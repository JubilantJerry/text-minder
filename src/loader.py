import os
import re
import string
from math import ceil
from collections import defaultdict, Counter
import codecs
from heapdict import heapdict
import numpy as np
import scipy.sparse
from threading import Thread, Lock
import json
import ipdb
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pattern.en
from allennlp.predictors.predictor import Predictor


SPEAKER_REGEXP = '(?:^|(?:[.!?]\\s))((\\w+ ?){1,3}):'
BRACKET_REGEXP = '[\\[(](.*?)[\\])]'
CONTEXT_RADIUS = 5
UPPER_CASE = True

NUM_CENTERS = 500
NUM_RESTARTS = 5
NUM_ITERS = 300
MAX_FRAC = 100

SEARCH_CUTOFF = 16
MAX_EXTRA_DIST = [6, 6, 10]
MAX_WORDNET_TAGS = [6, 6, 4]

POS_NOUN = 0
POS_VERB = 1
POS_DESCR = 2

VERB_TAGS = set(['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'MD', 'RP'])
NOUN_TAGS = set(['NN', 'NNP', 'NNPS', 'NNS', 'CD',
                 'PRP', 'PRP$', 'WP', 'WP$', 'WDT', 'EX', 'SYM'])
ADJ_TAGS = set(['JJ', 'JJS', 'JJR', 'PDT'])
VNA_TAGS = VERB_TAGS | NOUN_TAGS | ADJ_TAGS
IGNORE_TAGS = set(['RB', 'RBR', 'RBS', 'WRB', 'IN',
                   '.', ',', ':', '\'\'', '``', '-RRB-', '-LRB-', 'HYPH',
                   'FW', 'LS', 'NIL', 'XX', 'ADD', 'AFX',
                   'GW', 'CC', 'TO', 'NFP', 'POS', '_SP', 'DT', 'UH'])
BE_WORDS = set(['are', 'is', 'am', 'be', '\'re', '\'s', '\'m',
                'was', 'were', 'been', 'being'])
NEG_PREPS = set(['without', 'except', 'excluding', 'excepting', 'minus'])

OTHER_LEMMATIZE = {
    'me': 'i', 'my': 'i', 'mine': 'i',
    'your': 'you', 'yours': 'you',
    'him': 'he', 'his': 'he',
    'her': 'she', 'hers': 'she',
    'us': 'we', 'our': 'we', 'ours': 'we',
    'them': 'they', 'their': 'they', 'theirs': 'they',
    'its': 'it'
}
COMMON_NOUNS = set([
    'i', 'you', 'he', 'she', 'we', 'they', 'it',
    'this', 'that',
    'myself', 'yourself', 'himself', 'herself',
    'ourself', 'themself', 'itself',
])
COMMON_VERBS = set([
    'have', 'do', 'say', 'go', 'get', 'make',
    'know', 'think', 'take', 'see', 'come',
    'want', 'look', 'use', 'find', 'give',
    'tell', 'work', 'call', 'try', 'ask',
    'need', 'feel', 'become', 'leave',
    'put', 'mean', 'keep', 'let', 'begin',
    'seem', 'help', 'talk', 'turn', 'start',
    'show', 'hear', 'play', 'run', 'move',
    'like', 'live', 'believe', 'hold', 'bring',
    'happen', 'write', 'provide', 'sit', 'lose'])
DIRECT_TAG_MAP = {
    w: i for (i, w) in zip(range(len(COMMON_VERBS)),
                           sorted(list(COMMON_VERBS)))}

bad_words_set = set(stopwords.words('english'))
bad_words_set |= set([c for c in string.ascii_lowercase])

edge_costs = {
    'hyponym': 1,
    'to_lemma': 2,
    'hypernym': 10,
    'related_form': 6,
    'similar_to': 4,
    'same_pos_sense': 5,
    'diff_pos_sense': 8,
}

pos_inds = {'n': POS_NOUN, 'v': POS_VERB,
            'a': POS_DESCR, 's': POS_DESCR, 'r': POS_DESCR}
lemmatizer = WordNetLemmatizer()


def clean_line(line):
    line = re.sub(SPEAKER_REGEXP, '', line)
    line = re.sub(BRACKET_REGEXP, '', line)
    line = line.strip(' -\n')
    line = re.sub('([\\.,\\\'!\\?])', ' \\1', line)
    line = re.sub(r'[^a-zA-Z\\.,\\\'!\\?]', ' ', line)
    return line.strip()


def fully_clean_line(line):
    line = clean_line(line)
    tokens = line.split()
    cases = [
        ((not(w.isupper()) or len(w) == 1) and
         w[0].isupper()) for w in tokens]
    result = []
    for (i, token) in enumerate(tokens):
        token = token.lower()
        case = cases[i]
        if i != 0 and token[0] not in '.,\'!?':
            result.append(' ')
        if case == UPPER_CASE:
            result.append(token.title())
        else:
            result.append(token)
    return ''.join(result)


def strip_bad_words(tokens):
    return [w for w in tokens if w not in bad_words_set]


def build_vocab(files):
    vocab = defaultdict(int)

    for file in files:
        with open(file, 'r') as f:
            for line in f:
                line = clean_line(line)
                if not line:
                    continue

                tokens = line.split()
                for t in tokens:
                    vocab[t.lower()] += 1

    vocab_list = list(vocab.keys())
    vocab_counts = np.array([vocab[w] for w in vocab_list])
    vocab_map = dict(zip(vocab_list, range(len(vocab))))
    vocab_inv_map = dict(zip(range(len(vocab)), vocab_list))
    return vocab_map, vocab_inv_map, vocab_counts


def build_corpus(files, vocab_map):
    all_tokens = []
    all_token_cases = []
    curr_pos = 0
    split_pos = []

    for file in files:
        with open(file, 'r') as f:
            for line in f:
                line = clean_line(line)
                if not line:
                    continue

                split_pos.append(curr_pos)
                tokens = line.split()
                curr_pos += len(tokens)
                all_tokens.extend(vocab_map[w.lower()] for w in tokens)
                all_token_cases.extend((
                    (not(w.isupper()) or len(w) == 1) and
                    w[0].isupper()) for w in tokens)

    split_pos.append(curr_pos)
    return np.array(all_tokens), np.array(all_token_cases), np.array(split_pos)


def build_cooc(all_tokens, split_pos, vocab_map):
    vocab_size = len(vocab_map)
    num_lines = split_pos.shape[0] - 1
    bad_ids = set(vocab_map.get(w, -1) for w in bad_words_set)
    counts_by_row = [defaultdict(int) for _ in range(vocab_size)]

    for i in range(num_lines):
        start = split_pos[i]
        end = split_pos[i + 1]
        line = all_tokens[start:end]
        line_len = (end - start)

        for word_pos in range(line_len):
            window_start = max(0, word_pos - CONTEXT_RADIUS)
            window_end = min(line_len, word_pos + CONTEXT_RADIUS + 1)

            target_dict = counts_by_row[line[word_pos]]
            for w in line[window_start:window_end]:
                target_dict[w] += 1

    counts = np.zeros(vocab_size)
    for (i, d) in enumerate(counts_by_row):
        if i in bad_ids:
            continue
        for (j, v) in d.items():
            if i == j or j in bad_ids:
                continue
            counts[i] += v
    bad_ids.update(np.flatnonzero(counts == 0))
    good_ids = np.array(list(set(range(vocab_size)) - bad_ids))

    data = []
    row_ind = []
    col_ind = []
    for (i, d) in enumerate(counts_by_row):
        if i in bad_ids:
            continue
        count = counts[i]
        for (j, v) in d.items():
            if i == j or j in bad_ids:
                continue
            data.append(v / count)
            row_ind.append(i)
            col_ind.append(j)
        d.clear()
    data = np.array(data, dtype=np.float32)

    cooc_matrix = scipy.sparse.csr_matrix(
        (data, (row_ind, col_ind)),
        shape=(vocab_size, vocab_size))
    cooc_matrix = cooc_matrix[good_ids, :][:, good_ids]

    return cooc_matrix, good_ids


def build_glove(vocab_map, data_dir):
    selected_entries = []
    good_ids = []
    glove_vectors = np.load(os.path.join(data_dir, 'glove_all.npy'))
    glove_labels = np.load(os.path.join(data_dir, 'glove_labels.npy'))
    for (i, w) in enumerate(glove_labels):
        if w not in vocab_map:
            continue
        selected_entries.append(i)
        good_ids.append(vocab_map[w])
    vectors = glove_vectors[np.array(selected_entries)]
    return vectors, np.array(good_ids)


def get_possible_pos(vocab_inv_map, good_ids):
    mask = np.zeros((good_ids.shape[0], 3), dtype='bool')
    for (i, word_id) in enumerate(good_ids):
        word = vocab_inv_map[word_id]
        pos_options = set(s.pos() for s in wordnet.synsets(word))
        if 'n' in pos_options:
            mask[i, POS_NOUN] = True
        if 'v' in pos_options:
            mask[i, POS_VERB] = True
        if any(v in pos_options for v in 'asr'):
            mask[i, POS_DESCR] = True
    return mask


def k_means(points, weights,
            num_centers=NUM_CENTERS,
            max_frac=MAX_FRAC,
            num_restarts=NUM_RESTARTS,
            num_iters=NUM_ITERS):
    # learner = sklearn.cluster.KMeans(
    #     n_clusters=num_centers,
    #     n_init=num_restarts,
    #     verbose=True,
    #     precompute_distances=True)
    # assignments = learner.fit_predict(points, sample_weight=weights)
    # return assignments

    print("Started k-means")
    weights = weights / weights.sum()
    num_points = points.shape[0]
    best_loss = None
    best_assignments = None
    norms = np.array(np.square(points).sum(axis=1)).T

    for _ in range(num_restarts):
        centers = points[np.random.choice(num_points, num_centers)]
        if not(isinstance(centers, np.ndarray)):
            centers = centers.toarray()
        prev_loss = None
        curr_loss = None

        while True:
            dists = (
                norms + (centers ** 2).sum(axis=1, keepdims=True) -
                2 * (centers @ points.T))
            assignments = np.argmin(dists, axis=0)
            prev_loss = curr_loss
            curr_loss = (
                weights * dists[assignments, np.arange(num_points)]).sum()
            print(curr_loss, end='')
            too_few_count = 0
            too_many_count = 0

            for i in range(num_centers):
                group_points_ids = np.flatnonzero(assignments == i)
                group_weights = weights[group_points_ids]
                total_group_weight = group_weights.sum()
                too_few = total_group_weight == 0
                too_many = (
                    group_points_ids.shape[0] > 1 and
                    total_group_weight > max_frac / num_centers)

                if too_few or too_many:
                    result = points[np.random.choice(num_points)]
                    if not(isinstance(result, np.ndarray)):
                        result = result.toarray()
                    centers[i] = result
                    if too_few:
                        too_few_count += 1
                    if too_many:
                        too_many_count += 1
                else:
                    group_points = points[group_points_ids]
                    group_weights /= group_weights.sum()
                    centers[i] = group_weights @ group_points

            print(' %d %d' % (too_few_count, too_many_count))

            if not(prev_loss is None) and abs(prev_loss - curr_loss) < 1e-8:
                break

        if best_loss is None or curr_loss < best_loss:
            best_loss = curr_loss
            best_assignments = assignments
            print("Updated best")

    return best_assignments


def build_word_vector_matrix(vector_file, n_words=None):
    numpy_arrays = []
    labels_array = []
    with codecs.open(vector_file, 'r', 'utf-8') as f:
        for c, r in enumerate(f):
            sr = r.split()
            labels_array.append(sr[0])
            numpy_arrays.append(np.array([float(i) for i in sr[1:]]))

            if c == n_words:
                return np.array(numpy_arrays), labels_array
            if c % 1000 == 0:
                print(".", flush=True, end='')

    return np.array(numpy_arrays), labels_array


def synset_name(synset):
    return synset.name().split('.')[0]


def canonical_word(word):
    synsets = wordnet.synsets(word)
    if not synsets:
        return None

    counter = Counter()
    for s in synsets:
        s_name = synset_name(s)
        counter[s_name] += 3
        lemma_names = s.lemma_names()
        for n in lemma_names:
            counter[n] += 1 / len(lemma_names)

    missing_names = []
    for name in counter.keys():
        if name not in [synset_name(s) for s in wordnet.synsets(name)]:
            missing_names.append(name)
    for name in missing_names:
        del(counter[name])

    if len(counter) == 0:
        return None

    top_group = set(v[0] for v in counter.most_common(
        2 + ceil(0.1 * len(counter))))
    if word in top_group:
        return word
    else:
        return counter.most_common(1)[0][0]


def wordnet_search(manual_words, vocab_inv_map, search_cutoff=SEARCH_CUTOFF):
    targets = []
    target_words_map = {}

    for (word_id, word) in vocab_inv_map.items():
        c_word = canonical_word(word)
        if c_word is None:
            continue

        if c_word in target_words_map:
            to_write = target_words_map[c_word]
            targets[to_write].append(word_id)
        else:
            target_words_map[c_word] = len(targets)
            targets.append([word_id])

    num_targets = len(targets)
    dists = float('inf') * np.ones(
        (num_targets, 3, len(manual_words)))
    paths = {}

    visited = set()
    frontier = heapdict()
    token = 0
    for (i, synset) in enumerate(manual_words):
        key = (synset, i, token, ((synset.name(), 'origin'),))
        frontier[key] = 0.0
        token += 1

    ipdb.set_trace()

    while len(frontier) > 0:
        ((synset, origin, _, path), dist) = frontier.popitem()
        if dist > search_cutoff:
            break
        if (synset, origin) in visited:
            continue

        found_word = []
        found_word_lemma = []
        name = synset_name(synset)
        if name in target_words_map:
            found_word = [name]

        pos_ind = pos_inds[synset.pos()]
        if pos_ind == POS_NOUN:
            other_forms = [pattern.en.pluralize(name)]
        elif pos_ind == POS_VERB:
            other_forms = [
                pattern.en.conjugate(name, 'inf'),
                pattern.en.conjugate(name, 'p'),
                pattern.en.conjugate(name, 'part'),
                pattern.en.conjugate(name, 'ppart'),
            ]
        elif pos_ind == POS_DESCR:
            other_forms = [lmmatz(name, POS_DESCR)]
        other_forms = set(other_forms)
        if name in other_forms:
            other_forms.remove(name)
        other_forms = list(other_forms)

        for name in synset.lemma_names() + other_forms:
            if name in target_words_map:
                found_word_lemma.append(name)

        def register(name, dist, path):
            to_write = target_words_map[name]
            pos_ind = pos_inds[synset.pos()]
            orig_dist = dists[to_write, pos_ind, origin]
            if dist < orig_dist:
                dists[to_write, pos_ind, origin] = dist
                paths[(targets[to_write][0], pos_ind, origin)] = list(path)

        for name in found_word:
            register(name, dist, path)

        for name in found_word_lemma:
            new_path = path + ((name, 'to_lemma'),)
            new_dist = dist + edge_costs['to_lemma']
            register(name, new_dist, new_path)

        visited.add((synset, origin))

        edge_cost = edge_costs['hyponym']
        if dist + edge_cost <= search_cutoff:
            for next_synset in synset.hyponyms():
                if (next_synset, origin) in visited:
                    continue
                next_path = path + ((next_synset.name(), 'hyponym'),)
                key = (next_synset, origin, token, next_path)
                frontier[key] = dist + edge_cost
                token += 1

        edge_cost = edge_costs['related_form']
        if dist + edge_cost <= search_cutoff:
            next_synsets = set(
                rel_lemma.synset()
                for lemma in synset.lemmas() for rel_lemma in
                lemma.derivationally_related_forms()
                if lemma.name() == name)
            for next_synset in next_synsets:
                if (next_synset, origin) in visited:
                    continue
                next_path = path + ((next_synset.name(), 'related_form'),)
                key = (next_synset, origin, token, next_path)
                frontier[key] = dist + edge_cost
                token += 1

        edge_cost = edge_costs['hypernym']
        if dist + edge_cost <= search_cutoff:
            for next_synset in synset.hypernyms():
                if (next_synset, origin) in visited:
                    continue
                next_path = path + ((next_synset.name(), 'hypernym'),)
                key = (next_synset, origin, token, next_path)
                frontier[key] = dist + edge_cost
                token += 1

        edge_cost = edge_costs['similar_to']
        if dist + edge_cost <= search_cutoff:
            for next_synset in synset.similar_tos():
                if (next_synset, origin) in visited:
                    continue
                next_path = path + ((next_synset.name(), 'similar_to'),)
                key = (next_synset, origin, token, next_path)
                frontier[key] = dist + edge_cost
                token += 1

        same_pos_cost = edge_costs['same_pos_sense']
        diff_pos_cost = edge_costs['diff_pos_sense']
        required_name = synset_name(synset)
        required_pos = synset.pos()
        next_synsets = [
            s for s in wordnet.synsets(required_name)
            if synset_name(s) == required_name]
        for next_synset in next_synsets:
            if (next_synset, origin) in visited:
                continue
            if next_synset.pos() == required_pos:
                edge_cost = same_pos_cost
                next_path = path + ((next_synset.name(), 'same_pos_sense'),)
            else:
                edge_cost = diff_pos_cost
                next_path = path + ((next_synset.name(), 'diff_pos_sense'),)
            if dist + edge_cost <= search_cutoff:
                key = (next_synset, origin, token, next_path)
                frontier[key] = dist + edge_cost
                token += 1

    ipdb.set_trace()
    mask = (dists.min(axis=2) < float('inf'))

    targets_temp = targets
    targets = []
    mask_ids = []
    for i in range(mask.shape[0]):
        if mask[i].max():
            targets.append(targets_temp[i])
            mask_ids.append(i)
    dists = dists[mask.max(axis=1)]

    target_words_map = {}
    for (i, word_ids) in enumerate(targets):
        for word_id in word_ids:
            word = vocab_inv_map[word_id]
            target_words_map[word] = i

    fill_words = {}
    for (i, word_ids) in enumerate(targets):
        for word_id in word_ids:
            word = vocab_inv_map[word_id]
            mask_word = mask[mask_ids[i]]
            for pos_id in range(3):
                if mask_word[pos_id]:
                    base_word = lmmatz(word, pos_id)
                    if base_word not in target_words_map and \
                            base_word not in fill_words:
                        fill_words[base_word] = word

    return targets, target_words_map, fill_words, dists, paths


def construct_parser(path, cuda=False):
    if cuda:
        return Predictor.from_path(path, cuda_device=0)
    else:
        return Predictor.from_path(path)


def get_parse(parser, sentences, postproc=None):
    raw_outputs = parser.predict_batch_json(
        [{'sentence': s} for s in sentences])

    def visit(node):
        data = (node['word'], node['attributes'][0])
        children = {}
        if 'children' not in node:
            return (*data, children)
        for child in node['children']:
            node_type = child['nodeType']
            if node_type not in children:
                children[node_type] = []
            children[node_type].append(visit(child))
        return (*data, children)

    if postproc is None:
        def postproc(x):
            return x

    return [postproc(visit(raw_output['hierplane_tree']['root']))
            for raw_output in raw_outputs]


def process(parse):
    result = []
    noun_set = set()

    if parse[1] in IGNORE_TAGS:
        sections = search_down(parse)
        for section in sections:
            result += process_root(section, noun_set)
    else:
        result = process_root(parse, noun_set)

    result += squeeze_other(parse, noun_set)
    return result


def process_root(parse, noun_set):
    pos_tag = parse[1]
    if pos_tag in VERB_TAGS:
        result = process_verb(parse, noun_set)[1]
    elif pos_tag in NOUN_TAGS:
        result = process_np(parse, noun_set)[1]
    elif pos_tag in ADJ_TAGS:
        result = process_adj(parse, noun_set)[1]
    else:
        raise RuntimeError()
    return result


def scan_nested_prep(preps):
    for p in preps:
        yield p
        if 'prep' in p[2]:
            scan_nested_prep(p[2]['prep'])


def process_verb(parse, noun_set, subj='[NONE]', obj='[NONE]',
                 polarity='POS', isreal='REAL'):
    if parse[1] not in ['MD']:
        verb = parse[0]
    else:
        verb = '[UNK]'
    children = parse[2]
    bits = []

    passive = ('nsubjpass' in children)

    if 'nsubj' in children:
        subj, subj_bits = process_np(children['nsubj'][-1], noun_set)
        bits += subj_bits
    elif passive and 'prep' in children:
        for p in children['prep']:
            if 'pobj' in p[2] and p[0].lower() in [
                    'by', 'with', 'using', 'via']:
                subj, subj_bits = process_np(p[2]['pobj'][0], noun_set)
                bits += subj_bits
                break

    for link in ['iobj', 'dobj']:
        if link in children:
            obj, obj_bits = process_np(children[link][0], noun_set)
            bits += obj_bits
    if obj == '[NONE]' and passive:
        obj, obj_bits = process_np(children['nsubjpass'][0], noun_set)
        bits += obj_bits

    if 'neg' in children:
        polarity = 'NEG'

    if 'aux' in children:
        for aux_c in children['aux']:
            if aux_c[1] in ['MD'] and aux_c[0].lower() not in ['will', '\'ll']:
                isreal = 'HYPO'
                break

    if 'advmod' in children:
        for adv in children['advmod']:
            if adv[1] in ['RB'] and adv[0].lower() not in ['ever']:
                bits.append(('DESCR', subj, 'ACTION', adv[0], polarity))
            elif adv[1] in ['WRB'] and adv[0].lower() not in ['how']:
                isreal = 'HYPO'

    for link in ['conj', 'partmod']:
        if link in children:
            for conj in children[link]:
                if conj[1] in VERB_TAGS:
                    _, inner_bits = process_verb(conj, noun_set, subj=subj)
                    bits += inner_bits
                elif conj[1] in VNA_TAGS:
                    inner_bits = process_root(conj, noun_set)
                    bits += inner_bits

    for link in ['xcomp', 'infmod']:
        if link in children:
            if not passive:
                obj = '[CMPLX]'
                implicit = subj
            else:
                implicit = obj

            for comp in children[link]:
                if comp[1] in VERB_TAGS:
                    _, inner_bits = process_verb(
                        comp, noun_set, subj=implicit, isreal='TOPIC')
                    bits += inner_bits
                elif comp[1] in VNA_TAGS:
                    inner_bits = process_root(comp, noun_set)
                    bits += inner_bits

    if 'ccomp' in children:
        obj = '[CMPLX]'
        for comp in children['ccomp']:
            if comp[1] in VERB_TAGS:
                _, inner_bits = process_verb(comp, noun_set, isreal='TOPIC')
                bits += inner_bits
            elif comp[1] in VNA_TAGS:
                inner_bits = process_root(comp, noun_set)
                bits += inner_bits

    for link in ['dep', 'parataxis', 'acomp']:
        if link in children:
            for dep in children[link]:
                if dep[1] in VNA_TAGS:
                    inner_bits = process_root(dep, noun_set)
                    bits += inner_bits

    if 'advcl' in children:
        for clause in children['advcl']:
            if clause[1] in VERB_TAGS:
                inner_isreal = 'REAL'
                inner_ch = clause[2]
                if 'mark' in inner_ch and \
                        inner_ch['mark'][0][0].lower() in ['if']:
                    inner_isreal = 'HYPO'
                _, inner_bits = process_verb(
                    clause, noun_set, isreal=inner_isreal)
                bits += inner_bits

    if verb.lower() in BE_WORDS:
        verb = '[BE]'
    if verb == '[BE]' and obj == '[NONE]':
        verb = '[EXIST]'

    if 'prep' in children:
        for p in scan_nested_prep(children['prep']):
            inner_polarity = 'POS'
            if p[0].lower() in NEG_PREPS:
                inner_polarity = 'NEG'

            if 'pobj' in p[2]:
                tiewd, tiewd_bits = process_np(p[2]['pobj'][0], noun_set)
                bits += tiewd_bits
                connection = '[ACT_LINK]'
                if p[0].lower() == 'for':
                    connection = '[ACT_FOR]'
                bits.append(('ACTION', subj, connection,
                             tiewd, inner_polarity, 'REAL'))
                if p[1] in VERB_TAGS:
                    bits += [('ACTION', '[CMPLX]',
                              p[0], tiewd, 'POS', 'REAL')]
            elif 'pcomp' in p[2]:
                _, inner_bits = process_pcomp(p, noun_set, inner_polarity)
                bits += inner_bits

    if 'poss' in children and verb in ['[BE]', '[EXIST]']:
        for poss in children['poss']:
            posswd = poss[0]
            bits.append(('ACTION', posswd, '[POSSESS]', subj, 'POS', 'REAL'))

    main = ('ACTION', subj, verb, obj, polarity, isreal)
    bits.append(main)
    return verb, bits


def process_np(parse, noun_set):
    if parse[1] in NOUN_TAGS or parse[1] in ['DT']:
        subj = parse[0]
    else:
        subj = '[CMPLX]'
    children = parse[2]
    bits = []

    if 'nn' in children:
        chain = []
        for mod in children['nn']:
            chain.append(mod[0])
            noun_set.add(mod[0])
        chain.append(subj)
        noun_set.add(subj)
        subj = ' '.join(chain)

    if 'amod' in children:
        for adj in children['amod']:
            if subj != '[NONE]':
                _, inner_bits = process_adj(adj, noun_set, subj=subj)
                bits += inner_bits

    negate = False
    if 'det' in children:
        for det in children['det']:
            if subj != '[NONE]' and det[0].lower() in ['no']:
                negate = True
                break
    if 'neg' in children:
        negate = True
    if negate:
        bits.append(('DESCR', subj, 'DIRECT', '[IS_NOT]', 'POS'))

    if 'prep' in children:
        for p in scan_nested_prep(children['prep']):
            inner_polarity = 'POS'
            if p[0].lower() in NEG_PREPS:
                inner_polarity = 'NEG'
            if 'pobj' in p[2] and subj != '[NONE]':
                tiewd, tiewd_bits = process_np(p[2]['pobj'][0], noun_set)
                bits += tiewd_bits
                bits.append(('ACTION', subj, '[LINKED_TO]',
                             tiewd, inner_polarity, 'REAL'))
            elif p[1] in VERB_TAGS:
                _, inner_bits = process_verb(
                    p, noun_set, polarity=inner_polarity,
                    isreal='TOPIC')
                bits += inner_bits
            elif 'pcomp' in p[2]:
                _, inner_bits = process_pcomp(p, noun_set, inner_polarity)
                bits += inner_bits

    for link in ['num', 'number', 'npadvmod']:
        if link in children:
            for p in children[link]:
                if p[1] in NOUN_TAGS:
                    tiewd = p[0]
                    bits.append(('ACTION', subj, '[LINKED_TO]',
                                 tiewd, 'POS', 'REAL'))

    if 'poss' in children:
        for poss in children['poss']:
            posswd = poss[0]
            bits.append(('ACTION', posswd, '[POSSESS]', subj, 'POS', 'REAL'))

    if 'nsubj' in children:
        for ns in children['nsubj']:
            tiewd, tiewd_bits = process_np(ns, noun_set)
            bits += tiewd_bits
            if 'cop' in children and children['cop'][0][0].lower() in BE_WORDS:
                if 'possessive' in children or subj.lower() in [
                        'mine', 'his', 'hers', 'ours', 'theirs', 'its']:
                    bits.append(('ACTION', subj, '[POSSESS]',
                                 tiewd, 'POS', 'REAL'))
                else:
                    bits.append(('ACTION', tiewd, '[BE]',
                                 subj, 'POS', 'REAL'))
            else:
                bits.append(('ACTION', subj, '[LINKED_TO]',
                             tiewd, 'POS', 'REAL'))

    for link in ['dep', 'ccomp', 'xcomp', 'conj', 'partmod',
                 'infmod', 'parataxis', 'acomp']:
        if link in children:
            for dep in children[link]:
                if dep[1] in VERB_TAGS:
                    _, inner_bits = process_verb(dep, noun_set, isreal='TOPIC')
                    bits += inner_bits
                elif dep[1] in VNA_TAGS:
                    inner_bits = process_root(dep, noun_set)
                    bits += inner_bits

    if 'rcmod' in children:
        for rcmod in children['rcmod']:
            if rcmod[1] in VERB_TAGS and subj != '[NONE]':
                verb, inner_bits = process_verb(rcmod, noun_set)
                bits += inner_bits
                bits.append(('DESCR', subj, 'THROUGH_ACT', verb, 'POS'))
            elif rcmod[1] in VNA_TAGS:
                inner_bits = process_root(rcmod, noun_set)
                bits += inner_bits

    noun_set.add(subj)
    if subj not in ['[CMPLX]']:
        bits.append(('RELEVANT', subj))
    return subj, bits


def process_adj(parse, noun_set, subj='[NONE]'):
    descr = parse[0]
    bits = []
    children = parse[2]

    polarity = 'POS'
    if 'neg' in children:
        polarity = 'NEG'

    if 'nsubj' in children:
        subj, subj_bits = process_np(children['nsubj'][-1], noun_set)
        bits += subj_bits
        noun_set.add(subj)

    if 'prep' in children:
        for p in scan_nested_prep(children['prep']):
            inner_polarity = 'POS'
            if p[0].lower() in NEG_PREPS:
                inner_polarity = 'NEG'
            if 'pobj' in p[2] and subj != '[NONE]':
                tiewd, tiewd_bits = process_np(p[2]['pobj'][0], noun_set)
                bits += tiewd_bits
                bits.append(('ACTION', subj, '[LINKED_TO]',
                             tiewd, inner_polarity, 'REAL'))
            elif p[1] in VERB_TAGS:
                _, inner_bits = process_verb(
                    p, noun_set, polarity=inner_polarity,
                    isreal='TOPIC')
                bits += inner_bits
            elif 'pcomp' in p[2]:
                _, inner_bits = process_pcomp(p, noun_set, inner_polarity)
                bits += inner_bits

    for link in ['dep', 'ccomp', 'xcomp', 'conj', 'partmod',
                 'infmod', 'parataxis', 'acomp']:
        if link in children:
            for comp in children[link]:
                if comp[1] in VERB_TAGS:
                    _, inner_bits = process_verb(
                        comp, noun_set, isreal='TOPIC')
                    bits += inner_bits
                elif comp[1] in VNA_TAGS:
                    inner_bits = process_root(comp, noun_set)
                    bits += inner_bits

    main = None
    if subj != '[NONE]':
        main = ('DESCR', subj, 'DIRECT', descr, polarity)
        bits.append(main)
    return descr, bits


def process_pcomp(p, noun_set, inner_polarity):
    bits = []

    inner = p[2]['pcomp'][0]
    if inner[1] in VERB_TAGS:
        _, inner_bits = process_verb(
            inner, noun_set, polarity=inner_polarity,
            isreal='TOPIC')
        bits += inner_bits
    elif inner[1] in VNA_TAGS:
        inner_bits = process_root(inner, noun_set)
        bits += inner_bits
    elif 'pcomp' in inner[2]:
        _, inner_bits = process_pcomp(inner, noun_set, inner_polarity)
        bits += inner_bits
    if p[1] in VERB_TAGS:
        bits += [('ACTION', '[CMPLX]',
                  p[0], '[CMPLX]', 'POS', 'REAL')]

    return p[0], bits


def search_down(parse):
    if parse[1] in VNA_TAGS:
        return [parse]

    results = []
    for c in parse[2].values():
        for inst in c:
            test = search_down(inst)
            results += test

    return results


def squeeze_other(parse, noun_set):
    bits = []
    children = parse[2]

    if parse[1] in NOUN_TAGS and parse[0] not in noun_set:
        bits += process_np(parse, noun_set)[1]
    if parse[0] == '?':
        bits.append(('QUESTION',))
    if parse[1] == 'UH':
        bits.append(('UTTERANCE',))
    for c in children.values():
        for inst in c:
            bits += squeeze_other(inst, noun_set)

    return bits


def lmmatz(word, pos):
    if word.startswith('['):
        return word

    word = word.lower()

    if pos == POS_NOUN:
        if word in OTHER_LEMMATIZE:
            return OTHER_LEMMATIZE[word]
        return lemmatizer.lemmatize(word, pos='n')
    elif pos == POS_VERB:
        return lemmatizer.lemmatize(word, pos='v')
    else:
        shortest = word
        for pos in ['a', 's', 'r']:
            cand = lemmatizer.lemmatize(word, pos)
            if len(cand) < len(shortest):
                shortest = cand
        return shortest


def clean_up_bits(bits):
    by_noun = defaultdict(set)
    actions = set()
    misc = set()
    for bit in bits:
        if bit[0] == 'RELEVANT':
            new_bit = (bit[0], lmmatz(bit[1], POS_NOUN))
            by_noun[new_bit[1]].add(new_bit)
        elif bit[0] == 'DESCR':
            pos_snd = POS_DESCR
            if bit[2] == 'THROUGH_ACT':
                pos_snd = POS_VERB
            new_bit = (bit[0], lmmatz(bit[1], POS_NOUN),
                       bit[2], lmmatz(bit[3], pos_snd), *bit[4:])
            by_noun[new_bit[1]].add(new_bit)
        elif bit[0] == 'ACTION':
            new_bit = (bit[0], lmmatz(bit[1], POS_NOUN),
                       lmmatz(bit[2], POS_VERB),
                       lmmatz(bit[3], POS_NOUN), *bit[4:])
            actions.add(new_bit)
        elif bit[0] in ['QUESTION', 'UTTERANCE']:
            misc.add(bit)
    return [list(s) for s in by_noun.values()], list(actions), list(misc)


def replace_with_tags(corpus, by_noun, actions, misc, train=True):
    new_by_noun = []
    new_actions = []
    new_misc = []

    def merge(tags_pair):
        if train:
            return tags_pair[0] + 3 * tags_pair[1]
        else:
            if len(tags_pair[1]) > 0:
                return tags_pair[1]
            return tags_pair[0]

    for section in by_noun:
        new_section = []
        for bit in section:
            if bit[0] == 'RELEVANT':
                assert not(bit[1].startswith('['))
                if bit[1] not in COMMON_NOUNS:
                    tags = merge(corpus.find_tags(bit[1], POS_NOUN))
                else:
                    tags = ()
                new_section.append((*bit, tags))
            elif bit[0] == 'DESCR':
                pos_snd = POS_DESCR
                if bit[2] == 'THROUGH_ACT':
                    pos_snd = POS_VERB
                if bit[3].startswith('['):
                    tags = bit[3]
                else:
                    tags = merge(corpus.find_tags(bit[3], pos_snd))
                    if len(tags) == 0 and bit[2] == 'DIRECT':
                        tags = merge(corpus.find_tags(bit[3], POS_NOUN))
                if len(tags) > 0:
                    new_section.append((*bit[:-2], tags, bit[-1]))
        new_by_noun.append(new_section)

    for bit in actions:
        if bit[2].startswith('['):
            tags = bit[2]
        elif bit[2] not in COMMON_VERBS:
            tags = merge(corpus.find_tags(bit[2], POS_VERB))
        else:
            tags = "{%s}" % bit[2]
        if len(tags) > 0:
            new_actions.append((*bit[:2], tags, *bit[3:]))

    new_misc = misc.copy()
    return new_by_noun, new_actions, new_misc


def sc_to_tokens(by_noun, actions, misc):
    by_noun = [sorted(v, key=str) for v in sorted(by_noun, key=str)]
    actions = sorted(actions, key=str)
    misc = sorted(misc, key=str)

    def tag_to_token(tag):
        if isinstance(tag, tuple):
            return ['('] + list(tag) + [')']
        else:
            return [tag]

    result = []
    for section in by_noun:
        for bit in section:
            if bit[0] == 'RELEVANT':
                result += ['|[', bit[0], bit[1], ']|']
            elif bit[1] == 'DESCR':
                result += [
                    '|[', bit[0], bit[1], bit[2],
                    *tag_to_token(bit[3]), ']|']
    for bit in actions:
        result += [
            '|[', bit[0], bit[1], *tag_to_token(bit[2]),
            bit[3], bit[4], bit[5], ']|']
    for bit in misc:
        result += ['|[', bit[0], ']|']
    return result


bleu_smoother = SmoothingFunction().method2


def bleu(toks_ref, toks_hypo):
    def adjustment(length):
        if length == 1:
            return 1.6817928305074292
        elif length == 2:
            return 1.414213562373095
        elif length == 3:
            return 1.189207115002721
        else:
            return 1.0
    if len(toks_ref) == 0:
        return 1.0 if len(toks_hypo) == 0 else 0.0

    raw_value = sentence_bleu(
        [toks_ref], toks_hypo, smoothing_function=bleu_smoother)
    return adjustment(len(toks_ref)) * raw_value


def sc_similarity(sc_ref, sc_hypo):
    toks_ref = sc_to_tokens(*sc_ref)
    toks_hypo = sc_to_tokens(*sc_hypo)
    return bleu(toks_ref, toks_hypo)


KEYWORD_OFFSETS = {
    'RELEVANT': (0, {}),
    'DESCR': (1, {'DIRECT': 2, 'ACTION': 3,
                  'THROUGH_ACT': 4, 'POS': 5, 'NEG': 6}),
    'ACTION': (7, {'LHS': 8, 'RHS': 9, 'POS': 10, 'NEG': 11,
                   'REAL': 12, 'HYPO': 13, 'TOPIC': 14}),
    'QUESTION': (15, {}),
    'UTTERANCE': (16, {}),
}
NUM_KEYWORDS = 17

PLACEHOLDER_OFFSETS = {
    'NONE': 0,
    'CMPLX': 1,
}

RELATION_OFFSETS = {
    'LINKED_TO': 0,
    'POSSESS': 1,
    'BE': 2,
    'EXIST': 3,
    'ACT_LINK': 4,
    'ACT_FOR': 5,
    'IS_NOT': 6,
    'UNK': 7
}


def to_bitvecs(corpus, by_noun, actions, misc):
    keyword_tmp = np.zeros(NUM_KEYWORDS)
    placeholder_tmp = np.zeros(len(PLACEHOLDER_OFFSETS))
    relation_tmp = np.zeros(len(RELATION_OFFSETS))
    dtag_tmp = np.zeros(len(DIRECT_TAG_MAP))
    tag_tmp = np.zeros(len(corpus.to_tag_id))

    keyword_tmp.flags['WRITEABLE'] = False
    placeholder_tmp.flags['WRITEABLE'] = False
    relation_tmp.flags['WRITEABLE'] = False
    dtag_tmp.flags['WRITEABLE'] = False
    tag_tmp.flags['WRITEABLE'] = False

    new_by_noun = []
    new_actions = []
    new_misc = []

    for section in by_noun:
        new_section = []
        for bit in section:
            keyword_cpy = keyword_tmp.copy()
            placeholder_cpy = placeholder_tmp.copy()
            tag_cpy = tag_tmp.copy()
            if bit[0] == 'RELEVANT':
                noun = bit[1]
                if noun.startswith('['):
                    noun = None
                    placeholder_cpy[PLACEHOLDER_OFFSETS[bit[1][1:-1]]] = 1
                cabinet = KEYWORD_OFFSETS['RELEVANT']
                keyword_cpy[cabinet[0]] = 1
                assert isinstance(bit[2], tuple)
                for tag in bit[2]:
                    tag_cpy[corpus.to_tag_id[tag]] = 1
                new_section.append((noun, np.hstack([
                    keyword_cpy, placeholder_cpy, relation_tmp,
                    dtag_tmp, tag_cpy])))
            else:
                assert bit[0] == 'DESCR'
                noun = bit[1]
                if noun.startswith('['):
                    noun = None
                    placeholder_cpy[PLACEHOLDER_OFFSETS[bit[1][1:-1]]] = 1
                relation_cpy = relation_tmp.copy()
                cabinet = KEYWORD_OFFSETS['DESCR']
                keyword_cpy[cabinet[0]] = 1
                keyword_cpy[cabinet[1][bit[2]]] = 1
                if isinstance(bit[3], str):
                    relation_cpy[RELATION_OFFSETS[bit[3][1:-1]]] = 1
                else:
                    for tag in bit[3]:
                        tag_cpy[corpus.to_tag_id[tag]] = 1
                new_section.append((noun, np.hstack([
                    keyword_cpy, placeholder_cpy, relation_cpy,
                    dtag_tmp, tag_cpy])))
        new_by_noun.append(new_section)

    for bit in actions:
        keyword_cpy = keyword_tmp.copy()
        relation_cpy = relation_tmp.copy()
        dtag_cpy = dtag_tmp.copy()
        tag_cpy = tag_tmp.copy()
        cabinet = KEYWORD_OFFSETS['ACTION']

        keyword_cpy[cabinet[0]] = 1
        keyword_cpy[cabinet[1][bit[4]]] = 1
        keyword_cpy[cabinet[1][bit[5]]] = 1
        if isinstance(bit[2], str):
            if bit[2].startswith('['):
                relation_cpy[RELATION_OFFSETS[bit[2][1:-1]]] = 1
            else:
                assert bit[2].startswith('{')
                dtag_cpy[DIRECT_TAG_MAP[bit[2][1:-1]]] = 1
        else:
            for tag in bit[2]:
                tag_cpy[corpus.to_tag_id[tag]] = 1

        keyword_cpy[cabinet[1]['LHS']] = 1
        placeholder_cpy = placeholder_tmp.copy()
        noun = bit[1]
        if noun.startswith('['):
            noun = None
            placeholder_cpy[PLACEHOLDER_OFFSETS[bit[1][1:-1]]] = 1
        lhs_part = (noun, np.hstack([
            keyword_cpy, placeholder_cpy, relation_cpy,
            dtag_cpy, tag_cpy]))

        keyword_cpy[cabinet[1]['LHS']] = 0
        keyword_cpy[cabinet[1]['RHS']] = 1
        placeholder_cpy = placeholder_tmp.copy()
        noun = bit[3]
        if noun.startswith('['):
            noun = None
            placeholder_cpy[PLACEHOLDER_OFFSETS[bit[3][1:-1]]] = 1
        rhs_part = (noun, np.hstack([
            keyword_cpy, placeholder_cpy, relation_cpy,
            dtag_cpy, tag_cpy]))
        new_actions.append((lhs_part, rhs_part))

    for bit in misc:
        keyword_cpy = keyword_tmp.copy()
        if bit[0] == 'QUESTION':
            keyword_cpy[KEYWORD_OFFSETS['QUESTION'][0]] = 1
        else:
            assert bit[0] == 'UTTERANCE'
            keyword_cpy[KEYWORD_OFFSETS['UTTERANCE'][0]] = 1
        new_misc.append((None, np.hstack([
            keyword_cpy, placeholder_tmp, relation_tmp,
            dtag_tmp, tag_tmp])))

    return new_by_noun, new_actions, new_misc


class Corpus:
    def __init__(self, data_dir, cache_dir, cache=True, less_mem=False):
        lines_dir = os.path.join(data_dir, 'lines')
        files = sorted(
            [os.path.join(lines_dir, p) for p in os.listdir(lines_dir)])

        vocab_path = os.path.join(cache_dir, 'vocab.npy')
        if cache and os.path.exists(vocab_path):
            print("Loading vocab")
            vocab_items = np.load(vocab_path, allow_pickle=True)
        else:
            print("Rebuilding vocab")
            vocab_items = build_vocab(files)
            np.save(vocab_path, vocab_items)
        self.vocab_map = vocab_items[0]
        self.vocab_inv_map = vocab_items[1]
        self.vocab_counts = vocab_items[2]

        corpus_path = os.path.join(cache_dir, 'corpus.npy')
        if cache and os.path.exists(corpus_path):
            print("Loading corpus")
            corpus_items = np.load(corpus_path, allow_pickle=True)
        else:
            print("Rebuilding corpus")
            corpus_items = build_corpus(files, self.vocab_map)
            np.save(corpus_path, corpus_items)
        self.all_tokens = corpus_items[0]
        self.all_token_cases = corpus_items[1]
        self.split_pos = corpus_items[2]

        # cooc_path = os.path.join(cache_dir, 'cooc.npz')
        # good_ids_path = os.path.join(cache_dir, 'good_ids.npy')
        # if cache and os.path.exists(cooc_path):
        #     self.cooc_matrix = scipy.sparse.load_npz(cooc_path)
        #     self.good_ids = np.load(good_ids_path)
        # else:
        #     self.cooc_matrix, self.good_ids = build_cooc(
        #         self.all_tokens, self.split_pos, self.vocab_map)
        #     scipy.sparse.save_npz(cooc_path, self.cooc_matrix)
        #     np.save(good_ids_path, self.good_ids)

        glove_path = os.path.join(cache_dir, 'glove.npy')
        if cache and os.path.exists(glove_path):
            print("Loading glove")
            glove_items = np.load(glove_path, allow_pickle=True)
        else:
            print("Rebuilding glove")
            glove_items = build_glove(self.vocab_map, data_dir)
            glove_items = (*glove_items, None)
            np.save(glove_path, glove_items)
        if not less_mem:
            self.glove_vecs = glove_items[0]
        self.good_ids = glove_items[1]

        assignments_path = os.path.join(cache_dir, 'assignments.npy')
        if cache and os.path.exists(assignments_path):
            print("Loading assignments")
            assignments_items = np.load(assignments_path, allow_pickle=True)
        else:
            print("Rebuilding assignments")
            if less_mem:
                raise RuntimeError("Rerun with full initialization!")
            possible_pos = get_possible_pos(self.vocab_inv_map, self.good_ids)
            good_counts = self.vocab_counts[self.good_ids]
            assignments = [k_means(
                self.glove_vecs[possible_pos[:, pos]],
                good_counts[possible_pos[:, pos]]) for pos in range(3)]
            assignments_items = (possible_pos, assignments, None)
            np.save(assignments_path, assignments_items)

        possible_pos = assignments_items[0]
        assignments = assignments_items[1]
        self.assignments_arr = []
        self.assignments_map = []
        for pos in range(3):
            mask = possible_pos[:, pos]
            good_ids_m = self.good_ids[mask]
            assignments_pos = assignments[pos]

            weights = self.vocab_counts[good_ids_m]
            weights = weights / weights.sum()
            group_weight_sum = np.zeros(NUM_CENTERS)
            for i in range(NUM_CENTERS):
                group_points_ids = np.flatnonzero(assignments_pos == i)
                group_weights = weights[group_points_ids]
                group_weight_sum[i] = group_weights.sum()
            max_weight_order = np.argsort(-group_weight_sum)
            assignment_perm = np.argsort(max_weight_order)
            new_assignments = assignment_perm[assignments_pos]

            self.assignments_arr.append(np.vstack([
                good_ids_m, new_assignments]))
            self.assignments_map.append(dict(zip(
                good_ids_m, new_assignments)))

        print("Loading manual words")
        manual_words_path = os.path.join(data_dir, 'manual_words.txt')
        self.manual_words = []
        with open(manual_words_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.manual_words.append(wordnet.synset(line))

        wordnet_path = os.path.join(cache_dir, 'wordnet.npy')
        if cache and os.path.exists(wordnet_path):
            print("Loading wordnet")
            wordnet_items = np.load(wordnet_path, allow_pickle=True)
        else:
            print("Rebuilding wordnet")
            wordnet_items = wordnet_search(
                self.manual_words, self.vocab_inv_map)
            np.save(wordnet_path, wordnet_items)
        self.wn_targets = wordnet_items[0]
        self.target_words_map = wordnet_items[1]
        self.fill_words = wordnet_items[2]
        self.wn_dists = wordnet_items[3]
        if not less_mem:
            self.wn_paths = wordnet_items[4]

        def load_tags(file_name):
            tags_path = os.path.join(data_dir, file_name)
            result = []
            with open(tags_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    result.append(line[1:].split())
            return result

        print("Loading tags")
        self.tags_m = load_tags('tags_m.txt')
        tags_n = load_tags('tags_n.txt')
        tags_v = load_tags('tags_v.txt')
        tags_d = load_tags('tags_d.txt')
        self.tags_pos = [tags_n, tags_v, tags_d]
        unique_tags = sorted(list(set([
            w for i in [self.tags_m] + self.tags_pos
            for j in i for w in j])))
        self.to_tag_id = {
            w: i for (i, w) in
            zip(range(len(unique_tags)), unique_tags)}

        def process_ceng_tags(dest, words, tags, pos):
            for (i, word) in enumerate(words):
                if len(tags[i]) == 0:
                    continue
                word = lmmatz(word, pos)
                if word in dest:
                    dest[word] = list(set(tags[i]) | set(dest[word]))
                else:
                    dest[word] = tags[i]

        print("Loading common English tags")
        ceng_words_path = os.path.join(data_dir, 'common_english.txt')
        ceng_words = []
        with open(ceng_words_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ceng_words.append(line)
        ceng_tags_n = load_tags('tags_common_n.txt')
        ceng_tags_v = load_tags('tags_common_v.txt')
        ceng_tags_d = load_tags('tags_common_d.txt')
        self.ceng_tags = [{}, {}, {}]
        process_ceng_tags(
            self.ceng_tags[POS_NOUN], ceng_words, ceng_tags_n, POS_NOUN)
        process_ceng_tags(
            self.ceng_tags[POS_VERB], ceng_words, ceng_tags_v, POS_VERB)
        process_ceng_tags(
            self.ceng_tags[POS_DESCR], ceng_words, ceng_tags_d, POS_DESCR)

        parse_path = os.path.join(cache_dir, 'parse.npy')
        if cache and os.path.exists(parse_path):
            print("Loading parse")
            parse_items = np.load(parse_path, allow_pickle=True)
        else:
            print("Rebuilding parse")
            ipdb.set_trace()
            path = os.path.join(data_dir, 'parser.tar.gz')
            parser = construct_parser(path, cuda=True)
            num_sentences = (self.split_pos.shape[0] - 1)

            num_workers = 2
            parse_shards = {}
            progress = 0
            lock = Lock()

            def worker(wid):
                nonlocal progress
                chunk_size = 200
                for i in range(
                        chunk_size * wid, num_sentences,
                        chunk_size * num_workers):
                    shard = get_parse(parser, [
                        self.sentence(j) for j in range(
                            i, min(num_sentences, i + chunk_size))],
                        json.dumps)

                    lock.acquire()
                    progress += chunk_size
                    if progress >= 10000:
                        print('.', end='', flush=True)
                        progress -= 10000
                    parse_shards[i] = shard
                    lock.release()

            threads = []
            for i in range(num_workers):
                threads.append(Thread(target=worker, args=(i,)))
                threads[i].start()
            for i in range(num_workers):
                threads[i].join()

            compact_parses = []
            sorted_inds = sorted(list(parse_shards.keys()))
            for i in sorted_inds:
                compact_parses += parse_shards[i]
                del(parse_shards[i])
            parse_items = (compact_parses, None)
            np.save(parse_path, parse_items)
        self.compact_parses = parse_items[0]

    def cluster_top_words(self, k=10):
        for pos in range(3):
            if pos == POS_NOUN:
                print("For nouns:")
            elif pos == POS_VERB:
                print("For verbs:")
            elif pos == POS_DESCR:
                print("For descriptors:")

            good_ids_m, assignments = self.assignments_arr[pos]
            weights = self.vocab_counts[good_ids_m]
            weights = weights / weights.sum()

            for i in range(NUM_CENTERS):
                group_points_ids = np.flatnonzero(assignments == i)
                group_weights = weights[group_points_ids]
                top_inds = group_points_ids[np.argsort(-group_weights)[:k]]
                print([self.vocab_inv_map[good_ids_m[i]] for i in top_inds],
                      group_weights.sum() * NUM_CENTERS,
                      group_weights.shape[0])
            print('')

    def num_sentences(self):
        return self.split_pos.shape[0] - 1

    def sentence(self, index):
        start = self.split_pos[index]
        end = self.split_pos[index + 1]
        tokens = self.all_tokens[start:end]
        cases = self.all_token_cases[start:end]

        result = []
        for i in range(tokens.shape[0]):
            token = self.vocab_inv_map[tokens[i]]
            case = cases[i]
            if i != 0 and token[0] not in '.,\'!?':
                result.append(' ')
            if case == UPPER_CASE:
                result.append(token.title())
            else:
                result.append(token)
        return ''.join(result)

    def parse(self, index):
        return json.loads(self.compact_parses[index])

    def sem_constrs(self, index, train=True):
        parse = self.parse(index)
        bits = process(parse)
        cleaned = clean_up_bits(bits)
        result = replace_with_tags(self, *cleaned, train)
        return result

    def bitvec_size(self):
        return (
            NUM_KEYWORDS +
            len(PLACEHOLDER_OFFSETS) +
            len(RELATION_OFFSETS) +
            len(DIRECT_TAG_MAP) +
            len(self.to_tag_id)
        )

    def print_senses(self, word):
        lemmas = [set([
            n for s in wordnet.synsets(word)
            for n in s.lemma_names() if s.pos() in p])
            for p in ['n', 'v', 'asr']]
        c_word = canonical_word(word)
        alt_words = [lg & self.target_words_map.keys() for lg in lemmas]
        print("Canonical word: %s\n" % c_word)
        print("Alternative nouns:\n%s\n" % str(alt_words[POS_NOUN]))
        print("Alternative verbs:\n%s\n" % str(alt_words[POS_VERB]))
        print("Alternative descriptors:\n%s\n" % str(alt_words[POS_DESCR]))

        target_id = self.target_words_map[word]
        dists = self.wn_dists[target_id]
        print("Target vocab ID: %d" % self.wn_targets[target_id][0])

        def print_for_ind(i):
            count = (dists[i] < float('inf')).sum()
            order = np.argsort(dists[i])[:count]
            words = [self.manual_words[j].name() for j in order]
            sel_dists = dists[i][order]
            print(list(zip(words, sel_dists, order)))

        print("As a noun:")
        print_for_ind(POS_NOUN)
        print("\nAs a verb:")
        print_for_ind(POS_VERB)
        print("\nAs a descriptor:")
        print_for_ind(POS_DESCR)

    def find_tags(self, word, pos):
        tags = set()

        if word in self.fill_words:
            word = self.fill_words[word]

        if word in self.target_words_map:
            target_id = self.target_words_map[word]
            dists = self.wn_dists[target_id]
            order = np.argsort(dists[pos])
            starting_dist = dists[pos][order[0]]
            max_extra_dist = MAX_EXTRA_DIST[pos]
            max_wordnet_tags = MAX_WORDNET_TAGS[pos]

            for j in order:
                dist = dists[pos][j]
                if dist == float('inf') or \
                        dist > starting_dist + max_extra_dist:
                    break
                new_tags = set(self.tags_m[j])
                new_tags.difference_update(tags)
                if len(tags) + len(new_tags) > max_wordnet_tags:
                    break
                tags.update(new_tags)

        if word in self.vocab_map:
            word_id = self.vocab_map[word]
            assignments_map = self.assignments_map[pos]
            if word_id in assignments_map:
                assignment = assignments_map[word_id]
                tags.update(self.tags_pos[pos][assignment])

        added_tags = tuple()
        lword = lmmatz(word, pos)
        if lword in self.ceng_tags[pos]:
            added_tags = tuple(self.ceng_tags[pos][lword])

        return tuple(tags), added_tags

    def print_tags(self, word):
        print("As a noun: ", end='')
        print(self.find_tags(word, POS_NOUN))
        print("As a verb: ", end='')
        print(self.find_tags(word, POS_VERB))
        print("As a descriptor: ", end='')
        print(self.find_tags(word, POS_DESCR))
