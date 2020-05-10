import os
import sys
import random
import torch
import numpy as np
from scipy.stats import norm
from sklearn.decomposition import TruncatedSVD
from transformers import GPT2Tokenizer
from external.modeling_gpt2 import GPT2LMHeadModel
import loader
import sc_gpt2_model
import ipdb

CONTEXT_MIN_SIZE = 3
DATA_DIR = '../data'
CACHE_DIR = '../data/cache'

LENGTH_SPLITS = [(80, 2), (160, 4), (320, 8)]
BATCH_SIZE = 8
PCA_BATCHES = 8000

FILL_TOKEN = 9101
SC_DROPOUT_PROB = 0.25
TAG_LAM = 3.0


def load_model(device):
    tokenizer = GPT2Tokenizer.from_pretrained(
        'gpt2-medium', cache_dir='../data/gpt2')
    model = GPT2LMHeadModel.from_pretrained(
        'gpt2-medium', cache_dir='../data/gpt2')
    model = model.to(device)
    return tokenizer, model


class LinesDataset(torch.utils.data.Dataset):
    def __init__(self, corpus, context_min_size, split_token,
                 split=False, sc_changer=None, train=True):
        self.corpus = corpus
        self.context_min_size = context_min_size
        self.split_token = split_token
        self.split = split
        self.sc_changer = sc_changer
        self.train = train

    def __len__(self):
        count = self.corpus.num_sentences()
        return count // self.context_min_size - 1

    def make_sem_constrs(self, lid):
        raw = self.corpus.sem_constrs(lid, self.train)
        if self.sc_changer is None:
            return raw
        else:
            return self.sc_changer(raw)

    def __getitem__(self, index):
        start = self.context_min_size * index
        val_range = tuple(range(start, start + 2 * self.context_min_size))
        mid = self.split_token.join(
            self.corpus.sentence(i) for i in val_range)
        line = self.split_token + mid + self.split_token
        if self.split:
            return line, tuple(
                self.make_sem_constrs(lid) for lid in val_range)
        else:
            return line


class LinesListDataset(torch.utils.data.Dataset):
    def __init__(self, corpus, parser, data, context_min_size,
                 split_token, split=False, sc_changer=None, train=False):
        self.corpus = corpus
        self.parser = parser
        self.data = data
        self.context_min_size = context_min_size
        self.split_token = split_token
        self.split = split
        self.sc_changer = sc_changer
        self.train = train

    def __len__(self):
        count = len(self.data)
        return count // self.context_min_size - 1

    def make_sem_constrs(self, lid):
        sentence = self.data[lid]
        parse = loader.get_parse(self.parser, [sentence])[0]
        constr = loader.clean_up_bits(loader.process(parse))
        raw = loader.replace_with_tags(self.corpus, *constr, self.train)
        if self.sc_changer is None:
            return raw
        else:
            return self.sc_changer(raw)

    def __getitem__(self, index):
        start = self.context_min_size * index
        val_range = tuple(range(start, start + 2 * self.context_min_size))
        mid = self.split_token.join(self.data[i] for i in val_range)
        line = self.split_token + mid + self.split_token
        if self.split:
            return line, tuple(
                self.make_sem_constrs(lid) for lid in val_range)
        else:
            return line


def prepare_batch(tokenizer, ldata, device, single=False):
    ltids = [tokenizer.encode(inst) for inst in ldata]
    longest = max(len(t) for t in ltids)

    if single:
        piece_size = len(ltids)
        total_size = len(ltids)
    else:
        split_categ = 0
        split_count = 1
        for (size, count) in LENGTH_SPLITS:
            if longest > size and size > split_categ:
                split_categ = size
                split_count = count
        piece_size = BATCH_SIZE // split_count
        total_size = BATCH_SIZE

    ltids_list = []
    lmasks_list = []
    for start in range(0, total_size, piece_size):
        end = start + piece_size
        ltids_p = ltids[start:end]
        plen = max(len(t) for t in ltids_p)

        lmasks_p = [[1] * len(t) + [0] * (plen - len(t)) for t in ltids_p]
        ltids_p = [t + [0] * (plen - len(t)) for t in ltids_p]

        ltids_p = torch.tensor(
            ltids_p, dtype=torch.long).to(device)
        lmasks_p = torch.tensor(
            lmasks_p, dtype=torch.float).to(device)
        ltids_list.append(ltids_p)
        lmasks_list.append(lmasks_p)

    return ltids_list, lmasks_list


def learn_pca(device='cpu', stuff=None):
    if stuff is None:
        tokenizer, model = load_model(device=device)
        corpus = loader.Corpus(DATA_DIR, CACHE_DIR, less_mem=True)
        np.stuff = (tokenizer, model, corpus)
    else:
        tokenizer, model, corpus = stuff

    lines_dataset = LinesDataset(
        corpus, CONTEXT_MIN_SIZE, tokenizer.eos_token)
    lines_loader = torch.utils.data.DataLoader(
        lines_dataset, BATCH_SIZE, shuffle=True)
    model.transformer.output_hidden_states = True

    mean = None
    var = None
    count = 0

    for i in range(PCA_BATCHES):
        ldata = next(iter(lines_loader))
        ltids_list, lmasks_list = prepare_batch(tokenizer, ldata, device)

        for id_bit, m_bit in zip(ltids_list, lmasks_list):
            with torch.no_grad():
                outputs = model(id_bit, attention_mask=m_bit)
                embed_dim = outputs[2][0].shape[-1]
                pieces = [
                    (outputs[2][i] * m_bit[:, :, None]).view(-1, embed_dim)
                    for i in range(1, len(outputs[2]))
                ]
                vec = torch.cat(pieces, dim=1)
                vec = torch.sign(vec) * torch.log(1 + torch.abs(vec))
                del(outputs)
                del(pieces)

                if mean is None:
                    mean = vec.new_zeros(vec.shape[1])
                    var = vec.new_zeros(vec.shape[1])
                mean += vec.sum(dim=0)
                torch.addmm(var, vec.T, vec, out=var)
                piece_count = m_bit.sum()
                count += piece_count
        print('%d ' % (i * BATCH_SIZE), end='', flush=True)

    mean /= count
    var /= count
    torch.addmm(var, -mean[:, None], mean[None, :], out=var)

    var_np = var.cpu().data.numpy()
    learner = TruncatedSVD(n_components=1024, n_iter=8)
    learner.fit(var_np)
    np.save('../data/gpt2/mean.npy', mean.cpu().data.numpy())
    np.save('../data/gpt2/e_vecs.npy', learner.components_)
    np.save('../data/gpt2/e_vals.npy', learner.singular_values_)

    explained_var = learner.singular_values_.sum()
    total_var = torch.diag(var).sum().item()
    print(1 - (explained_var / total_var))


def compress(z_vals):
    packed = z_vals
    packed = norm.cdf(packed / 2)
    packed = 2 * packed - 1
    packed = np.round(packed * 127).astype(np.int8)
    return packed


def decompress(packed):
    z_vals = packed.astype(np.float).clip(min=-126.75, max=126.75) / 127
    z_vals = (z_vals + 1) / 2
    z_vals = norm.ppf(z_vals) * 2
    return z_vals.astype(np.float32)


def memoize_gpt2_data(device='cpu', stuff=None):
    if stuff is None:
        tokenizer, model = load_model(device=device)
        corpus = loader.Corpus(DATA_DIR, CACHE_DIR, less_mem=True)
        np.stuff = (tokenizer, model, corpus)
    else:
        tokenizer, model, corpus = stuff

    lines_dataset = LinesDataset(
        corpus, CONTEXT_MIN_SIZE, tokenizer.eos_token)
    total_size = len(lines_dataset)
    model.transformer.output_hidden_states = True
    model.eval()

    mean = torch.tensor(np.load('../data/gpt2/mean.npy')).to(device)
    e_vecs = torch.tensor(np.load('../data/gpt2/e_vecs.npy')).to(device)
    sqrt_e_vals = torch.sqrt(
        torch.tensor(np.load('../data/gpt2/e_vals.npy')).to(device))

    lines_perm = np.random.permutation(total_size)
    start = 0
    arena_num = 0
    pos_arr = []
    lines_perm_dir = '../data/gpt2/memo/lines_perm.npy'
    start_dir = '../data/gpt2/memo/start.npy'
    arena_num_dir = '../data/gpt2/memo/arena_num.npy'
    pos_arr_dir = '../data/gpt2/memo/pos.npy'
    if os.path.exists(lines_perm_dir):
        lines_perm = np.load(lines_perm_dir)
        start = np.load(start_dir)
        arena_num = np.load(arena_num_dir)
        pos_arr = np.load(pos_arr_dir).tolist()
    else:
        # np.save(lines_perm_dir, lines_perm)
        # np.save(start_dir, start)
        # np.save(arena_num_dir, arena_num)
        # np.save(pos_arr_dir, pos_arr)
        pass

    arena = []
    pointer = 0
    num_added = 0
    print("Starting from instance %d for arena %d" % (start, arena_num))

    for batch_start in range(start, total_size, BATCH_SIZE):
        elems = lines_perm[batch_start:batch_start + BATCH_SIZE]
        ldata = [lines_dataset[j] for j in elems]
        ltids_list, lmasks_list = prepare_batch(tokenizer, ldata, device)
        new_pointer = pointer

        for id_bit, m_bit in zip(ltids_list, lmasks_list):
            with torch.no_grad():
                outputs = model(id_bit, attention_mask=m_bit)
                embed_dim = outputs[2][0].shape[-1]
                pieces = [
                    (outputs[2][i] * m_bit[:, :, None]).view(-1, embed_dim)
                    for i in range(1, len(outputs[2]))
                ]
                vec = torch.cat(pieces, dim=1)
                vec = torch.sign(vec) * torch.log(1 + torch.abs(vec))
                del(outputs)
                del(pieces)
                vec = vec[m_bit.view(-1) > 0.5]

                transf = (vec - mean) @ e_vecs.T / sqrt_e_vals
                packed = transf.cpu().data.numpy()
                packed = compress(packed)

                arena.append(packed)
                new_pointer += packed.shape[0]

        pos_arr.append((arena_num, pointer))
        pointer = new_pointer
        arena.append(np.ones((1, e_vecs.shape[0]), dtype=np.int8) * (-128))
        pointer += 1
        num_added += 1
        print('%d' % ((batch_start % 10000) // 1000), end='', flush=True)

        if num_added >= 3000 or batch_start + BATCH_SIZE >= total_size:
            joined = np.concatenate(arena, axis=0)
            # np.save('../data/gpt2/memo/%d.npy' % arena_num, joined)
            # np.save(pos_arr_dir, np.array(pos_arr))
            # np.save(arena_num_dir, arena_num + 1)
            # np.save(start_dir, batch_start + BATCH_SIZE)
            print("\nSaved up to instance %d for arena %d" % (
                batch_start + BATCH_SIZE, arena_num))
            arena = []
            arena_num += 1
            pointer = 0
            num_added = 0


def shuffle_drop(entries, param, method='binomial'):
    count = len(entries)
    if method == 'binomial':
        prob = param
        to_keep = np.random.binomial(count, 1 - prob)
    else:
        assert method == 'poisson'
        lam = param
        to_keep = np.random.poisson(lam)
    new_entries = list(entries).copy()
    random.shuffle(new_entries)
    result = []
    for i in range(len(new_entries)):
        if new_entries[i] not in result:
            result.append(new_entries[i])
        if len(result) >= to_keep:
            break
    if isinstance(entries, tuple):
        result = tuple(result)
    return result


def peturb_constrs(by_noun, actions, misc):
    by_noun_temp = by_noun.copy()
    random.shuffle(by_noun_temp)
    by_noun = []
    for group in by_noun_temp:
        group = [list(x) for x in shuffle_drop(group, SC_DROPOUT_PROB)]
        if len(group) == 0:
            continue
        for bit in group:
            if bit[0] == 'RELEVANT':
                bit[2] = shuffle_drop(bit[2], TAG_LAM, method='poisson')
            elif bit[0] == 'DESCR' and isinstance(bit[3], tuple):
                bit[3] = shuffle_drop(bit[3], TAG_LAM, method='poisson')
        by_noun.append(group)

    actions = [list(x) for x in shuffle_drop(actions, SC_DROPOUT_PROB)]
    for bit in actions:
        if isinstance(bit[2], tuple):
            bit[2] = shuffle_drop(bit[2], TAG_LAM, method='poisson')

    misc = shuffle_drop(misc, SC_DROPOUT_PROB)
    return by_noun, actions, misc


def nouns_only(constrs):
    by_noun, actions, misc = constrs
    new_by_noun = []
    for section in by_noun:
        new_section = []
        for bit in section:
            if bit[0] == 'RELEVANT':
                new_section.append((bit[0], bit[1], ()))
        new_by_noun.append(new_section)
    return new_by_noun, [], []


def no_constr(constrs):
    return [], [], []


def prepare_sc_batch(corpus, tokenizer, ldata, transf_batch,
                     device, peturb=True, single=False,
                     remove_last_eos=False):
    eos_id = tokenizer.eos_token_id
    bitvec_size = corpus.bitvec_size()
    if transf_batch is not None:
        transf_size = transf_batch.shape[1]
    else:
        transf_size = 0

    tpos = 0
    pieces = []
    for (line, s_constrs_list) in ldata:
        encoder_input_ids_s = []
        sc_tagalong_s = []

        for s_constrs in s_constrs_list:
            if peturb:
                s_constrs = peturb_constrs(*s_constrs)
            by_noun, actions, misc = loader.to_bitvecs(corpus, *s_constrs)
            line_eids = []
            line_tgs = []

            def encode(noun, bitvec, spacer):
                if noun is None:
                    noun = '_'
                encoded = tokenizer.encode(noun)
                spacer = tokenizer.encode(spacer)
                spacer_zero = np.zeros_like(bitvec)[np.newaxis, :]
                bitvec = np.tile(bitvec, [len(encoded), 1])
                assert len(spacer) == 1
                line_eids.append(encoded)
                line_tgs.append(bitvec)
                line_eids.append(spacer)
                line_tgs.append(spacer_zero)

            for group in by_noun:
                for (noun, bitvec) in group:
                    encode(noun, bitvec, '|')
            for ((noun_lhs, bitvec_lhs), (noun_rhs, bitvec_rhs)) in actions:
                encode(noun_lhs, bitvec_lhs, '-')
                encode(noun_rhs, bitvec_rhs, '|')
            for (noun, bitvec) in misc:
                encode(noun, bitvec, '|')

            if len(line_eids) > 0:
                line_eids.pop()
                line_tgs.pop()
            line_eids.append([eos_id])
            line_tgs.append(np.zeros((1, bitvec_size)))
            line_eids = np.array([x for l in line_eids for x in l])
            line_tgs = np.concatenate(line_tgs, axis=0)
            assert line_eids.shape[0] == line_tgs.shape[0]
            encoder_input_ids_s.append(line_eids)
            sc_tagalong_s.append(line_tgs)

        decoder_input_ids_s = np.array(tokenizer.encode(line))[:-1]
        total_count = decoder_input_ids_s.shape[0]
        old_tpos = tpos
        tpos = tpos + total_count + 1
        if transf_batch is not None:
            transf_s = transf_batch[old_tpos:tpos - 1]

        split_places = np.where(decoder_input_ids_s == eos_id)[0][1:]
        decoder_input_ids_s = np.split(decoder_input_ids_s, split_places)
        split_sizes = tuple(np.diff((0, *split_places, total_count)))
        if transf_batch is not None:
            transf_s = torch.split(transf_s, split_sizes, dim=0)
        else:
            transf_s = [torch.zeros((0, 0), device=device)] * len(split_sizes)

        assert len(encoder_input_ids_s) == len(decoder_input_ids_s)
        encoder_attention_mask = []
        encoder_input_ids = []
        sc_tagalong = []
        decoder_attention_mask = []
        decoder_input_ids = []
        transf = []
        labels = []

        for i in range(len(encoder_input_ids_s)):
            line_eids = encoder_input_ids_s[i]
            encoder_attention_mask.append(np.ones(line_eids.shape[0]))
            encoder_input_ids.append(line_eids)
            sc_tagalong.append(sc_tagalong_s[i])
            decoder_attention_mask.append(np.zeros(line_eids.shape[0]))
            decoder_input_ids.append(np.ones_like(line_eids) * FILL_TOKEN)
            transf.append(transf_s[i].new_zeros(
                (line_eids.shape[0], transf_size)))
            labels.append(np.ones_like(line_eids) * FILL_TOKEN)

            line_dids = decoder_input_ids_s[i]
            decoder_attention_mask.append(np.ones(line_dids.shape[0]))
            decoder_input_ids.append(line_dids)
            labels.append(np.concatenate([line_dids[1:], [eos_id]], axis=0))
            transf.append(transf_s[i])
            encoder_attention_mask.append(np.zeros(line_dids.shape[0]))
            encoder_input_ids.append(np.ones_like(line_dids) * FILL_TOKEN)
            sc_tagalong.append(np.zeros((line_dids.shape[0], bitvec_size)))

        encoder_attention_mask = np.concatenate(
            encoder_attention_mask, axis=0).astype(np.float32)
        encoder_input_ids = np.concatenate(
            encoder_input_ids, axis=0).astype(np.int64)
        sc_tagalong = np.concatenate(sc_tagalong, axis=0).astype(np.float32)
        decoder_attention_mask = np.concatenate(
            decoder_attention_mask, axis=0).astype(np.float32)
        decoder_input_ids = np.concatenate(
            decoder_input_ids, axis=0).astype(np.int64)
        transf = torch.cat(transf, dim=0)
        labels = np.concatenate(labels, axis=0).astype(np.int64)
        if remove_last_eos:
            encoder_attention_mask = encoder_attention_mask[:-1]
            encoder_input_ids = encoder_input_ids[:-1]
            sc_tagalong = sc_tagalong[:-1]
            decoder_attention_mask = decoder_attention_mask[:-1]
            decoder_input_ids = decoder_input_ids[:-1]
            transf = transf[:-1]
            labels = labels[:-1]
        pieces.append((
            encoder_attention_mask,
            encoder_input_ids,
            sc_tagalong,
            decoder_attention_mask,
            decoder_input_ids,
            transf,
            labels,
        ))

    longest = max(t[0].shape[0] for t in pieces)

    if single:
        piece_size = len(pieces)
        total_size = len(pieces)
    else:
        split_categ = 0
        split_count = 1
        for (size, count) in LENGTH_SPLITS:
            if longest > size and size > split_categ:
                split_categ = size
                split_count = count
        piece_size = BATCH_SIZE // split_count
        total_size = BATCH_SIZE

    split_pieces = []
    for start in range(0, total_size, piece_size):
        plen = max(t[0].shape[0] for t in pieces[start:start + piece_size])
        encoder_attention_mask = torch.tensor(np.stack([
            np.pad(p[0], (0, plen - p[0].shape[0])) for p in pieces[
                start:start + piece_size]], axis=0), device=device)
        encoder_input_ids = torch.tensor(np.stack([
            np.pad(p[1], (0, plen - p[1].shape[0])) for p in pieces[
                start:start + piece_size]], axis=0), device=device)
        sc_tagalong = torch.tensor(np.stack([
            np.pad(p[2], ((0, plen - p[2].shape[0]), (0, 0))) for p in pieces[
                start:start + piece_size]], axis=0), device=device)
        decoder_attention_mask = torch.tensor(np.stack([
            np.pad(p[3], (0, plen - p[2].shape[0])) for p in pieces[
                start:start + piece_size]], axis=0), device=device)
        decoder_input_ids = torch.tensor(np.stack([
            np.pad(p[4], (0, plen - p[4].shape[0])) for p in pieces[
                start:start + piece_size]], axis=0), device=device)
        if transf_batch is not None:
            transf = torch.stack([
                torch.nn.functional.pad(
                    p[5], (0, 0, 0, plen - p[5].shape[0])) for p in pieces[
                    start:start + piece_size]], dim=0)
        else:
            transf = None
        labels = torch.tensor(np.stack([
            np.pad(p[6], (0, plen - p[6].shape[0])) for p in pieces[
                start:start + piece_size]], axis=0), device=device)
        to_add = {
            'encoder_attention_mask': encoder_attention_mask,
            'encoder_input_ids': encoder_input_ids,
            'sc_tagalong': sc_tagalong,
            'decoder_attention_mask': decoder_attention_mask,
            'decoder_input_ids': decoder_input_ids,
            'transf': transf,
            'labels': labels,
        }
        if transf_batch is None:
            del(to_add['transf'])
        split_pieces.append(to_add)
    return split_pieces


def train_model(device='cpu', stuff=None):
    if stuff is None:
        corpus = loader.Corpus(DATA_DIR, CACHE_DIR, less_mem=True)
        np.stuff = corpus
    else:
        corpus = stuff

    sqrt_e_vals = torch.sqrt(
        torch.tensor(np.load('../data/gpt2/e_vals.npy')).to(device))
    lines_perm = np.load('../data/gpt2/memo/lines_perm.npy')
    pos_arr = np.load('../data/gpt2/memo/pos.npy')

    tokenizer, gpt2_model = load_model(device=device)
    lines_dataset = LinesDataset(
        corpus, CONTEXT_MIN_SIZE, tokenizer.eos_token, split=True)

    encoder_config = sc_gpt2_model.GPT2Config(
        n_layer=2, n_head=16, n_embd=1024)
    decoder_config = sc_gpt2_model.GPT2Config(
        n_layer=4, n_head=16, n_embd=1024)
    config = sc_gpt2_model.SCGPT2Config(
        corpus.bitvec_size(), sqrt_e_vals.shape[0],
        encoder_config, decoder_config)
    model = sc_gpt2_model.SCGPT2HeadModel(config).to(device)
    model.embedding.sabotage_gpt2(gpt2_model)
    # model.load_state_dict(torch.load(
    #     '../data/gpt2/checkpt/nouns_model_0_50000.pth'))  # !!!
    model.train()
    del(gpt2_model)

    novel_named_parameters = model.novel_named_parameters()
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in novel_named_parameters
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": 1e-2,
        },
        {"params": [p for n, p in novel_named_parameters
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    num_epochs = 4
    starting_epoch = 0
    starting_lr = 2e-4  # !!!
    optim = torch.optim.AdamW(optimizer_grouped_parameters, lr=starting_lr)

    # log = open('../data/gpt2/checkpt/log.txt', 'w')

    def log_print(*args, **kwargs):
        print(*args, **kwargs)
        # print(*args, **kwargs, file=log)

    # ipdb.set_trace()

    # with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for epoch in range(starting_epoch, num_epochs + 1):
        log_print('Now on epoch %d' % epoch)
        if epoch == num_epochs:
            break

        prev_arena = -1
        arena_data = None
        mean_loss = None
        kill_run = False
        for batch_num in range(len(pos_arr)):
            if os.path.exists('../data/gpt2/checkpt/decay_%d_%d.txt' %
                              (epoch, batch_num)):
                for group in optim.param_groups:
                    lr = group['lr'] / 2
                    break
                for group in optim.param_groups:
                    group['lr'] = lr
                log_print("\nDecayed learning rate to %0.3e" % lr)

            if os.path.exists('../data/gpt2/checkpt/save_%d_%d.txt' %
                              (epoch, batch_num)):
                torch.save(
                    model.state_dict(),
                    '../data/gpt2/checkpt/model_%d_%d.pth' %
                    (epoch, batch_num))
                torch.save(
                    optim.state_dict(),
                    '../data/gpt2/checkpt/optim_%d_%d.pth' %
                    (epoch, batch_num))
                pass

            if os.path.exists('../data/gpt2/checkpt/kill_%d_%d.txt' %
                              (epoch, batch_num)):
                kill_run = True
                break

            arena, pointer = pos_arr[batch_num]
            if arena != prev_arena:
                arena_data = np.load(
                    '../data/gpt2/memo/%d.npy' % arena, mmap_mode='r')
                prev_arena = arena
            if batch_num == (len(pos_arr) - 1):
                transf_batch = arena_data[pointer:-1]
            else:
                next_arena, next_pointer = pos_arr[batch_num + 1]
                if next_arena != arena:
                    transf_batch = arena_data[pointer:-1]
                else:
                    transf_batch = arena_data[pointer:next_pointer - 1]
            transf_batch = torch.tensor(decompress(transf_batch)).to(device)
            transf_batch = transf_batch * sqrt_e_vals

            # if batch_num == 50:
            #     kill_run = True
            #     break

            entries = lines_perm[
                batch_num * BATCH_SIZE:(batch_num + 1) * BATCH_SIZE]
            ldata = [lines_dataset[i] for i in entries]
            if batch_num == 0:
                print(ldata[0])

            split_pieces = prepare_sc_batch(
                corpus, tokenizer, ldata, transf_batch, device)
            counts = [p['decoder_attention_mask'].sum()
                      for p in split_pieces]
            total_count = sum(counts)

            if mean_loss is not None:
                log_print('%0.3f ' % mean_loss.item(), end='', flush=True)

            optim.zero_grad()
            mean_loss = torch.zeros(1, device=device)
            for i in range(len(split_pieces)):
                outputs = model.forward_full(**split_pieces[i])
                loss = outputs[0]
                scaled_loss = loss * counts[i] / total_count
                scaled_loss.backward()
                mean_loss += scaled_loss.data
            optim.step()
        log_print('')

        if kill_run:
            break
    # print(prof.key_averages().table(sort_by="cuda_time_total"))
    # log.close()


def evaluate(checkpt, count, sc_changer=None, stuff=None,
             device='cpu', seed=1337):
    if stuff is None:
        corpus = loader.Corpus(DATA_DIR, CACHE_DIR, less_mem=True)
        tokenizer, gpt2_model = load_model(device=device)
        np.stuff = (corpus, tokenizer, gpt2_model)
    else:
        corpus, tokenizer, gpt2_model = stuff

    lines_dir = '../data/eval_lines'
    files = sorted(
        [os.path.join(lines_dir, p) for p in os.listdir(lines_dir)])
    lines = []
    for file in files:
        with open(file, 'r') as f:
            for line in f:
                line = loader.fully_clean_line(line)
                if line != '':
                    lines.append(line)

    parser = loader.construct_parser('../data/parser.tar.gz')
    lines_dataset = LinesListDataset(
        corpus, parser, lines,
        CONTEXT_MIN_SIZE, tokenizer.eos_token,
        split=True, sc_changer=sc_changer)

    mean = torch.tensor(np.load('../data/gpt2/mean.npy')).to(device)
    e_vecs = torch.tensor(np.load('../data/gpt2/e_vecs.npy')).to(device)

    encoder_config = sc_gpt2_model.GPT2Config(
        n_layer=2, n_head=16, n_embd=1024)
    decoder_config = sc_gpt2_model.GPT2Config(
        n_layer=4, n_head=16, n_embd=1024)
    config = sc_gpt2_model.SCGPT2Config(
        corpus.bitvec_size(), e_vecs.shape[0],
        encoder_config, decoder_config)
    model = sc_gpt2_model.SCGPT2HeadModel(config)
    model.load_state_dict(torch.load(checkpt, map_location='cpu'))
    model = model.to(device)
    model.attach_gpt2_model(gpt2_model, mean, e_vecs)
    model.eval()

    np.random.seed(seed)
    sample_ids = np.random.choice(len(lines_dataset), size=count)
    eos_len = len(tokenizer.eos_token)

    avg_sentence_bleu = 0.0
    avg_sc_bleu = 0.0
    avg_loss = 0.0
    num_losses = 0

    eval_logfile = open('../data/gpt2/eval_logs.txt', 'w')

    def log_print(*args, **kwargs):
        print(*args, **kwargs)
        print(*args, **kwargs, file=eval_logfile, flush=True)

    torch.random.manual_seed(seed)

    for i in sample_ids:
        raw_ldata = lines_dataset[i]
        last_line_ind = raw_ldata[0][:-eos_len].rindex(tokenizer.eos_token)
        past = raw_ldata[0][:last_line_ind] + tokenizer.eos_token
        context = raw_ldata[0][:last_line_ind] + (tokenizer.eos_token * 2)
        target = raw_ldata[0][last_line_ind:]
        parse = loader.get_parse(parser, [target[eos_len:-eos_len]])[0]
        ref_constr = loader.clean_up_bits(loader.process(parse))
        ref_constr_tagged = loader.replace_with_tags(
            corpus, *ref_constr, train=False)
        ldata = [(context, raw_ldata[1])]
        all_sentences = []

        if i == sample_ids[0]:
            print(ldata[0][1][-1])
            print(ref_constr_tagged)

        with torch.no_grad():
            for j in range(5):
                split_pieces = prepare_sc_batch(
                    corpus, tokenizer, ldata, None, device,
                    peturb=False, single=True, remove_last_eos=True)
                del(split_pieces[0]['labels'])
                model.prepare_forward(**split_pieces[0], num_copies=5)
                output = model.generate(
                    max_length=100, num_return_sequences=5,
                    do_sample=True, pad_token_id=tokenizer.eos_token_id,
                    temperature=0.5)
                for sentence in output:
                    eos_places = (sentence == tokenizer.eos_token_id).nonzero()
                    if eos_places.shape[0] > 1:
                        end = eos_places[1, 0]
                    else:
                        end = sentence.shape[0]
                    sentence = tokenizer.decode(sentence[1:end])
                    all_sentences.append(sentence)

        best_similarity = None
        best_sentence = None
        for sentence in all_sentences:
            parse = loader.get_parse(parser, [sentence])[0]
            hypo_constr = loader.clean_up_bits(loader.process(parse))
            hypo_constr_tagged = loader.replace_with_tags(
                corpus, *hypo_constr, train=False)
            similarity = loader.sc_similarity(
                ref_constr_tagged, hypo_constr_tagged)
            if best_similarity is None or similarity > best_similarity:
                best_similarity = similarity
                best_sentence = sentence

        sentence_bleu = loader.bleu(target[eos_len:-eos_len], best_sentence)
        avg_sentence_bleu += sentence_bleu / count
        sc_bleu = best_similarity
        avg_sc_bleu += sc_bleu / count

        ldata = [(past, raw_ldata[1][:-1])]
        split_pieces = prepare_sc_batch(
            corpus, tokenizer, ldata, None, device,
            peturb=False, single=True)
        model_outputs = model.forward_with_gpt2(**split_pieces[0])
        gpt2_past = model_outputs[2]
        past_eam = split_pieces[0]['encoder_attention_mask']
        past_dam = split_pieces[0]['decoder_attention_mask']

        ldata = [(target, raw_ldata[1][-1:])]
        split_pieces = prepare_sc_batch(
            corpus, tokenizer, ldata, None, device,
            peturb=False, single=True)
        split_pieces[0]['encoder_attention_mask'] = torch.cat([
            past_eam, split_pieces[0]['encoder_attention_mask']], dim=1)
        split_pieces[0]['decoder_attention_mask'] = torch.cat([
            past_dam, split_pieces[0]['decoder_attention_mask']], dim=1)
        loss = model.forward_with_gpt2(**split_pieces[0], past=gpt2_past)[0]
        loss = loss.item()
        loss_count = split_pieces[0]['decoder_attention_mask'].sum().item()
        avg_loss += loss * loss_count
        num_losses += loss_count

        orig_sentences = raw_ldata[0].replace(tokenizer.eos_token, '|')
        log_print("\"%s\" ==> \"%s\", (%0.4f, %0.4f, %0.4f)" % (
            orig_sentences, best_sentence, sentence_bleu, sc_bleu, loss))

    avg_loss /= num_losses
    log_print("Averages: (%0.4f, %0.4f, %0.4f)" % (
        avg_sentence_bleu, avg_sc_bleu, avg_loss))
    eval_logfile.close()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if 'memo' in sys.argv[1]:
            memoize_gpt2_data(device='cuda')
        elif 'train' in sys.argv[1]:
            train_model(device='cuda')
