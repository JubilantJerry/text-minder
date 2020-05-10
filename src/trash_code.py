

class AddMultConst(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mult = torch.nn.Parameter(torch.ones(dim))
        self.bias = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, vals):
        return vals * self.mult + self.bias


class ByteAutoencoder(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

        encode_layers = []
        decode_layers = []
        rdims = dims[::-1]
        for i in range(len(dims) - 2):
            encode_layers += [
                torch.nn.Linear(dims[i], dims[i + 1]),
                AddMultConst(dims[i + 1]),
                torch.nn.LeakyReLU(negative_slope=0.1)
            ]
            decode_layers += [
                torch.nn.Linear(rdims[i], rdims[i + 1]),
                AddMultConst(rdims[i + 1]),
                torch.nn.LeakyReLU(negative_slope=0.1)
            ]
        encode_layers += [
            torch.nn.Linear(dims[-2], dims[-1]),
            AddMultConst(dims[-1]),
            torch.nn.Tanh()
        ]
        decode_layers += [
            torch.nn.Linear(rdims[-2], rdims[-1]),
            AddMultConst(rdims[-1])
        ]
        self.encode_nn = torch.nn.Sequential(*encode_layers)
        self.decode_nn = torch.nn.Sequential(*decode_layers)

        for p in self.encode_nn.parameters():
            if p.dim() >= 2:
                torch.nn.init.xavier_normal_(p.data)

        for p in self.decode_nn.parameters():
            if p.dim() >= 2:
                torch.nn.init.xavier_normal_(p.data)

    def noisy_pass(self, vals):
        encoded = self.encode_nn(vals)
        noise = (2 * torch.rand_like(encoded)) - 1
        noise = noise / 127
        noisy = encoded + noise * 0
        decoded = self.decode_nn(noisy)
        loss = ((decoded - vals) ** 2).mean(axis=1)
        return loss

    def encode(self, vals):
        with torch.no_grad():
            encoded = self.encode_nn(vals)
            return torch.round(encoded * 127).to(torch.int8)

    def decode(self, levels):
        with torch.no_grad():
            encoded = levels.to(torch.float) / 127
            return self.decode_nn(encoded)


def playground_1(device='cpu', stuff=None):
    # if stuff is None:
    #     tokenizer = None  # load_model(device=device)
    #     corpus = None  # loader.Corpus(DATA_DIR, CACHE_DIR, less_mem=True)
    #     np.stuff = (tokenizer, model, corpus)
    # else:
    #     tokenizer, model, corpus = stuff

    inner_config = sc_gpt2_model.GPT2Config(n_layer=2, n_ctx=2048)
    config = sc_gpt2_model.SCGPT2Config(5, 17, inner_config, inner_config)
    model = sc_gpt2_model.SCGPT2HeadModel(config).to(device)
    model.encoder.output_hidden_states = True
    model.decoder.output_hidden_states = True

    encoder_input_ids = torch.tensor([[
        999, 999, 999, 999, 999, 999, 10919, 257, 30256,
        999, 999, 999, 999, 999, 999, 999, 999, 999]]).to(device)
    sc_tagalong = torch.ones(1, encoder_input_ids.shape[1], 5).to(device)
    decoder_input_ids = torch.tensor([[
        3666, 1204, 329, 38230, 333, 0, 999, 999, 999,
        198, 1639, 1276, 5678, 3224, 279, 2645, 684, 13]]).to(device)
    transf = torch.ones(1, encoder_input_ids.shape[1], 17).to(device)
    encoder_attention_mask = torch.tensor([[
        0, 0, 0, 0, 0, 0, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0]]).to(device).to(torch.float)
    decoder_attention_mask = torch.tensor([[
        1, 1, 1, 1, 1, 1, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1]]).to(device).to(torch.float)
    labels = torch.tensor([[
        1204, 329, 38230, 333, 0, 198, 999, 999, 999,
        1639, 1276, 5678, 3224, 279, 2645, 684, 13, 198]]).to(device)

    ipdb.set_trace()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    try:
        for i in range(1000):
            outputs = model.forward(
                encoder_input_ids, sc_tagalong,
                decoder_input_ids, transf,
                encoder_attention_mask, decoder_attention_mask,
                labels=labels)
            optim.zero_grad()
            outputs[0].backward()
            optim.step()
            print(outputs[0].item())
    finally:
        print("I tried to delete stuff but you wouldn't listen.")
        del(model)


def playground_2(device='cpu', stuff=None):
    if stuff is None:
        corpus = loader.Corpus(DATA_DIR, CACHE_DIR, less_mem=True)
        np.stuff = corpus
    else:
        corpus = stuff

    mean = torch.tensor(np.load('../data/gpt2/mean.npy')).to(device)
    e_vecs = torch.tensor(np.load('../data/gpt2/e_vecs.npy')).to(device)
    sqrt_e_vals = torch.sqrt(
        torch.tensor(np.load('../data/gpt2/e_vals.npy')).to(device))

    tokenizer, gpt2_model = load_model(device=device)
    gpt2_model.transformer.output_hidden_states = True
    lines_dataset = LinesDataset(
        corpus, CONTEXT_MIN_SIZE, tokenizer.eos_token, split=True)

    ldata = [lines_dataset[928387]]
    ipdb.set_trace()
    ldata[0] = list(ldata[0])
    ldata[0][0] = ldata[0][0][:257]
    # orig_str = ldata[0][0]
    print(ldata[0][0])

    constr = (
        [[('RELEVANT', 'he')],
         [('RELEVANT', 'john')],
         [('DESCR', 'amount', 'DIRECT', 'huge', 'POS'),
          ('RELEVANT', 'amount')],
         [('RELEVANT', 'drug')],
         [('RELEVANT', 'they')],
         [('RELEVANT', 'customs')],
         [('RELEVANT', 'that')]],
        [('ACTION', 'john', 'pass', 'they', 'POS', 'REAL'),
         ('ACTION', 'amount', '[LINKED_TO]', 'drug', 'POS', 'REAL'),
         ('ACTION', '[NONE]', 'do', 'that', 'POS', 'REAL'),
         ('ACTION', 'john', 'buy', 'amount', 'POS', 'TOPIC'),
         ('ACTION', 'he', 'get', '[CMPLX]', 'POS', 'REAL'),
         ('ACTION', 'he', '[ACT_LINK]', 'customs', 'POS', 'REAL')],
        [])
    # constr = (
    #     [[('DESCR', 'he', 'DIRECT', 'lucky', 'POS'),
    #       ('RELEVANT', 'he')],
    #      [('RELEVANT', 'friend'),
    #       ('DESCR', 'friend', 'DIRECT', 'good', 'POS')],
    #      [('RELEVANT', 'guard'),
    #       ('DESCR', 'guard', 'ACTION', 'just', 'POS')],
    #      [('RELEVANT', 'prison')]],
    #     [('ACTION', 'he', '[BE]', 'friend', 'POS', 'REAL'),
    #      ('ACTION', 'guard', 'release', 'he', 'POS', 'TOPIC'),
    #      ('ACTION', 'guard', 'decide', '[CMPLX]', 'POS', 'TOPIC'),
    #      ('ACTION', 'guard', '[ACT_LINK]', 'prison', 'POS', 'REAL'),
    #      ('ACTION', 'friend', '[LINKED_TO]', 'prison', 'POS', 'REAL')],
    #     [])
    ldata[0][1] = list(ldata[0][1])
    ldata[0][1][-1] = loader.replace_with_tags(
        corpus, *constr, train=False)

    encoder_config = sc_gpt2_model.GPT2Config(
        n_layer=2, n_head=16, n_embd=1024)
    decoder_config = sc_gpt2_model.GPT2Config(
        n_layer=4, n_head=16, n_embd=1024)
    config = sc_gpt2_model.SCGPT2Config(
        corpus.bitvec_size(), sqrt_e_vals.shape[0],
        encoder_config, decoder_config)
    model = sc_gpt2_model.SCGPT2HeadModel(config)
    model.load_state_dict(torch.load(
        '../data/gpt2/checkpt/model_3_202657.pth', map_location='cpu'))
    model = model.to(device)
    model.attach_gpt2_model(gpt2_model, mean, e_vecs)
    model.eval()
    # past = None
    all_sentences = []
    constr_tagged = loader.replace_with_tags(
        corpus, *constr, train=False)

    for i in range(10):
        ldata_fake = [[ldata[0][0] + tokenizer.eos_token, ldata[0][1]]]
        # id_bit, m_bit = prepare_batch(
        #     tokenizer, [ldata_fake[0][0]],
        #     device, single=True)
        # id_bit = id_bit[0]
        # m_bit = m_bit[0]
        with torch.no_grad():
            # outputs = gpt2_model(id_bit, attention_mask=m_bit)
            # embed_dim = outputs[2][0].shape[-1]
            # pieces = [
            #     (outputs[2][i] * m_bit[:, :, None]).view(-1, embed_dim)
            #     for i in range(1, len(outputs[2]))
            # ]
            # vec = torch.cat(pieces, dim=1)
            # vec = torch.sign(vec) * torch.log(1 + torch.abs(vec))
            # del(outputs)
            # del(pieces)
            # vec = vec[m_bit.view(-1) > 0.5]
            # transf_batch = (vec - mean) @ e_vecs.T

            split_pieces = prepare_sc_batch(
                corpus, tokenizer, ldata_fake, None, device,
                peturb=False, single=True, remove_last_eos=True)
            # if past is not None:
            #     split_pieces[0]['encoder_input_ids'] = \
            #         split_pieces[0]['encoder_input_ids'][:, -1:]
            #     split_pieces[0]['decoder_input_ids'] = \
            #         split_pieces[0]['decoder_input_ids'][:, -1:]
            #     split_pieces[0]['sc_tagalong'] = \
            #         split_pieces[0]['sc_tagalong'][:, -1:]
            #     split_pieces[0]['labels'] = \
            #         split_pieces[0]['labels'][:, -1:]
            del(split_pieces[0]['labels'])
            model.prepare_forward(**split_pieces[0], num_copies=3)
            output = model.generate(
                max_length=100,
                num_return_sequences=3, do_sample=True)
            for sentence in output:
                eos_places = (sentence == tokenizer.eos_token_id).nonzero()
                if eos_places.shape[0] > 1:
                    end = eos_places[1, 0]
                else:
                    end = sentence.shape[0]
                sentence = tokenizer.decode(sentence[1:end])
                all_sentences.append(sentence)
                print(sentence)
            # outputs = model.forward_with_gpt2(**split_pieces[0], past=past)
            # logits = outputs[0]
            # gumbel = -torch.log(
            #     -torch.log(torch.rand(logits.shape[2]))).to(device)
            # choice = (logits[0, -1] + gumbel).argmax()
            # text = tokenizer.decode(choice.item())
            # ldata[0][0] = ldata[0][0] + text
            # past = outputs[1]
            # print(ldata[0][0])
            # if ldata[0][0].endswith('<|endoftext|>'):
            #     ldata[0][0] = orig_str
            #     past = None

    best_similarity = None
    best_sentence = None
    parser = loader.construct_parser('../data/parser.tar.gz')
    for sentence in all_sentences:
        parse = loader.get_parse(parser, [sentence])[0]
        hypo_constr = loader.clean_up_bits(loader.process(parse))
        hypo_constr_tagged = loader.replace_with_tags(
            corpus, *hypo_constr, train=False)
        similarity = loader.sc_similarity(constr_tagged, hypo_constr_tagged)
        if best_similarity is None or similarity > best_similarity:
            best_similarity = similarity
            best_sentence = sentence
    print("Best sentence was: %s" % best_sentence)
