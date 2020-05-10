import torch
from external.modeling_gpt2 import *
from external.configuration_gpt2 import *


class SCGPT2Config(PretrainedConfig):
    def __init__(self, n_tagalong, n_transf, encoder, decoder):
        self.n_tagalong = n_tagalong
        self.n_transf = n_transf
        self.encoder = encoder
        self.decoder = decoder
        self.initializer_range = self.encoder.initializer_range
        if encoder.n_embd != decoder.n_embd:
            raise RuntimeError("Embedding size should be the same")


def get_shrink_data(attention_mask):
    batch_size = attention_mask.shape[0]
    lens = attention_mask.sum(axis=1)
    longest = lens.max().item()
    full = attention_mask.shape[1]
    fill = torch.nonzero(torch.arange(
        longest, device=attention_mask.device)[None, :] <
        lens[:, None], as_tuple=True)
    take = torch.nonzero(attention_mask, as_tuple=True)
    return ((batch_size, full, longest), (take, fill))


def concat_shrink_data(past_shrink_data, new_shrink_data):
    (pbatch_size, pfull, plongest), (ptake, pfill) = past_shrink_data
    (nbatch_size, nfull, nlongest), (ntake, nfill) = new_shrink_data
    batch_size = pbatch_size
    full = pfull + nfull
    longest = plongest + nlongest
    take = (
        torch.cat([ptake[0], ntake[0]], dim=0),
        torch.cat([ptake[1], pfull + ntake[1]], dim=0))
    fill = (
        torch.cat([pfill[0], nfill[0]], dim=0),
        torch.cat([pfill[1], plongest + nfill[1]], dim=0))
    return (batch_size, full, longest), (take, fill)


class TwoSourceAttention(nn.Module):
    def __init__(self, nx, ny, n_ctx, config, scale=False):
        super().__init__()
        self.output_attentions = config.output_attentions

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to
        # keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.register_buffer(
            "bias", torch.tril(
                torch.ones((n_ctx, n_ctx), dtype=torch.uint8)).view(
                1, 1, n_ctx, n_ctx)
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))
        self.n_head = config.n_head
        self.split_size = n_state
        self.split_size_y = ny
        self.scale = scale

        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_attn_2a = Conv1D(n_state, nx)
        self.c_attn_2b = Conv1D(n_state * 2, ny)
        self.c_proj = Conv1D(n_state, n_state * 2)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.n_head, self.split_size // self.n_head)
        # Convert to set and emove already pruned heads
        heads = set(heads) - self.pruned_heads
        for head in heads:
            # Compute how many pruned heads are before the head
            # and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        index_3 = torch.cat(
            [index, index + self.split_size, index + (2 * self.split_size)])
        index_2 = torch.cat(
            [index, index + self.split_size])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_3, dim=1)
        self.c_attn_2a = prune_conv1d_layer(self.c_attn_2a, index, dim=1)
        self.c_attn_2b = prune_conv1d_layer(self.c_attn_2b, index_2, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index_2, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.n_head) * \
            (self.n_head - len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _expand(self, w, v, q_shrink_data, k_shrink_data):
        batch_size, head, _, head_features = v.shape
        (_, full_q, longest_q), (q_take, q_fill) = q_shrink_data
        (_, full_k, longest_k), (k_take, k_fill) = k_shrink_data

        temp = w.new_zeros(batch_size, head, longest_q, full_k)
        temp[k_take[0], :, :, k_take[1]] = w[k_fill[0], :, :, k_fill[1]]
        new_w = w.new_zeros(batch_size, head, full_q, full_k)
        new_w[q_take[0], :, q_take[1], :] = temp[q_fill[0], :, q_fill[1], :]

        new_v = v.new_zeros(batch_size, head, full_k, head_features)
        new_v[k_take[0], :, k_take[1], :] = v[k_fill[0], :, k_fill[1], :]

        new_attention_mask = v.new_ones(
            batch_size, 1, 1, full_k) * -10000
        new_attention_mask[k_take[0], :, :, k_take[1]] = 0
        return new_w, new_v, new_attention_mask

    def _shrink(self, a, q_shrink_data):
        batch_size, head, _, head_features = a.shape
        (_, full_q, longest_q), (q_take, q_fill) = q_shrink_data
        new_a = a.new_zeros(batch_size, head, longest_q, head_features)
        new_a[q_fill[0], :, q_fill[1], :] = a[q_take[0], :, q_take[1], :]
        return new_a

    def _attn(self, q, k, v, attention_mask, head_mask=None, expand=None):
        w = torch.matmul(q, k)
        if expand is not None:
            (q_shrink_data, k_shrink_data) = expand
            w, v, attention_mask = self._expand(
                w, v, q_shrink_data, k_shrink_data)

        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns - nd: ns, :ns]
        w = w * b - 1e4 * (1 - b)

        # Apply the attention mask
        w = w + attention_mask

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        a = torch.matmul(w, v)
        if expand is not None:
            (q_shrink_data, k_shrink_data) = expand
            a = self._shrink(a, q_shrink_data)

        outputs = [a]
        if self.output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            # (batch, head, head_features, seq_length)
            return x.permute(0, 2, 3, 1)
        else:
            # (batch, head, seq_length, head_features)
            return x.permute(0, 2, 1, 3)

    def setup(self, y, attention_mask_y, y_shrink_data, x_shrink_data):
        self.y = y
        self.attention_mask_y = attention_mask_y
        self.y_shrink_data = y_shrink_data
        self.x_shrink_data = x_shrink_data

    def forward(self, x, layer_past=None,
                attention_mask=None, head_mask=None):
        x_old = x
        attention_mask_x = attention_mask
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            # transpose back cf below
            past_key, past_value = layer_past[0].transpose(
                -2, -1), layer_past[1]
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)
        present = torch.stack((key.transpose(-2, -1), value))

        attn_outputs = self._attn(
            query, key, value, attention_mask_x, head_mask,
            expand=None)
        a = attn_outputs[0]
        a = self.merge_heads(a)

        x = x_old
        y = self.y
        attention_mask_y = self.attention_mask_y
        y_shrink_data = self.y_shrink_data
        x_shrink_data = self.x_shrink_data
        del(self.y)
        del(self.attention_mask_y)
        del(self.y_shrink_data)
        del(self.x_shrink_data)

        x = self.c_attn_2a(x)
        query = x
        y = self.c_attn_2b(y)
        key, value = y.split(self.split_size_y, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        attn_outputs_2 = self._attn(
            query, key, value, attention_mask_y, head_mask,
            expand=(x_shrink_data, y_shrink_data))
        a_2 = attn_outputs_2[0]
        a_2 = self.merge_heads(a_2)

        a_both = self.c_proj(torch.cat([a, a_2], dim=2))
        a_both = self.resid_dropout(a_both)

        outputs = [a_both, present] + list(
            zip(attn_outputs[1:], attn_outputs_2[1:]))
        return outputs  # a, present, (attentions)


class TwoSourceBlock(Block):
    def __init__(self, n_ctx, config, scale=False):
        super().__init__(n_ctx, config.decoder)
        nx = config.decoder.n_embd
        ny = config.encoder.n_embd
        self.ln_1 = nn.LayerNorm(nx, eps=config.decoder.layer_norm_epsilon)
        self.attn = TwoSourceAttention(nx, ny, n_ctx, config.decoder, scale)
        self.ln_2 = nn.LayerNorm(nx, eps=config.decoder.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config.decoder)

    def setup(self, y, attention_mask_y, y_shrink_data, x_shrink_data):
        self.attn.setup(y, attention_mask_y, y_shrink_data, x_shrink_data)


class AddMultConst(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mult = torch.nn.Parameter(torch.ones(dim))
        self.bias = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, vals):
        return vals * self.mult + self.bias


class SCGPT2EmbeddingModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(
            config.decoder.vocab_size, config.decoder.n_embd)
        self.lm_head = nn.Linear(
            config.decoder.n_embd, config.decoder.vocab_size, bias=False)
        self.encoder_rescale = AddMultConst(config.n_tagalong)
        self.encoder_proj = nn.Linear(
            config.encoder.n_embd + config.n_tagalong, config.encoder.n_embd)
        self.decoder_proj = nn.Linear(
            config.decoder.n_embd + config.n_transf, config.decoder.n_embd)
        torch.nn.init.normal_(
            self.encoder_proj.weight.data, std=config.initializer_range)
        torch.nn.init.normal_(
            self.decoder_proj.weight.data, std=config.initializer_range)

    def sabotage_gpt2(self, gpt2):
        self.wte.load_state_dict(
            gpt2.get_input_embeddings().state_dict())
        self.lm_head.load_state_dict(
            gpt2.get_output_embeddings().state_dict())

    def novel_parameters(self):
        novel_modules = [
            self.encoder_rescale,
            self.encoder_proj,
            self.decoder_proj
        ]
        return [p for g in novel_modules for p in g.parameters()]

    def novel_named_parameters(self):
        novel_modules = [
            ('encoder_rescale', self.encoder_rescale),
            ('encoder_proj', self.encoder_proj),
            ('decoder_proj', self.decoder_proj)
        ]
        return [(h + '.' + n, p) for h, g in novel_modules
                for n, p in g.named_parameters()]

    def embed_inputs_encoder(self, input_ids, tagalong):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        embed = self.wte(input_ids)
        tagalong = self.encoder_rescale(tagalong)
        return self.encoder_proj(
            torch.cat([embed, tagalong], dim=2))

    def embed_inputs_decoder(self, input_ids, transf):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        embed = self.wte(input_ids)
        return self.decoder_proj(
            torch.cat([embed, transf], dim=2))

    def embed_outputs(self, last_hidden):
        return self.lm_head(last_hidden)


class SCGPT2EncoderModel(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        del(self.wte)

    def forward(
        self,
        past_data,
        attention_mask,
        position_ids,
        inputs_embeds,
        shrink_data,
        token_type_ids=None,
        head_mask=None,
    ):
        (batch_size, full, longest), (take, fill) = shrink_data
        if longest == 0:
            data = attention_mask.new_zeros(batch_size, 0, self.config.n_embd)
            return (data, past_data[1])

        past_shrink_data, past = past_data
        if past_shrink_data is None:
            pfull, plongest = 0, 0
            attention_mask_s = attention_mask.new_zeros(
                batch_size, longest)
        else:
            (_, pfull, plongest), (ptake, pfill) = past_shrink_data
            attention_mask_s = attention_mask.new_zeros(
                batch_size, plongest + longest)
            attention_mask_s[:, :plongest][pfill] = \
                attention_mask[:, :pfull][ptake]
        inputs_embeds_s = inputs_embeds.new_zeros(
            batch_size, longest, inputs_embeds.shape[2])
        position_ids_s = position_ids.new_zeros(batch_size, longest)
        attention_mask_s[:, plongest:][fill] = attention_mask[:, pfull:][take]
        inputs_embeds_s[fill] = inputs_embeds[take]
        position_ids_s[fill] = position_ids[:, pfull:][take]

        return super().forward(
            past=past,
            attention_mask=attention_mask_s,
            position_ids=position_ids_s,
            inputs_embeds=inputs_embeds_s,
            token_type_ids=token_type_ids,
            head_mask=head_mask
        )[:2]


class SCGPT2DecoderModel(GPT2Model):
    def __init__(self, config):
        super().__init__(config.decoder)
        del(self.wte)
        self.h = nn.ModuleList(
            [TwoSourceBlock(config.decoder.n_ctx, config, scale=True)
             for _ in range(config.decoder.n_layer)])
        self.init_weights()

    def setup(self, inputs_embeds, attention_mask_l, position_ids_l,
              encoder_shrink_data, decoder_shrink_data, token_type_ids=None):
        (batch_size, full, longest), (take, fill) = encoder_shrink_data
        attention_mask = attention_mask_l.new_zeros(batch_size, longest)
        position_ids = position_ids_l.new_zeros(batch_size, longest)
        attention_mask[fill] = attention_mask_l[take]
        position_ids[fill] = position_ids_l[take]

        input_shape = inputs_embeds.size()[:-1]
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        position_ids = position_ids.view(-1, input_shape[-1])

        assert batch_size > 0, "batch_size has to be defined and > 0"
        attention_mask = attention_mask.view(batch_size, -1)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = attention_mask.to(
            dtype=next(self.parameters()).dtype)
        attention_mask = (1.0 - attention_mask) * -10000.0

        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)

        for b in self.h:
            b.setup(hidden_states, attention_mask,
                    encoder_shrink_data, decoder_shrink_data)

    def forward(
        self,
        past_data,
        attention_mask,
        position_ids,
        inputs_embeds,
        shrink_data,
        last_encoded_s,
        encoder_attention_mask,
        encoder_shrink_data,
        token_type_ids=None,
        head_mask=None,
    ):
        (batch_size, full, longest), (take, fill) = shrink_data
        if longest == 0:
            data = attention_mask.new_zeros(batch_size, 0, self.config.n_embd)
            return (data, past_data[1])

        self.setup(
            last_encoded_s, encoder_attention_mask, position_ids,
            encoder_shrink_data, shrink_data, token_type_ids)

        past_shrink_data, past = past_data
        if past_shrink_data is None:
            pfull, plongest = 0, 0
            attention_mask_s = attention_mask.new_zeros(
                batch_size, longest)
        else:
            (_, pfull, plongest), (ptake, pfill) = past_shrink_data
            attention_mask_s = attention_mask.new_zeros(
                batch_size, plongest + longest)
            attention_mask_s[:, :plongest][pfill] = \
                attention_mask[:, :pfull][ptake]
        inputs_embeds_s = inputs_embeds.new_zeros(
            batch_size, longest, inputs_embeds.shape[2])
        position_ids_s = position_ids.new_zeros(batch_size, longest)
        attention_mask_s[:, plongest:][fill] = attention_mask[:, pfull:][take]
        inputs_embeds_s[fill] = inputs_embeds[take]
        position_ids_s[fill] = position_ids[:, pfull:][take]

        return super().forward(
            past=past,
            attention_mask=attention_mask_s,
            position_ids=position_ids_s,
            inputs_embeds=inputs_embeds_s,
            token_type_ids=token_type_ids,
            head_mask=head_mask
        )[:2]


class SCGPT2PreTrainedModel(GPT2PreTrainedModel):
    config_class = SCGPT2Config
    pretrained_model_archive_map = {}
    load_tf_weights = load_tf_weights_in_gpt2
    base_model_prefix = ""


class SCGPT2HeadModel(SCGPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config.decoder)
        self.embedding = SCGPT2EmbeddingModel(config)
        self.encoder = SCGPT2EncoderModel(config.encoder)
        self.decoder = SCGPT2DecoderModel(config)
        self.encoder.init_weights()
        self.gpt2_model = None
        self.mean = None
        self.e_vecs = None
        self.stored_info = None
        self.n_tagalong = config.n_tagalong

    def novel_parameters(self):
        return list(self.embedding.novel_parameters()) + list(
            self.encoder.parameters()) + list(self.decoder.parameters())

    def novel_named_parameters(self):
        return [
            ('embedding.%s' % n, p) for n, p in
            self.embedding.novel_named_parameters()] + [
            ('encoder.%s' % n, p) for n, p in
            self.encoder.named_parameters()] + [
            ('decoder.%s' % n, p) for n, p in
            self.decoder.named_parameters()]

    def get_output_embeddings(self):
        return self.embedding.decoder_proj

    def attach_gpt2_model(
        self,
        gpt2_model,
        mean,
        e_vecs,
    ):
        gpt2_model.transformer.output_hidden_states = True
        gpt2_model.lm_head = self.embedding.lm_head
        gpt2_model.transformer.set_input_embeddings(self.embedding.wte)
        self.gpt2_model = gpt2_model
        self.mean = mean
        self.e_vecs = e_vecs

    def prepare_forward(
        self,
        encoder_attention_mask,
        encoder_input_ids,
        sc_tagalong,
        decoder_attention_mask,
        decoder_input_ids,
        num_copies=1,
        past=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
    ):
        encoder_attention_mask = encoder_attention_mask.expand(num_copies, -1)
        encoder_input_ids = encoder_input_ids.expand(num_copies, -1)
        sc_tagalong = sc_tagalong.expand(num_copies, -1, -1)
        decoder_attention_mask = decoder_attention_mask.expand(num_copies, -1)
        decoder_input_ids = decoder_input_ids.expand(num_copies, -1)

        result = self.forward_with_gpt2(
            encoder_attention_mask,
            encoder_input_ids,
            sc_tagalong,
            decoder_attention_mask,
            decoder_input_ids,
            past,
            token_type_ids,
            position_ids,
            head_mask,
        )
        self.stored_info = (
            result[1], encoder_attention_mask, decoder_attention_mask)

    def forward(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
    ):
        if self.stored_info is None:
            raise RuntimeError("Forward hasn't been prepared")
        if inputs_embeds is not None:
            raise RuntimeError("Only support input ids")
        if self.stored_info[1].shape[0] != input_ids.shape[0]:
            raise RuntimeError("Forward preparation had the wrong batch size")
        if attention_mask is not None:
            raise RuntimeError("Cannot change attention mask")

        if past is None:
            past = list(self.stored_info[0])
        else:
            limited_past = past
            past = list(self.stored_info[0])
            num_decoder_past = len(past[4])
            past[4] = limited_past[:num_decoder_past]
            past[5] = limited_past[num_decoder_past:]
            extra_plen = past[4][0].shape[3] - past[3][0][2]
            past[0] = concat_shrink_data(
                past[0], get_shrink_data(input_ids.new_zeros(
                    batch_size, extra_plen, dtype=torch.bool)))
            past[3] = concat_shrink_data(
                past[3], get_shrink_data(input_ids.new_ones(
                    batch_size, extra_plen, dtype=torch.bool)))
        pfull = past[0][0][1]
        batch_size = input_ids.shape[0]
        extra_len = (
            pfull + input_ids.shape[1] - self.stored_info[1].shape[1])
        input_len = input_ids.shape[1]

        encoder_attention_mask = torch.cat([
            self.stored_info[1],
            input_ids.new_zeros(
                batch_size, extra_len, dtype=torch.float)], dim=1)
        decoder_attention_mask = torch.cat([
            self.stored_info[2],
            input_ids.new_ones(
                batch_size, extra_len, dtype=torch.float)], dim=1)
        result = self.forward_with_gpt2(
            encoder_attention_mask,
            input_ids,
            input_ids.new_zeros(
                batch_size, input_len, self.n_tagalong, dtype=torch.float),
            decoder_attention_mask,
            input_ids,
            past,
            token_type_ids,
            position_ids,
            head_mask
        )
        limited_past = result[1][4] + result[1][5]
        return (result[0], limited_past)

    def forward_with_gpt2(
        self,
        encoder_attention_mask,
        encoder_input_ids,
        sc_tagalong,
        decoder_attention_mask,
        decoder_input_ids,
        past=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        labels=None,
    ):
        if self.gpt2_model is None:
            raise RuntimeError("GPT2 model hasn't been attached")
        if position_ids is None:
            input_shape = decoder_attention_mask.size()
            device = decoder_attention_mask.device
            id_range = torch.arange(
                0, input_shape[-1],
                dtype=torch.long, device=device).unsqueeze(0)
            dam_bool = decoder_attention_mask > 0.5
            gpt2_position_ids = (id_range - torch.cumsum(
                ~dam_bool, dim=1)) * dam_bool

        if past is None:
            past_shrink_data, gpt2_past = None, None
            remain_past = None
            pfull = 0
        else:
            past_shrink_data, gpt2_past = past[3], past[5]
            remain_past = past[:-1]
            pfull = past[0][0][1]
        gpt2_position_ids = gpt2_position_ids[:, pfull:]

        decoder_shrink_data = get_shrink_data(
            decoder_attention_mask[:, pfull:] > 0.5)

        (batch_size, full, longest), (take, fill) = decoder_shrink_data
        if past_shrink_data is None:
            pfull, plongest = 0, 0
            attention_mask_s = decoder_attention_mask.new_zeros(
                batch_size, longest)
        else:
            (_, pfull, plongest), (ptake, pfill) = past_shrink_data
            attention_mask_s = decoder_attention_mask.new_zeros(
                batch_size, plongest + longest)
            attention_mask_s[:, :plongest][pfill] = \
                decoder_attention_mask[:, :pfull][ptake]
        inputs_id_s = decoder_input_ids.new_zeros(batch_size, longest)
        position_ids_s = gpt2_position_ids.new_zeros(batch_size, longest)
        attention_mask_s[:, plongest:][fill] = \
            decoder_attention_mask[:, pfull:][take]
        inputs_id_s[fill] = decoder_input_ids[take]
        position_ids_s[fill] = gpt2_position_ids[take]

        if longest == 0:
            transf = attention_mask_s.new_zeros(
                batch_size, full, self.config.n_transf)
        else:
            gpt2_outputs = self.gpt2_model(
                inputs_id_s,
                attention_mask=attention_mask_s,
                position_ids=position_ids_s,
                past=gpt2_past)
            pieces = [
                gpt2_outputs[2][i] for i in range(1, len(gpt2_outputs[2]))
            ]
            vec = torch.cat(pieces, dim=2)
            vec = torch.sign(vec) * torch.log(1 + torch.abs(vec))
            gpt2_present = gpt2_outputs[1]
            del(gpt2_outputs)
            del(pieces)
            transf_s = (vec - self.mean) @ self.e_vecs.T
            transf = transf_s.new_zeros(batch_size, full, transf_s.shape[2])
            transf[take] = transf_s[fill]

        result = self.forward_full(
            encoder_attention_mask,
            encoder_input_ids,
            sc_tagalong,
            decoder_attention_mask,
            decoder_input_ids,
            transf,
            remain_past,
            token_type_ids,
            position_ids,
            head_mask,
            labels
        )
        if labels is not None:
            present = (*result[2], gpt2_present)
            return (*result[:2], present, *result[3:])
        else:
            present = (*result[1], gpt2_present)
            return (result[0], present, *result[2:])

    def forward_full(
        self,
        encoder_attention_mask,
        encoder_input_ids,
        sc_tagalong,
        decoder_attention_mask,
        decoder_input_ids,
        transf,
        past=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        labels=None,
    ):
        encoder_embeds = self.embedding.embed_inputs_encoder(
            encoder_input_ids, sc_tagalong)
        decoder_embeds = self.embedding.embed_inputs_decoder(
            decoder_input_ids, transf)

        if position_ids is None:
            input_shape = encoder_attention_mask.size()
            device = encoder_attention_mask.device
            id_range = torch.arange(
                0, input_shape[-1],
                dtype=torch.long, device=device).unsqueeze(0)
            eam_bool = encoder_attention_mask > 0.5
            dam_bool = decoder_attention_mask > 0.5
            encoder_position_ids = (id_range - torch.cumsum(
                ~eam_bool, dim=1)) * eam_bool
            decoder_position_ids = (id_range - torch.cumsum(
                ~dam_bool, dim=1)) * dam_bool
            position_ids = encoder_position_ids + decoder_position_ids

        if past is None:
            past = (None, None, None, None, None)
            pfull = 0
        else:
            pfull = past[0][0][1]

        encoder_shrink_data = get_shrink_data(
            encoder_attention_mask[:, pfull:] > 0.5)
        encoder_outputs = self.encoder(
            past_data=(past[0], past[1]),
            attention_mask=encoder_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=encoder_embeds,
            shrink_data=encoder_shrink_data
        )

        last_encoded_s = encoder_outputs[0]
        if past[2] is not None:
            last_encoded_s = torch.cat([past[2], last_encoded_s], dim=1)
            encoder_shrink_data = concat_shrink_data(
                past[0], encoder_shrink_data)

        decoder_shrink_data = get_shrink_data(
            decoder_attention_mask[:, pfull:] > 0.5)
        decoder_outputs = self.decoder(
            past_data=(past[3], past[4]),
            attention_mask=decoder_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=decoder_embeds,
            shrink_data=decoder_shrink_data,
            last_encoded_s=last_encoded_s,
            encoder_attention_mask=encoder_attention_mask,
            encoder_shrink_data=encoder_shrink_data,
        )

        hidden_states_s = decoder_outputs[0]
        lm_logits_s = self.embedding.embed_outputs(hidden_states_s)
        if labels is not None:
            (batch_size, full, longest), (take, fill) = decoder_shrink_data
            labels_s = labels.new_zeros(batch_size, longest)
            labels_s[fill] = labels[take]
            decoder_attention_mask_s = \
                decoder_attention_mask[:, pfull:].new_zeros(
                    batch_size, longest)
            decoder_attention_mask_s[fill] = decoder_attention_mask[
                :, pfull:][take]

        if past[3] is not None:
            decoder_shrink_data = concat_shrink_data(
                past[3], decoder_shrink_data)
        present = (
            encoder_shrink_data,
            encoder_outputs[1],
            last_encoded_s,
            decoder_shrink_data,
            decoder_outputs[1]
        )
        outputs = (lm_logits_s, present)
        if labels is not None:
            batch_shape = lm_logits_s.shape[:-1]
            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(lm_logits_s.view(
                -1, lm_logits_s.size(-1)),
                labels_s.view(-1)).view(*batch_shape)
            loss = (loss * decoder_attention_mask_s).sum()
            loss = loss / decoder_attention_mask_s.sum()
            outputs = (loss,) + outputs

        return outputs
