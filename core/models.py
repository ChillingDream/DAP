import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from typing import List, Optional, Tuple, Union

import transformers
from transformers import RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead, BertOnlyMLMHead, BertEmbeddings
from transformers.activations import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions, MaskedLMOutput

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls" or \
        cls.model_args.pooler_type == 'cls_before_pooler' and cls.model_args.do_rtl:
        cls.mlp = MLPLayer(config)
    if cls.model_args.rtl_pattern in ['source', 'both', 'TLM']:
        cls.decoder_lang_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()


def cl_forward(cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    langs=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    rtl_mask=None,
    rtl_labels=None,
    tlm_input_ids=None,
    tlm_labels=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)
    len_sent = input_ids.size(2)

    if cls.model_args.do_rtl:
        src_attention_mask, tgt_attention_mask = attention_mask[:, 0, :], attention_mask[:, 1, :]
        source_segments = torch.zeros_like(src_attention_mask, device=input_ids.device)
        target_segments = torch.zeros_like(tgt_attention_mask, device=input_ids.device)
        # prepare [MASK] sequence
        masked_src_ids = torch.where(src_attention_mask.byte(),
                                   torch.tensor(cls.model_args.mask_token_id, device=input_ids.device),
                                   torch.tensor(cls.model_args.pad_token_id, device=input_ids.device))
        masked_tgt_ids = torch.where(tgt_attention_mask.byte(),
                                     torch.tensor(cls.model_args.mask_token_id, device=input_ids.device),
                                     torch.tensor(cls.model_args.pad_token_id, device=input_ids.device))

    tlm_outputs = None
    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    langs = langs.view(-1, langs.size(-1))
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        langs=langs,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    # TLM auxiliary objective
    if tlm_input_ids is not None:
        tlm_input_ids = tlm_input_ids.view((tlm_input_ids.size(0), -1)) # (bs, num_sent * len)
        tlm_langs = langs.view((tlm_input_ids.size(0), -1))
        tlm_attention_mask = attention_mask.view_as(tlm_input_ids)
        tlm_token_type_ids = token_type_ids.view_as(tlm_input_ids) if token_type_ids is not None else None
        tlm_position_ids = torch.arange(input_ids.size(-1), device=tlm_input_ids.device).repeat((tlm_input_ids.size(0), 2))
        tlm_outputs = encoder(
            tlm_input_ids,
            langs=tlm_langs,
            attention_mask=tlm_attention_mask,
            token_type_ids=tlm_token_type_ids,
            position_ids=tlm_position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )

    # Pooling
    output_seq = outputs.last_hidden_state.view(batch_size, num_sent, len_sent, outputs.last_hidden_state.size(-1))
    src_seq, tgt_seq = output_seq[:, 0, :, :].contiguous(), output_seq[:, 1, :, :].contiguous()
    pooler_output = cls.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.model_args.do_rtl and cls.pooler_type == 'cls_before_pooler':
        z1, z2 = pooler_output[:,0], pooler_output[:,1]
        pooler_output = cls.mlp(pooler_output)
        r1, r2 = pooler_output[:,0], pooler_output[:,1]
    else:
        if cls.pooler_type == "cls":
            pooler_output = cls.mlp(pooler_output)
        # Separate representation
        z1, z2 = pooler_output[:,0], pooler_output[:,1]
        r1, r2 = z1, z2

    if cls.model_args.do_rtl:
        src_mask_embedding = getattr(cls.decoder, cls.decoder.base_model_prefix).embeddings(
            input_ids=masked_src_ids,
            token_type_ids=source_segments,
        )
        tgt_mask_embedding = getattr(cls.decoder, cls.decoder.base_model_prefix).embeddings(
            input_ids=masked_tgt_ids,
            token_type_ids=target_segments,
        )

        rtl_loss = 0.
        if cls.model_args.rtl_pattern == 'TLM':
            # mask and reconstruction both source and target representation
            src_mask_embedding = torch.where(rtl_mask[0].unsqueeze(-1), src_mask_embedding, src_seq)
            tgt_mask_embedding = torch.where(rtl_mask[1].unsqueeze(-1), tgt_mask_embedding, tgt_seq)
            rtlm_embedding = torch.cat([src_mask_embedding, tgt_mask_embedding], dim=1)
            rtlm_attn_mask = torch.cat([src_attention_mask, tgt_attention_mask], dim=1)
            rtlm_attn_mask[:, [0, src_mask_embedding.size(1)]] = 0
            rtlm_labels = rtl_labels.clone()
            rtlm_labels[:, [0, src_mask_embedding.size(1)]] = -100
            rtl_loss = cls.decoder(
                inputs_embeds=rtlm_embedding,
                token_type_ids=target_segments.repeat(1, 2),
                attention_mask=rtlm_attn_mask,
                labels=rtlm_labels
            ).loss
        if cls.model_args.rtl_pattern in ['target', 'both']:
            # mask and reconstruct target sequence
            if cls.model_args.do_masked_rtl:
                # mask source representation
                src_mask = torch.rand((batch_size, len_sent)) < 0.5
                src_mask = src_mask.to(src_mask_embedding.device)
                masked_src_seq = torch.where(src_mask.unsqueeze(-1), src_mask_embedding, src_seq)
            else:
                masked_src_seq = src_seq
            tgt_mask_embedding = torch.where(rtl_mask[1].unsqueeze(-1), tgt_mask_embedding, tgt_seq)
            tgt_expanded_mask_embedding = torch.cat([masked_src_seq, tgt_mask_embedding], dim=1)
            tgt_mask_attn_mask = torch.cat([src_attention_mask, tgt_attention_mask], dim=1)
            # block [CLS] token
            tgt_mask_attn_mask[:, [0, masked_src_seq.size(1)]] = 0
            tgt_rtl_labels = rtl_labels.clone()
            tgt_rtl_labels[:, :masked_src_seq.size(1)] = -100
            tgt_rtl_labels[:, masked_src_seq.size(1)] = -100
            if cls.model_args.rtl_pattern == 'both':
                half_bs = batch_size // 2
                tgt_expanded_mask_embedding = tgt_expanded_mask_embedding[half_bs:]
                target_segments = target_segments[half_bs:]
                tgt_mask_attn_mask = tgt_mask_attn_mask[half_bs:]
                tgt_rtl_labels = tgt_rtl_labels[half_bs:]
            rtl_loss += cls.decoder(
                inputs_embeds=tgt_expanded_mask_embedding,
                token_type_ids=target_segments.repeat(1, 2),
                attention_mask=tgt_mask_attn_mask,
                labels=tgt_rtl_labels
            ).loss
        if cls.model_args.rtl_pattern in ['source', 'both']:
            # mask and reconstruct source sequence
            src_mask_embedding = torch.where(rtl_mask[0].unsqueeze(-1), src_mask_embedding, src_seq)
            langs = langs.view(-1, 2, langs.size(-1))
            src_mask_embedding += cls.decoder_lang_embedding(langs[:, 0])
            src_expanded_mask_embedding = torch.cat([src_mask_embedding, tgt_seq], dim=1)
            src_mask_attn_mask = torch.cat([src_attention_mask, tgt_attention_mask], dim=1)
            src_mask_attn_mask[:, [0, src_mask_embedding.size(1)]] = 0
            src_rtl_labels = rtl_labels.clone()
            src_rtl_labels[:, src_mask_embedding.size(1):] = -100
            src_rtl_labels[:, 0] = -100
            if cls.model_args.rtl_pattern == 'both':
                half_bs = batch_size // 2
                src_expanded_mask_embedding = src_expanded_mask_embedding[:half_bs]
                source_segments = source_segments[:half_bs]
                src_mask_attn_mask = src_mask_attn_mask[:half_bs]
                src_rtl_labels = src_rtl_labels[:half_bs]
            rtl_loss += cls.decoder(
                inputs_embeds=src_expanded_mask_embedding,
                token_type_ids=source_segments.repeat(1, 2),
                attention_mask=src_mask_attn_mask,
                labels=src_rtl_labels
            ).loss
        if cls.model_args.rtl_pattern == 'both':
            rtl_loss /= 2
    
    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:
        # Dummy vectors for allgather
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        # Allgather
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        z1_list[dist.get_rank()] = z1
        z2_list[dist.get_rank()] = z2
        # Get full batch embeddings: (bs x N, hidden)
        z1 = torch.cat(z1_list, 0)
        z2 = torch.cat(z2_list, 0)

    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    assert cos_sim.size(0) == cos_sim.size(1)
    cos_sim -= torch.eye(cos_sim.size(0), device=cos_sim.device) * cls.model_args.ams_margin

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = nn.CrossEntropyLoss()

    if cls.model_args.cl_dir == 'fwd':
        loss = loss_fct(cos_sim, labels)
    elif cls.model_args.cl_dir == 'both':
        loss = (loss_fct(cos_sim, labels)  + loss_fct(cos_sim.transpose(0, 1), labels)) / 2
    else:
        raise NotImplementedError
    if cls.model_args.do_rtl:
        loss += rtl_loss * cls.model_args.rtl_weight
    else:
        rtl_loss = torch.tensor(0.0).to(cls.device)

    # Calculate loss for tlm
    if tlm_outputs is not None and tlm_labels is not None:
        prediction_scores = cls.lm_head(tlm_outputs.last_hidden_state)
        tlm_loss = loss_fct(prediction_scores.view(-1, cls.config.vocab_size), tlm_labels.view(-1))
        loss = loss + cls.model_args.tlm_weight * tlm_loss
    else:
        tlm_loss = torch.tensor(0.0).to(cls.device)

    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    loss = loss, rtl_loss, tlm_loss
    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    langs=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    batch_size = input_ids.size(0)
    input_ids = input_ids.view(-1, input_ids.size(-1))
    langs = langs.view(-1, langs.size(-1))
    attention_mask = attention_mask.view(-1, attention_mask.size(-1))
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))

    outputs = encoder(
        input_ids,
        langs=langs,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)
    pooler_output = pooler_output.view(batch_size, -1, pooler_output.size(-1))

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


class SimCSEModel:
    def forward(self,
        input_ids=None,
        langs=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        rtl_mask=None,
        rtl_labels=None,
        tlm_input_ids=None,
        tlm_labels=None,
    ):
        model = getattr(self, self.base_model_prefix)
        if sent_emb:
            return sentemb_forward(self, model,
                input_ids=input_ids,
                langs=langs,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, model,
                input_ids=input_ids,
                langs=langs,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                rtl_mask=rtl_mask,
                rtl_labels=rtl_labels,
                tlm_input_ids=tlm_input_ids,
                tlm_labels=tlm_labels,
            )


class BertOnlyRTLHead(BertOnlyMLMHead):
    def shrink(self, ids):
        self.predictions.decoder.weight = nn.Parameter(self.predictions.decoder.weight.data[ids])
        self.predictions.bias = nn.Parameter(self.predictions.bias.data[ids])
        self.predictions.decoder.bias = self.predictions.bias
        self.predictions.decoder.out_features = len(ids)

class RobertaRTLHead(RobertaLMHead):
    def shrink(self, ids):
        self.decoder.weight = nn.Parameter(self.decoder.weight.data[ids])
        self.bias = nn.Parameter(self.bias.data[ids])
        self.decoder.bias = self.bias
        self.decoder.out_features = len(ids)


class BertForRTL(SimCSEModel, BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        BertPreTrainedModel.__init__(self, config)
        self.model_args = model_kargs["model_args"]
        self.bert = XLMBertModel(config, add_pooling_layer=False)

        if self.model_args.do_tlm:
            self.lm_head = BertLMPredictionHead(config)

        cl_init(self, config)
    

class RobertaForRTL(SimCSEModel, RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        RobertaPreTrainedModel.__init__(self, config)
        self.model_args = model_kargs["model_args"]
        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)

        if self.model_args.do_tlm:
            self.lm_head = RobertaLMHead(config)

        cl_init(self, config)


class XLMEmbeddings(BertEmbeddings):
    def __init__(self, config):
        super().__init__(config)
        if hasattr(config, 'n_langs') and config.n_langs > 1 and config.use_lang_emb:
            self.lang_embeddings = nn.Embedding(config.n_langs, config.hidden_size)

    def forward(
        self, input_ids=None, langs=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings

        if langs is not None and hasattr(self, 'lang_embeddings'):
            lang_embeddings = self.lang_embeddings(langs)
            embeddings += lang_embeddings

        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class XLMBertModel(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.embeddings = XLMEmbeddings(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        langs: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            langs=langs,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class XLMRobertaModel(RobertaModel):
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        langs: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
