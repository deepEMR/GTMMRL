# coding=utf-8
# Copyright 2018 Hao Tan, Mohit Bansal, and the HuggingFace team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch GTX model. """

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, SmoothL1Loss
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from configuration import GTXConfig
from transformers.activations import ACT2FN, gelu
from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    # add_start_docstrings_to_callable,
    replace_return_docstrings,
)
from transformers.modeling_utils import PreTrainedModel
# Logging tool
from utils.notifier import logging, log_formatter
import copy

notifier = logging.getLogger(__name__)
notifier.addHandler(log_formatter())


class GeLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


@dataclass
class CrossEncoderOutput(ModelOutput):
    lang_feats: Optional[torch.FloatTensor] = None
    kg_feats: Optional[torch.FloatTensor] = None
    kg_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    language_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    cross_encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class GTXModelOutput(ModelOutput):
    """
    GTX's outputs that contain the last hidden states, pooled outputs, and attention probabilites for the language,
    visual, and, cross-modality encoders. (note: the visual encoder in GTX is referred to as the "relation-ship"
    encoder")


    Args:
        language_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the language encoder.
        vision_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the visual encoder.
        pooled_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification, CLS, token) further processed
            by a Linear layer and a Tanh activation function. The Linear
        language_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for input features + one for the output of each cross-modality
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.
        vision_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for input features + one for the output of each cross-modality
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.
        language_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        vision_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        cross_encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
    """
    loss_mlm: [torch.FloatTensor] = None
    loss_ita: [torch.FloatTensor] = None
    loss_gtm: [torch.FloatTensor] = None
    loss_rc: [torch.FloatTensor] = None
    loss_mlp: [torch.FloatTensor] = None
    language_prediction_scores: [torch.FloatTensor] = None
    kg_prediction_scores: [torch.FloatTensor] = None
    language_output: Optional[torch.FloatTensor] = None
    kg_output: Optional[torch.FloatTensor] = None
    pooled_output: Optional[torch.FloatTensor] = None
    language_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    kg_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    language_attentions: Optional[Tuple[torch.FloatTensor]] = None
    kg_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class GTXForPreTrainingOutput(ModelOutput):
    """
    Output type of :class:`~transformers.GTXForPreTrainingModel`.

    Args:
        loss (`optional`, returned when ``labels`` is provided, ``torch.FloatTensor`` of shape :obj:`(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        cross_relationship_score: (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, 2)`):
            Prediction scores of the textual matching objective (classification) head (scores of True/False
            continuation before SoftMax).
        question_answering_score: (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, n_qa_answers)`):
            Prediction scores of question answering objective (classification).
        language_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for input features + one for the output of each cross-modality
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.
        vision_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for input features + one for the output of each cross-modality
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.
        language_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        vision_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        cross_encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.

    """

    loss: [torch.FloatTensor] = None
    loss_dict: Optional[dict] = None
    lang_prediction_logits: Optional[torch.FloatTensor] = None
    kg_prediction_logits: Optional[torch.FloatTensor] = None
    cross_relationship_score: Optional[torch.FloatTensor] = None
    language_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    kg_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    language_attentions: Optional[Tuple[torch.FloatTensor]] = None
    kg_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class GTXForDownstreamOutput(ModelOutput):
    """
    Output type of :class:`~transformers.GTXForPreTrainingModel`.

    Args:
        loss (`optional`, returned when ``labels`` is provided, ``torch.FloatTensor`` of shape :obj:`(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        pooled_logits: (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, 2)`):
            Pooled logit from two [CLS] pooling token
        question_answering_score: (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, n_qa_answers)`):
            Prediction scores of question answering objective (classification).
        language_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for input features + one for the output of each cross-modality
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.
        vision_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for input features + one for the output of each cross-modality
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.
        language_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        vision_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        cross_encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.

    """

    loss: [torch.FloatTensor] = None
    loss_dict: Optional[dict] = None
    lang_prediction_logits: Optional[torch.FloatTensor] = None
    kg_prediction_logits: Optional[torch.FloatTensor] = None
    pooled_logits: Optional[torch.FloatTensor] = None
    language_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    kg_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    language_attentions: Optional[Tuple[torch.FloatTensor]] = None
    kg_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


def load_tf_weights_in_GTX(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re

        import numpy as np
        import tensorflow as tf
    except ImportError:
        notifier.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    notifier.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        notifier.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
                n
                in [
                    "adam_v",
                    "adam_m",
                    "AdamWeightDecayOptimizer",
                    "AdamWeightDecayOptimizer_1",
                    "global_step",
                ]
                for n in name
        ):
            notifier.info("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    notifier.info("Skipping {}".format("/".join(name)))
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        notifier.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


class GTXEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config, input_type=None):
        super().__init__()
        # print('test ----- uuuuu')
        self.word_embeddings = nn.Embedding(config.vocab_size[input_type], config.hidden_size)
        if "linearize" in config.KnowMix:
            self.position_embeddings = nn.Embedding(2048, config.hidden_size)
        elif config.max_position_embeddings[input_type] > 0:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings[input_type], config.hidden_size)
        else:
            self.position_embeddings = None
        if config.type_vocab_size[input_type] > 0:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size[input_type], config.hidden_size)
        else:
            self.token_type_embeddings = None
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, inputs_embeds=None):

        # notifier.critical("test ------ 1.2.1.1")
        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        else:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        seq_length = input_shape[1]

        # notifier.critical("test ------ 1.2.1.2")

        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)

        # notifier.critical("test ------ 1.2.1.3")

        if token_type_ids is None and self.token_type_embeddings is not None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=position_ids.device)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        embeddings = inputs_embeds

        # notifier.critical("test ------ 1.2.1.4")

        if self.position_embeddings:
            # print('test ------ position_embeddings s', position_ids.shape)
            position_embeddings = self.position_embeddings(position_ids)
            # print('test ------ position_embeddings e')
            # print(position_embeddings.shape)
        if self.token_type_embeddings:
            # print('test ------ token_type_embeddings', token_type_ids.shape)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings += token_type_embeddings
        # notifier.critical("test ------ 1.2.1.5")

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class GTXAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.head_size = self.num_attention_heads * self.attention_head_size

        # visual_dim = 2048
        if ctx_dim is None:
            ctx_dim = config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.head_size)
        self.key = nn.Linear(ctx_dim, self.head_size)
        self.value = nn.Linear(ctx_dim, self.head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None, output_attentions=False):
        # notifier.warning(hidden_states.size())
        # notifier.warning(attention_mask.size())
        # notifier.warning(context.size())
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class GTXAttentionOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor, KnowMix_indices=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if KnowMix_indices is None:
            hidden_states = input_tensor + hidden_states
        else:
            if isinstance(KnowMix_indices, int):
                input_tensor[:, KnowMix_indices] = input_tensor[:, KnowMix_indices] + hidden_states.squeeze(1)
            else:
                input_tensor[KnowMix_indices, :] = input_tensor[KnowMix_indices, :] + hidden_states.squeeze(1)
            hidden_states = input_tensor
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class GTXCrossAttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = GTXAttention(config)
        self.output = GTXAttentionOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None, KnowMix_indices=None, output_attentions=False):
        if KnowMix_indices is None:
            output = self.att(input_tensor, ctx_tensor, ctx_att_mask, output_attentions=output_attentions)
        else:
            if isinstance(KnowMix_indices, int):
                output = self.att(input_tensor[:, KnowMix_indices].unsqueeze(1), ctx_tensor, ctx_att_mask,
                                  output_attentions=output_attentions)
            else:
                output = self.att(input_tensor[KnowMix_indices, :].unsqueeze(1), ctx_tensor[KnowMix_indices, :],
                                  ctx_att_mask[KnowMix_indices.unsqueeze(1), :].unsqueeze(1).unsqueeze(2),
                                  output_attentions=output_attentions)
        if output_attentions:
            attention_probs = output[1]
        attention_output = self.output(output[0], input_tensor, KnowMix_indices)
        outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)
        return outputs


class GTXSelfAttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = GTXAttention(config)
        self.output = GTXAttentionOutput(config)

    def forward(self, input_tensor, attention_mask, output_attentions=False):
        # Self attention attends to itself, thus keys and querys are the same (input_tensor).
        output = self.self(
            input_tensor,
            input_tensor,
            attention_mask,
            output_attentions=output_attentions,
        )
        if output_attentions:
            attention_probs = output[1]
        attention_output = self.output(output[0], input_tensor)
        outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)
        return outputs


class GTXIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class GTXOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class GTXLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = GTXSelfAttentionLayer(config)
        self.intermediate = GTXIntermediate(config)
        self.output = GTXOutput(config)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        outputs = self.attention(hidden_states, attention_mask, output_attentions=output_attentions)
        attention_output = outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs[1:]  # add attentions if we output them
        return outputs


class GTXXLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.cross_att_type = config.cross_att_type if 'cross_att_type' in vars(config).keys() else 'cross'

        # The cross-attention Layer
        self.cross_attention = GTXCrossAttentionLayer(config)

        # Self-attention Layers
        self.lang_self_att = GTXSelfAttentionLayer(config)
        self.visn_self_att = GTXSelfAttentionLayer(config)

        # Intermediate and Output Layers (FFNs)
        self.lang_inter = GTXIntermediate(config)
        self.lang_output = GTXOutput(config)
        self.visn_inter = GTXIntermediate(config)
        self.visn_output = GTXOutput(config)

    def cross_att(
            self,
            lang_input,
            lang_attention_mask,
            visual_input,
            visual_attention_mask,
            output_x_attentions=False,
    ):
        lang_att_output = self.cross_attention(
            lang_input,
            visual_input,
            ctx_att_mask=visual_attention_mask,
            output_attentions=output_x_attentions,
        )
        visual_att_output = self.cross_attention(
            visual_input,
            lang_input,
            ctx_att_mask=lang_attention_mask,
            output_attentions=output_x_attentions,
        )
        return lang_att_output, visual_att_output

    def no_cross_att(
            self,
            lang_input,
            lang_attention_mask,
            visual_input,
            visual_attention_mask,
            output_x_attentions=False,
    ):
        lang_att_output = self.cross_attention(
            lang_input,
            lang_input,
            ctx_att_mask=lang_attention_mask,
            output_attentions=output_x_attentions,
        )
        visual_att_output = self.cross_attention(
            visual_input,
            visual_input,
            ctx_att_mask=visual_attention_mask,
            output_attentions=output_x_attentions,
        )
        return lang_att_output, visual_att_output

    def unilm_cross_att(
            self,
            lang_input,
            lang_attention_mask,
            visual_input,
            visual_attention_mask,
            output_x_attentions=False,
    ):
        lang_att_output = self.cross_attention(
            lang_input,
            visual_input,
            ctx_att_mask=visual_attention_mask,  # in fact, visual_padding_masks
            output_attentions=output_x_attentions,
        )
        visual_att_output = (visual_input, None) if output_x_attentions else (visual_input,)
        return lang_att_output, visual_att_output

    def self_att(self, lang_input, lang_attention_mask, visual_input, visual_attention_mask):
        # Self Attention
        lang_att_output = self.lang_self_att(lang_input, lang_attention_mask, output_attentions=False)
        visual_att_output = self.visn_self_att(visual_input, visual_attention_mask, output_attentions=False)
        return lang_att_output[0], visual_att_output[0]

    def output_fc(self, lang_input, visual_input):
        # FC layers
        lang_inter_output = self.lang_inter(lang_input)
        visual_inter_output = self.visn_inter(visual_input)

        # Layer output
        lang_output = self.lang_output(lang_inter_output, lang_input)
        visual_output = self.visn_output(visual_inter_output, visual_input)

        return lang_output, visual_output

    def forward(
            self,
            lang_feats,
            lang_attention_mask,
            visual_feats,
            visual_attention_mask,
            visual_padding_mask,
            output_attentions=False,
    ):
        if self.cross_att_type == 'single':
            lang_att_output, visual_att_output = self.no_cross_att(
                lang_input=lang_feats,
                lang_attention_mask=lang_attention_mask,
                visual_input=visual_feats,
                visual_attention_mask=visual_padding_mask,
                output_x_attentions=output_attentions,
            )
        elif self.cross_att_type == 'unilm':
            lang_att_output, visual_att_output = self.unilm_cross_att(
                lang_input=lang_feats,
                lang_attention_mask=lang_attention_mask,
                visual_input=visual_feats,
                visual_attention_mask=visual_padding_mask,
                output_x_attentions=output_attentions,
            )
        else:  # original cross attention
            lang_att_output, visual_att_output = self.cross_att(
                lang_input=lang_feats,
                lang_attention_mask=lang_attention_mask,
                visual_input=visual_feats,
                visual_attention_mask=visual_padding_mask,
                output_x_attentions=output_attentions,
            )
        attention_probs = {'txt->kg': lang_att_output[-1],
                           'kg->txt': visual_att_output[-1]}

        lang_att_output, visual_att_output = self.self_att(
            lang_att_output[0],
            lang_attention_mask,
            visual_att_output[0],
            visual_attention_mask,
        )

        lang_output, visual_output = self.output_fc(lang_att_output, visual_att_output)
        return (
            (
                lang_output,
                visual_output,
                attention_probs,
            )
            if output_attentions
            else (lang_output, visual_output)
        )


class GTXKnowMixLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # The cross-attention Layer
        self.cross_attention = GTXCrossAttentionLayer(config)

        # Self-attention Layers
        self.self_attention = GTXSelfAttentionLayer(config)

        # Intermediate and Output Layers (FFNs)
        self.inter = GTXIntermediate(config)
        self.output = GTXOutput(config)

    def mixup(
            self,
            inputs,
            attention_masks,
            contexts,
            summaries,
            summary_attention_masks,
            KnowMix_indices,
            output_x_attentions=False,
    ):
        err_stack = 0
        if KnowMix_indices[0] is not None:
            att_output = self.cross_attention(
                inputs,
                contexts,
                ctx_att_mask=attention_masks,
                KnowMix_indices=KnowMix_indices[0],
                output_attentions=output_x_attentions,
            )
        else:
            err_stack += 1
            att_output = (inputs,)
        if KnowMix_indices[1] is not None:
            att_output = self.cross_attention(
                att_output[0],
                summaries,
                ctx_att_mask=summary_attention_masks,
                KnowMix_indices=KnowMix_indices[1],
                output_attentions=output_x_attentions,
            )
        else:
            err_stack += 1

        if err_stack == 2:
            raise ValueError("************** Something goes wrong! ****************")

        return att_output

    def self_att(self, inputs, attention_masks):
        # Self Attention
        att_output = self.self_attention(inputs, attention_masks, output_attentions=False)
        return att_output[0]

    def output_fc(self, inputs):
        # FC layers
        inter_output = self.inter(inputs)

        # Layer output
        output = self.output(inter_output, inputs)

        return output

    def forward(
            self,
            inputs,
            attention_masks,
            contexts,
            context_attention_masks,
            summaries,
            summary_attention_masks,
            KnowMix_indices,
            output_attentions=False,
    ):
        # Select fusion level
        if "abs" in self.config.KnowMix:
            actual_KnowMix_indices = KnowMix_indices.bool().any(-1)
        elif "adm" in self.config.KnowMix:
            actual_KnowMix_indices = 0
        else:
            actual_KnowMix_indices = None

        att_output = self.mixup(
            inputs=inputs,
            attention_masks=context_attention_masks,
            contexts=contexts,
            summaries=summaries,
            summary_attention_masks=summary_attention_masks,
            KnowMix_indices=(actual_KnowMix_indices, 0 if summaries is not None else None),
            output_x_attentions=output_attentions,
        )
        attention_probs = att_output[-1]

        att_output = self.self_att(
            att_output[0],
            attention_masks,
        )

        output = self.output_fc(att_output)
        return (
            (
                output,
                attention_probs,
            )
            if output_attentions
            else (output,)
        )


# class GTXKGFeatureEncoder(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#
#     def forward(self):
#         """
#         To-Do : Some GCNs will be integrated in future
#         """
#         return None

class GTXEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        if 'encoder_type' not in vars(config).keys():
            notifier.warning("You have not specific encoder type in config, so that you don't use any kinds of LSTM")
            self.config.encoder_type = {'lang': ''}
        self.encoder_type = self.config.encoder_type['lang'].lower()

        # Number of layers
        self.num_l_layers = config.l_layers
        self.num_x_layers = config.x_layers
        self.num_r_layers = config.r_layers

        # Layers
        # Using self.layer instead of self.l_layer to support loading BERT weights.
        self.layer = nn.ModuleList([GTXLayer(config) for _ in range(self.num_l_layers)])
        notifier.warning(
            f"This model has a {config.cross_att_type if 'cross_att_type' in vars(config).keys() else 'cross'} type of x_attention architecture.")
        self.x_layers = nn.ModuleList([GTXXLayer(config) for _ in range(self.num_x_layers)])
        if ("lit" in self.config.KnowMix) or ("abs" in self.config.KnowMix) or ("summary" in self.config.KnowMix) or (
                "adm" in self.config.KnowMix):
            notifier.critical(f"Use Knowledge Mixup Layer on {config.KnowMix} nodes")
            self.r_layers = nn.ModuleList([GTXKnowMixLayer(config) for _ in range(self.num_r_layers)])
        else:
            notifier.critical("Use Standard GAT Layer")
            self.r_layers = nn.ModuleList([GTXLayer(config) for _ in range(self.num_r_layers)])

            # Lang Encoder Architecture
        # LSTM for generation, BiLSTM for pretraining/other donwstream tasks
        if self.encoder_type in ['bilstm', 'lstm']:
            self.convert_lang_encoder_to_RNN()

    def re_init_to_pretrained_lang_model(self):
        if isinstance(self.layer, nn.LSTM):
            notifier.warning("You've already used RNN-Style Architecture so that cannot re-init with PLMs.")
        else:
            """ If we use lm to language part, then we re-init our encoder.layer """
            plm_usage = self.config.pretrained_lang_model
            from transformers import AutoModel, AutoConfig
            if plm_usage['use_weight']:
                notifier.warning("Warm start for language part")
                self.layer = AutoModel.from_pretrained(plm_usage['model_name']).encoder.layer
            else:
                notifier.warning("Cold start for language part")
                plm_config = AutoConfig.from_pretrained(plm_usage['model_name'])
                self.layer = AutoModel.from_config(plm_config).encoder.layer

    def convert_lang_encoder_to_RNN(self):
        if self.encoder_type == 'lstm':
            notifier.critical("Use LSTM Decoder instead of BERT")
            self.layer = nn.LSTM(input_size=self.config.hidden_size,
                                 hidden_size=self.config.hidden_size,
                                 num_layers=1,
                                 dropout=self.config.hidden_dropout_prob,
                                 batch_first=True,
                                 bidirectional=False)
        elif self.encoder_type == 'bilstm':
            notifier.critical("Use BiLSTM Encoder instead of BERT")
            self.layer = nn.LSTM(input_size=self.config.hidden_size,
                                 hidden_size=self.config.hidden_size,
                                 num_layers=1,
                                 dropout=self.config.hidden_dropout_prob,
                                 batch_first=True,
                                 bidirectional=True)
        else:
            raise NotImplementedError("not implemented yet, a such kind of architecture for language encoder:",
                                      self.encoder_type)

    def forward(
            self,
            lang_feats,
            lang_attention_mask,
            kg_feats,
            kg_attention_mask,
            kg_padding_mask,
            kg_ext_input_ids=None,
            kg_ext_attention_mask=None,
            kg_ext_sum_input_ids=None,
            kg_ext_sum_attention_mask=None,
            output_attentions=None,
    ):

        kg_hidden_states = ()
        language_hidden_states = ()
        kg_attentions = () if output_attentions or self.config.output_attentions else None
        language_attentions = () if output_attentions or self.config.output_attentions else None
        cross_encoder_attentions = {'txt->kg': (),
                                    'kg->txt': ()} if output_attentions or self.config.output_attentions else None

        # Run language layers
        ## use RNN Encoder
        if self.encoder_type in ['bilstm', 'lstm']:
            l_outputs = self.layer(lang_feats)
            lang_feats = l_outputs[0]
            if self.layer.bidirectional:
                bsz, seq_len = lang_feats.shape[0], lang_feats.shape[1]
                lang_feats = lang_feats.view(bsz, seq_len, 2, -1).sum(axis=2)
            language_hidden_states = language_hidden_states + (lang_feats,)
        ## use BERT Encoder
        else:
            for layer_module in self.layer:
                l_outputs = layer_module(lang_feats, lang_attention_mask, output_attentions=output_attentions)
                lang_feats = l_outputs[0]
                language_hidden_states = language_hidden_states + (lang_feats,)
                if language_attentions is not None:
                    language_attentions = language_attentions + (l_outputs[1],)

        # Run relational layers
        ## Process the KG attention mask
        if kg_ext_attention_mask is not None:
            if len(kg_ext_attention_mask.shape) == 2:
                extended_kg_ext_attention_mask = kg_ext_attention_mask.unsqueeze(1).unsqueeze(2)
            elif len(kg_ext_attention_mask.shape) == 3:
                extended_kg_ext_attention_mask = kg_ext_attention_mask.unsqueeze(1)
            extended_kg_ext_attention_mask = extended_kg_ext_attention_mask.to(dtype=lang_attention_mask.dtype)
            extended_kg_ext_attention_mask = (1.0 - extended_kg_ext_attention_mask) * -10000.0
        else:
            extended_kg_ext_attention_mask = None
        if kg_ext_sum_attention_mask is not None:
            if len(kg_ext_sum_attention_mask.shape) == 2:
                extended_kg_ext_sum_attention_mask = kg_ext_sum_attention_mask.unsqueeze(1).unsqueeze(2)
            elif len(kg_ext_sum_attention_mask.shape) == 3:
                extended_kg_ext_sum_attention_mask = kg_ext_sum_attention_mask.unsqueeze(1)
            extended_kg_ext_sum_attention_mask = extended_kg_ext_sum_attention_mask.to(dtype=lang_attention_mask.dtype)
            extended_kg_ext_sum_attention_mask = (1.0 - extended_kg_ext_sum_attention_mask) * -10000.0
        else:
            extended_kg_ext_sum_attention_mask = None

        for layer_module in self.r_layers:
            if ("abs" in self.config.KnowMix) or ("summary" in self.config.KnowMix) or ("adm" in self.config.KnowMix):
                kg_outputs = layer_module(
                    kg_feats,
                    kg_attention_mask,
                    contexts=kg_ext_input_ids,
                    context_attention_masks=extended_kg_ext_attention_mask,
                    summaries=kg_ext_sum_input_ids,
                    summary_attention_masks=extended_kg_ext_sum_attention_mask,
                    KnowMix_indices=kg_ext_attention_mask,
                    output_attentions=output_attentions
                )
            else:
                kg_outputs = layer_module(kg_feats, kg_attention_mask, output_attentions=output_attentions)
            kg_feats = kg_outputs[0]
            kg_hidden_states = kg_hidden_states + (kg_feats,)
            if kg_attentions is not None:
                kg_attentions = kg_attentions + (kg_outputs[1],)

        # Run cross-modality layers
        for layer_module in self.x_layers:
            x_outputs = layer_module(
                lang_feats,
                lang_attention_mask,
                kg_feats,
                kg_padding_mask,
                kg_padding_mask,
                output_attentions=output_attentions,
            )
            lang_feats, kg_feats = x_outputs[:2]
            kg_hidden_states = kg_hidden_states + (kg_feats,)
            language_hidden_states = language_hidden_states + (lang_feats,)
            if cross_encoder_attentions is not None:
                cross_encoder_attentions = {k: cross_encoder_attentions[k] + (x_outputs[2][k],) for k in
                                            cross_encoder_attentions}
        kg_encoder_outputs = (
            kg_hidden_states,
            kg_attentions if output_attentions else None,
        )
        lang_encoder_outputs = (
            language_hidden_states,
            language_attentions if output_attentions else None,
        )
        return (
            kg_encoder_outputs,
            lang_encoder_outputs,
            cross_encoder_attentions if output_attentions else None,
        )


class GTXPooler(nn.Module):
    def __init__(self, config):
        super(GTXPooler, self).__init__()
        self.multi_pooler = nn.Sequential(nn.Linear(config.hidden_size * 2, config.hidden_size * 2),
                                          nn.Tanh(),
                                          nn.Linear(config.hidden_size * 2, 2))
        self.ce_pooler = nn.Sequential(nn.Linear(config.hidden_size * 2, config.hidden_size * 2),
                                       nn.Tanh(),
                                       nn.Linear(config.hidden_size * 2, 2))
        self.use_ce_pooler = config.use_ce_pooler

    # def forward(self, hidden_states):
    def forward(self, kg_hidden_states, lang_hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensors = torch.cat([kg_hidden_states[:, 0], lang_hidden_states[:, 0]], dim=1)
        if self.use_ce_pooler:
            pooled_output = self.ce_pooler(first_token_tensors)
        else:
            pooled_output = self.multi_pooler(first_token_tensors)
        return pooled_output


class GTXPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(GTXPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act]
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class GTXLMPredictionHead(nn.Module):
    def __init__(self, config, GTX_model_embedding_weights):
        super(GTXLMPredictionHead, self).__init__()
        self.transform = GTXPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(
            GTX_model_embedding_weights.size(1),
            GTX_model_embedding_weights.size(0),
            bias=False,
        )
        self.decoder.weight = GTX_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(GTX_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class GTXPreTrainingHeads(nn.Module):
    def __init__(self, config, GTX_model_embedding_weights):
        super(GTXPreTrainingHeads, self).__init__()
        self.predictions = GTXLMPredictionHead(config, GTX_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)

        return prediction_scores


class GTXPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GTXConfig
    load_tf_weights = load_tf_weights_in_GTX
    base_model_prefix = "GTX"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


GTX_START_DOCSTRING = r"""

    The GTX model was proposed in `GTX: Learning Cross-Modality Encoder Representations from Transformers
    <https://arxiv.org/abs/1908.07490>`__ by Hao Tan and Mohit Bansal. It's a vision and language transformer model,
    pretrained on a variety of multi-modal datasets comprising of GQA, VQAv2.0, MCSCOCO captions, and Visual genome,
    using a combination of masked language modeling, region of interest feature regression, cross entropy loss for
    question answering attribute prediction, and object tag predicition.

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.GTXConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

GTX_INPUTS_DOCSTRING = r"""

    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.GTXTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        visual_feats: (:obj:`torch.FloatTensor` of shape :obj:՝(batch_size, num_visual_features, visual_feat_dim)՝):
            This input represents visual features. They ROI pooled object features from bounding boxes using a
            faster-RCNN model)

            These are currently not provided by the transformers library.
        visual_pos: (:obj:`torch.FloatTensor` of shape :obj:՝(batch_size, num_visual_features, visual_pos_dim)՝):
            This input represents spacial features corresponding to their relative (via index) visual features. The
            pre-trained GTX model expects these spacial features to be normalized bounding boxes on a scale of 0 to
            1.

            These are currently not provided by the transformers library.
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        visual_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`__
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`({0}, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


class text_encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_l_layers = config.l_layers
        # definition of l_layers: used for language encoding
        self.text_layers = nn.ModuleList([GTXLayer(config) for _ in range(self.num_l_layers)])

    def forward(self,
                lang_feats,
                lang_attention_mask,
                language_hidden_states,
                language_attentions,
                output_attentions,
                ):
        # Run language layers
        ## use RNN Encoder
        if self.config.encoder_type['lang'].lower() in ['bilstm', 'lstm']:
            l_outputs = self.text_layers(lang_feats)
            lang_feats = l_outputs[0]
            if self.layer.bidirectional:
                bsz, seq_len = lang_feats.shape[0], lang_feats.shape[1]
                lang_feats = lang_feats.view(bsz, seq_len, 2, -1).sum(axis=2)
            language_hidden_states = language_hidden_states + (lang_feats,)
        ## use BERT Encoder
        else:
            for layer_module in self.text_layers:
                l_outputs = layer_module(lang_feats, lang_attention_mask, output_attentions=output_attentions)
                lang_feats = l_outputs[0]
                language_hidden_states = language_hidden_states + (lang_feats,)
                if language_attentions is not None:
                    language_attentions = language_attentions + (l_outputs[1],)

        return lang_feats, language_hidden_states, language_attentions


# definition of graph_encoder
class graph_encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_r_layers = config.r_layers
        # definition of r_layers: used for graph encodding
        if ("lit" in self.config.KnowMix) or ("abs" in self.config.KnowMix) or ("summary" in self.config.KnowMix) or (
                "adm" in self.config.KnowMix):
            notifier.critical(f"Use Knowledge Mixup Layer on {config.KnowMix} nodes")
            self.graph_layers = nn.ModuleList([GTXKnowMixLayer(config) for _ in range(self.num_r_layers)])
        else:
            notifier.critical("Use Standard GAT Layer")
            self.graph_layers = nn.ModuleList([GTXLayer(config) for _ in range(self.num_r_layers)])

    def forward(self,
                kg_feats,
                kg_hidden_states,
                kg_attentions,
                kg_attention_mask,
                kg_ext_input_ids,
                extended_kg_ext_attention_mask,
                kg_ext_sum_input_ids,
                extended_kg_ext_sum_attention_mask,
                kg_ext_attention_mask,
                output_attentions
                ):
        for layer_module in self.graph_layers:
            if ("abs" in self.config.KnowMix) or ("summary" in self.config.KnowMix) or ("adm" in self.config.KnowMix):
                kg_outputs = layer_module(
                    kg_feats,
                    kg_attention_mask,
                    contexts=kg_ext_input_ids,
                    context_attention_masks=extended_kg_ext_attention_mask,
                    summaries=kg_ext_sum_input_ids,
                    summary_attention_masks=extended_kg_ext_sum_attention_mask,
                    KnowMix_indices=kg_ext_attention_mask,
                    output_attentions=output_attentions
                )
            else:
                kg_outputs = layer_module(kg_feats, kg_attention_mask, output_attentions=output_attentions)
            kg_feats = kg_outputs[0]
            kg_hidden_states = kg_hidden_states + (kg_feats,)
            if kg_attentions is not None:
                kg_attentions = kg_attentions + (kg_outputs[1],)

        return kg_feats, kg_hidden_states, kg_attentions


class cross_encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_x_layers = config.x_layers
        self.x_layers = nn.ModuleList([GTXXLayer(config) for _ in range(self.num_x_layers)])

    def forward(self,
                lang_feats,
                language_hidden_states,
                lang_attention_mask,
                kg_feats,
                kg_hidden_states,
                kg_padding_mask,
                cross_encoder_attentions,
                output_attentions,
                ):
        # Run cross-modality layers
        for layer_module in self.x_layers:
            x_outputs = layer_module(
                lang_feats,
                lang_attention_mask,
                kg_feats,
                kg_padding_mask,
                kg_padding_mask,
                output_attentions=output_attentions,
            )
            lang_feats, kg_feats = x_outputs[:2]
            kg_hidden_states = kg_hidden_states + (kg_feats,)
            language_hidden_states = language_hidden_states + (lang_feats,)
            if cross_encoder_attentions is not None:
                cross_encoder_attentions = {k: cross_encoder_attentions[k] + (x_outputs[2][k],) for k in
                                            cross_encoder_attentions}

            return CrossEncoderOutput(lang_feats, kg_feats, kg_hidden_states, language_hidden_states,
                                      cross_encoder_attentions)


@add_start_docstrings(
    "The bare GTX Model transformer outputting raw hidden-states without any specific head on top.",
    GTX_START_DOCSTRING,
)
class GTXModel(GTXPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.lang_embeddings = GTXEmbeddings(config, input_type='lang')
        self.kg_embeddings = GTXEmbeddings(config, input_type='kg')
        self.encoder_type = self.config.encoder_type['lang'].lower()
        self.config = config
        self.temp = nn.Parameter(torch.ones([]) * self.config.tmp)
        self.queue_size = self.config.queue_size
        self.momentum = self.config.momentum
        self.num_labels = config.num_labels
        self.num_kg_labels = config.num_kg_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # self.encoder = GTXEncoder(config)
        # self.pooler = GTXPooler(config)
        self.itm_head = GTXPooler(self.config)
        self.mlm_head = GTXPreTrainingHeads(self.config, self.lang_embeddings.word_embeddings.weight)
        self.classifier = nn.Linear(config.hidden_size, config.num_kg_labels)
        self.edge_classifier = nn.Sequential(nn.Linear(config.hidden_size * 2, config.hidden_size * 2),
                                             nn.Tanh(),
                                             nn.Linear(config.hidden_size * 2, config.num_relations))

        self.mlm_head_m = GTXPreTrainingHeads(self.config, self.lang_embeddings.word_embeddings.weight)
        self.classifier_m = nn.Linear(config.hidden_size, config.num_kg_labels)
        self.edge_classifier_m = nn.Sequential(nn.Linear(config.hidden_size * 2, config.hidden_size * 2),
                                             nn.Tanh(),
                                             nn.Linear(config.hidden_size * 2, config.num_relations))

        self.graph_encoder = graph_encoder(self.config)
        self.text_encoder = text_encoder(self.config)
        self.cross_encoder = cross_encoder(self.config)
        self.graph_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.text_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size)

        # create momentum models
        self.graph_encoder_m = graph_encoder(self.config)
        self.text_encoder_m = text_encoder(self.config)
        self.cross_encoder_m = cross_encoder(self.config)
        self.graph_proj_m = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.text_proj_m = nn.Linear(self.config.hidden_size, self.config.hidden_size)

        self.model_pairs = [[self.graph_encoder, self.graph_encoder_m],
                            [self.text_encoder, self.text_encoder_m],
                            [self.graph_proj, self.graph_proj_m],
                            [self.text_proj, self.text_proj_m],
                            [self.cross_encoder, self.cross_encoder_m],
                            [self.classifier, self.classifier_m],
                            [self.edge_classifier, self.edge_classifier_m],
                            [self.mlm_head, self.mlm_head_m]
                            ]

        # create the queue
        self.register_buffer("graph_queue", torch.randn(config.hidden_size, self.queue_size))
        self.register_buffer("text_queue", torch.randn(config.hidden_size, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.graph_queue = nn.functional.normalize(self.graph_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        notifier.warning(
            f"This model has a {config.cross_att_type if 'cross_att_type' in vars(config).keys() else 'cross'} type of x_attention architecture.")
        # LSTM for generation, BiLSTM for pretraining/other donwstream tasks
        self.encoder_type = self.config.encoder_type['lang'].lower()
        if self.encoder_type in ['bilstm', 'lstm']:
            self.convert_lang_encoder_to_RNN()

        if "init,enc" == config.KnowMix:
            notifier.warning("Prepare sentence encoder for KG init.")
            self.kg_init_enc = nn.GRU(config.hidden_size, config.hidden_size, num_layers=1, batch_first=True,
                                      bidirectional=False)

        # self.init_weights()

        # Loss functions
        self.loss_fcts = {
            "l2": SmoothL1Loss(reduction="none"),
            "mse": MSELoss(reduction="none"),
            "ce": CrossEntropyLoss(),
            "tri": nn.TripletMarginLoss()  # (margin=config.margin)
        }

    def re_init_to_pretrained_lang_model(self):
        if isinstance(self.text_encoder.text_layers, nn.LSTM):
            notifier.warning("You've already used RNN-Style Architecture so that cannot re-init with PLMs.")
        else:
            """ If we use lm to language part, then we re-init our encoder.layer """
            plm_usage = self.config.pretrained_lang_model
            from transformers import AutoModel, AutoConfig
            if plm_usage['use_weight']:
                notifier.warning("Warm start for language part")
                self.text_encoder.text_layers = AutoModel.from_pretrained(plm_usage['model_name']).encoder.layer
            else:
                notifier.warning("Cold start for language part")
                plm_config = AutoConfig.from_pretrained(plm_usage['model_name'])
                self.text_encoder.text_layers = AutoModel.from_config(plm_config).encoder.layer

    def convert_lang_encoder_to_RNN(self):
        if self.encoder_type == 'lstm':
            notifier.critical("Use LSTM Decoder instead of BERT")
            self.text_encoder.text_layers = nn.LSTM(input_size=self.config.hidden_size,
                                                    hidden_size=self.config.hidden_size,
                                                    num_layers=1,
                                                    dropout=self.config.hidden_dropout_prob,
                                                    batch_first=True,
                                                    bidirectional=False)
        elif self.encoder_type == 'bilstm':
            notifier.critical("Use BiLSTM Encoder instead of BERT")
            self.text_encoder.text_layers = nn.LSTM(input_size=self.config.hidden_size,
                                                    hidden_size=self.config.hidden_size,
                                                    num_layers=1,
                                                    dropout=self.config.hidden_dropout_prob,
                                                    batch_first=True,
                                                    bidirectional=True)
        else:
            raise NotImplementedError("not implemented yet, a such kind of architecture for language encoder:",
                                      self.encoder_type)

    def get_lang_embeddings(self):
        return self.lang_embeddings.word_embeddings

    def set_lang_embeddings(self, new_embeddings):
        self.lang_embeddings.word_embeddings = new_embeddings

    def get_kg_embeddings(self):
        return self.kg_embeddings.word_embeddings

    def set_kg_embeddings(self, new_embedding):  # , lit2word=None):
        # if lit2word is None:
        # if len(self.config.kg_special_token_ids)>0:
        shape = self.kg_embeddings.word_embeddings.weight.data[len(self.config.kg_special_token_ids):, :].shape
        self.kg_embeddings.word_embeddings.weight.data[len(self.config.kg_special_token_ids):, :] = new_embedding.data[
                                                                                                    :shape[0]]

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, graph_feat, text_feat):
        # gather keys before updating queue
        # graph_feats = concat_all_gather(graph_feat)
        # text_feats = concat_all_gather(text_feat)
        graph_feats = graph_feat
        text_feats = text_feat

        batch_size = graph_feats.shape[0]

        ptr = int(self.queue_ptr)

        # print('>>>')
        # print('self.queue_ptr[0]',self.queue_ptr[0],'self.queue_size',self.queue_size,'batch_size',batch_size,self.queue_size/batch_size,':',graph_feats.shape,text_feats.shape)

        # assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        if (ptr + batch_size) > self.queue_size:
            ptr_cur = batch_size - (self.queue_size - ptr)
            # self.graph_queue[:, :, ptr:] = graph_feats.permute(1, 2, 0)[:, :, 0: batch_size - ptr_cur]
            # self.text_queue[:, :, ptr:] = text_feats.permute(1, 2, 0)[:, :, 0:batch_size - ptr_cur]
            # self.graph_queue[:, :, 0: ptr_cur] = graph_feats.permute(1, 2, 0)[:, :, batch_size - ptr_cur: batch_size]
            # self.text_queue[:, :, 0: ptr_cur] = text_feats.permute(1, 2, 0)[:, :, batch_size - ptr_cur: batch_size]
            self.graph_queue[:, ptr:] = graph_feats.permute(1, 0)[:, 0: batch_size - ptr_cur]
            self.text_queue[:, ptr:] = text_feats.permute(1, 0)[:, 0:batch_size - ptr_cur]
            self.graph_queue[:, 0: ptr_cur] = graph_feats.permute(1, 0)[:, batch_size - ptr_cur: batch_size]
            self.text_queue[:, 0: ptr_cur] = text_feats.permute(1, 0)[:, batch_size - ptr_cur: batch_size]
            ptr = ptr_cur  # move pointer
        else:
            self.graph_queue[:, ptr:ptr + batch_size] = graph_feats.permute(1, 0)
            self.text_queue[:, ptr:ptr + batch_size] = text_feats.permute(1, 0)
            ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    def forward(
            self,
            lang_input_ids=None,  # orignal text inputs
            kg_input_ids=None,  # orignal kg inputs
            lang_inputs_embeds=None,  # text embeds
            kg_inputs_embeds=None,  # kg inputs
            lang_attention_mask=None,
            kg_attention_mask=None, # equivalent to the M_{adj}
            kg_padding_mask=None,
            kg_ext_input_ids=None,
            kg_ext_attention_mask=None,
            kg_ext_sum_input_ids=None,  # corresponding to the batch_encoding['knowledge']
            kg_ext_sum_attention_mask=None,  # language mask generated by self.tokenizer
            kg_langinit_input_ids=None,
            kg_langinit_attention_mask=None,
            token_type_ids=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            lm_label=None,
            kg_label=None,
            cross_label=None,
            rc_indeces=None,
            kg_label_mask=None,
    ):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)
        device = lang_input_ids.device
        alpha = self.config.alpha
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_attentions_neg = output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if lang_input_ids is not None and lang_inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif kg_input_ids is not None and kg_inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if lang_attention_mask is not None:
            if len(lang_attention_mask.shape) == 2:  # (batch_size, seq_length)
                extended_lang_attention_mask = lang_attention_mask.unsqueeze(1).unsqueeze(2)
            elif len(lang_attention_mask.shape) == 3:  # (batch_size, seq_length, seq_length)
                extended_lang_attention_mask = lang_attention_mask.unsqueeze(1)
            elif len(lang_attention_mask.shape) == 4:  # (batch_size, 1, seq_length, seq_length)
                extended_lang_attention_mask = lang_attention_mask
            else:
                raise ValueError(
                    "Only supports (batch_size, seq_length) or (batch_size, seq_length, seq_length) or even full extended")
        else:
            raise ValueError("there is no attention mask for langauge part")

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_lang_attention_mask = extended_lang_attention_mask.to(dtype=self.dtype)
        extended_lang_attention_mask = (1.0 - extended_lang_attention_mask) * -10000.0

        # Process the KG attention mask
        if kg_attention_mask is not None:
            if len(kg_attention_mask.shape) == 2:
                # Process KG-side self attention mask
                extended_kg_attention_mask = kg_attention_mask.unsqueeze(1).unsqueeze(2)
                extended_kg_attention_mask = extended_kg_attention_mask.to(dtype=self.dtype)
                extended_kg_attention_mask = (1.0 - extended_kg_attention_mask) * -10000.0
                # Process KG padding mask for cross attention
                extended_kg_padding_mask = kg_padding_mask.unsqueeze(1).unsqueeze(2)
                extended_kg_padding_mask = extended_kg_padding_mask.to(dtype=self.dtype)
                extended_kg_padding_mask = (1.0 - extended_kg_padding_mask) * -10000.0
            elif len(kg_attention_mask.shape) == 3:
                # Process KG-side self attention mask
                extended_kg_attention_mask = kg_attention_mask.unsqueeze(1)
                extended_kg_attention_mask = extended_kg_attention_mask.to(dtype=self.dtype)
                extended_kg_attention_mask = (1.0 - extended_kg_attention_mask) * -10000.0
                # Process KG padding mask for cross attention
                extended_kg_padding_mask = kg_padding_mask.unsqueeze(1).unsqueeze(2)
                extended_kg_padding_mask = extended_kg_padding_mask.to(dtype=self.dtype)
                extended_kg_padding_mask = (1.0 - extended_kg_padding_mask) * -10000.0
            elif len(kg_attention_mask.shape) == 4:
                # Process KG-side self attention mask
                extended_kg_attention_mask = kg_attention_mask
                extended_kg_attention_mask = extended_kg_attention_mask.to(dtype=self.dtype)
                extended_kg_attention_mask = (1.0 - extended_kg_attention_mask) * -10000.0
                # Process KG padding mask for cross attention
                extended_kg_padding_mask = kg_padding_mask.unsqueeze(1).unsqueeze(2)
                extended_kg_padding_mask = extended_kg_padding_mask.to(dtype=self.dtype)
                extended_kg_padding_mask = (1.0 - extended_kg_padding_mask) * -10000.0
            else:
                raise ValueError("Only supports seq_len X seq_len mask or batch_size X # head X seq_len X seq_len")
        else:
            # Process KG padding mask for cross attention
            extended_kg_padding_mask = kg_padding_mask.unsqueeze(1).unsqueeze(2)
            extended_kg_padding_mask = extended_kg_padding_mask.to(dtype=self.dtype)
            extended_kg_padding_mask = (1.0 - extended_kg_padding_mask) * -10000.0
            extended_kg_attention_mask = extended_kg_padding_mask.clone().detach()

        # Positional Word Embeddings
        lang_embedding_output = self.lang_embeddings(lang_input_ids, token_type_ids, lang_inputs_embeds)
        if "linearize" in self.config.KnowMix:
            kg_inputs_embeds = self.lang_embeddings.word_embeddings(kg_input_ids)
        else:
            kg_inputs_embeds = self.kg_embeddings.word_embeddings(kg_input_ids)
        if kg_langinit_input_ids is not None:
            initializable_idx = kg_langinit_attention_mask.any(-1)
            kg_init_ids = kg_langinit_input_ids[initializable_idx]
            if "mean" in self.config.KnowMix:
                kg_init_mask = kg_langinit_attention_mask[initializable_idx].unsqueeze(-1)
                kg_init_embedding_output = self.lang_embeddings.word_embeddings(kg_init_ids) * kg_init_mask
                kg_inputs_embeds[initializable_idx] = kg_init_embedding_output.sum(-2) / kg_init_mask.sum(-2)
            elif "enc" in self.config.KnowMix:
                kg_init_embedding_output = self.lang_embeddings.word_embeddings(kg_init_ids)
                kg_init_input_lengths = (kg_init_ids != 0).sum(-1)
                packed_init_embeds = pack_padded_sequence(kg_init_embedding_output, kg_init_input_lengths.cpu(),
                                                          batch_first=True, enforce_sorted=False)
                packed_init_output, _ = self.kg_init_enc(packed_init_embeds)
                kg_init_output = \
                    pad_packed_sequence(packed_init_output, batch_first=True, total_length=kg_init_ids.size(-1))[0][
                        torch.arange(kg_init_ids.size(0)), kg_init_input_lengths - 1]
                kg_inputs_embeds[initializable_idx] = kg_init_output.float()
            else:
                raise ValueError("Invalid initalization option!")
        kg_embedding_output = self.kg_embeddings(None, None, kg_inputs_embeds)
        if kg_ext_input_ids is not None:
            kg_ext_embedding_output = self.lang_embeddings.word_embeddings(kg_ext_input_ids)
        else:
            kg_ext_embedding_output = None
        if kg_ext_sum_input_ids is not None:
            kg_ext_sum_embedding_output = self.lang_embeddings.word_embeddings(kg_ext_sum_input_ids)
        else:
            kg_ext_sum_embedding_output = None

        # ================        rename varients
        lang_feats = lang_embedding_output
        lang_attention_mask = extended_lang_attention_mask
        kg_feats = kg_embedding_output
        kg_attention_mask = extended_kg_attention_mask
        kg_padding_mask = extended_kg_padding_mask
        kg_ext_input_ids = kg_ext_embedding_output
        kg_ext_attention_mask = kg_ext_attention_mask
        kg_ext_sum_input_ids = kg_ext_sum_embedding_output
        kg_ext_sum_attention_mask = kg_ext_sum_attention_mask
        output_attentions = output_attentions

        kg_hidden_states = ()
        language_hidden_states = ()
        kg_attentions = () if output_attentions or self.config.output_attentions else None
        language_attentions = () if output_attentions or self.config.output_attentions else None
        cross_encoder_attentions = {'txt->kg': (),
                                    'kg->txt': ()} if output_attentions or self.config.output_attentions else None
        cross_encoder_attentions_neg = {'txt->kg': (),
                                        'kg->txt': ()} if output_attentions or self.config.output_attentions else None

        ## Process the KG attention mask
        if kg_ext_attention_mask is not None:
            if len(kg_ext_attention_mask.shape) == 2:
                extended_kg_ext_attention_mask = kg_ext_attention_mask.unsqueeze(1).unsqueeze(2)
            elif len(kg_ext_attention_mask.shape) == 3:
                extended_kg_ext_attention_mask = kg_ext_attention_mask.unsqueeze(1)
            extended_kg_ext_attention_mask = extended_kg_ext_attention_mask.to(dtype=lang_attention_mask.dtype)
            extended_kg_ext_attention_mask = (1.0 - extended_kg_ext_attention_mask) * -10000.0
        else:
            extended_kg_ext_attention_mask = None
        if kg_ext_sum_attention_mask is not None:
            if len(kg_ext_sum_attention_mask.shape) == 2:
                extended_kg_ext_sum_attention_mask = kg_ext_sum_attention_mask.unsqueeze(1).unsqueeze(2)
            elif len(kg_ext_sum_attention_mask.shape) == 3:
                extended_kg_ext_sum_attention_mask = kg_ext_sum_attention_mask.unsqueeze(1)
            extended_kg_ext_sum_attention_mask = extended_kg_ext_sum_attention_mask.to(dtype=lang_attention_mask.dtype)
            extended_kg_ext_sum_attention_mask = (1.0 - extended_kg_ext_sum_attention_mask) * -10000.0
        else:
            extended_kg_ext_sum_attention_mask = None

        # ====================     back up             ==========================
        lang_feats_org = lang_embedding_output.clone()
        language_hidden_states_org = ()
        language_attentions_org = () if output_attentions or self.config.output_attentions else None
        kg_feats_org = kg_embedding_output.clone()
        kg_hidden_states_org = ()
        kg_attentions_org = () if output_attentions or self.config.output_attentions else None

        # ====================     text encoder and graph encoder             ==========================
        lang_feats, language_hidden_states, language_attentions = \
            self.text_encoder(lang_feats,
                              lang_attention_mask,
                              language_hidden_states,
                              language_attentions,
                              output_attentions, )

        lang_feats_proj = F.normalize(self.text_proj(lang_feats[:, 0, :]), dim=-1)

        kg_feats, kg_hidden_states, kg_attentions = \
            self.graph_encoder(kg_feats,
                               kg_hidden_states,
                               kg_attentions,
                               kg_attention_mask,
                               kg_ext_input_ids,
                               extended_kg_ext_attention_mask,
                               kg_ext_sum_input_ids,
                               extended_kg_ext_sum_attention_mask,
                               kg_ext_attention_mask,
                               output_attentions, )

        kg_feats_proj = F.normalize(self.graph_proj(kg_feats[:, 0, :]), dim=-1)

        # ====================     gta            ==========================
        # get momentum features
        with torch.no_grad():
            self._momentum_update()  # update model parameters by momentum mechanism
            lang_feats_m, language_hidden_states_m, language_attentions_m = \
                self.text_encoder_m(lang_feats_org,
                                    lang_attention_mask,
                                    language_hidden_states_org,
                                    language_attentions_org,
                                    output_attentions, )

            lang_feats_proj_m = F.normalize(self.text_proj_m(lang_feats_m[:, 0, :]), dim=-1)
            lang_feats_proj_m_all = torch.cat([lang_feats_proj_m.t(), self.text_queue.clone().detach()], dim=1)

            kg_feats_m, kg_hidden_states_m, kg_attentions_m = \
                self.graph_encoder_m(kg_feats_org,
                                     kg_hidden_states_org,
                                     kg_attentions_org,
                                     kg_attention_mask,
                                     kg_ext_input_ids,
                                     extended_kg_ext_attention_mask,
                                     kg_ext_sum_input_ids,
                                     extended_kg_ext_sum_attention_mask,
                                     kg_ext_attention_mask,
                                     output_attentions, )

            kg_feats_proj_m = F.normalize(self.graph_proj_m(kg_feats_m[:, 0, :]), dim=-1)
            kg_feats_proj_m_all = torch.cat([kg_feats_proj_m.t(), self.graph_queue.clone().detach()], dim=1)

            sim_g2t_m = kg_feats_proj_m @ lang_feats_proj_m_all / self.temp
            sim_t2g_m = lang_feats_proj_m @ kg_feats_proj_m_all / self.temp

            sim_targets = torch.zeros(sim_g2t_m.size()).cuda()
            sim_targets.fill_diagonal_(1)  # sim_targets 表示y
            # sim_i2t_m 代表momentum distillation中的 q^{i2t}
            sim_g2t_targets = alpha * F.softmax(sim_g2t_m, dim=1) + (1 - alpha) * sim_targets
            # sim_t2i_m 代表momentum distillation中的 q^{t2i}
            sim_t2g_targets = alpha * F.softmax(sim_t2g_m, dim=1) + (1 - alpha) * sim_targets

        sim_g2t = kg_feats_proj @ lang_feats_proj_m_all / self.temp
        sim_t2g = lang_feats_proj @ kg_feats_proj_m_all / self.temp

        loss_g2t = -torch.sum(F.log_softmax(sim_g2t, dim=1) * sim_g2t_targets, dim=1).mean()
        loss_t2g = -torch.sum(F.log_softmax(sim_t2g, dim=1) * sim_t2g_targets, dim=1).mean()

        self._dequeue_and_enqueue(kg_feats_proj_m, lang_feats_proj_m)

        if self.config.GTA:
            # compute the itc loss with momentum distillation (contrastive loss / align loss)
            loss_ita = (loss_g2t + loss_t2g) / 2
            # Maintenance of two queues
        else:
            loss_ita = None

        # ====================     cross encoder         ==========================
        cross_output_pos = self.cross_encoder(lang_feats,
                                              language_hidden_states,
                                              lang_attention_mask,
                                              kg_feats,
                                              kg_hidden_states,
                                              kg_padding_mask,
                                              cross_encoder_attentions,
                                              output_attentions, )
        if self.config.MoD:
            with torch.no_grad():
                cross_output_pos_m = self.cross_encoder_m(lang_feats_m,
                                                          language_hidden_states_m,
                                                          lang_attention_mask,
                                                          kg_feats_m,
                                                          kg_hidden_states_m,
                                                          kg_padding_mask,
                                                          cross_encoder_attentions,
                                                          output_attentions, )

        # ====================     gtm             ==========================
        # forward the positve graph-text pairs
        if self.config.GTM and self.config.hard:
            with torch.no_grad():
                bs = kg_feats.size(0)
                weights_g2t = F.softmax(sim_g2t[:, :bs], dim=1)
                weights_t2g = F.softmax(sim_t2g[:, :bs], dim=1)

                weights_g2t.fill_diagonal_(0)
                weights_t2g.fill_diagonal_(0)

            # select a negative graph for each graph
            kg_feats_neg = []
            kg_padding_mask_neg = []
            kg_hidden_states_neg_tmp = [[] for i in range(len(kg_hidden_states))]
            for b in range(bs):
                # torch.multinomial(input, num_samples,replacement=False, out=None) → LongTensor
                # 对input中的每一行做n_samples次取值，每一次取值时的输出张量是input当前行对应元素的下标
                neg_idx = torch.multinomial(weights_t2g[b], 1).item()
                kg_feats_neg.append(kg_feats[neg_idx])
                kg_padding_mask_neg.append(kg_padding_mask[neg_idx])
                for i in range(len(kg_hidden_states)):
                    kg_hidden_states_neg_tmp[i].append(kg_hidden_states[i][neg_idx])
            kg_feats_neg = torch.stack(kg_feats_neg, dim=0)
            kg_padding_mask_neg = torch.stack(kg_padding_mask_neg, dim=0)
            for i in range(len(kg_hidden_states)):
                kg_hidden_states_neg_tmp[i] = torch.stack(kg_hidden_states_neg_tmp[i], dim=0)
            kg_hidden_states_neg = tuple(kg_hidden_states_neg_tmp)

            # select a negative text for each text
            lang_feats_neg = []
            lang_attention_mask_neg = []
            language_hidden_states_neg_tmp = [[] for i in range(len(language_hidden_states))]
            for b in range(bs):
                neg_idx = torch.multinomial(weights_g2t[b], 1).item()
                lang_feats_neg.append(lang_feats[neg_idx])
                lang_attention_mask_neg.append(lang_attention_mask[neg_idx])
                for i in range(len(language_hidden_states)):
                    language_hidden_states_neg_tmp[i].append(language_hidden_states[i][neg_idx])
            lang_feats_neg = torch.stack(lang_feats_neg, dim=0)
            lang_attention_mask_neg = torch.stack(lang_attention_mask_neg, dim=0)
            for i in range(len(language_hidden_states)):
                language_hidden_states_neg_tmp[i] = torch.stack(language_hidden_states_neg_tmp[i], dim=0)
            language_hidden_states_neg = tuple(language_hidden_states_neg_tmp)

            lang_feats_all = torch.cat([lang_feats, lang_feats_neg], dim=0)
            lang_attention_mask_all = torch.cat([lang_attention_mask, lang_attention_mask_neg], dim=0)
            language_hidden_states_all = tuple(
                [torch.cat([language_hidden_states[i], language_hidden_states_neg[i]], dim=0) for i in
                 range(len(language_hidden_states))])
            kg_feats_all = torch.cat([kg_feats_neg, kg_feats], dim=0)
            kg_padding_mask_all = torch.cat([kg_padding_mask_neg, kg_padding_mask], dim=0)
            kg_hidden_states_all = tuple(
                [torch.cat([kg_hidden_states_neg[i], kg_hidden_states[i]], dim=0) for i in
                 range(len(kg_hidden_states))])

            cross_output_neg = self.cross_encoder(lang_feats_all,
                                                  language_hidden_states_all,
                                                  lang_attention_mask_all,
                                                  kg_feats_all,
                                                  kg_hidden_states_all,
                                                  kg_padding_mask_all,
                                                  cross_encoder_attentions_neg,
                                                  output_attentions_neg, )

            # lang_feats, kg_feats, kg_hidden_states, language_hidden_states, cross_encoder_attentions = \
            kg_hidden_embeddings = torch.cat([cross_output_pos.kg_feats, cross_output_neg.kg_feats], dim=0)
            lang_hidden_embeddings = torch.cat([cross_output_pos.lang_feats, cross_output_neg.lang_feats], dim=0)
            pooled_output_all = self.itm_head(kg_hidden_embeddings, lang_hidden_embeddings)

            gtm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                                   dim=0).cuda()
            loss_gtm = F.cross_entropy(pooled_output_all.float(), gtm_labels)
        elif self.config.GTM and not self.config.hard:
            # negative_sampling
            batch_size = kg_feats.size(0)
            lang_feats_gtm = torch.cat(
                [lang_feats.detach().clone()[(torch.arange(batch_size) + idx) % batch_size] for idx in range(2)], dim=0)
            language_hidden_states_gtm = ()
            for i in range(len(language_hidden_states)):
                language_hidden_states_gtm = language_hidden_states_gtm + (torch.cat(
                    [language_hidden_states[i].detach().clone()[(torch.arange(batch_size) + idx) % batch_size] for idx
                     in range(2)], dim=0),)
            lang_attention_mask_gtm = torch.cat(
                [lang_attention_mask.detach().clone()[(torch.arange(batch_size) + idx) % batch_size] for idx in
                 range(2)], dim=0)

            kg_feats_gtm = torch.cat([kg_feats.detach().clone() for _ in range(2)], dim=0)
            kg_hidden_states_gtm = ()
            for i in range(len(kg_hidden_states)):
                kg_hidden_states_gtm = kg_hidden_states_gtm + (
                    torch.cat([kg_hidden_states[i].detach().clone() for _ in range(2)], dim=0),)
            kg_padding_mask_gtm = torch.cat([kg_padding_mask.detach().clone() for _ in range(2)], dim=0)

            output_cross_gtm = self.cross_encoder(lang_feats_gtm,
                                                  language_hidden_states_gtm,
                                                  lang_attention_mask_gtm,
                                                  kg_feats_gtm,
                                                  kg_hidden_states_gtm,
                                                  kg_padding_mask_gtm,
                                                  cross_encoder_attentions,
                                                  output_attentions, )
            gtm_labels = torch.cat(
                [torch.ones(batch_size, dtype=torch.long), torch.zeros(batch_size, dtype=torch.long)], dim=0).cuda()

            pooled_output_all = self.itm_head(output_cross_gtm.kg_feats, output_cross_gtm.lang_feats)
            loss_gtm = F.cross_entropy(pooled_output_all.float(), gtm_labels)
        else:
            loss_gtm = None

        # ====================     MLM             ==========================
        output_mlm = copy.copy(cross_output_pos)
        prediction_scores = self.mlm_head(output_mlm.lang_feats)

        if self.config.MLM:
            if lm_label is not None:
                _lm_label = lm_label.view(-1)
                # _lm_label = lm_label
                positive_batch_size = _lm_label.size(0)
                masked_lm_loss = self.loss_fcts["ce"](
                    prediction_scores.view(-1, self.config.vocab_size['lang'])[:positive_batch_size],
                    _lm_label,
                )

                if self.config.MoD:
                    output_mlm_m = copy.copy(cross_output_pos_m)
                    with torch.no_grad():
                        prediction_scores_m = self.mlm_head_m(output_mlm_m.lang_feats)

                    soft_labels = F.softmax(
                        prediction_scores_m.view(-1, self.config.vocab_size['lang'])[:positive_batch_size],
                        dim=-1)

                    if soft_labels is not None:
                        loss_distill = -torch.sum(
                            F.log_softmax(
                                prediction_scores.view(-1, self.config.vocab_size['lang'])[:positive_batch_size],
                                dim=-1) * soft_labels, dim=-1)
                        loss_distill = loss_distill.view(-1)[_lm_label != -100].mean()
                        masked_lm_loss = (1 - alpha) * masked_lm_loss + alpha * loss_distill

            loss_mlm = masked_lm_loss
        else:
            loss_mlm = None

        # ====================     MLP  (Masked Literal Prediction)           ==========================
        if kg_label is not None and self.config.MLP:
            output_mlp = copy.copy(cross_output_pos)
            kg_prediction_scores = self.classifier(self.dropout(output_mlp.kg_feats))

            active_logits = kg_prediction_scores.view(-1, self.num_kg_labels)
            active_labels = torch.where(
                kg_label_mask.view(-1), kg_label.view(-1),
                torch.tensor(self.loss_fcts['ce'].ignore_index).type_as(kg_label)
            )
            positive_batch_size = active_labels.size(0)
            loss_mlp = self.loss_fcts['ce'](active_logits[:positive_batch_size], active_labels)

            if self.config.MoD:
                output_mlp_m = copy.copy(cross_output_pos_m)
                with torch.no_grad():
                    kg_prediction_scores_m = self.classifier_m(self.dropout(output_mlp_m.kg_feats))

                soft_labels_mlp = F.softmax(kg_prediction_scores_m.view(-1, self.num_kg_labels)[:positive_batch_size],
                                            dim=-1)

                loss_distill_mlp = -torch.sum(
                    F.log_softmax(kg_prediction_scores.view(-1, self.num_kg_labels)[:positive_batch_size],
                                  dim=-1) * soft_labels_mlp, dim=-1)
                loss_distill = loss_distill_mlp.view(-1)[active_labels != -100].mean()
                loss_mlp = (1 - alpha) * loss_mlp + alpha * loss_distill
        else:
            loss_mlp = None
            kg_prediction_scores = self.classifier(self.dropout(cross_output_pos.kg_feats))

        # ====================     # Relation Classification (RC)           ==========================
        if rc_indeces is not None and self.config.RC:
            output_rc = copy.copy(cross_output_pos)
            rc_labels = list()
            rc_inputs = list()

            for idx, sample_rc_indeces in enumerate(rc_indeces):
                for rc_idx in sample_rc_indeces:
                    rc_inputs.append(torch.cat([output_rc.kg_feats[idx, rc_idx[0]],
                                                output_rc.kg_feats[idx, rc_idx[1]]], dim=-1))
                    rc_labels.append(rc_idx[2])

            rc_outputs = self.edge_classifier(torch.stack(rc_inputs, dim=0))
            loss_rc = self.loss_fcts['ce'](rc_outputs, torch.tensor(rc_labels, dtype=torch.long, device=device))

            if self.config.MoD:
                with torch.no_grad():
                    output_rc_m = copy.copy(cross_output_pos_m)

                rc_soft_inputs = list()
                for idx, sample_rc_indeces in enumerate(rc_indeces):
                    for rc_idx in sample_rc_indeces:
                        rc_soft_inputs.append(torch.cat([output_rc_m.kg_feats[idx, rc_idx[0]],
                                                         output_rc_m.kg_feats[idx, rc_idx[1]]], dim=-1))

                rc_soft_outputs = self.edge_classifier_m(torch.stack(rc_soft_inputs, dim=0))
                rc_soft_labels = F.softmax(rc_soft_outputs, dim=-1)

                loss_distill_rc = -torch.sum(F.log_softmax(rc_outputs, dim=-1) * rc_soft_labels, dim=-1)
                loss_distill = loss_distill_rc.view(-1).mean()
                loss_rc = (1 - alpha) * loss_rc + alpha * loss_distill
        else:
            loss_rc = None

        # ==============    prepare outputs  ===================
        kg_encoder_outputs = (
            cross_output_pos.kg_hidden_states,
            kg_attentions if output_attentions else None,
        )
        lang_encoder_outputs = (
            cross_output_pos.language_hidden_states,
            language_attentions if output_attentions else None,
        )

        # ========  after the encoder before rewrite
        encoder_outputs = (
            kg_encoder_outputs,
            lang_encoder_outputs,
            cross_output_pos.cross_encoder_attentions if output_attentions else None,
        )

        kg_encoder_outputs, lang_encoder_outputs = encoder_outputs[:2]
        kg_hidden_states = kg_encoder_outputs[0]
        language_hidden_states = lang_encoder_outputs[0]

        all_attentions = ()
        if output_attentions:
            language_attentions = lang_encoder_outputs[1]
            kg_attentions = kg_encoder_outputs[1]
            cross_encoder_attentions = encoder_outputs[2]
            all_attentions = (
                language_attentions,
                kg_attentions,
                cross_encoder_attentions,
            )

        hidden_states = (language_hidden_states, kg_hidden_states) if output_hidden_states else ()
        kg_output = kg_hidden_states[-1]
        lang_output = language_hidden_states[-1]
        pooled_output = self.itm_head(kg_output, lang_output)

        if not return_dict:
            return (lang_output, kg_output, pooled_output) + hidden_states + all_attentions

        return GTXModelOutput(
            loss_mlm=loss_mlm,
            loss_ita=loss_ita,
            loss_gtm=loss_gtm,
            loss_rc=loss_rc,
            loss_mlp=loss_mlp,
            language_prediction_scores=prediction_scores,
            kg_prediction_scores=kg_prediction_scores,
            pooled_output=pooled_output,
            language_output=lang_output,
            kg_output=kg_output,
            language_hidden_states=language_hidden_states if output_hidden_states else None,
            kg_hidden_states=kg_hidden_states if output_hidden_states else None,
            language_attentions=language_attentions if output_attentions else None,
            kg_attentions=kg_attentions if output_attentions else None,
            cross_encoder_attentions=cross_encoder_attentions if output_attentions else None,
        )


class GAT(GTXPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Layers
        self.layer = nn.ModuleList([GTXLayer(config) for _ in range(6)])

        # Weight initialization
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            inputs_embeds=None,
            attention_mask=None,
    ):

        if len(attention_mask.shape) == 2:
            # Process KG-side self attention mask
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=self.dtype)
            attention_mask = (1.0 - attention_mask) * -10000.0

        elif len(attention_mask.shape) == 3:
            # Process KG-side self attention mask
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.to(dtype=self.dtype)
            attention_mask = (1.0 - attention_mask) * -10000.0

        elif len(attention_mask.shape) == 4:
            # Process KG-side self attention mask
            attention_mask = attention_mask
            attention_mask = attention_mask.to(dtype=self.dtype)
            attention_mask = (1.0 - attention_mask) * -10000.0

        for layer_module in self.layer:
            outputs = layer_module(inputs_embeds, attention_mask, output_attentions=False)
            feats = outputs[0]

        return (
            feats,
            None,
            None,
        )


class MedModelForPre(GTXPreTrainedModel):
    def __init__(self, config):  # , lit2word=None):
        super().__init__(config)  # , lit2word)
        # Configuration
        self.config = config
        # self.num_labels = config.num_labels
        # self.num_kg_labels = config.num_kg_labels

        # GTX backbone
        self.GTX = GTXModel(config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size, config.num_kg_labels)
        # self.edge_classifier = nn.Sequential(nn.Linear(config.hidden_size * 2, config.hidden_size * 2),
        #                                      nn.Tanh(),
        #                                      nn.Linear(config.hidden_size * 2, config.num_relations))

        # Pre-training heads
        # self.lm_head = GTXPreTrainingHeads(config, self.GTX.lang_embeddings.word_embeddings.weight)

        # Weight initialization
        self.init_weights()

        # Warm start KG embedding
        if not config.gcn and config.pretrained_kg_embedding:
            notifier.critical("Load pretrained embedding for translation based KG-GTX")
            new_embedding = torch.load(config.pretrained_kg_embedding)
            self.GTX.set_kg_embeddings(new_embedding)
            del new_embedding
        # elif lit2word:
        #     self.GTX.set_kg_embeddings(None, lit2word)

        # Use Pretrained-LM in Language Part
        # If we use random Language , mask this method
        self.GTX.re_init_to_pretrained_lang_model()

        self.GTX.copy_params()

        # Loss functions
        self.loss_fcts = {
            "l2": SmoothL1Loss(reduction="none"),
            "mse": MSELoss(reduction="none"),
            "ce": CrossEntropyLoss(),
            "tri": nn.TripletMarginLoss()  # (margin=config.margin)
        }

    def forward(
            self,
            lang_input_ids=None,
            kg_input_ids=None,
            lang_inputs_embeds=None,
            kg_inputs_embeds=None,
            lang_attention_mask=None,
            kg_attention_mask=None,
            kg_padding_mask=None,
            kg_label_mask=None,
            lm_label=None,
            kg_label=None,
            cross_label=None,
            token_type_ids=None,
            rc_indeces=None,
            kg_ext_input_ids=None,
            kg_ext_attention_mask=None,
            kg_ext_sum_input_ids=None,
            kg_ext_sum_attention_mask=None,
            kg_langinit_input_ids=None,
            kg_langinit_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True,
    ):
        r"""
        masked_lm_labels (``torch.LongTensor`` of shape ``(batch_size, sequence_length)``, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        obj_labels: (``Dict[Str: Tuple[Torch.FloatTensor, Torch.FloatTensor]]``, `optional`):
            each key is named after each one of the visual losses and each element of the tuple is of the shape
            ``(batch_size, num_features)`` and ``(batch_size, num_features, visual_feature_dim)`` for each the label id
            and the label score respectively
        matched_label (``torch.printLongTensor`` of shape ``(batch_size,)``, `optional`):
            Labels for computing the whether or not the text input matches the image (classification) loss. Input
            should be a sequence pair (see :obj:`input_ids` docstring) Indices should be in ``[0, 1]``:

            - 0 indicates that the sentence does not match the image,
            - 1 indicates that the sentence does match the image.
        ans: (``Torch.Tensor`` of shape ``(batch_size)``, `optional`):
            a one hot representation hof the correct answer `optional`

        Returns:
        """

        device = lang_input_ids.device  # if lang_input_ids is not None else inputs_embeds.device
        GTX_output = self.GTX(
            lang_input_ids=lang_input_ids,
            kg_input_ids=kg_input_ids,
            lang_inputs_embeds=lang_inputs_embeds,
            kg_inputs_embeds=kg_inputs_embeds,
            lang_attention_mask=lang_attention_mask,
            kg_attention_mask=kg_attention_mask,
            kg_padding_mask=kg_padding_mask,
            token_type_ids=token_type_ids,
            kg_ext_input_ids=kg_ext_input_ids,
            kg_ext_attention_mask=kg_ext_attention_mask,
            kg_ext_sum_input_ids=kg_ext_sum_input_ids,
            kg_ext_sum_attention_mask=kg_ext_sum_attention_mask,
            kg_langinit_input_ids=kg_langinit_input_ids,
            kg_langinit_attention_mask=kg_langinit_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            lm_label=lm_label,
            kg_label=kg_label,
            cross_label=cross_label,
            rc_indeces=rc_indeces,
            kg_label_mask=kg_label_mask,
        )

        lang_output, kg_output, lang_prediction_scores, kg_prediction_scores, cross_relationship_score = (
            GTX_output.language_output,
            GTX_output.kg_output,
            GTX_output.language_prediction_scores,
            GTX_output.kg_prediction_scores,
            GTX_output.pooled_output,
        )

        total_loss = (
            None
            if (lm_label is None and kg_label is None)
            else torch.tensor(0.0, device=device)
        )
        loss_dict = dict()

        # Masked Literal Prediction
        if self.config.MLP:
            loss_dict['loss_mlp'] = GTX_output.loss_mlp.mean().detach()
            total_loss += GTX_output.loss_mlp

        # Relation Classification (RC)
        if rc_indeces is not None and self.config.RC:
            loss_dict['loss_rc'] = GTX_output.loss_rc.mean().detach()
            total_loss += GTX_output.loss_rc

        # Image-Text Contrastive Loss /Graph Text Align
        if self.config.GTA:
            loss_dict['loss_ita'] = GTX_output.loss_ita.mean().detach()
            total_loss += GTX_output.loss_ita

        # Cross-modal Alignment Prediction (AP) / Graph-Text matching
        if self.config.GTM:
            loss_dict['loss_gtm'] = GTX_output.loss_gtm.mean().detach()
            total_loss += GTX_output.loss_gtm

        # Masked Language Modeling (MLM)
        if self.config.MLM:
            loss_dict['loss_mlm'] = GTX_output.loss_mlm.mean().detach()
            total_loss += GTX_output.loss_mlm

        loss_dict['loss'] = total_loss.mean().detach()

        if kg_prediction_scores is not None:
            kg_prediction_logits = kg_prediction_scores.detach()
        else:
            kg_prediction_logits = kg_prediction_scores

        if not return_dict:
            output = (
                         loss_dict,
                         lang_prediction_scores,
                         kg_prediction_scores,
                         cross_relationship_score,

                     ) + GTX_output[10:]
            return ((total_loss,) + output) if total_loss is not None else output

        return GTXForPreTrainingOutput(
            loss=total_loss,
            loss_dict=loss_dict,
            lang_prediction_logits=lang_prediction_scores.detach(),
            kg_prediction_logits=kg_prediction_logits,
            cross_relationship_score=cross_relationship_score.detach(),
            language_hidden_states=GTX_output.language_hidden_states,
            kg_hidden_states=GTX_output.kg_hidden_states,
            language_attentions=GTX_output.language_attentions,
            kg_attentions=GTX_output.kg_attentions,
            cross_encoder_attentions=GTX_output.cross_encoder_attentions,
        )


class UnimodalEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size["kg"], config.hidden_size)
        self.position_embeddings = nn.Embedding(1024, config.hidden_size)
        self.token_type_embeddings = None

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        else:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        seq_length = input_shape[1]

        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)

        if token_type_ids is None and self.token_type_embeddings is not None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=position_ids.device)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        embeddings = inputs_embeds
        if self.position_embeddings:
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        if self.token_type_embeddings:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings += token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class GAT(GTXPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Layers
        self.layer = nn.ModuleList([GTXLayer(config) for _ in range(6)])

        # Weight initialization
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            inputs_embeds=None,
            attention_mask=None,
    ):

        if len(attention_mask.shape) == 2:
            # Process KG-side self attention mask
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=self.dtype)
            attention_mask = (1.0 - attention_mask) * -10000.0

        elif len(attention_mask.shape) == 3:
            # Process KG-side self attention mask
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.to(dtype=self.dtype)
            attention_mask = (1.0 - attention_mask) * -10000.0

        elif len(attention_mask.shape) == 4:
            # Process KG-side self attention mask
            attention_mask = attention_mask
            attention_mask = attention_mask.to(dtype=self.dtype)
            attention_mask = (1.0 - attention_mask) * -10000.0

        for layer_module in self.layer:
            outputs = layer_module(inputs_embeds, attention_mask, output_attentions=False)
            feats = outputs[0]

        return (
            feats,
            None,
            None,
        )


class GTXForUnimodalPrediction(GTXPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # Configuration
        self.config = config
        self.task = config.task

        # Embeddings
        self.kg_embeddings = UnimodalEmbeddings(config)

        # Classifiers
        if self.task == "AdmPred":
            self.cls = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                     nn.Tanh(),
                                     nn.Linear(config.hidden_size, config.num_labels))
        elif self.task in ['ReAdm', 'Death30']:
            self.cls = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                     nn.Tanh(),
                                     nn.Linear(config.hidden_size, 1))
        else:
            self.cls = nn.Linear(config.hidden_size, 1)

        # Loss functions
        self.loss_fcts = {
            "bce": nn.BCEWithLogitsLoss(reduction='none'),
            "ce": nn.CrossEntropyLoss(),
        }
        self.class_weight = None

        # Weight initialization
        self.init_weights()

    def forward(
            self,
            lang_input_ids=None,
            kg_input_ids=None,
            lang_inputs_embeds=None,
            kg_inputs_embeds=None,
            lang_attention_mask=None,
            kg_attention_mask=None,
            kg_padding_mask=None,
            kg_label_mask=None,
            lm_label=None,
            label=None,
            kg_label=None,
            token_type_ids=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True,
    ):
        r"""
        masked_lm_labels (``torch.LongTensor`` of shape ``(batch_size, sequence_length)``, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        obj_labels: (``Dict[Str: Tuple[Torch.FloatTensor, Torch.FloatTensor]]``, `optional`):
            each key is named after each one of the visual losses and each element of the tuple is of the shape
            ``(batch_size, num_features)`` and ``(batch_size, num_features, visual_feature_dim)`` for each the label id
            and the label score respectively
        matched_label (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`):
            Labels for computing the whether or not the text input matches the image (classification) loss. Input
            should be a sequence pair (see :obj:`input_ids` docstring) Indices should be in ``[0, 1]``:

            - 0 indicates that the sentence does not match the image,
            - 1 indicates that the sentence does match the image.
        ans: (``Torch.Tensor`` of shape ``(batch_size)``, `optional`):
            a one hot representation hof the correct answer `optional`

        Returns:
        """
        loss_dict = dict()

        if self.config.unimodal == "graph":
            attention_mask = kg_attention_mask
        elif self.config.unimodal == "text":
            attention_mask = lang_attention_mask

        kg_inputs_embeds = self.kg_embeddings(kg_input_ids)
        encoder_outputs = self.encoder(
            input_ids=None if self.config.unimodal == "graph" else lang_input_ids,
            inputs_embeds=kg_inputs_embeds if self.config.unimodal == "graph" else None,
            attention_mask=attention_mask,
        )

        if self.task == 'AdmPred':
            classifier_input = encoder_outputs[0][:, 0]
            score = self.cls(classifier_input)
            total_loss = self.loss_fcts["bce"](score, label)
            if self.class_weight is not None:
                total_loss = total_loss * torch.tensor(self.class_weight, requires_grad=False, device=label.device)
            total_loss = total_loss.mean()
            loss_dict['loss'] = total_loss.detach()
        elif self.task == 'ErrDetect':
            classifier_input = encoder_outputs[0]
            _size = classifier_input.shape[:-1]
            score = self.cls(classifier_input.view(-1, self.config.hidden_size)).view(_size)
            total_loss = self.loss_fcts["bce"](score, kg_label).mean()
            loss_dict['loss'] = total_loss.detach()
        elif self.task in ['ReAdm', 'Death30']:
            classifier_input = encoder_outputs[0][:, 0]
            score = self.cls(classifier_input)
            total_loss = self.loss_fcts["bce"](score.squeeze(dim=1), label.float())
            total_loss = total_loss.mean()
            loss_dict['loss'] = total_loss.detach()
        else:
            raise ValueError("Task not supports")

        return GTXForDownstreamOutput(
            loss=total_loss,
            loss_dict=loss_dict,
            pooled_logits=score.detach(),
            # language_hidden_states=GTX_output.language_hidden_states,
            # kg_hidden_states=GTX_output.kg_hidden_states,
            # language_attentions=GTX_output.language_attentions,
            # kg_attentions=GTX_output.kg_attentions,
            # cross_encoder_attentions=GTX_output.cross_encoder_attentions,
        )

    @torch.no_grad()
    def get_out_put(
            self,
            lang_input_ids=None,
            kg_input_ids=None,
            lang_inputs_embeds=None,
            kg_inputs_embeds=None,
            lang_attention_mask=None,
            kg_attention_mask=None,
            kg_padding_mask=None,
            kg_label_mask=None,
            lm_label=None,
            label=None,
            kg_label=None,
            token_type_ids=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True,
    ):
        loss_dict = dict()

        if self.config.unimodal == "graph":
            attention_mask = kg_attention_mask
        elif self.config.unimodal == "text":
            attention_mask = lang_attention_mask

        kg_inputs_embeds = self.kg_embeddings(kg_input_ids)
        encoder_outputs = self.encoder(
            input_ids=None if self.config.unimodal == "graph" else lang_input_ids,
            inputs_embeds=kg_inputs_embeds if self.config.unimodal == "graph" else None,
            attention_mask=attention_mask,
        )

        if self.task == 'AdmPred':
            classifier_input = encoder_outputs[0][:, 0]
            score = self.cls(classifier_input)
            total_loss = self.loss_fcts["bce"](score, label)
            if self.class_weight is not None:
                total_loss = total_loss * torch.tensor(self.class_weight, requires_grad=False, device=label.device)
            total_loss = total_loss.mean()
            loss_dict['loss'] = total_loss.detach()
        elif self.task == 'ErrDetect':
            classifier_input = encoder_outputs[0]
            _size = classifier_input.shape[:-1]
            score = self.cls(classifier_input.view(-1, self.config.hidden_size)).view(_size)
            total_loss = self.loss_fcts["bce"](score, kg_label).mean()
            loss_dict['loss'] = total_loss.detach()
        elif self.task in ['ReAdm', 'Death30']:
            classifier_input = encoder_outputs[0][:, 0]
            score = self.cls(classifier_input)
            total_loss = self.loss_fcts["bce"](score.squeeze(dim=1), label.float())
            total_loss = total_loss.mean()
            loss_dict['loss'] = total_loss.detach()
        else:
            raise ValueError("Task not supports")

        return GTXForDownstreamOutput(
            loss=total_loss,
            loss_dict=loss_dict,
            pooled_logits=score.detach(),
        )
