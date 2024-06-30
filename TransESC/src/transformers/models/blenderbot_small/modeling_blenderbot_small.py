# coding=utf-8
# Copyright 2021 The Facebook, Inc. and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch BlenderbotSmall model. """

from curses import noecho
import math
import random
from typing import Optional, Tuple
from torch.nn.utils.rnn import pad_sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...file_utils import (
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_blenderbot_small import BlenderbotSmallConfig

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "BlenderbotSmallConfig"
_TOKENIZER_FOR_DOC = "BlenderbotSmallTokenizer"

BLENDERBOT_SMALL_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/blenderbot_small-90M",
    # See all BlenderbotSmall models at https://huggingface.co/models?filter=blenderbot_small
]


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903 
    图注意力层
    input: (B,N,C_in)
    output: (B,N,C_out)
    """

    def __init__(self, in_features, out_features, dropout, alpha=0.01, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features  # 节点表示向量的输入特征数
        self.out_features = out_features  # 节点表示向量的输出特征数
        self.dropout = dropout  # dropout参数
        self.alpha = alpha  # leakyrelu激活的参数
        self.concat = concat  # 如果为true, 再进行elu激活

        # 定义可训练参数，即论文中的W和a
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # 初始化
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  # 初始化

        # 定义leakyrelu激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj):
        """
        inp: input_fea [B,N, in_features]  in_features表示节点的输入特征向量元素个数
        adj: 图的邻接矩阵  [N, N] 非零即一 数据结构基本知识
        """
        h = torch.matmul(inp, self.W)  # [B, N, out_features]
        N = h.size()[1]  # N 图的节点数

        a_input = torch.cat([h.repeat(1, 1, N).view(-1, N * N, self.out_features), h.repeat(1, N, 1)], dim=-1).view(-1,
                                                                                                                    N,
                                                                                                                    N,
                                                                                                                    2 * self.out_features)
        # [B, N, N, 2*out_features]

        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        # [B, N, N, 1] => [B, N, N] 图注意力的相关系数（未归一化）

        # zero_vec = -1e12 * torch.ones_like(e_intra)    # 将没有连接的边置为负无穷

        # attention = torch.where(adj>0, e, zero_vec)   # [B, N, N]
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，
        # 否则需要mask并置为非常小的值，原因是softmax的时候这个最小值会不考虑。
        attention = F.softmax(e.masked_fill(adj == 0, -1e9), dim=-1)  # softmax形状保持不变 [B, N, N]，得到归一化的注意力权重！
        attention = F.dropout(attention, self.dropout, training=self.training)  # dropout，防止过拟合
        h_prime = torch.matmul(attention, h)  # [B, N, N].[B, N, out_features] => [B, N, out_features]
        # 得到由周围节点通过注意力权重进行更新的表示
        if self.concat:
            return F.relu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nheads=4, dropout=0.1, alpha=0.01):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        # nfeat为feature大小也就是idx_features_labels[:, 1:-1]第二列到倒数第二列，nhid自己设定
        # for _ in range(nheads)多头注意力机制，这里面nheads为8
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        # 创建8个多头注意力机制模块

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha,
                                           concat=False)  # 第二层(最后一层)的attention layer
        # nhid * nheads输入维度8*8，输出维度当时分的类，这个数据集里面是7个

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)  # 将每层attention拼接
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))  # 第二层的attention layer
        return x


class Predictor(nn.Module):
    def __init__(self, input_dim, mlp_dim, n_classes, mlp_dropout=0.1):
        super(Predictor, self).__init__()
        self.input_dim = input_dim
        self.mlp_dim = mlp_dim
        self.mlp_dropout = mlp_dropout
        self.mlp = nn.Sequential(nn.Linear(input_dim, mlp_dim, False), nn.ReLU(), nn.Dropout(mlp_dropout),
                                 nn.Linear(mlp_dim, mlp_dim, False), nn.ReLU(), nn.Dropout(mlp_dropout))
        self.predictor_weight = nn.Linear(mlp_dim, n_classes, False)

    def forward(self, x):
        predict_score = self.predictor_weight(self.mlp(x)).squeeze(-1)
        # predict_score = F.log_softmax(predict_score, dim=-1)
        return predict_score


class MaskedSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_attention_heads):
        super(MaskedSelfAttention, self).__init__()
        if hidden_dim % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_dim, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_dim / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(hidden_dim, self.all_head_size)
        self.key = nn.Linear(hidden_dim, self.all_head_size)
        self.value = nn.Linear(hidden_dim, self.all_head_size)

        self.relation_embedding = nn.Embedding(8, self.all_head_size, padding_idx=0)

        self.dropout = nn.Dropout(0.1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, q_hidden_states, k_hidden_states, v_hidden_states, trans_mask=None, edge_type=None):
        mixed_query_layer = self.query(q_hidden_states)
        mixed_key_layer = self.key(k_hidden_states)
        mixed_value_layer = self.value(v_hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        if edge_type is not None:
            rel_emb = self.relation_embedding(edge_type)
            attention_scores = torch.zeros((query_layer.size()[:-2] + (query_layer.size(-2), query_layer.size(-2))),
                                           device=torch.device('cuda'))
            for i in range(rel_emb.size(1)):
                cur_mixed_rel_embd = rel_emb[:, i, :, :]
                cur_rel_layer = self.transpose_for_scores(cur_mixed_rel_embd)
                query_layer += cur_rel_layer
                key_layer += cur_rel_layer

                cur_attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
                cur_attention_scores = cur_attention_scores / math.sqrt(self.attention_head_size)
                attention_scores[:, :, i] = cur_attention_scores[:, :, i]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        else:
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # if attention_mask is not None:
        #     attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        if trans_mask is not None:
            trans_mask = trans_mask.unsqueeze(1).expand_as(attention_scores)
            attention_probs = nn.Softmax(dim=-1)(attention_scores.masked_fill(trans_mask == 0, -1e9))
        else:
            attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer  # , torch.mean(attention_scores, dim=1)


# Copied from transformers.models.bart.modeling_bart.shift_tokens_right
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), float("-inf"))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


class Mutual_Attn(nn.Module):
    "The implemention of the mutual attention between two representations X and Y in the same hidden dim. "

    def __init__(
            self,
            embed_dim: int,
            num_heads: int = 1,
            dropout: float = 0.0,
            layer_norm_eps: float = 1e-8,
            bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.layer_norm_eps = layer_norm_eps
        self.dropout = dropout
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
                self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.embed_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v1_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v2_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.layerNorm_1 = nn.LayerNorm(embed_dim, eps=self.layer_norm_eps)
        self.layerNorm_2 = nn.LayerNorm(embed_dim, eps=self.layer_norm_eps)
        self.FN = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.layerNorm_3 = nn.LayerNorm(embed_dim, eps=self.layer_norm_eps)

    def forward(
            self,
            hidden_states_1,
            hidden_states_2,
            attention_mask_1=None,
            attention_mask_2=None,
            output_attentions=False
    ):
        """Input shape: Batch x Time x Channel"""

        bsz, len_1, embed_dim = hidden_states_1.size()
        bsz, len_2, embed_dim = hidden_states_2.size()
        query_states = self.q_proj(hidden_states_1)
        key_states = self.k_proj(hidden_states_2)
        value_states_1 = self.v1_proj(hidden_states_1)
        value_states_2 = self.v2_proj(hidden_states_2)

        # print(query_states.shape, key_states.shape)
        if attention_mask_1 is not None:
            mask1 = torch.where(attention_mask_1 == 1, torch.zeros_like(attention_mask_1, dtype=torch.float),
                                -1e8 * torch.ones_like(attention_mask_1, dtype=torch.float))

        if attention_mask_2 is not None:
            mask2 = torch.where(attention_mask_2 == 1, torch.zeros_like(attention_mask_2, dtype=torch.float),
                                -1e8 * torch.ones_like(attention_mask_2, dtype=torch.float))

        attn = torch.bmm(query_states, key_states.transpose(1, 2)) / self.scaling
        if attention_mask_2 is not None:
            attn_weight_1 = F.softmax(attn + mask2.unsqueeze(1).repeat([1, len_1, 1]), dim=-1)
        else:
            attn_weight_1 = F.softmax(attn, dim=-1)

        if attention_mask_1 is not None:
            attn_weight_2 = F.softmax(attn.transpose(1, 2) + mask1.unsqueeze(1).repeat([1, len_2, 1]), dim=-1)
        else:
            attn_weight_2 = F.softmax(attn.transpose(1, 2), dim=-1)

        # print(attn_weight_1.shape, value_states_1.shape)

        # temp_value_states_1 = value_states_1
        # temp_value_states_2 = value_states_2

        # value_states_1 = self.layerNorm_1(torch.bmm(attn_weight_1, temp_value_states_2) + value_states_1)
        value_states_2 = self.layerNorm_2(torch.bmm(attn_weight_2, value_states_1) + value_states_2)
        # value_states_2 = F.relu(F.dropout(self.FN(value_states_2), p=self.dropout, training=self.training))
        # value_states_2 = self.layerNorm_3(F.relu(F.dropout(self.FN(value_states_2), p=self.dropout, training=self.training)) + value_states_2)
        value_states_2 = self.layerNorm_3(
            F.dropout(F.relu(self.FN(value_states_2)), p=self.dropout, training=self.training) + value_states_2)

        return value_states_1, value_states_2, attn_weight_1


# Copied from transformers.models.blenderbot.modeling_blenderbot.BlenderbotLearnedPositionalEmbedding with Blenderbot->BlenderbotSmall
class BlenderbotSmallLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
        assert padding_idx is not None, "`padding_idx` should not be None, but of type int"
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions)


# Copied from transformers.models.bart.modeling_bart.BartAttention with Bart->BlenderbotSmall
class BlenderbotSmallAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            is_decoder: bool = False,
            bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
                self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]

        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        assert attn_weights.size() == (
            bsz * self.num_heads,
            tgt_len,
            src_len,
        ), f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"

        if attention_mask is not None:
            assert attention_mask.size() == (
                bsz,
                1,
                tgt_len,
                src_len,
            ), f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        assert attn_output.size() == (
            bsz * self.num_heads,
            tgt_len,
            self.head_dim,
        ), f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"

        attn_output = (
            attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
            .transpose(1, 2)
            .reshape(bsz, tgt_len, embed_dim)
        )

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


# Copied from transformers.models.bart.modeling_bart.BartEncoderLayer with Bart->BlenderbotSmall
class BlenderbotSmallEncoderLayer(nn.Module):
    def __init__(self, config: BlenderbotSmallConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = BlenderbotSmallAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, output_attentions: bool = False):

        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = {'hidden_states': hidden_states}
        if output_attentions:
            outputs['attn_weights'] = attn_weights

        return outputs


# Copied from transformers.models.bart.modeling_bart.BartDecoderLayer with Bart->BlenderbotSmall
class BlenderbotSmallDecoderLayer(nn.Module):
    def __init__(self, config: BlenderbotSmallConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = BlenderbotSmallAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = BlenderbotSmallAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = True,
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (:obj:`torch.FloatTensor`): cross attention input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_attention_mask (:obj:`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            past_key_value (:obj:`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None

        cross_attn_weights = None
        if encoder_hidden_states is not None:
            # residual_root = hidden_states
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None

            hidden_states_encoder, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states_encoder = F.dropout(hidden_states_encoder, p=self.dropout, training=self.training)

            hidden_states = residual + hidden_states_encoder
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class BlenderbotSmallPreTrainedModel(PreTrainedModel):
    config_class = BlenderbotSmallConfig
    base_model_prefix = "model"

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
            "decoder_input_ids": input_ids,
        }
        return dummy_inputs


BLENDERBOT_SMALL_START_DOCSTRING = r"""
    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.BlenderbotSmallConfig`):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

BLENDERBOT_SMALL_GENERATION_EXAMPLE = r"""
    Conversation example::

        >>> from transformers import BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration
        >>> mname = 'facebook/blenderbot_small-90M'
        >>> model = BlenderbotSmallForConditionalGeneration.from_pretrained(mname)
        >>> tokenizer = BlenderbotSmallTokenizer.from_pretrained(mname)
        >>> UTTERANCE = "My friends are cool but they eat too many carbs."
        >>> print("Human: ", UTTERANCE)
        >>> inputs = tokenizer([UTTERANCE], return_tensors='pt')
        >>> inputs.pop("token_type_ids")
        >>> reply_ids = model.generate(**inputs)
        >>> print("Bot: ", tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0])
        what kind of carbs do they eat? i don't know much about carbs.

        >>> REPLY = "I'm not sure"
        >>> print("Human: ", REPLY)
        >>> NEXT_UTTERANCE = (
        ... "My friends are cool but they eat too many carbs.</s> "
        ... "<s>what kind of carbs do they eat? i don't know much about carbs.</s> "
        ... "<s>I'm not sure."
        ... )
        >>> inputs = tokenizer([NEXT_UTTERANCE], return_tensors='pt')
        >>> inputs.pop("token_type_ids")
        >>> next_reply_ids = model.generate(**inputs)
        >>> print("Bot: ", tokenizer.batch_decode(next_reply_ids, skip_special_tokens=True)[0])
"""

BLENDERBOT_SMALL_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using :class:`~transformers.BlenderbotSmallTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        decoder_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.BlenderbotSmallTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__

            BlenderbotSmall uses the :obj:`bos_token_id` as the starting token for :obj:`decoder_input_ids` generation.
            If :obj:`past_key_values` is used, optionally only the last :obj:`decoder_input_ids` have to be input (see
            :obj:`past_key_values`).
        decoder_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Default behavior: generate a tensor that ignores pad tokens in :obj:`decoder_input_ids`. Causal mask will
            also be used by default.

            If you want to change padding behavior, you should read
            :func:`modeling_blenderbot_small._prepare_decoder_inputs` and modify to your needs. See diagram 1 in `the
            paper <https://arxiv.org/abs/1910.13461>`__ for more information on the default strategy.
        encoder_outputs (:obj:`tuple(tuple(torch.FloatTensor)`, `optional`):
            Tuple consists of (:obj:`last_hidden_state`, `optional`: :obj:`hidden_states`, `optional`:
            :obj:`attentions`) :obj:`last_hidden_state` of shape :obj:`(batch_size, sequence_length, hidden_size)`,
            `optional`) is a sequence of hidden-states at the output of the last layer of the encoder. Used in the
            cross-attention of the decoder.
        past_key_values (:obj:`Tuple[Tuple[torch.Tensor]]` of length :obj:`config.n_layers` with each tuple having 2 tuples each of which has 2 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size, sequence_length)`.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        decoder_inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, target_sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`decoder_input_ids` you can choose to directly pass an embedded
            representation. If :obj:`past_key_values` is used, optionally only the last :obj:`decoder_inputs_embeds`
            have to be input (see :obj:`past_key_values`). This is useful if you want more control over how to convert
            :obj:`decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

            If :obj:`decoder_input_ids` and :obj:`decoder_inputs_embeds` are both unset, :obj:`decoder_inputs_embeds`
            takes the value of :obj:`inputs_embeds`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


class BlenderbotSmallEncoder(BlenderbotSmallPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`BlenderbotSmallEncoderLayer`.

    Args:
        config: BlenderbotSmallConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BlenderbotSmallConfig, embed_tokens: Optional[nn.Embedding] = None,
                 transition_type='MHA', use_strategy=False, use_emotion=False, use_bow=False, add_position=False,
                 use_csk=False):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = BlenderbotSmallLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
            self.padding_idx,
        )

        self.layers = nn.ModuleList([BlenderbotSmallEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.trans_interact = Transition_Interact(config, transition_type, use_strategy, use_emotion, use_bow,
                                                  add_position, use_csk)

        if use_emotion:
            self.encoder_attn_emotion = BlenderbotSmallAttention(
                config.d_model,
                4,  # config.decoder_attention_heads
                dropout=config.attention_dropout,
                is_decoder=True,
            )

        self.fusion = nn.Linear(2 * config.d_model, config.d_model)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            inputs_embeds=None,
            trans_cls_index=None,
            emo_trans=None,
            trans_graph=None,
            inter_graph=None,
            trans_edge_type=None,
            inter_edge_type=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            cross_emotion_mask=None
    ):

        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.BlenderbotSmallTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_shape)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        post_hidden_states = None
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if getattr(self.config, "gradient_checkpointing", False):

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                    )
                else:
                    layer_outputs = encoder_layer(hidden_states, attention_mask)

                hidden_states = layer_outputs['hidden_states']

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs['attn_weights'],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        # if comet_hidden_states is not None:
        last_stre_state, trans_emotion_state, last_semantic_delta, sem_cls_embs, \
            strategy_logits, emotion_logits, \
            bow_logits = self.trans_interact(hidden_states, trans_cls_index, \
                                             trans_graph, inter_graph, trans_edge_type, \
                                             inter_edge_type, emo_trans)

        if cross_emotion_mask is not None:  # cross_emotion_mask
            cross_emotion_mask = _expand_mask(cross_emotion_mask, hidden_states.dtype, tgt_len=input_shape[-1])

        # if last_emotion_state is not None:
        hidden_states_emotion, _, _ = self.encoder_attn_emotion(
            hidden_states=hidden_states,
            key_value_states=trans_emotion_state,
            attention_mask=cross_emotion_mask
        )
        hidden_states_emotion = F.dropout(hidden_states_emotion, p=self.dropout, training=self.training)

        gate = nn.Sigmoid()(self.fusion(torch.cat([hidden_states, hidden_states_emotion], dim=-1)))
        hidden_states = gate * hidden_states + (1 - gate) * hidden_states_emotion

        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions,
            last_stre_state=last_stre_state, trans_emotion_state=trans_emotion_state, sem_cls_embs=sem_cls_embs,
            last_semantic_delta=last_semantic_delta, strategy_logits=strategy_logits, emotion_logits=emotion_logits,
            bow_logits=bow_logits)


class BlenderbotSmallDecoder(BlenderbotSmallPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a
    :class:`BlenderbotSmallDecoderLayer`

    Args:
        config: BlenderbotSmallConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BlenderbotSmallConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.embed_positions = BlenderbotSmallLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            self.padding_idx,
        )
        self.layers = nn.ModuleList([BlenderbotSmallDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.state_fusion = nn.Linear(2 * config.d_model, config.d_model)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            last_stre_state=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            decoder_emotion_mask=None,
            return_dict=None,
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.BlenderbotSmallTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            encoder_hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, encoder_sequence_length, hidden_size)`, `optional`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, encoder_sequence_length)`, `optional`):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            past_key_values (:obj:`Tuple[Tuple[torch.Tensor]]` of length :obj:`config.n_layers` with each tuple having 2 tuples each of which has 2 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up
                decoding.

                If :obj:`past_key_values` are used, the user can optionally input only the last
                :obj:`decoder_input_ids` (those that don't have their past key value states given to this model) of
                shape :obj:`(batch_size, 1)` instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size,
                sequence_length)`.
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(self.device)

        if attention_mask is not None and combined_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            combined_attention_mask = combined_attention_mask + _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        if decoder_emotion_mask is not None:
            decoder_emotion_mask = _expand_mask(decoder_emotion_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        # BlenderbotSmall applies layer norm on hidden_states
        gate = nn.Sigmoid()(self.state_fusion(
            torch.cat([inputs_embeds, last_stre_state.unsqueeze(1).expand_as(inputs_embeds)], dim=-1)))
        inputs_embeds = gate * inputs_embeds + (1 - gate) * last_stre_state.unsqueeze(1).expand_as(inputs_embeds)

        hidden_states = inputs_embeds + positions

        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False):
                if use_cache:
                    raise ValueError(
                        "When using `gradient_checkpointing, make sure that `use_cache=False` and `config.use_cache=False`."
                    )

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    combined_attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    None,
                )
            else:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=combined_attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        # hidden_states = hidden_states[:, 1:, :]
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


@add_start_docstrings(
    "The bare BlenderbotSmall Model outputting raw hidden-states without any specific head on top.",
    BLENDERBOT_SMALL_START_DOCSTRING,
)
class BlenderbotSmallModel(BlenderbotSmallPreTrainedModel):
    def __init__(self, config: BlenderbotSmallConfig, transition_type, use_strategy, use_emotion, use_bow, add_position,
                 use_csk):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BlenderbotSmallEncoder(config, self.shared, transition_type, use_strategy, use_emotion, use_bow,
                                              add_position, use_csk)
        self.decoder = BlenderbotSmallDecoder(config, self.shared)

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(BLENDERBOT_SMALL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_role_ids=None,
            decoder_turn_ids=None,
            role_ids=None,
            turn_ids=None,
            decoder_inputs_embeds=None,
            comet_embs=None,
            comet_mask=None,
            comet_embs_st=None,
            comet_mask_st=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        Returns:

        Example::

            >>> from transformers import BlenderbotSmallTokenizer, BlenderbotSmallModel

            >>> model = BlenderbotSmallModel.from_pretrained("facebook/blenderbot_small-90M")
            >>> tokenizer = BlenderbotSmallTokenizer.from_pretrained("facebook/blenderbot_small-90M")

            >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

            >>> last_hidden_states = outputs.last_hidden_state
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            role_ids=decoder_role_ids,
            turn_ids=decoder_turn_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            last_comet_hidden_state=encoder_outputs.last_comet_hidden_state,
            last_comet_hidden_state_st=encoder_outputs.last_comet_hidden_state_st,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class MLP(nn.Module):
    def __init__(self, config, hidden_size, inner_dim):  # in MLP: n_state=3072 (4 * n_embd)
        super().__init__()
        self.c_fc = nn.Linear(hidden_size, inner_dim)
        self.c_proj = nn.Linear(inner_dim, hidden_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h = self.dropout(h)
        h2 = self.c_proj(h)
        h2 = self.dropout(h2)
        return h2


class TransitionBlock(nn.Module):
    def __init__(self, config: BlenderbotSmallConfig, num_heads):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = MaskedSelfAttention(config.d_model, num_heads)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        residual = hidden_states
        hidden_states = self.self_attn(hidden_states, hidden_states, hidden_states, attention_mask)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class Transition_Interact(nn.Module):
    def __init__(self, config, transition_type, use_strategy, use_emotion, use_bow, add_position, use_csk):
        super(Transition_Interact, self).__init__()
        self.transition_type = transition_type
        self.use_strategy = use_strategy
        self.use_emotion = use_emotion
        self.use_bow = use_bow
        self.add_position = add_position
        self.use_csk = use_csk

        if add_position:
            self.strategy_pos_embedding = nn.Embedding(128, config.d_model, padding_idx=0)

        self.fusion = nn.Linear(2 * config.d_model, config.d_model)
        self.transition = MaskedSelfAttention(config.d_model, 16)

        if self.use_emotion:
            self.emotion_head = nn.Linear(config.d_model, 6)

        if self.use_strategy:
            self.strategy_head = nn.Linear(config.d_model, 8)

        if self.use_bow:
            self.bow_head = nn.Linear(config.d_model, 54945, bias=False)

        if self.use_csk:
            self.batchNorm_csk = nn.BatchNorm1d(1024)
            self.emotion_csk_lin = nn.Linear(1024, config.d_model)

    def _compute_delta(self, utt_hidden, utt_trans_hidden):
        delta_list = []
        for i in range(utt_hidden.size(1)):
            delta_list.append((utt_trans_hidden[:, i, :] - utt_hidden[:, i, :]).unsqueeze(1))
        return torch.cat(delta_list, dim=1)

    def _get_max_utterance_repre(self, cur_conv, idx, type='max'):
        cur_batch = []
        text_len = torch.sum(idx != 1).item()
        cur_index = idx[:text_len]
        for j in range(text_len - 1):  # seq_len
            cur_utt = cur_conv[cur_index[j]: cur_index[j + 1], :]
            if type == 'max':
                cur_batch.append(torch.max(cur_utt, dim=0)[0].unsqueeze(0))
            else:
                cur_batch.append(torch.mean(cur_utt, dim=0).unsqueeze(0))
        cur_batch.append(cur_conv[idx[text_len - 1]].unsqueeze(0))
        cur_batch = torch.cat(cur_batch, dim=0)
        return cur_batch

    def _prepare_for_transition(self, hidden_states, trans_cls_index, emotion_csk):  # comet_hidden_states
        transition_representation = []
        semantics_index = []
        for conv, idx, emo_csk in zip(hidden_states, trans_cls_index,
                                      emotion_csk):  # trans_cls_index comet_hidden_states
            cur_trans_repre = []
            cur_idx = []
            # [CLS] token
            cls_emb = torch.index_select(conv, 0, idx)
            text_len = torch.sum(idx != 1).item()
            for idx, item in enumerate(cls_emb):
                item = item.unsqueeze(0)
                if idx < text_len - 1:
                    cur_trans_repre.extend([item, item, item + emo_csk[idx]])
                else:
                    cur_trans_repre.extend([item, item, item])
                cur_idx.append(3 * idx)
            transition_representation.append(torch.cat(cur_trans_repre, dim=0))
            semantics_index.append(cur_idx)
        transition_representation = pad_sequence(transition_representation, batch_first=True, padding_value=0)
        return transition_representation, semantics_index

    def forward(self, hidden_states, trans_cls_index, trans_graph, inter_graph, trans_edge_type, inter_edge_type,
                emo_csk):
        window_text_len = torch.sum(trans_cls_index != 1, dim=-1)

        sem_cls_embs = []
        for item, idx in zip(hidden_states, trans_cls_index):  # trans_cls_index
            cls_emb = torch.index_select(item, 0, idx)
            sem_cls_embs.append(cls_emb)
        sem_cls_embs = pad_sequence(sem_cls_embs, batch_first=True, padding_value=0)

        res_emo_state = None
        res_stra_state = None
        last_delta = None

        # Transition Module
        if self.use_csk:
            emo_csk = self.emotion_csk_lin(
                self.batchNorm_csk(emo_csk.reshape(-1, 1024)).reshape(-1, emo_csk.size(1), 1024))

        transition_inputs, semantics_index = self._prepare_for_transition(hidden_states, trans_cls_index, emo_csk)

        transition_outputs = self.transition(transition_inputs, transition_inputs, transition_inputs, trans_graph,
                                             trans_edge_type)
        interaction_outputs = self.transition(transition_outputs, transition_outputs, transition_outputs, inter_graph,
                                              inter_edge_type)
        gate = nn.Sigmoid()(self.fusion(torch.cat([transition_outputs, interaction_outputs], dim=-1)))
        outputs = (1 - gate) * interaction_outputs + gate * transition_outputs

        sem_trans = []
        for item, idx in zip(outputs, semantics_index):
            sem_cls_emb = torch.index_select(item, 0, torch.tensor(idx, device='cuda'))
            sem_trans.append(sem_cls_emb)
        sem_trans = pad_sequence(sem_trans, batch_first=True, padding_value=0)

        stra_trans_states = []
        for item, idx in zip(outputs, semantics_index):
            str_cls_emb = torch.index_select(item, 0, torch.tensor(idx, device='cuda') + 1)
            stra_trans_states.append(str_cls_emb)
        stra_trans_states = pad_sequence(stra_trans_states, batch_first=True, padding_value=0)

        last_stra_state = []
        for item, idx in zip(stra_trans_states, window_text_len - 1):
            last_stra_state.append(item[idx.item()])
        last_stra_state = torch.stack(last_stra_state, dim=0)

        emo_trans_states = []
        for item, idx in zip(outputs, semantics_index):
            emo_cls_emb = torch.index_select(item, 0, torch.tensor(idx, device='cuda') + 2)
            emo_trans_states.append(emo_cls_emb)
        emo_trans_states = pad_sequence(emo_trans_states, batch_first=True, padding_value=0)

        strategy_logits = None
        emotion_logits = None
        bow_logits = None

        if self.use_bow:
            sentence_delta_hidden = self._compute_delta(sem_cls_embs, sem_trans)

        last_delta = []
        for item, idx in zip(sentence_delta_hidden, window_text_len - 1):
            last_delta.append(item[idx.item()])
        last_delta = torch.stack(last_delta, dim=0)

        if self.use_bow:
            bow_logits = self.bow_head(sentence_delta_hidden)  # sentence_delta_hidden

        if self.use_emotion:
            emotion_logits = self.emotion_head(emo_trans_states)

        if self.use_strategy:
            strategy_logits = self.strategy_head(stra_trans_states)

        return last_stra_state, emo_trans_states, last_delta, sem_cls_embs, strategy_logits, emotion_logits, bow_logits


def RankingLoss(score, label, margin=0.01):
    if score.sum() == 0:
        return 0
    LossFunc = torch.nn.MarginRankingLoss(0.0, reduction='sum')
    loss_mask = (score != 0).long()
    total_loss = LossFunc(score, score, loss_mask) / loss_mask.sum()

    n = score.size(1)
    LossFunc = torch.nn.MarginRankingLoss(margin, reduction='sum')
    for i in range(1, n):
        neg_score, neg_label = score[:, :-i], label[:, :-i]
        pos_score, pos_label = score[:, i:], label[:, i:]
        neg_score = neg_score.contiguous().view(-1)
        pos_score = pos_score.contiguous().view(-1)
        label_mask = ((pos_label != -1) & (neg_label != -1)).long()
        loss_mask = ((pos_label - neg_label) * label_mask).view(-1)
        if loss_mask.sum() == 0:
            continue
        loss = LossFunc(pos_score, neg_score, loss_mask) / loss_mask.abs().sum()
        total_loss += loss

    return total_loss


@add_start_docstrings(
    "The BlenderbotSmall Model with a language modeling head. Can be used for summarization.",
    BLENDERBOT_SMALL_START_DOCSTRING,
)
class BlenderbotSmallForConditionalGeneration(BlenderbotSmallPreTrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder\.version",
        r"decoder\.version",
        r"lm_head\.weight",
    ]

    def __init__(self, config: BlenderbotSmallConfig):
        super().__init__(config)
        self.transition_type = 'MHA'
        self.use_strategy = True
        self.use_bow = True
        self.use_emotion = True
        self.add_position = False
        self.use_csk = True
        self.model = BlenderbotSmallModel(config, self.transition_type, self.use_strategy, self.use_emotion,
                                          self.use_bow, self.add_position, self.use_csk)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        self.dropout = config.dropout

        if self.use_bow:
            self.fusion = nn.Linear(2 * config.d_model, config.d_model)

        self.init_weights()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @add_start_docstrings_to_model_forward(BLENDERBOT_SMALL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(BLENDERBOT_SMALL_GENERATION_EXAMPLE)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            trans_cls_index=None,
            emo_trans=None,
            decoder_inputs_embeds=None,
            strategy_labels=None,
            labels=None,
            trans_emotion_labels=None,
            use_cache=None,
            decoder_emotion_mask=None,
            trans_graph=None,
            inter_graph=None,
            trans_edge_type=None,
            inter_edge_type=None,
            keyword_ids=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            candidate_ids=None,
            candidate_labels=None,
            candidate_mask=None,
            length_penalty=1,
    ):

        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                trans_cls_index=trans_cls_index,
                emo_trans=emo_trans,
                trans_graph=trans_graph,
                inter_graph=inter_graph,
                trans_edge_type=trans_edge_type,
                inter_edge_type=inter_edge_type,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cross_emotion_mask=decoder_emotion_mask
            )

        encoder_hidden_states = encoder_outputs.last_hidden_state
        last_stre_state = encoder_outputs.last_stre_state
        last_semantic_delta = encoder_outputs.last_semantic_delta
        if candidate_labels != None:
            batch_size, candidate_num = candidate_ids.size(0), candidate_ids.size(1)
            assert torch.all(candidate_ids[:, 0, :decoder_input_ids.size(1)] == decoder_input_ids)
            decoder_input_ids = candidate_ids.view(batch_size * candidate_num, candidate_ids.size(-1))
            encoder_hidden_states = torch.repeat_interleave(encoder_hidden_states, candidate_num, dim=0)
            attention_mask = torch.repeat_interleave(attention_mask, candidate_num, dim=0)
            last_stre_state = torch.repeat_interleave(last_stre_state, candidate_num, dim=0)
            last_semantic_delta = torch.repeat_interleave(last_semantic_delta, candidate_num, dim=0)

        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            last_stre_state=last_stre_state,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,  # utterance_mask
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)

        # if encoder_outputs.last_semantic_delta is not None:
        gate = nn.Sigmoid()(
            self.fusion(
                torch.cat([decoder_outputs[0], last_semantic_delta.unsqueeze(1).expand_as(decoder_outputs[0])], dim=-1)
            )
        )
        generate_states = gate * decoder_outputs[0] \
                          + (1 - gate) * last_semantic_delta.unsqueeze(1).expand_as(decoder_outputs[0])
        lm_logits = self.lm_head(generate_states) + self.final_logits_bias

        gold_score, candidate_scores, ranking_loss = None, None, None
        if candidate_labels != None:
            all_lm_logits = lm_logits.view(batch_size, candidate_num, lm_logits.size(1), lm_logits.size(2))
            # [bz, cand_num, seq_len, word_dim]
            lm_logits = all_lm_logits[:, 0, :labels.size(1)].contiguous()
            _output = F.softmax(all_lm_logits, dim=3)
            scores = torch.gather(_output, 3, candidate_ids.unsqueeze(-1)).squeeze(-1)
            candidate_mask = candidate_mask.float()
            scores = torch.mul(scores, candidate_mask).sum(-1) \
                     / ((candidate_mask.sum(-1) + 1e-8) ** length_penalty)
            gold_score, candidate_scores = scores[:, 0], scores[:, 1:]
            ranking_loss = RankingLoss(candidate_scores, candidate_labels)

        loss = None
        masked_lm_loss = None
        emotion_loss = None
        strategy_loss = None
        response_strategy_loss = None
        bow_loss = None
        encoder_bow_loss = None
        decoder_bow_loss = None

        emotion_logits = encoder_outputs.emotion_logits
        strategy_logits = encoder_outputs.strategy_logits
        bow_logits = encoder_outputs.bow_logits

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.reshape(-1, self.config.vocab_size), labels.reshape(-1))
            loss = masked_lm_loss.clone()

        if keyword_ids is not None and bow_logits is not None:
            bow_loss_fct = CrossEntropyLoss()
            bow_logits = bow_logits.unsqueeze(2).expand(-1, -1, keyword_ids.size(2), -1)
            bow_loss = bow_loss_fct(bow_logits.contiguous().view(-1, self.config.vocab_size),
                                    keyword_ids.contiguous().view(-1))
            loss += 0.2 * bow_loss

        if strategy_labels is not None and strategy_logits is not None:
            strategy_loss_fct = CrossEntropyLoss(ignore_index=8)
            strategy_loss = strategy_loss_fct(strategy_logits.view(-1, 8), strategy_labels.view(-1))
            loss += strategy_loss

        if trans_emotion_labels is not None and emotion_logits is not None:
            emotion_loss_fct = CrossEntropyLoss(ignore_index=8)
            emotion_loss = emotion_loss_fct(emotion_logits.view(-1, 6), trans_emotion_labels.view(-1))
            loss += emotion_loss

        if ranking_loss is not None:
            loss = masked_lm_loss.clone() + ranking_loss + 0.01 * bow_loss + 0.1 * strategy_loss + 0.1 * emotion_loss

        return Seq2SeqLMOutput(
            loss=loss,
            lm_loss=masked_lm_loss,
            response_strategy_loss=response_strategy_loss,
            strategy_loss=strategy_loss,
            emotion_loss=emotion_loss,
            bow_loss=bow_loss,
            ranking_loss=ranking_loss,
            encoder_bow_loss=encoder_bow_loss,
            decoder_bow_loss=decoder_bow_loss,
            lm_logits=lm_logits,
            strategy_logits=strategy_logits,
            emotion_logits=emotion_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self, decoder_input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def adjust_logits_during_generation(self, logits, cur_len, max_length):
        if cur_len == max_length - 1 and self.config.eos_token_id is not None:
            self._force_token_id_to_be_generated(logits, self.config.eos_token_id)
        return logits

    @staticmethod
    def _force_token_id_to_be_generated(scores, token_id) -> None:
        """force one of token_ids to be generated by setting prob of all other tokens to 0 (logprob=-float("inf"))"""
        scores[:, [x for x in range(scores.shape[1]) if x != token_id]] = -float("inf")

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past
