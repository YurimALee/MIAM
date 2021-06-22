import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np
from help_physionet import random_mask
from attention_graph_util import *

import copy


class Multi_Duration_Pipeline_Residual(nn.Module):
    def __init__(self, input_dim, d_model, d_ff, num_stack, num_heads, max_length, n_iter):
        super().__init__()

        self.n_iter = n_iter

        self.obs_embed = Embedder(input_dim, d_model)
        self.mask_embed = Embedder(input_dim, d_model)
        self.deltas_embed = Embedder(input_dim, d_model)

        self.pe = PositionalEncoder_TimeDescriptor(d_model, max_length)

        self.obs_encoding_block = Encoding_Block(d_model, max_length, num_heads, d_ff, num_stack)
        self.mask_encoding_block = Encoding_Block(d_model, max_length, num_heads, d_ff, num_stack)
        self.deltas_encoding_block = Encoding_Block(d_model, max_length, num_heads, d_ff, num_stack)
        self.comb_encoding_block = Encoding_Block(d_model, max_length, num_heads, d_ff, num_stack)
        self.missing_comb_encoding_block = Encoding_Block(d_model, max_length, num_heads, d_ff, num_stack)

        self.fin_block = Encoding_Block(d_model, max_length, num_heads, d_ff, num_stack)

        obs_embed_weight = self.obs_embed.embed.weight
        n, v = obs_embed_weight.size()
        self.decoder = nn.Linear(n, v, bias=False)
        self.decoder.weight.data = obs_embed_weight.transpose(1, 0)
        self.decoder_bias = nn.Parameter(torch.zeros(v))

        self.classifier_los = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.BatchNorm1d(d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, 1),)

        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.BatchNorm1d(d_model),
            # nn.Sigmoid(),
            nn.LeakyReLU(),
            nn.Linear(d_model, 1),
        )

        self.classifier_phe = nn.Sequential(
            nn.Linear(d_model*2, d_model),
            nn.BatchNorm1d(d_model),
            nn.Tanh(),
            # nn.LeakyReLU(),
            nn.Linear(d_model, 25)
        )

        self.dropout = nn.Dropout(0.3)

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.size()) == 1:
                continue
            stv = 1. / math.sqrt(weight.size(1))
            nn.init.uniform_(weight, -stv, stv)

    def forward(self, data, mask, times, deltas, attn_mask):
        """
        :param src: Batch x Max_seq_len x Variable
        :param mask: Batch x Max_seq_len x Max_seq_len
        """

        # make attn_mask
        batch_size, seq_len, var_num = data.size()
        attn_mask = attn_mask.unsqueeze(1)
        attn_mask = attn_mask.expand(batch_size, seq_len, seq_len)

        # Datas
        d_z = data#[:, 0, :, :]

        # Input embedding
        x_z = self.obs_embed(d_z)
        m = self.mask_embed(mask)
        d = self.deltas_embed(deltas)

        # Positional encoding
        x_z, m, d = self.pe(x_z, m, d, times)

        # H_X, H_M, H_D
        x_z = self.obs_encoding_block(x_z, x_z, attn_mask)
        m = self.mask_encoding_block(m, m, attn_mask)
        d = self.deltas_encoding_block(d, d, attn_mask)

        # Attention Distillation
        missing_comb_z = self.missing_comb_encoding_block(d, m, attn_mask)
        for n in range(self.n_iter):
            comb_z = self.comb_encoding_block(missing_comb_z, x_z, attn_mask)
            x_z = self.obs_encoding_block(comb_z, x_z, attn_mask)

        """Imputation Part"""
        rx, ox, mmsk = random_mask(d_z)
        # Input Embedding
        x_mskd = self.obs_embed(rx.to(data.device))
        m_mskd = self.mask_embed(mask.to(data.device))
        d_mskd = self.deltas_embed(deltas)

        # Positional encoding
        x_mskd, m_mskd, d_mskd = self.pe(x_mskd, m_mskd, d_mskd, times)

        # Masked MHA
        x_d = self.obs_encoding_block(x_mskd, x_mskd, attn_mask)

        # Encoder-decoder Attention
        x_d = self.obs_encoding_block(x_z, x_d, attn_mask)
        x_final = x_d + x_z

        x_dd = self.decoder(x_final) + self.decoder_bias

        # Classification
        x_avg = x_z.mean(dim=1)
        m_avg = missing_comb_z.mean(dim=1)
        x_m_cat = torch.stack((x_avg, m_avg), 1).reshape([x_avg.shape[0], -1])
        phe_out = torch.sigmoid(self.classifier_phe(x_m_cat))
        out = self.classifier(x_m_cat)
        reg_out = self.classifier_los(x_m_cat)
        y = torch.sigmoid(out).squeeze(-1)

        return reg_out.squeeze(-1), phe_out, y, x_dd


class Encoding_Block(nn.Module):
    def __init__(self, d_model, max_length, num_heads, d_ff, num_stack):
        super().__init__()

        self.N = num_stack

        self.layers = get_clones(EncoderLayer(d_model, max_length, num_heads, d_ff), num_stack)
        self.norm = Norm(d_model)

    def forward(self, q, k, attn_mask):
        # MHA Encoding
        for i in range(self.N):
            q, k = self.layers[i](q, k, attn_mask)

        # Normalize
        encoded_data = self.norm(q)
        return encoded_data


def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Linear(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model

        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, m, delta, t):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        m = m * math.sqrt(self.d_model)
        delta = delta * math.sqrt(self.d_model)

        # add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:, :seq_len], requires_grad=False).cuda()
        m = m + Variable(self.pe[:, :seq_len], requires_grad=False).cuda()
        delta = delta + Variable(self.pe[:, :seq_len], requires_grad=False).cuda()
        return x, m, delta


class PositionalEncoder_TimeDescriptor(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

    def get_sinusoid_encoding_table(self, max_seq_len, d_model, t_set):
        def cal_angle(position, hid_idx):
            return position / np.power(max_seq_len, 2 * (hid_idx // 2) / d_model)

        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_j) for hid_j in range(d_model)]

        t_set = t_set.detach().cpu().numpy()

        # sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(max_seq_len)])
        sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in t_set])
        sinusoid_table[:, 0::2, :] = np.sin(sinusoid_table[:, 0::2, :])  # dim 2i
        sinusoid_table[:, 1::2, :] = np.cos(sinusoid_table[:, 1::2, :])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table)

    def time_encoding(self, t):
        batch_size = t.size(0)

        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(batch_size, self.max_seq_len, self.d_model)
        pe = self.get_sinusoid_encoding_table(self.max_seq_len, self.d_model, t)
        # for b in range(batch_size):
        #     pe[b] = self.get_sinusoid_encoding_table(self.max_seq_len, self.d_model, t[b])
        return pe.permute(0, 2, 1)

    def forward(self, x, m, delta, t):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        m = m * math.sqrt(self.d_model)
        delta = delta * math.sqrt(self.d_model)

        pos = self.time_encoding(t)

        x = x + Variable(pos, requires_grad=False).cuda()
        m = m + Variable(pos, requires_grad=False).cuda()
        delta = delta + Variable(pos, requires_grad=False).cuda()

        # seq_len = x.size(1)
        # x = x + Variable(self.pe[:, :seq_len], requires_grad=False).cuda()
        return x, m, delta


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.2):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, tp, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)  # [batch_size * len_q * n_heads * hidden_dim]
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)  # [batch_size * len_q * n_heads * hidden_dim]
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)  # [batch_size * len_q * n_heads * hidden_dim]

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)  # [batch_size * n_heads * len_q * hidden_dim]
        q = q.transpose(1, 2)  # [batch_size * n_heads * len_q * hidden_dim]
        v = v.transpose(1, 2)  # [batch_size * n_heads * len_q * hidden_dim]

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.h, 1, 1)  # [batch_size x n_heads x len_q x len_k]


        if tp == 1:  # if transpose
            k = k.transpose(2, 3)  # [batch_size * n_heads * hidden_dim * len_q]
            q = q.transpose(2, 3)  # [batch_size * n_heads * hidden_dim * len_q]
            v = v.transpose(2, 3)  # [batch_size * n_heads * hidden_dim * len_q]


        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        if tf == 1:
            scores = scores.transpose(2, 3)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)

        output = self.out(concat)

        return output


def attention(q, k, v, d_k, mask=None, dropout=None):
    """
    :param q: Batch x n_head x max_seq_len x variable
    :param k: Batch x n_head x max_seq_len x variable
    :param v: Batch x n_head x max_seq_len x variable
    :param d_k:
    :param mask: Batch x n_had x max_seq_len x max_seq_len
    :param dropout:
    :return:
    """

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # [Batch x n_head x max_seq_len x max_seq_len]

    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, max_length, heads, d_ff, dropout=0):
        super().__init__()
        self.norm_q = Norm(d_model)
        self.norm_k = Norm(d_model)
        self.norm_q_attn = Norm(d_model)

        self.attn = MultiHeadAttention(heads, d_model)
        self.time_attn = MultiHeadAttention(heads, d_model)
        self.var_attn = MultiHeadAttention(heads, d_model)

        self.ff = FeedForward(d_model, d_ff)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

        self.cat_lin = nn.Linear(128, 64)

    def forward(self, q, k, mask):
        method = 'none'

        if method == 'none':
            q2 = self.norm_q(q)
            k2 = self.norm_k(k)
            q = q + self.dropout_1(self.attn(q2, k2, k2, 0, mask))

            q2 = self.norm_q_attn(q)
            q = q + self.dropout_2(self.ff(q2))

        elif method == 't_first':
            q2 = self.norm_q(q)
            k2 = self.norm_k(k)
            q = q + self.dropout_1(self.time_attn(q2, k2, k2, 0, mask))

            q2 = self.norm_q_attn(q)
            q = q + self.dropout_2(self.ff(q2))

            q2 = self.norm_q(q)
            k2 = self.norm_k(k)
            q = q + self.dropout_1(self.var_attn(q2, k2, k2, 1, mask=None))

            q2 = self.norm_q_attn(q)
            q = q + self.dropout_2(self.ff(q2))

        elif method == 'v_first':
            q2 = self.norm_q(q)
            k2 = self.norm_k(k)
            q = q + self.dropout_1(self.var_attn(q2, k2, k2, 1, mask=None))

            q2 = self.norm_q_attn(q)
            q = q + self.dropout_2(self.ff(q2))

            q2 = self.norm_q(q)
            k2 = self.norm_k(k)
            q = q + self.dropout_1(self.time_attn(q2, k2, k2, 0, mask))

            q2 = self.norm_q_attn(q)
            q = q + self.dropout_2(self.ff(q2))

        elif method == 'multiply':
            q2 = self.norm_q(q)
            k2 = self.norm_k(k)

            q = q + torch.mul(self.dropout_1(self.time_attn(q2, k2, k2, 0, mask)),\
                              self.dropout_1(self.var_attn(q2, k2, k2, 1, mask=None)))

            q2 = self.norm_q_attn(q)
            q = q + self.dropout_2(self.ff(q2))

        elif method == 'withoutlast':
            q2 = self.norm_q(q)
            k2 = self.norm_k(k)
            q = q + self.dropout_1(self.attn(q2, k2, k2, 0, mask))

            # q2 = self.norm_q_attn(q)
            q = q + self.dropout_2(self.ff(q))

        elif method == 'concat':
            q2 = self.norm_q(q)
            k2 = self.norm_k(k)

            # q_t = self.dropout_1(self.time_attn(q2, k2, k2, 0, mask))
            # q_v = self.dropout_1(self.var_attn(q2, k2, k2, 1, mask=None))
            # q_cat = self.cat_lin(torch.cat((q_t, q_v), 2)

            cat = torch.cat((self.dropout_1(self.time_attn(q2, k2, k2, 0, mask)), self.dropout_1(self.var_attn(q2, k2, k2, 1, mask=None))),2)
            q_cat = self.cat_lin(cat)

            q = q + q_cat#torch.cat(self.dropout_1(self.time_attn(q2, k2, k2, 0, mask)), self.dropout_1(self.var_attn(q2, k2, k2, 1, mask=None)))

            q2 = self.norm_q_attn(q)
            q = q + self.dropout_2(self.ff(q2))


        else:
            pass

        # stack_matrix = torch.cat((q, qtrans), -1)
        # combine_q = self.combine_nn(stack_matrix)

        return q, k
        # return combine_q, k

