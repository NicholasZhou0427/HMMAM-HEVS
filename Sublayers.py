import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class ScaledDotProductAttention(nn.Module):
    def __init__(self, attn_dropout):
        super().__init__()
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-1e20'))

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, attn_dropout):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(attn_dropout)

    def forward(self, q, k, v, mask=None):
        batch_size, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        assert len_k == len_v, "计算注意力时k和v的长度相同"
        q = self.w_q(q).view(batch_size, len_q, self.n_head, -1).transpose(1, 2)
        k = self.w_k(k).view(batch_size, len_k, self.n_head, -1).transpose(1, 2)
        v = self.w_v(v).view(batch_size, len_v, self.n_head, -1).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # mask在头维度广播

        out = self.attention(q, k, v, mask=mask)
        out = out.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        out = self.w_concat(out)
        return out


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, FF_dim, FF_dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, FF_dim)
        self.linear2 = nn.Linear(FF_dim, d_model)
        self.dropout = nn.Dropout(p=FF_dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, len_seq, d_model):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(len_seq, d_model))

    def _get_sinusoid_encoding_table(self, len_seq, d_model):
        """ Sinusoid position encoding table """
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_model) for hid_j in range(d_model)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(len_seq)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        return self.pos_table[:, :seq_len].clone().detach()


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, FF_dim, dropout_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head, attn_dropout=dropout_prob)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.LN1 = nn.LayerNorm(d_model, eps=1e-6)

        self.FFN = PositionwiseFeedForward(d_model=d_model, FF_dim=FF_dim, FF_dropout=dropout_prob)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.LN2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, src_mask=None):
        x_res = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)
        x = self.dropout1(x)
        x = self.LN1(x + x_res)

        x_res = x
        x = self.FFN(x)
        x = self.dropout2(x)
        x = self.LN2(x + x_res)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, FF_dim, dropout_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head, attn_dropout=dropout_prob)
        self.LN1 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(p=dropout_prob)

        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head, attn_dropout=dropout_prob)
        self.LN2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout2 = nn.Dropout(p=dropout_prob)

        self.FFN = PositionwiseFeedForward(d_model=d_model, FF_dim=FF_dim, FF_dropout=dropout_prob)
        self.LN3 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout3 = nn.Dropout(p=dropout_prob)

    def forward(self, dec, enc, trg_mask, src_mask):
        x_res = dec
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)
        x = self.dropout1(x)
        x = self.LN1(x + x_res)

        x_res = x
        x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)
        x = self.dropout2(x)
        x = self.LN2(x + x_res)

        x_res = x
        x = self.FFN(x)
        x = self.dropout3(x)
        x = self.LN3(x + x_res)
        return x
