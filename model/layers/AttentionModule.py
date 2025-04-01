import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """
    def __init__(self, num_hidden, num_heads, attn_drop_rate=0., res_attn_score=False):
        super(ScaledDotProductAttention, self).__init__()
        self.attn_dropout = nn.Dropout(p=attn_drop_rate)
        self.res_attn_score = res_attn_score
        num_hidden_per_head = num_hidden // num_heads
        self.scale = nn.Parameter(torch.tensor(num_hidden_per_head ** -0.5), requires_grad=False)

    def forward(self, query, key, value, pre_softmax_attn_score=None, key_mask=None, attn_mask=None):
        """
        query: [batch, num_heads, query_len, num_hidden]
        key: [batch, num_heads, num_hidden, key_len]
        value: [batch, num_heads, key_len, num_hidden]
        pre_softmax_attn_score: [batch, num_head, query_len, key_len]
        key_mask: [batch, key_len]
        attn_mask: [1, query_len, query_len]
        """
        attn_score = torch.matmul(query, key) * self.scale
        if pre_softmax_attn_score is not None:
            attn_score = attn_score + pre_softmax_attn_score
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_score.masked_fill_(attn_mask, -np.inf)
            else:
                attn_score += attn_mask

        if key_mask is not None:
            attn_score.masked_fill_(key_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        attn_weights = F.softmax(attn_score, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        output = torch.matmul(attn_weights, value)

        if self.res_attn_score:
            return output, attn_weights, attn_score
        else:
            return output, attn_weights


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """
    def __init__(self, num_hidden, num_heads, num_hidden_key_per_head=None, num_hidden_value_per_head=None,
                 res_attn_score=False, attn_drop_rate=0., proj_drop_rate=0.):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.num_hidden_key_per_head = num_hidden // num_heads if num_hidden_key_per_head == 'None' else \
            num_hidden_key_per_head
        self.num_hidden_value_per_head = num_hidden // num_heads if num_hidden_value_per_head == 'None' \
            else num_hidden_value_per_head

        self.linear_q = nn.Linear(num_hidden, num_heads * self.num_hidden_key_per_head)
        self.linear_k = nn.Linear(num_hidden, num_heads * self.num_hidden_key_per_head)
        self.linear_v = nn.Linear(num_hidden, num_heads * self.num_hidden_value_per_head)

        self.res_attn_score = res_attn_score
        self.linear = nn.Linear(self.num_hidden_value_per_head * num_heads, num_hidden)

        self.attention = ScaledDotProductAttention(num_hidden=num_hidden, num_heads=num_heads,
                                                   attn_drop_rate=attn_drop_rate, res_attn_score=res_attn_score)
        self.dropout = nn.Dropout(proj_drop_rate)

    def forward(self, query, key, value, pre_softmax_attn_score=None, key_mask=None, attn_mask=None):
        batch_mul_nodes = query.shape[0]
        if key is None:
            key = query
        if value is None:
            value = query
        # q_s    : [bs x n_heads x max_q_len x d_k]
        # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        # v_s    : [bs x n_heads x q_len x d_v]
        query = self.linear_q(query).view(batch_mul_nodes, -1, self.num_heads, self.num_hidden_key_per_head).transpose(1, 2)
        key = self.linear_k(key).view(batch_mul_nodes, -1, self.num_heads, self.num_hidden_key_per_head).permute(0, 2, 3, 1)
        value = self.linear_v(value).view(batch_mul_nodes, -1, self.num_heads, self.num_hidden_value_per_head).transpose(1, 2)

        if self.res_attn_score:
            output, attn_weights, attn_scores = self.attention(query, key, value, pre_softmax_attn_score, key_mask,
                                                               attn_mask)
        else:
            output, attn_weights = self.attention(query, key, value, key_mask=key_mask, attn_mask=attn_mask)

        output = output.transpose(1, 2).contiguous().view(batch_mul_nodes, -1, self.num_heads * self.num_hidden_value_per_head)
        output = self.dropout(self.linear(output))

        if self.res_attn_score:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights
