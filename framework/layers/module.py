import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .AttentionModule import MultiHeadAttention
from .PositionalModule import positional_encoding
from .LocalFeatureEnhancement import LELayer


class TSTEncoderLayer(nn.Module):
    def __init__(self, num_hidden, num_heads, num_hidden_key_per_head=None, num_hidden_value_per_head=None,
                 num_hidden_ff=256, attn_drop_rate=0., drop_rate=0., res_attn_scores=False):
        super(TSTEncoderLayer, self).__init__()
        num_hidden_key_per_head = num_hidden // num_heads if num_hidden_key_per_head is None else \
            num_hidden_key_per_head
        num_hidden_value_per_head = num_hidden // num_heads if num_hidden_value_per_head is None else \
            num_hidden_value_per_head
        self.res_attn_scores = res_attn_scores
        self.attn = MultiHeadAttention(num_hidden, num_heads, num_hidden_key_per_head, num_hidden_value_per_head,
                                       res_attn_scores, attn_drop_rate=attn_drop_rate, proj_drop_rate=drop_rate)

        self.dropout_attn = nn.Dropout(drop_rate)
        self.norm_attn = nn.BatchNorm1d(num_hidden)

        self.linear1 = nn.Linear(num_hidden, num_hidden_ff)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(num_hidden_ff, num_hidden)
        self.dropout = nn.Dropout(drop_rate)

        self.dropout_ffn = nn.Dropout(drop_rate)
        self.norm_ffn = nn.BatchNorm1d(num_hidden)

    def forward(self, x, pre_softmax_attn_scores=None, key_mask=None, attn_mask=None):
        # x.shape [batch*num_nodes, in_lens, num_hidden]
        if self.res_attn_scores:
            x_out, attn_weights, attn_scores = self.attn(x, x, x, pre_softmax_attn_scores, key_mask, attn_mask)
        else:
            x_out, attn_weights = self.attn(x, x, x, key_mask, attn_mask)

        x = x + self.dropout_attn(x_out)

        x = x.transpose(1, 2)
        x = self.norm_attn(x)
        x = x.transpose(1, 2)

        x_out = self.activation(self.linear1(x))
        x_out = self.dropout(x_out)
        x_out = self.linear2(x_out)
        x = x + self.dropout_ffn(x_out)

        x = x.transpose(1, 2)
        x = self.norm_ffn(x)
        x = x.transpose(1, 2)

        if self.res_attn_scores:
            return x, attn_scores
        else:
            return x


class TSTEncoder(nn.Module):
    def __init__(self, num_hidden, num_heads, num_hidden_key_per_head=None, num_hidden_value_per_head=None,
                 num_hidden_ff=None, attn_drop_rate=0., drop_rate=0., res_attn_scores=False, num_layers=1):
        super(TSTEncoder, self).__init__()

        self.encoder_layers = nn.ModuleList([TSTEncoderLayer(num_hidden, num_heads, num_hidden_key_per_head,
                                                             num_hidden_value_per_head, num_hidden_ff,
                                                             attn_drop_rate, drop_rate,
                                                             res_attn_scores=res_attn_scores) for _ in range(num_layers)])
        self.res_attn_scores = res_attn_scores

    def forward(self, x, key_mask=None, attn_mask=None):
        output = x
        scores = None
        if self.res_attn_scores:
            for encoder_layer in self.encoder_layers:
                output, scores = encoder_layer(output, pre_softmax_attn_scores=scores, key_mask=key_mask,
                                               attn_mask=attn_mask)
            return output
        else:
            for encoder_layer in self.encoder_layers:
                output = encoder_layer(output, key_mask=key_mask, attn_mask=attn_mask)
            return output


class TSTiEncoder(nn.Module):
    def __init__(self, num_patches, patch_lens, num_layers=3, num_hidden=128, num_heads=16, num_hidden_key_per_head=None,
                 num_hidden_value_per_head=None, num_hidden_ff=256, attn_drop_rate=0.0, drop_rate=0.0,
                 res_attn_scores=False):
        """
        num_patches = N
        patch_length = P
        """
        super(TSTiEncoder, self).__init__()

        self.num_patches = num_patches
        self.patch_lens = patch_lens
        
        self.parallel_embedding = LELayer(patch_len=patch_lens, model_dim=num_hidden, 
                                                         tcn_channels=[128, 256, num_hidden])

        # Positional Encoding
        self.pos_embedding = positional_encoding(num_patches, num_hidden)
        self.dropout = nn.Dropout(drop_rate)

        self.encoder = TSTEncoder(num_hidden, num_heads, num_hidden_key_per_head, num_hidden_value_per_head,
                                  num_hidden_ff, attn_drop_rate, drop_rate, res_attn_scores, num_layers)

    def forward(self, x):
        # x.shape [batch, nodes, num_patches, patch_lens]
        num_nodes = x.shape[1]

        x = self.parallel_embedding(x)  # 输出维度 [batch, nodes, num_patches, num_hidden]
        
        # Reshape for positional embedding
        y = torch.reshape(x, (x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))
        y = self.dropout(y + self.pos_embedding)

        output = self.encoder(y)
        output = torch.reshape(output, (-1, num_nodes, output.shape[-2], output.shape[-1]))
        # output.shape [batch, num_nodes, num_patches, num_hidden]
        return output


class Flatten_Head(nn.Module):
    def __init__(self, num_hidden_flatten, out_lens, drop_rate=0.):
        super(Flatten_Head, self).__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(num_hidden_flatten, out_lens)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        # x.shape [batch, nodes, num_patches, patch_lens]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x
