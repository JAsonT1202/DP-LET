import torch
import torch.nn as nn
import torch.nn.functional as F
from .InstanceNormModule import RevIN
from .module import TSTiEncoder, Flatten_Head
from .TCN import TemporalConvNet
from .TSVDR import TSVDR

class Predictor(nn.Module):
    def __init__(self, **configs):
        super(Predictor, self).__init__()
        model_configs = configs['model']

        num_nodes = model_configs['num_nodes']
        in_lens = model_configs['in_lens']
        out_lens = model_configs['out_lens']

        num_layers = model_configs['num_layers']
        num_heads = model_configs['num_heads']
        num_hidden = model_configs['num_hidden']
        num_hidden_key_per_head = model_configs['num_hidden_key_per_head']
        num_hidden_value_per_head = model_configs['num_hidden_value_per_head']
        num_hidden_ff = model_configs['num_hidden_ff']
        drop_rate = model_configs['drop_rate']
        attn_drop_rate = model_configs['attn_drop_rate']
        flatten_drop_rate = model_configs['flatten_drop_rate']
        res_attn_scores = model_configs['res_attn_scores']
        self.patch_lens = model_configs['patch_lens']
        self.stride = model_configs['stride']

        num_patches = int((in_lens - self.patch_lens) / self.stride + 1)
        num_patches += 1

        if configs['model']['dim'] == 'time_feat':
            self.revin_layer = RevIN(num_features=5)
        else:
            self.revin_layer = RevIN(num_nodes)

        self.if_revin = configs['model']['if_revin']
        self.if_decompose = configs['model']['if_decomp']
        self.if_denoise = configs['model'].get('if_denoise', False)
        self.svd_cut = configs['model'].get('svd_cut', 31)  # Default number of singular values to retain


        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.model = TSTiEncoder(num_patches, self.patch_lens, num_layers, num_hidden, num_heads,
                                    num_hidden_key_per_head, num_hidden_value_per_head, num_hidden_ff,
                                    attn_drop_rate, drop_rate, res_attn_scores)
        num_hidden_flatten = num_patches * num_hidden
        self.linear_head = Flatten_Head(num_hidden_flatten, out_lens, flatten_drop_rate)

    def forward(self, x): 
        if self.if_denoise:
            x = TSVDR(x, self.svd_cut) 

        if self.if_revin:
            x = self.revin_layer(x, 'norm')
            
        x = x.permute(0, 2, 1)
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_lens, step=self.stride)

        # Encoder
        x = self.model(x)

        # Prediction Head
        x = self.linear_head(x)

        x = x.permute(0, 2, 1)
        if self.if_revin:
            x = self.revin_layer(x, 'denorm')
            
        return x
