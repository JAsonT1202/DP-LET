import torch
import torch.nn as nn


def positional_encoding(seq_len, num_hidden):
    """
    :param seq_len: = num_patches
    :param num_hidden:  = patch_lens->num_hidden
    :return:
    """
    W_pos = torch.empty(size=(seq_len, num_hidden))
    nn.init.uniform_(W_pos, -0.02, 0.02)
    return nn.Parameter(W_pos, requires_grad=True)