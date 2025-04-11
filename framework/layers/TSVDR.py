import torch

def TSVDR(x, cut):

    x_ = x.clone().detach()
    U, S, V = torch.svd(x)
    S[:, cut:] = 0 
    return U @ torch.diag(S[0, :]) @ V