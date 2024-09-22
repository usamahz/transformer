import torch
import numpy as np

def create_src_mask(seq):
    return (seq != 0).unsqueeze(1).unsqueeze(2)

def create_tgt_mask(seq):
    seq_len = seq.size(1)
    mask = (seq != 0).unsqueeze(1).unsqueeze(2)
    subsequent_mask = torch.triu(torch.ones((1, seq_len, seq_len), device=seq.device), diagonal=1).bool()
    return mask & ~subsequent_mask

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0