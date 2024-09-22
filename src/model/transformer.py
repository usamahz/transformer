import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model, num_heads, num_layers, d_ff, dropout, max_seq_length):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            vocab=src_vocab,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_length=max_seq_length
        )
        self.decoder = Decoder(
            vocab=tgt_vocab,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_length=max_seq_length
        )
        self.generator = nn.Linear(d_model, tgt_vocab)
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        return self.generator(dec_output)

    def encode(self, src, src_mask):
        return self.encoder(src, src_mask)

    def decode(self, tgt, memory, src_mask, tgt_mask):
        return self.decoder(tgt, memory, src_mask, tgt_mask)

    def generate(self, src, src_mask, max_len):
        memory = self.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(2).long().to(src.device)  # start token
        for i in range(max_len-1):
            tgt_mask = (torch.triu(torch.ones(1, i+1, i+1)) == 1).transpose(1, 2).type(torch.bool).to(src.device)
            out = self.decode(ys, memory, src_mask, tgt_mask)
            prob = self.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()
            ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
            if next_word == 3:  # end token
                break
        return ys