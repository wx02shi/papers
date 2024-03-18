import torch
import torch.nn as nn

from layers.decoder import Decoder

class DecoderStack(nn.Module):

    def __init__(self, d_embedding, n_head, d_ff, dropout, n_layer):
        
        super().__init__()
        
        self.layers = nn.ModuleList([Decoder(d_embedding, n_head, d_ff, dropout) for _ in range(n_layer)])

    
    def forward(self, x, enc_out, src_mask, tgt_mask):
        
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)

        return x
