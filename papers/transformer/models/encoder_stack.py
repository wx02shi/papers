import torch
import torch.nn as nn

from layers.encoder import Encoder

class EncoderStack(nn.Module):

    def __init__(self, d_embedding, n_head, d_ff, dropout, n_layer):
        
        super().__init__()
        
        self.layers = nn.ModuleList([Encoder(d_embedding, n_head, d_ff, dropout) for _ in range(n_layer)])

    
    def forward(self, x, src_mask):
        
        for layer in self.layers:
            x = layer(x, src_mask)

        return x
