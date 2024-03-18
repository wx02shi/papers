import torch
import torch.nn as nn

from sublayers.multi_head_attention import MultiHeadAttention
from sublayers.position_wise_feed_forward import PositionWiseFeedForward

class Decoder(nn.Module):

    def __init__(self, d_embedding, n_head, d_ff, dropout):
        
        super().__init__()
        
        self.self_attention = MultiHeadAttention(d_embedding, n_head)
        self.cross_attention = MultiHeadAttention(d_embedding, n_head)
        self.feed_forward = PositionWiseFeedForward(d_embedding, d_ff)
        self.norm = nn.LayerNorm(d_embedding)
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, x, enc_out, src_mask, tgt_mask):
        
        out1 = self.self_attention(x, x, x, tgt_mask)
        out1 = self.dropout(out1)
        out1 = out1 + x
        out1 = self.norm(out1)
        out2 = self.cross_attention(out1, enc_out, enc_out, src_mask)
        out2 = self.dropout(out2)
        out2 = out2 + out1
        out2 = self.norm(out2)
        out3 = self.feed_forward(out2)
        out3 = self.dropout(out3)
        out3 = out3 + out2
        out3 = self.norm(out3)

        return out3