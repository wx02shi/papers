import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):

    def __init__(self, d_embedding, max_len = 5000):
        
        super().__init__()
        
        pe = torch.zeros(max_len, d_embedding)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_embedding, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_embedding))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

    
    def forward(self, x):
        
        return x + self.pe[:, :x.size(1)]
