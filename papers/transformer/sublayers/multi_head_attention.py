import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):

    def __init__(self, d_embedding, n_head):
        
        super().__init__()
        assert d_embedding % n_head == 0

        self.d_embedding = d_embedding
        self.n_head = n_head
        self.d_K = d_embedding // n_head

        self.W_Q = nn.Linear(d_embedding, d_embedding)
        self.W_K = nn.Linear(d_embedding, d_embedding)
        self.W_V = nn.Linear(d_embedding, d_embedding)
        self.W_O = nn.Linear(d_embedding, d_embedding)

    
    def attention(self, Q, K, V, mask = None):
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_K)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_probs = F.softmax(attn_scores, dim = -1)
        out = torch.matmul(attn_probs, V)

        return out, attn_probs
    

    def make_heads(self, x):
            
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.n_head, self.d_K).transpose(1, 2)


    def join_heads(self, x):
            
        batch_size = x.size(0)
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_embedding)


    def forward(self, Q, K, V, mask = None):
        
        QW_q = self.W_Q(Q)
        KW_k = self.W_K(K)
        VW_v = self.W_V(V)

        QW_q = self.make_heads(QW_q)
        KW_k = self.make_heads(KW_k)
        VW_v = self.make_heads(VW_v)

        attn, _ = self.attention(QW_q, KW_k, VW_v, mask)
        attn = self.join_heads(attn)

        out = self.W_O(attn)

        return out
