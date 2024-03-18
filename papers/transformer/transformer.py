import torch
import torch.nn as nn
from embeddings.positional_encoding import PositionalEncoding
from models.encoder_stack import EncoderStack
from models.decoder_stack import DecoderStack

class Transformer(nn.Module):

    def __init__(self, src_vocab_size, tgt_vocab_size, d_embedding, n_head, d_ff, n_layer, max_seq_len, dropout):
        
        super().__init__()
        
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_embedding)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_embedding)
        self.positional_encoding = PositionalEncoding(d_embedding, max_seq_len)
        
        self.encoder_stack = EncoderStack(d_embedding, n_head, d_ff, dropout, n_layer)
        self.decoder_stack = DecoderStack(d_embedding, n_head, d_ff, dropout, n_layer)
        
        self.fc = nn.Linear(d_embedding, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)


    def generate_mask(self, src, tgt):

        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        seq_len = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        
        return src_mask, tgt_mask


    def forward(self, src, tgt):

        src_mask, tgt_mask = self.generate_mask(src, tgt)
        
        src = self.encoder_embedding(src)
        tgt = self.decoder_embedding(tgt)
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        
        enc_out = self.encoder_stack(src, src_mask)
        dec_out = self.decoder_stack(tgt, enc_out, src_mask, tgt_mask)
        
        out = self.fc(dec_out)
        
        return out
