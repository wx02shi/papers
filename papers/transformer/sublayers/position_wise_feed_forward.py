import torch
import torch.nn as nn

class PositionWiseFeedForward(nn.Module):
        
    def __init__(self, d_embedding, d_ff):
        
        super().__init__()
        
        self.fc1 = nn.Linear(d_embedding, d_ff)
        self.fc2 = nn.Linear(d_ff, d_embedding)
        self.relu = nn.ReLU()


    def forward(self, x):
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x