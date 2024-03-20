import torch
from torch import nn
from torch.nn import functional as F

class BitLinear158(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(BitLinear158, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None
        self.eps = 1e-5

    def forward(self, x):
        Wq = self.quantize_weights(self.weight)
        return F.linear(x, Wq, self.bias)

    def quantize_weights(self, W):
        gamma = torch.mean(torch.abs(W))
        Wq = torch.maximum(-1, torch.minimum(1, torch.round(W / (gamma + self.eps))))
        return Wq