import torch
from torch import nn
from torch.nn import functional as F

def clip(x, a, b):
    return torch.maximum(a, torch.minimum(b, x))

class BitLinear158(nn.Module):
    def __init__(self, in_features, out_features, bias=True, b=8):
        super(BitLinear158, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None
        self.eps = 1e-5
        self.Q_b = 2 ** (b-1)

    def forward(self, x):
        Wq = self.quantize_weights(self.weight)
        return F.linear(x, Wq, self.bias)

    def quantize_weights(self, W):
        gamma = torch.mean(torch.abs(W))
        Wscale = torch.round(W / (gamma + self.eps))
        Wq = clip(Wscale, -1, 1)
        return Wq

    def quantize_activations(self, x):
        gamma = torch.norm(x, float('inf'))
        xscale = (x - torch.min(x)) * self.Q_b / gamma
        xq = clip(xscale, -self.Q_b + self.eps, self.Q_b - self.eps)
        return xq