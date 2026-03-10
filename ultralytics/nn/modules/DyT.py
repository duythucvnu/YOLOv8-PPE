import torch
import torch.nn as nn
 
__all__ = ['DynamicTanh']

class DynamicTanh(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.5)
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1) * 1.0)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        return self.gamma * torch.tanh(self.alpha * x) + self.beta