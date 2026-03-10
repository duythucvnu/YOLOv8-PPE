import torch
import torch.nn as nn
 
__all__ = ['DyT']

class DyT(nn.Module):
    # input x has the shape of[B,T，C]
 
    # B:batch size, T:tokens, C:dimension
 
    def __init__(self, C, init_alpha):
        super(DyT, self).__init__()
 
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        self.beta = nn.Parameter(torch.ones(C))
        self.gamma = nn.Parameter(torch.zeros(C))
        self.act = nn.Tanh()
 
    def forward(self, x):
 
        b, c, h, w = x.shape
        x = x.reshape(b, c, -1).permute(0, 2, 1)
 
        x = self.act(x * self.alpha)
        x = self.beta * x + self.gamma
 
        return x.permute(0, 2, 1).reshape(b, c, h, w)
    
 
if __name__ == '__main__':
    x = torch.randn(1, 100, 512)
    model = DyT(512, 0.1)
    y = model(x)
    print(y.shape)