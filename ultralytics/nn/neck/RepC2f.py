import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = ['Rep_Conv', 'Rep_C2f']

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else[d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    default_act = nn.SiLU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Rep_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, act=nn.SiLU()):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.deploy = False # Cờ trạng thái
        self.act = act if act else nn.Identity()

        if self.deploy:
            self.rbr_reparam = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=True)
        else:
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0, groups=groups, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            self.rbr_identity = nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None

    def forward(self, x):
        if hasattr(self, 'rbr_reparam'):
            return self.act(self.rbr_reparam(x))
        
        id_out = 0 if self.rbr_identity is None else self.rbr_identity(x)
        return self.act(self.rbr_dense(x) + self.rbr_1x1(x) + id_out)

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None: return 0
        return torch.nn.functional.pad(kernel1x1,[1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None: return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel, bn = branch[0].weight, branch[1]
        else:
            bn = branch
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(bn.weight.device)
            kernel = self.id_tensor

        running_mean, running_var = bn.running_mean, bn.running_var
        gamma, beta, eps = bn.weight, bn.bias, bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'): return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(self.in_channels, self.out_channels, 3, 1, 1, groups=self.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'): self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'): self.__delattr__('id_tensor')
        self.deploy = True


class RepBottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        self.cv1 = Rep_Conv(c1, c_, kernel_size=3, stride=1, padding=1, act=nn.SiLU())
        self.cv2 = Rep_Conv(c_, c2, kernel_size=3, stride=1, padding=1, act=nn.SiLU())
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Rep_C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(RepBottleneck(self.c, self.c, shortcut, e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))