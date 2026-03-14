import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import SqueezeExcite

__all__ = ['repvit_m0_91', 'repvit_m1_01', 'repvit_m1_11', 'repvit_m1_51', 'repvit_m2_31']

class DynamicTanh(nn.Module):
    def __init__(self, channels, gamma_init=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.5)
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1) * gamma_init)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        return self.gamma * torch.tanh(self.alpha * x) + self.beta


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Conv2d_DyT(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, dyt_gamma_init=1.0, resolution=-10000):
        super().__init__()
        self.in_channels = a
        self.add_module('c', nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('dyt', DynamicTanh(b, gamma_init=dyt_gamma_init))

    @torch.no_grad()
    def fuse_self(self):
        print("Warning: DynamicTanh cannot be mathematically fused with Conv2d. Returning original module.")
        return self


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

    @torch.no_grad()
    def fuse_self(self):
        return self


class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_DyT(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = torch.nn.Conv2d(ed, ed, 1, 1, 0, groups=ed, bias=False)
        self.dim = ed
        self.dyt = DynamicTanh(ed)

    def forward(self, x):
        return self.dyt((self.conv(x) + self.conv1(x)) + x)

    @torch.no_grad()
    def fuse_self(self):
        print("Warning: RepVGGDW with DynamicTanh cannot be fused. Returning original module.")
        return self


class RepViTBlock(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(RepViTBlock, self).__init__()
        assert stride in [1, 2]
        self.identity = stride == 1 and inp == oup
        assert(hidden_dim == 2 * inp)

        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_DyT(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_DyT(inp, oup, ks=1, stride=1, pad=0)
            )
            self.channel_mixer = Residual(nn.Sequential(
                Conv2d_DyT(oup, 2 * oup, 1, 1, 0),
                nn.GELU() if use_hs else nn.GELU(),
                Conv2d_DyT(2 * oup, oup, 1, 1, 0, dyt_gamma_init=0.0),
            ))
        else:
            assert(self.identity)
            self.token_mixer = nn.Sequential(
                RepVGGDW(inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(nn.Sequential(
                Conv2d_DyT(inp, hidden_dim, 1, 1, 0),
                nn.GELU() if use_hs else nn.GELU(),
                Conv2d_DyT(hidden_dim, oup, 1, 1, 0, dyt_gamma_init=0.0),
            ))

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))


class RepViT(nn.Module):
    def __init__(self, cfgs):
        super(RepViT, self).__init__()
        self.cfgs = cfgs

        input_channel = self.cfgs[0][2]
        self.patch_embed = torch.nn.Sequential(
            Conv2d_DyT(3, input_channel // 2, 3, 2, 1), torch.nn.GELU(),
            Conv2d_DyT(input_channel // 2, input_channel, 3, 2, 1)
        )

        layers = [self.patch_embed]
        block = RepViTBlock

        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel

        self.features = nn.ModuleList(layers)
        self.channel = [i.size(1) for i in self.forward(torch.randn(1, 3, 640, 640))]

    def forward(self, x):
        input_size = x.size(2)
        scale = [4, 8, 16, 32]
        features = [None, None, None, None]

        for f in self.features:
            x = f(x)
            if input_size // x.size(2) in scale:
                features[scale.index(input_size // x.size(2))] = x

        return features

    def switch_to_deploy(self):
        pass


def update_weight(model_dict, weight_dict):
    idx, temp_dict = 0, {}

    for k, v in weight_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            idx += 1

    model_dict.update(temp_dict)

    print(f'Loading weights... {idx}/{len(model_dict)} items.')
    print(f'WARNING: Missed {len(model_dict) - idx} items. This is EXPECTED because BatchNorm weights cannot map to DynamicTanh.')

    return model_dict


def repvit_m1_5(weights=''):
    cfgs = [
        [3,2,64,0,0,1],[3,2,64,0,0,1],[3,2,128,0,0,2],[3,2,128,1,0,1],
        [3,2,128,0,0,1],[3,2,128,1,0,1],[3,2,128,0,0,1],[3,2,128,0,0,1],
        [3,2,256,0,1,2],[3,2,256,1,1,1],[3,2,256,0,1,1],[3,2,256,1,1,1],
        [3,2,256,0,1,1],[3,2,256,1,1,1],[3,2,256,0,1,1],[3,2,256,1,1,1],
        [3,2,256,0,1,1],[3,2,256,1,1,1],[3,2,256,0,1,1],[3,2,256,1,1,1],
        [3,2,256,0,1,1],[3,2,256,1,1,1],[3,2,256,0,1,1],[3,2,256,1,1,1],
        [3,2,256,0,1,1],[3,2,256,1,1,1],[3,2,256,0,1,1],[3,2,256,1,1,1],
        [3,2,256,0,1,1],[3,2,256,1,1,1],[3,2,256,0,1,1],[3,2,256,1,1,1],
        [3,2,256,0,1,1],[3,2,256,0,1,1],[3,2,512,0,1,2],[3,2,512,1,1,1],
        [3,2,512,0,1,1],[3,2,512,1,1,1],[3,2,512,0,1,1]
    ]

    model = RepViT(cfgs)

    if weights:
        model.load_state_dict(
            update_weight(
                model.state_dict(),
                torch.load(weights, map_location='cpu')['model']
            ),
            strict=False
        )

    return model


if __name__ == '__main__':
    model = repvit_m1_5()
    inputs = torch.randn((1, 3, 640, 640))
    res = model(inputs)

    print("Output shapes from RepViT-DyT:")
    for i in res:
        if i is not None:
            print(i.size())