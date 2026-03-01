import torch
import torch.nn as nn

__all__ = ['C2fMLLABlock']

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def autopad(k, p=None, d=1): 
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
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

class RoPE(torch.nn.Module):
    def __init__(self, base=10000):
        super(RoPE, self).__init__()
        self.base = base

    def generate_rotations(self, x):
        feature_dim = x.shape[-1]
        channel_dims = x.shape[1:-1]
        k_max = feature_dim // (2 * len(channel_dims))
        theta_ks = 1 / (self.base ** (torch.arange(k_max, dtype=torch.float32, device=x.device) / k_max))
        grids = [torch.arange(d, dtype=torch.float32, device=x.device) for d in channel_dims]
        meshes = torch.meshgrid(grids, indexing='ij')
        angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in meshes], dim=-1)
        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1)
        return rotations

    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, x):
        orig_dtype = x.dtype
        x = x.float()
        rotations = self.generate_rotations(x)
        x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2).contiguous())
        rot_complex = torch.view_as_complex(rotations.contiguous())
        pe_x = rot_complex * x_complex
        out = torch.view_as_real(pe_x).flatten(-2)
        return out.to(orig_dtype)

class LinearAttention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.rope = RoPE()

    def forward(self, x):
        b, n, c = x.shape
        h_img = int(n ** 0.5)
        w_img = int(n ** 0.5)
        num_heads = self.num_heads
        head_dim = c // num_heads

        qk = self.qk(x).reshape(b, n, 2, c).permute(2, 0, 1, 3).contiguous()
        q, k, v = qk[0], qk[1], x

        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0

        q_img = q.view(b, h_img, w_img, c)
        k_img = k.view(b, h_img, w_img, c)

        q_rope = self.rope(q_img).view(b, n, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        k_rope = self.rope(k_img).view(b, n, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()

        q = q.view(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.view(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.view(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k_rope.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
        x_out = q_rope @ kv * z

        x_out = x_out.transpose(1, 2).reshape(b, n, c).contiguous()

        v_img = v.transpose(1, 2).reshape(b, c, h_img, w_img).contiguous()
        lepe_out = self.lepe(v_img).flatten(2).transpose(1, 2)

        return x_out + lepe_out

class MLLABlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.in_proj = nn.Linear(dim, dim)
        self.act_proj = nn.Linear(dim, dim)
        self.dwc = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.SiLU()
        self.attn = LinearAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x + self.cpe1(x)

        shortcut = x.flatten(2).transpose(1, 2).contiguous()
        x_flat = shortcut

        x_flat = self.norm1(x_flat)
        act_res = self.act(self.act_proj(x_flat))
        x_flat = self.in_proj(x_flat)

        x_4d = x_flat.transpose(1, 2).reshape(B, C, H, W).contiguous()
        x_4d = self.act(self.dwc(x_4d))

        x_flat = x_4d.flatten(2).transpose(1, 2).contiguous()
        x_flat = self.attn(x_flat)
        x_flat = self.out_proj(x_flat * act_res)

        x_flat = shortcut + self.drop_path(x_flat)

        x_4d = x_flat.transpose(1, 2).reshape(B, C, H, W).contiguous()
        x_4d = x_4d + self.cpe2(x_4d)
        x_flat = x_4d.flatten(2).transpose(1, 2).contiguous()

        x_flat = x_flat + self.drop_path(self.mlp(self.norm2(x_flat)))

        return x_flat.transpose(1, 2).reshape(B, C, H, W).contiguous()

class C2fMLLABlock(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(MLLABlock(self.c) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

if __name__ == "__main__":
    image_size = (1, 64, 64, 64)
    image = torch.rand(*image_size).cuda()
    model = C2fMLLABlock(64, 64, 1).cuda()
    out = model(image)
    print("Output shape:", out.size())