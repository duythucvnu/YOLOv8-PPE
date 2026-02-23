import torch
import torch.nn as nn

__all__ = ['MLLAttention']

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


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, dropout=0, norm=nn.BatchNorm2d, act_func=nn.ReLU):
        super(ConvLayer, self).__init__()
        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            groups=groups,
            bias=bias,
        )
        self.norm = norm(num_features=out_channels) if norm else None
        self.act = act_func() if act_func else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class RoPE(torch.nn.Module):
    def __init__(self, base=10000):
        super(RoPE, self).__init__()
        self.base = base

    def generate_rotations(self, x):
        feature_dim = x.shape[-1]
        channel_dims = x.shape[1:-1] 
        
        k_max = feature_dim // (2 * len(channel_dims))
        assert feature_dim % k_max == 0, "Feature dimension must be divisible"

        theta_ks = 1 / (self.base ** (torch.arange(k_max, dtype=torch.float32, device=x.device) / k_max))
        
        grids = [torch.arange(d, dtype=torch.float32, device=x.device) for d in channel_dims]
        meshes = torch.meshgrid(grids, indexing='ij')
        
        angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in meshes], dim=-1)

        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        
        rotations = torch.cat([rotations_re, rotations_im], dim=-1)
        return rotations

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        orig_dtype = x.dtype
        x = x.float()
        
        rotations = self.generate_rotations(x)
        
        x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        rot_complex = torch.view_as_complex(rotations)
        
        pe_x = rot_complex * x_complex
        
        out = torch.view_as_real(pe_x).flatten(-2)
        
        return out.to(orig_dtype)


class MLLAttention(nn.Module):
    def __init__(self, dim=3, input_resolution=[160, 160], num_heads=4, qkv_bias=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.rope = RoPE()

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        
        x_flat = x.flatten(2).transpose(1, 2).contiguous() 
        
        qk = self.qk(x_flat).reshape(B, N, 2, C).permute(2, 0, 1, 3).contiguous()
        q, k, v = qk[0], qk[1], x_flat

        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0
        
        q_reshaped = q.view(B, H, W, C)
        k_reshaped = k.view(B, H, W, C)
        
        head_dim = C // self.num_heads
        
        q_rope = self.rope(q_reshaped).view(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
        k_rope = self.rope(k_reshaped).view(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
        
        v = v.view(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
        q = q.view(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.view(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        
        kv = (k_rope.transpose(-2, -1) * (N ** -0.5)) @ (v * (N ** -0.5))
        
        x_out = q_rope @ kv * z

        x_out = x_out.transpose(1, 2).reshape(B, N, C).contiguous()
        
        v_img = v.transpose(1, 2).reshape(B, C, H, W).contiguous()
        
        lepe_out = self.lepe(v_img).flatten(2).transpose(1, 2)
        x_out = x_out + lepe_out
        
        x_out = x_out.transpose(1, 2).reshape(B, C, H, W).contiguous()
        
        return x_out

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'


if __name__ == "__main__":
    image_size = (1, 64, 160, 160)
    image = torch.rand(*image_size)
    model = MLLAttention(64)
    out = model(image)
    print(out.size())