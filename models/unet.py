from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, ch)
        self.q = nn.Conv2d(ch, ch, 1)
        self.k = nn.Conv2d(ch, ch, 1)
        self.v = nn.Conv2d(ch, ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        h = self.norm(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        b, c, f, t = q.shape
        q = q.reshape(b, c, f*t).permute(0, 2, 1)
        k = k.reshape(b, c, f*t)
        attn = torch.bmm(q, k) * (c ** -0.5)
        attn = F.softmax(attn, dim=2)
        v = v.reshape(b, c, f*t)
        out = torch.bmm(v, attn.permute(0, 2, 1)).reshape(b, c, f, t)
        return x + self.proj(out)


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(
        -torch.arange(half, device=device, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / (half - 1))
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


class DiffusionUNet(nn.Module):
    def __init__(self, in_channels: int = 2, out_channels: int = 1, base: int = 64, channels_mult: List[int] = [1, 2, 4], time_emb_dim: int = 128):
        super().__init__()
        chs = [base * m for m in channels_mult]
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, base * 4), nn.SiLU(), nn.Linear(base * 4, base * 4)
        )

        # Encoder
        self.enc0 = ResidualBlock(in_channels, chs[0], base * 4)
        self.down1 = nn.Conv2d(chs[0], chs[0], 3, stride=2, padding=1)
        self.enc1 = ResidualBlock(chs[0], chs[1], base * 4)
        self.down2 = nn.Conv2d(chs[1], chs[1], 3, stride=2, padding=1)
        self.enc2 = ResidualBlock(chs[1], chs[2], base * 4)
        self.attn2 = AttentionBlock(chs[2])

        # Decoder
        self.up1 = nn.ConvTranspose2d(chs[2], chs[1], 4, stride=2, padding=1)
        self.dec1 = ResidualBlock(chs[1], chs[1], base * 4)
        self.up0 = nn.ConvTranspose2d(chs[1], chs[0], 4, stride=2, padding=1)
        self.dec0 = ResidualBlock(chs[0], chs[0], base * 4)

        self.out = nn.Conv2d(chs[0], out_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(sinusoidal_embedding(t, self.time_mlp[0].in_features))

        e0 = self.enc0(x, t_emb)
        d1 = self.down1(e0)
        e1 = self.enc1(d1, t_emb)
        d2 = self.down2(e1)
        e2 = self.enc2(d2, t_emb)
        e2 = self.attn2(e2)

        u1 = self.up1(e2)
        u1 = self.dec1(u1, t_emb)
        u0 = self.up0(u1)
        u0 = self.dec0(u0, t_emb)
        return self.out(u0)
