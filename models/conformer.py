from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import sinusoidal_embedding


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ConformerFF(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + 0.5 * self.net(x)


class ConformerMHSA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        h = self.ln(x)
        out, _ = self.attn(h, h, h, need_weights=False)
        return x + self.do(out)


class ConformerConvModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.pw1 = nn.Linear(d_model, 2 * d_model)
        self.act = nn.GLU(dim=-1)
        # depthwise conv over time implemented via Conv1d on (B, C, T)
        self.dw = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2, groups=d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.swish = Swish()
        self.pw2 = nn.Linear(d_model, d_model)
        self.do = nn.Dropout(dropout)

    def forward(self, x):  # x: (B, T, D)
        h = self.ln(x)
        h = self.pw1(h)
        h = self.act(h)
        # to (B, D, T) for depthwise conv
        h = h.transpose(1, 2)  # (B, D, T)
        h = self.dw(h)
        h = self.bn(h)
        h = self.swish(h)
        h = h.transpose(1, 2)  # (B, T, D)
        h = self.pw2(h)
        return x + self.do(h)


class ConformerBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int, kernel_size: int = 31, dropout: float = 0.0):
        super().__init__()
        self.ff1 = ConformerFF(d_model, d_ff, dropout)
        self.mhsa = ConformerMHSA(d_model, n_heads, dropout)
        self.conv = ConformerConvModule(d_model, kernel_size, dropout)
        self.ff2 = ConformerFF(d_model, d_ff, dropout)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.ff1(x)
        x = self.mhsa(x)
        x = self.conv(x)
        x = self.ff2(x)
        return self.ln(x)


class ConformerDiffusion(nn.Module):
    """Conformer-based epsilon predictor operating per-frequency across time.

    Input x: (B, Cin, F, T). We embed channels to d_model via 1x1 conv,
    then reshape to (B*F, T, d_model) and run N Conformer blocks.
    Finally project to Cout and reshape back to (B, Cout, F, T).
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        num_layers: int = 6,
        kernel_size: int = 31,
        dropout: float = 0.0,
        axis: str = "time",  # 'time' | 'freq' | 'mixed'
        time_emb_dim: int = 128,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.d_model = d_model
        self.axis = (axis or "time").lower()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.in_proj = nn.Conv2d(in_channels, d_model, kernel_size=1)
        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, d_ff, n_heads, kernel_size, dropout) for _ in range(num_layers)
        ])
        self.out_linear = nn.Linear(d_model, out_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (B, Cin, F, T)
        B, _, F, Tlen = x.shape
        h2d = self.in_proj(x)  # (B, D, F, T)

        def run_time(h2d: torch.Tensor) -> torch.Tensor:
            # (B, D, F, T) -> (B*F, T, D)
            h = h2d.permute(0, 2, 3, 1).contiguous().view(B * F, Tlen, self.d_model)
            t_emb = self.time_mlp(sinusoidal_embedding(t, self.time_mlp[0].in_features))  # (B, D)
            t_add = t_emb[:, None, :].repeat(1, F, 1).view(B * F, 1, self.d_model)
            h = h + t_add
            for blk in self.blocks:
                h = blk(h)
            # project and back to (B, Cout, F, T)
            y = self.out_linear(h).view(B, F, Tlen, self.out_channels).permute(0, 3, 1, 2).contiguous()
            return y

        def run_freq(h2d: torch.Tensor) -> torch.Tensor:
            # (B, D, F, T) -> (B*T, F, D)
            h = h2d.permute(0, 3, 2, 1).contiguous().view(B * Tlen, F, self.d_model)
            t_emb = self.time_mlp(sinusoidal_embedding(t, self.time_mlp[0].in_features))  # (B, D)
            # broadcast time embedding across F, then across T positions implicitly by tiling B times
            t_add = t_emb[:, None, :].repeat(1, 1, 1).view(B, 1, self.d_model)
            t_add = t_add.repeat_interleave(Tlen, dim=0)  # (B*T, 1, D)
            h = h + t_add
            for blk in self.blocks:
                h = blk(h)
            # project and back to (B, Cout, F, T)
            y = self.out_linear(h).view(B, Tlen, F, self.out_channels).permute(0, 3, 2, 1).contiguous()
            return y

        if self.axis == "time":
            return run_time(h2d)
        if self.axis == "freq":
            return run_freq(h2d)

        # mixed: alternate axes per block
        # We'll iteratively update h2d; reuse blocks sequentially with axis flip
        h = h2d
        # time embedding prepared once
        t_emb = self.time_mlp(sinusoidal_embedding(t, self.time_mlp[0].in_features))  # (B, D)
        for i, blk in enumerate(self.blocks):
            if i % 2 == 0:
                # time axis
                ht = h.permute(0, 2, 3, 1).contiguous().view(B * F, Tlen, self.d_model)
                t_add = t_emb[:, None, :].repeat(1, F, 1).view(B * F, 1, self.d_model)
                ht = ht + t_add
                ht = blk(ht)
                h = ht.view(B, F, Tlen, self.d_model).permute(0, 3, 1, 2).contiguous()
            else:
                # freq axis
                hf = h.permute(0, 3, 2, 1).contiguous().view(B * Tlen, F, self.d_model)
                t_add = t_emb[:, None, :].repeat(1, 1, 1).view(B, 1, self.d_model)
                t_add = t_add.repeat_interleave(Tlen, dim=0)
                hf = hf + t_add
                hf = blk(hf)
                h = hf.view(B, Tlen, F, self.d_model).permute(0, 3, 2, 1).contiguous()

        y = self.out_linear(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        return y
