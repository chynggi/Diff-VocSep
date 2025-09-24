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
        # Memory helpers
        freq_chunk: Optional[int] = None,  # for axis='time': process frequencies in chunks to cap memory
        time_chunk: Optional[int] = None,  # for axis='freq': process time in chunks to cap memory
        use_checkpoint: bool = False,      # gradient checkpointing to reduce memory at extra compute cost
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.d_model = d_model
        self.axis = (axis or "time").lower()
        self.freq_chunk = freq_chunk
        self.time_chunk = time_chunk
        self.use_checkpoint = use_checkpoint
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
            # process along time with optional frequency chunking to reduce memory
            t_emb = self.time_mlp(sinusoidal_embedding(t, self.time_mlp[0].in_features))  # (B, D)
            chunk = self.freq_chunk if (self.freq_chunk and self.freq_chunk > 0) else F
            y_out = x.new_zeros((B, self.out_channels, F, Tlen))

            def run_blocks(h: torch.Tensor) -> torch.Tensor:
                # h: (B*Fc, T, D)
                if self.use_checkpoint and self.training:
                    for blk in self.blocks:
                        h = torch.utils.checkpoint.checkpoint(blk, h)
                else:
                    for blk in self.blocks:
                        h = blk(h)
                return h

            for f0 in range(0, F, chunk):
                f1 = min(F, f0 + chunk)
                Fc = f1 - f0
                # slice and reshape: (B, D, Fc, T) -> (B*Fc, T, D)
                h = h2d[:, :, f0:f1, :].permute(0, 2, 3, 1).contiguous().view(B * Fc, Tlen, self.d_model)
                # add time embedding broadcast over frequencies in the chunk
                t_add = t_emb[:, None, :].repeat(1, Fc, 1).view(B * Fc, 1, self.d_model)
                h = h + t_add
                h = run_blocks(h)
                y = self.out_linear(h).view(B, Fc, Tlen, self.out_channels).permute(0, 3, 1, 2).contiguous()
                y_out[:, :, f0:f1, :] = y
            return y_out

        def run_freq(h2d: torch.Tensor) -> torch.Tensor:
            # process along frequency with optional time chunking (safe)
            t_emb = self.time_mlp(sinusoidal_embedding(t, self.time_mlp[0].in_features))  # (B, D)
            chunk = self.time_chunk if (self.time_chunk and self.time_chunk > 0) else Tlen
            y_out = x.new_zeros((B, self.out_channels, F, Tlen))

            def run_blocks(h: torch.Tensor) -> torch.Tensor:
                # h: (B*Tc, F, D)
                if self.use_checkpoint and self.training:
                    for blk in self.blocks:
                        h = torch.utils.checkpoint.checkpoint(blk, h)
                else:
                    for blk in self.blocks:
                        h = blk(h)
                return h

            for t0 in range(0, Tlen, chunk):
                t1 = min(Tlen, t0 + chunk)
                Tc = t1 - t0
                # slice and reshape: (B, D, F, Tc) -> (B*Tc, F, D)
                h = h2d[:, :, :, t0:t1].permute(0, 3, 2, 1).contiguous().view(B * Tc, F, self.d_model)
                # add time embedding (tile across chunked time)
                t_add = t_emb[:, None, :].view(B, 1, self.d_model).repeat_interleave(Tc, dim=1).view(B * Tc, 1, self.d_model)
                h = h + t_add
                h = run_blocks(h)
                y = self.out_linear(h).view(B, Tc, F, self.out_channels).permute(0, 3, 2, 1).contiguous()
                y_out[:, :, :, t0:t1] = y
            return y_out

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
                if self.use_checkpoint and self.training:
                    ht = torch.utils.checkpoint.checkpoint(blk, ht)
                else:
                    ht = blk(ht)
                h = ht.view(B, F, Tlen, self.d_model).permute(0, 3, 1, 2).contiguous()
            else:
                # freq axis
                hf = h.permute(0, 3, 2, 1).contiguous().view(B * Tlen, F, self.d_model)
                t_add = t_emb[:, None, :].repeat(1, 1, 1).view(B, 1, self.d_model)
                t_add = t_add.repeat_interleave(Tlen, dim=0)
                hf = hf + t_add
                if self.use_checkpoint and self.training:
                    hf = torch.utils.checkpoint.checkpoint(blk, hf)
                else:
                    hf = blk(hf)
                h = hf.view(B, Tlen, F, self.d_model).permute(0, 3, 2, 1).contiguous()

        y = self.out_linear(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        return y
