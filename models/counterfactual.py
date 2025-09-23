import torch
import torch.nn as nn

from .unet import DiffusionUNet
from .diffusion import GaussianDiffusion, DiffusionConfig


class CounterfactualDiffusion(nn.Module):
    def __init__(self, in_channels: int = 2, out_channels: int = 1, base: int = 64, channels_mult = [1,2,4], timesteps: int = 200, beta_start: float = 1e-4, beta_end: float = 2e-2):
        super().__init__()
        self.unet = DiffusionUNet(in_channels=in_channels, out_channels=out_channels, base=base, channels_mult=channels_mult)
        self.diffusion = GaussianDiffusion(DiffusionConfig(timesteps=timesteps, beta_start=beta_start, beta_end=beta_end))

    def forward(self, x_noisy_and_cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # UNet downsamples twice by stride=2 → require spatial dims divisible by 4.
        # Pad to nearest multiple of 4, run UNet, then crop back to original (F, T).
        B, C, F, T = x_noisy_and_cond.shape
        pad_f = (4 - (F % 4)) % 4
        pad_t = (4 - (T % 4)) % 4

        if pad_f != 0 or pad_t != 0:
            x_padded = torch.nn.functional.pad(
                x_noisy_and_cond,
                pad=(0, pad_t, 0, pad_f),  # (left, right, top, bottom) for last two dims (T, F)
                mode="reflect",
            )
        else:
            x_padded = x_noisy_and_cond

        y = self.unet(x_padded, t)
        # crop back to original size
        y = y[:, :, :F, :T]
        return y

    @torch.no_grad()
    def generate_instrumental(
        self,
        mixture_spec: torch.Tensor,
        use_ddim: bool = False,
        ddim_steps: int = 50,
        eta: float = 0.0,
        progress: bool = False,
    ) -> torch.Tensor:
        # mixture_spec: (B, 1, F, T) normalized magnitude
        b = mixture_spec.size(0)
        x_shape = (b, 1, mixture_spec.size(2), mixture_spec.size(3))
        # diffusion.sample will concat [x, cond] internally in our usage below
        instrumental = self.diffusion.sample(
            self,
            x_shape,
            cond=mixture_spec,
            use_ddim=use_ddim,
            ddim_steps=ddim_steps,
            eta=eta,
            progress=progress,
        )
        return instrumental
