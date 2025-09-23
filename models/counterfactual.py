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
        return self.unet(x_noisy_and_cond, t)

    @torch.no_grad()
    def generate_instrumental(self, mixture_spec: torch.Tensor) -> torch.Tensor:
        # mixture_spec: (B, 1, F, T) normalized magnitude
        b = mixture_spec.size(0)
        x_shape = (b, 1, mixture_spec.size(2), mixture_spec.size(3))
        # diffusion.sample will concat [x, cond] internally in our usage below
        instrumental = self.diffusion.sample(self, x_shape, cond=mixture_spec)
        return instrumental
