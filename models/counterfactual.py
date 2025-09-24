import torch
import torch.nn as nn

from .unet import DiffusionUNet
from .conformer import ConformerDiffusion
from .diffusion import GaussianDiffusion, DiffusionConfig


class CounterfactualDiffusion(nn.Module):
    def __init__(self, in_channels: int = 2, out_channels: int = 1, base: int = 64, channels_mult = [1,2,4], timesteps: int = 200, beta_start: float = 1e-4, beta_end: float = 2e-2, model_type: str = "unet", model_kwargs: dict | None = None):
        super().__init__()
        model_type = (model_type or "unet").lower()
        model_kwargs = model_kwargs or {}
        if model_type == "unet":
            self.unet = DiffusionUNet(in_channels=in_channels, out_channels=out_channels, base=base, channels_mult=channels_mult)
        elif model_type in ("conformer", "naivev2", "naivev2diff"):
            self.unet = ConformerDiffusion(in_channels=in_channels, out_channels=out_channels, **model_kwargs)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
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
        # Shallow diffusion options
        shallow_init: torch.Tensor | None = None,
        k_step: int | None = None,
        add_forward_noise: bool = True,
        sampler: str | None = None,
    ) -> torch.Tensor:
        # mixture_spec: (B, 1, F, T) normalized magnitude
        b = mixture_spec.size(0)
        x_shape = (b, 1, mixture_spec.size(2), mixture_spec.size(3))
        # diffusion.sample will concat [x, cond] internally in our usage below
        # Simple sampler switch (reserved for future DPM/UniPC); current choices map to DDPM/DDIM
        use_ddim_flag = use_ddim or (isinstance(sampler, str) and sampler.lower() == "ddim")
        instrumental = self.diffusion.sample(
            self,
            x_shape,
            cond=mixture_spec,
            use_ddim=use_ddim_flag,
            ddim_steps=ddim_steps,
            eta=eta,
            progress=progress,
            shallow_init=shallow_init,
            k_step=k_step,
            add_forward_noise=add_forward_noise,
        )
        return instrumental
