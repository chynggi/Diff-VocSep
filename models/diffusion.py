from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class DiffusionConfig:
    timesteps: int = 200
    beta_start: float = 1e-4
    beta_end: float = 2e-2


class GaussianDiffusion(nn.Module):
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.timesteps = config.timesteps
        betas = torch.linspace(config.beta_start, config.beta_end, config.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_ac = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_ac * x_start + sqrt_om * noise

    def p_mean_variance(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Model predicts noise (epsilon) on current x
        eps = model(x, t)
        beta_t = self.betas[t].view(-1, 1, 1, 1)
        alpha_t = (1.0 - beta_t)
        alpha_bar_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        # DDPM mean prediction
        mean = (1.0 / torch.sqrt(alpha_t)) * (x - beta_t / torch.sqrt(1.0 - alpha_bar_t) * eps)
        var = beta_t
        return mean, var

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        mean, var = self.p_mean_variance(model, x, t)
        if (t == 0).all():
            return mean
        noise = torch.randn_like(x)
        return mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(self, model: nn.Module, shape, cond: torch.Tensor = None) -> torch.Tensor:
        # model should accept concat([x, cond]) as input channels when cond is provided
        x = torch.randn(shape, device=cond.device if cond is not None else None)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=x.device, dtype=torch.long)
            x_in = torch.cat([x, cond], dim=1) if cond is not None else x
            x = self.p_sample(model, x_in, t)
        return x
