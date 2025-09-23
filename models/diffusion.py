from dataclasses import dataclass
import torch
import torch.nn as nn
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


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

    def p_mean_variance(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        # Model predicts noise (epsilon) on current x; if cond is provided, pass concat([x, cond]) into the model
        x_in = torch.cat([x, cond], dim=1) if cond is not None else x
        eps = model(x_in, t)
        beta_t = self.betas[t].view(-1, 1, 1, 1)
        alpha_t = (1.0 - beta_t)
        alpha_bar_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        # DDPM mean prediction
        mean = (1.0 / torch.sqrt(alpha_t)) * (x - beta_t / torch.sqrt(1.0 - alpha_bar_t) * eps)
        var = beta_t
        return mean, var

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        mean, var = self.p_mean_variance(model, x, t, cond=cond)
        if (t == 0).all():
            return mean
        noise = torch.randn_like(x)
        return mean + torch.sqrt(var) * noise

    def _get_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        return self.alphas_cumprod[t].view(-1, 1, 1, 1)

    @torch.no_grad()
    def sample_ddim(
        self,
        model: nn.Module,
        shape,
        cond: torch.Tensor,
        steps: int = 50,
        eta: float = 0.0,
        progress: bool = False,
    ) -> torch.Tensor:
        """DDIM sampling with optional stochasticity via eta.

        Args:
            model: noise-prediction model taking (x_in, t)
            shape: output tensor shape (B, C, F, T)
            cond: conditioning spectrogram (B, 1, F, T)
            steps: number of DDIM steps (<= self.timesteps)
            eta: 0.0 for deterministic DDIM, >0 to add noise
        """
        device = cond.device
        x = torch.randn(shape, device=device)

        # Create an evenly spaced timestep schedule
        if steps >= self.timesteps:
            schedule = list(range(self.timesteps))
        else:
            # include last index (T-1)
            schedule = torch.linspace(0, self.timesteps - 1, steps, device=device).long().tolist()
        # iterate backwards
        iterator = reversed(range(len(schedule)))
        if progress and tqdm is not None:
            iterator = tqdm(iterator, total=len(schedule), desc="DDIM", unit="step")
        for i in iterator:
            t_i = schedule[i]
            t = torch.full((shape[0],), t_i, device=device, dtype=torch.long)
            eps = model(torch.cat([x, cond], dim=1), t)

            alpha_bar_t = self._get_alpha_bar(t)
            sqrt_ab_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_ab_t = torch.sqrt(1.0 - alpha_bar_t)
            # predict x0 from current x and eps
            x0_pred = (x - sqrt_one_minus_ab_t * eps) / (sqrt_ab_t + 1e-8)

            if i == 0:
                # final step maps to t=-1 => alpha_bar_prev=1
                alpha_bar_prev = torch.ones_like(alpha_bar_t)
            else:
                t_prev_scalar = schedule[i - 1]
                t_prev = torch.full((shape[0],), t_prev_scalar, device=device, dtype=torch.long)
                alpha_bar_prev = self._get_alpha_bar(t_prev)

            # DDIM sigma
            sigma_t = 0.0
            if eta > 0.0:
                sigma_t = eta * torch.sqrt(
                    (1 - alpha_bar_prev) / (1 - alpha_bar_t + 1e-8)
                    * (1 - alpha_bar_t / (alpha_bar_prev + 1e-8))
                )
            # direction pointing to x_t
            dir_xt = torch.sqrt(torch.clamp(1 - alpha_bar_prev - (sigma_t ** 2), min=0.0)) * eps
            noise = torch.randn_like(x) if eta > 0.0 and i > 0 else 0.0
            x = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt + (sigma_t * noise if isinstance(noise, torch.Tensor) else 0.0)

        return x

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape,
        cond: torch.Tensor = None,
        use_ddim: bool = False,
        ddim_steps: int = 50,
        eta: float = 0.0,
        progress: bool = False,
    ) -> torch.Tensor:
        # model should accept concat([x, cond]) as input channels when cond is provided
        if cond is None:
            device = None
        else:
            device = cond.device

        if use_ddim and cond is not None:
            return self.sample_ddim(model, shape, cond=cond, steps=ddim_steps, eta=eta, progress=progress)

        x = torch.randn(shape, device=device)
        iterator = reversed(range(self.timesteps))
        if progress and tqdm is not None:
            iterator = tqdm(iterator, total=self.timesteps, desc="DDPM", unit="step")
        for i in iterator:
            t = torch.full((shape[0],), i, device=x.device, dtype=torch.long)
            x = self.p_sample(model, x, t, cond=cond)
        return x
