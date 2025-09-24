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
    beta_schedule: str = "linear"  # 'linear' | 'cosine'
    ct_embed: str = "tau"  # 'tau' | 'logsnr'


class GaussianDiffusion(nn.Module):
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.timesteps = config.timesteps

        def make_betas(cfg: DiffusionConfig) -> torch.Tensor:
            schedule = (cfg.beta_schedule or "linear").lower()
            if schedule == "linear":
                return torch.linspace(cfg.beta_start, cfg.beta_end, cfg.timesteps)
            elif schedule == "cosine":
                # Cosine schedule following Nichol & Dhariwal (2021)
                # Construct alpha_bar via cosine, then derive betas
                s = 0.008
                steps = cfg.timesteps
                t = torch.linspace(0, steps, steps + 1, dtype=torch.float32)
                f = lambda u: torch.cos((u / steps + s) / (1 + s) * torch.pi * 0.5) ** 2
                alpha_bar = f(t) / f(torch.tensor(0.0))
                betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
                return betas.clamp(1e-8, 0.999)
            else:
                raise ValueError(f"Unknown beta_schedule: {schedule}")

        betas = make_betas(config)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # cache schedule type for CT samplers
        self._beta_schedule = getattr(config, "beta_schedule", "linear")
        self._ct_embed = getattr(config, "ct_embed", "tau")

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

    # ---------- Continuous-time utilities ----------
    def _alpha_bar_ct(self, tau: torch.Tensor) -> torch.Tensor:
        """Continuous-time alpha_bar as a function of tau in [0,1].

        Uses the same family as beta_schedule for consistency.
        """
        schedule = (self._beta_schedule or "linear").lower()
        # ensure shape (B,1,1,1)
        tau = tau.view(-1, 1, 1, 1).clamp(0.0, 1.0)
        if schedule == "cosine":
            s = 0.008
            v = (tau + s) / (1 + s)
            alpha_bar = torch.cos(v * torch.pi * 0.5) ** 2
            return alpha_bar
        # linear in discrete sense approximated by linear interpolation over indices
        # map tau to discrete index and fetch alpha_bar
        idx = (tau.view(-1) * (self.timesteps - 1)).round().long().clamp(0, self.timesteps - 1)
        return self.alphas_cumprod[idx].view(-1, 1, 1, 1)

    def _logsnr_from_ab(self, ab: torch.Tensor) -> torch.Tensor:
        # logsnr = log(ab / (1 - ab)) with clamp for stability
        ab = ab.clamp(1e-6, 1 - 1e-6)
        return torch.log(ab) - torch.log1p(-ab)

    def _time_embed_from_tau(self, tau: torch.Tensor) -> torch.Tensor:
        """Convert continuous tau to model time embedding scalar according to ct_embed.

        - tau: in [0,1], shape (B,)
        - returns shape (B,) floats suitable for sinusoidal embedding
        """
        if (self._ct_embed or "tau").lower() == "logsnr":
            ab = self._alpha_bar_ct(tau)
            logsnr = self._logsnr_from_ab(ab).view(-1)
            return logsnr
        # default: scaled tau to discrete range
        return tau * (self.timesteps - 1)

    @torch.no_grad()
    def q_sample_ct(self, x_start: torch.Tensor, tau: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        ab = self._alpha_bar_ct(tau)
        return torch.sqrt(ab) * x_start + torch.sqrt(1.0 - ab) * noise

    @torch.no_grad()
    def q_sample_at_t(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """Forward diffuse x_start to an arbitrary timestep t (per-batch)."""
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_ab = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_ab = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_ab * x_start + sqrt_one_minus_ab * noise

    @torch.no_grad()
    def sample_ddim(
        self,
        model: nn.Module,
        shape,
        cond: torch.Tensor,
        steps: int = 50,
        eta: float = 0.0,
        progress: bool = False,
        t_start: int | None = None,
        x_init: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """DDIM sampling with optional stochasticity via eta.

        Args:
            model: noise-prediction model taking (x_in, t)
            shape: output tensor shape (B, C, F, T)
            cond: conditioning spectrogram (B, 1, F, T)
            steps: number of DDIM steps (<= self.timesteps)
            eta: 0.0 for deterministic DDIM, >0 to add noise
            t_start: optional start timestep (inclusive) to begin reverse process from
            x_init: optional initial x at t_start (if provided, used as initial state)
        """
        device = cond.device
        if x_init is None:
            x = torch.randn(shape, device=device)
        else:
            x = x_init

        # Create an evenly spaced timestep schedule
        max_t = self.timesteps - 1 if t_start is None else int(t_start)
        max_t = max(0, min(max_t, self.timesteps - 1))
        if steps >= (max_t + 1):
            schedule = list(range(max_t + 1))
        else:
            # include last index (max_t)
            schedule = torch.linspace(0, max_t, steps, device=device).long().tolist()
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
    def sample_ddim_ct(
        self,
        model: nn.Module,
        shape,
        cond: torch.Tensor,
        steps: int = 50,
        progress: bool = False,
        t_start: float | None = None,
        x_init: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = cond.device
        x = x_init if x_init is not None else torch.randn(shape, device=device)

        tau_max = 1.0 if t_start is None else float(t_start)
        tau_max = float(max(0.0, min(1.0, tau_max)))
        # build a monotonically increasing list [0..tau_max], iterate backwards
        taus = torch.linspace(0.0, tau_max, steps, device=device)
        iterator = reversed(range(len(taus)))
        if progress and tqdm is not None:
            iterator = tqdm(iterator, total=len(taus), desc="DDIM-CT", unit="step")
        for i in iterator:
            tau_i = taus[i]
            tau = torch.full((shape[0],), float(tau_i.item()), device=device)
            ab_t = self._alpha_bar_ct(tau)
            sqrt_ab_t = torch.sqrt(ab_t)
            sqrt_one_minus_ab_t = torch.sqrt(1.0 - ab_t)
            # eps at tau_i
            # For time embedding, scale tau to discrete range for model
            t_embed = tau * (self.timesteps - 1)
            eps = model(torch.cat([x, cond], dim=1), t_embed)
            x0_pred = (x - sqrt_one_minus_ab_t * eps) / (sqrt_ab_t + 1e-8)

            if i == 0:
                ab_prev = torch.ones_like(ab_t)
            else:
                tau_prev = torch.full((shape[0],), float(taus[i - 1].item()), device=device)
                ab_prev = self._alpha_bar_ct(tau_prev)
            x = torch.sqrt(ab_prev) * x0_pred + torch.sqrt(1.0 - ab_prev) * eps
        return x

    @torch.no_grad()
    def sample_unipc(
        self,
        model: nn.Module,
        shape,
        cond: torch.Tensor,
        steps: int = 20,
        progress: bool = False,
        t_start: int | None = None,
        x_init: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Lightweight UniPC-style Predictor-Corrector using epsilon model.

        Uses two model evaluations per step: predictor to get x_t->x_{t-1}, then corrector recomputes eps at t-1 and blends.
        """
        device = cond.device
        x = x_init if x_init is not None else torch.randn(shape, device=device)

        max_t = self.timesteps - 1 if t_start is None else int(t_start)
        max_t = max(0, min(max_t, self.timesteps - 1))

        if steps >= (max_t + 1):
            schedule = list(range(max_t + 1))
        else:
            schedule = torch.linspace(0, max_t, steps, device=device).long().tolist()

        iterator = reversed(range(len(schedule)))
        if progress and tqdm is not None:
            iterator = tqdm(iterator, total=len(schedule), desc="UniPC", unit="step")

        for i in iterator:
            t_i = schedule[i]
            t = torch.full((shape[0],), t_i, device=device, dtype=torch.long)
            alpha_bar_t = self._get_alpha_bar(t)
            sqrt_ab_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_ab_t = torch.sqrt(1.0 - alpha_bar_t)

            # predictor using eps at t
            eps_t = model(torch.cat([x, cond], dim=1), t)
            x0_pred = (x - sqrt_one_minus_ab_t * eps_t) / (sqrt_ab_t + 1e-8)

            if i == 0:
                alpha_bar_prev = torch.ones_like(alpha_bar_t)
                x = torch.sqrt(alpha_bar_prev) * x0_pred
                break
            else:
                t_prev_scalar = schedule[i - 1]
                t_prev = torch.full((shape[0],), t_prev_scalar, device=device, dtype=torch.long)
                alpha_bar_prev = self._get_alpha_bar(t_prev)

            # predictor step to x_{t-1}
            x_pred = torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(1.0 - alpha_bar_prev) * eps_t

            # corrector: recompute eps at t-1 and blend (Heun-like)
            eps_prev = model(torch.cat([x_pred, cond], dim=1), t_prev)
            eps_corr = 0.5 * (eps_t + eps_prev)
            x = torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(1.0 - alpha_bar_prev) * eps_corr

        return x

    @torch.no_grad()
    def sample_unipc_ct(
        self,
        model: nn.Module,
        shape,
        cond: torch.Tensor,
        steps: int = 20,
        progress: bool = False,
        t_start: float | None = None,
        x_init: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = cond.device
        x = x_init if x_init is not None else torch.randn(shape, device=device)
        tau_max = 1.0 if t_start is None else float(t_start)
        tau_max = float(max(0.0, min(1.0, tau_max)))
        taus = torch.linspace(0.0, tau_max, steps, device=device)
        iterator = reversed(range(len(taus)))
        if progress and tqdm is not None:
            iterator = tqdm(iterator, total=len(taus), desc="UniPC-CT", unit="step")
        for i in iterator:
            tau_i = taus[i]
            tau = torch.full((shape[0],), float(tau_i.item()), device=device)
            ab_t = self._alpha_bar_ct(tau)
            sqrt_ab_t = torch.sqrt(ab_t)
            sqrt_one_minus_ab_t = torch.sqrt(1.0 - ab_t)
            t_embed = tau * (self.timesteps - 1)
            eps_t = model(torch.cat([x, cond], dim=1), t_embed)
            x0_pred = (x - sqrt_one_minus_ab_t * eps_t) / (sqrt_ab_t + 1e-8)
            if i == 0:
                ab_prev = torch.ones_like(ab_t)
                x = torch.sqrt(ab_prev) * x0_pred
                break
            tau_prev = torch.full((shape[0],), float(taus[i - 1].item()), device=device)
            ab_prev = self._alpha_bar_ct(tau_prev)
            # predictor
            x_pred = torch.sqrt(ab_prev) * x0_pred + torch.sqrt(1.0 - ab_prev) * eps_t
            # corrector
            t_prev_embed = tau_prev * (self.timesteps - 1)
            eps_prev = model(torch.cat([x_pred, cond], dim=1), t_prev_embed)
            eps_corr = 0.5 * (eps_t + eps_prev)
            x = torch.sqrt(ab_prev) * x0_pred + torch.sqrt(1.0 - ab_prev) * eps_corr
        return x

    @torch.no_grad()
    def sample_dpm_solver(
        self,
        model: nn.Module,
        shape,
        cond: torch.Tensor,
        steps: int = 20,
        progress: bool = False,
        t_start: int | None = None,
        x_init: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Lightweight DPM-Solver (2nd-order style) approximation using epsilon model.

        Uses a midpoint correction: predict x_{t-1} with eps_t, estimate eps at mid t_mid, then finalize.
        """
        device = cond.device
        x = x_init if x_init is not None else torch.randn(shape, device=device)

        max_t = self.timesteps - 1 if t_start is None else int(t_start)
        max_t = max(0, min(max_t, self.timesteps - 1))

        if steps >= (max_t + 1):
            schedule = list(range(max_t + 1))
        else:
            schedule = torch.linspace(0, max_t, steps, device=device).long().tolist()

        iterator = reversed(range(len(schedule)))
        if progress and tqdm is not None:
            iterator = tqdm(iterator, total=len(schedule), desc="DPM-Solver", unit="step")

        for i in iterator:
            t_i = schedule[i]
            t = torch.full((shape[0],), t_i, device=device, dtype=torch.long)
            alpha_bar_t = self._get_alpha_bar(t)
            sqrt_ab_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_ab_t = torch.sqrt(1.0 - alpha_bar_t)

            eps_t = model(torch.cat([x, cond], dim=1), t)
            x0_pred = (x - sqrt_one_minus_ab_t * eps_t) / (sqrt_ab_t + 1e-8)

            if i == 0:
                alpha_bar_prev = torch.ones_like(alpha_bar_t)
                x = torch.sqrt(alpha_bar_prev) * x0_pred
                break
            else:
                t_prev_scalar = schedule[i - 1]
                t_prev = torch.full((shape[0],), t_prev_scalar, device=device, dtype=torch.long)
                alpha_bar_prev = self._get_alpha_bar(t_prev)

            # Midpoint time index (discrete approximation)
            t_mid_scalar = (t_i + t_prev_scalar) // 2
            t_mid = torch.full((shape[0],), int(t_mid_scalar), device=device, dtype=torch.long)
            alpha_bar_mid = self._get_alpha_bar(t_mid)

            # Predict x at mid using eps_t
            x_mid = torch.sqrt(alpha_bar_mid) * x0_pred + torch.sqrt(1.0 - alpha_bar_mid) * eps_t
            eps_mid = model(torch.cat([x_mid, cond], dim=1), t_mid)

            # Final update to t-1 using eps_mid
            x = torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(1.0 - alpha_bar_prev) * eps_mid

        return x

    @torch.no_grad()
    def sample_dpm_solver_ct(
        self,
        model: nn.Module,
        shape,
        cond: torch.Tensor,
        steps: int = 20,
        progress: bool = False,
        t_start: float | None = None,
        x_init: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = cond.device
        x = x_init if x_init is not None else torch.randn(shape, device=device)
        tau_max = 1.0 if t_start is None else float(t_start)
        tau_max = float(max(0.0, min(1.0, tau_max)))
        taus = torch.linspace(0.0, tau_max, steps, device=device)
        iterator = reversed(range(len(taus)))
        if progress and tqdm is not None:
            iterator = tqdm(iterator, total=len(taus), desc="DPM-CT", unit="step")
        for i in iterator:
            tau_i = taus[i]
            tau = torch.full((shape[0],), float(tau_i.item()), device=device)
            ab_t = self._alpha_bar_ct(tau)
            sqrt_ab_t = torch.sqrt(ab_t)
            sqrt_one_minus_ab_t = torch.sqrt(1.0 - ab_t)
            t_embed = tau * (self.timesteps - 1)
            eps_t = model(torch.cat([x, cond], dim=1), t_embed)
            x0_pred = (x - sqrt_one_minus_ab_t * eps_t) / (sqrt_ab_t + 1e-8)
            if i == 0:
                ab_prev = torch.ones_like(ab_t)
                x = torch.sqrt(ab_prev) * x0_pred
                break
            tau_prev_scalar = float(taus[i - 1].item())
            tau_prev = torch.full((shape[0],), tau_prev_scalar, device=device)
            ab_prev = self._alpha_bar_ct(tau_prev)
            # midpoint tau
            tau_mid_scalar = 0.5 * (float(tau_i.item()) + tau_prev_scalar)
            tau_mid = torch.full((shape[0],), tau_mid_scalar, device=device)
            ab_mid = self._alpha_bar_ct(tau_mid)
            # midpoint state using eps_t
            x_mid = torch.sqrt(ab_mid) * x0_pred + torch.sqrt(1.0 - ab_mid) * eps_t
            eps_mid = model(torch.cat([x_mid, cond], dim=1), tau_mid * (self.timesteps - 1))
            x = torch.sqrt(ab_prev) * x0_pred + torch.sqrt(1.0 - ab_prev) * eps_mid
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
        # Shallow diffusion options
        shallow_init: torch.Tensor | None = None,
        k_step: int | None = None,
        add_forward_noise: bool = True,
        sampler: str | None = None,
    ) -> torch.Tensor:
        # model should accept concat([x, cond]) as input channels when cond is provided
        if cond is None:
            device = None
        else:
            device = cond.device

        # Shallow diffusion: start from x_t where t = k_step, initialized from shallow_init (x0 prior)
        if shallow_init is not None:
            assert cond is not None, "cond is required for conditional sampling"
            assert k_step is not None, "k_step must be provided when using shallow_init"
            k = int(k_step)
            k = max(0, min(k, self.timesteps - 1))
            # Switch by sampler
            sampler_name = (sampler or ("ddim" if use_ddim else "ddpm")).lower()
            if sampler_name in ("ddim", "ddpm"):
                if add_forward_noise:
                    t_vec = torch.full((shallow_init.size(0),), k, device=shallow_init.device, dtype=torch.long)
                    x_k = self.q_sample_at_t(shallow_init, t_vec)
                else:
                    x_k = shallow_init
                return self.sample_ddim(
                    model,
                    x_k.shape,
                    cond=cond,
                    steps=min(ddim_steps, k + 1),
                    eta=eta,
                    progress=progress,
                    t_start=k,
                    x_init=x_k,
                )
            elif sampler_name in ("ddim-ct",):
                tau_k = k / max(1, (self.timesteps - 1))
                x_k = self.q_sample_ct(shallow_init, torch.full((shallow_init.size(0),), tau_k, device=shallow_init.device)) if add_forward_noise else shallow_init
                return self.sample_ddim_ct(
                    model,
                    x_k.shape,
                    cond=cond,
                    steps=min(ddim_steps, k + 1),
                    progress=progress,
                    t_start=tau_k,
                    x_init=x_k,
                )
            elif sampler_name in ("unipc", "uni-pc"):
                return self.sample_unipc(
                    model,
                    x_k.shape,
                    cond=cond,
                    steps=min(ddim_steps, k + 1),
                    progress=progress,
                    t_start=k,
                    x_init=x_k,
                )
            elif sampler_name in ("unipc-ct", "uni-pc-ct"):
                tau_k = k / max(1, (self.timesteps - 1))
                x_k = self.q_sample_ct(shallow_init, torch.full((shallow_init.size(0),), tau_k, device=shallow_init.device)) if add_forward_noise else shallow_init
                return self.sample_unipc_ct(
                    model,
                    x_k.shape,
                    cond=cond,
                    steps=min(ddim_steps, k + 1),
                    progress=progress,
                    t_start=tau_k,
                    x_init=x_k,
                )
            elif sampler_name in ("dpm", "dpm-solver", "dpmsolver"):
                return self.sample_dpm_solver(
                    model,
                    x_k.shape,
                    cond=cond,
                    steps=min(ddim_steps, k + 1),
                    progress=progress,
                    t_start=k,
                    x_init=x_k,
                )
            elif sampler_name in ("dpm-ct", "dpm-solver-ct", "dpmsolver-ct"):
                tau_k = k / max(1, (self.timesteps - 1))
                x_k = self.q_sample_ct(shallow_init, torch.full((shallow_init.size(0),), tau_k, device=shallow_init.device)) if add_forward_noise else shallow_init
                return self.sample_dpm_solver_ct(
                    model,
                    x_k.shape,
                    cond=cond,
                    steps=min(ddim_steps, k + 1),
                    progress=progress,
                    t_start=tau_k,
                    x_init=x_k,
                )
            # DDPM reverse sampling from k..0
            x = x_k
            iterator = reversed(range(k + 1))
            if progress and tqdm is not None:
                iterator = tqdm(iterator, total=(k + 1), desc="DDPM(shallow)", unit="step")
            for i in iterator:
                t = torch.full((x.shape[0],), i, device=x.device, dtype=torch.long)
                x = self.p_sample(model, x, t, cond=cond)
            return x

        if cond is not None:
            sampler_name = (sampler or ("ddim" if use_ddim else "ddpm")).lower()
            if sampler_name == "ddim":
                return self.sample_ddim(model, shape, cond=cond, steps=ddim_steps, eta=eta, progress=progress)
            if sampler_name in ("ddim-ct",):
                return self.sample_ddim_ct(model, shape, cond=cond, steps=ddim_steps, progress=progress)
            if sampler_name in ("unipc", "uni-pc"):
                return self.sample_unipc(model, shape, cond=cond, steps=ddim_steps, progress=progress)
            if sampler_name in ("dpm", "dpm-solver", "dpmsolver"):
                return self.sample_dpm_solver(model, shape, cond=cond, steps=ddim_steps, progress=progress)
            if sampler_name in ("unipc-ct", "uni-pc-ct"):
                return self.sample_unipc_ct(model, shape, cond=cond, steps=ddim_steps, progress=progress)
            if sampler_name in ("dpm-ct", "dpm-solver-ct", "dpmsolver-ct"):
                return self.sample_dpm_solver_ct(model, shape, cond=cond, steps=ddim_steps, progress=progress)

        x = torch.randn(shape, device=device)
        iterator = reversed(range(self.timesteps))
        if progress and tqdm is not None:
            iterator = tqdm(iterator, total=self.timesteps, desc="DDPM", unit="step")
        for i in iterator:
            t = torch.full((shape[0],), i, device=x.device, dtype=torch.long)
            x = self.p_sample(model, x, t, cond=cond)
        return x
