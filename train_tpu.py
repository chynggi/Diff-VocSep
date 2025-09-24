import os
import argparse
import yaml
import math
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler
import importlib

# Lazy-import torch_xla modules to avoid import errors on non-TPU environments
try:
    xm = importlib.import_module("torch_xla.core.xla_model")
    xmp = importlib.import_module("torch_xla.distributed.xla_multiprocessing")
    pl = importlib.import_module("torch_xla.distributed.parallel_loader")
except Exception:
    xm = None
    xmp = None
    pl = None

from utils.data_loader import MUSDB18Dataset, MusDB18HQ
from utils.data_setup import ensure_musdbhq
from utils.audio_utils import AudioProcessor
from utils.metrics import evaluate_waveforms
from models.counterfactual import CounterfactualDiffusion


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--val-steps", type=int, default=0, help="Validate every N steps (0 to disable)")
    return p.parse_args()


def _build_dataloaders(cfg, device):
    dataset_kind = cfg["data"].get("dataset", "musdb").lower()
    if dataset_kind == "musdbhq":
        ensure_musdbhq(cfg["data"]["musdbhq_root"])  # ensures available
        train_set = MusDB18HQ(
            dataset_path=cfg["data"]["musdbhq_root"],
            subset="Train",
            segment_seconds=cfg["data"]["segment_seconds"],
            sr=cfg["audio"]["sample_rate"],
            n_fft=cfg["audio"]["n_fft"],
            hop_length=cfg["audio"]["hop_length"],
            win_length=cfg["audio"]["win_length"],
            center=cfg["audio"].get("center", True),
        )
        val_set = MusDB18HQ(
            dataset_path=cfg["data"]["musdbhq_root"],
            subset="Test",
            segment_seconds=cfg["data"]["segment_seconds"],
            sr=cfg["audio"]["sample_rate"],
            n_fft=cfg["audio"]["n_fft"],
            hop_length=cfg["audio"]["hop_length"],
            win_length=cfg["audio"]["win_length"],
            center=cfg["audio"].get("center", True),
        )
    else:
        train_set = MUSDB18Dataset(
            cfg["data"]["musdb_root"],
            subset="train",
            segment_length=cfg["data"]["segment_seconds"],
            sr=cfg["audio"]["sample_rate"],
            n_fft=cfg["audio"]["n_fft"],
            hop_length=cfg["audio"]["hop_length"],
            win_length=cfg["audio"]["win_length"],
            center=cfg["audio"].get("center", True),
        )
        val_set = MUSDB18Dataset(
            cfg["data"]["musdb_root"],
            subset="test",
            segment_length=cfg["data"]["segment_seconds"],
            sr=cfg["audio"]["sample_rate"],
            n_fft=cfg["audio"]["n_fft"],
            hop_length=cfg["audio"]["hop_length"],
            win_length=cfg["audio"]["win_length"],
            center=cfg["audio"].get("center", True),
        )

    # Optional pitch augmentation setup for training set
    pitch_aug_cfg = cfg["data"].get("pitch_aug", None)
    if pitch_aug_cfg and pitch_aug_cfg.get("enabled", False):
        lo, hi = pitch_aug_cfg.get("semitones", (-2.0, 2.0))
        prob = pitch_aug_cfg.get("prob", 0.5)
        if hasattr(train_set, "set_pitch_aug"):
            train_set.set_pitch_aug(True, (float(lo), float(hi)), float(prob))

    if xm is None:
        raise RuntimeError("torch-xla is required for TPU training. Please install torch-xla and run on a TPU runtime.")

    # Use stable XLA APIs (PJRT-first, fallback to legacy XRT)
    if hasattr(xm, "xla_device_world_size"):
        world_size = xm.xla_device_world_size()
    else:
        world_size = xm.xrt_world_size()
    rank = xm.get_ordinal()
    
    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )
    # For validation, run only on master to simplify aggregation
    val_loader_master = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["train"]["batch_size"],
        sampler=train_sampler,
        num_workers=0,
        drop_last=True,
    )
    # Prefetch to device via MpDeviceLoader
    mp_train_loader = pl.MpDeviceLoader(train_loader, device)

    return mp_train_loader, train_sampler, val_loader_master


def _validate(model, val_loader, cfg, device, max_batches=10):
    # Master-only validation to avoid aggregation complexity
    if xm is None or not xm.is_master_ordinal():
        return float("nan")

    model.eval()
    total_sdr = 0.0
    count = 0
    with torch.no_grad():
        proc = AudioProcessor(
            sr=cfg["audio"]["sample_rate"],
            n_fft=cfg["audio"]["n_fft"],
            hop_length=cfg["audio"]["hop_length"],
            win_length=cfg["audio"]["win_length"],
            center=cfg["audio"].get("center", True),
        )
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            mix = batch["mixture"].to(device)
            acc = batch["accompaniment"].to(device)

            # Validation-time sampler/shallow settings aligned with GPU path
            val_sampler = cfg["diffusion"].get("sampler", None)
            val_use_ddim = cfg["diffusion"].get("use_ddim", False)
            val_ddim_steps = cfg["diffusion"].get("val_ddim_steps", cfg["diffusion"].get("ddim_steps", 50))
            val_eta = cfg["diffusion"].get("eta", 0.0)
            shallow_cfg = cfg["diffusion"].get("validate_use_shallow", False)
            shallow_k = cfg["diffusion"].get("shallow_k", None)
            add_forward_noise = cfg["diffusion"].get("add_forward_noise", True)

            instrumental_norm = model.generate_instrumental(
                mix,
                use_ddim=val_use_ddim,
                ddim_steps=val_ddim_steps,
                eta=val_eta,
                sampler=val_sampler,
                shallow_init=(mix if shallow_cfg else None),
                k_step=shallow_k,
                add_forward_noise=add_forward_noise,
            )
            vocals_est_norm = torch.clamp(mix - instrumental_norm, -1.0, 1.0)

            stats = batch.get("mix_norm_stats")
            if isinstance(stats, torch.Tensor) and stats.numel() >= 2:
                s_min = float(stats.view(-1)[0].item())
                s_max = float(stats.view(-1)[1].item())
            else:
                s_min, s_max = -1.0, 1.0

            mix_phase = batch["mixture_phase"].squeeze(0).cpu()
            voc_mag = proc.denormalize_mag(vocals_est_norm.squeeze(0).squeeze(0).detach().cpu(), s_min, s_max)
            voc_wav = proc.istft(voc_mag, mix_phase).numpy()

            # Prefer GT vocals if present; fallback to mix - accompaniment
            if "vocals" in batch and batch["vocals"] is not None:
                voc_gt_mag = proc.denormalize_mag(batch["vocals"].squeeze(0).squeeze(0).detach().cpu(), s_min, s_max)
                target_voc = proc.istft(voc_gt_mag, mix_phase).numpy()
            else:
                acc_mag = proc.denormalize_mag(acc.squeeze(0).squeeze(0).detach().cpu(), s_min, s_max)
                acc_wav = proc.istft(acc_mag, mix_phase).numpy()
                mix_mag = proc.denormalize_mag(mix.squeeze(0).squeeze(0).detach().cpu(), s_min, s_max)
                mix_wav = proc.istft(mix_mag, mix_phase).numpy()
                target_voc = mix_wav - acc_wav

            m = evaluate_waveforms(voc_wav, target_voc, sr=cfg["audio"]["sample_rate"], use_museval=False)
            sdr = m.get("SDR", float("nan"))
            if not math.isnan(sdr):
                total_sdr += sdr
                count += 1

    model.train()
    return (total_sdr / max(1, count)) if count > 0 else float("nan")


def _mp_fn(index, args):
    # Ensure PJRT is configured by default for torch-xla 2.8+
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    os.environ.setdefault("XLA_USE_SPMD", "1")
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if xm is None:
        raise RuntimeError("torch-xla not found. Install torch-xla and run on a TPU-enabled environment.")
    device = xm.xla_device()

    # Model
    model = CounterfactualDiffusion(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        base=cfg["model"]["base_channels"],
        channels_mult=cfg["model"]["channels_mult"],
        timesteps=cfg["diffusion"]["timesteps"],
        beta_start=cfg["diffusion"]["beta_start"],
        beta_end=cfg["diffusion"]["beta_end"],
        model_type=cfg["model"].get("model_type", "unet"),
        model_kwargs=cfg["model"].get("model_kwargs", {}),
    ).to(device)

    opt = optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=max(1, cfg["train"]["epochs"]) * 1000,  # fallback if no __len__
    )

    # Data
    train_loader, train_sampler, val_loader_master = _build_dataloaders(cfg, device)

    # Logging on master only
    writer = None
    if cfg["log"].get("use_tensorboard", True) and xm.is_master_ordinal():
        writer = SummaryWriter(log_dir=cfg["log"]["tb_log_dir"])

    use_amp = bool(cfg["train"].get("amp", True))

    global_step = 0
    best_sdr = float("-inf")

    epochs = cfg["train"]["epochs"]
    for epoch in range(epochs):
        # Set epoch for sampler to reshuffle each epoch
        train_sampler.set_epoch(epoch)
        running = 0.0

        for i, batch in enumerate(train_loader):
            mix = batch["mixture"].to(device)
            acc = batch["accompaniment"].to(device)

            b = mix.size(0)
            t = torch.randint(0, model.diffusion.timesteps, (b,), device=device, dtype=torch.long)

            with torch.autocast(device_type="xla", dtype=torch.bfloat16, enabled=use_amp):
                x_start = acc
                noise = torch.randn_like(x_start)
                x_t = model.diffusion.q_sample(x_start, t, noise=noise)
                x_in = torch.cat([x_t, mix], dim=1)
                pred_noise = model(x_in, t)
                loss_cf = torch.nn.functional.l1_loss(pred_noise, noise)

                alpha_bar_t = model.diffusion.alphas_cumprod[t].view(-1, 1, 1, 1)
                sqrt_ab = torch.sqrt(alpha_bar_t)
                sqrt_one_minus_ab = torch.sqrt(1.0 - alpha_bar_t)
                x0_pred = (x_t - sqrt_one_minus_ab * pred_noise) / (sqrt_ab + 1e-8)
                vocals_est = torch.clamp(mix - x0_pred, -1.0, 1.0)
                if "vocals" in batch:
                    voc = batch["vocals"].to(device)
                    loss_voc = torch.nn.functional.l1_loss(vocals_est, voc)
                    w = float(cfg["train"].get("loss_voc_weight", 0.5))
                    loss = loss_cf + w * loss_voc
                else:
                    loss = loss_cf

            opt.zero_grad()
            loss.backward()
            if cfg["train"].get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])

            # XLA-specific optimizer step
            xm.optimizer_step(opt, barrier=True)
            if scheduler is not None:
                scheduler.step()

            running += loss.item()

            # Logging on master
            if writer and (global_step % cfg["train"]["log_interval"] == 0):
                writer.add_scalar("train/loss", loss.item(), global_step)

            # Optional validation
            val_every = args.val_steps or cfg["train"].get("val_every_steps", 0)
            if val_every and global_step > 0 and (global_step % val_every == 0):
                val_sdr = _validate(model, val_loader_master, cfg, device, max_batches=cfg["train"].get("val_batches", 10))
                if xm.is_master_ordinal():
                    if writer:
                        writer.add_scalar("val/SDR", val_sdr, global_step)
                    xm.master_print(f"Validation at step {global_step}: SDR {val_sdr:.3f} dB")
                    os.makedirs("checkpoints", exist_ok=True)
                    if cfg["log"].get("save_last", True):
                        xm.save(model.state_dict(), os.path.join("checkpoints", "last.pt"))
                    if (val_sdr > best_sdr) and cfg["log"].get("save_best", True):
                        best_sdr = val_sdr
                        xm.save(model.state_dict(), os.path.join("checkpoints", "best.pt"))

            global_step += 1
            # Mark step for XLA execution
            xm.mark_step()

        # Epoch end: report average loss on master
        avg_loss = running / max(1, (i + 1))
        if xm.is_master_ordinal():
            xm.master_print(f"Epoch {epoch+1}/{epochs} | avg_loss={avg_loss:.4f} | best_sdr={(best_sdr if best_sdr!=-float('inf') else float('nan'))}")

    if xm.is_master_ordinal():
        os.makedirs("checkpoints", exist_ok=True)
        xm.save(model.state_dict(), os.path.join("checkpoints", "last.pt"))
        if writer:
            writer.close()


def main():
    args = parse_args()
    if xmp is None:
        raise RuntimeError("torch-xla not found. Install torch-xla and run on a TPU-enabled environment.")
    # Ensure PJRT defaults for parent process as well
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    os.environ.setdefault("XLA_USE_SPMD", "1")
    # Use all available TPU cores by default
    xmp.spawn(_mp_fn, args=(args,), nprocs=None, start_method='spawn')


if __name__ == "__main__":
    main()
