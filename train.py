import os
import argparse
import yaml
import torch
from torch import optim
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm, trange

from utils.data_loader import create_loader, create_musdbhq_loader
from utils.data_setup import ensure_musdbhq
from utils.audio_utils import AudioProcessor
from utils.metrics import evaluate_waveforms
from models.counterfactual import CounterfactualDiffusion


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataloaders (musdb or musdbhq)
    dataset_kind = cfg["data"].get("dataset", "musdb").lower()
    if dataset_kind == "musdbhq":
        # Ensure dataset exists, download if missing
        ensure_musdbhq(cfg["data"]["musdbhq_root"])
        train_loader = create_musdbhq_loader(
            root_dir=cfg["data"]["musdbhq_root"],
            batch_size=cfg["train"]["batch_size"],
            segment_seconds=cfg["data"]["segment_seconds"],
            sr=cfg["audio"]["sample_rate"],
            n_fft=cfg["audio"]["n_fft"],
            hop=cfg["audio"]["hop_length"],
            win_length=cfg["audio"]["win_length"],
            center=cfg["audio"].get("center", True),
            num_workers=0,
        )
        # val 로더는 간단히 동일 구조에서 batch_size=1로 생성
        val_loader = create_musdbhq_loader(
            root_dir=cfg["data"]["musdbhq_root"],
            batch_size=1,
            segment_seconds=cfg["data"]["segment_seconds"],
            sr=cfg["audio"]["sample_rate"],
            n_fft=cfg["audio"]["n_fft"],
            hop=cfg["audio"]["hop_length"],
            win_length=cfg["audio"]["win_length"],
            center=cfg["audio"].get("center", True),
            num_workers=0,
        )
    else:
        train_loader = create_loader(
            root_dir=cfg["data"]["musdb_root"],
            subset="train",
            batch_size=cfg["train"]["batch_size"],
            segment_seconds=cfg["data"]["segment_seconds"],
            sr=cfg["audio"]["sample_rate"],
            n_fft=cfg["audio"]["n_fft"],
            hop=cfg["audio"]["hop_length"],
            win_length=cfg["audio"]["win_length"],
            center=cfg["audio"].get("center", True),
            num_workers=0,
        )
        val_loader = create_loader(
            root_dir=cfg["data"]["musdb_root"],
            subset="test",  # MSST와 유사하게 별도 검증 세트 사용 가능
            batch_size=1,
            segment_seconds=cfg["data"]["segment_seconds"],
            sr=cfg["audio"]["sample_rate"],
            n_fft=cfg["audio"]["n_fft"],
            hop=cfg["audio"]["hop_length"],
            win_length=cfg["audio"]["win_length"],
            center=cfg["audio"].get("center", True),
            num_workers=0,
        )

    # Model
    model = CounterfactualDiffusion(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        base=cfg["model"]["base_channels"],
        channels_mult=cfg["model"]["channels_mult"],
        timesteps=cfg["diffusion"]["timesteps"],
        beta_start=cfg["diffusion"]["beta_start"],
        beta_end=cfg["diffusion"]["beta_end"],
    ).to(device)

    opt = optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg["train"]["epochs"]) * max(1, len(train_loader)))
    scaler = GradScaler(enabled=cfg["train"]["amp"])

    # Logging
    writer = SummaryWriter(log_dir=cfg["log"]["tb_log_dir"]) if cfg["log"].get("use_tensorboard", True) else None

    model.train()
    global_step = 0
    best_sdr = float("-inf")

    def validate(max_batches: int = 10):
        model.eval()
        total_sdr = 0.0
        count = 0
        with torch.no_grad():
            total = min(max_batches, len(val_loader)) if hasattr(val_loader, "__len__") else max_batches
            val_iter = enumerate(val_loader)
            val_pbar = tqdm(val_iter, total=total, desc="Validate", dynamic_ncols=True, leave=False)
            for i, batch in val_pbar:
                mix = batch["mixture"].to(device)   # (1,1,F,T)
                acc = batch["accompaniment"].to(device)
                # 샘플링으로 악기 스펙 추정 -> 보컬 = 혼합 - 악기
                instrumental_norm = model.generate_instrumental(
                    mix,
                    use_ddim=cfg["diffusion"].get("use_ddim", False),
                    ddim_steps=cfg["diffusion"].get("ddim_steps", 50),
                    eta=cfg["diffusion"].get("eta", 0.0),
                )
                vocals_est_norm = torch.clamp(mix - instrumental_norm, -1.0, 1.0)

                stats = batch.get("mix_norm_stats")
                if stats is not None:
                    # stats can be shaped (B, 2) due to DataLoader batching or (2,) per sample
                    try:
                        if isinstance(stats, torch.Tensor):
                            if stats.ndim == 2 and stats.size(-1) == 2:
                                s_min = float(stats[0, 0].item())
                                s_max = float(stats[0, 1].item())
                            elif stats.ndim == 1 and stats.numel() == 2:
                                s_min = float(stats[0].item())
                                s_max = float(stats[1].item())
                            else:
                                # Fallback: use global min/max
                                s_min = float(stats.min().item())
                                s_max = float(stats.max().item())
                        elif isinstance(stats, (list, tuple)) and len(stats) >= 2:
                            s_min = float(stats[0])
                            s_max = float(stats[1])
                        else:
                            s_min, s_max = -1.0, 1.0
                    except Exception:
                        s_min, s_max = -1.0, 1.0
                else:
                    s_min, s_max = -1.0, 1.0

                proc = AudioProcessor(sr=cfg["audio"]["sample_rate"], n_fft=cfg["audio"]["n_fft"], hop_length=cfg["audio"]["hop_length"], win_length=cfg["audio"]["win_length"], center=cfg["audio"].get("center", True))
                mix_phase = batch["mixture_phase"].squeeze(0).cpu()
                voc_mag = proc.denormalize_mag(vocals_est_norm.squeeze(0).squeeze(0).cpu(), s_min, s_max)
                voc_wav = proc.istft(voc_mag, mix_phase).numpy()

                # GT 보컬 파형이 없어서 proxy로 mixture - accompaniment(denorm, iSTFT)를 사용
                acc_mag = proc.denormalize_mag(acc.squeeze(0).squeeze(0).cpu(), s_min, s_max)
                acc_wav = proc.istft(acc_mag, mix_phase).numpy()
                mix_mag = proc.denormalize_mag(mix.squeeze(0).squeeze(0).cpu(), s_min, s_max)
                mix_wav = proc.istft(mix_mag, mix_phase).numpy()
                target_voc = mix_wav - acc_wav

                m = evaluate_waveforms(voc_wav, target_voc, sr=cfg["audio"]["sample_rate"], use_museval=False)
                sdr = m.get("SDR", float("nan"))
                if not np.isnan(sdr):
                    total_sdr += sdr
                    count += 1
                if i + 1 >= max_batches:
                    break
                if count > 0:
                    val_pbar.set_postfix(SDR=(total_sdr / max(1, count)))
            val_pbar.close()
        model.train()
        return (total_sdr / max(1, count)) if count > 0 else float("nan")



    # Training loop
    epochs = cfg["train"]["epochs"]
    steps_per_epoch = len(train_loader) if hasattr(train_loader, "__len__") else None
    epoch_bar = trange(epochs, desc="Epochs", dynamic_ncols=True)
    for epoch in epoch_bar:
        running_loss = 0.0
        # step-level progress bar for this epoch
        if steps_per_epoch is not None:
            step_bar = tqdm(train_loader, total=steps_per_epoch, desc=f"Train {epoch+1}/{epochs}", dynamic_ncols=True, leave=False)
        else:
            step_bar = tqdm(train_loader, desc=f"Train {epoch+1}/{epochs}", dynamic_ncols=True, leave=False)

        for i, batch in enumerate(step_bar):
            mix = batch["mixture"].to(device)
            acc = batch["accompaniment"].to(device)

            b = mix.size(0)
            t = torch.randint(0, model.diffusion.timesteps, (b,), device=device, dtype=torch.long)

            # forward noising on target instruments
            with autocast(enabled=cfg["train"]["amp"]):
                x_start = acc
                noise = torch.randn_like(x_start)
                x_t = model.diffusion.q_sample(x_start, t, noise=noise)
                x_in = torch.cat([x_t, mix], dim=1)
                pred_noise = model(x_in, t)
                loss_cf = torch.nn.functional.l1_loss(pred_noise, noise)

                # Add secondary vocal separation objective in spec domain (optional)
                # Estimate instruments via one reverse step preview (cheap): use current x_t prediction to approximate x0
                alpha_bar_t = model.diffusion.alphas_cumprod[t].view(-1, 1, 1, 1)
                sqrt_ab = torch.sqrt(alpha_bar_t)
                sqrt_one_minus_ab = torch.sqrt(1.0 - alpha_bar_t)
                x0_pred = (x_t - sqrt_one_minus_ab * pred_noise) / (sqrt_ab + 1e-8)
                vocals_est = torch.clamp(mix - x0_pred, -1.0, 1.0)
                # supervise with provided vocals spec if present
                if "vocals" in batch:
                    voc = batch["vocals"].to(device)
                    loss_voc = torch.nn.functional.l1_loss(vocals_est, voc)
                    w = float(cfg["train"].get("loss_voc_weight", 0.5))
                    loss = loss_cf + w * loss_voc
                else:
                    loss = loss_cf

            opt.zero_grad()
            scaler.scale(loss).backward()
            if cfg["train"]["grad_clip"]:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            scaler.step(opt)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

            if writer and (global_step % cfg["train"]["log_interval"] == 0):
                writer.add_scalar("train/loss", loss.item(), global_step)
            if global_step % cfg["train"]["log_interval"] == 0:
                # update progress bar with current loss and LR
                lr_val = opt.param_groups[0]["lr"] if len(opt.param_groups) > 0 else None
                postfix = {"loss": f"{loss.item():.4f}"}
                if lr_val is not None:
                    postfix["lr"] = f"{lr_val:.2e}"
                step_bar.set_postfix(postfix)

            # Validate and checkpoint every N steps
            if cfg["train"].get("val_every_steps") and (global_step % cfg["train"]["val_every_steps"] == 0) and global_step > 0:
                val_sdr = validate(cfg["train"].get("val_batches", 10))
                if writer:
                    writer.add_scalar("val/SDR", val_sdr, global_step)
                tqdm.write(f"Validation at step {global_step}: SDR {val_sdr:.3f} dB")
                os.makedirs("checkpoints", exist_ok=True)
                if cfg["log"].get("save_last", True):
                    torch.save(model.state_dict(), os.path.join("checkpoints", "last.pt"))
                if (val_sdr > best_sdr) and cfg["log"].get("save_best", True):
                    best_sdr = val_sdr
                    torch.save(model.state_dict(), os.path.join("checkpoints", "best.pt"))
            global_step += 1
            running_loss += loss.item()

        # end epoch: close step bar and update epoch bar postfix
        step_bar.close()
        avg_loss = running_loss / max(1, (i + 1))
        epoch_bar.set_postfix({"avg_loss": f"{avg_loss:.4f}", "best_sdr": f"{best_sdr:.3f}" if best_sdr != float('-inf') else "-inf"})

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("checkpoints", "last.pt"))
    if writer:
        writer.close()


if __name__ == "__main__":
    main()
