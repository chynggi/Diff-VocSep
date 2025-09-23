import os
import argparse
import yaml
import torch
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from utils.data_loader import create_loader
from utils.audio_utils import AudioProcessor
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

    # Dataloaders
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
    scaler = GradScaler(enabled=cfg["train"]["amp"])

    # Logging
    writer = SummaryWriter(log_dir=cfg["log"]["tb_log_dir"]) if cfg["log"].get("use_tensorboard", True) else None

    model.train()
    global_step = 0
    best_val = float("inf")

    def validate(max_batches: int = 10):
        model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                mix = batch["mixture"].to(device)
                acc = batch["accompaniment"].to(device)
                b = mix.size(0)
                t = torch.randint(0, model.diffusion.timesteps, (b,), device=device, dtype=torch.long)
                x_start = acc
                noise = torch.randn_like(x_start)
                x_t = model.diffusion.q_sample(x_start, t, noise=noise)
                x_in = torch.cat([x_t, mix], dim=1)
                pred_noise = model(x_in, t)
                l = torch.nn.functional.l1_loss(pred_noise, noise).item()
                total += l
                count += 1
                if i + 1 >= max_batches:
                    break
        model.train()
        return total / max(1, count)

    for epoch in range(cfg["train"]["epochs"]):
        for i, batch in enumerate(train_loader):
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
                loss = torch.nn.functional.l1_loss(pred_noise, noise)

            opt.zero_grad()
            scaler.scale(loss).backward()
            if cfg["train"]["grad_clip"]:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            scaler.step(opt)
            scaler.update()

            if writer and (global_step % cfg["train"]["log_interval"] == 0):
                writer.add_scalar("train/loss", loss.item(), global_step)
            if global_step % cfg["train"]["log_interval"] == 0:
                print(f"epoch {epoch} step {global_step} loss {loss.item():.4f}")

            # Validate and checkpoint every N steps
            if cfg["train"].get("val_every_steps") and (global_step % cfg["train"]["val_every_steps"] == 0) and global_step > 0:
                val_loss = validate(cfg["train"].get("val_batches", 10))
                if writer:
                    writer.add_scalar("val/loss", val_loss, global_step)
                print(f"Validation at step {global_step}: loss {val_loss:.4f}")
                os.makedirs("checkpoints", exist_ok=True)
                if cfg["log"].get("save_last", True):
                    torch.save(model.state_dict(), os.path.join("checkpoints", "last.pt"))
                if val_loss < best_val and cfg["log"].get("save_best", True):
                    best_val = val_loss
                    torch.save(model.state_dict(), os.path.join("checkpoints", "best.pt"))
            global_step += 1

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("checkpoints", "last.pt"))
    if writer:
        writer.close()


if __name__ == "__main__":
    main()
