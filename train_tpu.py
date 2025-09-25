import os
import argparse
import yaml
import math
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import importlib

# torch_xla는 TPU 환경에서 lazy import로 사용합니다 (torch-xla 2.8+ PJRT 호환성)
# import torch_xla.core.xla_model as xm
# import torch_xla.distributed.parallel_loader as pl

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


def _build_dataloaders(cfg):
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
    else: # "musdb"
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

    # 표준 DistributedSampler 사용
    train_sampler = DistributedSampler(
        train_set, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True, drop_last=True
    )
    
    # Validation은 master rank에서만 수행
    val_loader_master = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    # MpDeviceLoader를 제거하고 표준 DataLoader 사용 (PJRT 호환성)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["train"]["batch_size"],
        sampler=train_sampler,
        num_workers=0,
        drop_last=True,
    )

    return train_loader, train_sampler, val_loader_master


def _validate(model, val_loader, cfg, device, max_batches=10):
    # Lazy import xm for PJRT compatibility
    xm = importlib.import_module("torch_xla.core.xla_model")
    
    # Master-only validation (PJRT 호환성으로 MpDeviceLoader 사용하지 않음)
    if not xm.is_master_ordinal():
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
            # device로 데이터 이동
            mix_cpu = batch["mixture"]
            acc_cpu = batch["accompaniment"]
            mix = mix_cpu.to(device)
            acc = acc_cpu.to(device)

            # Validation-time sampler/shallow settings
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
            s_min, s_max = (-1.0, 1.0)
            if isinstance(stats, torch.Tensor) and stats.numel() >= 2:
                s_min = float(stats.view(-1)[0].item())
                s_max = float(stats.view(-1)[1].item())

            mix_phase = batch["mixture_phase"].squeeze(0).cpu()
            voc_mag = proc.denormalize_mag(vocals_est_norm.squeeze(0).squeeze(0).detach().cpu(), s_min, s_max)
            voc_wav = proc.istft(voc_mag, mix_phase).numpy()
            
            # Ground truth 보컬 생성
            if "vocals" in batch and batch["vocals"] is not None:
                voc_gt_mag = proc.denormalize_mag(batch["vocals"].squeeze(0).squeeze(0).cpu(), s_min, s_max)
                target_voc = proc.istft(voc_gt_mag, mix_phase).numpy()
            else:
                acc_mag = proc.denormalize_mag(acc_cpu.squeeze(0).squeeze(0).cpu(), s_min, s_max)
                acc_wav = proc.istft(acc_mag, mix_phase).numpy()
                mix_mag = proc.denormalize_mag(mix_cpu.squeeze(0).squeeze(0).cpu(), s_min, s_max)
                mix_wav = proc.istft(mix_mag, mix_phase).numpy()
                target_voc = mix_wav - acc_wav

            m = evaluate_waveforms(voc_wav, target_voc, sr=cfg["audio"]["sample_rate"], use_museval=False)
            sdr = m.get("SDR", float("nan"))
            if not math.isnan(sdr):
                total_sdr += sdr
                count += 1

    model.train()
    return (total_sdr / max(1, count)) if count > 0 else float("nan")


def main():
    # Configure PJRT defaults for torch-xla 2.8+ compatibility
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    os.environ.setdefault("XLA_USE_SPMD", "1")
    
    # Lazy import torch_xla after PJRT environment setup (torch-xla 2.8+ compatibility)
    xm = importlib.import_module("torch_xla.core.xla_model")
    
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. 분산 환경 초기화 - torchrun 사용시 자동으로 설정됨
    dist.init_process_group("xla")
    device = xm.xla_device()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

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
    
    # DataLoader에서 데이터 길이를 가져와 스케줄러 설정
    # 임시 DataLoader를 사용하여 길이 계산
    temp_train_set = MUSDB18Dataset(cfg["data"]["musdb_root"], subset="train")
    steps_per_epoch = len(temp_train_set) // (cfg["train"]["batch_size"] * world_size)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, cfg["train"]["epochs"]) * steps_per_epoch
    )

    # Data
    train_loader, train_sampler, val_loader_master = _build_dataloaders(cfg)

    # Logging on master only
    writer = None
    if cfg["log"].get("use_tensorboard", True) and xm.is_master_ordinal():
        writer = SummaryWriter(log_dir=cfg["log"]["tb_log_dir"])

    use_amp = bool(cfg["train"].get("amp", True))
    global_step = 0
    best_sdr = float("-inf")
    epochs = cfg["train"]["epochs"]

    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        
        for i, batch in enumerate(train_loader):
            # 루프 내에서 데이터를 디바이스로 이동
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
            
            # 2. 표준 optimizer.step() 사용
            opt.step()
            
            # 3. xm.mark_step()은 그래프 실행을 위해 여전히 필요
            xm.mark_step()
            
            if scheduler is not None:
                scheduler.step()

            # Logging on master
            if writer and (global_step % cfg["train"]["log_interval"] == 0):
                # xm.all_gather를 사용하여 모든 코어의 loss를 모아 평균을 낼 수 있습니다.
                loss_reduced = xm.all_gather(loss).mean()
                writer.add_scalar("train/loss", loss_reduced.item(), global_step)

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
            if i % 20 == 0 and xm.is_master_ordinal():
                print(f"Epoch {epoch+1}/{epochs}, Step {i}, Loss: {loss.item():.4f}")

    if xm.is_master_ordinal():
        os.makedirs("checkpoints", exist_ok=True)
        xm.save(model.state_dict(), os.path.join("checkpoints", "last.pt"))
        if writer:
            writer.close()


if __name__ == "__main__":
    # torch-xla 2.8+ 호환성: torchrun 사용 권장 (xmp.spawn 대신)
    # torchrun --nproc_per_node=8 train_tpu.py --config config.yaml
    main()