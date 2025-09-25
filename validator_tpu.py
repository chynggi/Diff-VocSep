import argparse
import yaml
import math
import torch
from torch.utils.data import DataLoader, DistributedSampler
import importlib
import os

from utils.data_loader import MUSDB18Dataset, MusDB18HQ
from utils.audio_utils import AudioProcessor
from utils.metrics import evaluate_waveforms
from models.counterfactual import CounterfactualDiffusion


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--max-batches", type=int, default=50)
    return p.parse_args()


def _build_val_loader(cfg):
    dataset_kind = cfg["data"].get("dataset", "musdb").lower()
    if dataset_kind == "musdbhq":
        ds = MusDB18HQ(
            dataset_path=cfg["data"]["musdbhq_root"],
            segment_seconds=cfg["data"]["segment_seconds"],
            sr=cfg["audio"]["sample_rate"],
            n_fft=cfg["audio"]["n_fft"],
            hop_length=cfg["audio"]["hop_length"],
            win_length=cfg["audio"]["win_length"],
            center=cfg["audio"].get("center", True),
        )
    else:
        ds = MUSDB18Dataset(
            cfg["data"]["musdb_root"],
            subset="test",
            segment_length=cfg["data"]["segment_seconds"],
            sr=cfg["audio"]["sample_rate"],
            n_fft=cfg["audio"]["n_fft"],
            hop_length=cfg["audio"]["hop_length"],
            win_length=cfg["audio"]["win_length"],
            center=cfg["audio"].get("center", True),
        )
    return ds


def _evaluate_loop(model, loader, cfg, device, max_batches: int):
    proc = AudioProcessor(
        sr=cfg["audio"]["sample_rate"],
        n_fft=cfg["audio"]["n_fft"],
        hop_length=cfg["audio"]["hop_length"],
        win_length=cfg["audio"]["win_length"],
        center=cfg["audio"].get("center", True),
    )
    total_sdr = 0.0
    count = 0
    with torch.no_grad(), torch.autocast(device_type="xla", dtype=torch.bfloat16, enabled=True):
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            mix = batch["mixture"].to(device)
            acc = batch["accompaniment"].to(device)

            instrumental_norm = model.generate_instrumental(
                mix,
                use_ddim=cfg["diffusion"].get("use_ddim", False),
                ddim_steps=cfg["diffusion"].get("ddim_steps", 50),
                eta=cfg["diffusion"].get("eta", 0.0),
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
    return total_sdr, count


def _mp_fn(index, args):
    # Ensure PJRT defaults
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    os.environ.setdefault("XLA_USE_SPMD", "1")
    xm = importlib.import_module("torch_xla.core.xla_model")
    xmp = importlib.import_module("torch_xla.distributed.xla_multiprocessing")
    pl = importlib.import_module("torch_xla.distributed.parallel_loader")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = xm.xla_device()

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

    # Use best/last checkpoint if present
    ckpt = None
    for path in ("checkpoints/best.pt", "checkpoints/last.pt"):
        if os.path.exists(path):
            ckpt = path
            break
    if ckpt:
        sd = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(sd, strict=False)
    model.eval()

    ds = _build_val_loader(cfg)
    # Use xla_device_world_size() for torch-xla 2.8+ compatibility
    world_size = xm.xla_device_world_size()
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=xm.get_ordinal(), shuffle=False)
    dl = DataLoader(ds, batch_size=1, sampler=sampler, num_workers=0)
    # Note: MpDeviceLoader may not be needed with PJRT in torch-xla 2.8+
    # For now, keeping it for backward compatibility, but consider removing if issues arise
    mp_dl = pl.MpDeviceLoader(dl, device)

    total_sdr, count = _evaluate_loop(model, mp_dl, cfg, device, args.max_batches)

    # All-reduce across TPU processes (sum) for PJRT compatibility
    total_sdr_t = torch.tensor([total_sdr], device=device, dtype=torch.float32)
    count_t = torch.tensor([count], device=device, dtype=torch.float32)
    total_sdr_t = xm.all_reduce(xm.REDUCE_SUM, [total_sdr_t])[0]
    count_t = xm.all_reduce(xm.REDUCE_SUM, [count_t])[0]
    total_sdr = float(total_sdr_t.cpu().item())
    count = int(count_t.cpu().item())

    if xm.is_master_ordinal():
        avg = (total_sdr / max(1, count)) if count > 0 else float("nan")
        xm.master_print(f"Validation SDR: {avg:.3f} dB (over {count} samples)")


def main():
    args = parse_args()
    # Ensure PJRT defaults for parent process (torch-xla 2.8+ compatibility)
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    os.environ.setdefault("XLA_USE_SPMD", "1")
    
    # For torch-xla 2.8+, consider using torchrun instead of xmp.spawn:
    # torchrun --nproc_per_node=8 validator_tpu.py --config config.yaml
    xmp = importlib.import_module("torch_xla.distributed.xla_multiprocessing")
    xmp.spawn(_mp_fn, args=(args,), nprocs=None, start_method='spawn')


if __name__ == "__main__":
    main()
