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
    sampler = DistributedSampler(ds, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=False)
    dl = DataLoader(ds, batch_size=1, sampler=sampler, num_workers=0)
    mp_dl = pl.MpDeviceLoader(dl, device)

    total_sdr, count = _evaluate_loop(model, mp_dl, cfg, device, args.max_batches)

    # reduce to master
    total_sdr = xm.mesh_reduce("sum_sdr", total_sdr, sum)
    count = xm.mesh_reduce("sum_count", count, sum)

    if xm.is_master_ordinal():
        avg = (total_sdr / max(1, count)) if count > 0 else float("nan")
        xm.master_print(f"Validation SDR: {avg:.3f} dB (over {count} samples)")


def main():
    args = parse_args()
    xmp = importlib.import_module("torch_xla.distributed.xla_multiprocessing")
    xmp.spawn(_mp_fn, args=(args,), nprocs=None, start_method='spawn')


if __name__ == "__main__":
    main()
