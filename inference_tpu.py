import argparse
import yaml
import torch
import torchaudio
import importlib

from utils.audio_utils import AudioProcessor
from models.counterfactual import CounterfactualDiffusion
from utils.segment_infer import spectrogram_segment_infer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--model_path", type=str, default="checkpoints/last.pt")
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, default="vocals.wav")
    p.add_argument("--segment-seconds", type=float, default=0.0, help="Segment length in seconds for chunked inference (0 for full).")
    p.add_argument("--overlap-seconds", type=float, default=0.0, help="Overlap in seconds between segments.")
    return p.parse_args()


def main():
    # lazy import torch_xla
    xm = importlib.import_module("torch_xla.core.xla_model")

    args = parse_args()
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
    ).to(device)

    sd = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(sd, strict=False)
    model.eval()

    proc = AudioProcessor(
        sr=cfg["audio"]["sample_rate"],
        n_fft=cfg["audio"]["n_fft"],
        hop_length=cfg["audio"]["hop_length"],
        win_length=cfg["audio"]["win_length"],
        center=cfg["audio"].get("center", True),
    )

    wav, _ = proc.load_audio(args.input, target_sr=cfg["audio"]["sample_rate"])
    mag, phase = proc.stft(wav)
    mag_norm, s_min, s_max = proc.normalize_mag(mag)

    with torch.no_grad(), torch.autocast(device_type="xla", dtype=torch.bfloat16, enabled=True):
        mix_bchw = mag_norm.unsqueeze(0).unsqueeze(0).to(device)
        gen_kwargs = dict(
            use_ddim=cfg["diffusion"].get("use_ddim", False),
            ddim_steps=cfg["diffusion"].get("ddim_steps", 50),
            eta=cfg["diffusion"].get("eta", 0.0),
        )
        seg_secs = float(args.segment_seconds)
        if seg_secs and seg_secs > 0:
            frames_per_sec = mag.shape[-1] / (wav.shape[-1] / cfg["audio"]["sample_rate"] + 1e-8)
            seg_frames = max(1, int(round(seg_secs * frames_per_sec)))
            ov_secs = max(0.0, float(args.overlap_seconds))
            ov_frames = max(0, int(round(ov_secs * frames_per_sec)))
            instrumental_norm = spectrogram_segment_infer(
                model,
                mix_bchw,
                segment_frames=seg_frames,
                overlap_frames=ov_frames,
                device=device,
                generate_kwargs=gen_kwargs,
            )
        else:
            instrumental_norm = model.generate_instrumental(mix_bchw, **gen_kwargs)
        vocals_norm = torch.clamp(mix_bchw - instrumental_norm, -1.0, 1.0)
        vocals_mag = proc.denormalize_mag(vocals_norm.squeeze(0).squeeze(0).cpu(), s_min, s_max)
        vocals_wav = proc.istft(vocals_mag, phase)

    torchaudio.save(args.output, vocals_wav.unsqueeze(0).cpu(), cfg["audio"]["sample_rate"])
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
