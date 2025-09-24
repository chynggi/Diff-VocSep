import argparse
import os
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
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars during inference.")
    p.add_argument("--stereo-output", action="store_true", help="If set, keep input stereo image and output stereo vocals via TF mask.")
    # Shallow diffusion options
    p.add_argument("--init-instrumental", type=str, default=None, help="Optional path to an initial instrumental WAV to use as shallow diffusion prior (same length as input).")
    p.add_argument("--shallow-k", type=int, default=None, help="If provided, run shallow diffusion starting from timestep k.")
    p.add_argument("--no-forward-noise", action="store_true", help="With shallow init, treat provided prior as already at x_k (skip forward noising).")
    return p.parse_args()


def main():
    # Configure PJRT defaults for torch-xla 2.8+
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    os.environ.setdefault("XLA_USE_SPMD", "1")
    # lazy import torch_xla after env setup
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
        model_type=cfg["model"].get("model_type", "unet"),
        model_kwargs=cfg["model"].get("model_kwargs", {}),
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

    # Keep stereo if requested
    keep_stereo = bool(args.stereo_output)
    wav, _ = proc.load_audio(args.input, target_sr=cfg["audio"]["sample_rate"], keep_stereo=keep_stereo)
    if keep_stereo and wav.ndim == 2 and wav.size(0) > 1:
        wav_mono = wav.mean(dim=0)
    else:
        wav_mono = wav if wav.ndim == 1 else wav.squeeze(0)
    mag, phase = proc.stft(wav_mono)
    mag_norm, s_min, s_max = proc.normalize_mag(mag)

    with torch.no_grad(), torch.autocast(device_type="xla", dtype=torch.bfloat16, enabled=True):
        mix_bchw = mag_norm.unsqueeze(0).unsqueeze(0).to(device)
        gen_kwargs = dict(
            use_ddim=cfg["diffusion"].get("use_ddim", False),
            ddim_steps=cfg["diffusion"].get("ddim_steps", 50),
            eta=cfg["diffusion"].get("eta", 0.0),
            progress=not args.no_progress,
            sampler=cfg["diffusion"].get("sampler", None),
        )

        # Prepare shallow diffusion init if provided
        shallow_init = None
        if args.init_instrumental is not None:
            init_wav, _ = proc.load_audio(args.init_instrumental, target_sr=cfg["audio"]["sample_rate"])
            # Match length
            if init_wav.numel() != wav.numel():
                L = wav.shape[-1]
                if init_wav.shape[-1] < L:
                    pad = L - init_wav.shape[-1]
                    init_wav = torch.nn.functional.pad(init_wav, (0, pad))
                else:
                    init_wav = init_wav[..., :L]
            init_mag, _ = proc.stft(init_wav if init_wav.ndim == 1 else init_wav.squeeze(0))
            init_mag_norm = proc.normalize_mag_with_stats(init_mag, s_min, s_max)
            shallow_init = init_mag_norm.unsqueeze(0).unsqueeze(0).to(device)
            gen_kwargs.update(
                dict(
                    shallow_init=shallow_init,
                    k_step=args.shallow_k if args.shallow_k is not None else cfg["diffusion"].get("shallow_k", None),
                    add_forward_noise=not args.no_forward_noise,
                )
            )
        elif args.shallow_k is not None or cfg["diffusion"].get("shallow_k", None) is not None:
            shallow_init = mix_bchw
            gen_kwargs.update(
                dict(
                    shallow_init=shallow_init,
                    k_step=args.shallow_k if args.shallow_k is not None else cfg["diffusion"].get("shallow_k", None),
                    add_forward_noise=True,
                )
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
        vocals_mag_mono = proc.denormalize_mag(vocals_norm.squeeze(0).squeeze(0).cpu(), s_min, s_max)

        if keep_stereo and isinstance(wav, torch.Tensor) and wav.ndim == 2 and wav.size(0) > 1:
            eps = 1e-8
            mix_mono_mag = proc.denormalize_mag(mix_bchw.squeeze(0).squeeze(0).cpu(), s_min, s_max)
            mask = torch.clamp(vocals_mag_mono / (mix_mono_mag + eps), 0.0, 1.0)
            chans = wav.size(0)
            ch_wavs = []
            for ch in range(chans):
                ch_mag, ch_phase = proc.stft(wav[ch])
                ch_vocal_mag = torch.clamp(mask * ch_mag, min=0.0)
                ch_vocal_wav = proc.istft(ch_vocal_mag, ch_phase)
                if ch_vocal_wav.shape[-1] < wav.shape[-1]:
                    pad = wav.shape[-1] - ch_vocal_wav.shape[-1]
                    ch_vocal_wav = torch.nn.functional.pad(ch_vocal_wav, (0, pad))
                ch_wavs.append(ch_vocal_wav[: wav.shape[-1]])
            vocals_wav = torch.stack(ch_wavs, dim=0)
        else:
            vocals_wav = proc.istft(vocals_mag_mono, phase)
            vocals_wav = vocals_wav.unsqueeze(0)

    torchaudio.save(args.output, vocals_wav.cpu(), cfg["audio"]["sample_rate"])
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
