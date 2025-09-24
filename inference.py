import argparse
import yaml
import torch
import torchaudio

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
    # Shallow diffusion options
    p.add_argument("--init-instrumental", type=str, default=None, help="Optional path to an initial instrumental WAV to use as shallow diffusion prior (same length as input).")
    p.add_argument("--shallow-k", type=int, default=None, help="If provided, run shallow diffusion starting from timestep k.")
    p.add_argument("--no-forward-noise", action="store_true", help="With shallow init, treat provided prior as already at x_k (skip forward noising).")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    sd = torch.load(args.model_path, map_location=device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    proc = AudioProcessor(sr=cfg["audio"]["sample_rate"], n_fft=cfg["audio"]["n_fft"], hop_length=cfg["audio"]["hop_length"], win_length=cfg["audio"]["win_length"], center=cfg["audio"].get("center", True))    
    wav, sr = proc.load_audio(args.input, target_sr=cfg["audio"]["sample_rate"]) 
    mag, phase = proc.stft(wav)
    mag_norm, s_min, s_max = proc.normalize_mag(mag)

    with torch.no_grad():
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
            # Match length to input
            if init_wav.numel() != wav.numel():
                L = wav.shape[-1]
                if init_wav.shape[-1] < L:
                    pad = L - init_wav.shape[-1]
                    init_wav = torch.nn.functional.pad(init_wav, (0, pad))
                else:
                    init_wav = init_wav[..., :L]
            init_mag, _ = proc.stft(init_wav)
            # Normalize using mixture stats so subtraction remains consistent
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
            # Shallow with no explicit init: use mixture as a weak prior (common in shallow diffusion usage)
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
            # Convert seconds to STFT frames: frames ~= ceil((N - win)/hop) + 1, approximate via T from mag
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
                show_progress=not args.no_progress,
            )
        else:
            instrumental_norm = model.generate_instrumental(mix_bchw, **gen_kwargs)
        # vocals = mix - instrumental in normalized space
        vocals_norm = torch.clamp(mix_bchw - instrumental_norm, -1.0, 1.0)
        vocals_mag = proc.denormalize_mag(vocals_norm.squeeze(0).squeeze(0).cpu(), s_min, s_max)
        vocals_wav = proc.istft(vocals_mag, phase)

    torchaudio.save(args.output, vocals_wav.unsqueeze(0).cpu(), cfg["audio"]["sample_rate"])
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
