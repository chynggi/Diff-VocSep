import argparse
import yaml
import torch
import torchaudio

from utils.audio_utils import AudioProcessor
from models.counterfactual import CounterfactualDiffusion


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--model_path", type=str, default="checkpoints/last.pt")
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, default="vocals.wav")
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
        instrumental_norm = model.generate_instrumental(mix_bchw)
        # vocals = mix - instrumental in normalized space
        vocals_norm = mix_bchw - instrumental_norm
        vocals_mag = proc.denormalize_mag(vocals_norm.squeeze(0).squeeze(0).cpu(), s_min, s_max)
        vocals_wav = proc.istft(vocals_mag, phase)

    torchaudio.save(args.output, vocals_wav.unsqueeze(0).cpu(), cfg["audio"]["sample_rate"])
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
