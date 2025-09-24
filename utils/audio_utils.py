import torch
import torchaudio
import numpy as np
from typing import Tuple


def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    # waveform: (channels, time) or (time,)
    if waveform.ndim == 1:
        return waveform
    if waveform.size(0) > 1:
        return waveform.mean(dim=0)
    return waveform.squeeze(0)


class AudioProcessor:
    def __init__(self, sr: int = 44100, n_fft: int = 2048, hop_length: int = 512, win_length: int = 2048, window: str = "hann", center: bool = True):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center
        window = (window or "hann").lower()
        if window == "hann":
            self.window = torch.hann_window(win_length)
        elif window == "hamming":
            self.window = torch.hamming_window(win_length)
        else:
            self.window = torch.hann_window(win_length)

    def load_audio(self, path: str, target_sr: int = None) -> Tuple[torch.Tensor, int]:
        wav, sr = torchaudio.load(path)
        wav = to_mono(wav)
        if target_sr and sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            wav = resampler(wav)
            sr = target_sr
        return wav, sr

    def stft(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(waveform.device),
            center=self.center,
            return_complex=True,
        )
        mag = torch.abs(stft)
        phase = torch.angle(stft)
        return mag, phase

    def istft(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        complex_spec = torch.polar(magnitude, phase)
        wav = torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(magnitude.device),
            center=self.center,
        )
        return wav

    @staticmethod
    def normalize_mag(spec: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        # log1p + minmax → return stats for inverse if needed
        log_spec = torch.log1p(spec)
        s_min = float(log_spec.min())
        s_max = float(log_spec.max())
        norm = (log_spec - s_min) / (s_max - s_min + 1e-8)
        norm = norm * 2.0 - 1.0
        return norm, s_min, s_max

    def normalize_mag_with_stats(self, mag: torch.Tensor, s_min: float, s_max: float) -> torch.Tensor:
        """Normalize magnitude using externally provided stats (e.g., from mixture).

        Args:
            mag: (F, T) magnitude
            s_min, s_max: stats computed from reference spectrogram after log1p (floats)
        Returns:
            mag_norm: (F, T) normalized with provided stats (still in [0,1] before mapping if needed)
        """
        m = torch.log1p(torch.clamp(mag, min=0.0))
        norm01 = (m - s_min) / (s_max - s_min + 1e-8)
        return norm01 * 2.0 - 1.0

    @staticmethod
    def denormalize_mag(norm: torch.Tensor, s_min: float, s_max: float) -> torch.Tensor:
        log_spec = (norm + 1.0) * 0.5 * (s_max - s_min) + s_min
        spec = torch.expm1(log_spec)
        return spec
