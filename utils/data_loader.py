import random
from typing import Dict, Optional, Tuple
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import musdb
import torchaudio

from .audio_utils import AudioProcessor, to_mono
try:
    from pedalboard import Pedalboard, PitchShift
    _HAS_PEDALBOARD = True
except Exception:
    _HAS_PEDALBOARD = False


def _exists(x) -> bool:
    return x is not None

class MUSDB18Dataset(Dataset):
    def __init__(self, root_dir: str, subset: str = "train", segment_length: float = 4.0, sr: int = 44100, n_fft: int = 2048, hop_length: int = 512, win_length: int = 2048, center: bool = True):
        self.db = musdb.DB(root=root_dir, subsets=subset)
        self.segment_length = segment_length
        self.processor = AudioProcessor(sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=center)
        # Optional pitch augmentation config
        self.pitch_aug = {"enabled": False, "semitones": (-2.0, 2.0), "prob": 0.0}

    def set_pitch_aug(self, enabled: bool, semitone_range: Tuple[float, float], prob: float = 0.5):
        self.pitch_aug = {"enabled": enabled, "semitones": (float(semitone_range[0]), float(semitone_range[1])), "prob": float(prob)}

    def __len__(self):
        # Heuristic: 10 segments per track
        return len(self.db) * 10

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        track = self.db[idx // 10]
        start = random.uniform(0, max(0.0, track.duration - self.segment_length))
        track.chunk_duration = self.segment_length
        track.chunk_start = start

        mixture = torch.tensor(track.audio.T)
        vocals = torch.tensor(track.targets['vocals'].audio.T)
        # accompaniment = drums + bass + other
        accomp_np = track.targets['drums'].audio + track.targets['bass'].audio + track.targets['other'].audio
        accompaniment = torch.tensor(accomp_np.T)

        mixture = to_mono(mixture)
        vocals = to_mono(vocals)
        accompaniment = to_mono(accompaniment)

        # Apply optional pitch augmentation consistently across sources
        if self.pitch_aug.get("enabled", False) and _HAS_PEDALBOARD and random.random() < float(self.pitch_aug.get("prob", 0.0)):
            lo, hi = self.pitch_aug.get("semitones", (-2.0, 2.0))
            shift = random.uniform(float(lo), float(hi))
            board = Pedalboard([PitchShift(semitones=shift)])
            sr = self.processor.sr
            def _do(w: torch.Tensor) -> torch.Tensor:
                x = w.detach().cpu().numpy().astype("float32")[None, :]
                y = board(x, sample_rate=sr)[0]
                return torch.from_numpy(y)
            mixture = _do(mixture)
            vocals = _do(vocals)
            accompaniment = _do(accompaniment)

        mix_mag, mix_phase = self.processor.stft(mixture)
        voc_mag, _ = self.processor.stft(vocals)
        acc_mag, _ = self.processor.stft(accompaniment)

        # Normalize EVERYTHING with mixture stats so subtraction is valid
        mix_norm, mix_min, mix_max = self.processor.normalize_mag(mix_mag)
        voc_norm = self.processor.normalize_mag_with_stats(voc_mag, mix_min, mix_max)
        acc_norm = self.processor.normalize_mag_with_stats(acc_mag, mix_min, mix_max)

        return {
            "mixture": mix_norm.unsqueeze(0),
            "vocals": voc_norm.unsqueeze(0),
            "accompaniment": acc_norm.unsqueeze(0),
            "mixture_phase": mix_phase.unsqueeze(0),
            "mix_norm_stats": torch.tensor([mix_min, mix_max], dtype=torch.float32),
        }


def create_loader(root_dir: str, subset: str, batch_size: int, segment_seconds: float, sr: int, n_fft: int, hop: int, win_length: int, center: bool, num_workers: int = 0, pitch_aug: Optional[Dict] = None) -> DataLoader:
    dataset = MUSDB18Dataset(root_dir, subset=subset, segment_length=segment_seconds, sr=sr, n_fft=n_fft, hop_length=hop, win_length=win_length, center=center)
    if pitch_aug and pitch_aug.get("enabled", False):
        lo, hi = pitch_aug.get("semitones", (-2.0, 2.0))
        prob = pitch_aug.get("prob", 0.5)
        dataset.set_pitch_aug(True, (float(lo), float(hi)), float(prob))
    return DataLoader(dataset, batch_size=batch_size, shuffle=(subset == "train"), num_workers=num_workers)


class MusDB18HQ(Dataset):
    """
    MUSDB18-HQ local folder dataset.
    Expects track folders each containing: mixture.wav and stems (default: drums.wav, bass.wav, vocals.wav, other.wav)

    Inspired by: https://github.com/lucidrains/HS-TasNet (dataset pattern)
    Returns dict compatible with current training: normalized spectrograms and phase.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        subset: str | None = None,
        sep_filenames: tuple = ("drums", "bass", "vocals", "other"),
        segment_seconds: Optional[float] = 4.0,
        sr: int = 44100,
        n_fft: int = 8192,
        hop_length: int = 1024,
        win_length: int = 8192,
        center: bool = True,
    ):
        if isinstance(dataset_path, str):
            dataset_path = Path(dataset_path)

        # If subset provided, narrow the search to that subfolder (e.g., 'Train' or 'Test')
        base = dataset_path / subset if isinstance(subset, str) and len(subset) > 0 else dataset_path

        paths = []
        mixture_paths = base.glob("**/*/mixture.wav")
        for mixture_path in mixture_paths:
            parent = mixture_path.parent
            if not all((parent / f"{name}.wav").exists() for name in sep_filenames):
                continue
            paths.append(parent)

        self.paths = sorted(paths)
        self.sep_filenames = sep_filenames
        self.subset = subset
        self.segment_seconds = segment_seconds
        self.sr = sr
        self.processor = AudioProcessor(sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=center)
        self.pitch_aug = {"enabled": False, "semitones": (-2.0, 2.0), "prob": 0.0}

    def set_pitch_aug(self, enabled: bool, semitone_range: Tuple[float, float], prob: float = 0.5):
        self.pitch_aug = {"enabled": enabled, "semitones": (float(semitone_range[0]), float(semitone_range[1])), "prob": float(prob)}

    def __len__(self):
        return len(self.paths)

    def _load_mono(self, wav_path: Path) -> torch.Tensor:
        wav, sr = torchaudio.load(str(wav_path))
        if sr != self.sr:
            wav = torchaudio.transforms.Resample(sr, self.sr)(wav)
        return to_mono(wav)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        folder = self.paths[idx]
        mixture = self._load_mono(folder / "mixture.wav")

        # Load stems
        stems = {}
        for name in self.sep_filenames:
            stems[name] = self._load_mono(folder / f"{name}.wav")

        # Compose accompaniment = drums + bass + other
        accompaniment = stems.get("drums", torch.zeros_like(mixture))
        accompaniment = accompaniment + stems.get("bass", 0)
        accompaniment = accompaniment + stems.get("other", 0)

        vocals = stems.get("vocals", torch.zeros_like(mixture))

        # Random segment
        if _exists(self.segment_seconds) and self.segment_seconds and self.segment_seconds > 0:
            seg_len = int(self.segment_seconds * self.sr)
            total = mixture.shape[-1]
            if total > seg_len:
                start = random.randrange(0, total - seg_len)
                end = start + seg_len
                mixture = mixture[..., start:end]
                accompaniment = accompaniment[..., start:end]
                vocals = vocals[..., start:end]
            elif total < seg_len:
                # pad to seg_len
                pad = seg_len - total
                mixture = torch.nn.functional.pad(mixture, (0, pad))
                accompaniment = torch.nn.functional.pad(accompaniment, (0, pad))
                vocals = torch.nn.functional.pad(vocals, (0, pad))

        # Optional pitch augmentation (apply consistently to all stems)
        if self.pitch_aug.get("enabled", False) and _HAS_PEDALBOARD and random.random() < float(self.pitch_aug.get("prob", 0.0)):
            lo, hi = self.pitch_aug.get("semitones", (-2.0, 2.0))
            shift = random.uniform(float(lo), float(hi))
            board = Pedalboard([PitchShift(semitones=shift)])
            sr = self.processor.sr
            def _do(w: torch.Tensor) -> torch.Tensor:
                x = w.detach().cpu().numpy().astype("float32")[None, :]
                y = board(x, sample_rate=sr)[0]
                return torch.from_numpy(y)
            mixture = _do(mixture)
            accompaniment = _do(accompaniment)
            vocals = _do(vocals)

        # STFT and normalization
        mix_mag, mix_phase = self.processor.stft(mixture)
        voc_mag, _ = self.processor.stft(vocals)
        acc_mag, _ = self.processor.stft(accompaniment)

        # Use mixture stats for all stems
        mix_norm, s_min, s_max = self.processor.normalize_mag(mix_mag)
        voc_norm = self.processor.normalize_mag_with_stats(voc_mag, s_min, s_max)
        acc_norm = self.processor.normalize_mag_with_stats(acc_mag, s_min, s_max)

        return {
            "mixture": mix_norm.unsqueeze(0),
            "vocals": voc_norm.unsqueeze(0),
            "accompaniment": acc_norm.unsqueeze(0),
            "mixture_phase": mix_phase.unsqueeze(0),
            "mix_norm_stats": torch.tensor([s_min, s_max], dtype=torch.float32),
        }


def create_musdbhq_loader(root_dir: str | Path, subset: str, batch_size: int, segment_seconds: float, sr: int, n_fft: int, hop: int, win_length: int, center: bool, num_workers: int = 0, pitch_aug: Optional[Dict] = None) -> DataLoader:
    dataset = MusDB18HQ(
        dataset_path=root_dir,
        subset=subset,
        segment_seconds=segment_seconds,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win_length,
        center=center,
    )
    if pitch_aug and pitch_aug.get("enabled", False):
        lo, hi = pitch_aug.get("semitones", (-2.0, 2.0))
        prob = pitch_aug.get("prob", 0.5)
        dataset.set_pitch_aug(True, (float(lo), float(hi)), float(prob))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
