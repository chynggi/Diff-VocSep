import random
from typing import Dict

import torch
from torch.utils.data import Dataset, DataLoader
import musdb

from .audio_utils import AudioProcessor, to_mono


class MUSDB18Dataset(Dataset):
    def __init__(self, root_dir: str, subset: str = "train", segment_length: float = 4.0, sr: int = 44100, n_fft: int = 2048, hop_length: int = 512, win_length: int = 2048, center: bool = True):
        self.db = musdb.DB(root=root_dir, subsets=subset)
        self.segment_length = segment_length
        self.processor = AudioProcessor(sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=center)

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

        mix_mag, mix_phase = self.processor.stft(mixture)
        voc_mag, _ = self.processor.stft(vocals)
        acc_mag, _ = self.processor.stft(accompaniment)

        mix_norm, mix_min, mix_max = self.processor.normalize_mag(mix_mag)
        voc_norm, _, _ = self.processor.normalize_mag(voc_mag)
        acc_norm, _, _ = self.processor.normalize_mag(acc_mag)

        return {
            "mixture": mix_norm.unsqueeze(0),
            "vocals": voc_norm.unsqueeze(0),
            "accompaniment": acc_norm.unsqueeze(0),
            "mixture_phase": mix_phase.unsqueeze(0),
            "mix_norm_stats": torch.tensor([mix_min, mix_max], dtype=torch.float32),
        }


def create_loader(root_dir: str, subset: str, batch_size: int, segment_seconds: float, sr: int, n_fft: int, hop: int, win_length: int, center: bool, num_workers: int = 0) -> DataLoader:
    dataset = MUSDB18Dataset(root_dir, subset=subset, segment_length=segment_seconds, sr=sr, n_fft=n_fft, hop_length=hop, win_length=win_length, center=center)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(subset == "train"), num_workers=num_workers)
