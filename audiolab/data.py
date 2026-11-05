"""Paired noisy/clean speech from a prepared VoiceBank-DEMAND folder."""
import random
from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import Dataset


def list_pairs(data_dir, split):
    root = Path(data_dir) / split
    names = sorted(p.name for p in (root / "clean").glob("*.wav"))
    if not names:
        raise FileNotFoundError(f"No wav files in {root / 'clean'} (run scripts/prepare_data.py)")
    return [(root / "noisy" / n, root / "clean" / n) for n in names]


def load(path):
    wav, _ = sf.read(path, dtype="float32")
    return torch.from_numpy(wav)


class CropDataset(Dataset):
    """Random fixed-length crops for training; short files are zero-padded."""

    def __init__(self, data_dir, split, crop_samples, seed=0):
        self.pairs = list_pairs(data_dir, split)
        self.crop = crop_samples
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        noisy, clean = (load(p) for p in self.pairs[i])
        n = noisy.shape[-1]
        if n >= self.crop:
            start = self.rng.randint(0, n - self.crop)
            return noisy[start:start + self.crop], clean[start:start + self.crop]
        pad = self.crop - n
        return torch.nn.functional.pad(noisy, (0, pad)), torch.nn.functional.pad(clean, (0, pad))
