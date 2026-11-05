"""Resample VoiceBank-DEMAND (48 kHz) to 16 kHz and lay it out for training.

Download and unzip the four archives from https://datashare.ed.ac.uk/handle/10283/2791
(clean/noisy trainset 28spk, clean/noisy testset) into one raw folder, then run:

    python scripts/prepare_data.py --raw-dir data/raw --out-dir data/vb16k
"""
import argparse
from pathlib import Path

import soundfile as sf
import torch
import torchaudio.functional as AF

SPLITS = {
    "train": ("clean_trainset_28spk_wav", "noisy_trainset_28spk_wav"),
    "test": ("clean_testset_wav", "noisy_testset_wav"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data/vb16k"))
    ap.add_argument("--sample-rate", type=int, default=16000)
    args = ap.parse_args()

    for split, (clean_name, noisy_name) in SPLITS.items():
        clean_dir, noisy_dir = args.raw_dir / clean_name, args.raw_dir / noisy_name
        files = sorted(p.name for p in clean_dir.glob("*.wav"))
        if not files:
            raise SystemExit(f"No wav files in {clean_dir}")
        for kind, src in (("clean", clean_dir), ("noisy", noisy_dir)):
            (args.out_dir / split / kind).mkdir(parents=True, exist_ok=True)
        for i, name in enumerate(files):
            for kind, src in (("clean", clean_dir), ("noisy", noisy_dir)):
                wav, sr = sf.read(src / name, dtype="float32")
                wav = AF.resample(torch.from_numpy(wav), sr, args.sample_rate)
                sf.write(args.out_dir / split / kind / name, wav.numpy(), args.sample_rate)
            if i % 500 == 0:
                print(f"{split}: {i}/{len(files)}")
        print(f"{split}: {len(files)} pairs done")


if __name__ == "__main__":
    main()
