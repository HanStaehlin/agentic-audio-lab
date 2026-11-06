"""Train and evaluate one speech-enhancement run.

    python -m audiolab.train --config configs/baseline.json --run-dir runs/baseline

Writes into the run directory:
    config.json   the exact config used
    status.json   state (running / done / failed), step, last loss, error
    train.log     one line per log interval
    metrics.json  test-set PESQ / STOI / SI-SDR (enhanced and noisy input) and latency
    model.pt      final weights
"""
import argparse
import json
import math
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from audiolab.data import CropDataset, list_pairs, load
from audiolab.metrics import si_sdr, speech_scores, streaming_latency
from audiolab.model import MaskGRU


def write_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)  # atomic, so readers never see half a file


def spectral_loss(est, ref, power=0.3):
    """Multi-resolution loss on power-law compressed magnitudes (closer to perception than SI-SDR)."""
    total = 0.0
    for n_fft, hop in ((256, 64), (512, 128), (1024, 256)):
        win = torch.hann_window(n_fft, device=est.device)
        e = torch.stft(est, n_fft, hop, window=win, return_complex=True).abs().clamp_min(1e-8).pow(power)
        r = torch.stft(ref, n_fft, hop, window=win, return_complex=True).abs().clamp_min(1e-8).pow(power)
        total = total + torch.nn.functional.mse_loss(e, r)
    return total / 3


def _score(args):
    est, noisy, clean, sr = args
    return speech_scores(torch.from_numpy(est), torch.from_numpy(clean), sr), \
        speech_scores(torch.from_numpy(noisy), torch.from_numpy(clean), sr)


def evaluate(model, cfg, device):
    """Enhance the test files on the GPU, then score them on all CPU cores (PESQ is slow)."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    import os

    model.eval()
    pairs = list_pairs(cfg["data_dir"], "test")[: cfg["eval_files"]]
    jobs = []
    with torch.no_grad():
        for noisy_path, clean_path in pairs:
            noisy, clean = load(noisy_path), load(clean_path)
            est = model(noisy.unsqueeze(0).to(device)).squeeze(0).cpu()
            jobs.append((est.numpy(), noisy.numpy(), clean.numpy(), cfg["sample_rate"]))
    with ProcessPoolExecutor(max_workers=os.cpu_count(), mp_context=mp.get_context("spawn")) as pool:
        scores = list(pool.map(_score, jobs, chunksize=8))
    enhanced, noisy_in = [s[0] for s in scores], [s[1] for s in scores]
    mean = lambda rows: {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    return {"enhanced": mean(enhanced), "noisy_input": mean(noisy_in), "eval_files": len(pairs)}


def train(cfg, run_dir):
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    status_path, log = run_dir / "status.json", open(run_dir / "train.log", "a")
    status = {"state": "running", "step": 0, "loss": None, "device": device, "started": time.time()}
    write_json(status_path, status)

    model = MaskGRU(cfg["n_fft"], cfg["hop"], cfg["hidden"], cfg["layers"], cfg.get("feature", "log1p")).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["steps"])
             if cfg.get("lr_schedule", "constant") == "cosine" else None)
    spec_weight = cfg.get("spec_weight", 0.0)
    data = CropDataset(cfg["data_dir"], "train", int(cfg["crop_seconds"] * cfg["sample_rate"]), cfg["seed"])
    loader = DataLoader(data, batch_size=cfg["batch_size"], shuffle=True, num_workers=2, drop_last=True)

    step, running = 0, []
    while step < cfg["steps"]:
        for noisy, clean in loader:
            model.train()
            noisy, clean = noisy.to(device), clean.to(device)
            est = model(noisy)
            loss = -si_sdr(est, clean).mean()
            if spec_weight:
                loss = loss + spec_weight * spectral_loss(est, clean)
            if not math.isfinite(loss.item()):
                raise FloatingPointError(f"loss is {loss.item()} at step {step}")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            if sched:
                sched.step()
            step += 1
            running.append(loss.item())
            if step % cfg["log_every"] == 0 or step == cfg["steps"]:
                avg = float(np.mean(running))
                running = []
                print(f"step {step} loss {avg:.3f} ({time.time() - status['started']:.0f}s)", file=log, flush=True)
                status.update(step=step, loss=avg)
                write_json(status_path, status)
            if step >= cfg["steps"]:
                break

    torch.save(model.state_dict(), run_dir / "model.pt")
    status["state"] = "evaluating"
    write_json(status_path, status)
    metrics = evaluate(model, cfg, device)
    metrics["latency"] = streaming_latency(model, cfg["sample_rate"])
    metrics["train_seconds"] = time.time() - status["started"]
    write_json(run_dir / "metrics.json", metrics)
    status["state"] = "done"
    write_json(status_path, status)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides as key=value (value parsed as JSON)")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    for kv in args.set:
        key, value = kv.split("=", 1)
        cfg[key] = json.loads(value)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.run_dir / "config.json", cfg)
    try:
        train(cfg, args.run_dir)
    except BaseException as exc:  # also Ctrl-C / Colab "interrupt", so the run never looks alive
        status_path = args.run_dir / "status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        status.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(status_path, status)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
