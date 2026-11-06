"""Speech-enhancement metrics: SI-SDR, wideband PESQ, STOI, and streaming latency."""
import time

import numpy as np
import torch
from pesq import pesq
from pystoi import stoi


def si_sdr(est, ref, eps=1e-8):
    """Scale-invariant SDR in dB; works on (..., samples) tensors."""
    est = est - est.mean(-1, keepdim=True)
    ref = ref - ref.mean(-1, keepdim=True)
    target = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + eps) * ref
    noise = est - target
    return 10 * torch.log10(target.pow(2).sum(-1) / (noise.pow(2).sum(-1) + eps) + eps)


def speech_scores(est, ref, sr):
    """PESQ (wideband), STOI and SI-SDR for one utterance."""
    e, r = est.numpy().astype(np.float64), ref.numpy().astype(np.float64)
    return {
        "pesq": float(pesq(sr, r, e, "wb")),
        "stoi": float(stoi(r, e, sr, extended=False)),
        "si_sdr": float(si_sdr(est, ref)),
    }


@torch.no_grad()
def streaming_latency(model, sr, frames=2000):
    """Time one hop of streaming inference on one CPU thread, batch 1.

    Returns compute time per hop, the real-time factor (compute / hop duration;
    below 1 means faster than real time) and the algorithmic latency (one window).
    """
    model = model.to("cpu").eval()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)  # one core, closer to an embedded target than a many-core server
    bins = model.n_fft // 2 + 1
    frame = torch.rand(1, 1, bins)
    state = None
    for _ in range(50):  # warm-up
        _, state = model.mask(frame, state)
    start = time.perf_counter()
    for _ in range(frames):
        _, state = model.mask(frame, state)
    per_hop = (time.perf_counter() - start) / frames
    torch.set_num_threads(threads)
    hop_s = model.hop / sr
    return {
        "ms_per_hop": per_hop * 1e3,
        "rtf": per_hop / hop_s,
        "algorithmic_latency_ms": model.n_fft / sr * 1e3,
        "params": sum(p.numel() for p in model.parameters()),
    }
