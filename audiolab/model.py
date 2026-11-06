"""A small GRU that predicts a spectral mask, frame by frame.

The GRU only looks at past frames, so the model can run as a stream: one STFT
frame in, one masked frame out. The algorithmic latency is one window (n_fft).
"""
import torch
from torch import nn


class MaskGRU(nn.Module):
    def __init__(self, n_fft=512, hop=128, hidden=256, layers=2, feature="log1p"):
        super().__init__()
        self.n_fft, self.hop, self.feature = n_fft, hop, feature
        bins = n_fft // 2 + 1
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.inp = nn.Linear(bins, hidden)
        self.gru = nn.GRU(hidden, hidden, layers, batch_first=True)
        self.out = nn.Linear(hidden, bins)

    def mask(self, logmag, state=None):
        """logmag: (B, frames, bins) -> mask in [0, 1] of the same shape."""
        h, state = self.gru(torch.relu(self.inp(logmag)), state)
        return torch.sigmoid(self.out(h)), state

    def features(self, mag):
        """log1p, or power-law compression (|X|^0.3), which is closer to perceived loudness."""
        return mag.clamp_min(1e-8).pow(0.3) if self.feature == "pow0.3" else torch.log1p(mag)

    def forward(self, wav):
        """wav: (B, samples) noisy -> (B, samples) enhanced."""
        spec = torch.stft(wav, self.n_fft, self.hop, window=self.window, return_complex=True)
        m, _ = self.mask(self.features(spec.abs()).transpose(1, 2))
        return torch.istft(spec * m.transpose(1, 2), self.n_fft, self.hop,
                           window=self.window, length=wav.shape[-1])
