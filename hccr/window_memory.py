"""Window-local historical response for [batch, time, channel] series."""

import torch
from torch import nn

from .patch_mixer import PatchMixer


def gather_neighbors(series, preceding=3):
    """Gather the original boundary-clamped triplet followed by its anchor."""
    length = series.size(1)
    time = torch.arange(length, device=series.device)
    offsets = torch.arange(preceding, device=series.device)
    starts = (time - preceding).clamp_min(0)
    indices = (starts[:, None] + offsets[None, :]).clamp_max(length - 1)
    return torch.cat((series[:, indices], series.unsqueeze(2)), dim=2)


class WindowMemory(nn.Module):
    def __init__(self, length, channels):
        super().__init__()
        tokens = 4 * length
        width = 16
        self.start_linear = nn.Linear(1, width)
        self.layers = nn.ModuleList([PatchMixer(tokens, channels, width)])
        self.projections = nn.Sequential(nn.Linear(tokens * width, length))

    def forward(self, series):
        batch, _, channels = series.shape
        neighbors = gather_neighbors(series).reshape(batch, -1, channels, 1)
        encoded = self.start_linear(neighbors)
        for mixer in self.layers:
            encoded = encoded + mixer(encoded)
        # Preserve the linearization order learned by the checkpoint's readout.
        window = self.projections(encoded.reshape(batch, channels, -1))
        return window.reshape(batch, -1, channels)
