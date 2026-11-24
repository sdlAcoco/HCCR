"""HCCR decomposition, noise reconstruction and channel readout."""
from dataclasses import dataclass

import torch
from torch import nn

from .channel_groups import ChannelRouter, GroupProjection
from .signal_path import NoiseSchedule, RestoreSignal, SplitSignal
from .window_memory import WindowMemory


@dataclass(frozen=True)
class NetworkSpec:
    length: int = 64
    groups: int = 4
    readout: str = "grouped"
    width: int = 512
    hidden_width: int = 2048
    head_width: int = 64
    depth: int = 2
    heads: int = 8
    drop_rate: float = 0.6
    noise_steps: int = 1000
    noise_floor: float = 0.0001
    noise_ceiling: float = 0.02
    noise_level: int = 500
    shift_radius: int = 30

    def __post_init__(self):
        if self.readout not in ("grouped", "separate"):
            raise ValueError("readout must be grouped or separate")
        if min(self.length, self.groups, self.width, self.hidden_width,
               self.head_width, self.depth, self.heads) < 1:
            raise ValueError("network dimensions must be positive")
        if self.width % 2 or self.length > 5000:
            raise ValueError("position encoding requires even width and length <= 5000")
        if not 0 <= self.noise_level < self.noise_steps:
            raise ValueError("noise level must be a valid index into the schedule")
        if not 0 < self.noise_floor <= self.noise_ceiling < 1:
            raise ValueError("noise limits must fall between zero and one")
        if not 0 <= self.shift_radius <= self.length or not 0 <= self.drop_rate <= 1:
            raise ValueError("invalid shift radius or dropout probability")


def read_channels(layer, signals, routing):
    """Apply a time projection to each channel of a [batch, time, channel] series."""
    channels_first = signals.transpose(1, 2)
    if routing is None:
        projected = torch.stack([head(channels_first[:, i]) for i, head in enumerate(layer)], dim=1)
    else:
        projected = layer(channels_first, routing)
    return projected.transpose(1, 2)


class HCCR(nn.Module):
    def __init__(self, spec, channels, clock_features, device):
        super().__init__()
        self.spec = spec
        self.channels = channels
        self.clock_features = clock_features
        common = (spec.length, spec.width, spec.hidden_width, spec.head_width,
                  channels, clock_features, spec.depth, spec.heads, spec.drop_rate)
        self.separator = SplitSignal(*common, spec.shift_radius)
        self.schedule = NoiseSchedule(spec.noise_steps, spec.noise_floor, spec.noise_ceiling, device)
        self.restorer = RestoreSignal(*common)
        self.memory = WindowMemory(spec.length, channels)
        if spec.readout == "grouped":
            self.outputs = nn.ModuleList([
                GroupProjection(spec.groups, spec.length, spec.length)
                for _ in range(3)
            ])
            self.router = ChannelRouter(channels, spec.groups, spec.length, 64, device)
            self.register_buffer("centers", self.router.initial_centers, persistent=False)
        else:
            self.outputs = nn.ModuleList([nn.ModuleList() for _ in range(3)])
            for _ in range(channels):
                for head in self.outputs:
                    head.append(nn.Linear(spec.length, spec.length))
            self.router = None
        self.to(device)

    def forward(self, values, clock, shift_scale=0.0, refresh_groups=False):
        history = self.memory(values)
        routing = None
        if self.router is not None:
            routing, next_centers = self.router(values, self.centers)
            if refresh_groups:
                self.centers = nn.Parameter(next_centers)
        # The offset is sampled on the CPU, including when its amplitude is zero.
        offset = torch.rand(values.shape[0], 1, values.shape[2], device="cpu").to(values.device) * shift_scale
        shifted = values + offset
        seasonal, trend = self.separator(shifted, clock)
        step = torch.full((len(values),), self.spec.noise_level, device=values.device)
        noisy = self.schedule.q_sample(shifted, trend, step, torch.randn_like(shifted))
        recovered = self.restorer(noisy, trend, clock) + history
        seasonal, trend, recovered = (
            read_channels(head, signal, routing)
            for head, signal in zip(self.outputs, (seasonal, trend, recovered))
        )
        return seasonal, trend - offset, recovered - offset
