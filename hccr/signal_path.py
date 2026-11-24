"""Learn window components, diffuse a series around its trend and reconstruct it."""
import math

import torch
from torch import nn

from .signal_blocks import AxisMix, ExtractComponent, WindowMix, initialize_filter


def layer_rates(depth, drop):
    """Hidden stages use dropout. The final stage is deterministic."""
    return [drop] * max(depth - 1, 0) + ([0.0] if depth > 0 else [])


def encode(layers, values):
    for layer in layers:
        values = layer(values)
    return values


class EmbedColumns(nn.Module):
    def __init__(self, inputs, width, *, normalize=False):
        super().__init__()
        self.conv = initialize_filter(nn.Conv1d(inputs, width, 1))
        self.norm = nn.LayerNorm(width) if normalize else nn.Identity()

    def forward(self, columns):
        return self.norm(self.conv(columns.transpose(1, 2)).transpose(1, 2))


class WindowPositions(nn.Module):
    def __init__(self, width, limit=5000):
        super().__init__()
        steps = torch.arange(limit, dtype=torch.float).unsqueeze(1)
        frequency = torch.exp(torch.arange(0, width, 2).float() * (-math.log(10000.0) / width))
        phase = steps * frequency
        table = torch.empty(limit, width)
        table[:, 0::2] = phase.sin()
        table[:, 1::2] = phase[:, :width // 2].cos()
        self.register_buffer("pe", table.unsqueeze(0))
        self.norm = nn.LayerNorm(width)

    def forward(self, values):
        return self.norm(self.pe[:, :values.size(1)])


class SensorEncoder(nn.Module):
    def __init__(self, span, width, hidden, key_width, sensors, depth, heads, drop):
        super().__init__()
        self.data_embedding = EmbedColumns(sensors, width)
        self.position_embedding = WindowPositions(width)
        self.encoder_blocks = nn.ModuleList(
            WindowMix(span, width, hidden, key_width, heads, rate)
            for rate in layer_rates(depth, drop)
        )

    def forward(self, values):
        return encode(self.encoder_blocks, self.data_embedding(values) + self.position_embedding(values))


class CalendarEncoder(nn.Module):
    def __init__(self, width, hidden, key_width, date_fields, depth, heads, drop):
        super().__init__()
        self.time_embed = EmbedColumns(date_fields, width, normalize=True)
        self.encoder_blocks = nn.ModuleList(
            AxisMix(width, hidden, key_width, heads, rate) for rate in layer_rates(depth, drop)
        )

    def forward(self, dates):
        return encode(self.encoder_blocks, self.time_embed(dates))


def nearby_indices(span, radius):
    # For ties, consider the centre first, then later times and finally earlier times.
    offsets = torch.arange(1, radius + 1)
    order = torch.cat((torch.zeros(1, dtype=torch.long), offsets, -offsets))
    return (torch.arange(span)[:, None] + order).clamp(0, span - 1)


def local_remainder(values, baseline, indices):
    batch, span, sensors = values.shape
    candidates = baseline[:, indices, :].transpose(-1, -2)
    differences = values.unsqueeze(-1) - candidates
    nearest = differences.abs().argmin(dim=-1, keepdim=True)
    return differences.gather(-1, nearest).reshape(batch, span, sensors)


class SplitSignal(nn.Module):
    def __init__(self, span, width, hidden, key_width, sensors, date_fields, depth, heads, drop, radius):
        super().__init__()
        self.data_encoder = SensorEncoder(span, width, hidden, key_width, sensors, depth, heads, drop)
        self.time_encoder = CalendarEncoder(width, hidden, key_width, date_fields, depth, heads, drop)
        self.decomposition_blocks = nn.ModuleList(
            ExtractComponent(width, hidden, key_width, sensors, heads, rate)
            for rate in layer_rates(depth, drop)
        )
        self.register_buffer("neighbors", nearby_indices(span, radius), persistent=False)

    def forward(self, values, dates):
        signal = self.data_encoder(values)
        calendar = self.time_encoder(dates)
        baseline = torch.zeros_like(values)
        for extract in self.decomposition_blocks:
            component, signal = extract(signal, calendar)
            baseline = baseline + component
        remainder = local_remainder(values, baseline, self.neighbors)
        trend = remainder.mean(dim=1, keepdim=True).expand_as(values)
        return baseline, trend


class RestoreSignal(nn.Module):
    def __init__(self, span, width, hidden, key_width, sensors, date_fields, depth, heads, drop):
        super().__init__()
        self.time_embed = EmbedColumns(date_fields, width, normalize=True)
        self.data_embedding = EmbedColumns(sensors, width)
        self.position_embedding = WindowPositions(width)
        self.decoder_blocks = nn.ModuleList(
            WindowMix(span, width, hidden, key_width, heads, rate)
            for rate in layer_rates(depth, drop)
        )
        self.fc1 = nn.Linear(width, sensors)

    def forward(self, noisy, trend, dates):
        background = self.data_embedding(trend)
        residual = self.data_embedding(noisy) - background
        tokens = residual + self.position_embedding(noisy) + self.time_embed(dates)
        return self.fc1(encode(self.decoder_blocks, tokens) + background)


class NoiseSchedule(nn.Module):
    def __init__(self, steps=1000, start=0.0001, stop=0.02, device="cpu"):
        super().__init__()
        beta = torch.linspace(start, stop, steps).float().to(device)
        retention = (1.0 - beta).cumprod(0)
        self.register_buffer("signal_gain", retention.sqrt(), persistent=False)
        self.register_buffer("noise_gain", (1.0 - retention).sqrt(), persistent=False)

    def q_sample(self, signal, trend, steps, noise):
        shape = (steps.size(0),) + (1,) * (signal.ndim - 1)
        keep = self.signal_gain[steps].reshape(shape)
        spread = self.noise_gain[steps].reshape(shape)
        return keep * signal + spread * noise + (1.0 - keep) * trend
