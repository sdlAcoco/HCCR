"""Window attention and feed-forward blocks shared by the signal paths.

Layer attribute names are part of the on-disk checkpoint schema.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def split_heads(project, tokens, heads):
    batch, length, _ = tokens.shape
    return project(tokens).reshape(batch, length, heads, -1).transpose(1, 2)


def affinity(queries, keys):
    return (queries @ keys.transpose(-1, -2)) / math.sqrt(queries.size(-1))


def mix_values(logits, values):
    context = logits.softmax(dim=-1) @ values
    return context.transpose(1, 2).flatten(start_dim=2)


class TokenMix(nn.Module):
    """Attend along one axis of a window or jointly match its values and dates."""

    def __init__(self, width, key_width, heads, drop, *, use_dates=False):
        super().__init__()
        self.heads = heads
        projected = heads * key_width
        projection_names = (
            ("W_Q_data", "W_Q_time", "W_K_data", "W_K_time", "W_V_time")
            if use_dates else ("W_Q", "W_K", "W_V")
        )
        for name in projection_names:
            self.add_module(name, nn.Linear(width, projected))
        self.fc = nn.Linear(projected, width)
        self.dropout = nn.Dropout(drop)
        self.norm = nn.LayerNorm(width)

    def forward(self, tokens, dates=None):
        if dates is None:
            query = split_heads(self.W_Q, tokens, self.heads)
            key = split_heads(self.W_K, tokens, self.heads)
            value = split_heads(self.W_V, tokens, self.heads)
            mixed = mix_values(affinity(query, key), value)
            return self.norm(self.dropout(self.fc(mixed)) + tokens)

        data_sim = affinity(split_heads(self.W_Q_data, tokens, self.heads),
                            split_heads(self.W_K_data, tokens, self.heads))
        time_sim = affinity(split_heads(self.W_Q_time, dates, self.heads),
                            split_heads(self.W_K_time, dates, self.heads))
        mixed = mix_values(data_sim + time_sim, split_heads(self.W_V_time, dates, self.heads))
        return self.norm(self.dropout(self.fc(mixed)))


def initialize_filter(layer):
    nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="leaky_relu")
    return layer


def make_filters(inputs, hidden, outputs):
    first = nn.Conv1d(inputs, hidden, 1)
    second = nn.Conv1d(hidden, outputs, 1)
    return initialize_filter(first), initialize_filter(second)


def transform_tokens(tokens, first, second, drop):
    hidden = F.gelu(first(tokens.transpose(1, 2)))
    return drop(second(hidden).transpose(1, 2))


class AxisMix(nn.Module):
    def __init__(self, width, hidden, key_width, heads, drop, *, feature_span=None):
        super().__init__()
        self.across_features = feature_span is not None
        self.attention = TokenMix(feature_span if self.across_features else width,
                                  key_width, heads, drop)
        self.conv1, self.conv2 = make_filters(width, hidden, width)
        self.dropout = nn.Dropout(drop)
        self.norm = nn.LayerNorm(width)

    def forward(self, tokens):
        sequence = tokens.transpose(1, 2) if self.across_features else tokens
        mixed = self.attention(sequence)
        if self.across_features:
            mixed = mixed.transpose(1, 2)
        return self.norm(transform_tokens(mixed, self.conv1, self.conv2, self.dropout) + mixed)


class WindowMix(nn.Module):
    def __init__(self, span, width, hidden, key_width, heads, drop):
        super().__init__()
        self.time_block = AxisMix(width, hidden, key_width, heads, drop)
        self.feature_block = AxisMix(width, hidden, key_width, heads, drop, feature_span=span)
        self.conv1, self.conv2 = make_filters(2 * width, hidden, width)
        self.dropout = nn.Dropout(drop)
        self.norm1 = nn.LayerNorm(2 * width)
        self.norm2 = nn.LayerNorm(width)

    def forward(self, tokens):
        joint = torch.cat((self.time_block(tokens), self.feature_block(tokens)), dim=-1)
        encoded = transform_tokens(self.norm1(joint), self.conv1, self.conv2, self.dropout)
        return self.norm2(encoded)


class ExtractComponent(nn.Module):
    def __init__(self, width, hidden, key_width, sensors, heads, drop):
        super().__init__()
        self.mixed_attention = TokenMix(width, key_width, heads, drop, use_dates=True)
        self.ordinary_attention = TokenMix(width, key_width, heads, drop)
        self.conv1, self.conv2 = make_filters(width, hidden, width)
        self.fc1 = nn.Linear(width, hidden)
        self.fc2 = nn.Linear(hidden, sensors)
        self.dropout = nn.Dropout(drop)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)

    def forward(self, signal, dates):
        tokens = self.ordinary_attention(self.mixed_attention(signal, dates))
        component = self.norm1(transform_tokens(tokens, self.conv1, self.conv2, self.dropout) + tokens)
        remainder = self.norm2(signal - component)
        return self.fc2(F.gelu(self.fc1(component))), remainder
