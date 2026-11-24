"""Mix neighboring observations within patches and across a complete window."""

import math

import torch
from torch import nn


def parameter_pair(rows, cols):
    return nn.ParameterList([nn.Parameter(torch.empty(rows, cols)) for _ in range(2)])


def initialize_factors(*pairs):
    for pair in pairs:
        for weight in pair:
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))


class ChannelFactors(nn.Module):
    """Generate the two channel-specific maps for the local keys and values."""

    def __init__(self, channels, width):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(channels, 16))
        self.generator = nn.Sequential(
            nn.Linear(16, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 100)
        )
        self.P = parameter_pair(width, 10)
        self.Q = parameter_pair(10, width)
        self.B = parameter_pair(100, width)
        initialize_factors(self.P, self.Q, self.B)

    def forward(self):
        memory = self.generator(self.memory.unsqueeze(1))
        biases = [(memory @ basis).squeeze(1) for basis in self.B]
        matrix = memory.reshape(-1, 10, 10)
        weights = [(left @ matrix) @ right for left, right in zip(self.P, self.Q)]
        return weights, biases


class SharedMaps(nn.Module):
    """A pair of pointwise maps shared across all channels."""

    def __init__(self, width):
        super().__init__()
        self.P = parameter_pair(width, width)
        self.B = parameter_pair(1, width)
        initialize_factors(self.P)
        bound = 1 / math.sqrt(width)
        for bias in self.B:
            nn.init.uniform_(bias, -bound, bound)

    def forward(self):
        return self.P, self.B


def channel_map(values, weights, bias):
    return (values.unsqueeze(3) @ weights).squeeze(3) + bias


def split_local_heads(values):
    left, right = values.chunk(2, dim=-1)
    return torch.cat((left, right), dim=0).transpose(1, 2)


def summarize_patch(token, observed, channel_maps, shared_maps):
    """Attend from one learned token to itself and its observed patch."""
    keys_values = torch.cat((token, observed), dim=1)
    weights, biases = channel_maps
    queries = split_local_heads(token)
    keys = split_local_heads(channel_map(keys_values, weights[0], biases[0]))
    values = split_local_heads(channel_map(keys_values, weights[1], biases[1]))
    logits = queries @ keys.transpose(-2, -1)
    attention = (logits / math.sqrt(queries.size(-1))).softmax(dim=-1)
    result = (attention @ values).transpose(1, 2)
    result = torch.cat(result.chunk(2, dim=0), dim=-1)
    weights, biases = shared_maps
    result = channel_map(result, weights[0], biases[0]).relu()
    return channel_map(result, weights[1], biases[1])


class ScaledWeights(nn.Module):
    def __init__(self, depth):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(depth ** -0.5), requires_grad=False)

    def forward(self, queries, keys):
        return ((queries @ keys.transpose(-2, -1)) * self.scale).softmax(dim=-1)


class WindowAttention(nn.Module):
    def __init__(self, width):
        super().__init__()
        # One scalar per head for the one-step patches used by this model.
        self.heads = width
        self.W_Q = nn.Linear(width, width)
        self.W_K = nn.Linear(width, width)
        self.W_V = nn.Linear(width, width)
        self.sdp_attn = ScaledWeights(1)
        self.to_out = nn.Sequential(nn.Linear(width, width), nn.Dropout(0.1))

    def forward(self, series):
        batch, length, width = series.shape

        def heads(projection):
            return projection(series).view(batch, length, self.heads, 1).transpose(1, 2)

        weights = self.sdp_attn(heads(self.W_Q), heads(self.W_K))
        result = (weights @ heads(self.W_V)).transpose(1, 2)
        return self.to_out(result.contiguous().view(batch, length, width))


class SwapTimeAndFeatures(nn.Module):
    def forward(self, values):
        return values.transpose(1, 2)


def normalize_features(width):
    return nn.Sequential(SwapTimeAndFeatures(), nn.BatchNorm1d(width), SwapTimeAndFeatures())


class PatchMixer(nn.Module):
    def __init__(self, tokens, channels, width):
        super().__init__()
        # Registered by the training architecture and present in saved weights.
        self.start_fc = nn.Linear(1, width)
        self.intra_embeddings = nn.Parameter(torch.rand(tokens, 1, 1, channels, 16))
        self.embeddings_generator = nn.ModuleList(
            nn.Sequential(nn.Linear(16, width)) for _ in range(tokens)
        )
        self.weights_generator_distinct = ChannelFactors(channels, width)
        self.weights_generator_shared = SharedMaps(width)
        self.intra_Linear = nn.Linear(tokens, tokens)
        self.emb_linear = nn.Linear(width, width)
        position = torch.empty(tokens, width)
        self.W_pos = nn.Parameter(nn.init.uniform_(position, -0.02, 0.02))
        self.inter_patch_attention = WindowAttention(width)
        self.norm_attn = normalize_features(width)
        self.norm_ffn = normalize_features(width)
        self.dropout = nn.Dropout(0.1)
        self.ff = nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, width))

    def mix_neighbors(self, series):
        shared = self.weights_generator_shared()
        individual = self.weights_generator_distinct()
        summaries = []
        for i, project in enumerate(self.embeddings_generator):
            query = project(self.intra_embeddings[i]).expand(series.size(0), -1, -1, -1)
            summaries.append(summarize_patch(query, series[:, i:i+1], individual, shared))
        patches = torch.cat(summaries, dim=1).transpose(1, 3)
        return self.intra_Linear(patches).transpose(1, 3)

    def mix_window(self, series):
        batch, tokens, channels, width = series.shape
        sequence = series.transpose(1, 2).reshape(batch * channels, tokens, width)
        sequence = self.dropout(self.emb_linear(sequence) + self.W_pos)
        mixed = self.inter_patch_attention(sequence)
        return mixed.reshape(batch, tokens, channels, width)

    def forward(self, series):
        batch, tokens, channels, width = series.shape
        joined = series + self.mix_neighbors(series) + self.mix_window(series)
        flattened = self.norm_attn(joined.reshape(batch * channels, tokens, width))
        residual = self.dropout(flattened)
        return self.norm_ffn(self.ff(residual) + residual).reshape_as(series)
