"""Soft channel grouping and group-specific linear readouts."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def group_weights(similarity, temperature):
    """Normalize across groups for each channel."""
    weights = (similarity / temperature).exp()
    return weights / weights.sum(dim=-1, keepdim=True)


def sample_membership(probability):
    """Draw a continuous Bernoulli mask for the center update."""
    jitter = torch.empty_like(probability).uniform_(1e-10, 1 - 1e-10)
    noise = jitter.log() - (1 - jitter).log()
    logits = (probability + 1e-10).log() - (1 - probability + 1e-10).log()
    return ((logits + noise) / 0.07).sigmoid()


class ChannelRouter(nn.Module):
    def __init__(self, channels, groups, length, width, device, temperature=0.05):
        super().__init__()
        self.channels = channels
        self.temperature = temperature
        self.linear = nn.Linear(length, width)
        # Legacy weights contain the learned projection, but no initial centers.
        centers = torch.empty(groups, width).to(device)
        nn.init.kaiming_uniform_(centers, a=math.sqrt(5))
        self.register_buffer("initial_centers", centers, persistent=False)

    def forward(self, series, centers):
        batch, _, channels = series.shape
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} channels, received {channels}")
        encoded = self.linear(series.transpose(1, 2)).reshape(batch * channels, -1)
        similarity = torch.mm(F.normalize(encoded, dim=1), F.normalize(centers, dim=1).T)
        sampled = sample_membership(group_weights(similarity, self.temperature))
        updates = (encoded.T @ sampled) / (sampled.sum(dim=0) + 1e-6)
        mean_similarity = similarity.reshape(batch, channels, -1).mean(dim=0)
        return group_weights(mean_similarity, self.temperature), centers + updates.T


class GroupProjection(nn.Module):
    def __init__(self, groups, input_length, output_length):
        super().__init__()
        self.linears = nn.ModuleList(nn.Linear(input_length, output_length) for _ in range(groups))

    def forward(self, series, weights):
        responses = torch.stack([projection(series) for projection in self.linears], dim=-1)
        return (responses @ weights.unsqueeze(-1)).squeeze(-1)
