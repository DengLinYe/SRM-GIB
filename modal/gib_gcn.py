from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv


class GIB_GCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.5,
        temperature: float = 0.5,
        prior_keep_rate: float = 0.1,
        beta: float = 0.01,
        hard_threshold: float | None = None,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if prior_keep_rate <= 0 or prior_keep_rate >= 1:
            raise ValueError("prior_keep_rate must be in (0, 1).")
        if beta < 0:
            raise ValueError("beta must be non-negative.")
        if hard_threshold is not None and (hard_threshold <= 0 or hard_threshold >= 1):
            raise ValueError("hard_threshold must be in (0, 1) when provided.")

        self.encoder = GCNConv(in_channels, hidden_channels)
        self.classifier = GCNConv(hidden_channels, out_channels)
        self.edge_scorer = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )
        self.dropout = dropout
        self.temperature = temperature
        self.prior_keep_rate = prior_keep_rate
        self.beta = beta
        self.hard_threshold = hard_threshold

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        return_info: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x, edge_index, edge_weight=edge_weight)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        edge_prob = self.edge_probability(h, edge_index)
        sampled_weight = self.sample_edge_weight(edge_prob)
        logits = self.classifier(h, edge_index, edge_weight=sampled_weight)
        if return_info:
            return logits, edge_prob, sampled_weight
        return logits

    def edge_probability(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        edge_feat = torch.cat([h[src], h[dst]], dim=-1)
        return torch.sigmoid(self.edge_scorer(edge_feat)).view(-1)

    def sample_edge_weight(self, edge_prob: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.gumbel_sigmoid(edge_prob, self.temperature)
        if self.hard_threshold is not None:
            return (edge_prob > self.hard_threshold).to(edge_prob.dtype)
        return edge_prob

    def kl_loss(self, edge_prob: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(edge_prob.dtype).eps
        p = edge_prob.clamp(eps, 1 - eps)
        r = torch.as_tensor(self.prior_keep_rate, dtype=p.dtype, device=p.device)
        return (p * torch.log(p / r) + (1 - p) * torch.log((1 - p) / (1 - r))).mean()

    def total_loss(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
        edge_prob: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ce_loss = F.cross_entropy(logits[mask], y[mask])
        kl = self.kl_loss(edge_prob)
        return ce_loss + self.beta * kl, ce_loss, kl

    @staticmethod
    def gumbel_sigmoid(edge_prob: torch.Tensor, temperature: float) -> torch.Tensor:
        eps = torch.finfo(edge_prob.dtype).eps
        p = edge_prob.clamp(eps, 1 - eps)
        u = torch.rand_like(p).clamp(eps, 1 - eps)
        logistic_noise = torch.log(u) - torch.log1p(-u)
        logits = torch.log(p) - torch.log1p(-p)
        return torch.sigmoid((logits + logistic_noise) / temperature)
