from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv


class WeightedSAGEConv(SAGEConv):
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            x_src = x_dst = x
        else:
            x_src, x_dst = x

        self._edge_weight = edge_weight
        out = self.propagate(edge_index, x=x_dst, size=size)
        out = self.lin_l(out)
        if self.root_weight:
            out = out + self.lin_r(x_src)
        return out

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        edge_weight = getattr(self, "_edge_weight", None)
        if edge_weight is None:
            return x_j
        return edge_weight.view(-1, 1) * x_j


class GIB_GCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.5,
        temperature: float = 1.0,
        prior_keep_rate: float = 0.5,
        beta_n: float = 0.05,
        beta_x: float = 0.001,
        hard_threshold: float | None = None,
        use_jaccard: bool = True,
        use_sage: bool = True,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if prior_keep_rate <= 0 or prior_keep_rate >= 1:
            raise ValueError("prior_keep_rate must be in (0, 1).")
        if beta_n < 0 or beta_x < 0:
            raise ValueError("beta must be non-negative.")
        if hard_threshold is not None and (hard_threshold <= 0 or hard_threshold >= 1):
            raise ValueError("hard_threshold must be in (0, 1) when provided.")

        self.mu_proj = nn.Linear(in_channels, hidden_channels)
        self.logstd_proj = nn.Linear(in_channels, hidden_channels)
        self.bn_mu = nn.BatchNorm1d(hidden_channels)

        edge_in_dim = hidden_channels * 2 + 3 + int(use_jaccard)
        self.edge_scorer = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )

        if use_sage:
            self.encoder = WeightedSAGEConv(hidden_channels, hidden_channels)
            self.classifier = WeightedSAGEConv(hidden_channels, out_channels)
        else:
            self.encoder = GCNConv(hidden_channels, hidden_channels)
            self.classifier = GCNConv(hidden_channels, out_channels)
        self.bn_enc = nn.BatchNorm1d(hidden_channels)

        self.dropout = dropout
        self.temperature = temperature
        self.prior_keep_rate = prior_keep_rate
        self.beta_n = beta_n
        self.beta_x = beta_x
        self.hard_threshold = hard_threshold
        self.use_jaccard = use_jaccard
        self.use_sage = use_sage
        self.reset_edge_scorer()

    def reset_edge_scorer(self) -> None:
        nn.init.xavier_uniform_(self.edge_scorer[0].weight)
        nn.init.zeros_(self.edge_scorer[0].bias)
        nn.init.zeros_(self.edge_scorer[2].weight)
        prior_logit = torch.logit(torch.tensor(float(self.prior_keep_rate)))
        nn.init.constant_(self.edge_scorer[2].bias, float(prior_logit))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_jaccard: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
        return_info: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.bn_mu(self.mu_proj(x))
        logstd = self.logstd_proj(x).clamp(-20, 2)

        if self.training:
            std = torch.exp(0.5 * logstd)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu

        kl_loss_x = -0.5 * torch.sum(1 + logstd - mu.pow(2) - logstd.exp(), dim=1).mean()

        edge_prob = self.edge_probability(z, x, edge_index, edge_jaccard)
        sampled_weight = self.sample_edge_weight(edge_prob)

        if edge_weight is not None:
            sampled_weight = sampled_weight * edge_weight

        h = self.encoder(z, edge_index, edge_weight=sampled_weight)
        h = self.bn_enc(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        logits = self.classifier(h, edge_index, edge_weight=sampled_weight)

        if return_info:
            return logits, edge_prob, sampled_weight, kl_loss_x
        return logits

    def edge_probability(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_jaccard: torch.Tensor | None = None,
    ) -> torch.Tensor:
        src, dst = edge_index
        degree = self.normalized_degree(
            edge_index,
            z.size(0),
            z.dtype,
            z.device,
        )
        feat_cos = F.cosine_similarity(x[src], x[dst], dim=-1).unsqueeze(-1)
        feature_blocks: list[torch.Tensor] = [
            z[src],
            z[dst],
            degree[src].unsqueeze(-1),
            degree[dst].unsqueeze(-1),
            feat_cos,
        ]
        if self.use_jaccard:
            if edge_jaccard is None:
                raise ValueError("edge_jaccard is required when use_jaccard=True.")
            jacc = edge_jaccard.to(z.device, dtype=z.dtype).unsqueeze(-1)
            feature_blocks.append(jacc)
        edge_feat = torch.cat(feature_blocks, dim=-1)
        return torch.sigmoid(self.edge_scorer(edge_feat)).view(-1)

    def sample_edge_weight(self, edge_prob: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.gumbel_sigmoid(edge_prob, self.temperature)
        if self.hard_threshold is not None:
            return (edge_prob > self.hard_threshold).to(edge_prob.dtype)
        return edge_prob

    def kl_loss_n(self, edge_prob: torch.Tensor) -> torch.Tensor:
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
        kl_loss_x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ce_loss = F.cross_entropy(logits[mask], y[mask])
        kl_n = self.kl_loss_n(edge_prob)
        total = ce_loss + self.beta_n * kl_n + self.beta_x * kl_loss_x
        return total, ce_loss, kl_n, kl_loss_x

    @staticmethod
    def gumbel_sigmoid(edge_prob: torch.Tensor, temperature: float) -> torch.Tensor:
        eps = torch.finfo(edge_prob.dtype).eps
        p = edge_prob.clamp(eps, 1 - eps)
        u = torch.rand_like(p).clamp(eps, 1 - eps)
        logistic_noise = torch.log(u) - torch.log1p(-u)
        logits = torch.log(p) - torch.log1p(-p)
        return torch.sigmoid((logits + logistic_noise) / temperature)

    @staticmethod
    def normalized_degree(
        edge_index: torch.Tensor,
        num_nodes: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        dst = edge_index[1]
        degree = torch.zeros(num_nodes, dtype=dtype, device=device)
        degree.scatter_add_(0, dst, torch.ones_like(dst, dtype=dtype, device=device))
        degree = torch.log1p(degree)
        max_degree = degree.max().clamp_min(torch.finfo(dtype).eps)
        return degree / max_degree
