from __future__ import annotations

import torch


def normalize_features(x: torch.Tensor, train_mask: torch.Tensor) -> torch.Tensor:
    x = x.float()
    train_x = x[train_mask.bool()]
    if train_x.numel() == 0:
        return x
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0).clamp_min(1e-6)
    return (x - mean) / std


def normalize_features_with_clean_train_mask(
    x: torch.Tensor,
    clean_train_mask: torch.Tensor,
    num_original_nodes: int,
) -> torch.Tensor:
    x = x.float()
    mask = clean_train_mask.bool()
    train_x = x[:num_original_nodes][mask]
    if train_x.numel() == 0:
        return x
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0).clamp_min(1e-6)
    return (x - mean) / std
