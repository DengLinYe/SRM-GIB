from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from utils.split_utils import unlabeled_mask


PSEUDO_CONFIDENCE: float = 0.95


def class_weights(y: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    y_train = y[loss_mask]
    counts = torch.bincount(y_train, minlength=int(y.max().item()) + 1).float()
    counts = counts.clamp_min(1.0)
    return (counts.sum() / (counts * counts.numel())).to(y.device)


@dataclass
class SupervisionTargets:
    y: torch.Tensor
    loss_mask: torch.Tensor
    pseudo_added: int


def build_supervision(
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    use_pseudo: bool,
    teacher: nn.Module | None,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    pseudo_confidence: float = PSEUDO_CONFIDENCE,
    transductive_pseudo: bool = False,
) -> SupervisionTargets:
    pseudo_y = y.clone()
    loss_mask = train_mask.bool().clone()

    if not use_pseudo or teacher is None:
        return SupervisionTargets(y=pseudo_y, loss_mask=loss_mask, pseudo_added=0)

    if transductive_pseudo:
        candidate_mask = ~train_mask.bool() & ~test_mask.bool()
    else:
        candidate_mask = unlabeled_mask(train_mask, val_mask, test_mask)

    teacher.eval()
    with torch.no_grad():
        logits = teacher(x, edge_index)
        probs = F.softmax(logits, dim=1)
        max_probs, preds = probs.max(dim=1)

    confident = candidate_mask & (max_probs > pseudo_confidence)
    pseudo_y[confident] = preds[confident]
    loss_mask = loss_mask | confident
    return SupervisionTargets(
        y=pseudo_y,
        loss_mask=loss_mask,
        pseudo_added=int(confident.sum().item()),
    )
