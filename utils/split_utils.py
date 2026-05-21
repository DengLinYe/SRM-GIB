from __future__ import annotations

import torch


def unlabeled_mask(
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
) -> torch.Tensor:
    return ~(train_mask.bool() | val_mask.bool() | test_mask.bool())


def original_node_test_mask(
    test_mask: torch.Tensor,
    num_original_nodes: int | None,
) -> torch.Tensor:
    if num_original_nodes is None:
        return test_mask.bool()
    out = test_mask.bool().clone()
    if out.numel() > num_original_nodes:
        out[num_original_nodes:] = False
    return out


def attach_split_fields(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    train_mask = tensors["train_mask"]
    val_mask = tensors["val_mask"]
    test_mask = tensors["test_mask"]
    tensors["unlabeled_mask"] = unlabeled_mask(train_mask, val_mask, test_mask)
    tensors["original_test_mask"] = original_node_test_mask(
        test_mask,
        tensors.get("num_original_nodes"),
    )
    return tensors
