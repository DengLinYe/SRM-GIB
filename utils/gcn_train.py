from __future__ import annotations

import torch
from torch.nn import functional as F

from model.gcn import VanillaGCN
from utils.config import (
    DROPOUT,
    EPOCHS,
    HIDDEN_CHANNELS,
    LEARNING_RATE,
    PATIENCE,
    TRAIN_SEED,
    WEIGHT_DECAY,
)
from utils.supervision import build_supervision, class_weights


def accuracy(
    model: VanillaGCN,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)
        pred = logits.argmax(dim=1)
        correct = (pred[mask] == y[mask]).sum().item()
        total = int(mask.sum())
    return correct / total if total > 0 else 0.0


def train_gcn(
    tensors: dict[str, torch.Tensor],
    device: torch.device,
    tag: str,
    teacher_model: VanillaGCN | None = None,
    use_pseudo: bool = False,
    transductive_pseudo: bool = False,
    verbose: bool = True,
    hidden_channels: int | None = None,
    use_class_weights: bool = True,
) -> VanillaGCN:
    torch.manual_seed(TRAIN_SEED)
    x = tensors["x"].to(device)
    y = tensors["y"].to(device)
    edge_index = tensors["edge_index"].to(device)
    train_mask = tensors["train_mask"].to(device)
    val_mask = tensors["val_mask"].to(device)
    test_mask = tensors["test_mask"].to(device)

    supervision = build_supervision(
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        use_pseudo=use_pseudo,
        teacher=teacher_model,
        x=x,
        edge_index=edge_index,
        transductive_pseudo=transductive_pseudo,
    )
    if supervision.pseudo_added > 0 and verbose:
        print(f"[{tag}] Pseudo-labeling: Added {supervision.pseudo_added} unlabeled nodes.")

    hid = HIDDEN_CHANNELS if hidden_channels is None else hidden_channels
    model = VanillaGCN(
        in_channels=int(x.size(1)),
        hidden_channels=hid,
        out_channels=int(y.max().item()) + 1,
        dropout=DROPOUT,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    best_val = -1.0
    best_state = None
    stale_epochs = 0
    weight = (
        class_weights(supervision.y, supervision.loss_mask)
        if use_class_weights
        else None
    )

    if verbose:
        print(f"--- Train GCN on {tag} graph ---")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x, edge_index)
        loss = F.cross_entropy(
            logits[supervision.loss_mask],
            supervision.y[supervision.loss_mask],
            weight=weight,
        )
        loss.backward()
        optimizer.step()

        val_acc = accuracy(model, x, edge_index, y, val_mask)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        if verbose and (epoch == 1 or epoch % 20 == 0 or epoch == EPOCHS):
            train_acc = accuracy(model, x, edge_index, y, train_mask)
            print(
                f"[{tag}] Epoch {epoch:03d} | Loss {loss.item():.4f} | "
                f"Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f}"
            )

        if stale_epochs >= PATIENCE:
            if verbose:
                print(f"[{tag}] Early stop at epoch {epoch} (best val {best_val:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate_gcn(
    model: VanillaGCN,
    tensors: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    x = tensors["x"].to(device)
    y = tensors["y"].to(device)
    edge_index = tensors["edge_index"].to(device)
    train_mask = tensors["train_mask"].to(device)
    val_mask = tensors["val_mask"].to(device)
    test_mask = tensors["original_test_mask"].to(device)
    return {
        "train": accuracy(model, x, edge_index, y, train_mask),
        "val": accuracy(model, x, edge_index, y, val_mask),
        "test": accuracy(model, x, edge_index, y, test_mask),
    }
