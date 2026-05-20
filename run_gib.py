from __future__ import annotations

TWITCH_LANG: str = "PT"
HIDDEN_CHANNELS: int = 64
DROPOUT: float = 0.5
LEARNING_RATE: float = 0.01
WEIGHT_DECAY: float = 5e-4
EPOCHS: int = 200
TRAIN_SEED: int = 42
GIB_TEMPERATURE: float = 0.5
GIB_PRIOR_KEEP_RATE: float = 0.1
GIB_BETA: float = 0.01
GIB_HARD_THRESHOLD: float | None = 0.5

import torch
from torch.nn import functional as F

from modal.gcn import VanillaGCN
from modal.gib_gcn import GIB_GCN
from run_baseline import (
    build_or_load_poison_bundle,
    load_clean_tensors,
    poison_bundle_to_tensors,
)


def accuracy_vanilla(
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


def accuracy_gib(
    model: GIB_GCN,
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


def train_vanilla_on_poison(
    tensors: dict[str, torch.Tensor],
    device: torch.device,
) -> VanillaGCN:
    torch.manual_seed(TRAIN_SEED)
    x = tensors["x"].to(device)
    y = tensors["y"].to(device)
    edge_index = tensors["edge_index"].to(device)
    train_mask = tensors["train_mask"].to(device)
    val_mask = tensors["val_mask"].to(device)

    model = VanillaGCN(
        in_channels=int(x.size(1)),
        hidden_channels=HIDDEN_CHANNELS,
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

    print("--- Vanilla GCN on poisoned graph ---")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x, edge_index)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        val_acc = accuracy_vanilla(model, x, edge_index, y, val_mask)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 20 == 0 or epoch == EPOCHS:
            train_acc = accuracy_vanilla(model, x, edge_index, y, train_mask)
            print(
                f"[Vanilla] Epoch {epoch:03d} | L_CE {loss.item():.4f} | "
                f"Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_gib_on_poison(
    tensors: dict[str, torch.Tensor],
    device: torch.device,
) -> GIB_GCN:
    torch.manual_seed(TRAIN_SEED)
    x = tensors["x"].to(device)
    y = tensors["y"].to(device)
    edge_index = tensors["edge_index"].to(device)
    train_mask = tensors["train_mask"].to(device)
    val_mask = tensors["val_mask"].to(device)

    model = GIB_GCN(
        in_channels=int(x.size(1)),
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=int(y.max().item()) + 1,
        dropout=DROPOUT,
        temperature=GIB_TEMPERATURE,
        prior_keep_rate=GIB_PRIOR_KEEP_RATE,
        beta=GIB_BETA,
        hard_threshold=GIB_HARD_THRESHOLD,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    best_val = -1.0
    best_state = None

    print("--- GIB-GCN on poisoned graph ---")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        logits, edge_prob, _ = model(x, edge_index, return_info=True)
        loss, ce_loss, kl_loss = model.total_loss(logits, y, train_mask, edge_prob)
        loss.backward()
        optimizer.step()

        val_acc = accuracy_gib(model, x, edge_index, y, val_mask)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 20 == 0 or epoch == EPOCHS:
            train_acc = accuracy_gib(model, x, edge_index, y, train_mask)
            print(
                f"[GIB] Epoch {epoch:03d} | L_CE {ce_loss.item():.4f} | "
                f"L_KL {kl_loss.item():.4f} | Total {loss.item():.4f} | "
                f"Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate_poisoned(
    vanilla_model: VanillaGCN,
    gib_model: GIB_GCN,
    tensors: dict[str, torch.Tensor],
    device: torch.device,
) -> None:
    x = tensors["x"].to(device)
    y = tensors["y"].to(device)
    edge_index = tensors["edge_index"].to(device)
    train_mask = tensors["train_mask"].to(device)
    val_mask = tensors["val_mask"].to(device)
    test_mask = tensors["test_mask"].to(device)

    vanilla_metrics = {
        "train": accuracy_vanilla(vanilla_model, x, edge_index, y, train_mask),
        "val": accuracy_vanilla(vanilla_model, x, edge_index, y, val_mask),
        "test": accuracy_vanilla(vanilla_model, x, edge_index, y, test_mask),
    }
    gib_metrics = {
        "train": accuracy_gib(gib_model, x, edge_index, y, train_mask),
        "val": accuracy_gib(gib_model, x, edge_index, y, val_mask),
        "test": accuracy_gib(gib_model, x, edge_index, y, test_mask),
    }

    print("=== Poisoned graph evaluation ===")
    print(
        f"Vanilla GCN | Train {vanilla_metrics['train']:.4f} | "
        f"Val {vanilla_metrics['val']:.4f} | Test {vanilla_metrics['test']:.4f}"
    )
    print(
        f"GIB-GCN     | Train {gib_metrics['train']:.4f} | "
        f"Val {gib_metrics['val']:.4f} | Test {gib_metrics['test']:.4f}"
    )
    gain = gib_metrics["test"] - vanilla_metrics["test"]
    print(f"Test accuracy gain (GIB - Vanilla): {gain:.4f}")
    if gain > 0:
        print("GIB improves over Vanilla on poisoned test set.")
    else:
        print("GIB did not beat Vanilla on poisoned test set; try tuning GIB_BETA / GIB_PRIOR_KEEP_RATE.")


def main() -> int:
    lang = TWITCH_LANG.strip().upper()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean_tensors = load_clean_tensors(lang)
    poison_bundle = build_or_load_poison_bundle(clean_tensors, lang)
    poison_tensors = poison_bundle_to_tensors(poison_bundle)

    vanilla_model = train_vanilla_on_poison(poison_tensors, device)
    gib_model = train_gib_on_poison(poison_tensors, device)
    evaluate_poisoned(vanilla_model, gib_model, poison_tensors, device)
    print(f"Language: {lang}")
    print(f"Device: {device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
