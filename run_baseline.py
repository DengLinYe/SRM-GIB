from __future__ import annotations

TWITCH_LANG: str = "PT"  # Twitch 语言子图
HIDDEN_CHANNELS: int = 64  # 第一层 GCN 输出维度
DROPOUT: float = 0.5  # 第一层 ReLU 后 dropout 概率
LEARNING_RATE: float = 0.01  # Adam 学习率
WEIGHT_DECAY: float = 5e-4  # Adam L2 权重衰减系数
EPOCHS: int = 200  # 每个图上的训练轮数
TRAIN_SEED: int = 42  # 随机种子

from pathlib import Path

import torch
from torch.nn import functional as F

from data.topology_attack import (
    ATTACK_SEED,
    BOT_FEATURE_STD,
    FOLLOWERS_PER_VICTIM,
    FOLLOWER_INTRA_RANDOM_PAIR_COUNT,
    FOLLOWER_INTRA_RING,
    VICTIM_RATIO,
    inject_topology_attack,
)
from modal.gcn import VanillaGCN


def project_root() -> Path:
    return Path(__file__).resolve().parent


def clean_twitch_dir(lang: str) -> Path:
    return project_root() / "dataset" / "clean" / f"twitch_{lang}"


def poisoned_twitch_dir(lang: str) -> Path:
    return project_root() / "dataset" / "poisoned" / f"twitch_{lang}"


def load_clean_tensors(lang: str) -> dict[str, torch.Tensor]:
    clean_dir = clean_twitch_dir(lang)
    graph_path = clean_dir / "graph.pt"
    splits_path = clean_dir / "splits.pt"
    if not graph_path.is_file():
        raise FileNotFoundError(f"Missing graph file: {graph_path}")
    if not splits_path.is_file():
        raise FileNotFoundError(f"Missing splits file: {splits_path}")
    graph = torch.load(graph_path, weights_only=False)
    splits = torch.load(splits_path, weights_only=False)
    return {
        "x": graph["x"],
        "y": graph["y"].view(-1).long(),
        "edge_index": graph["edge_index"],
        "train_mask": splits["train_mask"],
        "val_mask": splits["val_mask"],
        "test_mask": splits["test_mask"],
    }


def build_or_load_poison_bundle(tensors: dict[str, torch.Tensor], lang: str) -> dict[str, torch.Tensor]:
    out_dir = poisoned_twitch_dir(lang)
    graph_path = out_dir / "poisoned_graph.pt"
    if graph_path.is_file():
        bundle = torch.load(graph_path, weights_only=False)
        if isinstance(bundle, dict) and "x" in bundle and "edge_index" in bundle:
            return bundle

    bundle = inject_topology_attack(
        x=tensors["x"],
        edge_index=tensors["edge_index"],
        train_mask=tensors["train_mask"],
        val_mask=tensors["val_mask"],
        test_mask=tensors["test_mask"],
        y=tensors["y"],
        num_nodes=int(tensors["x"].size(0)),
        victim_ratio=VICTIM_RATIO,
        followers_per_victim=FOLLOWERS_PER_VICTIM,
        seed=ATTACK_SEED,
        follower_intra_ring=FOLLOWER_INTRA_RING,
        follower_intra_random_pair_count=FOLLOWER_INTRA_RANDOM_PAIR_COUNT,
        bot_feature_std=BOT_FEATURE_STD,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, graph_path)
    torch.save(bundle["edge_index"], out_dir / "poisoned_edge_index.pt")
    torch.save(
        {
            "lang": lang,
            "seed": ATTACK_SEED,
            "victim_ratio": VICTIM_RATIO,
            "followers_per_victim": FOLLOWERS_PER_VICTIM,
            "follower_intra_ring": FOLLOWER_INTRA_RING,
            "follower_intra_random_pair_count": FOLLOWER_INTRA_RANDOM_PAIR_COUNT,
            "attack_mode": "new_node_cross_label_intra",
            "num_original_nodes": int(tensors["x"].size(0)),
            "num_bot_nodes": int(bundle["x"].size(0)) - int(tensors["x"].size(0)),
            "bot_feature_std": BOT_FEATURE_STD,
            "victims": bundle["victims"],
            "injected_edge_index": bundle["injected_edge_index"],
        },
        out_dir / "attack_meta.pt",
    )
    return bundle


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


def poison_bundle_to_tensors(bundle: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "x": bundle["x"],
        "y": bundle["y"].view(-1).long(),
        "edge_index": bundle["edge_index"],
        "train_mask": bundle["train_mask"],
        "val_mask": bundle["val_mask"],
        "test_mask": bundle["test_mask"],
    }


def train_model(
    tensors: dict[str, torch.Tensor],
    device: torch.device,
    tag: str,
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

    print(f"--- Train GCN on {tag} graph ---")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x, edge_index)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        val_acc = accuracy(model, x, edge_index, y, val_mask)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        if epoch == 1 or epoch % 20 == 0 or epoch == EPOCHS:
            train_acc = accuracy(model, x, edge_index, y, train_mask)
            print(
                f"[{tag}] Epoch {epoch:03d} | Loss {loss.item():.4f} | "
                f"Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate_on_graph(
    model: VanillaGCN,
    tensors: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    x = tensors["x"].to(device)
    y = tensors["y"].to(device)
    edge_index = tensors["edge_index"].to(device)
    train_mask = tensors["train_mask"].to(device)
    val_mask = tensors["val_mask"].to(device)
    test_mask = tensors["test_mask"].to(device)
    return {
        "train": accuracy(model, x, edge_index, y, train_mask),
        "val": accuracy(model, x, edge_index, y, val_mask),
        "test": accuracy(model, x, edge_index, y, test_mask),
    }


def main() -> int:
    lang = TWITCH_LANG.strip().upper()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean_tensors = load_clean_tensors(lang)
    poison_bundle = build_or_load_poison_bundle(clean_tensors, lang)
    poison_tensors = poison_bundle_to_tensors(poison_bundle)

    clean_model = train_model(clean_tensors, device, "clean")
    poison_model = train_model(poison_tensors, device, "poisoned")

    clean_metrics = evaluate_on_graph(clean_model, clean_tensors, device)
    poison_metrics = evaluate_on_graph(poison_model, poison_tensors, device)

    print("=== Results (each model on its own graph) ===")
    print(f"Language: {lang}")
    print(f"Device: {device}")
    print(
        f"Clean  | Train {clean_metrics['train']:.4f} | "
        f"Val {clean_metrics['val']:.4f} | Test {clean_metrics['test']:.4f}"
    )
    print(
        f"Poison | Train {poison_metrics['train']:.4f} | "
        f"Val {poison_metrics['val']:.4f} | Test {poison_metrics['test']:.4f}"
    )
    print(f"Test accuracy gap (clean - poisoned): {clean_metrics['test'] - poison_metrics['test']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
