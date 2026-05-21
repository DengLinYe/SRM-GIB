from __future__ import annotations

from pathlib import Path

import torch

from data.graph_utils import normalize_features, normalize_features_with_clean_train_mask
from data.topology_attack import (
    ATTACK_SEED,
    ATTACK_SPLITS,
    BOT_FEATURE_STD,
    FOLLOWERS_PER_VICTIM,
    FOLLOWER_INTRA_RANDOM_PAIR_COUNT,
    FOLLOWER_INTRA_RING,
    INJECTED_EDGE_BUDGET_RATIO,
    MAX_FOLLOWERS_AT_FULL_BUDGET,
    VICTIM_RATIO,
    VICTIM_SELECTION,
    inject_topology_attack,
)
from utils.config import TWITCH_LANG
from utils.split_utils import attach_split_fields


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def clean_twitch_dir(lang: str) -> Path:
    return project_root() / "dataset" / "clean" / f"twitch_{lang}"


def poisoned_twitch_dir(lang: str, budget_ratio: float | None = None) -> Path:
    base = project_root() / "dataset" / "poisoned" / f"twitch_{lang}"
    if budget_ratio is None:
        return base
    tag = f"budget_{budget_ratio:.4f}".replace(".", "p")
    return base / tag


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
    x = graph["x"]
    train_mask = splits["train_mask"]
    return attach_split_fields(
        {
            "x": normalize_features(x, train_mask),
            "y": graph["y"].view(-1).long(),
            "edge_index": graph["edge_index"],
            "train_mask": train_mask,
            "val_mask": splits["val_mask"],
            "test_mask": splits["test_mask"],
        }
    )


def build_or_load_poison_bundle(
    tensors: dict[str, torch.Tensor],
    lang: str,
    injected_edge_budget_ratio: float = INJECTED_EDGE_BUDGET_RATIO,
) -> dict[str, torch.Tensor]:
    out_dir = poisoned_twitch_dir(lang, injected_edge_budget_ratio)
    graph_path = out_dir / "poisoned_graph.pt"
    if graph_path.is_file():
        bundle = torch.load(graph_path, weights_only=False)
        if (
            isinstance(bundle, dict)
            and "x" in bundle
            and "edge_index" in bundle
            and bundle.get("num_original_nodes") is not None
            and abs(
                float(bundle.get("injected_edge_budget_ratio", -1.0))
                - injected_edge_budget_ratio
            )
            < 1e-6
            and bundle.get("victim_selection") == VICTIM_SELECTION
            and int(bundle.get("max_followers_at_full_budget", -1))
            == MAX_FOLLOWERS_AT_FULL_BUDGET
        ):
            return bundle

    graph = torch.load(clean_twitch_dir(lang) / "graph.pt", weights_only=False)
    raw_x = graph["x"]

    bundle = inject_topology_attack(
        x=raw_x,
        edge_index=tensors["edge_index"],
        train_mask=tensors["train_mask"],
        val_mask=tensors["val_mask"],
        test_mask=tensors["test_mask"],
        y=tensors["y"],
        num_nodes=int(raw_x.size(0)),
        victim_ratio=VICTIM_RATIO,
        followers_per_victim=FOLLOWERS_PER_VICTIM,
        seed=ATTACK_SEED,
        follower_intra_ring=FOLLOWER_INTRA_RING,
        follower_intra_random_pair_count=FOLLOWER_INTRA_RANDOM_PAIR_COUNT,
        bot_feature_std=BOT_FEATURE_STD,
        attack_splits=ATTACK_SPLITS,
        injected_edge_budget_ratio=injected_edge_budget_ratio,
        victim_selection=VICTIM_SELECTION,
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
            "attack_mode": "new_node_cross_label_intra_blend",
            "num_original_nodes": int(tensors["x"].size(0)),
            "num_bot_nodes": int(bundle["x"].size(0)) - int(tensors["x"].size(0)),
            "bot_feature_std": BOT_FEATURE_STD,
            "attack_splits": ATTACK_SPLITS,
            "injected_edge_budget_ratio": injected_edge_budget_ratio,
            "victim_selection": VICTIM_SELECTION,
            "max_followers_at_full_budget": MAX_FOLLOWERS_AT_FULL_BUDGET,
            "followers_per_victim": bundle.get("followers_per_victim"),
            "victims": bundle["victims"],
            "injected_edge_index": bundle["injected_edge_index"],
            "victim_bot_edge_index": bundle["victim_bot_edge_index"],
            "bot_bot_edge_index": bundle["bot_bot_edge_index"],
        },
        out_dir / "attack_meta.pt",
    )
    return bundle


def poison_bundle_to_tensors(
    bundle: dict[str, torch.Tensor],
    clean_train_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    num_orig = int(bundle["num_original_nodes"])
    x = normalize_features_with_clean_train_mask(
        bundle["x"],
        clean_train_mask,
        num_orig,
    )
    return attach_split_fields(
        {
            "x": x,
            "y": bundle["y"].view(-1).long(),
            "edge_index": bundle["edge_index"],
            "train_mask": bundle["train_mask"],
            "val_mask": bundle["val_mask"],
            "test_mask": bundle["test_mask"],
            "num_original_nodes": bundle.get("num_original_nodes"),
        }
    )


def load_default_lang() -> str:
    return TWITCH_LANG.strip().upper()
