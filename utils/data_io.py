from __future__ import annotations

from pathlib import Path

import torch

from data.graph_utils import normalize_features, normalize_features_with_clean_train_mask
from data.topology_attack import (
    ATTACK_SEED,
    ATTACK_SPLITS,
    BOT_BUDGET_RATIO,
    BOT_FEATURE_STD,
    FOLLOWERS_PER_VICTIM,
    FOLLOWER_INTRA_RANDOM_PAIR_COUNT,
    FOLLOWER_INTRA_RING,
    MAX_FOLLOWERS_AT_FULL_BUDGET,
    VICTIM_RATIO,
    VICTIM_SELECTION,
    inject_topology_attack,
    legacy_poisoned_twitch_dir,
    poisoned_twitch_dir,
)
from utils.config import TWITCH_LANG
from utils.split_utils import attach_split_fields


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def clean_twitch_dir(lang: str) -> Path:
    return project_root() / "dataset" / "clean" / f"twitch_{lang}"


def _bundle_bot_budget_ratio(bundle: dict) -> float:
    if "bot_budget_ratio" in bundle:
        return float(bundle["bot_budget_ratio"])
    return float(bundle.get("injected_edge_budget_ratio", -1.0))


def _normalize_attack_splits(splits: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(s.strip().lower() for s in splits)


def _bundle_attack_splits(bundle: dict) -> tuple[str, ...]:
    raw = bundle.get("attack_splits", ATTACK_SPLITS)
    return _normalize_attack_splits(tuple(raw))


def poison_cache_matches(
    bundle: dict,
    bot_budget_ratio: float,
) -> bool:
    if not isinstance(bundle, dict):
        return False
    if "x" not in bundle or "edge_index" not in bundle:
        return False
    if bundle.get("num_original_nodes") is None:
        return False
    if abs(_bundle_bot_budget_ratio(bundle) - bot_budget_ratio) >= 1e-6:
        return False
    if bundle.get("victim_selection") != VICTIM_SELECTION:
        return False
    if int(bundle.get("max_followers_at_full_budget", -1)) != MAX_FOLLOWERS_AT_FULL_BUDGET:
        return False
    if int(bundle.get("attack_seed", -1)) != ATTACK_SEED:
        return False
    if abs(float(bundle.get("victim_ratio", -1.0)) - VICTIM_RATIO) >= 1e-6:
        return False
    if _bundle_attack_splits(bundle) != _normalize_attack_splits(ATTACK_SPLITS):
        return False
    if abs(float(bundle.get("bot_feature_std", -1.0)) - BOT_FEATURE_STD) >= 1e-6:
        return False
    if bool(bundle.get("follower_intra_ring", None)) != FOLLOWER_INTRA_RING:
        return False
    if int(bundle.get("follower_intra_random_pair_count", -1)) != FOLLOWER_INTRA_RANDOM_PAIR_COUNT:
        return False
    if FOLLOWERS_PER_VICTIM is not None and bundle.get("followers_per_victim") != FOLLOWERS_PER_VICTIM:
        return False
    return True


def resolve_poison_cache_paths(lang: str, bot_budget_ratio: float) -> tuple[Path, Path]:
    primary = poisoned_twitch_dir(lang, bot_budget_ratio) / "poisoned_graph.pt"
    if primary.is_file():
        return primary.parent, primary
    legacy = legacy_poisoned_twitch_dir(lang, bot_budget_ratio) / "poisoned_graph.pt"
    if legacy.is_file():
        return legacy.parent, legacy
    return primary.parent, primary


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
    bot_budget_ratio: float = BOT_BUDGET_RATIO,
) -> dict[str, torch.Tensor]:
    out_dir, graph_path = resolve_poison_cache_paths(lang, bot_budget_ratio)
    if graph_path.is_file():
        bundle = torch.load(graph_path, weights_only=False)
        if poison_cache_matches(bundle, bot_budget_ratio):
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
        bot_budget_ratio=bot_budget_ratio,
        victim_selection=VICTIM_SELECTION,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, graph_path)
    torch.save(bundle["edge_index"], out_dir / "poisoned_edge_index.pt")
    torch.save(
        {
            "lang": lang,
            "attack_seed": ATTACK_SEED,
            "victim_ratio": VICTIM_RATIO,
            "followers_per_victim": FOLLOWERS_PER_VICTIM,
            "follower_intra_ring": FOLLOWER_INTRA_RING,
            "follower_intra_random_pair_count": FOLLOWER_INTRA_RANDOM_PAIR_COUNT,
            "attack_mode": "new_node_cross_label_intra_blend",
            "num_original_nodes": int(tensors["x"].size(0)),
            "num_bot_nodes": int(bundle["x"].size(0)) - int(tensors["x"].size(0)),
            "bot_feature_std": BOT_FEATURE_STD,
            "attack_splits": ATTACK_SPLITS,
            "bot_budget_ratio": bot_budget_ratio,
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
