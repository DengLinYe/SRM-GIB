from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.utils import to_undirected

TWITCH_LANG: str = "EN"
VICTIM_RATIO: float = 0.2
FOLLOWERS_PER_VICTIM: int | None = None
MAX_FOLLOWERS_AT_FULL_BUDGET: int = 30
BOT_BUDGET_RATIO: float = 1
VICTIM_SELECTION: str = "random"
ATTACK_SEED: int = 42
FOLLOWER_INTRA_RING: bool = True # 组内连边
FOLLOWER_INTRA_RANDOM_PAIR_COUNT: int = 100
BOT_FEATURE_STD: float = 0.05 # 水军特征高斯扰动幅度
ATTACK_SPLITS: tuple[str, ...] = ("train", "val", "test")


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def clean_twitch_dir(lang: str) -> Path:
    return project_root() / "dataset" / "clean" / f"twitch_{lang}"


def poisoned_twitch_dir(lang: str, bot_budget_ratio: float | None = None) -> Path:
    base = project_root() / "dataset" / "poisoned" / f"twitch_{lang}"
    if bot_budget_ratio is None:
        return base
    tag = f"bot_budget_{bot_budget_ratio:.4f}".replace(".", "p")
    return base / tag


def legacy_poisoned_twitch_dir(lang: str, bot_budget_ratio: float) -> Path:
    base = project_root() / "dataset" / "poisoned" / f"twitch_{lang}"
    tag = f"budget_{bot_budget_ratio:.4f}".replace(".", "p")
    return base / tag


def load_clean_data(lang: str) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    clean_dir = clean_twitch_dir(lang)
    graph_path = clean_dir / "graph.pt"
    splits_path = clean_dir / "splits.pt"
    if not graph_path.is_file():
        raise FileNotFoundError(f"Missing graph file: {graph_path}")
    if not splits_path.is_file():
        raise FileNotFoundError(f"Missing splits file: {splits_path}")
    graph = torch.load(graph_path, weights_only=False)
    splits = torch.load(splits_path, weights_only=False)
    return graph, splits


def _intra_follower_edge_blocks(
    followers: torch.Tensor,
    generator: torch.Generator,
    device: torch.device,
    ring: bool,
    num_random_pairs: int,
) -> list[torch.Tensor]:
    blocks: list[torch.Tensor] = []
    k = int(followers.numel())
    if k < 2:
        return blocks
    if ring:
        ring_src = followers
        ring_dst = torch.cat([followers[1:], followers[:1]], dim=0)
        blocks.append(torch.stack([ring_src, ring_dst], dim=0))
    if num_random_pairs > 0:
        triu_i, triu_j = torch.triu_indices(k, k, offset=1, device=device)
        pair_count = triu_i.numel()
        take = min(num_random_pairs, pair_count)
        perm = torch.randperm(pair_count, generator=generator, device=device)
        sel = perm[:take]
        u = followers[triu_i[sel]]
        v = followers[triu_j[sel]]
        blocks.append(torch.stack([u, v], dim=0))
    return blocks


def _node_degrees(edge_index: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
    deg = torch.zeros(num_nodes, dtype=torch.long, device=device)
    ones = torch.ones(edge_index.size(1), dtype=torch.long, device=device)
    deg.scatter_add_(0, edge_index[0], ones)
    deg.scatter_add_(0, edge_index[1], ones)
    return deg


def _select_victims(
    candidate_nodes: torch.Tensor,
    victim_count: int,
    edge_index: torch.Tensor,
    num_nodes: int,
    victim_selection: str,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    if victim_count >= candidate_nodes.numel():
        return candidate_nodes
    if victim_selection == "low_degree":
        deg = _node_degrees(edge_index, num_nodes, device)
        victim_deg = deg[candidate_nodes].float()
        order = torch.argsort(victim_deg)
        return candidate_nodes[order[:victim_count]]
    perm = torch.randperm(candidate_nodes.numel(), generator=generator, device=device)
    return candidate_nodes[perm[:victim_count]]


def _followers_per_victim_from_budget(
    bot_budget_ratio: float,
    followers_per_victim: int | None,
    max_followers_at_full_budget: int = MAX_FOLLOWERS_AT_FULL_BUDGET,
) -> int:
    if followers_per_victim is not None and followers_per_victim > 0:
        return followers_per_victim
    ratio = min(1.0, max(0.0, bot_budget_ratio))
    return max(1, int(round(max_followers_at_full_budget * ratio)))


def _attack_nodes_from_splits(
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    attack_splits: tuple[str, ...],
    device: torch.device,
) -> torch.Tensor:
    split_map = {
        "train": train_mask,
        "val": val_mask,
        "test": test_mask,
    }
    selected = []
    for split in attack_splits:
        key = split.strip().lower()
        if key not in split_map:
            raise ValueError(f"Unknown attack split: {split}")
        selected.append(split_map[key].to(device).bool())
    merged = torch.stack(selected, dim=0).any(dim=0)
    return merged.nonzero(as_tuple=False).view(-1)


def _sample_bot_features(
    x: torch.Tensor,
    y_flat: torch.Tensor,
    num_nodes: int,
    bot_label: int,
    count: int,
    victim_feat: torch.Tensor,
    generator: torch.Generator,
    device: torch.device,
    bot_feature_std: float,
) -> torch.Tensor:
    pool = (y_flat[:num_nodes] == bot_label).nonzero(as_tuple=False).view(-1)
    if pool.numel() == 0:
        pool = torch.arange(num_nodes, device=device)
    if pool.numel() >= count:
        perm = torch.randperm(pool.numel(), generator=generator, device=device)[:count]
        proto_idx = pool[perm]
    else:
        proto_idx = pool[
            torch.randint(0, pool.numel(), (count,), generator=generator, device=device)
        ]
    feats = x[proto_idx].clone()
    victim_feat = victim_feat.view(1, -1).to(device=device, dtype=feats.dtype)
    feats = 0.7 * feats + 0.3 * victim_feat
    if bot_feature_std > 0:
        feats = feats + torch.randn(
            feats.shape,
            generator=generator,
            device=device,
            dtype=feats.dtype,
        ) * bot_feature_std
    return feats


def inject_topology_attack(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    y: torch.Tensor,
    num_nodes: int,
    victim_ratio: float,
    followers_per_victim: int | None,
    seed: int,
    follower_intra_ring: bool,
    follower_intra_random_pair_count: int,
    bot_feature_std: float,
    attack_splits: tuple[str, ...],
    bot_budget_ratio: float = 1,
    victim_selection: str = "low_degree",
) -> dict[str, torch.Tensor]:
    if victim_ratio <= 0 or victim_ratio > 1:
        raise ValueError("victim_ratio must be in (0, 1].")
    if bot_budget_ratio <= 0:
        raise ValueError("bot_budget_ratio must be positive.")
    if victim_selection not in {"random", "low_degree"}:
        raise ValueError("victim_selection must be 'random' or 'low_degree'.")

    device = edge_index.device
    x = x.to(device)
    y_flat = y.view(-1).long().to(device)
    if y_flat.numel() != num_nodes or int(x.size(0)) != num_nodes:
        raise ValueError("x rows and y length must match num_nodes.")
    num_classes = int(y_flat.max().item()) + 1

    generator = torch.Generator(device=device).manual_seed(seed)
    candidate_nodes = _attack_nodes_from_splits(
        train_mask,
        val_mask,
        test_mask,
        attack_splits,
        device,
    )
    victim_count = max(1, int(candidate_nodes.numel() * victim_ratio))
    victims = _select_victims(
        candidate_nodes,
        victim_count,
        edge_index,
        num_nodes,
        victim_selection,
        generator,
        device,
    )
    k_per_victim = _followers_per_victim_from_budget(
        bot_budget_ratio,
        followers_per_victim,
    )
    injected_blocks: list[torch.Tensor] = []
    victim_bot_blocks: list[torch.Tensor] = []
    bot_bot_blocks: list[torch.Tensor] = []
    new_x_rows: list[torch.Tensor] = []
    new_y_rows: list[torch.Tensor] = []
    next_new_id = num_nodes

    for victim in victims:
        victim = victim.view(()).long()
        k = k_per_victim
        bot_ids = torch.arange(
            next_new_id,
            next_new_id + k,
            device=device,
            dtype=torch.long,
        )
        next_new_id += k
        vl = int(y_flat[victim].item())
        if num_classes == 2:
            bot_label = 1 - vl
        else:
            bot_label = (vl + 1) % num_classes

        bot_features = _sample_bot_features(
            x,
            y_flat,
            num_nodes,
            bot_label,
            k,
            x[victim],
            generator,
            device,
            bot_feature_std,
        )
        new_x_rows.append(bot_features)
        new_y_rows.append(torch.full((k,), bot_label, device=device, dtype=torch.long))

        victim_nodes = victim.expand(k)
        victim_bot_edges = torch.stack([victim_nodes, bot_ids], dim=0)
        intra_pairs = min(
            follower_intra_random_pair_count,
            max(0, k * (k - 1) // 2),
        )
        bot_bot_edges = _intra_follower_edge_blocks(
            bot_ids,
            generator,
            device,
            follower_intra_ring,
            intra_pairs,
        )
        injected_blocks.append(victim_bot_edges)
        injected_blocks.extend(bot_bot_edges)
        victim_bot_blocks.append(victim_bot_edges)
        bot_bot_blocks.extend(bot_bot_edges)

    if not injected_blocks:
        injected_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        victim_bot_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        bot_bot_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        x_poison = x
        y_poison = y_flat
    else:
        injected_edge_index = torch.cat(injected_blocks, dim=1)
        victim_bot_edge_index = torch.cat(victim_bot_blocks, dim=1)
        if bot_bot_blocks:
            bot_bot_edge_index = torch.cat(bot_bot_blocks, dim=1)
        else:
            bot_bot_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        x_poison = torch.cat([x] + new_x_rows, dim=0)
        y_poison = torch.cat([y_flat] + new_y_rows, dim=0)

    num_total = int(x_poison.size(0))
    poisoned_edge_index = torch.cat([edge_index.to(device), injected_edge_index], dim=1)
    poisoned_edge_index = to_undirected(poisoned_edge_index, num_nodes=num_total)

    num_new = num_total - num_nodes
    z = torch.zeros(num_new, dtype=torch.bool, device=device)
    train_ext = torch.cat([train_mask.to(device).bool(), z], dim=0)
    val_ext = torch.cat([val_mask.to(device).bool(), z], dim=0)
    test_ext = torch.cat([test_mask.to(device).bool(), z], dim=0)

    return {
        "edge_index": poisoned_edge_index,
        "injected_edge_index": injected_edge_index,
        "victim_bot_edge_index": victim_bot_edge_index,
        "bot_bot_edge_index": bot_bot_edge_index,
        "victims": victims,
        "x": x_poison,
        "y": y_poison,
        "train_mask": train_ext,
        "val_mask": val_ext,
        "test_mask": test_ext,
        "num_original_nodes": num_nodes,
        "followers_per_victim": k_per_victim,
        "bot_budget_ratio": bot_budget_ratio,
        "attack_seed": seed,
        "victim_ratio": victim_ratio,
        "attack_splits": tuple(attack_splits),
        "bot_feature_std": bot_feature_std,
        "follower_intra_ring": follower_intra_ring,
        "follower_intra_random_pair_count": follower_intra_random_pair_count,
        "victim_selection": victim_selection,
        "max_followers_at_full_budget": MAX_FOLLOWERS_AT_FULL_BUDGET,
    }


def main() -> int:
    lang = TWITCH_LANG.strip().upper()
    graph, splits = load_clean_data(lang)
    x = graph["x"]
    y = graph["y"].view(-1).long()
    edge_index = graph["edge_index"]
    train_mask = splits["train_mask"]
    val_mask = splits["val_mask"]
    test_mask = splits["test_mask"]
    num_nodes = int(x.size(0))

    bundle = inject_topology_attack(
        x=x,
        edge_index=edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        y=y,
        num_nodes=num_nodes,
        victim_ratio=VICTIM_RATIO,
        followers_per_victim=FOLLOWERS_PER_VICTIM,
        seed=ATTACK_SEED,
        follower_intra_ring=FOLLOWER_INTRA_RING,
        follower_intra_random_pair_count=FOLLOWER_INTRA_RANDOM_PAIR_COUNT,
        bot_feature_std=BOT_FEATURE_STD,
        attack_splits=ATTACK_SPLITS,
        bot_budget_ratio=BOT_BUDGET_RATIO,
        victim_selection=VICTIM_SELECTION,
    )

    out_dir = poisoned_twitch_dir(lang, BOT_BUDGET_RATIO)
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_path = out_dir / "poisoned_graph.pt"
    edge_only_path = out_dir / "poisoned_edge_index.pt"
    meta_path = out_dir / "attack_meta.pt"
    torch.save(bundle, graph_path)
    torch.save(bundle["edge_index"], edge_only_path)
    torch.save(
        {
            "lang": lang,
            "seed": ATTACK_SEED,
            "victim_ratio": VICTIM_RATIO,
            "followers_per_victim": bundle.get("followers_per_victim", FOLLOWERS_PER_VICTIM),
            "bot_budget_ratio": bundle.get(
                "bot_budget_ratio",
                bundle.get("injected_edge_budget_ratio", BOT_BUDGET_RATIO),
            ),
            "attack_seed": bundle.get("attack_seed", ATTACK_SEED),
            "victim_selection": bundle.get("victim_selection", VICTIM_SELECTION),
            "max_followers_at_full_budget": bundle.get(
                "max_followers_at_full_budget",
                MAX_FOLLOWERS_AT_FULL_BUDGET,
            ),
            "follower_intra_ring": FOLLOWER_INTRA_RING,
            "follower_intra_random_pair_count": FOLLOWER_INTRA_RANDOM_PAIR_COUNT,
            "attack_mode": "new_node_cross_label_intra_blend",
            "num_original_nodes": num_nodes,
            "num_bot_nodes": int(bundle["x"].size(0)) - num_nodes,
            "bot_feature_std": BOT_FEATURE_STD,
            "attack_splits": ATTACK_SPLITS,
            "victims": bundle["victims"],
            "injected_edge_index": bundle["injected_edge_index"],
            "victim_bot_edge_index": bundle["victim_bot_edge_index"],
            "bot_bot_edge_index": bundle["bot_bot_edge_index"],
        },
        meta_path,
    )

    print(f"Language: {lang}")
    print(f"Bot budget ratio: {BOT_BUDGET_RATIO}")
    print(f"Followers per victim: {bundle.get('followers_per_victim')}")
    print(f"Victim selection: {VICTIM_SELECTION}")
    print(f"Victims: {int(bundle['victims'].numel())}")
    print(f"Bot nodes added: {int(bundle['x'].size(0)) - num_nodes}")
    print(f"Injected directed edges: {int(bundle['injected_edge_index'].size(1))}")
    print(f"Clean edge_index columns: {int(edge_index.size(1))}")
    print(f"Poisoned edge_index columns: {int(bundle['edge_index'].size(1))}")
    print(f"Wrote: {graph_path}")
    print(f"Wrote: {edge_only_path}")
    print(f"Wrote: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
