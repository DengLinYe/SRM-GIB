from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.utils import to_undirected

TWITCH_LANG: str = "PT"
VICTIM_RATIO: float = 0.4 # 受害者比例
FOLLOWERS_PER_VICTIM: int = 50 # 每个受害者跟随者数量
ATTACK_SEED: int = 42 
FOLLOWER_INTRA_RING: bool = True # 组内连边
FOLLOWER_INTRA_RANDOM_PAIR_COUNT: int = 100 # 组内连边数
BOT_FEATURE_STD: float = 0.01 # 水军特征高斯扰动幅度


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def clean_twitch_dir(lang: str) -> Path:
    return project_root() / "dataset" / "clean" / f"twitch_{lang}"


def poisoned_twitch_dir(lang: str) -> Path:
    return project_root() / "dataset" / "poisoned" / f"twitch_{lang}"


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


def inject_topology_attack(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    y: torch.Tensor,
    num_nodes: int,
    victim_ratio: float,
    followers_per_victim: int,
    seed: int,
    follower_intra_ring: bool,
    follower_intra_random_pair_count: int,
    bot_feature_std: float,
) -> dict[str, torch.Tensor]:
    if victim_ratio <= 0 or victim_ratio > 1:
        raise ValueError("victim_ratio must be in (0, 1].")
    if followers_per_victim <= 0:
        raise ValueError("followers_per_victim must be positive.")

    device = edge_index.device
    x = x.to(device)
    y_flat = y.view(-1).long().to(device)
    if y_flat.numel() != num_nodes or int(x.size(0)) != num_nodes:
        raise ValueError("x rows and y length must match num_nodes.")
    num_classes = int(y_flat.max().item()) + 1

    generator = torch.Generator(device=device).manual_seed(seed)
    test_nodes = test_mask.nonzero(as_tuple=False).view(-1).to(device)
    victim_count = max(1, int(test_nodes.numel() * victim_ratio))
    victim_perm = torch.randperm(test_nodes.numel(), generator=generator, device=device)
    victims = test_nodes[victim_perm[:victim_count]]
    injected_blocks: list[torch.Tensor] = []
    new_x_rows: list[torch.Tensor] = []
    new_y_rows: list[torch.Tensor] = []
    next_new_id = num_nodes
    feat_dim = int(x.size(1))

    for victim in victims:
        victim = victim.view(()).long()
        k = followers_per_victim
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
            
        target_class_indices = (y_flat[:num_nodes] == bot_label).nonzero(as_tuple=False).view(-1)
        
        sampled_indices = target_class_indices[
            torch.randint(
                0, 
                target_class_indices.numel(), 
                (k,), 
                generator=generator, 
                device=device
            )
        ]
        
        bot_features = x[sampled_indices].clone()
        new_x_rows.append(bot_features)
        
        new_y_rows.append(torch.full((k,), bot_label, device=device, dtype=torch.long))

        victim_nodes = victim.expand(k)
        injected_blocks.append(torch.stack([victim_nodes, bot_ids], dim=0))
        injected_blocks.extend(
            _intra_follower_edge_blocks(
                bot_ids,
                generator,
                device,
                follower_intra_ring,
                follower_intra_random_pair_count,
            )
        )

    if not injected_blocks:
        injected_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        x_poison = x
        y_poison = y_flat
    else:
        injected_edge_index = torch.cat(injected_blocks, dim=1)
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
        "victims": victims,
        "x": x_poison,
        "y": y_poison,
        "train_mask": train_ext,
        "val_mask": val_ext,
        "test_mask": test_ext,
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
    )

    out_dir = poisoned_twitch_dir(lang)
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
            "followers_per_victim": FOLLOWERS_PER_VICTIM,
            "follower_intra_ring": FOLLOWER_INTRA_RING,
            "follower_intra_random_pair_count": FOLLOWER_INTRA_RANDOM_PAIR_COUNT,
            "attack_mode": "new_node_cross_label_intra",
            "num_original_nodes": num_nodes,
            "num_bot_nodes": int(bundle["x"].size(0)) - num_nodes,
            "bot_feature_std": BOT_FEATURE_STD,
            "victims": bundle["victims"],
            "injected_edge_index": bundle["injected_edge_index"],
        },
        meta_path,
    )

    print(f"Language: {lang}")
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
