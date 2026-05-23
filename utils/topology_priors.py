from __future__ import annotations

import torch
from torch_geometric.utils import to_undirected


def compute_edge_jaccard(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Per-edge Jaccard of neighbor sets (static topology prior, computed once)."""
    device = edge_index.device
    undirected = to_undirected(edge_index, num_nodes=num_nodes).cpu()
    neighbors: list[set[int]] = [set() for _ in range(num_nodes)]
    us = undirected[0].tolist()
    vs = undirected[1].tolist()
    for u, v in zip(us, vs):
        if u == v:
            continue
        neighbors[u].add(v)
        neighbors[v].add(u)

    src = edge_index[0].cpu().tolist()
    dst = edge_index[1].cpu().tolist()
    scores: list[float] = []
    for u, v in zip(src, dst):
        if u == v:
            scores.append(1.0)
            continue
        nu = neighbors[u]
        nv = neighbors[v]
        if not nu and not nv:
            scores.append(0.0)
            continue
        inter = len(nu & nv)
        union = len(nu | nv)
        scores.append(inter / union if union > 0 else 0.0)

    return torch.tensor(scores, dtype=torch.float32, device=device)
