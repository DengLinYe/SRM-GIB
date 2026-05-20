from __future__ import annotations

import csv
import io
import json
import sys
import zipfile
from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

SNAP_MEMBER_PREFIX = "twitch"
LANG_TO_SNAP_FOLDER: dict[str, str] = {
    "DE": "DE",
    "EN": "ENGB",
    "ES": "ES",
    "FR": "FR",
    "PT": "PTBR",
    "RU": "RU",
}


TWITCH_LANG: str = "PT"     # 所选语言子图
TRAIN_RATIO: float = 0.1    # 训练集比例
VAL_RATIO: float = 0.2      # 验证集比例
SPLIT_SEED: int = 42        # 随机种子
N_FEATS: int = 128          # 特征维度


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def twitch_zip_path() -> Path:
    return project_root() / "dataset" / "twitch_zip" / "twitch.zip"


def clean_twitch_dir(lang: str) -> Path:
    return project_root() / "dataset" / "clean" / f"twitch_{lang}"


def _feature_zip_member(snap_folder: str) -> str:
    if snap_folder == "DE":
        return f"{SNAP_MEMBER_PREFIX}/{snap_folder}/musae_{snap_folder}.json"
    return f"{SNAP_MEMBER_PREFIX}/{snap_folder}/musae_{snap_folder}_features.json"


def load_twitch_from_local_zip(zip_path: Path, lang: str) -> Data:
    if lang not in LANG_TO_SNAP_FOLDER:
        raise ValueError(f"Unsupported language code: {lang}")
    snap_folder = LANG_TO_SNAP_FOLDER[lang]
    edges_member = f"{SNAP_MEMBER_PREFIX}/{snap_folder}/musae_{snap_folder}_edges.csv"
    target_member = f"{SNAP_MEMBER_PREFIX}/{snap_folder}/musae_{snap_folder}_target.csv"
    feat_member = _feature_zip_member(snap_folder)

    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(target_member) as fp:
            target_rows = list(csv.DictReader(io.TextIOWrapper(fp, "utf-8")))
        with zf.open(feat_member) as fp:
            features = json.load(io.TextIOWrapper(fp, "utf-8"))
        with zf.open(edges_member) as fp:
            edge_reader = csv.reader(io.TextIOWrapper(fp, "utf-8"))
            header = next(edge_reader)
            if header[0].lower() == "from":
                edge_pairs = [[int(a), int(b)] for a, b in edge_reader]
            else:
                edge_pairs = [[int(a), int(b)] for a, b in edge_reader]

    n = len(target_rows)
    by_new_id: dict[int, dict[str, str]] = {}
    for row in target_rows:
        nid = int(row["new_id"])
        by_new_id[nid] = row
    if set(by_new_id.keys()) != set(range(n)):
        raise RuntimeError("Expected new_id to be a dense 0..n-1 index set.")

    x_list: list[list[float]] = []
    y_list: list[int] = []
    for i in range(n):
        row = by_new_id[i]
        raw = features.get(str(i), [])
        if len(raw) >= N_FEATS:
            vec = [float(v) for v in raw[:N_FEATS]]
        else:
            vec = [float(v) for v in raw] + [0.0] * (N_FEATS - len(raw))
        x_list.append(vec)
        mature = row["mature"].strip().lower()
        y_list.append(1 if mature in ("true", "1", "yes") else 0)

    x = torch.tensor(x_list, dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.long)
    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index)
    return Data(x=x, y=y, edge_index=edge_index)


def random_node_masks(
    num_nodes: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("require train_ratio > 0, val_ratio > 0, train_ratio + val_ratio < 1")
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_nodes, generator=generator)
    n_train = int(train_ratio * num_nodes)
    n_val = int(val_ratio * num_nodes)
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def main() -> int:
    lang = TWITCH_LANG.strip().upper()
    if lang not in LANG_TO_SNAP_FOLDER:
        print(
            f"Invalid TWITCH_LANG={TWITCH_LANG!r}; must be one of {sorted(LANG_TO_SNAP_FOLDER)}.",
            file=sys.stderr,
        )
        return 1

    zip_path = twitch_zip_path()
    if not zip_path.is_file():
        print(f"Missing zip file: {zip_path}", file=sys.stderr)
        print("See data/DATA_SOURCE.md for where to download twitch.zip and how to place it.", file=sys.stderr)
        return 1

    try:
        data = load_twitch_from_local_zip(zip_path, lang)
    except (OSError, KeyError, ValueError, zipfile.BadZipFile, RuntimeError) as exc:
        print(f"Failed to read Twitch/{lang} from zip: {exc}", file=sys.stderr)
        return 1

    n = data.num_nodes
    train_mask, val_mask, test_mask = random_node_masks(
        n, TRAIN_RATIO, VAL_RATIO, SPLIT_SEED
    )

    out_dir = clean_twitch_dir(lang)
    out_dir.mkdir(parents=True, exist_ok=True)

    graph_path = out_dir / "graph.pt"
    splits_path = out_dir / "splits.pt"
    y_1d = data.y.view(-1).long()

    torch.save(
        {
            "lang": lang,
            "x": data.x,
            "y": y_1d,
            "edge_index": data.edge_index,
        },
        graph_path,
    )
    torch.save(
        {
            "lang": lang,
            "seed": SPLIT_SEED,
            "train_mask": train_mask,
            "val_mask": val_mask,
            "test_mask": test_mask,
        },
        splits_path,
    )

    num_edges = int(data.edge_index.size(1))
    feat_dim = int(data.x.size(1)) if data.x is not None else 0

    print(f"Language: {lang}")
    print(f"Output directory: {out_dir}")
    print(f"Number of nodes: {n}")
    print(f"Number of edges (edge_index columns): {num_edges}")
    print(f"Node feature dimension: {feat_dim}")
    print(
        "Split counts - train / val / test: "
        f"{int(train_mask.sum())} / {int(val_mask.sum())} / {int(test_mask.sum())}"
    )
    print(
        "Split ratios - train / val / test: "
        f"{train_mask.float().mean():.4f} / {val_mask.float().mean():.4f} / "
        f"{test_mask.float().mean():.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
