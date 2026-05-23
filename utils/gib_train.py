from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from modal.rm_gib import RM_GIB
from modal.srm_gib import GIB_GCN
from utils.config import (
    DROPOUT,
    EPOCHS,
    GIB_BETA_N,
    GIB_BETA_X,
    GIB_PRIOR_KEEP_RATE,
    GIB_SSL_JACCARD_THRESHOLD,
    GIB_TEMPERATURE,
    HIDDEN_CHANNELS,
    LEARNING_RATE,
    PATIENCE,
    TRAIN_SEED,
    WEIGHT_DECAY,
)
from utils.supervision import build_supervision, class_weights
from utils.topology_priors import compute_edge_jaccard

GIBVariant = str
VALID_GIB_VARIANTS: tuple[GIBVariant, ...] = (
    "rm_gib",
    "srm_gib",
    "srm_gib_no_jaccard",
    "srm_gib_gcn",
)

VARIANT_LABELS: dict[str, str] = {
    "rm_gib": "RM-GIB (GCN + legacy edge scorer, cosine SSL)",
    "srm_gib": "Social-RM-GIB (SAGE + Jaccard edge/SSL)",
    "srm_gib_no_jaccard": "SRM-GIB w/o Jaccard (SAGE + cosine SSL)",
    "srm_gib_gcn": "SRM-GIB w/ GCN backbone (Jaccard edge/SSL)",
}


def _normalize_variant(model_variant: str) -> str:
    if model_variant not in VALID_GIB_VARIANTS:
        valid = ", ".join(VALID_GIB_VARIANTS)
        raise ValueError(f"Unknown model_variant={model_variant!r}. Expected one of: {valid}.")
    return model_variant


def variant_needs_jaccard(model_variant: str) -> bool:
    return _normalize_variant(model_variant) in ("srm_gib", "srm_gib_gcn")


def variant_ssl_uses_jaccard(model_variant: str) -> bool:
    return _normalize_variant(model_variant) in ("srm_gib", "srm_gib_gcn")


def _build_model(
    model_variant: str,
    in_channels: int,
    out_channels: int,
    device: torch.device,
) -> nn.Module:
    variant = _normalize_variant(model_variant)
    common_kwargs = dict(
        in_channels=in_channels,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=out_channels,
        dropout=DROPOUT,
        temperature=GIB_TEMPERATURE,
        prior_keep_rate=GIB_PRIOR_KEEP_RATE,
        beta_n=GIB_BETA_N,
        beta_x=GIB_BETA_X,
        hard_threshold=None,
    )
    if variant == "rm_gib":
        return RM_GIB(**common_kwargs).to(device)
    if variant == "srm_gib":
        return GIB_GCN(**common_kwargs, use_jaccard=True, use_sage=True).to(device)
    if variant == "srm_gib_no_jaccard":
        return GIB_GCN(**common_kwargs, use_jaccard=False, use_sage=True).to(device)
    return GIB_GCN(**common_kwargs, use_jaccard=True, use_sage=False).to(device)


def _model_forward(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_jaccard: torch.Tensor | None,
    return_info: bool = False,
):
    if isinstance(model, RM_GIB):
        return model(x, edge_index, return_info=return_info)
    if isinstance(model, GIB_GCN) and not model.use_jaccard:
        return model(x, edge_index, edge_jaccard=None, return_info=return_info)
    return model(x, edge_index, edge_jaccard, return_info=return_info)


def _build_ssl_target(
    model_variant: str,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_jaccard: torch.Tensor | None,
) -> torch.Tensor:
    variant = _normalize_variant(model_variant)
    if variant_ssl_uses_jaccard(variant):
        if edge_jaccard is None:
            raise ValueError("edge_jaccard is required for Jaccard-based SSL.")
        if GIB_SSL_JACCARD_THRESHOLD is None:
            threshold = edge_jaccard.mean()
        else:
            threshold = torch.as_tensor(
                GIB_SSL_JACCARD_THRESHOLD,
                dtype=edge_jaccard.dtype,
                device=edge_jaccard.device,
            )
        return (edge_jaccard > threshold).float()
    src, dst = edge_index
    raw_sim = F.cosine_similarity(x[src], x[dst], dim=-1)
    return (raw_sim > raw_sim.mean()).float()


def accuracy_gib(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_jaccard: torch.Tensor | None,
    y: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    model.eval()
    with torch.no_grad():
        logits = _model_forward(model, x, edge_index, edge_jaccard, return_info=False)
        pred = logits.argmax(dim=1)
        correct = (pred[mask] == y[mask]).sum().item()
        total = int(mask.sum())
    return correct / total if total > 0 else 0.0


def train_gib(
    tensors: dict[str, torch.Tensor],
    device: torch.device,
    tag: str,
    teacher_model: nn.Module | None = None,
    use_pseudo: bool = False,
    use_ssl: bool = False,
    transductive_pseudo: bool = False,
    verbose: bool = True,
    model_variant: str = "srm_gib",
    train_seed: int | None = None,
) -> nn.Module:
    variant = _normalize_variant(model_variant)
    seed = TRAIN_SEED if train_seed is None else train_seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    x = tensors["x"].to(device)
    y = tensors["y"].to(device)
    edge_index = tensors["edge_index"].to(device)
    train_mask = tensors["train_mask"].to(device)
    val_mask = tensors["val_mask"].to(device)
    test_mask = tensors["test_mask"].to(device)

    edge_jaccard = (
        compute_edge_jaccard(edge_index, int(x.size(0)))
        if variant_needs_jaccard(variant)
        else None
    )

    supervision = build_supervision(
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        use_pseudo=use_pseudo,
        teacher=teacher_model if use_pseudo else None,
        x=x,
        edge_index=edge_index,
        edge_jaccard=edge_jaccard,
        transductive_pseudo=transductive_pseudo,
    )
    if supervision.pseudo_added > 0 and verbose:
        print(f"[{tag}] Pseudo-labeling: Added {supervision.pseudo_added} unlabeled nodes.")

    model = _build_model(
        variant,
        in_channels=int(x.size(1)),
        out_channels=int(y.max().item()) + 1,
        device=device,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val = -1.0
    best_state = None
    stale_epochs = 0
    weight = class_weights(supervision.y, supervision.loss_mask)

    if verbose:
        print(
            f"--- Train {variant} ({tag}) | pseudo={use_pseudo} ssl={use_ssl} "
            f"transductive={transductive_pseudo} ---"
        )
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        logits, edge_prob, _, kl_loss_x = _model_forward(
            model,
            x,
            edge_index,
            edge_jaccard,
            return_info=True,
        )
        ce_loss = F.cross_entropy(
            logits[supervision.loss_mask],
            supervision.y[supervision.loss_mask],
            weight=weight,
        )
        kl_loss_n = model.kl_loss_n(edge_prob)
        loss = ce_loss + model.beta_n * kl_loss_n + model.beta_x * kl_loss_x

        ssl_loss = edge_prob.new_zeros(())
        if use_ssl:
            ssl_target = _build_ssl_target(variant, x, edge_index, edge_jaccard)
            ssl_loss = F.binary_cross_entropy(edge_prob, ssl_target)
            loss = loss + ssl_loss

        loss.backward()
        optimizer.step()

        val_acc = accuracy_gib(model, x, edge_index, edge_jaccard, y, val_mask)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        if verbose and (epoch == 1 or epoch % 20 == 0 or epoch == EPOCHS):
            train_acc = accuracy_gib(model, x, edge_index, edge_jaccard, y, train_mask)
            msg = (
                f"[{tag}] Epoch {epoch:03d} | L_CE {ce_loss.item():.4f} | "
                f"L_KL_N {kl_loss_n.item():.4f} | L_KL_X {kl_loss_x.item():.4f}"
            )
            if use_ssl:
                msg += f" | L_SSL {ssl_loss.item():.4f}"
            msg += f" | Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f}"
            print(msg)

        if stale_epochs >= PATIENCE:
            if verbose:
                print(f"[{tag}] Early stop at epoch {epoch} (best val {best_val:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate_gib(
    model: nn.Module,
    tensors: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    x = tensors["x"].to(device)
    y = tensors["y"].to(device)
    edge_index = tensors["edge_index"].to(device)
    train_mask = tensors["train_mask"].to(device)
    val_mask = tensors["val_mask"].to(device)
    test_mask = tensors["original_test_mask"].to(device)
    edge_jaccard = (
        compute_edge_jaccard(edge_index, int(x.size(0)))
        if isinstance(model, GIB_GCN) and model.use_jaccard
        else None
    )
    return {
        "train": accuracy_gib(model, x, edge_index, edge_jaccard, y, train_mask),
        "val": accuracy_gib(model, x, edge_index, edge_jaccard, y, val_mask),
        "test": accuracy_gib(model, x, edge_index, edge_jaccard, y, test_mask),
    }


def _edge_membership_mask(
    edge_index: torch.Tensor,
    target_edges: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    if target_edges.numel() == 0:
        return torch.zeros(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
    target_edges = target_edges.to(edge_index.device)
    edge_code = edge_index[0] * num_nodes + edge_index[1]
    target_code = target_edges[0] * num_nodes + target_edges[1]
    reverse_code = target_edges[1] * num_nodes + target_edges[0]
    target_code = torch.unique(torch.cat([target_code, reverse_code], dim=0))
    return torch.isin(edge_code, target_code)


def edge_probability_diagnostics(
    model: nn.Module,
    tensors: dict[str, torch.Tensor],
    device: torch.device,
    tag: str,
    victim_bot_edge_index: torch.Tensor | None = None,
    bot_bot_edge_index: torch.Tensor | None = None,
) -> None:
    model.eval()
    x = tensors["x"].to(device)
    edge_index = tensors["edge_index"].to(device)
    edge_jaccard = compute_edge_jaccard(edge_index, int(x.size(0)))
    with torch.no_grad():
        _, edge_prob, _, _ = _model_forward(
            model,
            x,
            edge_index,
            edge_jaccard,
            return_info=True,
        )

    print(f"=== Edge keep probability ({tag}, post-hoc only) ===")
    print(f"All edges mean: {edge_prob.mean().item():.4f}")
    if victim_bot_edge_index is None or bot_bot_edge_index is None:
        return

    num_nodes = int(x.size(0))
    victim_bot_mask = _edge_membership_mask(edge_index, victim_bot_edge_index, num_nodes)
    bot_bot_mask = _edge_membership_mask(edge_index, bot_bot_edge_index, num_nodes)
    clean_mask = ~(victim_bot_mask | bot_bot_mask)

    if clean_mask.any():
        print(f"Original edges mean: {edge_prob[clean_mask].mean().item():.4f}")
    if victim_bot_mask.any():
        print(f"Victim-bot edges mean: {edge_prob[victim_bot_mask].mean().item():.4f}")
    if bot_bot_mask.any():
        print(f"Bot-bot edges mean: {edge_prob[bot_bot_mask].mean().item():.4f}")
