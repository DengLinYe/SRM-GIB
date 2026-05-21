from __future__ import annotations

from utils.config import ABLATION_BUDGET_RATIO, TWITCH_LANG
from utils.data_io import (
    build_or_load_poison_bundle,
    load_clean_tensors,
    poison_bundle_to_tensors,
)
from utils.gib_train import evaluate_gib, train_gib

ABLATION_SPECS: tuple[tuple[str, bool, bool], ...] = (
    ("gib_ce_kl", False, False),
    ("gib_ssl", True, False),
    ("gib_pseudo", False, True),
    ("gib_full", True, True),
)


def run_one(
    name: str,
    use_ssl: bool,
    use_pseudo: bool,
    clean_tensors: dict,
    poison_tensors: dict,
    device,
) -> dict[str, float]:
    clean_model = train_gib(
        clean_tensors,
        device,
        f"{name}/clean",
        use_pseudo=False,
        use_ssl=use_ssl,
        verbose=False,
    )
    poison_model = train_gib(
        poison_tensors,
        device,
        f"{name}/poison",
        teacher_model=clean_model if use_pseudo else None,
        use_pseudo=use_pseudo,
        use_ssl=use_ssl,
        verbose=False,
    )
    clean_test = evaluate_gib(clean_model, clean_tensors, device)["test"]
    poison_test = evaluate_gib(poison_model, poison_tensors, device)["test"]

    gap = clean_test - poison_test
    print(f"[{name}] clean_test={clean_test:.4f} poison_test={poison_test:.4f} gap={gap:.4f}")
    return {"clean_test": clean_test, "poison_test": poison_test, "gap": gap}


def main() -> int:
    import torch

    lang = TWITCH_LANG.strip().upper()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean_tensors = load_clean_tensors(lang)
    poison_bundle = build_or_load_poison_bundle(
        clean_tensors,
        lang,
        injected_edge_budget_ratio=ABLATION_BUDGET_RATIO,
    )
    poison_tensors = poison_bundle_to_tensors(poison_bundle, clean_tensors["train_mask"])

    print("=== Ablation study (fair pseudo on unlabeled_mask only) ===")
    print(f"Language: {lang} | budget={ABLATION_BUDGET_RATIO}")
    print(f"followers_per_victim={poison_bundle.get('followers_per_victim')}")

    rows: list[tuple[str, float, float, float]] = []
    for name, use_ssl, use_pseudo in ABLATION_SPECS:
        metrics = run_one(
            name,
            use_ssl,
            use_pseudo,
            clean_tensors,
            poison_tensors,
            device,
        )
        rows.append((name, metrics["clean_test"], metrics["poison_test"], metrics["gap"]))

    print("\n=== Summary (original-node test only) ===")
    print(f"{'Ablation':<12} {'Clean':>8} {'Poison':>8} {'Gap':>8}")
    for name, c, p, g in rows:
        print(f"{name:<12} {c:8.4f} {p:8.4f} {g:8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
