from __future__ import annotations

import statistics
from dataclasses import dataclass

from utils.config import ABLATION_BOT_BUDGET_RATIO, ABLATION_TRAIN_SEEDS, TWITCH_LANG
from utils.data_io import (
    build_or_load_poison_bundle,
    load_clean_tensors,
    poison_bundle_to_tensors,
)
from utils.gib_train import VARIANT_LABELS, evaluate_gib, train_gib


@dataclass(frozen=True)
class AblationSpec:
    name: str
    model_variant: str
    use_ssl_clean: bool
    use_ssl_poison: bool
    use_pseudo_poison: bool


ABLATION_SPECS: tuple[AblationSpec, ...] = (
    AblationSpec(
        "rm_gib",
        "rm_gib",
        use_ssl_clean=True,
        use_ssl_poison=True,
        use_pseudo_poison=True,
    ),
    AblationSpec(
        "srm_gib",
        "srm_gib",
        use_ssl_clean=True,
        use_ssl_poison=True,
        use_pseudo_poison=True,
    ),
    AblationSpec(
        "srm_no_jaccard",
        "srm_gib_no_jaccard",
        use_ssl_clean=True,
        use_ssl_poison=True,
        use_pseudo_poison=True,
    ),
    AblationSpec(
        "srm_gcn_backbone",
        "srm_gib_gcn",
        use_ssl_clean=True,
        use_ssl_poison=True,
        use_pseudo_poison=True,
    ),
)


def run_one(
    spec: AblationSpec,
    clean_tensors: dict,
    poison_tensors: dict,
    device,
    train_seed: int,
) -> dict[str, float]:
    clean_model = train_gib(
        clean_tensors,
        device,
        f"{spec.name}/clean/s{train_seed}",
        use_pseudo=False,
        use_ssl=spec.use_ssl_clean,
        verbose=False,
        model_variant=spec.model_variant,
        train_seed=train_seed,
    )
    poison_model = train_gib(
        poison_tensors,
        device,
        f"{spec.name}/poison/s{train_seed}",
        teacher_model=clean_model if spec.use_pseudo_poison else None,
        use_pseudo=spec.use_pseudo_poison,
        use_ssl=spec.use_ssl_poison,
        verbose=False,
        model_variant=spec.model_variant,
        train_seed=train_seed,
    )
    clean_test = evaluate_gib(clean_model, clean_tensors, device)["test"]
    poison_test = evaluate_gib(poison_model, poison_tensors, device)["test"]
    gap = clean_test - poison_test
    return {"clean_test": clean_test, "poison_test": poison_test, "gap": gap}


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, 0.0
    return mean, statistics.stdev(values)


def main() -> int:
    import torch

    lang = TWITCH_LANG.strip().upper()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean_tensors = load_clean_tensors(lang)
    poison_bundle = build_or_load_poison_bundle(
        clean_tensors,
        lang,
        bot_budget_ratio=ABLATION_BOT_BUDGET_RATIO,
    )
    poison_tensors = poison_bundle_to_tensors(poison_bundle, clean_tensors["train_mask"])

    print("=== Model-variant ablation (original-node test only) ===")
    print(f"Language: {lang} | bot_budget_ratio={ABLATION_BOT_BUDGET_RATIO}")
    print(f"followers_per_victim={poison_bundle.get('followers_per_victim')}")
    print(f"Train seeds: {ABLATION_TRAIN_SEEDS}")
    print("Training recipe: clean=SSL only; poison=SSL + pseudo (fair across variants)")
    for spec in ABLATION_SPECS:
        print(f"  - {spec.name}: {VARIANT_LABELS[spec.model_variant]}")

    per_seed_rows: list[tuple[int, str, float, float, float]] = []
    agg: dict[str, dict[str, list[float]]] = {
        spec.name: {"clean_test": [], "poison_test": [], "gap": []}
        for spec in ABLATION_SPECS
    }

    for train_seed in ABLATION_TRAIN_SEEDS:
        print(f"\n--- Seed {train_seed} ---")
        for spec in ABLATION_SPECS:
            metrics = run_one(spec, clean_tensors, poison_tensors, device, train_seed)
            for key in ("clean_test", "poison_test", "gap"):
                agg[spec.name][key].append(metrics[key])
            per_seed_rows.append(
                (
                    train_seed,
                    spec.name,
                    metrics["clean_test"],
                    metrics["poison_test"],
                    metrics["gap"],
                )
            )
            print(
                f"[seed={train_seed}][{spec.name}] "
                f"clean={metrics['clean_test']:.4f} "
                f"poison={metrics['poison_test']:.4f} "
                f"gap={metrics['gap']:.4f}"
            )

    print("\n=== Per-seed detail ===")
    print(f"{'Seed':>6} {'Ablation':<16} {'Clean':>8} {'Poison':>8} {'Gap':>8}")
    for seed, name, c, p, g in per_seed_rows:
        print(f"{seed:6d} {name:<16} {c:8.4f} {p:8.4f} {g:8.4f}")

    print("\n=== Summary mean ± std over seeds ===")
    print(f"{'Ablation':<16} {'Clean':>16} {'Poison':>16} {'Gap':>16}")
    for spec in ABLATION_SPECS:
        c_mean, c_std = mean_std(agg[spec.name]["clean_test"])
        p_mean, p_std = mean_std(agg[spec.name]["poison_test"])
        g_mean, g_std = mean_std(agg[spec.name]["gap"])
        print(
            f"{spec.name:<16} "
            f"{c_mean:6.4f}±{c_std:5.4f}   "
            f"{p_mean:6.4f}±{p_std:5.4f}   "
            f"{g_mean:6.4f}±{g_std:5.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
