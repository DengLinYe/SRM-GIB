from __future__ import annotations

from utils.config import ATTACK_CURVE_HIDDEN, ATTACK_CURVE_USE_CLASS_WEIGHTS, TWITCH_LANG
from utils.data_io import (
    build_or_load_poison_bundle,
    load_clean_tensors,
    poison_bundle_to_tensors,
)
from utils.gcn_train import evaluate_gcn, train_gcn

ATTACK_BUDGET_RATIOS: tuple[float, ...] = (0.05, 0.10, 0.20, 0.50, 1.00)


def main() -> int:
    import torch

    lang = TWITCH_LANG.strip().upper()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean_tensors = load_clean_tensors(lang)

    gcn_kw = dict(
        verbose=False,
        hidden_channels=ATTACK_CURVE_HIDDEN,
        use_class_weights=ATTACK_CURVE_USE_CLASS_WEIGHTS,
    )

    clean_model = train_gcn(clean_tensors, device, "clean", **gcn_kw)
    clean_test = evaluate_gcn(clean_model, clean_tensors, device)["test"]

    print("=== GCN bot-budget curve (original-node test) ===")
    print(f"GCN profile: hidden={ATTACK_CURVE_HIDDEN}, class_weights={ATTACK_CURVE_USE_CLASS_WEIGHTS}")
    print(f"Language: {lang} | clean_test={clean_test:.4f}")
    print(
        f"{'BotBudget':>9} {'k/victim':>10} {'bots':>8} "
        f"{'inj_edges':>12} {'poison_test':>12} {'drop':>8}"
    )

    for ratio in ATTACK_BUDGET_RATIOS:
        bundle = build_or_load_poison_bundle(clean_tensors, lang, bot_budget_ratio=ratio)
        poison_tensors = poison_bundle_to_tensors(bundle, clean_tensors["train_mask"])
        poison_model = train_gcn(poison_tensors, device, f"poison_{ratio:.2f}", **gcn_kw)
        poison_test = evaluate_gcn(poison_model, poison_tensors, device)["test"]
        drop = clean_test - poison_test
        num_bots = int(bundle["x"].size(0)) - int(bundle["num_original_nodes"])
        print(
            f"{ratio:8.2f} "
            f"{int(bundle.get('followers_per_victim', 0)):10d} "
            f"{num_bots:8d} "
            f"{int(bundle['injected_edge_index'].size(1)):12d} "
            f"{poison_test:12.4f} "
            f"{drop:8.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
