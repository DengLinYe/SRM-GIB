from __future__ import annotations

EPOCHS: int = 300
PATIENCE: int = 40
TRAIN_SEED: int = 42
TWITCH_LANG: str = "EN"
HIDDEN_CHANNELS: int = 128
DROPOUT: float = 0.5
LEARNING_RATE: float = 0.01
WEIGHT_DECAY: float = 5e-4

GIB_BETA_N: float = 0.05
GIB_BETA_X: float = 0.0005
GIB_PRIOR_KEEP_RATE: float = 0.5
GIB_TEMPERATURE: float = 1.0

ABLATION_BUDGET_RATIO: float = 0.5

ATTACK_CURVE_HIDDEN: int = 64
ATTACK_CURVE_USE_CLASS_WEIGHTS: bool = False
