import numpy as np
import pandas as pd
from typing import Optional, Callable, Dict, Literal, Tuple
from dataclasses import dataclass, field

@dataclass
class UtilityConfig:
    utility_mode: Literal["topq_mean", "policy_mean", "custom_callable"]
    topq: float = 0.2
    direction: Literal["higher_is_better", "lower_is_better"] = "higher_is_better"
    custom_callable_path: Optional[str] = None
    cost_per_trade: float = 0.0
    min_edge: float = 0.0
    position_clip: float = 1.0
    use_weights: bool = False
    sample_weight_col: Optional[str] = None
    threshold: float = 0.0 # Used for policy_mean


@dataclass
class FeatureSelectConfig:
    n_repeats_perm: int = 10
    min_features: int = 10
    max_features: Optional[int] = None
    utility_drop_tol: float = 0.0
    confirm_drop: bool = True
    confirm_mode: Literal["full_cv", "single_seed_fast"] = "single_seed_fast"
    topk_presence: int = 20
    min_presence: float = 0.4
    weights: dict = field(default_factory=lambda: {"wU": 0.5, "wS": 0.3, "wSt": 0.2})
    eps: float = 1e-12
    shap_sample: int = 5000
    perm_sample: int = 20000


def compute_utility(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    utility_config: UtilityConfig,
    X_val: Optional[pd.DataFrame] = None,
) -> float:
    """Computes generic OOS utility."""
    if utility_config.utility_mode == "topq_mean":
        target = y_true
        if X_val is not None and "realized_utility" in X_val.columns:
            target = X_val["realized_utility"].values

        k = max(1, int(len(y_pred) * utility_config.topq))

        if utility_config.direction == "higher_is_better":
            idx = np.argsort(y_pred)[-k:]
        else:
            idx = np.argsort(y_pred)[:k]

        return float(np.mean(target[idx]))

    elif utility_config.utility_mode == "policy_mean":
        target = y_true
        if X_val is not None and "realized_utility" in X_val.columns:
            target = X_val["realized_utility"].values

        mask = y_pred > utility_config.threshold
        if not np.any(mask):
            return 0.0

        returns = target[mask] - utility_config.cost_per_trade
        return float(np.mean(returns))

    elif utility_config.utility_mode == "custom_callable":
        if not utility_config.custom_callable_path:
            raise ValueError("custom_callable_path is required when utility_mode='custom_callable'")
        import importlib
        module_path, func_name = utility_config.custom_callable_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        func = getattr(module, func_name)
        return float(func(y_true, y_pred, X_val, utility_config))

    else:
        raise ValueError(f"Unknown utility_mode: {utility_config.utility_mode}")

def compute_bootstrap_ci(utilities: np.ndarray, n_boot: int = 1000, seed: int = 42) -> Tuple[float, float, float]:
    """Computes mean and 95% bootstrap confidence interval."""
    rng = np.random.RandomState(seed)
    n = len(utilities)
    if n == 0:
        return 0.0, 0.0, 0.0
    if n == 1:
        return float(utilities[0]), float(utilities[0]), float(utilities[0])

    samples = rng.choice(utilities, size=(n_boot, n), replace=True)
    means = np.mean(samples, axis=1)
    return float(np.mean(utilities)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def compute_composite_score(
    perm_mean: np.ndarray,
    perm_std: np.ndarray,
    shap_mean: np.ndarray,
    shap_std: np.ndarray,
    shap_presence: np.ndarray,
    config: FeatureSelectConfig
) -> np.ndarray:
    """Computes the composite score for feature selection."""
    eps = config.eps
    wU = config.weights.get("wU", 0.5)
    wS = config.weights.get("wS", 0.3)
    wSt = config.weights.get("wSt", 0.2)

    def rank_scale(arr: np.ndarray) -> np.ndarray:
        if len(arr) <= 1:
            return np.ones_like(arr)
        ranks = np.argsort(np.argsort(arr))
        return ranks / (len(arr) - 1)

    U = rank_scale(perm_mean)
    S = rank_scale(shap_mean)

    perm_stability = 1.0 / (1.0 + perm_std / (np.abs(perm_mean) + eps))
    shap_stability = 1.0 / (1.0 + shap_std / (np.abs(shap_mean) + eps))

    St = 0.5 * shap_stability + 0.5 * perm_stability

    composite = wU * U + wS * S + wSt * St

    presence_mask = shap_presence < config.min_presence
    composite[presence_mask] *= 0.5

    composite[perm_mean <= 0.0] *= 0.1

    return composite
