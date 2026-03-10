import math
from typing import Any, Mapping

import numpy as np


DEFAULT_CLASSIFIER_POSITIVE_FRACTION = 0.01
DEFAULT_REGRESSOR_TOTAL_FRACTION = 0.002
DEFAULT_MIN_LEAF_FLOOR = 20
DEFAULT_MIN_LEAF_CAP_FRACTION = 0.20


def compute_min_samples_leaf(
    y_train,
    task_type: str,
    positive_label: Any = 1,
    classifier_positive_fraction: float = DEFAULT_CLASSIFIER_POSITIVE_FRACTION,
    regressor_total_fraction: float = DEFAULT_REGRESSOR_TOTAL_FRACTION,
    min_leaf_floor: int = DEFAULT_MIN_LEAF_FLOOR,
    min_leaf_cap_fraction: float = DEFAULT_MIN_LEAF_CAP_FRACTION,
) -> int:
    y_arr = np.asarray(y_train)
    total_count = int(y_arr.shape[0])
    if total_count == 0:
        raise ValueError("y_train must contain at least one sample")

    task = (task_type or "").strip().lower()
    if task == "classification":
        positive_count = int(np.sum(y_arr == positive_label))
        if positive_count == 0:
            leaf = int(min_leaf_floor)
        else:
            leaf = int(math.ceil(positive_count * float(classifier_positive_fraction)))
    elif task == "regression":
        leaf = int(math.ceil(total_count * float(regressor_total_fraction)))
    else:
        raise ValueError(f"Unsupported task_type='{task_type}', expected 'classification' or 'regression'")

    cap = max(1, int(math.floor(total_count * float(min_leaf_cap_fraction))))
    leaf = max(int(min_leaf_floor), leaf)
    leaf = min(leaf, cap)
    return int(leaf)


def tree_regularization_params(
    y_train,
    task_type: str,
    positive_label: Any = 1,
    cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    c = cfg or {}
    min_samples_leaf = compute_min_samples_leaf(
        y_train=y_train,
        task_type=task_type,
        positive_label=positive_label,
        classifier_positive_fraction=float(c.get("classifier_positive_fraction", DEFAULT_CLASSIFIER_POSITIVE_FRACTION)),
        regressor_total_fraction=float(c.get("regressor_total_fraction", DEFAULT_REGRESSOR_TOTAL_FRACTION)),
        min_leaf_floor=int(c.get("min_leaf_floor", DEFAULT_MIN_LEAF_FLOOR)),
        min_leaf_cap_fraction=float(c.get("min_leaf_cap_fraction", DEFAULT_MIN_LEAF_CAP_FRACTION)),
    )
    return {
        "min_samples_leaf": int(min_samples_leaf),
        "min_samples_split": int(max(2, 2 * min_samples_leaf)),
        "bootstrap": False,
        "ccp_alpha": 1e-4,
        "max_leaf_nodes": 512,
    }
