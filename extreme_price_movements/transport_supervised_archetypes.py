"""Contracts for transport-supervised market-archetype discovery.

The discovery pool is every configured meta feature physically present in the
canonical panel.  Individual folds screen that pool using training rows only;
the screen is a compute proxy, not a narrower feature contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.special import expit


FORBIDDEN_TOKENS = ("_event", "_exit_", "_pnl", "_gross_bps", "_net_bps", "__label", "target__", "path_arch_")


def configured_available_meta_features(config: dict, panel_columns: Iterable[str]) -> list[str]:
    """All configured meta fields that exist in the panel, excluding labels."""
    available = set(panel_columns); result: list[str] = []
    for key, values in config.items():
        if "META" not in str(key) or not isinstance(values, (list, tuple)):
            continue
        for value in values:
            if not isinstance(value, str) or value not in available:
                continue
            if any(token in value for token in FORBIDDEN_TOKENS):
                continue
            if value not in result:
                result.append(value)
    return result


def training_univariate_screen(
    frame: pd.DataFrame, features: Sequence[str], target: Sequence[float], *, maximum: int = 64
) -> list[str]:
    """Rank the *entire* candidate universe using only a training fold.

    This avoids a dense all-feature rule matrix while every eligible meta
    feature remains eligible in each independent fold/head/side screen.
    """
    y = np.asarray(target, dtype=float); scores: list[tuple[float, str]] = []
    for name in features:
        x = pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
        good = np.isfinite(x) & np.isfinite(y)
        if good.sum() < 100 or np.nanstd(x[good]) < 1e-12:
            continue
        corr = pd.Series(x[good]).corr(pd.Series(y[good]), method="spearman")
        if np.isfinite(corr):
            scores.append((abs(float(corr)), name))
    return [name for _score, name in sorted(scores, key=lambda row: (-row[0], row[1]))[:maximum]]


@dataclass(frozen=True)
class SoftRule:
    features: tuple[str, ...]
    directions: tuple[int, ...]  # +1 x>threshold, -1 x<threshold
    thresholds: tuple[float, ...]
    temperatures: tuple[float, ...]


def soft_membership(frame: pd.DataFrame, rule: SoftRule) -> np.ndarray:
    """Overlapping geometric-mean sigmoid rule membership, never a simplex."""
    values: list[np.ndarray] = []
    for name, direction, threshold, temperature in zip(rule.features, rule.directions, rule.thresholds, rule.temperatures):
        x = pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
        signed = direction * (x - threshold) / max(float(temperature), 1e-6)
        values.append(np.clip(expit(signed), 1e-8, 1.0))
    return np.exp(np.mean(np.log(np.column_stack(values)), axis=1)).astype(np.float32)


def effective_support(membership: Sequence[float]) -> float:
    value = np.asarray(membership, dtype=float)
    return float(value.sum() ** 2 / max(np.square(value).sum(), 1e-12))


__all__ = ["SoftRule", "configured_available_meta_features", "effective_support", "soft_membership", "training_univariate_screen"]
