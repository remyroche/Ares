#!/usr/bin/env python3
"""Validate held-out side/archetype interactions in residual meta context."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_train_meta_residual_archetype_enhancement import (
    DEFAULT_OUT_DIR,  # noqa: E402
)

NUMERIC_FEATURES = (
    "meta_resid_arch_expected_hit_surprise",
    "meta_resid_arch_expected_ev",
    "meta_resid_arch_expected_bad_mae",
    "meta_resid_arch_expected_timeout",
    "meta_resid_arch_expected_dirty_positive",
    "meta_resid_arch_entropy",
    "meta_resid_arch_confidence",
    "meta_resid_ae_gmm_posterior_max",
    "meta_resid_ae_gmm_entropy",
    "meta_resid_ae_min_mahalanobis",
    "meta_resid_ae_dae_reconstruction_error_zscore",
)
FOLDS = (
    (("2026-04",), "2026-05"),
    (("2026-04", "2026-05"), "2026-06"),
)


def _group_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["side_name"].astype(str)
        + "||"
        + frame["archetype_policy_key"].astype(str)
    )


def _one_hot(values: pd.Series, categories: list[str]) -> np.ndarray:
    mapping = {name: idx for idx, name in enumerate(categories)}
    positions = values.map(mapping).fillna(-1).to_numpy(dtype=np.int32)
    output = np.zeros((len(values), len(categories)), dtype=np.float32)
    valid = positions >= 0
    output[np.flatnonzero(valid), positions[valid]] = 1.0
    return output


def _numeric(
    train: pd.DataFrame,
    valid: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x = (
        train.reindex(columns=NUMERIC_FEATURES)
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float32)
    )
    valid_x = (
        valid.reindex(columns=NUMERIC_FEATURES)
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float32)
    )
    medians = np.nanmedian(train_x, axis=0).astype(np.float32)
    medians = np.nan_to_num(medians, nan=0.0)
    train_x = np.where(np.isfinite(train_x), train_x, medians)
    valid_x = np.where(np.isfinite(valid_x), valid_x, medians)
    center = np.mean(train_x, axis=0, dtype=np.float64).astype(np.float32)
    scale = np.std(train_x, axis=0, dtype=np.float64).astype(np.float32)
    scale = np.where(scale > 1e-6, scale, 1.0).astype(np.float32)
    return (train_x - center) / scale, (valid_x - center) / scale, center, scale


def _design(
    numeric: np.ndarray,
    group: np.ndarray,
    *,
    interaction: bool,
) -> np.ndarray:
    blocks = [numeric, group]
    if interaction:
        blocks.append(
            (group[:, :, None] * numeric[:, None, :]).reshape(len(numeric), -1)
        )
    return np.concatenate(blocks, axis=1).astype(np.float32, copy=False)


def _metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - y
    rho = spearmanr(y, prediction, nan_policy="omit").statistic
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "mse": float(np.mean(error * error)),
        "mae": float(np.mean(np.abs(error))),
        "r2": float(1.0 - np.sum(error * error) / denominator)
        if denominator > 0.0
        else np.nan,
        "spearman_ic": float(rho) if np.isfinite(rho) else np.nan,
    }


def _fit_fold(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    categories: list[str],
    *,
    train_group_override: pd.Series | None = None,
) -> tuple[dict[str, float], dict[str, float], Ridge, Ridge]:
    train_numeric, valid_numeric, _center, _scale = _numeric(train, valid)
    train_group_values = (
        train_group_override if train_group_override is not None else _group_key(train)
    )
    train_group = _one_hot(train_group_values, categories)
    valid_group = _one_hot(_group_key(valid), categories)
    y_train = pd.to_numeric(train["hit_surprise"], errors="coerce").to_numpy(
        dtype=np.float32
    )
    y_valid = pd.to_numeric(valid["hit_surprise"], errors="coerce").to_numpy(
        dtype=np.float32
    )
    additive = Ridge(alpha=10.0, fit_intercept=True).fit(
        _design(train_numeric, train_group, interaction=False),
        y_train,
    )
    interaction = Ridge(alpha=10.0, fit_intercept=True).fit(
        _design(train_numeric, train_group, interaction=True),
        y_train,
    )
    additive_prediction = additive.predict(
        _design(valid_numeric, valid_group, interaction=False)
    )
    interaction_prediction = interaction.predict(
        _design(valid_numeric, valid_group, interaction=True)
    )
    return (
        _metrics(y_valid, additive_prediction),
        _metrics(y_valid, interaction_prediction),
        additive,
        interaction,
    )


def _permuted_groups(train: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    output = _group_key(train).copy()
    rank_band = (
        np.floor(
            pd.to_numeric(
                train["historical_rank_current_reference"], errors="coerce"
            ).fillna(0.0)
            * 10.0
        )
        .clip(0, 9)
        .astype(np.int8)
    )
    day = pd.to_datetime(train["__ts__"], utc=True, errors="coerce").dt.floor("D")
    keys = pd.DataFrame(
        {"day": day, "side": train["side_name"].astype(str), "rank": rank_band}
    )
    for positions in keys.groupby(["day", "side", "rank"], sort=False).groups.values():
        idx = np.asarray(list(positions))
        output.loc[idx] = rng.permutation(output.loc[idx].to_numpy())
    return output


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    root = DEFAULT_OUT_DIR
    report_dir = root / "final_report"
    frame = pd.read_parquet(
        root / "historical_rank_oos" / "oos_predictions_historical_rank.parquet"
    )
    frame = frame[
        pd.to_numeric(frame["historical_rank_current_reference"], errors="coerce").ge(
            0.80
        )
    ].copy()
    frame["hit_surprise"] = (
        pd.to_numeric(frame["clean_exec"], errors="coerce")
        - pd.to_numeric(frame["hit_prob_current_reference"], errors="coerce")
    ).astype(np.float32)
    categories = sorted(_group_key(frame).unique().tolist())
    fold_rows: list[dict[str, Any]] = []
    permutation_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(20260711)
    for train_months, valid_month in FOLDS:
        train = frame[
            frame["calendar_month"].astype(str).isin(train_months)
        ].reset_index(drop=True)
        valid = frame[frame["calendar_month"].astype(str).eq(valid_month)].reset_index(
            drop=True
        )
        additive, interaction, _add_model, _int_model = _fit_fold(
            train, valid, categories
        )
        fold_rows.append(
            {
                "train_months": ",".join(train_months),
                "valid_month": valid_month,
                "train_rows": int(len(train)),
                "valid_rows": int(len(valid)),
                **{f"additive_{name}": value for name, value in additive.items()},
                **{f"interaction_{name}": value for name, value in interaction.items()},
                "mse_improvement": additive["mse"] - interaction["mse"],
                "mae_improvement": additive["mae"] - interaction["mae"],
                "spearman_improvement": interaction["spearman_ic"]
                - additive["spearman_ic"],
            }
        )
        for draw in range(100):
            permuted = _permuted_groups(train, rng)
            perm_add, perm_interaction, _a, _i = _fit_fold(
                train,
                valid,
                categories,
                train_group_override=permuted,
            )
            permutation_rows.append(
                {
                    "valid_month": valid_month,
                    "draw": draw,
                    "mse_improvement": perm_add["mse"] - perm_interaction["mse"],
                }
            )
    folds = pd.DataFrame(fold_rows)
    permutations = pd.DataFrame(permutation_rows)
    folds.to_csv(report_dir / "stage3_interaction_oos_folds.csv", index=False)
    permutations.to_csv(
        report_dir / "stage3_interaction_policy_permutations.csv", index=False
    )
    actual_mean = float(folds["mse_improvement"].mean())
    placebo_by_draw = permutations.groupby("draw")["mse_improvement"].mean()
    p_value = float(
        (1 + np.sum(placebo_by_draw >= actual_mean)) / (1 + len(placebo_by_draw))
    )
    manifest = {
        "schema": "meta_residual_side_archetype_interaction_validation_v1",
        "population": "causal_historical_top20",
        "folds": int(len(folds)),
        "folds_mse_improved": int(folds["mse_improvement"].gt(0.0).sum()),
        "mean_mse_improvement": actual_mean,
        "mean_mae_improvement": float(folds["mae_improvement"].mean()),
        "mean_spearman_improvement": float(folds["spearman_improvement"].mean()),
        "permutation_p_value": p_value,
        "interaction_pass": bool(
            folds["mse_improvement"].gt(0.0).all()
            and actual_mean > 0.0
            and p_value <= 0.05
        ),
        "shrinkage": "Ridge alpha=10 on side/archetype-specific latent-state slopes",
        "leakage_contract": (
            "Only prior OOS-generated context trains each fold; May and June are evaluated strictly after their train months."
        ),
    }
    (report_dir / "stage3_interaction_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2),
        encoding="utf-8",
    )
    print(json.dumps(_json_safe(manifest), indent=2), flush=True)


if __name__ == "__main__":
    main()
