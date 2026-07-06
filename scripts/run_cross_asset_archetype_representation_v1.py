#!/usr/bin/env python3
"""Cross-asset representation acceptance smoke for archetype-aware meta inputs.

This runner evaluates shared live-predictable representation outputs as
candidate train_meta features.  It deliberately does not promote hard gates or
standalone archetype specialists.  The first implementation covers the Model A
cross-market challenger from ``docs/cross_asset_archetype_meta_plan.md``:

* fit month-forward models on prior months only;
* export OOF/prior-fold risk and execution scores;
* evaluate those outputs by archetype x side cells;
* compare against matched random controls;
* keep outcome/path labels out of decision-time features.

The produced artifacts are acceptance diagnostics, not frozen replay evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    from lightgbm import LGBMClassifier, LGBMRegressor

    _LIGHTGBM_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    LGBMClassifier = None
    LGBMRegressor = None
    _LIGHTGBM_AVAILABLE = False


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_HANDOFF_DIR = Path(
    "data_perp/reports/s52_trailing_profit_best_pointwise_scored_ledger_20260705_v1"
    "/s52_trailing_regime_meta_handoff_longsplit_v2"
)
DEFAULT_OUT_DIR = DEFAULT_HANDOFF_DIR / "cross_asset_archetype_representation_v1"

KEY_COLUMNS = ("__ts__", "__symbol__", "side_name")
TOP_FRACS = (0.30, 0.20, 0.10)
TARGETS = {
    "exec_margin": {"kind": "regression", "score": "cross_lgbm_exec_margin_score", "higher_is_better": True},
    "full_path_bad_mae_1r": {"kind": "classification", "score": "cross_lgbm_bad_mae_score", "higher_is_better": False},
    "timeout": {"kind": "classification", "score": "cross_lgbm_timeout_score", "higher_is_better": False},
    "dirty_positive": {"kind": "classification", "score": "cross_lgbm_dirty_positive_score", "higher_is_better": False},
}
OUTCOME_COLUMNS = {
    "target_soft",
    "target_hard",
    "first_pass_good",
    "first_pass_bad",
    "u_policy_net",
    "ret_net",
    "is_timeout",
    "mae_norm",
    "mfe_norm",
    "first_touch_net",
    "first_touch_gross",
    "first_touch_mae_norm",
    "first_touch_mfe_norm",
    "first_touch_full_path_mae_norm",
    "first_touch_bad_mae_1r",
    "mfe_1r_before_mae_1r",
    "mae_1r_before_mfe_1r",
    "mfe_before_mae_1r",
    "mae_before_mfe_1r",
    "max_adverse_before_mfe_1r",
    "underwater_bars_before_mfe_1r",
    "underwater_fraction_before_mfe_1r",
    "exec_margin",
    "ev_after_1pct",
    "clean_exec",
    "dirty_positive",
    "timeout",
    "full_path_bad_mae_1r",
    "positive_exec_margin",
    "clean_exec_label",
    "bad_path_label",
    "long_path_full_bad_mae_1r",
    "long_path_time_to_profit_bars",
    "long_path_slow_profit",
    "long_path_post_mfe_drawdown_norm",
    "long_path_post_mfe_bad_drawdown",
    "long_trailing_activated",
    "long_trailing_success",
    "long_path_clean_exec_label",
    "long_path_dirty_positive_label",
    "long_path_quality_soft",
    "long_bad_path_label",
}
NEVER_FEATURE_COLUMNS = {
    "__ts__",
    "__symbol__",
    "month",
    "variant",
    "side",
    "selected_top10",
    "selected_top20",
    "selected_top30",
}
CROSS_FEATURE_PREFIXES = (
    "q_tail_",
    "q_iqr_",
    "q_width_",
    "width_",
    "tail_",
    "asym_",
    "iqr_",
    "pct_assets_",
    "cs_",
    "cs_rank_",
    "btc_",
    "eth_",
    "eth_btc_",
    "xs_dispersion_",
    "trend_dispersion_",
    "spectral_",
    "state_spectral_",
    "xasset_",
    "mkt_",
    "eig_",
    "market_breadth_",
    "market_dispersion_",
    "xasset_mkt_",
    "market_index_",
    "cross_asset_",
    "median_asset_",
    "top_decile_asset_",
    "cross_asset_correlation_",
    "avg_pairwise_corr_",
)
LIVE_CONTEXT_PREFIXES = (
    "gmm_",
    "aegmm_",
    "side_aegmm_",
    "cluster_",
    "latent_",
    "dae_",
    "AE_",
    "mahalanobis",
    "expected_mahalanobis",
    "min_mahalanobis",
    "regime_",
    "source_",
    "long_source_",
    "calendar_",
    "structural_",
    "meta_action_",
    "meta_context_",
    "meta_threshold_",
    "base_score_",
    "score",
)
CONTROL_NAMES = ("perm", "block_perm", "noise_ar1")
AE_FAMILY_PREFIXES: dict[str, tuple[str, ...]] = {
    "tail": ("q_tail_", "tail_"),
    "breadth": ("pct_assets_", "market_breadth_", "mkt_breadth_"),
    "dispersion": (
        "xs_dispersion_",
        "trend_dispersion_",
        "market_dispersion_",
        "iqr_",
        "q_iqr_",
        "width_",
        "q_width_",
    ),
    "btc_eth": ("btc_", "eth_", "eth_btc_"),
    "corr_spectral": (
        "state_spectral_",
        "spectral_",
        "eig_",
        "cross_asset_correlation_",
        "avg_pairwise_corr_",
    ),
    "xasset": ("xasset_", "xasset_mkt_", "cross_asset_", "market_index_", "median_asset_", "top_decile_asset_"),
    "gmm_ae": (
        "gmm_",
        "cluster_",
        "latent_",
        "dae_",
        "AE_",
        "mahalanobis",
        "expected_mahalanobis",
        "min_mahalanobis",
    ),
    "regime_context": ("regime_", "meta_action_", "meta_context_", "meta_threshold_"),
}
AE_OUTPUT_COLUMNS = (
    "market_z_0",
    "market_z_1",
    "market_z_2",
    "market_z_3",
    "market_ae_recon_error",
    "market_ae_recon_error_pct",
    "market_ae_mahalanobis_diag",
    "family_recon_error_tail",
    "family_recon_error_breadth",
    "family_recon_error_dispersion",
    "family_recon_error_btc_eth",
    "family_recon_error_corr_spectral",
    "family_recon_error_xasset",
    "family_recon_error_gmm_ae",
    "family_recon_error_regime_context",
    "family_recon_error_other",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _num(values: Any, *, index: pd.Index | None = None, default: float = np.nan) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    if values is None:
        if index is None:
            return pd.Series(dtype=np.float32)
        return pd.Series(default, index=index, dtype=np.float32)
    return pd.to_numeric(pd.Series(values, index=index), errors="coerce")


def _rate(values: Any) -> float:
    arr = _num(values).replace([np.inf, -np.inf], np.nan).dropna()
    return float(arr.clip(0.0, 1.0).mean()) if len(arr) else float("nan")


def _mean(values: Any) -> float:
    arr = _num(values).replace([np.inf, -np.inf], np.nan).dropna()
    return float(arr.mean()) if len(arr) else float("nan")


def _safe_auc(y_true: pd.Series, score: pd.Series) -> float:
    y = _num(y_true)
    s = _num(score)
    valid = y.notna() & s.notna()
    if int(valid.sum()) < 20 or int(y.loc[valid].nunique(dropna=True)) < 2:
        return float("nan")
    return float(roc_auc_score(y.loc[valid].astype(int), s.loc[valid]))


def _safe_ap(y_true: pd.Series, score: pd.Series) -> float:
    y = _num(y_true)
    s = _num(score)
    valid = y.notna() & s.notna()
    if int(valid.sum()) < 20 or int(y.loc[valid].nunique(dropna=True)) < 2:
        return float("nan")
    return float(average_precision_score(y.loc[valid].astype(int), s.loc[valid]))


def _safe_spearman(y_true: pd.Series, score: pd.Series) -> float:
    y = _num(y_true)
    s = _num(score)
    valid = y.notna() & s.notna()
    if int(valid.sum()) < 20 or int(y.loc[valid].nunique(dropna=True)) < 2 or int(s.loc[valid].nunique(dropna=True)) < 2:
        return float("nan")
    return float(y.loc[valid].rank(method="average").corr(s.loc[valid].rank(method="average")))


def _read_parquet_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    try:
        import pyarrow.parquet as pq

        available = set(pq.read_schema(path).names)
        use_cols = [col for col in columns if col in available]
        return pd.read_parquet(path, columns=use_cols)
    except Exception:
        frame = pd.read_parquet(path)
        return frame[[col for col in columns if col in frame.columns]].copy()


def _candidate_column(frontier: str) -> str:
    normalized = str(frontier).lower().replace("top", "")
    return f"selected_top{int(normalized)}"


def _load_joined_frame(handoff_dir: Path, ledger_path: Path | None, frontier: str) -> pd.DataFrame:
    handoff_path = handoff_dir / "train_meta_regime_handoff.parquet"
    if ledger_path is None:
        ledger_path = handoff_dir / "s52_trailing_regime_scored_ledger.parquet"
    handoff = pd.read_parquet(handoff_path)
    selected_col = _candidate_column(frontier)
    ledger_cols = list(KEY_COLUMNS) + ["month", "score", selected_col] + sorted(OUTCOME_COLUMNS)
    ledger = _read_parquet_columns(ledger_path, ledger_cols)
    missing = [col for col in KEY_COLUMNS if col not in handoff.columns or col not in ledger.columns]
    if missing:
        raise ValueError(f"Missing join key columns: {missing}")
    merged = handoff.merge(
        ledger,
        on=list(KEY_COLUMNS),
        how="left",
        suffixes=("", "__ledger"),
        validate="one_to_one",
    )
    for col in ("month", "score", selected_col):
        ledger_col = f"{col}__ledger"
        if ledger_col in merged.columns:
            if col not in merged.columns:
                merged[col] = merged[ledger_col]
            else:
                merged[col] = merged[col].where(merged[col].notna(), merged[ledger_col])
            merged = merged.drop(columns=[ledger_col])
    if "month" not in merged.columns or merged["month"].isna().all():
        merged["month"] = pd.to_datetime(merged["__ts__"], utc=True, errors="coerce").dt.to_period("M").astype(str)
    merged["month"] = merged["month"].astype(str)
    if selected_col not in merged.columns:
        raise ValueError(f"Missing frontier column: {selected_col}")
    merged[selected_col] = _num(merged[selected_col], index=merged.index, default=0.0).fillna(0.0).gt(0.5)
    if "timeout" not in merged.columns and "is_timeout" in merged.columns:
        merged["timeout"] = _num(merged["is_timeout"], index=merged.index)
    if "mfe_before_mae_1r" not in merged.columns and "mfe_1r_before_mae_1r" in merged.columns:
        merged["mfe_before_mae_1r"] = _num(merged["mfe_1r_before_mae_1r"], index=merged.index)
    if "mae_before_mfe_1r" not in merged.columns and "mae_1r_before_mfe_1r" in merged.columns:
        merged["mae_before_mfe_1r"] = _num(merged["mae_1r_before_mfe_1r"], index=merged.index)
    if "first_touch_bad_mae_1r" not in merged.columns and "first_touch_full_path_mae_norm" in merged.columns:
        merged["first_touch_bad_mae_1r"] = _num(merged["first_touch_full_path_mae_norm"], index=merged.index).ge(1.0).astype(float)
    if "full_path_bad_mae_1r" not in merged.columns and "first_touch_bad_mae_1r" in merged.columns:
        merged["full_path_bad_mae_1r"] = _num(merged["first_touch_bad_mae_1r"], index=merged.index)
    if "exec_margin" not in merged.columns:
        candidate = "ev_after_1pct" if "ev_after_1pct" in merged.columns else "ret_net"
        merged["exec_margin"] = _num(merged.get(candidate), index=merged.index)
    merged["positive_exec_margin"] = _num(merged.get("exec_margin"), index=merged.index).gt(0.0).astype(float)
    if "clean_exec" not in merged.columns:
        merged["clean_exec"] = (
            merged["positive_exec_margin"].gt(0.5)
            & _num(merged.get("full_path_bad_mae_1r"), index=merged.index, default=0.0).fillna(0.0).le(0.5)
            & _num(merged.get("timeout"), index=merged.index, default=0.0).fillna(0.0).le(0.5)
        ).astype(float)
    if "dirty_positive" not in merged.columns:
        merged["dirty_positive"] = (
            merged["positive_exec_margin"].gt(0.5)
            & (
                _num(merged.get("full_path_bad_mae_1r"), index=merged.index, default=0.0).fillna(0.0).gt(0.5)
                | _num(merged.get("timeout"), index=merged.index, default=0.0).fillna(0.0).gt(0.5)
            )
        ).astype(float)
    merged["clean_exec_label"] = _num(merged.get("clean_exec"), index=merged.index, default=0.0).fillna(0.0)
    merged["bad_path_label"] = (
        _num(merged.get("full_path_bad_mae_1r"), index=merged.index, default=0.0).fillna(0.0).gt(0.5)
        | _num(merged.get("timeout"), index=merged.index, default=0.0).fillna(0.0).gt(0.5)
    ).astype(float)
    return merged


def _is_candidate_feature(col: str, series: pd.Series) -> bool:
    if col in NEVER_FEATURE_COLUMNS or col in OUTCOME_COLUMNS or col.endswith("__ledger"):
        return False
    if col.startswith("selected_top"):
        return False
    if col.startswith("cross_lgbm_"):
        return False
    if col == "score":
        return True
    if col.startswith(CROSS_FEATURE_PREFIXES) or col.startswith(LIVE_CONTEXT_PREFIXES):
        return True
    if col in {"side_name", "source_family", "source_semantic_family", "source_semantic_family_base", "long_source_regime_split"}:
        return True
    return pd.api.types.is_numeric_dtype(series) and (
        col.startswith("meta_") or col.startswith("base_") or col.startswith("ctx_")
    )


def _feature_columns(frame: pd.DataFrame, *, soft_only: bool) -> tuple[list[str], list[str]]:
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    for col in frame.columns:
        if not _is_candidate_feature(col, frame[col]):
            continue
        if soft_only and col in {"gmm_cluster_id", "aegmm_cluster", "side_aegmm_cluster"}:
            continue
        if pd.api.types.is_numeric_dtype(frame[col]) or frame[col].dtype == bool:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return sorted(set(numeric_cols)), sorted(set(categorical_cols))


def _make_xy(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    numeric_cols: list[str],
    categorical_cols: list[str],
    clip: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    parts_train: list[pd.DataFrame] = []
    parts_valid: list[pd.DataFrame] = []
    if numeric_cols:
        train_num = train.loc[:, numeric_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid_num = valid.loc[:, numeric_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        missing_cols = [f"{col}__missing" for col in numeric_cols]
        train_missing = train_num.isna().astype(np.float32)
        valid_missing = valid_num.isna().astype(np.float32)
        train_missing.columns = missing_cols
        valid_missing.columns = missing_cols
        med = train_num.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        q25 = train_num.quantile(0.25, numeric_only=True)
        q75 = train_num.quantile(0.75, numeric_only=True)
        scale = (q75 - q25).replace([np.inf, -np.inf], np.nan).abs().fillna(0.0)
        std = train_num.std(numeric_only=True).replace([np.inf, -np.inf], np.nan).abs().fillna(0.0)
        scale = scale.where(scale > 1e-9, std).where(lambda s: s > 1e-9, 1.0)
        train_scaled = ((train_num.fillna(med) - med) / scale).clip(-float(clip), float(clip)).astype(np.float32)
        valid_scaled = ((valid_num.fillna(med) - med) / scale).clip(-float(clip), float(clip)).astype(np.float32)
        parts_train.extend([train_scaled, train_missing])
        parts_valid.extend([valid_scaled, valid_missing])
    if categorical_cols:
        train_cat = pd.get_dummies(train.loc[:, categorical_cols].astype(str).fillna("missing"), dummy_na=False)
        valid_cat = pd.get_dummies(valid.loc[:, categorical_cols].astype(str).fillna("missing"), dummy_na=False)
        valid_cat = valid_cat.reindex(columns=train_cat.columns, fill_value=0)
        parts_train.append(train_cat.astype(np.float32))
        parts_valid.append(valid_cat.astype(np.float32))
    if not parts_train:
        raise ValueError("No cross-asset/archetype feature columns available.")
    x_train = pd.concat(parts_train, axis=1)
    x_valid = pd.concat(parts_valid, axis=1).reindex(columns=x_train.columns, fill_value=0.0)
    return x_train, x_valid, list(x_train.columns)


def _classification_weights(y: pd.Series, train: pd.DataFrame) -> np.ndarray:
    target = _num(y).fillna(0.0).astype(int)
    weights = np.ones(len(target), dtype=np.float32)
    pos = int(target.sum())
    neg = int(len(target) - pos)
    if pos > 0 and neg > 0:
        weights[target.to_numpy(dtype=bool)] = neg / max(pos, 1)
    weights *= 1.0 + 0.50 * _num(train.get("dirty_positive"), index=train.index, default=0.0).fillna(0.0).to_numpy(dtype=np.float32)
    weights *= 1.0 + 0.50 * _num(train.get("full_path_bad_mae_1r"), index=train.index, default=0.0).fillna(0.0).to_numpy(dtype=np.float32)
    weights = np.clip(weights, 0.20, 8.0)
    return weights / max(float(weights.mean()), 1e-12)


def _regression_weights(y: pd.Series, train: pd.DataFrame) -> np.ndarray:
    target = _num(y).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    weights = np.ones(len(target), dtype=np.float32)
    if int(target.nunique(dropna=True)) > 1:
        weights *= np.where(target.ge(float(target.quantile(0.80))), 1.75, 1.0)
        weights *= np.where(target.le(float(target.quantile(0.20))), 1.25, 1.0)
    weights *= np.where(target.gt(0.0), 1.50, 1.0)
    weights *= 1.0 + 0.40 * _num(train.get("dirty_positive"), index=train.index, default=0.0).fillna(0.0).to_numpy(dtype=np.float32)
    weights = np.clip(weights, 0.20, 6.0)
    return weights / max(float(weights.mean()), 1e-12)


def _fit_classifier(x: pd.DataFrame, y: pd.Series, train: pd.DataFrame, seed: int) -> Any:
    target = _num(y).fillna(0.0).astype(int)
    if int(target.nunique(dropna=True)) < 2:
        return None
    weights = _classification_weights(target, train)
    if _LIGHTGBM_AVAILABLE and LGBMClassifier is not None:
        model = LGBMClassifier(
            objective="binary",
            n_estimators=180,
            learning_rate=0.035,
            num_leaves=17,
            min_child_samples=35,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_alpha=0.10,
            reg_lambda=8.0,
            random_state=int(seed),
            n_jobs=2,
            verbosity=-1,
        )
        model.fit(x, target, sample_weight=weights)
        return model
    model = ExtraTreesClassifier(
        n_estimators=220,
        max_depth=7,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced",
        random_state=int(seed),
        n_jobs=2,
    )
    model.fit(x, target, sample_weight=weights)
    return model


def _fit_regressor(x: pd.DataFrame, y: pd.Series, train: pd.DataFrame, seed: int) -> Any:
    target = _num(y).replace([np.inf, -np.inf], np.nan)
    valid = target.notna()
    if int(valid.sum()) < 50 or float(target.loc[valid].std()) <= 1e-12:
        return None
    weights = _regression_weights(target.loc[valid], train.loc[valid])
    if _LIGHTGBM_AVAILABLE and LGBMRegressor is not None:
        model = LGBMRegressor(
            objective="regression",
            n_estimators=220,
            learning_rate=0.035,
            num_leaves=17,
            min_child_samples=35,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_alpha=0.10,
            reg_lambda=10.0,
            random_state=int(seed),
            n_jobs=2,
            verbosity=-1,
        )
        model.fit(x.loc[valid], target.loc[valid].astype(np.float32), sample_weight=weights)
        return model
    model = ExtraTreesRegressor(
        n_estimators=260,
        max_depth=7,
        min_samples_leaf=20,
        max_features="sqrt",
        random_state=int(seed),
        n_jobs=2,
    )
    model.fit(x.loc[valid], target.loc[valid].astype(np.float32), sample_weight=weights)
    return model


def _predict(model: Any, x: pd.DataFrame, *, classifier: bool) -> pd.Series:
    if model is None:
        return pd.Series(np.nan, index=x.index, dtype=np.float32)
    if classifier and hasattr(model, "predict_proba"):
        pred = model.predict_proba(x)
        if np.asarray(pred).ndim == 2 and pred.shape[1] >= 2:
            return pd.Series(pred[:, 1], index=x.index, dtype=np.float32)
    return pd.Series(np.asarray(model.predict(x), dtype=np.float32), index=x.index)


def _feature_importance(model: Any, feature_names: list[str], target: str, test_month: str) -> pd.DataFrame:
    if model is None or not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["test_month", "target", "feature", "importance"])
    values = np.asarray(model.feature_importances_, dtype=np.float64)
    if len(values) != len(feature_names):
        return pd.DataFrame(columns=["test_month", "target", "feature", "importance"])
    out = pd.DataFrame({"feature": feature_names, "importance": values})
    out = out[out["importance"] > 0].sort_values("importance", ascending=False).head(100)
    out.insert(0, "target", str(target))
    out.insert(0, "test_month", str(test_month))
    return out


def _robust_numeric_matrices(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    numeric_cols: list[str],
    *,
    clip: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not numeric_cols:
        return (
            np.zeros((len(train), 0), dtype=np.float32),
            np.zeros((len(valid), 0), dtype=np.float32),
            [],
        )
    train_num = train.loc[:, numeric_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid_num = valid.loc[:, numeric_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    finite_cols = [col for col in numeric_cols if train_num[col].notna().sum() >= 10]
    if not finite_cols:
        return (
            np.zeros((len(train), 0), dtype=np.float32),
            np.zeros((len(valid), 0), dtype=np.float32),
            [],
        )
    train_num = train_num.loc[:, finite_cols]
    valid_num = valid_num.loc[:, finite_cols]
    med = train_num.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    q25 = train_num.quantile(0.25, numeric_only=True)
    q75 = train_num.quantile(0.75, numeric_only=True)
    scale = (q75 - q25).replace([np.inf, -np.inf], np.nan).abs().fillna(0.0)
    std = train_num.std(numeric_only=True).replace([np.inf, -np.inf], np.nan).abs().fillna(0.0)
    scale = scale.where(scale > 1e-9, std).where(lambda s: s > 1e-9, 1.0)
    train_scaled = ((train_num.fillna(med) - med) / scale).clip(-float(clip), float(clip)).astype(np.float32)
    valid_scaled = ((valid_num.fillna(med) - med) / scale).clip(-float(clip), float(clip)).astype(np.float32)
    return train_scaled.to_numpy(dtype=np.float32), valid_scaled.to_numpy(dtype=np.float32), finite_cols


def _fit_linear_denoising_ae(
    train_x: np.ndarray,
    valid_x: np.ndarray,
    *,
    n_components: int,
    seed: int,
) -> dict[str, np.ndarray]:
    if train_x.shape[1] == 0:
        n_valid = valid_x.shape[0]
        return {
            "z_valid": np.zeros((n_valid, 0), dtype=np.float32),
            "valid_error": np.full(n_valid, np.nan, dtype=np.float32),
            "valid_error_pct": np.full(n_valid, np.nan, dtype=np.float32),
            "valid_mahal": np.full(n_valid, np.nan, dtype=np.float32),
            "valid_recon": np.zeros_like(valid_x, dtype=np.float32),
        }
    rng = np.random.default_rng(int(seed))
    train_clean = np.asarray(train_x, dtype=np.float32)
    valid_clean = np.asarray(valid_x, dtype=np.float32)
    center = np.nanmean(train_clean, axis=0, dtype=np.float64).astype(np.float32)
    train_centered = np.nan_to_num(train_clean - center, nan=0.0, posinf=0.0, neginf=0.0)
    valid_centered = np.nan_to_num(valid_clean - center, nan=0.0, posinf=0.0, neginf=0.0)
    noisy = train_centered + rng.normal(0.0, 0.03, size=train_centered.shape).astype(np.float32)
    rank = max(1, min(int(n_components), noisy.shape[0] - 1, noisy.shape[1]))
    try:
        _, _, vt = np.linalg.svd(noisy, full_matrices=False)
        components = vt[:rank].astype(np.float32)
    except np.linalg.LinAlgError:
        components = np.eye(noisy.shape[1], dtype=np.float32)[:rank]
    z_train = train_centered @ components.T
    z_valid = valid_centered @ components.T
    train_recon = z_train @ components + center
    valid_recon = z_valid @ components + center
    train_error = np.mean(np.square(train_clean - train_recon), axis=1).astype(np.float32)
    valid_error = np.mean(np.square(valid_clean - valid_recon), axis=1).astype(np.float32)
    sorted_train_error = np.sort(train_error[np.isfinite(train_error)])
    if len(sorted_train_error):
        valid_error_pct = (
            np.searchsorted(sorted_train_error, valid_error, side="right") / max(len(sorted_train_error), 1)
        ).astype(np.float32)
    else:
        valid_error_pct = np.full(len(valid_error), np.nan, dtype=np.float32)
    z_mean = np.nanmean(z_train, axis=0, dtype=np.float64).astype(np.float32)
    z_std = np.nanstd(z_train, axis=0).astype(np.float32)
    z_std = np.where(z_std > 1e-6, z_std, 1.0).astype(np.float32)
    valid_mahal = np.mean(np.square((z_valid - z_mean) / z_std), axis=1).astype(np.float32)
    return {
        "z_valid": z_valid.astype(np.float32),
        "valid_error": valid_error,
        "valid_error_pct": valid_error_pct,
        "valid_mahal": valid_mahal,
        "valid_recon": valid_recon.astype(np.float32),
    }


def _feature_family(col: str) -> str:
    for family, prefixes in AE_FAMILY_PREFIXES.items():
        if col.startswith(prefixes):
            return family
    return "other"


def _add_market_ae_outputs(
    scored: pd.DataFrame,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    numeric_cols: list[str],
    clip: float,
    seed: int,
) -> dict[str, int]:
    train_x, valid_x, finite_cols = _robust_numeric_matrices(train, valid, numeric_cols, clip=clip)
    result = _fit_linear_denoising_ae(train_x, valid_x, n_components=4, seed=seed)
    z_valid = result["z_valid"]
    for idx in range(4):
        scored[f"market_z_{idx}"] = (
            z_valid[:, idx].astype(np.float32) if z_valid.shape[1] > idx else np.full(len(scored), np.nan, dtype=np.float32)
        )
    scored["market_ae_recon_error"] = result["valid_error"]
    scored["market_ae_recon_error_pct"] = result["valid_error_pct"]
    scored["market_ae_mahalanobis_diag"] = result["valid_mahal"]
    valid_recon = result["valid_recon"]
    family_counts: dict[str, int] = {}
    families = {family: [] for family in [*AE_FAMILY_PREFIXES.keys(), "other"]}
    for pos, col in enumerate(finite_cols):
        families[_feature_family(col)].append(pos)
    for family, positions in families.items():
        out_col = f"family_recon_error_{family}"
        family_counts[family] = len(positions)
        if positions and valid_x.shape[1]:
            scored[out_col] = np.mean(np.square(valid_x[:, positions] - valid_recon[:, positions]), axis=1).astype(np.float32)
        else:
            scored[out_col] = np.full(len(scored), np.nan, dtype=np.float32)
    return family_counts


def _select_top(frame: pd.DataFrame, score_col: str, frac: float, *, ascending: bool = False) -> pd.DataFrame:
    valid = frame[_num(frame.get(score_col), index=frame.index).notna()].copy()
    if valid.empty:
        return valid
    n = max(1, int(math.ceil(len(valid) * float(frac))))
    return valid.sort_values(score_col, ascending=ascending, kind="mergesort").head(n)


def _path_metrics(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "exec_margin": _mean(frame.get("exec_margin")),
        "ev_after_1pct": _mean(frame.get("ev_after_1pct")),
        "ret_net": _mean(frame.get("ret_net")),
        "clean_exec_precision": _rate(frame.get("clean_exec")),
        "positive_exec_margin_rate": _rate(_num(frame.get("exec_margin")).gt(0.0)),
        "dirty_positive_rate": _rate(frame.get("dirty_positive")),
        "full_path_bad_mae": _rate(frame.get("full_path_bad_mae_1r")),
        "first_touch_bad_mae": _rate(frame.get("first_touch_bad_mae_1r")),
        "timeout": _rate(frame.get("timeout")),
        "mfe_before_mae": _rate(frame.get("mfe_before_mae_1r")),
        "mae_before_mfe": _rate(frame.get("mae_before_mfe_1r")),
        "max_adverse_before_mfe": _mean(frame.get("max_adverse_before_mfe_1r")),
        "underwater_bars": _mean(frame.get("underwater_bars_before_mfe_1r")),
    }


def _oracle_recall(frame: pd.DataFrame, selected: pd.DataFrame) -> float:
    if selected.empty or "exec_margin" not in frame.columns:
        return float("nan")
    oracle = frame.sort_values("exec_margin", ascending=False, kind="mergesort").head(len(selected))
    overlap = len(set(selected.index.tolist()) & set(oracle.index.tolist()))
    return float(overlap / max(len(oracle), 1))


def _topk_value(frame: pd.DataFrame, score_col: str, frac: float, *, ascending: bool = False) -> dict[str, Any]:
    selected = _select_top(frame, score_col, frac, ascending=ascending)
    out: dict[str, Any] = {
        "selected_rows": int(len(selected)),
        "oracle_recall": _oracle_recall(frame, selected),
    }
    out.update(_path_metrics(selected))
    return out


def _control_scores(scores: pd.Series, frame: pd.DataFrame, *, seed: int) -> dict[str, pd.Series]:
    rng = np.random.default_rng(int(seed))
    base = _num(scores).replace([np.inf, -np.inf], np.nan)
    valid_values = base.dropna().to_numpy(dtype=np.float64)
    if len(valid_values) == 0:
        valid_values = np.array([0.0], dtype=np.float64)
    perm_values = valid_values.copy()
    rng.shuffle(perm_values)
    perm = pd.Series(np.resize(perm_values, len(base)), index=base.index, dtype=np.float32)
    block = pd.Series(np.nan, index=base.index, dtype=np.float32)
    if "month" in frame.columns:
        for _, idx in frame.groupby("month", dropna=False).groups.items():
            vals = base.loc[idx].dropna().to_numpy(dtype=np.float64)
            if len(vals) == 0:
                vals = valid_values
            vals = vals.copy()
            rng.shuffle(vals)
            block.loc[idx] = np.resize(vals, len(idx)).astype(np.float32)
    else:
        block = perm.copy()
    noise = rng.normal(size=len(base)).astype(np.float64)
    for i in range(1, len(noise)):
        noise[i] = 0.75 * noise[i - 1] + noise[i]
    std = float(np.nanstd(valid_values)) if len(valid_values) else 1.0
    mean = float(np.nanmean(valid_values)) if len(valid_values) else 0.0
    noise = (noise - float(np.nanmean(noise))) / max(float(np.nanstd(noise)), 1e-9)
    noise_ar1 = pd.Series(mean + std * noise, index=base.index, dtype=np.float32)
    return {"perm": perm, "block_perm": block, "noise_ar1": noise_ar1}


def _support_stats(frame: pd.DataFrame, train: pd.DataFrame, group_cols: list[str]) -> dict[str, Any]:
    train_cell = train
    valid_cell = frame
    major_asset_share = float(valid_cell["__symbol__"].value_counts(normalize=True).iloc[0]) if len(valid_cell) else float("nan")
    ts = pd.to_datetime(valid_cell["__ts__"], errors="coerce")
    week_share = float(ts.dt.to_period("W").astype(str).value_counts(normalize=True).iloc[0]) if len(valid_cell) else float("nan")
    return {
        "grouping": "+".join(group_cols),
        "train_rows": int(len(train_cell)),
        "valid_rows": int(len(valid_cell)),
        "train_clean_rows": int(_num(train_cell.get("clean_exec"), index=train_cell.index, default=0.0).fillna(0.0).gt(0.5).sum()),
        "valid_clean_rows": int(_num(valid_cell.get("clean_exec"), index=valid_cell.index, default=0.0).fillna(0.0).gt(0.5).sum()),
        "valid_months": int(valid_cell["month"].nunique()) if "month" in valid_cell.columns else 0,
        "max_asset_share": major_asset_share,
        "max_week_share": week_share,
    }


def _is_supported(row: dict[str, Any], args: argparse.Namespace) -> bool:
    return (
        row["train_rows"] >= int(args.min_train_cell_rows)
        and row["valid_rows"] >= int(args.min_valid_cell_rows)
        and row["train_clean_rows"] >= int(args.min_train_clean_rows)
        and row["valid_clean_rows"] >= int(args.min_valid_clean_rows)
        and row["max_asset_share"] <= float(args.max_single_asset_share)
        and row["max_week_share"] <= float(args.max_single_week_share)
    )


def _cell_diagnostics_for_group(
    *,
    scored: pd.DataFrame,
    train: pd.DataFrame,
    group_cols: list[str],
    args: argparse.Namespace,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not all(col in scored.columns and col in train.columns for col in group_cols):
        return rows
    train_groups = {key: part for key, part in train.groupby(group_cols, dropna=False)}
    for key, cell in scored.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        train_cell = train_groups.get(key, train.iloc[0:0])
        base_support = _support_stats(cell, train_cell, group_cols)
        base_support["supported_cell"] = bool(_is_supported(base_support, args))
        for col, value in zip(group_cols, key, strict=False):
            base_support[col] = value
        for score_col, meta in {
            "score": {"ascending": False, "label": "base_score"},
            "cross_lgbm_exec_margin_score": {"ascending": False, "label": "cross_exec_margin"},
            "cross_lgbm_clean_risk_composite": {"ascending": False, "label": "cross_clean_risk_composite"},
            "cross_lgbm_bad_mae_score": {"ascending": True, "label": "cross_low_bad_mae"},
            "cross_lgbm_timeout_score": {"ascending": True, "label": "cross_low_timeout"},
            "cross_lgbm_dirty_positive_score": {"ascending": True, "label": "cross_low_dirty_positive"},
            "market_ae_recon_error_pct": {"ascending": True, "label": "market_ae_low_recon_error"},
            "market_ae_mahalanobis_diag": {"ascending": True, "label": "market_ae_low_mahalanobis"},
        }.items():
            if score_col not in cell.columns:
                continue
            for frac in TOP_FRACS:
                base_metrics = _topk_value(cell, "score", frac, ascending=False) if "score" in cell.columns else {}
                cand_metrics = _topk_value(cell, score_col, frac, ascending=bool(meta["ascending"]))
                control_vals: list[float] = []
                for control_name, control_score in _control_scores(cell[score_col], cell, seed=seed + len(rows)).items():
                    tmp_col = f"__control_{control_name}"
                    tmp = cell.copy()
                    tmp[tmp_col] = control_score
                    control_vals.append(float(_topk_value(tmp, tmp_col, frac, ascending=bool(meta["ascending"]))["exec_margin"]))
                control_median = float(np.nanmedian(control_vals)) if control_vals else float("nan")
                control_std = float(np.nanstd(control_vals)) if control_vals else float("nan")
                row = dict(base_support)
                row.update(
                    {
                        "selector": str(meta["label"]),
                        "score_col": score_col,
                        "top_frac": float(frac),
                        "base_exec_margin": base_metrics.get("exec_margin", float("nan")),
                        "delta_exec_margin_vs_base": cand_metrics["exec_margin"] - base_metrics.get("exec_margin", float("nan")),
                        "delta_clean_precision_vs_base": cand_metrics["clean_exec_precision"] - base_metrics.get("clean_exec_precision", float("nan")),
                        "delta_bad_mae_vs_base": cand_metrics["full_path_bad_mae"] - base_metrics.get("full_path_bad_mae", float("nan")),
                        "delta_timeout_vs_base": cand_metrics["timeout"] - base_metrics.get("timeout", float("nan")),
                        "delta_mfe_before_mae_vs_base": cand_metrics["mfe_before_mae"] - base_metrics.get("mfe_before_mae", float("nan")),
                        "delta_mae_before_mfe_vs_base": cand_metrics["mae_before_mfe"] - base_metrics.get("mae_before_mfe", float("nan")),
                        "control_exec_margin_median": control_median,
                        "control_exec_margin_std": control_std,
                        "control_adjusted_exec_margin": cand_metrics["exec_margin"] - control_median - 0.5 * control_std,
                    }
                )
                row.update({f"top{int(frac * 100):02d}_{k}": v for k, v in cand_metrics.items()})
                rows.append(row)
    return rows


def _fold_metrics(scored: pd.DataFrame, test_month: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    target_score_map = {
        "exec_margin": "cross_lgbm_exec_margin_score",
        "full_path_bad_mae_1r": "cross_lgbm_bad_mae_score",
        "timeout": "cross_lgbm_timeout_score",
        "dirty_positive": "cross_lgbm_dirty_positive_score",
        "clean_exec": "cross_lgbm_clean_risk_composite",
    }
    for target, score_col in target_score_map.items():
        if target not in scored.columns or score_col not in scored.columns:
            continue
        row = {
            "test_month": str(test_month),
            "target": target,
            "score_col": score_col,
            "rows": int(len(scored)),
            "spearman_exec_margin": _safe_spearman(scored.get("exec_margin", pd.Series(dtype=float)), scored[score_col]),
        }
        if target in {"exec_margin"}:
            row["spearman_target"] = _safe_spearman(scored[target], scored[score_col])
            row["auc_target"] = float("nan")
            row["ap_target"] = float("nan")
        elif target == "clean_exec":
            row["spearman_target"] = _safe_spearman(scored[target], scored[score_col])
            row["auc_target"] = _safe_auc(scored[target], scored[score_col])
            row["ap_target"] = _safe_ap(scored[target], scored[score_col])
        else:
            # Risk scores are better when high for the risk class.
            row["spearman_target"] = _safe_spearman(scored[target], scored[score_col])
            row["auc_target"] = _safe_auc(scored[target], scored[score_col])
            row["ap_target"] = _safe_ap(scored[target], scored[score_col])
        rows.append(row)
    return rows


def _summarize_cells(cell_df: pd.DataFrame) -> pd.DataFrame:
    if cell_df.empty:
        return cell_df
    cell_df = cell_df.copy()
    supported = cell_df["supported_cell"].astype(bool)
    non_base = cell_df["selector"].astype(str).ne("base_score")
    control_positive = _num(cell_df["control_adjusted_exec_margin"]).gt(0.0)
    path_or_utility_help = (
        _num(cell_df["delta_exec_margin_vs_base"]).gt(0.0)
        | _num(cell_df["delta_clean_precision_vs_base"]).gt(0.0)
        | _num(cell_df["delta_bad_mae_vs_base"]).lt(0.0)
        | _num(cell_df["delta_timeout_vs_base"]).lt(0.0)
        | _num(cell_df["delta_mfe_before_mae_vs_base"]).gt(0.0)
        | _num(cell_df["delta_mae_before_mfe_vs_base"]).lt(0.0)
    )
    cell_df["supported_control_adjusted_exec_margin"] = np.where(
        supported,
        _num(cell_df["control_adjusted_exec_margin"]),
        np.nan,
    )
    cell_df["positive_supported_cell"] = (supported & non_base & control_positive & path_or_utility_help).astype(int)
    summary = (
        cell_df.groupby(["selector", "top_frac"], dropna=False)
        .agg(
            rows=("score_col", "size"),
            supported_cells=("supported_cell", "sum"),
            mean_delta_exec_margin_vs_base=("delta_exec_margin_vs_base", "mean"),
            mean_delta_clean_precision_vs_base=("delta_clean_precision_vs_base", "mean"),
            mean_delta_bad_mae_vs_base=("delta_bad_mae_vs_base", "mean"),
            mean_delta_timeout_vs_base=("delta_timeout_vs_base", "mean"),
            best_supported_control_adjusted_exec_margin=(
                "supported_control_adjusted_exec_margin",
                lambda x: float(np.nanmax(x)) if len(x.dropna()) else float("nan"),
            ),
            positive_supported_cells=("positive_supported_cell", "sum"),
        )
        .reset_index()
    )
    summary["acceptance_status"] = np.where(
        (summary["supported_cells"] > 0)
        & (summary["positive_supported_cells"] > 0)
        & (summary["best_supported_control_adjusted_exec_margin"] > 0.0),
        "candidate_for_meta_ablation",
        np.where(summary["selector"].astype(str).eq("base_score"), "base_comparator", "diagnostic_or_shadow"),
    )
    return summary.sort_values(
        ["acceptance_status", "best_supported_control_adjusted_exec_margin", "mean_delta_bad_mae_vs_base"],
        ascending=[True, False, True],
    )


def run(
    *,
    handoff_dir: Path,
    ledger_path: Path | None,
    out_dir: Path,
    frontier: str,
    train_scope: str,
    seed: int,
    soft_only: bool,
    clip: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = _load_joined_frame(handoff_dir, ledger_path, frontier)
    selected_col = _candidate_column(frontier)
    if train_scope == "selected":
        data = data[data[selected_col]].copy()
    elif train_scope != "all":
        raise ValueError("--train-scope must be selected or all")
    months = sorted(str(m) for m in data["month"].dropna().unique())
    if len(months) < 2:
        raise ValueError(f"Need at least two months for month-forward scoring, got {months}")
    numeric_cols, categorical_cols = _feature_columns(data, soft_only=soft_only)
    predictions: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    importances: list[pd.DataFrame] = []
    ae_family_counts_by_fold: dict[str, dict[str, int]] = {}
    for fold_idx, test_month in enumerate(months[1:], start=1):
        train = data[data["month"].astype(str).lt(str(test_month))].copy()
        valid = data[data["month"].astype(str).eq(str(test_month))].copy()
        if len(train) < int(args.min_fold_train_rows) or len(valid) < int(args.min_fold_valid_rows):
            continue
        print(
            json.dumps(
                {
                    "event": "cross_asset_representation_fold_start",
                    "test_month": test_month,
                    "train_rows": int(len(train)),
                    "valid_rows": int(len(valid)),
                    "numeric_features": int(len(numeric_cols)),
                    "categorical_features": int(len(categorical_cols)),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        x_train, x_valid, feature_names = _make_xy(
            train,
            valid,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            clip=clip,
        )
        models: dict[str, Any] = {}
        for target, spec in TARGETS.items():
            if target not in train.columns:
                continue
            if spec["kind"] == "classification":
                models[target] = _fit_classifier(x_train, train[target], train, seed + fold_idx + len(models) * 17)
            else:
                models[target] = _fit_regressor(x_train, train[target], train, seed + fold_idx + len(models) * 17)
        scored = valid.copy()
        for target, spec in TARGETS.items():
            score_col = str(spec["score"])
            model = models.get(target)
            scored[score_col] = _predict(model, x_valid, classifier=spec["kind"] == "classification")
            importances.append(_feature_importance(model, feature_names, target, test_month))
        scored["cross_lgbm_clean_risk_composite"] = (
            _num(scored.get("cross_lgbm_exec_margin_score"), index=scored.index, default=0.0).fillna(0.0)
            + 0.0040 * _num(scored.get("cross_lgbm_clean_exec_proxy"), index=scored.index, default=0.0).fillna(0.0)
            - 0.0045 * _num(scored.get("cross_lgbm_bad_mae_score"), index=scored.index, default=1.0).fillna(1.0)
            - 0.0025 * _num(scored.get("cross_lgbm_timeout_score"), index=scored.index, default=1.0).fillna(1.0)
            - 0.0035 * _num(scored.get("cross_lgbm_dirty_positive_score"), index=scored.index, default=1.0).fillna(1.0)
        ).astype(np.float32)
        ae_family_counts_by_fold[str(test_month)] = _add_market_ae_outputs(
            scored,
            train,
            valid,
            numeric_cols=numeric_cols,
            clip=clip,
            seed=seed + 5000 + fold_idx,
        )
        fold_rows.extend(_fold_metrics(scored, test_month))
        groupings = [
            ["side_name", "source_semantic_family"],
            ["side_name", "source_semantic_family_base"],
            ["side_name", "long_source_regime_split"],
            ["side_name", "aegmm_cluster"],
            ["side_name", "source_semantic_family", "aegmm_cluster"],
            ["side_name", "aegmm_entropy_bin"],
            ["side_name", "aegmm_distance_bin"],
            ["side_name", "reconstruction_bin"],
        ]
        for group_cols in groupings:
            cell_rows.extend(
                _cell_diagnostics_for_group(
                    scored=scored,
                    train=train,
                    group_cols=group_cols,
                    args=args,
                    seed=seed + fold_idx * 1000,
                )
            )
        keep_cols = [
            "__ts__",
            "__symbol__",
            "side_name",
            "month",
            "source_semantic_family",
            "source_semantic_family_base",
            "long_source_regime_split",
            "aegmm_cluster",
            "side_aegmm_cluster",
            "aegmm_entropy_bin",
            "aegmm_distance_bin",
            "reconstruction_bin",
            "score",
            "exec_margin",
            "ev_after_1pct",
            "ret_net",
            "clean_exec",
            "dirty_positive",
            "full_path_bad_mae_1r",
            "first_touch_bad_mae_1r",
            "timeout",
            "mfe_before_mae_1r",
            "mae_before_mfe_1r",
            "max_adverse_before_mfe_1r",
            "underwater_bars_before_mfe_1r",
            "cross_lgbm_exec_margin_score",
            "cross_lgbm_bad_mae_score",
            "cross_lgbm_timeout_score",
            "cross_lgbm_dirty_positive_score",
            "cross_lgbm_clean_risk_composite",
            *AE_OUTPUT_COLUMNS,
        ]
        predictions.append(scored[[col for col in keep_cols if col in scored.columns]].copy())
    pred_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    folds_df = pd.DataFrame(fold_rows)
    cells_df = pd.DataFrame(cell_rows)
    cell_summary = _summarize_cells(cells_df)
    importance_df = (
        pd.concat([part for part in importances if not part.empty], ignore_index=True)
        if any(not part.empty for part in importances)
        else pd.DataFrame(columns=["test_month", "target", "feature", "importance"])
    )
    if not pred_df.empty:
        pred_df.to_parquet(out_dir / "cross_asset_representation_v1_predictions.parquet", index=False)
    folds_df.to_csv(out_dir / "cross_asset_representation_v1_fold_metrics.csv", index=False)
    cells_df.to_csv(out_dir / "cross_asset_representation_v1_cell_diagnostics.csv", index=False)
    cell_summary.to_csv(out_dir / "cross_asset_representation_v1_summary.csv", index=False)
    importance_df.to_csv(out_dir / "cross_asset_representation_v1_feature_importance.csv", index=False)
    manifest = {
        "generated_by": "run_cross_asset_archetype_representation_v1",
        "handoff_dir": str(handoff_dir),
        "ledger_path": str(ledger_path) if ledger_path is not None else str(handoff_dir / "s52_trailing_regime_scored_ledger.parquet"),
        "out_dir": str(out_dir),
        "frontier": str(frontier),
        "train_scope": str(train_scope),
        "months": months,
        "scored_months": sorted(pred_df["month"].astype(str).unique().tolist()) if not pred_df.empty else [],
        "rows": int(len(data)),
        "prediction_rows": int(len(pred_df)),
        "numeric_feature_count": int(len(numeric_cols)),
        "categorical_feature_count": int(len(categorical_cols)),
        "feature_columns": numeric_cols + categorical_cols,
        "excluded_outcome_columns": sorted(OUTCOME_COLUMNS),
        "ae_output_columns": list(AE_OUTPUT_COLUMNS),
        "ae_family_counts_by_fold": ae_family_counts_by_fold,
        "soft_only_gmm_inputs": bool(soft_only),
        "lightgbm_available": bool(_LIGHTGBM_AVAILABLE),
        "model_a_status": "implemented",
        "model_b_status": "implemented_compact_linear_denoising_ae",
        "support_thresholds": {
            "min_train_cell_rows": int(args.min_train_cell_rows),
            "min_valid_cell_rows": int(args.min_valid_cell_rows),
            "min_train_clean_rows": int(args.min_train_clean_rows),
            "min_valid_clean_rows": int(args.min_valid_clean_rows),
            "max_single_asset_share": float(args.max_single_asset_share),
            "max_single_week_share": float(args.max_single_week_share),
        },
        "leakage_contract": {
            "split": "month_forward_train_past_validate_next_month",
            "scaler": "robust median/IQR fit on train fold only",
            "features": "handoff pre-entry live-predictable columns only",
            "targets": "ledger outcomes joined only for training labels and validation diagnostics",
            "representation_outputs": "OOF/prior-fold for scored months",
            "promotion_scope": "candidate meta features only, not hard gates",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))
    _write_markdown(out_dir, manifest, folds_df, cell_summary)
    return manifest


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        val = float(value)
    except Exception:
        return "nan"
    if not math.isfinite(val):
        return "nan"
    return f"{val:.{digits}f}"


def _write_markdown(out_dir: Path, manifest: dict[str, Any], folds: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = [
        "# Cross-Asset Archetype Representation V1",
        "",
        "## Scope",
        "",
        "Model A cross-market LGBM challenger plus Model B compact linear denoising-AE/OOD outputs with month-forward OOF representation outputs.",
        "Outputs are candidate meta features only; they are not hard gates or frozen replay evidence.",
        "",
        "## Contract",
        "",
        f"- rows: `{manifest.get('rows')}`",
        f"- prediction rows: `{manifest.get('prediction_rows')}`",
        f"- train scope: `{manifest.get('train_scope')}`",
        f"- months: `{', '.join(manifest.get('months', []))}`",
        f"- scored months: `{', '.join(manifest.get('scored_months', []))}`",
        f"- numeric features: `{manifest.get('numeric_feature_count')}`",
        f"- categorical features: `{manifest.get('categorical_feature_count')}`",
        f"- AE/OOD outputs: `{len(manifest.get('ae_output_columns', []))}`",
        f"- hard GMM ids excluded from inputs: `{manifest.get('soft_only_gmm_inputs')}`",
        "",
        "## Fold Metrics",
        "",
    ]
    if folds.empty:
        lines.append("No valid folds produced.")
    else:
        display_cols = ["test_month", "target", "score_col", "rows", "spearman_target", "auc_target", "ap_target", "spearman_exec_margin"]
        lines.append(folds[[col for col in display_cols if col in folds.columns]].to_markdown(index=False))
    lines.extend(["", "## Acceptance Summary", ""])
    if summary.empty:
        lines.append("No cell diagnostics produced.")
    else:
        display_cols = [
            "selector",
            "top_frac",
            "acceptance_status",
            "supported_cells",
            "positive_supported_cells",
            "best_supported_control_adjusted_exec_margin",
            "mean_delta_exec_margin_vs_base",
            "mean_delta_bad_mae_vs_base",
            "mean_delta_timeout_vs_base",
        ]
        lines.append(summary[[col for col in display_cols if col in summary.columns]].head(30).to_markdown(index=False))
    lines.extend(
        [
            "",
            "## Leakage Notes",
            "",
            "- Scalers and models are fit on prior months only.",
            "- Outcome/path columns are excluded from features and used only as labels/diagnostics.",
            "- Representation outputs are OOF/prior-fold for scored months.",
            "- Model B uses a compact train-fold linear denoising AE/PCA-style encoder; reconstruction quality alone is diagnostic, not promotion evidence.",
            "",
        ]
    )
    (out_dir / "cross_asset_representation_v1_report.md").write_text("\n".join(lines))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-dir", type=Path, default=DEFAULT_HANDOFF_DIR)
    parser.add_argument("--ledger-path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--frontier", default="10", help="Candidate frontier suffix, e.g. 10 for selected_top10.")
    parser.add_argument("--train-scope", choices=("selected", "all"), default="selected")
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--clip", type=float, default=8.0)
    parser.add_argument("--include-hard-gmm-id", action="store_true", help="Allow hard GMM IDs as model inputs for an explicit ablation.")
    parser.add_argument("--min-fold-train-rows", type=int, default=100)
    parser.add_argument("--min-fold-valid-rows", type=int, default=30)
    parser.add_argument("--min-train-cell-rows", type=int, default=100)
    parser.add_argument("--min-valid-cell-rows", type=int, default=30)
    parser.add_argument("--min-train-clean-rows", type=int, default=10)
    parser.add_argument("--min-valid-clean-rows", type=int, default=5)
    parser.add_argument("--max-single-asset-share", type=float, default=0.80)
    parser.add_argument("--max-single-week-share", type=float, default=0.80)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run(
        handoff_dir=args.handoff_dir,
        ledger_path=args.ledger_path,
        out_dir=args.out_dir,
        frontier=args.frontier,
        train_scope=args.train_scope,
        seed=args.seed,
        soft_only=not bool(args.include_hard_gmm_id),
        clip=args.clip,
        args=args,
    )
    print(json.dumps(_json_safe({"event": "cross_asset_representation_done", **manifest}), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
