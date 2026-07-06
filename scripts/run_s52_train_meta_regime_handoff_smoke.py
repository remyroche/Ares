#!/usr/bin/env python3
"""Month-forward train_meta smoke for S52 regime handoff artifacts.

This is a controlled diagnostic, not production train_meta.  It consumes the
row-level ``train_meta_regime_handoff.parquet`` produced by
``report_s52_trailing_regime_meta_handoff.py`` and evaluates whether those
pre-entry regime/source/action features help filter a base candidate frontier.

Leakage contract:

* model inputs come from the handoff artifact plus the base score only;
* outcome columns are joined only as train labels and validation metrics;
* each validation month is scored by models fit on strictly earlier months;
* top-k precision/EV/path metrics are primary, AUC is secondary.
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

from scripts.run_label_weighted_proxy_ablation import _weight_series as _base_weight_series  # noqa: E402
from extreme_price_movements.lgbm_pipeline import train_lgbm_stability_candidate  # noqa: E402


DEFAULT_REPORT_ROOT = Path(
    "data_perp/reports/s52_trailing_profit_best_pointwise_scored_ledger_20260705_v1"
)
DEFAULT_HANDOFF_DIR = DEFAULT_REPORT_ROOT / "s52_trailing_regime_meta_handoff_v1"
DEFAULT_LEDGER = DEFAULT_HANDOFF_DIR / "s52_trailing_regime_scored_ledger.parquet"
DEFAULT_OUT_DIR = DEFAULT_HANDOFF_DIR / "train_meta_regime_handoff_smoke_v1"

KEY_COLUMNS = ("__ts__", "__symbol__", "side_name")
TOP_KEEP_FRACTIONS = (1.00, 0.50, 0.30, 0.20, 0.15, 0.10, 0.05)
POLICY_BUDGET_FRACTIONS = (0.30, 0.20, 0.15, 0.10)
BAD_PATH_CAPS = (0.45, 0.50, 0.55, 0.60, 0.65)
CLEAN_EXEC_FLOORS = (0.0, 0.45, 0.55, 0.65)
POSITIVE_MARGIN_FLOORS = (0.0, 0.45, 0.55)
OUTCOME_COLUMNS = (
    "__first_touch_target_soft__",
    "__first_touch_policy_soft__",
    "base_model_target_mode",
    "exec_margin",
    "ev_after_1pct",
    "first_touch_gross",
    "first_touch_bad_mae_1r",
    "full_path_bad_mae_1r",
    "timeout",
    "mfe_before_mae_1r",
    "mae_before_mfe_1r",
    "clean_exec",
    "dirty_positive",
    "u_policy_net",
    "ret_net",
    "mae_norm",
    "mfe_norm",
    "first_touch_net",
    "first_touch_full_path_mae_norm",
    "underwater_bars_before_mfe_1r",
    "long_path_full_bad_mae_1r",
    "long_path_time_to_profit_bars",
    "long_path_slow_profit",
    "long_path_post_mfe_drawdown_norm",
    "long_path_post_mfe_bad_drawdown",
    "long_path_clean_exec_label",
    "long_path_dirty_positive_label",
    "long_path_quality_soft",
    "long_bad_path_label",
)
BASE_SOFT_LABEL_COLUMNS = (
    "__first_touch_target_soft__",
    "target_soft",
    "__target_soft__",
)
LEDGER_CONTEXT_COLUMNS = (
    "__archetype_label_family__",
    "__archetype_label_source__",
    "__archetype_policy_key__",
    "__archetype_policy_role__",
    "__archetype_policy_confidence__",
    "__archetype_policy_tp_r__",
    "__archetype_policy_sl_r__",
    "__archetype_policy_trail_r__",
    "__archetype_policy_max_bars_to_mfe__",
    "__archetype_policy_max_barrier__",
)
LEDGER_CONTEXT_FEATURE_ALIASES = {
    "__archetype_label_family__": "archetype_label_family",
    "__archetype_label_source__": "archetype_label_source",
    "__archetype_policy_key__": "archetype_policy_key",
    "__archetype_policy_role__": "archetype_policy_role",
    "__archetype_policy_confidence__": "archetype_policy_confidence",
    "__archetype_policy_tp_r__": "archetype_policy_tp_r",
    "__archetype_policy_sl_r__": "archetype_policy_sl_r",
    "__archetype_policy_trail_r__": "archetype_policy_trail_r",
    "__archetype_policy_max_bars_to_mfe__": "archetype_policy_max_bars_to_mfe",
    "__archetype_policy_max_barrier__": "archetype_policy_max_barrier",
}
NEVER_FEATURE_COLUMNS = {
    "__ts__",
    "__symbol__",
    "month",
    "selected_top10",
    "selected_top20",
    "selected_top30",
}
BASE_PRIOR_NUMERIC_FEATURES = (
    "base_margin_to_cutoff",
    "base_margin_to_cutoff_z",
    "base_signal_zscore_within_archetype",
    "base_score_rank_pct_train_prior",
)
BASE_PRIOR_CATEGORICAL_FEATURES = (
    "base_rank_band",
    "base_margin_band",
)
RELIABILITY_NUMERIC_FEATURES = (
    "rel_rankband_rows_log1p",
    "rel_rankband_clean_rate",
    "rel_rankband_bad_mae_rate",
    "rel_rankband_timeout_rate",
    "rel_rankband_dirty_positive_rate",
    "rel_rankband_exec_margin_mean",
    "rel_rankband_edge",
    "rel_marginband_rows_log1p",
    "rel_marginband_clean_rate",
    "rel_marginband_bad_mae_rate",
    "rel_marginband_timeout_rate",
    "rel_marginband_dirty_positive_rate",
    "rel_marginband_exec_margin_mean",
    "rel_marginband_edge",
)
SUPPORT_DRIFT_NUMERIC_FEATURES = (
    "support_min_log_count",
    "support_mean_log_count",
    "support_min_frequency",
    "support_mean_frequency",
    "support_unseen_bucket_count",
    "support_unseen_bucket_share",
    "support_rare_bucket_count",
    "support_rare_bucket_share",
)
HIT_SURPRISE_HALFLIFE_DAYS = (3.0, 7.0, 14.0)
HIT_SURPRISE_NUMERIC_FEATURES = tuple(
    name
    for hl in (3, 7, 14)
    for name in (
        f"base_arch_hit_recent_rate_hl{hl}d",
        f"base_arch_hit_expected_rate_hl{hl}d",
        f"base_arch_hit_surprise_hl{hl}d",
        f"base_arch_hit_surprise_z_hl{hl}d",
        f"base_arch_hit_support_log1p_hl{hl}d",
        f"base_arch_hit_effective_n_hl{hl}d",
    )
)
SUPPORT_DRIFT_COLUMNS = (
    "source_tag",
    "source_semantic_family",
    "source_semantic_family_base",
    "long_source_regime_split",
    "aegmm_cluster",
    "side_aegmm_cluster",
    "aegmm_entropy_bin",
    "aegmm_distance_bin",
    "aegmm_expected_distance_bin",
    "reconstruction_bin",
    "dae_reconstruction_bin",
    "latent_speed_bin",
    "regime_lgbm_leaf_bad_mae_k4",
    "regime_lgbm_leaf_exec_margin_k4",
    "regime_first_touch_bad_mae_score_bin",
    "regime_timeout_score_bin",
    "regime_dirty_positive_score_bin",
    "regime_clean_exec_score_bin",
)
DEFAULT_LGBM_CLASSIFIER_PARAMS = {
    "n_estimators": 180,
    "learning_rate": 0.035,
    "num_leaves": 17,
    "max_depth": -1,
    "min_child_samples": 35,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.10,
    "reg_lambda": 8.0,
}
DEFAULT_LGBM_REGRESSOR_PARAMS = {
    "n_estimators": 220,
    "learning_rate": 0.035,
    "num_leaves": 17,
    "max_depth": -1,
    "min_child_samples": 35,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.10,
    "reg_lambda": 10.0,
}
META_HPO_PRESETS = (
    {
        "name": "base_winner_target_soft_w7",
        "classifier": {
            "n_estimators": 147,
            "learning_rate": 0.023186706725487574,
            "num_leaves": 15,
            "max_depth": 4,
            "min_child_samples": 98,
            "subsample": 0.7858416187773574,
            "colsample_bytree": 0.9119108731134247,
            "reg_alpha": 0.027300426078104418,
            "reg_lambda": 0.34080226291305005,
        },
        "regressor": {
            "n_estimators": 147,
            "learning_rate": 0.023186706725487574,
            "num_leaves": 15,
            "max_depth": 4,
            "min_child_samples": 98,
            "subsample": 0.7858416187773574,
            "colsample_bytree": 0.9119108731134247,
            "reg_alpha": 0.027300426078104418,
            "reg_lambda": 0.34080226291305005,
        },
    },
    {
        "name": "baseline",
        "classifier": DEFAULT_LGBM_CLASSIFIER_PARAMS,
        "regressor": DEFAULT_LGBM_REGRESSOR_PARAMS,
    },
    {
        "name": "regularized_slow",
        "classifier": {
            "n_estimators": 260,
            "learning_rate": 0.022,
            "num_leaves": 15,
            "max_depth": 4,
            "min_child_samples": 60,
            "subsample": 0.78,
            "colsample_bytree": 0.70,
            "reg_alpha": 0.25,
            "reg_lambda": 14.0,
        },
        "regressor": {
            "n_estimators": 300,
            "learning_rate": 0.022,
            "num_leaves": 15,
            "max_depth": 4,
            "min_child_samples": 60,
            "subsample": 0.78,
            "colsample_bytree": 0.70,
            "reg_alpha": 0.25,
            "reg_lambda": 16.0,
        },
    },
    {
        "name": "wider_context",
        "classifier": {
            "n_estimators": 220,
            "learning_rate": 0.028,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 45,
            "subsample": 0.82,
            "colsample_bytree": 0.82,
            "reg_alpha": 0.08,
            "reg_lambda": 8.0,
        },
        "regressor": {
            "n_estimators": 260,
            "learning_rate": 0.028,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 45,
            "subsample": 0.82,
            "colsample_bytree": 0.82,
            "reg_alpha": 0.08,
            "reg_lambda": 10.0,
        },
    },
)
_FEATURE_SELECTION_CACHE: dict[tuple[Any, ...], tuple[list[str], pd.DataFrame]] = {}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
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
    if pd.isna(value):
        return None
    return value


def _num(values: Any, *, index: pd.Index | None = None, default: float = np.nan) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    if values is None:
        if index is None:
            return pd.Series(dtype=np.float32)
        return pd.Series(default, index=index, dtype=np.float32)
    return pd.to_numeric(pd.Series(values, index=index), errors="coerce")


def _mean(values: Any) -> float:
    arr = _num(values).replace([np.inf, -np.inf], np.nan).dropna()
    return float(arr.mean()) if len(arr) else float("nan")


def _rate(values: Any) -> float:
    arr = _num(values).replace([np.inf, -np.inf], np.nan).dropna()
    return float(arr.clip(0.0, 1.0).mean()) if len(arr) else float("nan")


def _safe_spearman(y_true: pd.Series, score: pd.Series) -> float:
    y = _num(y_true)
    s = _num(score)
    valid = y.notna() & s.notna()
    if int(valid.sum()) < 20 or y.loc[valid].nunique(dropna=True) < 2 or s.loc[valid].nunique(dropna=True) < 2:
        return float("nan")
    return float(y.loc[valid].rank(method="average").corr(s.loc[valid].rank(method="average")))


def _safe_auc(y_true: pd.Series, score: pd.Series) -> float:
    y = _num(y_true)
    s = _num(score)
    valid = y.notna() & s.notna()
    if int(valid.sum()) < 20 or y.loc[valid].nunique(dropna=True) < 2:
        return float("nan")
    return float(roc_auc_score(y.loc[valid].astype(int), s.loc[valid]))


def _safe_ap(y_true: pd.Series, score: pd.Series) -> float:
    y = _num(y_true)
    s = _num(score)
    valid = y.notna() & s.notna()
    if int(valid.sum()) < 20 or y.loc[valid].nunique(dropna=True) < 2:
        return float("nan")
    return float(average_precision_score(y.loc[valid].astype(int), s.loc[valid]))


def _candidate_column(frontier: str) -> str:
    normalized = str(frontier).lower().replace("top", "")
    return f"selected_top{int(normalized)}"


def _load_joined_frame(handoff_path: Path, ledger_path: Path, frontier: str) -> pd.DataFrame:
    handoff = pd.read_parquet(handoff_path)
    ledger_cols = list(KEY_COLUMNS) + ["month", "score", _candidate_column(frontier)]
    ledger_cols += [col for col in OUTCOME_COLUMNS if col not in ledger_cols]
    ledger_cols += [col for col in LEDGER_CONTEXT_COLUMNS if col not in ledger_cols]
    try:
        import pyarrow.parquet as pq

        ledger_schema_cols = set(pq.read_schema(ledger_path).names)
    except Exception:
        ledger_schema_cols = set(pd.read_parquet(ledger_path).columns)
    ledger = pd.read_parquet(ledger_path, columns=[col for col in ledger_cols if col in ledger_schema_cols])
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
    for col in ("month", "score", _candidate_column(frontier), *LEDGER_CONTEXT_COLUMNS):
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
    selected_col = _candidate_column(frontier)
    if selected_col not in merged.columns:
        raise ValueError(f"Missing candidate frontier column {selected_col!r}")
    merged[selected_col] = _num(merged[selected_col]).fillna(0.0).gt(0.5)
    merged["positive_exec_margin"] = _num(merged.get("exec_margin"), index=merged.index).gt(0.0).astype(float)
    merged["clean_exec_label"] = _num(merged.get("clean_exec"), index=merged.index, default=0.0).fillna(0.0).gt(0.5).astype(float)
    merged["bad_path_label"] = (
        _num(merged.get("full_path_bad_mae_1r"), index=merged.index, default=0.0).fillna(0.0).gt(0.5)
        | _num(merged.get("timeout"), index=merged.index, default=0.0).fillna(0.0).gt(0.5)
    ).astype(float)
    merged["path_mfe_before_mae_label"] = (
        _num(merged.get("mfe_before_mae_1r"), index=merged.index, default=0.0).fillna(0.0).gt(0.5).astype(float)
    )
    merged["path_mae_before_mfe_label"] = (
        _num(merged.get("mae_before_mfe_1r"), index=merged.index, default=0.0).fillna(0.0).gt(0.5).astype(float)
    )
    underwater = _num(merged.get("underwater_bars_before_mfe_1r"), index=merged.index, default=np.nan)
    merged["path_underwater_duration_target"] = np.log1p(underwater.clip(lower=0.0)).astype(np.float32)
    is_long = merged["side_name"].astype(str).str.lower().eq("long")
    if "long_path_clean_exec_label" not in merged.columns:
        merged["long_path_clean_exec_label"] = np.where(is_long, merged["clean_exec_label"], np.nan).astype(np.float32)
    if "long_path_dirty_positive_label" not in merged.columns:
        merged["long_path_dirty_positive_label"] = np.where(
            is_long,
            _num(merged.get("dirty_positive"), index=merged.index, default=0.0).fillna(0.0),
            np.nan,
        ).astype(np.float32)
    if "long_path_post_mfe_bad_drawdown" not in merged.columns:
        merged["long_path_post_mfe_bad_drawdown"] = np.nan
    if "long_path_slow_profit" not in merged.columns:
        merged["long_path_slow_profit"] = np.nan
    merged["long_bad_path_label"] = np.where(
        is_long,
        (
            _num(merged.get("long_path_dirty_positive_label"), index=merged.index, default=0.0).fillna(0.0).gt(0.5)
            | _num(merged.get("long_path_post_mfe_bad_drawdown"), index=merged.index, default=0.0).fillna(0.0).gt(0.5)
            | _num(merged.get("long_path_slow_profit"), index=merged.index, default=0.0).fillna(0.0).gt(0.5)
        ).astype(float),
        np.nan,
    ).astype(np.float32)
    # Pre-entry base-score context available at decision time from the base
    # candidate cross-section. Fold-specific train-prior versions are recomputed
    # inside run_smoke before model fitting to avoid validation-month priors.
    score = _num(merged.get("score"), index=merged.index)
    ts = pd.to_datetime(merged["__ts__"], utc=True, errors="coerce") if "__ts__" in merged.columns else None
    if ts is not None:
        merged["base_rank_pct_by_timestamp"] = score.groupby(ts).rank(pct=True).astype(np.float32)
        merged["base_rank_pct_by_timestamp_side"] = score.groupby([ts, merged["side_name"].astype(str)]).rank(pct=True).astype(np.float32)
        mean_ts = score.groupby(ts).transform("mean")
        std_ts = score.groupby(ts).transform("std").replace(0.0, np.nan)
        merged["base_score_z_by_timestamp"] = ((score - mean_ts) / std_ts).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
        mean_tss = score.groupby([ts, merged["side_name"].astype(str)]).transform("mean")
        std_tss = score.groupby([ts, merged["side_name"].astype(str)]).transform("std").replace(0.0, np.nan)
        merged["base_score_z_by_timestamp_side"] = ((score - mean_tss) / std_tss).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    for col in BASE_PRIOR_NUMERIC_FEATURES:
        if col not in merged.columns:
            merged[col] = np.float32(0.0)
    for col in BASE_PRIOR_CATEGORICAL_FEATURES:
        if col not in merged.columns:
            merged[col] = "missing"
    for col in RELIABILITY_NUMERIC_FEATURES:
        if col not in merged.columns:
            merged[col] = np.float32(0.0)
    for col in SUPPORT_DRIFT_NUMERIC_FEATURES:
        if col not in merged.columns:
            merged[col] = np.float32(0.0)
    for col in HIT_SURPRISE_NUMERIC_FEATURES:
        if col not in merged.columns:
            merged[col] = np.float32(0.0)
    for source_col, alias_col in LEDGER_CONTEXT_FEATURE_ALIASES.items():
        if source_col in merged.columns and alias_col not in merged.columns:
            merged[alias_col] = merged[source_col]
    return merged


def _archetype_key(frame: pd.DataFrame) -> pd.Series:
    if "source_tag" in frame.columns:
        return frame["source_tag"].astype(str)
    if "source_semantic_family" in frame.columns:
        return frame["side_name"].astype(str).str.lower() + "__" + frame["source_semantic_family"].astype(str)
    return frame["side_name"].astype(str).str.lower() + "__unknown"


def _quantile_labels_from_train(
    train_values: pd.Series,
    values: pd.Series,
    *,
    prefix: str,
    q: int,
) -> pd.Series:
    train_clean = pd.to_numeric(train_values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    out = pd.Series(f"{prefix}__missing", index=values.index, dtype=object)
    if train_clean.nunique(dropna=True) < 2:
        return out
    probs = np.linspace(0.0, 1.0, int(q) + 1)[1:-1]
    edges = np.unique(train_clean.quantile(probs).to_numpy(dtype=np.float64))
    vals = pd.to_numeric(values, errors="coerce")
    bins = np.searchsorted(edges, vals.to_numpy(dtype=np.float64), side="right")
    labels = np.asarray([f"{prefix}__q{int(i)}" for i in bins], dtype=object)
    out.loc[vals.notna()] = labels[vals.notna().to_numpy()]
    return out


def _empirical_pct_from_train(train_values: pd.Series, values: pd.Series) -> pd.Series:
    train_clean = pd.to_numeric(train_values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().sort_values()
    vals = pd.to_numeric(values, errors="coerce")
    out = pd.Series(np.nan, index=values.index, dtype=np.float32)
    if train_clean.empty:
        return out.fillna(0.5)
    ranks = np.searchsorted(train_clean.to_numpy(dtype=np.float64), vals.to_numpy(dtype=np.float64), side="right")
    pct = ranks / max(len(train_clean), 1)
    out.loc[vals.notna()] = pct[vals.notna().to_numpy()].astype(np.float32)
    return out.fillna(0.5).clip(0.0, 1.0).astype(np.float32)


def _add_fold_base_prior_features(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    selected_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add validation-safe base-score priors fit on the training months only."""

    train = train.copy()
    valid = valid.copy()
    train_score = _num(train.get("score"), index=train.index)
    valid_score = _num(valid.get("score"), index=valid.index)
    train_arch = _archetype_key(train)
    valid_arch = _archetype_key(valid)
    train_keys = pd.DataFrame({"side": train["side_name"].astype(str).str.lower(), "arch": train_arch}, index=train.index)
    valid_keys = pd.DataFrame({"side": valid["side_name"].astype(str).str.lower(), "arch": valid_arch}, index=valid.index)
    selected = train.get(selected_col, pd.Series(False, index=train.index)).astype(bool)
    global_cutoff = float(train_score.loc[selected & train_score.notna()].min()) if bool((selected & train_score.notna()).any()) else float(train_score.quantile(0.90))
    global_mean = float(train_score.mean()) if train_score.notna().any() else 0.0
    global_std = float(train_score.std()) if train_score.notna().sum() > 1 else 1.0
    if not np.isfinite(global_std) or global_std <= 1e-12:
        global_std = 1.0

    prior_rows: list[dict[str, Any]] = []
    train_tmp = train_keys.copy()
    train_tmp["score"] = train_score
    train_tmp["selected"] = selected
    for (side, arch), group in train_tmp.groupby(["side", "arch"], dropna=False):
        scores = pd.to_numeric(group["score"], errors="coerce").dropna()
        selected_scores = pd.to_numeric(group.loc[group["selected"].astype(bool), "score"], errors="coerce").dropna()
        cutoff = float(selected_scores.min()) if len(selected_scores) else (float(scores.quantile(0.90)) if len(scores) else global_cutoff)
        mean = float(scores.mean()) if len(scores) else global_mean
        std = float(scores.std()) if len(scores) > 1 else global_std
        if not np.isfinite(std) or std <= 1e-12:
            std = global_std
        prior_rows.append({"side": side, "arch": arch, "cutoff": cutoff, "mean": mean, "std": std, "rows": int(len(group))})
    priors = pd.DataFrame(prior_rows)

    def attach(frame: pd.DataFrame, keys: pd.DataFrame, score: pd.Series) -> pd.DataFrame:
        joined = keys.merge(priors, on=["side", "arch"], how="left")
        cutoff = pd.to_numeric(joined["cutoff"], errors="coerce").fillna(global_cutoff)
        mean = pd.to_numeric(joined["mean"], errors="coerce").fillna(global_mean)
        std = pd.to_numeric(joined["std"], errors="coerce").replace(0.0, np.nan).fillna(global_std)
        frame["base_margin_to_cutoff"] = (score.reset_index(drop=True) - cutoff).astype(np.float32).to_numpy()
        frame["base_margin_to_cutoff_z"] = ((score.reset_index(drop=True) - cutoff) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32).to_numpy()
        frame["base_signal_zscore_within_archetype"] = ((score.reset_index(drop=True) - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32).to_numpy()
        return frame

    train = attach(train, train_keys, train_score)
    valid = attach(valid, valid_keys, valid_score)
    train["base_score_rank_pct_train_prior"] = _empirical_pct_from_train(train_score, train_score)
    valid["base_score_rank_pct_train_prior"] = _empirical_pct_from_train(train_score, valid_score)
    train["base_rank_band"] = _quantile_labels_from_train(train_score, train_score, prefix="base_rank_band", q=5)
    valid["base_rank_band"] = _quantile_labels_from_train(train_score, valid_score, prefix="base_rank_band", q=5)
    train["base_margin_band"] = _quantile_labels_from_train(train["base_margin_to_cutoff"], train["base_margin_to_cutoff"], prefix="base_margin_band", q=5)
    valid["base_margin_band"] = _quantile_labels_from_train(train["base_margin_to_cutoff"], valid["base_margin_to_cutoff"], prefix="base_margin_band", q=5)
    return train, valid


def _add_fold_reliability_features(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    shrinkage_k: float = 60.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add train-derived reliability priors by side/archetype/base band.

    Validation rows receive only statistics computed on earlier training
    months. Training rows use leave-one-out group statistics to avoid directly
    injecting their own label into the feature value.
    """

    train = train.copy()
    valid = valid.copy()
    train_arch = _archetype_key(train).astype(str)
    valid_arch = _archetype_key(valid).astype(str)
    train["_rel_side"] = train["side_name"].astype(str).str.lower()
    valid["_rel_side"] = valid["side_name"].astype(str).str.lower()
    train["_rel_arch"] = train_arch
    valid["_rel_arch"] = valid_arch

    clean = _num(train.get("clean_exec_label"), index=train.index, default=0.0).fillna(0.0).clip(0.0, 1.0)
    bad = _num(train.get("full_path_bad_mae_1r"), index=train.index, default=0.0).fillna(0.0).clip(0.0, 1.0)
    timeout = _num(train.get("timeout"), index=train.index, default=0.0).fillna(0.0).clip(0.0, 1.0)
    dirty = _num(train.get("dirty_positive"), index=train.index, default=0.0).fillna(0.0).clip(0.0, 1.0)
    exec_margin = _num(train.get("exec_margin"), index=train.index, default=0.0).fillna(0.0)
    global_stats = {
        "clean": float(clean.mean()) if len(clean) else 0.0,
        "bad": float(bad.mean()) if len(bad) else 0.0,
        "timeout": float(timeout.mean()) if len(timeout) else 0.0,
        "dirty": float(dirty.mean()) if len(dirty) else 0.0,
        "exec": float(exec_margin.mean()) if len(exec_margin) else 0.0,
    }
    global_stats["edge"] = (
        global_stats["clean"] - global_stats["bad"] - 0.50 * global_stats["timeout"] + global_stats["exec"]
    )
    work = train[["_rel_side", "_rel_arch"]].copy()
    work["_clean"] = clean
    work["_bad"] = bad
    work["_timeout"] = timeout
    work["_dirty"] = dirty
    work["_exec"] = exec_margin

    def attach_for_band(band_col: str, prefix: str) -> None:
        nonlocal train, valid
        if band_col not in train.columns:
            train[band_col] = f"{band_col}__missing"
        if band_col not in valid.columns:
            valid[band_col] = f"{band_col}__missing"
        tmp = work.copy()
        tmp["_band"] = train[band_col].astype(str).fillna("missing")
        grouped = (
            tmp.groupby(["_rel_side", "_rel_arch", "_band"], dropna=False)
            .agg(
                rows=("_clean", "size"),
                clean_sum=("_clean", "sum"),
                bad_sum=("_bad", "sum"),
                timeout_sum=("_timeout", "sum"),
                dirty_sum=("_dirty", "sum"),
                exec_sum=("_exec", "sum"),
            )
            .reset_index()
        )

        def fill_from_stats(frame: pd.DataFrame, *, loo: bool) -> pd.DataFrame:
            keys = pd.DataFrame(
                {
                    "_rel_side": frame["_rel_side"].astype(str),
                    "_rel_arch": frame["_rel_arch"].astype(str),
                    "_band": frame[band_col].astype(str).fillna("missing"),
                },
                index=frame.index,
            )
            merged = keys.merge(grouped, on=["_rel_side", "_rel_arch", "_band"], how="left")
            rows = pd.to_numeric(merged["rows"], errors="coerce").fillna(0.0)
            clean_sum = pd.to_numeric(merged["clean_sum"], errors="coerce").fillna(0.0)
            bad_sum = pd.to_numeric(merged["bad_sum"], errors="coerce").fillna(0.0)
            timeout_sum = pd.to_numeric(merged["timeout_sum"], errors="coerce").fillna(0.0)
            dirty_sum = pd.to_numeric(merged["dirty_sum"], errors="coerce").fillna(0.0)
            exec_sum = pd.to_numeric(merged["exec_sum"], errors="coerce").fillna(0.0)
            if loo:
                rows = (rows - 1.0).clip(lower=0.0)
                clean_sum = (clean_sum - _num(frame.get("clean_exec_label"), index=frame.index, default=0.0).reset_index(drop=True).fillna(0.0)).clip(lower=0.0)
                bad_sum = (bad_sum - _num(frame.get("full_path_bad_mae_1r"), index=frame.index, default=0.0).reset_index(drop=True).fillna(0.0)).clip(lower=0.0)
                timeout_sum = (timeout_sum - _num(frame.get("timeout"), index=frame.index, default=0.0).reset_index(drop=True).fillna(0.0)).clip(lower=0.0)
                dirty_sum = (dirty_sum - _num(frame.get("dirty_positive"), index=frame.index, default=0.0).reset_index(drop=True).fillna(0.0)).clip(lower=0.0)
                exec_sum = exec_sum - _num(frame.get("exec_margin"), index=frame.index, default=0.0).reset_index(drop=True).fillna(0.0)
            denom = rows.replace(0.0, np.nan)
            weight = (rows / (rows + float(shrinkage_k))).clip(0.0, 1.0)
            clean_rate = (clean_sum / denom).fillna(global_stats["clean"])
            bad_rate = (bad_sum / denom).fillna(global_stats["bad"])
            timeout_rate = (timeout_sum / denom).fillna(global_stats["timeout"])
            dirty_rate = (dirty_sum / denom).fillna(global_stats["dirty"])
            exec_mean = (exec_sum / denom).fillna(global_stats["exec"])
            frame[f"{prefix}_rows_log1p"] = np.log1p(rows).astype(np.float32).to_numpy()
            frame[f"{prefix}_clean_rate"] = (weight * clean_rate + (1.0 - weight) * global_stats["clean"]).astype(np.float32).to_numpy()
            frame[f"{prefix}_bad_mae_rate"] = (weight * bad_rate + (1.0 - weight) * global_stats["bad"]).astype(np.float32).to_numpy()
            frame[f"{prefix}_timeout_rate"] = (weight * timeout_rate + (1.0 - weight) * global_stats["timeout"]).astype(np.float32).to_numpy()
            frame[f"{prefix}_dirty_positive_rate"] = (weight * dirty_rate + (1.0 - weight) * global_stats["dirty"]).astype(np.float32).to_numpy()
            frame[f"{prefix}_exec_margin_mean"] = (weight * exec_mean + (1.0 - weight) * global_stats["exec"]).astype(np.float32).to_numpy()
            frame[f"{prefix}_edge"] = (
                frame[f"{prefix}_clean_rate"]
                - frame[f"{prefix}_bad_mae_rate"]
                - 0.50 * frame[f"{prefix}_timeout_rate"]
                + frame[f"{prefix}_exec_margin_mean"]
            ).astype(np.float32)
            return frame

        train = fill_from_stats(train, loo=True)
        valid = fill_from_stats(valid, loo=False)

    attach_for_band("base_rank_band", "rel_rankband")
    attach_for_band("base_margin_band", "rel_marginband")
    train = train.drop(columns=["_rel_side", "_rel_arch"], errors="ignore")
    valid = valid.drop(columns=["_rel_side", "_rel_arch"], errors="ignore")
    return train, valid


def _add_fold_support_drift_features(train: pd.DataFrame, valid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add row-level train-support features for AE/GMM/context/leaf buckets."""

    train = train.copy()
    valid = valid.copy()
    cols = [col for col in SUPPORT_DRIFT_COLUMNS if col in train.columns and col in valid.columns]
    if not cols:
        for col in SUPPORT_DRIFT_NUMERIC_FEATURES:
            train[col] = np.float32(0.0)
            valid[col] = np.float32(0.0)
        return train, valid

    min_count = 20.0
    min_freq = 0.01
    train_logs: list[np.ndarray] = []
    valid_logs: list[np.ndarray] = []
    train_freqs: list[np.ndarray] = []
    valid_freqs: list[np.ndarray] = []
    train_unseen: list[np.ndarray] = []
    valid_unseen: list[np.ndarray] = []
    train_rare: list[np.ndarray] = []
    valid_rare: list[np.ndarray] = []
    n_train = max(float(len(train)), 1.0)
    for col in cols:
        train_values = train[col].astype(str).fillna("missing")
        valid_values = valid[col].astype(str).fillna("missing")
        counts = train_values.value_counts(dropna=False)
        train_count = train_values.map(counts).fillna(0.0).astype(float).to_numpy()
        valid_count = valid_values.map(counts).fillna(0.0).astype(float).to_numpy()
        train_freq = train_count / n_train
        valid_freq = valid_count / n_train
        train_logs.append(np.log1p(train_count))
        valid_logs.append(np.log1p(valid_count))
        train_freqs.append(train_freq)
        valid_freqs.append(valid_freq)
        train_unseen.append((train_count <= 0.0).astype(float))
        valid_unseen.append((valid_count <= 0.0).astype(float))
        train_rare.append(((train_count < min_count) | (train_freq < min_freq)).astype(float))
        valid_rare.append(((valid_count < min_count) | (valid_freq < min_freq)).astype(float))

    def assign(frame: pd.DataFrame, logs: list[np.ndarray], freqs: list[np.ndarray], unseen: list[np.ndarray], rare: list[np.ndarray]) -> pd.DataFrame:
        log_mat = np.vstack(logs).T
        freq_mat = np.vstack(freqs).T
        unseen_mat = np.vstack(unseen).T
        rare_mat = np.vstack(rare).T
        frame["support_min_log_count"] = np.nanmin(log_mat, axis=1).astype(np.float32)
        frame["support_mean_log_count"] = np.nanmean(log_mat, axis=1).astype(np.float32)
        frame["support_min_frequency"] = np.nanmin(freq_mat, axis=1).astype(np.float32)
        frame["support_mean_frequency"] = np.nanmean(freq_mat, axis=1).astype(np.float32)
        frame["support_unseen_bucket_count"] = np.nansum(unseen_mat, axis=1).astype(np.float32)
        frame["support_unseen_bucket_share"] = (np.nanmean(unseen_mat, axis=1)).astype(np.float32)
        frame["support_rare_bucket_count"] = np.nansum(rare_mat, axis=1).astype(np.float32)
        frame["support_rare_bucket_share"] = (np.nanmean(rare_mat, axis=1)).astype(np.float32)
        return frame

    return (
        assign(train, train_logs, train_freqs, train_unseen, train_rare),
        assign(valid, valid_logs, valid_freqs, valid_unseen, valid_rare),
    )


def _timestamp_days(frame: pd.DataFrame) -> pd.Series:
    if "__ts__" not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=np.float64)
    ts = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    out = pd.Series(np.nan, index=frame.index, dtype=np.float64)
    valid = ts.notna()
    out.loc[valid] = ts.loc[valid].astype("int64").astype(np.float64) / (1e9 * 86400.0)
    return out


def _add_fold_hit_surprise_features(train: pd.DataFrame, valid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add decayed recent clean-hit surprise by side/base archetype.

    Validation rows are encoded from training rows only. Training rows are
    encoded from strictly earlier rows in the same training fold, excluding
    same-timestamp rows to avoid direct path-label leakage.
    """

    train = train.copy()
    valid = valid.copy()
    train_days = _timestamp_days(train)
    valid_days = _timestamp_days(valid)
    hit = _num(train.get("clean_exec_label"), index=train.index, default=0.0).fillna(0.0).clip(0.0, 1.0)
    train_key = train["side_name"].astype(str).str.lower() + "__" + _archetype_key(train).astype(str)
    valid_key = valid["side_name"].astype(str).str.lower() + "__" + _archetype_key(valid).astype(str)
    global_hit = float(hit.mean()) if len(hit) else 0.0
    shrinkage_k = 40.0

    source = pd.DataFrame(
        {
            "_key": train_key.astype(str),
            "_day": train_days,
            "_hit": hit.astype(np.float64),
        },
        index=train.index,
    ).replace([np.inf, -np.inf], np.nan)
    source = source[source["_day"].notna()].sort_values(["_key", "_day"], kind="mergesort")
    source_groups: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for key, group in source.groupby("_key", sort=False):
        days = group["_day"].to_numpy(dtype=np.float64)
        hits = group["_hit"].to_numpy(dtype=np.float64)
        source_groups[str(key)] = (days, hits, np.cumsum(hits))

    def assign(target: pd.DataFrame, target_days: pd.Series, target_key: pd.Series) -> pd.DataFrame:
        for hl in HIT_SURPRISE_HALFLIFE_DAYS:
            suffix = int(hl)
            recent = np.full(len(target), global_hit, dtype=np.float32)
            expected = np.full(len(target), global_hit, dtype=np.float32)
            surprise = np.zeros(len(target), dtype=np.float32)
            zscore = np.zeros(len(target), dtype=np.float32)
            support_log = np.zeros(len(target), dtype=np.float32)
            effective_n = np.zeros(len(target), dtype=np.float32)
            target_tmp = pd.DataFrame(
                {
                    "_pos": np.arange(len(target), dtype=np.int64),
                    "_key": target_key.astype(str).to_numpy(),
                    "_day": target_days.to_numpy(dtype=np.float64),
                },
                index=target.index,
            )
            for key, group in target_tmp.groupby("_key", sort=False):
                source_tuple = source_groups.get(str(key))
                if source_tuple is None:
                    continue
                src_days, src_hits, src_cumsum = source_tuple
                pos = group["_pos"].to_numpy(dtype=np.int64)
                day = group["_day"].to_numpy(dtype=np.float64)
                valid_day = np.isfinite(day)
                if not bool(np.any(valid_day)):
                    continue
                pos = pos[valid_day]
                day = day[valid_day]
                right = np.searchsorted(src_days, day, side="left").astype(np.int64)
                has_prior = right > 0
                if not bool(np.any(has_prior)):
                    continue
                pos = pos[has_prior]
                day = day[has_prior]
                right = right[has_prior]
                prior_hits = src_cumsum[right - 1]
                prior_count = right.astype(np.float64)
                prior_raw = prior_hits / np.maximum(prior_count, 1.0)
                prior_weight = prior_count / (prior_count + shrinkage_k)
                prior = prior_weight * prior_raw + (1.0 - prior_weight) * global_hit
                left = np.searchsorted(src_days, day - 4.0 * float(hl), side="left").astype(np.int64)
                alpha = math.log(2.0) / float(hl)
                exp1 = np.exp(alpha * src_days)
                exp2 = np.exp(2.0 * alpha * src_days)
                hit_exp1 = src_hits * exp1
                c_exp1 = np.concatenate([[0.0], np.cumsum(exp1)])
                c_hit_exp1 = np.concatenate([[0.0], np.cumsum(hit_exp1)])
                c_exp2 = np.concatenate([[0.0], np.cumsum(exp2)])
                win_exp1 = c_exp1[right] - c_exp1[left]
                win_hit_exp1 = c_hit_exp1[right] - c_hit_exp1[left]
                win_exp2 = c_exp2[right] - c_exp2[left]
                scale1 = np.exp(-alpha * day)
                scale2 = np.exp(-2.0 * alpha * day)
                weight_sum = scale1 * win_exp1
                weighted_hit = np.divide(
                    scale1 * win_hit_exp1,
                    np.maximum(weight_sum, 1e-12),
                    out=prior.copy(),
                    where=weight_sum > 1e-12,
                )
                weight_sq_sum = scale2 * win_exp2
                eff_n = np.divide(
                    weight_sum * weight_sum,
                    np.maximum(weight_sq_sum, 1e-12),
                    out=np.zeros_like(weight_sum),
                    where=weight_sum > 1e-12,
                )
                diff = weighted_hit - prior
                denom = np.sqrt(np.maximum(prior * (1.0 - prior), 1e-6) / np.maximum(eff_n, 1.0))
                recent[pos] = weighted_hit.astype(np.float32)
                expected[pos] = prior.astype(np.float32)
                surprise[pos] = diff.astype(np.float32)
                zscore[pos] = np.clip(diff / denom, -8.0, 8.0).astype(np.float32)
                support_log[pos] = np.log1p(weight_sum).astype(np.float32)
                effective_n[pos] = eff_n.astype(np.float32)
            target[f"base_arch_hit_recent_rate_hl{suffix}d"] = recent
            target[f"base_arch_hit_expected_rate_hl{suffix}d"] = expected
            target[f"base_arch_hit_surprise_hl{suffix}d"] = surprise
            target[f"base_arch_hit_surprise_z_hl{suffix}d"] = zscore
            target[f"base_arch_hit_support_log1p_hl{suffix}d"] = support_log
            target[f"base_arch_hit_effective_n_hl{suffix}d"] = effective_n
        return target

    return assign(train, train_days, train_key), assign(valid, valid_days, valid_key)


def _feature_columns(
    frame: pd.DataFrame,
    *,
    enable_base_prior_features: bool = False,
    enable_reliability_features: bool = False,
    enable_support_drift_features: bool = False,
    enable_hit_surprise_features: bool = False,
) -> tuple[list[str], list[str]]:
    excluded = set(NEVER_FEATURE_COLUMNS)
    excluded.update(OUTCOME_COLUMNS)
    excluded.update(
        {
            "positive_exec_margin",
            "clean_exec_label",
            "bad_path_label",
            "path_mfe_before_mae_label",
            "path_mae_before_mfe_label",
            "path_underwater_duration_target",
            "source_recipe_name",
            "target_soft",
            "target_hard",
            "first_pass_good",
            "first_pass_bad",
        }
    )
    if not enable_base_prior_features:
        excluded.update(BASE_PRIOR_NUMERIC_FEATURES)
        excluded.update(BASE_PRIOR_CATEGORICAL_FEATURES)
    if not enable_reliability_features:
        excluded.update(RELIABILITY_NUMERIC_FEATURES)
    if not enable_support_drift_features:
        excluded.update(SUPPORT_DRIFT_NUMERIC_FEATURES)
    if not enable_hit_surprise_features:
        excluded.update(HIT_SURPRISE_NUMERIC_FEATURES)
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    for col in frame.columns:
        if col in excluded or col.endswith("__ledger"):
            continue
        if col.startswith("selected_top"):
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            numeric_cols.append(col)
        elif frame[col].dtype == bool:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    # Side is both categorical and already encoded by descriptors in places; keep it categorical.
    return sorted(numeric_cols), sorted(categorical_cols)


def _make_xy(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_parts: list[pd.DataFrame] = []
    valid_parts: list[pd.DataFrame] = []
    if numeric_cols:
        train_num = train.loc[:, numeric_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid_num = valid.loc[:, numeric_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        med = train_num.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        train_parts.append(train_num.fillna(med).fillna(0.0).astype(np.float32))
        valid_parts.append(valid_num.fillna(med).fillna(0.0).astype(np.float32))
    if categorical_cols:
        train_cat = pd.get_dummies(train.loc[:, categorical_cols].astype(str).fillna("missing"), dummy_na=False)
        valid_cat = pd.get_dummies(valid.loc[:, categorical_cols].astype(str).fillna("missing"), dummy_na=False)
        valid_cat = valid_cat.reindex(columns=train_cat.columns, fill_value=0)
        train_parts.append(train_cat.astype(np.float32))
        valid_parts.append(valid_cat.astype(np.float32))
    if not train_parts:
        raise ValueError("No meta feature columns available.")
    x_train = pd.concat(train_parts, axis=1)
    x_valid = pd.concat(valid_parts, axis=1).reindex(columns=x_train.columns, fill_value=0.0)
    return x_train, x_valid, list(x_train.columns)


def _classification_weights(target: pd.Series, train: pd.DataFrame) -> np.ndarray:
    y = _num(target).fillna(0.0).astype(int)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    weights = np.ones(len(y), dtype=np.float32)
    if pos > 0 and neg > 0:
        weights[y.to_numpy(dtype=bool)] = neg / max(pos, 1)
    weights *= 1.0 + 0.75 * _num(train.get("dirty_positive"), index=train.index, default=0.0).fillna(0.0).to_numpy(dtype=np.float32)
    weights *= 1.0 + 0.50 * _num(train.get("full_path_bad_mae_1r"), index=train.index, default=0.0).fillna(0.0).to_numpy(dtype=np.float32)
    weights = np.clip(weights, 0.25, 8.0)
    return weights / max(float(weights.mean()), 1e-12)


def _regression_weights(target: pd.Series, train: pd.DataFrame) -> np.ndarray:
    y = _num(target).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    weights = np.ones(len(y), dtype=np.float32)
    weights *= np.where(y.gt(0.0), 2.0, 1.0)
    if len(y) and y.nunique(dropna=True) > 1:
        weights *= np.where(y.ge(float(y.quantile(0.80))), 1.5, 1.0)
    weights *= 1.0 + 0.60 * _num(train.get("dirty_positive"), index=train.index, default=0.0).fillna(0.0).to_numpy(dtype=np.float32)
    weights = np.clip(weights, 0.25, 6.0)
    return weights / max(float(weights.mean()), 1e-12)


def _lgbm_params(defaults: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    params = dict(defaults)
    if override:
        params.update({str(k): v for k, v in override.items()})
    for key in ("n_estimators", "num_leaves", "max_depth", "min_child_samples"):
        params[key] = int(float(params[key]))
    for key in ("learning_rate", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"):
        params[key] = float(params[key])
    return params


def _time_spread_cap_rows(n_rows: int, max_rows: int) -> np.ndarray:
    if int(max_rows) <= 0 or n_rows <= int(max_rows):
        return np.arange(n_rows, dtype=np.int64)
    n = int(n_rows)
    k = int(max_rows)
    base = k // 3
    rem = k - base * 3
    sizes = [base + (1 if i < rem else 0) for i in range(3)]
    windows = [(0, n // 3), (n // 3, (2 * n) // 3), ((2 * n) // 3, n)]
    parts: list[np.ndarray] = []
    for size, (start, end) in zip(sizes, windows):
        size = min(int(size), max(int(end - start), 0))
        if size <= 0:
            continue
        if size >= end - start:
            parts.append(np.arange(start, end, dtype=np.int64))
        else:
            parts.append(np.linspace(start, end - 1, size, dtype=np.int64))
    if not parts:
        return np.arange(0, min(n, k), dtype=np.int64)
    return np.unique(np.concatenate(parts).astype(np.int64))


def _feature_selection_target(train: pd.DataFrame, target_name: str) -> pd.Series:
    if target_name == "exec_margin":
        return _num(train.get("exec_margin"), index=train.index, default=0.0).fillna(0.0)
    if target_name == "ev_frontier":
        ev = _num(train.get("ev_after_1pct"), index=train.index)
        if ev.isna().all():
            ev = _num(train.get("exec_margin"), index=train.index, default=0.0)
        clean = _num(train.get("clean_exec_label"), index=train.index, default=0.0).fillna(0.0).clip(0.0, 1.0)
        dirty = _num(train.get("dirty_positive"), index=train.index, default=0.0).fillna(0.0).clip(0.0, 1.0)
        bad = _num(train.get("bad_path_label"), index=train.index, default=0.0).fillna(0.0).clip(0.0, 1.0)
        first_bad = _num(train.get("first_touch_bad_mae_1r"), index=train.index, default=0.0).fillna(0.0).clip(0.0, 1.0)
        full_bad = _num(train.get("full_path_bad_mae_1r"), index=train.index, default=0.0).fillna(0.0).clip(0.0, 1.0)
        bad_or_stop = pd.concat([bad, first_bad, full_bad], axis=1).max(axis=1).fillna(0.0)
        ev_pct = ev.fillna(0.0).clip(-0.05, 0.05) * 100.0
        return (
            0.45 * ev_pct
            + 0.20 * ((clean - dirty) + (clean - bad_or_stop))
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if target_name == "clean_minus_bad":
        clean = _num(train.get("clean_exec_label"), index=train.index, default=0.0).fillna(0.0)
        bad = _num(train.get("bad_path_label"), index=train.index, default=0.0).fillna(0.0)
        timeout = _num(train.get("timeout"), index=train.index, default=0.0).fillna(0.0)
        margin = _num(train.get("exec_margin"), index=train.index, default=0.0).fillna(0.0).clip(-0.03, 0.03)
        return clean + 0.50 * margin - 0.70 * bad - 0.30 * timeout
    if target_name == "clean_exec":
        return _num(train.get("clean_exec_label"), index=train.index, default=0.0).fillna(0.0)
    raise ValueError(f"Unknown feature-selection target: {target_name}")


def _base_soft_label_target(frame: pd.DataFrame) -> tuple[pd.Series, str]:
    for col in BASE_SOFT_LABEL_COLUMNS:
        if col in frame.columns:
            target = _num(frame.get(col), index=frame.index).replace([np.inf, -np.inf], np.nan).clip(0.0, 1.0)
            if int(target.notna().sum()) > 0:
                return target, col
    raise ValueError(
        "Single-head meta mode requires the base economic soft label. "
        f"Expected one of: {', '.join(BASE_SOFT_LABEL_COLUMNS)}"
    )


def _feature_selection_label_context(train: pd.DataFrame) -> dict[str, np.ndarray]:
    n = int(len(train))
    side_name = train.get("side_name", pd.Series("unknown", index=train.index)).astype(str).str.lower()
    side_num = np.where(side_name.eq("short"), -1.0, 1.0).astype(np.float32)
    family = (
        train.get("archetype_label_family")
        if "archetype_label_family" in train.columns
        else train.get("__archetype_label_family__", train.get("source_semantic_family", "unknown"))
    )
    family_ser = pd.Series(family, index=train.index).astype(str).fillna("unknown")
    policy = (
        train.get("archetype_policy_key")
        if "archetype_policy_key" in train.columns
        else train.get("__archetype_policy_key__", train.get("source_tag", "unknown"))
    )
    policy_ser = pd.Series(policy, index=train.index).astype(str).fillna("unknown")
    source_family = pd.Series(train.get("source_semantic_family", family_ser), index=train.index).astype(str).fillna("unknown")
    feature_arch = (side_name + "__" + family_ser + "__" + policy_ser).to_numpy(dtype=object)
    behavior_arch = (
        side_name
        + "__"
        + source_family
        + "__bad"
        + _num(train.get("first_touch_bad_mae_1r"), index=train.index, default=0.0).fillna(0.0).gt(0.5).astype(int).astype(str)
        + "__to"
        + _num(train.get("timeout"), index=train.index, default=0.0).fillna(0.0).gt(0.5).astype(int).astype(str)
    ).to_numpy(dtype=object)
    return {
        "feature_selection_archetype": feature_arch,
        "archetype": feature_arch,
        "label_archetype": family_ser.to_numpy(dtype=object),
        "source_archetype": source_family.to_numpy(dtype=object),
        "geometry_archetype": policy_ser.to_numpy(dtype=object),
        "side_name": side_name.to_numpy(dtype=object),
        "side": side_num,
        "clean_exec": _num(train.get("clean_exec_label"), index=train.index, default=0.0).fillna(0.0).to_numpy(dtype=np.float32),
        "bad_mae": _num(train.get("first_touch_bad_mae_1r"), index=train.index, default=0.0).fillna(0.0).to_numpy(dtype=np.float32),
        "timeout": _num(train.get("timeout"), index=train.index, default=0.0).fillna(0.0).to_numpy(dtype=np.float32),
        "dirty_positive": _num(train.get("dirty_positive"), index=train.index, default=0.0).fillna(0.0).to_numpy(dtype=np.float32),
        "utility": _num(train.get("exec_margin"), index=train.index, default=0.0).fillna(0.0).to_numpy(dtype=np.float32),
        "behavior_archetype": behavior_arch,
    }


def _select_features_by_lgbm_pipeline(
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    train: pd.DataFrame,
    *,
    target_name: str,
    top_n: int,
    fold: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame]:
    y = _feature_selection_target(train, target_name).replace([np.inf, -np.inf], np.nan)
    valid_y = y.notna().to_numpy(dtype=bool)
    if int(valid_y.sum()) < 500 or int(y.loc[valid_y].nunique(dropna=True)) < 3:
        raise ValueError("Canonical lgbm_pipeline feature selection requires at least 500 rows and target variation.")
    selector_x = x_train.loc[valid_y].reset_index(drop=True).astype(np.float32, copy=False)
    selector_train = train.loc[valid_y].reset_index(drop=True)
    selector_y = y.loc[valid_y].reset_index(drop=True).astype(np.float32).to_numpy(dtype=np.float32)
    base_soft, _base_soft_col = _base_soft_label_target(selector_train)
    sample_weight = _base_style_weights_for_soft_label(selector_train, base_soft)
    timestamps = pd.to_datetime(selector_train.get("__ts__"), utc=True, errors="coerce")
    assets = selector_train.get("__symbol__", pd.Series("unknown", index=selector_train.index)).astype(str).to_numpy()
    returns = _num(selector_train.get("ev_after_1pct"), index=selector_train.index)
    if returns.isna().all():
        returns = _num(selector_train.get("exec_margin"), index=selector_train.index, default=0.0)
    cfg = {
        "mda_config": {
            "enabled": True,
            "objective": "topk_opportunity_precision",
            "topk_fracs": [0.20, 0.10, 0.15],
            "topk_frac_weights": [0.20, 0.15, 0.20],
            "positive_label": 1,
            "use_sample_weight": True,
        }
    }
    candidate = train_lgbm_stability_candidate(
        selector_x,
        selector_y,
        sample_weight=sample_weight,
        random_state=int(seed),
        mode="regressor",
        timestamps=timestamps,
        assets=assets,
        returns=returns.fillna(0.0).to_numpy(dtype=np.float32),
        hpo_objective_mode="train_meta",
        cfg=cfg,
        label_context=_feature_selection_label_context(selector_train),
        reference_artifact_dir=Path("/private/tmp") / f"s52_meta_lgbm_pipeline_fs_{abs(hash(str(fold))) % 10_000_000}",
    )
    if not candidate:
        raise RuntimeError("Canonical lgbm_pipeline feature selection returned no candidate.")
    selected = [str(c) for c in candidate.get("selected_feature_names", []) if str(c) in x_train.columns]
    if int(top_n) > 0:
        selected = selected[: int(top_n)]
    if not selected:
        raise RuntimeError("Canonical lgbm_pipeline feature selection returned no selected feature names.")
    stats = candidate.get("feature_stats")
    if isinstance(stats, pd.DataFrame) and "feature" in stats.columns:
        rows = stats.copy()
    else:
        rows = pd.DataFrame({"feature": list(x_train.columns)})
    if "feature" not in rows.columns:
        rows["feature"] = list(x_train.columns[: len(rows)])
    selected_set = set(selected)
    rows["feature"] = rows["feature"].astype(str)
    rows["selected"] = rows["feature"].isin(selected_set)
    if "rank" not in rows.columns:
        rows["_rank_score"] = (
            pd.to_numeric(rows.get("feature_score"), errors="coerce")
            .fillna(pd.to_numeric(rows.get("mda_mean"), errors="coerce"))
            .fillna(pd.to_numeric(rows.get("permutation_score"), errors="coerce"))
            .fillna(0.0)
        )
        rows["rank"] = rows["_rank_score"].rank(method="first", ascending=False).astype(int)
        rows = rows.drop(columns=["_rank_score"], errors="ignore")
    if "score" not in rows.columns:
        score_source = None
        for col in ("feature_score", "mda_mean", "median_perm", "permutation_score", "gain_rank_score"):
            if col in rows.columns:
                score_source = col
                break
        rows["score"] = pd.to_numeric(rows[score_source], errors="coerce") if score_source else np.nan
    rows["fold"] = str(fold)
    rows["feature_selection_target"] = str(target_name)
    rows["feature_selection_method"] = "lgbm_pipeline_staged"
    rows["feature_selection_status"] = "ok"
    rows["feature_selection_requested_top_n"] = int(top_n)
    rows["feature_selection_auto_selected_count"] = int(len(selected))
    rows["feature_selection_auto_mode"] = "lgbm_pipeline_iterative_prune" if int(top_n) <= 0 else "explicit_top_n_cap"
    rows["lgbm_pipeline_selected_count"] = int(len(selected))
    rows["lgbm_pipeline_input_feature_count"] = int(x_train.shape[1])
    rows["lgbm_pipeline_selector_rows"] = int(len(selector_x))
    return x_train.loc[:, selected], x_valid.reindex(columns=selected, fill_value=0.0), selected, rows


def _fit_classifier(
    x: pd.DataFrame,
    y: pd.Series,
    train: pd.DataFrame,
    seed: int,
    *,
    lgbm_params: dict[str, Any] | None = None,
) -> Any:
    target = _num(y).fillna(0.0).astype(int)
    if int(target.nunique(dropna=True)) < 2:
        return None
    weights = _classification_weights(target, train)
    if _LIGHTGBM_AVAILABLE and LGBMClassifier is not None:
        params = _lgbm_params(DEFAULT_LGBM_CLASSIFIER_PARAMS, lgbm_params)
        model = LGBMClassifier(
            objective="binary",
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            num_leaves=params["num_leaves"],
            max_depth=params["max_depth"],
            min_child_samples=params["min_child_samples"],
            subsample=params["subsample"],
            subsample_freq=1,
            colsample_bytree=params["colsample_bytree"],
            reg_alpha=params["reg_alpha"],
            reg_lambda=params["reg_lambda"],
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


def _base_style_metrics_for_weights(train: pd.DataFrame) -> pd.DataFrame:
    index = train.index
    return pd.DataFrame(
        {
            "u_policy_net": _num(train.get("u_policy_net", train.get("exec_margin")), index=index, default=0.0).fillna(0.0),
            "mae_norm": _num(train.get("mae_norm", train.get("first_touch_full_path_mae_norm")), index=index, default=0.0).fillna(0.0),
            "mfe_norm": _num(train.get("mfe_norm", train.get("first_touch_full_path_mfe_norm")), index=index, default=0.0).fillna(0.0),
            "bars_to_mfe": _num(train.get("first_touch_bar"), index=index, default=24.0).fillna(24.0),
            "barrier": _num(train.get("__archetype_policy_max_barrier__"), index=index, default=0.0).fillna(0.0),
            "is_timeout": _num(train.get("timeout"), index=index, default=0.0).fillna(0.0),
            "side": np.where(train.get("side_name", pd.Series("long", index=index)).astype(str).str.lower().eq("short"), -1.0, 1.0),
        },
        index=index,
    )


def _base_style_weights_for_soft_label(
    train: pd.DataFrame,
    target: pd.Series,
    *,
    weight_arm: str = "W7_timestamp_balanced",
) -> pd.Series:
    frame = train.copy(deep=False)
    if "__ts__" in frame.columns:
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    target_frame = pd.DataFrame({"target_soft": _num(target, index=train.index).clip(0.0, 1.0)}, index=train.index)
    try:
        weights = _base_weight_series(
            frame=frame,
            metrics=_base_style_metrics_for_weights(frame),
            target=target_frame,
            arm=str(weight_arm),
        )
    except Exception:
        return pd.Series(1.0, index=train.index, dtype=np.float32)
    return _num(weights, index=train.index, default=1.0).replace([np.inf, -np.inf], np.nan).fillna(1.0).astype(np.float32)


def _fit_base_soft_label_model(
    x: pd.DataFrame,
    y: pd.Series,
    train: pd.DataFrame,
    seed: int,
    *,
    lgbm_params: dict[str, Any] | None = None,
) -> Any:
    target = _num(y).replace([np.inf, -np.inf], np.nan).clip(0.0, 1.0)
    valid = target.notna()
    if int(valid.sum()) < 50 or float(target.loc[valid].std()) <= 1e-12:
        return None
    weights = _base_style_weights_for_soft_label(train.loc[valid], target.loc[valid])
    if _LIGHTGBM_AVAILABLE and LGBMRegressor is not None:
        params = _lgbm_params(DEFAULT_LGBM_CLASSIFIER_PARAMS, lgbm_params)
        model = LGBMRegressor(
            objective="regression",
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            num_leaves=params["num_leaves"],
            max_depth=params["max_depth"],
            min_child_samples=params["min_child_samples"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            reg_alpha=params["reg_alpha"],
            reg_lambda=params["reg_lambda"],
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


def _fit_regressor(
    x: pd.DataFrame,
    y: pd.Series,
    train: pd.DataFrame,
    seed: int,
    *,
    lgbm_params: dict[str, Any] | None = None,
) -> Any:
    target = _num(y).replace([np.inf, -np.inf], np.nan)
    valid = target.notna()
    if int(valid.sum()) < 50 or float(target.loc[valid].std()) <= 1e-12:
        return None
    weights = _regression_weights(target.loc[valid], train.loc[valid])
    if _LIGHTGBM_AVAILABLE and LGBMRegressor is not None:
        params = _lgbm_params(DEFAULT_LGBM_REGRESSOR_PARAMS, lgbm_params)
        model = LGBMRegressor(
            objective="regression",
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            num_leaves=params["num_leaves"],
            max_depth=params["max_depth"],
            min_child_samples=params["min_child_samples"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            reg_alpha=params["reg_alpha"],
            reg_lambda=params["reg_lambda"],
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


def _fit_side_classifier(
    x: pd.DataFrame,
    y: pd.Series,
    train: pd.DataFrame,
    *,
    side: str,
    seed: int,
    min_rows: int = 80,
    lgbm_params: dict[str, Any] | None = None,
) -> Any:
    side_mask = train["side_name"].astype(str).str.lower().eq(str(side).lower()) if "side_name" in train.columns else pd.Series(False, index=train.index)
    target = _num(y).replace([np.inf, -np.inf], np.nan)
    valid = side_mask & target.notna()
    if int(valid.sum()) < int(min_rows) or int(target.loc[valid].nunique(dropna=True)) < 2:
        return None
    return _fit_classifier(x.loc[valid], target.loc[valid], train.loc[valid], seed, lgbm_params=lgbm_params)


def _predict(model: Any, x: pd.DataFrame, *, classifier: bool) -> pd.Series:
    if model is None:
        return pd.Series(np.nan, index=x.index, dtype=np.float32)
    if classifier and hasattr(model, "predict_proba"):
        pred = model.predict_proba(x)
        if np.asarray(pred).ndim == 2 and pred.shape[1] >= 2:
            return pd.Series(pred[:, 1], index=x.index, dtype=np.float32)
    return pd.Series(np.asarray(model.predict(x), dtype=np.float32), index=x.index)


def _top_selected(frame: pd.DataFrame, score_col: str, keep_frac: float) -> pd.DataFrame:
    valid = frame[_num(frame[score_col]).notna()].copy()
    if valid.empty:
        return valid
    n = max(1, int(math.ceil(len(valid) * float(keep_frac))))
    return valid.sort_values(score_col, ascending=False, kind="mergesort").head(n)


def _budget_selected(
    frame: pd.DataFrame,
    score_col: str,
    *,
    budget_frac: float,
    eligible: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    valid = frame[_num(frame[score_col]).notna()].copy()
    if valid.empty:
        return valid
    target_rows = max(1, int(math.ceil(len(frame) * float(budget_frac))))
    if eligible is not None:
        eligible_s = pd.Series(eligible, index=frame.index).reindex(valid.index).fillna(False).astype(bool)
        valid = valid[eligible_s]
    if valid.empty:
        return valid
    return valid.sort_values(score_col, ascending=False, kind="mergesort").head(target_rows)


def _metric_row(frame: pd.DataFrame, score_col: str, keep_frac: float) -> dict[str, Any]:
    selected = _top_selected(frame, score_col, keep_frac)
    tag = f"keep{int(round(keep_frac * 100)):03d}"
    oracle = frame.sort_values("exec_margin", ascending=False, kind="mergesort").head(len(selected))
    overlap = len(set(selected.index.tolist()) & set(oracle.index.tolist())) if len(selected) else 0
    selected_ts = pd.to_datetime(selected.get("__ts__"), utc=True, errors="coerce") if len(selected) else pd.Series(dtype="datetime64[ns, UTC]")
    selected_ev = _num(selected.get("ev_after_1pct"), index=selected.index) if len(selected) else pd.Series(dtype=float)
    if len(selected) and selected_ts.notna().any():
        week_key = selected_ts.dt.tz_localize(None).dt.to_period("W").astype(str)
        worst_week_ev = float(selected_ev.groupby(week_key).mean().min())
    else:
        worst_week_ev = float("nan")
    out = {
        f"{tag}_rows": int(len(selected)),
        f"{tag}_exec_margin": _mean(selected.get("exec_margin")),
        f"{tag}_ev_after_1pct": _mean(selected.get("ev_after_1pct")),
        f"{tag}_worst_week_ev_after_1pct": worst_week_ev,
        f"{tag}_ret_net": _mean(selected.get("ret_net")),
        f"{tag}_u_policy_net": _mean(selected.get("u_policy_net")),
        f"{tag}_gross": _mean(selected.get("first_touch_gross")),
        f"{tag}_positive_margin_rate": _rate(_num(selected.get("exec_margin")).gt(0.0)),
        f"{tag}_clean_exec_precision": _rate(selected.get("clean_exec")),
        f"{tag}_dirty_positive_rate": _rate(selected.get("dirty_positive")),
        f"{tag}_first_touch_bad_mae": _rate(selected.get("first_touch_bad_mae_1r")),
        f"{tag}_full_path_bad_mae": _rate(selected.get("full_path_bad_mae_1r")),
        f"{tag}_timeout": _rate(selected.get("timeout")),
        f"{tag}_mfe_before_mae": _rate(selected.get("mfe_before_mae_1r")),
        f"{tag}_mae_before_mfe": _rate(selected.get("mae_before_mfe_1r")),
        f"{tag}_underwater_bars": _mean(selected.get("underwater_bars_before_mfe_1r")),
        f"{tag}_oracle_recall_exec_margin": float(overlap / max(len(oracle), 1)) if len(selected) else float("nan"),
        f"{tag}_long_share": float(selected["side_name"].astype(str).eq("long").mean()) if len(selected) else float("nan"),
        f"{tag}_short_share": float(selected["side_name"].astype(str).eq("short").mean()) if len(selected) else float("nan"),
    }
    return out


def _policy_metric_row(
    frame: pd.DataFrame,
    score_col: str,
    *,
    selector: str,
    policy_id: str,
    test_month: str,
    budget_frac: float,
    eligible: pd.Series | np.ndarray | None,
) -> dict[str, Any]:
    selected = _budget_selected(frame, score_col, budget_frac=budget_frac, eligible=eligible)
    target_rows = max(1, int(math.ceil(len(frame) * float(budget_frac)))) if len(frame) else 0
    oracle = frame.sort_values("exec_margin", ascending=False, kind="mergesort").head(len(selected))
    overlap = len(set(selected.index.tolist()) & set(oracle.index.tolist())) if len(selected) else 0
    selected_ts = pd.to_datetime(selected.get("__ts__"), utc=True, errors="coerce") if len(selected) else pd.Series(dtype="datetime64[ns, UTC]")
    selected_ev = _num(selected.get("ev_after_1pct"), index=selected.index) if len(selected) else pd.Series(dtype=float)
    if len(selected) and selected_ts.notna().any():
        week_key = selected_ts.dt.tz_localize(None).dt.to_period("W").astype(str)
        worst_week_ev = float(selected_ev.groupby(week_key).mean().min())
    else:
        worst_week_ev = float("nan")
    return {
        "selector": str(selector),
        "policy_id": str(policy_id),
        "test_month": str(test_month),
        "budget_frac": float(budget_frac),
        "candidate_rows": int(len(frame)),
        "target_rows": int(target_rows),
        "selected_rows": int(len(selected)),
        "fill_rate": float(len(selected) / max(target_rows, 1)) if target_rows else float("nan"),
        "no_trade_rate": float(1.0 - len(selected) / max(target_rows, 1)) if target_rows else float("nan"),
        "exec_margin": _mean(selected.get("exec_margin")),
        "ev_after_1pct": _mean(selected.get("ev_after_1pct")),
        "worst_week_ev_after_1pct": worst_week_ev,
        "positive_margin_rate": _rate(_num(selected.get("exec_margin")).gt(0.0)),
        "clean_exec_precision": _rate(selected.get("clean_exec")),
        "dirty_positive_rate": _rate(selected.get("dirty_positive")),
        "first_touch_bad_mae": _rate(selected.get("first_touch_bad_mae_1r")),
        "full_path_bad_mae": _rate(selected.get("full_path_bad_mae_1r")),
        "timeout": _rate(selected.get("timeout")),
        "mfe_before_mae": _rate(selected.get("mfe_before_mae_1r")),
        "mae_before_mfe": _rate(selected.get("mae_before_mfe_1r")),
        "oracle_recall_exec_margin": float(overlap / max(len(oracle), 1)) if len(selected) else float("nan"),
        "long_share": float(selected["side_name"].astype(str).eq("long").mean()) if len(selected) else float("nan"),
        "short_share": float(selected["side_name"].astype(str).eq("short").mean()) if len(selected) else float("nan"),
    }


def _selector_metrics(frame: pd.DataFrame, score_col: str, selector: str, test_month: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "selector": selector,
        "test_month": str(test_month),
        "candidate_rows": int(len(frame)),
        "scorable_rows": int(_num(frame[score_col]).notna().sum()),
        "base_candidate_exec_margin": _mean(frame.get("exec_margin")),
        "base_candidate_full_path_bad_mae": _rate(frame.get("full_path_bad_mae_1r")),
        "base_candidate_timeout": _rate(frame.get("timeout")),
        "spearman_exec_margin": _safe_spearman(frame.get("exec_margin", pd.Series(dtype=float)), frame[score_col]),
        "spearman_ev_after_1pct": _safe_spearman(frame.get("ev_after_1pct", pd.Series(dtype=float)), frame[score_col]),
        "spearman_underwater_bars": _safe_spearman(
            -_num(frame.get("underwater_bars_before_mfe_1r", pd.Series(dtype=float))).fillna(0.0),
            frame[score_col],
        ),
        "auc_clean_exec": _safe_auc(frame.get("clean_exec", pd.Series(dtype=float)), frame[score_col]),
        "ap_clean_exec": _safe_ap(frame.get("clean_exec", pd.Series(dtype=float)), frame[score_col]),
        "auc_mfe_before_mae": _safe_auc(frame.get("mfe_before_mae_1r", pd.Series(dtype=float)), frame[score_col]),
        "auc_avoids_mae_before_mfe": _safe_auc(
            1.0 - _num(frame.get("mae_before_mfe_1r", pd.Series(dtype=float))).fillna(0.0),
            frame[score_col],
        ),
        "auc_positive_margin": _safe_auc(frame.get("positive_exec_margin", pd.Series(dtype=float)), frame[score_col]),
        "ap_positive_margin": _safe_ap(frame.get("positive_exec_margin", pd.Series(dtype=float)), frame[score_col]),
    }
    for frac in TOP_KEEP_FRACTIONS:
        row.update(_metric_row(frame, score_col, frac))
    return row


def _breakdown_rows(frame: pd.DataFrame, score_col: str, selector: str, test_month: str, keep_frac: float) -> list[dict[str, Any]]:
    selected = _top_selected(frame, score_col, keep_frac)
    rows: list[dict[str, Any]] = []
    if selected.empty:
        return rows
    group_cols = [
        ["side_name"],
        ["source_semantic_family"],
        ["side_name", "source_semantic_family"],
        ["side_name", "source_semantic_family", "aegmm_cluster"],
        ["side_name", "long_source_regime_split"],
        ["side_name", "regime_first_touch_bad_mae_score_bin"],
        ["side_name", "aegmm_expected_distance_bin"],
        ["side_name", "regime_lgbm_leaf_bad_mae_k4"],
    ]
    for cols in group_cols:
        if not all(col in selected.columns for col in cols):
            continue
        for key, group in selected.groupby(cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            record: dict[str, Any] = {
                "selector": selector,
                "test_month": str(test_month),
                "keep_frac": float(keep_frac),
                "grouping": "+".join(cols),
                "rows": int(len(group)),
                "exec_margin": _mean(group.get("exec_margin")),
                "clean_exec_precision": _rate(group.get("clean_exec")),
                "full_path_bad_mae": _rate(group.get("full_path_bad_mae_1r")),
                "timeout": _rate(group.get("timeout")),
                "mfe_before_mae": _rate(group.get("mfe_before_mae_1r")),
            }
            for col, value in zip(cols, key, strict=False):
                record[col] = value
            rows.append(record)
    return rows


def _base_conditioned_diagnostic_rows(
    frame: pd.DataFrame,
    *,
    test_month: str,
    selector: str,
    score_col: str,
    keep_frac: float = 0.30,
) -> list[dict[str, Any]]:
    selected = _top_selected(frame, score_col, keep_frac)
    rows: list[dict[str, Any]] = []
    if frame.empty:
        return rows
    candidate = frame.copy()
    candidate["selection_scope"] = "candidate"
    selected = selected.copy()
    selected["selection_scope"] = f"selected_keep{int(round(keep_frac * 100)):02d}"
    combined = pd.concat([candidate, selected], axis=0, ignore_index=False)
    groupings = [
        ["side_name", "source_tag", "base_score_decile"],
        ["side_name", "source_tag", "base_rank_band"],
        ["side_name", "source_tag", "base_margin_band"],
        ["side_name", "source_semantic_family", "base_score_decile"],
        ["side_name", "source_semantic_family", "base_margin_band"],
    ]
    for cols in groupings:
        if not all(col in combined.columns for col in cols):
            continue
        for key, group in combined.groupby(["selection_scope"] + cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            record: dict[str, Any] = {
                "test_month": str(test_month),
                "selector": str(selector),
                "score_col": str(score_col),
                "grouping": "+".join(cols),
                "selection_scope": str(key[0]),
                "rows": int(len(group)),
                "mean_score": _mean(group.get(score_col)),
                "mean_base_score": _mean(group.get("score")),
                "mean_base_margin_to_cutoff": _mean(group.get("base_margin_to_cutoff")),
                "clean_exec_precision": _rate(group.get("clean_exec")),
                "full_path_bad_mae": _rate(group.get("full_path_bad_mae_1r")),
                "timeout": _rate(group.get("timeout")),
                "mfe_before_mae": _rate(group.get("mfe_before_mae_1r")),
                "mae_before_mfe": _rate(group.get("mae_before_mfe_1r")),
                "underwater_bars": _mean(group.get("underwater_bars_before_mfe_1r")),
                "exec_margin": _mean(group.get("exec_margin")),
            }
            for col, value in zip(cols, key[1:], strict=False):
                record[col] = value
            rows.append(record)
    return rows


def _feature_importance(model: Any, feature_names: list[str], label: str, test_month: str) -> pd.DataFrame:
    if model is None or not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["test_month", "model", "feature", "importance"])
    values = np.asarray(model.feature_importances_, dtype=np.float64)
    if len(values) != len(feature_names):
        return pd.DataFrame(columns=["test_month", "model", "feature", "importance"])
    out = pd.DataFrame({"feature": feature_names, "importance": values})
    out = out[out["importance"] > 0].sort_values("importance", ascending=False).head(80)
    out.insert(0, "model", label)
    out.insert(0, "test_month", str(test_month))
    return out


def _summarize(folds: pd.DataFrame) -> pd.DataFrame:
    if folds.empty:
        return folds
    agg = (
        folds.groupby("selector", dropna=False)
        .agg(
            folds=("test_month", "nunique"),
            mean_candidate_rows=("candidate_rows", "mean"),
            mean_keep100_exec_margin=("keep100_exec_margin", "mean"),
            mean_keep100_full_path_bad_mae=("keep100_full_path_bad_mae", "mean"),
            mean_keep100_timeout=("keep100_timeout", "mean"),
            mean_keep050_exec_margin=("keep050_exec_margin", "mean"),
            mean_keep050_full_path_bad_mae=("keep050_full_path_bad_mae", "mean"),
            mean_keep030_exec_margin=("keep030_exec_margin", "mean"),
            mean_keep030_clean_exec_precision=("keep030_clean_exec_precision", "mean"),
            mean_keep030_full_path_bad_mae=("keep030_full_path_bad_mae", "mean"),
            mean_keep030_timeout=("keep030_timeout", "mean"),
            mean_keep030_oracle_recall=("keep030_oracle_recall_exec_margin", "mean"),
            mean_keep020_exec_margin=("keep020_exec_margin", "mean"),
            mean_keep020_ev_after_1pct=("keep020_ev_after_1pct", "mean"),
            mean_keep020_ret_net=("keep020_ret_net", "mean"),
            mean_keep020_u_policy_net=("keep020_u_policy_net", "mean"),
            mean_keep020_clean_exec_precision=("keep020_clean_exec_precision", "mean"),
            mean_keep020_dirty_positive_rate=("keep020_dirty_positive_rate", "mean"),
            mean_keep020_full_path_bad_mae=("keep020_full_path_bad_mae", "mean"),
            mean_keep020_first_touch_bad_mae=("keep020_first_touch_bad_mae", "mean"),
            mean_keep020_timeout=("keep020_timeout", "mean"),
            mean_keep020_oracle_recall=("keep020_oracle_recall_exec_margin", "mean"),
            mean_keep015_exec_margin=("keep015_exec_margin", "mean"),
            mean_keep015_ev_after_1pct=("keep015_ev_after_1pct", "mean"),
            mean_keep015_clean_exec_precision=("keep015_clean_exec_precision", "mean"),
            mean_keep015_dirty_positive_rate=("keep015_dirty_positive_rate", "mean"),
            mean_keep015_full_path_bad_mae=("keep015_full_path_bad_mae", "mean"),
            mean_keep015_first_touch_bad_mae=("keep015_first_touch_bad_mae", "mean"),
            mean_keep015_timeout=("keep015_timeout", "mean"),
            mean_keep010_exec_margin=("keep010_exec_margin", "mean"),
            mean_keep010_ev_after_1pct=("keep010_ev_after_1pct", "mean"),
            mean_keep010_ret_net=("keep010_ret_net", "mean"),
            mean_keep010_u_policy_net=("keep010_u_policy_net", "mean"),
            worst_keep010_exec_margin=("keep010_exec_margin", "min"),
            worst_keep010_ev_after_1pct=("keep010_ev_after_1pct", "min"),
            worst_week_keep010_ev_after_1pct=("keep010_worst_week_ev_after_1pct", "min"),
            mean_keep010_clean_exec_precision=("keep010_clean_exec_precision", "mean"),
            mean_keep010_dirty_positive_rate=("keep010_dirty_positive_rate", "mean"),
            mean_keep010_full_path_bad_mae=("keep010_full_path_bad_mae", "mean"),
            mean_keep010_first_touch_bad_mae=("keep010_first_touch_bad_mae", "mean"),
            mean_keep010_timeout=("keep010_timeout", "mean"),
            mean_keep010_oracle_recall=("keep010_oracle_recall_exec_margin", "mean"),
            mean_keep005_exec_margin=("keep005_exec_margin", "mean"),
            mean_keep005_ret_net=("keep005_ret_net", "mean"),
            mean_keep005_u_policy_net=("keep005_u_policy_net", "mean"),
            worst_keep005_exec_margin=("keep005_exec_margin", "min"),
            mean_keep005_clean_exec_precision=("keep005_clean_exec_precision", "mean"),
            mean_keep005_full_path_bad_mae=("keep005_full_path_bad_mae", "mean"),
            mean_keep005_timeout=("keep005_timeout", "mean"),
            mean_keep005_oracle_recall=("keep005_oracle_recall_exec_margin", "mean"),
            mean_spearman_exec_margin=("spearman_exec_margin", "mean"),
            mean_spearman_underwater_bars=("spearman_underwater_bars", "mean"),
            mean_auc_clean_exec=("auc_clean_exec", "mean"),
            mean_ap_clean_exec=("ap_clean_exec", "mean"),
            mean_auc_mfe_before_mae=("auc_mfe_before_mae", "mean"),
            mean_auc_avoids_mae_before_mfe=("auc_avoids_mae_before_mfe", "mean"),
        )
        .reset_index()
    )
    agg["meta_ev_frontier_objective"] = (
        100.0
        * (
            0.20 * _num(agg.get("mean_keep020_ev_after_1pct"), index=agg.index, default=0.0).fillna(0.0)
            + 0.15 * _num(agg.get("mean_keep010_ev_after_1pct"), index=agg.index, default=0.0).fillna(0.0)
            + 0.10 * _num(agg.get("worst_keep010_ev_after_1pct"), index=agg.index, default=0.0).fillna(0.0)
            + 0.10 * _num(agg.get("worst_week_keep010_ev_after_1pct"), index=agg.index, default=0.0).fillna(0.0)
        )
        + 0.20
        * (
            (
                _num(agg.get("mean_keep015_clean_exec_precision"), index=agg.index, default=0.0).fillna(0.0)
                - _num(agg.get("mean_keep015_dirty_positive_rate"), index=agg.index, default=0.0).fillna(0.0)
            )
            + (
                _num(agg.get("mean_keep015_clean_exec_precision"), index=agg.index, default=0.0).fillna(0.0)
                - pd.concat(
                    [
                        _num(agg.get("mean_keep015_first_touch_bad_mae"), index=agg.index, default=0.0),
                        _num(agg.get("mean_keep015_full_path_bad_mae"), index=agg.index, default=0.0),
                    ],
                    axis=1,
                ).max(axis=1).fillna(0.0)
            )
        )
    )
    agg["meta_smoke_status"] = np.where(
        (agg["mean_keep030_exec_margin"] > 0.0)
        & (agg["mean_keep030_timeout"] <= 0.12)
        & (agg["mean_keep030_full_path_bad_mae"] < agg["mean_keep100_full_path_bad_mae"])
        & (agg["mean_keep020_exec_margin"] > 0.0),
        "candidate_for_deeper_meta_eval",
        "diagnostic_or_fail",
    )
    return agg.sort_values(
        ["meta_smoke_status", "meta_ev_frontier_objective", "mean_keep020_ev_after_1pct"],
        ascending=[True, False, False],
    )


def _threshold_policy_rows(scored: pd.DataFrame, selector_cols: dict[str, str], test_month: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bad = _num(scored.get("score_meta_bad_path"), index=scored.index).fillna(1.0)
    timeout = _num(scored.get("score_meta_timeout"), index=scored.index).fillna(1.0)
    clean = _num(scored.get("score_meta_clean_exec"), index=scored.index).fillna(0.0)
    side = scored.get("side_name", pd.Series("", index=scored.index)).astype(str).str.lower()
    long_clean = _num(scored.get("score_meta_long_clean_exec"), index=scored.index)
    side_clean = clean.where(~side.eq("long"), long_clean.fillna(clean))
    long_bad = _num(scored.get("score_meta_long_bad_path"), index=scored.index)
    side_bad = bad.where(~side.eq("long"), long_bad.fillna(bad))
    pos = _num(scored.get("score_meta_positive_margin"), index=scored.index).fillna(0.0)
    for selector, score_col in selector_cols.items():
        if score_col not in scored.columns:
            continue
        policies: list[tuple[str, pd.Series]] = [
            ("no_cap", pd.Series(True, index=scored.index)),
        ]
        for cap in BAD_PATH_CAPS:
            policies.append((f"bad_path_le_{cap:.2f}", bad.le(float(cap))))
        for cap in (0.55, 0.60, 0.65):
            policies.append((f"bad_path_le_{cap:.2f}_timeout_le_0.50", bad.le(float(cap)) & timeout.le(0.50)))
        for floor in CLEAN_EXEC_FLOORS:
            if floor <= 0.0:
                continue
            policies.append((f"clean_ge_{floor:.2f}", clean.ge(float(floor))))
            policies.append((f"side_clean_ge_{floor:.2f}", side_clean.ge(float(floor))))
        for long_floor, short_floor in ((0.55, 0.65), (0.60, 0.65), (0.65, 0.65)):
            policies.append(
                (
                    f"side_clean_ge_long_{long_floor:.2f}_short_{short_floor:.2f}",
                    np.where(side.eq("long"), side_clean.ge(float(long_floor)), clean.ge(float(short_floor))),
                )
            )
        for cap in (0.50, 0.55, 0.60):
            policies.append((f"side_bad_path_le_{cap:.2f}", side_bad.le(float(cap))))
            policies.append(
                (
                    f"side_bad_path_le_{cap:.2f}_side_clean_ge_0.55",
                    side_bad.le(float(cap)) & side_clean.ge(0.55),
                )
            )
        for floor in POSITIVE_MARGIN_FLOORS:
            if floor <= 0.0:
                continue
            policies.append((f"positive_margin_ge_{floor:.2f}", pos.ge(float(floor))))
        for cap in (0.55, 0.60, 0.65):
            for clean_floor in (0.45, 0.55):
                policies.append(
                    (
                        f"bad_path_le_{cap:.2f}_clean_ge_{clean_floor:.2f}",
                        bad.le(float(cap)) & clean.ge(float(clean_floor)),
                    )
                )
        for policy_id, eligible in policies:
            for budget_frac in POLICY_BUDGET_FRACTIONS:
                rows.append(
                    _policy_metric_row(
                        scored,
                        score_col,
                        selector=selector,
                        policy_id=policy_id,
                        test_month=test_month,
                        budget_frac=budget_frac,
                        eligible=eligible,
                    )
                )
    return rows


def _summarize_threshold_policies(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    summary = (
        rows.groupby(["selector", "policy_id", "budget_frac"], dropna=False)
        .agg(
            folds=("test_month", "nunique"),
            mean_target_rows=("target_rows", "mean"),
            mean_selected_rows=("selected_rows", "mean"),
            min_selected_rows=("selected_rows", "min"),
            mean_fill_rate=("fill_rate", "mean"),
            mean_no_trade_rate=("no_trade_rate", "mean"),
            mean_exec_margin=("exec_margin", "mean"),
            worst_exec_margin=("exec_margin", "min"),
            mean_ev_after_1pct=("ev_after_1pct", "mean"),
            worst_week_ev_after_1pct=("worst_week_ev_after_1pct", "min"),
            mean_positive_margin_rate=("positive_margin_rate", "mean"),
            mean_clean_exec_precision=("clean_exec_precision", "mean"),
            mean_dirty_positive_rate=("dirty_positive_rate", "mean"),
            mean_first_touch_bad_mae=("first_touch_bad_mae", "mean"),
            mean_full_path_bad_mae=("full_path_bad_mae", "mean"),
            max_full_path_bad_mae=("full_path_bad_mae", "max"),
            mean_timeout=("timeout", "mean"),
            max_timeout=("timeout", "max"),
            mean_mfe_before_mae=("mfe_before_mae", "mean"),
            mean_mae_before_mfe=("mae_before_mfe", "mean"),
            mean_oracle_recall=("oracle_recall_exec_margin", "mean"),
            mean_long_share=("long_share", "mean"),
            mean_short_share=("short_share", "mean"),
        )
        .reset_index()
    )
    summary["threshold_policy_status"] = np.where(
        (summary["mean_exec_margin"] > 0.0)
        & (summary["worst_exec_margin"] > 0.0)
        & (summary["mean_full_path_bad_mae"] <= 0.50)
        & (summary["mean_timeout"] <= 0.12)
        & (summary["min_selected_rows"] >= 20)
        & (summary["mean_fill_rate"] >= 0.20),
        "bad_mae_bar_candidate",
        np.where(
            (summary["mean_exec_margin"] > 0.0)
            & (summary["mean_full_path_bad_mae"] < 0.58)
            & (summary["mean_timeout"] <= 0.12),
            "risk_improved_diagnostic",
            "diagnostic_or_fail",
        ),
    )
    objective_by_policy: dict[tuple[str, str], float] = {}
    for key, group in summary.groupby(["selector", "policy_id"], dropna=False):
        budget = _num(group.get("budget_frac"), index=group.index, default=0.0).fillna(0.0).round(4)

        def pick_metric(budget_value: float, col: str, default: float = 0.0) -> float:
            match = group.loc[budget.eq(round(float(budget_value), 4)), col]
            if match.empty:
                return float(default)
            return _mean(match)

        top20_ev = pick_metric(0.20, "mean_ev_after_1pct")
        top10_ev = pick_metric(0.10, "mean_ev_after_1pct")
        worst_month_top10_ev = pick_metric(0.10, "worst_exec_margin")
        worst_week_top10_ev = pick_metric(0.10, "worst_week_ev_after_1pct")
        top15_clean = pick_metric(0.15, "mean_clean_exec_precision")
        top15_dirty = pick_metric(0.15, "mean_dirty_positive_rate")
        top15_bad_or_stop = max(
            pick_metric(0.15, "mean_first_touch_bad_mae"),
            pick_metric(0.15, "mean_full_path_bad_mae"),
        )
        objective_by_policy[key] = float(
            100.0
            * (
                0.20 * top20_ev
                + 0.15 * top10_ev
                + 0.10 * worst_month_top10_ev
                + 0.10 * worst_week_top10_ev
            )
            + 0.20 * ((top15_clean - top15_dirty) + (top15_clean - top15_bad_or_stop))
        )
    summary["meta_ev_frontier_objective"] = [
        objective_by_policy.get((row.selector, row.policy_id), float("-inf")) for row in summary.itertuples(index=False)
    ]
    return summary.sort_values(
        [
            "threshold_policy_status",
            "meta_ev_frontier_objective",
            "mean_ev_after_1pct",
            "mean_fill_rate",
        ],
        ascending=[True, False, False, False],
    )


def run_smoke(
    *,
    handoff_dir: Path,
    ledger_path: Path | None,
    out_dir: Path,
    frontier: str,
    seed: int,
    train_scope: str = "selected",
    enable_base_prior_features: bool = False,
    enable_reliability_features: bool = False,
    enable_support_drift_features: bool = False,
    enable_hit_surprise_features: bool = False,
    enable_path_order_heads: bool = False,
    enable_path_order_blends: bool = False,
    feature_selection_top_n: int = 0,
    feature_selection_target: str = "ev_frontier",
    feature_selection_method: str = "auto",
    max_oos_model_age_days: int = 0,
    validation_scope: str = "all",
    model_train_max_rows: int = 0,
    model_params: dict[str, Any] | None = None,
    model_profile_name: str = "baseline",
    meta_head_mode: str = "multi",
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    handoff_path = handoff_dir / "train_meta_regime_handoff.parquet"
    if ledger_path is None:
        ledger_path = handoff_dir / "s52_trailing_regime_scored_ledger.parquet"
    data = _load_joined_frame(handoff_path, ledger_path, frontier)
    selected_col = _candidate_column(frontier)
    if train_scope == "selected":
        data = data[data[selected_col]].copy()
    elif train_scope != "all":
        raise ValueError("--train-scope must be selected or all")
    months = sorted(str(m) for m in data["month"].dropna().unique())
    if len(months) < 2:
        raise ValueError(f"Need at least two months, got {months}")
    ts_utc = pd.to_datetime(data["__ts__"], utc=True, errors="coerce")
    validation_windows: list[dict[str, Any]] = []
    for month in months[1:]:
        month_start = pd.Timestamp(pd.Period(month).start_time, tz="UTC")
        month_end = pd.Timestamp((pd.Period(month) + 1).start_time, tz="UTC")
        if int(max_oos_model_age_days) > 0:
            start = month_start
            step = pd.Timedelta(days=int(max_oos_model_age_days))
            while start < month_end:
                end = min(start + step, month_end)
                validation_windows.append(
                    {
                        "fold": f"{start:%Y-%m-%d}_{end:%Y-%m-%d}",
                        "month": str(month),
                        "valid_start": start,
                        "valid_end": end,
                    }
                )
                start = end
        else:
            validation_windows.append(
                {
                    "fold": str(month),
                    "month": str(month),
                    "valid_start": month_start,
                    "valid_end": month_end,
                }
            )
    eligible_windows: list[dict[str, Any]] = []
    for window in validation_windows:
        train_rows = int(ts_utc.lt(window["valid_start"]).sum())
        valid_rows = int((ts_utc.ge(window["valid_start"]) & ts_utc.lt(window["valid_end"])).sum())
        if train_rows < 100 or valid_rows < 30:
            continue
        enriched = dict(window)
        enriched["train_rows_estimate"] = int(train_rows)
        enriched["valid_rows_estimate"] = int(valid_rows)
        eligible_windows.append(enriched)
    calibration_candidates = [w for w in eligible_windows if int(w["valid_rows_estimate"]) >= 1000]
    if not calibration_candidates:
        calibration_candidates = eligible_windows
    calibration_window = max(
        calibration_candidates,
        key=lambda w: (int(w["train_rows_estimate"]), int(w["valid_rows_estimate"])),
    ) if calibration_candidates else None
    calibration_fold = str(calibration_window["fold"]) if calibration_window is not None else None
    if str(validation_scope).strip().lower() == "largest":
        validation_windows = [calibration_window] if calibration_window is not None else []
    elif calibration_window is not None:
        validation_windows = [calibration_window] + [
            w for w in eligible_windows if str(w["fold"]) != calibration_fold
        ]
    else:
        validation_windows = []
    numeric_cols, categorical_cols = _feature_columns(
        data,
        enable_base_prior_features=enable_base_prior_features,
        enable_reliability_features=enable_reliability_features,
        enable_support_drift_features=enable_support_drift_features,
        enable_hit_surprise_features=enable_hit_surprise_features,
    )
    fold_rows: list[dict[str, Any]] = []
    threshold_policy_rows: list[dict[str, Any]] = []
    breakdown: list[dict[str, Any]] = []
    base_conditioned_diagnostics: list[dict[str, Any]] = []
    importances: list[pd.DataFrame] = []
    feature_selection_frames: list[pd.DataFrame] = []
    selected_features_by_fold: dict[str, list[str]] = {}
    prediction_frames: list[pd.DataFrame] = []
    classifier_params = dict((model_params or {}).get("classifier", DEFAULT_LGBM_CLASSIFIER_PARAMS))
    regressor_params = dict((model_params or {}).get("regressor", DEFAULT_LGBM_REGRESSOR_PARAMS))
    meta_head_mode = str(meta_head_mode).strip().lower()
    if meta_head_mode not in {"multi", "single_base_soft_label"}:
        raise ValueError("--meta-head-mode must be multi or single_base_soft_label")
    fs_cache_key = (
        str(handoff_path.resolve()),
        str(Path(ledger_path).resolve()) if ledger_path is not None else None,
        str(frontier),
        str(train_scope),
        bool(enable_base_prior_features),
        bool(enable_reliability_features),
        bool(enable_support_drift_features),
        bool(enable_hit_surprise_features),
        int(feature_selection_top_n),
        str(feature_selection_target),
        str(feature_selection_method),
        int(max_oos_model_age_days),
        str(meta_head_mode),
    )
    cached_feature_selection = _FEATURE_SELECTION_CACHE.get(fs_cache_key)
    global_feature_names: list[str] | None = list(cached_feature_selection[0]) if cached_feature_selection else None
    global_feature_selection_df: pd.DataFrame | None = (
        cached_feature_selection[1].copy() if cached_feature_selection else None
    )
    feature_selection_recorded = False
    meta_target_columns_used: set[str] = set()
    for fold_idx, window in enumerate(validation_windows, start=1):
        test_fold = str(window["fold"])
        test_month = str(window["month"])
        train = data[ts_utc.lt(window["valid_start"])].copy()
        valid = data[ts_utc.ge(window["valid_start"]) & ts_utc.lt(window["valid_end"])].copy()
        if len(train) < 100 or len(valid) < 30:
            continue
        print(
            json.dumps(
                {
                    "event": "s52_train_meta_fold_start",
                    "frontier": frontier,
                    "test_fold": test_fold,
                    "test_month": test_month,
                    "valid_start": str(window["valid_start"]),
                    "valid_end": str(window["valid_end"]),
                    "max_oos_model_age_days": int(max_oos_model_age_days),
                    "train_rows": int(len(train)),
                    "valid_rows": int(len(valid)),
                },
                sort_keys=True,
            ),
                flush=True,
            )
        train, valid = _add_fold_base_prior_features(train, valid, selected_col=selected_col)
        if enable_reliability_features:
            train, valid = _add_fold_reliability_features(train, valid)
        if enable_support_drift_features:
            train, valid = _add_fold_support_drift_features(train, valid)
        if enable_hit_surprise_features:
            train, valid = _add_fold_hit_surprise_features(train, valid)
        # HPO/feature-selection trials only need the capped, time-spread training
        # sample. Build the feature matrix after sampling so the largest-fold
        # search does not materialize a full 1M+ row x feature frame just to
        # discard most rows before fitting.
        train_matrix = train
        if int(model_train_max_rows) > 0:
            train_matrix_idx = _time_spread_cap_rows(len(train), int(model_train_max_rows))
            train_matrix = train.iloc[train_matrix_idx].reset_index(drop=True)
        x_train, x_valid, feature_names = _make_xy(
            train_matrix,
            valid,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
        )
        if global_feature_names is None:
            fs_fold_name = f"largest_train_before_{window['valid_start']:%Y-%m-%d}"
            method = str(feature_selection_method).strip().lower()
            if method not in {"auto", "lgbm_pipeline", "lgbm_staged"}:
                raise ValueError(
                    "--feature-selection-method now routes through lgbm_pipeline.py only. "
                    "Use auto/lgbm_pipeline/lgbm_staged."
                )
            x_train, x_valid, feature_names, feature_selection_df = _select_features_by_lgbm_pipeline(
                x_train,
                x_valid,
                train_matrix,
                target_name=str(feature_selection_target),
                top_n=int(feature_selection_top_n),
                fold=fs_fold_name,
                seed=int(seed) + fold_idx,
            )
            global_feature_names = list(feature_names)
            global_feature_selection_df = feature_selection_df.copy()
            _FEATURE_SELECTION_CACHE[fs_cache_key] = (list(global_feature_names), global_feature_selection_df.copy())
            feature_selection_frames.append(feature_selection_df)
            feature_selection_recorded = True
        else:
            x_train = x_train.reindex(columns=global_feature_names, fill_value=0.0)
            x_valid = x_valid.reindex(columns=global_feature_names, fill_value=0.0)
            feature_names = list(global_feature_names)
            if not feature_selection_recorded and global_feature_selection_df is not None:
                feature_selection_frames.append(global_feature_selection_df.copy())
                feature_selection_recorded = True
        selected_features_by_fold[str(test_fold)] = list(feature_names)
        fit_idx = _time_spread_cap_rows(len(x_train), int(model_train_max_rows))
        x_train_fit = x_train.iloc[fit_idx].reset_index(drop=True)
        train_fit = train_matrix.iloc[fit_idx].reset_index(drop=True)
        scored = valid.copy()
        scored["score_base"] = _num(scored.get("score"), index=scored.index)
        if meta_head_mode == "single_base_soft_label":
            base_soft_target, base_soft_target_col = _base_soft_label_target(train_fit)
            meta_target_columns_used.add(str(base_soft_target_col))
            models = {
                "base_soft_label": _fit_base_soft_label_model(
                    x_train_fit,
                    base_soft_target,
                    train_fit,
                    seed + fold_idx,
                    lgbm_params=classifier_params,
                )
            }
            scored["score_meta_base_soft_label"] = _predict(models["base_soft_label"], x_valid, classifier=False)
            scored["meta_base_soft_label_target_col"] = base_soft_target_col
            selector_cols = {
                "base_score": "score_base",
                "meta_base_soft_label": "score_meta_base_soft_label",
            }
        else:
            models = {
                "clean_exec": _fit_classifier(
                    x_train_fit,
                    train_fit["clean_exec_label"],
                    train_fit,
                    seed + fold_idx,
                    lgbm_params=classifier_params,
                ),
                "positive_margin": _fit_classifier(
                    x_train_fit,
                    train_fit["positive_exec_margin"],
                    train_fit,
                    seed + 11 + fold_idx,
                    lgbm_params=classifier_params,
                ),
                "bad_path": _fit_classifier(
                    x_train_fit,
                    train_fit["bad_path_label"],
                    train_fit,
                    seed + 23 + fold_idx,
                    lgbm_params=classifier_params,
                ),
                "timeout": _fit_classifier(
                    x_train_fit,
                    train_fit["timeout"],
                    train_fit,
                    seed + 31 + fold_idx,
                    lgbm_params=classifier_params,
                ),
                "exec_margin": _fit_regressor(
                    x_train_fit,
                    train_fit["exec_margin"],
                    train_fit,
                    seed + 43 + fold_idx,
                    lgbm_params=regressor_params,
                ),
                "long_clean_exec": _fit_side_classifier(
                    x_train_fit,
                    train_fit["long_path_clean_exec_label"],
                    train_fit,
                    side="long",
                    seed=seed + 53 + fold_idx,
                    lgbm_params=classifier_params,
                ),
                "long_bad_path": _fit_side_classifier(
                    x_train_fit,
                    train_fit["long_bad_path_label"],
                    train_fit,
                    side="long",
                    seed=seed + 61 + fold_idx,
                    lgbm_params=classifier_params,
                ),
            }
            if enable_path_order_heads:
                models.update(
                    {
                        "mfe_before_mae": _fit_classifier(
                            x_train_fit,
                            train_fit["path_mfe_before_mae_label"],
                            train_fit,
                            seed + 47 + fold_idx,
                            lgbm_params=classifier_params,
                        ),
                        "mae_before_mfe": _fit_classifier(
                            x_train_fit,
                            train_fit["path_mae_before_mfe_label"],
                            train_fit,
                            seed + 49 + fold_idx,
                            lgbm_params=classifier_params,
                        ),
                        "underwater_duration": _fit_regressor(
                            x_train_fit,
                            train_fit["path_underwater_duration_target"],
                            train_fit,
                            seed + 51 + fold_idx,
                            lgbm_params=regressor_params,
                        ),
                    }
                )
            scored["score_meta_clean_exec"] = _predict(models["clean_exec"], x_valid, classifier=True)
            scored["score_meta_positive_margin"] = _predict(models["positive_margin"], x_valid, classifier=True)
            scored["score_meta_bad_path"] = _predict(models["bad_path"], x_valid, classifier=True)
            scored["score_meta_timeout"] = _predict(models["timeout"], x_valid, classifier=True)
            scored["score_meta_exec_margin"] = _predict(models["exec_margin"], x_valid, classifier=False)
            if enable_path_order_heads:
                scored["score_meta_mfe_before_mae"] = _predict(models["mfe_before_mae"], x_valid, classifier=True)
                scored["score_meta_mae_before_mfe"] = _predict(models["mae_before_mfe"], x_valid, classifier=True)
                scored["score_meta_underwater_duration"] = _predict(models["underwater_duration"], x_valid, classifier=False)
                underwater_penalty = _num(scored["score_meta_underwater_duration"], index=scored.index).fillna(0.0).clip(lower=0.0)
                scored["score_meta_path_order"] = (
                    scored["score_meta_mfe_before_mae"].fillna(0.0)
                    - 0.85 * scored["score_meta_mae_before_mfe"].fillna(1.0)
                    - 0.08 * underwater_penalty
                )
            else:
                underwater_penalty = pd.Series(0.0, index=scored.index, dtype=np.float32)
            long_valid = scored["side_name"].astype(str).str.lower().eq("long")
            scored["score_meta_long_clean_exec"] = _predict(models["long_clean_exec"], x_valid, classifier=True)
            scored.loc[~long_valid, "score_meta_long_clean_exec"] = np.nan
            scored["score_meta_long_bad_path"] = _predict(models["long_bad_path"], x_valid, classifier=True)
            scored.loc[~long_valid, "score_meta_long_bad_path"] = np.nan
            scored["score_meta_clean_minus_risk"] = (
                scored["score_meta_clean_exec"].fillna(0.0)
                + 0.60 * scored["score_meta_positive_margin"].fillna(0.0)
                - 0.70 * scored["score_meta_bad_path"].fillna(0.0)
                - 0.30 * scored["score_meta_timeout"].fillna(0.0)
            )
            if enable_path_order_heads:
                scored["score_meta_path_order_clean_minus_risk"] = (
                    scored["score_meta_clean_minus_risk"].fillna(0.0)
                    + 0.35 * scored["score_meta_mfe_before_mae"].fillna(0.0)
                    - 0.35 * scored["score_meta_mae_before_mfe"].fillna(1.0)
                    - 0.04 * underwater_penalty
                )
            scored["score_meta_exec_margin_risk_blend"] = (
                scored["score_meta_exec_margin"].fillna(0.0)
                + 0.0030 * scored["score_meta_clean_exec"].fillna(0.0)
                + 0.0020 * scored["score_meta_positive_margin"].fillna(0.0)
                - 0.0040 * scored["score_meta_bad_path"].fillna(0.0)
                - 0.0020 * scored["score_meta_timeout"].fillna(0.0)
            )
            if enable_path_order_blends and enable_path_order_heads:
                scored["score_meta_exec_margin_risk_blend"] = (
                    scored["score_meta_exec_margin_risk_blend"].fillna(0.0)
                    + 0.0015 * scored["score_meta_mfe_before_mae"].fillna(0.0)
                    - 0.0015 * scored["score_meta_mae_before_mfe"].fillna(1.0)
                    - 0.0002 * underwater_penalty
                )
            scored["score_meta_context_hint_blend"] = (
                scored["score_meta_exec_margin_risk_blend"].fillna(0.0)
                + 0.0010 * _num(scored.get("meta_context_weight_hint"), index=scored.index, default=0.0).fillna(0.0)
                - 0.0010 * _num(scored.get("meta_threshold_adjustment_hint"), index=scored.index, default=0.0).fillna(0.0)
            )
            long_clean_score = scored["score_meta_long_clean_exec"].where(
                scored["score_meta_long_clean_exec"].notna(),
                scored["score_meta_clean_exec"],
            )
            long_bad_score = scored["score_meta_long_bad_path"].where(
                scored["score_meta_long_bad_path"].notna(),
                scored["score_meta_bad_path"],
            )
            scored["score_meta_long_aware_clean_minus_risk"] = scored["score_meta_clean_minus_risk"]
            long_aware = (
                long_clean_score.loc[long_valid].fillna(0.0)
                + 0.55 * scored.loc[long_valid, "score_meta_positive_margin"].fillna(0.0)
                - 0.80 * long_bad_score.loc[long_valid].fillna(1.0)
                - 0.25 * scored.loc[long_valid, "score_meta_timeout"].fillna(1.0)
            )
            if enable_path_order_blends and enable_path_order_heads:
                long_aware = (
                    long_aware
                    + 0.25 * scored.loc[long_valid, "score_meta_mfe_before_mae"].fillna(0.0)
                    - 0.25 * scored.loc[long_valid, "score_meta_mae_before_mfe"].fillna(1.0)
                    - 0.03 * underwater_penalty.loc[long_valid]
                )
            scored.loc[long_valid, "score_meta_long_aware_clean_minus_risk"] = long_aware
            selector_cols = {
                "base_score": "score_base",
                "meta_clean_exec": "score_meta_clean_exec",
                "meta_positive_margin": "score_meta_positive_margin",
                "meta_exec_margin": "score_meta_exec_margin",
                "meta_clean_minus_risk": "score_meta_clean_minus_risk",
                "meta_exec_margin_risk_blend": "score_meta_exec_margin_risk_blend",
                "meta_context_hint_blend": "score_meta_context_hint_blend",
                "meta_long_aware_clean_minus_risk": "score_meta_long_aware_clean_minus_risk",
            }
            if enable_path_order_heads:
                selector_cols["meta_path_order"] = "score_meta_path_order"
                selector_cols["meta_path_order_clean_minus_risk"] = "score_meta_path_order_clean_minus_risk"
        scored["oos_fold"] = str(test_fold)
        scored["calendar_month"] = str(test_month)
        scored["valid_start"] = window["valid_start"]
        scored["valid_end"] = window["valid_end"]
        scored["max_oos_model_age_days"] = int(max_oos_model_age_days)
        for selector, score_col in selector_cols.items():
            selector_row = _selector_metrics(scored, score_col, selector, test_fold)
            selector_row.update(
                {
                    "calendar_month": str(test_month),
                    "valid_start": window["valid_start"],
                    "valid_end": window["valid_end"],
                    "max_oos_model_age_days": int(max_oos_model_age_days),
                }
            )
            fold_rows.append(selector_row)
            for keep_frac in (0.30, 0.20, 0.10):
                for row in _breakdown_rows(scored, score_col, selector, test_fold, keep_frac):
                    row.update(
                        {
                            "calendar_month": str(test_month),
                            "valid_start": window["valid_start"],
                            "valid_end": window["valid_end"],
                            "max_oos_model_age_days": int(max_oos_model_age_days),
                        }
                    )
                    breakdown.append(row)
            if selector in {
                "base_score",
                "meta_long_aware_clean_minus_risk",
                "meta_path_order_clean_minus_risk",
                "meta_exec_margin_risk_blend",
            }:
                base_conditioned_diagnostics.extend(
                    {
                        **row,
                        "calendar_month": str(test_month),
                        "valid_start": window["valid_start"],
                        "valid_end": window["valid_end"],
                        "max_oos_model_age_days": int(max_oos_model_age_days),
                    }
                    for row in _base_conditioned_diagnostic_rows(
                        scored,
                        test_month=test_fold,
                        selector=selector,
                        score_col=score_col,
                        keep_frac=0.30,
                    )
                )
        for row in _threshold_policy_rows(scored, selector_cols, test_fold):
            row.update(
                {
                    "calendar_month": str(test_month),
                    "valid_start": window["valid_start"],
                    "valid_end": window["valid_end"],
                    "max_oos_model_age_days": int(max_oos_model_age_days),
                }
            )
            threshold_policy_rows.append(row)
        for label, model in models.items():
            importances.append(_feature_importance(model, feature_names, label, test_fold))
        keep_cols = [
            "__ts__",
            "__symbol__",
            "side_name",
            "month",
            "oos_fold",
            "calendar_month",
            "valid_start",
            "valid_end",
            "max_oos_model_age_days",
            "source_semantic_family",
            "source_semantic_family_base",
            "long_source_regime_split",
            "aegmm_cluster",
            "side_aegmm_cluster",
            "aegmm_expected_distance_bin",
            "reconstruction_bin",
            "exec_margin",
            "ev_after_1pct",
            "ret_net",
            "u_policy_net",
            "first_touch_gross",
            "first_touch_bad_mae_1r",
            "full_path_bad_mae_1r",
            "timeout",
            "mfe_before_mae_1r",
            "mae_before_mfe_1r",
            "clean_exec",
            "dirty_positive",
            "underwater_bars_before_mfe_1r",
            "long_path_clean_exec_label",
            "long_path_dirty_positive_label",
            "long_path_post_mfe_drawdown_norm",
            "long_path_time_to_profit_bars",
            "long_path_slow_profit",
            "long_path_post_mfe_bad_drawdown",
            "long_bad_path_label",
            "__archetype_label_family__",
            "__archetype_label_source__",
            "__archetype_policy_key__",
            "__archetype_policy_role__",
            "__archetype_policy_confidence__",
            "__archetype_policy_tp_r__",
            "__archetype_policy_sl_r__",
            "__archetype_policy_trail_r__",
            "__archetype_policy_max_bars_to_mfe__",
            "__archetype_policy_max_barrier__",
            "score_meta_bad_path",
            "score_meta_timeout",
            "score_meta_mfe_before_mae",
            "score_meta_mae_before_mfe",
            "score_meta_underwater_duration",
            "score_meta_path_order",
            "score_meta_path_order_clean_minus_risk",
            "score_meta_long_clean_exec",
            "score_meta_long_bad_path",
            "base_margin_to_cutoff",
            "base_margin_to_cutoff_z",
            "base_signal_zscore_within_archetype",
            "base_score_rank_pct_train_prior",
            "base_rank_band",
            "base_margin_band",
        ] + list(RELIABILITY_NUMERIC_FEATURES) + list(SUPPORT_DRIFT_NUMERIC_FEATURES) + list(HIT_SURPRISE_NUMERIC_FEATURES) + list(selector_cols.values())
        export_cols = [col for col in dict.fromkeys(keep_cols) if col in scored.columns]
        prediction_frames.append(scored[export_cols].copy())
    folds = pd.DataFrame(fold_rows)
    summary = _summarize(folds)
    threshold_policies = pd.DataFrame(threshold_policy_rows)
    threshold_summary = _summarize_threshold_policies(threshold_policies)
    breakdown_df = pd.DataFrame(breakdown)
    base_conditioned_df = pd.DataFrame(base_conditioned_diagnostics)
    importance_df = (
        pd.concat([part for part in importances if not part.empty], ignore_index=True)
        if any(not part.empty for part in importances)
        else pd.DataFrame(columns=["test_month", "model", "feature", "importance"])
    )
    feature_selection_all = (
        pd.concat(feature_selection_frames, ignore_index=True)
        if feature_selection_frames
        else pd.DataFrame(
            columns=[
                "fold",
                "feature",
                "score",
                "rank",
                "selected",
                "feature_selection_target",
                "feature_selection_status",
            ]
        )
    )
    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    folds.to_csv(out_dir / "s52_train_meta_regime_handoff_smoke_folds.csv", index=False)
    summary.to_csv(out_dir / "s52_train_meta_regime_handoff_smoke_summary.csv", index=False)
    threshold_policies.to_csv(out_dir / "s52_train_meta_regime_handoff_threshold_policy_folds.csv", index=False)
    threshold_summary.to_csv(out_dir / "s52_train_meta_regime_handoff_threshold_policy_summary.csv", index=False)
    breakdown_df.to_csv(out_dir / "s52_train_meta_regime_handoff_smoke_breakdown.csv", index=False)
    base_conditioned_df.to_csv(out_dir / "s52_train_meta_base_conditioned_diagnostics.csv", index=False)
    importance_df.to_csv(out_dir / "s52_train_meta_regime_handoff_smoke_feature_importance.csv", index=False)
    feature_selection_all.to_csv(out_dir / "s52_train_meta_feature_selection_by_fold.csv", index=False)
    if not predictions.empty:
        predictions.to_parquet(out_dir / "s52_train_meta_regime_handoff_smoke_predictions.parquet", index=False)
    best = summary.iloc[0].to_dict() if not summary.empty else {}
    best_threshold = threshold_summary.iloc[0].to_dict() if not threshold_summary.empty else {}
    selected_sets = [set(v) for v in selected_features_by_fold.values()]
    selected_union = sorted(set().union(*selected_sets)) if selected_sets else []
    selected_intersection = sorted(set.intersection(*selected_sets)) if selected_sets else []
    manifest = {
        "generated_by": "run_s52_train_meta_regime_handoff_smoke",
        "handoff_dir": str(handoff_dir),
        "handoff_path": str(handoff_path),
        "ledger_path": str(ledger_path),
        "frontier": str(frontier),
        "train_scope": str(train_scope),
        "months": months,
        "rows": int(len(data)),
        "numeric_feature_count": int(len(numeric_cols)),
        "categorical_feature_count": int(len(categorical_cols)),
        "feature_selection_scope": "single_global_largest_train_window",
        "feature_selection_calibration_fold": calibration_fold,
        "calibration_min_valid_rows": 1000,
        "validation_scope": str(validation_scope),
        "global_feature_selection_fold": (
            str(feature_selection_all["fold"].iloc[0]) if not feature_selection_all.empty else None
        ),
        "feature_selection_method": str(feature_selection_method),
        "feature_selection_top_n": int(feature_selection_top_n),
        "feature_selection_target": str(feature_selection_target),
        "meta_head_mode": str(meta_head_mode),
        "meta_target_columns_used": sorted(meta_target_columns_used),
        "single_head_model_contract": (
            "LGBMRegressor objective=regression on base target_soft with W7_timestamp_balanced weights"
            if meta_head_mode == "single_base_soft_label"
            else "legacy_multi_head"
        ),
        "max_oos_model_age_days": int(max_oos_model_age_days),
        "model_train_max_rows": int(model_train_max_rows),
        "model_train_sampling": (
            "beginning_middle_end_time_spread"
            if int(model_train_max_rows) > 0
            else "full_train_rows"
        ),
        "fold_windows": [
            {
                "fold": row.get("test_month"),
                "calendar_month": row.get("calendar_month"),
                "valid_start": row.get("valid_start"),
                "valid_end": row.get("valid_end"),
                "max_oos_model_age_days": row.get("max_oos_model_age_days"),
            }
            for row in fold_rows
            if row.get("selector") == "base_score"
        ],
        "selected_features_by_fold": selected_features_by_fold,
        "selected_feature_union": selected_union,
        "selected_feature_intersection": selected_intersection,
        "selected_feature_union_count": int(len(selected_union)),
        "selected_feature_intersection_count": int(len(selected_intersection)),
        "feature_selection_output": str(out_dir / "s52_train_meta_feature_selection_by_fold.csv"),
        "model_profile_name": str(model_profile_name),
        "classifier_params": _json_safe(classifier_params),
        "regressor_params": _json_safe(regressor_params),
        "added_path_order_targets": [
            "path_mfe_before_mae_label",
            "path_mae_before_mfe_label",
            "path_underwater_duration_target",
        ],
        "added_base_conditioned_features": list(BASE_PRIOR_NUMERIC_FEATURES) + list(BASE_PRIOR_CATEGORICAL_FEATURES),
        "added_reliability_features": list(RELIABILITY_NUMERIC_FEATURES),
        "added_support_drift_features": list(SUPPORT_DRIFT_NUMERIC_FEATURES),
        "added_hit_surprise_features": list(HIT_SURPRISE_NUMERIC_FEATURES),
        "hit_surprise_half_life_days": list(HIT_SURPRISE_HALFLIFE_DAYS),
        "hit_surprise_window_cap": "4x_half_life",
        "hit_surprise_target": "clean_exec_label",
        "support_drift_source_columns": [col for col in SUPPORT_DRIFT_COLUMNS if col in data.columns],
        "enable_base_prior_features": bool(enable_base_prior_features),
        "enable_reliability_features": bool(enable_reliability_features),
        "enable_support_drift_features": bool(enable_support_drift_features),
        "enable_hit_surprise_features": bool(enable_hit_surprise_features),
        "enable_path_order_heads": bool(enable_path_order_heads),
        "enable_path_order_blends": bool(enable_path_order_blends),
        "base_conditioned_diagnostics": str(out_dir / "s52_train_meta_base_conditioned_diagnostics.csv"),
        "lightgbm_available": bool(_LIGHTGBM_AVAILABLE),
        "best_selector": _json_safe(best),
        "best_threshold_policy": _json_safe(best_threshold),
        "leakage_contract": {
            "feature_source": "train_meta_regime_handoff.parquet",
            "outcomes_joined_for": "training_labels_and_validation_metrics_only",
            "split": (
                "expanding_window_with_oos_age_cap"
                if int(max_oos_model_age_days) > 0
                else "month_forward_train_past_validate_next_month"
            ),
            "max_oos_model_age_days": int(max_oos_model_age_days),
            "primary_metrics": "top-k exec_margin/clean precision/bad-MAE/timeout",
            "threshold_policy_note": "fixed smoke templates evaluated OOS; not validation-optimized production thresholds",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))
    _write_markdown(out_dir, summary, folds, threshold_summary, manifest)
    return manifest


def _fmt_pct(value: Any) -> str:
    try:
        val = float(value)
    except Exception:
        return "nan"
    if not math.isfinite(val):
        return "nan"
    return f"{val * 100:.2f}%"


def _write_markdown(
    out_dir: Path,
    summary: pd.DataFrame,
    folds: pd.DataFrame,
    threshold_summary: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    lines = [
        "# S52 Train-Meta Regime Handoff Smoke",
        "",
        "## Scope",
        "",
        "Controlled month-forward diagnostic over the S52 row-level regime handoff artifact.",
        "This is not production train_meta and is not frozen replay evidence.",
        "",
        "## Contract",
        "",
        f"- frontier: `{manifest.get('frontier')}`",
        f"- train scope: `{manifest.get('train_scope')}`",
        f"- rows: `{manifest.get('rows')}`",
        f"- numeric features: `{manifest.get('numeric_feature_count')}`",
        f"- categorical features: `{manifest.get('categorical_feature_count')}`",
        f"- months: `{', '.join(manifest.get('months', []))}`",
        "",
        "## Best Summary",
        "",
    ]
    if summary.empty:
        lines.append("No valid folds were produced.")
    else:
        display_cols = [
            "selector",
            "meta_smoke_status",
            "mean_keep030_exec_margin",
            "mean_keep030_clean_exec_precision",
            "mean_keep030_full_path_bad_mae",
            "mean_keep030_timeout",
            "mean_keep020_exec_margin",
            "mean_keep020_ret_net",
            "mean_keep010_exec_margin",
            "mean_keep010_ret_net",
            "mean_keep005_exec_margin",
            "mean_keep005_ret_net",
            "worst_keep010_exec_margin",
            "mean_keep030_oracle_recall",
        ]
        head = summary[[col for col in display_cols if col in summary.columns]].head(8)
        lines.append(head.to_markdown(index=False))
        best = summary.iloc[0]
        lines.extend(
            [
                "",
                "## Read",
                "",
                f"- best selector: `{best.get('selector')}`",
                f"- keep30 executable margin: `{_fmt_pct(best.get('mean_keep030_exec_margin'))}`",
                f"- keep30 clean precision: `{_fmt_pct(best.get('mean_keep030_clean_exec_precision'))}`",
                f"- keep30 full-path bad-MAE: `{_fmt_pct(best.get('mean_keep030_full_path_bad_mae'))}`",
                f"- keep30 timeout: `{_fmt_pct(best.get('mean_keep030_timeout'))}`",
                f"- keep20 average net return: `{_fmt_pct(best.get('mean_keep020_ret_net'))}`",
                f"- keep10 average net return: `{_fmt_pct(best.get('mean_keep010_ret_net'))}`",
                f"- keep5 average net return: `{_fmt_pct(best.get('mean_keep005_ret_net'))}`",
                f"- status: `{best.get('meta_smoke_status')}`",
            ]
        )
    if not folds.empty:
        lines.extend(["", "## Fold Rows", "", folds.head(30).to_markdown(index=False)])
    if not threshold_summary.empty:
        display_cols = [
            "selector",
            "policy_id",
            "budget_frac",
            "threshold_policy_status",
            "mean_selected_rows",
            "mean_fill_rate",
            "mean_exec_margin",
            "worst_exec_margin",
            "mean_clean_exec_precision",
            "mean_full_path_bad_mae",
            "mean_timeout",
            "mean_oracle_recall",
        ]
        lines.extend(
            [
                "",
                "## Threshold / Abstention Sweep",
                "",
                "Fixed risk templates evaluated OOS. These are diagnostic threshold candidates, not production-optimized thresholds.",
                "",
                threshold_summary[[col for col in display_cols if col in threshold_summary.columns]]
                .head(15)
                .to_markdown(index=False),
            ]
        )
    (out_dir / "s52_train_meta_regime_handoff_smoke.md").write_text("\n".join(lines) + "\n")


def _meta_trial_objective(manifest: dict[str, Any]) -> float:
    best = dict(manifest.get("best_selector", {}) or {})

    def f(value: Any, default: float = 0.0) -> float:
        try:
            val = float(value)
        except Exception:
            return float(default)
        return val if math.isfinite(val) else float(default)

    # Threshold templates are diagnostic abstention policies. HPO should tune
    # the meta model for top-k ranking quality, not sparse post-hoc gating.
    return f(best.get("meta_ev_frontier_objective"), float("-inf"))


def _hpo_param_grid(seed: int, requested_trials: int) -> list[dict[str, Any]]:
    trials = [{**preset, "trial_source": "preset"} for preset in META_HPO_PRESETS]
    rng = np.random.default_rng(int(seed))
    while len(trials) < int(requested_trials):
        classifier = {
            "n_estimators": int(rng.integers(120, 361)),
            "learning_rate": float(np.exp(rng.uniform(np.log(0.015), np.log(0.080)))),
            "num_leaves": int(rng.choice([15, 23, 31, 47, 63])),
            "max_depth": int(rng.choice([-1, 4, 5, 6, 8])),
            "min_child_samples": int(rng.integers(25, 111)),
            "subsample": float(rng.uniform(0.65, 0.95)),
            "colsample_bytree": float(rng.uniform(0.55, 0.95)),
            "reg_alpha": float(np.exp(rng.uniform(np.log(1e-4), np.log(3.0)))),
            "reg_lambda": float(np.exp(rng.uniform(np.log(0.3), np.log(12.0)))),
        }
        regressor = dict(classifier)
        trials.append(
            {
                "name": f"random_{len(trials):03d}",
                "trial_source": "random",
                "classifier": classifier,
                "regressor": regressor,
            }
        )
    return trials[: int(requested_trials)]


def run_hpo_smoke(
    *,
    handoff_dir: Path,
    ledger_path: Path | None,
    out_dir: Path,
    frontier: str,
    seed: int,
    train_scope: str,
    enable_base_prior_features: bool,
    enable_reliability_features: bool,
    enable_support_drift_features: bool,
    enable_hit_surprise_features: bool,
    enable_path_order_heads: bool,
    enable_path_order_blends: bool,
    feature_selection_top_n: int,
    feature_selection_target: str,
    feature_selection_method: str,
    hpo_trials: int,
    hpo_max_train_rows: int,
    max_oos_model_age_days: int = 0,
    meta_head_mode: str = "multi",
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    trial_rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for trial_idx, profile in enumerate(_hpo_param_grid(seed, hpo_trials)):
        trial_name = str(profile["name"])
        trial_dir = out_dir / f"trial_{trial_idx:03d}_{trial_name}"
        manifest = run_smoke(
            handoff_dir=handoff_dir,
            ledger_path=ledger_path,
            out_dir=trial_dir,
            frontier=frontier,
            seed=int(seed) + 10_000 * trial_idx,
            train_scope=train_scope,
            enable_base_prior_features=enable_base_prior_features,
            enable_reliability_features=enable_reliability_features,
            enable_support_drift_features=enable_support_drift_features,
            enable_hit_surprise_features=enable_hit_surprise_features,
            enable_path_order_heads=enable_path_order_heads,
            enable_path_order_blends=enable_path_order_blends,
            feature_selection_top_n=feature_selection_top_n,
            feature_selection_target=feature_selection_target,
            feature_selection_method=feature_selection_method,
            max_oos_model_age_days=int(max_oos_model_age_days),
            validation_scope="largest",
            model_train_max_rows=int(hpo_max_train_rows),
            model_params={"classifier": profile["classifier"], "regressor": profile["regressor"]},
            model_profile_name=trial_name,
            meta_head_mode=str(meta_head_mode),
        )
        objective = _meta_trial_objective(manifest)
        best = dict(manifest.get("best_selector", {}) or {})
        best_threshold = dict(manifest.get("best_threshold_policy", {}) or {})
        trial_rows.append(
            {
                "trial_idx": int(trial_idx),
                "trial_name": trial_name,
                "trial_source": str(profile.get("trial_source", "")),
                "objective": float(objective),
                "trial_dir": str(trial_dir),
                "best_selector": best.get("selector"),
                "best_selector_meta_ev_frontier_objective": best.get("meta_ev_frontier_objective"),
                "best_keep020_ev_after_1pct": best.get("mean_keep020_ev_after_1pct"),
                "best_keep020_clean_exec_precision": best.get("mean_keep020_clean_exec_precision"),
                "best_keep020_dirty_positive_rate": best.get("mean_keep020_dirty_positive_rate"),
                "best_keep010_exec_margin": best.get("mean_keep010_exec_margin"),
                "best_keep010_ev_after_1pct": best.get("mean_keep010_ev_after_1pct"),
                "best_worst_keep010_ev_after_1pct": best.get("worst_keep010_ev_after_1pct"),
                "best_keep010_ret_net": best.get("mean_keep010_ret_net"),
                "best_keep010_clean_exec_precision": best.get("mean_keep010_clean_exec_precision"),
                "best_keep010_dirty_positive_rate": best.get("mean_keep010_dirty_positive_rate"),
                "best_keep010_first_touch_bad_mae": best.get("mean_keep010_first_touch_bad_mae"),
                "best_keep010_bad_mae": best.get("mean_keep010_full_path_bad_mae"),
                "best_keep010_timeout": best.get("mean_keep010_timeout"),
                "best_threshold_selector": best_threshold.get("selector"),
                "best_threshold_policy": best_threshold.get("policy_id"),
                "best_threshold_budget_frac": best_threshold.get("budget_frac"),
                "best_threshold_meta_ev_frontier_objective": best_threshold.get("meta_ev_frontier_objective"),
                "best_threshold_ev_after_1pct": best_threshold.get("mean_ev_after_1pct"),
                "best_threshold_exec_margin": best_threshold.get("mean_exec_margin"),
                "best_threshold_worst_exec_margin": best_threshold.get("worst_exec_margin"),
                "best_threshold_clean_exec_precision": best_threshold.get("mean_clean_exec_precision"),
                "best_threshold_dirty_positive_rate": best_threshold.get("mean_dirty_positive_rate"),
                "best_threshold_first_touch_bad_mae": best_threshold.get("mean_first_touch_bad_mae"),
                "best_threshold_bad_mae": best_threshold.get("mean_full_path_bad_mae"),
                "best_threshold_timeout": best_threshold.get("mean_timeout"),
                "selected_feature_union_count": manifest.get("selected_feature_union_count"),
                "selected_feature_intersection_count": manifest.get("selected_feature_intersection_count"),
                **{f"classifier_{k}": v for k, v in profile["classifier"].items()},
                **{f"regressor_{k}": v for k, v in profile["regressor"].items()},
            }
        )
        manifests.append(manifest)
    trials = pd.DataFrame(trial_rows).sort_values("objective", ascending=False).reset_index(drop=True)
    trials.insert(0, "rank", np.arange(1, len(trials) + 1, dtype=np.int32))
    trials_path = out_dir / "s52_train_meta_hpo_trials.csv"
    trials.to_csv(trials_path, index=False)
    best_trial = trials.iloc[0].to_dict() if not trials.empty else {}
    best_idx = int(best_trial.get("trial_idx", 0)) if best_trial else 0
    best_manifest = manifests[best_idx] if manifests else {}
    best_profile = _hpo_param_grid(seed, hpo_trials)[best_idx] if manifests else None
    final_manifest = {}
    if best_profile is not None:
        final_dir = out_dir / "best_full_oos"
        final_manifest = run_smoke(
            handoff_dir=handoff_dir,
            ledger_path=ledger_path,
            out_dir=final_dir,
            frontier=frontier,
            seed=int(seed) + 900_000 + best_idx,
            train_scope=train_scope,
            enable_base_prior_features=enable_base_prior_features,
            enable_reliability_features=enable_reliability_features,
            enable_support_drift_features=enable_support_drift_features,
            enable_hit_surprise_features=enable_hit_surprise_features,
            enable_path_order_heads=enable_path_order_heads,
            enable_path_order_blends=enable_path_order_blends,
            feature_selection_top_n=feature_selection_top_n,
            feature_selection_target=feature_selection_target,
            feature_selection_method=feature_selection_method,
            max_oos_model_age_days=int(max_oos_model_age_days),
            validation_scope="all",
            model_train_max_rows=0,
            model_params={"classifier": best_profile["classifier"], "regressor": best_profile["regressor"]},
            model_profile_name=f"best_full_oos_{best_profile['name']}",
            meta_head_mode=str(meta_head_mode),
        )
    best_params = {
        "trial": _json_safe(best_trial),
        "classifier_params": best_manifest.get("classifier_params", {}),
        "regressor_params": best_manifest.get("regressor_params", {}),
        "selected_feature_union": best_manifest.get("selected_feature_union", []),
        "selected_feature_intersection": best_manifest.get("selected_feature_intersection", []),
        "best_full_oos_manifest": final_manifest.get("output_dir"),
    }
    (out_dir / "s52_train_meta_hpo_best.json").write_text(json.dumps(_json_safe(best_params), indent=2), encoding="utf-8")
    manifest = {
        "generated_by": "run_s52_train_meta_regime_handoff_smoke_hpo",
        "handoff_dir": str(handoff_dir),
        "ledger_path": str(ledger_path) if ledger_path is not None else None,
        "frontier": str(frontier),
        "train_scope": str(train_scope),
        "hpo_trials": int(hpo_trials),
        "hpo_scope": "single_largest_train_fold",
        "hpo_sampling": "beginning_middle_end_time_spread_for_lgbm_pipeline_feature_selection_and_model_fit",
        "hpo_max_train_rows": int(hpo_max_train_rows),
        "best_full_oos_model_train_rows": "full_train_rows",
        "feature_selection_top_n": int(feature_selection_top_n),
        "feature_selection_target": str(feature_selection_target),
        "feature_selection_method": str(feature_selection_method),
        "max_oos_model_age_days": int(max_oos_model_age_days),
        "meta_head_mode": str(meta_head_mode),
        "trial_summary": str(trials_path),
        "best_params": str(out_dir / "s52_train_meta_hpo_best.json"),
        "best_trial": _json_safe(best_trial),
        "best_trial_manifest": best_manifest.get("output_dir"),
        "best_full_oos_manifest": final_manifest.get("output_dir"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-dir", type=Path, default=DEFAULT_HANDOFF_DIR)
    parser.add_argument("--ledger", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--frontier", choices=["top10", "top20", "top30"], default="top10")
    parser.add_argument("--train-scope", choices=["selected", "all"], default="selected")
    parser.add_argument("--enable-base-prior-features", action="store_true")
    parser.add_argument("--enable-reliability-features", action="store_true")
    parser.add_argument("--enable-support-drift-features", action="store_true")
    parser.add_argument("--enable-hit-surprise-features", action="store_true")
    parser.add_argument("--enable-path-order-heads", action="store_true")
    parser.add_argument("--enable-path-order-blends", action="store_true")
    parser.add_argument(
        "--meta-head-mode",
        choices=["multi", "single_base_soft_label"],
        default="multi",
        help="Use the legacy multi-head meta stack or one classifier trained on the base economic soft label.",
    )
    parser.add_argument(
        "--feature-selection-top-n",
        type=int,
        default=0,
        help="0 lets lgbm_pipeline.py pick the feature count automatically; >0 caps the selected list explicitly.",
    )
    parser.add_argument(
        "--feature-selection-target",
        choices=["ev_frontier", "clean_minus_bad", "exec_margin", "clean_exec"],
        default="ev_frontier",
    )
    parser.add_argument(
        "--feature-selection-method",
        choices=["auto", "lgbm_pipeline", "lgbm_staged"],
        default="auto",
        help="Canonical staged selector in lgbm_pipeline.py: univariate + ReliefF + redundancy clustering + iterative MDA pruning.",
    )
    parser.add_argument("--hpo-trials", type=int, default=0)
    parser.add_argument(
        "--hpo-max-train-rows",
        type=int,
        default=300000,
        help="Rows used to fit each meta model during HPO, sampled from beginning/middle/end of the largest calibration fold. Final full-OOS replay uses all prior rows.",
    )
    parser.add_argument(
        "--max-oos-model-age-days",
        type=int,
        default=0,
        help="Split each OOS month into expanding-window validation slices capped to this many days.",
    )
    parser.add_argument("--seed", type=int, default=20260705)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if int(args.hpo_trials) > 0:
        manifest = run_hpo_smoke(
            handoff_dir=args.handoff_dir,
            ledger_path=args.ledger,
            out_dir=args.out_dir,
            frontier=args.frontier,
            seed=int(args.seed),
            train_scope=str(args.train_scope),
            enable_base_prior_features=bool(args.enable_base_prior_features),
            enable_reliability_features=bool(args.enable_reliability_features),
            enable_support_drift_features=bool(args.enable_support_drift_features),
            enable_hit_surprise_features=bool(args.enable_hit_surprise_features),
            enable_path_order_heads=bool(args.enable_path_order_heads),
            enable_path_order_blends=bool(args.enable_path_order_blends),
            feature_selection_top_n=int(args.feature_selection_top_n),
            feature_selection_target=str(args.feature_selection_target),
            feature_selection_method=str(args.feature_selection_method),
            hpo_trials=int(args.hpo_trials),
            hpo_max_train_rows=int(args.hpo_max_train_rows),
            max_oos_model_age_days=int(args.max_oos_model_age_days),
            meta_head_mode=str(args.meta_head_mode),
        )
    else:
        manifest = run_smoke(
            handoff_dir=args.handoff_dir,
            ledger_path=args.ledger,
            out_dir=args.out_dir,
            frontier=args.frontier,
            seed=int(args.seed),
            train_scope=str(args.train_scope),
            enable_base_prior_features=bool(args.enable_base_prior_features),
            enable_reliability_features=bool(args.enable_reliability_features),
            enable_support_drift_features=bool(args.enable_support_drift_features),
            enable_hit_surprise_features=bool(args.enable_hit_surprise_features),
            enable_path_order_heads=bool(args.enable_path_order_heads),
            enable_path_order_blends=bool(args.enable_path_order_blends),
            feature_selection_top_n=int(args.feature_selection_top_n),
            feature_selection_target=str(args.feature_selection_target),
            feature_selection_method=str(args.feature_selection_method),
            max_oos_model_age_days=int(args.max_oos_model_age_days),
            model_train_max_rows=0,
            meta_head_mode=str(args.meta_head_mode),
        )
    print(json.dumps(_json_safe({"event": "s52_train_meta_handoff_smoke_done", **manifest}), sort_keys=True))


if __name__ == "__main__":
    main()
