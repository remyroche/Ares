#!/usr/bin/env python3
"""Report positive conditional-GMM feature pairs and AE/GMM archetypes."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.features_gmm_ae import (  # noqa: E402
    AE_GMM_FEATURE_COLUMNS,
    fit_ae_gmm_state,
    transform_ae_gmm_features,
)
from scripts.run_conditional_gmm_feature_selection import (  # noqa: E402
    build_side_aware_targets,
    _load_feature_store_columns,
    _load_labels,
)


DEFAULT_SELECTION_DIR = Path(
    "data_perp/reports/conditional_gmm_feature_selection_20260702_lowcost_strict_econ_target"
)
DEFAULT_LABELS_PATH = Path(
    "data_perp/artifacts/"
    "20260702_211500_single_head_monthly_walkforward_bidirectional_sideaware_"
    "lowcost_strict_economic_target_labels/labels"
)
DEFAULT_FEATURE_DIR = Path("data_perp/features/20260629_050000")

UPSIDE_TARGETS = {
    "side_adjusted_return",
    "utility",
    "risk_adjusted_utility",
    "favorable_excursion",
    "lower_tail_utility",
}
RISK_TARGETS = {"bad_MAE", "timeout", "adverse_excursion"}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def summarize_beneficial_pairs(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return beneficial selected pairs, feature rollup, and label/archetype counts."""
    if pairs.empty:
        empty = pd.DataFrame()
        return empty, empty, empty
    out = pairs.copy()
    numeric = [
        "pair_score",
        "global_spearman_ic",
        "mean_bucket_spearman_ic_shrunk",
        "long_spearman_ic",
        "short_spearman_ic",
        "long_short_ic_difference",
        "sign_flip_rate",
        "bucket_ic_std",
    ]
    for col in numeric:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = np.nan

    target = out["target"].astype(str)
    direction = np.where(target.isin(RISK_TARGETS), -1.0, 1.0)
    out["beneficial_direction"] = np.where(target.isin(RISK_TARGETS), "lower_is_better", "higher_is_better")
    out["good_direction_ic"] = direction * out["global_spearman_ic"]
    out["good_direction_bucket_ic"] = direction * out["mean_bucket_spearman_ic_shrunk"]
    out["beneficial_kind"] = np.where(
        target.isin(RISK_TARGETS),
        "risk_protective_negative_ic",
        "upside_or_tail_buffer_positive_ic",
    )
    beneficial = out[out["good_direction_ic"] > 0.0].copy()
    beneficial = beneficial.sort_values(
        ["pair_score", "good_direction_ic"],
        ascending=[False, False],
    ).reset_index(drop=True)

    if beneficial.empty:
        rollup = pd.DataFrame()
    else:
        rollup = (
            beneficial.groupby(["feature", "family"], dropna=False)
            .agg(
                best_pair_score=("pair_score", "max"),
                best_good_direction_ic=("good_direction_ic", "max"),
                pair_count=("target", "size"),
                beneficial_targets=("target", lambda s: ",".join(sorted(set(map(str, s))))),
                primary_categories=(
                    "primary_category",
                    lambda s: ",".join(sorted(set(map(str, s)))),
                ),
                beneficial_kinds=(
                    "beneficial_kind",
                    lambda s: ",".join(sorted(set(map(str, s)))),
                ),
            )
            .reset_index()
            .sort_values(
                ["best_pair_score", "best_good_direction_ic"],
                ascending=[False, False],
            )
        )

    counts = []
    for column in ("target", "primary_category", "family"):
        if column in out.columns:
            vc = out[column].astype(str).value_counts(dropna=False)
            counts.extend(
                {
                    "dimension": column,
                    "value": str(key),
                    "selected_pair_count": int(value),
                }
                for key, value in vc.items()
            )
    counts_df = pd.DataFrame(counts)
    return beneficial, rollup.reset_index(drop=True), counts_df


def _read_selected_features(selection_dir: Path) -> list[str]:
    candidates = [
        selection_dir / "conditional_gmm_training_feature_list.csv",
        selection_dir / "conditional_selected_features.csv",
    ]
    for path in candidates:
        if path.exists():
            frame = pd.read_csv(path)
            if "used_by_model" in frame.columns:
                used = frame["used_by_model"].astype(str).str.lower().isin({"1", "true", "yes", "y"})
                frame = frame[used].copy()
            if "selected_feature_position" in frame.columns:
                frame = frame.sort_values("selected_feature_position")
            if "feature" in frame.columns:
                return [str(v) for v in frame["feature"].dropna().drop_duplicates().tolist()]
    raise FileNotFoundError(f"No selected-feature CSV found under {selection_dir}")


def _load_selected_feature_frame(
    labels_path: Path,
    feature_dir: Path,
    selected_features: list[str],
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    labels = _load_labels(labels_path)
    to_load = [feature for feature in selected_features if feature not in labels.columns]
    matrix, feature_report = _load_feature_store_columns(
        labels,
        feature_dir=feature_dir,
        selected_features=to_load,
    )
    frame = pd.concat([labels, matrix], axis=1) if not matrix.empty else labels.copy()
    available = [feature for feature in selected_features if feature in frame.columns]
    return frame, available, feature_report


def _sample_frame(frame: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if int(max_rows) > 0 and len(frame) > int(max_rows):
        idx = np.linspace(0, len(frame) - 1, int(max_rows), dtype=int)
        return frame.iloc[idx].reset_index(drop=True)
    return frame.reset_index(drop=True)


def _normalise_side(work: pd.DataFrame) -> pd.Series:
    side_col = "side" if "side" in work.columns else "__side__" if "__side__" in work.columns else ""
    if not side_col:
        return pd.Series(1, index=work.index, dtype=np.int8)
    raw = pd.to_numeric(work[side_col], errors="coerce").fillna(1.0)
    return pd.Series(np.where(raw < 0.0, -1, 1), index=work.index, dtype=np.int8)


def _timestamp_month(work: pd.DataFrame) -> pd.Series:
    for col in ("timestamp", "__ts__", "ts", "entry_ts"):
        if col in work.columns:
            ts = pd.to_datetime(work[col], utc=True, errors="coerce")
            return pd.Series(ts.dt.to_period("M").astype(str), index=work.index)
    return pd.Series("unknown", index=work.index)


def _spread_bucket(work: pd.DataFrame) -> pd.Series:
    candidates = [
        "p75_spread_bps",
        "median_spread_bps",
        "spread_bps",
        "effective_spread_bps",
        "quoted_spread_bps",
        "cost_bps",
    ]
    for col in candidates:
        if col in work.columns:
            spread = pd.to_numeric(work[col], errors="coerce")
            if int(spread.notna().sum()) < 20 or float(spread.nunique(dropna=True)) < 2:
                continue
            try:
                bucket = pd.qcut(
                    spread.rank(method="first"),
                    q=min(4, int(spread.notna().sum())),
                    labels=["low_spread", "mid_low_spread", "mid_high_spread", "high_spread"][: min(4, int(spread.notna().sum()))],
                    duplicates="drop",
                )
                return pd.Series(bucket.astype(str), index=work.index).fillna("spread_unknown")
            except Exception:
                break
    return pd.Series("spread_unknown", index=work.index)


def _full_stop_loss_flag(work: pd.DataFrame) -> pd.Series:
    for col in ("full_stop_loss", "full_sl", "stop_loss", "__hit_sl__", "__is_stop__"):
        if col in work.columns:
            return pd.to_numeric(work[col], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=work.index, dtype=np.float32)


def _gmm_economic_targets(
    work: pd.DataFrame,
    targets: pd.DataFrame | None = None,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    if targets is None:
        targets, _target_report = build_side_aware_targets(work)
    utility = pd.to_numeric(work["__u_econ_net__"], errors="coerce")
    bad_mae = pd.to_numeric(targets["bad_MAE"], errors="coerce").fillna(0.0)
    timeout = pd.to_numeric(targets["timeout"], errors="coerce").fillna(0.0)
    full_stop = _full_stop_loss_flag(work)
    positive = utility > 0.0
    clean_positive = positive & bad_mae.lt(0.5) & timeout.lt(0.5) & full_stop.lt(0.5)
    dirty_positive = positive & (bad_mae.ge(0.5) | timeout.ge(0.5) | full_stop.ge(0.5))
    month_codes = pd.factorize(_timestamp_month(work).astype(str), sort=True)[0].astype(np.float32)
    economic_targets: dict[str, np.ndarray] = {
        "returns": utility.to_numpy(dtype=np.float32),
        "target": pd.to_numeric(work["__y_econ_soft__"], errors="coerce").to_numpy(dtype=np.float32),
        "bad_mae_1r": bad_mae.to_numpy(dtype=np.float32),
        "timeout": timeout.to_numpy(dtype=np.float32),
        "adverse_excursion": pd.to_numeric(targets["adverse_excursion"], errors="coerce").to_numpy(dtype=np.float32),
        "favorable_excursion": pd.to_numeric(targets["favorable_excursion"], errors="coerce").to_numpy(dtype=np.float32),
        "lower_tail_utility": pd.to_numeric(targets["lower_tail_utility"], errors="coerce").to_numpy(dtype=np.float32),
        "full_stop_loss": full_stop.to_numpy(dtype=np.float32),
        "clean_positive": clean_positive.astype(np.float32).to_numpy(),
        "dirty_positive": dirty_positive.astype(np.float32).to_numpy(),
        "time_bucket": month_codes,
    }
    side = _normalise_side(work)
    economic_targets["side"] = side.to_numpy(dtype=np.float32)
    return economic_targets, targets


def _feature_theme(features: list[str]) -> str:
    joined = " ".join(str(v).lower() for v in features)
    if any(token in joined for token in ("spread", "liquid", "slippage", "depth")):
        return "liquidity_stress"
    if any(token in joined for token in ("oi", "open_interest", "funding")):
        return "open_interest_pressure"
    if any(token in joined for token in ("shock", "range", "vol", "atr", "compression")):
        return "volatility_range"
    if any(token in joined for token in ("dist", "pullback", "vwap", "loc_", "zscore")):
        return "reversion_location"
    if any(token in joined for token in ("adx", "trend", "ema", "breakout", "slope")):
        return "trend_expansion"
    return "mixed_path_state"


def _archetype_label(row: pd.Series, baseline: dict[str, float], top_features: list[str]) -> str:
    theme = _feature_theme(top_features)
    u = _safe_float(row.get("u_econ_net_mean"))
    hit = _safe_float(row.get("u_econ_hit"))
    bad = _safe_float(row.get("bad_MAE_mean"))
    timeout = _safe_float(row.get("timeout_mean"))
    if u >= baseline["u_econ_net_mean"] + 0.0005 and hit >= baseline["u_econ_hit"]:
        prefix = "positive"
    elif bad >= baseline["bad_MAE_mean"] + 0.04 or timeout >= baseline["timeout_mean"] + 0.03:
        prefix = "risk"
    elif u <= baseline["u_econ_net_mean"] - 0.0005:
        prefix = "negative"
    else:
        prefix = "neutral"
    return f"{prefix}_{theme}"


def _cluster_metric_row(
    *,
    group_name: str,
    group_value: str,
    cluster: int,
    mask: pd.Series | np.ndarray,
    total: int,
    utility: pd.Series,
    targets: pd.DataFrame,
    side: pd.Series,
    full_stop: pd.Series,
) -> dict[str, Any]:
    mask_arr = np.asarray(mask, dtype=bool)
    local_u = utility.loc[mask_arr]
    bad = pd.to_numeric(targets.loc[mask_arr, "bad_MAE"], errors="coerce").fillna(0.0)
    timeout = pd.to_numeric(targets.loc[mask_arr, "timeout"], errors="coerce").fillna(0.0)
    stop = pd.to_numeric(full_stop.loc[mask_arr], errors="coerce").fillna(0.0)
    positive = local_u > 0.0
    clean_positive = positive & bad.lt(0.5) & timeout.lt(0.5) & stop.lt(0.5)
    dirty_positive = positive & (bad.ge(0.5) | timeout.ge(0.5) | stop.ge(0.5))
    return {
        "slice": str(group_name),
        "slice_value": str(group_value),
        "cluster": int(cluster),
        "rows": int(mask_arr.sum()),
        "share_in_slice": float(mask_arr.sum() / max(total, 1)),
        "u_econ_net_mean": float(local_u.mean()) if len(local_u) else float("nan"),
        "u_econ_net_q10": float(local_u.quantile(0.10)) if len(local_u) else float("nan"),
        "u_econ_hit": float(positive.mean()) if len(local_u) else float("nan"),
        "clean_positive_rate": float(clean_positive.mean()) if len(local_u) else float("nan"),
        "dirty_positive_rate": float(dirty_positive.mean()) if len(local_u) else float("nan"),
        "bad_MAE_mean": float(bad.mean()) if len(bad) else float("nan"),
        "timeout_mean": float(timeout.mean()) if len(timeout) else float("nan"),
        "full_stop_loss_mean": float(stop.mean()) if len(stop) else float("nan"),
        "side_short_share": float((side.loc[mask_arr] < 0).mean()) if int(mask_arr.sum()) else float("nan"),
        "adverse_excursion_mean": float(pd.to_numeric(targets.loc[mask_arr, "adverse_excursion"], errors="coerce").mean()),
        "favorable_excursion_mean": float(pd.to_numeric(targets.loc[mask_arr, "favorable_excursion"], errors="coerce").mean()),
        "lower_tail_utility_mean": float(pd.to_numeric(targets.loc[mask_arr, "lower_tail_utility"], errors="coerce").mean()),
    }


def _slice_cluster_metrics(
    work: pd.DataFrame,
    targets: pd.DataFrame,
    clusters: np.ndarray,
) -> pd.DataFrame:
    utility = pd.to_numeric(work["__u_econ_net__"], errors="coerce")
    side = _normalise_side(work)
    full_stop = _full_stop_loss_flag(work)
    cluster_series = pd.Series(np.asarray(clusters, dtype=np.int32), index=work.index)
    slices = {
        "side": pd.Series(np.where(side < 0, "short", "long"), index=work.index),
        "month": _timestamp_month(work),
        "spread_bucket": _spread_bucket(work),
    }
    rows: list[dict[str, Any]] = []
    for slice_name, slice_values in slices.items():
        for value, value_idx in slice_values.groupby(slice_values, dropna=False).groups.items():
            value_mask = work.index.isin(value_idx)
            total = int(np.sum(value_mask))
            if total <= 0:
                continue
            for cluster in sorted(set(int(v) for v in cluster_series.loc[value_mask].dropna().tolist())):
                mask = value_mask & cluster_series.eq(cluster).to_numpy()
                if int(np.sum(mask)) <= 0:
                    continue
                rows.append(
                    _cluster_metric_row(
                        group_name=slice_name,
                        group_value=str(value),
                        cluster=int(cluster),
                        mask=mask,
                        total=total,
                        utility=utility,
                        targets=targets,
                        side=side,
                        full_stop=full_stop,
                    )
                )
    return pd.DataFrame(rows)


def _summarize_side_specific_clusters(
    side_name: str,
    work: pd.DataFrame,
    x: pd.DataFrame,
    targets: pd.DataFrame,
    state: dict[str, Any],
) -> pd.DataFrame:
    transformed = transform_ae_gmm_features(x, state)
    if "gmm_cluster_id" not in transformed.columns:
        return pd.DataFrame()
    clusters = pd.to_numeric(transformed["gmm_cluster_id"], errors="coerce").fillna(0).astype(np.int32).to_numpy()
    metrics = _slice_cluster_metrics(work, targets, clusters)
    if metrics.empty:
        return pd.DataFrame()
    side_rows = metrics[metrics["slice"].eq("side")].copy()
    if side_rows.empty:
        side_rows = metrics.copy()
    side_rows.insert(0, "side_fit", str(side_name))
    selected = dict(state.get("selected_config", {}) or {})
    side_rows["selected_n_components"] = int(state.get("gmm_n_components", 0) or 0)
    side_rows["selected_path_cleanliness_score"] = _safe_float(selected.get("path_cleanliness_score"))
    side_rows["selected_final_score"] = _safe_float(selected.get("final_score"))
    return side_rows.reset_index(drop=True)


def build_cluster_archetypes(
    labels_path: Path,
    feature_dir: Path,
    selected_features: list[str],
    *,
    max_rows: int = 8000,
    random_state: int = 913,
    max_train_rows: int = 4000,
    ae_max_iter: int = 8,
    cluster_candidates: str | None = None,
    reg_covar_candidates: str | None = None,
    smooth_lambda_candidates: str | None = None,
    require_both_sides: bool = True,
    min_side_cluster_frac: float = 0.02,
    min_side_cluster_rows: int = 10,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame, available, feature_report = _load_selected_feature_frame(labels_path, feature_dir, selected_features)
    work = _sample_frame(frame, max_rows=max_rows)
    x = (
        work.reindex(columns=available)
        .apply(pd.to_numeric, errors="coerce")
        .astype(np.float32)
    )
    economic_targets, targets = _gmm_economic_targets(work)
    side_col = "side" if "side" in work.columns else "__side__" if "__side__" in work.columns else ""
    state = fit_ae_gmm_state(
        x,
        economic_targets=economic_targets,
        random_state=int(random_state),
        max_train_rows=int(max_train_rows),
        ae_max_iter=int(ae_max_iter),
        cluster_candidates=cluster_candidates,
        reg_covar_candidates=reg_covar_candidates,
        smooth_lambda_candidates=smooth_lambda_candidates,
        require_both_sides=bool(require_both_sides and side_col),
        min_side_cluster_frac=float(min_side_cluster_frac),
        min_side_cluster_rows=int(min_side_cluster_rows),
    )
    transformed = transform_ae_gmm_features(x, state)
    n_components = int(state.get("gmm_n_components", 0) or 0)
    prob_cols = [f"gmm_prob_{i}" for i in range(n_components) if f"gmm_prob_{i}" in transformed.columns]
    if not prob_cols:
        return pd.DataFrame(), {
            "state_enabled": bool(state.get("enabled", False)),
            "reason": str(state.get("reason", "no_probability_columns")),
            "selected_feature_count": int(len(selected_features)),
            "available_feature_count": int(len(available)),
            "feature_store": feature_report,
        }
    clusters = transformed[prob_cols].to_numpy(dtype=np.float32).argmax(axis=1)
    raw_z = x.copy()
    raw_z = (raw_z - raw_z.mean()) / raw_z.std(ddof=0).replace(0.0, np.nan)
    side = _normalise_side(work)
    utility = pd.to_numeric(work["__u_econ_net__"], errors="coerce")
    full_stop = _full_stop_loss_flag(work)
    clean_positive = utility.gt(0.0) & targets["bad_MAE"].lt(0.5) & targets["timeout"].lt(0.5) & full_stop.lt(0.5)
    dirty_positive = utility.gt(0.0) & (
        targets["bad_MAE"].ge(0.5) | targets["timeout"].ge(0.5) | full_stop.ge(0.5)
    )
    baseline = {
        "u_econ_net_mean": float(utility.mean()),
        "u_econ_hit": float((utility > 0.0).mean()),
        "bad_MAE_mean": float(targets["bad_MAE"].mean()),
        "timeout_mean": float(targets["timeout"].mean()),
        "clean_positive_rate": float(clean_positive.mean()),
        "dirty_positive_rate": float(dirty_positive.mean()),
    }
    rows: list[dict[str, Any]] = []
    for cluster in sorted(set(int(v) for v in clusters)):
        mask = clusters == cluster
        local_bad = targets.loc[mask, "bad_MAE"]
        local_timeout = targets.loc[mask, "timeout"]
        local_stop = full_stop.loc[mask]
        local_positive = utility.loc[mask].gt(0.0)
        local_clean = local_positive & local_bad.lt(0.5) & local_timeout.lt(0.5) & local_stop.lt(0.5)
        local_dirty = local_positive & (local_bad.ge(0.5) | local_timeout.ge(0.5) | local_stop.ge(0.5))
        zmeans = raw_z.loc[mask].mean().sort_values(key=lambda s: s.abs(), ascending=False).head(8)
        top_features = [str(feature) for feature in zmeans.index]
        row = {
            "cluster": int(cluster),
            "rows": int(mask.sum()),
            "share": float(mask.mean()),
            "u_econ_net_mean": float(utility.loc[mask].mean()),
            "u_econ_net_q10": float(utility.loc[mask].quantile(0.10)),
            "u_econ_hit": float((utility.loc[mask] > 0.0).mean()),
            "y_econ_soft_mean": float(pd.to_numeric(work.loc[mask, "__y_econ_soft__"], errors="coerce").mean()),
            "side_short_share": float((side.loc[mask] < 0.0).mean()),
            "clean_positive_rate": float(local_clean.mean()),
            "dirty_positive_rate": float(local_dirty.mean()),
            "bad_MAE_mean": float(targets.loc[mask, "bad_MAE"].mean()),
            "timeout_mean": float(targets.loc[mask, "timeout"].mean()),
            "full_stop_loss_mean": float(local_stop.mean()),
            "adverse_excursion_mean": float(targets.loc[mask, "adverse_excursion"].mean()),
            "favorable_excursion_mean": float(targets.loc[mask, "favorable_excursion"].mean()),
            "lower_tail_utility_mean": float(targets.loc[mask, "lower_tail_utility"].mean()),
            "top_feature_z_deviations": "; ".join(
                f"{feature}:{value:+.2f}" for feature, value in zmeans.items()
            ),
        }
        row["archetype_label"] = _archetype_label(pd.Series(row), baseline, top_features)
        rows.append(row)
    clusters_df = pd.DataFrame(rows).sort_values("u_econ_net_mean", ascending=False).reset_index(drop=True)
    slice_metrics = _slice_cluster_metrics(work, targets, clusters)
    side_specific_rows: list[pd.DataFrame] = []
    for side_value, side_name in ((1, "long"), (-1, "short")):
        side_mask = side.eq(side_value).to_numpy()
        if int(np.sum(side_mask)) < max(100, int(max_train_rows) // 20):
            continue
        side_work = work.loc[side_mask].reset_index(drop=True)
        side_x = x.loc[side_mask].reset_index(drop=True)
        side_targets_frame = targets.loc[side_mask].reset_index(drop=True)
        side_econ_targets, side_targets_frame = _gmm_economic_targets(side_work, side_targets_frame)
        side_state = fit_ae_gmm_state(
            side_x,
            economic_targets=side_econ_targets,
            random_state=int(random_state + (101 if side_value > 0 else 202)),
            max_train_rows=int(max_train_rows),
            ae_max_iter=int(ae_max_iter),
            cluster_candidates=cluster_candidates,
            reg_covar_candidates=reg_covar_candidates,
            smooth_lambda_candidates=smooth_lambda_candidates,
            require_both_sides=False,
            min_side_cluster_frac=0.0,
            min_side_cluster_rows=0,
        )
        side_specific = _summarize_side_specific_clusters(
            side_name,
            side_work,
            side_x,
            side_targets_frame,
            side_state,
        )
        if not side_specific.empty:
            side_specific_rows.append(side_specific)
    side_specific_df = (
        pd.concat(side_specific_rows, axis=0, ignore_index=True)
        if side_specific_rows
        else pd.DataFrame()
    )
    report = {
        "state_enabled": bool(state.get("enabled", False)),
        "reason": str(state.get("reason", "")),
        "selected_config": state.get("selected_config", {}),
        "sample_rows": int(len(work)),
        "selected_feature_count": int(len(selected_features)),
        "available_feature_count": int(len(available)),
        "missing_selected_features": [feature for feature in selected_features if feature not in available],
        "output_feature_count": int(transformed.shape[1]),
        "expected_feature_count": int(len(AE_GMM_FEATURE_COLUMNS)),
        "columns_match_contract": list(transformed.columns) == list(AE_GMM_FEATURE_COLUMNS),
        "finite_output_frac": float(np.isfinite(transformed.to_numpy(dtype=np.float32)).mean()) if transformed.size else 0.0,
        "feature_store": feature_report,
        "hpo_grid": state.get("hpo_grid", {}),
        "hpo_report_count": int(state.get("hpo_report_count", len(state.get("hpo_reports", [])))),
        "hpo_reports": state.get("hpo_reports", state.get("top_configs", [])),
        "baseline": baseline,
        "slice_metrics": slice_metrics.to_dict(orient="records"),
        "side_specific_archetypes": side_specific_df.to_dict(orient="records"),
    }
    return clusters_df, report


def _write_markdown(
    path: Path,
    *,
    beneficial: pd.DataFrame,
    rollup: pd.DataFrame,
    counts: pd.DataFrame,
    clusters: pd.DataFrame,
    slice_metrics: pd.DataFrame,
    side_specific: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    def table(df: pd.DataFrame, columns: list[str], limit: int = 20) -> str:
        if df.empty:
            return "No rows."
        return df.loc[:, [c for c in columns if c in df.columns]].head(limit).to_markdown(index=False)

    target_counts = counts[counts["dimension"].eq("target")] if not counts.empty else pd.DataFrame()
    category_counts = counts[counts["dimension"].eq("primary_category")] if not counts.empty else pd.DataFrame()
    lines = [
        "# Conditional GMM Archetype Report",
        "",
        f"Selection dir: `{manifest['selection_dir']}`",
        f"Labels path: `{manifest['labels_path']}`",
        f"Promotion status: `{manifest.get('target_readiness', {}).get('promotion_status', 'unknown')}`",
        "",
        "## Target Labels",
        "",
        table(target_counts, ["value", "selected_pair_count"], limit=20),
        "",
        "## Selection Archetypes",
        "",
        table(category_counts, ["value", "selected_pair_count"], limit=20),
        "",
        "## Positive Feature Rollup",
        "",
        table(
            rollup,
            [
                "feature",
                "family",
                "best_pair_score",
                "best_good_direction_ic",
                "beneficial_targets",
                "primary_categories",
            ],
            limit=30,
        ),
        "",
        "## Top Beneficial Pairs",
        "",
        table(
            beneficial,
            [
                "feature",
                "target",
                "family",
                "primary_category",
                "beneficial_kind",
                "pair_score",
                "global_spearman_ic",
                "good_direction_ic",
            ],
            limit=40,
        ),
        "",
        "## AE/GMM Cluster Archetypes",
        "",
        table(
            clusters,
            [
                "cluster",
                "archetype_label",
                "share",
                "u_econ_net_mean",
                "u_econ_hit",
                "clean_positive_rate",
                "dirty_positive_rate",
                "bad_MAE_mean",
                "timeout_mean",
                "top_feature_z_deviations",
            ],
            limit=20,
        ),
        "",
        "## AE/GMM Cluster Metrics By Side/Month/Spread",
        "",
        table(
            slice_metrics,
            [
                "slice",
                "slice_value",
                "cluster",
                "rows",
                "share_in_slice",
                "u_econ_net_mean",
                "clean_positive_rate",
                "dirty_positive_rate",
                "bad_MAE_mean",
                "timeout_mean",
            ],
            limit=60,
        ),
        "",
        "## Side-Specific AE/GMM Fits",
        "",
        table(
            side_specific,
            [
                "side_fit",
                "cluster",
                "rows",
                "share_in_slice",
                "u_econ_net_mean",
                "clean_positive_rate",
                "dirty_positive_rate",
                "bad_MAE_mean",
                "timeout_mean",
                "selected_path_cleanliness_score",
            ],
            limit=40,
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_report(
    *,
    selection_dir: Path,
    labels_path: Path,
    feature_dir: Path,
    output_dir: Path | None = None,
    max_rows: int = 8000,
    random_state: int = 913,
    max_train_rows: int = 4000,
    ae_max_iter: int = 8,
    cluster_candidates: str | None = None,
    reg_covar_candidates: str | None = None,
    smooth_lambda_candidates: str | None = None,
    require_both_sides: bool = True,
    min_side_cluster_frac: float = 0.02,
    min_side_cluster_rows: int = 10,
) -> dict[str, Any]:
    selection_dir = selection_dir.resolve()
    output_dir = (output_dir or selection_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = selection_dir / "manifest.json"
    selection_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    pairs = pd.read_csv(selection_dir / "conditional_selected_feature_target_pairs.csv")
    selected_features = _read_selected_features(selection_dir)
    beneficial, rollup, counts = summarize_beneficial_pairs(pairs)
    clusters, cluster_report = build_cluster_archetypes(
        labels_path=labels_path,
        feature_dir=feature_dir,
        selected_features=selected_features,
        max_rows=max_rows,
        random_state=random_state,
        max_train_rows=max_train_rows,
        ae_max_iter=ae_max_iter,
        cluster_candidates=cluster_candidates,
        reg_covar_candidates=reg_covar_candidates,
        smooth_lambda_candidates=smooth_lambda_candidates,
        require_both_sides=require_both_sides,
        min_side_cluster_frac=min_side_cluster_frac,
        min_side_cluster_rows=min_side_cluster_rows,
    )
    paths = {
        "beneficial_pairs": output_dir / "conditional_gmm_positive_selected_pairs.csv",
        "beneficial_feature_rollup": output_dir / "conditional_gmm_positive_feature_rollup.csv",
        "target_archetype_counts": output_dir / "conditional_gmm_target_archetype_counts.csv",
        "cluster_archetypes": output_dir / "conditional_gmm_cluster_archetypes.csv",
        "cluster_slice_metrics": output_dir / "conditional_gmm_cluster_slice_metrics.csv",
        "side_specific_archetypes": output_dir / "conditional_gmm_side_specific_cluster_archetypes.csv",
        "hpo_configs": output_dir / "conditional_gmm_hpo_configs.csv",
        "manifest": output_dir / "conditional_gmm_archetype_report_manifest.json",
        "markdown": output_dir / "conditional_gmm_archetype_report.md",
    }
    slice_metrics = pd.DataFrame(cluster_report.get("slice_metrics", []))
    side_specific = pd.DataFrame(cluster_report.get("side_specific_archetypes", []))
    beneficial.to_csv(paths["beneficial_pairs"], index=False)
    rollup.to_csv(paths["beneficial_feature_rollup"], index=False)
    counts.to_csv(paths["target_archetype_counts"], index=False)
    clusters.to_csv(paths["cluster_archetypes"], index=False)
    slice_metrics.to_csv(paths["cluster_slice_metrics"], index=False)
    side_specific.to_csv(paths["side_specific_archetypes"], index=False)
    pd.DataFrame(cluster_report.get("hpo_reports", [])).to_csv(paths["hpo_configs"], index=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": "conditional_gmm_archetype_report_v1",
        "selection_dir": str(selection_dir),
        "labels_path": str(labels_path),
        "feature_dir": str(feature_dir),
        "output_dir": str(output_dir),
        "target_readiness": selection_manifest.get("target_readiness", {}),
        "counts": {
            "selected_pairs": int(len(pairs)),
            "beneficial_pairs": int(len(beneficial)),
            "beneficial_features": int(len(rollup)),
            "cluster_archetypes": int(len(clusters)),
            "cluster_slice_metric_rows": int(len(slice_metrics)),
            "side_specific_archetype_rows": int(len(side_specific)),
        },
        "cluster_report": cluster_report,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    _write_markdown(
        paths["markdown"],
        beneficial=beneficial,
        rollup=rollup,
        counts=counts,
        clusters=clusters,
        slice_metrics=slice_metrics,
        side_specific=side_specific,
        manifest=manifest,
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-dir", type=Path, default=DEFAULT_SELECTION_DIR)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=8000)
    parser.add_argument("--random-state", type=int, default=913)
    parser.add_argument("--max-train-rows", type=int, default=4000)
    parser.add_argument("--ae-max-iter", type=int, default=8)
    parser.add_argument("--cluster-candidates", default="2,3,4,5,6")
    parser.add_argument("--reg-covar-candidates", default="1e-5,3e-5,1e-4,3e-4,1e-3,3e-3")
    parser.add_argument("--smooth-lambda-candidates", default="0.5,0.8,0.925,0.97")
    parser.add_argument("--allow-single-side-clusters", action="store_true")
    parser.add_argument("--min-side-cluster-frac", type=float, default=0.02)
    parser.add_argument("--min-side-cluster-rows", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_report(
        selection_dir=args.selection_dir,
        labels_path=args.labels_path,
        feature_dir=args.feature_dir,
        output_dir=args.output_dir,
        max_rows=int(args.max_rows),
        random_state=int(args.random_state),
        max_train_rows=int(args.max_train_rows),
        ae_max_iter=int(args.ae_max_iter),
        cluster_candidates=args.cluster_candidates,
        reg_covar_candidates=args.reg_covar_candidates,
        smooth_lambda_candidates=args.smooth_lambda_candidates,
        require_both_sides=not bool(args.allow_single_side_clusters),
        min_side_cluster_frac=float(args.min_side_cluster_frac),
        min_side_cluster_rows=int(args.min_side_cluster_rows),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
