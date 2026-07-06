#!/usr/bin/env python3
"""Source/event feature gap for Stage 71 path-safe recoverability.

This is a diagnostic-only report. It reconstructs selected rows for the
Stage 71 gated-oracle recoverability specs, then compares June clean standalone
rows with June positive-dirty rows selected by Apr-May fit-stable specs.

It does not train a model, tune a production threshold, or promote a label.
Because the clean standalone rows are identified using June, this report is
post-holdout attribution only.
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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_label_matched_clean_dirty_feature_gap import (  # noqa: E402
    _auc_clean_high,
    _bucket_key,
    _feature_family,
    _rank_within_bucket,
    _table,
)
from scripts.run_label_first_touch_execution_proxy_ablation import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_HOLDOUT_MONTH,
    _first_touch_metrics,
    _safe_mean,
    _safe_numeric,
    _safe_quantile,
)
from scripts.run_label_first_touch_soft_recipe_proxy_ablation import (  # noqa: E402
    _global_bad_soft,
    _timestamp_top_k_indices,
)
from scripts.run_label_gated_oracle_recoverability_proxy import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_STAGE_DIR,
    _apply_bad_and_floor,
    _dirty_target,
    _json_safe,
    _proxy_score,
    _run_entry_score,
)
from scripts.run_label_quality_proxy_diagnostics import _feature_columns  # noqa: E402
from scripts.run_label_two_head_abstention_utility_proxy import _utility_targets  # noqa: E402
from scripts.run_soft_label_economic_proxy_ablation import (  # noqa: E402
    DEFAULT_EVENT_FEATURE_STORE_FEATURES,
    DEFAULT_PRIOR_WINDOWS_DAYS,
    DEFAULT_STATE_PATH_PRIOR_FEATURES,
)
from scripts.diagnose_label_matched_clean_dirty_feature_gap import (  # noqa: E402
    DEFAULT_LABELS_PATH,
    _build_frame,
)
from scripts.run_label_first_touch_execution_proxy_ablation import (  # noqa: E402
    _target_components as _first_touch_target_components,
)


DEFAULT_STAGE71_DIR = Path(
    "data_perp/reports/label_gated_oracle_recoverability_proxy_stage71_pathsafe_lower_floors_v1"
)
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/label_gated_oracle_pathsafe_source_gap_stage72_v1"
)
DEFAULT_MATCH_MODES = ("day_side", "regime_side", "timestamp_side")


def _parse_csv(value: str | None, default: tuple[str, ...]) -> list[str]:
    if value is None or not str(value).strip():
        return list(default)
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _load_stage_specs(
    stage_dir: Path,
    *,
    max_clean_specs: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected_path = stage_dir / "gated_oracle_recoverability_selected_by_fit.csv"
    fit_path = stage_dir / "gated_oracle_recoverability_fit_holdout.csv"
    if not selected_path.exists():
        raise FileNotFoundError(selected_path)
    if not fit_path.exists():
        raise FileNotFoundError(fit_path)
    selected = pd.read_csv(selected_path)
    fit = pd.read_csv(fit_path)

    dirty_specs = selected.copy()
    dirty_specs["spec_role"] = "fit_selected_dirty_holdout"

    clean_specs = fit[
        ~fit["selector"].astype(str).eq("oracle_ceiling")
        & fit["holdout_bounded_standalone_pass"].astype(bool)
        & ~fit["fit_bounded_pass"].astype(bool)
    ].copy()
    clean_specs = clean_specs.sort_values(
        ["holdout_objective", "holdout_mean_month_u", "holdout_clean_exec_actual_rate"],
        ascending=[False, False, False],
    ).head(int(max_clean_specs))
    clean_specs["spec_role"] = "june_clean_standalone"

    keep_cols = [
        "spec_role",
        "selector",
        "utility_target",
        "oracle_gate",
        "proxy_method",
        "proxy_top_k",
        "bad_threshold",
        "score_floor",
        "run_entry_gap_hours",
        "top_k",
    ]
    specs = pd.concat([dirty_specs[keep_cols], clean_specs[keep_cols]], ignore_index=True)
    specs = specs.drop_duplicates(keep_cols, keep="first").reset_index(drop=True)
    return specs, {
        "selected_by_fit_rows": int(len(selected)),
        "clean_standalone_candidate_rows": int(len(clean_specs)),
        "spec_rows": int(len(specs)),
        "stage_selected_by_fit": str(selected_path),
        "stage_fit_holdout": str(fit_path),
    }


def _selector_scores(
    *,
    selector: str,
    soft_proxy: pd.Series,
    cleanft_proxy: pd.Series,
    early_adverse_proxy: pd.Series,
    slow_timeout_proxy: pd.Series,
    path_dirty_proxy: pd.Series,
) -> pd.Series:
    cleanft = _safe_numeric(cleanft_proxy).reset_index(drop=True).fillna(0.5).clip(0.0, 1.0)
    low_early = (1.0 - _safe_numeric(early_adverse_proxy).reset_index(drop=True).fillna(0.5)).clip(0.0, 1.0)
    low_slow = (1.0 - _safe_numeric(slow_timeout_proxy).reset_index(drop=True).fillna(0.5)).clip(0.0, 1.0)
    low_dirty = (1.0 - _safe_numeric(path_dirty_proxy).reset_index(drop=True).fillna(0.5)).clip(0.0, 1.0)
    soft = _safe_numeric(soft_proxy).reset_index(drop=True)
    path_safe = ((cleanft + low_early + low_slow + low_dirty) / 4.0).clip(0.0, 1.0)
    if selector == "soft_proxy_bad_gate":
        return soft
    if selector == "soft_cleanft_blend_bad_gate":
        return 0.50 * soft + 0.50 * cleanft
    if selector == "soft_pathsafe_blend_bad_gate":
        return 0.55 * soft + 0.45 * path_safe
    if selector == "soft_low_adverse_blend_bad_gate":
        return 0.65 * soft + 0.35 * low_early
    if selector == "soft_low_dirty_blend_bad_gate":
        return 0.65 * soft + 0.35 * low_dirty
    if selector == "cleanft_low_dirty_blend_bad_gate":
        return 0.50 * cleanft + 0.50 * low_dirty
    raise ValueError(f"Unsupported selector for source-gap diagnostic: {selector}")


def _metric_cols(frame: pd.DataFrame, ft: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
    out = frame.iloc[idx][["__ts__", "__symbol__"]].reset_index(drop=True).copy()
    for col in ("side", "side_name", "__side__", "timeframe", "candidate_id", "primary_source_tag"):
        if col in frame.columns:
            out[col] = frame.iloc[idx][col].to_numpy()
    if "side" not in out.columns and "side" in ft.columns:
        out["side"] = ft["side"].iloc[idx].to_numpy()
    if "side_name" not in out.columns and "side" in out.columns:
        out["side_name"] = np.where(_safe_numeric(out["side"]) < 0.0, "short", "long")
    metric_cols = [
        "u_policy_net",
        "ret_net",
        "barrier",
        "clean_exec_actual",
        "first_touch_hit",
        "first_touch_stop",
        "first_touch_timeout",
        "first_touch_same_bar",
        "first_touch_bar",
        "first_touch_mae_to_sl",
        "first_touch_mfe_to_tp",
    ]
    for col in metric_cols:
        if col in ft.columns:
            out[col] = ft[col].iloc[idx].to_numpy()
    return out


def _reconstruct_selected_rows(
    *,
    frame: pd.DataFrame,
    ft: pd.DataFrame,
    specs: pd.DataFrame,
    holdout_month: str,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    month_series = frame["__ts__"].dt.to_period("M").astype(str)
    train_mask = month_series < str(holdout_month)
    valid_mask = month_series == str(holdout_month)
    train = frame.loc[train_mask].copy()
    valid = frame.loc[valid_mask].copy()
    valid_frame = valid.reset_index(drop=True)
    valid_ft = ft.loc[valid_mask].copy().reset_index(drop=True)

    bad_soft = _global_bad_soft(ft)
    train_bad_soft = bad_soft.loc[train_mask]
    components = _first_touch_target_components(ft)
    utility_map = _utility_targets(frame, ft)

    diagnostics: list[dict[str, Any]] = []
    ledgers: list[pd.DataFrame] = []
    proxy_cache: dict[tuple[str, int], dict[str, Any]] = {}

    for (proxy_method, proxy_top_k), group in specs.groupby(["proxy_method", "proxy_top_k"], dropna=False):
        cache_key = (str(proxy_method), int(proxy_top_k))
        if cache_key not in proxy_cache:
            bad_proxy, bad_diag = _proxy_score(
                train=train,
                valid=valid,
                features=feature_cols,
                target_train=train_bad_soft,
                top_k=int(proxy_top_k),
                method=str(proxy_method),
                tail_frac=0.05,
            )
            cleanft_proxy, cleanft_diag = _proxy_score(
                train=train,
                valid=valid,
                features=feature_cols,
                target_train=components["clean_first_touch"].loc[train_mask],
                top_k=int(proxy_top_k),
                method=str(proxy_method),
                tail_frac=0.05,
            )
            early_adverse_proxy, early_diag = _proxy_score(
                train=train,
                valid=valid,
                features=feature_cols,
                target_train=components["early_adverse"].loc[train_mask],
                top_k=int(proxy_top_k),
                method=str(proxy_method),
                tail_frac=0.05,
            )
            slow_timeout_proxy, slow_diag = _proxy_score(
                train=train,
                valid=valid,
                features=feature_cols,
                target_train=components["slow_timeout"].loc[train_mask],
                top_k=int(proxy_top_k),
                method=str(proxy_method),
                tail_frac=0.05,
            )
            path_dirty_proxy, dirty_diag = _proxy_score(
                train=train,
                valid=valid,
                features=feature_cols,
                target_train=components["dirty"].loc[train_mask],
                top_k=int(proxy_top_k),
                method=str(proxy_method),
                tail_frac=0.05,
            )
            proxy_cache[cache_key] = {
                "bad_proxy": _safe_numeric(bad_proxy).reset_index(drop=True),
                "cleanft_proxy": _safe_numeric(cleanft_proxy).reset_index(drop=True),
                "early_adverse_proxy": _safe_numeric(early_adverse_proxy).reset_index(drop=True),
                "slow_timeout_proxy": _safe_numeric(slow_timeout_proxy).reset_index(drop=True),
                "path_dirty_proxy": _safe_numeric(path_dirty_proxy).reset_index(drop=True),
                "bad_features": bad_diag.get("features", []),
                "cleanft_features": cleanft_diag.get("features", []),
                "early_adverse_features": early_diag.get("features", []),
                "slow_timeout_features": slow_diag.get("features", []),
                "path_dirty_features": dirty_diag.get("features", []),
            }
        cache = proxy_cache[cache_key]
        for utility_target, target_group in group.groupby("utility_target", dropna=False):
            utility_soft = utility_map[str(utility_target)]
            soft_proxy, soft_diag = _proxy_score(
                train=train,
                valid=valid,
                features=feature_cols,
                target_train=utility_soft.loc[train_mask],
                top_k=int(proxy_top_k),
                method=str(proxy_method),
                tail_frac=0.05,
            )
            soft_proxy = _safe_numeric(soft_proxy).reset_index(drop=True)
            for _, spec in target_group.iterrows():
                selector = str(spec["selector"])
                raw_score = _selector_scores(
                    selector=selector,
                    soft_proxy=soft_proxy,
                    cleanft_proxy=cache["cleanft_proxy"],
                    early_adverse_proxy=cache["early_adverse_proxy"],
                    slow_timeout_proxy=cache["slow_timeout_proxy"],
                    path_dirty_proxy=cache["path_dirty_proxy"],
                )
                score = _apply_bad_and_floor(
                    raw_score,
                    bad_proxy=cache["bad_proxy"],
                    bad_threshold=float(spec["bad_threshold"]) if pd.notna(spec["bad_threshold"]) else None,
                    score_floor=float(spec["score_floor"]) if pd.notna(spec["score_floor"]) else None,
                )
                score = _run_entry_score(
                    valid_frame,
                    score,
                    gap_hours=float(spec["run_entry_gap_hours"]),
                )
                idx = _timestamp_top_k_indices(valid_frame, score, int(spec["top_k"]))
                selected = _metric_cols(valid_frame, valid_ft, idx)
                selected.insert(0, "valid_pos", idx.astype(int))
                for col in (
                    "spec_role",
                    "selector",
                    "utility_target",
                    "oracle_gate",
                    "proxy_method",
                    "proxy_top_k",
                    "bad_threshold",
                    "score_floor",
                    "run_entry_gap_hours",
                    "top_k",
                ):
                    selected[col] = spec[col]
                selected["score"] = score.iloc[idx].to_numpy(dtype=np.float32, copy=False)
                selected["soft_proxy"] = soft_proxy.iloc[idx].to_numpy(dtype=np.float32, copy=False)
                selected["bad_proxy"] = cache["bad_proxy"].iloc[idx].to_numpy(dtype=np.float32, copy=False)
                selected["cleanft_proxy"] = cache["cleanft_proxy"].iloc[idx].to_numpy(dtype=np.float32, copy=False)
                selected["early_adverse_proxy"] = cache["early_adverse_proxy"].iloc[idx].to_numpy(
                    dtype=np.float32,
                    copy=False,
                )
                selected["slow_timeout_proxy"] = cache["slow_timeout_proxy"].iloc[idx].to_numpy(
                    dtype=np.float32,
                    copy=False,
                )
                selected["path_dirty_proxy"] = cache["path_dirty_proxy"].iloc[idx].to_numpy(
                    dtype=np.float32,
                    copy=False,
                )
                ledgers.append(selected)
                diagnostics.append(
                    {
                        "spec_role": spec["spec_role"],
                        "selector": selector,
                        "utility_target": str(utility_target),
                        "bad_threshold": float(spec["bad_threshold"]) if pd.notna(spec["bad_threshold"]) else np.nan,
                        "score_floor": float(spec["score_floor"]) if pd.notna(spec["score_floor"]) else np.nan,
                        "run_entry_gap_hours": float(spec["run_entry_gap_hours"]),
                        "top_k": int(spec["top_k"]),
                        "selected_rows": int(len(idx)),
                        "soft_features": ",".join(soft_diag.get("features", [])),
                        "bad_features": ",".join(cache["bad_features"]),
                        "cleanft_features": ",".join(cache["cleanft_features"]),
                        "early_adverse_features": ",".join(cache["early_adverse_features"]),
                        "slow_timeout_features": ",".join(cache["slow_timeout_features"]),
                        "path_dirty_features": ",".join(cache["path_dirty_features"]),
                    }
                )
    ledger = pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame()
    return ledger, pd.DataFrame(diagnostics)


def _add_row_flags(ledger: pd.DataFrame) -> pd.DataFrame:
    out = ledger.copy()
    u = _safe_numeric(out["u_policy_net"])
    bad_mae = _safe_numeric(out["first_touch_mae_to_sl"]).ge(1.0)
    timeout = _safe_numeric(out["first_touch_timeout"]).gt(0.5)
    stop = _safe_numeric(out["first_touch_stop"]).gt(0.5)
    same_bar = _safe_numeric(out["first_touch_same_bar"]).gt(0.5)
    wide = _safe_numeric(out["barrier"]).gt(0.025)
    clean = _safe_numeric(out["clean_exec_actual"]).gt(0.5)
    out["row_clean_path"] = clean & u.gt(0.0)
    out["row_positive_dirty"] = u.gt(0.0) & ~clean & (bad_mae | timeout | stop | same_bar | wide)
    out["row_bad_mae_1r"] = bad_mae
    out["row_timeout"] = timeout
    out["row_wide25"] = wide
    return out


def _pool_rows(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        return ledger
    clean = ledger[
        ledger["spec_role"].astype(str).eq("june_clean_standalone") & ledger["row_clean_path"].astype(bool)
    ].copy()
    positive_dirty = ledger[
        ledger["spec_role"].astype(str).eq("fit_selected_dirty_holdout")
        & ledger["row_positive_dirty"].astype(bool)
    ].copy()
    bad_mae_tail = ledger[
        ledger["spec_role"].astype(str).eq("fit_selected_dirty_holdout")
        & ledger["row_bad_mae_1r"].astype(bool)
    ].copy()
    wide_positive = ledger[
        ledger["spec_role"].astype(str).eq("fit_selected_dirty_holdout")
        & ledger["row_wide25"].astype(bool)
        & _safe_numeric(ledger["u_policy_net"]).gt(0.0)
    ].copy()
    clean["pool"] = "clean_june_standalone"
    positive_dirty["pool"] = "positive_dirty_fit_selected_june"
    bad_mae_tail["pool"] = "badmae_tail_fit_selected_june"
    wide_positive["pool"] = "wide_positive_fit_selected_june"
    pool = pd.concat([clean, positive_dirty, bad_mae_tail, wide_positive], ignore_index=True)
    if pool.empty:
        return pool
    pool = pool.sort_values(["pool", "score"], ascending=[True, False])
    return pool.drop_duplicates(["pool", "valid_pos"], keep="first").reset_index(drop=True)


def _pool_summary(pool: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, group in pool.groupby("pool", dropna=False, sort=False):
        rows.append(
            {
                "pool": name,
                "rows": int(len(group)),
                "symbols": int(group["__symbol__"].nunique(dropna=True)),
                "timestamps": int(pd.to_datetime(group["__ts__"], errors="coerce").nunique(dropna=True)),
                "mean_u": _safe_mean(group["u_policy_net"]),
                "mean_return_net": _safe_mean(group["ret_net"]),
                "clean_exec_rate": _safe_mean(group["row_clean_path"].astype(float)),
                "positive_dirty_rate": _safe_mean(group["row_positive_dirty"].astype(float)),
                "bad_mae_1r_rate": _safe_mean(group["row_bad_mae_1r"].astype(float)),
                "timeout_rate": _safe_mean(group["row_timeout"].astype(float)),
                "wide25_rate": _safe_mean(group["row_wide25"].astype(float)),
                "p90_mae_to_sl": _safe_quantile(group["first_touch_mae_to_sl"], 0.90),
                "median_first_touch_bar": _safe_quantile(group["first_touch_bar"], 0.50),
                "top_symbols": ",".join(group["__symbol__"].astype(str).value_counts().head(8).index.tolist()),
                "top_selectors": ",".join(group["selector"].astype(str).value_counts().head(5).index.tolist()),
            }
        )
    return pd.DataFrame(rows)


def _feature_contrast(
    *,
    valid_frame: pd.DataFrame,
    valid_ft: pd.DataFrame,
    pool: pd.DataFrame,
    feature_cols: list[str],
    match_modes: list[str],
    min_class_rows: int,
) -> pd.DataFrame:
    if pool.empty:
        return pd.DataFrame()
    clean_pos = set(pool.loc[pool["pool"].eq("clean_june_standalone"), "valid_pos"].astype(int).tolist())
    rows: list[dict[str, Any]] = []
    clean_mask = pd.Series(np.arange(len(valid_frame))).isin(clean_pos).reset_index(drop=True)
    if int(clean_mask.sum()) < min_class_rows:
        return pd.DataFrame()
    for dirty_pool in sorted(pool["pool"].dropna().unique().tolist()):
        if dirty_pool == "clean_june_standalone":
            continue
        dirty_pos = set(pool.loc[pool["pool"].eq(dirty_pool), "valid_pos"].astype(int).tolist())
        dirty_mask = pd.Series(np.arange(len(valid_frame))).isin(dirty_pos).reset_index(drop=True)
        if int(dirty_mask.sum()) < min_class_rows:
            continue
        selected_mask = clean_mask | dirty_mask
        labels = clean_mask[selected_mask].reset_index(drop=True)
        for match_mode in match_modes:
            bucket = _bucket_key(valid_frame, valid_ft, match_mode).reset_index(drop=True)
            selected_bucket = bucket[selected_mask].reset_index(drop=True)
            for feature in feature_cols:
                values = _safe_numeric(valid_frame[feature]).reset_index(drop=True)
                ranks = _rank_within_bucket(values, bucket)
                clean_ranks = ranks[clean_mask].dropna()
                dirty_ranks = ranks[dirty_mask].dropna()
                if len(clean_ranks) < min_class_rows or len(dirty_ranks) < min_class_rows:
                    continue
                selected_ranks = ranks[selected_mask].reset_index(drop=True)
                auc = _auc_clean_high(selected_ranks, labels)
                bucket_frame = pd.DataFrame(
                    {
                        "bucket": selected_bucket,
                        "rank": selected_ranks,
                        "clean": labels,
                    }
                ).dropna(subset=["rank"])
                by_bucket = bucket_frame.groupby(["bucket", "clean"], dropna=False)["rank"].mean().unstack()
                bucket_gap = pd.Series(dtype=float)
                if True in by_bucket.columns and False in by_bucket.columns:
                    bucket_gap = (by_bucket[True] - by_bucket[False]).dropna()
                best_auc = max(float(auc), 1.0 - float(auc)) if math.isfinite(float(auc)) else np.nan
                rows.append(
                    {
                        "comparison": f"clean_june_standalone_vs_{dirty_pool}",
                        "dirty_pool": dirty_pool,
                        "match_mode": match_mode,
                        "feature": feature,
                        "feature_family": _feature_family(feature),
                        "clean_rows": int(len(clean_ranks)),
                        "dirty_rows": int(len(dirty_ranks)),
                        "matched_buckets": int(len(bucket_gap)),
                        "clean_rank_mean": float(clean_ranks.mean()),
                        "dirty_rank_mean": float(dirty_ranks.mean()),
                        "clean_minus_dirty_rank_mean": float(clean_ranks.mean() - dirty_ranks.mean()),
                        "bucket_equal_weight_gap_mean": float(bucket_gap.mean()) if len(bucket_gap) else np.nan,
                        "bucket_gap_positive_rate": float((bucket_gap > 0.0).mean()) if len(bucket_gap) else np.nan,
                        "auc_clean_high": float(auc),
                        "best_auc": float(best_auc),
                        "best_direction": (
                            "clean_high" if math.isfinite(float(auc)) and float(auc) >= 0.5 else "clean_low"
                        ),
                        "clean_median": float(values[clean_mask].median()),
                        "dirty_median": float(values[dirty_mask].median()),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["abs_rank_gap"] = out["clean_minus_dirty_rank_mean"].abs()
    out["abs_bucket_gap"] = out["bucket_equal_weight_gap_mean"].abs()
    return out.sort_values(["best_auc", "abs_rank_gap"], ascending=[False, False])


def _family_summary(contrast: pd.DataFrame, *, min_best_auc: float) -> pd.DataFrame:
    if contrast.empty:
        return pd.DataFrame()
    strong = contrast[contrast["best_auc"].ge(float(min_best_auc))].copy()
    if strong.empty:
        strong = contrast.sort_values("best_auc", ascending=False).head(100).copy()
    rows: list[dict[str, Any]] = []
    for family, group in strong.groupby("feature_family", dropna=False, sort=False):
        top = group.sort_values("best_auc", ascending=False)["feature"].drop_duplicates().head(10)
        rows.append(
            {
                "feature_family": family,
                "rows": int(len(group)),
                "match_modes": ",".join(sorted(group["match_mode"].astype(str).unique())),
                "mean_best_auc": _safe_mean(group["best_auc"]),
                "max_best_auc": _safe_quantile(group["best_auc"], 1.0),
                "mean_abs_rank_gap": _safe_mean(group["abs_rank_gap"]),
                "top_features": ",".join(top.astype(str).tolist()),
            }
        )
    return pd.DataFrame(rows).sort_values(["max_best_auc", "rows"], ascending=[False, False])


def _write_markdown(
    *,
    output_dir: Path,
    specs: pd.DataFrame,
    pool_summary: pd.DataFrame,
    contrast: pd.DataFrame,
    family_summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "gated_oracle_pathsafe_source_gap.md"
    spec_cols = [
        "spec_role",
        "selector",
        "utility_target",
        "bad_threshold",
        "score_floor",
        "run_entry_gap_hours",
        "top_k",
    ]
    pool_cols = [
        "pool",
        "rows",
        "symbols",
        "timestamps",
        "mean_u",
        "mean_return_net",
        "clean_exec_rate",
        "positive_dirty_rate",
        "bad_mae_1r_rate",
        "timeout_rate",
        "wide25_rate",
        "p90_mae_to_sl",
        "median_first_touch_bar",
        "top_symbols",
        "top_selectors",
    ]
    contrast_cols = [
        "comparison",
        "dirty_pool",
        "match_mode",
        "feature",
        "feature_family",
        "best_auc",
        "best_direction",
        "clean_rank_mean",
        "dirty_rank_mean",
        "clean_minus_dirty_rank_mean",
        "bucket_equal_weight_gap_mean",
        "bucket_gap_positive_rate",
        "clean_median",
        "dirty_median",
    ]
    family_cols = [
        "feature_family",
        "rows",
        "match_modes",
        "mean_best_auc",
        "max_best_auc",
        "mean_abs_rank_gap",
        "top_features",
    ]
    diag_cols = [
        "spec_role",
        "selector",
        "utility_target",
        "bad_threshold",
        "score_floor",
        "run_entry_gap_hours",
        "top_k",
        "selected_rows",
        "soft_features",
        "cleanft_features",
        "early_adverse_features",
        "path_dirty_features",
    ]
    lines = [
        "# Gated-Oracle Path-Safe Source Gap",
        "",
        "Scope: diagnostic-only, no model training. Clean standalone rows are selected using June evidence, so this is attribution for label redesign, not clean promotion evidence.",
        "",
        f"Stage directory: `{manifest['stage_dir']}`",
        f"Holdout month: `{manifest['holdout_month']}`",
        f"Feature count: `{manifest['feature_count']}`",
        f"Selected ledger rows: `{manifest['selected_ledger_rows']}`",
        f"Contrast pool rows: `{manifest['pool_rows']}`",
        "",
        "## Reconstructed Specs",
        "",
        _table(specs, spec_cols, limit=40),
        "",
        "## Pool Summary",
        "",
        _table(pool_summary, pool_cols, limit=20),
        "",
        "## Strongest Clean-vs-Dirty Separators",
        "",
        _table(contrast, contrast_cols, limit=80),
        "",
        "## Repeated Feature Families",
        "",
        _table(family_summary, family_cols, limit=40),
        "",
        "## Proxy Feature Diagnostics",
        "",
        _table(diagnostics, diag_cols, limit=40),
        "",
        "## Outputs",
        "",
        f"- Selected ledger: `{manifest['outputs']['selected_ledger_csv']}`",
        f"- Pool ledger: `{manifest['outputs']['pool_ledger_csv']}`",
        f"- Feature contrast: `{manifest['outputs']['feature_contrast']}`",
        f"- Family summary: `{manifest['outputs']['family_summary']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_report(
    *,
    stage_dir: Path,
    labels_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    holdout_month: str,
    max_feature_store_features: int | None,
    max_clean_specs: int,
    min_class_rows: int,
    match_modes: list[str],
    prior_embargo_hours: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    specs, stage_report = _load_stage_specs(stage_dir, max_clean_specs=max_clean_specs)
    frame, metrics, reports = _build_frame(
        labels_path=labels_path,
        feature_dir=feature_dir,
        feature_list_csv=feature_list_csv,
        max_feature_store_features=max_feature_store_features,
        include_causal_outcome_priors=True,
        include_causal_state_path_priors=True,
        include_event_confirmation_features=True,
        include_adverse_path_composites=True,
        prior_windows_days=list(DEFAULT_PRIOR_WINDOWS_DAYS),
        prior_embargo_hours=float(prior_embargo_hours),
        state_path_prior_features=list(DEFAULT_STATE_PATH_PRIOR_FEATURES),
        event_feature_store_features=list(DEFAULT_EVENT_FEATURE_STORE_FEATURES),
    )
    feature_cols = _feature_columns(frame)
    ft = _first_touch_metrics(frame, metrics)
    _ = _dirty_target(ft)
    ledger, diagnostics = _reconstruct_selected_rows(
        frame=frame,
        ft=ft,
        specs=specs,
        holdout_month=holdout_month,
        feature_cols=feature_cols,
    )
    ledger = _add_row_flags(ledger)
    pool = _pool_rows(ledger)

    month_series = frame["__ts__"].dt.to_period("M").astype(str)
    valid_mask = month_series == str(holdout_month)
    valid_frame = frame.loc[valid_mask].copy().reset_index(drop=True)
    valid_ft = ft.loc[valid_mask].copy().reset_index(drop=True)
    contrast = _feature_contrast(
        valid_frame=valid_frame,
        valid_ft=valid_ft,
        pool=pool,
        feature_cols=feature_cols,
        match_modes=match_modes,
        min_class_rows=min_class_rows,
    )
    families = _family_summary(contrast, min_best_auc=0.68)
    pool_summary = _pool_summary(pool)

    paths = {
        "selected_ledger_csv": output_dir / "gated_oracle_pathsafe_selected_rows.csv",
        "pool_ledger_csv": output_dir / "gated_oracle_pathsafe_contrast_pool.csv",
        "feature_contrast": output_dir / "gated_oracle_pathsafe_feature_contrast.csv",
        "family_summary": output_dir / "gated_oracle_pathsafe_family_summary.csv",
        "diagnostics": output_dir / "gated_oracle_pathsafe_proxy_diagnostics.csv",
        "manifest": output_dir / "manifest.json",
    }
    ledger.to_csv(paths["selected_ledger_csv"], index=False)
    pool.to_csv(paths["pool_ledger_csv"], index=False)
    contrast.to_csv(paths["feature_contrast"], index=False)
    families.to_csv(paths["family_summary"], index=False)
    diagnostics.to_csv(paths["diagnostics"], index=False)

    manifest = {
        "scope": "diagnostic_only_stage71_pathsafe_source_gap",
        "stage_dir": str(stage_dir),
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "holdout_month": str(holdout_month),
        "rows": int(len(frame)),
        "valid_rows": int(valid_mask.sum()),
        "feature_count": int(len(feature_cols)),
        "selected_ledger_rows": int(len(ledger)),
        "pool_rows": int(len(pool)),
        "clean_pool_rows": int(pool["pool"].eq("clean_june_standalone").sum()) if not pool.empty else 0,
        "positive_dirty_pool_rows": int(pool["pool"].eq("positive_dirty_fit_selected_june").sum())
        if not pool.empty
        else 0,
        "badmae_tail_pool_rows": int(pool["pool"].eq("badmae_tail_fit_selected_june").sum())
        if not pool.empty
        else 0,
        "wide_positive_pool_rows": int(pool["pool"].eq("wide_positive_fit_selected_june").sum())
        if not pool.empty
        else 0,
        "match_modes": list(match_modes),
        "min_class_rows": int(min_class_rows),
        "max_clean_specs": int(max_clean_specs),
        "stage": stage_report,
        "outputs": {key: str(value) for key, value in paths.items()},
        **reports,
    }
    markdown = _write_markdown(
        output_dir=output_dir,
        specs=specs,
        pool_summary=pool_summary,
        contrast=contrast,
        family_summary=families,
        diagnostics=diagnostics,
        manifest=manifest,
    )
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE71_DIR)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--holdout-month", default=DEFAULT_HOLDOUT_MONTH)
    parser.add_argument("--max-feature-store-features", type=int, default=None)
    parser.add_argument("--max-clean-specs", type=int, default=4)
    parser.add_argument("--min-class-rows", type=int, default=10)
    parser.add_argument("--match-modes", type=lambda value: _parse_csv(value, DEFAULT_MATCH_MODES), default=",".join(DEFAULT_MATCH_MODES))
    parser.add_argument("--prior-embargo-hours", type=float, default=24.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_report(
        stage_dir=args.stage_dir,
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        holdout_month=str(args.holdout_month),
        max_feature_store_features=args.max_feature_store_features,
        max_clean_specs=int(args.max_clean_specs),
        min_class_rows=int(args.min_class_rows),
        match_modes=list(args.match_modes),
        prior_embargo_hours=float(args.prior_embargo_hours),
    )
    summary = {
        key: manifest.get(key)
        for key in (
            "output_dir",
            "holdout_month",
            "rows",
            "valid_rows",
            "feature_count",
            "selected_ledger_rows",
            "pool_rows",
            "clean_pool_rows",
            "positive_dirty_pool_rows",
            "badmae_tail_pool_rows",
            "wide_positive_pool_rows",
            "outputs",
        )
    }
    print(json.dumps(_json_safe(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
