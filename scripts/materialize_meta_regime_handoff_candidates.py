#!/usr/bin/env python3
"""Materialize passing meta-regime candidate streams for downstream replay.

The input is the month-forward OOS prediction ledger produced by
``report_meta_regime_context_filter_oos.py``.  This script does not refit
models, choose thresholds from validation rows, or alter scores.  It freezes
the selectors already marked as passing in the OOS summary and writes selected
top-k candidate ledgers plus diagnostics for handoff to downstream replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_CONTEXT_DIR = Path(
    "data_perp/reports/ae_gmm_archetype_validation_status_20260704/"
    "meta_prefeature_regime_source_interaction_audit_v1/"
    "meta_regime_context_filter_oos_v1"
)
DEFAULT_OUTPUT_DIR = DEFAULT_CONTEXT_DIR / "meta_regime_handoff_candidates_v1"
DEFAULT_TOP_FRAC = 0.10
PASS_STATUS = "local_path_filter_pass"
KEY_COLUMNS = ("timestamp", "symbol", "side_name", "month")
EXECUTION_KEY_COLUMNS = ("timestamp", "symbol", "side_name")
MIN_SIDE_ROWS_FOR_GATE = 20
SIDE_SHARE_WARNING_CAP = 0.75
OUTCOME_COLUMNS = (
    "ev_after_cost",
    "clean_exec",
    "dirty_positive",
    "bad_mae",
    "timeout",
    "mfe_before_mae_1r",
    "mae_before_mfe_1r",
)
EVAL_ONLY_COLUMNS = set(OUTCOME_COLUMNS)
CONTEXT_COLUMNS = (
    "source_tag",
    "source_family",
    "candidate_liquidity_bin",
    "candidate_activity_liquidity_bin",
    "candidate_volatility_bin",
    "candidate_volatility_zscore_bin",
    "candidate_directional_vol_imbalance_bin",
    "candidate_market_dispersion_bin",
    "candidate_aegmm_entropy_bin",
    "candidate_aegmm_distance_bin",
    "candidate_reconstruction_bin",
    "candidate_archetype_side_aegmm_entropy_bin",
    "candidate_archetype_side_aegmm_distance_bin",
    "candidate_archetype_side_liquidity_bin",
    "candidate_archetype_side_volatility_bin",
    "candidate_archetype_side_activity_liquidity_bin",
    "candidate_archetype_side_directional_vol_imbalance_bin",
    "candidate_archetype_side_market_dispersion_bin",
    "candidate_volatility_shape_bin",
    "candidate_exec_move_speed_bin",
    "candidate_archetype_side_exec_move_speed_bin",
    "candidate_exec_signal_to_spread_bin",
    "candidate_archetype_side_exec_signal_to_spread_bin",
    "candidate_exec_slow_resolution_risk_bin",
    "candidate_archetype_side_exec_slow_resolution_risk_bin",
    "candidate_exec_adverse_path_pressure_bin",
    "candidate_archetype_side_exec_adverse_path_pressure_bin",
    "candidate_exec_opportunity_pressure_bin",
    "candidate_archetype_side_exec_opportunity_pressure_bin",
    "ctx_exec_spread_bps_proxy",
    "ctx_exec_liquidity_rank_proxy",
    "ctx_exec_spread_pressure_proxy",
    "ctx_exec_volatility_rank_proxy",
    "ctx_exec_move_speed_proxy",
    "ctx_exec_signal_to_spread_proxy",
    "ctx_exec_aegmm_uncertainty_proxy",
    "ctx_exec_model_risk_pressure_proxy",
    "ctx_exec_adverse_path_pressure_proxy",
    "ctx_exec_slow_resolution_risk_proxy",
    "ctx_exec_opportunity_pressure_proxy",
)
POLICY_OVERLAY_RULES = (
    {
        "policy_overlay": "abstain_vol_shape_q3_dir_imbalance_q1",
        "column": "candidate_volatility_shape_bin",
        "values": ("vol_shape__volatility_q3__dir_vol_imbalance_q1",),
        "action": "abstain",
        "evidence": (
            "train_only_regime_action_table marks the corresponding volatility-shape bucket as "
            "downweight_or_meta_filter with elevated bad-MAE/timeout path quality."
        ),
    },
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if pd.isna(value):
        return None
    return value


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_num(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _safe_mean(values: Any) -> float:
    arr = _safe_num(values).replace([np.inf, -np.inf], np.nan).dropna()
    return float(arr.mean()) if len(arr) else float("nan")


def _rate(values: Any) -> float:
    arr = _safe_num(values).replace([np.inf, -np.inf], np.nan).dropna()
    return float(arr.clip(0.0, 1.0).mean()) if len(arr) else float("nan")


def _score_col(feature_set: str, selector: str) -> str:
    return f"score_{feature_set}_{selector}"


def _week_start(values: pd.Series) -> pd.Series:
    ts = pd.to_datetime(values, errors="coerce", utc=True)
    naive = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    return naive.dt.to_period("W-SUN").dt.start_time.astype(str)


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    side = frame["side_name"].astype(str) if "side_name" in frame.columns else pd.Series(dtype=str)
    return {
        "rows": int(len(frame)),
        "ev": _safe_mean(frame["ev_after_cost"]) if "ev_after_cost" in frame.columns else float("nan"),
        "clean_precision": _rate(frame["clean_exec"]) if "clean_exec" in frame.columns else float("nan"),
        "dirty_positive": _rate(frame["dirty_positive"]) if "dirty_positive" in frame.columns else float("nan"),
        "bad_mae": _rate(frame["bad_mae"]) if "bad_mae" in frame.columns else float("nan"),
        "timeout": _rate(frame["timeout"]) if "timeout" in frame.columns else float("nan"),
        "mfe_before_mae_1r": _rate(frame["mfe_before_mae_1r"]) if "mfe_before_mae_1r" in frame.columns else float("nan"),
        "mae_before_mfe_1r": _rate(frame["mae_before_mfe_1r"]) if "mae_before_mfe_1r" in frame.columns else float("nan"),
        "long_share": float(side.eq("long").mean()) if len(side) else float("nan"),
        "short_share": float(side.eq("short").mean()) if len(side) else float("nan"),
        "symbols": int(frame["symbol"].astype(str).nunique()) if "symbol" in frame.columns and len(frame) else 0,
    }


def _gate_from_aggregate_and_slices(
    *,
    primary: pd.DataFrame,
    primary_monthly: pd.DataFrame,
    primary_side: pd.DataFrame,
    primary_month_side: pd.DataFrame,
) -> dict[str, Any]:
    side_eligible = (
        primary_side[_safe_num(primary_side["rows"]).ge(MIN_SIDE_ROWS_FOR_GATE)].copy()
        if not primary_side.empty and "rows" in primary_side.columns
        else pd.DataFrame()
    )
    month_side_eligible = (
        primary_month_side[_safe_num(primary_month_side["rows"]).ge(MIN_SIDE_ROWS_FOR_GATE)].copy()
        if not primary_month_side.empty and "rows" in primary_month_side.columns
        else pd.DataFrame()
    )
    gate = {
        "mean_ev": _safe_mean(primary_monthly["ev"]) if "ev" in primary_monthly.columns else float("nan"),
        "worst_month_ev": float(_safe_num(primary_monthly["ev"]).min())
        if "ev" in primary_monthly.columns and len(primary_monthly)
        else float("nan"),
        "bad_mae": _rate(primary["bad_mae"]) if "bad_mae" in primary.columns else float("nan"),
        "max_month_bad_mae": float(_safe_num(primary_monthly["bad_mae"]).max())
        if "bad_mae" in primary_monthly.columns and len(primary_monthly)
        else float("nan"),
        "timeout": _rate(primary["timeout"]) if "timeout" in primary.columns else float("nan"),
        "max_month_timeout": float(_safe_num(primary_monthly["timeout"]).max())
        if "timeout" in primary_monthly.columns and len(primary_monthly)
        else float("nan"),
        "selected_rows": int(len(primary)),
        "symbols": int(primary["symbol"].astype(str).nunique()) if "symbol" in primary.columns and len(primary) else 0,
        "long_share": float(primary["side_name"].astype(str).eq("long").mean())
        if "side_name" in primary.columns and len(primary)
        else float("nan"),
        "short_share": float(primary["side_name"].astype(str).eq("short").mean())
        if "side_name" in primary.columns and len(primary)
        else float("nan"),
        "side_min_ev": float(_safe_num(side_eligible["ev"]).min())
        if "ev" in side_eligible.columns and len(side_eligible)
        else float("nan"),
        "side_max_bad_mae": float(_safe_num(side_eligible["bad_mae"]).max())
        if "bad_mae" in side_eligible.columns and len(side_eligible)
        else float("nan"),
        "side_max_timeout": float(_safe_num(side_eligible["timeout"]).max())
        if "timeout" in side_eligible.columns and len(side_eligible)
        else float("nan"),
        "month_side_min_ev": float(_safe_num(month_side_eligible["ev"]).min())
        if "ev" in month_side_eligible.columns and len(month_side_eligible)
        else float("nan"),
        "month_side_max_bad_mae": float(_safe_num(month_side_eligible["bad_mae"]).max())
        if "bad_mae" in month_side_eligible.columns and len(month_side_eligible)
        else float("nan"),
        "month_side_max_timeout": float(_safe_num(month_side_eligible["timeout"]).max())
        if "timeout" in month_side_eligible.columns and len(month_side_eligible)
        else float("nan"),
    }
    gate["local_path_filter_status"] = (
        "pass"
        if gate["mean_ev"] > 0.0
        and gate["worst_month_ev"] > 0.0
        and gate["bad_mae"] <= 0.50
        and gate["max_month_bad_mae"] <= 0.50
        and gate["timeout"] <= 0.12
        and gate["max_month_timeout"] <= 0.12
        and gate["selected_rows"] >= MIN_SIDE_ROWS_FOR_GATE
        else "fail"
    )
    gate["side_share_status"] = (
        "pass"
        if max(gate.get("long_share", 0.0), gate.get("short_share", 0.0)) <= SIDE_SHARE_WARNING_CAP
        else "warning"
    )
    gate["side_quality_status"] = (
        "pass"
        if (
            not len(side_eligible)
            or (
                gate["side_min_ev"] > 0.0
                and gate["side_max_bad_mae"] <= 0.50
                and gate["side_max_timeout"] <= 0.12
            )
        )
        else "fail"
    )
    gate["month_side_quality_status"] = (
        "pass"
        if (
            not len(month_side_eligible)
            or (
                gate["month_side_min_ev"] > 0.0
                and gate["month_side_max_bad_mae"] <= 0.50
                and gate["month_side_max_timeout"] <= 0.12
            )
        )
        else "fail"
    )
    gate["candidate_handoff_status"] = (
        "candidate_handoff_pass"
        if gate["local_path_filter_status"] == "pass"
        and gate["side_share_status"] == "pass"
        and gate["side_quality_status"] == "pass"
        and gate["month_side_quality_status"] == "pass"
        else "local_path_pass_side_or_slice_rework"
        if gate["local_path_filter_status"] == "pass"
        else "fail_or_diagnostic"
    )
    return gate


def _select_top_by_month(frame: pd.DataFrame, score_col: str, top_frac: float) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for month, group in frame.groupby("month", dropna=False, sort=True):
        scores = _safe_num(group[score_col])
        valid = group.loc[scores.notna()].copy()
        if valid.empty:
            continue
        n = max(1, int(math.ceil(float(top_frac) * len(valid))))
        selected = valid.sort_values(score_col, ascending=False, kind="mergesort").head(n).copy()
        selected["selection_month"] = str(month)
        selected["month_scorable_rows"] = int(len(valid))
        selected["month_top_frac"] = float(top_frac)
        rows.append(selected)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _build_scored_frame(predictions: pd.DataFrame, pass_rows: pd.DataFrame, top_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored_frames: list[pd.DataFrame] = []
    selected_frames: list[pd.DataFrame] = []
    base_cols = [
        col
        for col in list(KEY_COLUMNS) + list(CONTEXT_COLUMNS) + list(OUTCOME_COLUMNS)
        if col in predictions.columns
    ]
    for rank, row in enumerate(pass_rows.itertuples(index=False), start=1):
        score_col = _score_col(str(row.feature_set), str(row.selector))
        if score_col not in predictions.columns:
            continue
        work = predictions.loc[:, base_cols + [score_col]].copy()
        work = work.rename(columns={score_col: "meta_regime_score"})
        work["selector_rank"] = int(rank)
        work["primary_selector"] = bool(rank == 1)
        work["feature_set"] = str(row.feature_set)
        work["model_target"] = str(row.model_target)
        work["selector"] = str(row.selector)
        work["selector_key"] = f"{row.feature_set}::{row.model_target}::{row.selector}"
        work["score_column"] = score_col
        work["configured_top_frac"] = float(top_frac)
        work["scorable"] = _safe_num(work["meta_regime_score"]).notna()
        work["score_rank_pct_by_month"] = (
            work.groupby("month", dropna=False)["meta_regime_score"]
            .transform(lambda s: _safe_num(s).rank(method="average", pct=True))
            .astype("float32")
        )
        work["accepted"] = False
        work = work.reset_index(names="prediction_row_id")
        selected = _select_top_by_month(work, "meta_regime_score", top_frac)
        selected["accepted"] = True
        if len(selected):
            work.loc[work["prediction_row_id"].isin(selected["prediction_row_id"]), "accepted"] = True
        scored_frames.append(work)
        selected_frames.append(selected)
    scored = pd.concat(scored_frames, ignore_index=True, sort=False) if scored_frames else pd.DataFrame()
    selected = pd.concat(selected_frames, ignore_index=True, sort=False) if selected_frames else pd.DataFrame()
    if not selected.empty:
        selected["handoff_candidate_id"] = np.arange(len(selected), dtype=np.int64)
    return scored, selected


def _group_metrics(frame: pd.DataFrame, group_cols: list[str], scope: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    present = [col for col in group_cols if col in frame.columns]
    if not present:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(present, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = {"scope": scope}
        row.update({col: str(value) for col, value in zip(present, key)})
        row.update(_metrics(group))
        rows.append(row)
    return pd.DataFrame(rows)


def _selector_group_columns(frame: pd.DataFrame) -> list[str]:
    cols = ["selector_key", "feature_set", "model_target", "selector"]
    return [col for col in cols if col in frame.columns]


def _decision_time_export(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    prohibited = set(EVAL_ONLY_COLUMNS)
    prohibited.update(col for col in frame.columns if str(col).startswith("__"))
    keep = [col for col in frame.columns if col not in prohibited]
    return frame.loc[:, keep].copy()


def _execution_decision_export(frame: pd.DataFrame) -> pd.DataFrame:
    return _decision_time_export(_execution_selected_export(frame))


def _execution_selected_export(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    key_cols = [col for col in EXECUTION_KEY_COLUMNS if col in out.columns]
    if len(key_cols) != len(EXECUTION_KEY_COLUMNS):
        return out
    sort_cols = [
        col
        for col in (
            "meta_regime_score",
            "score_rank_pct_by_month",
            "selector_rank",
        )
        if col in out.columns
    ]
    ascending = [False if col != "selector_rank" else True for col in sort_cols]
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=ascending, kind="mergesort")
    return out.drop_duplicates(key_cols, keep="first").reset_index(drop=True)


def _execution_duplicate_report(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"rows": 0, "unique_execution_keys": 0, "duplicate_execution_rows": 0}
    key_cols = [col for col in EXECUTION_KEY_COLUMNS if col in frame.columns]
    if len(key_cols) != len(EXECUTION_KEY_COLUMNS):
        return {
            "rows": int(len(frame)),
            "unique_execution_keys": 0,
            "duplicate_execution_rows": 0,
            "missing_execution_key_columns": sorted(set(EXECUTION_KEY_COLUMNS) - set(key_cols)),
        }
    unique_keys = int(frame.loc[:, key_cols].drop_duplicates().shape[0])
    return {
        "rows": int(len(frame)),
        "unique_execution_keys": unique_keys,
        "duplicate_execution_rows": int(len(frame) - unique_keys),
        "duplicate_execution_row_rate": float((len(frame) - unique_keys) / max(len(frame), 1)),
    }


def _apply_policy_overlay(frame: pd.DataFrame, rule: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    if frame.empty:
        return frame.copy(), {
            "policy_overlay": str(rule.get("policy_overlay", "unknown")),
            "removed_rows": 0,
            "kept_rows": 0,
            "removed_share": float("nan"),
        }
    column = str(rule["column"])
    values = {str(value) for value in rule.get("values", ())}
    if column not in frame.columns:
        kept = frame.copy()
        kept["policy_overlay"] = str(rule.get("policy_overlay", "unknown"))
        return kept, {
            "policy_overlay": str(rule.get("policy_overlay", "unknown")),
            "removed_rows": 0,
            "kept_rows": int(len(kept)),
            "removed_share": 0.0,
            "missing_column": column,
        }
    remove = frame[column].astype(str).isin(values)
    kept = frame.loc[~remove].copy()
    kept["policy_overlay"] = str(rule.get("policy_overlay", "unknown"))
    kept["policy_overlay_action"] = str(rule.get("action", "filter"))
    kept["policy_overlay_evidence"] = str(rule.get("evidence", ""))
    return kept, {
        "policy_overlay": str(rule.get("policy_overlay", "unknown")),
        "column": column,
        "values": sorted(values),
        "removed_rows": int(remove.sum()),
        "kept_rows": int(len(kept)),
        "removed_share": float(remove.mean()) if len(frame) else float("nan"),
        "action": str(rule.get("action", "filter")),
        "evidence": str(rule.get("evidence", "")),
    }


def _gate_for_selected_frame(frame: pd.DataFrame, selector_group_cols: list[str]) -> pd.DataFrame:
    if frame.empty or not selector_group_cols:
        return pd.DataFrame()
    monthly = _group_metrics(frame, selector_group_cols + ["month"], "selector_month")
    side = _group_metrics(frame, selector_group_cols + ["side_name"], "selector_side")
    month_side = _group_metrics(frame, selector_group_cols + ["month", "side_name"], "selector_month_side")
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(selector_group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        identity = {col: str(value) for col, value in zip(selector_group_cols, key)}
        selector_key = identity.get("selector_key", "")
        group_monthly = monthly[monthly["selector_key"].eq(selector_key)].copy() if selector_key and not monthly.empty else pd.DataFrame()
        group_side = side[side["selector_key"].eq(selector_key)].copy() if selector_key and not side.empty else pd.DataFrame()
        group_month_side = (
            month_side[month_side["selector_key"].eq(selector_key)].copy()
            if selector_key and not month_side.empty
            else pd.DataFrame()
        )
        row = dict(identity)
        row.update(
            _gate_from_aggregate_and_slices(
                primary=group,
                primary_monthly=group_monthly,
                primary_side=group_side,
                primary_month_side=group_month_side,
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _execution_gate_for_selected_frame(frame: pd.DataFrame, selector_group_cols: list[str]) -> pd.DataFrame:
    if frame.empty or not selector_group_cols:
        return pd.DataFrame()
    deduped_frames: list[pd.DataFrame] = []
    for _key, group in frame.groupby(selector_group_cols, dropna=False, sort=True):
        deduped_frames.append(_execution_selected_export(group))
    execution_selected = (
        pd.concat(deduped_frames, ignore_index=True, sort=False)
        if deduped_frames
        else pd.DataFrame()
    )
    if execution_selected.empty:
        return pd.DataFrame()
    gate = _gate_for_selected_frame(execution_selected, selector_group_cols)
    if not gate.empty:
        gate["gate_scope"] = "executable_key"
    return gate


def run_materialization(
    *,
    context_dir: Path,
    output_dir: Path,
    top_frac: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = context_dir / "meta_regime_context_filter_oos_predictions.parquet"
    summary_path = context_dir / "meta_regime_context_filter_summary.csv"
    week_guard_path = context_dir / "week_health_guard_v1" / "week_health_guard_summary.csv"
    predictions = pd.read_parquet(predictions_path)
    summary = pd.read_csv(summary_path)
    pass_rows = summary[summary["gate3_candidate_status"].astype(str).eq(PASS_STATUS)].copy()
    if pass_rows.empty:
        raise RuntimeError(f"No summary rows with gate3_candidate_status={PASS_STATUS!r}")
    pass_rows = pass_rows.sort_values(
        ["mean_top10_ev", "worst_month_top10_ev", "mean_top10_bad_mae"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    pass_rows["selector_rank"] = np.arange(1, len(pass_rows) + 1)
    pass_rows["score_column"] = [_score_col(str(r.feature_set), str(r.selector)) for r in pass_rows.itertuples(index=False)]
    pass_rows["selector_key"] = [
        f"{r.feature_set}::{r.model_target}::{r.selector}" for r in pass_rows.itertuples(index=False)
    ]
    pass_rows["score_column_available"] = pass_rows["score_column"].isin(predictions.columns)

    scored, selected = _build_scored_frame(predictions, pass_rows[pass_rows["score_column_available"]], top_frac)
    if not selected.empty and "timestamp" in selected.columns:
        selected["timestamp"] = pd.to_datetime(selected["timestamp"], errors="coerce", utc=True)
        selected["week_start"] = _week_start(selected["timestamp"])
    if not scored.empty and "timestamp" in scored.columns:
        scored["timestamp"] = pd.to_datetime(scored["timestamp"], errors="coerce", utc=True)

    monthly_rows = []
    selector_rows = []
    selector_group_cols = _selector_group_columns(selected)
    for key, group in selected.groupby(selector_group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        selector_row = {col: str(value) for col, value in zip(selector_group_cols, key)}
        selector_row.update(_metrics(group))
        selector_rows.append(selector_row)
        for month, month_group in group.groupby("month", dropna=False, sort=True):
            row = {col: str(value) for col, value in zip(selector_group_cols, key)}
            row["month"] = str(month)
            row.update(_metrics(month_group))
            monthly_rows.append(row)
    selector_metrics = pd.DataFrame(selector_rows)
    monthly_metrics = pd.DataFrame(monthly_rows)
    week_metrics = _group_metrics(selected, selector_group_cols + ["week_start"], "selector_week")
    side_metrics = _group_metrics(selected, selector_group_cols + ["side_name"], "selector_side")
    month_side_metrics = _group_metrics(selected, selector_group_cols + ["month", "side_name"], "selector_month_side")
    source_metrics = _group_metrics(selected, selector_group_cols + ["source_family"], "selector_source")
    regime_metrics = pd.concat(
        [
            _group_metrics(selected, selector_group_cols + [col], f"selector_{col}")
            for col in CONTEXT_COLUMNS
            if col.startswith("candidate_") and col in selected.columns
        ],
        ignore_index=True,
        sort=False,
    )

    primary = selected[selected["primary_selector"].astype(bool)].copy() if "primary_selector" in selected.columns else pd.DataFrame()
    primary_selector_key = str(pass_rows.iloc[0]["score_column"])
    if not primary.empty and "selector_key" in primary.columns:
        primary_selector_key = str(primary["selector_key"].iloc[0])
    primary_monthly = monthly_metrics[
        monthly_metrics["selector_key"].eq(primary_selector_key)
    ].copy() if not monthly_metrics.empty else pd.DataFrame()
    primary_side = side_metrics[
        side_metrics["selector_key"].eq(primary_selector_key)
    ].copy() if not side_metrics.empty else pd.DataFrame()
    primary_month_side = month_side_metrics[
        month_side_metrics["selector_key"].eq(primary_selector_key)
    ].copy() if not month_side_metrics.empty else pd.DataFrame()
    primary_gate = _gate_from_aggregate_and_slices(
        primary=primary,
        primary_monthly=primary_monthly,
        primary_side=primary_side,
        primary_month_side=primary_month_side,
    )
    selector_gate_summary = _gate_for_selected_frame(selected, selector_group_cols)
    overlay_frames: list[pd.DataFrame] = []
    overlay_reports: list[dict[str, Any]] = []
    for rule in POLICY_OVERLAY_RULES:
        overlay_selected, overlay_report = _apply_policy_overlay(selected, rule)
        overlay_frames.append(overlay_selected)
        overlay_reports.append(overlay_report)
    overlay_selected_all = (
        pd.concat(overlay_frames, ignore_index=True, sort=False)
        if overlay_frames
        else pd.DataFrame()
    )
    overlay_group_cols = ["policy_overlay"] + selector_group_cols
    overlay_gate_summary = _gate_for_selected_frame(overlay_selected_all, overlay_group_cols)
    execution_selector_gate_summary = _execution_gate_for_selected_frame(selected, selector_group_cols)
    overlay_execution_gate_summary = _execution_gate_for_selected_frame(overlay_selected_all, overlay_group_cols)
    primary_overlay_selected = (
        overlay_selected_all[
            overlay_selected_all["primary_selector"].astype(bool)
            & overlay_selected_all["policy_overlay"].eq(POLICY_OVERLAY_RULES[0]["policy_overlay"])
        ].copy()
        if not overlay_selected_all.empty and "primary_selector" in overlay_selected_all.columns
        else pd.DataFrame()
    )

    outputs = {
        "decision_candidates": output_dir / "meta_regime_handoff_decision_candidates.parquet",
        "primary_decision_candidates": output_dir / "meta_regime_handoff_primary_decision_candidates.parquet",
        "selected_candidates": output_dir / "meta_regime_handoff_selected_candidates.parquet",
        "primary_selected_candidates": output_dir / "meta_regime_handoff_primary_selected_candidates.parquet",
        "primary_policy_overlay_decision_candidates": output_dir / "meta_regime_handoff_primary_policy_overlay_decision_candidates.parquet",
        "execution_decision_candidates": output_dir / "meta_regime_handoff_execution_decision_candidates.parquet",
        "primary_execution_decision_candidates": output_dir / "meta_regime_handoff_primary_execution_decision_candidates.parquet",
        "primary_policy_overlay_execution_decision_candidates": output_dir / "meta_regime_handoff_primary_policy_overlay_execution_decision_candidates.parquet",
        "execution_selected_candidates": output_dir / "meta_regime_handoff_execution_selected_candidates.parquet",
        "primary_execution_selected_candidates": output_dir / "meta_regime_handoff_primary_execution_selected_candidates.parquet",
        "primary_policy_overlay_execution_selected_candidates": output_dir / "meta_regime_handoff_primary_policy_overlay_execution_selected_candidates.parquet",
        "primary_policy_overlay_selected_candidates": output_dir / "meta_regime_handoff_primary_policy_overlay_selected_candidates.parquet",
        "scored_candidates": output_dir / "meta_regime_handoff_scored_candidates.parquet",
        "passing_selectors": output_dir / "meta_regime_handoff_passing_selectors.csv",
        "selector_metrics": output_dir / "meta_regime_handoff_selector_metrics.csv",
        "selector_gate_summary": output_dir / "meta_regime_handoff_selector_gate_summary.csv",
        "execution_selector_gate_summary": output_dir / "meta_regime_handoff_execution_selector_gate_summary.csv",
        "policy_overlay_gate_summary": output_dir / "meta_regime_handoff_policy_overlay_gate_summary.csv",
        "policy_overlay_execution_gate_summary": output_dir / "meta_regime_handoff_policy_overlay_execution_gate_summary.csv",
        "policy_overlay_report": output_dir / "meta_regime_handoff_policy_overlay_report.csv",
        "monthly_metrics": output_dir / "meta_regime_handoff_monthly_metrics.csv",
        "week_metrics": output_dir / "meta_regime_handoff_week_metrics.csv",
        "month_side_metrics": output_dir / "meta_regime_handoff_month_side_metrics.csv",
        "side_metrics": output_dir / "meta_regime_handoff_side_metrics.csv",
        "source_metrics": output_dir / "meta_regime_handoff_source_metrics.csv",
        "regime_metrics": output_dir / "meta_regime_handoff_regime_metrics.csv",
        "manifest": output_dir / "meta_regime_handoff_manifest.json",
        "report": output_dir / "meta_regime_handoff_report.md",
    }
    decision_candidates = _decision_time_export(selected)
    primary_selected = selected[selected["primary_selector"].astype(bool)].copy() if "primary_selector" in selected.columns else pd.DataFrame()
    primary_decision_candidates = _decision_time_export(primary_selected)
    primary_policy_overlay_decision_candidates = _decision_time_export(primary_overlay_selected)
    execution_selected_candidates = _execution_selected_export(selected)
    primary_execution_selected_candidates = _execution_selected_export(primary_selected)
    primary_policy_overlay_execution_selected_candidates = _execution_selected_export(primary_overlay_selected)
    execution_decision_candidates = _execution_decision_export(selected)
    primary_execution_decision_candidates = _execution_decision_export(primary_selected)
    primary_policy_overlay_execution_decision_candidates = _execution_decision_export(primary_overlay_selected)
    decision_candidates.to_parquet(outputs["decision_candidates"], index=False)
    primary_decision_candidates.to_parquet(outputs["primary_decision_candidates"], index=False)
    execution_decision_candidates.to_parquet(outputs["execution_decision_candidates"], index=False)
    primary_execution_decision_candidates.to_parquet(outputs["primary_execution_decision_candidates"], index=False)
    primary_policy_overlay_execution_decision_candidates.to_parquet(
        outputs["primary_policy_overlay_execution_decision_candidates"],
        index=False,
    )
    execution_selected_candidates.to_parquet(outputs["execution_selected_candidates"], index=False)
    primary_execution_selected_candidates.to_parquet(outputs["primary_execution_selected_candidates"], index=False)
    primary_policy_overlay_execution_selected_candidates.to_parquet(
        outputs["primary_policy_overlay_execution_selected_candidates"],
        index=False,
    )
    selected.to_parquet(outputs["selected_candidates"], index=False)
    primary_selected.to_parquet(outputs["primary_selected_candidates"], index=False)
    primary_policy_overlay_decision_candidates.to_parquet(outputs["primary_policy_overlay_decision_candidates"], index=False)
    primary_overlay_selected.to_parquet(outputs["primary_policy_overlay_selected_candidates"], index=False)
    scored.to_parquet(outputs["scored_candidates"], index=False)
    pass_rows.to_csv(outputs["passing_selectors"], index=False)
    selector_metrics.to_csv(outputs["selector_metrics"], index=False)
    selector_gate_summary.to_csv(outputs["selector_gate_summary"], index=False)
    execution_selector_gate_summary.to_csv(outputs["execution_selector_gate_summary"], index=False)
    overlay_gate_summary.to_csv(outputs["policy_overlay_gate_summary"], index=False)
    overlay_execution_gate_summary.to_csv(outputs["policy_overlay_execution_gate_summary"], index=False)
    pd.DataFrame(overlay_reports).to_csv(outputs["policy_overlay_report"], index=False)
    monthly_metrics.to_csv(outputs["monthly_metrics"], index=False)
    week_metrics.to_csv(outputs["week_metrics"], index=False)
    month_side_metrics.to_csv(outputs["month_side_metrics"], index=False)
    side_metrics.to_csv(outputs["side_metrics"], index=False)
    source_metrics.to_csv(outputs["source_metrics"], index=False)
    regime_metrics.to_csv(outputs["regime_metrics"], index=False)

    week_guard_report: dict[str, Any] = {"available": False}
    if week_guard_path.exists():
        week_guard = pd.read_csv(week_guard_path)
        week_guard_report = {
            "available": True,
            "path": str(week_guard_path),
            "pass_rows": int(week_guard["gate_status"].astype(str).eq("week_health_pass").sum())
            if "gate_status" in week_guard.columns
            else 0,
        }
    primary_execution_gate = (
        execution_selector_gate_summary[
            execution_selector_gate_summary["selector_key"].astype(str).eq(primary_selector_key)
        ].head(1).to_dict("records")[0]
        if len(execution_selector_gate_summary) and "selector_key" in execution_selector_gate_summary.columns
        else {}
    )
    primary_overlay_execution_gate = (
        overlay_execution_gate_summary[
            overlay_execution_gate_summary["selector_key"].astype(str).eq(primary_selector_key)
        ].head(1).to_dict("records")[0]
        if len(overlay_execution_gate_summary) and "selector_key" in overlay_execution_gate_summary.columns
        else {}
    )

    manifest = {
        "scope": "meta_regime_handoff_candidates",
        "context_dir": str(context_dir),
        "predictions_path": str(predictions_path),
        "summary_path": str(summary_path),
        "output_dir": str(output_dir),
        "top_frac_by_month": float(top_frac),
        "selector_count": int(len(pass_rows)),
        "available_selector_count": int(pass_rows["score_column_available"].sum()),
        "selected_rows": int(len(selected)),
        "decision_candidate_columns": int(decision_candidates.shape[1]),
        "primary_selected_rows": int(len(primary_selected)),
        "primary_policy_overlay_selected_rows": int(len(primary_overlay_selected)),
        "primary_decision_candidate_columns": int(primary_decision_candidates.shape[1]),
        "primary_policy_overlay_decision_candidate_columns": int(primary_policy_overlay_decision_candidates.shape[1]),
        "execution_decision_rows": int(len(execution_decision_candidates)),
        "primary_execution_decision_rows": int(len(primary_execution_decision_candidates)),
        "primary_policy_overlay_execution_decision_rows": int(len(primary_policy_overlay_execution_decision_candidates)),
        "execution_selected_rows": int(len(execution_selected_candidates)),
        "primary_execution_selected_rows": int(len(primary_execution_selected_candidates)),
        "primary_policy_overlay_execution_selected_rows": int(len(primary_policy_overlay_execution_selected_candidates)),
        "execution_duplicate_report": {
            "all_selected": _execution_duplicate_report(selected),
            "primary_selected": _execution_duplicate_report(primary_selected),
            "primary_policy_overlay_selected": _execution_duplicate_report(primary_overlay_selected),
        },
        "scored_rows": int(len(scored)),
        "decision_export_excluded_eval_columns": sorted(col for col in EVAL_ONLY_COLUMNS if col in selected.columns),
        "primary_decision_export_excluded_eval_columns": sorted(
            col for col in EVAL_ONLY_COLUMNS if col in primary_selected.columns
        ),
        "primary_selector": pass_rows.iloc[0].to_dict(),
        "primary_gate": primary_gate,
        "primary_policy_overlay_gate": overlay_gate_summary[
            overlay_gate_summary["selector_key"].astype(str).eq(primary_selector_key)
        ].head(1).to_dict("records")[0]
        if len(overlay_gate_summary) and "selector_key" in overlay_gate_summary.columns
        else {},
        "primary_execution_gate": primary_execution_gate,
        "primary_policy_overlay_execution_gate": primary_overlay_execution_gate,
        "policy_overlay_reports": overlay_reports,
        "side_balance_warning": (
            "primary_selector_short_share_above_75pct"
            if primary_gate.get("short_share", float("nan")) > 0.75
            else "none"
        ),
        "week_guard_report": week_guard_report,
        "selector_gate_status_counts": selector_gate_summary["candidate_handoff_status"].value_counts(dropna=False).to_dict()
        if "candidate_handoff_status" in selector_gate_summary.columns
        else {},
        "execution_selector_gate_status_counts": execution_selector_gate_summary["candidate_handoff_status"].value_counts(dropna=False).to_dict()
        if "candidate_handoff_status" in execution_selector_gate_summary.columns
        else {},
        "policy_overlay_execution_gate_status_counts": overlay_execution_gate_summary["candidate_handoff_status"].value_counts(dropna=False).to_dict()
        if "candidate_handoff_status" in overlay_execution_gate_summary.columns
        else {},
        "best_side_quality_candidate": selector_gate_summary.sort_values(
            ["side_min_ev", "mean_ev", "side_max_bad_mae"],
            ascending=[False, False, True],
        ).head(1).to_dict("records")[0]
        if len(selector_gate_summary)
        else {},
        "leakage_contract": (
            "No models or thresholds are fitted here. Selectors are taken from the month-forward OOS summary; "
            "rows are selected by fixed top fraction within each validation month using existing OOS score columns."
        ),
        "outputs": {key: str(path) for key, path in outputs.items()},
        "input_hashes": {
            "predictions_sha256": _sha256_path(predictions_path),
            "summary_sha256": _sha256_path(summary_path),
        },
    }
    with open(outputs["manifest"], "w", encoding="utf-8") as fh:
        json.dump(_json_safe(manifest), fh, indent=2, sort_keys=True)

    display_cols = [
        "selector_rank",
        "feature_set",
        "selector",
        "mean_top10_ev",
        "worst_month_top10_ev",
        "mean_top10_bad_mae",
        "max_month_top10_bad_mae",
        "mean_top10_timeout",
        "max_month_top10_timeout",
    ]
    primary_lines = [f"- `{key}`: `{_json_safe(value)}`" for key, value in primary_gate.items()]
    primary_overlay_gate = (
        overlay_gate_summary[
            overlay_gate_summary["selector_key"].astype(str).eq(primary_selector_key)
        ].head(1).to_dict("records")[0]
        if len(overlay_gate_summary) and "selector_key" in overlay_gate_summary.columns
        else {}
    )
    primary_overlay_lines = [f"- `{key}`: `{_json_safe(value)}`" for key, value in primary_overlay_gate.items()]
    primary_execution_lines = [
        f"- `{key}`: `{_json_safe(value)}`"
        for key, value in manifest["primary_execution_gate"].items()
    ]
    primary_overlay_execution_lines = [
        f"- `{key}`: `{_json_safe(value)}`"
        for key, value in manifest["primary_policy_overlay_execution_gate"].items()
    ]
    execution_duplicate_lines = [
        f"- `{key}`: `{_json_safe(value)}`"
        for key, value in manifest["execution_duplicate_report"].items()
    ]
    lines = [
        "# Meta Regime Handoff Candidates",
        "",
        manifest["leakage_contract"],
        "",
        "## Executable Decision Collapse",
        "",
        *execution_duplicate_lines,
        "",
        "## Primary Gate",
        "",
        *primary_lines,
        "",
        "## Primary Executable-Key Gate",
        "",
        *primary_execution_lines,
        "",
        "## Primary Policy Overlay Gate",
        "",
        *primary_overlay_lines,
        "",
        "## Primary Policy Overlay Executable-Key Gate",
        "",
        *primary_overlay_execution_lines,
        "",
        "## Passing Selectors",
        "",
        pass_rows[[col for col in display_cols if col in pass_rows.columns]].head(25).to_markdown(index=False),
        "",
        "## Outputs",
        "",
    ]
    for key, path in outputs.items():
        lines.append(f"- `{key}`: `{path}`")
    outputs["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-dir", type=Path, default=DEFAULT_CONTEXT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-frac", type=float, default=DEFAULT_TOP_FRAC)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_materialization(
        context_dir=args.context_dir,
        output_dir=args.output_dir,
        top_frac=float(args.top_frac),
    )
    print(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
