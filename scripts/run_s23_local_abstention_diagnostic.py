#!/usr/bin/env python3
"""S23 leakage-safe local abstention diagnostic for Gate 3 source streams."""

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


DEFAULT_REPORT_DIR = Path(
    "data_perp/reports/gmm_cluster_policy_smoke_20260702_wide_sidebalanced"
)
DEFAULT_OUTPUT_SUBDIR = "s23_local_abstention_spread170_top05_ctx_v1"
DEFAULT_LEDGER_PATH = (
    DEFAULT_REPORT_DIR
    / "gmm_train_meta_path_filter_smoke_s21_posclean_spread170_top05_ctx_v1"
    / "base_candidate_streams"
    / "label_feature_store_model_smoke_candidate_ledger.csv"
)
DEFAULT_SPREAD_BASELINE_PATH = Path(
    "data_perp/exchanges/krakenfutures/spread_model/per_asset_spread_baseline_latest.csv"
)
DEFAULT_FEATURE_DIR = Path("data_perp/features/20260629_050000")
CTX_BUCKET_COLUMNS = (
    "ctx_state_spectral_top3_reconstruction_error",
    "ctx_q_iqr__bars_in_high_vol_state_log_norm",
    "ctx_q_tail_width__bars_in_high_vol_state_log_norm",
)
POLICY_BUCKET_COLUMNS = (
    "side_bucket",
    "spread_bucket",
    "liquidity_bucket",
    "ctx_state_spectral_top3_reconstruction_error_bucket",
    "ctx_q_iqr__bars_in_high_vol_state_log_norm_bucket",
    "ctx_q_tail_width__bars_in_high_vol_state_log_norm_bucket",
)
NON_SIDE_BUCKET_COLUMNS = tuple(c for c in POLICY_BUCKET_COLUMNS if c != "side_bucket")
DEFAULT_THRESHOLDS = {
    "min_mean_u": 0.0,
    "max_bad_mae_1r_rate": 0.65,
    "max_timeout_rate": 0.15,
    "min_clean_dirty_gap": 0.0,
    "min_bucket_final_oracle_recall": 0.0,
    "min_selected_rows": 20,
    "min_oracle_hits": 1,
    "min_prior_months": 1,
    "require_prior_month_stability": False,
    "max_side_share": 0.70,
    "min_final_oracle_recall": 0.02,
    "min_fold_selected_rows": 5,
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if np.isfinite(out) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _safe_mean(values: Any) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    return float(series.mean()) if series.notna().any() else float("nan")


def _safe_min(values: Any) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    return float(series.min()) if series.notna().any() else float("nan")


def _safe_max(values: Any) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    return float(series.max()) if series.notna().any() else float("nan")


def _symbol_to_feature_path(symbol: str, feature_dir: Path) -> Path:
    return feature_dir / f"symbol={str(symbol).replace('/', '_')}.parquet"


def _load_liquidity_feature(
    ledger: pd.DataFrame,
    *,
    feature_dir: Path,
    liquidity_column: str,
) -> pd.Series:
    out = pd.Series(np.nan, index=ledger.index, dtype=np.float32)
    if liquidity_column in ledger.columns:
        return pd.to_numeric(ledger[liquidity_column], errors="coerce").astype(np.float32)
    timestamps = pd.to_datetime(ledger["timestamp"], utc=True, errors="coerce")
    for symbol, idx in ledger.groupby("symbol", sort=False).groups.items():
        path = _symbol_to_feature_path(str(symbol), feature_dir)
        if not path.exists():
            continue
        try:
            features = pd.read_parquet(path, columns=[liquidity_column])
        except Exception:
            continue
        if not isinstance(features.index, pd.DatetimeIndex):
            continue
        feature_index = pd.to_datetime(features.index, utc=True, errors="coerce")
        values = pd.Series(
            pd.to_numeric(features[liquidity_column], errors="coerce").to_numpy(
                dtype=np.float32,
                copy=False,
            ),
            index=feature_index,
        )
        local_ts = timestamps.loc[idx]
        out.loc[idx] = values.reindex(local_ts).to_numpy(dtype=np.float32, copy=False)
    return out


def _add_static_spread(
    ledger: pd.DataFrame,
    *,
    spread_baseline_path: Path,
    spread_column: str,
) -> pd.DataFrame:
    if spread_column in ledger.columns:
        return ledger
    if not spread_baseline_path.exists():
        ledger[spread_column] = np.nan
        return ledger
    spread = pd.read_csv(spread_baseline_path)
    if "symbol" not in spread.columns or spread_column not in spread.columns:
        ledger[spread_column] = np.nan
        return ledger
    keep = spread[["symbol", spread_column]].drop_duplicates("symbol")
    return ledger.merge(keep, on="symbol", how="left")


def _bucket_from_train(
    train_values: pd.Series,
    values: pd.Series,
    *,
    bins: int,
) -> pd.Series:
    train_num = pd.to_numeric(train_values, errors="coerce").to_numpy(dtype=np.float64)
    value_num = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    finite_train = train_num[np.isfinite(train_num)]
    out = np.full(len(value_num), -1, dtype=np.int16)
    finite_values = np.isfinite(value_num)
    if len(finite_train) < max(20, bins * 8):
        out[finite_values] = 0
        return pd.Series(out, index=values.index, dtype=np.int16)
    edges = np.unique(np.nanquantile(finite_train, np.linspace(0.0, 1.0, bins + 1)[1:-1]))
    if len(edges) == 0:
        out[finite_values] = 0
    else:
        out[finite_values] = np.searchsorted(
            edges,
            value_num[finite_values],
            side="right",
        ).astype(np.int16)
    return pd.Series(out, index=values.index, dtype=np.int16)


def _oracle_total(frame: pd.DataFrame) -> int:
    if frame.empty or "oracle_rows_total" not in frame.columns:
        return 0
    totals = pd.to_numeric(frame["oracle_rows_total"], errors="coerce")
    if "period" in frame.columns:
        return int(totals.groupby(frame["period"].astype(str), sort=False).max().fillna(0).sum())
    return int(totals.max() if totals.notna().any() else 0)


def _metric_row(frame: pd.DataFrame) -> dict[str, Any]:
    selected_rows = int(len(frame))
    side = pd.to_numeric(frame.get("side"), errors="coerce")
    long_rows = int((side > 0.0).sum()) if selected_rows else 0
    short_rows = int((side < 0.0).sum()) if selected_rows else 0
    max_side_share = (
        max(long_rows, short_rows) / float(selected_rows) if selected_rows else float("nan")
    )
    oracle_hits = int(frame.get("oracle_top", pd.Series(False, index=frame.index)).astype(bool).sum())
    oracle_total = _oracle_total(frame)
    score = pd.to_numeric(frame.get("selector_score"), errors="coerce")
    clean_mask = frame.get("clean_positive", pd.Series(False, index=frame.index)).astype(bool)
    dirty_mask = frame.get("dirty_positive", pd.Series(False, index=frame.index)).astype(bool)
    clean_score = _safe_mean(score[clean_mask])
    dirty_score = _safe_mean(score[dirty_mask])
    return {
        "selected_rows": selected_rows,
        "symbol_count": int(frame["symbol"].nunique()) if selected_rows and "symbol" in frame else 0,
        "selected_long_rows": long_rows,
        "selected_short_rows": short_rows,
        "max_side_share": max_side_share,
        "mean_u": _safe_mean(frame.get("u_policy_net")),
        "bad_mae_1r_rate": _safe_mean(frame.get("bad_mae_1r")),
        "timeout_rate": _safe_mean(
            pd.to_numeric(frame.get("is_timeout"), errors="coerce") > 0.5
        ),
        "clean_positive_rate": _safe_mean(frame.get("clean_positive")),
        "dirty_positive_rate": _safe_mean(frame.get("dirty_positive")),
        "oracle_hit_rows": oracle_hits,
        "oracle_rows_total": oracle_total,
        "final_oracle_recall": (
            float(oracle_hits / oracle_total) if oracle_total > 0 else float("nan")
        ),
        "mean_score_clean_positive": clean_score,
        "mean_score_dirty_positive": dirty_score,
        "clean_dirty_score_gap": (
            float(clean_score - dirty_score)
            if np.isfinite(clean_score) and np.isfinite(dirty_score)
            else float("nan")
        ),
    }


def _period_worst_month_u(frame: pd.DataFrame) -> float:
    if frame.empty:
        return float("nan")
    mean_by_period = frame.groupby(frame["period"].astype(str), sort=False)["u_policy_net"].mean()
    return float(mean_by_period.min()) if len(mean_by_period) else float("nan")


def _build_period_buckets(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    bins: int,
    spread_column: str,
    liquidity_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_b = train.copy()
    valid_b = valid.copy()
    train_b["side_bucket"] = np.where(
        pd.to_numeric(train_b["side"], errors="coerce").fillna(1.0) < 0.0,
        "short",
        "long",
    )
    valid_b["side_bucket"] = np.where(
        pd.to_numeric(valid_b["side"], errors="coerce").fillna(1.0) < 0.0,
        "short",
        "long",
    )
    bucket_sources = {
        "spread_bucket": spread_column,
        "liquidity_bucket": liquidity_column,
        "ctx_state_spectral_top3_reconstruction_error_bucket": (
            "ctx_state_spectral_top3_reconstruction_error"
        ),
        "ctx_q_iqr__bars_in_high_vol_state_log_norm_bucket": (
            "ctx_q_iqr__bars_in_high_vol_state_log_norm"
        ),
        "ctx_q_tail_width__bars_in_high_vol_state_log_norm_bucket": (
            "ctx_q_tail_width__bars_in_high_vol_state_log_norm"
        ),
    }
    for bucket_col, source_col in bucket_sources.items():
        if source_col not in train_b.columns:
            train_b[bucket_col] = -1
            valid_b[bucket_col] = -1
            continue
        train_b[bucket_col] = _bucket_from_train(
            train_b[source_col],
            train_b[source_col],
            bins=bins,
        )
        valid_b[bucket_col] = _bucket_from_train(
            train_b[source_col],
            valid_b[source_col],
            bins=bins,
        )
    train_b["diagnostic_month_bucket"] = train_b["period"].astype(str)
    valid_b["diagnostic_month_bucket"] = valid_b["period"].astype(str)
    return train_b, valid_b


def _active_bucket_policy(
    train: pd.DataFrame,
    *,
    valid_period: str,
    thresholds: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    diagnostics: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    grouped = train.groupby(list(POLICY_BUCKET_COLUMNS), sort=False, dropna=False)
    parent_side_share: dict[tuple[Any, ...], float] = {}
    for parent_key, parent in train.groupby(list(NON_SIDE_BUCKET_COLUMNS), sort=False, dropna=False):
        parent_side_share[tuple(parent_key if isinstance(parent_key, tuple) else (parent_key,))] = (
            _metric_row(parent)["max_side_share"]
        )
    for key, group in grouped:
        key_tuple = tuple(key if isinstance(key, tuple) else (key,))
        month_metrics = []
        for period, period_group in group.groupby(group["period"].astype(str), sort=False):
            row = {
                "valid_period": valid_period,
                "train_period": str(period),
                **dict(zip(POLICY_BUCKET_COLUMNS, key_tuple, strict=False)),
                **_metric_row(period_group),
            }
            diagnostics.append(row)
            month_metrics.append(row)
        metrics = _metric_row(group)
        metrics["worst_month_mean_u"] = _period_worst_month_u(group)
        metrics["prior_months"] = int(len(month_metrics))
        metrics["positive_prior_months"] = int(
            sum(
                float(row.get("mean_u", float("nan"))) > float(thresholds["min_mean_u"])
                for row in month_metrics
            )
        )
        metrics["max_month_bad_mae_1r_rate"] = _safe_max(
            [row.get("bad_mae_1r_rate") for row in month_metrics]
        )
        metrics["max_month_timeout_rate"] = _safe_max(
            [row.get("timeout_rate") for row in month_metrics]
        )
        metrics["min_month_clean_dirty_score_gap"] = _safe_min(
            [row.get("clean_dirty_score_gap") for row in month_metrics]
        )
        metrics["min_month_final_oracle_recall"] = _safe_min(
            [row.get("final_oracle_recall") for row in month_metrics]
        )
        parent_key = tuple(
            value
            for column, value in zip(POLICY_BUCKET_COLUMNS, key_tuple, strict=False)
            if column != "side_bucket"
        )
        metrics["prior_parent_max_side_share"] = parent_side_share.get(
            parent_key,
            float("nan"),
        )
        month_stable = (
            metrics["prior_months"] >= int(thresholds["min_prior_months"])
            and metrics["positive_prior_months"] == metrics["prior_months"]
            and metrics["worst_month_mean_u"] > float(thresholds["min_mean_u"])
            and metrics["max_month_bad_mae_1r_rate"]
            <= float(thresholds["max_bad_mae_1r_rate"])
            and metrics["max_month_timeout_rate"] <= float(thresholds["max_timeout_rate"])
            and metrics["min_month_clean_dirty_score_gap"]
            > float(thresholds["min_clean_dirty_gap"])
        )
        if float(thresholds.get("min_bucket_final_oracle_recall", 0.0)) > 0.0:
            month_stable = month_stable and (
                metrics["min_month_final_oracle_recall"]
                >= float(thresholds["min_bucket_final_oracle_recall"])
            )
        active = (
            metrics["mean_u"] > float(thresholds["min_mean_u"])
            and metrics["bad_mae_1r_rate"] <= float(thresholds["max_bad_mae_1r_rate"])
            and metrics["timeout_rate"] <= float(thresholds["max_timeout_rate"])
            and metrics["clean_dirty_score_gap"] > float(thresholds["min_clean_dirty_gap"])
            and metrics["final_oracle_recall"]
            >= float(thresholds.get("min_bucket_final_oracle_recall", 0.0))
            and metrics["selected_rows"] >= int(thresholds["min_selected_rows"])
            and metrics["oracle_hit_rows"] >= int(thresholds["min_oracle_hits"])
            and (
                (not bool(thresholds.get("require_prior_month_stability", False)))
                or month_stable
            )
        )
        fail_reasons = []
        if not metrics["mean_u"] > float(thresholds["min_mean_u"]):
            fail_reasons.append("mean_u")
        if not metrics["bad_mae_1r_rate"] <= float(thresholds["max_bad_mae_1r_rate"]):
            fail_reasons.append("bad_mae")
        if not metrics["timeout_rate"] <= float(thresholds["max_timeout_rate"]):
            fail_reasons.append("timeout")
        if not metrics["clean_dirty_score_gap"] > float(thresholds["min_clean_dirty_gap"]):
            fail_reasons.append("clean_dirty_gap")
        if not metrics["final_oracle_recall"] >= float(
            thresholds.get("min_bucket_final_oracle_recall", 0.0)
        ):
            fail_reasons.append("bucket_oracle_recall")
        if not metrics["selected_rows"] >= int(thresholds["min_selected_rows"]):
            fail_reasons.append("selected_rows")
        if not metrics["oracle_hit_rows"] >= int(thresholds["min_oracle_hits"]):
            fail_reasons.append("oracle_hits")
        if bool(thresholds.get("require_prior_month_stability", False)) and not month_stable:
            fail_reasons.append("prior_month_stability")
        policy_rows.append(
            {
                "valid_period": valid_period,
                **dict(zip(POLICY_BUCKET_COLUMNS, key_tuple, strict=False)),
                **metrics,
                "prior_month_stable": bool(month_stable),
                "active": bool(active),
                "fail_reasons": ",".join(fail_reasons),
            }
        )
    return pd.DataFrame(diagnostics), pd.DataFrame(policy_rows)


def _apply_policy(valid: pd.DataFrame, policy: pd.DataFrame) -> pd.DataFrame:
    if valid.empty or policy.empty:
        out = valid.iloc[:0].copy()
        out["s23_bucket_active"] = False
        return out
    active_keys = policy.loc[policy["active"].astype(bool), list(POLICY_BUCKET_COLUMNS)]
    if active_keys.empty:
        out = valid.iloc[:0].copy()
        out["s23_bucket_active"] = False
        return out
    keyed = valid.merge(
        active_keys.drop_duplicates(),
        on=list(POLICY_BUCKET_COLUMNS),
        how="inner",
    )
    keyed["s23_bucket_active"] = True
    return keyed


def _side_application_metrics(
    *,
    selector: str,
    period: str,
    accepted: pd.DataFrame,
    valid: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    accepted_side = pd.to_numeric(
        accepted.get("side", pd.Series(dtype=np.float32)),
        errors="coerce",
    )
    valid_side = pd.to_numeric(
        valid.get("side", pd.Series(dtype=np.float32)),
        errors="coerce",
    )
    for side_name, accepted_mask, valid_mask in (
        ("long", accepted_side > 0.0, valid_side > 0.0),
        ("short", accepted_side < 0.0, valid_side < 0.0),
    ):
        side_accepted = accepted.loc[accepted_mask].copy() if not accepted.empty else accepted.copy()
        side_valid = valid.loc[valid_mask].copy() if not valid.empty else valid.copy()
        metrics = _metric_row(side_accepted)
        side_oracle_total = _oracle_total(side_valid)
        metrics["oracle_rows_total"] = side_oracle_total
        metrics["final_oracle_recall"] = (
            float(metrics["oracle_hit_rows"] / side_oracle_total)
            if side_oracle_total > 0
            else float("nan")
        )
        rows.append(
            {
                "selector_variant": selector,
                "period": period,
                "side_bucket": side_name,
                **metrics,
            }
        )
    return rows


def run_s23(
    *,
    ledger_path: Path,
    output_dir: Path,
    spread_baseline_path: Path,
    feature_dir: Path,
    bins: int,
    thresholds: dict[str, float],
    spread_column: str,
    liquidity_column: str,
    run_family: str = "s23",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = pd.read_csv(ledger_path)
    if ledger.empty:
        raise ValueError(f"empty candidate ledger: {ledger_path}")
    ledger["period"] = ledger["period"].astype(str)
    ledger["timestamp"] = pd.to_datetime(ledger["timestamp"], utc=True, errors="coerce")
    ledger = _add_static_spread(
        ledger,
        spread_baseline_path=spread_baseline_path,
        spread_column=spread_column,
    )
    ledger[liquidity_column] = _load_liquidity_feature(
        ledger,
        feature_dir=feature_dir,
        liquidity_column=liquidity_column,
    )
    for col in ("u_policy_net", "bad_mae_1r", "is_timeout", "selector_score"):
        if col in ledger.columns:
            ledger[col] = pd.to_numeric(ledger[col], errors="coerce").astype(np.float32)

    local_rows: list[pd.DataFrame] = []
    policy_rows: list[pd.DataFrame] = []
    application_rows: list[dict[str, Any]] = []
    application_by_side_rows: list[dict[str, Any]] = []
    accepted_rows: list[pd.DataFrame] = []
    periods = sorted(ledger["period"].dropna().unique())
    for selector, selector_rows in ledger.groupby("selector_variant", sort=False):
        selector_rows = selector_rows.reset_index(drop=True)
        for valid_period in periods[1:]:
            train = selector_rows[selector_rows["period"] < valid_period].copy()
            valid = selector_rows[selector_rows["period"].eq(valid_period)].copy()
            if valid.empty:
                continue
            if train.empty:
                application_rows.append(
                    {
                        "selector_variant": selector,
                        "period": valid_period,
                        "eval_status": "insufficient_prior_rows",
                        **_metric_row(valid.iloc[:0]),
                    }
                )
                continue
            train_b, valid_b = _build_period_buckets(
                train,
                valid,
                bins=bins,
                spread_column=spread_column,
                liquidity_column=liquidity_column,
            )
            local_diag, policy = _active_bucket_policy(
                train_b,
                valid_period=valid_period,
                thresholds=thresholds,
            )
            if not local_diag.empty:
                local_diag.insert(0, "selector_variant", selector)
                local_rows.append(local_diag)
            if not policy.empty:
                policy.insert(0, "selector_variant", selector)
                policy_rows.append(policy)
            accepted = _apply_policy(valid_b, policy)
            if not accepted.empty:
                accepted.insert(0, "s23_valid_period", valid_period)
                accepted_rows.append(accepted)
            app_metrics = _metric_row(accepted)
            valid_oracle_total = _oracle_total(valid_b)
            app_metrics["oracle_rows_total"] = valid_oracle_total
            app_metrics["final_oracle_recall"] = (
                float(app_metrics["oracle_hit_rows"] / valid_oracle_total)
                if valid_oracle_total > 0
                else float("nan")
            )
            app_metrics["worst_month_mean_u"] = _period_worst_month_u(accepted)
            application_rows.append(
                {
                    "selector_variant": selector,
                    "period": valid_period,
                    "eval_status": "ok",
                    "candidate_rows": int(len(valid_b)),
                    "active_bucket_count": int(policy["active"].sum()) if not policy.empty else 0,
                    "policy_bucket_count": int(len(policy)),
                    **app_metrics,
                    "gate_mean_u_pass": bool(app_metrics["mean_u"] > thresholds["min_mean_u"]),
                    "gate_bad_mae_candidate_pass": bool(
                        app_metrics["bad_mae_1r_rate"] <= thresholds["max_bad_mae_1r_rate"]
                    ),
                    "gate_timeout_candidate_pass": bool(
                        app_metrics["timeout_rate"] <= thresholds["max_timeout_rate"]
                    ),
                    "gate_min_rows_pass": bool(
                        app_metrics["selected_rows"] >= thresholds["min_fold_selected_rows"]
                    ),
                    "gate_side_share_pass": bool(
                        app_metrics["max_side_share"] <= thresholds["max_side_share"]
                    ),
                }
            )
            application_by_side_rows.extend(
                _side_application_metrics(
                    selector=selector,
                    period=valid_period,
                    accepted=accepted,
                    valid=valid_b,
                )
            )

    local = pd.concat(local_rows, ignore_index=True) if local_rows else pd.DataFrame()
    policy = pd.concat(policy_rows, ignore_index=True) if policy_rows else pd.DataFrame()
    application = pd.DataFrame(application_rows)
    application_by_side = pd.DataFrame(application_by_side_rows)
    accepted = pd.concat(accepted_rows, ignore_index=True) if accepted_rows else pd.DataFrame()
    readiness_rows = []
    for selector, group in application.groupby("selector_variant", sort=False):
        ok = group[group["eval_status"].eq("ok")].copy()
        oracle_hits = int(pd.to_numeric(ok["oracle_hit_rows"], errors="coerce").fillna(0).sum())
        oracle_total = int(pd.to_numeric(ok["oracle_rows_total"], errors="coerce").fillna(0).sum())
        selected_rows = pd.to_numeric(ok["selected_rows"], errors="coerce")
        mean_u = pd.to_numeric(ok["mean_u"], errors="coerce")
        bad = pd.to_numeric(ok["bad_mae_1r_rate"], errors="coerce")
        timeout = pd.to_numeric(ok["timeout_rate"], errors="coerce")
        side_share = pd.to_numeric(ok["max_side_share"], errors="coerce")
        row = {
            "selector_variant": selector,
            "evaluable_months": int(len(ok)),
            "positive_months": int((mean_u > thresholds["min_mean_u"]).sum()),
            "no_trade_months": int((selected_rows <= 0).sum()),
            "mean_u": _safe_mean(mean_u),
            "worst_month_mean_u": _safe_min(mean_u),
            "bad_mae_1r_rate": _safe_mean(bad),
            "timeout_rate": _safe_mean(timeout),
            "max_month_bad_mae_1r_rate": _safe_max(bad),
            "max_month_timeout_rate": _safe_max(timeout),
            "final_oracle_recall": (
                float(oracle_hits / oracle_total) if oracle_total > 0 else float("nan")
            ),
            "oracle_hit_rows": oracle_hits,
            "oracle_rows_total": oracle_total,
            "mean_selected_rows": _safe_mean(selected_rows),
            "min_selected_rows": int(selected_rows.min()) if selected_rows.notna().any() else 0,
            "max_selected_side_share": _safe_max(side_share),
            "symbol_count": int(accepted.loc[
                accepted["selector_variant"].eq(selector),
                "symbol",
            ].nunique())
            if not accepted.empty
            else 0,
        }
        row["gate3_candidate_ready"] = (
            row["evaluable_months"] > 0
            and row["positive_months"] == row["evaluable_months"]
            and row["no_trade_months"] == 0
            and row["mean_u"] > thresholds["min_mean_u"]
            and row["worst_month_mean_u"] > thresholds["min_mean_u"]
            and row["bad_mae_1r_rate"] <= thresholds["max_bad_mae_1r_rate"]
            and row["timeout_rate"] <= thresholds["max_timeout_rate"]
            and row["final_oracle_recall"] >= thresholds["min_final_oracle_recall"]
            and row["min_selected_rows"] >= thresholds["min_fold_selected_rows"]
            and row["max_selected_side_share"] <= thresholds["max_side_share"]
        )
        readiness_rows.append(row)
    readiness = pd.DataFrame(readiness_rows).sort_values(
        ["gate3_candidate_ready", "final_oracle_recall", "mean_u"],
        ascending=[False, False, False],
    )

    suffix = "".join(ch if ch.isalnum() else "_" for ch in str(run_family).lower()).strip("_")
    suffix = suffix or "s23"
    paths = {
        "local_bucket_diagnostics": output_dir / f"local_bucket_diagnostics_{suffix}.csv",
        "prior_fold_bucket_policy": output_dir / f"prior_fold_bucket_policy_{suffix}.csv",
        "oos_bucket_application": output_dir / f"oos_bucket_application_{suffix}.csv",
        "oos_bucket_application_by_side": (
            output_dir / f"oos_bucket_application_by_side_{suffix}.csv"
        ),
        "readiness": output_dir / f"gate3_source_readiness_{suffix}.csv",
        "accepted_rows": output_dir / f"accepted_rows_{suffix}.parquet",
        "markdown": output_dir / f"gate3_source_readiness_{suffix}.md",
        "manifest": output_dir / "manifest.json",
    }
    local.to_csv(paths["local_bucket_diagnostics"], index=False)
    policy.to_csv(paths["prior_fold_bucket_policy"], index=False)
    application.to_csv(paths["oos_bucket_application"], index=False)
    application_by_side.to_csv(paths["oos_bucket_application_by_side"], index=False)
    readiness.to_csv(paths["readiness"], index=False)
    accepted.to_parquet(paths["accepted_rows"], index=False)

    best = readiness.iloc[0].to_dict() if not readiness.empty else {}
    report = _render_markdown(
        readiness=readiness,
        application=application,
        application_by_side=application_by_side,
        best=best,
        thresholds=thresholds,
        paths=paths,
        run_family=suffix.upper(),
    )
    paths["markdown"].write_text(report, encoding="utf-8")
    manifest = {
        "run_family": suffix,
        "status": "pass" if bool(best.get("gate3_candidate_ready", False)) else "fail",
        "ledger_path": str(ledger_path),
        "output_dir": str(output_dir),
        "spread_baseline_path": str(spread_baseline_path),
        "feature_dir": str(feature_dir),
        "spread_column": spread_column,
        "liquidity_column": liquidity_column,
        "bins": int(bins),
        "thresholds": thresholds,
        "best_selector": best,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def _render_markdown(
    *,
    readiness: pd.DataFrame,
    application: pd.DataFrame,
    application_by_side: pd.DataFrame,
    best: dict[str, Any],
    thresholds: dict[str, float],
    paths: dict[str, Path],
    run_family: str,
) -> str:
    lines: list[str] = [
        f"# {run_family} Local Abstention Gate 3 Source Readiness",
        "",
        f"Generated: {pd.Timestamp.utcnow().date().isoformat()}",
        "",
        "Scope: source-side local abstention diagnostic. Bucket policies are learned from prior periods only and applied to the next OOS period. This is not a train_meta run, simple_policy run, or frozen replay.",
        "",
        "Promotion keys: side, spread bucket, liquidity bucket, and context buckets. Month is used as a prior-fold stability diagnostic, not as an OOS lookahead key.",
        "",
        "## Outputs",
        "",
    ]
    for key in (
        "local_bucket_diagnostics",
        "prior_fold_bucket_policy",
        "oos_bucket_application",
        "oos_bucket_application_by_side",
        "readiness",
        "accepted_rows",
    ):
        lines.append(f"- `{paths[key]}`")
    lines.extend(["", "## Gate Answers", ""])
    final_recall = float(best.get("final_oracle_recall", float("nan")))
    lines.append(
        f"- Did local abstention preserve final oracle recall >= 2%? "
        f"{'yes' if final_recall >= thresholds['min_final_oracle_recall'] else 'no'} "
        f"(`{final_recall:.4%}` best)."
    )
    bad = float(best.get("bad_mae_1r_rate", float("nan")))
    lines.append(
        f"- Did it reduce bad-MAE versus S21/S21-context? "
        f"Best S23 bad-MAE is `{bad:.2%}`; compare against repaired S21-context best-mean row at about `52.85%` and lowest row at about `49.72%`."
    )
    june = application[application["period"].astype(str).eq("2026-06")]
    june_trade = int((pd.to_numeric(june.get("selected_rows"), errors="coerce") > 0).sum())
    lines.append(
        f"- Did June stop being the contradictory failure bucket? "
        f"{'partially' if june_trade else 'no'}; June traded in `{june_trade}` selector applications, but readiness depends on the selected best row below."
    )
    no_trade = int(best.get("no_trade_months", 0) or 0)
    lines.append(
        f"- Did all evaluable OOS months trade? {'yes' if no_trade == 0 else 'no'} "
        f"(`{no_trade}` no-trade months for best row)."
    )
    side = float(best.get("max_selected_side_share", float("nan")))
    rows = int(best.get("min_selected_rows", 0) or 0)
    lines.append(
        f"- Did side/share and breadth floors remain acceptable? "
        f"{'yes' if side <= thresholds['max_side_share'] and rows >= thresholds['min_fold_selected_rows'] else 'no'} "
        f"(max side share `{side:.2%}`, min monthly rows `{rows}`)."
    )
    lines.extend(["", "## Readiness Summary", ""])
    if readiness.empty:
        lines.append("No readiness rows were produced.")
    else:
        cols = [
            "selector_variant",
            "gate3_candidate_ready",
            "evaluable_months",
            "positive_months",
            "no_trade_months",
            "mean_u",
            "worst_month_mean_u",
            "bad_mae_1r_rate",
            "timeout_rate",
            "final_oracle_recall",
            "mean_selected_rows",
            "min_selected_rows",
            "max_selected_side_share",
        ]
        lines.append(readiness[cols].head(20).to_markdown(index=False, floatfmt=".6f"))
    lines.extend(["", "## OOS Month-Side Application", ""])
    if application_by_side.empty:
        lines.append("No side application rows were produced.")
    else:
        side_cols = [
            "selector_variant",
            "period",
            "side_bucket",
            "selected_rows",
            "mean_u",
            "bad_mae_1r_rate",
            "timeout_rate",
            "final_oracle_recall",
        ]
        lines.append(
            application_by_side[side_cols].head(40).to_markdown(index=False, floatfmt=".6f")
        )
    lines.extend(["", "## Stop Rule", ""])
    if final_recall < thresholds["min_final_oracle_recall"]:
        lines.append(
            "S23 remains below the `2%` final oracle recall bar. Per the stop rule, do not tune local abstention thresholds harder; the next Gate 3 repair should be S24, a true broad path-first source objective."
        )
    else:
        lines.append(
            "S23 preserved the `2%` final oracle recall bar. The next step would be to package the active bucket policy as a source-conditioned candidate stream and rerun Gate 3 readiness."
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-path", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR / DEFAULT_OUTPUT_SUBDIR,
    )
    parser.add_argument("--spread-baseline-path", type=Path, default=DEFAULT_SPREAD_BASELINE_PATH)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--spread-column", type=str, default="p75_spread_bps")
    parser.add_argument("--liquidity-column", type=str, default="log_quote_volume")
    parser.add_argument("--bins", type=int, default=3)
    parser.add_argument("--min-selected-rows", type=int, default=int(DEFAULT_THRESHOLDS["min_selected_rows"]))
    parser.add_argument("--min-oracle-hits", type=int, default=int(DEFAULT_THRESHOLDS["min_oracle_hits"]))
    parser.add_argument("--max-bad-mae-rate", type=float, default=DEFAULT_THRESHOLDS["max_bad_mae_1r_rate"])
    parser.add_argument("--max-timeout-rate", type=float, default=DEFAULT_THRESHOLDS["max_timeout_rate"])
    parser.add_argument("--min-clean-dirty-gap", type=float, default=DEFAULT_THRESHOLDS["min_clean_dirty_gap"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    thresholds = dict(DEFAULT_THRESHOLDS)
    thresholds.update(
        {
            "min_selected_rows": int(args.min_selected_rows),
            "min_oracle_hits": int(args.min_oracle_hits),
            "max_bad_mae_1r_rate": float(args.max_bad_mae_rate),
            "max_timeout_rate": float(args.max_timeout_rate),
            "min_clean_dirty_gap": float(args.min_clean_dirty_gap),
        }
    )
    manifest = run_s23(
        ledger_path=args.ledger_path,
        output_dir=args.output_dir,
        spread_baseline_path=args.spread_baseline_path,
        feature_dir=args.feature_dir,
        bins=int(args.bins),
        thresholds=thresholds,
        spread_column=str(args.spread_column),
        liquidity_column=str(args.liquidity_column),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0 if manifest["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
