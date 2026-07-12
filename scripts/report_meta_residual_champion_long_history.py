#!/usr/bin/env python3
"""Long-history diagnostics for the +0.7906% residual-meta champion.

The final champion only has exact OOS predictions from April 2026 onward.
Earlier rows are retained as a causal historical-rank baseline context stream;
they are never relabeled as champion OOS evidence.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from extreme_price_movements.meta_historical_rank import HistoricalScoreRankReference

ROOT = Path("data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1")
FINAL_DIR = ROOT / (
    "lifecycle_residual_aware_ae_gmm_overlay_pca8_clip8_baseline_"
    "globaloverlay_sparse_shock_composite"
)
CONTEXT_PATH = ROOT / "cache/compact_reference_with_lifecycle.parquet"
JULY_PATH = ROOT / "july_predictions_through_20260711/july_predictions_combined.parquet"
JULY_COMPLETE_PATH = (
    ROOT / "july_complete_08_10/july_08_10_complete_predictions.parquet"
)
OUTPUT_DIR = ROOT / "champion_long_history_202503_20260710"

BASELINE = "current_reference"
PREVIOUS = "pca8_globaloverlay_parent"
LATEST = "pca8_globaloverlay_sparse_shock"


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _monday(values: pd.Series) -> pd.Series:
    day = pd.to_datetime(values, utc=True, errors="coerce").dt.floor("D")
    return day - pd.to_timedelta(day.dt.weekday.to_numpy(), unit="D")


def _fit_calibrator(frame: pd.DataFrame) -> LogisticRegression | None:
    x = pd.to_numeric(frame["score"], errors="coerce").to_numpy(dtype=np.float32)
    y = pd.to_numeric(frame["clean_exec"], errors="coerce").to_numpy(dtype=np.float32)
    rank = pd.to_numeric(frame["historical_rank"], errors="coerce").to_numpy(
        dtype=np.float32
    )
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(rank) & (rank >= 0.90)
    if int(valid.sum()) < 500 or np.unique((y[valid] >= 0.5).astype(np.int8)).size < 2:
        return None
    model = LogisticRegression(C=0.20, solver="lbfgs", max_iter=300)
    model.fit(x[valid, None], (y[valid] >= 0.5).astype(np.int8))
    return model


def _predict_probability(
    frame: pd.DataFrame, calibrators: dict[str, LogisticRegression | None]
) -> np.ndarray:
    score = (
        pd.to_numeric(frame["score"], errors="coerce")
        .fillna(0.5)
        .to_numpy(dtype=np.float32)
    )
    side = frame["side_name"].astype(str).str.lower().to_numpy()
    output = np.clip(score, 0.0, 1.0).astype(np.float32, copy=True)
    for key in np.unique(side):
        mask = side == key
        model = calibrators.get(str(key))
        if model is not None:
            output[mask] = model.predict_proba(score[mask, None])[:, 1].astype(
                np.float32
            )
    return output


def build_baseline_context() -> pd.DataFrame:
    columns = [
        "__ts__",
        "__symbol__",
        "side_name",
        "archetype_policy_key",
        "score_meta_base_soft_label",
        "ev_after_1pct",
        "clean_exec",
        "dirty_positive",
        "first_touch_bad_mae_1r",
        "full_path_bad_mae_1r",
        "timeout",
    ]
    data = pd.read_parquet(CONTEXT_PATH, columns=columns)
    data["__ts__"] = pd.to_datetime(data["__ts__"], utc=True, errors="coerce")
    data = data[
        data["__ts__"].ge(pd.Timestamp("2025-02-01", tz="UTC"))
        & data["__ts__"].lt(pd.Timestamp("2026-04-01", tz="UTC"))
    ].copy()
    data["calendar_month"] = data["__ts__"].dt.to_period("M").astype(str)
    data["score"] = pd.to_numeric(
        data["score_meta_base_soft_label"], errors="coerce"
    ).astype(np.float32)
    prior = data[data["calendar_month"].eq("2025-02")].copy()
    prior_rank_state = HistoricalScoreRankReference(score_col="score").fit(prior)
    prior["historical_rank"] = prior_rank_state.transform(prior, "score")
    parts: list[pd.DataFrame] = []
    for month in sorted(
        name
        for name in data["calendar_month"].dropna().unique()
        if "2025-03" <= str(name) <= "2026-03"
    ):
        valid = data[data["calendar_month"].eq(month)].copy()
        rank_state = HistoricalScoreRankReference(score_col="score").fit(prior)
        valid["historical_rank"] = rank_state.transform(valid, "score")
        calibrators: dict[str, LogisticRegression | None] = {}
        for side in ("long", "short"):
            side_prior = prior[prior["side_name"].astype(str).str.lower().eq(side)]
            calibrators[side] = _fit_calibrator(side_prior)
        valid["hit_probability"] = _predict_probability(valid, calibrators)
        valid["selected_for_monitor"] = valid["historical_rank"].ge(0.90)
        valid["monitor_selection_contract"] = "causal_side_historical_rank_ge_0.90"
        valid["selector"] = BASELINE
        valid["evidence_phase"] = "causal_baseline_oos_context"
        parts.append(valid)
        prior = pd.concat([prior, valid], ignore_index=True, copy=False)
    output = pd.concat(parts, ignore_index=True)
    output["week_start"] = _monday(output["__ts__"])
    return output


def _champion_apr_jun() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    raw = pd.read_parquet(FINAL_DIR / "oos_predictions_historical_rank.parquet")
    raw["__ts__"] = pd.to_datetime(raw["__ts__"], utc=True, errors="coerce")
    common = [
        "__ts__",
        "__symbol__",
        "side_name",
        "archetype_policy_key",
        "ev_after_1pct",
        "clean_exec",
        "dirty_positive",
        "first_touch_bad_mae_1r",
        "full_path_bad_mae_1r",
        "timeout",
    ]
    mappings = {
        BASELINE: (
            "score_current_reference",
            "historical_rank_current_reference",
            "hit_prob_current_reference",
        ),
        PREVIOUS: (
            "score_alternative",
            "historical_rank_alternative",
            "hit_prob_alternative",
        ),
        LATEST: (
            "score_adjusted",
            "historical_rank_adjusted",
            "hit_prob_adjusted",
        ),
    }
    versions: dict[str, pd.DataFrame] = {}
    for selector, (score_col, rank_col, probability_col) in mappings.items():
        frame = raw[common].copy()
        frame["score"] = pd.to_numeric(raw[score_col], errors="coerce").astype(
            np.float32
        )
        frame["historical_rank"] = pd.to_numeric(raw[rank_col], errors="coerce").astype(
            np.float32
        )
        frame["hit_probability"] = pd.to_numeric(
            raw[probability_col], errors="coerce"
        ).astype(np.float32)
        frame["selected_for_monitor"] = frame["historical_rank"].ge(0.90)
        frame["monitor_selection_contract"] = "causal_side_historical_rank_ge_0.90"
        frame["selector"] = selector
        frame["evidence_phase"] = "exact_champion_oos_apr_jun"
        frame["calendar_month"] = frame["__ts__"].dt.to_period("M").astype(str)
        frame["week_start"] = _monday(frame["__ts__"])
        versions[selector] = frame
    return versions[LATEST], versions


def _champion_july() -> pd.DataFrame:
    early = pd.read_parquet(JULY_PATH)
    early["__ts__"] = pd.to_datetime(early["__ts__"], utc=True, errors="coerce")
    early = early[
        early["__ts__"].lt(pd.Timestamp("2026-07-08", tz="UTC"))
        & early["outcomes_available"].fillna(False)
    ].copy()
    early["score"] = pd.to_numeric(early["score_shock_adjusted"], errors="coerce")
    early["historical_rank"] = pd.to_numeric(
        early["historical_rank_alternative"], errors="coerce"
    )
    early["hit_probability"] = pd.to_numeric(early["hit_probability"], errors="coerce")
    early["selected_for_monitor"] = (
        early["threshold_alternative_selected"].fillna(False).astype(bool)
    )
    early["monitor_selection_contract"] = "production_dynamic_ev_threshold_admission"
    complete = pd.read_parquet(JULY_COMPLETE_PATH)
    complete["__ts__"] = pd.to_datetime(complete["__ts__"], utc=True, errors="coerce")
    complete = complete[complete["ev_after_1pct"].notna()].copy()
    complete["score"] = pd.to_numeric(complete["score_shock_adjusted"], errors="coerce")
    complete["historical_rank"] = pd.to_numeric(
        complete["historical_rank"], errors="coerce"
    )
    complete["hit_probability"] = pd.to_numeric(
        complete["hit_probability"], errors="coerce"
    )
    complete["selected_for_monitor"] = (
        complete["score"]
        .groupby(pd.to_datetime(complete["__ts__"], utc=True), sort=False)
        .rank(method="first", pct=True)
        .ge(0.90)
    )
    complete["monitor_selection_contract"] = (
        "global_within_timestamp_top10_proxy_missing_dynamic_threshold_fields"
    )
    common = [
        "__ts__",
        "__symbol__",
        "side_name",
        "archetype_policy_key",
        "ev_after_1pct",
        "clean_exec",
        "dirty_positive",
        "first_touch_bad_mae_1r",
        "full_path_bad_mae_1r",
        "timeout",
        "score",
        "historical_rank",
        "hit_probability",
        "selected_for_monitor",
        "monitor_selection_contract",
    ]
    output = pd.concat(
        [early.reindex(columns=common), complete.reindex(columns=common)],
        ignore_index=True,
    ).drop_duplicates(["__ts__", "__symbol__", "side_name"], keep="last")
    output = output[output["__ts__"].lt(pd.Timestamp("2026-07-11", tz="UTC"))].copy()
    output["selector"] = LATEST
    output["evidence_phase"] = "frozen_champion_oos_july"
    output["calendar_month"] = "2026-07"
    output["week_start"] = _monday(output["__ts__"])
    return output


def _daily_cells(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame[frame["selected_for_monitor"].fillna(False).astype(bool)].copy()
    selected["day"] = pd.to_datetime(
        selected["__ts__"], utc=True, errors="coerce"
    ).dt.floor("D")
    selected["residual"] = pd.to_numeric(
        selected["clean_exec"], errors="coerce"
    ) - pd.to_numeric(selected["hit_probability"], errors="coerce")
    selected["variance"] = (
        pd.to_numeric(selected["hit_probability"], errors="coerce")
        * (1.0 - pd.to_numeric(selected["hit_probability"], errors="coerce"))
    ).clip(lower=1e-4)
    selected["negative_surprise"] = (-selected["residual"]).clip(lower=0.0)
    selected["positive_surprise"] = selected["residual"].clip(lower=0.0)
    scopes = {
        "global": [],
        "side": ["side_name"],
        "side_archetype": ["side_name", "archetype_policy_key"],
    }
    rows: list[pd.DataFrame] = []
    for scope, group_cols in scopes.items():
        keys = ["day", *group_cols]
        grouped = (
            selected.groupby(keys, dropna=False, observed=True)
            .agg(
                selected_rows=("clean_exec", "size"),
                mean_ev_after_1pct=("ev_after_1pct", "mean"),
                sum_ev_after_1pct=("ev_after_1pct", "sum"),
                clean_exec_rate=("clean_exec", "mean"),
                expected_clean_rate=("hit_probability", "mean"),
                signed_surprise=("residual", "mean"),
                mean_negative_surprise=("negative_surprise", "mean"),
                mean_positive_surprise=("positive_surprise", "mean"),
                residual_sum=("residual", "sum"),
                residual_variance=("variance", "sum"),
                evidence_phase=("evidence_phase", "last"),
                monitor_selection_contract=("monitor_selection_contract", "last"),
            )
            .reset_index()
        )
        grouped["scope"] = scope
        rows.append(grouped)
    output = pd.concat(rows, ignore_index=True, sort=False)
    output["surprise_z"] = output["residual_sum"] / np.sqrt(
        output["residual_variance"].clip(lower=1e-6)
    )
    output["two_sided_p"] = (
        output["surprise_z"]
        .abs()
        .map(lambda value: math.erfc(float(value) / math.sqrt(2.0)))
    )
    output["significance_threshold"] = 1.96
    output["minimum_support"] = 10
    output["significant"] = output["surprise_z"].abs().ge(1.96) & output[
        "selected_rows"
    ].ge(10)
    output["surprise_sign"] = np.select(
        [output["surprise_z"].ge(1.96), output["surprise_z"].le(-1.96)],
        ["positive", "negative"],
        default="not_significant",
    )
    output.loc[~output["significant"], "surprise_sign"] = "not_significant"
    return output.sort_values(
        ["day", "scope", "side_name", "archetype_policy_key"],
        kind="stable",
        na_position="first",
    )


def _autocorrelation_rows(
    daily: pd.DataFrame, *, group_by_phase: bool = False
) -> pd.DataFrame:
    source = daily[daily["scope"].eq("side_archetype")].copy()
    rows: list[dict[str, Any]] = []
    group_cols = ["side_name", "archetype_policy_key"]
    if group_by_phase:
        group_cols = ["evidence_phase", *group_cols]
    for keys, group in source.groupby(group_cols, dropna=False, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        if group_by_phase:
            phase, side, archetype = keys
        else:
            side, archetype = keys
            phase = "all_available_phases"
        group = group.sort_values("day", kind="stable")
        ev = pd.to_numeric(group["mean_ev_after_1pct"], errors="coerce")
        surprise = pd.to_numeric(group["signed_surprise"], errors="coerce")
        loss = (-ev).clip(lower=0.0)
        gain = ev.clip(lower=0.0)
        negative_surprise = pd.to_numeric(
            group["mean_negative_surprise"], errors="coerce"
        )
        positive_surprise = pd.to_numeric(
            group["mean_positive_surprise"], errors="coerce"
        )
        rows.append(
            {
                "evidence_phase": phase,
                "side_name": side,
                "archetype_policy_key": archetype,
                "days": int(len(group)),
                "selected_rows": int(group["selected_rows"].sum()),
                "mean_daily_ev": float(ev.mean()),
                "mean_ev_per_trade": float(
                    group["sum_ev_after_1pct"].sum()
                    / max(group["selected_rows"].sum(), 1)
                ),
                "negative_day_rate": float(ev.lt(0.0).mean()),
                "ev_autocorr_lag1": float(ev.autocorr(1)),
                "ev_autocorr_lag3": float(ev.autocorr(3)),
                "loss_autocorr_lag1": float(loss.autocorr(1)),
                "positive_ev_autocorr_lag1": float(gain.autocorr(1)),
                "signed_surprise_autocorr_lag1": float(surprise.autocorr(1)),
                "negative_surprise_autocorr_lag1": float(negative_surprise.autocorr(1)),
                "positive_surprise_autocorr_lag1": float(positive_surprise.autocorr(1)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["evidence_phase", "side_name", "archetype_policy_key"], kind="stable"
    )


def _version_metrics(
    frame: pd.DataFrame, selector: str, *, use_monitor_selection: bool = False
) -> dict[str, Any]:
    selected_mask = (
        frame["selected_for_monitor"].fillna(False).astype(bool)
        if use_monitor_selection
        else pd.to_numeric(frame["historical_rank"], errors="coerce").ge(0.90)
    )
    selected = frame[selected_mask].copy()
    selected["calendar_month"] = (
        pd.to_datetime(selected["__ts__"], utc=True).dt.to_period("M").astype(str)
    )
    selected["week_start"] = _monday(selected["__ts__"])
    week = selected.groupby("week_start", observed=True)["ev_after_1pct"].mean()
    month = selected.groupby("calendar_month", observed=True)["ev_after_1pct"].mean()
    daily = _daily_cells(frame)
    local = _autocorrelation_rows(daily)
    return {
        "selector": selector,
        "candidate_rows": int(len(frame)),
        "selected_rows": int(len(selected)),
        "mean_ev_after_1pct": float(selected["ev_after_1pct"].mean()),
        "clean_exec_precision": float(selected["clean_exec"].mean()),
        "full_path_bad_mae_rate": float(selected["full_path_bad_mae_1r"].mean()),
        "timeout_rate": float(selected["timeout"].mean()),
        "worst_week_ev": float(week.min()),
        "worst_month_ev": float(month.min()),
        "positive_weeks": int(week.gt(0.0).sum()),
        "weeks": int(len(week)),
        "mean_abs_signed_surprise_autocorr_lag1": float(
            local["signed_surprise_autocorr_lag1"].abs().mean()
        ),
        "mean_abs_negative_surprise_autocorr_lag1": float(
            local["negative_surprise_autocorr_lag1"].abs().mean()
        ),
        "mean_abs_positive_surprise_autocorr_lag1": float(
            local["positive_surprise_autocorr_lag1"].abs().mean()
        ),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline_context = build_baseline_context()
    champion, versions = _champion_apr_jun()
    july = _champion_july()
    monitor = pd.concat([baseline_context, champion, july], ignore_index=True)
    monitor = monitor.sort_values(
        ["__ts__", "__symbol__", "side_name"], kind="stable"
    ).reset_index(drop=True)
    monitor.to_parquet(
        OUTPUT_DIR / "monitoring_stream_202503_20260710.parquet",
        index=False,
        compression="zstd",
    )
    daily = _daily_cells(monitor)
    daily["coverage_status"] = "covered"
    expected_days = pd.date_range("2025-01-01", "2026-07-10", freq="D", tz="UTC")
    candidate_days = pd.DatetimeIndex(
        pd.to_datetime(monitor["__ts__"], utc=True).dt.floor("D").unique()
    )
    selected_days = pd.DatetimeIndex(pd.to_datetime(daily["day"], utc=True).unique())
    no_selected_days = candidate_days.difference(selected_days)
    pre_source_days = expected_days[
        expected_days < pd.Timestamp("2025-02-01", tz="UTC")
    ]
    calibration_warmup_days = expected_days[
        (expected_days >= pd.Timestamp("2025-02-01", tz="UTC"))
        & (expected_days < pd.Timestamp("2025-03-01", tz="UTC"))
    ]
    missing_days = expected_days[
        expected_days >= pd.Timestamp("2025-03-01", tz="UTC")
    ].difference(candidate_days)
    placeholder_parts: list[pd.DataFrame] = []
    if len(no_selected_days):
        placeholder_parts.append(
            pd.DataFrame(
                {
                    "day": no_selected_days,
                    "scope": "global",
                    "selected_rows": 0,
                    "significant": False,
                    "surprise_sign": "no_selected_rows",
                    "coverage_status": "candidate_rows_no_selection",
                    "evidence_phase": "causal_rank_cold_start_or_no_selection",
                    "monitor_selection_contract": "unavailable_or_no_admission",
                    "significance_threshold": 1.96,
                    "minimum_support": 10,
                }
            )
        )
    if len(missing_days):
        placeholder_parts.append(
            pd.DataFrame(
                {
                    "day": missing_days,
                    "scope": "global",
                    "selected_rows": 0,
                    "significant": False,
                    "surprise_sign": "missing_source_rows",
                    "coverage_status": "missing_source_rows",
                    "evidence_phase": "missing_source_rows",
                    "monitor_selection_contract": "unavailable",
                    "significance_threshold": 1.96,
                    "minimum_support": 10,
                }
            )
        )
    if len(pre_source_days):
        placeholder_parts.append(
            pd.DataFrame(
                {
                    "day": pre_source_days,
                    "scope": "global",
                    "selected_rows": 0,
                    "significant": False,
                    "surprise_sign": "source_unavailable",
                    "coverage_status": "source_unavailable",
                    "evidence_phase": "pre_monitoring_source",
                    "monitor_selection_contract": "unavailable",
                    "significance_threshold": 1.96,
                    "minimum_support": 10,
                }
            )
        )
    if len(calibration_warmup_days):
        placeholder_parts.append(
            pd.DataFrame(
                {
                    "day": calibration_warmup_days,
                    "scope": "global",
                    "selected_rows": 0,
                    "significant": False,
                    "surprise_sign": "calibration_warmup_no_meta_score",
                    "coverage_status": "calibration_warmup_no_meta_score",
                    "evidence_phase": "calibration_warmup",
                    "monitor_selection_contract": "unavailable",
                    "significance_threshold": 1.96,
                    "minimum_support": 10,
                }
            )
        )
    if placeholder_parts:
        placeholders = pd.DataFrame(
            pd.concat(placeholder_parts, ignore_index=True, sort=False)
        )
        daily = pd.concat([daily, placeholders], ignore_index=True, sort=False)
        daily = daily.sort_values(
            ["day", "scope", "side_name", "archetype_policy_key"],
            kind="stable",
            na_position="first",
        )
    significant = daily[daily["significant"].fillna(False).astype(bool)].copy()
    autocorrelation = _autocorrelation_rows(daily)
    autocorrelation_by_phase = _autocorrelation_rows(daily, group_by_phase=True)
    daily.to_csv(OUTPUT_DIR / "daily_surprise_calendar_all_cells.csv", index=False)
    significant.to_csv(
        OUTPUT_DIR / "significant_positive_negative_surprise_calendar.csv",
        index=False,
    )
    autocorrelation.to_csv(
        OUTPUT_DIR / "side_archetype_daily_ev_autocorrelation.csv", index=False
    )
    autocorrelation_by_phase.to_csv(
        OUTPUT_DIR / "side_archetype_daily_ev_autocorrelation_by_phase.csv",
        index=False,
    )

    comparison = pd.DataFrame(
        [_version_metrics(frame, selector) for selector, frame in versions.items()]
    )
    baseline = comparison.loc[comparison["selector"].eq(BASELINE)].iloc[0]
    previous = comparison.loc[comparison["selector"].eq(PREVIOUS)].iloc[0]
    for column in (
        "mean_ev_after_1pct",
        "worst_week_ev",
        "worst_month_ev",
        "mean_abs_signed_surprise_autocorr_lag1",
        "mean_abs_negative_surprise_autocorr_lag1",
        "mean_abs_positive_surprise_autocorr_lag1",
    ):
        comparison[f"delta_vs_baseline__{column}"] = comparison[column] - float(
            baseline[column]
        )
        comparison[f"delta_vs_previous__{column}"] = comparison[column] - float(
            previous[column]
        )
    comparison.to_csv(OUTPUT_DIR / "champion_comparison_apr_jun.csv", index=False)

    latest_extended = _version_metrics(
        pd.concat([versions[LATEST], july], ignore_index=True),
        f"{LATEST}_through_2026_07_10",
        use_monitor_selection=True,
    )
    pd.DataFrame([latest_extended]).to_csv(
        OUTPUT_DIR / "latest_champion_extended_through_20260710.csv", index=False
    )
    coverage_observed = (
        monitor.assign(day=pd.to_datetime(monitor["__ts__"], utc=True).dt.floor("D"))
        .groupby(["day", "evidence_phase"], observed=True)
        .agg(
            candidate_rows=("__symbol__", "size"),
            timestamps=("__ts__", "nunique"),
            symbols=("__symbol__", "nunique"),
            finite_outcomes=("ev_after_1pct", lambda values: int(values.notna().sum())),
        )
        .reset_index()
    )
    coverage_index = pd.DataFrame({"day": expected_days})
    coverage = coverage_index.merge(coverage_observed, on="day", how="left")
    coverage["coverage_status"] = "missing_source_rows"
    coverage.loc[
        coverage["day"].lt(pd.Timestamp("2025-02-01", tz="UTC")),
        "coverage_status",
    ] = "source_unavailable"
    coverage.loc[
        coverage["day"].between(
            pd.Timestamp("2025-02-01", tz="UTC"),
            pd.Timestamp("2025-02-28", tz="UTC"),
        ),
        "coverage_status",
    ] = "calibration_warmup_no_meta_score"
    coverage.loc[coverage["candidate_rows"].fillna(0).gt(0), "coverage_status"] = (
        "covered"
    )
    coverage.to_csv(OUTPUT_DIR / "daily_coverage.csv", index=False)
    manifest = {
        "schema": "meta_residual_champion_long_history_v1",
        "comparison_period": "2026-04-01 through 2026-06-30 exact OOS",
        "calendar_period": "2025-01-01 through 2026-07-10",
        "monitoring_period": "2025-03-01 through 2026-07-10",
        "baseline_context_period": "2025-03-01 through 2026-03-31",
        "champion_exact_oos_period": "2026-04-01 through 2026-06-30",
        "champion_frozen_july_period": "2026-07-01 through 2026-07-10",
        "selection_contract": (
            "side-aware causal historical score percentile >= 0.90; no forced "
            "selection within weak timestamps"
        ),
        "cost_contract": "ev_after_1pct includes 1% round-trip cost",
        "significance_contract": (
            "two-sided calibration-residual z test, |z| >= 1.96, minimum 10 "
            "selected rows per day/cell"
        ),
        "important_limitation": (
            "Rows before April 2026 are causal baseline OOS context, not a "
            "retrospective champion backtest. This avoids applying future-selected "
            "champion coefficients to 2025."
        ),
        "comparison": comparison.to_dict(orient="records"),
        "latest_extended": latest_extended,
        "significant_calendar_rows": int(len(significant)),
        "autocorrelation_rows": int(len(autocorrelation)),
        "coverage_days_expected": int(len(expected_days)),
        "coverage_days_observed": int(coverage["candidate_rows"].notna().sum()),
        "pre_monitoring_source_days": [str(day) for day in pre_source_days],
        "calibration_warmup_days": [str(day) for day in calibration_warmup_days],
        "candidate_days_without_selected_rows": [str(day) for day in no_selected_days],
        "missing_source_days": [str(day) for day in missing_days],
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(_safe_json(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(_safe_json(manifest), indent=2), flush=True)


if __name__ == "__main__":
    main()
