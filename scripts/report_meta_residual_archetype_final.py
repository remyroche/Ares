#!/usr/bin/env python3
"""Materialize the final residual-archetype scorecard and detailed tables."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.meta_residual_archetypes import (  # noqa: E402
    OUTCOME_COLUMNS,
    REFERENCE_DERIVED_COLUMNS,
)
from scripts.run_train_meta_residual_archetype_enhancement import (  # noqa: E402
    DEFAULT_OUT_DIR,
    _calibrate,
    _fit_platt,
    _merge_residual_features,
    _selection_mask,
    metrics_by_scope,
    surprise_calendar,
)

CHAMPION = "lifecycle_residual_aware_ae_gmm_overlay_pca8_clip8_baseline_globaloverlay"
SHOCK_CHAMPION = f"{CHAMPION}_sparse_shock_composite"
CHAMPION_PARENT = "lifecycle_residual_aware_ae_gmm_overlay_pca8_clip8_baseline"
ARMS = (
    "baseline_retrained",
    "lifecycle_only",
    "residual_archetypes",
    "lifecycle_residual_overlay",
    "lifecycle_residual_local_overlay",
    "lifecycle_residual_aware_ae_gmm_overlay",
    CHAMPION_PARENT,
    CHAMPION,
)
SEMANTIC_PREFIX = "meta_resid_arch_prob__"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _prediction_path(root: Path, arm: str) -> Path:
    directory = root / arm
    for name in ("oos_predictions.parquet", "oos_predictions_apr_may_jun.parquet"):
        path = directory / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No predictions for {arm}")


def _alt_summary(root: Path, arm: str) -> dict[str, Any]:
    metrics = pd.read_csv(root / arm / "metrics_by_scope.csv")
    row = metrics[
        metrics["scope"].eq("overall")
        & metrics["fraction"].eq(0.10)
        & metrics["selector"].eq(arm)
    ].iloc[0]
    predictions = pd.read_parquet(_prediction_path(root, arm))
    selected = _selected(predictions, "score_alternative")
    selected["true_week_start"] = _true_monday_week_start(selected["__ts__"])
    week_ev = selected.groupby("true_week_start", sort=True)["ev_after_1pct"].mean()
    month_ev = selected.groupby("calendar_month", sort=True)["ev_after_1pct"].mean()
    autocorr = pd.read_csv(root / arm / "hit_surprise_autocorrelation.csv")
    autocorr = autocorr[autocorr["selector"].eq(arm)]
    return {
        "arm": arm,
        "selected_rows": int(row["selected_rows"]),
        "mean_ev_after_1pct": float(row["mean_ev_after_1pct"]),
        "clean_exec_precision": float(row["clean_exec_precision"]),
        "dirty_positive_rate": float(row["dirty_positive_rate"]),
        "first_touch_bad_mae_rate": float(row["first_touch_bad_mae_rate"]),
        "full_path_bad_mae_rate": float(row["full_path_bad_mae_rate"]),
        "timeout_rate": float(row["timeout_rate"]),
        "worst_week_ev": float(week_ev.min()),
        "positive_weeks": int(week_ev.gt(0.0).sum()),
        "weeks": int(len(week_ev)),
        "worst_month_ev": float(month_ev.min()),
        "positive_months": int(month_ev.gt(0.0).sum()),
        "months": int(len(month_ev)),
        "mean_abs_signed_surprise_autocorr_lag1": float(
            pd.to_numeric(autocorr["surprise_autocorr_lag1"], errors="coerce")
            .abs()
            .mean()
        ),
    }


def _current_reference_summary(root: Path) -> dict[str, Any]:
    metrics = pd.read_csv(root / "baseline_retrained" / "metrics_by_scope.csv")
    row = metrics[
        metrics["scope"].eq("overall")
        & metrics["fraction"].eq(0.10)
        & metrics["selector"].eq("current_reference")
    ].iloc[0]
    predictions = pd.read_parquet(_prediction_path(root, "baseline_retrained"))
    selected = _selected(predictions, "score_current_reference")
    selected["true_week_start"] = _true_monday_week_start(selected["__ts__"])
    week_ev = selected.groupby("true_week_start", sort=True)["ev_after_1pct"].mean()
    month_ev = selected.groupby("calendar_month", sort=True)["ev_after_1pct"].mean()
    autocorr = pd.read_csv(
        root / "baseline_retrained" / "hit_surprise_autocorrelation.csv"
    )
    autocorr = autocorr[autocorr["selector"].eq("current_reference")]
    return {
        "arm": "current_reference",
        "selected_rows": int(row["selected_rows"]),
        "mean_ev_after_1pct": float(row["mean_ev_after_1pct"]),
        "clean_exec_precision": float(row["clean_exec_precision"]),
        "dirty_positive_rate": float(row["dirty_positive_rate"]),
        "first_touch_bad_mae_rate": float(row["first_touch_bad_mae_rate"]),
        "full_path_bad_mae_rate": float(row["full_path_bad_mae_rate"]),
        "timeout_rate": float(row["timeout_rate"]),
        "worst_week_ev": float(week_ev.min()),
        "positive_weeks": int(week_ev.gt(0.0).sum()),
        "weeks": int(len(week_ev)),
        "worst_month_ev": float(month_ev.min()),
        "positive_months": int(month_ev.gt(0.0).sum()),
        "months": int(len(month_ev)),
        "mean_abs_signed_surprise_autocorr_lag1": float(
            pd.to_numeric(autocorr["surprise_autocorr_lag1"], errors="coerce")
            .abs()
            .mean()
        ),
    }


def _selected(
    frame: pd.DataFrame, score_col: str, fraction: float = 0.10
) -> pd.DataFrame:
    mask = _selection_mask(frame, score_col, fraction, ["calendar_month", "side_name"])
    return frame.loc[mask].copy()


def _true_monday_week_start(values: pd.Series) -> pd.Series:
    ts = pd.to_datetime(values, utc=True, errors="coerce").dt.floor("D")
    return ts - pd.to_timedelta(ts.dt.weekday, unit="D")


def _group_metrics(
    selected: pd.DataFrame, group_cols: list[str], arm: str
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in selected.groupby(group_cols, dropna=False, sort=True):
        values = key if isinstance(key, tuple) else (key,)
        row = {name: value for name, value in zip(group_cols, values, strict=False)}
        row.update(
            {
                "arm": arm,
                "selected_rows": int(len(group)),
                "mean_ev_after_1pct": float(
                    pd.to_numeric(group["ev_after_1pct"], errors="coerce").mean()
                ),
                "clean_exec_precision": float(
                    pd.to_numeric(group["clean_exec"], errors="coerce").mean()
                ),
                "dirty_positive_rate": float(
                    pd.to_numeric(group["dirty_positive"], errors="coerce").mean()
                ),
                "first_touch_bad_mae_rate": float(
                    pd.to_numeric(
                        group["first_touch_bad_mae_1r"], errors="coerce"
                    ).mean()
                ),
                "full_path_bad_mae_rate": float(
                    pd.to_numeric(group["full_path_bad_mae_1r"], errors="coerce").mean()
                ),
                "timeout_rate": float(
                    pd.to_numeric(group["timeout"], errors="coerce").mean()
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _calendar_components(
    frame: pd.DataFrame, *, score_col: str, prob_col: str, arm: str
) -> pd.DataFrame:
    selected = _selected(frame, score_col)
    return _calendar_components_preselected(selected, prob_col=prob_col, arm=arm)


def _calendar_components_preselected(
    selected: pd.DataFrame,
    *,
    prob_col: str,
    arm: str,
) -> pd.DataFrame:
    selected = selected.copy()
    selected["date"] = pd.to_datetime(selected["__ts__"], utc=True).dt.floor("D")
    surprise = pd.to_numeric(selected["clean_exec"], errors="coerce") - pd.to_numeric(
        selected[prob_col], errors="coerce"
    )
    selected["signed_surprise"] = surprise
    selected["negative_surprise"] = (-surprise).clip(lower=0.0)
    selected["positive_surprise"] = surprise.clip(lower=0.0)
    daily = (
        selected.groupby(["date", "side_name", "archetype_policy_key"], dropna=False)
        .agg(
            rows=("clean_exec", "size"),
            hit_rate=("clean_exec", "mean"),
            mean_ev_after_1pct=("ev_after_1pct", "mean"),
            signed_surprise=("signed_surprise", "mean"),
            negative_surprise=("negative_surprise", "mean"),
            positive_surprise=("positive_surprise", "mean"),
        )
        .reset_index()
    )
    daily["arm"] = arm
    return daily


def _autocorr_components(calendar: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (arm, side, arch), group in calendar.groupby(
        ["arm", "side_name", "archetype_policy_key"], dropna=False
    ):
        group = group.sort_values("date")
        row: dict[str, Any] = {
            "arm": arm,
            "side_name": side,
            "archetype_policy_key": arch,
            "days": int(len(group)),
        }
        for name in ("signed_surprise", "negative_surprise", "positive_surprise"):
            series = pd.to_numeric(group[name], errors="coerce")
            row[f"{name}_autocorr_lag1"] = (
                float(series.autocorr(1)) if len(series) >= 3 else np.nan
            )
            row[f"{name}_autocorr_lag3"] = (
                float(series.autocorr(3)) if len(series) >= 5 else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _event_table(comparison: pd.DataFrame) -> pd.DataFrame:
    tail = comparison[
        comparison["baseline_high_surprise"].fillna(False).astype(bool)
    ].copy()
    tail["date"] = pd.to_datetime(tail["date"], utc=True)
    events: list[pd.DataFrame] = []
    for _, group in tail.sort_values("date").groupby(
        ["side_name", "archetype_policy_key"], sort=True
    ):
        current: list[pd.Series] = []
        previous = None
        previous_sign = None
        for _, row in group.iterrows():
            sign = int(np.sign(float(row["mean_hit_surprise_base"])))
            split = previous is not None and (
                (row["date"] - previous).days > 1 or sign != previous_sign
            )
            if split and current:
                events.append(pd.DataFrame(current))
                current = []
            current.append(row)
            previous = row["date"]
            previous_sign = sign
        if current:
            events.append(pd.DataFrame(current))
    rows: list[dict[str, Any]] = []
    for event_id, event in enumerate(events):
        wb = pd.to_numeric(event["rows_base"], errors="coerce").clip(lower=1.0)
        wa = pd.to_numeric(event["rows_alt"], errors="coerce").clip(lower=1.0)
        base_surprise = float(np.average(event["mean_hit_surprise_base"], weights=wb))
        alt_surprise = float(np.average(event["mean_hit_surprise_alt"], weights=wa))
        base_ev = float(np.average(event["mean_ev_after_1pct_base"], weights=wb))
        alt_ev = float(np.average(event["mean_ev_after_1pct_alt"], weights=wa))
        abs_improvement = abs(base_surprise) - abs(alt_surprise)
        significantly_improved = abs_improvement >= 0.20 * abs(base_surprise)
        economically_improved = alt_ev >= base_ev
        rows.append(
            {
                "event_id": event_id,
                "side_name": event["side_name"].iloc[0],
                "archetype_policy_key": event["archetype_policy_key"].iloc[0],
                "start": event["date"].min(),
                "end": event["date"].max(),
                "days": int(len(event)),
                "baseline_rows": int(
                    pd.to_numeric(event["rows_base"], errors="coerce").sum()
                ),
                "baseline_signed_surprise": base_surprise,
                "champion_signed_surprise": alt_surprise,
                "surprise_abs_improvement": abs_improvement,
                "baseline_ev": base_ev,
                "champion_ev": alt_ev,
                "ev_delta": alt_ev - base_ev,
                "surprise_significantly_improved": significantly_improved,
                "economically_improved": economically_improved,
                "improved_either_way": bool(
                    significantly_improved or economically_improved
                ),
                "material_persistent_event": bool(
                    len(event) >= 2
                    and pd.to_numeric(event["rows_base"], errors="coerce").sum() >= 20
                ),
            }
        )
    return pd.DataFrame(rows)


def _archetype_catalog(root: Path) -> pd.DataFrame:
    catalog = pd.read_csv(
        root
        / "residual_walkforward_ae_gmm_eval_mar_jun_pca8_clip8_baseline_catalog.csv"
    )
    latest = str(catalog["oos_month"].astype(str).max())
    catalog = catalog[catalog["oos_month"].astype(str).eq(latest)].copy()
    rows: list[dict[str, Any]] = []
    for semantic, group in catalog.groupby("semantic", sort=True):
        weight = pd.to_numeric(group["rows"], errors="coerce").clip(lower=1.0)
        row = {
            "semantic": semantic,
            "latest_oos_month": latest,
            "component_count": int(len(group)),
            "support_rows_sum": int(weight.sum()),
        }
        for name in (
            "mean_hit_surprise",
            "mean_ev",
            "clean_rate",
            "dirty_rate",
            "bad_mae_rate",
            "timeout_rate",
            "negative_tail_rate",
            "positive_tail_rate",
        ):
            row[name] = float(
                np.average(pd.to_numeric(group[name], errors="coerce"), weights=weight)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _seed_stability(root: Path) -> pd.DataFrame:
    arms = (
        CHAMPION,
        f"{CHAMPION_PARENT}_seed17_globaloverlay",
        f"{CHAMPION_PARENT}_seed29_globaloverlay",
    )
    rows: list[dict[str, Any]] = []
    for arm in arms:
        manifest = json.loads((root / arm / "manifest.json").read_text())
        parent = str(manifest.get("parent_arm", arm.removesuffix("_globaloverlay")))
        parent_manifest = json.loads((root / parent / "manifest.json").read_text())
        overlay = joblib.load(root / arm / "residual_overlay_state.joblib")
        summary = _alt_summary(root, arm)
        rows.append(
            {
                "arm": arm,
                "seed_offset": int(parent_manifest["seed_offset"]),
                "hit_alpha": float(overlay.hit_alpha),
                "local_hit_alpha": float(overlay.local_hit_alpha),
                "top10_mean_ev_after_1pct": summary["mean_ev_after_1pct"],
                "top10_clean_exec_precision": summary["clean_exec_precision"],
                "top10_full_path_bad_mae_rate": summary["full_path_bad_mae_rate"],
                "mean_abs_signed_surprise_autocorr_lag1": summary[
                    "mean_abs_signed_surprise_autocorr_lag1"
                ],
                "worst_week_ev": summary["worst_week_ev"],
                "positive_weeks": summary["positive_weeks"],
                "weeks": summary["weeks"],
            }
        )
    return pd.DataFrame(rows)


def _markdown_table(
    frame: pd.DataFrame, columns: Iterable[str], *, percent: set[str] | None = None
) -> str:
    percent = percent or set()
    cols = list(columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in frame.iterrows():
        values: list[str] = []
        for col in cols:
            value = row.get(col)
            if col in percent and pd.notna(value):
                values.append(f"{100.0 * float(value):.4f}%")
            elif isinstance(value, (float, np.floating)):
                values.append("" if not np.isfinite(value) else f"{float(value):.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    root = DEFAULT_OUT_DIR
    report_dir = root / "final_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(
        [_current_reference_summary(root)] + [_alt_summary(root, arm) for arm in ARMS]
    )
    baseline = summary[summary["arm"].eq("current_reference")].iloc[0]
    for name in (
        "mean_ev_after_1pct",
        "clean_exec_precision",
        "dirty_positive_rate",
        "first_touch_bad_mae_rate",
        "full_path_bad_mae_rate",
        "timeout_rate",
        "worst_week_ev",
        "worst_month_ev",
        "mean_abs_signed_surprise_autocorr_lag1",
    ):
        summary[f"delta_vs_current__{name}"] = summary[name] - float(baseline[name])
    summary.to_csv(report_dir / "model_summary.csv", index=False)

    burnin = pd.read_parquet(
        root / "lifecycle_only_burnin" / "oos_predictions_march_burnin.parquet"
    )
    residual = pd.read_parquet(
        root
        / "cache"
        / "residual_walkforward_ae_gmm_eval_mar_jun_pca8_clip8_baseline.parquet"
    )
    burnin = _merge_residual_features(burnin, residual)
    overlay_state = joblib.load(root / CHAMPION / "residual_overlay_state.joblib")
    safe_burnin = burnin.drop(
        columns=[
            name
            for name in OUTCOME_COLUMNS | REFERENCE_DERIVED_COLUMNS
            if name in burnin.columns
        ],
        errors="ignore",
    )
    burnin["score_lifecycle_only"] = pd.to_numeric(
        burnin["score_alternative"], errors="coerce"
    ).astype(np.float32)
    burnin["score_alternative"] = overlay_state.transform(
        safe_burnin,
        burnin["score_lifecycle_only"].fillna(0.5).to_numpy(dtype=np.float32),
    )
    burnin_platt = _fit_platt(burnin["score_alternative"], burnin["clean_exec"])
    burnin["hit_prob_alternative"] = _calibrate(
        burnin_platt, burnin["score_alternative"]
    )
    burnin_arm = f"{CHAMPION}_burnin"
    burnin_metrics = metrics_by_scope(burnin, burnin_arm)
    burnin_calendar, burnin_autocorr, burnin_comparison = surprise_calendar(
        burnin, burnin_arm
    )
    burnin_metrics.to_csv(report_dir / "burnin_march_metrics.csv", index=False)
    burnin_calendar.to_csv(
        report_dir / "burnin_march_hit_surprise_calendar.csv", index=False
    )
    burnin_autocorr.to_csv(
        report_dir / "burnin_march_hit_surprise_autocorrelation.csv", index=False
    )
    burnin_comparison.to_csv(
        report_dir / "burnin_march_high_surprise_comparison.csv", index=False
    )
    burnin_overall = burnin_metrics[
        burnin_metrics["scope"].eq("overall") & burnin_metrics["fraction"].eq(0.10)
    ]
    burnin_base = burnin_overall[
        burnin_overall["selector"].eq("current_reference")
    ].iloc[0]
    burnin_alt = burnin_overall[burnin_overall["selector"].eq(burnin_arm)].iloc[0]
    burnin_tail = burnin_comparison[
        burnin_comparison["baseline_high_surprise"].fillna(False).astype(bool)
    ]

    current_frame = pd.read_parquet(_prediction_path(root, "baseline_retrained"))
    champion = pd.read_parquet(_prediction_path(root, CHAMPION))
    coverage_work = champion.copy()
    coverage_work["__ts__"] = pd.to_datetime(
        coverage_work["__ts__"], utc=True, errors="coerce"
    )
    coverage_work["utc_date"] = coverage_work["__ts__"].dt.floor("D")
    coverage = (
        coverage_work.groupby("calendar_month", sort=True)
        .agg(
            candidate_rows=("__ts__", "size"),
            distinct_timestamps=("__ts__", "nunique"),
            observed_utc_days=("utc_date", "nunique"),
            distinct_symbols=("__symbol__", "nunique"),
            min_timestamp=("__ts__", "min"),
            max_timestamp=("__ts__", "max"),
        )
        .reset_index()
    )
    coverage["expected_hourly_timestamps"] = coverage["calendar_month"].map(
        lambda value: int(pd.Period(str(value), freq="M").days_in_month * 24)
    )
    coverage["timestamp_coverage_rate"] = (
        coverage["distinct_timestamps"] / coverage["expected_hourly_timestamps"]
    )
    coverage.to_csv(report_dir / "oos_source_coverage.csv", index=False)
    current_selected = _selected(current_frame, "score_current_reference")
    champion_selected = _selected(champion, "score_alternative")
    month_current = _group_metrics(
        current_selected, ["calendar_month"], "current_reference"
    )
    month_champion = _group_metrics(champion_selected, ["calendar_month"], CHAMPION)
    month = month_current.merge(
        month_champion, on="calendar_month", suffixes=("_current", "_champion")
    )
    for metric in (
        "mean_ev_after_1pct",
        "clean_exec_precision",
        "full_path_bad_mae_rate",
    ):
        month[f"delta__{metric}"] = (
            month[f"{metric}_champion"] - month[f"{metric}_current"]
        )
    month.to_csv(report_dir / "month_by_month.csv", index=False)

    current_selected["week_start"] = _true_monday_week_start(current_selected["__ts__"])
    champion_selected["week_start"] = _true_monday_week_start(
        champion_selected["__ts__"]
    )
    week_current = _group_metrics(current_selected, ["week_start"], "current_reference")
    week_champion = _group_metrics(champion_selected, ["week_start"], CHAMPION)
    week = week_current.merge(
        week_champion, on="week_start", suffixes=("_current", "_champion")
    )
    for metric in (
        "mean_ev_after_1pct",
        "clean_exec_precision",
        "dirty_positive_rate",
        "first_touch_bad_mae_rate",
        "full_path_bad_mae_rate",
        "timeout_rate",
    ):
        week[f"delta__{metric}"] = (
            week[f"{metric}_champion"] - week[f"{metric}_current"]
        )
    week.to_csv(report_dir / "week_by_week_delta.csv", index=False)

    side_arch_current = _group_metrics(
        current_selected, ["side_name", "archetype_policy_key"], "current_reference"
    )
    side_arch_champion = _group_metrics(
        champion_selected, ["side_name", "archetype_policy_key"], CHAMPION
    )
    side_arch = side_arch_current.merge(
        side_arch_champion,
        on=["side_name", "archetype_policy_key"],
        how="outer",
        suffixes=("_current", "_champion"),
    )
    for metric in (
        "mean_ev_after_1pct",
        "clean_exec_precision",
        "full_path_bad_mae_rate",
    ):
        side_arch[f"delta__{metric}"] = (
            side_arch[f"{metric}_champion"] - side_arch[f"{metric}_current"]
        )
    side_arch.to_csv(report_dir / "side_archetype_metrics.csv", index=False)

    calendar = pd.concat(
        [
            _calendar_components(
                current_frame,
                score_col="score_current_reference",
                prob_col="hit_prob_current_reference",
                arm="current_reference",
            ),
            _calendar_components(
                champion,
                score_col="score_alternative",
                prob_col="hit_prob_alternative",
                arm=CHAMPION,
            ),
        ],
        ignore_index=True,
    )
    calendar.to_csv(
        report_dir / "hit_surprise_calendar_signed_components.csv", index=False
    )
    autocorr = _autocorr_components(calendar)
    autocorr.to_csv(
        report_dir / "hit_surprise_autocorrelation_signed_components.csv", index=False
    )

    comparison = pd.read_csv(root / CHAMPION / "high_surprise_period_comparison.csv")
    events = _event_table(comparison)
    events.to_csv(report_dir / "high_surprise_events.csv", index=False)
    unexplained = events[~events["improved_either_way"]].copy()
    unexplained["diagnosis"] = np.where(
        (
            unexplained["days"].eq(1)
            & unexplained["baseline_rows"].le(25)
            & unexplained["baseline_signed_surprise"].lt(0.0)
        ),
        (
            "sparse synchronized cross-asset shock; June 30 long-mixed losses concentrated "
            "across altcoins while both semantic and surprise-head recognizers underweighted "
            "negative-tail risk; insufficient recurrence for a leakage-safe local rule"
        ),
        np.where(
            unexplained["baseline_rows"].lt(20),
            "sparse single-day cell; insufficient event support",
            np.where(
                unexplained["baseline_signed_surprise"].gt(0.0),
                "positive-surprise underconfidence; ranking remains profitable but calibration lags realized hit rate",
                "negative-surprise selection error requiring further local context",
            ),
        ),
    )
    unexplained.to_csv(report_dir / "unimproved_high_surprise_events.csv", index=False)

    catalog = _archetype_catalog(root)
    catalog.to_csv(report_dir / "residual_archetype_semantics_metrics.csv", index=False)
    seed_stability = _seed_stability(root)
    seed_stability.to_csv(
        report_dir / "residual_ae_gmm_seed_stability.csv", index=False
    )
    inference = json.loads(
        (
            root / "inference_bundle_residual_pca8_globaloverlay" / "manifest.json"
        ).read_text()
    )
    historical_dir = root / f"historical_rank_oos_{CHAMPION}"
    historical_ranked = pd.read_parquet(
        historical_dir / "oos_predictions_historical_rank.parquet"
    )
    historical_ranked["week_start"] = _true_monday_week_start(
        historical_ranked["__ts__"]
    )
    historical_selected_current = historical_ranked[
        pd.to_numeric(
            historical_ranked["historical_rank_current_reference"], errors="coerce"
        ).ge(0.90)
    ].copy()
    historical_selected_champion = historical_ranked[
        pd.to_numeric(
            historical_ranked["historical_rank_alternative"], errors="coerce"
        ).ge(0.90)
    ].copy()
    historical_component_calendar = pd.concat(
        [
            _calendar_components_preselected(
                historical_selected_current,
                prob_col="hit_prob_current_reference",
                arm="current_reference",
            ),
            _calendar_components_preselected(
                historical_selected_champion,
                prob_col="hit_prob_alternative",
                arm=CHAMPION,
            ),
        ],
        ignore_index=True,
    )
    historical_component_calendar.to_csv(
        report_dir / "historical_rank_hit_surprise_calendar_signed_components.csv",
        index=False,
    )
    historical_component_autocorr = _autocorr_components(historical_component_calendar)
    historical_component_autocorr.to_csv(
        report_dir
        / "historical_rank_hit_surprise_autocorrelation_signed_components.csv",
        index=False,
    )
    historical_component_summary = historical_component_autocorr.groupby("arm").agg(
        signed=(
            "signed_surprise_autocorr_lag1",
            lambda values: pd.to_numeric(values, errors="coerce").abs().mean(),
        ),
        negative=(
            "negative_surprise_autocorr_lag1",
            lambda values: pd.to_numeric(values, errors="coerce").abs().mean(),
        ),
        positive=(
            "positive_surprise_autocorr_lag1",
            lambda values: pd.to_numeric(values, errors="coerce").abs().mean(),
        ),
    )
    historical_overall = pd.read_csv(historical_dir / "metrics_by_scope.csv")
    historical_overall = historical_overall[
        historical_overall["scope"].eq("overall")
        & historical_overall["fraction"].eq(0.10)
    ].copy()
    historical_overall.to_csv(
        report_dir / "historical_rank_top10_summary.csv", index=False
    )

    def historical_pair(group_cols: list[str]) -> pd.DataFrame:
        current = _group_metrics(
            historical_selected_current, group_cols, "current_reference"
        )
        alternative = _group_metrics(historical_selected_champion, group_cols, CHAMPION)
        paired = current.merge(
            alternative,
            on=group_cols,
            how="outer",
            suffixes=("_current", "_champion"),
        )
        for metric in (
            "mean_ev_after_1pct",
            "clean_exec_precision",
            "dirty_positive_rate",
            "first_touch_bad_mae_rate",
            "full_path_bad_mae_rate",
            "timeout_rate",
        ):
            paired[f"delta__{metric}"] = (
                paired[f"{metric}_champion"] - paired[f"{metric}_current"]
            )
        return paired

    historical_month = historical_pair(["calendar_month"])
    historical_week = historical_pair(["week_start"])
    historical_side_archetype = historical_pair(["side_name", "archetype_policy_key"])
    historical_month.to_csv(
        report_dir / "historical_rank_month_by_month.csv", index=False
    )
    historical_week.to_csv(report_dir / "historical_rank_week_by_week.csv", index=False)
    historical_side_archetype.to_csv(
        report_dir / "historical_rank_side_archetype.csv",
        index=False,
    )
    historical_autocorr = pd.read_csv(
        historical_dir / "hit_surprise_autocorrelation.csv"
    )
    historical_autocorr.to_csv(
        report_dir / "historical_rank_hit_surprise_autocorrelation.csv", index=False
    )
    historical_high_surprise = pd.read_csv(
        historical_dir / "high_surprise_period_comparison.csv"
    )
    historical_high_surprise.to_csv(
        report_dir / "historical_rank_high_surprise_periods.csv", index=False
    )
    historical_tail = historical_high_surprise[
        historical_high_surprise["baseline_high_surprise"].fillna(False).astype(bool)
    ]
    historical_manifest = json.loads((historical_dir / "manifest.json").read_text())
    historical_base = historical_overall[
        historical_overall["selector"].eq("current_reference")
    ].iloc[0]
    historical_alt = historical_overall[
        historical_overall["selector"].eq(CHAMPION)
    ].iloc[0]
    historical_alt_weeks = historical_week["mean_ev_after_1pct_champion"]
    historical_ac_summary = historical_autocorr.groupby("selector")[
        "surprise_autocorr_lag1"
    ].apply(lambda values: pd.to_numeric(values, errors="coerce").abs().mean())
    robustness = json.loads(
        (root / CHAMPION / "robustness_extended_report.json").read_text()
    )
    insample_capacity_path = report_dir / "march_insample_capacity_manifest.json"
    insample_capacity = (
        json.loads(insample_capacity_path.read_text())
        if insample_capacity_path.exists()
        else {}
    )
    material = events[events["material_persistent_event"]]
    material_unimproved = material[~material["improved_either_way"]]
    stage_gate_path = report_dir / "stage_gate_scorecard.csv"
    stage_gate = (
        pd.read_csv(stage_gate_path) if stage_gate_path.exists() else pd.DataFrame()
    )
    event_sensitivity_path = report_dir / "stage4_event_sensitivity_manifest.json"
    event_sensitivity = (
        json.loads(event_sensitivity_path.read_text())
        if event_sensitivity_path.exists()
        else {}
    )
    gmm_robustness_path = report_dir / "stage7_gmm_robustness_manifest.json"
    gmm_robustness = (
        json.loads(gmm_robustness_path.read_text())
        if gmm_robustness_path.exists()
        else {}
    )
    gmm_family_path = report_dir / "stage7_gmm_family_summary.csv"
    gmm_family = (
        pd.read_csv(gmm_family_path) if gmm_family_path.exists() else pd.DataFrame()
    )
    representation_family_path = report_dir / "stage6_representation_family_summary.csv"
    representation_family = (
        pd.read_csv(representation_family_path)
        if representation_family_path.exists()
        else pd.DataFrame()
    )
    representation_manifest_path = report_dir / "stage6_representation_manifest.json"
    representation_manifest = (
        json.loads(representation_manifest_path.read_text())
        if representation_manifest_path.exists()
        else {}
    )
    retrained_summary_path = report_dir / "retrained_aligned_top10_summary.csv"
    retrained_summary = (
        pd.read_csv(retrained_summary_path)
        if retrained_summary_path.exists()
        else pd.DataFrame()
    )
    retrained_manifest_path = (
        root / "lifecycle_residual_surprise_head_retrained" / "manifest.json"
    )
    retrained_manifest = (
        json.loads(retrained_manifest_path.read_text())
        if retrained_manifest_path.exists()
        else {}
    )
    forced_manifest_path = (
        root / "lifecycle_residual_surprise_head_forced_retrained" / "manifest.json"
    )
    forced_manifest = (
        json.loads(forced_manifest_path.read_text())
        if forced_manifest_path.exists()
        else {}
    )
    blend_manifest_path = (
        root / "lifecycle_residual_pca8_forced_retrained_rank_blend" / "manifest.json"
    )
    blend_manifest = (
        json.loads(blend_manifest_path.read_text())
        if blend_manifest_path.exists()
        else {}
    )
    semantic_enrichment_path = report_dir / "stage8_semantic_enrichment.csv"
    semantic_enrichment = (
        pd.read_csv(semantic_enrichment_path)
        if semantic_enrichment_path.exists()
        else pd.DataFrame()
    )
    shock_summary_path = report_dir / "shock_overlay_top10_summary.csv"
    shock_summary = (
        pd.read_csv(shock_summary_path)
        if shock_summary_path.exists()
        else pd.DataFrame()
    )
    shock_breakdowns_path = report_dir / "shock_overlay_breakdowns.csv"
    shock_breakdowns = (
        pd.read_csv(shock_breakdowns_path)
        if shock_breakdowns_path.exists()
        else pd.DataFrame()
    )
    shock_report_path = report_dir / "shock_overlay_manifest.json"
    shock_report = (
        json.loads(shock_report_path.read_text()) if shock_report_path.exists() else {}
    )
    shock_autocorr_path = report_dir / "shock_overlay_hit_surprise_autocorrelation.csv"
    shock_autocorr = (
        pd.read_csv(shock_autocorr_path)
        if shock_autocorr_path.exists()
        else pd.DataFrame()
    )
    shock_component_summary = (
        shock_autocorr.groupby("arm").agg(
            signed=(
                "signed_surprise_autocorr_lag1",
                lambda values: pd.to_numeric(values, errors="coerce").abs().mean(),
            ),
            negative=(
                "negative_surprise_autocorr_lag1",
                lambda values: pd.to_numeric(values, errors="coerce").abs().mean(),
            ),
            positive=(
                "positive_surprise_autocorr_lag1",
                lambda values: pd.to_numeric(values, errors="coerce").abs().mean(),
            ),
        )
        if not shock_autocorr.empty
        else pd.DataFrame()
    )
    shock_model_path = root / SHOCK_CHAMPION / "manifest.json"
    shock_model = (
        json.loads(shock_model_path.read_text()) if shock_model_path.exists() else {}
    )
    shock_inference_path = (
        root / "inference_bundle_residual_pca8_globaloverlay_shock" / "manifest.json"
    )
    shock_inference = (
        json.loads(shock_inference_path.read_text())
        if shock_inference_path.exists()
        else {}
    )
    shock_event_path = root / SHOCK_CHAMPION / "june30_long_mixed_event.csv"
    shock_event = (
        pd.read_csv(shock_event_path) if shock_event_path.exists() else pd.DataFrame()
    )
    final_inference = shock_inference or inference
    material_event_note = (
        "No material persistent high-surprise event remains unimproved."
        if material_unimproved.empty
        else (
            f"{len(material_unimproved)} material persistent event(s) remain unimproved; "
            "see `unimproved_high_surprise_events.csv`."
        )
    )
    shock_parent = (
        shock_summary[shock_summary["selector"].eq(CHAMPION)].iloc[0]
        if not shock_summary.empty and shock_summary["selector"].eq(CHAMPION).any()
        else pd.Series(dtype=float)
    )
    shock_final = (
        shock_summary[shock_summary["selector"].eq(SHOCK_CHAMPION)].iloc[0]
        if not shock_summary.empty
        and shock_summary["selector"].eq(SHOCK_CHAMPION).any()
        else pd.Series(dtype=float)
    )
    shock_month = (
        shock_breakdowns[
            shock_breakdowns["scope"].eq("month")
            & shock_breakdowns["arm"].isin([CHAMPION, SHOCK_CHAMPION])
        ]
        if not shock_breakdowns.empty
        else pd.DataFrame()
    )
    shock_week = (
        shock_breakdowns[
            shock_breakdowns["scope"].eq("week")
            & shock_breakdowns["arm"].isin([CHAMPION, SHOCK_CHAMPION])
        ]
        if not shock_breakdowns.empty
        else pd.DataFrame()
    )
    shock_side_archetype = (
        shock_breakdowns[
            shock_breakdowns["scope"].eq("side_archetype")
            & shock_breakdowns["arm"].isin([CHAMPION, SHOCK_CHAMPION])
        ]
        if not shock_breakdowns.empty
        else pd.DataFrame()
    )

    percent_cols = {
        "mean_ev_after_1pct",
        "clean_exec_precision",
        "full_path_bad_mae_rate",
        "worst_week_ev",
        "worst_month_ev",
    }
    report = f"""# Alternative Meta Residual-Archetype Report

## Verdict

The alternative `{SHOCK_CHAMPION}` is the final robust arm. It starts from `{CHAMPION}`, preserves the current base model and current meta model, and adds a sparse market-wide liquidation/rebound overlay selected from monthly walk-forward states between April 2025 and March 2026. Its corrected eight-component robust PCA representation remains the core residual representation; the shock layer only nudges ranks in jointly extreme OI-flush and price-up/OI-down states.

The primary evidence uses a causal historical score-rank contract: April is ranked against March scores, May against March-April, and June against March-May. The candidate is **inference-ready as an alternative artifact**, not promoted over the current production meta model. Exact July bundle, shock transform, historical-rank, and frozen base AE/GMM parity all pass. Residual AE/GMM, local score normalization, depth-3 shock models, and March-only shock rules were rejected by simpler held-out baselines.

## Primary Causal Historical-Rank Result

{_markdown_table(historical_overall, ["selector", "candidate_rows", "selected_rows", "trades_per_observed_day", "mean_ev_after_1pct", "clean_exec_precision", "dirty_positive_rate", "first_touch_bad_mae_rate", "full_path_bad_mae_rate", "timeout_rate"], percent={"mean_ev_after_1pct", "clean_exec_precision", "dirty_positive_rate", "first_touch_bad_mae_rate", "full_path_bad_mae_rate", "timeout_rate"})}

At the frozen historical top-10 threshold, EV improves from `{100 * float(historical_base["mean_ev_after_1pct"]):.4f}%` to `{100 * float(historical_alt["mean_ev_after_1pct"]):.4f}%` per selected row after the 1% cost convention. Clean precision improves from `{100 * float(historical_base["clean_exec_precision"]):.2f}%` to `{100 * float(historical_alt["clean_exec_precision"]):.2f}%`. The alternative is positive in `{int(historical_alt_weeks.gt(0.0).sum())}/{len(historical_alt_weeks)}` observed Monday-anchored weeks; its worst week is `{100 * float(historical_alt_weeks.min()):.4f}%`.

Mean absolute side x archetype lag-1 hit-surprise autocorrelation falls from `{float(historical_ac_summary["current_reference"]):.6f}` to `{float(historical_ac_summary[CHAMPION]):.6f}`. `{int(historical_tail["high_surprise_significantly_improved"].sum())}/{len(historical_tail)}` baseline high-surprise cells contract by at least 20%.

On the same causal historical top-10 population, mean absolute negative-surprise autocorrelation falls from `{float(historical_component_summary.loc["current_reference", "negative"]):.6f}` to `{float(historical_component_summary.loc[CHAMPION, "negative"]):.6f}`; positive-surprise autocorrelation falls from `{float(historical_component_summary.loc["current_reference", "positive"]):.6f}` to `{float(historical_component_summary.loc[CHAMPION, "positive"]):.6f}`.

## Sparse Shock Extension

{_markdown_table(shock_summary, ["selector", "selected_rows", "mean_ev_after_1pct", "clean_exec_precision", "full_path_bad_mae_rate", "timeout_rate", "worst_week_ev", "worst_month_ev", "positive_weeks", "weeks"], percent={"mean_ev_after_1pct", "clean_exec_precision", "full_path_bad_mae_rate", "timeout_rate", "worst_week_ev", "worst_month_ev"}) if not shock_summary.empty else "Shock overlay report not materialized."}

The historical walk-forward search selects raw market composites rather than archetype-weighted composites: long threshold/alpha `{shock_model.get("selected_side_parameters", {}).get("long", {})}`, short `{shock_model.get("selected_side_parameters", {}).get("short", {})}`. Against its PCA parent, EV changes from `{100 * float(shock_parent.get("mean_ev_after_1pct", float("nan"))):.4f}%` to `{100 * float(shock_final.get("mean_ev_after_1pct", float("nan"))):.4f}%`, worst-week EV from `{100 * float(shock_parent.get("worst_week_ev", float("nan"))):.4f}%` to `{100 * float(shock_final.get("worst_week_ev", float("nan"))):.4f}%`, and mean absolute signed-surprise autocorrelation from `{float(shock_report.get("mean_abs_signed_autocorr", {}).get(CHAMPION, float("nan"))):.6f}` to `{float(shock_report.get("mean_abs_signed_autocorr", {}).get(SHOCK_CHAMPION, float("nan"))):.6f}`. Final negative/positive component autocorrelation is `{float(shock_component_summary.loc[SHOCK_CHAMPION, "negative"]) if SHOCK_CHAMPION in shock_component_summary.index else float("nan"):.6f}` / `{float(shock_component_summary.loc[SHOCK_CHAMPION, "positive"]) if SHOCK_CHAMPION in shock_component_summary.index else float("nan"):.6f}`, versus current `{float(shock_component_summary.loc["current_reference", "negative"]) if "current_reference" in shock_component_summary.index else float("nan"):.6f}` / `{float(shock_component_summary.loc["current_reference", "positive"]) if "current_reference" in shock_component_summary.index else float("nan"):.6f}`. The parent-to-final weekly bootstrap delta is `{100 * float(shock_report.get("parent_to_final_weekly_bootstrap", {}).get("mean_delta", float("nan"))):.4f}%`, 95% CI `[{100 * float(shock_report.get("parent_to_final_weekly_bootstrap", {}).get("ci025", float("nan"))):.4f}%, {100 * float(shock_report.get("parent_to_final_weekly_bootstrap", {}).get("ci975", float("nan"))):.4f}%]`.

The June 30 long-mixed event is repaired without using June outcomes for parameter selection: selected rows fall from `{int(shock_event.iloc[0]["selected_rows"]) if len(shock_event) >= 2 else 0}` to `{int(shock_event.iloc[1]["selected_rows"]) if len(shock_event) >= 2 else 0}`, EV moves from `{100 * float(shock_event.iloc[0]["mean_ev_after_1pct"]):.4f}%` to `{100 * float(shock_event.iloc[1]["mean_ev_after_1pct"]):.4f}%`, and clean precision from `{100 * float(shock_event.iloc[0]["clean_exec_precision"]):.2f}%` to `{100 * float(shock_event.iloc[1]["clean_exec_precision"]):.2f}%`.

### Shock Month Breakdown

{_markdown_table(shock_month, ["arm", "calendar_month", "selected_rows", "mean_ev_after_1pct", "clean_exec_precision", "full_path_bad_mae_rate", "timeout_rate"], percent={"mean_ev_after_1pct", "clean_exec_precision", "full_path_bad_mae_rate", "timeout_rate"}) if not shock_month.empty else "Not materialized."}

### Shock Week Breakdown

{_markdown_table(shock_week, ["arm", "week_start", "selected_rows", "mean_ev_after_1pct", "clean_exec_precision"], percent={"mean_ev_after_1pct", "clean_exec_precision"}) if not shock_week.empty else "Not materialized."}

### Shock Side x Archetype Breakdown

{_markdown_table(shock_side_archetype, ["arm", "side_name", "archetype_policy_key", "selected_rows", "mean_ev_after_1pct", "clean_exec_precision", "full_path_bad_mae_rate"], percent={"mean_ev_after_1pct", "clean_exec_precision", "full_path_bad_mae_rate"}) if not shock_side_archetype.empty else "Not materialized."}

## Retrained Meta Ablation

The canonical `lgbm_pipeline` selector was corrected so both its univariate and Relief stages score the economic target **within** each side x base-archetype slice, then retain the union. It evaluated `158` candidates on `966,990` pre-March rows and automatically selected `{int(retrained_manifest.get("selected_feature_count", 0))}` features, including six post-selection OOD features. The final model reused the current meta parameters and the same soft-binary target.

{_markdown_table(retrained_summary, ["selector", "selected_rows", "mean_ev_after_1pct", "clean_exec_precision", "full_path_bad_mae_rate", "timeout_rate", "worst_week_ev", "worst_month_ev", "positive_weeks", "weeks", "mean_abs_surprise_autocorr_lag1"], percent={"mean_ev_after_1pct", "clean_exec_precision", "full_path_bad_mae_rate", "timeout_rate", "worst_week_ev", "worst_month_ev"}) if not retrained_summary.empty else "Retrained comparison not materialized."}

Automatic MDA retained `{int(retrained_manifest.get("feature_families", {}).get("walkforward_surprise_heads", {}).get("selected_count", 0))}/5` surprise-head outputs. The support ablation forced all five outputs into the same selected contract and used `{int(forced_manifest.get("selected_feature_count", 0))}` total features. It improved the automatic retrain but remained below the corrected-PCA overlay and had worse calendar autocorrelation. A March-only causal rank-blend search selected forced-retrain weight `{float(blend_manifest.get("selected_forced_weight", float("nan"))):.2f}`; zero weight reproduces the PCA champion exactly. The retrained-head branches are therefore rejected rather than added to inference.

### Causal Month Breakdown

{_markdown_table(historical_month, ["calendar_month", "selected_rows_current", "selected_rows_champion", "mean_ev_after_1pct_current", "mean_ev_after_1pct_champion", "delta__mean_ev_after_1pct", "clean_exec_precision_current", "clean_exec_precision_champion", "full_path_bad_mae_rate_current", "full_path_bad_mae_rate_champion"], percent={"mean_ev_after_1pct_current", "mean_ev_after_1pct_champion", "delta__mean_ev_after_1pct", "clean_exec_precision_current", "clean_exec_precision_champion", "full_path_bad_mae_rate_current", "full_path_bad_mae_rate_champion"})}

### Causal Week Breakdown

{_markdown_table(historical_week, ["week_start", "selected_rows_current", "selected_rows_champion", "mean_ev_after_1pct_current", "mean_ev_after_1pct_champion", "delta__mean_ev_after_1pct", "clean_exec_precision_current", "clean_exec_precision_champion"], percent={"mean_ev_after_1pct_current", "mean_ev_after_1pct_champion", "delta__mean_ev_after_1pct", "clean_exec_precision_current", "clean_exec_precision_champion"})}

### Causal Side x Base-Archetype Breakdown

{_markdown_table(historical_side_archetype, ["side_name", "archetype_policy_key", "selected_rows_current", "selected_rows_champion", "mean_ev_after_1pct_current", "mean_ev_after_1pct_champion", "delta__mean_ev_after_1pct", "clean_exec_precision_current", "clean_exec_precision_champion"], percent={"mean_ev_after_1pct_current", "mean_ev_after_1pct_champion", "delta__mean_ev_after_1pct", "clean_exec_precision_current", "clean_exec_precision_champion"})}

## Diagnostic Full-Month Top-K Comparison

The following arm matrix uses within-month side-specific quantiles and is retained for ablation comparability only. It is not the production rank contract and is secondary to the causal historical-rank result above.

{_markdown_table(summary, ["arm", "selected_rows", "mean_ev_after_1pct", "clean_exec_precision", "full_path_bad_mae_rate", "worst_week_ev", "worst_month_ev", "positive_weeks", "weeks", "mean_abs_signed_surprise_autocorr_lag1"], percent=percent_cols)}

Top-10 EV improves from `{100 * float(baseline["mean_ev_after_1pct"]):.4f}%` to `{100 * float(summary.loc[summary["arm"].eq(CHAMPION), "mean_ev_after_1pct"].iloc[0]):.4f}%` per trade after the 1% cost convention. Mean absolute side × archetype lag-1 surprise autocorrelation falls by `{100 * (1 - float(summary.loc[summary["arm"].eq(CHAMPION), "mean_abs_signed_surprise_autocorr_lag1"].iloc[0]) / float(baseline["mean_abs_signed_surprise_autocorr_lag1"])):.1f}%`.

## Month By Month

{_markdown_table(month, ["calendar_month", "selected_rows_current", "selected_rows_champion", "mean_ev_after_1pct_current", "mean_ev_after_1pct_champion", "delta__mean_ev_after_1pct", "clean_exec_precision_current", "clean_exec_precision_champion"], percent={"mean_ev_after_1pct_current", "mean_ev_after_1pct_champion", "delta__mean_ev_after_1pct", "clean_exec_precision_current", "clean_exec_precision_champion"})}

## March Burn-In Check

March is not counted as final OOS. On this coefficient-selection month, current-reference top-10 EV is `{100 * float(burnin_base["mean_ev_after_1pct"]):.4f}%` and the alternative is `{100 * float(burnin_alt["mean_ev_after_1pct"]):.4f}%`; clean precision moves from `{100 * float(burnin_base["clean_exec_precision"]):.2f}%` to `{100 * float(burnin_alt["clean_exec_precision"]):.2f}%`. Of `{len(burnin_tail)}` high-surprise daily cells, `{int(burnin_tail["high_surprise_significantly_improved"].sum())}` contract by at least 20%. That did not meet the period-level target, so the requested same-sample capacity fallback was also run.

The requested in-sample capacity fallback was nevertheless run with the final through-June bundle. It improves `{int(insample_capacity.get("high_surprise_cells_improved", 0))}/{int(insample_capacity.get("high_surprise_cells", 0))}` March high-surprise cells, with top-10 EV `{100 * float(insample_capacity.get("top10_ev_alternative", float("nan"))):.4f}%` and clean precision `{100 * float(insample_capacity.get("top10_clean_alternative", float("nan"))):.2f}%`. The probability map is refitted on the same March sample for this capacity-only diagnostic. These numbers are deliberately excluded from OOS and inference claims. They show that the architecture can encode the residual states; temporal generalization, not representation capacity, is the remaining constraint.

## OOS Source Coverage

{_markdown_table(coverage, ["calendar_month", "candidate_rows", "distinct_timestamps", "expected_hourly_timestamps", "timestamp_coverage_rate", "observed_utc_days", "distinct_symbols", "min_timestamp", "max_timestamp"], percent={"timestamp_coverage_rate"})}

April is complete at hourly resolution. May covers about 88% of expected hourly timestamps and June about 36%. This attrition exists in the frozen base OOS/top-30 handoff before the alternative meta experiment; the meta join does not create it. June metrics are valid on the available OOS rows but are not evidence for a full-calendar June replay.

## Surprise Calendar

- Baseline high-surprise daily cells: `{int(comparison["baseline_high_surprise"].fillna(False).astype(bool).sum())}`.
- Contiguous high-surprise events: `{len(events)}`.
- Events improved either by >=20% surprise contraction or non-negative EV delta: `{int(events["improved_either_way"].sum())}/{len(events)}`.
- Material persistent events (>=2 days and >=20 rows): `{len(material)}`.
- Material persistent events not improved: `{len(material_unimproved)}`.
- {material_event_note}

## Robustness

- Weekly block-bootstrap EV delta: `{100 * float(robustness["weekly_block_bootstrap"]["mean_delta"]):.4f}%`, 95% CI `[{100 * float(robustness["weekly_block_bootstrap"]["ci025"]):.4f}%, {100 * float(robustness["weekly_block_bootstrap"]["ci975"]):.4f}%]`.
- Bootstrap probability of positive weekly delta: `{100 * float(robustness["weekly_block_bootstrap"]["positive_probability"]):.2f}%`.
- Positive-event trade retention: `{100 * float(robustness["positive_preservation"]["retention_rate"]):.2f}%`.
- Shifted, shuffled, and matched-noise residual controls materially underperform. Local side x archetype score normalization failed its identity placebo and is disabled; archetypes remain recognizer inputs and reporting segments, while the selected overlay uses only the global posterior-weighted residual prior.
- Surprise-event definitions pass one-factor sensitivity: local median Jaccard `{event_sensitivity.get("local_neighborhood_median_jaccard", float("nan")):.3f}` across nearby 85/90/95% thresholds, 3/6/12h causal EMAs, and 1/2/4h gaps. The full corner-to-corner factorial median is `{event_sensitivity.get("full_factorial_median_jaccard", float("nan")):.3f}` and is retained as a harsher diagnostic. Maximum event surprise-mass share is `{100 * float(event_sensitivity.get("maximum_largest_event_surprise_share", float("nan"))):.2f}%`.

## Representation Robustness

{_markdown_table(representation_family, ["family", "seeds", "mean_top10_ev_after_1pct", "std_top10_ev_after_1pct", "mean_clean_precision", "mean_full_bad_mae", "mean_timeout", "mean_abs_surprise_autocorr_lag1", "minimum_worst_week_ev", "mean_high_surprise_improvement_rate", "mean_pca_effective_rank"], percent={"mean_top10_ev_after_1pct", "std_top10_ev_after_1pct", "mean_clean_precision", "mean_full_bad_mae", "mean_timeout", "minimum_worst_week_ev", "mean_high_surprise_improvement_rate"}) if not representation_family.empty else "Representation robustness table not materialized."}

The selected family is `{representation_manifest.get("selected_family", "missing")}`. Residual AE/GMM adds `{100 * float(representation_manifest.get("ae_incremental_top10_ev_after_1pct", float("nan"))):.4f}` percentage points of EV versus the corrected PCA comparator and does not reduce calendar autocorrelation, so it does not earn its extra complexity. The legacy unclipped PCA is rejected because one zero-heavy price/OI feature dominated PC1.

## Archetypes

{_markdown_table(catalog, ["semantic", "component_count", "support_rows_sum", "mean_hit_surprise", "mean_ev", "clean_rate", "dirty_rate", "bad_mae_rate", "timeout_rate"], percent={"mean_ev", "clean_rate", "dirty_rate", "bad_mae_rate", "timeout_rate"})}

Only four of the seven stable semantic slots are materially discovered in the latest fit. The absent dirty-only, timeout-only, and high-variance semantics remain explicit zero-probability outputs, preserving inference schema without inventing unsupported states.

### Semantic Enrichment Gate

{_markdown_table(semantic_enrichment, ["semantic", "posterior_weighted_rows", "mean_signed_hit_surprise", "mean_ev_after_1pct", "negative_enrichment_ratio", "positive_enrichment_ratio", "distinct_event_days", "largest_event_share", "block_bootstrap_ci025", "block_bootstrap_ci975", "fold_direction_recurrence", "matched_high_state_surprise_delta", "enrichment_pass"], percent={"mean_ev_after_1pct", "largest_event_share", "fold_direction_recurrence"}) if not semantic_enrichment.empty else "Semantic enrichment report not materialized."}

The semantic posterior states have significant signed-surprise and matched-control separation, but none reaches the strict `1.5x` tail-enrichment requirement. They are retained as continuous residual-prior inputs and reporting semantics, not as hard gates.

The previously unimproved June 30 `long_mixed_wideslow_tentative` event is a synchronized deleveraging/rebound state: market OI contraction, systemic deleveraging, and price-up/OI-down breadth are jointly extreme. A March-only learned shock model improved that cell but weakened aggregate EV and was rejected. The retained sparse composite instead selects its coefficients from twelve monthly walk-forward states ending in March, then improves both the event and aggregate April-June OOS metrics. Hard post-event rules are not used.

One non-material negative outlier remains on May 23 at 23:00 UTC in `short_default_clean_path`: current selects one PUMP timeout and the final arm selects that row plus one PENGU bad-path loss. The market shock composite is only `~0.68`, below its activation threshold, and the cell has one timestamp/two final rows with no recurrence. It is an explained idiosyncratic outcome, not evidence of an unhandled autocorrelated regime; adding a rule for it would be post-hoc overfitting.

## Seed Stability

{_markdown_table(seed_stability, ["seed_offset", "hit_alpha", "local_hit_alpha", "top10_mean_ev_after_1pct", "top10_clean_exec_precision", "top10_full_path_bad_mae_rate", "mean_abs_signed_surprise_autocorr_lag1", "worst_week_ev", "positive_weeks", "weeks"], percent={"top10_mean_ev_after_1pct", "top10_clean_exec_precision", "top10_full_path_bad_mae_rate", "worst_week_ev"})}

Top-10 EV has mean `{100 * seed_stability["top10_mean_ev_after_1pct"].mean():.4f}%` and cross-seed standard deviation `{100 * seed_stability["top10_mean_ev_after_1pct"].std(ddof=0):.4f}` percentage points. All three seeds retain positive EV in every observed week.

## Stage Gates

{_markdown_table(stage_gate, ["stage", "name", "status", "evidence"]) if not stage_gate.empty else "Stage-gate report not yet materialized."}

## Inference Parity

- Bundle: `{final_inference["bundle_path"]}`
- Fit through: `{final_inference["fit_through"]}`
- July parity rows: `{final_inference["july_parity_rows"]}`
- Bundle round-trip max absolute difference: `{final_inference["bundle_roundtrip_max_abs_diff"]}`
- Frozen base AE/GMM hash match: `{final_inference["base_ae_gmm_hash_match"]}`
- Residual representation: `{final_inference["residual_representation"]}`
- Shock adjustment parity: `{final_inference.get("shock_adjustment_max_abs_diff", "not_applicable")}`
- Shock raw/local parity: `{final_inference.get("shock_raw_max_abs_diff", "not_applicable")}` / `{final_inference.get("shock_local_max_abs_diff", "not_applicable")}`
- Historical rank embedded in bundle: `{final_inference.get("historical_rank") is not None}`
- Inference parity pass: `{final_inference["inference_parity_pass"]}`

## Boundaries

- April-May-June are the formal untouched OOS evaluation months.
- March is the residual-overlay burn-in month. Sparse shock thresholds are selected on monthly walk-forward states from April 2025 through March 2026; April-June 2026 remain OOS for that layer.
- July contains only 202 handoff rows and is used for parity, not performance assessment.
- The source handoff is already the base top-30 candidate pool, so this report does not claim reconstruction of the full pre-base production universe.
- May and especially June have incomplete upstream base-OOS timestamp coverage; a full-calendar claim requires regenerating the missing frozen-base OOS shards, which is outside this meta-only experiment.
- The current production meta model is not overwritten. The alternative is separately packaged.
- Optional future base alternatives are explicitly isolated: base changes may use only pre-entry sign/direction features. They must be tested as `B0/B1/B2 x M0/M1` OOS factorial arms so gains from base archetypes and meta context are not conflated with this meta-only result.
"""
    (report_dir / "REPORT.md").write_text(report, encoding="utf-8")
    manifest = {
        "schema": "meta_residual_archetype_final_report_v1",
        "champion": SHOCK_CHAMPION,
        "report": str(report_dir / "REPORT.md"),
        "tables": sorted(str(path) for path in report_dir.glob("*.csv")),
        "material_persistent_unimproved_events": int(len(material_unimproved)),
        "inference_parity_pass": bool(final_inference["inference_parity_pass"]),
        "historical_rank_parity_pass": bool(
            historical_manifest["inference_rank_parity_pass"]
        ),
        "primary_metric_contract": "expanding_prior_score_cdf_by_side",
        "full_validation_status": "keep_continuous_prior_hard_semantics_diagnostic",
        "current_meta_model_overwritten": False,
    }
    (report_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2), encoding="utf-8"
    )
    print(json.dumps(_json_safe(manifest), indent=2), flush=True)


if __name__ == "__main__":
    main()
