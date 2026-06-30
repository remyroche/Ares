#!/usr/bin/env python3
"""Run head-agnostic T16 dynamic HR-surprise meta/similarity ablations."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_CANDIDATES = Path(
    "data_perp/reports/performance_market_state_modulator_ablation_report_20260627_badregime_v2/"
    "continuous_broad_candidate_ledger_audit_mixed/continuous_broad_candidate_ledger.parquet"
)
DEFAULT_POLICY_PARAMS = Path(
    "data_perp/artifacts/20260620_185313_no_mkt4_evband002_policy_uncertainty_ev/"
    "best_policy_params.json"
)
DEFAULT_ROOT = Path(
    "data_perp/reports/finalfit_broad_candidate_regen_20260627/"
    "dynamic_hr_surprise_meta_similarity_ablation_20260628"
)


@dataclass(frozen=True)
class AblationSpec:
    trial_id: str
    name: str
    description: str
    overrides: tuple[str, ...]


T16_OVERRIDES: tuple[str, ...] = (
    "--threshold-selection-objective",
    "recent_daily_quantile",
    "--recent-quantile-days",
    "30.0",
    "--recent-quantile-level",
    "0.42",
    "--recent-quantile-weight-mode",
    "bucket",
    "--recent-quantile-weight-last-7",
    "1.1",
    "--recent-quantile-weight-prev-7",
    "0.3",
    "--recent-quantile-weight-older",
    "0.5",
    "--recent-validation-guard",
    "--recent-validation-min-hit-rate",
    "0.35",
)


META_CONSERVATIVE: tuple[str, ...] = (
    "--use-meta-context-features",
    "--disable-meta-context-enable-tuning",
    "--meta-context-transform",
    "causal_percentile",
    "--meta-context-feature-aggregation",
    "mean",
    "--meta-context-timestamp-aggregation",
    "mean",
    "--meta-drift-raise-min",
    "0.0",
    "--meta-drift-raise-max",
    "0.35",
    "--meta-drift-floor-min",
    "0.60",
    "--meta-drift-floor-max",
    "0.98",
    "--meta-uncertainty-raise-min",
    "0.0",
    "--meta-uncertainty-raise-max",
    "0.60",
    "--meta-uncertainty-floor-min",
    "0.60",
    "--meta-uncertainty-floor-max",
    "0.98",
)


LINEAR_CONTEXT_MILD: tuple[str, ...] = (
    "--context-linear-density-raise",
    "0.20",
    "--context-linear-density-floor",
    "0.55",
    "--context-linear-relaxation-dampen",
    "1.25",
    "--context-linear-pressure-raise",
    "0.25",
    "--context-linear-lowering-penalty-strength",
    "4.0",
    "--context-linear-z-weight",
    "1.0",
    "--context-linear-similarity-weight",
    "0.50",
    "--context-linear-density-weight",
    "0.75",
    "--context-linear-drift-weight",
    "0.50",
    "--context-linear-uncertainty-weight",
    "0.75",
    "--context-linear-z-scale",
    "1.0",
    "--context-linear-similarity-scale",
    "6.0",
    "--context-linear-meta-center",
    "0.50",
)


LINEAR_CONTEXT_STRONG: tuple[str, ...] = (
    "--context-linear-density-raise",
    "0.35",
    "--context-linear-density-floor",
    "0.45",
    "--context-linear-relaxation-dampen",
    "1.75",
    "--context-linear-pressure-raise",
    "0.40",
    "--context-linear-lowering-penalty-strength",
    "7.0",
    "--context-linear-z-weight",
    "1.0",
    "--context-linear-similarity-weight",
    "0.60",
    "--context-linear-density-weight",
    "1.0",
    "--context-linear-drift-weight",
    "0.75",
    "--context-linear-uncertainty-weight",
    "1.0",
    "--context-linear-z-scale",
    "1.0",
    "--context-linear-similarity-scale",
    "6.0",
    "--context-linear-meta-center",
    "0.50",
)


LINEAR_CONTEXT_GRID_ONLY: tuple[str, ...] = (
    "--context-linear-lowering-penalty-strength",
    "5.0",
    "--context-linear-z-weight",
    "1.0",
    "--context-linear-similarity-weight",
    "0.60",
    "--context-linear-density-weight",
    "1.0",
    "--context-linear-drift-weight",
    "0.75",
    "--context-linear-uncertainty-weight",
    "1.0",
    "--context-linear-z-scale",
    "1.0",
    "--context-linear-similarity-scale",
    "6.0",
    "--context-linear-meta-center",
    "0.50",
)


def _linear_context_grid_only_with_strength(strength: str) -> tuple[str, ...]:
    return (
        "--context-linear-lowering-penalty-strength",
        strength,
        "--context-linear-z-weight",
        "1.0",
        "--context-linear-similarity-weight",
        "0.60",
        "--context-linear-density-weight",
        "0.75",
        "--context-linear-drift-weight",
        "0.50",
        "--context-linear-uncertainty-weight",
        "0.75",
        "--context-linear-z-scale",
        "1.0",
        "--context-linear-similarity-scale",
        "6.0",
        "--context-linear-meta-center",
        "0.50",
    )


LINEAR_CONTEXT_GRID_ONLY_WEAK: tuple[str, ...] = _linear_context_grid_only_with_strength("1.0")
LINEAR_CONTEXT_GRID_ONLY_MEDIUM: tuple[str, ...] = _linear_context_grid_only_with_strength("2.0")


def _a09_top3_similarity() -> tuple[str, ...]:
    return (
        "--similarity-prior-enable",
        "--similarity-prior-ev-weight",
        "0.50",
        "--similarity-prior-top-k-days",
        "3",
        "--similarity-prior-min-days",
        "3",
        "--similarity-prior-temperature",
        "1.0",
    )


A09_TOP3_SIMILARITY: tuple[str, ...] = _a09_top3_similarity()


ABLATIONS: tuple[AblationSpec, ...] = (
    AblationSpec(
        "A00",
        "t16_base_meta_ledger",
        "T16 reproduced on the meta-rich continuous ledger.",
        T16_OVERRIDES,
    ),
    AblationSpec(
        "A01",
        "t16_meta_raise_only",
        "T16 plus head-agnostic per-head drift/uncertainty raise-only threshold pressure.",
        T16_OVERRIDES + META_CONSERVATIVE + ("--meta-context-action-mode", "raise"),
    ),
    AblationSpec(
        "A02",
        "t16_meta_dampen_relaxation",
        "T16 plus head-agnostic per-head drift/uncertainty dampening of threshold relaxation.",
        T16_OVERRIDES + META_CONSERVATIVE + ("--meta-context-action-mode", "dampen_relaxation"),
    ),
    AblationSpec(
        "A03",
        "t16_meta_bad_surprise_raise",
        "T16 plus head-agnostic per-head meta raise only when HR surprise is negative.",
        T16_OVERRIDES
        + META_CONSERVATIVE
        + ("--meta-context-action-mode", "bad_surprise_raise", "--meta-context-bad-z-threshold", "0.0"),
    ),
    AblationSpec(
        "A04",
        "t16_similarity_ev025",
        "T16 plus causal same-head similar-period EV prior in daily Y selection.",
        T16_OVERRIDES
        + (
            "--similarity-prior-enable",
            "--similarity-prior-ev-weight",
            "0.25",
            "--similarity-prior-top-k-days",
            "5",
            "--similarity-prior-min-days",
            "3",
            "--similarity-prior-temperature",
            "1.0",
        ),
    ),
    AblationSpec(
        "A05",
        "t16_similarity_ev050",
        "T16 plus stronger causal same-head similar-period EV prior in daily Y selection.",
        T16_OVERRIDES
        + (
            "--similarity-prior-enable",
            "--similarity-prior-ev-weight",
            "0.50",
            "--similarity-prior-top-k-days",
            "5",
            "--similarity-prior-min-days",
            "3",
            "--similarity-prior-temperature",
            "1.0",
        ),
    ),
    AblationSpec(
        "A06",
        "t16_meta_dampen_plus_similarity_ev025",
        "T16 plus both generic meta relaxation dampening and causal similar-period EV prior.",
        T16_OVERRIDES
        + META_CONSERVATIVE
        + (
            "--meta-context-action-mode",
            "dampen_relaxation",
            "--similarity-prior-enable",
            "--similarity-prior-ev-weight",
            "0.25",
            "--similarity-prior-top-k-days",
            "5",
            "--similarity-prior-min-days",
            "3",
            "--similarity-prior-temperature",
            "1.0",
        ),
    ),
    AblationSpec(
        "A07",
        "t16_similarity_ev075",
        "T16 plus stronger causal same-head similar-period EV prior in daily Y selection.",
        T16_OVERRIDES
        + (
            "--similarity-prior-enable",
            "--similarity-prior-ev-weight",
            "0.75",
            "--similarity-prior-top-k-days",
            "5",
            "--similarity-prior-min-days",
            "3",
            "--similarity-prior-temperature",
            "1.0",
        ),
    ),
    AblationSpec(
        "A08",
        "t16_similarity_ev100",
        "T16 plus very strong causal same-head similar-period EV prior in daily Y selection.",
        T16_OVERRIDES
        + (
            "--similarity-prior-enable",
            "--similarity-prior-ev-weight",
            "1.00",
            "--similarity-prior-top-k-days",
            "5",
            "--similarity-prior-min-days",
            "3",
            "--similarity-prior-temperature",
            "1.0",
        ),
    ),
    AblationSpec(
        "A09",
        "t16_similarity_ev050_top3",
        "T16 plus EV prior using only the 3 most similar prior days.",
        T16_OVERRIDES + A09_TOP3_SIMILARITY,
    ),
    AblationSpec(
        "A10",
        "t16_similarity_ev050_top10",
        "T16 plus EV prior using the 10 most similar prior days.",
        T16_OVERRIDES
        + (
            "--similarity-prior-enable",
            "--similarity-prior-ev-weight",
            "0.50",
            "--similarity-prior-top-k-days",
            "10",
            "--similarity-prior-min-days",
            "3",
            "--similarity-prior-temperature",
            "1.0",
        ),
    ),
    AblationSpec(
        "A11",
        "t16_similarity_ev050_hr025",
        "T16 plus EV prior and a modest similar-period hit-rate bonus.",
        T16_OVERRIDES
        + (
            "--similarity-prior-enable",
            "--similarity-prior-ev-weight",
            "0.50",
            "--similarity-prior-hr-weight",
            "0.25",
            "--similarity-prior-hr-floor",
            "0.35",
            "--similarity-prior-top-k-days",
            "5",
            "--similarity-prior-min-days",
            "3",
            "--similarity-prior-temperature",
            "1.0",
        ),
    ),
    AblationSpec(
        "A12",
        "t16_meta_bad_raise_plus_similarity_ev050",
        "T16 plus generic bad-surprise meta raise and causal similar-period EV prior.",
        T16_OVERRIDES
        + META_CONSERVATIVE
        + (
            "--meta-context-action-mode",
            "bad_surprise_raise",
            "--meta-context-bad-z-threshold",
            "0.0",
            "--similarity-prior-enable",
            "--similarity-prior-ev-weight",
            "0.50",
            "--similarity-prior-top-k-days",
            "5",
            "--similarity-prior-min-days",
            "3",
            "--similarity-prior-temperature",
            "1.0",
        ),
    ),
    AblationSpec(
        "A13",
        "t16_linear_context_mild_top3",
        "T16 top3 similarity with continuous linear meta/density dampening and lowering penalty.",
        T16_OVERRIDES
        + META_CONSERVATIVE
        + (
            "--meta-context-action-mode",
            "linear_context",
            "--similarity-prior-enable",
            "--similarity-prior-ev-weight",
            "0.50",
            "--similarity-prior-top-k-days",
            "3",
            "--similarity-prior-min-days",
            "3",
            "--similarity-prior-temperature",
            "1.0",
        )
        + LINEAR_CONTEXT_MILD,
    ),
    AblationSpec(
        "A14",
        "t16_linear_context_strong_top3",
        "T16 top3 similarity with stronger continuous linear context dampening.",
        T16_OVERRIDES
        + META_CONSERVATIVE
        + (
            "--meta-context-action-mode",
            "linear_context",
            "--similarity-prior-enable",
            "--similarity-prior-ev-weight",
            "0.50",
            "--similarity-prior-top-k-days",
            "3",
            "--similarity-prior-min-days",
            "3",
            "--similarity-prior-temperature",
            "1.0",
        )
        + LINEAR_CONTEXT_STRONG,
    ),
    AblationSpec(
        "A15",
        "t16_linear_context_grid_only_top3",
        "T16 top3 similarity with continuous context-aware Y selection but no row-level context offset.",
        T16_OVERRIDES
        + META_CONSERVATIVE
        + (
            "--meta-context-action-mode",
            "dampen_relaxation",
            "--context-linear-density-raise",
            "0.0",
            "--context-linear-relaxation-dampen",
            "0.0",
            "--context-linear-pressure-raise",
            "0.0",
            "--similarity-prior-enable",
            "--similarity-prior-ev-weight",
            "0.50",
            "--similarity-prior-top-k-days",
            "3",
            "--similarity-prior-min-days",
            "3",
            "--similarity-prior-temperature",
            "1.0",
        )
        + LINEAR_CONTEXT_GRID_ONLY,
    ),
    AblationSpec(
        "A16",
        "t16_linear_context_grid_only_weak_top3",
        "T16 top3 similarity with weak continuous context-aware Y lowering penalty.",
        T16_OVERRIDES
        + META_CONSERVATIVE
        + (
            "--meta-context-action-mode",
            "dampen_relaxation",
            "--context-linear-density-raise",
            "0.0",
            "--context-linear-relaxation-dampen",
            "0.0",
            "--context-linear-pressure-raise",
            "0.0",
            "--similarity-prior-enable",
            "--similarity-prior-ev-weight",
            "0.50",
            "--similarity-prior-top-k-days",
            "3",
            "--similarity-prior-min-days",
            "3",
            "--similarity-prior-temperature",
            "1.0",
        )
        + LINEAR_CONTEXT_GRID_ONLY_WEAK,
    ),
    AblationSpec(
        "A17",
        "t16_linear_context_grid_only_medium_top3",
        "T16 top3 similarity with medium continuous context-aware Y lowering penalty.",
        T16_OVERRIDES
        + META_CONSERVATIVE
        + (
            "--meta-context-action-mode",
            "dampen_relaxation",
            "--context-linear-density-raise",
            "0.0",
            "--context-linear-relaxation-dampen",
            "0.0",
            "--context-linear-pressure-raise",
            "0.0",
            "--similarity-prior-enable",
            "--similarity-prior-ev-weight",
            "0.50",
            "--similarity-prior-top-k-days",
            "3",
            "--similarity-prior-min-days",
            "3",
            "--similarity-prior-temperature",
            "1.0",
        )
        + LINEAR_CONTEXT_GRID_ONLY_MEDIUM,
    ),
    AblationSpec(
        "A18",
        "t16_linear_context_true_grid_only_weak_top3",
        "T16 top3 similarity with weak continuous Y penalty and no row-level context offset.",
        T16_OVERRIDES
        + META_CONSERVATIVE
        + (
            "--meta-context-action-mode",
            "linear_context",
            "--context-linear-density-raise",
            "0.0",
            "--context-linear-relaxation-dampen",
            "0.0",
            "--context-linear-pressure-raise",
            "0.0",
            "--similarity-prior-enable",
            "--similarity-prior-ev-weight",
            "0.50",
            "--similarity-prior-top-k-days",
            "3",
            "--similarity-prior-min-days",
            "3",
            "--similarity-prior-temperature",
            "1.0",
        )
        + LINEAR_CONTEXT_GRID_ONLY_WEAK,
    ),
    AblationSpec(
        "A19",
        "t16_linear_context_true_grid_only_medium_top3",
        "T16 top3 similarity with medium continuous Y penalty and no row-level context offset.",
        T16_OVERRIDES
        + META_CONSERVATIVE
        + (
            "--meta-context-action-mode",
            "linear_context",
            "--context-linear-density-raise",
            "0.0",
            "--context-linear-relaxation-dampen",
            "0.0",
            "--context-linear-pressure-raise",
            "0.0",
            "--similarity-prior-enable",
            "--similarity-prior-ev-weight",
            "0.50",
            "--similarity-prior-top-k-days",
            "3",
            "--similarity-prior-min-days",
            "3",
            "--similarity-prior-temperature",
            "1.0",
        )
        + LINEAR_CONTEXT_GRID_ONLY_MEDIUM,
    ),
    AblationSpec(
        "A20",
        "a09_q50",
        "A09 with a stricter daily Y quantile level of 0.50.",
        T16_OVERRIDES
        + A09_TOP3_SIMILARITY
        + ("--recent-quantile-level", "0.50"),
    ),
    AblationSpec(
        "A21",
        "a09_q55",
        "A09 with a stricter daily Y quantile level of 0.55.",
        T16_OVERRIDES
        + A09_TOP3_SIMILARITY
        + ("--recent-quantile-level", "0.55"),
    ),
    AblationSpec(
        "A22",
        "a09_q50_hr40",
        "A09 q50 with a stricter recent-validation hit-rate guard of 40%.",
        T16_OVERRIDES
        + A09_TOP3_SIMILARITY
        + ("--recent-quantile-level", "0.50", "--recent-validation-min-hit-rate", "0.40"),
    ),
    AblationSpec(
        "A23",
        "a09_q55_hr40",
        "A09 q55 with a stricter recent-validation hit-rate guard of 40%.",
        T16_OVERRIDES
        + A09_TOP3_SIMILARITY
        + ("--recent-quantile-level", "0.55", "--recent-validation-min-hit-rate", "0.40"),
    ),
    AblationSpec(
        "A24",
        "a09_q50_wlower015",
        "A09 q50 with lower maximum threshold-lowering sensitivity.",
        T16_OVERRIDES
        + A09_TOP3_SIMILARITY
        + ("--recent-quantile-level", "0.50", "--w-lower-max", "0.15"),
    ),
    AblationSpec(
        "A25",
        "a09_q55_wlower015",
        "A09 q55 with lower maximum threshold-lowering sensitivity.",
        T16_OVERRIDES
        + A09_TOP3_SIMILARITY
        + ("--recent-quantile-level", "0.55", "--w-lower-max", "0.15"),
    ),
    AblationSpec(
        "A26",
        "a09_q50_rank075",
        "A09 q50 with a stricter top-rank floor of 0.75.",
        T16_OVERRIDES
        + A09_TOP3_SIMILARITY
        + ("--recent-quantile-level", "0.50", "--top-rank-floor", "0.75"),
    ),
    AblationSpec(
        "A27",
        "a09_q55_rank075",
        "A09 q55 with a stricter top-rank floor of 0.75.",
        T16_OVERRIDES
        + A09_TOP3_SIMILARITY
        + ("--recent-quantile-level", "0.55", "--top-rank-floor", "0.75"),
    ),
    AblationSpec(
        "A28",
        "a09_quality_hr45_keep80",
        "A09 with causal per-head p_hit floors fitted to keep at least 80% of dynamic selections while targeting 45% training HR.",
        T16_OVERRIDES
        + A09_TOP3_SIMILARITY
        + (
            "--quality-gate-enable",
            "--quality-gate-target-hit-rate",
            "0.45",
            "--quality-gate-min-keep-fraction",
            "0.80",
            "--quality-gate-min-total-pnl",
            "0.0",
        ),
    ),
    AblationSpec(
        "A29",
        "a09_quality_hr45_keep70",
        "A09 with causal per-head p_hit floors fitted to keep at least 70% of dynamic selections while targeting 45% training HR.",
        T16_OVERRIDES
        + A09_TOP3_SIMILARITY
        + (
            "--quality-gate-enable",
            "--quality-gate-target-hit-rate",
            "0.45",
            "--quality-gate-min-keep-fraction",
            "0.70",
            "--quality-gate-min-total-pnl",
            "0.0",
        ),
    ),
    AblationSpec(
        "A30",
        "a09_quality_hr44_keep80",
        "A09 with causal per-head p_hit floors fitted to keep at least 80% of dynamic selections while targeting 44% training HR.",
        T16_OVERRIDES
        + A09_TOP3_SIMILARITY
        + (
            "--quality-gate-enable",
            "--quality-gate-target-hit-rate",
            "0.44",
            "--quality-gate-min-keep-fraction",
            "0.80",
            "--quality-gate-min-total-pnl",
            "0.0",
        ),
    ),
    AblationSpec(
        "A31",
        "a09_quality_hr45_keep80_deact",
        "A09 with causal per-head p_hit floors targeting 45% training HR and allowing full head closure when no qualifying floor exists.",
        T16_OVERRIDES
        + A09_TOP3_SIMILARITY
        + (
            "--quality-gate-enable",
            "--quality-gate-target-hit-rate",
            "0.45",
            "--quality-gate-min-keep-fraction",
            "0.80",
            "--quality-gate-min-total-pnl",
            "0.0",
            "--quality-gate-allow-deactivation",
            "--quality-gate-deactivate-if-no-pass",
        ),
    ),
)


def _common_compare_args(
    candidates: Path,
    policy_params: Path,
    output_dir: Path,
    *,
    calendar_eval_start: str,
    calendar_eval_end: str,
    calendar_xw_min_train_days: float,
    calendar_xw_max_train_days: float,
    calendar_y_train_days: float,
) -> list[str]:
    return [
        sys.executable,
        "-u",
        "scripts/compare_dynamic_hr_surprise_threshold.py",
        "--candidates",
        str(candidates),
        "--policy-params",
        str(policy_params),
        "--output-dir",
        str(output_dir),
        "--calendar-only",
        "--calendar-eval-start",
        calendar_eval_start,
        "--calendar-eval-end",
        calendar_eval_end,
        "--calendar-xw-min-train-days",
        str(calendar_xw_min_train_days),
        "--calendar-xw-max-train-days",
        str(calendar_xw_max_train_days),
        "--calendar-y-train-days",
        str(calendar_y_train_days),
        "--disable-deployed-threshold-floor",
        "--head-optimization-mode",
        "independent",
        "--threshold-refresh-mode",
        "grid",
        "--top-rank-floor",
        "0.70",
        "--trials",
        "120",
        "--threshold-grid-size",
        "201",
        "--x-min-days",
        "1.0",
        "--x-max-days",
        "28.0",
        "--w-lower-min",
        "0.0",
        "--w-lower-max",
        "0.25",
        "--w-raise-min",
        "0.0",
        "--w-raise-max",
        "0.60",
        "--y-min",
        "-0.50",
        "--y-max",
        "1.50",
        "--z-clip",
        "5.0",
        "--subwindow-constraints-mode",
        "penalty",
        "--subwindow-days",
        "5.0",
        "--min-subwindows",
        "4",
        "--min-positive-objective-fraction",
        "0.25",
        "--subwindow-q15-floor",
        "-1.00",
        "--subwindow-drawdown-floor",
        "-3.00",
        "--lambda-iqr",
        "0.25",
        "--lambda-tail",
        "0.50",
        "--subwindow-constraint-penalty",
        "10.0",
        "--min-threshold-selected-count",
        "0",
        "--min-threshold-active-subwindows",
        "0",
        "--deployed-threshold-soft-prior-strength",
        "8.0",
        "--deployed-threshold-soft-prior-deadband",
        "0.03",
        "--deployed-threshold-soft-prior-power",
        "2.0",
        "--deployed-threshold-soft-prior-activity-weight",
        "0.25",
    ]


def _week_start(ts: pd.Series) -> pd.Series:
    dates = pd.to_datetime(ts, utc=True, errors="coerce").dt.floor("D")
    return dates - pd.to_timedelta(dates.dt.weekday, unit="D")


def _metrics(
    output_dir: Path,
    spec: AblationSpec,
    *,
    eval_start: str,
    eval_end: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    selected_path = output_dir / "calendar_dynamic_hr_surprise_selected_rows.parquet"
    selected = pd.read_parquet(selected_path).copy() if selected_path.exists() else pd.DataFrame()
    if len(selected):
        selected["timestamp"] = pd.to_datetime(selected["timestamp"], utc=True, errors="coerce")
        selected["week_start"] = _week_start(selected["timestamp"])
        selected["hit"] = pd.to_numeric(selected["net_return"], errors="coerce").gt(0.0).astype(float)
    eval_start_ts = pd.Timestamp(eval_start)
    if eval_start_ts.tzinfo is None:
        eval_start_ts = eval_start_ts.tz_localize("UTC")
    else:
        eval_start_ts = eval_start_ts.tz_convert("UTC")
    eval_end_ts = pd.Timestamp(eval_end)
    if eval_end_ts.tzinfo is None:
        eval_end_ts = eval_end_ts.tz_localize("UTC")
    else:
        eval_end_ts = eval_end_ts.tz_convert("UTC")
    week_index = pd.date_range(
        eval_start_ts.floor("D") - pd.Timedelta(days=eval_start_ts.weekday()),
        eval_end_ts,
        freq="W-MON",
    )
    if len(selected):
        weekly = (
            selected.groupby("week_start", observed=True)
            .agg(pnl_net_spread=("net_return", "sum"), trades=("net_return", "size"), hits=("hit", "sum"))
            .reindex(week_index, fill_value=0.0)
            .rename_axis("week_start")
            .reset_index()
        )
    else:
        weekly = pd.DataFrame({"week_start": week_index, "pnl_net_spread": 0.0, "trades": 0.0, "hits": 0.0})
    weekly["trial_id"] = spec.trial_id
    weekly["variant"] = spec.name
    weekly["hit_rate"] = np.divide(
        weekly["hits"].to_numpy(dtype=float),
        weekly["trades"].to_numpy(dtype=float),
        out=np.full(len(weekly), np.nan),
        where=weekly["trades"].to_numpy(dtype=float) > 0,
    )
    weekly["pnl_per_trade"] = np.divide(
        weekly["pnl_net_spread"].to_numpy(dtype=float),
        weekly["trades"].to_numpy(dtype=float),
        out=np.full(len(weekly), np.nan),
        where=weekly["trades"].to_numpy(dtype=float) > 0,
    )
    by_head = (
        selected.groupby("head", observed=True)
        .agg(pnl_net_spread=("net_return", "sum"), trades=("net_return", "size"), hits=("hit", "sum"))
        .reset_index()
        if len(selected)
        else pd.DataFrame(columns=["head", "pnl_net_spread", "trades", "hits"])
    )
    by_head["trial_id"] = spec.trial_id
    by_head["variant"] = spec.name
    by_head["hit_rate"] = np.divide(
        by_head["hits"].to_numpy(dtype=float),
        by_head["trades"].to_numpy(dtype=float),
        out=np.full(len(by_head), np.nan),
        where=by_head["trades"].to_numpy(dtype=float) > 0,
    )
    by_head["pnl_per_trade"] = np.divide(
        by_head["pnl_net_spread"].to_numpy(dtype=float),
        by_head["trades"].to_numpy(dtype=float),
        out=np.full(len(by_head), np.nan),
        where=by_head["trades"].to_numpy(dtype=float) > 0,
    )
    total_pnl = float(weekly["pnl_net_spread"].sum())
    trades = int(weekly["trades"].sum())
    hits = float(weekly["hits"].sum())
    metrics = {
        "trial_id": spec.trial_id,
        "variant": spec.name,
        "description": spec.description,
        "total_pnl_net_spread": total_pnl,
        "trades": trades,
        "hit_rate": float(hits / trades) if trades else np.nan,
        "pnl_per_trade": float(total_pnl / trades) if trades else np.nan,
        "worst_week_pnl": float(weekly["pnl_net_spread"].min()) if len(weekly) else 0.0,
        "q15_week_pnl": float(weekly["pnl_net_spread"].quantile(0.15)) if len(weekly) else 0.0,
        "q05_week_pnl": float(weekly["pnl_net_spread"].quantile(0.05)) if len(weekly) else 0.0,
        "positive_week_fraction": float((weekly["pnl_net_spread"] > 0.0).mean()) if len(weekly) else 0.0,
        "output_dir": str(output_dir),
    }
    return metrics, weekly, by_head


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6g}")
        elif pd.api.types.is_datetime64_any_dtype(display[col]):
            display[col] = display[col].astype(str)
        else:
            display[col] = display[col].astype(str)
    headers = [str(col) for col in display.columns]
    rows = display.astype(str).values.tolist()
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def _write_ledger(path: Path, summary: pd.DataFrame, weekly: pd.DataFrame, by_head: pd.DataFrame) -> None:
    lines = [
        "# Dynamic HR Surprise Meta/Similarity Ablation Ledger",
        "",
        f"Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "All variants are head-agnostic: parameters are fitted per head, but no variant names or code paths special-case a named head.",
        "",
        "## Summary",
        "",
        _markdown_table(summary.sort_values("total_pnl_net_spread", ascending=False)),
        "",
        "## Weekly",
        "",
        _markdown_table(weekly.sort_values(["variant", "week_start"])),
        "",
        "## By Head",
        "",
        _markdown_table(by_head.sort_values(["variant", "pnl_net_spread"], ascending=[True, False])),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--policy-params", type=Path, default=DEFAULT_POLICY_PARAMS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--only", nargs="*", default=None, help="Optional trial ids or names to run.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--calendar-eval-start", default="2026-05-01")
    parser.add_argument("--calendar-eval-end", default="2026-06-25T23:59:59Z")
    parser.add_argument("--calendar-xw-min-train-days", type=float, default=14.0)
    parser.add_argument("--calendar-xw-max-train-days", type=float, default=183.0)
    parser.add_argument("--calendar-y-train-days", type=float, default=20.0)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    selected = set(args.only or [])
    specs = [spec for spec in ABLATIONS if not selected or spec.trial_id in selected or spec.name in selected]
    if not specs:
        raise SystemExit("No ablations selected")

    all_metrics: list[dict[str, Any]] = []
    weekly_frames: list[pd.DataFrame] = []
    by_head_frames: list[pd.DataFrame] = []
    manifest = {
        "candidates": str(args.candidates),
        "policy_params": str(args.policy_params),
        "output_root": str(args.output_root),
        "calendar_eval_start": args.calendar_eval_start,
        "calendar_eval_end": args.calendar_eval_end,
        "calendar_xw_min_train_days": args.calendar_xw_min_train_days,
        "calendar_xw_max_train_days": args.calendar_xw_max_train_days,
        "calendar_y_train_days": args.calendar_y_train_days,
        "variants": [spec.__dict__ for spec in specs],
    }
    (args.output_root / "ablation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for spec in specs:
        output_dir = args.output_root / f"{spec.trial_id}_{spec.name}"
        done = output_dir / "calendar_dynamic_hr_surprise_selected_rows.parquet"
        if not (args.skip_existing and done.exists()):
            cmd = _common_compare_args(
                args.candidates,
                args.policy_params,
                output_dir,
                calendar_eval_start=args.calendar_eval_start,
                calendar_eval_end=args.calendar_eval_end,
                calendar_xw_min_train_days=args.calendar_xw_min_train_days,
                calendar_xw_max_train_days=args.calendar_xw_max_train_days,
                calendar_y_train_days=args.calendar_y_train_days,
            ) + list(spec.overrides)
            print(f"\n=== Running {spec.trial_id} {spec.name} ===", flush=True)
            print(" ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)
        metrics, weekly, by_head = _metrics(
            output_dir,
            spec,
            eval_start=args.calendar_eval_start,
            eval_end=args.calendar_eval_end,
        )
        all_metrics.append(metrics)
        weekly_frames.append(weekly)
        by_head_frames.append(by_head)

    summary = pd.DataFrame(all_metrics).sort_values("total_pnl_net_spread", ascending=False)
    weekly = pd.concat(weekly_frames, ignore_index=True) if weekly_frames else pd.DataFrame()
    by_head = pd.concat(by_head_frames, ignore_index=True) if by_head_frames else pd.DataFrame()
    summary.to_csv(args.output_root / "ablation_summary.csv", index=False)
    weekly.to_csv(args.output_root / "ablation_weekly.csv", index=False)
    by_head.to_csv(args.output_root / "ablation_by_head.csv", index=False)
    _write_ledger(args.output_root / "ablation_ledger.md", summary, weekly, by_head)
    print("\nSummary:")
    print(summary.to_string(index=False))
    print(f"\nWrote {args.output_root}")


if __name__ == "__main__":
    main()
