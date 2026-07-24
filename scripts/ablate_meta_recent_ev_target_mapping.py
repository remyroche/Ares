#!/usr/bin/env python3
"""Causally compare recent EV-target rank mappings around the MLP overlay.

The mapping mirrors the reachable-EV policy idea: the historical global top
10% EV is the target; each side x archetype finds the recent score threshold
whose tail best reaches that target.  The local/global threshold ratio is a
bounded, support-shrunk multiplier applied to the MLP rank.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TOP_FRACTION = 0.10
GLOBAL_MIN_ROWS = 200
LOCAL_MIN_ROWS = 40
LOCAL_SUPPORT_TARGET = 160.0
MULTIPLIER_MIN = 0.50
MULTIPLIER_MAX = 1.50


@dataclass(frozen=True)
class Arm:
    name: str
    timing: str
    window_days: int
    smoothing: str
    half_life_days: int | None


def _arms() -> list[Arm]:
    result: list[Arm] = []
    for timing in ("before_mlp", "after_mlp"):
        for window in (8, 12, 16, 20, 24, 28):
            result.append(
                Arm(
                    name=f"{timing}_evtarget_{window}d_flat",
                    timing=timing,
                    window_days=window,
                    smoothing="flat",
                    half_life_days=None,
                )
            )
            for half_life in (5, 10, 20):
                result.append(
                    Arm(
                        name=(
                            f"{timing}_evtarget_{window}d_"
                            f"ewma_hl{half_life}d"
                        ),
                        timing=timing,
                        window_days=window,
                        smoothing="ewma",
                        half_life_days=half_life,
                    )
                )
    return result


def _finite(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)[np.isfinite(values)]


def _global_top_fraction_ev(
    score: np.ndarray, ev: np.ndarray, fraction: float = TOP_FRACTION
) -> tuple[float, float]:
    valid = np.isfinite(score) & np.isfinite(ev)
    if int(valid.sum()) < GLOBAL_MIN_ROWS:
        return np.nan, np.nan
    score, ev = score[valid], ev[valid]
    threshold = float(np.quantile(score, 1.0 - fraction))
    selected = score >= threshold
    return float(np.mean(ev[selected])), threshold


def _effective_n(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    total = float(np.sum(weights))
    square_total = float(np.sum(np.square(weights)))
    if total <= 0.0 or square_total <= 0.0:
        return 0.0
    return total * total / square_total


def _recent_weights(
    timestamps: np.ndarray,
    day: pd.Timestamp,
    arm: Arm,
) -> np.ndarray:
    if arm.smoothing == "flat":
        return np.ones(len(timestamps), dtype=np.float64)
    timestamp_ns = pd.to_datetime(timestamps, utc=True).asi8
    age_days = (float(day.value) - timestamp_ns.astype(np.float64)) / 86_400_000_000_000.0
    return np.power(
        0.5,
        np.maximum(age_days, 0.0) / max(float(arm.half_life_days or 1), 1e-6),
    )


def _threshold_for_target_ev(
    score: np.ndarray,
    ev: np.ndarray,
    weights: np.ndarray,
    *,
    target_ev: float,
    min_rows: int,
) -> float:
    valid = np.isfinite(score) & np.isfinite(ev) & np.isfinite(weights) & (weights > 0)
    if int(valid.sum()) < int(min_rows) or not np.isfinite(target_ev):
        return np.nan
    score, ev, weights = score[valid], ev[valid], weights[valid]
    order = np.argsort(score, kind="stable")
    score, ev, weights = score[order], ev[order], weights[order]
    weighted_ev_suffix = np.cumsum((weights * ev)[::-1])[::-1]
    weight_suffix = np.cumsum(weights[::-1])[::-1]
    row_suffix = np.arange(len(score), 0, -1)
    candidates = np.unique(
        np.searchsorted(
            score,
            np.quantile(score, np.linspace(0.70, 0.99, 60)),
            side="left",
        )
    )
    candidates = candidates[candidates < len(score)]
    if not len(candidates):
        return np.nan
    means = weighted_ev_suffix[candidates] / np.maximum(weight_suffix[candidates], 1e-12)
    allowed = row_suffix[candidates] >= int(min_rows)
    meets_target = allowed & (means >= float(target_ev))
    if np.any(meets_target):
        candidate_positions = np.flatnonzero(meets_target)
        best = candidate_positions[np.argmin(np.abs(means[meets_target] - target_ev))]
        return float(score[candidates[best]])
    # No recent tail attains the historical target: use the strictest supported
    # tail, which naturally demotes this side x archetype for the day.
    supported = np.flatnonzero(allowed)
    return float(score[candidates[supported[-1]]]) if len(supported) else np.nan


def _safe_multiplier(global_threshold: float, local_threshold: float, support: float) -> float:
    if not np.isfinite(global_threshold) or not np.isfinite(local_threshold):
        return 1.0
    raw = np.clip(
        global_threshold / max(local_threshold, 1e-8),
        MULTIPLIER_MIN,
        MULTIPLIER_MAX,
    )
    confidence = np.clip(float(support) / LOCAL_SUPPORT_TARGET, 0.0, 1.0)
    return float(1.0 + confidence * (raw - 1.0))


def _week_start(timestamps: pd.Series) -> pd.Series:
    days = pd.to_datetime(timestamps, utc=True).dt.floor("D")
    return days - pd.to_timedelta(days.dt.weekday, unit="D")


def _metrics(selected: pd.DataFrame, source: pd.DataFrame) -> dict[str, float]:
    if selected.empty:
        return {
            "selected_rows": 0,
            "trades_per_day": 0.0,
            "mean_net_ev_after_1pct": np.nan,
            "sum_net_ev_after_1pct": 0.0,
            "cumulative_net_ev": 0.0,
            "positive_ev_rate": np.nan,
            "q01_week_net_ev": np.nan,
            "q10_week_net_ev": np.nan,
            "q25_week_net_ev": np.nan,
            "worst_week_net_ev": np.nan,
        }
    ev = pd.to_numeric(selected["ev_after_1pct"], errors="coerce").to_numpy(dtype=np.float64)
    week_values = (
        selected.assign(__week__=_week_start(selected["__ts__"]))
        .groupby("__week__", observed=True)["ev_after_1pct"]
        .mean()
        .to_numpy(dtype=np.float64)
    )
    return {
        "selected_rows": int(len(selected)),
        "trades_per_day": float(len(selected) / max(source["__day__"].nunique(), 1)),
        "mean_net_ev_after_1pct": float(np.nanmean(ev)),
        "sum_net_ev_after_1pct": float(np.nansum(ev)),
        # Candidate rows overlap in time; compounding trade-level returns would
        # falsely assume sequential full-notional execution. Additive EV is the
        # comparable pre-portfolio cumulative measure.
        "cumulative_net_ev": float(np.nansum(ev)),
        "positive_ev_rate": float(np.nanmean(ev > 0.0)),
        "q01_week_net_ev": float(np.nanquantile(week_values, 0.01)),
        "q10_week_net_ev": float(np.nanquantile(week_values, 0.10)),
        "q25_week_net_ev": float(np.nanquantile(week_values, 0.25)),
        "worst_week_net_ev": float(np.nanmin(week_values)),
    }


def _apply_arm(rows: pd.DataFrame, arm: Arm, eval_start: pd.Timestamp) -> pd.DataFrame:
    calibration_col = "policy_parent_rank" if arm.timing == "before_mlp" else "rank_mlp_direct"
    apply_col = "rank_mlp_direct"
    records: list[pd.DataFrame] = []
    days = sorted(rows.loc[rows["__day__"].ge(eval_start), "__day__"].unique())
    for day_value in days:
        day = pd.Timestamp(day_value)
        day = day.tz_localize("UTC") if day.tzinfo is None else day.tz_convert("UTC")
        current = rows.loc[rows["__day__"].eq(day)].copy()
        prior = rows.loc[rows["__day__"].lt(day)]
        if current.empty or len(prior) < GLOBAL_MIN_ROWS:
            continue
        recent = prior.loc[prior["__day__"].ge(day - pd.Timedelta(days=arm.window_days))]
        reference = recent if len(recent) >= GLOBAL_MIN_ROWS else prior
        ref_score = pd.to_numeric(reference[calibration_col], errors="coerce").to_numpy(dtype=np.float64)
        ref_ev = pd.to_numeric(reference["ev_after_1pct"], errors="coerce").to_numpy(dtype=np.float64)
        target_ev, global_threshold = _global_top_fraction_ev(
            pd.to_numeric(prior[calibration_col], errors="coerce").to_numpy(dtype=np.float64),
            pd.to_numeric(prior["ev_after_1pct"], errors="coerce").to_numpy(dtype=np.float64),
        )
        apply_scores = pd.to_numeric(prior[apply_col], errors="coerce").to_numpy(dtype=np.float64)
        apply_scores = apply_scores[np.isfinite(apply_scores)]
        if not np.isfinite(global_threshold) or len(apply_scores) < GLOBAL_MIN_ROWS:
            continue
        apply_global_cutoff = float(np.quantile(apply_scores, 1.0 - TOP_FRACTION))
        weights = _recent_weights(
            pd.to_datetime(reference["__ts__"], utc=True).to_numpy(), day, arm
        )
        global_recent_threshold = _threshold_for_target_ev(
            ref_score, ref_ev, weights, target_ev=target_ev, min_rows=GLOBAL_MIN_ROWS
        )
        if not np.isfinite(global_recent_threshold):
            global_recent_threshold = global_threshold
        multipliers = np.ones(len(current), dtype=np.float64)
        local_thresholds = np.full(len(current), global_recent_threshold, dtype=np.float64)
        local_support = np.zeros(len(current), dtype=np.float64)
        fallback = np.ones(len(current), dtype=bool)
        group_key = current["side_name"].astype(str) + "||" + current["archetype_policy_key"].astype(str)
        ref_key = reference["side_name"].astype(str) + "||" + reference["archetype_policy_key"].astype(str)
        for key in group_key.unique():
            idx = np.flatnonzero(group_key.to_numpy() == key)
            local = reference.loc[ref_key.eq(key)]
            local_score = pd.to_numeric(local[calibration_col], errors="coerce").to_numpy(dtype=np.float64)
            local_ev = pd.to_numeric(local["ev_after_1pct"], errors="coerce").to_numpy(dtype=np.float64)
            local_weights = _recent_weights(
                pd.to_datetime(local["__ts__"], utc=True).to_numpy(), day, arm
            )
            finite = np.isfinite(local_score) & np.isfinite(local_ev)
            support = _effective_n(local_weights[finite])
            local_support[idx] = support
            if int(finite.sum()) < LOCAL_MIN_ROWS:
                continue
            threshold = _threshold_for_target_ev(
                local_score, local_ev, local_weights, target_ev=target_ev,
                min_rows=LOCAL_MIN_ROWS,
            )
            if not np.isfinite(threshold):
                continue
            local_thresholds[idx] = threshold
            multipliers[idx] = _safe_multiplier(
                global_recent_threshold, threshold, support
            )
            fallback[idx] = False
        current["ev_target_multiplier"] = multipliers.astype(np.float32)
        current["ev_target_local_threshold"] = local_thresholds.astype(np.float32)
        current["ev_target_global_threshold"] = float(global_recent_threshold)
        current["ev_target_global_ev"] = float(target_ev)
        current["ev_target_local_support"] = local_support.astype(np.float32)
        current["ev_target_global_fallback"] = fallback
        current["ev_target_apply_cutoff"] = apply_global_cutoff
        current["mapped_rank"] = np.clip(
            pd.to_numeric(current[apply_col], errors="coerce").to_numpy(dtype=np.float64)
            * multipliers,
            0.0,
            1.0,
        )
        current["selected"] = current["mapped_rank"] >= apply_global_cutoff
        records.append(current)
    return pd.concat(records, ignore_index=True, copy=False) if records else rows.iloc[0:0].copy()


def _report_arm(scored: pd.DataFrame, arm: Arm) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    scored["month"] = scored["__ts__"].dt.strftime("%Y-%m")
    for month, source in scored.groupby("month", observed=True, sort=True):
        selected = source.loc[source["selected"]].copy()
        record = {**asdict(arm), "month": str(month), **_metrics(selected, source)}
        record.update(
            {
                "mean_multiplier": float(source["ev_target_multiplier"].mean()),
                "global_fallback_rate": float(source["ev_target_global_fallback"].mean()),
                "mean_global_target_ev": float(source["ev_target_global_ev"].mean()),
            }
        )
        records.append(record)
    selected = scored.loc[scored["selected"]].copy()
    records.append(
        {
            **asdict(arm),
            "month": "overall",
            **_metrics(selected, scored),
            "mean_multiplier": float(scored["ev_target_multiplier"].mean()),
            "global_fallback_rate": float(scored["ev_target_global_fallback"].mean()),
            "mean_global_target_ev": float(scored["ev_target_global_ev"].mean()),
        }
    )
    return pd.DataFrame(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank-history", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-start", default="2026-04-01")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    needed = [
        "__ts__", "side_name", "archetype_policy_key", "policy_parent_rank",
        "rank_mlp_direct", "ev_after_1pct", "__fold__",
    ]
    rows = pd.read_parquet(args.rank_history, columns=needed)
    rows["__ts__"] = pd.to_datetime(rows["__ts__"], utc=True)
    rows = rows.dropna(subset=["__ts__", "policy_parent_rank", "rank_mlp_direct", "ev_after_1pct"])
    rows = rows.sort_values("__ts__", kind="stable").reset_index(drop=True)
    rows["__day__"] = rows["__ts__"].dt.floor("D")
    eval_start = pd.Timestamp(args.eval_start, tz="UTC")
    reports: list[pd.DataFrame] = []
    diagnostics: list[pd.DataFrame] = []
    for number, arm in enumerate(_arms(), start=1):
        scored = _apply_arm(rows, arm, eval_start)
        reports.append(_report_arm(scored, arm))
        diagnostics.append(
            scored.loc[
                :, [
                    "__ts__", "side_name", "archetype_policy_key", "__fold__",
                    "policy_parent_rank", "rank_mlp_direct", "mapped_rank", "selected",
                    "ev_target_multiplier", "ev_target_local_threshold",
                    "ev_target_global_threshold", "ev_target_global_ev",
                    "ev_target_apply_cutoff",
                    "ev_target_local_support", "ev_target_global_fallback",
                ]
            ].assign(variant=arm.name)
        )
        print(f"Completed {number}/{len(_arms())}: {arm.name}", flush=True)
    report = pd.concat(reports, ignore_index=True)
    report.to_csv(args.output_dir / "monthly_metrics.csv", index=False)
    pd.concat(diagnostics, ignore_index=True).to_parquet(
        args.output_dir / "row_diagnostics.parquet", index=False, compression="zstd"
    )
    manifest = {
        "schema": "meta_recent_ev_target_mapping_ablation_v1",
        "rank_history": str(args.rank_history),
        "eval_start": eval_start.isoformat(),
        "eval_end": rows["__ts__"].max().isoformat(),
        "cost_contract": "ev_after_1pct is the sole realized metric; no additional fee is subtracted.",
        "causal_contract": (
            "Each day uses only prior OOS rows. Global top-10 EV uses all prior rows; "
            "local thresholds use only the selected recent window, with a global fallback."
        ),
        "mapping_contract": (
            "The local/global score-threshold ratio is a bounded, support-shrunk "
            "multiplier applied to rank_mlp_direct. before_mlp derives that multiplier "
            "from policy_parent_rank; after_mlp derives it from rank_mlp_direct."
        ),
        "arms": [asdict(arm) for arm in _arms()],
        "constants": {
            "top_fraction": TOP_FRACTION,
            "global_min_rows": GLOBAL_MIN_ROWS,
            "local_min_rows": LOCAL_MIN_ROWS,
            "local_support_target": LOCAL_SUPPORT_TARGET,
            "multiplier_min": MULTIPLIER_MIN,
            "multiplier_max": MULTIPLIER_MAX,
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(report.loc[report["month"].eq("overall")].sort_values("mean_net_ev_after_1pct", ascending=False).head(12).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
