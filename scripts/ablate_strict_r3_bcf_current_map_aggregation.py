#!/usr/bin/env python3
"""Causal post-mapper aggregation ablation for frozen BCF/current-v5 MC1.

This does not fit or retune a score family, mapper, policy, or portfolio.  It
only compares ways to combine their already strictly prequential expected-EV
outputs at a common +30-bps admission boundary.  Invalid policy rows are
excluded *after* the frozen score population is formed and never reserve
portfolio capacity.

The dynamic-reliability variants use only earlier resolved policy outcomes:
per calendar day, inverse 21-day MAE mapper weights are calculated from rows
whose policy label was already available before that day.  They are clipped
to [25%, 75%] and are therefore a conservative, deployable calibration
reliability proxy rather than a new ranker.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ablate_strict_r3_bcf_current_v5_agreement_blend import (  # noqa: E402
    CAUSAL_AUCTION_CURVE,
    _common_panel,
    _daily_metrics,
    _read,
    _replay,
    _summary,
)

DEFAULT_BCF = ROOT / (
    "data_perp/artifacts/strict_r3_score_family_matched_mc1_canonical_policy_"
    "20260817_v7_bcf/predictions_bcf_mc1_d2.parquet"
)
DEFAULT_CURRENT = ROOT / (
    "data_perp/artifacts/strict_r3_score_family_matched_mc1_canonical_policy_"
    "20260817_v7_current/predictions_current_v5_mc1_d2.parquet"
)
DEFAULT_OUT = ROOT / "data_perp/artifacts/strict_r3_bcf_current_map_aggregation_20260817_v1"
THRESHOLD = 30.0


def _assert_new(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"immutable output already exists: {path}")
    path.mkdir(parents=True, exist_ok=False)


def _valid_label(panel: pd.DataFrame) -> pd.Series:
    return (
        panel["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(panel["policy_net_bps"], errors="coerce"))
        & np.isfinite(pd.to_numeric(panel["bcf_mc1_expected_bps"], errors="coerce"))
        & np.isfinite(pd.to_numeric(panel["current_mc1_expected_bps"], errors="coerce"))
    )


def _prior_reliability_weight(panel: pd.DataFrame, *, days: int = 21, min_rows: int = 500) -> pd.DataFrame:
    """Return causal daily BCF weights from prior fully resolved MAE only."""
    work = panel.copy()
    work["day"] = work["__decision_ts__"].dt.normalize()
    work["policy_label_available_ts"] = pd.to_datetime(work["policy_label_available_ts"], utc=True, errors="raise")
    valid = _valid_label(work)
    day_values = sorted(pd.Timestamp(value) for value in work["day"].unique())
    rows: list[dict[str, float | int | pd.Timestamp]] = []
    y = pd.to_numeric(work["policy_net_bps"], errors="coerce")
    b = pd.to_numeric(work["bcf_mc1_expected_bps"], errors="coerce")
    c = pd.to_numeric(work["current_mc1_expected_bps"], errors="coerce")
    for day in day_values:
        start = day - pd.Timedelta(days=days)
        prior = valid & work["__decision_ts__"].ge(start) & work["policy_label_available_ts"].lt(day)
        support = int(prior.sum())
        if support < min_rows:
            weight = 0.5
            bcf_mae = float("nan")
            current_mae = float("nan")
        else:
            bcf_mae = float(np.abs(y.loc[prior] - b.loc[prior]).mean())
            current_mae = float(np.abs(y.loc[prior] - c.loc[prior]).mean())
            inv_b = 1.0 / max(bcf_mae, 1e-9)
            inv_c = 1.0 / max(current_mae, 1e-9)
            weight = float(np.clip(inv_b / (inv_b + inv_c), 0.25, 0.75))
        rows.append({
            "day": day, "reliability_bcf_weight": weight, "reliability_support": support,
            "prior_bcf_mae": bcf_mae, "prior_current_mae": current_mae,
        })
    return pd.DataFrame(rows)


def _variants(panel: pd.DataFrame) -> dict[str, tuple[pd.Series, pd.Series]]:
    b = pd.to_numeric(panel["bcf_mc1_expected_bps"], errors="raise")
    c = pd.to_numeric(panel["current_mc1_expected_bps"], errors="raise")
    mean = 0.5 * (b + c)
    low = np.minimum(b, c)
    high = np.maximum(b, c)
    # Penalise only disagreement at the floor.  When both maps agree on the
    # side of the boundary, this remains the ordinary equal-weight mean.
    straddles_floor = (low < THRESHOLD) & (high >= THRESHOLD)
    low75 = mean.where(~straddles_floor, 0.75 * low + 0.25 * high)
    low90 = mean.where(~straddles_floor, 0.90 * low + 0.10 * high)
    rel = pd.to_numeric(panel["reliability_bcf_weight"], errors="raise")
    rel_score = rel * b + (1.0 - rel) * c
    rel_low75 = rel_score.where(~straddles_floor, 0.75 * low + 0.25 * high)
    return {
        "dual_and_t30_bcf_priority": (b.ge(THRESHOLD) & c.ge(THRESHOLD), b),
        "mean50_t30": (mean.ge(THRESHOLD), mean),
        "bcf75_t30": ((0.75 * b + 0.25 * c).ge(THRESHOLD), 0.75 * b + 0.25 * c),
        "current75_t30": ((0.25 * b + 0.75 * c).ge(THRESHOLD), 0.25 * b + 0.75 * c),
        "floor_low75_t30": (low75.ge(THRESHOLD), low75),
        "floor_low90_t30": (low90.ge(THRESHOLD), low90),
        "reliability_invmae_t30": (rel_score.ge(THRESHOLD), rel_score),
        "reliability_invmae_floor75_t30": (rel_low75.ge(THRESHOLD), rel_low75),
    }


def run(args: argparse.Namespace) -> Path:
    out = Path(args.out_dir).resolve()
    _assert_new(out)
    bcf, current = _read(Path(args.bcf_predictions)), _read(Path(args.current_predictions))
    panel = _common_panel(bcf, current)
    panel = panel.loc[panel["__decision_ts__"].dt.year.isin(args.years)].copy().reset_index(drop=True)
    reliability = _prior_reliability_weight(panel)
    panel["day"] = panel["__decision_ts__"].dt.normalize()
    panel = panel.merge(reliability, on="day", how="left", validate="many_to_one")
    if panel["reliability_bcf_weight"].isna().any():
        raise AssertionError("causal reliability weight missing for a decision day")

    metrics: list[dict[str, object]] = []
    daily: list[pd.DataFrame] = []
    decisions_root = out / "decisions"
    decisions_root.mkdir()
    start = panel["__decision_ts__"].min()
    end = panel["__decision_ts__"].max() + pd.Timedelta(days=1)
    for arm, (admission, priority) in _variants(panel).items():
        rows, decisions = _replay(
            panel, arm=arm, admission=admission, priority=priority,
            years=tuple(sorted(set(args.years))), out_dir=decisions_root,
        )
        for row in rows:
            row["admission_threshold_bps"] = THRESHOLD
        metrics.extend(rows)
        daily.append(_daily_metrics(decisions, arm=arm, start=start, end=end))
    metrics_frame = pd.DataFrame(metrics)
    daily_frame = pd.concat(daily, ignore_index=True)
    summary = _summary(metrics_frame, daily_frame, baseline_arm="dual_and_t30_bcf_priority")
    valid = _valid_label(panel)
    diagnostics = pd.DataFrame([{
        "rows": int(len(panel)), "valid_policy_rows": int(valid.sum()),
        "start": start, "end_exclusive": end,
        "mean_reliability_bcf_weight": float(panel["reliability_bcf_weight"].mean()),
        "p05_reliability_bcf_weight": float(panel["reliability_bcf_weight"].quantile(.05)),
        "p95_reliability_bcf_weight": float(panel["reliability_bcf_weight"].quantile(.95)),
        "mean_reliability_support": float(panel["reliability_support"].mean()),
    }])
    metrics_frame.to_parquet(out / "portfolio_metrics.parquet", index=False, compression="zstd")
    metrics_frame.to_csv(out / "portfolio_metrics.csv", index=False)
    daily_frame.to_parquet(out / "daily_metrics.parquet", index=False, compression="zstd")
    summary.to_parquet(out / "summary_metrics.parquet", index=False, compression="zstd")
    summary.to_csv(out / "summary_metrics.csv", index=False)
    reliability.to_parquet(out / "causal_reliability_weights.parquet", index=False, compression="zstd")
    diagnostics.to_parquet(out / "diagnostics.parquet", index=False, compression="zstd")
    (out / "run_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_bcf_current_map_aggregation_v1", "status": "complete",
        "purpose": "post-mapper admission/priority overlay; no score/map/policy retraining",
        "threshold_bps": THRESHOLD, "years": list(args.years),
        "common_population": "candidate IDs present in both frozen prequential BCF and current-v5 mapper ledgers",
        "base_route_limitation": "historical common ledger lacks the current timestamp-local raw base top-30 field; results isolate mapper aggregation, not the later live base route",
        "reliability": "inverse MAE over preceding 21 days of policy labels resolved before decision day; 25%-75% clipped; min 500 rows otherwise 50/50",
        "portfolio": "long-only global auction, 7x, 10% slots, 80% margin, two entries/decision, eight concurrent; invalid outcomes excluded before capacity",
    }, indent=2, sort_keys=True) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bcf-predictions", type=Path, default=DEFAULT_BCF)
    parser.add_argument("--current-predictions", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--years", type=int, nargs="+", default=[2025, 2026])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()
