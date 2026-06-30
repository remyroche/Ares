#!/usr/bin/env python3
"""Run pre-head symbol-guard threshold sweeps with recomputed calendar replay.

Each trial materializes a guarded candidate ledger first, then feeds that ledger
through the dynamic HR-surprise calendar replay. This keeps daily Y refits and
monthly X/W updates honest after rows are removed or rank-penalized.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_CANDIDATES = (
    "data_perp/artifacts/finalfit_candidate_mask_native_candidates_20260627_6mo/"
    "simple_policy_optimiser/simple_policy_candidates_broad.parquet"
)
DEFAULT_POLICY_PARAMS = "data_perp/artifacts/20260629_050000_lgbm_mda/policy_params/best_policy_params.json"


@dataclass(frozen=True)
class SweepTrial:
    trial_id: str
    variant: str
    args: tuple[str, ...]
    description: str


TRIALS: tuple[SweepTrial, ...] = (
    SweepTrial(
        "A1_default_l3of4_24h",
        "A1_loss_cooldown_3of4_24h",
        (),
        "Default A1: hard block after 3 losses in the last 4 eligible opportunities within 24h.",
    ),
    SweepTrial(
        "A1_l4of4_24h",
        "A1_loss_cooldown_3of4_24h",
        ("--loss-threshold", "4"),
        "Stricter A1: hard block only after 4 losses in the last 4 opportunities.",
    ),
    SweepTrial(
        "A1_l3of4_12h",
        "A1_loss_cooldown_3of4_24h",
        ("--cooldown-hours", "12"),
        "More reactive A1 recovery: same 3/4 trigger but only a 12h cooldown.",
    ),
    SweepTrial(
        "A1_l4of5_24h",
        "A1_loss_cooldown_3of4_24h",
        ("--loss-window", "5", "--loss-threshold", "4"),
        "Less brittle A1: 4 losses in the last 5 opportunities within 24h.",
    ),
    SweepTrial(
        "A3_default_z150_m5",
        "A3_symbol_z_guard_7d_m5_zneg15",
        (),
        "Default A3: 7d standardized HR surprise z <= -1.50 with at least 5 opportunities.",
    ),
    SweepTrial(
        "A3_z175_m5",
        "A3_symbol_z_guard_7d_m5_zneg15",
        ("--z-threshold", "-1.75"),
        "Less aggressive A3: require z <= -1.75.",
    ),
    SweepTrial(
        "A3_z200_m5",
        "A3_symbol_z_guard_7d_m5_zneg15",
        ("--z-threshold", "-2.00"),
        "Conservative A3: require z <= -2.00.",
    ),
    SweepTrial(
        "A3_z175_m8",
        "A3_symbol_z_guard_7d_m5_zneg15",
        ("--z-threshold", "-1.75", "--z-min-count", "8"),
        "Conservative A3 with more evidence: z <= -1.75 and at least 8 opportunities.",
    ),
    SweepTrial(
        "A4_default_p05_p10",
        "A4_soft_raise_loss2_zneg125",
        (),
        "Default A4: soft rank penalty 0.05/severe 0.10 on weak symbol state.",
    ),
    SweepTrial(
        "A4_light_p03_p07",
        "A4_soft_raise_loss2_zneg125",
        ("--soft-penalty", "0.03", "--severe-penalty", "0.07"),
        "Lighter A4 penalties with the same weakness thresholds.",
    ),
    SweepTrial(
        "A4_loose_l3_z175_p03_p07",
        "A4_soft_raise_loss2_zneg125",
        (
            "--soft-loss-threshold",
            "3",
            "--soft-z-threshold",
            "-1.75",
            "--severe-z-threshold",
            "-2.50",
            "--soft-penalty",
            "0.03",
            "--severe-penalty",
            "0.07",
        ),
        "A4 middle ground: require stronger weakness before a lighter penalty.",
    ),
    SweepTrial(
        "A4_moderate_z150_p03_p07",
        "A4_soft_raise_loss2_zneg125",
        (
            "--soft-z-threshold",
            "-1.50",
            "--severe-z-threshold",
            "-2.25",
            "--soft-penalty",
            "0.03",
            "--severe-penalty",
            "0.07",
        ),
        "A4 moderate z thresholds with lighter penalties.",
    ),
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _calendar_replay_args(policy_params: str) -> list[str]:
    return [
        "scripts/compare_dynamic_hr_surprise_threshold.py",
        "--policy-params",
        policy_params,
        "--calendar-only",
        "--calendar-eval-start",
        "2026-05-01",
        "--calendar-eval-end",
        "2026-06-25T23:59:59Z",
        "--calendar-xw-min-train-days",
        "90.0",
        "--calendar-xw-max-train-days",
        "183.0",
        "--calendar-y-train-days",
        "20.0",
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
        "--y-min=-0.50",
        "--y-max",
        "1.50",
        "--subwindow-constraints-mode",
        "penalty",
        "--subwindow-days",
        "5.0",
        "--min-subwindows",
        "4",
        "--min-positive-objective-fraction",
        "0.25",
        "--subwindow-q15-floor=-1.0",
        "--subwindow-drawdown-floor=-3.0",
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
        "--per-head-min-objective=-1e18",
        "--per-head-min-q05-week-pnl=-1e18",
        "--per-head-min-q15-week-pnl=-1e18",
        "--per-head-min-robust-objective=-1e18",
        "--deployed-threshold-soft-prior-strength",
        "8.0",
        "--deployed-threshold-soft-prior-deadband",
        "0.03",
        "--deployed-threshold-soft-prior-power",
        "2.0",
        "--deployed-threshold-soft-prior-activity-weight",
        "0.25",
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
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument("--policy-params", default=DEFAULT_POLICY_PARAMS)
    parser.add_argument("--output-dir", default="data_perp/reports/prehead_symbol_guard_threshold_sweep_20260630")
    parser.add_argument("--trial-filter", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    filters = {item.strip() for item in args.trial_filter.split(",") if item.strip()}
    selected = [trial for trial in TRIALS if not filters or trial.trial_id in filters]
    if not selected:
        raise SystemExit(f"No matching trials for filter={sorted(filters)}")

    manifest = {
        "candidates": args.candidates,
        "policy_params": args.policy_params,
        "output_dir": str(out_root),
        "trials": [asdict(trial) for trial in selected],
    }
    (out_root / "threshold_sweep_manifest.json").write_text(
        json.dumps(manifest, default=_json_default, indent=2),
        encoding="utf-8",
    )

    for trial in selected:
        trial_dir = out_root / trial.trial_id
        materialized_dir = trial_dir / "materialized"
        replay_dir = trial_dir / "T16_recomputed_calendar_replay"
        replay_summary = replay_dir / "calendar_dynamic_hr_surprise_policy_summary.parquet"
        if replay_summary.exists() and not args.force:
            print(f"SKIP {trial.trial_id}: {replay_summary} exists", flush=True)
            continue

        materialize_cmd = [
            sys.executable,
            "-u",
            "scripts/materialize_prehead_symbol_guard_ablation_candidates.py",
            "--candidates",
            args.candidates,
            "--output-dir",
            str(materialized_dir),
            "--variants",
            trial.variant,
            "--top-rank-floor",
            "0.70",
            "--engine",
            "fast",
            "--max-blacklisted-asset-fraction",
            "0.10",
            "--require-relative-symbol-weakness",
            "--relative-peer-min-symbols",
            "20",
            "--relative-z-peer-quantile",
            "0.25",
            "--relative-z-margin",
            "0.50",
            "--relative-loss-peer-quantile",
            "0.75",
            "--relative-loss-margin",
            "1.0",
            *trial.args,
        ]
        _run(materialize_cmd, dry_run=bool(args.dry_run))
        guarded_candidates = materialized_dir / trial.variant / "simple_policy_candidates_broad.parquet"
        replay_cmd = [
            sys.executable,
            "-u",
            *_calendar_replay_args(str(args.policy_params)),
            "--candidates",
            str(guarded_candidates),
            "--output-dir",
            str(replay_dir),
        ]
        _run(replay_cmd, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()
