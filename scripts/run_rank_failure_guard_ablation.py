#!/usr/bin/env python3
"""Run A/B tests for a generic per-head recent rank-failure guard.

The tested policy chain is:

    A1_l4of5_24h symbol guard -> optional per-head rank-failure guard -> T16 calendar replay

This keeps the guard symmetric across heads and recomputes the daily dynamic
threshold grid after candidate rows are removed or rank-penalized.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_prehead_symbol_guard_threshold_sweep import (  # noqa: E402
    DEFAULT_CANDIDATES,
    DEFAULT_POLICY_PARAMS,
    _calendar_replay_args,
)


@dataclass(frozen=True)
class RankFailureTrial:
    trial_id: str
    rank_variant: str | None
    rank_args: tuple[str, ...]
    description: str


TRIALS: tuple[RankFailureTrial, ...] = (
    RankFailureTrial(
        "A1_l4of5_24h_only",
        None,
        (),
        "Current default pre-head symbol guard before T16 replay.",
    ),
    RankFailureTrial(
        "A1_l4of5_24h__rank_soft_7d",
        "A8_rank_failure_soft_7d",
        (),
        "Soft per-head rank penalty when recent top decile underperforms the 70-90 rank band.",
    ),
    RankFailureTrial(
        "A1_l4of5_24h__rank_hard_7d",
        "A8_rank_failure_hard_7d",
        (),
        "Hard per-head day pause when recent top decile underperforms the 70-90 rank band.",
    ),
    RankFailureTrial(
        "A1_l4of5_24h__rank_soft_3d",
        "A8_rank_failure_soft_3d",
        (),
        "More reactive 3-day soft per-head rank penalty.",
    ),
    RankFailureTrial(
        "A1_l4of5_24h__rank_soft_7d_hr15",
        "A8_rank_failure_soft_7d",
        ("--rank-failure-hr-margin", "0.15"),
        "Less sensitive 7-day soft rank penalty requiring a 15pp HR underperformance.",
    ),
    RankFailureTrial(
        "A1_l4of5_24h__rank_soft_7d_both_edges",
        "A8_rank_failure_soft_7d",
        ("--rank-failure-require-both-edges",),
        "Narrow 7-day soft rank penalty requiring both PnL-edge and HR-edge inversion.",
    ),
    RankFailureTrial(
        "A1_l4of5_24h__rank_hard_7d_both_edges",
        "A8_rank_failure_hard_7d",
        ("--rank-failure-require-both-edges",),
        "Narrow 7-day hard per-head pause requiring both PnL-edge and HR-edge inversion.",
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


def _materialize_a1_cmd(candidates: str, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-u",
        "scripts/materialize_prehead_symbol_guard_ablation_candidates.py",
        "--candidates",
        candidates,
        "--output-dir",
        str(output_dir),
        "--variants",
        "A1_loss_cooldown_3of4_24h",
        "--top-rank-floor",
        "0.70",
        "--engine",
        "fast",
        "--loss-window",
        "5",
        "--loss-threshold",
        "4",
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
        "--parquet-only",
    ]


def _materialize_rank_cmd(candidates: Path, output_dir: Path, trial: RankFailureTrial) -> list[str]:
    if trial.rank_variant is None:
        raise ValueError("rank_variant is required")
    return [
        sys.executable,
        "-u",
        "scripts/materialize_prehead_symbol_guard_ablation_candidates.py",
        "--candidates",
        str(candidates),
        "--output-dir",
        str(output_dir),
        "--variants",
        trial.rank_variant,
        "--top-rank-floor",
        "0.70",
        "--engine",
        "fast",
        "--rank-failure-top-floor",
        "0.90",
        "--rank-failure-lower-floor",
        "0.70",
        "--rank-failure-lower-ceiling",
        "0.90",
        "--rank-failure-min-top-count",
        "20",
        "--rank-failure-min-lower-count",
        "40",
        "--rank-failure-mean-margin",
        "0.0",
        "--rank-failure-hr-margin",
        "0.10",
        "--rank-failure-soft-penalty",
        "0.07",
        "--rank-failure-severe-penalty",
        "0.12",
        "--parquet-only",
        *trial.rank_args,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument("--policy-params", default=DEFAULT_POLICY_PARAMS)
    parser.add_argument(
        "--output-dir",
        default="data_perp/reports/rank_failure_guard_ablation_20260630",
    )
    parser.add_argument("--trial-filter", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    filters = {item.strip() for item in str(args.trial_filter).split(",") if item.strip()}
    selected = [trial for trial in TRIALS if not filters or trial.trial_id in filters]
    if not selected:
        raise SystemExit(f"No matching trials for filter={sorted(filters)}")

    manifest = {
        "candidates": args.candidates,
        "policy_params": args.policy_params,
        "output_dir": str(out_root),
        "trials": [asdict(trial) for trial in selected],
    }
    (out_root / "rank_failure_guard_ablation_manifest.json").write_text(
        json.dumps(manifest, default=_json_default, indent=2),
        encoding="utf-8",
    )

    shared_a1_dir = out_root / "_shared_a1_materialized"
    shared_a1_candidate = shared_a1_dir / "A1_loss_cooldown_3of4_24h" / "simple_policy_candidates_broad.parquet"
    if bool(args.force) or not shared_a1_candidate.exists():
        _run(_materialize_a1_cmd(str(args.candidates), shared_a1_dir), dry_run=bool(args.dry_run))

    for trial in selected:
        trial_dir = out_root / trial.trial_id
        rank_dir = trial_dir / "rank_failure_materialized"
        replay_dir = trial_dir / "T16_recomputed_calendar_replay"
        replay_summary = replay_dir / "calendar_dynamic_hr_surprise_policy_summary.parquet"
        if replay_summary.exists() and not bool(args.force):
            print(f"SKIP {trial.trial_id}: {replay_summary} exists", flush=True)
            continue

        final_candidate = shared_a1_candidate

        if trial.rank_variant is not None:
            rank_candidate = rank_dir / trial.rank_variant / "simple_policy_candidates_broad.parquet"
            if bool(args.force) or not rank_candidate.exists():
                _run(_materialize_rank_cmd(shared_a1_candidate, rank_dir, trial), dry_run=bool(args.dry_run))
            final_candidate = rank_candidate

        replay_cmd = [
            sys.executable,
            "-u",
            *_calendar_replay_args(str(args.policy_params)),
            "--candidates",
            str(final_candidate),
            "--output-dir",
            str(replay_dir),
        ]
        _run(replay_cmd, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()
