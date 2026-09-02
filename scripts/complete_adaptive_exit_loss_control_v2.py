#!/usr/bin/env python3
"""Complete portfolio/frontier/manifest outputs after a finished V2 model pass."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_adaptive_exit_loss_control_v2 import (
    ARM_ORDER,
    DEFAULT_A5,
    DEFAULT_ATR,
    DEFAULT_HOURLY,
    DEFAULT_POLICY,
    DEFAULT_V1,
    STOP_LEVELS,
    _load_frozen_policy,
    _portfolio,
    _portfolio_metrics,
    _safe,
    _sha,
    _v1_population,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--v1-dir", type=Path, default=DEFAULT_V1)
    parser.add_argument("--hourly-dir", type=Path, default=DEFAULT_HOURLY)
    parser.add_argument("--a5", type=Path, default=DEFAULT_A5)
    parser.add_argument("--atr", type=Path, default=DEFAULT_ATR)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    args = parser.parse_args()

    required = [
        args.run_dir / "v2_candidate_replay.parquet",
        args.run_dir / "loss_state_counterfactuals.parquet",
        args.run_dir / "loss_heads_oof_predictions.parquet",
        args.run_dir / "loss_head_fit_audit.parquet",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"V2 completion lacks model artifacts: {missing}")
    if (args.run_dir / "run_manifest.json").exists():
        raise FileExistsError("completed immutable V2 run already has a manifest")

    policy, _, _ = _load_frozen_policy(args.policy, "unused")
    v1, rows, _states, _paths, _baseline = _v1_population(
        args.v1_dir, args.hourly_dir, args.a5, args.atr,
    )
    replay = pd.read_parquet(args.run_dir / "v2_candidate_replay.parquet")
    population = v1.copy()
    portfolio_root = args.run_dir / "portfolio"
    portfolio_root.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for arm in ARM_ORDER:
        chosen = replay[replay.arm.eq(arm)].set_index("candidate_id")
        if len(chosen) != len(population):
            raise RuntimeError(f"{arm} candidate replay is incomplete")
        frame = population.copy()
        ids = frame.candidate_id.astype(str)
        for destination, source in (
            ("adaptive_net_bps", "adaptive_net_bps"),
            ("adaptive_gross_bps", "adaptive_gross_bps"),
            ("adaptive_exit_bar", "adaptive_exit_bar"),
            ("adaptive_exit_reason", "adaptive_exit_reason"),
        ):
            frame[destination] = ids.map(chosen[source]).to_numpy()
        summaries[arm] = _portfolio(frame, arm.upper(), portfolio_root)
    portfolio_metrics = _portfolio_metrics(args.run_dir, summaries)

    labels = pd.read_parquet(args.run_dir / "loss_state_counterfactuals.parquet")
    frontier = []
    for allowance in (0, 25, 50, 100):
        gain = labels[f"loss_oracle_gain_cost{allowance}_bps"]
        cost = labels[f"loss_oracle_profit_cost{allowance}_bps"]
        selected = labels[f"loss_oracle_action_cost{allowance}"]
        frontier.append({
            "profit_cost_allowance_bps": allowance,
            "states": len(labels),
            "oracle_gain_bps": gain.mean(),
            "mean_profit_cost_bps": cost.mean(),
            "intervention_fraction": (selected > 0).mean(),
            "avoidable_remaining_loss_bps": labels.avoidable_remaining_loss_bps.mean(),
        })
    pd.DataFrame(frontier).to_parquet(
        args.run_dir / "loss_protection_oracle_frontier.parquet", index=False,
    )
    predictions = pd.read_parquet(args.run_dir / "loss_heads_oof_predictions.parquet")
    candidates = pd.read_parquet(args.run_dir / "loss_feature_candidates.parquet")
    drawdown = pd.read_parquet(args.run_dir / "max_drawdown_episode.parquet")
    manifest = {
        "schema": "adaptive_exit_loss_control_v2_v1",
        "status": "COMPLETED_NOT_PROMOTED",
        "promotion": "none",
        "v1_dir": args.v1_dir,
        "hourly_dir": args.hourly_dir,
        "policy": args.policy,
        "policy_sha256": _sha(args.policy),
        "v1_frozen": True,
        "stop_levels": STOP_LEVELS,
        "arms": ARM_ORDER,
        "selection_period": "2025",
        "confirmation_period": "2026",
        "decision_clock": "completed hourly bar",
        "action_effective": "next hourly bar",
        "cost_bps_once": 100,
        "feature_candidates": len(candidates),
        "oof_states": len(predictions),
        "drawdown_episode": drawdown.to_dict("records"),
        "portfolio": portfolio_metrics.to_dict("records"),
        "frozen_policy": policy,
    }
    (args.run_dir / "run_manifest.json").write_text(
        json.dumps(_safe(manifest), indent=2) + "\n"
    )
    print(json.dumps({
        "status": manifest["status"], "oof_states": len(predictions),
        "output": str(args.run_dir),
    }))


if __name__ == "__main__":
    main()
