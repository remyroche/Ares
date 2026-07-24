#!/usr/bin/env python3
"""Verify the promoted fixed-EV selector against its portfolio matrix arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from extreme_price_movements.inference.threshold_basis_policy import (
    _load_reference,
    _select_side_archetype_expected_ev_batch,
    load_threshold_basis_policy,
)
from scripts.ablate_side_archetype_ev_portfolio_matrix import (
    Arm,
    _corrected_ev_for_arm,
    _daily_stats,
    _portfolio_replay,
)
from scripts.evaluate_side_archetype_expected_ev_policy import _load_rows


def _keys(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(
        frame[["__ts__", "__symbol__", "side_name"]].reset_index(drop=True)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oos-predictions", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--matrix-trades", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    policy = load_threshold_basis_policy(args.policy)
    start = pd.Timestamp("2026-04-01T00:00:00Z")
    source = _load_rows(args.oos_predictions, start).reset_index(drop=True)
    source["outcome_day"] = (source["__ts__"] + pd.Timedelta(hours=12)).dt.floor("D")
    source["residual"] = (
        pd.to_numeric(source["ev_after_1pct"], errors="coerce")
        - pd.to_numeric(
            source["expected_net_ev_after_1pct_mlp_direct"], errors="coerce"
        )
    )
    finite = np.isfinite(source["residual"].to_numpy(dtype=np.float64, copy=False))
    residual_rows = source.loc[
        finite, ["outcome_day", "side_name", "policy_archetype", "residual"]
    ]
    arm = Arm(0.007, 0.10, 21)
    matrix_corrected, _ = _corrected_ev_for_arm(
        source,
        arm,
        global_daily=_daily_stats(residual_rows, []),
        side_daily=_daily_stats(residual_rows, ["side_name"]),
        local_daily=_daily_stats(
            residual_rows, ["side_name", "policy_archetype"]
        ),
    )

    reference = _load_reference(policy)
    production_corrected = np.full(len(source), np.nan, dtype=np.float64)
    for day, positions in source.groupby(
        source["__ts__"].dt.floor("D"), sort=True, observed=True
    ).groups.items():
        asof = pd.Timestamp(day)
        all_prior = reference.loc[
            reference["timestamp"].lt(asof)
            & reference["outcome_resolved_at"].lt(asof)
        ]
        recent = all_prior.loc[
            all_prior["outcome_resolved_at"].ge(asof - pd.Timedelta(days=21))
        ]
        idx = np.asarray(list(positions), dtype=np.int64)
        batch = source.loc[
            idx, ["side_name", "policy_archetype", "policy_parent_rank"]
        ].rename(columns={"policy_parent_rank": "parent_rank"})
        batch["mapped_expected_ev"] = pd.to_numeric(
            source.loc[idx, "expected_net_ev_after_1pct_mlp_direct"],
            errors="coerce",
        ).to_numpy(dtype=np.float64, copy=False)
        _, metadata = _select_side_archetype_expected_ev_batch(
            batch, recent_ref=recent, all_prior=all_prior, policy=policy
        )
        production_corrected[idx] = metadata["corrected_expected_ev"].to_numpy(
            dtype=np.float64, copy=False
        )

    finite_compare = np.isfinite(matrix_corrected) & np.isfinite(production_corrected)
    diff = np.abs(
        matrix_corrected[finite_compare].astype(np.float64)
        - production_corrected[finite_compare]
    )
    production_idx = _portfolio_replay(
        source,
        production_corrected,
        target_ev=0.007,
        max_new_entries_per_bar=2,
        max_concurrent_positions=8,
        outcome_horizon_hours=12,
    )
    matrix_trades = pd.read_parquet(args.matrix_trades)
    matrix_trades = matrix_trades.loc[
        matrix_trades["arm"].eq("ev70bps_trim10_period21d")
    ].copy()
    production_keys = _keys(source.iloc[production_idx])
    matrix_keys = _keys(matrix_trades)
    missing = matrix_keys.difference(production_keys)
    extra = production_keys.difference(matrix_keys)
    payload = {
        "schema": "fixed_ev_policy_matrix_parity_v1",
        "policy_id": policy.get("policy_id"),
        "rows": int(len(source)),
        "finite_corrected_rows": int(finite_compare.sum()),
        "corrected_ev_max_abs_diff": float(diff.max()) if diff.size else None,
        "corrected_ev_mean_abs_diff": float(diff.mean()) if diff.size else None,
        "matrix_trades": int(len(matrix_trades)),
        "production_trades": int(len(production_idx)),
        "missing_matrix_trades": int(len(missing)),
        "extra_production_trades": int(len(extra)),
        "pass": bool(
            diff.size == len(source)
            and float(diff.max()) <= 1e-7
            and len(missing) == 0
            and len(extra) == 0
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
