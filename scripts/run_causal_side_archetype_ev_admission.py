#!/usr/bin/env python3
"""Apply the production fixed-EV admission policy to a strict OOS window.

The hierarchical EV map is assumed to have been fitted upstream.  This runner
only estimates the causal recent realized-minus-mapped EV correction.  Test
outcomes may be present in the reference ledger, but become visible only after
``outcome_resolved_at`` through the production threshold-policy implementation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from extreme_price_movements.inference.threshold_basis_policy import (
    apply_threshold_basis_policy_to_decisions,
    load_threshold_basis_policy,
)


POLICY_ID = "side_archetype_hier_ev_fixed70_trim10_21d_v1"


def _load(path: Path) -> pd.DataFrame:
    rows = pd.read_parquet(path)
    ts_col = "__ts__" if "__ts__" in rows else "timestamp"
    symbol_col = "__symbol__" if "__symbol__" in rows else "symbol"
    required = {
        ts_col,
        symbol_col,
        "side_name",
        "archetype_policy_key",
        "rank_mlp_direct",
        "expected_net_ev_after_1pct_mlp_direct",
        "ev_after_1pct",
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    rows = rows.copy(deep=False)
    rows["timestamp"] = pd.to_datetime(rows[ts_col], utc=True, errors="coerce")
    rows["symbol"] = rows[symbol_col].astype(str)
    rows["side_name"] = rows["side_name"].astype(str).str.lower()
    archetype = rows["archetype_policy_key"].astype("string").fillna("missing")
    for side in ("long", "short"):
        prefix = f"{side}__"
        mask = rows["side_name"].eq(side) & archetype.str.startswith(prefix)
        archetype.loc[mask] = archetype.loc[mask].str[len(prefix) :]
    rows["policy_archetype"] = archetype
    return rows.sort_values(
        ["timestamp", "symbol", "side_name"], kind="stable"
    ).reset_index(drop=True)


def _reference(rows: pd.DataFrame, outcome_horizon_hours: int) -> pd.DataFrame:
    reference = rows.loc[
        :,
        [
            "timestamp",
            "symbol",
            "side_name",
            "policy_archetype",
            "rank_mlp_direct",
            "expected_net_ev_after_1pct_mlp_direct",
            "ev_after_1pct",
        ],
    ].rename(
        columns={
            "expected_net_ev_after_1pct_mlp_direct": "mapped_expected_ev"
        }
    )
    reference["outcome_resolved_at"] = reference["timestamp"] + pd.Timedelta(
        hours=int(outcome_horizon_hours)
    )
    return reference.drop_duplicates(
        ["timestamp", "symbol", "side_name", "policy_archetype"], keep="last"
    )


def _policy(reference_path: Path, reference_rows: int, horizon_hours: int) -> dict[str, Any]:
    return {
        "schema_version": "threshold_basis_policy_v3",
        "policy_id": POLICY_ID,
        "policy_name": "s59_packb_meta_hierev_sidearch_ev70_trim10_21d_v1",
        "enabled": True,
        "family": "side_archetype_expected_ev_recent_correction",
        "window_days": 21,
        "selection_mode": "fixed_corrected_ev_threshold",
        "fixed_target_net_ev": 0.007,
        "recalibration_frequency": "1d_at_00_utc",
        "robust_daily_residual_trim_fraction": 0.10,
        "robust_daily_residual_normalization": "median_iqr",
        "top_fraction": 0.10,
        "min_reference_rows": 40,
        "side_support_target": 320.0,
        "local_support_target": 160.0,
        "recent_ev_correction_cap": 0.03,
        "ev_rank_blend_weight": 1.0,
        "rank_blend_parent_col": "v9_tail95_predecessor_rank",
        "mapped_expected_ev_col": "expected_net_ev_after_1pct_side_archetype",
        "reference_mapped_expected_ev_col": "mapped_expected_ev",
        "live_score_col": "expected_ev_rank_score",
        "return_col": "ev_after_1pct",
        "reference_candidates_path": reference_path.name,
        "reference_columns": [
            "timestamp",
            "symbol",
            "side_name",
            "policy_archetype",
            "rank_mlp_direct",
            "mapped_expected_ev",
            "ev_after_1pct",
            "outcome_resolved_at",
        ],
        "reference_rows": int(reference_rows),
        "outcome_horizon_hours": int(horizon_hours),
        "cost_contract": (
            "mapped and realized EV are already net of the pooled per-symbol "
            "p90 full spread plus the sole 0.15% round-trip fee; this layer "
            "subtracts no additional cost"
        ),
        "causal_contract": (
            "At t, reference timestamp and outcome_resolved_at must both precede "
            "the daily UTC as-of boundary; test outcomes enter only after resolution"
        ),
    }


def _decisions(rows: pd.DataFrame) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for row in rows.itertuples(index=False):
        decisions.append(
            {
                "signal_bar_ts": row.timestamp,
                "symbol": row.symbol,
                "side_name": row.side_name,
                "archetype_policy_key": row.policy_archetype,
                "rank_mlp_direct": float(row.rank_mlp_direct),
                "policy_rank_pct": float(row.rank_mlp_direct),
                "v9_tail95_predecessor_rank": float(
                    getattr(row, "policy_parent_rank", row.rank_mlp_direct)
                ),
                "expected_net_ev_after_1pct_side_archetype": float(
                    row.expected_net_ev_after_1pct_mlp_direct
                ),
            }
        )
    return decisions


def _metrics(rows: pd.DataFrame, *, scope: str) -> dict[str, Any]:
    if rows.empty:
        return {
            "scope": scope,
            "rows": 0,
            "days": 0,
            "trades_per_day": 0.0,
            "mean_net_ev_after_1pct": np.nan,
            "sum_net_ev_after_1pct": 0.0,
            "positive_ev_rate": np.nan,
        }
    ev = pd.to_numeric(rows["ev_after_1pct"], errors="coerce")
    days = int(rows["timestamp"].dt.floor("D").nunique())
    return {
        "scope": scope,
        "rows": int(len(rows)),
        "days": days,
        "trades_per_day": float(len(rows) / max(days, 1)),
        "mean_net_ev_after_1pct": float(ev.mean()),
        "sum_net_ev_after_1pct": float(ev.sum()),
        "positive_ev_rate": float(ev.gt(0.0).mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-start", required=True)
    parser.add_argument("--test-end", required=True)
    parser.add_argument("--outcome-horizon-hours", type=int, default=12)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(args.test_start, tz="UTC")
    end = pd.Timestamp(args.test_end, tz="UTC")
    history = _load(args.history)
    test = _load(args.test)
    history = history.loc[history["timestamp"].lt(start)].reset_index(drop=True)
    test = test.loc[
        test["timestamp"].ge(start) & test["timestamp"].lt(end)
    ].reset_index(drop=True)
    if history.empty or test.empty:
        raise ValueError(
            f"empty split after UTC bounds: history={len(history)} test={len(test)}"
        )
    if not history["timestamp"].max() < start:
        raise AssertionError("history overlaps the OOS test window")

    # Include test rows in the persisted reference so the same artifact can be
    # replayed causally and used live later. The production selector filters
    # both timestamp and outcome_resolved_at at every decision as-of boundary.
    reference = _reference(
        pd.concat([history, test], ignore_index=True, copy=False),
        args.outcome_horizon_hours,
    )
    reference_path = output_dir / "threshold_basis_reference_sidearch_ev21d.parquet"
    reference.to_parquet(reference_path, index=False, compression="zstd")
    policy_path = output_dir / "threshold_basis_policy_sidearch_ev70_trim10_21d.json"
    policy_path.write_text(
        json.dumps(
            _policy(reference_path, len(reference), args.outcome_horizon_hours),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    policy = load_threshold_basis_policy(policy_path)
    decisions = _decisions(test)
    apply_threshold_basis_policy_to_decisions(decisions, policy=policy)
    diagnostics = pd.DataFrame(decisions)
    diagnostics = pd.concat(
        [test.reset_index(drop=True), diagnostics.reset_index(drop=True)], axis=1
    )
    diagnostics = diagnostics.loc[:, ~diagnostics.columns.duplicated(keep="last")]
    diagnostics["calibrated_score"] = pd.to_numeric(
        diagnostics["threshold_basis_rank_score"], errors="coerce"
    ).astype(np.float32)
    diagnostics["rank_pct"] = diagnostics["calibrated_score"]
    diagnostics["policy_rank_pct"] = diagnostics["calibrated_score"]
    selected = diagnostics.loc[
        diagnostics["threshold_basis_selected"].fillna(False).astype(bool)
    ].copy()

    diagnostics.to_parquet(
        output_dir / "admission_diagnostics.parquet", index=False, compression="zstd"
    )
    selected.to_parquet(
        output_dir / "admitted_oos_rows.parquet", index=False, compression="zstd"
    )
    summaries = pd.DataFrame(
        [_metrics(test, scope="all_oos"), _metrics(selected, scope="admitted")]
    )
    summaries.to_csv(output_dir / "summary.csv", index=False)
    weekly = selected.assign(
        week_start=selected["timestamp"].dt.floor("D")
        - pd.to_timedelta(selected["timestamp"].dt.weekday, unit="D")
    ).groupby("week_start", observed=True).agg(
        selected_rows=("timestamp", "size"),
        days=("timestamp", lambda value: value.dt.floor("D").nunique()),
        mean_net_ev_after_1pct=("ev_after_1pct", "mean"),
        sum_net_ev_after_1pct=("ev_after_1pct", "sum"),
        positive_ev_rate=("ev_after_1pct", lambda value: value.gt(0.0).mean()),
    ).reset_index()
    weekly["trades_per_day"] = weekly["selected_rows"] / weekly["days"].clip(lower=1)
    weekly.to_csv(output_dir / "weekly_metrics.csv", index=False)
    manifest = {
        "schema": "causal_side_archetype_ev_admission_oos_v1",
        "policy_id": POLICY_ID,
        "history_source": str(args.history),
        "test_source": str(args.test),
        "test_start": start.isoformat(),
        "test_end_exclusive": end.isoformat(),
        "history_rows": int(len(history)),
        "test_rows": int(len(test)),
        "reference_rows": int(len(reference)),
        "admitted_rows": int(len(selected)),
        "leakage_contract": (
            "hierarchical EV is supplied by upstream OOS models; admission uses "
            "the production selector and only rows resolved before each UTC as-of"
        ),
        "cost_contract": "ev_after_1pct is net of one 1% round-trip cost",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(summaries.to_string(index=False))
    print(weekly.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
