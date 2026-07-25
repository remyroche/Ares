#!/usr/bin/env python3
"""Replay the frozen April-fitted simple policy under portfolio constraints."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    fit_hierarchical_ev_curves,
    portfolio_policy_params_from_live_config,
    replay_candidates,
)
from extreme_price_movements.simple_policy_optimiser import (  # noqa: E402
    _build_simple_policy_candidate_rows,
)
from scripts.ablate_simple_policy_exit_geometry import (  # noqa: E402
    _load_bundles,
    _prepare_rows,
)
from scripts.run_s52_side_archetype_simple_policy_optimiser import (  # noqa: E402
    _attach_policy_archetype_column,
    _geometry_params_from_archetype_row,
    _params_from_parent_summary_row,
)

DEFAULT_ROOT = ROOT / "data_perp/artifacts/label_hpo_policy_replay_20260725_v1"
DEFAULT_CONFIG = (
    ROOT / "data_perp/artifacts/"
    "s59_s52_base66_residualstate_meta_v9tail95_mlp_hierev_ev70_geometry_20260717_v2/"
    "policy_params/optimized_portfolio_policy_config.json"
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _lookup(
    parent_path: Path, archetype_path: Path
) -> tuple[
    dict[str, tuple[dict[str, Any], float]], dict[tuple[str, str], Mapping[str, Any]]
]:
    parents: dict[str, tuple[dict[str, Any], float]] = {}
    for _, row in pd.read_csv(parent_path).iterrows():
        parents[str(row["strategy_id"])] = _params_from_parent_summary_row(
            row.to_dict()
        )
    archetypes: dict[tuple[str, str], Mapping[str, Any]] = {}
    for _, row in pd.read_csv(archetype_path).iterrows():
        archetypes[(str(row["strategy_id"]), str(row["policy_archetype"]))] = (
            row.to_dict()
        )
    return parents, archetypes


def _materialize(
    bundles: list[Any],
    *,
    parents: Mapping[str, tuple[dict[str, Any], float]],
    archetypes: Mapping[tuple[str, str], Mapping[str, Any]],
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for bundle in bundles:
        strategy = str(bundle.strategy_id)
        parent_params, parent_size = parents[strategy]
        work = _attach_policy_archetype_column(bundle.rows.copy(), strategy_id=strategy)
        for archetype, positions in work.groupby(
            "policy_archetype", sort=True
        ).indices.items():
            idx = np.asarray(positions, dtype=np.int64)
            rows = work.iloc[idx].reset_index(drop=True)
            paths = tuple(values[idx] for values in bundle.paths)
            contract = archetypes.get((strategy, str(archetype)))
            if contract is None:
                params, size_power = parent_params, parent_size
                source = "side_parent_fallback"
            else:
                params, size_power = _geometry_params_from_archetype_row(
                    contract,
                    parent_params=parent_params,
                    parent_size_power=parent_size,
                )
                source = "side_archetype_shrunk_geometry"
            frame = _build_simple_policy_candidate_rows(
                strategy_id=strategy,
                df_top=rows,
                paths=paths,
                cost_pct=0.005,
                best_params=params,
                best_size_power=size_power,
                base_strategy_threshold=0.90,
                market_mode="perps",
            )
            if not frame.empty:
                frame["policy_archetype"] = str(archetype)
                frame["policy_source"] = source
                outputs.append(frame)
    if not outputs:
        raise RuntimeError("no simple-policy candidates materialized")
    return pd.concat(outputs, ignore_index=True)


def _monthly(decisions: pd.DataFrame) -> list[dict[str, Any]]:
    accepted = decisions.loc[decisions["accepted"].astype(bool)].copy()
    accepted["month"] = pd.to_datetime(accepted["timestamp"], utc=True).dt.strftime(
        "%Y-%m"
    )
    rows: list[dict[str, Any]] = []
    for month, part in accepted.groupby("month", sort=True):
        returns = pd.to_numeric(part["position_net_return"], errors="coerce")
        rows.append(
            {
                "month": month,
                "accepted_trades": len(part),
                "mean_net_return": float(returns.mean()),
                "net_return_sum": float(returns.sum()),
                "positive_rate": float((returns > 0).mean()),
                "long_trades": int(
                    part["side"].astype(str).str.lower().eq("long").sum()
                ),
                "short_trades": int(
                    part["side"].astype(str).str.lower().eq("short").sum()
                ),
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output.mkdir(parents=True, exist_ok=True)
    rows = _prepare_rows(
        args.candidates,
        min_rank=0.90,
        rank_score_col="rank_pct",
        rank_scope="timestamp_side",
        apply_regime_ev_calibration_artifact=False,
    )
    april = rows.loc[
        rows["timestamp"].ge(pd.Timestamp("2026-04-01", tz="UTC"))
        & rows["timestamp"].lt(pd.Timestamp("2026-05-01", tz="UTC"))
    ].copy()
    holdout = rows.loc[
        rows["timestamp"].ge(pd.Timestamp("2026-05-01", tz="UTC"))
        & rows["timestamp"].lt(pd.Timestamp("2026-07-18", tz="UTC"))
    ].copy()
    parents, archetypes = _lookup(args.parent_summary, args.archetype_summary)
    april_candidates = _materialize(
        _load_bundles(
            april,
            data_root=str(args.data_root),
            market_mode="perps",
            path_len=96,
            min_rows_per_strategy=20,
        ),
        parents=parents,
        archetypes=archetypes,
    )
    holdout_candidates = _materialize(
        _load_bundles(
            holdout,
            data_root=str(args.data_root),
            market_mode="perps",
            path_len=96,
            min_rows_per_strategy=20,
        ),
        parents=parents,
        archetypes=archetypes,
    )
    april_candidates.to_parquet(
        args.output / "april_policy_candidates.parquet", index=False
    )
    holdout_candidates.to_parquet(
        args.output / "may_july17_policy_candidates.parquet", index=False
    )
    config = json.loads(args.portfolio_config.read_text(encoding="utf-8"))
    params = portfolio_policy_params_from_live_config(config)
    params = replace(
        params,
        strategy_ids=("long_label_hpo_winner", "short_label_hpo_winner"),
        strategy_cores=("label_hpo_winner",),
    )
    ev_curve = fit_hierarchical_ev_curves(april_candidates)
    decisions, equity, metrics = replay_candidates(
        holdout_candidates,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode="perps",
    )
    decisions.to_parquet(args.output / "portfolio_decisions.parquet", index=False)
    equity.to_parquet(args.output / "portfolio_equity.parquet", index=False)
    summary = {
        "chronology": {
            "simple_policy_and_ev_curve_fit": "2026-04-01/2026-05-01",
            "frozen_portfolio_replay": "2026-05-01/2026-07-18",
            "latest_candidate_timestamp": holdout_candidates["timestamp"].max(),
        },
        "coverage": {
            "april_candidate_rows": len(april_candidates),
            "holdout_candidate_rows": len(holdout_candidates),
            "accepted_rows": int(decisions["accepted"].astype(bool).sum()),
        },
        "constraints": {
            "source": str(args.portfolio_config),
            "max_new_entries_per_bar": params.max_new_entries_per_bar,
            "max_concurrent_per_symbol": params.max_concurrent_per_symbol,
            "wallet_allocation_cap": params.max_total_wallet_allocation_pct,
            "position_count_cap_enforced": params.enforce_position_count_cap,
            "max_concurrent_positions_emergency_bound": params.max_concurrent_positions,
        },
        "portfolio_metrics": metrics,
        "monthly_accepted_metrics": _monthly(decisions),
    }
    (args.output / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=DEFAULT_ROOT / "policy_handoff_apr_to_jul17.parquet",
    )
    parser.add_argument(
        "--parent-summary",
        type=Path,
        default=DEFAULT_ROOT / "simple_policy_optimizer/side_parent_policy_summary.csv",
    )
    parser.add_argument(
        "--archetype-summary",
        type=Path,
        default=DEFAULT_ROOT
        / "simple_policy_optimizer/side_archetype_policy_summary.csv",
    )
    parser.add_argument("--portfolio-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data_perp")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_ROOT / "portfolio_replay"
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(_json_safe(run(parse_args())), indent=2))
