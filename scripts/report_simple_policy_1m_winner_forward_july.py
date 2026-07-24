#!/usr/bin/env python3
"""Extend the frozen joint-trailing + raw-Bayesian replay through July 16.

The extension uses the canonical current-policy admission frontier materialized
from historical inference.  Geometry and sizing remain frozen at the fold-3
May 1--June 14 fit; no July outcome is used to refit either component.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.simple_policy_1m_constrained import (  # noqa: E402
    FAMILY_TRAILING_ONLY,
    ConstrainedReplaySpec,
)
from extreme_price_movements.simple_policy_optimiser import (  # noqa: E402
    _with_policy_spread_cost_columns,
)
from scripts.report_simple_policy_1m_winner_weekly import (  # noqa: E402
    _empty_outputs,
    _json_safe,
    _weekly_rows,
)
from scripts.run_simple_policy_1m_capital_ablation import (  # noqa: E402
    _load_deployed_side_params,
    _load_or_build_path_cache,
)
from scripts.run_simple_policy_1m_constrained_search import (  # noqa: E402
    ExperimentData,
    _causal_entry_atr,
    _indices_between,
)
from scripts.run_simple_policy_1m_contextual_ablation import (  # noqa: E402
    _bayesian_sizes,
    _load_atr,
    _load_context,
)


BASE = Path(
    "data_perp/reports/meta_v9_recovery_20260717/"
    "residual_state_mda95_hier_newaegmm_downstream_retrain_v1"
)
CHAMPION = Path(
    "data_perp/reports/"
    "simple_policy_1m_joint_trailing_raw_bayesian_champion_20260718_v1"
)
FORWARD_SOURCE = Path(
    "data_perp/reports/july_01_16_current_policy_metrics_20260717/"
    "current_policy_candidates_through_july16.parquet"
)
SOURCE_CONTEXTS = (
    Path(
        "data_perp/reports/july_01_16_current_policy_metrics_20260717/"
        "batch_predictions_july01_16_with_expert/july_08_10_complete_predictions.parquet"
    ),
    Path(
        "data_perp/reports/july_01_16_current_policy_metrics_20260717/"
        "batch_predictions_july15_16/july_08_10_complete_predictions.parquet"
    ),
)


def _assign_outputs(
    target: dict[str, np.ndarray], positions: np.ndarray, source: Mapping[str, np.ndarray]
) -> None:
    for key in target:
        target[key][positions] = source[key]


def _forward_context(rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    parts: list[pd.DataFrame] = []
    columns = [
        "__ts__",
        "__symbol__",
        "side_name",
        "threshold_basis_selected",
        "expected_net_ev_after_1pct",
        "meta_hit_probability_uncertainty_p1mp",
        "gmm_ood_score",
        "cluster_entropy_norm",
    ]
    for source_i, path in enumerate(SOURCE_CONTEXTS):
        part = pd.read_parquet(path, columns=columns)
        # Context is model-produced before admission.  The canonical frontier
        # was assembled from several parity-safe historical batches, so use the
        # complete scored rows here and join only the already-admitted keys.
        part = part.copy()
        part["timestamp"] = pd.to_datetime(part.pop("__ts__"), utc=True)
        part["symbol"] = part.pop("__symbol__").astype(str)
        # The second source is authoritative from the start of its tail replay.
        if source_i == 0:
            part = part.loc[part["timestamp"] < pd.Timestamp("2026-07-15 16:00", tz="UTC")]
        else:
            part = part.loc[part["timestamp"] >= pd.Timestamp("2026-07-15 16:00", tz="UTC")]
        parts.append(part)
    source = pd.concat(parts, ignore_index=True, copy=False)
    key = ["timestamp", "symbol", "side_name"]
    if source.duplicated(key).any():
        raise RuntimeError("Forward Bayesian context key is not unique")
    context_columns = [
        "expected_net_ev_after_1pct",
        "meta_hit_probability_uncertainty_p1mp",
        "gmm_ood_score",
        "cluster_entropy_norm",
    ]
    merged = rows[key].merge(
        source[key + context_columns], on=key, how="left", validate="one_to_one", indicator=True
    )
    exact = merged["_merge"].eq("both")
    merged = merged.drop(columns="_merge").rename(
        columns={"expected_net_ev_after_1pct": "expected_net_ev_after_1pct_mlp_direct"}
    )
    finite = merged.drop(columns=key).apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    return merged.drop(columns=key), {
        "source_ledgers": [str(path) for path in SOURCE_CONTEXTS],
        "rows": int(len(merged)),
        "exact_key_rows": int(exact.sum()),
        "exact_key_coverage": float(exact.mean()),
        "fully_finite_rows_before_fallback": int(finite.sum()),
        "fully_finite_coverage_before_fallback": float(finite.mean()),
        "join_key": key,
        "expected_ev_alias": "expected_net_ev_after_1pct -> expected_net_ev_after_1pct_mlp_direct",
    }


def _combine_outputs(parts: list[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CHAMPION / "forward_replay_jul11_17_v1",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    old_candidates = BASE / "execution_candidates_may_july_v1/simple_policy_candidates_with_archetypes.parquet"
    rich = BASE / "admission_may_july_oos_v1/admitted_oos_rows_execution_ledger.parquet"
    posterior = BASE / "complete_parent_state_july_v1/complete_oos_residual_event_states.parquet"
    parent_summary = BASE / "simple_policy_mayjune_fit_july_holdout_v1/side_parent_policy_summary.csv"
    params_path = CHAMPION / "evidence/nested_params.json"
    old_atr_path = CHAMPION / "replay/causal_entry_atr_audit.parquet"
    old_cache_path = CHAMPION / "replay/path_cache"
    store_root = Path("data_perp/exchanges/krakenfutures/execution_1m")

    old_rows = pd.read_parquet(old_candidates)
    old_rows["timestamp"] = pd.to_datetime(old_rows["timestamp"], utc=True)
    old_rows = old_rows.sort_values(
        ["timestamp", "rank_pct"], ascending=[True, False], kind="mergesort"
    ).reset_index(drop=True)
    old_context, _, old_context_audit = _load_context(old_rows, rich, posterior)
    old_atr = _load_atr(old_rows, old_atr_path)
    deployed, _ = _load_deployed_side_params(parent_summary)
    spec = ConstrainedReplaySpec()
    old_open, old_high, old_low, old_close, old_valid, old_path_manifest = _load_or_build_path_cache(
        old_rows, store_root=store_root, cache_dir=old_cache_path, spec=spec, rebuild=False
    )
    old_data = ExperimentData(
        old_rows, old_open, old_high, old_low, old_close, old_valid, old_atr, spec, deployed
    )
    train_idx = _indices_between(old_data, "2026-05-01", "2026-06-14")

    forward = pd.read_parquet(FORWARD_SOURCE)
    forward["timestamp"] = pd.to_datetime(forward["timestamp"], utc=True)
    forward = forward.loc[
        forward["timestamp"].ge(pd.Timestamp("2026-07-11", tz="UTC"))
        & forward["timestamp"].lt(pd.Timestamp("2026-07-17", tz="UTC"))
    ].copy()
    forward = _with_policy_spread_cost_columns(forward, market_mode="perps")
    forward = forward.sort_values(
        ["timestamp", "rank_pct"], ascending=[True, False], kind="mergesort"
    ).reset_index(drop=True)
    forward_context, forward_context_audit = _forward_context(forward)
    fallback_counts: dict[str, int] = {}
    for column in forward_context.columns:
        values = pd.to_numeric(forward_context[column], errors="coerce")
        missing = ~np.isfinite(values.to_numpy(dtype=np.float64))
        fallback_counts[column] = int(missing.sum())
        if missing.any():
            frozen_median = float(
                np.nanmedian(
                    pd.to_numeric(old_context.iloc[train_idx][column], errors="coerce").to_numpy(
                        dtype=np.float64
                    )
                )
            )
            values.loc[missing] = frozen_median
        forward_context[column] = values
    forward_context_audit["frozen_train_median_fallback_counts"] = fallback_counts
    forward_context_audit["fallback_contract"] = (
        "Missing historical context is replaced by the May1-Jun14 frozen training median; "
        "this is neutral under the winner's robust z-scaling and consumes no July outcome."
    )
    if not np.isfinite(forward_context.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("Forward Bayesian context remains non-finite after frozen fallback")

    forward_atr, forward_atr_audit, forward_atr_manifest = _causal_entry_atr(
        forward,
        store_root=store_root,
        deployed_by_side=deployed,
        parent_summary=parent_summary,
        warmup_hours=48,
    )
    forward_atr_audit.to_parquet(args.output_dir / "causal_entry_atr_audit.parquet", index=False)
    if forward_atr_manifest["coverage"] < 0.999:
        raise RuntimeError(f"Forward causal ATR coverage is {forward_atr_manifest['coverage']:.2%}")
    forward_open, forward_high, forward_low, forward_close, forward_valid, forward_path_manifest = (
        _load_or_build_path_cache(
            forward,
            store_root=store_root,
            cache_dir=args.output_dir / "path_cache",
            spec=spec,
            rebuild=False,
        )
    )
    forward_data = ExperimentData(
        forward,
        forward_open,
        forward_high,
        forward_low,
        forward_close,
        forward_valid,
        forward_atr,
        spec,
        deployed,
    )
    if forward_data.valid.mean() < 0.999:
        raise RuntimeError(f"Forward exact replay coverage is {forward_data.valid.mean():.2%}")

    params = json.loads(params_path.read_text())
    fold3_params = params["fold_3"]["full_train_parent"]
    fold3_sizing = params["fold_3"]["sizing"]
    train_outputs = old_data.simulate(train_idx, fold3_params, FAMILY_TRAILING_ONLY)

    combined_rows = pd.concat([old_rows, forward], ignore_index=True, copy=False)
    combined_context = pd.concat([old_context, forward_context], ignore_index=True, copy=False)
    sizing_data = SimpleNamespace(
        rows=combined_rows,
        side=pd.to_numeric(combined_rows["side"], errors="coerce").to_numpy(dtype=np.float64),
        rank=pd.to_numeric(combined_rows["rank_pct"], errors="coerce").to_numpy(dtype=np.float64),
    )
    forward_combined_idx = np.arange(len(old_rows), len(combined_rows), dtype=np.int64)
    combined_sizes, sizing_state = _bayesian_sizes(
        sizing_data,
        train_idx,
        forward_combined_idx,
        train_outputs,
        combined_context,
        strength=float(fold3_sizing["strength"]),
        ood_weight=float(fold3_sizing["ood_weight"]),
    )
    forward_sizes = combined_sizes[forward_combined_idx]
    forward_updated = forward_data.simulate(
        np.arange(len(forward), dtype=np.int64), fold3_params, FAMILY_TRAILING_ONLY
    )
    forward_deployed = forward_data.simulate_deployed(np.arange(len(forward), dtype=np.int64))

    # Rebuild the full June/July stream so capacity admission carries cleanly
    # across the July 10/11 boundary and the Jul 6--12 week is not split.
    old_ts = pd.to_datetime(old_rows["timestamp"], utc=True)
    old_report_idx = np.flatnonzero(
        old_ts.ge(pd.Timestamp("2026-06-01", tz="UTC")).to_numpy() & old_data.valid
    )
    old_report_rows = old_rows.iloc[old_report_idx]
    old_updated = _empty_outputs(len(old_report_idx))
    old_sizes = np.ones(len(old_report_idx), dtype=np.float64)
    for fold, apply_start, apply_end, train_start, train_end in (
        ("fold_2", "2026-06-01", "2026-06-15", "2026-05-01", "2026-05-31"),
        ("fold_3", "2026-06-15", "2026-07-11", "2026-05-01", "2026-06-14"),
    ):
        positions = np.flatnonzero(
            old_report_rows["timestamp"].ge(pd.Timestamp(apply_start, tz="UTC")).to_numpy()
            & old_report_rows["timestamp"].lt(pd.Timestamp(apply_end, tz="UTC")).to_numpy()
        )
        apply_idx = old_report_idx[positions]
        local_train_idx = _indices_between(old_data, train_start, train_end)
        fold_params = params[fold]["full_train_parent"]
        fit_outputs = old_data.simulate(local_train_idx, fold_params, FAMILY_TRAILING_ONLY)
        apply_outputs = old_data.simulate(apply_idx, fold_params, FAMILY_TRAILING_ONLY)
        sizing = params[fold]["sizing"]
        size_all, _ = _bayesian_sizes(
            old_data,
            local_train_idx,
            apply_idx,
            fit_outputs,
            old_context,
            strength=float(sizing["strength"]),
            ood_weight=float(sizing["ood_weight"]),
        )
        _assign_outputs(old_updated, positions, apply_outputs)
        old_sizes[positions] = size_all[apply_idx]
    old_deployed = old_data.simulate_deployed(old_report_idx)

    report_rows = pd.concat([old_report_rows, forward], ignore_index=True, copy=False)
    if not report_rows["timestamp"].is_monotonic_increasing:
        raise RuntimeError("Combined report stream is not chronological")
    report_data = SimpleNamespace(rows=report_rows)
    updated_outputs = _combine_outputs([old_updated, forward_updated])
    deployed_outputs = _combine_outputs([old_deployed, forward_deployed])
    updated_sizes = np.concatenate([old_sizes, forward_sizes])
    all_idx = np.arange(len(report_rows), dtype=np.int64)
    updated_weekly, updated_breakdown, updated_ledger = _weekly_rows(
        report_data,
        all_idx,
        updated_outputs,
        updated_sizes,
        policy="joint_trailing_plus_bayesian_raw",
    )
    deployed_weekly, deployed_breakdown, deployed_ledger = _weekly_rows(
        report_data,
        all_idx,
        deployed_outputs,
        np.ones(len(report_rows), dtype=np.float64),
        policy="current_deployed_reference",
    )
    weekly = pd.concat([updated_weekly, deployed_weekly], ignore_index=True)
    frontier_end_exclusive = pd.Timestamp("2026-07-17", tz="UTC")
    weekly["partial_week"] = pd.to_datetime(weekly["week_end_utc"], utc=True).ge(
        frontier_end_exclusive
    )
    weekly["frontier_observed_through_utc"] = [
        min(pd.Timestamp(value), frontier_end_exclusive - pd.Timedelta(nanoseconds=1))
        for value in pd.to_datetime(weekly["week_end_utc"], utc=True)
    ]
    reference = deployed_weekly.set_index("week")
    weekly["delta_net_pnl_vs_deployed"] = [
        float(row.net_pnl_bankroll - reference.loc[row.week, "net_pnl_bankroll"])
        if row.policy != "current_deployed_reference"
        else 0.0
        for row in weekly.itertuples()
    ]
    weekly["delta_hit_rate_vs_deployed"] = [
        float(row.hit_rate - reference.loc[row.week, "hit_rate"])
        if row.policy != "current_deployed_reference"
        else 0.0
        for row in weekly.itertuples()
    ]
    weekly = weekly.sort_values(["week_start_utc", "policy"]).reset_index(drop=True)
    breakdown = pd.concat([updated_breakdown, deployed_breakdown], ignore_index=True)
    ledger = pd.concat([updated_ledger, deployed_ledger], ignore_index=True)

    weekly.to_csv(args.output_dir / "weekly_metrics.csv", index=False)
    breakdown.to_csv(args.output_dir / "weekly_side_archetype_metrics.csv", index=False)
    ledger.to_parquet(args.output_dir / "selected_trade_ledger.parquet", index=False)
    forward.to_parquet(args.output_dir / "forward_candidates_jul11_16.parquet", index=False)
    manifest = {
        "status": "complete",
        "evidence": "frozen forward-OOS policy replay; no July outcome used for geometry or sizing fit",
        "policy": "joint_trailing_total_mfe_raw_bayesian_v1",
        "forward_candidate_source": str(FORWARD_SOURCE),
        "forward_entry_start_utc": str(forward["timestamp"].min()),
        "forward_entry_cutoff_utc": str(forward["timestamp"].max()),
        "latest_exit_observable_utc": str(forward["timestamp"].max() + pd.Timedelta(minutes=1440)),
        "forward_candidate_rows": int(len(forward)),
        "forward_valid_path_rows": int(forward_data.valid.sum()),
        "july17_status": "pending: a full causal 24h outcome path was not observable at run time",
        "fit_contract": {
            "geometry_and_sizing_train_start": "2026-05-01T00:00:00Z",
            "geometry_and_sizing_train_end_exclusive": "2026-06-14T00:00:00Z",
            "fold": "fold_3/full_train_parent",
            "strength": float(fold3_sizing["strength"]),
            "ood_weight": float(fold3_sizing["ood_weight"]),
            "sizing_state": sizing_state,
        },
        "comparison": "same candidates, exact 1m paths, causal entry-frozen ATR, 1% fee once, spread baseline, 8-open/2-new capacity",
        "old_context_audit": old_context_audit,
        "forward_context_audit": forward_context_audit,
        "forward_atr_manifest": forward_atr_manifest,
        "old_path_manifest": old_path_manifest,
        "forward_path_manifest": forward_path_manifest,
        "outputs": [
            "weekly_metrics.csv",
            "weekly_side_archetype_metrics.csv",
            "selected_trade_ledger.parquet",
            "forward_candidates_jul11_16.parquet",
            "causal_entry_atr_audit.parquet",
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(weekly.loc[weekly["week_start_utc"] >= pd.Timestamp("2026-06-01", tz="UTC")].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
