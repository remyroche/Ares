#!/usr/bin/env python3
"""Matched multi-era validation for the path-based adaptive exit overlay.

The runner samples only from strict-prequential, causally admitted long rows,
then obtains complete 48x15m paths from the already-downloaded market store.
Path availability is an evaluation-coverage property and is never used by the
upstream score/admission process.  Every counterfactual uses the same frozen
dynamic SimplePolicyOptimiser baseline, entry, ATR, paths, and 100-bps cost.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.path_based_exit_optimisation import (  # noqa: E402
    AdaptiveExitConfig,
    build_action_grid,
    run_after_global_policy_optimisation,
    run_strict_oof_static_ablation,
)
from extreme_price_movements.simple_policy_optimiser import (  # noqa: E402
    _policy_params_from_deployment_strategy,
    _without_concurrency_param,
    simulate_and_score,
)
from scripts.replay_strict_r3_simple_policy_15m import (  # noqa: E402
    HORIZON_BARS,
    _coarse_causal_atr,
    _load_15m,
    _paths_for_group,
)


DEFAULT_LEDGERS = (
    ROOT / "data_perp/artifacts/strict_r3_lockstep_exactreserve_sourcealigned_policyexact_long_2025_janmar_reserve_seeded_admission_20260812_v1/reserve_seeded_causal_admission_ledger.parquet",
    ROOT / "data_perp/artifacts/strict_r3_lockstep_exactreserve_sourcealigned_policyexact_long_2025_aprjul_reserve_seeded_admission_20260812_v1/reserve_seeded_causal_admission_ledger.parquet",
    ROOT / "data_perp/artifacts/strict_r3_lockstep_exactreserve_monthstore_strictfull_long_2026_janjul_reserve_seeded_optimised_policy_20260812_v1/reserve_seeded_causal_admission_ledger.parquet",
)
DEFAULT_POLICY = (
    ROOT
    / "data_perp/artifacts/s59_s52_base66_residualstate_meta_v9tail95_mlp_hierev_ev70_geometry_20260717_v2/simple_policy_optimiser/deployment/best_policy_params.json"
)
DEFAULT_OUTPUT = (
    ROOT / "data_perp/artifacts/path_based_exit_portability_long_2025_2026_20260812_v1"
)

CONTEXT_FIELDS = (
    "base_score", "base_rank42", "base_anchor_bps",
    "conditional_consensus_rank", "upstream", "correctness_rank", "final_score",
    "rule_support_effective", "rule_support_p05", "rule_support_p95",
    "rule_support_contribution_weighted", "rule_ood_marginal",
    "rule_ood_joint_factorised", "path_support_effective_28d",
    "path_ood_marginal", "path_ood_conditioned", "model_ood_marginal",
    "model_ood_mahalanobis_diag", "model_drift_prototype_psi",
    "model_drift_prototype_ks", "k9_cluster_weighted_fit_support",
    "k9_cluster_weighted_distance", "k9_cluster_weighted_ood",
    "k9_cluster_weighted_mahalanobis_train",
    "k9_cluster_timestamp_cov_break_train", "k9_cluster_timestamp_corr_break_train",
    "k9_cluster_timestamp_mahalanobis_train",
    "k9_cluster_timestamp_support_weighted", "k9_cluster_timestamp_ood_weighted",
    "leaf_support_effective", "leaf_support_contribution_weighted",
    "leaf_ood_marginal", "leaf_ood_joint_rms", "k9_entropy", "k9_top2_margin",
    "k9_ood_distance", "k9_path_support_effective_28d", "k9_model_ood_marginal",
    "k9_model_drift_psi", "causal_21d_side_expected_net_bps",
    "causal_21d_side_positive_outcome_probability", "causal_21d_side_reference_rows",
    "causal_42d_side_reference_rows", "causal_84d_side_reference_rows",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _load_population(paths: list[Path], cap: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "candidate_id", "__decision_ts__", "__symbol__", "side_name",
        "policy_path_valid", "policy_label_available_ts",
        "causal_21d_side_admitted_ge_50bps", *CONTEXT_FIELDS,
    ]
    pieces = []
    coverage_rows = []
    for path in paths:
        available = set(pd.read_parquet(path, columns=[]).columns)
        # pandas cannot expose a parquet schema through columns=[] on all
        # engines; use pyarrow only for the column-name contract.
        import pyarrow.parquet as pq

        available = set(pq.ParquetFile(path).schema_arrow.names)
        selected_columns = [column for column in columns if column in available]
        part = pd.read_parquet(path, columns=selected_columns)
        part["__decision_ts__"] = pd.to_datetime(part["__decision_ts__"], utc=True)
        if "policy_label_available_ts" in part:
            part["policy_label_available_ts"] = pd.to_datetime(
                part["policy_label_available_ts"], utc=True
            )
        part = part.loc[part["side_name"].astype(str).str.lower().eq("long")].copy()
        admitted = part.get(
            "causal_21d_side_admitted_ge_50bps", pd.Series(False, index=part.index)
        ).fillna(False).astype(bool)
        coverage_rows.append(
            {
                "source": str(path),
                "rows": int(len(part)),
                "admitted_rows": int(admitted.sum()),
                "declared_policy_path_valid_rows": int(
                    part.get("policy_path_valid", pd.Series(False, index=part.index))
                    .fillna(False).sum()
                ),
            }
        )
        pieces.append(part.loc[admitted])
    frame = pd.concat(pieces, ignore_index=True)
    frame = frame.sort_values(["__decision_ts__", "candidate_id"], kind="stable")
    frame = frame.drop_duplicates("candidate_id", keep="last")
    frame["month"] = frame["__decision_ts__"].dt.strftime("%Y-%m")
    groups = list(frame.groupby("month", sort=True))
    quota = max(1, cap // max(len(groups), 1))
    sample = pd.concat(
        [
            part.sort_values("candidate_id", kind="stable").head(quota)
            for _, part in groups
        ],
        ignore_index=True,
    )
    if len(sample) < cap:
        remaining = frame.loc[~frame["candidate_id"].isin(sample["candidate_id"])]
        sample = pd.concat(
            [sample, remaining.sort_values("candidate_id", kind="stable").head(cap - len(sample))],
            ignore_index=True,
        )
    return sample.head(cap), pd.DataFrame(coverage_rows)


def _materialize_paths(
    population: pd.DataFrame,
) -> tuple[pd.DataFrame, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], pd.DataFrame]:
    count = len(population)
    arrays = tuple(
        np.full((count, HORIZON_BARS), np.nan, dtype=np.float32) for _ in range(4)
    )
    atr = np.full(count, np.nan, dtype=float)
    valid = np.zeros(count, dtype=bool)
    audits = []
    work = population.copy().reset_index(drop=True)
    work["__ts__"] = work["__decision_ts__"] - pd.Timedelta(hours=1)
    for symbol, group in work.groupby("__symbol__", sort=True):
        ts, opens, highs, lows, closes = _load_15m(str(symbol))
        if not len(ts):
            audits.append({"symbol": symbol, "rows": int(len(group)), "valid": 0})
            continue
        local_valid, *local_arrays = _paths_for_group(
            group, ts, opens, highs, lows, closes
        )
        local_positions = np.flatnonzero(local_valid)
        global_positions = group.index.to_numpy(int)[local_positions]
        if len(local_positions):
            for target, source in zip(arrays, local_arrays):
                target[global_positions] = source
            atr_series = _coarse_causal_atr(ts, opens, highs, lows, closes)
            local_atr = pd.to_datetime(
                group.iloc[local_positions]["__decision_ts__"], utc=True
            ).map(atr_series).to_numpy(float)
            atr[global_positions] = local_atr
            usable = np.isfinite(local_atr) & (local_atr > 0.0)
            valid[global_positions[usable]] = True
        audits.append(
            {"symbol": symbol, "rows": int(len(group)), "valid": int(valid[group.index].sum())}
        )
    selected = np.flatnonzero(valid)
    result = work.iloc[selected].copy().reset_index(drop=True)
    selected_arrays = tuple(array[selected] for array in arrays)
    result["timestamp"] = result["__decision_ts__"]
    result["symbol"] = result["__symbol__"].astype(str)
    result["side"] = 1.0
    result["rank_pct"] = pd.to_numeric(result["final_score"], errors="coerce").rank(pct=True)
    result["barrier_pct"] = atr[selected] / np.maximum(selected_arrays[0][:, 0], 1.0e-12)
    result["expected_half_spread_bps"] = 0.0
    result["exit_quote_half_spread_bps"] = 0.0
    result["entry_slippage_proxy_bps"] = 0.0
    result["market_mode"] = "perps"
    return result, selected_arrays, pd.DataFrame(audits)


def _load_policy(path: Path, strategy_id: str) -> tuple[dict[str, Any], float, float]:
    payload = json.loads(path.read_text())
    strategies = [
        strategy for strategy in payload.get("strategies", [])
        if str(strategy.get("strategy_id")) == strategy_id
    ]
    if len(strategies) != 1:
        raise ValueError(f"expected one strategy {strategy_id!r}, found {len(strategies)}")
    params, size_power, _threshold = _policy_params_from_deployment_strategy(
        strategies[0], payload.get("selection_rules", {})
    )
    if not all(field in params for field in (
        "sl_mult", "trailing_activation_mult", "trailing_power", "giveback_beta"
    )):
        raise ValueError("selected baseline lacks dynamic trailing fields")
    cost_pct = float(strategies[0].get("cost_pct_per_side", 0.005))
    if not np.isclose(cost_pct * 2.0, 0.01):
        raise ValueError("validation requires the declared 100-bps round-trip cost")
    params["replay_timeframe"] = "15m"
    params["market_mode"] = "perps"
    params["adverse_exit_enabled"] = False
    return _without_concurrency_param(params), size_power, cost_pct


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, action="append")
    parser.add_argument("--policy-json", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--strategy-id", default="long_s52_meta_threshold_handoff")
    parser.add_argument("--max-trades", type=int, default=2_000)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    ledgers = list(args.ledger or DEFAULT_LEDGERS)
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    population, source_coverage = _load_population(ledgers, args.max_trades)
    rows, paths, path_coverage = _materialize_paths(population)
    if len(rows) < 500:
        raise ValueError(f"insufficient complete-path support: {len(rows)}")
    params, size_power, cost_pct = _load_policy(args.policy_json, args.strategy_id)
    config = AdaptiveExitConfig(
        max_counterfactual_trades=len(rows),
        min_train_trades=200,
        min_validation_trades=50,
        action_batch_size=25,
    )
    counterfactual_dir = args.out_dir / "counterfactuals"
    manifest = run_after_global_policy_optimisation(
        rows=rows,
        paths=paths,
        baseline_params=params,
        cost_pct=cost_pct,
        size_power=size_power,
        output_dir=counterfactual_dir,
        simulator=simulate_and_score,
        config=config,
        entry_feature_columns=[field for field in CONTEXT_FIELDS if field in rows],
    )
    payload = np.load(counterfactual_dir / "counterfactual_action_values.npz")
    states = pd.read_parquet(counterfactual_dir / "adaptive_exit_decision_states.parquet")
    entry_states = states.loc[states["path_bar"].eq(0)].copy()
    identity_order = pd.Index(payload["candidate_id"].astype(str))
    entry_states = entry_states.set_index(entry_states["candidate_id"].astype(str)).loc[identity_order].reset_index(drop=True)
    actions = build_action_grid(config)
    decisions, metrics, oof_summary = run_strict_oof_static_ablation(
        entry_states,
        actions,
        payload["delta_q_bps"],
        payload["net_bps"],
        candidate_features=[
            column for column in entry_states
            if column.startswith("entry__") or column in {
                "atr_frac", "pnl_atr", "mfe_atr", "mae_atr",
            }
        ],
        output_dir=args.out_dir / "oof_static",
        config=config,
    )
    source_coverage.to_parquet(args.out_dir / "source_coverage.parquet", index=False)
    path_coverage.to_parquet(args.out_dir / "path_coverage.parquet", index=False)
    top = metrics.head(5).to_dict("records")
    final = {
        "schema": "path_based_exit_portability_validation_v1",
        "status": "STATIC_OOF_COMPLETE_SEQUENTIAL_REPLAY_REQUIRED",
        "ledgers": [{"path": str(path), "sha256": _sha(path)} for path in ledgers],
        "policy_json": str(args.policy_json),
        "policy_json_sha256": _sha(args.policy_json),
        "strategy_id": args.strategy_id,
        "sampled_admitted_rows": int(len(population)),
        "complete_path_rows": int(len(rows)),
        "complete_path_fraction": float(len(rows) / max(len(population), 1)),
        "months": sorted(rows["month"].astype(str).unique()),
        "baseline": params,
        "round_trip_cost_bps": float(cost_pct * 2.0 * 10_000.0),
        "counterfactual_manifest": manifest,
        "oof_summary": oof_summary,
        "top_static_arms": top,
        "promotion": "prohibited_without_sequential_oof_replay",
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(_safe(final), indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(_safe({"event": "complete", **final})))


if __name__ == "__main__":
    main()
