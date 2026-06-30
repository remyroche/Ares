#!/usr/bin/env python3
"""Build live-replay-derived action rows and defensive-success labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from extreme_price_movements.portfolio_policy_replay import fit_hierarchical_ev_curves, normalise_candidate_table
from scripts.materialize_size_action_live_scorer import score_size_action_frame
from scripts.run_exact_state_counterfactual_oracle import _capture_snapshots
from scripts.run_exact_state_size_action_learning import SIZE_ACTIONS, _feature_row
from scripts.run_global_portfolio_period_multiplier import _load_policy_params
from scripts.run_size_action_live_scorer_replay import _accepted_trades, _head_from_strategy, _load_candidates, _prepare_accepted, _summarise


def _strategy_notional_at_timestamp(accepted: pd.DataFrame, timestamp: pd.Timestamp, strategy_id: str) -> float:
    if accepted.empty:
        return 0.0
    ts = pd.to_datetime(accepted["timestamp"], utc=True, errors="coerce")
    rows = accepted.loc[ts.eq(pd.Timestamp(timestamp)) & accepted["strategy_id"].astype(str).eq(str(strategy_id))]
    if rows.empty:
        return 0.0
    return float(pd.to_numeric(rows.get("position_size"), errors="coerce").fillna(0.0).sum())


def _build_action_feature_rows(
    candidates: pd.DataFrame,
    baseline_decisions: pd.DataFrame,
    snapshots: dict[pd.Timestamp, Any],
    params: Any,
    *,
    multipliers: tuple[float, ...],
) -> pd.DataFrame:
    baseline_accepted = _accepted_trades(candidates, baseline_decisions)
    strategy_ids = sorted(str(x) for x in candidates["strategy_id"].dropna().astype(str).unique())
    rows: list[dict[str, Any]] = []
    for timestamp in pd.DatetimeIndex(candidates["timestamp"].dropna().drop_duplicates().sort_values()):
        ts = pd.Timestamp(timestamp)
        state = snapshots.get(ts)
        if state is None:
            continue
        for strategy_id in strategy_ids:
            local = candidates.loc[candidates["timestamp"].eq(ts) & candidates["strategy_id"].astype(str).eq(strategy_id)]
            if local.empty:
                continue
            affected = _strategy_notional_at_timestamp(baseline_accepted, ts, strategy_id)
            for multiplier in multipliers:
                # This panel is for live scorer features, not exact counterfactual
                # labels. Binding is approximated by whether the baseline accepted
                # any notional for that strategy at this timestamp.
                rows.append(
                    _feature_row(
                        candidates,
                        ts,
                        strategy_id,
                        state,
                        params,
                        float(multiplier),
                        action_binds=bool(affected > 0.0 and float(multiplier) < 1.0),
                        affected_notional=affected,
                    )
                )
    return pd.DataFrame(rows)


def _attach_defensive_labels(scores: pd.DataFrame, baseline_accepted: pd.DataFrame) -> pd.DataFrame:
    labels = baseline_accepted.groupby(["timestamp", "strategy_id"], dropna=False).agg(
        baseline_group_net_pnl=("net_pnl", "sum"),
        baseline_group_winner_pnl=("net_pnl", lambda s: float(s[s > 0].sum())),
        baseline_group_loser_loss=("net_pnl", lambda s: float(-s[s < 0].sum())),
        baseline_group_trades=("net_pnl", "size"),
    ).reset_index()
    out = scores.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["strategy_id"] = out["strategy_id"].astype(str)
    out["head"] = out["strategy_id"].map(_head_from_strategy)
    out["selected_multiplier"] = pd.to_numeric(out.get("selected_multiplier"), errors="coerce").fillna(1.0)
    out["scorer_intervention"] = out["selected_multiplier"].lt(1.0)
    out = out.merge(labels, on=["timestamp", "strategy_id"], how="left")
    for col in ("baseline_group_net_pnl", "baseline_group_winner_pnl", "baseline_group_loser_loss", "baseline_group_trades"):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out["defensive_success_value"] = -out["baseline_group_net_pnl"]
    out["defensive_success_target"] = out["defensive_success_value"].gt(0.0).astype(int)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--broad-candidates", type=Path, required=True)
    parser.add_argument("--deployable-candidates", type=Path, required=True)
    parser.add_argument("--scorer-bundle", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--policy-variant", default="refit_bar4_strategy_bar2")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--market-mode", default="perps")
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    params, _payload = _load_policy_params(args.policy_manifest, args.policy_variant)
    candidates = _load_candidates(args.broad_candidates, start=start, end=end)
    deployable = normalise_candidate_table(pd.read_parquet(args.deployable_candidates))
    deployable_train = deployable.loc[deployable["timestamp"].lt(start)].copy()
    if deployable_train.empty:
        raise RuntimeError(
            f"No deployable rows before {start}; choose a later start so EV curves are causal."
        )
    ev_curve = fit_hierarchical_ev_curves(deployable_train)
    baseline_decisions, _equity, snapshots = _capture_snapshots(candidates, params, ev_curve, args.market_mode)
    baseline_accepted = _prepare_accepted(_accepted_trades(candidates, baseline_decisions), "baseline")
    baseline_accepted.to_csv(args.out_dir / "baseline_accepted_trades.csv", index=False)
    _summarise(baseline_accepted, ["arm"]).to_csv(args.out_dir / "baseline_overall.csv", index=False)

    action_features = _build_action_feature_rows(
        candidates,
        baseline_decisions,
        snapshots,
        params,
        multipliers=tuple(float(x) for x in SIZE_ACTIONS),
    )
    action_features.to_parquet(args.out_dir / "action_feature_rows.parquet", index=False)
    scores = score_size_action_frame(args.scorer_bundle, action_features)
    scores["timestamp"] = pd.to_datetime(scores["timestamp"], utc=True, errors="coerce")
    scores["multiplier"] = pd.to_numeric(scores.get("selected_multiplier"), errors="coerce").fillna(1.0).clip(0.0, 1.0)
    scores.to_csv(args.out_dir / "size_action_scores.csv", index=False)
    labels = _attach_defensive_labels(scores, baseline_accepted)
    labels.to_csv(args.out_dir / "defensive_success_training_labels.csv", index=False)
    manifest = {
        "generated_by": "build_size_action_live_replay_training_panel",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "broad_candidates": str(args.broad_candidates),
        "deployable_candidates": str(args.deployable_candidates),
        "scorer_bundle": str(args.scorer_bundle),
        "policy_manifest": str(args.policy_manifest),
        "policy_variant": str(args.policy_variant),
        "market_mode": str(args.market_mode),
        "candidate_rows": int(len(candidates)),
        "candidate_timestamps": int(candidates["timestamp"].nunique()) if "timestamp" in candidates.columns else 0,
        "action_feature_rows": int(len(action_features)),
        "score_groups": int(len(scores)),
        "scorer_interventions": int(labels["scorer_intervention"].sum()),
        "intervention_positive_rate": float(labels.loc[labels["scorer_intervention"], "defensive_success_target"].mean())
        if bool(labels["scorer_intervention"].any())
        else 0.0,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
