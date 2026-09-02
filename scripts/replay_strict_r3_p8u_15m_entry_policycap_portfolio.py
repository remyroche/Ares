#!/usr/bin/env python3
"""Portfolio-constrained OOS replay for a capped rich policy plus entry head.

Each arm consumes an immutable target-free feature panel, its separately
materialised rich-policy labels, and strict-OOS entry-head predictions.  The
entry head is a demoter only: it cannot manufacture a candidate below the
named dual-MC1 floor.  The stateful continuation head is intentionally not
applied here; it is frozen to its original parent-policy state contract and
would need its own re-labelling/retraining to be comparable across new parent
exit geometries.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.portfolio_policy_replay import (
    compute_replay_metrics,
    normalise_candidate_table,
    replay_candidates,
)
from scripts.report_strict_r3_mc1_d2_controlled_portfolio import CAUSAL_AUCTION_CURVE, _params as portfolio_params
from scripts.replay_strict_r3_p8u_15m_continuation_portfolio import _attach_ids, _period_metrics


def _labels(root: Path) -> pd.DataFrame:
    paths = sorted(root.resolve().glob("policy_parts/symbol=*/policy_labels.parquet"))
    if not paths:
        raise FileNotFoundError(f"no materialised rich-policy labels under {root}")
    columns = [
        "candidate_id", "policy_path_valid", "policy_gross_bps", "policy_net_bps",
        "policy_exit_bar_15m", "policy_entry_price", "policy_exit_price",
        "policy_exit_reason", "policy_cost_bps",
    ]
    frame = pd.concat([pd.read_parquet(path, columns=columns) for path in paths], ignore_index=True)
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    if frame.candidate_id.duplicated().any():
        raise AssertionError("policy label root contains duplicate candidate identities")
    valid = frame.policy_path_valid.fillna(False).astype(bool)
    if not np.isclose(
        pd.to_numeric(frame.loc[valid, "policy_gross_bps"], errors="coerce")
        - pd.to_numeric(frame.loc[valid, "policy_net_bps"], errors="coerce"),
        100.0, atol=1e-8, rtol=0.0,
    ).all():
        raise AssertionError("rich policy labels must embed exactly one 100-bps cost")
    return frame


def _selected(predictions: Path, *, floor: float, model: str, interaction: str) -> pd.DataFrame:
    frame = pd.read_parquet(predictions)
    selected_column = f"selected__{interaction}"
    required = {"candidate_id", "floor_bps", "model_spec", selected_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"entry predictions lack {sorted(missing)}")
    frame = frame.loc[
        pd.to_numeric(frame.floor_bps, errors="coerce").eq(float(floor))
        & frame.model_spec.eq(model)
        & frame[selected_column].fillna(False).astype(bool)
    ].copy()
    frame["candidate_id"] = frame.candidate_id.astype(str)
    if frame.candidate_id.duplicated().any():
        raise AssertionError("entry OOS predictions select a candidate more than once")
    return frame.loc[:, ["candidate_id"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-panel", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--entry-predictions", type=Path, required=True)
    parser.add_argument("--floor", type=float, default=30.0)
    parser.add_argument("--entry-model", default="lgb_huber_bps")
    parser.add_argument("--entry-interaction", default="veto_pred_ge_0")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    selected = _selected(
        args.entry_predictions.resolve(), floor=args.floor,
        model=args.entry_model, interaction=args.entry_interaction,
    )
    features = pd.read_parquet(
        args.feature_panel.resolve(),
        columns=["candidate_id", "__decision_ts__", "__symbol__", "bcf_mc1_expected_bps", "dual_mc1_min_bps"],
    )
    features["candidate_id"] = features.candidate_id.astype(str)
    features["__decision_ts__"] = pd.to_datetime(features["__decision_ts__"], utc=True, errors="raise")
    if features.candidate_id.duplicated().any():
        raise AssertionError("target-free feature panel contains duplicate candidate identities")
    frame = selected.merge(features, on="candidate_id", how="left", validate="one_to_one")
    frame = frame.merge(_labels(args.labels_root), on="candidate_id", how="left", validate="one_to_one")
    if frame.isna().any(axis=None) and frame[["__decision_ts__", "__symbol__", "bcf_mc1_expected_bps", "policy_entry_price"]].isna().any(axis=None):
        raise AssertionError("selected candidate lacks source-aligned feature or policy label")
    valid = frame.policy_path_valid.fillna(False).astype(bool)
    if not valid.all():
        raise AssertionError("entry prediction selected an invalid policy-label path")
    if not pd.to_numeric(frame.dual_mc1_min_bps, errors="coerce").ge(float(args.floor)).all():
        raise AssertionError("entry head selected below the named dual-MC1 floor")
    exit_bar = pd.to_numeric(frame.policy_exit_bar_15m, errors="raise").astype(int)
    gross_bps = pd.to_numeric(frame.policy_gross_bps, errors="raise")
    candidates = pd.DataFrame({
        "timestamp": frame["__decision_ts__"],
        "candidate_id": frame.candidate_id,
        "symbol": frame["__symbol__"].astype(str),
        "side": "long",
        "strategy_id": "strict_r3_p8u_15m_entry_policycap_long",
        "policy_archetype": str(args.arm),
        "normalized_rank_score": 1.0,
        "strategy_rank_pct": 1.0,
        "base_strategy_threshold": 0.0,
        "calibrated_score": 1.0,
        "portfolio_priority_adjustment": pd.to_numeric(frame.bcf_mc1_expected_bps, errors="raise"),
        "entry_price": pd.to_numeric(frame.policy_entry_price, errors="raise"),
        "exit_timestamp": frame["__decision_ts__"] + pd.to_timedelta((exit_bar + 1) * 15, unit="m"),
        "exit_price": pd.to_numeric(frame.policy_exit_price, errors="raise"),
        "net_return": pd.to_numeric(frame.policy_net_bps, errors="raise") / 10_000.0,
        "gross_return": gross_bps / 10_000.0,
        "holding_bars": exit_bar + 1,
        "simple_policy_exit_reason": frame.policy_exit_reason.astype(str),
        "fees_bps": 100.0,
        "expected_friction_bps": 0.0,
        "price_gap_bps": 0.0,
        "liquidity_capacity_weight": 1.0,
    })
    candidates = normalise_candidate_table(candidates)
    params = portfolio_params()
    decisions, equity, _ = replay_candidates(
        candidates, params, mode="global_auction", ev_curve=CAUSAL_AUCTION_CURVE, market_mode="perp"
    )
    decisions = _attach_ids(decisions, candidates)
    accepted = decisions.loc[decisions.accepted.fillna(False).astype(bool)].copy()
    out.mkdir(parents=True, exist_ok=False)
    candidates.to_parquet(out / "candidates.parquet", index=False, compression="zstd")
    decisions.to_parquet(out / "decisions.parquet", index=False, compression="zstd")
    accepted.to_parquet(out / "accepted.parquet", index=False, compression="zstd")
    equity.to_parquet(out / "equity.parquet", index=False, compression="zstd")
    _period_metrics(accepted, "month").to_parquet(out / "monthly.parquet", index=False)
    _period_metrics(accepted, "day").to_parquet(out / "daily.parquet", index=False)
    metrics = compute_replay_metrics(candidates, decisions, equity, params=params)
    pd.DataFrame([{
        "arm": args.arm, "entry_selected": len(candidates), "portfolio_accepted": len(accepted), **metrics,
    }]).to_parquet(out / "summary.parquet", index=False)
    manifest = {
        "schema": "strict_r3_p8u_15m_entry_policycap_portfolio_v1",
        "scope": "offline strict-OOS only; no live state, exchange IO, or order submission",
        "arm": args.arm,
        "entry": {"floor_bps": args.floor, "model": args.entry_model, "interaction": args.entry_interaction, "authority": "demotion-only"},
        "continuation_head": "held unchanged and excluded: parent-policy-specific state labels cannot be reused across cap changes",
        "priority": "BCF MC1 expected bps only",
        "portfolio": asdict(params),
        "exit": "arm-specific frozen rich policy labels; 15-minute aggregate proxy; 100-bps cost once",
        "source_files": {"features": str(args.feature_panel.resolve()), "labels": str(args.labels_root.resolve()), "entry_predictions": str(args.entry_predictions.resolve())},
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
