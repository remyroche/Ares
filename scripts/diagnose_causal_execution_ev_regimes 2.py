#!/usr/bin/env python3
"""Strict weekly forward diagnostic for causal execution-EV state inputs.

The fitted state is based only on contemporaneously available alpha, peak,
CatBoost and base-score geometry.  It is intentionally not a regime expert,
not calendar-labelled, and does not weight training examples by economics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.causal_execution_regimes import (  # noqa: E402
    STATE_SCHEMA,
    CausalRegimeStateModel,
    add_regime_transition_labels,
)

OLD = ROOT / "data_perp/artifacts/execution_ev_repaired_heads_representation_handoff_20260726_v7/joined.parquet"
FORWARD = ROOT / "data_perp/artifacts/execution_ev_context_head_clean_forward_july19_20260726_v2/strict_forward_winner_inputs_and_raw_scores.parquet"
OUTPUT = ROOT / "data_perp/artifacts/causal_execution_ev_regime_diagnostic_may_july19_20260726_v2"
ID = ["__ts__", "__symbol__", "side_name", "candidate_id"]
STATE_FEATURES = [
    "existing_alpha_ev", "alpha_prediction_uncertainty", "alpha_leaf_support",
    "pred_peak_MFE_12h_ATR", "catboost_entropy", "base_oof_score",
    "base_margin_to_cutoff", "base_margin_to_cutoff_z", "oof_clean_favorable_probability",
    *[f"catboost_p_{i}" for i in range(7)],
]
HEAD_AVAILABILITY = {
    "existing_alpha_ev": "alpha_available_at",
    "alpha_prediction_uncertainty": "alpha_available_at",
    "alpha_leaf_support": "alpha_available_at",
    "pred_peak_MFE_12h_ATR": "peak_mfe_available_at",
    **{f"catboost_p_{i}": "catboost_available_at" for i in range(7)},
    "catboost_entropy": "catboost_available_at",
}


def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    out = pd.read_parquet(path, columns=columns)
    for col in ["__ts__", "execution_decision_utc", "execution_label_end_utc", *set(HEAD_AVAILABILITY.values())]:
        if col in out:
            out[col] = pd.to_datetime(out[col], utc=True, errors="raise")
    return out


def _state_availability_mask(frame: pd.DataFrame) -> pd.Series:
    decision = pd.to_datetime(frame["execution_decision_utc"], utc=True, errors="raise")
    valid = np.isfinite(frame.loc[:, STATE_FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)).all(axis=1)
    for stamp in sorted(set(HEAD_AVAILABILITY.values())):
        value = pd.to_datetime(frame[stamp], utc=True, errors="coerce")
        valid &= value.notna().to_numpy() & (value <= decision).to_numpy()
    return pd.Series(valid, index=frame.index)


def _state_rows(frame: pd.DataFrame, *, week: pd.Timestamp, side: str, k: int) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    grouped = frame.groupby("causal_regime_state", observed=True).agg(
        rows=("candidate_id", "size"),
        mean_net_ev=("execution_net_ev_12h", "mean"),
        positive_net_ev_rate=("execution_net_ev_12h", lambda s: float((s > 0).mean())),
        mean_change_within_6h=("causal_regime_change_within_6h", "mean"),
        resolved_transition_rows=("causal_regime_change_within_6h", "count"),
        mean_ood_z=("causal_regime_ood_z", "mean"),
        mean_entropy=("causal_regime_entropy", "mean"),
    ).reindex(range(k), fill_value=0).reset_index()
    grouped["week_start"] = week; grouped["side_name"] = side
    return grouped


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    needed = list(dict.fromkeys([*ID, "execution_decision_utc", "execution_label_end_utc", "execution_net_ev_12h", *STATE_FEATURES, *HEAD_AVAILABILITY.values()]))
    old = _read(args.old, needed)
    forward = _read(args.forward, needed)
    data = pd.concat([old, forward], ignore_index=True).drop_duplicates(ID, keep="last")
    data = data.loc[data["execution_decision_utc"].ge(pd.Timestamp(args.start, tz="UTC"))].copy()
    data["state_input_available_by_decision"] = _state_availability_mask(data)
    data = data.loc[data["state_input_available_by_decision"]].sort_values("execution_decision_utc").reset_index(drop=True)
    observed_through = data["execution_decision_utc"].max()
    week = pd.Timestamp(args.first_eval, tz="UTC")
    summaries: list[dict[str, object]] = []; state_summaries: list[pd.DataFrame] = []; output_rows: list[pd.DataFrame] = []
    while week <= observed_through:
        end = min(week + pd.Timedelta(days=7), observed_through + pd.Timedelta(nanoseconds=1))
        evaluation = data.loc[(data["execution_decision_utc"] >= week) & (data["execution_decision_utc"] < end)].copy()
        if evaluation.empty:
            week += pd.Timedelta(days=7); continue
        for side in ("long", "short"):
            training = data.loc[(data["execution_decision_utc"] < week) & data["side_name"].eq(side)].copy()
            current = evaluation.loc[evaluation["side_name"].eq(side)].copy()
            if len(training) < args.min_train_rows or current.empty:
                continue
            state_model = CausalRegimeStateModel.fit(training, STATE_FEATURES)
            transformed_train = state_model.transform(training)
            transformed_eval = state_model.transform(current)
            # The 6h label buffer is transformed with the same frozen weekly
            # model.  It is used only to resolve labels for final-week rows;
            # buffered rows are never emitted as that week's feature rows.
            label_end = min(end + pd.Timedelta(hours=6), observed_through + pd.Timedelta(nanoseconds=1))
            label_context = data.loc[(data["execution_decision_utc"] >= week) & (data["execution_decision_utc"] < label_end) & data["side_name"].eq(side)].copy()
            label_context = pd.concat([label_context.reset_index(drop=True), state_model.transform(label_context).reset_index(drop=True)], axis=1)
            # Supporting labels only become known at decision+6h.  The trainer
            # does not receive them; they are emitted for later label modelling.
            label_context = add_regime_transition_labels(label_context, observed_through=min(observed_through, end + pd.Timedelta(hours=6)), time_column="execution_decision_utc")
            current = label_context.loc[label_context["execution_decision_utc"].lt(end)].copy()
            drift = state_model.training_drift(transformed_train, transformed_eval)
            summaries.append({
                "week_start": week.isoformat(), "week_end_exclusive": end.isoformat(), "side_name": side,
                "train_rows": int(len(training)), "evaluation_rows": int(len(current)), "selected_k": state_model.selected_k,
                "selection": state_model.selection, "mean_net_ev": float(current["execution_net_ev_12h"].mean()),
                "positive_net_ev_rate": float((current["execution_net_ev_12h"] > 0).mean()),
                "resolved_transition_rows": int(current["causal_regime_change_within_6h"].notna().sum()),
                "transition_change_rate": float(current["causal_regime_change_within_6h"].mean()), **drift,
            })
            state_summaries.append(_state_rows(current, week=week, side=side, k=state_model.selected_k))
            current["regime_week_start"] = week; current["regime_week_end_exclusive"] = end; current["regime_fit_cutoff_utc"] = week
            current["regime_schema"] = STATE_SCHEMA; current["regime_feature_count"] = len(STATE_FEATURES)
            output_rows.append(current)
        week += pd.Timedelta(days=7)
    all_rows = pd.concat(output_rows, ignore_index=True)
    all_states = pd.concat(state_summaries, ignore_index=True)
    args.output_dir.mkdir(parents=True)
    all_rows.to_parquet(args.output_dir / "weekly_forward_regime_state_rows.parquet", index=False)
    all_states.to_parquet(args.output_dir / "weekly_forward_state_economics_and_transitions.parquet", index=False)
    contract = {
        "schema": STATE_SCHEMA,
        "state_features": STATE_FEATURES,
        "availability_contract": {
            "head_features": HEAD_AVAILABILITY,
            "base_and_clean_inputs": "materialized candidate inputs with no separate availability field; source handoff contract asserts they were known by execution_decision_utc",
            "market_volatility_liquidity": "not included because no common decision-time fields exist in both expanded old and strict-forward handoffs",
        },
        "training_rule": "per side, every weekly model fits only rows whose decision timestamp is before that week; no execution outcome, calendar label, or sample weighting is used",
        "supporting_labels": {
            "causal_regime_change_within_6h": "whether any later candidate for same symbol/side receives another causal state before decision+6h; null until resolution",
            "causal_regime_persistence_6h": "1 - causal_regime_change_within_6h; null until resolution",
            "causal_regime_change_6h_label_resolution_utc": "decision timestamp + 6h",
        },
        "downstream_input_rule": "causal_regime_state is a categorical diagnostic ID, never numeric/ordinal; all transition/persistence columns are labels only and never same-row features. posterior_* coordinates are fold-local because K and centroid ordering may change, hence permitted only inside one frozen-fit ablation. Across weekly refits, use only permutation-invariant stable geometry: entropy, top2 margin, nearest_distance2, distance_percentile, and distance_exceedance. OOD z remains diagnostic only because a tiny train MAD can make it numerically large.",
        "state_selection": "predeclared K=3..5 selection by seed-replicated assignment ARI plus minimum training occupancy only",
    }
    payload = {"contract": contract, "observed_through": observed_through.isoformat(), "weekly_summaries": summaries, "outputs": {"rows": str(args.output_dir / "weekly_forward_regime_state_rows.parquet"), "state_summaries": str(args.output_dir / "weekly_forward_state_economics_and_transitions.parquet")}}
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    (args.output_dir / "state_feature_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, default=OLD); parser.add_argument("--forward", type=Path, default=FORWARD)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--start", default="2026-05-01"); parser.add_argument("--first-eval", default="2026-06-01")
    parser.add_argument("--min-train-rows", type=int, default=500)
    return parser


if __name__ == "__main__":
    print(json.dumps(run(_parser().parse_args()), indent=2, default=str))
