#!/usr/bin/env python3
"""Replay Stage-4 native-MC1 finalists through the frozen dual portfolio.

The only candidate differences are the current-native C0/C2 MC1 outputs.
BCF remains separately native and retains the specified +30/+30 admission
and BCF-EV auction authority.  This is a label-complete research replay;
invalid policy paths are excluded before capacity allocation.
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
MC1_ROOT = ROOT / "data_perp/artifacts/strict_r3_stage4_mc1_native_funnel_20260823_v1"
POLICY = ROOT / (
    "data_perp/artifacts/strict_r3_score_family_matched_mc1_canonical_policy_"
    "20260817_v7_bcf/canonical_policy_union.parquet"
)
BCF_SOURCE = ROOT / (
    "data_perp/artifacts/strict_r3_score_family_matched_mc1_canonical_policy_"
    "20260817_v7_bcf/predictions_bcf_mc1_d2.parquet"
)
DEFAULT_OUT = ROOT / "data_perp/artifacts/strict_r3_stage4_dual_portfolio_20260823_v1"

from extreme_price_movements.portfolio_policy_replay import normalise_candidate_table, replay_candidates  # noqa: E402
from scripts.report_strict_r3_mc1_d2_controlled_portfolio import CAUSAL_AUCTION_CURVE, _metrics, _params  # noqa: E402


SPLITS = {
    "development_2025_q1q3": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2025-10-01T00:00:00Z")),
    "holdout_2025_q4": (pd.Timestamp("2025-10-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "portability_2026_janjul": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}


def _policy_candidates(
    shared: pd.DataFrame,
    policy: pd.DataFrame,
    symbols: pd.DataFrame,
    *,
    admission_threshold_bps: float,
) -> pd.DataFrame:
    data = shared.merge(policy, on="candidate_id", how="left", validate="one_to_one")
    data = data.merge(symbols, on="candidate_id", how="left", validate="one_to_one")
    if data["policy_path_valid"].isna().any() or data["__symbol__"].isna().any():
        raise ValueError("dual prediction identity lacks canonical policy or BCF symbol lineage")
    valid = (
        data["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(data["policy_net_bps"], errors="coerce"))
        & np.isfinite(pd.to_numeric(data["policy_gross_bps"], errors="coerce"))
        & np.isfinite(pd.to_numeric(data["policy_exit_bar_15m"], errors="coerce"))
    )
    data = data.loc[valid].copy()
    data["dual_admitted"] = (
        pd.to_numeric(data["current_mc1_expected_bps"], errors="coerce").ge(admission_threshold_bps)
        & pd.to_numeric(data["bcf_mc1_expected_bps"], errors="coerce").ge(admission_threshold_bps)
    )
    data = data.loc[data["dual_admitted"]].copy()
    # BCF expected net is the sole priority authority.  The timestamp-local
    # percentile simply encodes that fixed descending order for the unchanged
    # portfolio engine; it is not a new calibration or ranker.
    data["auction_rank"] = data.groupby("__decision_ts__", sort=False)["bcf_mc1_expected_bps"].rank(
        pct=True, method="first", ascending=True,
    )
    decision = pd.to_datetime(data["__decision_ts__"], utc=True)
    exit_bar = pd.to_numeric(data["policy_exit_bar_15m"], errors="raise").astype(int)
    candidate = pd.DataFrame({
        "timestamp": decision,
        "symbol": data["__symbol__"].astype(str),
        "side": "long",
        "strategy_id": "strict_r3_stage4_dual_long",
        "policy_archetype": "strict_r3_stage4_dual_long",
        "normalized_rank_score": data["auction_rank"].to_numpy(float),
        "strategy_rank_pct": data["auction_rank"].to_numpy(float),
        "base_strategy_threshold": 0.0,
        "calibrated_score": data["auction_rank"].to_numpy(float),
        "entry_price": pd.to_numeric(data["policy_entry_price"], errors="raise"),
        "exit_timestamp": decision + pd.to_timedelta((exit_bar + 1) * 15, unit="min"),
        "exit_price": pd.to_numeric(data["policy_exit_price"], errors="raise"),
        "net_return": pd.to_numeric(data["policy_net_bps"], errors="raise") / 10_000.0,
        "gross_return": pd.to_numeric(data["policy_gross_bps"], errors="raise") / 10_000.0,
        "holding_bars": exit_bar + 1,
        "simple_policy_exit_reason": data["policy_exit_reason"].astype(str),
        "fees_bps": pd.to_numeric(data.get("policy_cost_bps", 100.0), errors="coerce").fillna(100.0),
        "slippage_bps": 0.0,
        "expected_friction_bps": 100.0,
        "price_gap_bps": 0.0,
        "liquidity_capacity_weight": 1.0,
        "source_month": decision.dt.strftime("%Y-%m"),
        "candidate_id": data["candidate_id"].astype(str),
        "mapped_expected_net_bps": pd.to_numeric(data["bcf_mc1_expected_bps"], errors="raise"),
        "policy_outcome_available": True,
    })
    return normalise_candidate_table(candidate)


def _split_metrics(
    decisions: pd.DataFrame,
    equity: pd.DataFrame,
    arm: str,
    *,
    splits: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = SPLITS,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    timestamp = pd.to_datetime(decisions["timestamp"], utc=True) if not decisions.empty else pd.Series(dtype="datetime64[ns, UTC]")
    for name, (start, end) in splits.items():
        selected = decisions.loc[timestamp.ge(start) & timestamp.lt(end)].copy() if not decisions.empty else decisions.copy()
        equity_time = pd.to_datetime(equity["timestamp"], utc=True) if not equity.empty else pd.Series(dtype="datetime64[ns, UTC]")
        equity_selected = equity.loc[equity_time.ge(start) & equity_time.lt(end)].copy() if not equity.empty else equity.copy()
        metric = _metrics(selected, equity_selected, arm, name)
        metric["period_start"] = start
        metric["period_end_exclusive"] = end
        rows.append(metric)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--mc1-root", type=Path, default=MC1_ROOT)
    parser.add_argument(
        "--current-predictions", type=Path,
        help="Optional current-native prediction parquet; permits memory-bounded paired M4 replays.",
    )
    parser.add_argument(
        "--bcf-predictions", type=Path,
        help="Optional separately materialised BCF-native prediction parquet.",
    )
    parser.add_argument("--policy-path", type=Path, default=POLICY,
                        help="Canonical policy ledger. Defaults to the legacy source-aligned parent ledger.")
    parser.add_argument("--symbols-path", type=Path,
                        help="Optional candidate-id/symbol source when the policy ledger has no __symbol__ column.")
    parser.add_argument(
        "--current-arms", default="c0_m0,c2_m0,c2_m1_anchor,c2_m2_agreement",
        help="Comma-separated current-native arms to replay.",
    )
    parser.add_argument("--bcf-arm", default="bcf_m0")
    parser.add_argument("--admission-thresholds", default="30",
                        help="Comma-separated common dual-MC1 floors in bps.")
    parser.add_argument("--start", help="Optional inclusive decision timestamp for a matched replay window.")
    parser.add_argument("--end", help="Optional exclusive decision timestamp for a matched replay window.")
    parser.add_argument("--period-label", default="matched_window",
                        help="Metric label when --start/--end are supplied.")
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    default_predictions = args.mc1_root / "native_mc1_predictions.parquet"
    print(json.dumps({"event": "load_current_predictions"}), flush=True)
    current_predictions = pd.read_parquet(args.current_predictions or default_predictions)
    print(json.dumps({"event": "load_bcf_predictions"}), flush=True)
    bcf_predictions = pd.read_parquet(args.bcf_predictions or default_predictions)
    current_predictions["__decision_ts__"] = pd.to_datetime(current_predictions["__decision_ts__"], utc=True, errors="raise")
    bcf_predictions["__decision_ts__"] = pd.to_datetime(bcf_predictions["__decision_ts__"], utc=True, errors="raise")
    bcf = bcf_predictions.loc[bcf_predictions["family"].eq("bcf")].copy()
    if args.bcf_arm and "arm" in bcf.columns:
        bcf = bcf.loc[bcf["arm"].eq(args.bcf_arm)].copy()
    if bcf.empty:
        raise ValueError(f"missing requested BCF-native arm: {args.bcf_arm}")
    if bcf["candidate_id"].duplicated().any():
        raise AssertionError("BCF-native Stage-4 prediction identities are not unique")
    bcf = bcf.loc[:, ["candidate_id", "mc1_expected_bps"]].rename(columns={"mc1_expected_bps": "bcf_mc1_expected_bps"})
    print(json.dumps({"event": "load_policy_and_symbols"}), flush=True)
    policy = pd.read_parquet(args.policy_path)
    policy["__decision_ts__"] = pd.to_datetime(policy["__decision_ts__"], utc=True, errors="raise")
    # The current target-free score establishes the decision identity.  The
    # policy union contributes outcomes only; retaining its duplicate score
    # timestamp/final-score columns would obscure the causal score lineage.
    if "__symbol__" in policy.columns:
        symbols = policy.loc[:, ["candidate_id", "__symbol__"]].copy()
    elif args.symbols_path:
        symbols = pd.read_parquet(args.symbols_path, columns=["candidate_id", "__symbol__"])
    else:
        symbols = pd.read_parquet(BCF_SOURCE, columns=["candidate_id", "__symbol__"])
    policy = policy.drop(columns=["__decision_ts__", "final_score", "__symbol__"], errors="ignore")
    if policy["candidate_id"].duplicated().any() or symbols["candidate_id"].duplicated().any():
        raise AssertionError("canonical BCF policy or symbol identities are not unique")
    rows: list[dict[str, object]] = []
    counts: list[dict[str, object]] = []
    requested_arms = tuple(arm.strip() for arm in args.current_arms.split(",") if arm.strip())
    if not requested_arms:
        raise ValueError("at least one current-native arm is required")
    thresholds = tuple(float(value) for value in args.admission_thresholds.split(",") if value.strip())
    if not thresholds or any(value < 0.0 for value in thresholds):
        raise ValueError("admission thresholds must be nonnegative numbers")
    window_start = pd.Timestamp(args.start, tz="UTC") if args.start else None
    window_end = pd.Timestamp(args.end, tz="UTC") if args.end else None
    splits = (
        {args.period_label: (window_start, window_end)}
        if window_start is not None and window_end is not None else SPLITS
    )
    if (window_start is None) != (window_end is None):
        raise ValueError("--start and --end must be supplied together")
    for arm in requested_arms:
        print(json.dumps({"event": "prepare_arm", "arm": arm}), flush=True)
        current = current_predictions.loc[current_predictions["family"].isin(("current", "current_v5"))].copy()
        if "arm" in current.columns:
            current = current.loc[current["arm"].eq(arm)].copy()
        if current["candidate_id"].duplicated().any():
            raise AssertionError(f"{arm} current-native identities are not unique")
        current = current.loc[:, ["candidate_id", "__decision_ts__", "mc1_expected_bps"]].rename(
            columns={"mc1_expected_bps": "current_mc1_expected_bps"},
        )
        shared = current.merge(bcf, on="candidate_id", how="inner", validate="one_to_one")
        if window_start is not None:
            shared = shared.loc[shared["__decision_ts__"].ge(window_start) & shared["__decision_ts__"].lt(window_end)].copy()
        for threshold in thresholds:
            label = f"{arm}_t{threshold:g}"
            candidate = _policy_candidates(shared, policy, symbols, admission_threshold_bps=threshold)
            print(json.dumps({"event": "replay_arm", "arm": label, "admitted": int(len(candidate))}), flush=True)
            decisions, equity, _ = replay_candidates(
                candidate, _params(), mode="global_auction", ev_curve=CAUSAL_AUCTION_CURVE,
                market_mode="perps", initial_wallet=1000.0,
            )
            if not decisions.empty:
                provenance = candidate.loc[:, ["candidate_id", "policy_outcome_available"]].reset_index(drop=True)
                provenance.index.name = "candidate_index"
                decisions = decisions.merge(provenance, on="candidate_index", how="left", validate="many_to_one")
            decisions.to_parquet(args.out_dir / f"{label}_decisions.parquet", index=False, compression="zstd")
            equity.to_parquet(args.out_dir / f"{label}_equity.parquet", index=False, compression="zstd")
            rows.extend(_split_metrics(decisions, equity, label, splits=splits))
            counts.append({
                "arm": label, "shared_native_rows": int(len(shared)), "dual_admitted_valid_rows": int(len(candidate)),
                "portfolio_accepted_rows": int(decisions["accepted"].fillna(False).sum()) if not decisions.empty else 0,
                "admission_threshold_bps": float(threshold),
            })
            print(json.dumps({"event": "stage4_dual_portfolio_complete", "arm": label, "accepted": counts[-1]["portfolio_accepted_rows"]}), flush=True)
    pd.DataFrame(rows).to_parquet(args.out_dir / "portfolio_metrics.parquet", index=False)
    pd.DataFrame(counts).to_parquet(args.out_dir / "admission_counts.parquet", index=False)
    manifest = {
        "schema": "strict_r3_stage4_dual_portfolio_v1",
        "scope": "offline long-only research; no live/canonical/execution artifact modified",
        "admission": "current-native MC1 and separately native BCF MC1 clear the same requested threshold",
        "priority": "BCF-native MC1 expected net descending within timestamp",
        "portfolio": "8 concurrent, 2 entries per timestamp, 1 per asset, 80% margin cap, 10% wallet margin slots, 7x; frozen rich parent policy",
        "outcome": "canonical source-aligned frozen rich parent policy net; invalid paths excluded before capacity",
        "mc1_root": str(args.mc1_root),
        "current_arms": list(requested_arms),
        "bcf_arm": args.bcf_arm,
        "current_predictions": str(args.current_predictions or default_predictions),
        "bcf_predictions": str(args.bcf_predictions or default_predictions),
        "policy_path": str(args.policy_path), "symbols_path": str(args.symbols_path) if args.symbols_path else None,
        "admission_thresholds_bps": list(thresholds),
        "matched_window": [str(window_start), str(window_end)] if window_start is not None else None,
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
