#!/usr/bin/env python3
"""Compare hourly and quarter-hour decision cadence without changing Strict-R3.

The frozen model contract is hourly.  This is consequently a *carry-forward
cadence* experiment: score/model features at a completed UTC hour are frozen,
then the identical score is actionable at :00, :15, :30 and :45.  It never
injects partially completed hour or 15-minute values into the hourly models.

Both arms retain the immutable common BCF/current-v5 historical route:
both MC1 maps must be at least +30 bps and BCF MC1 expected bps is the sole
auction priority.  Outcomes are joined only after that target-free route.
The execution outcome is the current frozen rich SimplePolicyOptimiser parent
on complete Kraken Futures one-minute bars, entered five minutes after each
decision.  Adaptive Exit V1 is intentionally not replayed here because no
historical per-decision V1 inference artifact exists; it is held constant out
of scope for both arms rather than approximated from future outcomes.
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

from extreme_price_movements.data_store import PartitionedOHLCVStore, canonical_kraken_execution_1m_root
from extreme_price_movements.portfolio_policy_replay import normalise_candidate_table, replay_candidates
from extreme_price_movements.strict_r3_rich_policy import RichPolicyParams, simulate_rich_policy
from scripts.replay_strict_r3_bcf_exact5m_1m import _minute_aggregated_atr, _one_minute_paths
from scripts.report_strict_r3_mc1_d2_controlled_portfolio import CAUSAL_AUCTION_CURVE, _params


HORIZON_MINUTES = 12 * 60
COST_BPS = 100.0
DEFAULT_BCF = ROOT / "data_perp/artifacts/strict_r3_score_family_matched_mc1_canonical_policy_20260817_v7_bcf/predictions_bcf_mc1_d2.parquet"
DEFAULT_CURRENT = ROOT / "data_perp/artifacts/strict_r3_score_family_matched_mc1_canonical_policy_20260817_v7_current/predictions_current_v5_mc1_d2.parquet"
DEFAULT_POLICY = ROOT / "data_perp/artifacts/strict_r3_rich_policy_smooth_protection_long_20260817_v1/frozen_policy.json"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    return value


def _read(path: Path, *, family: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    columns = [
        "candidate_id", "__decision_ts__", "__symbol__", "side_name", "final_score",
        "base_rank42", "conditional_consensus_rank", "ordinary_shadow_consensus_rank",
        "correctness_rank", "mc1_expected_bps",
    ]
    frame = pd.read_parquet(path, columns=columns)
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True)
    frame = frame.loc[
        frame["__decision_ts__"].ge(start) & frame["__decision_ts__"].lt(end)
        & frame["side_name"].astype(str).str.lower().eq("long")
    ].copy()
    if frame["candidate_id"].duplicated().any():
        raise AssertionError(f"{family} source has duplicate identities")
    frame = frame.rename(columns={
        "final_score": f"{family}_final_score",
        "base_rank42": f"{family}_base_rank42",
        "conditional_consensus_rank": f"{family}_conditional_consensus_rank",
        "ordinary_shadow_consensus_rank": f"{family}_ordinary_shadow_consensus_rank",
        "correctness_rank": f"{family}_correctness_rank",
        "mc1_expected_bps": f"{family}_mc1_expected_bps",
    })
    return frame


def _target_free_route(bcf: pd.DataFrame, current: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    merged = bcf.merge(
        current, on="candidate_id", how="inner", validate="one_to_one", suffixes=("_bcf", "_current"),
    )
    for field in ("__decision_ts__", "__symbol__", "side_name"):
        left, right = f"{field}_bcf", f"{field}_current"
        if not merged[left].eq(merged[right]).all():
            raise AssertionError(f"BCF/current common score identity differs for {field}")
    merged = merged.rename(columns={
        "__decision_ts___bcf": "__decision_ts__",
        "__symbol___bcf": "__symbol__",
        "side_name_bcf": "side_name",
    }).drop(columns=["__decision_ts___current", "__symbol___current", "side_name_current"])
    bcf_ev = pd.to_numeric(merged["bcf_mc1_expected_bps"], errors="raise")
    current_ev = pd.to_numeric(merged["current_mc1_expected_bps"], errors="raise")
    routed = merged.loc[bcf_ev.ge(30.0) & current_ev.ge(30.0)].copy()
    routed["source_decision_ts"] = routed["__decision_ts__"]
    routed["priority_bps"] = bcf_ev.loc[routed.index].to_numpy(float)
    routed["mapped_expected_net_bps"] = routed["priority_bps"]
    return routed, {
        "bcf_rows": int(len(bcf)), "current_rows": int(len(current)),
        "common_score_rows": int(len(merged)), "dual_admitted_source_rows": int(len(routed)),
    }


def _expand(routed: pd.DataFrame, *, offsets: list[int]) -> pd.DataFrame:
    copies: list[pd.DataFrame] = []
    for minute in offsets:
        work = routed.copy()
        work["feature_age_minutes"] = int(minute)
        work["__decision_ts__"] = work["source_decision_ts"] + pd.Timedelta(minutes=int(minute))
        work["candidate_id"] = (
            work["candidate_id"].astype(str)
            + "|cadence=" + work["__decision_ts__"].dt.strftime("%Y%m%dT%H%M%SZ")
        )
        work["entry_ts"] = work["__decision_ts__"] + pd.Timedelta(minutes=5)
        copies.append(work)
    output = pd.concat(copies, ignore_index=True)
    if output["candidate_id"].duplicated().any():
        raise AssertionError("cadence expansion created duplicate identities")
    return output.sort_values(["__decision_ts__", "priority_bps", "candidate_id"], ascending=[True, False, True], kind="stable").reset_index(drop=True)


def _empty_labels(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.loc[:, ["candidate_id", "__decision_ts__", "source_decision_ts", "__symbol__", "entry_ts"]].copy()
    out["policy_path_valid"] = False
    out["policy_gross_bps"] = np.nan
    out["policy_net_bps"] = np.nan
    out["policy_entry_price"] = np.nan
    out["policy_exit_price"] = np.nan
    out["policy_exit_minutes"] = np.nan
    out["policy_exit_timestamp"] = pd.NaT
    out["policy_exit_reason"] = "invalid_exact_1m_path"
    out["policy_atr"] = np.nan
    out["policy_label_available_ts"] = out["entry_ts"] + pd.Timedelta(hours=12)
    return out


def _labels_for_symbol(
    rows: pd.DataFrame,
    *,
    minute_store: PartitionedOHLCVStore,
    params: RichPolicyParams,
    median_atr_fraction: float,
) -> pd.DataFrame:
    out = _empty_labels(rows).reset_index(drop=True)
    symbol = str(rows["__symbol__"].iloc[0])
    source_decisions = pd.to_datetime(rows["source_decision_ts"], utc=True).reset_index(drop=True)
    entries = pd.to_datetime(rows["entry_ts"], utc=True).reset_index(drop=True)
    start = min(entries.min(), source_decisions.min() - pd.Timedelta(hours=100))
    end = entries.max() + pd.Timedelta(minutes=HORIZON_MINUTES)
    minute = minute_store.load(symbol, columns=["ts", "open", "high", "low", "close"], start_ts=start, end_ts=end)
    opens, highs, lows, closes, path_ok = _one_minute_paths(minute, entries)
    atr_series = _minute_aggregated_atr(minute_store, symbol, decisions=source_decisions)
    atr_values = source_decisions.map(atr_series).to_numpy(dtype=np.float64)
    entry = opens[:, 0].astype(np.float64)
    valid = path_ok & np.isfinite(atr_values) & (atr_values > 0.0) & np.isfinite(entry) & (entry > 0.0)
    if not valid.any():
        return out
    positions = np.flatnonzero(valid)
    result = simulate_rich_policy(
        entry=entry[positions], atr=atr_values[positions], highs=highs[positions], lows=lows[positions], closes=closes[positions],
        params=params, median_atr_fraction=median_atr_fraction,
    )
    realised = np.asarray(result["path_valid"], dtype=bool) & np.isfinite(np.asarray(result["net_bps"], dtype=float))
    loc = positions[realised]
    if len(loc):
        exit_bar = np.asarray(result["exit_bar"], dtype=np.int64)[realised]
        duration = exit_bar + 1
        out.loc[loc, "policy_path_valid"] = True
        out.loc[loc, "policy_gross_bps"] = np.asarray(result["gross_bps"], dtype=float)[realised]
        out.loc[loc, "policy_net_bps"] = np.asarray(result["net_bps"], dtype=float)[realised]
        out.loc[loc, "policy_entry_price"] = entry[loc]
        # Barriers/timeout use the modelled price returned by the rich policy;
        # infer it from realised gross rather than invent an intrabar fill.
        out.loc[loc, "policy_exit_price"] = entry[loc] * (1.0 + out.loc[loc, "policy_gross_bps"].to_numpy(float) / 10_000.0)
        out.loc[loc, "policy_exit_minutes"] = duration
        out.loc[loc, "policy_exit_timestamp"] = entries.iloc[loc].to_numpy() + pd.to_timedelta(duration, unit="min")
        out.loc[loc, "policy_exit_reason"] = np.asarray(result["exit_reason"], dtype=object)[realised]
        out.loc[loc, "policy_atr"] = atr_values[loc]
    valid_rows = out["policy_path_valid"].to_numpy(bool)
    if valid_rows.any() and not np.allclose(
        out.loc[valid_rows, "policy_net_bps"].to_numpy(float),
        out.loc[valid_rows, "policy_gross_bps"].to_numpy(float) - COST_BPS,
        rtol=0.0, atol=1e-9,
    ):
        raise AssertionError("policy cost must be charged exactly once")
    return out


def _materialize_labels(
    candidates: pd.DataFrame, *, data_root: Path, params: RichPolicyParams, median_atr_fraction: float,
) -> pd.DataFrame:
    store = PartitionedOHLCVStore(str(canonical_kraken_execution_1m_root(data_root)), timeframe="1m")
    pieces: list[pd.DataFrame] = []
    total = candidates["__symbol__"].nunique()
    for number, (_, group) in enumerate(candidates.groupby("__symbol__", sort=True), start=1):
        pieces.append(_labels_for_symbol(group.reset_index(drop=True), minute_store=store, params=params, median_atr_fraction=median_atr_fraction))
        if number == 1 or number % 20 == 0 or number == total:
            print(json.dumps({"event": "label_progress", "symbols_complete": number, "symbols_total": total}), flush=True)
    labels = pd.concat(pieces, ignore_index=True).sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    if len(labels) != len(candidates) or labels["candidate_id"].duplicated().any():
        raise AssertionError("exact label materialisation altered target-free identity")
    return labels


def _portfolio_candidates(source: pd.DataFrame, labels: pd.DataFrame, *, arm: str) -> pd.DataFrame:
    frame = source.merge(labels, on=["candidate_id", "__decision_ts__", "source_decision_ts", "__symbol__", "entry_ts"], how="inner", validate="one_to_one")
    valid = frame["policy_path_valid"].fillna(False).astype(bool) & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce"))
    admitted = frame.loc[valid].copy()
    if admitted.empty:
        return pd.DataFrame()
    admitted["auction_rank"] = admitted.groupby("entry_ts", sort=False)["priority_bps"].rank(pct=True, method="average")
    candidate = pd.DataFrame({
        "timestamp": pd.to_datetime(admitted["entry_ts"], utc=True),
        "symbol": admitted["__symbol__"].astype(str), "side": "long",
        "strategy_id": f"strict_r3_hourly_feature_{arm}", "policy_archetype": f"strict_r3_hourly_feature_{arm}",
        "normalized_rank_score": admitted["auction_rank"].to_numpy(float), "strategy_rank_pct": admitted["auction_rank"].to_numpy(float),
        "base_strategy_threshold": 0.0, "calibrated_score": admitted["priority_bps"].to_numpy(float),
        "entry_price": pd.to_numeric(admitted["policy_entry_price"], errors="raise"),
        "exit_timestamp": pd.to_datetime(admitted["policy_exit_timestamp"], utc=True),
        "exit_price": pd.to_numeric(admitted["policy_exit_price"], errors="raise"),
        "net_return": pd.to_numeric(admitted["policy_net_bps"], errors="raise") / 10_000.0,
        "gross_return": pd.to_numeric(admitted["policy_gross_bps"], errors="raise") / 10_000.0,
        "holding_bars": pd.to_numeric(admitted["policy_exit_minutes"], errors="raise"),
        "simple_policy_exit_reason": admitted["policy_exit_reason"].astype(str),
        "fees_bps": COST_BPS, "slippage_bps": 0.0, "expected_friction_bps": COST_BPS, "price_gap_bps": 0.0,
        "liquidity_capacity_weight": 1.0, "candidate_id": admitted["candidate_id"].astype(str),
        "mapped_expected_net_bps": admitted["mapped_expected_net_bps"].to_numpy(float),
        "bcf_mc1_expected_bps": admitted["bcf_mc1_expected_bps"].to_numpy(float),
        "current_v5_mc1_expected_bps": admitted["current_mc1_expected_bps"].to_numpy(float),
        "source_decision_ts": pd.to_datetime(admitted["source_decision_ts"], utc=True),
        "feature_age_minutes": admitted["feature_age_minutes"].to_numpy(int),
        "policy_outcome_available": True,
    })
    return normalise_candidate_table(candidate)


def _metrics(decisions: pd.DataFrame, *, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    accepted = decisions.loc[decisions["accepted"].fillna(False).astype(bool)].copy()
    days = max((end - start).total_seconds() / 86_400.0, 1.0)
    if accepted.empty:
        return {"entries": 0, "trades_per_day": 0.0, "net_ev_bps_per_trade": np.nan, "total_net_bps": 0.0}
    net_bps = accepted["position_net_return"].to_numpy(float) * 10_000.0
    return {
        "entries": int(len(accepted)), "trades_per_day": float(len(accepted) / days),
        "net_ev_bps_per_trade": float(np.mean(net_bps)), "total_net_bps": float(np.sum(net_bps)),
        "worst_trade_bps": float(np.min(net_bps)), "median_net_bps": float(np.median(net_bps)),
    }


def _periods(decisions: pd.DataFrame, *, frequency: str) -> pd.DataFrame:
    accepted = decisions.loc[decisions["accepted"].fillna(False).astype(bool)].copy()
    if accepted.empty:
        return pd.DataFrame(columns=["period", "trades", "net_ev_bps_per_trade", "total_net_bps"])
    accepted["period"] = pd.to_datetime(accepted["timestamp"], utc=True).dt.to_period(frequency).astype(str)
    return accepted.groupby("period", as_index=False).agg(
        trades=("accepted", "size"),
        net_ev_bps_per_trade=("position_net_return", lambda value: float(np.mean(value) * 10_000.0)),
        total_net_bps=("position_net_return", lambda value: float(np.sum(value) * 10_000.0)),
        symbols=("symbol", "nunique"),
    )


def _run_arm(name: str, source: pd.DataFrame, labels: pd.DataFrame, *, start: pd.Timestamp, end: pd.Timestamp, out: Path) -> dict[str, Any]:
    candidates = _portfolio_candidates(source, labels, arm=name)
    decisions, equity, _ = replay_candidates(candidates, _params(), mode="global_auction", ev_curve=CAUSAL_AUCTION_CURVE, market_mode="perps", initial_wallet=1000.0)
    candidates.to_parquet(out / f"{name}_portfolio_candidates.parquet", index=False, compression="zstd")
    decisions.to_parquet(out / f"{name}_portfolio_decisions.parquet", index=False, compression="zstd")
    equity.to_parquet(out / f"{name}_portfolio_equity.parquet", index=False, compression="zstd")
    _periods(decisions, frequency="M").to_parquet(out / f"{name}_monthly_metrics.parquet", index=False)
    _periods(decisions, frequency="W-MON").to_parquet(out / f"{name}_weekly_metrics.parquet", index=False)
    return {
        "arm": name, "routed_target_free_candidates": int(len(source)),
        "valid_exact_1m_outcomes": int(labels["policy_path_valid"].sum()),
        "invalid_exact_1m_outcomes_excluded_after_route": int((~labels["policy_path_valid"].astype(bool)).sum()),
        **_metrics(decisions, start=start, end=end),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2026-05-01T00:00:00Z")
    parser.add_argument("--end", default="2026-08-01T00:00:00Z")
    parser.add_argument("--bcf", type=Path, default=DEFAULT_BCF)
    parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data_perp")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    if start.minute or end.minute or start.second or end.second or end <= start:
        raise ValueError("start/end must be ordered whole UTC hours")
    payload = json.loads(args.policy.read_text(encoding="utf-8"))
    params = RichPolicyParams.from_mapping(payload["params"])
    median = float(payload["median_atr_fraction_fitted_on_complete_2024_development"])
    if float(payload["cost_bps"]) != COST_BPS:
        raise AssertionError("policy cost contract changed")
    bcf, current = _read(args.bcf, family="bcf", start=start, end=end), _read(args.current, family="current", start=start, end=end)
    routed, route_audit = _target_free_route(bcf, current)
    arms = {
        "hourly_control": _expand(routed, offsets=[0]),
        "quarter_hour_carry": _expand(routed, offsets=[0, 15, 30, 45]),
    }
    out.mkdir(parents=True)
    # Materialise the broad arm first.  Its :00 candidate identities are
    # identical to the control's identities, so the control reuses those exact
    # one-minute outcomes instead of reading the same historical panels twice.
    for name, source in arms.items():
        source.to_parquet(out / f"{name}_target_free_candidates.parquet", index=False, compression="zstd")
    quarter_labels = _materialize_labels(
        arms["quarter_hour_carry"], data_root=args.data_root, params=params, median_atr_fraction=median,
    )
    control_ids = set(arms["hourly_control"]["candidate_id"].astype(str))
    control_labels = quarter_labels.loc[quarter_labels["candidate_id"].astype(str).isin(control_ids)].copy()
    if len(control_labels) != len(arms["hourly_control"]):
        raise AssertionError("quarter-hour outcome materialisation omitted a :00 control identity")
    label_frames = {"hourly_control": control_labels, "quarter_hour_carry": quarter_labels}
    results: list[dict[str, Any]] = []
    for name, source in arms.items():
        labels = label_frames[name]
        labels.to_parquet(out / f"{name}_exact_1m_policy_labels.parquet", index=False, compression="zstd")
        results.append(_run_arm(name, source, labels, start=start, end=end, out=out))
    metrics = pd.DataFrame(results)
    control = metrics.loc[metrics["arm"].eq("hourly_control")].iloc[0]
    for field in ("entries", "trades_per_day", "net_ev_bps_per_trade", "total_net_bps"):
        metrics[f"delta_vs_hourly_{field}"] = metrics[field] - control[field]
    metrics.to_parquet(out / "summary_metrics.parquet", index=False)
    manifest = {
        "schema": "strict_r3_hourly_feature_quarter_cadence_replay_v1",
        "purpose": "research-only cadence comparison; no live configuration or model artifact changed",
        "period": {"start": start, "end": end},
        "score_contract": "frozen BCF/current-v5 common historical score IDs; both MC1 >= +30 bps; BCF MC1 priority",
        "feature_cadence": "hourly snapshot reused at +0/+15/+30/+45 minutes; model features/outputs are not recomputed intrahour",
        "entry": "complete Kraken Futures 1m open at decision +5 minutes",
        "outcome": "720 complete post-entry 1m bars; missing paths excluded only after target-free routing",
        "policy": {"path": str(args.policy), "sha256": _sha(args.policy), "cost_bps_once": COST_BPS, "adaptive_exit_v1": "not replayed; no historical causal V1 artifact"},
        "portfolio": "unchanged existing long-only global auction (_params, CAUSAL_AUCTION_CURVE)",
        "sources": {"bcf": {"path": str(args.bcf), "sha256": _sha(args.bcf)}, "current": {"path": str(args.current), "sha256": _sha(args.current)}},
        "route_audit": route_audit, "results": results,
    }
    (out / "run_manifest.json").write_text(json.dumps(_json(manifest), indent=2, sort_keys=True) + "\n")
    print(json.dumps(_json({"event": "complete", "out_dir": str(out), "results": results}), indent=2))


if __name__ == "__main__":
    main()
