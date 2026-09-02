#!/usr/bin/env python3
"""Offline portfolio replay for the source-aligned P8U 15-minute C1 challenger.

This is deliberately a second-stage research utility.  It consumes only
already materialised, strict-OOS entry outcomes, applies the existing BCF-MC1
priority and fixed chronological portfolio contract separately to each arm,
and never scores, fetches data, or communicates with an exchange.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
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
from scripts import run_strict_r3_p8u_15m_continuation_walkforward as continuation
from scripts.report_strict_r3_mc1_d2_controlled_portfolio import (
    CAUSAL_AUCTION_CURVE,
    _params as canonical_portfolio_params,
)


DEFAULT_STATE_ROOT = (
    ROOT
    / "data_perp/artifacts/strict_r3_p8u_15m_continuation_predictive_observed25h_20260830_v3"
    / "target_free_continuation_state_parts"
)
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_continuation_portfolio_20260830_v1"
REQUIRED_ARMS = ("C0_parent", "C1_activation_only")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_inputs(patterns: list[str]) -> tuple[pd.DataFrame, list[Path]]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(sorted(ROOT.glob(pattern)))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError("no entry_outcomes.parquet paths matched --input-glob")
    frames = [pd.read_parquet(path) for path in paths]
    rows = pd.concat(frames, ignore_index=True)
    required = {
        "candidate_id", "__symbol__", "entry_decision_ts", "arm",
        "mc1_threshold_bps", "c1_gross_bps", "c1_net_bps", "c1_exit_bar",
        "c1_exit_reason",
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"entry outcomes lack {missing}")
    rows["entry_decision_ts"] = pd.to_datetime(rows["entry_decision_ts"], utc=True, errors="raise")
    rows["candidate_id"] = rows["candidate_id"].astype(str)
    return rows, paths


def _entry_prices(state_root: Path) -> pd.DataFrame:
    states = continuation._read_state_panel(state_root)
    needed = states.loc[:, ["candidate_id", "entry_price"]].copy()
    needed["candidate_id"] = needed["candidate_id"].astype(str)
    needed["entry_price"] = pd.to_numeric(needed["entry_price"], errors="coerce")
    if needed.groupby("candidate_id")["entry_price"].nunique(dropna=True).gt(1).any():
        raise AssertionError("one candidate has multiple continuation entry prices")
    return needed.drop_duplicates("candidate_id", keep="first")


def _bcf_priority() -> pd.DataFrame:
    source = pd.read_parquet(
        continuation.DUAL,
        columns=["candidate_id", "__decision_ts__", "bcf_mc1_expected_bps", "side_name"],
    ).copy()
    source["candidate_id"] = source["candidate_id"].astype(str)
    source["__decision_ts__"] = pd.to_datetime(source["__decision_ts__"], utc=True, errors="raise")
    source["bcf_mc1_expected_bps"] = pd.to_numeric(source["bcf_mc1_expected_bps"], errors="coerce")
    if source.candidate_id.duplicated().any():
        raise AssertionError("sealed dual source has duplicate candidate identities")
    return source


def _candidate_table(rows: pd.DataFrame, prices: pd.DataFrame, priorities: pd.DataFrame) -> pd.DataFrame:
    out = rows.merge(prices, on="candidate_id", how="left", validate="many_to_one")
    out = out.merge(priorities, on="candidate_id", how="left", validate="many_to_one")
    if out["entry_price"].isna().any() or out["bcf_mc1_expected_bps"].isna().any():
        raise AssertionError("C1 portfolio input lacks source-aligned entry price or BCF priority")
    if not out["entry_decision_ts"].eq(out["__decision_ts__"]).all():
        raise AssertionError("continuation entry timestamps do not match sealed dual decisions")
    if not out["side_name"].astype(str).str.lower().eq("long").all():
        raise AssertionError("continuation portfolio replay is long-only")
    if out.duplicated(["arm", "mc1_threshold_bps", "candidate_id"]).any():
        raise AssertionError("duplicate candidate within an arm/threshold replay")
    exit_bars = pd.to_numeric(out["c1_exit_bar"], errors="coerce")
    gross_bps = pd.to_numeric(out["c1_gross_bps"], errors="coerce")
    if (~np.isfinite(exit_bars)).any() or (~np.isfinite(gross_bps)).any() or (exit_bars < 0).any():
        raise AssertionError("C1 outcomes contain invalid exit metadata")
    exit_ts = out["entry_decision_ts"] + pd.to_timedelta((exit_bars.astype(int) + 1) * 15, unit="m")
    candidates = pd.DataFrame({
        "timestamp": out["entry_decision_ts"],
        "candidate_id": out["candidate_id"],
        "symbol": out["__symbol__"].astype(str),
        "side": "long",
        "strategy_id": "strict_r3_p8u_15m_continuation_long",
        "policy_archetype": "strict_r3_p8u_15m_continuation_long",
        # Keep BCF-MC1 as the only auction-ordering authority.  The rank is
        # inert in this controlled replay and avoids creating a new selector.
        "normalized_rank_score": 1.0,
        "strategy_rank_pct": 1.0,
        "base_strategy_threshold": 0.0,
        "calibrated_score": 1.0,
        "portfolio_priority_adjustment": out["bcf_mc1_expected_bps"],
        "entry_price": out["entry_price"],
        "exit_timestamp": exit_ts,
        "exit_price": out["entry_price"] * (1.0 + gross_bps / 10_000.0),
        "net_return": pd.to_numeric(out["c1_net_bps"], errors="raise") / 10_000.0,
        "gross_return": gross_bps / 10_000.0,
        "holding_bars": (exit_bars.astype(int) + 1),
        "simple_policy_exit_reason": out["c1_exit_reason"].astype(str),
        "fees_bps": 100.0,
        "expected_friction_bps": 0.0,
        "price_gap_bps": 0.0,
        "liquidity_capacity_weight": 1.0,
    })
    return normalise_candidate_table(candidates)


def _attach_ids(decisions: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    out = decisions.copy()
    indices = pd.to_numeric(out["candidate_index"], errors="raise").astype(int)
    if (indices < 0).any() or (indices >= len(candidates)).any():
        raise AssertionError("portfolio replay decision references outside candidate table")
    for field in ("candidate_id", "timestamp", "symbol"):
        out[field] = candidates.iloc[indices.to_numpy()][field].to_numpy()
    return out


def _period_metrics(accepted: pd.DataFrame, period: str) -> pd.DataFrame:
    if accepted.empty:
        return pd.DataFrame(columns=[period, "trades", "net_bps_per_trade", "net_sum_bps", "win_rate"])
    work = accepted.copy()
    timestamp = pd.to_datetime(work["timestamp"], utc=True)
    work[period] = timestamp.dt.strftime("%Y-%m" if period == "month" else "%Y-%m-%d")
    work["net_bps"] = pd.to_numeric(work["position_net_return"], errors="raise") * 10_000.0
    return work.groupby(period, as_index=False).agg(
        trades=("candidate_id", "size"),
        net_bps_per_trade=("net_bps", "mean"),
        net_sum_bps=("net_bps", "sum"),
        win_rate=("net_bps", lambda series: float((series > 0.0).mean())),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", action="append", required=True, help="workspace-relative glob(s) for entry_outcomes.parquet")
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument("--mc1-threshold", type=float, action="append", required=True)
    parser.add_argument("--arm", action="append", choices=REQUIRED_ARMS, default=list(REQUIRED_ARMS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    rows, inputs = _read_inputs(args.input_glob)
    thresholds = {float(value) for value in args.mc1_threshold}
    arms = tuple(dict.fromkeys(args.arm))
    rows = rows.loc[rows["arm"].isin(arms) & pd.to_numeric(rows["mc1_threshold_bps"], errors="coerce").isin(thresholds)].copy()
    if rows.empty:
        raise RuntimeError("no selected C0/C1 outcomes")
    prices = _entry_prices(args.state_root.resolve())
    priorities = _bcf_priority()
    params = canonical_portfolio_params()
    summaries: list[dict[str, object]] = []
    output.mkdir(parents=True, exist_ok=False)
    for threshold in sorted(thresholds, reverse=True):
        for arm in arms:
            subset = rows.loc[(rows["arm"] == arm) & pd.to_numeric(rows["mc1_threshold_bps"], errors="coerce").eq(threshold)].copy()
            if subset.empty:
                continue
            candidates = _candidate_table(subset, prices, priorities)
            decisions, equity, _ = replay_candidates(
                candidates, params, mode="global_auction", ev_curve=CAUSAL_AUCTION_CURVE, market_mode="perp"
            )
            decisions = _attach_ids(decisions, candidates)
            accepted = decisions.loc[decisions["accepted"].fillna(False).astype(bool)].copy()
            tag = f"mc1_{threshold:g}_{arm}"
            candidates.to_parquet(output / f"{tag}_candidates.parquet", index=False, compression="zstd")
            decisions.to_parquet(output / f"{tag}_decisions.parquet", index=False, compression="zstd")
            accepted.to_parquet(output / f"{tag}_accepted.parquet", index=False, compression="zstd")
            equity.to_parquet(output / f"{tag}_equity.parquet", index=False, compression="zstd")
            _period_metrics(accepted, "day").to_parquet(output / f"{tag}_daily.parquet", index=False)
            _period_metrics(accepted, "month").to_parquet(output / f"{tag}_monthly.parquet", index=False)
            metrics = compute_replay_metrics(candidates, decisions, equity, params=params)
            summaries.append({
                "mc1_threshold_bps": threshold,
                "arm": arm,
                "routed_candidates": len(candidates),
                "portfolio_accepted": len(accepted),
                **metrics,
            })
    summary = pd.DataFrame(summaries)
    summary.to_parquet(output / "portfolio_summary.parquet", index=False)
    manifest = {
        "schema": "strict-r3-p8u-15m-continuation-c1-portfolio-v1",
        "scope": "offline strict-OOS research; no feature generation, model fitting, exchange IO, or order submission",
        "inputs": [str(path) for path in inputs],
        "input_sha256": {str(path): _sha256(path) for path in inputs},
        "state_root": str(args.state_root.resolve()),
        "state_source": "source-aligned continuation entry prices; one immutable price per candidate",
        "priority": "sealed BCF MC1 expected bps only; no C1 or outcome-derived ranking authority",
        "exit": "C0 parent or C1 activation-only resimulated rich-policy path; 15-minute state update applies to the next interval",
        "portfolio": asdict(canonical_portfolio_params()),
        "thresholds": sorted(thresholds),
        "arms": arms,
        "cost": "100 bps embedded once in each routed outcome",
        "stage": "challenger-only; no live/canonical mutation",
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
