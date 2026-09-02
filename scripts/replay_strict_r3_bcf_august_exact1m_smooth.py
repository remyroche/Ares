#!/usr/bin/env python3
"""Replay the frozen BCF +5m route with the frozen exact-1m smooth policy.

This is an offline, fixed-contract producer.  It deliberately does *not*
train, calibrate, select, or alter the BCF/MC1 route.  The supplied exact-path
dataset has already frozen the target-free BCF admission population; outcomes
are used only here, after that routing decision, to replay the policy and the
normal constrained portfolio auction.

The policy is the winning rich-policy V2 extension: 1.5 ATR smooth-protection
activation, 0.5 strength, and 1.5 power.  It is the same frozen +5-minute
research execution contract used for the 2026 sensitivity, not a new live
policy promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.exact_1m_rich_policy_contract import (  # noqa: E402
    Exact1mRichV2ExecutionContract,
    RichExitExtensions,
    exact_1m_rich_v2_receipt,
    replay_exact_1m_rich_policy_v2,
)
from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    compute_replay_metrics,
    normalise_candidate_table,
    replay_candidates,
)
from scripts.report_strict_r3_mc1_d2_controlled_portfolio import (  # noqa: E402
    CAUSAL_AUCTION_CURVE,
    _params as canonical_portfolio_params,
)
from scripts.run_strict_r3_exact_1m_rich_extensions_hpo import (  # noqa: E402
    _load_dataset,
    _load_frozen_policy,
    _resort,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_empty(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"immutable output already exists: {path}")
    path.mkdir(parents=True)


def _frozen_extensions(path: Path) -> RichExitExtensions:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = dict(payload.get("extensions") or {})
    expected = {
        "protection_activation_atr": 1.5,
        "protection_strength": 0.5,
        "protection_power": 1.5,
    }
    for key, value in expected.items():
        if not np.isclose(float(values.get(key, np.nan)), value):
            raise AssertionError(f"frozen extension has unexpected {key}")
    extension = RichExitExtensions(**values)
    extension.validate()
    return extension


def _portfolio_table(paths: Any, replay: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = paths.rows
    net = np.asarray(replay["net_bps"], dtype=float)
    gross = np.asarray(replay["gross_bps"], dtype=float)
    exit_timestamp = pd.to_datetime(np.asarray(replay["exit_timestamp"]), utc=True)
    if not np.isfinite(net).all() or exit_timestamp.isna().any():
        raise AssertionError("complete exact paths did not resolve into complete policy outcomes")
    exit_minutes = np.asarray(replay["exit_minute"], dtype=float)
    priority = pd.to_numeric(rows["priority_bps"], errors="raise").to_numpy(float)
    table = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(rows["entry_ts"], utc=True),
            "decision_timestamp": pd.to_datetime(rows["timestamp"], utc=True),
            "candidate_id": rows["candidate_id"].astype(str),
            "symbol": rows["symbol"].astype(str),
            "side": "long",
            "strategy_id": "strict_r3_bcf_mc1_30_exact1m_tplus5_smooth_long",
            "policy_archetype": "strict_r3_bcf_mc1_30_exact1m_tplus5_smooth_long",
            # The candidate population was admitted with BCF MC1 >= +30 bps
            # before any path was touched.  Preserve that BCF value as the
            # sole auction priority; no within-held-period rank is computed.
            "normalized_rank_score": 1.0,
            "strategy_rank_pct": 1.0,
            "base_strategy_threshold": 0.0,
            "calibrated_score": priority,
            "portfolio_priority_adjustment": priority,
            "mapped_expected_net_bps": priority,
            "entry_price": np.asarray(paths.entry, dtype=float),
            "exit_timestamp": exit_timestamp,
            "exit_price": np.asarray(replay["exit_price"], dtype=float),
            # V2 net contains the 100-bps contract cost exactly once.  The
            # auction receives no second realised-cost debit.
            "net_return": net / 10_000.0,
            "gross_return": gross / 10_000.0,
            "holding_bars": np.maximum(1.0, np.ceil((exit_minutes + 1.0) / 15.0)),
            "simple_policy_exit_reason": np.asarray(replay["exit_reason"], dtype=object),
            "fees_bps": 100.0,
            "expected_friction_bps": 0.0,
            "price_gap_bps": 0.0,
            "liquidity_capacity_weight": 1.0,
            "policy_outcome_available": True,
        }
    )
    return normalise_candidate_table(table)


def _attach_identity(decisions: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    indices = pd.to_numeric(decisions["candidate_index"], errors="raise").astype(int).to_numpy()
    result = decisions.copy()
    for column in (
        "candidate_id", "decision_timestamp", "symbol", "mapped_expected_net_bps",
        "entry_price", "exit_timestamp", "exit_price", "simple_policy_exit_reason",
    ):
        result[column] = candidates.iloc[indices][column].to_numpy()
    result["net_bps"] = pd.to_numeric(result["position_net_return"], errors="coerce") * 10_000.0
    return result


def _daily(paths: Any, decisions: pd.DataFrame) -> pd.DataFrame:
    rows = paths.rows.copy()
    rows["day"] = pd.to_datetime(rows["timestamp"], utc=True).dt.floor("D")
    routed = rows.groupby("day", as_index=False).agg(
        bcf_mc1_admitted_candidates=("candidate_id", "size"),
        mean_mapped_ev_bps=("priority_bps", "mean"),
        max_mapped_ev_bps=("priority_bps", "max"),
    )
    out = decisions.copy()
    out["day"] = pd.to_datetime(out["decision_timestamp"], utc=True).dt.floor("D")
    accepted = out["accepted"].fillna(False).astype(bool)
    auction = out.groupby("day", as_index=False).agg(
        auction_evaluated=("accepted", "size"),
        portfolio_accepted=("accepted", "sum"),
    )
    accepted_rows = out.loc[accepted].copy()
    if accepted_rows.empty:
        accepted_metrics = pd.DataFrame(columns=["day", "net_ev_bps_per_trade", "net_sum_bps", "mean_expected_ev_bps", "mean_exit_minutes"])
        exits = pd.DataFrame(columns=["day"])
    else:
        accepted_rows["exit_minutes"] = (
            pd.to_datetime(accepted_rows["exit_timestamp"], utc=True)
            - pd.to_datetime(accepted_rows["timestamp"], utc=True)
        ).dt.total_seconds() / 60.0
        accepted_metrics = accepted_rows.groupby("day", as_index=False).agg(
            net_ev_bps_per_trade=("net_bps", "mean"),
            net_sum_bps=("net_bps", "sum"),
            mean_expected_ev_bps=("mapped_expected_net_bps", "mean"),
            mean_exit_minutes=("exit_minutes", "mean"),
        )
        exits = (
            accepted_rows.pivot_table(
                index="day", columns="simple_policy_exit_reason", values="candidate_id",
                aggfunc="size", fill_value=0,
            )
            .add_prefix("exit_").reset_index()
        )
    start = rows["day"].min()
    end = rows["day"].max()
    calendar = pd.DataFrame({"day": pd.date_range(start, end, freq="D", tz="UTC")})
    result = calendar.merge(routed, on="day", how="left").merge(auction, on="day", how="left")
    result = result.merge(accepted_metrics, on="day", how="left").merge(exits, on="day", how="left")
    for column in ("bcf_mc1_admitted_candidates", "auction_evaluated", "portfolio_accepted"):
        result[column] = result[column].fillna(0).astype(int)
    return result


def run(args: argparse.Namespace) -> Path:
    out = Path(args.out_dir).resolve()
    _assert_empty(out)
    paths = _resort(_load_dataset(Path(args.dataset), expected_delay=5))
    route = dict(paths.manifest.get("candidate_source") or {})
    selection_inputs = {str(value) for value in route.get("selection_inputs") or []}
    is_dual_mc1 = {"bcf_mc1_expected_bps", "current_v5_mc1_expected_bps"}.issubset(selection_inputs)
    params, median_atr, policy_audit = _load_frozen_policy(Path(args.base_policy))
    extension = _frozen_extensions(Path(args.extensions_winner))
    contract = Exact1mRichV2ExecutionContract(entry_delay_minutes=5)
    replay = replay_exact_1m_rich_policy_v2(
        entry=paths.entry,
        atr=paths.atr,
        highs=paths.high,
        lows=paths.low,
        closes=paths.close,
        entry_timestamps=paths.rows["entry_ts"],
        params=params,
        median_atr_fraction=median_atr,
        extensions=extension,
        contract=contract,
    )
    if not np.asarray(replay["path_valid"], dtype=bool).all():
        raise AssertionError("the materialised exact path dataset unexpectedly contains invalid paths")
    candidates = _portfolio_table(paths, replay)
    decisions, equity, _ = replay_candidates(
        candidates,
        canonical_portfolio_params(),
        mode="global_auction",
        ev_curve=CAUSAL_AUCTION_CURVE,
        market_mode="perp",
        initial_wallet=1000.0,
    )
    # Compute the common metrics before adding convenience identity columns.
    # ``compute_replay_metrics`` owns the candidate/decision merge and expects
    # those names not to be present on both sides of that merge.
    raw = compute_replay_metrics(candidates, decisions, equity, params=canonical_portfolio_params())
    decisions = _attach_identity(decisions, candidates)
    accepted = decisions.loc[decisions["accepted"].fillna(False).astype(bool)].copy()
    outcome = pd.DataFrame(
        {
            "candidate_id": paths.rows["candidate_id"].astype(str),
            "decision_timestamp": pd.to_datetime(paths.rows["timestamp"], utc=True),
            "entry_timestamp": pd.to_datetime(paths.rows["entry_ts"], utc=True),
            "symbol": paths.rows["symbol"].astype(str),
            "mapped_expected_net_bps": pd.to_numeric(paths.rows["priority_bps"], errors="raise"),
            "entry_price": np.asarray(paths.entry, dtype=float),
            "exit_price": np.asarray(replay["exit_price"], dtype=float),
            "exit_timestamp": pd.to_datetime(np.asarray(replay["exit_timestamp"]), utc=True),
            "gross_bps": np.asarray(replay["gross_bps"], dtype=float),
            "net_bps": np.asarray(replay["net_bps"], dtype=float),
            "exit_minute": np.asarray(replay["exit_minute"], dtype=int),
            "exit_reason": np.asarray(replay["exit_reason"], dtype=object),
        }
    )
    daily = _daily(paths, decisions)
    metrics = {
        "routed_bcf_mc1_admitted_candidates": int(len(paths.rows)),
        "portfolio_entries": int(len(accepted)),
        "net_ev_bps_per_trade": float(accepted["net_bps"].mean()) if len(accepted) else float("nan"),
        "total_net_bps": float(accepted["net_bps"].sum()) if len(accepted) else 0.0,
        "mean_exit_minutes": float(
            (pd.to_datetime(accepted["exit_timestamp"], utc=True) - pd.to_datetime(accepted["timestamp"], utc=True))
            .dt.total_seconds().mean() / 60.0
        ) if len(accepted) else float("nan"),
        "sortino": float(raw.get("sortino", np.nan)),
        "max_drawdown": float(raw.get("max_drawdown", np.nan)),
        "worst_week_return": float(raw.get("worst_week", np.nan)),
        "days_with_no_portfolio_entries": int((daily["portfolio_accepted"] == 0).sum()),
    }
    metrics["routed_candidates"] = int(len(paths.rows))
    if is_dual_mc1:
        metrics["routed_dual_mc1_admitted_candidates"] = int(len(paths.rows))
    candidates.to_parquet(out / "auction_candidates.parquet", index=False, compression="zstd")
    outcome.to_parquet(out / "exact1m_policy_outcomes.parquet", index=False, compression="zstd")
    decisions.to_parquet(out / "portfolio_decisions.parquet", index=False, compression="zstd")
    accepted.to_parquet(out / "portfolio_accepted.parquet", index=False, compression="zstd")
    equity.to_parquet(out / "portfolio_equity.parquet", index=False, compression="zstd")
    daily.to_parquet(out / "daily_portfolio_metrics.parquet", index=False, compression="zstd")
    (out / "portfolio_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema": "strict_r3_bcf_mc1_30_august_exact1m_tplus5_smooth_replay_v1",
        "purpose": "fixed, target-free score-route; frozen +5m exact-1m smooth policy replay",
        "research_only": True,
        "no_retraining_or_retuning": True,
        "dataset": paths.audit,
        "admission_route": {
            "selection_inputs": sorted(selection_inputs),
            "selection_predicate": route.get("predicate"),
            "strict_dual_bcf_current_mc1": is_dual_mc1,
            "auction_priority": "frozen BCF MC1 mapped EV",
        },
        "base_policy": policy_audit,
        "extensions_winner": {"path": str(Path(args.extensions_winner).resolve()), "sha256": _sha256(Path(args.extensions_winner)), **asdict(extension)},
        "contract": contract.to_dict(),
        "contract_sha256": contract.hash,
        "policy_replay_receipt": exact_1m_rich_v2_receipt(params=params, extensions=extension, replay=replay, contract=contract),
        "portfolio_contract": {"params": canonical_portfolio_params(), "mode": "global_auction", "priority": "frozen BCF MC1 mapped EV"},
        "cost": "exact one 100-bps policy cost in rich-policy net; no second auction debit",
        "metrics": metrics,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--base-policy", required=True, type=Path)
    parser.add_argument("--extensions-winner", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
