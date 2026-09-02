#!/usr/bin/env python3
"""Ablate only the per-decision entry cap for the dual BCF/current stack.

The input must be the target-free dual-admission population after its fixed
policy outcomes have been attached.  It deliberately leaves the BCF-MC1
priority, exact one-minute rich exit outcomes, 7x leverage, 10%-wallet margin
slots, eight-position concurrency cap and 80% margin cap unchanged.

This is research-only: it never reads live state or calls an exchange.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    compute_replay_metrics,
    replay_candidates,
)
from scripts.report_strict_r3_mc1_d2_controlled_portfolio import (  # noqa: E402
    CAUSAL_AUCTION_CURVE,
    _params as canonical_portfolio_params,
)
from scripts.run_strict_r3_exact_1m_rich_matched_attribution import (  # noqa: E402
    _attach_candidate_ids,
)


SCHEMA = "strict_r3_dual_bcf_current_entry_cap_ablation_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: pd.Series) -> pd.Series:
    return pd.to_datetime(value, utc=True, errors="raise")


def _drawdown(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float((clean / clean.cummax() - 1.0).min())


def _period_metrics(
    accepted: pd.DataFrame,
    equity: pd.DataFrame,
    *,
    cap: int,
    period: str,
) -> dict[str, Any]:
    work = accepted.copy()
    net_bps = pd.to_numeric(work.get("position_net_return"), errors="coerce") * 10_000.0
    gross_bps = pd.to_numeric(work.get("position_gross_return"), errors="coerce") * 10_000.0
    timestamps = _utc(work["decision_timestamp"]) if len(work) else pd.Series(dtype="datetime64[ns, UTC]")
    days = max(int(timestamps.dt.normalize().nunique()), 1) if len(timestamps) else 0
    months = (
        pd.DataFrame({"month": timestamps.dt.strftime("%Y-%m"), "net_bps": net_bps})
        .groupby("month", sort=True)["net_bps"].mean()
        if len(work) else pd.Series(dtype=float)
    )
    weeks = (
        pd.DataFrame({"week": timestamps.dt.strftime("%G-W%V"), "net_bps": net_bps})
        .groupby("week", sort=True)["net_bps"].mean()
        if len(work) else pd.Series(dtype=float)
    )
    return {
        "entry_cap_per_candle": int(cap),
        "period": str(period),
        "entries": int(len(work)),
        "trades_per_active_day": float(len(work) / days) if days else 0.0,
        "net_ev_bps_per_trade": float(net_bps.mean()) if len(work) else float("nan"),
        "gross_ev_bps_per_trade": float(gross_bps.mean()) if len(work) else float("nan"),
        "net_sum_bps": float(net_bps.sum()) if len(work) else 0.0,
        "gross_sum_bps": float(gross_bps.sum()) if len(work) else 0.0,
        "worst_month_net_ev_bps": float(months.min()) if len(months) else float("nan"),
        "worst_week_net_ev_bps": float(weeks.min()) if len(weeks) else float("nan"),
        "positive_month_fraction": float(months.gt(0.0).mean()) if len(months) else float("nan"),
        "portfolio_final_wallet": float(pd.to_numeric(equity["wallet"], errors="coerce").iloc[-1]),
        "portfolio_growth_pct": float(
            (pd.to_numeric(equity["wallet"], errors="coerce").iloc[-1]
             / pd.to_numeric(equity["wallet"], errors="coerce").iloc[0] - 1.0)
            * 100.0
        ),
        "portfolio_max_drawdown_pct": _drawdown(equity["wallet"]) * 100.0,
    }


def _monthly_metrics(accepted: pd.DataFrame, cap: int) -> pd.DataFrame:
    if accepted.empty:
        return pd.DataFrame()
    work = accepted.copy()
    work["month"] = _utc(work["decision_timestamp"]).dt.strftime("%Y-%m")
    work["net_bps"] = pd.to_numeric(work["position_net_return"], errors="coerce") * 10_000.0
    work["gross_bps"] = pd.to_numeric(work["position_gross_return"], errors="coerce") * 10_000.0
    out = work.groupby("month", sort=True).agg(
        entries=("candidate_id", "size"),
        net_ev_bps_per_trade=("net_bps", "mean"),
        gross_ev_bps_per_trade=("gross_bps", "mean"),
        net_sum_bps=("net_bps", "sum"),
        gross_sum_bps=("gross_bps", "sum"),
    ).reset_index()
    out.insert(0, "entry_cap_per_candle", int(cap))
    return out


def _parse_caps(raw: str) -> list[int]:
    values = sorted({int(value.strip()) for value in raw.split(",") if value.strip()})
    if not values or any(value < 1 for value in values):
        raise ValueError("caps must be one or more positive integers")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--caps", default="1,2,3,4")
    parser.add_argument(
        "--baseline-decisions", type=Path,
        help="Optional existing two-entry decision ledger; when supplied cap=2 must match it exactly.",
    )
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")
    caps = _parse_caps(args.caps)
    candidates = pd.read_parquet(args.candidates).copy()
    required = {
        "candidate_id", "timestamp", "decision_timestamp", "symbol", "side",
        "portfolio_priority_adjustment", "net_return", "gross_return",
        "exit_timestamp", "policy_outcome_available",
    }
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise ValueError(f"candidate table lacks {missing}")
    if not candidates["candidate_id"].astype(str).is_unique:
        raise ValueError("candidate IDs must be unique")
    if not candidates["policy_outcome_available"].fillna(False).astype(bool).all():
        raise ValueError("all provided replay candidates must have exact valid outcomes")
    if not np.isfinite(pd.to_numeric(candidates["portfolio_priority_adjustment"], errors="coerce")).all():
        raise ValueError("BCF-MC1 priority is incomplete")

    baseline = None
    if args.baseline_decisions is not None:
        baseline = pd.read_parquet(args.baseline_decisions).copy()

    summaries: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    monthly_frames: list[pd.DataFrame] = []
    validation: dict[str, Any] = {}
    args.out_dir.mkdir(parents=True)

    for cap in caps:
        params = replace(
            canonical_portfolio_params(),
            max_new_entries_per_bar=int(cap),
            max_new_entries_per_strategy_per_bar=int(cap),
            portfolio_policy_version=f"{SCHEMA}:cap{cap}",
        )
        decisions, equity, raw_metrics = replay_candidates(
            candidates, params, mode="global_auction", ev_curve=CAUSAL_AUCTION_CURVE,
            market_mode="perp",
        )
        decisions = _attach_candidate_ids(decisions, candidates)
        accepted = decisions.loc[decisions["accepted"].fillna(False).astype(bool)].copy()
        # The portfolio engine records the shared long-only strategy limiter
        # separately from the global bar limiter.  Both are the same swept
        # per-candle cap in this one-strategy ablation.
        rejected_by_cap = decisions["rejection_reason"].isin(
            {
                "max_new_entries_per_bar_reached",
                "max_new_entries_per_strategy_per_bar_reached",
            }
        ).sum()
        headline = _period_metrics(accepted, equity, cap=cap, period="all")
        headline.update({
            "input_rows": int(len(candidates)),
            "rejected_by_per_candle_cap": int(rejected_by_cap),
            "max_concurrent_positions": int(params.max_concurrent_positions),
            "max_total_wallet_allocation_pct": float(params.max_total_wallet_allocation_pct),
            "margin_slot_wallet_fraction": float(params.margin_slot_wallet_fraction or 0.0),
            "leverage": float(params.perp_default_leverage),
            "replay_net_pnl_quote": float(raw_metrics.get("net_pnl", np.nan)),
            "replay_sortino": float(raw_metrics.get("sortino", np.nan)),
        })
        summaries.append(headline)
        monthly_frames.append(_monthly_metrics(accepted, cap))
        years = _utc(accepted["decision_timestamp"]).dt.year.astype(str) if len(accepted) else pd.Series(dtype=str)
        for year in sorted(years.unique()):
            mask = years.eq(year)
            year_equity = equity.loc[_utc(equity["timestamp"]).dt.year.astype(str).eq(year)].copy()
            if not year_equity.empty:
                # Preserve the incoming wallet for an interpretable annual growth rate.
                prior = equity.loc[_utc(equity["timestamp"]) < _utc(year_equity["timestamp"]).iloc[0], "wallet"]
                if not prior.empty:
                    year_equity = pd.concat([
                        pd.DataFrame({"timestamp": [pd.NaT], "wallet": [float(prior.iloc[-1])]}), year_equity,
                    ], ignore_index=True)
            yearly_rows.append(_period_metrics(
                accepted.loc[mask].copy(), year_equity if not year_equity.empty else equity.iloc[:1],
                cap=cap, period=str(year),
            ))
        decisions.to_parquet(args.out_dir / f"cap_{cap}_portfolio_decisions.parquet", index=False, compression="zstd")
        accepted.to_parquet(args.out_dir / f"cap_{cap}_accepted_trades.parquet", index=False, compression="zstd")
        equity.to_parquet(args.out_dir / f"cap_{cap}_portfolio_equity.parquet", index=False, compression="zstd")

        if cap == 2 and baseline is not None:
            baseline_accepted = baseline.loc[baseline["accepted"].fillna(False).astype(bool)].copy()
            current_ids = accepted["candidate_id"].astype(str).tolist()
            baseline_ids = baseline_accepted["candidate_id"].astype(str).tolist()
            validation["cap2_matches_existing_control"] = current_ids == baseline_ids
            validation["cap2_accepted_rows"] = int(len(current_ids))
            validation["existing_control_accepted_rows"] = int(len(baseline_ids))
            if not validation["cap2_matches_existing_control"]:
                raise AssertionError("cap=2 does not reproduce the supplied exact-policy control")

    pd.DataFrame(summaries).sort_values("entry_cap_per_candle").to_parquet(
        args.out_dir / "summary_metrics.parquet", index=False,
    )
    pd.DataFrame(yearly_rows).sort_values(["entry_cap_per_candle", "period"]).to_parquet(
        args.out_dir / "yearly_metrics.parquet", index=False,
    )
    pd.concat(monthly_frames, ignore_index=True).sort_values(
        ["entry_cap_per_candle", "month"],
    ).to_parquet(args.out_dir / "monthly_metrics.parquet", index=False)
    manifest = {
        "schema": SCHEMA,
        "purpose": "research-only; entry-cap ablation only; no live state or exchange I/O",
        "input": {"path": str(args.candidates), "sha256": _sha256(args.candidates), "rows": int(len(candidates))},
        "caps": caps,
        "fixed_contract": {
            "admission": "pre-materialised target-free BCF MC1 >=30 AND current-v5 MC1 >=30",
            "auction_priority": "portfolio_priority_adjustment = BCF MC1 expected bps",
            "outcomes": "exact one-minute rich policy outcomes; policy cost already applied once",
            "concurrency": "8 positions, one per symbol, 80% margin cap, 10% wallet margin slots, 7x leverage",
        },
        "validation": validation,
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({"event": "complete", "summary": summaries, "validation": validation}, sort_keys=True))


if __name__ == "__main__":
    main()
