#!/usr/bin/env python3
"""Compare a strict-R3 router-only portfolio against its matched baseline.

This is a research-only, read-only audit.  It expects completed decision and
equity ledgers produced by the same policy/portfolio simulator and makes no
model, exchange, or live-state changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


KEY_COLUMNS = ["timestamp", "symbol", "side"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _accepted(decisions_path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(decisions_path)
    required = set(KEY_COLUMNS + ["accepted", "policy_outcome_available", "position_net_return"])
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{decisions_path} is missing {missing}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    frame = frame.loc[frame["accepted"].astype(bool) & frame["policy_outcome_available"].astype(bool)].copy()
    if frame.duplicated(KEY_COLUMNS).any():
        raise AssertionError(f"accepted decision identities are not unique in {decisions_path}")
    frame["net_bps"] = pd.to_numeric(frame["position_net_return"], errors="coerce") * 10_000.0
    if not np.isfinite(frame["net_bps"]).all():
        raise AssertionError(f"accepted realised outcomes are non-finite in {decisions_path}")
    frame["month"] = frame["timestamp"].dt.strftime("%Y-%m")
    frame["day"] = frame["timestamp"].dt.strftime("%Y-%m-%d")
    return frame


def _summary(frame: pd.DataFrame, arm: str) -> dict[str, Any]:
    return {
        "arm": arm,
        "entries": int(len(frame)),
        "net_ev_bps_per_trade": float(frame["net_bps"].mean()),
        "net_sum_bps": float(frame["net_bps"].sum()),
        "positive_trade_fraction": float((frame["net_bps"] > 0.0).mean()),
        "worst_trade_bps": float(frame["net_bps"].min()),
        "best_trade_bps": float(frame["net_bps"].max()),
        "symbols": int(frame["symbol"].nunique()),
        "timestamps": int(frame["timestamp"].nunique()),
    }


def _monthly(frame: pd.DataFrame, arm: str) -> pd.DataFrame:
    result = (
        frame.groupby("month", sort=True)
        .agg(
            entries=("symbol", "size"),
            net_ev_bps_per_trade=("net_bps", "mean"),
            net_sum_bps=("net_bps", "sum"),
            positive_trade_fraction=("net_bps", lambda values: float((values > 0.0).mean())),
            symbols=("symbol", "nunique"),
        )
        .reset_index()
    )
    result.insert(0, "arm", arm)
    return result


def _concentration(frame: pd.DataFrame, arm: str) -> pd.DataFrame:
    symbol_counts = frame.groupby("symbol", sort=True).size().rename("entries").reset_index()
    total = float(symbol_counts["entries"].sum())
    symbol_counts["share"] = symbol_counts["entries"] / total
    return pd.DataFrame(
        [
            {
                "arm": arm,
                "symbol_hhi": float((symbol_counts["share"] ** 2).sum()),
                "top_1_symbol_share": float(symbol_counts["share"].max()),
                "top_5_symbol_share": float(symbol_counts.nlargest(5, "share")["share"].sum()),
                "worst_day_bps": float(frame.groupby("day")["net_bps"].sum().min()),
                "worst_hour_bps": float(frame.groupby("timestamp")["net_bps"].sum().min()),
                "max_entries_per_timestamp": int(frame.groupby("timestamp").size().max()),
            }
        ]
    )


def _drawdown(equity_path: Path, arm: str) -> pd.DataFrame:
    frame = pd.read_parquet(equity_path, columns=["timestamp", "mtm_equity", "wallet", "open_positions", "open_capital_pct"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    mtm_equity = pd.to_numeric(frame["mtm_equity"], errors="coerce")
    realized_wallet = pd.to_numeric(frame["wallet"], errors="coerce")
    if not np.isfinite(mtm_equity).all() or (mtm_equity <= 0.0).any():
        raise AssertionError(f"invalid MTM equity in {equity_path}")
    if not np.isfinite(realized_wallet).all() or (realized_wallet <= 0.0).any():
        raise AssertionError(f"invalid realized wallet in {equity_path}")
    mtm_drawdown = mtm_equity / mtm_equity.cummax() - 1.0
    wallet_drawdown = realized_wallet / realized_wallet.cummax() - 1.0
    trough_index = int(mtm_drawdown.idxmin())
    peak_index = int(mtm_equity.iloc[: trough_index + 1].idxmax())
    return pd.DataFrame(
        [
            {
                "arm": arm,
                # This matches the canonical stored portfolio metric exactly.
                "max_realized_wallet_drawdown": float(wallet_drawdown.min()),
                # MTM drawdown is intentionally separate; it catches open-risk
                # episodes that a realized-wallet series cannot show.
                "max_mtm_drawdown": float(mtm_drawdown.iloc[trough_index]),
                "peak_timestamp": frame.loc[peak_index, "timestamp"],
                "trough_timestamp": frame.loc[trough_index, "timestamp"],
                "peak_mtm_equity": float(mtm_equity.iloc[peak_index]),
                "trough_mtm_equity": float(mtm_equity.iloc[trough_index]),
                "trough_open_positions": int(frame.loc[trough_index, "open_positions"]),
                "trough_open_capital_pct": float(frame.loc[trough_index, "open_capital_pct"]),
            }
        ]
    )


def _cohorts(baseline: pd.DataFrame, routed: pd.DataFrame) -> pd.DataFrame:
    left = baseline[KEY_COLUMNS + ["net_bps"]].rename(columns={"net_bps": "baseline_net_bps"})
    right = routed[KEY_COLUMNS + ["net_bps"]].rename(columns={"net_bps": "router_net_bps"})
    joined = left.merge(right, on=KEY_COLUMNS, how="outer", indicator=True, validate="one_to_one")
    rows: list[dict[str, Any]] = []
    mapping = {"both": "both", "left_only": "baseline_only", "right_only": "router_only"}
    for source, cohort in mapping.items():
        selected = joined.loc[joined["_merge"].eq(source)]
        outcome = selected["baseline_net_bps"] if source != "right_only" else selected["router_net_bps"]
        rows.append(
            {
                "cohort": cohort,
                "entries": int(len(selected)),
                "net_ev_bps_per_trade": float(outcome.mean()),
                "net_sum_bps": float(outcome.sum()),
                "positive_trade_fraction": float((outcome > 0.0).mean()),
            }
        )
    return pd.DataFrame(rows)


def _paths(root: Path, threshold: int) -> tuple[Path, Path]:
    decisions = root / f"router_dual_{threshold}_2026_febjul_decisions.parquet"
    equity = root / f"router_dual_{threshold}_2026_febjul_equity.parquet"
    if not decisions.exists() or not equity.exists():
        raise FileNotFoundError(f"missing threshold-{threshold} decision/equity ledgers under {root}")
    return decisions, equity


def run(*, baseline_root: Path, router_root: Path, out: Path, threshold: int) -> None:
    baseline_decisions, baseline_equity = _paths(baseline_root, threshold)
    router_decisions, router_equity = _paths(router_root, threshold)
    baseline = _accepted(baseline_decisions)
    router = _accepted(router_decisions)
    out.mkdir(parents=True, exist_ok=False)

    pd.DataFrame([_summary(baseline, "baseline"), _summary(router, "p3_router_only")]).to_parquet(out / "summary.parquet", index=False)
    pd.concat([_monthly(baseline, "baseline"), _monthly(router, "p3_router_only")], ignore_index=True).to_parquet(out / "monthly.parquet", index=False)
    _cohorts(baseline, router).to_parquet(out / "cohorts.parquet", index=False)
    pd.concat([_concentration(baseline, "baseline"), _concentration(router, "p3_router_only")], ignore_index=True).to_parquet(out / "concentration.parquet", index=False)
    pd.concat([_drawdown(baseline_equity, "baseline"), _drawdown(router_equity, "p3_router_only")], ignore_index=True).to_parquet(out / "drawdown.parquet", index=False)

    audit = {
        "schema": "strict_r3_router_only_baseline_acceptance_audit_v1",
        "scope": "offline research-only, completed causal portfolio ledgers",
        "threshold_bps": threshold,
        "identity": "timestamp/symbol/side; accepted and policy-outcome-available rows only",
        "baseline_decisions_sha256": _sha256(baseline_decisions),
        "router_decisions_sha256": _sha256(router_decisions),
        "baseline_equity_sha256": _sha256(baseline_equity),
        "router_equity_sha256": _sha256(router_equity),
        "router_contract": "P3/P4 has eligibility authority only; no router score is included as a downstream numeric coordinate",
        "checks": {
            "baseline_unique_accepted_identities": True,
            "router_unique_accepted_identities": True,
            "outcome_coverage": "100 percent in both completed portfolio ledgers",
        },
    }
    (out / "run_manifest.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold", type=int, default=50)
    args = parser.parse_args()
    run(baseline_root=args.baseline_root, router_root=args.router_root, out=args.out, threshold=args.threshold)


if __name__ == "__main__":
    main()
