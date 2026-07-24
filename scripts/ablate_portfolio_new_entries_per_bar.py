#!/usr/bin/env python3
"""Compare fixed per-bar entry caps on identical executable candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_canonical_exit_policy_replay import (  # noqa: E402
    _json_safe,
    _load_ev_curve,
    _portfolio_candidates,
)
from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    load_portfolio_policy_params,
    replay_candidates,
)


def _accepted_metrics(decisions: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    accepted = decisions.loc[decisions["accepted"].fillna(False)].copy()
    accepted["timestamp"] = pd.to_datetime(
        accepted["timestamp"], utc=True, errors="coerce"
    )
    accepted["week_start"] = (
        accepted["timestamp"].dt.floor("D")
        - pd.to_timedelta(accepted["timestamp"].dt.weekday, unit="D")
    )
    accepted["net_return"] = pd.to_numeric(
        accepted["position_net_return"], errors="coerce"
    )
    accepted["notional"] = pd.to_numeric(
        accepted["position_size"], errors="coerce"
    )
    accepted["net_pnl"] = accepted["notional"] * accepted["net_return"]
    accepted["positive"] = accepted["net_return"] > 0.0
    accepted["full_sl"] = accepted["position_exit_reason"].astype(str).eq("full_sl")
    accepted["timeout"] = accepted["position_exit_reason"].astype(str).eq("timeout")
    weekly = (
        accepted.groupby("week_start", observed=True)
        .agg(
            trades=("net_return", "size"),
            net_ev_per_trade=("net_return", "mean"),
            net_pnl=("net_pnl", "sum"),
            positive_rate=("positive", "mean"),
            full_sl_rate=("full_sl", "mean"),
            timeout_rate=("timeout", "mean"),
        )
        .reset_index()
    )
    weekly_ev = weekly["net_ev_per_trade"].to_numpy(dtype=np.float64)
    elapsed_days = max(
        (
            accepted["timestamp"].max() - accepted["timestamp"].min()
        ).total_seconds()
        / 86_400.0,
        1.0,
    ) if len(accepted) else 1.0
    metrics = {
        "trades": int(len(accepted)),
        "trades_per_day": float(len(accepted) / elapsed_days),
        "net_ev_per_trade": float(accepted["net_return"].mean()),
        "net_pnl": float(accepted["net_pnl"].sum()),
        "positive_rate": float(accepted["positive"].mean()),
        "full_sl_rate": float(accepted["full_sl"].mean()),
        "timeout_rate": float(accepted["timeout"].mean()),
        "mean_week_ev": float(np.mean(weekly_ev)),
        "std_week_ev": float(np.std(weekly_ev, ddof=0)),
        "worst_week_ev": float(np.min(weekly_ev)),
        "positive_weeks": int(np.sum(weekly_ev > 0.0)),
        "weeks": int(weekly_ev.size),
    }
    metrics["stable_week_ev_objective"] = float(
        metrics["mean_week_ev"]
        - 0.5 * metrics["std_week_ev"]
        + 0.25 * metrics["worst_week_ev"]
    )
    return metrics, weekly


def _set_nested_cap(payload: dict[str, Any], cap: int) -> bool:
    changed = False
    if isinstance(payload.get("concurrency"), dict):
        payload["concurrency"]["max_new_entries_per_bar"] = int(cap)
        changed = True
    if isinstance(payload.get("portfolio_policy"), dict):
        changed = _set_nested_cap(payload["portfolio_policy"], cap) or changed
    if isinstance(payload.get("deployment_policy"), dict):
        changed = _set_nested_cap(payload["deployment_policy"], cap) or changed
    return changed


def _promote_cap(policy_config: Path, cap: int, output_dir: Path) -> list[str]:
    run_root = policy_config.parent.parent
    targets = [
        policy_config,
        run_root / "policy_params" / "training_live_parity_contract.json",
        run_root / "simple_policy_optimiser" / "training_live_parity_contract.json",
        run_root / "simple_policy_optimiser" / "deployment" / "best_policy_params.json",
        run_root
        / "simple_policy_optimiser"
        / "deployment"
        / "best_policy_params_perps.json",
    ]
    updated: list[str] = []
    for path in targets:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if path == policy_config:
            payload.setdefault("concurrency", {})["max_new_entries_per_bar"] = int(cap)
            changed = True
        else:
            changed = _set_nested_cap(payload, cap)
        if not changed:
            continue
        backup = output_dir / "pre_promotion" / path.relative_to(run_root)
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            backup.write_bytes(path.read_bytes())
        path.write_text(
            json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        updated.append(str(path))

    manifest_path = run_root / "policy_params" / "promoted_policy_manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        backup = output_dir / "pre_promotion" / manifest_path.relative_to(run_root)
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            backup.write_bytes(manifest_path.read_bytes())
        payload["max_new_entries_per_bar"] = int(cap)
        digest = hashlib.sha256(policy_config.read_bytes()).hexdigest()
        payload.setdefault("file_sha256", {})[
            "policy_params/optimized_portfolio_policy_config.json"
        ] = digest
        manifest_path.write_text(
            json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        updated.append(str(manifest_path))
    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exit-rows", type=Path, required=True)
    parser.add_argument("--policy-config", type=Path, required=True)
    parser.add_argument("--portfolio-ev-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--caps", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--promote", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_parquet(args.exit_rows)
    candidates = _portfolio_candidates(rows)
    baseline = load_portfolio_policy_params(args.policy_config)
    ev_curve = _load_ev_curve(args.portfolio_ev_reference)
    records: list[dict[str, Any]] = []
    weekly_parts: list[pd.DataFrame] = []
    summaries: dict[str, Any] = {}

    for cap in sorted(set(args.caps)):
        params = replace(baseline, max_new_entries_per_bar=int(cap))
        decisions, equity, summary = replay_candidates(
            candidates,
            params,
            mode="global_auction",
            ev_curve=ev_curve,
            market_mode="perps",
        )
        metrics, weekly = _accepted_metrics(decisions)
        record = {
            "max_new_entries_per_bar": int(cap),
            "portfolio_objective": float(summary["objective"]),
            "compounded_return": float(summary["compounded_return"]),
            "max_drawdown": float(summary["max_drawdown"]),
            "sortino": float(summary["sortino"]),
            **metrics,
        }
        records.append(record)
        weekly.insert(0, "max_new_entries_per_bar", int(cap))
        weekly_parts.append(weekly)
        summaries[str(cap)] = summary
        decisions.to_parquet(
            args.output_dir / f"decisions_cap{cap}.parquet",
            index=False,
            compression="zstd",
        )
        equity.to_parquet(
            args.output_dir / f"equity_cap{cap}.parquet",
            index=False,
            compression="zstd",
        )

    comparison = pd.DataFrame(records).sort_values(
        ["stable_week_ev_objective", "portfolio_objective", "net_ev_per_trade"],
        ascending=False,
        kind="stable",
    )
    winner = int(comparison.iloc[0]["max_new_entries_per_bar"])
    comparison["promoted"] = comparison["max_new_entries_per_bar"].eq(winner)
    comparison.to_csv(args.output_dir / "comparison.csv", index=False)
    pd.concat(weekly_parts, ignore_index=True).to_csv(
        args.output_dir / "weekly_comparison.csv", index=False
    )
    updated = _promote_cap(args.policy_config, winner, args.output_dir) if args.promote else []
    manifest = {
        "schema": "portfolio_new_entries_per_bar_ablation_v1",
        "exit_rows": str(args.exit_rows),
        "policy_config": str(args.policy_config),
        "portfolio_ev_reference": str(args.portfolio_ev_reference),
        "caps": sorted(set(args.caps)),
        "promotion_metric": (
            "mean_week_ev - 0.5*std_week_ev + 0.25*worst_week_ev"
        ),
        "winner": winner,
        "promoted": bool(args.promote),
        "updated_artifacts": updated,
        "summaries": summaries,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(comparison.to_string(index=False))
    print(json.dumps({"winner": winner, "updated_artifacts": updated}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
