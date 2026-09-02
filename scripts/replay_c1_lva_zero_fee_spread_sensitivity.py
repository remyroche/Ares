#!/usr/bin/env python3
"""Replay fixed C1-LVA portfolio selections under empirical spread scenarios.

This is a transaction-cost sensitivity, not a re-selection experiment.  It
keeps the sealed target-free C1-LVA admissions and constrained portfolio entry
identities fixed, then replaces the parent replay's fixed 100-bps cost with:

``exact 1m gross outcome - one empirical full bid/ask spread``.

For a long entered at the ask and exited at the bid, the entry half-spread plus
the exit half-spread is one full spread.  No fee, impact, or extra slippage is
added in these scenarios.  The exact five-minute entry delay and one-minute
exit path remain embedded in ``exact_gross_bps``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPLAY = ROOT / "data_perp/artifacts/p8u_c1_lva_vs_core_exact1m_parent_mayjul_20260901_v9_all_active_sources_clean"
DEFAULT_SPREAD_ROOT = ROOT / "data_perp/exchanges/krakenfutures/spread_snapshots"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/c1_lva_zero_fee_per_asset_spread_sensitivity_20260901_v4"
PERCENTILES = (0.50, 0.60, 0.70, 0.80)
INITIAL_WALLET = 10_000.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_spread(path: Path) -> tuple[Path, pd.DataFrame | None, str | None]:
    try:
        frame = pd.read_parquet(path, columns=["observed_ts", "symbol", "spread_bps"])
    except Exception as exc:  # source integrity is audited below, never imputed
        return path, None, f"{type(exc).__name__}: {exc}"
    return path, frame, None


def _read_spreads(paths: Iterable[Path], workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = sorted(paths)
    if not ordered:
        raise FileNotFoundError("no Kraken spread-training parquet files found")
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        reads = list(pool.map(_read_spread, ordered))
    audit = pd.DataFrame([
        {"source_file": path.name, "readable": frame is not None, "error": error}
        for path, frame, error in reads
    ])
    frames = [frame for _path, frame, _error in reads if frame is not None]
    if not frames:
        raise RuntimeError("no readable Kraken spread-training parquet files")
    result = pd.concat(frames, ignore_index=True)
    result["observed_ts"] = pd.to_datetime(result["observed_ts"], utc=True, errors="coerce")
    result["symbol"] = result["symbol"].astype(str)
    result["spread_bps"] = pd.to_numeric(result["spread_bps"], errors="coerce")
    result = result.loc[
        result["observed_ts"].notna()
        & result["symbol"].ne("")
        & np.isfinite(result["spread_bps"])
        & result["spread_bps"].gt(0.0)
    ].copy()
    # Several collection jobs can write the same source timestamp.  Its
    # median preserves the actual observation without letting duplicate files
    # alter an empirical asset percentile.
    result = result.groupby(["symbol", "observed_ts"], as_index=False, sort=True)["spread_bps"].median()
    return result, audit


def _asset_quantiles(spreads: pd.DataFrame) -> pd.DataFrame:
    grouped = spreads.groupby("symbol", sort=True)["spread_bps"]
    result = grouped.agg(spread_observations="count", spread_min_bps="min", spread_max_bps="max").reset_index()
    for percentile in PERCENTILES:
        label = f"spread_p{int(percentile * 100):02d}_bps"
        result[label] = grouped.quantile(percentile).to_numpy(float)
    dates = spreads.groupby("symbol", sort=True)["observed_ts"].agg(["min", "max"]).reset_index()
    return result.merge(dates, on="symbol", how="left", validate="one_to_one").rename(
        columns={"min": "spread_first_observed_ts", "max": "spread_last_observed_ts"}
    )


def _realize_wallet(trades: pd.DataFrame, *, net_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay the original accepted identities with their fixed 70% notional rule.

    Entry selection/capacity is deliberately held fixed.  Each entry keeps its
    recorded constrained ``position_size / wallet_before`` fraction (normally
    70%, lower only when the historical wallet-cap bound was binding).  PnL
    changes as the synthetic fee/spread cost changes.  This is therefore a
    clean *cost* sensitivity rather than a hidden re-optimisation.
    """
    work = trades.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="raise")
    work["position_exit_timestamp"] = pd.to_datetime(work["position_exit_timestamp"], utc=True, errors="raise")
    work = work.sort_values(["timestamp", "candidate_id"], kind="stable").reset_index(drop=True)
    work["recorded_entry_fraction"] = work["position_size"].to_numpy(float) / work["wallet_before"].to_numpy(float)
    if (work["recorded_entry_fraction"] <= 0.0).any() or (work["recorded_entry_fraction"] > .70 + 1e-12).any():
        raise ValueError("accepted portfolio has an invalid constrained entry fraction")

    wallet = INITIAL_WALLET
    high_water = wallet
    pending: list[dict[str, object]] = []
    realized: list[dict[str, object]] = []
    enriched: list[dict[str, object]] = []

    def settle(through: pd.Timestamp) -> None:
        nonlocal wallet, high_water
        due = sorted((item for item in pending if item["exit_ts"] <= through), key=lambda item: (item["exit_ts"], item["candidate_id"]))
        if not due:
            return
        due_ids = {str(item["candidate_id"]) for item in due}
        pending[:] = [item for item in pending if str(item["candidate_id"]) not in due_ids]
        for item in due:
            pnl = float(item["notional"]) * float(item["net_bps"]) / 10_000.0
            wallet += pnl
            high_water = max(high_water, wallet)
            realized.append({
                "exit_timestamp": item["exit_ts"], "candidate_id": item["candidate_id"],
                "wallet": wallet, "realized_pnl": pnl,
                "realized_drawdown": wallet / high_water - 1.0,
            })

    for timestamp, group in work.groupby("timestamp", sort=True):
        settle(timestamp)
        entry_wallet = wallet
        for row in group.itertuples(index=False):
            record = row._asdict()
            notional = entry_wallet * float(record["recorded_entry_fraction"])
            record["scenario_entry_wallet"] = entry_wallet
            record["scenario_notional"] = notional
            record["scenario_pnl"] = notional * float(record[net_column]) / 10_000.0
            enriched.append(record)
            pending.append({
                "exit_ts": record["position_exit_timestamp"], "candidate_id": record["candidate_id"],
                "notional": notional, "net_bps": float(record[net_column]),
            })
    settle(pd.Timestamp.max.tz_localize("UTC"))
    trade_frame = pd.DataFrame(enriched)
    realized_frame = pd.DataFrame(realized).sort_values(["exit_timestamp", "candidate_id"], kind="stable")
    return trade_frame, realized_frame


def _metrics(trades: pd.DataFrame, realized: pd.DataFrame, *, scenario: str, net_column: str) -> dict[str, object]:
    net = trades[net_column].to_numpy(float)
    total_pnl = float(trades["scenario_pnl"].sum())
    final_wallet = INITIAL_WALLET + total_pnl
    daily = trades.assign(decision_day=pd.to_datetime(trades["timestamp"], utc=True).dt.normalize()).groupby("decision_day", sort=True)[net_column].mean()
    downside = daily.loc[daily.lt(0.0)].to_numpy(float)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside)))) if len(downside) else 0.0
    sortino = float(daily.mean() / downside_deviation) if downside_deviation > 0 else np.nan
    max_drawdown = float(realized["realized_drawdown"].min()) if len(realized) else 0.0
    return {
        "scenario": scenario,
        "trades": int(len(trades)),
        "mean_net_bps_per_trade": float(np.mean(net)),
        "median_net_bps_per_trade": float(np.median(net)),
        "notional_weighted_net_bps": float(np.average(net, weights=trades["scenario_notional"].to_numpy(float))),
        "total_net_bps": float(np.sum(net)),
        "win_rate": float(np.mean(net > 0.0)),
        "gross_policy_total_bps": float(np.sum(trades["exact_gross_bps"].to_numpy(float))),
        "spread_cost_total_bps": float(np.sum(trades["exact_gross_bps"].to_numpy(float) - net)),
        "portfolio_net_pnl_quote": total_pnl,
        "portfolio_final_wallet": final_wallet,
        "portfolio_compounded_return": final_wallet / INITIAL_WALLET - 1.0,
        "realized_wallet_max_drawdown": max_drawdown,
        "daily_sortino_bps": sortino,
        "trades_per_decision_day": float(len(trades) / max(int(daily.size), 1)),
        "worst_decision_day_mean_net_bps": float(daily.min()) if len(daily) else np.nan,
    }


def _period_metrics(trades: pd.DataFrame, *, scenario: str, net_column: str, frequency: str) -> pd.DataFrame:
    frame = trades.copy()
    frame["period"] = pd.to_datetime(frame["timestamp"], utc=True).dt.to_period(frequency).astype(str)
    result = frame.groupby("period", sort=True).agg(
        trades=("candidate_id", "size"),
        net_bps_per_trade=(net_column, "mean"),
        total_net_bps=(net_column, "sum"),
        net_pnl_quote=("scenario_pnl", "sum"),
        win_rate=(net_column, lambda values: float((values > 0.0).mean())),
        mean_spread_cost_bps=("scenario_spread_cost_bps", "mean"),
    ).reset_index()
    result.insert(0, "scenario", scenario)
    result.insert(1, "frequency", frequency)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--spread-root", type=Path, default=DEFAULT_SPREAD_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    replay_root, spread_root, output = args.replay_root.resolve(), args.spread_root.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output must be immutable: {output}")

    accepted_path = replay_root / "C1_LVA_refit_core_plus_causal_sr_portfolio_accepted.parquet"
    outcome_path = replay_root / "exact_1m_rich_parent_outcomes.parquet"
    summary_path = replay_root / "portfolio_summary.parquet"
    accepted = pd.read_parquet(accepted_path)
    outcomes = pd.read_parquet(outcome_path)
    trades = accepted.merge(
        outcomes.loc[:, ["candidate_id", "exact_gross_bps", "exact_net_bps", "exact_exit_ts", "exact_exit_reason"]],
        on="candidate_id", how="inner", validate="one_to_one",
    )
    if len(trades) != len(accepted):
        raise AssertionError("an accepted C1-LVA trade lacks an exact one-minute outcome")
    if not np.allclose(trades["net_bps"].to_numpy(float), trades["exact_net_bps"].to_numpy(float), atol=1e-9, rtol=0.0):
        raise AssertionError("accepted C1 net bps does not match the exact one-minute outcome")
    if not np.allclose(trades["exact_gross_bps"].to_numpy(float) - trades["exact_net_bps"].to_numpy(float), 100.0, atol=1e-8, rtol=0.0):
        raise AssertionError("parent replay cost is not the expected single 100-bps fixed cost")

    files = sorted(spread_root.glob("*spread_training*.parquet"))
    spreads, spread_source_audit = _read_spreads(files, args.workers)
    quantiles = _asset_quantiles(spreads)
    merged = trades.merge(quantiles, on="symbol", how="left", validate="many_to_one")
    if merged["spread_observations"].isna().any():
        missing = sorted(merged.loc[merged["spread_observations"].isna(), "symbol"].unique())
        raise ValueError(f"accepted C1 trades lack per-asset spread coverage: {missing[:10]}")

    output.mkdir(parents=True, exist_ok=False)
    spread_source_audit.to_parquet(output / "spread_source_audit.parquet", index=False, compression="zstd")
    quantiles.to_parquet(output / "per_asset_spread_quantiles.parquet", index=False, compression="zstd")
    metrics: list[dict[str, object]] = []
    monthly: list[pd.DataFrame] = []
    daily: list[pd.DataFrame] = []
    trade_scenarios: list[pd.DataFrame] = []
    scenarios: dict[str, pd.Series] = {"parent_fixed_100bps": merged["exact_net_bps"]}
    for percentile in PERCENTILES:
        name = f"zero_fee_asset_spread_p{int(percentile * 100):02d}"
        scenarios[name] = merged["exact_gross_bps"] - merged[f"spread_p{int(percentile * 100):02d}_bps"]

    for name, values in scenarios.items():
        scenario = merged.copy()
        scenario["scenario_net_bps"] = values.to_numpy(float)
        scenario["scenario_spread_cost_bps"] = scenario["exact_gross_bps"] - scenario["scenario_net_bps"]
        enriched, realized = _realize_wallet(scenario, net_column="scenario_net_bps")
        metrics.append(_metrics(enriched, realized, scenario=name, net_column="scenario_net_bps"))
        monthly.append(_period_metrics(enriched, scenario=name, net_column="scenario_net_bps", frequency="M"))
        daily.append(_period_metrics(enriched, scenario=name, net_column="scenario_net_bps", frequency="D"))
        columns = [
            "candidate_id", "timestamp", "symbol", "exact_exit_ts", "exact_exit_reason",
            "exact_gross_bps", "exact_net_bps", "scenario_net_bps", "scenario_spread_cost_bps",
            "scenario_notional", "scenario_pnl", "spread_observations", "spread_p50_bps",
            "spread_p60_bps", "spread_p70_bps", "spread_p80_bps",
        ]
        rendered = enriched.loc[:, columns].copy()
        rendered.insert(0, "scenario", name)
        trade_scenarios.append(rendered)

    parent = next(item for item in metrics if item["scenario"] == "parent_fixed_100bps")
    summary = pd.read_parquet(summary_path)
    reference = summary.loc[summary["arm"].eq("C1_LVA_refit_core_plus_causal_sr")]
    if len(reference) != 1:
        raise AssertionError("exact C1 parent portfolio summary is not uniquely identifiable")
    expected_wallet = float(reference.iloc[0]["final_wallet"])
    if not np.isclose(float(parent["portfolio_final_wallet"]), expected_wallet, atol=1e-5, rtol=0.0):
        raise AssertionError("fixed-selection cost replay does not reproduce the sealed parent wallet")

    pd.DataFrame(metrics).to_parquet(output / "scenario_metrics.parquet", index=False, compression="zstd")
    pd.concat(monthly, ignore_index=True).to_parquet(output / "monthly_metrics.parquet", index=False, compression="zstd")
    pd.concat(daily, ignore_index=True).to_parquet(output / "daily_metrics.parquet", index=False, compression="zstd")
    pd.concat(trade_scenarios, ignore_index=True).to_parquet(output / "trade_sensitivity.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "c1_lva_zero_fee_per_asset_spread_sensitivity_v1",
        "scope": "fixed target-free C1-LVA admissions and fixed portfolio entry identities; transaction-cost sensitivity only",
        "replay_root": str(replay_root.relative_to(ROOT)),
        "accepted_sha256": _sha256(accepted_path),
        "outcome_sha256": _sha256(outcome_path),
        "portfolio_summary_sha256": _sha256(summary_path),
        "spread_root": str(spread_root.relative_to(ROOT)),
        "spread_files": len(files),
        "spread_files_readable": int(spread_source_audit["readable"].sum()),
        "spread_files_unreadable": int((~spread_source_audit["readable"]).sum()),
        "spread_rows_deduplicated": int(len(spreads)),
        "spread_observed_start": spreads["observed_ts"].min().isoformat(),
        "spread_observed_end": spreads["observed_ts"].max().isoformat(),
        "trade_count": int(len(merged)),
        "selected_symbol_count": int(merged["symbol"].nunique()),
        "selected_min_asset_spread_observations": int(merged["spread_observations"].min()),
        "selected_median_asset_spread_observations": float(merged["spread_observations"].median()),
        "formula": "zero_fee_net_bps = exact_gross_bps - per_asset_full_bid_ask_spread_percentile_bps",
        "cost_definition": "one full bid/ask spread = entry half-spread plus exit half-spread; zero fees, zero additional impact, zero additional slippage",
        "execution_path": "the source exact_gross_bps already embeds decision+5m entry and one-minute rich-policy exit",
        "portfolio": "accepted identities/capacity are held fixed; each trade reuses its recorded constrained entry-notional fraction and is wallet-scaled without reoptimising admission or auction",
        "percentiles": list(PERCENTILES),
        "no_outcome_selection": True,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
