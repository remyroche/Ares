#!/usr/bin/env python3
"""Stress the parent and retained H4 continuation outcomes against adverse fills.

Historical full-depth books do not exist for this all-symbol period.  This is
therefore explicitly a conservative *15-minute-close* stress, not a VWAP
reconstruction: a long exit receives the worse of its ideal policy fill and
the closing price of its terminal 15-minute bar, then a fixed extra 0/10/30/50
bps exit friction is applied.  Entry, labels, portfolio state, and exit bars
remain unchanged.  Research only; no live-policy mutation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_strict_r3_p8u_15m_continuation_v2_advantage_ablation as stable
from scripts import run_strict_r3_p8u_15m_continuation_c1_ablation as c1


OUTCOMES = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_continuation_4m_matched_hpo_20260830_v1/H4_l1_d4_l15_leaf5_reg20_entry_outcomes.parquet"
STATE_PANEL = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_activation50_advantage_20260830_v1/activation50_advantage_states.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_continuation_execution_stress_20260830_v1"
SELECTION_END = pd.Timestamp("2026-08-01", tz="UTC")
STRESS_BPS = (0.0, 10.0, 30.0, 50.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _entry_context(state_path: Path) -> pd.DataFrame:
    states = pd.read_parquet(state_path, columns=["candidate_id", "__symbol__", "entry_decision_ts", "entry_price"])
    states["candidate_id"] = states.candidate_id.astype(str)
    states["entry_decision_ts"] = pd.to_datetime(states.entry_decision_ts, utc=True, errors="raise")
    first = states.sort_values(["candidate_id", "entry_decision_ts"], kind="stable").drop_duplicates("candidate_id", keep="first")
    return first


def _adverse_close_gross(row: pd.Series, bars: c1.CompactBars, *, gross_column: str, exit_bar_column: str) -> float:
    path = c1._bar_path(bars, pd.Timestamp(row.entry_decision_ts))
    if path is None:
        raise RuntimeError(f"missing exact 15m replay path for {row.candidate_id}")
    _, _, close = path
    index = int(row[exit_bar_column])
    if not 0 <= index < len(close):
        raise RuntimeError(f"terminal bar outside H12 path for {row.candidate_id}")
    ideal = float(row[gross_column])
    close_gross = (float(close[index]) / float(row.entry_price) - 1.0) * 10_000.0
    # Long exits must not be credited above the policy's ideal threshold fill.
    return min(ideal, close_gross)


def _scope_replays(detail: pd.DataFrame, arm: str, output: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for scope, frame in (
        ("selection_jun_jul", detail.loc[pd.to_datetime(detail.entry_decision_ts, utc=True).lt(SELECTION_END)].copy()),
        ("august_holdout", detail.loc[pd.to_datetime(detail.entry_decision_ts, utc=True).ge(SELECTION_END)].copy()),
        ("all_oos", detail),
    ):
        if frame.empty:
            continue
        metrics = stable._replay_portfolio(frame, f"{arm}__{scope}", output)
        metrics["model_arm"], metrics["evaluation_scope"] = arm, scope
        records.append(metrics)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", type=Path, default=OUTCOMES)
    parser.add_argument("--state-panel", type=Path, default=STATE_PANEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    outcome_path, state_path = args.outcomes.resolve(), args.state_panel.resolve()
    rows = pd.read_parquet(outcome_path)
    rows["candidate_id"] = rows.candidate_id.astype(str)
    rows["entry_decision_ts"] = pd.to_datetime(rows.entry_decision_ts, utc=True, errors="raise")
    if rows.candidate_id.duplicated().any():
        raise AssertionError("input continuation outcomes must be candidate-unique")
    rows = rows.merge(_entry_context(state_path), on=["candidate_id", "__symbol__", "entry_decision_ts"], how="inner", validate="one_to_one")
    if len(rows) == 0:
        raise RuntimeError("no exact entry contexts joined")
    output.mkdir(parents=True, exist_ok=False)
    cache: dict[str, c1.CompactBars | None] = {}
    parent_close: list[float] = []
    c50_close: list[float] = []
    for _, row in rows.iterrows():
        symbol = str(row.__symbol__)
        if symbol not in cache:
            cache[symbol] = c1._load_symbol_bars(symbol)
        bars = cache[symbol]
        if bars is None:
            raise RuntimeError(f"missing bars for {symbol}")
        parent_close.append(_adverse_close_gross(row, bars, gross_column="baseline_gross_bps", exit_bar_column="baseline_exit_bar"))
        c50_close.append(_adverse_close_gross(row, bars, gross_column="c1_gross_bps", exit_bar_column="c1_exit_bar"))
    rows["parent_adverse_close_gross_bps"] = parent_close
    rows["c50_adverse_close_gross_bps"] = c50_close
    rows["parent_ideal_gap_bps"] = rows.parent_adverse_close_gross_bps - rows.baseline_gross_bps
    rows["c50_ideal_gap_bps"] = rows.c50_adverse_close_gross_bps - rows.c1_gross_bps
    rows.to_parquet(output / "candidate_stress_paths.parquet", index=False, compression="zstd")

    summaries: list[dict[str, object]] = []
    for variant, gross, exit_bar, exit_reason in (
        ("P0_parent", "parent_adverse_close_gross_bps", "baseline_exit_bar", "baseline_exit_reason"),
        ("C50_direct", "c50_adverse_close_gross_bps", "c1_exit_bar", "c1_exit_reason"),
    ):
        for friction in STRESS_BPS:
            detail = rows.loc[:, ["candidate_id", "__symbol__", "entry_decision_ts", "baseline_net_bps", "baseline_gross_bps", "baseline_exit_bar", "baseline_exit_reason"]].copy()
            detail["c1_gross_bps"] = rows[gross]
            detail["c1_net_bps"] = rows[gross] - 100.0 - friction
            detail["c1_exit_bar"] = rows[exit_bar].astype(int)
            detail["c1_exit_reason"] = rows[exit_reason].astype(str)
            detail["model_calls"] = 0
            detail["action_calls"] = 0
            arm = f"{variant}_close_stress_{int(friction):02d}bps"
            detail.to_parquet(output / f"{arm}_entry_outcomes.parquet", index=False, compression="zstd")
            for metric in _scope_replays(detail, arm, output):
                metric["variant"], metric["extra_exit_friction_bps"] = variant, friction
                summaries.append(metric)
    summary = pd.DataFrame(summaries)
    summary["total_ev_per_abs_drawdown"] = summary.total_policy_net_bps / summary.max_drawdown.abs().replace(0.0, np.nan)
    parent = summary.loc[summary.variant.eq("P0_parent")].set_index(["evaluation_scope", "extra_exit_friction_bps"])
    for metric in ("portfolio_accepted", "policy_net_bps_per_trade", "total_policy_net_bps", "max_drawdown", "worst_week", "total_ev_per_abs_drawdown"):
        summary[f"delta_vs_parent_{metric}"] = summary.apply(lambda row: row[metric] - parent.loc[(row.evaluation_scope, row.extra_exit_friction_bps), metric], axis=1)
    summary.to_parquet(output / "portfolio_summary.parquet", index=False)
    coverage = {
        "candidates": len(rows),
        "parent_mean_close_gap_bps": float(rows.parent_ideal_gap_bps.mean()),
        "parent_p05_close_gap_bps": float(np.quantile(rows.parent_ideal_gap_bps, .05)),
        "c50_mean_close_gap_bps": float(rows.c50_ideal_gap_bps.mean()),
        "c50_p05_close_gap_bps": float(np.quantile(rows.c50_ideal_gap_bps, .05)),
    }
    pd.DataFrame([coverage]).to_parquet(output / "stress_path_summary.parquet", index=False)
    manifest = {
        "schema": "strict-r3-p8u-15m-continuation-execution-stress-v1",
        "scope": "offline stress test only; no live/canonical mutation",
        "outcomes": str(outcome_path), "outcomes_sha256": _sha256(outcome_path),
        "state_panel": str(state_path), "state_panel_sha256": _sha256(state_path),
        "stress": "worse-of-ideal-policy-fill-or-terminal-15m-close for long exits, then fixed extra exit friction",
        "extra_exit_friction_bps": list(STRESS_BPS), "cost": "existing 100-bps policy cost plus stress friction exactly once",
        "limitation": "not a historical full-depth VWAP reconstruction; designed as a conservative bar-close robustness bound",
        "selection_period": "June--July 2026; August is reporting-only",
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
