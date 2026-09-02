#!/usr/bin/env python3
"""Research-only interaction check for the independently selected exit controls."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.exact_1m_h4_overlay_research import replay_exact_1m_gradual_h4_overlay
from scripts import run_strict_r3_p8u_v58_gradual_exit_grid as grid
from scripts.run_strict_r3_p8u_v58_gradual_exit_finalists import _frame, _load_context
from scripts.run_strict_r3_p8u_v58_e2_50_exact_matched import DEFAULT_PATH_ROOT, DEFAULT_POLICY

FINALISTS = ROOT / "data_perp/artifacts/strict_r3_p8u_v58_gradual_exit_finalists_20260830_v5"
STATE = ROOT / "data_perp/artifacts/strict_r3_p8u_v58_exact_h4_states_20260830_v1"
SCORE = ROOT / "data_perp/artifacts/strict_r3_p8u_v58_gradual_exit_grid_screen_a_20260830_v1/calibrated_h4_scores_target_free.parquet"
DEFAULT_OUT = ROOT / "data_perp/artifacts/strict_r3_p8u_v58_gradual_exit_composition_20260830_v1"


def _outcome(configs: list[dict[str, object]], ids: list[str]) -> pd.DataFrame:
    c = grid._WORKER_CONTEXT
    rows: list[dict[str, object]] = []
    for candidate in ids:
        route = c["route_by_id"][candidate]
        path = int(route["__path_index__"])
        schedule = c["schedule"].get(candidate, {})

        def modulator(state: dict[str, float], schedule=schedule, configs=configs) -> dict[str, float] | None:
            probability = schedule.get(pd.Timestamp(state["state_decision_ts"]))
            if probability is None:
                return None
            result = {"activation_multiplier": 1.0, "giveback_multiplier": 1.0, "sl_distance_multiplier": 1.0}
            key = {"activation": "activation_multiplier", "giveback": "giveback_multiplier", "stop": "sl_distance_multiplier"}
            for config in configs:
                partial = grid._multipliers(float(probability), config)
                result[key[str(config["control"])]] = partial[key[str(config["control"])]]
            return result

        allow_extension = any(str(x["control"]) == "stop" and str(x["mode"]) in {"extend", "both"} for x in configs)
        trace = replay_exact_1m_gradual_h4_overlay(
            entry_price=float(c["entry"][path]), signal_atr=float(c["atr"][path]), entry_ts=route["entry_ts"],
            highs=c["high"][path], lows=c["low"][path], closes=c["close"][path], params=c["params"],
            median_atr_fraction=float(c["median"]), mc1_expected_bps=float(route["bcf_mc1_expected_bps"]),
            state_modulator=modulator, allow_stop_extension=allow_extension, max_stop_loss_fraction=.05,
        )
        rows.append({"candidate_id": candidate, "exact_entry_price": float(c["entry"][path]),
            "exact_net_bps": float(trace["net_bps"]), "exact_gross_bps": float(trace["gross_bps"]),
            "exact_exit_ts": pd.Timestamp(trace["exit_timestamp"]), "exact_exit_price": float(trace["exit_price"]),
            "exact_exit_minute": int(trace["exit_minute"]), "exact_exit_reason": str(trace["exit_reason"])})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalists", type=Path, default=FINALISTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    winners = pd.read_parquet(args.finalists / "per_control_winners.parquet")
    by_control = {str(r.control): {key: r[key] for key in ("control", "mode", "extension_ratio", "threshold", "strength", "power")} for _, r in winners.iterrows()}
    route, parent = _load_context(STATE, DEFAULT_PATH_ROOT, DEFAULT_POLICY, SCORE)
    parent_frame = _frame(route, parent)
    variants = {
        "activation_giveback": [by_control["activation"], by_control["giveback"]],
        "activation_stop": [by_control["activation"], by_control["stop"]],
        "activation_giveback_stop": [by_control["activation"], by_control["giveback"], by_control["stop"]],
    }
    context = mp.get_context("fork")
    outcomes: dict[str, pd.DataFrame] = {}
    with ProcessPoolExecutor(max_workers=min(int(args.workers), len(variants)), mp_context=context) as pool:
        futures = {pool.submit(_outcome, configs, route["candidate_id"].tolist()): name for name, configs in variants.items()}
        for future in as_completed(futures):
            outcomes[futures[future]] = future.result()
    _, _, _, c0 = grid._portfolio(parent_frame, output=output, arm="C0_parent")
    summary = [{"arm": "C0_parent", **c0}]
    for name, configs in variants.items():
        frame = grid._replace_month(parent_frame, outcomes[name], pd.Timestamp("2026-05-01", tz="UTC"))
        _, _, accepted, metrics = grid._portfolio(frame, output=output, arm=name)
        accepted.to_parquet(output / f"{name}_accepted.parquet", index=False, compression="zstd")
        summary.append({"arm": name, "controls": json.dumps(configs, sort_keys=True), **metrics})
    result = pd.DataFrame(summary)
    for field in ("portfolio_accepted", "net_bps_per_trade", "total_net_bps", "max_drawdown", "worst_week", "sortino", "compounded_return"):
        result[f"delta_vs_parent_{field}"] = result[field] - c0[field]
    result.to_parquet(output / "composition_summary.parquet", index=False)
    (output / "manifest.json").write_text(json.dumps({
        "scope": "offline research-only composition check; no live modification", "selection": "individual components selected strictly from May-June screen and July holdout before this composition check", "route": "same full v58 exact source-valid Router50 dual-MC1>=50 +5m portfolio", "stop_cap": .05,
    }, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
