#!/usr/bin/env python3
"""Select and test gradual exact-exit finalists from immutable screen blocks.

The May--June coarse screen is used only to nominate top controls.  July is a
strict chronological portfolio-constrained selection holdout.  The selected
winner for each independent control is then replayed across the full common
May--July route.  This script is research-only and has no exchange imports.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_strict_r3_p8u_v58_gradual_exit_grid as grid
from scripts.run_strict_r3_p8u_v58_e2_50_exact_matched import (
    DEFAULT_PATH_ROOT,
    DEFAULT_POLICY,
    _load_policy,
)

DEFAULT_STATE = ROOT / "data_perp/artifacts/strict_r3_p8u_v58_exact_h4_states_20260830_v1"
DEFAULT_BLOCK_GLOB = "strict_r3_p8u_v58_gradual_exit_grid_screen_*_20260830_v1/coarse_grid_screen.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/strict_r3_p8u_v58_gradual_exit_finalists_20260830_v1"
SEED = 1729
JULY = pd.Timestamp("2026-07-01", tz="UTC")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_context(state_root: Path, path_root: Path, policy: Path, score_source: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    route = pd.read_parquet(state_root / "target_free_route.parquet")
    route["candidate_id"] = route["candidate_id"].astype(str)
    route["timestamp"] = pd.to_datetime(route["timestamp"], utc=True, errors="raise")
    route["entry_ts"] = pd.to_datetime(route["entry_ts"], utc=True, errors="raise")
    parent = pd.read_parquet(state_root / "exact_parent_outcomes.parquet")
    parent["candidate_id"] = parent["candidate_id"].astype(str)
    rows = pd.read_parquet(path_root / "valid_exact_paths_rows.parquet")
    index = pd.Series(np.arange(len(rows), dtype=np.int64), index=rows["candidate_id"].astype(str))
    route["__path_index__"] = route["candidate_id"].map(index)
    if route["__path_index__"].isna().any() or parent.candidate_id.nunique() != len(route):
        raise AssertionError("route, parent, and exact path identities must agree")
    archive = np.load(path_root / "exact_paths.npz", allow_pickle=False)
    params, median, _ = _load_policy(policy)
    scores = pd.read_parquet(score_source)
    scores["candidate_id"] = scores["candidate_id"].astype(str)
    scores["state_decision_ts"] = pd.to_datetime(scores["state_decision_ts"], utc=True, errors="raise")
    schedule = {
        candidate: {pd.Timestamp(ts): float(value) for ts, value in zip(group["state_decision_ts"], group["calibrated_h4_probability_positive"], strict=True)}
        for candidate, group in scores.groupby("candidate_id", sort=False)
    }
    grid._WORKER_CONTEXT = {
        "route_by_id": {str(row.candidate_id): row.to_dict() for _, row in route.iterrows()},
        "schedule": schedule,
        "entry": np.asarray(archive["entry"], dtype=float),
        "atr": np.asarray(archive["atr"], dtype=float),
        "high": np.asarray(archive["high"], dtype=np.float32),
        "low": np.asarray(archive["low"], dtype=np.float32),
        "close": np.asarray(archive["close"], dtype=np.float32),
        "params": params,
        "median": median,
    }
    return route, parent


def _run(config: dict[str, object], ids: list[str]) -> tuple[dict[str, object], pd.DataFrame]:
    return config, grid._run_candidates(config, ids)


def _parallel_outcomes(configs: list[dict[str, object]], ids: list[str], workers: int) -> dict[str, pd.DataFrame]:
    key = lambda c: json.dumps(c, sort_keys=True)
    context = mp.get_context("fork")
    results: dict[str, pd.DataFrame] = {}
    with ProcessPoolExecutor(max_workers=min(int(workers), len(configs)), mp_context=context) as pool:
        futures = [pool.submit(_run, config, ids) for config in configs]
        for future in as_completed(futures):
            config, outcome = future.result()
            results[key(config)] = outcome
    return results


def _frame(route: pd.DataFrame, parent: pd.DataFrame) -> pd.DataFrame:
    entry = grid._WORKER_CONTEXT["entry"]
    return grid._parent_frame(route, parent, entry)


def _july_metrics(parent_frame: pd.DataFrame, outcome: pd.DataFrame, output: Path, arm: str) -> dict[str, float]:
    frame = grid._replace_month(parent_frame, outcome, JULY)
    table, _, accepted, metrics = grid._portfolio(frame, output=output, arm=arm)
    return grid._accepted_metrics(table, accepted, start=JULY) | {
        "max_drawdown": float(metrics["max_drawdown"]),
        "sortino": float(metrics["sortino"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-glob", default=DEFAULT_BLOCK_GLOB)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--path-root", type=Path, default=DEFAULT_PATH_ROOT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-per-control", type=int, default=3)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    if args.top_per_control < 1:
        raise ValueError("top-per-control must be positive")

    screen_paths = sorted((ROOT / "data_perp/artifacts").glob(str(args.screen_glob)))
    if len(screen_paths) != 16:
        raise AssertionError(f"expected 16 immutable screen blocks, found {len(screen_paths)}")
    screen = pd.concat([pd.read_parquet(path) for path in screen_paths], ignore_index=True)
    key_columns = ["control", "mode", "extension_ratio", "threshold", "strength", "power"]
    if len(screen) != 432 or screen.duplicated(key_columns).any():
        raise AssertionError("screen does not cover the full unique 432-cell grid")
    screen.to_parquet(output / "combined_coarse_screen.parquet", index=False)
    finalists: list[dict[str, object]] = []
    for _, group in screen.groupby("control", sort=True):
        selected = group.sort_values(["screen_score", "screen_delta_bps_per_trade"], ascending=False, kind="stable").head(int(args.top_per_control))
        finalists.extend(selected.loc[:, key_columns].to_dict("records"))
    finalist_frame = pd.DataFrame(finalists)
    finalist_frame.to_parquet(output / "may_june_finalists.parquet", index=False)

    # All blocks must have the same strict prequential score schedule.
    score_sources = [path.parent / "calibrated_h4_scores_target_free.parquet" for path in screen_paths]
    hashes = {_sha256(path) for path in score_sources}
    if len(hashes) != 1:
        raise AssertionError("screen blocks do not share the same calibrated H4 schedule")
    route, parent = _load_context(args.state_root.resolve(), args.path_root.resolve(), args.policy.resolve(), score_sources[0])
    parent_frame = _frame(route, parent)
    july_ids = route.loc[route["timestamp"].ge(JULY), "candidate_id"].tolist()
    july_outcomes = _parallel_outcomes(finalists, july_ids, int(args.workers))
    parent_table, _, parent_accepted, parent_metrics = grid._portfolio(parent_frame, output=output, arm="C0_parent")
    parent_july = grid._accepted_metrics(parent_table, parent_accepted, start=JULY)
    rows: list[dict[str, object]] = []
    for number, config in enumerate(finalists):
        arm = f"july_finalist_{number:02d}_{config['control']}_{config['mode']}"
        observed = _july_metrics(parent_frame, july_outcomes[json.dumps(config, sort_keys=True)], output, arm)
        rows.append({**config, "arm": arm, **{f"july_{name}": value for name, value in observed.items()},
            "july_delta_net_bps_per_trade": observed["net_bps_per_trade"] - parent_july["net_bps_per_trade"],
            "july_delta_total_net_bps": observed["total_net_bps"] - parent_july["total_net_bps"]})
    holdout = pd.DataFrame(rows)
    holdout["holdout_score"] = holdout["july_delta_total_net_bps"] + 50.0 * holdout["july_delta_net_bps_per_trade"]
    holdout.to_parquet(output / "july_constrained_holdout.parquet", index=False)

    winners: list[dict[str, object]] = []
    for _, group in holdout.groupby("control", sort=True):
        winners.append(group.sort_values(["holdout_score", "july_delta_net_bps_per_trade"], ascending=False, kind="stable").iloc[0].loc[key_columns].to_dict())
    winner_frame = pd.DataFrame(winners)
    winner_frame.to_parquet(output / "per_control_winners.parquet", index=False)
    full_outcomes = _parallel_outcomes(winners, route["candidate_id"].tolist(), int(args.workers))

    summary: list[dict[str, object]] = [{"arm": "C0_parent", "control": "parent", **parent_metrics}]
    c0 = parent_metrics
    for config in winners:
        arm = f"winner_{config['control']}"
        frame = grid._replace_month(parent_frame, full_outcomes[json.dumps(config, sort_keys=True)], pd.Timestamp("2026-05-01", tz="UTC"))
        _, _, accepted, metrics = grid._portfolio(frame, output=output, arm=arm)
        accepted.to_parquet(output / f"{arm}_accepted.parquet", index=False, compression="zstd")
        summary.append({"arm": arm, **config, **metrics})
    output_summary = pd.DataFrame(summary)
    for field in ("portfolio_accepted", "net_bps_per_trade", "total_net_bps", "max_drawdown", "worst_week", "sortino", "compounded_return"):
        output_summary[f"delta_vs_parent_{field}"] = output_summary[field] - c0[field]
    output_summary.to_parquet(output / "full_exact_constrained_summary.parquet", index=False)
    (output / "run_manifest.json").write_text(json.dumps({
        "schema": "strict-r3-p8u-v58-gradual-exit-finalists-v1",
        "scope": "research only: causal exact-one-minute replay, no live execution changes",
        "screen": "May-June 432-cell balanced screen; top three independently nominated per control",
        "holdout": "July common source-valid route, normal global chronological portfolio constraints",
        "route": "Router50, dual BCF/current MC1 >=50, BCF priority, +5-minute entry",
        "calibrator": "frozen strict-prequential final-28-day isotonic H4 probability schedule shared by all screens",
        "full_result": "per-control July selected winner replayed May-July; components are not composed",
        "screen_blocks": [str(path.resolve()) for path in screen_paths],
        "policy": str(args.policy.resolve()), "policy_sha256": _sha256(args.policy.resolve()),
        "workers": int(args.workers), "seed": SEED,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # The desktop launcher can detach a long child from its terminal.
        # Persist a diagnostic receipt beside the immutable partial output so
        # a focused research repair never has to infer an exception from files.
        import traceback

        argv = sys.argv[1:]
        if "--output" in argv:
            candidate = Path(argv[argv.index("--output") + 1]).resolve()
            if candidate.exists():
                (candidate / "failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
