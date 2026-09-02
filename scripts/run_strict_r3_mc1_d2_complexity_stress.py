#!/usr/bin/env python3
"""Offline complexity/seed stress test for frozen MC1_d2.

Only the static HGB geometry changes.  The six MC1 inputs, full-universe
day-balanced source, strict availability rule, frozen daily residual shift,
+50-bps admission, final-score auction, and portfolio contract remain fixed.
The producer writes compact metrics only; no live or model-bundle state changes.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_strict_r3_mc1_d2_controlled_portfolio import (  # noqa: E402
    CAUSAL_AUCTION_CURVE,
    _candidate_table,
    _metrics,
    _params,
)
from scripts.run_strict_r3_mc1_d2_historical_parity import (  # noqa: E402
    CORE,
    day_balanced,
    history,
    utc,
)
from extreme_price_movements.portfolio_policy_replay import replay_candidates  # noqa: E402


ARMS = (
    ("d1_seed1729_leaf100", 1, 1729, 100),
    ("d2_seed0017_leaf100", 2, 17, 100),
    ("d2_seed2718_leaf100", 2, 2718, 100),
    ("d2_seed1729_leaf200", 2, 1729, 200),
    ("d3_seed1729_leaf100", 3, 1729, 100),
)


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, depth: int, seed: int, leaf: int) -> np.ndarray:
    medians = train.loc[:, CORE].apply(pd.to_numeric, errors="coerce").median(numeric_only=True)
    x = train.loc[:, CORE].apply(pd.to_numeric, errors="coerce").fillna(medians)
    y = pd.to_numeric(train.net, errors="coerce")
    if len(x) > 50_000:
        take = x.sample(50_000, random_state=seed).index
        x, y = x.loc[take], y.loc[take]
    model = HistGradientBoostingRegressor(
        max_depth=depth, max_iter=80, learning_rate=.04, l2_regularization=20.0,
        min_samples_leaf=leaf, random_state=seed,
    ).fit(x, y)
    z = test.loc[:, CORE].apply(pd.to_numeric, errors="coerce").fillna(medians)
    return np.asarray(model.predict(z), dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--control-predictions", type=Path, required=True,
                        help="strict MC1 control; supplies the frozen daily residual shift")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-08-01")
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    columns = list(dict.fromkeys([
        "candidate_id", "__decision_ts__", "__symbol__", "side_name",
        "policy_label_available_ts", "policy_path_valid", "policy_net_bps",
        "policy_gross_bps", "policy_exit_bar_15m", "policy_entry_price",
        "policy_exit_price", "policy_exit_reason", "final_score", *CORE,
    ]))
    full = pd.read_parquet(args.ledger, columns=columns)
    full["__decision_ts__"] = pd.to_datetime(full["__decision_ts__"], utc=True)
    full["policy_label_available_ts"] = pd.to_datetime(full["policy_label_available_ts"], utc=True)
    if not full.side_name.astype(str).str.lower().eq("long").all():
        raise ValueError("MC1 complexity stress test is long-only")
    full["day"] = full.__decision_ts__.dt.normalize()
    source = day_balanced(full)
    control = pd.read_parquet(args.control_predictions, columns=["candidate_id", "recent_shift_bps"])
    score = full.merge(control, on="candidate_id", how="inner", validate="one_to_one")
    if len(score) != len(control):
        raise ValueError("control prediction and source identity mismatch")
    policy = full.loc[:, [
        "candidate_id", "policy_path_valid", "policy_net_bps", "policy_gross_bps",
        "policy_exit_bar_15m", "policy_entry_price", "policy_exit_price", "policy_exit_reason",
    ]].copy()
    start, end = utc(args.start), utc(args.end)
    starts = pd.date_range(start, end, freq="MS", inclusive="left", tz="UTC")
    metrics: list[dict[str, object]] = []
    for arm, depth, seed, leaf in ARMS:
        pieces: list[pd.DataFrame] = []
        for fold in starts:
            stop = fold + pd.offsets.MonthBegin(1)
            train = history(source, fold, None, inclusive=False)
            test = score.loc[score.__decision_ts__.between(fold, stop, inclusive="left")].copy()
            if len(train) < 5_000 or test.empty:
                continue
            test["static_expected_bps"] = _fit_predict(train, test, depth, seed, leaf)
            test["mc1_expected_bps"] = test.static_expected_bps + pd.to_numeric(test.recent_shift_bps, errors="coerce")
            pieces.append(test)
            del train, test
            gc.collect()
        prediction = pd.concat(pieces, ignore_index=True)
        for year in (2025, 2026):
            part = prediction.loc[prediction.__decision_ts__.dt.year.eq(year)].copy()
            candidates = _candidate_table(part, policy, 50.0)
            decisions, equity, _ = replay_candidates(
                candidates, _params(), mode="global_auction", ev_curve=CAUSAL_AUCTION_CURVE,
                market_mode="perps", initial_wallet=1000.0,
            )
            if decisions.empty:
                decisions = decisions.copy()
                decisions["policy_outcome_available"] = pd.Series(dtype=bool)
            else:
                lookup = candidates.loc[:, ["candidate_id", "policy_outcome_available"]].reset_index(drop=True)
                lookup.index.name = "candidate_index"
                decisions = decisions.merge(lookup, on="candidate_index", how="left", validate="many_to_one")
            row = _metrics(decisions, equity, arm, str(year))
            row.update({"max_depth": depth, "seed": seed, "min_samples_leaf": leaf})
            metrics.append(row)
        print(json.dumps({"event": "arm_complete", "arm": arm}), flush=True)
        del prediction
        gc.collect()
    pd.DataFrame(metrics).to_parquet(args.out_dir / "complexity_stress_metrics.parquet", index=False)
    (args.out_dir / "run_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_mc1_d2_complexity_stress_v1", "status": "complete",
        "purpose": "static-HGB architecture falsification only",
        "arms": [dict(name=a, max_depth=d, seed=s, min_samples_leaf=l) for a, d, s, l in ARMS],
        "fixed": {
            "features": list(CORE), "training": "full-universe day-balanced 50k cap; monthly strict prequential refit",
            "shift": "frozen strict daily 21d residual shift", "admission": "+50 bps", "auction": "final_score",
        },
        "exclusions": ["live state", "exchange I/O", "upstream model changes"],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
