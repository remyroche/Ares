#!/usr/bin/env python3
"""Evaluate M6 transport within causal market-state proxies across 2022 and 2023--24.

Regime thresholds are fitted from rows available before each scored era.  The
states use only broad-market 24-hour return and negative breadth, both present
in the older and newer candidate stores.  They are diagnostics, not model
inputs, so this audit cannot manufacture a regime-aware winner.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data_perp/artifacts/historical_m6_regime_transport_20260809_v1"
TOPS = (.01, .05)


def _regime(history: pd.DataFrame, scored: pd.DataFrame, ret: str, breadth: str) -> tuple[pd.Series, dict[str, float]]:
    r1, r2 = history[ret].quantile([1 / 3, 2 / 3]).to_numpy(float)
    b1, b2 = history[breadth].quantile([1 / 3, 2 / 3]).to_numpy(float)
    if not (r1 < r2 and b1 < b2):
        return pd.Series("support_insufficient", index=scored.index), {"ret_q33": r1, "ret_q67": r2, "breadth_q33": b1, "breadth_q67": b2}
    trend = np.select([scored[ret] <= r1, scored[ret] >= r2], ["risk_off", "risk_on"], default="neutral")
    state = np.select([scored[breadth] >= b2, scored[breadth] <= b1], ["broad_selloff", "benign_breadth"], default="mixed_breadth")
    combined = pd.Series(trend, index=scored.index).str.cat(pd.Series(state, index=scored.index), sep="|")
    return combined, {"ret_q33": r1, "ret_q67": r2, "breadth_q33": b1, "breadth_q67": b2}


def _top_rows(frame: pd.DataFrame, *, era: str, period: str, regime: str, score: str) -> list[dict[str, object]]:
    rows = []
    ordered = frame.sort_values([score, "candidate_id"], ascending=[False, True], kind="mergesort")
    for fraction in TOPS:
        take = ordered.head(max(1, int(np.ceil(len(ordered) * fraction))))
        rows.append({"era": era, "period": period, "regime": regime, "top_fraction": fraction, "n": len(take),
                     "all_rows": len(frame), "gross_bps": float(take.gross_bps.mean()), "net_bps": float(take.net_bps.mean())})
    return rows


def _newer() -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    root = ROOT / "data_perp/artifacts"
    pred = pd.read_parquet(root / "historical_2023_2024_r3_m6_rolling_20260809_v1/predictions.parquet")
    # Recover the complete strict base-OOF history to fit unsupervised causal
    # regime thresholds before each test fold.
    all_oof = []
    for side in ("long", "short"):
        for fold in range(4):
            x = pd.read_parquet(root / f"tp6_r3_r5_{side}_baseoof_fold{fold}_20260802_v1/base_oof_predictions.parquet")
            x["historical_fold"] = fold
            all_oof.append(x[["candidate_id", "__ts__", "historical_fold"]])
    all_oof = pd.concat(all_oof, ignore_index=True)
    needed = ["candidate_id", "mkt_ret_eq_24h", "negative_breadth_pct"]
    context = []
    ids = set(all_oof.candidate_id)
    for part in sorted((root / "full_universe_t2_t4_panel_20260801_v3/parts").glob("*.parquet")):
        x = pd.read_parquet(part, columns=needed)
        x = x[x.candidate_id.isin(ids)]
        if not x.empty: context.append(x)
    all_oof = all_oof.merge(pd.concat(context, ignore_index=True), on="candidate_id", how="inner", validate="one_to_one")
    pred = pred.merge(all_oof[["candidate_id", "mkt_ret_eq_24h", "negative_breadth_pct"]], on="candidate_id", how="inner", validate="one_to_one")
    rows, thresholds = [], []
    labelled = []
    for fold, test in pred.groupby("historical_fold", sort=True):
        prior = all_oof[all_oof.historical_fold < fold]
        state, cuts = _regime(prior, test, "mkt_ret_eq_24h", "negative_breadth_pct")
        test = test.copy(); test["regime"] = state; labelled.append(test)
        thresholds.append({"era": "2023_2024", "period": f"fold_{fold}", **cuts, "prior_rows": len(prior), "scored_rows": len(test)})
        for regime, x in test.groupby("regime", sort=True):
            rows += _top_rows(x, era="2023_2024", period=f"fold_{fold}", regime=regime, score="side_calibrated_score_bps")
    return pd.concat(labelled, ignore_index=True), rows, thresholds


def _older() -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    root = ROOT / "data_perp/artifacts"
    pred = pd.read_parquet(root / "historical_2022_r3_m6_rolling_20260809_v1/rolling_predictions.parquet")
    pred["__ts__"] = pd.to_datetime(pred["__ts__"], utc=True)
    labels = pd.concat([pd.read_parquet(p, columns=["__ts__", "market_median_ret_24h", "market_negative_breadth_24h"])
                        for p in sorted((root / "historical_2022_tp6_sl4_h12_20260809_v2/parts").glob("*.parquet"))], ignore_index=True)
    labels["__ts__"] = pd.to_datetime(labels["__ts__"], utc=True)
    rows, thresholds, labelled = [], [], []
    for month, test in pred.groupby("month", sort=True):
        start = test.__ts__.min()
        prior = labels[labels.__ts__ < start]
        state, cuts = _regime(prior, test, "market_median_ret_24h", "market_negative_breadth_24h")
        test = test.copy(); test["regime"] = state; labelled.append(test)
        thresholds.append({"era": "2022", "period": month, **cuts, "prior_rows": len(prior), "scored_rows": len(test)})
        test["gross_bps"] = test.net_bps + 100.
        for regime, x in test.groupby("regime", sort=True):
            rows += _top_rows(x, era="2022", period=month, regime=regime, score="m6_probability")
    return pd.concat(labelled, ignore_index=True), rows, thresholds


def main() -> None:
    if OUT.exists(): raise FileExistsError(OUT)
    newer, new_rows, new_cuts = _newer()
    older, old_rows, old_cuts = _older()
    OUT.mkdir(parents=True)
    pd.concat([older.assign(era="2022"), newer.assign(era="2023_2024")], ignore_index=True).to_parquet(OUT / "regime_scored_predictions.parquet", index=False)
    results = pd.DataFrame(old_rows + new_rows)
    results.to_parquet(OUT / "regime_transport_metrics.parquet", index=False)
    pd.DataFrame(old_cuts + new_cuts).to_parquet(OUT / "regime_thresholds.parquet", index=False)
    summary = results[results.top_fraction.eq(.01)].groupby(["era", "regime"], as_index=False).agg(periods=("period", "nunique"), rows=("all_rows", "sum"), net_bps=("net_bps", "mean"), worst_period_net_bps=("net_bps", "min"))
    summary.to_parquet(OUT / "regime_transport_summary.parquet", index=False)
    (OUT / "manifest.json").write_text(json.dumps({"status": "COMPLETED", "causal_regime_contract": "train-history quantiles of 24h broad-market return and negative breadth", "regime_is_inference_feature": False, "top_fractions": TOPS}, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__": main()
