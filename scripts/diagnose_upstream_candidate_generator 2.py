#!/usr/bin/env python3
"""Audit why the historical directional candidate stream is economically adverse.

This is descriptive only: it joins the preserved candidate-stage score and
archetype fields to exact H12 outcomes.  It deliberately does not fit a new
model or use any outcome-derived value as a candidate input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "data_perp/artifacts/controlled_target_supportive_prepared_ledger_20260801_v5/prepared_target_supportive_ledger.parquet"
DEFAULT_STAGE = ROOT / "data_perp/artifacts/failure_2024_transition_exact1m_request_stage_20260730_v2/staged_candidates.parquet"


def _bin(values: pd.Series, bins: int = 10) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    ranks = numeric.rank(method="first", pct=True)
    return pd.cut(ranks, bins=np.linspace(0.0, 1.0, bins + 1), labels=[f"Q{i:02d}" for i in range(1, bins + 1)], include_lowest=True)


def _summary(frame: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    return (
        frame.groupby(group, observed=True)
        .agg(rows=("candidate_id", "size"), gross_bps=("gross_bps", "mean"), net_bps=("net_bps", "mean"), favorable_rate=("favorable_first", "mean"), adverse_rate=("adverse_first", "mean"))
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    columns = [
        "candidate_id", "__ts__", "__symbol__", "side_name", "oof_fold", "execution_gross_ev_12h", "execution_net_ev_12h",
        "favorable_first", "adverse_first", "mkt_ret_24h", "mkt_rv_ratio_1h_24h", "market_breadth_24h", "atr_percentile",
    ]
    ledger = pd.read_parquet(args.ledger, columns=columns)
    ledger = ledger.loc[ledger["oof_fold"].eq("meta_train")].copy()
    stage = pd.read_parquet(args.stage)
    fields = ["candidate_id", "base_score", "score_meta_base_soft_label", "historical_rank", "archetype_policy_key", "selected_for_monitor", "signal_timestamp", "decision_timestamp"]
    stage = stage.loc[:, fields]
    if stage.candidate_id.duplicated().any():
        raise ValueError("preserved stage has duplicate candidate IDs")
    work = ledger.merge(stage, on="candidate_id", how="left", validate="one_to_one", indicator=True)
    if not work["_merge"].eq("both").all():
        raise ValueError(f"stage join missing {(~work['_merge'].eq('both')).sum()} candidate rows")
    work = work.drop(columns="_merge")
    work["gross_bps"] = pd.to_numeric(work.execution_gross_ev_12h, errors="coerce") * 1e4
    work["net_bps"] = pd.to_numeric(work.execution_net_ev_12h, errors="coerce") * 1e4
    work["signal_timestamp"] = pd.to_datetime(work.signal_timestamp, utc=True)
    work["decision_timestamp"] = pd.to_datetime(work.decision_timestamp, utc=True)
    work["entry_lag_hours"] = (work.decision_timestamp - work.signal_timestamp).dt.total_seconds() / 3600.0
    work["base_score_decile"] = work.groupby("side_name", observed=True)["base_score"].transform(_bin)
    work["historical_rank_decile"] = work.groupby("side_name", observed=True)["historical_rank"].transform(_bin)
    work["market_return_quartile"] = _bin(work["mkt_ret_24h"], bins=4)
    work["market_vol_quartile"] = _bin(work["mkt_rv_ratio_1h_24h"], bins=4)
    work["breadth_quartile"] = _bin(work["market_breadth_24h"], bins=4)
    summaries = {
        "by_side": _summary(work, ["side_name"]),
        "by_source_score_decile_side": _summary(work, ["side_name", "base_score_decile"]),
        "by_historical_rank_decile_side": _summary(work, ["side_name", "historical_rank_decile"]),
        "by_archetype_side": _summary(work, ["side_name", "archetype_policy_key"]),
        "by_market_return_quartile_side": _summary(work, ["side_name", "market_return_quartile"]),
        "by_market_vol_quartile_side": _summary(work, ["side_name", "market_vol_quartile"]),
        "by_breadth_quartile_side": _summary(work, ["side_name", "breadth_quartile"]),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in summaries.items():
        table.to_csv(args.out_dir / f"{name}.csv", index=False)
        table.to_parquet(args.out_dir / f"{name}.parquet", index=False)
    paired = work.pivot_table(index=["__ts__", "__symbol__"], columns="side_name", values="gross_bps", aggfunc="first")
    paired = paired.dropna(subset=["long", "short"])
    paired["long_minus_short_bps"] = paired["long"] - paired["short"]
    paired["long_plus_short_bps"] = paired["long"] + paired["short"]
    paired.to_parquet(args.out_dir / "paired_direction_counterfactuals.parquet")
    correlation = (
        work.groupby("side_name", observed=True)
        .apply(lambda x: pd.Series({"base_score_vs_gross_spearman": x.base_score.corr(x.gross_bps, method="spearman"), "historical_rank_vs_gross_spearman": x.historical_rank.corr(x.gross_bps, method="spearman")} ), include_groups=False)
        .reset_index()
    )
    correlation.to_csv(args.out_dir / "source_score_correlations.csv", index=False)
    report = {
        "schema": "upstream_candidate_generator_diagnostic_v1",
        "scope": "2024-04..07 meta-train candidate-conditioned population only",
        "rows": int(len(work)),
        "source_stage_join_exact": True,
        "all_selected_for_monitor": bool(work.selected_for_monitor.fillna(False).all()),
        "entry_lag_hours_unique": sorted(map(float, work.entry_lag_hours.dropna().unique())),
        "paired_timestamp_symbol_rows": int(len(paired)),
        "paired_long_gross_bps": float(paired["long"].mean()),
        "paired_short_gross_bps": float(paired["short"].mean()),
        "paired_long_plus_short_gross_bps": float(paired["long_plus_short_bps"].mean()),
        "correlations": correlation.to_dict(orient="records"),
        "tables": {name: str((args.out_dir / f"{name}.csv").resolve()) for name in summaries},
        "interpretation_limit": "The stage contains selected candidate rows, not the rejected base-universe rows; this identifies failure modes within the selected stream but cannot estimate the original selector's acceptance curve without its full scored population.",
    }
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
