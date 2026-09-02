#!/usr/bin/env python3
"""Measure residual-score reliability conditional on historical transition state.

Read-only diagnostic: it does not fit or select a state-specific mapping.  It
establishes whether a future causal calibration ablation has enough stable
state-conditioned support to be worth running.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONTEXT = ROOT / "data_perp/artifacts/failure_2022_2023_pf_exact1m_transition_context_continuation_20260730_v1/context.parquet"
SCORES = ROOT / "data_perp/artifacts/reconstructed_base_residual_stack_2022_2024_20260730_v4/oof_scores.parquet"
SPINE = ROOT / "data_perp/artifacts/failure_2022_2023_pf_exact1m_transition_context_continuation_20260730_v1/frozen_v3_market_spine/hourly_transition_dataset.parquet"
OUT = ROOT / "data_perp/artifacts/historical_residual_transition_reliability_20260731_v1"
IDENTITY = ["candidate_id", "__ts__", "__symbol__", "side_name"]
SCORE = "score_residual_expected_ev"
TARGET = "execution_net_ev_12h"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write(path: Path, value: object) -> None:
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(partial, path)


def calibration(frame: pd.DataFrame) -> dict[str, float | int]:
    score = frame[SCORE].to_numpy(float)
    target = frame[TARGET].to_numpy(float)
    slope, intercept = np.polyfit(score, target, 1)
    return {"rows": int(len(frame)), "rank_ic": float(frame[SCORE].rank().corr(frame[TARGET].rank())), "mean_score_bps": float(score.mean() * 1e4), "mean_net_bps": float(target.mean() * 1e4), "slope": float(slope), "intercept_bps": float(intercept * 1e4), "mean_error_bps": float((score - target).mean() * 1e4), "rmse_bps": float(np.sqrt(np.mean((score - target) ** 2)) * 1e4)}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    context = pd.read_parquet(CONTEXT, columns=[*IDENTITY, "state_context__current_state", "state_context__switch_1h", "state_context__state_age_hours"])
    scores = pd.read_parquet(SCORES, columns=[*IDENTITY, SCORE, TARGET, "execution_gross_ev_12h", "execution_cost_return", "residual_is_oof"])
    spine = pd.read_parquet(SPINE, columns=["source_utc", "target__transition_active", "target__onset_within_3h"])
    for frame, time in ((context, "__ts__"), (scores, "__ts__"), (spine, "source_utc")):
        frame[time] = pd.to_datetime(frame[time], utc=True, errors="raise")
    if not scores.residual_is_oof.astype(bool).all():
        raise ValueError("non-OOF residual rows")
    if not np.allclose(scores.execution_gross_ev_12h - scores.execution_cost_return, scores[TARGET], atol=1e-12, rtol=0):
        raise ValueError("gross-cost net assertion")
    panel = context.merge(scores.drop(columns="residual_is_oof"), on=IDENTITY, how="inner", validate="one_to_one")
    panel = panel.merge(spine.rename(columns={"source_utc": "__ts__"}), on="__ts__", how="left", validate="many_to_one")
    if len(panel) != len(context) or panel[["target__transition_active", "target__onset_within_3h"]].isna().any().any():
        raise ValueError("incomplete exact context/score/spine identity")
    rank = panel.sort_values([SCORE, "candidate_id"], ascending=[False, True], kind="stable").index[:int(np.ceil(len(panel) * .1))]
    panel["selected_global_top10"] = panel.index.isin(rank)
    records = []
    for dimension in ("state_context__current_state", "target__transition_active", "target__onset_within_3h", "state_context__switch_1h"):
        for value, local in panel.groupby(dimension, sort=True):
            selected = local.loc[local.selected_global_top10]
            records.append({"dimension": dimension, "value": str(value), **calibration(local), "full_population_share": float(len(local) / len(panel)), "selected_rows": int(len(selected)), "selected_share": float(len(selected) / panel.selected_global_top10.sum()), "selected_net_bps": float(selected[TARGET].mean() * 1e4) if len(selected) else float("nan"), "selected_rank_ic": float(selected[SCORE].rank().corr(selected[TARGET].rank())) if len(selected) > 2 else float("nan")})
    deciles = []
    for state, local in panel.groupby("state_context__current_state", sort=True):
        local = local.copy()
        local["score_decile"] = pd.qcut(local[SCORE].rank(method="first"), 10, labels=False) + 1
        for decile, cell in local.groupby("score_decile", sort=True):
            deciles.append({"state": int(state), "decile": int(decile), "rows": int(len(cell)), "mean_score_bps": float(cell[SCORE].mean() * 1e4), "mean_net_bps": float(cell[TARGET].mean() * 1e4), "positive_fraction": float(cell[TARGET].gt(0).mean())})
    OUT.mkdir(parents=True)
    pd.DataFrame(records).to_csv(OUT / "state_reliability.csv", index=False)
    pd.DataFrame(deciles).to_csv(OUT / "state_score_deciles.csv", index=False)
    contract = {"scope": "read_only_historical_residual_reliability_diagnostic", "promotion_eligible": False, "selection": "one pooled global top10 once, before state splits", "interpretation": "state-specific metrics are diagnostic only; no state gate, quota or calibrator is selected", "economics": "exact frozen-policy H12 gross minus cost", "scores": "reconstructed residual expected EV is already OOF"}
    write(OUT / "contract.json", contract)
    manifest = {"schema": "historical_residual_transition_reliability_v1", "status": "COMPLETE_RESEARCH_ONLY", "rows": int(len(panel)), "sources": {str(p): sha(p) for p in (CONTEXT, SCORES, SPINE)}, "outputs_sha256": {p.name: sha(p) for p in (OUT / "state_reliability.csv", OUT / "state_score_deciles.csv", OUT / "contract.json")}, **contract}
    write(OUT / "manifest.json", manifest)
    (OUT / "manifest.sha256").write_text(f"{sha(OUT / 'manifest.json')}  manifest.json\n")


if __name__ == "__main__":
    main()
