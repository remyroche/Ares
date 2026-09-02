#!/usr/bin/env python3
"""Select a research-only B/E/T rank blend from strict-OOF counterparts.

All three input ledgers must be generated on exactly the same routed candidate
identities and blocked held months.  Scores are converted to deterministic
timestamp-local percentile ranks before blending: a raw LambdaRank B score and
the two Huber supportive-head scores do not share a meaningful absolute scale.

This is a model-selection diagnostic, not a final untouched evaluation.  It
does not train a model, alter any downstream consumer, or write an inference,
consensus, MC1, admission, portfolio, or execution artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_strict_r3_base_stability_selector_v2 import IDENTITY, _rank01, _timestamp_metrics
from run_strict_r3_direct_head_crossyear_hpo_v1 import _objective_value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _exclusive(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _read_head(path: Path, head: str) -> pd.DataFrame:
    required = [*IDENTITY, "held_month", "head_score"]
    frame = pd.read_parquet(path, columns=required).copy()
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    frame["held_month"] = frame["held_month"].astype(str)
    frame["head_score"] = pd.to_numeric(frame["head_score"], errors="coerce")
    if frame.duplicated(list(IDENTITY)).any():
        raise AssertionError(f"{head}: duplicate strict-OOF identities")
    if not np.isfinite(frame["head_score"]).all():
        raise AssertionError(f"{head}: non-finite strict-OOF scores")
    return frame.rename(columns={"head_score": f"{head}_score", "held_month": f"{head}_held_month"})


def _read_policy(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["candidate_id", "policy_path_valid", "policy_net_bps"]).copy()
    frame["policy_path_valid"] = frame["policy_path_valid"].fillna(False).astype(bool)
    frame["policy_net_bps"] = pd.to_numeric(frame["policy_net_bps"], errors="coerce")
    if frame.candidate_id.duplicated().any():
        raise AssertionError("canonical policy labels contain duplicate candidate IDs")
    return frame


def _build(b_path: Path, e_path: Path, t_path: Path, policy_path: Path) -> pd.DataFrame:
    b = _read_head(b_path, "B")
    e = _read_head(e_path, "E")
    t = _read_head(t_path, "T")
    base = b.merge(e, on=list(IDENTITY), how="inner", validate="one_to_one")
    base = base.merge(t, on=list(IDENTITY), how="inner", validate="one_to_one")
    if len(base) != len(b) or len(base) != len(e) or len(base) != len(t):
        raise AssertionError("B/E/T OOF ledgers do not cover exactly the same candidate identities")
    if not (base.B_held_month.eq(base.E_held_month) & base.B_held_month.eq(base.T_held_month)).all():
        raise AssertionError("B/E/T counterpart ledgers disagree on held-month provenance")
    policy = _read_policy(policy_path)
    base = base.merge(policy, on="candidate_id", how="inner", validate="one_to_one")
    base = base.loc[base.policy_path_valid & np.isfinite(base.policy_net_bps)].copy()
    if base.empty:
        raise AssertionError("no valid canonical-policy outcome rows")
    sizes = base.groupby("__decision_ts__", sort=False).size()
    if int(sizes.min()) < 10:
        raise AssertionError("a held timestamp has fewer than ten exact counterpart candidates")
    for head in ("B", "E", "T"):
        rank_input = base.loc[:, ["__decision_ts__", "candidate_id", f"{head}_score"]].rename(columns={f"{head}_score": "score"})
        base[f"{head}_rank_ts"] = _rank01(rank_input, "score")
    base["held_month"] = base.B_held_month
    return base.drop(columns=["B_held_month", "E_held_month", "T_held_month", "policy_path_valid"])


def _weights(step: float) -> list[tuple[float, float, float]]:
    units = int(round(1.0 / step))
    if not np.isclose(units * step, 1.0) or units < 2:
        raise ValueError("--step must partition 1.0 into at least two equal units")
    values = [
        (b / units, e / units, (units - b - e) / units)
        for b in range(units + 1)
        for e in range(units - b + 1)
    ]
    # The incumbent equal blend is the mandatory control even when the chosen
    # coarse grid cannot represent one third exactly (for example, 5% steps).
    equal = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    if not any(np.allclose(item, equal) for item in values):
        values.append(equal)
    return values


def _metrics(frame: pd.DataFrame, score: np.ndarray, name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scored = frame.loc[:, ["__decision_ts__", "candidate_id", "policy_net_bps", "held_month"]].copy()
    scored["blend_score"] = score
    overall = _timestamp_metrics(scored, "blend_score")
    overall["contract"] = name
    overall["selection_objective"] = _objective_value(overall)
    monthly: list[dict[str, Any]] = []
    for month, part in scored.groupby("held_month", sort=True):
        item = _timestamp_metrics(part, "blend_score")
        item.update({"contract": name, "held_month": month})
        monthly.append(item)
    return overall, monthly


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b-oof", type=Path, required=True)
    parser.add_argument("--e-oof", type=Path, required=True)
    parser.add_argument("--t-oof", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--step", type=float, default=.05)
    parser.add_argument("--top-save", type=int, default=12)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    frame = _build(args.b_oof, args.e_oof, args.t_oof, args.policy_path)
    held = pd.PeriodIndex(frame.held_month, freq="M")
    months = tuple(sorted(held.unique()))
    span = (months[-1].year - months[0].year) * 12 + months[-1].month - months[0].month
    if len(months) < 5 or len({month.year for month in months}) < 2 or span < 8:
        raise AssertionError("blend selection needs five held months spanning two years and eight months")
    args.out.mkdir(parents=True)
    _exclusive(args.out / "run_manifest.json", {
        "schema": "strict_r3_threeway_rank_blend_grid_v1",
        "scope": "offline research-only B/E/T blend selection; no live/inference/consensus/MC1/admission/portfolio/execution mutation",
        "inputs": {"B": str(args.b_oof), "E": str(args.e_oof), "T": str(args.t_oof), "policy": str(args.policy_path)},
        "input_sha256": {"B": _sha(args.b_oof), "E": _sha(args.e_oof), "T": _sha(args.t_oof), "policy": _sha(args.policy_path)},
        "candidate_contract": "identical strict-OOF B/E/T identities; frozen 50% router inherited from ledgers",
        "score_contract": "deterministic timestamp-local percentile rank per head; weighted B/E/T rank blend",
        "policy_outcome": "canonical rich-policy net bps joined only after all three target-free score ledgers",
        "held_months": [str(month) for month in months],
        "selection": "same cross-year OOF folds used for HPO; model-selection evidence, not untouched promotion evidence",
        "objective": "0.27 Top-1 + 0.23 Top-2 + 0.20 Top-5 + 0.10 Top-10-percent timestamp-local EV plus monthly/weekly stability terms",
        "grid_step": args.step,
    })
    records: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    saved_scores: list[tuple[str, np.ndarray]] = []
    for b_weight, e_weight, t_weight in _weights(args.step):
        name = f"B{b_weight:.2f}_E{e_weight:.2f}_T{t_weight:.2f}"
        score = (
            b_weight * frame.B_rank_ts.to_numpy(float)
            + e_weight * frame.E_rank_ts.to_numpy(float)
            + t_weight * frame.T_rank_ts.to_numpy(float)
        )
        overall, monthly = _metrics(frame, score, name)
        overall.update({"B_weight": b_weight, "E_weight": e_weight, "T_weight": t_weight})
        for row in monthly:
            row.update({"B_weight": b_weight, "E_weight": e_weight, "T_weight": t_weight})
        records.append(overall)
        monthly_rows.extend(monthly)
        saved_scores.append((name, score))
    summary = pd.DataFrame(records)
    control = summary.loc[np.isclose(summary.B_weight, 1 / 3) & np.isclose(summary.E_weight, 1 / 3) & np.isclose(summary.T_weight, 1 / 3)]
    if len(control) != 1:
        raise AssertionError("rank-grid does not contain the equal B/E/T control")
    control = control.iloc[0]
    for metric in ("selection_objective", "ts_top01_ev", "ts_top02_ev", "ts_top05_ev", "ts_top10_ev", "weekly_q10_top02", "worst_month_top10", "fixed_k1_ev", "fixed_k2_ev", "fixed_k5_ev", "fixed_k10_ev"):
        summary[f"delta_{metric}_vs_equal"] = summary[metric] - float(control[metric])
    summary = summary.sort_values(
        ["selection_objective", "weekly_q10_top02", "worst_month_top10", "fixed_k1_ev", "B_weight", "E_weight", "T_weight"],
        ascending=[False, False, False, False, True, True, True], kind="stable",
    ).reset_index(drop=True)
    selected = summary.iloc[0].to_dict()
    selected_name = str(selected["contract"])
    score_lookup = dict(saved_scores)
    selected_rows = frame.loc[:, [*IDENTITY, "held_month", "policy_net_bps", "B_score", "E_score", "T_score", "B_rank_ts", "E_rank_ts", "T_rank_ts"]].copy()
    selected_rows["blend_score"] = score_lookup[selected_name]
    selected_rows["B_weight"] = float(selected["B_weight"])
    selected_rows["E_weight"] = float(selected["E_weight"])
    selected_rows["T_weight"] = float(selected["T_weight"])
    summary.to_parquet(args.out / "grid_summary.parquet", index=False, compression="zstd")
    pd.DataFrame(monthly_rows).to_parquet(args.out / "grid_monthly_metrics.parquet", index=False, compression="zstd")
    selected_rows.to_parquet(args.out / "selected_oof_predictions.parquet", index=False, compression="zstd")
    _exclusive(args.out / "winner.json", {
        "selected": selected,
        "equal_rank_control": control.to_dict(),
        "selection_only": True,
        "acceptance": "requires downstream meta/MC1 replay on a later frozen period before any promotion",
        "selected_prediction_rows": len(selected_rows),
        "selected_timestamps": int(selected_rows.__decision_ts__.nunique()),
    })


if __name__ == "__main__":
    main()
