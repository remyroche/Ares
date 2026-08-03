#!/usr/bin/env python3
"""Causal monthly calibration test for residual-score × transition-state interactions.

The sole question is whether the observed current state can repair conversion
of an already-OOF residual score into common expected H12 net.  It is not a
state gate, side quota, transition forecast, or portfolio rule.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CONTEXT = ROOT / "data_perp/artifacts/failure_2022_2023_pf_exact1m_transition_context_continuation_20260730_v1/context.parquet"
SCORES = ROOT / "data_perp/artifacts/reconstructed_base_residual_stack_2022_2024_20260730_v4/oof_scores.parquet"
OUT = ROOT / "data_perp/artifacts/historical_causal_state_calibration_ablation_20260731_v1"
IDENTITY = ["candidate_id", "__ts__", "__symbol__", "side_name"]
TARGET = "execution_net_ev_12h"
RAW = "score_residual_expected_ev"


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


def features(frame: pd.DataFrame, arm: str) -> pd.DataFrame:
    score = frame[RAW].astype(float)
    side = frame.side_name.eq("long").astype(float)
    base = pd.DataFrame({"residual_score": score, "is_long": side}, index=frame.index)
    if arm == "common":
        return base
    state = pd.to_numeric(frame["state_context__current_state"], errors="raise").astype(int)
    for value in (2, 3):
        indicator = state.eq(value).astype(float)
        base[f"state_{value}"] = indicator
        base[f"residual_x_state_{value}"] = score * indicator
        base[f"long_x_state_{value}"] = side * indicator
    return base


def evaluate(frame: pd.DataFrame, score: str) -> dict[str, object]:
    selected = frame.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable").head(int(np.ceil(len(frame) * .1)))
    admitted = frame.loc[frame[score].gt(0.0)]
    slope, intercept = np.polyfit(frame[score].to_numpy(float), frame[TARGET].to_numpy(float), 1)
    return {"rows": int(len(frame)), "net_rank_ic": float(frame[score].rank().corr(frame[TARGET].rank())), "calibration_slope": float(slope), "calibration_intercept_bps": float(intercept * 1e4), "top10_rows": int(len(selected)), "top10_net_bps": float(selected[TARGET].mean() * 1e4), "top10_positive_fraction": float(selected[TARGET].gt(0).mean()), "threshold_rows": int(len(admitted)), "threshold_net_bps": float(admitted[TARGET].mean() * 1e4) if len(admitted) else float("nan")}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    context = pd.read_parquet(CONTEXT, columns=[*IDENTITY, "state_context__current_state"])
    scores = pd.read_parquet(SCORES, columns=[*IDENTITY, RAW, TARGET, "execution_gross_ev_12h", "execution_cost_return", "residual_is_oof"])
    for frame in (context, scores):
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    if not scores.residual_is_oof.astype(bool).all() or not np.allclose(scores.execution_gross_ev_12h - scores.execution_cost_return, scores[TARGET], atol=1e-12, rtol=0):
        raise ValueError("invalid score/economics contract")
    data = context.merge(scores.drop(columns="residual_is_oof"), on=IDENTITY, how="inner", validate="one_to_one").sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    starts = pd.date_range("2023-01-01", "2023-12-01", freq="MS", tz="UTC")
    predictions = []
    folds = []
    for arm in ("common", "state_2_3_interactions"):
        for start in starts:
            end = start + pd.offsets.MonthBegin(1)
            evaluate_rows = data.loc[data.__ts__.ge(start) & data.__ts__.lt(end)].copy()
            train = data.loc[data.__ts__.lt(start - pd.Timedelta(hours=12)) & data.__ts__.ge(start - pd.Timedelta(days=90))].copy()
            if len(train) < 1_000 or len(evaluate_rows) == 0:
                raise ValueError(f"inadequate causal support at {start}")
            x_train = features(train, arm)
            x_eval = features(evaluate_rows, arm)
            model = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("ridge", Ridge(alpha=30.0))])
            model.fit(x_train, train[TARGET].to_numpy(float))
            evaluate_rows["mapped_expected_net"] = model.predict(x_eval)
            evaluate_rows["arm"] = arm
            evaluate_rows["fold_month"] = start.strftime("%Y-%m")
            predictions.append(evaluate_rows)
            folds.append({"arm": arm, "fold_month": start.strftime("%Y-%m"), "train_rows": int(len(train)), "evaluation_rows": int(len(evaluate_rows)), "train_label_end_max": str(train.__ts__.max() + pd.Timedelta(hours=12)), "feature_columns": list(x_train.columns), "causal": True})
    oof = pd.concat(predictions, ignore_index=True)
    records = []
    monthly = []
    side = []
    for arm, local in oof.groupby("arm", sort=True):
        records.append({"arm": arm, **evaluate(local, "mapped_expected_net")})
        for month, group in local.groupby("fold_month", sort=True):
            monthly.append({"arm": arm, "month": month, **evaluate(group, "mapped_expected_net")})
        selected = local.sort_values(["mapped_expected_net", "candidate_id"], ascending=[False, True], kind="stable").head(int(np.ceil(len(local) * .1)))
        for name, group in selected.groupby("side_name", sort=True):
            side.append({"arm": arm, "side_name": name, "selected_rows": int(len(group)), "selected_share": float(len(group) / len(selected)), "net_bps": float(group[TARGET].mean() * 1e4), "positive_fraction": float(group[TARGET].gt(0).mean())})
    temp = Path(tempfile.mkdtemp(dir=OUT.parent, prefix=f".{OUT.name}."))
    try:
        oof.to_parquet(temp / "causal_monthly_predictions.parquet", index=False)
        pd.DataFrame(folds).to_csv(temp / "fold_provenance.csv", index=False)
        pd.DataFrame(records).to_csv(temp / "aggregate_metrics.csv", index=False)
        pd.DataFrame(monthly).to_csv(temp / "monthly_metrics.csv", index=False)
        pd.DataFrame(side).to_csv(temp / "global_top10_side_metrics.csv", index=False)
        contract = {"evidence_scope": "strictly_prior_resolved_monthly_calibration_ablation", "promotion_eligible": False, "selection": "one pooled global top10 after each arm emits expected H12 net; no state or side quota", "arms": {"common": "residual score plus side", "state_2_3_interactions": "common plus current-state 2/3 intercept and score/side interactions"}, "forbidden": "no future path, transition target, action head, state gate or state-specific selection rule", "acceptance": "must beat common on positive threshold and top10, latest month, both sides, and calibration; otherwise reject"}
        write(temp / "contract.json", contract)
        files = [temp / n for n in ("causal_monthly_predictions.parquet", "fold_provenance.csv", "aggregate_metrics.csv", "monthly_metrics.csv", "global_top10_side_metrics.csv", "contract.json")]
        manifest = {"schema": "historical_causal_state_calibration_ablation_v1", "status": "COMPLETE_RESEARCH_ONLY", "rows": int(len(oof) / 2), "coverage": [str(oof.__ts__.min()), str(oof.__ts__.max())], "sources": {str(p): sha(p) for p in (CONTEXT, SCORES)}, "outputs_sha256": {p.name: sha(p) for p in files}, **contract}
        write(temp / "manifest.json", manifest)
        (temp / "manifest.sha256").write_text(f"{sha(temp / 'manifest.json')}  manifest.json\n")
        os.replace(temp, OUT)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
