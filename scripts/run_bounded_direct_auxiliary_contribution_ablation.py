#!/usr/bin/env python3
"""Strict March-OOF / April-confirmation auxiliary contribution screen.

This is deliberately a ranking-only screen: it uses one pooled global book,
stable candidate-ID ties, and no execution/timing action.  Peak contribution
is the hurdle expectation P(hit) * E(peak | hit), while slope is its separate
strict-OOF diagnostic prediction.  The only fitted post-processing is a
causal 21-day isotonic admission map; each day can see only labels resolved
before that day.  It is not a portfolio replay or a promotion decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data_perp/artifacts/marapr2025_all_score_ic_ev_waterfall_20260730_v1/all_score_waterfall.parquet"
PEAK = ROOT / "data_perp/artifacts/febapr2025_historical_auxiliary_oof_20260729_v2/oof_predictions.parquet"
SLOPE = ROOT / "data_perp/artifacts/febapr2025_historical_future_slope_fixed_geometry_oof_20260730_v1/oof_predictions.parquet"
ID = ["candidate_id", "side_name", "__symbol__", "__ts__"]
TIME, END, NET = "execution_decision_utc", "execution_label_end_utc", "execution_net_ev_12h"
FRACTIONS = (0.01, 0.05, 0.10, 0.20)
WEIGHTS = (0.0, 0.10, 0.25)
ARMS = ("control", "peak_contribution", "future_slope", "both")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, item: Any) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(item, indent=2, default=str, sort_keys=True) + "\n")
    os.replace(temp, path)


def ordered(frame: pd.DataFrame, score: str, fraction: float) -> pd.DataFrame:
    n = max(1, int(math.ceil(len(frame) * fraction)))
    return frame.sort_values([score, "candidate_id", "__ts__", "__symbol__", "side_name"], ascending=[False, True, True, True, True], kind="mergesort").iloc[:n].copy()


def zscore(reference: pd.Series, value: pd.Series) -> tuple[np.ndarray, dict[str, float]]:
    median = float(reference.median())
    scale = float(reference.std(ddof=0))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return ((value.to_numpy(float) - median) / scale), {"median": median, "std": scale}


def causal_map(history: pd.DataFrame, evaluation: pd.DataFrame, score: str) -> np.ndarray:
    """Map per calendar day using at most 21 earlier, label-resolved days."""
    out = np.full(len(evaluation), np.nan, dtype=float)
    for day in sorted(evaluation[TIME].dt.floor("D").unique()):
        valid = evaluation[TIME].dt.floor("D").eq(day).to_numpy()
        train = history.loc[
            (history[TIME] < day)
            & (history[END] < day)
            & (history[TIME] >= day - pd.Timedelta(days=21))
            & np.isfinite(history[score])
        ]
        if len(train) >= 300 and train[score].nunique() > 1:
            out[valid] = IsotonicRegression(out_of_bounds="clip").fit(train[score], train[NET]).predict(evaluation.loc[valid, score])
        else:
            out[valid] = evaluation.loc[valid, score].to_numpy(float)
    return out


def metric_rows(frame: pd.DataFrame, *, arm: str, weight: float, stage: str, month: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metrics: list[dict[str, Any]] = []
    sides: list[dict[str, Any]] = []
    assets: list[dict[str, Any]] = []
    for score_kind, score in (("raw", "raw_score"), ("causal_21d_admission_map", "mapped_score")):
        for fraction in FRACTIONS:
            selected = ordered(frame, score, fraction)
            cutoff = float(selected[score].iloc[-1])
            tie_rows = int(np.isclose(frame[score].to_numpy(float), cutoff, rtol=0.0, atol=1e-14).sum())
            metrics.append({
                "arm": arm, "weight": weight, "stage": stage, "month": month, "score_kind": score_kind,
                "selection": "one_pooled_global_top_k_stable_candidate_id_ties", "top_fraction": fraction,
                "rows": len(selected), "net_bps": float(selected[NET].mean() * 1e4),
                "gross_bps": float(selected.execution_gross_ev_12h.mean() * 1e4),
                "cost_bps": float(selected.execution_cost_return.mean() * 1e4),
                "positive_rate": float(selected[NET].gt(0).mean()),
                "full_rank_ic": float(frame[score].corr(frame[NET], method="spearman")),
                "cutoff": cutoff, "cutoff_tie_rows": tie_rows,
                "cutoff_tie_fraction_of_book": float(tie_rows / len(selected)),
                "distinct_scores": int(frame[score].nunique()),
                "latest_fold_coverage": bool(month == "2025-04"),
            })
            for side, part in selected.groupby("side_name", sort=True):
                sides.append({"arm": arm, "weight": weight, "stage": stage, "month": month, "score_kind": score_kind, "top_fraction": fraction, "side_name": side, "selected_rows": len(part), "share": float(len(part) / len(selected)), "net_bps": float(part[NET].mean() * 1e4), "positive_rate": float(part[NET].gt(0).mean())})
            for asset, part in selected.groupby("__symbol__", sort=True):
                assets.append({"arm": arm, "weight": weight, "stage": stage, "month": month, "score_kind": score_kind, "top_fraction": fraction, "__symbol__": asset, "selected_rows": len(part), "share": float(len(part) / len(selected)), "net_bps": float(part[NET].mean() * 1e4)})
    return metrics, sides, assets


def load(source: Path, peak: Path, slope: Path) -> pd.DataFrame:
    x = pd.read_parquet(source)
    p = pd.read_parquet(peak, columns=ID + ["pred_peak_mfe_12h_atr__p_hit", "pred_peak_mfe_12h_atr__conditional_mean"])
    s = pd.read_parquet(slope, columns=ID + ["pred_future_slope_atr_per_hour__diagnostic"])
    for part in (x, p, s):
        part["__ts__"] = pd.to_datetime(part["__ts__"], utc=True, errors="raise")
        if part.duplicated(ID).any():
            raise ValueError("source identities must be unique")
    x = x.merge(p, on=ID, how="inner", validate="one_to_one").merge(s, on=ID, how="inner", validate="one_to_one")
    x[TIME] = pd.to_datetime(x[TIME], utc=True, errors="raise")
    x[END] = pd.to_datetime(x[END], utc=True, errors="raise")
    if len(x) != 140_682 or x[ID].duplicated().any() or not x["candidate_month"].isin(("2025-03", "2025-04")).all():
        raise ValueError("common-ID March/April 140,682-row contract failed")
    if not (x[END] == x[TIME] + pd.Timedelta(hours=12)).all():
        raise ValueError("execution labels must resolve exactly 12 hours after decision")
    if not np.allclose(x.execution_gross_ev_12h - x.execution_cost_return, x[NET], rtol=0, atol=1e-12):
        raise ValueError("gross-cost-net identity failed")
    if not np.isfinite(x[["direct_q25_return", "pred_peak_mfe_12h_atr__p_hit", "pred_peak_mfe_12h_atr__conditional_mean", "pred_future_slope_atr_per_hour__diagnostic"]]).all().all():
        raise ValueError("one or more frozen predictions is unavailable")
    return x


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    x = load(args.source, args.peak, args.slope)
    april_start = pd.Timestamp("2025-04-01T01:00:00Z")
    development = x.loc[x.candidate_month.eq("2025-03") & x[END].lt(april_start)].copy()
    confirmation = x.loc[x.candidate_month.eq("2025-04")].copy()
    if len(development) == 0 or development[END].max() >= april_start:
        raise ValueError("March development contains labels unavailable at April confirmation start")
    # Freeze global component scale from causally available March OOF only.
    development["peak_contribution"] = development.pred_peak_mfe_12h_atr__p_hit * development.pred_peak_mfe_12h_atr__conditional_mean
    confirmation["peak_contribution"] = confirmation.pred_peak_mfe_12h_atr__p_hit * confirmation.pred_peak_mfe_12h_atr__conditional_mean
    components = {"control": [], "peak_contribution": ["peak_contribution"], "future_slope": ["pred_future_slope_atr_per_hour__diagnostic"], "both": ["peak_contribution", "pred_future_slope_atr_per_hour__diagnostic"]}
    base_dev, base_scale = zscore(development.direct_q25_return, development.direct_q25_return)
    base_conf, _ = zscore(development.direct_q25_return, confirmation.direct_q25_return)
    scale: dict[str, Any] = {"direct_q25_return": base_scale}
    for column in ("peak_contribution", "pred_future_slope_atr_per_hour__diagnostic"):
        development[column + "_z"], scale[column] = zscore(development[column], development[column])
        confirmation[column + "_z"], _ = zscore(development[column], confirmation[column])
    development["base_z"] = base_dev
    confirmation["base_z"] = base_conf
    candidates: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    all_sides: list[dict[str, Any]] = []
    all_assets: list[dict[str, Any]] = []
    ledgers: list[pd.DataFrame] = []
    for arm in ARMS:
        for weight in WEIGHTS:
            if arm == "control" and weight != 0.0:
                continue
            dev = development.copy()
            conf = confirmation.copy()
            additions = sum((dev[column + "_z"] for column in components[arm]), start=pd.Series(0.0, index=dev.index))
            additions_conf = sum((conf[column + "_z"] for column in components[arm]), start=pd.Series(0.0, index=conf.index))
            dev["raw_score"] = dev.base_z + weight * additions
            conf["raw_score"] = conf.base_z + weight * additions_conf
            # March OOF metrics use a sequential causal map; April map sees March
            # only, never April labels.  Scores are unmodified outcome-free inputs.
            dev["mapped_score"] = causal_map(dev, dev, "raw_score")
            conf["mapped_score"] = causal_map(dev, conf, "raw_score")
            for stage, month, sample in (("development_oof", "2025-03", dev), ("confirmation", "2025-04", conf)):
                m, si, a = metric_rows(sample, arm=arm, weight=weight, stage=stage, month=month)
                all_metrics.extend(m); all_sides.extend(si); all_assets.extend(a)
            select = ordered(dev, "mapped_score", 0.10)
            candidates.append({"arm": arm, "weight": weight, "march_oof_mapped_top10_net_bps": float(select[NET].mean() * 1e4), "march_oof_mapped_top10_rows": len(select)})
            ledgers.append(conf[ID + [TIME, END, NET, "raw_score", "mapped_score"]].assign(arm=arm, weight=weight))
    choices = pd.DataFrame(candidates).sort_values(["march_oof_mapped_top10_net_bps", "arm", "weight"], ascending=[False, True, True], kind="mergesort")
    winner = choices.iloc[0].to_dict()
    temp = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent))
    pd.DataFrame(all_metrics).to_csv(temp / "global_metrics.csv", index=False)
    pd.DataFrame(all_sides).to_csv(temp / "side_metrics.csv", index=False)
    pd.DataFrame(all_assets).to_csv(temp / "asset_metrics.csv", index=False)
    choices.to_csv(temp / "march_oof_weight_selection.csv", index=False)
    pd.concat(ledgers, ignore_index=True).to_parquet(temp / "april_confirmation_predictions.parquet", index=False, compression="zstd")
    outputs = {path.name: sha(path) for path in sorted(temp.iterdir()) if path.is_file()}
    manifest = {
        "schema": "bounded_direct_auxiliary_contribution_ablation_v1",
        "status": "COMPLETED_RESEARCH_ONLY_NO_PORTFOLIO_REPLAY",
        "promotion_eligible": False,
        "contract": {
            "population": "exact common IDs, March development / April untouched confirmation",
            "development_label_availability": "execution_label_end_utc < 2025-04-01T01:00:00Z",
            "selection": "one pooled global top K, stable candidate-ID tie ordering; sides/assets are attribution only",
            "arms": list(ARMS), "weights": list(WEIGHTS),
            "peak_formula": "strict OOF P(meaningful hit) × strict OOF E(peak MFE ATR | meaningful hit)",
            "future_slope": "strict OOF diagnostic prediction only; realised slope is not an input",
            "mapping": "causal rolling 21-day isotonic admission map; each day uses only earlier label-resolved rows",
            "actions": "timing, MAE, target price, and wait actions excluded",
            "portfolio_replay": "NOT_RUN",
        },
        "sources": {str(p): sha(p) for p in (args.source, args.peak, args.slope)},
        "scale_frozen_from_march_oof": scale,
        "frozen_april_winner_from_march_oof": winner,
        "outputs_sha256": outputs,
        "runner_sha256": sha(Path(__file__)),
    }
    write_json(temp / "manifest.json", manifest)
    (temp / "manifest.sha256").write_text(sha(temp / "manifest.json") + "  manifest.json\n")
    os.replace(temp, args.output_dir)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--peak", type=Path, default=PEAK)
    parser.add_argument("--slope", type=Path, default=SLOPE)
    parser.add_argument("--output-dir", type=Path, required=True)
    print(json.dumps(run(parser.parse_args()), indent=2, default=str))
