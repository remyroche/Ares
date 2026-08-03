#!/usr/bin/env python3
"""Evaluate the selected R3 target screen by causal side/time/geometry strata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def _safe_metric(fn, *args: np.ndarray) -> float:
    try:
        return float(fn(*args))
    except ValueError:
        return float("nan")


def _summary(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for values, group in frame.groupby(keys, observed=True, dropna=False):
        values = values if isinstance(values, tuple) else (values,)
        clear = group.target_class.eq(2).to_numpy(int)
        probs = group[["prob_adverse", "prob_weak", "prob_clear"]].to_numpy(float)
        score = group.score_bps.to_numpy(float)
        net = group.net_bps.to_numpy(float)
        row = dict(zip(keys, values, strict=True))
        row.update({"n": len(group), "clear_rate": clear.mean(),
                    "clear_auc": _safe_metric(roc_auc_score, clear, probs[:, 2]),
                    "clear_pr_auc": _safe_metric(average_precision_score, clear, probs[:, 2]),
                    "clear_brier": _safe_metric(brier_score_loss, clear, probs[:, 2]),
                    "multiclass_log_loss": _safe_metric(lambda y, p: log_loss(y, p, labels=[0, 1, 2]), group.target_class.to_numpy(int), probs),
                    "net_spearman": float(pd.Series(score).corr(pd.Series(net), method="spearman"))})
        for fraction in (.01, .05, .10):
            n = int(np.ceil(len(group) * fraction))
            top = group.sort_values(["score_bps", "candidate_id"], ascending=[False, True], kind="mergesort").head(n)
            row[f"top_{int(fraction*100)}_net_bps"] = top.net_bps.mean()
            row[f"top_{int(fraction*100)}_gross_bps"] = top.gross_bps.mean()
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, nargs="+", default=[
        ROOT / "data_perp/artifacts/tp6_screen_long_r3_composite_20260802_v2/target_repair_oof_predictions.parquet",
        ROOT / "data_perp/artifacts/tp6_screen_short_r3_composite_20260802_v2/target_repair_oof_predictions.parquet",
    ])
    p.add_argument("--robust", type=Path, default=ROOT / "data_perp/artifacts/tp6_sl4_robust_clear_labels_20260802_v1")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    predictions = pd.concat([pd.read_parquet(path) for path in args.predictions], ignore_index=True)
    required = {"prob_adverse", "prob_weak", "prob_clear", "target_class"}
    if not required <= set(predictions):
        raise ValueError("R3 component probabilities are required")
    # Keep the audit's memory proportional to evaluated predictions rather
    # than the 1.85m-row full population.  This remains an exact ID join.
    evaluated_ids = set(predictions.candidate_id)
    label_pieces = []
    for part in sorted((args.robust / "parts").glob("*.parquet")):
        piece = pd.read_parquet(part, columns=["candidate_id", "atr_bps"])
        piece = piece.loc[piece.candidate_id.isin(evaluated_ids)]
        if not piece.empty:
            label_pieces.append(piece)
    labels = pd.concat(label_pieces, ignore_index=True)
    frame = predictions.merge(labels, on="candidate_id", how="left", validate="one_to_one")
    if frame.atr_bps.isna().any():
        raise ValueError("missing decision-time ATR for a prediction")
    frame["month"] = pd.to_datetime(frame.__ts__, utc=True).dt.strftime("%Y-%m")
    frame["cost_to_tp"] = 100. / (6. * frame.atr_bps)
    frame["diagnostic_cost_atr_regime"] = pd.cut(
        frame.cost_to_tp, [-np.inf, .5, 1., np.inf],
        labels=["headroom", "thin_margin", "cost_dominated"],
    ).astype("string")
    by_side = _summary(frame, ["side_name"])
    by_month = _summary(frame, ["side_name", "month"])
    by_regime = _summary(frame, ["side_name", "diagnostic_cost_atr_regime"])
    deciles = []
    for (side, regime), group in frame.groupby(["side_name", "diagnostic_cost_atr_regime"], observed=True):
        group = group.copy(); group["score_decile"] = pd.qcut(group.score_bps.rank(method="first"), 10, labels=False) + 1
        for decile, chunk in group.groupby("score_decile", observed=True):
            deciles.append({"side_name": side, "diagnostic_cost_atr_regime": regime, "score_decile": int(decile),
                            "n": len(chunk), "gross_bps": chunk.gross_bps.mean(), "net_bps": chunk.net_bps.mean(),
                            "clear_rate": chunk.target_class.eq(2).mean()})
    args.out.mkdir(parents=True)
    by_side.to_parquet(args.out / "target_repair_model_side_metrics.parquet", index=False)
    by_month.to_parquet(args.out / "target_repair_model_month_metrics.parquet", index=False)
    by_regime.to_parquet(args.out / "target_repair_model_regime_metrics.parquet", index=False)
    pd.DataFrame(deciles).to_parquet(args.out / "target_regime_decile_economics.parquet", index=False)
    manifest = {"schema": "tp6_sl4_r3_causal_diagnostics_v1", "status": "COMPLETED",
                "contract": {"target": "R3: robust clear b25 / adverse-first / weak-or-unresolved",
                             "geometry": "selected TP6/SL4/H12", "cost_bps": 100,
                             "regimes": "decision-time cost / (6 * ATR), diagnostic only"},
                "side_metrics": by_side.to_dict("records")}
    (args.out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
    print(by_side.to_string(index=False))


if __name__ == "__main__":
    main()
