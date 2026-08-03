#!/usr/bin/env python3
"""Report strictly OOF six-class path-head metrics from sealed side outputs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.audit_febapr2025_historical_catboost_six_class_gate import CLASS_ORDER


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    probabilities = frame[[f"prob_{c}" for c in CLASS_ORDER]].to_numpy(dtype=float)
    y = pd.Categorical(frame.class_label, categories=CLASS_ORDER).codes
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == y
    log_loss = float(-np.log(np.clip(probabilities[np.arange(len(y)), y], 1e-12, 1)).mean())
    onehot = np.eye(len(CLASS_ORDER))[y]
    bins = np.minimum((confidence * 10).astype(int), 9)
    ece = sum(len(confidence[bins == b]) / len(frame) * abs(confidence[bins == b].mean() - correct[bins == b].mean())
              for b in range(10) if (bins == b).any())
    calibration = []
    discrimination = []
    for j, label in enumerate(CLASS_ORDER):
        actual = float((y == j).mean())
        predicted = float(probabilities[:, j].mean())
        calibration.append({"class_label": label, "actual_share": actual, "mean_predicted_probability": predicted,
                            "signed_gap": predicted - actual})
        binary = (y == j).astype(np.int8)
        discrimination.append({"class_label": label,
                               "one_vs_rest_roc_auc": float(roc_auc_score(binary, probabilities[:, j])),
                               "one_vs_rest_average_precision": float(average_precision_score(binary, probabilities[:, j]))})
    return {"rows": int(len(frame)), "multiclass_logloss": log_loss,
            "accuracy": float(correct.mean()),
            "macro_f1": float(f1_score(y, probabilities.argmax(axis=1), average="macro")),
            "balanced_accuracy": float(balanced_accuracy_score(y, probabilities.argmax(axis=1))),
            "multiclass_brier": float(np.square(probabilities - onehot).sum(axis=1).mean()),
            "top_confidence_ece_10bin": float(ece), "class_calibration": calibration,
            "class_discrimination": discrimination}


def _economics(frame: pd.DataFrame) -> dict[str, Any]:
    # This is only a class-probability diagnostic.  It is deliberately not an
    # EV mapping, admission calibrator, or production top-k policy.
    favourable = (frame.prob_fast_realization_winner + frame.prob_late_breakout + frame.prob_slow_grinder)
    ranked = frame.assign(raw_actionable_probability=favourable).sort_values("raw_actionable_probability", ascending=False, kind="stable")
    top_n = int(np.ceil(len(ranked) * .10))
    top = ranked.head(top_n)
    def summary(x: pd.DataFrame) -> dict[str, Any]:
        return {"rows": int(len(x)), "mean_net_ev": float(x.realized_execution_net_ev_12h.mean()),
                "median_net_ev": float(x.realized_execution_net_ev_12h.median()),
                "positive_ev_rate": float(x.realized_execution_net_ev_12h.gt(0).mean())}
    return {"all_rows": summary(ranked),
            "top_10pct_global_by_raw_actionable_probability": {**summary(top),
                "threshold": float(top.raw_actionable_probability.iloc[-1]),
                "definition": "global across both sides and both OOF months; no per-timestamp ranking"}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root
    sides = []
    for side in ("long", "short"):
        manifest = json.loads((root / side / "oof_manifest.json").read_text())
        if manifest.get("stage") != "strict_oof" or not manifest.get("hpo_convergence", {}).get("accepted"):
            raise ValueError(f"{side} OOF is not converged strict OOF")
        sides.append(pd.read_parquet(root / side / "oof.parquet").assign(side=side))
    all_rows = pd.concat(sides, ignore_index=True)
    by_side_month = []
    for (side, month), group in all_rows.groupby(["side", "oof_month"], sort=True):
        economics = _economics(group)
        by_side_month.append({"side": side, "oof_month": month, "classification": _metrics(group),
                              "economics_all_rows": economics["all_rows"],
                              "raw_actionable_top_10pct_within_side_month": economics["top_10pct_global_by_raw_actionable_probability"]})
    result = {"schema": "febapr2025_historical_six_class_catboost_oof_report_v1",
              "scope": "strict Mar-Apr OOF from converged 128-tree HPO only",
              "class_order": list(CLASS_ORDER), "overall_classification": _metrics(all_rows),
              "by_side_month": by_side_month, "economics_diagnostic": _economics(all_rows),
              "not_a_deployed_policy": True,
              "warning": "raw actionable probability is not an execution-EV mapping or admission-calibrated trading score."}
    target = root / "strict_oof_metrics_calibration_economics.json"
    _write(target, result)
    print(json.dumps(result, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
