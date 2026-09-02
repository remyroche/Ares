#!/usr/bin/env python3
"""Falsify frozen learned Meta-HPO proxy predictions on unseen MC1 labels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


SCHEMA = "strict_r3_p8u_meta_proxy_holdout_evaluation_v1"
MODELS = ("P0_ridge", "P1_elastic_net", "P2_depth2_gbdt", "P3_pairwise", "mean")
TARGETS = ("priority", "gate")


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _clean(root: Path) -> None:
    report = json.loads((root / "correctness_report.json").read_text())
    if not all(value is True for value in report.values()):
        raise AssertionError(f"{root}: upstream correctness receipt is not clean")


def _metrics(part: pd.DataFrame, *, score: str, target: str) -> dict[str, object]:
    values = pd.to_numeric(part[target], errors="coerce").to_numpy(float)
    scores = pd.to_numeric(part[score], errors="coerce").to_numpy(float)
    valid = np.isfinite(values) & np.isfinite(scores)
    values, scores = values[valid], scores[valid]
    order = np.argsort(-scores, kind="stable")
    actual = np.argsort(-values, kind="stable")
    row: dict[str, object] = {
        "rows": int(len(values)),
        "spearman": float(spearmanr(scores, values).statistic) if len(values) >= 3 else float("nan"),
    }
    for k in (1, 3, 5):
        selected = order[: min(k, len(order))]
        best = actual[: min(k, len(actual))]
        row[f"top{k}_precision"] = float(len(set(selected).intersection(best)) / max(1, len(selected)))
        row[f"regret_at{k}"] = float(np.max(values) - np.max(values[selected]))
        row[f"winner_in_top{k}"] = bool(int(np.argmax(values)) in set(selected))
        row[f"selected_{target}_mean"] = float(np.mean(values[selected]))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prediction_root, label_root, out = args.prediction_root.resolve(), args.label_root.resolve(), args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    _clean(prediction_root); _clean(label_root)
    prediction = pd.read_parquet(prediction_root / "holdout_proxy_predictions.parquet")
    labels = pd.read_parquet(label_root / "downstream_trial_labels.parquet")
    frame = prediction.merge(labels, on="trial", how="inner", validate="one_to_one")
    if len(frame) != len(prediction) or len(frame) != len(labels):
        raise AssertionError("frozen prediction and MC1-label holdout identities differ")
    rows: list[dict[str, object]] = []
    selected: list[pd.DataFrame] = []
    for family in TARGETS:
        target = f"d{family}_shrunk"
        for model in MODELS:
            score = f"proxy_{family}_{model}"
            metric = _metrics(frame, score=score, target=target)
            metric.update({"family": family, "target": target, "model": model})
            rows.append(metric)
            for k in (1, 3, 5):
                subset = frame.nlargest(k, score, keep="all").copy()
                subset["family"] = family; subset["model"] = model; subset["k"] = k
                subset["proxy_score"] = subset[score]
                selected.append(subset)
    out.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(out / "holdout_proxy_metrics.parquet", index=False, compression="zstd")
    pd.concat(selected, ignore_index=True).to_parquet(out / "holdout_proxy_selected_trials.parquet", index=False, compression="zstd")
    _once(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline frozen-proxy falsification only; no refit, no HPO selection, no score/admission/portfolio/live mutation",
        "prediction_root": str(prediction_root), "label_root": str(label_root), "rows": int(len(frame)),
        "metrics": ["Spearman", "Top-k precision", "winner containment", "regret@k", "selected downstream label mean"],
        "selection_authority": "none; must pass this unseen trial-level test before any proxy-guided HPO funnel",
    })
    _once(out / "correctness_report.json", {
        "frozen_proxy_predictions_were_sealed_before_holdout_mc1_labels": True,
        "prediction_and_label_identities_are_exact": True,
        "no_proxy_refit_on_holdout": True,
        "no_live_or_model_score_mutation": True,
    })
    print(out)


if __name__ == "__main__":
    main()
