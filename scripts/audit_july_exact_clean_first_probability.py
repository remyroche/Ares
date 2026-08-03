#!/usr/bin/env python3
"""Audit frozen clean-first probabilities against exact July 1m event labels.

This is an identical-identity semantic audit.  It evaluates the already frozen
``catboost_hard_clean_first__probability`` against newly materialized exact
1-minute h12_u1p5atr competing-risk labels; it neither refits nor selects a
model or threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "july_exact_clean_first_probability_audit_v1"
IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
PROBABILITY = "catboost_hard_clean_first__probability"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probability_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """Use the same fixed-bin ECE definition as the gate challenger."""

    y = np.asarray(target, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    if len(y) != len(p) or not len(y):
        raise ValueError("probability metrics require non-empty aligned vectors")
    bins = np.minimum((p * 10).astype(int), 9)
    output = {
        "rows": int(len(y)),
        "prevalence": float(y.mean()),
        "mean_probability": float(p.mean()),
        "auc": float("nan"),
        "pr_auc": float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece_10": float(sum((bins == index).mean() * abs(float(p[bins == index].mean()) - float(y[bins == index].mean())) for index in np.unique(bins))),
    }
    if np.unique(y).size == 2:
        output["auc"] = float(roc_auc_score(y, p))
        output["pr_auc"] = float(average_precision_score(y, p))
    return output


def _top_decile(frame: pd.DataFrame) -> dict[str, Any]:
    count = max(1, int(np.ceil(0.10 * len(frame))))
    chosen = frame.sort_values([PROBABILITY, "candidate_id"], ascending=[False, True], kind="stable").head(count)
    return {
        "top10_rows": int(len(chosen)),
        "top10_exact_clean_first_precision": float(chosen["exact_clean_first"].mean()),
        "top10_adverse_first_or_conflict_rate": float(chosen["exact_adverse_first_or_conflict"].mean()),
        "top10_timeout_rate": float(chosen["exact_timeout"].mean()),
        "top10_net_ev_bps": float(pd.to_numeric(chosen["execution_net_ev_12h"], errors="raise").mean() * 1e4),
        "top10_positive_net_precision": float(pd.to_numeric(chosen["execution_net_ev_12h"], errors="raise").gt(0.0).mean()),
    }


def _calibration_rows(frame: pd.DataFrame, scope: str) -> list[dict[str, Any]]:
    values = pd.to_numeric(frame[PROBABILITY], errors="raise").to_numpy(float)
    bins = np.minimum((np.clip(values, 0.0, 1.0) * 10).astype(int), 9)
    rows: list[dict[str, Any]] = []
    for index in range(10):
        mask = bins == index
        if not mask.any():
            continue
        local = frame.loc[mask]
        rows.append({
            "scope": scope,
            "bin": int(index),
            "lower_probability": index / 10.0,
            "upper_probability": (index + 1) / 10.0,
            "rows": int(mask.sum()),
            "mean_probability": float(pd.to_numeric(local[PROBABILITY], errors="raise").mean()),
            "exact_clean_first_rate": float(local["exact_clean_first"].mean()),
        })
    return rows


def run(*, predictions_path: Path, labels_path: Path, labels_manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    manifest = json.loads(labels_manifest_path.read_text())
    if manifest.get("schema") != "july_exact1m_clean_first_labels_v1":
        raise ValueError("exact labels are not the expected July h12 1m clean-first artifact")
    if _sha256(labels_path) != manifest["outputs"]["labels"]["sha256"]:
        raise ValueError("exact clean-first label hash does not match its manifest")
    predictions = pd.read_parquet(predictions_path, columns=[*IDENTITY, PROBABILITY, "execution_net_ev_12h"])
    labels = pd.read_parquet(labels_path, columns=[*IDENTITY, "__soft_tb_first_event__"])
    for name, frame in (("predictions", predictions), ("exact labels", labels)):
        if frame.duplicated(list(IDENTITY), keep=False).any():
            raise ValueError(f"{name} have duplicate candidate identities")
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
        frame["side_name"] = frame["side_name"].astype(str).str.lower()
    joined = predictions.merge(labels, on=list(IDENTITY), how="inner", validate="one_to_one")
    if len(joined) != len(predictions) or len(joined) != len(labels):
        raise ValueError("frozen probability and exact-label identities are not identical")
    event = joined["__soft_tb_first_event__"].astype(str)
    if not event.isin(("favorable_first", "adverse_first_or_conflict", "timeout")).all():
        raise ValueError("unexpected exact competing-risk event")
    joined["exact_clean_first"] = event.eq("favorable_first").astype(np.int8)
    joined["exact_adverse_first_or_conflict"] = event.eq("adverse_first_or_conflict").astype(np.int8)
    joined["exact_timeout"] = event.eq("timeout").astype(np.int8)
    metric_rows: list[dict[str, Any]] = []
    calibration: list[dict[str, Any]] = []
    for scope, local in [("pooled", joined), *[(side, joined.loc[joined.side_name.eq(side)].copy()) for side in ("long", "short")]]:
        metrics = probability_metrics(local["exact_clean_first"].to_numpy(), local[PROBABILITY].to_numpy())
        metric_rows.append({"scope": scope, **metrics, **_top_decile(local)})
        calibration.extend(_calibration_rows(local, scope))
    output_dir.mkdir(parents=True)
    metric_path = output_dir / "metrics_by_scope.csv"
    calibration_path = output_dir / "calibration_by_scope_bin.csv"
    pd.DataFrame(metric_rows).to_csv(metric_path, index=False)
    pd.DataFrame(calibration).to_csv(calibration_path, index=False)
    report = {
        "schema": SCHEMA,
        "status": "completed_identical_row_semantic_audit_not_model_selection",
        "prediction": {"path": str(predictions_path), "sha256": _sha256(predictions_path), "column": PROBABILITY, "frozen": True},
        "exact_label": {"path": str(labels_path), "sha256": _sha256(labels_path), "event": "favorable_first under executable spread-adjusted h12_u1p5atr"},
        "identity": list(IDENTITY),
        "rows": int(len(joined)),
        "metric_contract": {"AUC": "ROC AUC", "PR_AUC": "average precision", "Brier": "mean squared probability error", "ECE_10": "fixed decile-bin expected calibration error", "top10": "one deterministic score-desc/candidate-id global or side-local decile"},
        "outputs": {
            "metrics": {"path": str(metric_path), "sha256": _sha256(metric_path)},
            "calibration": {"path": str(calibration_path), "sha256": _sha256(calibration_path)},
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _parser() -> argparse.ArgumentParser:
    root = ROOT / "data_perp/artifacts"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=root / "historical_to_july_meaningful_mfe_gate_challenger_20260730_v2/current_predictions.parquet")
    parser.add_argument("--labels", type=Path, default=root / "july_exact1m_clean_first_labels_20260730_v1/exact_clean_first_labels.parquet")
    parser.add_argument("--labels-manifest", type=Path, default=root / "july_exact1m_clean_first_labels_20260730_v1/manifest.json")
    parser.add_argument("--output-dir", type=Path, default=root / "july_exact_clean_first_probability_audit_20260730_v1")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(json.dumps(run(predictions_path=args.predictions, labels_path=args.labels, labels_manifest_path=args.labels_manifest, output_dir=args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
