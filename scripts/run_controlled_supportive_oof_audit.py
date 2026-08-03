#!/usr/bin/env python3
"""Audit the strict OOF supportive heads used by the target matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.controlled_target_supportive_ablation import (
    GROUPED_SUPPORT_LABELS,
    SUPPORT_LABELS,
    strict_oof_support_predictions,
    validate_causal_raw_features,
)
from scripts.run_controlled_target_supportive_ablation import _predictor


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rank_ic(left: pd.Series, right: pd.Series) -> float:
    x = pd.to_numeric(left, errors="coerce")
    y = pd.to_numeric(right, errors="coerce")
    valid = x.notna() & y.notna()
    if int(valid.sum()) < 3 or x[valid].nunique() < 2 or y[valid].nunique() < 2:
        return float("nan")
    return float(x[valid].rank().corr(y[valid].rank()))


def _attach_oof_lineage(
    frame: pd.DataFrame,
    support: pd.DataFrame,
    *,
    fold_column: str,
    features_sha256: str,
    semantic_contract_sha256: str,
    support_labels: tuple[tuple[str, str, str, str], ...] = SUPPORT_LABELS,
) -> pd.DataFrame:
    """Persist the exact fold/model lineage for each emitted OOF vector.

    The original audit artifact retained predictions and fold names but did
    not retain the fit-end, logical generation timestamp, model identity, or
    OOF flag required by the semantic-support gate.  Fit ends are recomputed
    from the same training mask used by ``strict_oof_support_predictions``;
    they are not inferred from a broad fold maximum.
    """
    required = {"candidate_id", fold_column, "fold_order", "__ts__", "__decision_ts__", "__label_available_at__"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"support OOF lineage frame is missing columns: {missing}")
    output = support.copy()
    source = frame.copy()
    for column in ("__ts__", "__decision_ts__", "__label_available_at__"):
        output[column] = pd.to_datetime(output[column], utc=True, errors="raise")
        source[column] = pd.to_datetime(source[column], utc=True, errors="raise")
    if output["candidate_id"].duplicated().any():
        raise ValueError("support OOF lineage requires one prediction vector per candidate")
    fold_starts = (
        source.groupby(fold_column, observed=True)["__ts__"]
        .min()
        .sort_values(kind="mergesort")
    )
    fold_position = {fold: position for position, fold in enumerate(fold_starts.index)}
    fit_end_by_fold: dict[Any, pd.Timestamp] = {}
    model_id_by_fold: dict[Any, str] = {}
    head_names = [name for _, name, _, _ in support_labels]
    for fold, test_start in fold_starts.items():
        position = fold_position[fold]
        if position == 0:
            continue
        earlier_folds = set(fold_starts.index[:position])
        train_mask = source[fold_column].isin(earlier_folds) & source["__label_available_at__"].lt(test_start)
        if not bool(train_mask.any()):
            raise ValueError(f"no strictly resolved training rows for support fold {fold!r}")
        fit_end = source.loc[train_mask, "__label_available_at__"].max()
        if not bool(fit_end < test_start):
            raise ValueError(f"support fold {fold!r} fit end is not before its test start")
        fit_end_by_fold[fold] = fit_end
        model_id_by_fold[fold] = (
            "controlled_supportive_oof_v2:lightgbm:"
            f"heads-{','.join(head_names)}:fold-{fold}:features-{features_sha256[:16]}"
        )
    if pd.to_numeric(output["fold_order"], errors="raise").lt(1).any():
        raise ValueError("support OOF lineage cannot emit warmup-fold predictions")
    output["prediction_fold_id"] = output[fold_column].map(lambda value: str(value))
    output["prediction_fit_end_ts"] = output[fold_column].map(fit_end_by_fold)
    output["prediction_generated_ts"] = output["__decision_ts__"]
    output["prediction_model_id"] = output[fold_column].map(model_id_by_fold)
    output["is_oof"] = True
    output["semantic_target_contract_sha256"] = semantic_contract_sha256
    output["support_head_lineage"] = output[fold_column].map(
        lambda fold: json.dumps(
            {
                "schema": "supportive_head_lineage_v2",
                "heads": {
                    name: {
                        "model_id": f"{model_id_by_fold[fold]}:{name}",
                        "fold_id": str(fold),
                        "fit_end_ts": fit_end_by_fold[fold].isoformat(),
                        "generated_ts": "decision_ts",
                    }
                    for name in head_names
                },
            },
            sort_keys=True,
        )
    )
    return output


def run(
    *,
    ledger: Path,
    features_json: Path,
    semantic_contract: Path,
    output: Path,
    fold_column: str,
    support_labels: tuple[tuple[str, str, str, str], ...] = SUPPORT_LABELS,
    support_spec: str = "legacy",
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    frame = pd.read_parquet(ledger)
    payload = json.loads(features_json.read_text())
    semantic_contract_sha256 = _sha256(semantic_contract)
    features = (
        (payload.get("raw_feature_columns") or payload.get("feature_columns"))
        if isinstance(payload, dict)
        else payload
    )
    features = list(validate_causal_raw_features(features))
    support = strict_oof_support_predictions(
        frame,
        feature_columns=features,
        fold_column=fold_column,
        predictor=_predictor,
        support_labels=support_labels,
    )
    usable = support["fold_order"].ge(1) if "fold_order" in support else support[fold_column].ne("base_train")
    oof = support.loc[usable].copy()
    oof = _attach_oof_lineage(
        frame,
        oof,
        fold_column=fold_column,
        features_sha256=_sha256(features_json),
        semantic_contract_sha256=semantic_contract_sha256,
        support_labels=support_labels,
    )
    rows: list[dict[str, object]] = []
    for stage, name, label, kind in support_labels:
        prediction = oof[f"support_oof__{name}"]
        actual = pd.to_numeric(oof[label], errors="coerce")
        valid = prediction.notna() & actual.notna()
        selected = oof.loc[valid].assign(_score=prediction[valid])
        selected = selected.sort_values(["_score", "candidate_id"], ascending=[False, True], kind="mergesort").head(max(1, int(np.ceil(len(selected) * 0.10))))
        row: dict[str, object] = {
            "stage": stage, "head": name, "label": label, "kind": kind,
            "oof_rows": int(valid.sum()), "oof_folds": sorted(map(str, oof.loc[valid, fold_column].unique())),
            "target_mean": float(actual[valid].mean()), "prediction_mean": float(prediction[valid].mean()),
            "top10_rows": int(len(selected)), "top10_target_mean": float(pd.to_numeric(selected[label], errors="coerce").mean()),
            "rank_ic": _rank_ic(prediction[valid], actual[valid]),
            "mae": float(np.abs(prediction[valid].to_numpy(float) - actual[valid].to_numpy(float)).mean()),
            "rmse": float(np.sqrt(np.square(prediction[valid].to_numpy(float) - actual[valid].to_numpy(float)).mean())),
        }
        if kind == "binary":
            y = (actual[valid] > 0.5).astype(int)
            row["auc"] = float(roc_auc_score(y, prediction[valid])) if y.nunique() == 2 else float("nan")
            row["brier"] = float(brier_score_loss(y, prediction[valid]))
        rows.append(row)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        oof.to_parquet(stage / "supportive_head_oof_predictions.parquet", index=False, compression="zstd")
        metrics = pd.DataFrame(rows)
        metrics.to_parquet(stage / "supportive_head_metrics.parquet", index=False, compression="zstd")
        manifest = {
            "schema": "controlled_supportive_oof_audit_v1",
            "status": "STRICT_OOF_RESEARCH_DIAGNOSTIC",
            "ledger": str(ledger), "ledger_sha256": _sha256(ledger),
            "features_json": str(features_json), "features_json_sha256": _sha256(features_json),
            "semantic_target_contract": str(semantic_contract),
            "semantic_target_contract_sha256": semantic_contract_sha256,
            "feature_count": len(features), "support_spec": support_spec, "support_heads": [dict(stage=s, name=n, label=l, kind=k) for s, n, l, k in support_labels],
            "oof_rule": "fit_end labels available strictly before each test fold; first warmup fold has no emitted prediction",
            "prediction_generated_timestamp_semantics": "logical decision-time availability; generated_ts equals candidate decision_ts",
            "prediction_lineage_columns": [
                "is_oof", "prediction_fit_end_ts", "prediction_generated_ts",
                "prediction_model_id", "prediction_fold_id",
            ],
            "outputs_sha256": {
                name: _sha256(stage / name)
                for name in ("supportive_head_oof_predictions.parquet", "supportive_head_metrics.parquet")
            },
        }
        (stage / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
        os.replace(stage, output)
        return manifest
    except Exception:
        import shutil
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--features-json", type=Path, required=True)
    parser.add_argument("--semantic-contract", type=Path, required=True)
    parser.add_argument("--fold-column", default="oof_fold")
    parser.add_argument("--support-spec", choices=("legacy", "grouped"), default="legacy")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labels = GROUPED_SUPPORT_LABELS if args.support_spec == "grouped" else SUPPORT_LABELS
    print(json.dumps(run(
        ledger=args.ledger,
        features_json=args.features_json,
        semantic_contract=args.semantic_contract,
        output=args.output,
        fold_column=args.fold_column,
        support_labels=labels,
        support_spec=args.support_spec,
    ), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
