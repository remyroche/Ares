#!/usr/bin/env python3
"""Seven-day block bootstrap for frozen calibrated transition challengers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ABLATION = ROOT / (
    "data_perp/artifacts/cross_era_regime_transition_classifier_ablation_"
    "20260729_v3"
)
DEFAULT_OUTPUT = ROOT / (
    "data_perp/artifacts/cross_era_transition_classifier_bootstrap_"
    "20260729_v1"
)
SCHEMA = "cross_era_transition_classifier_bootstrap_v1"
BOOTSTRAP_DRAWS = 2_000
RANDOM_STATE = 20260729

FROZEN_ARMS = (
    {
        "arm": "canonical_active",
        "setup": "canonical_spread",
        "target": "target__active_adverse",
        "feature_set": "coordinates_plus_raw_state",
        "model": "extra_trees_shrunk",
        "provenance": None,
    },
    {
        "arm": "canonical_onset",
        "setup": "canonical_spread",
        "target": "target__adverse_onset_within_3h",
        "feature_set": "coordinates_only",
        "model": "logistic_shrunk",
        "provenance": None,
    },
    {
        "arm": "current_active_strict_oof",
        "setup": "current_exact_spread",
        "target": "target__active_adverse",
        "feature_set": "coordinates_plus_raw_state",
        "model": "logistic_shrunk",
        "provenance": "strict_oof",
    },
    {
        "arm": "current_onset_strict_oof",
        "setup": "current_exact_spread",
        "target": "target__adverse_onset_within_3h",
        "feature_set": "coordinates_plus_raw_state",
        "model": "logistic_shrunk",
        "provenance": "strict_oof",
    },
    {
        "arm": "reconstructed_active",
        "setup": "reconstructed_fee_only",
        "target": "target__active_adverse",
        "feature_set": "past_transitions_only",
        "model": "logistic_shrunk",
        "provenance": None,
    },
    {
        "arm": "reconstructed_onset",
        "setup": "reconstructed_fee_only",
        "target": "target__adverse_onset_within_3h",
        "feature_set": "coordinates_only",
        "model": "logistic_shrunk",
        "provenance": None,
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _metrics(y: np.ndarray, prediction: np.ndarray) -> tuple[float, float, float]:
    brier = float(brier_score_loss(y, prediction))
    average_precision = (
        float(average_precision_score(y, prediction))
        if y.sum() > 0
        else float("nan")
    )
    auc = (
        float(roc_auc_score(y, prediction))
        if np.unique(y).size == 2
        else float("nan")
    )
    return brier, average_precision, auc


def paired_block_bootstrap(
    frame: pd.DataFrame,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    required = {
        "target",
        "model_prediction",
        "prior_prediction",
        "cv_group_id",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"bootstrap frame lacks {missing}")
    groups = {
        str(group): indices.to_numpy()
        for group, indices in frame.groupby("cv_group_id", sort=True).groups.items()
    }
    names = np.asarray(sorted(groups), dtype=object)
    if len(names) < 5:
        raise ValueError("block bootstrap requires at least five UTC groups")
    y = frame["target"].to_numpy(float)
    model = np.clip(frame["model_prediction"].to_numpy(float), 1e-8, 1 - 1e-8)
    prior = np.clip(frame["prior_prediction"].to_numpy(float), 1e-8, 1 - 1e-8)
    model_point = _metrics(y, model)
    prior_point = _metrics(y, prior)
    rng = np.random.default_rng(int(random_state))
    records: list[tuple[float, float, float]] = []
    for _ in range(int(draws)):
        sampled = rng.choice(names, size=len(names), replace=True)
        indices = np.concatenate([groups[str(group)] for group in sampled])
        local_y = y[indices]
        model_metric = _metrics(local_y, model[indices])
        prior_metric = _metrics(local_y, prior[indices])
        records.append(
            (
                model_metric[0] - prior_metric[0],
                model_metric[1] - prior_metric[1],
                model_metric[2] - prior_metric[2],
            )
        )
    values = np.asarray(records, dtype=float)
    result: dict[str, Any] = {
        "rows": int(len(frame)),
        "groups": int(len(names)),
        "positive_rows": int(y.sum()),
        "prevalence": float(y.mean()),
        "model_brier": model_point[0],
        "prior_brier": prior_point[0],
        "delta_brier": model_point[0] - prior_point[0],
        "model_average_precision": model_point[1],
        "prior_average_precision": prior_point[1],
        "delta_average_precision": model_point[1] - prior_point[1],
        "model_roc_auc": model_point[2],
        "prior_roc_auc": prior_point[2],
        "delta_roc_auc": model_point[2] - prior_point[2],
    }
    for index, metric in enumerate(
        ("delta_brier", "delta_average_precision", "delta_roc_auc")
    ):
        finite = values[:, index][np.isfinite(values[:, index])]
        result[f"{metric}_ci_low"] = float(np.quantile(finite, 0.025))
        result[f"{metric}_ci_high"] = float(np.quantile(finite, 0.975))
        if metric == "delta_brier":
            result["bootstrap_probability_brier_improves"] = float(
                (finite < 0.0).mean()
            )
        else:
            result[f"bootstrap_probability_{metric}_improves"] = float(
                (finite > 0.0).mean()
            )
    return result


def _paired_arm(predictions: pd.DataFrame, arm: Mapping[str, Any]) -> pd.DataFrame:
    common = (
        predictions["setup"].eq(arm["setup"])
        & predictions["target_name"].eq(arm["target"])
        & predictions["feature_set"].eq(arm["feature_set"])
    )
    model = predictions.loc[common & predictions["model"].eq(arm["model"])].copy()
    prior = predictions.loc[common & predictions["model"].eq("prior")].copy()
    provenance = arm.get("provenance")
    if provenance:
        model = model.loc[model["mapping_provenance_role"].eq(provenance)]
        prior = prior.loc[prior["mapping_provenance_role"].eq(provenance)]
    keys = [
        "cohort_anchor_utc",
        "horizon_hours",
        "source_family",
        "cv_group_id",
    ]
    if model.duplicated(keys).any() or prior.duplicated(keys).any():
        raise ValueError(f"bootstrap arm has duplicate identities: {arm['arm']}")
    paired = model.loc[:, [*keys, "target", "prediction"]].merge(
        prior.loc[:, [*keys, "target", "prediction"]],
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_model", "_prior"),
    )
    if len(paired) != len(model) or len(paired) != len(prior):
        raise ValueError(f"bootstrap arm model/prior coverage differs: {arm['arm']}")
    if not paired["target_model"].eq(paired["target_prior"]).all():
        raise ValueError(f"bootstrap arm target parity fails: {arm['arm']}")
    return paired.rename(
        columns={
            "target_model": "target",
            "prediction_model": "model_prediction",
            "prediction_prior": "prior_prediction",
        }
    ).drop(columns="target_prior")


def run(args: argparse.Namespace) -> dict[str, Any]:
    ablation = Path(args.ablation)
    prediction_path = ablation / "grouped_oof_predictions.parquet"
    manifest_path = ablation / "manifest.json"
    sidecar = ablation / "manifest.sha256"
    if not all(path.is_file() for path in (prediction_path, manifest_path, sidecar)):
        raise FileNotFoundError("transition classifier artifact is incomplete")
    if sidecar.read_text(encoding="utf-8").split()[0] != sha256(manifest_path):
        raise ValueError("transition classifier manifest checksum fails")
    predictions = pd.read_parquet(prediction_path)
    records: list[dict[str, Any]] = []
    for index, arm in enumerate(FROZEN_ARMS):
        result = paired_block_bootstrap(
            _paired_arm(predictions, arm),
            draws=int(args.draws),
            random_state=RANDOM_STATE + index,
        )
        records.append({**dict(arm), **result})
    summary = pd.DataFrame(records)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.")
    )
    summary_path = temporary / "bootstrap_summary.csv"
    summary.to_csv(summary_path, index=False)
    manifest = {
        "schema": SCHEMA,
        "status": "PAIRED_UTC7D_BLOCK_BOOTSTRAP_COMPLETE",
        "draws": int(args.draws),
        "arms": [dict(arm) for arm in FROZEN_ARMS],
        "selection_disclosure": "exploratory arms frozen from the completed classifier ablation; intervals do not correct for that prior arm selection",
        "source": {
            "predictions": str(prediction_path),
            "sha256": sha256(prediction_path),
        },
        "outputs": {
            "summary": {
                "path": summary_path.name,
                "sha256": sha256(summary_path),
            }
        },
    }
    (temporary / "manifest.json").write_text(
        json.dumps(_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (temporary / "manifest.sha256").write_text(
        f"{sha256(temporary / 'manifest.json')}  manifest.json\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return {"output": str(output), "arms": len(summary), "draws": int(args.draws)}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--ablation", type=Path, default=DEFAULT_ABLATION)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    return result


def main() -> None:
    print(json.dumps(_safe(run(parser().parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
