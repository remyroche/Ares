#!/usr/bin/env python3
"""Train on one transition era and evaluate unchanged on another era."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from scripts.run_cross_era_regime_transition_classifier_ablation import (
        ECONOMIC_TARGETS,
        _metric_record,
        _nested_shrunk_prediction,
        feature_sets,
    )
except ModuleNotFoundError:
    from run_cross_era_regime_transition_classifier_ablation import (
        ECONOMIC_TARGETS,
        _metric_record,
        _nested_shrunk_prediction,
        feature_sets,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = ROOT / (
    "data_perp/artifacts/cross_era_global_book_transition_research_panel_"
    "20260729_v3"
)
DEFAULT_OUTPUT = ROOT / (
    "data_perp/artifacts/cross_era_transition_source_transfer_20260729_v1"
)
SCHEMA = "cross_era_transition_source_transfer_v1"
SOURCE_SETUPS = {
    "reconstructed_fee_only": "reconstructed_exact1m_janapr2025",
    "canonical_spread": "canonical_spread_febapr2025",
    "current_exact_spread_strict_oof": "current_exact_spread_mayjul2026",
}


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


def _source_frame(panel: pd.DataFrame, setup: str) -> pd.DataFrame:
    family = SOURCE_SETUPS[setup]
    result = panel.loc[
        panel["source_family"].eq(family)
        & panel["horizon_hours"].eq(12)
        & panel["context_available"].astype(bool)
    ].copy()
    if setup == "current_exact_spread_strict_oof":
        result = result.loc[
            result["mapping_provenance_role"].eq("strict_oof")
        ].copy()
    return result


def _prior_top10_selection(frame: pd.DataFrame) -> np.ndarray:
    """Return a deterministic score-independent tie break for a constant prior."""
    count = max(1, int(math.ceil(0.10 * len(frame))))
    tie_columns = [
        column
        for column in (
            "cohort_anchor_utc",
            "source_family",
            "mapping_provenance_role",
        )
        if column in frame.columns
    ]
    tie_hash = pd.util.hash_pandas_object(
        frame.loc[:, tie_columns], index=True
    ).to_numpy(dtype=np.uint64)
    selected = np.zeros(len(frame), dtype=bool)
    selected[np.argsort(tie_hash, kind="stable")[:count]] = True
    return selected


def run_transfer_matrix(
    panel: pd.DataFrame,
    feature_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    families = feature_sets(feature_columns)
    metrics: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    skipped: list[dict[str, Any]] = []
    for train_setup in SOURCE_SETUPS:
        train_source = _source_frame(panel, train_setup)
        for evaluation_setup in SOURCE_SETUPS:
            if evaluation_setup == train_setup:
                continue
            evaluation_source = _source_frame(panel, evaluation_setup)
            for target in ECONOMIC_TARGETS:
                train_valid = train_source[target].notna()
                evaluation_valid = evaluation_source[target].notna()
                train = train_source.loc[train_valid].reset_index(drop=True)
                evaluation = evaluation_source.loc[
                    evaluation_valid
                ].reset_index(drop=True)
                train_y = pd.to_numeric(train[target], errors="raise").astype(int)
                evaluation_y = pd.to_numeric(
                    evaluation[target], errors="raise"
                ).astype(int)
                if (
                    train_y.nunique() < 2
                    or evaluation_y.nunique() < 2
                    or train["cv_group_id"].nunique() < 5
                ):
                    skipped.append(
                        {
                            "train_setup": train_setup,
                            "evaluation_setup": evaluation_setup,
                            "target": target,
                            "reason": "insufficient_binary_or_group_support",
                        }
                    )
                    continue
                for feature_name, columns in families.items():
                    usable = [
                        column
                        for column in columns
                        if train[column].notna().sum()
                        >= max(20, int(0.50 * len(train)))
                        and pd.to_numeric(
                            train[column], errors="coerce"
                        ).nunique(dropna=True)
                        > 1
                    ]
                    if not usable:
                        skipped.append(
                            {
                                "train_setup": train_setup,
                                "evaluation_setup": evaluation_setup,
                                "target": target,
                                "feature_set": feature_name,
                                "reason": "no_usable_training_features",
                            }
                        )
                        continue
                    prior = float(train_y.mean())
                    for model_name in (
                        "logistic_shrunk",
                        "extra_trees_shrunk",
                    ):
                        try:
                            score, weight = _nested_shrunk_prediction(
                                train,
                                train_y,
                                evaluation,
                                columns=usable,
                                base_model_name=model_name.removesuffix(
                                    "_shrunk"
                                ),
                            )
                        except ValueError as error:
                            skipped.append(
                                {
                                    "train_setup": train_setup,
                                    "evaluation_setup": evaluation_setup,
                                    "target": target,
                                    "feature_set": feature_name,
                                    "model": model_name,
                                    "reason": str(error),
                                }
                            )
                            continue
                        count = max(1, int(math.ceil(0.10 * len(evaluation))))
                        order = np.lexsort(
                            (
                                evaluation["cohort_anchor_utc"]
                                .astype("int64")
                                .to_numpy(),
                                -score,
                            )
                        )
                        selected = np.zeros(len(evaluation), dtype=bool)
                        selected[order[:count]] = True
                        base_columns = [
                            "cohort_anchor_utc",
                            "horizon_hours",
                            "source_family",
                            "economics_tier",
                            "mapping_provenance_role",
                            "cv_group_id",
                        ]
                        prediction = evaluation.loc[:, base_columns].copy()
                        prediction["target_name"] = target
                        prediction["target"] = evaluation_y.to_numpy(float)
                        prediction["prediction"] = score
                        prediction["selected_top10"] = selected
                        prediction["actual_onset"] = pd.to_numeric(
                            evaluation["target__adverse_onset"],
                            errors="coerce",
                        ).to_numpy(float)
                        prediction["train_setup"] = train_setup
                        prediction["evaluation_setup"] = evaluation_setup
                        prediction["feature_set"] = feature_name
                        prediction["model"] = model_name
                        prediction["calibration_shrinkage_weight"] = weight
                        predictions.append(prediction)
                        setup_name = f"{train_setup}=>{evaluation_setup}"
                        metrics.append(
                            {
                                **_metric_record(
                                    prediction,
                                    setup=setup_name,
                                    horizon=12,
                                    target=target,
                                    feature_set=feature_name,
                                    model=model_name,
                                    scope="transfer",
                                ),
                                "train_setup": train_setup,
                                "evaluation_setup": evaluation_setup,
                                "train_rows": int(len(train)),
                                "train_positive_rows": int(train_y.sum()),
                                "train_prevalence": prior,
                                "calibration_shrinkage_weight": weight,
                            }
                        )
                        prior_prediction = prediction.copy()
                        prior_prediction["prediction"] = prior
                        prior_prediction["selected_top10"] = (
                            _prior_top10_selection(evaluation)
                        )
                        metrics.append(
                            {
                                **_metric_record(
                                    prior_prediction,
                                    setup=setup_name,
                                    horizon=12,
                                    target=target,
                                    feature_set=feature_name,
                                    model="train_prior",
                                    scope="transfer",
                                ),
                                "train_setup": train_setup,
                                "evaluation_setup": evaluation_setup,
                                "train_rows": int(len(train)),
                                "train_positive_rows": int(train_y.sum()),
                                "train_prevalence": prior,
                                "calibration_shrinkage_weight": 0.0,
                            }
                        )
    if not predictions:
        raise ValueError("no source-transfer arm was evaluable")
    return (
        pd.DataFrame(metrics),
        pd.concat(predictions, ignore_index=True),
        pd.DataFrame(skipped),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    panel_root = Path(args.panel)
    panel_path = panel_root / "transition_research_panel.parquet"
    manifest_path = panel_root / "manifest.json"
    sidecar = panel_root / "manifest.sha256"
    if not all(path.is_file() for path in (panel_path, manifest_path, sidecar)):
        raise FileNotFoundError("cross-era transition panel is incomplete")
    if sidecar.read_text(encoding="utf-8").split()[0] != sha256(manifest_path):
        raise ValueError("cross-era transition panel manifest checksum fails")
    panel_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics, predictions, skipped = run_transfer_matrix(
        pd.read_parquet(panel_path), panel_manifest["feature_columns"]
    )
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.")
    )
    paths = {
        "metrics": temporary / "transfer_metrics.csv",
        "predictions": temporary / "transfer_predictions.parquet",
        "skipped": temporary / "skipped_arms.csv",
    }
    metrics.to_csv(paths["metrics"], index=False)
    predictions.to_parquet(
        paths["predictions"], index=False, compression="zstd"
    )
    skipped.to_csv(paths["skipped"], index=False)
    manifest = {
        "schema": SCHEMA,
        "status": "SOURCE_TO_SOURCE_TRANSFER_DIAGNOSTIC_COMPLETE",
        "contracts": {
            "training": "one source family only; current training excludes frozen forward-OOS rows",
            "calibration": "three-fold grouped/36h-purged training-source OOF chooses shrinkage before the evaluation source is scored",
            "evaluation": "entire destination source scored once; reverse-time arms are symmetric diagnosis, never promotion evidence",
            "features": "source ID/calendar/outcomes excluded; fixed common raw-state and causal-coordinate families",
            "economics": "fee-only and spread-aware labels remain separate; no raw-PnL pooling",
        },
        "metric_rows": int(len(metrics)),
        "prediction_rows": int(len(predictions)),
        "skipped_rows": int(len(skipped)),
        "source": {
            "panel": str(panel_path),
            "sha256": sha256(panel_path),
        },
        "outputs": {
            name: {"path": path.name, "sha256": sha256(path)}
            for name, path in paths.items()
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
    return {
        "output": str(output),
        "metric_rows": int(len(metrics)),
        "prediction_rows": int(len(predictions)),
        "skipped_rows": int(len(skipped)),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return result


def main() -> None:
    print(json.dumps(_safe(run(parser().parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
