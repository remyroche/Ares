#!/usr/bin/env python3
"""Audit score dispersion and tie-aware top-decile metrics without refitting.

Constant or cutoff-tied calibrated predictions do not define a ranking.  The
original deterministic timestamp tie-break is reproducible, but its realized
precision is not model discrimination.  This audit reports the expected
precision and exact best/worst bounds under uniform allocation within the
cutoff plateau, and marks zero-shrink/constant arms non-ranking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLASSIFIER = ROOT / "data_perp/artifacts/pooled_historical_current_transition_classifier_20260730_v1"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/pooled_historical_current_transition_score_tie_audit_20260730_v1"
SCHEMA = "pooled_historical_current_transition_score_tie_audit_v1"


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
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def tie_aware_top10(frame: pd.DataFrame) -> dict[str, Any]:
    score = pd.to_numeric(frame["prediction"], errors="raise").to_numpy(float)
    target = pd.to_numeric(frame["target"], errors="raise").to_numpy(float)
    if not np.isfinite(score).all() or not np.isfinite(target).all():
        raise ValueError("tie audit needs finite scores and targets")
    rows = len(frame)
    selected_rows = max(1, int(math.ceil(0.10 * rows)))
    cutoff = float(np.sort(score)[-selected_rows])
    above = score > cutoff
    plateau = score == cutoff
    above_rows, plateau_rows = int(above.sum()), int(plateau.sum())
    needed = selected_rows - above_rows
    if needed < 0 or needed > plateau_rows:
        raise ValueError("cutoff plateau accounting is inconsistent")
    above_positive = int(target[above].sum())
    plateau_positive = int(target[plateau].sum())
    expected_positive = above_positive + needed * plateau_positive / plateau_rows
    lower_positive = above_positive + max(0, needed - (plateau_rows - plateau_positive))
    upper_positive = above_positive + min(needed, plateau_positive)
    prevalence = float(target.mean())
    selected = frame["selected_top10"].astype(bool).to_numpy() if "selected_top10" in frame else np.zeros(rows, dtype=bool)
    original_precision = float(target[selected].mean()) if selected.any() else float("nan")
    expected_precision = float(expected_positive / selected_rows)
    return {
        "rows": rows, "positive_rows": int(target.sum()), "unweighted_prevalence": prevalence,
        "selected_rows": selected_rows, "cutoff": cutoff, "strictly_above_rows": above_rows,
        "cutoff_plateau_rows": plateau_rows, "cutoff_plateau_fraction": plateau_rows / rows,
        "needed_from_plateau": needed, "cutoff_is_ambiguous": bool(plateau_rows > needed),
        "original_timestamp_tiebreak_precision": original_precision,
        "tie_aware_expected_precision": expected_precision,
        "tie_aware_lower_precision": float(lower_positive / selected_rows),
        "tie_aware_upper_precision": float(upper_positive / selected_rows),
        "tie_aware_expected_lift_unweighted": float(expected_precision / prevalence) if prevalence else float("nan"),
    }


def dispersion(frame: pd.DataFrame) -> dict[str, Any]:
    score = pd.to_numeric(frame["prediction"], errors="raise").to_numpy(float)
    rounded = np.round(score, 12)
    _, counts = np.unique(rounded, return_counts=True)
    shrink = pd.to_numeric(frame["calibration_shrinkage_weight"], errors="coerce")
    return {
        "score_min": float(score.min()), "score_max": float(score.max()),
        "score_range": float(score.max() - score.min()), "score_std": float(score.std()),
        "unique_scores_12dp": int(len(counts)), "largest_score_tie_rows_12dp": int(counts.max()),
        "largest_score_tie_fraction_12dp": float(counts.max() / len(frame)),
        "zero_shrink_row_fraction": float(shrink.eq(0.0).mean()),
        "shrinkage_min": float(shrink.min()), "shrinkage_max": float(shrink.max()),
        "ranking_informative": bool(len(counts) > 1 and score.max() - score.min() > 1e-12 and not shrink.eq(0.0).all()),
    }


def audit_predictions(grouped: pd.DataFrame, transfer: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    arms: list[dict[str, Any]] = []
    compositions: list[dict[str, Any]] = []
    for keys, local in grouped.groupby(["target_name", "model"], sort=True):
        record = {"evaluation_kind": "grouped_oof", "target": keys[0], "model": keys[1], "train_source": "POOLED", "evaluation_source": "POOLED"}
        record.update(dispersion(local)); record.update(tie_aware_top10(local)); arms.append(record)
        cutoff = record["cutoff"]
        for source, source_rows in local.groupby("source_family", sort=True):
            compositions.append({"evaluation_kind": "grouped_oof", "target": keys[0], "model": keys[1], "train_source": "POOLED", "evaluation_source": "POOLED", "source_family": source, "rows": int(len(source_rows)), "selected_rows_original": int(source_rows["selected_top10"].sum()), "strictly_above_cutoff_rows": int(source_rows["prediction"].gt(cutoff).sum()), "cutoff_plateau_rows": int(source_rows["prediction"].eq(cutoff).sum()), "score_std": float(source_rows["prediction"].std(ddof=0))})
    for keys, local in transfer.groupby(["target_name", "train_source", "evaluation_source", "model"], sort=True):
        record = {"evaluation_kind": "source_transfer", "target": keys[0], "train_source": keys[1], "evaluation_source": keys[2], "model": keys[3]}
        record.update(dispersion(local)); record.update(tie_aware_top10(local)); arms.append(record)
    arms_frame = pd.DataFrame(arms)
    invalid = arms_frame.loc[~arms_frame["ranking_informative"] | arms_frame["cutoff_is_ambiguous"]].copy()
    return arms_frame, pd.DataFrame(compositions), invalid


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.classifier)
    manifest_path, sidecar = source / "manifest.json", source / "manifest.sha256"
    grouped_path, transfer_path = source / "grouped_oof_predictions.parquet", source / "source_transfer_predictions.parquet"
    if not all(path.is_file() for path in (manifest_path, sidecar, grouped_path, transfer_path)):
        raise FileNotFoundError("classifier artifact is incomplete")
    if sidecar.read_text(encoding="utf-8").split()[0] != sha256(manifest_path):
        raise ValueError("classifier manifest checksum fails")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for path in (grouped_path, transfer_path):
        if manifest.get("outputs_sha256", {}).get(path.name) != sha256(path):
            raise ValueError(f"classifier output checksum fails: {path.name}")
    arms, compositions, invalid = audit_predictions(pd.read_parquet(grouped_path), pd.read_parquet(transfer_path))
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}."))
    paths = {"score_tie_audit.csv": arms, "grouped_selection_composition.csv": compositions, "nonranking_or_tied_arms.csv": invalid}
    for name, value in paths.items():
        value.to_csv(temporary / name, index=False)
    output_manifest = {
        "schema": SCHEMA, "status": "IMMUTABLE_TIE_AWARE_AUDIT_COMPLETE",
        "arms": int(len(arms)), "nonranking_or_tied_arms": int(len(invalid)),
        "contracts": {
            "no_refit": "reads frozen grouped-OOF and source-transfer predictions only",
            "constant_scores": "zero-shrink/constant predictions have no ranking; deterministic timestamp top10 precision is not model evidence",
            "tie_aware": "expected precision and exact lower/upper bounds allocate the required cutoff slots within the tied plateau; lift uses the arm's unweighted prevalence",
            "promotion": "audit only; supersedes interpretation of timestamp-tie-broken top10 lift but does not alter frozen predictions",
        },
        "source": {"classifier": str(source), "manifest_sha256": sha256(manifest_path), "grouped_sha256": sha256(grouped_path), "transfer_sha256": sha256(transfer_path)},
        "outputs_sha256": {name: sha256(temporary / name) for name in paths},
        "promotion_eligible": False,
    }
    _write_json(temporary / "manifest.json", output_manifest)
    (temporary / "manifest.sha256").write_text(f"{sha256(temporary / 'manifest.json')}  manifest.json\n")
    os.replace(temporary, output)
    return output_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier", type=Path, default=DEFAULT_CLASSIFIER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(_safe(run(parse_args())), indent=2, sort_keys=True))
