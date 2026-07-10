"""Frozen live policy-archetype assignment for replay parity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


ARTIFACT_FILENAME = "live_policy_archetype_classifier.joblib"
MANIFEST_FILENAME = "live_policy_archetype_classifier_manifest.json"


def _side_norm(side: Any) -> str:
    text = str(side or "").strip().lower()
    return "short" if text.startswith("short") else "long"


def normalize_policy_archetype_label(side: Any, value: Any) -> str:
    """Return the policy-threshold key form used by replay artifacts."""
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        return ""
    side_name = _side_norm(side)
    if text.startswith(f"{side_name}__"):
        return text
    if text.startswith("long__") or text.startswith("short__"):
        return text
    return f"{side_name}__{text}"


def _artifact_paths(data_root: str | Path, run_id: str) -> tuple[Path, Path]:
    root = Path(data_root) / "artifacts" / str(run_id) / "policy_params"
    return root / ARTIFACT_FILENAME, root / MANIFEST_FILENAME


def load_live_policy_archetype_classifier(
    *,
    data_root: str | Path,
    run_id: str,
    warn: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    """Load the frozen classifier payload if present.

    Missing artifacts are non-fatal because older policies may not use
    archetype-specific HR modulation.
    """
    artifact_path, manifest_path = _artifact_paths(data_root, run_id)
    if not artifact_path.exists():
        if warn is not None:
            warn(f"Live policy archetype classifier not found: {artifact_path}")
        return {}
    try:
        import joblib

        payload = joblib.load(artifact_path)
        if not isinstance(payload, Mapping):
            raise TypeError(f"expected mapping payload, got {type(payload)!r}")
        payload = dict(payload)
        payload["artifact_path"] = str(artifact_path)
        if manifest_path.exists():
            try:
                payload["manifest"] = json.loads(manifest_path.read_text())
            except Exception:
                payload["manifest"] = {}
        return payload
    except Exception as exc:
        if warn is not None:
            warn(f"Failed to load live policy archetype classifier {artifact_path}: {exc}")
        return {}


def _first_row_mapping(frame: Optional[pd.DataFrame]) -> dict[str, Any]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return {}
    row = frame.iloc[0]
    return {str(k): row.get(k) for k in row.index}


def _coerce_single_row(
    *,
    side: Any,
    feature_columns: Sequence[str],
    medians: Mapping[str, Any],
    candidate_feature_row: Optional[pd.DataFrame],
    meta_model_input_row: Optional[pd.DataFrame],
) -> pd.DataFrame:
    raw: dict[str, Any] = {}
    raw.update(_first_row_mapping(candidate_feature_row))
    raw.update(_first_row_mapping(meta_model_input_row))
    side_name = _side_norm(side)
    raw["__live_side_is_long"] = 1.0 if side_name == "long" else 0.0
    raw["__live_side_is_short"] = 1.0 if side_name == "short" else 0.0

    values: dict[str, float] = {}
    for col in feature_columns:
        val = raw.get(col, medians.get(col, 0.0))
        try:
            fval = float(val)
        except (TypeError, ValueError):
            fval = float(medians.get(col, 0.0) or 0.0)
        if not np.isfinite(fval):
            try:
                fval = float(medians.get(col, 0.0) or 0.0)
            except (TypeError, ValueError):
                fval = 0.0
        values[col] = fval
    return pd.DataFrame([values], columns=list(feature_columns))


def predict_live_policy_archetype(
    *,
    side: Any,
    payload: Mapping[str, Any],
    candidate_feature_row: Optional[pd.DataFrame] = None,
    meta_model_input_row: Optional[pd.DataFrame] = None,
) -> str:
    """Predict a replay-compatible side-prefixed policy archetype."""
    if not isinstance(payload, Mapping) or not payload:
        return ""
    model = payload.get("model")
    feature_columns = list(payload.get("feature_columns") or [])
    if model is None or not feature_columns:
        return ""
    X = _coerce_single_row(
        side=side,
        feature_columns=feature_columns,
        medians=payload.get("feature_medians") or {},
        candidate_feature_row=candidate_feature_row,
        meta_model_input_row=meta_model_input_row,
    )
    try:
        pred = model.predict(X)
    except Exception:
        return ""
    if pred is None or len(pred) == 0:
        return ""
    label = normalize_policy_archetype_label(side, pred[0])
    side_name = _side_norm(side)
    if label.startswith(f"{side_name}__"):
        return label
    side_defaults = payload.get("side_defaults") or {}
    return normalize_policy_archetype_label(side, side_defaults.get(side_name, ""))
