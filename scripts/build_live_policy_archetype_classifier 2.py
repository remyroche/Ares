#!/usr/bin/env python3
"""Build the frozen live policy-archetype classifier used by inference.

The classifier is a parity fallback for live rows that do not carry the
materialized replay `policy_archetype` field. It trains only on pre-entry
features and pre-OOS rows, then emits side-prefixed policy keys matching the
simple_policy_optimiser handoff.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from extreme_price_movements.inference.live_policy_archetype import (
    ARTIFACT_FILENAME,
    MANIFEST_FILENAME,
    normalize_policy_archetype_label,
)


SIDE_FEATURES = ["__live_side_is_long", "__live_side_is_short"]


def _read_feature_names(columns_path: Path) -> list[str]:
    payload = json.loads(columns_path.read_text())
    names = payload.get("feature_names") or payload.get("features") or []
    if not isinstance(names, list) or not names:
        raise ValueError(f"no feature_names found in {columns_path}")
    return [str(c) for c in names]


def _coerce_target(frame: pd.DataFrame) -> pd.Series:
    side = frame.get("side_name")
    if side is None:
        side = frame.get("side")
    if side is None:
        raise ValueError("handoff is missing side_name/side")
    source = None
    for col in (
        "policy_archetype",
        "archetype_policy_key",
        "__archetype_policy_key__",
        "local_side_archetype",
    ):
        if col in frame.columns:
            source = frame[col]
            break
    if source is None:
        raise ValueError("handoff is missing policy archetype columns")
    return pd.Series(
        [
            normalize_policy_archetype_label(s, v)
            for s, v in zip(side.astype(str), source.astype(str))
        ],
        index=frame.index,
        name="policy_archetype_live_key",
    )


def _coerce_matrix(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    medians: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    out = pd.DataFrame(index=frame.index)
    side = frame.get("side_name")
    if side is None:
        side = frame.get("side")
    side_text = side.astype(str).str.lower() if side is not None else pd.Series("", index=frame.index)
    for col in feature_columns:
        if col == "__live_side_is_long":
            out[col] = side_text.str.startswith("long").astype(float).to_numpy()
        elif col == "__live_side_is_short":
            out[col] = side_text.str.startswith("short").astype(float).to_numpy()
        elif col in frame.columns:
            out[col] = pd.to_numeric(frame[col], errors="coerce")
        else:
            out[col] = np.nan
    if medians is None:
        medians = {}
        for col in feature_columns:
            values = pd.to_numeric(out[col], errors="coerce")
            med = float(values.median()) if values.notna().any() else 0.0
            if not np.isfinite(med):
                med = 0.0
            medians[col] = med
    for col in feature_columns:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        out[col] = out[col].fillna(float(medians.get(col, 0.0)))
    return out.astype(np.float32), medians


def _value_counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff", required=True, type=Path)
    parser.add_argument("--columns", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--train-end", default="2026-07-01T00:00:00Z")
    parser.add_argument("--valid-start", default="2026-07-01T00:00:00Z")
    parser.add_argument("--random-state", default=42, type=int)
    args = parser.parse_args()

    feature_columns = _read_feature_names(args.columns) + SIDE_FEATURES
    needed = set(feature_columns)
    needed.update(
        {
            "__ts__",
            "side_name",
            "side",
            "policy_archetype",
            "archetype_policy_key",
            "__archetype_policy_key__",
            "local_side_archetype",
        }
    )
    available_cols = pd.read_parquet(args.handoff, columns=None).columns
    read_cols = [c for c in available_cols if c in needed]
    frame = pd.read_parquet(args.handoff, columns=read_cols)
    if "__ts__" not in frame.columns:
        raise ValueError("handoff is missing __ts__")
    ts = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    y = _coerce_target(frame)
    valid_target = y.astype(str).str.len() > 0
    train_end = pd.Timestamp(args.train_end, tz="UTC")
    valid_start = pd.Timestamp(args.valid_start, tz="UTC")
    train_mask = (ts < train_end) & valid_target
    valid_mask = (ts >= valid_start) & valid_target
    if int(train_mask.sum()) < 1000:
        raise ValueError(f"not enough train rows: {int(train_mask.sum())}")

    X_train, medians = _coerce_matrix(
        frame.loc[train_mask],
        feature_columns=feature_columns,
        medians=None,
    )
    y_train = y.loc[train_mask].astype(str)

    try:
        from lightgbm import LGBMClassifier

        model: Any = LGBMClassifier(
            objective="multiclass",
            n_estimators=180,
            learning_rate=0.045,
            num_leaves=31,
            max_depth=7,
            min_child_samples=80,
            subsample=0.90,
            colsample_bytree=0.90,
            reg_alpha=0.01,
            reg_lambda=0.25,
            class_weight="balanced",
            random_state=args.random_state,
            n_jobs=-1,
            verbose=-1,
        )
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier

        model = HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.045,
            max_leaf_nodes=31,
            l2_regularization=0.25,
            random_state=args.random_state,
        )
    model.fit(X_train, y_train)

    metrics: dict[str, Any] = {
        "train_rows": int(train_mask.sum()),
        "valid_rows": int(valid_mask.sum()),
        "train_end_exclusive": train_end.isoformat(),
        "valid_start_inclusive": valid_start.isoformat(),
        "train_class_counts": _value_counts(y_train),
    }
    if int(valid_mask.sum()) > 0:
        X_valid, _ = _coerce_matrix(
            frame.loc[valid_mask],
            feature_columns=feature_columns,
            medians=medians,
        )
        y_valid = y.loc[valid_mask].astype(str)
        pred = pd.Series(model.predict(X_valid), index=y_valid.index).astype(str)
        metrics["valid_accuracy"] = float((pred == y_valid).mean())
        metrics["valid_class_counts"] = _value_counts(y_valid)
        metrics["valid_pred_counts"] = _value_counts(pred)
        metrics["valid_side_mismatch_rate"] = float(
            (
                pred.str.split("__", n=1).str[0].fillna("")
                != y_valid.str.split("__", n=1).str[0].fillna("")
            ).mean()
        )

    side_defaults = {}
    for side_name in ("long", "short"):
        s = y_train[y_train.str.startswith(f"{side_name}__")]
        if not s.empty:
            side_defaults[side_name] = str(s.value_counts().index[0])

    payload = {
        "schema": "live_policy_archetype_classifier_v1",
        "model": model,
        "feature_columns": feature_columns,
        "feature_medians": {str(k): float(v) for k, v in medians.items()},
        "classes": [str(c) for c in getattr(model, "classes_", [])],
        "side_defaults": side_defaults,
        "train_end_exclusive": train_end.isoformat(),
        "source_handoff": str(args.handoff),
        "source_columns": str(args.columns),
        "metrics": metrics,
    }
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = args.artifact_dir / ARTIFACT_FILENAME
    manifest_path = args.artifact_dir / MANIFEST_FILENAME
    import joblib

    joblib.dump(payload, artifact_path)
    manifest = {
        "schema": payload["schema"],
        "artifact_path": str(artifact_path),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "classes": payload["classes"],
        "side_defaults": side_defaults,
        "leakage_contract": {
            "target": "side-prefixed policy_archetype from train_meta handoff",
            "train_filter": f"__ts__ < {train_end.isoformat()}",
            "inference_inputs": "pre-entry selected meta features plus live side indicators",
            "oos_outcomes_used_at_inference": False,
        },
        "metrics": metrics,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
