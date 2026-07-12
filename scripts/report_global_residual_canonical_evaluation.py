#!/usr/bin/env python3
"""Materialize calibrated canonical HPO metrics from a completed state run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_global_residual_champion_enhancement import (  # noqa: E402
    KEY_COLUMNS,
    OUTCOME_COLUMNS,
    _revision_summary,
    _score_external_prediction_vector,
)

DEFAULT_RUN = ROOT / (
    "data_perp/reports/global_residual_state_discovery_20260711_v1/"
    "champion_greedy_enhancement_local_phasefs_20260712_v2"
)
DEFAULT_COMPACT = ROOT / (
    "data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1/"
    "cache/compact_reference_with_lifecycle.parquet"
)
DEFAULT_JULY = ROOT / (
    "data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1/"
    "champion_frozen_single_source_202501_20260710/prediction_shards/"
    "predictions_2026-07.parquet"
)


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _columns(path: Path) -> set[str]:
    return set(pq.ParquetFile(path).schema_arrow.names)


def _load_rows(path: Path) -> pd.DataFrame:
    available = _columns(path)
    requested = [
        name
        for name in (
            *KEY_COLUMNS,
            "archetype_label_family",
            *OUTCOME_COLUMNS,
            "score_meta_base_soft_label",
            "__first_touch_target_soft__",
        )
        if name in available
    ]
    frame = pd.read_parquet(path, columns=requested)
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    frame["side_name"] = frame["side_name"].astype(str).str.lower()
    frame["archetype_policy_key"] = frame["archetype_policy_key"].astype(str)
    return frame.sort_values(
        ["__ts__", "__symbol__", "side_name"], kind="stable"
    ).reset_index(drop=True)


def _alignment_audit(run_root: Path, final_rows: pd.DataFrame) -> dict[str, Any]:
    comparison_path = run_root / "final_test_predictions.parquet"
    if not comparison_path.exists():
        return {"available": False}
    expected = pd.read_parquet(comparison_path, columns=list(KEY_COLUMNS))
    expected["__ts__"] = pd.to_datetime(expected["__ts__"], utc=True, errors="coerce")
    received = final_rows.loc[:, list(KEY_COLUMNS)].reset_index(drop=True)
    expected = expected.reset_index(drop=True)
    matches = len(expected) == len(received)
    if matches:
        matches = bool(
            np.array_equal(
                pd.util.hash_pandas_object(expected, index=False).to_numpy(),
                pd.util.hash_pandas_object(received, index=False).to_numpy(),
            )
        )
    return {
        "available": True,
        "expected_rows": int(len(expected)),
        "received_rows": int(len(received)),
        "exact_order_match": matches,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_root = Path(args.run_root)
    train_end = pd.Timestamp(args.train_end, tz="UTC")
    selection_start = pd.Timestamp(args.selection_start, tz="UTC")
    selection_end = pd.Timestamp(args.selection_end, tz="UTC")
    evaluation_end = pd.Timestamp(args.evaluation_end, tz="UTC")

    historical = _load_rows(Path(args.compact))
    train = historical.loc[historical["__ts__"].lt(train_end)].reset_index(drop=True)
    selection = historical.loc[
        historical["__ts__"].ge(selection_start)
        & historical["__ts__"].lt(selection_end)
    ].reset_index(drop=True)
    final_rows = _load_rows(Path(args.july))
    final_rows = final_rows.loc[
        final_rows["__ts__"].ge(selection_end) & final_rows["__ts__"].lt(evaluation_end)
    ].reset_index(drop=True)
    for name in selection.columns:
        if name not in final_rows:
            final_rows[name] = np.nan
    final_rows = final_rows.reindex(columns=selection.columns)
    evaluation = pd.concat(
        [selection, final_rows], ignore_index=True, sort=False, copy=False
    )

    canonical_root = run_root / "canonical_final_fit"
    model = joblib.load(canonical_root / "model.joblib")
    predictions = np.load(canonical_root / "evaluation_predictions.npy")
    scored = _score_external_prediction_vector(
        train,
        evaluation,
        alternative_score=predictions,
        alternative_train_oof_score=np.asarray(model.oof_probs),
        arm="canonical_final_hpo",
    )
    metrics, summary = _revision_summary(scored, "canonical_final_hpo")
    selection_scored = scored.loc[scored["__ts__"].lt(selection_end)].reset_index(
        drop=True
    )
    selection_metrics, selection_summary = _revision_summary(
        selection_scored, "canonical_final_hpo"
    )
    final_scored = scored.loc[scored["__ts__"].ge(selection_end)].reset_index(drop=True)
    final_metrics, final_summary = _revision_summary(
        final_scored, "canonical_final_hpo"
    )
    alignment = _alignment_audit(run_root, final_rows)
    if not alignment.get("exact_order_match", False):
        raise ValueError(f"Canonical evaluation row alignment failed: {alignment}")

    scored.to_parquet(
        canonical_root / "evaluation_scored.parquet",
        index=False,
        compression="zstd",
    )
    metrics.to_csv(canonical_root / "evaluation_metrics.csv", index=False)
    selection_metrics.to_csv(canonical_root / "selection_metrics.csv", index=False)
    final_metrics.to_csv(canonical_root / "final_test_metrics.csv", index=False)
    report = {
        "schema": "global_residual_canonical_evaluation_v1",
        "train_end_exclusive": train_end,
        "selection_start": selection_start,
        "selection_end_exclusive": selection_end,
        "evaluation_end_exclusive": evaluation_end,
        "train_rows": int(len(train)),
        "selection_rows": int(len(selection)),
        "final_test_rows": int(len(final_rows)),
        "evaluation_rows": int(len(evaluation)),
        "train_oof_finite_rows": int(np.isfinite(np.asarray(model.oof_probs)).sum()),
        "alignment": alignment,
        "evaluation_summary": summary,
        "selection_summary": selection_summary,
        "final_test_summary": final_summary,
        "leakage_contract": (
            "Platt calibration uses only sparse forward OOF train scores through the "
            "purged train boundary. April-June and July predictions are frozen OOS scores."
        ),
    }
    (canonical_root / "evaluation_report.json").write_text(
        json.dumps(_safe(report), indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--compact", type=Path, default=DEFAULT_COMPACT)
    parser.add_argument("--july", type=Path, default=DEFAULT_JULY)
    parser.add_argument("--train-end", default="2026-03-31 12:00:00")
    parser.add_argument("--selection-start", default="2026-04-01")
    parser.add_argument("--selection-end", default="2026-07-01")
    parser.add_argument("--evaluation-end", default="2026-07-11")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(_safe(run(parse_args())), indent=2, sort_keys=True))
