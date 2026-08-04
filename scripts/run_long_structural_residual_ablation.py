#!/usr/bin/env python3
"""Fold-local long residual-ranker ablation for structural tree representations.

The structural sidecar already contains strictly OOF base scores and its
token-free explanations.  This runner deliberately keeps the residual target
on every candidate row::

    realised_net_bps - base_expected_bps

``query_id`` (decision timestamp × side) is used solely by LambdaRank.  The
reported tails are pooled globally after scoring; no path/support label is a
model feature, a fitting filter, or an evaluation admission rule.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.support_aware_residual_ablation import (
    bps_residual_grade,
    hybrid_economic_grade,
)
from scripts.run_long_only_executable_net_lambdarank import (
    _fit_bps_map,
    _fit_ranker,
    _predict_ranker,
    _tail_metrics,
    _write_json,
)


SCHEMA = "long_structural_residual_ablation_v1"
SIDE = "long"
TAILS = (0.01, 0.03, 0.05, 0.10)

_IDENTITY_COLUMNS = {
    "candidate_id", "__ts__", "side_name", "gross_bps", "net_bps",
    "label_valid", "barrier_relevance_0_5", "mfe_mae_label_valid", "atr_bps",
    "label_available_ts", "query_id", "month", "meta_partition", "fold",
    "feature_contract_sha256", "base_raw_score", "base_expected_bps",
}
_HISTORICAL_HEALTH = (
    "structural_health__active_posterior_mass",
    "structural_health__historical_log_support",
    "structural_health__historical_correctness",
    "structural_health__historical_residual_bps",
    "structural_health__historical_residual_std_bps",
)
_PORTABILITY_HEALTH = (
    "structural_health__completed_period_count",
    "structural_health__period_residual_std_bps",
    "structural_health__period_sign_reversal_rate",
    "structural_health__period_worst_residual_bps",
)
_COMPATIBILITY_HEALTH = (
    "structural_health__context_compatibility",
    "structural_health__contextual_residual_bps",
)


def _raw_aegmm_columns(columns: Iterable[str]) -> list[str]:
    """Return only the frozen raw MDA + AE/GMM contract from the sidecar."""

    fields = [
        str(column) for column in columns
        if str(column) not in _IDENTITY_COLUMNS
        and not str(column).startswith("base_reasoning__")
        and not str(column).startswith("base_structural_family__")
        and not str(column).startswith("structural_health__")
    ]
    forbidden = [
        field for field in fields
        if "support_h12" in field.lower() or "leaf" in field.lower()
    ]
    if forbidden:
        raise ValueError(f"raw/AE-GMM contract unexpectedly contains path/leaf fields: {forbidden}")
    if len(fields) < 30:
        raise ValueError("raw/AE-GMM contract is unexpectedly small")
    return fields


def feature_arms(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Construct the predeclared nested feature arms without raw leaf tokens."""

    raw_aegmm = _raw_aegmm_columns(frame.columns)
    reasoning = sorted(column for column in frame if column.startswith("base_reasoning__"))
    memberships = sorted(column for column in frame if column.startswith("base_structural_family__"))
    missing = sorted(
        set((*_HISTORICAL_HEALTH, *_PORTABILITY_HEALTH, *_COMPATIBILITY_HEALTH)).difference(frame.columns)
    )
    if missing:
        raise ValueError(f"structural sidecar misses declared health features: {missing}")
    if not reasoning or not memberships:
        raise ValueError("structural sidecar misses invariant base reasoning or direct family memberships")
    base = [*raw_aegmm, "base_expected_bps"]
    result = {
        "R0_raw_aegmm_base": base,
        "R1_reasoning_memberships": [*base, *reasoning, *memberships],
        "R2_historical_health": [*base, *reasoning, *memberships, *_HISTORICAL_HEALTH],
        "R3_portability_health": [*base, *reasoning, *memberships, *_PORTABILITY_HEALTH],
        "R4_compatibility_health": [*base, *reasoning, *memberships, *_COMPATIBILITY_HEALTH],
    }
    for name, fields in result.items():
        if len(fields) != len(set(fields)):
            raise ValueError(f"{name}: duplicate feature fields")
        raw_leaf = [
            field for field in fields
            if field.startswith("leaf_") or "leaf_index" in field.lower() or "leaf_id" in field.lower()
        ]
        if raw_leaf:
            raise ValueError(f"{name}: raw leaf token fields are prohibited: {raw_leaf}")
    return result


def _finite_target(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    net = pd.to_numeric(frame["net_bps"], errors="coerce").to_numpy(float)
    base = pd.to_numeric(frame["base_expected_bps"], errors="coerce").to_numpy(float)
    atr = pd.to_numeric(frame["atr_bps"], errors="coerce").to_numpy(float)
    if not np.isfinite(net).all() or not np.isfinite(base).all():
        raise ValueError("universal residual target needs finite realised net and base expectation")
    if not np.isfinite(atr).all() or (atr <= 0.0).any():
        raise ValueError("hybrid residual target needs positive decision-time ATR bps on every candidate")
    return net - base, (net - base) / atr


def _target_arms(train: pd.DataFrame) -> dict[str, np.ndarray]:
    residual_bps, residual_atr = _finite_target(train)
    return {
        "hybrid_50_150": hybrid_economic_grade(
            residual_atr, residual_bps, moderate_bps=50.0, severe_bps=150.0,
        ),
        "bps_50_150": bps_residual_grade(
            residual_bps, moderate_bps=50.0, severe_bps=150.0,
        ),
    }


def _validate_fold_partitions(frame: pd.DataFrame, fold: str) -> None:
    required = {"meta_train", "meta_calibration", "test"}
    actual = set(frame["meta_partition"].astype(str))
    if actual != required:
        raise ValueError(f"{fold}: expected exactly {sorted(required)}, got {sorted(actual)}")
    train = frame.loc[frame["meta_partition"].eq("meta_train")]
    calibration = frame.loc[frame["meta_partition"].eq("meta_calibration")]
    test = frame.loc[frame["meta_partition"].eq("test")]
    if min(len(train), len(calibration), len(test)) < 2:
        raise ValueError(f"{fold}: insufficient partition rows")
    if train["__ts__"].max() >= calibration["__ts__"].min():
        raise ValueError(f"{fold}: meta train decisions do not precede calibration")
    # Calibration labels must resolve before the test starts.  Meta-train is
    # separately purged against calibration, preserving the small boundary
    # overlap in the source sidecar for audit rather than concealing it.
    if calibration["label_available_ts"].max() >= test["__ts__"].min():
        raise ValueError(f"{fold}: calibration overlaps the test label horizon")


def _purged_meta_train(train: pd.DataFrame, calibration: pd.DataFrame, fold: str) -> pd.DataFrame:
    """Remove only H12-unresolved training rows at the calibration boundary."""

    calibration_start = calibration["__ts__"].min()
    purged = train.loc[train["label_available_ts"] < calibration_start].copy()
    if len(purged) < 2:
        raise ValueError(f"{fold}: horizon purge leaves too few meta training rows")
    if purged["label_available_ts"].max() >= calibration_start:
        raise AssertionError(f"{fold}: residual meta horizon purge failed")
    return purged


def _tail_metrics_with_fold_rows(predictions: pd.DataFrame, score: str) -> list[dict]:
    """Use canonical pooled/month metrics, then add identically global fold rows."""

    rows = []
    for row in _tail_metrics(predictions, score):
        rows.append({**row, "period_scope": "global" if row["period"] == "pooled" else "month"})
    for fold, block in predictions.groupby("fold", sort=True, observed=True):
        ordered = block.sort_values(score, ascending=False, kind="stable")
        for tail in TAILS:
            count = max(1, int(np.ceil(len(ordered) * tail)))
            selected = ordered.iloc[:count]
            rows.append({
                "score": score, "period": str(fold), "period_scope": "fold", "tail": tail,
                "trades": count,
                "gross_bps_per_trade": float(selected["gross_bps"].mean()),
                "net_bps_per_trade": float(selected["net_bps"].mean()),
                "win_rate_net": float((selected["net_bps"] > 0.0).mean()),
            })
    return rows


def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    required = _IDENTITY_COLUMNS.difference({"month"})
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"structural sidecar misses required columns: {missing}")
    # Rolling folds intentionally reuse historical candidates.  Identity must
    # instead be unique inside a fitted base/fold contract.
    if frame.duplicated(["fold", "candidate_id"]).any():
        raise ValueError("structural sidecar duplicates candidate identities within a fold")
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    frame["label_available_ts"] = pd.to_datetime(frame["label_available_ts"], utc=True, errors="coerce")
    if frame[["__ts__", "label_available_ts"]].isna().any().any():
        raise ValueError("sidecar has invalid UTC timestamps")
    if not frame["label_available_ts"].ge(frame["__ts__"]).all():
        raise ValueError("sidecar label availability predates a decision")
    if not frame["side_name"].astype(str).str.lower().eq(SIDE).all():
        raise ValueError("runner is explicitly long-only")
    # The strict reasoning sidecar intentionally carries no redundant monthly
    # display column; derive it only for diagnostics after all causal scoring.
    frame["month"] = frame["__ts__"].dt.strftime("%Y-%m")
    if "support_h12" in frame.columns:
        # It is allowed in the sidecar for audit provenance but is never read
        # by this runner.  An explicit field makes accidental inclusion easier
        # to detect in review rather than silently filtering it away.
        pass
    return frame.sort_values(["fold", "__ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def run(args: argparse.Namespace) -> Path:
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"refusing to overwrite populated output directory {output}")
    output.mkdir(parents=True, exist_ok=True)
    frame = _load(Path(args.sidecar))
    arms = feature_arms(frame)
    coverage = frame.loc[:, sorted({field for fields in arms.values() for field in fields})].notna().mean()
    coverage.rename("coverage").rename_axis("feature").reset_index().to_parquet(
        output / "feature_coverage.parquet", index=False, compression="zstd"
    )

    prediction_parts: list[pd.DataFrame] = []
    audit_rows: list[dict] = []
    for fold_number, (fold, block) in enumerate(frame.groupby("fold", sort=True, observed=True), start=1):
        _validate_fold_partitions(block, str(fold))
        train_unpurged = block.loc[block["meta_partition"].eq("meta_train")].copy()
        calibration = block.loc[block["meta_partition"].eq("meta_calibration")].copy()
        test = block.loc[block["meta_partition"].eq("test")].copy()
        train = _purged_meta_train(train_unpurged, calibration, str(fold))
        result = test.loc[:, ["candidate_id", "__ts__", "month", "fold", "gross_bps", "net_bps", "base_expected_bps"]].copy()
        for target_name, label in _target_arms(train).items():
            for arm_number, (arm_name, fields) in enumerate(arms.items(), start=1):
                # Universal by construction: no path/support/outcome condition
                # can remove a candidate from this direct residual learner.
                model, fit_audit = _fit_ranker(
                    train, fields, label, seed=20260840 + fold_number * 100 + arm_number,
                )
                calibration_raw = _predict_ranker(model, calibration, fields)
                calibration_residual, _ = _finite_target(calibration)
                mapper = _fit_bps_map(calibration_raw, calibration_residual)
                test_raw = _predict_ranker(model, test, fields)
                score = f"{arm_name}__{target_name}"
                result[score] = (
                    test["base_expected_bps"].to_numpy(float) + mapper.predict(test_raw)
                ).astype(np.float32)
                audit_rows.append({
                    "fold": str(fold), "feature_arm": arm_name, "target_arm": target_name,
                    "score": score,
                    "meta_train_rows_before_horizon_purge": int(len(train_unpurged)),
                    "meta_train_rows": int(len(train)),
                    "meta_calibration_rows": int(len(calibration)), "test_rows": int(len(test)),
                    "feature_count": int(len(fields)), "universal_candidate_training": True,
                    "query_definition": "decision_timestamp_x_side_only_for_lambdarank",
                    "residual_target": "net_bps_minus_base_expected_bps",
                    **fit_audit,
                })
        prediction_parts.append(result)

    predictions = pd.concat(prediction_parts, ignore_index=True)
    score_columns = [column for column in predictions if "__" in column and column.startswith("R")]
    if not score_columns:
        raise RuntimeError("no residual prediction scores were materialised")
    predictions.to_parquet(output / "raw_oof_oos_predictions.parquet", index=False, compression="zstd")
    pd.DataFrame(audit_rows).to_parquet(output / "arm_fold_audit.parquet", index=False, compression="zstd")
    metrics = pd.DataFrame(
        [metric for score in score_columns for metric in _tail_metrics_with_fold_rows(predictions, score)]
    )
    metrics.to_parquet(output / "ablation_metrics.parquet", index=False, compression="zstd")
    _write_json(output / "run_manifest.json", {
        "schema": SCHEMA,
        "status": "complete",
        "side": SIDE,
        "sidecar": str(Path(args.sidecar)),
        "base_prediction_provenance": "strict OOF base score and calibration from structural sidecar",
        "residual_target": "direct net_bps - base_expected_bps",
        "feature_arms": {name: {"feature_count": len(fields), "fields": fields} for name, fields in arms.items()},
        "target_arms": {
            "hybrid_50_150": "hybrid ATR/bps grade; bps moderate=50 severe=150",
            "bps_50_150": "direct bps residual grade; moderate=50 severe=150",
        },
        "training_population": "all meta_train candidates; no support/path/outcome filtering",
        "forbidden_inference_inputs": "support_h12 and all raw leaf tokens",
        "evaluation": "pooled global top 1/3/5/10% after common-bps score; monthly and fold rows are diagnostic",
        "fold_count": int(frame["fold"].nunique()),
    })
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sidecar", type=Path,
        default=ROOT / "data_perp/artifacts/long_structural_tree_meta_sidecar_20260804_v4/tree_meta_candidate_sidecar.parquet",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
