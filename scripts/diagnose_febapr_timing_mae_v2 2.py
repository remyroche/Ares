#!/usr/bin/env python3
"""Immutable, action-layer diagnostics for the frozen timing/MAE OOF artifact.

This script deliberately has no execution-EV import or output.  It is a
post-hoc diagnostic and action-layer preparation step: the four independently
trained timing probabilities are projected onto a valid cumulative
distribution, and the three MAE heads are combined into an expected adverse
excursion estimate.  The completed source OOF artifact is never modified.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.path_auxiliary_role_training import _role_metrics
from scripts.run_febapr2025_historical_auxiliary_oof import (
    DEFAULT_CONTEXT,
    DEFAULT_LABEL_DIR,
    DEFAULT_STRICT_RESIDUAL,
    _identity_sha,
    load_inputs,
)

SOURCE = ROOT / "data_perp/artifacts/febapr2025_historical_auxiliary_oof_20260729_v2_timing_mae"
OUTPUT = ROOT / "data_perp/artifacts/febapr2025_historical_auxiliary_oof_20260729_v2_timing_mae_diagnostics"
HORIZONS = (2, 4, 8, 12)
IDENTITY = ("candidate_id", "side_name", "__ts__")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _record_final_output(temporary_path: Path, final_path: Path) -> dict[str, str]:
    """Bind a temp-published file to its stable post-rename path."""
    return {"path": str(final_path.resolve()), "sha256": _sha256(temporary_path)}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, default=str, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def pava_non_decreasing(row: np.ndarray) -> np.ndarray:
    """Unit-weight PAVA for one short CDF vector, independent of outcomes."""
    values: list[float] = []
    weights: list[float] = []
    counts: list[int] = []
    for raw in np.asarray(row, dtype=float):
        values.append(float(raw))
        weights.append(1.0)
        counts.append(1)
        while len(values) >= 2 and values[-2] > values[-1]:
            total_weight = weights[-2] + weights[-1]
            pooled = (values[-2] * weights[-2] + values[-1] * weights[-1]) / total_weight
            values[-2:] = [pooled]
            weights[-2:] = [total_weight]
            counts[-2:] = [counts[-2] + counts[-1]]
    return np.asarray([value for value, count in zip(values, counts) for _ in range(count)], dtype=float)


def _binary_metrics(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    return _role_metrics(target, prediction, mask, task_kind="binary", quantile_alpha=0.8)


def _regression_metrics(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    return _role_metrics(target, prediction, mask, task_kind="regression", quantile_alpha=0.8)


def _cdf_action_relevance(frame: pd.DataFrame) -> dict[str, Any]:
    """Outcome-stratified diagnostics; explicitly not a trade-policy result."""
    hit_2h = (
        frame["__meaningful_mfe_reached_12h__"].astype(bool)
        & frame["__time_to_first_meaningful_mfe_hours_12h__"].le(2)
    ).to_numpy()
    score = frame["timing_cdf_projected_2h"].to_numpy(float)
    n = len(frame)
    top_count = max(1, n // 10)
    top = np.argsort(score, kind="stable")[-top_count:]
    return {
        "top_10pct_fast_hit_rate": float(hit_2h[top].mean()),
        "all_fast_hit_rate": float(hit_2h.mean()),
        "top_10pct_support": int(top_count),
    }


def _mae_action_relevance(frame: pd.DataFrame, valid: np.ndarray) -> dict[str, Any]:
    score = frame["pred_expected_mae_atr_mae_p_hit"].to_numpy(float)
    observed = frame["__mae_before_meaningful_mfe_atr_12h__"].to_numpy(float)
    eligible = np.flatnonzero(valid & np.isfinite(score) & np.isfinite(observed))
    top_count = max(1, len(eligible) // 10)
    top = eligible[np.argsort(score[eligible], kind="stable")[-top_count:]]
    return {
        "top_10pct_predicted_adverse_mae_atr": float(observed[top].mean()),
        "all_observed_adverse_mae_atr": float(observed[eligible].mean()),
        "top_10pct_support": int(top_count),
    }


def _month_side_report(frame: pd.DataFrame, raw: np.ndarray, projected: np.ndarray) -> dict[str, Any]:
    report: dict[str, Any] = {"timing": {}, "mae": {}}
    reached = frame["__meaningful_mfe_reached_12h__"].astype(bool).to_numpy()
    elapsed = pd.to_numeric(frame["__time_to_first_meaningful_mfe_hours_12h__"], errors="coerce").to_numpy(float)
    for index, horizon in enumerate(HORIZONS):
        target = (reached & (elapsed <= horizon)).astype(float)
        mask = np.isfinite(target)
        report["timing"][str(horizon)] = {
            "raw": _binary_metrics(target, raw[:, index], mask),
            "projected": _binary_metrics(target, projected[:, index], mask),
        }
    valid = frame["__path_auxiliary_target_valid__"].eq(1).to_numpy()
    observed_mae = frame["__mae_before_meaningful_mfe_atr_12h__"].to_numpy(float)
    p_hit = frame["pred_mae_before_meaningful_mfe_atr__p_hit"].to_numpy(float)
    if_hit = frame["pred_mae_before_meaningful_mfe_atr__if_hit"].to_numpy(float)
    if_no_hit = frame["pred_mae_before_meaningful_mfe_atr__if_no_hit"].to_numpy(float)
    mae_p_hit_mixture = frame["pred_expected_mae_atr_mae_p_hit"].to_numpy(float)
    timing_p_hit_mixture = frame["pred_expected_mae_atr_timing_12h_p_hit"].to_numpy(float)
    report["mae"] = {
        "p_hit": _binary_metrics(reached.astype(float), p_hit, valid),
        "if_hit": _regression_metrics(observed_mae, if_hit, valid & reached),
        "if_no_hit": _regression_metrics(observed_mae, if_no_hit, valid & ~reached),
        "expected_mae_mixture_using_mae_p_hit": _regression_metrics(observed_mae, mae_p_hit_mixture, valid),
        "expected_mae_mixture_using_projected_timing_12h_p_hit": _regression_metrics(
            observed_mae, timing_p_hit_mixture, valid
        ),
        "action_relevance": _mae_action_relevance(frame, valid),
    }
    report["timing_action_relevance"] = _cdf_action_relevance(frame)
    raw_violations = (np.diff(raw, axis=1) < 0).any(axis=1)
    report["cdf_projection"] = {
        "raw_violation_rows": int(raw_violations.sum()),
        "raw_violation_fraction": float(raw_violations.mean()),
        "mean_abs_projection_delta": float(np.abs(projected - raw).mean()),
        "max_abs_projection_delta": float(np.abs(projected - raw).max()),
    }
    return report


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite immutable diagnostic output: {OUTPUT}")
    source_oof = SOURCE / "oof_predictions.parquet"
    source_manifest = SOURCE / "manifest.json"
    source_checkpoint = SOURCE / "checkpoint.json"
    if not all(path.is_file() for path in (source_oof, source_manifest, source_checkpoint)):
        raise FileNotFoundError("completed timing/MAE source artifact is incomplete")

    predictions = pd.read_parquet(source_oof)
    frame, _, _, strict_contract = load_inputs(DEFAULT_CONTEXT, DEFAULT_LABEL_DIR, DEFAULT_STRICT_RESIDUAL)
    labels = frame.loc[
        frame["__strict_residual_oof__"],
        [*IDENTITY, "__path_auxiliary_target_valid__",
         "__time_to_first_meaningful_mfe_hours_12h__", "__mae_before_meaningful_mfe_atr_12h__"],
    ]
    result = predictions.merge(labels, on=list(IDENTITY), how="inner", validate="one_to_one")
    if len(predictions) != 140_682 or len(result) != len(predictions):
        raise ValueError("timing/MAE OOF output does not exactly cover the strict 140,682-row ledger")
    if _identity_sha(predictions) != str(strict_contract["identity_sha256"]):
        raise ValueError("timing/MAE OOF identity differs from strict residual OOF ledger")
    if "__meaningful_mfe_reached_12h__" not in result:
        raise ValueError("timing/MAE OOF must retain the realized meaningful-MFE event for action evaluation")

    raw = np.column_stack([
        result[f"pred_time_to_first_meaningful_mfe__hit_by_{horizon}h"].to_numpy(float)
        for horizon in HORIZONS
    ])
    if not np.isfinite(raw).all() or ((raw < 0) | (raw > 1)).any():
        raise ValueError("raw timing probabilities must be finite values in [0, 1]")
    projected = np.vstack([pava_non_decreasing(row) for row in raw])
    if not np.all(np.diff(projected, axis=1) >= -1e-12) or ((projected < 0) | (projected > 1)).any():
        raise AssertionError("PAVA failed to produce a bounded non-decreasing CDF")
    for index, horizon in enumerate(HORIZONS):
        result[f"timing_cdf_projected_{horizon}h"] = projected[:, index]
    result["pred_expected_mae_atr_mae_p_hit"] = (
        result["pred_mae_before_meaningful_mfe_atr__p_hit"]
        * result["pred_mae_before_meaningful_mfe_atr__if_hit"]
        + (1.0 - result["pred_mae_before_meaningful_mfe_atr__p_hit"])
        * result["pred_mae_before_meaningful_mfe_atr__if_no_hit"]
    )
    result["pred_expected_mae_atr_timing_12h_p_hit"] = (
        result["timing_cdf_projected_12h"] * result["pred_mae_before_meaningful_mfe_atr__if_hit"]
        + (1.0 - result["timing_cdf_projected_12h"])
        * result["pred_mae_before_meaningful_mfe_atr__if_no_hit"]
    )
    result["timing_interval_mass_0_2h"] = projected[:, 0]
    result["timing_interval_mass_2_4h"] = projected[:, 1] - projected[:, 0]
    result["timing_interval_mass_4_8h"] = projected[:, 2] - projected[:, 1]
    result["timing_interval_mass_8_12h"] = projected[:, 3] - projected[:, 2]
    result["timing_interval_mass_no_hit_12h"] = 1.0 - projected[:, 3]
    mass_columns = [column for column in result if column.startswith("timing_interval_mass_")]
    masses = result.loc[:, mass_columns].to_numpy(float)
    if (masses < -1e-12).any() or not np.allclose(masses.sum(axis=1), 1.0, atol=1e-12, rtol=0):
        raise AssertionError("timing interval masses are incoherent")
    result["realized_labels_for_action_evaluation_only"] = True

    raw_violation_rows = (np.diff(raw, axis=1) < 0).any(axis=1)
    report: dict[str, Any] = {
        "schema": "febapr_timing_mae_action_layer_diagnostics_v1",
        "status": "ACTION_LAYER_ONLY_NOT_EXECUTION_EV",
        "identity": strict_contract,
        "source": {
            "timing_mae_oof": _record(source_oof),
            "timing_mae_manifest": _record(source_manifest),
            "timing_mae_checkpoint": _record(source_checkpoint),
            "strict_runner_source": _record(ROOT / "scripts/run_febapr2025_historical_auxiliary_oof.py"),
            "diagnostic_source": _record(Path(__file__)),
            "context_index": _record(DEFAULT_CONTEXT / "context_index.parquet"),
            "context_manifest": _record(DEFAULT_CONTEXT / "manifest.json"),
            "strict_residual": _record(DEFAULT_STRICT_RESIDUAL),
            "label_sources": {
                side: _record(DEFAULT_LABEL_DIR / f"train_global_{side}_3.parquet")
                for side in ("long", "short")
            },
        },
        "cdf_projection": {
            "method": "deterministic_unit_weight_pava_no_outcomes",
            "raw_violation_rows": int(raw_violation_rows.sum()),
            "raw_violation_fraction": float(raw_violation_rows.mean()),
            "mean_abs_projection_delta": float(np.abs(projected - raw).mean()),
            "max_abs_projection_delta": float(np.abs(projected - raw).max()),
            "interval_mass_columns": mass_columns,
            "interval_mass_sum_min": float(masses.sum(axis=1).min()),
            "interval_mass_sum_max": float(masses.sum(axis=1).max()),
        },
        "by_side_month": {},
        "notes": [
            "Raw timing heads are independently trained binary probabilities; PAVA only imposes CDF coherence.",
            "Action relevance is target-proximal stratification, not an execution-EV or portfolio-policy backtest.",
            "Timing, MAE, target price, and wait actions remain outside the execution-EV feature path.",
            "Realized label columns in action_layer_predictions.parquet are forbidden model inputs and retained only for action evaluation.",
        ],
    }
    for side in ("long", "short"):
        for month in ("2025-03", "2025-04"):
            mask = result["side_name"].eq(side) & result["__ts__"].dt.strftime("%Y-%m").eq(month)
            local = result.loc[mask].reset_index(drop=True)
            report["by_side_month"][f"{side}/{month}"] = _month_side_report(
                local, raw[mask.to_numpy()], projected[mask.to_numpy()]
            )

    temporary = Path(tempfile.mkdtemp(prefix=f".{OUTPUT.name}.", dir=OUTPUT.parent))
    try:
        result.to_parquet(temporary / "action_layer_predictions.parquet", index=False, compression="zstd")
        _write_json(temporary / "manifest.json", report)
        _write_json(
            temporary / "output_hashes.json",
            {
                "schema": "febapr_timing_mae_action_layer_output_hashes_v1",
                "outputs": {
                    "action_layer_predictions": _record_final_output(
                        temporary / "action_layer_predictions.parquet", OUTPUT / "action_layer_predictions.parquet"
                    ),
                    "manifest": _record_final_output(temporary / "manifest.json", OUTPUT / "manifest.json"),
                },
            },
        )
        os.replace(temporary, OUTPUT)
    except BaseException:
        for child in temporary.glob("*"):
            child.unlink()
        temporary.rmdir()
        raise


if __name__ == "__main__":
    main()
