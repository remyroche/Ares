#!/usr/bin/env python3
"""Finalize an extended meaningful-MFE CatBoost v2 checkpoint without fitting.

The model runner writes the OOF and exact-policy parquet checkpoints before it
writes its JSON summary.  This recovery path treats those parquet files as
immutable inputs: it verifies incumbent-overlap parity, checks the paired
checkpoint against the OOF checkpoint, and rebuilds all evaluation reports.
It contains no estimator construction or fitting code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.exact_policy_soft_binary_ablation import (
    economic_metrics,
)
from extreme_price_movements.meaningful_mfe_event_ablation import (
    first_21d_admission,
)

DEFAULT_INCUMBENT_DIR = (
    ROOT / "data_perp/artifacts/meaningful_mfe_catboost_v2_ablation_20260725_v1"
)
DEFAULT_EXTENSION_DIR = (
    ROOT
    / "data_perp/artifacts/"
    "meaningful_mfe_catboost_v2_ablation_july20_20260726_v1"
)
IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
TARGET_COLUMNS = (
    "tb_hard_label",
    "tb_soft_label",
    "meaningful_mfe_reached",
    "risk_class",
    "order_ambiguous",
)
RETURN_COLUMN = "execution_net_ev_12h"
SCHEMA = "meaningful_mfe_catboost_v2_extension_finalization_v1"


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "bytes": int(stat.st_size),
    }


def _require_columns(
    frame: pd.DataFrame,
    required: Sequence[str],
    *,
    source: str,
) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing required columns: {', '.join(missing)}")


def _timestamps(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    out = frame.copy()
    out["__ts__"] = pd.to_datetime(out["__ts__"], utc=True, errors="raise")
    if out["candidate_id"].duplicated().any():
        duplicates = int(out["candidate_id"].duplicated(keep=False).sum())
        raise ValueError(f"{source} contains {duplicates} duplicate candidate_id rows")
    return out


def _strict_column_audit(
    expected: pd.Series,
    observed: pd.Series,
) -> dict[str, Any]:
    left = expected.to_numpy()
    right = observed.to_numpy()
    if pd.api.types.is_numeric_dtype(expected.dtype) and pd.api.types.is_numeric_dtype(
        observed.dtype
    ):
        left_number = pd.to_numeric(expected, errors="coerce").to_numpy(np.float64)
        right_number = pd.to_numeric(observed, errors="coerce").to_numpy(np.float64)
        same_missing = np.array_equal(
            np.isnan(left_number),
            np.isnan(right_number),
        )
        finite = np.isfinite(left_number) & np.isfinite(right_number)
        deltas = np.abs(left_number[finite] - right_number[finite])
        max_delta = float(deltas.max()) if len(deltas) else 0.0
        exact = bool(same_missing and max_delta == 0.0)
        mismatch = (left_number != right_number) & ~(
            np.isnan(left_number) & np.isnan(right_number)
        )
        mismatch_rows = int(mismatch.sum())
        return {
            "exact": exact,
            "mismatch_rows": mismatch_rows,
            "max_abs_delta": max_delta,
            "missing_pattern_exact": bool(same_missing),
        }
    left_object = expected.astype("string").fillna("<NA>").to_numpy()
    right_object = observed.astype("string").fillna("<NA>").to_numpy()
    mismatch = left_object != right_object
    return {
        "exact": bool(not mismatch.any()),
        "mismatch_rows": int(mismatch.sum()),
        "max_abs_delta": None,
        "missing_pattern_exact": bool(
            np.array_equal(expected.isna().to_numpy(), observed.isna().to_numpy())
        ),
    }


def audit_incumbent_overlap(
    incumbent: pd.DataFrame,
    extension: pd.DataFrame,
    *,
    score_columns: Sequence[str],
) -> dict[str, Any]:
    """Require every incumbent row and score to be bit-identical in extension."""

    required = [*IDENTITY, *TARGET_COLUMNS, *score_columns]
    _require_columns(incumbent, required, source="incumbent OOF")
    _require_columns(extension, required, source="extension OOF")
    incumbent = _timestamps(incumbent.loc[:, required], source="incumbent OOF")
    extension = _timestamps(extension.loc[:, required], source="extension OOF")
    incumbent_ids = pd.Index(incumbent["candidate_id"])
    extension_ids = pd.Index(extension["candidate_id"])
    missing_ids = incumbent_ids.difference(extension_ids)
    if len(missing_ids):
        raise ValueError(
            f"extension OOF is missing {len(missing_ids)} incumbent candidate rows"
        )
    old = incumbent.set_index("candidate_id", drop=False)
    new = extension.set_index("candidate_id", drop=False).loc[incumbent_ids]
    column_audits = {
        column: _strict_column_audit(old[column], new[column])
        for column in [*IDENTITY[:-1], *TARGET_COLUMNS, *score_columns]
    }
    failures = [column for column, audit in column_audits.items() if not audit["exact"]]
    if failures:
        raise ValueError(
            "extension OOF incumbent-overlap parity failed for: "
            + ", ".join(failures)
        )
    extension_only = extension.loc[
        ~extension["candidate_id"].isin(incumbent_ids), "__ts__"
    ]
    return {
        "status": "passed",
        "identity_key": "candidate_id",
        "incumbent_rows": int(len(incumbent)),
        "extension_rows": int(len(extension)),
        "overlap_rows": int(len(incumbent)),
        "extension_only_rows": int(len(extension_only)),
        "incumbent_timestamp_max": incumbent["__ts__"].max(),
        "extension_only_timestamp_min": (
            extension_only.min() if len(extension_only) else None
        ),
        "extension_only_timestamp_max": (
            extension_only.max() if len(extension_only) else None
        ),
        "prediction_columns": list(score_columns),
        "global_prediction_max_abs_delta": float(
            max(column_audits[column]["max_abs_delta"] for column in score_columns)
        ),
        "columns": column_audits,
    }


def audit_paired_consistency(
    oof: pd.DataFrame,
    paired: pd.DataFrame,
    *,
    score_columns: Sequence[str],
) -> dict[str, Any]:
    """Require paired labels and predictions to equal their OOF source rows."""

    required = [*IDENTITY, *TARGET_COLUMNS, *score_columns]
    _require_columns(oof, required, source="extension OOF")
    _require_columns(
        paired,
        [*required, RETURN_COLUMN],
        source="extension exact-policy paired",
    )
    oof = _timestamps(oof.loc[:, required], source="extension OOF")
    paired = _timestamps(
        paired.loc[:, [*required, RETURN_COLUMN]],
        source="extension exact-policy paired",
    )
    paired_ids = pd.Index(paired["candidate_id"])
    missing_ids = paired_ids.difference(pd.Index(oof["candidate_id"]))
    if len(missing_ids):
        raise ValueError(
            f"paired checkpoint has {len(missing_ids)} rows absent from extension OOF"
        )
    source = oof.set_index("candidate_id", drop=False).loc[paired_ids]
    observed = paired.set_index("candidate_id", drop=False)
    audits = {
        column: _strict_column_audit(source[column], observed[column])
        for column in [*IDENTITY[:-1], *TARGET_COLUMNS, *score_columns]
    }
    failures = [column for column, audit in audits.items() if not audit["exact"]]
    if failures:
        raise ValueError(
            "extension exact-policy/OOF consistency failed for: "
            + ", ".join(failures)
        )
    return {
        "status": "passed",
        "paired_rows": int(len(paired)),
        "matched_oof_rows": int(len(paired)),
        "prediction_columns": list(score_columns),
        "global_prediction_max_abs_delta": float(
            max(audits[column]["max_abs_delta"] for column in score_columns)
        ),
        "columns": audits,
    }


def classification_metrics(
    hard: np.ndarray,
    soft: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, Any]:
    """Reproduce the original runner's classification metric contract."""

    prediction = np.clip(np.asarray(prediction, dtype=np.float64), 1e-6, 1 - 1e-6)
    hard = np.asarray(hard, dtype=np.float64)
    soft = np.asarray(soft, dtype=np.float64)
    order = np.argsort(-prediction, kind="stable")
    n10 = max(1, int(np.ceil(len(order) * 0.10)))
    selected = order[:n10]
    bins = pd.qcut(prediction, 10, labels=False, duplicates="drop")
    calibration = (
        pd.DataFrame({"bin": bins, "prediction": prediction, "hard": hard})
        .groupby("bin", sort=True)
        .agg(
            rows=("hard", "size"),
            prediction=("prediction", "mean"),
            observed=("hard", "mean"),
        )
        .reset_index()
    )
    return {
        "rows": int(len(hard)),
        "prevalence": float(np.mean(hard)),
        "roc_auc": float(roc_auc_score(hard, prediction)),
        "average_precision": float(average_precision_score(hard, prediction)),
        "brier_hard": float(brier_score_loss(hard, prediction)),
        "brier_soft": float(np.mean((prediction - soft) ** 2)),
        "log_loss_hard": float(log_loss(hard, prediction, labels=[0.0, 1.0])),
        "spearman_soft": float(spearmanr(prediction, soft).statistic),
        "top10_rows": int(n10),
        "top10_precision": float(np.mean(hard[selected])),
        "top10_recall": float(np.sum(hard[selected]) / max(np.sum(hard), 1.0)),
        "ece": float(
            np.average(
                np.abs(calibration["prediction"] - calibration["observed"]),
                weights=calibration["rows"],
            )
        ),
        "calibration_bins": calibration.to_dict(orient="records"),
    }


def recompute_reports(
    oof: pd.DataFrame,
    paired: pd.DataFrame,
    *,
    score_columns: Sequence[str],
) -> dict[str, Any]:
    """Recompute classification, exact-policy, month, and admission reports."""

    required = [*IDENTITY, *TARGET_COLUMNS, *score_columns]
    _require_columns(oof, required, source="extension OOF")
    _require_columns(
        paired,
        [*required, RETURN_COLUMN],
        source="extension exact-policy paired",
    )
    oof = _timestamps(oof, source="extension OOF")
    paired = _timestamps(paired, source="extension exact-policy paired")
    clean_hard = pd.to_numeric(oof["tb_hard_label"], errors="coerce").to_numpy(
        np.float64
    )
    clean_soft = pd.to_numeric(oof["tb_soft_label"], errors="coerce").to_numpy(
        np.float64
    )
    literal = pd.to_numeric(
        oof["meaningful_mfe_reached"], errors="coerce"
    ).to_numpy(np.float64)
    reports: dict[str, Any] = {}
    for name in score_columns:
        values = pd.to_numeric(oof[name], errors="coerce").to_numpy(np.float64)
        finite = (
            np.isfinite(values)
            & np.isfinite(clean_hard)
            & np.isfinite(clean_soft)
            & np.isfinite(literal)
        )
        report: dict[str, Any] = {
            "oof_rows": int(finite.sum()),
            "clean_event": classification_metrics(
                clean_hard[finite],
                clean_soft[finite],
                values[finite],
            ),
            "literal_event": classification_metrics(
                literal[finite],
                literal[finite],
                values[finite],
            ),
            "clean_event_by_side_month": [],
        }
        local_oof = oof.loc[finite].copy()
        local_oof["_month"] = local_oof["__ts__"].dt.strftime("%Y-%m")
        for (side, month), group in local_oof.groupby(
            ["side_name", "_month"],
            sort=True,
        ):
            report["clean_event_by_side_month"].append(
                {
                    "side": side,
                    "month": month,
                    **classification_metrics(
                        group["tb_hard_label"].to_numpy(np.float64),
                        group["tb_soft_label"].to_numpy(np.float64),
                        group[name].to_numpy(np.float64),
                    ),
                }
            )

        paired_score = pd.to_numeric(paired[name], errors="coerce").to_numpy(
            np.float64
        )
        paired_finite = np.isfinite(paired_score)
        report["exact_policy"] = economic_metrics(
            paired.loc[paired_finite].reset_index(drop=True),
            paired_score[paired_finite],
        )
        admission = first_21d_admission(
            paired["__ts__"],
            paired_score,
            pd.to_numeric(paired[RETURN_COLUMN], errors="coerce").to_numpy(
                np.float64
            ),
        )
        evaluation = np.asarray(admission["evaluation_mask"], dtype=bool)
        admitted = np.asarray(admission["admitted_mask"], dtype=bool)
        report["post_21d_admission"] = {
            "contract": {
                key: value
                for key, value in admission.items()
                if key
                not in {
                    "evaluation_mask",
                    "admitted_mask",
                    "calibrated_expected_net_return",
                }
            },
            "raw_after_fit_window": economic_metrics(
                paired.loc[evaluation].reset_index(drop=True),
                paired_score[evaluation],
            ),
            "admitted": economic_metrics(
                paired,
                paired_score,
                admitted=admitted,
            ),
        }
        local_paired = paired.loc[paired_finite].copy()
        local_paired["_month"] = local_paired["__ts__"].dt.strftime("%Y-%m")
        report["exact_policy_by_side_month"] = [
            {
                "side": side,
                "month": month,
                **economic_metrics(
                    group.reset_index(drop=True),
                    group[name].to_numpy(np.float64),
                ),
            }
            for (side, month), group in local_paired.groupby(
                ["side_name", "_month"],
                sort=True,
            )
        ]
        favorable = finite & (clean_hard > 0.5)
        quality = np.clip((clean_soft[favorable] - 0.75) / 0.25, 0.0, 1.0)
        report["conditional_quality_ic_on_favorable"] = float(
            spearmanr(values[favorable], quality).statistic
        )
        reports[name] = report
    return reports


def _range(values: pd.Series) -> dict[str, Any]:
    timestamps = pd.to_datetime(values, utc=True, errors="coerce").dropna()
    return {
        "rows": int(len(timestamps)),
        "min": timestamps.min() if len(timestamps) else None,
        "max": timestamps.max() if len(timestamps) else None,
    }


def data_ceiling(
    oof: pd.DataFrame,
    paired: pd.DataFrame,
    *,
    score_columns: Sequence[str],
) -> dict[str, Any]:
    oof = _timestamps(oof, source="extension OOF")
    paired = _timestamps(paired, source="extension exact-policy paired")
    result: dict[str, Any] = {
        "interpretation": (
            "Observed checkpoint ceilings only; no claim beyond serialized rows."
        ),
        "oof_candidates": _range(oof["__ts__"]),
        "exact_policy_candidates": _range(paired["__ts__"]),
        "oof_finite_by_score": {},
        "exact_policy_finite_by_score": {},
    }
    for name in score_columns:
        oof_finite = pd.to_numeric(oof[name], errors="coerce").notna()
        paired_finite = pd.to_numeric(paired[name], errors="coerce").notna()
        result["oof_finite_by_score"][name] = _range(oof.loc[oof_finite, "__ts__"])
        result["exact_policy_finite_by_score"][name] = _range(
            paired.loc[paired_finite, "__ts__"]
        )
    for column in (
        "execution_decision_utc",
        "execution_label_end_utc",
        "execution_label_available_at",
    ):
        if column in paired:
            result[column] = _range(paired[column])
    return result


def _load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("reports"), dict) or not payload["reports"]:
        raise ValueError("incumbent summary has no report contract")
    return payload


def finalize(
    *,
    incumbent_summary_path: Path,
    incumbent_oof_path: Path,
    incumbent_paired_path: Path,
    extension_oof_path: Path,
    extension_paired_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Audit checkpoint inputs, rebuild reports, and atomically write summary."""

    sources = {
        "incumbent_summary": incumbent_summary_path,
        "incumbent_oof_checkpoint": incumbent_oof_path,
        "incumbent_exact_policy_checkpoint": incumbent_paired_path,
        "extension_oof_checkpoint": extension_oof_path,
        "extension_exact_policy_checkpoint": extension_paired_path,
    }
    missing = {name: str(path) for name, path in sources.items() if not path.is_file()}
    if missing:
        raise FileNotFoundError(
            "missing required finalization sources: "
            + json.dumps(missing, sort_keys=True)
        )
    if output_path.resolve() in {path.resolve() for path in sources.values()}:
        raise ValueError("output path must not overwrite a checkpoint source")

    incumbent_summary = _load_summary(incumbent_summary_path)
    score_columns = tuple(incumbent_summary["reports"].keys())
    read_columns = [*IDENTITY, *TARGET_COLUMNS, *score_columns]
    incumbent_oof = pd.read_parquet(incumbent_oof_path, columns=read_columns)
    extension_oof = pd.read_parquet(extension_oof_path, columns=read_columns)
    paired_columns = [*read_columns, RETURN_COLUMN]
    available_paired_columns = pq.read_schema(extension_paired_path).names
    paired_columns.extend(
        column
        for column in (
            "execution_decision_utc",
            "execution_label_end_utc",
            "execution_label_available_at",
        )
        if column in available_paired_columns
    )
    extension_paired = pd.read_parquet(
        extension_paired_path,
        columns=paired_columns,
    )

    overlap = audit_incumbent_overlap(
        incumbent_oof,
        extension_oof,
        score_columns=score_columns,
    )
    consistency = audit_paired_consistency(
        extension_oof,
        extension_paired,
        score_columns=score_columns,
    )
    reports = recompute_reports(
        extension_oof,
        extension_paired,
        score_columns=score_columns,
    )
    ceilings = data_ceiling(
        extension_oof,
        extension_paired,
        score_columns=score_columns,
    )
    source_records = {name: _source(path) for name, path in sources.items()}
    summary = {
        "schema": SCHEMA,
        "status": "finalized_from_completed_checkpoints_no_refit",
        "finalized_at_utc": datetime.now(timezone.utc),
        "recovery_contract": {
            "model_fit_performed": False,
            "checkpoint_inputs_are_read_only": True,
            "incumbent_report_schema": incumbent_summary.get("schema"),
            "score_columns_from_incumbent_report_contract": list(score_columns),
            "metric_definitions": (
                "same clean/literal classification, exact-policy economics, "
                "side-month breakdown, and causal first-21d admission contract "
                "as the original runner"
            ),
            "failure_context": (
                "resume after checkpoints completed; original final hash step "
                "received an incorrect missing incumbent path"
            ),
        },
        "chronology": {
            **incumbent_summary.get("chronology", {}),
            "extension": (
                "checkpoint-only July extension; overlap with incumbent OOF is "
                "required to be exactly prediction-identical"
            ),
        },
        "incumbent_contract": {
            "schema": incumbent_summary.get("schema"),
            "status": incumbent_summary.get("status"),
            "rows": incumbent_summary.get("rows"),
            "sources": incumbent_summary.get("sources"),
        },
        "sources": source_records,
        "overlap_parity": overlap,
        "checkpoint_consistency": consistency,
        "data_ceiling": ceilings,
        "rows": {
            "valid_labels": int(len(extension_oof)),
            "outer_oof": int(
                pd.to_numeric(
                    extension_oof["catboost_hard_ensemble"],
                    errors="coerce",
                ).notna().sum()
            ),
            "exact_policy_paired": int(len(extension_paired)),
            "order_ambiguous": int(
                pd.to_numeric(
                    extension_oof["order_ambiguous"],
                    errors="coerce",
                ).fillna(0).astype(bool).sum()
            ),
        },
        "reports": reports,
        "outputs": {
            "predictions": str(extension_oof_path.resolve()),
            "exact_policy_paired": str(extension_paired_path.resolve()),
            "summary": str(output_path.resolve()),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--incumbent-summary-path",
        type=Path,
        default=DEFAULT_INCUMBENT_DIR / "summary.json",
    )
    parser.add_argument(
        "--incumbent-oof-path",
        type=Path,
        default=DEFAULT_INCUMBENT_DIR / "oof_predictions.parquet",
    )
    parser.add_argument(
        "--incumbent-paired-path",
        type=Path,
        default=DEFAULT_INCUMBENT_DIR / "exact_policy_paired.parquet",
    )
    parser.add_argument(
        "--extension-oof-path",
        type=Path,
        default=DEFAULT_EXTENSION_DIR / "oof_predictions.parquet",
    )
    parser.add_argument(
        "--extension-paired-path",
        type=Path,
        default=DEFAULT_EXTENSION_DIR / "exact_policy_paired.parquet",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_EXTENSION_DIR / "summary.json",
    )
    args = parser.parse_args(argv)
    summary = finalize(**vars(args))
    print(
        json.dumps(
            {
                "status": summary["status"],
                "rows": summary["rows"],
                "overlap_parity": {
                    key: summary["overlap_parity"][key]
                    for key in (
                        "status",
                        "overlap_rows",
                        "extension_only_rows",
                        "global_prediction_max_abs_delta",
                    )
                },
                "output": summary["outputs"]["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
