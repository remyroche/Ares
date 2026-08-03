#!/usr/bin/env python3
"""Frozen head-by-head diagnostics for the cross-era tail-payoff challenger.

This is deliberately an audit, not a training entry point.  It reads the
already-frozen challenger predictions, restores exact-1m competing-risk
targets for the May--July 2026 rows, and reports head quality without
changing predictions, calibrators, models, or the source artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
IDENTITY = ("candidate_id", "__ts__", "__symbol__", "side_name")
CLASS_NAMES = ("positive", "adverse_negative", "timeout_negative", "other_negative")
SCHEMA = "cross_era_tail_payoff_challenger_head_audit_v1"

PROBABILITY_SPECS = tuple(
    [
        {"head": f"raw_{name}", "column": f"raw_p_{name}", "class_code": code}
        for code, name in enumerate(CLASS_NAMES)
    ]
    + [
        {"head": f"calibrated_{name}", "column": f"p_{name}", "class_code": code}
        for code, name in enumerate(CLASS_NAMES)
    ]
)
QUANTILE_SPECS = (
    {"head": "q25_positive", "column": "q25_positive_bps", "class_code": 0, "alpha": 0.25, "target": "positive_payoff_bps", "higher_is_favorable": True},
    {"head": "q50_positive", "column": "q50_positive_bps", "class_code": 0, "alpha": 0.50, "target": "positive_payoff_bps", "higher_is_favorable": True},
    {"head": "q50_adverse", "column": "q50_adverse_bps", "class_code": 1, "alpha": 0.50, "target": "loss_magnitude_bps", "higher_is_favorable": False},
    {"head": "q85_adverse", "column": "q85_adverse_bps", "class_code": 1, "alpha": 0.85, "target": "loss_magnitude_bps", "higher_is_favorable": False},
    {"head": "q75_timeout", "column": "q75_timeout_bps", "class_code": 2, "alpha": 0.75, "target": "loss_magnitude_bps", "higher_is_favorable": False},
    {"head": "q75_other", "column": "q75_other_bps", "class_code": 3, "alpha": 0.75, "target": "loss_magnitude_bps", "higher_is_favorable": False},
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def _normalise_identity(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["__ts__"] = pd.to_datetime(result["__ts__"], utc=True, errors="raise")
    result["__symbol__"] = result["__symbol__"].astype(str).str.replace("/", "_", regex=False)
    return result


def derive_exact_event_code(frame: pd.DataFrame) -> np.ndarray:
    """Economic 4-class target using exact-1m first-touch labels.

    Positive net payoff takes precedence, exactly as the frozen challenger
    target.  Negative rows split into exact adverse, exact timeout and other.
    """
    positive = pd.to_numeric(frame["positive_net"], errors="raise").astype(bool).to_numpy()
    event = frame["__soft_tb_first_event__"].astype(str).to_numpy()
    result = np.full(len(frame), 3, dtype=np.int8)
    negative = ~positive
    result[negative & (event == "adverse_first_or_conflict")] = 1
    result[negative & (event == "timeout")] = 2
    result[positive] = 0
    return result


def attach_exact_1m_event_targets(
    predictions: pd.DataFrame, labels: pd.DataFrame
) -> pd.DataFrame:
    """Strictly attach exact-1m labels; missing or duplicate identities fail."""
    left = _normalise_identity(predictions)
    right = _normalise_identity(labels)
    right = right.loc[:, [*IDENTITY, "__soft_tb_first_event__"]]
    if left.duplicated(list(IDENTITY)).any() or right.duplicated(list(IDENTITY)).any():
        raise ValueError("candidate identity must be unique before exact-1m join")
    joined = left.merge(right, on=list(IDENTITY), how="left", validate="one_to_one")
    if joined["__soft_tb_first_event__"].isna().any():
        raise ValueError("exact-1m label coverage is incomplete for challenger predictions")
    joined["event_code"] = derive_exact_event_code(joined)
    joined["event_target_origin"] = "harmonized_exact_1m"
    return joined


def _finite_binary_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(target) & np.isfinite(probability)
    y = target[finite].astype(int)
    p = np.clip(probability[finite], 0.0, 1.0)
    result: dict[str, float] = {
        "rows": int(len(y)), "positive_rows": int(y.sum()),
        "prevalence": float(y.mean()) if len(y) else float("nan"),
        "mean_prediction": float(p.mean()) if len(p) else float("nan"),
        "auc": float("nan"), "pr_auc": float("nan"), "brier": float("nan"),
        "brier_skill_vs_prevalence": float("nan"), "ece_10": float("nan"),
        "calibration_mean_gap": float("nan"),
    }
    if not len(y):
        return result
    result["brier"] = float(brier_score_loss(y, p))
    baseline = float(y.mean() * (1.0 - y.mean()))
    if baseline > 0:
        result["brier_skill_vs_prevalence"] = float(1.0 - result["brier"] / baseline)
    result["calibration_mean_gap"] = float(p.mean() - y.mean())
    bins = np.minimum((p * 10).astype(int), 9)
    result["ece_10"] = float(sum(
        (bins == bin_index).mean() * abs(float(p[bins == bin_index].mean()) - float(y[bins == bin_index].mean()))
        for bin_index in np.unique(bins)
    ))
    if np.unique(y).size == 2:
        result["auc"] = float(roc_auc_score(y, p))
        result["pr_auc"] = float(average_precision_score(y, p))
    return result


def probability_calibration_bins(target: np.ndarray, probability: np.ndarray) -> pd.DataFrame:
    finite = np.isfinite(target) & np.isfinite(probability)
    y = target[finite].astype(int)
    p = np.clip(probability[finite], 0.0, 1.0)
    if not len(y):
        return pd.DataFrame(columns=["calibration_bin", "rows", "mean_prediction", "observed_rate", "absolute_gap"])
    bins = np.minimum((p * 10).astype(int), 9)
    rows = []
    for index in range(10):
        mask = bins == index
        if mask.any():
            mean_prediction, observed = float(p[mask].mean()), float(y[mask].mean())
            rows.append({"calibration_bin": index, "rows": int(mask.sum()), "mean_prediction": mean_prediction, "observed_rate": observed, "absolute_gap": abs(mean_prediction - observed)})
    return pd.DataFrame(rows)


def pinball_loss(target: np.ndarray, prediction: np.ndarray, alpha: float) -> float:
    error = target - prediction
    return float(np.mean(np.maximum(alpha * error, (alpha - 1.0) * error)))


def quantile_metrics(target: np.ndarray, prediction: np.ndarray, signed_net_bps: np.ndarray, alpha: float) -> dict[str, float]:
    finite = np.isfinite(target) & np.isfinite(prediction) & np.isfinite(signed_net_bps)
    y, p, net = target[finite], prediction[finite], signed_net_bps[finite]
    result: dict[str, float] = {
        "rows": int(len(y)), "pinball_loss": float("nan"), "baseline_pinball_loss": float("nan"),
        "baseline_pinball_skill": float("nan"), "conditional_spearman": float("nan"),
        "dispersion_ratio": float("nan"), "bias_bps": float("nan"),
        "bottom_decile_realized_payoff_bps": float("nan"), "top_decile_realized_payoff_bps": float("nan"),
        "top_minus_bottom_realized_payoff_bps": float("nan"),
    }
    if not len(y):
        return result
    result["pinball_loss"] = pinball_loss(y, p, alpha)
    baseline_prediction = float(np.quantile(y, alpha))
    result["baseline_pinball_loss"] = pinball_loss(y, np.full(len(y), baseline_prediction), alpha)
    if result["baseline_pinball_loss"] > 0:
        result["baseline_pinball_skill"] = float(1.0 - result["pinball_loss"] / result["baseline_pinball_loss"])
    if len(y) >= 3 and np.unique(y).size > 1 and np.unique(p).size > 1:
        result["conditional_spearman"] = float(spearmanr(p, y).statistic)
    std_y = float(np.std(y))
    if std_y > 0:
        result["dispersion_ratio"] = float(np.std(p) / std_y)
    result["bias_bps"] = float(np.mean(p - y))
    take = max(1, int(math.ceil(0.10 * len(y))))
    order = np.argsort(p, kind="stable")
    bottom, top = net[order[:take]], net[order[-take:]]
    result["bottom_decile_realized_payoff_bps"] = float(bottom.mean())
    result["top_decile_realized_payoff_bps"] = float(top.mean())
    result["top_minus_bottom_realized_payoff_bps"] = float(top.mean() - bottom.mean())
    return result


def _diagnose_probability(row: Mapping[str, Any]) -> str:
    auc = row["auc"]
    if not np.isfinite(auc):
        return "insufficient_class_support"
    if auc <= 0.47:
        return "inverted"
    if auc >= 0.53 and row["brier_skill_vs_prevalence"] > 0:
        return "learnable"
    return "weak_or_miscalibrated"


def _diagnose_quantile(row: Mapping[str, Any]) -> str:
    rho = row["conditional_spearman"]
    if not np.isfinite(rho):
        return "insufficient_conditional_support"
    if rho <= -0.03:
        return "inverted"
    if rho >= 0.05 and row["baseline_pinball_skill"] > 0:
        return "learnable"
    return "weak_or_unhelpful"


def evaluate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    probability_rows: list[dict[str, Any]] = []
    quantile_rows: list[dict[str, Any]] = []
    calibration_parts: list[pd.DataFrame] = []
    group_columns = ["evaluation_set", "era", "month", "side_name"]
    for group_key, local in frame.groupby(group_columns, sort=True, dropna=False):
        group = dict(zip(group_columns, group_key, strict=True))
        event = pd.to_numeric(local["event_code"], errors="raise").to_numpy(int)
        net_bps = pd.to_numeric(local["execution_net_ev_12h"], errors="raise").to_numpy(float) * 1e4
        for spec in PROBABILITY_SPECS:
            probability = pd.to_numeric(local[spec["column"]], errors="coerce").to_numpy(float)
            target = (event == spec["class_code"]).astype(int)
            row = {**group, **spec, **_finite_binary_metrics(target, probability)}
            row["diagnosis"] = _diagnose_probability(row)
            probability_rows.append(row)
            bins = probability_calibration_bins(target, probability)
            if not bins.empty:
                bins.insert(0, "head", spec["head"])
                for key, value in reversed(tuple(group.items())):
                    bins.insert(0, key, value)
                calibration_parts.append(bins)
        for spec in QUANTILE_SPECS:
            conditional = event == spec["class_code"]
            target = net_bps[conditional] if spec["class_code"] == 0 else -net_bps[conditional]
            prediction = pd.to_numeric(local.loc[conditional, spec["column"]], errors="coerce").to_numpy(float)
            signed = net_bps[conditional]
            row = {**group, **spec, **quantile_metrics(target, prediction, signed, spec["alpha"])}
            row["favorable_top_minus_bottom_realized_payoff_bps"] = row["top_minus_bottom_realized_payoff_bps"] if spec["higher_is_favorable"] else -row["top_minus_bottom_realized_payoff_bps"]
            row["diagnosis"] = _diagnose_quantile(row)
            quantile_rows.append(row)
    probability = pd.DataFrame(probability_rows)
    quantiles = pd.DataFrame(quantile_rows)
    calibration = pd.concat(calibration_parts, ignore_index=True) if calibration_parts else pd.DataFrame()
    diagnosis = pd.concat([
        probability.assign(head_type="probability"),
        quantiles.assign(head_type="quantile"),
    ], ignore_index=True, sort=False)
    diagnosis = diagnosis.loc[:, ["evaluation_set", "era", "month", "side_name", "head", "head_type", "diagnosis"]]
    return probability, quantiles, calibration, diagnosis


def load_evaluation_frame(
    challenger_dir: Path, historical_exact_labels_path: Path, current_exact_labels_path: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    historical = pd.read_parquet(challenger_dir / "historical_oof_winner.parquet")
    current_predictions = pd.read_parquet(challenger_dir / "current_predictions_before_outcomes.parquet")
    current_economics = pd.read_parquet(challenger_dir / "current_scored_exact.parquet")
    historical_exact_labels = pd.read_parquet(historical_exact_labels_path)
    current_exact_labels = pd.read_parquet(current_exact_labels_path)
    historical = _normalise_identity(historical)
    current_predictions = _normalise_identity(current_predictions)
    current_economics = _normalise_identity(current_economics)
    historical_exact_labels = _normalise_identity(historical_exact_labels)
    current_exact_labels = _normalise_identity(current_exact_labels)
    current_outcomes = current_economics.loc[:, [*IDENTITY, "execution_net_ev_12h", "positive_net"]]
    current = current_predictions.merge(current_outcomes, on=list(IDENTITY), how="inner", validate="one_to_one")
    if len(current) != len(current_predictions):
        raise ValueError("frozen current predictions lost identities when exact economics were attached")
    historical["event_target_origin"] = "frozen_historical_exact_1m"
    # Every 2026 historical OOF row must use the already-materialised 1m target,
    # replacing the hourly-grid event code without touching any prediction.
    recent_historical = historical[historical["era"].astype(str).str.startswith("2026")].copy()
    old_historical = historical[~historical["era"].astype(str).str.startswith("2026")].copy()
    recent_historical["positive_net"] = (
        pd.to_numeric(recent_historical["execution_net_ev_12h"], errors="raise") > 0.0
    ).astype(np.int8)
    recent_historical = recent_historical.drop(columns=["event_code", "event_target_origin"])
    recent_historical = attach_exact_1m_event_targets(recent_historical, historical_exact_labels)
    current = attach_exact_1m_event_targets(current, current_exact_labels)
    old_historical["event_code"] = pd.to_numeric(old_historical["event_code"], errors="raise").astype(np.int8)
    old_historical["evaluation_set"] = "historical_oof"
    recent_historical["evaluation_set"] = "historical_oof"
    current["evaluation_set"] = "current_july_exact_1m"
    combined = pd.concat([old_historical, recent_historical, current], ignore_index=True, sort=False)
    combined["month"] = pd.to_datetime(combined["__ts__"], utc=True).dt.strftime("%Y-%m")
    combined["era"] = combined["era"].fillna("2026_current_jul20_23")
    expected = set(column for spec in PROBABILITY_SPECS for column in [spec["column"]]) | set(spec["column"] for spec in QUANTILE_SPECS)
    missing = expected.difference(combined.columns)
    if missing:
        raise ValueError(f"missing frozen prediction heads: {sorted(missing)}")
    lineage = {
        "historical_oof_rows": int(len(historical)), "current_frozen_prediction_rows": int(len(current_predictions)),
        "current_rows_with_exact_1m_labels": int(len(current)), "recent_historical_rows_with_exact_1m_labels": int(len(recent_historical)),
        "old_historical_rows_with_frozen_exact_1m_event_code": int(len(old_historical)),
    }
    return combined, lineage


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    frame, lineage = load_evaluation_frame(
        args.challenger_dir, args.historical_exact_labels, args.current_exact_labels
    )
    probability, quantiles, calibration, diagnosis = evaluate(frame)
    args.output_dir.mkdir(parents=True)
    outputs: dict[str, dict[str, Any]] = {}
    for name, data in {
        "probability_metrics": probability,
        "quantile_metrics": quantiles,
        "calibration_bins": calibration,
        "head_diagnosis": diagnosis,
    }.items():
        path = args.output_dir / f"{name}.csv"
        data.to_csv(path, index=False)
        outputs[name] = {"path": str(path), "rows": int(len(data)), "sha256": sha256(path)}
    summary = {
        "schema": SCHEMA, "status": "completed_frozen_prediction_audit", "source_challenger": {"path": str(args.challenger_dir), "manifest_sha256": sha256(args.challenger_dir / "manifest.json")},
        "exact_1m_labels": {
            "historical_may_jul19": {"path": str(args.historical_exact_labels), "sha256": sha256(args.historical_exact_labels)},
            "current_jul20_23": {"path": str(args.current_exact_labels), "sha256": sha256(args.current_exact_labels)},
        },
        "lineage": lineage,
        "contract": {
            "predictions": "frozen artifact only; no refit, rescore, calibration or model mutation",
            "recent_event_targets": "harmonized exact-1m, executable spread-adjusted h12_u1p5atr competing-risk labels",
            "quantile_targets": "conditional realised exact-policy net payoff for positive; loss magnitude for each negative event class",
            "calibration": "10 equal-width probability bins; ECE and mean calibration gap",
        },
        "diagnosis_counts": diagnosis.groupby(["head_type", "diagnosis"], dropna=False).size().to_dict(),
        "outputs": outputs,
    }
    report_path = args.output_dir / "report.json"
    _write_json(report_path, summary)
    manifest_path = args.output_dir / "manifest.json"
    _write_json(manifest_path, {"schema": SCHEMA, "status": summary["status"], "report": {"path": str(report_path), "sha256": sha256(report_path)}, "outputs": outputs})
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenger-dir", type=Path, default=Path("data_perp/artifacts/cross_era_tail_payoff_challenger_20260730_v2"))
    parser.add_argument("--historical-exact-labels", type=Path, default=Path("data_perp/artifacts/harmonized_mayjul19_exact1m_clean_first_labels_20260730_v1/exact_clean_first_labels.parquet"))
    parser.add_argument("--current-exact-labels", type=Path, default=Path("data_perp/artifacts/july_exact1m_clean_first_labels_20260730_v1/exact_clean_first_labels.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_perp/artifacts/cross_era_tail_payoff_challenger_head_audit_20260730_v1"))
    return parser.parse_args()


if __name__ == "__main__":
    report = run(parse_args())
    print(json.dumps(_safe({"status": report["status"], "outputs": report["outputs"]}), sort_keys=True))
