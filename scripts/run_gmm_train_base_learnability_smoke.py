#!/usr/bin/env python3
"""Train-base-style OOF learnability smoke for active GMM label candidates.

This is a handoff guard, not a final production training claim. It loads the
active rows from the GMM cluster-policy readiness matrix, runs the existing
month-forward feature-store model smoke on those label/weight arms, and writes
an explicit pass/fail artifact that the readiness checker can consume before
train_meta or policy-exit work starts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_label_feature_store_model_smoke import FIXED_ARTIFACT_LABEL_ARMS, run_smoke


DEFAULT_REPORT_DIR = Path("data_perp/reports/gmm_cluster_policy_smoke_20260702_wide_sidebalanced")
DEFAULT_SMOKE_SUBDIR = "gmm_train_base_learnability_smoke"
DEFAULT_OUTPUT_NAME = "gmm_train_base_learnability_check.json"
DEFAULT_TARGET_VARIANT_ARMS = (
    "OPTIMIZED_ECONOMIC_PATH_SAFE_TARGET",
    "OPTIMIZED_ECONOMIC_BAD_MAE_CONTRAST_TARGET",
    "OPTIMIZED_ECONOMIC_CLEAN_RANK_TARGET",
    "OPTIMIZED_ECONOMIC_TIMEOUT_SAFE_TARGET",
    "OPTIMIZED_ECONOMIC_STRICT_PATH_FIRST_TARGET",
    "OPTIMIZED_ECONOMIC_CLEAN_UTILITY_RANK_TARGET",
)
GMM_CONTEXT_TOKENS = (
    "gmm",
    "cluster",
    "state",
    "posterior",
    "entropy",
    "reconstruct",
    "spectral",
    "ae_",
    "_ae",
)
DEFAULT_THRESHOLDS = {
    "min_months": 3,
    "min_positive_month_share": 2.0 / 3.0,
    "min_mean_u": 0.0,
    "min_worst_month_mean_u": 0.0,
    "min_score_ic_u": 0.0,
    "min_top_bottom_decile_spread_u": 0.0,
    "max_bad_mae_1r_rate": 0.50,
    "max_timeout_rate": 0.12,
    "max_wide_barrier_25bps_rate": 0.08,
    "max_selected_side_share": 0.70,
    "min_mean_selected_rows": 25,
    "min_selected_rows": 10,
    "min_retained_features": 25,
    "min_stage_a_oracle_recall": 0.50,
    "min_final_oracle_recall": 0.02,
    "max_hard_risk_cap_no_trade_rate": 0.95,
    "candidate_min_positive_month_share": 1.0,
    "candidate_min_worst_month_mean_u": 0.0,
    "candidate_max_bad_mae_1r_rate": 0.65,
    "candidate_max_timeout_rate": 0.15,
    "candidate_min_stage_a_oracle_recall": 0.50,
    "candidate_min_final_oracle_recall": 0.02,
    "candidate_min_clean_positive_rate": 0.01,
    "candidate_min_dirty_positive_rate": 0.01,
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if np.isfinite(out) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _read_report_manifest(report_dir: Path) -> dict[str, Any]:
    path = report_dir / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_int_csv(value: str | None, fallback: list[int]) -> list[int]:
    if value is None or not str(value).strip():
        return list(fallback)
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def _as_bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False).astype(bool)
    return values.astype(str).str.lower().isin({"1", "true", "yes", "y"})


def _active_readiness_rows(report_dir: Path) -> pd.DataFrame:
    readiness = _read_csv(report_dir / "gmm_train_meta_readiness.csv")
    if readiness.empty:
        raise ValueError("gmm_train_meta_readiness.csv has no candidate rows")
    if "is_final_promotion_ready" in readiness.columns:
        final_ready = _as_bool_series(readiness["is_final_promotion_ready"])
        readiness = readiness.loc[~final_ready].copy()
    if readiness.empty:
        raise ValueError("no non-final GMM readiness candidates found")
    return readiness.reset_index(drop=True)


def _expanded_target_candidate_rows(
    candidates: pd.DataFrame,
    *,
    target_label_arms: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, candidate in candidates.iterrows():
        base = candidate.to_dict()
        label_arms = [str(base.get("label_arm", ""))]
        label_arms.extend(arm for arm in target_label_arms if arm and arm not in label_arms)
        for label_arm in label_arms:
            row = dict(base)
            row["label_arm"] = label_arm
            row["source_label_arm"] = base.get("label_arm")
            row["is_target_variant"] = str(label_arm) != str(base.get("label_arm"))
            rows.append(row)
    return pd.DataFrame(rows)


def _candidate_id(row: pd.Series, index: int) -> str:
    parts = [
        str(row.get("label_arm", "unknown_label")),
        str(row.get("weight_arm", "unknown_weight")),
        str(row.get("cluster_policy", "unknown_policy")),
        f"top{float(row.get('top_frac', 0.0)):.4f}",
    ]
    return f"{index}:" + "::".join(parts)


def _candidate_output_dir(report_dir: Path, row: pd.Series, index: int, multiple: bool) -> Path:
    base = report_dir / DEFAULT_SMOKE_SUBDIR
    if not multiple:
        return base
    raw = _candidate_id(row, index)
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw)
    return base / safe[:160]


def _metric(row: pd.Series | dict[str, Any], column: str) -> float:
    if column not in row:
        return float("nan")
    value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else float("nan")


def _passes_min(value: float, threshold: float) -> bool:
    return math.isfinite(value) and value >= float(threshold)


def _passes_max(value: float, threshold: float) -> bool:
    return math.isfinite(value) and value <= float(threshold)


def _failed_checks(checks: list[tuple[str, float, str, float, bool]]) -> list[dict[str, Any]]:
    return [
        {
            "metric": metric,
            "value": value,
            "operator": operator,
            "threshold": threshold,
        }
        for metric, value, operator, threshold, passed in checks
        if not passed
    ]


def _check_records(checks: list[tuple[str, float, str, float, bool]]) -> list[dict[str, Any]]:
    return [
        {
            "metric": metric,
            "value": value,
            "operator": operator,
            "threshold": threshold,
            "passed": bool(passed),
        }
        for metric, value, operator, threshold, passed in checks
    ]


def _matching_aggregate_rows(aggregate: pd.DataFrame, candidate: pd.Series) -> pd.DataFrame:
    label_arm = str(candidate.get("label_arm", ""))
    weight_arm = str(candidate.get("weight_arm", ""))
    top_frac = float(candidate.get("top_frac", 0.0))
    subset = aggregate.copy()
    if "label_arm" in subset.columns:
        subset = subset[subset["label_arm"].astype(str).eq(label_arm)]
    if "weight_arm" in subset.columns:
        subset = subset[subset["weight_arm"].astype(str).eq(weight_arm)]
    if "top_frac" in subset.columns:
        numeric_top_frac = pd.to_numeric(subset["top_frac"], errors="coerce")
        subset = subset[np.isclose(numeric_top_frac, top_frac)]
    if subset.empty:
        raise ValueError(
            "No aggregate row matched active candidate "
            f"label={label_arm} weight={weight_arm} top_frac={top_frac}"
        )
    sort_cols = [
        col
        for col in (
            "mean_u",
            "worst_month_mean_u",
            "bad_mae_1r_rate",
            "max_selected_side_share",
        )
        if col in subset.columns
    ]
    if sort_cols:
        ascending = [False if col in {"mean_u", "worst_month_mean_u"} else True for col in sort_cols]
        subset = subset.sort_values(sort_cols, ascending=ascending)
    return subset.reset_index(drop=True)


def _feature_contract(smoke_manifest: dict[str, Any]) -> dict[str, Any]:
    features = [str(v) for v in smoke_manifest.get("features", [])]
    lowered = [feature.lower() for feature in features]
    context_features = [
        feature
        for feature, lower in zip(features, lowered)
        if any(token in lower for token in GMM_CONTEXT_TOKENS)
    ]
    feature_store = smoke_manifest.get("feature_store") or {}
    return {
        "feature_count": int(smoke_manifest.get("feature_count", len(features)) or 0),
        "retained_feature_store_features": int(feature_store.get("retained_features", 0) or 0),
        "requested_feature_store_features": int(feature_store.get("requested_features", 0) or 0),
        "mean_feature_finite_frac": float(feature_store.get("mean_feature_finite_frac", 0.0) or 0.0),
        "min_feature_finite_frac": float(feature_store.get("min_feature_finite_frac", 0.0) or 0.0),
        "gmm_context_feature_count": int(len(context_features)),
        "gmm_context_features": context_features,
    }


def _gate_candidate(
    *,
    candidate: pd.Series,
    aggregate_row: dict[str, Any],
    smoke_manifest: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    feature_contract = _feature_contract(smoke_manifest)
    months = _metric(aggregate_row, "months")
    positive_months = _metric(aggregate_row, "positive_months")
    min_positive_months = math.ceil(months * thresholds["min_positive_month_share"]) if math.isfinite(months) else math.inf
    checks = [
        ("months", months, ">=", thresholds["min_months"], _passes_min(months, thresholds["min_months"])),
        (
            "positive_months",
            positive_months,
            ">=",
            min_positive_months,
            _passes_min(positive_months, min_positive_months),
        ),
        ("mean_u", _metric(aggregate_row, "mean_u"), ">", thresholds["min_mean_u"], _metric(aggregate_row, "mean_u") > thresholds["min_mean_u"]),
        (
            "worst_month_mean_u",
            _metric(aggregate_row, "worst_month_mean_u"),
            ">=",
            thresholds["min_worst_month_mean_u"],
            _passes_min(_metric(aggregate_row, "worst_month_mean_u"), thresholds["min_worst_month_mean_u"]),
        ),
        (
            "score_ic_u",
            _metric(aggregate_row, "score_ic_u"),
            ">",
            thresholds["min_score_ic_u"],
            _metric(aggregate_row, "score_ic_u") > thresholds["min_score_ic_u"],
        ),
        (
            "top_bottom_decile_spread_u",
            _metric(aggregate_row, "top_bottom_decile_spread_u"),
            ">",
            thresholds["min_top_bottom_decile_spread_u"],
            _metric(aggregate_row, "top_bottom_decile_spread_u")
            > thresholds["min_top_bottom_decile_spread_u"],
        ),
        (
            "bad_mae_1r_rate",
            _metric(aggregate_row, "bad_mae_1r_rate"),
            "<=",
            thresholds["max_bad_mae_1r_rate"],
            _passes_max(_metric(aggregate_row, "bad_mae_1r_rate"), thresholds["max_bad_mae_1r_rate"]),
        ),
        (
            "timeout_rate",
            _metric(aggregate_row, "timeout_rate"),
            "<=",
            thresholds["max_timeout_rate"],
            _passes_max(_metric(aggregate_row, "timeout_rate"), thresholds["max_timeout_rate"]),
        ),
        (
            "wide_barrier_25bps_rate",
            _metric(aggregate_row, "wide_barrier_25bps_rate"),
            "<=",
            thresholds["max_wide_barrier_25bps_rate"],
            _passes_max(
                _metric(aggregate_row, "wide_barrier_25bps_rate"),
                thresholds["max_wide_barrier_25bps_rate"],
            ),
        ),
        (
            "max_selected_side_share",
            _metric(aggregate_row, "max_selected_side_share"),
            "<=",
            thresholds["max_selected_side_share"],
            _passes_max(
                _metric(aggregate_row, "max_selected_side_share"),
                thresholds["max_selected_side_share"],
            ),
        ),
        (
            "mean_selected_rows",
            _metric(aggregate_row, "mean_selected_rows"),
            ">=",
            thresholds["min_mean_selected_rows"],
            _passes_min(_metric(aggregate_row, "mean_selected_rows"), thresholds["min_mean_selected_rows"]),
        ),
        (
            "min_selected_rows",
            _metric(aggregate_row, "min_selected_rows"),
            ">=",
            thresholds["min_selected_rows"],
            _passes_min(_metric(aggregate_row, "min_selected_rows"), thresholds["min_selected_rows"]),
        ),
        (
            "retained_feature_store_features",
            float(feature_contract["retained_feature_store_features"]),
            ">=",
            thresholds["min_retained_features"],
            _passes_min(
                float(feature_contract["retained_feature_store_features"]),
                thresholds["min_retained_features"],
            ),
        ),
    ]
    stage_a_oracle_recall = _metric(aggregate_row, "stageA_candidate_oracle_recall")
    if math.isfinite(stage_a_oracle_recall):
        checks.append(
            (
                "stageA_candidate_oracle_recall",
                stage_a_oracle_recall,
                ">=",
                thresholds["min_stage_a_oracle_recall"],
                _passes_min(stage_a_oracle_recall, thresholds["min_stage_a_oracle_recall"]),
            )
        )
    final_oracle_recall = _metric(aggregate_row, "final_oracle_recall")
    if math.isfinite(final_oracle_recall):
        checks.append(
            (
                "final_oracle_recall",
                final_oracle_recall,
                ">=",
                thresholds["min_final_oracle_recall"],
                _passes_min(final_oracle_recall, thresholds["min_final_oracle_recall"]),
            )
        )
    hard_no_trade_rate = _metric(aggregate_row, "hard_risk_cap_no_trade_rate")
    if math.isfinite(hard_no_trade_rate):
        checks.append(
            (
                "hard_risk_cap_no_trade_rate",
                hard_no_trade_rate,
                "<=",
                thresholds["max_hard_risk_cap_no_trade_rate"],
                _passes_max(hard_no_trade_rate, thresholds["max_hard_risk_cap_no_trade_rate"]),
            )
        )
    final_failed = _failed_checks(checks)

    candidate_min_positive_months = (
        math.ceil(months * thresholds["candidate_min_positive_month_share"])
        if math.isfinite(months)
        else math.inf
    )
    candidate_checks = [
        ("months", months, ">=", thresholds["min_months"], _passes_min(months, thresholds["min_months"])),
        (
            "positive_months",
            positive_months,
            ">=",
            candidate_min_positive_months,
            _passes_min(positive_months, candidate_min_positive_months),
        ),
        ("mean_u", _metric(aggregate_row, "mean_u"), ">", thresholds["min_mean_u"], _metric(aggregate_row, "mean_u") > thresholds["min_mean_u"]),
        (
            "worst_month_mean_u",
            _metric(aggregate_row, "worst_month_mean_u"),
            ">=",
            thresholds["candidate_min_worst_month_mean_u"],
            _passes_min(
                _metric(aggregate_row, "worst_month_mean_u"),
                thresholds["candidate_min_worst_month_mean_u"],
            ),
        ),
        (
            "score_ic_u",
            _metric(aggregate_row, "score_ic_u"),
            ">",
            thresholds["min_score_ic_u"],
            _metric(aggregate_row, "score_ic_u") > thresholds["min_score_ic_u"],
        ),
        (
            "top_bottom_decile_spread_u",
            _metric(aggregate_row, "top_bottom_decile_spread_u"),
            ">",
            thresholds["min_top_bottom_decile_spread_u"],
            _metric(aggregate_row, "top_bottom_decile_spread_u")
            > thresholds["min_top_bottom_decile_spread_u"],
        ),
        (
            "bad_mae_1r_rate",
            _metric(aggregate_row, "bad_mae_1r_rate"),
            "<=",
            thresholds["candidate_max_bad_mae_1r_rate"],
            _passes_max(
                _metric(aggregate_row, "bad_mae_1r_rate"),
                thresholds["candidate_max_bad_mae_1r_rate"],
            ),
        ),
        (
            "timeout_rate",
            _metric(aggregate_row, "timeout_rate"),
            "<=",
            thresholds["candidate_max_timeout_rate"],
            _passes_max(
                _metric(aggregate_row, "timeout_rate"),
                thresholds["candidate_max_timeout_rate"],
            ),
        ),
        (
            "wide_barrier_25bps_rate",
            _metric(aggregate_row, "wide_barrier_25bps_rate"),
            "<=",
            thresholds["max_wide_barrier_25bps_rate"],
            _passes_max(
                _metric(aggregate_row, "wide_barrier_25bps_rate"),
                thresholds["max_wide_barrier_25bps_rate"],
            ),
        ),
        (
            "max_selected_side_share",
            _metric(aggregate_row, "max_selected_side_share"),
            "<=",
            thresholds["max_selected_side_share"],
            _passes_max(
                _metric(aggregate_row, "max_selected_side_share"),
                thresholds["max_selected_side_share"],
            ),
        ),
        (
            "mean_selected_rows",
            _metric(aggregate_row, "mean_selected_rows"),
            ">=",
            thresholds["min_mean_selected_rows"],
            _passes_min(
                _metric(aggregate_row, "mean_selected_rows"),
                thresholds["min_mean_selected_rows"],
            ),
        ),
        (
            "min_selected_rows",
            _metric(aggregate_row, "min_selected_rows"),
            ">=",
            thresholds["min_selected_rows"],
            _passes_min(_metric(aggregate_row, "min_selected_rows"), thresholds["min_selected_rows"]),
        ),
        (
            "retained_feature_store_features",
            float(feature_contract["retained_feature_store_features"]),
            ">=",
            thresholds["min_retained_features"],
            _passes_min(
                float(feature_contract["retained_feature_store_features"]),
                thresholds["min_retained_features"],
            ),
        ),
    ]
    if math.isfinite(stage_a_oracle_recall):
        candidate_checks.append(
            (
                "stageA_candidate_oracle_recall",
                stage_a_oracle_recall,
                ">=",
                thresholds["candidate_min_stage_a_oracle_recall"],
                _passes_min(
                    stage_a_oracle_recall,
                    thresholds["candidate_min_stage_a_oracle_recall"],
                ),
            )
        )
    if math.isfinite(final_oracle_recall):
        candidate_checks.append(
            (
                "final_oracle_recall",
                final_oracle_recall,
                ">=",
                thresholds["candidate_min_final_oracle_recall"],
                _passes_min(final_oracle_recall, thresholds["candidate_min_final_oracle_recall"]),
            )
        )
    if math.isfinite(hard_no_trade_rate):
        candidate_checks.append(
            (
                "hard_risk_cap_no_trade_rate",
                hard_no_trade_rate,
                "<=",
                thresholds["max_hard_risk_cap_no_trade_rate"],
                _passes_max(hard_no_trade_rate, thresholds["max_hard_risk_cap_no_trade_rate"]),
            )
        )
    clean_positive_rate = _metric(aggregate_row, "clean_positive_rate")
    if math.isfinite(clean_positive_rate):
        candidate_checks.append(
            (
                "clean_positive_rate",
                clean_positive_rate,
                ">=",
                thresholds["candidate_min_clean_positive_rate"],
                _passes_min(
                    clean_positive_rate,
                    thresholds["candidate_min_clean_positive_rate"],
                ),
            )
        )
    dirty_positive_rate = _metric(aggregate_row, "dirty_positive_rate")
    if math.isfinite(dirty_positive_rate):
        candidate_checks.append(
            (
                "dirty_positive_rate",
                dirty_positive_rate,
                ">=",
                thresholds["candidate_min_dirty_positive_rate"],
                _passes_min(
                    dirty_positive_rate,
                    thresholds["candidate_min_dirty_positive_rate"],
                ),
            )
        )
    candidate_failed = _failed_checks(candidate_checks)
    return {
        "candidate_id": _candidate_id(candidate, 0),
        "label_arm": candidate.get("label_arm"),
        "weight_arm": candidate.get("weight_arm"),
        "cluster_policy": candidate.get("cluster_policy"),
        "top_frac": float(candidate.get("top_frac", 0.0)),
        "evaluation_utility_source": candidate.get("evaluation_utility_source"),
        "selector_variant": aggregate_row.get("selector_variant"),
        "arm": aggregate_row.get("arm"),
        "status": "pass" if not final_failed else "fail",
        "candidate_readiness_status": "pass" if not candidate_failed else "fail",
        "final_policy_status": "pass" if not final_failed else "fail",
        "failed_checks": final_failed,
        "checks": _check_records(checks),
        "candidate_readiness_failed_checks": candidate_failed,
        "candidate_readiness_checks": _check_records(candidate_checks),
        "metrics": {
            key: aggregate_row.get(key)
            for key in (
                "months",
                "active_selected_months",
                "no_trade_months",
                "no_trade_month_share",
                "positive_months",
                "mean_u",
                "worst_month_mean_u",
                "hit_u",
                "q10_u",
                "delta_mean_u_vs_period",
                "score_ic_u",
                "score_ic_label",
                "decile_spearman_u",
                "top_bottom_decile_spread_u",
                "ts_rank_hr10_u",
                "ts_rank_hr20_u",
                "ts_rank_hr30_u",
                "ts_rank_ndcg30_u",
                "ts_rank_top30_bad_mae_1r_rate",
                "ts_rank_top30_timeout_rate",
                "bad_mae_1r_rate",
                "clean_positive_rate",
                "dirty_positive_rate",
                "wide_barrier_25bps_rate",
                "timeout_rate",
                "selected_long_share",
                "selected_short_share",
                "max_selected_side_share",
                "worst_month_selected_side_share",
                "selected_pred_bad_mae_mean",
                "selected_pred_timeout_mean",
                "score_ic_bad_mae",
                "mean_selected_rows",
                "min_selected_rows",
                "top_symbol_share",
                "hard_risk_cap_no_trade_rate",
                "stageA_candidate_oracle_recall",
                "stageA_candidate_long_oracle_recall",
                "stageA_candidate_short_oracle_recall",
                "stageA_candidate_bad_mae_1r_rate",
                "stageA_candidate_timeout_rate",
                "final_oracle_recall",
                "final_long_oracle_recall",
                "final_short_oracle_recall",
                "final_bad_mae_1r_rate",
                "final_timeout_rate",
            )
            if key in aggregate_row
        },
        "feature_contract": feature_contract,
    }


def build_learnability_check(
    *,
    report_dir: Path,
    smoke_manifest: dict[str, Any],
    aggregate: pd.DataFrame,
    candidates: pd.DataFrame,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    candidate_checks: list[dict[str, Any]] = []
    for idx, candidate in candidates.reset_index(drop=True).iterrows():
        for variant_idx, aggregate_row in _matching_aggregate_rows(aggregate, candidate).iterrows():
            gate = _gate_candidate(
                candidate=candidate,
                aggregate_row=aggregate_row.to_dict(),
                smoke_manifest=smoke_manifest,
                thresholds=thresholds,
            )
            gate["candidate_id"] = f"{_candidate_id(candidate, int(idx))}::variant{int(variant_idx)}"
            candidate_checks.append(gate)

    passed = [row for row in candidate_checks if row.get("final_policy_status") == "pass"]
    candidate_ready = [
        row for row in candidate_checks if row.get("candidate_readiness_status") == "pass"
    ]
    failed = [row for row in candidate_checks if row.get("final_policy_status") != "pass"]
    best_passing = (
        sorted(
            passed,
            key=lambda row: (
                float((row.get("metrics") or {}).get("mean_u") or -1.0e9),
                float((row.get("metrics") or {}).get("worst_month_mean_u") or -1.0e9),
                -float((row.get("metrics") or {}).get("bad_mae_1r_rate") or 1.0e9),
            ),
            reverse=True,
        )[0]
        if passed
        else None
    )
    best_candidate_readiness = (
        sorted(
            candidate_ready,
            key=lambda row: (
                float((row.get("metrics") or {}).get("mean_u") or -1.0e9),
                float((row.get("metrics") or {}).get("worst_month_mean_u") or -1.0e9),
                float((row.get("metrics") or {}).get("final_oracle_recall") or -1.0e9),
                -float((row.get("metrics") or {}).get("bad_mae_1r_rate") or 1.0e9),
            ),
            reverse=True,
        )[0]
        if candidate_ready
        else None
    )
    status = (
        "pass"
        if passed
        else "candidate_for_train_meta_path_filter_smoke"
        if candidate_ready
        else "fail"
    )
    meta_filter_handoff = {
        "status": "ready" if candidate_ready else "blocked",
        "source_gate": "train_base_candidate_readiness",
        "best_candidate_stream": best_candidate_readiness,
        "base_final_policy_status": "pass" if passed else "fail",
        "target_contract": {
            "positive_label": "u_policy_net > 0 and mae_norm < 1.0 and is_timeout == 0",
            "negative_label": (
                "u_policy_net <= 0 or mae_norm >= 1.0 or is_timeout == 1; "
                "dirty-positive rows are explicit negatives/low relevance"
            ),
        },
        "required_oof_inputs": [
            "base_oof_score",
            "base_timestamp_local_rank_pct",
            "base_side_local_rank_pct",
            "base_candidate_pool_rank",
            "bad_mae_probability",
            "timeout_probability",
            "clean_path_probability",
            "dirty_positive_probability",
            "gmm_cluster_state",
            "gmm_posterior_features",
            "ae_reconstruction_error",
            "side",
            "month_or_regime_features",
            "market_state_features",
        ],
        "promotion_bar": {
            "bad_mae_1r_rate_max": thresholds["max_bad_mae_1r_rate"],
            "timeout_rate_max": thresholds["max_timeout_rate"],
            "final_oracle_recall_min": thresholds["min_final_oracle_recall"],
            "mean_u_min": thresholds["min_mean_u"],
            "worst_month_mean_u_min": thresholds["min_worst_month_mean_u"],
            "max_selected_side_share": thresholds["max_selected_side_share"],
            "min_selected_rows": thresholds["min_selected_rows"],
        },
    }
    return {
        "status": status,
        "report_dir": str(report_dir),
        "candidate_count": int(len(candidate_checks)),
        "passed_variant_count": int(len(passed)),
        "candidate_ready_variant_count": int(len(candidate_ready)),
        "failed_variant_count": int(len(failed)),
        "best_passing_candidate": best_passing,
        "best_candidate_readiness": best_candidate_readiness,
        "candidate_checks": candidate_checks,
        "gate_1a_train_base_candidate_readiness": {
            "status": "pass" if candidate_ready else "fail",
            "passed_variant_count": int(len(candidate_ready)),
            "best_candidate": best_candidate_readiness,
            "purpose": "Provide a broad, side-aware, non-degenerate candidate stream for train_meta.",
        },
        "gate_1b_train_base_final_policy_readiness": {
            "status": "pass" if passed else "fail",
            "passed_variant_count": int(len(passed)),
            "best_candidate": best_passing,
            "purpose": "Use train_base directly as a final selector under strict path-risk standards.",
        },
        "meta_filter_handoff": meta_filter_handoff,
        "thresholds": thresholds,
        "smoke_manifest": smoke_manifest,
        "outputs": smoke_manifest.get("outputs", {}),
        "sequential_pipeline_plan": [
            {
                "stage": "train_base_candidate_readiness",
                "goal": "A pre-existing-feature OOF selector must preserve broad, side-aware opportunities for train_meta.",
                "advance_when": "At least one active GMM candidate stream is positive in every month, has adequate recall, enough rows, both sides, and non-extreme path risk.",
            },
            {
                "stage": "train_base_final_policy_readiness",
                "goal": "Verify whether train_base can be used directly as a final selector.",
                "advance_when": "The base selector itself clears strict bad-MAE, timeout, monthly utility, recall, and side-exposure gates.",
            },
            {
                "stage": "train_meta_oos_path_filter",
                "goal": "Feed passed base candidate streams plus GMM/AE/risk context into train_meta and verify OOS path filtering.",
                "advance_when": "train_meta improves bad-MAE and timeout without killing utility, oracle recall, row count, or side balance.",
            },
            {
                "stage": "simple_policy_optimiser_exit_policy",
                "goal": "Fit exits/thresholds on allowed periods only, with GMM/AE state as context rather than hard gates.",
                "advance_when": "Policy-OOS and frozen replay pass monthly, weekly-tail, exposure, and cost guardrails.",
            },
            {
                "stage": "frozen_threshold_replay",
                "goal": "Replay final thresholds without retuning on the reported period.",
                "advance_when": "Frozen replay remains positive with acceptable no-trade/exposure and tail risk.",
            },
            {
                "stage": "leakage_and_feature_parity_audit",
                "goal": "Confirm no future labels, post-selection leakage, or train/inference feature mismatch.",
                "advance_when": "Artifacts prove feature parity and split purity for the promoted configuration.",
            },
        ],
        "passed_next_check": (
            "train_base_final_policy_readiness"
            if passed
            else "train_base_candidate_readiness"
            if candidate_ready
            else None
        ),
        "failed_next_check": (
            None
            if passed
            else "train_base_final_policy_readiness"
            if candidate_ready
            else "train_base_candidate_readiness"
        ),
    }


def run_learnability_smoke(
    *,
    report_dir: Path,
    smoke_output_dir: Path | None,
    output: Path | None,
    seeds: list[int] | None,
    train_lookback_months: int | None,
    max_feature_store_features: int | None,
    thresholds: dict[str, float],
    target_label_arms: list[str],
) -> dict[str, Any]:
    report_manifest = _read_report_manifest(report_dir)
    candidates = _active_readiness_rows(report_dir)
    if seeds is None:
        raw_seeds = report_manifest.get("seeds")
        seeds = [int(v) for v in raw_seeds] if isinstance(raw_seeds, list) and raw_seeds else [42]
    if train_lookback_months is None:
        raw_lookback = report_manifest.get("train_lookback_months")
        train_lookback_months = int(raw_lookback) if raw_lookback is not None else None

    if len(candidates) != 1:
        raise ValueError(
            "Current learnability smoke expects exactly one active GMM candidate; "
            f"found {len(candidates)}"
        )

    candidate = candidates.iloc[0]
    candidate_output = smoke_output_dir or _candidate_output_dir(report_dir, candidate, 0, multiple=False)
    label_arm = str(candidate.get("label_arm", "")).strip()
    weight_arm = str(candidate.get("weight_arm", "")).strip()
    if not label_arm or not weight_arm:
        raise ValueError("Active GMM candidate must include label_arm and weight_arm")
    expanded_candidates = _expanded_target_candidate_rows(
        candidates,
        target_label_arms=target_label_arms,
    )
    label_arms = expanded_candidates["label_arm"].dropna().astype(str).drop_duplicates().tolist()

    smoke_manifest = run_smoke(
        labels_path=Path(str(candidate.get("labels_path"))),
        output_dir=candidate_output,
        feature_dir=Path(str(candidate.get("feature_dir"))),
        feature_list_csv=Path(str(candidate.get("feature_list_csv"))),
        evaluation_utility_column=str(candidate.get("evaluation_utility_source") or "").strip() or None,
        max_feature_store_features=max_feature_store_features,
        label_arms=label_arms,
        weight_arms=[weight_arm],
        seeds=seeds,
        model_feature_selector="all",
        model_feature_tail_frac=0.01,
        top_fracs=[float(candidate.get("top_frac"))],
        train_lookback_months=train_lookback_months,
        include_risk_selector_variants=True,
        side_cap_max_share=float(thresholds["max_selected_side_share"]),
    )
    aggregate_path = Path(str(smoke_manifest["outputs"]["aggregate"]))
    aggregate = pd.read_csv(aggregate_path)
    check = build_learnability_check(
        report_dir=report_dir,
        smoke_manifest=smoke_manifest,
        aggregate=aggregate,
        candidates=expanded_candidates,
        thresholds=thresholds,
    )
    output_path = output or (report_dir / DEFAULT_OUTPUT_NAME)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_safe(check), indent=2, sort_keys=True), encoding="utf-8")
    return check


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--smoke-output-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--train-lookback-months", type=int, default=None)
    parser.add_argument("--max-feature-store-features", type=int, default=None)
    parser.add_argument(
        "--target-label-arms",
        type=str,
        default=",".join(DEFAULT_TARGET_VARIANT_ARMS),
        help=(
            "Comma-separated target variants to evaluate alongside the active label arm. "
            "Use an empty string to evaluate only the active readiness label."
        ),
    )
    parser.add_argument("--min-months", type=int, default=int(DEFAULT_THRESHOLDS["min_months"]))
    parser.add_argument(
        "--min-positive-month-share",
        type=float,
        default=float(DEFAULT_THRESHOLDS["min_positive_month_share"]),
    )
    parser.add_argument("--min-mean-u", type=float, default=float(DEFAULT_THRESHOLDS["min_mean_u"]))
    parser.add_argument(
        "--min-worst-month-mean-u",
        type=float,
        default=float(DEFAULT_THRESHOLDS["min_worst_month_mean_u"]),
    )
    parser.add_argument(
        "--min-score-ic-u",
        type=float,
        default=float(DEFAULT_THRESHOLDS["min_score_ic_u"]),
    )
    parser.add_argument(
        "--min-top-bottom-decile-spread-u",
        type=float,
        default=float(DEFAULT_THRESHOLDS["min_top_bottom_decile_spread_u"]),
    )
    parser.add_argument(
        "--max-bad-mae-1r-rate",
        type=float,
        default=float(DEFAULT_THRESHOLDS["max_bad_mae_1r_rate"]),
    )
    parser.add_argument(
        "--max-timeout-rate",
        type=float,
        default=float(DEFAULT_THRESHOLDS["max_timeout_rate"]),
    )
    parser.add_argument(
        "--max-wide-barrier-25bps-rate",
        type=float,
        default=float(DEFAULT_THRESHOLDS["max_wide_barrier_25bps_rate"]),
    )
    parser.add_argument(
        "--max-selected-side-share",
        type=float,
        default=float(DEFAULT_THRESHOLDS["max_selected_side_share"]),
    )
    parser.add_argument(
        "--min-mean-selected-rows",
        type=float,
        default=float(DEFAULT_THRESHOLDS["min_mean_selected_rows"]),
    )
    parser.add_argument(
        "--min-selected-rows",
        type=float,
        default=float(DEFAULT_THRESHOLDS["min_selected_rows"]),
    )
    parser.add_argument(
        "--min-retained-features",
        type=float,
        default=float(DEFAULT_THRESHOLDS["min_retained_features"]),
    )
    parser.add_argument(
        "--min-stage-a-oracle-recall",
        type=float,
        default=float(DEFAULT_THRESHOLDS["min_stage_a_oracle_recall"]),
    )
    parser.add_argument(
        "--min-final-oracle-recall",
        type=float,
        default=float(DEFAULT_THRESHOLDS["min_final_oracle_recall"]),
    )
    parser.add_argument(
        "--max-hard-risk-cap-no-trade-rate",
        type=float,
        default=float(DEFAULT_THRESHOLDS["max_hard_risk_cap_no_trade_rate"]),
    )
    parser.add_argument(
        "--candidate-min-positive-month-share",
        type=float,
        default=float(DEFAULT_THRESHOLDS["candidate_min_positive_month_share"]),
    )
    parser.add_argument(
        "--candidate-min-worst-month-mean-u",
        type=float,
        default=float(DEFAULT_THRESHOLDS["candidate_min_worst_month_mean_u"]),
    )
    parser.add_argument(
        "--candidate-max-bad-mae-1r-rate",
        type=float,
        default=float(DEFAULT_THRESHOLDS["candidate_max_bad_mae_1r_rate"]),
    )
    parser.add_argument(
        "--candidate-max-timeout-rate",
        type=float,
        default=float(DEFAULT_THRESHOLDS["candidate_max_timeout_rate"]),
    )
    parser.add_argument(
        "--candidate-min-stage-a-oracle-recall",
        type=float,
        default=float(DEFAULT_THRESHOLDS["candidate_min_stage_a_oracle_recall"]),
    )
    parser.add_argument(
        "--candidate-min-final-oracle-recall",
        type=float,
        default=float(DEFAULT_THRESHOLDS["candidate_min_final_oracle_recall"]),
    )
    parser.add_argument(
        "--candidate-min-clean-positive-rate",
        type=float,
        default=float(DEFAULT_THRESHOLDS["candidate_min_clean_positive_rate"]),
    )
    parser.add_argument(
        "--candidate-min-dirty-positive-rate",
        type=float,
        default=float(DEFAULT_THRESHOLDS["candidate_min_dirty_positive_rate"]),
    )
    return parser.parse_args()


def _thresholds_from_args(args: argparse.Namespace) -> dict[str, float]:
    return {
        "min_months": float(args.min_months),
        "min_positive_month_share": float(args.min_positive_month_share),
        "min_mean_u": float(args.min_mean_u),
        "min_worst_month_mean_u": float(args.min_worst_month_mean_u),
        "min_score_ic_u": float(args.min_score_ic_u),
        "min_top_bottom_decile_spread_u": float(args.min_top_bottom_decile_spread_u),
        "max_bad_mae_1r_rate": float(args.max_bad_mae_1r_rate),
        "max_timeout_rate": float(args.max_timeout_rate),
        "max_wide_barrier_25bps_rate": float(args.max_wide_barrier_25bps_rate),
        "max_selected_side_share": float(args.max_selected_side_share),
        "min_mean_selected_rows": float(args.min_mean_selected_rows),
        "min_selected_rows": float(args.min_selected_rows),
        "min_retained_features": float(args.min_retained_features),
        "min_stage_a_oracle_recall": float(args.min_stage_a_oracle_recall),
        "min_final_oracle_recall": float(args.min_final_oracle_recall),
        "max_hard_risk_cap_no_trade_rate": float(args.max_hard_risk_cap_no_trade_rate),
        "candidate_min_positive_month_share": float(args.candidate_min_positive_month_share),
        "candidate_min_worst_month_mean_u": float(args.candidate_min_worst_month_mean_u),
        "candidate_max_bad_mae_1r_rate": float(args.candidate_max_bad_mae_1r_rate),
        "candidate_max_timeout_rate": float(args.candidate_max_timeout_rate),
        "candidate_min_stage_a_oracle_recall": float(
            args.candidate_min_stage_a_oracle_recall
        ),
        "candidate_min_final_oracle_recall": float(args.candidate_min_final_oracle_recall),
        "candidate_min_clean_positive_rate": float(args.candidate_min_clean_positive_rate),
        "candidate_min_dirty_positive_rate": float(args.candidate_min_dirty_positive_rate),
    }


def main() -> int:
    args = parse_args()
    report = run_learnability_smoke(
        report_dir=args.report_dir,
        smoke_output_dir=args.smoke_output_dir,
        output=args.output,
        seeds=_parse_int_csv(args.seeds, []) if args.seeds else None,
        train_lookback_months=args.train_lookback_months,
        max_feature_store_features=args.max_feature_store_features,
        thresholds=_thresholds_from_args(args),
        target_label_arms=[
            arm.strip()
            for arm in str(args.target_label_arms or "").split(",")
            if arm.strip() and arm.strip() in FIXED_ARTIFACT_LABEL_ARMS
        ],
    )
    print(json.dumps(_json_safe(report), indent=2, sort_keys=True))
    return 0 if report["status"] in {"pass", "candidate_for_train_meta_path_filter_smoke"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
