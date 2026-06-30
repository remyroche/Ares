#!/usr/bin/env python3
"""Audit market-state head-priority artifacts against the shadow contract.

Head-priority modulation is not part of the executable threshold controller.
It is a shadow portfolio-routing experiment that may propose bounded
auction-priority action columns per head and timestamp. This audit verifies the
artifact shape and source contract before any economic interpretation.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_PRIORITY_DIR = Path(
    "data_perp/reports/market_state_head_priority_learning_topcandidate_replayaware_forced_shadow_20260626_jun15_22_v1"
)
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/market_state_head_priority_contract_audit_20260626"
)
BASELINE_ARM = "P0_static_priority"
REQUIRED_FILES = [
    "manifest.json",
    "head_priority_training_targets.parquet",
    "head_priority_score_rows.parquet",
    "head_priority_learned_schedule.parquet",
    "head_priority_learning_replay_summary.csv",
    "head_priority_learning_by_head.csv",
    "head_priority_learning_model_diagnostics.csv",
    "head_priority_learning_accepted_overlap.csv",
    "head_priority_learning_accepted_swap_utility.csv",
    "head_priority_score_feature_coverage.csv",
    "head_priority_config_selection.csv",
    "head_priority_config_fold_validation.csv",
]
FORBIDDEN_STATE_COLUMN_NAMES = {
    "strategy_id",
    "strategy",
    "head",
    "side",
    "symbol",
    "candidate_count",
    "accepted_trade_count",
    "accepted",
    "portfolio_pnl",
    "net_pnl",
    "gross_pnl",
    "net_return",
    "label",
    "target",
    "y",
    "rank",
    "rank_pct",
    "policy_rank_pct",
    "strategy_rank_pct",
    "score",
    "calibrated_score",
}
FORBIDDEN_STATE_COLUMN_TOKENS = {
    "accepted",
    "anchor",
    "candidate",
    "confidence",
    "decision",
    "fail",
    "failure",
    "headhealth",
    "leaf",
    "ledger",
    "margin",
    "meta",
    "model",
    "pnl",
    "policy",
    "portfolio",
    "prediction",
    "qfail",
    "rank",
    "reliability",
    "score",
    "side",
    "strategy",
    "symbol",
    "target",
    "trade",
    "y",
    "ybin",
}
ORDER_BOOK_TOKENS = {
    "ask",
    "bid",
    "book",
    "cancel",
    "cancellation",
    "depth",
    "imbalance",
    "level2",
    "l2",
    "microprice",
    "orderbook",
    "quote",
    "replenishment",
    "spread",
}
ALLOWED_MARKET_STATE_SEMANTIC_FEATURES = {
    # Explicit plan-approved market-state reliability channels. This keeps the
    # semantic audit strict for generic score/model/rank/performance fields.
    "state_drift_score",
    "state_ood_score",
    "state_ood_score_mean",
    "state_ood_score_max",
}
STATE_META_COLUMNS = {
    "timestamp",
    "fold",
    "split",
    "state_arm",
    "state_level",
    "prediction_contract",
    "state_feature_count",
    "head",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        try:
            return pd.read_csv(path)
        except EmptyDataError:
            return pd.DataFrame()
    return pd.DataFrame()


def _state_value_columns(frame: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in frame.columns:
        if col in STATE_META_COLUMNS:
            continue
        name = str(col)
        if not (name.startswith("state_") or name.startswith("forecast_")):
            continue
        if pd.to_numeric(frame[col], errors="coerce").notna().any():
            cols.append(name)
    return cols


def _unsafe_orderbook_columns(columns: list[str]) -> list[str]:
    unsafe: list[str] = []
    for col in columns:
        tokens = {part for part in str(col).lower().replace("-", "_").split("_") if part}
        if tokens.intersection(ORDER_BOOK_TOKENS):
            unsafe.append(str(col))
    return sorted(set(unsafe))


def _unsafe_semantic_state_columns(columns: list[str]) -> list[str]:
    unsafe: list[str] = []
    for col in columns:
        normalized = str(col).lower().replace("-", "_")
        if normalized in ALLOWED_MARKET_STATE_SEMANTIC_FEATURES:
            continue
        tokens = {part for part in normalized.split("_") if part}
        suffix = normalized.removeprefix("state_").removeprefix("forecast_")
        if suffix in FORBIDDEN_STATE_COLUMN_NAMES or tokens.intersection(FORBIDDEN_STATE_COLUMN_TOKENS):
            unsafe.append(str(col))
    return sorted(set(unsafe))


def _duplicate_count(frame: pd.DataFrame, keys: list[str]) -> int:
    if not all(key in frame.columns for key in keys):
        return -1
    return int(frame.duplicated(keys).sum())


def _audit_state_features(frame: pd.DataFrame, *, name: str) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    state_cols = _state_value_columns(frame)
    if not state_cols:
        failures.append(f"{name} has no numeric state_/forecast_ value columns")
        return state_cols, failures
    forbidden = sorted(set(state_cols).intersection(FORBIDDEN_STATE_COLUMN_NAMES))
    if forbidden:
        failures.append(f"{name} state features contain forbidden columns: {forbidden}")
    semantic_forbidden = _unsafe_semantic_state_columns(state_cols)
    if semantic_forbidden:
        failures.append(
            f"{name} state features contain strategy/model/performance-like columns: {semantic_forbidden[:10]}"
        )
    unsafe = _unsafe_orderbook_columns(state_cols)
    if unsafe:
        failures.append(f"{name} state features contain actual order-book-like columns: {unsafe[:10]}")
    return state_cols, failures


def _audit_constant_state_by_group(
    frame: pd.DataFrame,
    *,
    state_cols: list[str],
    keys: list[str],
    name: str,
) -> list[str]:
    failures: list[str] = []
    missing = [key for key in keys if key not in frame.columns]
    if missing:
        failures.append(f"{name} missing state invariance keys: {missing}")
        return failures
    for col in state_cols:
        counts = frame.groupby(keys, dropna=False)[col].nunique(dropna=False)
        bad = int((counts > 1).sum())
        if bad:
            failures.append(f"{name}.{col} varies within {keys}: {bad} groups")
    return failures


def _audit_manifest(manifest: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if manifest.get("generated_by") != "run_market_state_head_priority_learning":
        failures.append("manifest.generated_by is unexpected")
    if manifest.get("purpose") != "learned_market_state_head_priority_modulation_shadow_ablation":
        failures.append("manifest.purpose is not the shadow head-priority ablation")
    contract = manifest.get("contract")
    if not isinstance(contract, dict):
        failures.append("manifest.contract is missing")
        return failures
    expected_false = [
        "changes_scores_or_ranks",
        "changes_thresholds",
        "changes_position_sizing",
        "qfail_active",
        "head_health_active",
        "market_state_threshold_controller_active",
    ]
    for key in expected_false:
        if contract.get(key) is not False:
            failures.append(f"manifest.contract.{key} is not false")
    if contract.get("changes_auction_ordering") is not True:
        failures.append("manifest.contract.changes_auction_ordering is not true for shadow priority modulation")
    if contract.get("priority_adjustment_column") != "portfolio_priority_adjustment":
        failures.append("manifest.contract.priority_adjustment_column is not portfolio_priority_adjustment")
    multiplier_col = contract.get("priority_multiplier_column")
    if multiplier_col not in (None, "portfolio_priority_multiplier"):
        failures.append(
            "manifest.contract.priority_multiplier_column is not portfolio_priority_multiplier"
        )
    if contract.get("operational_status") != "shadow_only":
        failures.append("manifest.contract.operational_status is not shadow_only")
    if contract.get("execution_enabled") is not False:
        failures.append("manifest.contract.execution_enabled is not false")
    if contract.get("production_eligible") is not False:
        failures.append("manifest.contract.production_eligible is not false")
    if contract.get("requires_promotion_gate") is not True:
        failures.append("manifest.contract.requires_promotion_gate is not true")
    if contract.get("market_state_encoder_uses_candidate_features") is not False:
        failures.append("manifest.contract.market_state_encoder_uses_candidate_features is not false")
    return failures


def _audit_schedule(schedule: pd.DataFrame, manifest: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if schedule.empty:
        return ["head_priority_learned_schedule is empty"]
    missing = [col for col in ("timestamp", "head", "portfolio_priority_adjustment") if col not in schedule.columns]
    if missing:
        return [f"head_priority_learned_schedule missing columns: {missing}"]
    dupes = _duplicate_count(schedule, ["timestamp", "head"])
    if dupes > 0:
        failures.append(f"head_priority_learned_schedule duplicate timestamp/head rows: {dupes}")
    adjustment = pd.to_numeric(schedule["portfolio_priority_adjustment"], errors="coerce")
    if bool(adjustment.isna().any()):
        failures.append("head_priority_learned_schedule.portfolio_priority_adjustment has non-finite values")
    params = manifest.get("params") or {}
    max_adjustment = float(params.get("max_adjustment", 0.20) or 0.20)
    selected = (manifest.get("selection") or {}).get("selected")
    if isinstance(selected, dict) and selected.get("config_max_adjustment") is not None:
        max_adjustment = float(selected.get("config_max_adjustment") or max_adjustment)
    if bool((adjustment.abs() > abs(max_adjustment) + 1e-12).any()):
        failures.append("head_priority_learned_schedule.portfolio_priority_adjustment exceeds max_adjustment")
    centered = schedule.groupby("timestamp", dropna=False)["portfolio_priority_adjustment"].sum()
    if bool((centered.abs() > 1e-6).any()):
        failures.append("head_priority_learned_schedule adjustments are not timestamp-centered")
    if "portfolio_priority_multiplier" in schedule.columns:
        multiplier = pd.to_numeric(schedule["portfolio_priority_multiplier"], errors="coerce")
        if bool(multiplier.isna().any()):
            failures.append("head_priority_learned_schedule.portfolio_priority_multiplier has non-finite values")
        if bool((multiplier < 0.0).any()):
            failures.append("head_priority_learned_schedule.portfolio_priority_multiplier is negative")
        max_multiplier = float(params.get("max_priority_multiplier", 1.0) or 1.0)
        if isinstance(selected, dict) and selected.get("config_max_priority_multiplier") is not None:
            max_multiplier = float(selected.get("config_max_priority_multiplier") or max_multiplier)
        max_multiplier = max(max_multiplier, 1.0)
        if bool((multiplier > max_multiplier + 1e-12).any()):
            failures.append(
                "head_priority_learned_schedule.portfolio_priority_multiplier exceeds max_priority_multiplier"
            )
        min_multiplier = 1.0 / max(max_multiplier, 1e-12)
        if bool((multiplier < min_multiplier - 1e-12).any()):
            failures.append(
                "head_priority_learned_schedule.portfolio_priority_multiplier below reciprocal max_priority_multiplier"
            )
    return failures


def _audit_activation_filter(
    manifest: dict[str, Any],
    *,
    training_state_cols: list[str],
    score_state_cols: list[str],
) -> list[str]:
    failures: list[str] = []
    activation = manifest.get("state_head_activation_filter")
    if activation is None:
        return failures
    if not isinstance(activation, dict):
        return ["manifest.state_head_activation_filter is not an object"]
    if activation.get("enabled") is not True:
        return failures
    allowed = set(map(str, activation.get("allowed_state_heads") or []))
    if not allowed:
        return ["manifest.state_head_activation_filter is enabled with no allowed_state_heads"]
    for name, cols in (
        ("head_priority_training_targets", training_state_cols),
        ("head_priority_score_rows", score_state_cols),
    ):
        unexpected = sorted(set(cols).difference(allowed))
        if unexpected:
            failures.append(f"{name} contains state heads outside activation filter: {unexpected[:20]}")
    missing_from_both = sorted(allowed.difference(set(training_state_cols)).difference(set(score_state_cols)))
    if missing_from_both:
        failures.append(
            "manifest.state_head_activation_filter.allowed_state_heads are absent from both "
            f"training and score artifacts: {missing_from_both[:20]}"
        )
    return failures


def _audit_selection(
    selection: pd.DataFrame,
    diagnostics: pd.DataFrame,
    manifest: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    params = manifest.get("params") or {}
    grid_enabled = bool(params.get("select_config_grid") is True)
    if grid_enabled and selection.empty:
        failures.append("head_priority_config_selection is empty")
    elif not selection.empty and "selection_gate_passed" not in selection.columns:
        failures.append("head_priority_config_selection missing selection_gate_passed")
    if diagnostics.empty:
        failures.append("head_priority_learning_model_diagnostics is empty")
    else:
        for col in ("selection_gate_passed", "selection_objective"):
            if col not in diagnostics.columns:
                failures.append(f"head_priority_learning_model_diagnostics missing {col}")
    return failures


def audit_head_priority_contract(priority_dir: Path) -> dict[str, Any]:
    failures: list[str] = []
    missing = [name for name in REQUIRED_FILES if not (priority_dir / name).exists()]
    if missing:
        failures.append(f"missing required files: {missing}")

    manifest = _read_json(priority_dir / "manifest.json")
    failures.extend(_audit_manifest(manifest))

    training = _read_frame(priority_dir / "head_priority_training_targets.parquet")
    score_rows = _read_frame(priority_dir / "head_priority_score_rows.parquet")
    schedule = _read_frame(priority_dir / "head_priority_learned_schedule.parquet")
    selection = _read_frame(priority_dir / "head_priority_config_selection.csv")
    diagnostics = _read_frame(priority_dir / "head_priority_learning_model_diagnostics.csv")

    if training.empty:
        failures.append("head_priority_training_targets is empty")
        training_state_cols: list[str] = []
    else:
        training_state_cols, state_failures = _audit_state_features(
            training,
            name="head_priority_training_targets",
        )
        failures.extend(state_failures)
        failures.extend(
            _audit_constant_state_by_group(
                training,
                state_cols=training_state_cols,
                keys=["fold", "timestamp"],
                name="head_priority_training_targets",
            )
        )

    if score_rows.empty:
        failures.append("head_priority_score_rows is empty")
        score_state_cols: list[str] = []
    else:
        score_state_cols, state_failures = _audit_state_features(
            score_rows,
            name="head_priority_score_rows",
        )
        failures.extend(state_failures)
        dupes = _duplicate_count(score_rows, ["timestamp", "head"])
        if dupes > 0:
            failures.append(f"head_priority_score_rows duplicate timestamp/head rows: {dupes}")
        failures.extend(
            _audit_constant_state_by_group(
                score_rows,
                state_cols=score_state_cols,
                keys=["timestamp"],
                name="head_priority_score_rows",
            )
        )

    feature_mismatch = sorted(set(training_state_cols).symmetric_difference(score_state_cols))
    if feature_mismatch:
        failures.append(f"training and score state feature sets differ: {feature_mismatch[:20]}")

    failures.extend(
        _audit_activation_filter(
            manifest,
            training_state_cols=training_state_cols,
            score_state_cols=score_state_cols,
        )
    )
    failures.extend(_audit_schedule(schedule, manifest))
    failures.extend(_audit_selection(selection, diagnostics, manifest))

    payload = {
        "generated_by": "audit_market_state_head_priority_contract",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "priority_dir": str(priority_dir),
        "passed": not failures,
        "failures": failures,
        "required_files_checked": REQUIRED_FILES,
        "training_rows": int(len(training)),
        "score_rows": int(len(score_rows)),
        "schedule_rows": int(len(schedule)),
        "training_state_feature_count": int(len(training_state_cols)),
        "score_state_feature_count": int(len(score_state_cols)),
        "selection_gate_passed": (
            bool(diagnostics.iloc[0].get("selection_gate_passed"))
            if not diagnostics.empty and "selection_gate_passed" in diagnostics.columns
            else None
        ),
        "replay_aware_selection": bool(
            int((manifest.get("params") or {}).get("selection_replay_top_n", 0) or 0) > 0
        ),
    }
    return payload


def _render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Market-State Head-Priority Contract Audit",
        "",
        f"Priority dir: `{payload['priority_dir']}`",
        f"Passed: `{payload['passed']}`",
        f"Training rows: `{payload['training_rows']}`",
        f"Score rows: `{payload['score_rows']}`",
        f"Schedule rows: `{payload['schedule_rows']}`",
        f"Training state features: `{payload['training_state_feature_count']}`",
        f"Score state features: `{payload['score_state_feature_count']}`",
        f"Selection gate passed: `{payload['selection_gate_passed']}`",
        f"Replay-aware selection: `{payload['replay_aware_selection']}`",
        "",
        "## Failures",
        "",
    ]
    failures = payload.get("failures") or []
    if failures:
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.append("_None._")
    return "\n".join(lines) + "\n"


def write_audit(payload: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / "market_state_head_priority_contract_audit.json",
        "report": output_dir / "market_state_head_priority_contract_audit.md",
    }
    paths["json"].write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    paths["report"].write_text(_render_report(payload), encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--priority-dir", type=Path, default=DEFAULT_PRIORITY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    payload = audit_head_priority_contract(args.priority_dir)
    paths = write_audit(payload, args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "passed": bool(payload["passed"]),
                "failure_count": len(payload["failures"]),
                "report": str(paths["report"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
