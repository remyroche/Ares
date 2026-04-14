"""Shared parity helpers for inference and inference_backtest paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Set

import pandas as pd

from extreme_price_movements.simple_position_sizer import (
    calibrate_score,
    load_calibration_contract,
)
from extreme_price_movements.utils import tprint


def load_strategy_acceptance_filter(data_root: str, run_id: str) -> Optional[Set[str]]:
    """Load accepted strategy identifiers from policy optimiser artifacts."""
    paths = [
        Path(data_root) / "artifacts" / run_id / "strategy_final_acceptation.json",
        Path(data_root)
        / "artifacts"
        / run_id
        / "policy_params"
        / "strategy_final_acceptation.json",
    ]

    for path in paths:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
            strategies = payload.get("strategies", [])
            accepted = {
                str(s["strategy_id"])
                for s in strategies
                if isinstance(s, dict) and s.get("strategy_id")
            }
            tprint(
                f"[StrategyFilter] Loaded {len(accepted)} accepted strategies from {path}"
            )
            return accepted
        except Exception as exc:
            tprint(f"[StrategyFilter] Error loading {path}: {exc}")
    return None


def apply_strategy_acceptance_filter(
    df: pd.DataFrame,
    accepted_strategies: Optional[Set[str]],
    strategy_col: str = "strategy",
) -> pd.DataFrame:
    """Filter rows to strategies accepted by policy optimisation."""
    if accepted_strategies is None:
        return df
    n_before = len(df)
    out = df[df[strategy_col].astype(str).isin(accepted_strategies)].copy()
    tprint(f"[StrategyFilter] {n_before} -> {len(out)} rows after acceptance filtering")
    return out


def calibrated_score_and_threshold(
    raw_score: float,
    strategy_id: str,
    calibration_data: Dict[str, Dict[str, Any]],
    default_threshold: float = 0.5,
) -> tuple[float, float]:
    """Return calibrated score and p75 threshold for a strategy."""
    if not calibration_data:
        return float(raw_score), float(default_threshold)

    sid = str(strategy_id)
    calib = calibration_data.get(sid, {}) if isinstance(calibration_data, dict) else {}
    calibrated = float(calibrate_score(raw_score, sid, calibration_data))
    p75 = float(calib.get("p75_threshold", default_threshold) or default_threshold)
    return calibrated, p75


def passes_rank_filter(
    raw_score: float,
    strategy_id: str,
    calibration_data: Dict[str, Dict[str, Any]],
    default_threshold: float = 0.5,
) -> bool:
    """Check if a score passes strategy-specific confidence rank threshold."""
    calibrated, threshold = calibrated_score_and_threshold(
        raw_score=raw_score,
        strategy_id=strategy_id,
        calibration_data=calibration_data,
        default_threshold=default_threshold,
    )
    return bool(calibrated >= threshold)


def calibration_size_multiplier(
    raw_score: float,
    strategy_id: str,
    calibration_data: Dict[str, Dict[str, Any]],
    default_threshold: float = 0.5,
    max_mult: float = 2.0,
) -> float:
    """Convert calibrated rank strength into a bounded sizing multiplier."""
    calibrated, threshold = calibrated_score_and_threshold(
        raw_score=raw_score,
        strategy_id=strategy_id,
        calibration_data=calibration_data,
        default_threshold=default_threshold,
    )
    den = max(float(threshold), 1e-6)
    rel = max(0.0, float(calibrated) / den)
    return float(min(rel, float(max_mult)))


def validate_calibration_artifacts(
    data_root: str,
    run_id: str,
    calibration_data: Dict[str, Dict[str, Any]],
    *,
    strict: bool = True,
) -> bool:
    """Validate calibration artifact schema expected by inference runtime."""
    contract = load_calibration_contract(data_root, run_id)
    if not contract:
        if strict and calibration_data:
            raise ValueError(
                "Calibration data exists but confidence_calibration.contract.json is missing"
            )
        return False
    req = list(contract.get("required_strategy_fields", []) or [])
    for sid, row in (calibration_data or {}).items():
        missing = [k for k in req if k not in row]
        if missing:
            raise ValueError(
                f"Calibration artifact schema mismatch for strategy {sid}: missing={missing}"
            )
    return True
