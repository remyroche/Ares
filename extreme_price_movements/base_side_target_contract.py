"""Reusable side-specific continuous base-target and weight contract."""

from __future__ import annotations

import json
import os
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from extreme_price_movements.hierarchical_label_weights import (
    TargetStrengthWeightSpec,
    build_target_strength_weights,
)


DEFAULT_CONTRACT_PATH = Path(__file__).resolve().parents[1] / "docs/promoted_base_side_target_contract.json"
TARGET_MODE = "side_continuous_geometry_v1"
WEIGHT_ARM = "W_side_target_strength_v1"
_MFE_KEYS = ("05", "075", "1", "125", "15")
_MAE_KEYS = ("05", "075", "1", "15")
_R_VALUES = {"05": 0.50, "075": 0.75, "1": 1.00, "125": 1.25, "15": 1.50}


def _contract_path() -> Path:
    configured = str(os.environ.get("EPM_BASE_SIDE_TARGET_CONTRACT_JSON", "")).strip()
    return Path(configured) if configured else DEFAULT_CONTRACT_PATH


def load_promoted_side_target_contract(path: Path | None = None) -> dict[str, Any]:
    source = Path(path or _contract_path())
    payload = json.loads(source.read_text(encoding="utf-8"))
    winners = dict(payload.get("winner_by_side") or {})
    if set(winners) != {"long", "short"}:
        raise ValueError(f"Invalid promoted side-target contract: {source}")
    return payload


def promoted_side_target_provenance(
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the split target/weight contracts persisted at the meta handoff."""

    payload = dict(contract or load_promoted_side_target_contract())
    winners = dict(payload["winner_by_side"])
    target_contract = {
        "schema": "base_soft_label_contract_v1",
        "target_column": "__first_touch_target_soft__",
        "target_mode": str(payload.get("target_mode", TARGET_MODE)),
        "geometry_by_side": {
            side: dict(dict(winners[side])["geometry"])
            for side in ("long", "short")
        },
        "selection_scope": payload.get("selection_scope"),
        "source": payload.get("source"),
        "execution_geometry_promoted": bool(
            payload.get("execution_geometry_promoted", False)
        ),
    }
    weight_contract = {
        "schema": "target_strength_weight_v1",
        "weight_arm": str(payload.get("weight_arm", WEIGHT_ARM)),
        "spec_by_side": {
            side: {
                "exponent": float(dict(dict(winners[side])["weight"])["target_exponent"]),
                "weight_range_ratio": float(
                    dict(dict(winners[side])["weight"])["weight_range_ratio"]
                ),
            }
            for side in ("long", "short")
        },
        "normalization": "train_fold_side_mean_one_then_side_x_archetype_context",
        "selection_scope": payload.get("selection_scope"),
        "source": payload.get("source"),
    }

    def stable_hash(value: Mapping[str, Any]) -> str:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    return {
        "base_target_contract": target_contract,
        "base_target_contract_hash": stable_hash(target_contract),
        "base_sample_weight_spec": weight_contract,
        "base_sample_weight_spec_hash": stable_hash(weight_contract),
    }


def _side_names(frame: pd.DataFrame) -> np.ndarray:
    raw = frame.get("side_name", frame.get("side", frame.get("__side__", 1.0)))
    values = pd.Series(raw, index=frame.index)
    numeric = pd.to_numeric(values, errors="coerce")
    text = values.astype(str).str.lower()
    return np.where(
        text.str.contains("short", regex=False).to_numpy()
        | numeric.lt(0.0).fillna(False).to_numpy(),
        "short",
        "long",
    )


def _archetypes(frame: pd.DataFrame) -> pd.Series:
    for column in (
        "__archetype_label_family__",
        "archetype_label_family",
        "policy_archetype",
        "local_side_archetype",
        "source_archetype",
    ):
        if column in frame.columns:
            return frame[column].fillna("__missing__").astype(str)
    return pd.Series("__missing__", index=frame.index, dtype="string")


def _finite(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(f"Side-target contract requires label column {column}")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Side-target contract requires finite label column {column}")
    return values


def _event_bars(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(f"Side-target contract requires label column {column}")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
    if np.isinf(values).any():
        raise ValueError(f"Side-target contract found infinite event bars in {column}")
    values = np.where(np.isnan(values), -1.0, values)
    if np.any((values != -1.0) & (values < 1.0)):
        raise ValueError(f"Side-target contract found invalid event bars in {column}")
    return values


def _side_target(frame: pd.DataFrame, geometry: Mapping[str, Any]) -> np.ndarray:
    barrier = _finite(frame, "__barrier_pct__")
    timeout_gross = _finite(frame, "__y_ret__") + _finite(
        frame, "__first_touch_round_trip_cost__"
    )
    cost = _finite(frame, "__first_touch_round_trip_cost__")
    if not np.allclose(cost, 0.01, rtol=0.0, atol=1e-8):
        raise ValueError("Side-target contract requires exactly one stored 1% round-trip cost")
    tp_r = float(geometry["tp_r"])
    sl_r = float(geometry["sl_r"])
    tp_key = next(key for key in _MFE_KEYS if np.isclose(_R_VALUES[key], tp_r, rtol=0.0, atol=1e-9))
    sl_key = next(key for key in _MAE_KEYS if np.isclose(_R_VALUES[key], sl_r, rtol=0.0, atol=1e-9))
    tp_bar = _event_bars(frame, f"__bars_to_mfe_{tp_key}r__")
    sl_bar = _event_bars(frame, f"__bars_to_mae_{sl_key}r__")
    max_bars = int(geometry["max_profit_bars"])
    tp_hit = (tp_bar > 0.0) & (tp_bar <= max_bars)
    sl_hit = sl_bar > 0.0
    tp_first = tp_hit & (~sl_hit | (tp_bar < sl_bar))
    sl_first = sl_hit & ~tp_first
    timeout = ~(tp_first | sl_first)
    gross = timeout_gross.copy()
    gross[tp_first] = tp_r * barrier[tp_first]
    gross[sl_first] = -sl_r * barrier[sl_first]
    resolved = np.where(tp_first, tp_bar, np.where(sl_first, sl_bar, float(max_bars)))
    temperature = max(float(geometry["temperature"]), 1e-4)
    if "net_edge" in geometry:
        economic_outcome = gross - cost
        economic_edge = float(geometry["net_edge"])
    else:
        # Backward compatibility for the already promoted gross-target contract.
        economic_outcome = gross
        economic_edge = float(geometry["gross_edge"])
    edge = 1.0 / (
        1.0
        + np.exp(
            -np.clip((economic_outcome - economic_edge) / temperature, -50.0, 50.0)
        )
    )
    speed = np.exp(-np.maximum(resolved, 0.0) / max(float(geometry["slow_profit_bars"]), 1.0))
    multiplier = (
        0.50
        + 0.20 * tp_first
        + 0.20 * speed
        - float(geometry["mae_penalty"]) * 0.25 * sl_first
        - float(geometry["timeout_penalty"]) * 0.20 * timeout
        - float(geometry["first_pass_penalty"]) * 0.10 * sl_first
    )
    return np.clip(edge * np.clip(multiplier, 0.0, 1.5), 0.0, 1.0).astype(np.float32)


def build_promoted_side_target(
    frame: pd.DataFrame,
    *,
    contract: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    payload = dict(contract or load_promoted_side_target_contract())
    winners = dict(payload["winner_by_side"])
    sides = _side_names(frame)
    target = np.zeros(len(frame), dtype=np.float32)
    for side in ("long", "short"):
        positions = np.flatnonzero(sides == side)
        if not len(positions):
            continue
        target[positions] = _side_target(
            frame.iloc[positions], dict(winners[side])["geometry"]
        )
    return pd.DataFrame(
        {
            "target_soft": target,
            "target_hard": (target >= 0.5).astype(np.float32),
        },
        index=frame.index,
    )


def build_promoted_side_weights(
    frame: pd.DataFrame,
    target: pd.DataFrame,
    *,
    contract: Mapping[str, Any] | None = None,
) -> pd.Series:
    payload = dict(contract or load_promoted_side_target_contract())
    winners = dict(payload["winner_by_side"])
    sides = _side_names(frame)
    soft = pd.to_numeric(target["target_soft"], errors="coerce").to_numpy(np.float64)
    weights = np.ones(len(frame), dtype=np.float32)
    archetypes = _archetypes(frame)
    for side in ("long", "short"):
        positions = np.flatnonzero(sides == side)
        if not len(positions):
            continue
        weight_config = dict(dict(winners[side])["weight"])
        local, _ = build_target_strength_weights(
            soft[positions],
            timestamps=frame.iloc[positions]["__ts__"],
            archetypes=archetypes.iloc[positions],
            spec=TargetStrengthWeightSpec(
                exponent=float(weight_config["target_exponent"]),
                weight_range_ratio=float(weight_config["weight_range_ratio"]),
            ),
        )
        weights[positions] = local.astype(np.float32, copy=False)
    return pd.Series(weights, index=frame.index, name="sample_weight", dtype=np.float32)
