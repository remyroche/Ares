"""Train-only certainty diagnostics for nearby executable-label contracts.

This module deliberately consumes realised paths.  Its outputs are label
metadata and sample-weight inputs, never decision-time features.  Keeping the
contract and the inference guard together makes accidental future leakage
detectable at the boundary where a training ledger is assembled.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


HORIZON_MINUTES = 720
CERTAINTY_PREFIX = "__label_certainty__"
CERTAINTY_SCHEMA = "label_certainty_stability_v1"


@dataclass(frozen=True)
class PerturbationContract:
    """One predeclared nearby triple-barrier path contract."""

    contract_id: str
    horizon_minutes: int = 720
    entry_delay_minutes: int = 0
    upper_atr_multiple: float = 1.5
    lower_atr_multiple: float = 1.0
    atr_source: str = "reference"


# A balanced 10-contract neighbourhood, intentionally not a factorial grid.
DEFAULT_PERTURBATION_CONTRACTS: tuple[PerturbationContract, ...] = (
    PerturbationContract("reference"),
    PerturbationContract("barrier_minus_10", upper_atr_multiple=1.35, lower_atr_multiple=0.90),
    PerturbationContract("barrier_plus_10", upper_atr_multiple=1.65, lower_atr_multiple=1.10),
    PerturbationContract("barrier_minus_20", upper_atr_multiple=1.20, lower_atr_multiple=0.80),
    PerturbationContract("barrier_plus_20", upper_atr_multiple=1.80, lower_atr_multiple=1.20),
    PerturbationContract("horizon_8h", horizon_minutes=480),
    PerturbationContract("horizon_16h", horizon_minutes=960),
    PerturbationContract("entry_delay_1m", entry_delay_minutes=1),
    PerturbationContract("entry_delay_5m", entry_delay_minutes=5),
    PerturbationContract("atr_long", atr_source="long"),
)


def contracts_payload(contracts: Sequence[PerturbationContract] = DEFAULT_PERTURBATION_CONTRACTS) -> dict[str, object]:
    """Serializable frozen contract, including the explicit train-only rule."""
    identifiers = [item.contract_id for item in contracts]
    if len(identifiers) != len(set(identifiers)) or "reference" not in identifiers:
        raise ValueError("certainty contracts require one unique reference contract")
    return {
        "schema": CERTAINTY_SCHEMA,
        "contracts": [asdict(item) for item in contracts],
        "balanced_predeclared_subset": True,
        "not_a_factorial_search": True,
        "use": "training_label_diagnostic_and_sample_weight_only",
        "inference_feature_policy": "forbidden",
    }


def assert_no_label_certainty_inference_features(columns: Iterable[str]) -> None:
    """Fail closed if outcome-derived certainty reaches an inference matrix."""
    forbidden = sorted(str(column) for column in columns if str(column).startswith(CERTAINTY_PREFIX))
    if forbidden:
        raise ValueError("label-certainty diagnostics are training-only and cannot be inference features: " + ", ".join(forbidden))


def _first_index(mask: np.ndarray) -> np.ndarray:
    return np.where(mask.any(axis=1), mask.argmax(axis=1), -1).astype(np.int16)


def _validate_paths(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> tuple[int, int]:
    arrays = tuple(np.asarray(value, dtype=np.float64) for value in (open_, high, low, close))
    if any(value.ndim != 2 for value in arrays) or not (arrays[0].shape == arrays[1].shape == arrays[2].shape == arrays[3].shape):
        raise ValueError("OHLC paths must be same-shape two-dimensional arrays")
    rows, minutes = arrays[0].shape
    if rows == 0 or minutes < HORIZON_MINUTES:
        raise ValueError("paths must have at least a complete reference 12h horizon")
    if not np.isfinite(np.stack(arrays, axis=2)).all() or (np.stack(arrays, axis=2) <= 0.0).any() or (arrays[1] < arrays[2]).any():
        raise ValueError("paths must be finite positive OHLC with high >= low")
    return rows, minutes


def _atr_for_source(reference: np.ndarray, source: str, alternatives: Mapping[str, Sequence[float]] | None) -> np.ndarray:
    if source == "reference":
        return reference
    if alternatives is None or source not in alternatives:
        raise ValueError(f"perturbation requires materialized ATR source '{source}', not a proxy")
    result = np.asarray(alternatives[source], dtype=np.float64)
    if result.shape != reference.shape or not np.isfinite(result).all() or (result <= 0.0).any():
        raise ValueError(f"ATR source '{source}' must be finite, positive, and row-aligned")
    return result


def materialize_perturbed_barrier_targets(
    *,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    side_sign: Sequence[float],
    atr_reference: Sequence[float],
    cost_return: Sequence[float],
    atr_alternatives: Mapping[str, Sequence[float]] | None = None,
    contracts: Sequence[PerturbationContract] = DEFAULT_PERTURBATION_CONTRACTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a small contract neighbourhood on full realised OHLC paths.

    ``target_value`` is a deliberately simple path outcome: +1 clean upper
    first, -1 lower first, otherwise terminal signed net return.  This is a
    diagnostic neighbour, not a replacement for the authoritative policy-EV
    label (whose exit path need not be reconstructible from OHLC).
    """
    rows, minutes = _validate_paths(open_, high, low, close)
    sign = np.asarray(side_sign, dtype=np.float64)
    atr = np.asarray(atr_reference, dtype=np.float64)
    cost = np.asarray(cost_return, dtype=np.float64)
    if sign.shape != (rows,) or not np.isin(sign, (-1.0, 1.0)).all() or atr.shape != (rows,) or cost.shape != (rows,):
        raise ValueError("side, ATR, and cost vectors must be row-aligned; side must be +/-1")
    if not np.isfinite(atr).all() or (atr <= 0.0).any() or not np.isfinite(cost).all() or (cost < 0.0).any():
        raise ValueError("ATR must be positive; cost must be finite and non-negative")
    records: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    for contract in contracts:
        end = contract.entry_delay_minutes + contract.horizon_minutes
        if contract.entry_delay_minutes < 0 or contract.horizon_minutes <= 0 or end > minutes:
            raise ValueError(f"contract {contract.contract_id} is outside the materialized path horizon")
        entry = open_[:, contract.entry_delay_minutes]
        current_atr = _atr_for_source(atr, contract.atr_source, atr_alternatives)
        upper = np.maximum(contract.upper_atr_multiple * current_atr, cost)
        lower = contract.lower_atr_multiple * current_atr
        high_slice, low_slice, close_slice = high[:, contract.entry_delay_minutes:end], low[:, contract.entry_delay_minutes:end], close[:, contract.entry_delay_minutes:end]
        favorable = np.where(sign[:, None] > 0.0, high_slice / entry[:, None] - 1.0 >= upper[:, None], 1.0 - low_slice / entry[:, None] >= upper[:, None])
        adverse = np.where(sign[:, None] > 0.0, 1.0 - low_slice / entry[:, None] >= lower[:, None], high_slice / entry[:, None] - 1.0 >= lower[:, None])
        first_fav, first_adv = _first_index(favorable), _first_index(adverse)
        has_fav, has_adv = first_fav >= 0, first_adv >= 0
        clean = has_fav & (~has_adv | (first_fav < first_adv))
        adverse_first = has_adv & (~has_fav | (first_adv <= first_fav))
        terminal_gross = np.where(sign > 0.0, close_slice[:, -1] / entry - 1.0, 1.0 - close_slice[:, -1] / entry)
        terminal_net = terminal_gross - cost
        event = np.full(rows, "timeout", dtype=object)
        event[clean], event[adverse_first] = "clean_first", "adverse_first"
        value = terminal_net.copy()
        value[clean], value[adverse_first] = 1.0, -1.0
        same_bar = has_fav & has_adv & (first_fav == first_adv)
        nearest = np.minimum(np.abs(terminal_gross - upper), np.abs(terminal_gross + lower))
        records.append(pd.DataFrame({
            "__certainty_row_id__": np.arange(rows, dtype=np.int64),
            "contract_id": contract.contract_id, "target_value": value, "target_sign": np.sign(value).astype(np.int8),
            "event": event, "top_state": (value >= upper).astype(np.int8), "bottom_state": (value <= -lower).astype(np.int8),
            "nearest_boundary_distance": nearest, "same_bar_conflict": same_bar.astype(np.int8), "path_complete": np.ones(rows, dtype=np.int8),
            "horizon_minutes": contract.horizon_minutes, "entry_delay_minutes": contract.entry_delay_minutes, "atr_source": contract.atr_source,
        }))
        diagnostics.append({"contract_id": contract.contract_id, "rows": rows, "clean_first_rate": float(clean.mean()), "adverse_first_rate": float(adverse_first.mean()), "same_bar_conflict_rate": float(same_bar.mean())})
    return pd.concat(records, ignore_index=True), pd.DataFrame(diagnostics)


def build_label_certainty(variants: pd.DataFrame, *, reference_target: Sequence[float] | None = None) -> pd.DataFrame:
    """Return individual certainty components plus one bounded, auditable score."""
    required = {"__certainty_row_id__", "contract_id", "target_value", "target_sign", "event", "top_state", "bottom_state", "nearest_boundary_distance", "same_bar_conflict", "path_complete", "entry_delay_minutes", "atr_source"}
    missing = required.difference(variants.columns)
    if missing:
        raise ValueError(f"variant surface lacks certainty fields: {sorted(missing)}")
    reference = variants.loc[variants.contract_id.eq("reference")].reset_index(drop=True)
    if reference.empty or variants.groupby("contract_id", sort=False).size().nunique() != 1:
        raise ValueError("variants must have complete equally sized contracts including reference")
    rows = len(reference)
    contracts = list(dict.fromkeys(variants.contract_id.astype(str)))
    pivot = lambda value: variants.pivot(index="__certainty_row_id__", columns="contract_id", values=value).loc[:, contracts]
    matrix = pivot("target_value").to_numpy(dtype=float)
    event = pivot("event").astype(str).to_numpy()
    sign = np.sign(matrix)
    ref_value = np.asarray(reference_target, dtype=float) if reference_target is not None else reference.target_value.to_numpy(dtype=float)
    if ref_value.shape != (rows,) or not np.isfinite(ref_value).all():
        raise ValueError("reference_target must be finite and row-aligned")
    ref_sign = np.sign(ref_value)
    agreement = (event == reference.event.to_numpy(dtype=str)[:, None]).mean(axis=1)
    sign_agreement = (sign == ref_sign[:, None]).mean(axis=1)
    top_bottom = ((pivot("top_state").to_numpy() == reference.top_state.to_numpy()[:, None]) & (pivot("bottom_state").to_numpy() == reference.bottom_state.to_numpy()[:, None])).mean(axis=1)
    dispersion = np.nanstd(matrix, axis=1)
    scale = float(np.nanmedian(np.abs(ref_value))) or 1.0
    dispersion_score = 1.0 / (1.0 + dispersion / scale)
    boundary_score = np.tanh(reference.nearest_boundary_distance.to_numpy(dtype=float) / np.maximum(np.nanmedian(reference.nearest_boundary_distance.to_numpy(dtype=float)), 1e-8))
    conflict_score = 1.0 - pivot("same_bar_conflict").to_numpy(dtype=float).mean(axis=1)
    completeness = pivot("path_complete").to_numpy(dtype=float).mean(axis=1)
    delay_ids = [index for index, value in enumerate(variants.groupby("contract_id", sort=False).first().loc[contracts, "entry_delay_minutes"].to_numpy()) if value > 0]
    atr_ids = [index for index, value in enumerate(variants.groupby("contract_id", sort=False).first().loc[contracts, "atr_source"].astype(str).to_numpy()) if value != "reference"]
    delay_sensitivity = np.mean(np.abs(matrix[:, delay_ids] - matrix[:, [contracts.index("reference")]]), axis=1) if delay_ids else np.zeros(rows)
    atr_sensitivity = np.mean(np.abs(matrix[:, atr_ids] - matrix[:, [contracts.index("reference")]]), axis=1) if atr_ids else np.zeros(rows)
    sensitivity_score = 1.0 / (1.0 + delay_sensitivity / scale + atr_sensitivity / scale)
    certainty = np.clip(0.20 * agreement + 0.15 * sign_agreement + 0.15 * top_bottom + 0.15 * dispersion_score + 0.10 * boundary_score + 0.10 * conflict_score + 0.10 * completeness + 0.05 * sensitivity_score, 0.0, 1.0)
    return pd.DataFrame({
        f"{CERTAINTY_PREFIX}event_agreement_rate": agreement, f"{CERTAINTY_PREFIX}target_sign_agreement_rate": sign_agreement,
        f"{CERTAINTY_PREFIX}target_value_dispersion": dispersion, f"{CERTAINTY_PREFIX}top_bottom_state_agreement": top_bottom,
        f"{CERTAINTY_PREFIX}nearest_boundary_distance": reference.nearest_boundary_distance.to_numpy(dtype=float),
        f"{CERTAINTY_PREFIX}same_bar_conflict_flag": (conflict_score < 1.0).astype(np.int8), f"{CERTAINTY_PREFIX}path_completeness": completeness,
        f"{CERTAINTY_PREFIX}entry_delay_sensitivity": delay_sensitivity, f"{CERTAINTY_PREFIX}atr_sensitivity": atr_sensitivity,
        f"{CERTAINTY_PREFIX}score": certainty, f"{CERTAINTY_PREFIX}weight_c1": 0.5 + 0.5 * certainty, f"{CERTAINTY_PREFIX}weight_c2": 0.25 + 0.75 * certainty,
        f"{CERTAINTY_PREFIX}consensus_target": matrix.mean(axis=1), f"{CERTAINTY_PREFIX}training_only": np.ones(rows, dtype=np.int8),
    })
