"""Frozen, inference-safe mechanism families for residual-state composites."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


FAMILY_NAMES = (
    "correlation_fragmentation",
    "participation_failure",
    "liquidity_dislocation",
    "deleveraging",
    "leverage_rebuild",
    "liquidation_exhaustion",
    "recovery_failure",
    "volcompression_failure",
)


def mechanism_family(base_feature: str, gate_feature: str) -> str:
    text = f"{base_feature} {gate_feature}".lower()
    if "compression" in text:
        return "volcompression_failure"
    if any(token in text for token in ("recovery", "followthrough", "range_climax")):
        return "recovery_failure"
    if any(token in text for token in ("rebuild", "covering")):
        return "leverage_rebuild"
    if any(token in text for token in ("exhaustion", "flush")):
        return "liquidation_exhaustion"
    if any(token in text for token in ("deleveraging", "oi_dominance")):
        return "deleveraging"
    if any(token in text for token in ("breadth", "new_low", "breakout")):
        return "participation_failure"
    if any(token in text for token in ("correlation", "decoupling", "dispersion")):
        return "correlation_fragmentation"
    return "liquidity_dislocation"


def _location_scale(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size < 20:
        return 0.0, 1.0
    location = float(np.median(finite))
    mad = float(np.median(np.abs(finite - location)))
    scale = max(1.4826 * mad, float(np.std(finite)) * 0.10, 1e-8)
    return location, scale


def _scaled(values: np.ndarray, location: float, scale: float) -> np.ndarray:
    return np.clip((values - location) / max(scale, 1e-8), -5.0, 5.0)


def _score(a: np.ndarray, b: np.ndarray, form: str, gate: float) -> np.ndarray:
    ap = np.maximum(a, 0.0)
    if form == "positive":
        raw = ap * np.maximum(b, 0.0)
    elif form == "negative":
        raw = ap * np.maximum(-b, 0.0)
    elif form == "threshold":
        raw = ap * (b > gate)
    elif form == "contrast":
        raw = ap * np.maximum(b - a, 0.0)
    else:
        raise ValueError(f"Unsupported residual-state form: {form}")
    return np.arcsinh(raw).astype(np.float32)


@dataclass(frozen=True)
class ResidualStateDefinition:
    side_name: str
    archetype_policy_key: str
    family: str
    base_feature: str
    gate_feature: str
    form: str
    base_location: float
    base_scale: float
    gate_location: float
    gate_scale: float
    gate_direction: float
    gate_threshold: float
    percentile_knots: tuple[float, ...]
    weight: float
    status: str

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        a = _scaled(
            pd.to_numeric(frame[self.base_feature], errors="coerce").to_numpy(float),
            self.base_location,
            self.base_scale,
        )
        b = self.gate_direction * _scaled(
            pd.to_numeric(frame[self.gate_feature], errors="coerce").to_numpy(float),
            self.gate_location,
            self.gate_scale,
        )
        raw = _score(a, b, self.form, self.gate_threshold)
        knots = np.asarray(self.percentile_knots, dtype=np.float32)
        quantiles = np.linspace(0.0, 1.0, len(knots), dtype=np.float32)
        return np.interp(raw, knots, quantiles, left=0.0, right=1.0).astype(np.float32)


@dataclass(frozen=True)
class ResidualStateFamilyContract:
    schema_version: int
    definitions: tuple[ResidualStateDefinition, ...]
    source_feature_contract_hash: str
    fit_end: str
    contract_hash: str = ""

    def with_hash(self) -> "ResidualStateFamilyContract":
        payload = self.to_dict(include_hash=False)
        stable = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return ResidualStateFamilyContract(
            schema_version=self.schema_version,
            definitions=self.definitions,
            source_feature_contract_hash=self.source_feature_contract_hash,
            fit_end=self.fit_end,
            contract_hash="sha256:" + sha256(stable.encode()).hexdigest(),
        )

    def to_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "source_feature_contract_hash": self.source_feature_contract_hash,
            "fit_end": self.fit_end,
            "definitions": [asdict(definition) for definition in self.definitions],
        }
        if include_hash:
            payload["contract_hash"] = self.contract_hash
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ResidualStateFamilyContract":
        definitions = tuple(
            ResidualStateDefinition(
                **{
                    **definition,
                    "percentile_knots": tuple(definition["percentile_knots"]),
                    "status": str(definition.get("status", "discovery_only")),
                }
            )
            for definition in payload.get("definitions", [])
        )
        contract = cls(
            schema_version=int(payload["schema_version"]),
            definitions=definitions,
            source_feature_contract_hash=str(payload["source_feature_contract_hash"]),
            fit_end=str(payload["fit_end"]),
            contract_hash=str(payload.get("contract_hash", "")),
        )
        expected = contract.with_hash().contract_hash
        if contract.contract_hash and contract.contract_hash != expected:
            raise ValueError("Residual-state family contract hash mismatch")
        return contract

    def transform(
        self,
        frame: pd.DataFrame,
        side: Iterable[object],
        archetype: Iterable[object],
    ) -> pd.DataFrame:
        side_values = np.asarray(list(side), dtype=str)
        archetype_values = np.asarray(list(archetype), dtype=str)
        output = pd.DataFrame(index=frame.index)
        family_values = {
            family: np.zeros(len(frame), dtype=np.float32) for family in FAMILY_NAMES
        }
        family_weights = {
            family: np.zeros(len(frame), dtype=np.float32) for family in FAMILY_NAMES
        }
        family_active = {
            family: np.zeros(len(frame), dtype=np.float32) for family in FAMILY_NAMES
        }
        family_computable = {
            family: np.zeros(len(frame), dtype=np.float32) for family in FAMILY_NAMES
        }
        for definition in self.definitions:
            mask = (side_values == definition.side_name) & (
                archetype_values == definition.archetype_policy_key
            )
            if not mask.any():
                continue
            values = definition.transform(frame)
            computable = (
                pd.to_numeric(frame[definition.base_feature], errors="coerce").notna().to_numpy()
                & pd.to_numeric(frame[definition.gate_feature], errors="coerce").notna().to_numpy()
            )
            family_values[definition.family][mask] += definition.weight * values[mask]
            family_weights[definition.family][mask] += abs(definition.weight)
            family_active[definition.family][mask] = 1.0
            family_computable[definition.family][mask & computable] = 1.0
        for family in FAMILY_NAMES:
            denominator = family_weights[family]
            values = np.divide(
                family_values[family],
                denominator,
                out=np.zeros(len(frame), dtype=np.float32),
                where=denominator > 0,
            )
            output[f"residual_state_family_{family}_pct"] = values
            output[f"residual_state_family_{family}_active"] = family_active[family]
            output[f"residual_state_family_{family}_computable"] = family_computable[family]
        score_columns = [
            f"residual_state_family_{family}_pct" for family in FAMILY_NAMES
        ]
        output["residual_state_family_gated_composite_max_pct"] = output[
            score_columns
        ].max(axis=1)
        return output.astype(np.float32)


def fit_definition(
    frame: pd.DataFrame,
    target: np.ndarray,
    row: Mapping[str, object],
) -> ResidualStateDefinition:
    a_raw = pd.to_numeric(frame[str(row["base_feature"])], errors="coerce").to_numpy(float)
    b_raw = pd.to_numeric(frame[str(row["gate_feature"])], errors="coerce").to_numpy(float)
    a_location, a_scale = _location_scale(a_raw)
    b_location, b_scale = _location_scale(b_raw)
    a = _scaled(a_raw, a_location, a_scale)
    b = _scaled(b_raw, b_location, b_scale)
    broad = a >= np.nanquantile(a, 0.80)
    adverse = b[broad & target]
    benign = b[broad & ~target]
    direction = 1.0 if len(adverse) and len(benign) and np.nanmedian(adverse) >= np.nanmedian(benign) else -1.0
    b *= direction
    gate = float(np.nanquantile(b[broad], 0.75)) if broad.any() else 0.0
    raw = _score(a, b, str(row["form"]), gate)
    knots = tuple(float(value) for value in np.nanquantile(raw, np.linspace(0, 1, 101)))
    lift_q25 = max(float(row.get("lift_q25", row.get("mean_lift", 1.0))), 1e-8)
    fpr_q75 = np.clip(float(row.get("fpr_q75", row.get("mean_fpr", 1.0))), 0.0, 1.0)
    stability = max(float(row.get("fold_stability", 1.0)), 0.0)
    support = max(float(row.get("adverse_support", 0.0)), 0.0)
    weight = (
        np.log1p(lift_q25)
        * (1.0 - fpr_q75)
        * stability
        * (support / (support + 10.0))
    )
    return ResidualStateDefinition(
        side_name=str(row["side_name"]),
        archetype_policy_key=str(row["archetype_policy_key"]),
        family=mechanism_family(str(row["base_feature"]), str(row["gate_feature"])),
        base_feature=str(row["base_feature"]),
        gate_feature=str(row["gate_feature"]),
        form=str(row["form"]),
        base_location=a_location,
        base_scale=a_scale,
        gate_location=b_location,
        gate_scale=b_scale,
        gate_direction=direction,
        gate_threshold=gate,
        percentile_knots=knots,
        weight=float(weight),
        status=str(row.get("status", "discovery_only")),
    )
