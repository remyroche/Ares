"""Inference-safe sparse market-shock overlay for the residual meta model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .meta_residual_archetypes import OUTCOME_COLUMNS, REFERENCE_DERIVED_COLUMNS

MARKET_SHOCK_COMPONENTS: dict[str, float] = {
    "mkt_median_oi_chg_1h_rz": -1.0,
    "mkt_median_oi_chg_4h_rz": -1.0,
    "mkt_median_oi_drawdown_from_peak_24h": -1.0,
    "mkt_pct_oi_chg_4h_rz_lt_minus2": 1.0,
    "mkt_oi_flush_breadth_accel_1h": 1.0,
    "mkt_systemic_deleveraging_score": 1.0,
    "mkt_pct_price_up_oi_down_1h": 1.0,
}


@dataclass
class ResidualShockOverlayState:
    """Frozen percentile references and side/archetype support multipliers."""

    references: dict[str, np.ndarray]
    archetype_multipliers: dict[str, float]
    train_end: str
    components: dict[str, float] = field(
        default_factory=lambda: dict(MARKET_SHOCK_COMPONENTS)
    )

    def required_features(self) -> list[str]:
        return list(self.components)

    def _validate(self, frame: pd.DataFrame) -> None:
        forbidden = sorted(
            (OUTCOME_COLUMNS | REFERENCE_DERIVED_COLUMNS).intersection(frame.columns)
        )
        if forbidden:
            raise ValueError(f"Shock overlay received outcomes: {forbidden[:12]}")
        missing = [name for name in self.components if name not in frame.columns]
        if missing:
            raise ValueError(f"Shock overlay is missing features: {missing[:12]}")

    def transform_raw(self, frame: pd.DataFrame) -> np.ndarray:
        self._validate(frame)
        percentiles: list[np.ndarray] = []
        for name, direction in self.components.items():
            query = pd.to_numeric(frame[name], errors="coerce").to_numpy(
                dtype=np.float32
            )
            query = np.float32(direction) * query
            reference = np.asarray(self.references[name], dtype=np.float32)
            finite = np.isfinite(query)
            pct = np.full(len(frame), 0.5, dtype=np.float32)
            pct[finite] = np.searchsorted(reference, query[finite], side="right") / max(
                len(reference), 1
            )
            percentiles.append(pct)
        matrix = np.column_stack(percentiles).astype(np.float32, copy=False)
        flush = matrix[:, :-1].mean(axis=1)
        rebound = matrix[:, -1]
        return np.sqrt(np.clip(flush * rebound, 0.0, 1.0)).astype(np.float32)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        composite = self.transform_raw(frame)
        side = (
            frame.get("side_name", pd.Series("missing", index=frame.index))
            .astype(str)
            .str.lower()
        )
        arch = frame.get(
            "archetype_policy_key", pd.Series("missing", index=frame.index)
        ).astype(str)
        multiplier = np.fromiter(
            (
                self.archetype_multipliers.get(
                    f"{side_value}||{arch_value}",
                    self.archetype_multipliers.get(f"{side_value}||*", 1.0),
                )
                for side_value, arch_value in zip(side, arch)
            ),
            dtype=np.float32,
            count=len(frame),
        )
        return np.clip(composite * multiplier, 0.0, 1.0).astype(np.float32, copy=False)

    def adjust_scores(
        self,
        frame: pd.DataFrame,
        base_scores: Any,
        side_parameters: Mapping[str, Mapping[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raw = self.transform_raw(frame)
        local = self.transform(frame)
        adjusted = np.asarray(base_scores, dtype=np.float32).reshape(-1).copy()
        if len(adjusted) != len(frame):
            raise ValueError(
                f"Shock score length {len(adjusted)} != frame length {len(frame)}"
            )
        side = (
            frame.get("side_name", pd.Series("missing", index=frame.index))
            .astype(str)
            .str.lower()
        )
        for side_key, params in side_parameters.items():
            mask = side.eq(str(side_key).lower()).to_numpy()
            if not mask.any():
                continue
            source = raw if str(params.get("variant", "raw")) == "raw" else local
            threshold = float(params.get("threshold", 1.0))
            alpha = float(params.get("alpha", 0.0))
            intensity = np.clip(
                (source[mask] - threshold) / max(1.0 - threshold, 1e-3), 0.0, 1.0
            )
            adjusted[mask] -= np.float32(alpha) * intensity.astype(np.float32)
        return np.clip(adjusted, 0.0, 1.0), raw, local

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "meta_residual_shock_overlay_state_v1",
            "train_end": self.train_end,
            "components": dict(self.components),
            "reference_rows": {
                name: int(len(values)) for name, values in self.references.items()
            },
            "archetype_multiplier_count": int(len(self.archetype_multipliers)),
            "leakage_contract": (
                "References and multipliers are fitted on prior rows only; transform rejects realized "
                "outcomes and uses only market-wide pre-entry features plus side/archetype identity."
            ),
        }
