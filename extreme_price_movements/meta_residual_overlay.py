"""Frozen, inference-safe score overlay for residual meta archetypes.

The residual recognizer emits causal train-prior features.  This module only
normalizes those outputs with statistics fitted on a prior calibration period
and combines them with an already-computed meta score.  Realized outcomes are
neither required nor accepted by the transform.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .meta_residual_archetypes import OUTCOME_COLUMNS, REFERENCE_DERIVED_COLUMNS

HIT_FEATURE = "meta_resid_arch_expected_hit_surprise"
DIRTY_FEATURE = "meta_resid_arch_expected_dirty_positive"


def _safe_key(side: Any, archetype: Any) -> str:
    return f"{str(side).strip().lower()}||{str(archetype).strip()}"


def _finite_mean_std(values: pd.Series, *, min_std: float) -> tuple[float, float]:
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0, 1.0
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=0))
    return mean, max(std if np.isfinite(std) else 0.0, float(min_std))


@dataclass(frozen=True)
class ResidualOverlayStats:
    hit_mean: float
    hit_std: float
    dirty_mean: float
    dirty_std: float
    rows: int


@dataclass
class ResidualOverlayState:
    """Parameters and train-fitted normalization for a residual score overlay."""

    hit_alpha: float = 0.0
    dirty_lambda: float = 0.0
    local_hit_alpha: float = 0.0
    local_dirty_lambda: float = 0.0
    side_col: str = "side_name"
    archetype_col: str = "archetype_policy_key"
    hit_feature: str = HIT_FEATURE
    dirty_feature: str = DIRTY_FEATURE
    min_std: float = 0.02
    z_clip: float = 3.0
    calibration_start: str | None = None
    calibration_end: str | None = None
    group_stats: dict[str, ResidualOverlayStats] = field(default_factory=dict)
    side_stats: dict[str, ResidualOverlayStats] = field(default_factory=dict)
    global_stats: ResidualOverlayStats = field(
        default_factory=lambda: ResidualOverlayStats(0.0, 1.0, 0.0, 1.0, 0)
    )

    def fit_normalization(self, frame: pd.DataFrame) -> "ResidualOverlayState":
        missing = [
            name
            for name in (self.hit_feature, self.dirty_feature)
            if name not in frame.columns
        ]
        if missing:
            raise ValueError(
                f"Residual overlay calibration is missing features: {missing}"
            )
        side = (
            frame.get(self.side_col, pd.Series("missing", index=frame.index))
            .astype(str)
            .str.lower()
        )
        arch = frame.get(
            self.archetype_col, pd.Series("missing", index=frame.index)
        ).astype(str)
        work = pd.DataFrame(
            {
                "side": side,
                "arch": arch,
                "hit": pd.to_numeric(frame[self.hit_feature], errors="coerce"),
                "dirty": pd.to_numeric(frame[self.dirty_feature], errors="coerce"),
            },
            index=frame.index,
        )
        self.group_stats = {}
        self.side_stats = {}
        for (side_key, arch_key), group in work.groupby(
            ["side", "arch"], sort=True, dropna=False
        ):
            hm, hs = _finite_mean_std(group["hit"], min_std=self.min_std)
            dm, ds = _finite_mean_std(group["dirty"], min_std=self.min_std)
            self.group_stats[_safe_key(side_key, arch_key)] = ResidualOverlayStats(
                hm, hs, dm, ds, len(group)
            )
        for side_key, group in work.groupby("side", sort=True, dropna=False):
            hm, hs = _finite_mean_std(group["hit"], min_std=self.min_std)
            dm, ds = _finite_mean_std(group["dirty"], min_std=self.min_std)
            self.side_stats[str(side_key).lower()] = ResidualOverlayStats(
                hm, hs, dm, ds, len(group)
            )
        hm, hs = _finite_mean_std(work["hit"], min_std=self.min_std)
        dm, ds = _finite_mean_std(work["dirty"], min_std=self.min_std)
        self.global_stats = ResidualOverlayStats(hm, hs, dm, ds, len(work))
        if "__ts__" in frame.columns:
            ts = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
            self.calibration_start = str(ts.min())
            self.calibration_end = str(ts.max())
        return self

    def _stats_arrays(
        self, frame: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        side = (
            frame.get(self.side_col, pd.Series("missing", index=frame.index))
            .astype(str)
            .str.lower()
        )
        arch = frame.get(
            self.archetype_col, pd.Series("missing", index=frame.index)
        ).astype(str)
        h_mean = np.empty(len(frame), dtype=np.float32)
        h_std = np.empty(len(frame), dtype=np.float32)
        d_mean = np.empty(len(frame), dtype=np.float32)
        d_std = np.empty(len(frame), dtype=np.float32)
        for pos, (side_value, arch_value) in enumerate(zip(side, arch)):
            stats = self.group_stats.get(_safe_key(side_value, arch_value))
            if stats is None:
                stats = self.side_stats.get(str(side_value).lower(), self.global_stats)
            h_mean[pos] = np.float32(stats.hit_mean)
            h_std[pos] = np.float32(max(stats.hit_std, self.min_std))
            d_mean[pos] = np.float32(stats.dirty_mean)
            d_std[pos] = np.float32(max(stats.dirty_std, self.min_std))
        return h_mean, h_std, d_mean, d_std

    def transform(self, frame: pd.DataFrame, base_scores: Any) -> np.ndarray:
        forbidden = sorted(
            (OUTCOME_COLUMNS | REFERENCE_DERIVED_COLUMNS).intersection(frame.columns)
        )
        if forbidden:
            raise ValueError(
                f"Residual overlay transform received outcomes: {forbidden[:12]}"
            )
        missing = [
            name
            for name in (self.hit_feature, self.dirty_feature)
            if name not in frame.columns
        ]
        if missing:
            raise ValueError(
                f"Residual overlay transform is missing features: {missing}"
            )
        base = np.asarray(base_scores, dtype=np.float32).reshape(-1)
        if len(base) != len(frame):
            raise ValueError(
                f"base score length {len(base)} != frame length {len(frame)}"
            )
        hit = pd.to_numeric(frame[self.hit_feature], errors="coerce").to_numpy(
            dtype=np.float32
        )
        dirty = pd.to_numeric(frame[self.dirty_feature], errors="coerce").to_numpy(
            dtype=np.float32
        )
        h_mean, h_std, d_mean, d_std = self._stats_arrays(frame)
        hit = np.where(np.isfinite(hit), hit, h_mean)
        dirty = np.where(np.isfinite(dirty), dirty, d_mean)
        hit_z = np.clip((hit - h_mean) / h_std, -self.z_clip, self.z_clip)
        dirty_z = np.clip((dirty - d_mean) / d_std, -self.z_clip, self.z_clip)
        score = (
            base
            + np.float32(self.hit_alpha) * hit
            - np.float32(self.dirty_lambda) * dirty
            + np.float32(self.local_hit_alpha) * hit_z
            - np.float32(self.local_dirty_lambda) * dirty_z
        )
        return np.clip(score, 0.0, 1.0).astype(np.float32, copy=False)

    def manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema"] = "meta_residual_overlay_state_v1"
        payload["generated_from_outcomes"] = False
        payload["inference_inputs"] = [
            self.side_col,
            self.archetype_col,
            self.hit_feature,
            self.dirty_feature,
            "base_meta_score",
        ]
        payload["leakage_contract"] = (
            "Normalization statistics and coefficients are frozen before evaluation; "
            "transform rejects realized outcomes and reference-derived labels."
        )
        return payload


def overlay_state_from_mapping(payload: Mapping[str, Any]) -> ResidualOverlayState:
    """Reconstruct a state from a JSON-compatible manifest."""

    data = dict(payload)
    data.pop("schema", None)
    data.pop("generated_from_outcomes", None)
    data.pop("inference_inputs", None)
    data.pop("leakage_contract", None)
    data["group_stats"] = {
        str(key): ResidualOverlayStats(**value) if isinstance(value, Mapping) else value
        for key, value in dict(data.get("group_stats", {})).items()
    }
    data["side_stats"] = {
        str(key): ResidualOverlayStats(**value) if isinstance(value, Mapping) else value
        for key, value in dict(data.get("side_stats", {})).items()
    }
    if isinstance(data.get("global_stats"), Mapping):
        data["global_stats"] = ResidualOverlayStats(**dict(data["global_stats"]))
    return ResidualOverlayState(**data)
