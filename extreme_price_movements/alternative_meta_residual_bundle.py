"""Inference bundle for the alternative lifecycle + residual-archetype meta model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .meta_historical_rank import HistoricalScoreRankReference
from .meta_residual_archetypes import (
    ResidualArchetypeRecognizer,
    strip_outcomes_for_oos,
)
from .meta_residual_overlay import ResidualOverlayState
from .meta_residual_shock_overlay import ResidualShockOverlayState


def _apply_ood_state(frame: pd.DataFrame, state: dict[str, Any]) -> pd.DataFrame:
    cols = [str(name) for name in state.get("columns", [])]
    out = frame.copy(deep=False)
    if not cols:
        return out
    values = out.reindex(columns=cols).to_numpy(dtype=np.float32, copy=True)
    finite = np.isfinite(values)
    mean = np.asarray(state["mean"], dtype=np.float32)
    std = np.asarray(state["std"], dtype=np.float32)
    q25 = np.asarray(state["q25"], dtype=np.float32)
    q75 = np.asarray(state["q75"], dtype=np.float32)
    iqr = np.maximum(q75 - q25, 1e-6)
    filled = np.where(finite, values, mean)
    z = (filled - mean) / std
    abs_z = np.abs(z)
    exceed = ((filled < q25 - 1.5 * iqr) | (filled > q75 + 1.5 * iqr)) & finite
    out = out.copy()
    out["meta_sel_ood_abs_z_mean"] = np.mean(abs_z, axis=1).astype(np.float32)
    out["meta_sel_ood_abs_z_max"] = np.max(abs_z, axis=1).astype(np.float32)
    out["meta_sel_ood_abs_z_p95"] = np.quantile(abs_z, 0.95, axis=1).astype(np.float32)
    out["meta_sel_ood_iqr_exceed_frac"] = np.mean(exceed, axis=1).astype(np.float32)
    out["meta_sel_ood_missing_frac"] = np.mean(~finite, axis=1).astype(np.float32)
    out["meta_sel_ood_centroid_l2"] = np.sqrt(np.mean(z * z, axis=1)).astype(np.float32)
    return out


def _apply_residual_representation(
    frame: pd.DataFrame,
    state: dict[str, Any] | None,
    *,
    batch_rows: int = 100_000,
) -> pd.DataFrame:
    """Apply an optional frozen pre-recognizer representation transform."""

    if not state:
        return frame
    kind = str(state.get("kind", ""))
    if kind != "robust_pca":
        raise ValueError(f"Unsupported residual representation: {kind!r}")
    columns = [str(name) for name in state.get("columns", [])]
    pca = state["pca"]
    output_columns = list(
        state.get(
            "output_columns",
            [f"meta_resid_pca_{idx:02d}" for idx in range(len(pca.components_))],
        )
    )
    medians = np.asarray(state["medians"], dtype=np.float32)
    low = np.asarray(state["low"], dtype=np.float32)
    high = np.asarray(state["high"], dtype=np.float32)
    scaled_clip = state.get("scaled_clip")
    output = np.empty((len(frame), len(output_columns)), dtype=np.float32)
    for start in range(0, len(frame), int(batch_rows)):
        stop = min(start + int(batch_rows), len(frame))
        values = (
            frame.iloc[start:stop]
            .reindex(columns=columns)
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float32)
        )
        values = np.where(np.isfinite(values), values, medians)
        values = np.clip(values, low, high)
        values = state["scaler"].transform(values).astype(np.float32)
        if scaled_clip is not None and float(scaled_clip) > 0.0:
            values = np.clip(values, -float(scaled_clip), float(scaled_clip)).astype(
                np.float32,
                copy=False,
            )
        output[start:stop] = pca.transform(values).astype(np.float32)
    out = frame.copy(deep=False)
    for idx, name in enumerate(output_columns):
        out[name] = output[:, idx]
    return out


@dataclass
class AlternativeMetaResidualBundle:
    """Frozen alternative model; it never refits any component at inference."""

    lifecycle_model: Any
    selected_features: list[str]
    raw_selected_features: list[str]
    feature_medians: dict[str, float]
    ood_state: dict[str, Any]
    residual_recognizer: ResidualArchetypeRecognizer
    overlay_state: ResidualOverlayState
    residual_representation_state: dict[str, Any] | None = None
    hit_calibrator: Any = None
    historical_rank_reference: HistoricalScoreRankReference | None = None
    shock_overlay_state: ResidualShockOverlayState | None = None
    shock_side_parameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    fit_through: str | None = None
    frozen_ae_gmm_sha256: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def required_input_features(self) -> list[str]:
        required = list(self.raw_selected_features)
        if self.residual_representation_state:
            required.extend(self.residual_representation_state.get("columns", []))
        for model in list(self.residual_recognizer.side_models.values()) + list(
            self.residual_recognizer.local_models.values()
        ):
            required.extend(model.feature_columns)
        required.extend(
            [
                self.residual_recognizer.config.side_col,
                self.residual_recognizer.config.archetype_col,
            ]
        )
        shock_state = getattr(self, "shock_overlay_state", None)
        if shock_state is not None:
            required.extend(shock_state.required_features())
        return list(dict.fromkeys(str(name) for name in required))

    def _lifecycle_matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        matrix = frame.reindex(columns=self.raw_selected_features).apply(
            pd.to_numeric, errors="coerce"
        )
        for name in self.raw_selected_features:
            matrix[name] = (
                matrix[name]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(float(self.feature_medians.get(name, 0.0)))
            )
        matrix = matrix.astype(np.float32)
        matrix = _apply_ood_state(matrix, self.ood_state)
        return matrix.reindex(columns=self.selected_features, fill_value=0.0).astype(
            np.float32
        )

    def predict(self, pre_entry_frame: pd.DataFrame) -> pd.DataFrame:
        safe = strip_outcomes_for_oos(pre_entry_frame)
        lifecycle_x = self._lifecycle_matrix(safe)
        lifecycle = np.asarray(
            self.lifecycle_model.predict(lifecycle_x), dtype=np.float32
        ).reshape(-1)
        residual_input = _apply_residual_representation(
            safe,
            self.residual_representation_state,
        )
        residual = self.residual_recognizer.transform_oos(residual_input)
        residual = residual.set_axis(safe.index)
        overlay_input = pd.concat([safe, residual], axis=1, copy=False)
        overlay = self.overlay_state.transform(overlay_input, lifecycle)
        shock_state = getattr(self, "shock_overlay_state", None)
        if shock_state is None:
            adjusted = overlay
            shock_raw = np.zeros(len(safe), dtype=np.float32)
            shock_local = np.zeros(len(safe), dtype=np.float32)
        else:
            adjusted, shock_raw, shock_local = shock_state.adjust_scores(
                safe,
                overlay,
                getattr(self, "shock_side_parameters", {}),
            )
        if self.hit_calibrator is None:
            hit_prob = np.clip(adjusted, 0.0, 1.0)
        else:
            hit_prob = self.hit_calibrator.predict_proba(adjusted.reshape(-1, 1))[:, 1]
        output = pd.DataFrame(
            {
                "score_lifecycle_only": lifecycle.astype(np.float32),
                "score_residual_overlay": overlay.astype(np.float32),
                "score_shock_adjusted": adjusted.astype(np.float32),
                "shock_composite_raw": shock_raw.astype(np.float32),
                "shock_composite_local": shock_local.astype(np.float32),
                "hit_probability": np.asarray(hit_prob, dtype=np.float32),
            },
            index=pre_entry_frame.index,
        )
        rank_reference = getattr(self, "historical_rank_reference", None)
        if rank_reference is not None:
            rank_frame = pd.DataFrame(index=safe.index)
            rank_frame[rank_reference.side_col] = safe.get(
                rank_reference.side_col,
                pd.Series("missing", index=safe.index),
            )
            rank_frame[rank_reference.score_col] = adjusted
            output["historical_rank"] = rank_reference.transform(rank_frame).to_numpy(
                dtype=np.float32,
                copy=False,
            )
        return output

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "alternative_meta_residual_bundle_v1",
            "fit_through": self.fit_through,
            "selected_feature_count": len(self.selected_features),
            "raw_selected_feature_count": len(self.raw_selected_features),
            "required_input_feature_count": len(self.required_input_features()),
            "frozen_ae_gmm_sha256": self.frozen_ae_gmm_sha256,
            "residual_representation": (
                {
                    "kind": self.residual_representation_state.get("kind"),
                    "input_feature_count": len(
                        self.residual_representation_state.get("columns", [])
                    ),
                    "output_feature_count": len(
                        self.residual_representation_state.get("output_columns", [])
                    ),
                    "scaled_clip": self.residual_representation_state.get(
                        "scaled_clip"
                    ),
                }
                if self.residual_representation_state
                else None
            ),
            "residual_recognizer": self.residual_recognizer.manifest(),
            "overlay": self.overlay_state.manifest(),
            "shock_overlay": (
                getattr(self, "shock_overlay_state", None).manifest()
                if getattr(self, "shock_overlay_state", None) is not None
                else None
            ),
            "shock_side_parameters": getattr(self, "shock_side_parameters", {}),
            "historical_rank": (
                getattr(self, "historical_rank_reference", None).manifest()
                if getattr(self, "historical_rank_reference", None) is not None
                else None
            ),
            "metadata": self.metadata,
            "leakage_contract": (
                "All model, OOD, recognizer, prior, overlay, calibration, and AE/GMM states are frozen; "
                "the historical rank uses a frozen prior-score empirical CDF; predict rejects outcome "
                "columns through the residual transform."
            ),
        }
