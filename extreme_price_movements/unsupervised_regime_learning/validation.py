"""Validation helpers for unsupervised regime-learning artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from extreme_price_movements.unsupervised_regime_learning.regime_models import (
    AdvancedRegimeLearningArtifact,
)


@dataclass(frozen=True)
class RegimePipelineValidationConfig:
    min_top_total_score: float = 0.05
    min_top_useful_regime_score: float = 0.0
    min_top_min_support: float = 0.01
    min_effective_regime_count: float = 1.25
    min_model_feature_finite_fraction: float = 0.95
    require_final_feature_input_for_ae_mfa: bool = True


def _float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _row_by_step(steps: pd.DataFrame, step: str) -> Mapping[str, Any]:
    if not isinstance(steps, pd.DataFrame) or steps.empty or "step" not in steps.columns:
        return {}
    rows = steps.loc[steps["step"].astype(str).eq(str(step))]
    if rows.empty:
        return {}
    return rows.iloc[-1].to_dict()


def validate_regime_learning_artifact(
    artifact: AdvancedRegimeLearningArtifact,
    *,
    config: RegimePipelineValidationConfig = RegimePipelineValidationConfig(),
) -> pd.DataFrame:
    """Return per-step validation checks for a fitted regime-learning artifact.

    The checks are deliberately bounded and descriptive. They are intended for
    pipeline QA and artifact manifests, not for trading-label model selection.
    """

    rows: list[dict[str, Any]] = []
    steps = getattr(artifact, "pipeline_steps", pd.DataFrame())
    diag = getattr(artifact, "regime_diagnostics", pd.DataFrame())
    model_features = getattr(artifact, "model_regime_features", pd.DataFrame())
    required_steps = [
        "01_matrix_scaling",
        "02_real_vs_null_stability_selection",
        "03_leaf_and_raw_embeddings",
        "04_autoencoder_latents",
        "05_mixture_factor_analyzers",
        "06_regime_discovery_assessment",
        "07_regime_feature_generation",
        "08_model_regime_feature_package",
    ]
    present = set(steps["step"].astype(str)) if isinstance(steps, pd.DataFrame) and "step" in steps.columns else set()
    for step in required_steps:
        rows.append(
            {
                "step": step,
                "check": "step_present",
                "passed": step in present,
                "value": 1.0 if step in present else 0.0,
                "threshold": 1.0,
                "message": "present" if step in present else "missing",
            }
        )
    matrix_step = _row_by_step(steps, "01_matrix_scaling")
    usable = _float(matrix_step.get("usable_feature_count"), 0.0)
    rows.append(
        {
            "step": "01_matrix_scaling",
            "check": "usable_features",
            "passed": usable > 0.0,
            "value": usable,
            "threshold": 1.0,
            "message": f"usable_feature_count={usable:.0f}",
        }
    )
    selector_step = _row_by_step(steps, "02_real_vs_null_stability_selection")
    selected = _float(selector_step.get("selected_feature_count"), 0.0)
    rows.append(
        {
            "step": "02_real_vs_null_stability_selection",
            "check": "selected_features",
            "passed": selected > 0.0,
            "value": selected,
            "threshold": 1.0,
            "message": f"selected_feature_count={selected:.0f}",
        }
    )
    ae_step = _row_by_step(steps, "04_autoencoder_latents")
    ae_ok = (
        str(ae_step.get("sparse_input_source", "")) == "05_final_regime_learning_feature_set"
        and str(ae_step.get("contrastive_input_source", "")) == "05_final_regime_learning_feature_set"
    )
    rows.append(
        {
            "step": "04_autoencoder_latents",
            "check": "uses_final_feature_set",
            "passed": bool(ae_ok) if config.require_final_feature_input_for_ae_mfa else True,
            "value": 1.0 if ae_ok else 0.0,
            "threshold": 1.0,
            "message": (
                "sparse/contrastive AE use final feature set"
                if ae_ok
                else "sparse/contrastive AE input source is not final feature set"
            ),
        }
    )
    mfa_step = _row_by_step(steps, "05_mixture_factor_analyzers")
    mfa_ok = str(mfa_step.get("input_source", "")) == "05_final_regime_learning_feature_set"
    rows.append(
        {
            "step": "05_mixture_factor_analyzers",
            "check": "uses_final_feature_set",
            "passed": bool(mfa_ok) if config.require_final_feature_input_for_ae_mfa else True,
            "value": 1.0 if mfa_ok else 0.0,
            "threshold": 1.0,
            "message": "MFA uses final feature set" if mfa_ok else "MFA input source is not final feature set",
        }
    )
    if isinstance(diag, pd.DataFrame) and not diag.empty:
        total = pd.to_numeric(diag.get("TotalScore", pd.Series(dtype=float)), errors="coerce")
        top_score = float(total.max()) if total.notna().any() else 0.0
        rows.append(
            {
                "step": "06_regime_discovery_assessment",
                "check": "top_total_score",
                "passed": top_score >= float(config.min_top_total_score),
                "value": top_score,
                "threshold": float(config.min_top_total_score),
                "message": f"top_total_score={top_score:.4f}",
            }
        )
        useful = pd.to_numeric(diag.get("UsefulRegimeScore", pd.Series(dtype=float)), errors="coerce")
        if useful.notna().any():
            top_useful = float(useful.max())
            rows.append(
                {
                    "step": "06_regime_discovery_assessment",
                    "check": "top_useful_regime_score",
                    "passed": top_useful >= float(config.min_top_useful_regime_score),
                    "value": top_useful,
                    "threshold": float(config.min_top_useful_regime_score),
                    "message": f"top_useful_regime_score={top_useful:.4f}",
                }
            )
            top_idx = useful.idxmax()
        else:
            top_idx = total.idxmax() if total.notna().any() else None
        if top_idx is not None:
            top = diag.loc[top_idx]
            min_support = _float(top.get("min_support"), 0.0)
            regime_count = _float(top.get("regime_count"), 0.0)
            max_support = max(_float(top.get("label_max_support"), 0.0), min_support)
            effective = 1.0 / max(max_support, 1e-12) if max_support > 0.0 else regime_count
            rows.append(
                {
                    "step": "06_regime_discovery_assessment",
                    "check": "minimum_regime_support",
                    "passed": min_support >= float(config.min_top_min_support),
                    "value": min_support,
                    "threshold": float(config.min_top_min_support),
                    "message": f"top_min_support={min_support:.4f}",
                }
            )
            rows.append(
                {
                    "step": "06_regime_discovery_assessment",
                    "check": "effective_regime_count",
                    "passed": effective >= float(config.min_effective_regime_count),
                    "value": effective,
                    "threshold": float(config.min_effective_regime_count),
                    "message": f"effective_regime_count_proxy={effective:.4f}",
                }
            )
    else:
        rows.append(
            {
                "step": "06_regime_discovery_assessment",
                "check": "diagnostics_present",
                "passed": False,
                "value": 0.0,
                "threshold": 1.0,
                "message": "regime_diagnostics is empty",
            }
        )
    if isinstance(model_features, pd.DataFrame) and not model_features.empty:
        finite_fraction = float(
            np.isfinite(model_features.to_numpy(dtype=np.float32, copy=False)).mean()
        )
    else:
        finite_fraction = 0.0
    rows.append(
        {
            "step": "08_model_regime_feature_package",
            "check": "model_feature_finite_fraction",
            "passed": finite_fraction >= float(config.min_model_feature_finite_fraction),
            "value": finite_fraction,
            "threshold": float(config.min_model_feature_finite_fraction),
            "message": f"finite_fraction={finite_fraction:.4f}",
        }
    )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["passed"] = out["passed"].astype(bool)
    return out


def regime_pipeline_validation_summary(report: pd.DataFrame) -> dict[str, Any]:
    if not isinstance(report, pd.DataFrame) or report.empty:
        return {"passed": False, "check_count": 0, "failed_count": 0}
    passed = report["passed"].astype(bool) if "passed" in report.columns else pd.Series(False, index=report.index)
    failed = report.loc[~passed]
    return {
        "passed": bool(passed.all()),
        "check_count": int(len(report)),
        "failed_count": int((~passed).sum()),
        "failed_checks": failed[["step", "check", "message"]].to_dict("records")
        if {"step", "check", "message"}.issubset(failed.columns)
        else [],
    }
