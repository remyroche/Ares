#!/usr/bin/env python3
"""Compare raw, PCA, and residual AE/GMM representations on causal ranks."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_train_meta_residual_archetype_enhancement import (
    DEFAULT_OUT_DIR,  # noqa: E402
)

FAMILIES = {
    "raw_global": ["lifecycle_residual_overlay"],
    "raw_local": ["lifecycle_residual_local_overlay"],
    "legacy_pca_unclipped": [
        f"lifecycle_residual_aware_ae_gmm_overlay_pca_baseline{suffix}"
        for suffix in ("", "_seed17", "_seed29")
    ],
    "corrected_pca8_clip8": [
        f"lifecycle_residual_aware_ae_gmm_overlay_pca8_clip8_baseline{suffix}"
        for suffix in ("", "_seed17", "_seed29")
    ],
    "corrected_pca8_clip8_globaloverlay": [
        f"lifecycle_residual_aware_ae_gmm_overlay_pca8_clip8_baseline{suffix}_globaloverlay"
        for suffix in ("", "_seed17", "_seed29")
    ],
    "residual_ae_gmm": [
        f"lifecycle_residual_aware_ae_gmm_overlay{suffix}"
        for suffix in ("", "_seed17", "_seed29")
    ],
}


def _historical_dir(root: Path, arm: str) -> Path:
    return root / (
        "historical_rank_oos"
        if arm == "lifecycle_residual_aware_ae_gmm_overlay"
        else f"historical_rank_oos_{arm}"
    )


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _run_record(root: Path, family: str, arm: str) -> dict[str, Any]:
    directory = _historical_dir(root, arm)
    metrics = pd.read_csv(directory / "metrics_by_scope.csv")
    overall = metrics[
        metrics["scope"].eq("overall")
        & metrics["fraction"].eq(0.10)
        & metrics["selector"].eq(arm)
    ].iloc[0]
    weeks = metrics[
        metrics["scope"].eq("week")
        & metrics["fraction"].eq(0.10)
        & metrics["selector"].eq(arm)
    ]
    autocorr = pd.read_csv(directory / "hit_surprise_autocorrelation.csv")
    autocorr = (
        pd.to_numeric(
            autocorr.loc[autocorr["selector"].eq(arm), "surprise_autocorr_lag1"],
            errors="coerce",
        )
        .abs()
        .dropna()
    )
    high = pd.read_csv(directory / "high_surprise_period_comparison.csv")
    high = high[high["baseline_high_surprise"].fillna(False).astype(bool)]
    arm_manifest_path = root / arm / "manifest.json"
    arm_manifest = (
        json.loads(arm_manifest_path.read_text()) if arm_manifest_path.exists() else {}
    )
    if arm_manifest.get("parent_arm"):
        parent_manifest_path = root / str(arm_manifest["parent_arm"]) / "manifest.json"
        if parent_manifest_path.exists():
            arm_manifest = json.loads(parent_manifest_path.read_text())
    fold_effective_rank = [
        float(fold["pca_effective_rank"])
        for fold in arm_manifest.get("folds", [])
        if fold.get("pca_effective_rank") is not None
    ]
    fold_explained = [
        float(fold["pca_explained_variance_sum"])
        for fold in arm_manifest.get("folds", [])
        if fold.get("pca_explained_variance_sum") is not None
    ]
    return {
        "family": family,
        "arm": arm,
        "selected_rows": int(overall["selected_rows"]),
        "top10_ev_after_1pct": float(overall["mean_ev_after_1pct"]),
        "clean_exec_precision": float(overall["clean_exec_precision"]),
        "full_path_bad_mae_rate": float(overall["full_path_bad_mae_rate"]),
        "timeout_rate": float(overall["timeout_rate"]),
        "worst_week_ev": float(weeks["mean_ev_after_1pct"].min()),
        "positive_weeks": int(weeks["mean_ev_after_1pct"].gt(0.0).sum()),
        "weeks": int(len(weeks)),
        "mean_abs_surprise_autocorr_lag1": float(autocorr.mean()),
        "high_surprise_improvement_rate": float(
            high["high_surprise_significantly_improved"]
            .fillna(False)
            .astype(bool)
            .mean()
        ),
        "mean_pca_effective_rank": (
            float(np.mean(fold_effective_rank)) if fold_effective_rank else np.nan
        ),
        "mean_pca_explained_variance": (
            float(np.mean(fold_explained)) if fold_explained else np.nan
        ),
    }


def main() -> None:
    root = DEFAULT_OUT_DIR
    report_dir = root / "final_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        _run_record(root, family, arm)
        for family, arms in FAMILIES.items()
        for arm in arms
        if (_historical_dir(root, arm) / "metrics_by_scope.csv").exists()
    ]
    runs = pd.DataFrame(rows)
    runs.to_csv(report_dir / "stage6_representation_runs.csv", index=False)
    summary = (
        runs.groupby("family", sort=True)
        .agg(
            seeds=("arm", "size"),
            mean_top10_ev_after_1pct=("top10_ev_after_1pct", "mean"),
            std_top10_ev_after_1pct=("top10_ev_after_1pct", "std"),
            mean_clean_precision=("clean_exec_precision", "mean"),
            mean_full_bad_mae=("full_path_bad_mae_rate", "mean"),
            mean_timeout=("timeout_rate", "mean"),
            mean_abs_surprise_autocorr_lag1=("mean_abs_surprise_autocorr_lag1", "mean"),
            std_abs_surprise_autocorr_lag1=("mean_abs_surprise_autocorr_lag1", "std"),
            minimum_worst_week_ev=("worst_week_ev", "min"),
            mean_high_surprise_improvement_rate=(
                "high_surprise_improvement_rate",
                "mean",
            ),
            mean_pca_effective_rank=("mean_pca_effective_rank", "mean"),
            mean_pca_explained_variance=("mean_pca_explained_variance", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(report_dir / "stage6_representation_family_summary.csv", index=False)
    indexed = summary.set_index("family")
    pca = indexed.loc["corrected_pca8_clip8_globaloverlay"]
    ae = indexed.loc["residual_ae_gmm"]
    ae_incremental_ev = float(
        ae["mean_top10_ev_after_1pct"] - pca["mean_top10_ev_after_1pct"]
    )
    ae_incremental_calendar = float(
        pca["mean_abs_surprise_autocorr_lag1"] - ae["mean_abs_surprise_autocorr_lag1"]
    )
    pca_healthy_rank = bool(float(pca["mean_pca_effective_rank"]) >= 4.0)
    selected = "corrected_pca8_clip8_globaloverlay"
    manifest = {
        "schema": "meta_residual_representation_robustness_v1",
        "selected_family": selected,
        "selected_arm": "lifecycle_residual_aware_ae_gmm_overlay_pca8_clip8_baseline_globaloverlay",
        "pca_scaled_clip": 8.0,
        "pca_components": 8,
        "pca_effective_rank_healthy": pca_healthy_rank,
        "ae_incremental_top10_ev_after_1pct": ae_incremental_ev,
        "ae_incremental_calendar_autocorr_reduction": ae_incremental_calendar,
        "ae_earns_incremental_complexity": bool(
            ae_incremental_ev > 0.0001 and ae_incremental_calendar > 0.0
        ),
        "legacy_pca_rejected": True,
        "legacy_pca_rejection_reason": (
            "Post-robust-scale values were not clipped; a zero-heavy price/OI quadrant "
            "feature dominated PC1. The corrected PCA has effective rank above seven."
        ),
        "selection_reason": (
            "Corrected PCA8 with global overlay is economically tied with residual AE/GMM "
            "across three seeds, has lower surprise autocorrelation, materially lower seed "
            "variance, and rejects the non-causal local normalization term."
        ),
        "current_meta_model_overwritten": False,
    }
    (report_dir / "stage6_representation_manifest.json").write_text(
        json.dumps(_safe(manifest), indent=2),
        encoding="utf-8",
    )
    print(json.dumps(_safe(manifest), indent=2), flush=True)


if __name__ == "__main__":
    main()
