import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Literal, Dict
import json
import os

from extreme_price_movements.feature_select.scoring import UtilityConfig, FeatureSelectConfig
from extreme_price_movements.feature_select.cv import CVConfig
from extreme_price_movements.feature_select.rfe import run_rfe
from extreme_price_movements.utils import tprint

@dataclass
class FeatureSelectResult:
    selected_features: List[str]
    feature_scores: pd.DataFrame
    rfe_trace: pd.DataFrame


def run_feature_selection(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: Optional[np.ndarray],     # e.g. asset_id or day_id for blocking
    time_index: Optional[pd.Series],  # for time splits
    model_kind: Literal["binary", "regression", "quantile"],
    quantile_alpha: Optional[float],  # required if model_kind=="quantile"
    cv_config: CVConfig,
    lgbm_params: dict,
    utility_config: UtilityConfig,
    fs_config: FeatureSelectConfig,
    random_seed: int = 42,
    output_dir: str = "artifacts",
    max_samples: int = 8000,
) -> FeatureSelectResult:
    """
    Main entry point for LightGBM feature selection pipeline.
    """
    tprint("Starting Feature Selection Process...")

    selected_features, feature_scores, rfe_trace = run_rfe(
        X, y, groups, time_index, model_kind, quantile_alpha,
        cv_config, lgbm_params, utility_config, fs_config, random_seed,
        max_samples=max_samples
    )

    result = FeatureSelectResult(
        selected_features=selected_features,
        feature_scores=feature_scores,
        rfe_trace=rfe_trace
    )

    # Save artifacts
    os.makedirs(output_dir, exist_ok=True)

    # JSON Report
    report = {
        "selected_features": selected_features,
        "n_dropped": len(X.columns) - len(selected_features),
        "baseline_utility": rfe_trace.iloc[0]["oos_utility_mean"] if not rfe_trace.empty else 0.0,
        "final_utility": rfe_trace.iloc[-1]["oos_utility_mean"] if not rfe_trace.empty else 0.0,
        "rfe_trace": rfe_trace.to_dict(orient="records"),
    }
    with open(os.path.join(output_dir, "feature_select_report.json"), "w") as f:
        json.dump(report, f, indent=4)

    # CSV Reports
    feature_scores.to_csv(os.path.join(output_dir, "feature_select_report.csv"), index=False)
    rfe_trace.to_csv(os.path.join(output_dir, "rfe_trace.csv"), index=False)

    tprint(f"Feature Selection complete. Selected {len(selected_features)} features.")

    return result
