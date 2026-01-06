import os
import json
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from econml.dml import LinearDML
from src.utils.tprint import tprint_info, tprint_warning, tprint_error

class OfflineCausalDiscovery:
    """
    Phase 3: Causal Discovery & Specialist Training (Offline Step)
    """
    def __init__(self, artifact_dir: str = "versioned_artifacts/causal_discovery"):
        self.artifact_dir = artifact_dir
        self.universal_drivers: List[str] = []

    def train_global_dml(self, X: pd.DataFrame, Y: pd.Series, T_col: str, W_cols: List[str]) -> float:
        tprint_info(f"   [Causal] Training DML for treatment: {T_col}")
        Y_vec = Y.values
        T_vec = X[T_col].values
        W_mat = X[W_cols].values

        est = LinearDML(
            model_y=RandomForestRegressor(n_estimators=50, max_depth=5, n_jobs=-1),
            model_t=RandomForestRegressor(n_estimators=50, max_depth=5, n_jobs=-1),
            discrete_treatment=False,
            random_state=42
        )
        try:
            est.fit(Y_vec, T_vec, W=W_mat)
            ate = est.ate(W_mat)
            tprint_info(f"   [Causal] ATE for {T_col}: {ate:.5f}")
            return ate
        except Exception as e:
            tprint_warning(f"DML failed for {T_col}: {e}")
            return 0.0

    def discover_drivers(self, df: pd.DataFrame, target_col: str, feature_cols: List[str],
                         nuisance_cols: List[str]):
        tprint_info("Starting Offline Causal Discovery...")
        T = df[feature_cols].values
        Y = df[target_col].values
        W = df[nuisance_cols].values

        tprint_info(f"   [Causal] Global DML Fit (Features: {len(feature_cols)}, Nuisance: {len(nuisance_cols)})...")
        est = LinearDML(
            model_y=RandomForestRegressor(n_estimators=100, max_depth=5, n_jobs=-1),
            model_t=RandomForestRegressor(n_estimators=100, max_depth=5, n_jobs=-1),
            discrete_treatment=False,
            random_state=42
        )
        est.fit(Y, T, W=W)

        effect_matrix = est.const_marginal_effect(df[nuisance_cols].values)
        global_importance = np.abs(effect_matrix).mean(axis=0)

        pred_importance = np.ones_like(global_importance)

        self.universal_drivers = []
        for i, feat in enumerate(feature_cols):
            if global_importance[i] > 0.05 * pred_importance[i]:
                self.universal_drivers.append(feat)

        tprint_info(f"   [Causal] Discovery Complete. Drivers: {len(self.universal_drivers)}")
        self.save_artifacts()

    def save_artifacts(self):
        tprint_info(f"   [Causal] Saving artifacts to {self.artifact_dir}...")
        os.makedirs(self.artifact_dir, exist_ok=True)
        with open(f"{self.artifact_dir}/universal_drivers.json", 'w') as f:
            json.dump(self.universal_drivers, f)

    def load_artifacts(self) -> List[str]:
        path = f"{self.artifact_dir}/universal_drivers.json"
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return []
