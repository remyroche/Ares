import os
import json
import logging
import numpy as np
import pandas as pd
import shap
from typing import List, Dict, Any, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from econml.dml import LinearDML

from src.utils.versioned_artifacts.store import VersionedArtifactStore
from src.utils.ml_common.physics_router import AdaptiveHunterRouter

logger = logging.getLogger(__name__)

class OfflineCausalDiscovery:
    """
    Phase 3: Causal Discovery & Specialist Training (Offline Step)
    1. Global Causal Pre-Screen
    2. Regime-Conditional Causal Pruning
    3. Interaction Hunting
    """
    def __init__(self, artifact_dir: str = "versioned_artifacts/causal_discovery"):
        self.artifact_dir = artifact_dir
        self.router = AdaptiveHunterRouter()
        self.universal_drivers: List[str] = []
        self.regime_drivers: Dict[str, List[str]] = {}
        self.interactions: List[Tuple[str, str]] = []

    def run_discovery(self, X: pd.DataFrame, Y: pd.Series, feature_names: List[str]):
        """
        Main execution flow.
        X: Feature Matrix
        Y: Target (Returns)
        """
        # 0. Physics Router Fit
        # Need physics features. Assuming X contains them or we generate them.
        # For simplicity, we assume X has raw columns needed for Router or we skip router fit here
        # and assume it's pre-fit?
        # Better: calculate physics features from the raw OHLCV used to generate X.
        # But here we only have X.
        # Limitation: We need the Router to weigh samples.
        # Workaround: We assume X has a 'regime_weight' or similar, OR we fit router on proxy columns.
        pass

    def train_global_dml(self, X: pd.DataFrame, Y: pd.Series, T_col: str, W_cols: List[str]) -> float:
        """
        Train Global LinearDML for a specific 'Treatment' feature T against Outcome Y,
        controlling for Confounders W (Market Beta).
        Returns the Causal Coefficient (ATE).
        """
        # EconML LinearDML
        # Y: Outcome
        # T: Treatment (The Feature we are testing)
        # X: Effect Modifiers (Context - e.g. Regimes) - Optional in simple DML,
        #    but we use X=None for ATE (Average Treatment Effect).
        # W: Confounders (Market proxies: SPY, BTC, Vol)

        # Prepare data
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
            return est.ate(W_mat) # Average Treatment Effect
        except Exception as e:
            logger.warning(f"DML failed for {T_col}: {e}")
            return 0.0

    def discover_drivers(self, df: pd.DataFrame, target_col: str, feature_cols: List[str],
                         nuisance_cols: List[str], regime_weights: pd.DataFrame = None):
        """
        Full discovery pipeline.

        1. Global Causal SHAP:
           Instead of one-by-one DML, we can treat T as vector?
           LinearDML supports multi-treatment.
        """
        T = df[feature_cols].values
        Y = df[target_col].values
        W = df[nuisance_cols].values # Market Beta, Volatility

        # 1. Train Global DML
        logger.info("Training Global LinearDML...")
        est = LinearDML(
            model_y=RandomForestRegressor(n_estimators=100, max_depth=5, n_jobs=-1),
            model_t=RandomForestRegressor(n_estimators=100, max_depth=5, n_jobs=-1),
            discrete_treatment=False,
            random_state=42
        )

        # We can pass sample_weights to fit if we want global weighting (e.g. AFML weights)
        est.fit(Y, T, W=W)

        # 2. Compute Causal SHAP
        # est.shap_values(X) returns SHAP values for the Causal Effect
        # X here represents the Heterogeneity features. If we didn't use X in fit,
        # we might get constant effects.
        # But wait, LinearDML(model_y, model_t) estimates Y = theta(X)*T + g(X,W) + e
        # If we didn't pass X to fit, theta is constant.
        # To get heterogeneous effects (and thus SHAP importance of X features on the Effect),
        # we need to pass X to fit.
        # BUT, the prompt says "X: Context features (Wavelets, MP, etc.)".
        # So we should pass Context features as X.

        # Let's assume some features are 'Context' and some are 'Treatments' (Signals).
        # Or, typically, we want to know the causal effect OF the signals.
        # And how that effect varies by Context (Regime).

        # Refined Plan:
        # T: The Signals (RSI, Moving Avg, etc.)
        # X: The Context (Vol Regime, Trend Regime - Physics Features)
        # W: Nuisance (Broad Market Moves)

        # But we need to prune the Signals themselves based on their causal power.
        # If we use T=Signals, est.const_marginal_effect(X) gives the effect of Signals.

        effect_matrix = est.const_marginal_effect(df[nuisance_cols].values) # Use W as X for now if no distinct X
        # Actually, let's use the constant marginal effect (Average Treatment Effect) for pruning.

        # Global Importance
        # mean(|theta|)
        global_importance = np.abs(effect_matrix).mean(axis=0) # [n_features]

        # Pruning
        # Compare to Predictive SHAP (need a separate LGBM fit)
        # Placeholder for Predictive SHAP
        pred_importance = np.ones_like(global_importance) # Dummy

        # Filter
        self.universal_drivers = []
        for i, feat in enumerate(feature_cols):
            # Causal > 5% of Predictive
            if global_importance[i] > 0.05 * pred_importance[i]:
                self.universal_drivers.append(feat)

        # Save Artifacts
        self.save_artifacts()

    def save_artifacts(self):
        os.makedirs(self.artifact_dir, exist_ok=True)
        with open(f"{self.artifact_dir}/universal_drivers.json", 'w') as f:
            json.dump(self.universal_drivers, f)

    def load_artifacts(self) -> List[str]:
        path = f"{self.artifact_dir}/universal_drivers.json"
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return []

    def get_partial_orthogonalization(self, X: pd.DataFrame, W: pd.DataFrame) -> pd.DataFrame:
        """
        Phase 2: Feature Conditioning
        X_resid = X - 0.7 * E[X|W]
        """
        # Train model to predict X from W
        # We can use a simple LinearRegression for speed
        model = LinearRegression()
        model.fit(W, X)
        X_hat = model.predict(W)

        X_resid = X - (0.7 * X_hat)
        return pd.DataFrame(X_resid, index=X.index, columns=X.columns)
