import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple, Any
import logging
from scipy.stats import spearmanr
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

# Configure logging
logger = logging.getLogger(__name__)

class MultipleGeometriesFeatureSelector:
    """
    Two-stage feature selection for Multiple Geometries:
    1. Global Trimmer (Multi-Horizon Proxy): Reduces feature space to ~150 features valid for various horizons.
    2. Geometry Refinement: Selects ~70 features tailored to specific barrier configurations.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def select_global_multi_horizon_features(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        horizons: List[int] = [12, 48, 120],  # Short, Medium, Long
        target_n: int = 150,
        regime_features: List[str] = ['volatility_1d', 'trend_regime', 'vol_regime', 'volatility_regime']
    ) -> List[str]:
        """
        Stage 1: Select features that are predictive across multiple time horizons.

        Args:
            df: DataFrame containing features and close price.
            feature_columns: List of candidate feature names.
            horizons: List of forward horizons (bars) to create proxy targets.
            target_n: Target number of unique features to select.
            regime_features: List of mandatory features to include (will be verified).

        Returns:
            List of selected feature names.
        """
        if 'close' not in df.columns:
            logger.warning("Column 'close' missing for proxy target generation. Returning all features.")
            return feature_columns[:target_n]

        valid_features = [f for f in feature_columns if f in df.columns]
        if not valid_features:
            return []

        # 1. Generate Proxy Targets
        targets = {}
        valid_indices = df.index

        for h in horizons:
            # Future return (magnitude or signed? Use signed correlation to capture direction)
            # We use absolute correlation of signed return to capture both trend following and mean reversion potential.
            ret = df['close'].pct_change(h).shift(-h)

            # Fill NaNs with 0 for correlation or drop? Drop is better.
            target_name = f"proxy_ret_{h}"
            targets[target_name] = ret

        # 2. Calculate Correlation per Horizon
        feature_scores: Dict[str, float] = {f: 0.0 for f in valid_features}

        # Subsample for speed if needed (e.g. max 50k rows)
        if len(df) > 50000:
            sample_idx = np.linspace(0, len(df)-1, 50000, dtype=int)
            df_sub = df.iloc[sample_idx]
            targets_sub = {k: v.iloc[sample_idx] for k, v in targets.items()}
        else:
            df_sub = df
            targets_sub = targets

        X = df_sub[valid_features].fillna(0.0)

        for h, target_name in zip(horizons, targets.keys()):
            y = targets_sub[target_name]
            mask = y.notna()

            if mask.sum() < 100:
                continue

            X_masked = X.loc[mask]
            y_masked = y.loc[mask]

            # Vectorized correlation?
            # Pandas corrwith is cleaner
            corrs = X_masked.corrwith(y_masked, method='spearman').abs()

            for f, score in corrs.items():
                if np.isfinite(score):
                    # Max-pooling score across horizons:
                    # A feature is good if it is good for AT LEAST one horizon.
                    feature_scores[f] = max(feature_scores.get(f, 0.0), float(score))

        # 3. Select Top Features
        # Sort by score descending
        sorted_feats = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)

        # Mandatory inclusion
        mandatory = set([f for f in regime_features if f in df.columns])

        selected = set(mandatory)

        # Fill remaining slots
        for f, score in sorted_feats:
            if len(selected) >= target_n:
                break
            selected.add(f)

        final_list = list(selected)

        logger.info(f"Global Multi-Horizon Selection: Selected {len(final_list)} features (Target: {target_n})")
        return final_list

    def refine_features_for_geometry(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        target_n: int = 70,
        params: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Stage 2: Refine feature set for a specific geometry using LGBM Importance.

        Args:
            X: Feature matrix (subset from Stage 1).
            y: Binary labels for the specific geometry.
            target_n: Number of features to keep.
            params: Optional geometry parameters (for logging/context).

        Returns:
            List of refined feature names.
        """
        if X.empty or y.empty:
            return list(X.columns)

        # Align X and y
        common_idx = X.index.intersection(y.index)
        if len(common_idx) < 50:
            logger.warning("Not enough samples for refinement. Returning input columns.")
            return list(X.columns)

        X_aligned = X.loc[common_idx].fillna(0.0)
        y_aligned = y.loc[common_idx]

        # Valid labels check
        if y_aligned.nunique() < 2:
             return list(X.columns)

        # Train fast Probe
        # Use TimeSeriesSplit for validation or just simple train/val split?
        # Simple split is faster for selection.
        split = int(len(X_aligned) * 0.8)
        X_train, X_val = X_aligned.iloc[:split], X_aligned.iloc[split:]
        y_train, y_val = y_aligned.iloc[:split], y_aligned.iloc[split:]

        if y_train.nunique() < 2:
             return list(X.columns)

        lgbm_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'max_depth': 5,
            'objective': 'binary',
            'metric': 'auc',
            'verbose': -1,
            'n_jobs': 1,
            'random_state': self.random_state
        }

        try:
            model = lgb.LGBMClassifier(**lgbm_params)

            # Early stopping callbacks
            callbacks = [lgb.early_stopping(20, verbose=False)]

            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks
            )

            # Get Gain Importance
            # booster_ is accessible after fit
            importance = model.booster_.feature_importance(importance_type='gain')
            feature_names = X_aligned.columns.tolist()

            # Rank
            feat_imp = zip(feature_names, importance)
            sorted_feat = sorted(feat_imp, key=lambda x: x[1], reverse=True)

            top_k = [f for f, imp in sorted_feat[:target_n]]

            # Ensure we don't return empty if importance is 0
            if not top_k:
                return list(X.columns)[:target_n]

            return top_k

        except Exception as e:
            logger.warning(f"Feature refinement failed: {e}. Returning input columns.")
            return list(X.columns)[:target_n]
