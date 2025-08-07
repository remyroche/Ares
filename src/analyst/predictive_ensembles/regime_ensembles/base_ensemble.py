import logging
import os  # For path manipulation
import warnings
from typing import Any

import joblib  # For saving/loading models
import numpy as np
import optuna
import pandas as pd

# Import SMOTE with fallback
try:
    from imblearn.over_sampling import SMOTE

    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

    # Create a dummy SMOTE class for fallback
    class SMOTE:
        def __init__(self, *args, **kwargs):
            pass

        def fit_resample(self, X, y):
            return X, y


from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning, module="arch")
optuna.logging.set_verbosity(optuna.logging.WARNING)


class BaseEnsemble:
    """
    Base class for all child ensembles to train highly optimized and robust models.
    Includes common utilities for training, prediction, and now, model persistence.
    Enhanced with L1-L2 regularization support.
    """

    def __init__(self, config: dict, ensemble_name: str):
        self.config = config.get("analyst", {}).get(ensemble_name, {})
        self.ensemble_name = ensemble_name
        self.logger = logging.getLogger(self.__class__.__name__)
        self.models: dict[Any, Any] = {}
        self.meta_learner = None
        self.trained = False
        self.pca = None
        self.meta_feature_scaler = StandardScaler()
        self.best_meta_params: dict[Any, Any] = {}
        self.label_encoder = LabelEncoder()
        self.n_pca_components = self.config.get("n_pca_components", 15)
        self.use_smote = self.config.get("use_smote", True)
        self.tune_base_models = self.config.get("tune_base_models", True)
        self.ensemble_weights = {self.ensemble_name: 1.0}  # Default initial weight

        # Regularization configuration - will be set by TrainingManager
        self.regularization_config = None

        # Unified Feature Lists - ENHANCED FOR MORE COMPREHENSIVE COVERAGE
        self.sequence_features = [
            "close",
            "volume",
            "ADX",
            "MACD_HIST",
            "ATR",
            "volume_delta",
            "autoencoder_reconstruction_error",
            "funding_rate",
            # Add more time-series relevant features from FE
            "Realized_Volatility",
            "Volatility_Regime_Numeric",
            "Hour_Sin",
            "Hour_Cos",
            "DayOfWeek_Sin",
            "DayOfWeek_Cos",
            "VROC",
            "OBV_Divergence",
            "Buy_Sell_Pressure_Ratio",
            "Order_Flow_Imbalance",
            "Funding_Momentum",
            "Funding_Divergence",
            "Funding_Extreme",
            "Price_Momentum",
            "Volatility_Momentum",
            "RSI_MACD_Divergence",
            "Volume_Price_Divergence",
            # SR Context Features (Phase 1 Enhancement)
            "distance_to_sr",
            "sr_strength",
            "sr_type",
            "price_position",
            "momentum_5",
            "momentum_10",
            "volume_ratio",
            "volatility",
        ]
        # Define comprehensive feature sets including liquidity features
        self.flat_features = [
            "RSI_14",
            "MACD_12_26_9",
            "MACDs_12_26_9",
            "MACDh_12_26_9",
            "BBU_20_2.0",
            "BBM_20_2.0",
            "BBL_20_2.0",
            "BBW_20_2.0",
            "BBP_20_2.0",
            "STOCHk_14_3_3",
            "STOCHd_14_3_3",
            "ATR_14",
            "ADX_14",
            "OBV",
            "VWAP",
            "SMA_9",
            "SMA_21",
            "SMA_50",
            "EMA_12",
            "EMA_26",
            "CCI_14",
            "MFI_14",
            "ROC_10",
            "Williams_R_14",
            "Parabolic_SAR",
            "SuperTrend_10_2.0",
            "DCU_20",
            "DCL_20",
            "DCM_20",
            "ATRr_14",
            "Volatility_Regime_Numeric",
            "Hour_Sin",
            "Hour_Cos",
            "DayOfWeek_Sin",
            "DayOfWeek_Cos",
            "VROC",
            "OBV_Divergence",
            "Buy_Sell_Pressure_Ratio",
            "Order_Flow_Imbalance",
            "Large_Order_Count",
            "Liquidity_Score",
            "Funding_Momentum",
            "Funding_Divergence",
            "Funding_Extreme",
            "Price_Momentum",
            "Price_Acceleration",
            "Volume_Momentum",
            "Volume_Acceleration",
            "Volatility_Momentum",
            "RSI_MACD_Divergence",
            "Volume_Price_Divergence",
            "Price_SMA_9_Ratio",
            "Price_SMA_21_Ratio",
            "Price_SMA_50_Ratio",
            "Volatility_Regime",
            # Advanced liquidity features
            "volume_liquidity",
            "price_impact",
            "spread_liquidity",
            "liquidity_regime",
            "liquidity_percentile",
            "kyle_lambda",
            "amihud_illiquidity",
            "order_flow_imbalance",
            "large_order_ratio",
            "vwap",
            "volume_roc",
            "volume_ma_ratio",
            "liquidity_stress",
            "liquidity_health",
            "realized_volatility",
            "parkinson_volatility",
            "garman_klass_volatility",
            "volatility_regime",
            "volatility_percentile",
            "autocorrelation_5",
            "autocorrelation_20",
            "cross_timeframe_correlation",
            "momentum_5",
            "momentum_20",
            "momentum_50",
            "momentum_acceleration",
            "momentum_strength",
            "momentum_divergence",
            "adaptive_sma",
            "adaptive_ema",
            "adaptive_period",
        ]
        self.order_flow_features = [
            "volume",
            "volume_delta",
            "cvd_slope",
            "OBV",
            "CMF",
            # Advanced liquidity features from advanced feature engineering
            "volume_liquidity",
            "price_impact",
            "spread_liquidity",
            "liquidity_regime",
            "liquidity_percentile",
            "kyle_lambda",
            "amihud_illiquidity",
            "order_flow_imbalance",
            "large_order_ratio",
            "vwap",
            "volume_roc",
            "volume_ma_ratio",
            "liquidity_stress",
            "liquidity_health",
            "Buy_Sell_Pressure_Ratio",
            "Order_Flow_Imbalance",
            "Large_Order_Count",
            "Liquidity_Score",
        ]

    def train_ensemble(
        self,
        historical_features: pd.DataFrame,
        historical_targets: pd.Series,
    ):
        self.logger.info(f"Starting full training pipeline for {self.ensemble_name}...")
        if historical_features.empty:
            self.logger.warning(
                f"No historical features for {self.ensemble_name}. Skipping training.",
            )
            return

        # Calculate SR context features (Phase 1 Enhancement)
        sr_features = self._calculate_sr_context_features(historical_features)
        historical_features = pd.concat([historical_features, sr_features], axis=1)

        # Ensure all expected features are present, fill missing with 0.0
        # Create a union of all features used across different types
        all_expected_features = list(
            set(self.sequence_features + self.flat_features + self.order_flow_features),
        )
        for col in all_expected_features:
            if col not in historical_features.columns:
                historical_features[col] = 0.0

        aligned_data = historical_features.join(
            historical_targets.rename("target"),
        ).dropna()
        if aligned_data.empty:
            self.logger.warning(
                f"Aligned data is empty for {self.ensemble_name} after dropping NaNs. Skipping training.",
            )
            return

        # Encode targets
        try:
            y_encoded = self.label_encoder.fit_transform(aligned_data["target"])
        except ValueError as e:
            self.logger.error(
                f"Error encoding labels for {self.ensemble_name}: {e}. Skipping training.",
                exc_info=True,
            )
            return

        self._train_base_models(aligned_data, y_encoded)

        # Prepare meta-features for meta-learner
        meta_features_train = self._get_meta_features(aligned_data, is_live=False)

        # Ensure meta_features_train is a DataFrame and has an index
        if (
            not isinstance(meta_features_train, pd.DataFrame)
            or meta_features_train.empty
        ):
            self.logger.warning(
                f"Meta-features are empty for {self.ensemble_name}. Cannot train meta-learner.",
            )
            return

        # Align meta-features with targets
        # Re-align y_encoded to the index of meta_features_train
        y_meta_train = (
            pd.Series(y_encoded, index=aligned_data.index)
            .loc[meta_features_train.index]
            .values
        )
        X_meta_train = meta_features_train

        if X_meta_train.empty or len(np.unique(y_meta_train)) < 2:
            self.logger.warning(
                f"Insufficient or single-class data for meta-learner in {self.ensemble_name}. Skipping meta-learner training.",
            )
            return

        self.logger.info("Applying PCA to meta-features...")
        X_meta_scaled = self.meta_feature_scaler.fit_transform(X_meta_train)
        n_components = min(self.n_pca_components, X_meta_scaled.shape[1])
        self.pca = PCA(n_components=n_components)
        X_meta_pca = self.pca.fit_transform(X_meta_scaled)
        X_meta_pca_df = pd.DataFrame(X_meta_pca, index=X_meta_train.index)

        self.logger.info("Tuning hyperparameters for meta-learner...")
        self.best_meta_params = self._tune_hyperparameters(
            LGBMClassifier,
            self._get_lgbm_search_space,
            X_meta_pca_df,
            y_meta_train,
        )
        self._train_meta_learner(X_meta_pca_df, y_meta_train, self.best_meta_params)
        self.trained = True
        self.logger.info(f"Training pipeline for {self.ensemble_name} complete.")

    def get_prediction(self, current_features: pd.DataFrame, **kwargs) -> dict:
        if not self.trained:
            self.logger.warning(
                f"Ensemble {self.ensemble_name} not trained. Returning HOLD.",
            )
            return {"prediction": "HOLD", "confidence": 0.0}

        # Calculate SR context features for prediction (Phase 1 Enhancement)
        sr_features = self._calculate_sr_context_features(current_features)
        current_features = pd.concat([current_features, sr_features], axis=1)

        # Ensure current_features has all expected columns, fill missing with 0.0
        all_expected_features = list(
            set(self.sequence_features + self.flat_features + self.order_flow_features),
        )
        for col in all_expected_features:
            if col not in current_features.columns:
                current_features[col] = 0.0

        meta_features = self._get_meta_features(
            current_features,
            is_live=True,
            **kwargs,
        )

        # Ensure meta_features contains all columns the scaler was fitted on
        # Create a DataFrame from the dictionary, then reindex
        meta_input_df = pd.DataFrame([meta_features])
        if hasattr(self.meta_feature_scaler, "feature_names_in_"):
            meta_input_df = meta_input_df.reindex(
                columns=self.meta_feature_scaler.feature_names_in_,
                fill_value=0,
            )
        else:
            self.logger.error(
                "Scaler not fitted with feature names. Cannot ensure correct feature order for prediction. Attempting with current columns.",
            )
            # Fallback: if feature_names_in_ is not available, assume current order is fine, but log warning.
            # This might happen if the scaler was loaded from an older checkpoint.

        meta_input_scaled = self.meta_feature_scaler.transform(meta_input_df)
        meta_input_pca = self.pca.transform(meta_input_scaled)
        return self._get_meta_prediction(meta_input_pca)

    def _train_with_smote(self, model, X, y):
        """Applies SMOTE to balance the dataset before training."""
        if self.use_smote and len(np.unique(y)) > 1:
            try:
                smote = SMOTE(random_state=42)
                X_res, y_res = smote.fit_resample(X, y)
                self.logger.info(
                    f"Applied SMOTE: Original size {X.shape[0]}, Resampled size {X_res.shape[0]}",
                )
                model.fit(X_res, y_res)
            except Exception as e:
                self.logger.warning(f"SMOTE failed: {e}. Training on original data.")
                model.fit(X, y)
        else:
            model.fit(X, y)
        return model

    def _tune_hyperparameters(self, model_class, search_space_func, X, y, n_trials=25):
        """Reusable Optuna hyperparameter tuning function."""
        if not self.tune_base_models:
            self.logger.info("Base model tuning is disabled. Using default parameters.")
            return {}  # Return empty dict, meaning default params will be used

        def objective(trial):
            params = search_space_func(trial)
            model = model_class(**params, random_state=42, verbose=-1)
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = []
            for train_idx, val_idx in cv.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                model.fit(X_train, y_train)
                scores.append(model.score(X_val, y_val))
            return np.mean(scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, n_jobs=-1)
        self.logger.info(
            f"Optuna best params for {model_class.__name__}: {study.best_params}",
        )
        return study.best_params

    def _get_lgbm_search_space(self, trial):
        """Enhanced LightGBM search space with regularization from config."""
        base_space = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        }

        # Use regularization config if available, otherwise use default optuna suggestions
        if self.regularization_config and "lightgbm" in self.regularization_config:
            reg_config = self.regularization_config["lightgbm"]
            base_space.update(
                {
                    "reg_alpha": reg_config.get(
                        "reg_alpha",
                        0.01,
                    ),  # Fixed L1 from config
                    "reg_lambda": reg_config.get(
                        "reg_lambda",
                        0.001,
                    ),  # Fixed L2 from config
                },
            )
            self.logger.info(
                f"Using configured regularization: L1={base_space['reg_alpha']}, L2={base_space['reg_lambda']}",
            )
        else:
            # Fallback to optuna optimization if no config available
            base_space.update(
                {
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                    "reg_lambda": trial.suggest_float(
                        "reg_lambda",
                        1e-3,
                        10.0,
                        log=True,
                    ),
                },
            )
            self.logger.info("Using optuna-optimized regularization parameters")

        return base_space

    def _get_regularized_logistic_regression(self):
        """Create a LogisticRegression model with L1-L2 regularization."""
        if self.regularization_config and "sklearn" in self.regularization_config:
            sklearn_config = self.regularization_config["sklearn"]

            # Use ElasticNet penalty for combined L1/L2 regularization
            model = LogisticRegression(
                penalty="elasticnet",
                C=sklearn_config.get("C", 1.0),
                l1_ratio=sklearn_config.get("l1_ratio", 0.5),  # 0.5 = equal L1/L2
                solver="saga",  # Required for elasticnet
                random_state=42,
                max_iter=1000,
            )
            self.logger.info(
                f"Created regularized LogisticRegression with C={sklearn_config.get('C')}, l1_ratio={sklearn_config.get('l1_ratio')}",
            )
        else:
            # Fallback to standard LogisticRegression
            model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                solver="liblinear",
            )
            self.logger.info(
                "Created standard LogisticRegression (no regularization config available)",
            )

        return model

    def _get_svm_search_space(self, trial):
        return {
            "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 1e-1, log=True),
            "kernel": trial.suggest_categorical("kernel", ["rbf", "poly"]),
            "probability": True,
        }

    def _train_meta_learner(self, X, y, params):
        self.meta_learner = LGBMClassifier(**params, random_state=42, verbose=-1)
        self.meta_learner.fit(X, y)

    def _get_meta_prediction(self, meta_input_pca):
        if not self.meta_learner:
            return {"prediction": "HOLD", "confidence": 0.0}
        proba = self.meta_learner.predict_proba(meta_input_pca)[0]
        idx = np.argmax(proba)
        return {
            "prediction": self.label_encoder.inverse_transform([idx])[0],
            "confidence": proba[idx],
        }

    def save_model(self, path: str):
        """Saves the entire ensemble instance to a file."""
        try:
            # Save relevant components
            model_data = {
                "models": self.models,
                "meta_learner": self.meta_learner,
                "pca": self.pca,
                "meta_feature_scaler": self.meta_feature_scaler,
                "label_encoder": self.label_encoder,
                "trained": self.trained,
                "best_meta_params": self.best_meta_params,
                "ensemble_weights": self.ensemble_weights,
            }
            joblib.dump(model_data, path)
            self.logger.info(f"Ensemble {self.ensemble_name} model saved to {path}")
        except Exception as e:
            self.logger.error(
                f"Error saving {self.ensemble_name} model to {path}: {e}",
                exc_info=True,
            )

    def load_model(self, path: str) -> bool:
        """Loads the entire ensemble instance from a file."""
        if not os.path.exists(path):
            self.logger.warning(
                f"Ensemble {self.ensemble_name} model file not found at {path}. Cannot load.",
            )
            self.trained = False
            return False
        try:
            model_data = joblib.load(path)
            self.models = model_data.get("models", {})
            self.meta_learner = model_data.get("meta_learner")
            self.pca = model_data.get("pca")
            self.meta_feature_scaler = model_data.get("meta_feature_scaler")
            self.label_encoder = model_data.get("label_encoder")
            self.trained = model_data.get("trained", False)
            self.best_meta_params = model_data.get("best_meta_params", {})
            self.ensemble_weights = model_data.get(
                "ensemble_weights",
                {self.ensemble_name: 1.0},
            )
            self.logger.info(f"Ensemble {self.ensemble_name} model loaded from {path}")
            return True
        except Exception as e:
            self.logger.error(
                f"Error loading {self.ensemble_name} model from {path}: {e}",
                exc_info=True,
            )
            self.trained = False
            return False

    def _train_base_models(self, aligned_data: pd.DataFrame, y_encoded: np.ndarray):
        raise NotImplementedError

    def _calculate_sr_context_features(
        self,
        df: pd.DataFrame,
        sr_analyzer=None,
    ) -> pd.DataFrame:
        """
        Calculate SR context features using unified regime classifier's S/R levels.

        Args:
            df: DataFrame with OHLCV data
            sr_analyzer: Optional SR analyzer instance (unified regime classifier)

        Returns:
            DataFrame with SR context features
        """
        try:
            # Initialize SR features with defaults
            sr_features = pd.DataFrame(index=df.index)
            sr_features["distance_to_sr"] = 1.0  # Default: far from SR
            sr_features["sr_strength"] = 0.0  # Default: no SR
            sr_features["sr_type"] = 0.5  # Default: neutral
            
            # New features for unified S/R levels
            sr_features["distance_to_pivot_support"] = 1.0
            sr_features["distance_to_pivot_resistance"] = 1.0
            sr_features["distance_to_hvn_support"] = 1.0
            sr_features["distance_to_hvn_resistance"] = 1.0
            sr_features["is_pivot_support"] = 0.0
            sr_features["is_pivot_resistance"] = 0.0
            sr_features["is_hvn_support"] = 0.0
            sr_features["is_hvn_resistance"] = 0.0
            sr_features["is_confluence"] = 0.0
            sr_features["is_open_range"] = 1.0
            
            # Strength features for S/R levels
            sr_features["nearest_pivot_strength"] = 0.0
            sr_features["nearest_pivot_touches"] = 0.0
            sr_features["nearest_pivot_volume"] = 0.0
            sr_features["nearest_pivot_age"] = 0.0
            sr_features["nearest_pivot_resistance_strength"] = 0.0
            sr_features["nearest_pivot_resistance_touches"] = 0.0
            sr_features["nearest_pivot_resistance_volume"] = 0.0
            sr_features["nearest_pivot_resistance_age"] = 0.0
            sr_features["nearest_hvn_strength"] = 0.0
            sr_features["nearest_hvn_touches"] = 0.0
            sr_features["nearest_hvn_volume"] = 0.0
            sr_features["nearest_hvn_age"] = 0.0
            sr_features["nearest_hvn_resistance_strength"] = 0.0
            sr_features["nearest_hvn_resistance_touches"] = 0.0
            sr_features["nearest_hvn_resistance_volume"] = 0.0
            sr_features["nearest_hvn_resistance_age"] = 0.0

            # Use unified regime classifier if available
            if sr_analyzer and hasattr(sr_analyzer, '_calculate_features'):
                try:
                    # Calculate features using unified regime classifier
                    features_df = sr_analyzer._calculate_features(df)
                    if not features_df.empty:
                        # Get location classification
                        location_labels = sr_analyzer._classify_location(features_df)
                        
                        # Extract S/R levels
                        pivot_levels = self._extract_pivot_levels(sr_analyzer, features_df)
                        hvn_levels = self._extract_hvn_levels(sr_analyzer, features_df)
                        
                        # Calculate S/R features for each row
                        for i in range(len(df)):
                            try:
                                current_price = df.iloc[i]["close"]
                                current_location = location_labels[i] if i < len(location_labels) else "OPEN_RANGE"
                                
                                # Calculate distances to S/R levels
                                self._calculate_sr_distances(
                                    sr_features, i, current_price, pivot_levels, hvn_levels, current_location
                                )
                                
                            except Exception as e:
                                self.logger.warning(
                                    f"Error calculating SR features for row {i}: {e}",
                                )
                                continue
                except Exception as e:
                    self.logger.warning(f"Error using unified regime classifier: {e}")
                    # Fall back to simplified approach
                    self._calculate_simplified_sr_features(sr_features, df)
            else:
                # Fall back to simplified approach
                self._calculate_simplified_sr_features(sr_features, df)

            # Calculate additional momentum and volatility features
            if len(df) >= 5:
                sr_features["momentum_5"] = df["close"].pct_change(5).fillna(0)
            else:
                sr_features["momentum_5"] = 0.0

            if len(df) >= 10:
                sr_features["momentum_10"] = df["close"].pct_change(10).fillna(0)
            else:
                sr_features["momentum_10"] = 0.0

            # Volume ratio
            if "volume" in df.columns:
                volume_ma = df["volume"].rolling(window=20, min_periods=1).mean()
                sr_features["volume_ratio"] = (df["volume"] / volume_ma).fillna(1.0)
            else:
                sr_features["volume_ratio"] = 1.0

            # Volatility
            if len(df) >= 20:
                price_volatility = (
                    df["close"].rolling(window=20).std()
                    / df["close"].rolling(window=20).mean()
                )
                sr_features["volatility"] = price_volatility.fillna(0.0)
            else:
                sr_features["volatility"] = 0.0

            # Price position within recent range
            if len(df) >= 20:
                high_20 = df["high"].rolling(window=20, min_periods=1).max()
                low_20 = df["low"].rolling(window=20, min_periods=1).min()
                sr_features["price_position"] = (
                    (df["close"] - low_20) / (high_20 - low_20)
                ).fillna(0.5)
            else:
                sr_features["price_position"] = 0.5

            return sr_features

        except Exception as e:
            self.logger.error(f"Error calculating SR context features: {e}")
            # Return default features
            default_features = pd.DataFrame(index=df.index)
            for feature in [
                "distance_to_sr",
                "sr_strength",
                "sr_type",
                "price_position",
                "momentum_5",
                "momentum_10",
                "volume_ratio",
                "volatility",
                "distance_to_pivot_support",
                "distance_to_pivot_resistance",
                "distance_to_hvn_support",
                "distance_to_hvn_resistance",
                "is_pivot_support",
                "is_pivot_resistance",
                "is_hvn_support",
                "is_hvn_resistance",
                "is_confluence",
                "is_open_range",
                "nearest_pivot_strength",
                "nearest_pivot_touches",
                "nearest_pivot_volume",
                "nearest_pivot_age",
                "nearest_pivot_resistance_strength",
                "nearest_pivot_resistance_touches",
                "nearest_pivot_resistance_volume",
                "nearest_pivot_resistance_age",
                "nearest_hvn_strength",
                "nearest_hvn_touches",
                "nearest_hvn_volume",
                "nearest_hvn_age",
                "nearest_hvn_resistance_strength",
                "nearest_hvn_resistance_touches",
                "nearest_hvn_resistance_volume",
                "nearest_hvn_resistance_age",
            ]:
                default_features[feature] = 0.0
            return default_features

    def _extract_pivot_levels(self, sr_analyzer, features_df: pd.DataFrame) -> dict[str, list[float]]:
        """
        Extract pivot levels from features DataFrame.
        
        Args:
            sr_analyzer: Unified regime classifier
            features_df: Features DataFrame
            
        Returns:
            dict: Pivot support and resistance levels
        """
        try:
            supports = []
            resistances = []
            
            # Calculate rolling pivots for the last few periods
            for i in range(max(0, len(features_df) - 24), len(features_df)):
                if i < 5:  # Need at least 5 data points
                    continue
                    
                window = features_df.iloc[max(0, i-24):i+1]
                if len(window) < 5:
                    continue
                    
                pivots = sr_analyzer._calculate_rolling_pivots(window)
                
                if pivots["s1"] > 0:
                    supports.append(pivots["s1"])
                if pivots["s2"] > 0:
                    supports.append(pivots["s2"])
                if pivots["r1"] > 0:
                    resistances.append(pivots["r1"])
                if pivots["r2"] > 0:
                    resistances.append(pivots["r2"])
            
            return {
                "supports": list(set(supports)),  # Remove duplicates
                "resistances": list(set(resistances))
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting pivot levels: {e}")
            return {"supports": [], "resistances": []}

    def _extract_hvn_levels(self, sr_analyzer, features_df: pd.DataFrame) -> dict[str, list[float]]:
        """
        Extract HVN levels from features DataFrame.
        
        Args:
            sr_analyzer: Unified regime classifier
            features_df: Features DataFrame
            
        Returns:
            dict: HVN support and resistance levels
        """
        try:
            supports = []
            resistances = []
            
            # Analyze volume levels for the last period
            if len(features_df) >= 720:  # Need enough data for HVN analysis
                hvn_window = features_df.iloc[-720:]
                volume_levels = sr_analyzer._analyze_volume_levels(hvn_window)
                
                if volume_levels:
                    for level_data in volume_levels.values():
                        level_price = level_data["price"]
                        current_price = features_df["close"].iloc[-1]
                        
                        # Classify as support or resistance based on current price
                        if current_price > level_price:
                            supports.append(level_price)
                        else:
                            resistances.append(level_price)
            
            return {
                "supports": list(set(supports)),  # Remove duplicates
                "resistances": list(set(resistances))
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting HVN levels: {e}")
            return {"supports": [], "resistances": []}

    def _calculate_sr_distances(
        self, 
        sr_features: pd.DataFrame, 
        row_idx: int, 
        current_price: float, 
        pivot_levels: dict, 
        hvn_levels: dict, 
        current_location: str
    ) -> None:
        """
        Calculate S/R distances and strength metrics for a specific row.
        
        Args:
            sr_features: DataFrame to update
            row_idx: Row index to update
            current_price: Current price
            pivot_levels: Pivot support/resistance levels with strength data
            hvn_levels: HVN support/resistance levels with strength data
            current_location: Current location classification
        """
        try:
            # Calculate distances to pivot levels
            pivot_supports = pivot_levels.get('supports', [])
            pivot_resistances = pivot_levels.get('resistances', [])
            pivot_strengths = pivot_levels.get('strengths', {})
            
            if pivot_supports:
                nearest_pivot_support = min(pivot_supports, key=lambda x: abs(x - current_price))
                sr_features.iloc[row_idx, sr_features.columns.get_loc("distance_to_pivot_support")] = (
                    current_price - nearest_pivot_support
                ) / current_price
                
                # Add pivot strength features
                if pivot_strengths:
                    strength_data = self._get_nearest_level_strength_data(nearest_pivot_support, pivot_supports, pivot_strengths)
                    if strength_data:
                        sr_features.iloc[row_idx, sr_features.columns.get_loc("nearest_pivot_strength")] = strength_data.get("strength", 0.0)
                        sr_features.iloc[row_idx, sr_features.columns.get_loc("nearest_pivot_touches")] = strength_data.get("touches", 0.0)
                        sr_features.iloc[row_idx, sr_features.columns.get_loc("nearest_pivot_volume")] = strength_data.get("volume", 0.0)
                        sr_features.iloc[row_idx, sr_features.columns.get_loc("nearest_pivot_age")] = strength_data.get("age", 0.0)
                
            if pivot_resistances:
                nearest_pivot_resistance = min(pivot_resistances, key=lambda x: abs(x - current_price))
                sr_features.iloc[row_idx, sr_features.columns.get_loc("distance_to_pivot_resistance")] = (
                    nearest_pivot_resistance - current_price
                ) / current_price
                
                # Add pivot resistance strength features
                if pivot_strengths:
                    strength_data = self._get_nearest_level_strength_data(nearest_pivot_resistance, pivot_resistances, pivot_strengths)
                    if strength_data:
                        sr_features.iloc[row_idx, sr_features.columns.get_loc("nearest_pivot_resistance_strength")] = strength_data.get("strength", 0.0)
                        sr_features.iloc[row_idx, sr_features.columns.get_loc("nearest_pivot_resistance_touches")] = strength_data.get("touches", 0.0)
                        sr_features.iloc[row_idx, sr_features.columns.get_loc("nearest_pivot_resistance_volume")] = strength_data.get("volume", 0.0)
                        sr_features.iloc[row_idx, sr_features.columns.get_loc("nearest_pivot_resistance_age")] = strength_data.get("age", 0.0)
            
            # Calculate distances to HVN levels
            hvn_supports = hvn_levels.get('supports', [])
            hvn_resistances = hvn_levels.get('resistances', [])
            hvn_strengths = hvn_levels.get('strengths', {})
            
            if hvn_supports:
                nearest_hvn_support = min(hvn_supports, key=lambda x: abs(x - current_price))
                sr_features.iloc[row_idx, sr_features.columns.get_loc("distance_to_hvn_support")] = (
                    current_price - nearest_hvn_support
                ) / current_price
                
                # Add HVN strength features
                if hvn_strengths:
                    strength_data = self._get_nearest_level_strength_data(nearest_hvn_support, hvn_supports, hvn_strengths)
                    if strength_data:
                        sr_features.iloc[row_idx, sr_features.columns.get_loc("nearest_hvn_strength")] = strength_data.get("strength", 0.0)
                        sr_features.iloc[row_idx, sr_features.columns.get_loc("nearest_hvn_touches")] = strength_data.get("touches", 0.0)
                        sr_features.iloc[row_idx, sr_features.columns.get_loc("nearest_hvn_volume")] = strength_data.get("volume", 0.0)
                        sr_features.iloc[row_idx, sr_features.columns.get_loc("nearest_hvn_age")] = strength_data.get("age", 0.0)
                
            if hvn_resistances:
                nearest_hvn_resistance = min(hvn_resistances, key=lambda x: abs(x - current_price))
                sr_features.iloc[row_idx, sr_features.columns.get_loc("distance_to_hvn_resistance")] = (
                    nearest_hvn_resistance - current_price
                ) / current_price
                
                # Add HVN resistance strength features
                if hvn_strengths:
                    strength_data = self._get_nearest_level_strength_data(nearest_hvn_resistance, hvn_resistances, hvn_strengths)
                    if strength_data:
                        sr_features.iloc[row_idx, sr_features.columns.get_loc("nearest_hvn_resistance_strength")] = strength_data.get("strength", 0.0)
                        sr_features.iloc[row_idx, sr_features.columns.get_loc("nearest_hvn_resistance_touches")] = strength_data.get("touches", 0.0)
                        sr_features.iloc[row_idx, sr_features.columns.get_loc("nearest_hvn_resistance_volume")] = strength_data.get("volume", 0.0)
                        sr_features.iloc[row_idx, sr_features.columns.get_loc("nearest_hvn_resistance_age")] = strength_data.get("age", 0.0)
            
            # Set location classification features
            sr_features.iloc[row_idx, sr_features.columns.get_loc("is_pivot_support")] = (
                1.0 if 'PIVOT_S' in current_location else 0.0
            )
            sr_features.iloc[row_idx, sr_features.columns.get_loc("is_pivot_resistance")] = (
                1.0 if 'PIVOT_R' in current_location else 0.0
            )
            sr_features.iloc[row_idx, sr_features.columns.get_loc("is_hvn_support")] = (
                1.0 if 'HVN_SUPPORT' in current_location else 0.0
            )
            sr_features.iloc[row_idx, sr_features.columns.get_loc("is_hvn_resistance")] = (
                1.0 if 'HVN_RESISTANCE' in current_location else 0.0
            )
            sr_features.iloc[row_idx, sr_features.columns.get_loc("is_confluence")] = (
                1.0 if 'CONFLUENCE' in current_location else 0.0
            )
            sr_features.iloc[row_idx, sr_features.columns.get_loc("is_open_range")] = (
                1.0 if current_location == 'OPEN_RANGE' else 0.0
            )
            
        except Exception as e:
            self.logger.warning(f"Error calculating S/R distances for row {row_idx}: {e}")

    def _get_nearest_level_strength_data(self, nearest_level: float, levels: list[float], strengths: dict) -> dict:
        """Get strength data for the nearest level."""
        try:
            if not levels or not strengths:
                return {}
            
            # Find the strength data for the nearest level
            for level_key, strength_data in strengths.items():
                if isinstance(strength_data, dict):
                    return strength_data
            
            return {}
        except Exception as e:
            self.logger.warning(f"Error getting nearest level strength data: {e}")
            return {}

    def _calculate_simplified_sr_features(self, sr_features: pd.DataFrame, df: pd.DataFrame) -> None:
        """
        Calculate simplified S/R features as fallback.
        
        Args:
            sr_features: DataFrame to update
            df: Input DataFrame
        """
        try:
            for i in range(len(df)):
                try:
                    current_price = df.iloc[i]["close"]

                    # Simplified SR context calculation
                    # Default values for SR features
                    sr_features.iloc[
                        i,
                        sr_features.columns.get_loc("distance_to_sr"),
                    ] = 1.0  # Default distance
                    sr_features.iloc[i, sr_features.columns.get_loc("sr_strength")] = (
                        0.0  # Default strength
                    )
                    sr_features.iloc[i, sr_features.columns.get_loc("sr_type")] = (
                        0.5  # Default type
                    )

                except Exception as e:
                    self.logger.warning(
                        f"Error calculating simplified SR features for row {i}: {e}",
                    )
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error calculating simplified SR features: {e}")

    def _get_meta_features(
        self,
        df: pd.DataFrame,
        is_live: bool = False,
        **kwargs,
    ) -> pd.DataFrame | dict:
        raise NotImplementedError
