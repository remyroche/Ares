import logging
import os  # For path manipulation
import warnings
from typing import Any

import joblib  # For saving/loading models
import numpy as np
import optuna
import pandas as pd

from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.warning_symbols import (
    error,
    failed,
    warning,
)

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
from src.utils.purged_kfold import PurgedKFoldTime
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning, module="arch")
optuna.logging.set_verbosity(optuna.logging.WARNING)


class BaseEnsemble:
    """
    Base class for all child ensembles to train highly optimized and robust models.
    Includes common utilities for training, prediction, and now, model persistence.
    Enhanced with L1-L2 regularization support and comprehensive feature normalization.
    """

    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError, TypeError),
        default_return=None,
        context="ensemble initialization",
    )
    def __init__(self, config: dict, ensemble_name: str):
        self.config = config.get("analyst", {}).get(ensemble_name, {})
        self.ensemble_name = ensemble_name
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing {self.ensemble_name} ensemble...")
        
        # Initialize model components
        self.models: dict[Any, Any] = {}
        self.meta_learner = None
        self.trained = False
        self.pca = None
        self.meta_feature_scaler = StandardScaler()
        self.best_meta_params: dict[Any, Any] = {}
        self.label_encoder = LabelEncoder()
        
        # Load configuration parameters with defaults
        self.n_pca_components = self.config.get("n_pca_components", 15)
        self.use_smote = self.config.get("use_smote", True)
        self.tune_base_models = self.config.get("tune_base_models", True)
        self.ensemble_weights = {self.ensemble_name: 1.0}  # Default initial weight

        # Regularization configuration - will be set by TrainingManager
        self.regularization_config = None

        # Feature normalization configuration
        self.normalization_windows = {
            "short": 20,  # For ROC, momentum, short-term changes
            "medium": 60,  # For rolling means, z-scores
            "long": 120,  # For longer-term normalization
        }

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
            # Normalized features (Step 4 Enhancement)
            "volume_log_diff",
            "volume_pct_change",
            "volume_z_score",
            "spread_liquidity_bps",
            "spread_liquidity_z_score",
            "spread_liquidity_change",
            "spread_liquidity_pct_change",
            "price_impact_bps",
            "price_impact_z_score",
            "price_impact_change",
            "price_impact_pct_change",
            "kyle_lambda_z_score",
            "kyle_lambda_change",
            "kyle_lambda_pct_change",
            "amihud_illiquidity_z_score",
            "amihud_illiquidity_change",
            "amihud_illiquidity_pct_change",
            "volume_liquidity_log",
            "volume_liquidity_z_score",
            "volume_liquidity_change",
            "liquidity_percentile_z_score",
            "liquidity_stress_log",
            "liquidity_stress_z_score",
            "liquidity_stress_change",
            "liquidity_health_z_score",
            "liquidity_health_change",
            "order_flow_imbalance_bounded",
            "order_flow_imbalance_z_score",
            "order_flow_imbalance_change_1",
            "order_flow_imbalance_change_3",
            "Order_Flow_Imbalance_bounded",
            "Order_Flow_Imbalance_z_score",
            "Order_Flow_Imbalance_change_1",
            "Order_Flow_Imbalance_change_3",
            "Buy_Sell_Pressure_Ratio_bounded",
            "Buy_Sell_Pressure_Ratio_z_score",
            "Buy_Sell_Pressure_Ratio_change_1",
            "Buy_Sell_Pressure_Ratio_change_3",
            "vwap_deviation",
            "vwap_deviation_z_score",
            "large_order_ratio_bounded",
            "large_order_ratio_z_score",
            "funding_rate_z_score",
            "funding_rate_change",
            "funding_rate_acceleration",
            "realized_volatility_log",
            "realized_volatility_z_score",
            "realized_volatility_change",
            "realized_volatility_pct_change",
            "parkinson_volatility_log",
            "parkinson_volatility_z_score",
            "parkinson_volatility_change",
            "parkinson_volatility_pct_change",
            "garman_klass_volatility_log",
            "garman_klass_volatility_z_score",
            "garman_klass_volatility_change",
            "garman_klass_volatility_pct_change",
            "momentum_5_z_score",
            "momentum_5_acceleration",
            "momentum_10_z_score",
            "momentum_10_acceleration",
            "momentum_20_z_score",
            "momentum_20_acceleration",
            "momentum_50_z_score",
            "momentum_50_acceleration",
            # Newly engineered order book features (stationary)
            "nearest_bid_wall_dist_pct",
            "nearest_ask_wall_dist_pct",
            "nearest_bid_wall_size_change",
            "nearest_ask_wall_size_change",
            "nearest_bid_wall_size_returns",
            "nearest_ask_wall_size_returns",
            "orderbook_wall_imbalance",
            "weighted_mid_price_returns",
            "weighted_mid_price_change",
            "depth_profile_slope_proxy",
            "orderbook_pressure",
            "trade_to_order_ratio",
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
            # Normalized order flow features (Step 4 Enhancement)
            "volume_log_diff",
            "volume_pct_change",
            "volume_z_score",
            "spread_liquidity_bps",
            "spread_liquidity_z_score",
            "spread_liquidity_change",
            "price_impact_bps",
            "price_impact_z_score",
            "price_impact_change",
            "kyle_lambda_z_score",
            "kyle_lambda_change",
            "amihud_illiquidity_z_score",
            "amihud_illiquidity_change",
            "volume_liquidity_log",
            "volume_liquidity_z_score",
            "volume_liquidity_change",
            "liquidity_percentile_z_score",
            "liquidity_stress_log",
            "liquidity_stress_z_score",
            "liquidity_stress_change",
            "liquidity_health_z_score",
            "liquidity_health_change",
            "order_flow_imbalance_bounded",
            "order_flow_imbalance_z_score",
            "order_flow_imbalance_change_1",
            "order_flow_imbalance_change_3",
            "Order_Flow_Imbalance_bounded",
            "Order_Flow_Imbalance_z_score",
            "Order_Flow_Imbalance_change_1",
            "Order_Flow_Imbalance_change_3",
            "Buy_Sell_Pressure_Ratio_bounded",
            "Buy_Sell_Pressure_Ratio_z_score",
            "Buy_Sell_Pressure_Ratio_change_1",
            "Buy_Sell_Pressure_Ratio_change_3",
            "vwap_deviation",
            "vwap_deviation_z_score",
            "large_order_ratio_bounded",
            "large_order_ratio_z_score",
            # Newly engineered order book features (stationary) for order-flow models
            "nearest_bid_wall_dist_pct",
            "nearest_ask_wall_dist_pct",
            "nearest_bid_wall_size_change",
            "nearest_ask_wall_size_change",
            "nearest_bid_wall_size_returns",
            "nearest_ask_wall_size_returns",
            "orderbook_wall_imbalance",
            "weighted_mid_price_returns",
            "weighted_mid_price_change",
            "depth_profile_slope_proxy",
            "orderbook_pressure",
            "trade_to_order_ratio",
        ]

    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError, TypeError),
        default_return=None,
        context="ensemble training",
    )
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

        # SR context features are now provided upstream (step4 unified S/R system)

        # Apply comprehensive feature normalization (Step 4 Enhancement)
        self.logger.info("Applying comprehensive feature normalization...")
        historical_features = self.normalize_non_price_features(historical_features)

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

        # Fit scaler and PCA on training data only
        self.logger.info("Scaling and applying PCA to meta-features (train-only fit)...")
        self.meta_feature_scaler = StandardScaler()
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
        
        # Validate ensemble state after training
        self._validate_ensemble_state()

    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError, TypeError),
        default_return=False,
        context="ensemble state validation",
    )
    def _validate_ensemble_state(self) -> bool:
        """Validate that the ensemble is properly trained and ready for prediction."""
        try:
            if not self.trained:
                self.logger.warning(f"{self.ensemble_name}: Ensemble not marked as trained")
                return False
                
            if not self.models:
                self.logger.warning(f"{self.ensemble_name}: No base models found")
                return False
                
            if not self.meta_learner:
                self.logger.warning(f"{self.ensemble_name}: No meta-learner found")
                return False
                
            if not self.meta_feature_scaler:
                self.logger.warning(f"{self.ensemble_name}: No meta-feature scaler found")
                return False
                
            if not self.label_encoder:
                self.logger.warning(f"{self.ensemble_name}: No label encoder found")
                return False
                
            self.logger.info(f"{self.ensemble_name}: Ensemble state validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"{self.ensemble_name}: Error validating ensemble state: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError, TypeError),
        default_return={"prediction": "HOLD", "confidence": 0.0},
        context="ensemble prediction",
    )
    def get_prediction(self, current_features: pd.DataFrame, **kwargs) -> dict:
        if not self.trained:
            self.logger.warning(
                f"Ensemble {self.ensemble_name} not trained. Returning HOLD.",
            )
            return {"prediction": "HOLD", "confidence": 0.0}

        # SR context features are expected upstream (step4 unified S/R system)

        # Apply comprehensive feature normalization (Step 4 Enhancement)
        self.logger.info("Applying comprehensive feature normalization for prediction...")
        current_features = self.normalize_non_price_features(current_features)

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
            missing_cols = list(set(self.meta_feature_scaler.feature_names_in_) - set(meta_input_df.columns))
            if missing_cols:
                self.logger.warning(f"Missing meta features at inference: {missing_cols}")
            meta_input_df = meta_input_df.reindex(
                columns=self.meta_feature_scaler.feature_names_in_,
            ).fillna(0)
        else:
            self.logger.error(
                "Scaler not fitted with feature names. Cannot ensure correct feature order for prediction. Attempting with current columns.",
            )
            # Fallback: if feature_names_in_ is not available, assume current order is fine, but log warning.
            # This might happen if the scaler was loaded from an older checkpoint.

        meta_input_scaled = self.meta_feature_scaler.transform(meta_input_df)
        meta_input_pca = self.pca.transform(meta_input_scaled)
        return self._get_meta_prediction(meta_input_pca)

    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError, TypeError),
        default_return=None,
        context="SMOTE training",
    )
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
            except Exception:
                self.print(failed("SMOTE failed: {e}. Training on original data."))
                model.fit(X, y)
        else:
            model.fit(X, y)
        return model

    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError, TypeError),
        default_return={},
        context="hyperparameter tuning",
    )
    def _tune_hyperparameters(self, model_class, search_space_func, X, y, n_trials=25):
        """Reusable Optuna hyperparameter tuning function."""
        if not self.tune_base_models:
            self.logger.info("Base model tuning is disabled. Using default parameters.")
            return {}  # Return empty dict, meaning default params will be used

        def objective(trial):
            params = search_space_func(trial)
            model = model_class(**params, random_state=42, verbose=-1)
            # Prefer purged CV for time series with DatetimeIndex
            if isinstance(X, pd.DataFrame) and isinstance(X.index, pd.DatetimeIndex):
                cv = PurgedKFoldTime(n_splits=3)
                splits = cv.split(X)
            else:
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                splits = cv.split(X, y)
            scores = []
            for train_idx, val_idx in splits:
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

    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError, TypeError),
        default_return={},
        context="LightGBM search space",
    )
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

    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError, TypeError),
        default_return=LogisticRegression(),
        context="regularized logistic regression",
    )
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

    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError, TypeError),
        default_return={},
        context="SVM search space",
    )
    def _get_svm_search_space(self, trial):
        return {
            "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 1e-1, log=True),
            "kernel": trial.suggest_categorical("kernel", ["rbf", "poly"]),
            "probability": True,
        }

    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError, TypeError),
        default_return=None,
        context="meta learner training",
    )
    def _train_meta_learner(self, X, y, params):
        self.meta_learner = LGBMClassifier(**params, random_state=42, verbose=-1)
        self.meta_learner.fit(X, y)

    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError, TypeError),
        default_return={"prediction": "HOLD", "confidence": 0.0},
        context="meta prediction",
    )
    def _get_meta_prediction(self, meta_input_pca):
        if not self.meta_learner:
            return {"prediction": "HOLD", "confidence": 0.0}
        proba = self.meta_learner.predict_proba(meta_input_pca)[0]
        idx = np.argmax(proba)
        return {
            "prediction": self.label_encoder.inverse_transform([idx])[0],
            "confidence": proba[idx],
        }

    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError, TypeError),
        default_return=pd.DataFrame(),
        context="historical prediction",
    )
    def get_prediction_on_historical_data(self, historical_features: pd.DataFrame) -> pd.DataFrame:
        """Get predictions for historical data with comprehensive error handling."""
        try:
            if not self.trained:
                self.logger.warning(f"{self.ensemble_name}: Ensemble not trained, returning empty DataFrame")
                return pd.DataFrame()
                
            if historical_features.empty:
                self.logger.warning(f"{self.ensemble_name}: Empty historical features provided")
                return pd.DataFrame()
                
            # SR context features are expected upstream (step4 unified S/R system)
            
            # Apply feature normalization
            historical_features = self.normalize_non_price_features(historical_features)
            
            # Ensure all expected features are present
            all_expected_features = list(
                set(self.sequence_features + self.flat_features + self.order_flow_features),
            )
            for col in all_expected_features:
                if col not in historical_features.columns:
                    historical_features[col] = 0.0
                    
            # Get meta features
            meta_features = self._get_meta_features(historical_features, is_live=False)
            
            if not isinstance(meta_features, pd.DataFrame) or meta_features.empty:
                self.logger.warning(f"{self.ensemble_name}: Empty meta features generated")
                return pd.DataFrame()
                
            # Ensure meta features have correct columns
            if hasattr(self.meta_feature_scaler, "feature_names_in_"):
                missing_cols = list(set(self.meta_feature_scaler.feature_names_in_) - set(meta_features.columns))
                if missing_cols:
                    self.logger.warning(f"Missing meta features for historical prediction: {missing_cols}")
                meta_features = meta_features.reindex(
                    columns=self.meta_feature_scaler.feature_names_in_,
                ).fillna(0)
                
            # Transform and predict
            meta_input_scaled = self.meta_feature_scaler.transform(meta_features)
            meta_input_pca = self.pca.transform(meta_input_scaled)
            
            # Get predictions for all rows
            predictions = []
            for i in range(len(meta_input_pca)):
                try:
                    pred_result = self._get_meta_prediction(meta_input_pca[i:i+1])
                    predictions.append({
                        "prediction": pred_result["prediction"],
                        "confidence": pred_result["confidence"]
                    })
                except Exception as e:
                    self.logger.warning(f"{self.ensemble_name}: Error predicting row {i}: {e}")
                    predictions.append({"prediction": "HOLD", "confidence": 0.0})
                    
            result_df = pd.DataFrame(predictions, index=historical_features.index)
            self.logger.info(f"{self.ensemble_name}: Generated predictions for {len(result_df)} historical samples")
            return result_df
            
        except Exception as e:
            self.logger.error(f"{self.ensemble_name}: Error in historical prediction: {e}")
            return pd.DataFrame()

    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError, TypeError),
        default_return={"status": "unhealthy", "issues": ["Unknown error"]},
        context="ensemble health check",
    )
    def check_ensemble_health(self) -> dict[str, Any]:
        """Check the health status of the ensemble and return detailed diagnostics."""
        try:
            issues = []
            status = "healthy"
            
            # Check training status
            if not self.trained:
                issues.append("Ensemble not trained")
                status = "unhealthy"
                
            # Check base models
            if not self.models:
                issues.append("No base models available")
                status = "unhealthy"
            else:
                for model_name, model in self.models.items():
                    if model is None:
                        issues.append(f"Base model '{model_name}' is None")
                        status = "degraded"
                        
            # Check meta-learner
            if not self.meta_learner:
                issues.append("No meta-learner available")
                status = "unhealthy"
                
            # Check scalers and encoders
            if not self.meta_feature_scaler:
                issues.append("No meta-feature scaler available")
                status = "unhealthy"
                
            if not self.label_encoder:
                issues.append("No label encoder available")
                status = "unhealthy"
                
            # Check PCA
            if not self.pca:
                issues.append("No PCA transformer available")
                status = "degraded"
                
            # Check configuration
            if not self.config:
                issues.append("No configuration available")
                status = "degraded"
                
            health_report = {
                "status": status,
                "ensemble_name": self.ensemble_name,
                "trained": self.trained,
                "num_base_models": len(self.models) if self.models else 0,
                "has_meta_learner": self.meta_learner is not None,
                "has_scaler": self.meta_feature_scaler is not None,
                "has_encoder": self.label_encoder is not None,
                "has_pca": self.pca is not None,
                "issues": issues,
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
            if status == "healthy":
                self.logger.info(f"{self.ensemble_name}: Ensemble health check passed")
            elif status == "degraded":
                self.logger.warning(f"{self.ensemble_name}: Ensemble health check shows degraded status: {issues}")
            else:
                self.logger.error(f"{self.ensemble_name}: Ensemble health check failed: {issues}")
                
            return health_report
            
        except Exception as e:
            self.logger.error(f"{self.ensemble_name}: Error during health check: {e}")
            return {
                "status": "error",
                "ensemble_name": self.ensemble_name,
                "issues": [f"Health check error: {e}"],
                "timestamp": pd.Timestamp.now().isoformat()
            }

    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError, TypeError, OSError),
        default_return=None,
        context="model saving",
    )
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

    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError, TypeError, OSError),
        default_return=False,
        context="model loading",
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

    # SR context features were moved to step4 unified S/R system.

    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError, TypeError),
        default_return={"support": [], "resistance": []},
        context="pivot levels extraction",
    )
    def _extract_pivot_levels(
        self,
        sr_analyzer,
        features_df: pd.DataFrame,
    ) -> dict[str, list[float]]:
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

                window = features_df.iloc[max(0, i - 24) : i + 1]
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
                "resistances": list(set(resistances)),
            }

        except Exception:
            self.print(error("Error extracting pivot levels: {e}"))
            return {"supports": [], "resistances": []}

    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError, TypeError),
        default_return={"support": [], "resistance": []},
        context="HVN levels extraction",
    )
    def _extract_hvn_levels(
        self,
        sr_analyzer,
        features_df: pd.DataFrame,
    ) -> dict[str, list[float]]:
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
                "resistances": list(set(resistances)),
            }

        except Exception:
            self.print(error("Error extracting HVN levels: {e}"))
            return {"supports": [], "resistances": []}

    def _calculate_sr_distances(
        self,
        sr_features: pd.DataFrame,
        row_idx: int,
        current_price: float,
        pivot_levels: dict,
        hvn_levels: dict,
        current_location: str,
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
            pivot_supports = pivot_levels.get("supports", [])
            pivot_resistances = pivot_levels.get("resistances", [])
            pivot_strengths = pivot_levels.get("strengths", {})

            if pivot_supports:
                nearest_pivot_support = min(
                    pivot_supports,
                    key=lambda x: abs(x - current_price),
                )
                sr_features.iloc[
                    row_idx,
                    sr_features.columns.get_loc("distance_to_pivot_support"),
                ] = (current_price - nearest_pivot_support) / current_price

                # Add pivot strength features
                if pivot_strengths:
                    strength_data = self._get_nearest_level_strength_data(
                        nearest_pivot_support,
                        pivot_supports,
                        pivot_strengths,
                    )
                    if strength_data:
                        sr_features.iloc[
                            row_idx,
                            sr_features.columns.get_loc("nearest_pivot_strength"),
                        ] = strength_data.get("strength", 0.0)
                        sr_features.iloc[
                            row_idx,
                            sr_features.columns.get_loc("nearest_pivot_touches"),
                        ] = strength_data.get("touches", 0.0)
                        sr_features.iloc[
                            row_idx,
                            sr_features.columns.get_loc("nearest_pivot_volume"),
                        ] = strength_data.get("volume", 0.0)
                        sr_features.iloc[
                            row_idx,
                            sr_features.columns.get_loc("nearest_pivot_age"),
                        ] = strength_data.get("age", 0.0)

            if pivot_resistances:
                nearest_pivot_resistance = min(
                    pivot_resistances,
                    key=lambda x: abs(x - current_price),
                )
                sr_features.iloc[
                    row_idx,
                    sr_features.columns.get_loc("distance_to_pivot_resistance"),
                ] = (nearest_pivot_resistance - current_price) / current_price

                # Add pivot resistance strength features
                if pivot_strengths:
                    strength_data = self._get_nearest_level_strength_data(
                        nearest_pivot_resistance,
                        pivot_resistances,
                        pivot_strengths,
                    )
                    if strength_data:
                        sr_features.iloc[
                            row_idx,
                            sr_features.columns.get_loc(
                                "nearest_pivot_resistance_strength",
                            ),
                        ] = strength_data.get("strength", 0.0)
                        sr_features.iloc[
                            row_idx,
                            sr_features.columns.get_loc(
                                "nearest_pivot_resistance_touches",
                            ),
                        ] = strength_data.get("touches", 0.0)
                        sr_features.iloc[
                            row_idx,
                            sr_features.columns.get_loc(
                                "nearest_pivot_resistance_volume",
                            ),
                        ] = strength_data.get("volume", 0.0)
                        sr_features.iloc[
                            row_idx,
                            sr_features.columns.get_loc("nearest_pivot_resistance_age"),
                        ] = strength_data.get("age", 0.0)

            # Calculate distances to HVN levels
            hvn_supports = hvn_levels.get("supports", [])
            hvn_resistances = hvn_levels.get("resistances", [])
            hvn_strengths = hvn_levels.get("strengths", {})

            if hvn_supports:
                nearest_hvn_support = min(
                    hvn_supports,
                    key=lambda x: abs(x - current_price),
                )
                sr_features.iloc[
                    row_idx,
                    sr_features.columns.get_loc("distance_to_hvn_support"),
                ] = (current_price - nearest_hvn_support) / current_price

                # Add HVN strength features
                if hvn_strengths:
                    strength_data = self._get_nearest_level_strength_data(
                        nearest_hvn_support,
                        hvn_supports,
                        hvn_strengths,
                    )
                    if strength_data:
                        sr_features.iloc[
                            row_idx,
                            sr_features.columns.get_loc("nearest_hvn_strength"),
                        ] = strength_data.get("strength", 0.0)
                        sr_features.iloc[
                            row_idx,
                            sr_features.columns.get_loc("nearest_hvn_touches"),
                        ] = strength_data.get("touches", 0.0)
                        sr_features.iloc[
                            row_idx,
                            sr_features.columns.get_loc("nearest_hvn_volume"),
                        ] = strength_data.get("volume", 0.0)
                        sr_features.iloc[
                            row_idx,
                            sr_features.columns.get_loc("nearest_hvn_age"),
                        ] = strength_data.get("age", 0.0)

            if hvn_resistances:
                nearest_hvn_resistance = min(
                    hvn_resistances,
                    key=lambda x: abs(x - current_price),
                )
                sr_features.iloc[
                    row_idx,
                    sr_features.columns.get_loc("distance_to_hvn_resistance"),
                ] = (nearest_hvn_resistance - current_price) / current_price

                # Add HVN resistance strength features
                if hvn_strengths:
                    strength_data = self._get_nearest_level_strength_data(
                        nearest_hvn_resistance,
                        hvn_resistances,
                        hvn_strengths,
                    )
                    if strength_data:
                        sr_features.iloc[
                            row_idx,
                            sr_features.columns.get_loc(
                                "nearest_hvn_resistance_strength",
                            ),
                        ] = strength_data.get("strength", 0.0)
                        sr_features.iloc[
                            row_idx,
                            sr_features.columns.get_loc(
                                "nearest_hvn_resistance_touches",
                            ),
                        ] = strength_data.get("touches", 0.0)
                        sr_features.iloc[
                            row_idx,
                            sr_features.columns.get_loc(
                                "nearest_hvn_resistance_volume",
                            ),
                        ] = strength_data.get("volume", 0.0)
                        sr_features.iloc[
                            row_idx,
                            sr_features.columns.get_loc("nearest_hvn_resistance_age"),
                        ] = strength_data.get("age", 0.0)

            # Set location classification features
            sr_features.iloc[
                row_idx,
                sr_features.columns.get_loc("is_pivot_support"),
            ] = 1.0 if "PIVOT_S" in current_location else 0.0
            sr_features.iloc[
                row_idx,
                sr_features.columns.get_loc("is_pivot_resistance"),
            ] = 1.0 if "PIVOT_R" in current_location else 0.0
            sr_features.iloc[row_idx, sr_features.columns.get_loc("is_hvn_support")] = (
                1.0 if "HVN_SUPPORT" in current_location else 0.0
            )
            sr_features.iloc[
                row_idx,
                sr_features.columns.get_loc("is_hvn_resistance"),
            ] = 1.0 if "HVN_RESISTANCE" in current_location else 0.0
            sr_features.iloc[row_idx, sr_features.columns.get_loc("is_confluence")] = (
                1.0 if "CONFLUENCE" in current_location else 0.0
            )
            sr_features.iloc[row_idx, sr_features.columns.get_loc("is_open_range")] = (
                1.0 if current_location == "OPEN_RANGE" else 0.0
            )

        except Exception as e:
            self.logger.warning(
                f"Error calculating S/R distances for row {row_idx}: {e}",
            )

    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError, TypeError),
        default_return={"strength": 0.0, "touches": 0, "volume": 0.0, "age": 0.0},
        context="level strength data extraction",
    )
    def _get_nearest_level_strength_data(
        self,
        nearest_level: float,
        levels: list[float],
        strengths: dict,
    ) -> dict:
        """Get strength data for the nearest level."""
        try:
            if not levels or not strengths:
                return {}

            # Find the strength data for the nearest level
            for strength_data in strengths.values():
                if isinstance(strength_data, dict):
                    return strength_data

            return {}
        except Exception:
            self.print(warning("Error getting nearest level strength data: {e}"))
            return {}

    def _calculate_simplified_sr_features(
        self,
        sr_features: pd.DataFrame,
        df: pd.DataFrame,
    ) -> None:
        """
        Calculate simplified S/R features as fallback.

        Args:
            sr_features: DataFrame to update
            df: Input DataFrame
        """
        try:
            for i in range(len(df)):
                try:
                    df.iloc[i]["close"]

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

        except Exception:
            self.print(error("Error calculating simplified SR features: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError, TypeError),
        default_return={},
        context="meta features extraction",
    )
    def _get_meta_features(
        self,
        df: pd.DataFrame,
        is_live: bool = False,
        **kwargs,
    ) -> pd.DataFrame | dict:
        raise NotImplementedError

    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError, TypeError),
        default_return=None,
        context="feature normalization",
    )
    def normalize_non_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize non-price series using relative/normalized changes and rolling z-scores.
        
        Implements the comprehensive normalization strategy:
        - Volume: log1p + first-difference, pct_change, rolling z-score
        - Spreads: convert to bps, rolling z-score, changes
        - Liquidity: relative normalization, log-transform, percentiles
        - Order flow: bounded normalization, changes
        - VWAP: deviations from mid, rolling z-score
        
        Args:
            df: DataFrame with features to normalize
            
        Returns:
            DataFrame with normalized features added
        """
        try:
            normalized_df = df.copy()
            
            # 1. Volume normalization
            if "volume" in df.columns:
                # Log1p + first-difference for stationarity
                normalized_df["volume_log_diff"] = np.log1p(df["volume"]).diff()
                
                # Percentage change
                normalized_df["volume_pct_change"] = df["volume"].pct_change()
                
                # Rolling z-score of log volume
                log_volume = np.log1p(df["volume"])
                normalized_df["volume_z_score"] = self._calculate_rolling_z_score(
                    log_volume, self.normalization_windows["medium"]
                )
                
                # Volume relative to rolling turnover (if available)
                if "volume_ma_ratio" not in df.columns:
                    volume_ma_20 = df["volume"].rolling(window=20, min_periods=1).mean()
                    normalized_df["volume_ma_ratio"] = df["volume"] / (volume_ma_20 + 1e-8)
            
            # 2. Spread and microstructure normalization
            spread_features = ["spread_liquidity", "price_impact", "kyle_lambda", "amihud_illiquidity"]
            for feature in spread_features:
                if feature in df.columns:
                    # Convert to bps if not already (assuming raw values)
                    if feature in ["spread_liquidity", "price_impact"]:
                        # These might already be in bps, but ensure they're properly scaled
                        normalized_df[f"{feature}_bps"] = df[feature] * 1e4
                    
                    # Rolling z-score
                    normalized_df[f"{feature}_z_score"] = self._calculate_rolling_z_score(
                        df[feature], self.normalization_windows["medium"]
                    )
                    
                    # Changes
                    normalized_df[f"{feature}_change"] = df[feature].diff()
                    normalized_df[f"{feature}_pct_change"] = df[feature].pct_change()
            
            # 3. Liquidity depth normalization
            liquidity_features = ["volume_liquidity", "liquidity_percentile", "liquidity_stress", "liquidity_health"]
            for feature in liquidity_features:
                if feature in df.columns:
                    # Log-transform heavy-tailed metrics
                    if feature in ["volume_liquidity", "liquidity_stress"]:
                        normalized_df[f"{feature}_log"] = np.log1p(np.abs(df[feature]))
                    
                    # Rolling z-score
                    normalized_df[f"{feature}_z_score"] = self._calculate_rolling_z_score(
                        df[feature], self.normalization_windows["medium"]
                    )
                    
                    # Changes for non-bounded features
                    if feature not in ["liquidity_percentile"]:  # Percentiles are already bounded
                        normalized_df[f"{feature}_change"] = df[feature].diff()
            
            # 4. Order flow imbalance normalization
            ofi_features = ["order_flow_imbalance", "Order_Flow_Imbalance", "Buy_Sell_Pressure_Ratio"]
            for feature in ofi_features:
                if feature in df.columns:
                    # Ensure bounded to [-1, 1]
                    normalized_df[f"{feature}_bounded"] = np.clip(df[feature], -1, 1)
                    
                    # Rolling z-score of bounded values
                    normalized_df[f"{feature}_z_score"] = self._calculate_rolling_z_score(
                        normalized_df[f"{feature}_bounded"], self.normalization_windows["medium"]
                    )
                    
                    # Short-horizon changes (avoid over-differencing bounded ratios)
                    normalized_df[f"{feature}_change_1"] = normalized_df[f"{feature}_bounded"].diff(1)
                    normalized_df[f"{feature}_change_3"] = normalized_df[f"{feature}_bounded"].diff(3)
            
            # 5. VWAP normalization
            if "vwap" in df.columns and "close" in df.columns:
                # VWAP deviation from mid price
                mid_price = df["close"]
                normalized_df["vwap_deviation"] = (df["vwap"] - mid_price) / (mid_price + 1e-8)
                
                # Rolling z-score of VWAP deviation
                normalized_df["vwap_deviation_z_score"] = self._calculate_rolling_z_score(
                    normalized_df["vwap_deviation"], self.normalization_windows["medium"]
                )
            
            # 6. Large order ratio normalization (already bounded)
            if "large_order_ratio" in df.columns:
                # Clip to [0, 1] and rolling z-score
                normalized_df["large_order_ratio_bounded"] = np.clip(df["large_order_ratio"], 0, 1)
                
                normalized_df["large_order_ratio_z_score"] = self._calculate_rolling_z_score(
                    normalized_df["large_order_ratio_bounded"], self.normalization_windows["medium"]
                )
            
            # 7. Funding rate normalization
            if "funding_rate" in df.columns:
                # Funding rates are already in percentage form, normalize with rolling z-score
                normalized_df["funding_rate_z_score"] = self._calculate_rolling_z_score(
                    df["funding_rate"], self.normalization_windows["medium"]
                )
                
                # Funding rate changes
                normalized_df["funding_rate_change"] = df["funding_rate"].diff()
                normalized_df["funding_rate_acceleration"] = normalized_df["funding_rate_change"].diff()
            
            # 8. Volatility normalization
            volatility_features = ["realized_volatility", "parkinson_volatility", "garman_klass_volatility"]
            for feature in volatility_features:
                if feature in df.columns:
                    # Log-transform for heavy-tailed volatility
                    normalized_df[f"{feature}_log"] = np.log1p(df[feature])
                    
                    # Rolling z-score
                    normalized_df[f"{feature}_z_score"] = self._calculate_rolling_z_score(
                        normalized_df[f"{feature}_log"], self.normalization_windows["medium"]
                    )
                    
                    # Volatility changes
                    normalized_df[f"{feature}_change"] = df[feature].diff()
                    normalized_df[f"{feature}_pct_change"] = df[feature].pct_change()
            
            # 9. Momentum normalization
            momentum_features = ["momentum_5", "momentum_10", "momentum_20", "momentum_50"]
            for feature in momentum_features:
                if feature in df.columns:
                    # Rolling z-score of momentum
                    normalized_df[f"{feature}_z_score"] = self._calculate_rolling_z_score(
                        df[feature], self.normalization_windows["medium"]
                    )
                    
                    # Momentum acceleration
                    normalized_df[f"{feature}_acceleration"] = df[feature].diff()
            
            # 10. Winsorize outliers before final scaling
            self._winsorize_features(normalized_df)
            
            # 11. Final cleanup: handle any remaining NaN values
            normalized_df = normalized_df.fillna(0)
            
            self.logger.info(f"Applied comprehensive feature normalization to {len(normalized_df.columns)} features")
            return normalized_df
            
        except Exception as e:
            self.logger.error(f"Error in feature normalization: {e}", exc_info=True)
            return df  # Return original if normalization fails
    
    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError, TypeError),
        default_return=pd.Series(dtype=float),
        context="rolling z-score calculation",
    )
    def _calculate_rolling_z_score(self, series: pd.Series, window: int = 60) -> pd.Series:
        """
        Calculate rolling z-score with proper handling of infinite values.
        
        Args:
            series: Input series
            window: Rolling window size
            
        Returns:
            Series with rolling z-scores
        """
        try:
            rolling_mean = series.rolling(window=window, min_periods=1).mean()
            rolling_std = series.rolling(window=window, min_periods=1).std()
            z_score = (series - rolling_mean) / (rolling_std + 1e-8)
            # Handle infinite values
            z_score = z_score.replace([np.inf, -np.inf], 0)
            return z_score
        except Exception as e:
            self.logger.warning(f"Error calculating rolling z-score: {e}")
            return pd.Series(0, index=series.index)

    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError, TypeError),
        default_return=None,
        context="feature winsorization",
    )
    def _winsorize_features(self, df: pd.DataFrame, percentile: float = 0.01) -> None:
        """
        Winsorize outliers in the DataFrame to improve numerical stability.
        
        Args:
            df: DataFrame to winsorize
            percentile: Percentile to clip at (default 1%)
        """
        try:
            for col in df.columns:
                if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    # Skip binary/categorical features
                    if df[col].nunique() <= 2:
                        continue
                    
                    # Handle NaN values first
                    if df[col].isna().any():
                        df[col] = df[col].fillna(df[col].median())
                    
                    # Calculate percentiles
                    lower_percentile = df[col].quantile(percentile)
                    upper_percentile = df[col].quantile(1 - percentile)
                    
                    # Clip outliers
                    df[col] = np.clip(df[col], lower_percentile, upper_percentile)
                    
        except Exception as e:
            self.logger.warning(f"Error in winsorization: {e}")
