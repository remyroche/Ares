from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Any, Callable
import logging
import os  # For path manipulation
import warnings
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
    # Create a minimal no-op SMOTE fallback
    class SMOTE:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.params = {"args": args, "kwargs": kwargs}

        def fit_resample(self, X: Any, y: Any) -> tuple[Any, Any]:
            return X, y

from lightgbm import LGBMClassifier
from src.utils.error_handler import handle_errors
from src.utils.purged_kfold import PurgedKFoldTime

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
        default_return=None, context="ensemble initialization",
    )
    def __init__(self, config: dict, ensemble_name: str):
        self.config = config.get("analyst", {}).get(ensemble_name, {})
        self.ensemble_name = ensemble_name
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing {self.ensemble_name} ensemble...")

        # Initialize model components
        self.models: dict[Any, Any] = {}
        self.meta_learner: Any | None = None
        self.trained = False
        self.pca: PCA | None = None
        self.meta_feature_scaler = StandardScaler()
        self.best_meta_params: dict[Any, Any] = {}
        self.label_encoder = LabelEncoder()

        # Load configuration parameters with defaults
        self.n_pca_components = self.config.get("n_pca_components", 15)
        self.use_smote = self.config.get("use_smote", True)
        self.tune_base_models = self.config.get("tune_base_models", True)
        self.ensemble_weights = {self.ensemble_name: 1.0}  # Default initial weight

        # Regularization configuration - will be set by TrainingManager
        self.regularization_config: dict[str, Any] | None = None

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
        default_return=None, context="ensemble training",
    )
    def train_ensemble(
        self, historical_features: pd.DataFrame,
        historical_targets: pd.Series | None = None) -> None:
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

        if historical_targets is None:
            self.logger.warning(
                f"No historical targets for {self.ensemble_name}. Skipping training.",
            )
            return

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
        self.logger.info(
            "Scaling and applying PCA to meta-features (train-only fit)...",
        )
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
            X_meta_pca_df, y_meta_train,
        )
        self._train_meta_learner(X_meta_pca_df, y_meta_train, self.best_meta_params)
        self.trained = True
        self.logger.info(f"Training pipeline for {self.ensemble_name} complete.")

        # Validate ensemble state after training
        self._validate_ensemble_state()

    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError, TypeError),
        default_return=False, context="ensemble state validation",
    )
    def _validate_ensemble_state(self) -> bool:
        """Validate that the ensemble is properly trained and ready for prediction."""
        try:
            if not self.trained:
                self.logger.warning(
                    f"{self.ensemble_name}: Ensemble not marked as trained",
                )
                return False

            if not self.models:
                self.logger.warning(f"{self.ensemble_name}: No base models found")
                return False

            if not self.meta_learner:
                self.logger.warning(f"{self.ensemble_name}: No meta-learner found")
                return False

            if not self.meta_feature_scaler:
                self.logger.warning(
                    f"{self.ensemble_name}: No meta-feature scaler found",
                )
                return False

            if not self.label_encoder:
                self.logger.warning(f"{self.ensemble_name}: No label encoder found")
                return False

            self.logger.info(f"{self.ensemble_name}: Ensemble state validation passed")
            return True

        except Exception as e:
            self.logger.exception(
                f"{self.ensemble_name}: Error validating ensemble state: {e}",
            )
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError, TypeError),
        default_return={"prediction": "HOLD", "confidence": 0.0},
        context="ensemble prediction",
    )
    def get_prediction(self, current_features: pd.DataFrame, **kwargs: Any) -> dict:
        if not self.trained:
            self.logger.warning(
                f"Ensemble {self.ensemble_name} not trained. Returning HOLD.",
            )
            return {"prediction": "HOLD", "confidence": 0.0}

        # SR context features are expected upstream (step4 unified S/R system)

        # Apply comprehensive feature normalization (Step 4 Enhancement)
        self.logger.info(
            "Applying comprehensive feature normalization for prediction...",
        )
        current_features = self.normalize_non_price_features(current_features)

        # Ensure current_features has all expected columns, fill missing with 0.0
        all_expected_features = list(
            set(self.sequence_features + self.flat_features + self.order_flow_features),
        )
        for col in all_expected_features:
            if col not in current_features.columns:
                current_features[col] = 0.0

        meta_features = self._get_meta_features(
            current_features, is_live=True, **kwargs
        )

        # Ensure meta_features contains all columns the scaler was fitted on
        # Create a DataFrame from the dictionary, then reindex
        meta_input_df = pd.DataFrame([meta_features])
        missing_cols: list[str] = []
        if hasattr(self.meta_feature_scaler, "feature_names_in_"):
            missing_cols = list(
                set(self.meta_feature_scaler.feature_names_in_)
                - set(meta_input_df.columns),
            )
            if missing_cols:
                self.logger.warning(
                    f"Missing meta features at inference: {missing_cols}",
                )
                meta_input_df = meta_input_df.reindex(
                    columns=self.meta_feature_scaler.feature_names_in_
                ).fillna(0)
        else:
            self.logger.warning(
                "Scaler not fitted with feature names. Attempting with current columns.",
            )
            # Proceed with current ordering

        meta_input_scaled = self.meta_feature_scaler.transform(meta_input_df)
        meta_input_pca = self.pca.transform(meta_input_scaled) if self.pca else meta_input_scaled
        return self._get_meta_prediction(meta_input_pca)

    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError, TypeError),
        default_return=None, context="SMOTE training",
    )
    def _train_with_smote(self, model: Any, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> Any:
        """Applies SMOTE to balance the dataset before training."""
        if self.use_smote and len(np.unique(y)) > 1:
            try:
                smote = SMOTE(random_state=42)
                X_res, y_res = smote.fit_resample(X, y)
                self.logger.info(
                    f"Applied SMOTE: Original size {np.shape(X)[0]}, Resampled size {np.shape(X_res)[0]}",
                )
                model.fit(X_res, y_res)
                return model
            except Exception as e:
                self.logger.warning(f"SMOTE failed: {e}. Training on original data.")
        model.fit(X, y)
        return model

    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError, TypeError),
        default_return={},
        context="hyperparameter tuning",
    )
    def _tune_hyperparameters(self, model_class: Callable[..., Any], search_space_func: Callable[[Any], dict[str, Any]], X: pd.DataFrame, y: np.ndarray, n_trials: int = 25) -> dict[str, Any]:
        """Reusable Optuna hyperparameter tuning function."""
        if not self.tune_base_models:
            self.logger.info("Base model tuning is disabled. Using default parameters.")
            return {}

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
    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError, TypeError),
        default_return=LogisticRegression(),
        context="regularized logistic regression",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError, TypeError),
        default_return={},
        context="SVM search space",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError, TypeError),
        default_return=None, context="meta learner training",
    )
    def _train_meta_learner(self, X: pd.DataFrame, y: np.ndarray, params: dict[str, Any]) -> None:
        self.meta_learner = LGBMClassifier(**params, random_state=42, verbose=-1)
        self.meta_learner.fit(X, y)

    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError, TypeError),
        default_return={"prediction": "HOLD", "confidence": 0.0},
        context="meta prediction",
    )
    def _get_meta_prediction(self, meta_input_pca: np.ndarray) -> dict[str, Any]:
        if not self.meta_learner:
            return {"prediction": "HOLD", "confidence": 0.0}
        proba = self.meta_learner.predict_proba(meta_input_pca)[0]
        idx = int(np.argmax(proba))
        return {
            "prediction": self.label_encoder.inverse_transform([idx])[0],
            "confidence": float(proba[idx]),
        }

    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError, TypeError),
        default_return=pd.DataFrame(),
        context="historical prediction",
    )
    def get_prediction_on_historical_data(
        self, historical_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Get predictions for historical data with comprehensive error handling."""
        try:
            if not self.trained:
                self.logger.warning(
                    f"{self.ensemble_name}: Ensemble not trained, returning empty DataFrame",
                )
                return pd.DataFrame()

            if historical_features.empty:
                self.logger.warning(
                    f"{self.ensemble_name}: Empty historical features provided",
                )
                return pd.DataFrame()

            # SR context features are expected upstream (step4 unified S/R system)

            # Apply feature normalization
            historical_features = self.normalize_non_price_features(historical_features)

            # Ensure all expected features are present
            all_expected_features = list(
                set(
                    self.sequence_features
                    + self.flat_features
                    + self.order_flow_features
                ),
            )
            for col in all_expected_features:
                if col not in historical_features.columns:
                    historical_features[col] = 0.0

            # Get meta features
            meta_features = self._get_meta_features(historical_features, is_live=False)

            if not isinstance(meta_features, pd.DataFrame) or meta_features.empty:
                self.logger.warning(
                    f"{self.ensemble_name}: Empty meta features generated",
                )
                return pd.DataFrame()

            # Ensure meta features have correct columns
            missing_cols: list[str] = []
            if hasattr(self.meta_feature_scaler, "feature_names_in_"):
                missing_cols = list(
                    set(self.meta_feature_scaler.feature_names_in_)
                    - set(meta_features.columns),
                )
                if missing_cols:
                    self.logger.warning(
                        f"Missing meta features for historical prediction: {missing_cols}",
                    )
                    meta_features = meta_features.reindex(
                        columns=self.meta_feature_scaler.feature_names_in_
                    ).fillna(0)

            # Transform and predict
            meta_input_scaled = self.meta_feature_scaler.transform(meta_features)
            meta_input_pca = self.pca.transform(meta_input_scaled) if self.pca else meta_input_scaled

            # Get predictions for all rows
            predictions: list[dict[str, Any]] = []
            for i in range(len(meta_input_pca)):
                try:
                    pred_result = self._get_meta_prediction(meta_input_pca[i : i + 1])
                    predictions.append(
                        {
                            "prediction": pred_result["prediction"],
                            "confidence": pred_result["confidence"],
                        },
                    )
                except Exception as e:
                    self.logger.warning(
                        f"{self.ensemble_name}: Error predicting row {i}: {e}",
                    )
                    predictions.append({"prediction": "HOLD", "confidence": 0.0})

            result_df = pd.DataFrame(predictions, index=historical_features.index)
            self.logger.info(
                f"{self.ensemble_name}: Generated predictions for {len(result_df)} historical samples",
            )
            return result_df

        except Exception as e:
            self.logger.exception(
                f"{self.ensemble_name}: Error in historical prediction: {e}",
            )
            return pd.DataFrame()

    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError, TypeError),
        default_return={"status": "unhealthy", "issues": ["Unknown error"]},
        context="ensemble health check",
    )
    def check_ensemble_health(self) -> dict[str, Any]:
        """Check the health status of the ensemble and return detailed diagnostics."""
        try:
            issues: list[str] = []
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
                "status": status, "ensemble_name": self.ensemble_name,
                "trained": self.trained,
                "num_base_models": len(self.models) if self.models else 0,
                "has_meta_learner": self.meta_learner is not None, "has_scaler": self.meta_feature_scaler is not None,
                "has_encoder": self.label_encoder is not None, "has_pca": self.pca is not None,
                "issues": issues,
                "timestamp": pd.Timestamp.now().isoformat(),
            }

            if status == "healthy":
                self.logger.info(f"{self.ensemble_name}: Ensemble health check passed")
            elif status == "degraded":
                self.logger.warning(
                    f"{self.ensemble_name}: Ensemble health check shows degraded status: {issues}",
                )
            else:
                self.logger.error(
                    f"{self.ensemble_name}: Ensemble health check failed: {issues}",
                )

            return health_report

        except Exception as e:
            self.logger.exception(
                f"{self.ensemble_name}: Error during health check: {e}",
            )
            return {
                "status": "error",
                "ensemble_name": self.ensemble_name, "issues": [f"Health check error: {e}"],
                "timestamp": pd.Timestamp.now().isoformat(),
            }

    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError, TypeError, OSError),
        default_return=None, context="model saving",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError, TypeError, OSError),
        default_return=False, context="model loading",
    )
    def _train_base_models(self, aligned_data: pd.DataFrame, y_encoded: np.ndarray):
        raise NotImplementedError

    # SR context features were moved to step4 unified S/R system.

    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError, TypeError),
        default_return={"support": [], "resistance": []},
        context="pivot levels extraction",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError, TypeError),
        default_return={"support": [], "resistance": []},
        context="HVN levels extraction",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError, TypeError),
        default_return={"strength": 0.0, "touches": 0, "volume": 0.0, "age": 0.0},
        context="level strength data extraction",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError, TypeError),
        default_return={},
        context="meta features extraction",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError, TypeError),
        default_return=None, context="feature normalization",
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
                    log_volume, self.normalization_windows["medium"],
                )

            # Volume relative to rolling turnover (if available)
            if "volume_ma_ratio" not in df.columns and "volume" in df.columns:
                volume_ma_20 = df["volume"].rolling(window=20, min_periods=1).mean()
                normalized_df["volume_ma_ratio"] = df["volume"] / (
                    volume_ma_20 + 1e-8
                )

            # 2. Spread and microstructure normalization
            spread_features = [
                "spread_liquidity",
                "price_impact",
                "kyle_lambda",
                "amihud_illiquidity",
            ]
            for feature in spread_features:
                if feature in df.columns:
                    # Convert to bps if not already (assuming raw values)
                    if feature in ["spread_liquidity", "price_impact"]:
                        # These might already be in bps, but ensure they're properly scaled
                        normalized_df[f"{feature}_bps"] = df[feature] * 1e4

                    # Rolling z-score
                    normalized_df[f"{feature}_z_score"] = (
                        self._calculate_rolling_z_score(
                            df[feature],
                            self.normalization_windows["medium"],
                        )
                    )

                    # Changes - use percentage change to avoid perfect correlation with base features
                    normalized_df[f"{feature}_pct_change"] = df[feature].pct_change()
                    # Use standard difference for better predictive power while avoiding perfect correlation
                    normalized_df[f"{feature}_change"] = df[feature].diff().fillna(0)

            # 3. Liquidity depth normalization
            liquidity_features = [
                "volume_liquidity",
                "liquidity_percentile",
                "liquidity_health",
            ]
            for feature in liquidity_features:
                if feature in df.columns:
                    # Log-transform heavy-tailed metrics
                    if feature in ["volume_liquidity"]:
                        normalized_df[f"{feature}_log"] = np.log1p(np.abs(df[feature]))

                    # Rolling z-score
                    normalized_df[f"{feature}_z_score"] = (
                        self._calculate_rolling_z_score(
                            df[feature],
                            self.normalization_windows["medium"],
                        )
                    )

                    # Changes for non-bounded features - use standard difference for better predictive power
                    if feature not in [
                        "liquidity_percentile",
                    ]:  # Percentiles are already bounded
                        normalized_df[f"{feature}_change"] = (
                            df[feature].diff().fillna(0)
                        )

            # 4. Order flow imbalance normalization
            ofi_features = [
                "order_flow_imbalance",
                "Order_Flow_Imbalance",
                "Buy_Sell_Pressure_Ratio",
            ]
            for feature in ofi_features:
                if feature in df.columns:
                    # Ensure bounded to [-1, 1]
                    normalized_df[f"{feature}_bounded"] = np.clip(df[feature], -1, 1)

                    # Rolling z-score of bounded values
                    normalized_df[f"{feature}_z_score"] = (
                        self._calculate_rolling_z_score(
                            normalized_df[f"{feature}_bounded"],
                            self.normalization_windows["medium"],
                        )
                    )

                    # Short-horizon changes (avoid over-differencing bounded ratios)
                    normalized_df[f"{feature}_change_1"] = normalized_df[
                        f"{feature}_bounded"
                    ].diff(1)
                    normalized_df[f"{feature}_change_3"] = normalized_df[
                        f"{feature}_bounded"
                    ].diff(3)

            # 5. VWAP normalization
            if "vwap" in df.columns and "close" in df.columns:
                # VWAP deviation from mid price
                mid_price = df["close"]
                normalized_df["vwap_deviation"] = (df["vwap"] - mid_price) / (
                    mid_price + 1e-8
                )

                # Rolling z-score of VWAP deviation
                normalized_df["vwap_deviation_z_score"] = (
                    self._calculate_rolling_z_score(
                        normalized_df["vwap_deviation"],
                        self.normalization_windows["medium"],
                    )
                )

            # 6. Large order ratio normalization (already bounded)
            if "large_order_ratio" in df.columns:
                # Clip to [0, 1] and rolling z-score
                normalized_df["large_order_ratio_bounded"] = np.clip(
                    df["large_order_ratio"],
                    0,
                    1,
                )

                normalized_df["large_order_ratio_z_score"] = (
                    self._calculate_rolling_z_score(
                        normalized_df["large_order_ratio_bounded"],
                        self.normalization_windows["medium"],
                    )
                )

            # 7. Funding rate normalization
            if "funding_rate" in df.columns:
                # Funding rates are already in percentage form, normalize with rolling z-score
                normalized_df["funding_rate_z_score"] = self._calculate_rolling_z_score(
                    df["funding_rate"],
                    self.normalization_windows["medium"],
                )

                # Funding rate changes - use multi-period difference to reduce correlation
                normalized_df["funding_rate_change"] = (
                    df["funding_rate"].diff(3).fillna(0)
                )
                normalized_df["funding_rate_acceleration"] = (
                    normalized_df["funding_rate_change"].diff(2).fillna(0)
                )

            # 8. Volatility normalization
            volatility_features = [
                "realized_volatility",
                "parkinson_volatility",
                "garman_klass_volatility",
            ]
            for feature in volatility_features:
                if feature in df.columns:
                    # Log-transform for heavy-tailed volatility
                    if feature in ["realized_volatility", "parkinson_volatility", "garman_klass_volatility"]:
                        normalized_df[f"{feature}_log"] = np.log1p(df[feature])

                    # Rolling z-score
                    normalized_df[f"{feature}_z_score"] = (
                        self._calculate_rolling_z_score(
                            normalized_df.get(f"{feature}_log", df[feature]),
                            self.normalization_windows["medium"],
                        )
                    )

                    # Volatility changes - use percentage change to avoid perfect correlation
                    normalized_df[f"{feature}_pct_change"] = df[feature].pct_change()
                    # Use multi-period difference for change to reduce correlation
                    normalized_df[f"{feature}_change"] = df[feature].diff(3).fillna(0)

            # 9. Momentum normalization
            momentum_features = [
                "momentum_5",
                "momentum_10",
                "momentum_20",
                "momentum_50",
            ]
            for feature in momentum_features:
                if feature in df.columns:
                    # Rolling z-score of momentum
                    normalized_df[f"{feature}_z_score"] = (
                        self._calculate_rolling_z_score(
                            df[feature],
                            self.normalization_windows["medium"],
                        )
                    )

                    # Momentum acceleration - use multi-period difference to reduce correlation
                    normalized_df[f"{feature}_acceleration"] = (
                        df[feature].diff(3).fillna(0)
                    )

            # 10. Winsorize outliers before final scaling
            self._winsorize_features(normalized_df)

            # 11. Final cleanup: handle any remaining NaN values
            normalized_df = normalized_df.fillna(0)

            self.logger.info(
                f"Applied comprehensive feature normalization to {len(normalized_df.columns)} features",
            )
            return normalized_df

        except Exception as e:
            self.logger.error(f"Error in feature normalization: {e}", exc_info=True)
            return df  # Return original if normalization fails

    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError, TypeError),
        default_return=pd.Series(dtype=float),
        context="rolling z-score calculation",
    )
    def _calculate_rolling_z_score(
        self, series: pd.Series,
        window: int = 60,
    ) -> pd.Series:
        """
        Calculate rolling z-score with proper handling of infinite values.

        Args:
            series: Input series
            window: Rolling window size

        Returns:
            Series with rolling z-scores
        """
        try:
            rolling_mean = series.rolling(window, min_periods=1).mean()
            rolling_std = series.rolling(window, min_periods=1).std()
            z_score = (series - rolling_mean) / (rolling_std + 1e-8)
            # Handle infinite values
            return z_score.replace([np.inf, -np.inf], 0)
        except Exception as e:
            self.logger.warning(f"Error calculating rolling z-score: {e}")
            return pd.Series(0, index=series.index)

    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError, TypeError),
        default_return=None, context="feature winsorization",
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
                if df[col].dtype in ["float64", "float32", "int64", "int32"]:
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
