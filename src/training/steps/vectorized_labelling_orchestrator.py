# src/training/steps/vectorized_labelling_orchestrator.py

"""
Vectorized Labelling Orchestrator for comprehensive feature engineering and labeling pipeline.
Coordinates optimized_triple_barrier_labeling.py, vectorized_advanced_feature_engineering.py 
and autoencoder_feature_generator.py with advanced preprocessing and feature selection.
"""

import numpy as np
import pandas as pd
import pywt
from typing import Any, Dict, List, Optional, Tuple
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger


class VectorizedLabellingOrchestrator:
    """
    Comprehensive vectorized labeling orchestrator that coordinates all feature generation 
    and labeling components with advanced preprocessing and feature selection.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("VectorizedLabellingOrchestrator")

        # Configuration
        self.orchestrator_config = config.get("vectorized_labelling_orchestrator", {})
        self.enable_stationary_checks = self.orchestrator_config.get("enable_stationary_checks", True)
        self.enable_data_normalization = self.orchestrator_config.get("enable_data_normalization", True)
        self.enable_lookahead_bias_handling = self.orchestrator_config.get("enable_lookahead_bias_handling", True)
        self.enable_feature_selection = self.orchestrator_config.get("enable_feature_selection", True)
        self.enable_memory_efficient_types = self.orchestrator_config.get("enable_memory_efficient_types", True)
        self.enable_parquet_saving = self.orchestrator_config.get("enable_parquet_saving", True)

        # Feature selection configuration
        self.feature_selection_config = self.orchestrator_config.get("feature_selection", {})
        self.vif_threshold = self.feature_selection_config.get("vif_threshold", 5.0)
        self.mutual_info_threshold = self.feature_selection_config.get("mutual_info_threshold", 0.01)
        self.lightgbm_importance_threshold = self.feature_selection_config.get("lightgbm_importance_threshold", 0.01)

        # Multi-timeframe configuration
        self.timeframes = ["1m", "5m", "15m", "30m"]

        # Initialize components
        self.triple_barrier_labeler = None
        self.advanced_feature_engineer = None
        self.autoencoder_generator = None
        self.stationarity_checker = None
        self.feature_selector = None
        self.data_normalizer = None

        self.is_initialized = False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="vectorized labelling orchestrator initialization",
    )
    async def initialize(self) -> bool:
        """Initialize vectorized labeling orchestrator components."""
        try:
            self.logger.info("🚀 Initializing vectorized labeling orchestrator...")

            # Initialize triple barrier labeler
            from src.training.steps.step4_analyst_labeling_feature_engineering.optimized_triple_barrier_labeling import OptimizedTripleBarrierLabeling
            self.triple_barrier_labeler = OptimizedTripleBarrierLabeling(
                profit_take_multiplier=self.orchestrator_config.get("profit_take_multiplier", 0.002),
                stop_loss_multiplier=self.orchestrator_config.get("stop_loss_multiplier", 0.001),
                time_barrier_minutes=self.orchestrator_config.get("time_barrier_minutes", 30),
                max_lookahead=self.orchestrator_config.get("max_lookahead", 100),
            )

            # Initialize advanced feature engineer
            from src.training.steps.vectorized_advanced_feature_engineering import VectorizedAdvancedFeatureEngineering
            self.advanced_feature_engineer = VectorizedAdvancedFeatureEngineering(self.config)
            await self.advanced_feature_engineer.initialize()

            # Initialize autoencoder generator
            from src.analyst.autoencoder_feature_generator import AutoencoderFeatureGenerator
            self.autoencoder_generator = AutoencoderFeatureGenerator(self.config)

            # Initialize stationarity checker
            self.stationarity_checker = VectorizedStationarityChecker(self.config)

            # Initialize feature selector
            self.feature_selector = VectorizedFeatureSelector(self.config)

            # Initialize data normalizer
            self.data_normalizer = VectorizedDataNormalizer(self.config)

            self.is_initialized = True
            self.logger.info("✅ Vectorized labeling orchestrator initialized successfully")
            return True

        except Exception as e:
            self.logger.error(
                f"❌ Error initializing vectorized labeling orchestrator: {e}",
            )
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="vectorized labeling orchestration",
    )
    async def orchestrate_labeling_and_feature_engineering(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        order_flow_data: pd.DataFrame | None = None,
        sr_levels: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Orchestrate the complete labeling and feature engineering pipeline.

        Args:
            price_data: OHLCV price data
            volume_data: Volume and trade flow data
            order_flow_data: Order book and flow data (optional)
            sr_levels: Support/resistance levels (optional)

        Returns:
            Dictionary containing processed data and metadata
        """
        try:
            if not self.is_initialized:
                self.logger.error("Vectorized labeling orchestrator not initialized")
                return {}

            self.logger.info("🎯 Starting comprehensive vectorized labeling and feature engineering orchestration...")

            # 1. Stationary checks
            if self.enable_stationary_checks:
                self.logger.info("📊 Performing stationary checks...")
                stationary_data = await self.stationarity_checker.check_and_transform_stationarity(
                    price_data,
                    volume_data,
                    order_flow_data,
                )
                price_data = stationary_data.get("price_data", price_data)
                volume_data = stationary_data.get("volume_data", volume_data)
                order_flow_data = stationary_data.get("order_flow_data", order_flow_data)

            # 2. Advanced feature engineering
            self.logger.info("🔧 Generating advanced features...")
            advanced_features = await self.advanced_feature_engineer.engineer_features(
                price_data,
                volume_data,
                order_flow_data,
                sr_levels,
            )

            # 3. Triple barrier labeling
            self.logger.info("🏷️ Applying triple barrier labeling...")
            labeled_data = self.triple_barrier_labeler.apply_triple_barrier_labeling_vectorized(
                price_data.copy()
            )

            # 4. Combine features and labels
            self.logger.info("🔗 Combining features and labels...")
            combined_data = self._combine_features_and_labels_vectorized(
                labeled_data,
                advanced_features,
            )

            # 5. Feature selection
            if self.enable_feature_selection:
                self.logger.info("🎯 Performing feature selection...")
                selected_features = await self.feature_selector.select_optimal_features(
                    combined_data,
                    labeled_data["label"] if "label" in labeled_data.columns else None,
                )
                combined_data = selected_features

            # 6. Data normalization
            if self.enable_data_normalization:
                self.logger.info("📏 Normalizing data...")
                normalized_data = await self.data_normalizer.normalize_data(
                    combined_data,
                )
                combined_data = normalized_data

            # 7. Autoencoder feature generation
            self.logger.info("🤖 Generating autoencoder features...")
            autoencoder_features = self.autoencoder_generator.generate_features(
                combined_data,
                "vectorized_regime",
                labeled_data["label"].values if "label" in labeled_data.columns else np.zeros(len(combined_data)),
            )

            # 8. Final data preparation
            self.logger.info("🎨 Preparing final data...")
            final_data = self._prepare_final_data_vectorized(
                autoencoder_features,
                labeled_data,
            )

            # 9. Memory optimization
            if self.enable_memory_efficient_types:
                self.logger.info("💾 Optimizing memory usage...")
                final_data = self._optimize_memory_usage_vectorized(final_data)

            # 10. Save data
            if self.enable_parquet_saving:
                self.logger.info("💾 Saving data as Parquet...")
                self._save_data_as_parquet(final_data)

            self.logger.info(
                f"🎉 Vectorized labeling and feature engineering completed! Final shape: {final_data.shape}",
            )

            return {
                "data": final_data,
                "metadata": {
                    "total_features": len(final_data.columns),
                    "total_samples": len(final_data),
                    "feature_engineering_completed": True,
                    "labeling_completed": True,
                    "autoencoder_features_generated": True,
                    "stationary_checks_performed": self.enable_stationary_checks,
                    "data_normalized": self.enable_data_normalization,
                    "feature_selection_performed": self.enable_feature_selection,
                    "memory_optimized": self.enable_memory_efficient_types,
                    "saved_as_parquet": self.enable_parquet_saving,
                }
            }

        except Exception as e:
            self.logger.error(f"Error in vectorized labeling orchestration: {e}")
            return {}

    def _combine_features_and_labels_vectorized(
        self,
        labeled_data: pd.DataFrame,
        advanced_features: dict[str, Any],
    ) -> pd.DataFrame:
        """Combine features and labels using vectorized operations."""
        try:
            # Convert advanced features to DataFrame
            features_df = pd.DataFrame([advanced_features])
            
            # Replicate features for all rows
            features_df = pd.concat([features_df] * len(labeled_data), ignore_index=True)
            features_df.index = labeled_data.index

            # Combine with labeled data
            combined_data = pd.concat([labeled_data, features_df], axis=1)

            # Remove duplicate columns
            combined_data = combined_data.loc[:, ~combined_data.columns.duplicated()]

            return combined_data

        except Exception as e:
            self.logger.error(f"Error combining features and labels: {e}")
            return labeled_data

    def _prepare_final_data_vectorized(
        self,
        autoencoder_features: pd.DataFrame,
        labeled_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Prepare final data using vectorized operations."""
        try:
            # Combine autoencoder features with labeled data
            final_data = pd.concat([autoencoder_features, labeled_data], axis=1)

            # Remove duplicate columns
            final_data = final_data.loc[:, ~final_data.columns.duplicated()]

            # Ensure label column is present
            if "label" not in final_data.columns and "label" in labeled_data.columns:
                final_data["label"] = labeled_data["label"]

            # Remove infinite values
            final_data = final_data.replace([np.inf, -np.inf], np.nan)

            # Fill remaining NaN values
            final_data = final_data.fillna(method="ffill").fillna(method="bfill").fillna(0)

            # Remove columns with all NaN values
            final_data = final_data.dropna(axis=1, how="all")

            return final_data

        except Exception as e:
            self.logger.error(f"Error preparing final data: {e}")
            return autoencoder_features

    def _optimize_memory_usage_vectorized(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize memory usage using efficient data types."""
        try:
            optimized_data = data.copy()

            # Optimize numeric columns
            for col in optimized_data.select_dtypes(include=[np.number]).columns:
                col_min = optimized_data[col].min()
                col_max = optimized_data[col].max()
                
                if col_min >= 0 and col_max <= 255:
                    optimized_data[col] = optimized_data[col].astype(np.uint8)
                elif col_min >= -32768 and col_max <= 32767:
                    optimized_data[col] = optimized_data[col].astype(np.int16)
                elif col_min >= -2147483648 and col_max <= 2147483647:
                    optimized_data[col] = optimized_data[col].astype(np.int32)
                else:
                    optimized_data[col] = optimized_data[col].astype(np.float32)

            # Optimize categorical columns
            for col in optimized_data.select_dtypes(include=['object']).columns:
                if optimized_data[col].nunique() < 255:
                    optimized_data[col] = optimized_data[col].astype('category')

            return optimized_data

        except Exception as e:
            self.logger.error(f"Error optimizing memory usage: {e}")
            return data

    def _save_data_as_parquet(self, data: pd.DataFrame) -> None:
        """Save data as Parquet file."""
        try:
            import os
            from datetime import datetime

            # Create output directory
            output_dir = "data/vectorized_features"
            os.makedirs(output_dir, exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"vectorized_features_{timestamp}.parquet"
            filepath = os.path.join(output_dir, filename)

            # Save as Parquet
            data.to_parquet(filepath, index=True, compression='snappy')
            self.logger.info(f"💾 Data saved as Parquet: {filepath}")

        except Exception as e:
            self.logger.error(f"Error saving data as Parquet: {e}")


class VectorizedStationarityChecker:
    """Check and transform data for stationarity using vectorized operations."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("VectorizedStationarityChecker")

    async def check_and_transform_stationarity(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        order_flow_data: pd.DataFrame | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Check and transform data for stationarity using vectorized operations."""
        try:
            self.logger.info("🔍 Checking data stationarity...")

            # Check price data stationarity
            price_stationary = self._check_price_stationarity_vectorized(price_data)
            if not price_stationary:
                self.logger.info("📈 Transforming price data for stationarity...")
                price_data = self._transform_price_stationarity_vectorized(price_data)

            # Check volume data stationarity
            volume_stationary = self._check_volume_stationarity_vectorized(volume_data)
            if not volume_stationary:
                self.logger.info("📊 Transforming volume data for stationarity...")
                volume_data = self._transform_volume_stationarity_vectorized(volume_data)

            # Check order flow data stationarity
            order_flow_stationary = True
            if order_flow_data is not None:
                order_flow_stationary = self._check_order_flow_stationarity_vectorized(order_flow_data)
                if not order_flow_stationary:
                    self.logger.info("🔄 Transforming order flow data for stationarity...")
                    order_flow_data = self._transform_order_flow_stationarity_vectorized(order_flow_data)

            return {
                "price_data": price_data,
                "volume_data": volume_data,
                "order_flow_data": order_flow_data,
                "price_stationary": price_stationary,
                "volume_stationary": volume_stationary,
                "order_flow_stationary": order_flow_stationary,
            }

        except Exception as e:
            self.logger.error(f"Error checking and transforming stationarity: {e}")
            return {
                "price_data": price_data,
                "volume_data": volume_data,
                "order_flow_data": order_flow_data,
                "price_stationary": False,
                "volume_stationary": False,
                "order_flow_stationary": False,
            }

    def _check_price_stationarity_vectorized(self, price_data: pd.DataFrame) -> bool:
        """Check price data stationarity using vectorized operations."""
        try:
            # Calculate returns
            returns = price_data["close"].pct_change().dropna()
            
            # Augmented Dickey-Fuller test (simplified)
            # For vectorized implementation, we'll use a simplified approach
            # In practice, you might want to use statsmodels.tsa.stattools.adfuller
            
            # Check for trend
            trend = np.polyfit(range(len(returns)), returns, 1)[0]
            trend_threshold = 0.001
            
            # Check for autocorrelation
            autocorr = returns.autocorr()
            autocorr_threshold = 0.1
            
            # Check for variance stability
            rolling_std = returns.rolling(20).std()
            variance_ratio = rolling_std.iloc[-1] / rolling_std.iloc[0] if len(rolling_std) > 0 else 1.0
            variance_threshold = 2.0
            
            is_stationary = (
                abs(trend) < trend_threshold and
                abs(autocorr) < autocorr_threshold and
                variance_ratio < variance_threshold
            )
            
            return is_stationary

        except Exception as e:
            self.logger.error(f"Error checking price stationarity: {e}")
            return False

    def _transform_price_stationarity_vectorized(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Transform price data for stationarity using vectorized operations."""
        try:
            transformed_data = price_data.copy()
            
            # Calculate returns (first difference)
            transformed_data["close_returns"] = transformed_data["close"].pct_change()
            transformed_data["open_returns"] = transformed_data["open"].pct_change()
            transformed_data["high_returns"] = transformed_data["high"].pct_change()
            transformed_data["low_returns"] = transformed_data["low"].pct_change()
            
            # Log transformation for variance stabilization
            transformed_data["close_log"] = np.log(transformed_data["close"])
            transformed_data["open_log"] = np.log(transformed_data["open"])
            transformed_data["high_log"] = np.log(transformed_data["high"])
            transformed_data["low_log"] = np.log(transformed_data["low"])
            
            # Remove trend using rolling mean
            window = 20
            transformed_data["close_detrended"] = (
                transformed_data["close"] - transformed_data["close"].rolling(window).mean()
            )
            
            return transformed_data

        except Exception as e:
            self.logger.error(f"Error transforming price stationarity: {e}")
            return price_data

    def _check_volume_stationarity_vectorized(self, volume_data: pd.DataFrame) -> bool:
        """Check volume data stationarity using vectorized operations."""
        try:
            if "volume" not in volume_data.columns:
                return True
                
            volume = volume_data["volume"]
            
            # Check for trend
            trend = np.polyfit(range(len(volume)), volume, 1)[0]
            trend_threshold = 0.001
            
            # Check for autocorrelation
            autocorr = volume.autocorr()
            autocorr_threshold = 0.1
            
            # Check for variance stability
            rolling_std = volume.rolling(20).std()
            variance_ratio = rolling_std.iloc[-1] / rolling_std.iloc[0] if len(rolling_std) > 0 else 1.0
            variance_threshold = 2.0
            
            is_stationary = (
                abs(trend) < trend_threshold and
                abs(autocorr) < autocorr_threshold and
                variance_ratio < variance_threshold
            )
            
            return is_stationary

        except Exception as e:
            self.logger.error(f"Error checking volume stationarity: {e}")
            return False

    def _transform_volume_stationarity_vectorized(self, volume_data: pd.DataFrame) -> pd.DataFrame:
        """Transform volume data for stationarity using vectorized operations."""
        try:
            transformed_data = volume_data.copy()
            
            if "volume" in transformed_data.columns:
                # Log transformation
                transformed_data["volume_log"] = np.log(transformed_data["volume"])
                
                # Remove trend
                window = 20
                transformed_data["volume_detrended"] = (
                    transformed_data["volume"] - transformed_data["volume"].rolling(window).mean()
                )
                
                # Normalize volume
                transformed_data["volume_normalized"] = (
                    transformed_data["volume"] / transformed_data["volume"].rolling(window).mean()
                )
            
            return transformed_data

        except Exception as e:
            self.logger.error(f"Error transforming volume stationarity: {e}")
            return volume_data

    def _check_order_flow_stationarity_vectorized(self, order_flow_data: pd.DataFrame) -> bool:
        """Check order flow data stationarity using vectorized operations."""
        try:
            # Simplified check for order flow data
            # In practice, you might want more sophisticated checks
            
            # Check for any numeric columns
            numeric_cols = order_flow_data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                return True
                
            # Check first numeric column for stationarity
            first_col = numeric_cols[0]
            data = order_flow_data[first_col]
            
            # Check for trend
            trend = np.polyfit(range(len(data)), data, 1)[0]
            trend_threshold = 0.001
            
            # Check for autocorrelation
            autocorr = data.autocorr()
            autocorr_threshold = 0.1
            
            is_stationary = (
                abs(trend) < trend_threshold and
                abs(autocorr) < autocorr_threshold
            )
            
            return is_stationary

        except Exception as e:
            self.logger.error(f"Error checking order flow stationarity: {e}")
            return False

    def _transform_order_flow_stationarity_vectorized(self, order_flow_data: pd.DataFrame) -> pd.DataFrame:
        """Transform order flow data for stationarity using vectorized operations."""
        try:
            transformed_data = order_flow_data.copy()
            
            # Apply transformations to numeric columns
            numeric_cols = transformed_data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                # Log transformation (if positive)
                if (transformed_data[col] > 0).all():
                    transformed_data[f"{col}_log"] = np.log(transformed_data[col])
                
                # Remove trend
                window = 20
                transformed_data[f"{col}_detrended"] = (
                    transformed_data[col] - transformed_data[col].rolling(window).mean()
                )
                
                # Normalize
                transformed_data[f"{col}_normalized"] = (
                    transformed_data[col] / transformed_data[col].rolling(window).mean()
                )
            
            return transformed_data

        except Exception as e:
            self.logger.error(f"Error transforming order flow stationarity: {e}")
            return order_flow_data


class VectorizedFeatureSelector:
    """Select optimal features using VIF, Mutual Information, and LightGBM importance."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("VectorizedFeatureSelector")
        
        # Configuration
        self.feature_selection_config = config.get("feature_selection", {})
        self.vif_threshold = self.feature_selection_config.get("vif_threshold", 5.0)
        self.mutual_info_threshold = self.feature_selection_config.get("mutual_info_threshold", 0.01)
        self.lightgbm_importance_threshold = self.feature_selection_config.get("lightgbm_importance_threshold", 0.01)

    async def select_optimal_features(
        self,
        data: pd.DataFrame,
        labels: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Select optimal features using multiple methods."""
        try:
            self.logger.info("🎯 Starting feature selection...")
            
            # Remove non-numeric columns
            numeric_data = data.select_dtypes(include=[np.number])
            
            # 1. Remove constant features
            constant_features = numeric_data.columns[numeric_data.std() == 0]
            numeric_data = numeric_data.drop(columns=constant_features)
            self.logger.info(f"Removed {len(constant_features)} constant features")
            
            # 2. Remove highly correlated features
            correlated_features = self._remove_correlated_features_vectorized(numeric_data)
            numeric_data = numeric_data.drop(columns=correlated_features)
            self.logger.info(f"Removed {len(correlated_features)} highly correlated features")
            
            # 3. Remove features with high VIF
            if len(numeric_data.columns) > 1:
                high_vif_features = self._remove_high_vif_features_vectorized(numeric_data)
                numeric_data = numeric_data.drop(columns=high_vif_features)
                self.logger.info(f"Removed {len(high_vif_features)} high VIF features")
            
            # 4. Select features based on mutual information (if labels available)
            if labels is not None and len(numeric_data.columns) > 0:
                low_mi_features = self._remove_low_mutual_info_features_vectorized(numeric_data, labels)
                numeric_data = numeric_data.drop(columns=low_mi_features)
                self.logger.info(f"Removed {len(low_mi_features)} low mutual information features")
            
            # 5. Select features based on LightGBM importance (if labels available)
            if labels is not None and len(numeric_data.columns) > 0:
                low_importance_features = self._remove_low_importance_features_vectorized(numeric_data, labels)
                numeric_data = numeric_data.drop(columns=low_importance_features)
                self.logger.info(f"Removed {len(low_importance_features)} low importance features")
            
            self.logger.info(f"Feature selection completed. Final features: {len(numeric_data.columns)}")
            return numeric_data

        except Exception as e:
            self.logger.error(f"Error in feature selection: {e}")
            return data

    def _remove_correlated_features_vectorized(self, data: pd.DataFrame) -> List[str]:
        """Remove highly correlated features using vectorized operations."""
        try:
            correlation_matrix = data.corr()
            upper_triangle = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            high_correlation = np.abs(correlation_matrix) > 0.95
            high_correlation = high_correlation & upper_triangle
            
            to_drop = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i + 1, len(correlation_matrix.columns)):
                    if high_correlation.iloc[i, j]:
                        to_drop.append(correlation_matrix.columns[j])
            
            return list(set(to_drop))

        except Exception as e:
            self.logger.error(f"Error removing correlated features: {e}")
            return []

    def _remove_high_vif_features_vectorized(self, data: pd.DataFrame) -> List[str]:
        """Remove features with high Variance Inflation Factor using vectorized operations."""
        try:
            from sklearn.linear_model import LinearRegression
            
            vif_scores = {}
            for col in data.columns:
                # Use other features to predict this feature
                other_cols = [c for c in data.columns if c != col]
                if len(other_cols) > 0:
                    X = data[other_cols]
                    y = data[col]
                    
                    # Fit linear regression
                    reg = LinearRegression()
                    reg.fit(X, y)
                    
                    # Calculate R-squared
                    y_pred = reg.predict(X)
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    
                    # Calculate VIF
                    vif = 1 / (1 - r_squared) if r_squared != 1 else np.inf
                    vif_scores[col] = vif
            
            # Return features with high VIF
            high_vif_features = [col for col, vif in vif_scores.items() if vif > self.vif_threshold]
            return high_vif_features

        except Exception as e:
            self.logger.error(f"Error removing high VIF features: {e}")
            return []

    def _remove_low_mutual_info_features_vectorized(
        self,
        data: pd.DataFrame,
        labels: np.ndarray,
    ) -> List[str]:
        """Remove features with low mutual information using vectorized operations."""
        try:
            from sklearn.feature_selection import mutual_info_classif
            
            # Calculate mutual information scores
            mi_scores = mutual_info_classif(data, labels, random_state=42)
            
            # Create feature importance dictionary
            feature_importance = dict(zip(data.columns, mi_scores))
            
            # Return features with low mutual information
            low_mi_features = [
                col for col, score in feature_importance.items()
                if score < self.mutual_info_threshold
            ]
            
            return low_mi_features

        except Exception as e:
            self.logger.error(f"Error removing low mutual information features: {e}")
            return []

    def _remove_low_importance_features_vectorized(
        self,
        data: pd.DataFrame,
        labels: np.ndarray,
    ) -> List[str]:
        """Remove features with low LightGBM importance using vectorized operations."""
        try:
            import lightgbm as lgb
            
            # Train LightGBM model
            model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                verbose=-1,
            )
            model.fit(data, labels)
            
            # Get feature importance
            feature_importance = dict(zip(data.columns, model.feature_importances_))
            
            # Return features with low importance
            low_importance_features = [
                col for col, importance in feature_importance.items()
                if importance < self.lightgbm_importance_threshold
            ]
            
            return low_importance_features

        except Exception as e:
            self.logger.error(f"Error removing low importance features: {e}")
            return []


class VectorizedDataNormalizer:
    """Normalize data using various scaling methods with vectorized operations."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("VectorizedDataNormalizer")
        
        # Configuration
        self.normalization_config = config.get("data_normalization", {})
        self.scaling_method = self.normalization_config.get("scaling_method", "robust")
        self.outlier_handling = self.normalization_config.get("outlier_handling", "clip")

    async def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize data using vectorized operations."""
        try:
            self.logger.info("📏 Normalizing data...")
            
            normalized_data = data.copy()
            
            # Handle outliers
            if self.outlier_handling == "clip":
                normalized_data = self._clip_outliers_vectorized(normalized_data)
            elif self.outlier_handling == "remove":
                normalized_data = self._remove_outliers_vectorized(normalized_data)
            
            # Apply scaling
            if self.scaling_method == "robust":
                normalized_data = self._apply_robust_scaling_vectorized(normalized_data)
            elif self.scaling_method == "standard":
                normalized_data = self._apply_standard_scaling_vectorized(normalized_data)
            elif self.scaling_method == "minmax":
                normalized_data = self._apply_minmax_scaling_vectorized(normalized_data)
            
            self.logger.info("✅ Data normalization completed")
            return normalized_data

        except Exception as e:
            self.logger.error(f"Error normalizing data: {e}")
            return data

    def _clip_outliers_vectorized(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clip outliers using vectorized operations."""
        try:
            clipped_data = data.copy()
            
            for col in clipped_data.select_dtypes(include=[np.number]).columns:
                Q1 = clipped_data[col].quantile(0.25)
                Q3 = clipped_data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                clipped_data[col] = clipped_data[col].clip(lower=lower_bound, upper=upper_bound)
            
            return clipped_data

        except Exception as e:
            self.logger.error(f"Error clipping outliers: {e}")
            return data

    def _remove_outliers_vectorized(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using vectorized operations."""
        try:
            cleaned_data = data.copy()
            
            for col in cleaned_data.select_dtypes(include=[np.number]).columns:
                Q1 = cleaned_data[col].quantile(0.25)
                Q3 = cleaned_data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Create mask for outliers
                outlier_mask = (cleaned_data[col] < lower_bound) | (cleaned_data[col] > upper_bound)
                cleaned_data = cleaned_data[~outlier_mask]
            
            return cleaned_data

        except Exception as e:
            self.logger.error(f"Error removing outliers: {e}")
            return data

    def _apply_robust_scaling_vectorized(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply robust scaling using vectorized operations."""
        try:
            from sklearn.preprocessing import RobustScaler
            
            scaler = RobustScaler()
            numeric_data = data.select_dtypes(include=[np.number])
            
            if len(numeric_data.columns) > 0:
                scaled_data = scaler.fit_transform(numeric_data)
                data[numeric_data.columns] = scaled_data
            
            return data

        except Exception as e:
            self.logger.error(f"Error applying robust scaling: {e}")
            return data

    def _apply_standard_scaling_vectorized(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply standard scaling using vectorized operations."""
        try:
            from sklearn.preprocessing import StandardScaler
            
            scaler = StandardScaler()
            numeric_data = data.select_dtypes(include=[np.number])
            
            if len(numeric_data.columns) > 0:
                scaled_data = scaler.fit_transform(numeric_data)
                data[numeric_data.columns] = scaled_data
            
            return data

        except Exception as e:
            self.logger.error(f"Error applying standard scaling: {e}")
            return data

    def _apply_minmax_scaling_vectorized(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply min-max scaling using vectorized operations."""
        try:
            from sklearn.preprocessing import MinMaxScaler
            
            scaler = MinMaxScaler()
            numeric_data = data.select_dtypes(include=[np.number])
            
            if len(numeric_data.columns) > 0:
                scaled_data = scaler.fit_transform(numeric_data)
                data[numeric_data.columns] = scaled_data
            
            return data

        except Exception as e:
            self.logger.error(f"Error applying min-max scaling: {e}")
            return data
