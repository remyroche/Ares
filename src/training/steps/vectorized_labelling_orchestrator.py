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

            # Initialize triple barrier labeler using the proper implementation
            from src.training.steps.step4_analyst_labeling_feature_engineering_components.optimized_triple_barrier_labeling import OptimizedTripleBarrierLabeling
            
            self.triple_barrier_labeler = OptimizedTripleBarrierLabeling(
                profit_take_multiplier=self.orchestrator_config.get("profit_take_multiplier", 0.002),
                stop_loss_multiplier=self.orchestrator_config.get("stop_loss_multiplier", 0.001),
                time_barrier_minutes=self.orchestrator_config.get("time_barrier_minutes", 30),
                max_lookahead=self.orchestrator_config.get("max_lookahead", 100),
            )

            # Initialize advanced feature engineer with error handling
            try:
                from src.training.steps.vectorized_advanced_feature_engineering import VectorizedAdvancedFeatureEngineering
                self.advanced_feature_engineer = VectorizedAdvancedFeatureEngineering(self.config)
                await self.advanced_feature_engineer.initialize()
            except Exception as e:
                self.logger.warning(f"Failed to initialize advanced feature engineer: {e}")
                self.advanced_feature_engineer = None

            # Initialize autoencoder generator with error handling
            try:
                from src.analyst.autoencoder_feature_generator import AutoencoderFeatureGenerator
                self.autoencoder_generator = AutoencoderFeatureGenerator(self.config)
            except Exception as e:
                self.logger.warning(f"Failed to initialize autoencoder generator: {e}")
                self.autoencoder_generator = None

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
            # Set basic components even if advanced ones fail
            from src.training.steps.step4_analyst_labeling_feature_engineering_components.optimized_triple_barrier_labeling import OptimizedTripleBarrierLabeling
            self.triple_barrier_labeler = OptimizedTripleBarrierLabeling()
            self.stationarity_checker = VectorizedStationarityChecker(self.config)
            self.feature_selector = VectorizedFeatureSelector(self.config)
            self.data_normalizer = VectorizedDataNormalizer(self.config)
            self.is_initialized = True
            self.logger.warning("⚠️ Vectorized labeling orchestrator initialized with fallback components")
            return True

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
            if self.advanced_feature_engineer is not None:
                advanced_features = await self.advanced_feature_engineer.engineer_features(
                    price_data,
                    volume_data,
                    order_flow_data,
                    sr_levels,
                )
            else:
                self.logger.warning("Advanced feature engineer not available, using basic features")
                advanced_features = {}

            # 3. Triple barrier labeling
            self.logger.info("🏷️ Applying triple barrier labeling...")
            labeled_data = self.triple_barrier_labeler.apply_triple_barrier_labeling_vectorized(
                price_data.copy()
            )

            # 4. Combine engineered features (time-aligned) with labels only
            self.logger.info("🔗 Combining engineered features and labels...")
            combined_data = self._combine_features_and_labels_vectorized(
                labeled_data,
                advanced_features,
            )

            # Validate feature set before selection
            feature_cols = [c for c in combined_data.columns if c != "label"]
            if len(feature_cols) == 0:
                self.logger.warning("No engineered features available after combination; skipping selection and normalization")
            else:
                # 5. Feature selection on engineered features only
                if self.enable_feature_selection:
                    self.logger.info("🎯 Performing feature selection...")
                    features_only = combined_data.drop(columns=["label"], errors="ignore")
                    # Exclude raw OHLCV if present as extra safety
                    ohlcv_cols = ["open", "high", "low", "close", "volume"]
                    features_only = features_only.drop(columns=[c for c in ohlcv_cols if c in features_only.columns], errors="ignore")
                    selected_features = await self.feature_selector.select_optimal_features(
                        features_only,
                        labeled_data["label"] if "label" in labeled_data.columns else None,
                    )
                    # Re-attach label only
                    combined_data = pd.concat([selected_features, combined_data[["label"]]], axis=1)

                # 6. Data normalization (exclude label)
                if self.enable_data_normalization:
                    self.logger.info("📏 Normalizing data...")
                    normalized_data = await self.data_normalizer.normalize_data(
                        combined_data,
                    )
                    combined_data = normalized_data

            # 7. Autoencoder feature generation on features only
            self.logger.info("🤖 Generating autoencoder features...")
            if self.autoencoder_generator is not None:
                ae_input = combined_data.drop(columns=["label"], errors="ignore")
                autoencoder_features = self.autoencoder_generator.generate_features(
                    ae_input,
                    "vectorized_regime",
                    labeled_data["label"].values if "label" in labeled_data.columns else np.zeros(len(combined_data)),
                )
            else:
                self.logger.warning("Autoencoder generator not available, skipping autoencoder feature generation")
                autoencoder_features = combined_data.drop(columns=["label"], errors="ignore")

            # 8. Final data preparation (features + label only)
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
                    "feature_engineering_completed": self.advanced_feature_engineer is not None,
                    "labeling_completed": True,
                    "autoencoder_features_generated": self.autoencoder_generator is not None,
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
        """Combine engineered features and label. Excludes raw OHLCV from features."""
        try:
            # Ensure label column exists
            if "label" not in labeled_data.columns:
                labeled_data = labeled_data.copy()
                labeled_data["label"] = 0

            # Build a time-aligned features DataFrame from engineered features
            features_df = self._build_features_dataframe(advanced_features, labeled_data.index)

            # Drop raw OHLCV from features if somehow present
            ohlcv_cols = ["open", "high", "low", "close", "volume"]
            features_df = features_df.drop(columns=[c for c in ohlcv_cols if c in features_df.columns], errors="ignore")

            # Keep only numeric feature columns
            features_df = features_df.select_dtypes(include=[np.number])

            # Drop constant features pre-selection
            nunique = features_df.nunique()
            constant_cols = nunique[nunique <= 1].index.tolist()
            if constant_cols:
                features_df = features_df.drop(columns=constant_cols)

            # Only return engineered features + label (no raw OHLCV)
            combined = pd.concat([features_df, labeled_data[["label"]]], axis=1)
            return combined

        except Exception as e:
            self.logger.error(f"Error combining features and labels: {e}")
            # Fallback: return label only to avoid leaking raw OHLCV
            return labeled_data[["label"]] if "label" in labeled_data.columns else pd.DataFrame({"label": []})

    def _build_features_dataframe(self, advanced_features: dict[str, Any], index: pd.Index) -> pd.DataFrame:
        """Construct a time-aligned features DataFrame from a heterogeneous features dict.
        - Uses only per-row series/arrays aligned to the target index length
        - Skips scalar/global features to avoid constant columns across the dataset
        """
        try:
            if not advanced_features:
                return pd.DataFrame(index=index)

            features_df = pd.DataFrame(index=index)
            target_len = len(index)

            for name, value in advanced_features.items():
                # Pandas Series with same length
                if isinstance(value, pd.Series):
                    if len(value) == target_len:
                        # Align by index if possible, otherwise reindex by position
                        try:
                            aligned = value.reindex(index)
                        except Exception:
                            aligned = pd.Series(value.values, index=index)
                        features_df[name] = aligned
                    else:
                        continue
                # Numpy array or list with same length
                elif isinstance(value, (np.ndarray, list)):
                    if len(value) == target_len:
                        features_df[name] = pd.Series(value, index=index)
                    else:
                        continue
                # Other types (scalar, dict, etc.) are skipped to avoid constant leakage
                else:
                    continue

            return features_df
        except Exception as e:
            self.logger.error(f"Error building features DataFrame: {e}")
            return pd.DataFrame(index=index)

    def _remove_datetime_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove datetime columns to prevent dtype conflicts in ML training."""
        try:
            # Get columns with datetime dtype
            datetime_columns = []
            for col in data.columns:
                if data[col].dtype == 'datetime64[ns]' or 'datetime' in str(data[col].dtype).lower():
                    datetime_columns.append(col)
            
            # Remove datetime columns
            if datetime_columns:
                self.logger.info(f"Removing datetime columns: {datetime_columns}")
                data = data.drop(columns=datetime_columns)
            
            # Also remove any timestamp columns that might cause issues
            timestamp_columns = [col for col in data.columns if 'timestamp' in col.lower()]
            if timestamp_columns:
                self.logger.info(f"Removing timestamp columns: {timestamp_columns}")
                data = data.drop(columns=timestamp_columns)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error removing datetime columns: {e}")
            return data

    def _prepare_final_data_vectorized(
        self,
        autoencoder_features: pd.DataFrame,
        labeled_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Prepare final data as engineered/selected/AE features + label only."""
        try:
            # Ensure label column present
            labels = labeled_data[["label"]] if "label" in labeled_data.columns else pd.DataFrame({"label": [0] * len(autoencoder_features)}, index=autoencoder_features.index)

            # Combine engineered features (post-AE) with label only
            final_data = pd.concat([autoencoder_features, labels], axis=1)

            # Remove duplicate columns
            final_data = final_data.loc[:, ~final_data.columns.duplicated()]

            # Remove infinite values
            final_data = final_data.replace([np.inf, -np.inf], np.nan)

            # Fill remaining NaN values
            final_data = final_data.fillna(method="ffill").fillna(method="bfill").fillna(0)

            # Remove columns with all NaN values
            final_data = final_data.dropna(axis=1, how="all")

            return final_data

        except Exception as e:
            self.logger.error(f"Error preparing final data: {e}")
            # Fallback to AE features only
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
            
            # Ensure we preserve the original OHLCV columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in transformed_data.columns for col in required_columns):
                self.logger.warning(f"Missing required OHLCV columns. Available: {transformed_data.columns.tolist()}")
                return price_data
            
            # Calculate returns (first difference) - ADD to existing data, don't replace
            transformed_data["close_returns"] = transformed_data["close"].pct_change()
            transformed_data["open_returns"] = transformed_data["open"].pct_change()
            transformed_data["high_returns"] = transformed_data["high"].pct_change()
            transformed_data["low_returns"] = transformed_data["low"].pct_change()
            
            # Log transformation for variance stabilization - ADD to existing data
            transformed_data["close_log"] = np.log(transformed_data["close"])
            transformed_data["open_log"] = np.log(transformed_data["open"])
            transformed_data["high_log"] = np.log(transformed_data["high"])
            transformed_data["low_log"] = np.log(transformed_data["low"])
            
            # Remove trend using rolling mean - ADD to existing data
            window = 20
            transformed_data["close_detrended"] = (
                transformed_data["close"] - transformed_data["close"].rolling(window).mean()
            )
            
            # Ensure original OHLCV columns are still present
            self.logger.info(f"✅ Transformed price data shape: {transformed_data.shape}, columns: {transformed_data.columns.tolist()}")
            
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
            
            # Ensure we preserve the original volume column
            if "volume" not in transformed_data.columns:
                self.logger.warning(f"Missing volume column. Available: {transformed_data.columns.tolist()}")
                return volume_data
            
            # Log transformation - ADD to existing data, don't replace
            transformed_data["volume_log"] = np.log(transformed_data["volume"])
            
            # Remove trend - ADD to existing data
            window = 20
            transformed_data["volume_detrended"] = (
                transformed_data["volume"] - transformed_data["volume"].rolling(window).mean()
            )
            
            # Normalize volume - ADD to existing data
            transformed_data["volume_normalized"] = (
                transformed_data["volume"] / transformed_data["volume"].rolling(window).mean()
            )
            
            # Ensure original volume column is still present
            self.logger.info(f"✅ Transformed volume data shape: {transformed_data.shape}, columns: {transformed_data.columns.tolist()}")
            
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
    """Vectorized feature selector using multiple selection methods."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("VectorizedFeatureSelector")

        # Feature selection configuration
        self.feature_selection_config = config.get("feature_selection", {})
        self.vif_threshold = self.feature_selection_config.get("vif_threshold", 10.0)  # Increased from 5.0
        self.mutual_info_threshold = self.feature_selection_config.get("mutual_info_threshold", 0.001)  # Decreased from 0.01
        self.lightgbm_importance_threshold = self.feature_selection_config.get("lightgbm_importance_threshold", 0.001)  # Decreased from 0.01
        self.min_features_to_keep = self.feature_selection_config.get("min_features_to_keep", 10)  # New parameter
        self.correlation_threshold = self.feature_selection_config.get("correlation_threshold", 0.98)  # Increased from 0.95
        self.max_removal_percentage = self.feature_selection_config.get("max_removal_percentage", 0.3)
        
        # Enable/disable flags
        self.enable_constant_removal = self.feature_selection_config.get("enable_constant_removal", True)
        self.enable_correlation_removal = self.feature_selection_config.get("enable_correlation_removal", True)
        self.enable_vif_removal = self.feature_selection_config.get("enable_vif_removal", True)
        self.enable_mutual_info_removal = self.feature_selection_config.get("enable_mutual_info_removal", True)
        self.enable_importance_removal = self.feature_selection_config.get("enable_importance_removal", True)
        
        # Safety settings
        self.enable_safety_checks = self.feature_selection_config.get("enable_safety_checks", True)
        self.return_original_on_failure = self.feature_selection_config.get("return_original_on_failure", True)

    def _remove_datetime_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove datetime columns to prevent dtype conflicts in ML training."""
        try:
            # Get columns with datetime dtype
            datetime_columns = []
            for col in data.columns:
                if data[col].dtype == 'datetime64[ns]' or 'datetime' in str(data[col].dtype).lower():
                    datetime_columns.append(col)
            
            # Remove datetime columns
            if datetime_columns:
                self.logger.info(f"Removing datetime columns: {datetime_columns}")
                data = data.drop(columns=datetime_columns)
            
            # Also remove any timestamp columns that might cause issues
            timestamp_columns = [col for col in data.columns if 'timestamp' in col.lower()]
            if timestamp_columns:
                self.logger.info(f"Removing timestamp columns: {timestamp_columns}")
                data = data.drop(columns=timestamp_columns)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error removing datetime columns: {e}")
            return data

    async def select_optimal_features(
        self,
        data: pd.DataFrame,
        labels: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Select optimal features using vectorized operations."""
        try:
            self.logger.info("🎯 Starting feature selection...")
            initial_features = len(data.columns)
            original_data = data.copy()  # Keep original for fallback
            
            # Remove datetime columns first
            data = self._remove_datetime_columns(data)

            # Exclude label and raw OHLCV from selection (defense in depth)
            ohlcv_cols = ["open", "high", "low", "close", "volume"]
            data = data.drop(columns=["label"], errors="ignore")
            data = data.drop(columns=[c for c in ohlcv_cols if c in data.columns], errors="ignore")

            # Keep numeric columns only
            data = data.select_dtypes(include=[np.number])
            
            # Handle NaN values in feature selection
            self.logger.info("Handling NaN values in feature selection...")
            data = data.fillna(method="ffill").fillna(method="bfill").fillna(0)

            # Compute dynamic minimum features to keep based on initial dimensionality
            dynamic_min_features = max(self.min_features_to_keep, max(10, int(0.05 * max(1, initial_features))))

            # Remove constant features (if enabled)
            if self.enable_constant_removal:
                constant_features = []
                for col in data.columns:
                    if data[col].nunique() <= 1:
                        constant_features.append(col)
                
                if constant_features:
                    self.logger.info(f"Removed {len(constant_features)} constant features")
                    data = data.drop(columns=constant_features)

            # Safety check after constant removal
            if self.enable_safety_checks and len(data.columns) < dynamic_min_features:
                self.logger.warning(f"Too few features after constant removal ({len(data.columns)}). Skipping further selection.")
                return data

            # Remove highly correlated features (only if enabled and we have enough features)
            if (self.enable_correlation_removal and 
                len(data.columns) > dynamic_min_features and 
                len(data.columns) > 2):  # Need at least 2 features for correlation
                
                correlated_features = self._remove_correlated_features_vectorized(data)
                if correlated_features:
                    # Check removal percentage
                    removal_percentage = len(correlated_features) / len(data.columns)
                    if (removal_percentage <= self.max_removal_percentage and 
                        len(data.columns) - len(correlated_features) >= dynamic_min_features):
                        self.logger.info(f"Removed {len(correlated_features)} highly correlated features")
                        data = data.drop(columns=correlated_features)
                    else:
                        self.logger.info(f"Skipping correlated feature removal (removal %: {removal_percentage:.2f})")

            # Remove high VIF features (only if enabled and we have enough features)
            if (self.enable_vif_removal and 
                len(data.columns) > dynamic_min_features and 
                len(data.columns) > 2):  # Need at least 2 features for VIF
                
                high_vif_features = self._remove_high_vif_features_vectorized(data)
                if high_vif_features:
                    # Check removal percentage
                    removal_percentage = len(high_vif_features) / len(data.columns)
                    if (removal_percentage <= self.max_removal_percentage and 
                        len(data.columns) - len(high_vif_features) >= dynamic_min_features):
                        self.logger.info(f"Removed {len(high_vif_features)} high VIF features")
                        data = data.drop(columns=high_vif_features)
                    else:
                        self.logger.info(f"Skipping VIF feature removal (removal %: {removal_percentage:.2f})")

            # Remove low mutual information features (if enabled, labels provided, and we have enough features)
            if (self.enable_mutual_info_removal and 
                labels is not None and len(labels) > 0 and 
                len(data.columns) > dynamic_min_features):
                
                low_mi_features = self._remove_low_mutual_info_features_vectorized(data, labels)
                if low_mi_features:
                    # Check removal percentage
                    removal_percentage = len(low_mi_features) / len(data.columns)
                    if (removal_percentage <= self.max_removal_percentage and 
                        len(data.columns) - len(low_mi_features) >= dynamic_min_features):
                        self.logger.info(f"Removed {len(low_mi_features)} low mutual information features")
                        data = data.drop(columns=low_mi_features)
                    else:
                        self.logger.info(f"Skipping mutual info feature removal (removal %: {removal_percentage:.2f})")

            # Remove low importance features (if enabled, labels provided, and we have enough features)
            if (self.enable_importance_removal and 
                labels is not None and len(labels) > 0 and 
                len(data.columns) > dynamic_min_features):
                
                low_importance_features = self._remove_low_importance_features_vectorized(data, labels)
                if low_importance_features:
                    # Check removal percentage
                    removal_percentage = len(low_importance_features) / len(data.columns)
                    if (removal_percentage <= self.max_removal_percentage and 
                        len(data.columns) - len(low_importance_features) >= dynamic_min_features):
                        self.logger.info(f"Removed {len(low_importance_features)} low importance features")
                        data = data.drop(columns=low_importance_features)
                    else:
                        self.logger.info(f"Skipping importance feature removal (removal %: {removal_percentage:.2f})")

            final_features = len(data.columns)
            self.logger.info(f"Feature selection completed. Initial features: {initial_features}, Final features: {final_features}")
            
            # Final safety check
            if final_features == 0:
                self.logger.error("No features remaining after selection!")
                if self.return_original_on_failure:
                    self.logger.info("Returning original engineered features (sans OHLCV/label) as fallback.")
                    cleaned_original = original_data.drop(columns=["label"], errors="ignore")
                    cleaned_original = cleaned_original.drop(columns=[c for c in ["open","high","low","close","volume"] if c in cleaned_original.columns], errors="ignore")
                    cleaned_original = cleaned_original.select_dtypes(include=[np.number])
                    return self._remove_datetime_columns(cleaned_original)
                else:
                    raise ValueError("No features remaining after selection and fallback disabled")
            
            return data

        except Exception as e:
            self.logger.error(f"Error in feature selection: {e}")
            if self.return_original_on_failure:
                self.logger.info("Returning original engineered features as fallback due to error.")
                cleaned_original = original_data.drop(columns=["label"], errors="ignore")
                cleaned_original = cleaned_original.drop(columns=[c for c in ["open","high","low","close","volume"] if c in cleaned_original.columns], errors="ignore")
                cleaned_original = cleaned_original.select_dtypes(include=[np.number])
                return self._remove_datetime_columns(cleaned_original)
            else:
                raise

    def _remove_correlated_features_vectorized(self, data: pd.DataFrame) -> List[str]:
        """Remove highly correlated features using vectorized operations."""
        try:
            # Handle NaN values by imputing with median for correlation calculation
            from sklearn.impute import SimpleImputer
            
            imputer = SimpleImputer(strategy='median')
            data_imputed = pd.DataFrame(
                imputer.fit_transform(data),
                columns=data.columns,
                index=data.index
            )
            
            correlation_matrix = data_imputed.corr()
            upper_triangle = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            high_correlation = np.abs(correlation_matrix) > self.correlation_threshold
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
            from sklearn.impute import SimpleImputer
            
            # Handle NaN values by imputing with median
            imputer = SimpleImputer(strategy='median')
            data_imputed = pd.DataFrame(
                imputer.fit_transform(data),
                columns=data.columns,
                index=data.index
            )
            
            vif_scores = {}
            for col in data_imputed.columns:
                # Use other features to predict this feature
                other_cols = [c for c in data_imputed.columns if c != col]
                if len(other_cols) > 0:
                    X = data_imputed[other_cols]
                    y = data_imputed[col]
                    
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
            from sklearn.impute import SimpleImputer
            
            # Handle NaN values by imputing with median
            imputer = SimpleImputer(strategy='median')
            data_imputed = pd.DataFrame(
                imputer.fit_transform(data),
                columns=data.columns,
                index=data.index
            )
            
            # Calculate mutual information scores
            mi_scores = mutual_info_classif(data_imputed, labels, random_state=42)
            
            # Create feature importance dictionary
            feature_importance = dict(zip(data_imputed.columns, mi_scores))
            
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
            from sklearn.impute import SimpleImputer
            
            # Handle NaN values by imputing with median
            imputer = SimpleImputer(strategy='median')
            data_imputed = pd.DataFrame(
                imputer.fit_transform(data),
                columns=data.columns,
                index=data.index
            )
            
            # Train LightGBM model
            model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                verbose=-1,
            )
            model.fit(data_imputed, labels)
            
            # Get feature importance
            feature_importance = dict(zip(data_imputed.columns, model.feature_importances_))
            
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
            
            # Remove datetime columns first
            data = self._remove_datetime_columns(data)

            # Separate label to avoid scaling it
            label_series = None
            if "label" in data.columns:
                label_series = data["label"].copy()
                data = data.drop(columns=["label"])  # features only
            
            # Handle outliers
            data = self._clip_outliers_vectorized(data)
            
            # Apply robust scaling
            data = self._apply_robust_scaling_vectorized(data)

            # Re-attach label if present
            if label_series is not None:
                data["label"] = label_series
            
            self.logger.info("✅ Data normalization completed")
            return data

        except Exception as e:
            self.logger.error(f"Error normalizing data: {e}")
            return data

    def _remove_datetime_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove datetime columns to prevent dtype conflicts in ML training."""
        try:
            # Get columns with datetime dtype
            datetime_columns = []
            for col in data.columns:
                if data[col].dtype == 'datetime64[ns]' or 'datetime' in str(data[col].dtype).lower():
                    datetime_columns.append(col)
            
            # Remove datetime columns
            if datetime_columns:
                self.logger.info(f"Removing datetime columns: {datetime_columns}")
                data = data.drop(columns=datetime_columns)
            
            # Also remove any timestamp columns that might cause issues
            timestamp_columns = [col for col in data.columns if 'timestamp' in col.lower()]
            if timestamp_columns:
                self.logger.info(f"Removing timestamp columns: {timestamp_columns}")
                data = data.drop(columns=timestamp_columns)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error removing datetime columns: {e}")
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
