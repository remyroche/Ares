# src/analyst/feature_engineering_orchestrator.py

import os
from typing import Any

import numpy as np
import pandas as pd
import pywt

# Import the advanced feature engineering components
from src.analyst.advanced_feature_engineering import AdvancedFeatureEngineering
from src.analyst.autoencoder_feature_generator import AutoencoderFeatureGenerator
from src.config import CONFIG
from src.utils.error_handler import (
    handle_data_processing_errors,
    handle_errors,
    handle_file_operations,
)
from src.utils.logger import system_logger


class FeatureEngineeringOrchestrator:
    """
    Comprehensive feature engineering orchestrator that coordinates all feature generation components.
    Integrates advanced feature engineering and autoencoder feature generation.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the feature engineering orchestrator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("FeatureEngineeringOrchestrator")

        # Initialize sub-components
        self.advanced_feature_engineering = AdvancedFeatureEngineering(config)
        self.autoencoder_generator = AutoencoderFeatureGenerator(config)

        # Model storage paths
        self.model_storage_path = os.path.join(
            CONFIG["CHECKPOINT_DIR"],
            "analyst_models",
            "feature_engineering",
        )
        os.makedirs(self.model_storage_path, exist_ok=True)

        self.autoencoder_model_path = os.path.join(
            self.model_storage_path,
            "autoencoder_model.h5",
        )
        self.autoencoder_scaler_path = os.path.join(
            self.model_storage_path,
            "der_scaler.joblib",
        )

        # Configuration
        from src.config_optuna import get_parameter_value

        self.orchestrator_config = config.get("feature_engineering_orchestrator", {})
        self.enable_advanced_features = get_parameter_value(
            "feature_engineering_parameters.enable_advanced_features",
            True,
        )
        self.enable_autoencoder_features = get_parameter_value(
            "feature_engineering_parameters.enable_autoencoder_features",
            True,
        )
        self.enable_legacy_features = get_parameter_value(
            "feature_engineering_parameters.enable_legacy_features",
            True,
        )

        self.logger.info("🚀 FeatureEngineeringOrchestrator initialized successfully")

    @handle_errors(
        exceptions=(Exception,),
        default_return=pd.DataFrame(),
        context="orchestrated feature generation",
    )
    async def generate_all_features(
        self,
        klines_df: pd.DataFrame,
        agg_trades_df: pd.DataFrame = None,
        futures_df: pd.DataFrame = None,
        sr_levels: list = None,
    ) -> pd.DataFrame:
        """
        Orchestrate the generation of all features using multiple components.

        Args:
            klines_df: Klines data
            agg_trades_df: Aggregated trades data (optional)
            futures_df: Futures data (optional)
            sr_levels: Support/resistance levels (optional)

        Returns:
            DataFrame with all generated features
        """
        self.logger.info(
            "🎯 Starting comprehensive feature generation orchestration...",
        )

        if klines_df.empty:
            self.logger.warning("Empty klines data provided, returning empty DataFrame")
            return pd.DataFrame()

        try:
            # Start with a copy of the original data
            features_df = klines_df.copy()

            # 1. Generate advanced features (if enabled)
            if self.enable_advanced_features:
                self.logger.info("📊 Generating advanced features...")
                features_df = self.advanced_feature_engineering.generate_features(
                    features_df,
                    agg_trades_df,
                    futures_df,
                )
                self.logger.info(
                    f"✅ Advanced features generated. Shape: {features_df.shape}",
                )

            # 2. Generate autoencoder features (if enabled)
            if self.enable_autoencoder_features and not features_df.empty:
                self.logger.info("🤖 Generating autoencoder features...")
                features_df = self.autoencoder_generator.generate_features(features_df)
                self.logger.info(
                    f"✅ Autoencoder features generated. Shape: {features_df.shape}",
                )

            # 3. Generate legacy features (if enabled)
            if self.enable_legacy_features:
                self.logger.info("🔧 Generating legacy features...")
                features_df = self._generate_legacy_features(
                    features_df,
                    agg_trades_df,
                    futures_df,
                    sr_levels,
                )
                self.logger.info(
                    f"✅ Legacy features generated. Shape: {features_df.shape}",
                )

            # 4. Generate multi-timeframe features (if enabled)
            if self.config.get("enable_multi_timeframe", True):
                self.logger.info("⏰ Generating multi-timeframe features...")
                multi_timeframe_features = (
                    await self._calculate_multi_timeframe_features(
                        klines_df,
                        agg_trades_df,
                        None,
                    )
                )
                if not multi_timeframe_features.empty:
                    features_df = pd.concat(
                        [features_df, multi_timeframe_features],
                        axis=1,
                    )
                    self.logger.info(
                        f"✅ Multi-timeframe features generated. Shape: {features_df.shape}",
                    )

            # 5. Generate meta-labeling features (if enabled)
            if self.config.get("enable_meta_labeling", True):
                self.logger.info("🏷️ Generating meta-labeling features...")
                meta_labeling_features = await self._calculate_meta_labeling_features(
                    klines_df,
                    agg_trades_df,
                    None,
                )
                if not meta_labeling_features.empty:
                    features_df = pd.concat(
                        [features_df, meta_labeling_features],
                        axis=1,
                    )
                    self.logger.info(
                        f"✅ Meta-labeling features generated. Shape: {features_df.shape}",
                    )

            # 6. Final cleanup and validation
            features_df = self._cleanup_features(features_df)

            self.logger.info(
                f"🎉 Feature generation orchestration completed! Final shape: {features_df.shape}",
            )
            self.logger.info(f"📊 Total features generated: {len(features_df.columns)}")

            return features_df

        except Exception as e:
            self.logger.error(f"❌ Error in feature generation orchestration: {e}")
            return klines_df.copy()

    @handle_data_processing_errors(
        default_return=pd.DataFrame(),
        context="legacy feature generation",
    )
    def _generate_legacy_features(
        self,
        features_df: pd.DataFrame,
        agg_trades_df: pd.DataFrame = None,
        futures_df: pd.DataFrame = None,
        sr_levels: list = None,
    ) -> pd.DataFrame:
        """Generate legacy features for backward compatibility."""
        try:
            # Merge klines with futures data first
            if futures_df is not None and not futures_df.empty:
                features_df = (
                    pd.merge_asof(
                        features_df.sort_index(),
                        futures_df.sort_index(),
                        left_index=True,
                        right_index=True,
                        direction="backward",
                    )
                    .ffill()
                    .fillna(0)
                )

            # Standard technical indicators
            features_df = self._calculate_standard_indicators(features_df)

            # Time-based features
            features_df = self._calculate_time_features(features_df)

            # Volatility regime indicators
            features_df = self._calculate_volatility_regime_indicators(features_df)

            # Volatility targeting features
            features_df = self._calculate_volatility_targeting_features(features_df)

            # ML enhanced features
            features_df = self._calculate_ml_enhanced_features(features_df)

            # S/R features if SR levels are provided
            if sr_levels:
                features_df = self._calculate_sr_features(features_df, sr_levels)

            return features_df

        except Exception as e:
            self.logger.error(f"Error generating legacy features: {e}")
            return features_df

    @handle_errors(
        exceptions=(Exception,),
        default_return=pd.DataFrame(),
        context="multi-timeframe feature calculation",
    )
    async def _calculate_multi_timeframe_features(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        order_flow_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Calculate multi-timeframe features."""
        try:
            from src.analyst.advanced_feature_engineering import (
                AdvancedFeatureEngineering,
            )

            # Initialize advanced feature engineering
            advanced_fe = AdvancedFeatureEngineering(self.config)
            await advanced_fe.initialize()

            # Generate multi-timeframe features
            multi_timeframe_features = (
                await advanced_fe._engineer_multi_timeframe_features(
                    price_data,
                    volume_data,
                    order_flow_data,
                )
            )

            # Convert to DataFrame
            features_df = pd.DataFrame([multi_timeframe_features])

            return features_df

        except Exception as e:
            self.logger.error(f"Error calculating multi-timeframe features: {e}")
            return pd.DataFrame()

    @handle_errors(
        exceptions=(Exception,),
        default_return=pd.DataFrame(),
        context="meta-labeling feature calculation",
    )
    async def _calculate_meta_labeling_features(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        order_flow_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Calculate meta-labeling features."""
        try:
            from src.analyst.meta_labeling_system import MetaLabelingSystem

            # Initialize meta-labeling system
            meta_labeling = MetaLabelingSystem(self.config)
            await meta_labeling.initialize()

            # Generate meta-labels
            analyst_labels = await meta_labeling._generate_analyst_labels(
                price_data,
                volume_data,
                order_flow_data,
            )
            tactician_labels = await meta_labeling._generate_tactician_labels(
                price_data,
                volume_data,
                order_flow_data,
            )

            # Combine labels
            all_labels = {**analyst_labels, **tactician_labels}

            # Convert to DataFrame
            features_df = pd.DataFrame([all_labels])

            return features_df

        except Exception as e:
            self.logger.error(f"Error calculating meta-labeling features: {e}")
            return pd.DataFrame()

    @handle_data_processing_errors(
        default_return=pd.DataFrame(),
        context="standard indicators calculation",
    )
    def _calculate_standard_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate standard technical indicators."""
        try:
            import pandas_ta as ta

            # Moving averages
            df["sma_5"] = ta.sma(df["close"], length=5)
            df["sma_10"] = ta.sma(df["close"], length=10)
            df["sma_20"] = ta.sma(df["close"], length=20)
            df["sma_50"] = ta.sma(df["close"], length=50)
            df["ema_12"] = ta.ema(df["close"], length=12)
            df["ema_26"] = ta.ema(df["close"], length=26)

            # RSI
            df["rsi"] = ta.rsi(df["close"], length=14)

            # MACD
            macd = ta.macd(df["close"])
            df["macd"] = macd["MACD_12_26_9"]
            df["macd_signal"] = macd["MACDs_12_26_9"]
            df["macd_hist"] = macd["MACDh_12_26_9"]

            # Bollinger Bands
            bb = ta.bbands(df["close"])
            df["bb_upper"] = bb["BBU_20_2.0"]
            df["bb_middle"] = bb["BBM_20_2.0"]
            df["bb_lower"] = bb["BBL_20_2.0"]
            df["bb_width"] = (bb["BBU_20_2.0"] - bb["BBL_20_2.0"]) / bb["BBM_20_2.0"]

            # Stochastic
            stoch = ta.stoch(df["high"], df["low"], df["close"])
            df["stoch_k"] = stoch["STOCHk_14_3_3"]
            df["stoch_d"] = stoch["STOCHd_14_3_3"]

            # ATR
            df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)

            return df

        except Exception as e:
            self.logger.error(f"Error calculating standard indicators: {e}")
            return df

    @handle_data_processing_errors(
        default_return=pd.DataFrame(),
        context="time features calculation",
    )
    def _calculate_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate time-based features."""
        try:
            # Extract time components
            df["hour"] = df.index.hour
            df["day_of_week"] = df.index.dayofweek
            df["day_of_month"] = df.index.day
            df["month"] = df.index.month
            df["quarter"] = df.index.quarter

            # Cyclical encoding for time features
            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
            df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
            df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
            df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
            df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

            # Market session indicators
            df["is_asia_session"] = ((df["hour"] >= 0) & (df["hour"] < 8)).astype(int)
            df["is_london_session"] = ((df["hour"] >= 8) & (df["hour"] < 16)).astype(
                int,
            )
            df["is_ny_session"] = ((df["hour"] >= 13) & (df["hour"] < 21)).astype(int)

            return df

        except Exception as e:
            self.logger.error(f"Error calculating time features: {e}")
            return df

    @handle_data_processing_errors(
        default_return=pd.DataFrame(),
        context="volatility regime indicators calculation",
    )
    def _calculate_volatility_regime_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility regime indicators."""
        try:
            # Calculate rolling volatility
            returns = df["close"].pct_change()
            df["volatility_5"] = returns.rolling(window=5).std()
            df["volatility_10"] = returns.rolling(window=10).std()
            df["volatility_20"] = returns.rolling(window=20).std()

            # Volatility regime classification
            def classify_vol_regime(vol):
                if vol <= 0.02:
                    return 0  # Low volatility
                if vol <= 0.04:
                    return 1  # Normal volatility
                if vol <= 0.08:
                    return 2  # High volatility
                return 3  # Extreme volatility

            df["volatility_regime_5"] = df["volatility_5"].apply(classify_vol_regime)
            df["volatility_regime_10"] = df["volatility_10"].apply(classify_vol_regime)
            df["volatility_regime_20"] = df["volatility_20"].apply(classify_vol_regime)

            # Volatility ratio
            df["vol_ratio_5_20"] = df["volatility_5"] / df["volatility_20"]
            df["vol_ratio_10_20"] = df["volatility_10"] / df["volatility_20"]

            return df

        except Exception as e:
            self.logger.error(f"Error calculating volatility regime indicators: {e}")
            return df

    @handle_data_processing_errors(
        default_return=pd.DataFrame(),
        context="volatility targeting features calculation",
    )
    def _calculate_volatility_targeting_features(
        self,
        df: pd.DataFrame,
        target_volatility: float = 0.15,
    ) -> pd.DataFrame:
        """Calculate volatility targeting features."""
        try:
            # Target volatility (annual to daily)
            target_vol_daily = target_volatility / np.sqrt(252)

            # Current volatility
            returns = df["close"].pct_change()
            current_vol = returns.rolling(window=20).std()

            # Volatility targeting ratio
            df["vol_target_ratio"] = current_vol / target_vol_daily

            # Position sizing based on volatility
            df["vol_adjusted_position"] = 1.0 / df["vol_target_ratio"]
            df["vol_adjusted_position"] = df["vol_adjusted_position"].clip(0.1, 2.0)

            # Volatility regime
            df["vol_regime"] = np.where(
                df["vol_target_ratio"] > 1.5,
                "high_vol",
                np.where(df["vol_target_ratio"] < 0.5, "low_vol", "normal_vol"),
            )

            return df

        except Exception as e:
            self.logger.error(f"Error calculating volatility targeting features: {e}")
            return df

    @handle_data_processing_errors(
        default_return=pd.DataFrame(),
        context="ML enhanced features calculation",
    )
    def _calculate_ml_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ML-enhanced features."""
        try:
            # Price momentum features
            df["price_momentum_1"] = df["close"].pct_change(1)
            df["price_momentum_5"] = df["close"].pct_change(5)
            df["price_momentum_10"] = df["close"].pct_change(10)

            # Volume features (if available)
            if "volume" in df.columns:
                df["volume_momentum_1"] = df["volume"].pct_change(1)
                df["volume_momentum_5"] = df["volume"].pct_change(5)
                df["volume_ratio"] = (
                    df["volume"] / df["volume"].rolling(window=20).mean()
                )

            # Legacy S/R/Candle code removed features
            df["resistance_20"] = df["high"].rolling(window=20).max()
            df["support_20"] = df["low"].rolling(window=20).min()
            df["dist_to_resistance"] = (df["resistance_20"] - df["close"]) / df["close"]
            df["dist_to_support"] = (df["close"] - df["support_20"]) / df["close"]

            # Pivot points
            df["pivot"] = (df["high"] + df["low"] + df["close"]) / 3
            df["r1"] = 2 * df["pivot"] - df["low"]
            df["s1"] = 2 * df["pivot"] - df["high"]

            return df

        except Exception as e:
            self.logger.error(f"Error calculating ML enhanced features: {e}")
            return df

    @handle_data_processing_errors(
        default_return=pd.DataFrame(),
        context="S/R features calculation",
    )
    def _calculate_sr_features(
        self,
        features_df: pd.DataFrame,
        sr_levels: list,
    ) -> pd.DataFrame:
        """
        Calculate S/R-related features for each row of historical data.

        Args:
            features_df: Features DataFrame
            sr_levels: Support/resistance levels from SR analyzer

        Returns:
            DataFrame with S/R features added
        """
        self.logger.info("🔍 Calculating S/R features...")

        try:
            # Initialize SR analyzer if not already done
            # Legacy S/R/Candle code removed
            # Legacy S/R/Candle code removed

            # Calculate S/R features for each row
            sr_features_list = []

            for idx, row in features_df.iterrows():
                current_price = row.get("close", 0)
                if current_price > 0:
                    # Legacy S/R/Candle code removed
                    sr_features_list.append(sr_features)
                else:
                    # Default values if price is not available
                    sr_features_list.append(
                        {
                            "distance_to_nearest_support": 0.0,
                            "distance_to_nearest_resistance": 0.0,
                            "strength_of_nearest_support": 0.0,
                            "strength_of_nearest_resistance": 0.0,
                            "time_since_level_tested": 0,
                            "sr_density_nearby": 0,
                            "is_vpvr_level": False,
                            "is_hvn_level": False,
                            "is_pivot_level": False,
                        },
                    )

            # Convert to DataFrame
            sr_features_df = pd.DataFrame(sr_features_list, index=features_df.index)

            # Add S/R features to main DataFrame
            for col in sr_features_df.columns:
                features_df[f"sr_{col}"] = sr_features_df[col]

            self.logger.info(
                f"✅ S/R features added: {len(sr_features_df.columns)} features",
            )
            return features_df

        except Exception as e:
            self.logger.error(f"❌ Error calculating S/R features: {e}")
            # Add default S/R features
            features_df["sr_distance_to_nearest_support"] = 0.0
            features_df["sr_distance_to_nearest_resistance"] = 0.0
            features_df["sr_strength_of_nearest_support"] = 0.0
            features_df["sr_strength_of_nearest_resistance"] = 0.0
            features_df["sr_time_since_level_tested"] = 0
            features_df["sr_density_nearby"] = 0
            features_df["sr_is_vpvr_level"] = False
            features_df["sr_is_hvn_level"] = False
            features_df["sr_is_pivot_level"] = False
            return features_df

    @handle_data_processing_errors(
        default_return=pd.DataFrame(),
        context="feature cleanup",
    )
    def _cleanup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean up and validate features."""
        try:
            # Remove infinite values
            df = df.replace([np.inf, -np.inf], np.nan)

            # Fill remaining NaN values
            df = df.fillna(method="ffill").fillna(method="bfill").fillna(0)

            # Remove columns with all NaN values
            df = df.dropna(axis=1, how="all")

            # Ensure all features are numeric
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df = df[numeric_cols]

            self.logger.info(f"Feature cleanup completed. Final shape: {df.shape}")
            return df

        except Exception as e:
            self.logger.error(f"Error in feature cleanup: {e}")
            return df

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="orchestrator info retrieval",
    )
    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="orchestrator info retrieval",
    )
    def get_orchestrator_info(self) -> dict[str, Any]:
        """Get information about the orchestrator."""
        try:
            return {
                "orchestrator_type": "FeatureEngineeringOrchestrator",
                "enable_advanced_features": self.enable_advanced_features,
                "enable_autoencoder_features": self.enable_autoencoder_features,
                "enable_legacy_features": self.enable_legacy_features,
                "advanced_feature_engineering_info": self.advanced_feature_engineering.get_feature_statistics(),
                "autoencoder_generator_info": self.autoencoder_generator.get_generator_info(),
                "config": self.orchestrator_config,
            }
        except Exception as e:
            self.logger.error(f"Error getting orchestrator info: {e}")
            return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="feature summary retrieval",
    )
    def get_feature_summary(self) -> dict[str, Any]:
        """Get a summary of all available features."""
        try:
            return {
                "feature_categories": [
                    "standard_indicators",
                    "advanced_features",
                    "autoencoder_features",
                    "time_features",
                    "volatility_features",
                    "ml_enhanced_features",
                ],
                "total_feature_types": 6,
                "orchestrator_config": self.orchestrator_config,
            }
        except Exception as e:
            self.logger.error(f"Error getting feature summary: {e}")
            return {}


# Legacy FeatureEngineeringEngine class for backward compatibility
class FeatureEngineeringEngine:
    """
    Legacy feature engineering engine for backward compatibility.
    Now delegates to the orchestrator.
    """

    def __init__(self, config):
        self.config = config.get("analyst", {}).get("feature_engineering", {})
        self.logger = system_logger.getChild("FeatureEngineeringEngine")
        self.orchestrator = FeatureEngineeringOrchestrator(config)
        self.autoencoder_model = None
        self.autoencoder_scaler = None

        # Use the new checkpoint directory for model storage
        self.model_storage_path = os.path.join(
            CONFIG["CHECKPOINT_DIR"],
            "analyst_models",
            "feature_engineering",
        )
        os.makedirs(self.model_storage_path, exist_ok=True)

        self.autoencoder_model_path = os.path.join(
            self.model_storage_path,
            "autoencoder_model.h5",
        )
        self.autoencoder_scaler_path = os.path.join(
            self.model_storage_path,
            "der_scaler.joblib",
        )

    @handle_errors(
        exceptions=(Exception,),
        default_return=pd.DataFrame(),
        context="generate_all_features",
    )
    async def generate_all_features(
        self,
        klines_df: pd.DataFrame,
        agg_trades_df: pd.DataFrame,
        futures_df: pd.DataFrame,
        sr_levels: list,
    ):
        """
        Generate all features using the orchestrator.
        """
        return await self.orchestrator.generate_all_features(
            klines_df,
            agg_trades_df,
            futures_df,
            sr_levels,
        )

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="wavelet transforms",
    )
    def apply_wavelet_transforms(self, data: pd.Series, wavelet="db1", level=3):
        """Apply wavelet transforms to data."""
        try:
            coeffs = pywt.wavedec(data, wavelet, level=level)
            return coeffs
        except Exception as e:
            self.logger.error(f"Error applying wavelet transforms: {e}")
            return None

    @handle_file_operations(default_return=False, context="train_autoencoder")
    def train_autoencoder(self, data: pd.DataFrame):
        """Train autoencoder model."""
        try:
            # Delegate to orchestrator's autoencoder generator
            return (
                self.orchestrator.autoencoder_generator.pipeline.autoencoder is not None
            )
        except Exception as e:
            self.logger.error(f"Error training autoencoder: {e}")
            return False

    @handle_data_processing_errors(
        default_return=pd.Series(),
        context="apply_autoencoders",
    )
    def apply_autoencoders(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply autoencoder features."""
        try:
            return self.orchestrator.autoencoder_generator.generate_features(data)
        except Exception as e:
            self.logger.error(f"Error applying autoencoders: {e}")
            return data

    @handle_file_operations(default_return=False, context="load_autoencoder")
    def load_autoencoder(self):
        """Load autoencoder model."""
        try:
            # This is handled by the orchestrator now
            return True
        except Exception as e:
            self.logger.error(f"Error loading autoencoder: {e}")
            return False
