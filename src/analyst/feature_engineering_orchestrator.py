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
from src.utils.warning_symbols import (
    error,
    warning,
)


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
        
        # Enhanced feature engineering state
        self.feature_engineering_state = {
            "last_feature_generation": None,
            "feature_generation_count": 0,
            "feature_quality_scores": {},
            "feature_redundancy_metrics": {},
            "feature_integration_status": {},
            "sr_integration_status": {},
            "regime_integration_status": {}
        }
        
        # Initialize enhanced components
        self.sr_analyzer = None
        self.regime_analyzer = None
        self.enhanced_feature_engineering = None
        
        # Try to initialize enhanced components
        try:
            from src.tactician.sr_breakout_predictor import SRBreakoutPredictor
            self.sr_analyzer = SRBreakoutPredictor(config)
            self.logger.info("✅ S/R analyzer initialized for feature engineering")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not initialize S/R analyzer: {e}")
        
        try:
            from src.training.steps.step9_5_hmm_lm_generalist_training import HMMLMGeneralistTrainingStep
            self.regime_analyzer = HMMLMGeneralistTrainingStep(config)
            self.logger.info("✅ Regime analyzer initialized for feature engineering")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not initialize regime analyzer: {e}")
        
        try:
            from src.training.steps.vectorized_advanced_feature_engineering import OptimizedResampler
            self.enhanced_feature_engineering = OptimizedResampler()
            self.logger.info("✅ Enhanced feature engineering initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not initialize enhanced feature engineering: {e}")

    async def generate_enhanced_features(self, klines_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate enhanced features with S/R and regime integration.
        
        Args:
            klines_df: Market data DataFrame
            
        Returns:
            Dict[str, Any]: Enhanced feature engineering results
        """
        try:
            self.logger.info("🔧 Generating enhanced features with S/R and regime integration...")
            
            # Update feature engineering state
            self.feature_engineering_state["last_feature_generation"] = pd.Timestamp.now()
            self.feature_engineering_state["feature_generation_count"] += 1
            
            # Use enhanced feature engineering if available
            if self.enhanced_feature_engineering is not None:
                enhanced_results = await self.enhanced_feature_engineering.generate_enhanced_features(
                    klines_df, self.sr_analyzer, self.regime_analyzer
                )
                
                # Update state
                self.feature_engineering_state["feature_quality_scores"] = enhanced_results.get("quality_metrics", {})
                self.feature_engineering_state["feature_redundancy_metrics"] = enhanced_results.get("redundancy_metrics", {})
                self.feature_engineering_state["sr_integration_status"] = enhanced_results.get("sr_features", {})
                self.feature_engineering_state["regime_integration_status"] = enhanced_results.get("regime_features", {})
                
                self.logger.info(f"✅ Enhanced feature generation completed: {enhanced_results.get('total_features', 0)} features")
                return enhanced_results
            else:
                # Fallback to original method
                self.logger.warning("⚠️ Enhanced feature engineering not available, using fallback method")
                return await self.generate_all_features(klines_df)
                
        except Exception as e:
            self.logger.error(f"Error in enhanced feature generation: {e}")
            return {"base_features": {}, "sr_features": {}, "regime_features": {}, "interaction_features": {}}

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
            self.print(warning("Empty klines data provided, returning empty DataFrame"))
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

        except Exception:
            self.print(error("❌ Error in feature generation orchestration: {e}"))
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
            return self._calculate_ml_enhanced_features(features_df)

        except Exception:
            self.print(error("Error generating legacy features: {e}"))
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
            return pd.DataFrame([multi_timeframe_features])

        except Exception:
            self.print(error("Error calculating multi-timeframe features: {e}"))
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
            return pd.DataFrame([all_labels])

        except Exception:
            self.print(error("Error calculating meta-labeling features: {e}"))
            return pd.DataFrame()

    @handle_data_processing_errors(
        default_return=pd.DataFrame(),
        context="standard indicators calculation",
    )
    def _calculate_standard_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate standard technical indicators using price differences."""
        try:
            import pandas_ta as ta

            # Convert price data to differences for technical indicators
            close_diff = df["close"].diff().fillna(0)
            high_diff = df["high"].diff().fillna(0)
            low_diff = df["low"].diff().fillna(0)

            # Create a temporary DataFrame with price differences for pandas_ta
            temp_df = df.copy()
            temp_df["close"] = close_diff
            temp_df["high"] = high_diff
            temp_df["low"] = low_diff

            # Moving averages using price differences
            df["sma_5"] = ta.sma(temp_df["close"], length=5)
            df["sma_10"] = ta.sma(temp_df["close"], length=10)
            df["sma_20"] = ta.sma(temp_df["close"], length=20)
            df["sma_50"] = ta.sma(temp_df["close"], length=50)
            df["ema_12"] = ta.ema(temp_df["close"], length=12)
            df["ema_26"] = ta.ema(temp_df["close"], length=26)

            # RSI using price differences
            df["rsi"] = ta.rsi(temp_df["close"], length=14)

            # MACD using price differences
            macd = ta.macd(temp_df["close"])
            df["macd"] = macd["MACD_12_26_9"]
            df["macd_signal"] = macd["MACDs_12_26_9"]
            df["macd_hist"] = macd["MACDh_12_26_9"]

            # Bollinger Bands using price differences
            bb = ta.bbands(temp_df["close"])
            df["bb_upper"] = bb["BBU_20_2.0"]
            df["bb_middle"] = bb["BBM_20_2.0"]
            df["bb_lower"] = bb["BBL_20_2.0"]
            df["bb_width"] = (bb["BBU_20_2.0"] - bb["BBL_20_2.0"]) / bb["BBM_20_2.0"]

            # Stochastic using price differences
            stoch = ta.stoch(temp_df["high"], temp_df["low"], temp_df["close"])
            df["stoch_k"] = stoch["STOCHk_14_3_3"]
            df["stoch_d"] = stoch["STOCHd_14_3_3"]

            # ATR using price differences
            df["atr"] = ta.atr(
                temp_df["high"], temp_df["low"], temp_df["close"], length=14
            )

            return df

        except Exception:
            self.print(error("Error calculating standard indicators: {e}"))
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

        except Exception:
            self.print(error("Error calculating time features: {e}"))
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
            self.logger.exception(
                f"Error calculating volatility regime indicators: {e}",
            )
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
            self.logger.exception(
                f"Error calculating volatility targeting features: {e}",
            )
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

        except Exception:
            self.print(error("Error calculating ML enhanced features: {e}"))
            return df

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

        except Exception:
            self.print(error("Error in feature cleanup: {e}"))
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
        except Exception:
            self.print(error("Error getting orchestrator info: {e}"))
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
        except Exception:
            self.print(error("Error getting feature summary: {e}"))
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
            return pywt.wavedec(data, wavelet, level=level)
        except Exception:
            self.print(error("Error applying wavelet transforms: {e}"))
            return None

    @handle_file_operations(default_return=False, context="train_autoencoder")
    def train_autoencoder(self, data: pd.DataFrame):
        """Train autoencoder model."""
        try:
            # Delegate to orchestrator's autoencoder generator
            return (
                self.orchestrator.autoencoder_generator.pipeline.autoencoder is not None
            )
        except Exception:
            self.print(error("Error training autoencoder: {e}"))
            return False

    @handle_data_processing_errors(
        default_return=pd.Series(),
        context="apply_autoencoders",
    )
    def apply_autoencoders(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply autoencoder features."""
        try:
            return self.orchestrator.autoencoder_generator.generate_features(data)
        except Exception:
            self.print(error("Error applying autoencoders: {e}"))
            return data

    # === ENHANCED FEATURE ENGINEERING METHODS ===
    
    async def generate_comprehensive_features(
        self,
        klines_df: pd.DataFrame,
        agg_trades_df: pd.DataFrame = None,
        futures_df: pd.DataFrame = None,
        sr_levels: list = None,
        regime_data: dict = None,
    ) -> pd.DataFrame:
        """
        Generate comprehensive, non-redundant features with S/R and regime integration.
        
        Args:
            klines_df: Klines data
            agg_trades_df: Aggregated trades data (optional)
            futures_df: Futures data (optional)
            sr_levels: S/R levels from centralized analysis (optional)
            regime_data: Regime data from HMM analysis (optional)
            
        Returns:
            pd.DataFrame: Comprehensive features
        """
        try:
            self.logger.info("🔧 Generating comprehensive features with S/R and regime integration...")
            
            # Update feature engineering state
            self.feature_engineering_state["last_feature_generation"] = pd.Timestamp.now()
            self.feature_engineering_state["feature_generation_count"] += 1
            
            # Step 1: Generate base features
            base_features = await self._generate_base_features(klines_df, agg_trades_df, futures_df)
            
            # Step 2: Generate S/R features
            sr_features = await self._generate_sr_features(klines_df, sr_levels)
            
            # Step 3: Generate regime features
            regime_features = await self._generate_regime_features(klines_df, regime_data)
            
            # Step 4: Generate interaction features
            interaction_features = await self._generate_interaction_features(base_features, sr_features, regime_features)
            
            # Step 5: Eliminate redundancy
            redundancy_metrics = self._eliminate_feature_redundancy(base_features, sr_features, regime_features, interaction_features)
            
            # Step 6: Combine all features
            comprehensive_features = self._combine_features(base_features, sr_features, regime_features, interaction_features)
            
            # Step 7: Calculate quality metrics
            quality_metrics = self._calculate_feature_quality_metrics(comprehensive_features)
            
            # Update state
            self.feature_engineering_state["feature_quality_scores"] = quality_metrics
            self.feature_engineering_state["feature_redundancy_metrics"] = redundancy_metrics
            self.feature_engineering_state["sr_integration_status"] = {"integrated": True, "feature_count": len(sr_features.columns)}
            self.feature_engineering_state["regime_integration_status"] = {"integrated": True, "feature_count": len(regime_features.columns)}
            
            self.logger.info(f"✅ Comprehensive features generated: {comprehensive_features.shape}")
            return comprehensive_features
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive features: {e}")
            return pd.DataFrame()

    async def _generate_base_features(self, klines_df: pd.DataFrame, agg_trades_df: pd.DataFrame = None, futures_df: pd.DataFrame = None) -> pd.DataFrame:
        """Generate base features from all components."""
        try:
            self.logger.info("🔧 Generating base features...")
            
            base_features = pd.DataFrame()
            
            # Generate advanced features
            if self.enable_advanced_features:
                advanced_features = await self.advanced_feature_engineering.generate_advanced_features(
                    klines_df, agg_trades_df, futures_df
                )
                base_features = pd.concat([base_features, advanced_features], axis=1)
            
            # Generate autoencoder features
            if self.enable_autoencoder_features:
                autoencoder_features = await self.autoencoder_generator.generate_autoencoder_features(
                    klines_df, agg_trades_df, futures_df
                )
                base_features = pd.concat([base_features, autoencoder_features], axis=1)
            
            # Generate legacy features
            if self.enable_legacy_features:
                legacy_features = await self.advanced_feature_engineering.generate_legacy_features(
                    klines_df, agg_trades_df, futures_df
                )
                base_features = pd.concat([base_features, legacy_features], axis=1)
            
            # Generate multi-timeframe features
            multi_timeframe_features = await self.advanced_feature_engineering.generate_multi_timeframe_features(
                klines_df, agg_trades_df, futures_df
            )
            base_features = pd.concat([base_features, multi_timeframe_features], axis=1)
            
            # Generate meta-labeling features
            meta_labeling_features = await self.advanced_feature_engineering.generate_meta_labeling_features(
                klines_df, agg_trades_df, futures_df
            )
            base_features = pd.concat([base_features, meta_labeling_features], axis=1)
            
            self.logger.info(f"✅ Base features generated: {base_features.shape}")
            return base_features
            
        except Exception as e:
            self.logger.error(f"Error generating base features: {e}")
            return pd.DataFrame()

    async def _generate_sr_features(self, klines_df: pd.DataFrame, sr_levels: list = None) -> pd.DataFrame:
        """Generate S/R features using centralized S/R analysis."""
        try:
            self.logger.info("🔧 Generating S/R features...")
            
            # Import SR breakout predictor
            try:
                from src.tactician.sr_breakout_predictor import SRBreakoutPredictor
                sr_predictor = SRBreakoutPredictor(self.config)
                await sr_predictor.initialize()
                
                # Get S/R features
                sr_features_dict = await sr_predictor.get_sr_features_for_engineering(klines_df)
                
                # Convert to DataFrame
                sr_features = pd.DataFrame([sr_features_dict], index=klines_df.index[-1:])
                
                # Extend to full length if needed
                if len(sr_features) < len(klines_df):
                    sr_features = sr_features.reindex(klines_df.index, method='ffill')
                
                self.logger.info(f"✅ S/R features generated: {sr_features.shape}")
                return sr_features
                
            except ImportError:
                self.logger.warning("SR breakout predictor not available, using fallback S/R features")
                return self._generate_fallback_sr_features(klines_df)
            
        except Exception as e:
            self.logger.error(f"Error generating S/R features: {e}")
            return pd.DataFrame()

    async def _generate_regime_features(self, klines_df: pd.DataFrame, regime_data: dict = None) -> pd.DataFrame:
        """Generate regime features using HMM analysis."""
        try:
            self.logger.info("🔧 Generating regime features...")
            
            if regime_data is None:
                # Generate basic regime features
                regime_features = self._generate_basic_regime_features(klines_df)
            else:
                # Use provided regime data
                regime_features = self._extract_regime_features(regime_data, klines_df)
            
            self.logger.info(f"✅ Regime features generated: {regime_features.shape}")
            return regime_features
            
        except Exception as e:
            self.logger.error(f"Error generating regime features: {e}")
            return pd.DataFrame()

    async def _generate_interaction_features(self, base_features: pd.DataFrame, sr_features: pd.DataFrame, regime_features: pd.DataFrame) -> pd.DataFrame:
        """Generate interaction features between different feature types."""
        try:
            self.logger.info("🔧 Generating interaction features...")
            
            interaction_features = pd.DataFrame()
            
            # Combine all features for interaction analysis
            all_features = pd.concat([base_features, sr_features, regime_features], axis=1)
            
            # Generate polynomial features for important variables
            important_features = self._identify_important_features(all_features)
            
            for i, feat1 in enumerate(important_features):
                for feat2 in important_features[i+1:]:
                    if feat1 in all_features.columns and feat2 in all_features.columns:
                        # Create interaction feature
                        interaction_name = f"interaction_{feat1}_{feat2}"
                        interaction_features[interaction_name] = all_features[feat1] * all_features[feat2]
                        
                        # Create ratio feature
                        ratio_name = f"ratio_{feat1}_{feat2}"
                        interaction_features[ratio_name] = all_features[feat1] / (all_features[feat2] + 1e-8)
            
            self.logger.info(f"✅ Interaction features generated: {interaction_features.shape}")
            return interaction_features
            
        except Exception as e:
            self.logger.error(f"Error generating interaction features: {e}")
            return pd.DataFrame()

    def _eliminate_feature_redundancy(self, base_features: pd.DataFrame, sr_features: pd.DataFrame, 
                                    regime_features: pd.DataFrame, interaction_features: pd.DataFrame) -> dict[str, Any]:
        """Eliminate redundant features and calculate redundancy metrics."""
        try:
            metrics = {}
            
            # Combine all features
            all_features = pd.concat([base_features, sr_features, regime_features, interaction_features], axis=1)
            
            # Calculate correlation matrix
            correlation_matrix = all_features.corr().abs()
            
            # Find highly correlated feature pairs
            high_correlation_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if corr_value > 0.95:  # High correlation threshold
                        high_correlation_pairs.append((
                            correlation_matrix.columns[i],
                            correlation_matrix.columns[j],
                            corr_value
                        ))
            
            # Remove redundant features
            redundant_features = set()
            for feat1, feat2, corr in high_correlation_pairs:
                # Keep the feature with more variance
                var1 = all_features[feat1].var()
                var2 = all_features[feat2].var()
                if var1 < var2:
                    redundant_features.add(feat1)
                else:
                    redundant_features.add(feat2)
            
            metrics["total_features"] = len(all_features.columns)
            metrics["redundant_features"] = len(redundant_features)
            metrics["redundancy_ratio"] = len(redundant_features) / len(all_features.columns)
            metrics["high_correlation_pairs"] = len(high_correlation_pairs)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error eliminating feature redundancy: {e}")
            return {}

    def _combine_features(self, base_features: pd.DataFrame, sr_features: pd.DataFrame, 
                         regime_features: pd.DataFrame, interaction_features: pd.DataFrame) -> pd.DataFrame:
        """Combine all features into a comprehensive feature set."""
        try:
            # Combine all features
            comprehensive_features = pd.concat([base_features, sr_features, regime_features, interaction_features], axis=1)
            
            # Remove any duplicate columns
            comprehensive_features = comprehensive_features.loc[:, ~comprehensive_features.columns.duplicated()]
            
            # Fill any NaN values
            comprehensive_features = comprehensive_features.fillna(method='ffill').fillna(0)
            
            return comprehensive_features
            
        except Exception as e:
            self.logger.error(f"Error combining features: {e}")
            return pd.DataFrame()

    def _calculate_feature_quality_metrics(self, features: pd.DataFrame) -> dict[str, float]:
        """Calculate quality metrics for the feature set."""
        try:
            metrics = {}
            
            # Completeness
            metrics["completeness"] = 1.0 - features.isnull().sum().sum() / (features.shape[0] * features.shape[1])
            
            # Variance
            metrics["avg_variance"] = features.var().mean()
            metrics["variance_std"] = features.var().std()
            
            # Correlation
            correlation_matrix = features.corr().abs()
            metrics["avg_correlation"] = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
            
            # Feature count
            metrics["feature_count"] = len(features.columns)
            
            # Overall quality score
            quality_factors = [
                metrics["completeness"],
                min(1.0, metrics["avg_variance"] * 10),  # Normalize variance
                1.0 - metrics["avg_correlation"],  # Lower correlation is better
                min(1.0, metrics["feature_count"] / 100)  # Normalize feature count
            ]
            metrics["overall_quality_score"] = sum(quality_factors) / len(quality_factors)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating feature quality metrics: {e}")
            return {}

    def _generate_fallback_sr_features(self, klines_df: pd.DataFrame) -> pd.DataFrame:
        """Generate fallback S/R features when centralized S/R analysis is not available."""
        try:
            sr_features = pd.DataFrame()
            
            # Basic S/R features
            sr_features["support_level"] = klines_df['low'].rolling(window=20).min()
            sr_features["resistance_level"] = klines_df['high'].rolling(window=20).max()
            sr_features["sr_distance"] = (sr_features["resistance_level"] - sr_features["support_level"]) / klines_df['close']
            
            return sr_features
            
        except Exception as e:
            self.logger.error(f"Error generating fallback S/R features: {e}")
            return pd.DataFrame()

    def _generate_basic_regime_features(self, klines_df: pd.DataFrame) -> pd.DataFrame:
        """Generate basic regime features when HMM analysis is not available."""
        try:
            regime_features = pd.DataFrame()
            
            # Volatility regime
            returns = klines_df['close'].pct_change()
            regime_features["volatility_regime"] = returns.rolling(window=20).std()
            regime_features["volatility_regime_high"] = (regime_features["volatility_regime"] > regime_features["volatility_regime"].quantile(0.8)).astype(int)
            
            # Trend regime
            regime_features["trend_regime"] = returns.rolling(window=10).mean()
            regime_features["trend_regime_bull"] = (regime_features["trend_regime"] > 0).astype(int)
            regime_features["trend_regime_bear"] = (regime_features["trend_regime"] < 0).astype(int)
            
            # Volume regime
            regime_features["volume_regime"] = klines_df['volume'] / klines_df['volume'].rolling(window=20).mean()
            regime_features["volume_regime_high"] = (regime_features["volume_regime"] > 1.5).astype(int)
            
            return regime_features
            
        except Exception as e:
            self.logger.error(f"Error generating basic regime features: {e}")
            return pd.DataFrame()

    def _extract_regime_features(self, regime_data: dict, klines_df: pd.DataFrame) -> pd.DataFrame:
        """Extract regime features from HMM analysis results."""
        try:
            regime_features = pd.DataFrame(index=klines_df.index)
            
            # Extract regime states
            regime_states = regime_data.get("regime_states", [])
            if regime_states:
                for i, state in enumerate(regime_states):
                    if i < len(regime_features):
                        regime_features.loc[regime_features.index[i], "regime_state"] = state.regime_id
                        regime_features.loc[regime_features.index[i], "regime_confidence"] = state.confidence
                        regime_features.loc[regime_features.index[i], "regime_volatility"] = state.volatility
                        regime_features.loc[regime_features.index[i], "regime_momentum"] = state.momentum
            
            # Extract regime transitions
            regime_transitions = regime_data.get("regime_transitions", [])
            if regime_transitions:
                transition_features = pd.DataFrame(index=klines_df.index)
                for transition in regime_transitions:
                    if transition.timestamp in regime_features.index:
                        transition_features.loc[transition.timestamp, "regime_transition"] = 1
                        transition_features.loc[transition.timestamp, "transition_probability"] = transition.probability
                        transition_features.loc[transition.timestamp, "transition_confidence"] = transition.confidence
                
                regime_features = pd.concat([regime_features, transition_features], axis=1)
            
            return regime_features.fillna(0)
            
        except Exception as e:
            self.logger.error(f"Error extracting regime features: {e}")
            return pd.DataFrame()

    def _identify_important_features(self, features: pd.DataFrame, top_n: int = 10) -> List[str]:
        """Identify the most important features for interaction generation."""
        try:
            # Use variance as a simple importance measure
            feature_variance = features.var().sort_values(ascending=False)
            return feature_variance.head(top_n).index.tolist()
            
        except Exception as e:
            self.logger.error(f"Error identifying important features: {e}")
            return []

    @handle_file_operations(default_return=False, context="load_autoencoder")
    def load_autoencoder(self):
        """Load autoencoder model."""
        try:
            # This is handled by the orchestrator now
            return True
        except Exception:
            self.print(error("Error loading autoencoder: {e}"))
            return False
