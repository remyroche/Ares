import logging
import warnings
import os
from typing import Any, Dict
import pywt
import numpy as np
import pandas as pd
import pandas_ta as ta
        # Use existing feature engineering from src.feature_generation.utils
        from src.feature_generation.utils.step06_enhanced_feature_engineering import EnhancedFeatureEngineeringStep
        from .autoencoder_feature_generator import AutoencoderFeatureGenerator
        from .utils.limited_microstructure_features import LimitedMicrostructureFeatures
        from ..training.steps.data_collection.feature_generation.utils.feature_components import EntropyFeatureEngine
        from ..utils.step06_utilities import CrossTimeframeFeatureGenerator

        # Enhanced feature engineering integration
        from src.feature_generation.enhanced_feature_engineering_integration import EnhancedFeatureEngineer
from ..config import CONFIG
from ..core.domain import handle_data_processing_errors, handle_file_operations
from ..utils.logger import system_logger
from ..config_optuna import get_parameter_value
from ..core.decorators import handles_errors

class FeatureEngineeringOrchestrator:
    """
    Comprehensive feature engineering orchestrator that coordinates all feature generation components.
    Integrates advanced feature engineering and autoencoder feature generation.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize the feature engineering orchestrator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger = system_logger.getChild('FeatureEngineeringOrchestrator')
        self.advanced_feature_engineering = EnhancedFeatureEngineeringStep(config)
        self.autoencoder_generator = AutoencoderFeatureGenerator(config)
        self.microstructure_features = LimitedMicrostructureFeatures(config)
        self.entropy_engine = EntropyFeatureEngine(config)
        self.cross_timeframe_generator = CrossTimeframeFeatureGenerator()

        # Enhanced feature engineering
        self.enhanced_feature_engineer = EnhancedFeatureEngineer(config)
        self.model_storage_path = os.path.join(CONFIG['CHECKPOINT_DIR'], 'analyst_models', 'feature_engineering')
        os.makedirs(self.model_storage_path, exist_ok = True)
        self.autoencoder_model_path = os.path.join(self.model_storage_path, 'autoencoder_model.h5')
        self.autoencoder_scaler_path = os.path.join(self.model_storage_path, 'der_scaler.joblib')
        
        self.orchestrator_config = config.get('feature_engineering_orchestrator', {})
        self.enable_advanced_features = get_parameter_value('feature_engineering_parameters.enable_advanced_features', True)
        self.enable_autoencoder_features = get_parameter_value('feature_engineering_parameters.enable_autoencoder_features', True)
        self.enable_legacy_features = get_parameter_value('feature_engineering_parameters.enable_legacy_features', True)
        self.enable_microstructure_features = get_parameter_value('feature_engineering_parameters.enable_microstructure_features', True)
        self.enable_entropy_features = get_parameter_value('feature_engineering_parameters.enable_entropy_features', True)
        self.logger.info('🚀 FeatureEngineeringOrchestrator initialized successfully')

    @handles_errors(exceptions=(Exception,), default_return = pd.DataFrame(), context='orchestrated feature generation')
    async def generate_all_features(self, klines_df: pd.DataFrame, agg_trades_df: pd.DataFrame = None, futures_df: pd.DataFrame = None, sr_levels: list = None) -> pd.DataFrame:
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
        self.logger.info('🎯 Starting comprehensive feature generation orchestration...')
        if klines_df.empty:
            self.logger.warning('Empty klines data provided, returning empty DataFrame')
            return pd.DataFrame()
        try:
            features_df = klines_df.copy()
            if self.enable_advanced_features:
                self.logger.info('📊 Generating advanced features...')
                features_df = self.advanced_feature_generation.utils.generate_features(features_df, agg_trades_df, futures_df)
                self.logger.info(f'✅ Advanced features generated. Shape: {features_df.shape}')

                # Add enhanced features from the new feature engineering system
                self.logger.info('🚀 Generating enhanced features...')
                try:
                    enhanced_features = await self.enhanced_feature_engineer.generate_comprehensive_features(
                        {timeframe: features_df for timeframe in ['1m', '5m', '15m', '1h']},
                        include_categories=['normalization', 'interaction', 'cross_timeframe']
                    )

                    # Merge enhanced features into main dataframe
                    for tf, tf_features in enhanced_features.items():
                        if not tf_features.empty:
                            features_df = pd.concat([features_df, tf_features], axis=1)

                    self.logger.info(f'✅ Enhanced features integrated. Shape: {features_df.shape}')

                except Exception as e:
                    self.logger.warning(f'⚠️ Enhanced features generation failed: {e}')
            if self.enable_autoencoder_features and (not features_df.empty):
                self.logger.info('🤖 Generating autoencoder features...')
                features_df = self.autoencoder_generator.generate_features(features_df)
                self.logger.info(f'✅ Autoencoder features generated. Shape: {features_df.shape}')
            if self.enable_microstructure_features and (not features_df.empty):
                self.logger.info('📈 Generating microstructure features...')
                microstructure_features = await self._generate_microstructure_features(features_df)
                if not microstructure_features.empty:
                    features_df = pd.concat([features_df, microstructure_features], axis = 1)
                    self.logger.info(f'✅ Microstructure features generated. Shape: {features_df.shape}')
            if self.enable_legacy_features:
                self.logger.info('🔧 Generating legacy features...')
                features_df = self._generate_legacy_features(features_df, agg_trades_df, futures_df, sr_levels)
                self.logger.info(f'✅ Legacy features generated. Shape: {features_df.shape}')
            if self.config.get('enable_multi_timeframe', True):
                self.logger.info('⏰ Generating cross-timeframe features...')
                cross_timeframe_features = await self._generate_cross_timeframe_features(klines_df, agg_trades_df)
                if not cross_timeframe_features.empty:
                    features_df = pd.concat([features_df, cross_timeframe_features], axis = 1)
                    self.logger.info(f'✅ Cross-timeframe features generated. Shape: {features_df.shape}')
                
                # Also generate legacy multi-timeframe features for compatibility
                self.logger.info('⏰ Generating legacy multi-timeframe features...')
                multi_timeframe_features = await self._calculate_multi_timeframe_features(klines_df, agg_trades_df, None)
                if not multi_timeframe_features.empty:
                    features_df = pd.concat([features_df, multi_timeframe_features], axis = 1)
                    self.logger.info(f'✅ Legacy multi-timeframe features generated. Shape: {features_df.shape}')
            if self.config.get('enable_meta_labeling', True):
                self.logger.info('🏷️ Generating meta-labeling features...')
                meta_labeling_features = await self._calculate_meta_labeling_features(klines_df, agg_trades_df, None)
                if not meta_labeling_features.empty:
                    features_df = pd.concat([features_df, meta_labeling_features], axis = 1)
                    self.logger.info(f'✅ Meta-labeling features generated. Shape: {features_df.shape}')
            if self.enable_entropy_features and (not features_df.empty):
                self.logger.info('🧠 Generating entropy features...')
                features_df = self.entropy_engine.create_entropy_features(features_df)
                self.logger.info(f'✅ Entropy features generated. Shape: {features_df.shape}')
            features_df = self._cleanup_features(features_df)
            self.logger.info(f'🎉 Feature generation orchestration completed! Final shape: {features_df.shape}')
            self.logger.info(f'📊 Total features generated: {len(features_df.columns)}')
            return features_df
        except Exception:
            self.logger.error('❌ Error in feature generation orchestration: {e}')
            return klines_df.copy()

    @handles_errors(exceptions=(Exception,), default_return = pd.DataFrame(), context='microstructure feature generation')
    async def _generate_microstructure_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Generate microstructure features from available market data."""
        try:
            microstructure_features_df = pd.DataFrame(index = features_df.index)
            
            # Extract available market data for microstructure analysis
            for idx, row in features_df.iterrows():
                # Create market data dictionary from available features
                market_data = {
                    'bid': row.get('close', 0),  # Use close as bid approximation
                    'ask': row.get('close', 0) * 1.0001,  # Add small spread
                    'last_price': row.get('close', 0),
                    'volume': row.get('volume', 0),
                    'high': row.get('high', 0),
                    'low': row.get('low', 0)
                }
                
                # Extract microstructure features
                features = await self.microstructure_features.extract_features(market_data, features_df.head(idx))
                
                if features:
                    # Add features to DataFrame
                    for feature_name, feature_value in features.items():
                        if isinstance(feature_value, (int, float)) and not pd.isna(feature_value):
                            if feature_name not in microstructure_features_df.columns:
                                microstructure_features_df[feature_name] = 0.0
                            microstructure_features_df.loc[idx, feature_name] = feature_value
            
            return microstructure_features_df
            
        except Exception as e:
            self.logger.error(f"Error generating microstructure features: {e}")
            return pd.DataFrame()

    @handle_data_processing_errors(default_return = pd.DataFrame(), context='legacy feature generation')
    def _generate_legacy_features(self, features_df: pd.DataFrame, agg_trades_df: pd.DataFrame = None, futures_df: pd.DataFrame = None, sr_levels: list = None) -> pd.DataFrame:
        """Generate legacy features for backward compatibility."""
        try:
            if futures_df is not None and (not futures_df.empty):
                features_df = pd.merge_asof(features_df.sort_index(), futures_df.sort_index(), left_index = True, right_index = True, direction='backward').ffill().fillna(0)
            features_df = self._calculate_standard_indicators(features_df)
            features_df = self._calculate_time_features(features_df)
            features_df = self._calculate_volatility_regime_indicators(features_df)
            features_df = self._calculate_volatility_targeting_features(features_df)
            return self._calculate_ml_enhanced_features(features_df)
        except Exception:
            self.logger.error('Error generating legacy features: {e}')
            return features_df

    @handles_errors(exceptions=(Exception,), default_return = pd.DataFrame(), context='multi-timeframe feature calculation')
    async def _generate_cross_timeframe_features(self, price_data: pd.DataFrame, volume_data: pd.DataFrame = None) -> pd.DataFrame:
        """Generate cross-timeframe features using the CrossTimeframeFeatureGenerator from step06_utilities."""
        try:
            # Prepare data for cross-timeframe feature generation
            if volume_data is not None and not volume_data.empty:
                # Use volume data if available
                cross_features = self.cross_timeframe_generator.generate_cross_timeframe_features(price_data, volume_data)
            else:
                # Create mock volume data if not available
                mock_volume = pd.DataFrame({'volume': price_data.get('volume', 1000.0)})
                cross_features = self.cross_timeframe_generator.generate_cross_timeframe_features(price_data, mock_volume)
            
            # Convert dictionary to DataFrame
            if isinstance(cross_features, dict) and cross_features:
                cross_features_df = pd.DataFrame(cross_features, index=price_data.index)
                return cross_features_df
            else:
                self.logger.warning("No cross-timeframe features generated")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error generating cross-timeframe features: {e}")
            return pd.DataFrame()
    
    async def _calculate_multi_timeframe_features(self, price_data: pd.DataFrame, volume_data: pd.DataFrame, order_flow_data: pd.DataFrame | None = None) -> pd.DataFrame:
        """Calculate multi-timeframe features."""
        try:
            # Use existing feature engineering from src.feature_generation.utils
            from src.feature_generation.utils.step06_enhanced_feature_engineering import EnhancedFeatureEngineering
            advanced_fe = EnhancedFeatureEngineering(self.config)
            await advanced_fe.initialize()
            multi_timeframe_features = await advanced_fe._engineer_multi_timeframe_features(price_data, volume_data, order_flow_data)

            # Convert dictionary to DataFrame properly
            if isinstance(multi_timeframe_features, dict) and multi_timeframe_features:
                # If it's a dict of DataFrames, concatenate them
                if all(isinstance(v, pd.DataFrame) for v in multi_timeframe_features.values()):
                    return pd.concat(list(multi_timeframe_features.values()), axis=1)
                # If it's a dict of Series/values, create DataFrame from dict
                else:
                    return pd.DataFrame(multi_timeframe_features)
            elif isinstance(multi_timeframe_features, pd.DataFrame):
                return multi_timeframe_features
            else:
                self.logger.warning(f"Unexpected multi_timeframe_features type: {type(multi_timeframe_features)}")
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f'Error calculating multi-timeframe features: {e}')
            return pd.DataFrame()

    @handles_errors(exceptions=(Exception,), default_return = pd.DataFrame(), context='meta-labeling feature calculation')
    async def _calculate_meta_labeling_features(self, price_data: pd.DataFrame, volume_data: pd.DataFrame, order_flow_data: pd.DataFrame | None = None) -> pd.DataFrame:
        """Calculate meta-labeling features."""
        try:
            from .meta_labeling_system import MetaLabelingSystem

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

except ImportError:
    
    cp = None

            meta_labeling = MetaLabelingSystem(self.config)
            await meta_labeling.initialize()
            analyst_labels = await meta_labeling._generate_analyst_labels(price_data, volume_data, order_flow_data)
            tactician_labels = await meta_labeling._generate_tactician_labels(price_data, volume_data, order_flow_data)
            all_labels = {**analyst_labels, **tactician_labels}
            return pd.DataFrame([all_labels])
        except Exception:
            self.logger.error('Error calculating meta-labeling features: {e}')
            return pd.DataFrame()

    @handle_data_processing_errors(default_return = pd.DataFrame(), context='standard indicators calculation')
    def _calculate_standard_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate standard technical indicators using price differences."""
        try:
            
            close_diff = df['close'].diff().fillna(0)
            high_diff = df['high'].diff().fillna(0)
            low_diff = df['low'].diff().fillna(0)
            temp_df = df.copy()
            temp_df['close'] = close_diff
            temp_df['high'] = high_diff
            temp_df['low'] = low_diff
            df['sma_5'] = ta.sma(temp_df['close'], length = 5)
            df['sma_10'] = ta.sma(temp_df['close'], length = 10)
            df['sma_20'] = ta.sma(temp_df['close'], length = 20)
            df['sma_50'] = ta.sma(temp_df['close'], length = 50)
            df['ema_12'] = ta.ema(temp_df['close'], length = 12)
            df['ema_26'] = ta.ema(temp_df['close'], length = 26)
            df['rsi'] = ta.rsi(temp_df['close'], length = 14)
            macd = ta.macd(temp_df['close'])
            df['macd'] = macd['MACD_12_26_9']
            df['macd_signal'] = macd['MACDs_12_26_9']
            df['macd_hist'] = macd['MACDh_12_26_9']
            bb = ta.bbands(temp_df['close'])
            df['bb_upper'] = bb['BBU_20_2.0']
            df['bb_middle'] = bb['BBM_20_2.0']
            df['bb_lower'] = bb['BBL_20_2.0']
            df['bb_width'] = (bb['BBU_20_2.0'] - bb['BBL_20_2.0']) / bb['BBM_20_2.0']
            stoch = ta.stoch(temp_df['high'], temp_df['low'], temp_df['close'])
            df['stoch_k'] = stoch['STOCHk_14_3_3']
            df['stoch_d'] = stoch['STOCHd_14_3_3']
            df['atr'] = ta.atr(temp_df['high'], temp_df['low'], temp_df['close'], length = 14)
            return df
        except Exception:
            self.logger.error('Error calculating standard indicators: {e}')
            return df

    @handle_data_processing_errors(default_return = pd.DataFrame(), context='time features calculation')
    def _calculate_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate time-based features."""
        try:
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['day_of_month'] = df.index.day
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df['is_asia_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
            df['is_london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
            df['is_ny_session'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
            return df
        except Exception:
            self.logger.error('Error calculating time features: {e}')
            return df

    @handle_data_processing_errors(default_return = pd.DataFrame(), context='volatility regime indicators calculation')
    def _calculate_volatility_regime_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility regime indicators."""
        try:
            returns = df['close'].pct_change()
            df['volatility_5'] = returns.rolling(window = 5).std()
            df['volatility_10'] = returns.rolling(window = 10).std()
            df['volatility_20'] = returns.rolling(window = 20).std()

            def classify_vol_regime(vol: Any) -> int:
                if vol <= 0.02:
                    return 0
                if vol <= 0.04:
                    return 1
                if vol <= 0.08:
                    return 2
                return 3
            df['volatility_regime_5'] = df['volatility_5'].apply(classify_vol_regime)
            df['volatility_regime_10'] = df['volatility_10'].apply(classify_vol_regime)
            df['volatility_regime_20'] = df['volatility_20'].apply(classify_vol_regime)
            df['vol_ratio_5_20'] = df['volatility_5'] / df['volatility_20']
            df['vol_ratio_10_20'] = df['volatility_10'] / df['volatility_20']
            return df
        except Exception as e:
            self.logger.exception(f'Error calculating volatility regime indicators: {e}')
            return df

    @handle_data_processing_errors(default_return = pd.DataFrame(), context='volatility targeting features calculation')
    def _calculate_volatility_targeting_features(self, df: pd.DataFrame, target_volatility: float = 0.15) -> pd.DataFrame:
        """Calculate volatility targeting features."""
        try:
            target_vol_daily = target_volatility / np.sqrt(252)
            returns = df['close'].pct_change()
            current_vol = returns.rolling(window = 20).std()
            df['vol_target_ratio'] = current_vol / target_vol_daily
            df['vol_adjusted_position'] = 1.0 / df['vol_target_ratio']
            df['vol_adjusted_position'] = df['vol_adjusted_position'].clip(0.1, 2.0)
            df['vol_regime'] = np.where(df['vol_target_ratio'] > 1.5, 'high_vol', np.where(df['vol_target_ratio'] < 0.5, 'low_vol', 'normal_vol'))
            return df
        except Exception as e:
            self.logger.exception(f'Error calculating volatility targeting features: {e}')
            return df

    @handle_data_processing_errors(default_return = pd.DataFrame(), context='ML enhanced features calculation')
    def _calculate_ml_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ML-enhanced features."""
        try:
            df['price_momentum_5'] = df['close'].pct_change(5)
            if 'volume' in df.columns:
                df['volume_momentum_5'] = df['volume'].pct_change(5)
                df['volume_ratio'] = df['volume'] / df['volume'].rolling(window = 20).mean()
            df['resistance_20'] = df['high'].rolling(window = 20).max()
            df['support_20'] = df['low'].rolling(window = 20).min()
            df['dist_to_resistance'] = (df['resistance_20'] - df['close']) / df['close']
            df['dist_to_support'] = (df['close'] - df['support_20']) / df['close']
            df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
            df['r1'] = 2 * df['pivot'] - df['low']
            df['s1'] = 2 * df['pivot'] - df['high']
            return df
        except Exception:
            self.logger.error('Error calculating ML enhanced features: {e}')
            return df

    @handle_data_processing_errors(default_return = pd.DataFrame(), context='feature cleanup')
    def _cleanup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean up and validate features."""
        try:
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            df = df.dropna(axis = 1, how='all')
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df = df[numeric_cols]
            self.logger.info(f'Feature cleanup completed. Final shape: {df.shape}')
            return df
        except Exception:
            self.logger.error('Error in feature cleanup: {e}')
            return df

    @handles_errors(exceptions=(Exception,), default_return={}, context='orchestrator info retrieval')
    @handles_errors(exceptions=(Exception,), default_return={}, context='orchestrator info retrieval')
    def get_orchestrator_info(self) -> dict[str, Any]:
        """Get information about the orchestrator."""
        try:
            return {'orchestrator_type': 'FeatureEngineeringOrchestrator', 'enable_advanced_features': self.enable_advanced_features, 'enable_autoencoder_features': self.enable_autoencoder_features, 'enable_legacy_features': self.enable_legacy_features, 'enable_entropy_features': self.enable_entropy_features, 'advanced_feature_engineering_info': self.advanced_feature_generation.utils.get_feature_statistics(), 'autoencoder_generator_info': self.autoencoder_generator.get_generator_info(), 'config': self.orchestrator_config}
        except Exception:
            self.logger.error('Error getting orchestrator info: {e}')
            return {}

    @handles_errors(exceptions=(Exception,), default_return={}, context='feature summary retrieval')
    def get_feature_summary(self) -> dict[str, Any]:
        """Get a summary of all available features."""
        try:
            return {'feature_categories': ['standard_indicators', 'advanced_features', 'autoencoder_features', 'time_features', 'volatility_features', 'ml_enhanced_features', 'entropy_features'], 'total_feature_types': 7, 'orchestrator_config': self.orchestrator_config}
        except Exception:
            self.logger.error('Error getting feature summary: {e}')
            return {}

class FeatureEngineeringEngine:
    """
    Legacy feature engineering engine for backward compatibility.
    Now delegates to the orchestrator.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config.get('analyst', {}).get('feature_engineering', {})
        self.logger = system_logger.getChild('FeatureEngineeringEngine')
        self.orchestrator = FeatureEngineeringOrchestrator(config)
        self.autoencoder_model = None
        self.autoencoder_scaler = None
        self.model_storage_path = os.path.join(CONFIG['CHECKPOINT_DIR'], 'analyst_models', 'feature_engineering')
        os.makedirs(self.model_storage_path, exist_ok = True)
        self.autoencoder_model_path = os.path.join(self.model_storage_path, 'autoencoder_model.h5')
        self.autoencoder_scaler_path = os.path.join(self.model_storage_path, 'der_scaler.joblib')

    @handles_errors(exceptions=(Exception,), default_return = pd.DataFrame(), context='generate_all_features')
    async def generate_all_features(self, klines_df: pd.DataFrame, agg_trades_df: pd.DataFrame, futures_df: pd.DataFrame, sr_levels: list) -> None:
        """
        Generate all features using the orchestrator.
        """
        return await self.orchestrator.generate_all_features(klines_df, agg_trades_df, futures_df, sr_levels)

    @handles_errors(exceptions=(Exception,), default_return = None, context='wavelet transforms')
    def apply_wavelet_transforms(self, data: pd.Series, wavelet: Any='db1', level: Any = 3) -> None:
        """Apply wavelet transforms to data."""
        try:
            return pywt.wavedec(data, wavelet, level = level)
        except Exception:
            self.logger.error('Error applying wavelet transforms: {e}')
            return None

    @handle_file_operations(default_return = False, context='train_autoencoder')
    def train_autoencoder(self, data: pd.DataFrame) -> Any:
        """Train autoencoder model."""
        try:
            return self.orchestrator.autoencoder_generator.pipeline.autoencoder is not None
        except Exception:
            self.logger.error('Error training autoencoder: {e}')
            return False

    @handle_data_processing_errors(default_return = pd.Series(), context='apply_autoencoders')
    def apply_autoencoders(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply autoencoder features."""
        try:
            return self.orchestrator.autoencoder_generator.generate_features(data)
        except Exception:
            self.logger.error('Error applying autoencoders: {e}')
            return data

    @handle_file_operations(default_return = False, context='load_autoencoder')
    def load_autoencoder(self) -> Any:
        """Load autoencoder model."""
        try:
            return True
        except Exception:
            self.logger.error('Error loading autoencoder: {e}')
            return False

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _vectorbt_apply_operation(self, data: pd.Series, func, 
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)
        
        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
