
import pandas as pd
import numpy as np
from src.core.decorators import handles_errors
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation

# Optional scipy sparse support
try:
    import scipy.sparse as sp
    SCIPY_SPARSE_AVAILABLE = True
except ImportError:
    SCIPY_SPARSE_AVAILABLE = False
    sp = None

"""Step 6: Advanced Feature Engineering - Refactored and modular.

This module generates advanced features including technical indicators,
wavelet features, and market microstructure features.
"""
from pathlib import Path
from typing import Any, Dict, List, Tuple
from src.training.base_step import BaseStep
# Optional advanced components; fallback to basic if unavailable
try:
    from src.analyst.feature_engineering_utils import TechnicalIndicatorCalculator as _TechnicalIndicatorCalculator  # type: ignore
    import logging
except Exception:
    _TechnicalIndicatorCalculator = None  # type: ignore

# Import new enhanced feature engineering components
try:
    from .feature_components import (
        DataResampler,
        WaveletAnalyzer,
        EnhancedFeatureInteractionEngine,
        EnhancedRegimeAwareFeatureEngine,
        MarketProfileFeatureEngine,
        IchimokuFeatureEngine,
        HarmonicPatternFeatureEngine,
        SentimentFeatureEngine
    )
except ImportError as e:
    # Fallback implementations if import fails
    class DataResampler:
        def __init__(self, config):
            self.config = config
            self.logger = logging.getLogger(__name__)

        def create_multi_timeframe_features(self, data, base_timeframe, target_timeframes):
            """Basic fallback implementation."""
            return {base_timeframe: data}

    class WaveletAnalyzer:
        def __init__(self, config):
            self.config = config
            self.logger = logging.getLogger(__name__)

        def analyze_wavelet_features(self, data):
            """Basic fallback implementation."""
            return data

    class EnhancedFeatureInteractionEngine:
        def __init__(self, config):
            self.config = config
            self.logger = logging.getLogger(__name__)

        def create_feature_interactions(self, data):
            """Basic fallback implementation."""
            return data

    class EnhancedRegimeAwareFeatureEngine:
        def __init__(self, config):
            self.config = config
            self.logger = logging.getLogger(__name__)

        def create_regime_aware_features(self, data):
            """Basic fallback implementation."""
            return data

    class MarketProfileFeatureEngine:
        def __init__(self, config):
            self.config = config
            self.logger = logging.getLogger(__name__)

        def create_market_profile_features(self, data):
            """Basic fallback implementation."""
            return data

    class IchimokuFeatureEngine:
        def __init__(self, config):
            self.config = config
            self.logger = logging.getLogger(__name__)

        def create_ichimoku_features(self, data):
            """Basic fallback implementation."""
            return data

    class HarmonicPatternFeatureEngine:
        def __init__(self, config):
            self.config = config
            self.logger = logging.getLogger(__name__)

        def detect_harmonic_patterns(self, data):
            """Basic fallback implementation."""
            return data

    class SentimentFeatureEngine:
        def __init__(self, config):
            self.config = config
            self.logger = logging.getLogger(__name__)

        def create_sentiment_features(self, data):
            """Basic fallback implementation."""
            return data

class AdvancedFeatureEngineeringStep(BaseStep):
    """Step 6: Advanced Feature Engineering using modular components."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize advanced feature engineering step.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config, '06', 'advanced_feature_engineering')
        self.resampler = None
        self.wavelet_analyzer = None
        self.indicator_calculator = None
        self.interaction_engine = None
        self.regime_engine = None
        self.feature_config = config.get('feature_engineering', {})
        self.enable_wavelets = self.feature_config.get('enable_wavelets', True)
        self.enable_multi_timeframe = self.feature_config.get('enable_multi_timeframe', True)
        self.enable_feature_interactions = self.feature_config.get('enable_feature_interactions', True)
        self.enable_regime_features = self.feature_config.get('enable_regime_features', True)
        self.timeframes = self.feature_config.get('timeframes', ['30m', '1h', '4h', '1d'])
    def _initialize_step(self) -> None:
        """Initialize step-specific components."""
        # Initialize multi-timeframe resampler
        if DataResampler is not None:
            self.resampler = DataResampler(self.feature_config)
        else:
            self.resampler = None

        # Initialize wavelet analyzer
        if WaveletAnalyzer is not None:
            self.wavelet_analyzer = WaveletAnalyzer(self.feature_config)
        else:
            self.wavelet_analyzer = None

        # Initialize feature interaction engine
        if EnhancedFeatureInteractionEngine is not None:
            self.interaction_engine = EnhancedFeatureInteractionEngine(self.feature_config)
        else:
            self.interaction_engine = None

        # Initialize regime-aware feature engine
        if EnhancedRegimeAwareFeatureEngine is not None:
            self.regime_engine = EnhancedRegimeAwareFeatureEngine(self.feature_config)
        else:
            self.regime_engine = None

        # Initialize new feature engines with regime-aware configuration
        regime_config = self._get_regime_specific_config()
        enhanced_config = {**self.config, **regime_config}

        if MarketProfileFeatureEngine is not None:
            self.market_profile_engine = MarketProfileFeatureEngine(enhanced_config)
        else:
            self.market_profile_engine = None

        if IchimokuFeatureEngine is not None:
            self.ichimoku_engine = IchimokuFeatureEngine(enhanced_config)
        else:
            self.ichimoku_engine = None

        if HarmonicPatternFeatureEngine is not None:
            self.harmonic_engine = HarmonicPatternFeatureEngine(enhanced_config)
        else:
            self.harmonic_engine = None

        if SentimentFeatureEngine is not None:
            self.sentiment_engine = SentimentFeatureEngine(enhanced_config)
        else:
            self.sentiment_engine = None

        # Enable flags for new features
        self.enable_market_profile = self.config.get('feature_engineering_parameters', {}).get('enable_market_profile_features', True)
        self.enable_ichimoku = self.config.get('feature_engineering_parameters', {}).get('enable_ichimoku_features', True)
        self.enable_harmonic = self.config.get('feature_engineering_parameters', {}).get('enable_harmonic_features', True)
        self.enable_sentiment = self.config.get('feature_engineering_parameters', {}).get('enable_sentiment_features', True)

    def _get_regime_specific_config(self) -> Dict[str, Any]:
        """Get regime-specific configuration for feature parameters.

        Returns:
            Dict containing regime-specific parameter overrides
        """
        # This would be enhanced to detect current market regime
        # For now, return default regime parameters
        ab_testing_config = self.config.get('ab_testing', {})
        hmm_config = ab_testing_config.get('hmm_regime_ab_testing', {})

        # Check if we should use conservative or aggressive regime configs
        # This is a simplified implementation - in production, this would be based on actual regime detection
        regime_type = self._detect_current_regime_type()

        if regime_type == 'conservative':
            regime_params = hmm_config.get('conservative_regime_configs', {})
        elif regime_type == 'aggressive':
            regime_params = hmm_config.get('aggressive_regime_configs', {})
        else:
            regime_params = {}

        # Extract feature-specific parameters from regime config
        step17_config = {}

        if 'market_profile_periods' in regime_params:
            step17_config['market_profile'] = {
                'profile_periods': regime_params['market_profile_periods']
            }

        if 'ichimoku_periods' in regime_params:
            ichimoku_periods = regime_params['ichimoku_periods']
            if len(ichimoku_periods) >= 4:
                step17_config['ichimoku'] = {
                    'tenkan_period': ichimoku_periods[0],
                    'kijun_period': ichimoku_periods[1],
                    'senkou_span_b_period': ichimoku_periods[2],
                    'displacement': ichimoku_periods[3]
                }

        if 'harmonic_tolerance' in regime_params:
            step17_config['harmonic_patterns'] = {
                'pattern_tolerance': regime_params['harmonic_tolerance']
            }

        if 'sentiment_lookback' in regime_params:
            step17_config['sentiment_features'] = {
                'greed_fear_lookback': regime_params['sentiment_lookback']
            }

        return {'step17_optimization': step17_config}

    def _detect_current_regime_type(self) -> str:
        """Detect current market regime type for parameter selection.

        Returns:
            String indicating regime type ('conservative', 'aggressive', or 'neutral')
        """
        # Simplified regime detection - in production, this would use actual market data
        # For now, return 'neutral' to use default parameters
        return 'neutral'

        # Prefer optional component if available; otherwise, implement a simple calculator inline
        if _TechnicalIndicatorCalculator is not None:
            self.indicator_calculator = _TechnicalIndicatorCalculator([
                {"name": "RSI", "params": {"period": 14}},
                {"name": "SMA", "params": {"period": 20}},
                {"name": "EMA", "params": {"period": 12}},
            ])
        else:
            self.indicator_calculator = None

        self.logger.info('✅ Enhanced feature engineering components initialized')
    @log_step_functions

    def validate_inputs(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Tuple[bool, list]:
        """Validate step inputs.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        if 'labeled_data' not in pipeline_state:
            errors.append('No labeled_data found in pipeline state')
        if 'labeled_data' in pipeline_state:
            data_path = Path(pipeline_state['labeled_data'])
            if data_path.exists():
                try:
                    # Optimized Parquet reading with performance settings
                    read_options = {
                        'engine': 'pyarrow' if hasattr(pd, 'ArrowDtype') else 'fastparquet',
                        'columns': ['open', 'high', 'low', 'close', 'volume']
                    }
                    sample = pd.read_parquet(data_path, **read_options)
                    required_cols = ['open', 'high', 'low', 'close', 'volume']
                    missing = set(required_cols) - set(sample.columns)
                    if missing:
                        errors.append(f'Missing required columns: {missing}')
                except Exception as e:
                    errors.append(f'Failed to validate data file: {e}')
            else:
                errors.append(f'Labeled data file not found: {data_path}')
        return (len(errors) == 0, errors)

    @handles_errors(exceptions=(Exception,), default_return={'success': False}, context='advanced feature engineering')
    async def execute_logic(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute advanced feature engineering.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Updated pipeline state
        """
        symbol = training_input['symbol']
        exchange = training_input['exchange']
        base_timeframe = training_input.get('timeframe', '1m')
        self.logger.info(f'🔧 Engineering advanced features for {symbol} ({base_timeframe})')
        labeled_data_path = Path(pipeline_state['labeled_data'])
        # Optimized Parquet reading for main data
        read_options = {
            'engine': 'pyarrow' if hasattr(pd, 'ArrowDtype') else 'fastparquet',
            'use_threads': True  # Enable multi-threading for faster reads
        }
        data = pd.read_parquet(labeled_data_path, **read_options)

        # Advanced NumPy optimizations for memory and performance
        data = self._optimize_dataframe_for_numpy(data)
        self.logger.info(f'📊 Loaded {len(data)} rows of labeled data')
        all_features = {}
        self.logger.info('📈 Calculating technical indicators...')
        chunk_size = int(self.feature_config.get('chunk_size', 300000))
        tech_chunks: List[pd.DataFrame] = []
        for start in range(0, len(data), chunk_size):
            end = min(len(data), start + chunk_size)
            part = data.iloc[start:end].copy()
            tech = (
                self.indicator_calculator.calculate(part)
                if self.indicator_calculator is not None
                else self._basic_indicators(part)
            )
            tech_chunks.append(tech)
        technical_features = pd.concat(tech_chunks, axis = 0, ignore_index = True)
        all_features['technical'] = technical_features
        # Wavelet analyzer optional; skip if not available
        if self.enable_wavelets and self.wavelet_analyzer is not None:
            self.logger.info('🌊 Calculating wavelet features...')
            wavelet_features = self.wavelet_analyzer.extract_wavelet_features(data, price_column='close', symbol = symbol, timeframe = base_timeframe)
            all_features['wavelet'] = wavelet_features
            self.logger.info(f'✅ Wavelet features: {len(wavelet_features.columns) if wavelet_features is not None else 0} features')
        if self.enable_multi_timeframe:
            self.logger.info('⏰ Calculating multi-timeframe features...')
            mtf_features = await self._calculate_multi_timeframe_features(data, base_timeframe, symbol)
            # Ensure we have a DataFrame, not a coroutine
            if hasattr(mtf_features, 'columns'):
                all_features['multi_timeframe'] = mtf_features
            else:
                self.logger.error('❌ Multi-timeframe feature calculation returned invalid result')
                all_features['multi_timeframe'] = pd.DataFrame(index=data.index)

        self.logger.info('🔬 Calculating market microstructure features...')
        microstructure_features = self._calculate_microstructure_features(data)
        all_features['microstructure'] = microstructure_features

        if self.enable_feature_interactions and self.interaction_engine is not None:
            self.logger.info('🔗 Creating feature interactions...')
            interaction_features = await self._create_feature_interactions(data)
            # Ensure we have a DataFrame, not a coroutine
            if hasattr(interaction_features, 'columns'):
                all_features['interactions'] = interaction_features
                self.logger.info(f'✅ Feature interactions: {len(interaction_features.columns)} features')
            else:
                self.logger.warning('⚠️ Feature interactions returned invalid result, skipping')
                all_features['interactions'] = pd.DataFrame(index=data.index)

        if self.enable_regime_features and self.regime_engine is not None:
            self.logger.info('🎭 Creating regime-aware features...')
            regime_features = await self._create_regime_aware_features(data, pipeline_state)
            # Ensure we have a DataFrame, not a coroutine
            if hasattr(regime_features, 'columns'):
                all_features['regime'] = regime_features
                self.logger.info(f'✅ Regime features: {len(regime_features.columns)} features')
            else:
                self.logger.warning('⚠️ Regime features returned invalid result, skipping')
                all_features['regime'] = pd.DataFrame(index=data.index)

        # New advanced feature categories
        if self.enable_market_profile and self.market_profile_engine is not None:
            self.logger.info('📊 Creating market profile features...')
            market_profile_features = self.market_profile_engine.create_market_profile_features(data)
            all_features['market_profile'] = market_profile_features
            self.logger.info(f'✅ Market profile features: {len(market_profile_features.columns) if market_profile_features is not None else 0} features')

        if self.enable_ichimoku and self.ichimoku_engine is not None:
            self.logger.info('☁️ Creating Ichimoku Cloud features...')
            ichimoku_features = self.ichimoku_engine.create_ichimoku_features(data)
            all_features['ichimoku'] = ichimoku_features
            self.logger.info(f'✅ Ichimoku features: {len(ichimoku_features.columns) if ichimoku_features is not None else 0} features')

        if self.enable_harmonic and self.harmonic_engine is not None:
            self.logger.info('🎵 Creating harmonic pattern features...')
            harmonic_features = self.harmonic_engine.create_harmonic_features(data)
            all_features['harmonic'] = harmonic_features
            self.logger.info(f'✅ Harmonic features: {len(harmonic_features.columns) if harmonic_features is not None else 0} features')

        if self.enable_sentiment and self.sentiment_engine is not None:
            self.logger.info('😀 Creating sentiment features...')
            sentiment_features = self.sentiment_engine.create_sentiment_features(data)
            all_features['sentiment'] = sentiment_features
            self.logger.info(f'✅ Sentiment features: {len(sentiment_features.columns) if sentiment_features is not None else 0} features')

        # Validate and fill NaN values in all feature groups
        for feature_type, feature_df in all_features.items():
            if feature_df is not None and hasattr(feature_df, 'columns') and len(feature_df.columns) > 0:
                all_features[feature_type] = self._validate_and_fill_features(feature_df, data)

        self.logger.info('🔗 Combining all features...')
        combined_features = self._combine_features(data, all_features)
        train_features, val_features = self._split_features(combined_features, pipeline_state.get('train_end_idx', int(len(combined_features) * 0.8)))
        output_dir = Path(training_input.get('data_dir', 'data/training'))
        output_dir.mkdir(parents = True, exist_ok = True)
        train_path = output_dir / f'{exchange}_{symbol}_{base_timeframe}_features_train.parquet'
        val_path = output_dir / f'{exchange}_{symbol}_{base_timeframe}_features_val.parquet'
        # Optimized Parquet writing with performance settings
        parquet_options = {
            'compression': 'snappy',
            'engine': 'pyarrow' if hasattr(pd, 'ArrowDtype') else 'fastparquet',
            'index': False,  # Don't save index to reduce file size
        }

        # Add row group size for PyArrow if available
        try:
            import pyarrow as pa
            parquet_options['row_group_size'] = 50000  # Optimize for reading performance
        except ImportError:
            pass

        train_features.to_parquet(train_path, **parquet_options)
        val_features.to_parquet(val_path, **parquet_options)
        self.logger.info(f'✅ Saved features - Train: {len(train_features)} rows, Val: {len(val_features)} rows, Features: {len(train_features.columns)} columns')
        pipeline_state['advanced_features'] = {
            'train': str(train_path),
            'val': str(val_path),
            'n_features': len(train_features.columns),
            'feature_groups': list(all_features.keys()),
            'feature_names': list(train_features.columns),
            'wavelet_enabled': self.enable_wavelets and self.wavelet_analyzer is not None,
            'multitimeframe_enabled': self.enable_multi_timeframe and self.resampler is not None,
            'interactions_enabled': self.enable_feature_interactions and self.interaction_engine is not None,
            'regime_features_enabled': self.enable_regime_features and self.regime_engine is not None,
            'market_profile_enabled': self.enable_market_profile and self.market_profile_engine is not None,
            'ichimoku_enabled': self.enable_ichimoku and self.ichimoku_engine is not None,
            'harmonic_enabled': self.enable_harmonic and self.harmonic_engine is not None,
            'sentiment_enabled': self.enable_sentiment and self.sentiment_engine is not None
        }
        pipeline_state['feature_statistics'] = self._calculate_feature_statistics(train_features)
        return pipeline_state

    async def _calculate_multi_timeframe_features(self, data: pd.DataFrame, base_timeframe: str, symbol: str) -> pd.DataFrame:
        """Calculate features from multiple timeframes."""
        mtf_features = pd.DataFrame(index=data.index)

        if self.resampler is None:
            self.logger.warning("DataResampler not available, using basic multi-timeframe features")
            return await self._basic_multi_timeframe_features(data, base_timeframe)

        try:
            # Use the enhanced resampler
            mtf_data = self.resampler.create_multi_timeframe_features(data, base_timeframe, self.timeframes)

            for tf, tf_data in mtf_data.items():
                if tf == base_timeframe or tf_data.empty:
                    continue

                # Calculate indicators for each timeframe
                if self.indicator_calculator:
                    tf_indicators = self.indicator_calculator.calculate(tf_data)
                    # Handle async calculator if needed
                    if hasattr(tf_indicators, '__await__'):
                        tf_indicators = await tf_indicators
                else:
                    tf_indicators = self._basic_indicators(tf_data)

                key_features = ['rsi_14', 'volatility_20', 'sma_20', 'ema_12', 'volume_ratio_10']
                for feat in key_features:
                    if feat in tf_indicators.columns:
                        # Align to original timeframe using forward fill
                        aligned = tf_indicators[feat].reindex(data.index, method='ffill')
                        mtf_features[f'{feat}_{tf}'] = aligned

            self.logger.info(f"✅ Created {len(mtf_features.columns)} multi-timeframe features")
            return mtf_features

        except Exception as e:
            self.logger.error(f"❌ Multi-timeframe feature calculation failed: {e}")
            return await self._basic_multi_timeframe_features(data, base_timeframe)

    async def _basic_multi_timeframe_features(self, data: pd.DataFrame, base_timeframe: str) -> pd.DataFrame:
        """Fallback multi-timeframe features when resampler is not available."""
        mtf_features = pd.DataFrame(index=data.index)

        # Simple higher timeframe approximations using rolling windows
        for tf in self.timeframes:
            if tf == base_timeframe:
                continue

            # Extract multiplier from timeframe string
            multiplier = self._get_timeframe_multiplier(tf, base_timeframe)
            if multiplier > 1:
                # Approximate higher timeframe features
                mtf_features[f'close_{tf}_approx'] = data['close'].rolling(multiplier).mean()
                mtf_features[f'volume_{tf}_approx'] = data['volume'].rolling(multiplier).sum()
                mtf_features[f'volatility_{tf}_approx'] = data['close'].pct_change().rolling(multiplier).std()

        return mtf_features

    def _get_timeframe_multiplier(self, target_tf: str, base_tf: str) -> int:
        """Get multiplier between timeframes."""
        multipliers = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240, '1d': 1440}

        base_mult = multipliers.get(base_tf, 1)
        target_mult = multipliers.get(target_tf, 60)

        if target_mult % base_mult == 0:
            return target_mult // base_mult
        return 1  # Default fallback
    @log_all_calls

    def _calculate_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate market microstructure features."""
        features = pd.DataFrame(index = data.index)
        features['spread'] = data['high'] - data['low']
        features['spread_pct'] = features['spread'] / data['close']
        features['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
        features['vwap'] = (features['typical_price'] * data['volume']).cumsum() / data['volume'].cumsum()
        features['price_to_vwap'] = data['close'] / features['vwap']
        features['dollar_volume'] = data['close'] * data['volume']
        features['log_dollar_volume'] = np.log1p(features['dollar_volume'])
        features['price_impact'] = data['close'].pct_change().abs() / (data['volume'] + 1)
        features['kyle_lambda'] = features['price_impact'].rolling(20).mean()
        features['order_flow_imbalance'] = np.where(data['close'] > data['open'], data['volume'], -data['volume'])
        features['ofi_cumsum'] = features['order_flow_imbalance'].cumsum()
        return features

    async def _create_feature_interactions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create sophisticated feature interactions."""
        if self.interaction_engine is not None:
            return await self.interaction_engine.create_interactions(data)
        else:
            return await self._basic_feature_interactions(data)

    async def _basic_feature_interactions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Basic feature interactions when enhanced engine is not available."""
        interactions = pd.DataFrame(index=data.index)

        # Price and volume interactions
        if 'close' in data.columns and 'volume' in data.columns:
            interactions['price_volume_interaction'] = data['close'].pct_change() * data['volume']

        # RSI and momentum interactions
        rsi_cols = [col for col in data.columns if 'rsi' in col.lower()]
        if rsi_cols and 'close' in data.columns:
            rsi_col = rsi_cols[0]
            interactions['rsi_momentum_interaction'] = data[rsi_col] * data['close'].pct_change()

        # Moving average crossovers
        ma_cols = [col for col in data.columns if 'sma' in col.lower() or 'ema' in col.lower()]
        if len(ma_cols) >= 2 and 'close' in data.columns:
            ma1, ma2 = ma_cols[0], ma_cols[1]
            crossover_signal = (data[ma1] > data[ma2]).fillna(False).astype(int)
            interactions['ma_crossover_signal'] = crossover_signal

        return interactions

    async def _create_regime_aware_features(self, data: pd.DataFrame, pipeline_state: Dict[str, Any]) -> pd.DataFrame:
        """Create regime-aware features."""
        regime_characteristics = pipeline_state.get('regime_characteristics', {})

        if self.regime_engine is not None:
            return await self.regime_engine.create_regime_features(data, regime_characteristics)
        else:
            return await self._basic_regime_aware_features(data, pipeline_state)

    async def _basic_regime_aware_features(self, data: pd.DataFrame, pipeline_state: Dict[str, Any]) -> pd.DataFrame:
        """Basic regime-aware features when enhanced engine is not available."""
        regime_features = pd.DataFrame(index=data.index)

        # Check for regime labels
        regime_col = None
        if 'regime_label' in data.columns:
            regime_col = 'regime_label'
        elif 'composite_cluster_id' in data.columns:
            regime_col = 'composite_cluster_id'

        if regime_col is None:
            self.logger.warning("No regime labels found for regime-aware features")
            return regime_features

        # Basic regime features
        regime_dummies = pd.get_dummies(data[regime_col], prefix='regime')
        regime_features = pd.concat([regime_features, regime_dummies], axis=1)

        # Regime duration
        regime_changed = (data[regime_col] != data[regime_col].shift(1)).fillna(False).astype(int)
        regime_features['regime_changed'] = regime_changed
        regime_features['regime_duration'] = data.groupby(
            (data[regime_col] != data[regime_col].shift()).cumsum()
        ).cumcount()

        return regime_features
    @log_all_calls

    def _combine_features(self, original_data: pd.DataFrame, feature_groups: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine all feature groups into a single DataFrame."""
        combined = original_data[['open', 'high', 'low', 'close', 'volume']].copy()
        if 'label' in original_data.columns:
            combined['label'] = original_data['label']
        for group_name, features in feature_groups.items():
            self.logger.info(f'Adding {len(features.columns)} {group_name} features')
            features = features.reindex(combined.index)
            if group_name != 'technical':
                features = features.add_prefix(f'{group_name}_')
            combined = pd.concat([combined, features], axis = 1)
        return combined
    @log_all_calls

    def _split_features(self, features: pd.DataFrame, train_end_idx: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split features into train and validation sets."""
        train_features = features.iloc[:train_end_idx]
        val_features = features.iloc[train_end_idx:]
        return (train_features, val_features)
    @log_all_calls

    def _basic_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fallback simple technical indicators if advanced calculator is missing."""
        out = pd.DataFrame(index = data.index)
        # RSI 14
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss.replace(0, np.nan))
        out['rsi_14'] = 100 - (100 / (1 + rs))
        # SMA/EMA
        out['sma_20'] = data['close'].rolling(20).mean()
        out['ema_12'] = data['close'].ewm(span = 12).mean()
        # Volatility
        out['volatility_20'] = data['close'].pct_change().rolling(20).std()
        # Volume ratio
        if 'volume' in data.columns:
            vma = data['volume'].rolling(20).mean()
            out['volume_ratio_10'] = data['volume'] / (vma.replace(0, np.nan))
        return out

    def _validate_and_fill_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Generic function to validate and fill NaN values in features.

        Args:
            features: DataFrame with features to validate
            data: Original market data for fill values

        Returns:
            DataFrame with validated and filled features

        Raises:
            ValueError: If any feature has >5% NaN values (relaxed threshold for technical indicators)
        """
        for col in features.columns:
            nan_pct = features[col].isna().mean() * 100

            # Check for small data gaps that can be forward-filled
            if nan_pct > 0 and nan_pct <= 0.5:  # Small gaps (< 0.5%)
                # Check if gaps are due to small time differences (< 2 seconds)
                if hasattr(data, 'index') and hasattr(data.index, 'to_series'):
                    # Calculate time gaps if data has timestamp index
                    time_gaps = data.index.to_series().diff().dt.total_seconds()
                    max_gap = time_gaps.max() if not time_gaps.empty else 0
                    if max_gap < 2:  # Small gaps can be forward-filled
                        features[col] = features[col].fillna(method='ffill')
                        nan_pct = features[col].isna().mean() * 100
                        if nan_pct == 0:
                            continue  # Successfully filled

            # Selective relaxed threshold only for indicators that naturally have NaN at the beginning
            # RSI needs lookback period, others can be stricter
            if 'rsi' in col.lower():
                threshold = 5.0  # RSI has natural NaN at start due to lookback
            elif any(keyword in col.lower() for keyword in ['stoch', 'williams', 'cci', 'ma', 'sma', 'ema', 'bb_', 'atr']):
                threshold = 1.0  # Other technical indicators get moderate threshold
            else:
                threshold = 0.1  # Strict threshold for all other features
            if nan_pct > threshold:
                raise ValueError(f'❌ Excessive NaN values in {col}: {nan_pct:.2f}% (threshold: {threshold}%)')

            # Apply appropriate fill strategy based on feature type
            if any(keyword in col.lower() for keyword in ['rsi', 'stoch', 'williams', 'cci']):
                # Oscillators: fill with neutral values
                if 'rsi' in col.lower():
                    features[col] = features[col].fillna(50)
                elif 'stoch' in col.lower():
                    features[col] = features[col].fillna(50)
                elif 'williams' in col.lower():
                    features[col] = features[col].fillna(-50)
                elif 'cci' in col.lower():
                    features[col] = features[col].fillna(0)
            elif any(keyword in col.lower() for keyword in ['ma', 'sma', 'ema', 'bb_', 'vwap']):
                # Price-based features: fill with current price
                features[col] = features[col].fillna(data['close'])
            elif any(keyword in col.lower() for keyword in ['volatility', 'atr']):
                # Volatility features: fill with rolling mean or zero
                features[col] = features[col].fillna(features[col].rolling(50).mean().fillna(0))
            elif any(keyword in col.lower() for keyword in ['momentum', 'roc', 'macd']):
                # Momentum features: fill with zero
                features[col] = features[col].fillna(0)
            elif any(keyword in col.lower() for keyword in ['ratio', 'position']):
                # Ratio/position features: fill with neutral values
                features[col] = features[col].fillna(0.5 if 'position' in col.lower() else 1.0)
            else:
                # Default: fill with zero
                features[col] = features[col].fillna(0)

        return features

    @log_all_calls

    def _calculate_feature_statistics(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistics about the features."""
        numeric_features = features.select_dtypes(include=[np.number])
        return {'n_samples': len(features), 'n_features': len(numeric_features.columns), 'missing_values': numeric_features.isnull().sum().to_dict(), 'feature_means': numeric_features.mean().to_dict(), 'feature_stds': numeric_features.std().to_dict(), 'correlation_matrix_sample': numeric_features.corr().iloc[:5, :5].to_dict()}

    def validate_outputs(self, pipeline_state: Dict[str, Any]) -> Tuple[bool, list]:
        """Validate step outputs.
        
        Args:
            pipeline_state: Updated pipeline state
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        if 'advanced_features' not in pipeline_state:
            errors.append('No advanced_features in pipeline state')
            return (False, errors)
        features_info = pipeline_state['advanced_features']
        for split in ['train', 'val']:
            if split not in features_info:
                errors.append(f'No {split} features path')
            else:
                path = Path(features_info[split])
                if not path.exists():
                    errors.append(f'{split} features file not found: {path}')
        if features_info.get('n_features', 0) < 10:
            errors.append(f"Too few features: {features_info.get('n_features', 0)}")
        return (len(errors) == 0, errors)

    def _create_sparse_feature_matrix(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, Any]:
        """Create sparse feature matrix if beneficial for memory efficiency."""
        if not SCIPY_SPARSE_AVAILABLE:
            return features_df, None

        try:
            # Calculate feature sparsity
            numeric_features = features_df.select_dtypes(include=[np.number])
            if numeric_features.empty:
                return features_df, None

            matrix_data = numeric_features.values

            # Calculate sparsity
            total_elements = matrix_data.size
            zero_elements = np.sum(np.abs(matrix_data) < 1e-10)  # Near-zero threshold
            sparsity = zero_elements / total_elements

            if sparsity >= 0.7:  # Use sparse matrix if >= 70% sparse
                self.logger.info(f'🔧 Creating sparse feature matrix (sparsity: {sparsity:.2f})')

                # Create sparse matrix
                sparse_matrix = sp.csr_matrix(matrix_data)

                # Calculate memory savings
                dense_memory = matrix_data.nbytes
                sparse_memory = (sparse_matrix.data.nbytes +
                               sparse_matrix.indices.nbytes +
                               sparse_matrix.indptr.nbytes)
                memory_savings = (dense_memory - sparse_memory) / dense_memory

                self.logger.info(f'💾 Sparse matrix memory savings: {memory_savings:.1%}')

                # Replace dense columns with sparse representation
                sparse_features_df = features_df.copy()
                for i, col in enumerate(numeric_features.columns):
                    # Store sparse column data as a custom attribute
                    setattr(sparse_features_df[col], '_sparse_data', sparse_matrix[:, i])

                return sparse_features_df, sparse_matrix
            else:
                self.logger.info(f'📊 Features not sparse enough (sparsity: {sparsity:.2f}), keeping dense')
                return features_df, None

        except Exception as e:
            self.logger.warning(f'Failed to create sparse feature matrix: {e}')
            return features_df, None

    def _optimize_dataframe_for_numpy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for NumPy operations and memory efficiency."""
        try:
            optimized_df = df.copy()

            # NumPy dtype optimizations
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']

            for col in numeric_cols:
                if col in optimized_df.columns:
                    if optimized_df[col].dtype == np.float64:
                        # Convert to float32 for memory efficiency
                        optimized_df[col] = optimized_df[col].astype(np.float32)

            # Optimize categorical data
            if 'composite_cluster_id' in optimized_df.columns:
                optimized_df['composite_cluster_id'] = optimized_df['composite_cluster_id'].astype('category')

            # Optimize datetime columns
            if 'timestamp' in optimized_df.columns and optimized_df['timestamp'].dtype == 'object':
                optimized_df['timestamp'] = pd.to_datetime(optimized_df['timestamp'])

            # Memory usage optimization
            memory_before = optimized_df.memory_usage(deep=True).sum()

            # Convert string columns to categorical if they have low cardinality
            for col in optimized_df.select_dtypes(include=['object']).columns:
                if len(optimized_df[col].unique()) / len(optimized_df[col]) < 0.1:  # Less than 10% unique
                    optimized_df[col] = optimized_df[col].astype('category')

            memory_after = optimized_df.memory_usage(deep=True).sum()
            memory_savings = (memory_before - memory_after) / memory_before * 100

            if memory_savings > 1:  # Only log significant savings
                self.logger.info(f"💾 NumPy optimization saved {memory_savings:.1f}% memory")

            return optimized_df

        except Exception as e:
            self.logger.warning(f"DataFrame optimization failed: {e}, returning original")
            return df

    def get_required_inputs(self) -> list:
        """Get list of required inputs for this step."""
        return ['symbol', 'exchange', 'labeled_data']

    def get_produced_outputs(self) -> list:
        """Get list of outputs produced by this step."""
        return ['advanced_features', 'feature_statistics']

    def get_dependencies(self) -> list:
        """Get list of step dependencies."""
        return ['05']