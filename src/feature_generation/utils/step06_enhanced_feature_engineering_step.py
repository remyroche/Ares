"""
Enhanced Step06 Feature Engineering Step with Modular Approach

This module implements a modular, memory-efficient feature engineering step with:
- Reduced nested functions using modular approach
- Integration of enhanced feature engineering components
- Strict temporal validation and lookahead bias prevention
- Memory-efficient chunking for large datasets
- Comprehensive error handling and validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from pathlib import Path
import json
import time
import warnings
from contextlib import nullcontext

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

# Import base step and utilities
# Import BaseStep from training module
from src.training.base_step import BaseStep
from src.utils.comprehensive_function_logger import (
    log_step_functions, log_important_calls, log_all_calls,
    log_internal_call, log_step_progress, log_data_operation
)
# Import math validation functions from shared module
from .math_validation import safe_divide, safe_log, safe_sqrt, validate_positive

# Import enhanced components
from .step06_enhanced_feature_engineering import EnhancedFeatureEngineering

# Import validation framework
try:
    from .step06_enhanced_validation_framework import (
        step06_function_validator, step06_function_tracker,
        step06_validation_context, ValidationLevel, FunctionStatus
    )
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False

    def step06_function_validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def step06_function_tracker(func):
        return func

    def step06_validation_context(*args, **kwargs):
        return nullcontext()

    class ValidationLevel:
        BASIC = 'basic'
        DETAILED = 'detailed'
        COMPREHENSIVE = 'comprehensive'

logger = logging.getLogger(__name__)

class EnhancedFeatureEngineeringStep(BaseStep):
    """
    Enhanced Step 6: Feature Engineering with modular approach and advanced optimizations.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize enhanced feature engineering step."""
        super().__init__(config, '06', 'enhanced_feature_engineering')

        # Initialize enhanced feature engineering
        self.enhanced_engine = EnhancedFeatureEngineering(config)

        # Configuration
        self.feature_config = config.get('step06_feature_engineering', {
            'use_technical_indicators': True,
            'use_interaction_features': True,
            'use_regime_features': True,
            'use_sr_features': True,
            'use_order_flow_proxies': True,  # Enable order flow proxies (replaces aggtrades)
            'use_dynamic_lookback': True,
            'chunk_size': 10000,
            'max_features': 500,
            'polynomial_degree': 2,
            'correlation_threshold': 0.95,
            'memory_limit_mb': 1000,
            'lookback_periods': {
                'RSI': [7, 14, 21],
                'MACD': [12, 26, 52],
                'Bollinger_Bands': [10, 20, 50],
                'SMA': [5, 20, 100],
                'EMA': [8, 21, 55],
                'ATR': [7, 14, 30],
                'Stochastic': [7, 14, 30],
                'ADX': [7, 14, 25],
                'OBV': [10, 20, 50],
                'MFI': [7, 14, 30]
            },
            # SR-specific configuration
            'sr_detection_window': 20,
            'min_touches_required': 3,
            'touch_tolerance': 0.002,
            'min_bounce_strength': 0.001,
            'volume_threshold_multiplier': 1.5,
            'use_pre_optimized_sr_parameters': True,
            'sr_optimization_config': {
                'optimization_method': 'adaptive_grid_search',
                'enable_hardware_optimization': True,
                'enable_parallel_processing': True,
                'max_parallel_workers': None
            }
        })

        # Performance tracking
        self.performance_metrics = {
            'total_processing_time': 0.0,
            'total_memory_used_mb': 0.0,
            'features_created': 0,
            'chunks_processed': 0,
            'validation_errors': 0
        }

        self.logger.info("🚀 Enhanced Feature Engineering Step initialized")
        self.logger.info(f"   Chunk size: {self.feature_config['chunk_size']}")
        self.logger.info(f"   Max features: {self.feature_config['max_features']}")
        self.logger.info(f"   Polynomial degree: {self.feature_config['polynomial_degree']}")

    @log_step_functions
    def _initialize_step(self) -> None:
        """Initialize step-specific components."""
        try:
            # Initialize enhanced feature engineering components
            self.enhanced_engine = EnhancedFeatureEngineering(self.config)
            self.logger.info('✅ Enhanced feature engineering components initialized')
        except Exception as e:
            self.logger.error(f'❌ Failed to initialize enhanced components: {e}')
            raise

    @step06_function_validator(validation_level=ValidationLevel.COMPREHENSIVE)
    def validate_inputs(self, training_input: Dict[str, Any],
                       pipeline_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate step inputs with enhanced validation.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            Tuple of (is_valid, errors)
        """
        with step06_validation_context('validate_inputs', 'feature_engineering'):
            self.logger.info(f'🔍 Starting enhanced input validation')
            self.logger.info(f'   Training input keys: {list(training_input.keys())}')
            self.logger.info(f'   Pipeline state keys: {list(pipeline_state.keys())}')

        errors = []

        # Check for labeled data
        if 'labeled_data' not in pipeline_state:
            if not any(f'{split}_data' in pipeline_state for split in ['train', 'val', 'test']):
                errors.append('No labeled data from step 5')

        # Validate data quality
        data_to_validate = None
        for key in ['labeled_data', 'train_data', 'val_data', 'test_data']:
            if key in pipeline_state:
                data_to_validate = pipeline_state[key]
                break

        if data_to_validate is not None:
            validation_result = self._validate_data_quality(data_to_validate)
            if not validation_result['is_valid']:
                errors.extend(validation_result['errors'])

        # Validate configuration
        config_errors = self._validate_configuration()
        errors.extend(config_errors)

        is_valid = len(errors) == 0
        if not is_valid:
            self.performance_metrics['validation_errors'] += len(errors)
            self.logger.error(f'❌ Input validation failed: {errors}')
        else:
            self.logger.info('✅ Input validation passed')

        return is_valid, errors

    def _validate_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality for feature engineering."""
        errors = []

        try:
            # Check data shape
            if len(data) < 50:
                errors.append(f'Insufficient data: {len(data)} rows (minimum 50 required)')

            # Check required columns
            required_columns = ['open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                errors.append(f'Missing required columns: {missing_columns}')

            # Check for valid prices
            for col in required_columns:
                if col in data.columns:
                    if (data[col] <= 0).any():
                        errors.append(f'Invalid prices in {col}: non-positive values found')
                    if data[col].isna().any():
                        errors.append(f'NaN values in {col}')

            # Check temporal consistency
            if isinstance(data.index, pd.DatetimeIndex):
                if not data.index.is_monotonic_increasing:
                    errors.append('Data index is not temporally ordered')

        except Exception as e:
            errors.append(f'Data validation error: {e}')

        return {'is_valid': len(errors) == 0, 'errors': errors}

    def _validate_configuration(self) -> List[str]:
        """Validate configuration parameters."""
        errors = []

        try:
            # Validate chunk size
            chunk_size = self.feature_config.get('chunk_size', 10000)
            if chunk_size < 100:
                errors.append(f'Chunk size too small: {chunk_size} (minimum 100)')
            if chunk_size > 100000:
                errors.append(f'Chunk size too large: {chunk_size} (maximum 100000)')

            # Validate max features
            max_features = self.feature_config.get('max_features', 500)
            if max_features < 10:
                errors.append(f'Max features too small: {max_features} (minimum 10)')
            if max_features > 10000:
                errors.append(f'Max features too large: {max_features} (maximum 10000)')

            # Validate polynomial degree
            poly_degree = self.feature_config.get('polynomial_degree', 2)
            if poly_degree < 1 or poly_degree > 3:
                errors.append(f'Polynomial degree invalid: {poly_degree} (must be 1-3)')

        except Exception as e:
            errors.append(f'Configuration validation error: {e}')

        return errors

    @step06_function_validator(validation_level=ValidationLevel.COMPREHENSIVE)
    async def execute_logic(self, training_input: Dict[str, Any],
                           pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute enhanced feature engineering logic with modular approach.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            Updated pipeline state
        """
        start_time = time.time()

        with step06_validation_context('execute_logic', 'feature_engineering'):
            self.logger.info(f'🔧 Starting enhanced feature engineering execution')
            self.logger.info(f'   Training input keys: {list(training_input.keys())}')
            self.logger.info(f'   Pipeline state keys: {list(pipeline_state.keys())}')
            self.logger.info(f'   Feature config: technical={self.feature_config.get("use_technical_indicators")}, '
                           f'interactions={self.feature_config.get("use_interaction_features")}, '
                           f'regime={self.feature_config.get("use_regime_features")}, '
                           f'sr={self.feature_config.get("use_sr_features")}, '
                           f'order_flow={self.feature_config.get("use_order_flow_proxies")}')
            self.logger.info(f'   Order flow proxies enabled: {self.feature_config.get("use_order_flow_proxies", True)} (replaces aggtrades features)')

        try:
            # Store pipeline state for SR feature extraction
            self.pipeline_state = pipeline_state

            # Step 1: Get and validate data
            data_dict = self._get_data_to_process(pipeline_state)
            self.logger.info(f'📊 Processing {len(data_dict)} data splits')

            # Step 2: Process each data split
            engineered_data = {}
            feature_statistics = {}

            for split_name, data in data_dict.items():
                self.logger.info(f'🔧 Processing {split_name} split: {data.shape}')

                # Process with enhanced feature engineering
                engineered_split = await self._process_data_split(data, split_name)

                # Calculate statistics
                stats = self._calculate_feature_statistics(engineered_split)

                engineered_data[split_name] = engineered_split
                feature_statistics[split_name] = stats

                self.logger.info(f'✅ {split_name} split processed: {engineered_split.shape}')

            # Step 3: Feature selection and optimization
            if self.feature_config.get('feature_selection', {}).get('enabled', True):
                self.logger.info('🎯 Performing feature selection...')
                engineered_data, selected_features = await self._perform_feature_selection(
                    engineered_data, feature_statistics
                )
                self.logger.info(f'✅ Feature selection complete: {len(selected_features)} features selected')
            else:
                selected_features = self._get_all_feature_columns(engineered_data)

            # Step 4: Generate reports
            reports = self._generate_feature_reports(engineered_data, feature_statistics, selected_features)

            # Step 5: Update pipeline state
            pipeline_state.update({
                'engineered_data': engineered_data,
                'feature_statistics': feature_statistics,
                'selected_features': selected_features,
                'feature_reports': reports,
                'feature_config': self.feature_config,
                'performance_metrics': self.performance_metrics.copy()
            })

            # Update individual splits
            for split in ['train', 'val', 'test']:
                if split in engineered_data and f'{split}_data' in pipeline_state:
                    pipeline_state[f'{split}_data'] = engineered_data[split]

            # Step 6: Save outputs
            await self._save_outputs(training_input, pipeline_state)

            # Update performance metrics
            processing_time = time.time() - start_time
            self.performance_metrics['total_processing_time'] += processing_time
            self.performance_metrics['features_created'] = len(selected_features)

            self.logger.info(f'✅ Enhanced feature engineering completed in {processing_time:.2f}s')
            self.logger.info(f'   Features created: {len(selected_features)}')
            self.logger.info(f'   Total processing time: {self.performance_metrics["total_processing_time"]:.2f}s')

            return pipeline_state

        except Exception as e:
            self.logger.error(f'❌ Enhanced feature engineering failed: {e}')
            self.performance_metrics['validation_errors'] += 1
            raise

    async def _process_data_split(self, data: pd.DataFrame, split_name: str) -> pd.DataFrame:
        """
        Process a single data split with enhanced feature engineering.

        Args:
            data: Data to process
            split_name: Name of the data split

        Returns:
            Processed data with enhanced features
        """
        try:
            # Start with base data
            processed_data = data.copy()

            # Step 1: Extract technical indicators using batch processing
            if self.feature_config.get('use_technical_indicators', True):
                self.logger.info(f'📊 Extracting technical indicators for {split_name}')

                lookback_periods = self.feature_config.get('lookback_periods', {})
                indicators = self.enhanced_engine.extract_indicators_batch(
                    processed_data, lookback_periods
                )

                # Combine with original data
                processed_data = pd.concat([processed_data, indicators], axis=1)
                self.logger.info(f'   Added {indicators.shape[1]} technical indicators')

            # Step 2: Create sophisticated interactions
            if self.feature_config.get('use_interaction_features', True):
                self.logger.info(f'🔗 Creating sophisticated interactions for {split_name}')

                # Apply temporal validation to prevent lookahead bias
                current_idx = len(processed_data) - 1  # Use all data up to current point
                interactions = self.enhanced_engine.create_sophisticated_interactions(
                    processed_data, current_idx
                )

                # Use only the new interaction features
                interaction_cols = [col for col in interactions.columns
                                  if col not in processed_data.columns]
                if interaction_cols:
                    processed_data = pd.concat([
                        processed_data,
                        interactions[interaction_cols]
                    ], axis=1)
                    self.logger.info(f'   Added {len(interaction_cols)} interaction features')

            # Step 3: Add regime-aware features
            if self.feature_config.get('use_regime_features', True) and 'regime_label' in data.columns:
                self.logger.info(f'🏛️ Adding regime-aware features for {split_name}')
                regime_features = self._create_regime_features(processed_data)
                processed_data = pd.concat([processed_data, regime_features], axis=1)
                self.logger.info(f'   Added {regime_features.shape[1]} regime features')

            # Step 4: Add support/resistance features
            if self.feature_config.get('use_sr_features', True):
                self.logger.info(f'📈 Adding S/R features for {split_name}')
                sr_features = self._create_sr_features(processed_data)
                processed_data = pd.concat([processed_data, sr_features], axis=1)
                self.logger.info(f'   Added {sr_features.shape[1]} S/R features')

            # Step 5: Add time-based features
            time_features = self._create_time_features(processed_data)
            processed_data = pd.concat([processed_data, time_features], axis=1)

            # Step 6: Add order flow proxies (replaces aggtrades features)
            if self.feature_config.get('use_order_flow_proxies', True):
                self.logger.info(f'💹 Adding order flow proxies for {split_name}')
                order_flow_features = self._create_order_flow_proxies(processed_data)
                processed_data = pd.concat([processed_data, order_flow_features], axis=1)
                self.logger.info(f'   Added {order_flow_features.shape[1]} order flow proxy features')

            # Step 7: Clean and validate final data
            processed_data = self._clean_and_validate_data(processed_data)

            return processed_data

        except Exception as e:
            self.logger.error(f'❌ Failed to process {split_name} split: {e}')
            raise

    def _create_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create regime-aware features."""
        regime_features = pd.DataFrame(index=data.index)

        if 'regime_label' in data.columns:
            # Create regime dummy variables
            regime_dummies = pd.get_dummies(data['regime_label'], prefix='regime')
            regime_features = pd.concat([regime_features, regime_dummies], axis=1)

            # Create regime transition features
            regime_features['regime_changed'] = (data['regime_label'] != data['regime_label'].shift(1)).astype(int)
            regime_features['time_in_regime'] = data.groupby(
                (data['regime_label'] != data['regime_label'].shift()).cumsum()
            ).cumcount()

            # Create regime-specific technical indicators
            for regime in data['regime_label'].unique():
                if pd.notna(regime):
                    regime_mask = data['regime_label'] == regime
                    regime_features[f'regime_{regime}_count'] = regime_mask.cumsum()

        return regime_features

    def _create_order_flow_proxies(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive order flow features using real taker data + enhanced kline proxies.

        This leverages the actual taker_buy_base and taker_buy_quote columns from Binance API
        to create much more accurate order flow features, supplemented by kline-based proxies.
        """
        order_flow_features = pd.DataFrame(index=data.index)

        try:
            # Import numpy for mathematical operations

            # === TAKER DATA FEATURES (HIGH ACCURACY) ===
            # Check if we have real taker data from Binance API
            has_taker_data = False
            taker_base = None
            taker_quote = None

            # Try different column names for taker data
            taker_base_cols = ['taker_buy_base_asset_volume', 'taker_buy_base']
            taker_quote_cols = ['taker_buy_quote_asset_volume', 'taker_buy_quote']

            for col in taker_base_cols:
                if col in data.columns:
                    taker_base = data[col]
                    has_taker_data = True
                    break

            for col in taker_quote_cols:
                if col in data.columns:
                    taker_quote = data[col]
                    has_taker_data = True
                    break

            if has_taker_data and taker_base is not None and taker_quote is not None:
                self.logger.info("✅ Using real taker data for enhanced order flow features")

                # 1. REAL TAKER BUY/SELL FLOW (Direct from API)
                total_volume = data['volume']
                taker_ratio = safe_divide(taker_base, total_volume)
                order_flow_features['taker_buy_ratio'] = taker_ratio
                order_flow_features['taker_sell_ratio'] = 1.0 - taker_ratio  # Passive (maker) volume ratio

                # 2. TAKER VALUE FLOW (Quote volume based)
                total_quote_volume = data['quote_volume']
                taker_quote_ratio = safe_divide(taker_quote, total_quote_volume)
                order_flow_features['taker_quote_ratio'] = taker_quote_ratio

                # 3. MARKET AGGRESSION INDEX (Real taker vs passive)
                maker_volume = total_volume - taker_base
                aggression_index = safe_divide(taker_base, maker_volume)
                order_flow_features['market_aggression_index'] = aggression_index
                order_flow_features['aggression_score'] = (aggression_index * 100).clip(0, 1000)  # Scaled for readability

                # 4. TAKER BUYING PRESSURE (Average price paid by aggressive buyers)
                taker_avg_price = safe_divide(taker_quote, taker_base)
                market_price = data['close']
                order_flow_features['taker_avg_price'] = taker_avg_price
                order_flow_features['taker_price_deviation'] = safe_divide((taker_avg_price - market_price), market_price)

                # 5. ORDER FLOW IMBALANCE (Real buy vs sell pressure)
                order_flow_features['order_flow_imbalance'] = safe_divide((taker_base - maker_volume), total_volume)

                # 6. TAKER VOLUME MOMENTUM
                order_flow_features['taker_volume_momentum'] = taker_base.pct_change(5)
                order_flow_features['taker_quote_momentum'] = taker_quote.pct_change(5)

                # 7. TAKER PARTICIPATION RATE (How much of total volume is aggressive)
                order_flow_features['taker_participation_rate'] = safe_divide(taker_base, total_volume)

                # 8. TAKER EFFICIENCY (Value per volume for taker trades)
                order_flow_features['taker_efficiency'] = safe_divide(taker_quote, taker_base)

                # 9. TAKER FLOW DIRECTION (Net aggressive buying/selling)
                taker_flow = taker_base - maker_volume
                order_flow_features['taker_flow'] = taker_flow
                order_flow_features['taker_flow_ratio'] = safe_divide(taker_flow, total_volume)

                # 10. INSTITUTIONAL vs RETAIL INDICATOR (High participation + stable pricing = institutional)
                taker_stability = taker_avg_price.rolling(10).std()
                participation_rate = order_flow_features['taker_participation_rate']
                order_flow_features['institutional_indicator'] = participation_rate / (taker_stability + 0.001)

                # 11. TAKER VOLUME VOLATILITY (How erratic aggressive trading is)
                order_flow_features['taker_volume_volatility'] = taker_base.rolling(20).std() / taker_base.rolling(20).mean().replace(0, 1)

                # 12. BUY/SELL PRESSURE RATIO
                order_flow_features['buy_sell_pressure_ratio'] = safe_divide(taker_base, maker_volume)

                # 13. TAKER CONCENTRATION (How concentrated aggressive buying is at certain price levels)
                order_flow_features['taker_concentration'] = safe_divide(taker_quote, taker_base)  # Price per unit volume

                # 14. MARKET IMPACT PROXY (Real price impact from taker activity)
                price_change = data['close'].pct_change()
                order_flow_features['taker_market_impact'] = price_change * safe_sqrt(taker_base)

                # 15. TAKER TREND ANALYSIS
                order_flow_features['taker_trend_5'] = taker_base.rolling(5).mean() / taker_base.rolling(20).mean().replace(0, 1)
                order_flow_features['taker_trend_10'] = taker_base.rolling(10).mean() / taker_base.rolling(30).mean().replace(0, 1)

            else:
                self.logger.info("ℹ️ No real taker data available, using enhanced kline proxies")

            # === ENHANCED KLINE-BASED PROXIES (FALLBACK OR SUPPLEMENT) ===

            # 1. BUYER/SELLER INITIATED TRADE FLOW PROXY (Enhanced with taker data if available)
            close_position = safe_divide((data['close'] - data['open']), (data['high'] - data['low']))
            base_flow_proxy = np.sign(close_position)

            # Enhance with taker data if available
            if has_taker_data and taker_base is not None:
                taker_weighted_flow = np.sign(taker_base - taker_base.shift(1))
                order_flow_features['buyer_seller_flow_proxy'] = 0.7 * base_flow_proxy + 0.3 * taker_weighted_flow
            else:
                order_flow_features['buyer_seller_flow_proxy'] = base_flow_proxy

            # 2. ORDER MARKET IMBALANCE (OMI) PROXY (Enhanced with taker data)
            midpoint = (data['high'] + data['low']) / 2
            volume_weighted_deviation = ((data['close'] - midpoint) / midpoint) * safe_sqrt(data['volume'])

            if has_taker_data and taker_base is not None:
                # Use real taker data for more accurate imbalance calculation
                taker_weighted_deviation = volume_weighted_deviation * (taker_base / data['volume'].replace(0, 1))
                omi_base = taker_weighted_deviation
            else:
                omi_base = volume_weighted_deviation

            omi_mean = omi_base.rolling(20).mean()
            omi_std = omi_base.rolling(20).std()
            omi_zscore = safe_divide((omi_base - omi_mean), omi_std)
            order_flow_features['omi_proxy'] = omi_zscore

            # 3. ORDER BOOK PRESSURE PROXY (Enhanced)
            price_position = safe_divide((data['close'] - data['low']), (data['high'] - data['low']))
            volume_normalized = safe_divide(data['volume'], data['volume'].rolling(20).mean())

            if has_taker_data and taker_base is not None:
                # Weight by taker participation
                taker_participation = safe_divide(taker_base, data['volume'].replace(0, 1))
                pressure_proxy = price_position * safe_log(volume_normalized + 1) * (1 + taker_participation)
            else:
                pressure_proxy = price_position * safe_log(volume_normalized + 1)

            order_flow_features['order_book_pressure_proxy'] = pressure_proxy

            # 4. MARKET MAKER vs RETAIL ORDER FLOW PROXY (Enhanced with taker data)
            intrabar_range = safe_divide((data['high'] - data['low']), data['close'])
            volume_per_range = safe_divide(data['volume'], intrabar_range)
            volume_per_range_ma = volume_per_range.rolling(10).mean()

            if has_taker_data and taker_base is not None:
                # Adjust for taker participation - high taker ratio suggests more institutional activity
                taker_adjustment = safe_divide(taker_base, data['volume'].replace(0, 1))
                retail_proxy = safe_divide(volume_per_range, volume_per_range_ma) * (2 - taker_adjustment)  # Higher taker = more institutional
            else:
                retail_proxy = safe_divide(volume_per_range, volume_per_range_ma)

            order_flow_features['market_maker_retail_proxy'] = retail_proxy

            # 5. ORDER FLOW TOXICITY PROXY (Enhanced with taker data)
            returns = data['close'].pct_change()

            if has_taker_data and taker_base is not None:
                # Use taker volume for more accurate price impact calculation
                taker_returns = returns * safe_sqrt(taker_base)
                toxicity_proxy = taker_returns.rolling(5).std()
            else:
                volume_returns = returns * safe_sqrt(data['volume'])
                toxicity_proxy = volume_returns.rolling(5).std()

            order_flow_features['order_flow_toxicity_proxy'] = toxicity_proxy

            # 6. REAL ORDER DIRECTION CLASSIFICATION PROXY
            price_direction = np.sign(data['close'] - data['open'])
            volume_confirmation = np.sign(data['volume'] - data['volume'].shift(1))
            order_flow_features['order_direction_proxy'] = price_direction * volume_confirmation

            # 7. TRUE ORDER MARKET IMBALANCE PROXY (Enhanced)
            price_momentum = data['close'].pct_change(3)

            if has_taker_data and taker_base is not None:
                taker_momentum = taker_base.pct_change(3)
                true_omi = price_momentum - taker_momentum
            else:
                volume_momentum = data['volume'].pct_change(3)
                true_omi = price_momentum - volume_momentum

            order_flow_features['true_omi_proxy'] = true_omi

            # 8. BID/ASK PRESSURE ANALYSIS PROXY
            upper_pressure = safe_divide((data['high'] - data['close']), (data['high'] - data['low']))
            lower_pressure = safe_divide((data['close'] - data['low']), (data['high'] - data['low']))
            order_flow_features['bid_pressure_proxy'] = lower_pressure
            order_flow_features['ask_pressure_proxy'] = upper_pressure

            # 9. TRADE SOURCE IDENTIFICATION PROXY (Enhanced)
            volatility = data['close'].pct_change().rolling(10).std()
            vol_volatility_ratio = safe_divide(data['volume'], volatility)
            vol_volatility_ratio_ma = vol_volatility_ratio.rolling(20).mean()

            if has_taker_data and taker_base is not None:
                # Use taker data for more accurate institutional vs retail classification
                taker_volatility = taker_base.rolling(10).std()
                taker_vol_ratio = safe_divide(taker_base, taker_volatility.replace(0, 1))
                trade_source_proxy = safe_divide(taker_vol_ratio, taker_vol_ratio.rolling(20).mean().replace(0, 1))
            else:
                trade_source_proxy = safe_divide(vol_volatility_ratio, vol_volatility_ratio_ma)

            order_flow_features['trade_source_proxy'] = trade_source_proxy

            # Handle any NaN values that might have been created
            order_flow_features = order_flow_features.fillna(0.0)

            # Log successful creation
            feature_count = len(order_flow_features.columns)
            if has_taker_data:
                self.logger.info(f"✅ Created {feature_count} enhanced order flow features using real taker data + kline proxies")
            else:
                self.logger.debug(f"✅ Created {feature_count} order flow proxy features using kline data only")

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to create order flow features: {e}")
            # Return empty DataFrame if creation fails
            order_flow_features = pd.DataFrame(index=data.index)

        return order_flow_features

    def _create_sr_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create support/resistance features with three-tier system: Enhanced → Basic → Fallback."""

        # Extract SR levels if available in pipeline state
        sr_levels = None
        if hasattr(self, 'pipeline_state') and 'sr_levels' in self.pipeline_state:
            sr_levels = self.pipeline_state['sr_levels']

        # Get regime labels if available
        regime_labels = None
        if 'regime_label' in data.columns:
            regime_labels = data['regime_label']

        # Tier 1: Try Enhanced SR Feature Extractor with Historical Integration
        try:
            from .enhanced_sr_feature_extractor import (
                get_enhanced_sr_feature_extractor, SRFeatureConfig, HistoricalSRConfig
            )

            # Create SR feature configuration
            sr_config = SRFeatureConfig(
                enable_basic_sr_features=True,
                enable_advanced_sr_features=True,
                enable_sr_bounce_signals=True,
                enable_sr_strength_calculation=True,
                enable_regime_aware_sr=True,
                use_pre_optimized_parameters=True,
                sr_detection_window=self.feature_config.get('sr_detection_window', 20),
                min_touches_required=self.feature_config.get('min_touches_required', 3),
                touch_tolerance=self.feature_config.get('touch_tolerance', 0.002),
                min_bounce_strength=self.feature_config.get('min_bounce_strength', 0.001),
                volume_threshold_multiplier=self.feature_config.get('volume_threshold_multiplier', 1.5)
            )

            # Create historical configuration
            historical_config = HistoricalSRConfig(
                load_historical_levels=True,
                historical_data_path=self.feature_config.get('historical_data_path', 'sr_levels_history.json'),
                current_levels_path=self.feature_config.get('current_levels_path', 'sr_levels.json'),
                enable_level_persistence_features=True,
                enable_historical_touch_analysis=True,
                enable_bounce_success_analysis=True,
                enable_level_evolution_features=True,
                enable_ml_ready_features=True,
                enable_trading_features=True
            )

            # Get enhanced SR feature extractor
            sr_extractor = get_enhanced_sr_feature_extractor(sr_config, historical_config)

            # Extract enhanced SR features with historical integration
            sr_features = sr_extractor.extract_historical_sr_features(data, sr_levels, regime_labels)

            self.logger.info(f"✅ Tier 1: Extracted {sr_features.shape[1]} enhanced SR features with historical integration")
            return sr_features

        except ImportError as e:
            self.logger.warning(f"Tier 1 failed: Enhanced SR feature extractor not available: {e}")
        except Exception as e:
            self.logger.warning(f"Tier 1 failed: Enhanced SR feature extraction error: {e}")

        # Tier 2: Try Basic SR Feature Extractor
        try:
            from .sr_feature_extractor import get_sr_feature_extractor, SRFeatureConfig

            sr_config = SRFeatureConfig(
                enable_basic_sr_features=True,
                enable_advanced_sr_features=True,
                enable_sr_bounce_signals=True,
                enable_sr_strength_calculation=True,
                enable_regime_aware_sr=True,
                use_pre_optimized_parameters=True,
                sr_detection_window=self.feature_config.get('sr_detection_window', 20),
                min_touches_required=self.feature_config.get('min_touches_required', 3),
                touch_tolerance=self.feature_config.get('touch_tolerance', 0.002),
                min_bounce_strength=self.feature_config.get('min_bounce_strength', 0.001),
                volume_threshold_multiplier=self.feature_config.get('volume_threshold_multiplier', 1.5)
            )

            sr_extractor = get_sr_feature_extractor(sr_config)

            # Extract basic SR features
            sr_features = sr_extractor.extract_sr_features(data, sr_levels, regime_labels)

            self.logger.info(f"✅ Tier 2: Extracted {sr_features.shape[1]} basic SR features")
            return sr_features

        except ImportError as e:
            self.logger.warning(f"Tier 2 failed: Basic SR feature extractor not available: {e}")
        except Exception as e:
            self.logger.warning(f"Tier 2 failed: Basic SR feature extraction error: {e}")

        # Tier 3: Use Fallback SR Features
        self.logger.warning("Tier 3: Using fallback SR features")
        return self._create_fallback_sr_features(data)

    def _create_fallback_sr_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create basic support/resistance features as fallback."""
        sr_features = pd.DataFrame(index=data.index)

        # Price position features
        sr_features['price_position_20'] = safe_divide(
            data['close'] - data['low'].rolling(20).min(),
            data['high'].rolling(20).max() - data['low'].rolling(20).min(),
            default=0.5
        )

        # Distance to support/resistance
        sr_features['dist_to_high_20'] = safe_divide(
            data['high'].rolling(20).max() - data['close'],
            data['close'],
            default=0.0
        )
        sr_features['dist_to_low_20'] = safe_divide(
            data['close'] - data['low'].rolling(20).min(),
            data['close'],
            default=0.0
        )

        # Breakout features
        sr_features['breakout_high_20'] = (data['close'] > data['high'].rolling(20).max().shift(1)).astype(int)
        sr_features['breakout_low_20'] = (data['close'] < data['low'].rolling(20).min().shift(1)).astype(int)

        return sr_features

    def _create_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        time_features = pd.DataFrame(index=data.index)

        if hasattr(data.index, 'hour'):
            time_features['hour'] = data.index.hour
            time_features['minute'] = data.index.minute
            time_features['hour_sin'] = np.sin(2 * np.pi * data.index.hour / 24)
            time_features['hour_cos'] = np.cos(2 * np.pi * data.index.hour / 24)

        if hasattr(data.index, 'dayofweek'):
            time_features['dayofweek'] = data.index.dayofweek
            time_features['is_monday'] = (data.index.dayofweek == 0).astype(int)
            time_features['is_friday'] = (data.index.dayofweek == 4).astype(int)

        return time_features

    def _clean_and_validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate processed data."""
        # Remove infinite values
        data = data.replace([np.inf, -np.inf], np.nan)

        # Forward fill and then fill remaining NaN with 0
        data = data.ffill().fillna(0)

        # Remove duplicate columns
        data = data.loc[:, ~data.columns.duplicated()]

        return data

    def _get_data_to_process(self, pipeline_state: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Get data splits to process."""
        data_dict = {}

        for split in ['train', 'val', 'test']:
            if f'{split}_data' in pipeline_state:
                data_dict[split] = pipeline_state[f'{split}_data'].copy()

        if not data_dict and 'labeled_data' in pipeline_state:
            data_dict['all'] = pipeline_state['labeled_data'].copy()

        return data_dict

    def _calculate_feature_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistics for engineered features."""
        feature_cols = [
            col for col in data.columns
            if col.startswith(('feature_', 'regime_', 'sr_', 'time_', 'RSI_', 'MACD_', 'SMA_', 'EMA_', 'ATR_', 'BB_', 'Stoch_', 'ADX_', 'OBV', 'MFI_', 'poly_', 'cross_', 'pattern_', 'momentum_', 'regime_'))
        ]

        stats = {
            'n_features': len(feature_cols),
            'feature_names': feature_cols,
            'missing_values': {},
            'zero_variance': [],
            'high_correlation_pairs': []
        }

        for col in feature_cols:
            missing_pct = data[col].isna().sum() / len(data) * 100
            if missing_pct > 0:
                stats['missing_values'][col] = missing_pct

        for col in feature_cols:
            if data[col].std() < 1e-10:
                stats['zero_variance'].append(col)

        # Calculate correlations for a sample
        if len(feature_cols) > 1:
            sample_size = min(1000, len(data))
            sample_data = data[feature_cols].sample(n=sample_size)
            corr_matrix = sample_data.corr()

            for i in range(len(feature_cols)):
                for j in range(i + 1, len(feature_cols)):
                    if abs(corr_matrix.iloc[i, j]) > 0.95:
                        stats['high_correlation_pairs'].append((
                            feature_cols[i], feature_cols[j], corr_matrix.iloc[i, j]
                        ))

        return stats

    async def _perform_feature_selection(self, engineered_data: Dict[str, pd.DataFrame],
                                       feature_statistics: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        """Perform feature selection."""
        train_data = engineered_data.get('train', next(iter(engineered_data.values())))
        all_features = [
            col for col in train_data.columns
            if col.startswith(('feature_', 'regime_', 'sr_', 'time_', 'RSI_', 'MACD_', 'SMA_', 'EMA_', 'ATR_', 'BB_', 'Stoch_', 'ADX_', 'OBV', 'MFI_', 'poly_', 'cross_', 'pattern_', 'momentum_', 'regime_'))
        ]

        # Remove zero variance features
        zero_var_features = set()
        for stats in feature_statistics.values():
            zero_var_features.update(stats.get('zero_variance', []))

        valid_features = [f for f in all_features if f not in zero_var_features]

        # Remove highly correlated features
        to_remove = set()
        for stats in feature_statistics.values():
            for feat1, feat2, corr in stats.get('high_correlation_pairs', []):
                to_remove.add(feat2)

        valid_features = [f for f in valid_features if f not in to_remove]

        # Limit to max features
        max_features = self.feature_config.get('max_features', 500)
        if len(valid_features) > max_features:
            valid_features = valid_features[:max_features]

        # Select features
        selected_data = {}
        base_columns = [
            col for col in train_data.columns
            if not col.startswith(('feature_', 'regime_', 'sr_', 'time_', 'RSI_', 'MACD_', 'SMA_', 'EMA_', 'ATR_', 'BB_', 'Stoch_', 'ADX_', 'OBV', 'MFI_', 'poly_', 'cross_', 'pattern_', 'momentum_', 'regime_'))
        ]
        selected_columns = base_columns + valid_features

        for split_name, data in engineered_data.items():
            selected_data[split_name] = data[selected_columns]

        self.logger.info(f'✅ Selected {len(valid_features)} features from {len(all_features)} total')
        return selected_data, valid_features

    def _get_all_feature_columns(self, engineered_data: Dict[str, pd.DataFrame]) -> List[str]:
        """Get all feature columns from engineered data."""
        all_features = set()
        for data in engineered_data.values():
            if isinstance(data, pd.DataFrame):
                features = [
                    col for col in data.columns
                    if col.startswith(('feature_', 'regime_', 'sr_', 'time_', 'RSI_', 'MACD_', 'SMA_', 'EMA_', 'ATR_', 'BB_', 'Stoch_', 'ADX_', 'OBV', 'MFI_', 'poly_', 'cross_', 'pattern_', 'momentum_', 'regime_'))
                ]
                all_features.update(features)
        return sorted(list(all_features))

    def _generate_feature_reports(self, engineered_data: Dict[str, pd.DataFrame],
                                feature_statistics: Dict[str, Dict[str, Any]],
                                selected_features: List[str]) -> Dict[str, str]:
        """Generate feature engineering reports."""
        reports = {}

        # Summary report
        summary_lines = [
            'Enhanced Feature Engineering Summary',
            '=' * 50,
            f'Total features created: {len(selected_features)}',
            f'Processing time: {self.performance_metrics["total_processing_time"]:.2f}s',
            f'Memory used: {self.performance_metrics["total_memory_used_mb"]:.1f}MB',
            f'Chunks processed: {self.performance_metrics["chunks_processed"]}',
            '',
            'Data splits:'
        ]

        for split_name, data in engineered_data.items():
            if isinstance(data, pd.DataFrame):
                summary_lines.append(f'  {split_name}: {data.shape[0]} rows × {data.shape[1]} columns')

        reports['summary'] = '\n'.join(summary_lines)

        # Statistics report
        stats_lines = ['Feature Statistics', '=' * 30]
        for split_name, stats in feature_statistics.items():
            stats_lines.extend([
                '',
                f'{split_name.upper()} split:',
                f"  Total features: {stats.get('n_features', 0)}",
                f"  Features with missing values: {len(stats.get('missing_values', {}))}",
                f"  Zero variance features: {len(stats.get('zero_variance', []))}",
                f"  High correlation pairs: {len(stats.get('high_correlation_pairs', []))}"
            ])

        reports['statistics'] = '\n'.join(stats_lines)

        return reports

    async def _save_outputs(self, training_input: Dict[str, Any],
                          pipeline_state: Dict[str, Any]) -> None:
        """Save step outputs to disk."""
        output_dir = Path(training_input.get('output_dir', 'output')) / 'step06_enhanced_feature_engineering'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save engineered data
        if 'engineered_data' in pipeline_state:
            for split_name, data in pipeline_state['engineered_data'].items():
                if isinstance(data, pd.DataFrame):
                    file_path = output_dir / f'{split_name}_engineered.parquet'
                    data.to_parquet(file_path)
                    self.logger.info(f'💾 Saved {split_name} engineered data to {file_path}')

        # Save selected features
        if 'selected_features' in pipeline_state:
            features_path = output_dir / 'selected_features.json'
            with open(features_path, 'w') as f:
                json.dump(pipeline_state['selected_features'], f, indent=2)
            self.logger.info(f'💾 Saved selected features to {features_path}')

        # Save performance metrics
        metrics_path = output_dir / 'performance_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.performance_metrics, f, indent=2)
        self.logger.info(f'💾 Saved performance metrics to {metrics_path}')

    def validate_outputs(self, pipeline_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate step outputs."""
        errors = []

        if 'engineered_data' not in pipeline_state:
            errors.append('No engineered data in pipeline state')
            return False, errors

        engineered_data = pipeline_state['engineered_data']
        total_features = 0

        for split_name, data in engineered_data.items():
            if isinstance(data, pd.DataFrame):
                feature_cols = [col for col in data.columns if col.startswith(('feature_', 'regime_', 'sr_', 'time_'))]
                total_features += len(feature_cols)

        if total_features == 0:
            errors.append('No features were engineered')

        if 'selected_features' in pipeline_state:
            selected = pipeline_state['selected_features']
            if len(selected) == 0:
                errors.append('No features were selected')

        return len(errors) == 0, errors

    def get_required_inputs(self) -> List[str]:
        """Get list of required inputs for this step."""
        return ['labeled_data or split data with labels']

    def get_produced_outputs(self) -> List[str]:
        """Get list of outputs produced by this step."""
        return [
            'engineered_data', 'feature_statistics', 'selected_features',
            'feature_reports', 'performance_metrics'
        ]

    def get_dependencies(self) -> List[str]:
        """Get list of step dependencies."""
        return ['05_labeling']

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self.performance_metrics.copy()
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and
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
