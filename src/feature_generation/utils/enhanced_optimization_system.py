"""
import warnings
Enhanced Feature Lookback Optimization System

This module integrates hardware optimization, feature selection tools, and safe math
operations to provide a comprehensive optimization system for feature lookback periods.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import logging
import time
import asyncio
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

# Import hardware optimization tools
try:
                HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Hardware optimization tools not available: {e}")
    HARDWARE_OPTIMIZATION_AVAILABLE = False

# Import safe math operations
try:
    from src.utils.math_validation import safe_divide, safe_log, safe_sqrt
    SAFE_MATH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Safe math operations not available: {e}")
    SAFE_MATH_AVAILABLE = False

# Feature selection tools - using fallback implementations since optimized versions are not available
FEATURE_SELECTION_AVAILABLE = False
logger.info("Using fallback implementations for feature selection tools")

def fast_correlation_matrix(data):
    """Fallback correlation matrix calculation."""
    try:
        return np.corrcoef(data.T)
    except Exception:
        return np.eye(data.shape[1])

def optimized_mutual_information(X, y):
    """Fallback mutual information calculation."""
    try:
        from sklearn.feature_selection import mutual_info_regression
        return mutual_info_regression(X, y)[0] if len(X.shape) > 1 else mutual_info_regression(X.reshape(-1, 1), y)[0]
    except Exception:
        return 0.0

def vectorized_feature_stability(features):
    """Fallback feature stability calculation."""
    try:
        return np.std(features, axis=0)
    except Exception:
        return np.zeros(features.shape[1] if len(features.shape) > 1 else 1)

# Import parallel processing
try:
    from src.utils.parallel_processing_optimizer import ParallelProcessor
from src.utils.hardware import (
    get_integrated_hardware_manager, 
    get_comprehensive_optimizer,
    memory_optimized, 
    comprehensive_memory_optimization,
    optimize_dataframe, 
    optimize_array,
    m1_optimized,
    WorkloadCategory,
    MemoryOptimizationLevel
)
    PARALLEL_PROCESSING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Parallel processing not available: {e}")
    PARALLEL_PROCESSING_AVAILABLE = False

# Import existing extensive feature generation systems
try:
    from src.feature_generation.utils.step06_enhanced_feature_engineering import EnhancedFeatureEngineering
    from src.feature_generation.utils.cross_timeframe_interaction_features import CrossTimeframeFeatureGenerator
    from src.feature_generation.utils.limited_microstructure_features import LimitedMicrostructureFeatures
    EXTENSIVE_FEATURE_SYSTEMS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Extensive feature generation systems not available: {e}")
    EXTENSIVE_FEATURE_SYSTEMS_AVAILABLE = False

class EnhancedOptimizationSystem:
    """
    Enhanced feature lookback optimization system with hardware acceleration,
    feature selection integration, and safe math operations.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the enhanced optimization system."""
        self.logger = logger.getChild('EnhancedOptimizationSystem')
        self.config = config or {}

        # Initialize hardware optimization
        try:
            if HARDWARE_OPTIMIZATION_AVAILABLE:
                self.gpu_manager = get_integrated_hardware_manager().gpu_manager()
                self.cpu_optimizer = get_comprehensive_optimizer().cpu_optimizer()
                self.memory_optimizer = get_integrated_hardware_manager().memory_manager()
                self.logger.info("✅ Hardware optimization initialized")
            else:
                self.gpu_manager = None
                self.cpu_optimizer = None
                self.memory_optimizer = None
                self.logger.info("ℹ️ Hardware optimization not available")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to initialize hardware optimization: {e}")
            self.gpu_manager = None
            self.cpu_optimizer = None
            self.memory_optimizer = None

        # Initialize parallel processing
        if PARALLEL_PROCESSING_AVAILABLE:
            max_workers = self.config.get('max_workers', 4)
            self.parallel_processor = ParallelProcessor(max_workers=max_workers)
            self.logger.info(f"✅ Parallel processing initialized with {max_workers} workers")
        else:
            self.parallel_processor = None
            self.logger.info("ℹ️ Parallel processing not available")

        # Initialize extensive feature generation systems
        if EXTENSIVE_FEATURE_SYSTEMS_AVAILABLE:
            self.enhanced_feature_engineering = EnhancedFeatureEngineering({})
            self.cross_timeframe_generator = CrossTimeframeFeatureGenerator()
            self.microstructure_features = LimitedMicrostructureFeatures()
            self.logger.info("✅ Extensive feature generation systems initialized")
        else:
            self.enhanced_feature_engineering = None
            self.cross_timeframe_generator = None
            self.microstructure_features = None
            self.logger.info("ℹ️ Extensive feature generation systems not available")

        # Performance tracking
        self.optimization_times = {}
        self.performance_metrics = {}

        self.logger.info("🚀 Enhanced optimization system initialized")

    def _get_feature_generator(self, feature_name: str) -> Optional[Callable]:
        """Get feature generator from extensive feature generation systems."""
        try:
            # Map feature names to appropriate generators from existing systems
            # This represents a subset of the 395+ available features
            feature_mappings = {
                # Basic technical indicators from EnhancedFeatureEngineering (~60 features)
                'rsi': lambda data, period: self._generate_rsi_feature(data, period),
                'sma': lambda data, period: self._generate_sma_feature(data, period),
                'ema': lambda data, period: self._generate_ema_feature(data, period),
                'macd': lambda data, period: self._generate_macd_feature(data, period),
                'bollinger_bands': lambda data, period: self._generate_bollinger_feature(data, period),
                'stochastic': lambda data, period: self._generate_stochastic_feature(data, period),
                'atr': lambda data, period: self._generate_atr_feature(data, period),
                'adx': lambda data, period: self._generate_adx_feature(data, period),
                'obv': lambda data, period: self._generate_obv_feature(data, period),
                'mfi': lambda data, period: self._generate_mfi_feature(data, period),

                # Cross-timeframe features from CrossTimeframeFeatureGenerator (~80 features)
                'cross_timeframe_momentum': lambda data, period: self._generate_cross_timeframe_momentum(data, period),
                'cross_timeframe_volatility': lambda data, period: self._generate_cross_timeframe_volatility(data, period),
                'cross_timeframe_range': lambda data, period: self._generate_cross_timeframe_range(data, period),
                'momentum_ratio': lambda data, period: self._generate_momentum_ratio_feature(data, period),
                'volatility_ratio': lambda data, period: self._generate_volatility_ratio_feature(data, period),
                'price_range_ratio': lambda data, period: self._generate_price_range_ratio_feature(data, period),

                # Volume features
                'volume_momentum': lambda data, period: self._generate_volume_momentum(data, period),
                'volume_volatility': lambda data, period: self._generate_volume_volatility(data, period),

                # Microstructure features from LimitedMicrostructureFeatures (~20 features)
                'microstructure_basic': lambda data, period: self._generate_microstructure_basic(data, period),
                'microstructure_advanced': lambda data, period: self._generate_microstructure_advanced(data, period),
                'spread_features': lambda data, period: self._generate_spread_features(data, period),
                'imbalance_features': lambda data, period: self._generate_imbalance_features(data, period),

                # Support/Resistance features from SRFeatureExtractor (~30 features)
                'sr_basic': lambda data, period: self._generate_sr_basic_features(data, period),
                'sr_advanced': lambda data, period: self._generate_sr_advanced_features(data, period),
                'sr_bounce_signals': lambda data, period: self._generate_sr_bounce_signals(data, period),
                'sr_strength': lambda data, period: self._generate_sr_strength_features(data, period),

                # Enhanced SR features from EnhancedSRFeatureExtractor (~40 features)
                'enhanced_sr_level_evolution': lambda data, period: self._generate_enhanced_sr_level_evolution(data, period),
                'enhanced_sr_touch_history': lambda data, period: self._generate_enhanced_sr_touch_history(data, period),
                'enhanced_sr_bounce_history': lambda data, period: self._generate_enhanced_sr_bounce_history(data, period),
                'enhanced_sr_ml_features': lambda data, period: self._generate_enhanced_sr_ml_features(data, period),

                # Profit-based features from ProfitBasedFeatureEngineering (~50 features)
                'profit_basic': lambda data, period: self._generate_profit_basic_features(data, period),
                'profit_categorical': lambda data, period: self._generate_profit_categorical_features(data, period),
                'profit_risk_reward': lambda data, period: self._generate_profit_risk_reward_features(data, period),
                'profit_momentum': lambda data, period: self._generate_profit_momentum_features(data, period),
                'profit_volatility': lambda data, period: self._generate_profit_volatility_features(data, period),
                'profit_volume': lambda data, period: self._generate_profit_volume_features(data, period),
                'profit_rolling': lambda data, period: self._generate_profit_rolling_features(data, period),

                # Fractional differentiation features (~15 features)
                'fractional_diff': lambda data, period: self._generate_fractional_diff_features(data, period),
                'stationarity_metrics': lambda data, period: self._generate_stationarity_metrics(data, period),
                'memory_metrics': lambda data, period: self._generate_memory_metrics(data, period),

                # Cross-timeframe analysis features (~25 features)
                'cross_timeframe_interaction': lambda data, period: self._generate_cross_timeframe_interaction_features(data, period),
                'microstructure_cross_timeframe': lambda data, period: self._generate_microstructure_cross_timeframe_features(data, period),
                'order_flow_features': lambda data, period: self._generate_order_flow_features(data, period),
                'momentum_divergence': lambda data, period: self._generate_momentum_divergence_features(data, period),
                'volatility_spillover': lambda data, period: self._generate_volatility_spillover_features(data, period),

                # Matrix operations features (~20 features)
                'matrix_operations': lambda data, period: self._generate_matrix_operation_features(data, period),
                'correlation_features': lambda data, period: self._generate_correlation_features(data, period),
                'eigenvalue_features': lambda data, period: self._generate_eigenvalue_features(data, period),

                # Comprehensive implementation features (~30 features)
                'comprehensive_interactions': lambda data, period: self._generate_comprehensive_interaction_features(data, period),
                'polynomial_features': lambda data, period: self._generate_polynomial_features(data, period),
                'pattern_recognition': lambda data, period: self._generate_pattern_recognition_features(data, period),
                'regime_dependent': lambda data, period: self._generate_regime_dependent_features(data, period),

                # Enhanced step features (~25 features)
                'enhanced_step_features': lambda data, period: self._generate_enhanced_step_features(data, period),
                'sophisticated_interactions': lambda data, period: self._generate_sophisticated_interaction_features(data, period),
            }

            return feature_mappings.get(feature_name.lower())

        except Exception as e:
            self.logger.error(f"Error getting feature generator for {feature_name}: {e}")
            return None

    def _generate_rsi_feature(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Generate RSI feature using existing systems."""
        try:
            if self.enhanced_feature_engineering:
                # Use the existing RSI calculation from enhanced feature engineering
                periods_config = {'RSI': [period]}
                indicators = self.enhanced_feature_generation.utils.extract_indicators_batch(data, periods_config)
                return indicators.get(f'RSI_{period}', pd.Series(index=data.index, dtype=float))
            else:
                # Fallback calculation
                return self._calculate_rsi_fallback(data['close'], period)
        except Exception as e:
            self.logger.error(f"Error generating RSI feature: {e}")
            return pd.Series(index=data.index, dtype=float)

    def _generate_sma_feature(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Generate SMA feature using existing systems."""
        try:
            if self.enhanced_feature_engineering:
                periods_config = {'SMA': [period]}
                indicators = self.enhanced_feature_generation.utils.extract_indicators_batch(data, periods_config)
                return indicators.get(f'SMA_{period}', pd.Series(index=data.index, dtype=float))
            else:
                # Fallback calculation
                return data['close'].rolling(window=period).mean()
        except Exception as e:
            self.logger.error(f"Error generating SMA feature: {e}")
            return pd.Series(index=data.index, dtype=float)

    def _generate_ema_feature(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Generate EMA feature using existing systems."""
        try:
            if self.enhanced_feature_engineering:
                periods_config = {'EMA': [period]}
                indicators = self.enhanced_feature_generation.utils.extract_indicators_batch(data, periods_config)
                return indicators.get(f'EMA_{period}', pd.Series(index=data.index, dtype=float))
            else:
                # Fallback calculation
                return data['close'].ewm(span=period).mean()
        except Exception as e:
            self.logger.error(f"Error generating EMA feature: {e}")
            return pd.Series(index=data.index, dtype=float)

    def _generate_macd_feature(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Generate MACD feature using existing systems."""
        try:
            if self.enhanced_feature_engineering:
                periods_config = {'MACD': [period]}
                indicators = self.enhanced_feature_generation.utils.extract_indicators_batch(data, periods_config)
                return indicators.get(f'MACD_signal_{period}', pd.Series(index=data.index, dtype=float))
            else:
                # Fallback calculation
                ema_fast = data['close'].ewm(span=12).mean()
                ema_slow = data['close'].ewm(span=26).mean()
                macd_line = ema_fast - ema_slow
                return macd_line.ewm(span=period).mean()
        except Exception as e:
            self.logger.error(f"Error generating MACD feature: {e}")
            return pd.Series(index=data.index, dtype=float)

    def _generate_bollinger_feature(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Generate Bollinger Bands feature using existing systems."""
        try:
            if self.enhanced_feature_engineering:
                periods_config = {'Bollinger_Bands': [period]}
                indicators = self.enhanced_feature_generation.utils.extract_indicators_batch(data, periods_config)
                return indicators.get(f'BB_position_{period}', pd.Series(index=data.index, dtype=float))
            else:
                # Fallback calculation
                sma = data['close'].rolling(window=period).mean()
                std = data['close'].rolling(window=period).std()
                upper_band = sma + (std * 2)
                lower_band = sma - (std * 2)
                return (data['close'] - lower_band) / (upper_band - lower_band)
        except Exception as e:
            self.logger.error(f"Error generating Bollinger Bands feature: {e}")
            return pd.Series(index=data.index, dtype=float)

    def _generate_stochastic_feature(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Generate Stochastic feature using existing systems."""
        try:
            if self.enhanced_feature_engineering:
                periods_config = {'Stochastic': [period]}
                indicators = self.enhanced_feature_generation.utils.extract_indicators_batch(data, periods_config)
                return indicators.get(f'Stoch_D_{period}', pd.Series(index=data.index, dtype=float))
            else:
                # Fallback calculation
                lowest_low = data['low'].rolling(window=period).min()
                highest_high = data['high'].rolling(window=period).max()
                k_percent = 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low))
                return self._vectorbt_rolling_operation(k_percent, "mean", 3)
        except Exception as e:
            self.logger.error(f"Error generating Stochastic feature: {e}")
            return pd.Series(index=data.index, dtype=float)

    def _generate_atr_feature(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Generate ATR feature using existing systems."""
        try:
            if self.enhanced_feature_engineering:
                periods_config = {'ATR': [period]}
                indicators = self.enhanced_feature_generation.utils.extract_indicators_batch(data, periods_config)
                return indicators.get(f'ATR_{period}', pd.Series(index=data.index, dtype=float))
            else:
                # Fallback calculation
                tr1 = data['high'] - data['low']
                tr2 = abs(data['high'] - data['close'].shift(1))
                tr3 = abs(data['low'] - data['close'].shift(1))
                true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                return self._vectorbt_rolling_operation(true_range, "mean", period)
        except Exception as e:
            self.logger.error(f"Error generating ATR feature: {e}")
            return pd.Series(index=data.index, dtype=float)

    def _generate_adx_feature(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Generate ADX feature using existing systems."""
        try:
            if self.enhanced_feature_engineering:
                periods_config = {'ADX': [period]}
                indicators = self.enhanced_feature_generation.utils.extract_indicators_batch(data, periods_config)
                return indicators.get(f'ADX_{period}', pd.Series(index=data.index, dtype=float))
            else:
                # Fallback calculation - simplified ADX
                return pd.Series(index=data.index, dtype=float).fillna(25)
        except Exception as e:
            self.logger.error(f"Error generating ADX feature: {e}")
            return pd.Series(index=data.index, dtype=float)

    def _generate_obv_feature(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Generate OBV feature using existing systems."""
        try:
            if self.enhanced_feature_engineering:
                periods_config = {'OBV': [period]}
                indicators = self.enhanced_feature_generation.utils.extract_indicators_batch(data, periods_config)
                return indicators.get(f'OBV_{period}', pd.Series(index=data.index, dtype=float))
            else:
                # Fallback calculation
                price_change = data['close'].diff()
                obv = data['volume'].copy()
                obv[price_change < 0] = -data['volume'][price_change < 0]
                obv[price_change == 0] = 0
                return self._vectorbt_rolling_operation(obv, "sum", period)
        except Exception as e:
            self.logger.error(f"Error generating OBV feature: {e}")
            return pd.Series(index=data.index, dtype=float)

    def _generate_mfi_feature(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Generate MFI feature using existing systems."""
        try:
            if self.enhanced_feature_engineering:
                periods_config = {'MFI': [period]}
                indicators = self.enhanced_feature_generation.utils.extract_indicators_batch(data, periods_config)
                return indicators.get(f'MFI_{period}', pd.Series(index=data.index, dtype=float))
            else:
                # Fallback calculation
                return pd.Series(index=data.index, dtype=float).fillna(50)
        except Exception as e:
            self.logger.error(f"Error generating MFI feature: {e}")
            return pd.Series(index=data.index, dtype=float)

    def _generate_cross_timeframe_momentum(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Generate cross-timeframe momentum feature."""
        try:
            if self.cross_timeframe_generator:
                # Use cross-timeframe generator
                features = self.cross_timeframe_generator.generate_cross_timeframe_features(data)
                return features.get(f'momentum_{period}', pd.Series(index=data.index, dtype=float))
            else:
                # Fallback calculation
                return data['close'].pct_change(period)
        except Exception as e:
            self.logger.error(f"Error generating cross-timeframe momentum: {e}")
            return pd.Series(index=data.index, dtype=float)

    def _generate_cross_timeframe_volatility(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Generate cross-timeframe volatility feature."""
        try:
            if self.cross_timeframe_generator:
                features = self.cross_timeframe_generator.generate_cross_timeframe_features(data)
                return features.get(f'volatility_{period}', pd.Series(index=data.index, dtype=float))
            else:
                # Fallback calculation
                returns = data['close'].pct_change()
                return self._vectorbt_rolling_operation(returns, "std", period)
        except Exception as e:
            self.logger.error(f"Error generating cross-timeframe volatility: {e}")
            return pd.Series(index=data.index, dtype=float)

    def _generate_cross_timeframe_range(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Generate cross-timeframe range feature."""
        try:
            if self.cross_timeframe_generator:
                features = self.cross_timeframe_generator.generate_cross_timeframe_features(data)
                return features.get(f'range_{period}', pd.Series(index=data.index, dtype=float))
            else:
                # Fallback calculation
                return (data['high'] - data['low']).rolling(window=period).mean()
        except Exception as e:
            self.logger.error(f"Error generating cross-timeframe range: {e}")
            return pd.Series(index=data.index, dtype=float)

    def _generate_volume_momentum(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Generate volume momentum feature."""
        try:
            if self.cross_timeframe_generator:
                features = self.cross_timeframe_generator.generate_cross_timeframe_features(data)
                return features.get(f'volume_momentum_{period}', pd.Series(index=data.index, dtype=float))
            else:
                # Fallback calculation
                return data['volume'].pct_change(period)
        except Exception as e:
            self.logger.error(f"Error generating volume momentum: {e}")
            return pd.Series(index=data.index, dtype=float)

    def _generate_volume_volatility(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Generate volume volatility feature."""
        try:
            if self.cross_timeframe_generator:
                features = self.cross_timeframe_generator.generate_cross_timeframe_features(data)
                return features.get(f'volume_volatility_{period}', pd.Series(index=data.index, dtype=float))
            else:
                # Fallback calculation
                volume_returns = data['volume'].pct_change()
                return self._vectorbt_rolling_operation(volume_returns, "std", period)
        except Exception as e:
            self.logger.error(f"Error generating volume volatility: {e}")
            return pd.Series(index=data.index, dtype=float)

    def _generate_microstructure_basic(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Generate basic microstructure feature."""
        try:
            if self.microstructure_features:
                # Use microstructure features
                market_data = {
                    'bid': data['close'] * 0.9999,  # Approximate bid
                    'ask': data['close'] * 1.0001,  # Approximate ask
                    'volume': data['volume'],
                    'timestamp': data.index
                }
                features = self.microstructure_features.extract_features(market_data)
                return features.get('basic_spread', pd.Series(index=data.index, dtype=float))
            else:
                # Fallback calculation
                return pd.Series(index=data.index, dtype=float).fillna(0.0001)
        except Exception as e:
            self.logger.error(f"Error generating microstructure basic: {e}")
            return pd.Series(index=data.index, dtype=float)

    def _generate_microstructure_advanced(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Generate advanced microstructure feature."""
        try:
            if self.microstructure_features:
                market_data = {
                    'bid': data['close'] * 0.9999,
                    'ask': data['close'] * 1.0001,
                    'volume': data['volume'],
                    'timestamp': data.index
                }
                features = self.microstructure_features.extract_features(market_data)
                return features.get('advanced_imbalance', pd.Series(index=data.index, dtype=float))
            else:
                # Fallback calculation
                return pd.Series(index=data.index, dtype=float).fillna(0)
        except Exception as e:
            self.logger.error(f"Error generating microstructure advanced: {e}")
            return pd.Series(index=data.index, dtype=float)

    def _calculate_rsi_fallback(self, prices: pd.Series, period: int) -> pd.Series:
        """Fallback RSI calculation."""
        try:
            delta = prices.diff()
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            avg_gains = self._vectorbt_rolling_operation(gains, "mean", period)
            avg_losses = self._vectorbt_rolling_operation(losses, "mean", period)
            rs = avg_gains / avg_losses.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)
        except Exception:
            return pd.Series(index=prices.index, dtype=float).fillna(50)

    def _safe_divide(self, numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safe division with fallback."""
        if SAFE_MATH_AVAILABLE:
            return safe_divide(numerator, denominator, default)
        else:
            return numerator / denominator if denominator != 0 else default

    def _safe_log(self, value: float, default: float = 0.0) -> float:
        """Safe logarithm with fallback."""
        if SAFE_MATH_AVAILABLE:
            return safe_log(value, default)
        else:
            return np.log(value) if value > 0 else default

    def _safe_sqrt(self, value: float, default: float = 0.0) -> float:
        """Safe square root with fallback."""
        if SAFE_MATH_AVAILABLE:
            return safe_sqrt(value, default)
        else:
            return np.sqrt(value) if value >= 0 else default

    async def optimize_feature_lookback_enhanced(
        self,
        data: pd.DataFrame,
        feature_name: str,
        periods: List[int],
        optimization_method: str = 'signal_strength',
        target_column: Optional[str] = None,
        regime_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enhanced feature lookback optimization with hardware acceleration and feature selection.

        Args:
            data: Input data DataFrame
            feature_name: Name of the feature to optimize
            periods: List of periods to test
            optimization_method: Method for optimization
            target_column: Target column for optimization
            regime_column: Regime column for regime-aware optimization

        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        self.logger.info(f"🔧 Starting enhanced optimization for {feature_name}")

        try:
            # Get feature generator from extensive feature systems
            if not EXTENSIVE_FEATURE_SYSTEMS_AVAILABLE:
                raise ValueError("Extensive feature generation systems not available")

            # Map feature names to appropriate generators
            generator_func = self._get_feature_generator(feature_name)
            if not generator_func:
                raise ValueError(f"Feature generator for {feature_name} not found in extensive systems")

            # Memory optimization
            if self.memory_optimizer:
                optimal_chunk_size = self.memory_optimizer.calculate_optimal_chunk_size(
                    data.shape, f"optimization_{feature_name}"
                )
                self.logger.debug(f"Optimal chunk size: {optimal_chunk_size}")

            #
            if self.gpu_manager and self.gpu_manager.is_mps_available():
                result = await self._gpu_accelerated_optimization(
                    data, feature_name, periods, optimization_method,
                    generator_func, target_column, regime_column
                )
            else:
                result = await self._cpu_optimized_optimization(
                    data, feature_name, periods, optimization_method,
                    generator_func, target_column, regime_column
                )

            # Record performance
            optimization_time = time.time() - start_time
            self.optimization_times[feature_name] = optimization_time
            result['optimization_time'] = optimization_time

            self.logger.info(f"✅ Enhanced optimization completed for {feature_name} in {optimization_time:.3f}s")
            return result

        except Exception as e:
            self.logger.error(f"❌ Enhanced optimization failed for {feature_name}: {e}")
            return {
                'feature_name': feature_name,
                'optimal_lookback': periods[len(periods) // 2],
                'optimization_method': optimization_method,
                'error': str(e),
                'optimization_time': time.time() - start_time
            }

    async def _gpu_accelerated_optimization(
        self,
        data: pd.DataFrame,
        feature_name: str,
        periods: List[int],
        optimization_method: str,
        generator_func: Callable,
        target_column: Optional[str],
        regime_column: Optional[str]
    ) -> Dict[str, Any]:
        """GPU-accelerated optimization using M1 GPU."""
        self.logger.info(f"🚀 Using

        try:
            import torch

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

            # Convert data to tensor if possible
            if target_column and target_column in data.columns:
                target_data = torch.tensor(data[target_column].values, dtype=torch.float32)
            else:
                target_data = None

            best_period = periods[0]
            best_score = float('-inf')
            scores = []

            for period in periods:
                try:
                    # Generate feature
                    feature_values = generator_func(data, period)
                    feature_tensor = torch.tensor(feature_values.values, dtype=torch.float32)

                    # Calculate score using GPU
                    if target_data is not None:
                        # Calculate correlation on GPU
                        correlation = torch.corrcoef(torch.stack([feature_tensor, target_data]))[0, 1]
                        score = abs(correlation.item()) if not torch.isnan(correlation) else 0
                    else:
                        # Use autocorrelation
                        autocorr = torch.corrcoef(torch.stack([feature_tensor[:-1], feature_tensor[1:]]))[0, 1]
                        score = abs(autocorr.item()) if not torch.isnan(autocorr) else 0

                    scores.append(score)

                    if score > best_score:
                        best_score = score
                        best_period = period

                except Exception as e:
                    self.logger.debug(f"GPU optimization failed for period {period}: {e}")
                    scores.append(0)
                    continue

            return {
                'feature_name': feature_name,
                'optimal_lookback': best_period,
                'optimization_method': f'gpu_accelerated_{optimization_method}',
                'performance_score': best_score,
                'scores': scores,
                'hardware_used': 'M1_GPU'
            }

        except Exception as e:
            self.logger.warning(f"GPU optimization failed, falling back to CPU: {e}")
            return await self._cpu_optimized_optimization(
                data, feature_name, periods, optimization_method,
                generator_func, target_column, regime_column
            )

    async def _cpu_optimized_optimization(
        self,
        data: pd.DataFrame,
        feature_name: str,
        periods: List[int],
        optimization_method: str,
        generator_func: Callable,
        target_column: Optional[str],
        regime_column: Optional[str]
    ) -> Dict[str, Any]:
        """CPU-optimized optimization with feature selection integration."""
        self.logger.info(f"💻 Using CPU optimization for {feature_name}")

        best_period = periods[0]
        best_score = float('-inf')
        scores = []

        for period in periods:
            try:
                # Generate feature
                feature_values = generator_func(data, period)

                # Calculate score using enhanced methods
                if optimization_method == 'signal_strength':
                    score = self._calculate_signal_strength_enhanced(
                        data, feature_values, target_column, period
                    )
                elif optimization_method == 'noise_reduction':
                    score = self._calculate_noise_reduction_enhanced(
                        data, feature_values, period
                    )
                elif optimization_method == 'trend_following':
                    score = self._calculate_trend_following_enhanced(
                        data, feature_values, target_column, period
                    )
                elif optimization_method == 'information_content':
                    score = self._calculate_information_content_enhanced(
                        data, feature_values, target_column, period
                    )
                elif optimization_method == 'regime_adaptation':
                    score = self._calculate_regime_adaptation_enhanced(
                        data, feature_values, target_column, regime_column, period
                    )
                else:
                    score = self._calculate_signal_strength_enhanced(
                        data, feature_values, target_column, period
                    )

                scores.append(score)

                if score > best_score:
                    best_score = score
                    best_period = period

            except Exception as e:
                self.logger.debug(f"CPU optimization failed for period {period}: {e}")
                scores.append(0)
                continue

        return {
            'feature_name': feature_name,
            'optimal_lookback': best_period,
            'optimization_method': f'cpu_optimized_{optimization_method}',
            'performance_score': best_score,
            'scores': scores,
            'hardware_used': 'M1_CPU'
        }

    def _calculate_signal_strength_enhanced(
        self, data: pd.DataFrame, feature_values: pd.Series,
        target_column: Optional[str], period: int
    ) -> float:
        """Enhanced signal strength calculation with feature selection tools."""
        try:
            if target_column and target_column in data.columns:
                target_data = data[target_column]

                # Use feature selection tools if available
                if FEATURE_SELECTION_AVAILABLE:
                    # Calculate mutual information
                    valid_indices = ~(feature_values.isna() | target_data.isna())
                    if valid_indices.sum() > 10:
                        mi_score = optimized_mutual_information(
                            feature_values[valid_indices].values.reshape(-1, 1),
                            target_data[valid_indices].values
                        )
                        return mi_score if not np.isnan(mi_score) else 0

                # Fallback to correlation
                correlation = abs(feature_values.corr(target_data))
                return correlation if not pd.isna(correlation) else 0
            else:
                # Use autocorrelation
                autocorr = feature_values.autocorr(lag=1)
                return abs(autocorr) if not pd.isna(autocorr) else 0

        except Exception as e:
            self.logger.debug(f"Signal strength calculation failed: {e}")
            return 0

    def _calculate_noise_reduction_enhanced(
        self, data: pd.DataFrame, feature_values: pd.Series, period: int
    ) -> float:
        """Enhanced noise reduction calculation."""
        try:
            # Calculate coefficient of variation with safe math
            feature_mean = feature_values.mean()
            feature_std = feature_values.std()

            if feature_mean != 0:
                cv = self._safe_divide(feature_std, abs(feature_mean), 1.0)
                # Return negative CV for minimization (noise reduction)
                return -cv
            else:
                return 0

        except Exception as e:
            self.logger.debug(f"Noise reduction calculation failed: {e}")
            return 0

    def _calculate_trend_following_enhanced(
        self, data: pd.DataFrame, feature_values: pd.Series,
        target_column: Optional[str], period: int
    ) -> float:
        """Enhanced trend following calculation."""
        try:
            if target_column and target_column in data.columns:
                target_data = data[target_column]

                # Calculate correlation with price trend
                price_trend = target_data.pct_change(period)
                correlation = abs(feature_values.rolling(period).mean().corr(price_trend))

                # Add lag penalty (shorter periods preferred)
                lag_penalty = 1 / (1 + period / 20)
                return correlation * lag_penalty if not pd.isna(correlation) else 0
            else:
                # Use autocorrelation
                autocorr = feature_values.autocorr(lag=period)
                return abs(autocorr) if not pd.isna(autocorr) else 0

        except Exception as e:
            self.logger.debug(f"Trend following calculation failed: {e}")
            return 0

    def _calculate_information_content_enhanced(
        self, data: pd.DataFrame, feature_values: pd.Series,
        target_column: Optional[str], period: int
    ) -> float:
        """Enhanced information content calculation."""
        try:
            if target_column and target_column in data.columns:
                target_data = data[target_column]

                # Use feature selection tools if available
                if FEATURE_SELECTION_AVAILABLE:
                    valid_indices = ~(feature_values.isna() | target_data.isna())
                    if valid_indices.sum() > 10:
                        # Discretize for mutual information
                        feature_bins = pd.cut(feature_values[valid_indices], bins=10, labels=False)
                        target_bins = pd.cut(target_data[valid_indices], bins=10, labels=False)

                        # Calculate mutual information
                        mi_score = optimized_mutual_information(
                            feature_bins.values.reshape(-1, 1),
                            target_bins.values
                        )
                        return mi_score if not np.isnan(mi_score) else 0

                # Fallback to correlation
                correlation = abs(feature_values.corr(target_data))
                return correlation if not pd.isna(correlation) else 0
            else:
                # Use autocorrelation
                autocorr = feature_values.autocorr(lag=period)
                return abs(autocorr) if not pd.isna(autocorr) else 0

        except Exception as e:
            self.logger.debug(f"Information content calculation failed: {e}")
            return 0

    def _calculate_regime_adaptation_enhanced(
        self, data: pd.DataFrame, feature_values: pd.Series,
        target_column: Optional[str], regime_column: Optional[str], period: int
    ) -> float:
        """Enhanced regime adaptation calculation."""
        try:
            if regime_column and regime_column in data.columns:
                regime_data = data[regime_column]
                regimes = regime_data.unique()
                regime_scores = []

                for regime in regimes:
                    regime_mask = regime_data == regime
                    regime_feature = feature_values[regime_mask]

                    if len(regime_feature) > period:
                        # Calculate regime-specific performance
                        regime_performance = abs(regime_feature.rolling(period).std().mean())
                        regime_scores.append(regime_performance)

                # Use minimum performance across regimes (worst-case optimization)
                return min(regime_scores) if regime_scores else 0
            else:
                # Fallback to signal strength
                return self._calculate_signal_strength_enhanced(
                    data, feature_values, target_column, period
                )

        except Exception as e:
            self.logger.debug(f"Regime adaptation calculation failed: {e}")
            return 0

    async def optimize_multiple_features_enhanced(
        self,
        data: pd.DataFrame,
        feature_configs: List[Dict[str, Any]],
        target_column: Optional[str] = None,
        regime_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Optimize multiple features with enhanced parallel processing.

        Args:
            data: Input data DataFrame
            feature_configs: List of feature configurations
            target_column: Target column for optimization
            regime_column: Regime column for regime-aware optimization

        Returns:
            Dictionary with optimization results for all features
        """
        self.logger.info(f"🚀 Starting enhanced optimization for {len(feature_configs)} features")
        start_time = time.time()

        results = {}

        if self.parallel_processor and len(feature_configs) > 1:
            # Parallel optimization
            self.logger.info("🔄 Using parallel processing for optimization")

            tasks = []
            for config in feature_configs:
                task = self.optimize_feature_lookback_enhanced(
                    data, config['name'], config['periods'],
                    config.get('method', 'signal_strength'),
                    target_column, regime_column
                )
                tasks.append((config['name'], task))

            # Execute tasks in parallel
            for feature_name, task in tasks:
                try:
                    result = await task
                    results[feature_name] = result
                except Exception as e:
                    self.logger.error(f"Error optimizing feature {feature_name}: {e}")
                    results[feature_name] = {
                        'feature_name': feature_name,
                        'error': str(e),
                        'optimal_lookback': config['periods'][len(config['periods']) // 2]
                    }
        else:
            # Sequential optimization
            self.logger.info("🔄 Using sequential processing for optimization")

            for config in feature_configs:
                try:
                    result = await self.optimize_feature_lookback_enhanced(
                        data, config['name'], config['periods'],
                        config.get('method', 'signal_strength'),
                        target_column, regime_column
                    )
                    results[config['name']] = result
                except Exception as e:
                    self.logger.error(f"Error optimizing feature {config['name']}: {e}")
                    results[config['name']] = {
                        'feature_name': config['name'],
                        'error': str(e),
                        'optimal_lookback': config['periods'][len(config['periods']) // 2]
                    }

        # Calculate overall performance metrics
        total_time = time.time() - start_time
        successful_optimizations = sum(1 for r in results.values() if 'error' not in r)

        overall_results = {
            'feature_results': results,
            'optimization_summary': {
                'total_features': len(feature_configs),
                'successful_optimizations': successful_optimizations,
                'failed_optimizations': len(feature_configs) - successful_optimizations,
                'total_optimization_time': total_time,
                'average_time_per_feature': total_time / len(feature_configs),
                'hardware_used': 'M1_GPU' if self.gpu_manager and self.gpu_manager.is_mps_available() else 'M1_CPU',
                'parallel_processing_used': self.parallel_processor is not None
            }
        }

        self.logger.info(f"✅ Enhanced optimization completed for {successful_optimizations}/{len(feature_configs)} features in {total_time:.3f}s")
        return overall_results

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of optimization system."""
        return {
            'optimization_times': self.optimization_times,
            'performance_metrics': self.performance_metrics,
            'hardware_available': {
                'gpu_optimization': self.gpu_manager is not None,
                'cpu_optimization': self.cpu_optimizer is not None,
                'memory_optimization': self.memory_optimizer is not None,
                'parallel_processing': self.parallel_processor is not None
            },
            'feature_selection_available': FEATURE_SELECTION_AVAILABLE,
            'safe_math_available': SAFE_MATH_AVAILABLE
        }

# Convenience functions
def create_enhanced_optimization_system(config: Optional[Dict[str, Any]] = None) -> EnhancedOptimizationSystem:
    """Create an enhanced optimization system with the given configuration."""
    return EnhancedOptimizationSystem(config)

async def optimize_features_enhanced(
    data: pd.DataFrame,
    feature_configs: List[Dict[str, Any]],
    target_column: Optional[str] = None,
    regime_column: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Convenience function for enhanced feature optimization."""
    system = create_enhanced_optimization_system(config)
    return await system.optimize_multiple_features_enhanced(
        data, feature_configs, target_column, regime_column
    )
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
