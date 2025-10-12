"""
Microstructure Feature Generator

This module provides feature generators for microstructure-based indicators,
including bid-ask spread, order flow, and other high-frequency features.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union

# Optimization utilities
try:
    from ..utils.vectorization_optimizer import get_vectorization_optimizer
    from ..utils.optimized_feature_pipeline import get_optimized_feature_pipeline
    from ..utils.unified_vectorization_manager import get_unified_vectorization_manager, UnifiedVectorizationManager
    from ..utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    get_unified_vectorization_manager = None
    UnifiedVectorizationManager = None
    get_vectorbt_rolling_optimizer = None

from ..core.feature_generator import (
    FeatureGenerator,
    FeatureConfig,
    FeatureCategory,
    VectorizedFeatureGenerator
)
from ..base_calculations import (
    BaseCalculationType,
    create_base_calculator
)
from ...utils.math_validation import validate_finite

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

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

class MicrostructureFeatureGenerator(VectorizedFeatureGenerator):
    """Feature generator for microstructure-based features."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        # Initialize unified vectorization manager
        self.vectorization_manager = get_unified_vectorization_manager() if OPTIMIZATION_AVAILABLE else None
        self.rolling_optimizer = get_vectorbt_rolling_optimizer() if OPTIMIZATION_AVAILABLE else None
        
        # Initialize Unified Vectorization Manager
        try:
            from ...utils.ml_common.unified_vectorization_manager import (
                get_unified_vectorization_manager, UnifiedVectorizationManager, 
                OperationType, OptimizationStrategy, OperationConfig
            )
            self.unified_manager = get_unified_vectorization_manager()
            self.UNIFIED_MANAGER_AVAILABLE = True
        except ImportError:
            self.unified_manager = None
            self.UNIFIED_MANAGER_AVAILABLE = False
        
        # Performance tracking
        self.performance_stats = {
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'unified_manager_operations': 0,
            'batch_operations': 0,
            'total_operations': 0,
            'total_time': 0.0
        }
    
    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="microstructure_features",
            category=FeatureCategory.MICROSTRUCTURE,
            description="Comprehensive microstructure features including bid-ask spread, order flow, and trade intensity",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume", "bid", "ask"],
            default_lookback=20,
            min_lookback=1,
            max_lookback=100,
            parameters={
                "spread_windows": [5, 10, 20],
                "order_flow_windows": [5, 10, 20],
                "trade_intensity_windows": [5, 10, 20]
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
    
    @classmethod
    def create_default(cls) -> 'MicrostructureFeatureGenerator':
        return cls()
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close_prices = data['close'].values
        ms = np.zeros_like(close_prices)
        return pd.Series(ms, index=data.index, name='ms_placeholder')

# Bid-Ask Spread Generator
    
    def generate_optimized_microstructure_features(self, data: pd.DataFrame, 
                                                 feature_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Generate multiple microstructure features using optimized batch processing.
        
        Args:
            data: OHLCV data
            feature_configs: List of feature configuration dictionaries
            
        Returns:
            DataFrame with generated microstructure features
        """
        if hasattr(self, 'unified_manager') and self.unified_manager and len(data) > 100:
            try:
                # Use Unified Vectorization Manager for batch processing
                batch_result = self.unified_manager.optimize_operation(
                    OperationType.FEATURE_ENGINEERING,
                    {
                        'data': data,
                        'feature_configs': feature_configs,
                        'operation_type': 'microstructure_batch'
                    },
                    OperationConfig(
                        operation_type=OperationType.FEATURE_ENGINEERING,
                        data_size=len(data),
                        data_dimensions=data.shape,
                        memory_budget_mb=1024.0
                    )
                )
                return batch_result.result
            except Exception as e:
                self.logger.warning(f"Unified Vectorization Manager batch processing failed: {e}, using fallback")
                # Fallback to individual processing
                return self._process_microstructure_features_individually(data, feature_configs)
        else:
            return self._process_microstructure_features_individually(data, feature_configs)
    
    def _process_microstructure_features_individually(self, data: pd.DataFrame, feature_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """Process microstructure features individually as fallback when batch processing fails."""
        results = {}
        for config in feature_configs:
            feature_name = config['name']
            feature_type = config.get('type', 'microstructure')
            params = config.get('params', {})
            
            try:
                if feature_type == 'bid_ask_spread':
                    window = params.get('window', 20)
                    if 'bid' in data.columns and 'ask' in data.columns:
                        spread = data['ask'] - data['bid']
                        if hasattr(self, 'rolling_optimizer') and self.rolling_optimizer:
                            results[feature_name] = self.rolling_optimizer.rolling_std(spread, window)
                        else:
                            results[feature_name] = spread.rolling(window).std()
                    else:
                        # Fallback to high-low spread
                        spread = data['high'] - data['low']
                        if hasattr(self, 'rolling_optimizer') and self.rolling_optimizer:
                            results[feature_name] = self.rolling_optimizer.rolling_std(spread, window)
                        else:
                            results[feature_name] = spread.rolling(window).std()
                
            except Exception as e:
                self.logger.warning(f"Microstructure feature {feature_name} failed: {e}")
                results[feature_name] = pd.Series(np.nan, index=data.index)
        
        return pd.DataFrame(results, index=data.index)

class BidAskSpreadGenerator(VectorizedFeatureGenerator):
    """Generator for bid-ask spread features."""
    
    def __init__(self, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        if 'bid' not in required_columns:
            required_columns.append('bid')
        if 'ask' not in required_columns:
            required_columns.append('ask')
        
        config = FeatureConfig(
            name=f"bid_ask_spread_{window}_{base_calculation.value}",
            category=FeatureCategory.MICROSTRUCTURE,
            description=f"Bid-ask spread over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate bid-ask spread or fallback to price volatility when bid/ask not available."""
        # Check if bid and ask columns are available
        if 'bid' in data.columns and 'ask' in data.columns:
            if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
                bid = data['bid']
                ask = data['ask']
                spread = ask - bid
            else:
                base_values = self.base_calculator.calculate(data)
                # Use VectorBT rolling operations if available
                if hasattr(self, 'rolling_optimizer') and self.rolling_optimizer:
                    spread = self.rolling_optimizer.rolling_std(base_values, window=self.window)
                else:
                    spread = base_values.rolling(window=self.window).std()
        else:
            # Fallback: use high-low spread as proxy for bid-ask spread
            self.logger.warning(f"⚠️ Bid/ask columns not available, using high-low spread as proxy")
            if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
                spread = data['high'] - data['low']
            else:
                base_values = self.base_calculator.calculate(data)
                # Use VectorBT rolling operations if available
                if hasattr(self, 'rolling_optimizer') and self.rolling_optimizer:
                    spread = self.rolling_optimizer.rolling_std(base_values, window=self.window)
                else:
                    spread = base_values.rolling(window=self.window).std()

        # Validate that all values are finite and provide detailed information
        try:
            validate_finite(spread.values, f"BidAskSpread_{self.window}_{self.base_calculation.value}")
        except ValueError as e:
            # Get detailed information about where the NaN/inf values are
            non_finite_mask = ~np.isfinite(spread.values)
            if np.any(non_finite_mask):
                non_finite_indices = np.where(non_finite_mask)[0]
                total_count = len(non_finite_indices)

                # Show first few and last few problematic indices
                if total_count <= 10:
                    indices_str = f"indices {non_finite_indices.tolist()}"
                else:
                    first_5 = non_finite_indices[:5].tolist()
                    last_5 = non_finite_indices[-5:].tolist()
                    indices_str = f"indices {first_5} ... {last_5} (total: {total_count})"

                # Only log once per feature globally to reduce verbosity
                feature_key = f"BidAskSpread_{self.window}_{self.base_calculation.value}"
                # Use class-level tracking to prevent duplicate warnings across all instances
                if not hasattr(BidAskSpreadGenerator, '_logged_warnings'):
                    BidAskSpreadGenerator._logged_warnings = set()
                if feature_key not in BidAskSpreadGenerator._logged_warnings:
                    self.logger.warning(f"⚠️ {e} - {indices_str}")
                    BidAskSpreadGenerator._logged_warnings.add(feature_key)
            else:
                self.logger.warning(f"⚠️ {e}")

        return spread

# Order Flow Imbalance Generator
    
class OrderFlowImbalanceGenerator(VectorizedFeatureGenerator):
    """Generator for order flow imbalance features."""
    
    def __init__(self, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        if 'volume' not in required_columns:
            required_columns.append('volume')
        
        config = FeatureConfig(
            name=f"order_flow_imbalance_{window}_{base_calculation.value}",
            category=FeatureCategory.MICROSTRUCTURE,
            description=f"Order flow imbalance over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate order flow imbalance."""
        base_values = self.base_calculator.calculate(data)
        volume = data['volume']
        
        # Use VectorBT rolling operations if available
        if hasattr(self, 'rolling_optimizer') and self.rolling_optimizer:
            numerator = self.rolling_optimizer.rolling_sum(base_values * volume, window=self.window)
            denominator = self.rolling_optimizer.rolling_sum(volume, window=self.window)
        else:
            numerator = (base_values * volume).rolling(window=self.window).sum()
            denominator = volume.rolling(window=self.window).sum()
        
        order_flow_imbalance = numerator / denominator
        return order_flow_imbalance

# Trade Size Imbalance Generator
    
class TradeSizeImbalanceGenerator(VectorizedFeatureGenerator):
    """Generator for trade size imbalance features."""
    
    def __init__(self, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        if 'volume' not in required_columns:
            required_columns.append('volume')
        
        config = FeatureConfig(
            name=f"trade_size_imbalance_{window}_{base_calculation.value}",
            category=FeatureCategory.MICROSTRUCTURE,
            description=f"Trade size imbalance over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate trade size imbalance."""
        base_values = self.base_calculator.calculate(data)
        volume = data['volume']
        
        # Use VectorBT rolling operations if available
        if hasattr(self, 'rolling_optimizer') and self.rolling_optimizer:
            numerator = self.rolling_optimizer.rolling_sum(base_values * volume, window=self.window)
            denominator = self.rolling_optimizer.rolling_sum(volume, window=self.window)
        else:
            numerator = (base_values * volume).rolling(window=self.window).sum()
            denominator = volume.rolling(window=self.window).sum()
        
        trade_size_imbalance = numerator / denominator
        return trade_size_imbalance

# Price Impact Generator
    
class PriceImpactGenerator(VectorizedFeatureGenerator):
    """Generator for price impact features."""
    
    def __init__(self, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        if 'volume' not in required_columns:
            required_columns.append('volume')
        
        config = FeatureConfig(
            name=f"price_impact_{window}_{base_calculation.value}",
            category=FeatureCategory.MICROSTRUCTURE,
            description=f"Price impact over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate price impact."""
        base_values = self.base_calculator.calculate(data)
        volume = data['volume']
        
        # Use VectorBT rolling operations if available
        if hasattr(self, 'rolling_optimizer') and self.rolling_optimizer:
            numerator = self.rolling_optimizer.rolling_sum(base_values * volume, window=self.window)
            denominator = self.rolling_optimizer.rolling_sum(volume, window=self.window)
        else:
            numerator = (base_values * volume).rolling(window=self.window).sum()
            denominator = volume.rolling(window=self.window).sum()
        
        price_impact = numerator / denominator
        return price_impact

# Volume Weighted Price Generator
    
class VolumeWeightedPriceGenerator(VectorizedFeatureGenerator):
    """Generator for volume weighted price features."""
    
    def __init__(self, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        if 'volume' not in required_columns:
            required_columns.append('volume')
        
        config = FeatureConfig(
            name=f"volume_weighted_price_{window}_{base_calculation.value}",
            category=FeatureCategory.MICROSTRUCTURE,
            description=f"Volume weighted price over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate volume weighted price."""
        base_values = self.base_calculator.calculate(data)
        volume = data['volume']
        
        # Use VectorBT rolling operations if available
        if hasattr(self, 'rolling_optimizer') and self.rolling_optimizer:
            numerator = self.rolling_optimizer.rolling_sum(base_values * volume, window=self.window)
            denominator = self.rolling_optimizer.rolling_sum(volume, window=self.window)
        else:
            numerator = (base_values * volume).rolling(window=self.window).sum()
            denominator = volume.rolling(window=self.window).sum()
        
        volume_weighted_price = numerator / denominator
        return volume_weighted_price

# Trade Intensity Generator
    
class TradeIntensityGenerator(VectorizedFeatureGenerator):
    """Generator for trade intensity features."""
    
    def __init__(self, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        if 'volume' not in required_columns:
            required_columns.append('volume')
        
        config = FeatureConfig(
            name=f"trade_intensity_{window}_{base_calculation.value}",
            category=FeatureCategory.MICROSTRUCTURE,
            description=f"Trade intensity over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate trade intensity."""
        base_values = self.base_calculator.calculate(data)
        volume = data['volume']
        
        # Use VectorBT rolling operations if available
        if hasattr(self, 'rolling_optimizer') and self.rolling_optimizer:
            numerator = self.rolling_optimizer.rolling_sum(base_values * volume, window=self.window)
            denominator = self.rolling_optimizer.rolling_sum(volume, window=self.window)
        else:
            numerator = (base_values * volume).rolling(window=self.window).sum()
            denominator = volume.rolling(window=self.window).sum()
        
        trade_intensity = numerator / denominator
        return trade_intensity

# Liquidity Proxy Generator
    
class LiquidityProxyGenerator(VectorizedFeatureGenerator):
    """Generator for liquidity proxy features."""
    
    def __init__(self, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        if 'volume' not in required_columns:
            required_columns.append('volume')
        
        config = FeatureConfig(
            name=f"liquidity_proxy_{window}_{base_calculation.value}",
            category=FeatureCategory.MICROSTRUCTURE,
            description=f"Liquidity proxy over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate liquidity proxy."""
        base_values = self.base_calculator.calculate(data)
        volume = data['volume']
        
        # Use VectorBT rolling operations if available
        if hasattr(self, 'rolling_optimizer') and self.rolling_optimizer:
            numerator = self.rolling_optimizer.rolling_sum(base_values * volume, window=self.window)
            denominator = self.rolling_optimizer.rolling_sum(volume, window=self.window)
        else:
            numerator = (base_values * volume).rolling(window=self.window).sum()
            denominator = volume.rolling(window=self.window).sum()
        
        liquidity_proxy = numerator / denominator
        return liquidity_proxy

# Market Depth Generator
    
class MarketDepthGenerator(VectorizedFeatureGenerator):
    """Generator for market depth features."""
    
    def __init__(self, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        if 'volume' not in required_columns:
            required_columns.append('volume')
        
        config = FeatureConfig(
            name=f"market_depth_{window}_{base_calculation.value}",
            category=FeatureCategory.MICROSTRUCTURE,
            description=f"Market depth over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate market depth."""
        base_values = self.base_calculator.calculate(data)
        volume = data['volume']
        
        # Use VectorBT rolling operations if available
        if hasattr(self, 'rolling_optimizer') and self.rolling_optimizer:
            numerator = self.rolling_optimizer.rolling_sum(base_values * volume, window=self.window)
            denominator = self.rolling_optimizer.rolling_sum(volume, window=self.window)
        else:
            numerator = (base_values * volume).rolling(window=self.window).sum()
            denominator = volume.rolling(window=self.window).sum()
        
        market_depth = numerator / denominator
        return market_depth

def create_default_microstructure_generators() -> List[FeatureGenerator]:
    """Create default microstructure feature generators with VectorBT optimization."""
    windows = [5, 10, 20]
    
    generators = []
    
    # Create generators for each window
    for window in windows:
        # Create generators and add VectorBT optimization
        generator_list = [
            BidAskSpreadGenerator(window),
            OrderFlowImbalanceGenerator(window),
            TradeSizeImbalanceGenerator(window),
            PriceImpactGenerator(window),
            VolumeWeightedPriceGenerator(window),
            TradeIntensityGenerator(window),
            LiquidityProxyGenerator(window),
            MarketDepthGenerator(window),
        ]
        
        # Add VectorBT optimization to each generator
        for generator in generator_list:
            if OPTIMIZATION_AVAILABLE:
                generator.rolling_optimizer = get_vectorbt_rolling_optimizer()
                generator.vectorization_manager = get_unified_vectorization_manager()
        
        generators.extend(generator_list)

    # Analyst Features - Microstructure generators
    class AnalystSpreadNormalizedGenerator(VectorizedFeatureGenerator):
        """Generator for normalized spread feature."""

        def __init__(self):
            config = FeatureConfig(
                name="analyst_spread_normalized",
                category=FeatureCategory.MICROSTRUCTURE,
                description="Analyst normalized bid-ask spread using ATR",
                required_columns=["high", "low", "close"],
                default_lookback=20,
                min_lookback=10,
                max_lookback=100,
                parameters={}
            )
            super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

        def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
            # Optimize DataFrame for processing
            if hasattr(self, 'optimize_dataframe_processing'):
                data = self.optimize_dataframe_processing(data)

            """Generate normalized spread feature."""
            # Spread calculation (using high-low as proxy)
            spread = (data['high'] - data['low']) / data['close']

            # ATR for normalization (using simplified ATR calculation)
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift(1))
            low_close = np.abs(data['low'] - data['close'].shift(1))
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            
            # Use VectorBT rolling operations if available
            if hasattr(self, 'rolling_optimizer') and self.rolling_optimizer:
                atr = self.rolling_optimizer.rolling_mean(true_range, window=14)
            else:
                atr = true_range.rolling(14).mean()

            spread_normalized = spread / atr.replace(0, 1)
            return spread_normalized

    class AnalystTickImbalanceGenerator(VectorizedFeatureGenerator):
        """Generator for tick imbalance feature."""

        def __init__(self, lookback: int = 100):
            config = FeatureConfig(
                name="analyst_tick_imbalance",
                category=FeatureCategory.MICROSTRUCTURE,
                description="Analyst tick imbalance ((upticks - downticks) / total_ticks)",
                required_columns=["close"],
                default_lookback=lookback,
                min_lookback=50,
                max_lookback=200,
                parameters={"lookback": lookback}
            )
            super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
            self.lookback = lookback

        def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
            # Optimize DataFrame for processing
            if hasattr(self, 'optimize_dataframe_processing'):
                data = self.optimize_dataframe_processing(data)

            """Generate tick imbalance feature."""
            price_changes = data['close'].diff()

            # Count upticks vs downticks in rolling window
            # Use VectorBT rolling operations if available
            if hasattr(self, 'rolling_optimizer') and self.rolling_optimizer:
                upticks = self.rolling_optimizer.rolling_sum((price_changes > 0).astype(int), window=self.lookback)
                downticks = self.rolling_optimizer.rolling_sum((price_changes < 0).astype(int), window=self.lookback)
            else:
                upticks = (price_changes > 0).rolling(self.lookback).sum()
                downticks = (price_changes < 0).rolling(self.lookback).sum()
            
            total_ticks = upticks + downticks

            tick_imbalance = (upticks - downticks) / total_ticks.replace(0, 1)
            return tick_imbalance

    # Add VectorBT optimization to analyst features
    analyst_generators = [AnalystSpreadNormalizedGenerator(), AnalystTickImbalanceGenerator()]
    for generator in analyst_generators:
        if OPTIMIZATION_AVAILABLE:
            generator.rolling_optimizer = get_vectorbt_rolling_optimizer()
            generator.vectorization_manager = get_unified_vectorization_manager()
    
    generators.extend(analyst_generators)

    # New microstructure interaction features
    class CorwinSchultzSpreadMomentumGenerator(VectorizedFeatureGenerator):
        """Generator for Corwin-Schultz spread proxy × momentum interaction feature."""

        def __init__(self, spread_window: int = 20, momentum_period: int = 5):
            config = FeatureConfig(
                name="cs_spread_momentum",
                category=FeatureCategory.MICROSTRUCTURE,
                description="Corwin-Schultz spread proxy × momentum interaction (wide spreads → trend breaks sooner)",
                required_columns=["high", "low", "close"],
                default_lookback=max(spread_window, momentum_period),
                min_lookback=10,
                max_lookback=100,
                parameters={"spread_window": spread_window, "momentum_period": momentum_period}
            )
            super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
            self.spread_window = spread_window
            self.momentum_period = momentum_period

        def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
            # Optimize DataFrame for processing
            if hasattr(self, 'optimize_dataframe_processing'):
                data = self.optimize_dataframe_processing(data)

            """Generate Corwin-Schultz spread proxy × momentum interaction."""
            # Calculate Corwin-Schultz spread proxy
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Corwin-Schultz spread proxy: (high - low) / close
            cs_spread = (high - low) / close
            
            # Calculate momentum (5-period price change)
            momentum = close.pct_change(self.momentum_period)
            
            # Interaction: CS spread × momentum
            interaction = cs_spread * momentum
            
            return interaction

    class AmihudIlliquidityVWAPDistanceGenerator(VectorizedFeatureGenerator):
        """Generator for Amihud illiquidity × VWAP distance interaction feature."""

        def __init__(self, illiquidity_window: int = 20, vwap_window: int = 20):
            config = FeatureConfig(
                name="amihud_illiquidity_vwap_distance",
                category=FeatureCategory.MICROSTRUCTURE,
                description="Amihud illiquidity × VWAP distance (big price move per $ volume → mean reversion risk)",
                required_columns=["high", "low", "close", "volume"],
                default_lookback=max(illiquidity_window, vwap_window),
                min_lookback=10,
                max_lookback=100,
                parameters={"illiquidity_window": illiquidity_window, "vwap_window": vwap_window}
            )
            super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
            self.illiquidity_window = illiquidity_window
            self.vwap_window = vwap_window

        def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
            # Optimize DataFrame for processing
            if hasattr(self, 'optimize_dataframe_processing'):
                data = self.optimize_dataframe_processing(data)

            """Generate Amihud illiquidity × VWAP distance interaction."""
            close = data['close']
            volume = data['volume']
            
            # Calculate returns
            returns = close.pct_change()
            
            # Amihud illiquidity: |returns| / volume
            amihud_illiquidity = np.abs(returns) / volume
            
            # Use VectorBT rolling operations if available
            if hasattr(self, 'rolling_optimizer') and self.rolling_optimizer:
                amihud_illiquidity = self.rolling_optimizer.rolling_mean(amihud_illiquidity, window=self.illiquidity_window)
                # Calculate VWAP
                vwap_numerator = self.rolling_optimizer.rolling_sum(close * volume, window=self.vwap_window)
                vwap_denominator = self.rolling_optimizer.rolling_sum(volume, window=self.vwap_window)
                vwap = vwap_numerator / vwap_denominator
            else:
                amihud_illiquidity = amihud_illiquidity.rolling(window=self.illiquidity_window).mean()
                # Calculate VWAP
                vwap = (close * volume).rolling(window=self.vwap_window).sum() / volume.rolling(window=self.vwap_window).sum()
            
            # VWAP distance: (close - vwap) / vwap
            vwap_distance = (close - vwap) / vwap
            
            # Interaction: Amihud illiquidity × VWAP distance
            interaction = amihud_illiquidity * vwap_distance
            
            return interaction

    class RollLambdaRVShortGenerator(VectorizedFeatureGenerator):
        """Generator for Roll's λ (signed autocov) × rv_short interaction feature."""

        def __init__(self, roll_window: int = 20, rv_window: int = 5):
            config = FeatureConfig(
                name="roll_lambda_rv_short",
                category=FeatureCategory.MICROSTRUCTURE,
                description="Roll's λ (signed autocov) × rv_short (implicit spread/high trans. costs amplify vol impact)",
                required_columns=["close"],
                default_lookback=max(roll_window, rv_window),
                min_lookback=10,
                max_lookback=100,
                parameters={"roll_window": roll_window, "rv_window": rv_window}
            )
            super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
            self.roll_window = roll_window
            self.rv_window = rv_window

        def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
            # Optimize DataFrame for processing
            if hasattr(self, 'optimize_dataframe_processing'):
                data = self.optimize_dataframe_processing(data)

            """Generate Roll's λ × rv_short interaction."""
            close = data['close']
            
            # Calculate returns
            returns = close.pct_change()
            
            # Roll's λ: signed autocovariance of returns
            # λ = -2 * cov(returns_t, returns_{t-1})
            if hasattr(self, 'rolling_optimizer') and self.rolling_optimizer:
                roll_lambda = -2 * self.rolling_optimizer.rolling_apply(
                    returns, lambda x: np.cov(x[:-1], x[1:])[0, 1] if len(x) > 1 else 0, window=self.roll_window
                )
                # Realized volatility (short-term)
                rv_short = self.rolling_optimizer.rolling_std(returns, window=self.rv_window)
            else:
                roll_lambda = -2 * returns.rolling(window=self.roll_window).apply(
                    lambda x: np.cov(x[:-1], x[1:])[0, 1] if len(x) > 1 else 0, raw=False
                )
                # Realized volatility (short-term)
                rv_short = returns.rolling(window=self.rv_window).std()
            
            # Interaction: Roll's λ × rv_short
            interaction = roll_lambda * rv_short
            
            return interaction

    class RangeVolumeShockOpen30Generator(VectorizedFeatureGenerator):
        """Generator for Range/Volume shock × open30 interaction feature."""

        def __init__(self, range_volume_window: int = 20, open30_window: int = 30):
            config = FeatureConfig(
                name="range_volume_shock_open30",
                category=FeatureCategory.MICROSTRUCTURE,
                description="Range/Volume shock × open30 (thin-open shock filter)",
                required_columns=["high", "low", "open", "volume"],
                default_lookback=max(range_volume_window, open30_window),
                min_lookback=10,
                max_lookback=100,
                parameters={"range_volume_window": range_volume_window, "open30_window": open30_window}
            )
            super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
            self.range_volume_window = range_volume_window
            self.open30_window = open30_window

        def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
            # Optimize DataFrame for processing
            if hasattr(self, 'optimize_dataframe_processing'):
                data = self.optimize_dataframe_processing(data)

            """Generate Range/Volume shock × open30 interaction."""
            high = data['high']
            low = data['low']
            open_price = data['open']
            volume = data['volume']
            
            # Range/Volume ratio: (high - low) / volume
            range_volume_ratio = (high - low) / volume
            
            # Z-score of range/volume ratio
            if hasattr(self, 'rolling_optimizer') and self.rolling_optimizer:
                range_volume_mean = self.rolling_optimizer.rolling_mean(range_volume_ratio, window=self.range_volume_window)
                range_volume_std = self.rolling_optimizer.rolling_std(range_volume_ratio, window=self.range_volume_window)
                range_volume_z = (range_volume_ratio - range_volume_mean) / range_volume_std
            else:
                range_volume_z = (range_volume_ratio - range_volume_ratio.rolling(window=self.range_volume_window).mean()) / range_volume_ratio.rolling(window=self.range_volume_window).std()
            
            # Open30: 30-period open price change
            open30 = open_price.pct_change(self.open30_window)
            
            # Interaction: (high-low)/volume_z × open30
            interaction = range_volume_z * open30
            
            return interaction

    # Add VectorBT optimization to interaction features
    interaction_generators = [
        CorwinSchultzSpreadMomentumGenerator(),
        AmihudIlliquidityVWAPDistanceGenerator(),
        RollLambdaRVShortGenerator(),
        RangeVolumeShockOpen30Generator()
    ]
    
    for generator in interaction_generators:
        if OPTIMIZATION_AVAILABLE:
            generator.rolling_optimizer = get_vectorbt_rolling_optimizer()
            generator.vectorization_manager = get_unified_vectorization_manager()
    
    generators.extend(interaction_generators)

    return generators
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
    
    def generate_optimized_microstructure_features(self, data: pd.DataFrame, 
                                                 feature_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Generate multiple microstructure features using optimized batch processing.
        
        Args:
            data: OHLCV data
            feature_configs: List of feature configuration dictionaries
            
        Returns:
            DataFrame with generated microstructure features
        """
        if self.vectorization_manager:
            try:
                # Use Unified Vectorization Manager for batch processing
                return self.vectorization_manager.batch_process_features(data, feature_configs)
            except Exception as e:
                self.logger.warning(f"Unified Vectorization Manager batch processing failed: {e}, using VectorBT fallback")
                return self._generate_batch_with_vectorbt(data, feature_configs)
        elif self.rolling_optimizer:
            try:
                return self._generate_batch_with_vectorbt(data, feature_configs)
            except Exception as e:
                self.logger.warning(f"VectorBT batch processing failed: {e}, using pandas fallback")
                return self._generate_batch_with_pandas(data, feature_configs)
        else:
            return self._generate_batch_with_pandas(data, feature_configs)
    
    def _generate_batch_with_vectorbt(self, data: pd.DataFrame, 
                                    feature_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """Generate batch features using VectorBT rolling optimizer."""
        results = {}
        
        for config in feature_configs:
            feature_name = config['name']
            feature_type = config.get('type', 'microstructure')
            params = config.get('params', {})
            
            try:
                if feature_type == 'microstructure':
                    window = params.get('window', 20)
                    column = params.get('column', 'close')
                    
                    if column in data.columns:
                        series_data = data[column]
                        
                        # Calculate microstructure features using VectorBT
                        if feature_name == 'spread_volatility':
                            # Calculate rolling volatility of price changes
                            price_changes = series_data.pct_change()
                            results[feature_name] = self.rolling_optimizer.rolling_std(price_changes, window=window)
                        
                        elif feature_name == 'trade_intensity':
                            # Calculate rolling mean of volume
                            if 'volume' in data.columns:
                                volume_data = data['volume']
                                results[feature_name] = self.rolling_optimizer.rolling_mean(volume_data, window=window)
                            else:
                                results[feature_name] = pd.Series(np.nan, index=data.index)
                        
                        elif feature_name == 'price_efficiency':
                            # Calculate rolling correlation between price and volume
                            if 'volume' in data.columns:
                                volume_data = data['volume']
                                results[feature_name] = self.rolling_optimizer.rolling_corr(series_data, volume_data, window=window)
                            else:
                                results[feature_name] = pd.Series(np.nan, index=data.index)
                        
                        else:
                            # Default to rolling mean
                            results[feature_name] = self.rolling_optimizer.rolling_mean(series_data, window=window)
                
                elif feature_type == 'rolling':
                    operation = params.get('operation', 'mean')
                    window = params.get('window', 20)
                    column = params.get('column', 'close')
                    
                    if column in data.columns:
                        series_data = data[column]
                        if operation == 'mean':
                            results[feature_name] = self.rolling_optimizer.rolling_mean(series_data, window=window)
                        elif operation == 'std':
                            results[feature_name] = self.rolling_optimizer.rolling_std(series_data, window=window)
                        elif operation == 'min':
                            results[feature_name] = self.rolling_optimizer.rolling_min(series_data, window=window)
                        elif operation == 'max':
                            results[feature_name] = self.rolling_optimizer.rolling_max(series_data, window=window)
                        elif operation == 'sum':
                            results[feature_name] = self.rolling_optimizer.rolling_sum(series_data, window=window)
                
            except Exception as e:
                self.logger.warning(f"Microstructure feature {feature_name} failed: {e}")
                results[feature_name] = pd.Series(np.nan, index=data.index)
        
        return pd.DataFrame(results, index=data.index)
    
    def _generate_batch_with_pandas(self, data: pd.DataFrame, 
                                  feature_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """Generate batch features using pandas fallback."""
        results = {}
        
        for config in feature_configs:
            feature_name = config['name']
            feature_type = config.get('type', 'microstructure')
            params = config.get('params', {})
            
            try:
                if feature_type == 'microstructure':
                    window = params.get('window', 20)
                    column = params.get('column', 'close')
                    
                    if column in data.columns:
                        series_data = data[column]
                        
                        # Calculate microstructure features using pandas
                        if feature_name == 'spread_volatility':
                            # Calculate rolling volatility of price changes
                            price_changes = series_data.pct_change()
                            results[feature_name] = price_changes.rolling(window=window).std()
                        
                        elif feature_name == 'trade_intensity':
                            # Calculate rolling mean of volume
                            if 'volume' in data.columns:
                                volume_data = data['volume']
                                results[feature_name] = volume_data.rolling(window=window).mean()
                            else:
                                results[feature_name] = pd.Series(np.nan, index=data.index)
                        
                        elif feature_name == 'price_efficiency':
                            # Calculate rolling correlation between price and volume
                            if 'volume' in data.columns:
                                volume_data = data['volume']
                                results[feature_name] = series_data.rolling(window=window).corr(volume_data)
                            else:
                                results[feature_name] = pd.Series(np.nan, index=data.index)
                        
                        else:
                            # Default to rolling mean
                            results[feature_name] = series_data.rolling(window=window).mean()
                
                elif feature_type == 'rolling':
                    operation = params.get('operation', 'mean')
                    window = params.get('window', 20)
                    column = params.get('column', 'close')
                    
                    if column in data.columns:
                        series_data = data[column]
                        rolling_obj = series_data.rolling(window=window)
                        if operation == 'mean':
                            results[feature_name] = rolling_obj.mean()
                        elif operation == 'std':
                            results[feature_name] = rolling_obj.std()
                        elif operation == 'min':
                            results[feature_name] = rolling_obj.min()
                        elif operation == 'max':
                            results[feature_name] = rolling_obj.max()
                        elif operation == 'sum':
                            results[feature_name] = rolling_obj.sum()
                
            except Exception as e:
                self.logger.warning(f"Microstructure feature {feature_name} failed: {e}")
                results[feature_name] = pd.Series(np.nan, index=data.index)
        
        return pd.DataFrame(results, index=data.index)
