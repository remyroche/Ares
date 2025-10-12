"""Order Flow features"""
import pandas as pd
from typing import List, Optional
from ..core.feature_generator import FeatureGenerator, FeatureConfig, FeatureCategory, VectorizedFeatureGenerator

# Optimization utilities
try:
    from ..utils.vectorization_optimizer import get_vectorization_optimizer
    from ..utils.optimized_feature_pipeline import get_optimized_feature_pipeline
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
from ..base_calculations import BaseCalculationType, create_base_calculator

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

# VectorBT-optimized order flow generators will be defined inline below
VECTORBT_ORDER_FLOW_AVAILABLE = VECTORBT_AVAILABLE

class TakerBuyRatioGenerator(VectorizedFeatureGenerator):
    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name=f"taker_buy_ratio_{window}",
            category=FeatureCategory.ORDER_FLOW,
            description=f"Taker buy ratio over {window} periods",
            required_columns=["close", "volume"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close']
        volume = data['volume']
        
        # Simulate taker buy ratio based on price movement and volume
        price_change = close.pct_change()
        buy_pressure = (price_change > 0).astype(int) * volume
        total_volume = volume.rolling(window=self.window).sum()
        buy_volume = buy_pressure.rolling(window=self.window).sum()
        
        return buy_volume / total_volume.replace(0, 1)
    
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

    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class TakerSellRatioGenerator(FeatureGenerator):
    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name=f"taker_sell_ratio_{window}",
            category=FeatureCategory.ORDER_FLOW,
            description=f"Taker sell ratio over {window} periods",
            required_columns=["close", "volume"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close']
        volume = data['volume']
        
        # Simulate taker sell ratio based on price movement and volume
        price_change = close.pct_change()
        sell_pressure = (price_change < 0).astype(int) * volume
        total_volume = volume.rolling(window=self.window).sum()
        sell_volume = sell_pressure.rolling(window=self.window).sum()
        
        return sell_volume / total_volume.replace(0, 1)

    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class MarketAggressionIndexGenerator(FeatureGenerator):
    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name=f"market_aggression_index_{window}",
            category=FeatureCategory.ORDER_FLOW,
            description=f"Market aggression index over {window} periods",
            required_columns=["close", "volume"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close']
        volume = data['volume']
        
        # Calculate market aggression based on price velocity and volume
        price_velocity = close.pct_change().abs()
        aggression = price_velocity * volume
        
        return aggression.rolling(window=self.window).mean()

    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class OrderFlowImbalanceGenerator(FeatureGenerator):
    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name=f"order_flow_imbalance_{window}",
            category=FeatureCategory.ORDER_FLOW,
            description=f"Order flow imbalance over {window} periods",
            required_columns=["close", "volume"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close']
        volume = data['volume']
        
        # Calculate order flow imbalance
        price_change = close.pct_change()
        buy_volume = (price_change > 0).astype(int) * volume
        sell_volume = (price_change < 0).astype(int) * volume
        
        buy_sum = buy_volume.rolling(window=self.window).sum()
        sell_sum = sell_volume.rolling(window=self.window).sum()
        
        return (buy_sum - sell_sum) / (buy_sum + sell_sum).replace(0, 1)

def create_default_order_flow_generators() -> List[FeatureGenerator]:
    generators = []
    
    # Use VectorBT generators if available, otherwise fall back to legacy generators
    if VECTORBT_ORDER_FLOW_AVAILABLE and VECTORBT_AVAILABLE:
        # Use VectorBT-optimized generators
        generators.extend(create_default_vectorbt_order_flow_generators())
    else:
        # Fall back to legacy generators
        windows = [5, 10, 20]
        
        for window in windows:
            generators.extend([
                TakerBuyRatioGenerator(window),
                TakerSellRatioGenerator(window),
                MarketAggressionIndexGenerator(window),
                OrderFlowImbalanceGenerator(window),
            ])

        # Analyst Features - Order flow
        generators.append(AnalystBidAskImbalanceGenerator())
        generators.append(AnalystMarketOrderFlowGenerator())

    return generators

# Analyst Features - Order flow generators
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class AnalystBidAskImbalanceGenerator(FeatureGenerator):
    """Generator for bid-ask imbalance feature."""

    def __init__(self):
        config = FeatureConfig(
            name="analyst_bid_ask_imbalance",
            category=FeatureCategory.ORDER_FLOW,
            description="Analyst bid-ask imbalance ((bid_size - ask_size) / (bid_size + ask_size))",
            required_columns=["bid", "ask"],
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

        """Generate bid-ask imbalance feature."""
        if 'bid' in data.columns and 'ask' in data.columns:
            bid_size = data['bid']
            ask_size = data['ask']

            bid_ask_imbalance = (bid_size - ask_size) / (bid_size + ask_size).replace(0, 1)
            return bid_ask_imbalance
        else:
            # Return neutral value if bid/ask data not available
            return pd.Series([0.0] * len(data), index=data.index, name=self.config.name)

    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class AnalystMarketOrderFlowGenerator(FeatureGenerator):
    """Generator for market order flow feature."""

    def __init__(self):
        config = FeatureConfig(
            name="analyst_market_order_flow",
            category=FeatureCategory.ORDER_FLOW,
            description="Analyst market order flow (market_buys - market_sells)",
            required_columns=["market_buys", "market_sells"],
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

        """Generate market order flow feature."""
        if 'market_buys' in data.columns and 'market_sells' in data.columns:
            market_buys = data['market_buys']
            market_sells = data['market_sells']

            market_order_flow = market_buys - market_sells
            return market_order_flow
        else:
            # Return neutral value if market order data not available
            return pd.Series([0.0] * len(data), index=data.index, name=self.config.name)
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
