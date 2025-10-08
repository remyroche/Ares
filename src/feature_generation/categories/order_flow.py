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