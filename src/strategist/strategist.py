"""
Base Strategist Implementation
Provides a foundation for all trading strategies with comprehensive decorators.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.enhanced_data_quality_decorators import validate_memory_optimized_data_quality
from src.utils.trading_decorators import performance_monitor
from src.utils.logger import system_logger


@dataclass
class StrategyConfig:
    """Base configuration for all strategies."""
    
    strategy_name: str
    enabled: bool = True
    risk_tolerance: float = 0.02  # 2% risk per trade
    max_position_size: float = 0.25  # 25% max position
    enable_stop_loss: bool = True
    enable_take_profit: bool = True
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.04  # 4% take profit


class BaseStrategist(ABC):
    """
    Abstract base class for all trading strategies.
    
    Provides common functionality and ensures proper decorator usage
    across all strategy implementations.
    """
    
    def __init__(self, config: StrategyConfig):
        """Initialize base strategist."""
        self.config = config
        self.logger = system_logger.getChild(f"BaseStrategist.{config.strategy_name}")
        self.is_active = False
        self.performance_metrics = {}
        
        self.logger.info(f"Initialized {config.strategy_name} strategy")
    
    @performance_monitor
    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid strategy configuration"),
            AttributeError: (False, "Missing required strategy parameters"),
        },
        default_return=False,
        context="strategy initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize the strategy with proper error handling.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            if not self._validate_configuration():
                self.logger.error("Strategy configuration validation failed")
                return False
            
            await self._initialize_strategy_specific()
            self.is_active = True
            
            self.logger.info(f"✅ {self.config.strategy_name} strategy initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.config.strategy_name} strategy: {e}")
            return False
    
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """Validate strategy configuration."""
        if not self.config.strategy_name:
            self.logger.error("Strategy name is required")
            return False
        
        if self.config.risk_tolerance <= 0 or self.config.risk_tolerance > 1:
            self.logger.error("Risk tolerance must be between 0 and 1")
            return False
        
        if self.config.max_position_size <= 0 or self.config.max_position_size > 1:
            self.logger.error("Max position size must be between 0 and 1")
            return False
        
        return True
    
    @abstractmethod
    async def _initialize_strategy_specific(self) -> None:
        """Initialize strategy-specific components. Must be implemented by subclasses."""
        pass
    
    @performance_monitor
    @validate_memory_optimized_data_quality
    @handle_errors(
        exceptions=(ValueError, TypeError, KeyError),
        default_return=None,
        context="strategy signal generation",
    )
    async def generate_signal(
        self,
        market_data: pd.DataFrame,
        analysis_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate trading signal with comprehensive validation.
        
        Args:
            market_data: Market price and volume data
            analysis_data: Optional analysis results
            
        Returns:
            Optional[Dict[str, Any]]: Trading signal or None if no signal
        """
        if not self.is_active:
            self.logger.warning("Strategy is not active")
            return None
        
        if market_data.empty:
            self.logger.warning("Empty market data provided")
            return None
        
        try:
            signal = await self._generate_strategy_signal(market_data, analysis_data)
            
            if signal:
                signal = self._apply_risk_management(signal, market_data)
                self._log_signal_generation(signal)
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            return None
    
    @abstractmethod
    async def _generate_strategy_signal(
        self,
        market_data: pd.DataFrame,
        analysis_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Generate strategy-specific signal. Must be implemented by subclasses."""
        pass
    
    @handle_errors(
        exceptions=(ValueError, TypeError),
        default_return=None,
        context="risk management application",
    )
    def _apply_risk_management(
        self,
        signal: Dict[str, Any],
        market_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Apply risk management rules to the signal."""
        if not signal:
            return signal
        
        current_price = market_data['close'].iloc[-1] if 'close' in market_data.columns else 0
        
        # Apply stop loss
        if self.config.enable_stop_loss and current_price > 0:
            signal['stop_loss'] = current_price * (1 - self.config.stop_loss_pct)
        
        # Apply take profit
        if self.config.enable_take_profit and current_price > 0:
            signal['take_profit'] = current_price * (1 + self.config.take_profit_pct)
        
        # Apply position size limits
        if 'position_size' in signal:
            signal['position_size'] = min(
                signal['position_size'],
                self.config.max_position_size
            )
        
        return signal
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="signal logging",
    )
    def _log_signal_generation(self, signal: Dict[str, Any]) -> None:
        """Log signal generation details."""
        if signal:
            self.logger.info(
                f"Generated signal: {signal.get('action', 'unknown')} "
                f"with confidence: {signal.get('confidence', 0):.2f}"
            )
    
    @performance_monitor
    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="performance metrics calculation",
    )
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get strategy performance metrics."""
        return {
            'strategy_name': self.config.strategy_name,
            'is_active': self.is_active,
            'risk_tolerance': self.config.risk_tolerance,
            'max_position_size': self.config.max_position_size,
            'performance_metrics': self.performance_metrics.copy(),
        }
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="strategy shutdown",
    )
    async def shutdown(self) -> None:
        """Shutdown the strategy gracefully."""
        self.logger.info(f"🛑 Shutting down {self.config.strategy_name} strategy")
        
        try:
            await self._shutdown_strategy_specific()
            self.is_active = False
            self.logger.info(f"✅ {self.config.strategy_name} strategy shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during strategy shutdown: {e}")
    
    @abstractmethod
    async def _shutdown_strategy_specific(self) -> None:
        """Shutdown strategy-specific components. Must be implemented by subclasses."""
        pass


class SimpleMovingAverageStrategist(BaseStrategist):
    """
    Simple Moving Average Strategy implementation.
    
    Example implementation showing proper decorator usage.
    """
    
    def __init__(self, config: StrategyConfig, short_window: int = 10, long_window: int = 30):
        """Initialize SMA strategy."""
        super().__init__(config)
        self.short_window = short_window
        self.long_window = long_window
    
    async def _initialize_strategy_specific(self) -> None:
        """Initialize SMA-specific components."""
        self.logger.info(f"Initialized SMA strategy with windows: {self.short_window}/{self.long_window}")
    
    @performance_monitor
    @validate_memory_optimized_data_quality
    @handle_errors(
        exceptions=(ValueError, TypeError),
        default_return=None,
        context="SMA signal generation",
    )
    async def _generate_strategy_signal(
        self,
        market_data: pd.DataFrame,
        analysis_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Generate SMA-based trading signal."""
        if len(market_data) < self.long_window:
            return None
        
        # Calculate moving averages
        short_ma = market_data['close'].rolling(window=self.short_window).mean()
        long_ma = market_data['close'].rolling(window=self.long_window).mean()
        
        current_short_ma = short_ma.iloc[-1]
        current_long_ma = long_ma.iloc[-1]
        prev_short_ma = short_ma.iloc[-2]
        prev_long_ma = long_ma.iloc[-2]
        
        # Generate signal based on MA crossover
        if current_short_ma > current_long_ma and prev_short_ma <= prev_long_ma:
            # Golden cross - buy signal
            return {
                'action': 'buy',
                'confidence': 0.7,
                'position_size': self.config.risk_tolerance,
                'reason': f'SMA crossover: {self.short_window}MA > {self.long_window}MA',
            }
        elif current_short_ma < current_long_ma and prev_short_ma >= prev_long_ma:
            # Death cross - sell signal
            return {
                'action': 'sell',
                'confidence': 0.7,
                'position_size': self.config.risk_tolerance,
                'reason': f'SMA crossover: {self.short_window}MA < {self.long_window}MA',
            }
        
        return None
    
    async def _shutdown_strategy_specific(self) -> None:
        """Shutdown SMA-specific components."""
        self.logger.info("SMA strategy specific shutdown complete")


# Example usage
if __name__ == "__main__":
    import asyncio
    import numpy as np
    
    async def test_strategy():
        config = StrategyConfig(
            strategy_name="SMA_Strategy",
            risk_tolerance=0.02,
            max_position_size=0.25,
        )
        
        strategy = SimpleMovingAverageStrategist(config)
        
        # Initialize
        success = await strategy.initialize()
        print(f"Strategy initialization: {'Success' if success else 'Failed'}")
        
        # Generate sample data
        dates = pd.date_range("2023-01-01", "2024-01-01", freq="D")
        prices = 100 * (1 + 0.001 * np.cumsum(np.random.randn(len(dates))))
        
        sample_data = pd.DataFrame({
            'close': prices,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'volume': np.random.randint(1000, 10000, len(dates)),
        }, index=dates)
        
        # Generate signal
        signal = await strategy.generate_signal(sample_data)
        print(f"Generated signal: {signal}")
        
        # Get performance metrics
        metrics = strategy.get_performance_metrics()
        print(f"Performance metrics: {metrics}")
        
        # Shutdown
        await strategy.shutdown()
    
    # Run test
    asyncio.run(test_strategy())