"""Strategist configuration utilities."""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging

class MarketIndicators:
    """Market indicators configuration and management."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize market indicators."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Default indicator configurations
        self.default_indicators = {
            'rsi': {'period': 14, 'overbought': 70, 'oversold': 30},
            'macd': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
            'bollinger_bands': {'period': 20, 'std_dev': 2},
            'sma': {'periods': [20, 50, 200]},
            'ema': {'periods': [12, 26]},
            'atr': {'period': 14},
            'stochastic': {'k_period': 14, 'd_period': 3},
            'williams_r': {'period': 14},
            'cci': {'period': 20},
            'adx': {'period': 14}
        }
        
        # Load custom configurations
        self.indicators = self._load_indicator_configs()
    
    def _load_indicator_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load indicator configurations from config."""
        indicators = self.default_indicators.copy()
        
        if 'indicators' in self.config:
            indicators.update(self.config['indicators'])
        
        return indicators
    
    def get_indicator_config(self, indicator_name: str) -> Dict[str, Any]:
        """Get configuration for a specific indicator."""
        return self.indicators.get(indicator_name, {})
    
    def update_indicator_config(self, indicator_name: str, config: Dict[str, Any]) -> None:
        """Update configuration for a specific indicator."""
        self.indicators[indicator_name] = config
        self.logger.info(f"Updated indicator config for {indicator_name}")
    
    def get_all_indicators(self) -> Dict[str, Dict[str, Any]]:
        """Get all indicator configurations."""
        return self.indicators.copy()
    
    def validate_indicator_config(self, indicator_name: str, config: Dict[str, Any]) -> bool:
        """Validate indicator configuration."""
        try:
            if indicator_name == 'rsi':
                return all(key in config for key in ['period', 'overbought', 'oversold'])
            elif indicator_name == 'macd':
                return all(key in config for key in ['fast_period', 'slow_period', 'signal_period'])
            elif indicator_name == 'bollinger_bands':
                return all(key in config for key in ['period', 'std_dev'])
            elif indicator_name in ['sma', 'ema']:
                return 'periods' in config and isinstance(config['periods'], list)
            elif indicator_name == 'atr':
                return 'period' in config
            elif indicator_name == 'stochastic':
                return all(key in config for key in ['k_period', 'd_period'])
            elif indicator_name == 'williams_r':
                return 'period' in config
            elif indicator_name == 'cci':
                return 'period' in config
            elif indicator_name == 'adx':
                return 'period' in config
            else:
                return True  # Unknown indicator, assume valid
        except Exception as e:
            self.logger.error(f"Error validating indicator config: {e}")
            return False

@dataclass
class StrategistConfig:
    """Strategist configuration with comprehensive settings."""
    
    # Basic configuration
    symbol: str = "BTCUSDT"
    timeframe: str = "1h"
    initial_capital: float = 10000.0
    
    # Risk management
    risk_per_trade: float = 0.02
    max_position_size: float = 0.1
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.1
    
    # Strategy parameters
    strategy_type: str = "trend_following"
    lookback_period: int = 100
    min_confidence: float = 0.6
    
    # Market indicators
    indicators: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Performance settings
    enable_optimization: bool = True
    optimization_method: str = "genetic_algorithm"
    optimization_params: Dict[str, Any] = field(default_factory=dict)
    
    # Data settings
    data_source: str = "binance"
    data_quality_threshold: float = 0.95
    max_missing_data_pct: float = 0.05
    
    # Logging and monitoring
    log_level: str = "INFO"
    enable_performance_monitoring: bool = True
    performance_report_interval: int = 1000  # trades
    
    # Advanced settings
    enable_ml_features: bool = False
    ml_model_path: Optional[str] = None
    feature_engineering: Dict[str, Any] = field(default_factory=dict)
    
    # Backtesting settings
    enable_backtesting: bool = True
    backtest_start_date: Optional[str] = None
    backtest_end_date: Optional[str] = None
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        self._validate_config()
        self._setup_defaults()
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        
        if not 0 < self.risk_per_trade <= 0.1:
            raise ValueError("Risk per trade must be between 0 and 0.1")
        
        if not 0 < self.max_position_size <= 1.0:
            raise ValueError("Max position size must be between 0 and 1.0")
        
        if not 0 < self.stop_loss_pct <= 0.5:
            raise ValueError("Stop loss percentage must be between 0 and 0.5")
        
        if not 0 < self.take_profit_pct <= 1.0:
            raise ValueError("Take profit percentage must be between 0 and 1.0")
        
        if not 0 <= self.min_confidence <= 1.0:
            raise ValueError("Minimum confidence must be between 0 and 1.0")
    
    def _setup_defaults(self) -> None:
        """Setup default values for optional parameters."""
        if not self.indicators:
            self.indicators = {
                'rsi': {'period': 14, 'overbought': 70, 'oversold': 30},
                'macd': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
                'bollinger_bands': {'period': 20, 'std_dev': 2}
            }
        
        if not self.optimization_params:
            self.optimization_params = {
                'population_size': 50,
                'generations': 100,
                'mutation_rate': 0.1,
                'crossover_rate': 0.8
            }
        
        if not self.feature_engineering:
            self.feature_engineering = {
                'enable_technical_indicators': True,
                'enable_price_features': True,
                'enable_volume_features': True,
                'enable_time_features': False
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'initial_capital': self.initial_capital,
            'risk_per_trade': self.risk_per_trade,
            'max_position_size': self.max_position_size,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'strategy_type': self.strategy_type,
            'lookback_period': self.lookback_period,
            'min_confidence': self.min_confidence,
            'indicators': self.indicators,
            'enable_optimization': self.enable_optimization,
            'optimization_method': self.optimization_method,
            'optimization_params': self.optimization_params,
            'data_source': self.data_source,
            'data_quality_threshold': self.data_quality_threshold,
            'max_missing_data_pct': self.max_missing_data_pct,
            'log_level': self.log_level,
            'enable_performance_monitoring': self.enable_performance_monitoring,
            'performance_report_interval': self.performance_report_interval,
            'enable_ml_features': self.enable_ml_features,
            'ml_model_path': self.ml_model_path,
            'feature_engineering': self.feature_engineering,
            'enable_backtesting': self.enable_backtesting,
            'backtest_start_date': self.backtest_start_date,
            'backtest_end_date': self.backtest_end_date,
            'commission_rate': self.commission_rate,
            'slippage_rate': self.slippage_rate
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'StrategistConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logging.getLogger(__name__).warning(f"Unknown configuration key: {key}")
        
        # Re-validate after updates
        self._validate_config()

@dataclass
class StrategyResult:
    """Strategy execution result with comprehensive metrics."""
    
    # Basic result information
    strategy_name: str
    symbol: str
    timeframe: str
    start_time: datetime
    end_time: datetime
    
    # Performance metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    
    # Trade metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    
    # Risk metrics
    var_95: float = 0.0  # Value at Risk 95%
    cvar_95: float = 0.0  # Conditional Value at Risk 95%
    max_consecutive_losses: int = 0
    max_consecutive_wins: int = 0
    
    # Additional metrics
    total_fees: float = 0.0
    net_profit: float = 0.0
    final_capital: float = 0.0
    
    # Metadata
    config_used: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Post-initialization calculations."""
        self._calculate_derived_metrics()
    
    def _calculate_derived_metrics(self) -> None:
        """Calculate derived metrics from basic metrics."""
        # Calmar ratio
        if self.max_drawdown != 0:
            self.calmar_ratio = self.annualized_return / abs(self.max_drawdown)
        
        # Net profit
        self.net_profit = self.total_return - self.total_fees
        
        # Final capital (assuming initial capital from config)
        initial_capital = self.config_used.get('initial_capital', 10000.0)
        self.final_capital = initial_capital * (1 + self.total_return)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'strategy_name': self.strategy_name,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'calmar_ratio': self.calmar_ratio,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'profit_factor': self.profit_factor,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95,
            'max_consecutive_losses': self.max_consecutive_losses,
            'max_consecutive_wins': self.max_consecutive_wins,
            'total_fees': self.total_fees,
            'net_profit': self.net_profit,
            'final_capital': self.final_capital,
            'config_used': self.config_used,
            'execution_time': self.execution_time,
            'errors': self.errors,
            'warnings': self.warnings
        }
    
    def get_summary(self) -> str:
        """Get a summary string of the strategy result."""
        return f"""
Strategy: {self.strategy_name} ({self.symbol} {self.timeframe})
Period: {self.start_time.strftime('%Y-%m-%d')} to {self.end_time.strftime('%Y-%m-%d')}

Performance:
- Total Return: {self.total_return:.2%}
- Annualized Return: {self.annualized_return:.2%}
- Sharpe Ratio: {self.sharpe_ratio:.2f}
- Max Drawdown: {self.max_drawdown:.2%}
- Calmar Ratio: {self.calmar_ratio:.2f}

Trading:
- Total Trades: {self.total_trades}
- Win Rate: {self.win_rate:.2%}
- Profit Factor: {self.profit_factor:.2f}
- Avg Win: {self.avg_win:.2f}
- Avg Loss: {self.avg_loss:.2f}

Risk:
- VaR 95%: {self.var_95:.2%}
- CVaR 95%: {self.cvar_95:.2%}
- Max Consecutive Losses: {self.max_consecutive_losses}
- Max Consecutive Wins: {self.max_consecutive_wins}

Execution Time: {self.execution_time:.2f} seconds
        """.strip()
    
    def is_profitable(self) -> bool:
        """Check if the strategy is profitable."""
        return self.total_return > 0
    
    def is_acceptable_risk(self, max_drawdown_threshold: float = 0.2) -> bool:
        """Check if the strategy has acceptable risk levels."""
        return abs(self.max_drawdown) <= max_drawdown_threshold
    
    def get_risk_score(self) -> float:
        """Get a risk score (0-1, where 1 is highest risk)."""
        # Combine multiple risk factors
        drawdown_risk = min(abs(self.max_drawdown) / 0.5, 1.0)  # Normalize to 50% max drawdown
        volatility_risk = min(self.volatility / 0.3, 1.0)  # Normalize to 30% volatility
        consecutive_loss_risk = min(self.max_consecutive_losses / 10, 1.0)  # Normalize to 10 consecutive losses
        
        # Weighted average
        risk_score = (drawdown_risk * 0.5 + volatility_risk * 0.3 + consecutive_loss_risk * 0.2)
        return min(risk_score, 1.0)