"""
Event-Based Bar Construction for Volatility-Aware Labeling

This module implements event-based bar construction utilities that create
volatility-normalized bars for more robust profit labeling.

Key Features:
- Event-based bar construction using volume, volatility, and time triggers
- Volatility-normalized bar sizes for consistent signal quality
- Adaptive bar construction based on market conditions
- Integration with existing ML optimization utilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta

# Import existing utilities
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.common_operations import (
    safe_divide, safe_log, safe_sqrt, safe_mean, safe_std,
    validate_finite, validate_positive, validate_range
)
from src.utils.math_validation import MathValidation

# Import ML optimization utilities
try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer
    BAYESIAN_OPTIMIZER_AVAILABLE = True
except ImportError:
    BAYESIAN_OPTIMIZER_AVAILABLE = False
    tprint_warning("⚠️ Bayesian TPE optimizer not available, using grid search")


class BarTriggerType(Enum):
    """Enumeration of bar construction trigger types."""
    VOLUME = "volume"  # Volume-based triggers
    VOLATILITY = "volatility"  # Volatility-based triggers
    TIME = "time"  # Time-based triggers
    HYBRID = "hybrid"  # Combined approach


@dataclass
class BarConstructionConfig:
    """Configuration for event-based bar construction."""
    
    # Trigger settings
    trigger_type: BarTriggerType = BarTriggerType.HYBRID
    
    # Volume-based triggers
    volume_threshold: float = 1000.0  # Minimum volume for bar completion
    volume_multiplier: float = 1.5  # Volume multiplier for adaptive sizing
    
    # Volatility-based triggers
    volatility_threshold: float = 0.01  # Minimum volatility for bar completion
    volatility_multiplier: float = 2.0  # Volatility multiplier for adaptive sizing
    
    # Time-based triggers
    max_bar_duration: timedelta = timedelta(minutes=5)  # Maximum bar duration
    min_bar_duration: timedelta = timedelta(seconds=30)  # Minimum bar duration
    
    # Adaptive sizing
    enable_adaptive_sizing: bool = True
    adaptive_window: int = 20  # Window for adaptive parameter calculation
    
    # Quality checks
    min_bar_samples: int = 10
    max_price_change_ratio: float = 0.1  # Maximum price change ratio within a bar
    
    # Data-driven optimization
    enable_optimization: bool = True
    optimization_metric: str = "sharpe_ratio"  # Optimization target metric
    
    def _validate_config(self) -> None:
        """Validate bar construction configuration parameters."""
        if self.volume_threshold <= 0:
            raise ValueError("volume_threshold must be positive")
        if self.volatility_threshold <= 0:
            raise ValueError("volatility_threshold must be positive")
        if self.max_bar_duration <= self.min_bar_duration:
            raise ValueError("max_bar_duration must be greater than min_bar_duration")
        if self.min_bar_samples < 1:
            raise ValueError("min_bar_samples must be at least 1")
        if self.max_price_change_ratio <= 0:
            raise ValueError("max_price_change_ratio must be positive")


@dataclass
class BarConstructionResult:
    """Result container for bar construction."""
    
    # Core results
    constructed_bars: pd.DataFrame
    construction_metadata: Dict[str, Any]
    
    # Statistics
    total_bars: int = 0
    avg_bar_duration: float = 0.0
    avg_bar_volume: float = 0.0
    avg_bar_volatility: float = 0.0
    
    # Quality metrics
    bar_quality_score: float = 0.0
    volatility_consistency: float = 0.0
    volume_consistency: float = 0.0
    
    # Metadata
    config_used: BarConstructionConfig = None
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class EventBasedBarConstructor:
    """
    Event-Based Bar Constructor for Volatility-Aware Labeling
    
    This class implements sophisticated bar construction that creates
    volatility-normalized bars for more robust profit labeling.
    
    Key Features:
    1. **Event-Based Construction**: Bars are created based on volume, volatility, and time triggers
    2. **Volatility Normalization**: Bar sizes are normalized by volatility for consistency
    3. **Adaptive Sizing**: Bar parameters adapt to changing market conditions
    4. **Quality Validation**: Comprehensive bar quality assessment
    5. **Data-Driven Optimization**: Parameters optimized using historical data
    """
    
    def __init__(self, config: Optional[BarConstructionConfig] = None):
        """Initialize event-based bar constructor."""
        self.config = config or BarConstructionConfig()
        self.logger = logging.getLogger('EventBasedBarConstructor')
        
        # Validate configuration
        self.config._validate_config()
        
        # Initialize optimization if available
        if BAYESIAN_OPTIMIZER_AVAILABLE and self.config.enable_optimization:
            self.optimizer = BayesianTPEOptimizer()
            tprint_info("   → Bayesian optimization: Available")
        else:
            self.optimizer = None
            tprint_warning("   → Bayesian optimization: Not available, using fixed parameters")
        
        tprint_info("📊 Event-Based Bar Constructor initialized")
        tprint_info(f"   → Trigger type: {self.config.trigger_type.value}")
        tprint_info(f"   → Volume threshold: {self.config.volume_threshold}")
        tprint_info(f"   → Volatility threshold: {self.config.volatility_threshold}")
        tprint_info(f"   → Adaptive sizing: {self.config.enable_adaptive_sizing}")
    
    def construct_bars(self, tick_data: pd.DataFrame) -> BarConstructionResult:
        """
        Construct event-based bars from tick data.
        
        Args:
            tick_data: Tick data with OHLCV and timestamp columns
            
        Returns:
            BarConstructionResult with constructed bars and metadata
        """
        start_time = datetime.now()
        tprint_info("📊 Constructing event-based bars")
        
        # Initialize result container
        result = BarConstructionResult(
            constructed_bars=pd.DataFrame(),
            construction_metadata={},
            config_used=self.config
        )
        
        try:
            # Validate input data
            if not self._validate_input_data(tick_data):
                return result
            
            # Optimize parameters if enabled
            if self.config.enable_optimization and self.optimizer:
                tprint_info("🔧 Step 1: Optimizing bar construction parameters")
                optimized_config = self._optimize_parameters(tick_data)
                self.config = optimized_config
            
            # Calculate adaptive parameters
            if self.config.enable_adaptive_sizing:
                tprint_info("📈 Step 2: Calculating adaptive parameters")
                adaptive_params = self._calculate_adaptive_parameters(tick_data)
            else:
                adaptive_params = self._get_default_parameters()
            
            # Construct bars based on trigger type
            tprint_info("🔨 Step 3: Constructing bars")
            if self.config.trigger_type == BarTriggerType.VOLUME:
                bars = self._construct_volume_based_bars(tick_data, adaptive_params)
            elif self.config.trigger_type == BarTriggerType.VOLATILITY:
                bars = self._construct_volatility_based_bars(tick_data, adaptive_params)
            elif self.config.trigger_type == BarTriggerType.TIME:
                bars = self._construct_time_based_bars(tick_data, adaptive_params)
            else:  # HYBRID
                bars = self._construct_hybrid_bars(tick_data, adaptive_params)
            
            result.constructed_bars = bars
            
            # Calculate statistics and quality metrics
            tprint_info("📊 Step 4: Calculating statistics and quality metrics")
            stats = self._calculate_bar_statistics(bars)
            result.total_bars = stats['total_bars']
            result.avg_bar_duration = stats['avg_bar_duration']
            result.avg_bar_volume = stats['avg_bar_volume']
            result.avg_bar_volatility = stats['avg_bar_volatility']
            
            quality_metrics = self._calculate_bar_quality(bars)
            result.bar_quality_score = quality_metrics['quality_score']
            result.volatility_consistency = quality_metrics['volatility_consistency']
            result.volume_consistency = quality_metrics['volume_consistency']
            
            # Store construction metadata
            result.construction_metadata = {
                'trigger_type': self.config.trigger_type.value,
                'adaptive_params': adaptive_params,
                'optimization_enabled': self.config.enable_optimization,
                'bars_constructed': len(bars)
            }
            
        except Exception as e:
            tprint_error(f"❌ Bar construction failed: {e}")
            return result
        
        # Calculate processing time
        result.processing_time = (datetime.now() - start_time).total_seconds()
        
        tprint_success("✅ Bar construction completed")
        tprint_info(f"   → Bars constructed: {result.total_bars}")
        tprint_info(f"   → Avg duration: {result.avg_bar_duration:.2f}s")
        tprint_info(f"   → Quality score: {result.bar_quality_score:.3f}")
        
        return result
    
    def _validate_input_data(self, tick_data: pd.DataFrame) -> bool:
        """Validate input tick data."""
        try:
            # Check if DataFrame is empty
            if tick_data.empty:
                tprint_warning("⚠️ Input tick data is empty")
                return False
            
            # Check required columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = set(required_columns) - set(tick_data.columns)
            if missing_columns:
                tprint_warning(f"⚠️ Missing required columns: {missing_columns}")
                return False
            
            # Check minimum samples
            if len(tick_data) < self.config.min_bar_samples:
                tprint_warning(f"⚠️ Insufficient samples: {len(tick_data)} < {self.config.min_bar_samples}")
                return False
            
            # Check for non-finite values
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            if tick_data[numeric_columns].isnull().any().any():
                tprint_warning("⚠️ Data contains null values")
                return False
            
            if not np.isfinite(tick_data[numeric_columns].values).all():
                tprint_warning("⚠️ Data contains non-finite values")
                return False
            
            return True
            
        except Exception as e:
            tprint_error(f"❌ Data validation failed: {e}")
            return False
    
    def _optimize_parameters(self, tick_data: pd.DataFrame) -> BarConstructionConfig:
        """Optimize bar construction parameters using historical data."""
        try:
            if not self.optimizer:
                return self.config
            
            # Define parameter space for optimization
            param_space = {
                'volume_threshold': (100.0, 10000.0),
                'volatility_threshold': (0.001, 0.05),
                'volume_multiplier': (1.0, 3.0),
                'volatility_multiplier': (1.0, 4.0)
            }
            
            # Define objective function
            def objective(params):
                # Create temporary config with optimized parameters
                temp_config = BarConstructionConfig(
                    trigger_type=self.config.trigger_type,
                    volume_threshold=params['volume_threshold'],
                    volatility_threshold=params['volatility_threshold'],
                    volume_multiplier=params['volume_multiplier'],
                    volatility_multiplier=params['volatility_multiplier'],
                    max_bar_duration=self.config.max_bar_duration,
                    min_bar_duration=self.config.min_bar_duration,
                    enable_adaptive_sizing=self.config.enable_adaptive_sizing,
                    adaptive_window=self.config.adaptive_window,
                    min_bar_samples=self.config.min_bar_samples,
                    max_price_change_ratio=self.config.max_price_change_ratio,
                    enable_optimization=False  # Prevent recursive optimization
                )
                
                # Create temporary constructor
                temp_constructor = EventBasedBarConstructor(temp_config)
                
                # Construct bars and evaluate quality
                result = temp_constructor.construct_bars(tick_data)
                
                # Return quality score (higher is better)
                return result.bar_quality_score
            
            # Run optimization
            best_params = self.optimizer.optimize(
                objective_function=objective,
                param_space=param_space,
                n_trials=50,
                random_state=42
            )
            
            # Update config with optimized parameters
            optimized_config = BarConstructionConfig(
                trigger_type=self.config.trigger_type,
                volume_threshold=best_params['volume_threshold'],
                volatility_threshold=best_params['volatility_threshold'],
                volume_multiplier=best_params['volume_multiplier'],
                volatility_multiplier=best_params['volatility_multiplier'],
                max_bar_duration=self.config.max_bar_duration,
                min_bar_duration=self.config.min_bar_duration,
                enable_adaptive_sizing=self.config.enable_adaptive_sizing,
                adaptive_window=self.config.adaptive_window,
                min_bar_samples=self.config.min_bar_samples,
                max_price_change_ratio=self.config.max_price_change_ratio,
                enable_optimization=False
            )
            
            tprint_success("✅ Parameter optimization completed")
            return optimized_config
            
        except Exception as e:
            tprint_warning(f"⚠️ Parameter optimization failed: {e}")
            return self.config
    
    def _calculate_adaptive_parameters(self, tick_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate adaptive parameters based on historical data."""
        try:
            # Calculate rolling statistics
            window = min(self.config.adaptive_window, len(tick_data) // 2)
            
            # Volume statistics
            volume_mean = tick_data['volume'].rolling(window=window).mean().iloc[-1]
            volume_std = tick_data['volume'].rolling(window=window).std().iloc[-1]
            
            # Volatility statistics
            returns = tick_data['close'].pct_change().dropna()
            volatility_mean = returns.rolling(window=window).std().iloc[-1]
            volatility_std = returns.rolling(window=window).std().std()
            
            # Adaptive thresholds
            adaptive_volume_threshold = max(
                volume_mean + 0.5 * volume_std,
                self.config.volume_threshold
            )
            
            adaptive_volatility_threshold = max(
                volatility_mean + 0.5 * volatility_std,
                self.config.volatility_threshold
            )
            
            # Adaptive multipliers based on market conditions
            volume_multiplier = 1.0 + (volume_std / volume_mean) if volume_mean > 0 else 1.0
            volatility_multiplier = 1.0 + (volatility_std / volatility_mean) if volatility_mean > 0 else 1.0
            
            return {
                'volume_threshold': adaptive_volume_threshold,
                'volatility_threshold': adaptive_volatility_threshold,
                'volume_multiplier': min(volume_multiplier, 3.0),  # Cap at 3.0
                'volatility_multiplier': min(volatility_multiplier, 4.0)  # Cap at 4.0
            }
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating adaptive parameters: {e}")
            return self._get_default_parameters()
    
    def _get_default_parameters(self) -> Dict[str, float]:
        """Get default parameters when adaptive calculation fails."""
        return {
            'volume_threshold': self.config.volume_threshold,
            'volatility_threshold': self.config.volatility_threshold,
            'volume_multiplier': self.config.volume_multiplier,
            'volatility_multiplier': self.config.volatility_multiplier
        }
    
    def _construct_volume_based_bars(self, tick_data: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
        """Construct bars based on volume triggers."""
        try:
            bars = []
            current_bar = None
            cumulative_volume = 0.0
            
            for idx, row in tick_data.iterrows():
                if current_bar is None:
                    # Start new bar
                    current_bar = {
                        'timestamp': row['timestamp'],
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close'],
                        'volume': row['volume']
                    }
                    cumulative_volume = row['volume']
                else:
                    # Update current bar
                    current_bar['high'] = max(current_bar['high'], row['high'])
                    current_bar['low'] = min(current_bar['low'], row['low'])
                    current_bar['close'] = row['close']
                    current_bar['volume'] += row['volume']
                    cumulative_volume += row['volume']
                    
                    # Check if volume threshold is reached
                    if cumulative_volume >= params['volume_threshold'] * params['volume_multiplier']:
                        bars.append(current_bar)
                        current_bar = None
                        cumulative_volume = 0.0
            
            # Add final bar if exists
            if current_bar is not None:
                bars.append(current_bar)
            
            return pd.DataFrame(bars)
            
        except Exception as e:
            tprint_warning(f"⚠️ Error constructing volume-based bars: {e}")
            return pd.DataFrame()
    
    def _construct_volatility_based_bars(self, tick_data: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
        """Construct bars based on volatility triggers."""
        try:
            bars = []
            current_bar = None
            bar_returns = []
            
            for idx, row in tick_data.iterrows():
                if current_bar is None:
                    # Start new bar
                    current_bar = {
                        'timestamp': row['timestamp'],
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close'],
                        'volume': row['volume']
                    }
                    bar_returns = []
                else:
                    # Update current bar
                    current_bar['high'] = max(current_bar['high'], row['high'])
                    current_bar['low'] = min(current_bar['low'], row['low'])
                    current_bar['close'] = row['close']
                    current_bar['volume'] += row['volume']
                    
                    # Calculate return
                    return_val = (row['close'] - current_bar['open']) / current_bar['open']
                    bar_returns.append(return_val)
                    
                    # Check if volatility threshold is reached
                    if len(bar_returns) > 1:
                        volatility = np.std(bar_returns)
                        if volatility >= params['volatility_threshold'] * params['volatility_multiplier']:
                            bars.append(current_bar)
                            current_bar = None
                            bar_returns = []
            
            # Add final bar if exists
            if current_bar is not None:
                bars.append(current_bar)
            
            return pd.DataFrame(bars)
            
        except Exception as e:
            tprint_warning(f"⚠️ Error constructing volatility-based bars: {e}")
            return pd.DataFrame()
    
    def _construct_time_based_bars(self, tick_data: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
        """Construct bars based on time triggers."""
        try:
            bars = []
            current_bar = None
            bar_start_time = None
            
            for idx, row in tick_data.iterrows():
                if current_bar is None:
                    # Start new bar
                    current_bar = {
                        'timestamp': row['timestamp'],
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close'],
                        'volume': row['volume']
                    }
                    bar_start_time = row['timestamp']
                else:
                    # Update current bar
                    current_bar['high'] = max(current_bar['high'], row['high'])
                    current_bar['low'] = min(current_bar['low'], row['low'])
                    current_bar['close'] = row['close']
                    current_bar['volume'] += row['volume']
                    
                    # Check if time threshold is reached
                    bar_duration = row['timestamp'] - bar_start_time
                    if bar_duration >= self.config.max_bar_duration:
                        bars.append(current_bar)
                        current_bar = None
                        bar_start_time = None
            
            # Add final bar if exists
            if current_bar is not None:
                bars.append(current_bar)
            
            return pd.DataFrame(bars)
            
        except Exception as e:
            tprint_warning(f"⚠️ Error constructing time-based bars: {e}")
            return pd.DataFrame()
    
    def _construct_hybrid_bars(self, tick_data: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
        """Construct bars using hybrid approach (volume + volatility + time)."""
        try:
            bars = []
            current_bar = None
            cumulative_volume = 0.0
            bar_returns = []
            bar_start_time = None
            
            for idx, row in tick_data.iterrows():
                if current_bar is None:
                    # Start new bar
                    current_bar = {
                        'timestamp': row['timestamp'],
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close'],
                        'volume': row['volume']
                    }
                    cumulative_volume = row['volume']
                    bar_returns = []
                    bar_start_time = row['timestamp']
                else:
                    # Update current bar
                    current_bar['high'] = max(current_bar['high'], row['high'])
                    current_bar['low'] = min(current_bar['low'], row['low'])
                    current_bar['close'] = row['close']
                    current_bar['volume'] += row['volume']
                    
                    # Update tracking variables
                    cumulative_volume += row['volume']
                    return_val = (row['close'] - current_bar['open']) / current_bar['open']
                    bar_returns.append(return_val)
                    
                    # Check multiple triggers
                    volume_trigger = cumulative_volume >= params['volume_threshold'] * params['volume_multiplier']
                    volatility_trigger = len(bar_returns) > 1 and np.std(bar_returns) >= params['volatility_threshold'] * params['volatility_multiplier']
                    time_trigger = (row['timestamp'] - bar_start_time) >= self.config.max_bar_duration
                    
                    # Complete bar if any trigger is met
                    if volume_trigger or volatility_trigger or time_trigger:
                        bars.append(current_bar)
                        current_bar = None
                        cumulative_volume = 0.0
                        bar_returns = []
                        bar_start_time = None
            
            # Add final bar if exists
            if current_bar is not None:
                bars.append(current_bar)
            
            return pd.DataFrame(bars)
            
        except Exception as e:
            tprint_warning(f"⚠️ Error constructing hybrid bars: {e}")
            return pd.DataFrame()
    
    def _calculate_bar_statistics(self, bars: pd.DataFrame) -> Dict[str, Any]:
        """Calculate bar construction statistics."""
        try:
            if bars.empty:
                return {
                    'total_bars': 0,
                    'avg_bar_duration': 0.0,
                    'avg_bar_volume': 0.0,
                    'avg_bar_volatility': 0.0
                }
            
            # Basic statistics
            total_bars = len(bars)
            
            # Duration statistics
            if 'timestamp' in bars.columns and len(bars) > 1:
                durations = bars['timestamp'].diff().dt.total_seconds().dropna()
                avg_duration = durations.mean() if not durations.empty else 0.0
            else:
                avg_duration = 0.0
            
            # Volume statistics
            avg_volume = bars['volume'].mean() if 'volume' in bars.columns else 0.0
            
            # Volatility statistics
            if 'open' in bars.columns and 'close' in bars.columns:
                returns = (bars['close'] - bars['open']) / bars['open']
                avg_volatility = returns.std() if not returns.empty else 0.0
            else:
                avg_volatility = 0.0
            
            return {
                'total_bars': total_bars,
                'avg_bar_duration': avg_duration,
                'avg_bar_volume': avg_volume,
                'avg_bar_volatility': avg_volatility
            }
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating bar statistics: {e}")
            return {
                'total_bars': 0,
                'avg_bar_duration': 0.0,
                'avg_bar_volume': 0.0,
                'avg_bar_volatility': 0.0
            }
    
    def _calculate_bar_quality(self, bars: pd.DataFrame) -> Dict[str, float]:
        """Calculate bar quality metrics."""
        try:
            if bars.empty:
                return {
                    'quality_score': 0.0,
                    'volatility_consistency': 0.0,
                    'volume_consistency': 0.0
                }
            
            # Quality score based on multiple factors
            quality_factors = []
            
            # Volume consistency
            if 'volume' in bars.columns and len(bars) > 1:
                volume_cv = bars['volume'].std() / bars['volume'].mean() if bars['volume'].mean() > 0 else 1.0
                volume_consistency = max(0.0, 1.0 - volume_cv)
                quality_factors.append(volume_consistency)
            else:
                volume_consistency = 0.0
                quality_factors.append(0.0)
            
            # Volatility consistency
            if 'open' in bars.columns and 'close' in bars.columns and len(bars) > 1:
                returns = (bars['close'] - bars['open']) / bars['open']
                volatility_cv = returns.std() / returns.mean() if returns.mean() > 0 else 1.0
                volatility_consistency = max(0.0, 1.0 - volatility_cv)
                quality_factors.append(volatility_consistency)
            else:
                volatility_consistency = 0.0
                quality_factors.append(0.0)
            
            # Price consistency (high-low range)
            if 'high' in bars.columns and 'low' in bars.columns and 'open' in bars.columns:
                price_ranges = (bars['high'] - bars['low']) / bars['open']
                range_cv = price_ranges.std() / price_ranges.mean() if price_ranges.mean() > 0 else 1.0
                price_consistency = max(0.0, 1.0 - range_cv)
                quality_factors.append(price_consistency)
            else:
                quality_factors.append(0.0)
            
            # Overall quality score
            quality_score = np.mean(quality_factors) if quality_factors else 0.0
            
            return {
                'quality_score': quality_score,
                'volatility_consistency': volatility_consistency,
                'volume_consistency': volume_consistency
            }
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating bar quality: {e}")
            return {
                'quality_score': 0.0,
                'volatility_consistency': 0.0,
                'volume_consistency': 0.0
            }


# Convenience functions
def create_bar_construction_manager(config: Optional[BarConstructionConfig] = None) -> EventBasedBarConstructor:
    """Create bar construction manager with specified configuration."""
    return EventBasedBarConstructor(config)


def construct_bars(tick_data: pd.DataFrame,
                  config: Optional[BarConstructionConfig] = None) -> BarConstructionResult:
    """Construct bars with default configuration."""
    constructor = EventBasedBarConstructor(config)
    return constructor.construct_bars(tick_data)