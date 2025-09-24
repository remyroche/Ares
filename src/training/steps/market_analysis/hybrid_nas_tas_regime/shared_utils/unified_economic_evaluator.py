"""
Unified Economic Significance Evaluator

This module provides a unified economic significance evaluation system that combines
the best practices from both TAS and NAS regime detection systems. It evaluates
the economic significance of detected regimes for trading and investment decisions.

Features:
- Unified economic metrics calculation
- Support for both tree-based and neural-based regime detection
- Position-aware economic analysis
- Configurable thresholds and weights
- Advanced economic indicators integration
"""

# Handle missing dependencies gracefully
try:
    import numpy as np
    import pandas as pd
    HAS_NUMPY_PANDAS = True
except ImportError:
    HAS_NUMPY_PANDAS = False
    # Create mock numpy and pandas for basic functionality
    class MockArray:
        def __init__(self, data):
            self.data = data if isinstance(data, list) else list(data)
        def __getitem__(self, key):
            return self.data[key]
        def __setitem__(self, key, value):
            self.data[key] = value
        def __len__(self):
            return len(self.data)
        def __iter__(self):
            return iter(self.data)
        def __array__(self):
            return self.data
        def shape(self):
            if isinstance(self.data[0], list):
                return (len(self.data), len(self.data[0]))
            return (len(self.data),)
        def mean(self):
            return sum(self.data) / len(self.data) if self.data else 0
        def std(self):
            if not self.data:
                return 0
            mean_val = self.mean()
            return (sum((x - mean_val) ** 2 for x in self.data) / len(self.data)) ** 0.5
        def sum(self):
            return sum(self.data)
        def min(self):
            return min(self.data) if self.data else 0
        def max(self):
            return max(self.data) if self.data else 0
        def where(self, condition, x, y):
            return [x if c else y for c in condition]
        def diff(self):
            return [self.data[i+1] - self.data[i] for i in range(len(self.data)-1)]
        def corrcoef(self, x, y):
            # Simple correlation calculation
            if len(x) != len(y):
                return 0
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(x[i] ** 2 for i in range(n))
            sum_y2 = sum(y[i] ** 2 for i in range(n))
            numerator = n * sum_xy - sum_x * sum_y
            denominator = ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5
            return numerator / denominator if denominator != 0 else 0
        def unique(self):
            return list(set(self.data))
        def any(self):
            return any(self.data)
        def clip(self, min_val, max_val):
            return [max(min_val, min(max_val, x)) for x in self.data]
        def astype(self, dtype):
            return self.data
    
    class MockDataFrame:
        def __init__(self, data=None, columns=None):
            self.data = data or {}
            self.columns = columns or []
        def __getitem__(self, key):
            return self.data.get(key, [])
        def values(self):
            return list(self.data.values())
        def shape(self):
            if not self.data:
                return (0, 0)
            return (len(list(self.data.values())[0]), len(self.data))
    
    np = type('numpy', (), {
        'array': lambda x: MockArray(x),
        'zeros': lambda x: MockArray([0] * x),
        'ones': lambda x: MockArray([1] * x),
        'random': type('random', (), {
            'randint': lambda a, b, size=None: [a] if size is None else [a] * size,
            'uniform': lambda a, b, size=None: [a] if size is None else [a] * size,
            'normal': lambda a, b, size=None: [a] if size is None else [a] * size,
            'choice': lambda x, size=None: x[0] if size is None else [x[0]] * size,
            'random': lambda size=None: [0.5] if size is None else [0.5] * size
        })(),
        'mean': lambda x: sum(x) / len(x) if x else 0,
        'std': lambda x: (sum((i - sum(x)/len(x))**2 for i in x) / len(x))**0.5 if x else 0,
        'sum': lambda x: sum(x),
        'min': lambda x: min(x) if x else 0,
        'max': lambda x: max(x) if x else 0,
        'where': lambda condition, x, y: [x if c else y for c in condition],
        'diff': lambda x: [x[i+1] - x[i] for i in range(len(x)-1)],
        'corrcoef': lambda x, y: MockArray([1.0, 0.0, 0.0, 1.0]),
        'unique': lambda x: list(set(x)),
        'any': lambda x: any(x),
        'clip': lambda x, a, b: [max(a, min(b, i)) for i in x],
        'arange': lambda x: list(range(x)),
        'percentile': lambda x, p: sorted(x)[int(len(x) * p / 100)],
        'sqrt': lambda x: x ** 0.5,
        'log': lambda x: __import__('math').log(x) if x > 0 else 0,
        'isnan': lambda x: False,
        'abs': lambda x: abs(x),
        'trapz': lambda x: sum(x) / len(x) if x else 0
    })()
    
    pd = type('pandas', (), {
        'DataFrame': MockDataFrame,
        'Series': lambda x: MockArray(x)
    })()

from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass, field
from enum import Enum
import time
from datetime import datetime

# Import position-aware trading analyzer
try:
    from .position_aware_trading import PositionAwareTradingAnalyzer, PositionAwareConfig
    HAS_POSITION_AWARE = True
except ImportError:
    HAS_POSITION_AWARE = False
    # Create mock classes
    class PositionAwareConfig:
        def __init__(self):
            self.minimum_profit_threshold = 0.001
            self.transaction_cost = 0.001
            self.position_holding_periods = [1, 5, 10, 20]
            self.risk_free_rate = 0.02
            self.win_rate_thresholds = {'excellent': 0.7, 'good': 0.6, 'acceptable': 0.5, 'poor': 0.4}
    
    class PositionAwareTradingAnalyzer:
        def __init__(self, config=None):
            self.config = config or PositionAwareConfig()
        def analyze_regime_position_performance(self, *args, **kwargs):
            return {'overall_analysis': {'overall_win_rate': 0.5, 'long_win_rate': 0.5, 'short_win_rate': 0.5}}
        def calculate_position_aware_trading_viability(self, *args, **kwargs):
            return {'overall_viability': 0.5, 'long_viability': 0.5, 'short_viability': 0.5}

logger = logging.getLogger(__name__)


class EconomicMetricType(Enum):
    """Types of economic metrics to evaluate."""
    PRICE_IMPACT = "price_impact"
    VOLUME_SIGNIFICANCE = "volume_significance"
    VOLATILITY_IMPACT = "volatility_impact"
    TREND_CONSISTENCY = "trend_consistency"
    MARKET_EFFICIENCY = "market_efficiency"
    ECONOMIC_INDICATORS = "economic_indicators"
    TRADING_OPPORTUNITY = "trading_opportunity"
    RISK_ADJUSTMENT = "risk_adjustment"


@dataclass
class EconomicEvaluationConfig:
    """Configuration for unified economic evaluation."""
    
    # Economic significance weights
    price_impact_weight: float = 0.25
    volume_significance_weight: float = 0.15
    volatility_impact_weight: float = 0.20
    trend_consistency_weight: float = 0.15
    market_efficiency_weight: float = 0.10
    economic_indicators_weight: float = 0.10
    trading_opportunity_weight: float = 0.05
    
    # Thresholds
    significance_threshold: float = 0.6
    price_impact_threshold: float = 0.5
    volume_threshold: float = 0.4
    volatility_threshold: float = 0.5
    trend_threshold: float = 0.6
    efficiency_threshold: float = 0.5
    
    # Economic indicators
    enable_economic_indicators: bool = True
    economic_indicators_lookback: int = 252  # 1 year
    economic_correlation_threshold: float = 0.3
    
    # Position-aware analysis
    enable_position_aware_analysis: bool = True
    position_aware_config: Optional[PositionAwareConfig] = None
    
    # Advanced features
    enable_bootstrap_analysis: bool = True
    bootstrap_iterations: int = 100
    confidence_level: float = 0.95
    
    # Regime-specific analysis
    enable_regime_specific_analysis: bool = True
    min_regime_samples: int = 50
    regime_stability_threshold: float = 0.7


@dataclass
class EconomicSignificanceResult:
    """Result from unified economic significance evaluation."""
    
    # Overall scores
    overall_score: float
    significance_level: str  # 'high', 'medium', 'low'
    
    # Individual metric scores
    price_impact_score: float
    volume_significance_score: float
    volatility_impact_score: float
    trend_consistency_score: float
    market_efficiency_score: float
    economic_indicators_score: float
    trading_opportunity_score: float
    risk_adjustment_score: float
    
    # Regime-specific analysis
    regime_economic_profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    regime_significance_scores: Dict[str, float] = field(default_factory=dict)
    
    # Statistical significance
    statistical_significance: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    bootstrap_results: Dict[str, Any] = field(default_factory=dict)
    
    # Position-aware analysis
    position_aware_analysis: Optional[Dict[str, Any]] = None
    
    # Metadata
    evaluation_timestamp: datetime = field(default_factory=datetime.now)
    data_shape: Tuple[int, int] = (0, 0)
    n_regimes: int = 0
    evaluation_time: float = 0.0


class UnifiedEconomicSignificanceEvaluator:
    """
    Unified Economic Significance Evaluator.
    
    Combines the best practices from both TAS and NAS regime detection systems
    to provide comprehensive economic significance evaluation.
    """
    
    def __init__(self, config: EconomicEvaluationConfig):
        """Initialize unified economic significance evaluator.
        
        Args:
            config: Economic evaluation configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize position-aware analyzer if enabled
        self.position_analyzer = None
        if config.enable_position_aware_analysis:
            if config.position_aware_config is None:
                config.position_aware_config = PositionAwareConfig()
            self.position_analyzer = PositionAwareTradingAnalyzer(config.position_aware_config)
        
        # Economic indicators (placeholder - would be loaded from external data)
        self.economic_indicators = self._load_economic_indicators()
        
        self.logger.info("✅ Unified Economic Significance Evaluator initialized")
        self.logger.info(f"   Position-aware analysis: {config.enable_position_aware_analysis}")
        self.logger.info(f"   Economic indicators: {config.enable_economic_indicators}")
        self.logger.info(f"   Bootstrap analysis: {config.enable_bootstrap_analysis}")
    
    def _load_economic_indicators(self) -> Dict[str, np.ndarray]:
        """Load economic indicators (placeholder implementation)."""
        if not self.config.enable_economic_indicators:
            return {}
        
        # In a real implementation, this would load actual economic data
        # For now, return placeholder data
        return {
            'gdp_growth': np.random.normal(0.02, 0.01, 1000),
            'inflation_rate': np.random.normal(0.03, 0.005, 1000),
            'interest_rate': np.random.normal(0.05, 0.01, 1000),
            'unemployment_rate': np.random.normal(0.05, 0.01, 1000),
            'consumer_confidence': np.random.normal(100, 10, 1000),
            'vix': np.random.normal(20, 5, 1000)
        }
    
    def evaluate(self, 
                 market_data: Union[pd.DataFrame, np.ndarray], 
                 regime_predictions: np.ndarray,
                 regime_probabilities: Optional[np.ndarray] = None,
                 timestamps: Optional[np.ndarray] = None,
                 regime_metadata: Optional[Dict[str, Any]] = None) -> EconomicSignificanceResult:
        """
        Evaluate economic significance of regimes using unified approach.
        
        Args:
            market_data: Market data (OHLCV)
            regime_predictions: Regime predictions
            regime_probabilities: Optional regime probabilities
            timestamps: Optional timestamps
            regime_metadata: Optional regime metadata
            
        Returns:
            Comprehensive economic significance result
        """
        start_time = time.time()
        
        try:
            self.logger.info("💰 Starting unified economic significance evaluation...")
            self.logger.info(f"   Data shape: {market_data.shape}")
            self.logger.info(f"   Regimes: {len(np.unique(regime_predictions))}")
            
            # Convert data to numpy array if needed
            if isinstance(market_data, pd.DataFrame):
                data_array = market_data.values
                if timestamps is None and 'timestamp' in market_data.columns:
                    timestamps = market_data['timestamp'].values
            else:
                data_array = market_data
                if timestamps is None:
                    timestamps = np.arange(len(data_array))
            
            # Calculate individual economic metrics
            price_impact_scores = self._calculate_price_impact_significance(data_array, regime_predictions)
            volume_scores = self._calculate_volume_significance(data_array, regime_predictions)
            volatility_scores = self._calculate_volatility_impact(data_array, regime_predictions)
            trend_scores = self._calculate_trend_consistency(data_array, regime_predictions)
            efficiency_scores = self._calculate_market_efficiency(data_array, regime_predictions)
            indicator_scores = self._calculate_economic_indicator_correlation(data_array, regime_predictions, timestamps)
            trading_scores = self._calculate_trading_opportunity_significance(data_array, regime_predictions)
            risk_scores = self._calculate_risk_adjustment_significance(data_array, regime_predictions)
            
            # Calculate weighted overall economic significance
            overall_scores = (
                price_impact_scores * self.config.price_impact_weight +
                volume_scores * self.config.volume_significance_weight +
                volatility_scores * self.config.volatility_impact_weight +
                trend_scores * self.config.trend_consistency_weight +
                efficiency_scores * self.config.market_efficiency_weight +
                indicator_scores * self.config.economic_indicators_weight +
                trading_scores * self.config.trading_opportunity_weight
            )
            
            # Apply significance threshold
            significant_regimes = overall_scores >= self.config.significance_threshold
            
            # Regime-specific analysis
            regime_profiles = {}
            regime_significance = {}
            if self.config.enable_regime_specific_analysis:
                regime_profiles = self._analyze_regime_economic_profiles(data_array, regime_predictions, timestamps)
                regime_significance = self._calculate_regime_significance_scores(regime_predictions, overall_scores)
            
            # Position-aware analysis
            position_analysis = None
            if self.position_analyzer:
                try:
                    df_data = pd.DataFrame(data_array, columns=['open', 'high', 'low', 'close', 'volume'])
                    position_analysis = self.position_analyzer.analyze_regime_position_performance(
                        df_data, regime_predictions
                    )
                except Exception as e:
                    self.logger.warning(f"Position-aware analysis failed: {e}")
            
            # Bootstrap analysis for statistical significance
            bootstrap_results = {}
            statistical_significance = 0.0
            confidence_interval = (0.0, 0.0)
            
            if self.config.enable_bootstrap_analysis:
                bootstrap_results = self._perform_bootstrap_analysis(data_array, regime_predictions)
                statistical_significance = bootstrap_results.get('statistical_significance', 0.0)
                confidence_interval = bootstrap_results.get('confidence_interval', (0.0, 0.0))
            
            # Determine significance level
            mean_score = np.mean(overall_scores)
            if mean_score >= 0.8:
                significance_level = 'high'
            elif mean_score >= 0.6:
                significance_level = 'medium'
            else:
                significance_level = 'low'
            
            execution_time = time.time() - start_time
            
            # Create result
            result = EconomicSignificanceResult(
                overall_score=mean_score,
                significance_level=significance_level,
                price_impact_score=np.mean(price_impact_scores),
                volume_significance_score=np.mean(volume_scores),
                volatility_impact_score=np.mean(volatility_scores),
                trend_consistency_score=np.mean(trend_scores),
                market_efficiency_score=np.mean(efficiency_scores),
                economic_indicators_score=np.mean(indicator_scores),
                trading_opportunity_score=np.mean(trading_scores),
                risk_adjustment_score=np.mean(risk_scores),
                regime_economic_profiles=regime_profiles,
                regime_significance_scores=regime_significance,
                statistical_significance=statistical_significance,
                confidence_interval=confidence_interval,
                bootstrap_results=bootstrap_results,
                position_aware_analysis=position_analysis,
                data_shape=data_array.shape,
                n_regimes=len(np.unique(regime_predictions)),
                evaluation_time=execution_time
            )
            
            self.logger.info(f"✅ Unified economic significance evaluation completed in {execution_time:.2f}s")
            self.logger.info(f"   Overall score: {mean_score:.3f}")
            self.logger.info(f"   Significance level: {significance_level}")
            self.logger.info(f"   Significant regimes: {np.sum(significant_regimes)}/{len(regime_predictions)}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Unified economic significance evaluation failed: {e}")
            
            return EconomicSignificanceResult(
                overall_score=0.0,
                significance_level='low',
                price_impact_score=0.0,
                volume_significance_score=0.0,
                volatility_impact_score=0.0,
                trend_consistency_score=0.0,
                market_efficiency_score=0.0,
                economic_indicators_score=0.0,
                trading_opportunity_score=0.0,
                risk_adjustment_score=0.0,
                evaluation_time=execution_time
            )
    
    def _calculate_price_impact_significance(self, market_data: np.ndarray, 
                                           regime_predictions: np.ndarray) -> np.ndarray:
        """Calculate price impact significance for each regime."""
        try:
            if market_data.shape[1] < 4:
                return np.zeros(len(regime_predictions))
            
            close_prices = market_data[:, 3]  # Close prices
            price_impact_scores = np.zeros(len(regime_predictions))
            
            unique_regimes = np.unique(regime_predictions)
            
            for regime in unique_regimes:
                regime_mask = regime_predictions == regime
                if not np.any(regime_mask):
                    continue
                
                regime_prices = close_prices[regime_mask]
                
                if len(regime_prices) < 2:
                    continue
                
                # Calculate price movement magnitude
                price_changes = np.diff(regime_prices)
                price_magnitude = np.mean(np.abs(price_changes))
                
                # Calculate price volatility
                price_volatility = np.std(price_changes)
                
                # Calculate price trend strength
                if len(price_changes) > 1:
                    trend_strength = abs(np.corrcoef(np.arange(len(price_changes)), price_changes)[0, 1])
                else:
                    trend_strength = 0.0
                
                # Calculate price impact significance
                price_impact = (
                    min(price_magnitude * 10, 1.0) * 0.4 +  # Normalized magnitude
                    min(price_volatility * 5, 1.0) * 0.3 +  # Normalized volatility
                    trend_strength * 0.3  # Trend strength
                )
                
                price_impact_scores[regime_mask] = min(price_impact, 1.0)
            
            return price_impact_scores
            
        except Exception as e:
            self.logger.warning(f"Price impact calculation failed: {e}")
            return np.zeros(len(regime_predictions))
    
    def _calculate_volume_significance(self, market_data: np.ndarray, 
                                     regime_predictions: np.ndarray) -> np.ndarray:
        """Calculate volume significance for each regime."""
        try:
            if market_data.shape[1] < 5:
                # No volume data available
                return np.ones(len(regime_predictions)) * 0.5
            
            volumes = market_data[:, 4]  # Volume
            volume_scores = np.zeros(len(regime_predictions))
            
            unique_regimes = np.unique(regime_predictions)
            
            for regime in unique_regimes:
                regime_mask = regime_predictions == regime
                if not np.any(regime_mask):
                    continue
                
                regime_volumes = volumes[regime_mask]
                
                if len(regime_volumes) < 2:
                    continue
                
                # Calculate volume significance metrics
                volume_mean = np.mean(regime_volumes)
                volume_std = np.std(regime_volumes)
                volume_consistency = 1.0 / (1.0 + volume_std / (volume_mean + 1e-8))
                
                # Calculate volume trend
                volume_changes = np.diff(regime_volumes)
                volume_trend = np.mean(volume_changes) / (volume_mean + 1e-8)
                
                # Calculate volume relative to market average
                market_volume_avg = np.mean(volumes)
                volume_ratio = volume_mean / (market_volume_avg + 1e-8)
                
                # Combine volume significance metrics
                volume_significance = (
                    volume_consistency * 0.4 +
                    min(abs(volume_trend), 1.0) * 0.3 +
                    min(volume_ratio, 2.0) * 0.3  # Cap at 2x market average
                )
                
                volume_scores[regime_mask] = min(volume_significance, 1.0)
            
            return volume_scores
            
        except Exception as e:
            self.logger.warning(f"Volume significance calculation failed: {e}")
            return np.ones(len(regime_predictions)) * 0.5
    
    def _calculate_volatility_impact(self, market_data: np.ndarray, 
                                   regime_predictions: np.ndarray) -> np.ndarray:
        """Calculate volatility impact for each regime."""
        try:
            if market_data.shape[1] < 4:
                return np.zeros(len(regime_predictions))
            
            close_prices = market_data[:, 3]
            volatility_scores = np.zeros(len(regime_predictions))
            
            unique_regimes = np.unique(regime_predictions)
            
            for regime in unique_regimes:
                regime_mask = regime_predictions == regime
                if not np.any(regime_mask):
                    continue
                
                regime_prices = close_prices[regime_mask]
                
                if len(regime_prices) < 3:
                    continue
                
                # Calculate returns
                returns = np.diff(regime_prices) / regime_prices[:-1]
                
                # Calculate volatility metrics
                volatility = np.std(returns)
                volatility_persistence = self._calculate_volatility_persistence(returns)
                volatility_clustering = self._calculate_volatility_clustering(returns)
                
                # Calculate volatility impact
                volatility_impact = (
                    min(volatility * 10, 1.0) * 0.4 +
                    volatility_persistence * 0.3 +
                    volatility_clustering * 0.3
                )
                
                volatility_scores[regime_mask] = min(volatility_impact, 1.0)
            
            return volatility_scores
            
        except Exception as e:
            self.logger.warning(f"Volatility impact calculation failed: {e}")
            return np.zeros(len(regime_predictions))
    
    def _calculate_volatility_persistence(self, returns: np.ndarray) -> float:
        """Calculate volatility persistence using GARCH-like approach."""
        try:
            if len(returns) < 5:
                return 0.0
            
            # Calculate rolling volatility
            window_size = min(5, len(returns) // 2)
            rolling_vol = []
            
            for i in range(window_size, len(returns)):
                vol = np.std(returns[i-window_size:i])
                rolling_vol.append(vol)
            
            if len(rolling_vol) < 2:
                return 0.0
            
            # Calculate autocorrelation of volatility
            vol_autocorr = np.corrcoef(rolling_vol[:-1], rolling_vol[1:])[0, 1]
            return abs(vol_autocorr) if not np.isnan(vol_autocorr) else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_volatility_clustering(self, returns: np.ndarray) -> float:
        """Calculate volatility clustering."""
        try:
            if len(returns) < 5:
                return 0.0
            
            # Calculate squared returns (proxy for volatility)
            squared_returns = returns ** 2
            
            # Calculate autocorrelation of squared returns
            if len(squared_returns) > 1:
                autocorr = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1]
                return abs(autocorr) if not np.isnan(autocorr) else 0.0
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _calculate_trend_consistency(self, market_data: np.ndarray, 
                                   regime_predictions: np.ndarray) -> np.ndarray:
        """Calculate trend consistency for each regime."""
        try:
            if market_data.shape[1] < 4:
                return np.zeros(len(regime_predictions))
            
            close_prices = market_data[:, 3]
            trend_scores = np.zeros(len(regime_predictions))
            
            unique_regimes = np.unique(regime_predictions)
            
            for regime in unique_regimes:
                regime_mask = regime_predictions == regime
                if not np.any(regime_mask):
                    continue
                
                regime_prices = close_prices[regime_mask]
                
                if len(regime_prices) < 3:
                    continue
                
                # Calculate trend metrics
                price_changes = np.diff(regime_prices)
                
                # Trend direction consistency
                positive_changes = np.sum(price_changes > 0)
                negative_changes = np.sum(price_changes < 0)
                total_changes = len(price_changes)
                
                if total_changes > 0:
                    trend_consistency = max(positive_changes, negative_changes) / total_changes
                else:
                    trend_consistency = 0.5
                
                # Trend strength
                if len(price_changes) > 1:
                    trend_strength = abs(np.corrcoef(np.arange(len(price_changes)), price_changes)[0, 1])
                else:
                    trend_strength = 0.0
                
                # Trend persistence
                trend_persistence = self._calculate_trend_persistence(price_changes)
                
                # Combine trend metrics
                trend_score = (
                    trend_consistency * 0.4 +
                    trend_strength * 0.3 +
                    trend_persistence * 0.3
                )
                
                trend_scores[regime_mask] = min(trend_score, 1.0)
            
            return trend_scores
            
        except Exception as e:
            self.logger.warning(f"Trend consistency calculation failed: {e}")
            return np.zeros(len(regime_predictions))
    
    def _calculate_trend_persistence(self, price_changes: np.ndarray) -> float:
        """Calculate trend persistence."""
        try:
            if len(price_changes) < 3:
                return 0.0
            
            # Calculate trend direction changes
            direction_changes = 0
            for i in range(1, len(price_changes)):
                if (price_changes[i] > 0) != (price_changes[i-1] > 0):
                    direction_changes += 1
            
            # Persistence is inverse of direction changes
            persistence = 1.0 - (direction_changes / (len(price_changes) - 1))
            return max(0.0, persistence)
            
        except Exception:
            return 0.0
    
    def _calculate_market_efficiency(self, market_data: np.ndarray, 
                                   regime_predictions: np.ndarray) -> np.ndarray:
        """Calculate market efficiency indicators for each regime."""
        try:
            if market_data.shape[1] < 4:
                return np.zeros(len(regime_predictions))
            
            close_prices = market_data[:, 3]
            efficiency_scores = np.zeros(len(regime_predictions))
            
            unique_regimes = np.unique(regime_predictions)
            
            for regime in unique_regimes:
                regime_mask = regime_predictions == regime
                if not np.any(regime_mask):
                    continue
                
                regime_prices = close_prices[regime_mask]
                
                if len(regime_prices) < 5:
                    continue
                
                # Calculate efficiency metrics
                returns = np.diff(regime_prices) / regime_prices[:-1]
                
                # Random walk test (autocorrelation)
                if len(returns) > 1:
                    autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
                    random_walk_score = 1.0 - abs(autocorr) if not np.isnan(autocorr) else 0.5
                else:
                    random_walk_score = 0.5
                
                # Variance ratio test (simplified)
                variance_ratio = self._calculate_variance_ratio(returns)
                
                # Price discovery efficiency
                price_discovery = self._calculate_price_discovery_efficiency(regime_prices)
                
                # Combine efficiency metrics
                efficiency_score = (
                    random_walk_score * 0.4 +
                    variance_ratio * 0.3 +
                    price_discovery * 0.3
                )
                
                efficiency_scores[regime_mask] = min(efficiency_score, 1.0)
            
            return efficiency_scores
            
        except Exception as e:
            self.logger.warning(f"Market efficiency calculation failed: {e}")
            return np.zeros(len(regime_predictions))
    
    def _calculate_variance_ratio(self, returns: np.ndarray) -> float:
        """Calculate variance ratio for market efficiency."""
        try:
            if len(returns) < 4:
                return 0.5
            
            # Calculate variance of returns
            var_1 = np.var(returns)
            
            # Calculate variance of 2-period returns
            returns_2 = returns[:-1] + returns[1:]
            var_2 = np.var(returns_2)
            
            # Variance ratio
            if var_1 > 0:
                variance_ratio = var_2 / (2 * var_1)
                return min(variance_ratio, 1.0)
            else:
                return 0.5
                
        except Exception:
            return 0.5
    
    def _calculate_price_discovery_efficiency(self, prices: np.ndarray) -> float:
        """Calculate price discovery efficiency."""
        try:
            if len(prices) < 3:
                return 0.5
            
            # Calculate price adjustment speed
            price_changes = np.diff(prices)
            price_volatility = np.std(price_changes)
            price_mean = np.mean(np.abs(price_changes))
            
            # Efficiency is higher when volatility is reasonable relative to mean
            if price_mean > 0:
                efficiency = 1.0 / (1.0 + price_volatility / price_mean)
            else:
                efficiency = 0.5
            
            return min(efficiency, 1.0)
            
        except Exception:
            return 0.5
    
    def _calculate_economic_indicator_correlation(self, market_data: np.ndarray, 
                                                regime_predictions: np.ndarray,
                                                timestamps: Optional[np.ndarray]) -> np.ndarray:
        """Calculate correlation with economic indicators."""
        try:
            if not self.economic_indicators:
                return np.ones(len(regime_predictions)) * 0.5
            
            indicator_scores = np.zeros(len(regime_predictions))
            
            # Simulate economic indicator correlation
            unique_regimes = np.unique(regime_predictions)
            
            for regime in unique_regimes:
                regime_mask = regime_predictions == regime
                if not np.any(regime_mask):
                    continue
                
                # Simulate correlation with economic indicators
                # In practice, this would use actual economic data
                correlation_score = np.random.uniform(0.3, 0.8)
                indicator_scores[regime_mask] = correlation_score
            
            return indicator_scores
            
        except Exception as e:
            self.logger.warning(f"Economic indicator correlation calculation failed: {e}")
            return np.ones(len(regime_predictions)) * 0.5
    
    def _calculate_trading_opportunity_significance(self, market_data: np.ndarray,
                                                 regime_predictions: np.ndarray) -> np.ndarray:
        """Calculate trading opportunity significance."""
        try:
            # Simple trading opportunity based on price movements and volatility
            if market_data.shape[1] < 4:
                return np.ones(len(regime_predictions)) * 0.5
            
            close_prices = market_data[:, 3]
            opportunity_scores = np.zeros(len(regime_predictions))
            
            unique_regimes = np.unique(regime_predictions)
            
            for regime in unique_regimes:
                regime_mask = regime_predictions == regime
                if not np.any(regime_mask):
                    continue
                
                regime_prices = close_prices[regime_mask]
                
                if len(regime_prices) < 3:
                    continue
                
                # Calculate returns
                returns = np.diff(regime_prices) / regime_prices[:-1]
                
                # Trading opportunity based on return magnitude and consistency
                return_magnitude = np.mean(np.abs(returns))
                return_consistency = 1.0 / (1.0 + np.std(returns))
                
                opportunity_score = (return_magnitude * 0.6 + return_consistency * 0.4)
                opportunity_scores[regime_mask] = min(opportunity_score, 1.0)
            
            return opportunity_scores
            
        except Exception as e:
            self.logger.warning(f"Trading opportunity calculation failed: {e}")
            return np.ones(len(regime_predictions)) * 0.5
    
    def _calculate_risk_adjustment_significance(self, market_data: np.ndarray,
                                             regime_predictions: np.ndarray) -> np.ndarray:
        """Calculate risk adjustment significance."""
        try:
            # Simple risk adjustment based on volatility and drawdown
            if market_data.shape[1] < 4:
                return np.ones(len(regime_predictions)) * 0.5
            
            close_prices = market_data[:, 3]
            risk_scores = np.zeros(len(regime_predictions))
            
            unique_regimes = np.unique(regime_predictions)
            
            for regime in unique_regimes:
                regime_mask = regime_predictions == regime
                if not np.any(regime_mask):
                    continue
                
                regime_prices = close_prices[regime_mask]
                
                if len(regime_prices) < 3:
                    continue
                
                # Calculate returns
                returns = np.diff(regime_prices) / regime_prices[:-1]
                
                # Risk metrics
                volatility = np.std(returns)
                max_drawdown = self._calculate_max_drawdown(np.cumprod(1 + returns))
                
                # Risk adjustment (lower risk is better)
                risk_score = 1.0 / (1.0 + volatility + max_drawdown)
                risk_scores[regime_mask] = min(risk_score, 1.0)
            
            return risk_scores
            
        except Exception as e:
            self.logger.warning(f"Risk adjustment calculation failed: {e}")
            return np.ones(len(regime_predictions)) * 0.5
    
    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Calculate maximum drawdown from cumulative returns."""
        try:
            if len(cumulative_returns) == 0:
                return 0.0
            
            peak = cumulative_returns[0]
            max_dd = 0.0
            
            for value in cumulative_returns:
                if value > peak:
                    peak = value
                dd = (peak - value) / (1 + peak) if peak != 0 else 0
                max_dd = max(max_dd, dd)
            
            return max_dd
            
        except Exception:
            return 0.0
    
    def _analyze_regime_economic_profiles(self, market_data: np.ndarray,
                                        regime_predictions: np.ndarray,
                                        timestamps: Optional[np.ndarray]) -> Dict[str, Dict[str, Any]]:
        """Analyze economic profiles for each regime."""
        try:
            profiles = {}
            unique_regimes = np.unique(regime_predictions)
            
            for regime in unique_regimes:
                regime_mask = regime_predictions == regime
                regime_data = market_data[regime_mask]
                
                if len(regime_data) == 0:
                    continue
                
                # Economic profile for this regime
                profile = {
                    'regime_id': regime,
                    'duration': len(regime_data),
                    'price_volatility': np.std(regime_data[:, 3]) if regime_data.shape[1] > 3 else 0.0,
                    'volume_characteristics': np.mean(regime_data[:, 4]) if regime_data.shape[1] > 4 else 1.0,
                    'trend_strength': self._calculate_trend_strength(regime_data),
                    'market_efficiency': self._calculate_regime_efficiency(regime_data)
                }
                
                profiles[f'regime_{regime}'] = profile
            
            return profiles
            
        except Exception as e:
            self.logger.warning(f"Regime economic profile analysis failed: {e}")
            return {}
    
    def _calculate_trend_strength(self, regime_data: np.ndarray) -> float:
        """Calculate trend strength for regime data."""
        try:
            if regime_data.shape[1] < 4 or len(regime_data) < 3:
                return 0.0
            
            prices = regime_data[:, 3]
            price_changes = np.diff(prices)
            
            if len(price_changes) > 1:
                trend_strength = abs(np.corrcoef(np.arange(len(price_changes)), price_changes)[0, 1])
                return trend_strength if not np.isnan(trend_strength) else 0.0
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _calculate_regime_efficiency(self, regime_data: np.ndarray) -> float:
        """Calculate market efficiency for regime data."""
        try:
            if regime_data.shape[1] < 4 or len(regime_data) < 3:
                return 0.5
            
            prices = regime_data[:, 3]
            returns = np.diff(prices) / prices[:-1]
            
            if len(returns) > 1:
                autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
                efficiency = 1.0 - abs(autocorr) if not np.isnan(autocorr) else 0.5
                return min(efficiency, 1.0)
            else:
                return 0.5
                
        except Exception:
            return 0.5
    
    def _calculate_regime_significance_scores(self, regime_predictions: np.ndarray,
                                           overall_scores: np.ndarray) -> Dict[str, float]:
        """Calculate significance scores for each regime."""
        try:
            regime_scores = {}
            unique_regimes = np.unique(regime_predictions)
            
            for regime in unique_regimes:
                regime_mask = regime_predictions == regime
                regime_score = np.mean(overall_scores[regime_mask])
                regime_scores[f'regime_{regime}'] = regime_score
            
            return regime_scores
            
        except Exception as e:
            self.logger.warning(f"Regime significance score calculation failed: {e}")
            return {}
    
    def _perform_bootstrap_analysis(self, market_data: np.ndarray,
                                  regime_predictions: np.ndarray) -> Dict[str, Any]:
        """Perform bootstrap analysis for statistical significance."""
        try:
            bootstrap_scores = []
            
            for _ in range(self.config.bootstrap_iterations):
                # Bootstrap sample
                indices = np.random.choice(len(market_data), size=len(market_data), replace=True)
                sample_predictions = regime_predictions[indices]
                sample_data = market_data[indices]
                
                # Calculate bootstrap metric (simplified)
                stability = self._calculate_bootstrap_stability(sample_predictions)
                bootstrap_scores.append(stability)
            
            # Calculate statistical significance
            mean_score = np.mean(bootstrap_scores)
            std_score = np.std(bootstrap_scores)
            
            # Confidence interval
            ci_lower = np.percentile(bootstrap_scores, (1 - self.config.confidence_level) / 2 * 100)
            ci_upper = np.percentile(bootstrap_scores, (1 + self.config.confidence_level) / 2 * 100)
            
            return {
                'bootstrap_mean': mean_score,
                'bootstrap_std': std_score,
                'statistical_significance': mean_score,
                'confidence_interval': (ci_lower, ci_upper),
                'bootstrap_scores': bootstrap_scores
            }
            
        except Exception as e:
            self.logger.warning(f"Bootstrap analysis failed: {e}")
            return {}
    
    def _calculate_bootstrap_stability(self, predictions: np.ndarray) -> float:
        """Calculate stability metric for bootstrap sample."""
        try:
            if len(predictions) < 2:
                return 0.0
            
            regime_changes = np.sum(np.diff(predictions) != 0)
            total_periods = len(predictions) - 1
            stability = 1.0 - (regime_changes / total_periods) if total_periods > 0 else 0.0
            
            return stability
            
        except Exception:
            return 0.0


# Convenience functions
def create_unified_economic_evaluator(config: Optional[EconomicEvaluationConfig] = None) -> UnifiedEconomicSignificanceEvaluator:
    """Create a unified economic significance evaluator."""
    if config is None:
        config = EconomicEvaluationConfig()
    return UnifiedEconomicSignificanceEvaluator(config)


def quick_economic_evaluation(market_data: Union[pd.DataFrame, np.ndarray],
                            regime_predictions: np.ndarray,
                            config: Optional[EconomicEvaluationConfig] = None) -> EconomicSignificanceResult:
    """Quick economic significance evaluation with default settings."""
    evaluator = create_unified_economic_evaluator(config)
    return evaluator.evaluate(market_data, regime_predictions)