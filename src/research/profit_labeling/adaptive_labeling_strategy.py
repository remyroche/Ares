"""
Adaptive Market Regime-Aware Labeling Strategy

This module provides adaptive labeling strategies that dynamically adjust parameters
based on current market conditions and detected market regimes. It replaces static
labeling parameters with intelligent, data-driven parameter selection.

Key Components:
1. Market Regime Detection (Volatility, Trend, Mean-Reversion regimes)
2. Contextual Parameter Optimization for each regime
3. Real-time Regime Classification
4. Adaptive Configuration Generation
5. Performance Tracking across Regimes
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import json
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy import stats
import warnings

from src.utils.logger import get_logger
from src.training.steps.pre_training.profit_labeling.consolidated_profit_labeler import MultiHorizonConfig

class MarketRegime(Enum):
    """Enumeration of market regimes."""
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    BREAKOUT = "breakout"
    CONSOLIDATION = "consolidation"
    CRISIS = "crisis"
    UNKNOWN = "unknown"

class RegimeDetectionMethod(Enum):
    """Enumeration of regime detection methods."""
    VOLATILITY_CLUSTERING = "volatility_clustering"
    TREND_ANALYSIS = "trend_analysis"
    GAUSSIAN_MIXTURE = "gaussian_mixture"
    KMEANS_CLUSTERING = "kmeans_clustering"
    STATISTICAL_TESTS = "statistical_tests"
    ENSEMBLE = "ensemble"

@dataclass
class AdaptiveLabelingConfig:
    """Configuration for adaptive labeling strategy."""
    # Regime detection parameters
    regime_detection_method: RegimeDetectionMethod = RegimeDetectionMethod.ENSEMBLE
    regime_lookback_window: int = 100
    regime_update_frequency: int = 20  # Update regime every N periods
    min_regime_duration: int = 10  # Minimum periods to stay in regime

    # Volatility regime parameters
    volatility_window: int = 20
    high_volatility_threshold: float = 0.75  # 75th percentile
    low_volatility_threshold: float = 0.25   # 25th percentile

    # Trend detection parameters
    trend_window: int = 50
    trend_strength_threshold: float = 0.6

    # Mean reversion parameters
    mean_reversion_window: int = 30
    mean_reversion_threshold: float = 2.0  # Standard deviations

    # Parameter adaptation settings
    adaptation_speed: float = 0.1  # How quickly to adapt parameters
    parameter_smoothing: float = 0.8  # Exponential smoothing factor
    regime_confidence_threshold: float = 0.6

    # Performance tracking
    track_regime_performance: bool = True
    performance_window: int = 200
    min_samples_for_optimization: int = 50

    # Default parameter sets for each regime
    regime_parameters: Dict[MarketRegime, Dict[str, Any]] = field(default_factory=lambda: {
        MarketRegime.HIGH_VOLATILITY: {
            'profit_targets': {'micro': 0.005, 'small': 0.008, 'medium': 0.012, 'good': 0.020},
            'time_horizons': {'immediate': 1, 'short': 3},
            'speed_weight': 0.4, 'risk_weight': 0.3, 'profitability_weight': 0.3
        },
        MarketRegime.LOW_VOLATILITY: {
            'profit_targets': {'micro': 0.002, 'small': 0.003, 'medium': 0.005, 'good': 0.008},
            'time_horizons': {'immediate': 3, 'short': 6},
            'speed_weight': 0.2, 'risk_weight': 0.5, 'profitability_weight': 0.3
        },
        MarketRegime.TRENDING_UP: {
            'profit_targets': {'micro': 0.003, 'small': 0.006, 'medium': 0.010, 'good': 0.015},
            'time_horizons': {'immediate': 2, 'short': 5},
            'speed_weight': 0.3, 'risk_weight': 0.2, 'profitability_weight': 0.5
        },
        MarketRegime.TRENDING_DOWN: {
            'profit_targets': {'micro': 0.004, 'small': 0.007, 'medium': 0.011, 'good': 0.018},
            'time_horizons': {'immediate': 2, 'short': 4},
            'speed_weight': 0.4, 'risk_weight': 0.4, 'profitability_weight': 0.2
        },
        MarketRegime.MEAN_REVERTING: {
            'profit_targets': {'micro': 0.002, 'small': 0.004, 'medium': 0.006, 'good': 0.010},
            'time_horizons': {'immediate': 1, 'short': 2},
            'speed_weight': 0.5, 'risk_weight': 0.3, 'profitability_weight': 0.2
        },
        MarketRegime.BREAKOUT: {
            'profit_targets': {'micro': 0.006, 'small': 0.010, 'medium': 0.015, 'good': 0.025},
            'time_horizons': {'immediate': 2, 'short': 4},
            'speed_weight': 0.3, 'risk_weight': 0.2, 'profitability_weight': 0.5
        },
        MarketRegime.CONSOLIDATION: {
            'profit_targets': {'micro': 0.002, 'small': 0.003, 'medium': 0.004, 'good': 0.006},
            'time_horizons': {'immediate': 4, 'short': 8},
            'speed_weight': 0.2, 'risk_weight': 0.6, 'profitability_weight': 0.2
        }
    })

@dataclass
class RegimeDetectionResult:
    """Result container for regime detection."""
    current_regime: MarketRegime
    regime_confidence: float
    regime_probabilities: Dict[MarketRegime, float]
    regime_features: Dict[str, float]
    detection_method: RegimeDetectionMethod
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class AdaptiveConfigResult:
    """Result container for adaptive configuration."""
    config: MultiHorizonConfig
    regime: MarketRegime
    regime_confidence: float
    parameter_adjustments: Dict[str, Any]
    performance_metrics: Dict[str, float]
    metadata: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

class MarketRegimeDetector:
    """
    Market regime detection using multiple methods.

    Detects different market regimes based on price action, volatility,
    and statistical patterns to enable regime-specific parameter optimization.
    """

    def __init__(self, config: Optional[AdaptiveLabelingConfig] = None):
        """Initialize the market regime detector."""
        self.config = config or AdaptiveLabelingConfig()
        self.logger = get_logger('MarketRegimeDetector')

        # Detection state
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_history: List[Tuple[datetime, MarketRegime, float]] = []
        self.feature_scalers: Dict[str, StandardScaler] = {}

        # Clustering models
        self.clustering_models: Dict[str, Any] = {}

        self.logger.info('🔍 Market Regime Detector initialized')
        self.logger.info(f'   → Detection method: {self.config.regime_detection_method.value}')

    def detect_regime(self, market_data: pd.DataFrame) -> RegimeDetectionResult:
        """
        Detect current market regime.

        Args:
            market_data: OHLCV market data

        Returns:
            RegimeDetectionResult with detected regime and confidence
        """
        if len(market_data) < self.config.regime_lookback_window:
            return RegimeDetectionResult(
                current_regime=MarketRegime.UNKNOWN,
                regime_confidence=0.0,
                regime_probabilities={MarketRegime.UNKNOWN: 1.0},
                regime_features={},
                detection_method=self.config.regime_detection_method
            )

        # Extract regime features
        features = self._extract_regime_features(market_data)

        # Detect regime using selected method
        if self.config.regime_detection_method == RegimeDetectionMethod.ENSEMBLE:
            result = self._detect_regime_ensemble(features, market_data)
        elif self.config.regime_detection_method == RegimeDetectionMethod.VOLATILITY_CLUSTERING:
            result = self._detect_regime_volatility(features, market_data)
        elif self.config.regime_detection_method == RegimeDetectionMethod.TREND_ANALYSIS:
            result = self._detect_regime_trend(features, market_data)
        elif self.config.regime_detection_method == RegimeDetectionMethod.GAUSSIAN_MIXTURE:
            result = self._detect_regime_gaussian_mixture(features, market_data)
        elif self.config.regime_detection_method == RegimeDetectionMethod.KMEANS_CLUSTERING:
            result = self._detect_regime_kmeans(features, market_data)
        else:
            result = self._detect_regime_statistical(features, market_data)

        # Update regime history
        self.regime_history.append((
            datetime.now(),
            result.current_regime,
            result.regime_confidence
        ))

        # Keep limited history
        if len(self.regime_history) > 1000:
            self.regime_history = self.regime_history[-1000:]

        self.current_regime = result.current_regime

        self.logger.info(f'🎯 Detected regime: {result.current_regime.value} (confidence: {result.regime_confidence:.2f})')

        return result

    def _extract_regime_features(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Extract features for regime detection."""
        features = {}

        if 'close' not in market_data.columns:
            return features

        # Price-based features
        prices = market_data['close'].values
        returns = np.diff(prices) / prices[:-1]

        # Volatility features
        features['volatility'] = np.std(returns[-self.config.volatility_window:])
        features['volatility_ma'] = np.mean([
            np.std(returns[i:i+10]) for i in range(len(returns)-10, len(returns))
        ]) if len(returns) > 10 else features['volatility']

        # Trend features
        if len(prices) >= self.config.trend_window:
            trend_window = prices[-self.config.trend_window:]
            slope, _, r_value, _, _ = stats.linregress(range(len(trend_window)), trend_window)
            features['trend_slope'] = slope / np.mean(trend_window)  # Normalized slope
            features['trend_strength'] = abs(r_value)
            features['trend_direction'] = 1.0 if slope > 0 else -1.0

        # Mean reversion features
        if len(prices) >= self.config.mean_reversion_window:
            recent_prices = prices[-self.config.mean_reversion_window:]
            price_mean = np.mean(recent_prices)
            price_std = np.std(recent_prices)
            current_price = prices[-1]

            if price_std > 0:
                features['mean_reversion_z'] = (current_price - price_mean) / price_std
                features['mean_reversion_strength'] = abs(features['mean_reversion_z'])

        # Range and momentum features
        if len(market_data) >= 20:
            high_20 = market_data['high'].rolling(20).max().iloc[-1]
            low_20 = market_data['low'].rolling(20).min().iloc[-1]
            current_close = market_data['close'].iloc[-1]

            if high_20 != low_20:
                features['price_position'] = (current_close - low_20) / (high_20 - low_20)

            # Momentum
            features['momentum_5'] = (current_close / market_data['close'].iloc[-6]) - 1
            features['momentum_20'] = (current_close / market_data['close'].iloc[-21]) - 1 if len(market_data) > 20 else 0

        # Volume features (if available)
        if 'volume' in market_data.columns:
            volume = market_data['volume'].values
            features['volume_trend'] = (np.mean(volume[-5:]) / np.mean(volume[-20:])) - 1 if len(volume) >= 20 else 0
            features['volume_volatility'] = np.std(volume[-20:]) / np.mean(volume[-20:]) if len(volume) >= 20 else 0

        # Handle missing values
        features = {k: v for k, v in features.items() if not (np.isnan(v) or np.isinf(v))}

        return features

    def _detect_regime_ensemble(self, features: Dict[str, float], market_data: pd.DataFrame) -> RegimeDetectionResult:
        """Detect regime using ensemble of methods."""
        methods = [
            self._detect_regime_volatility,
            self._detect_regime_trend,
            self._detect_regime_statistical
        ]

        regime_votes = {}
        confidence_sum = 0.0

        for method in methods:
            try:
                result = method(features, market_data)
                regime = result.current_regime
                confidence = result.regime_confidence

                if regime not in regime_votes:
                    regime_votes[regime] = 0.0
                regime_votes[regime] += confidence
                confidence_sum += confidence

            except Exception as e:
                self.logger.warning(f'Ensemble method failed: {e}')

        if not regime_votes:
            return RegimeDetectionResult(
                current_regime=MarketRegime.UNKNOWN,
                regime_confidence=0.0,
                regime_probabilities={MarketRegime.UNKNOWN: 1.0},
                regime_features=features,
                detection_method=RegimeDetectionMethod.ENSEMBLE
            )

        # Normalize votes to probabilities
        regime_probabilities = {}
        if confidence_sum > 0:
            for regime, vote in regime_votes.items():
                regime_probabilities[regime] = vote / confidence_sum

        # Select regime with highest vote
        best_regime = max(regime_votes.keys(), key=lambda k: regime_votes[k])
        best_confidence = regime_probabilities.get(best_regime, 0.0)

        return RegimeDetectionResult(
            current_regime=best_regime,
            regime_confidence=best_confidence,
            regime_probabilities=regime_probabilities,
            regime_features=features,
            detection_method=RegimeDetectionMethod.ENSEMBLE
        )

    def _detect_regime_volatility(self, features: Dict[str, float], market_data: pd.DataFrame) -> RegimeDetectionResult:
        """Detect regime based on volatility clustering."""
        volatility = features.get('volatility', 0.0)

        # Calculate volatility percentiles from recent history
        if len(market_data) >= 100:
            returns = market_data['close'].pct_change().dropna()
            rolling_vol = returns.rolling(self.config.volatility_window).std()

            high_vol_threshold = rolling_vol.quantile(self.config.high_volatility_threshold)
            low_vol_threshold = rolling_vol.quantile(self.config.low_volatility_threshold)

            if volatility > high_vol_threshold:
                regime = MarketRegime.HIGH_VOLATILITY
                confidence = min(1.0, (volatility - high_vol_threshold) / high_vol_threshold + 0.6)
            elif volatility < low_vol_threshold:
                regime = MarketRegime.LOW_VOLATILITY
                confidence = min(1.0, (low_vol_threshold - volatility) / low_vol_threshold + 0.6)
            else:
                # Check for breakout conditions
                price_position = features.get('price_position', 0.5)
                if price_position > 0.9 or price_position < 0.1:
                    regime = MarketRegime.BREAKOUT
                    confidence = abs(price_position - 0.5) * 2
                else:
                    regime = MarketRegime.CONSOLIDATION
                    confidence = 0.5
        else:
            regime = MarketRegime.UNKNOWN
            confidence = 0.0

        return RegimeDetectionResult(
            current_regime=regime,
            regime_confidence=confidence,
            regime_probabilities={regime: confidence, MarketRegime.UNKNOWN: 1.0 - confidence},
            regime_features=features,
            detection_method=RegimeDetectionMethod.VOLATILITY_CLUSTERING
        )

    def _detect_regime_trend(self, features: Dict[str, float], market_data: pd.DataFrame) -> RegimeDetectionResult:
        """Detect regime based on trend analysis."""
        trend_strength = features.get('trend_strength', 0.0)
        trend_direction = features.get('trend_direction', 0.0)

        if trend_strength > self.config.trend_strength_threshold:
            if trend_direction > 0:
                regime = MarketRegime.TRENDING_UP
            else:
                regime = MarketRegime.TRENDING_DOWN
            confidence = trend_strength
        else:
            # Check for mean reversion
            mean_reversion_strength = features.get('mean_reversion_strength', 0.0)
            if mean_reversion_strength > self.config.mean_reversion_threshold:
                regime = MarketRegime.MEAN_REVERTING
                confidence = min(1.0, mean_reversion_strength / 3.0)
            else:
                regime = MarketRegime.CONSOLIDATION
                confidence = 0.5

        return RegimeDetectionResult(
            current_regime=regime,
            regime_confidence=confidence,
            regime_probabilities={regime: confidence, MarketRegime.UNKNOWN: 1.0 - confidence},
            regime_features=features,
            detection_method=RegimeDetectionMethod.TREND_ANALYSIS
        )

    def _detect_regime_statistical(self, features: Dict[str, float], market_data: pd.DataFrame) -> RegimeDetectionResult:
        """Detect regime using statistical tests."""
        if len(market_data) < 50:
            return RegimeDetectionResult(
                current_regime=MarketRegime.UNKNOWN,
                regime_confidence=0.0,
                regime_probabilities={MarketRegime.UNKNOWN: 1.0},
                regime_features=features,
                detection_method=RegimeDetectionMethod.STATISTICAL_TESTS
            )

        returns = market_data['close'].pct_change().dropna()
        recent_returns = returns.tail(30)

        # Test for normality (crisis detection)
        try:
            _, normality_p = stats.jarque_bera(recent_returns)
            if normality_p < 0.01:  # Reject normality
                regime = MarketRegime.CRISIS
                confidence = 1.0 - normality_p
            else:
                # Use volatility and trend features
                volatility = features.get('volatility', 0.0)
                trend_strength = features.get('trend_strength', 0.0)

                if volatility > 0.02:  # High volatility threshold
                    regime = MarketRegime.HIGH_VOLATILITY
                    confidence = min(1.0, volatility / 0.05)
                elif trend_strength > 0.7:
                    trend_direction = features.get('trend_direction', 0.0)
                    regime = MarketRegime.TRENDING_UP if trend_direction > 0 else MarketRegime.TRENDING_DOWN
                    confidence = trend_strength
                else:
                    regime = MarketRegime.CONSOLIDATION
                    confidence = 0.6

        except Exception:
            regime = MarketRegime.UNKNOWN
            confidence = 0.0

        return RegimeDetectionResult(
            current_regime=regime,
            regime_confidence=confidence,
            regime_probabilities={regime: confidence, MarketRegime.UNKNOWN: 1.0 - confidence},
            regime_features=features,
            detection_method=RegimeDetectionMethod.STATISTICAL_TESTS
        )

    def _detect_regime_gaussian_mixture(self, features: Dict[str, float], market_data: pd.DataFrame) -> RegimeDetectionResult:
        """Detect regime using Gaussian Mixture Models."""
        # This is a placeholder for more advanced regime detection
        # Would require historical feature data to train GMM
        return self._detect_regime_statistical(features, market_data)

    def _detect_regime_kmeans(self, features: Dict[str, float], market_data: pd.DataFrame) -> RegimeDetectionResult:
        """Detect regime using K-means clustering."""
        # This is a placeholder for clustering-based regime detection
        # Would require historical feature data to train clustering model
        return self._detect_regime_statistical(features, market_data)

class ContextualParameterOptimizer:
    """
    Optimize parameters for specific market regimes.

    Maintains regime-specific parameter sets and optimizes them based on
    historical performance in each regime.
    """

    def __init__(self, config: Optional[AdaptiveLabelingConfig] = None):
        """Initialize the contextual parameter optimizer."""
        self.config = config or AdaptiveLabelingConfig()
        self.logger = get_logger('ContextualParameterOptimizer')

        # Parameter storage
        self.regime_parameters = self.config.regime_parameters.copy()
        self.parameter_history: Dict[MarketRegime, List[Dict[str, Any]]] = {}
        self.performance_history: Dict[MarketRegime, List[float]] = {}

        # Optimization state
        self.optimization_counter = 0

        self.logger.info('⚙️ Contextual Parameter Optimizer initialized')
        self.logger.info(f'   → Tracking {len(self.regime_parameters)} regime parameter sets')

    def optimize_for_regime(self, regime: MarketRegime, performance_data: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Optimize parameters for a specific market regime.

        Args:
            regime: Market regime to optimize for
            performance_data: Historical performance data for this regime

        Returns:
            Optimized parameter dictionary
        """
        if regime not in self.regime_parameters:
            self.logger.warning(f'Unknown regime {regime.value}, using default parameters')
            regime = MarketRegime.CONSOLIDATION

        base_params = self.regime_parameters[regime].copy()

        # If we have performance data, use it for optimization
        if performance_data and len(performance_data) >= self.config.min_samples_for_optimization:
            optimized_params = self._optimize_parameters_with_performance(regime, base_params, performance_data)
        else:
            # Use base parameters with small random adjustments
            optimized_params = self._apply_random_adjustments(base_params)

        # Store parameter history
        if regime not in self.parameter_history:
            self.parameter_history[regime] = []
        self.parameter_history[regime].append(optimized_params.copy())

        # Keep limited history
        if len(self.parameter_history[regime]) > 100:
            self.parameter_history[regime] = self.parameter_history[regime][-100:]

        return optimized_params

    def _optimize_parameters_with_performance(self,
                                            regime: MarketRegime,
                                            base_params: Dict[str, Any],
                                            performance_data: List[float]) -> Dict[str, Any]:
        """Optimize parameters based on performance feedback."""
        optimized_params = base_params.copy()

        # Calculate performance metrics
        avg_performance = np.mean(performance_data)
        performance_trend = np.polyfit(range(len(performance_data)), performance_data, 1)[0]

        # Adjust parameters based on performance
        if avg_performance < 0.5:  # Poor performance
            # Reduce targets and extend horizons for better hit rates
            for target_name in optimized_params['profit_targets']:
                optimized_params['profit_targets'][target_name] *= 0.9

            for horizon_name in optimized_params['time_horizons']:
                optimized_params['time_horizons'][horizon_name] = min(
                    optimized_params['time_horizons'][horizon_name] + 1, 10
                )

        elif avg_performance > 0.8:  # Good performance
            # Increase targets for higher profits
            for target_name in optimized_params['profit_targets']:
                optimized_params['profit_targets'][target_name] *= 1.05

        # Adjust quality weights based on regime
        if regime == MarketRegime.HIGH_VOLATILITY:
            # Emphasize speed in volatile conditions
            optimized_params['speed_weight'] = min(0.5, optimized_params['speed_weight'] * 1.1)
        elif regime == MarketRegime.TRENDING_UP or regime == MarketRegime.TRENDING_DOWN:
            # Emphasize profitability in trending markets
            optimized_params['profitability_weight'] = min(0.6, optimized_params['profitability_weight'] * 1.1)

        # Normalize weights
        total_weight = (optimized_params['speed_weight'] +
                       optimized_params['risk_weight'] +
                       optimized_params['profitability_weight'])
        if total_weight > 0:
            optimized_params['speed_weight'] /= total_weight
            optimized_params['risk_weight'] /= total_weight
            optimized_params['profitability_weight'] /= total_weight

        return optimized_params

    def _apply_random_adjustments(self, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply small random adjustments to parameters."""
        adjusted_params = base_params.copy()

        # Small random adjustments to profit targets (±5%)
        for target_name in adjusted_params['profit_targets']:
            adjustment = np.random.uniform(0.95, 1.05)
            adjusted_params['profit_targets'][target_name] *= adjustment

        # Small adjustments to quality weights (±10%)
        for weight_name in ['speed_weight', 'risk_weight', 'profitability_weight']:
            if weight_name in adjusted_params:
                adjustment = np.random.uniform(0.9, 1.1)
                adjusted_params[weight_name] *= adjustment

        # Normalize weights
        total_weight = (adjusted_params.get('speed_weight', 0.33) +
                       adjusted_params.get('risk_weight', 0.33) +
                       adjusted_params.get('profitability_weight', 0.33))
        if total_weight > 0:
            for weight_name in ['speed_weight', 'risk_weight', 'profitability_weight']:
                if weight_name in adjusted_params:
                    adjusted_params[weight_name] /= total_weight

        return adjusted_params

    def update_regime_performance(self, regime: MarketRegime, performance_score: float):
        """Update performance history for a regime."""
        if regime not in self.performance_history:
            self.performance_history[regime] = []

        self.performance_history[regime].append(performance_score)

        # Keep limited history
        if len(self.performance_history[regime]) > self.config.performance_window:
            self.performance_history[regime] = self.performance_history[regime][-self.config.performance_window:]

class AdaptiveLabelingStrategy:
    """
    Main adaptive labeling strategy that coordinates regime detection and parameter optimization.

    This class combines regime detection with contextual parameter optimization to provide
    adaptive labeling configurations that respond to changing market conditions.
    """

    def __init__(self, config: Optional[AdaptiveLabelingConfig] = None):
        """Initialize the adaptive labeling strategy."""
        self.config = config or AdaptiveLabelingConfig()
        self.logger = get_logger('AdaptiveLabelingStrategy')

        # Components
        self.regime_detector = MarketRegimeDetector(self.config)
        self.parameter_optimizer = ContextualParameterOptimizer(self.config)

        # State tracking
        self.current_config: Optional[MultiHorizonConfig] = None
        self.last_regime_update = datetime.now() - timedelta(days=1)
        self.regime_stability_counter = 0

        # Performance tracking
        self.adaptation_history: List[AdaptiveConfigResult] = []

        self.logger.info('🎯 Adaptive Labeling Strategy initialized')
        self.logger.info(f'   → Regime update frequency: every {self.config.regime_update_frequency} periods')

    def get_adaptive_config(self, market_data: pd.DataFrame) -> AdaptiveConfigResult:
        """
        Generate adaptive labeling configuration based on current market conditions.

        Args:
            market_data: Recent market data for regime detection

        Returns:
            AdaptiveConfigResult with optimized configuration
        """
        self.logger.info('🔄 Generating adaptive labeling configuration')

        # Detect current market regime
        regime_result = self.regime_detector.detect_regime(market_data)

        # Check if we should update the configuration
        should_update = self._should_update_config(regime_result)

        if should_update or self.current_config is None:
            # Get optimized parameters for current regime
            performance_data = self.parameter_optimizer.performance_history.get(regime_result.current_regime)
            optimized_params = self.parameter_optimizer.optimize_for_regime(
                regime_result.current_regime, performance_data
            )

            # Create new configuration
            new_config = self._create_multi_horizon_config(optimized_params)

            # Apply parameter smoothing if we have a previous configuration
            if self.current_config is not None:
                new_config = self._smooth_config_transition(self.current_config, new_config)

            self.current_config = new_config
            self.last_regime_update = datetime.now()

            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(regime_result, optimized_params)

            # Create result
            result = AdaptiveConfigResult(
                config=new_config,
                regime=regime_result.current_regime,
                regime_confidence=regime_result.regime_confidence,
                parameter_adjustments=optimized_params,
                performance_metrics=performance_metrics,
                metadata={
                    'regime_features': regime_result.regime_features,
                    'detection_method': regime_result.detection_method.value,
                    'update_triggered': True
                }
            )

            self.logger.info(f'✅ Updated configuration for {regime_result.current_regime.value} regime')

        else:
            # Use existing configuration
            result = AdaptiveConfigResult(
                config=self.current_config,
                regime=regime_result.current_regime,
                regime_confidence=regime_result.regime_confidence,
                parameter_adjustments={},
                performance_metrics={},
                metadata={
                    'regime_features': regime_result.regime_features,
                    'detection_method': regime_result.detection_method.value,
                    'update_triggered': False
                }
            )

        # Store adaptation history
        self.adaptation_history.append(result)
        if len(self.adaptation_history) > 1000:
            self.adaptation_history = self.adaptation_history[-1000:]

        return result

    def update_performance_feedback(self, regime: MarketRegime, performance_score: float):
        """Update performance feedback for regime-specific optimization."""
        self.parameter_optimizer.update_regime_performance(regime, performance_score)
        self.logger.info(f'📈 Updated performance for {regime.value}: {performance_score:.3f}')

    def _should_update_config(self, regime_result: RegimeDetectionResult) -> bool:
        """Determine if configuration should be updated."""
        # Check time-based update frequency
        time_since_update = datetime.now() - self.last_regime_update
        time_threshold = timedelta(minutes=self.config.regime_update_frequency * 5)  # Assume 5-min periods

        if time_since_update > time_threshold:
            return True

        # Check regime change with confidence threshold
        if (regime_result.current_regime != self.regime_detector.current_regime and
            regime_result.regime_confidence > self.config.regime_confidence_threshold):

            # Check regime stability (avoid frequent switching)
            if regime_result.current_regime == self.regime_detector.current_regime:
                self.regime_stability_counter += 1
            else:
                self.regime_stability_counter = 0

            if self.regime_stability_counter >= self.config.min_regime_duration:
                return True

        return False

    def _create_multi_horizon_config(self, optimized_params: Dict[str, Any]) -> MultiHorizonConfig:
        """Create MultiHorizonConfig from optimized parameters."""
        config = MultiHorizonConfig()

        # Set profit targets
        if 'profit_targets' in optimized_params:
            config.profit_targets = optimized_params['profit_targets'].copy()

        # Set time horizons
        if 'time_horizons' in optimized_params:
            config.time_horizons = optimized_params['time_horizons'].copy()

        # Set quality weights
        config.speed_weight = optimized_params.get('speed_weight', 0.3)
        config.risk_weight = optimized_params.get('risk_weight', 0.4)
        config.profitability_weight = optimized_params.get('profitability_weight', 0.3)

        # Enable quality scoring
        config.enable_quality_scoring = True

        return config

    def _smooth_config_transition(self,
                                old_config: MultiHorizonConfig,
                                new_config: MultiHorizonConfig) -> MultiHorizonConfig:
        """Apply exponential smoothing to configuration transitions."""
        smoothed_config = MultiHorizonConfig()

        # Smooth profit targets
        smoothed_config.profit_targets = {}
        for target_name in new_config.profit_targets:
            old_value = old_config.profit_targets.get(target_name, new_config.profit_targets[target_name])
            new_value = new_config.profit_targets[target_name]

            smoothed_value = (self.config.parameter_smoothing * old_value +
                            (1 - self.config.parameter_smoothing) * new_value)
            smoothed_config.profit_targets[target_name] = smoothed_value

        # Time horizons (discrete values, no smoothing)
        smoothed_config.time_horizons = new_config.time_horizons.copy()

        # Smooth quality weights
        smoothed_config.speed_weight = (
            self.config.parameter_smoothing * old_config.speed_weight +
            (1 - self.config.parameter_smoothing) * new_config.speed_weight
        )
        smoothed_config.risk_weight = (
            self.config.parameter_smoothing * old_config.risk_weight +
            (1 - self.config.parameter_smoothing) * new_config.risk_weight
        )
        smoothed_config.profitability_weight = (
            self.config.parameter_smoothing * old_config.profitability_weight +
            (1 - self.config.parameter_smoothing) * new_config.profitability_weight
        )

        # Copy other settings
        smoothed_config.transaction_cost = new_config.transaction_cost
        smoothed_config.enable_quality_scoring = new_config.enable_quality_scoring
        smoothed_config.leverage_aware = new_config.leverage_aware
        smoothed_config.small_move_emphasis = new_config.small_move_emphasis

        return smoothed_config

    def _calculate_performance_metrics(self,
                                     regime_result: RegimeDetectionResult,
                                     optimized_params: Dict[str, Any]) -> Dict[str, float]:
        """Calculate performance metrics for the adaptive configuration."""
        metrics = {}

        # Regime confidence
        metrics['regime_confidence'] = regime_result.regime_confidence

        # Parameter diversity (how much parameters changed)
        if self.adaptation_history:
            last_params = self.adaptation_history[-1].parameter_adjustments
            if last_params:
                param_changes = []
                for key in optimized_params:
                    if key in last_params:
                        if isinstance(optimized_params[key], dict):
                            # Handle nested dictionaries (like profit_targets)
                            for subkey in optimized_params[key]:
                                if subkey in last_params.get(key, {}):
                                    old_val = last_params[key][subkey]
                                    new_val = optimized_params[key][subkey]
                                    if old_val != 0:
                                        change = abs(new_val - old_val) / abs(old_val)
                                        param_changes.append(change)
                        else:
                            old_val = last_params[key]
                            new_val = optimized_params[key]
                            if old_val != 0:
                                change = abs(new_val - old_val) / abs(old_val)
                                param_changes.append(change)

                metrics['parameter_diversity'] = np.mean(param_changes) if param_changes else 0.0

        # Adaptation frequency
        recent_adaptations = [r for r in self.adaptation_history[-20:] if r.metadata.get('update_triggered', False)]
        metrics['adaptation_frequency'] = len(recent_adaptations) / min(20, len(self.adaptation_history))

        return metrics

    def get_regime_performance_summary(self) -> Dict[MarketRegime, Dict[str, float]]:
        """Get performance summary for all regimes."""
        summary = {}

        for regime, performance_data in self.parameter_optimizer.performance_history.items():
            if performance_data:
                summary[regime] = {
                    'mean_performance': np.mean(performance_data),
                    'std_performance': np.std(performance_data),
                    'sample_count': len(performance_data),
                    'recent_performance': np.mean(performance_data[-10:]) if len(performance_data) >= 10 else np.mean(performance_data)
                }

        return summary

    def save_adaptation_state(self, output_path: Union[str, Path]):
        """Save adaptation state to disk."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        state_data = {
            'config': self.config.__dict__,
            'regime_parameters': {k.value: v for k, v in self.parameter_optimizer.regime_parameters.items()},
            'performance_history': {k.value: v for k, v in self.parameter_optimizer.performance_history.items()},
            'regime_history': [(t.isoformat(), r.value, c) for t, r, c in self.regime_detector.regime_history],
            'current_regime': self.regime_detector.current_regime.value,
            'adaptation_count': len(self.adaptation_history)
        }

        with open(output_path, 'w') as f:
            json.dump(state_data, f, indent=2)

        self.logger.info(f'💾 Adaptation state saved to {output_path}')

    def load_adaptation_state(self, input_path: Union[str, Path]):
        """Load adaptation state from disk."""
        input_path = Path(input_path)

        if not input_path.exists():
            self.logger.warning(f'Adaptation state file not found: {input_path}')
            return

        with open(input_path, 'r') as f:
            state_data = json.load(f)

        # Restore performance history
        if 'performance_history' in state_data:
            self.parameter_optimizer.performance_history = {
                MarketRegime(k): v for k, v in state_data['performance_history'].items()
            }

        # Restore regime history
        if 'regime_history' in state_data:
            self.regime_detector.regime_history = [
                (datetime.fromisoformat(t), MarketRegime(r), c)
                for t, r, c in state_data['regime_history']
            ]

        # Restore current regime
        if 'current_regime' in state_data:
            self.regime_detector.current_regime = MarketRegime(state_data['current_regime'])

        self.logger.info(f'📂 Adaptation state loaded from {input_path}')

# Convenience functions
def create_adaptive_labeling_strategy(config: Optional[AdaptiveLabelingConfig] = None) -> AdaptiveLabelingStrategy:
    """Create an adaptive labeling strategy."""
    return AdaptiveLabelingStrategy(config)

def get_regime_adaptive_config(market_data: pd.DataFrame,
                             config: Optional[AdaptiveLabelingConfig] = None) -> AdaptiveConfigResult:
    """Get regime-adaptive configuration for current market conditions."""
    strategy = AdaptiveLabelingStrategy(config)
    return strategy.get_adaptive_config(market_data)
