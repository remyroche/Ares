"""
Ensemble Labeling System for Multi-Horizon Profit Labeling

This module provides ensemble labeling approaches that combine multiple labeling
strategies for improved robustness and accuracy. It implements various ensemble
methods and combination strategies.

Key Ensemble Components:
1. Multiple Labeling Strategy Implementations
2. Ensemble Combination Methods (Voting, Weighted, Stacking)
3. Dynamic Weight Optimization
4. Confidence-Based Selection
5. Performance-Based Adaptation
6. Diversity Measures and Ensemble Optimization

Now inherits from the production-ready BaseLabelingStrategy in core module.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
from pathlib import Path
import json
from datetime import datetime
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import the production-ready BaseLabelingStrategy
from src.core.abstract_base_classes import BaseLabelingStrategy as ProductionBaseLabelingStrategy, LabelingResult, LabelingStrategy

# ML imports for ensemble methods
from sklearn.ensemble import VotingRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger
from src.training.steps.pre_training.profit_labeling.consolidated_profit_labeler import (
    MultiHorizonProfitLabeler,
    MultiHorizonConfig
)

class LabelingStrategy(Enum):
    """Enumeration of labeling strategies."""
    MULTI_HORIZON = "multi_horizon"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    REGIME_AWARE = "regime_aware"
    ML_PREDICTIVE = "ml_predictive"
    MOMENTUM_BASED = "momentum_based"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT_FOCUSED = "breakout_focused"
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"

class EnsembleCombinationMethod(Enum):
    """Enumeration of ensemble combination methods."""
    SIMPLE_AVERAGE = "simple_average"
    WEIGHTED_AVERAGE = "weighted_average"
    PERFORMANCE_WEIGHTED = "performance_weighted"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    VOTING = "voting"
    STACKING = "stacking"
    DYNAMIC_SELECTION = "dynamic_selection"
    RANK_BASED = "rank_based"

class DiversityMeasure(Enum):
    """Enumeration of diversity measures for ensemble optimization."""
    CORRELATION_DIVERSITY = "correlation_diversity"
    DISAGREEMENT_DIVERSITY = "disagreement_diversity"
    ENTROPY_DIVERSITY = "entropy_diversity"
    Q_STATISTIC = "q_statistic"
    KOHAVI_WOLPERT = "kohavi_wolpert"

@dataclass
class EnsembleLabelingConfig:
    """Configuration for ensemble labeling system."""
    # Strategy selection
    enabled_strategies: List[LabelingStrategy] = field(default_factory=lambda: [
        LabelingStrategy.MULTI_HORIZON,
        LabelingStrategy.VOLATILITY_ADJUSTED,
        LabelingStrategy.REGIME_AWARE,
        LabelingStrategy.MOMENTUM_BASED,
        LabelingStrategy.MEAN_REVERSION
    ])

    # Ensemble combination
    combination_method: EnsembleCombinationMethod = EnsembleCombinationMethod.PERFORMANCE_WEIGHTED

    # Weight optimization
    optimize_weights: bool = True
    weight_optimization_window: int = 200
    weight_update_frequency: int = 50
    min_weight: float = 0.05  # Minimum weight for any strategy

    # Performance tracking
    performance_window: int = 100
    performance_metrics: List[str] = field(default_factory=lambda: [
        'correlation', 'hit_rate', 'sharpe_ratio', 'information_ratio'
    ])

    # Diversity optimization
    target_diversity: float = 0.7
    diversity_measure: DiversityMeasure = DiversityMeasure.CORRELATION_DIVERSITY
    diversity_weight: float = 0.3  # Weight of diversity in ensemble optimization

    # Confidence thresholds
    min_confidence: float = 0.1
    confidence_threshold: float = 0.6

    # Stacking parameters (if using stacking)
    stacking_meta_learner: str = "ridge"  # "linear", "ridge", "tree"
    stacking_cv_folds: int = 5

    # Dynamic selection parameters
    dynamic_selection_k: int = 3  # Select top k strategies
    selection_window: int = 50

    # Parallel processing
    n_jobs: int = -1
    timeout_seconds: int = 300

@dataclass
class StrategyResult:
    """Result container for individual labeling strategy."""
    strategy: LabelingStrategy
    labels: pd.DataFrame
    confidence_scores: pd.Series
    performance_metrics: Dict[str, float]
    metadata: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class EnsembleResult:
    """Result container for ensemble labeling."""
    ensemble_labels: pd.DataFrame
    strategy_results: Dict[LabelingStrategy, StrategyResult]
    combination_weights: Dict[LabelingStrategy, float]
    ensemble_confidence: pd.Series
    diversity_score: float
    performance_metrics: Dict[str, float]
    metadata: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

class BaseLabelingStrategy(ProductionBaseLabelingStrategy):
    """Base class for labeling strategies."""

    @abstractmethod
    def generate_labels(self, market_data: pd.DataFrame) -> StrategyResult:
        """
        Generate labels using this strategy.
        
        Args:
            market_data: OHLCV market data
            
        Returns:
            StrategyResult containing labels, confidence scores, and metadata
        """
        raise NotImplementedError("Subclasses must implement generate_labels method")

    @abstractmethod
    def calculate_confidence(self, labels: pd.DataFrame, market_data: pd.DataFrame) -> pd.Series:
        """
        Calculate confidence scores for labels.
        
        Args:
            labels: Generated labels DataFrame
            market_data: Original market data
            
        Returns:
            Series of confidence scores (0-1 range)
        """
        raise NotImplementedError("Subclasses must implement calculate_confidence method")

class MultiHorizonStrategy(BaseLabelingStrategy):
    """Multi-horizon profit labeling strategy."""

    def __init__(self, config: Optional[MultiHorizonConfig] = None):
        """Initialize multi-horizon strategy."""
        self.config = config or MultiHorizonConfig()
        self.labeler = MultiHorizonProfitLabeler(self.config)

    def generate_labels(self, market_data: pd.DataFrame) -> StrategyResult:
        """Generate labels using multi-horizon approach."""
        labeled_data = self.labeler.generate_labels(market_data.copy())
        confidence_scores = self.calculate_confidence(labeled_data, market_data)

        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(labeled_data, market_data)

        return StrategyResult(
            strategy=LabelingStrategy.MULTI_HORIZON,
            labels=labeled_data,
            confidence_scores=confidence_scores,
            performance_metrics=performance_metrics,
            metadata={'config': self.config.__dict__}
        )

    def calculate_confidence(self, labels: pd.DataFrame, market_data: pd.DataFrame) -> pd.Series:
        """Calculate confidence based on quality scores."""
        quality_columns = [col for col in labels.columns if col.endswith('_quality_score')]

        if quality_columns:
            # Average quality scores as confidence
            confidence = labels[quality_columns].mean(axis=1)
        else:
            # Use overall opportunity as proxy
            confidence = labels.get('overall_opportunity', pd.Series(0.5, index=labels.index))

        return confidence.fillna(0.5)

    def _calculate_performance_metrics(self, labels: pd.DataFrame, market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics for the strategy."""
        metrics = {}

        if 'overall_opportunity' in labels.columns and 'close' in market_data.columns:
            opportunity = labels['overall_opportunity'].fillna(0)
            returns = market_data['close'].pct_change().shift(-1).fillna(0)

            # Correlation
            common_idx = opportunity.index.intersection(returns.index)
            if len(common_idx) > 10:
                corr = np.corrcoef(opportunity.loc[common_idx], returns.loc[common_idx])[0, 1]
                metrics['correlation'] = abs(corr) if not np.isnan(corr) else 0.0

                # Hit rate (when opportunity > 0.5, return should be positive)
                signals = (opportunity.loc[common_idx] > 0.5).astype(int)
                hit_rate = ((signals == 1) & (returns.loc[common_idx] > 0)).sum() / max(signals.sum(), 1)
                metrics['hit_rate'] = hit_rate

                # Sharpe-like ratio
                strategy_returns = signals * returns.loc[common_idx]
                if strategy_returns.std() > 0:
                    metrics['sharpe_ratio'] = strategy_returns.mean() / strategy_returns.std()
                else:
                    metrics['sharpe_ratio'] = 0.0

        return metrics

class VolatilityAdjustedStrategy(BaseLabelingStrategy):
    """Volatility-adjusted labeling strategy."""

    def __init__(self, volatility_window: int = 20):
        """Initialize volatility-adjusted strategy."""
        self.volatility_window = volatility_window

    def generate_labels(self, market_data: pd.DataFrame) -> StrategyResult:
        """Generate volatility-adjusted labels."""
        if 'close' not in market_data.columns:
            # Return empty result
            empty_labels = pd.DataFrame(index=market_data.index)
            empty_labels['volatility_opportunity'] = 0.0
            return StrategyResult(
                strategy=LabelingStrategy.VOLATILITY_ADJUSTED,
                labels=empty_labels,
                confidence_scores=pd.Series(0.0, index=market_data.index),
                performance_metrics={},
                metadata={}
            )

        # Calculate rolling volatility
        returns = market_data['close'].pct_change()
        volatility = returns.rolling(self.volatility_window).std()

        # Create volatility-based opportunity scores
        vol_percentile = volatility.rolling(100).rank(pct=True)

        # Higher volatility = higher opportunity (but also higher risk)
        volatility_opportunity = vol_percentile * 0.8  # Scale to 0-0.8 range

        # Adjust profit targets based on volatility
        base_targets = {'micro': 0.003, 'small': 0.005, 'medium': 0.007, 'good': 0.010}
        vol_multiplier = 1.0 + volatility.fillna(0.02) * 50  # Scale volatility

        labels = pd.DataFrame(index=market_data.index)
        labels['volatility_opportunity'] = volatility_opportunity.fillna(0.0)

        # Add volatility-adjusted target probabilities
        for target_name, base_target in base_targets.items():
            adjusted_target = base_target * vol_multiplier
            # Simple probability based on how achievable the target is
            target_prob = np.clip(1.0 - (adjusted_target - base_target) / base_target, 0.0, 1.0)
            labels[f'{target_name}_vol_prob'] = target_prob.fillna(0.0)

        confidence_scores = self.calculate_confidence(labels, market_data)
        performance_metrics = self._calculate_performance_metrics(labels, market_data)

        return StrategyResult(
            strategy=LabelingStrategy.VOLATILITY_ADJUSTED,
            labels=labels,
            confidence_scores=confidence_scores,
            performance_metrics=performance_metrics,
            metadata={'volatility_window': self.volatility_window}
        )

    def calculate_confidence(self, labels: pd.DataFrame, market_data: pd.DataFrame) -> pd.Series:
        """Calculate confidence based on volatility consistency."""
        if 'volatility_opportunity' in labels.columns:
            # Confidence is higher when volatility is more predictable
            opportunity = labels['volatility_opportunity']
            rolling_std = opportunity.rolling(20).std()
            confidence = 1.0 - (rolling_std / (opportunity.std() + 1e-10))
            return confidence.fillna(0.5).clip(0.1, 1.0)

        return pd.Series(0.5, index=labels.index)

    def _calculate_performance_metrics(self, labels: pd.DataFrame, market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics."""
        metrics = {}

        if 'volatility_opportunity' in labels.columns and 'close' in market_data.columns:
            opportunity = labels['volatility_opportunity']
            returns = market_data['close'].pct_change().shift(-1).fillna(0)

            common_idx = opportunity.index.intersection(returns.index)
            if len(common_idx) > 10:
                corr = np.corrcoef(opportunity.loc[common_idx], returns.loc[common_idx])[0, 1]
                metrics['correlation'] = abs(corr) if not np.isnan(corr) else 0.0

        return metrics

class MomentumBasedStrategy(BaseLabelingStrategy):
    """Momentum-based labeling strategy."""

    def __init__(self, momentum_windows: List[int] = [5, 10, 20]):
        """Initialize momentum-based strategy."""
        self.momentum_windows = momentum_windows

    def generate_labels(self, market_data: pd.DataFrame) -> StrategyResult:
        """Generate momentum-based labels."""
        if 'close' not in market_data.columns:
            empty_labels = pd.DataFrame(index=market_data.index)
            empty_labels['momentum_opportunity'] = 0.0
            return StrategyResult(
                strategy=LabelingStrategy.MOMENTUM_BASED,
                labels=empty_labels,
                confidence_scores=pd.Series(0.0, index=market_data.index),
                performance_metrics={},
                metadata={}
            )

        prices = market_data['close']
        labels = pd.DataFrame(index=market_data.index)

        # Calculate momentum for different windows
        momentum_scores = []
        for window in self.momentum_windows:
            if len(prices) > window:
                momentum = (prices / prices.shift(window)) - 1
                momentum_scores.append(momentum.fillna(0))

        if momentum_scores:
            # Combine momentum scores
            avg_momentum = pd.concat(momentum_scores, axis=1).mean(axis=1)

            # Convert momentum to opportunity (absolute momentum indicates opportunity)
            momentum_opportunity = np.tanh(abs(avg_momentum) * 10) * 0.8  # Scale to 0-0.8
            labels['momentum_opportunity'] = momentum_opportunity.fillna(0.0)

            # Direction-based probabilities
            labels['upward_momentum_prob'] = np.clip((avg_momentum + 0.05) / 0.1, 0.0, 1.0).fillna(0.5)
            labels['downward_momentum_prob'] = np.clip((-avg_momentum + 0.05) / 0.1, 0.0, 1.0).fillna(0.5)
        else:
            labels['momentum_opportunity'] = 0.0
            labels['upward_momentum_prob'] = 0.5
            labels['downward_momentum_prob'] = 0.5

        confidence_scores = self.calculate_confidence(labels, market_data)
        performance_metrics = self._calculate_performance_metrics(labels, market_data)

        return StrategyResult(
            strategy=LabelingStrategy.MOMENTUM_BASED,
            labels=labels,
            confidence_scores=confidence_scores,
            performance_metrics=performance_metrics,
            metadata={'momentum_windows': self.momentum_windows}
        )

    def calculate_confidence(self, labels: pd.DataFrame, market_data: pd.DataFrame) -> pd.Series:
        """Calculate confidence based on momentum consistency."""
        if 'momentum_opportunity' in labels.columns:
            opportunity = labels['momentum_opportunity']
            # Higher opportunity = higher confidence
            return opportunity.clip(0.1, 1.0)

        return pd.Series(0.5, index=labels.index)

    def _calculate_performance_metrics(self, labels: pd.DataFrame, market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics."""
        metrics = {}

        if 'momentum_opportunity' in labels.columns and 'close' in market_data.columns:
            opportunity = labels['momentum_opportunity']
            returns = market_data['close'].pct_change().shift(-1).fillna(0)

            common_idx = opportunity.index.intersection(returns.index)
            if len(common_idx) > 10:
                corr = np.corrcoef(opportunity.loc[common_idx], returns.loc[common_idx])[0, 1]
                metrics['correlation'] = abs(corr) if not np.isnan(corr) else 0.0

        return metrics

class MeanReversionStrategy(BaseLabelingStrategy):
    """Mean reversion labeling strategy."""

    def __init__(self, lookback_window: int = 30, threshold_std: float = 2.0):
        """Initialize mean reversion strategy."""
        self.lookback_window = lookback_window
        self.threshold_std = threshold_std

    def generate_labels(self, market_data: pd.DataFrame) -> StrategyResult:
        """Generate mean reversion labels."""
        if 'close' not in market_data.columns:
            empty_labels = pd.DataFrame(index=market_data.index)
            empty_labels['reversion_opportunity'] = 0.0
            return StrategyResult(
                strategy=LabelingStrategy.MEAN_REVERSION,
                labels=empty_labels,
                confidence_scores=pd.Series(0.0, index=market_data.index),
                performance_metrics={},
                metadata={}
            )

        prices = market_data['close']
        labels = pd.DataFrame(index=market_data.index)

        # Calculate mean reversion signals
        rolling_mean = prices.rolling(self.lookback_window).mean()
        rolling_std = prices.rolling(self.lookback_window).std()

        # Z-score (deviation from mean in standard deviations)
        z_score = (prices - rolling_mean) / (rolling_std + 1e-10)

        # Reversion opportunity is higher when price is far from mean
        reversion_opportunity = np.tanh(abs(z_score) / self.threshold_std) * 0.8
        labels['reversion_opportunity'] = reversion_opportunity.fillna(0.0)

        # Direction-based probabilities (expect reversion to mean)
        labels['revert_down_prob'] = np.clip((z_score - 1.0) / 2.0, 0.0, 1.0).fillna(0.5)
        labels['revert_up_prob'] = np.clip((-z_score - 1.0) / 2.0, 0.0, 1.0).fillna(0.5)

        # Strength of mean reversion signal
        labels['reversion_strength'] = abs(z_score).fillna(0.0) / self.threshold_std

        confidence_scores = self.calculate_confidence(labels, market_data)
        performance_metrics = self._calculate_performance_metrics(labels, market_data)

        return StrategyResult(
            strategy=LabelingStrategy.MEAN_REVERSION,
            labels=labels,
            confidence_scores=confidence_scores,
            performance_metrics=performance_metrics,
            metadata={
                'lookback_window': self.lookback_window,
                'threshold_std': self.threshold_std
            }
        )

    def calculate_confidence(self, labels: pd.DataFrame, market_data: pd.DataFrame) -> pd.Series:
        """Calculate confidence based on reversion strength."""
        if 'reversion_strength' in labels.columns:
            strength = labels['reversion_strength']
            # Confidence increases with reversion strength but caps at 1.0
            return np.clip(strength * 0.5 + 0.2, 0.1, 1.0)

        return pd.Series(0.5, index=labels.index)

    def _calculate_performance_metrics(self, labels: pd.DataFrame, market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics."""
        metrics = {}

        if 'reversion_opportunity' in labels.columns and 'close' in market_data.columns:
            opportunity = labels['reversion_opportunity']
            returns = market_data['close'].pct_change().shift(-1).fillna(0)

            common_idx = opportunity.index.intersection(returns.index)
            if len(common_idx) > 10:
                # For mean reversion, we expect negative correlation (high opportunity when price reverts)
                corr = np.corrcoef(opportunity.loc[common_idx], returns.loc[common_idx])[0, 1]
                metrics['correlation'] = abs(corr) if not np.isnan(corr) else 0.0

        return metrics

class RegimeAwareStrategy(BaseLabelingStrategy):
    """Regime-aware labeling strategy that adapts to market conditions."""

    def __init__(self, regime_window: int = 50, volatility_threshold: float = 0.02):
        """Initialize regime-aware strategy."""
        self.regime_window = regime_window
        self.volatility_threshold = volatility_threshold

    def generate_labels(self, market_data: pd.DataFrame) -> StrategyResult:
        """Generate regime-aware labels."""
        if 'close' not in market_data.columns:
            empty_labels = pd.DataFrame(index=market_data.index)
            empty_labels['regime_opportunity'] = 0.0
            return StrategyResult(
                strategy=LabelingStrategy.REGIME_AWARE,
                labels=empty_labels,
                confidence_scores=pd.Series(0.0, index=market_data.index),
                performance_metrics={},
                metadata={}
            )

        prices = market_data['close']
        labels = pd.DataFrame(index=market_data.index)

        # Detect market regimes
        returns = prices.pct_change()
        volatility = returns.rolling(self.regime_window).std()
        trend = prices.rolling(self.regime_window).apply(lambda x: (x.iloc[-1] / x.iloc[0]) - 1)

        # Classify regimes
        high_vol = volatility > self.volatility_threshold
        uptrend = trend > 0.05
        downtrend = trend < -0.05

        # Regime-based opportunity calculation
        regime_opportunity = pd.Series(0.0, index=market_data.index)

        # High volatility regime - focus on mean reversion
        high_vol_mask = high_vol.fillna(False)
        if high_vol_mask.any():
            z_scores = (prices - prices.rolling(self.regime_window).mean()) / (volatility + 1e-10)
            regime_opportunity[high_vol_mask] = np.tanh(abs(z_scores[high_vol_mask]) / 2) * 0.8

        # Trending regime - follow momentum
        trend_mask = (uptrend | downtrend).fillna(False)
        if trend_mask.any():
            momentum = trend[trend_mask]
            regime_opportunity[trend_mask] = np.tanh(abs(momentum) * 5) * 0.6

        # Low volatility regime - breakout potential
        low_vol_mask = (~high_vol & ~trend_mask).fillna(False)
        if low_vol_mask.any():
            # Look for consolidation patterns
            price_range = prices.rolling(self.regime_window).max() - prices.rolling(self.regime_window).min()
            normalized_range = price_range / prices.rolling(self.regime_window).mean()
            regime_opportunity[low_vol_mask] = np.tanh(normalized_range[low_vol_mask] * 10) * 0.4

        labels['regime_opportunity'] = regime_opportunity.fillna(0.0)

        # Add regime-specific probabilities
        labels['trend_following_prob'] = uptrend.fillna(0.5).clip(0, 1)
        labels['mean_reversion_prob'] = high_vol.fillna(0.5).clip(0, 1)
        labels['breakout_prob'] = low_vol_mask.fillna(0.5).clip(0, 1)

        # Regime strength indicator
        labels['regime_strength'] = (abs(trend) + volatility.fillna(0)).clip(0, 1)

        confidence_scores = self.calculate_confidence(labels, market_data)
        performance_metrics = self._calculate_performance_metrics(labels, market_data)

        return StrategyResult(
            strategy=LabelingStrategy.REGIME_AWARE,
            labels=labels,
            confidence_scores=confidence_scores,
            performance_metrics=performance_metrics,
            metadata={
                'regime_window': self.regime_window,
                'volatility_threshold': self.volatility_threshold
            }
        )

    def calculate_confidence(self, labels: pd.DataFrame, market_data: pd.DataFrame) -> pd.Series:
        """Calculate confidence based on regime clarity."""
        if 'regime_strength' in labels.columns:
            strength = labels['regime_strength']
            # Higher confidence when regime is clearly defined
            return np.clip(strength * 0.8 + 0.2, 0.1, 1.0)

        return pd.Series(0.5, index=labels.index)

    def _calculate_performance_metrics(self, labels: pd.DataFrame, market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics."""
        metrics = {}

        if 'regime_opportunity' in labels.columns and 'close' in market_data.columns:
            opportunity = labels['regime_opportunity']
            returns = market_data['close'].pct_change().shift(-1).fillna(0)

            common_idx = opportunity.index.intersection(returns.index)
            if len(common_idx) > 10:
                corr = np.corrcoef(opportunity.loc[common_idx], returns.loc[common_idx])[0, 1]
                metrics['correlation'] = abs(corr) if not np.isnan(corr) else 0.0

        return metrics

class MLPredictiveStrategy(BaseLabelingStrategy):
    """ML-based predictive labeling strategy."""

    def __init__(self, lookback_window: int = 20, prediction_horizon: int = 5):
        """Initialize ML predictive strategy."""
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.scaler = StandardScaler()

    def generate_labels(self, market_data: pd.DataFrame) -> StrategyResult:
        """Generate ML-based predictive labels."""
        if 'close' not in market_data.columns or len(market_data) < self.lookback_window + self.prediction_horizon:
            empty_labels = pd.DataFrame(index=market_data.index)
            empty_labels['ml_opportunity'] = 0.0
            return StrategyResult(
                strategy=LabelingStrategy.ML_PREDICTIVE,
                labels=empty_labels,
                confidence_scores=pd.Series(0.0, index=market_data.index),
                performance_metrics={},
                metadata={}
            )

        prices = market_data['close']
        labels = pd.DataFrame(index=market_data.index)

        try:
            # Create features
            features = self._create_features(prices)
            
            if len(features) < 50:  # Need sufficient data for ML
                labels['ml_opportunity'] = 0.0
                labels['ml_confidence'] = 0.0
            else:
                # Simple linear regression for prediction
                X = features[:-self.prediction_horizon]
                y = (prices.shift(-self.prediction_horizon) / prices - 1).iloc[:-self.prediction_horizon]
                
                # Remove NaN values
                valid_mask = ~(X.isna().any(axis=1) | y.isna())
                X_clean = X[valid_mask]
                y_clean = y[valid_mask]
                
                if len(X_clean) > 20:
                    # Scale features
                    X_scaled = self.scaler.fit_transform(X_clean)
                    
                    # Train simple model
                    from sklearn.linear_model import Ridge
                    self.model = Ridge(alpha=1.0)
                    self.model.fit(X_scaled, y_clean)
                    
                    # Make predictions
                    X_pred = features.iloc[-self.prediction_horizon:]
                    X_pred_scaled = self.scaler.transform(X_pred)
                    predictions = self.model.predict(X_pred_scaled)
                    
                    # Convert predictions to opportunity scores
                    ml_opportunity = np.tanh(predictions * 10) * 0.8
                    ml_opportunity = np.clip(ml_opportunity, 0, 1)
                    
                    # Pad with zeros for the rest of the data
                    full_opportunity = pd.Series(0.0, index=market_data.index)
                    full_opportunity.iloc[-self.prediction_horizon:] = ml_opportunity
                    
                    labels['ml_opportunity'] = full_opportunity
                    labels['ml_confidence'] = np.abs(ml_opportunity)  # Confidence based on prediction strength
                else:
                    labels['ml_opportunity'] = 0.0
                    labels['ml_confidence'] = 0.0

        except Exception as e:
            labels['ml_opportunity'] = 0.0
            labels['ml_confidence'] = 0.0

        confidence_scores = self.calculate_confidence(labels, market_data)
        performance_metrics = self._calculate_performance_metrics(labels, market_data)

        return StrategyResult(
            strategy=LabelingStrategy.ML_PREDICTIVE,
            labels=labels,
            confidence_scores=confidence_scores,
            performance_metrics=performance_metrics,
            metadata={
                'lookback_window': self.lookback_window,
                'prediction_horizon': self.prediction_horizon
            }
        )

    def _create_features(self, prices: pd.Series) -> pd.DataFrame:
        """Create features for ML model."""
        features = pd.DataFrame(index=prices.index)
        
        # Price-based features
        features['returns'] = prices.pct_change()
        features['log_returns'] = np.log(prices / prices.shift(1))
        features['volatility'] = features['returns'].rolling(5).std()
        
        # Technical indicators
        features['sma_5'] = prices.rolling(5).mean() / prices
        features['sma_10'] = prices.rolling(10).mean() / prices
        features['sma_20'] = prices.rolling(20).mean() / prices
        
        # Momentum features
        features['momentum_5'] = prices / prices.shift(5) - 1
        features['momentum_10'] = prices / prices.shift(10) - 1
        
        # Volatility features
        features['vol_ratio'] = features['volatility'] / features['volatility'].rolling(20).mean()
        
        return features.dropna()

    def calculate_confidence(self, labels: pd.DataFrame, market_data: pd.DataFrame) -> pd.Series:
        """Calculate confidence based on ML model performance."""
        if 'ml_confidence' in labels.columns:
            return labels['ml_confidence'].clip(0.1, 1.0)

        return pd.Series(0.5, index=labels.index)

    def _calculate_performance_metrics(self, labels: pd.DataFrame, market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics."""
        metrics = {}

        if 'ml_opportunity' in labels.columns and 'close' in market_data.columns:
            opportunity = labels['ml_opportunity']
            returns = market_data['close'].pct_change().shift(-1).fillna(0)

            common_idx = opportunity.index.intersection(returns.index)
            if len(common_idx) > 10:
                corr = np.corrcoef(opportunity.loc[common_idx], returns.loc[common_idx])[0, 1]
                metrics['correlation'] = abs(corr) if not np.isnan(corr) else 0.0

        return metrics

class BreakoutFocusedStrategy(BaseLabelingStrategy):
    """Breakout-focused labeling strategy."""

    def __init__(self, breakout_window: int = 20, breakout_threshold: float = 0.02):
        """Initialize breakout-focused strategy."""
        self.breakout_window = breakout_window
        self.breakout_threshold = breakout_threshold

    def generate_labels(self, market_data: pd.DataFrame) -> StrategyResult:
        """Generate breakout-focused labels."""
        if 'close' not in market_data.columns:
            empty_labels = pd.DataFrame(index=market_data.index)
            empty_labels['breakout_opportunity'] = 0.0
            return StrategyResult(
                strategy=LabelingStrategy.BREAKOUT_FOCUSED,
                labels=empty_labels,
                confidence_scores=pd.Series(0.0, index=market_data.index),
                performance_metrics={},
                metadata={}
            )

        prices = market_data['close']
        labels = pd.DataFrame(index=market_data.index)

        # Calculate support and resistance levels
        rolling_high = prices.rolling(self.breakout_window).max()
        rolling_low = prices.rolling(self.breakout_window).min()
        rolling_range = rolling_high - rolling_low

        # Detect breakouts
        resistance_breakout = (prices > rolling_high.shift(1)) & (rolling_range > 0)
        support_breakout = (prices < rolling_low.shift(1)) & (rolling_range > 0)

        # Calculate breakout strength
        resistance_strength = np.where(
            resistance_breakout,
            (prices - rolling_high.shift(1)) / (rolling_range.shift(1) + 1e-10),
            0
        )
        support_strength = np.where(
            support_breakout,
            (rolling_low.shift(1) - prices) / (rolling_range.shift(1) + 1e-10),
            0
        )

        # Combine breakout signals
        breakout_strength = np.maximum(resistance_strength, support_strength)
        breakout_opportunity = np.tanh(breakout_strength * 5) * 0.8

        labels['breakout_opportunity'] = pd.Series(breakout_opportunity, index=market_data.index).fillna(0.0)

        # Add breakout direction probabilities
        labels['resistance_breakout_prob'] = pd.Series(resistance_strength, index=market_data.index).clip(0, 1).fillna(0.0)
        labels['support_breakout_prob'] = pd.Series(support_strength, index=market_data.index).clip(0, 1).fillna(0.0)

        # Consolidation detection (low volatility before breakout)
        volatility = prices.pct_change().rolling(10).std()
        consolidation = (volatility < volatility.rolling(20).quantile(0.3)).fillna(False)
        labels['consolidation_signal'] = consolidation.astype(float)

        confidence_scores = self.calculate_confidence(labels, market_data)
        performance_metrics = self._calculate_performance_metrics(labels, market_data)

        return StrategyResult(
            strategy=LabelingStrategy.BREAKOUT_FOCUSED,
            labels=labels,
            confidence_scores=confidence_scores,
            performance_metrics=performance_metrics,
            metadata={
                'breakout_window': self.breakout_window,
                'breakout_threshold': self.breakout_threshold
            }
        )

    def calculate_confidence(self, labels: pd.DataFrame, market_data: pd.DataFrame) -> pd.Series:
        """Calculate confidence based on breakout strength and consolidation."""
        if 'breakout_opportunity' in labels.columns and 'consolidation_signal' in labels.columns:
            opportunity = labels['breakout_opportunity']
            consolidation = labels['consolidation_signal']
            
            # Higher confidence when breakout is strong and preceded by consolidation
            confidence = opportunity * (0.5 + 0.5 * consolidation)
            return confidence.clip(0.1, 1.0)

        return pd.Series(0.5, index=labels.index)

    def _calculate_performance_metrics(self, labels: pd.DataFrame, market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics."""
        metrics = {}

        if 'breakout_opportunity' in labels.columns and 'close' in market_data.columns:
            opportunity = labels['breakout_opportunity']
            returns = market_data['close'].pct_change().shift(-1).fillna(0)

            common_idx = opportunity.index.intersection(returns.index)
            if len(common_idx) > 10:
                corr = np.corrcoef(opportunity.loc[common_idx], returns.loc[common_idx])[0, 1]
                metrics['correlation'] = abs(corr) if not np.isnan(corr) else 0.0

        return metrics

class ConservativeStrategy(BaseLabelingStrategy):
    """Conservative labeling strategy with lower risk tolerance."""

    def __init__(self, risk_threshold: float = 0.01, min_confidence: float = 0.7):
        """Initialize conservative strategy."""
        self.risk_threshold = risk_threshold
        self.min_confidence = min_confidence

    def generate_labels(self, market_data: pd.DataFrame) -> StrategyResult:
        """Generate conservative labels."""
        if 'close' not in market_data.columns:
            empty_labels = pd.DataFrame(index=market_data.index)
            empty_labels['conservative_opportunity'] = 0.0
            return StrategyResult(
                strategy=LabelingStrategy.CONSERVATIVE,
                labels=empty_labels,
                confidence_scores=pd.Series(0.0, index=market_data.index),
                performance_metrics={},
                metadata={}
            )

        prices = market_data['close']
        labels = pd.DataFrame(index=market_data.index)

        # Calculate volatility and trend
        returns = prices.pct_change()
        volatility = returns.rolling(20).std()
        trend = prices.rolling(50).apply(lambda x: (x.iloc[-1] / x.iloc[0]) - 1)

        # Conservative opportunity: only when conditions are very favorable
        low_vol = volatility < self.risk_threshold
        strong_trend = abs(trend) > 0.05
        stable_conditions = (volatility / volatility.rolling(50).mean()) < 1.2

        # Conservative opportunity score
        conservative_opportunity = pd.Series(0.0, index=market_data.index)
        
        # Only signal when all conditions are met
        favorable_conditions = low_vol & strong_trend & stable_conditions
        conservative_opportunity[favorable_conditions] = 0.6  # Moderate opportunity even in best conditions

        labels['conservative_opportunity'] = conservative_opportunity.fillna(0.0)

        # Add conservative-specific features
        labels['risk_score'] = (volatility / self.risk_threshold).clip(0, 2)
        labels['trend_strength'] = abs(trend).clip(0, 1)
        labels['stability_score'] = (1.0 / (volatility / volatility.rolling(50).mean() + 1e-10)).clip(0, 1)

        confidence_scores = self.calculate_confidence(labels, market_data)
        performance_metrics = self._calculate_performance_metrics(labels, market_data)

        return StrategyResult(
            strategy=LabelingStrategy.CONSERVATIVE,
            labels=labels,
            confidence_scores=confidence_scores,
            performance_metrics=performance_metrics,
            metadata={
                'risk_threshold': self.risk_threshold,
                'min_confidence': self.min_confidence
            }
        )

    def calculate_confidence(self, labels: pd.DataFrame, market_data: pd.DataFrame) -> pd.Series:
        """Calculate confidence based on conservative criteria."""
        if 'conservative_opportunity' in labels.columns and 'stability_score' in labels.columns:
            opportunity = labels['conservative_opportunity']
            stability = labels['stability_score']
            
            # High confidence only when opportunity exists and conditions are stable
            confidence = np.where(
                opportunity > 0,
                np.minimum(opportunity + stability * 0.3, 1.0),
                0.1
            )
            return pd.Series(confidence, index=labels.index).clip(0.1, 1.0)

        return pd.Series(0.5, index=labels.index)

    def _calculate_performance_metrics(self, labels: pd.DataFrame, market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics."""
        metrics = {}

        if 'conservative_opportunity' in labels.columns and 'close' in market_data.columns:
            opportunity = labels['conservative_opportunity']
            returns = market_data['close'].pct_change().shift(-1).fillna(0)

            common_idx = opportunity.index.intersection(returns.index)
            if len(common_idx) > 10:
                corr = np.corrcoef(opportunity.loc[common_idx], returns.loc[common_idx])[0, 1]
                metrics['correlation'] = abs(corr) if not np.isnan(corr) else 0.0

        return metrics

class AggressiveStrategy(BaseLabelingStrategy):
    """Aggressive labeling strategy with higher risk tolerance."""

    def __init__(self, risk_multiplier: float = 2.0, min_opportunity: float = 0.3):
        """Initialize aggressive strategy."""
        self.risk_multiplier = risk_multiplier
        self.min_opportunity = min_opportunity

    def generate_labels(self, market_data: pd.DataFrame) -> StrategyResult:
        """Generate aggressive labels."""
        if 'close' not in market_data.columns:
            empty_labels = pd.DataFrame(index=market_data.index)
            empty_labels['aggressive_opportunity'] = 0.0
            return StrategyResult(
                strategy=LabelingStrategy.AGGRESSIVE,
                labels=empty_labels,
                confidence_scores=pd.Series(0.0, index=market_data.index),
                performance_metrics={},
                metadata={}
            )

        prices = market_data['close']
        labels = pd.DataFrame(index=market_data.index)

        # Calculate various opportunity signals
        returns = prices.pct_change()
        volatility = returns.rolling(10).std()
        momentum = prices / prices.shift(5) - 1
        trend = prices.rolling(20).apply(lambda x: (x.iloc[-1] / x.iloc[0]) - 1)

        # Aggressive opportunity: amplify signals and take more risks
        momentum_opportunity = np.tanh(abs(momentum) * self.risk_multiplier * 5) * 0.9
        volatility_opportunity = np.tanh(volatility * self.risk_multiplier * 20) * 0.8
        trend_opportunity = np.tanh(abs(trend) * self.risk_multiplier * 3) * 0.7

        # Combine opportunities with aggressive weighting
        combined_opportunity = (momentum_opportunity * 0.4 + 
                              volatility_opportunity * 0.3 + 
                              trend_opportunity * 0.3)

        # Apply minimum threshold
        aggressive_opportunity = np.where(
            combined_opportunity >= self.min_opportunity,
            combined_opportunity,
            0.0
        )

        labels['aggressive_opportunity'] = pd.Series(aggressive_opportunity, index=market_data.index).fillna(0.0)

        # Add aggressive-specific features
        labels['momentum_signal'] = momentum_opportunity
        labels['volatility_signal'] = volatility_opportunity
        labels['trend_signal'] = trend_opportunity
        labels['risk_appetite'] = (volatility * self.risk_multiplier).clip(0, 2)

        confidence_scores = self.calculate_confidence(labels, market_data)
        performance_metrics = self._calculate_performance_metrics(labels, market_data)

        return StrategyResult(
            strategy=LabelingStrategy.AGGRESSIVE,
            labels=labels,
            confidence_scores=confidence_scores,
            performance_metrics=performance_metrics,
            metadata={
                'risk_multiplier': self.risk_multiplier,
                'min_opportunity': self.min_opportunity
            }
        )

    def calculate_confidence(self, labels: pd.DataFrame, market_data: pd.DataFrame) -> pd.Series:
        """Calculate confidence based on signal strength."""
        if 'aggressive_opportunity' in labels.columns:
            opportunity = labels['aggressive_opportunity']
            # Higher confidence for stronger signals
            confidence = np.where(
                opportunity > 0,
                np.minimum(opportunity * 1.2, 1.0),  # Boost confidence for aggressive signals
                0.1
            )
            return pd.Series(confidence, index=labels.index).clip(0.1, 1.0)

        return pd.Series(0.5, index=labels.index)

    def _calculate_performance_metrics(self, labels: pd.DataFrame, market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics."""
        metrics = {}

        if 'aggressive_opportunity' in labels.columns and 'close' in market_data.columns:
            opportunity = labels['aggressive_opportunity']
            returns = market_data['close'].pct_change().shift(-1).fillna(0)

            common_idx = opportunity.index.intersection(returns.index)
            if len(common_idx) > 10:
                corr = np.corrcoef(opportunity.loc[common_idx], returns.loc[common_idx])[0, 1]
                metrics['correlation'] = abs(corr) if not np.isnan(corr) else 0.0

        return metrics

class EnsembleCombiner:
    """Combines results from multiple labeling strategies."""

    def __init__(self, method: EnsembleCombinationMethod = EnsembleCombinationMethod.PERFORMANCE_WEIGHTED):
        """Initialize ensemble combiner."""
        self.method = method
        self.logger = get_logger('EnsembleCombiner')

        # State for adaptive methods
        self.strategy_weights: Dict[LabelingStrategy, float] = {}
        self.performance_history: Dict[LabelingStrategy, List[float]] = {}

    def combine_labels(self,
                      strategy_results: Dict[LabelingStrategy, StrategyResult],
                      market_data: pd.DataFrame,
                      config: EnsembleLabelingConfig) -> EnsembleResult:
        """
        Combine labels from multiple strategies.

        Args:
            strategy_results: Results from individual strategies
            market_data: Original market data
            config: Ensemble configuration

        Returns:
            EnsembleResult with combined labels
        """
        if not strategy_results:
            return self._create_empty_result()

        # Determine combination weights
        combination_weights = self._calculate_combination_weights(strategy_results, config)

        # Combine labels using selected method
        if self.method == EnsembleCombinationMethod.SIMPLE_AVERAGE:
            ensemble_labels = self._simple_average_combination(strategy_results)
        elif self.method == EnsembleCombinationMethod.WEIGHTED_AVERAGE:
            ensemble_labels = self._weighted_average_combination(strategy_results, combination_weights)
        elif self.method == EnsembleCombinationMethod.PERFORMANCE_WEIGHTED:
            ensemble_labels = self._performance_weighted_combination(strategy_results, combination_weights)
        elif self.method == EnsembleCombinationMethod.CONFIDENCE_WEIGHTED:
            ensemble_labels = self._confidence_weighted_combination(strategy_results)
        elif self.method == EnsembleCombinationMethod.STACKING:
            ensemble_labels = self._stacking_combination(strategy_results, market_data, config)
        elif self.method == EnsembleCombinationMethod.DYNAMIC_SELECTION:
            ensemble_labels = self._dynamic_selection_combination(strategy_results, config)
        else:
            ensemble_labels = self._simple_average_combination(strategy_results)

        # Calculate ensemble confidence
        ensemble_confidence = self._calculate_ensemble_confidence(strategy_results, combination_weights)

        # Calculate diversity score
        diversity_score = self._calculate_diversity_score(strategy_results, config.diversity_measure)

        # Calculate ensemble performance metrics
        performance_metrics = self._calculate_ensemble_performance(ensemble_labels, market_data)

        return EnsembleResult(
            ensemble_labels=ensemble_labels,
            strategy_results=strategy_results,
            combination_weights=combination_weights,
            ensemble_confidence=ensemble_confidence,
            diversity_score=diversity_score,
            performance_metrics=performance_metrics,
            metadata={
                'combination_method': self.method.value,
                'n_strategies': len(strategy_results)
            }
        )

    def _calculate_combination_weights(self,
                                     strategy_results: Dict[LabelingStrategy, StrategyResult],
                                     config: EnsembleLabelingConfig) -> Dict[LabelingStrategy, float]:
        """Calculate combination weights for strategies."""
        if self.method == EnsembleCombinationMethod.SIMPLE_AVERAGE:
            # Equal weights
            n_strategies = len(strategy_results)
            return {strategy: 1.0 / n_strategies for strategy in strategy_results.keys()}

        elif self.method == EnsembleCombinationMethod.PERFORMANCE_WEIGHTED:
            # Weight by performance metrics
            weights = {}
            total_performance = 0.0

            for strategy, result in strategy_results.items():
                # Use correlation as primary performance metric
                performance = result.performance_metrics.get('correlation', 0.0)
                # Add small base weight to avoid zero weights
                performance = max(performance, config.min_weight)
                weights[strategy] = performance
                total_performance += performance

            # Normalize weights
            if total_performance > 0:
                weights = {k: v / total_performance for k, v in weights.items()}

            return weights

        else:
            # Default to equal weights
            n_strategies = len(strategy_results)
            return {strategy: 1.0 / n_strategies for strategy in strategy_results.keys()}

    def _simple_average_combination(self, strategy_results: Dict[LabelingStrategy, StrategyResult]) -> pd.DataFrame:
        """Combine using simple average."""
        if not strategy_results:
            return pd.DataFrame()

        # Find common columns across all strategies
        all_columns = set()
        for result in strategy_results.values():
            all_columns.update(result.labels.columns)

        # Get common index
        common_index = None
        for result in strategy_results.values():
            if common_index is None:
                common_index = result.labels.index
            else:
                common_index = common_index.intersection(result.labels.index)

        if common_index is None or len(common_index) == 0:
            return pd.DataFrame()

        # Combine labels
        ensemble_labels = pd.DataFrame(index=common_index)

        for col in all_columns:
            col_values = []
            for strategy, result in strategy_results.items():
                if col in result.labels.columns:
                    values = result.labels[col].reindex(common_index).fillna(0)
                    col_values.append(values)

            if col_values:
                ensemble_labels[f'ensemble_{col}'] = pd.concat(col_values, axis=1).mean(axis=1)

        # Add overall ensemble opportunity
        opportunity_cols = [col for col in ensemble_labels.columns if 'opportunity' in col]
        if opportunity_cols:
            ensemble_labels['ensemble_opportunity'] = ensemble_labels[opportunity_cols].mean(axis=1)

        return ensemble_labels

    def _weighted_average_combination(self,
                                    strategy_results: Dict[LabelingStrategy, StrategyResult],
                                    weights: Dict[LabelingStrategy, float]) -> pd.DataFrame:
        """Combine using weighted average."""
        if not strategy_results:
            return pd.DataFrame()

        # Find common columns
        all_columns = set()
        for result in strategy_results.values():
            all_columns.update(result.labels.columns)

        # Get common index
        common_index = None
        for result in strategy_results.values():
            if common_index is None:
                common_index = result.labels.index
            else:
                common_index = common_index.intersection(result.labels.index)

        if common_index is None or len(common_index) == 0:
            return pd.DataFrame()

        # Combine labels with weights
        ensemble_labels = pd.DataFrame(index=common_index)

        for col in all_columns:
            weighted_values = []
            total_weight = 0.0

            for strategy, result in strategy_results.items():
                if col in result.labels.columns:
                    weight = weights.get(strategy, 0.0)
                    values = result.labels[col].reindex(common_index).fillna(0) * weight
                    weighted_values.append(values)
                    total_weight += weight

            if weighted_values and total_weight > 0:
                ensemble_labels[f'ensemble_{col}'] = sum(weighted_values) / total_weight

        # Add overall ensemble opportunity
        opportunity_cols = [col for col in ensemble_labels.columns if 'opportunity' in col]
        if opportunity_cols:
            ensemble_labels['ensemble_opportunity'] = ensemble_labels[opportunity_cols].mean(axis=1)

        return ensemble_labels

    def _performance_weighted_combination(self,
                                        strategy_results: Dict[LabelingStrategy, StrategyResult],
                                        weights: Dict[LabelingStrategy, float]) -> pd.DataFrame:
        """Combine using performance-based weights (same as weighted average)."""
        return self._weighted_average_combination(strategy_results, weights)

    def _confidence_weighted_combination(self, strategy_results: Dict[LabelingStrategy, StrategyResult]) -> pd.DataFrame:
        """Combine using confidence-based weights."""
        if not strategy_results:
            return pd.DataFrame()

        # Get common index
        common_index = None
        for result in strategy_results.values():
            if common_index is None:
                common_index = result.labels.index
            else:
                common_index = common_index.intersection(result.labels.index)

        if common_index is None or len(common_index) == 0:
            return pd.DataFrame()

        ensemble_labels = pd.DataFrame(index=common_index)

        # For each time point, weight by confidence
        for col in ['opportunity']:  # Focus on main opportunity columns
            confidence_weighted_values = pd.DataFrame(index=common_index)
            confidence_weights = pd.DataFrame(index=common_index)

            for strategy, result in strategy_results.items():
                # Find relevant columns
                relevant_cols = [c for c in result.labels.columns if col in c.lower()]
                if relevant_cols:
                    main_col = relevant_cols[0]  # Use first relevant column
                    values = result.labels[main_col].reindex(common_index).fillna(0)
                    confidence = result.confidence_scores.reindex(common_index).fillna(0.1)

                    confidence_weighted_values[strategy.value] = values * confidence
                    confidence_weights[strategy.value] = confidence

            # Calculate weighted average
            if not confidence_weighted_values.empty and not confidence_weights.empty:
                total_weights = confidence_weights.sum(axis=1)
                total_weights = total_weights.replace(0, 1)  # Avoid division by zero

                ensemble_labels[f'ensemble_{col}'] = confidence_weighted_values.sum(axis=1) / total_weights

        return ensemble_labels

    def _stacking_combination(self,
                            strategy_results: Dict[LabelingStrategy, StrategyResult],
                            market_data: pd.DataFrame,
                            config: EnsembleLabelingConfig) -> pd.DataFrame:
        """Combine using stacking (meta-learning)."""
        if not strategy_results or 'close' not in market_data.columns:
            return self._simple_average_combination(strategy_results)

        try:
            # Prepare features (predictions from base strategies)
            features_df = pd.DataFrame()

            for strategy, result in strategy_results.items():
                # Use main opportunity column as feature
                opportunity_cols = [col for col in result.labels.columns if 'opportunity' in col]
                if opportunity_cols:
                    main_col = opportunity_cols[0]
                    features_df[f'{strategy.value}_pred'] = result.labels[main_col]

            if features_df.empty:
                return self._simple_average_combination(strategy_results)

            # Prepare target (future returns)
            returns = market_data['close'].pct_change().shift(-1).fillna(0)

            # Align data
            common_idx = features_df.index.intersection(returns.index)
            if len(common_idx) < 50:  # Need sufficient data for stacking
                return self._simple_average_combination(strategy_results)

            X = features_df.loc[common_idx].fillna(0)
            y = returns.loc[common_idx]

            # Train meta-learner
            if config.stacking_meta_learner == "ridge":
                meta_learner = Ridge(alpha=1.0)
            elif config.stacking_meta_learner == "tree":
                meta_learner = DecisionTreeRegressor(max_depth=3, random_state=42)
            else:
                meta_learner = LinearRegression()

            # Cross-validation to avoid overfitting
            from sklearn.model_selection import cross_val_predict
            meta_predictions = cross_val_predict(meta_learner, X, y, cv=config.stacking_cv_folds)

            # Create ensemble result
            ensemble_labels = pd.DataFrame(index=common_idx)
            ensemble_labels['ensemble_opportunity'] = pd.Series(meta_predictions, index=common_idx).clip(0, 1)

            return ensemble_labels

        except Exception as e:
            self.logger.warning(f'Stacking combination failed: {e}')
            return self._simple_average_combination(strategy_results)

    def _dynamic_selection_combination(self,
                                     strategy_results: Dict[LabelingStrategy, StrategyResult],
                                     config: EnsembleLabelingConfig) -> pd.DataFrame:
        """Combine using dynamic strategy selection."""
        if not strategy_results:
            return pd.DataFrame()

        # Select top k strategies based on recent performance
        strategy_performance = {}
        for strategy, result in strategy_results.items():
            performance = result.performance_metrics.get('correlation', 0.0)
            strategy_performance[strategy] = performance

        # Select top k strategies
        top_strategies = sorted(strategy_performance.items(), key=lambda x: x[1], reverse=True)
        selected_strategies = {k: v for k, v in top_strategies[:config.dynamic_selection_k]}

        # Combine selected strategies using weighted average
        selected_results = {k: strategy_results[k] for k in selected_strategies.keys()}
        weights = {k: v for k, v in selected_strategies.items()}

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        return self._weighted_average_combination(selected_results, weights)

    def _calculate_ensemble_confidence(self,
                                     strategy_results: Dict[LabelingStrategy, StrategyResult],
                                     weights: Dict[LabelingStrategy, float]) -> pd.Series:
        """Calculate ensemble confidence scores."""
        if not strategy_results:
            return pd.Series(dtype=float)

        # Get common index
        common_index = None
        for result in strategy_results.values():
            if common_index is None:
                common_index = result.confidence_scores.index
            else:
                common_index = common_index.intersection(result.confidence_scores.index)

        if common_index is None or len(common_index) == 0:
            return pd.Series(dtype=float)

        # Weight confidence scores
        weighted_confidences = []
        total_weight = 0.0

        for strategy, result in strategy_results.items():
            weight = weights.get(strategy, 0.0)
            confidence = result.confidence_scores.reindex(common_index).fillna(0.5)
            weighted_confidences.append(confidence * weight)
            total_weight += weight

        if weighted_confidences and total_weight > 0:
            ensemble_confidence = sum(weighted_confidences) / total_weight
        else:
            ensemble_confidence = pd.Series(0.5, index=common_index)

        return ensemble_confidence

    def _calculate_diversity_score(self,
                                 strategy_results: Dict[LabelingStrategy, StrategyResult],
                                 diversity_measure: DiversityMeasure) -> float:
        """Calculate diversity score for the ensemble."""
        if len(strategy_results) < 2:
            return 0.0

        try:
            # Extract main opportunity predictions from each strategy
            predictions = {}
            for strategy, result in strategy_results.items():
                opportunity_cols = [col for col in result.labels.columns if 'opportunity' in col]
                if opportunity_cols:
                    predictions[strategy] = result.labels[opportunity_cols[0]].fillna(0)

            if len(predictions) < 2:
                return 0.0

            # Calculate pairwise correlations
            correlations = []
            strategies = list(predictions.keys())

            for i in range(len(strategies)):
                for j in range(i + 1, len(strategies)):
                    pred1 = predictions[strategies[i]]
                    pred2 = predictions[strategies[j]]

                    # Align predictions
                    common_idx = pred1.index.intersection(pred2.index)
                    if len(common_idx) > 10:
                        corr = np.corrcoef(pred1.loc[common_idx], pred2.loc[common_idx])[0, 1]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))

            if not correlations:
                return 0.0

            # Diversity is inverse of average correlation
            avg_correlation = np.mean(correlations)
            diversity_score = 1.0 - avg_correlation

            return max(0.0, min(1.0, diversity_score))

        except Exception:
            return 0.0

    def _calculate_ensemble_performance(self, ensemble_labels: pd.DataFrame, market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics for ensemble."""
        metrics = {}

        if 'ensemble_opportunity' in ensemble_labels.columns and 'close' in market_data.columns:
            opportunity = ensemble_labels['ensemble_opportunity']
            returns = market_data['close'].pct_change().shift(-1).fillna(0)

            common_idx = opportunity.index.intersection(returns.index)
            if len(common_idx) > 10:
                # Correlation
                corr = np.corrcoef(opportunity.loc[common_idx], returns.loc[common_idx])[0, 1]
                metrics['correlation'] = abs(corr) if not np.isnan(corr) else 0.0

                # Hit rate
                signals = (opportunity.loc[common_idx] > 0.5).astype(int)
                if signals.sum() > 0:
                    hit_rate = ((signals == 1) & (returns.loc[common_idx] > 0)).sum() / signals.sum()
                    metrics['hit_rate'] = hit_rate

                # Sharpe-like ratio
                strategy_returns = signals * returns.loc[common_idx]
                if strategy_returns.std() > 0:
                    metrics['sharpe_ratio'] = strategy_returns.mean() / strategy_returns.std()

        return metrics

    def _create_empty_result(self) -> EnsembleResult:
        """Create empty ensemble result."""
        return EnsembleResult(
            ensemble_labels=pd.DataFrame(),
            strategy_results={},
            combination_weights={},
            ensemble_confidence=pd.Series(dtype=float),
            diversity_score=0.0,
            performance_metrics={},
            metadata={}
        )

class EnsembleLabelingSystem:
    """
    Main ensemble labeling system that coordinates multiple strategies.

    This class manages multiple labeling strategies and combines their results
    using various ensemble methods for improved robustness and accuracy.
    """

    def __init__(self, config: Optional[EnsembleLabelingConfig] = None):
        """Initialize ensemble labeling system."""
        self.config = config or EnsembleLabelingConfig()
        self.logger = get_logger('EnsembleLabelingSystem')

        # Initialize strategies
        self.strategies: Dict[LabelingStrategy, BaseLabelingStrategy] = {}
        self._initialize_strategies()

        # Initialize combiner
        self.combiner = EnsembleCombiner(self.config.combination_method)

        # Performance tracking
        self.ensemble_history: List[EnsembleResult] = []

        self.logger.info('🎭 Ensemble Labeling System initialized')
        self.logger.info(f'   → Enabled strategies: {[s.value for s in self.strategies.keys()]}')
        self.logger.info(f'   → Combination method: {self.config.combination_method.value}')

    def _initialize_strategies(self):
        """Initialize individual labeling strategies."""
        for strategy_type in self.config.enabled_strategies:
            try:
                if strategy_type == LabelingStrategy.MULTI_HORIZON:
                    self.strategies[strategy_type] = MultiHorizonStrategy()
                elif strategy_type == LabelingStrategy.VOLATILITY_ADJUSTED:
                    self.strategies[strategy_type] = VolatilityAdjustedStrategy()
                elif strategy_type == LabelingStrategy.MOMENTUM_BASED:
                    self.strategies[strategy_type] = MomentumBasedStrategy()
                elif strategy_type == LabelingStrategy.MEAN_REVERSION:
                    self.strategies[strategy_type] = MeanReversionStrategy()
                elif strategy_type == LabelingStrategy.REGIME_AWARE:
                    self.strategies[strategy_type] = RegimeAwareStrategy()
                elif strategy_type == LabelingStrategy.ML_PREDICTIVE:
                    self.strategies[strategy_type] = MLPredictiveStrategy()
                elif strategy_type == LabelingStrategy.BREAKOUT_FOCUSED:
                    self.strategies[strategy_type] = BreakoutFocusedStrategy()
                elif strategy_type == LabelingStrategy.CONSERVATIVE:
                    self.strategies[strategy_type] = ConservativeStrategy()
                elif strategy_type == LabelingStrategy.AGGRESSIVE:
                    self.strategies[strategy_type] = AggressiveStrategy()

            except Exception as e:
                self.logger.warning(f'Failed to initialize {strategy_type.value}: {e}')

    def generate_ensemble_labels(self, market_data: pd.DataFrame) -> EnsembleResult:
        """
        Generate ensemble labels using all enabled strategies.

        Args:
            market_data: OHLCV market data

        Returns:
            EnsembleResult with combined labels and metadata
        """
        self.logger.info('🚀 Generating ensemble labels')

        if len(market_data) < 50:
            self.logger.warning('⚠️ Insufficient data for ensemble labeling')
            return self.combiner._create_empty_result()

        # Generate labels from each strategy
        strategy_results = {}

        for strategy_type, strategy in self.strategies.items():
            try:
                self.logger.info(f'   → Running {strategy_type.value} strategy')
                result = strategy.generate_labels(market_data.copy())
                strategy_results[strategy_type] = result

            except Exception as e:
                self.logger.error(f'Strategy {strategy_type.value} failed: {e}')

        if not strategy_results:
            self.logger.error('❌ No strategies produced results')
            return self.combiner._create_empty_result()

        # Combine results
        ensemble_result = self.combiner.combine_labels(strategy_results, market_data, self.config)

        # Store in history
        self.ensemble_history.append(ensemble_result)
        if len(self.ensemble_history) > 100:  # Keep limited history
            self.ensemble_history = self.ensemble_history[-100:]

        self.logger.info('✅ Ensemble labeling completed')
        self.logger.info(f'   → Combined {len(strategy_results)} strategies')
        self.logger.info(f'   → Diversity score: {ensemble_result.diversity_score:.3f}')

        return ensemble_result

    def optimize_ensemble_weights(self, market_data: pd.DataFrame, target_returns: pd.Series) -> Dict[LabelingStrategy, float]:
        """Optimize ensemble weights based on historical performance."""
        if not self.ensemble_history:
            self.logger.warning('No ensemble history for weight optimization')
            return {}

        # This would implement weight optimization logic
        # For now, return current weights
        if self.ensemble_history:
            return self.ensemble_history[-1].combination_weights

        return {}

    def get_ensemble_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of the ensemble system."""
        if not self.ensemble_history:
            return {}

        recent_results = self.ensemble_history[-10:]  # Last 10 results

        summary = {
            'average_diversity': np.mean([r.diversity_score for r in recent_results]),
            'average_correlation': np.mean([r.performance_metrics.get('correlation', 0) for r in recent_results]),
            'average_hit_rate': np.mean([r.performance_metrics.get('hit_rate', 0) for r in recent_results]),
            'strategy_weights': recent_results[-1].combination_weights if recent_results else {},
            'n_strategies': len(recent_results[-1].strategy_results) if recent_results else 0
        }

        return summary

    def save_ensemble_state(self, output_path: Union[str, Path]):
        """Save ensemble state to disk."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        state_data = {
            'config': {
                'enabled_strategies': [s.value for s in self.config.enabled_strategies],
                'combination_method': self.config.combination_method.value,
                'target_diversity': self.config.target_diversity
            },
            'recent_performance': self.get_ensemble_performance_summary(),
            'ensemble_history_count': len(self.ensemble_history)
        }

        with open(output_path, 'w') as f:
            json.dump(state_data, f, indent=2)

        self.logger.info(f'💾 Ensemble state saved to {output_path}')

# Convenience functions
def create_ensemble_labeling_system(config: Optional[EnsembleLabelingConfig] = None) -> EnsembleLabelingSystem:
    """Create ensemble labeling system."""
    return EnsembleLabelingSystem(config)

def generate_ensemble_labels(market_data: pd.DataFrame,
                           strategies: Optional[List[LabelingStrategy]] = None,
                           combination_method: EnsembleCombinationMethod = EnsembleCombinationMethod.PERFORMANCE_WEIGHTED) -> EnsembleResult:
    """Convenience function to generate ensemble labels."""
    config = EnsembleLabelingConfig()
    if strategies:
        config.enabled_strategies = strategies
    config.combination_method = combination_method

    system = EnsembleLabelingSystem(config)
    return system.generate_ensemble_labels(market_data)
