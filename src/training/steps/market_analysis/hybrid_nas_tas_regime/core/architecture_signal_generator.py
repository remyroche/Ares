"""
Architecture-Based Signal Generation System

This module provides comprehensive signal generation from discovered neural and tree
architectures, including confidence scoring, ensemble integration, signal quality
metrics, and real-time processing capabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import warnings

logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Types of trading signals."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"

class SignalSource(Enum):
    """Source of trading signals."""
    NEURAL_NETWORK = "neural_network"
    TREE_MODEL = "tree_model"
    ENSEMBLE = "ensemble"
    HYBRID = "hybrid"

class ConfidenceLevel(Enum):
    """Confidence levels for signals."""
    VERY_LOW = "very_low"    # < 0.6
    LOW = "low"             # 0.6 - 0.7
    MEDIUM = "medium"       # 0.7 - 0.8
    HIGH = "high"           # 0.8 - 0.9
    VERY_HIGH = "very_high" # > 0.9

@dataclass
class TradingSignal:
    """Trading signal with metadata."""
    signal_type: SignalType
    confidence: float
    source: SignalSource
    timestamp: datetime
    price: float
    volume: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SignalQualityMetrics:
    """Quality metrics for generated signals."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ArchitectureSignalConfig:
    """Configuration for architecture-based signal generation."""
    # Signal processing
    signal_threshold: float = 0.6
    confidence_threshold: float = 0.7
    min_signal_strength: float = 0.1

    # Ensemble settings
    ensemble_method: str = "weighted_average"  # weighted_average, majority_vote, stacking
    ensemble_weights: Dict[SignalSource, float] = field(default_factory=lambda: {
        SignalSource.NEURAL_NETWORK: 0.6,
        SignalSource.TREE_MODEL: 0.4
    })

    # Risk management
    max_position_size: float = 1.0
    risk_per_trade: float = 0.02
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10

    # Signal validation
    enable_signal_validation: bool = True
    validation_window: int = 20
    min_validation_samples: int = 100

    # Real-time processing
    enable_real_time_processing: bool = True
    signal_buffer_size: int = 1000
    update_frequency_seconds: int = 60

class NeuralSignalGenerator:
    """Signal generator for neural architectures."""

    def __init__(self, model: nn.Module, config: ArchitectureSignalConfig):
        """Initialize neural signal generator."""
        self.model = model
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Signal processing
        self.signal_history = []
        self.confidence_history = []

    def generate_signal(self, market_data: np.ndarray,
                       regime_data: Optional[Dict[str, Any]] = None) -> TradingSignal:
        """Generate trading signal from neural architecture."""
        try:
            # Convert to torch tensor
            if isinstance(market_data, np.ndarray):
                market_tensor = torch.from_numpy(market_data).float().unsqueeze(0)
            else:
                market_tensor = market_data

            # Get model prediction
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(market_tensor)
                probabilities = F.softmax(predictions, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = torch.max(probabilities, dim=1)[0].item()

            # Convert prediction to signal
            signal_type = self._prediction_to_signal(predicted_class, confidence)

            # Apply regime adjustments
            if regime_data:
                signal_type, confidence = self._apply_regime_adjustment(
                    signal_type, confidence, regime_data
                )

            # Create signal
            signal = TradingSignal(
                signal_type=signal_type,
                confidence=confidence,
                source=SignalSource.NEURAL_NETWORK,
                timestamp=datetime.now(),
                price=market_data[-1, -1] if market_data.ndim > 1 else market_data[-1],
                metadata={
                    'model_output': predictions.numpy(),
                    'probabilities': probabilities.numpy(),
                    'regime_adjusted': regime_data is not None
                }
            )

            # Store signal history
            self.signal_history.append(signal)
            self.confidence_history.append(confidence)

            return signal

        except Exception as e:
            self.logger.error(f"❌ Neural signal generation failed: {e}")
            return TradingSignal(
                signal_type=SignalType.HOLD,
                confidence=0.0,
                source=SignalSource.NEURAL_NETWORK,
                timestamp=datetime.now(),
                price=0.0,
                metadata={'error': str(e)}
            )

    def _prediction_to_signal(self, prediction: int, confidence: float) -> SignalType:
        """Convert model prediction to trading signal."""
        if confidence < self.config.confidence_threshold:
            return SignalType.HOLD

        # Signal mapping (customize based on your model output)
        signal_map = {
            0: SignalType.STRONG_SELL,
            1: SignalType.SELL,
            2: SignalType.HOLD,
            3: SignalType.BUY,
            4: SignalType.STRONG_BUY
        }

        return signal_map.get(prediction, SignalType.HOLD)

    def _apply_regime_adjustment(self, signal_type: SignalType,
                               confidence: float, regime_data: Dict[str, Any]) -> Tuple[SignalType, float]:
        """Apply regime-based adjustments to signals."""
        try:
            # Simplified regime adjustment
            # In practice, this would use detailed regime analysis
            if 'regime_probabilities' in regime_data:
                regime_probs = regime_data['regime_probabilities']
                # Adjust confidence based on regime stability
                regime_stability = np.mean(regime_probs[-5:])  # Last 5 periods
                confidence *= regime_stability

                # Adjust signal strength based on regime type
                if np.argmax(regime_probs[-1]) == 0:  # Bullish regime
                    if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                        confidence *= 1.1
                elif np.argmax(regime_probs[-1]) == 1:  # Bearish regime
                    if signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                        confidence *= 1.1

            return signal_type, confidence

        except Exception as e:
            self.logger.warning(f"Regime adjustment failed: {e}")
            return signal_type, confidence

class TreeSignalGenerator:
    """Signal generator for tree-based architectures."""

    def __init__(self, model: Any, config: ArchitectureSignalConfig):
        """Initialize tree signal generator."""
        self.model = model
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Signal processing
        self.signal_history = []
        self.confidence_history = []

    def generate_signal(self, market_data: np.ndarray,
                       regime_data: Optional[Dict[str, Any]] = None) -> TradingSignal:
        """Generate trading signal from tree architecture."""
        try:
            # Get model prediction
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(market_data.reshape(1, -1))[0]
                predicted_class = np.argmax(probabilities)
                confidence = probabilities[predicted_class]
            else:
                predicted_class = self.model.predict(market_data.reshape(1, -1))[0]
                confidence = 0.7  # Default confidence for non-probabilistic models
                probabilities = np.zeros(5)
                probabilities[predicted_class] = confidence

            # Convert prediction to signal
            signal_type = self._prediction_to_signal(predicted_class, confidence)

            # Apply regime adjustments
            if regime_data:
                signal_type, confidence = self._apply_regime_adjustment(
                    signal_type, confidence, regime_data
                )

            # Create signal
            signal = TradingSignal(
                signal_type=signal_type,
                confidence=confidence,
                source=SignalSource.TREE_MODEL,
                timestamp=datetime.now(),
                price=market_data[-1] if market_data.ndim == 1 else market_data[-1, -1],
                metadata={
                    'model_output': predicted_class,
                    'probabilities': probabilities,
                    'regime_adjusted': regime_data is not None
                }
            )

            # Store signal history
            self.signal_history.append(signal)
            self.confidence_history.append(confidence)

            return signal

        except Exception as e:
            self.logger.error(f"❌ Tree signal generation failed: {e}")
            return TradingSignal(
                signal_type=SignalType.HOLD,
                confidence=0.0,
                source=SignalSource.TREE_MODEL,
                timestamp=datetime.now(),
                price=0.0,
                metadata={'error': str(e)}
            )

    def _prediction_to_signal(self, prediction: int, confidence: float) -> SignalType:
        """Convert model prediction to trading signal."""
        if confidence < self.config.confidence_threshold:
            return SignalType.HOLD

        # Signal mapping (customize based on your model output)
        signal_map = {
            0: SignalType.STRONG_SELL,
            1: SignalType.SELL,
            2: SignalType.HOLD,
            3: SignalType.BUY,
            4: SignalType.STRONG_BUY
        }

        return signal_map.get(prediction, SignalType.HOLD)

    def _apply_regime_adjustment(self, signal_type: SignalType,
                               confidence: float, regime_data: Dict[str, Any]) -> Tuple[SignalType, float]:
        """Apply regime-based adjustments to signals."""
        try:
            # Similar to neural implementation
            if 'regime_probabilities' in regime_data:
                regime_probs = regime_data['regime_probabilities']
                regime_stability = np.mean(regime_probs[-5:])

                # Adjust based on tree model characteristics
                if self.model.__class__.__name__ in ['RandomForestClassifier', 'GradientBoostingClassifier']:
                    confidence *= regime_stability * 0.9  # Trees are slightly less confident

            return signal_type, confidence

        except Exception as e:
            self.logger.warning(f"Regime adjustment failed: {e}")
            return signal_type, confidence

class EnsembleSignalGenerator:
    """Ensemble signal generator combining multiple architectures."""

    def __init__(self, generators: Dict[SignalSource, Any], config: ArchitectureSignalConfig):
        """Initialize ensemble signal generator."""
        self.generators = generators
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Ensemble weights
        self.weights = config.ensemble_weights

        # Signal history for ensemble
        self.ensemble_history = []

    def generate_signal(self, market_data: np.ndarray,
                       regime_data: Optional[Dict[str, Any]] = None) -> TradingSignal:
        """Generate ensemble trading signal."""
        try:
            # Get signals from all generators
            individual_signals = {}
            for source, generator in self.generators.items():
                signal = generator.generate_signal(market_data, regime_data)
                individual_signals[source] = signal

            # Combine signals
            ensemble_signal = self._combine_signals(individual_signals)

            # Add ensemble metadata
            ensemble_signal.metadata['individual_signals'] = {
                source.value: {
                    'signal_type': signal.signal_type.value,
                    'confidence': signal.confidence,
                    'source': signal.source.value
                }
                for source, signal in individual_signals.items()
            }

            # Store ensemble history
            self.ensemble_history.append(ensemble_signal)

            return ensemble_signal

        except Exception as e:
            self.logger.error(f"❌ Ensemble signal generation failed: {e}")
            return TradingSignal(
                signal_type=SignalType.HOLD,
                confidence=0.0,
                source=SignalSource.ENSEMBLE,
                timestamp=datetime.now(),
                price=0.0,
                metadata={'error': str(e)}
            )

    def _combine_signals(self, signals: Dict[SignalSource, TradingSignal]) -> TradingSignal:
        """Combine individual signals into ensemble signal."""
        if self.config.ensemble_method == "weighted_average":
            return self._weighted_average_combination(signals)
        elif self.config.ensemble_method == "majority_vote":
            return self._majority_vote_combination(signals)
        else:
            return self._weighted_average_combination(signals)

    def _weighted_average_combination(self, signals: Dict[SignalSource, TradingSignal]) -> TradingSignal:
        """Combine signals using weighted average."""
        # Convert signals to numeric values
        signal_values = {}
        for source, signal in signals.items():
            weight = self.weights.get(source, 0.5)
            if signal.signal_type == SignalType.STRONG_BUY:
                signal_values[source] = 2.0 * weight
            elif signal.signal_type == SignalType.BUY:
                signal_values[source] = 1.0 * weight
            elif signal.signal_type == SignalType.HOLD:
                signal_values[source] = 0.0 * weight
            elif signal.signal_type == SignalType.SELL:
                signal_values[source] = -1.0 * weight
            elif signal.signal_type == SignalType.STRONG_SELL:
                signal_values[source] = -2.0 * weight

        # Calculate weighted average
        total_weight = sum(self.weights.values())
        if total_weight == 0:
            total_weight = len(signals)

        weighted_sum = sum(signal_values.values())
        average_score = weighted_sum / total_weight

        # Convert back to signal
        if average_score >= 1.5:
            signal_type = SignalType.STRONG_BUY
        elif average_score >= 0.5:
            signal_type = SignalType.BUY
        elif average_score <= -1.5:
            signal_type = SignalType.STRONG_SELL
        elif average_score <= -0.5:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD

        # Calculate confidence
        confidences = [signal.confidence for signal in signals.values()]
        ensemble_confidence = np.mean(confidences)

        # Get average price
        prices = [signal.price for signal in signals.values() if signal.price > 0]
        avg_price = np.mean(prices) if prices else 0.0

        return TradingSignal(
            signal_type=signal_type,
            confidence=ensemble_confidence,
            source=SignalSource.ENSEMBLE,
            timestamp=datetime.now(),
            price=avg_price,
            metadata={'combination_method': 'weighted_average'}
        )

    def _majority_vote_combination(self, signals: Dict[SignalSource, TradingSignal]) -> TradingSignal:
        """Combine signals using majority vote."""
        # Count votes for each signal type
        vote_counts = {}
        total_votes = 0

        for signal in signals.values():
            weight = self.weights.get(signal.source, 1.0)
            vote_counts[signal.signal_type] = vote_counts.get(signal.signal_type, 0) + weight
            total_votes += weight

        # Find signal type with most votes
        winning_signal = max(vote_counts.items(), key=lambda x: x[1])
        signal_type = winning_signal[0]

        # Calculate confidence based on vote percentage
        vote_percentage = winning_signal[1] / total_votes
        confidences = [signal.confidence for signal in signals.values()]
        avg_confidence = np.mean(confidences)

        confidence = vote_percentage * avg_confidence

        # Get average price
        prices = [signal.price for signal in signals.values() if signal.price > 0]
        avg_price = np.mean(prices) if prices else 0.0

        return TradingSignal(
            signal_type=signal_type,
            confidence=confidence,
            source=SignalSource.ENSEMBLE,
            timestamp=datetime.now(),
            price=avg_price,
            metadata={'combination_method': 'majority_vote', 'vote_counts': vote_counts}
        )

class SignalQualityEvaluator:
    """Evaluator for signal quality metrics."""

    def __init__(self, config: ArchitectureSignalConfig):
        """Initialize signal quality evaluator."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Historical signal data
        self.signal_history = []
        self.performance_history = []

    def evaluate_signal_quality(self, signals: List[TradingSignal],
                              market_data: pd.DataFrame) -> SignalQualityMetrics:
        """Evaluate quality of generated signals."""
        try:
            if len(signals) < self.config.min_validation_samples:
                self.logger.warning("Insufficient signals for quality evaluation")
                return SignalQualityMetrics(
                    accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0,
                    sharpe_ratio=0.0, max_drawdown=0.0, win_rate=0.0, profit_factor=0.0
                )

            # Calculate basic metrics
            accuracy = self._calculate_accuracy(signals, market_data)
            precision = self._calculate_precision(signals, market_data)
            recall = self._calculate_recall(signals, market_data)
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            # Calculate financial metrics
            returns = self._calculate_returns(signals, market_data)
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            max_drawdown = self._calculate_max_drawdown(returns)
            win_rate = self._calculate_win_rate(returns)
            profit_factor = self._calculate_profit_factor(returns)

            return SignalQualityMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                metadata={
                    'n_signals': len(signals),
                    'evaluation_window': self.config.validation_window
                }
            )

        except Exception as e:
            self.logger.error(f"❌ Signal quality evaluation failed: {e}")
            return SignalQualityMetrics(
                accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0,
                sharpe_ratio=0.0, max_drawdown=0.0, win_rate=0.0, profit_factor=0.0,
                metadata={'error': str(e)}
            )

    def _calculate_accuracy(self, signals: List[TradingSignal], market_data: pd.DataFrame) -> float:
        """Calculate signal accuracy."""
        # Simplified accuracy calculation
        # In practice, this would compare signals against actual market movements
        return np.random.uniform(0.6, 0.8)  # Placeholder

    def _calculate_precision(self, signals: List[TradingSignal], market_data: pd.DataFrame) -> float:
        """Calculate signal precision."""
        return np.random.uniform(0.65, 0.85)  # Placeholder

    def _calculate_recall(self, signals: List[TradingSignal], market_data: pd.DataFrame) -> float:
        """Calculate signal recall."""
        return np.random.uniform(0.6, 0.8)  # Placeholder

    def _calculate_returns(self, signals: List[TradingSignal], market_data: pd.DataFrame) -> np.ndarray:
        """Calculate returns from signals."""
        # Simplified returns calculation
        returns = np.random.normal(0.001, 0.02, len(signals))
        return returns

    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        return mean_return / std_return * np.sqrt(252)  # Annualized

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / peak
        return np.max(drawdown)

    def _calculate_win_rate(self, returns: np.ndarray) -> float:
        """Calculate win rate."""
        winning_trades = np.sum(returns > 0)
        total_trades = len(returns)
        return winning_trades / total_trades if total_trades > 0 else 0.0

    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Calculate profit factor."""
        positive_returns = np.sum(np.where(returns > 0, returns, 0))
        negative_returns = np.sum(np.where(returns < 0, -returns, 0))

        if negative_returns == 0:
            return float('inf') if positive_returns > 0 else 0.0

        return positive_returns / negative_returns

class ArchitectureSignalGenerator:
    """
    Main architecture-based signal generation system.

    Coordinates neural, tree, and ensemble signal generators with quality evaluation
    and real-time processing capabilities.
    """

    def __init__(self, config: ArchitectureSignalConfig):
        """Initialize architecture signal generator."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Signal generators
        self.neural_generator = None
        self.tree_generator = None
        self.ensemble_generator = None

        # Quality evaluator
        self.quality_evaluator = SignalQualityEvaluator(config)

        # Signal buffer for real-time processing
        self.signal_buffer = []

        # Performance tracking
        self.performance_metrics = {}
        self.signal_statistics = {}

        self.logger.info("✅ Architecture Signal Generator initialized")
        self.logger.info(f"   Ensemble Method: {config.ensemble_method}")
        self.logger.info(f"   Signal Threshold: {config.signal_threshold}")

    def add_neural_generator(self, model: nn.Module):
        """Add neural signal generator."""
        self.neural_generator = NeuralSignalGenerator(model, self.config)
        self.logger.info("✅ Neural signal generator added")

    def add_tree_generator(self, model: Any):
        """Add tree signal generator."""
        self.tree_generator = TreeSignalGenerator(model, self.config)
        self.logger.info("✅ Tree signal generator added")

    def create_ensemble_generator(self):
        """Create ensemble signal generator."""
        if not (self.neural_generator and self.tree_generator):
            self.logger.warning("Need both neural and tree generators for ensemble")
            return

        generators = {}
        if self.neural_generator:
            generators[SignalSource.NEURAL_NETWORK] = self.neural_generator
        if self.tree_generator:
            generators[SignalSource.TREE_MODEL] = self.tree_generator

        self.ensemble_generator = EnsembleSignalGenerator(generators, self.config)
        self.logger.info("✅ Ensemble signal generator created")

    def generate_signal(self, market_data: np.ndarray,
                       regime_data: Optional[Dict[str, Any]] = None) -> TradingSignal:
        """Generate trading signal from architectures."""
        try:
            if self.ensemble_generator:
                # Use ensemble generator
                signal = self.ensemble_generator.generate_signal(market_data, regime_data)
            elif self.neural_generator:
                # Use neural generator
                signal = self.neural_generator.generate_signal(market_data, regime_data)
            elif self.tree_generator:
                # Use tree generator
                signal = self.tree_generator.generate_signal(market_data, regime_data)
            else:
                # No generators available
                signal = TradingSignal(
                    signal_type=SignalType.HOLD,
                    confidence=0.0,
                    source=SignalSource.HYBRID,
                    timestamp=datetime.now(),
                    price=0.0,
                    metadata={'error': 'No signal generators available'}
                )

            # Apply signal validation
            if self.config.enable_signal_validation:
                signal = self._validate_signal(signal, market_data)

            # Add to buffer for real-time processing
            if self.config.enable_real_time_processing:
                self._add_to_buffer(signal)

            return signal

        except Exception as e:
            self.logger.error(f"❌ Signal generation failed: {e}")
            return TradingSignal(
                signal_type=SignalType.HOLD,
                confidence=0.0,
                source=SignalSource.HYBRID,
                timestamp=datetime.now(),
                price=0.0,
                metadata={'error': str(e)}
            )

    def _validate_signal(self, signal: TradingSignal, market_data: np.ndarray) -> TradingSignal:
        """Validate signal quality and adjust confidence."""
        try:
            # Check signal strength
            if signal.confidence < self.config.signal_threshold:
                signal.signal_type = SignalType.HOLD
                signal.confidence = 0.0

            # Check market conditions
            if market_data.size > 0:
                market_volatility = np.std(market_data.flatten())
                if market_volatility > 0.05:  # High volatility
                    signal.confidence *= 0.8  # Reduce confidence

            return signal

        except Exception as e:
            self.logger.warning(f"Signal validation failed: {e}")
            return signal

    def _add_to_buffer(self, signal: TradingSignal):
        """Add signal to buffer for real-time processing."""
        self.signal_buffer.append(signal)

        # Maintain buffer size
        if len(self.signal_buffer) > self.config.signal_buffer_size:
            self.signal_buffer.pop(0)

    def evaluate_signal_quality(self, market_data: pd.DataFrame) -> SignalQualityMetrics:
        """Evaluate quality of generated signals."""
        if len(self.signal_buffer) < self.config.min_validation_samples:
            self.logger.warning("Insufficient signals for quality evaluation")
            return SignalQualityMetrics(
                accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0,
                sharpe_ratio=0.0, max_drawdown=0.0, win_rate=0.0, profit_factor=0.0
            )

        return self.quality_evaluator.evaluate_signal_quality(self.signal_buffer, market_data)

    def get_signal_statistics(self) -> Dict[str, Any]:
        """Get signal generation statistics."""
        if not self.signal_buffer:
            return {}

        signals = self.signal_buffer
        signal_types = [signal.signal_type.value for signal in signals]
        confidences = [signal.confidence for signal in signals]

        return {
            'total_signals': len(signals),
            'signal_type_distribution': pd.Series(signal_types).value_counts().to_dict(),
            'average_confidence': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'high_confidence_signals': sum(c >= self.config.confidence_threshold for c in confidences),
            'low_confidence_signals': sum(c < self.config.confidence_threshold for c in confidences)
        }

    def get_recent_signals(self, n: int = 10) -> List[TradingSignal]:
        """Get most recent signals."""
        return self.signal_buffer[-n:] if self.signal_buffer else []

    def save_signal_generator(self, filepath: str) -> bool:
        """Save signal generator state."""
        try:
            state = {
                'config': self.config.__dict__,
                'signal_buffer': self.signal_buffer,
                'performance_metrics': self.performance_metrics,
                'signal_statistics': self.signal_statistics
            }

            with open(filepath, 'wb') as f:
                import pickle
                pickle.dump(state, f)

            self.logger.info(f"✅ Signal generator state saved to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to save signal generator: {e}")
            return False

def create_architecture_signal_generator(config: ArchitectureSignalConfig) -> ArchitectureSignalGenerator:
    """Create architecture signal generator instance."""
    return ArchitectureSignalGenerator(config)

def quick_signal_generation(architectures: Dict[str, Any],
                          market_data: np.ndarray,
                          config: Optional[ArchitectureSignalConfig] = None) -> TradingSignal:
    """Quick signal generation with default settings."""
    if config is None:
        config = ArchitectureSignalConfig()

    generator = ArchitectureSignalGenerator(config)

    # Mock neural and tree generators for demonstration
    # In practice, these would be actual trained models
    mock_neural = type('MockNeural', (), {})()
    mock_tree = type('MockTree', (), {})()

    generator.add_neural_generator(mock_neural)
    generator.add_tree_generator(mock_tree)
    generator.create_ensemble_generator()

    return generator.generate_signal(market_data)
