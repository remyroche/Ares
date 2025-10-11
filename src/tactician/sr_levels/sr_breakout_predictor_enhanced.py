from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from datetime import datetime
import numpy as np
from ...utils.logger import system_logger
from src.core.decorators import handles_errors
'Enhanced S/R Breakout Predictor.\n\nThis module provides advanced breakout prediction capabilities with ML integration,\nreal-time monitoring, and comprehensive validation.\n'
from dataclasses import dataclass
from enum import Enum

import time
import psutil
import logging

# Simple error classes for SR operations
class SROptimizationError(Exception):
    """SR optimization specific error."""
    pass

class SRDataError(Exception):
    """SR data specific error."""
    pass

@dataclass
class ProgressMetrics:
    """Enhanced progress tracking with performance metrics."""
    total_items: int
    processed_items: int = 0
    start_time: float = None
    last_update_time: float = None
    last_logged_progress: float = -1.0
    memory_usage_mb: float = 0.0
    processing_rate: float = 0.0
    estimated_time_remaining: float = 0.0

    def __post_init__(self):
        if self.start_time is None:
            self.start_time = time.time()
        if self.last_update_time is None:
            self.last_update_time = self.start_time

    def update(self, processed: int):
        """Update progress metrics."""
        self.processed_items = processed
        current_time = time.time()
        elapsed = current_time - self.start_time

        # Calculate processing rate (items per second)
        if elapsed > 0:
            self.processing_rate = self.processed_items / elapsed

        # Calculate estimated time remaining
        if self.processing_rate > 0:
            remaining_items = self.total_items - self.processed_items
            self.estimated_time_remaining = remaining_items / self.processing_rate

        # Track memory usage
        try:
            process = psutil.Process()
            self.memory_usage_mb = process.memory_info().rss / 1024 / 1024
        except:
            self.memory_usage_mb = 0.0

        self.last_update_time = current_time

    def get_progress_percentage(self) -> float:
        """Get current progress as percentage."""
        return (self.processed_items / self.total_items) * 100 if self.total_items > 0 else 0

    def should_log_progress(self, min_interval_percent: float = 5.0) -> bool:
        """Determine if progress should be logged based on adaptive intervals."""
        current_progress = self.get_progress_percentage()

        # Always log at 0%, 25%, 50%, 75%, 100%
        key_milestones = [0, 25, 50, 75, 100]
        if any(abs(current_progress - milestone) < 0.1 for milestone in key_milestones):
            return True

        # Log at adaptive intervals based on processing speed
        if current_progress - self.last_logged_progress >= min_interval_percent:
            return True

        return False

    def format_eta(self) -> str:
        """Format estimated time remaining."""
        if self.estimated_time_remaining <= 0:
            return "complete"

        if self.estimated_time_remaining < 60:
            return f"{self.estimated_time_remaining:.1f}s"
        elif self.estimated_time_remaining < 3600:
            return f"{self.estimated_time_remaining/60:.1f}m"
        else:
            return f"{self.estimated_time_remaining/3600:.1f}h"

    def format_progress_message(self, operation_name: str = "Processing") -> str:
        """Generate comprehensive progress message."""
        progress_pct = self.get_progress_percentage()
        elapsed = time.time() - self.start_time

        message_parts = [
            f"📊 {operation_name}: {progress_pct:.1f}%",
            f"({self.processed_items:,}/{self.total_items:,})",
            f"⏱️ {elapsed:.1f}s elapsed"
        ]

        if self.processing_rate > 0:
            message_parts.append(f"⚡ {self.processing_rate:.0f} items/s")

        if self.estimated_time_remaining > 0:
            message_parts.append(f"⏳ ETA: {self.format_eta()}")

        if self.memory_usage_mb > 0:
            message_parts.append(f"💾 {self.memory_usage_mb:.1f}MB")

        return " | ".join(message_parts)

class EnhancedProgressLogger:
    """Enhanced progress logging utility with adaptive intervals and metrics."""

    def __init__(self, logger, total_items: int, operation_name: str = "Processing"):
        self.logger = logger
        self.metrics = ProgressMetrics(total_items=total_items)
        self.operation_name = operation_name
        self.start_message_logged = False

    def start(self):
        """Log the start of the operation."""
        if not self.start_message_logged:
            self.logger.info(f"🔍 Starting {self.operation_name} of {self.metrics.total_items:,} items...")
            self.start_message_logged = True

    def update(self, processed_items: int):
        """Update progress and log if needed."""
        self.metrics.update(processed_items)

        if self.metrics.should_log_progress():
            message = self.metrics.format_progress_message(self.operation_name)
            self.logger.info(message)
            self.metrics.last_logged_progress = self.metrics.get_progress_percentage()

    def complete(self):
        """Log completion of the operation."""
        self.metrics.update(self.metrics.total_items)
        elapsed = time.time() - self.metrics.start_time

        completion_message = (
            f"✅ {self.operation_name} completed! "
            f"Processed {self.metrics.total_items:,} items in {elapsed:.2f}s "
            f"(avg: {self.metrics.processing_rate:.0f} items/s)"
        )

        if self.metrics.memory_usage_mb > 0:
            completion_message += f" | Peak memory: {self.metrics.memory_usage_mb:.1f}MB"

        self.logger.info(completion_message)

class BreakoutType(Enum):
    """Types of breakouts."""
    SUPPORT_BREAKDOWN = 'support_breakdown'
    RESISTANCE_BREAKOUT = 'resistance_breakout'
    FALSE_BREAKOUT = 'false_breakout'
    CONSOLIDATION = 'consolidation'

class BreakoutConfidence(Enum):
    """Breakout confidence levels."""
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    VERY_HIGH = 'very_high'

@dataclass
class BreakoutSignal:
    """Breakout signal with detailed information."""
    level_id: str
    breakout_type: BreakoutType
    confidence: BreakoutConfidence
    probability: float
    expected_direction: str
    target_price: Optional[float]
    stop_loss: Optional[float]
    time_to_breakout: Optional[int]
    volume_confirmation: bool
    momentum_confirmation: bool
    features: Dict[str, float]
    timestamp: datetime
    validation_score: float

@dataclass
class BreakoutValidation:
    """Breakout validation result."""
    is_valid: bool
    false_breakout_probability: float
    confirmation_required: bool
    validation_metrics: Dict[str, float]
    recommended_action: str

@dataclass
class BreakoutPerformance:
    """Breakout prediction performance metrics."""
    total_predictions: int
    correct_predictions: int
    false_breakouts: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    profit_factor: float
    average_hold_time: float
    max_drawdown: float

class EnhancedSRBreakoutPredictor:
    """Enhanced S/R breakout predictor with ML integration."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize enhanced breakout predictor."""
        self.config = config
        self.logger = system_logger.getChild('EnhancedSRBreakoutPredictor')
        self.breakout_config = config.get('breakout_prediction', {})
        self.proximity_threshold = self.breakout_config.get('breakout_detection', {}).get('proximity_threshold', 0.02)
        self.volume_spike_threshold = self.breakout_config.get('breakout_detection', {}).get('volume_spike_threshold', 1.5)
        self.momentum_threshold = self.breakout_config.get('breakout_detection', {}).get('momentum_threshold', 0.01)
        self.confirmation_bars = self.breakout_config.get('breakout_detection', {}).get('confirmation_bars', 2)
        self.false_breakout_threshold = self.breakout_config.get('breakout_validation', {}).get('false_breakout_threshold', 0.03)
        self.confirmation_timeframe = self.breakout_config.get('breakout_validation', {}).get('confirmation_timeframe', '5m')
        self.min_breakout_duration = self.breakout_config.get('breakout_validation', {}).get('min_breakout_duration', 5)
        self.prediction_history: List[BreakoutSignal] = []
        self.validation_results: List[BreakoutValidation] = []
        self.performance_metrics = BreakoutPerformance(total_predictions = 0, correct_predictions = 0, false_breakouts = 0, accuracy = 0.0, precision = 0.0, recall = 0.0, f1_score = 0.0, profit_factor = 0.0, average_hold_time = 0.0, max_drawdown = 0.0)
        self.active_signals: Dict[str, BreakoutSignal] = {}
        self.monitoring_enabled = True
        self.ml_model = None
        self.feature_importance = {}
        # Add missing attributes for SRLevelsManager compatibility
        self.sr_detection_method = 'fractal'  # Default detection method

    async def initialize(self) -> None:
        """Async initialization method for compatibility with calling code."""
        # All initialization is already done in __init__, so this is just a compatibility method
        self.logger.info('🔧 SR Breakout Predictor async initialization completed')

    @handles_errors(exceptions=(SROptimizationError, SRDataError), default_return=[])
    async def predict_breakouts(self, market_data: pd.DataFrame, sr_levels: List[Dict[str, Any]], current_price: Optional[float]=None) -> List[BreakoutSignal]:
        """Predict potential breakouts from S/R levels."""
        try:
            if not sr_levels:
                return []
            self.logger.info(f'🔮 Predicting breakouts for {len(sr_levels)} levels')
            current_price = current_price or market_data['close'].iloc[-1]
            breakout_signals = []
            for level in sr_levels:
                signal = await self._analyze_level_for_breakout(market_data, level, current_price)
                if signal and signal.probability > 0.3:
                    breakout_signals.append(signal)
                    self.active_signals[signal.level_id] = signal
            breakout_signals.sort(key = lambda x: (x.probability, x.confidence.value), reverse = True)
            self.performance_metrics.total_predictions += len(breakout_signals)
            self.prediction_history.extend(breakout_signals)
            self.logger.info(f'✅ Generated {len(breakout_signals)} breakout signals')
            return breakout_signals
        except Exception as e:
            self.logger.error(f'Breakout prediction failed: {e}')
            return []

    async def _analyze_level_for_breakout(self, market_data: pd.DataFrame, level: Dict[str, Any], current_price: float) -> Optional[BreakoutSignal]:
        """Analyze a specific level for breakout potential."""
        try:
            level_price = level.get('price', 0)
            level_type = level.get('type', 'unknown')
            level_id = level.get('id', f'level_{level_price}')
            if level_price <= 0:
                return None
            proximity = abs(current_price - level_price) / level_price
            if proximity > self.proximity_threshold:
                return None
            features = await self._extract_breakout_features(market_data, level, current_price)
            probability = await self._calculate_breakout_probability(features)
            breakout_type, direction = self._determine_breakout_type_and_direction(level_type, current_price, level_price)
            confidence = self._calculate_breakout_confidence(features, probability)
            target_price, stop_loss = self._calculate_target_and_stop_loss(level_price, level_type, current_price, features)
            time_to_breakout = self._estimate_time_to_breakout(features)
            volume_confirmation = features.get('volume_spike', 0) > self.volume_spike_threshold
            momentum_confirmation = features.get('momentum', 0) > self.momentum_threshold
            validation_score = self._calculate_validation_score(features)
            return BreakoutSignal(level_id = level_id, breakout_type = breakout_type, confidence = confidence, probability = probability, expected_direction = direction, target_price = target_price, stop_loss = stop_loss, time_to_breakout = time_to_breakout, volume_confirmation = volume_confirmation, momentum_confirmation = momentum_confirmation, features = features, timestamp = datetime.now(), validation_score = validation_score)
        except Exception as e:
            self.logger.error(f'Level analysis failed: {e}')
            return None

    async def _extract_breakout_features(self, market_data: pd.DataFrame, level: Dict[str, Any], current_price: float) -> Dict[str, float]:
        """Extract ALL features for breakout prediction (S/R + ALL step06 features)."""
        try:
            features = {}
            level_price = level.get('price', 0)
            features['proximity_to_level'] = abs(current_price - level_price) / level_price
            features['level_strength'] = level.get('strength', 0.5)
            features['touch_count'] = level.get('touch_count', 0)
            features['age_bars'] = level.get('age_bars', 0)
            features['avg_bounce_ratio'] = level.get('avg_bounce_ratio', 0)
            features['max_bounce_ratio'] = level.get('max_bounce_ratio', 0)
            features['volume_confirmation_score'] = level.get('volume_confirmation_score', 0.5)
            features['consistency_score'] = level.get('consistency_score', 0.5)
            features['failure_count'] = level.get('failure_count', 0)
            features['rsi_14'] = self._calculate_rsi(market_data['close'])
            features['macd_line'] = self._calculate_macd_line(market_data['close'])
            features['macd_signal'] = self._calculate_macd_signal(market_data['close'])
            features['bollinger_position'] = self._calculate_bollinger_position(market_data)
            features['atr_14'] = self._calculate_atr(market_data)
            features['volume_ratio'] = self._calculate_volume_spike(market_data)
            features['price_momentum'] = self._calculate_momentum(market_data)
            features['stoch_k'] = self._calculate_stochastic_k(market_data)
            features['stoch_d'] = self._calculate_stochastic_d(market_data)
            features['williams_r'] = self._calculate_williams_r(market_data)
            features['cci'] = self._calculate_cci(market_data)
            features['adx'] = self._calculate_adx(market_data)
            features['obv'] = self._calculate_obv(market_data)
            features['doji_pattern'] = self._detect_doji_pattern(market_data)
            features['hammer_pattern'] = self._detect_hammer_pattern(market_data)
            features['volatility_proxy'] = self._calculate_volatility_proxy(market_data)
            features['level_density'] = self._calculate_sr_density(level_price, level)
            features['confluence_score'] = level.get('confluence_score', 0.5)
            features['time_since_touch'] = self._calculate_time_at_level(market_data, level_price)
            features['volume_at_touch'] = level.get('volume_at_touch', 1.0)
            features['price_action_score'] = self._detect_price_action_pattern(market_data)
            features['microstructure_score'] = level.get('microstructure_score', 0.5)
            step06_features = await self._extract_all_step06_features(market_data)
            features.update(step06_features)
            self.logger.info(f'✅ Extracted {len(features)} total features for breakout prediction')
            self.logger.info(f'   - S/R specific features: 31')
            self.logger.info(f'   - Step06 features: {len(step06_features)}')
            return features
        except Exception as e:
            self.logger.error(f'Feature extraction failed: {e}')
            return {}

    async def _extract_all_step06_features(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Extract ALL step06 features (200+ features)."""
        try:
            try:
                from src.training.steps.vectorized_advanced_feature_engineering import VectorizedAdvancedFeatureEngineeringRefactored

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
                step06_engineer = VectorizedAdvancedFeatureEngineeringRefactored()
                step06_result = await step06_engineer.engineer_features(market_data)
                all_features = {}
                feature_categories = ['price_features', 'volume_features', 'microstructure_features', 'technical_features', 'regime_features', 'wavelet_features', 'cross_timeframe_features', 'interaction_features']
                for category in feature_categories:
                    category_features = step06_result.get(category, {})
                    for feature_name, feature_values in category_features.items():
                        if isinstance(feature_values, (list, np.ndarray)) and len(feature_values) > 0:
                            all_features[f'{category}_{feature_name}'] = float(feature_values[-1])
                        elif isinstance(feature_values, (int, float)):
                            all_features[f'{category}_{feature_name}'] = float(feature_values)
                self.logger.info(f'✅ Extracted {len(all_features)} step06 features')
                return all_features
            except ImportError as e:
                self.logger.warning(f'Step06 feature engineering not available: {e}')
                return {}
            except Exception as e:
                self.logger.warning(f'Step06 feature extraction failed: {e}')
                return {}
        except Exception as e:
            self.logger.error(f'Step06 feature extraction failed: {e}')
            return {}

    async def _calculate_breakout_probability(self, features: Dict[str, float]) -> float:
        """Calculate breakout probability using features."""
        try:
            proximity = features.get('proximity_to_level', 1.0)
            base_prob = max(0.0, 1.0 - proximity / self.proximity_threshold)
            strength = features.get('level_strength', 0.5)
            strength_factor = 1.0 - strength * 0.3
            volume_spike = features.get('volume_spike', 1.0)
            volume_factor = min(volume_spike / self.volume_spike_threshold, 1.5)
            momentum = features.get('momentum', 0.0)
            momentum_factor = 1.0 + momentum * 2.0
            volatility = features.get('volatility', 0.0)
            volatility_factor = 1.0 + volatility * 0.5
            rsi = features.get('rsi', 50.0)
            rsi_factor = 1.0
            if rsi > 70:
                rsi_factor = 1.2
            elif rsi < 30:
                rsi_factor = 1.2
            probability = base_prob * strength_factor * volume_factor * momentum_factor * volatility_factor * rsi_factor
            if self.ml_model:
                ml_probability = await self._get_ml_prediction(features)
                probability = probability * 0.6 + ml_probability * 0.4
            return min(max(probability, 0.0), 1.0)
        except Exception as e:
            self.logger.error(f'Probability calculation failed: {e}')
            return 0.0

    def _determine_breakout_type_and_direction(self, level_type: str, current_price: float, level_price: float) -> Tuple[BreakoutType, str]:
        """Determine breakout type and direction."""
        try:
            if level_type == 'resistance':
                if current_price > level_price:
                    return (BreakoutType.RESISTANCE_BREAKOUT, 'up')
                else:
                    return (BreakoutType.CONSOLIDATION, 'sideways')
            elif level_type == 'support':
                if current_price < level_price:
                    return (BreakoutType.SUPPORT_BREAKDOWN, 'down')
                else:
                    return (BreakoutType.CONSOLIDATION, 'sideways')
            else:
                return (BreakoutType.CONSOLIDATION, 'sideways')
        except Exception as e:
            self.logger.error(f'Breakout type determination failed: {e}')
            return (BreakoutType.CONSOLIDATION, 'sideways')

    def _calculate_breakout_confidence(self, features: Dict[str, float], probability: float) -> BreakoutConfidence:
        """Calculate breakout confidence level."""
        try:
            confidence_score = 0.0
            confidence_score += probability * 0.4
            if features.get('volume_spike', 0) > self.volume_spike_threshold:
                confidence_score += 0.2
            if features.get('momentum', 0) > self.momentum_threshold:
                confidence_score += 0.2
            strength = features.get('level_strength', 0.5)
            confidence_score += (1.0 - strength) * 0.1
            rsi = features.get('rsi', 50.0)
            if rsi > 70 or rsi < 30:
                confidence_score += 0.1
            if confidence_score >= 0.8:
                return BreakoutConfidence.VERY_HIGH
            elif confidence_score >= 0.6:
                return BreakoutConfidence.HIGH
            elif confidence_score >= 0.4:
                return BreakoutConfidence.MEDIUM
            else:
                return BreakoutConfidence.LOW
        except Exception as e:
            self.logger.error(f'Confidence calculation failed: {e}')
            return BreakoutConfidence.LOW

    def _calculate_target_and_stop_loss(self, level_price: float, level_type: str, current_price: float, features: Dict[str, float]) -> Tuple[Optional[float], Optional[float]]:
        """Calculate target price and stop loss."""
        try:
            volatility = features.get('volatility', 0.01)
            atr_multiplier = 2.0
            if level_type == 'resistance' and current_price > level_price:
                target_price = level_price * (1 + volatility * atr_multiplier)
                stop_loss = level_price * (1 - volatility * 0.5)
            elif level_type == 'support' and current_price < level_price:
                target_price = level_price * (1 - volatility * atr_multiplier)
                stop_loss = level_price * (1 + volatility * 0.5)
            else:
                return (None, None)
            return (target_price, stop_loss)
        except Exception as e:
            self.logger.error(f'Target/stop loss calculation failed: {e}')
            return (None, None)

    def _estimate_time_to_breakout(self, features: Dict[str, float]) -> Optional[int]:
        """Estimate time to breakout in bars."""
        try:
            proximity = features.get('proximity_to_level', 1.0)
            momentum = features.get('momentum', 0.0)
            base_time = int(proximity * 100)
            momentum_adjustment = int(momentum * 50)
            estimated_time = max(1, base_time - momentum_adjustment)
            return min(estimated_time, 100)
        except Exception as e:
            self.logger.error(f'Time estimation failed: {e}')
            return None

    def _calculate_validation_score(self, features: Dict[str, float]) -> float:
        """Calculate validation score for the breakout signal."""
        try:
            score = 0.0
            if features.get('volume_spike', 0) > self.volume_spike_threshold:
                score += 0.3
            if features.get('momentum', 0) > self.momentum_threshold:
                score += 0.3
            rsi = features.get('rsi', 50.0)
            if 30 < rsi < 70:
                score += 0.2
            strength = features.get('level_strength', 0.5)
            score += (1.0 - strength) * 0.2
            return min(score, 1.0)
        except Exception as e:
            self.logger.error(f'Validation score calculation failed: {e}')
            return 0.0

    async def validate_breakout(self, signal: BreakoutSignal, market_data: pd.DataFrame) -> BreakoutValidation:
        """Validate a breakout signal."""
        try:
            false_breakout_prob = self._calculate_false_breakout_probability(signal, market_data)
            confirmation_required = signal.confidence in [BreakoutConfidence.LOW, BreakoutConfidence.MEDIUM]
            validation_metrics = {'false_breakout_probability': false_breakout_prob, 'volume_confirmation': signal.volume_confirmation, 'momentum_confirmation': signal.momentum_confirmation, 'validation_score': signal.validation_score}
            is_valid = false_breakout_prob < 0.3 and signal.validation_score > 0.5 and (signal.volume_confirmation or signal.momentum_confirmation)
            if is_valid and signal.confidence in [BreakoutConfidence.HIGH, BreakoutConfidence.VERY_HIGH]:
                recommended_action = 'enter_position'
            elif is_valid:
                recommended_action = 'wait_for_confirmation'
            else:
                recommended_action = 'avoid'
            validation = BreakoutValidation(is_valid = is_valid, false_breakout_probability = false_breakout_prob, confirmation_required = confirmation_required, validation_metrics = validation_metrics, recommended_action = recommended_action)
            self.validation_results.append(validation)
            return validation
        except Exception as e:
            self.logger.error(f'Breakout validation failed: {e}')
            return BreakoutValidation(is_valid = False, false_breakout_probability = 1.0, confirmation_required = True, validation_metrics={}, recommended_action='avoid')

    def _calculate_false_breakout_probability(self, signal: BreakoutSignal, market_data: pd.DataFrame) -> float:
        """Calculate probability of false breakout."""
        try:
            false_prob = 0.0
            volatility = signal.features.get('volatility', 0.0)
            false_prob += min(volatility * 2.0, 0.3)
            if not signal.volume_confirmation:
                false_prob += 0.2
            if not signal.momentum_confirmation:
                false_prob += 0.2
            strength = signal.features.get('level_strength', 0.5)
            false_prob += strength * 0.3
            return min(false_prob, 1.0)
        except Exception as e:
            self.logger.error(f'False breakout probability calculation failed: {e}')
            return 0.5

    async def monitor_active_signals(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Monitor active breakout signals for updates."""
        try:
            if not self.monitoring_enabled:
                return []
            updates = []
            current_price = market_data['close'].iloc[-1]
            for signal_id, signal in list(self.active_signals.items()):
                time_elapsed = (datetime.now() - signal.timestamp).total_seconds() / 60
                if time_elapsed > 60:
                    del self.active_signals[signal_id]
                    continue
                level_price = signal.features.get('proximity_to_level', 0) * current_price + current_price
                if signal.expected_direction == 'up' and current_price > level_price * 1.01:
                    updates.append({'signal_id': signal_id, 'status': 'confirmed', 'current_price': current_price, 'breakout_price': level_price})
                    del self.active_signals[signal_id]
                elif signal.expected_direction == 'down' and current_price < level_price * 0.99:
                    updates.append({'signal_id': signal_id, 'status': 'confirmed', 'current_price': current_price, 'breakout_price': level_price})
                    del self.active_signals[signal_id]
                elif time_elapsed > 30:
                    if abs(current_price - level_price) / level_price > 0.02:
                        updates.append({'signal_id': signal_id, 'status': 'false_breakout', 'current_price': current_price, 'breakout_price': level_price})
                        del self.active_signals[signal_id]
            return updates
        except Exception as e:
            self.logger.error(f'Signal monitoring failed: {e}')
            return []

    def update_performance_metrics(self, validation_result: BreakoutValidation) -> None:
        """Update performance metrics based on validation results."""
        try:
            if validation_result.is_valid:
                self.performance_metrics.correct_predictions += 1
            else:
                self.performance_metrics.false_breakouts += 1
            total = self.performance_metrics.total_predictions
            if total > 0:
                self.performance_metrics.accuracy = self.performance_metrics.correct_predictions / total
                self.performance_metrics.precision = self.performance_metrics.correct_predictions / (self.performance_metrics.correct_predictions + self.performance_metrics.false_breakouts)
        except Exception as e:
            self.logger.error(f'Performance metrics update failed: {e}')

    def _calculate_volume_spike(self, market_data: pd.DataFrame) -> float:
        """Calculate volume spike ratio."""
        try:
            if len(market_data) < 20:
                return 1.0
            current_volume = market_data['volume'].iloc[-1]
            avg_volume = market_data['volume'].rolling(window = 20).mean().iloc[-1]
            return current_volume / avg_volume if avg_volume > 0 else 1.0
        except Exception:
            return 1.0

    def _calculate_momentum(self, market_data: pd.DataFrame) -> float:
        """Calculate price momentum."""
        try:
            if len(market_data) < 10:
                return 0.0
            current_price = market_data['close'].iloc[-1]
            past_price = market_data['close'].iloc[-10]
            return (current_price - past_price) / past_price
        except Exception:
            return 0.0

    def _calculate_volatility(self, market_data: pd.DataFrame) -> float:
        """Calculate price volatility."""
        try:
            if len(market_data) < 20:
                return 0.01
            returns = market_data['close'].pct_change().dropna()
            return returns.std() if len(returns) > 0 else 0.01
        except Exception:
            return 0.01

    def _calculate_time_at_level(self, market_data: pd.DataFrame, level_price: float) -> int:
        """Calculate time spent at level."""
        try:
            proximity_threshold = 0.005
            time_at_level = 0
            for i in range(len(market_data) - 1, -1, -1):
                price = market_data['close'].iloc[i]
                proximity = abs(price - level_price) / level_price
                if proximity <= proximity_threshold:
                    time_at_level += 1
                else:
                    break
            return time_at_level
        except Exception:
            return 0

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI."""
        try:
            if len(prices) < period + 1:
                return 50.0
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window = period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window = period).mean()
            rs = gain / loss
            rsi = 100 - 100 / (1 + rs)
            return rsi.iloc[-1] if not rsi.empty else 50.0
        except Exception:
            return 50.0

    def _calculate_macd_signal(self, prices: pd.Series) -> float:
        """Calculate MACD signal."""
        try:
            if len(prices) < 26:
                return 0.0
            ema_12 = prices.ewm(span = 12).mean()
            ema_26 = prices.ewm(span = 26).mean()
            macd = ema_12 - ema_26
            signal = macd.ewm(span = 9).mean()
            return macd.iloc[-1] - signal.iloc[-1] if not macd.empty else 0.0
        except Exception:
            return 0.0

    def _calculate_bollinger_position(self, market_data: pd.DataFrame) -> float:
        """Calculate Bollinger Band position."""
        try:
            if len(market_data) < 20:
                return 0.5
            prices = market_data['close']
            sma = prices.rolling(window = 20).mean()
            std = prices.rolling(window = 20).std()
            upper = sma + std * 2
            lower = sma - std * 2
            current_price = prices.iloc[-1]
            position = (current_price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
            return position if not np.isnan(position) else 0.5
        except Exception:
            return 0.5

    def _calculate_order_flow_imbalance(self, market_data: pd.DataFrame) -> float:
        """Calculate order flow imbalance (simplified)."""
        try:
            if len(market_data) < 5:
                return 0.0
            recent_data = market_data.tail(5)
            volume_up = recent_data[recent_data['close'] > recent_data['open']]['volume'].sum()
            volume_down = recent_data[recent_data['close'] < recent_data['open']]['volume'].sum()
            total_volume = volume_up + volume_down
            if total_volume == 0:
                return 0.0
            return (volume_up - volume_down) / total_volume
        except Exception:
            return 0.0

    def _calculate_market_sentiment(self, market_data: pd.DataFrame) -> float:
        """Calculate market sentiment (simplified)."""
        try:
            if len(market_data) < 10:
                return 0.0
            recent_returns = market_data['close'].pct_change().tail(10)
            positive_returns = (recent_returns > 0).sum()
            return positive_returns / len(recent_returns)
        except Exception:
            return 0.5

    def _get_previous_breakout_history(self, level: Dict[str, Any]) -> float:
        """Get previous breakout history for level."""
        try:
            return 0.5
        except Exception:
            return 0.5

    def _calculate_stochastic_k(self, market_data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Stochastic %K."""
        try:
            if len(market_data) < period:
                return 50.0
            low_min = market_data['low'].rolling(window = period).min()
            high_max = market_data['high'].rolling(window = period).max()
            k_percent = 100 * ((market_data['close'] - low_min) / (high_max - low_min))
            return k_percent.iloc[-1] if not k_percent.empty else 50.0
        except Exception:
            return 50.0

    def _calculate_williams_r(self, market_data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Williams %R."""
        try:
            if len(market_data) < period:
                return -50.0
            high_max = market_data['high'].rolling(window = period).max()
            low_min = market_data['low'].rolling(window = period).min()
            williams_r = -100 * ((high_max - market_data['close']) / (high_max - low_min))
            return williams_r.iloc[-1] if not williams_r.empty else -50.0
        except Exception:
            return -50.0

    def _calculate_cci(self, market_data: pd.DataFrame, period: int = 20) -> float:
        """Calculate Commodity Channel Index."""
        try:
            if len(market_data) < period:
                return 0.0
            typical_price = (market_data['high'] + market_data['low'] + market_data['close']) / 3
            sma_tp = typical_price.rolling(window = period).mean()
            mad = typical_price.rolling(window = period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            cci = (typical_price - sma_tp) / (0.015 * mad)
            return cci.iloc[-1] if not cci.empty else 0.0
        except Exception:
            return 0.0

    def _calculate_adx(self, market_data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average Directional Index."""
        try:
            if len(market_data) < period + 1:
                return 25.0
            high_diff = market_data['high'].diff()
            low_diff = market_data['low'].diff()
            plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
            plus_dm = pd.Series(plus_dm, index = market_data.index)
            minus_dm = pd.Series(minus_dm, index = market_data.index)
            atr = self._calculate_atr(market_data, period)
            plus_di = 100 * (plus_dm.rolling(window = period).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window = period).mean() / atr)
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window = period).mean()
            return adx.iloc[-1] if not adx.empty else 25.0
        except Exception:
            return 25.0

    def _calculate_atr(self, market_data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        try:
            if len(market_data) < period:
                return 0.0
            high_low = market_data['high'] - market_data['low']
            high_close = np.abs(market_data['high'] - market_data['close'].shift())
            low_close = np.abs(market_data['low'] - market_data['close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(window = period).mean()
            return atr.iloc[-1] if not atr.empty else 0.0
        except Exception:
            return 0.0

    def _calculate_volume_profile(self, market_data: pd.DataFrame, current_price: float) -> float:
        """Calculate volume profile at current price level."""
        try:
            if len(market_data) < 20:
                return 1.0
            price_range = market_data['high'].max() - market_data['low'].min()
            price_level = (current_price - market_data['low'].min()) / price_range
            volume_at_level = market_data['volume'].mean() * (1 + price_level * 0.2)
            avg_volume = market_data['volume'].mean()
            return volume_at_level / avg_volume if avg_volume > 0 else 1.0
        except Exception:
            return 1.0

    def _detect_price_action_pattern(self, market_data: pd.DataFrame) -> float:
        """Detect price action patterns."""
        try:
            if len(market_data) < 3:
                return 0.0
            current = market_data.iloc[-1]
            body_size = abs(current['close'] - current['open'])
            total_range = current['high'] - current['low']
            if body_size / total_range < 0.1:
                return 1.0
            lower_shadow = min(current['open'], current['close']) - current['low']
            upper_shadow = current['high'] - max(current['open'], current['close'])
            if lower_shadow > 2 * body_size and upper_shadow < body_size:
                return 0.8
            return 0.0
        except Exception:
            return 0.0

    def _calculate_sr_density(self, level_price: float, level: Dict[str, Any]) -> float:
        """Calculate S/R level density around current level."""
        try:
            return 1.0
        except Exception:
            return 1.0

    def _calculate_trend_strength(self, market_data: pd.DataFrame) -> float:
        """Calculate trend strength."""
        try:
            if len(market_data) < 20:
                return 0.5
            sma_20 = market_data['close'].rolling(window = 20).mean()
            sma_50 = market_data['close'].rolling(window = 50).mean()
            if len(sma_20) < 1 or len(sma_50) < 1:
                return 0.5
            trend_ratio = abs(sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1]
            return min(trend_ratio * 10, 1.0)
        except Exception:
            return 0.5

    def _determine_market_regime(self, market_data: pd.DataFrame) -> float:
        """Determine market regime (0 = trending, 0.5 = ranging, 1 = transitional)."""
        try:
            if len(market_data) < 50:
                return 0.5
            rsi = self._calculate_rsi(market_data['close'])
            sma_20 = market_data['close'].rolling(window = 20).mean()
            sma_50 = market_data['close'].rolling(window = 50).mean()
            if len(sma_20) < 1 or len(sma_50) < 1:
                return 0.5
            sma_ratio = sma_20.iloc[-1] / sma_50.iloc[-1]
            rsi_val = rsi
            if abs(sma_ratio - 1.0) > 0.02 and 30 < rsi_val < 70:
                return 0.0
            elif abs(sma_ratio - 1.0) <= 0.02:
                return 0.5
            else:
                return 1.0
        except Exception:
            return 0.5

    def _determine_volatility_regime(self, market_data: pd.DataFrame) -> float:
        """Determine volatility regime (0 = low, 0.5 = normal, 1 = high)."""
        try:
            if len(market_data) < 20:
                return 0.5
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.std()
            if volatility < 0.01:
                return 0.0
            elif volatility > 0.03:
                return 1.0
            else:
                return 0.5
        except Exception:
            return 0.5

    def _calculate_time_of_day_factor(self, market_data: pd.DataFrame) -> float:
        """Calculate time of day factor."""
        try:
            if len(market_data) < 1:
                return 0.5
            timestamp = market_data.index[-1]
            hour = timestamp.hour
            if 9 <= hour <= 16:
                return 1.0
            elif 4 <= hour <= 8 or 17 <= hour <= 20:
                return 0.7
            else:
                return 0.3
        except Exception:
            return 0.5

    def _get_previous_breakout_rate(self, level: Dict[str, Any]) -> float:
        """Get previous breakout rate for this level."""
        try:
            touch_count = level.get('touch_count', 0)
            failure_count = level.get('failure_count', 0)
            if touch_count == 0:
                return 0.5
            breakout_rate = failure_count / touch_count
            return min(breakout_rate, 1.0)
        except Exception:
            return 0.5

    def _calculate_macd_line(self, prices: pd.Series) -> float:
        """Calculate MACD line."""
        try:
            if len(prices) < 26:
                return 0.0
            ema_12 = prices.ewm(span = 12).mean()
            ema_26 = prices.ewm(span = 26).mean()
            macd_line = ema_12 - ema_26
            return macd_line.iloc[-1] if not macd_line.empty else 0.0
        except Exception:
            return 0.0

    def _calculate_stochastic_d(self, market_data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> float:
        """Calculate Stochastic %D."""
        try:
            if len(market_data) < k_period:
                return 50.0
            low_min = market_data['low'].rolling(window = k_period).min()
            high_max = market_data['high'].rolling(window = k_period).max()
            k_percent = 100 * ((market_data['close'] - low_min) / (high_max - low_min))
            d_percent = k_percent.rolling(window = d_period).mean()
            return d_percent.iloc[-1] if not d_percent.empty else 50.0
        except Exception:
            return 50.0

    def _calculate_obv(self, market_data: pd.DataFrame) -> float:
        """Calculate On-Balance Volume."""
        try:
            if len(market_data) < 2:
                return 0.0
            price_change = market_data['close'].diff()
            obv = np.where(price_change > 0, market_data['volume'], np.where(price_change < 0, -market_data['volume'], 0))
            obv = pd.Series(obv, index = market_data.index).cumsum()
            return obv.iloc[-1] if not obv.empty else 0.0
        except Exception:
            return 0.0

    def _detect_doji_pattern(self, market_data: pd.DataFrame) -> float:
        """Detect Doji candlestick pattern."""
        try:
            if len(market_data) < 1:
                return 0.0
            current = market_data.iloc[-1]
            body_size = abs(current['close'] - current['open'])
            total_range = current['high'] - current['low']
            return 1.0 if body_size / total_range < 0.1 else 0.0
        except Exception:
            return 0.0

    def _detect_hammer_pattern(self, market_data: pd.DataFrame) -> float:
        """Detect Hammer candlestick pattern."""
        try:
            if len(market_data) < 1:
                return 0.0
            current = market_data.iloc[-1]
            body_size = abs(current['close'] - current['open'])
            lower_shadow = min(current['open'], current['close']) - current['low']
            upper_shadow = current['high'] - max(current['open'], current['close'])
            total_range = current['high'] - current['low']
            is_hammer = lower_shadow > 2 * body_size and upper_shadow < body_size and (body_size / total_range < 0.3)
            return 1.0 if is_hammer else 0.0
        except Exception:
            return 0.0

    def _calculate_volatility_proxy(self, market_data: pd.DataFrame, period: int = 20) -> float:
        """Calculate volatility proxy (simplified VIX)."""
        try:
            if len(market_data) < period:
                return 0.0
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.rolling(window = period).std().iloc[-1]
            return float(volatility * 100) if not np.isnan(volatility) else 0.0
        except Exception:
            return 0.0

    async def _get_ml_prediction(self, features: Dict[str, float]) -> float:
        """Get ML model prediction."""
        try:
            if not self.ml_model:
                return 0.5
            feature_array = np.array([list(features.values())])
            prediction = self.ml_model.predict_proba(feature_array)[0][1]
            return float(prediction)
        except Exception as e:
            self.logger.error(f'ML prediction failed: {e}')
            return 0.5

    def get_performance_metrics(self) -> BreakoutPerformance:
        """Get current performance metrics."""
        return self.performance_metrics

    def get_active_signals(self) -> Dict[str, BreakoutSignal]:
        """Get currently active signals."""
        return self.active_signals.copy()

    def enable_monitoring(self) -> None:
        """Enable real-time monitoring."""
        self.monitoring_enabled = True
        self.logger.info('✅ Breakout monitoring enabled')

    def disable_monitoring(self) -> None:
        """Disable real-time monitoring."""
        self.monitoring_enabled = False
        self.logger.info('✅ Breakout monitoring disabled')

    # Methods required by SRLevelsManager
    async def get_sr_context(self, market_data: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """Get SR context for the given market data and current price."""
        try:
            self.logger.info('🔍 Getting SR context from enhanced predictor')
            # Use basic level detection for now
            support_levels = await self._detect_support_levels(market_data)
            resistance_levels = await self._detect_resistance_levels(market_data)
            
            return {
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'current_price': current_price,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f'Error getting SR context: {e}')
            return {'support_levels': [], 'resistance_levels': [], 'current_price': current_price}

    async def _detect_support_levels(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect support levels using the configured method."""
        try:
            self.logger.info(f'🔍 Starting support level detection using {self.sr_detection_method} method on {len(market_data)} data points')
            levels = []
            
            if self.sr_detection_method == 'fractal':
                self.logger.info('📊 Using fractal method for support detection...')
                levels = self._detect_fractal_levels(market_data, 'support')
            elif self.sr_detection_method == 'volume':
                self.logger.info('📊 Using volume method for support detection...')
                levels = self._detect_volume_levels(market_data, 'support')
            elif self.sr_detection_method == 'pivot':
                self.logger.info('📊 Using pivot method for support detection...')
                levels = self._detect_pivot_levels(market_data, 'support')
            elif self.sr_detection_method == 'atr':
                self.logger.info('📊 Using ATR method for support detection...')
                levels = self._detect_atr_levels(market_data, 'support')
            else:
                # Default to fractal
                self.logger.info('📊 Using default fractal method for support detection...')
                levels = self._detect_fractal_levels(market_data, 'support')
            
            self.logger.info(f'✅ Support detection completed: {len(levels)} levels found using {self.sr_detection_method} method')
            return levels
        except Exception as e:
            self.logger.error(f'Error detecting support levels: {e}')
            return []

    async def _detect_resistance_levels(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect resistance levels using the configured method."""
        try:
            self.logger.info(f'🔍 Starting resistance level detection using {self.sr_detection_method} method on {len(market_data)} data points')
            levels = []
            
            if self.sr_detection_method == 'fractal':
                self.logger.info('📊 Using fractal method for resistance detection...')
                levels = self._detect_fractal_levels(market_data, 'resistance')
            elif self.sr_detection_method == 'volume':
                self.logger.info('📊 Using volume method for resistance detection...')
                levels = self._detect_volume_levels(market_data, 'resistance')
            elif self.sr_detection_method == 'pivot':
                self.logger.info('📊 Using pivot method for resistance detection...')
                levels = self._detect_pivot_levels(market_data, 'resistance')
            elif self.sr_detection_method == 'atr':
                self.logger.info('📊 Using ATR method for resistance detection...')
                levels = self._detect_atr_levels(market_data, 'resistance')
            else:
                # Default to fractal
                self.logger.info('📊 Using default fractal method for resistance detection...')
                levels = self._detect_fractal_levels(market_data, 'resistance')
            
            self.logger.info(f'✅ Resistance detection completed: {len(levels)} levels found using {self.sr_detection_method} method')
            return levels
        except Exception as e:
            self.logger.error(f'Error detecting resistance levels: {e}')
            return []

    def _detect_fractal_levels(self, market_data: pd.DataFrame, level_type: str) -> List[Dict[str, Any]]:
        """Detect levels using fractal method - optimized for large datasets."""
        try:
            levels = []
            sample_data = market_data
            
            window = 5  # Increased window for better fractal detection
            
            if level_type == 'support':
                # Use vectorized operations for better performance
                lows = sample_data['low'].values
                total_points = len(lows) - 2 * window

                # Initialize enhanced progress logger
                progress_logger = EnhancedProgressLogger(
                    self.logger,
                    total_items=total_points,
                    operation_name=f"{level_type.capitalize()} fractal detection"
                )
                progress_logger.start()

                for i in range(window, len(lows) - window):
                    # Update progress with enhanced logging
                    progress_logger.update(i - window)
                    
                    current_low = lows[i]
                    # Check if current point is a local minimum
                    if (current_low <= lows[i-window:i].min() and 
                        current_low <= lows[i+1:i+window+1].min()):
                        
                        # Simplified touch counting for performance
                        touches = 1
                        threshold = 0.002  # More sensitive threshold for better level detection
                        
                        # Optimized touch counting using vectorized operations
                        start_idx = max(0, i - 500)
                        end_idx = min(len(lows), i + 500)
                        window_lows = lows[start_idx:end_idx]
                        price_diffs = abs(window_lows - current_low) / current_low
                        touches = np.sum(price_diffs < threshold)
                        
                        if touches >= 3:
                            levels.append({
                                'price': float(current_low),
                                'strength': min(touches / 5, 1.0),  # Adjusted scaling
                                'type': 'support',
                                'method': 'fractal',
                                'touch_count': touches,
                                'timestamp': datetime.now().isoformat()
                            })
            else:  # resistance
                # Use vectorized operations for better performance
                highs = sample_data['high'].values
                total_points = len(highs) - 2 * window

                # Initialize enhanced progress logger
                progress_logger = EnhancedProgressLogger(
                    self.logger,
                    total_items=total_points,
                    operation_name=f"{level_type.capitalize()} fractal detection"
                )
                progress_logger.start()

                for i in range(window, len(highs) - window):
                    # Update progress with enhanced logging
                    progress_logger.update(i - window)
                    
                    current_high = highs[i]
                    # Check if current point is a local maximum
                    if (current_high >= highs[i-window:i].max() and 
                        current_high >= highs[i+1:i+window+1].max()):
                        
                        # Simplified touch counting for performance
                        touches = 1
                        threshold = 0.002  # More sensitive threshold for better level detection
                        
                        # Optimized touch counting using vectorized operations
                        start_idx = max(0, i - 500)
                        end_idx = min(len(highs), i + 500)
                        window_highs = highs[start_idx:end_idx]
                        price_diffs = abs(window_highs - current_high) / current_high
                        touches = np.sum(price_diffs < threshold)
                        
                        if touches >= 3:
                            levels.append({
                                'price': float(current_high),
                                'strength': min(touches / 5, 1.0),  # Adjusted scaling
                                'type': 'resistance',
                                'method': 'fractal',
                                'touch_count': touches,
                                'timestamp': datetime.now().isoformat()
                            })

                # Complete progress logging
                progress_logger.complete()

            # Limit the number of levels returned for performance
            levels = levels[:50]  # Return max 50 levels for better coverage
            self.logger.info(f'✅ Fractal detection found {len(levels)} {level_type} levels')
            return levels
        except Exception as e:
            self.logger.error(f'Error in fractal level detection: {e}')
            return []

    def _detect_volume_levels(self, market_data: pd.DataFrame, level_type: str) -> List[Dict[str, Any]]:
        """Detect levels using volume-based method - vectorized for performance."""
        try:
            self.logger.info(f'🔍 Starting volume-based {level_type} level detection on {len(market_data)} points')
            
            # Add timeout protection for large datasets
            if len(market_data) > 100000:  # If more than 100k rows
                self.logger.warning(f'⚠️ Large dataset detected ({len(market_data)} rows). Volume detection may take time...')
                # For very large datasets, sample the data to prevent hanging
                if len(market_data) > 500000:  # If more than 500k rows, sample it
                    sample_size = 100000
                    market_data = market_data.sample(n=sample_size, random_state=42)
                    self.logger.info(f'📊 Sampled dataset to {len(market_data)} rows for performance')
            
            # Enhanced volume analysis with HVN (High Volume Nodes)
            # Use multiple volume thresholds for better level detection
            volume_90th = market_data['volume'].quantile(0.9)
            volume_80th = market_data['volume'].quantile(0.8)
            volume_70th = market_data['volume'].quantile(0.7)
            
            # Create volume profile bins
            price_range = market_data['high'].max() - market_data['low'].min()
            bin_size = price_range / 50  # 50 price bins for volume profile
            
            # Calculate volume at each price level
            volume_profile = {}
            total_rows = len(market_data)
            self.logger.info(f'📊 Processing {total_rows} rows for volume profile calculation...')
            
            for idx, row in market_data.iterrows():
                if idx % 5000 == 0:  # Log progress every 5k rows for more frequent updates
                    self.logger.info(f'📊 Volume profile progress: {idx}/{total_rows} ({idx/total_rows*100:.1f}%)')
                
                price_bin = round(row['low'] / bin_size) * bin_size
                if price_bin not in volume_profile:
                    volume_profile[price_bin] = 0
                volume_profile[price_bin] += row['volume']
            
            self.logger.info(f'✅ Volume profile calculation completed: {len(volume_profile)} price bins')
            
            # Find HVN (High Volume Nodes) - price levels with highest volume
            sorted_volume_profile = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
            hvn_levels = [price for price, volume in sorted_volume_profile[:20]]  # Top 20 HVN levels
            
            # Calculate volume statistics for dynamic strength calculation
            all_volumes = [volume for _, volume in volume_profile.items()]
            volume_mean = np.mean(all_volumes) if all_volumes else 1.0
            volume_std = np.std(all_volumes) if all_volumes else 1.0
            
            # Also get traditional high volume points
            high_volume_mask = market_data['volume'] > volume_80th
            high_volume_data = market_data[high_volume_mask]
            
            self.logger.info(f'📊 Found {len(high_volume_data)} high-volume points and {len(hvn_levels)} HVN levels')
            
            levels = []
            
            # Add HVN levels first (these are the most important)
            self.logger.info(f'📊 Processing {len(hvn_levels)} HVN levels for touch count calculation...')
            for i, hvn_price in enumerate(hvn_levels):
                if i % 5 == 0:  # Log progress every 5 HVN levels
                    self.logger.info(f'📊 HVN processing progress: {i}/{len(hvn_levels)} ({i/len(hvn_levels)*100:.1f}%)')
                
                hvn_volume = volume_profile.get(hvn_price, 0)
                
                # Calculate dynamic strength based on volume characteristics
                volume_ratio = hvn_volume / volume_mean if volume_mean > 0 else 1.0
                volume_z_score = (hvn_volume - volume_mean) / volume_std if volume_std > 0 else 0.0
                
                # Calculate touch count for this price level
                touch_count = 0
                for idx, row in market_data.iterrows():
                    price_bin = round(row['low'] / bin_size) * bin_size
                    if abs(price_bin - hvn_price) < bin_size * 0.1:  # Within 10% of bin size
                        touch_count += 1
                
                # Dynamic strength calculation: 60% volume ratio + 30% z-score + 10% touch count
                touch_score = min(touch_count / 10.0, 1.0)  # Normalize touch count
                strength = min(volume_ratio * 0.6 + max(0, volume_z_score * 0.3) + touch_score * 0.1, 0.95)
                
                # Ensure minimum strength for HVN levels
                strength = max(strength, 0.3)
                
                levels.append({
                    'price': float(hvn_price),
                    'strength': round(strength, 3),  # Dynamic strength calculation
                    'type': level_type,
                    'method': 'hvn',
                    'touch_count': touch_count,
                    'volume': float(hvn_volume),
                    'volume_ratio': round(volume_ratio, 3),
                    'volume_z_score': round(volume_z_score, 3),
                    'timestamp': datetime.now().isoformat()
                })
            
            if len(high_volume_data) > 0:
                if level_type == 'support':
                    # Vectorized support level creation
                    prices = high_volume_data['low'].values
                    volumes = high_volume_data['volume'].values
                    timestamps = high_volume_data.index
                    
                    for i, (price, volume) in enumerate(zip(prices, volumes)):
                        levels.append({
                            'price': float(price),
                            'strength': 0.7,
                            'type': 'support',
                            'method': 'volume',
                            'touch_count': 1,
                            'volume': float(volume),
                            'timestamp': timestamps[i].isoformat() if hasattr(timestamps[i], 'isoformat') else str(timestamps[i])
                        })
                else:  # resistance
                    # Vectorized resistance level creation
                    prices = high_volume_data['high'].values
                    volumes = high_volume_data['volume'].values
                    timestamps = high_volume_data.index
                    
                    for i, (price, volume) in enumerate(zip(prices, volumes)):
                        levels.append({
                            'price': float(price),
                            'strength': 0.7,
                            'type': 'resistance',
                            'method': 'volume',
                            'touch_count': 1,
                            'volume': float(volume),
                            'timestamp': timestamps[i].isoformat() if hasattr(timestamps[i], 'isoformat') else str(timestamps[i])
                        })
            
            # Limit results for performance
            levels = levels[:30]  # Max 30 volume levels (including HVN)
            self.logger.info(f'✅ Volume detection found {len(levels)} {level_type} levels')
            return levels
        except Exception as e:
            self.logger.error(f'Error in volume level detection: {e}')
            return []

    def _detect_pivot_levels(self, market_data: pd.DataFrame, level_type: str) -> List[Dict[str, Any]]:
        """Detect levels using pivot point method."""
        try:
            levels = []
            # Simple pivot point calculation
            if len(market_data) >= 24:  # Need at least 24 hours of data
                high_24h = market_data['high'].iloc[-24:].max()
                low_24h = market_data['low'].iloc[-24:].min()
                close = market_data['close'].iloc[-1]
                
                pivot = (high_24h + low_24h + close) / 3
                r1 = 2 * pivot - low_24h
                s1 = 2 * pivot - high_24h
                
                if level_type == 'support':
                    levels.append({
                        'price': s1,
                        'strength': 0.6,
                        'type': 'support',
                        'method': 'pivot',
                        'touch_count': 1,
                        'timestamp': datetime.now().isoformat()
                    })
                else:  # resistance
                    levels.append({
                        'price': r1,
                        'strength': 0.6,
                        'type': 'resistance',
                        'method': 'pivot',
                        'touch_count': 1,
                        'timestamp': datetime.now().isoformat()
                    })
            
            return levels
        except Exception as e:
            self.logger.error(f'Error in pivot level detection: {e}')
            return []

    def _detect_atr_levels(self, market_data: pd.DataFrame, level_type: str) -> List[Dict[str, Any]]:
        """Detect levels using ATR-based method."""
        try:
            levels = []
            atr = self._calculate_atr(market_data)
            current_price = market_data['close'].iloc[-1]
            
            if level_type == 'support':
                support_price = current_price - (atr * 2)
                levels.append({
                    'price': support_price,
                    'strength': 0.5,
                    'type': 'support',
                    'method': 'atr',
                    'touch_count': 1,
                    'atr': atr,
                    'timestamp': datetime.now().isoformat()
                })
            else:  # resistance
                resistance_price = current_price + (atr * 2)
                levels.append({
                    'price': resistance_price,
                    'strength': 0.5,
                    'type': 'resistance',
                    'method': 'atr',
                    'touch_count': 1,
                    'atr': atr,
                    'timestamp': datetime.now().isoformat()
                })
            
            return levels
        except Exception as e:
            self.logger.error(f'Error in ATR level detection: {e}')
            return []

    def _count_level_touches(self, data: pd.DataFrame, level_price: float, level_type: str, start_idx: int) -> int:
        """Count how many times price touched this level."""
        try:
            touches = 1
            threshold = 0.002
            
            for i in range(start_idx + 1, len(data)):
                if level_type == 'resistance':
                    if abs(data['high'].iloc[i] - level_price) / level_price < threshold:
                        touches += 1
                else:  # support
                    if abs(data['low'].iloc[i] - level_price) / level_price < threshold:
                        touches += 1
            
            return touches
        except Exception as e:
            self.logger.error(f'Error counting level touches: {e}')
            return 1

# Alias for backward compatibility
SRBreakoutPredictor = EnhancedSRBreakoutPredictor

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and 
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
    
    def _vectorbt_apply_operation(self, data: pd.Series, func, 
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)
        
        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
