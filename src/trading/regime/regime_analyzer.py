"""
Regime Analyzer

Advanced regime analysis including regime transitions,
stability analysis, and regime-specific feature extraction.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, log_execution_time
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success
from ..config.regime_config import RegimeConfig, RegimeType

logger = system_logger.getChild('RegimeAnalyzer')

class RegimeAnalyzer:
    """
    Advanced regime analysis for market regime detection.

    Provides regime transition analysis, stability metrics,
    and regime-specific feature extraction.
    """

    def __init__(self, config: RegimeConfig):
        self.config = config
        self.logger = logger.getChild('RegimeAnalyzer')

        # Analysis parameters
        self.lookback_periods = {
            'short': 20,
            'medium': 50,
            'long': 100
        }

        # Regime transition matrix
        self.transition_matrix: Dict[Tuple[RegimeType, RegimeType], float] = {}

        # Regime stability metrics
        self.stability_metrics: Dict[RegimeType, Dict[str, float]] = {}

        # Analysis history
        self.analysis_history: List[Dict[str, Any]] = []
        self.max_history = 1000

    @handles_errors
    async def initialize(self) -> bool:
        """Initialize regime analyzer."""
        try:
            tprint_info("🔄 Initializing Regime Analyzer...")

            # Initialize transition matrix
            await self._initialize_transition_matrix()

            # Initialize stability metrics
            await self._initialize_stability_metrics()

            # Load historical analysis data
            await self._load_historical_data()

            tprint_success("✅ Regime Analyzer initialized")
            return True

        except Exception as e:
            tprint_error(f"❌ Failed to initialize Regime Analyzer: {e}")
            return False

    async def _initialize_transition_matrix(self):
        """Initialize regime transition probability matrix."""
        try:
            # Initialize with default transition probabilities
            regimes = list(RegimeType)

            for from_regime in regimes:
                for to_regime in regimes:
                    if from_regime == to_regime:
                        # High probability of staying in same regime
                        self.transition_matrix[(from_regime, to_regime)] = 0.8
                    else:
                        # Low probability of transitioning to different regime
                        self.transition_matrix[(from_regime, to_regime)] = 0.2 / (len(regimes) - 1)

            self.logger.info("✅ Transition matrix initialized")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize transition matrix: {e}")

    async def _initialize_stability_metrics(self):
        """Initialize regime stability metrics."""
        try:
            for regime in RegimeType:
                self.stability_metrics[regime] = {
                    'average_duration': 10.0,  # Average regime duration in periods
                    'volatility': 0.5,  # Regime volatility
                    'persistence': 0.8,  # Regime persistence score
                    'confidence': 0.7  # Default confidence
                }

            self.logger.info("✅ Stability metrics initialized")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize stability metrics: {e}")

    async def _load_historical_data(self):
        """Load historical analysis data if available."""
        try:
            # Try to load from cache
            import os
            import json

            cache_file = "data_cache/regime_analysis_history.json"
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    data = json.load(f)

                # Load transition matrix
                if 'transition_matrix' in data:
                    for key_str, value in data['transition_matrix'].items():
                        # Convert string key back to tuple
                        from_regime, to_regime = key_str.split('|')
                        self.transition_matrix[(RegimeType(from_regime), RegimeType(to_regime))] = value

                # Load stability metrics
                if 'stability_metrics' in data:
                    for regime_str, metrics in data['stability_metrics'].items():
                        self.stability_metrics[RegimeType(regime_str)] = metrics

                self.logger.info("✅ Historical analysis data loaded")
            else:
                self.logger.info("📝 No historical analysis data found, using defaults")

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load historical data: {e}")

    @handles_errors
    @traced(span_name="regime_analysis")
    @log_execution_time()
    async def analyze_regime_stability(
        self,
        regime_history: List[Dict[str, Any]],
        current_regime: RegimeType,
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Analyze regime stability and transition patterns.

        Args:
            regime_history: Historical regime detections
            current_regime: Current detected regime
            market_data: Market data for analysis

        Returns:
            Dictionary with stability analysis results
        """
        try:
            if not regime_history:
                return self._default_stability_analysis()

            # Analyze regime transitions
            transition_analysis = await self._analyze_transitions(regime_history)

            # Calculate regime persistence
            persistence_analysis = await self._analyze_persistence(regime_history, current_regime)

            # Analyze regime duration patterns
            duration_analysis = await self._analyze_durations(regime_history)

            # Calculate stability scores
            stability_scores = await self._calculate_stability_scores(
                transition_analysis, persistence_analysis, duration_analysis
            )

            # Market condition analysis
            market_condition_analysis = await self._analyze_market_conditions(
                market_data, current_regime
            )

            # Combine all analyses
            analysis_result = {
                'timestamp': datetime.now().isoformat(),
                'current_regime': current_regime.value,
                'stability_score': stability_scores.get('overall', 0.5),
                'transition_analysis': transition_analysis,
                'persistence_analysis': persistence_analysis,
                'duration_analysis': duration_analysis,
                'stability_scores': stability_scores,
                'market_conditions': market_condition_analysis,
                'regime_forecast': await self._forecast_regime_changes(regime_history, current_regime)
            }

            # Store in history
            self._store_analysis(analysis_result)

            return analysis_result

        except Exception as e:
            self.logger.error(f"❌ Regime stability analysis failed: {e}")
            return self._default_stability_analysis()

    async def _analyze_transitions(self, regime_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze regime transition patterns."""
        try:
            if len(regime_history) < 2:
                return {'transition_count': 0, 'transition_rate': 0.0}

            transitions = []
            transition_count = 0

            for i in range(1, len(regime_history)):
                prev_regime = RegimeType(regime_history[i-1]['primary_regime'])
                curr_regime = RegimeType(regime_history[i]['primary_regime'])

                if prev_regime != curr_regime:
                    transitions.append((prev_regime, curr_regime))
                    transition_count += 1

            # Calculate transition rate
            transition_rate = transition_count / len(regime_history)

            # Update transition matrix
            await self._update_transition_matrix(transitions)

            # Most common transitions
            transition_counts = {}
            for transition in transitions:
                transition_counts[transition] = transition_counts.get(transition, 0) + 1

            most_common = sorted(transition_counts.items(), key=lambda x: x[1], reverse=True)[:3]

            return {
                'transition_count': transition_count,
                'transition_rate': transition_rate,
                'total_periods': len(regime_history),
                'most_common_transitions': [
                    {
                        'from': trans[0][0].value,
                        'to': trans[0][1].value,
                        'count': trans[1],
                        'probability': trans[1] / transition_count if transition_count > 0 else 0.0
                    }
                    for trans in most_common
                ]
            }

        except Exception as e:
            self.logger.error(f"❌ Transition analysis failed: {e}")
            return {'transition_count': 0, 'transition_rate': 0.0}

    async def _update_transition_matrix(self, transitions: List[Tuple[RegimeType, RegimeType]]):
        """Update transition probability matrix based on observed transitions."""
        try:
            if not transitions:
                return

            # Count transitions
            transition_counts = {}
            for transition in transitions:
                transition_counts[transition] = transition_counts.get(transition, 0) + 1

            # Update transition matrix with exponential smoothing
            alpha = 0.1  # Learning rate

            for transition, count in transition_counts.items():
                current_prob = self.transition_matrix.get(transition, 0.0)
                observed_prob = count / len(transitions)

                # Exponential smoothing update
                new_prob = alpha * observed_prob + (1 - alpha) * current_prob
                self.transition_matrix[transition] = new_prob

        except Exception as e:
            self.logger.warning(f"⚠️ Transition matrix update failed: {e}")

    async def _analyze_persistence(
        self,
        regime_history: List[Dict[str, Any]],
        current_regime: RegimeType
    ) -> Dict[str, Any]:
        """Analyze regime persistence patterns."""
        try:
            if not regime_history:
                return {'current_duration': 0, 'persistence_score': 0.5}

            # Calculate current regime duration
            current_duration = 0
            for i in range(len(regime_history) - 1, -1, -1):
                if RegimeType(regime_history[i]['primary_regime']) == current_regime:
                    current_duration += 1
                else:
                    break

            # Calculate regime durations for all regimes
            regime_durations = {regime: [] for regime in RegimeType}
            current_regime_type = None
            current_duration_temp = 0

            for detection in regime_history:
                regime = RegimeType(detection['primary_regime'])

                if regime == current_regime_type:
                    current_duration_temp += 1
                else:
                    if current_regime_type is not None and current_duration_temp > 0:
                        regime_durations[current_regime_type].append(current_duration_temp)
                    current_regime_type = regime
                    current_duration_temp = 1

            # Add final duration
            if current_regime_type is not None:
                regime_durations[current_regime_type].append(current_duration_temp)

            # Calculate persistence metrics
            avg_duration = np.mean(regime_durations[current_regime]) if regime_durations[current_regime] else 1.0
            max_duration = np.max(regime_durations[current_regime]) if regime_durations[current_regime] else 1.0

            # Persistence score based on current duration vs average
            persistence_score = min(current_duration / avg_duration, 2.0) / 2.0 if avg_duration > 0 else 0.5

            return {
                'current_duration': current_duration,
                'average_duration': avg_duration,
                'max_duration': max_duration,
                'persistence_score': persistence_score,
                'regime_durations': {k.value: v for k, v in regime_durations.items() if v}
            }

        except Exception as e:
            self.logger.error(f"❌ Persistence analysis failed: {e}")
            return {'current_duration': 0, 'persistence_score': 0.5}

    async def _analyze_durations(self, regime_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze regime duration statistics."""
        try:
            if len(regime_history) < 2:
                return {'total_regimes': 0, 'avg_duration': 0.0}

            # Extract regime sequences
            regime_sequences = []
            current_regime = None
            current_start = 0

            for i, detection in enumerate(regime_history):
                regime = RegimeType(detection['primary_regime'])

                if regime != current_regime:
                    if current_regime is not None:
                        duration = i - current_start
                        regime_sequences.append({
                            'regime': current_regime,
                            'duration': duration,
                            'start_index': current_start,
                            'end_index': i - 1
                        })
                    current_regime = regime
                    current_start = i

            # Add final sequence
            if current_regime is not None:
                duration = len(regime_history) - current_start
                regime_sequences.append({
                    'regime': current_regime,
                    'duration': duration,
                    'start_index': current_start,
                    'end_index': len(regime_history) - 1
                })

            # Calculate duration statistics
            durations = [seq['duration'] for seq in regime_sequences]

            if durations:
                avg_duration = np.mean(durations)
                median_duration = np.median(durations)
                std_duration = np.std(durations)
                min_duration = np.min(durations)
                max_duration = np.max(durations)
            else:
                avg_duration = median_duration = std_duration = min_duration = max_duration = 0.0

            return {
                'total_regimes': len(regime_sequences),
                'avg_duration': avg_duration,
                'median_duration': median_duration,
                'std_duration': std_duration,
                'min_duration': min_duration,
                'max_duration': max_duration,
                'regime_sequences': [
                    {
                        'regime': seq['regime'].value,
                        'duration': seq['duration'],
                        'start_index': seq['start_index'],
                        'end_index': seq['end_index']
                    }
                    for seq in regime_sequences[-10:]  # Last 10 sequences
                ]
            }

        except Exception as e:
            self.logger.error(f"❌ Duration analysis failed: {e}")
            return {'total_regimes': 0, 'avg_duration': 0.0}

    async def _calculate_stability_scores(
        self,
        transition_analysis: Dict[str, Any],
        persistence_analysis: Dict[str, Any],
        duration_analysis: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate overall stability scores."""
        try:
            # Transition stability (lower transition rate = higher stability)
            transition_rate = transition_analysis.get('transition_rate', 0.5)
            transition_stability = 1.0 - min(transition_rate, 1.0)

            # Persistence stability
            persistence_score = persistence_analysis.get('persistence_score', 0.5)

            # Duration stability (higher average duration = higher stability)
            avg_duration = duration_analysis.get('avg_duration', 1.0)
            duration_stability = min(avg_duration / 10.0, 1.0)  # Normalize to max 10 periods

            # Overall stability (weighted combination)
            overall_stability = (
                0.4 * transition_stability +
                0.4 * persistence_score +
                0.2 * duration_stability
            )

            return {
                'overall': overall_stability,
                'transition_stability': transition_stability,
                'persistence_stability': persistence_score,
                'duration_stability': duration_stability
            }

        except Exception as e:
            self.logger.error(f"❌ Stability score calculation failed: {e}")
            return {'overall': 0.5}

    async def _analyze_market_conditions(
        self,
        market_data: pd.DataFrame,
        current_regime: RegimeType
    ) -> Dict[str, Any]:
        """Analyze current market conditions in context of regime."""
        try:
            if len(market_data) < 20:
                return {'condition': 'insufficient_data'}

            # Calculate market metrics
            close_prices = market_data['close'].values
            returns = np.diff(close_prices) / close_prices[:-1]

            # Volatility metrics
            short_vol = np.std(returns[-5:]) if len(returns) >= 5 else 0.0
            medium_vol = np.std(returns[-20:]) if len(returns) >= 20 else 0.0

            # Trend metrics
            short_trend = np.mean(returns[-5:]) if len(returns) >= 5 else 0.0
            medium_trend = np.mean(returns[-20:]) if len(returns) >= 20 else 0.0

            # Volume metrics (if available)
            if 'volume' in market_data.columns:
                volume_trend = (
                    market_data['volume'].iloc[-5:].mean() /
                    market_data['volume'].iloc[-20:-5].mean()
                    if len(market_data) >= 20 else 1.0
                )
            else:
                volume_trend = 1.0

            # Regime consistency check
            regime_consistency = await self._check_regime_consistency(
                current_regime, short_vol, medium_vol, short_trend, medium_trend
            )

            return {
                'volatility': {
                    'short_term': short_vol,
                    'medium_term': medium_vol,
                    'ratio': short_vol / medium_vol if medium_vol > 0 else 1.0
                },
                'trend': {
                    'short_term': short_trend,
                    'medium_term': medium_trend,
                    'consistency': 1.0 if short_trend * medium_trend >= 0 else 0.0
                },
                'volume': {
                    'trend': volume_trend,
                    'strength': 'high' if volume_trend > 1.2 else 'low' if volume_trend < 0.8 else 'normal'
                },
                'regime_consistency': regime_consistency,
                'market_phase': await self._determine_market_phase(short_vol, medium_vol, short_trend, medium_trend)
            }

        except Exception as e:
            self.logger.error(f"❌ Market conditions analysis failed: {e}")
            return {'condition': 'analysis_error'}

    async def _check_regime_consistency(
        self,
        regime: RegimeType,
        short_vol: float,
        medium_vol: float,
        short_trend: float,
        medium_trend: float
    ) -> Dict[str, Any]:
        """Check if current market conditions are consistent with detected regime."""
        try:
            consistency_score = 0.5  # Default
            reasons = []

            # Volatility consistency
            if regime == RegimeType.HIGH_VOLATILITY:
                if short_vol > 0.03:  # High volatility threshold
                    consistency_score += 0.2
                    reasons.append("High volatility matches regime")
                else:
                    reasons.append("Low volatility inconsistent with high volatility regime")
            elif regime == RegimeType.LOW_VOLATILITY:
                if short_vol < 0.01:  # Low volatility threshold
                    consistency_score += 0.2
                    reasons.append("Low volatility matches regime")
                else:
                    reasons.append("High volatility inconsistent with low volatility regime")

            # Trend consistency
            if regime == RegimeType.TRENDING_UP:
                if short_trend > 0.001:
                    consistency_score += 0.2
                    reasons.append("Positive trend matches upward regime")
                else:
                    reasons.append("Negative trend inconsistent with upward regime")
            elif regime == RegimeType.TRENDING_DOWN:
                if short_trend < -0.001:
                    consistency_score += 0.2
                    reasons.append("Negative trend matches downward regime")
                else:
                    reasons.append("Positive trend inconsistent with downward regime")
            elif regime == RegimeType.SIDEWAYS:
                if abs(short_trend) < 0.001:
                    consistency_score += 0.2
                    reasons.append("Neutral trend matches sideways regime")
                else:
                    reasons.append("Strong trend inconsistent with sideways regime")

            return {
                'score': min(max(consistency_score, 0.0), 1.0),
                'reasons': reasons,
                'is_consistent': consistency_score > 0.6
            }

        except Exception as e:
            self.logger.error(f"❌ Regime consistency check failed: {e}")
            return {'score': 0.5, 'reasons': [], 'is_consistent': False}

    async def _determine_market_phase(
        self,
        short_vol: float,
        medium_vol: float,
        short_trend: float,
        medium_trend: float
    ) -> str:
        """Determine current market phase."""
        try:
            # High volatility phases
            if short_vol > 0.03 or medium_vol > 0.025:
                if abs(short_trend) > 0.002:
                    return "high_volatility_trending"
                else:
                    return "high_volatility_choppy"

            # Low volatility phases
            elif short_vol < 0.01 and medium_vol < 0.015:
                if abs(medium_trend) > 0.001:
                    return "low_volatility_trending"
                else:
                    return "low_volatility_consolidation"

            # Medium volatility phases
            else:
                if short_trend * medium_trend > 0 and abs(medium_trend) > 0.001:
                    return "medium_volatility_trending"
                else:
                    return "medium_volatility_mixed"

        except Exception as e:
            self.logger.error(f"❌ Market phase determination failed: {e}")
            return "unknown"

    async def _forecast_regime_changes(
        self,
        regime_history: List[Dict[str, Any]],
        current_regime: RegimeType
    ) -> Dict[str, Any]:
        """Forecast potential regime changes."""
        try:
            if len(regime_history) < 10:
                return {'forecast': 'insufficient_data', 'probability': 0.5}

            # Calculate transition probabilities from current regime
            next_regime_probs = {}
            for regime in RegimeType:
                prob = self.transition_matrix.get((current_regime, regime), 0.0)
                next_regime_probs[regime.value] = prob

            # Find most likely next regime (excluding current)
            next_regime_candidates = [(k, v) for k, v in next_regime_probs.items()
                                    if k != current_regime.value]

            if next_regime_candidates:
                most_likely_next = max(next_regime_candidates, key=lambda x: x[1])

                return {
                    'forecast': 'transition_analysis',
                    'current_regime': current_regime.value,
                    'most_likely_next': most_likely_next[0],
                    'transition_probability': most_likely_next[1],
                    'stay_probability': next_regime_probs[current_regime.value],
                    'all_probabilities': next_regime_probs
                }
            else:
                return {
                    'forecast': 'stay_current',
                    'probability': next_regime_probs.get(current_regime.value, 0.8)
                }

        except Exception as e:
            self.logger.error(f"❌ Regime forecast failed: {e}")
            return {'forecast': 'error', 'probability': 0.5}

    def _store_analysis(self, analysis_result: Dict[str, Any]):
        """Store analysis result in history."""
        self.analysis_history.append(analysis_result)

        # Maintain history size
        if len(self.analysis_history) > self.max_history:
            self.analysis_history.pop(0)

    def _default_stability_analysis(self) -> Dict[str, Any]:
        """Return default stability analysis."""
        return {
            'timestamp': datetime.now().isoformat(),
            'current_regime': 'unknown',
            'stability_score': 0.5,
            'transition_analysis': {'transition_count': 0, 'transition_rate': 0.0},
            'persistence_analysis': {'current_duration': 0, 'persistence_score': 0.5},
            'duration_analysis': {'total_regimes': 0, 'avg_duration': 0.0},
            'stability_scores': {'overall': 0.5},
            'market_conditions': {'condition': 'unknown'},
            'regime_forecast': {'forecast': 'unknown', 'probability': 0.5}
        }

    def get_analysis_history(self, n: int = 100) -> List[Dict[str, Any]]:
        """Get recent analysis history."""
        return self.analysis_history[-n:] if len(self.analysis_history) >= n else self.analysis_history.copy()

    def get_transition_matrix(self) -> Dict[str, Dict[str, float]]:
        """Get current transition matrix."""
        result = {}
        for (from_regime, to_regime), prob in self.transition_matrix.items():
            from_key = from_regime.value
            if from_key not in result:
                result[from_key] = {}
            result[from_key][to_regime.value] = prob
        return result

    async def save_analysis_data(self):
        """Save analysis data to cache."""
        try:

            # Prepare data for saving
            save_data = {
                'transition_matrix': {
                    f"{from_regime.value}|{to_regime.value}": prob
                    for (from_regime, to_regime), prob in self.transition_matrix.items()
                },
                'stability_metrics': {
                    regime.value: metrics
                    for regime, metrics in self.stability_metrics.items()
                },
                'analysis_count': len(self.analysis_history)
            }

            # Ensure directory exists
            os.makedirs("data_cache", exist_ok=True)

            # Save to file
            cache_file = "data_cache/regime_analysis_history.json"
            with open(cache_file, 'w') as f:
                json.dump(save_data, f, indent=2)

            self.logger.info("✅ Analysis data saved to cache")

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save analysis data: {e}")

    async def stop(self):
        """Stop regime analyzer."""
        try:
            self.logger.info("🛑 Stopping Regime Analyzer...")

            # Save analysis data
            await self.save_analysis_data()

            # Clear data
            self.analysis_history.clear()

            self.logger.info("✅ Regime Analyzer stopped")

        except Exception as e:
            self.logger.error(f"❌ Error stopping Regime Analyzer: {e}")
