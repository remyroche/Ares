"""
Tactician 5m Entry Optimizer - Trains on Analyst 15m Green Lights

This component implements the Tactician's core functionality:
- Operates on 5m timeframe data
- Trains exclusively on Analyst (15m) green light periods
- Finds optimal entry points with minimal adverse price movement
- Uses advanced entry timing optimization techniques

The Tactician is designed to be the "execution specialist" that takes
Analyst signals and finds the absolute best moment to enter positions.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import traceback

# Import utilities
try:
    from src.utils.logger import system_logger
    from src.utils.tprint import (
        tprint, tprint_info, tprint_warning, tprint_error, tprint_success,
        tprint_debug, tprint_progress, tprint_timer
    )
    UTILS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Failed to import utilities: {e}")
    UTILS_AVAILABLE = False


class EntryOptimizationMethod(Enum):
    """Entry optimization methods for Tactician."""
    PEAK_FINDER = "peak_finder"              # Current peak-finding approach
    MULTI_TIMEFRAME = "multi_timeframe"      # Multi-timeframe optimization
    PATTERN_BASED = "pattern_based"          # Pattern recognition
    ML_ENHANCED = "ml_enhanced"             # ML-enhanced scoring
    HYBRID_ENSEMBLE = "hybrid_ensemble"     # Ensemble of methods


@dataclass
class Tactician5mConfig:
    """Configuration for 5m Tactician entry optimization."""

    # Timeframes
    analyst_timeframe: str = "15m"  # Analyst operates on 15m
    tactician_timeframe: str = "5m" # Tactician operates on 5m

    # Analyst signal filtering
    analyst_confidence_threshold: float = 0.004  # 0.4% threshold for green lights
    min_green_period_duration: int = 3  # Minimum 3 candles in green period

    # Entry optimization parameters
    max_adverse_movement_pct: float = 0.5  # Max 0.5% adverse movement allowed
    min_favorable_movement_pct: float = 0.2  # Min 0.2% favorable movement expected
    max_entry_window_minutes: int = 60     # Max time to find entry within green period

    # Advanced optimization settings
    optimization_method: EntryOptimizationMethod = EntryOptimizationMethod.HYBRID_ENSEMBLE
    enable_multi_timeframe_analysis: bool = True
    enable_pattern_recognition: bool = True
    enable_ml_scoring: bool = False  # Disabled by default for interpretability

    # Risk management
    max_position_size_pct: float = 1.0  # Max position size as % of portfolio
    stop_loss_atr_multiplier: float = 2.0  # Stop loss distance in ATR units

    # Performance tracking
    enable_performance_tracking: bool = True
    save_entry_analysis: bool = True

    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EntryOptimizationResult:
    """Result of entry optimization process."""
    success: bool = False
    optimal_entries: List[Dict[str, Any]] = field(default_factory=list)
    entry_scores: List[float] = field(default_factory=list)
    green_periods_analyzed: int = 0
    total_entries_found: int = 0
    execution_time: float = 0.0
    error_message: Optional[str] = None

    # Performance metrics
    avg_entry_quality: float = 0.0
    best_entry_score: float = 0.0
    worst_entry_score: float = 0.0

    # Analysis metadata
    method_used: Optional[EntryOptimizationMethod] = None
    features_considered: List[str] = field(default_factory=list)


class Tactician5mEntryOptimizer:
    """
    Tactician 5m Entry Optimizer.

    Specialized for finding optimal entry points on 5m timeframe
    within Analyst 15m green light periods.

    Key Features:
    - Multi-timeframe analysis (5m entries within 15m analyst periods)
    - Advanced entry scoring with minimal adverse movement
    - Pattern recognition for entry timing
    - Ensemble optimization methods
    """

    def __init__(self, config: Optional[Tactician5mConfig] = None):
        """Initialize the Tactician 5m Entry Optimizer."""
        try:
            self.config = config or Tactician5mConfig()
            self.logger = system_logger.getChild('Tactician5mEntryOptimizer')

            tprint_success("✅ Tactician5mEntryOptimizer initialized")
            tprint_info(f"🎯 Analyst timeframe: {self.config.analyst_timeframe}")
            tprint_info(f"🎯 Tactician timeframe: {self.config.tactician_timeframe}")
            tprint_info(f"🎯 Optimization method: {self.config.optimization_method.value}")

        except Exception as e:
            tprint_error(f"❌ Failed to initialize Tactician5mEntryOptimizer: {e}")
            raise

    def _align_timeframes(self, data_5m: pd.DataFrame, data_15m: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Align 5m and 15m data to common time periods."""
        tprint_info("⏰ Aligning 5m and 15m timeframes...")

        # Ensure both datasets have datetime index
        if not isinstance(data_5m.index, pd.DatetimeIndex):
            data_5m.index = pd.to_datetime(data_5m.index)
        if not isinstance(data_15m.index, pd.DatetimeIndex):
            data_15m.index = pd.to_datetime(data_15m.index)

        # Find overlapping time period
        start_time = max(data_5m.index.min(), data_15m.index.min())
        end_time = min(data_5m.index.max(), data_15m.index.max())

        if start_time >= end_time:
            raise ValueError("No overlapping time period between 5m and 15m data")

        # Filter both datasets to overlapping period
        data_5m = data_5m[(data_5m.index >= start_time) & (data_5m.index <= end_time)]
        data_15m = data_15m[(data_15m.index >= start_time) & (data_15m.index <= end_time)]

        tprint_info(f"⏰ Aligned period: {start_time} to {end_time}")
        tprint_info(f"📊 5m data points: {len(data_5m)}")
        tprint_info(f"📊 15m data points: {len(data_15m)}")

        return data_5m, data_15m

    def _identify_analyst_green_periods(self, analyst_signals_15m: pd.Series) -> List[Dict[str, Any]]:
        """Identify contiguous periods of Analyst green lights (confidence > threshold)."""
        tprint_info("🔍 Identifying Analyst green light periods...")

        green_periods = []
        in_green_period = False
        start_idx = None

        for idx, (timestamp, signal) in enumerate(analyst_signals_15m.items()):
            if signal > self.config.analyst_confidence_threshold and not in_green_period:
                # Start of green period
                in_green_period = True
                start_idx = idx
            elif signal <= self.config.analyst_confidence_threshold and in_green_period:
                # End of green period
                in_green_period = False
                end_idx = idx

                # Check if period meets minimum duration
                if end_idx - start_idx >= self.config.min_green_period_duration:
                    period_start = analyst_signals_15m.index[start_idx]
                    period_end = analyst_signals_15m.index[end_idx - 1]

                    green_periods.append({
                        'start_time': period_start,
                        'end_time': period_end,
                        'duration': (period_end - period_start).total_seconds() / 60,  # minutes
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'signal_strength': analyst_signals_15m.iloc[start_idx:end_idx].mean()
                    })

        # Handle case where green period extends to end of data
        if in_green_period and start_idx is not None:
            end_idx = len(analyst_signals_15m)
            if end_idx - start_idx >= self.config.min_green_period_duration:
                period_start = analyst_signals_15m.index[start_idx]
                period_end = analyst_signals_15m.index[end_idx - 1]

                green_periods.append({
                    'start_time': period_start,
                    'end_time': period_end,
                    'duration': (period_end - period_start).total_seconds() / 60,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'signal_strength': analyst_signals_15m.iloc[start_idx:end_idx].mean()
                })

        tprint_info(f"🔍 Found {len(green_periods)} Analyst green light periods")
        for i, period in enumerate(green_periods[:3]):  # Show first 3 periods
            tprint_info(f"  Period {i+1}: {period['start_time']} to {period['end_time']} ({period['duration']:.1f} min)")

        return green_periods

    def _find_optimal_5m_entries_in_green_period(
        self,
        green_period: Dict[str, Any],
        data_5m: pd.DataFrame,
        data_15m: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Find optimal 5m entry points within a 15m Analyst green period."""
        tprint_info(f"🎯 Finding optimal 5m entries in green period: {green_period['start_time']} to {green_period['end_time']}")

        # Filter 5m data to green period
        period_5m_data = data_5m[
            (data_5m.index >= green_period['start_time']) &
            (data_5m.index <= green_period['end_time'])
        ]

        if len(period_5m_data) < 2:
            tprint_warning("⚠️ Insufficient 5m data in green period")
            return []

        optimal_entries = []

        # Use selected optimization method
        if self.config.optimization_method == EntryOptimizationMethod.PEAK_FINDER:
            entries = self._peak_finder_optimization(period_5m_data, data_15m)
        elif self.config.optimization_method == EntryOptimizationMethod.MULTI_TIMEFRAME:
            entries = self._multi_timeframe_optimization(period_5m_data, data_15m)
        elif self.config.optimization_method == EntryOptimizationMethod.PATTERN_BASED:
            entries = self._pattern_based_optimization(period_5m_data, data_15m)
        elif self.config.optimization_method == EntryOptimizationMethod.ML_ENHANCED:
            entries = self._ml_enhanced_optimization(period_5m_data, data_15m)
        else:  # HYBRID_ENSEMBLE
            entries = self._hybrid_ensemble_optimization(period_5m_data, data_15m)

        optimal_entries.extend(entries)

        tprint_info(f"🎯 Found {len(optimal_entries)} optimal entries in this green period")
        return optimal_entries

    def _peak_finder_optimization(self, data_5m: pd.DataFrame, data_15m: pd.DataFrame) -> List[Dict[str, Any]]:
        """Original peak-finding optimization method."""
        entries = []

        for i in range(len(data_5m) - 1):
            entry_time = data_5m.index[i]
            entry_price = data_5m.iloc[i]['close']

            # Look ahead in 5m data for future price movement
            future_5m_data = data_5m.iloc[i+1:]

            if len(future_5m_data) == 0:
                continue

            # Calculate entry quality metrics
            adverse_move = (entry_price - future_5m_data['low'].min()) / entry_price * 100
            favorable_move = (future_5m_data['high'].max() - entry_price) / entry_price * 100

            # Check if entry meets criteria
            if (adverse_move <= self.config.max_adverse_movement_pct and
                favorable_move >= self.config.min_favorable_movement_pct):

                entry_score = self._calculate_entry_score(
                    entry_price, future_5m_data, entry_time, data_15m
                )

                if entry_score > 0.5:  # Minimum quality threshold
                    entries.append({
                        'timestamp': entry_time,
                        'entry_price': entry_price,
                        'score': entry_score,
                        'adverse_move_pct': adverse_move,
                        'favorable_move_pct': favorable_move,
                        'method': 'peak_finder'
                    })

        return entries

    def _multi_timeframe_optimization(self, data_5m: pd.DataFrame, data_15m: pd.DataFrame) -> List[Dict[str, Any]]:
        """Multi-timeframe optimization using 15m context."""
        entries = []

        for i in range(len(data_5m) - 1):
            entry_time = data_5m.index[i]

            # Get 15m context around this 5m entry
            context_15m = self._get_15m_context(entry_time, data_15m)

            if context_15m is None:
                continue

            # Enhanced scoring with 15m context
            entry_price = data_5m.iloc[i]['close']
            future_5m_data = data_5m.iloc[i+1:]

            if len(future_5m_data) == 0:
                continue

            # Multi-timeframe entry score
            base_score = self._calculate_entry_score(entry_price, future_5m_data, entry_time, data_15m)
            context_bonus = self._calculate_context_bonus(context_15m, entry_time)
            timeframe_alignment_score = self._calculate_timeframe_alignment(entry_time, data_15m)

            combined_score = (base_score * 0.6 + context_bonus * 0.3 + timeframe_alignment_score * 0.1)

            if combined_score > 0.6:
                entries.append({
                    'timestamp': entry_time,
                    'entry_price': entry_price,
                    'score': combined_score,
                    'method': 'multi_timeframe',
                    'context_15m': context_15m
                })

        return entries

    def _pattern_based_optimization(self, data_5m: pd.DataFrame, data_15m: pd.DataFrame) -> List[Dict[str, Any]]:
        """Pattern-based entry optimization."""
        # Implementation would include:
        # - Breakout patterns
        # - Pullback entries
        # - Support/resistance bounces
        # - Volume-price divergences
        entries = []

        # Simplified pattern detection for now
        # In full implementation, would use sophisticated pattern recognition
        for i in range(1, len(data_5m) - 1):
            # Look for price consolidation followed by breakout
            current_price = data_5m.iloc[i]['close']
            prev_price = data_5m.iloc[i-1]['close']
            next_price = data_5m.iloc[i+1]['close']

            # Simple breakout pattern detection
            if (current_price > prev_price and
                next_price > current_price and
                self._is_within_analyst_green_period(data_5m.index[i], data_15m)):

                pattern_score = 0.7  # Would be calculated based on pattern strength

                entries.append({
                    'timestamp': data_5m.index[i],
                    'entry_price': current_price,
                    'score': pattern_score,
                    'method': 'pattern_based',
                    'pattern': 'breakout_consolidation'
                })

        return entries

    def _ml_enhanced_optimization(self, data_5m: pd.DataFrame, data_15m: pd.DataFrame) -> List[Dict[str, Any]]:
        """Machine learning enhanced entry optimization."""
        # Placeholder for ML-based optimization
        # Would use trained models to predict entry quality
        return []

    def _hybrid_ensemble_optimization(self, data_5m: pd.DataFrame, data_15m: pd.DataFrame) -> List[Dict[str, Any]]:
        """Hybrid ensemble combining multiple optimization methods."""
        all_entries = []

        # Get entries from different methods
        peak_entries = self._peak_finder_optimization(data_5m, data_15m)
        mtf_entries = self._multi_timeframe_optimization(data_5m, data_15m)
        pattern_entries = self._pattern_based_optimization(data_5m, data_15m)

        # Combine and deduplicate entries
        all_entries.extend(peak_entries)
        all_entries.extend(mtf_entries)
        all_entries.extend(pattern_entries)

        # Remove duplicate timestamps and keep highest scoring entry
        unique_entries = {}
        for entry in all_entries:
            timestamp = entry['timestamp']
            if timestamp not in unique_entries or entry['score'] > unique_entries[timestamp]['score']:
                unique_entries[timestamp] = entry

        # Ensemble scoring
        final_entries = []
        for entry in unique_entries.values():
            # Boost score if multiple methods agree
            method_count = sum(1 for e in all_entries if e['timestamp'] == entry['timestamp'])
            ensemble_score = entry['score'] * (1 + 0.1 * method_count)

            entry['score'] = min(ensemble_score, 1.0)  # Cap at 1.0
            entry['method'] = f"hybrid_ensemble_{method_count}_methods"
            final_entries.append(entry)

        # Sort by score and return top entries
        final_entries.sort(key=lambda x: x['score'], reverse=True)

        return final_entries

    def _calculate_entry_score(self, entry_price: float, future_data: pd.DataFrame,
                              entry_time: pd.Timestamp, data_15m: pd.DataFrame) -> float:
        """Calculate entry quality score based on price movement expectations."""
        if len(future_data) == 0:
            return 0.0

        min_future_low = future_data['low'].min()
        max_future_high = future_data['high'].max()

        adverse_move = max(entry_price - min_future_low, 0.0) / max(entry_price, 1e-8) * 100
        favorable_move = max(max_future_high - entry_price, 0.0) / max(entry_price, 1e-8) * 100

        if adverse_move > self.config.max_adverse_movement_pct:
            return 0.0

        if favorable_move < self.config.min_favorable_movement_pct:
            return 0.0

        risk_reward_ratio = favorable_move / (adverse_move + 1e-8)
        timing_score = 1.0 / (1.0 + len(future_data) / self.config.max_entry_window_minutes)
        volatility = future_data['close'].pct_change().std() or 0.0
        volatility_score = 1.0 / (1.0 + (volatility * 100) / 10.0)

        quality_score = (
            risk_reward_ratio * 0.4 +
            timing_score * 0.3 +
            volatility_score * 0.3
        )

        return float(min(max(quality_score, 0.0), 1.0))

    def _get_15m_context(self, timestamp_5m: pd.Timestamp, data_15m: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Get 15m context around a 5m timestamp."""
        try:
            # Find the 15m candle that contains this 5m timestamp
            context_15m = data_15m[data_15m.index <= timestamp_5m].tail(1)
            if len(context_15m) == 0:
                return None

            return {
                'candle_time': context_15m.index[0],
                'open': context_15m.iloc[0]['open'],
                'high': context_15m.iloc[0]['high'],
                'low': context_15m.iloc[0]['low'],
                'close': context_15m.iloc[0]['close'],
                'volume': context_15m.iloc[0]['volume']
            }
        except Exception:
            return None

    def _calculate_context_bonus(self, context_15m: Dict[str, Any], entry_time: pd.Timestamp) -> float:
        """Calculate bonus score based on 15m context."""
        # Higher bonus for entries in strong 15m candles
        candle_range = (context_15m['high'] - context_15m['low']) / context_15m['open'] * 100
        volume_ratio = context_15m['volume'] / context_15m['volume'].mean() if hasattr(context_15m['volume'], 'mean') else 1.0

        return min((candle_range / 5.0 + volume_ratio / 2.0) / 2.0, 0.3)

    def _calculate_timeframe_alignment(self, entry_time: pd.Timestamp, data_15m: pd.DataFrame) -> float:
        """Calculate how well 5m entry aligns with 15m trend."""
        # Check if entry aligns with 15m trend direction
        try:
            context_15m = self._get_15m_context(entry_time, data_15m)
            if context_15m is None:
                return 0.5

            # Simple trend alignment score
            return 0.7  # Placeholder - would analyze trend alignment
        except Exception:
            return 0.5

    def _is_within_analyst_green_period(self, timestamp_5m: pd.Timestamp, data_15m: pd.DataFrame) -> bool:
        """Check if 5m timestamp falls within a 15m Analyst green period."""
        # This would need to be implemented based on how Analyst signals are structured
        # For now, return True as placeholder
        return True

    def optimize_entries(
        self,
        data_5m: pd.DataFrame,
        analyst_signals_15m: pd.Series,
        data_15m: Optional[pd.DataFrame] = None
    ) -> EntryOptimizationResult:
        """Main entry optimization function."""
        start_time = tprint_timer()
        tprint_info("🚀 Starting Tactician 5m entry optimization...")
        tprint_info(f"📊 5m data points: {len(data_5m)}")
        tprint_info(f"📊 15m analyst signals: {len(analyst_signals_15m)}")

        result = EntryOptimizationResult()
        result.method_used = self.config.optimization_method

        try:
            # Align timeframes if both datasets provided
            if data_15m is not None:
                data_5m, data_15m = self._align_timeframes(data_5m, data_15m)

            # Identify Analyst green periods
            green_periods = self._identify_analyst_green_periods(analyst_signals_15m)
            result.green_periods_analyzed = len(green_periods)

            if len(green_periods) == 0:
                result.error_message = "No Analyst green periods found"
                tprint_warning("⚠️ No Analyst green periods found")
                return result

            # Find optimal entries in each green period
            all_optimal_entries = []

            for green_period in green_periods:
                period_entries = self._find_optimal_5m_entries_in_green_period(
                    green_period, data_5m, data_15m or pd.DataFrame()
                )
                all_optimal_entries.extend(period_entries)

            # Process and filter entries
            if all_optimal_entries:
                # Sort by score and take top entries
                all_optimal_entries.sort(key=lambda x: x['score'], reverse=True)

                # Apply quality filtering
                filtered_entries = [e for e in all_optimal_entries if e['score'] > 0.6]

                result.optimal_entries = filtered_entries[:50]  # Top 50 entries
                result.entry_scores = [e['score'] for e in result.optimal_entries]
                result.total_entries_found = len(all_optimal_entries)

                if result.entry_scores:
                    result.avg_entry_quality = np.mean(result.entry_scores)
                    result.best_entry_score = max(result.entry_scores)
                    result.worst_entry_score = min(result.entry_scores)

            result.success = True
            result.execution_time = tprint_timer(start_time)

            tprint_success(f"✅ Entry optimization completed in {result.execution_time:.2f}s")
            tprint_info(f"📊 Found {len(result.optimal_entries)} high-quality entries")
            tprint_info(f"📊 Average entry quality: {result.avg_entry_quality:.3f}")

            return result

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.execution_time = tprint_timer(start_time)

            tprint_error(f"❌ Entry optimization failed: {e}")
            return result


# Convenience function for external usage
def optimize_tactician_entries(
    data_5m: pd.DataFrame,
    analyst_signals_15m: pd.Series,
    data_15m: Optional[pd.DataFrame] = None,
    config: Optional[Tactician5mConfig] = None
) -> EntryOptimizationResult:
    """
    Optimize entry points for Tactician 5m timeframe using Analyst 15m signals.

    Args:
        data_5m: 5m timeframe price data
        analyst_signals_15m: Analyst confidence signals on 15m timeframe
        data_15m: Optional 15m price data for context
        config: Optional configuration

    Returns:
        EntryOptimizationResult with optimal entry points and scores
    """
    optimizer = Tactician5mEntryOptimizer(config)
    return optimizer.optimize_entries(data_5m, analyst_signals_15m, data_15m)