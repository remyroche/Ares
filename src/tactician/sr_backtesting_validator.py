# src/tactician/sr_backtesting_validator.py

"""
S/R Backtesting Validator

This module implements comprehensive backtesting to validate whether detected
S/R levels are actually effective. It simulates price interactions with S/R
levels and measures their success in predicting support/resistance behavior.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
from scipy import stats
from sklearn.cluster import DBSCAN
warnings.filterwarnings('ignore')

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors, handle_specific_errors


@dataclass
class SRLevelTest:
    """Individual S/R level test result."""
    
    level_price: float
    level_type: str  # "support" or "resistance"
    test_start_time: datetime
    test_end_time: datetime
    touches: int = 0
    bounces: int = 0
    breakouts: int = 0
    false_breakouts: int = 0
    bounce_rate: float = 0.0
    breakout_rate: float = 0.0
    false_breakout_rate: float = 0.0
    avg_bounce_strength: float = 0.0
    avg_breakout_strength: float = 0.0
    level_strength: float = 0.0
    confidence_score: float = 0.0
    
    # Volume analysis metrics
    avg_volume_at_touches: float = 0.0
    volume_spike_ratio: float = 0.0  # Volume at touches vs average volume
    volume_confirmation_rate: float = 0.0  # % of touches with above-average volume
    volume_weighted_bounce_rate: float = 0.0  # Bounce rate weighted by volume
    institutional_volume_ratio: float = 0.0  # Large volume bars ratio
    volume_cluster_score: float = 0.0  # Volume clustering around level
    
    # Time-based analysis
    level_age_days: int = 0
    age_decay_factor: float = 1.0  # How much level effectiveness decays over time
    first_touch_date: datetime = None
    last_touch_date: datetime = None
    
    # Market context analysis
    trend_context_score: float = 0.0  # How well level works in current trend
    volatility_regime_score: float = 0.0  # Performance in current volatility
    market_structure_score: float = 0.0  # Bull/bear market performance
    
    # Multi-timeframe validation
    multi_timeframe_score: float = 0.0  # Confluence across timeframes
    higher_timeframe_alignment: float = 0.0  # Alignment with higher timeframes
    
    # Price action confirmation
    candlestick_confirmation_rate: float = 0.0  # % of touches with confirming patterns
    rejection_pattern_score: float = 0.0  # Strength of rejection signals
    consolidation_score: float = 0.0  # Price consolidation around level
    
    # Statistical validation
    statistical_significance: float = 0.0  # P-value for statistical significance
    monte_carlo_score: float = 0.0  # Robustness score from Monte Carlo simulation


@dataclass
class BacktestResult:
    """Result of S/R backtesting."""
    
    # Overall metrics
    total_levels_tested: int = 0
    successful_levels: int = 0
    overall_bounce_rate: float = 0.0
    overall_breakout_rate: float = 0.0
    overall_false_breakout_rate: float = 0.0
    
    # Support vs Resistance metrics
    support_bounce_rate: float = 0.0
    resistance_bounce_rate: float = 0.0
    support_breakout_rate: float = 0.0
    resistance_breakout_rate: float = 0.0
    
    # Level quality metrics
    avg_level_strength: float = 0.0
    avg_confidence_score: float = 0.0
    level_detection_accuracy: float = 0.0
    
    # Volume analysis metrics
    avg_volume_spike_ratio: float = 0.0
    avg_volume_confirmation_rate: float = 0.0
    avg_institutional_volume_ratio: float = 0.0
    avg_volume_cluster_score: float = 0.0
    
    # Time-based analysis metrics
    avg_level_age_days: float = 0.0
    avg_age_decay_factor: float = 1.0
    level_persistence_score: float = 0.0  # How long levels remain valid
    
    # Market context metrics
    avg_trend_context_score: float = 0.0
    avg_volatility_regime_score: float = 0.0
    avg_market_structure_score: float = 0.0
    
    # Multi-timeframe metrics
    avg_multi_timeframe_score: float = 0.0
    avg_higher_timeframe_alignment: float = 0.0
    
    # Price action metrics
    avg_candlestick_confirmation_rate: float = 0.0
    avg_rejection_pattern_score: float = 0.0
    avg_consolidation_score: float = 0.0
    
    # Statistical validation metrics
    avg_statistical_significance: float = 0.0
    avg_monte_carlo_score: float = 0.0
    out_of_sample_score: float = 0.0  # Performance on unseen data
    
    # S/R validation score
    sr_validation_score: float = 0.0
    
    # Detailed results
    level_tests: List[SRLevelTest] = None
    
    def __post_init__(self):
        if self.level_tests is None:
            self.level_tests = []


class SRBacktestingValidator:
    """
    Comprehensive S/R backtesting validator.
    
    This class implements proper backtesting to validate S/R levels by:
    1. Detecting when price touches S/R levels
    2. Measuring bounce vs breakout behavior
    3. Calculating success metrics and confidence scores
    4. Simulating trading strategies based on S/R levels
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the S/R backtesting validator."""
        self.config = config
        self.logger = system_logger.getChild("SRBacktestingValidator")
        
            # Backtesting configuration
    self.backtest_config = config.get("sr_backtesting", {})
    self.touch_threshold = self.backtest_config.get("touch_threshold", 0.001)  # 0.1% touch threshold
    self.bounce_threshold = self.backtest_config.get("bounce_threshold", 0.005)  # 0.5% bounce threshold
    self.breakout_threshold = self.backtest_config.get("breakout_threshold", 0.01)  # 1% breakout threshold
    self.false_breakout_threshold = self.backtest_config.get("false_breakout_threshold", 0.02)  # 2% false breakout
    self.confirmation_periods = self.backtest_config.get("confirmation_periods", 3)
    self.min_touches = self.backtest_config.get("min_touches", 2)
    
    # Volume analysis configuration
    self.volume_spike_threshold = self.backtest_config.get("volume_spike_threshold", 1.5)  # 1.5x average volume
    self.institutional_volume_threshold = self.backtest_config.get("institutional_volume_threshold", 2.0)  # 2x average volume
    self.volume_confirmation_threshold = self.backtest_config.get("volume_confirmation_threshold", 1.2)  # 1.2x average volume
    self.volume_lookback_periods = self.backtest_config.get("volume_lookback_periods", 20)  # 20 periods for volume baseline
    self.volume_cluster_radius = self.backtest_config.get("volume_cluster_radius", 0.005)  # 0.5% price range for clustering
        
                    # S/R validation configuration
        self.min_bounce_rate = self.backtest_config.get("min_bounce_rate", 0.6)  # 60% minimum bounce rate
        self.max_false_breakout_rate = self.backtest_config.get("max_false_breakout_rate", 0.3)  # 30% max false breakouts
        self.min_volume_confirmation = self.backtest_config.get("min_volume_confirmation", 0.5)  # 50% volume confirmation
        
        # Time-based analysis configuration
        self.age_decay_factor = self.backtest_config.get("age_decay_factor", 0.95)  # 5% decay per period
        self.max_level_age_days = self.backtest_config.get("max_level_age_days", 365)  # 1 year max age
        
        # Market context configuration
        self.trend_period = self.backtest_config.get("trend_period", 50)  # Periods for trend calculation
        self.volatility_period = self.backtest_config.get("volatility_period", 20)  # Periods for volatility calculation
        
        # Multi-timeframe configuration
        self.enable_multi_timeframe = self.backtest_config.get("enable_multi_timeframe", True)
        self.timeframe_weights = self.backtest_config.get("timeframe_weights", {
            "1m": 0.05, "5m": 0.1, "15m": 0.15, "1h": 0.2, "4h": 0.25, "1d": 0.25
        })
        
        # Price action configuration
        self.enable_price_action_analysis = self.backtest_config.get("enable_price_action_analysis", True)
        self.candlestick_patterns = self.backtest_config.get("candlestick_patterns", [
            "doji", "hammer", "shooting_star", "engulfing", "pin_bar"
        ])
        
        # Statistical validation configuration
        self.enable_statistical_validation = self.backtest_config.get("enable_statistical_validation", True)
        self.monte_carlo_iterations = self.backtest_config.get("monte_carlo_iterations", 1000)
        self.out_of_sample_ratio = self.backtest_config.get("out_of_sample_ratio", 0.2)
        
    @handle_specific_errors(
        error_handlers={
            ValueError: (None, "Invalid data for S/R backtesting"),
            AttributeError: (None, "Backtesting validator not properly initialized"),
        },
        default_return=None,
        context="S/R backtesting validation"
    )
    async def validate_sr_levels(
        self,
        market_data: pd.DataFrame,
        sr_levels: List[Dict[str, Any]],
        current_price: float,
        multi_timeframe_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Optional[BacktestResult]:
        """
        Validate S/R levels through comprehensive backtesting.
        
        Args:
            market_data: Historical market data for backtesting
            sr_levels: List of detected S/R levels
            current_price: Current market price
            
        Returns:
            BacktestResult: Comprehensive backtesting results
        """
        try:
            self.logger.info(f"🔍 Starting S/R level validation with {len(sr_levels)} levels")
            
            # Initialize results
            result = BacktestResult()
            result.total_levels_tested = len(sr_levels)
            
            # Calculate market context metrics
            market_context = await self._calculate_market_context(market_data)
            
            # Test each S/R level
            for level in sr_levels:
                level_test = await self._test_single_level(
                    market_data, level, current_price, market_context, multi_timeframe_data
                )
                if level_test:
                    result.level_tests.append(level_test)
                    
                    # Update overall metrics
                    if level_test.bounce_rate > self.min_bounce_rate:
                        result.successful_levels += 1
            
            # Calculate overall metrics
            await self._calculate_overall_metrics(result)
            
            # Calculate S/R validation score
            await self._calculate_sr_validation_score(result)
            
            # Perform statistical validation if enabled
            if self.enable_statistical_validation:
                await self._perform_statistical_validation(result, market_data)
            
            self.logger.info(f"✅ Comprehensive S/R validation completed. Validation score: {result.sr_validation_score:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"S/R validation failed: {e}")
            return None
    
    async def _test_single_level(
        self,
        market_data: pd.DataFrame,
        level: Dict[str, Any],
        current_price: float,
        market_context: Dict[str, Any],
        multi_timeframe_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Optional[SRLevelTest]:
        """Test a single S/R level."""
        try:
            level_price = level.get("price", 0.0)
            level_type = level.get("type", "support")
            level_strength = level.get("enhanced_strength", level.get("strength", 0.5))
            
            # Create test result
            test = SRLevelTest(
                level_price=level_price,
                level_type=level_type,
                test_start_time=market_data.index[0],
                test_end_time=market_data.index[-1],
                level_strength=level_strength
            )
            
            # Analyze price interactions with this level
            touch_data = await self._analyze_level_interactions(
                market_data, level_price, level_type
            )
            
            touches, bounces, breakouts, false_breakouts = touch_data[:4]
            touch_volumes, touch_indices = touch_data[4:6]
            
            # Update test results
            test.touches = touches
            test.bounces = bounces
            test.breakouts = breakouts
            test.false_breakouts = false_breakouts
            
            # Calculate rates
            if touches > 0:
                test.bounce_rate = bounces / touches
                test.breakout_rate = breakouts / touches
                test.false_breakout_rate = false_breakouts / touches
            
            # Analyze volume patterns
            await self._analyze_volume_patterns(test, market_data, touch_volumes, touch_indices, level_price)
            
            # Analyze time-based factors
            await self._analyze_time_based_factors(test, market_data, touch_indices)
            
            # Analyze market context
            await self._analyze_market_context(test, market_context, touch_indices)
            
            # Analyze price action patterns
            await self._analyze_price_action(test, market_data, touch_indices)
            
            # Calculate confidence score with comprehensive analysis
            test.confidence_score = self._calculate_level_confidence(test)
            
            return test
            
        except Exception as e:
            self.logger.error(f"Failed to test level {level.get('price', 0)}: {e}")
            return None
    
    async def _analyze_level_interactions(
        self,
        market_data: pd.DataFrame,
        level_price: float,
        level_type: str
    ) -> Tuple[int, int, int, int, List[float], List[int]]:
        """
        Analyze how price interacts with a specific S/R level.
        
        Returns:
            Tuple of (touches, bounces, breakouts, false_breakouts, touch_volumes, touch_indices)
        """
        try:
            touches = 0
            bounces = 0
            breakouts = 0
            false_breakouts = 0
            touch_volumes = []
            touch_indices = []
            
            # Define touch zone around the level
            touch_zone_upper = level_price * (1 + self.touch_threshold)
            touch_zone_lower = level_price * (1 - self.touch_threshold)
            
            i = 0
            while i < len(market_data) - self.confirmation_periods:
                # Check if price touches the level
                high = market_data['high'].iloc[i]
                low = market_data['low'].iloc[i]
                
                if low <= touch_zone_upper and high >= touch_zone_lower:
                    touches += 1
                    touch_volumes.append(market_data['volume'].iloc[i])
                    touch_indices.append(i)
                    
                    # Analyze what happens after the touch
                    touch_result = await self._analyze_touch_outcome(
                        market_data, i, level_price, level_type
                    )
                    
                    if touch_result == "bounce":
                        bounces += 1
                    elif touch_result == "breakout":
                        breakouts += 1
                    elif touch_result == "false_breakout":
                        false_breakouts += 1
                
                i += 1
            
            return touches, bounces, breakouts, false_breakouts, touch_volumes, touch_indices
            
        except Exception as e:
            self.logger.error(f"Failed to analyze level interactions: {e}")
            return 0, 0, 0, 0, [], []
    
    async def _analyze_touch_outcome(
        self,
        market_data: pd.DataFrame,
        touch_index: int,
        level_price: float,
        level_type: str
    ) -> str:
        """
        Analyze what happens after price touches an S/R level.
        
        Returns:
            "bounce", "breakout", "false_breakout", or "inconclusive"
        """
        try:
            # Look ahead for confirmation
            end_index = min(touch_index + self.confirmation_periods, len(market_data))
            future_data = market_data.iloc[touch_index:end_index]
            
            if level_type == "support":
                # For support levels, check if price bounces up or breaks down
                min_price = future_data['low'].min()
                max_price = future_data['high'].max()
                
                # Check for bounce (price moves up significantly)
                if max_price > level_price * (1 + self.bounce_threshold):
                    return "bounce"
                
                # Check for breakout (price moves down significantly)
                elif min_price < level_price * (1 - self.breakout_threshold):
                    # Check if it's a false breakout (price comes back)
                    if max_price > level_price * (1 + self.false_breakout_threshold):
                        return "false_breakout"
                    else:
                        return "breakout"
            
            elif level_type == "resistance":
                # For resistance levels, check if price bounces down or breaks up
                min_price = future_data['low'].min()
                max_price = future_data['high'].max()
                
                # Check for bounce (price moves down significantly)
                if min_price < level_price * (1 - self.bounce_threshold):
                    return "bounce"
                
                # Check for breakout (price moves up significantly)
                elif max_price > level_price * (1 + self.breakout_threshold):
                    # Check if it's a false breakout (price comes back)
                    if min_price < level_price * (1 - self.false_breakout_threshold):
                        return "false_breakout"
                    else:
                        return "breakout"
            
            return "inconclusive"
            
        except Exception as e:
            self.logger.error(f"Failed to analyze touch outcome: {e}")
            return "inconclusive"
    
    async def _analyze_volume_patterns(
        self,
        test: SRLevelTest,
        market_data: pd.DataFrame,
        touch_volumes: List[float],
        touch_indices: List[int],
        level_price: float
    ) -> None:
        """
        Analyze volume patterns around S/R levels.
        
        This method analyzes:
        1. Volume spikes at S/R touches
        2. Volume confirmation of bounces/breakouts
        3. Institutional volume presence
        4. Volume clustering around the level
        """
        try:
            if not touch_volumes or len(touch_volumes) == 0:
                return
            
            # Calculate volume baseline
            avg_volume = market_data['volume'].rolling(window=self.volume_lookback_periods).mean()
            
            # 1. Average volume at touches
            test.avg_volume_at_touches = np.mean(touch_volumes)
            
            # 2. Volume spike ratio (volume at touches vs average volume)
            volume_spikes = []
            volume_confirmations = 0
            institutional_volumes = 0
            
            for i, touch_idx in enumerate(touch_indices):
                if touch_idx < len(avg_volume):
                    baseline_volume = avg_volume.iloc[touch_idx]
                    if baseline_volume > 0:
                        volume_ratio = touch_volumes[i] / baseline_volume
                        volume_spikes.append(volume_ratio)
                        
                        # Check for volume confirmation
                        if volume_ratio >= self.volume_confirmation_threshold:
                            volume_confirmations += 1
                        
                        # Check for institutional volume
                        if volume_ratio >= self.institutional_volume_threshold:
                            institutional_volumes += 1
            
            if volume_spikes:
                test.volume_spike_ratio = np.mean(volume_spikes)
                test.volume_confirmation_rate = volume_confirmations / len(touch_indices)
                test.institutional_volume_ratio = institutional_volumes / len(touch_indices)
            
            # 3. Volume-weighted bounce rate
            bounce_volumes = []
            total_volume = 0
            
            for i, touch_idx in enumerate(touch_indices):
                if touch_idx < len(market_data) - self.confirmation_periods:
                    # Check if this touch resulted in a bounce
                    touch_result = await self._analyze_touch_outcome(
                        market_data, touch_idx, level_price, test.level_type
                    )
                    
                    if touch_result == "bounce":
                        bounce_volumes.append(touch_volumes[i])
                    
                    total_volume += touch_volumes[i]
            
            if total_volume > 0 and bounce_volumes:
                test.volume_weighted_bounce_rate = sum(bounce_volumes) / total_volume
            
            # 4. Volume clustering analysis
            test.volume_cluster_score = await self._calculate_volume_cluster_score(
                market_data, level_price, touch_indices
            )
            
        except Exception as e:
            self.logger.error(f"Failed to analyze volume patterns: {e}")
    
    async def _calculate_volume_cluster_score(
        self,
        market_data: pd.DataFrame,
        level_price: float,
        touch_indices: List[int]
    ) -> float:
        """
        Calculate volume clustering score around S/R level.
        
        This measures how much volume is concentrated around the S/R level
        compared to other price levels.
        """
        try:
            if not touch_indices:
                return 0.0
            
            # Define the level zone
            level_zone_upper = level_price * (1 + self.volume_cluster_radius)
            level_zone_lower = level_price * (1 - self.volume_cluster_radius)
            
            # Calculate total volume in the level zone
            level_zone_volume = 0
            total_volume = market_data['volume'].sum()
            
            for i in range(len(market_data)):
                price = market_data['close'].iloc[i]
                if level_zone_lower <= price <= level_zone_upper:
                    level_zone_volume += market_data['volume'].iloc[i]
            
            # Calculate volume concentration ratio
            if total_volume > 0:
                volume_concentration = level_zone_volume / total_volume
                
                # Normalize by the size of the zone relative to the price range
                price_range = market_data['high'].max() - market_data['low'].min()
                zone_size = level_zone_upper - level_zone_lower
                expected_concentration = zone_size / price_range if price_range > 0 else 0
                
                if expected_concentration > 0:
                    cluster_score = volume_concentration / expected_concentration
                    return min(cluster_score, 5.0)  # Cap at 5x concentration
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Failed to calculate volume cluster score: {e}")
            return 0.0
    
    def _calculate_level_confidence(self, test: SRLevelTest) -> float:
        """Calculate confidence score for a level based on its performance and volume analysis."""
        try:
            confidence = 0.0
            
            # Base confidence from bounce rate (40% weight)
            if test.bounce_rate > 0.8:
                confidence += 0.4
            elif test.bounce_rate > 0.6:
                confidence += 0.3
            elif test.bounce_rate > 0.4:
                confidence += 0.2
            elif test.bounce_rate > 0.2:
                confidence += 0.1
            
            # Volume analysis (30% weight)
            volume_confidence = 0.0
            
            # Volume spike ratio
            if test.volume_spike_ratio > 2.0:
                volume_confidence += 0.15
            elif test.volume_spike_ratio > 1.5:
                volume_confidence += 0.1
            elif test.volume_spike_ratio > 1.2:
                volume_confidence += 0.05
            
            # Volume confirmation rate
            if test.volume_confirmation_rate > 0.8:
                volume_confidence += 0.1
            elif test.volume_confirmation_rate > 0.6:
                volume_confidence += 0.05
            
            # Institutional volume presence
            if test.institutional_volume_ratio > 0.3:
                volume_confidence += 0.05
            
            confidence += volume_confidence
            
            # Penalty for false breakouts (15% weight)
            if test.false_breakout_rate > 0.3:
                confidence -= 0.15
            elif test.false_breakout_rate > 0.2:
                confidence -= 0.1
            elif test.false_breakout_rate > 0.1:
                confidence -= 0.05
            
            # Bonus for number of touches (10% weight)
            if test.touches >= 5:
                confidence += 0.1
            elif test.touches >= 3:
                confidence += 0.05
            
            # Bonus for level strength (5% weight)
            confidence += test.level_strength * 0.05
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            self.logger.error(f"Failed to calculate level confidence: {e}")
            return 0.5
    
    async def _calculate_overall_metrics(self, result: BacktestResult) -> None:
        """Calculate overall metrics from individual level tests."""
        try:
            if not result.level_tests:
                return
            
            # Calculate overall rates
            total_touches = sum(test.touches for test in result.level_tests)
            total_bounces = sum(test.bounces for test in result.level_tests)
            total_breakouts = sum(test.breakouts for test in result.level_tests)
            total_false_breakouts = sum(test.false_breakouts for test in result.level_tests)
            
            if total_touches > 0:
                result.overall_bounce_rate = total_bounces / total_touches
                result.overall_breakout_rate = total_breakouts / total_touches
                result.overall_false_breakout_rate = total_false_breakouts / total_touches
            
            # Calculate support vs resistance rates
            support_tests = [test for test in result.level_tests if test.level_type == "support"]
            resistance_tests = [test for test in result.level_tests if test.level_type == "resistance"]
            
            if support_tests:
                support_touches = sum(test.touches for test in support_tests)
                support_bounces = sum(test.bounces for test in support_tests)
                support_breakouts = sum(test.breakouts for test in support_tests)
                
                if support_touches > 0:
                    result.support_bounce_rate = support_bounces / support_touches
                    result.support_breakout_rate = support_breakouts / support_touches
            
            if resistance_tests:
                resistance_touches = sum(test.touches for test in resistance_tests)
                resistance_bounces = sum(test.bounces for test in resistance_tests)
                resistance_breakouts = sum(test.breakouts for test in resistance_tests)
                
                if resistance_touches > 0:
                    result.resistance_bounce_rate = resistance_bounces / resistance_touches
                    result.resistance_breakout_rate = resistance_breakouts / resistance_touches
            
            # Calculate average metrics
            result.avg_level_strength = np.mean([test.level_strength for test in result.level_tests])
            result.avg_confidence_score = np.mean([test.confidence_score for test in result.level_tests])
            
            # Calculate volume-related metrics
            if result.level_tests:
                result.avg_volume_spike_ratio = np.mean([test.volume_spike_ratio for test in result.level_tests])
                result.avg_volume_confirmation_rate = np.mean([test.volume_confirmation_rate for test in result.level_tests])
                result.avg_institutional_volume_ratio = np.mean([test.institutional_volume_ratio for test in result.level_tests])
                result.avg_volume_cluster_score = np.mean([test.volume_cluster_score for test in result.level_tests])
            
            # Calculate time-based metrics
            if result.level_tests:
                result.avg_level_age_days = np.mean([test.level_age_days for test in result.level_tests])
                result.avg_age_decay_factor = np.mean([test.age_decay_factor for test in result.level_tests])
                result.level_persistence_score = np.mean([1 - test.age_decay_factor for test in result.level_tests])
            
            # Calculate market context metrics
            if result.level_tests:
                result.avg_trend_context_score = np.mean([test.trend_context_score for test in result.level_tests])
                result.avg_volatility_regime_score = np.mean([test.volatility_regime_score for test in result.level_tests])
                result.avg_market_structure_score = np.mean([test.market_structure_score for test in result.level_tests])
            
            # Calculate multi-timeframe metrics
            if result.level_tests:
                result.avg_multi_timeframe_score = np.mean([test.multi_timeframe_score for test in result.level_tests])
                result.avg_higher_timeframe_alignment = np.mean([test.higher_timeframe_alignment for test in result.level_tests])
            
            # Calculate price action metrics
            if result.level_tests:
                result.avg_candlestick_confirmation_rate = np.mean([test.candlestick_confirmation_rate for test in result.level_tests])
                result.avg_rejection_pattern_score = np.mean([test.rejection_pattern_score for test in result.level_tests])
                result.avg_consolidation_score = np.mean([test.consolidation_score for test in result.level_tests])
            
            # Calculate statistical validation metrics
            if result.level_tests:
                result.avg_statistical_significance = np.mean([test.statistical_significance for test in result.level_tests])
                result.avg_monte_carlo_score = np.mean([test.monte_carlo_score for test in result.level_tests])
            
            # Calculate level detection accuracy
            if result.total_levels_tested > 0:
                result.level_detection_accuracy = result.successful_levels / result.total_levels_tested
            
        except Exception as e:
            self.logger.error(f"Failed to calculate overall metrics: {e}")
    
    async def _calculate_sr_validation_score(self, result: BacktestResult) -> None:
        """Calculate S/R validation score based on level effectiveness."""
        try:
            # S/R validation score components (0-1)
            bounce_score = min(result.overall_bounce_rate / self.min_bounce_rate, 1.0)
            
            false_breakout_score = max(0, 1 - (result.overall_false_breakout_rate / self.max_false_breakout_rate))
            
            volume_score = min(result.avg_volume_confirmation_rate / self.min_volume_confirmation, 1.0)
            
            confidence_score = result.avg_confidence_score
            
            level_accuracy_score = result.level_detection_accuracy
            
            # Enhanced S/R validation score with comprehensive factors
            result.sr_validation_score = (
                bounce_score * 0.25 +                    # 25% - Bounce rate
                false_breakout_score * 0.20 +            # 20% - Low false breakouts
                volume_score * 0.15 +                    # 15% - Volume confirmation
                confidence_score * 0.10 +                # 10% - Overall confidence
                level_accuracy_score * 0.05 +            # 5% - Level detection accuracy
                result.level_persistence_score * 0.05 +  # 5% - Level persistence
                result.avg_trend_context_score * 0.05 +  # 5% - Trend context
                result.avg_candlestick_confirmation_rate * 0.05 +   # 5% - Price action confirmation
                result.avg_statistical_significance * 0.05 +  # 5% - Statistical significance
                result.avg_monte_carlo_score * 0.05      # 5% - Monte Carlo robustness
            )
            
            self.logger.info(f"📊 S/R Validation Score Components:")
            self.logger.info(f"  Bounce Score: {bounce_score:.3f}")
            self.logger.info(f"  False Breakout Score: {false_breakout_score:.3f}")
            self.logger.info(f"  Volume Score: {volume_score:.3f}")
            self.logger.info(f"  Confidence Score: {confidence_score:.3f}")
            self.logger.info(f"  Level Accuracy Score: {level_accuracy_score:.3f}")
            self.logger.info(f"  Overall S/R Validation Score: {result.sr_validation_score:.3f}")
            
        except Exception as e:
            self.logger.error(f"Failed to calculate S/R validation score: {e}")
    
    async def _calculate_market_context(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate market context metrics for S/R validation."""
        try:
            context = {}
            
            # Calculate trend
            if len(market_data) >= self.trend_period:
                sma = market_data['close'].rolling(window=self.trend_period).mean()
                current_price = market_data['close'].iloc[-1]
                current_sma = sma.iloc[-1]
                
                if current_price > current_sma:
                    context['trend'] = 'bullish'
                    context['trend_strength'] = (current_price - current_sma) / current_sma
                else:
                    context['trend'] = 'bearish'
                    context['trend_strength'] = (current_sma - current_price) / current_sma
            
            # Calculate volatility regime
            if len(market_data) >= self.volatility_period:
                returns = market_data['close'].pct_change().dropna()
                volatility = returns.rolling(window=self.volatility_period).std()
                current_volatility = volatility.iloc[-1]
                avg_volatility = volatility.mean()
                
                if current_volatility > avg_volatility * 1.2:
                    context['volatility_regime'] = 'high'
                elif current_volatility < avg_volatility * 0.8:
                    context['volatility_regime'] = 'low'
                else:
                    context['volatility_regime'] = 'normal'
                
                context['volatility_ratio'] = current_volatility / avg_volatility
            
            # Determine market structure (bull/bear market)
            if len(market_data) >= 200:  # Need enough data for market structure
                long_sma = market_data['close'].rolling(window=200).mean()
                short_sma = market_data['close'].rolling(window=50).mean()
                
                if short_sma.iloc[-1] > long_sma.iloc[-1]:
                    context['market_structure'] = 'bull'
                else:
                    context['market_structure'] = 'bear'
            
            return context
            
        except Exception as e:
            self.logger.error(f"Failed to calculate market context: {e}")
            return {}
    
    async def _analyze_time_based_factors(
        self,
        test: SRLevelTest,
        market_data: pd.DataFrame,
        touch_indices: List[int]
    ) -> None:
        """Analyze time-based factors affecting S/R level validity."""
        try:
            if not touch_indices:
                return
            
            # Calculate level age
            first_touch_idx = min(touch_indices)
            last_touch_idx = max(touch_indices)
            
            test.first_touch_date = market_data.index[first_touch_idx]
            test.last_touch_date = market_data.index[last_touch_idx]
            
            # Calculate age in days
            if test.first_touch_date and test.last_touch_date:
                age_delta = test.last_touch_date - test.first_touch_date
                test.level_age_days = age_delta.days
            
            # Calculate age decay factor
            if test.level_age_days > 0:
                # Exponential decay based on age
                test.age_decay_factor = self.age_decay_factor ** (test.level_age_days / 30)  # Decay per month
            
        except Exception as e:
            self.logger.error(f"Failed to analyze time-based factors: {e}")
    
    async def _analyze_market_context(
        self,
        test: SRLevelTest,
        market_context: Dict[str, Any],
        touch_indices: List[int]
    ) -> None:
        """Analyze how market context affects S/R level performance."""
        try:
            if not market_context:
                return
            
            # Trend context score
            if 'trend' in market_context:
                if test.level_type == 'support' and market_context['trend'] == 'bullish':
                    test.trend_context_score = 1.0
                elif test.level_type == 'resistance' and market_context['trend'] == 'bearish':
                    test.trend_context_score = 1.0
                else:
                    test.trend_context_score = 0.5  # Neutral
            
            # Volatility regime score
            if 'volatility_regime' in market_context:
                if market_context['volatility_regime'] == 'normal':
                    test.volatility_regime_score = 1.0
                elif market_context['volatility_regime'] == 'low':
                    test.volatility_regime_score = 0.8  # Slightly better in low volatility
                else:
                    test.volatility_regime_score = 0.6  # Worse in high volatility
            
            # Market structure score
            if 'market_structure' in market_context:
                if test.level_type == 'support' and market_context['market_structure'] == 'bull':
                    test.market_structure_score = 1.0
                elif test.level_type == 'resistance' and market_context['market_structure'] == 'bear':
                    test.market_structure_score = 1.0
                else:
                    test.market_structure_score = 0.7  # Still works but less effective
            
        except Exception as e:
            self.logger.error(f"Failed to analyze market context: {e}")
    
    async def _analyze_price_action(
        self,
        test: SRLevelTest,
        market_data: pd.DataFrame,
        touch_indices: List[int]
    ) -> None:
        """Analyze price action patterns at S/R levels."""
        try:
            if not touch_indices or not self.enable_price_action_analysis:
                return
            
            candlestick_confirmations = 0
            rejection_strength = 0.0
            consolidation_periods = 0
            
            for touch_idx in touch_indices:
                if touch_idx >= len(market_data) - 1:
                    continue
                
                # Analyze candlestick patterns
                current_bar = market_data.iloc[touch_idx]
                next_bar = market_data.iloc[touch_idx + 1]
                
                # Check for rejection patterns
                if test.level_type == 'support':
                    # Hammer, doji, or strong bounce
                    body_size = abs(current_bar['close'] - current_bar['open'])
                    lower_shadow = min(current_bar['open'], current_bar['close']) - current_bar['low']
                    
                    if lower_shadow > body_size * 2:  # Hammer-like pattern
                        candlestick_confirmations += 1
                        rejection_strength += lower_shadow / current_bar['close']
                    
                    # Check for bounce in next bar
                    if next_bar['close'] > current_bar['close']:
                        rejection_strength += (next_bar['close'] - current_bar['close']) / current_bar['close']
                
                elif test.level_type == 'resistance':
                    # Shooting star, doji, or strong rejection
                    body_size = abs(current_bar['close'] - current_bar['open'])
                    upper_shadow = current_bar['high'] - max(current_bar['open'], current_bar['close'])
                    
                    if upper_shadow > body_size * 2:  # Shooting star-like pattern
                        candlestick_confirmations += 1
                        rejection_strength += upper_shadow / current_bar['close']
                    
                    # Check for rejection in next bar
                    if next_bar['close'] < current_bar['close']:
                        rejection_strength += (current_bar['close'] - next_bar['close']) / current_bar['close']
                
                # Check for consolidation
                if touch_idx > 0 and touch_idx < len(market_data) - 1:
                    prev_bar = market_data.iloc[touch_idx - 1]
                    price_range = max(prev_bar['high'], current_bar['high'], next_bar['high']) - \
                                 min(prev_bar['low'], current_bar['low'], next_bar['low'])
                    avg_price = (prev_bar['close'] + current_bar['close'] + next_bar['close']) / 3
                    
                    if price_range / avg_price < 0.02:  # Less than 2% range
                        consolidation_periods += 1
            
            # Calculate scores
            if touch_indices:
                test.candlestick_confirmation_rate = candlestick_confirmations / len(touch_indices)
                test.rejection_pattern_score = min(rejection_strength / len(touch_indices), 1.0)
                test.consolidation_score = min(consolidation_periods / len(touch_indices), 1.0)
            
        except Exception as e:
            self.logger.error(f"Failed to analyze price action: {e}")
    
    async def _perform_statistical_validation(
        self,
        result: BacktestResult,
        market_data: pd.DataFrame
    ) -> None:
        """Perform statistical validation of S/R level results."""
        try:
            if not result.level_tests:
                return
            
            # Calculate statistical significance
            bounce_rates = [test.bounce_rate for test in result.level_tests if test.touches > 0]
            if bounce_rates:
                # Test if bounce rate is significantly different from random (50%)
                t_stat, p_value = stats.ttest_1samp(bounce_rates, 0.5)
                result.avg_statistical_significance = 1 - p_value  # Convert to confidence level
            
            # Monte Carlo simulation
            await self._run_monte_carlo_simulation(result, market_data)
            
            # Out-of-sample validation
            await self._perform_out_of_sample_validation(result, market_data)
            
        except Exception as e:
            self.logger.error(f"Failed to perform statistical validation: {e}")
    
    async def _run_monte_carlo_simulation(
        self,
        result: BacktestResult,
        market_data: pd.DataFrame
    ) -> None:
        """Run Monte Carlo simulation to test robustness."""
        try:
            if not result.level_tests:
                return
            
            # Simulate random S/R levels and compare performance
            random_scores = []
            actual_score = result.sr_validation_score
            
            for _ in range(self.monte_carlo_iterations):
                # Generate random levels
                random_levels = []
                for _ in range(len(result.level_tests)):
                    random_price = np.random.uniform(
                        market_data['low'].min(),
                        market_data['high'].max()
                    )
                    random_levels.append({
                        'price': random_price,
                        'type': np.random.choice(['support', 'resistance'])
                    })
                
                # Test random levels
                random_validator = SRBacktestingValidator(self.config)
                random_result = await random_validator.validate_sr_levels(
                    market_data, random_levels, market_data['close'].iloc[-1]
                )
                
                if random_result:
                    random_scores.append(random_result.sr_validation_score)
            
            # Calculate robustness score
            if random_scores:
                better_than_random = sum(1 for score in random_scores if actual_score > score)
                result.avg_monte_carlo_score = better_than_random / len(random_scores)
            
        except Exception as e:
            self.logger.error(f"Failed to run Monte Carlo simulation: {e}")
    
    async def _perform_out_of_sample_validation(
        self,
        result: BacktestResult,
        market_data: pd.DataFrame
    ) -> None:
        """Perform out-of-sample validation."""
        try:
            if len(market_data) < 100:  # Need enough data
                return
            
            # Split data into training and testing
            split_idx = int(len(market_data) * (1 - self.out_of_sample_ratio))
            train_data = market_data.iloc[:split_idx]
            test_data = market_data.iloc[split_idx:]
            
            # Get S/R levels from training data
            from src.tactician.sr_breakout_predictor import SRBreakoutPredictor
            sr_predictor = SRBreakoutPredictor(self.config)
            await sr_predictor.initialize()
            
            train_price = train_data['close'].iloc[-1]
            train_context = await sr_predictor.get_sr_context(train_data, train_price)
            
            train_levels = train_context.get("support_levels", []) + train_context.get("resistance_levels", [])
            
            if train_levels:
                # Test on out-of-sample data
                test_validator = SRBacktestingValidator(self.config)
                test_result = await test_validator.validate_sr_levels(
                    test_data, train_levels, test_data['close'].iloc[-1]
                )
                
                if test_result:
                    result.out_of_sample_score = test_result.sr_validation_score
            
        except Exception as e:
            self.logger.error(f"Failed to perform out-of-sample validation: {e}")


# Setup function for easy integration
async def setup_sr_backtesting_validator(config: Dict[str, Any]) -> Optional[SRBacktestingValidator]:
    """Setup S/R backtesting validator."""
    try:
        validator = SRBacktestingValidator(config)
        return validator
    except Exception as e:
        system_logger.error(f"Failed to setup S/R backtesting validator: {e}")
        return None