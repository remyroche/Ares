"""
Period Validator for Data-Driven Period Selection

This module provides validation, filtering, and ranking capabilities for
period selection based on data characteristics and quality metrics.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import logging
from collections import Counter

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance
)

from .period_analysis_utils import (
    PeriodAnalysisUtils, ValidationError, AnalysisError,
    performance_monitoring, safe_validate_and_execute
)

logger = logging.getLogger(__name__)


class PeriodValidator:
    """
    Validates, filters, and ranks periods based on data characteristics and quality metrics.
    
    This class handles all validation operations including period filtering,
    ranking, categorization, and confidence scoring.
    """
    
    def __init__(self, 
                 min_period: int = 2,
                 max_period: int = 200,
                 max_periods: int = 8,
                 min_data_points: int = 100):
        """
        Initialize the period validator.
        
        Args:
            min_period: Minimum period to consider
            max_period: Maximum period to consider
            max_periods: Maximum number of periods to return
            min_data_points: Minimum data points required for analysis
        """
        self.min_period = min_period
        self.max_period = max_period
        self.max_periods = max_periods
        self.min_data_points = min_data_points
        
        tprint_info(f"🔧 Period validator initialized")
        tprint_info(f"📊 Period range: {min_period} - {max_period}")
        tprint_info(f"📊 Max periods: {max_periods}")
        tprint_info(f"📊 Min data points: {min_data_points}")
    
    def filter_periods(self, periods: List[int], 
                      characteristics: Dict[str, Any]) -> List[int]:
        """
        Filter periods based on data characteristics and constraints.
        
        Args:
            periods: List of candidate periods
            characteristics: Data characteristics
            
        Returns:
            List of filtered periods
            
        Raises:
            ValidationError: If inputs are invalid
            AnalysisError: If filtering fails
        """
        def _validate_inputs():
            PeriodAnalysisUtils.validate_periods(periods, self.min_period, self.max_period, "period_filtering")
            if not isinstance(characteristics, dict):
                raise ValidationError("Characteristics must be a dictionary")
        
        def _filter_periods():
            filtered = []
            data_length = characteristics.get('data_length', 0)
            timeframe_minutes = characteristics.get('timeframe_minutes', 15)
            
            if data_length <= 0:
                raise AnalysisError("Data length must be positive")
            
            tprint_debug(f"📊 Filtering {len(periods)} periods")
            tprint_debug(f"📊 Data length: {data_length}, Timeframe: {timeframe_minutes}min")
            
            for period in periods:
                if not isinstance(period, int):
                    tprint_debug(f"⚠️ Skipping non-integer period: {period}")
                    continue
                
                # Check if period is within bounds
                if not (self.min_period <= period <= self.max_period):
                    tprint_debug(f"❌ Period {period} outside bounds [{self.min_period}, {self.max_period}]")
                    continue
                
                # Check if period is reasonable for data length
                if period > data_length // 4:  # Don't use periods longer than 1/4 of data
                    tprint_debug(f"❌ Period {period} too long for data length {data_length}")
                    continue
                
                # Check if period makes sense for timeframe
                if period < 2:  # At least 2 periods
                    tprint_debug(f"❌ Period {period} too short")
                    continue
                
                filtered.append(period)
                tprint_debug(f"✅ Period {period} passed all filters")
            
            result = sorted(list(set(filtered)))
            tprint_debug(f"✅ Filtered to {len(result)} periods: {result}")
            return result
        
        return safe_validate_and_execute(
            _validate_inputs, _filter_periods, "period_filtering"
        )
    
    def rank_periods(self, periods: List[int], 
                    data: pd.DataFrame, 
                    characteristics: Dict[str, Any]) -> List[int]:
        """
        Rank periods by their potential usefulness.
        
        Args:
            periods: List of periods to rank
            data: Input data
            characteristics: Data characteristics
            
        Returns:
            List of periods ranked by usefulness
            
        Raises:
            ValidationError: If inputs are invalid
            AnalysisError: If ranking fails
        """
        def _validate_inputs():
            PeriodAnalysisUtils.validate_periods(periods, self.min_period, self.max_period, "period_ranking")
            PeriodAnalysisUtils.validate_dataframe(data, operation_name="period_ranking")
            if not isinstance(characteristics, dict):
                raise ValidationError("Characteristics must be a dictionary")
        
        def _rank_periods():
            if not periods:
                return []
            
            scores = []
            data_length = characteristics.get('data_length', 0)
            
            if data_length <= 0:
                raise AnalysisError("Data length must be positive")
            
            tprint_debug(f"📊 Ranking {len(periods)} periods")
            
            for period in periods:
                if not isinstance(period, int):
                    continue
                
                score = 0
                score_components = {}
                
                # Diversity score (prefer periods that are different from others)
                other_periods = [p for p in periods if p != period and isinstance(p, int)]
                if other_periods:
                    min_diff = min(abs(period - p) for p in other_periods)
                    diversity_score = min_diff / max(period, 1)
                    score += diversity_score
                    score_components['diversity'] = diversity_score
                
                # Data coverage score (prefer periods that use more data)
                coverage = min(period, data_length) / data_length
                score += coverage
                score_components['coverage'] = coverage
                
                # Stability score (prefer periods that are stable across different windows)
                if 'close' in data.columns and len(data) > period * 2:
                    try:
                        returns = data['close'].pct_change().dropna()
                        if len(returns) > period:
                            rolling_vol = returns.rolling(period).std()
                            vol_stability = 1 / (rolling_vol.std() + 1e-8)
                            score += vol_stability
                            score_components['stability'] = vol_stability
                        else:
                            score_components['stability'] = 0
                    except Exception as e:
                        tprint_debug(f"⚠️ Stability calculation failed for period {period}: {e}")
                        score_components['stability'] = 0
                
                scores.append((score, period))
                tprint_debug(f"📊 Period {period}: score={score:.3f}, components={score_components}")
            
            if not scores:
                return []
            
            # Sort by score (descending)
            scores.sort(reverse=True)
            ranked_periods = [period for score, period in scores]
            
            tprint_debug(f"✅ Ranked periods: {ranked_periods}")
            return ranked_periods
        
        return safe_validate_and_execute(
            _validate_inputs, _rank_periods, "period_ranking"
        )
    
    def categorize_periods(self, periods: List[int], 
                          characteristics: Dict[str, Any]) -> Dict[str, List[int]]:
        """
        Categorize periods by their characteristics.
        
        Args:
            periods: List of periods to categorize
            characteristics: Data characteristics
            
        Returns:
            Dictionary with period categories
            
        Raises:
            ValidationError: If inputs are invalid
            AnalysisError: If categorization fails
        """
        def _validate_inputs():
            PeriodAnalysisUtils.validate_periods(periods, self.min_period, self.max_period, "period_categorization")
            if not isinstance(characteristics, dict):
                raise ValidationError("Characteristics must be a dictionary")
        
        def _categorize_periods():
            categories = {
                'short_term': [],
                'medium_term': [],
                'long_term': [],
                'volatility_driven': [],
                'trend_driven': [],
                'volume_driven': []
            }
            
            data_length = characteristics.get('data_length', 0)
            volatility_clusters = characteristics.get('volatility_clusters', [])
            trend_cycles = characteristics.get('trend_cycles', [])
            volume_patterns = characteristics.get('volume_patterns', {})
            volume_periods = volume_patterns.get('spike_periods', []) if isinstance(volume_patterns, dict) else []
            
            if data_length <= 0:
                raise AnalysisError("Data length must be positive")
            
            tprint_debug(f"📊 Categorizing {len(periods)} periods")
            tprint_debug(f"📊 Data length: {data_length}")
            tprint_debug(f"📊 Volatility clusters: {volatility_clusters}")
            tprint_debug(f"📊 Trend cycles: {trend_cycles}")
            tprint_debug(f"📊 Volume periods: {volume_periods}")
            
            for period in periods:
                if not isinstance(period, int):
                    continue
                
                # Time-based categorization
                if period <= data_length // 20:
                    categories['short_term'].append(period)
                    tprint_debug(f"✅ Period {period} categorized as short_term")
                elif period <= data_length // 10:
                    categories['medium_term'].append(period)
                    tprint_debug(f"✅ Period {period} categorized as medium_term")
                else:
                    categories['long_term'].append(period)
                    tprint_debug(f"✅ Period {period} categorized as long_term")
                
                # Pattern-based categorization
                if period in volatility_clusters:
                    categories['volatility_driven'].append(period)
                    tprint_debug(f"✅ Period {period} categorized as volatility_driven")
                
                if period in trend_cycles:
                    categories['trend_driven'].append(period)
                    tprint_debug(f"✅ Period {period} categorized as trend_driven")
                
                if period in volume_periods:
                    categories['volume_driven'].append(period)
                    tprint_debug(f"✅ Period {period} categorized as volume_driven")
            
            tprint_debug(f"✅ Period categorization complete: {categories}")
            return categories
        
        return safe_validate_and_execute(
            _validate_inputs, _categorize_periods, "period_categorization"
        )
    
    def calculate_confidence_score(self, periods: List[int], 
                                 characteristics: Dict[str, Any]) -> float:
        """
        Calculate confidence score for selected periods.
        
        Args:
            periods: List of selected periods
            characteristics: Data characteristics
            
        Returns:
            Confidence score between 0 and 1
            
        Raises:
            ValidationError: If inputs are invalid
        """
        def _validate_inputs():
            if not isinstance(periods, list):
                raise ValidationError("Periods must be a list")
            if not isinstance(characteristics, dict):
                raise ValidationError("Characteristics must be a dictionary")
        
        def _calculate_confidence():
            if not periods:
                return 0.0
            
            return PeriodAnalysisUtils.calculate_confidence_score(periods, characteristics)
        
        return safe_validate_and_execute(
            _validate_inputs, _calculate_confidence, "confidence_calculation"
        )
    
    def validate_period_quality(self, periods: List[int], 
                               data: pd.DataFrame,
                               characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the quality of selected periods.
        
        Args:
            periods: List of periods to validate
            data: Input data
            characteristics: Data characteristics
            
        Returns:
            Dictionary with quality metrics
            
        Raises:
            ValidationError: If inputs are invalid
            AnalysisError: If validation fails
        """
        def _validate_inputs():
            PeriodAnalysisUtils.validate_periods(periods, self.min_period, self.max_period, "period_quality_validation")
            PeriodAnalysisUtils.validate_dataframe(data, operation_name="period_quality_validation")
            if not isinstance(characteristics, dict):
                raise ValidationError("Characteristics must be a dictionary")
        
        def _validate_quality():
            quality_metrics = {
                'total_periods': len(periods),
                'valid_periods': 0,
                'diversity_score': 0.0,
                'coverage_score': 0.0,
                'stability_score': 0.0,
                'confidence_score': 0.0,
                'warnings': [],
                'recommendations': []
            }
            
            data_length = characteristics.get('data_length', 0)
            
            if data_length <= 0:
                raise AnalysisError("Data length must be positive")
            
            # Count valid periods
            valid_periods = [p for p in periods if isinstance(p, int) and self.min_period <= p <= self.max_period]
            quality_metrics['valid_periods'] = len(valid_periods)
            
            # Calculate diversity score
            if len(valid_periods) > 1:
                periods_sorted = sorted(valid_periods)
                min_gaps = [periods_sorted[i+1] - periods_sorted[i] for i in range(len(periods_sorted)-1)]
                if min_gaps:
                    quality_metrics['diversity_score'] = min(min_gaps) / max(valid_periods)
            
            # Calculate coverage score
            if valid_periods:
                max_period = max(valid_periods)
                quality_metrics['coverage_score'] = min(max_period, data_length) / data_length
            
            # Calculate stability score (simplified)
            if 'close' in data.columns and valid_periods:
                try:
                    returns = data['close'].pct_change().dropna()
                    stability_scores = []
                    for period in valid_periods[:3]:  # Check top 3 periods
                        if len(returns) > period:
                            rolling_vol = returns.rolling(period).std()
                            if not rolling_vol.empty:
                                stability = 1 / (rolling_vol.std() + 1e-8)
                                stability_scores.append(stability)
                    
                    if stability_scores:
                        quality_metrics['stability_score'] = np.mean(stability_scores)
                except Exception:
                    pass
            
            # Calculate confidence score
            quality_metrics['confidence_score'] = self.calculate_confidence_score(periods, characteristics)
            
            # Generate warnings and recommendations
            if len(valid_periods) < 3:
                quality_metrics['warnings'].append("Few periods selected - consider increasing max_periods")
            
            if quality_metrics['diversity_score'] < 0.1:
                quality_metrics['warnings'].append("Low period diversity - periods may be too similar")
            
            if quality_metrics['coverage_score'] < 0.1:
                quality_metrics['warnings'].append("Low data coverage - periods may be too short")
            
            if quality_metrics['confidence_score'] < 0.5:
                quality_metrics['recommendations'].append("Consider providing more data for better analysis")
            
            if len(valid_periods) > self.max_periods:
                quality_metrics['recommendations'].append(f"Consider reducing max_periods from {len(valid_periods)} to {self.max_periods}")
            
            tprint_debug(f"✅ Period quality validation complete: {quality_metrics}")
            return quality_metrics
        
        return safe_validate_and_execute(
            _validate_inputs, _validate_quality, "period_quality_validation"
        )
    
    def get_base_periods_from_timeframe(self, timeframe_minutes: int, 
                                       target_timeframe: Optional[str] = None) -> List[int]:
        """
        Get base periods based on timeframe.
        
        Args:
            timeframe_minutes: Current timeframe in minutes
            target_timeframe: Target timeframe string (e.g., "15m", "60m")
            
        Returns:
            List of base periods
            
        Raises:
            ValidationError: If inputs are invalid
        """
        def _validate_inputs():
            if not isinstance(timeframe_minutes, int) or timeframe_minutes <= 0:
                raise ValidationError("Timeframe minutes must be a positive integer")
        
        def _get_base_periods():
            if target_timeframe:
                # Parse target timeframe
                if target_timeframe.endswith('m'):
                    target_minutes = int(target_timeframe[:-1])
                elif target_timeframe.endswith('h'):
                    target_minutes = int(target_timeframe[:-1]) * 60
                elif target_timeframe.endswith('d'):
                    target_minutes = int(target_timeframe[:-1]) * 24 * 60
                else:
                    target_minutes = 15  # Default
                    tprint_debug("⚠️ Unknown target timeframe format, using default 15 minutes")
            else:
                target_minutes = timeframe_minutes
            
            tprint_debug(f"📊 Target minutes: {target_minutes}, Current timeframe: {timeframe_minutes}")
            
            # Calculate periods based on target timeframe
            base_periods = []
            
            # Short-term periods (2-10x current timeframe)
            for multiplier in [2, 3, 5, 10]:
                period = multiplier * (target_minutes // timeframe_minutes)
                if self.min_period <= period <= self.max_period:
                    base_periods.append(period)
                    tprint_debug(f"✅ Added short-term period: {period} (multiplier: {multiplier})")
            
            # Medium-term periods (20-50x current timeframe)
            for multiplier in [20, 30, 50]:
                period = multiplier * (target_minutes // timeframe_minutes)
                if self.min_period <= period <= self.max_period:
                    base_periods.append(period)
                    tprint_debug(f"✅ Added medium-term period: {period} (multiplier: {multiplier})")
            
            # Long-term periods (100x+ current timeframe)
            for multiplier in [100, 200]:
                period = multiplier * (target_minutes // timeframe_minutes)
                if self.min_period <= period <= self.max_period:
                    base_periods.append(period)
                    tprint_debug(f"✅ Added long-term period: {period} (multiplier: {multiplier})")
            
            tprint_debug(f"✅ Generated {len(base_periods)} base periods: {base_periods}")
            return base_periods
        
        return safe_validate_and_execute(
            _validate_inputs, _get_base_periods, "base_period_generation"
        )
    
    def select_optimal_periods(self, periods: List[int], 
                              data: pd.DataFrame,
                              characteristics: Dict[str, Any]) -> List[int]:
        """
        Select optimal periods from candidates using filtering and ranking.
        
        Args:
            periods: List of candidate periods
            data: Input data
            characteristics: Data characteristics
            
        Returns:
            List of optimal periods
            
        Raises:
            ValidationError: If inputs are invalid
            AnalysisError: If selection fails
        """
        def _validate_inputs():
            PeriodAnalysisUtils.validate_periods(periods, self.min_period, self.max_period, "optimal_period_selection")
            PeriodAnalysisUtils.validate_dataframe(data, operation_name="optimal_period_selection")
            if not isinstance(characteristics, dict):
                raise ValidationError("Characteristics must be a dictionary")
        
        def _select_optimal():
            # Filter periods
            filtered_periods = self.filter_periods(periods, characteristics)
            
            if not filtered_periods:
                tprint_warning("⚠️ No periods passed filtering")
                return []
            
            # Rank periods
            ranked_periods = self.rank_periods(filtered_periods, data, characteristics)
            
            # Select top periods
            optimal_periods = ranked_periods[:self.max_periods]
            
            tprint_success(f"✅ Selected {len(optimal_periods)} optimal periods: {optimal_periods}")
            return optimal_periods
        
        return safe_validate_and_execute(
            _validate_inputs, _select_optimal, "optimal_period_selection"
        )