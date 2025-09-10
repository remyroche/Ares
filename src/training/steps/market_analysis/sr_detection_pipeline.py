"""
Support/Resistance Detection Pipeline

This module provides comprehensive support and resistance level detection
using multiple algorithms and validation techniques.

Key Features:
- Multiple SR detection algorithms (pivot points, swing highs/lows, volume-based)
- Statistical validation of SR levels
- Strength scoring and ranking
- Data quality validation using existing utilities
- Integration with ML commons for enhanced analysis
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import json

from src.utils.logger import system_logger
from src.utils.data_processing_utils import DataFrameValidator, DataQualityReport
from src.utils.enhanced_data_quality_validator import EnhancedDataQualityValidator, QualityResult
from src.utils.ml_common.data_quality import DataQualityUtilities
from src.utils.common_operations import CommonOperations
from src.utils.math_validation import MathValidation
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler

logger = system_logger.getChild('SRDetectionPipeline')

@dataclass
class SRLevel:
    """Support/Resistance level representation."""
    level: float
    level_type: str  # 'support' or 'resistance'
    strength: float  # 0.0 to 1.0
    touches: int
    first_touch: int
    last_touch: int
    volume_at_level: float
    confidence: float
    algorithm: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SRDetectionConfig:
    """Configuration for SR detection."""
    # Detection parameters
    min_touches: int = 2
    max_levels: int = 50
    strength_threshold: float = 0.3
    lookback_periods: int = 100
    
    # Algorithm selection
    use_pivot_points: bool = True
    use_swing_levels: bool = True
    use_volume_levels: bool = True
    use_fractal_levels: bool = True
    
    # Validation parameters
    min_volume_ratio: float = 0.5
    max_price_deviation: float = 0.02  # 2%
    min_time_between_touches: int = 5
    
    # Data quality validation
    enable_data_quality_validation: bool = True
    quality_thresholds: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SRDetectionResult:
    """Result of SR detection."""
    levels: List[SRLevel]
    metrics: Dict[str, Any]
    quality_report: Optional[QualityResult] = None
    algorithm_performance: Dict[str, Any] = field(default_factory=dict)

class SRDetectionPipeline:
    """
    Support/Resistance Detection Pipeline.
    
    Provides comprehensive SR level detection with multiple algorithms
    and data quality validation.
    """
    
    def __init__(self, config: Optional[SRDetectionConfig] = None):
        """Initialize SR detection pipeline."""
        self.config = config or SRDetectionConfig()
        self.logger = logger.getChild('SRDetectionPipeline')
        self.common_ops = CommonOperations()
        self.math_validator = MathValidation()
        
        # Initialize data quality utilities
        self.data_quality_validator = EnhancedDataQualityValidator()
        self.ml_data_quality = None
        try:
            self.ml_data_quality = DataQualityUtilities()
            self.logger.info("✅ ML data quality utilities initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ ML data quality utilities not available: {e}")
    
    async def detect_sr_levels(
        self,
        data_dir: str,
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> SRDetectionResult:
        """
        Detect support and resistance levels.
        
        Args:
            data_dir: Data directory path
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Data timeframe
            
        Returns:
            SRDetectionResult with detected levels and metrics
        """
        self.logger.info(f"🔍 Starting SR detection for {symbol} on {exchange} ({timeframe})")
        
        try:
            # Load and validate data
            data = await self._load_and_validate_data(data_dir, symbol, exchange, timeframe)
            
            # Perform data quality validation
            quality_report = None
            if self.config.enable_data_quality_validation:
                quality_report = await self._validate_data_quality(data, symbol, exchange)
            
            # Detect SR levels using multiple algorithms
            all_levels = []
            algorithm_performance = {}
            
            if self.config.use_pivot_points:
                pivot_levels, pivot_perf = await self._detect_pivot_levels(data)
                all_levels.extend(pivot_levels)
                algorithm_performance['pivot_points'] = pivot_perf
            
            if self.config.use_swing_levels:
                swing_levels, swing_perf = await self._detect_swing_levels(data)
                all_levels.extend(swing_levels)
                algorithm_performance['swing_levels'] = swing_perf
            
            if self.config.use_volume_levels:
                volume_levels, volume_perf = await self._detect_volume_levels(data)
                all_levels.extend(volume_levels)
                algorithm_performance['volume_levels'] = volume_perf
            
            if self.config.use_fractal_levels:
                fractal_levels, fractal_perf = await self._detect_fractal_levels(data)
                all_levels.extend(fractal_levels)
                algorithm_performance['fractal_levels'] = fractal_perf
            
            # Merge and validate levels
            merged_levels = await self._merge_and_validate_levels(all_levels, data)
            
            # Calculate metrics
            metrics = await self._calculate_detection_metrics(merged_levels, data)
            
            result = SRDetectionResult(
                levels=merged_levels,
                metrics=metrics,
                quality_report=quality_report,
                algorithm_performance=algorithm_performance
            )
            
            self.logger.info(f"✅ SR detection completed: {len(merged_levels)} levels found")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ SR detection failed: {e}")
            raise
    
    async def _load_and_validate_data(
        self,
        data_dir: str,
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> pd.DataFrame:
        """Load and validate market data."""
        # Construct file path
        file_path = Path(data_dir) / f"aggtrades_{exchange}_{symbol}_consolidated.parquet"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Load data using standardized handler
        data = standardized_parquet_handler.read_parquet_standardized(file_path)
        
        # Basic validation
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Sort by timestamp if available
        if 'timestamp' in data.columns:
            data = data.sort_values('timestamp').reset_index(drop=True)
        
        self.logger.info(f"📊 Loaded {len(data)} data points for SR detection")
        return data
    
    async def _validate_data_quality(
        self,
        data: pd.DataFrame,
        symbol: str,
        exchange: str
    ) -> QualityResult:
        """Validate data quality using existing utilities."""
        self.logger.info("🔍 Performing data quality validation")
        
        try:
            # Use enhanced data quality validator
            quality_result = self.data_quality_validator.validate_dataframe(data)
            
            # Use ML data quality utilities if available
            if self.ml_data_quality:
                try:
                    ml_quality_report = await self.ml_data_quality.perform_comprehensive_validation(
                        data, symbol=symbol, exchange=exchange
                    )
                    
                    # Merge ML quality insights
                    if ml_quality_report.get('has_critical_issues', False):
                        for issue in ml_quality_report.get('critical_issues', []):
                            quality_result.add_issue('ml_critical', issue)
                    
                    if ml_quality_report.get('warnings', []):
                        for warning in ml_quality_report.get('warnings', []):
                            quality_result.add_warning('ml_warning', warning)
                    
                    self.logger.info("✅ ML-enhanced data quality validation completed")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ ML data quality validation failed: {e}")
            
            # Log quality results
            if quality_result.passed:
                self.logger.info("✅ Data quality validation passed")
            else:
                self.logger.warning(f"⚠️ Data quality issues found: {len(quality_result.issues)} issues, {len(quality_result.warnings)} warnings")
                for issue in quality_result.issues[:5]:  # Log first 5 issues
                    self.logger.warning(f"  - {issue}")
            
            return quality_result
            
        except Exception as e:
            self.logger.error(f"❌ Data quality validation failed: {e}")
            # Return a basic quality result
            return QualityResult(passed=False, issues=[f"Validation failed: {e}"])
    
    async def _detect_pivot_levels(self, data: pd.DataFrame) -> Tuple[List[SRLevel], Dict[str, Any]]:
        """Detect SR levels using pivot point analysis."""
        self.logger.info("📊 Detecting pivot point levels")
        
        levels = []
        performance_metrics = {
            'algorithm': 'pivot_points',
            'levels_found': 0,
            'execution_time': 0.0
        }
        
        try:
            import time
            start_time = time.time()
            
            # Calculate pivot points
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            
            # Find pivot highs and lows
            pivot_highs = []
            pivot_lows = []
            
            for i in range(self.config.lookback_periods, len(data) - self.config.lookback_periods):
                # Check for pivot high
                is_pivot_high = True
                for j in range(i - self.config.lookback_periods, i + self.config.lookback_periods + 1):
                    if j != i and high[j] >= high[i]:
                        is_pivot_high = False
                        break
                
                if is_pivot_high:
                    pivot_highs.append((i, high[i]))
                
                # Check for pivot low
                is_pivot_low = True
                for j in range(i - self.config.lookback_periods, i + self.config.lookback_periods + 1):
                    if j != i and low[j] <= low[i]:
                        is_pivot_low = False
                        break
                
                if is_pivot_low:
                    pivot_lows.append((i, low[i]))
            
            # Convert to SR levels
            for idx, price in pivot_highs:
                level = SRLevel(
                    level=price,
                    level_type='resistance',
                    strength=0.6,  # Base strength for pivot points
                    touches=1,
                    first_touch=idx,
                    last_touch=idx,
                    volume_at_level=data.iloc[idx]['volume'],
                    confidence=0.7,
                    algorithm='pivot_points',
                    metadata={'pivot_type': 'high', 'lookback_periods': self.config.lookback_periods}
                )
                levels.append(level)
            
            for idx, price in pivot_lows:
                level = SRLevel(
                    level=price,
                    level_type='support',
                    strength=0.6,  # Base strength for pivot points
                    touches=1,
                    first_touch=idx,
                    last_touch=idx,
                    volume_at_level=data.iloc[idx]['volume'],
                    confidence=0.7,
                    algorithm='pivot_points',
                    metadata={'pivot_type': 'low', 'lookback_periods': self.config.lookback_periods}
                )
                levels.append(level)
            
            performance_metrics['levels_found'] = len(levels)
            performance_metrics['execution_time'] = time.time() - start_time
            
            self.logger.info(f"📊 Pivot points: {len(levels)} levels found")
            
        except Exception as e:
            self.logger.error(f"❌ Pivot point detection failed: {e}")
            performance_metrics['error'] = str(e)
        
        return levels, performance_metrics
    
    async def _detect_swing_levels(self, data: pd.DataFrame) -> Tuple[List[SRLevel], Dict[str, Any]]:
        """Detect SR levels using swing high/low analysis."""
        self.logger.info("📊 Detecting swing levels")
        
        levels = []
        performance_metrics = {
            'algorithm': 'swing_levels',
            'levels_found': 0,
            'execution_time': 0.0
        }
        
        try:
            import time
            start_time = time.time()
            
            # Simple swing detection
            high = data['high'].values
            low = data['low'].values
            
            # Find swing highs (local maxima)
            swing_highs = []
            for i in range(2, len(data) - 2):
                if (high[i] > high[i-1] and high[i] > high[i-2] and 
                    high[i] > high[i+1] and high[i] > high[i+2]):
                    swing_highs.append((i, high[i]))
            
            # Find swing lows (local minima)
            swing_lows = []
            for i in range(2, len(data) - 2):
                if (low[i] < low[i-1] and low[i] < low[i-2] and 
                    low[i] < low[i+1] and low[i] < low[i+2]):
                    swing_lows.append((i, low[i]))
            
            # Convert to SR levels
            for idx, price in swing_highs:
                level = SRLevel(
                    level=price,
                    level_type='resistance',
                    strength=0.5,  # Base strength for swing levels
                    touches=1,
                    first_touch=idx,
                    last_touch=idx,
                    volume_at_level=data.iloc[idx]['volume'],
                    confidence=0.6,
                    algorithm='swing_levels',
                    metadata={'swing_type': 'high'}
                )
                levels.append(level)
            
            for idx, price in swing_lows:
                level = SRLevel(
                    level=price,
                    level_type='support',
                    strength=0.5,  # Base strength for swing levels
                    touches=1,
                    first_touch=idx,
                    last_touch=idx,
                    volume_at_level=data.iloc[idx]['volume'],
                    confidence=0.6,
                    algorithm='swing_levels',
                    metadata={'swing_type': 'low'}
                )
                levels.append(level)
            
            performance_metrics['levels_found'] = len(levels)
            performance_metrics['execution_time'] = time.time() - start_time
            
            self.logger.info(f"📊 Swing levels: {len(levels)} levels found")
            
        except Exception as e:
            self.logger.error(f"❌ Swing level detection failed: {e}")
            performance_metrics['error'] = str(e)
        
        return levels, performance_metrics
    
    async def _detect_volume_levels(self, data: pd.DataFrame) -> Tuple[List[SRLevel], Dict[str, Any]]:
        """Detect SR levels based on volume analysis."""
        self.logger.info("📊 Detecting volume-based levels")
        
        levels = []
        performance_metrics = {
            'algorithm': 'volume_levels',
            'levels_found': 0,
            'execution_time': 0.0
        }
        
        try:
            import time
            start_time = time.time()
            
            # Calculate volume-weighted average price (VWAP)
            volume = data['volume'].values
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            
            # Calculate typical price
            typical_price = (high + low + close) / 3
            
            # Calculate VWAP
            vwap = np.cumsum(typical_price * volume) / np.cumsum(volume)
            
            # Find significant volume levels
            volume_threshold = np.percentile(volume, 80)  # Top 20% volume
            
            for i in range(len(data)):
                if volume[i] > volume_threshold:
                    # Create level at the price where high volume occurred
                    price = close[i]
                    
                    # Determine if it's support or resistance based on price action
                    level_type = 'support' if price < vwap[i] else 'resistance'
                    
                    level = SRLevel(
                        level=price,
                        level_type=level_type,
                        strength=0.7,  # High strength for volume-based levels
                        touches=1,
                        first_touch=i,
                        last_touch=i,
                        volume_at_level=volume[i],
                        confidence=0.8,
                        algorithm='volume_levels',
                        metadata={'volume_ratio': volume[i] / np.mean(volume)}
                    )
                    levels.append(level)
            
            performance_metrics['levels_found'] = len(levels)
            performance_metrics['execution_time'] = time.time() - start_time
            
            self.logger.info(f"📊 Volume levels: {len(levels)} levels found")
            
        except Exception as e:
            self.logger.error(f"❌ Volume level detection failed: {e}")
            performance_metrics['error'] = str(e)
        
        return levels, performance_metrics
    
    async def _detect_fractal_levels(self, data: pd.DataFrame) -> Tuple[List[SRLevel], Dict[str, Any]]:
        """Detect SR levels using fractal analysis."""
        self.logger.info("📊 Detecting fractal levels")
        
        levels = []
        performance_metrics = {
            'algorithm': 'fractal_levels',
            'levels_found': 0,
            'execution_time': 0.0
        }
        
        try:
            import time
            start_time = time.time()
            
            # Simple fractal detection (5-point fractals)
            high = data['high'].values
            low = data['low'].values
            
            # Find fractal highs
            fractal_highs = []
            for i in range(2, len(data) - 2):
                if (high[i] > high[i-1] and high[i] > high[i-2] and 
                    high[i] > high[i+1] and high[i] > high[i+2]):
                    fractal_highs.append((i, high[i]))
            
            # Find fractal lows
            fractal_lows = []
            for i in range(2, len(data) - 2):
                if (low[i] < low[i-1] and low[i] < low[i-2] and 
                    low[i] < low[i+1] and low[i] < low[i+2]):
                    fractal_lows.append((i, low[i]))
            
            # Convert to SR levels
            for idx, price in fractal_highs:
                level = SRLevel(
                    level=price,
                    level_type='resistance',
                    strength=0.6,  # Base strength for fractal levels
                    touches=1,
                    first_touch=idx,
                    last_touch=idx,
                    volume_at_level=data.iloc[idx]['volume'],
                    confidence=0.7,
                    algorithm='fractal_levels',
                    metadata={'fractal_type': 'high'}
                )
                levels.append(level)
            
            for idx, price in fractal_lows:
                level = SRLevel(
                    level=price,
                    level_type='support',
                    strength=0.6,  # Base strength for fractal levels
                    touches=1,
                    first_touch=idx,
                    last_touch=idx,
                    volume_at_level=data.iloc[idx]['volume'],
                    confidence=0.7,
                    algorithm='fractal_levels',
                    metadata={'fractal_type': 'low'}
                )
                levels.append(level)
            
            performance_metrics['levels_found'] = len(levels)
            performance_metrics['execution_time'] = time.time() - start_time
            
            self.logger.info(f"📊 Fractal levels: {len(levels)} levels found")
            
        except Exception as e:
            self.logger.error(f"❌ Fractal level detection failed: {e}")
            performance_metrics['error'] = str(e)
        
        return levels, performance_metrics
    
    async def _merge_and_validate_levels(
        self,
        all_levels: List[SRLevel],
        data: pd.DataFrame
    ) -> List[SRLevel]:
        """Merge similar levels and validate them."""
        self.logger.info("🔗 Merging and validating SR levels")
        
        if not all_levels:
            return []
        
        # Group levels by proximity
        merged_levels = []
        used_indices = set()
        
        for i, level in enumerate(all_levels):
            if i in used_indices:
                continue
            
            # Find similar levels within price tolerance
            similar_levels = [level]
            similar_indices = [i]
            
            for j, other_level in enumerate(all_levels[i+1:], i+1):
                if j in used_indices:
                    continue
                
                price_diff = abs(level.level - other_level.level)
                price_tolerance = level.level * self.config.max_price_deviation
                
                if price_diff <= price_tolerance and level.level_type == other_level.level_type:
                    similar_levels.append(other_level)
                    similar_indices.append(j)
            
            # Merge similar levels
            if len(similar_levels) > 1:
                # Calculate weighted average level
                total_volume = sum(l.volume_at_level for l in similar_levels)
                weighted_level = sum(l.level * l.volume_at_level for l in similar_levels) / total_volume
                
                # Calculate combined strength
                combined_strength = min(1.0, sum(l.strength for l in similar_levels) / len(similar_levels) + 0.1)
                
                # Calculate combined confidence
                combined_confidence = min(1.0, sum(l.confidence for l in similar_levels) / len(similar_levels) + 0.1)
                
                merged_level = SRLevel(
                    level=weighted_level,
                    level_type=level.level_type,
                    strength=combined_strength,
                    touches=sum(l.touches for l in similar_levels),
                    first_touch=min(l.first_touch for l in similar_levels),
                    last_touch=max(l.last_touch for l in similar_levels),
                    volume_at_level=total_volume,
                    confidence=combined_confidence,
                    algorithm='merged',
                    metadata={
                        'merged_from': [l.algorithm for l in similar_levels],
                        'merge_count': len(similar_levels)
                    }
                )
                merged_levels.append(merged_level)
            else:
                merged_levels.append(level)
            
            # Mark indices as used
            used_indices.update(similar_indices)
        
        # Filter by strength threshold
        filtered_levels = [
            level for level in merged_levels 
            if level.strength >= self.config.strength_threshold
        ]
        
        # Sort by strength (descending)
        filtered_levels.sort(key=lambda x: x.strength, reverse=True)
        
        # Limit to max levels
        if len(filtered_levels) > self.config.max_levels:
            filtered_levels = filtered_levels[:self.config.max_levels]
        
        self.logger.info(f"🔗 Merged {len(all_levels)} levels into {len(filtered_levels)} final levels")
        return filtered_levels
    
    async def _calculate_detection_metrics(
        self,
        levels: List[SRLevel],
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate detection metrics."""
        if not levels:
            return {
                'total_levels': 0,
                'support_levels': 0,
                'resistance_levels': 0,
                'avg_strength': 0.0,
                'avg_confidence': 0.0,
                'algorithm_distribution': {}
            }
        
        support_levels = [l for l in levels if l.level_type == 'support']
        resistance_levels = [l for l in levels if l.level_type == 'resistance']
        
        # Calculate algorithm distribution
        algorithm_dist = {}
        for level in levels:
            algorithm = level.algorithm
            algorithm_dist[algorithm] = algorithm_dist.get(algorithm, 0) + 1
        
        metrics = {
            'total_levels': len(levels),
            'support_levels': len(support_levels),
            'resistance_levels': len(resistance_levels),
            'avg_strength': np.mean([l.strength for l in levels]),
            'avg_confidence': np.mean([l.confidence for l in levels]),
            'avg_touches': np.mean([l.touches for l in levels]),
            'algorithm_distribution': algorithm_dist,
            'strength_distribution': {
                'min': min(l.strength for l in levels),
                'max': max(l.strength for l in levels),
                'std': np.std([l.strength for l in levels])
            }
        }
        
        return metrics

# Convenience function
async def detect_sr_levels(
    data_dir: str,
    symbol: str,
    exchange: str,
    timeframe: str,
    config: Optional[SRDetectionConfig] = None
) -> SRDetectionResult:
    """Convenience function to detect SR levels."""
    pipeline = SRDetectionPipeline(config)
    return await pipeline.detect_sr_levels(data_dir, symbol, exchange, timeframe)