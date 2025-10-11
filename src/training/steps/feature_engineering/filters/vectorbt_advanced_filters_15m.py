"""
VectorBT Advanced Filters 15m - Enhanced Filter System

This module provides an enhanced unified interface for applying advanced filters
to 15-minute timeframe data using VectorBT for superior performance and analysis.

Features:
- VectorBT-optimized filter calculations
- Advanced pattern recognition and classification
- Multi-dimensional filtering with VectorBT indicators
- Parameter optimization and adaptive thresholds
- Comprehensive performance monitoring and validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
import warnings

# Import VectorBT base classes
from src.training.steps.feature_engineering.vectorbt_base import (
    VectorBTFeatureGenerator, VectorBTConfig, VectorBTTechnicalIndicators
)
from src.training.steps.feature_engineering.vectorbt_indicators_suite import (
    VectorBTIndicatorSuite, VectorBTIndicatorSuiteConfig
)

# Import VectorBT feature generators
from src.training.steps.feature_engineering.volatility.vectorbt_atr_volatility_ratio import (
    VectorBTATRVolatilityRatioGenerator
)
from src.training.steps.feature_engineering.trend.vectorbt_trend_coherence import (
    VectorBTTrendCoherenceGenerator
)
from src.training.steps.feature_engineering.price_action.vectorbt_bar_efficiency_ratio import (
    VectorBTBarEfficiencyRatioGenerator
)
from src.training.steps.feature_engineering.price_action.vectorbt_close_location_value import (
    VectorBTCloseLocationValueGenerator
)

# Import existing utilities
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.common_operations import (
    safe_divide, safe_log, safe_sqrt, safe_mean, safe_std,
    validate_finite, validate_positive, validate_range, safe_correlation,
    safe_dataframe_operation, validate_dataframe_columns, get_dataframe_info,
    create_data_quality_report, optimize_memory, memory_checkpoint
)
from src.utils.matrix_operations import (
    get_unified_matrix_operations, get_vectorized_processing_core,
    get_enhanced_matrix_operations, optimize_dataframe,
    vectorized_rolling_features, matrix_correlation_analysis,
    safe_correlation_matrix, compute_trading_indicators,
    get_hardware_performance_report
)


class VectorBTFilterType(Enum):
    """Enumeration of VectorBT filter types."""
    EFFICIENCY_RATIO = "efficiency_ratio"
    CLV = "clv"
    ATR_RATIO = "atr_ratio"
    TREND_COHERENCE = "trend_coherence"
    TECHNICAL_INDICATORS = "technical_indicators"
    PATTERN_RECOGNITION = "pattern_recognition"
    VOLUME_ANALYSIS = "volume_analysis"
    COMBINED = "combined"


@dataclass
class VectorBTAdvancedFiltersConfig:
    """Enhanced configuration for VectorBT advanced 15m timeframe filters."""
    
    # Global enable/disable
    enabled: bool = True
    
    # Individual filter enable/disable
    enable_efficiency_ratio: bool = True
    enable_clv: bool = True
    enable_atr_ratio: bool = True
    enable_trend_coherence: bool = True
    enable_technical_indicators: bool = True
    enable_pattern_recognition: bool = True
    enable_volume_analysis: bool = True
    
    # VectorBT specific settings
    enable_vectorbt_optimization: bool = True
    enable_parameter_optimization: bool = True
    optimization_runs: int = 100
    enable_caching: bool = True
    
    # Grading system (primary filtering method)
    use_grading_system: bool = True
    grade_threshold: float = 0.2
    grade_weights: Dict[str, float] = field(default_factory=lambda: {
        'efficiency': 0.15,
        'clv': 0.15,
        'atr_ratio': 0.15,
        'trend_coherence': 0.15,
        'technical_indicators': 0.20,
        'pattern_recognition': 0.10,
        'volume_analysis': 0.10
    })
    
    # Advanced filtering settings
    enable_adaptive_thresholds: bool = True
    enable_multi_dimensional_filtering: bool = True
    enable_regime_detection: bool = True
    enable_performance_monitoring: bool = True
    
    # Quality checks
    min_eligible_samples: int = 50
    max_filter_failure_rate: float = 0.7
    enable_cross_validation: bool = True
    cv_folds: int = 5
    
    # Performance settings
    enable_parallel_processing: bool = True
    n_jobs: int = -1
    chunk_size: int = 1000
    memory_efficient: bool = True


@dataclass
class VectorBTFilterResult:
    """Enhanced result container for VectorBT advanced filtering."""
    
    # Core results
    eligibility_mask: pd.Series
    eligibility_ratio: float
    
    # VectorBT grading system results
    average_grade: Optional[pd.Series] = None
    individual_grades: Dict[str, pd.Series] = field(default_factory=dict)
    vectorbt_indicators: Dict[str, pd.Series] = field(default_factory=dict)
    
    # Advanced filtering results
    pattern_recognition_results: Dict[str, Any] = field(default_factory=dict)
    regime_detection_results: Dict[str, Any] = field(default_factory=dict)
    volume_analysis_results: Dict[str, Any] = field(default_factory=dict)
    
    # Statistics
    n_total_samples: int = 0
    n_eligible_samples: int = 0
    n_filtered_samples: int = 0
    
    # Filter performance
    filter_effectiveness: Dict[str, float] = field(default_factory=dict)
    filter_statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    vectorbt_performance_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Quality metrics
    overall_quality_score: float = 0.0
    noise_reduction_ratio: float = 0.0
    vectorbt_optimization_score: float = 0.0
    
    # Metadata
    config_used: VectorBTAdvancedFiltersConfig = None
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class VectorBTAdvancedFilters15m:
    """
    VectorBT-Enhanced Advanced Filters for 15m Timeframe.
    
    This class provides a comprehensive filtering system using VectorBT
    for superior performance, advanced pattern recognition, and multi-dimensional analysis.
    """
    
    def __init__(self, config: Optional[VectorBTAdvancedFiltersConfig] = None):
        """Initialize VectorBT advanced filters for 15m timeframe."""
        self.config = config or VectorBTAdvancedFiltersConfig()
        self.logger = logging.getLogger('VectorBTAdvancedFilters15m')
        
        # Initialize VectorBT components
        self.vectorbt_config = VectorBTConfig(
            enable_optimization=self.config.enable_vectorbt_optimization,
            optimization_runs=self.config.optimization_runs,
            enable_caching=self.config.enable_caching,
            enable_parallel=self.config.enable_parallel_processing,
            n_jobs=self.config.n_jobs
        )
        
        self.indicators = VectorBTTechnicalIndicators(self.vectorbt_config)
        self.indicator_suite = VectorBTIndicatorSuite()
        
        # Initialize matrix operations for enhanced data processing
        self.matrix_ops = get_unified_matrix_operations()
        self.vectorized_core = get_vectorized_processing_core()
        self.enhanced_matrix_ops = get_enhanced_matrix_operations()
        
        # Initialize VectorBT feature generators
        self._initialize_vectorbt_generators()
        
        # Initialize performance monitoring
        self.performance_metrics = {}
        self.optimization_results = {}
        
        tprint_info("🔍 VectorBT Advanced Filters for 15m Timeframe initialized")
        tprint_info(f"   → VectorBT optimization: {self.config.enable_vectorbt_optimization}")
        tprint_info(f"   → Parameter optimization: {self.config.enable_parameter_optimization}")
        tprint_info(f"   → Multi-dimensional filtering: {self.config.enable_multi_dimensional_filtering}")
        tprint_info(f"   → Regime detection: {self.config.enable_regime_detection}")
        tprint_info(f"   → Performance monitoring: {self.config.enable_performance_monitoring}")
    
    def _initialize_vectorbt_generators(self) -> None:
        """Initialize VectorBT feature generators."""
        try:
            # Initialize VectorBT generators
            self.efficiency_generator = VectorBTBarEfficiencyRatioGenerator(
                lookback=3,
                enable_optimization=self.config.enable_vectorbt_optimization,
                enable_caching=self.config.enable_caching
            )
            
            self.clv_generator = VectorBTCloseLocationValueGenerator(
                lookback=8,
                enable_optimization=self.config.enable_vectorbt_optimization,
                enable_caching=self.config.enable_caching
            )
            
            self.atr_generator = VectorBTATRVolatilityRatioGenerator(
                lookback=4,
                enable_optimization=self.config.enable_vectorbt_optimization,
                enable_caching=self.config.enable_caching
            )
            
            self.trend_generator = VectorBTTrendCoherenceGenerator(
                lookback=8,
                enable_optimization=self.config.enable_vectorbt_optimization,
                enable_caching=self.config.enable_caching
            )
            
            tprint_info("✅ VectorBT generators initialized")
            
        except Exception as e:
            tprint_error(f"❌ Error initializing VectorBT generators: {e}")
            raise
    
    def apply_filters(self, data: pd.DataFrame) -> VectorBTFilterResult:
        """
        Apply VectorBT-enhanced advanced filters to 15m timeframe data.
        
        Args:
            data: OHLCV data with 15m timeframe
            
        Returns:
            VectorBTFilterResult with eligibility mask and comprehensive statistics
        """
        start_time = datetime.now()
        tprint_info("🔍 Applying VectorBT advanced 15m filters")
        
        # Validate input data
        self._validate_input_data(data)
        
        # Optimize data using matrix operations
        tprint_info("🧮 Optimizing data with VectorBT and matrix operations")
        original_shape = data.shape
        optimized_data = optimize_dataframe(data)
        if optimized_data is not data:
            data = optimized_data
            tprint_success(f"✅ Data optimized: {original_shape} → {data.shape}")
        
        # Initialize result container
        result = VectorBTFilterResult(
            eligibility_mask=pd.Series(True, index=data.index),
            eligibility_ratio=1.0,
            n_total_samples=len(data),
            config_used=self.config
        )
        
        try:
            # Use memory optimization context
            with memory_checkpoint("vectorbt_advanced_filters_15m"):
                if self.config.use_grading_system:
                    # Use VectorBT-enhanced grading system
                    tprint_info("📊 Using VectorBT-enhanced grading system for filter evaluation")
                    grades = {}
                    vectorbt_indicators = {}
                    
                    # Calculate VectorBT feature grades
                    if self.config.enable_efficiency_ratio:
                        tprint_info("📊 Calculating VectorBT efficiency ratio grade")
                        efficiency_features = self.efficiency_generator.generate_vectorbt_features(data)
                        if 'bar_efficiency_grade' in efficiency_features:
                            grades['efficiency'] = efficiency_features['bar_efficiency_grade']
                        vectorbt_indicators.update(efficiency_features)
                    
                    if self.config.enable_clv:
                        tprint_info("📊 Calculating VectorBT CLV grade")
                        clv_features = self.clv_generator.generate_vectorbt_features(data)
                        if 'clv_grade' in clv_features:
                            grades['clv'] = clv_features['clv_grade']
                        vectorbt_indicators.update(clv_features)
                    
                    if self.config.enable_atr_ratio:
                        tprint_info("📊 Calculating VectorBT ATR ratio grade")
                        atr_features = self.atr_generator.generate_vectorbt_features(data)
                        if 'atr_grade' in atr_features:
                            grades['atr_ratio'] = atr_features['atr_grade']
                        vectorbt_indicators.update(atr_features)
                    
                    if self.config.enable_trend_coherence:
                        tprint_info("📊 Calculating VectorBT trend coherence grade")
                        trend_features = self.trend_generator.generate_vectorbt_features(data)
                        if 'trend_coherence_grade' in trend_features:
                            grades['trend_coherence'] = trend_features['trend_coherence_grade']
                        vectorbt_indicators.update(trend_features)
                    
                    # Calculate technical indicators grade
                    if self.config.enable_technical_indicators:
                        tprint_info("📊 Calculating VectorBT technical indicators grade")
                        tech_indicators = self._calculate_technical_indicators_grade(data)
                        if tech_indicators:
                            grades['technical_indicators'] = tech_indicators
                            vectorbt_indicators.update(self.indicators.get_all_indicators(data))
                    
                    # Calculate pattern recognition grade
                    if self.config.enable_pattern_recognition:
                        tprint_info("📊 Calculating VectorBT pattern recognition grade")
                        pattern_grade = self._calculate_pattern_recognition_grade(data)
                        if pattern_grade is not None:
                            grades['pattern_recognition'] = pattern_grade
                    
                    # Calculate volume analysis grade
                    if self.config.enable_volume_analysis:
                        tprint_info("📊 Calculating VectorBT volume analysis grade")
                        volume_grade = self._calculate_volume_analysis_grade(data)
                        if volume_grade is not None:
                            grades['volume_analysis'] = volume_grade
                    
                    # Calculate weighted average grade
                    if grades:
                        weights = self.config.grade_weights
                        weighted_grades = []
                        for filter_name, grade in grades.items():
                            weight = weights.get(filter_name, 0.0)
                            weighted_grades.append(grade * weight)
                        
                        average_grade = pd.concat(weighted_grades, axis=1).sum(axis=1)
                        result.eligibility_mask = average_grade >= self.config.grade_threshold
                        result.average_grade = average_grade
                        result.individual_grades = grades
                        result.vectorbt_indicators = vectorbt_indicators
                        
                        tprint_info(f"   → Average grade: mean={average_grade.mean():.3f}, std={average_grade.std():.3f}")
                        tprint_info(f"   → Grade threshold: {self.config.grade_threshold}")
                    else:
                        result.eligibility_mask = pd.Series(True, index=data.index)
                        result.average_grade = pd.Series(1.0, index=data.index)
                        result.individual_grades = {}
                
                # Apply advanced filtering if enabled
                if self.config.enable_multi_dimensional_filtering:
                    result = self._apply_multi_dimensional_filtering(data, result)
                
                # Apply regime detection if enabled
                if self.config.enable_regime_detection:
                    result = self._apply_regime_detection(data, result)
                
                # Calculate final eligibility
                result.eligibility_ratio = result.eligibility_mask.mean()
                result.n_eligible_samples = result.eligibility_mask.sum()
                result.n_filtered_samples = result.n_total_samples - result.n_eligible_samples
                
                # Calculate quality metrics
                result.overall_quality_score = self._calculate_overall_quality_score(result)
                result.noise_reduction_ratio = result.n_filtered_samples / result.n_total_samples
                result.vectorbt_optimization_score = self._calculate_vectorbt_optimization_score(result)
                
                # Calculate performance metrics if enabled
                if self.config.enable_performance_monitoring:
                    result.vectorbt_performance_metrics = self._calculate_vectorbt_performance_metrics(data, result)
                
                # Log memory usage and performance
                memory_info = optimize_memory()
                data_info = get_dataframe_info(data)
                hardware_report = get_hardware_performance_report()
                tprint_info(f"📊 Data info: {data_info['shape']} shape, {data_info.get('memory_usage', 'N/A')} memory")
                tprint_info(f"🔧 Hardware performance: {hardware_report.get('cpu_cores', 'N/A')} cores, GPU: {hardware_report.get('gpu_available', 'N/A')}")
                
                result.processing_time = (datetime.now() - start_time).total_seconds()
                
                tprint_success(f"✅ VectorBT advanced filters applied: {result.n_eligible_samples}/{result.n_total_samples} samples eligible ({result.eligibility_ratio:.1%})")
                
                return result
            
        except Exception as e:
            tprint_error(f"❌ Error applying VectorBT advanced filters: {e}")
            raise
    
    def _calculate_technical_indicators_grade(self, data: pd.DataFrame) -> Optional[pd.Series]:
        """Calculate technical indicators grade using VectorBT."""
        try:
            # Get comprehensive technical indicators
            all_indicators = self.indicators.get_all_indicators(data)
            
            # Calculate composite grade from key indicators
            grade_components = []
            
            # RSI grade
            if 'rsi_14' in all_indicators:
                rsi = all_indicators['rsi_14']
                rsi_grade = 1.0 - np.abs(rsi - 50) / 50  # Higher grade for RSI closer to 50
                grade_components.append(rsi_grade)
            
            # MACD grade
            if 'macd' in all_indicators and 'macd_signal' in all_indicators:
                macd = all_indicators['macd']
                macd_signal = all_indicators['macd_signal']
                macd_grade = 1.0 - np.abs(macd - macd_signal) / (macd.abs().rolling(20).mean() + 1e-8)
                grade_components.append(macd_grade)
            
            # Bollinger Bands grade
            if 'bb_position_20' in all_indicators:
                bb_position = all_indicators['bb_position_20']
                bb_grade = 1.0 - np.abs(bb_position - 0.5) * 2  # Higher grade for position closer to middle
                grade_components.append(bb_grade)
            
            # ATR grade
            if 'atr_14' in all_indicators:
                atr = all_indicators['atr_14']
                atr_grade = 1.0 - np.clip(atr / atr.rolling(50).mean() - 1, 0, 1)  # Higher grade for moderate ATR
                grade_components.append(atr_grade)
            
            if grade_components:
                # Average the grade components
                technical_grade = pd.concat(grade_components, axis=1).mean(axis=1)
                return technical_grade
            else:
                return None
                
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating technical indicators grade: {e}")
            return None
    
    def _calculate_pattern_recognition_grade(self, data: pd.DataFrame) -> Optional[pd.Series]:
        """Calculate pattern recognition grade using VectorBT."""
        try:
            # Get pattern indicators
            pattern_indicators = self.indicators.get_pattern_indicators(data)
            
            # Calculate pattern strength grade
            if 'pattern_strength' in pattern_indicators:
                pattern_strength = pattern_indicators['pattern_strength']
                # Normalize pattern strength to 0-1 grade
                pattern_grade = np.clip(pattern_strength / pattern_strength.rolling(50).max(), 0, 1)
                return pattern_grade
            else:
                return None
                
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating pattern recognition grade: {e}")
            return None
    
    def _calculate_volume_analysis_grade(self, data: pd.DataFrame) -> Optional[pd.Series]:
        """Calculate volume analysis grade using VectorBT."""
        try:
            if 'volume' not in data.columns:
                return None
            
            # Get volume indicators
            volume_indicators = self.indicators.get_volume_indicators(data)
            
            # Calculate volume grade
            grade_components = []
            
            # Volume ratio grade
            if 'volume_ratio_20' in volume_indicators:
                volume_ratio = volume_indicators['volume_ratio_20']
                volume_grade = 1.0 - np.clip(np.abs(volume_ratio - 1), 0, 1)  # Higher grade for volume close to average
                grade_components.append(volume_grade)
            
            # VWAP deviation grade
            if 'vwap_deviation' in volume_indicators:
                vwap_deviation = volume_indicators['vwap_deviation']
                vwap_grade = 1.0 - np.clip(np.abs(vwap_deviation), 0, 1)  # Higher grade for price close to VWAP
                grade_components.append(vwap_grade)
            
            if grade_components:
                volume_grade = pd.concat(grade_components, axis=1).mean(axis=1)
                return volume_grade
            else:
                return None
                
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating volume analysis grade: {e}")
            return None
    
    def _apply_multi_dimensional_filtering(self, data: pd.DataFrame, result: VectorBTFilterResult) -> VectorBTFilterResult:
        """Apply multi-dimensional filtering using VectorBT indicators."""
        try:
            tprint_info("🔍 Applying multi-dimensional filtering")
            
            # Get comprehensive indicators
            all_indicators = self.indicators.get_all_indicators(data)
            
            # Create multi-dimensional filter
            multi_dim_mask = pd.Series(True, index=data.index)
            
            # RSI filter
            if 'rsi_14' in all_indicators:
                rsi = all_indicators['rsi_14']
                rsi_mask = (rsi > 20) & (rsi < 80)  # Avoid extreme RSI values
                multi_dim_mask = multi_dim_mask & rsi_mask
            
            # MACD filter
            if 'macd' in all_indicators and 'macd_signal' in all_indicators:
                macd = all_indicators['macd']
                macd_signal = all_indicators['macd_signal']
                macd_mask = np.abs(macd - macd_signal) < macd.abs().rolling(20).std() * 2
                multi_dim_mask = multi_dim_mask & macd_mask
            
            # Bollinger Bands filter
            if 'bb_position_20' in all_indicators:
                bb_position = all_indicators['bb_position_20']
                bb_mask = (bb_position > 0.1) & (bb_position < 0.9)  # Avoid extreme BB positions
                multi_dim_mask = multi_dim_mask & bb_mask
            
            # ATR filter
            if 'atr_14' in all_indicators:
                atr = all_indicators['atr_14']
                atr_mask = atr < atr.rolling(50).quantile(0.8)  # Avoid extremely high volatility
                multi_dim_mask = multi_dim_mask & atr_mask
            
            # Apply multi-dimensional filter
            result.eligibility_mask = result.eligibility_mask & multi_dim_mask
            
            tprint_info(f"   → Multi-dimensional filtering: {multi_dim_mask.sum()}/{len(data)} samples passed")
            
        except Exception as e:
            tprint_warning(f"⚠️ Error applying multi-dimensional filtering: {e}")
        
        return result
    
    def _apply_regime_detection(self, data: pd.DataFrame, result: VectorBTFilterResult) -> VectorBTFilterResult:
        """Apply regime detection filtering."""
        try:
            tprint_info("🔍 Applying regime detection filtering")
            
            # Detect market regimes using VectorBT indicators
            regime_mask = pd.Series(True, index=data.index)
            
            # Get trend indicators
            trend_indicators = self.indicators.get_trend_indicators(data)
            
            # ADX regime detection
            if 'adx' in trend_indicators:
                adx = trend_indicators['adx']
                # Filter out low trend strength periods
                adx_mask = adx > 20  # Minimum ADX threshold
                regime_mask = regime_mask & adx_mask
            
            # Volatility regime detection
            volatility_indicators = self.indicators.get_volatility_indicators(data)
            if 'atr_14' in volatility_indicators:
                atr = volatility_indicators['atr_14']
                # Filter out extreme volatility periods
                atr_percentile = atr.rolling(100).rank(pct=True)
                volatility_mask = (atr_percentile > 0.1) & (atr_percentile < 0.9)
                regime_mask = regime_mask & volatility_mask
            
            # Apply regime filter
            result.eligibility_mask = result.eligibility_mask & regime_mask
            
            # Store regime detection results
            result.regime_detection_results = {
                'adx_regime': trend_indicators.get('adx'),
                'volatility_regime': volatility_indicators.get('atr_14'),
                'regime_mask': regime_mask
            }
            
            tprint_info(f"   → Regime detection: {regime_mask.sum()}/{len(data)} samples passed")
            
        except Exception as e:
            tprint_warning(f"⚠️ Error applying regime detection: {e}")
        
        return result
    
    def _calculate_vectorbt_optimization_score(self, result: VectorBTFilterResult) -> float:
        """Calculate VectorBT optimization score."""
        try:
            if not result.vectorbt_indicators:
                return 0.0
            
            # Calculate score based on VectorBT indicator quality
            scores = []
            
            for name, series in result.vectorbt_indicators.items():
                if isinstance(series, pd.Series) and pd.api.types.is_numeric_dtype(series):
                    # Calculate stability score
                    stability = 1.0 / (1.0 + series.std())
                    scores.append(stability)
            
            if scores:
                return np.mean(scores)
            else:
                return 0.0
                
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating VectorBT optimization score: {e}")
            return 0.0
    
    def _calculate_vectorbt_performance_metrics(self, data: pd.DataFrame, result: VectorBTFilterResult) -> Dict[str, Any]:
        """Calculate VectorBT performance metrics."""
        try:
            metrics = {}
            
            # Calculate indicator performance
            if result.vectorbt_indicators:
                numeric_indicators = {
                    name: series for name, series in result.vectorbt_indicators.items()
                    if isinstance(series, pd.Series) and pd.api.types.is_numeric_dtype(series)
                }
                
                if len(numeric_indicators) > 1:
                    # Calculate correlation matrix
                    indicator_df = pd.DataFrame(numeric_indicators)
                    correlation_matrix = indicator_df.corr()
                    metrics['correlation_matrix'] = correlation_matrix.to_dict()
                
                # Calculate stability scores
                stability_scores = {}
                for name, series in numeric_indicators.items():
                    if len(series.dropna()) > 1:
                        stability_scores[name] = 1.0 / (1.0 + series.std())
                
                metrics['stability_scores'] = stability_scores
            
            # Calculate filter performance
            metrics['filter_performance'] = {
                'eligibility_ratio': result.eligibility_ratio,
                'noise_reduction_ratio': result.noise_reduction_ratio,
                'overall_quality_score': result.overall_quality_score,
                'vectorbt_optimization_score': result.vectorbt_optimization_score
            }
            
            return metrics
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating VectorBT performance metrics: {e}")
            return {}
    
    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """Validate input data format and requirements."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        if not validate_dataframe_columns(data, required_columns):
            missing_columns = set(required_columns) - set(data.columns)
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        min_required = max(20, 50)
        if len(data) < min_required:
            raise ValueError(f"Insufficient data: need at least {min_required} samples")
        
        # Check for valid OHLCV data
        for col in ['open', 'high', 'low', 'close']:
            if not pd.api.types.is_numeric_dtype(data[col]):
                tprint_warning(f"⚠️ Converting {col} to numeric")
                data = safe_dataframe_operation(data, pd.to_numeric, col, errors='coerce')
        
        # Validate OHLC relationships
        try:
            high_low_valid = (data['high'] >= data['low']).all()
            high_open_valid = (data['high'] >= data['open']).all()
            high_close_valid = (data['high'] >= data['close']).all()
            low_open_valid = (data['low'] <= data['open']).all()
            low_close_valid = (data['low'] <= data['close']).all()
            
            if not all([high_low_valid, high_open_valid, high_close_valid, low_open_valid, low_close_valid]):
                tprint_warning("⚠️ Found invalid OHLC relationships - data may need cleaning")
        except Exception as e:
            tprint_warning(f"⚠️ Error validating OHLC relationships: {e}")
    
    def _calculate_overall_quality_score(self, result: VectorBTFilterResult) -> float:
        """Calculate overall quality score based on filter results."""
        if result.n_total_samples == 0:
            return 0.0
        
        # Base score from eligibility ratio
        eligibility_score = result.eligibility_ratio
        
        # Bonus for good noise reduction (but not too much)
        noise_reduction_score = min(result.noise_reduction_ratio, 0.8)
        
        # VectorBT optimization bonus
        vectorbt_bonus = result.vectorbt_optimization_score * 0.2
        
        # Combine scores
        overall_score = (eligibility_score * 0.5) + (noise_reduction_score * 0.3) + (vectorbt_bonus * 0.2)
        
        return min(overall_score, 1.0)
    
    def optimize_parameters(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Optimize filter parameters using VectorBT."""
        if not self.config.enable_parameter_optimization:
            return {}
        
        try:
            tprint_info("🔍 Optimizing VectorBT filter parameters")
            
            optimization_results = {}
            
            # Optimize individual generators
            generators = {
                'efficiency': self.efficiency_generator,
                'clv': self.clv_generator,
                'atr': self.atr_generator,
                'trend': self.trend_generator
            }
            
            for name, generator in generators.items():
                try:
                    optimized_params = generator.optimize_parameters(data)
                    optimization_results[name] = optimized_params
                except Exception as e:
                    tprint_warning(f"⚠️ Error optimizing {name} generator: {e}")
            
            # Store optimization results
            self.optimization_results = optimization_results
            
            tprint_success(f"✅ Optimized parameters for {len(optimization_results)} generators")
            return optimization_results
            
        except Exception as e:
            tprint_error(f"❌ Error optimizing parameters: {e}")
            return {}
    
    def cleanup(self) -> None:
        """Clean up resources and optimize memory."""
        try:
            # Optimize memory usage
            memory_info = optimize_memory()
            if memory_info.get('success', False):
                tprint_info(f"🧠 Memory optimized: {memory_info.get('objects_collected', 0)} objects collected")
            
            # Clear performance metrics
            self.performance_metrics.clear()
            self.optimization_results.clear()
            
            tprint_success("✅ VectorBTAdvancedFilters15m cleanup completed")
        except Exception as e:
            tprint_warning(f"⚠️ Error during cleanup: {e}")


# Convenience function for external usage
def apply_vectorbt_advanced_filters_15m(
    data: pd.DataFrame,
    config: Optional[VectorBTAdvancedFiltersConfig] = None,
    **kwargs
) -> VectorBTFilterResult:
    """
    Apply VectorBT-enhanced advanced filters to 15m timeframe data.
    
    Args:
        data: OHLCV data with 15m timeframe
        config: Optional configuration
        **kwargs: Additional parameters
        
    Returns:
        VectorBTFilterResult with eligibility mask and comprehensive statistics
    """
    tprint_info("🚀 Starting VectorBT advanced filters 15m application")
    
    try:
        filter_system = VectorBTAdvancedFilters15m(config)
        result = filter_system.apply_filters(data, **kwargs)
        
        # Cleanup resources
        filter_system.cleanup()
        
        tprint_success("✅ VectorBT advanced filters 15m application completed")
        return result
        
    except Exception as e:
        tprint_error(f"❌ Error in VectorBT advanced filters 15m application: {e}")
        raise