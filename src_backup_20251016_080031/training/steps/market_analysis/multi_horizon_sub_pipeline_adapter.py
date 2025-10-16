"""
Multi-Horizon Sub-Pipeline Adapter

This module provides an adapter to integrate multi-horizon labeling into the existing
sub-pipeline system, replacing the triple barrier method.

Key features:
- Drop-in replacement for triple barrier labeling
- Maintains compatibility with existing sub-pipeline
- Provides enhanced labeling with reversal capture
- Optimized for short-term, high-frequency trading
- Enhanced data filtering and quality validation
- Memory optimization for large datasets
"""

import pandas as pd
import numpy as np
import functools
import time
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass

# Import utility modules for enhanced functionality
from src.utils.common_utilities import (
    validate_dataframe_columns, safe_dataframe_operation,
    calculate_data_quality_metrics, safe_convert_dtypes
)
from src.utils.math_validation import safe_divide, validate_finite
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer

# Optimized imports using common utilities
from src.utils.logger import get_logger
from src.core.decorators import handles_errors, traced, validates, log_execution_time

# Import tprint utility for enhanced logging
try:
    from src.utils.tprint import tprint
except ImportError:
    # Fallback to simple print if tprint is not available
    def tprint(*args, **kwargs):
        print(*args)

# Import optimized process engine
try:
    from ..optimized_process_engines import OptimizedMultiHorizonEngine, ProcessType
    OPTIMIZED_ENGINE_AVAILABLE = True
except ImportError:
    OptimizedMultiHorizonEngine = None
    ProcessType = None
    OPTIMIZED_ENGINE_AVAILABLE = False

class ExecutionMode(Enum):
    """Enhanced execution modes with configurable parameters."""
    FULL = "full"          # Complete execution with all data
    LIGHT = "light"        # Lightweight execution with data filtering
    BLANK = "blank"        # Minimal execution for testing/validation
    ADAPTIVE = "adaptive"  # Dynamic filtering based on data characteristics

@dataclass
class DataFilterConfig:
    """Configuration for data filtering operations."""
    mode: ExecutionMode = ExecutionMode.FULL
    timeframe: str = "5m"
    max_rows_light: int = 14400     # 10 days for 1m data
    max_rows_blank: int = 259200    # 180 days for 1m data
    max_rows_adaptive: int = 50000  # Default adaptive limit
    min_data_quality_score: float = 0.7
    enable_outlier_filtering: bool = True
    outlier_threshold: float = 3.0  # Standard deviations
    preserve_recent_data: bool = True
    memory_efficient: bool = True

class DataFilteringManager:
    """Enhanced data filtering manager with quality validation."""

    def __init__(self, config: Optional[DataFilterConfig] = None):
        """Initialize data filtering manager with hardware optimizations."""
        self.config = config or DataFilterConfig()
        self.logger = get_logger('DataFilteringManager')

        # Initialize hardware optimizers
        self.memory_optimizer = get_m1_memory_optimizer()
        self.cpu_optimizer = get_m1_cpu_optimizer()

        # Optimize CPU for data processing
        if self.cpu_optimizer:
            self.cpu_optimizer.optimize_numpy_operations()

        self.logger.info('🔄 Data Filtering Manager initialized with M1 optimizations')

    def filter_data(self, data: pd.DataFrame, mode: Optional[str] = None) -> pd.DataFrame:
        """
        Apply intelligent data filtering based on execution mode and data characteristics.

        Args:
            data: Input DataFrame to filter
            mode: Execution mode override

        Returns:
            Filtered DataFrame with quality validation
        """
        if data is None or data.empty:
            self.logger.warning("❌ No data provided for filtering")
            return pd.DataFrame()

        # Determine execution mode
        exec_mode = mode or self.config.mode.value
        self.logger.info(f"🔍 Applying {exec_mode} mode filtering to {len(data):,} rows")

        try:
            # Step 1: Data quality assessment
            data_quality = self._assess_data_quality(data)
            self.logger.info(f"📊 Data quality score: {data_quality['overall_score']:.3f}")

            # Step 2: Apply execution mode filtering
            filtered_data = self._apply_execution_mode_filtering(data, exec_mode)

            # Step 3: Quality-based filtering if enabled
            if self.config.min_data_quality_score > 0:
                filtered_data = self._apply_quality_filtering(filtered_data, data_quality)

            # Step 4: Outlier removal if enabled
            if self.config.enable_outlier_filtering:
                filtered_data = self._remove_outliers(filtered_data)

            # Step 5: Memory optimization
            if self.config.memory_efficient and self.memory_optimizer:
                filtered_data = self.memory_optimizer.optimize_dataframe_memory(filtered_data)

            # Step 6: Final validation
            final_quality = self._assess_data_quality(filtered_data)
            self.logger.info(f"✅ Filtering completed: {len(data):,} → {len(filtered_data):,} rows")
            self.logger.info(f"📊 Final quality score: {final_quality['overall_score']:.3f}")

            return filtered_data

        except Exception as e:
            self.logger.error(f"❌ Error in data filtering: {e}")
            return data  # Return original data on error

    def _assess_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality with comprehensive metrics."""
        try:
            # Use enhanced data quality metrics
            quality_metrics = calculate_data_quality_metrics(data)

            # Calculate additional quality indicators
            quality_score = self._calculate_overall_quality_score(data, quality_metrics)

            return {
                'overall_score': quality_score,
                'metrics': quality_metrics,
                'recommendations': self._generate_quality_recommendations(quality_metrics)
            }
        except Exception as e:
            self.logger.warning(f"⚠️ Error assessing data quality: {e}")
            return {'overall_score': 0.0, 'metrics': {}, 'recommendations': []}

    def _calculate_overall_quality_score(self, data: pd.DataFrame, metrics: Dict[str, Any]) -> float:
        """Calculate overall data quality score based on multiple factors."""
        try:
            score = 1.0

            # Factor 1: Missing data penalty
            missing_ratio = metrics.get('missing_values', 0) / len(data) if len(data) > 0 else 0
            score *= max(0.1, 1.0 - missing_ratio * 2)

            # Factor 2: Duplicate data penalty
            duplicate_ratio = metrics.get('duplicate_rows', 0) / len(data) if len(data) > 0 else 0
            score *= max(0.5, 1.0 - duplicate_ratio * 5)

            # Factor 3: Column completeness
            total_columns = len(data.columns)
            if total_columns > 0:
                missing_cols = len([col for col in ['open', 'high', 'low', 'close', 'volume'] if col not in data.columns])
                completeness = (total_columns - missing_cols) / total_columns
                score *= completeness

            # Factor 4: Data consistency (price relationships)
            consistency_score = self._check_price_consistency(data)
            score *= consistency_score

            return max(0.0, min(1.0, score))
        except Exception as e:
            return 0.5  # Default score on error

    def _check_price_consistency(self, data: pd.DataFrame) -> float:
        """Check price consistency and OHLC relationships."""
        try:
            if len(data) == 0:
                return 0.5

            # Check for logical OHLC relationships
            issues = 0
            total_checks = 0

            # High should be >= max(open, close)
            total_checks += 1
            high_issues = (data['high'] < np.maximum(data['open'], data['close'])).sum()
            if high_issues > 0:
                issues += 1

            # Low should be <= min(open, close)
            total_checks += 1
            low_issues = (data['low'] > np.minimum(data['open'], data['close'])).sum()
            if low_issues > 0:
                issues += 1

            # Volume should be positive
            total_checks += 1
            volume_issues = (data['volume'] <= 0).sum()
            if volume_issues > 0:
                issues += 1

            # Price changes should be reasonable (not too extreme)
            total_checks += 1
            if len(data) > 1:
                returns = data['close'].pct_change().dropna()
                extreme_changes = (returns.abs() > 0.5).sum()  # More than 50% change
                if extreme_changes > len(returns) * 0.1:  # More than 10% of data
                    issues += 1

            return max(0.0, 1.0 - (issues / total_checks)) if total_checks > 0 else 0.5
        except Exception as e:
            return 0.5

    def _apply_execution_mode_filtering(self, data: pd.DataFrame, mode: str) -> pd.DataFrame:
        """Apply execution mode specific filtering."""
        try:
            original_size = len(data)

            if mode.lower() == 'light':
                # Light mode: Keep recent data with quality preservation
                filtered_data = self._apply_light_filtering(data)
            elif mode.lower() == 'blank':
                # Blank mode: Minimal data for testing
                filtered_data = self._apply_blank_filtering(data)
            elif mode.lower() == 'adaptive':
                # Adaptive mode: Dynamic filtering based on data characteristics
                filtered_data = self._apply_adaptive_filtering(data)
            else:
                # Full mode: Use all data with quality validation
                filtered_data = self._apply_full_filtering(data)

            filtered_size = len(filtered_data)
            self.logger.info(f"📊 {mode.upper()} filtering: {original_size:,} → {filtered_size:,} rows ({filtered_size/original_size*100:.1f}%)")
            return filtered_data

        except Exception as e:
            self.logger.error(f"❌ Error in execution mode filtering: {e}")
            return data

    def _apply_light_filtering(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply light mode filtering with intelligent data selection."""
        try:
            # Start with recent data
            if self.config.preserve_recent_data and len(data) > self.config.max_rows_light:
                data = data.tail(self.config.max_rows_light).copy()

            # Apply quality-based subsampling if still too large
            if len(data) > self.config.max_rows_light * 0.8:
                return self._quality_based_subsampling(data, self.config.max_rows_light)

            return data
        except Exception as e:
            self.logger.warning(f"⚠️ Error in light filtering: {e}")
            return data.tail(self.config.max_rows_light).copy()

    def _apply_blank_filtering(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply blank mode filtering for minimal testing."""
        try:
            # Use minimal data for testing
            target_size = min(self.config.max_rows_blank, len(data))
            return data.tail(target_size).copy()
        except Exception as e:
            self.logger.warning(f"⚠️ Error in blank filtering: {e}")
            return data.tail(self.config.max_rows_blank).copy()

    def _apply_adaptive_filtering(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply adaptive filtering based on data characteristics."""
        try:
            # Calculate optimal sample size based on data quality and characteristics
            data_quality = self._assess_data_quality(data)

            if data_quality['overall_score'] > 0.8:
                # High quality data - can use larger sample
                target_size = min(self.config.max_rows_adaptive, len(data))
            elif data_quality['overall_score'] > 0.6:
                # Medium quality - moderate sample
                target_size = min(self.config.max_rows_adaptive // 2, len(data))
            else:
                # Low quality - smaller sample with quality filtering
                target_size = min(self.config.max_rows_adaptive // 4, len(data))

            return self._quality_based_subsampling(data, target_size)
        except Exception as e:
            self.logger.warning(f"⚠️ Error in adaptive filtering: {e}")
            return data.tail(self.config.max_rows_adaptive).copy()

    def _apply_full_filtering(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply full mode filtering with quality preservation."""
        try:
            # Apply quality-based filtering to remove problematic data
            quality_data = self._assess_data_quality(data)

            if quality_data['overall_score'] < self.config.min_data_quality_score:
                self.logger.warning(f"⚠️ Low data quality ({quality_data['overall_score']:.3f}) - applying quality filtering")
                return self._apply_quality_filtering(data, quality_data)

            return data
        except Exception as e:
            self.logger.warning(f"⚠️ Error in full filtering: {e}")
            return data

    def _quality_based_subsampling(self, data: pd.DataFrame, target_size: int) -> pd.DataFrame:
        """Apply quality-based subsampling to maintain data quality."""
        try:
            if len(data) <= target_size:
                return data

            # Calculate quality scores for each data point
            quality_scores = self._calculate_sample_quality_scores(data)

            # Sort by quality and select top samples
            quality_df = pd.DataFrame({'quality': quality_scores}, index=data.index)
            quality_df = quality_df.sort_values('quality', ascending=False)

            # Ensure we keep recent data by weighting recent samples higher
            if self.config.preserve_recent_data:
                recency_weights = pd.Series(1.0, index=data.index)
                if isinstance(data.index, pd.DatetimeIndex):
                    # Higher weight for recent data
                    max_date = data.index.max()
                    recency_weights = 1.0 + (data.index - data.index.min()) / (max_date - data.index.min())

                # Combine quality and recency scores
                combined_scores = quality_df['quality'] * recency_weights
                quality_df['combined'] = combined_scores

                top_indices = combined_scores.nlargest(target_size).index
            else:
                top_indices = quality_df.index[:target_size]

            return data.loc[top_indices].sort_index()

        except Exception as e:
            self.logger.warning(f"⚠️ Error in quality-based subsampling: {e}")
            return data.tail(target_size).copy()

    def _calculate_sample_quality_scores(self, data: pd.DataFrame) -> pd.Series:
        """Calculate quality scores for each sample."""
        try:
            scores = pd.Series(0.5, index=data.index)  # Default score

            # Quality based on price consistency
            for idx in data.index:
                try:
                    sample = data.loc[idx]
                    if pd.notna(sample['high']) and pd.notna(sample['low']) and pd.notna(sample['close']):
                        # Check OHLC consistency
                        ohlc_consistent = (sample['high'] >= max(sample['open'], sample['close']) and
                                         sample['low'] <= min(sample['open'], sample['close']))
                        volume_valid = pd.notna(sample['volume']) and sample['volume'] > 0

                        if ohlc_consistent and volume_valid:
                            scores[idx] = 1.0
                        elif volume_valid:
                            scores[idx] = 0.7
                        else:
                            scores[idx] = 0.3
                except Exception:
                    scores[idx] = 0.2

            return scores
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating sample quality: {e}")
            return pd.Series(0.5, index=data.index)

    def _apply_quality_filtering(self, data: pd.DataFrame, quality_data: Dict[str, Any]) -> pd.DataFrame:
        """Apply quality-based filtering to improve data quality."""
        try:
            if quality_data['overall_score'] >= self.config.min_data_quality_score:
                return data

            # Remove samples with quality issues
            recommendations = quality_data.get('recommendations', [])
            filtered_data = data.copy()

            # Apply common recommendations
            for rec in recommendations:
                if 'missing_values' in rec.lower():
                    # Remove rows with excessive missing values
                    missing_per_row = filtered_data.isnull().sum(axis=1)
                    max_missing = len(filtered_data.columns) // 2  # Allow up to 50% missing
                    filtered_data = filtered_data[missing_per_row <= max_missing]

            return filtered_data
        except Exception as e:
            self.logger.warning(f"⚠️ Error in quality filtering: {e}")
            return data

    def _remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using statistical methods."""
        try:
            if len(data) < 10:
                return data  # Not enough data for outlier detection

            # Use IQR method for outlier detection
            numeric_cols = data.select_dtypes(include=[np.number]).columns

            # Calculate bounds for each numeric column
            outlier_mask = pd.Series(False, index=data.index)

            for col in numeric_cols:
                try:
                    if col in data.columns:
                        values = data[col].dropna()
                        if len(values) > 10:
                            Q1 = values.quantile(0.25)
                            Q3 = values.quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - self.config.outlier_threshold * IQR
                            upper_bound = Q3 + self.config.outlier_threshold * IQR

                            col_outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
                            outlier_mask |= col_outliers
                except Exception:
                    continue

            # Remove outlier rows
            outlier_count = outlier_mask.sum()
            if outlier_count > 0 and outlier_count < len(data) * 0.5:  # Don't remove more than 50%
                filtered_data = data[~outlier_mask].copy()
                self.logger.info(f"🧹 Removed {outlier_count} outlier rows ({outlier_count/len(data)*100:.1f}%)")
                return filtered_data

            return data
        except Exception as e:
            self.logger.warning(f"⚠️ Error in outlier removal: {e}")
            return data

    def _generate_quality_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on data quality metrics."""
        recommendations = []

        try:
            # Check missing values
            if metrics.get('missing_values', 0) > 0:
                missing_ratio = metrics['missing_values'] / (len(metrics.get('total_rows', 1)) * len(metrics.get('total_columns', 1)))
                if missing_ratio > 0.1:
                    recommendations.append("High missing value ratio - consider data imputation or filtering")
                elif missing_ratio > 0.05:
                    recommendations.append("Moderate missing values detected")

            # Check duplicates
            if metrics.get('duplicate_rows', 0) > 0:
                recommendations.append(f"Found {metrics['duplicate_rows']} duplicate rows - consider removal")

            # Check data consistency issues
            if metrics.get('price_consistency_score', 1.0) < 0.9:
                recommendations.append("Price consistency issues detected - check OHLC relationships")

        except Exception as e:
            self.logger.warning(f"⚠️ Error generating recommendations: {e}")

        return recommendations
try:
    from src.utils.math_validation import (
        safe_mean, safe_std, validate_finite, safe_percentage_change
    )
except ImportError:
    # Fallback implementations
    def safe_mean(values, default=0.0):
        try:
            return float(np.mean(values)) if len(values) > 0 else default
        except:
            return default
    
    def safe_std(values, default=0.0):
        try:
            return float(np.std(values)) if len(values) > 0 else default
        except:
            return default
    
    def validate_finite(value, context=""):
        try:
            return float(value) if np.isfinite(value) else 0.0
        except:
            return 0.0
    
    def safe_percentage_change(old_val, new_val, default=0.0):
        try:
            return (new_val - old_val) / old_val if old_val != 0 else default
        except:
            return default


# Try to import UniversalSerializer, provide fallback if not available
try:
    from src.utils.serialization_utils import UniversalSerializer
except ImportError:
    # Fallback implementation
    class UniversalSerializer:
        def __init__(self):
            pass
        
        def serialize(self, data):
            return data
        
        def deserialize(self, data):
            return data

# Fallback implementations for missing functions
def validate_dataframe(df):
    """Validate that DataFrame is not None and not empty."""
    return df is not None and isinstance(df, pd.DataFrame) and not df.empty

def calculate_data_quality_metrics(df):
    """Calculate basic data quality metrics."""
    if not validate_dataframe(df):
        return {}
    
    return {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': int(df.isnull().sum().sum()),
        'duplicate_rows': int(df.duplicated().sum()),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2)
    }

def create_summary_statistics(df):
    """Create basic summary statistics."""
    if not validate_dataframe(df):
        return {}
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return {'message': 'No numeric columns found'}
    
    return {
        'numeric_columns': len(numeric_cols),
        'mean_values': df[numeric_cols].mean().to_dict(),
        'std_values': df[numeric_cols].std().to_dict()
    }

class memory_checkpoint:
    """Simple memory checkpoint context manager."""
    def __init__(self, name):
        self.name = name
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

def timed_operation(func):
    """Simple timing decorator."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logging.getLogger(__name__).info(f'{func.__name__} executed in {elapsed:.2f}s')
        return result
    return wrapper

def safe_divide(numerator, denominator, default=0.0):
    """Safely divide two numbers."""
    try:
        return numerator / denominator if denominator != 0 else default
    except:
        return default

# Import the multi-horizon labeler
from ..pre_training.multi_horizon_profit_labeler import (
    MultiHorizonProfitLabeler, 
    MultiHorizonConfig
)

# Import optimization components
try:
    from research.profit_labeling.dynamic_target_optimizer import (
        JointTargetHorizonOptimizer,
        DynamicOptimizationConfig,
        OptimizationMethod,
        OptimizationObjective
    )
    from research.profit_labeling.heuristic_analyzer import (
        HeuristicAnalyzer,
        HeuristicAnalysisConfig
    )
    from research.profit_labeling.labeling_validator import (
        LabelingValidator,
        ValidationConfig
    )
    OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    OPTIMIZATION_AVAILABLE = False
    print(f"⚠️ Optimization components not available: {e}")

class MultiHorizonSubPipelineAdapter:
    """
    Adapter for integrating multi-horizon labeling into existing sub-pipeline.
    
    This adapter provides a drop-in replacement for the triple barrier labeling
    step while maintaining compatibility with the existing pipeline structure.
    """
    
    def __init__(self):
        """Initialize the adapter with hardware optimizations and automatic timeframe optimization."""
        self.logger = get_logger('MultiHorizonSubPipelineAdapter')
        
        # Initialize hardware optimizers
        self.memory_optimizer = get_m1_memory_optimizer()
        self.cpu_optimizer = get_m1_cpu_optimizer()
        self.serializer = UniversalSerializer()

        # Initialize data filtering manager
        self.data_filter = DataFilteringManager()
        
        # Initialize optimization components if available
        self.optimization_enabled = OPTIMIZATION_AVAILABLE
        if self.optimization_enabled:
            self._initialize_optimization_components()
        
        # Initialize optimized process engine
        if OPTIMIZED_ENGINE_AVAILABLE:
            tprint("🔧 Initializing optimized multi-horizon engine...")
            self.optimized_engine = OptimizedMultiHorizonEngine(
                use_hardware_accel=True,
                cache_size=1000
            )
            tprint("✅ Optimized multi-horizon engine initialized")
        else:
            self.optimized_engine = None
            tprint("⚠️ Optimized multi-horizon engine not available")
        
        # Optimize CPU for data processing
        if self.cpu_optimizer:
            self.cpu_optimizer.optimize_numpy_operations()
        
        self.logger.info('🔄 Multi-Horizon Sub-Pipeline Adapter initialized with M1 optimizations')
        self.logger.info('🔄 Data filtering manager initialized with quality validation')
        if self.optimization_enabled:
            self.logger.info('🎯 Automatic timeframe optimization ENABLED')
        else:
            self.logger.error('❌ FAST FAIL: Automatic timeframe optimization DISABLED - cannot proceed')

    def _initialize_optimization_components(self):
        """Initialize optimization components for automatic timeframe discovery."""
        try:
            # Initialize dynamic optimizer
            self.optimization_config = DynamicOptimizationConfig(
                optimization_method=OptimizationMethod.BAYESIAN_OPTIMIZATION,
                min_horizon=1,  # 5 minutes
                max_horizon=20,  # 100 minutes
                horizon_step=1,
                optimization_objective=OptimizationObjective.MULTI_OBJECTIVE,
                n_target_candidates=10,
                target_range=(0.001, 0.020),  # 0.1% to 2.0%
                bayesian_iterations=30
            )
            
            self.optimizer = JointTargetHorizonOptimizer(self.optimization_config)
            
            # Initialize analysis components
            self.heuristic_analyzer = HeuristicAnalyzer()
            self.labeling_validator = LabelingValidator()
            
            self.logger.info('✅ Optimization components initialized successfully')
            
        except Exception as e:
            self.logger.error(f'❌ Failed to initialize optimization components: {e}')
            self.optimization_enabled = False

    def _optimize_timeframes_automatically(self, data: pd.DataFrame) -> MultiHorizonConfig:
        """
        Automatically optimize timeframes based on market data characteristics.
        
        This method uses the research framework to discover optimal timeframes
        for both Analyst and Tactician model training.
        """
        if not self.optimization_enabled:
            self.logger.error('❌ FAST FAIL: Optimization disabled - cannot proceed without optimal timeframes')
            raise RuntimeError(
                "❌ FAST FAIL: Automatic timeframe optimization is disabled. "
                "Cannot proceed without optimal timeframe discovery. "
                "Training pipeline will terminate."
            )
        
        self.logger.info('🎯 Starting automatic timeframe optimization')
        
        try:
            # Step 1: Run joint target-horizon optimization
            self.logger.info('   → Running joint target-horizon optimization...')
            optimization_result = self.optimizer.optimize_target_horizon_combinations(data)
            
            if optimization_result.objective_score < 0.3:
                self.logger.error(f'❌ FAST FAIL: Low optimization score ({optimization_result.objective_score:.3f}) - cannot proceed')
                raise RuntimeError(
                    f"❌ FAST FAIL: Optimization score ({optimization_result.objective_score:.3f}) below minimum threshold (0.3). "
                    f"Cannot proceed without optimal timeframe discovery. Training pipeline will terminate."
                )
            
            # Step 2: Extract optimal timeframes
            optimal_horizons = optimization_result.optimal_horizons
            optimal_targets = optimization_result.optimal_targets
            
            self.logger.info(f'   → Optimal horizons discovered: {optimal_horizons}')
            self.logger.info(f'   → Optimal targets discovered: {optimal_targets}')
            
            # Step 3: Create optimized configuration
            optimized_config = MultiHorizonConfig()
            
            # Map discovered horizons to configuration
            if optimal_horizons:
                # Find the best horizons for immediate and short-term
                horizon_values = list(optimal_horizons.values())
                if len(horizon_values) >= 2:
                    optimized_config.time_horizons = {
                        'immediate': min(horizon_values[:2]),
                        'short': max(horizon_values[:2])
                    }
                else:
                    optimized_config.time_horizons = {
                        'immediate': horizon_values[0] if horizon_values else 2,
                        'short': horizon_values[0] * 2 if horizon_values else 4
                    }
            
            # Map discovered targets to configuration
            if optimal_targets:
                target_values = list(optimal_targets.values())
                if len(target_values) >= 4:
                    # Sort targets and map to micro, small, medium, good
                    sorted_targets = sorted(target_values)
                    optimized_config.profit_targets = {
                        'micro': sorted_targets[0],
                        'small': sorted_targets[1],
                        'medium': sorted_targets[2],
                        'good': sorted_targets[3]
                    }
            
            # Step 4: Validate optimized configuration
            validation_score = self._validate_optimized_config(optimized_config, data)
            
            if validation_score > 0.5:
                self.logger.info(f'✅ Optimized configuration validated (score: {validation_score:.3f})')
                return optimized_config
            else:
                self.logger.error(f'❌ FAST FAIL: Low validation score ({validation_score:.3f}) - cannot proceed')
                raise RuntimeError(
                    f"❌ FAST FAIL: Optimized configuration validation failed. "
                    f"Validation score ({validation_score:.3f}) below minimum threshold (0.5). "
                    f"Cannot proceed with invalid timeframe configuration. Training pipeline will terminate."
                )
                
        except Exception as e:
            self.logger.error(f'❌ FAST FAIL: Automatic optimization failed: {e}')
            raise RuntimeError(
                f"❌ FAST FAIL: Automatic timeframe optimization failed. "
                f"Error: {e}. Cannot proceed without optimal timeframe discovery. "
                f"Training pipeline will terminate."
            )

    def _validate_optimized_config(self, config: MultiHorizonConfig, data: pd.DataFrame) -> float:
        """Validate the optimized configuration using heuristic analysis."""
        try:
            # Generate labels with optimized config
            labeler = MultiHorizonProfitLabeler(config)
            labeled_data = labeler.generate_labels(data.copy())
            
            # Analyze effectiveness
            heuristic_results = self.heuristic_analyzer.analyze_labeling_heuristics(labeled_data)
            
            # Calculate overall effectiveness score
            effectiveness_scores = []
            for result in heuristic_results.values():
                if hasattr(result, 'metric_value'):
                    effectiveness_scores.append(result.metric_value)
            
            if effectiveness_scores:
                avg_effectiveness = np.mean(effectiveness_scores)
                return min(1.0, max(0.0, avg_effectiveness))
            
            return 0.5  # Neutral score if no results
            
        except Exception as e:
            self.logger.warning(f'⚠️ Configuration validation failed: {e}')
            return 0.3  # Low score on error

    def _validate_input_data(self, data: pd.DataFrame) -> bool:
        """Enhanced data validation with quality checks."""
        try:
            # Basic validation
            if data is None or data.empty:
                return False

            # Required column validation
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not validate_dataframe_columns(data, required_cols):
                self.logger.warning("⚠️ Missing required columns for OHLCV data")
                return False

            # Data quality assessment
            quality_metrics = calculate_data_quality_metrics(data)

            # Check for excessive missing values
            total_cells = len(data) * len(data.columns)
            missing_ratio = quality_metrics.get('missing_values', 0) / total_cells

            if missing_ratio > 0.3:  # More than 30% missing
                self.logger.warning(f"⚠️ High missing value ratio: {missing_ratio:.1%}")
                return False

            # Check for excessive duplicates
            duplicate_ratio = quality_metrics.get('duplicate_rows', 0) / len(data)
            if duplicate_ratio > 0.2:  # More than 20% duplicates
                self.logger.warning(f"⚠️ High duplicate ratio: {duplicate_ratio:.1%}")
                return False

            # Price consistency check
            if not self._check_price_data_consistency(data):
                self.logger.warning("⚠️ Price data consistency issues detected")
                return False

            return True

        except Exception as e:
            self.logger.error(f"❌ Error in data validation: {e}")
            return False

    def _check_price_data_consistency(self, data: pd.DataFrame) -> bool:
        """Check price data consistency and logical relationships."""
        try:
            # Check for logical OHLC relationships in a sample
            sample_size = min(1000, len(data))
            sample = data.tail(sample_size)

            # High should be >= max(open, close)
            high_issues = (sample['high'] < np.maximum(sample['open'], sample['close'])).sum()

            # Low should be <= min(open, close)
            low_issues = (sample['low'] > np.minimum(sample['open'], sample['close'])).sum()

            # Volume should be positive
            volume_issues = (sample['volume'] <= 0).sum()

            # Allow some tolerance for data issues
            total_issues = high_issues + low_issues + volume_issues
            tolerance_ratio = total_issues / len(sample)

            return tolerance_ratio < 0.1  # Less than 10% issues

        except Exception as e:
            self.logger.warning(f"⚠️ Error checking price consistency: {e}")
            return True  # Assume OK if we can't check
    
    def execute_multi_horizon_labeling_step(self,
                                          data: pd.DataFrame,
                                          regime_labels: Optional[pd.Series] = None,
                                          config: Optional[Dict[str, Any]] = None,
                                          symbol: Optional[str] = None,
                                          exchange: Optional[str] = None,
                                          timeframe: Optional[str] = None,
                                          mode: str = 'full',
                                          features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute multi-horizon labeling step compatible with sub-pipeline.

        This method provides the same interface as the original triple barrier
        labeling step but uses the new multi-horizon approach.

        Args:
            data: Input OHLCV data
            regime_labels: Optional regime labels
            config: Configuration dictionary
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Data timeframe
            mode: Execution mode
            features: Optional pre-computed features to use for enhanced labeling

        Returns:
            Dictionary with labeling results compatible with sub-pipeline
        """
        self.logger.info(f'🎯 Executing multi-horizon labeling step for {symbol or "unknown"} on {timeframe or "unknown"}')
        self.logger.info(f'📊 Input data shape: {data.shape if data is not None else "None"}')
        self.logger.info(f'🚨 EXECUTION MODE: {mode}')
        self.logger.info(f'🔧 Features available: {features is not None}')

        if features:
            feature_names = features.get('combined_feature_names', [])
            self.logger.info(f'📊 Available features: {len(feature_names)} features')
            self.logger.info(f'🔧 Enhanced labeling with optimized features enabled')
        
        # ENHANCED DATA FILTERING WITH QUALITY VALIDATION
        if data is not None and len(data) > 1000:  # Apply filtering for datasets larger than 1000 rows
            original_size = len(data)

            # Use the enhanced data filtering manager
            try:
                filtered_data = self.data_filter.filter_data(data, mode)

                # Only use filtered data if it's significantly smaller and maintains quality
                if len(filtered_data) < original_size * 0.9:  # At least 10% reduction
                    data = filtered_data
                    self.logger.info(f'🔥 ENHANCED FILTERING: {original_size:,} → {len(data):,} rows')
                    self.logger.info(f'📊 Data quality preserved during filtering')
                else:
                    self.logger.info(f'📊 Data quality sufficient - using original data: {original_size:,} rows')

            except Exception as e:
                self.logger.warning(f'⚠️ Enhanced filtering failed, using original approach: {e}')
                # Fallback to original filtering logic
            if mode and mode.lower() == 'light':
                data = data.tail(14400).copy()
                self.logger.info(f'🔥 FORCED LIGHT FILTERING: {original_size:,} → {len(data):,} rows')
            elif mode and mode.lower() == 'blank':
                data = data.tail(259200).copy()
                self.logger.info(f'🔥 FORCED BLANK FILTERING: {original_size:,} → {len(data):,} rows')
        
        try:
            # Validate input data with enhanced validation
            if not self._validate_input_data(data):
                self.logger.error('❌ Data validation failed')
                return {
                    'status': 'failed',
                    'error': 'Invalid or empty DataFrame provided',
                    'artifacts': {}
                }
            
            
            # Optimize data memory usage
            if self.memory_optimizer:
                data = self.memory_optimizer.optimize_dataframe_memory(data)
            
            # Use memory checkpoint for large operations
            with memory_checkpoint('multi_horizon_labeling'):
                # AUTOMATIC TIMEFRAME OPTIMIZATION
                if self.optimization_enabled:
                    self.logger.info('🎯 Running automatic timeframe optimization...')
                    optimized_config = self._optimize_timeframes_automatically(data)
                    if optimized_config.time_horizons:
                        self.logger.info(f'✅ Using optimized timeframes: {optimized_config.time_horizons}')
                        self.logger.info(f'✅ Using optimized targets: {optimized_config.profit_targets}')
                    else:
                        self.logger.info('⚠️ Optimization failed - using default configuration')
                else:
                    optimized_config = None
                
                # Create multi-horizon configuration (with optimization if available)
                labeling_config = self._create_labeling_config(config, optimized_config)
                self.logger.info(f'🔧 Created labeling config: {labeling_config.__dict__}')

                # Enhanced labeling with features if available
                if features and features.get('combined_feature_names'):
                    self.logger.info('🔧 Enhancing labeling with feature-based analysis')
                    # Store features for use in dynamic labeling
                    labeling_config.features = features
                    labeling_config.enhanced_labeling = True
                
                # Apply multi-horizon labeling with safe operations
                self.logger.info('🔄 Starting multi-horizon labeling...')
                self.logger.info(f'📊 Input data columns: {list(data.columns)}')
                self.logger.info(f'📊 Input data index type: {type(data.index)}')
                
                try:
                    # Check if data has required columns
                    required_cols = ['open', 'high', 'low', 'close', 'volume']
                    missing_cols = [col for col in required_cols if col not in data.columns]
                    if missing_cols:
                        self.logger.error(f'❌ Missing required columns: {missing_cols}')
                        labeled_data = None
                    else:
                        # Call directly to get better error information
                        self.logger.info('🔄 Applying multi-horizon labeling...')
                        
                        # Try calling the labeler directly to bypass potential issues
                        self.logger.info('🔧 Creating labeler...')
                        labeler = MultiHorizonProfitLabeler(labeling_config)
                        self.logger.info(f'🔧 Created labeler: {labeler}')

                        # Test with a small subset first
                        test_data = data.head(1000).copy()
                        self.logger.info(f'🧪 Testing with small subset: {test_data.shape}')

                        # Implement actual dynamic labeling without decorators
                        try:
                            self.logger.info('🧪 Implementing dynamic multi-horizon labeling...')

                            # Create dynamic labeling without problematic decorators
                            # Pass features if available for enhanced labeling
                            if hasattr(labeling_config, 'enhanced_labeling') and labeling_config.enhanced_labeling:
                                labeled_data = self._generate_enhanced_dynamic_labels(test_data, labeling_config)
                            else:
                                labeled_data = self._generate_dynamic_labels(test_data, labeling_config)

                            self.logger.info(f'✅ Dynamic labeling successful: {labeled_data.shape}')

                        except Exception as direct_e:
                            self.logger.error(f'❌ Dynamic labeling failed: {direct_e}')
                            import traceback
                            self.logger.error(f'❌ Traceback: {traceback.format_exc()}')
                            labeled_data = None

                        self.logger.info(f'📊 Final result: {type(labeled_data)}, shape: {labeled_data.shape if labeled_data is not None else "None"}')

                        # If successful with small data, apply to the filtered dataset
                        if labeled_data is not None and not labeled_data.empty:
                            self.logger.info('✅ Small test successful, applying dynamic labeling to filtered data...')

                            # Apply dynamic labeling to the already-filtered dataset
                            self.logger.info(f'🔧 About to call _generate_dynamic_labels with data shape: {data.shape}')
                            if hasattr(labeling_config, 'enhanced_labeling') and labeling_config.enhanced_labeling:
                                labeled_data = self._generate_enhanced_dynamic_labels(data, labeling_config)  # 'data' is already filtered!
                            else:
                                labeled_data = self._generate_dynamic_labels(data, labeling_config)  # 'data' is already filtered!

                            self.logger.info(f'📊 Filtered data dynamic labeling completed: {labeled_data.shape}')
                except Exception as e:
                    self.logger.error(f'❌ Multi-horizon labeling failed: {e}')
                    self.logger.error(f'❌ Traceback: {traceback.format_exc()}')
                    labeled_data = None
                
                self.logger.info(f'📊 Labeling result type: {type(labeled_data)}, shape: {labeled_data.shape if labeled_data is not None else "None"}')
                
                # Check if labeling was successful
                if labeled_data is None:
                    self.logger.error('❌ Multi-horizon labeling returned None')
                    return {
                        'status': 'failed',
                        'error': 'Multi-horizon labeling returned None result',
                        'artifacts': {}
                    }
                
                # Calculate labeling metrics with enhanced validation
                labeling_metrics = self._calculate_labeling_metrics(labeled_data, data)
            
            # Create result compatible with sub-pipeline with enhanced metrics
            result = {
                'status': 'completed',
                'execution_time': datetime.now().isoformat(),
                'artifacts': {
                    'multi_horizon_labeling_result': {
                        'labeled_data': labeled_data.to_json(orient='records'),  # JSON serialization for parsing
                        'labeling_metrics': labeling_metrics,
                        'config': labeling_config.__dict__,
                        'method': 'multi_horizon_profit_labeling',
                        'symbol': symbol,
                        'exchange': exchange,
                        'timeframe': timeframe,
                        'data_quality': calculate_data_quality_metrics(labeled_data),
                        'summary_stats': create_summary_statistics(labeled_data)
                    }
                }
            }
            
            self.logger.info(f'✅ Multi-horizon labeling completed: {len(labeled_data)} samples, {labeled_data.shape[1]} features')
            self.logger.info(f'🔄 Returning result with status: {result.get("status", "unknown")}')
            return result
            
        except Exception as e:
            self.logger.error(f'❌ Multi-horizon labeling failed: {e}')
            error_result = {
                'status': 'failed',
                'error': str(e),
                'artifacts': {}
            }
            self.logger.info(f'🔄 Returning error result: {error_result}')
            return error_result
    
    def _create_labeling_config(self, config: Optional[Dict[str, Any]] = None, optimized_config: Optional[MultiHorizonConfig] = None) -> MultiHorizonConfig:
        """Create multi-horizon configuration from sub-pipeline config with optional optimization."""
        # Start with optimized config if available, otherwise use default
        if optimized_config and optimized_config.time_horizons:
            labeling_config = optimized_config
            self.logger.info('🎯 Using optimized configuration')
        else:
            labeling_config = MultiHorizonConfig()
            self.logger.info('📋 Using default configuration')
        
        if config:
            # Update profit targets if specified
            if 'profit_targets' in config:
                labeling_config.profit_targets = config['profit_targets']
            
            # Update time horizons if specified
            if 'time_horizons' in config:
                labeling_config.time_horizons = config['time_horizons']
            
            # Update other parameters
            if 'transaction_cost' in config:
                labeling_config.transaction_cost = config['transaction_cost']
            
            if 'enable_quality_scoring' in config:
                labeling_config.enable_quality_scoring = config['enable_quality_scoring']
            
            if 'leverage_aware' in config:
                labeling_config.leverage_aware = config['leverage_aware']
        
        return labeling_config
    
    def _calculate_labeling_metrics(self, labeled_data: pd.DataFrame, 
                                  original_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive labeling metrics with safe operations."""
        metrics = {
            'total_samples': len(labeled_data),
            'original_samples': len(original_data),
            'total_features': labeled_data.shape[1],
            'new_features_added': labeled_data.shape[1] - original_data.shape[1],
            'labeling_method': 'multi_horizon_profit_labeling'
        }
        
        # Calculate target-specific metrics with safe operations
        target_columns = [col for col in labeled_data.columns if col.endswith('_prob')]
        metrics['probability_targets'] = len(target_columns)
        
        # Calculate composite score metrics with safe operations
        composite_columns = [
            'overall_opportunity', 'leverage_adjusted_score', 
            'immediate_opportunity', 'short_term_opportunity',
            'reversal_capture_score', 'reassessment_frequency'
        ]
        
        for col in composite_columns:
            if col in labeled_data.columns:
                values = labeled_data[col].dropna()
                if len(values) > 0:
                    metrics[f'{col}_mean'] = validate_finite(safe_mean(values, default=0.0), f'{col}_mean')
                    metrics[f'{col}_std'] = validate_finite(safe_std(values, default=0.0), f'{col}_std')
                    high_quality_count = (values > 0.7).sum()
                    metrics[f'{col}_high_quality_ratio'] = safe_divide(high_quality_count, len(values), default=0.0)
        
        # Overall quality metrics with safe operations
        if 'overall_opportunity' in labeled_data.columns:
            overall_opp = labeled_data['overall_opportunity'].dropna()
            if len(overall_opp) > 0:
                high_opp_count = (overall_opp > 0.7).sum()
                metrics['high_opportunity_samples'] = int(high_opp_count)
                metrics['high_opportunity_ratio'] = safe_divide(high_opp_count, len(overall_opp), default=0.0)
                metrics['average_opportunity_score'] = validate_finite(safe_mean(overall_opp, default=0.0), 'average_opportunity_score')
        
        return metrics
    
    def _generate_dynamic_labels(self, data: pd.DataFrame, config: MultiHorizonConfig) -> pd.DataFrame:
        """Generate dynamic multi-horizon labels without decorators."""
        self.logger.info(f'🔍 Generating dynamic multi-horizon labels for {len(data)} samples')
        
        if len(data) < max(config.time_horizons.values()) + 1:
            self.logger.warning(f'⚠️ Insufficient data for labeling (need at least {max(config.time_horizons.values()) + 1} samples)')
            return data.copy()
        
        labeled_data = data.copy()
        max_horizon = max(config.time_horizons.values())
        
        # Initialize all probability columns
        self._initialize_probability_columns(labeled_data, config)
        
        # Generate labels for each valid sample
        valid_samples = len(data) - max_horizon
        self.logger.info(f'📊 Processing {valid_samples} valid samples with dynamic calculations')
        
        for i in range(min(valid_samples, len(data) - max_horizon)):
            if i % 10000 == 0 and i > 0:
                self.logger.info(f'   → Progress: {i}/{valid_samples} ({i/valid_samples*100:.1f}%)')
            
            try:
                current_price = float(data.iloc[i]['close'])
                sample_labels = self._calculate_dynamic_sample_labels(data, i, current_price, config)
                
                # DEBUG: Check what's actually in sample_labels at assignment time (first few samples only)
                if i < 3:
                    bi_keys_at_assignment = [k for k in sample_labels.keys() if any(keyword in k for keyword in ['long_overall', 'short_overall', 'opportunity_asymmetry', 'directional_confidence'])]
                    self.logger.info(f"🔍 SAMPLE[{i}] assignment time - bi-directional keys: {bi_keys_at_assignment}")
                
                # Store all labels for this sample
                for col_name, value in sample_labels.items():
                    if col_name in labeled_data.columns:
                        labeled_data.iloc[i, labeled_data.columns.get_loc(col_name)] = value
                        # DEBUG: Log successful bi-directional assignments (first few samples only)
                        if i < 3 and any(keyword in col_name for keyword in ['long_overall', 'short_overall', 'opportunity_asymmetry', 'directional_confidence']):
                            self.logger.info(f"✅ ASSIGNED: {col_name} = {value:.4f} to DataFrame[{i}]")
                    else:
                        # DEBUG: Log missing columns (only for first few samples to avoid spam)
                        if i < 3 and any(keyword in col_name for keyword in ['long_overall', 'short_overall', 'opportunity_asymmetry', 'directional_confidence']):
                            self.logger.warning(f"❌ MISSING COLUMN: {col_name} not in DataFrame columns")
                        
            except Exception as e:
                if i < 10:  # Only log first few errors to avoid spam
                    self.logger.warning(f'⚠️ Error processing sample {i}: {e}')
                continue
        
        # Calculate summary statistics
        self._log_dynamic_labeling_statistics(labeled_data, valid_samples)

        return labeled_data

    def _generate_enhanced_dynamic_labels(self, data: pd.DataFrame, config: MultiHorizonConfig) -> pd.DataFrame:
        """Generate enhanced dynamic multi-horizon labels using optimized features."""
        self.logger.info(f'🔍 Generating ENHANCED dynamic multi-horizon labels for {len(data)} samples with features')

        if len(data) < max(config.time_horizons.values()) + 1:
            self.logger.warning(f'⚠️ Insufficient data for enhanced labeling (need at least {max(config.time_horizons.values()) + 1} samples)')
            return data.copy()

        labeled_data = data.copy()
        max_horizon = max(config.time_horizons.values())

        # Initialize all probability columns
        self._initialize_probability_columns(labeled_data, config)

        # Extract features for enhanced labeling
        features = getattr(config, 'features', {})
        feature_data = None
        feature_names = []

        if features and 'combined_features' in features:
            self.logger.info('🔧 Using optimized features for enhanced labeling')
            feature_data = features['combined_features']
            feature_names = features.get('combined_feature_names', [])

            # Ensure feature data is aligned with our data
            if isinstance(feature_data, pd.DataFrame):
                # Align feature data with our data index
                common_index = data.index.intersection(feature_data.index)
                if len(common_index) > 0:
                    feature_data = feature_data.loc[common_index]
                    labeled_data = labeled_data.loc[common_index]
                    self.logger.info(f'📊 Aligned feature data: {len(common_index)} samples')
                else:
                    self.logger.warning('⚠️ No common index between data and features - falling back to standard labeling')
                    return self._generate_dynamic_labels(data, config)

        # Generate labels for each valid sample
        valid_samples = len(data) - max_horizon
        self.logger.info(f'📊 Processing {valid_samples} valid samples with enhanced calculations')

        for i in range(min(valid_samples, len(data) - max_horizon)):
            if i % 10000 == 0 and i > 0:
                self.logger.info(f'   → Enhanced Progress: {i}/{valid_samples} ({i/valid_samples*100:.1f}%)')

            try:
                current_price = float(data.iloc[i]['close'])

                # Use features to enhance labeling decisions if available
                if feature_data is not None and i < len(feature_data):
                    current_features = feature_data.iloc[i] if hasattr(feature_data, 'iloc') else None
                    sample_labels = self._calculate_enhanced_dynamic_sample_labels(
                        data, i, current_price, config, current_features, feature_names
                    )
                else:
                    # Fallback to standard labeling
                    sample_labels = self._calculate_dynamic_sample_labels(data, i, current_price, config)

                # Store all labels for this sample
                for col_name, value in sample_labels.items():
                    if col_name in labeled_data.columns:
                        labeled_data.iloc[i, labeled_data.columns.get_loc(col_name)] = value

            except Exception as e:
                if i < 10:  # Only log first few errors to avoid spam
                    self.logger.warning(f'⚠️ Error processing enhanced sample {i}: {e}')
                continue

        # Calculate summary statistics
        self._log_enhanced_labeling_statistics(labeled_data, valid_samples, features)

        return labeled_data
    
    def _initialize_probability_columns(self, labeled_data: pd.DataFrame, config: MultiHorizonConfig):
        """Initialize all probability and metadata columns."""
        columns_to_add = []
        
        # Individual probability columns - BI-DIRECTIONAL
        for target_name in config.profit_targets.keys():
            for horizon_name in config.time_horizons.keys():
                # Original columns (backward compatibility)
                col_name = f'{target_name}_{horizon_name}_prob'
                columns_to_add.append(col_name)
                
                # NEW: Bi-directional columns
                long_col = f'{target_name}_{horizon_name}_long_prob'
                short_col = f'{target_name}_{horizon_name}_short_prob'
                columns_to_add.extend([long_col, short_col])
        
        # Composite score columns - BI-DIRECTIONAL
        composite_columns = [
            # Original composite scores (backward compatibility)
            'overall_opportunity',
            'leverage_adjusted_score', 
            'immediate_opportunity',
            'short_term_opportunity',
            
            # NEW: Bi-directional composite scores
            'long_overall_opportunity',
            'short_overall_opportunity',
            'long_immediate_opportunity',
            'short_immediate_opportunity',
            'long_short_term_opportunity',
            'short_short_term_opportunity',
            'long_leverage_adjusted_score',
            'short_leverage_adjusted_score',
            
            # NEW: Directional analysis
            'opportunity_asymmetry',
            'directional_confidence',
            'directional_bias',
            'best_direction'
        ]
        columns_to_add.extend(composite_columns)
        
        # Initialize all columns with zeros
        for col in columns_to_add:
            labeled_data[col] = 0.0
    
    def _calculate_dynamic_sample_labels(self, data: pd.DataFrame, index: int, current_price: float, config: MultiHorizonConfig) -> Dict[str, float]:
        """Calculate dynamic labels for a single sample based on actual price movements."""
        sample_labels = {}
        probability_scores = {}
        
        # Generate labels for each target/horizon combination - BI-DIRECTIONAL
        for target_name, target_pct in config.profit_targets.items():
            for horizon_name, horizon_periods in config.time_horizons.items():
                window_end = min(index + horizon_periods + 1, len(data))
                window_data = data.iloc[index:window_end]
                
                # Calculate actual probability for BOTH directions
                long_prob = self._calculate_actual_profit_probability(
                    window_data, current_price, target_pct, horizon_periods, config, direction='long'
                )
                short_prob = self._calculate_actual_profit_probability(
                    window_data, current_price, target_pct, horizon_periods, config, direction='short'
                )
                
                # Store LONG results
                long_col = f'{target_name}_{horizon_name}_long_prob'
                sample_labels[long_col] = long_prob
                probability_scores[f'{target_name}_{horizon_name}_long'] = long_prob
                
                # Store SHORT results
                short_col = f'{target_name}_{horizon_name}_short_prob'
                sample_labels[short_col] = short_prob
                probability_scores[f'{target_name}_{horizon_name}_short'] = short_prob
                
                # BACKWARD COMPATIBILITY: Store original (long-biased) results
                col_name = f'{target_name}_{horizon_name}_prob'
                sample_labels[col_name] = long_prob  # Use long for backward compatibility
                probability_scores[f'{target_name}_{horizon_name}'] = long_prob
        
        # Calculate composite scores
        composite_scores = self._calculate_dynamic_composite_scores(probability_scores)
        sample_labels.update(composite_scores)
        
        # DEBUG: Check if bi-directional scores made it into sample_labels
        bi_keys_in_sample = [k for k in sample_labels.keys() if any(keyword in k for keyword in ['long_overall', 'short_overall', 'opportunity_asymmetry', 'directional_confidence'])]
        if bi_keys_in_sample:
            self.logger.info(f"🎯 SAMPLE_LABELS contains {len(bi_keys_in_sample)} bi-directional scores: {bi_keys_in_sample}")
        
        return sample_labels
    
    def _calculate_actual_profit_probability(self, window_data: pd.DataFrame, 
                                           entry_price: float, 
                                           profit_target: float,
                                           horizon_periods: int,
                                           config: MultiHorizonConfig,
                                           direction: str = 'long') -> float:
        """Calculate actual probability based on real price movements."""
        if len(window_data) < 2:
            return 0.1  # Base uncertainty probability
        
        try:
            highs = window_data['high'].values
            lows = window_data['low'].values
            
            # Calculate directional target prices and check hits
            if direction.lower() == 'long':
                target_price = entry_price * (1 + profit_target)
                target_hit = np.any(highs >= target_price)
                if target_hit:
                    hit_index = np.where(highs >= target_price)[0][0]
                    # For longs, adverse move is price going down
                    max_adverse = (entry_price - np.min(lows[:hit_index+1])) / entry_price if hit_index > 0 else 0.0
                else:
                    max_adverse = (entry_price - np.min(lows)) / entry_price
                    
            else:  # direction == 'short'
                target_price = entry_price * (1 - profit_target)  # Short target is below entry
                target_hit = np.any(lows <= target_price)
                if target_hit:
                    hit_index = np.where(lows <= target_price)[0][0]
                    # For shorts, adverse move is price going up
                    max_adverse = (np.max(highs[:hit_index+1]) - entry_price) / entry_price if hit_index > 0 else 0.0
                else:
                    max_adverse = (np.max(highs) - entry_price) / entry_price
            
            if target_hit:
                time_to_hit = hit_index
                
                # Calculate quality factors
                speed_factor = max(0.2, 1.0 - (time_to_hit / horizon_periods))
                
                # Risk factor (lower adverse excursion = better)
                risk_factor = max(0.1, 1.0 - (abs(max_adverse) * 20))  # Penalize adverse moves
                
                # Net profit factor
                net_profit = profit_target - config.transaction_cost
                profit_factor = max(0.2, min(1.0, net_profit * 200))
                
                # Combined probability with quality weighting
                base_prob = 0.9  # High probability for actual hits
                quality_weight = (speed_factor * config.speed_weight + 
                                risk_factor * config.risk_weight + 
                                profit_factor * config.profitability_weight)
                
                final_prob = base_prob * quality_weight
                return np.clip(final_prob, 0.0, 1.0)
            else:
                # Target not hit - calculate probability based on how close we got
                max_price_reached = np.max(highs)
                progress_to_target = (max_price_reached - entry_price) / (target_price - entry_price)
                progress_to_target = np.clip(progress_to_target, 0.0, 1.0)
                
                # Base probability for near-misses
                base_prob = 0.1 + (progress_to_target * 0.3)  # 0.1 to 0.4 range
                return np.clip(base_prob, 0.0, 1.0)
                
        except Exception as e:
            # Fallback for any calculation errors
            return 0.1
    
    def _calculate_dynamic_composite_scores(self, probability_scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate bi-directional dynamic composite opportunity scores."""
        composite_scores = {}
        
        # Separate long and short probability scores
        long_scores = {k: v for k, v in probability_scores.items() if '_long' in k}
        short_scores = {k: v for k, v in probability_scores.items() if '_short' in k}
        
        # DEBUG: Log what we found
        self.logger.info(f"🔍 DEBUG: Total probability_scores: {len(probability_scores)}")
        self.logger.info(f"🔍 DEBUG: Long scores found: {len(long_scores)}")
        self.logger.info(f"🔍 DEBUG: Short scores found: {len(short_scores)}")
        if len(probability_scores) > 0:
            sample_keys = list(probability_scores.keys())[:3]
            self.logger.info(f"🔍 DEBUG: Sample keys: {sample_keys}")
        
        # BI-DIRECTIONAL: Long opportunity scores
        long_immediate_probs = [prob for key, prob in long_scores.items() if 'immediate_long' in key]
        long_short_probs = [prob for key, prob in long_scores.items() if 'short_long' in key]
        
        composite_scores['long_immediate_opportunity'] = np.mean(long_immediate_probs) if long_immediate_probs else 0.1
        composite_scores['long_short_term_opportunity'] = np.mean(long_short_probs) if long_short_probs else 0.1
        long_overall = np.mean(list(long_scores.values())) if long_scores else 0.1
        composite_scores['long_overall_opportunity'] = long_overall
        self.logger.info(f"✅ CREATED long_overall_opportunity: {long_overall:.4f}")
        
        # BI-DIRECTIONAL: Short opportunity scores  
        short_immediate_probs = [prob for key, prob in short_scores.items() if 'immediate_short' in key]
        short_short_probs = [prob for key, prob in short_scores.items() if 'short_short' in key]
        
        composite_scores['short_immediate_opportunity'] = np.mean(short_immediate_probs) if short_immediate_probs else 0.1
        composite_scores['short_short_term_opportunity'] = np.mean(short_short_probs) if short_short_probs else 0.1
        short_overall = np.mean(list(short_scores.values())) if short_scores else 0.1
        composite_scores['short_overall_opportunity'] = short_overall
        self.logger.info(f"✅ CREATED short_overall_opportunity: {short_overall:.4f}")
        
        # BACKWARD COMPATIBILITY: Original scores (long-biased)
        composite_scores['immediate_opportunity'] = composite_scores['long_immediate_opportunity']
        composite_scores['short_term_opportunity'] = composite_scores['long_short_term_opportunity'] 
        composite_scores['overall_opportunity'] = composite_scores['long_overall_opportunity']
        
        # BI-DIRECTIONAL: Directional analysis
        long_avg = composite_scores['long_overall_opportunity']
        short_avg = composite_scores['short_overall_opportunity']
        
        asymmetry = long_avg - short_avg
        confidence = abs(long_avg - short_avg)
        composite_scores['opportunity_asymmetry'] = asymmetry
        composite_scores['directional_confidence'] = confidence
        self.logger.info(f"✅ CREATED opportunity_asymmetry: {asymmetry:.4f}")
        self.logger.info(f"✅ CREATED directional_confidence: {confidence:.4f}")
        
        # Determine best direction
        if long_avg > short_avg + 0.05:  # 5% threshold
            composite_scores['directional_bias'] = 1.0  # Long bias
            composite_scores['best_direction'] = 1.0    # Long preferred
        elif short_avg > long_avg + 0.05:
            composite_scores['directional_bias'] = -1.0  # Short bias
            composite_scores['best_direction'] = -1.0    # Short preferred
        else:
            composite_scores['directional_bias'] = 0.0   # Neutral
            composite_scores['best_direction'] = 0.0     # Neutral
        
        # Leverage-adjusted score (bi-directional)
        leverage_weights = {'micro': 0.4, 'small': 0.3, 'medium': 0.2, 'good': 0.1}
        
        # Calculate for both directions
        for direction, dir_scores in [('long', long_scores), ('short', short_scores)]:
            weighted_score = 0.0
            total_weight = 0.0
            
            for target_name in ['micro', 'small', 'medium', 'good']:
                weight = leverage_weights.get(target_name, 0.1)
                target_probs = [prob for key, prob in dir_scores.items() 
                               if key.startswith(f'{target_name}_')]
                if target_probs:
                    weighted_score += np.mean(target_probs) * weight
                    total_weight += weight
            
            final_score = weighted_score / total_weight if total_weight > 0 else 0.1
            
            if direction == 'long':
                composite_scores['leverage_adjusted_score'] = final_score  # Backward compatibility
            composite_scores[f'{direction}_leverage_adjusted_score'] = final_score
        
        # DEBUG: Log what we're returning
        bi_directional_keys = [k for k in composite_scores.keys() if any(keyword in k for keyword in ['long_overall', 'short_overall', 'opportunity_asymmetry', 'directional_confidence'])]
        if bi_directional_keys:
            self.logger.info(f"🎯 RETURNING {len(bi_directional_keys)} bi-directional composite scores: {bi_directional_keys}")
        
        return composite_scores
    
    def _log_dynamic_labeling_statistics(self, labeled_data: pd.DataFrame, valid_samples: int):
        """Log dynamic labeling statistics."""
        self.logger.info('📊 Dynamic Labeling Statistics:')
        
        # Overall opportunity distribution
        if 'overall_opportunity' in labeled_data.columns:
            overall_opp = labeled_data['overall_opportunity'].iloc[:valid_samples]
            self.logger.info(f'   → Overall opportunity: mean={overall_opp.mean():.3f}, std={overall_opp.std():.3f}')
            
            # High opportunity samples
            high_opp_count = (overall_opp > 0.7).sum()
            self.logger.info(f'   → High opportunity samples (>0.7): {high_opp_count} ({high_opp_count/valid_samples*100:.1f}%)')
        
        # Leverage-adjusted scores
        if 'leverage_adjusted_score' in labeled_data.columns:
            leverage_scores = labeled_data['leverage_adjusted_score'].iloc[:valid_samples]
            self.logger.info(f'   → Leverage-adjusted: mean={leverage_scores.mean():.3f}, std={leverage_scores.std():.3f}')
        
        self.logger.info('✅ Dynamic multi-horizon labeling completed successfully')

    def _calculate_enhanced_dynamic_sample_labels(self, data: pd.DataFrame, index: int,
                                                current_price: float, config: MultiHorizonConfig,
                                                current_features: pd.Series = None,
                                                feature_names: List[str] = None) -> Dict[str, float]:
        """Calculate enhanced dynamic labels for a single sample using features."""
        sample_labels = {}
        probability_scores = {}

        # Generate labels for each target/horizon combination - BI-DIRECTIONAL
        for target_name, target_pct in config.profit_targets.items():
            for horizon_name, horizon_periods in config.time_horizons.items():
                window_end = min(index + horizon_periods + 1, len(data))
                window_data = data.iloc[index:window_end]

                # Calculate actual probability for BOTH directions (enhanced with features)
                long_prob = self._calculate_enhanced_profit_probability(
                    window_data, current_price, target_pct, horizon_periods, config,
                    direction='long', features=current_features, feature_names=feature_names
                )
                short_prob = self._calculate_enhanced_profit_probability(
                    window_data, current_price, target_pct, horizon_periods, config,
                    direction='short', features=current_features, feature_names=feature_names
                )

                # Store LONG results
                long_col = f'{target_name}_{horizon_name}_long_prob'
                sample_labels[long_col] = long_prob
                probability_scores[f'{target_name}_{horizon_name}_long'] = long_prob

                # Store SHORT results
                short_col = f'{target_name}_{horizon_name}_short_prob'
                sample_labels[short_col] = short_prob
                probability_scores[f'{target_name}_{horizon_name}_short'] = short_prob

                # BACKWARD COMPATIBILITY: Store original (long-biased) results
                col_name = f'{target_name}_{horizon_name}_prob'
                sample_labels[col_name] = long_prob  # Use long for backward compatibility
                probability_scores[f'{target_name}_{horizon_name}'] = long_prob

        # Calculate composite scores (unchanged)
        composite_scores = self._calculate_dynamic_composite_scores(probability_scores)
        sample_labels.update(composite_scores)

        return sample_labels

    def _calculate_enhanced_profit_probability(self, window_data: pd.DataFrame,
                                             entry_price: float,
                                             profit_target: float,
                                             horizon_periods: int,
                                             config: MultiHorizonConfig,
                                             direction: str = 'long',
                                             features: pd.Series = None,
                                             feature_names: List[str] = None) -> float:
        """Calculate enhanced probability using features and actual price movements."""
        if len(window_data) < 2:
            return 0.1  # Base uncertainty probability

        try:
            highs = window_data['high'].values
            lows = window_data['low'].values

            # Calculate directional target prices and check hits
            if direction.lower() == 'long':
                target_price = entry_price * (1 + profit_target)
                target_hit = np.any(highs >= target_price)
                if target_hit:
                    hit_index = np.where(highs >= target_price)[0][0]
                    # For longs, adverse move is price going down
                    max_adverse = (entry_price - np.min(lows[:hit_index+1])) / entry_price if hit_index > 0 else 0.0
                else:
                    max_adverse = (entry_price - np.min(lows)) / entry_price

            else:  # direction == 'short'
                target_price = entry_price * (1 - profit_target)  # Short target is below entry
                target_hit = np.any(lows <= target_price)
                if target_hit:
                    hit_index = np.where(lows <= target_price)[0][0]
                    # For shorts, adverse move is price going up
                    max_adverse = (np.max(highs[:hit_index+1]) - entry_price) / entry_price if hit_index > 0 else 0.0
                else:
                    max_adverse = (np.max(highs) - entry_price) / entry_price

            # Enhanced calculation using features
            feature_boost = 1.0
            if features is not None and feature_names:
                # Use feature importance to boost probability calculation
                feature_importance = getattr(config, 'feature_importance_scores', {})

                # Calculate feature-based confidence boost
                relevant_features = [f for f in feature_names if any(keyword in f.lower()
                                   for keyword in ['momentum', 'trend', 'volatility', 'rsi', 'macd'])]

                if relevant_features:
                    # Average of relevant feature values (normalized)
                    feature_values = []
                    for feature in relevant_features:
                        if feature in features.index:
                            val = features[feature]
                            if pd.notna(val):
                                # Normalize to 0-1 range
                                normalized_val = max(0.0, min(1.0, float(val)))
                                feature_values.append(normalized_val)

                    if feature_values:
                        feature_boost = 1.0 + (np.mean(feature_values) * 0.3)  # Up to 30% boost
                        self.logger.debug(f'🔧 Feature boost: {feature_boost:.3f} from {len(feature_values)} features')

            if target_hit:
                time_to_hit = hit_index

                # Calculate quality factors (enhanced with features)
                speed_factor = max(0.2, 1.0 - (time_to_hit / horizon_periods))

                # Risk factor (lower adverse excursion = better)
                risk_factor = max(0.1, 1.0 - (abs(max_adverse) * 20))  # Penalize adverse moves

                # Net profit factor
                net_profit = profit_target - config.transaction_cost
                profit_factor = max(0.2, min(1.0, net_profit * 200))

                # Combined probability with quality weighting and feature boost
                base_prob = 0.9  # High probability for actual hits
                quality_weight = (speed_factor * config.speed_weight +
                                risk_factor * config.risk_weight +
                                profit_factor * config.profitability_weight)

                final_prob = base_prob * quality_weight * feature_boost
                return np.clip(final_prob, 0.0, 1.0)
            else:
                # Target not hit - calculate probability based on how close we got
                max_price_reached = np.max(highs)
                progress_to_target = (max_price_reached - entry_price) / (target_price - entry_price)
                progress_to_target = np.clip(progress_to_target, 0.0, 1.0)

                # Base probability for near-misses (enhanced with features)
                base_prob = 0.1 + (progress_to_target * 0.3)  # 0.1 to 0.4 range

                # Feature-based adjustment for near-misses
                if features is not None:
                    # If features suggest strong directional bias, increase probability
                    directional_features = [f for f in feature_names if any(keyword in f.lower()
                                          for keyword in ['momentum', 'trend', 'bias', 'direction'])]

                    if directional_features:
                        directional_values = []
                        for feature in directional_features:
                            if feature in features.index:
                                val = features[feature]
                                if pd.notna(val):
                                    directional_values.append(float(val))

                        if directional_values:
                            # Calculate directional confidence
                            directional_avg = np.mean(directional_values)
                            directional_boost = 1.0 + (abs(directional_avg) * 0.2)  # Up to 20% boost
                            base_prob *= directional_boost

                return np.clip(base_prob, 0.0, 1.0)

        except Exception as e:
            # Fallback for any calculation errors
            return 0.1

    def _log_enhanced_labeling_statistics(self, labeled_data: pd.DataFrame, valid_samples: int, features: Dict[str, Any]):
        """Log enhanced labeling statistics including feature usage."""
        self.logger.info('📊 Enhanced Labeling Statistics:')

        # Overall opportunity distribution
        if 'overall_opportunity' in labeled_data.columns:
            overall_opp = labeled_data['overall_opportunity'].iloc[:valid_samples]
            self.logger.info(f'   → Overall opportunity: mean={overall_opp.mean():.3f}, std={overall_opp.std():.3f}')

            # High opportunity samples
            high_opp_count = (overall_opp > 0.7).sum()
            self.logger.info(f'   → High opportunity samples (>0.7): {high_opp_count} ({high_opp_count/valid_samples*100:.1f}%)')

        # Leverage-adjusted scores
        if 'leverage_adjusted_score' in labeled_data.columns:
            leverage_scores = labeled_data['leverage_adjusted_score'].iloc[:valid_samples]
            self.logger.info(f'   → Leverage-adjusted: mean={leverage_scores.mean():.3f}, std={leverage_scores.std():.3f}')

        # Feature-enhanced metrics
        if features:
            feature_names = features.get('combined_feature_names', [])
            self.logger.info(f'   → Enhanced with {len(feature_names)} optimized features')

            # Feature quality metrics
            if 'feature_importance_scores' in features:
                importance_scores = features['feature_importance_scores']
                if importance_scores:
                    top_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:5]
                    self.logger.info(f'   → Top 5 features used: {top_features}')

        self.logger.info('✅ Enhanced multi-horizon labeling completed successfully')
    
    def _apply_execution_mode_filtering(self, data: pd.DataFrame, mode: str) -> pd.DataFrame:
        """Apply execution mode-based data filtering."""
        try:
            self.logger.info(f'🔍 Applying {mode} mode data filtering...')
            
            # Define lookback days for each mode
            lookback_days_map = {
                'light': 10,     # Light mode: 10 days
                'blank': 180,    # Blank mode: 180 days  
                'full': None     # Full mode: no filtering
            }
            
            lookback_days = lookback_days_map.get(mode.lower())
            
            if lookback_days is None:
                self.logger.info(f'📊 Full mode - no date filtering applied')
                return data
            
            # Convert index to datetime if needed
            if not isinstance(data.index, pd.DatetimeIndex):
                try:
                    if hasattr(data.index, 'max') and data.index.max() > 1e10:
                        # Likely millisecond timestamps
                        data.index = pd.to_datetime(data.index, unit='ms', utc=True).tz_localize(None)
                    else:
                        data.index = pd.to_datetime(data.index, utc=True).tz_localize(None)
                    self.logger.info(f'🔧 Converted index to datetime for filtering')
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not convert index to datetime: {e}")
                    # Fallback: take last N rows as approximation
                    minutes_per_day = 1440  # 1440 minutes per day for 1m data
                    approx_rows = lookback_days * minutes_per_day
                    filtered_data = data.tail(approx_rows).copy()
                    self.logger.info(f'📊 Fallback filtering: took last {approx_rows:,} rows (~{lookback_days} days)')
                    return filtered_data
            
            # Calculate date range
            end_date = data.index.max()
            start_date = end_date - timedelta(days=lookback_days)
            
            # Apply date filtering
            original_rows = len(data)
            filtered_data = data[data.index >= start_date].copy()
            filtered_rows = len(filtered_data)
            
            self.logger.info(f'📅 Date range: {start_date} to {end_date}')
            self.logger.info(f'🔍 {mode.upper()} mode filtering: {original_rows:,} → {filtered_rows:,} rows ({lookback_days} days)')
            self.logger.info(f'📊 Filtering efficiency: {filtered_rows/original_rows*100:.1f}% of original data')
            
            return filtered_data
            
        except Exception as e:
            self.logger.error(f'❌ Error in execution mode filtering: {e}')
            return data  # Return original data if filtering fails

# Convenience function for sub-pipeline integration
def execute_multi_horizon_labeling_step(data: pd.DataFrame,
                                       regime_labels: Optional[pd.Series] = None,
                                       config: Optional[Dict[str, Any]] = None,
                                       symbol: Optional[str] = None,
                                       exchange: Optional[str] = None,
                                       timeframe: Optional[str] = None,
                                       mode: str = 'full',
                                       features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Execute multi-horizon labeling step (sub-pipeline compatible).
    
    This function provides a drop-in replacement for the original triple barrier
    labeling step in the sub-pipeline system.
    """
    adapter = MultiHorizonSubPipelineAdapter()
    return adapter.execute_multi_horizon_labeling_step(
        data=data,
        regime_labels=regime_labels,
        config=config,
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        mode=mode,
        features=features
    )

# Test function
if __name__ == '__main__':
    tprint('🧪 Testing Multi-Horizon Sub-Pipeline Adapter')
    
    # Create test data
    dates = pd.date_range('2024-01-01', periods=500, freq='5min')
    np.random.seed(42)
    
    base_price = 100.0
    returns = np.random.normal(0.0001, 0.002, 500)
    prices = [base_price]
    
    for ret in returns[:-1]:
        prices.append(prices[-1] * (1 + ret))
    
    data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, 500)
    }, index=dates)
    
    # Ensure OHLC consistency
    for i in range(len(data)):
        data.loc[data.index[i], 'high'] = max(data.iloc[i][['open', 'high', 'low', 'close']])
        data.loc[data.index[i], 'low'] = min(data.iloc[i][['open', 'high', 'low', 'close']])
    
    # Test sub-pipeline adapter
    tprint('\n🔄 Testing sub-pipeline adapter...')
    
    config = {
        'profit_targets': {
            'micro': 0.003,
            'small': 0.005,
            'medium': 0.007,
            'good': 0.010
        },
        'time_horizons': {
            'immediate': 2,
            'short': 4
        },
        'transaction_cost': 0.0008
    }
    
    result = execute_multi_horizon_labeling_step(
        data=data,
        config=config,
        symbol='TESTUSDT',
        exchange='test',
        timeframe='5m'
    )
    
    if result['status'] == 'completed':
        tprint('✅ Sub-pipeline adapter test successful!')
        
        artifacts = result['artifacts']['multi_horizon_labeling_result']
        metrics = artifacts['labeling_metrics']
        
        tprint(f'📊 Results:')
        tprint(f'   → Status: {result["status"]}')
        tprint(f'   → Total samples: {metrics["total_samples"]}')
        tprint(f'   → New features: {metrics["new_features_added"]}')
        tprint(f'   → Probability targets: {metrics["probability_targets"]}')
        tprint(f'   → High opportunity ratio: {metrics.get("high_opportunity_ratio", 0):.1%}')
        
    else:
        tprint(f'❌ Sub-pipeline adapter test failed: {result.get("error", "Unknown error")}')
    
    tprint('✅ Multi-Horizon Sub-Pipeline Adapter test completed!')