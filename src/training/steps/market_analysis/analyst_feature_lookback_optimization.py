"""
Analyst Feature Lookback Optimization - No Long/Short Differentiation

This component optimizes feature lookback periods for Analyst models on 5m timeframe
without long/short differentiation. Provides unified optimization for overall opportunity assessment.

Key features:
- No long/short differentiation (unified approach)
- Optimized for 5m timeframe
- Simplified optimization process
- Focus on overall opportunity assessment
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Import tprint for consistent logging
from src.utils.tprint import tprint

# Core dependencies
import numpy as np
import pandas as pd

# Import common utilities
from src.utils.common_operations import (
    safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
    calculate_data_quality_metrics, safe_merge_dataframes, safe_groupby_operation,
    safe_apply_function, create_summary_statistics, safe_drop_columns,
    safe_rename_columns, validate_timestamp_column, safe_timestamp_conversion,
    get_dataframe_info, safe_filter_dataframe, create_data_quality_report,
    optimize_dataframe_dtypes, safe_fillna, safe_rolling, safe_groupby_operation,
    safe_apply_function, safe_filter_dataframe, create_summary_statistics,
    safe_to_parquet, safe_read_parquet, validate_dataframe_schema,
    guard_dataframe_nulls, memory_checkpoint, gpu_context, optimize_memory,
    get_memory_usage, integrate_with_m1_optimizers, get_m1_gpu_manager,
    get_m1_memory_optimizer, get_m1_cpu_optimizer, validate_dataframe
)

from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    validate_positive, validate_range, safe_kelly_calculation,
    safe_weighted_average, safe_percentage_change, safe_correlation,
    safe_covariance, safe_mean, safe_std, safe_percentile,
    validate_correlation_matrix, safe_matrix_inverse, math_safe,
    MathValidation, MathValidationError
)

# Import matrix operations for efficient computation
from src.utils.matrix_operations.unified_operations import (
    UnifiedMatrixOperations, safe_correlation_matrix,
    safe_matrix_inverse, get_unified_matrix_operations
)

# Import ML common utilities if available
try:
    from src.utils.ml_common.data_processing.data_quality import DataQualityUtilities
    from src.utils.ml_common.data_processing.feature_preparation import FeaturePreparator
    from src.utils.ml_common.validation.cv import purged_time_series_splits, PurgedSplitConfig
    from src.utils.ml_common.monitoring.enhanced_error_detector import (
        EnhancedErrorDetector, ErrorSeverity, ErrorCategory
    )
    ML_COMMON_AVAILABLE = True
except ImportError as e:
    ML_COMMON_AVAILABLE = False
    tprint(f"⚠️ ML common utilities not available: {e}")

class OptimizationStatus(Enum):
    """Status of optimization process."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"

@dataclass
class AnalystOptimizationConfig:
    """Configuration for Analyst feature lookback optimization."""
    # Optimization parameters
    max_lookback_periods: int = 20  # Maximum lookback periods to test
    min_lookback_periods: int = 5   # Minimum lookback periods to test
    optimization_steps: int = 5     # Number of optimization steps
    
    # Performance thresholds
    min_correlation_threshold: float = 0.1  # Minimum correlation for feature selection
    max_correlation_threshold: float = 0.9  # Maximum correlation to avoid redundancy
    min_importance_threshold: float = 0.01  # Minimum feature importance
    
    # Timeframe specific (5m for Analyst)
    timeframe_minutes: int = 5
    max_horizon_minutes: int = 20  # 4 periods * 5 minutes
    
    # Quality thresholds
    min_data_quality_score: float = 0.7
    min_feature_quality_score: float = 0.6
    
    # Memory optimization
    enable_memory_optimization: bool = True
    max_memory_usage_gb: float = 8.0
    batch_size: int = 1000

@dataclass
class AnalystOptimizationResult:
    """Result of Analyst feature lookback optimization."""
    # Optimization results
    optimal_lookback_periods: Dict[str, int]
    optimization_scores: Dict[str, float]
    feature_importance_scores: Dict[str, float]
    
    # Quality metrics
    overall_quality_score: float
    data_quality_score: float
    feature_quality_score: float
    
    # Performance metrics
    optimization_time: float
    total_features_optimized: int
    
    # Status
    optimization_status: OptimizationStatus
    success: bool
    error_message: Optional[str] = None

class AnalystFeatureLookbackOptimizer:
    """
    Analyst Feature Lookback Optimizer - NO LONG/SHORT DIFFERENTIATION.
    
    Optimizes feature lookback periods for Analyst models on 5m timeframe
    without long/short differentiation.
    """
    
    def __init__(self, config: Optional[AnalystOptimizationConfig] = None):
        """Initialize the Analyst feature lookback optimizer."""
        self.config = config or AnalystOptimizationConfig()
        self.logger = logging.getLogger('AnalystFeatureLookbackOptimizer')
        
        # Initialize matrix operations
        self.matrix_ops = get_unified_matrix_operations()
        
        # Initialize ML common utilities if available
        if ML_COMMON_AVAILABLE:
            self.data_quality_utils = DataQualityUtilities()
            self.feature_preparator = FeaturePreparator()
            self.error_detector = EnhancedErrorDetector()
        else:
            self.data_quality_utils = None
            self.feature_preparator = None
            self.error_detector = None
        
        # Initialize hardware optimizers
        self.memory_optimizer = get_m1_memory_optimizer() if self.config.enable_memory_optimization else None
        self.cpu_optimizer = get_m1_cpu_optimizer()
        
        if self.memory_optimizer:
            self.memory_optimizer.set_memory_limit(self.config.max_memory_usage_gb)
        
        self.logger.info("🚀 Analyst Feature Lookback Optimizer initialized (NO LONG/SHORT DIFFERENTIATION)")
        self.logger.info(f"   → Max lookback periods: {self.config.max_lookback_periods}")
        self.logger.info(f"   → Min lookback periods: {self.config.min_lookback_periods}")
        self.logger.info(f"   → Timeframe: {self.config.timeframe_minutes}m")
        self.logger.info(f"   → Memory optimization: {'Enabled' if self.config.enable_memory_optimization else 'Disabled'}")
    
    async def optimize_lookback_periods(self, 
                                      market_data: pd.DataFrame,
                                      target_data: Optional[pd.Series] = None) -> AnalystOptimizationResult:
        """
        Optimize feature lookback periods for Analyst (UNIFIED APPROACH).
        
        Args:
            market_data: Market data for optimization
            target_data: Target variable (optional, will use correlation-based optimization if not provided)
            
        Returns:
            AnalystOptimizationResult with optimization results
        """
        start_time = time.time()
        self.logger.info("🔍 Starting Analyst feature lookback optimization (UNIFIED APPROACH)")
        
        try:
            # Step 1: Validate input data
            validation_result = await self._validate_input_data(market_data, target_data)
            if not validation_result['is_valid']:
                return AnalystOptimizationResult(
                    optimal_lookback_periods={},
                    optimization_scores={},
                    feature_importance_scores={},
                    overall_quality_score=0.0,
                    data_quality_score=0.0,
                    feature_quality_score=0.0,
                    optimization_time=time.time() - start_time,
                    total_features_optimized=0,
                    optimization_status=OptimizationStatus.FAILED,
                    success=False,
                    error_message=validation_result['error_message']
                )
            
            # Step 2: Prepare data for optimization
            prepared_data = await self._prepare_data_for_optimization(market_data, target_data)
            
            # Step 3: Generate feature candidates
            feature_candidates = await self._generate_feature_candidates(prepared_data)
            
            # Step 4: Optimize lookback periods (UNIFIED APPROACH)
            optimization_result = await self._optimize_lookback_periods_unified(
                feature_candidates, prepared_data
            )
            
            # Step 5: Calculate quality scores
            quality_scores = await self._calculate_quality_scores(
                optimization_result, prepared_data
            )
            
            # Step 6: Create final result
            result = AnalystOptimizationResult(
                optimal_lookback_periods=optimization_result['optimal_periods'],
                optimization_scores=optimization_result['scores'],
                feature_importance_scores=optimization_result['importance_scores'],
                overall_quality_score=quality_scores['overall'],
                data_quality_score=quality_scores['data'],
                feature_quality_score=quality_scores['feature'],
                optimization_time=time.time() - start_time,
                total_features_optimized=len(optimization_result['optimal_periods']),
                optimization_status=OptimizationStatus.COMPLETED,
                success=True
            )
            
            self.logger.info(f"✅ Analyst optimization completed: {result.total_features_optimized} features optimized")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Analyst optimization failed: {e}")
            return AnalystOptimizationResult(
                optimal_lookback_periods={},
                optimization_scores={},
                feature_importance_scores={},
                overall_quality_score=0.0,
                data_quality_score=0.0,
                feature_quality_score=0.0,
                optimization_time=time.time() - start_time,
                total_features_optimized=0,
                optimization_status=OptimizationStatus.FAILED,
                success=False,
                error_message=str(e)
            )
    
    async def _validate_input_data(self, market_data: pd.DataFrame, 
                                 target_data: Optional[pd.Series]) -> Dict[str, Any]:
        """Validate input data for optimization."""
        try:
            # Check market data
            if market_data is None or market_data.empty:
                return {'is_valid': False, 'error_message': 'Market data is empty or None'}
            
            if len(market_data) < self.config.min_lookback_periods + 10:
                return {'is_valid': False, 'error_message': f'Insufficient data: {len(market_data)} rows'}
            
            # Check required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in market_data.columns]
            
            if len(missing_columns) == len(required_columns):
                return {'is_valid': False, 'error_message': 'No OHLCV columns found'}
            
            # Check data quality
            if self.data_quality_utils:
                quality_metrics = self.data_quality_utils.calculate_quality_metrics(market_data)
                if quality_metrics['overall_score'] < self.config.min_data_quality_score:
                    return {'is_valid': False, 'error_message': f'Low data quality: {quality_metrics["overall_score"]:.3f}'}
            
            # Check target data if provided
            if target_data is not None:
                if len(target_data) != len(market_data):
                    return {'is_valid': False, 'error_message': 'Target data length mismatch'}
                
                if target_data.isna().all():
                    return {'is_valid': False, 'error_message': 'Target data contains only NaN values'}
            
            return {'is_valid': True, 'error_message': None}
            
        except Exception as e:
            return {'is_valid': False, 'error_message': f'Validation error: {e}'}
    
    async def _prepare_data_for_optimization(self, market_data: pd.DataFrame, 
                                           target_data: Optional[pd.Series]) -> Dict[str, Any]:
        """Prepare data for optimization."""
        try:
            # Clean and prepare market data
            prepared_market_data = market_data.copy()
            
            # Handle missing values
            prepared_market_data = safe_fillna(prepared_market_data, method='forward')
            prepared_market_data = safe_fillna(prepared_market_data, method='backward')
            
            # Ensure numeric types
            for col in prepared_market_data.columns:
                if prepared_market_data[col].dtype == 'object':
                    try:
                        prepared_market_data[col] = pd.to_numeric(prepared_market_data[col], errors='coerce')
                    except:
                        prepared_market_data = prepared_market_data.drop(columns=[col])
            
            # Prepare target data
            prepared_target = None
            if target_data is not None:
                prepared_target = target_data.copy()
                prepared_target = prepared_target.fillna(method='forward').fillna(method='backward')
            
            return {
                'market_data': prepared_market_data,
                'target_data': prepared_target,
                'data_quality_score': self._calculate_data_quality_score(prepared_market_data)
            }
            
        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            raise
    
    async def _generate_feature_candidates(self, prepared_data: Dict[str, Any]) -> List[str]:
        """Generate feature candidates for optimization."""
        try:
            market_data = prepared_data['market_data']
            feature_candidates = []
            
            # Price-based features
            price_columns = ['open', 'high', 'low', 'close']
            available_price_columns = [col for col in price_columns if col in market_data.columns]
            
            for col in available_price_columns:
                feature_candidates.append(f'{col}_sma')
                feature_candidates.append(f'{col}_ema')
                feature_candidates.append(f'{col}_rsi')
                feature_candidates.append(f'{col}_momentum')
                feature_candidates.append(f'{col}_volatility')
            
            # Volume-based features
            if 'volume' in market_data.columns:
                feature_candidates.append('volume_sma')
                feature_candidates.append('volume_ema')
                feature_candidates.append('volume_ratio')
                feature_candidates.append('volume_momentum')
            
            # Technical indicators
            if len(available_price_columns) >= 4:  # Need OHLC
                feature_candidates.extend([
                    'bollinger_upper', 'bollinger_lower', 'bollinger_middle',
                    'atr', 'adx', 'macd', 'macd_signal', 'macd_histogram',
                    'stoch_k', 'stoch_d', 'williams_r', 'cci'
                ])
            
            # Cross-timeframe features (simplified for Analyst)
            feature_candidates.extend([
                'price_change_ratio', 'volume_price_trend', 'money_flow_index',
                'on_balance_volume', 'accumulation_distribution'
            ])
            
            self.logger.info(f"Generated {len(feature_candidates)} feature candidates")
            return feature_candidates
            
        except Exception as e:
            self.logger.error(f"Feature candidate generation failed: {e}")
            return []
    
    async def _optimize_lookback_periods_unified(self, feature_candidates: List[str], 
                                               prepared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize lookback periods using unified approach (NO LONG/SHORT DIFFERENTIATION)."""
        try:
            market_data = prepared_data['market_data']
            target_data = prepared_data['target_data']
            
            optimal_periods = {}
            scores = {}
            importance_scores = {}
            
            # Test different lookback periods for each feature
            for feature in feature_candidates:
                best_period = self.config.min_lookback_periods
                best_score = 0.0
                best_importance = 0.0
                
                for period in range(self.config.min_lookback_periods, 
                                  min(self.config.max_lookback_periods + 1, len(market_data) // 2)):
                    
                    try:
                        # Generate feature with current lookback period
                        feature_values = self._generate_feature_with_lookback(
                            market_data, feature, period
                        )
                        
                        if feature_values is None or len(feature_values) == 0:
                            continue
                        
                        # Calculate score based on target correlation or variance
                        if target_data is not None:
                            # Use correlation with target
                            correlation = self._calculate_feature_target_correlation(
                                feature_values, target_data
                            )
                            score = abs(correlation) if not np.isnan(correlation) else 0.0
                        else:
                            # Use variance as proxy for information content
                            variance = np.var(feature_values[~np.isnan(feature_values)])
                            score = variance if not np.isnan(variance) else 0.0
                        
                        # Update best period if score is better
                        if score > best_score:
                            best_score = score
                            best_period = period
                            best_importance = score
                        
                    except Exception as e:
                        self.logger.warning(f"Error optimizing {feature} with period {period}: {e}")
                        continue
                
                # Store results if we found a valid period
                if best_score > self.config.min_importance_threshold:
                    optimal_periods[feature] = best_period
                    scores[feature] = best_score
                    importance_scores[feature] = best_importance
            
            return {
                'optimal_periods': optimal_periods,
                'scores': scores,
                'importance_scores': importance_scores
            }
            
        except Exception as e:
            self.logger.error(f"Lookback optimization failed: {e}")
            return {
                'optimal_periods': {},
                'scores': {},
                'importance_scores': {}
            }
    
    def _generate_feature_with_lookback(self, market_data: pd.DataFrame, 
                                       feature_name: str, lookback_period: int) -> Optional[np.ndarray]:
        """Generate a feature with specified lookback period."""
        try:
            if feature_name.endswith('_sma'):
                base_col = feature_name.replace('_sma', '')
                if base_col in market_data.columns:
                    return market_data[base_col].rolling(window=lookback_period).mean().values
            
            elif feature_name.endswith('_ema'):
                base_col = feature_name.replace('_ema', '')
                if base_col in market_data.columns:
                    return market_data[base_col].ewm(span=lookback_period).mean().values
            
            elif feature_name.endswith('_rsi'):
                base_col = feature_name.replace('_rsi', '')
                if base_col in market_data.columns:
                    return self._calculate_rsi(market_data[base_col], lookback_period)
            
            elif feature_name.endswith('_momentum'):
                base_col = feature_name.replace('_momentum', '')
                if base_col in market_data.columns:
                    return market_data[base_col].pct_change(periods=lookback_period).values
            
            elif feature_name.endswith('_volatility'):
                base_col = feature_name.replace('_volatility', '')
                if base_col in market_data.columns:
                    return market_data[base_col].rolling(window=lookback_period).std().values
            
            elif feature_name == 'bollinger_upper':
                if 'close' in market_data.columns:
                    sma = market_data['close'].rolling(window=lookback_period).mean()
                    std = market_data['close'].rolling(window=lookback_period).std()
                    return (sma + 2 * std).values
            
            elif feature_name == 'bollinger_lower':
                if 'close' in market_data.columns:
                    sma = market_data['close'].rolling(window=lookback_period).mean()
                    std = market_data['close'].rolling(window=lookback_period).std()
                    return (sma - 2 * std).values
            
            elif feature_name == 'atr':
                if all(col in market_data.columns for col in ['high', 'low', 'close']):
                    return self._calculate_atr(market_data, lookback_period)
            
            # Add more feature types as needed
            return None
            
        except Exception as e:
            self.logger.warning(f"Error generating {feature_name} with period {lookback_period}: {e}")
            return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> np.ndarray:
        """Calculate RSI indicator."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.values
        except:
            return np.full(len(prices), np.nan)
    
    def _calculate_atr(self, market_data: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate Average True Range."""
        try:
            high = market_data['high']
            low = market_data['low']
            close = market_data['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            return atr.values
        except:
            return np.full(len(market_data), np.nan)
    
    def _calculate_feature_target_correlation(self, feature_values: np.ndarray, 
                                             target_values: pd.Series) -> float:
        """Calculate correlation between feature and target."""
        try:
            # Align lengths
            min_len = min(len(feature_values), len(target_values))
            if min_len < 10:  # Need minimum samples for correlation
                return 0.0
            
            feature_aligned = feature_values[-min_len:]
            target_aligned = target_values.iloc[-min_len:].values
            
            # Remove NaN values
            valid_mask = ~(np.isnan(feature_aligned) | np.isnan(target_aligned))
            if np.sum(valid_mask) < 10:
                return 0.0
            
            feature_clean = feature_aligned[valid_mask]
            target_clean = target_aligned[valid_mask]
            
            # Calculate correlation
            correlation = np.corrcoef(feature_clean, target_clean)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            self.logger.warning(f"Correlation calculation failed: {e}")
            return 0.0
    
    async def _calculate_quality_scores(self, optimization_result: Dict[str, Any], 
                                      prepared_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate quality scores for optimization results."""
        try:
            # Data quality score
            data_quality_score = prepared_data.get('data_quality_score', 0.5)
            
            # Feature quality score based on optimization results
            feature_scores = list(optimization_result['scores'].values())
            if feature_scores:
                feature_quality_score = np.mean(feature_scores)
            else:
                feature_quality_score = 0.0
            
            # Overall quality score
            overall_quality_score = (data_quality_score + feature_quality_score) / 2
            
            return {
                'data': data_quality_score,
                'feature': feature_quality_score,
                'overall': overall_quality_score
            }
            
        except Exception as e:
            self.logger.warning(f"Quality score calculation failed: {e}")
            return {
                'data': 0.5,
                'feature': 0.5,
                'overall': 0.5
            }
    
    def _calculate_data_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate data quality score."""
        try:
            if self.data_quality_utils:
                quality_metrics = self.data_quality_utils.calculate_quality_metrics(data)
                return quality_metrics['overall_score']
            else:
                # Basic quality assessment
                score = 1.0
                
                # Check for missing values
                missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
                score *= (1.0 - missing_ratio)
                
                # Check for constant columns
                constant_columns = data.nunique() <= 1
                constant_ratio = constant_columns.sum() / len(data.columns)
                score *= (1.0 - constant_ratio)
                
                return max(0.0, score)
                
        except Exception as e:
            self.logger.warning(f"Data quality calculation failed: {e}")
            return 0.5

# Convenience functions
def create_analyst_feature_lookback_optimizer(config: Optional[AnalystOptimizationConfig] = None) -> AnalystFeatureLookbackOptimizer:
    """Create Analyst feature lookback optimizer."""
    return AnalystFeatureLookbackOptimizer(config)

async def optimize_analyst_feature_lookback(market_data: pd.DataFrame,
                                          target_data: Optional[pd.Series] = None,
                                          config: Optional[AnalystOptimizationConfig] = None) -> AnalystOptimizationResult:
    """Optimize Analyst feature lookback periods."""
    optimizer = AnalystFeatureLookbackOptimizer(config)
    return await optimizer.optimize_lookback_periods(market_data, target_data)