"""
Multi-Horizon Sub-Pipeline Adapter

This module provides an adapter to integrate multi-horizon labeling into the existing
sub-pipeline system, replacing the triple barrier method.

Key features:
- Drop-in replacement for triple barrier labeling
- Maintains compatibility with existing sub-pipeline
- Provides enhanced labeling with reversal capture
- Optimized for short-term, high-frequency trading
"""

import pandas as pd
import numpy as np
import functools
import time
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime

# Optimized imports using common utilities
from src.utils.logger import get_logger
from src.core.decorators import handles_errors, traced, validates, log_execution_time
from src.utils.common_utilities import (
    safe_dataframe_operation, validate_dataframe_columns, 
    safe_convert_dtypes
)
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

from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer

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
from .multi_horizon_profit_labeler import (
    MultiHorizonProfitLabeler, 
    MultiHorizonConfig,
    apply_multi_horizon_labeling
)

class MultiHorizonSubPipelineAdapter:
    """
    Adapter for integrating multi-horizon labeling into existing sub-pipeline.
    
    This adapter provides a drop-in replacement for the triple barrier labeling
    step while maintaining compatibility with the existing pipeline structure.
    """
    
    def __init__(self):
        """Initialize the adapter with hardware optimizations."""
        self.logger = get_logger('MultiHorizonSubPipelineAdapter')
        
        # Initialize hardware optimizers
        self.memory_optimizer = get_m1_memory_optimizer()
        self.cpu_optimizer = get_m1_cpu_optimizer()
        self.serializer = UniversalSerializer()
        
        # Optimize CPU for data processing
        if self.cpu_optimizer:
            self.cpu_optimizer.optimize_numpy_operations()
        
        self.logger.info('🔄 Multi-Horizon Sub-Pipeline Adapter initialized with M1 optimizations')
    
    def execute_multi_horizon_labeling_step(self,
                                          data: pd.DataFrame,
                                          regime_labels: Optional[pd.Series] = None,
                                          config: Optional[Dict[str, Any]] = None,
                                          symbol: Optional[str] = None,
                                          exchange: Optional[str] = None,
                                          timeframe: Optional[str] = None,
                                          mode: str = 'full') -> Dict[str, Any]:
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
            
        Returns:
            Dictionary with labeling results compatible with sub-pipeline
        """
        self.logger.info(f'🎯 Executing multi-horizon labeling step for {symbol or "unknown"} on {timeframe or "unknown"}')
        self.logger.info(f'📊 Input data shape: {data.shape if data is not None else "None"}')
        self.logger.info(f'🚨 EXECUTION MODE: {mode}')
        
        # FORCE DATA FILTERING IMMEDIATELY
        if data is not None and len(data) > 50000:  # Only filter large datasets
            original_size = len(data)
            if mode and mode.lower() == 'light':
                data = data.tail(14400).copy()
                self.logger.info(f'🔥 FORCED LIGHT FILTERING: {original_size:,} → {len(data):,} rows')
            elif mode and mode.lower() == 'blank':
                data = data.tail(259200).copy()
                self.logger.info(f'🔥 FORCED BLANK FILTERING: {original_size:,} → {len(data):,} rows')
        
        try:
            # Validate input data with enhanced validation
            if not validate_dataframe(data):
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
                # Create multi-horizon configuration
                labeling_config = self._create_labeling_config(config)
                self.logger.info(f'🔧 Created labeling config: {labeling_config.__dict__}')
                
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
                        self.logger.info('🔄 Calling apply_multi_horizon_labeling...')
                        
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
                            labeled_data = self._generate_dynamic_labels(data, labeling_config)  # 'data' is already filtered!
                            
                            self.logger.info(f'📊 Filtered data dynamic labeling completed: {labeled_data.shape}')
                except Exception as e:
                    self.logger.error(f'❌ apply_multi_horizon_labeling failed: {e}')
                    import traceback
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
    
    def _create_labeling_config(self, config: Optional[Dict[str, Any]] = None) -> MultiHorizonConfig:
        """Create multi-horizon configuration from sub-pipeline config."""
        labeling_config = MultiHorizonConfig()
        
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
            from datetime import datetime, timedelta
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
                                       mode: str = 'full') -> Dict[str, Any]:
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
        mode=mode
    )

# Test function
if __name__ == '__main__':
    from src.utils.tprint import tprint
    import numpy as np
    
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