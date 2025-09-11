"""SR Clustering Stage: Generate SR clusters using backtesting-enhanced clustering."""

import asyncio
import sys
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
try:
    from collections.abc import Iterable
except ImportError:
    from typing import Iterable
import time
import json
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import traceback
import logging
import random
import gc

# Core imports
try:
    from src.training.base_step import BaseStep
except ImportError:
    # Fallback BaseStep class
    class BaseStep:
        def __init__(self, config):
            self.config = config
        
        async def execute(self, data):
            pass
        
        def validate_config(self):
            pass
        
        def get_status(self):
            return {}

from src.utils.logger import system_logger

# Initialize logger early to avoid usage before definition
logger = system_logger.getChild('SRClustering')

# Required utility modules - Comprehensive Integration
from src.utils.common_operations import (
    safe_json_load, safe_json_dump, safe_read_parquet, safe_to_parquet,
    ensure_directory, create_fallback_logger, create_fallback_decorator,
    safe_mean, safe_std, safe_float, safe_int, safe_append, safe_extend,
    safe_dict_get, safe_dict_items, safe_lower, safe_upper, safe_join,
    get_current_datetime, format_datetime, create_empty_dataframe,
    safe_fillna, safe_rolling, safe_copy, safe_deepcopy, safe_sleep,
    safe_gather, create_async_task, get_logger, setup_basic_logging,
    safe_exception_handler, suggest_float_uniform, suggest_int_uniform,
    validate_dataframe, validate_numeric_range, optimize_dataframe_dtypes,
    timed_operation, format_bytes, chunked_iterable, parallel_map,
    safe_log_metric, safe_log_params, safe_log_artifact, get_common_operations_health_status
)

# SR Clustering System Integration - Required
from src.utils.sr_clustering import (
    get_backtesting_enhanced_clustering, BacktestingEnhancedConfig,
    get_predictive_sr_engine, PredictiveConfig,
    get_trading_ml_integration, TradingMLConfig
)
logger.info('✅ SR clustering system loaded')

# Core decorators and errors
from src.core.decorators import handles_errors, error_boundary, converts_errors
from src.core.errors import (
    AppError, ValidationError, DataIntegrityError, 
    NotFoundError, BusinessRuleError
)

# Pipeline standards and utilities
from src.utils.pipeline_standards import PipelineStandards
from src.utils.monitoring_utils import (
    global_monitor, function_tracker, logging_patterns
)
from src.utils.comprehensive_function_logger import (
    log_step_functions, log_important_calls, log_all_calls, 
    log_internal_call, log_step_progress, log_data_operation
)

# M1 Optimization Utilities - Integrated via Common Operations
try:
    from src.utils.common_operations import (
        integrate_with_m1_optimizers, get_m1_gpu_manager, get_m1_memory_optimizer,
        get_m1_cpu_optimizer, cleanup_m1_optimizers, memory_checkpoint, gpu_context,
        optimize_memory, get_memory_usage
    )

    # Initialize M1 integration through common operations
    m1_integration_result = integrate_with_m1_optimizers()
    M1_GPU_AVAILABLE = m1_integration_result.get('gpu_manager', False)
    M1_MEMORY_AVAILABLE = m1_integration_result.get('memory_optimizer', False)
    M1_CPU_AVAILABLE = m1_integration_result.get('cpu_optimizer', False)
    M1_BATCH_AVAILABLE = M1_CPU_AVAILABLE  # Batch processor available if CPU optimizer is

    integration_status = m1_integration_result.get('integration_status', 'unknown')
    if integration_status == 'success':
        logger.info("✅ Complete M1 utilities integration successful")
    elif integration_status == 'partial':
        logger.info("⚠️ Partial M1 utilities integration - some components available")
    else:
        logger.warning("❌ M1 utilities integration failed")

except ImportError as e:
    M1_GPU_AVAILABLE = False
    M1_MEMORY_AVAILABLE = False
    M1_CPU_AVAILABLE = False
    M1_BATCH_AVAILABLE = False
    logger.warning(f"M1 utilities integration not available: {e}")
except Exception as e:
    M1_GPU_AVAILABLE = False
    M1_MEMORY_AVAILABLE = False
    M1_CPU_AVAILABLE = False
    M1_BATCH_AVAILABLE = False
    logger.error(f"Unexpected error in M1 utilities integration: {e}")

# Utility functions for memory management and validation
def get_memory_usage():
    try:
        import psutil
        return psutil.Process().memory_info().rss
    except ImportError:
        return 0

def format_bytes(bytes_val):
    return f"{bytes_val / 1024 / 1024:.1f} MB"

def memory_checkpoint(name):
    pass

def optimize_dataframe_dtypes(df):
    return df

def validate_dataframe(df):
    return True

def validate_finite(arr):
    return np.all(np.isfinite(arr))

def safe_divide(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b!=0)


class SRClusteringStep(BaseStep):
    """SR Clustering Stage: Generate SR clusters using backtesting-enhanced clustering."""

    @log_important_calls
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize SR clustering step."""
        super().__init__(config)
        self.logger = system_logger.getChild('SRClusteringStep')
        self.standards = PipelineStandards(self.logger)
        self.sr_optimization_config = config.get('sr_optimization', {
            'min_touches': 2, 
            'tolerance_pct': 0.5, 
            'lookback_periods': 100
        })

        # Adjust configuration for LIGHT mode
        training_mode = os.environ.get('LIGHT_TRAINING_MODE', '')
        if training_mode == '1' or config.get('training_mode') == 'light':
            self.sr_optimization_config['lookback_periods'] = 10
            self.logger.info('💡 LIGHT mode: Adjusted lookback_periods to 10 (was 100)')
        
        # Clustering configuration
        self.clustering_config = config.get('sr_clustering', {
            'min_levels_for_learning': 5,
            'quality_filter_threshold': 0.1,
            'proximity_adjustment_factor': 0.5
        })
        
        # Initialize automatic memory management
        try:
            from src.utils.hardware.memory_optimization import get_memory_manager, MemoryContext as memory_context
            self.memory_manager = get_memory_manager()
            self.memory_manager.start_monitoring()
            self.logger.info("🧠 Memory management initialized")
        except Exception as e:
            self.logger.warning(f"Memory manager initialization failed: {e}")
            # Fallback memory manager
            class FallbackMemoryManager:
                def start_monitoring(self):
                    pass
                def stop_monitoring(self):
                    pass
            self.memory_manager = FallbackMemoryManager()

    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the SR clustering stage."""
        self.logger.info('🚀 Starting SR Clustering Stage execution')
        start_time = time.time()

        try:
            # Get SR levels from previous stage or pipeline state
            sr_levels = pipeline_state.get('sr_levels')
            if sr_levels is None:
                raise ValueError("No SR levels found in pipeline state")

            self.logger.info(f'📊 SR levels loaded: {len(sr_levels.get("all_levels", []))} total levels')

            # Cluster SR levels
            clustered_levels = self._cluster_sr_levels(sr_levels)
            
            execution_time = time.time() - start_time
            self.logger.info(f'✅ SR Clustering completed in {execution_time:.2f} seconds')

            return {
                'success': True,
                'clustered_levels': clustered_levels,
                'execution_time': execution_time,
                'stage': 'sr_clustering'
            }

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f'❌ SR Clustering failed: {e}')
            return {
                'success': False,
                'error': str(e),
                'execution_time': execution_time,
                'stage': 'sr_clustering'
            }

    def _cluster_sr_levels(self, sr_levels: Dict[str, Any]) -> Dict[str, Any]:
        """Cluster SR levels using backtesting-enhanced clustering."""
        self.logger.info('🚀 ===== STARTING SR CLUSTERING SYSTEM =====')
        self.logger.info('🚀 Using SR clustering system with weight optimization...')
        clustering_start_time = time.time()

        try:
            # Extract levels from input
            all_levels = sr_levels.get('all_levels', [])
            if not all_levels:
                self.logger.error('❌ No SR levels provided for clustering.')
                raise ValueError("No SR levels provided for clustering.")

            self.logger.info(f'📊 Input levels for clustering: {len(all_levels)} total levels')

            # Use backtesting-enhanced clustering
            self.logger.info('🔧 Configuring backtesting-enhanced clustering...')
            clustering_config = BacktestingEnhancedConfig(
                min_levels_for_learning=self.clustering_config.get('min_levels_for_learning', 5),
                quality_filter_threshold=self.clustering_config.get('quality_filter_threshold', 0.1),
                proximity_adjustment_factor=self.clustering_config.get('proximity_adjustment_factor', 0.5)
            )
            
            self.logger.info(f'🔧 Clustering Configuration:')
            self.logger.info(f'   • Min levels for learning: {clustering_config.min_levels_for_learning}')
            self.logger.info(f'   • Quality filter threshold: {clustering_config.quality_filter_threshold}')
            self.logger.info(f'   • Proximity adjustment factor: {clustering_config.proximity_adjustment_factor}')
            
            clustering_creation_start = time.time()
            clustering = get_backtesting_enhanced_clustering(clustering_config)
            clustering_creation_time = time.time() - clustering_creation_start
            self.logger.info(f'✅ Backtesting-enhanced clustering created in {clustering_creation_time:.2f} seconds')
            
            # Convert levels to dict format for clustering if needed
            self.logger.info('🔄 Preparing levels for clustering...')
            levels_dict = []
            conversion_start = time.time()
            
            for i, level in enumerate(all_levels):
                if isinstance(level, dict):
                    # Already in dict format
                    levels_dict.append(level)
                else:
                    # Convert from object format
                    if hasattr(level, 'price'):
                        level_dict = {
                            'price': level.price,
                            'strength': getattr(level, 'strength', 0.5),
                            'level_type': getattr(level, 'type', 'support'),
                            'touch_count': getattr(level, 'touch_count', 2),
                            'first_touch': getattr(level, 'first_touch', datetime.now() - timedelta(days=30)),
                            'last_touch': getattr(level, 'last_touch', datetime.now() - timedelta(days=1))
                        }
                        levels_dict.append(level_dict)
                    
                # Log every 10th level for progress tracking
                if (i + 1) % 10 == 0 or i == len(all_levels) - 1:
                    self.logger.info(f'   📊 Prepared {i + 1}/{len(all_levels)} levels ({(i + 1)/len(all_levels)*100:.1f}%)')
            
            conversion_time = time.time() - conversion_start
            self.logger.info(f'✅ Level preparation completed in {conversion_time:.2f} seconds')
            self.logger.info(f'📊 Prepared {len(levels_dict)} levels for clustering')
            
            # Perform clustering
            self.logger.info('🔄 Starting SR level clustering...')
            clustering_start = time.time()
            
            try:
                clustered_results = clustering.cluster_levels(levels_dict)
                clustering_time = time.time() - clustering_start
                self.logger.info(f'✅ SR level clustering completed in {clustering_time:.2f} seconds')
                
                # Process clustering results
                if isinstance(clustered_results, dict):
                    clusters = clustered_results.get('clusters', [])
                    cluster_metadata = clustered_results.get('metadata', {})
                else:
                    clusters = clustered_results if isinstance(clustered_results, list) else []
                    cluster_metadata = {}
                
                self.logger.info(f'📊 Clustering results: {len(clusters)} clusters generated')
                
                # Analyze cluster quality
                cluster_analysis = self._analyze_cluster_quality(clusters)
                
                total_clustering_time = time.time() - clustering_start_time
                
                self.logger.info('🚀 ===== SR CLUSTERING SYSTEM COMPLETED =====')
                self.logger.info(f'✅ Total clustering time: {total_clustering_time:.2f} seconds')
                self.logger.info(f'📊 Final results: {len(clusters)} clusters with {cluster_analysis["total_levels"]} total levels')
                
                return {
                    'clusters': clusters,
                    'cluster_metadata': cluster_metadata,
                    'cluster_analysis': cluster_analysis,
                    'clustering_time': total_clustering_time,
                    'clustering_config': clustering_config.__dict__,
                    'input_levels_count': len(levels_dict),
                    'output_clusters_count': len(clusters)
                }
                
            except Exception as clustering_error:
                self.logger.error(f'❌ Clustering process failed: {clustering_error}')
                self.logger.error(f'❌ Clustering error details: {traceback.format_exc()}')
                
                # Return fallback results
                return self._get_fallback_clustering_results(levels_dict, clustering_start_time)

        except Exception as e:
            clustering_time = time.time() - clustering_start_time
            self.logger.error(f'❌ SR Clustering failed after {clustering_time:.2f} seconds: {e}')
            self.logger.error(f'❌ Error details: {traceback.format_exc()}')
            raise

    def _analyze_cluster_quality(self, clusters: List[Any]) -> Dict[str, Any]:
        """Analyze the quality of generated clusters."""
        self.logger.info('📊 Analyzing cluster quality...')
        
        if not clusters:
            return {
                'total_clusters': 0,
                'total_levels': 0,
                'average_cluster_size': 0,
                'cluster_size_distribution': {},
                'quality_metrics': {}
            }
        
        total_levels = 0
        cluster_sizes = []
        
        for i, cluster in enumerate(clusters):
            if isinstance(cluster, dict):
                cluster_size = len(cluster.get('levels', []))
            elif hasattr(cluster, 'levels'):
                cluster_size = len(cluster.levels)
            else:
                cluster_size = 1  # Single level cluster
            
            cluster_sizes.append(cluster_size)
            total_levels += cluster_size
        
        # Calculate statistics
        average_cluster_size = total_levels / len(clusters) if clusters else 0
        
        # Size distribution
        size_distribution = {}
        for size in cluster_sizes:
            size_distribution[size] = size_distribution.get(size, 0) + 1
        
        # Quality metrics
        quality_metrics = {
            'cluster_diversity': len(set(cluster_sizes)) / len(clusters) if clusters else 0,
            'size_consistency': 1 - (np.std(cluster_sizes) / np.mean(cluster_sizes)) if cluster_sizes and np.mean(cluster_sizes) > 0 else 0,
            'coverage_ratio': total_levels / max(total_levels, 1)
        }
        
        self.logger.info(f'📊 Cluster analysis completed:')
        self.logger.info(f'   • Total clusters: {len(clusters)}')
        self.logger.info(f'   • Total levels: {total_levels}')
        self.logger.info(f'   • Average cluster size: {average_cluster_size:.2f}')
        self.logger.info(f'   • Cluster diversity: {quality_metrics["cluster_diversity"]:.3f}')
        
        return {
            'total_clusters': len(clusters),
            'total_levels': total_levels,
            'average_cluster_size': average_cluster_size,
            'cluster_size_distribution': size_distribution,
            'quality_metrics': quality_metrics
        }

    def _get_fallback_clustering_results(self, levels_dict: List[Dict], start_time: float) -> Dict[str, Any]:
        """Get fallback clustering results when clustering fails."""
        self.logger.warning('⚠️ Using fallback clustering results due to clustering failure')
        
        # Create simple single-level clusters as fallback
        fallback_clusters = []
        for level in levels_dict:
            fallback_cluster = {
                'levels': [level],
                'center_price': level.get('price', 0),
                'cluster_strength': level.get('strength', 0.5),
                'cluster_type': level.get('level_type', 'support'),
                'fallback': True
            }
            fallback_clusters.append(fallback_cluster)
        
        clustering_time = time.time() - start_time
        
        return {
            'clusters': fallback_clusters,
            'cluster_metadata': {'fallback': True, 'method': 'single_level_clusters'},
            'cluster_analysis': {
                'total_clusters': len(fallback_clusters),
                'total_levels': len(levels_dict),
                'average_cluster_size': 1.0,
                'cluster_size_distribution': {1: len(fallback_clusters)},
                'quality_metrics': {'fallback': True}
            },
            'clustering_time': clustering_time,
            'clustering_config': {},
            'input_levels_count': len(levels_dict),
            'output_clusters_count': len(fallback_clusters),
            'fallback': True
        }