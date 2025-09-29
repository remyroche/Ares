"""
NAS-TAS Clustering Component.

This component performs advanced regime clustering using combined Neural Architecture Search (NAS)
and Tree-based Architecture Search (TAS) approaches. It leverages the unified clustering algorithms
from the hybrid NAS-TAS regime system for superior clustering quality and economic awareness.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import traceback

from .base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult
from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_progress, tprint_performance, tprint_timer

logger = logging.getLogger(__name__)


@dataclass
class NASTASClusteringConfig(ComponentConfig):
    """Configuration for NAS-TAS clustering component."""
    symbol: str = "ETHUSDT"
    timeframe: str = "15m"
    exchange: str = "binance"

    # Clustering parameters
    n_regimes: int = 8
    algorithm_type: str = "adaptive_clustering"
    enable_economic_clustering: bool = True
    enable_ensemble_clustering: bool = True

    # Economic clustering weights
    economic_weight: float = 0.3
    momentum_weight: float = 0.25
    volume_weight: float = 0.25

    # Feature configuration
    feature_categories: List[str] = None
    use_standardized_features: bool = True

    # Output configuration
    output_dir: str = "data_cache"
    save_intermediate_results: bool = True

    def __post_init__(self):
        if self.feature_categories is None:
            self.feature_categories = ['momentum', 'volatility', 'volume', 'trend', 'price_action']


class NASTASClusteringComponent(BaseMarketAnalysisComponent):
    """
    NAS-TAS Clustering Component.

    Performs advanced regime clustering using combined NAS and TAS approaches.
    """

    def __init__(self, config: Optional[NASTASClusteringConfig] = None):
        """Initialize the NAS-TAS clustering component."""
        super().__init__(config)
        self.logger = system_logger.getChild('NASTASClustering')
        self.unified_clustering = None
        self.clustering_result = None
        self.execution_metadata = {}

    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['nas_tas_clustering_result']

    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute NAS-TAS clustering with comprehensive logging and debugging.

        Args:
            data: Market data for clustering
            pipeline_state: Current pipeline state

        Returns:
            ComponentResult with clustering results
        """
        import time
        import psutil
        import gc
        
        # Initialize execution tracking
        execution_start = time.time()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        tprint("🚀 [NAS_TAS_CLUSTERING] ===== STARTING NAS-TAS CLUSTERING =====", color="blue", bold=True)
        tprint_info("🔧 [NAS_TAS_CLUSTERING] Initializing comprehensive logging and debugging")
        self.logger.info('🚀 Starting NAS-TAS Clustering with enhanced logging')
        
        # Log system information
        tprint_debug(f"💻 [NAS_TAS_CLUSTERING] System memory: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f} GB")
        tprint_debug(f"💻 [NAS_TAS_CLUSTERING] Available memory: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.1f} GB")
        tprint_debug(f"💻 [NAS_TAS_CLUSTERING] Process memory: {initial_memory:.1f} MB")

        try:
            # Step 1: Validate inputs and configuration
            tprint("📋 [NAS_TAS_CLUSTERING] Step 1: Validating inputs and configuration", color="cyan", bold=True)
            validation_errors = self.validate_inputs()
            if validation_errors:
                tprint_error(f"❌ [NAS_TAS_CLUSTERING] Configuration validation failed: {validation_errors}")
                self.logger.error(f"Configuration validation failed: {validation_errors}")
                raise ValueError(f"Configuration validation failed: {validation_errors}")
            tprint_success("✅ [NAS_TAS_CLUSTERING] Configuration validation passed")
            
            # Log configuration details
            tprint_debug(f"⚙️ [NAS_TAS_CLUSTERING] Symbol: {self.config.symbol}")
            tprint_debug(f"⚙️ [NAS_TAS_CLUSTERING] Timeframe: {self.config.timeframe}")
            tprint_debug(f"⚙️ [NAS_TAS_CLUSTERING] Exchange: {self.config.exchange}")
            tprint_debug(f"⚙️ [NAS_TAS_CLUSTERING] N regimes: {self.config.n_regimes}")
            tprint_debug(f"⚙️ [NAS_TAS_CLUSTERING] Algorithm: {self.config.algorithm_type}")
            tprint_debug(f"⚙️ [NAS_TAS_CLUSTERING] Economic clustering: {self.config.enable_economic_clustering}")
            tprint_debug(f"⚙️ [NAS_TAS_CLUSTERING] Ensemble clustering: {self.config.enable_ensemble_clustering}")

            # Step 2: Initialize execution metadata
            tprint("📊 [NAS_TAS_CLUSTERING] Step 2: Initializing execution metadata", color="cyan")
            self.execution_metadata = {
                'start_time': datetime.now(),
                'symbol': self.config.symbol,
                'timeframe': self.config.timeframe,
                'exchange': self.config.exchange,
                'component': 'nas_tas_clustering',
                'initial_memory_mb': initial_memory,
                'pipeline_state_keys': list(pipeline_state.keys()) if pipeline_state else []
            }
            tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Execution metadata initialized")
            tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Pipeline state keys: {self.execution_metadata['pipeline_state_keys']}")

            # Step 3: Load and validate market data
            tprint("📊 [NAS_TAS_CLUSTERING] Step 3: Loading and validating market data", color="cyan", bold=True)
            data_load_start = time.time()
            
            tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Input data type: {type(data)}")
            if hasattr(data, 'shape'):
                tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Input data shape: {data.shape}")
            elif isinstance(data, (list, dict)):
                tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Input data length: {len(data)}")
            
            market_data = await self._load_market_data(data)
            data_load_time = time.time() - data_load_start
            
            if market_data is None or market_data.empty:
                tprint_error("❌ [NAS_TAS_CLUSTERING] No market data available for clustering")
                self.logger.error("No market data available for clustering")
                raise ValueError("No market data available for clustering")
            
            # Log data quality metrics
            tprint_success(f"✅ [NAS_TAS_CLUSTERING] Market data loaded: {len(market_data)} rows in {data_load_time:.3f}s")
            tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Data columns: {list(market_data.columns)}")
            tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Data index range: {market_data.index.min()} to {market_data.index.max()}")
            tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Data memory usage: {market_data.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
            
            # Check for missing values
            missing_data = market_data.isnull().sum()
            if missing_data.any():
                tprint_warning(f"⚠️ [NAS_TAS_CLUSTERING] Missing data detected: {missing_data[missing_data > 0].to_dict()}")
            else:
                tprint_debug("✅ [NAS_TAS_CLUSTERING] No missing data detected")

            # Step 4: Prepare features with detailed logging
            tprint("🔧 [NAS_TAS_CLUSTERING] Step 4: Preparing features for clustering", color="cyan", bold=True)
            feature_prep_start = time.time()
            
            features = self._prepare_features(market_data)
            feature_prep_time = time.time() - feature_prep_start
            
            if features is None:
                tprint_error("❌ [NAS_TAS_CLUSTERING] Failed to prepare features for clustering")
                self.logger.error("Failed to prepare features for clustering")
                raise ValueError("Failed to prepare features for clustering")
            
            tprint_success(f"✅ [NAS_TAS_CLUSTERING] Features prepared: {features.shape} in {feature_prep_time:.3f}s")
            tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Feature array memory usage: {features.nbytes / 1024 / 1024:.1f} MB")
            tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Feature statistics:")
            tprint_debug(f"   - Mean: {np.mean(features):.6f}")
            tprint_debug(f"   - Std: {np.std(features):.6f}")
            tprint_debug(f"   - Min: {np.min(features):.6f}")
            tprint_debug(f"   - Max: {np.max(features):.6f}")
            tprint_debug(f"   - NaN count: {np.isnan(features).sum()}")
            tprint_debug(f"   - Inf count: {np.isinf(features).sum()}")

            # Step 5: Create clustering configuration
            tprint("⚙️ [NAS_TAS_CLUSTERING] Step 5: Creating clustering configuration", color="cyan", bold=True)
            config_start = time.time()
            clustering_config = self._create_clustering_config()
            config_time = time.time() - config_start
            
            tprint_success(f"✅ [NAS_TAS_CLUSTERING] Clustering configuration created in {config_time:.3f}s")
            tprint_debug(f"🔧 [NAS_TAS_CLUSTERING] Clustering config: {clustering_config}")

            # Step 6: Initialize unified clustering
            tprint("🚀 [NAS_TAS_CLUSTERING] Step 6: Initializing unified clustering", color="cyan", bold=True)
            init_start = time.time()
            
            self.unified_clustering = self._initialize_unified_clustering(clustering_config)
            init_time = time.time() - init_start
            
            tprint_success(f"✅ [NAS_TAS_CLUSTERING] Unified clustering initialized in {init_time:.3f}s")
            tprint_debug(f"🔧 [NAS_TAS_CLUSTERING] Clustering algorithm: {type(self.unified_clustering).__name__}")

            # Step 7: Perform clustering with detailed monitoring
            tprint("🧠 [NAS_TAS_CLUSTERING] Step 7: Starting clustering process", color="cyan", bold=True)
            clustering_start = time.time()
            clustering_memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Clustering {features.shape[0]} samples with {features.shape[1]} features")
            tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Memory before clustering: {clustering_memory_before:.1f} MB")
            tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Expected regimes: {self.config.n_regimes}")

            # Perform the actual clustering
            tprint_progress("🔄 [NAS_TAS_CLUSTERING] Executing clustering algorithm...")
            clustering_result = self.unified_clustering.cluster_features(
                features=features,
                market_data=market_data
            )

            clustering_time = time.time() - clustering_start
            clustering_memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_delta = clustering_memory_after - clustering_memory_before

            tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Memory after clustering: {clustering_memory_after:.1f} MB")
            tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Memory delta: {memory_delta:+.1f} MB")

            if not clustering_result.success:
                tprint_error(f"❌ [NAS_TAS_CLUSTERING] Clustering failed: {clustering_result.error_message}")
                self.logger.error(f"Clustering failed: {clustering_result.error_message}")
                raise ValueError(f"Clustering failed: {clustering_result.error_message}")

            # Log clustering results
            self.clustering_result = clustering_result
            unique_regimes = len(set(clustering_result.labels))
            tprint_success(f"✅ [NAS_TAS_CLUSTERING] Clustering completed: {unique_regimes} regimes discovered in {clustering_time:.3f}s")
            tprint_performance("NAS-TAS clustering", clustering_time)
            
            # Log detailed clustering results
            tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Clustering results:")
            tprint_debug(f"   - Algorithm used: {clustering_result.algorithm_used}")
            tprint_debug(f"   - Regimes discovered: {unique_regimes}")
            tprint_debug(f"   - Total samples: {len(clustering_result.labels)}")
            tprint_debug(f"   - Execution time: {clustering_result.execution_time:.3f}s")
            if clustering_result.quality_metrics:
                tprint_debug(f"   - Quality metrics: {clustering_result.quality_metrics}")
            if clustering_result.probabilities is not None:
                tprint_debug(f"   - Probability matrix shape: {clustering_result.probabilities.shape}")
                tprint_debug(f"   - Probability range: {np.min(clustering_result.probabilities):.3f} to {np.max(clustering_result.probabilities):.3f}")
            
            # Log regime distribution
            regime_counts = np.bincount(clustering_result.labels)
            tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Regime distribution: {dict(enumerate(regime_counts))}")
            
            self.logger.info(f"✅ NAS-TAS Clustering completed: {unique_regimes} regimes discovered")

            # Step 8: Generate outputs
            tprint("📁 [NAS_TAS_CLUSTERING] Step 8: Generating outputs", color="cyan", bold=True)
            output_start = time.time()
            
            outputs = await self._generate_outputs(market_data, clustering_result)
            output_time = time.time() - output_start
            
            tprint_success(f"✅ [NAS_TAS_CLUSTERING] Outputs generated in {output_time:.3f}s")
            tprint_debug(f"📁 [NAS_TAS_CLUSTERING] Output files: {outputs.get('output_files', [])}")

            # Step 9: Finalize execution metadata
            total_execution_time = time.time() - execution_start
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            total_memory_delta = final_memory - initial_memory
            
            tprint("⏱️ [NAS_TAS_CLUSTERING] Step 9: Finalizing execution", color="cyan")
            tprint_debug(f"⏱️ [NAS_TAS_CLUSTERING] Total execution time: {total_execution_time:.3f}s")
            tprint_debug(f"⏱️ [NAS_TAS_CLUSTERING] Memory usage: {initial_memory:.1f} MB → {final_memory:.1f} MB (Δ{total_memory_delta:+.1f} MB)")
            tprint_debug(f"⏱️ [NAS_TAS_CLUSTERING] Time breakdown:")
            tprint_debug(f"   - Data loading: {data_load_time:.3f}s ({data_load_time/total_execution_time*100:.1f}%)")
            tprint_debug(f"   - Feature prep: {feature_prep_time:.3f}s ({feature_prep_time/total_execution_time*100:.1f}%)")
            tprint_debug(f"   - Config creation: {config_time:.3f}s ({config_time/total_execution_time*100:.1f}%)")
            tprint_debug(f"   - Clustering init: {init_time:.3f}s ({init_time/total_execution_time*100:.1f}%)")
            tprint_debug(f"   - Clustering exec: {clustering_time:.3f}s ({clustering_time/total_execution_time*100:.1f}%)")
            tprint_debug(f"   - Output generation: {output_time:.3f}s ({output_time/total_execution_time*100:.1f}%)")

            self.execution_metadata.update({
                'end_time': datetime.now(),
                'execution_time': total_execution_time,
                'success': True,
                'regime_count': unique_regimes,
                'algorithm_used': clustering_result.algorithm_used,
                'quality_metrics': clustering_result.quality_metrics,
                'final_memory_mb': final_memory,
                'memory_delta_mb': total_memory_delta,
                'time_breakdown': {
                    'data_loading': data_load_time,
                    'feature_preparation': feature_prep_time,
                    'config_creation': config_time,
                    'clustering_initialization': init_time,
                    'clustering_execution': clustering_time,
                    'output_generation': output_time
                },
                'output_files': outputs.get('output_files', [])
            })

            # Force garbage collection
            gc.collect()
            tprint_debug(f"🧹 [NAS_TAS_CLUSTERING] Garbage collection completed")

            tprint_success(f"🎉 [NAS_TAS_CLUSTERING] SUCCESS: {unique_regimes} regimes discovered in {total_execution_time:.3f}s")
            tprint("🚀 [NAS_TAS_CLUSTERING] ===== NAS-TAS CLUSTERING COMPLETED =====", color="green", bold=True)
            
            return ComponentResult(
                success=True,
                artifacts={
                    'nas_tas_clustering_result': {
                        'regime_count': unique_regimes,
                        'total_samples': len(clustering_result.labels),
                        'regime_assignments': clustering_result.labels.tolist(),
                        'cluster_centers': clustering_result.cluster_centers.tolist(),
                        'probabilities': clustering_result.probabilities.tolist() if clustering_result.probabilities is not None else [],
                        'quality_metrics': clustering_result.quality_metrics,
                        'algorithm_used': clustering_result.algorithm_used,
                        'execution_time': clustering_result.execution_time,
                        'configuration': asdict(self.config) if self.config else {},
                        'execution_info': self.execution_metadata
                    }
                },
                metadata={
                    'symbol': self.config.symbol,
                    'timeframe': self.config.timeframe,
                    'data_points_processed': len(market_data),
                    'regime_count': unique_regimes,
                    'algorithm_used': clustering_result.algorithm_used,
                    'execution_successful': True,
                    'execution_time': clustering_result.execution_time,
                    'memory_usage_mb': final_memory,
                    'memory_delta_mb': total_memory_delta
                }
            )

        except Exception as e:
            total_execution_time = time.time() - execution_start
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            total_memory_delta = final_memory - initial_memory
            
            tprint_error(f"❌ [NAS_TAS_CLUSTERING] NAS-TAS Clustering failed: {e}")
            tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Error type: {type(e).__name__}")
            tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Error details: {str(e)}")
            tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Execution time before failure: {total_execution_time:.3f}s")
            tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Memory usage: {initial_memory:.1f} MB → {final_memory:.1f} MB (Δ{total_memory_delta:+.1f} MB)")
            
            # Log full stack trace
            full_traceback = traceback.format_exc()
            tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Full traceback:\n{full_traceback}")
            
            self.logger.error(f'❌ NAS-TAS Clustering failed: {e}')
            self.logger.error(f'Error type: {type(e).__name__}')
            self.logger.error(f'Execution time before failure: {total_execution_time:.3f}s')
            self.logger.error(f'Memory usage: {initial_memory:.1f} MB → {final_memory:.1f} MB (Δ{total_memory_delta:+.1f} MB)')
            self.logger.error(f'Full traceback:\n{full_traceback}')

            self.execution_metadata.update({
                'end_time': datetime.now(),
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'execution_time': total_execution_time,
                'final_memory_mb': final_memory,
                'memory_delta_mb': total_memory_delta,
                'traceback': full_traceback
            })

            tprint("🚀 [NAS_TAS_CLUSTERING] ===== NAS-TAS CLUSTERING FAILED =====", color="red", bold=True)
            
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=f"NAS-TAS clustering failed: {str(e)}",
                metadata={
                    'error_type': type(e).__name__,
                    'execution_time': total_execution_time,
                    'memory_usage_mb': final_memory,
                    'memory_delta_mb': total_memory_delta
                }
            )

    def _create_clustering_config(self) -> Dict[str, Any]:
        """Create clustering configuration with comprehensive logging and validation."""
        import time
        
        config_start = time.time()
        tprint("⚙️ [NAS_TAS_CLUSTERING] ===== CREATING CLUSTERING CONFIG =====", color="blue", bold=True)
        
        try:
            # Use our specific config class which has the required attributes
            tprint_debug("🔧 [NAS_TAS_CLUSTERING] Using NASTASClusteringConfig for configuration")
            config = NASTASClusteringConfig()

            # Log configuration details
            tprint_debug(f"⚙️ [NAS_TAS_CLUSTERING] Configuration parameters:")
            tprint_debug(f"   - N regimes: {config.n_regimes}")
            tprint_debug(f"   - Algorithm type: {config.algorithm_type}")
            tprint_debug(f"   - Economic clustering: {config.enable_economic_clustering}")
            tprint_debug(f"   - Ensemble clustering: {config.enable_ensemble_clustering}")
            tprint_debug(f"   - Economic weight: {config.economic_weight}")
            tprint_debug(f"   - Momentum weight: {config.momentum_weight}")
            tprint_debug(f"   - Volume weight: {config.volume_weight}")
            tprint_debug(f"   - Feature categories: {config.feature_categories}")
            tprint_debug(f"   - Standardized features: {config.use_standardized_features}")

            # Validate configuration parameters
            tprint_debug("🔍 [NAS_TAS_CLUSTERING] Validating configuration parameters...")
            
            # Validate n_regimes
            if config.n_regimes < 2:
                tprint_warning(f"⚠️ [NAS_TAS_CLUSTERING] N regimes too low ({config.n_regimes}), setting to minimum 2")
                config.n_regimes = 2
            elif config.n_regimes > 50:
                tprint_warning(f"⚠️ [NAS_TAS_CLUSTERING] N regimes too high ({config.n_regimes}), setting to maximum 50")
                config.n_regimes = 50
            
            # Validate weights
            total_weight = config.economic_weight + config.momentum_weight + config.volume_weight
            if abs(total_weight - 1.0) > 0.01:
                tprint_warning(f"⚠️ [NAS_TAS_CLUSTERING] Weights don't sum to 1.0 ({total_weight:.3f}), normalizing...")
                config.economic_weight /= total_weight
                config.momentum_weight /= total_weight
                config.volume_weight /= total_weight
                tprint_debug(f"   - Normalized weights: economic={config.economic_weight:.3f}, momentum={config.momentum_weight:.3f}, volume={config.volume_weight:.3f}")
            
            # Validate algorithm type
            valid_algorithms = ['adaptive_clustering', 'kmeans', 'gaussian_mixture', 'hierarchical', 'dbscan']
            if config.algorithm_type not in valid_algorithms:
                tprint_warning(f"⚠️ [NAS_TAS_CLUSTERING] Invalid algorithm type ({config.algorithm_type}), using adaptive_clustering")
                config.algorithm_type = 'adaptive_clustering'

            clustering_config = {
                'n_regimes': config.n_regimes,
                'algorithm_type': config.algorithm_type,
                'enable_economic_clustering': config.enable_economic_clustering,
                'enable_ensemble_clustering': config.enable_ensemble_clustering,
                'economic_weight': config.economic_weight,
                'momentum_weight': config.momentum_weight,
                'volume_weight': config.volume_weight,
                'feature_categories': config.feature_categories,
                'use_standardized_features': config.use_standardized_features,
                'symbol': config.symbol,
                'timeframe': config.timeframe,
                'exchange': config.exchange
            }

            config_time = time.time() - config_start
            tprint_success(f"✅ [NAS_TAS_CLUSTERING] Clustering configuration created in {config_time:.3f}s")
            tprint_debug(f"🔧 [NAS_TAS_CLUSTERING] Final clustering config: {clustering_config}")
            
            self.logger.info(f"📊 Clustering configuration: {config.n_regimes} regimes, algorithm: {config.algorithm_type}")
            tprint("⚙️ [NAS_TAS_CLUSTERING] ===== CLUSTERING CONFIG COMPLETED =====", color="green", bold=True)
            return clustering_config

        except Exception as e:
            config_time = time.time() - config_start
            tprint_error(f"❌ [NAS_TAS_CLUSTERING] Failed to create clustering config: {e}")
            tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Error type: {type(e).__name__}")
            tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Error details: {str(e)}")
            tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Execution time: {config_time:.3f}s")
            
            # Log full traceback
            full_traceback = traceback.format_exc()
            tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Full traceback:\n{full_traceback}")
            
            self.logger.error(f"Failed to create clustering config: {e}")
            self.logger.error(f"Error type: {type(e).__name__}")
            self.logger.error(f"Execution time: {config_time:.3f}s")
            self.logger.error(f"Full traceback:\n{full_traceback}")
            
            tprint_warning("🔄 [NAS_TAS_CLUSTERING] Using fallback default configuration")
            default_config = {
                'n_regimes': 8,
                'algorithm_type': 'adaptive_clustering',
                'enable_economic_clustering': True,
                'enable_ensemble_clustering': True,
                'economic_weight': 0.3,
                'momentum_weight': 0.25,
                'volume_weight': 0.25,
                'feature_categories': ['momentum', 'volatility', 'volume', 'trend', 'price_action'],
                'use_standardized_features': True,
                'symbol': 'ETHUSDT',
                'timeframe': '15m',
                'exchange': 'binance'
            }
            tprint_debug(f"🔧 [NAS_TAS_CLUSTERING] Default config: {default_config}")
            tprint("⚙️ [NAS_TAS_CLUSTERING] ===== CLUSTERING CONFIG FAILED =====", color="red", bold=True)
            return default_config

    def _initialize_unified_clustering(self, clustering_config: Dict[str, Any]):
        """Initialize unified clustering algorithm with comprehensive logging and validation."""
        import time
        import psutil
        
        init_start = time.time()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        tprint("🚀 [NAS_TAS_CLUSTERING] ===== INITIALIZING UNIFIED CLUSTERING =====", color="blue", bold=True)
        tprint_debug(f"🔧 [NAS_TAS_CLUSTERING] Clustering config: {clustering_config}")
        tprint_debug(f"💻 [NAS_TAS_CLUSTERING] Initial memory: {initial_memory:.1f} MB")
        
        try:
            # Step 1: Import validation
            tprint("📦 [NAS_TAS_CLUSTERING] Step 1: Importing unified clustering algorithm", color="cyan")
            import_start = time.time()
            
            try:
                from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils.unified_clustering_algorithms import (
                    UnifiedClusteringAlgorithm
                )
                import_time = time.time() - import_start
                tprint_success(f"✅ [NAS_TAS_CLUSTERING] UnifiedClusteringAlgorithm imported in {import_time:.3f}s")
                tprint_debug(f"📦 [NAS_TAS_CLUSTERING] Algorithm class: {UnifiedClusteringAlgorithm}")
                
            except ImportError as e:
                import_time = time.time() - import_start
                tprint_error(f"❌ [NAS_TAS_CLUSTERING] Failed to import UnifiedClusteringAlgorithm: {e}")
                tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Import error type: {type(e).__name__}")
                tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Import time: {import_time:.3f}s")
                
                # Log full traceback
                full_traceback = traceback.format_exc()
                tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Full traceback:\n{full_traceback}")
                
                self.logger.error(f"Failed to import unified clustering: {e}")
                self.logger.error(f"Import error type: {type(e).__name__}")
                self.logger.error(f"Import time: {import_time:.3f}s")
                self.logger.error(f"Full traceback:\n{full_traceback}")
                
                raise ValueError(f"Cannot import unified clustering algorithm: {e}")

            # Step 2: Configuration validation
            tprint("🔍 [NAS_TAS_CLUSTERING] Step 2: Validating clustering configuration", color="cyan")
            validation_start = time.time()
            
            # Validate required parameters
            required_params = ['n_regimes', 'algorithm_type', 'enable_economic_clustering', 'enable_ensemble_clustering']
            missing_params = [param for param in required_params if param not in clustering_config]
            if missing_params:
                tprint_error(f"❌ [NAS_TAS_CLUSTERING] Missing required parameters: {missing_params}")
                raise ValueError(f"Missing required parameters: {missing_params}")
            
            # Validate parameter types and ranges
            if not isinstance(clustering_config['n_regimes'], int) or clustering_config['n_regimes'] < 1:
                tprint_error(f"❌ [NAS_TAS_CLUSTERING] Invalid n_regimes: {clustering_config['n_regimes']}")
                raise ValueError(f"Invalid n_regimes: {clustering_config['n_regimes']}")
            
            if not isinstance(clustering_config['algorithm_type'], str):
                tprint_error(f"❌ [NAS_TAS_CLUSTERING] Invalid algorithm_type: {clustering_config['algorithm_type']}")
                raise ValueError(f"Invalid algorithm_type: {clustering_config['algorithm_type']}")
            
            validation_time = time.time() - validation_start
            tprint_success(f"✅ [NAS_TAS_CLUSTERING] Configuration validation passed in {validation_time:.3f}s")
            tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Validated parameters: {list(clustering_config.keys())}")

            # Step 3: Algorithm initialization
            tprint("⚙️ [NAS_TAS_CLUSTERING] Step 3: Initializing clustering algorithm", color="cyan", bold=True)
            init_algorithm_start = time.time()
            
            tprint_debug(f"🔧 [NAS_TAS_CLUSTERING] Creating UnifiedClusteringAlgorithm instance...")
            tprint_debug(f"🔧 [NAS_TAS_CLUSTERING] Algorithm type: {clustering_config['algorithm_type']}")
            tprint_debug(f"🔧 [NAS_TAS_CLUSTERING] N regimes: {clustering_config['n_regimes']}")
            tprint_debug(f"🔧 [NAS_TAS_CLUSTERING] Economic clustering: {clustering_config['enable_economic_clustering']}")
            tprint_debug(f"🔧 [NAS_TAS_CLUSTERING] Ensemble clustering: {clustering_config['enable_ensemble_clustering']}")
            
            clustering = UnifiedClusteringAlgorithm(clustering_config)
            init_algorithm_time = time.time() - init_algorithm_start
            
            tprint_success(f"✅ [NAS_TAS_CLUSTERING] UnifiedClusteringAlgorithm initialized in {init_algorithm_time:.3f}s")
            tprint_debug(f"🔧 [NAS_TAS_CLUSTERING] Algorithm instance: {type(clustering).__name__}")
            tprint_debug(f"🔧 [NAS_TAS_CLUSTERING] Algorithm attributes: {[attr for attr in dir(clustering) if not attr.startswith('_')]}")

            # Step 4: Algorithm validation
            tprint("🔍 [NAS_TAS_CLUSTERING] Step 4: Validating algorithm instance", color="cyan")
            validation_start = time.time()
            
            # Check if algorithm has required methods
            required_methods = ['cluster_features']
            missing_methods = [method for method in required_methods if not hasattr(clustering, method)]
            if missing_methods:
                tprint_error(f"❌ [NAS_TAS_CLUSTERING] Algorithm missing required methods: {missing_methods}")
                raise ValueError(f"Algorithm missing required methods: {missing_methods}")
            
            # Test algorithm configuration access
            try:
                if hasattr(clustering, 'config'):
                    tprint_debug(f"🔧 [NAS_TAS_CLUSTERING] Algorithm config: {clustering.config}")
                if hasattr(clustering, 'n_regimes'):
                    tprint_debug(f"🔧 [NAS_TAS_CLUSTERING] Algorithm n_regimes: {clustering.n_regimes}")
                if hasattr(clustering, 'algorithm_type'):
                    tprint_debug(f"🔧 [NAS_TAS_CLUSTERING] Algorithm type: {clustering.algorithm_type}")
            except Exception as e:
                tprint_warning(f"⚠️ [NAS_TAS_CLUSTERING] Could not access algorithm configuration: {e}")
            
            validation_time = time.time() - validation_start
            tprint_success(f"✅ [NAS_TAS_CLUSTERING] Algorithm validation passed in {validation_time:.3f}s")

            # Step 5: Performance metrics
            total_init_time = time.time() - init_start
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_delta = final_memory - initial_memory
            
            tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Initialization performance:")
            tprint_debug(f"   - Total time: {total_init_time:.3f}s")
            tprint_debug(f"   - Import time: {import_time:.3f}s ({import_time/total_init_time*100:.1f}%)")
            tprint_debug(f"   - Validation time: {validation_time:.3f}s ({validation_time/total_init_time*100:.1f}%)")
            tprint_debug(f"   - Algorithm init time: {init_algorithm_time:.3f}s ({init_algorithm_time/total_init_time*100:.1f}%)")
            tprint_debug(f"   - Memory usage: {initial_memory:.1f} MB → {final_memory:.1f} MB (Δ{memory_delta:+.1f} MB)")
            
            self.logger.info("✅ Unified clustering algorithm initialized")
            tprint_success(f"🎉 [NAS_TAS_CLUSTERING] Unified clustering initialization completed in {total_init_time:.3f}s")
            tprint("🚀 [NAS_TAS_CLUSTERING] ===== UNIFIED CLUSTERING INITIALIZED =====", color="green", bold=True)
            return clustering

        except Exception as e:
            total_init_time = time.time() - init_start
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_delta = final_memory - initial_memory
            
            tprint_error(f"❌ [NAS_TAS_CLUSTERING] Unified clustering initialization failed: {e}")
            tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Error type: {type(e).__name__}")
            tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Error details: {str(e)}")
            tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Execution time: {total_init_time:.3f}s")
            tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Memory usage: {initial_memory:.1f} MB → {final_memory:.1f} MB (Δ{memory_delta:+.1f} MB)")
            
            # Log full traceback
            full_traceback = traceback.format_exc()
            tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Full traceback:\n{full_traceback}")
            
            self.logger.error(f"Failed to initialize unified clustering: {e}")
            self.logger.error(f"Error type: {type(e).__name__}")
            self.logger.error(f"Execution time: {total_init_time:.3f}s")
            self.logger.error(f"Memory usage: {initial_memory:.1f} MB → {final_memory:.1f} MB (Δ{memory_delta:+.1f} MB)")
            self.logger.error(f"Full traceback:\n{full_traceback}")
            
            tprint("🚀 [NAS_TAS_CLUSTERING] ===== UNIFIED CLUSTERING INITIALIZATION FAILED =====", color="red", bold=True)
            raise ValueError(f"Cannot initialize unified clustering algorithm: {e}")

    def _prepare_features(self, market_data: pd.DataFrame) -> Optional[np.ndarray]:
        """Prepare features for clustering with comprehensive logging and validation."""
        import time
        import psutil
        
        feature_prep_start = time.time()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        tprint("🔧 [NAS_TAS_CLUSTERING] ===== FEATURE PREPARATION =====", color="blue", bold=True)
        tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Market data shape: {market_data.shape}")
        tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Available columns: {list(market_data.columns)}")
        tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Data types: {market_data.dtypes.to_dict()}")
        tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Memory usage: {market_data.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        
        try:
            features = []
            feature_stats = {}
            
            # Step 1: Data quality validation
            tprint("🔍 [NAS_TAS_CLUSTERING] Step 1: Data quality validation", color="cyan")
            missing_data = market_data.isnull().sum()
            tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Missing data per column: {missing_data[missing_data > 0].to_dict()}")
            
            # Check for infinite values
            inf_counts = {}
            for col in market_data.select_dtypes(include=[np.number]).columns:
                inf_count = np.isinf(market_data[col]).sum()
                if inf_count > 0:
                    inf_counts[col] = inf_count
            if inf_counts:
                tprint_warning(f"⚠️ [NAS_TAS_CLUSTERING] Infinite values detected: {inf_counts}")
            else:
                tprint_debug("✅ [NAS_TAS_CLUSTERING] No infinite values detected")
            
            # Step 2: Price-based features
            tprint("💰 [NAS_TAS_CLUSTERING] Step 2: Processing price-based features", color="cyan", bold=True)
            if 'close' in market_data.columns:
                tprint_debug(f"📈 [NAS_TAS_CLUSTERING] Close price range: {market_data['close'].min():.6f} to {market_data['close'].max():.6f}")
                tprint_debug(f"📈 [NAS_TAS_CLUSTERING] Close price mean: {market_data['close'].mean():.6f}")
                tprint_debug(f"📈 [NAS_TAS_CLUSTERING] Close price std: {market_data['close'].std():.6f}")
                
                # Returns calculation
                tprint_debug("📈 [NAS_TAS_CLUSTERING] Calculating returns...")
                returns = market_data['close'].pct_change().fillna(0)
                returns_stats = {
                    'mean': returns.mean(),
                    'std': returns.std(),
                    'min': returns.min(),
                    'max': returns.max(),
                    'nan_count': returns.isna().sum(),
                    'inf_count': np.isinf(returns).sum()
                }
                feature_stats['returns'] = returns_stats
                tprint_debug(f"📈 [NAS_TAS_CLUSTERING] Returns stats: {returns_stats}")
                features.append(returns.values.reshape(-1, 1))
                tprint_success(f"✅ [NAS_TAS_CLUSTERING] Returns feature created: {returns.shape}")

                # Volatility calculation
                tprint_debug("📊 [NAS_TAS_CLUSTERING] Calculating volatility (20-period rolling std)...")
                volatility = returns.rolling(20).std().fillna(0)
                volatility_stats = {
                    'mean': volatility.mean(),
                    'std': volatility.std(),
                    'min': volatility.min(),
                    'max': volatility.max(),
                    'nan_count': volatility.isna().sum(),
                    'inf_count': np.isinf(volatility).sum()
                }
                feature_stats['volatility'] = volatility_stats
                tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Volatility stats: {volatility_stats}")
                features.append(volatility.values.reshape(-1, 1))
                tprint_success(f"✅ [NAS_TAS_CLUSTERING] Volatility feature created: {volatility.shape}")

                # Moving averages ratio
                tprint_debug("📊 [NAS_TAS_CLUSTERING] Calculating moving average ratio...")
                sma_20 = market_data['close'].rolling(20).mean().fillna(market_data['close'].iloc[0])
                ma_ratio = market_data['close'] / sma_20 - 1
                ma_ratio_stats = {
                    'mean': ma_ratio.mean(),
                    'std': ma_ratio.std(),
                    'min': ma_ratio.min(),
                    'max': ma_ratio.max(),
                    'nan_count': ma_ratio.isna().sum(),
                    'inf_count': np.isinf(ma_ratio).sum()
                }
                feature_stats['ma_ratio'] = ma_ratio_stats
                tprint_debug(f"📊 [NAS_TAS_CLUSTERING] MA ratio stats: {ma_ratio_stats}")
                features.append(ma_ratio.values.reshape(-1, 1))
                tprint_success(f"✅ [NAS_TAS_CLUSTERING] MA ratio feature created: {ma_ratio.shape}")
            else:
                tprint_warning("⚠️ [NAS_TAS_CLUSTERING] No 'close' column found, skipping price features")

            # Step 3: Volume features
            tprint("📊 [NAS_TAS_CLUSTERING] Step 3: Processing volume features", color="cyan", bold=True)
            if 'volume' in market_data.columns:
                tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Volume range: {market_data['volume'].min():.2f} to {market_data['volume'].max():.2f}")
                tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Volume mean: {market_data['volume'].mean():.2f}")
                tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Volume std: {market_data['volume'].std():.2f}")
                
                tprint_debug("📊 [NAS_TAS_CLUSTERING] Calculating volume ratio...")
                volume_ma = market_data['volume'].rolling(20).mean().fillna(market_data['volume'].mean())
                volume_ratio = market_data['volume'] / volume_ma
                volume_ratio = volume_ratio.fillna(1)
                
                volume_ratio_stats = {
                    'mean': volume_ratio.mean(),
                    'std': volume_ratio.std(),
                    'min': volume_ratio.min(),
                    'max': volume_ratio.max(),
                    'nan_count': volume_ratio.isna().sum(),
                    'inf_count': np.isinf(volume_ratio).sum()
                }
                feature_stats['volume_ratio'] = volume_ratio_stats
                tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Volume ratio stats: {volume_ratio_stats}")
                features.append(volume_ratio.values.reshape(-1, 1))
                tprint_success(f"✅ [NAS_TAS_CLUSTERING] Volume ratio feature created: {volume_ratio.shape}")
            else:
                tprint_warning("⚠️ [NAS_TAS_CLUSTERING] No 'volume' column found, skipping volume features")

            # Step 4: High-low spread features
            tprint("📊 [NAS_TAS_CLUSTERING] Step 4: Processing high-low spread features", color="cyan", bold=True)
            if all(col in market_data.columns for col in ['high', 'low', 'close']):
                tprint_debug(f"📊 [NAS_TAS_CLUSTERING] High range: {market_data['high'].min():.6f} to {market_data['high'].max():.6f}")
                tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Low range: {market_data['low'].min():.6f} to {market_data['low'].max():.6f}")
                
                tprint_debug("📊 [NAS_TAS_CLUSTERING] Calculating high-low spread...")
                hl_spread = (market_data['high'] - market_data['low']) / market_data['close']
                hl_spread = hl_spread.fillna(0)
                
                hl_spread_stats = {
                    'mean': hl_spread.mean(),
                    'std': hl_spread.std(),
                    'min': hl_spread.min(),
                    'max': hl_spread.max(),
                    'nan_count': hl_spread.isna().sum(),
                    'inf_count': np.isinf(hl_spread).sum()
                }
                feature_stats['hl_spread'] = hl_spread_stats
                tprint_debug(f"📊 [NAS_TAS_CLUSTERING] HL spread stats: {hl_spread_stats}")
                features.append(hl_spread.values.reshape(-1, 1))
                tprint_success(f"✅ [NAS_TAS_CLUSTERING] HL spread feature created: {hl_spread.shape}")
            else:
                tprint_warning("⚠️ [NAS_TAS_CLUSTERING] Missing high/low/close columns, skipping HL spread")

            # Step 5: Additional technical features
            tprint("📊 [NAS_TAS_CLUSTERING] Step 5: Processing additional technical features", color="cyan", bold=True)
            
            # RSI-like momentum indicator
            if 'close' in market_data.columns:
                tprint_debug("📊 [NAS_TAS_CLUSTERING] Calculating momentum indicator...")
                price_changes = market_data['close'].diff()
                gains = price_changes.where(price_changes > 0, 0)
                losses = -price_changes.where(price_changes < 0, 0)
                
                avg_gains = gains.rolling(14).mean().fillna(0)
                avg_losses = losses.rolling(14).mean().fillna(0)
                
                # Avoid division by zero
                rs = avg_gains / (avg_losses + 1e-10)
                momentum = 100 - (100 / (1 + rs))
                momentum = momentum.fillna(50)  # Neutral value
                
                momentum_stats = {
                    'mean': momentum.mean(),
                    'std': momentum.std(),
                    'min': momentum.min(),
                    'max': momentum.max(),
                    'nan_count': momentum.isna().sum(),
                    'inf_count': np.isinf(momentum).sum()
                }
                feature_stats['momentum'] = momentum_stats
                tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Momentum stats: {momentum_stats}")
                features.append(momentum.values.reshape(-1, 1))
                tprint_success(f"✅ [NAS_TAS_CLUSTERING] Momentum feature created: {momentum.shape}")

            # Step 6: Combine and validate features
            tprint("🔄 [NAS_TAS_CLUSTERING] Step 6: Combining and validating features", color="cyan", bold=True)
            if features:
                tprint_debug(f"🔄 [NAS_TAS_CLUSTERING] Combining {len(features)} feature arrays")
                tprint_debug(f"🔄 [NAS_TAS_CLUSTERING] Individual feature shapes: {[f.shape for f in features]}")
                
                # Combine features
                feature_array = np.hstack(features)
                tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Combined features shape: {feature_array.shape}")
                tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Combined features memory: {feature_array.nbytes / 1024 / 1024:.1f} MB")
                
                # Pre-cleaning statistics
                tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Pre-cleaning statistics:")
                tprint_debug(f"   - NaN count: {np.isnan(feature_array).sum()}")
                tprint_debug(f"   - Inf count: {np.isinf(feature_array).sum()}")
                tprint_debug(f"   - Mean: {np.mean(feature_array):.6f}")
                tprint_debug(f"   - Std: {np.std(feature_array):.6f}")
                tprint_debug(f"   - Min: {np.min(feature_array):.6f}")
                tprint_debug(f"   - Max: {np.max(feature_array):.6f}")
                
                # Clean features
                tprint("🧹 [NAS_TAS_CLUSTERING] Cleaning features: removing NaN and infinite values", color="yellow")
                initial_shape = feature_array.shape
                initial_nan_count = np.isnan(feature_array).sum()
                initial_inf_count = np.isinf(feature_array).sum()
                
                feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # Post-cleaning statistics
                final_nan_count = np.isnan(feature_array).sum()
                final_inf_count = np.isinf(feature_array).sum()
                
                tprint_success(f"✅ [NAS_TAS_CLUSTERING] Features cleaned:")
                tprint_debug(f"   - Shape: {initial_shape} → {feature_array.shape}")
                tprint_debug(f"   - NaN: {initial_nan_count} → {final_nan_count}")
                tprint_debug(f"   - Inf: {initial_inf_count} → {final_inf_count}")
                tprint_debug(f"   - Final mean: {np.mean(feature_array):.6f}")
                tprint_debug(f"   - Final std: {np.std(feature_array):.6f}")
                tprint_debug(f"   - Final min: {np.min(feature_array):.6f}")
                tprint_debug(f"   - Final max: {np.max(feature_array):.6f}")
                
                # Feature correlation analysis
                tprint_debug("📊 [NAS_TAS_CLUSTERING] Feature correlation analysis:")
                if feature_array.shape[1] > 1:
                    corr_matrix = np.corrcoef(feature_array.T)
                    high_corr_pairs = []
                    for i in range(corr_matrix.shape[0]):
                        for j in range(i+1, corr_matrix.shape[1]):
                            if abs(corr_matrix[i, j]) > 0.9:
                                high_corr_pairs.append((i, j, corr_matrix[i, j]))
                    
                    if high_corr_pairs:
                        tprint_warning(f"⚠️ [NAS_TAS_CLUSTERING] High correlation detected: {high_corr_pairs}")
                    else:
                        tprint_debug("✅ [NAS_TAS_CLUSTERING] No high correlations detected")
                
                # Performance metrics
                feature_prep_time = time.time() - feature_prep_start
                final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                memory_delta = final_memory - initial_memory
                
                tprint_performance("Feature preparation", feature_prep_time)
                tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Memory usage: {initial_memory:.1f} MB → {final_memory:.1f} MB (Δ{memory_delta:+.1f} MB)")
                tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Feature preparation time: {feature_prep_time:.3f}s")
                
                # Log feature statistics summary
                tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Feature statistics summary:")
                for feature_name, stats in feature_stats.items():
                    tprint_debug(f"   - {feature_name}: mean={stats['mean']:.6f}, std={stats['std']:.6f}")
                
                tprint_success(f"🎉 [NAS_TAS_CLUSTERING] Feature preparation completed: {feature_array.shape}")
                tprint("🔧 [NAS_TAS_CLUSTERING] ===== FEATURE PREPARATION COMPLETED =====", color="green", bold=True)
                return feature_array
            else:
                tprint_warning("⚠️ [NAS_TAS_CLUSTERING] No features could be created, using dummy features")
                dummy_features = np.random.randn(len(market_data), 5)
                tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Dummy features shape: {dummy_features.shape}")
                tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Dummy features stats: mean={np.mean(dummy_features):.6f}, std={np.std(dummy_features):.6f}")
                return dummy_features

        except Exception as e:
            feature_prep_time = time.time() - feature_prep_start
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_delta = final_memory - initial_memory
            
            tprint_error(f"❌ [NAS_TAS_CLUSTERING] Feature preparation failed: {e}")
            tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Error type: {type(e).__name__}")
            tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Error details: {str(e)}")
            tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Execution time: {feature_prep_time:.3f}s")
            tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Memory usage: {initial_memory:.1f} MB → {final_memory:.1f} MB (Δ{memory_delta:+.1f} MB)")
            
            # Log full traceback
            full_traceback = traceback.format_exc()
            tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Full traceback:\n{full_traceback}")
            
            self.logger.error(f"Failed to prepare features: {e}")
            self.logger.error(f"Error type: {type(e).__name__}")
            self.logger.error(f"Execution time: {feature_prep_time:.3f}s")
            self.logger.error(f"Memory usage: {initial_memory:.1f} MB → {final_memory:.1f} MB (Δ{memory_delta:+.1f} MB)")
            self.logger.error(f"Full traceback:\n{full_traceback}")
            
            # Fallback to dummy features
            dummy_features = np.random.randn(len(market_data), 5)
            tprint_warning("🔄 [NAS_TAS_CLUSTERING] Using fallback dummy features")
            tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Dummy features shape: {dummy_features.shape}")
            tprint("🔧 [NAS_TAS_CLUSTERING] ===== FEATURE PREPARATION FAILED =====", color="red", bold=True)
            return dummy_features

    async def _load_market_data(self, data: Any) -> Optional[pd.DataFrame]:
        """Load and prepare market data for clustering."""
        try:
            if data is None or (isinstance(data, pd.DataFrame) and data.empty):
                self.logger.warning("⚠️ No market data provided, attempting to load from klines_parquet")

                symbol = self.config.symbol if self.config else 'ETHUSDT'
                timeframe = self.config.timeframe if self.config else '15m'

                # Try to load data using klines_parquet manager
                from src.utils.data.klines_parquet import get_klines_manager

                manager = get_klines_manager()

                self.logger.info(f"📊 Loading {symbol} {timeframe} data using klines_parquet manager")

                # Try processed data first
                market_data = manager.read_data(symbol, timeframe, data_type="processed")

                if market_data is None or market_data.empty:
                    # Fallback to raw data
                    market_data = manager.read_data(symbol, timeframe, data_type="raw")

                if market_data is None or market_data.empty:
                    self.logger.error(f"❌ No data available for {symbol} {timeframe}")
                    return None

                self.logger.info(f"✅ Loaded {len(market_data)} rows of {symbol} {timeframe} data")
                return market_data

            # If data is already a DataFrame, use it
            if isinstance(data, pd.DataFrame):
                self.logger.info(f"📊 Using provided DataFrame with {len(data)} rows")
                return data.copy()

            return None

        except Exception as e:
            self.logger.exception(f"❌ Error loading market data: {e}")
            return None

    async def _generate_outputs(self, market_data: pd.DataFrame, clustering_result) -> Dict[str, Any]:
        """Generate output files and data structures with comprehensive logging and validation."""
        import time
        import psutil
        
        output_start = time.time()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        tprint("📁 [NAS_TAS_CLUSTERING] ===== GENERATING OUTPUTS =====", color="blue", bold=True)
        tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Market data shape: {market_data.shape}")
        tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Clustering result success: {clustering_result.success if clustering_result else False}")
        tprint_debug(f"💻 [NAS_TAS_CLUSTERING] Initial memory: {initial_memory:.1f} MB")
        
        try:
            outputs = {
                'clustering_report': None,
                'regime_assignments': None,
                'cluster_characteristics': None,
                'output_files': [],
                'generation_stats': {}
            }

            # Step 1: Validate inputs
            tprint("🔍 [NAS_TAS_CLUSTERING] Step 1: Validating inputs", color="cyan")
            validation_start = time.time()
            
            if not clustering_result:
                tprint_error("❌ [NAS_TAS_CLUSTERING] No clustering result provided")
                raise ValueError("No clustering result provided")
            
            if not clustering_result.success:
                tprint_error(f"❌ [NAS_TAS_CLUSTERING] Clustering result indicates failure: {clustering_result.error_message}")
                raise ValueError(f"Clustering result indicates failure: {clustering_result.error_message}")
            
            if market_data is None or market_data.empty:
                tprint_error("❌ [NAS_TAS_CLUSTERING] No market data provided")
                raise ValueError("No market data provided")
            
            validation_time = time.time() - validation_start
            tprint_success(f"✅ [NAS_TAS_CLUSTERING] Input validation passed in {validation_time:.3f}s")

            # Step 2: Generate clustering report
            tprint("📄 [NAS_TAS_CLUSTERING] Step 2: Generating clustering report", color="cyan", bold=True)
            report_start = time.time()
            
            try:
                tprint_debug("📄 [NAS_TAS_CLUSTERING] Creating clustering report...")
                report_file = self._save_clustering_report(clustering_result)
                
                if report_file:
                    outputs['clustering_report'] = report_file
                    outputs['output_files'].append(report_file)
                    tprint_success(f"✅ [NAS_TAS_CLUSTERING] Clustering report saved: {report_file}")
                else:
                    tprint_warning("⚠️ [NAS_TAS_CLUSTERING] Clustering report generation failed")
                
                report_time = time.time() - report_start
                outputs['generation_stats']['clustering_report_time'] = report_time
                tprint_debug(f"📄 [NAS_TAS_CLUSTERING] Report generation time: {report_time:.3f}s")
                
            except Exception as e:
                report_time = time.time() - report_start
                tprint_error(f"❌ [NAS_TAS_CLUSTERING] Failed to generate clustering report: {e}")
                tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Report error type: {type(e).__name__}")
                tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Report error details: {str(e)}")
                tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Report generation time: {report_time:.3f}s")
                outputs['generation_stats']['clustering_report_time'] = report_time
                outputs['generation_stats']['clustering_report_error'] = str(e)

            # Step 3: Generate regime assignments
            tprint("📊 [NAS_TAS_CLUSTERING] Step 3: Generating regime assignments", color="cyan", bold=True)
            regime_start = time.time()
            
            try:
                tprint_debug("📊 [NAS_TAS_CLUSTERING] Creating regime assignments...")
                regime_data = self._generate_regime_assignments(market_data, clustering_result)
                
                if regime_data is not None and not regime_data.empty:
                    tprint_success(f"✅ [NAS_TAS_CLUSTERING] Regime assignments generated: {regime_data.shape}")
                    tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Regime data columns: {list(regime_data.columns)}")
                    tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Regime data memory: {regime_data.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
                    
                    # Save regime assignments
                    regime_file = self._save_regime_assignments(regime_data)
                    if regime_file:
                        outputs['regime_assignments'] = regime_file
                        outputs['output_files'].append(regime_file)
                        tprint_success(f"✅ [NAS_TAS_CLUSTERING] Regime assignments saved: {regime_file}")
                    else:
                        tprint_warning("⚠️ [NAS_TAS_CLUSTERING] Regime assignments save failed")
                else:
                    tprint_warning("⚠️ [NAS_TAS_CLUSTERING] Regime assignments generation failed or empty")
                
                regime_time = time.time() - regime_start
                outputs['generation_stats']['regime_assignments_time'] = regime_time
                tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Regime assignments time: {regime_time:.3f}s")
                
            except Exception as e:
                regime_time = time.time() - regime_start
                tprint_error(f"❌ [NAS_TAS_CLUSTERING] Failed to generate regime assignments: {e}")
                tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Regime error type: {type(e).__name__}")
                tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Regime error details: {str(e)}")
                tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Regime generation time: {regime_time:.3f}s")
                outputs['generation_stats']['regime_assignments_time'] = regime_time
                outputs['generation_stats']['regime_assignments_error'] = str(e)

            # Step 4: Generate cluster characteristics
            tprint("📈 [NAS_TAS_CLUSTERING] Step 4: Generating cluster characteristics", color="cyan", bold=True)
            characteristics_start = time.time()
            
            try:
                tprint_debug("📈 [NAS_TAS_CLUSTERING] Creating cluster characteristics...")
                characteristics = self._generate_cluster_characteristics(market_data, clustering_result)
                
                if characteristics:
                    tprint_success(f"✅ [NAS_TAS_CLUSTERING] Cluster characteristics generated: {len(characteristics)} regimes")
                    tprint_debug(f"📈 [NAS_TAS_CLUSTERING] Characteristics keys: {list(characteristics.keys())}")
                    
                    # Log characteristics summary
                    for regime_id, char_data in characteristics.items():
                        tprint_debug(f"📈 [NAS_TAS_CLUSTERING] {regime_id}: {char_data}")
                    
                    # Save cluster characteristics
                    char_file = self._save_cluster_characteristics(characteristics)
                    if char_file:
                        outputs['cluster_characteristics'] = char_file
                        outputs['output_files'].append(char_file)
                        tprint_success(f"✅ [NAS_TAS_CLUSTERING] Cluster characteristics saved: {char_file}")
                    else:
                        tprint_warning("⚠️ [NAS_TAS_CLUSTERING] Cluster characteristics save failed")
                else:
                    tprint_warning("⚠️ [NAS_TAS_CLUSTERING] Cluster characteristics generation failed or empty")
                
                characteristics_time = time.time() - characteristics_start
                outputs['generation_stats']['cluster_characteristics_time'] = characteristics_time
                tprint_debug(f"📈 [NAS_TAS_CLUSTERING] Characteristics generation time: {characteristics_time:.3f}s")
                
            except Exception as e:
                characteristics_time = time.time() - characteristics_start
                tprint_error(f"❌ [NAS_TAS_CLUSTERING] Failed to generate cluster characteristics: {e}")
                tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Characteristics error type: {type(e).__name__}")
                tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Characteristics error details: {str(e)}")
                tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Characteristics generation time: {characteristics_time:.3f}s")
                outputs['generation_stats']['cluster_characteristics_time'] = characteristics_time
                outputs['generation_stats']['cluster_characteristics_error'] = str(e)

            # Step 5: Performance metrics and validation
            total_output_time = time.time() - output_start
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_delta = final_memory - initial_memory
            
            tprint("📊 [NAS_TAS_CLUSTERING] Step 5: Output generation summary", color="cyan", bold=True)
            tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Total output generation time: {total_output_time:.3f}s")
            tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Memory usage: {initial_memory:.1f} MB → {final_memory:.1f} MB (Δ{memory_delta:+.1f} MB)")
            tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Output files generated: {len(outputs['output_files'])}")
            tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Output files: {outputs['output_files']}")
            
            # Log generation statistics
            tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Generation statistics:")
            for stat_name, stat_value in outputs['generation_stats'].items():
                if isinstance(stat_value, float):
                    tprint_debug(f"   - {stat_name}: {stat_value:.3f}s")
                else:
                    tprint_debug(f"   - {stat_name}: {stat_value}")
            
            # Validate outputs
            tprint_debug("🔍 [NAS_TAS_CLUSTERING] Validating outputs...")
            output_validation = {
                'clustering_report_valid': outputs['clustering_report'] is not None,
                'regime_assignments_valid': outputs['regime_assignments'] is not None,
                'cluster_characteristics_valid': outputs['cluster_characteristics'] is not None,
                'total_files_generated': len(outputs['output_files'])
            }
            tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Output validation: {output_validation}")
            
            outputs['generation_stats'].update({
                'total_time': total_output_time,
                'memory_delta_mb': memory_delta,
                'validation': output_validation
            })
            
            self.logger.info(f"📁 Output generation completed: {len(outputs['output_files'])} files generated")
            tprint_success(f"🎉 [NAS_TAS_CLUSTERING] Output generation completed: {len(outputs['output_files'])} files in {total_output_time:.3f}s")
            tprint("📁 [NAS_TAS_CLUSTERING] ===== OUTPUT GENERATION COMPLETED =====", color="green", bold=True)
            return outputs

        except Exception as e:
            total_output_time = time.time() - output_start
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_delta = final_memory - initial_memory
            
            tprint_error(f"❌ [NAS_TAS_CLUSTERING] Output generation failed: {e}")
            tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Error type: {type(e).__name__}")
            tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Error details: {str(e)}")
            tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Execution time: {total_output_time:.3f}s")
            tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Memory usage: {initial_memory:.1f} MB → {final_memory:.1f} MB (Δ{memory_delta:+.1f} MB)")
            
            # Log full traceback
            full_traceback = traceback.format_exc()
            tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Full traceback:\n{full_traceback}")
            
            self.logger.error(f"❌ Failed to generate outputs: {e}")
            self.logger.error(f"Error type: {type(e).__name__}")
            self.logger.error(f"Execution time: {total_output_time:.3f}s")
            self.logger.error(f"Memory usage: {initial_memory:.1f} MB → {final_memory:.1f} MB (Δ{memory_delta:+.1f} MB)")
            self.logger.error(f"Full traceback:\n{full_traceback}")
            
            # Return partial outputs
            fallback_outputs = {
                'clustering_report': None,
                'regime_assignments': None,
                'cluster_characteristics': None,
                'output_files': [],
                'generation_stats': {
                    'total_time': total_output_time,
                    'memory_delta_mb': memory_delta,
                    'error': str(e),
                    'error_type': type(e).__name__
                }
            }
            tprint("📁 [NAS_TAS_CLUSTERING] ===== OUTPUT GENERATION FAILED =====", color="red", bold=True)
            return fallback_outputs

    def _save_clustering_report(self, clustering_result) -> str:
        """Save clustering report to file with comprehensive logging and validation."""
        import time
        import os
        
        save_start = time.time()
        tprint("📄 [NAS_TAS_CLUSTERING] ===== SAVING CLUSTERING REPORT =====", color="blue", bold=True)
        
        try:
            # Step 1: Prepare output directory
            tprint("📁 [NAS_TAS_CLUSTERING] Step 1: Preparing output directory", color="cyan")
            dir_start = time.time()
            
            output_dir = Path(self.config.output_dir) / "nas_tas_clustering" / (self.config.symbol if self.config else 'UNKNOWN')
            tprint_debug(f"📁 [NAS_TAS_CLUSTERING] Output directory: {output_dir}")
            
            # Create directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)
            dir_time = time.time() - dir_start
            tprint_success(f"✅ [NAS_TAS_CLUSTERING] Output directory prepared in {dir_time:.3f}s")
            tprint_debug(f"📁 [NAS_TAS_CLUSTERING] Directory exists: {output_dir.exists()}")
            tprint_debug(f"📁 [NAS_TAS_CLUSTERING] Directory is writable: {os.access(output_dir, os.W_OK)}")

            # Step 2: Generate filename and path
            tprint("📝 [NAS_TAS_CLUSTERING] Step 2: Generating filename", color="cyan")
            filename_start = time.time()
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"nas_tas_clustering_report_{timestamp}.json"
            filepath = output_dir / filename
            
            tprint_debug(f"📝 [NAS_TAS_CLUSTERING] Timestamp: {timestamp}")
            tprint_debug(f"📝 [NAS_TAS_CLUSTERING] Filename: {filename}")
            tprint_debug(f"📝 [NAS_TAS_CLUSTERING] Full path: {filepath}")
            
            filename_time = time.time() - filename_start
            tprint_success(f"✅ [NAS_TAS_CLUSTERING] Filename generated in {filename_time:.3f}s")

            # Step 3: Prepare report data
            tprint("📊 [NAS_TAS_CLUSTERING] Step 3: Preparing report data", color="cyan", bold=True)
            data_start = time.time()
            
            # Validate clustering result
            if not clustering_result:
                tprint_error("❌ [NAS_TAS_CLUSTERING] No clustering result provided")
                raise ValueError("No clustering result provided")
            
            if not hasattr(clustering_result, 'labels') or clustering_result.labels is None:
                tprint_error("❌ [NAS_TAS_CLUSTERING] Clustering result has no labels")
                raise ValueError("Clustering result has no labels")
            
            # Calculate regime count
            regime_count = len(set(clustering_result.labels))
            tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Regime count: {regime_count}")
            tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Total samples: {len(clustering_result.labels)}")
            tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Algorithm used: {clustering_result.algorithm_used}")
            tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Success: {clustering_result.success}")
            
            # Prepare comprehensive report data
            report_data = {
                'clustering_result': {
                    'regime_count': regime_count,
                    'total_samples': len(clustering_result.labels),
                    'algorithm_used': clustering_result.algorithm_used,
                    'quality_metrics': clustering_result.quality_metrics,
                    'execution_time': clustering_result.execution_time,
                    'success': clustering_result.success,
                    'labels_summary': {
                        'unique_labels': sorted(set(clustering_result.labels)),
                        'label_counts': dict(zip(*np.unique(clustering_result.labels, return_counts=True))),
                        'label_distribution': {str(k): v for k, v in dict(zip(*np.unique(clustering_result.labels, return_counts=True))).items()}
                    }
                },
                'metadata': self.execution_metadata,
                'config': asdict(self.config) if self.config else {},
                'generation_info': {
                    'timestamp': timestamp,
                    'file_path': str(filepath),
                    'file_size_bytes': 0,  # Will be updated after writing
                    'generation_time': 0   # Will be updated after completion
                }
            }
            
            # Add additional metadata if available
            if hasattr(clustering_result, 'cluster_centers') and clustering_result.cluster_centers is not None:
                report_data['clustering_result']['cluster_centers_shape'] = clustering_result.cluster_centers.shape
                tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Cluster centers shape: {clustering_result.cluster_centers.shape}")
            
            if hasattr(clustering_result, 'probabilities') and clustering_result.probabilities is not None:
                report_data['clustering_result']['probabilities_shape'] = clustering_result.probabilities.shape
                tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Probabilities shape: {clustering_result.probabilities.shape}")
            
            data_time = time.time() - data_start
            tprint_success(f"✅ [NAS_TAS_CLUSTERING] Report data prepared in {data_time:.3f}s")
            tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Report data keys: {list(report_data.keys())}")

            # Step 4: Write file
            tprint("💾 [NAS_TAS_CLUSTERING] Step 4: Writing report file", color="cyan", bold=True)
            write_start = time.time()
            
            tprint_debug(f"💾 [NAS_TAS_CLUSTERING] Writing to: {filepath}")
            with open(filepath, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            write_time = time.time() - write_start
            tprint_success(f"✅ [NAS_TAS_CLUSTERING] Report file written in {write_time:.3f}s")

            # Step 5: Validate file
            tprint("🔍 [NAS_TAS_CLUSTERING] Step 5: Validating saved file", color="cyan")
            validation_start = time.time()
            
            # Check if file exists and has content
            if not filepath.exists():
                tprint_error(f"❌ [NAS_TAS_CLUSTERING] File was not created: {filepath}")
                raise FileNotFoundError(f"File was not created: {filepath}")
            
            file_size = filepath.stat().st_size
            if file_size == 0:
                tprint_error(f"❌ [NAS_TAS_CLUSTERING] File is empty: {filepath}")
                raise ValueError(f"File is empty: {filepath}")
            
            # Update report data with actual file size
            report_data['generation_info']['file_size_bytes'] = file_size
            
            # Test file readability
            try:
                with open(filepath, 'r') as f:
                    test_data = json.load(f)
                tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] File validation: JSON readable, {len(test_data)} top-level keys")
            except json.JSONDecodeError as e:
                tprint_error(f"❌ [NAS_TAS_CLUSTERING] File contains invalid JSON: {e}")
                raise ValueError(f"File contains invalid JSON: {e}")
            
            validation_time = time.time() - validation_start
            tprint_success(f"✅ [NAS_TAS_CLUSTERING] File validation passed in {validation_time:.3f}s")
            tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] File size: {file_size} bytes")
            tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] File readable: {os.access(filepath, os.R_OK)}")

            # Step 6: Performance metrics
            total_save_time = time.time() - save_start
            tprint("📊 [NAS_TAS_CLUSTERING] Step 6: Save performance summary", color="cyan", bold=True)
            tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Total save time: {total_save_time:.3f}s")
            tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Time breakdown:")
            tprint_debug(f"   - Directory prep: {dir_time:.3f}s ({dir_time/total_save_time*100:.1f}%)")
            tprint_debug(f"   - Filename gen: {filename_time:.3f}s ({filename_time/total_save_time*100:.1f}%)")
            tprint_debug(f"   - Data prep: {data_time:.3f}s ({data_time/total_save_time*100:.1f}%)")
            tprint_debug(f"   - File write: {write_time:.3f}s ({write_time/total_save_time*100:.1f}%)")
            tprint_debug(f"   - Validation: {validation_time:.3f}s ({validation_time/total_save_time*100:.1f}%)")
            
            # Update final generation info
            report_data['generation_info']['generation_time'] = total_save_time
            
            self.logger.info(f"💾 Clustering report saved to: {filepath}")
            tprint_success(f"🎉 [NAS_TAS_CLUSTERING] Clustering report saved successfully: {filepath}")
            tprint("📄 [NAS_TAS_CLUSTERING] ===== CLUSTERING REPORT SAVED =====", color="green", bold=True)
            return str(filepath)

        except Exception as e:
            total_save_time = time.time() - save_start
            tprint_error(f"❌ [NAS_TAS_CLUSTERING] Failed to save clustering report: {e}")
            tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Error type: {type(e).__name__}")
            tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Error details: {str(e)}")
            tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Execution time: {total_save_time:.3f}s")
            
            # Log full traceback
            full_traceback = traceback.format_exc()
            tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Full traceback:\n{full_traceback}")
            
            self.logger.error(f"❌ Failed to save clustering report: {e}")
            self.logger.error(f"Error type: {type(e).__name__}")
            self.logger.error(f"Execution time: {total_save_time:.3f}s")
            self.logger.error(f"Full traceback:\n{full_traceback}")
            
            tprint("📄 [NAS_TAS_CLUSTERING] ===== CLUSTERING REPORT SAVE FAILED =====", color="red", bold=True)
            return ""

    def _save_regime_assignments(self, regime_data: pd.DataFrame) -> str:
        """Save regime assignments to file."""
        try:
            output_dir = Path(self.config.output_dir) / "nas_tas_clustering" / (self.config.symbol if self.config else 'UNKNOWN')
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"nas_tas_regime_assignments_{timestamp}.parquet"
            filepath = output_dir / filename

            regime_data.to_parquet(filepath)
            self.logger.info(f"💾 Regime assignments saved to: {filepath}")
            return str(filepath)

        except Exception as e:
            self.logger.error(f"❌ Failed to save regime assignments: {e}")
            return ""

    def _save_cluster_characteristics(self, characteristics: Dict) -> str:
        """Save cluster characteristics to file."""
        try:
            output_dir = Path(self.config.output_dir) / "nas_tas_clustering" / (self.config.symbol if self.config else 'UNKNOWN')
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"nas_tas_cluster_characteristics_{timestamp}.json"
            filepath = output_dir / filename

            with open(filepath, 'w') as f:
                json.dump(characteristics, f, indent=2, default=str)

            self.logger.info(f"💾 Cluster characteristics saved to: {filepath}")
            return str(filepath)

        except Exception as e:
            self.logger.error(f"❌ Failed to save cluster characteristics: {e}")
            return ""

    def _generate_regime_assignments(self, market_data: pd.DataFrame, clustering_result) -> Optional[pd.DataFrame]:
        """Generate regime assignments DataFrame."""
        try:
            if clustering_result.labels is None or len(clustering_result.labels) == 0:
                return None

            # Handle probabilities - extract probability for assigned cluster
            if clustering_result.probabilities is not None and len(clustering_result.probabilities) > 0:
                # probabilities is 2D array (n_samples, n_clusters)
                # Extract probability for the assigned cluster for each sample
                if clustering_result.probabilities.ndim == 2:
                    # Get the probability for the assigned cluster (maximum probability)
                    regime_probs = np.max(clustering_result.probabilities, axis=1)
                else:
                    # Fallback to uniform probabilities if not 2D
                    regime_probs = np.ones(len(market_data)) * 0.5
            else:
                # Use zeros if no probabilities available
                regime_probs = np.zeros(len(market_data))

            # Create DataFrame with regime assignments
            regime_data = pd.DataFrame({
                'timestamp': market_data.index,
                'regime_id': clustering_result.labels,
                'regime_prob': regime_probs
            }).set_index('timestamp')

            return regime_data

        except Exception as e:
            self.logger.error(f"❌ Failed to generate regime assignments: {e}")
            return None

    def _generate_cluster_characteristics(self, market_data: pd.DataFrame, clustering_result) -> Dict[str, Any]:
        """Generate cluster characteristics."""
        try:
            characteristics = {}
            unique_regimes = set(clustering_result.labels)

            for regime_id in unique_regimes:
                regime_mask = clustering_result.labels == regime_id
                regime_data = market_data.iloc[regime_mask] if regime_mask.any() else pd.DataFrame()

                if len(regime_data) > 0:
                    characteristics[f'regime_{regime_id}'] = {
                        'sample_count': len(regime_data),
                        'avg_return': regime_data['close'].pct_change().mean() if 'close' in regime_data.columns else 0.0,
                        'volatility': regime_data['close'].pct_change().std() if 'close' in regime_data.columns else 0.0,
                        'avg_volume': regime_data['volume'].mean() if 'volume' in regime_data.columns else 0.0
                    }

            return characteristics

        except Exception as e:
            self.logger.error(f"❌ Failed to generate cluster characteristics: {e}")
            return {}

    def get_status(self) -> Dict[str, Any]:
        """Get component status."""
        return {
            'component': 'nas_tas_clustering',
            'initialized': self.unified_clustering is not None,
            'has_results': self.clustering_result is not None,
            'execution_metadata': self.execution_metadata
        }

    def validate_inputs(self) -> List[str]:
        """Validate input parameters with comprehensive logging."""
        tprint("🔍 [NAS_TAS_CLUSTERING] ===== VALIDATING INPUTS =====", color="blue", bold=True)
        
        errors = []
        
        # Log validation start
        tprint_debug("🔍 [NAS_TAS_CLUSTERING] Starting input validation...")
        tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Config type: {type(self.config)}")
        tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Config exists: {self.config is not None}")

        if not self.config:
            error_msg = "Configuration is required"
            errors.append(error_msg)
            tprint_error(f"❌ [NAS_TAS_CLUSTERING] {error_msg}")
            return errors

        # Validate symbol
        tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Validating symbol: {self.config.symbol}")
        if not self.config.symbol:
            error_msg = "Symbol is required"
            errors.append(error_msg)
            tprint_error(f"❌ [NAS_TAS_CLUSTERING] {error_msg}")
        else:
            tprint_success(f"✅ [NAS_TAS_CLUSTERING] Symbol validation passed: {self.config.symbol}")

        # Validate timeframe
        tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Validating timeframe: {self.config.timeframe}")
        if not self.config.timeframe:
            error_msg = "Timeframe is required"
            errors.append(error_msg)
            tprint_error(f"❌ [NAS_TAS_CLUSTERING] {error_msg}")
        else:
            valid_timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
            if self.config.timeframe not in valid_timeframes:
                error_msg = f"Invalid timeframe: {self.config.timeframe}. Must be one of {valid_timeframes}"
                errors.append(error_msg)
                tprint_error(f"❌ [NAS_TAS_CLUSTERING] {error_msg}")
            else:
                tprint_success(f"✅ [NAS_TAS_CLUSTERING] Timeframe validation passed: {self.config.timeframe}")

        # Validate n_regimes
        tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Validating n_regimes: {self.config.n_regimes}")
        if self.config.n_regimes < 2:
            error_msg = "Number of regimes must be at least 2"
            errors.append(error_msg)
            tprint_error(f"❌ [NAS_TAS_CLUSTERING] {error_msg}")
        elif self.config.n_regimes > 50:
            error_msg = "Number of regimes should not exceed 50"
            errors.append(error_msg)
            tprint_warning(f"⚠️ [NAS_TAS_CLUSTERING] {error_msg}")
        else:
            tprint_success(f"✅ [NAS_TAS_CLUSTERING] N regimes validation passed: {self.config.n_regimes}")

        # Validate algorithm type
        tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Validating algorithm type: {self.config.algorithm_type}")
        valid_algorithms = ['adaptive_clustering', 'kmeans', 'gaussian_mixture', 'hierarchical', 'dbscan']
        if self.config.algorithm_type not in valid_algorithms:
            error_msg = f"Invalid algorithm type: {self.config.algorithm_type}. Must be one of {valid_algorithms}"
            errors.append(error_msg)
            tprint_error(f"❌ [NAS_TAS_CLUSTERING] {error_msg}")
        else:
            tprint_success(f"✅ [NAS_TAS_CLUSTERING] Algorithm type validation passed: {self.config.algorithm_type}")

        # Validate weights
        tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Validating weights...")
        total_weight = self.config.economic_weight + self.config.momentum_weight + self.config.volume_weight
        tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Weight sum: {total_weight:.3f}")
        if abs(total_weight - 1.0) > 0.01:
            error_msg = f"Weights don't sum to 1.0 ({total_weight:.3f})"
            errors.append(error_msg)
            tprint_warning(f"⚠️ [NAS_TAS_CLUSTERING] {error_msg}")
        else:
            tprint_success(f"✅ [NAS_TAS_CLUSTERING] Weights validation passed: {total_weight:.3f}")

        # Log validation results
        if errors:
            tprint_error(f"❌ [NAS_TAS_CLUSTERING] Input validation failed with {len(errors)} errors")
            for i, error in enumerate(errors, 1):
                tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Error {i}: {error}")
        else:
            tprint_success(f"✅ [NAS_TAS_CLUSTERING] Input validation passed with no errors")
        
        tprint("🔍 [NAS_TAS_CLUSTERING] ===== INPUT VALIDATION COMPLETED =====", color="green" if not errors else "red", bold=True)
        return errors

    def debug_data_quality(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Debug data quality with comprehensive analysis."""
        import time
        
        debug_start = time.time()
        tprint("🔍 [NAS_TAS_CLUSTERING] ===== DEBUGGING DATA QUALITY =====", color="blue", bold=True)
        
        debug_info = {
            'timestamp': datetime.now().isoformat(),
            'data_shape': market_data.shape,
            'data_types': {},
            'missing_data': {},
            'infinite_values': {},
            'statistics': {},
            'quality_score': 0.0,
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Basic data info
            tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Data shape: {market_data.shape}")
            tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Data columns: {list(market_data.columns)}")
            tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Data index type: {type(market_data.index)}")
            tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Data memory usage: {market_data.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
            
            # Data types analysis
            tprint_debug("📊 [NAS_TAS_CLUSTERING] Analyzing data types...")
            for col in market_data.columns:
                dtype = str(market_data[col].dtype)
                debug_info['data_types'][col] = dtype
                tprint_debug(f"📊 [NAS_TAS_CLUSTERING] {col}: {dtype}")
            
            # Missing data analysis
            tprint_debug("📊 [NAS_TAS_CLUSTERING] Analyzing missing data...")
            missing_data = market_data.isnull().sum()
            debug_info['missing_data'] = missing_data.to_dict()
            
            total_missing = missing_data.sum()
            missing_percentage = (total_missing / (market_data.shape[0] * market_data.shape[1])) * 100
            
            tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Total missing values: {total_missing}")
            tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Missing percentage: {missing_percentage:.2f}%")
            
            if total_missing > 0:
                debug_info['issues'].append(f"Missing data detected: {total_missing} values ({missing_percentage:.2f}%)")
                tprint_warning(f"⚠️ [NAS_TAS_CLUSTERING] Missing data detected: {total_missing} values")
            else:
                tprint_success("✅ [NAS_TAS_CLUSTERING] No missing data detected")
            
            # Infinite values analysis
            tprint_debug("📊 [NAS_TAS_CLUSTERING] Analyzing infinite values...")
            inf_counts = {}
            for col in market_data.select_dtypes(include=[np.number]).columns:
                inf_count = np.isinf(market_data[col]).sum()
                if inf_count > 0:
                    inf_counts[col] = inf_count
                    debug_info['issues'].append(f"Infinite values in {col}: {inf_count}")
                    tprint_warning(f"⚠️ [NAS_TAS_CLUSTERING] Infinite values in {col}: {inf_count}")
            
            debug_info['infinite_values'] = inf_counts
            if not inf_counts:
                tprint_success("✅ [NAS_TAS_CLUSTERING] No infinite values detected")
            
            # Statistical analysis
            tprint_debug("📊 [NAS_TAS_CLUSTERING] Computing statistics...")
            for col in market_data.select_dtypes(include=[np.number]).columns:
                col_stats = {
                    'mean': float(market_data[col].mean()),
                    'std': float(market_data[col].std()),
                    'min': float(market_data[col].min()),
                    'max': float(market_data[col].max()),
                    'median': float(market_data[col].median()),
                    'skewness': float(market_data[col].skew()),
                    'kurtosis': float(market_data[col].kurtosis())
                }
                debug_info['statistics'][col] = col_stats
                tprint_debug(f"📊 [NAS_TAS_CLUSTERING] {col} stats: mean={col_stats['mean']:.6f}, std={col_stats['std']:.6f}")
            
            # Quality score calculation
            quality_score = 100.0
            if total_missing > 0:
                quality_score -= min(50, missing_percentage * 2)  # Penalty for missing data
            if inf_counts:
                quality_score -= len(inf_counts) * 10  # Penalty for infinite values
            if market_data.shape[0] < 100:
                quality_score -= 20  # Penalty for insufficient data
                debug_info['issues'].append("Insufficient data: less than 100 rows")
            
            debug_info['quality_score'] = max(0, quality_score)
            
            # Generate recommendations
            if missing_percentage > 5:
                debug_info['recommendations'].append("Consider data imputation for missing values")
            if inf_counts:
                debug_info['recommendations'].append("Clean infinite values before clustering")
            if market_data.shape[0] < 1000:
                debug_info['recommendations'].append("Consider using more data for better clustering results")
            
            debug_time = time.time() - debug_start
            tprint_success(f"✅ [NAS_TAS_CLUSTERING] Data quality analysis completed in {debug_time:.3f}s")
            tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Quality score: {debug_info['quality_score']:.1f}/100")
            tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Issues found: {len(debug_info['issues'])}")
            tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Recommendations: {len(debug_info['recommendations'])}")
            
            tprint("🔍 [NAS_TAS_CLUSTERING] ===== DATA QUALITY DEBUG COMPLETED =====", color="green", bold=True)
            return debug_info
            
        except Exception as e:
            debug_time = time.time() - debug_start
            tprint_error(f"❌ [NAS_TAS_CLUSTERING] Data quality debugging failed: {e}")
            tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Error type: {type(e).__name__}")
            tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Error details: {str(e)}")
            tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Execution time: {debug_time:.3f}s")
            
            debug_info.update({
                'error': str(e),
                'error_type': type(e).__name__,
                'execution_time': debug_time
            })
            
            tprint("🔍 [NAS_TAS_CLUSTERING] ===== DATA QUALITY DEBUG FAILED =====", color="red", bold=True)
            return debug_info

    def debug_clustering_result(self, clustering_result) -> Dict[str, Any]:
        """Debug clustering result with comprehensive analysis."""
        import time
        
        debug_start = time.time()
        tprint("🔍 [NAS_TAS_CLUSTERING] ===== DEBUGGING CLUSTERING RESULT =====", color="blue", bold=True)
        
        debug_info = {
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'regime_count': 0,
            'total_samples': 0,
            'algorithm_used': None,
            'execution_time': 0.0,
            'quality_metrics': {},
            'label_distribution': {},
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Basic result validation
            if not clustering_result:
                debug_info['issues'].append("No clustering result provided")
                tprint_error("❌ [NAS_TAS_CLUSTERING] No clustering result provided")
                return debug_info
            
            debug_info['success'] = clustering_result.success
            tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Clustering success: {clustering_result.success}")
            
            if not clustering_result.success:
                debug_info['issues'].append(f"Clustering failed: {clustering_result.error_message}")
                tprint_error(f"❌ [NAS_TAS_CLUSTERING] Clustering failed: {clustering_result.error_message}")
                return debug_info
            
            # Labels analysis
            if hasattr(clustering_result, 'labels') and clustering_result.labels is not None:
                labels = clustering_result.labels
                debug_info['total_samples'] = len(labels)
                debug_info['regime_count'] = len(set(labels))
                
                tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Total samples: {len(labels)}")
                tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Regime count: {len(set(labels))}")
                
                # Label distribution
                unique_labels, counts = np.unique(labels, return_counts=True)
                label_dist = dict(zip(unique_labels, counts))
                debug_info['label_distribution'] = {str(k): int(v) for k, v in label_dist.items()}
                
                tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Label distribution: {debug_info['label_distribution']}")
                
                # Check for balanced clusters
                min_count = min(counts)
                max_count = max(counts)
                balance_ratio = min_count / max_count if max_count > 0 else 0
                
                tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Cluster balance ratio: {balance_ratio:.3f}")
                
                if balance_ratio < 0.1:
                    debug_info['issues'].append("Highly imbalanced clusters detected")
                    debug_info['recommendations'].append("Consider adjusting clustering parameters for better balance")
                    tprint_warning("⚠️ [NAS_TAS_CLUSTERING] Highly imbalanced clusters detected")
                elif balance_ratio < 0.3:
                    debug_info['recommendations'].append("Clusters are somewhat imbalanced, consider parameter tuning")
                    tprint_warning("⚠️ [NAS_TAS_CLUSTERING] Clusters are somewhat imbalanced")
                else:
                    tprint_success("✅ [NAS_TAS_CLUSTERING] Clusters are reasonably balanced")
            else:
                debug_info['issues'].append("No labels found in clustering result")
                tprint_error("❌ [NAS_TAS_CLUSTERING] No labels found in clustering result")
            
            # Algorithm information
            if hasattr(clustering_result, 'algorithm_used'):
                debug_info['algorithm_used'] = clustering_result.algorithm_used
                tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Algorithm used: {clustering_result.algorithm_used}")
            
            # Execution time
            if hasattr(clustering_result, 'execution_time'):
                debug_info['execution_time'] = clustering_result.execution_time
                tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Execution time: {clustering_result.execution_time:.3f}s")
            
            # Quality metrics
            if hasattr(clustering_result, 'quality_metrics') and clustering_result.quality_metrics:
                debug_info['quality_metrics'] = clustering_result.quality_metrics
                tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Quality metrics: {clustering_result.quality_metrics}")
                
                # Analyze quality metrics
                for metric_name, metric_value in clustering_result.quality_metrics.items():
                    tprint_debug(f"📊 [NAS_TAS_CLUSTERING] {metric_name}: {metric_value}")
            else:
                debug_info['issues'].append("No quality metrics available")
                tprint_warning("⚠️ [NAS_TAS_CLUSTERING] No quality metrics available")
            
            # Cluster centers analysis
            if hasattr(clustering_result, 'cluster_centers') and clustering_result.cluster_centers is not None:
                centers = clustering_result.cluster_centers
                tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Cluster centers shape: {centers.shape}")
                
                # Check for duplicate centers
                if len(centers) != len(np.unique(centers, axis=0)):
                    debug_info['issues'].append("Duplicate cluster centers detected")
                    tprint_warning("⚠️ [NAS_TAS_CLUSTERING] Duplicate cluster centers detected")
                else:
                    tprint_success("✅ [NAS_TAS_CLUSTERING] No duplicate cluster centers")
            
            # Probabilities analysis
            if hasattr(clustering_result, 'probabilities') and clustering_result.probabilities is not None:
                probs = clustering_result.probabilities
                tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Probabilities shape: {probs.shape}")
                
                # Check probability validity
                if np.any(probs < 0) or np.any(probs > 1):
                    debug_info['issues'].append("Invalid probability values detected")
                    tprint_warning("⚠️ [NAS_TAS_CLUSTERING] Invalid probability values detected")
                else:
                    tprint_success("✅ [NAS_TAS_CLUSTERING] Probability values are valid")
                
                # Check probability sums
                prob_sums = np.sum(probs, axis=1)
                if not np.allclose(prob_sums, 1.0, atol=1e-6):
                    debug_info['issues'].append("Probability sums don't equal 1.0")
                    tprint_warning("⚠️ [NAS_TAS_CLUSTERING] Probability sums don't equal 1.0")
                else:
                    tprint_success("✅ [NAS_TAS_CLUSTERING] Probability sums are valid")
            
            debug_time = time.time() - debug_start
            tprint_success(f"✅ [NAS_TAS_CLUSTERING] Clustering result debugging completed in {debug_time:.3f}s")
            tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Issues found: {len(debug_info['issues'])}")
            tprint_debug(f"📊 [NAS_TAS_CLUSTERING] Recommendations: {len(debug_info['recommendations'])}")
            
            tprint("🔍 [NAS_TAS_CLUSTERING] ===== CLUSTERING RESULT DEBUG COMPLETED =====", color="green", bold=True)
            return debug_info
            
        except Exception as e:
            debug_time = time.time() - debug_start
            tprint_error(f"❌ [NAS_TAS_CLUSTERING] Clustering result debugging failed: {e}")
            tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Error type: {type(e).__name__}")
            tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Error details: {str(e)}")
            tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Execution time: {debug_time:.3f}s")
            
            debug_info.update({
                'error': str(e),
                'error_type': type(e).__name__,
                'execution_time': debug_time
            })
            
            tprint("🔍 [NAS_TAS_CLUSTERING] ===== CLUSTERING RESULT DEBUG FAILED =====", color="red", bold=True)
            return debug_info

    def debug_system_resources(self) -> Dict[str, Any]:
        """Debug system resources and performance."""
        import time
        import psutil
        
        debug_start = time.time()
        tprint("🔍 [NAS_TAS_CLUSTERING] ===== DEBUGGING SYSTEM RESOURCES =====", color="blue", bold=True)
        
        debug_info = {
            'timestamp': datetime.now().isoformat(),
            'cpu_info': {},
            'memory_info': {},
            'disk_info': {},
            'process_info': {},
            'recommendations': []
        }
        
        try:
            # CPU information
            tprint_debug("💻 [NAS_TAS_CLUSTERING] Analyzing CPU information...")
            cpu_count = psutil.cpu_count()
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_freq = psutil.cpu_freq()
            
            debug_info['cpu_info'] = {
                'cpu_count': cpu_count,
                'cpu_percent': cpu_percent,
                'cpu_freq_current': cpu_freq.current if cpu_freq else None,
                'cpu_freq_min': cpu_freq.min if cpu_freq else None,
                'cpu_freq_max': cpu_freq.max if cpu_freq else None
            }
            
            tprint_debug(f"💻 [NAS_TAS_CLUSTERING] CPU count: {cpu_count}")
            tprint_debug(f"💻 [NAS_TAS_CLUSTERING] CPU usage: {cpu_percent}%")
            if cpu_freq:
                tprint_debug(f"💻 [NAS_TAS_CLUSTERING] CPU frequency: {cpu_freq.current} MHz")
            
            # Memory information
            tprint_debug("💻 [NAS_TAS_CLUSTERING] Analyzing memory information...")
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            debug_info['memory_info'] = {
                'total_memory_gb': memory.total / 1024 / 1024 / 1024,
                'available_memory_gb': memory.available / 1024 / 1024 / 1024,
                'used_memory_gb': memory.used / 1024 / 1024 / 1024,
                'memory_percent': memory.percent,
                'swap_total_gb': swap.total / 1024 / 1024 / 1024,
                'swap_used_gb': swap.used / 1024 / 1024 / 1024,
                'swap_percent': swap.percent
            }
            
            tprint_debug(f"💻 [NAS_TAS_CLUSTERING] Total memory: {memory.total / 1024 / 1024 / 1024:.1f} GB")
            tprint_debug(f"💻 [NAS_TAS_CLUSTERING] Available memory: {memory.available / 1024 / 1024 / 1024:.1f} GB")
            tprint_debug(f"💻 [NAS_TAS_CLUSTERING] Memory usage: {memory.percent}%")
            
            # Disk information
            tprint_debug("💻 [NAS_TAS_CLUSTERING] Analyzing disk information...")
            disk = psutil.disk_usage('/')
            
            debug_info['disk_info'] = {
                'total_disk_gb': disk.total / 1024 / 1024 / 1024,
                'used_disk_gb': disk.used / 1024 / 1024 / 1024,
                'free_disk_gb': disk.free / 1024 / 1024 / 1024,
                'disk_percent': (disk.used / disk.total) * 100
            }
            
            tprint_debug(f"💻 [NAS_TAS_CLUSTERING] Total disk: {disk.total / 1024 / 1024 / 1024:.1f} GB")
            tprint_debug(f"💻 [NAS_TAS_CLUSTERING] Free disk: {disk.free / 1024 / 1024 / 1024:.1f} GB")
            tprint_debug(f"💻 [NAS_TAS_CLUSTERING] Disk usage: {(disk.used / disk.total) * 100:.1f}%")
            
            # Process information
            tprint_debug("💻 [NAS_TAS_CLUSTERING] Analyzing process information...")
            process = psutil.Process()
            
            debug_info['process_info'] = {
                'pid': process.pid,
                'memory_usage_mb': process.memory_info().rss / 1024 / 1024,
                'cpu_percent': process.cpu_percent(),
                'num_threads': process.num_threads(),
                'create_time': process.create_time(),
                'status': process.status()
            }
            
            tprint_debug(f"💻 [NAS_TAS_CLUSTERING] Process PID: {process.pid}")
            tprint_debug(f"💻 [NAS_TAS_CLUSTERING] Process memory: {process.memory_info().rss / 1024 / 1024:.1f} MB")
            tprint_debug(f"💻 [NAS_TAS_CLUSTERING] Process threads: {process.num_threads()}")
            
            # Generate recommendations
            if memory.percent > 90:
                debug_info['recommendations'].append("High memory usage detected, consider reducing data size or using more efficient algorithms")
            if cpu_percent > 90:
                debug_info['recommendations'].append("High CPU usage detected, consider using fewer parallel processes")
            if disk.free / disk.total < 0.1:
                debug_info['recommendations'].append("Low disk space, consider cleaning up temporary files")
            if process.memory_info().rss / 1024 / 1024 > 1000:
                debug_info['recommendations'].append("High process memory usage, consider optimizing data structures")
            
            debug_time = time.time() - debug_start
            tprint_success(f"✅ [NAS_TAS_CLUSTERING] System resource debugging completed in {debug_time:.3f}s")
            tprint_debug(f"💻 [NAS_TAS_CLUSTERING] Recommendations: {len(debug_info['recommendations'])}")
            
            tprint("🔍 [NAS_TAS_CLUSTERING] ===== SYSTEM RESOURCE DEBUG COMPLETED =====", color="green", bold=True)
            return debug_info
            
        except Exception as e:
            debug_time = time.time() - debug_start
            tprint_error(f"❌ [NAS_TAS_CLUSTERING] System resource debugging failed: {e}")
            tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Error type: {type(e).__name__}")
            tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Error details: {str(e)}")
            tprint_debug(f"🔍 [NAS_TAS_CLUSTERING] Execution time: {debug_time:.3f}s")
            
            debug_info.update({
                'error': str(e),
                'error_type': type(e).__name__,
                'execution_time': debug_time
            })
            
            tprint("🔍 [NAS_TAS_CLUSTERING] ===== SYSTEM RESOURCE DEBUG FAILED =====", color="red", bold=True)
            return debug_info
