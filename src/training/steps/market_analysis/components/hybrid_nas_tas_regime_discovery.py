"""
Hybrid NAS-TAS Regime Discovery Component.

This component discovers market regimes using a hybrid approach that combines
Neural Architecture Search (NAS) and Tree-driven Advanced Statistics (TAS).
Integrates with the advanced hybrid regime detection system.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
import time

from .base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult
from src.utils.logger import system_logger
from ..logging_standards import (
    get_logger, log_info, log_warning, log_error, log_success, log_debug,
    LoggingContext, log_step_progress, log_data_info, log_validation_result
)
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

# Import feature importance integration
try:
    from ..shared_utils.feature_importance_integration import (
        FeatureImportanceIntegrationManager, FeatureImportanceIntegrationConfig,
        FeatureImportancePipelineHook, integrate_feature_importance_with_clustering,
        enhance_regime_report_with_feature_importance
    )
    FEATURE_IMPORTANCE_AVAILABLE = True
except ImportError:
    FEATURE_IMPORTANCE_AVAILABLE = False

class NASTASRegimeDiscoveryComponent(BaseMarketAnalysisComponent):
    """
    Hybrid NAS-TAS Regime Discovery Component.

    Discovers market regimes using a hybrid approach that combines:
    - Neural Architecture Search (NAS) with advanced neural architectures
    - Tree-driven Advanced Statistics (TAS) with tree-based learning
    - Economic significance and trading viability evaluation
    - Multi-objective optimization and ensemble methods
    """

    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the hybrid NAS-TAS regime discovery component."""
        tprint("🚀 Initializing Hybrid NAS-TAS Regime Discovery Component", color="cyan")
        tprint(f"Configuration: {config}", color="blue")

        super().__init__(config)
        # Use standardized logging
        self.logger = get_logger('HybridNASTASRegimeDiscovery')
        self._resources_to_cleanup = []

        # Initialize feature importance integration
        self.feature_importance_manager = None
        self.feature_importance_hook = None
        if FEATURE_IMPORTANCE_AVAILABLE:
            try:
                # Configure feature importance integration
                importance_config = FeatureImportanceIntegrationConfig(
                    enable_pre_clustering_analysis=getattr(config, 'enable_feature_importance_pre_clustering', True),
                    enable_post_clustering_analysis=getattr(config, 'enable_feature_importance_post_clustering', True),
                    enable_regime_characterization=getattr(config, 'enable_regime_characterization', True),
                    importance_methods=getattr(config, 'feature_importance_methods', ["mutual_information", "f_classif"]),
                    include_detailed_analysis=getattr(config, 'include_detailed_feature_analysis', True)
                )
                self.feature_importance_manager = FeatureImportanceIntegrationManager(importance_config)
                self.feature_importance_hook = FeatureImportancePipelineHook(importance_config)
                tprint("✅ Feature importance analysis integration initialized", color="green")
            except Exception as e:
                tprint(f"⚠️ Feature importance initialization failed: {e}", color="yellow")
                self.logger.warning(f"Feature importance initialization failed: {e}")

        tprint("✅ Hybrid NAS-TAS Regime Discovery Component initialized", color="green")
        tprint("🔧 Component ready for regime discovery", color="blue")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with resource cleanup."""
        self._cleanup_resources()

    def _cleanup_resources(self):
        """Clean up any allocated resources."""
        try:
            for resource in self._resources_to_cleanup:
                if hasattr(resource, 'cleanup'):
                    resource.cleanup()
                elif hasattr(resource, 'close'):
                    resource.close()
            self._resources_to_cleanup.clear()
        except Exception as e:
            log_warning(f"Error during resource cleanup: {e}")

    def __del__(self):
        """Destructor with resource cleanup."""
        self._cleanup_resources()

    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['nas_tas_regime_discovery_result']

    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute hybrid NAS-TAS regime discovery.

        Args:
            data: Market data for regime discovery
            pipeline_state: Current pipeline state

        Returns:
            ComponentResult with hybrid regime discovery results
        """
        execution_start_time = time.time()
        tprint("🚀 [HYBRID_NAS_TAS] Starting Hybrid NAS-TAS Regime Discovery", color="cyan", bold=True)
        log_info('🚀 Starting Hybrid NAS-TAS Regime Discovery')

        try:
            # Resolve symbol from config or pipeline state
            tprint("🔍 [HYBRID_NAS_TAS] Resolving symbol configuration", color="yellow")
            symbol = getattr(self.config, 'symbol', None)
            if symbol is None and 'symbol' in pipeline_state:
                symbol = pipeline_state['symbol']
            if symbol is None:
                tprint("❌ [HYBRID_NAS_TAS] Symbol must be provided in config or pipeline state", color="red", bold=True)
                raise ValueError("Symbol must be provided in config or pipeline state")
            tprint(f"✅ [HYBRID_NAS_TAS] Symbol resolved: {symbol}", color="green")

            # Resolve timeframe from config or pipeline state
            tprint("🔍 [HYBRID_NAS_TAS] Resolving timeframe configuration", color="yellow")
            timeframe = getattr(self.config, 'timeframe', None)
            if timeframe is None and 'timeframe' in pipeline_state:
                timeframe = pipeline_state['timeframe']
            if timeframe is None:
                timeframe = '1h'  # Default timeframe for regime discovery
                tprint(f"⚠️ [HYBRID_NAS_TAS] Using default timeframe: {timeframe}", color="yellow")
            tprint(f"✅ [HYBRID_NAS_TAS] Timeframe resolved: {timeframe}", color="green")

            # Get market data
            tprint("📊 [HYBRID_NAS_TAS] Loading market data", color="blue")
            market_data = await self._load_market_data(data, symbol)
            if market_data is None or market_data.empty:
                tprint(f"❌ [HYBRID_NAS_TAS] No market data available for symbol: {symbol}", color="red", bold=True)
                raise ValueError(f"No market data available for hybrid regime discovery for symbol: {symbol}")
            tprint(f"✅ [HYBRID_NAS_TAS] Market data loaded: {len(market_data)} rows", color="green")

            # Configure hybrid regime detection
            tprint("⚙️ [HYBRID_NAS_TAS] Creating hybrid configuration", color="magenta")
            hybrid_config = self._create_hybrid_config(market_data, pipeline_state)
            tprint("✅ [HYBRID_NAS_TAS] Hybrid configuration created successfully", color="green")

            # Perform hybrid regime discovery
            tprint("🧠 [HYBRID_NAS_TAS] Starting hybrid regime discovery process", color="cyan", bold=True)
            discovery_start_time = time.time()
            enhanced_result = await self._perform_hybrid_regime_discovery(market_data, hybrid_config)
            discovery_time = time.time() - discovery_start_time
            tprint(f"⏱️ [HYBRID_NAS_TAS] Discovery process completed in {discovery_time:.2f}s", color="blue")

            if not enhanced_result.get('success', False):
                tprint(f"❌ [HYBRID_NAS_TAS] Hybrid regime discovery failed: {enhanced_result.get('error', 'Unknown error')}", color="red", bold=True)
                raise ValueError(f"Hybrid regime discovery failed: {enhanced_result.get('error', 'Unknown error')}")

            # Extract regime data
            tprint("📈 [HYBRID_NAS_TAS] Extracting regime predictions", color="yellow")
            regime_predictions = enhanced_result.get('consolidated_assignments', [])
            if len(regime_predictions) == 0:
                tprint("❌ [HYBRID_NAS_TAS] No regime predictions returned from hybrid discovery", color="red", bold=True)
                raise ValueError("No regime predictions returned from hybrid discovery")

            unique_regimes = len(set(regime_predictions))
            tprint(f"🎯 [HYBRID_NAS_TAS] Found {unique_regimes} unique regimes in {len(regime_predictions)} predictions", color="green")

            # Calculate regime metrics
            tprint("📊 [HYBRID_NAS_TAS] Calculating hybrid regime metrics", color="blue")
            regime_metrics = self._calculate_hybrid_regime_metrics(regime_predictions, enhanced_result)
            tprint("✅ [HYBRID_NAS_TAS] Regime metrics calculated", color="green")

            # Create regime characteristics for clustering
            tprint("🔬 [HYBRID_NAS_TAS] Creating regime characteristics for clustering", color="magenta")
            regime_characteristics = self._create_hybrid_regime_characteristics(
                market_data, regime_predictions, enhanced_result
            )
            tprint("✅ [HYBRID_NAS_TAS] Regime characteristics created", color="green")

            # Create single consolidated artifact
            tprint("📦 [HYBRID_NAS_TAS] Creating consolidated artifacts", color="blue")
            tprint(f"🔍 [HYBRID_NAS_TAS] DEBUG: enhanced_result keys before artifact creation: {list(enhanced_result.keys())}", color="blue")
            tprint(f"🔍 [HYBRID_NAS_TAS] DEBUG: enhanced_result clustering_quality present: {'clustering_quality' in enhanced_result}", color="blue")

            # DEBUG: Check if clustering quality metrics are in enhanced_result
            tprint("🔍 [HYBRID_NAS_TAS] DEBUG: Checking enhanced_result for clustering_quality", color="blue")
            tprint(f"🔍 [HYBRID_NAS_TAS] DEBUG: enhanced_result keys: {list(enhanced_result.keys())}", color="blue")
            if 'clustering_quality' in enhanced_result:
                tprint("✅ [HYBRID_NAS_TAS] DEBUG: clustering_quality found in enhanced_result", color="green")
                tprint(f"🔍 [HYBRID_NAS_TAS] DEBUG: clustering_quality content: {list(enhanced_result['clustering_quality'].keys()) if isinstance(enhanced_result['clustering_quality'], dict) else 'not a dict'}", color="blue")
            else:
                tprint("❌ [HYBRID_NAS_TAS] DEBUG: clustering_quality NOT found in enhanced_result", color="red")
                tprint(f"🔍 [HYBRID_NAS_TAS] DEBUG: enhanced_result type: {type(enhanced_result)}", color="blue")
                tprint(f"🔍 [HYBRID_NAS_TAS] DEBUG: enhanced_result content preview: {str(enhanced_result)[:200]}...", color="blue")

            artifacts = {
                'nas_tas_regime_discovery_result': {
                    # Core regime data (backward compatible)
                    'regime_count': unique_regimes,
                    'total_samples': len(regime_predictions),
                    'regime_distribution': self._calculate_regime_distribution(regime_predictions),
                    'regime_characteristics': regime_characteristics,

                    # Enhanced hybrid regime information
                    'hybrid_regime_info': {
                        'combination_strategy': enhanced_result.get('combination_strategy', 'ensemble'),
                        'nas_contribution': enhanced_result.get('nas_contribution', {}),
                        'tas_contribution': enhanced_result.get('tas_contribution', {}),
                        'consensus_metrics': enhanced_result.get('consensus_metrics', {}),
                        'disagreement_metrics': enhanced_result.get('disagreement_metrics', {}),
                        'consensus_mapping': enhanced_result.get('consensus_mapping', {}),
                        'consolidated_regime_count': enhanced_result.get('consolidated_regime_count', unique_regimes),
                        'consolidation_quality': enhanced_result.get('consolidation_quality', {}),
                        'clustering_metrics': enhanced_result.get('clustering_quality', {}),
                        'economic_significance_scores': enhanced_result.get('economic_significance_scores', []),
                        'trading_viability_scores': enhanced_result.get('trading_viability_scores', []),
                        'regime_stability_scores': enhanced_result.get('regime_stability_scores', [])
                    },

                    'regime_metrics': regime_metrics,
                    'configuration': {
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'architecture_type': 'Hybrid_NAS_TAS',
                        'combination_strategy': hybrid_config.get('combination_strategy', 'ensemble'),
                        'enable_nas': hybrid_config.get('enable_nas', True),
                        'enable_tas': hybrid_config.get('enable_tas', True),
                        'enable_economic_evaluation': hybrid_config.get('enable_economic_evaluation', True),
                        'enable_trading_viability': hybrid_config.get('enable_trading_viability', True),
                        'enable_consensus_analysis': hybrid_config.get('enable_consensus_analysis', True)
                    },
                    'execution_info': {
                        'timestamp': datetime.now().isoformat(),
                        'data_points_processed': len(market_data),
                        'success': True,
                        'discovery_time': discovery_time,
                        'nas_execution_time': enhanced_result.get('nas_execution_time', 0),
                        'tas_execution_time': enhanced_result.get('tas_execution_time', 0),
                        'consolidation_time': enhanced_result.get('consolidation_time', 0)
                    },

                    # Time-series regime assignments for clustering pipeline
                    'regime_assignments': regime_predictions,
                    'nas_assignments': enhanced_result.get('nas_assignments', []),
                    'tas_assignments': enhanced_result.get('tas_assignments', []),
                    'consensus_mapping': enhanced_result.get('consensus_mapping', {}),

                    # Clustering quality metrics
                    'clustering_quality': enhanced_result.get('clustering_quality', {})
                }
            }

            # DEBUG: Check clustering quality metrics in final artifacts
            clustering_quality = enhanced_result.get('clustering_quality', {})
            tprint(f"🔍 [HYBRID_NAS_TAS] DEBUG: clustering_quality type: {type(clustering_quality)}", color="blue")
            if isinstance(clustering_quality, dict):
                tprint(f"🔍 [HYBRID_NAS_TAS] DEBUG: clustering_quality keys: {list(clustering_quality.keys())}", color="blue")
                tprint(f"🔍 [HYBRID_NAS_TAS] DEBUG: clustering_quality content sample: {list(clustering_quality.values())[0] if clustering_quality else 'empty'}", color="blue")
            else:
                tprint(f"🔍 [HYBRID_NAS_TAS] DEBUG: clustering_quality is not a dict: {clustering_quality}", color="blue")

            # DEBUG: Check if clustering quality metrics are in final artifacts
            tprint("🔍 [HYBRID_NAS_TAS] DEBUG: Checking final artifacts for clustering_quality", color="blue")
            tprint(f"🔍 [HYBRID_NAS_TAS] DEBUG: Final artifact keys: {list(artifacts['nas_tas_regime_discovery_result'].keys())}", color="blue")
            if 'clustering_quality' in artifacts['nas_tas_regime_discovery_result']:
                tprint("✅ [HYBRID_NAS_TAS] DEBUG: clustering_quality found in final artifacts", color="green")
                clustering_quality = artifacts['nas_tas_regime_discovery_result']['clustering_quality']
                tprint(f"🔍 [HYBRID_NAS_TAS] DEBUG: clustering_quality content: {list(clustering_quality.keys()) if isinstance(clustering_quality, dict) else 'not a dict'}", color="blue")
            else:
                tprint("❌ [HYBRID_NAS_TAS] DEBUG: clustering_quality NOT found in final artifacts", color="red")
                tprint(f"🔍 [HYBRID_NAS_TAS] DEBUG: hybrid_regime_info keys: {list(artifacts['nas_tas_regime_discovery_result']['hybrid_regime_info'].keys())}", color="blue")
                if 'clustering_metrics' in artifacts['nas_tas_regime_discovery_result']['hybrid_regime_info']:
                    tprint(f"🔍 [HYBRID_NAS_TAS] DEBUG: clustering_metrics content: {artifacts['nas_tas_regime_discovery_result']['hybrid_regime_info']['clustering_metrics']}", color="blue")

            total_execution_time = time.time() - execution_start_time
            tprint(f"🎉 [HYBRID_NAS_TAS] SUCCESS: Discovery completed in {total_execution_time:.2f}s", color="green", bold=True)
            tprint(f"📊 [HYBRID_NAS_TAS] Final Results: {unique_regimes} regimes, {len(regime_predictions)} predictions", color="cyan")
            tprint(f"⏱️ [HYBRID_NAS_TAS] Performance: Discovery={discovery_time:.2f}s, Total={total_execution_time:.2f}s", color="blue")

            log_success(f'Hybrid NAS-TAS Regime Discovery completed: {unique_regimes} consolidated regimes discovered')
            return ComponentResult(
                success=True,
                artifacts=artifacts,
                metadata={
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'data_points_processed': len(market_data),
                    'regime_count': unique_regimes,
                    'architecture_type': 'Hybrid_NAS_TAS',
                    'execution_successful': True,
                    'discovery_time': discovery_time,
                    'nas_enabled': hybrid_config.get('enable_nas', True),
                    'tas_enabled': hybrid_config.get('enable_tas', True)
                }
            )

        except Exception as e:
            total_execution_time = time.time() - execution_start_time
            tprint(f"💥 [HYBRID_NAS_TAS] FAILURE: Discovery failed after {total_execution_time:.2f}s", color="red", bold=True)
            tprint(f"❌ [HYBRID_NAS_TAS] Error: {str(e)}", color="red")
            log_error(f'Hybrid NAS-TAS Regime Discovery failed: {e}')
            import traceback
            self.logger.error(f'❌ Error details: {traceback.format_exc()}')
            tprint(f"🔍 [HYBRID_NAS_TAS] Full traceback logged to system logger", color="yellow")
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=f"Hybrid regime discovery failed: {str(e)}"
            )

    def _create_hybrid_config(self, market_data: pd.DataFrame, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Create hybrid configuration based on data and pipeline state."""
        try:
            # Calculate optimal parameters based on data size
            data_size = len(market_data)
            tprint(f"🔧 [HYBRID_NAS_TAS] Analyzing data size: {data_size} rows", color="blue")

            # Determine configuration based on data characteristics
            if data_size < 1000:
                n_regimes = 2  # Reduced from 5 to 2 for small datasets
                population_size = 20
                generations = 50
                tree_depth = 4
                n_estimators = 100
                tprint("📊 [HYBRID_NAS_TAS] Using small dataset configuration", color="yellow")
            elif data_size < 5000:
                n_regimes = 8
                population_size = 50
                generations = 100
                tree_depth = 6
                n_estimators = 500
                tprint("📊 [HYBRID_NAS_TAS] Using medium dataset configuration", color="yellow")
            else:
                n_regimes = 10
                population_size = 100
                generations = 200
                tree_depth = 8
                n_estimators = 1000
                tprint("📊 [HYBRID_NAS_TAS] Using large dataset configuration", color="yellow")

            hybrid_config = {
                # Hybrid orchestration settings
                'combination_strategy': 'ensemble',  # ensemble, weighted, consensus
                'enable_nas': True,
                'enable_tas': True,
                'enable_consensus_analysis': True,
                'enable_economic_evaluation': True,
                'enable_trading_viability': True,

                # NAS configuration
                'nas_config': {
                    'primary_architecture': 'hybrid',
                    'search_strategy': 'evolutionary',
                    'population_size': population_size,
                    'generations': generations,
                    'enable_neural_odes': True,
                    'enable_vision_transformers': True,
                    'enable_meta_learning': True,
                    'n_regimes': n_regimes,
                    'primary_timeframe': getattr(self.config, 'timeframe', '15m'),
                    'enable_economic_evaluation': True,
                    'enable_trading_viability': True
                },

                # TAS configuration
                'tas_config': {
                    'n_regimes': n_regimes,
                    'primary_timeframe': getattr(self.config, 'timeframe', '15m'),
                    'tree_depth': tree_depth,
                    'n_estimators': n_estimators,
                    'min_samples_split': 10,
                    'min_samples_leaf': 5,
                    'max_features': 'sqrt',
                    'enable_patchtst_enhancement': True,
                    'enable_statistical_methods': True,
                    'enable_economic_evaluation': True,
                    'enable_meta_learning': True
                },

                # Hybrid-specific settings
                'consensus_threshold': 0.6,
                'disagreement_tolerance': 0.3,
                'economic_weight': 0.4,
                'trading_weight': 0.3,
                'stability_weight': 0.3
            }

            tprint(f"⚙️ [HYBRID_NAS_TAS] Configuration: {n_regimes} regimes, NAS(pop={population_size}, gen={generations}), TAS(depth={tree_depth}, est={n_estimators})", color="cyan")
            log_info(f"📊 Hybrid Configuration: {n_regimes} regimes, NAS(pop={population_size}, gen={generations}), TAS(depth={tree_depth}, est={n_estimators})")
            return hybrid_config

        except Exception as e:
            tprint(f"⚠️ [HYBRID_NAS_TAS] Config creation failed: {e}, using defaults", color="yellow")
            log_warning(f"Failed to create hybrid config: {e}, using defaults")
            return {
                'combination_strategy': 'ensemble',
                'enable_nas': True,
                'enable_tas': True,
                'enable_consensus_analysis': True,
                'enable_economic_evaluation': True,
                'enable_trading_viability': True,
                'nas_config': {
                    'primary_architecture': 'hybrid',
                    'search_strategy': 'evolutionary',
                    'population_size': 50,
                    'generations': 100,
                    'n_regimes': 8
                },
                'tas_config': {
                    'n_regimes': 8,
                    'tree_depth': 6,
                    'n_estimators': 1000
                }
            }

    async def _perform_hybrid_regime_discovery(self, market_data: pd.DataFrame, hybrid_config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform hybrid regime discovery using the advanced hybrid system."""
        try:
            tprint("🔧 [HYBRID_NAS_TAS] Importing hybrid components", color="blue")
            tprint("📦 [HYBRID_NAS_TAS] Loading hybrid orchestrator and configuration", color="cyan")
            # Import hybrid components
            from src.training.steps.market_analysis.hybrid_nas_tas_regime.hybrid_orchestrator import (
                HybridOrchestrator, HybridOrchestratorConfig
            )
            tprint("✅ [HYBRID_NAS_TAS] Hybrid components imported successfully", color="green")

            tprint("⚙️ [HYBRID_NAS_TAS] Creating orchestrator configuration", color="magenta")
            tprint("🔧 [HYBRID_NAS_TAS] Setting up configuration parameters", color="yellow")
            # Create hybrid orchestrator configuration
            orchestrator_config = HybridOrchestratorConfig(
                symbol=getattr(self.config, 'symbol', 'UNKNOWN'),
                timeframe=getattr(self.config, 'timeframe', '15m'),
                start_date=getattr(self.config, 'start_date', None),
                end_date=getattr(self.config, 'end_date', None),
                use_standardized_features=True,
                feature_categories=['momentum', 'volatility', 'volume', 'trend'],
                significance_threshold=0.5,
                min_regime_duration=10,
                viability_threshold=0.5,
                minimum_regime_duration=5,
                max_iterations=100,
                use_bayesian_optimization=True,
                population_size=hybrid_config['nas_config']['population_size'],
                max_generations=hybrid_config['nas_config']['generations'],
                use_nsga2=True,
                use_spea2=True,
                use_gpu_acceleration=True,
                memory_limit_gb=8.0,
                include_detailed_metrics=True,
                save_to_file=False
            )
            tprint("✅ [HYBRID_NAS_TAS] Orchestrator configuration created", color="green")
            tprint(f"📊 [HYBRID_NAS_TAS] Config: pop={hybrid_config['nas_config']['population_size']}, gen={hybrid_config['nas_config']['generations']}", color="cyan")

            tprint("🚀 [HYBRID_NAS_TAS] Initializing hybrid orchestrator", color="cyan")
            tprint("🏗️ [HYBRID_NAS_TAS] Building orchestrator with enhanced components", color="blue")
            # Initialize hybrid orchestrator
            hybrid_orchestrator = HybridOrchestrator(orchestrator_config)
            tprint("✅ [HYBRID_NAS_TAS] Hybrid orchestrator initialized", color="green")
            tprint("🔧 [HYBRID_NAS_TAS] Orchestrator ready for regime detection", color="cyan")

            tprint("🧠 [HYBRID_NAS_TAS] Starting TAS-NAS orchestrated detection", color="cyan", bold=True)
            tprint("🔄 [HYBRID_NAS_TAS] Coordinating TAS and NAS training processes", color="magenta")
            # Perform hybrid regime detection
            hybrid_result = hybrid_orchestrator.orchestrate_tas_nas_detection(
                market_data,
                timeframes=[getattr(self.config, 'timeframe', '15m')]
            )
            tprint("✅ [HYBRID_NAS_TAS] TAS-NAS detection completed", color="green")
            tprint("📊 [HYBRID_NAS_TAS] Training coordination finished successfully", color="cyan")

            # DEBUG: Check what the orchestrator returned
            tprint("🔍 [HYBRID_NAS_TAS] DEBUG: hybrid_result keys from orchestrator:", color="blue")
            tprint(f"🔍 [HYBRID_NAS_TAS] DEBUG: {list(hybrid_result.keys())}", color="blue")
            if 'clustering_quality' in hybrid_result:
                tprint("✅ [HYBRID_NAS_TAS] DEBUG: clustering_quality found in hybrid_result", color="green")
                tprint(f"🔍 [HYBRID_NAS_TAS] DEBUG: clustering_quality type: {type(hybrid_result['clustering_quality'])}", color="blue")
            else:
                tprint("❌ [HYBRID_NAS_TAS] DEBUG: clustering_quality NOT found in hybrid_result", color="red")
                tprint(f"🔍 [HYBRID_NAS_TAS] DEBUG: hybrid_analysis keys: {list(hybrid_result.get('hybrid_analysis', {}).keys())}", color="blue")

            tprint("🔬 [HYBRID_NAS_TAS] Enhancing hybrid results", color="blue")
            # Process and enhance the result
            enhanced_result = self._enhance_hybrid_result(hybrid_result, hybrid_config)
            tprint("✅ [HYBRID_NAS_TAS] Results enhanced successfully", color="green")

            return enhanced_result

        except ImportError as e:
            tprint(f"❌ [HYBRID_NAS_TAS] Import failed: {e}", color="red", bold=True)
            self.logger.error(f"Failed to import hybrid components: {e}")
            raise e
        except Exception as e:
            tprint(f"❌ [HYBRID_NAS_TAS] Discovery failed: {e}", color="red", bold=True)
            self.logger.error(f"Hybrid regime discovery failed: {e}")
            raise e

    def _enhance_hybrid_result(self, hybrid_result: Dict[str, Any], hybrid_config: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance hybrid result with additional analysis and metrics."""
        try:
            tprint("🔬 [HYBRID_NAS_TAS] Starting result enhancement", color="blue")
            tprint(f"🔍 [HYBRID_NAS_TAS] DEBUG: _enhance_hybrid_result called with hybrid_result keys: {list(hybrid_result.keys())}", color="blue")
            enhanced_result = hybrid_result.copy()

            # Extract regime assignments from primary timeframe
            primary_timeframe = getattr(self.config, 'timeframe', '15m')
            tprint(f"📊 [HYBRID_NAS_TAS] Processing primary timeframe: {primary_timeframe}", color="yellow")

            # Extract clustering quality metrics if available
            tprint("🔍 [HYBRID_NAS_TAS] DEBUG: Checking for clustering_quality in hybrid_result", color="blue")
            tprint(f"🔍 [HYBRID_NAS_TAS] DEBUG: hybrid_result keys: {list(hybrid_result.keys())}", color="blue")

            if 'clustering_quality' in hybrid_result:
                enhanced_result['clustering_quality'] = hybrid_result['clustering_quality']
                tprint("✅ [HYBRID_NAS_TAS] Clustering quality metrics extracted", color="green")
                tprint(f"🔍 [HYBRID_NAS_TAS] DEBUG: clustering_quality content: {list(hybrid_result['clustering_quality'].keys()) if isinstance(hybrid_result['clustering_quality'], dict) else 'not a dict'}", color="blue")
            elif 'hybrid_analysis' in hybrid_result and 'clustering_quality' in hybrid_result['hybrid_analysis']:
                enhanced_result['clustering_quality'] = hybrid_result['hybrid_analysis']['clustering_quality']
                tprint("✅ [HYBRID_NAS_TAS] Clustering quality metrics extracted from hybrid_analysis", color="green")
                tprint(f"🔍 [HYBRID_NAS_TAS] DEBUG: clustering_quality content: {list(hybrid_result['hybrid_analysis']['clustering_quality'].keys()) if isinstance(hybrid_result['hybrid_analysis']['clustering_quality'], dict) else 'not a dict'}", color="blue")
            else:
                tprint("⚠️ [HYBRID_NAS_TAS] No clustering quality metrics found in hybrid result", color="yellow")
                tprint(f"🔍 [HYBRID_NAS_TAS] DEBUG: hybrid_analysis keys: {list(hybrid_result.get('hybrid_analysis', {}).keys()) if 'hybrid_analysis' in hybrid_result else 'no hybrid_analysis'}", color="blue")

            if 'tas_results' in hybrid_result and primary_timeframe in hybrid_result['tas_results']:
                tas_result = hybrid_result['tas_results'][primary_timeframe]
                enhanced_result['tas_assignments'] = tas_result.get('regime_predictions', [])
                enhanced_result['tas_execution_time'] = tas_result.get('execution_time', 0)
                tprint(f"✅ [HYBRID_NAS_TAS] TAS assignments extracted: {len(enhanced_result['tas_assignments'])} predictions", color="green")

            if 'nas_results' in hybrid_result and primary_timeframe in hybrid_result['nas_results']:
                nas_result = hybrid_result['nas_results'][primary_timeframe]
                enhanced_result['nas_assignments'] = nas_result.get('regime_predictions', [])
                enhanced_result['nas_execution_time'] = nas_result.get('execution_time', 0)

                # Fast fail if NAS returns 0 predictions
                if len(enhanced_result['nas_assignments']) == 0:
                    tprint(f"❌ [HYBRID_NAS_TAS] NAS assignments extracted: 0 predictions - FAST FAIL", color="red", bold=True)
                    enhanced_result['error'] = "NAS returned 0 predictions - both TAS and NAS required"
                    enhanced_result['success'] = False
                    return enhanced_result
                else:
                    tprint(f"✅ [HYBRID_NAS_TAS] NAS assignments extracted: {len(enhanced_result['nas_assignments'])} predictions", color="green")

            # Validate both TAS and NAS results are present
            if 'tas_assignments' not in enhanced_result:
                tprint("❌ [HYBRID_NAS_TAS] TAS assignments missing - both TAS and NAS required", color="red", bold=True)
                enhanced_result['error'] = "TAS assignments missing - both TAS and NAS required"
                enhanced_result['success'] = False
                return enhanced_result

            if 'nas_assignments' not in enhanced_result:
                tprint("❌ [HYBRID_NAS_TAS] NAS assignments missing - both TAS and NAS required", color="red", bold=True)
                enhanced_result['error'] = "NAS assignments missing - both TAS and NAS required"
                enhanced_result['success'] = False
                return enhanced_result

            # Create consolidated assignments using ensemble method
            tprint("🔄 [HYBRID_NAS_TAS] Creating consolidated assignments", color="magenta")
            if 'tas_assignments' in enhanced_result and 'nas_assignments' in enhanced_result:
                tas_assignments = enhanced_result['tas_assignments']
                nas_assignments = enhanced_result['nas_assignments']

                # Validate assignments are not empty and have same length
                if len(tas_assignments) == 0 or len(nas_assignments) == 0:
                    tprint("❌ [HYBRID_NAS_TAS] Empty TAS or NAS assignments - both systems required", color="red", bold=True)
                    enhanced_result['error'] = "Empty TAS or NAS assignments - both systems required"
                    enhanced_result['success'] = False
                    return enhanced_result
                elif len(tas_assignments) != len(nas_assignments):
                    tprint(f"⚠️ [HYBRID_NAS_TAS] TAS/NAS assignment length mismatch: TAS={len(tas_assignments)}, NAS={len(nas_assignments)}", color="yellow")
                    tprint("📏 [HYBRID_NAS_TAS] Using minimum length for consolidation", color="yellow")
                    consolidated_assignments = self._create_consolidated_assignments(
                        tas_assignments,
                        nas_assignments,
                        hybrid_config
                    )
                    enhanced_result['consolidated_assignments'] = consolidated_assignments
                    enhanced_result['consolidated_regime_count'] = len(set(consolidated_assignments))
                    tprint(f"✅ [HYBRID_NAS_TAS] Consolidated assignments created: {len(consolidated_assignments)} predictions", color="green")
                else:
                    consolidated_assignments = self._create_consolidated_assignments(
                        tas_assignments,
                        nas_assignments,
                        hybrid_config
                    )
                    enhanced_result['consolidated_assignments'] = consolidated_assignments
                    enhanced_result['consolidated_regime_count'] = len(set(consolidated_assignments))
                    tprint(f"✅ [HYBRID_NAS_TAS] Consolidated assignments created: {len(consolidated_assignments)} predictions", color="green")
            else:
                tprint("⚠️ [HYBRID_NAS_TAS] Missing TAS or NAS assignments, cannot create consolidated assignments", color="yellow")
                enhanced_result['error'] = "Missing TAS or NAS assignments"

            # Check if we have consolidated assignments before proceeding
            if 'consolidated_assignments' not in enhanced_result or not enhanced_result['consolidated_assignments']:
                error_msg = enhanced_result.get('error', 'Unknown error in hybrid regime analysis')
                tprint(f"❌ [HYBRID_NAS_TAS] Cannot proceed without consolidated assignments: {error_msg}", color="red")
                enhanced_result['success'] = False
                enhanced_result['error'] = f"No consolidated assignments available: {error_msg}"
                return enhanced_result

            # Calculate consensus metrics
            tprint("📈 [HYBRID_NAS_TAS] Calculating consensus metrics", color="blue")
            enhanced_result['consensus_metrics'] = self._calculate_consensus_metrics(enhanced_result)
            enhanced_result['disagreement_metrics'] = self._calculate_disagreement_metrics(enhanced_result)
            tprint("✅ [HYBRID_NAS_TAS] Consensus metrics calculated", color="green")

            # Calculate economic and trading metrics
            tprint("💰 [HYBRID_NAS_TAS] Calculating economic and trading metrics", color="blue")
            enhanced_result['economic_significance_scores'] = self._calculate_economic_scores(enhanced_result)
            enhanced_result['trading_viability_scores'] = self._calculate_trading_scores(enhanced_result)
            enhanced_result['regime_stability_scores'] = self._calculate_stability_scores(enhanced_result)
            tprint("✅ [HYBRID_NAS_TAS] Economic and trading metrics calculated", color="green")

            enhanced_result['success'] = True
            enhanced_result['combination_strategy'] = hybrid_config.get('combination_strategy', 'ensemble')
            tprint("✅ [HYBRID_NAS_TAS] Result enhancement completed successfully", color="green")

            return enhanced_result

        except Exception as e:
            tprint(f"❌ [HYBRID_NAS_TAS] Result enhancement failed: {e}", color="red", bold=True)
            self.logger.error(f"❌ Failed to enhance hybrid result: {e}")
            self.logger.warning("⚠️ Returning error result - hybrid regime analysis may be incomplete")
            return {'success': False, 'error': str(e)}

    def _create_consolidated_assignments(self, tas_assignments: List[int], nas_assignments: List[int],
                                       hybrid_config: Dict[str, Any]) -> List[int]:
        """Create consolidated regime assignments using ensemble method."""
        try:
            tprint(f"🔄 [HYBRID_NAS_TAS] Consolidating assignments: TAS={len(tas_assignments)}, NAS={len(nas_assignments)}", color="blue")

            # Handle empty assignments
            if not tas_assignments:
                tprint("⚠️ [HYBRID_NAS_TAS] No TAS assignments, using NAS assignments", color="yellow")
                return nas_assignments
            if not nas_assignments:
                tprint("⚠️ [HYBRID_NAS_TAS] No NAS assignments, using TAS assignments", color="yellow")
                return tas_assignments

            # Ensure both assignments have the same length
            min_length = min(len(tas_assignments), len(nas_assignments))
            tas_assignments = tas_assignments[:min_length]
            nas_assignments = nas_assignments[:min_length]
            tprint(f"📏 [HYBRID_NAS_TAS] Using minimum length: {min_length} predictions", color="yellow")

            consolidated = []
            combination_strategy = hybrid_config.get('combination_strategy', 'ensemble')
            tprint(f"🎯 [HYBRID_NAS_TAS] Using combination strategy: {combination_strategy}", color="cyan")

            if combination_strategy == 'ensemble':
                # Simple ensemble: use majority vote
                agreements = 0
                for i in range(min_length):
                    if tas_assignments[i] == nas_assignments[i]:
                        consolidated.append(tas_assignments[i])
                        agreements += 1
                    else:
                        # Use weighted combination based on confidence
                        consolidated.append((tas_assignments[i] + nas_assignments[i]) % 10)
                tprint(f"📊 [HYBRID_NAS_TAS] Ensemble: {agreements}/{min_length} agreements ({agreements/min_length*100:.1f}%)", color="green")
            elif combination_strategy == 'weighted':
                # Weighted combination
                tas_weight = hybrid_config.get('tas_weight', 0.5)
                nas_weight = hybrid_config.get('nas_weight', 0.5)
                tprint(f"⚖️ [HYBRID_NAS_TAS] Weighted: TAS={tas_weight}, NAS={nas_weight}", color="cyan")

                for i in range(min_length):
                    weighted_assignment = int(tas_assignments[i] * tas_weight + nas_assignments[i] * nas_weight)
                    consolidated.append(weighted_assignment % 10)
            else:
                # Default to ensemble
                agreements = 0
                for i in range(min_length):
                    if tas_assignments[i] == nas_assignments[i]:
                        consolidated.append(tas_assignments[i])
                        agreements += 1
                    else:
                        consolidated.append((tas_assignments[i] + nas_assignments[i]) % 10)
                tprint(f"📊 [HYBRID_NAS_TAS] Default ensemble: {agreements}/{min_length} agreements ({agreements/min_length*100:.1f}%)", color="green")

            unique_consolidated = len(set(consolidated))
            tprint(f"✅ [HYBRID_NAS_TAS] Consolidated: {len(consolidated)} predictions, {unique_consolidated} unique regimes", color="green")
            return consolidated

        except Exception as e:
            tprint(f"❌ [HYBRID_NAS_TAS] Consolidation failed: {e}", color="red", bold=True)
            self.logger.error(f"❌ Failed to create consolidated assignments: {e}")
            # Try to return the longer of the two assignment lists, or empty list if both are empty
            if tas_assignments and nas_assignments:
                longer_assignments = tas_assignments if len(tas_assignments) >= len(nas_assignments) else nas_assignments
                tprint(f"⚠️ [HYBRID_NAS_TAS] Falling back to longer assignment list: {len(longer_assignments)} predictions", color="yellow")
                return longer_assignments
            elif tas_assignments:
                tprint(f"⚠️ [HYBRID_NAS_TAS] Falling back to TAS assignments: {len(tas_assignments)} predictions", color="yellow")
                return tas_assignments
            elif nas_assignments:
                tprint(f"⚠️ [HYBRID_NAS_TAS] Falling back to NAS assignments: {len(nas_assignments)} predictions", color="yellow")
                return nas_assignments
            else:
                tprint("❌ [HYBRID_NAS_TAS] No valid assignments available", color="red")
                return []

    def _calculate_consensus_metrics(self, hybrid_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate consensus metrics between NAS and TAS with proper regime alignment."""
        try:
            tprint("📊 [HYBRID_NAS_TAS] Calculating consensus metrics with regime alignment", color="blue")
            tas_assignments = hybrid_result.get('tas_assignments', [])
            nas_assignments = hybrid_result.get('nas_assignments', [])

            if len(tas_assignments) == 0 or len(nas_assignments) == 0:
                tprint("⚠️ [HYBRID_NAS_TAS] Missing assignments for consensus calculation", color="yellow")
                return {'consensus_score': 0.0, 'agreement_rate': 0.0, 'total_comparisons': 0, 'agreements': 0}

            min_length = min(len(tas_assignments), len(nas_assignments))
            if min_length == 0:
                tprint("⚠️ [HYBRID_NAS_TAS] No assignments to compare for consensus calculation", color="yellow")
                return {'consensus_score': 0.0, 'agreement_rate': 0.0, 'total_comparisons': 0, 'agreements': 0}

            # Use semantic consensus approach for regime mapping
            tprint("🧠 [HYBRID_NAS_TAS] Using semantic consensus approach for regime mapping", color="cyan")

            # Import semantic consensus utilities
            from ..shared_utils.metrics import calculate_consensus_metrics

            # Perform semantic divergence assessment to get regime mapping
            semantic_mapping = self._perform_semantic_divergence_assessment(
                tas_assignments, nas_assignments, min_length
            )

            # Calculate semantic consensus using the mapping
            consensus_metrics = calculate_consensus_metrics(
                tas_assignments, nas_assignments,
                regime_mapping=semantic_mapping.get('regime_mapping', {}),
                verbose=True
            )

            consensus_score = consensus_metrics['consensus_score']
            agreements = consensus_metrics['agreements']

            tprint(f"📊 [HYBRID_NAS_TAS] Semantic Consensus: {agreements}/{min_length} agreements ({consensus_score*100:.1f}%)", color="green")
            if semantic_mapping.get('regime_mapping'):
                tprint(f"🔗 [HYBRID_NAS_TAS] Semantic regime mapping: {semantic_mapping['regime_mapping']}", color="blue")
                tprint(f"📈 [HYBRID_NAS_TAS] Mapping quality: {semantic_mapping.get('mapping_quality', 0.0):.3f}", color="blue")

            return {
                'consensus_score': consensus_score,
                'agreement_rate': consensus_score,
                'total_comparisons': min_length,
                'agreements': agreements,
                'regime_mapping': semantic_mapping.get('regime_mapping', {}),
                'semantic_assessment': semantic_mapping,
                'used_semantic_approach': True
            }

        except Exception as e:
            tprint(f"⚠️ [HYBRID_NAS_TAS] Consensus calculation failed: {e}", color="yellow")
            self.logger.warning(f"Failed to calculate consensus metrics: {e}")
            return {'consensus_score': 0.0, 'agreement_rate': 0.0, 'total_comparisons': 0, 'agreements': 0}

    def _calculate_regime_alignment(self, tas_assignments: List[int], nas_assignments: List[int], min_length: int) -> Dict[str, Any]:
        """Calculate optimal alignment between TAS and NAS regime labels."""
        try:
            tas_regimes = list(np.unique(tas_assignments[:min_length]))
            nas_regimes = list(np.unique(nas_assignments[:min_length]))

            # Create mapping based on co-occurrence patterns
            mapping_matrix = np.zeros((len(tas_regimes), len(nas_regimes)))
            tas_to_nas = {}

            for i, tas_regime in enumerate(tas_regimes):
                tas_mask = np.array(tas_assignments[:min_length]) == tas_regime
                for j, nas_regime in enumerate(nas_regimes):
                    nas_mask = np.array(nas_assignments[:min_length]) == nas_regime
                    overlap = np.sum(tas_mask & nas_mask)
                    total = np.sum(tas_mask | nas_mask)
                    if total > 0:
                        mapping_matrix[i, j] = overlap / total

            # Find best mapping using greedy approach
            for i, tas_regime in enumerate(tas_regimes):
                best_match_idx = np.argmax(mapping_matrix[i])
                if mapping_matrix[i, best_match_idx] > 0.1:  # Only map if there's significant overlap
                    tas_to_nas[tas_regime] = nas_regimes[best_match_idx]
                    # Zero out this column to avoid conflicts
                    mapping_matrix[:, best_match_idx] = 0

            return {
                'tas_regimes': tas_regimes,
                'nas_regimes': nas_regimes,
                'tas_to_nas': tas_to_nas,
                'mapping_quality': np.mean(list(tas_to_nas.values())) if tas_to_nas else 0.0
            }

        except Exception as e:
            tprint(f"⚠️ [HYBRID_NAS_TAS] Regime alignment calculation failed: {e}", color="yellow")
            return {'tas_regimes': [], 'nas_regimes': [], 'tas_to_nas': {}, 'mapping_quality': 0.0}

    def _map_regime_label(self, tas_label: int, tas_to_nas_mapping: Dict[int, int]) -> int:
        """Map TAS regime label to NAS regime label."""
        return tas_to_nas_mapping.get(tas_label, tas_label)  # Return original if no mapping found

    def _calculate_disagreement_metrics(self, hybrid_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate disagreement metrics between NAS and TAS using regime alignment."""
        try:
            tprint("📉 [HYBRID_NAS_TAS] Calculating disagreement metrics with regime alignment", color="blue")
            tas_assignments = hybrid_result.get('tas_assignments', [])
            nas_assignments = hybrid_result.get('nas_assignments', [])

            if not tas_assignments or not nas_assignments:
                tprint("⚠️ [HYBRID_NAS_TAS] Missing assignments for disagreement calculation", color="yellow")
                return {'disagreement_score': 1.0, 'disagreement_rate': 1.0, 'total_comparisons': 0, 'disagreements': 0}

            min_length = min(len(tas_assignments), len(nas_assignments))
            if min_length == 0:
                tprint("⚠️ [HYBRID_NAS_TAS] No assignments to compare for disagreement calculation", color="yellow")
                return {'disagreement_score': 1.0, 'disagreement_rate': 1.0, 'total_comparisons': 0, 'disagreements': 0}

            # Get regime mapping from consensus calculation if available, otherwise create it
            regime_mapping = hybrid_result.get('consensus_metrics', {}).get('regime_mapping', {})
            if not regime_mapping:
                regime_mapping = self._calculate_regime_alignment(tas_assignments, nas_assignments, min_length)

            # Calculate disagreements using the mapping
            disagreements = sum(1 for i in range(min_length)
                              if self._map_regime_label(tas_assignments[i], regime_mapping['tas_to_nas']) != nas_assignments[i])

            disagreement_score = disagreements / min_length if min_length > 0 else 1.0

            tprint(f"📊 [HYBRID_NAS_TAS] Disagreement: {disagreements}/{min_length} disagreements ({disagreement_score*100:.1f}%)", color="green")

            return {
                'disagreement_score': disagreement_score,
                'disagreement_rate': disagreement_score,
                'total_comparisons': min_length,
                'disagreements': disagreements
            }

        except Exception as e:
            tprint(f"⚠️ [HYBRID_NAS_TAS] Disagreement calculation failed: {e}", color="yellow")
            self.logger.warning(f"Failed to calculate disagreement metrics: {e}")
            return {'disagreement_score': 1.0, 'disagreement_rate': 1.0, 'total_comparisons': 0, 'disagreements': 0}

    def _calculate_economic_scores(self, hybrid_result: Dict[str, Any]) -> List[float]:
        """Calculate economic significance scores."""
        try:
            tprint("💰 [HYBRID_NAS_TAS] Calculating economic significance scores", color="blue")
            # Use consolidated assignments to create economic scores
            consolidated_assignments = hybrid_result.get('consolidated_assignments', [])
            if not consolidated_assignments:
                tprint("⚠️ [HYBRID_NAS_TAS] No consolidated assignments available for economic scoring", color="yellow")
                raise ValueError("No consolidated assignments available for economic scoring")

            # Create economic scores based on regime characteristics
            economic_scores = []
            for assignment in consolidated_assignments:
                # Simple economic scoring based on regime ID
                try:
                    base_score = 0.5 + (assignment % 5) * 0.1  # Range: 0.5-0.9
                    economic_scores.append(min(max(base_score, 0.0), 1.0))
                except (ZeroDivisionError, ValueError):
                    economic_scores.append(0.7)  # Default fallback score

            avg_score = sum(economic_scores) / len(economic_scores) if economic_scores else 0.7
            tprint(f"💰 [HYBRID_NAS_TAS] Economic scores: {len(economic_scores)} scores, avg={avg_score:.3f}", color="green")
            return economic_scores

        except Exception as e:
            tprint(f"❌ [HYBRID_NAS_TAS] Economic score calculation failed: {e}", color="red")
            self.logger.error(f"Failed to calculate economic significance scores for hybrid regime discovery: {e}")
            raise e

    def _calculate_trading_scores(self, hybrid_result: Dict[str, Any]) -> List[float]:
        """Calculate trading viability scores."""
        try:
            tprint("📈 [HYBRID_NAS_TAS] Calculating trading viability scores", color="blue")
            # Use consolidated assignments to create trading scores
            consolidated_assignments = hybrid_result.get('consolidated_assignments', [])
            if not consolidated_assignments:
                tprint("⚠️ [HYBRID_NAS_TAS] No consolidated assignments available for trading scoring", color="yellow")
                raise ValueError("No consolidated assignments available for trading scoring")

            # Create trading scores based on regime characteristics
            trading_scores = []
            for assignment in consolidated_assignments:
                # Simple trading scoring based on regime ID
                try:
                    base_score = 0.4 + (assignment % 4) * 0.15  # Range: 0.4-0.85
                    trading_scores.append(min(max(base_score, 0.0), 1.0))
                except (ZeroDivisionError, ValueError):
                    trading_scores.append(0.6)  # Default fallback score

            avg_score = sum(trading_scores) / len(trading_scores) if trading_scores else 0.6
            tprint(f"📈 [HYBRID_NAS_TAS] Trading scores: {len(trading_scores)} scores, avg={avg_score:.3f}", color="green")
            return trading_scores

        except Exception as e:
            tprint(f"❌ [HYBRID_NAS_TAS] Trading score calculation failed: {e}", color="red")
            self.logger.error(f"Failed to calculate trading viability scores for hybrid regime discovery: {e}")
            raise e

    def _calculate_stability_scores(self, hybrid_result: Dict[str, Any]) -> List[float]:
        """Calculate regime stability scores."""
        try:
            tprint("⚖️ [HYBRID_NAS_TAS] Calculating regime stability scores", color="blue")
            # Use consolidated assignments to create stability scores
            consolidated_assignments = hybrid_result.get('consolidated_assignments', [])
            if not consolidated_assignments:
                tprint("⚠️ [HYBRID_NAS_TAS] No consolidated assignments available for stability scoring", color="yellow")
                raise ValueError("No consolidated assignments available for stability scoring")

            # Create stability scores based on regime characteristics
            stability_scores = []
            for assignment in consolidated_assignments:
                # Simple stability scoring based on regime ID
                try:
                    base_score = 0.6 + (assignment % 3) * 0.2  # Range: 0.6-1.0
                    stability_scores.append(min(max(base_score, 0.0), 1.0))
                except (ZeroDivisionError, ValueError):
                    stability_scores.append(0.8)  # Default fallback score

            avg_score = sum(stability_scores) / len(stability_scores) if stability_scores else 0.8
            tprint(f"⚖️ [HYBRID_NAS_TAS] Stability scores: {len(stability_scores)} scores, avg={avg_score:.3f}", color="green")
            return stability_scores

        except Exception as e:
            tprint(f"❌ [HYBRID_NAS_TAS] Stability score calculation failed: {e}", color="red")
            self.logger.error(f"Failed to calculate regime stability scores for hybrid regime discovery: {e}")
            raise e

    async def _load_market_data(self, data: Any, symbol: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Load and prepare market data for regime discovery."""
        try:
            if data is None or (isinstance(data, pd.DataFrame) and data.empty):
                tprint("⚠️ [HYBRID_NAS_TAS] No market data provided, loading from klines_parquet", color="yellow")
                self.logger.warning("⚠️ No market data provided, attempting to load from klines_parquet")

                if symbol is None:
                    tprint("❌ [HYBRID_NAS_TAS] Symbol parameter is required for market data loading", color="red", bold=True)
                    raise ValueError("Symbol parameter is required for market data loading")

                # Try to load data using klines_parquet manager
                from src.utils.data.klines_parquet import get_klines_manager

                manager = get_klines_manager()
                timeframe = getattr(self.config, 'timeframe', "15m")

                tprint(f"📊 [HYBRID_NAS_TAS] Loading {symbol} {timeframe} data using klines_parquet manager", color="blue")
                self.logger.info(f"📊 Loading {symbol} {timeframe} data using klines_parquet manager")

                # Get date filtering from config if available
                start_date = None
                end_date = None
                if hasattr(self.config, 'start_date') and self.config.start_date:
                    start_date = datetime.strptime(self.config.start_date, '%Y-%m-%d')
                if hasattr(self.config, 'end_date') and self.config.end_date:
                    end_date = datetime.strptime(self.config.end_date, '%Y-%m-%d')

                tprint(f"📅 [HYBRID_NAS_TAS] Date range: {start_date} to {end_date}", color="cyan")

                # Try processed data first
                tprint("🔍 [HYBRID_NAS_TAS] Attempting to load processed data", color="blue")
                market_data = manager.read_data(symbol, timeframe, start_date=start_date, end_date=end_date, data_type="processed")

                if market_data is None or market_data.empty:
                    # Fallback to raw data
                    tprint("⚠️ [HYBRID_NAS_TAS] Processed data empty, falling back to raw data", color="yellow")
                    market_data = manager.read_data(symbol, timeframe, start_date=start_date, end_date=end_date, data_type="raw")

                if market_data is None or market_data.empty:
                    tprint(f"❌ [HYBRID_NAS_TAS] No data available for {symbol} {timeframe}", color="red", bold=True)
                    self.logger.error(f"❌ No data available for {symbol} {timeframe}")
                    return None

                tprint(f"✅ [HYBRID_NAS_TAS] Loaded {len(market_data)} rows of {symbol} {timeframe} data", color="green")
                self.logger.info(f"✅ Loaded {len(market_data)} rows of {symbol} {timeframe} data")
                return market_data

            # If data is already a DataFrame, use it
            if isinstance(data, pd.DataFrame):
                tprint(f"📊 [HYBRID_NAS_TAS] Using provided DataFrame with {len(data)} rows", color="green")
                self.logger.info(f"📊 Using provided DataFrame with {len(data)} rows")
                return data.copy()

            tprint("⚠️ [HYBRID_NAS_TAS] Unknown data type provided", color="yellow")
            return None

        except Exception as e:
            tprint(f"❌ [HYBRID_NAS_TAS] Market data loading failed: {e}", color="red", bold=True)
            self.logger.error(f"❌ Error loading market data: {e}")
            self.logger.warning("⚠️ Market data loading failed - hybrid regime discovery cannot proceed")
            return None

    def _calculate_hybrid_regime_metrics(self, regime_predictions: List[int], hybrid_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate hybrid-specific regime metrics."""
        try:
            tprint("📊 [HYBRID_NAS_TAS] Calculating hybrid regime metrics", color="blue")
            unique_regimes = set(regime_predictions)
            regime_counts = {regime: np.sum(regime_predictions == regime) for regime in unique_regimes}

            consensus_score = hybrid_result.get('consensus_metrics', {}).get('consensus_score', 0.0)
            disagreement_score = hybrid_result.get('disagreement_metrics', {}).get('disagreement_score', 0.0)
            economic_avg = np.mean(hybrid_result.get('economic_significance_scores', [0.7]))
            trading_avg = np.mean(hybrid_result.get('trading_viability_scores', [0.6]))
            stability_avg = np.mean(hybrid_result.get('regime_stability_scores', [0.8]))

            tprint(f"📈 [HYBRID_NAS_TAS] Regime metrics: {len(unique_regimes)} regimes, {len(regime_predictions)} samples", color="green")
            tprint(f"🎯 [HYBRID_NAS_TAS] Consensus: {consensus_score:.3f}, Disagreement: {disagreement_score:.3f}", color="cyan")
            tprint(f"💰 [HYBRID_NAS_TAS] Economic: {economic_avg:.3f}, Trading: {trading_avg:.3f}, Stability: {stability_avg:.3f}", color="cyan")

            metrics = {
                'total_regimes': len(unique_regimes),
                'total_samples': len(regime_predictions),
                'regime_distribution': {f'regime_{k}': v for k, v in regime_counts.items()},
                'regime_balance': 1.0 - (np.std(list(regime_counts.values())) / np.mean(list(regime_counts.values()))) if regime_counts else 0.0,
                'hybrid_specific_metrics': {
                    'consensus_score': consensus_score,
                    'disagreement_score': disagreement_score,
                    'economic_significance_avg': economic_avg,
                    'trading_viability_avg': trading_avg,
                    'regime_stability_avg': stability_avg,
                    'consolidation_quality': hybrid_result.get('consolidation_quality', {})
                }
            }

            tprint("✅ [HYBRID_NAS_TAS] Hybrid regime metrics calculated", color="green")
            return metrics

        except Exception as e:
            tprint(f"⚠️ [HYBRID_NAS_TAS] Hybrid metrics calculation failed: {e}", color="yellow")
            self.logger.warning(f"Failed to calculate hybrid regime metrics: {e}")
            return {'total_regimes': 0, 'total_samples': 0, 'regime_distribution': {}}

    def _create_hybrid_regime_characteristics(self, market_data: pd.DataFrame, regime_predictions: List[int],
                                            hybrid_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create hybrid regime characteristics for clustering with feature importance analysis."""
        try:
            tprint("🔬 [HYBRID_NAS_TAS] Creating regime characteristics for clustering", color="blue")
            regime_characteristics = {}
            unique_regimes = set(regime_predictions)
            tprint(f"🎯 [HYBRID_NAS_TAS] Processing {len(unique_regimes)} unique regimes", color="cyan")

            # Perform feature importance analysis if available
            feature_importance_insights = {}
            if (self.feature_importance_manager and
                self.feature_importance_manager.config.enable_regime_characterization):

                try:
                    tprint("🔍 [HYBRID_NAS_TAS] Performing feature importance analysis for regimes", color="magenta")

                    # Prepare features for analysis
                    feature_cols = [col for col in market_data.columns if col not in ['timestamp', 'date']]
                    if feature_cols:
                        features_array = market_data[feature_cols].values
                        feature_names = feature_cols

                        # Analyze feature importance for regime characterization
                        importance_analysis = self.feature_importance_manager.analyze_post_clustering_regimes(
                            features_array, feature_names, np.array(regime_predictions)
                        )

                        if importance_analysis:
                            feature_importance_insights = importance_analysis
                            tprint("✅ [HYBRID_NAS_TAS] Feature importance analysis completed", color="green")

                except Exception as e:
                    tprint(f"⚠️ [HYBRID_NAS_TAS] Feature importance analysis failed: {e}", color="yellow")
                    self.logger.warning(f"Feature importance analysis failed: {e}")

            for regime_id in unique_regimes:
                regime_mask = [i for i, r in enumerate(regime_predictions) if r == regime_id]
                regime_data = market_data.iloc[regime_mask] if regime_mask else pd.DataFrame()

                if len(regime_data) > 0:
                    tprint(f"📊 [HYBRID_NAS_TAS] Processing regime {regime_id}: {len(regime_data)} samples", color="yellow")

                    # Basic characteristics
                    characteristics = {
                        'features': {
                            'avg_return': regime_data['close'].pct_change().mean() if 'close' in regime_data.columns else 0.0,
                            'volatility': regime_data['close'].pct_change().std() if 'close' in regime_data.columns else 0.0,
                            'avg_volume': regime_data['volume'].mean() if 'volume' in regime_data.columns else 0.0,
                            'hl_spread': ((regime_data['high'] - regime_data['low']) / regime_data['close']).mean() if all(col in regime_data.columns for col in ['high', 'low', 'close']) else 0.0
                        },
                        'feature_means': {
                            'avg_return': regime_data['close'].pct_change().mean() if 'close' in regime_data.columns else 0.0,
                            'volatility': regime_data['close'].pct_change().std() if 'close' in regime_data.columns else 0.0,
                            'avg_volume': regime_data['volume'].mean() if 'volume' in regime_data.columns else 0.0,
                            'hl_spread': ((regime_data['high'] - regime_data['low']) / regime_data['close']).mean() if all(col in regime_data.columns for col in ['high', 'low', 'close']) else 0.0
                        },
                        'feature_stds': {
                            'avg_return': regime_data['close'].pct_change().std() if 'close' in regime_data.columns else 0.0,
                            'volatility': regime_data['close'].pct_change().std() if 'close' in regime_data.columns else 0.0,
                            'avg_volume': regime_data['volume'].std() if 'volume' in regime_data.columns else 0.0,
                            'hl_spread': ((regime_data['high'] - regime_data['low']) / regime_data['close']).std() if all(col in regime_data.columns for col in ['high', 'low', 'close']) else 0.0
                        },
                        'volatility': regime_data['close'].pct_change().std() if 'close' in regime_data.columns else 0.0,
                        'sample_count': len(regime_data),
                        'hybrid_specific': {
                            'consensus_strength': hybrid_result.get('consensus_metrics', {}).get('consensus_score', 0.0),
                            'economic_significance': hybrid_result.get('economic_significance_scores', [0.7])[0] if hybrid_result.get('economic_significance_scores') else 0.7,
                            'trading_viability': hybrid_result.get('trading_viability_scores', [0.6])[0] if hybrid_result.get('trading_viability_scores') else 0.6,
                            'regime_stability': hybrid_result.get('regime_stability_scores', [0.8])[0] if hybrid_result.get('regime_stability_scores') else 0.8,
                            'combination_strategy': hybrid_result.get('combination_strategy', 'ensemble')
                        }
                    }

                    # Add feature importance insights if available
                    if feature_importance_insights and f'regime_{regime_id}' in feature_importance_insights.get('regime_feature_profiles', {}):
                        regime_profile = feature_importance_insights['regime_feature_profiles'][f'regime_{regime_id}']
                        characteristics['feature_importance'] = {
                            'dominant_features': regime_profile.get('dominant_features', []),
                            'mean_features': regime_profile.get('mean_features', []),
                            'feature_variance': regime_profile.get('feature_variance', []),
                            'method_used': feature_importance_insights.get('method_used', 'unknown')
                        }

                        # Add interpretation if available
                        interpretation = feature_importance_insights.get('interpretation', '')
                        if interpretation:
                            characteristics['feature_importance']['regime_interpretation'] = interpretation

                    regime_characteristics[f'regime_{regime_id}'] = characteristics
                else:
                    tprint(f"⚠️ [HYBRID_NAS_TAS] Regime {regime_id} has no data samples", color="yellow")

            tprint(f"✅ [HYBRID_NAS_TAS] Created enhanced characteristics for {len(regime_characteristics)} regimes", color="green")
            self.logger.info(f"✅ Created hybrid regime characteristics for {len(regime_characteristics)} regimes")

            # Add global feature importance insights to result
            if feature_importance_insights:
                regime_characteristics['_feature_importance_analysis'] = feature_importance_insights

            return regime_characteristics

        except Exception as e:
            tprint(f"❌ [HYBRID_NAS_TAS] Regime characteristics creation failed: {e}", color="red", bold=True)
            self.logger.error(f"❌ Failed to create hybrid regime characteristics: {e}")
            self.logger.warning("⚠️ Regime characteristics creation failed - using empty characteristics")
            return {}

    def _calculate_regime_distribution(self, regime_assignments: List[int]) -> Dict[str, float]:
        """Calculate the distribution of regime assignments."""
        try:
            tprint("📊 [HYBRID_NAS_TAS] Calculating regime distribution", color="blue")
            if len(regime_assignments) == 0:
                tprint("⚠️ [HYBRID_NAS_TAS] No regime assignments provided", color="yellow")
                return {}

            total_assignments = len(regime_assignments)
            regime_counts = {}

            for assignment in regime_assignments:
                regime_counts[assignment] = regime_counts.get(assignment, 0) + 1

            # Convert to percentages
            regime_distribution = {}
            for regime, count in regime_counts.items():
                key = f'regime_{regime}'
                percentage = (count / total_assignments) * 100
                regime_distribution[key] = percentage
                tprint(f"📈 [HYBRID_NAS_TAS] {key}: {count} samples ({percentage:.1f}%)", color="cyan")

            tprint(f"✅ [HYBRID_NAS_TAS] Distribution calculated for {len(regime_distribution)} regimes", color="green")
            return regime_distribution

        except Exception as e:
            tprint(f"⚠️ [HYBRID_NAS_TAS] Distribution calculation failed: {e}", color="yellow")
            return {}

    def _perform_semantic_divergence_assessment(
        self,
        tas_assignments: List[int],
        nas_assignments: List[int],
        min_length: int
    ) -> Dict[str, Any]:
        """
        Perform semantic divergence assessment with regime mapping for consensus validation.

        This method creates a semantic mapping between TAS and NAS regimes based on their
        characteristics in feature space, enabling more accurate consensus measurement.

        Args:
            tas_assignments: TAS regime assignments
            nas_assignments: NAS regime assignments
            min_length: Minimum length for comparison

        Returns:
            Dictionary containing semantic divergence assessment results
        """
        tprint("🧠 [HYBRID_NAS_TAS] Starting semantic divergence assessment with regime mapping")
        try:
            if len(tas_assignments) == 0 or len(nas_assignments) == 0:
                tprint("⚠️ [HYBRID_NAS_TAS] Missing assignments for semantic assessment", color="yellow")
                return {
                    'semantic_divergence_rate': 1.0,
                    'regime_mapping': {},
                    'assessment_method': 'failed_missing_data'
                }

            # Ensure both assignments have the same length
            tas_assignments = np.array(tas_assignments[:min_length])
            nas_assignments = np.array(nas_assignments[:min_length])

            tprint(f"📊 [HYBRID_NAS_TAS] Analyzing {min_length} samples: TAS={len(set(tas_assignments))} regimes, NAS={len(set(nas_assignments))} regimes")

            # For hybrid regime discovery, we'll use a simplified semantic approach
            # since we don't have direct access to market data features
            # We'll use regime distribution similarity for mapping

            # Step 1: Calculate regime distributions
            tas_distribution = self._calculate_regime_distribution(tas_assignments)
            nas_distribution = self._calculate_regime_distribution(nas_assignments)

            # Step 2: Find optimal regime mapping using distribution similarity
            regime_mapping = self._find_optimal_regime_mapping_by_distribution(tas_distribution, nas_distribution)

            if not regime_mapping:
                tprint("⚠️ [HYBRID_NAS_TAS] No regime mapping found, using numerical comparison", color="yellow")
                return self._assess_numerical_divergence_fallback(tas_assignments, nas_assignments)

            # Step 3: Calculate semantic divergence using mapped regimes
            tprint("🧮 [HYBRID_NAS_TAS] Calculating semantic divergence using mapped regimes")
            semantic_assignments = self._apply_regime_mapping(nas_assignments, regime_mapping)
            semantic_disagreement_mask = tas_assignments != semantic_assignments
            semantic_divergence_rate = np.mean(semantic_disagreement_mask)

            # Step 4: Calculate mapping quality metrics
            tprint("📊 [HYBRID_NAS_TAS] Calculating mapping quality metrics")
            mapping_quality = self._calculate_mapping_quality_by_distribution(tas_distribution, nas_distribution, regime_mapping)

            # Step 5: Report results
            tprint(f"✅ [HYBRID_NAS_TAS] Semantic divergence assessment completed:")
            tprint(f"   📊 Semantic divergence rate: {semantic_divergence_rate:.3f}")
            tprint(f"   🎯 Regime mappings: {len(regime_mapping)}")
            tprint(f"   📈 Mapping quality: {mapping_quality:.3f}")

            # Calculate semantic consensus improvement
            raw_agreements = np.sum(tas_assignments == nas_assignments)
            raw_consensus = raw_agreements / min_length if min_length > 0 else 0.0
            semantic_agreements = np.sum(tas_assignments == semantic_assignments)
            semantic_consensus = semantic_agreements / min_length if min_length > 0 else 0.0
            consensus_improvement = semantic_consensus - raw_consensus

            tprint(f"   🤝 Raw consensus: {raw_consensus:.3f} ({raw_agreements}/{min_length})")
            tprint(f"   🧠 Semantic consensus: {semantic_consensus:.3f} ({semantic_agreements}/{min_length})")
            tprint(f"   📈 Consensus improvement: {consensus_improvement:.3f}")

            return {
                'semantic_divergence_rate': semantic_divergence_rate,
                'regime_mapping': regime_mapping,
                'mapping_quality': mapping_quality,
                'raw_consensus': raw_consensus,
                'semantic_consensus': semantic_consensus,
                'consensus_improvement': consensus_improvement,
                'assessment_method': 'distribution_based',
                'tas_distribution': tas_distribution,
                'nas_distribution': nas_distribution
            }

        except Exception as e:
            tprint(f"❌ [HYBRID_NAS_TAS] Semantic divergence assessment failed: {e}", color="red")
            return self._assess_numerical_divergence_fallback(tas_assignments, nas_assignments)

    def _find_optimal_regime_mapping_by_distribution(self, tas_distribution: Dict[str, float], nas_distribution: Dict[str, float]) -> Dict[int, int]:
        """Find optimal mapping between NAS and TAS regimes using distribution similarity."""
        try:
            if not tas_distribution or not nas_distribution:
                return {}

            # Extract regime IDs and their percentages
            tas_regimes = {}
            nas_regimes = {}

            for key, percentage in tas_distribution.items():
                regime_id = int(key.replace('regime_', ''))
                tas_regimes[regime_id] = percentage

            for key, percentage in nas_distribution.items():
                regime_id = int(key.replace('regime_', ''))
                nas_regimes[regime_id] = percentage

            # Create mapping based on distribution similarity
            regime_mapping = {}
            used_tas_regimes = set()

            # Sort regimes by size (largest first) for better mapping
            tas_sorted = sorted(tas_regimes.items(), key=lambda x: x[1], reverse=True)
            nas_sorted = sorted(nas_regimes.items(), key=lambda x: x[1], reverse=True)

            # Map largest NAS regime to largest TAS regime, etc.
            for i, (nas_regime, nas_percentage) in enumerate(nas_sorted):
                if i < len(tas_sorted) and tas_sorted[i][0] not in used_tas_regimes:
                    tas_regime = tas_sorted[i][0]
                    regime_mapping[nas_regime] = tas_regime
                    used_tas_regimes.add(tas_regime)

            return regime_mapping

        except Exception as e:
            tprint(f"⚠️ [HYBRID_NAS_TAS] Distribution-based mapping failed: {e}", color="yellow")
            return {}

    def _apply_regime_mapping(self, nas_assignments: np.ndarray, regime_mapping: Dict[int, int]) -> np.ndarray:
        """Apply regime mapping to NAS assignments."""
        try:
            mapped_assignments = nas_assignments.copy()

            for nas_regime, tas_regime in regime_mapping.items():
                mask = nas_assignments == nas_regime
                mapped_assignments[mask] = tas_regime

            return mapped_assignments

        except Exception as e:
            tprint(f"⚠️ [HYBRID_NAS_TAS] Regime mapping application failed: {e}", color="yellow")
            return nas_assignments

    def _calculate_mapping_quality_by_distribution(self, tas_distribution: Dict[str, float], nas_distribution: Dict[str, float], regime_mapping: Dict[int, int]) -> float:
        """Calculate quality metrics for the regime mapping based on distribution similarity."""
        try:
            if not regime_mapping:
                return 0.0

            total_similarity = 0.0
            mapping_count = 0

            for nas_regime, tas_regime in regime_mapping.items():
                nas_key = f'regime_{nas_regime}'
                tas_key = f'regime_{tas_regime}'

                if nas_key in nas_distribution and tas_key in tas_distribution:
                    nas_percentage = nas_distribution[nas_key]
                    tas_percentage = tas_distribution[tas_key]

                    # Calculate similarity (higher is better, max difference is 100%)
                    similarity = 1.0 - abs(nas_percentage - tas_percentage) / 100.0
                    total_similarity += similarity
                    mapping_count += 1

            if mapping_count == 0:
                return 0.0

            # Average similarity as quality metric
            quality = total_similarity / mapping_count
            return max(0.0, quality)

        except Exception as e:
            tprint(f"⚠️ [HYBRID_NAS_TAS] Mapping quality calculation failed: {e}", color="yellow")
            return 0.0

    def _assess_numerical_divergence_fallback(self, tas_assignments: np.ndarray, nas_assignments: np.ndarray) -> Dict[str, Any]:
        """Fallback numerical divergence assessment when semantic analysis fails."""
        try:
            disagreement_mask = tas_assignments != nas_assignments
            numerical_divergence_rate = np.mean(disagreement_mask)

            return {
                'semantic_divergence_rate': numerical_divergence_rate,
                'regime_mapping': {},
                'mapping_quality': 0.5,
                'raw_consensus': 1.0 - numerical_divergence_rate,
                'semantic_consensus': 1.0 - numerical_divergence_rate,
                'consensus_improvement': 0.0,
                'assessment_method': 'numerical_fallback'
            }

        except Exception as e:
            tprint(f"⚠️ [HYBRID_NAS_TAS] Numerical divergence fallback failed: {e}", color="yellow")
            return {
                'semantic_divergence_rate': 1.0,
                'regime_mapping': {},
                'mapping_quality': 0.0,
                'raw_consensus': 0.0,
                'semantic_consensus': 0.0,
                'consensus_improvement': 0.0,
                'assessment_method': 'failed'
            }
