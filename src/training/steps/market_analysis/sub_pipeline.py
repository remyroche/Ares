"""
Market Analysis Sub-Pipeline - Complete 9-Step Pipeline

This module provides the complete market analysis sub-pipeline with exactly 9 required steps:

1. sr_parameter_optimization - Optimize SR detection levels
2. sr_detection - Detect Support/Resistance levels
3. sr_clustering - Generate SR clusters
4. nas_tas_regime_discovery - Discover market regimes using hybrid NAS-TAS approach
5. nas_tas_clustering - NAS-TAS-based regime clustering
        6. regime_models_training - Regime detection models training (CatBoost, Bayesian Rule Lists, ExtraTrees)
        7. regime_ensemble_training - Meta-learner training (stacker_lgbm_calibrated)
8. regime_data_splitting - Tag data by regimes
9. sr_feature_integration - Integrate SR-specific features into feature set
"""

import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.utils.enhanced_artifact_manager import get_artifact_manager
from src.utils.version_manager import get_version_manager
from src.utils.tprint import tprint
from src.training.config.data_locator import DataLocator, DataLocatorConfig, LocatorPaths
from .logging_standards import (
    get_logger, log_info, log_warning, log_error, log_success, log_debug,
    LoggingContext, log_step_progress, log_data_info, log_validation_result
)

# Import component system
from .components import ComponentFactory, ComponentConfig

# Import unified NAS/TAS pipeline
try:
    from src.nas_tas.unified_pipeline import (
        UnifiedNASPipeline, UnifiedTASPipeline, UnifiedHybridPipeline,
        create_nas_pipeline, create_tas_pipeline, create_hybrid_pipeline
    )
    UNIFIED_PIPELINE_AVAILABLE = True
except ImportError:
    UNIFIED_PIPELINE_AVAILABLE = False

# Import feature importance integration if available
try:
    from .shared_utils.feature_importance_integration import (
        FeatureImportanceIntegrationManager, FeatureImportanceIntegrationConfig,
        FeatureImportancePipelineHook
    )
    FEATURE_IMPORTANCE_AVAILABLE = True
except ImportError:
    FEATURE_IMPORTANCE_AVAILABLE = False

logger = system_logger.getChild('MarketAnalysisSubPipeline')

class ExecutionMode(Enum):
    """Execution modes for sub-pipelines."""
    FULL = "full"          # Complete execution with all features
    LIGHT = "light"        # Lightweight execution with essential features only
    BLANK = "blank"        # Minimal execution for testing/validation

class SubPipelineStatus(Enum):
    """Status of sub-pipeline execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class LoggingConfig:
    """Logging configuration for the sub-pipeline."""
    level: str = "INFO"
    enable_console: bool = True
    enable_file: bool = False
    log_file: Optional[str] = None

DEFAULT_DATA_DIR = "historical_data"


@dataclass
class SubPipelineConfig:
    """Configuration for sub-pipeline execution."""
    mode: ExecutionMode = ExecutionMode.FULL
    symbol: str = "ETHUSDT"
    exchange: str = "binance"
    timeframe: str = "4h"  # Updated default timeframe for regime detection
    data_dir: str = DEFAULT_DATA_DIR
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    force_rerun: bool = False
    parallel_processing: bool = True
    max_workers: int = 4
    validation_enabled: bool = True
    monitoring_enabled: bool = True
    fast_mode: bool = False
    skip_next_pipeline: bool = False
    single_stage_only: bool = False  # New parameter to control single vs sequential execution
    custom_params: Dict[str, Any] = field(default_factory=dict)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Direction control (optional, not used by market analysis but accepted for compatibility)
    enable_long_positions: bool = True
    enable_short_positions: bool = True
    
    # Unified pipeline configuration
    use_unified_pipeline: bool = True  # Default to unified pipeline
    unified_pipeline_mode: str = "hybrid"  # "nas", "tas", or "hybrid"
    unified_pipeline_fallback: bool = True  # Fallback to legacy if unified fails

    # Feature importance analysis configuration
    enable_feature_importance: bool = True
    feature_importance_methods: List[str] = field(default_factory=lambda: ["mutual_information", "f_classif"])
    enable_pre_clustering_analysis: bool = True
    enable_post_clustering_analysis: bool = True
    enable_regime_characterization: bool = True
    data_locator_config: DataLocatorConfig = field(default_factory=DataLocatorConfig)
    data_locator: Optional[DataLocator] = None
    data_dir_key: str = "market_data"
    cache_dir_key: str = "default"
    artifacts_dir_key: str = "default"
    generated_dir_key: str = "market_analysis"
    config_dir_key: str = "multi_horizon_labeling"
    _path_view: Optional[LocatorPaths] = field(default=None, init=False, repr=False)

    def attach_locator(self, locator: DataLocator) -> None:
        """Attach a :class:`DataLocator` instance to the configuration."""

        self.data_locator = locator
        self._path_view = LocatorPaths(locator)

    def _ensure_paths(self) -> LocatorPaths:
        if self.data_locator is None:
            self.attach_locator(DataLocator(self.data_locator_config))
        elif self._path_view is None or self._path_view.locator is not self.data_locator:
            self._path_view = LocatorPaths(self.data_locator)
        return self._path_view

    @property
    def paths(self) -> LocatorPaths:
        return self._ensure_paths()

    @property
    def data(self):
        return self.paths.data

    @property
    def cache(self):
        return self.paths.cache

    @property
    def artifacts(self):
        return self.paths.artifacts

    @property
    def generated(self):
        return self.paths.generated

    @property
    def config_paths(self):
        return self.paths.config

    @property
    def config_files(self):
        return self.paths.config

    @property
    def config_root(self) -> Path:
        return self.paths.config.root

    @property
    def config(self):
        return self.paths.config

@dataclass
class SubPipelineResult:
    """Result of sub-pipeline execution."""
    sub_pipeline_name: str
    status: SubPipelineStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    output_files: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        """Check if sub-pipeline completed successfully."""
        return self.status == SubPipelineStatus.COMPLETED and self.error_message is None
    
    @property
    def is_complete(self) -> bool:
        """Check if sub-pipeline produced a complete report with all required artifacts."""
        if not self.success:
            return False
        
        # Define required artifacts for each sub-pipeline
        required_artifacts = self._get_required_artifacts()
        
        # Check if all required artifacts are present and non-empty
        for artifact_name in required_artifacts:
            if artifact_name not in self.artifacts:
                return False
            artifact_value = self.artifacts[artifact_name]
            
            # Check for empty values
            if artifact_value is None:
                return False
            if isinstance(artifact_value, (list, dict)) and len(artifact_value) == 0:
                return False
            if isinstance(artifact_value, str) and artifact_value.strip() == "":
                return False
        
        return True
    
    def _get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts for this sub-pipeline."""
        artifact_requirements = {
            'sr_parameter_optimization': ['sr_parameter_optimization_result'],
            'sr_detection': ['sr_detection_result'],
            'sr_clustering': ['sr_clustering_result'],
            'nas_tas_regime_discovery': ['nas_tas_regime_discovery_result'],
            'nas_tas_clustering': ['optimal_regime_clustering_result'],
            'nas_tas_models_training': ['nas_tas_models_training_result'],
            'nas_tas_ensemble_training': ['nas_tas_ensemble_training_result'],
            'regime_data_splitting': ['regime_data_splitting_result'],
            # Support both legacy and current naming for the multi-horizon step
            'multi_horizon_labeling': ['multi_horizon_labeling_result'],
        }
        return artifact_requirements.get(self.sub_pipeline_name, [])
    
    @property
    def execution_time(self) -> float:
        """Get execution time in seconds."""
        return self.duration_seconds or 0.0

class MarketAnalysisSubPipeline:
    """
    Market Analysis Sub-Pipeline Manager.
    
    Provides granular control over market analysis processes with different
    execution modes and comprehensive monitoring.
    """
    
    def __init__(self, config: Optional[SubPipelineConfig] = None):
        """Initialize the market analysis sub-pipeline with backward compatibility."""
        # Handle both old dict config and new SubPipelineConfig
        if isinstance(config, dict):
            # Convert old config format to SubPipelineConfig
            self.original_config = config
            self.config = self._convert_old_config(config)
        else:
            # Use provided SubPipelineConfig or create default
            self.config = config or SubPipelineConfig()
            self.original_config = {}
        
        # Use standardized logging
        self.logger = get_logger('MarketAnalysisSubPipeline')
        self.results: List[SubPipelineResult] = []

        # Apply logging configuration
        self._apply_logging_config(self.config.logging)

        # Locator state for filesystem management
        self._data_locator: Optional[DataLocator] = None
        self._configuration_logged = False
        self._prepare_filesystem(self.config)

        # Initialize artifact and version managers
        self.artifact_manager = get_artifact_manager()
        self.version_manager = get_version_manager()
        
        # Initialize component factory
        self.component_factory = ComponentFactory()

        # Initialize feature importance manager if available
        self.feature_importance_manager = None
        if FEATURE_IMPORTANCE_AVAILABLE and self.config.enable_feature_importance:
            try:
                importance_config = FeatureImportanceIntegrationConfig(
                    enable_pre_clustering_analysis=self.config.enable_pre_clustering_analysis,
                    enable_post_clustering_analysis=self.config.enable_post_clustering_analysis,
                    enable_regime_characterization=self.config.enable_regime_characterization,
                    importance_methods=self.config.feature_importance_methods,
                    auto_integrate_with_clustering=True,
                    auto_integrate_with_reporting=True,
                    include_detailed_analysis=True
                )
                self.feature_importance_manager = FeatureImportanceIntegrationManager(importance_config)
                self.logger.info("✅ Feature importance manager initialized in sub-pipeline")
            except Exception as e:
                self.logger.warning(f"⚠️ Feature importance manager initialization failed: {e}")
                self.feature_importance_manager = None

        # Initialize pipeline state for component communication
        self._current_data = None
        self._current_pipeline_state = {}
        self._accumulated_artifacts = {}

    def _apply_logging_config(self, logging_cfg: LoggingConfig) -> None:
        try:
            import logging as _logging
            from pathlib import Path as _Path
            level = getattr(_logging, str(logging_cfg.level).upper(), _logging.INFO)
            self.logger.setLevel(level)
            if logging_cfg.enable_file and logging_cfg.log_file:
                has_same_file = any(
                    isinstance(h, _logging.FileHandler) and getattr(h, 'baseFilename', None) == str(_Path(logging_cfg.log_file).resolve())
                    for h in self.logger.handlers
                )
                if not has_same_file:
                    _Path(logging_cfg.log_file).parent.mkdir(parents=True, exist_ok=True)
                    fh = _logging.FileHandler(logging_cfg.log_file)
                    fh.setLevel(level)
                    formatter = _logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                    fh.setFormatter(formatter)
                    self.logger.addHandler(fh)
        except Exception as e:
            log_warning(f"Failed to apply logging configuration: {e}. Continuing with default logging settings.")
    
    def _validate_sub_pipeline_result(self, result: SubPipelineResult, stage_name: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Validate sub-pipeline result and return success status and error info.
        
        Returns:
            Tuple of (is_success, error_dict_or_none)
        """
        if result.is_complete:
            log_success(f"{stage_name} completed with complete report")
            return True, None
        elif result.success:
            log_warning(f"{stage_name} completed but report is incomplete")
            return False, {
                'success': False,
                'error': f"{stage_name} produced incomplete report - missing required artifacts",
                'stage': result.sub_pipeline_name,
                'incomplete_artifacts': result.artifacts
            }
        else:
            log_error(f"{stage_name} failed: {result.error_message}")
            return False, {
                'success': False,
                'error': f"{stage_name} failed: {result.error_message}",
                'stage': result.sub_pipeline_name
            }
    
    def _convert_to_component_config(self, sub_config: SubPipelineConfig) -> ComponentConfig:
        """Convert SubPipelineConfig to ComponentConfig."""
        return ComponentConfig(
            symbol=sub_config.symbol,
            exchange=sub_config.exchange,
            timeframe=sub_config.timeframe,
            data_dir=sub_config.data_dir,
            start_date=sub_config.start_date,
            end_date=sub_config.end_date,
            force_rerun=sub_config.force_rerun,
            validation_enabled=sub_config.validation_enabled,
            monitoring_enabled=sub_config.monitoring_enabled,
            fast_mode=sub_config.fast_mode,
            custom_params=sub_config.custom_params
        )

    def _resolve_data_locator(self, config: SubPipelineConfig) -> DataLocator:
        if isinstance(config.data_locator, DataLocator):
            locator = config.data_locator
        else:
            locator = DataLocator(config.data_locator_config)
            config.data_locator = locator
        config.attach_locator(locator)
        self._data_locator = locator
        return locator

    def _ensure_data_directory(self, config: SubPipelineConfig, locator: DataLocator) -> None:
        data_value = config.data_dir
        default_key = config.data_dir_key or "market_data"

        if data_value:
            candidate = Path(data_value).expanduser()
            if candidate.is_absolute():
                resolved = candidate
            elif data_value == DEFAULT_DATA_DIR:
                resolved = locator.data_path(default_key, ensure_exists=True)
            else:
                resolved = locator.data_path(default=data_value, ensure_exists=True)
        else:
            resolved = locator.data_path(default_key, ensure_exists=True)

        resolved.mkdir(parents=True, exist_ok=True)
        config.data_dir = str(resolved)

    def _emit_effective_configuration(self, config: SubPipelineConfig) -> None:
        summary = config.paths.summary()
        summary_json = json.dumps(summary, indent=2, sort_keys=True)
        self.logger.info('📁 Effective filesystem configuration:\n%s', summary_json)
        tprint(f"📁 Effective filesystem configuration:\n{summary_json}")
        self._configuration_logged = True

    def _prepare_filesystem(self, config: SubPipelineConfig) -> DataLocator:
        locator = self._resolve_data_locator(config)
        self._ensure_data_directory(config, locator)
        if not self._configuration_logged:
            self._emit_effective_configuration(config)
        return locator
    
    def _convert_old_config(self, config: Dict[str, Any]) -> SubPipelineConfig:
        """Convert old config format to SubPipelineConfig."""
        # Extract relevant configuration
        sr_config = config.get('sr_optimization', {})
        training_mode = config.get('training_mode', 'full')
        
        # Determine execution mode
        if training_mode == 'light':
            mode = ExecutionMode.LIGHT
        elif training_mode == 'blank':
            mode = ExecutionMode.BLANK
        else:
            mode = ExecutionMode.FULL
        
        # Create SubPipelineConfig
        sub_config = SubPipelineConfig(
            mode=mode,
            symbol=config.get('symbol', 'ETHUSDT'),
            exchange=config.get('exchange', 'binance'),
            timeframe=config.get('timeframe', '15m'),
            data_dir=config.get('data_dir', './data'),
            start_date=config.get('start_date'),
            end_date=config.get('end_date'),
            force_rerun=config.get('force_rerun', False),
            parallel_processing=config.get('parallel_processing', True),
            max_workers=config.get('max_workers', 4),
            validation_enabled=config.get('validation_enabled', True),
            monitoring_enabled=config.get('monitoring_enabled', True),
            fast_mode=config.get('fast_mode', False),
            skip_next_pipeline=config.get('skip_next_pipeline', False),
            custom_params=config.get('custom_params', {})
        )
        
        return sub_config
    
    async def _execute_unified_nas_tas_regime_discovery(self) -> SubPipelineResult:
        """Execute NAS-TAS regime discovery using unified pipeline."""
        try:
            self.logger.info("🚀 Using unified NAS-TAS pipeline for regime discovery")
            tprint("🚀 [UNIFIED_PIPELINE] Using unified NAS-TAS pipeline for regime discovery", color="cyan", bold=True)
            
            # Create unified pipeline based on mode
            if self.config.unified_pipeline_mode == "nas":
                pipeline = create_nas_pipeline()
            elif self.config.unified_pipeline_mode == "tas":
                pipeline = create_tas_pipeline()
            else:  # hybrid
                pipeline = create_hybrid_pipeline()
            
            # Execute unified pipeline
            result = await pipeline.execute_regime_discovery(
                symbol=self.config.symbol,
                timeframe=self.config.timeframe,
                data_dir=self.config.data_dir
            )
            
            # Convert unified pipeline result to SubPipelineResult format
            return SubPipelineResult(
                sub_pipeline_name='nas_tas_regime_discovery',
                status=SubPipelineStatus.COMPLETED,
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration_seconds=0.0,
                artifacts={'nas_tas_regime_discovery_result': result},
                metadata={'pipeline_type': 'unified', 'mode': self.config.unified_pipeline_mode}
            )
            
        except Exception as e:
            self.logger.error(f"❌ Unified NAS-TAS regime discovery failed: {e}")
            if self.config.unified_pipeline_fallback:
                self.logger.info("🔄 Falling back to legacy regime discovery")
                return await self.execute_sub_pipeline('nas_tas_regime_discovery', self.config)
            else:
                raise
    
    async def _execute_unified_nas_tas_clustering(self) -> SubPipelineResult:
        """Execute NAS-TAS clustering using unified pipeline."""
        try:
            self.logger.info("🚀 Using unified NAS-TAS pipeline for clustering")
            tprint("🚀 [UNIFIED_PIPELINE] Using unified NAS-TAS pipeline for clustering", color="cyan", bold=True)
            
            # Create unified pipeline based on mode
            if self.config.unified_pipeline_mode == "nas":
                pipeline = create_nas_pipeline()
            elif self.config.unified_pipeline_mode == "tas":
                pipeline = create_tas_pipeline()
            else:  # hybrid
                pipeline = create_hybrid_pipeline()
            
            # Execute unified pipeline
            result = await pipeline.execute_clustering(
                symbol=self.config.symbol,
                timeframe=self.config.timeframe,
                data_dir=self.config.data_dir
            )
            
            # Convert unified pipeline result to SubPipelineResult format
            return SubPipelineResult(
                sub_pipeline_name='nas_tas_clustering',
                status=SubPipelineStatus.COMPLETED,
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration_seconds=0.0,
                artifacts={'optimal_regime_clustering_result': result},
                metadata={'pipeline_type': 'unified', 'mode': self.config.unified_pipeline_mode}
            )
            
        except Exception as e:
            self.logger.error(f"❌ Unified NAS-TAS clustering failed: {e}")
            if self.config.unified_pipeline_fallback:
                self.logger.info("🔄 Falling back to legacy clustering")
                return await self.execute_sub_pipeline('nas_tas_clustering', self.config)
            else:
                raise
    
    async def _execute_unified_nas_tas_models_training(self) -> SubPipelineResult:
        """Execute NAS-TAS models training using unified pipeline."""
        try:
            self.logger.info("🚀 Using unified NAS-TAS pipeline for models training")
            tprint("🚀 [UNIFIED_PIPELINE] Using unified NAS-TAS pipeline for models training", color="cyan", bold=True)
            
            # Create unified pipeline based on mode
            if self.config.unified_pipeline_mode == "nas":
                pipeline = create_nas_pipeline()
            elif self.config.unified_pipeline_mode == "tas":
                pipeline = create_tas_pipeline()
            else:  # hybrid
                pipeline = create_hybrid_pipeline()
            
            # Execute unified pipeline
            result = await pipeline.execute_models_training(
                symbol=self.config.symbol,
                timeframe=self.config.timeframe,
                data_dir=self.config.data_dir
            )
            
            # Convert unified pipeline result to SubPipelineResult format
            return SubPipelineResult(
                sub_pipeline_name='nas_tas_models_training',
                status=SubPipelineStatus.COMPLETED,
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration_seconds=0.0,
                artifacts={'nas_tas_models_training_result': result},
                metadata={'pipeline_type': 'unified', 'mode': self.config.unified_pipeline_mode}
            )
            
        except Exception as e:
            self.logger.error(f"❌ Unified NAS-TAS models training failed: {e}")
            if self.config.unified_pipeline_fallback:
                self.logger.info("🔄 Falling back to legacy models training")
                return await self.execute_sub_pipeline('nas_tas_models_training', self.config)
            else:
                raise
    
    async def _execute_unified_nas_tas_ensemble_training(self) -> SubPipelineResult:
        """Execute NAS-TAS ensemble training using unified pipeline."""
        try:
            self.logger.info("🚀 Using unified NAS-TAS pipeline for ensemble training")
            tprint("🚀 [UNIFIED_PIPELINE] Using unified NAS-TAS pipeline for ensemble training", color="cyan", bold=True)
            
            # Create unified pipeline based on mode
            if self.config.unified_pipeline_mode == "nas":
                pipeline = create_nas_pipeline()
            elif self.config.unified_pipeline_mode == "tas":
                pipeline = create_tas_pipeline()
            else:  # hybrid
                pipeline = create_hybrid_pipeline()
            
            # Execute unified pipeline
            result = await pipeline.execute_ensemble_training(
                symbol=self.config.symbol,
                timeframe=self.config.timeframe,
                data_dir=self.config.data_dir
            )
            
            # Convert unified pipeline result to SubPipelineResult format
            return SubPipelineResult(
                sub_pipeline_name='nas_tas_ensemble_training',
                status=SubPipelineStatus.COMPLETED,
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration_seconds=0.0,
                artifacts={'nas_tas_ensemble_training_result': result},
                metadata={'pipeline_type': 'unified', 'mode': self.config.unified_pipeline_mode}
            )
            
        except Exception as e:
            self.logger.error(f"❌ Unified NAS-TAS ensemble training failed: {e}")
            if self.config.unified_pipeline_fallback:
                self.logger.info("🔄 Falling back to legacy ensemble training")
                return await self.execute_sub_pipeline('nas_tas_ensemble_training', self.config)
            else:
                raise
    
    async def _execute_nas_tas_clustering_with_new_structure(self) -> SubPipelineResult:
        """Execute NAS-TAS clustering using the new clustering directory structure."""
        start_time = datetime.now()
        
        try:
            self.logger.info("🚀 Using new clustering directory structure for NAS-TAS clustering")
            tprint("🚀 [CLUSTERING] Using new clustering directory structure for NAS-TAS clustering", color="cyan", bold=True)
            
            # Import the new clustering component
            from src.training.steps.market_analysis.clustering import NASTASClusteringComponent
            from src.training.steps.market_analysis.clustering.config.clustering_config import NASTASClusteringConfig
            
            # Create configuration for the clustering component
            clustering_config = NASTASClusteringConfig()
            
            # Initialize the clustering component
            clustering_component = NASTASClusteringComponent(config=clustering_config)
            
            # Prepare data for clustering
            # Use the current data and pipeline state
            data = self._current_data
            pipeline_state = self._current_pipeline_state.copy()
            
            # Add regime discovery results to pipeline state if available
            if 'regime_models' in self._current_pipeline_state:
                pipeline_state['regime_models'] = self._current_pipeline_state['regime_models']
            if 'regime_assignments' in self._current_pipeline_state:
                pipeline_state['regime_assignments'] = self._current_pipeline_state['regime_assignments']
            
            self.logger.info(f"📊 Data prepared for clustering: {data.shape if data is not None else 'None'}")
            self.logger.info(f"📊 Pipeline state keys: {list(pipeline_state.keys())}")
            
            # Execute clustering using the component's fit method
            if data is not None:
                clustering_result = await clustering_component.fit(data, None, pipeline_state)
                
                # Extract the actual clustering data from the component
                clustering_data = {}
                if hasattr(clustering_result, 'current_results') and clustering_result.current_results:
                    clustering_data = clustering_result.current_results
                else:
                    # Fallback: try to get data from the component's context
                    if hasattr(clustering_result, 'context') and clustering_result.context:
                        context = clustering_result.context
                        if hasattr(context, 'optimized_assignments') and context.optimized_assignments is not None:
                            # Convert numpy array to list for proper JSON serialization
                            assignments = context.optimized_assignments
                            if hasattr(assignments, 'tolist'):  # numpy array
                                clustering_data['cluster_assignments'] = assignments.tolist()
                            else:  # already a list or other type
                                clustering_data['cluster_assignments'] = assignments
                        if hasattr(context, 'optimal_k') and context.optimal_k is not None:
                            clustering_data['n_clusters'] = context.optimal_k
                        if hasattr(context, 'final_k') and context.final_k is not None:
                            clustering_data['final_k'] = context.final_k
                            clustering_data['n_clusters'] = context.final_k  # Update n_clusters with final value
                        if hasattr(context, 'optimized_results') and context.optimized_results:
                            clustering_data.update(context.optimized_results)
                
                # Create artifacts in the expected format with actual clustering data
                artifacts = {
                    'optimal_regime_clustering_result': {
                        'clustering_result': clustering_data,  # Store actual data instead of component object
                        'component_config': clustering_config.__dict__,
                        'execution_metadata': {
                            'component_type': 'NASTASClusteringComponent',
                            'execution_mode': 'new_structure',
                            'timestamp': datetime.now().isoformat()
                        }
                    }
                }
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                result = SubPipelineResult(
                    sub_pipeline_name='nas_tas_clustering',
                    status=SubPipelineStatus.COMPLETED,
                    start_time=start_time,
                    end_time=end_time,
                    duration_seconds=duration,
                    artifacts=artifacts,
                    metadata={'component_type': 'NASTASClusteringComponent', 'execution_mode': 'new_structure'}
                )
                
                self.logger.info("✅ NAS-TAS clustering completed successfully with new structure")
                tprint("✅ [CLUSTERING] NAS-TAS clustering completed successfully with new structure", color="green", bold=True)
                
                return result
            else:
                raise ValueError("No data available for clustering")
                
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.logger.error(f"❌ NAS-TAS clustering with new structure failed: {e}")
            tprint(f"❌ [CLUSTERING] NAS-TAS clustering with new structure failed: {e}", color="red", bold=True)
            
            # Fallback to legacy clustering if available
            self.logger.info("🔄 Falling back to legacy clustering")
            try:
                return await self.execute_sub_pipeline('nas_tas_clustering', self.config)
            except Exception as fallback_error:
                self.logger.error(f"❌ Legacy clustering fallback also failed: {fallback_error}")
                
                return SubPipelineResult(
                    sub_pipeline_name='nas_tas_clustering',
                    status=SubPipelineStatus.FAILED,
                    start_time=start_time,
                    end_time=end_time,
                    duration_seconds=duration,
                    error_message=f"New structure failed: {e}, Legacy fallback failed: {fallback_error}"
                )

    async def execute_all_steps_from_start(
        self, 
        config: Optional[SubPipelineConfig] = None
    ) -> Dict[str, Any]:
        """
        Execute all 13 market analysis steps automatically from the beginning.
        
        This is a convenience method that starts from step 1 (sr_parameter_optimization)
        and automatically triggers all subsequent steps when each completes.
        
        Args:
            config: Configuration for the sub-pipeline (optional)
            
        Returns:
            Dict with execution results and summary
        """
        if config is None:
            config = self.config

        # Reset results for a fresh run
        self.results = []

        self._prepare_filesystem(config)

        log_info('🚀 Starting automatic execution of all 13 market analysis steps')
        log_info('=' * 80)
        log_info('📋 Steps to be executed automatically:')
        log_info('   1. sr_parameter_optimization - Optimize SR detection levels')
        log_info('   2. sr_detection - Detect Support/Resistance levels')
        log_info('   3. sr_clustering - Generate SR clusters')
        log_info('   4. hybrid_nas_tas_regime_discovery - Discover market regimes using hybrid NAS-TAS approach')
        log_info('   5. nas_tas_clustering - NAS-TAS-based regime clustering')
        log_info('   6. regime_models_training - Regime detection models training (CatBoost, Bayesian Rule Lists, ExtraTrees)')
        log_info('   7. regime_ensemble_training - Meta-learner training (stacker_lgbm_calibrated)')
        log_info('   8. regime_data_splitting - Tag data by regimes')
        log_info('=' * 80)
        
        # Execute from the first step - this will automatically trigger all subsequent steps
        result = await self.execute_sub_pipeline_with_next('sr_parameter_optimization', config)
        
        # Get execution summary
        summary = self.get_execution_summary()
        
        return {
            'success': result.success,
            'first_step_result': result,
            'execution_summary': summary,
            'total_steps_executed': summary['total_sub_pipelines'],
            'successful_steps': summary['successful_sub_pipelines'],
            'failed_steps': summary['failed_sub_pipelines'],
            'total_execution_time': summary['total_execution_time']
        }
    
    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the complete market analysis sub-pipeline with backward compatible interface.

        This method orchestrates the complete market analysis pipeline with logical groupings:

        SR Steps (1-3):
        1. SR parameter optimization
        2. SR detection
        3. SR clustering

        Regime Steps (4-7):
        4. NAS-TAS regime discovery
        5. NAS-TAS clustering
        6. Regime detection models training (CatBoost, Bayesian Rule Lists, ExtraTrees)
        7. Regime detection ensemble training (stacker_lgbm_calibrated)

        Data Processing Steps (8-12):
        8. Regime data splitting
        9. Multi-horizon profit labeling
        10. Feature lookback optimization
        11. Interactive feature generation
        12. Final feature selection (120→100→80→60)
        """
        log_info('🎯 Starting Market Analysis Sub-Pipeline execution')
        tprint("🎯 [MARKET_ANALYSIS] Starting Market Analysis Sub-Pipeline execution", color="cyan", bold=True)
        # Reset results for a fresh run
        self.results = []

        self._prepare_filesystem(self.config)

        try:
            # Extract data from pipeline state
            data = pipeline_state.get('dataframe')
            if data is None:
                tprint("❌ [MARKET_ANALYSIS] No dataframe found in pipeline state", color="red", bold=True)
                raise ValueError("No dataframe found in pipeline state")
            tprint(f"📊 [MARKET_ANALYSIS] Data loaded: {data.shape[0]} rows, {data.shape[1]} columns", color="green")
            
            # Store data and pipeline state for component communication
            self._current_data = data
            self._current_pipeline_state = pipeline_state.copy()
            
            # Initialize results dictionary
            results = {}
            
            # ===== SR STEPS GROUP =====
            log_info('🎯 ===== STARTING SR STEPS GROUP =====')
            tprint("🎯 [MARKET_ANALYSIS] ===== STARTING SR STEPS GROUP =====", color="blue", bold=True)
            
            # Stage 1: SR Parameter Optimization (BEFORE detection and clustering)
            log_info('🎯 Executing Stage 1: SR Parameter Optimization')
            tprint("🎯 [MARKET_ANALYSIS] Executing Stage 1: SR Parameter Optimization", color="yellow")
            param_optimization_result = await self.execute_sub_pipeline('sr_parameter_optimization', self.config)
            is_success, error_info = self._validate_sub_pipeline_result(param_optimization_result, "SR Parameter Optimization")
            if not is_success:
                return error_info
            
            # Extract data from consolidated artifact
            sr_optimization_result = param_optimization_result.artifacts.get('sr_parameter_optimization_result', {})
            results['optimized_parameters'] = sr_optimization_result.get('optimized_parameters', {})
            results['quality_thresholds'] = sr_optimization_result.get('quality_thresholds', {})
            results['parameter_optimization_metrics'] = sr_optimization_result.get('parameter_optimization_metrics', {})
            
            # Update pipeline state for next components
            self._current_pipeline_state.update({
                'optimized_parameters': results['optimized_parameters'],
                'quality_thresholds': results['quality_thresholds']
            })
            
            # Stage 2: SR Detection (using optimized parameters)
            self.logger.info('📊 Executing Stage 2: SR Detection')
            sr_detection_result = await self.execute_sub_pipeline('sr_detection', self.config)
            is_success, error_info = self._validate_sub_pipeline_result(sr_detection_result, "SR Detection")
            if not is_success:
                return error_info
            
            # Extract data from consolidated artifact
            sr_detection_data = sr_detection_result.artifacts.get('sr_detection_result', {})
            results['sr_levels'] = sr_detection_data.get('sr_levels', {})
            results['detection_metrics'] = sr_detection_data.get('detection_metrics', {})
            
            # Update pipeline state for next components
            self._current_pipeline_state.update({
                'sr_levels': results['sr_levels']
            })
            
            # Stage 3: SR Clustering (using detected levels)
            self.logger.info('🔗 Executing Stage 3: SR Clustering')
            sr_clustering_result = await self.execute_sub_pipeline('sr_clustering', self.config)
            is_success, error_info = self._validate_sub_pipeline_result(sr_clustering_result, "SR Clustering")
            if not is_success:
                return error_info
            
            # Extract data from consolidated artifact
            sr_clustering_data = sr_clustering_result.artifacts.get('sr_clustering_result', {})
            results['sr_clusters'] = sr_clustering_data.get('sr_clusters', {})
            results['clustering_metrics'] = sr_clustering_data.get('clustering_metrics', {})
            
            # Update pipeline state for next components
            self._current_pipeline_state.update({
                'sr_clusters': results['sr_clusters']
            })
            
            # ===== HMM STEPS GROUP =====
            self.logger.info('🔍 ===== STARTING HMM STEPS GROUP =====')
            tprint("🔍 [MARKET_ANALYSIS] ===== STARTING HMM STEPS GROUP =====", color="blue", bold=True)
            
            # Stage 4: NAS-TAS Regime Discovery
            self.logger.info('🔍 Executing Stage 4: NAS-TAS Regime Discovery')
            tprint("🔍 [MARKET_ANALYSIS] Executing Stage 4: NAS-TAS Regime Discovery", color="yellow")
            
            # Use unified pipeline if available and enabled
            if (UNIFIED_PIPELINE_AVAILABLE and 
                self.config.use_unified_pipeline and 
                self.config.unified_pipeline_mode in ["nas", "hybrid"]):
                nas_tas_regime_discovery_result = await self._execute_unified_nas_tas_regime_discovery()
            else:
                nas_tas_regime_discovery_result = await self.execute_sub_pipeline('nas_tas_regime_discovery', self.config)
            
            is_success, error_info = self._validate_sub_pipeline_result(nas_tas_regime_discovery_result, "NAS-TAS Regime Discovery")
            if not is_success:
                return error_info

            # Extract data from consolidated artifact
            nas_regime_data = nas_tas_regime_discovery_result.artifacts.get('nas_tas_regime_discovery_result', {})
            results['regime_models'] = nas_regime_data.get('regime_models', {})
            results['regime_assignments'] = nas_regime_data.get('regime_assignments', {})
            results['regime_metrics'] = nas_regime_data.get('regime_metrics', {})
            
            # Update pipeline state for next components
            self._current_pipeline_state.update({
                'regime_models': results['regime_models'],
                'regime_assignments': results['regime_assignments'],
                'nas_tas_regime_discovery_result': nas_regime_data  # Pass the full regime discovery result
            })
            
            # Stage 5: NAS-TAS Clustering
            self.logger.info('🎯 Executing Stage 5: NAS-TAS Clustering')
            tprint("🎯 [MARKET_ANALYSIS] Executing Stage 5: NAS-TAS Clustering", color="yellow")
            
            # Use the new clustering directory structure
            nas_tas_clustering_result = await self._execute_nas_tas_clustering_with_new_structure()
            
            is_success, error_info = self._validate_sub_pipeline_result(nas_tas_clustering_result, "NAS-TAS Clustering")
            if not is_success:
                return error_info
            
            # Extract data from consolidated artifact
            nas_tas_clustering_data = nas_tas_clustering_result.artifacts.get('nas_tas_clustering_result', {})
            results['nas_tas_clusters'] = nas_tas_clustering_data.get('nas_tas_clusters', {})
            results['nas_tas_clustering_metrics'] = nas_tas_clustering_data.get('nas_tas_clustering_metrics', {})
            
            # Update pipeline state for next components
            cluster_assignments = nas_tas_clustering_data.get('cluster_assignments', [])
            self._current_pipeline_state.update({
                'nas_tas_clusters': nas_tas_clustering_data,  # Store the full result
                'cluster_assignments': cluster_assignments,  # Make cluster_assignments directly accessible
                'optimal_regime_clustering_result': nas_tas_clustering_data  # Add the key that regime_data_splitting expects
            })
            
            # Prepare data for HMM Models Training
            self.logger.info('📊 Preparing data for HMM Models Training...')
            try:
                # FORCE ORIGINAL MARKET DATA for feature bank integration
                self.logger.info('🔧 FORCING original market data for regime models training feature bank integration')
                tprint("🔧 [SUB_PIPELINE] FORCING original market data for regime models training", color="cyan", bold=True)
                
                # Don't use processed features - let the component generate comprehensive features from original data
                features = None  # Force None to trigger feature bank generation
                feature_names = []
                
                # Targets are not required: HMM training uses cluster_assignments as labels
                targets = None
                
                # Extract regime labels from regime assignments
                regime_labels = None
                if 'regime_assignments' in results and results['regime_assignments']:
                    regime_data = results['regime_assignments']
                    if isinstance(regime_data, list):
                        # regime_assignments is a list of regime assignments
                        regime_labels = regime_data
                    elif isinstance(regime_data, dict) and 'regime_labels' in regime_data:
                        # Legacy format - regime_assignments is a dict with regime_labels
                        regime_labels = regime_data['regime_labels']
                
                # Store original market data for feature bank generation
                self._current_pipeline_state.update({
                    'features': features,  # None to force feature bank generation
                    'targets': targets,
                    'regime_labels': regime_labels,
                    'feature_names': feature_names,
                    'original_data': data,  # Store original market data for feature bank
                    'force_feature_bank': True  # Flag to force feature bank usage
                })
                
                # Log data availability for debugging
                self.logger.info(f"📊 Data prepared for HMM Models Training:")
                self.logger.info(f"   - Features: {'✅' if features is not None else '❌'}")
                self.logger.info(f"   - Targets: {'✅' if targets is not None else '❌'} (HMM uses cluster_assignments)")
                self.logger.info(f"   - Regime Labels: {'✅' if regime_labels is not None else '❌'}")
                self.logger.info(f"   - Feature Names: {len(feature_names) if feature_names else 0}")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to prepare data for HMM Models Training: {e}")
                return self._create_error_result("Data preparation failed for HMM Models Training", str(e))
            
            # Stage 6: Regime Detection Models Training
            self.logger.info('🏋️ Executing Stage 6: Regime Detection Models Training')
            
            # Use the new regime detection models training component
            regime_models_training_result = await self.execute_sub_pipeline('regime_models_training', self.config)
            
            is_success, error_info = self._validate_sub_pipeline_result(regime_models_training_result, "Regime Detection Models Training")
            if not is_success:
                return error_info
            
            # Extract data from consolidated artifact
            regime_models_data = regime_models_training_result.artifacts.get('regime_models_training_result', {})
            results['regime_models'] = regime_models_data.get('regime_models', {})
            results['regime_training_metrics'] = regime_models_data.get('metrics', {})
            
            # Update pipeline state for next components
            self._current_pipeline_state.update({
                'regime_models': results['regime_models']
            })
            
            # Ensure features and targets are available for regime ensemble training
            # Extract features from optimized_features or interactive_features if available
            features = None
            feature_names = []
            targets = None
            
            if 'optimized_features' in results and results['optimized_features']:
                features_data = results['optimized_features']
                if isinstance(features_data, dict) and 'features' in features_data:
                    features = features_data['features']
                    feature_names = features_data.get('feature_names', [])
            
            if features is None and 'interactive_features' in results:
                interactive_features = results['interactive_features']
                if isinstance(interactive_features, dict) and 'combined_features' in interactive_features:
                    features = interactive_features['combined_features']
                    feature_names = interactive_features.get('combined_feature_names', [])
            
            # If no features available yet, use basic features from regime models training
            if features is None:
                # Extract basic features from the regime models training data
                # The regime models training component generates features internally
                self.logger.warning("⚠️ No optimized features available for regime ensemble training, using basic features from regime models training")
                
                # Get the features that were used in regime models training
                # These are generated internally by the regime models training component
                # We need to regenerate them or extract them from the training process
                try:
                    # Extract regime labels for feature generation
                    regime_labels = None
                    if 'regime_assignments' in results and results['regime_assignments']:
                        regime_data = results['regime_assignments']
                        if isinstance(regime_data, list):
                            regime_labels = regime_data
                        elif isinstance(regime_data, dict) and 'regime_labels' in regime_data:
                            regime_labels = regime_data['regime_labels']
                    
                    # Generate basic features similar to what regime models training does
                    if regime_labels is not None:
                        features, feature_names = self._generate_basic_features(data, regime_labels)
                        targets = regime_labels  # Use regime labels as targets
                        self.logger.info(f"✅ Generated basic features: {features.shape if features is not None else 'None'}")
                    else:
                        self.logger.warning("⚠️ No regime labels available for feature generation")
                        features = None
                        targets = None
                except Exception as e:
                    self.logger.error(f"❌ Failed to generate basic features: {e}")
                    features = None
                    targets = None
            
            # Update pipeline state with features and targets for regime ensemble training
            self._current_pipeline_state.update({
                'features': features,
                'targets': targets,
                'feature_names': feature_names
            })
            
            # Log data availability for debugging
            self.logger.info(f"📊 Data prepared for Regime Ensemble Training:")
            self.logger.info(f"   - Features: {'✅' if features is not None else '❌'}")
            self.logger.info(f"   - Targets: {'✅' if targets is not None else '❌'}")
            self.logger.info(f"   - Feature Names: {len(feature_names) if feature_names else 0}")
            
            # Stage 7: Regime Detection Ensemble Training
            self.logger.info('🎭 Executing Stage 7: Regime Detection Ensemble Training')
            
            # Use the new regime detection ensemble training component
            regime_ensemble_training_result = await self.execute_sub_pipeline('regime_ensemble_training', self.config)
            
            is_success, error_info = self._validate_sub_pipeline_result(regime_ensemble_training_result, "Regime Detection Ensemble Training")
            if not is_success:
                return error_info
            
            # Extract data from consolidated artifact
            regime_ensemble_data = regime_ensemble_training_result.artifacts.get('regime_ensemble_training_result', {})
            results['regime_ensemble'] = regime_ensemble_data.get('stacker_lgbm_calibrated', {})
            results['regime_ensemble_metrics'] = regime_ensemble_data.get('ensemble_metrics', {})
            
            # Update pipeline state for next components
            self._current_pipeline_state.update({
                'regime_ensemble': results['regime_ensemble']
            })
            
            # ===== DATA PROCESSING STEPS GROUP =====
            self.logger.info('✂️ ===== STARTING DATA PROCESSING STEPS GROUP =====')
            
            # Stage 8: Regime Data Splitting
            self.logger.info('✂️ Executing Stage 8: Regime Data Splitting')
            regime_data_splitting_result = await self.execute_sub_pipeline('regime_data_splitting', self.config)
            is_success, error_info = self._validate_sub_pipeline_result(regime_data_splitting_result, "Regime Data Splitting")
            if not is_success:
                return error_info
            
            # Extract data from consolidated artifact
            regime_splitting_data = regime_data_splitting_result.artifacts.get('regime_data_splitting_result', {})
            results['regime_data'] = regime_splitting_data.get('regime_data', {})
            results['regime_stats'] = regime_splitting_data.get('regime_stats', {})
            
            # Update pipeline state for next components
            self._current_pipeline_state.update({
                'regime_data': results['regime_data']
            })
            

            # Final success
            self.logger.info('🎉 Market Analysis Sub-Pipeline completed successfully')
            return {
                'success': True,
                'results': results,
                'execution_time': sum(result.execution_time for result in self.results),
                'total_stages': 8,
                'completed_stages': len(self.results)
            }
            
        except Exception as e:
            self.logger.error(f'❌ Market Analysis Sub-Pipeline failed: {e}')
            import traceback
            self.logger.error(f'❌ Error details: {traceback.format_exc()}')
            # Add feature importance analysis if available
            if self.feature_importance_manager:
                try:
                    log_info('🔍 Adding feature importance analysis to final results')
                    # Extract features and regime assignments from pipeline results
                    features = None
                    feature_names = []
                    regime_assignments = None

                    # Look for features in accumulated artifacts with improved extraction
                    if 'final_features' in self._accumulated_artifacts:
                        final_features = self._accumulated_artifacts['final_features']
                        if isinstance(final_features, dict):
                            features = final_features.get('features')
                            feature_names = final_features.get('feature_names', [])

                    # Try alternative sources if final_features not available
                    if features is None:
                        # Look for features from previous pipeline stages
                        for stage_result in self.results:
                            if hasattr(stage_result, 'artifacts') and stage_result.artifacts:
                                artifacts = stage_result.artifacts
                                if 'features' in artifacts and 'feature_names' in artifacts:
                                    features = artifacts['features']
                                    feature_names = artifacts['feature_names']
                                    break

                    # Look for regime assignments
                    if 'regime_assignments' in self._accumulated_artifacts:
                        regime_assignments = self._accumulated_artifacts['regime_assignments']

                    # Try to find regime assignments in pipeline results if not in artifacts
                    if regime_assignments is None:
                        for stage_result in self.results:
                            if hasattr(stage_result, 'artifacts') and stage_result.artifacts:
                                artifacts = stage_result.artifacts
                                if 'regime_assignments' in artifacts or 'regime_predictions' in artifacts:
                                    regime_assignments = artifacts.get('regime_assignments') or artifacts.get('regime_predictions')
                                    break

                    # Validate that we have the required data
                    if features is not None and len(feature_names) > 0 and regime_assignments is not None:
                        # Ensure feature_names matches the number of features
                        if len(feature_names) != features.shape[1]:
                            log_warning(f"⚠️ Feature names count ({len(feature_names)}) doesn't match features shape ({features.shape[1]})")
                            # Generate fallback feature names
                            feature_names = [f'feature_{i}' for i in range(features.shape[1])]

                        # Ensure regime assignments match features length
                        if len(regime_assignments) != features.shape[0]:
                            log_warning(f"⚠️ Regime assignments length ({len(regime_assignments)}) doesn't match features length ({features.shape[0]})")
                        else:
                            # Perform feature importance analysis
                            importance_analysis = self.feature_importance_manager.analyze_post_clustering_regimes(
                                features, feature_names, np.array(regime_assignments)
                            )

                            if importance_analysis:
                                results['feature_importance_analysis'] = importance_analysis
                                log_info('✅ Feature importance analysis added to final results')
                                log_info(f"📊 Analyzed {len(feature_names)} features across {len(np.unique(regime_assignments))} regimes")
                            else:
                                log_warning('⚠️ Feature importance analysis returned no results')
                    else:
                        missing_items = []
                        if features is None:
                            missing_items.append("features")
                        if len(feature_names) == 0:
                            missing_items.append("feature_names")
                        if regime_assignments is None:
                            missing_items.append("regime_assignments")
                        log_warning(f"⚠️ Cannot perform feature importance analysis - missing: {', '.join(missing_items)}")

                except Exception as e:
                    log_warning(f'⚠️ Feature importance analysis addition failed: {e}')
                    self.logger.error(f"Feature importance analysis error: {e}")

            return {
                'success': False,
                'error': str(e),
                'execution_time': sum(result.execution_time for result in self.results),
                'completed_stages': len(self.results)
            }
    
    def validate_config(self):
        """Validate the sub-pipeline configuration."""
        if not self.config.symbol:
            raise ValueError("Symbol is required")
        if not self.config.exchange:
            raise ValueError("Exchange is required")
        if not self.config.timeframe:
            raise ValueError("Timeframe is required")
    
    def get_status(self):
        """Get the current status of the sub-pipeline."""
        return {
            'config': self.config,
            'results_count': len(self.results),
            'completed_stages': [r.sub_pipeline_name for r in self.results if r.success]
        }
    
    def _log_sub_pipeline_completion(self, sub_pipeline_name: str, config: SubPipelineConfig, artifacts: Dict[str, Any]):
        """Log sub-pipeline completion with artifacts summary."""
        artifact_count = len(artifacts)
        artifact_keys = list(artifacts.keys())
        
        self.logger.info(f"✅ {sub_pipeline_name} completed successfully")
        # Ensure logs reflect single artifact expectation for nas_tas_clustering
        if sub_pipeline_name == 'nas_tas_clustering' and artifact_count > 1:
            self.logger.info(f"📊 Generated {artifact_count} artifacts: {artifact_keys} (note: consolidated into single artifact group)")
        else:
            self.logger.info(f"📊 Generated {artifact_count} artifacts: {artifact_keys}")
        
        # Log artifact sizes for monitoring
        for key, value in artifacts.items():
            if isinstance(value, (list, dict)):
                size = len(value)
                self.logger.info(f"  📁 {key}: {size} items")
            elif isinstance(value, str):
                size = len(value)
                self.logger.info(f"  📄 {key}: {size} characters")
            else:
                self.logger.info(f"  📦 {key}: {type(value).__name__}")
    
    async def execute_sub_pipeline(
        self, 
        sub_pipeline_name: str, 
        config: SubPipelineConfig
    ) -> SubPipelineResult:
        """
        Execute a specific sub-pipeline using the component system.
        
        Args:
            sub_pipeline_name: Name of the sub-pipeline to execute
            config: Configuration for the sub-pipeline
            
        Returns:
            SubPipelineResult with execution details
        """
        self._prepare_filesystem(config)
        start_time = datetime.now()
        self.logger.info(f'🚀 Starting {sub_pipeline_name} sub-pipeline')
        
        try:
            # Load market data if not already available
            if self._current_data is None:
                self.logger.info('📊 Loading market data for single-stage sub-pipeline execution...')
                await self._load_market_data(config)
            
            # Convert config to component config
            component_config = self._convert_to_component_config(config)
            # Enforce 4h timeframe for Regime components only (log warning if overriding)
            if sub_pipeline_name in ('nas_tas_models_training', 'nas_tas_ensemble_training', 'regime_models_training'):
                if component_config.timeframe != '4h':
                    self.logger.warning(f"⚠️ {sub_pipeline_name}: timeframe {component_config.timeframe} supplied; overriding to 4h")
                component_config.timeframe = '4h'

            # Enforce 1m timeframe for regime_data_splitting
            if sub_pipeline_name == 'regime_data_splitting':
                if component_config.timeframe != '1m':
                    self.logger.warning(f"⚠️ {sub_pipeline_name}: timeframe {component_config.timeframe} supplied; overriding to 1m")
                component_config.timeframe = '1m'
            
            # Create component using factory
            component = self.component_factory.create_component(sub_pipeline_name, component_config)
            
            if component is None:
                raise ValueError(f"Component '{sub_pipeline_name}' not found in factory")
            
            # Prepare pipeline state with accumulated artifacts
            pipeline_state_with_artifacts = self._current_pipeline_state.copy()
            pipeline_state_with_artifacts['artifacts'] = self._accumulated_artifacts.copy()

            # Ensure essential pipeline state parameters are present
            # These are required by many components (e.g., optimal_regime_clustering)
            essential_params = ['symbol', 'exchange', 'timeframe', 'data_dir']
            for param in essential_params:
                if param not in pipeline_state_with_artifacts:
                    # Try to get from config
                    if hasattr(config, param):
                        pipeline_state_with_artifacts[param] = getattr(config, param)
                    else:
                        # Use defaults
                        if param == 'data_dir':
                            pipeline_state_with_artifacts[param] = 'historical_data'
                        elif param == 'symbol':
                            pipeline_state_with_artifacts[param] = 'ETHUSDT'
                        elif param == 'exchange':
                            pipeline_state_with_artifacts[param] = 'binance'
                        elif param == 'timeframe':
                            pipeline_state_with_artifacts[param] = '1m'

            # Log missing parameters for debugging
            missing_params = [param for param in essential_params if param not in pipeline_state_with_artifacts]
            if missing_params:
                self.logger.warning(f"⚠️ Some essential parameters were missing and added: {missing_params}")
            
            # Execute component
            component_result = await component.execute(self._current_data, pipeline_state_with_artifacts)
            
            # Accumulate artifacts from this execution
            if component_result.success and component_result.artifacts:
                self._accumulated_artifacts.update(component_result.artifacts)
                self.logger.info(f'📦 Accumulated {len(component_result.artifacts)} artifacts from {sub_pipeline_name}')
                
                # Log artifact persistence status
                if component_result.metadata.get('artifacts_saved_persistently', False):
                    self.logger.info(f'💾 Artifacts from {sub_pipeline_name} saved persistently for cross-stage access')
                else:
                    self.logger.warning(f'⚠️ Artifacts from {sub_pipeline_name} may not be persistently saved')
            
            # Convert component result to sub-pipeline result
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result = SubPipelineResult(
                sub_pipeline_name=sub_pipeline_name,
                status=SubPipelineStatus.COMPLETED if component_result.success else SubPipelineStatus.FAILED,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                artifacts=component_result.artifacts,
                metadata=component_result.metadata,
                error_message=component_result.error_message
            )
            
            # Store result
            self.results.append(result)
            
            # Log completion
            if result.success:
                self._log_sub_pipeline_completion(sub_pipeline_name, config, result.artifacts)
            else:
                self.logger.error(f"❌ {sub_pipeline_name} failed: {result.error_message}")
            
            return result
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result = SubPipelineResult(
                sub_pipeline_name=sub_pipeline_name,
                status=SubPipelineStatus.FAILED,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                error_message=str(e)
            )
            
            # Store result
            self.results.append(result)
            
            self.logger.error(f"❌ {sub_pipeline_name} sub-pipeline failed: {e}")
            return result
    
    def get_available_sub_pipelines(self, stage: Optional[Any] = None) -> List[str]:
        """Get list of available sub-pipelines for a given stage."""
        # Import here to avoid circular imports
        from src.training.steps.main_training_pipeline import PipelineStage

        # If no stage specified or stage is MARKET_ANALYSIS, return market analysis sub-pipelines
        if stage is None or stage == PipelineStage.MARKET_ANALYSIS:
            # Market analysis sub-pipelines
            return [
                'sr_detection',
                'sr_clustering',
                'nas_tas_regime_discovery',
                'nas_tas_clustering',
                'nas_regime_discovery',  # DEPRECATED
                'nas_clustering',        # DEPRECATED
                'regime_models_training',  # Regime detection models training (CatBoost, Bayesian Rule Lists, ExtraTrees)
                'regime_ensemble_training', # Ensemble regime detection models training
                'hmm_training',
                'analyst_model_training',
                'analyst_ensemble_training',
                'tactician_pre_ml_orchestration',
                'tactician_training',
                'regime_specific_training',
                'regime_data_splitting',  # Tag data by regimes
                'model_validation',
                'model_persistence',
                'model_evaluation',
                'basic_backtesting_pre',
                'basic_backtesting_post',
                'walk_forward_validation',
                'monte_carlo_simulation'
            ]

        # For other stages, return empty list (they have their own sub-pipelines)
        return []
    
    def get_sub_pipeline_status(self, sub_pipeline_name: str) -> Optional[SubPipelineStatus]:
        """Get status of a specific sub-pipeline."""
        for result in self.results:
            if result.sub_pipeline_name == sub_pipeline_name:
                return result.status
        return None
    
    async def execute_sub_pipeline_with_next(
        self, 
        sub_pipeline_name: str, 
        config: SubPipelineConfig
    ) -> SubPipelineResult:
        """
        Execute a specific sub-pipeline and automatically trigger subsequent sub-pipelines.
        
        This method provides automatic sequential execution of all market analysis steps:
        1. sr_parameter_optimization - Optimize SR detection levels
        2. sr_detection - Detect Support/Resistance levels
        3. sr_clustering - Generate SR clusters
        4. hybrid_nas_tas_regime_discovery - Discover market regimes using hybrid NAS-TAS approach
        5. nas_tas_clustering - NAS-TAS-based regime clustering
        6. regime_models_training - Regime detection models training (CatBoost, Bayesian Rule Lists, ExtraTrees)
        7. regime_ensemble_training - Meta-learner training (stacker_lgbm_calibrated)
        8. regime_data_splitting - Tag data by regimes

        When one step completes successfully, it automatically triggers the next step.
        
        Args:
            sub_pipeline_name: Name of the sub-pipeline to execute (will trigger all subsequent steps)
            config: Configuration for the sub-pipeline
            
        Returns:
            SubPipelineResult with execution details
        """
        self._prepare_filesystem(config)
        self.logger.info(f'🚀 Starting {sub_pipeline_name} sub-pipeline')
        
        # Check if we should execute only a single stage
        if config.single_stage_only:
            self.logger.info('🎯 Single stage execution mode - executing only the requested sub-pipeline')
            return await self.execute_sub_pipeline(sub_pipeline_name, config)
        
        self.logger.info(f'🚀 Starting {sub_pipeline_name} sub-pipeline with sequential execution')
        
        # Reset accumulated artifacts for new sequence
        self._accumulated_artifacts = {}
        self.logger.info('🔄 Reset accumulated artifacts for new execution sequence')
        
        # Load market data if not already available
        if self._current_data is None:
            self.logger.info('📊 Loading market data for sub-pipeline execution...')
            await self._load_market_data(config)
        
        # Define logical execution groups - ALL sub-pipelines in market_analysis stage
        sr_steps = [
            'sr_parameter_optimization',
            'sr_detection', 
            'sr_clustering'
        ]
        
        regime_steps = [
            'nas_tas_regime_discovery',
            'nas_tas_clustering',
            'regime_models_training',
            'regime_ensemble_training'
        ]
        
        data_processing_steps = [
            'hybrid_nas_tas_regime_discovery',
            'nas_tas_clustering',
            'regime_models_training',
            'regime_ensemble_training',
            'regime_data_splitting',
        ]
        
        # Additional sub-pipelines that were missing
        additional_steps = [
            # cross_timeframe_analysis removed - replaced by interactive_feature_generation
        ]
        
        # Complete execution sequence - ALL sub-pipelines in market_analysis stage
        execution_sequence = sr_steps + regime_steps + data_processing_steps + additional_steps
        
        # Find the starting index
        try:
            start_index = execution_sequence.index(sub_pipeline_name)
        except ValueError:
            self.logger.error(f"❌ Unknown sub-pipeline: {sub_pipeline_name}")
            raise ValueError(f"Unknown sub-pipeline: {sub_pipeline_name}")
        
        # Determine which group we're starting from
        current_group = None
        if sub_pipeline_name in sr_steps:
            current_group = "SR Steps"
            self.logger.info('🎯 Starting from SR steps group - will complete all SR steps before moving to Regime')
        elif sub_pipeline_name in regime_steps:
            current_group = "Regime Steps"
            self.logger.info('🎯 Starting from Regime steps group - will complete all Regime steps before moving to data processing')
        elif sub_pipeline_name in data_processing_steps:
            current_group = "Data Processing Steps"
            self.logger.info('🎯 Starting from data processing steps group')
        elif sub_pipeline_name in additional_steps:
            current_group = "Additional Steps"
            self.logger.info('🎯 Starting from additional steps group')
        
        self.logger.info(f'📋 Execution sequence: {execution_sequence}')
        self.logger.info(f'🚀 Starting from index {start_index}: {sub_pipeline_name}')
        
        # Execute sub-pipelines starting from the specified one
        results = []
        for i in range(start_index, len(execution_sequence)):
            pipeline_name = execution_sequence[i]
            
            # Log group transitions
            if pipeline_name in sr_steps and current_group != "SR Steps":
                self.logger.info('🔄 Transitioning to SR steps group')
                current_group = "SR Steps"
            elif pipeline_name in regime_steps and current_group != "Regime Steps":
                self.logger.info('🔄 Transitioning to Regime steps group')
                current_group = "Regime Steps"
            elif pipeline_name in data_processing_steps and current_group != "Data Processing Steps":
                self.logger.info('🔄 Transitioning to data processing steps group')
                current_group = "Data Processing Steps"
            elif pipeline_name in additional_steps and current_group != "Additional Steps":
                self.logger.info('🔄 Transitioning to additional steps group')
                current_group = "Additional Steps"
            
            try:
                progress_info = f"({i+1-start_index}/{len(execution_sequence)-start_index})"
                self.logger.info(f'🔄 Executing {pipeline_name} {progress_info} [Group: {current_group}]')
                # Ensure 4h timeframe at dispatch time for Regime components only (log warning if overriding)
                if pipeline_name in ('nas_tas_models_training', 'nas_tas_ensemble_training'):
                    # Avoid mutating the shared config; create a scoped copy for this call
                    from dataclasses import replace as _dc_replace
                    scoped_config = _dc_replace(config, timeframe='4h')
                    if getattr(config, 'timeframe', None) != '4h':
                        self.logger.warning(f"⚠️ {pipeline_name}: timeframe {config.timeframe} supplied; overriding to 4h for this step only")
                    result = await self.execute_sub_pipeline(pipeline_name, scoped_config)
                else:
                    result = await self.execute_sub_pipeline(pipeline_name, config)
                results.append(result)
                
                # If this sub-pipeline failed, stop the sequence
                if not result.success:
                    self.logger.error(f"❌ {pipeline_name} failed, stopping execution sequence")
                    break
                    
            except Exception as e:
                self.logger.error(f"❌ Error executing {pipeline_name}: {e}")
                # Create a failed result
                failed_result = SubPipelineResult(
                    sub_pipeline_name=pipeline_name,
                    status=SubPipelineStatus.FAILED,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    duration_seconds=0.0,
                    error_message=str(e)
                )
                results.append(failed_result)
                break
        
        # Return the first result (the one that was requested)
        if results:
            return results[0]
        else:
            # Return a failed result if no execution occurred
            return SubPipelineResult(
                sub_pipeline_name=sub_pipeline_name,
                status=SubPipelineStatus.FAILED,
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration_seconds=0.0,
                error_message="No execution occurred"
            )

    async def _load_market_data(self, config: SubPipelineConfig) -> None:
        """
        Load market data for sub-pipeline execution.
        
        Args:
            config: Sub-pipeline configuration containing symbol, exchange, timeframe, etc.
        """
        try:
            # Import the klines data loader
            from src.utils.data.klines_parquet import load_klines_from_parquet
            
            self.logger.info(f'📊 Loading market data for {config.symbol} on {config.exchange} ({config.timeframe})')
            
            # Get date filtering from config if available
            start_date = None
            end_date = None

            # Check for date attributes on config object
            if hasattr(config, 'start_date') and config.start_date:
                start_date = datetime.strptime(config.start_date, '%Y-%m-%d')
                self.logger.info(f'📅 Using start_date filter: {start_date} (mode: {config.mode.value})')

            if hasattr(config, 'end_date') and config.end_date:
                end_date = datetime.strptime(config.end_date, '%Y-%m-%d')
                self.logger.info(f'📅 Using end_date filter: {end_date} (mode: {config.mode.value})')

            # For light mode, enforce strict 20-day limit regardless of configuration
            if config.mode.value == 'light':
                from datetime import timedelta
                # For light mode, use the last 20 days of available data instead of current date
                # This ensures we use actual historical data rather than future dates
                try:
                    # Try to determine the last available date from the data
                    from src.utils.data.klines_parquet import KlinesParquetManager
                    manager = KlinesParquetManager(data_dir=config.data_dir)

                    # Load a small sample of recent data to determine the date range
                    # Use last 30 days to get a representative sample without loading everything
                    recent_start = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                    recent_end = datetime.now().strftime('%Y-%m-%d')

                    sample_data = manager.read_data(
                        symbol=config.symbol.lower(),
                        interval=config.timeframe,
                        start_date=recent_start,
                        end_date=recent_end,
                        data_type="processed"
                    )

                    if sample_data is not None and not sample_data.empty:
                        # Get the last available date from the data
                        if 'timestamp' in sample_data.columns:
                            timestamps = pd.to_datetime(sample_data['timestamp'], unit='s')
                            end_date = timestamps.max()
                        elif hasattr(sample_data.index, 'max'):
                            end_date = sample_data.index.max()
                        else:
                            # Fallback to current date if we can't determine from data
                            end_date = datetime.now()

                        start_date = end_date - timedelta(days=20)
                        self.logger.info(f'📅 Light mode: Using last 20 days of available data: {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}')
                    else:
                        # No data available, use current date as fallback
                        end_date = datetime.now()
                        start_date = end_date - timedelta(days=20)
                        self.logger.info(f'📅 Light mode: No data available, using calculated range: {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}')
                except Exception as e:
                    # If there's any error, fall back to current date
                    self.logger.warning(f"⚠️ Could not determine available data range for light mode: {e}")
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=20)
                    self.logger.info(f'📅 Light mode: Using fallback date range: {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}')

                self.logger.info(f'📅 Light mode: Exactly 20 days of data for regime diversity')
            
            # Load the klines data directly
            market_data = load_klines_from_parquet(
                symbol=config.symbol.lower(),  # Convert to lowercase to match directory structure
                interval=config.timeframe,
                start_date=start_date,
                end_date=end_date,
                data_type="raw",  # Load raw klines data
                data_dir=config.data_dir
            )
            
            if market_data is None or market_data.empty:
                raise ValueError(f"No market data found for {config.symbol} on {config.exchange} ({config.timeframe})")
            
            self.logger.info(f'📊 Loaded full market data: {market_data.shape[0]} rows, {market_data.shape[1]} columns')
            
            # Skip additional date filtering since it's already handled at the pipeline configuration level
            # The KlinesParquetManager already applies the correct date filtering based on available data
            self.logger.info(f'📅 Date filtering already applied at pipeline level, using loaded data as-is')
            
            self.logger.info(f'✅ Final market data: {market_data.shape[0]} rows, {market_data.shape[1]} columns')
            self.logger.info(f'📊 Data columns: {list(market_data.columns)}')
            self.logger.info(f'📅 Date range: {market_data.index.min()} to {market_data.index.max()}')
            
            # Store the data for component communication
            self._current_data = market_data
            self._current_pipeline_state = {
                'dataframe': market_data,
                'validated_data': market_data,
                'symbol': config.symbol,
                'exchange': config.exchange,
                'timeframe': config.timeframe,
                'data_dir': config.data_dir
            }
            
        except Exception as e:
            self.logger.error(f'❌ Failed to load market data: {e}')
            raise

    def _generate_basic_features(self, data: pd.DataFrame, regime_labels: List[int]) -> Tuple[np.ndarray, List[str]]:
        """
        Generate basic features similar to what regime models training does.
        
        Args:
            data: Market data DataFrame
            regime_labels: List of regime labels
            
        Returns:
            Tuple of (features_array, feature_names)
        """
        try:
            import numpy as np

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
            
            features = []
            feature_names = []
            
            # Create basic price-based features
            if 'close' in data.columns:
                # Price returns
                returns = data['close'].pct_change().fillna(0)
                features.append(returns.values)
                feature_names.append('price_returns')
                
                # Multi-timeframe returns
                returns_5 = data['close'].pct_change(5).fillna(0)
                returns_10 = data['close'].pct_change(10).fillna(0)
                returns_20 = data['close'].pct_change(20).fillna(0)
                features.extend([returns_5.values, returns_10.values, returns_20.values])
                feature_names.extend(['returns_5', 'returns_10', 'returns_20'])
                
                # Moving averages
                sma_10 = data['close'].rolling(10).mean().fillna(data['close'].iloc[0])
                sma_20 = data['close'].rolling(20).mean().fillna(data['close'].iloc[0])
                sma_50 = data['close'].rolling(50).mean().fillna(data['close'].iloc[0])
                features.extend([sma_10.values, sma_20.values, sma_50.values])
                feature_names.extend(['sma_10', 'sma_20', 'sma_50'])
                
                # Price position relative to MAs
                price_to_sma10 = (data['close'] / sma_10 - 1).fillna(0)
                price_to_sma20 = (data['close'] / sma_20 - 1).fillna(0)
                price_to_sma50 = (data['close'] / sma_50 - 1).fillna(0)
                features.extend([price_to_sma10.values, price_to_sma20.values, price_to_sma50.values])
                feature_names.extend(['price_to_sma10', 'price_to_sma20', 'price_to_sma50'])
                
                # Volatility
                volatility_10 = returns.rolling(10).std().fillna(0)
                volatility_20 = returns.rolling(20).std().fillna(0)
                volatility_50 = returns.rolling(50).std().fillna(0)
                features.extend([volatility_10.values, volatility_20.values, volatility_50.values])
                feature_names.extend(['volatility_10', 'volatility_20', 'volatility_50'])
                
                # RSI
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                features.append(rsi.fillna(50).values)
                feature_names.append('rsi_14')
                
                # Momentum
                momentum_5 = data['close'].pct_change(5).fillna(0)
                momentum_10 = data['close'].pct_change(10).fillna(0)
                features.extend([momentum_5.values, momentum_10.values])
                feature_names.extend(['momentum_5', 'momentum_10'])
                
                # Price range features
                if 'high' in data.columns and 'low' in data.columns:
                    price_range = (data['high'] - data['low']) / data['close']
                    hl_position = (data['close'] - data['low']) / (data['high'] - data['low'])
                    features.extend([price_range.fillna(0).values, hl_position.fillna(0.5).values])
                    feature_names.extend(['price_range', 'hl_position'])
            
            # Volume features
            if 'volume' in data.columns:
                volume_change = data['volume'].pct_change().fillna(0)
                volume_ratio = data['volume'] / data['volume'].rolling(20).mean()
                volume_momentum = data['volume'].pct_change(5).fillna(0)
                features.extend([volume_change.values, volume_ratio.fillna(1).values, volume_momentum.values])
                feature_names.extend(['volume_change', 'volume_ratio', 'volume_momentum'])
            
            # Trend strength
            if 'close' in data.columns:
                # Simple trend strength based on price position
                trend_strength = (data['close'] - data['close'].rolling(20).mean()) / data['close'].rolling(20).std()
                features.append(trend_strength.fillna(0).values)
                feature_names.append('trend_strength')
            
            # Convert to numpy array
            if features:
                features_array = np.column_stack(features)
                self.logger.info(f"✅ Generated {len(feature_names)} basic features with shape {features_array.shape}")
                return features_array, feature_names
            else:
                self.logger.warning("⚠️ No features generated")
                return None, []
                
        except Exception as e:
            self.logger.error(f"❌ Failed to generate basic features: {e}")
            return None, []

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary with all results."""
        return {
            'total_sub_pipelines': len(self.results),
            'successful_sub_pipelines': len([r for r in self.results if r.success]),
            'failed_sub_pipelines': len([r for r in self.results if not r.success]),
            'total_execution_time': sum(r.execution_time for r in self.results),
            'sub_pipeline_results': [
                {
                    'name': r.sub_pipeline_name,
                    'status': r.status.value,
                    'success': r.success,
                    'is_complete': r.is_complete,
                    'execution_time': r.execution_time,
                    'artifact_count': len(r.artifacts),
                    'error_message': r.error_message
                }
                for r in self.results
            ]
        }

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _vectorbt_apply_operation(self, data: pd.Series, func, 
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)
        
        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
