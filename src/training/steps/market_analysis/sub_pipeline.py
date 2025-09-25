"""
Market Analysis Sub-Pipeline - Complete 10-Step Pipeline

This module provides the complete market analysis sub-pipeline with the 10 active steps that
are currently implemented and wired:

1. sr_parameter_optimization - Optimize SR detection levels
2. sr_detection - Detect Support/Resistance levels
3. sr_clustering - Generate SR clusters
4. hybrid_nas_tas_regime_discovery - Discover market regimes using hybrid NAS-TAS approach
5. nas_tas_clustering - Consolidated NAS/TAS clustering stage
6. regime_data_splitting - Tag data by regimes
7. feature_lookback_optimization - Optimize feature lookback periods
8. pid_based_feature_generation - PID-based feature generation with interaction, polynomial, and cross-timeframe features
9. multi_horizon_labeling - Apply multi-horizon profit labeling
10. final_feature_selection - Final multi-stage feature selection (120→100→80→60)
"""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
import pandas as pd

from src.utils.logger import system_logger
from src.utils.enhanced_artifact_manager import get_artifact_manager
from src.utils.version_manager import get_version_manager
from .logging_standards import (
    get_logger, log_info, log_warning, log_error, log_success, log_debug,
    LoggingContext, log_step_progress, log_data_info, log_validation_result
)
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, 
    tprint_error, tprint_success, tprint_progress, 
    tprint_performance, tprint_timer
)

# Import utilities with error handling
try:
    from src.utils.common_operations import (
        safe_json_dump, safe_json_load, ensure_directory,
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
        integrate_with_m1_optimizers, cleanup_m1_optimizers,
        safe_dataframe_operation, validate_dataframe_columns, optimize_dataframe_dtypes
    )
    from src.utils.math_validation import (
        validate_finite, validate_positive, safe_divide, safe_log, safe_sqrt,
        safe_correlation, safe_covariance, safe_mean, safe_std
    )
    from src.utils.serialization_utils import UniversalSerializer
    UTILS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Some utilities not available: {e}")
    UTILS_AVAILABLE = False

# Import component system
from .components import ComponentFactory, ComponentConfig

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

@dataclass
class SubPipelineConfig:
    """Configuration for sub-pipeline execution."""
    mode: ExecutionMode = ExecutionMode.FULL
    symbol: str = "BTCUSDT"
    exchange: str = "binance"
    timeframe: str = "30m"
    data_dir: str = "historical_data"
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
            'hybrid_nas_tas_regime_discovery': ['hybrid_nas_tas_regime_discovery_result'],
            'nas_tas_clustering': ['nas_tas_clustering_result'],
            # Backward-compatible alias for legacy component name
            'nas_clustering': ['nas_tas_clustering_result'],
            'regime_data_splitting': ['regime_data_splitting_result'],
            # Support both legacy and current naming for the multi-horizon step
            'multi_horizon_labeling': ['multi_horizon_labeling_result'],
            'multi_horizon_profit_labeler': ['multi_horizon_labeling_result'],
            'feature_lookback_optimization': ['feature_lookback_optimization_result'],
            'pid_based_feature_generation': ['pid_based_feature_generation_result'],
            'final_feature_selection': ['final_feature_selection_result']
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
        """Initialize the market analysis sub-pipeline with enhanced error handling and utility integration."""
        try:
            tprint("🚀 Starting MarketAnalysisSubPipeline initialization...")
            
            # Handle both old dict config and new SubPipelineConfig
            if isinstance(config, dict):
                self.original_config = config
                self.config = self._convert_old_config(config)
            else:
                self.config = config or SubPipelineConfig()
                self.original_config = {}
            
            # Use standardized logging
            self.logger = get_logger('MarketAnalysisSubPipeline')
            self.results: List[SubPipelineResult] = []
            
            # Apply logging configuration
            try:
                self._apply_logging_config(self.config.logging)
            except Exception as e:
                tprint_warning(f"⚠️ Logging configuration failed: {e}")
            
            # Initialize utility systems
            self.utils_available = UTILS_AVAILABLE
            self.serializer = None
            self.m1_optimizers = None
            
            if self.utils_available:
                try:
                    self._initialize_utilities()
                    tprint_success("✅ Utilities initialized")
                except Exception as e:
                    tprint_warning(f"⚠️ Utilities initialization failed: {e}")
                    self.utils_available = False
            
            # Initialize managers
            self.artifact_manager = get_artifact_manager()
            self.version_manager = get_version_manager()
            self.component_factory = ComponentFactory()
            
            # Initialize pipeline state
            self._current_data = None
            self._current_pipeline_state = {}
            self._accumulated_artifacts = {}
            
            tprint_success("✅ MarketAnalysisSubPipeline initialized successfully")
            
        except Exception as e:
            tprint_error(f"❌ MarketAnalysisSubPipeline initialization failed: {e}")
            raise
    
    def _initialize_utilities(self):
        """Initialize utility systems including serialization, M1 optimizers, and ML utilities."""
        try:
            # Initialize serialization utilities
            if self.utils_available:
                self.serializer = UniversalSerializer()
                tprint_info("✅ Serialization utilities initialized")
            
            # Initialize M1 optimizers
            try:
                self.m1_optimizers = integrate_with_m1_optimizers()
                if self.m1_optimizers.get('success', False):
                    tprint_success("✅ M1 optimizers integrated successfully")
                else:
                    tprint_warning("⚠️ M1 optimizers integration failed")
            except Exception as e:
                tprint_warning(f"⚠️ M1 optimizers not available: {e}")
                self.m1_optimizers = None
            
                
        except Exception as e:
            tprint_error(f"❌ Utility initialization failed: {e}")
            raise

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
            tprint_success(f"{stage_name} completed with complete report")
            log_success(f"{stage_name} completed with complete report")
            return True, None
        elif result.success:
            error_msg = f"{stage_name} completed but report is incomplete - missing required artifacts"
            tprint_warning(error_msg)
            log_warning(error_msg)
            return False, {
                'success': False,
                'error': error_msg,
                'stage': result.sub_pipeline_name,
                'incomplete_artifacts': result.artifacts
            }
        else:
            error_msg = f"{stage_name} failed: {result.error_message}"
            tprint_error(error_msg)
            log_error(error_msg)
            return False, {
                'success': False,
                'error': error_msg,
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
            symbol=config.get('symbol', 'BTCUSDT'),
            exchange=config.get('exchange', 'binance'),
            timeframe=config.get('timeframe', '1m'),
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

    async def execute_all_steps_from_start(
        self, 
        config: Optional[SubPipelineConfig] = None
    ) -> Dict[str, Any]:
        """
        Execute all 10 market analysis steps automatically from the beginning.
        
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
            
        log_info('🚀 Starting automatic execution of all 10 market analysis steps')
        log_info('=' * 80)
        log_info('📋 Steps to be executed automatically:')
        log_info('   1. sr_parameter_optimization - Optimize SR detection levels')
        log_info('   2. sr_detection - Detect Support/Resistance levels')
        log_info('   3. sr_clustering - Generate SR clusters')
        log_info('   4. hybrid_nas_tas_regime_discovery - Discover market regimes using hybrid NAS-TAS approach')
        log_info('   5. nas_tas_clustering - Consolidated NAS/TAS clustering stage')
        log_info('   6. regime_data_splitting - Tag data by regimes')
        log_info('   7. feature_lookback_optimization - Optimize feature lookback periods')
        log_info('   8. pid_based_feature_generation - PID-based feature generation with interaction, polynomial, and cross-timeframe features')
        log_info('   9. multi_horizon_profit_labeler - Apply multi-horizon profit labeling')
        log_info('   10. final_feature_selection - Final feature selection (120→100→80→60)')
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

        HMM Steps (4-5):
        4. Hybrid NAS/TAS regime discovery
        5. NAS/TAS clustering

        Data Processing Steps (6-10):
        6. Regime data splitting
        7. Feature lookback optimization
        8. PID-based feature generation
        9. Multi-horizon profit labeling
        10. Final feature selection (120→100→80→60)
        """
        log_info('🎯 Starting Market Analysis Sub-Pipeline execution')
        # Reset results for a fresh run
        self.results = []
        
        try:
            # Extract data from pipeline state
            data = pipeline_state.get('dataframe')
            if data is None:
                error_msg = "No dataframe found in pipeline state"
                tprint_error(error_msg)
                raise ValueError(error_msg)
            
            # Store data and pipeline state for component communication
            self._current_data = data
            self._current_pipeline_state = pipeline_state.copy()
            
            # Initialize results dictionary
            results = {}
            
            # ===== SR STEPS GROUP =====
            log_info('🎯 ===== STARTING SR STEPS GROUP =====')
            
            # Stage 1: SR Parameter Optimization (BEFORE detection and clustering)
            log_info('🎯 Executing Stage 1: SR Parameter Optimization')
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
            
            # Stage 4: Hybrid NAS-TAS Regime Discovery
            self.logger.info('🔍 Executing Stage 4: Hybrid NAS-TAS Regime Discovery')
            hybrid_nas_tas_regime_discovery_result = await self.execute_sub_pipeline('hybrid_nas_tas_regime_discovery', self.config)
            is_success, error_info = self._validate_sub_pipeline_result(hybrid_nas_tas_regime_discovery_result, "Hybrid NAS-TAS Regime Discovery")
            if not is_success:
                return error_info
            
            # Extract data from consolidated artifact
            nas_regime_data = hybrid_nas_tas_regime_discovery_result.artifacts.get('hybrid_nas_tas_regime_discovery_result', {})
            results['regime_models'] = nas_regime_data.get('regime_models', {})
            results['regime_assignments'] = nas_regime_data.get('regime_assignments', {})
            results['regime_metrics'] = nas_regime_data.get('regime_metrics', {})
            
            # Update pipeline state for next components
            self._current_pipeline_state.update({
                'regime_models': results['regime_models'],
                'regime_assignments': results['regime_assignments']
            })
            
            # Stage 5: NAS/TAS Clustering
            self.logger.info('🎯 Executing Stage 5: NAS/TAS Clustering')
            nas_clustering_result = await self.execute_sub_pipeline('nas_tas_clustering', self.config)
            is_success, error_info = self._validate_sub_pipeline_result(nas_clustering_result, "NAS Clustering")
            if not is_success:
                return error_info
            
            # Extract data from consolidated artifact
            nas_clustering_data = nas_clustering_result.artifacts.get('nas_tas_clustering_result', {})
            results['nas_clusters'] = nas_clustering_data.get('nas_clusters', {})
            results['nas_clustering_metrics'] = nas_clustering_data.get('nas_clustering_metrics', {})
            
            # Update pipeline state for next components
            cluster_assignments = nas_clustering_data.get('cluster_assignments', [])
            self._current_pipeline_state.update({
                'nas_clusters': nas_clustering_data,  # Store the full result
                'cluster_assignments': cluster_assignments  # Make cluster_assignments directly accessible
            })
            
            # Prepare data for HMM Models Training
            self.logger.info('📊 Preparing data for HMM Models Training...')
            try:
                # Extract features from optimized_features or pid_based_features
                features = None
                feature_names = []
                
                if 'optimized_features' in results and results['optimized_features']:
                    features_data = results['optimized_features']
                    if isinstance(features_data, dict) and 'features' in features_data:
                        features = features_data['features']
                        feature_names = features_data.get('feature_names', [])
                
                if features is None and 'pid_based_features' in results:
                    pid_features = results['pid_based_features']
                    if isinstance(pid_features, dict) and 'combined_features' in pid_features:
                        features = pid_features['combined_features']
                        feature_names = pid_features.get('combined_feature_names', [])
                
                # Targets are not required: HMM training uses cluster_assignments as labels
                targets = None
                
                # Extract regime labels from regime assignments
                regime_labels = None
                if 'regime_assignments' in results and results['regime_assignments']:
                    regime_data = results['regime_assignments']
                    if isinstance(regime_data, dict) and 'regime_labels' in regime_data:
                        regime_labels = regime_data['regime_labels']
                
                # Update pipeline state with prepared data
                self._current_pipeline_state.update({
                    'features': features,
                    'targets': targets,
                    'regime_labels': regime_labels,
                    'feature_names': feature_names
                })
                
                # Log data availability for debugging
                self.logger.info(f"📊 Data prepared for HMM Models Training:")
                self.logger.info(f"   - Features: {'✅' if features is not None else '❌'}")
                self.logger.info(f"   - Targets: {'✅' if targets is not None else '❌'} (HMM uses cluster_assignments)")
                self.logger.info(f"   - Regime Labels: {'✅' if regime_labels is not None else '❌'}")
                self.logger.info(f"   - Feature Names: {len(feature_names) if feature_names else 0}")
                
            except Exception as e:
                error_msg = f"Failed to prepare data for HMM Models Training: {e}"
                tprint_error(error_msg)
                self.logger.error(f"❌ {error_msg}")
                return self._create_error_result("Data preparation failed for HMM Models Training", str(e))
            
            # ===== DATA PROCESSING STEPS GROUP =====
            self.logger.info('✂️ ===== STARTING DATA PROCESSING STEPS GROUP =====')

            # Stage 6: Regime Data Splitting
            self.logger.info('✂️ Executing Stage 6: Regime Data Splitting')
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

            # Stage 7: Feature Lookback Optimization
            self.logger.info('⚙️ Executing Stage 7: Feature Lookback Optimization')
            feature_lookback_optimization_result = await self.execute_sub_pipeline('feature_lookback_optimization', self.config)
            is_success, error_info = self._validate_sub_pipeline_result(feature_lookback_optimization_result, "Feature Lookback Optimization")
            if not is_success:
                return error_info

            # Extract data from consolidated artifact
            feature_optimization_data = feature_lookback_optimization_result.artifacts.get('feature_lookback_optimization_result', {})
            results['optimized_features'] = feature_optimization_data.get('optimized_features', {})
            results['optimization_metrics'] = feature_optimization_data.get('optimization_metrics', {})

            # Update pipeline state for next components
            self._current_pipeline_state.update({
                'optimized_features': results['optimized_features']
            })

            # Stage 8: PID-Based Feature Generation
            self.logger.info('🔧 Executing Stage 8: PID-Based Feature Generation')
            pid_based_feature_generation_result = await self.execute_sub_pipeline('pid_based_feature_generation', self.config)
            is_success, error_info = self._validate_sub_pipeline_result(pid_based_feature_generation_result, "PID-Based Feature Generation")
            if not is_success:
                return error_info

            # Extract data from consolidated artifact
            pid_feature_data = pid_based_feature_generation_result.artifacts.get('pid_based_feature_generation_result', {})

            # Extract comprehensive PID-based feature generation results
            results['pid_based_features'] = {
                'combined_features': pid_feature_data.get('combined_features', {}),
                'combined_feature_names': pid_feature_data.get('combined_feature_names', []),
                'feature_importance_scores': pid_feature_data.get('feature_importance_scores', {}),
                'interaction_features': pid_feature_data.get('interaction_result', {}),
                'polynomial_features': pid_feature_data.get('polynomial_result', {}),
                'cross_timeframe_features': pid_feature_data.get('cross_timeframe_result', {})
            }

            results['pid_feature_metrics'] = {
                'generation_summary': pid_feature_data.get('generation_summary', {}),
                'quality_metrics': {
                    'overall_quality_score': pid_feature_data.get('overall_quality_score', 0.0),
                    'feature_diversity_score': pid_feature_data.get('feature_diversity_score', 0.0),
                    'redundancy_score': pid_feature_data.get('redundancy_score', 0.0),
                    'stability_score': pid_feature_data.get('stability_score', 0.0)
                },
                'optimization_metrics': {
                    'optimization_used': pid_feature_data.get('optimization_used', False),
                    'matrix_ops_used': pid_feature_data.get('matrix_ops_used', False),
                    'lookback_integration': pid_feature_data.get('lookback_integration', {})
                },
                'validation_result': pid_feature_data.get('validation_result', {}),
                'total_features_generated': pid_feature_data.get('total_features_generated', 0),
                'generation_status': pid_feature_data.get('generation_status', 'unknown')
            }

            # Update pipeline state with features for labeling
            self._current_pipeline_state.update({
                'pid_based_features': results['pid_based_features']
            })

            # Stage 9: Multi-Horizon Profit Labeler (moved here to use optimized features)
            self.logger.info('💰 Executing Stage 9: Multi-Horizon Profit Labeler')
            mh_component_result = await self.execute_sub_pipeline('multi_horizon_profit_labeler', self.config)
            is_success, error_info = self._validate_sub_pipeline_result(mh_component_result, "Multi-Horizon Profit Labeler")
            if not is_success:
                return error_info
            # Extract data from consolidated artifact using dict-safe access
            mh_result_dict = mh_component_result.artifacts
            multi_horizon_data = mh_result_dict.get('multi_horizon_labeling_result', {})
            results['labeled_data'] = multi_horizon_data.get('labeled_data', {})
            results['labeling_metrics'] = multi_horizon_data.get('labeling_metrics', {})
            # Update pipeline state for next components
            self._current_pipeline_state.update({
                'labeled_data': results['labeled_data'],
                'multi_horizon_labeling_result': multi_horizon_data
            })

            # ===== FINAL FEATURE SELECTION STEP =====
            self.logger.info('🏁 ===== STARTING FINAL FEATURE SELECTION =====')

            # Stage 10: Final Feature Selection
            self.logger.info('🏁 Executing Stage 10: Final Feature Selection')
            final_feature_selection_result = await self.execute_sub_pipeline('final_feature_selection', self.config)
            is_success, error_info = self._validate_sub_pipeline_result(final_feature_selection_result, "Final Feature Selection")
            if not is_success:
                return error_info

            # Extract data from consolidated artifact
            final_selection_data = final_feature_selection_result.artifacts.get('final_feature_selection_result', {})

            # Extract final feature selection results
            results['final_feature_selection'] = {
                'symbol': final_selection_data.get('symbol'),
                'exchange': final_selection_data.get('exchange'),
                'timeframe': final_selection_data.get('timeframe'),
                'data_dir': final_selection_data.get('data_dir'),
                'feature_selection_config': final_selection_data.get('feature_selection_config', {}),
                'execution_mode': final_selection_data.get('execution_mode', 'component'),
                'success': final_selection_data.get('success', True),
                'stage_reduction': final_selection_data.get('stage_reduction', {
                    'initial': 120,
                    'stage_1': 100,
                    'stage_2': 80,
                    'stage_3': 60
                })
            }

            # Final success
            self.logger.info('🎉 Market Analysis Sub-Pipeline completed successfully')
            return {
                'success': True,
                'results': results,
                'execution_time': sum(result.execution_time for result in self.results),
                'total_stages': 10,
                'completed_stages': len(self.results)
            }
            
        except Exception as e:
            error_msg = f"Market Analysis Sub-Pipeline failed: {e}"
            tprint_error(error_msg)
            self.logger.error(f'❌ {error_msg}')
            import traceback
            traceback_msg = f"Error details: {traceback.format_exc()}"
            tprint_error(traceback_msg)
            self.logger.error(f'❌ {traceback_msg}')
            return {
                'success': False,
                'error': str(e),
                'execution_time': sum(result.execution_time for result in self.results),
                'completed_stages': len(self.results)
            }
    
    def validate_config(self):
        """Validate the sub-pipeline configuration."""
        try:
            if not self.config.symbol:
                error_msg = "Symbol is required"
                tprint_error(error_msg)
                raise ValueError(error_msg)
            if not self.config.exchange:
                error_msg = "Exchange is required"
                tprint_error(error_msg)
                raise ValueError(error_msg)
            if not self.config.timeframe:
                error_msg = "Timeframe is required"
                tprint_error(error_msg)
                raise ValueError(error_msg)
            tprint_success("Configuration validation passed")
        except Exception as e:
            tprint_error(f"Configuration validation failed: {e}")
            raise
    
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
        if sub_pipeline_name in ('nas_tas_clustering', 'nas_clustering') and artifact_count > 1:
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
        start_time = datetime.now()
        self.logger.info(f'🚀 Starting {sub_pipeline_name} sub-pipeline')
        
        try:
            # Load market data if not already available
            if self._current_data is None:
                self.logger.info('📊 Loading market data for single-stage sub-pipeline execution...')
                await self._load_market_data(config)
            
            # Convert config to component config
            component_config = self._convert_to_component_config(config)
            # Enforce 15m timeframe for HMM components only (log warning if overriding)
            if sub_pipeline_name in ('hmm_models_training', 'hmm_ensemble_training'):
                if component_config.timeframe != '15m':
                    self.logger.warning(f"⚠️ {sub_pipeline_name}: timeframe {component_config.timeframe} supplied; overriding to 15m")
                component_config.timeframe = '15m'
            
            # Create component using factory
            component = self.component_factory.create_component(sub_pipeline_name, component_config)
            
            if component is None:
                error_msg = f"Component '{sub_pipeline_name}' not found in factory"
                tprint_error(error_msg)
                raise ValueError(error_msg)
            
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
                            pipeline_state_with_artifacts[param] = 'BTCUSDT'
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
    
    def get_available_sub_pipelines(self) -> List[str]:
        """Get list of available sub-pipelines."""
        return list(self.component_factory.get_available_components())
    
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
        5. nas_tas_clustering - Consolidated NAS/TAS clustering stage
        6. regime_data_splitting - Tag data by regimes
        7. feature_lookback_optimization - Optimize feature lookback periods
        8. pid_based_feature_generation - PID-based feature generation with interaction, polynomial, and cross-timeframe features
        9. multi_horizon_profit_labeler - Apply multi-horizon profit labeling
        10. final_feature_selection - Final feature selection (120→100→80→60)

        When one step completes successfully, it automatically triggers the next step.
        
        Args:
            sub_pipeline_name: Name of the sub-pipeline to execute (will trigger all subsequent steps)
            config: Configuration for the sub-pipeline
            
        Returns:
            SubPipelineResult with execution details
        """
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
        
        hmm_steps = [
            'hybrid_nas_tas_regime_discovery',
            'nas_tas_clustering'
        ]
        
        data_processing_steps = [
            'regime_data_splitting',
            'feature_lookback_optimization',
            'pid_based_feature_generation',
            'multi_horizon_profit_labeler',  # Uses regime-tagged data and generated features
            'final_feature_selection'
        ]
        
        # Additional sub-pipelines that were missing
        additional_steps = [
            # cross_timeframe_analysis removed - redundant wrapper for pid_based_feature_generation
        ]
        
        # Complete execution sequence - ALL sub-pipelines in market_analysis stage
        execution_sequence = sr_steps + hmm_steps + data_processing_steps + additional_steps
        
        # Find the starting index
        try:
            start_index = execution_sequence.index(sub_pipeline_name)
        except ValueError:
            error_msg = f"Unknown sub-pipeline: {sub_pipeline_name}"
            tprint_error(error_msg)
            self.logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)
        
        # Determine which group we're starting from
        current_group = None
        if sub_pipeline_name in sr_steps:
            current_group = "SR Steps"
            self.logger.info('🎯 Starting from SR steps group - will complete all SR steps before moving to HMM')
        elif sub_pipeline_name in hmm_steps:
            current_group = "HMM Steps"
            self.logger.info('🎯 Starting from HMM steps group - will complete all HMM steps before moving to data processing')
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
            elif pipeline_name in hmm_steps and current_group != "HMM Steps":
                self.logger.info('🔄 Transitioning to HMM steps group')
                current_group = "HMM Steps"
            elif pipeline_name in data_processing_steps and current_group != "Data Processing Steps":
                self.logger.info('🔄 Transitioning to data processing steps group')
                current_group = "Data Processing Steps"
            elif pipeline_name in additional_steps and current_group != "Additional Steps":
                self.logger.info('🔄 Transitioning to additional steps group')
                current_group = "Additional Steps"
            
            try:
                progress_info = f"({i+1-start_index}/{len(execution_sequence)-start_index})"
                self.logger.info(f'🔄 Executing {pipeline_name} {progress_info} [Group: {current_group}]')
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
            if hasattr(config, 'start_date') and config.start_date:
                start_date = datetime.strptime(config.start_date, '%Y-%m-%d')
                self.logger.info(f'📅 Using start_date filter: {start_date} (mode: {config.mode.value})')
            if hasattr(config, 'end_date') and config.end_date:
                end_date = datetime.strptime(config.end_date, '%Y-%m-%d')
                self.logger.info(f'📅 Using end_date filter: {end_date} (mode: {config.mode.value})')
            
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
                error_msg = f"No market data found for {config.symbol} on {config.exchange} ({config.timeframe})"
                tprint_error(error_msg)
                raise ValueError(error_msg)
            
            self.logger.info(f'📊 Loaded full market data: {market_data.shape[0]} rows, {market_data.shape[1]} columns')
            
            # Apply date filtering after loading if dates are specified
            if start_date is not None or end_date is not None:
                original_rows = len(market_data)
                
                # Convert index to datetime if it isn't already
                if not isinstance(market_data.index, pd.DatetimeIndex):
                    try:
                        if hasattr(market_data.index, 'max') and market_data.index.max() > 1e10:
                            # Likely millisecond timestamps
                            market_data.index = pd.to_datetime(market_data.index, unit='ms', utc=True).tz_localize(None)
                        else:
                            market_data.index = pd.to_datetime(market_data.index, utc=True).tz_localize(None)
                    except Exception as e:
                        self.logger.warning(f"⚠️ Could not convert index to datetime for filtering: {e}")
                
                # Apply date filtering
                if start_date is not None:
                    market_data = market_data[market_data.index >= start_date]
                    self.logger.info(f'📅 Applied start_date filter: {start_date}')
                
                if end_date is not None:
                    market_data = market_data[market_data.index <= end_date]
                    self.logger.info(f'📅 Applied end_date filter: {end_date}')
                
                filtered_rows = len(market_data)
                self.logger.info(f'🔍 Date filtering: {original_rows:,} → {filtered_rows:,} rows ({filtered_rows/original_rows*100:.1f}%)')
            
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
            error_msg = f"Failed to load market data: {e}"
            tprint_error(error_msg)
            self.logger.error(f'❌ {error_msg}')
            raise

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