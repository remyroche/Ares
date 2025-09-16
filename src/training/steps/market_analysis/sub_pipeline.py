"""
Market Analysis Sub-Pipeline - Complete 11-Step Pipeline

This module provides the complete market analysis sub-pipeline with exactly 11 required steps:

1. sr_parameter_optimization - Optimize SR detection levels
2. sr_detection - Detect Support/Resistance levels
3. sr_clustering - Generate SR clusters
4. hmm_regime_discovery - Discover market regimes
5. hmm_clustering - HMM-based regime clustering
6. hmm_models_training - Base models training, HPO, saving, metrics
7. hmm_ensemble_training - Meta-model, HPO, saving, metrics
8. regime_data_splitting - Tag data by regimes
9. triple_barrier_labeling - Apply triple barrier method
10. feature_lookback_optimization - Optimize feature lookback periods
11. pid_based_feature_generation - PID-based feature generation with interaction, polynomial, and cross-timeframe features
"""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

from src.utils.logger import system_logger
from src.utils.enhanced_artifact_manager import get_artifact_manager
from src.utils.version_manager import get_version_manager

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
            'hmm_regime_discovery': ['hmm_regime_discovery_result'],
            'hmm_clustering': ['hmm_clustering_result'],
            'hmm_models_training': ['hmm_models_training_result'],
            'hmm_ensemble_training': ['hmm_ensemble_training_result'],
            'regime_data_splitting': ['regime_data_splitting_result'],
            'triple_barrier_labeling': ['triple_barrier_labeling_result'],
            'feature_lookback_optimization': ['feature_lookback_optimization_result'],
            'pid_based_feature_generation': ['pid_based_feature_generation_result']
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
        
        self.logger = logger.getChild('MarketAnalysisSubPipeline')
        self.results: List[SubPipelineResult] = []
        
        # Initialize artifact and version managers
        self.artifact_manager = get_artifact_manager()
        self.version_manager = get_version_manager()
        
        # Initialize component factory
        self.component_factory = ComponentFactory()
        
        # Initialize pipeline state for component communication
        self._current_data = None
        self._current_pipeline_state = {}
        self._accumulated_artifacts = {}
    
    def _validate_sub_pipeline_result(self, result: SubPipelineResult, stage_name: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Validate sub-pipeline result and return success status and error info.
        
        Returns:
            Tuple of (is_success, error_dict_or_none)
        """
        if result.is_complete:
            self.logger.info(f"✅ {stage_name} completed with complete report")
            return True, None
        elif result.success:
            self.logger.warning(f"⚠️ {stage_name} completed but report is incomplete")
            return False, {
                'success': False,
                'error': f"{stage_name} produced incomplete report - missing required artifacts",
                'stage': result.sub_pipeline_name,
                'incomplete_artifacts': result.artifacts
            }
        else:
            self.logger.error(f"❌ {stage_name} failed: {result.error_message}")
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
    
    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the complete market analysis sub-pipeline with backward compatible interface.

        This method orchestrates the complete market analysis pipeline with logical groupings:
        
        SR Steps (1-3):
        1. SR parameter optimization
        2. SR detection  
        3. SR clustering
        
        HMM Steps (4-7):
        4. HMM regime discovery
        5. HMM clustering
        6. HMM models training with HPO
        7. HMM ensemble training (meta-model)
        
        Data Processing Steps (8-11):
        8. Regime data splitting
        9. Triple barrier labeling
        10. Feature lookback optimization
        11. PID-based feature generation
        """
        self.logger.info('🎯 Starting Market Analysis Sub-Pipeline execution')
        
        try:
            # Extract data from pipeline state
            data = pipeline_state.get('dataframe')
            if data is None:
                raise ValueError("No dataframe found in pipeline state")
            
            # Store data and pipeline state for component communication
            self._current_data = data
            self._current_pipeline_state = pipeline_state.copy()
            
            # Initialize results dictionary
            results = {}
            
            # ===== SR STEPS GROUP =====
            self.logger.info('🎯 ===== STARTING SR STEPS GROUP =====')
            
            # Stage 1: SR Parameter Optimization (BEFORE detection and clustering)
            self.logger.info('🎯 Executing Stage 1: SR Parameter Optimization')
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
            
            # Stage 4: HMM Regime Discovery
            self.logger.info('🔍 Executing Stage 4: HMM Regime Discovery')
            hmm_regime_discovery_result = await self.execute_sub_pipeline('hmm_regime_discovery', self.config)
            is_success, error_info = self._validate_sub_pipeline_result(hmm_regime_discovery_result, "HMM Regime Discovery")
            if not is_success:
                return error_info
            
            # Extract data from consolidated artifact
            hmm_regime_data = hmm_regime_discovery_result.artifacts.get('hmm_regime_discovery_result', {})
            results['regime_models'] = hmm_regime_data.get('regime_models', {})
            results['regime_assignments'] = hmm_regime_data.get('regime_assignments', {})
            results['regime_metrics'] = hmm_regime_data.get('regime_metrics', {})
            
            # Update pipeline state for next components
            self._current_pipeline_state.update({
                'regime_models': results['regime_models'],
                'regime_assignments': results['regime_assignments']
            })
            
            # Stage 5: HMM Clustering
            self.logger.info('🎯 Executing Stage 5: HMM Clustering')
            hmm_clustering_result = await self.execute_sub_pipeline('hmm_clustering', self.config)
            is_success, error_info = self._validate_sub_pipeline_result(hmm_clustering_result, "HMM Clustering")
            if not is_success:
                return error_info
            
            # Extract data from consolidated artifact
            hmm_clustering_data = hmm_clustering_result.artifacts.get('hmm_clustering_result', {})
            results['hmm_clusters'] = hmm_clustering_data.get('hmm_clusters', {})
            results['hmm_clustering_metrics'] = hmm_clustering_data.get('hmm_clustering_metrics', {})
            
            # Update pipeline state for next components
            self._current_pipeline_state.update({
                'hmm_clusters': results['hmm_clusters']
            })
            
            # Stage 6: HMM Models Training
            self.logger.info('🏋️ Executing Stage 6: HMM Models Training')
            hmm_models_training_result = await self.execute_sub_pipeline('hmm_models_training', self.config)
            is_success, error_info = self._validate_sub_pipeline_result(hmm_models_training_result, "HMM Models Training")
            if not is_success:
                return error_info
            
            # Extract data from consolidated artifact
            hmm_models_data = hmm_models_training_result.artifacts.get('hmm_models_training_result', {})
            results['hmm_models'] = hmm_models_data.get('hmm_models', {})
            results['hmm_training_metrics'] = hmm_models_data.get('hmm_training_metrics', {})
            
            # Update pipeline state for next components
            self._current_pipeline_state.update({
                'hmm_models': results['hmm_models']
            })
            
            # Stage 7: HMM Ensemble Training
            self.logger.info('🎭 Executing Stage 7: HMM Ensemble Training')
            hmm_ensemble_training_result = await self.execute_sub_pipeline('hmm_ensemble_training', self.config)
            is_success, error_info = self._validate_sub_pipeline_result(hmm_ensemble_training_result, "HMM Ensemble Training")
            if not is_success:
                return error_info
            
            # Extract data from consolidated artifact
            hmm_ensemble_data = hmm_ensemble_training_result.artifacts.get('hmm_ensemble_training_result', {})
            results['hmm_ensemble'] = hmm_ensemble_data.get('hmm_ensemble', {})
            results['hmm_ensemble_metrics'] = hmm_ensemble_data.get('hmm_ensemble_metrics', {})
            
            # Update pipeline state for next components
            self._current_pipeline_state.update({
                'hmm_ensemble': results['hmm_ensemble']
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
            
            # Stage 9: Triple Barrier Labeling
            self.logger.info('🏷️ Executing Stage 9: Triple Barrier Labeling')
            try:
                from src.training.steps.market_analysis.triple_barrier_labeling.step import execute_triple_barrier_labeling_step
                triple_barrier_labeling_result = await execute_triple_barrier_labeling_step(
                    data, self._current_pipeline_state, self.config
                )
                
                # Create a mock result object for compatibility
                class MockResult:
                    def __init__(self, result_dict):
                        self.artifacts = result_dict.get('artifacts', {})
                        self.status = result_dict.get('status', 'unknown')
                        self.metadata = result_dict.get('metadata', {})
                
                triple_barrier_labeling_result = MockResult(triple_barrier_labeling_result)
                
                if triple_barrier_labeling_result.status != 'completed':
                    return self._create_error_result("Triple Barrier Labeling failed", triple_barrier_labeling_result.artifacts)
                    
            except Exception as e:
                self.logger.error(f"Triple Barrier Labeling execution failed: {e}")
                return self._create_error_result("Triple Barrier Labeling execution failed", str(e))
            
            # Extract data from consolidated artifact
            triple_barrier_data = triple_barrier_labeling_result.artifacts.get('triple_barrier_labeling_result', {})
            results['labeled_data'] = triple_barrier_data.get('labeled_data', {})
            results['labeling_metrics'] = triple_barrier_data.get('labeling_metrics', {})
            
            # Update pipeline state for next components
            self._current_pipeline_state.update({
                'labeled_data': results['labeled_data']
            })
            
            # Stage 10: Feature Lookback Optimization
            self.logger.info('⚙️ Executing Stage 10: Feature Lookback Optimization')
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
            
            # Stage 11: PID-Based Feature Generation
            self.logger.info('🔧 Executing Stage 11: PID-Based Feature Generation')
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
            
            # Final success
            self.logger.info('🎉 Market Analysis Sub-Pipeline completed successfully')
            return {
                'success': True,
                'results': results,
                'execution_time': sum(result.execution_time for result in self.results),
                'total_stages': 11,
                'completed_stages': len(self.results)
            }
            
        except Exception as e:
            self.logger.error(f'❌ Market Analysis Sub-Pipeline failed: {e}')
            import traceback
            self.logger.error(f'❌ Error details: {traceback.format_exc()}')
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
            # Convert config to component config
            component_config = self._convert_to_component_config(config)
            
            # Create component using factory
            component = self.component_factory.create_component(sub_pipeline_name, component_config)
            
            if component is None:
                raise ValueError(f"Component '{sub_pipeline_name}' not found in factory")
            
            # Prepare pipeline state with accumulated artifacts
            pipeline_state_with_artifacts = self._current_pipeline_state.copy()
            pipeline_state_with_artifacts['artifacts'] = self._accumulated_artifacts.copy()
            
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
    
    async def execute_sub_pipeline_with_next(
        self, 
        sub_pipeline_name: str, 
        config: SubPipelineConfig
    ) -> SubPipelineResult:
        """
        Execute a specific sub-pipeline and then automatically execute the next sub-pipeline in sequence.
        
        Args:
            sub_pipeline_name: Name of the sub-pipeline to execute
            config: Configuration for the sub-pipeline
            
        Returns:
            SubPipelineResult with execution details
        """
        # Execute the requested sub-pipeline
        result = await self.execute_sub_pipeline(sub_pipeline_name, config)
        
        # If successful, determine and execute the next sub-pipeline
        if result.success:
            next_sub_pipeline = self._get_next_sub_pipeline(sub_pipeline_name)
            if next_sub_pipeline:
                self.logger.info(f"🔄 Auto-executing next sub-pipeline: {next_sub_pipeline}")
                next_result = await self.execute_sub_pipeline(next_sub_pipeline, config)
                
                # Update the result to include next pipeline execution
                result.metadata = result.metadata or {}
                result.metadata['next_pipeline_executed'] = next_sub_pipeline
                result.metadata['next_pipeline_success'] = next_result.success
                if not next_result.success:
                    result.metadata['next_pipeline_error'] = next_result.error_message
        
        return result
    
    def _get_next_sub_pipeline(self, current_sub_pipeline: str) -> Optional[str]:
        """
        Determine the next sub-pipeline to execute based on the current one.
        
        Args:
            current_sub_pipeline: Name of the current sub-pipeline
            
        Returns:
            Name of the next sub-pipeline, or None if no next pipeline
        """
        # Define the execution sequence for market analysis sub-pipelines
        sequence = [
            'sr_parameter_optimization',
            'sr_detection', 
            'sr_clustering',
            'hmm_regime_discovery',
            'hmm_clustering',
            'hmm_models_training',
            'hmm_ensemble_training',
            'regime_data_splitting',
            'triple_barrier_labeling',
            'feature_lookback_optimization',
            'pid_based_feature_generation'
        ]
        
        try:
            current_index = sequence.index(current_sub_pipeline)
            if current_index < len(sequence) - 1:
                return sequence[current_index + 1]
        except ValueError:
            # Current sub-pipeline not in sequence, no next pipeline
            pass
        
        return None
    
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
        Execute a specific sub-pipeline and conditionally trigger subsequent sub-pipelines.
        
        This method provides the interface expected by the main training pipeline for
        automatic sequential execution of sub-pipelines, following logical groupings:
        - SR steps: parameter optimization -> detection -> clustering
        - HMM steps: regime discovery -> clustering -> models -> ensemble
        - Data processing: regime splitting -> labeling -> feature optimization -> generation
        
        Args:
            sub_pipeline_name: Name of the sub-pipeline to execute
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
        
        # Define logical execution groups
        sr_steps = [
            'sr_parameter_optimization',
            'sr_detection', 
            'sr_clustering'
        ]
        
        hmm_steps = [
            'hmm_regime_discovery',
            'hmm_clustering',
            'hmm_models_training',
            'hmm_ensemble_training'
        ]
        
        data_processing_steps = [
            'regime_data_splitting',
            'triple_barrier_labeling',
            'feature_lookback_optimization',
            'pid_based_feature_generation'
        ]
        
        # Complete execution sequence
        execution_sequence = sr_steps + hmm_steps + data_processing_steps
        
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
            self.logger.info('🎯 Starting from SR steps group - will complete all SR steps before moving to HMM')
        elif sub_pipeline_name in hmm_steps:
            current_group = "HMM Steps"
            self.logger.info('🎯 Starting from HMM steps group - will complete all HMM steps before moving to data processing')
        elif sub_pipeline_name in data_processing_steps:
            current_group = "Data Processing Steps"
            self.logger.info('🎯 Starting from data processing steps group')
        
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
            # Import the unified data loader
            from ..data_collection.unified_data_loader import UnifiedDataLoader
            
            self.logger.info(f'📊 Loading market data for {config.symbol} on {config.exchange} ({config.timeframe})')
            
            # Create data loader
            data_loader = UnifiedDataLoader()
            
            # Load the data
            market_data = await data_loader.load_unified_data(
                symbol=config.symbol,
                exchange=config.exchange,
                timeframe=config.timeframe,
                data_dir=config.data_dir
            )
            
            if market_data is None or market_data.empty:
                raise ValueError(f"No market data found for {config.symbol} on {config.exchange} ({config.timeframe})")
            
            self.logger.info(f'✅ Loaded market data: {market_data.shape[0]} rows, {market_data.shape[1]} columns')
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