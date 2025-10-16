"""
Unified Pipeline for NAS/TAS Systems

This module provides the main entry point for the unified NAS/TAS pipeline,
combining all shared utilities into a comprehensive system.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer,
    tprint_structured, tprint_with_level, tprint_logged, configure_tprint,
    TPrintConfig, LogLevel, TimestampFormat
)

# Import all unified components
from .config.base_config import UnifiedArchitectureConfig, create_comprehensive_config
from .data.data_processor import UnifiedDataProcessor, DataProcessingConfig
from .evaluation.unified_evaluator import UnifiedEvaluator, EvaluationConfig
from .results.result_manager import ResultManager, UnifiedArchitectureResult
from .training.training_orchestrator import UnifiedTrainingOrchestrator, TrainingConfig
from .error_handling import UnifiedErrorHandler
from .logging import UnifiedLogger, LoggingConfig
from .interfaces import ArchitectureSearchInterface, TrainingPipelineInterface


@dataclass
class UnifiedPipelineConfig:
    """Configuration for the unified NAS/TAS pipeline."""
    
    # Component configurations
    architecture_config: Optional[UnifiedArchitectureConfig] = None
    data_config: Optional[DataProcessingConfig] = None
    evaluation_config: Optional[EvaluationConfig] = None
    training_config: Optional[TrainingConfig] = None
    logging_config: Optional[LoggingConfig] = None
    
    # Pipeline settings
    enable_data_processing: bool = True
    enable_architecture_search: bool = True
    enable_model_training: bool = True
    enable_evaluation: bool = True
    enable_result_storage: bool = True
    
    # Performance settings
    enable_parallel_processing: bool = True
    max_workers: int = 4
    
    # Output settings
    output_directory: str = "nas_tas_output"
    save_intermediate_results: bool = True
    generate_reports: bool = True
    
    # TPrint configuration
    tprint_config: Optional[TPrintConfig] = None
    enable_extensive_logging: bool = True
    log_level: LogLevel = LogLevel.INFO
    enable_performance_logging: bool = True
    enable_structured_logging: bool = True
    
    def __post_init__(self):
        """Initialize default configurations if not provided."""
        if self.architecture_config is None:
            self.architecture_config = create_comprehensive_config()
        
        if self.data_config is None:
            self.data_config = DataProcessingConfig()
        
        if self.evaluation_config is None:
            self.evaluation_config = EvaluationConfig()
        
        if self.training_config is None:
            self.training_config = TrainingConfig()
        
        if self.logging_config is None:
            self.logging_config = LoggingConfig()
        
        # Configure tprint for extensive logging
        if self.tprint_config is None:
            self.tprint_config = TPrintConfig(
                timestamp_format=TimestampFormat.WITH_MICROSECONDS,
                use_colors=True,
                output_to_console=True,
                output_to_file=True,
                output_file=f"{self.output_directory}/nas_tas_pipeline.log",
                min_log_level=self.log_level,
                enable_structured_logging=self.enable_structured_logging,
                integrate_with_logging=True,
                auto_log_prints=True,
                capture_print_to_tprint=True
            )
        
        # Apply tprint configuration
        configure_tprint(self.tprint_config)
        
        # Log pipeline configuration
        if self.enable_extensive_logging:
            tprint_info("Unified Pipeline Configuration initialized")
            tprint_structured({
                "pipeline_settings": {
                    "enable_data_processing": self.enable_data_processing,
                    "enable_architecture_search": self.enable_architecture_search,
                    "enable_model_training": self.enable_model_training,
                    "enable_evaluation": self.enable_evaluation,
                    "enable_result_storage": self.enable_result_storage,
                    "enable_parallel_processing": self.enable_parallel_processing,
                    "max_workers": self.max_workers
                },
                "output_settings": {
                    "output_directory": self.output_directory,
                    "save_intermediate_results": self.save_intermediate_results,
                    "generate_reports": self.generate_reports
                },
                "logging_settings": {
                    "enable_extensive_logging": self.enable_extensive_logging,
                    "log_level": self.log_level.value,
                    "enable_performance_logging": self.enable_performance_logging,
                    "enable_structured_logging": self.enable_structured_logging
                }
            }, LogLevel.INFO)


class UnifiedNASPipeline:
    """
    Unified pipeline for Neural Architecture Search.
    
    This class provides a complete pipeline for NAS implementations
    using the shared utilities framework.
    """
    
    def __init__(self, config: Optional[UnifiedPipelineConfig] = None):
        """Initialize unified NAS pipeline."""
        tprint_info("Initializing Unified NAS Pipeline")
        
        self.config = config or UnifiedPipelineConfig()
        
        # Log initialization start
        tprint_debug("Starting NAS pipeline component initialization")
        
        # Initialize components with detailed logging
        with tprint_timer("data_processor_initialization", LogLevel.DEBUG):
            self.data_processor = UnifiedDataProcessor(self.config.data_config)
            tprint_success("Data processor initialized")
        
        with tprint_timer("evaluator_initialization", LogLevel.DEBUG):
            self.evaluator = UnifiedEvaluator(self.config.evaluation_config)
            tprint_success("Evaluator initialized")
        
        with tprint_timer("training_orchestrator_initialization", LogLevel.DEBUG):
            self.training_orchestrator = UnifiedTrainingOrchestrator(
                self.config.training_config,
                self.config.architecture_config
            )
            tprint_success("Training orchestrator initialized")
        
        with tprint_timer("result_manager_initialization", LogLevel.DEBUG):
            self.result_manager = ResultManager(self.config.output_directory)
            tprint_success("Result manager initialized")
        
        with tprint_timer("error_handler_initialization", LogLevel.DEBUG):
            self.error_handler = UnifiedErrorHandler()
            tprint_success("Error handler initialized")
        
        with tprint_timer("logger_initialization", LogLevel.DEBUG):
            self.logger = UnifiedLogger(self.config.logging_config)
            tprint_success("Logger initialized")
        
        # Log pipeline configuration
        if self.config.enable_extensive_logging:
            tprint_structured({
                "pipeline_type": "NAS",
                "components_initialized": [
                    "data_processor", "evaluator", "training_orchestrator",
                    "result_manager", "error_handler", "logger"
                ],
                "configuration": {
                    "enable_data_processing": self.config.enable_data_processing,
                    "enable_architecture_search": self.config.enable_architecture_search,
                    "enable_model_training": self.config.enable_model_training,
                    "enable_evaluation": self.config.enable_evaluation,
                    "enable_result_storage": self.config.enable_result_storage,
                    "enable_parallel_processing": self.config.enable_parallel_processing,
                    "max_workers": self.config.max_workers
                }
            }, LogLevel.INFO)
        
        tprint_success("Unified NAS Pipeline initialized successfully")
    
    async def execute_pipeline(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        target: Optional[Union[np.ndarray, pd.Series]] = None,
        search_interface: Optional[ArchitectureSearchInterface] = None
    ) -> UnifiedArchitectureResult:
        """
        Execute complete NAS pipeline.
        
        Args:
            data: Training data
            target: Target variable
            search_interface: NAS search interface
            
        Returns:
            UnifiedArchitectureResult with complete pipeline results
        """
        tprint_info("Starting Unified NAS Pipeline execution")
        
        # Log input data information
        if self.config.enable_extensive_logging:
            data_shape = data.shape if hasattr(data, 'shape') else f"Unknown shape: {type(data)}"
            target_shape = target.shape if target is not None and hasattr(target, 'shape') else "No target"
            search_interface_type = type(search_interface).__name__ if search_interface else "No search interface"
            
            tprint_structured({
                "pipeline_execution": {
                    "pipeline_type": "NAS",
                    "data_shape": data_shape,
                    "target_shape": target_shape,
                    "search_interface": search_interface_type,
                    "timestamp": datetime.now().isoformat()
                }
            }, LogLevel.INFO)
        
        # Track execution with performance logging
        with tprint_timer("nas_pipeline_execution", LogLevel.INFO):
            try:
                # Execute training orchestration (which includes data processing, search, and training)
                tprint_debug("Executing training orchestration")
                result = await self.training_orchestrator.execute_training(
                    data, target, search_interface
                )
                
                # Log training results
                if self.config.enable_extensive_logging:
                    tprint_structured({
                        "training_results": {
                            "execution_successful": result.execution_info.status.value == "SUCCESS",
                            "duration_seconds": result.execution_info.duration_seconds,
                            "architecture_count": result.architecture_count,
                            "search_type": result.search_type,
                            "optimization_mode": result.optimization_mode
                        }
                    }, LogLevel.INFO)
                
                # Store results
                if self.config.enable_result_storage:
                    tprint_debug("Storing pipeline results")
                    with tprint_timer("result_storage", LogLevel.DEBUG):
                        self.result_manager.store_result(result)
                    tprint_success("Results stored successfully")
                else:
                    tprint_info("Result storage disabled - skipping storage")
                
                tprint_success("Unified NAS Pipeline execution completed successfully")
                return result
                
            except Exception as e:
                tprint_error(f"NAS Pipeline execution failed: {e}")
                if self.config.enable_extensive_logging:
                    tprint_structured({
                        "pipeline_error": {
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "pipeline_type": "NAS",
                            "timestamp": datetime.now().isoformat()
                        }
                    }, LogLevel.ERROR)
                raise


class UnifiedTASPipeline:
    """
    Unified pipeline for Tree Architecture Search.
    
    This class provides a complete pipeline for TAS implementations
    using the shared utilities framework.
    """
    
    def __init__(self, config: Optional[UnifiedPipelineConfig] = None):
        """Initialize unified TAS pipeline."""
        tprint_info("Initializing Unified TAS Pipeline")
        
        self.config = config or UnifiedPipelineConfig()
        
        # Log initialization start
        tprint_debug("Starting TAS pipeline component initialization")
        
        # Initialize components with detailed logging
        with tprint_timer("data_processor_initialization", LogLevel.DEBUG):
            self.data_processor = UnifiedDataProcessor(self.config.data_config)
            tprint_success("Data processor initialized")
        
        with tprint_timer("evaluator_initialization", LogLevel.DEBUG):
            self.evaluator = UnifiedEvaluator(self.config.evaluation_config)
            tprint_success("Evaluator initialized")
        
        with tprint_timer("training_orchestrator_initialization", LogLevel.DEBUG):
            self.training_orchestrator = UnifiedTrainingOrchestrator(
                self.config.training_config,
                self.config.architecture_config
            )
            tprint_success("Training orchestrator initialized")
        
        with tprint_timer("result_manager_initialization", LogLevel.DEBUG):
            self.result_manager = ResultManager(self.config.output_directory)
            tprint_success("Result manager initialized")
        
        with tprint_timer("error_handler_initialization", LogLevel.DEBUG):
            self.error_handler = UnifiedErrorHandler()
            tprint_success("Error handler initialized")
        
        with tprint_timer("logger_initialization", LogLevel.DEBUG):
            self.logger = UnifiedLogger(self.config.logging_config)
            tprint_success("Logger initialized")
        
        # Log pipeline configuration
        if self.config.enable_extensive_logging:
            tprint_structured({
                "pipeline_type": "TAS",
                "components_initialized": [
                    "data_processor", "evaluator", "training_orchestrator",
                    "result_manager", "error_handler", "logger"
                ],
                "configuration": {
                    "enable_data_processing": self.config.enable_data_processing,
                    "enable_architecture_search": self.config.enable_architecture_search,
                    "enable_model_training": self.config.enable_model_training,
                    "enable_evaluation": self.config.enable_evaluation,
                    "enable_result_storage": self.config.enable_result_storage,
                    "enable_parallel_processing": self.config.enable_parallel_processing,
                    "max_workers": self.config.max_workers
                }
            }, LogLevel.INFO)
        
        tprint_success("Unified TAS Pipeline initialized successfully")
    
    async def execute_pipeline(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        target: Optional[Union[np.ndarray, pd.Series]] = None,
        search_interface: Optional[ArchitectureSearchInterface] = None
    ) -> UnifiedArchitectureResult:
        """
        Execute complete TAS pipeline.
        
        Args:
            data: Training data
            target: Target variable
            search_interface: TAS search interface
            
        Returns:
            UnifiedArchitectureResult with complete pipeline results
        """
        tprint_info("Starting Unified TAS Pipeline execution")
        
        # Log input data information
        if self.config.enable_extensive_logging:
            data_shape = data.shape if hasattr(data, 'shape') else f"Unknown shape: {type(data)}"
            target_shape = target.shape if target is not None and hasattr(target, 'shape') else "No target"
            search_interface_type = type(search_interface).__name__ if search_interface else "No search interface"
            
            tprint_structured({
                "pipeline_execution": {
                    "pipeline_type": "TAS",
                    "data_shape": data_shape,
                    "target_shape": target_shape,
                    "search_interface": search_interface_type,
                    "timestamp": datetime.now().isoformat()
                }
            }, LogLevel.INFO)
        
        # Track execution with performance logging
        with tprint_timer("tas_pipeline_execution", LogLevel.INFO):
            try:
                # Execute training orchestration (which includes data processing, search, and training)
                tprint_debug("Executing training orchestration")
                result = await self.training_orchestrator.execute_training(
                    data, target, search_interface
                )
                
                # Log training results
                if self.config.enable_extensive_logging:
                    tprint_structured({
                        "training_results": {
                            "execution_successful": result.execution_info.status.value == "SUCCESS",
                            "duration_seconds": result.execution_info.duration_seconds,
                            "architecture_count": result.architecture_count,
                            "search_type": result.search_type,
                            "optimization_mode": result.optimization_mode
                        }
                    }, LogLevel.INFO)
                
                # Store results
                if self.config.enable_result_storage:
                    tprint_debug("Storing pipeline results")
                    with tprint_timer("result_storage", LogLevel.DEBUG):
                        self.result_manager.store_result(result)
                    tprint_success("Results stored successfully")
                else:
                    tprint_info("Result storage disabled - skipping storage")
                
                tprint_success("Unified TAS Pipeline execution completed successfully")
                return result
                
            except Exception as e:
                tprint_error(f"TAS Pipeline execution failed: {e}")
                if self.config.enable_extensive_logging:
                    tprint_structured({
                        "pipeline_error": {
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "pipeline_type": "TAS",
                            "timestamp": datetime.now().isoformat()
                        }
                    }, LogLevel.ERROR)
                raise


class UnifiedHybridPipeline:
    """
    Unified pipeline for Hybrid NAS/TAS systems.
    
    This class provides a complete pipeline that combines both
    Neural Architecture Search and Tree Architecture Search
    using the shared utilities framework.
    """
    
    def __init__(self, config: Optional[UnifiedPipelineConfig] = None):
        """Initialize unified hybrid pipeline."""
        tprint_info("Initializing Unified Hybrid NAS/TAS Pipeline")
        
        self.config = config or UnifiedPipelineConfig()
        
        # Log initialization start
        tprint_debug("Starting Hybrid pipeline component initialization")
        
        # Initialize components with detailed logging
        with tprint_timer("data_processor_initialization", LogLevel.DEBUG):
            self.data_processor = UnifiedDataProcessor(self.config.data_config)
            tprint_success("Data processor initialized")
        
        with tprint_timer("evaluator_initialization", LogLevel.DEBUG):
            self.evaluator = UnifiedEvaluator(self.config.evaluation_config)
            tprint_success("Evaluator initialized")
        
        with tprint_timer("training_orchestrator_initialization", LogLevel.DEBUG):
            self.training_orchestrator = UnifiedTrainingOrchestrator(
                self.config.training_config,
                self.config.architecture_config
            )
            tprint_success("Training orchestrator initialized")
        
        with tprint_timer("result_manager_initialization", LogLevel.DEBUG):
            self.result_manager = ResultManager(self.config.output_directory)
            tprint_success("Result manager initialized")
        
        with tprint_timer("error_handler_initialization", LogLevel.DEBUG):
            self.error_handler = UnifiedErrorHandler()
            tprint_success("Error handler initialized")
        
        with tprint_timer("logger_initialization", LogLevel.DEBUG):
            self.logger = UnifiedLogger(self.config.logging_config)
            tprint_success("Logger initialized")
        
        # Log pipeline configuration
        if self.config.enable_extensive_logging:
            tprint_structured({
                "pipeline_type": "HYBRID_NAS_TAS",
                "components_initialized": [
                    "data_processor", "evaluator", "training_orchestrator",
                    "result_manager", "error_handler", "logger"
                ],
                "configuration": {
                    "enable_data_processing": self.config.enable_data_processing,
                    "enable_architecture_search": self.config.enable_architecture_search,
                    "enable_model_training": self.config.enable_model_training,
                    "enable_evaluation": self.config.enable_evaluation,
                    "enable_result_storage": self.config.enable_result_storage,
                    "enable_parallel_processing": self.config.enable_parallel_processing,
                    "max_workers": self.config.max_workers
                }
            }, LogLevel.INFO)
        
        tprint_success("Unified Hybrid NAS/TAS Pipeline initialized successfully")
    
    async def execute_pipeline(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        target: Optional[Union[np.ndarray, pd.Series]] = None,
        nas_interface: Optional[ArchitectureSearchInterface] = None,
        tas_interface: Optional[ArchitectureSearchInterface] = None
    ) -> UnifiedArchitectureResult:
        """
        Execute complete hybrid NAS/TAS pipeline.
        
        Args:
            data: Training data
            target: Target variable
            nas_interface: NAS search interface
            tas_interface: TAS search interface
            
        Returns:
            UnifiedArchitectureResult with complete hybrid pipeline results
        """
        tprint_info("Starting Unified Hybrid NAS/TAS Pipeline execution")
        
        # Log input data information
        if self.config.enable_extensive_logging:
            data_shape = data.shape if hasattr(data, 'shape') else f"Unknown shape: {type(data)}"
            target_shape = target.shape if target is not None and hasattr(target, 'shape') else "No target"
            nas_interface_type = type(nas_interface).__name__ if nas_interface else "No NAS interface"
            tas_interface_type = type(tas_interface).__name__ if tas_interface else "No TAS interface"
            
            tprint_structured({
                "pipeline_execution": {
                    "pipeline_type": "HYBRID_NAS_TAS",
                    "data_shape": data_shape,
                    "target_shape": target_shape,
                    "nas_interface": nas_interface_type,
                    "tas_interface": tas_interface_type,
                    "timestamp": datetime.now().isoformat()
                }
            }, LogLevel.INFO)
        
        # Track execution with performance logging
        with tprint_timer("hybrid_pipeline_execution", LogLevel.INFO):
            try:
                # Execute hybrid training orchestration
                tprint_debug("Executing hybrid training orchestration")
                result = await self._execute_hybrid_training(
                    data, target, nas_interface, tas_interface
                )
                
                # Log training results
                if self.config.enable_extensive_logging:
                    tprint_structured({
                        "training_results": {
                            "execution_successful": result.execution_info.status.value == "SUCCESS",
                            "duration_seconds": result.execution_info.duration_seconds,
                            "architecture_count": result.architecture_count,
                            "search_type": result.search_type,
                            "optimization_mode": result.optimization_mode
                        }
                    }, LogLevel.INFO)
                
                # Store results
                if self.config.enable_result_storage:
                    tprint_debug("Storing hybrid pipeline results")
                    with tprint_timer("result_storage", LogLevel.DEBUG):
                        self.result_manager.store_result(result)
                    tprint_success("Results stored successfully")
                else:
                    tprint_info("Result storage disabled - skipping storage")
                
                tprint_success("Unified Hybrid NAS/TAS Pipeline execution completed successfully")
                return result
                
            except Exception as e:
                tprint_error(f"Hybrid Pipeline execution failed: {e}")
                if self.config.enable_extensive_logging:
                    tprint_structured({
                        "pipeline_error": {
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "pipeline_type": "HYBRID_NAS_TAS",
                            "timestamp": datetime.now().isoformat()
                        }
                    }, LogLevel.ERROR)
                raise
    
    async def _execute_hybrid_training(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        target: Optional[Union[np.ndarray, pd.Series]] = None,
        nas_interface: Optional[ArchitectureSearchInterface] = None,
        tas_interface: Optional[ArchitectureSearchInterface] = None
    ) -> UnifiedArchitectureResult:
        """Execute hybrid training combining NAS and TAS."""
        from .results.result_manager import ArchitectureResult
        from .config.base_config import ArchitectureType
        
        tprint_info("Starting hybrid training execution")
        
        # Collect architectures from both NAS and TAS
        all_architectures = []
        
        # NAS architectures
        if nas_interface:
            tprint_debug("Searching NAS architectures")
            with tprint_timer("nas_architecture_search", LogLevel.DEBUG):
                nas_result = await nas_interface.search(data, self.config.architecture_config)
                nas_count = len(nas_result.architectures)
                tprint_success(f"Found {nas_count} NAS architectures")
                
                for arch in nas_result.architectures:
                    arch.architecture_type = ArchitectureType.NEURAL_ONLY
                    all_architectures.append(arch)
        else:
            tprint_warning("No NAS interface provided - skipping neural architectures")
        
        # TAS architectures
        if tas_interface:
            tprint_debug("Searching TAS architectures")
            with tprint_timer("tas_architecture_search", LogLevel.DEBUG):
                tas_result = await tas_interface.search(data, self.config.architecture_config)
                tas_count = len(tas_result.architectures)
                tprint_success(f"Found {tas_count} TAS architectures")
                
                for arch in tas_result.architectures:
                    arch.architecture_type = ArchitectureType.TREE_ONLY
                    all_architectures.append(arch)
        else:
            tprint_warning("No TAS interface provided - skipping tree architectures")
        
        # Create hybrid architectures
        tprint_debug("Creating hybrid architectures")
        with tprint_timer("hybrid_architecture_creation", LogLevel.DEBUG):
            hybrid_architectures = self._create_hybrid_architectures(all_architectures)
            hybrid_count = len(hybrid_architectures)
            tprint_success(f"Created {hybrid_count} hybrid architectures")
            all_architectures.extend(hybrid_architectures)
        
        # Log architecture summary
        if self.config.enable_extensive_logging:
            tprint_structured({
                "architecture_summary": {
                    "total_architectures": len(all_architectures),
                    "nas_architectures": len([a for a in all_architectures if a.architecture_type == ArchitectureType.NEURAL_ONLY]),
                    "tas_architectures": len([a for a in all_architectures if a.architecture_type == ArchitectureType.TREE_ONLY]),
                    "hybrid_architectures": len([a for a in all_architectures if a.architecture_type == ArchitectureType.HYBRID_NEURAL_TREE])
                }
            }, LogLevel.INFO)
        
        # Execute unified training
        tprint_debug("Executing unified training with all architectures")
        with tprint_timer("unified_training_execution", LogLevel.INFO):
            result = await self.training_orchestrator.execute_training(
                data, target, None  # No single interface, we have all architectures
            )
        
        # Override with our collected architectures
        result.all_architectures = all_architectures
        result.architecture_count = len(all_architectures)
        result.search_type = "hybrid"
        
        tprint_success("Hybrid training execution completed")
        return result
    
    def _create_hybrid_architectures(
        self,
        architectures: List[ArchitectureResult]
    ) -> List[ArchitectureResult]:
        """Create hybrid architectures combining neural and tree components."""
        from .results.result_manager import ArchitectureResult
        from .config.base_config import ArchitectureType
        
        tprint_debug("Creating hybrid architectures from neural and tree components")
        
        hybrid_architectures = []
        
        # Find neural and tree architectures
        neural_archs = [arch for arch in architectures if arch.architecture_type == ArchitectureType.NEURAL_ONLY]
        tree_archs = [arch for arch in architectures if arch.architecture_type == ArchitectureType.TREE_ONLY]
        
        tprint_info(f"Found {len(neural_archs)} neural architectures and {len(tree_archs)} tree architectures")
        
        if not neural_archs or not tree_archs:
            tprint_warning("Cannot create hybrid architectures - missing neural or tree architectures")
            return hybrid_architectures
        
        # Create hybrid combinations
        combination_count = 0
        max_combinations = min(3, len(neural_archs), len(tree_archs))  # Limit combinations
        
        for i, neural_arch in enumerate(neural_archs[:max_combinations]):
            for j, tree_arch in enumerate(tree_archs[:max_combinations]):
                tprint_debug(f"Creating hybrid architecture {combination_count + 1}: Neural[{i}] + Tree[{j}]")
                
                hybrid_arch = ArchitectureResult(
                    architecture_type=ArchitectureType.HYBRID_NEURAL_TREE,
                    architecture_config={
                        "neural_config": neural_arch.architecture_config,
                        "tree_config": tree_arch.architecture_config,
                        "ensemble_method": "weighted_average",
                        "neural_weight": 0.6,
                        "tree_weight": 0.4,
                        "combination_id": f"hybrid_{i}_{j}"
                    }
                )
                hybrid_architectures.append(hybrid_arch)
                combination_count += 1
        
        tprint_success(f"Created {len(hybrid_architectures)} hybrid architectures")
        
        if self.config.enable_extensive_logging:
            tprint_structured({
                "hybrid_creation_summary": {
                    "neural_architectures_used": len(neural_archs[:max_combinations]),
                    "tree_architectures_used": len(tree_archs[:max_combinations]),
                    "hybrid_architectures_created": len(hybrid_architectures),
                    "max_combinations": max_combinations
                }
            }, LogLevel.INFO)
        
        return hybrid_architectures


# Convenience functions for quick pipeline creation
def create_nas_pipeline(config: Optional[UnifiedPipelineConfig] = None) -> UnifiedNASPipeline:
    """Create unified NAS pipeline."""
    tprint_info("Creating NAS pipeline")
    pipeline = UnifiedNASPipeline(config)
    tprint_success("NAS pipeline created successfully")
    return pipeline


def create_tas_pipeline(config: Optional[UnifiedPipelineConfig] = None) -> UnifiedTASPipeline:
    """Create unified TAS pipeline."""
    tprint_info("Creating TAS pipeline")
    pipeline = UnifiedTASPipeline(config)
    tprint_success("TAS pipeline created successfully")
    return pipeline


def create_hybrid_pipeline(config: Optional[UnifiedPipelineConfig] = None) -> UnifiedHybridPipeline:
    """Create unified hybrid pipeline."""
    tprint_info("Creating Hybrid NAS/TAS pipeline")
    pipeline = UnifiedHybridPipeline(config)
    tprint_success("Hybrid NAS/TAS pipeline created successfully")
    return pipeline


async def run_quick_example():
    """Run a quick example of the unified pipeline."""
    tprint_info("Starting quick example of unified NAS/TAS pipeline")
    
    # Create sample data
    tprint_debug("Generating sample data")
    np.random.seed(42)
    X = np.random.randn(1000, 20)
    y = np.random.randint(0, 2, 1000)
    
    tprint_success(f"Generated sample data: {X.shape[0]} samples, {X.shape[1]} features")
    tprint_info(f"Target distribution: {len(np.unique(y))} classes")
    
    # Create pipeline
    tprint_debug("Creating hybrid pipeline")
    pipeline = create_hybrid_pipeline()
    
    # Execute pipeline
    tprint_info("Executing hybrid pipeline")
    with tprint_timer("quick_example_execution", LogLevel.INFO):
        result = await pipeline.execute_pipeline(X, y)
    
    # Log results
    tprint_success("Quick example completed successfully")
    tprint_structured({
        "example_results": {
            "pipeline_status": result.execution_info.status.value,
            "architectures_found": result.architecture_count,
            "execution_time_seconds": result.execution_info.duration_seconds,
            "search_type": result.search_type,
            "optimization_mode": result.optimization_mode
        }
    }, LogLevel.INFO)
    
    tprint_info(f"Pipeline completed: {result.execution_info.status}")
    tprint_info(f"Architectures found: {result.architecture_count}")
    tprint_performance("Quick example execution", result.execution_info.duration_seconds)
    
    return result


if __name__ == "__main__":
    # Run example
    asyncio.run(run_quick_example())