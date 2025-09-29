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
    tprint_success, tprint_progress, tprint_performance, tprint_timer
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


class UnifiedNASPipeline:
    """
    Unified pipeline for Neural Architecture Search.
    
    This class provides a complete pipeline for NAS implementations
    using the shared utilities framework.
    """
    
    def __init__(self, config: Optional[UnifiedPipelineConfig] = None):
        """Initialize unified NAS pipeline."""
        self.config = config or UnifiedPipelineConfig()
        
        # Initialize components
        self.data_processor = UnifiedDataProcessor(self.config.data_config)
        self.evaluator = UnifiedEvaluator(self.config.evaluation_config)
        self.training_orchestrator = UnifiedTrainingOrchestrator(
            self.config.training_config,
            self.config.architecture_config
        )
        self.result_manager = ResultManager(self.config.output_directory)
        self.error_handler = UnifiedErrorHandler()
        self.logger = UnifiedLogger(self.config.logging_config)
        
        tprint_success("Unified NAS Pipeline initialized")
    
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
        
        with self.logger.log_execution_time("nas_pipeline", "unified_pipeline"):
            # Execute training orchestration (which includes data processing, search, and training)
            result = await self.training_orchestrator.execute_training(
                data, target, search_interface
            )
            
            # Store results
            if self.config.enable_result_storage:
                self.result_manager.store_result(result)
            
            tprint_success("Unified NAS Pipeline execution completed")
            return result


class UnifiedTASPipeline:
    """
    Unified pipeline for Tree Architecture Search.
    
    This class provides a complete pipeline for TAS implementations
    using the shared utilities framework.
    """
    
    def __init__(self, config: Optional[UnifiedPipelineConfig] = None):
        """Initialize unified TAS pipeline."""
        self.config = config or UnifiedPipelineConfig()
        
        # Initialize components
        self.data_processor = UnifiedDataProcessor(self.config.data_config)
        self.evaluator = UnifiedEvaluator(self.config.evaluation_config)
        self.training_orchestrator = UnifiedTrainingOrchestrator(
            self.config.training_config,
            self.config.architecture_config
        )
        self.result_manager = ResultManager(self.config.output_directory)
        self.error_handler = UnifiedErrorHandler()
        self.logger = UnifiedLogger(self.config.logging_config)
        
        tprint_success("Unified TAS Pipeline initialized")
    
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
        
        with self.logger.log_execution_time("tas_pipeline", "unified_pipeline"):
            # Execute training orchestration (which includes data processing, search, and training)
            result = await self.training_orchestrator.execute_training(
                data, target, search_interface
            )
            
            # Store results
            if self.config.enable_result_storage:
                self.result_manager.store_result(result)
            
            tprint_success("Unified TAS Pipeline execution completed")
            return result


class UnifiedHybridPipeline:
    """
    Unified pipeline for Hybrid NAS/TAS systems.
    
    This class provides a complete pipeline that combines both
    Neural Architecture Search and Tree Architecture Search
    using the shared utilities framework.
    """
    
    def __init__(self, config: Optional[UnifiedPipelineConfig] = None):
        """Initialize unified hybrid pipeline."""
        self.config = config or UnifiedPipelineConfig()
        
        # Initialize components
        self.data_processor = UnifiedDataProcessor(self.config.data_config)
        self.evaluator = UnifiedEvaluator(self.config.evaluation_config)
        self.training_orchestrator = UnifiedTrainingOrchestrator(
            self.config.training_config,
            self.config.architecture_config
        )
        self.result_manager = ResultManager(self.config.output_directory)
        self.error_handler = UnifiedErrorHandler()
        self.logger = UnifiedLogger(self.config.logging_config)
        
        tprint_success("Unified Hybrid Pipeline initialized")
    
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
        tprint_info("Starting Unified Hybrid Pipeline execution")
        
        with self.logger.log_execution_time("hybrid_pipeline", "unified_pipeline"):
            # Execute hybrid training orchestration
            result = await self._execute_hybrid_training(
                data, target, nas_interface, tas_interface
            )
            
            # Store results
            if self.config.enable_result_storage:
                self.result_manager.store_result(result)
            
            tprint_success("Unified Hybrid Pipeline execution completed")
            return result
    
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
        
        # Collect architectures from both NAS and TAS
        all_architectures = []
        
        # NAS architectures
        if nas_interface:
            nas_result = await nas_interface.search(data, self.config.architecture_config)
            for arch in nas_result.architectures:
                arch.architecture_type = ArchitectureType.NEURAL
                all_architectures.append(arch)
        
        # TAS architectures
        if tas_interface:
            tas_result = await tas_interface.search(data, self.config.architecture_config)
            for arch in tas_result.architectures:
                arch.architecture_type = ArchitectureType.TREE
                all_architectures.append(arch)
        
        # Create hybrid architectures
        hybrid_architectures = self._create_hybrid_architectures(all_architectures)
        all_architectures.extend(hybrid_architectures)
        
        # Execute unified training
        result = await self.training_orchestrator.execute_training(
            data, target, None  # No single interface, we have all architectures
        )
        
        # Override with our collected architectures
        result.all_architectures = all_architectures
        result.architecture_count = len(all_architectures)
        result.search_type = "hybrid"
        
        return result
    
    def _create_hybrid_architectures(
        self,
        architectures: List
    ) -> List:
        """Create hybrid architectures combining neural and tree components."""
        from .results.result_manager import ArchitectureResult
        from .config.base_config import ArchitectureType
        
        hybrid_architectures = []
        
        # Find neural and tree architectures
        neural_archs = [arch for arch in architectures if arch.architecture_type == ArchitectureType.NEURAL]
        tree_archs = [arch for arch in architectures if arch.architecture_type == ArchitectureType.TREE]
        
        # Create hybrid combinations
        for neural_arch in neural_archs[:3]:  # Limit combinations
            for tree_arch in tree_archs[:3]:
                hybrid_arch = ArchitectureResult(
                    architecture_type=ArchitectureType.HYBRID_NEURAL_TREE,
                    architecture_config={
                        "neural_config": neural_arch.architecture_config,
                        "tree_config": tree_arch.architecture_config,
                        "ensemble_method": "weighted_average",
                        "neural_weight": 0.6,
                        "tree_weight": 0.4
                    }
                )
                hybrid_architectures.append(hybrid_arch)
        
        return hybrid_architectures


# Convenience functions for quick pipeline creation
def create_nas_pipeline(config: Optional[UnifiedPipelineConfig] = None) -> UnifiedNASPipeline:
    """Create unified NAS pipeline."""
    return UnifiedNASPipeline(config)


def create_tas_pipeline(config: Optional[UnifiedPipelineConfig] = None) -> UnifiedTASPipeline:
    """Create unified TAS pipeline."""
    return UnifiedTASPipeline(config)


def create_hybrid_pipeline(config: Optional[UnifiedPipelineConfig] = None) -> UnifiedHybridPipeline:
    """Create unified hybrid pipeline."""
    return UnifiedHybridPipeline(config)


async def run_quick_example():
    """Run a quick example of the unified pipeline."""
    # Create sample data
    np.random.seed(42)
    X = np.random.randn(1000, 20)
    y = np.random.randint(0, 2, 1000)
    
    # Create pipeline
    pipeline = create_hybrid_pipeline()
    
    # Execute pipeline
    result = await pipeline.execute_pipeline(X, y)
    
    print(f"Pipeline completed: {result.execution_info.status}")
    print(f"Architectures found: {result.architecture_count}")
    print(f"Execution time: {result.execution_info.duration_seconds:.2f}s")
    
    return result


if __name__ == "__main__":
    # Run example
    asyncio.run(run_quick_example())