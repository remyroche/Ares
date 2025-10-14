"""
Refactored Unified Data-Driven Pipeline

This is the refactored version of the unified pipeline that uses modular stages
and simplified configuration presets.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import logging
import time
from pathlib import Path
from datetime import datetime

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)

# Import simplified configuration
from .core.simplified_config import (
    create_full_config, create_blank_config, create_light_config,
    create_config_by_intensity, PipelineIntensity
)

# Import pipeline stages
from .stages.data_validation_stage import (
    DataValidationStage, DataValidationResult, create_data_validation_stage
)
from .stages.feature_generation_stage import (
    FeatureGenerationStage, FeatureGenerationResult, create_feature_generation_stage
)
from .stages.feature_selection_stage import (
    FeatureSelectionStage, FeatureSelectionStageResult, create_feature_selection_stage
)
from .stages.optimization_stage import (
    OptimizationStage, OptimizationStageResult, create_optimization_stage
)

# Import enhanced components for reporting
from .enhanced_components.detailed_pipeline_reporter import (
    DetailedPipelineReporter, DetailedPipelineReport
)


@dataclass
class RefactoredPipelineResult:
    """Result from the refactored unified pipeline."""
    
    # Core results
    processed_data: pd.DataFrame
    selected_features: List[str]
    feature_metadata: Dict[str, Any]
    
    # Stage results
    validation_result: Optional[DataValidationResult] = None
    generation_result: Optional[FeatureGenerationResult] = None
    selection_result: Optional[FeatureSelectionStageResult] = None
    optimization_result: Optional[OptimizationStageResult] = None
    
    # Performance metrics
    total_processing_time: float = 0.0
    memory_usage: float = 0.0
    quality_score: float = 0.0
    
    # Pipeline metadata
    pipeline_metadata: Dict[str, Any] = None
    warnings: List[str] = None
    errors: List[str] = None
    
    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.pipeline_metadata is None:
            self.pipeline_metadata = {}
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the pipeline results."""
        return {
            'total_processing_time': self.total_processing_time,
            'memory_usage_mb': self.memory_usage,
            'quality_score': self.quality_score,
            'selected_features_count': len(self.selected_features),
            'processed_data_shape': self.processed_data.shape,
            'warnings_count': len(self.warnings),
            'errors_count': len(self.errors),
            'pipeline_metadata': self.pipeline_metadata
        }
    
    def save_result(self, output_path: Union[str, Path]) -> bool:
        """Save pipeline results to files.
        
        Args:
            output_path: Path to save results
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save processed data
            self.processed_data.to_csv(output_path / "processed_data.csv")
            
            # Save selected features
            pd.DataFrame({'feature': self.selected_features}).to_csv(
                output_path / "selected_features.csv", index=False
            )
            
            # Save metadata
            import json
            with open(output_path / "metadata.json", 'w') as f:
                json.dump(self.pipeline_metadata, f, indent=2, default=str)
            
            tprint_success(f"✅ Results saved to {output_path}")
            return True
            
        except Exception as e:
            tprint_error(f"❌ Failed to save results: {e}")
            return False


class RefactoredUnifiedPipeline:
    """
    Refactored Unified Data-Driven Pipeline.
    
    This is the refactored version that uses modular stages and simplified
    configuration presets for better maintainability and usability.
    """
    
    def __init__(self, 
                 config: Optional[Any] = None,
                 intensity: str = "full",
                 custom_overrides: Optional[Dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None):
        """Initialize the refactored unified pipeline.
        
        Args:
            config: Pipeline configuration (uses simplified config if None)
            intensity: Pipeline intensity ("full", "blank", "light")
            custom_overrides: Custom configuration overrides
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize configuration
        if config is None:
            self.config = create_config_by_intensity(intensity, custom_overrides)
        else:
            self.config = config
        
        # Initialize pipeline stages
        self.validation_stage = create_data_validation_stage(self.config, self.logger)
        self.generation_stage = create_feature_generation_stage(self.config, self.logger)
        self.selection_stage = create_feature_selection_stage(self.config, self.logger)
        self.optimization_stage = create_optimization_stage(self.config, self.logger)
        
        # Initialize reporting
        self.detailed_reporter = DetailedPipelineReporter(outcomes_dir="outcomes")
        
        tprint_info("🚀 Refactored Unified Data-Driven Pipeline initialized")
        tprint_info(f"📊 Intensity: {intensity}")
        tprint_info(f"📊 Configuration: {type(self.config).__name__}")
    
    async def process(self, 
                     data: pd.DataFrame, 
                     targets: Optional[pd.Series] = None,
                     feature_columns: Optional[List[str]] = None,
                     timeframe: str = "15m",
                     pipeline_state: Optional[Dict[str, Any]] = None) -> RefactoredPipelineResult:
        """Process data through the refactored unified pipeline.
        
        Args:
            data: Input data with OHLCV columns
            targets: Optional target series for supervised learning
            feature_columns: Optional list of feature columns to use
            timeframe: Target timeframe (e.g., "15m", "5m", "1h")
            pipeline_state: Optional pipeline state dictionary
            
        Returns:
            RefactoredPipelineResult with comprehensive results
        """
        tprint_info("🚀 Starting refactored unified pipeline processing")
        tprint_info(f"📊 Data shape: {data.shape}, timeframe: {timeframe}")
        
        # Start performance monitoring
        start_time = time.time()
        warnings = []
        errors = []
        
        try:
            # Stage 1: Data Validation
            tprint_info("Stage 1: Data Validation")
            validation_result = self.validation_stage.validate_dataframe_quality(
                data, context=f"pipeline_input_{timeframe}"
            )
            
            if not validation_result.is_valid:
                error_msg = f"Data validation failed: {validation_result.issues}"
                tprint_error(f"❌ {error_msg}")
                return self._create_error_result(start_time, error_msg, warnings, errors)
            
            # Stage 2: Feature Generation
            tprint_info("Stage 2: Feature Generation")
            generation_result = self.generation_stage.generate_features(
                data, targets=targets, feature_columns=feature_columns, timeframe=timeframe
            )
            
            if generation_result.errors:
                error_msg = f"Feature generation failed: {generation_result.errors}"
                tprint_error(f"❌ {error_msg}")
                return self._create_error_result(start_time, error_msg, warnings, errors)
            
            # Stage 3: Feature Selection
            tprint_info("Stage 3: Feature Selection")
            selection_result = self.selection_stage.select_features(
                generation_result.generated_features, 
                targets=targets, 
                feature_columns=feature_columns, 
                timeframe=timeframe
            )
            
            if selection_result.errors:
                error_msg = f"Feature selection failed: {selection_result.errors}"
                tprint_error(f"❌ {error_msg}")
                return self._create_error_result(start_time, error_msg, warnings, errors)
            
            # Stage 4: Optimization
            tprint_info("Stage 4: Optimization")
            optimization_result = self.optimization_stage.optimize_features(
                generation_result.generated_features,
                targets=targets,
                feature_columns=selection_result.selected_features,
                timeframe=timeframe
            )
            
            if optimization_result.errors:
                error_msg = f"Optimization failed: {optimization_result.errors}"
                tprint_error(f"❌ {error_msg}")
                return self._create_error_result(start_time, error_msg, warnings, errors)
            
            # Create final result
            total_time = time.time() - start_time
            memory_usage = generation_result.generated_features.memory_usage(deep=True).sum() / 1024 / 1024
            
            # Calculate overall quality score
            quality_score = self._calculate_overall_quality(
                validation_result, generation_result, selection_result, optimization_result
            )
            
            result = RefactoredPipelineResult(
                processed_data=generation_result.generated_features,
                selected_features=selection_result.selected_features,
                feature_metadata={
                    'validation_quality': validation_result.quality_score,
                    'generation_quality': generation_result.quality_score,
                    'selection_quality': selection_result.quality_score,
                    'optimization_quality': optimization_result.quality_score,
                    'timeframe': timeframe,
                    'pipeline_state': pipeline_state or {}
                },
                validation_result=validation_result,
                generation_result=generation_result,
                selection_result=selection_result,
                optimization_result=optimization_result,
                total_processing_time=total_time,
                memory_usage=memory_usage,
                quality_score=quality_score,
                pipeline_metadata={
                    'intensity': getattr(self.config, 'intensity', 'unknown'),
                    'timeframe': timeframe,
                    'processing_time': total_time,
                    'memory_usage_mb': memory_usage
                },
                warnings=warnings,
                errors=errors
            )
            
            tprint_success(f"✅ Pipeline processing completed in {total_time:.2f}s")
            tprint_info(f"📊 Memory usage: {memory_usage:.2f} MB")
            tprint_info(f"📈 Overall quality score: {quality_score:.3f}")
            tprint_info(f"🎯 Selected features: {len(selection_result.selected_features)}")
            
            return result
            
        except Exception as e:
            total_time = time.time() - start_time
            error_msg = f"Pipeline processing failed: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            return self._create_error_result(start_time, error_msg, warnings, errors)
    
    def _create_error_result(self, 
                           start_time: float, 
                           error_msg: str, 
                           warnings: List[str], 
                           errors: List[str]) -> RefactoredPipelineResult:
        """Create an error result when pipeline fails.
        
        Args:
            start_time: Start time of processing
            error_msg: Error message
            warnings: List of warnings
            errors: List of errors
            
        Returns:
            RefactoredPipelineResult with error information
        """
        errors.append(error_msg)
        total_time = time.time() - start_time
        
        return RefactoredPipelineResult(
            processed_data=pd.DataFrame(),
            selected_features=[],
            feature_metadata={},
            total_processing_time=total_time,
            memory_usage=0.0,
            quality_score=0.0,
            pipeline_metadata={'error': error_msg},
            warnings=warnings,
            errors=errors
        )
    
    def _calculate_overall_quality(self, 
                                 validation_result: DataValidationResult,
                                 generation_result: FeatureGenerationResult,
                                 selection_result: FeatureSelectionStageResult,
                                 optimization_result: OptimizationStageResult) -> float:
        """Calculate overall quality score from stage results.
        
        Args:
            validation_result: Data validation result
            generation_result: Feature generation result
            selection_result: Feature selection result
            optimization_result: Optimization result
            
        Returns:
            Overall quality score between 0 and 1
        """
        try:
            # Weighted average of stage quality scores
            weights = [0.2, 0.3, 0.3, 0.2]  # validation, generation, selection, optimization
            scores = [
                validation_result.quality_score,
                generation_result.quality_score,
                selection_result.quality_score,
                optimization_result.quality_score
            ]
            
            overall_score = sum(w * s for w, s in zip(weights, scores))
            return max(0.0, min(1.0, overall_score))
            
        except Exception as e:
            tprint_warning(f"⚠️ Quality score calculation failed: {e}")
            return 0.0
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get a summary of the pipeline configuration and status.
        
        Returns:
            Dictionary with pipeline summary
        """
        return {
            'pipeline_type': 'RefactoredUnifiedPipeline',
            'config_type': type(self.config).__name__,
            'stages': [
                'data_validation',
                'feature_generation', 
                'feature_selection',
                'optimization'
            ],
            'intensity': getattr(self.config, 'intensity', 'unknown'),
            'logger': self.logger.name if self.logger else 'default'
        }
    
    def cleanup(self) -> None:
        """Clean up pipeline resources."""
        tprint_info("🧹 Starting pipeline cleanup process")
        
        try:
            # Clean up stages
            self.validation_stage.cleanup()
            self.generation_stage.cleanup()
            self.selection_stage.cleanup()
            self.optimization_stage.cleanup()
            
            tprint_success("✅ Pipeline cleanup completed")
            
        except Exception as e:
            tprint_warning(f"⚠️ Error during cleanup: {e}")


# Convenience functions
def create_refactored_pipeline(intensity: str = "full", 
                             custom_overrides: Optional[Dict[str, Any]] = None,
                             logger: Optional[logging.Logger] = None) -> RefactoredUnifiedPipeline:
    """Create a refactored unified pipeline instance.
    
    Args:
        intensity: Pipeline intensity ("full", "blank", "light")
        custom_overrides: Custom configuration overrides
        logger: Optional logger instance
        
    Returns:
        RefactoredUnifiedPipeline instance
    """
    return RefactoredUnifiedPipeline(
        config=None,
        intensity=intensity,
        custom_overrides=custom_overrides,
        logger=logger
    )


def create_full_pipeline(custom_overrides: Optional[Dict[str, Any]] = None,
                        logger: Optional[logging.Logger] = None) -> RefactoredUnifiedPipeline:
    """Create a full intensity pipeline.
    
    Args:
        custom_overrides: Custom configuration overrides
        logger: Optional logger instance
        
    Returns:
        RefactoredUnifiedPipeline instance
    """
    return create_refactored_pipeline("full", custom_overrides, logger)


def create_blank_pipeline(custom_overrides: Optional[Dict[str, Any]] = None,
                         logger: Optional[logging.Logger] = None) -> RefactoredUnifiedPipeline:
    """Create a blank intensity pipeline (25% intensity).
    
    Args:
        custom_overrides: Custom configuration overrides
        logger: Optional logger instance
        
    Returns:
        RefactoredUnifiedPipeline instance
    """
    return create_refactored_pipeline("blank", custom_overrides, logger)


def create_light_pipeline(custom_overrides: Optional[Dict[str, Any]] = None,
                         logger: Optional[logging.Logger] = None) -> RefactoredUnifiedPipeline:
    """Create a light intensity pipeline (10% intensity).
    
    Args:
        custom_overrides: Custom configuration overrides
        logger: Optional logger instance
        
    Returns:
        RefactoredUnifiedPipeline instance
    """
    return create_refactored_pipeline("light", custom_overrides, logger)


# Export main classes and functions
__all__ = [
    'RefactoredUnifiedPipeline',
    'RefactoredPipelineResult',
    'create_refactored_pipeline',
    'create_full_pipeline',
    'create_blank_pipeline',
    'create_light_pipeline'
]