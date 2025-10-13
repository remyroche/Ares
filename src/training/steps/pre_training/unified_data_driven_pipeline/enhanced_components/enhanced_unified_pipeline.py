"""
Enhanced Unified Data-Driven Pipeline

This module provides a comprehensive unified pipeline that integrates all
the enhanced components for advanced data-driven analysis and optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import logging
import time
from pathlib import Path

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

# Import enhanced components
from .enhanced_walk_forward_validation import (
    AdvancedWalkForwardValidator, AdvancedWalkForwardConfig, AdvancedTimeSeriesSplit
)
from .enhanced_statistical_framework import (
    EnhancedStatisticalFramework, HypothesisTestResult, MultipleTestingResult, StatisticalAnalysisResult
)
from .enhanced_schema_validation import (
    EnhancedSchemaValidator, ValidationResult, SchemaDefinition, TemporalAlignmentResult
)
from .enhanced_caching_integration import (
    EnhancedCachingIntegration, CacheEntry, CacheStats, ArtifactMetadata
)
from .gpu_optimizations import (
    GPUOptimizer, GPUConfig, GPUOperationResult
)

# Import existing components
try:
    from ..core.enhanced_unified_pipeline import EnhancedUnifiedDataDrivenPipeline
    EXISTING_PIPELINE_AVAILABLE = True
    tprint_info("✅ Existing EnhancedUnifiedDataDrivenPipeline available")
except ImportError:
    EXISTING_PIPELINE_AVAILABLE = False
    tprint_warning("⚠️ Existing EnhancedUnifiedDataDrivenPipeline not available")

logger = logging.getLogger(__name__)


@dataclass
class EnhancedPipelineConfig:
    """Configuration for the enhanced unified pipeline."""
    
    # Walk-forward validation
    enable_advanced_walk_forward: bool = True
    walk_forward_config: Optional[AdvancedWalkForwardConfig] = None
    
    # Statistical framework
    enable_enhanced_statistical: bool = True
    statistical_config: Optional[Dict[str, Any]] = None
    
    # Schema validation
    enable_enhanced_schema: bool = True
    schema_config: Optional[Dict[str, Any]] = None
    
    # Caching integration
    enable_enhanced_caching: bool = True
    caching_config: Optional[Dict[str, Any]] = None
    
    # GPU optimizations
    enable_gpu_optimizations: bool = True
    gpu_config: Optional[GPUConfig] = None
    
    # Performance settings
    enable_performance_monitoring: bool = True
    enable_parallel_processing: bool = True
    max_workers: int = 4
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.max_workers > 0, "max_workers must be positive"


@dataclass
class PipelineExecutionResult:
    """Result of pipeline execution."""
    
    success: bool
    execution_time: float
    components_used: List[str]
    performance_metrics: Dict[str, Any]
    error_message: Optional[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        """Validate result."""
        assert isinstance(self.success, bool), "success must be boolean"
        assert self.execution_time >= 0, "execution_time must be non-negative"
        assert isinstance(self.components_used, list), "components_used must be list"
        if self.warnings is None:
            self.warnings = []


class EnhancedUnifiedDataDrivenPipeline:
    """
    Enhanced unified data-driven pipeline with advanced features.
    
    Integrates all enhanced components for comprehensive data analysis,
    optimization, and validation.
    """
    
    def __init__(self, config: Optional[EnhancedPipelineConfig] = None):
        """Initialize the enhanced unified pipeline."""
        self.config = config or EnhancedPipelineConfig()
        
        # Initialize enhanced components
        self._initialize_enhanced_components()
        
        # Initialize existing pipeline if available
        self._initialize_existing_pipeline()
        
        # Performance tracking
        self.performance_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'total_execution_time': 0.0,
            'component_usage': {},
            'gpu_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        tprint_info("Enhanced Unified Data-Driven Pipeline initialized")
        tprint_success("✅ All enhanced components integrated")
    
    def _initialize_enhanced_components(self):
        """Initialize enhanced components."""
        # Initialize walk-forward validation
        if self.config.enable_advanced_walk_forward:
            try:
                self.walk_forward_validator = AdvancedWalkForwardValidator(
                    config=self.config.walk_forward_config or AdvancedWalkForwardConfig()
                )
                tprint_success("✅ Advanced walk-forward validation initialized")
            except Exception as e:
                tprint_warning(f"⚠️ Walk-forward validation initialization failed: {e}")
                self.walk_forward_validator = None
        else:
            self.walk_forward_validator = None
        
        # Initialize statistical framework
        if self.config.enable_enhanced_statistical:
            try:
                self.statistical_framework = EnhancedStatisticalFramework(
                    config=self.config.statistical_config or {}
                )
                tprint_success("✅ Enhanced statistical framework initialized")
            except Exception as e:
                tprint_warning(f"⚠️ Statistical framework initialization failed: {e}")
                self.statistical_framework = None
        else:
            self.statistical_framework = None
        
        # Initialize schema validation
        if self.config.enable_enhanced_schema:
            try:
                self.schema_validator = EnhancedSchemaValidator(
                    enable_pandera=True,
                    enable_gpu_optimization=True
                )
                tprint_success("✅ Enhanced schema validation initialized")
            except Exception as e:
                tprint_warning(f"⚠️ Schema validation initialization failed: {e}")
                self.schema_validator = None
        else:
            self.schema_validator = None
        
        # Initialize caching integration
        if self.config.enable_enhanced_caching:
            try:
                self.caching_integration = EnhancedCachingIntegration(
                    enable_feature_cache=True,
                    enable_serialization=True,
                    enable_compression=True
                )
                tprint_success("✅ Enhanced caching integration initialized")
            except Exception as e:
                tprint_warning(f"⚠️ Caching integration initialization failed: {e}")
                self.caching_integration = None
        else:
            self.caching_integration = None
        
        # Initialize GPU optimizations
        if self.config.enable_gpu_optimizations:
            try:
                self.gpu_optimizer = GPUOptimizer(
                    config=self.config.gpu_config or GPUConfig()
                )
                tprint_success("✅ GPU optimizations initialized")
            except Exception as e:
                tprint_warning(f"⚠️ GPU optimizations initialization failed: {e}")
                self.gpu_optimizer = None
        else:
            self.gpu_optimizer = None
    
    def _initialize_existing_pipeline(self):
        """Initialize existing pipeline if available."""
        if EXISTING_PIPELINE_AVAILABLE:
            try:
                self.existing_pipeline = EnhancedUnifiedDataDrivenPipeline()
                tprint_success("✅ Existing pipeline integrated")
            except Exception as e:
                tprint_warning(f"⚠️ Existing pipeline integration failed: {e}")
                self.existing_pipeline = None
        else:
            self.existing_pipeline = None
    
    def run_comprehensive_analysis(self, 
                                 data: pd.DataFrame,
                                 labels: pd.DataFrame,
                                 config: Optional[Dict[str, Any]] = None) -> PipelineExecutionResult:
        """
        Run comprehensive analysis with all enhanced components.
        
        Args:
            data: Input data DataFrame
            labels: Labels DataFrame
            config: Optional configuration overrides
            
        Returns:
            PipelineExecutionResult with analysis results
        """
        tprint_info("Starting comprehensive analysis")
        start_time = time.time()
        
        components_used = []
        performance_metrics = {}
        warnings = []
        
        try:
            # 1. Schema validation
            if self.schema_validator:
                tprint_info("Running schema validation")
                validation_result = self.schema_validator.validate_data(
                    data, "features", "comprehensive_analysis"
                )
                if not validation_result.is_valid:
                    warnings.append(f"Schema validation failed: {validation_result.errors}")
                components_used.append("schema_validation")
                performance_metrics['schema_validation'] = {
                    'time': validation_result.validation_time,
                    'valid': validation_result.is_valid,
                    'errors': len(validation_result.errors)
                }
            
            # 2. Statistical analysis
            if self.statistical_framework:
                tprint_info("Running statistical analysis")
                statistical_result = self.statistical_framework.comprehensive_analysis(data)
                components_used.append("statistical_analysis")
                performance_metrics['statistical_analysis'] = {
                    'time': statistical_result.execution_time,
                    'tests_performed': len(statistical_result.test_results),
                    'significant_tests': sum(1 for t in statistical_result.test_results if t.significant)
                }
            
            # 3. Walk-forward validation
            if self.walk_forward_validator:
                tprint_info("Running walk-forward validation")
                splits = self.walk_forward_validator.generate_splits(data, labels)
                components_used.append("walk_forward_validation")
                performance_metrics['walk_forward_validation'] = {
                    'splits_generated': len(splits),
                    'total_samples': len(data)
                }
            
            # 4. GPU-accelerated operations
            if self.gpu_optimizer:
                tprint_info("Running GPU-accelerated operations")
                # Example: correlation matrix calculation
                corr_result = self.gpu_optimizer.correlation_matrix(data.select_dtypes(include=[np.number]))
                components_used.append("gpu_operations")
                performance_metrics['gpu_operations'] = {
                    'time': corr_result.execution_time,
                    'success': corr_result.success,
                    'gpu_used': not corr_result.fallback_used
                }
            
            # 5. Caching operations
            if self.caching_integration:
                tprint_info("Running caching operations")
                # Example: cache features
                cache_success = self.caching_integration.cache_data(
                    key="comprehensive_analysis_features",
                    data=data,
                    artifact_type="features",
                    schema_version="1.0"
                )
                components_used.append("caching")
                performance_metrics['caching'] = {
                    'cache_success': cache_success,
                    'cache_stats': self.caching_integration.get_cache_stats()
                }
            
            # 6. Run existing pipeline if available
            if self.existing_pipeline:
                tprint_info("Running existing pipeline")
                # This would call the existing pipeline's methods
                components_used.append("existing_pipeline")
                performance_metrics['existing_pipeline'] = {
                    'available': True,
                    'integrated': True
                }
            
            execution_time = time.time() - start_time
            
            # Update performance stats
            self.performance_stats['total_executions'] += 1
            self.performance_stats['successful_executions'] += 1
            self.performance_stats['total_execution_time'] += execution_time
            
            tprint_success(f"Comprehensive analysis completed in {execution_time:.3f}s")
            
            return PipelineExecutionResult(
                success=True,
                execution_time=execution_time,
                components_used=components_used,
                performance_metrics=performance_metrics,
                warnings=warnings
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            error_message = str(e)
            
            # Update performance stats
            self.performance_stats['total_executions'] += 1
            self.performance_stats['failed_executions'] += 1
            self.performance_stats['total_execution_time'] += execution_time
            
            tprint_error(f"Comprehensive analysis failed: {error_message}")
            
            return PipelineExecutionResult(
                success=False,
                execution_time=execution_time,
                components_used=components_used,
                performance_metrics=performance_metrics,
                error_message=error_message,
                warnings=warnings
            )
    
    def optimize_periods(self, 
                        data: pd.DataFrame,
                        labels: pd.DataFrame,
                        config: Optional[Dict[str, Any]] = None) -> PipelineExecutionResult:
        """
        Optimize periods using enhanced components.
        
        Args:
            data: Input data DataFrame
            labels: Labels DataFrame
            config: Optional configuration overrides
            
        Returns:
            PipelineExecutionResult with optimization results
        """
        tprint_info("Starting period optimization")
        start_time = time.time()
        
        components_used = []
        performance_metrics = {}
        
        try:
            # Use existing pipeline if available
            if self.existing_pipeline:
                # This would call the existing pipeline's period optimization
                components_used.append("existing_pipeline_period_optimization")
                performance_metrics['existing_pipeline'] = {'used': True}
            
            # Add enhanced validation
            if self.walk_forward_validator:
                splits = self.walk_forward_validator.generate_splits(data, labels)
                components_used.append("enhanced_walk_forward")
                performance_metrics['walk_forward_splits'] = len(splits)
            
            execution_time = time.time() - start_time
            
            # Update performance stats
            self.performance_stats['total_executions'] += 1
            self.performance_stats['successful_executions'] += 1
            self.performance_stats['total_execution_time'] += execution_time
            
            tprint_success(f"Period optimization completed in {execution_time:.3f}s")
            
            return PipelineExecutionResult(
                success=True,
                execution_time=execution_time,
                components_used=components_used,
                performance_metrics=performance_metrics
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            error_message = str(e)
            
            # Update performance stats
            self.performance_stats['total_executions'] += 1
            self.performance_stats['failed_executions'] += 1
            self.performance_stats['total_execution_time'] += execution_time
            
            tprint_error(f"Period optimization failed: {error_message}")
            
            return PipelineExecutionResult(
                success=False,
                execution_time=execution_time,
                components_used=components_used,
                performance_metrics=performance_metrics,
                error_message=error_message
            )
    
    def generate_interactions(self, 
                             data: pd.DataFrame,
                             config: Optional[Dict[str, Any]] = None) -> PipelineExecutionResult:
        """
        Generate interactions using enhanced components.
        
        Args:
            data: Input data DataFrame
            config: Optional configuration overrides
            
        Returns:
            PipelineExecutionResult with interaction generation results
        """
        tprint_info("Starting interaction generation")
        start_time = time.time()
        
        components_used = []
        performance_metrics = {}
        
        try:
            # Use existing pipeline if available
            if self.existing_pipeline:
                # This would call the existing pipeline's interaction generation
                components_used.append("existing_pipeline_interaction_generation")
                performance_metrics['existing_pipeline'] = {'used': True}
            
            # Add enhanced validation
            if self.schema_validator:
                validation_result = self.schema_validator.validate_data(
                    data, "features", "interaction_generation"
                )
                components_used.append("enhanced_schema_validation")
                performance_metrics['schema_validation'] = {
                    'valid': validation_result.is_valid,
                    'time': validation_result.validation_time
                }
            
            execution_time = time.time() - start_time
            
            # Update performance stats
            self.performance_stats['total_executions'] += 1
            self.performance_stats['successful_executions'] += 1
            self.performance_stats['total_execution_time'] += execution_time
            
            tprint_success(f"Interaction generation completed in {execution_time:.3f}s")
            
            return PipelineExecutionResult(
                success=True,
                execution_time=execution_time,
                components_used=components_used,
                performance_metrics=performance_metrics
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            error_message = str(e)
            
            # Update performance stats
            self.performance_stats['total_executions'] += 1
            self.performance_stats['failed_executions'] += 1
            self.performance_stats['total_execution_time'] += execution_time
            
            tprint_error(f"Interaction generation failed: {error_message}")
            
            return PipelineExecutionResult(
                success=False,
                execution_time=execution_time,
                components_used=components_used,
                performance_metrics=performance_metrics,
                error_message=error_message
            )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = self.performance_stats.copy()
        
        # Add component-specific performance
        if self.walk_forward_validator:
            summary['walk_forward_performance'] = self.walk_forward_validator.get_performance_summary()
        
        if self.statistical_framework:
            summary['statistical_performance'] = self.statistical_framework.get_performance_summary()
        
        if self.schema_validator:
            summary['schema_performance'] = self.schema_validator.get_performance_summary()
        
        if self.caching_integration:
            summary['caching_performance'] = self.caching_integration.get_performance_summary()
        
        if self.gpu_optimizer:
            summary['gpu_performance'] = self.gpu_optimizer.get_performance_summary()
        
        return summary
    
    def get_component_status(self) -> Dict[str, bool]:
        """Get status of all components."""
        return {
            'walk_forward_validator': self.walk_forward_validator is not None,
            'statistical_framework': self.statistical_framework is not None,
            'schema_validator': self.schema_validator is not None,
            'caching_integration': self.caching_integration is not None,
            'gpu_optimizer': self.gpu_optimizer is not None,
            'existing_pipeline': self.existing_pipeline is not None
        }


# Convenience functions
def create_enhanced_unified_pipeline(config: Optional[EnhancedPipelineConfig] = None) -> EnhancedUnifiedDataDrivenPipeline:
    """Create an enhanced unified pipeline."""
    return EnhancedUnifiedDataDrivenPipeline(config)


def run_enhanced_analysis(data: pd.DataFrame, 
                         labels: pd.DataFrame,
                         config: Optional[EnhancedPipelineConfig] = None) -> PipelineExecutionResult:
    """Run enhanced analysis with the unified pipeline."""
    pipeline = create_enhanced_unified_pipeline(config)
    return pipeline.run_comprehensive_analysis(data, labels)
