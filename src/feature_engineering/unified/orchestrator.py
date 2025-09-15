"""
Unified Feature Orchestrator

This module provides the main orchestration system that coordinates
all feature generation across the system with intelligent dependency
resolution and performance optimization.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from .core import (
    FeatureGenerator, 
    FeatureGeneratorConfig, 
    FeatureGenerationResult,
    FeatureCategory,
    FeaturePriority,
    CompositeFeatureGenerator
)
from .registry import FeatureRegistry, get_registry
from ...utils.logger import system_logger
from ...core.decorators import handles_errors


@dataclass
class OrchestrationConfig:
    """Configuration for feature orchestration."""
    enable_parallel_processing: bool = True
    max_parallel_generators: int = 4
    enable_dependency_resolution: bool = True
    enable_performance_optimization: bool = True
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    enable_validation: bool = True
    enable_quality_checks: bool = True
    timeout_seconds: Optional[int] = None
    memory_limit_mb: Optional[int] = None
    retry_failed_generators: bool = True
    max_retries: int = 3
    retry_delay_seconds: float = 1.0


@dataclass
class FeaturePipeline:
    """A pipeline of feature generators with execution plan."""
    name: str
    generators: List[FeatureGenerator]
    execution_order: List[str] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    estimated_duration: float = 0.0
    memory_estimate_mb: float = 0.0


class FeatureOrchestrator:
    """
    Unified feature orchestrator that coordinates all feature generation.
    
    Provides intelligent dependency resolution, parallel processing,
    performance optimization, and quality assurance.
    """
    
    def __init__(self, config: OrchestrationConfig):
        """
        Initialize the feature orchestrator.
        
        Args:
            config: Orchestration configuration
        """
        self.config = config
        self.logger = system_logger.getChild("FeatureOrchestrator")
        self.registry = get_registry()
        self._pipelines: Dict[str, FeaturePipeline] = {}
        self._cache: Dict[str, Any] = {}
        self._performance_metrics: Dict[str, Any] = {}
        self._initialized = False
        
    async def initialize(self) -> bool:
        """Initialize the orchestrator and registry."""
        try:
            self.logger.info("Initializing feature orchestrator...")
            
            # Initialize registry
            if not await self.registry.initialize():
                self.logger.error("Failed to initialize feature registry")
                return False
            
            # Create default pipelines
            await self._create_default_pipelines()
            
            self._initialized = True
            self.logger.info("Feature orchestrator initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing orchestrator: {e}")
            return False
    
    async def _create_default_pipelines(self) -> None:
        """Create default feature pipelines."""
        # Basic technical indicators pipeline
        basic_generators = await self._get_generators_by_categories([
            FeatureCategory.TECHNICAL_INDICATORS,
            FeatureCategory.STATISTICAL_FEATURES
        ])
        
        if basic_generators:
            basic_pipeline = FeaturePipeline(
                name="basic_indicators",
                generators=basic_generators
            )
            await self._optimize_pipeline(basic_pipeline)
            self._pipelines["basic_indicators"] = basic_pipeline
        
        # Advanced features pipeline
        advanced_generators = await self._get_generators_by_categories([
            FeatureCategory.MICROSTRUCTURE,
            FeatureCategory.VOLATILITY,
            FeatureCategory.MOMENTUM,
            FeatureCategory.PATTERN_RECOGNITION
        ])
        
        if advanced_generators:
            advanced_pipeline = FeaturePipeline(
                name="advanced_features",
                generators=advanced_generators
            )
            await self._optimize_pipeline(advanced_pipeline)
            self._pipelines["advanced_features"] = advanced_pipeline
        
        # Cross-timeframe pipeline
        cross_tf_generators = await self._get_generators_by_categories([
            FeatureCategory.CROSS_TIMEFRAME,
            FeatureCategory.META_LABELING
        ])
        
        if cross_tf_generators:
            cross_tf_pipeline = FeaturePipeline(
                name="cross_timeframe",
                generators=cross_tf_generators
            )
            await self._optimize_pipeline(cross_tf_pipeline)
            self._pipelines["cross_timeframe"] = cross_tf_pipeline
    
    async def _get_generators_by_categories(self, categories: List[FeatureCategory]) -> List[FeatureGenerator]:
        """Get generators for specific categories."""
        generators = []
        
        for category in categories:
            category_generators = self.registry.get_generators_by_category(category)
            for info in category_generators:
                if info.config.enabled:
                    instance = await self.registry.create_generator_instance(info.name)
                    if instance:
                        generators.append(instance)
        
        return generators
    
    async def _optimize_pipeline(self, pipeline: FeaturePipeline) -> None:
        """Optimize pipeline execution order and dependencies."""
        try:
            # Resolve dependencies
            if self.config.enable_dependency_resolution:
                pipeline.execution_order = await self._resolve_dependencies(pipeline.generators)
            else:
                pipeline.execution_order = [gen.config.name for gen in pipeline.generators]
            
            # Estimate performance
            pipeline.estimated_duration = await self._estimate_pipeline_duration(pipeline)
            pipeline.memory_estimate_mb = await self._estimate_pipeline_memory(pipeline)
            
            self.logger.info(f"Optimized pipeline {pipeline.name}: {len(pipeline.generators)} generators, "
                           f"estimated {pipeline.estimated_duration:.2f}s, {pipeline.memory_estimate_mb:.1f}MB")
            
        except Exception as e:
            self.logger.warning(f"Error optimizing pipeline {pipeline.name}: {e}")
    
    async def _resolve_dependencies(self, generators: List[FeatureGenerator]) -> List[str]:
        """Resolve dependencies between generators."""
        # Simple topological sort for dependency resolution
        # Can be enhanced with more sophisticated algorithms
        
        dependency_graph = {}
        in_degree = {}
        
        # Build dependency graph
        for generator in generators:
            name = generator.config.name
            dependencies = generator.config.dependencies
            dependency_graph[name] = dependencies
            in_degree[name] = len(dependencies)
        
        # Topological sort
        queue = [name for name, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            # Update in-degrees of dependent generators
            for name, deps in dependency_graph.items():
                if current in deps:
                    in_degree[name] -= 1
                    if in_degree[name] == 0:
                        queue.append(name)
        
        return result
    
    async def _estimate_pipeline_duration(self, pipeline: FeaturePipeline) -> float:
        """Estimate pipeline execution duration."""
        total_duration = 0.0
        
        for generator in pipeline.generators:
            # Use performance targets if available
            if 'avg_duration_ms' in generator.config.performance_targets:
                total_duration += generator.config.performance_targets['avg_duration_ms'] / 1000.0
            else:
                # Default estimate based on category
                category_estimates = {
                    FeatureCategory.TECHNICAL_INDICATORS: 0.1,
                    FeatureCategory.STATISTICAL_FEATURES: 0.05,
                    FeatureCategory.MICROSTRUCTURE: 0.5,
                    FeatureCategory.VOLATILITY: 0.2,
                    FeatureCategory.MOMENTUM: 0.1,
                    FeatureCategory.VOLUME: 0.1,
                    FeatureCategory.TIME_SERIES: 0.05,
                    FeatureCategory.CROSS_TIMEFRAME: 1.0,
                    FeatureCategory.META_LABELING: 0.5,
                    FeatureCategory.PATTERN_RECOGNITION: 0.3,
                    FeatureCategory.REGIME_DETECTION: 0.4,
                    FeatureCategory.LIQUIDITY: 0.2,
                    FeatureCategory.CUSTOM: 0.1
                }
                total_duration += category_estimates.get(generator.config.category, 0.1)
        
        return total_duration
    
    async def _estimate_pipeline_memory(self, pipeline: FeaturePipeline) -> float:
        """Estimate pipeline memory usage."""
        total_memory = 0.0
        
        for generator in pipeline.generators:
            if generator.config.memory_limit_mb:
                total_memory += generator.config.memory_limit_mb
            else:
                # Default memory estimates
                category_estimates = {
                    FeatureCategory.TECHNICAL_INDICATORS: 10.0,
                    FeatureCategory.STATISTICAL_FEATURES: 5.0,
                    FeatureCategory.MICROSTRUCTURE: 50.0,
                    FeatureCategory.VOLATILITY: 20.0,
                    FeatureCategory.MOMENTUM: 10.0,
                    FeatureCategory.VOLUME: 15.0,
                    FeatureCategory.TIME_SERIES: 5.0,
                    FeatureCategory.CROSS_TIMEFRAME: 100.0,
                    FeatureCategory.META_LABELING: 50.0,
                    FeatureCategory.PATTERN_RECOGNITION: 30.0,
                    FeatureCategory.REGIME_DETECTION: 40.0,
                    FeatureCategory.LIQUIDITY: 25.0,
                    FeatureCategory.CUSTOM: 10.0
                }
                total_memory += category_estimates.get(generator.config.category, 10.0)
        
        return total_memory
    
    @handles_errors(exceptions=(Exception,), default_return=FeatureGenerationResult(success=False), context="feature generation")
    async def generate_features(
        self,
        data: pd.DataFrame,
        pipeline_name: Optional[str] = None,
        generator_names: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> FeatureGenerationResult:
        """
        Generate features using specified pipeline or generators.
        
        Args:
            data: Input data for feature generation
            pipeline_name: Name of pipeline to use (optional)
            generator_names: Specific generators to use (optional)
            context: Additional context information
            
        Returns:
            FeatureGenerationResult containing generated features
        """
        if not self._initialized:
            return FeatureGenerationResult(
                success=False,
                errors=["Orchestrator not initialized"]
            )
        
        try:
            start_time = time.time()
            
            # Determine which generators to use
            if pipeline_name and pipeline_name in self._pipelines:
                generators = self._pipelines[pipeline_name].generators
                execution_order = self._pipelines[pipeline_name].execution_order
            elif generator_names:
                generators = await self._get_specific_generators(generator_names)
                execution_order = generator_names
            else:
                # Use all enabled generators
                generators = await self._get_all_enabled_generators()
                execution_order = [gen.config.name for gen in generators]
            
            if not generators:
                return FeatureGenerationResult(
                    success=False,
                    errors=["No generators available"]
                )
            
            # Generate features
            if self.config.enable_parallel_processing and len(generators) > 1:
                result = await self._generate_features_parallel(
                    data, generators, execution_order, context
                )
            else:
                result = await self._generate_features_sequential(
                    data, generators, execution_order, context
                )
            
            # Update performance metrics
            duration = time.time() - start_time
            self._performance_metrics[f"last_generation_duration"] = duration
            self._performance_metrics[f"total_generations"] = self._performance_metrics.get("total_generations", 0) + 1
            
            if result.success:
                self.logger.info(f"Feature generation completed in {duration:.2f}s, "
                               f"generated {len(result.features.columns)} features")
            else:
                self.logger.error(f"Feature generation failed after {duration:.2f}s: {result.errors}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in feature generation: {e}")
            return FeatureGenerationResult(
                success=False,
                errors=[f"Orchestration error: {str(e)}"]
            )
    
    async def _get_specific_generators(self, generator_names: List[str]) -> List[FeatureGenerator]:
        """Get specific generators by name."""
        generators = []
        
        for name in generator_names:
            instance = await self.registry.create_generator_instance(name)
            if instance:
                generators.append(instance)
            else:
                self.logger.warning(f"Could not create generator: {name}")
        
        return generators
    
    async def _get_all_enabled_generators(self) -> List[FeatureGenerator]:
        """Get all enabled generators."""
        generators = []
        enabled_infos = self.registry.get_enabled_generators()
        
        for info in enabled_infos:
            instance = await self.registry.create_generator_instance(info.name)
            if instance:
                generators.append(instance)
        
        return generators
    
    async def _generate_features_parallel(
        self,
        data: pd.DataFrame,
        generators: List[FeatureGenerator],
        execution_order: List[str],
        context: Optional[Dict[str, Any]]
    ) -> FeatureGenerationResult:
        """Generate features using parallel processing."""
        try:
            # Group generators by priority for execution
            priority_groups = {}
            for generator in generators:
                priority = generator.config.priority
                if priority not in priority_groups:
                    priority_groups[priority] = []
                priority_groups[priority].append(generator)
            
            all_features = []
            all_errors = []
            all_warnings = []
            performance_metrics = {}
            
            # Execute by priority groups
            for priority in sorted(priority_groups.keys(), key=lambda x: x.value):
                group_generators = priority_groups[priority]
                
                # Execute generators in parallel within each priority group
                with ThreadPoolExecutor(max_workers=self.config.max_parallel_generators) as executor:
                    futures = {
                        executor.submit(self._execute_generator, gen, data, context): gen
                        for gen in group_generators
                    }
                    
                    for future in as_completed(futures):
                        generator = futures[future]
                        try:
                            result = await asyncio.wrap_future(future)
                            
                            if result.success and result.features is not None:
                                all_features.append(result.features)
                            else:
                                all_errors.extend(result.errors)
                            
                            all_warnings.extend(result.warnings)
                            performance_metrics.update(result.performance_metrics)
                            
                        except Exception as e:
                            self.logger.error(f"Error executing generator {generator.config.name}: {e}")
                            all_errors.append(f"Generator {generator.config.name} failed: {str(e)}")
            
            # Combine features
            if all_features:
                combined_features = pd.concat(all_features, axis=1)
                # Remove duplicate columns
                combined_features = combined_features.loc[:, ~combined_features.columns.duplicated()]
                
                return FeatureGenerationResult(
                    success=True,
                    features=combined_features,
                    metadata={"parallel_execution": True, "generator_count": len(generators)},
                    errors=all_errors,
                    warnings=all_warnings,
                    performance_metrics=performance_metrics
                )
            else:
                return FeatureGenerationResult(
                    success=False,
                    errors=all_errors,
                    warnings=all_warnings,
                    performance_metrics=performance_metrics
                )
                
        except Exception as e:
            self.logger.error(f"Error in parallel feature generation: {e}")
            return FeatureGenerationResult(
                success=False,
                errors=[f"Parallel generation error: {str(e)}"]
            )
    
    async def _generate_features_sequential(
        self,
        data: pd.DataFrame,
        generators: List[FeatureGenerator],
        execution_order: List[str],
        context: Optional[Dict[str, Any]]
    ) -> FeatureGenerationResult:
        """Generate features using sequential processing."""
        all_features = []
        all_errors = []
        all_warnings = []
        performance_metrics = {}
        
        # Sort generators by execution order
        generator_map = {gen.config.name: gen for gen in generators}
        ordered_generators = [generator_map[name] for name in execution_order if name in generator_map]
        
        for generator in ordered_generators:
            try:
                result = await self._execute_generator(generator, data, context)
                
                if result.success and result.features is not None:
                    all_features.append(result.features)
                else:
                    all_errors.extend(result.errors)
                
                all_warnings.extend(result.warnings)
                performance_metrics.update(result.performance_metrics)
                
            except Exception as e:
                self.logger.error(f"Error executing generator {generator.config.name}: {e}")
                all_errors.append(f"Generator {generator.config.name} failed: {str(e)}")
        
        # Combine features
        if all_features:
            combined_features = pd.concat(all_features, axis=1)
            # Remove duplicate columns
            combined_features = combined_features.loc[:, ~combined_features.columns.duplicated()]
            
            return FeatureGenerationResult(
                success=True,
                features=combined_features,
                metadata={"sequential_execution": True, "generator_count": len(generators)},
                errors=all_errors,
                warnings=all_warnings,
                performance_metrics=performance_metrics
            )
        else:
            return FeatureGenerationResult(
                success=False,
                errors=all_errors,
                warnings=all_warnings,
                performance_metrics=performance_metrics
            )
    
    async def _execute_generator(
        self,
        generator: FeatureGenerator,
        data: pd.DataFrame,
        context: Optional[Dict[str, Any]]
    ) -> FeatureGenerationResult:
        """Execute a single generator with error handling and retries."""
        max_retries = self.config.max_retries if self.config.retry_failed_generators else 1
        
        for attempt in range(max_retries):
            try:
                # Initialize generator if needed
                if not generator.is_initialized():
                    if not await generator.initialize():
                        return FeatureGenerationResult(
                            success=False,
                            errors=[f"Failed to initialize generator {generator.config.name}"]
                        )
                
                # Execute generator
                result = await generator.generate_features(data, context)
                
                if result.success or attempt == max_retries - 1:
                    return result
                else:
                    self.logger.warning(f"Generator {generator.config.name} failed on attempt {attempt + 1}, retrying...")
                    await asyncio.sleep(self.config.retry_delay_seconds)
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    return FeatureGenerationResult(
                        success=False,
                        errors=[f"Generator {generator.config.name} failed after {max_retries} attempts: {str(e)}"]
                    )
                else:
                    self.logger.warning(f"Generator {generator.config.name} error on attempt {attempt + 1}: {e}")
                    await asyncio.sleep(self.config.retry_delay_seconds)
        
        return FeatureGenerationResult(success=False, errors=["Max retries exceeded"])
    
    def create_custom_pipeline(
        self,
        name: str,
        generator_names: List[str],
        config: Optional[OrchestrationConfig] = None
    ) -> bool:
        """Create a custom feature pipeline."""
        try:
            generators = []
            for gen_name in generator_names:
                instance = asyncio.run(self.registry.create_generator_instance(gen_name))
                if instance:
                    generators.append(instance)
                else:
                    self.logger.warning(f"Could not create generator {gen_name} for pipeline {name}")
            
            if not generators:
                self.logger.error(f"No valid generators for pipeline {name}")
                return False
            
            pipeline = FeaturePipeline(name=name, generators=generators)
            asyncio.run(self._optimize_pipeline(pipeline))
            self._pipelines[name] = pipeline
            
            self.logger.info(f"Created custom pipeline {name} with {len(generators)} generators")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating custom pipeline {name}: {e}")
            return False
    
    def get_pipeline_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a pipeline."""
        pipeline = self._pipelines.get(name)
        if not pipeline:
            return None
        
        return {
            "name": pipeline.name,
            "generator_count": len(pipeline.generators),
            "execution_order": pipeline.execution_order,
            "estimated_duration": pipeline.estimated_duration,
            "memory_estimate_mb": pipeline.memory_estimate_mb,
            "generators": [gen.get_info() for gen in pipeline.generators]
        }
    
    def list_pipelines(self) -> List[str]:
        """List all available pipelines."""
        return list(self._pipelines.keys())
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get orchestrator performance metrics."""
        return self._performance_metrics.copy()
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return self.registry.get_registry_stats()