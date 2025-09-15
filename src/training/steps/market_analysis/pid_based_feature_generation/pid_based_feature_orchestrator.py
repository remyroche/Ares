"""
PID-Based Feature Orchestrator

This module orchestrates all PID-based feature generation processes, integrating
interaction, polynomial, and cross-timeframe feature generation with optimized
lookback periods from feature_lookback_optimization.

Key Features:
- Orchestrates all three feature generation types
- Integrates optimized lookback periods
- Uses matrix_operations/ for all calculations
- Comprehensive validation and error handling
- Hardware-optimized computations
"""

import asyncio
import logging
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

# Core dependencies with fallback support
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

# Import feature generators
from .interaction_feature_generator import InteractionFeatureGenerator, InteractionConfig, InteractionResult
from .polynomial_feature_generator import PolynomialFeatureGenerator, PolynomialConfig, PolynomialResult
from .cross_timeframe_feature_generator import CrossTimeframeFeatureGenerator, CrossTimeframeConfig, CrossTimeframeResult

# Import matrix operations
try:
    from src.utils.matrix_operations import get_unified_matrix_operations
    MATRIX_OPS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Matrix operations not available: {e}")
    MATRIX_OPS_AVAILABLE = False

# Import logger
try:
    from src.utils.logger import system_logger
    logger = system_logger.getChild('PIDBasedFeatureOrchestrator')
except ImportError:
    logger = logging.getLogger('PIDBasedFeatureOrchestrator')
    logger.setLevel(logging.INFO)


class GenerationStatus(Enum):
    """Status of feature generation process."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class OrchestratorConfig:
    """Configuration for PID-based feature orchestrator."""
    # Feature Generation Limits
    max_interaction_features: int = 100
    max_polynomial_features: int = 50
    max_cross_timeframe_features: int = 50
    
    # PID Configuration
    synergy_threshold: float = 0.1
    redundancy_threshold: float = 0.15
    unique_info_threshold: float = 0.05
    
    # Computational Settings
    enable_parallel_processing: bool = True
    enable_gpu_acceleration: bool = True
    memory_limit_gb: float = 8.0
    
    # Generation Control
    enable_interaction_features: bool = True
    enable_polynomial_features: bool = True
    enable_cross_timeframe_features: bool = True
    
    # Validation
    min_feature_quality_score: float = 0.3
    max_redundancy_threshold: float = 0.8
    
    # Hardware Optimization
    chunk_size_mb: int = 256
    max_memory_percent: float = 0.7


@dataclass
class OrchestratorResult:
    """Result of PID-based feature orchestration."""
    # Individual Results
    interaction_result: Optional[InteractionResult] = None
    polynomial_result: Optional[PolynomialResult] = None
    cross_timeframe_result: Optional[CrossTimeframeResult] = None
    
    # Combined Results
    combined_features: Dict[str, np.ndarray] = field(default_factory=dict)
    combined_feature_names: List[str] = field(default_factory=list)
    feature_importance_scores: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    total_features_generated: int = 0
    execution_time: float = 0.0
    optimization_used: bool = False
    matrix_ops_used: bool = False
    generation_status: GenerationStatus = GenerationStatus.PENDING
    
    # Quality Metrics
    overall_quality_score: float = 0.0
    feature_diversity_score: float = 0.0
    redundancy_score: float = 0.0
    stability_score: float = 0.0


class PIDBasedFeatureOrchestrator:
    """
    PID-Based Feature Orchestrator.
    
    Orchestrates all PID-based feature generation processes, integrating
    interaction, polynomial, and cross-timeframe feature generation.
    """
    
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        """Initialize the PID-based feature orchestrator."""
        self.config = config or OrchestratorConfig()
        self.logger = logger.getChild('PIDBasedFeatureOrchestrator')
        
        # Initialize components
        self._initialize_components()
        
        self.logger.info("🔧 PIDBasedFeatureOrchestrator initialized")
        self.logger.info(f"📊 Max interaction features: {self.config.max_interaction_features}")
        self.logger.info(f"📊 Max polynomial features: {self.config.max_polynomial_features}")
        self.logger.info(f"📊 Max cross-timeframe features: {self.config.max_cross_timeframe_features}")
    
    def _initialize_components(self):
        """Initialize required components."""
        # Initialize feature generators
        if self.config.enable_interaction_features:
            interaction_config = InteractionConfig(
                synergy_threshold=self.config.synergy_threshold,
                redundancy_threshold=self.config.redundancy_threshold,
                unique_info_threshold=self.config.unique_info_threshold,
                max_interaction_features=self.config.max_interaction_features,
                enable_parallel_processing=self.config.enable_parallel_processing,
                enable_gpu_acceleration=self.config.enable_gpu_acceleration,
                memory_limit_gb=self.config.memory_limit_gb
            )
            self.interaction_generator = InteractionFeatureGenerator(interaction_config)
            self.logger.info("✅ Interaction Feature Generator initialized")
        else:
            self.interaction_generator = None
        
        if self.config.enable_polynomial_features:
            polynomial_config = PolynomialConfig(
                synergy_threshold=self.config.synergy_threshold,
                redundancy_threshold=self.config.redundancy_threshold,
                unique_info_threshold=self.config.unique_info_threshold,
                max_polynomial_features=self.config.max_polynomial_features,
                enable_parallel_processing=self.config.enable_parallel_processing,
                enable_gpu_acceleration=self.config.enable_gpu_acceleration,
                memory_limit_gb=self.config.memory_limit_gb
            )
            self.polynomial_generator = PolynomialFeatureGenerator(polynomial_config)
            self.logger.info("✅ Polynomial Feature Generator initialized")
        else:
            self.polynomial_generator = None
        
        if self.config.enable_cross_timeframe_features:
            cross_timeframe_config = CrossTimeframeConfig(
                synergy_threshold=self.config.synergy_threshold,
                redundancy_threshold=self.config.redundancy_threshold,
                unique_info_threshold=self.config.unique_info_threshold,
                max_cross_timeframe_features=self.config.max_cross_timeframe_features,
                enable_parallel_processing=self.config.enable_parallel_processing,
                enable_gpu_acceleration=self.config.enable_gpu_acceleration,
                memory_limit_gb=self.config.memory_limit_gb
            )
            self.cross_timeframe_generator = CrossTimeframeFeatureGenerator(cross_timeframe_config)
            self.logger.info("✅ Cross Timeframe Feature Generator initialized")
        else:
            self.cross_timeframe_generator = None
        
        # Initialize matrix operations
        if MATRIX_OPS_AVAILABLE:
            self.matrix_ops = get_unified_matrix_operations(
                enable_gpu=self.config.enable_gpu_acceleration,
                enable_memory_optimization=True,
                enable_parallel=self.config.enable_parallel_processing
            )
            self.logger.info("✅ Matrix Operations initialized")
        else:
            self.matrix_ops = None
            self.logger.warning("⚠️ Matrix Operations not available")
    
    async def orchestrate_feature_generation(
        self, 
        data: Union[np.ndarray, pd.DataFrame],
        feature_names: List[str],
        optimized_lookback_periods: Optional[Dict[str, int]] = None,
        target: Optional[np.ndarray] = None
    ) -> OrchestratorResult:
        """
        Orchestrate all PID-based feature generation processes.
        
        Args:
            data: Input feature matrix
            feature_names: List of feature names
            optimized_lookback_periods: Optimized lookback periods from feature_lookback_optimization
            target: Target variable for PID analysis (optional)
            
        Returns:
            OrchestratorResult with all generated features
        """
        start_time = time.time()
        self.logger.info("🔧 Starting PID-based feature orchestration...")
        
        result = OrchestratorResult()
        result.generation_status = GenerationStatus.IN_PROGRESS
        
        try:
            # Convert data to numpy array if needed
            if isinstance(data, pd.DataFrame):
                X = data.values
                if feature_names is None:
                    feature_names = list(data.columns)
            else:
                X = data
            
            self.logger.info(f"📊 Input data shape: {X.shape}")
            self.logger.info(f"📊 Feature count: {len(feature_names)}")
            
            # Track optimization usage
            if optimized_lookback_periods:
                result.optimization_used = True
                self.logger.info("✅ Optimized lookback periods will be applied")
            
            # Generate features in parallel if possible
            generation_tasks = []
            
            # Interaction features
            if self.interaction_generator:
                task = asyncio.create_task(
                    self.interaction_generator.generate_interaction_features(
                        X, feature_names, optimized_lookback_periods, target
                    )
                )
                generation_tasks.append(('interaction', task))
            
            # Polynomial features
            if self.polynomial_generator:
                task = asyncio.create_task(
                    self.polynomial_generator.generate_polynomial_features(
                        X, feature_names, optimized_lookback_periods, target
                    )
                )
                generation_tasks.append(('polynomial', task))
            
            # Cross-timeframe features
            if self.cross_timeframe_generator:
                task = asyncio.create_task(
                    self.cross_timeframe_generator.generate_cross_timeframe_features(
                        X, feature_names, optimized_lookback_periods, target
                    )
                )
                generation_tasks.append(('cross_timeframe', task))
            
            # Wait for all tasks to complete
            self.logger.info(f"🚀 Executing {len(generation_tasks)} feature generation tasks...")
            completed_tasks = await asyncio.gather(*[task for _, task in generation_tasks], return_exceptions=True)
            
            # Process results
            successful_generations = 0
            for (generation_type, _), task_result in zip(generation_tasks, completed_tasks):
                if isinstance(task_result, Exception):
                    self.logger.error(f"❌ {generation_type} feature generation failed: {task_result}")
                    continue
                
                if generation_type == 'interaction':
                    result.interaction_result = task_result
                elif generation_type == 'polynomial':
                    result.polynomial_result = task_result
                elif generation_type == 'cross_timeframe':
                    result.cross_timeframe_result = task_result
                
                successful_generations += 1
                self.logger.info(f"✅ {generation_type} feature generation completed")
            
            # Combine all generated features
            self.logger.info("🔧 Combining all generated features...")
            combined_features, combined_names, importance_scores = self._combine_features(result)
            
            # Store combined results
            result.combined_features = combined_features
            result.combined_feature_names = combined_names
            result.feature_importance_scores = importance_scores
            result.total_features_generated = len(combined_names)
            result.matrix_ops_used = self.matrix_ops is not None
            
            # Calculate quality metrics
            result.overall_quality_score = self._calculate_overall_quality_score(result)
            result.feature_diversity_score = self._calculate_feature_diversity_score(combined_names)
            result.redundancy_score = self._calculate_redundancy_score(combined_features)
            result.stability_score = self._calculate_stability_score(result)
            
            # Determine final status
            if successful_generations == len(generation_tasks):
                result.generation_status = GenerationStatus.COMPLETED
            elif successful_generations > 0:
                result.generation_status = GenerationStatus.PARTIAL
            else:
                result.generation_status = GenerationStatus.FAILED
            
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            self.logger.info(f"✅ PID-based feature orchestration completed in {execution_time:.3f}s")
            self.logger.info(f"📊 Generated {result.total_features_generated} total features")
            self.logger.info(f"📊 Overall quality score: {result.overall_quality_score:.3f}")
            self.logger.info(f"📊 Feature diversity score: {result.feature_diversity_score:.3f}")
            self.logger.info(f"📊 Generation status: {result.generation_status.value}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            result.generation_status = GenerationStatus.FAILED
            
            self.logger.error(f"❌ PID-based feature orchestration failed: {e}")
            self.logger.error(f"❌ Error details: {traceback.format_exc()}")
            
            return result
    
    def _combine_features(
        self, 
        result: OrchestratorResult
    ) -> Tuple[Dict[str, np.ndarray], List[str], Dict[str, float]]:
        """Combine features from all generators."""
        combined_features = {}
        combined_names = []
        importance_scores = {}
        
        # Add interaction features
        if result.interaction_result and result.interaction_result.interaction_features:
            for name, feature in result.interaction_result.interaction_features.items():
                combined_features[f"interaction_{name}"] = feature
                combined_names.append(f"interaction_{name}")
                importance_scores[f"interaction_{name}"] = result.interaction_result.interaction_scores.get(name, 0.0)
        
        # Add polynomial features
        if result.polynomial_result and result.polynomial_result.polynomial_features:
            for name, feature in result.polynomial_result.polynomial_features.items():
                combined_features[f"polynomial_{name}"] = feature
                combined_names.append(f"polynomial_{name}")
                importance_scores[f"polynomial_{name}"] = result.polynomial_result.polynomial_scores.get(name, 0.0)
        
        # Add cross-timeframe features
        if result.cross_timeframe_result and result.cross_timeframe_result.cross_timeframe_features:
            for name, feature in result.cross_timeframe_result.cross_timeframe_features.items():
                combined_features[f"cross_timeframe_{name}"] = feature
                combined_names.append(f"cross_timeframe_{name}")
                importance_scores[f"cross_timeframe_{name}"] = result.cross_timeframe_result.cross_timeframe_scores.get(name, 0.0)
        
        return combined_features, combined_names, importance_scores
    
    def _calculate_overall_quality_score(self, result: OrchestratorResult) -> float:
        """Calculate overall quality score."""
        try:
            scores = []
            
            # Individual quality scores
            if result.interaction_result:
                scores.append(result.interaction_result.feature_stability_score)
            
            if result.polynomial_result:
                scores.append(result.polynomial_result.feature_stability_score)
            
            if result.cross_timeframe_result:
                scores.append(result.cross_timeframe_result.feature_stability_score)
            
            # Generation success rate
            total_generators = sum([
                bool(result.interaction_result),
                bool(result.polynomial_result),
                bool(result.cross_timeframe_result)
            ])
            success_rate = total_generators / 3.0 if total_generators > 0 else 0.0
            scores.append(success_rate)
            
            return float(np.mean(scores)) if scores else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_feature_diversity_score(self, feature_names: List[str]) -> float:
        """Calculate feature diversity score based on naming patterns."""
        try:
            if not feature_names:
                return 0.0
            
            # Count different feature types
            interaction_count = sum(1 for name in feature_names if name.startswith('interaction_'))
            polynomial_count = sum(1 for name in feature_names if name.startswith('polynomial_'))
            cross_timeframe_count = sum(1 for name in feature_names if name.startswith('cross_timeframe_'))
            
            total_count = len(feature_names)
            
            # Calculate diversity as entropy
            proportions = [
                interaction_count / total_count,
                polynomial_count / total_count,
                cross_timeframe_count / total_count
            ]
            
            # Remove zero proportions
            proportions = [p for p in proportions if p > 0]
            
            if not proportions:
                return 0.0
            
            # Calculate entropy
            entropy = -sum(p * np.log2(p) for p in proportions)
            max_entropy = np.log2(len(proportions))
            
            return entropy / max_entropy if max_entropy > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_redundancy_score(self, combined_features: Dict[str, np.ndarray]) -> float:
        """Calculate redundancy score."""
        try:
            if len(combined_features) < 2:
                return 0.0
            
            # Convert to matrix
            feature_matrix = np.column_stack(list(combined_features.values()))
            
            if self.matrix_ops:
                corr_matrix = self.matrix_ops.safe_correlation_matrix(feature_matrix)
            else:
                corr_matrix = np.corrcoef(feature_matrix.T)
            
            # Count high correlations (>0.8)
            n = corr_matrix.shape[0]
            upper_triangle = corr_matrix[np.triu_indices(n, k=1)]
            high_correlations = np.sum(np.abs(upper_triangle) > 0.8)
            
            # Normalize by total possible correlations
            total_correlations = n * (n - 1) // 2
            redundancy_score = high_correlations / total_correlations if total_correlations > 0 else 0.0
            
            return float(redundancy_score)
            
        except Exception:
            return 0.0
    
    def _calculate_stability_score(self, result: OrchestratorResult) -> float:
        """Calculate overall stability score."""
        try:
            scores = []
            
            if result.interaction_result:
                scores.append(result.interaction_result.feature_stability_score)
            
            if result.polynomial_result:
                scores.append(result.polynomial_result.feature_stability_score)
            
            if result.cross_timeframe_result:
                scores.append(result.cross_timeframe_result.feature_stability_score)
            
            return float(np.mean(scores)) if scores else 0.0
            
        except Exception:
            return 0.0
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        metrics = {
            'orchestrator_config': {
                'max_interaction_features': self.config.max_interaction_features,
                'max_polynomial_features': self.config.max_polynomial_features,
                'max_cross_timeframe_features': self.config.max_cross_timeframe_features,
                'enable_interaction_features': self.config.enable_interaction_features,
                'enable_polynomial_features': self.config.enable_polynomial_features,
                'enable_cross_timeframe_features': self.config.enable_cross_timeframe_features
            },
            'component_availability': {
                'interaction_generator': self.interaction_generator is not None,
                'polynomial_generator': self.polynomial_generator is not None,
                'cross_timeframe_generator': self.cross_timeframe_generator is not None,
                'matrix_ops': self.matrix_ops is not None
            },
            'system_availability': {
                'numpy_available': NUMPY_AVAILABLE,
                'pandas_available': PANDAS_AVAILABLE,
                'matrix_ops_available': MATRIX_OPS_AVAILABLE
            }
        }
        
        # Add individual generator metrics
        if self.interaction_generator:
            metrics['interaction_generator_metrics'] = self.interaction_generator.get_performance_metrics()
        
        if self.polynomial_generator:
            metrics['polynomial_generator_metrics'] = self.polynomial_generator.get_performance_metrics()
        
        if self.cross_timeframe_generator:
            metrics['cross_timeframe_generator_metrics'] = self.cross_timeframe_generator.get_performance_metrics()
        
        if self.matrix_ops:
            metrics['matrix_ops_stats'] = self.matrix_ops.get_performance_stats()
            metrics['hardware_info'] = self.matrix_ops.get_hardware_info()
        
        return metrics