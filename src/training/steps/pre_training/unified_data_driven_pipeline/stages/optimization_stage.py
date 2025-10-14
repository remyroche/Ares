"""
Optimization Stage for Unified Data-Driven Pipeline

This module handles period optimization, lookback optimization, and interaction generation
for the unified pipeline.
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

# Import core components
from ..core.economic_evaluator import (
    EconomicPeriodEvaluator, EconomicEvaluationConfig, 
    EconomicPeriodEvaluationResult, create_economic_evaluator
)
from ..core.template_interaction_generator import (
    TemplateInteractionGenerator, TemplateConfig, 
    create_template_interaction_generator
)

# Import enhanced components
from ..enhanced_components.common_lookback_optimizer import (
    CommonLookbackOptimizer, LookbackOptimizationConfig,
    create_common_lookback_optimizer
)
from ..enhanced_components.advanced_lookback_optimizer import (
    AdvancedLookbackOptimizer, LookbackConstraints, OptimizationMethod
)
from ..enhanced_components.detailed_pipeline_reporter import DetailedPipelineReporter


@dataclass
class OptimizationStageResult:
    """Result from optimization stage."""
    
    optimized_periods: Dict[str, int]
    optimized_lookbacks: Dict[str, int]
    generated_interactions: List[str]
    optimization_metadata: Dict[str, Any]
    optimization_time: float
    memory_usage: float
    quality_score: float
    warnings: List[str]
    errors: List[str]


class OptimizationStage:
    """Optimization stage for the unified pipeline."""
    
    def __init__(self, config: Any, logger: Optional[logging.Logger] = None):
        """Initialize the optimization stage.
        
        Args:
            config: Pipeline configuration
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize optimization components
        self.economic_evaluator = create_economic_evaluator()
        self.template_generator = create_template_interaction_generator()
        self.common_lookback_optimizer = create_common_lookback_optimizer()
        self.advanced_lookback_optimizer = AdvancedLookbackOptimizer()
        self.detailed_reporter = DetailedPipelineReporter(outcomes_dir="outcomes")
        
        tprint_info("⚡ Optimization stage initialized")
    
    def optimize_features(self, 
                         data: pd.DataFrame, 
                         targets: Optional[pd.Series] = None,
                         feature_columns: Optional[List[str]] = None,
                         timeframe: str = "15m") -> OptimizationStageResult:
        """Optimize features using multiple optimization methods.
        
        Args:
            data: Input DataFrame with features
            targets: Optional target series for supervised learning
            feature_columns: Optional list of feature columns to optimize
            timeframe: Target timeframe (e.g., "15m", "5m", "1h")
            
        Returns:
            OptimizationStageResult with optimization results
        """
        start_time = time.time()
        warnings = []
        errors = []
        
        try:
            tprint_info(f"⚡ Optimizing features for timeframe: {timeframe}")
            
            # Start detailed reporting
            self.detailed_reporter.start_step("optimization", len(data.columns))
            
            # Step 1: Period optimization
            tprint_info("Step 1: Period optimization")
            optimized_periods = self._optimize_periods(data, targets, timeframe)
            
            # Step 2: Lookback optimization
            tprint_info("Step 2: Lookback optimization")
            optimized_lookbacks = self._optimize_lookbacks(data, targets, timeframe)
            
            # Step 3: Interaction generation
            tprint_info("Step 3: Interaction generation")
            generated_interactions = self._generate_interactions(data, targets, timeframe)
            
            # Calculate metrics
            optimization_time = time.time() - start_time
            memory_usage = data.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            quality_score = self._calculate_optimization_quality(
                optimized_periods, optimized_lookbacks, generated_interactions
            )
            
            # Create result
            result = OptimizationStageResult(
                optimized_periods=optimized_periods,
                optimized_lookbacks=optimized_lookbacks,
                generated_interactions=generated_interactions,
                optimization_metadata={
                    'timeframe': timeframe,
                    'optimization_methods': ['period', 'lookback', 'interaction'],
                    'total_optimizations': len(optimized_periods) + len(optimized_lookbacks) + len(generated_interactions)
                },
                optimization_time=optimization_time,
                memory_usage=memory_usage,
                quality_score=quality_score,
                warnings=warnings,
                errors=errors
            )
            
            # End detailed reporting
            self.detailed_reporter.end_step(
                "optimization", 
                len(optimized_periods) + len(optimized_lookbacks) + len(generated_interactions),
                optimization_time,
                memory_usage,
                True
            )
            
            tprint_success(f"✅ Optimization completed in {optimization_time:.2f}s")
            tprint_info(f"📊 Memory usage: {memory_usage:.2f} MB")
            tprint_info(f"📈 Quality score: {quality_score:.3f}")
            
            return result
            
        except Exception as e:
            optimization_time = time.time() - start_time
            error_msg = f"Optimization failed: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            
            return OptimizationStageResult(
                optimized_periods={},
                optimized_lookbacks={},
                generated_interactions=[],
                optimization_metadata={},
                optimization_time=optimization_time,
                memory_usage=0.0,
                quality_score=0.0,
                warnings=warnings,
                errors=[error_msg]
            )
    
    def _optimize_periods(self, 
                         data: pd.DataFrame, 
                         targets: Optional[pd.Series],
                         timeframe: str) -> Dict[str, int]:
        """Optimize periods using economic evaluation.
        
        Args:
            data: Input DataFrame
            targets: Optional target series
            timeframe: Target timeframe
            
        Returns:
            Dictionary of feature names to optimized periods
        """
        try:
            optimized_periods = {}
            
            # Get numeric columns for period optimization
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Limit the number of features to optimize based on config
            max_features = getattr(self.config, 'max_period_optimization_features', 10)
            if len(numeric_cols) > max_features:
                # Select top features by variance
                variances = data[numeric_cols].var().sort_values(ascending=False)
                numeric_cols = variances.head(max_features).index.tolist()
            
            for feature in numeric_cols:
                try:
                    # Use economic evaluator to find optimal period
                    evaluation_result = self.economic_evaluator.evaluate_periods(
                        data[feature], targets=targets
                    )
                    
                    if evaluation_result and hasattr(evaluation_result, 'top_periods'):
                        optimal_period = evaluation_result.top_periods[0] if evaluation_result.top_periods else 20
                        optimized_periods[feature] = optimal_period
                    else:
                        # Fallback to default period
                        optimized_periods[feature] = 20
                        
                except Exception as e:
                    tprint_debug(f"Period optimization failed for {feature}: {e}")
                    optimized_periods[feature] = 20  # Default period
            
            tprint_info(f"Optimized periods for {len(optimized_periods)} features")
            return optimized_periods
            
        except Exception as e:
            tprint_warning(f"⚠️ Period optimization failed: {e}")
            return {}
    
    def _optimize_lookbacks(self, 
                          data: pd.DataFrame, 
                          targets: Optional[pd.Series],
                          timeframe: str) -> Dict[str, int]:
        """Optimize lookbacks using advanced optimization.
        
        Args:
            data: Input DataFrame
            targets: Optional target series
            timeframe: Target timeframe
            
        Returns:
            Dictionary of feature names to optimized lookbacks
        """
        try:
            optimized_lookbacks = {}
            
            # Get numeric columns for lookback optimization
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Limit the number of features to optimize based on config
            max_features = getattr(self.config, 'max_lookback_optimization_features', 10)
            if len(numeric_cols) > max_features:
                # Select top features by variance
                variances = data[numeric_cols].var().sort_values(ascending=False)
                numeric_cols = variances.head(max_features).index.tolist()
            
            for feature in numeric_cols:
                try:
                    # Use common lookback optimizer
                    lookback_result = self.common_lookback_optimizer.optimize_lookback(
                        data[feature], targets=targets, timeframe=timeframe
                    )
                    
                    if lookback_result and hasattr(lookback_result, 'optimal_lookback'):
                        optimal_lookback = lookback_result.optimal_lookback
                        optimized_lookbacks[feature] = optimal_lookback
                    else:
                        # Fallback to default lookback
                        optimized_lookbacks[feature] = 20
                        
                except Exception as e:
                    tprint_debug(f"Lookback optimization failed for {feature}: {e}")
                    optimized_lookbacks[feature] = 20  # Default lookback
            
            tprint_info(f"Optimized lookbacks for {len(optimized_lookbacks)} features")
            return optimized_lookbacks
            
        except Exception as e:
            tprint_warning(f"⚠️ Lookback optimization failed: {e}")
            return {}
    
    def _generate_interactions(self, 
                             data: pd.DataFrame, 
                             targets: Optional[pd.Series],
                             timeframe: str) -> List[str]:
        """Generate feature interactions.
        
        Args:
            data: Input DataFrame
            targets: Optional target series
            timeframe: Target timeframe
            
        Returns:
            List of generated interaction feature names
        """
        try:
            # Use template interaction generator
            interaction_result = self.template_generator.generate_interactions(
                data, targets=targets, timeframe=timeframe
            )
            
            if interaction_result and hasattr(interaction_result, 'interactions'):
                interactions = interaction_result.interactions
                tprint_info(f"Generated {len(interactions)} interactions")
                return interactions
            else:
                # Fallback to basic interactions
                return self._generate_basic_interactions(data)
                
        except Exception as e:
            tprint_warning(f"⚠️ Interaction generation failed: {e}")
            return self._generate_basic_interactions(data)
    
    def _generate_basic_interactions(self, data: pd.DataFrame) -> List[str]:
        """Generate basic interactions as fallback.
        
        Args:
            data: Input DataFrame
            
        Returns:
            List of basic interaction feature names
        """
        try:
            interactions = []
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Generate simple pairwise interactions
            max_interactions = getattr(self.config, 'max_basic_interactions', 10)
            interaction_count = 0
            
            for i, col1 in enumerate(numeric_cols):
                for j, col2 in enumerate(numeric_cols[i+1:], i+1):
                    if interaction_count >= max_interactions:
                        break
                    
                    # Create interaction feature name
                    interaction_name = f"{col1}_x_{col2}"
                    interactions.append(interaction_name)
                    interaction_count += 1
                
                if interaction_count >= max_interactions:
                    break
            
            tprint_info(f"Generated {len(interactions)} basic interactions")
            return interactions
            
        except Exception as e:
            tprint_warning(f"⚠️ Basic interaction generation failed: {e}")
            return []
    
    def _calculate_optimization_quality(self, 
                                      optimized_periods: Dict[str, int],
                                      optimized_lookbacks: Dict[str, int],
                                      generated_interactions: List[str]) -> float:
        """Calculate quality score for the optimization results.
        
        Args:
            optimized_periods: Dictionary of optimized periods
            optimized_lookbacks: Dictionary of optimized lookbacks
            generated_interactions: List of generated interactions
            
        Returns:
            Quality score between 0 and 1
        """
        try:
            # Calculate various quality metrics
            period_count = len(optimized_periods)
            lookback_count = len(optimized_lookbacks)
            interaction_count = len(generated_interactions)
            
            # Calculate diversity score
            total_optimizations = period_count + lookback_count + interaction_count
            diversity_score = min(1.0, total_optimizations / 30.0)  # Prefer around 30 optimizations
            
            # Calculate balance score
            if total_optimizations > 0:
                period_ratio = period_count / total_optimizations
                lookback_ratio = lookback_count / total_optimizations
                interaction_ratio = interaction_count / total_optimizations
                
                # Prefer balanced optimization
                balance_score = 1.0 - abs(period_ratio - 0.33) - abs(lookback_ratio - 0.33) - abs(interaction_ratio - 0.33)
                balance_score = max(0.0, balance_score)
            else:
                balance_score = 0.0
            
            # Combine scores
            quality_score = (diversity_score + balance_score) / 2.0
            quality_score = max(0.0, min(1.0, quality_score))
            
            return quality_score
            
        except Exception as e:
            tprint_warning(f"⚠️ Optimization quality calculation failed: {e}")
            return 0.0
    
    def get_optimization_summary(self, result: OptimizationStageResult) -> Dict[str, Any]:
        """Get a summary of optimization results.
        
        Args:
            result: OptimizationStageResult to summarize
            
        Returns:
            Dictionary with optimization summary
        """
        return {
            'period_optimizations': len(result.optimized_periods),
            'lookback_optimizations': len(result.optimized_lookbacks),
            'interactions_generated': len(result.generated_interactions),
            'quality_score': result.quality_score,
            'optimization_time': result.optimization_time,
            'memory_usage_mb': result.memory_usage,
            'warnings_count': len(result.warnings),
            'errors_count': len(result.errors),
            'metadata': result.optimization_metadata
        }
    
    def cleanup(self) -> None:
        """Clean up resources."""
        tprint_debug("🧹 Cleaning up optimization stage")
        # Add any cleanup logic here if needed


def create_optimization_stage(config: Any, logger: Optional[logging.Logger] = None) -> OptimizationStage:
    """Create an optimization stage instance.
    
    Args:
        config: Pipeline configuration
        logger: Optional logger instance
        
    Returns:
        OptimizationStage instance
    """
    return OptimizationStage(config, logger)