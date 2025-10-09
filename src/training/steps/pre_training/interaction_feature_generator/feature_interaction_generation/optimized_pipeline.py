"""
Optimized Interactive Feature Generation Pipeline

This module integrates all the optimization components:
1. DAG executor with parallel processing
2. Memory-efficient model with PyArrow/Parquet and memmap
3. Content-addressed cache with dependency graph
4. Early filtering system
5. Interaction pruning system
6. Budgeted lookback optimization with TPE

Key Features:
- End-to-end optimized pipeline
- Parallel execution where safe
- Memory-efficient processing
- Intelligent caching
- Early feature filtering
- Smart interaction pruning
- Budgeted optimization
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance
)

# Import optimization components
from .dag_executor import DAGExecutor, DAGNode, NodeType, ExecutionContext
from .memory_model import MemoryEfficientProcessor, MemoryConfig
from .content_cache import ContentAddressedCache, CacheConfig, WarmStartData
from .early_filtering import EarlyFilteringSystem, EarlyFilteringConfig, FilteringResult
from .interaction_pruning import InteractionPruningSystem, InteractionPruningConfig, PruningResult
from .budgeted_optimization import BudgetedLookbackOptimizer, BudgetedOptimizationConfig, OptimizationResult

logger = logging.getLogger(__name__)


@dataclass
class OptimizedPipelineConfig:
    """Configuration for the optimized pipeline."""
    # DAG executor settings
    max_workers: int = 8
    use_processes: bool = False  # FIXED: Changed to threads to avoid pickling issues with thread locks
    
    # Memory model settings
    max_memory_gb: float = 8.0
    chunk_size_mb: float = 100.0
    use_parquet: bool = True
    use_memmap: bool = True
    
    # Cache settings
    l1_max_size_mb: float = 100.0
    l2_max_size_mb: float = 1000.0
    enable_dependency_tracking: bool = True
    
    # Early filtering settings
    downsample_ratio: float = 0.1
    variance_threshold: float = 1e-6
    top_k_per_family: int = 5
    
    # Interaction pruning settings
    max_interactions_per_domain: int = 6
    min_delta_ic: float = 0.01
    min_stability_score: float = 0.7
    
    # Budgeted optimization settings
    coarse_grid_points: int = 10
    fine_search_evals: int = 16
    early_stop_patience: int = 5
    
    # Pipeline settings
    enable_early_filtering: bool = True
    enable_interaction_pruning: bool = True
    enable_budgeted_optimization: bool = True
    enable_caching: bool = True
    enable_parallel_processing: bool = True


@dataclass
class OptimizedPipelineResult:
    """Result of the optimized pipeline."""
    features: pd.DataFrame
    feature_names: List[str]
    selected_features: List[str]
    interaction_features: pd.DataFrame
    cross_timeframe_features: pd.DataFrame
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    
    # Performance metrics
    memory_usage_mb: float = 0.0
    cache_hit_rate: float = 0.0
    parallel_efficiency: float = 0.0
    
    # Optimization results
    early_filtering_result: Optional[FilteringResult] = None
    interaction_pruning_result: Optional[PruningResult] = None
    budgeted_optimization_result: Optional[OptimizationResult] = None
    
    # Pipeline metadata
    pipeline_metadata: Dict[str, Any] = field(default_factory=dict)


class OptimizedInteractiveFeaturePipeline:
    """
    Optimized interactive feature generation pipeline.
    
    Integrates all optimization components for maximum efficiency and performance.
    """
    
    def __init__(self, config: Optional[OptimizedPipelineConfig] = None):
        """Initialize the optimized pipeline."""
        self.config = config or OptimizedPipelineConfig()
        
        # Initialize components
        self.dag_executor = DAGExecutor(
            max_workers=self.config.max_workers,
            use_processes=self.config.use_processes
        )
        
        memory_config = MemoryConfig(
            max_memory_gb=self.config.max_memory_gb,
            chunk_size_mb=self.config.chunk_size_mb,
            use_parquet=self.config.use_parquet,
            use_memmap=self.config.use_memmap
        )
        self.memory_processor = MemoryEfficientProcessor(memory_config)
        
        cache_config = CacheConfig(
            l1_max_size_mb=self.config.l1_max_size_mb,
            l2_max_size_mb=self.config.l2_max_size_mb,
            enable_dependency_tracking=self.config.enable_dependency_tracking
        )
        self.cache = ContentAddressedCache(cache_config)
        
        early_filtering_config = EarlyFilteringConfig(
            downsample_ratio=self.config.downsample_ratio,
            variance_threshold=self.config.variance_threshold,
            top_k_per_family=self.config.top_k_per_family
        )
        self.early_filtering = EarlyFilteringSystem(early_filtering_config)
        
        interaction_pruning_config = InteractionPruningConfig(
            max_interactions_per_domain=self.config.max_interactions_per_domain,
            min_delta_ic=self.config.min_delta_ic,
            min_stability_score=self.config.min_stability_score
        )
        self.interaction_pruning = InteractionPruningSystem(interaction_pruning_config)
        
        budgeted_optimization_config = BudgetedOptimizationConfig(
            coarse_grid_points=self.config.coarse_grid_points,
            fine_search_evals=self.config.fine_search_evals,
            early_stop_patience=self.config.early_stop_patience
        )
        self.budgeted_optimizer = BudgetedLookbackOptimizer(budgeted_optimization_config)
        
        tprint_success("🚀 Optimized Interactive Feature Pipeline initialized")
        tprint_info(f"📊 Max workers: {self.config.max_workers}")
        tprint_info(f"📊 Max memory: {self.config.max_memory_gb} GB")
        tprint_info(f"📊 Early filtering: {'✅' if self.config.enable_early_filtering else '❌'}")
        tprint_info(f"📊 Interaction pruning: {'✅' if self.config.enable_interaction_pruning else '❌'}")
        tprint_info(f"📊 Budgeted optimization: {'✅' if self.config.enable_budgeted_optimization else '❌'}")
    
    def _create_optimized_dag(self) -> DAGExecutor:
        """Create an optimized DAG for the pipeline."""
        executor = DAGExecutor(
            max_workers=self.config.max_workers,
            use_processes=self.config.use_processes
        )
        
        # Define optimized nodes
        nodes = [
            DAGNode(
                node_id="init",
                node_type=NodeType.INITIALIZATION,
                function=self._initialize_pipeline,
                can_parallelize=False,
                priority=10
            ),
            DAGNode(
                node_id="early_filtering",
                node_type=NodeType.FEATURE_ENGINEERING,
                function=self._early_filtering_stage,
                can_parallelize=True,
                priority=8
            ),
            DAGNode(
                node_id="feature_engineering",
                node_type=NodeType.FEATURE_ENGINEERING,
                function=self._feature_engineering_stage,
                can_parallelize=True,
                priority=8
            ),
            DAGNode(
                node_id="budgeted_optimization",
                node_type=NodeType.LOOKBACK_OPTIMIZATION,
                function=self._budgeted_optimization_stage,
                can_parallelize=True,
                priority=7
            ),
            DAGNode(
                node_id="interaction_pruning",
                node_type=NodeType.INTERACTION_GENERATION,
                function=self._interaction_pruning_stage,
                can_parallelize=True,
                priority=6
            ),
            DAGNode(
                node_id="cross_timeframe",
                node_type=NodeType.CROSS_TIMEFRAME,
                function=self._cross_timeframe_stage,
                can_parallelize=True,
                priority=6
            ),
            DAGNode(
                node_id="final_assembly",
                node_type=NodeType.FINAL_ASSEMBLY,
                function=self._final_assembly_stage,
                can_parallelize=False,
                priority=5
            ),
            DAGNode(
                node_id="validation",
                node_type=NodeType.VALIDATION,
                function=self._validation_stage,
                can_parallelize=False,
                priority=4
            )
        ]
        
        # Add nodes to executor
        for node in nodes:
            executor.add_node(node)
        
        # Add dependencies
        executor.add_dependency("early_filtering", "init")
        executor.add_dependency("feature_engineering", "init")
        executor.add_dependency("budgeted_optimization", "feature_engineering")
        executor.add_dependency("interaction_pruning", "feature_engineering")
        executor.add_dependency("cross_timeframe", "feature_engineering")
        executor.add_dependency("final_assembly", "interaction_pruning")
        executor.add_dependency("final_assembly", "cross_timeframe")
        executor.add_dependency("final_assembly", "budgeted_optimization")
        executor.add_dependency("validation", "final_assembly")
        
        return executor
    
    async def _initialize_pipeline(self, context: ExecutionContext) -> Dict[str, Any]:
        """Initialize the pipeline with memory optimization."""
        tprint_debug("🔧 Initializing optimized pipeline...")
        
        # Convert data to memory-efficient format
        if isinstance(context.data, pd.DataFrame):
            # Convert to PyArrow Table for efficient storage
            table = self.memory_processor.to_columnar(context.data, "input_data")
            context.data = table
        
        # Initialize cache
        if self.config.enable_caching:
            cache_key = self._compute_cache_key(context)
            context.pipeline_state['cache_key'] = cache_key
        
        return {
            'status': 'initialized',
            'memory_optimized': True,
            'cache_enabled': self.config.enable_caching
        }
    
    async def _early_filtering_stage(self, context: ExecutionContext) -> Dict[str, Any]:
        """Early filtering stage."""
        if not self.config.enable_early_filtering:
            return {'status': 'skipped', 'reason': 'disabled'}
        
        tprint_debug("🔍 Early filtering stage...")
        
        # Convert data back to DataFrame if needed
        if hasattr(context.data, 'to_pandas'):
            data = context.data.to_pandas()
        else:
            data = context.data
        
        # Perform early filtering
        target_column = context.pipeline_state.get('target_column', 'target')
        filtering_result = self.early_filtering.filter_features(data, target_column)
        
        # Update context with filtered features
        context.pipeline_state['early_filtering_result'] = filtering_result
        context.pipeline_state['filtered_features'] = filtering_result.selected_features
        
        return {
            'status': 'completed',
            'selected_features': len(filtering_result.selected_features),
            'rejected_features': len(filtering_result.rejected_features),
            'filtering_efficiency': filtering_result.performance_metrics.get('filtering_efficiency', 0.0)
        }
    
    async def _feature_engineering_stage(self, context: ExecutionContext) -> Dict[str, Any]:
        """Feature engineering stage with memory optimization."""
        tprint_debug("🏗️ Feature engineering stage...")
        
        # Convert data back to DataFrame if needed
        if hasattr(context.data, 'to_pandas'):
            data = context.data.to_pandas()
        else:
            data = context.data
        
        # Use filtered features if available
        if 'filtered_features' in context.pipeline_state:
            features_to_use = context.pipeline_state['filtered_features']
        else:
            features_to_use = [col for col in data.columns if col != 'target']
        
        # Generate features (simplified - would use actual feature generation)
        generated_features = data[features_to_use].copy()
        
        # Apply memory optimization
        optimized_features = self.memory_processor.optimize_dataframe_dtypes(generated_features)
        
        # Store in memory-efficient format
        if self.config.use_parquet:
            table = self.memory_processor.to_columnar(optimized_features, "generated_features")
            context.pipeline_state['generated_features'] = table
        else:
            context.pipeline_state['generated_features'] = optimized_features
        
        return {
            'status': 'completed',
            'features_generated': len(optimized_features.columns),
            'memory_optimized': True
        }
    
    async def _budgeted_optimization_stage(self, context: ExecutionContext) -> Dict[str, Any]:
        """Budgeted optimization stage."""
        if not self.config.enable_budgeted_optimization:
            return {'status': 'skipped', 'reason': 'disabled'}
        
        tprint_debug("🎯 Budgeted optimization stage...")
        
        # Get generated features
        if 'generated_features' in context.pipeline_state:
            if hasattr(context.pipeline_state['generated_features'], 'to_pandas'):
                features = context.pipeline_state['generated_features'].to_pandas()
            else:
                features = context.pipeline_state['generated_features']
        else:
            features = context.data
        
        # Get target
        target_column = context.pipeline_state.get('target_column', 'target')
        target = features[target_column] if target_column in features.columns else features.iloc[:, -1]
        
        # Get feature names
        feature_names = [col for col in features.columns if col != target_column]
        
        # Perform budgeted optimization
        optimization_result = self.budgeted_optimizer.optimize_lookbacks(
            features, target, feature_names
        )
        
        # Store result
        context.pipeline_state['budgeted_optimization_result'] = optimization_result
        
        return {
            'status': 'completed',
            'families_optimized': len(optimization_result.best_choices),
            'average_confidence': optimization_result.performance_metrics.get('average_confidence', 0.0),
            'execution_time': optimization_result.performance_metrics.get('execution_time', 0.0)
        }
    
    async def _interaction_pruning_stage(self, context: ExecutionContext) -> Dict[str, Any]:
        """Interaction pruning stage."""
        if not self.config.enable_interaction_pruning:
            return {'status': 'skipped', 'reason': 'disabled'}
        
        tprint_debug("🔗 Interaction pruning stage...")
        
        # Get generated features
        if 'generated_features' in context.pipeline_state:
            if hasattr(context.pipeline_state['generated_features'], 'to_pandas'):
                features = context.pipeline_state['generated_features'].to_pandas()
            else:
                features = context.pipeline_state['generated_features']
        else:
            features = context.data
        
        # Get target
        target_column = context.pipeline_state.get('target_column', 'target')
        target = features[target_column] if target_column in features.columns else features.iloc[:, -1]
        
        # Get feature names
        feature_names = [col for col in features.columns if col != target_column]
        
        # Perform interaction pruning
        pruning_result = self.interaction_pruning.prune_interactions_for_data(
            features, target, feature_names
        )
        
        # Store result
        context.pipeline_state['interaction_pruning_result'] = pruning_result
        
        return {
            'status': 'completed',
            'selected_interactions': len(pruning_result.selected_interactions),
            'rejected_interactions': len(pruning_result.rejected_interactions),
            'selection_rate': pruning_result.performance_metrics.get('selection_rate', 0.0)
        }
    
    async def _cross_timeframe_stage(self, context: ExecutionContext) -> Dict[str, Any]:
        """Cross-timeframe features stage."""
        tprint_debug("⏰ Cross-timeframe features stage...")
        
        # Get generated features
        if 'generated_features' in context.pipeline_state:
            if hasattr(context.pipeline_state['generated_features'], 'to_pandas'):
                features = context.pipeline_state['generated_features'].to_pandas()
            else:
                features = context.pipeline_state['generated_features']
        else:
            features = context.data
        
        # Generate cross-timeframe features (simplified)
        cross_timeframe_features = self._generate_cross_timeframe_features(features)
        
        # Store result
        context.pipeline_state['cross_timeframe_features'] = cross_timeframe_features
        
        return {
            'status': 'completed',
            'cross_timeframe_features': len(cross_timeframe_features.columns)
        }
    
    async def _final_assembly_stage(self, context: ExecutionContext) -> Dict[str, Any]:
        """Final assembly stage."""
        tprint_debug("🏁 Final assembly stage...")
        
        # Combine all features
        all_features = []
        
        # Add generated features
        if 'generated_features' in context.pipeline_state:
            if hasattr(context.pipeline_state['generated_features'], 'to_pandas'):
                features = context.pipeline_state['generated_features'].to_pandas()
            else:
                features = context.pipeline_state['generated_features']
            all_features.append(features)
        
        # Add cross-timeframe features
        if 'cross_timeframe_features' in context.pipeline_state:
            cross_tf = context.pipeline_state['cross_timeframe_features']
            all_features.append(cross_tf)
        
        # Combine features
        if all_features:
            final_features = pd.concat(all_features, axis=1)
            # Remove duplicate columns if any were introduced during concatenation
            if len(final_features.columns) != len(set(final_features.columns)):
                from src.utils.tprint import tprint_warning, tprint_debug
                tprint_warning(f"⚠️ Duplicate columns detected in final feature assembly")
                final_features = final_features.loc[:, ~final_features.columns.duplicated(keep='first')]
                tprint_debug(f"✅ Removed duplicate columns, now have {len(final_features.columns)} unique columns")
        else:
            final_features = pd.DataFrame()
        
        # Store result
        context.pipeline_state['final_features'] = final_features
        
        return {
            'status': 'completed',
            'total_features': len(final_features.columns),
            'final_shape': final_features.shape
        }
    
    async def _validation_stage(self, context: ExecutionContext) -> Dict[str, Any]:
        """Validation stage."""
        tprint_debug("✅ Validation stage...")
        
        # Get final features
        if 'final_features' in context.pipeline_state:
            final_features = context.pipeline_state['final_features']
        else:
            final_features = pd.DataFrame()
        
        # Perform validation
        validation_results = self._validate_features(final_features)
        
        return {
            'status': 'completed',
            'validation_passed': validation_results.get('passed', False),
            'quality_score': validation_results.get('quality_score', 0.0)
        }
    
    def _generate_cross_timeframe_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate cross-timeframe features."""
        cross_tf_features = {}
        
        # Timeframe aggregations
        timeframes = [5, 15, 30, 60]
        
        for tf in timeframes:
            for col in features.columns:
                if col.startswith('feature_') or col.startswith('momentum_') or col.startswith('volatility_'):
                    # Rolling aggregations
                    cross_tf_features[f'ctf_{tf}m_{col}_mean'] = features[col].rolling(tf).mean()
                    cross_tf_features[f'ctf_{tf}m_{col}_std'] = features[col].rolling(tf).std()
                    cross_tf_features[f'ctf_{tf}m_{col}_max'] = features[col].rolling(tf).max()
                    cross_tf_features[f'ctf_{tf}m_{col}_min'] = features[col].rolling(tf).min()
        
        # Create DataFrame
        if cross_tf_features:
            result = pd.DataFrame(cross_tf_features, index=features.index)
            result = result.dropna(axis=1, how='all')
            return result
        else:
            return pd.DataFrame(index=features.index)
    
    def _validate_features(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Validate generated features."""
        if features.empty:
            return {'passed': False, 'quality_score': 0.0}
        
        # Check for finite values
        finite_ratio = features.select_dtypes(include=[np.number]).apply(
            lambda x: np.isfinite(x).sum() / len(x)
        ).mean()
        
        # Check for constant features
        constant_ratio = (features.nunique() <= 1).sum() / len(features.columns)
        
        # Calculate quality score
        quality_score = finite_ratio * (1 - constant_ratio)
        
        return {
            'passed': quality_score > 0.8,
            'quality_score': quality_score,
            'finite_ratio': finite_ratio,
            'constant_ratio': constant_ratio
        }
    
    def _compute_cache_key(self, context: ExecutionContext) -> str:
        """Compute cache key for the pipeline."""
        # Simplified cache key computation
        data_hash = hash(str(context.data.shape))
        config_hash = hash(str(self.config))
        return f"pipeline_{data_hash}_{config_hash}"
    
    async def execute(self, data: pd.DataFrame, target_column: str = 'target',
                     pipeline_state: Optional[Dict[str, Any]] = None) -> OptimizedPipelineResult:
        """
        Execute the optimized pipeline.
        
        Args:
            data: Input DataFrame
            target_column: Name of target column
            pipeline_state: Optional pipeline state
            
        Returns:
            OptimizedPipelineResult with generated features
        """
        tprint_success("🚀 Starting optimized interactive feature generation pipeline")
        start_time = time.time()
        
        try:
            # Create execution context
            context = ExecutionContext(
                data=data,
                pipeline_state=pipeline_state or {},
                config=self.config.__dict__
            )
            context.pipeline_state['target_column'] = target_column
            
            # Create optimized DAG
            dag_executor = self._create_optimized_dag()
            
            # Execute DAG
            dag_results = await dag_executor.execute_dag(context)
            
            # Extract results
            final_features = context.pipeline_state.get('final_features', pd.DataFrame())
            early_filtering_result = context.pipeline_state.get('early_filtering_result')
            interaction_pruning_result = context.pipeline_state.get('interaction_pruning_result')
            budgeted_optimization_result = context.pipeline_state.get('budgeted_optimization_result')
            
            # Calculate performance metrics
            execution_time = time.time() - start_time
            memory_usage = self.memory_processor.get_memory_usage()
            cache_stats = self.cache.get_stats()
            dag_summary = dag_executor.get_execution_summary()
            
            # Create result
            result = OptimizedPipelineResult(
                features=final_features,
                feature_names=list(final_features.columns),
                selected_features=list(final_features.columns),  # Simplified
                interaction_features=pd.DataFrame(),  # Would be populated by interaction pruning
                cross_timeframe_features=context.pipeline_state.get('cross_timeframe_features', pd.DataFrame()),
                execution_time=execution_time,
                success=True,
                memory_usage_mb=memory_usage.get('rss_mb', 0.0),
                cache_hit_rate=cache_stats.get('hit_rate', 0.0),
                parallel_efficiency=dag_summary.get('parallel_efficiency', 0.0),
                early_filtering_result=early_filtering_result,
                interaction_pruning_result=interaction_pruning_result,
                budgeted_optimization_result=budgeted_optimization_result,
                pipeline_metadata={
                    'dag_summary': dag_summary,
                    'cache_stats': cache_stats,
                    'memory_usage': memory_usage
                }
            )
            
            tprint_success(f"✅ Optimized pipeline completed in {execution_time:.3f}s")
            tprint_info(f"📊 Generated {len(final_features.columns)} features")
            tprint_info(f"💾 Memory usage: {memory_usage.get('rss_mb', 0.0):.1f} MB")
            tprint_info(f"📈 Cache hit rate: {cache_stats.get('hit_rate', 0.0):.1%}")
            tprint_info(f"⚡ Parallel efficiency: {dag_summary.get('parallel_efficiency', 0.0):.1%}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_message = f"Optimized pipeline failed: {str(e)}"
            
            tprint_error(f"❌ {error_message}")
            
            return OptimizedPipelineResult(
                features=pd.DataFrame(),
                feature_names=[],
                selected_features=[],
                interaction_features=pd.DataFrame(),
                cross_timeframe_features=pd.DataFrame(),
                execution_time=execution_time,
                success=False,
                error_message=error_message
            )
    
    def cleanup(self) -> None:
        """Clean up resources."""
        tprint_debug("🧹 Cleaning up optimized pipeline...")
        
        # Cleanup components
        self.memory_processor.cleanup()
        self.cache.cleanup()
        
        tprint_success("✅ Pipeline cleanup completed")


# Convenience functions

def create_optimized_pipeline(config: Optional[OptimizedPipelineConfig] = None) -> OptimizedInteractiveFeaturePipeline:
    """Create an optimized interactive feature pipeline."""
    return OptimizedInteractiveFeaturePipeline(config)


async def execute_optimized_pipeline(data: pd.DataFrame, target_column: str = 'target',
                                   config: Optional[OptimizedPipelineConfig] = None) -> OptimizedPipelineResult:
    """Convenience function for executing the optimized pipeline."""
    pipeline = create_optimized_pipeline(config)
    try:
        return await pipeline.execute(data, target_column)
    finally:
        pipeline.cleanup()


# Example usage
if __name__ == "__main__":
    async def main():
        # Create sample data
        np.random.seed(42)
        n_samples = 1000
        
        data = pd.DataFrame({
            'target': np.random.randn(n_samples).cumsum(),
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples),
            'feature4': np.random.randn(n_samples),
            'feature5': np.random.randn(n_samples),
        })
        
        # Test optimized pipeline
        config = OptimizedPipelineConfig(
            max_workers=4,
            max_memory_gb=4.0,
            enable_early_filtering=True,
            enable_interaction_pruning=True,
            enable_budgeted_optimization=True
        )
        
        result = await execute_optimized_pipeline(data, 'target', config)
        
        print(f"Pipeline result:")
        print(f"  Success: {result.success}")
        print(f"  Execution time: {result.execution_time:.3f}s")
        print(f"  Features generated: {len(result.feature_names)}")
        print(f"  Memory usage: {result.memory_usage_mb:.1f} MB")
        print(f"  Cache hit rate: {result.cache_hit_rate:.1%}")
        print(f"  Parallel efficiency: {result.parallel_efficiency:.1%}")
        
        if result.early_filtering_result:
            print(f"  Early filtering: {result.early_filtering_result.performance_metrics}")
        
        if result.interaction_pruning_result:
            print(f"  Interaction pruning: {result.interaction_pruning_result.performance_metrics}")
        
        if result.budgeted_optimization_result:
            print(f"  Budgeted optimization: {result.budgeted_optimization_result.performance_metrics}")
    
    asyncio.run(main())