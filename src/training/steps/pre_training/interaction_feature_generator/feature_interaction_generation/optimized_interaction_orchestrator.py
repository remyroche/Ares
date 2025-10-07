"""
Optimized Interaction Feature Generation Orchestrator

This module provides a fully wired interaction feature generation pipeline that:
1. Gets features from feature_engineering bank
2. Selects features for lookback optimization
3. Generates cross-timeframe features and interaction features
4. Uses matrix operations and hardware optimization
5. Integrates with ares_launcher and sub_pipeline

Key Features:
- Extensive tprint logging throughout
- Vectorized computations using matrix_operations/
- M1 hardware optimization
- Integration with all utility modules
- Consistent with sub_pipeline architecture
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
import warnings
from pathlib import Path

# Import tprint utilities
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error, 
    tprint_debug, tprint_performance, tprint_progress
)

# Import common operations and utilities
from src.utils.common_operations import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
    optimize_memory_usage, parallel_processing_optimizer
)

# Import math validation
from src.utils.math_validation import (
    safe_divide as math_safe_divide, safe_log as math_safe_log,
    safe_sqrt as math_safe_sqrt, validate_finite as math_validate_finite
)

# Import matrix operations
try:
    from src.utils.matrix_operations import (
        get_unified_matrix_operations, get_vectorized_processing_core,
        get_batch_matrix_processor, safe_matrix_multiply,
        vectorized_rolling_features, parallel_feature_engineering,
        optimize_dataframe, get_hardware_performance_report
    )
    MATRIX_OPS_AVAILABLE = True
except ImportError as e:
    tprint_warning(f"Matrix operations not available: {e}")
    MATRIX_OPS_AVAILABLE = False

# Import ML common utilities
try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
        BayesianTPEOptimizer, OptimizationConfig
    )
    from src.utils.ml_common.cross_validation import PurgedKFold
    from src.utils.ml_common.feature_selection import FeatureSelector
    ML_COMMON_AVAILABLE = True
except ImportError as e:
    tprint_warning(f"ML common utilities not available: {e}")
    ML_COMMON_AVAILABLE = False

# Import data utilities
try:
    from src.utils.data.data_loader import DataLoader
    from src.utils.data.data_validation import DataValidator
    from src.utils.kline_parquet import KlineParquetLoader
    from src.utils.serialization_utils import save_pickle, load_pickle
    DATA_UTILS_AVAILABLE = True
except ImportError as e:
    tprint_warning(f"Data utilities not available: {e}")
    DATA_UTILS_AVAILABLE = False

# Import feature engineering components
from .feature_engineering.assembly_dag import AssemblyDAG, AssemblyConfig, AssemblyResult
from .feature_engineering.lookback_selection import LookbackSelector, create_feature_families
from .feature_engineering.transforms import TransformRouter, create_default_transform_config
from .feature_engineering.interactions import InteractionEngine, create_default_interaction_config
from .feature_engineering.feature_registry import FeatureRegistry, FeatureFamily

# Import orchestrator components
from .orchestrator import LookbackOptimizationOrchestrator, OptimizationResult
from .config import LookbackOptimizationConfig, create_default_config

# Setup logging
logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Pipeline execution stages."""
    INITIALIZATION = "initialization"
    FEATURE_ENGINEERING = "feature_engineering"
    LOOKBACK_OPTIMIZATION = "lookback_optimization"
    TRANSFORM_APPLICATION = "transform_application"
    INTERACTION_GENERATION = "interaction_generation"
    CROSS_TIMEFRAME = "cross_timeframe"
    FINAL_ASSEMBLY = "final_assembly"
    VALIDATION = "validation"
    COMPLETION = "completion"


@dataclass
class OptimizedInteractionConfig:
    """Configuration for optimized interaction feature generation."""
    # Pipeline configuration
    symbol: str = "ETHUSDT"
    exchange: str = "binance"
    timeframe: str = "15m"
    data_dir: str = "historical_data"
    
    # Feature engineering configuration
    feature_budget_pre: int = 120
    feature_budget_post: Tuple[int, int] = (30, 60)
    interactions_cap: int = 15
    transforms_per_parent: int = 1
    lookback_ceiling_minutes: int = 120
    latency_budget_ms: int = 50
    
    # Optimization configuration
    enable_matrix_optimization: bool = True
    enable_hardware_optimization: bool = True
    enable_parallel_processing: bool = True
    max_workers: int = 4
    batch_size: int = 1000
    
    # Lookback optimization configuration
    lookback_config: Optional[LookbackOptimizationConfig] = None
    
    # Validation configuration
    enable_validation: bool = True
    validation_threshold: float = 0.02
    
    # Logging configuration
    verbose_logging: bool = True
    log_performance: bool = True
    
    def __post_init__(self):
        if self.lookback_config is None:
            self.lookback_config = create_default_config()


@dataclass
class OptimizedInteractionResult:
    """Result of optimized interaction feature generation."""
    # Core results
    features: pd.DataFrame
    feature_names: List[str]
    selected_features: List[str]
    interaction_features: pd.DataFrame
    cross_timeframe_features: pd.DataFrame
    
    # Pipeline metadata
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    
    # Performance metrics
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    gpu_usage_percent: float = 0.0
    
    # Stage results
    stage_results: Dict[PipelineStage, Dict[str, Any]] = field(default_factory=dict)
    
    # Artifacts
    artifacts: Dict[str, Any] = field(default_factory=dict)


class OptimizedInteractionOrchestrator:
    """Main orchestrator for optimized interaction feature generation."""
    
    def __init__(self, config: OptimizedInteractionConfig):
        self.config = config
        self.logger = logger.getChild('OptimizedInteractionOrchestrator')
        
        # Initialize components
        self._initialize_components()
        
        # Performance tracking
        self.performance_metrics = {}
        self.stage_start_times = {}
        
        tprint_success("🚀 Optimized Interaction Orchestrator initialized")
        tprint_info(f"📊 Symbol: {config.symbol}, Exchange: {config.exchange}")
        tprint_info(f"⏰ Timeframe: {config.timeframe}")
        tprint_info(f"🔧 Matrix ops: {MATRIX_OPS_AVAILABLE}, ML common: {ML_COMMON_AVAILABLE}")
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        tprint_debug("🔧 Initializing pipeline components...")
        
        # Feature registry
        self.feature_registry = FeatureRegistry()
        tprint_debug(f"✅ Feature registry initialized with {len(self.feature_registry.get_all_features())} features")
        
        # Assembly DAG
        assembly_config = AssemblyConfig(
            feature_budget_pre=self.config.feature_budget_pre,
            feature_budget_post=self.config.feature_budget_post,
            interactions_cap=self.config.interactions_cap,
            transforms_per_parent=self.config.transforms_per_parent,
            lookback_ceiling_minutes=self.config.lookback_ceiling_minutes,
            latency_budget_ms=self.config.latency_budget_ms
        )
        self.assembly_dag = AssemblyDAG(assembly_config)
        tprint_debug("✅ Assembly DAG initialized")
        
        # Lookback optimization orchestrator
        self.lookback_orchestrator = LookbackOptimizationOrchestrator(self.config.lookback_config)
        tprint_debug("✅ Lookback optimization orchestrator initialized")
        
        # Lookback selector
        self.lookback_selector = LookbackSelector()
        tprint_debug("✅ Lookback selector initialized")
        
        # Matrix operations (if available)
        if MATRIX_OPS_AVAILABLE:
            self.matrix_ops = get_unified_matrix_operations()
            self.vectorized_core = get_vectorized_processing_core()
            self.batch_processor = get_batch_matrix_processor()
            tprint_debug("✅ Matrix operations initialized")
        else:
            self.matrix_ops = None
            self.vectorized_core = None
            self.batch_processor = None
            tprint_warning("⚠️ Matrix operations not available - using fallback methods")
        
        # Hardware optimizers (if available)
        try:
            self.m1_gpu_manager = get_m1_gpu_manager()
            self.m1_memory_optimizer = get_m1_memory_optimizer()
            self.m1_cpu_optimizer = get_m1_cpu_optimizer()
            tprint_debug("✅ M1 hardware optimizers initialized")
        except Exception as e:
            tprint_warning(f"⚠️ M1 hardware optimizers not available: {e}")
            self.m1_gpu_manager = None
            self.m1_memory_optimizer = None
            self.m1_cpu_optimizer = None
        
        # ML common utilities (if available)
        if ML_COMMON_AVAILABLE:
            self.bayesian_optimizer = BayesianTPEOptimizer()
            self.feature_selector = FeatureSelector()
            tprint_debug("✅ ML common utilities initialized")
        else:
            self.bayesian_optimizer = None
            self.feature_selector = None
            tprint_warning("⚠️ ML common utilities not available")
        
        # Data utilities (if available)
        if DATA_UTILS_AVAILABLE:
            self.data_loader = DataLoader()
            self.data_validator = DataValidator()
            self.kline_loader = KlineParquetLoader()
            tprint_debug("✅ Data utilities initialized")
        else:
            self.data_loader = None
            self.data_validator = None
            self.kline_loader = None
            tprint_warning("⚠️ Data utilities not available")
        
        tprint_success("✅ All components initialized successfully")
    
    async def generate_features(self, 
                              training_input: Dict[str, Any],
                              pipeline_state: Dict[str, Any]) -> OptimizedInteractionResult:
        """Generate optimized interaction features."""
        start_time = time.time()
        tprint_success("🚀 Starting optimized interaction feature generation")
        
        try:
            # Stage 1: Initialization
            await self._stage_initialization(training_input, pipeline_state)
            
            # Stage 2: Feature Engineering
            feature_engineering_result = await self._stage_feature_engineering(training_input, pipeline_state)
            
            # Stage 3: Lookback Optimization
            lookback_result = await self._stage_lookback_optimization(feature_engineering_result, pipeline_state)
            
            # Stage 4: Transform Application
            transform_result = await self._stage_transform_application(lookback_result, pipeline_state)
            
            # Stage 5: Interaction Generation
            interaction_result = await self._stage_interaction_generation(transform_result, pipeline_state)
            
            # Stage 6: Cross-timeframe Features
            cross_timeframe_result = await self._stage_cross_timeframe_features(interaction_result, pipeline_state)
            
            # Stage 7: Final Assembly
            final_result = await self._stage_final_assembly(cross_timeframe_result, pipeline_state)
            
            # Stage 8: Validation
            validation_result = await self._stage_validation(final_result, pipeline_state)
            
            # Stage 9: Completion
            completion_result = await self._stage_completion(validation_result, pipeline_state)
            
            execution_time = time.time() - start_time
            tprint_success(f"✅ Feature generation completed in {execution_time:.3f}s")
            
            return completion_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_message = f"Feature generation failed: {str(e)}"
            tprint_error(f"❌ {error_message}")
            self.logger.error(f"Feature generation failed: {error_message}", exc_info=True)
            
            return OptimizedInteractionResult(
                features=pd.DataFrame(),
                feature_names=[],
                selected_features=[],
                interaction_features=pd.DataFrame(),
                cross_timeframe_features=pd.DataFrame(),
                execution_time=execution_time,
                success=False,
                error_message=error_message
            )
    
    async def _stage_initialization(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 1: Initialize pipeline and validate inputs."""
        stage_start = time.time()
        tprint_info("🔧 Stage 1: Initialization")
        
        try:
            # Validate inputs
            if not training_input:
                raise ValueError("No training input provided")
            
            if not pipeline_state:
                raise ValueError("No pipeline state provided")
            
            # Extract data
            data = training_input.get('data')
            if data is None:
                raise ValueError("No data provided in training input")
            
            # Validate data
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Data must be a pandas DataFrame")
            
            if len(data) < 100:
                raise ValueError(f"Insufficient data: {len(data)} < 100 rows")
            
            # Check required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = set(required_columns) - set(data.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Initialize performance tracking
            self.performance_metrics = {
                'total_rows': len(data),
                'total_columns': len(data.columns),
                'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
                'data_quality_score': self._calculate_data_quality_score(data)
            }
            
            # Hardware optimization setup
            if self.m1_memory_optimizer:
                self.m1_memory_optimizer.optimize_dataframe(data)
                tprint_debug("✅ M1 memory optimization applied")
            
            stage_time = time.time() - stage_start
            tprint_performance("Initialization", stage_time)
            
            result = {
                'data': data,
                'performance_metrics': self.performance_metrics,
                'stage_time': stage_time,
                'success': True
            }
            
            self.stage_results[PipelineStage.INITIALIZATION] = result
            return result
            
        except Exception as e:
            stage_time = time.time() - stage_start
            tprint_error(f"❌ Initialization failed: {e}")
            raise
    
    async def _stage_feature_engineering(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 2: Generate parent features from market data."""
        stage_start = time.time()
        tprint_info("🔧 Stage 2: Feature Engineering")
        
        try:
            data = training_input['data']
            
            # Extract targets if available
            targets = training_input.get('targets', {})
            
            # Use assembly DAG to build parent features
            tprint_debug("Building parent features using assembly DAG...")
            assembly_result = self.assembly_dag.assemble(data, targets)
            
            if assembly_result.status.value != 'completed':
                raise ValueError(f"Assembly DAG failed: {assembly_result.status.value}")
            
            # Extract features
            parent_features = assembly_result.features
            feature_names = assembly_result.feature_names
            
            tprint_info(f"✅ Generated {len(feature_names)} parent features")
            tprint_debug(f"Feature families: {list(set([name.split('/')[1] for name in feature_names if '/' in name]))}")
            
            # Apply matrix optimization if available
            if self.vectorized_core and MATRIX_OPS_AVAILABLE:
                tprint_debug("Applying vectorized processing optimization...")
                parent_features = self.vectorized_core.optimize_dataframe_for_processing(parent_features)
                tprint_debug("✅ Vectorized processing optimization applied")
            
            # Memory optimization
            if self.m1_memory_optimizer:
                parent_features = self.m1_memory_optimizer.optimize_dataframe_memory(parent_features)
                tprint_debug("✅ Memory optimization applied")
            
            stage_time = time.time() - stage_start
            tprint_performance("Feature Engineering", stage_time)
            
            result = {
                'parent_features': parent_features,
                'feature_names': feature_names,
                'assembly_result': assembly_result,
                'stage_time': stage_time,
                'success': True
            }
            
            self.stage_results[PipelineStage.FEATURE_ENGINEERING] = result
            return result
            
        except Exception as e:
            stage_time = time.time() - stage_start
            tprint_error(f"❌ Feature engineering failed: {e}")
            raise
    
    async def _stage_lookback_optimization(self, feature_result: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 3: Optimize lookback periods for features."""
        stage_start = time.time()
        tprint_info("🔧 Stage 3: Lookback Optimization")
        
        try:
            parent_features = feature_result['parent_features']
            feature_names = feature_result['feature_names']
            
            # Extract targets
            targets = pipeline_state.get('targets', {})
            if not targets:
                tprint_warning("⚠️ No targets available for lookback optimization")
                # Create dummy targets for optimization
                targets = {1: pd.Series(0, index=parent_features.index)}
            
            # Create feature families
            tprint_debug("Creating feature families...")
            feature_families = create_feature_families(feature_names)
            tprint_debug(f"Created {len(feature_families)} feature families")
            
            # Use lookback selector
            tprint_debug("Selecting optimal lookbacks...")
            lookback_choices = self.lookback_selector.select_lookbacks(
                parent_features, 
                targets.get(1, pd.Series(0, index=parent_features.index)),
                feature_families
            )
            
            tprint_info(f"✅ Selected lookbacks for {len(lookback_choices)} feature families")
            
            # Log lookback choices
            for family, choice in lookback_choices.items():
                tprint_debug(f"  {family}: {choice.selected_lookback} (confidence: {choice.confidence_score:.3f})")
            
            stage_time = time.time() - stage_start
            tprint_performance("Lookback Optimization", stage_time)
            
            result = {
                'lookback_choices': lookback_choices,
                'feature_families': feature_families,
                'stage_time': stage_time,
                'success': True
            }
            
            self.stage_results[PipelineStage.LOOKBACK_OPTIMIZATION] = result
            return result
            
        except Exception as e:
            stage_time = time.time() - stage_start
            tprint_error(f"❌ Lookback optimization failed: {e}")
            raise
    
    async def _stage_transform_application(self, lookback_result: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 4: Apply transforms to parent features."""
        stage_start = time.time()
        tprint_info("🔧 Stage 4: Transform Application")
        
        try:
            # Get parent features from previous stage
            feature_result = self.stage_results[PipelineStage.FEATURE_ENGINEERING]
            parent_features = feature_result['parent_features']
            feature_names = feature_result['feature_names']
            
            # Create transform configuration
            tprint_debug("Creating transform configuration...")
            transform_config = create_default_transform_config(feature_names)
            tprint_debug(f"Created transform config for {len(transform_config)} features")
            
            # Initialize transform router
            transform_router = TransformRouter(transform_config)
            
            # Split data for transform fitting
            split_idx = int(len(parent_features) * 0.8)
            train_features = parent_features.iloc[:split_idx]
            val_features = parent_features.iloc[split_idx:]
            
            tprint_debug(f"Split data: train={len(train_features)}, val={len(val_features)}")
            
            # Apply transforms
            tprint_debug("Applying transforms...")
            transformed_results = transform_router.fit_transform(train_features, val_features)
            
            # Combine transformed features
            all_transformed = []
            for feature_name, results in transformed_results.items():
                all_transformed.append(results['train'])
            
            if all_transformed:
                transformed_features = pd.concat(all_transformed, axis=1)
                tprint_info(f"✅ Generated {len(transformed_features.columns)} transformed features")
            else:
                transformed_features = pd.DataFrame(index=parent_features.index)
                tprint_warning("⚠️ No transformed features generated")
            
            # Apply winsorization
            tprint_debug("Applying winsorization...")
            from .feature_engineering.transforms import apply_winsorization
            transformed_features = apply_winsorization(transformed_features)
            
            # Matrix optimization
            if self.vectorized_core and MATRIX_OPS_AVAILABLE:
                tprint_debug("Applying matrix optimization to transformed features...")
                transformed_features = self.vectorized_core.optimize_dataframe_for_processing(transformed_features)
            
            stage_time = time.time() - stage_start
            tprint_performance("Transform Application", stage_time)
            
            result = {
                'transformed_features': transformed_features,
                'transform_router': transform_router,
                'transform_config': transform_config,
                'stage_time': stage_time,
                'success': True
            }
            
            self.stage_results[PipelineStage.TRANSFORM_APPLICATION] = result
            return result
            
        except Exception as e:
            stage_time = time.time() - stage_start
            tprint_error(f"❌ Transform application failed: {e}")
            raise
    
    async def _stage_interaction_generation(self, transform_result: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 5: Generate interaction features."""
        stage_start = time.time()
        tprint_info("🔧 Stage 5: Interaction Generation")
        
        try:
            transformed_features = transform_result['transformed_features']
            
            # Create interaction configuration
            tprint_debug("Creating interaction configuration...")
            interaction_config = create_default_interaction_config()
            tprint_debug(f"Created interaction config for {len(interaction_config)} interactions")
            
            # Initialize interaction engine
            interaction_engine = InteractionEngine(interaction_config)
            
            # Extract patch features if available
            patch_features = pipeline_state.get('patch_features', {})
            if patch_features:
                tprint_debug(f"Using {len(patch_features)} patch features")
            else:
                tprint_debug("No patch features available")
            
            # Generate interactions
            tprint_debug("Generating interactions...")
            interactions = interaction_engine.build_interactions(transformed_features, patch_features)
            
            tprint_info(f"✅ Generated {len(interactions.columns)} interaction features")
            
            # Log interaction types
            interaction_types = {}
            for col in interactions.columns:
                interaction_type = col.split('/')[1] if '/' in col else 'unknown'
                interaction_types[interaction_type] = interaction_types.get(interaction_type, 0) + 1
            
            for interaction_type, count in interaction_types.items():
                tprint_debug(f"  {interaction_type}: {count} features")
            
            # Matrix optimization
            if self.vectorized_core and MATRIX_OPS_AVAILABLE:
                tprint_debug("Applying matrix optimization to interactions...")
                interactions = self.vectorized_core.optimize_dataframe_for_processing(interactions)
            
            stage_time = time.time() - stage_start
            tprint_performance("Interaction Generation", stage_time)
            
            result = {
                'interactions': interactions,
                'interaction_engine': interaction_engine,
                'interaction_config': interaction_config,
                'stage_time': stage_time,
                'success': True
            }
            
            self.stage_results[PipelineStage.INTERACTION_GENERATION] = result
            return result
            
        except Exception as e:
            stage_time = time.time() - stage_start
            tprint_error(f"❌ Interaction generation failed: {e}")
            raise
    
    async def _stage_cross_timeframe_features(self, interaction_result: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 6: Generate cross-timeframe features."""
        stage_start = time.time()
        tprint_info("🔧 Stage 6: Cross-timeframe Features")
        
        try:
            # Get transformed features and interactions
            transform_result = self.stage_results[PipelineStage.TRANSFORM_APPLICATION]
            transformed_features = transform_result['transformed_features']
            interactions = interaction_result['interactions']
            
            # Combine features
            all_features = pd.concat([transformed_features, interactions], axis=1)
            
            # Generate cross-timeframe features
            tprint_debug("Generating cross-timeframe features...")
            cross_timeframe_features = self._generate_cross_timeframe_features(all_features)
            
            tprint_info(f"✅ Generated {len(cross_timeframe_features.columns)} cross-timeframe features")
            
            # Matrix optimization
            if self.vectorized_core and MATRIX_OPS_AVAILABLE:
                tprint_debug("Applying matrix optimization to cross-timeframe features...")
                cross_timeframe_features = self.vectorized_core.optimize_dataframe_for_processing(cross_timeframe_features)
            
            stage_time = time.time() - stage_start
            tprint_performance("Cross-timeframe Features", stage_time)
            
            result = {
                'cross_timeframe_features': cross_timeframe_features,
                'all_features': all_features,
                'stage_time': stage_time,
                'success': True
            }
            
            self.stage_results[PipelineStage.CROSS_TIMEFRAME] = result
            return result
            
        except Exception as e:
            stage_time = time.time() - stage_start
            tprint_error(f"❌ Cross-timeframe features failed: {e}")
            raise
    
    async def _stage_final_assembly(self, cross_timeframe_result: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 7: Final feature assembly and selection."""
        stage_start = time.time()
        tprint_info("🔧 Stage 7: Final Assembly")
        
        try:
            all_features = cross_timeframe_result['all_features']
            cross_timeframe_features = cross_timeframe_result['cross_timeframe_features']
            
            # Combine all features
            final_features = pd.concat([all_features, cross_timeframe_features], axis=1)
            tprint_info(f"✅ Assembled {len(final_features.columns)} total features")
            
            # Feature selection
            tprint_debug("Performing feature selection...")
            selected_features = self._select_features(final_features, pipeline_state)
            tprint_info(f"✅ Selected {len(selected_features)} features within budget")
            
            # Create final feature matrix
            final_feature_matrix = final_features[selected_features] if selected_features else final_features
            
            # Memory optimization
            if self.m1_memory_optimizer:
                final_feature_matrix = self.m1_memory_optimizer.optimize_dataframe_memory(final_feature_matrix)
            
            stage_time = time.time() - stage_start
            tprint_performance("Final Assembly", stage_time)
            
            result = {
                'final_features': final_feature_matrix,
                'selected_features': selected_features,
                'all_feature_names': final_features.columns.tolist(),
                'stage_time': stage_time,
                'success': True
            }
            
            self.stage_results[PipelineStage.FINAL_ASSEMBLY] = result
            return result
            
        except Exception as e:
            stage_time = time.time() - stage_start
            tprint_error(f"❌ Final assembly failed: {e}")
            raise
    
    async def _stage_validation(self, final_result: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 8: Validate generated features."""
        stage_start = time.time()
        tprint_info("🔧 Stage 8: Validation")
        
        try:
            final_features = final_result['final_features']
            selected_features = final_result['selected_features']
            
            # Data quality validation
            tprint_debug("Performing data quality validation...")
            validation_results = self._validate_features(final_features)
            
            # Performance validation
            tprint_debug("Performing performance validation...")
            performance_results = self._validate_performance(final_features)
            
            # Memory validation
            memory_usage_mb = final_features.memory_usage(deep=True).sum() / 1024 / 1024
            tprint_info(f"✅ Memory usage: {memory_usage_mb:.2f} MB")
            
            stage_time = time.time() - stage_start
            tprint_performance("Validation", stage_time)
            
            result = {
                'validation_results': validation_results,
                'performance_results': performance_results,
                'memory_usage_mb': memory_usage_mb,
                'stage_time': stage_time,
                'success': True
            }
            
            self.stage_results[PipelineStage.VALIDATION] = result
            return result
            
        except Exception as e:
            stage_time = time.time() - stage_start
            tprint_error(f"❌ Validation failed: {e}")
            raise
    
    async def _stage_completion(self, validation_result: Dict[str, Any], pipeline_state: Dict[str, Any]) -> OptimizedInteractionResult:
        """Stage 9: Complete pipeline and return results."""
        stage_start = time.time()
        tprint_info("🔧 Stage 9: Completion")
        
        try:
            # Get results from all stages
            final_result = self.stage_results[PipelineStage.FINAL_ASSEMBLY]
            interaction_result = self.stage_results[PipelineStage.INTERACTION_GENERATION]
            cross_timeframe_result = self.stage_results[PipelineStage.CROSS_TIMEFRAME]
            
            # Extract features
            final_features = final_result['final_features']
            selected_features = final_result['selected_features']
            all_feature_names = final_result['all_feature_names']
            interactions = interaction_result['interactions']
            cross_timeframe_features = cross_timeframe_result['cross_timeframe_features']
            
            # Calculate performance metrics
            total_execution_time = sum(result['stage_time'] for result in self.stage_results.values())
            memory_usage_mb = validation_result.get('memory_usage_mb', 0.0)
            
            # Create artifacts
            artifacts = {
                'stage_results': self.stage_results,
                'performance_metrics': self.performance_metrics,
                'config': self.config,
                'feature_registry': self.feature_registry,
                'assembly_result': self.stage_results[PipelineStage.FEATURE_ENGINEERING].get('assembly_result')
            }
            
            stage_time = time.time() - stage_start
            tprint_performance("Completion", stage_time)
            
            # Final success message
            tprint_success("🎉 Optimized interaction feature generation completed successfully!")
            tprint_info(f"📊 Generated {len(all_feature_names)} total features")
            tprint_info(f"🎯 Selected {len(selected_features)} features")
            tprint_info(f"🔗 Generated {len(interactions.columns)} interactions")
            tprint_info(f"⏰ Generated {len(cross_timeframe_features.columns)} cross-timeframe features")
            tprint_info(f"💾 Memory usage: {memory_usage_mb:.2f} MB")
            tprint_info(f"⏱️ Total execution time: {total_execution_time:.3f}s")
            
            return OptimizedInteractionResult(
                features=final_features,
                feature_names=all_feature_names,
                selected_features=selected_features,
                interaction_features=interactions,
                cross_timeframe_features=cross_timeframe_features,
                execution_time=total_execution_time,
                success=True,
                memory_usage_mb=memory_usage_mb,
                stage_results=self.stage_results,
                artifacts=artifacts
            )
            
        except Exception as e:
            stage_time = time.time() - stage_start
            tprint_error(f"❌ Completion failed: {e}")
            raise
    
    def _generate_cross_timeframe_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate cross-timeframe features."""
        tprint_debug("Generating cross-timeframe features...")
        
        cross_timeframe_features = {}
        
        # Timeframe aggregations
        timeframes = [5, 15, 30, 60]  # minutes
        
        for tf in timeframes:
            # Rolling aggregations
            for col in features.columns:
                if col.startswith('t/'):  # Only transform features
                    # Rolling mean
                    cross_timeframe_features[f'ctf_{tf}m_{col}_mean'] = features[col].rolling(tf).mean()
                    
                    # Rolling std
                    cross_timeframe_features[f'ctf_{tf}m_{col}_std'] = features[col].rolling(tf).std()
                    
                    # Rolling max
                    cross_timeframe_features[f'ctf_{tf}m_{col}_max'] = features[col].rolling(tf).max()
                    
                    # Rolling min
                    cross_timeframe_features[f'ctf_{tf}m_{col}_min'] = features[col].rolling(tf).min()
        
        # Create DataFrame
        if cross_timeframe_features:
            result = pd.DataFrame(cross_timeframe_features, index=features.index)
            # Remove columns with all NaN values
            result = result.dropna(axis=1, how='all')
            return result
        else:
            return pd.DataFrame(index=features.index)
    
    def _select_features(self, features: pd.DataFrame, pipeline_state: Dict[str, Any]) -> List[str]:
        """Select features within budget constraints."""
        tprint_debug(f"Selecting features from {len(features.columns)} candidates...")
        
        if len(features.columns) <= self.config.feature_budget_pre:
            tprint_debug("All features within budget, selecting all")
            return features.columns.tolist()
        
        # Extract targets for selection
        targets = pipeline_state.get('targets', {})
        if not targets:
            tprint_warning("No targets available for feature selection, using random selection")
            return features.columns.tolist()[:self.config.feature_budget_pre]
        
        target_series = targets.get(1, pd.Series(0, index=features.index))
        
        # Calculate correlations
        correlations = []
        for col in features.columns:
            if not features[col].isna().all() and not target_series.isna().all():
                try:
                    corr = features[col].corr(target_series)
                    if not pd.isna(corr):
                        correlations.append((col, abs(corr)))
                except Exception as e:
                    tprint_debug(f"Failed to calculate correlation for {col}: {e}")
                    continue
        
        # Sort by correlation strength
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        # Select top features within budget
        selected = [col for col, _ in correlations[:self.config.feature_budget_pre]]
        
        tprint_debug(f"Selected {len(selected)} features based on correlation")
        return selected
    
    def _validate_features(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Validate generated features."""
        validation_results = {
            'total_features': len(features.columns),
            'finite_features': 0,
            'infinite_features': 0,
            'nan_features': 0,
            'constant_features': 0,
            'quality_score': 0.0
        }
        
        for col in features.columns:
            col_data = features[col].dropna()
            
            if len(col_data) == 0:
                validation_results['nan_features'] += 1
                continue
            
            # Check for finite values
            finite_count = np.isfinite(col_data).sum()
            if finite_count == len(col_data):
                validation_results['finite_features'] += 1
            else:
                validation_results['infinite_features'] += 1
            
            # Check for constant features
            if col_data.nunique() <= 1:
                validation_results['constant_features'] += 1
        
        # Calculate quality score
        total_features = validation_results['total_features']
        if total_features > 0:
            validation_results['quality_score'] = validation_results['finite_features'] / total_features
        
        return validation_results
    
    def _validate_performance(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Validate performance characteristics."""
        performance_results = {
            'memory_usage_mb': features.memory_usage(deep=True).sum() / 1024 / 1024,
            'shape': features.shape,
            'dtypes': features.dtypes.value_counts().to_dict()
        }
        
        return performance_results
    
    def _calculate_data_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate data quality score."""
        if len(data) == 0:
            return 0.0
        
        # Check for missing values
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        
        # Check for infinite values
        infinite_ratio = np.isinf(data.select_dtypes(include=[np.number])).sum().sum() / (len(data) * len(data.columns))
        
        # Calculate quality score (higher is better)
        quality_score = 1.0 - missing_ratio - infinite_ratio
        return max(0.0, quality_score)


# Convenience function for easy integration
async def generate_optimized_interaction_features(
    training_input: Dict[str, Any],
    pipeline_state: Dict[str, Any],
    config: Optional[OptimizedInteractionConfig] = None
) -> OptimizedInteractionResult:
    """
    Generate optimized interaction features with the given configuration.
    
    Args:
        training_input: Input data for feature generation
        pipeline_state: Current pipeline state
        config: Configuration for feature generation
        
    Returns:
        OptimizedInteractionResult with generated features
    """
    if config is None:
        config = OptimizedInteractionConfig()
    
    orchestrator = OptimizedInteractionOrchestrator(config)
    return await orchestrator.generate_features(training_input, pipeline_state)