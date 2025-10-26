"""
Feature Generation Final Feature Selection Step

This step performs final feature selection from all previously generated and selected features.
It creates multiple optimized feature sets (60, 50, 40 features) for different model configurations
and generates comprehensive SHAP values and selection metadata.

Features:
- Combines features from interaction generation steps
- Performs final feature ranking and selection
- Creates multiple feature sets (60, 50, 40 features)
- Generates SHAP values for interpretability
- Comprehensive selection metadata and reporting
"""

import asyncio
import logging
import warnings
import pandas as pd
import numpy as np

# Fix NumPy compatibility for older libraries
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'complex'):
    np.complex = complex
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime
from pathlib import Path
import json

# Import BaseStep and step registry
from src.training.steps.base_step import BaseStep

# Import feature selection component
from src.training.steps.pre_training.components.final_feature_selection import (
    FinalFeatureSelectionConfig,
    FinalFeatureSelectionComponent
)

# Import VectorBT optimization tools
from src.feature_generation.utils.vectorbt_rolling_optimizer import (
    VectorBTRollingOptimizer,
    get_vectorbt_rolling_optimizer
)

# Import unified vectorization manager
from src.feature_generation.utils.unified_vectorization_manager import (
    UnifiedVectorizationManager,
    VectorizationConfig,
    get_unified_vectorization_manager
)

# Note: Hardware optimization components are optional for feature selection

# Import hardware optimization tools
from src.utils.hardware.unified_hardware_manager import (
    UnifiedHardwareManager,
    HardwareConfig,
    WorkloadType,
    OptimizationLevel
)

# Import additional hardware optimization components
from src.utils.hardware.adaptive_optimization_engine import (
    AdaptiveOptimizationEngine,
    LearningAlgorithm
)

# Import CMI complementarity components for Tactician mode
try:
    # These modules don't exist yet - placeholder for future implementation
    CMIComplementarityScorer = None
    CMIComplementarityConfig = None
    AnalystSideInfoHandler = None
    CMI_COMPLEMENTARITY_AVAILABLE = False
    print("⚠️ CMI complementarity components not available - placeholder implementation")
except ImportError:
    CMI_COMPLEMENTARITY_AVAILABLE = False
    CMIComplementarityScorer = None
    CMIComplementarityConfig = None
    AnalystSideInfoHandler = None

# Import OptimizationStrategy from the correct location
from src.feature_generation.core.optimization_strategies import (
    OptimizationStrategy,
    ConservativeOptimizationStrategy,
    BalancedOptimizationStrategy,
    AggressiveOptimizationStrategy
)

from src.utils.hardware.advanced_cpu_optimizer import (
    AdvancedM1CPUOptimizer,
    WorkloadProfile,
    CoreType
)

from src.utils.hardware.enhanced_gpu_manager import (
    EnhancedM1GPUManager,
    GPUOperationType
)

from src.utils.hardware.advanced_memory_optimizer import (
    AdvancedM1MemoryOptimizer,
    MemoryStrategy
)

# Import utilities
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning, tprint_error, configure_tprint, TPrintConfig, LogLevel
from src.utils.artifact_manager import ArtifactManager

# Configure tprint for minimal mode to reduce overhead
configure_tprint(TPrintConfig(
    use_colors=False,
    output_to_file=False,
    log_to_python_logger=False,
    integrate_with_logging=False,
    min_log_level=LogLevel.INFO,
    enable_lazy_evaluation=True,
    cache_timestamps=True
))

logger = logging.getLogger(__name__)


class FeatureGenerationFinalFeatureSelectionStep(BaseStep):
    """
    Final feature selection step for the feature generation pipeline.

    This step combines all previously selected features and performs final selection
    to create optimized feature sets for model training.
    """

    def __init__(self, step_name: str = "feature_generation_final_feature_selection_step"):
        """Initialize the final feature selection step."""
        super().__init__(step_name)
        self.selection_component: Optional[FinalFeatureSelectionComponent] = None
        
        # Initialize VectorBT optimization components
        self.vectorization_manager: Optional[UnifiedVectorizationManager] = None
        self.rolling_optimizer: Optional[VectorBTRollingOptimizer] = None
        self.optimization_enabled: bool = True
        
        # Initialize hardware optimization components
        self.hardware_manager: Optional[UnifiedHardwareManager] = None
        self.adaptive_engine: Optional[AdaptiveOptimizationEngine] = None
        self.cpu_optimizer: Optional[AdvancedM1CPUOptimizer] = None
        self.gpu_manager: Optional[EnhancedM1GPUManager] = None
        self.memory_optimizer: Optional[AdvancedM1MemoryOptimizer] = None
        self.hardware_optimization_enabled: bool = True
        
        # Initialize CMI complementarity components for Tactician mode
        if CMI_COMPLEMENTARITY_AVAILABLE:
            self.cmi_config = CMIComplementarityConfig(
                per_family_budget=(5, 15),
                upstream_multiplier=3,
                max_total_features=60,
                enable_regime_awareness=True,
                compute_timeout_seconds=300.0,
                enable_synergy=True,
                beta_synergy=0.25
            )
            self.cmi_scorer = CMIComplementarityScorer(self.cmi_config)
            self.analyst_handler = AnalystSideInfoHandler()
            tprint_info("✅ CMI complementarity components initialized for final feature selection")
        else:
            self.cmi_scorer = None
            self.analyst_handler = None
            tprint_warning("⚠️ CMI complementarity components not available")

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute final feature selection.

        Args:
            config: Configuration dictionary containing:
                - symbol: Trading symbol
                - exchange: Exchange name
                - timeframe: Timeframe
                - execution_mode: Execution mode (light, full, etc.)
                - feature_set_sizes: List of feature set sizes to create [60, 50, 40]
                - selection_config: Optional selection configuration overrides

        Returns:
            Dict containing execution results and artifacts
        """
        try:
            tprint_info(f"🎯 Starting {self.step_name} execution...")

            # Get required data from previous steps
            # Look for artifacts created by labeling integration step
            labeled_df = self._get_artifact('labeled_data')
            targets = self._get_artifact('labeling_metadata')

            if labeled_df is None or targets is None:
                raise ValueError("Required artifacts 'labeled_data' and 'labeling_metadata' not found")

            # Get features from previous steps
            features_data = self._collect_features_from_previous_steps()

            # Combine all features
            combined_features_df = self._combine_features(features_data, labeled_df)

            if combined_features_df.empty:
                raise ValueError("No features available for final selection")

            # Setup selection configuration
            selection_config = self._setup_selection_config(config)

            # Initialize optimization components
            await self._initialize_optimization_components(config)
            await self._initialize_hardware_optimization_components(config)

            # Initialize selection component
            self.selection_component = FinalFeatureSelectionComponent(selection_config)

            # Perform feature selection for different set sizes
            feature_sets = self._perform_multi_size_selection(combined_features_df, targets, config)

            # Generate SHAP values for interpretability
            shap_values = self._generate_shap_values(feature_sets, combined_features_df, targets, config)

            # Generate artifacts
            artifacts = self._generate_artifacts(feature_sets, shap_values, config, combined_features_df)

            # Create comprehensive outcome report
            outcome_report = self._create_outcome_report(feature_sets, shap_values, config)

            # Save artifacts
            saved_artifacts = []
            for artifact_name, artifact_data in artifacts.items():
                artifact_path = self._save_artifact(
                    artifact_data,
                    artifact_name,
                    artifact_type="data"
                )
                saved_artifacts.append({
                    'name': artifact_name,
                    'path': artifact_path,
                    'type': 'data'
                })

            # Save outcome report (pickle format)
            report_path = self._save_artifact(
                outcome_report,
                "final_feature_selection_outcome_report",
                artifact_type="report"
            )
            
            # Generate and save markdown report
            markdown_report = self._generate_markdown_report(outcome_report, feature_sets, shap_values, config)
            markdown_path = self._save_markdown_report(markdown_report, "final_feature_selection_outcome_report")

            # Calculate metrics
            metrics = self._calculate_metrics(feature_sets, shap_values, config)
            
            # Add optimization performance metrics
            optimization_metrics = self._get_optimization_metrics()
            metrics.update(optimization_metrics)

            execution_result = {
                'success': True,
                'artifacts': saved_artifacts,
                'metrics': metrics,
                'feature_sets': {k: len(v) for k, v in feature_sets.items()},
                'shap_summary': self._summarize_shap_values(shap_values),
                'outcome_report_path': report_path,
                'markdown_report_path': markdown_path,
                'execution_time': 0.0,  # Will be set by base class
                'optimization_enabled': self.optimization_enabled,
                'vectorization_stats': self._get_vectorization_stats()
            }

            tprint_success(f"✅ {self.step_name} completed successfully")
            tprint_info(f"📊 Created feature sets: {metrics.get('total_features_selected', 0)} total features across {len(feature_sets)} sets")

            return execution_result

        except Exception as e:
            error_msg = f"Final feature selection step failed: {str(e)}"
            tprint_error(error_msg)

            return {
                'success': False,
                'error': error_msg,
                'artifacts': [],
                'metrics': {},
                'execution_time': 0.0
            }

    def _collect_features_from_previous_steps(self) -> Dict[str, Any]:
        """Collect features from previous steps in the pipeline."""
        features_data = {}

        # PRIORITY 1: Get features from main feature generation step (334+ features)
        try:
            # Try different possible artifact names for generated features
            generated_features = None
            for artifact_name in ['generated_features', 'generated_features_1h', 'generated_features_15m', 'generated_features_long']:
                try:
                    generated_features = self._get_artifact(artifact_name)
                    if generated_features is not None:
                        features_data['generated_features'] = generated_features
                        tprint_info(f"✅ Retrieved main generated features ({artifact_name}): {generated_features.shape if hasattr(generated_features, 'shape') else 'Unknown shape'}")
                        break
                except Exception:
                    continue
            
            if generated_features is None:
                tprint_warning("⚠️ Could not get main generated features from any artifact name")
        except Exception as e:
            tprint_warning(f"⚠️ Could not get main generated features: {e}")

        # PRIORITY 2: Get features from lookback optimization step (Most sophisticated engineered features)
        try:
            lookback_features = self._get_artifact('lookback_optimization')
            if lookback_features is not None:
                features_data['lookback_optimization'] = lookback_features
                tprint_info(f"✅ Retrieved lookback optimization features: {lookback_features.shape if hasattr(lookback_features, 'shape') else 'Unknown shape'}")
        except Exception as e:
            tprint_warning(f"⚠️ Could not get lookback optimization features: {e}")

        # PRIORITY 3: Get features from interaction generation steps (Complex feature interactions)
        try:
            analyst_interactions = self._get_artifact('analyst_interaction_features')
            if analyst_interactions is not None:
                features_data['analyst_interactions'] = analyst_interactions
                tprint_info(f"✅ Retrieved analyst interaction features: {analyst_interactions.shape if hasattr(analyst_interactions, 'shape') else 'Unknown shape'}")
        except Exception as e:
            tprint_warning(f"⚠️ Could not get analyst interaction features: {e}")

        try:
            tactician_interactions = self._get_artifact('tactician_interaction_features')
            if tactician_interactions is not None:
                features_data['tactician_interactions'] = tactician_interactions
                tprint_info(f"✅ Retrieved tactician interaction features: {tactician_interactions.shape if hasattr(tactician_interactions, 'shape') else 'Unknown shape'}")
        except Exception as e:
            tprint_warning(f"⚠️ Could not get tactician interaction features: {e}")

        # PRIORITY 4: Get features from feature selection step (Previously selected features)
        try:
            selected_features = self._get_artifact('selected_features')
            if selected_features is not None:
                features_data['selected_features'] = selected_features
                tprint_info(f"✅ Retrieved selected features: {len(selected_features) if isinstance(selected_features, list) else 'Unknown count'}")
        except Exception as e:
            tprint_warning(f"⚠️ Could not get selected features: {e}")

        # PRIORITY 5: Get feature dataframe from feature generation step (Other engineered features)
        try:
            feature_df = self._get_artifact('feature_dataframe')
            if feature_df is not None:
                features_data['feature_dataframe'] = feature_df
                tprint_info(f"✅ Retrieved feature dataframe: {feature_df.shape if hasattr(feature_df, 'shape') else 'Unknown shape'}")
        except Exception as e:
            tprint_warning(f"⚠️ Could not get feature dataframe: {e}")

        return features_data

    async def _initialize_optimization_components(self, config: Dict[str, Any]) -> None:
        """Initialize VectorBT optimization components."""
        try:
            tprint_info("🚀 Initializing VectorBT optimization components...")
            
            # Initialize unified vectorization manager
            vectorization_config = VectorizationConfig(
                enable_vectorbt=True,
                enable_gpu=config.get('enable_gpu', False),
                enable_parallel=config.get('enable_parallel', True),
                memory_efficient=config.get('memory_efficient', True),
                max_memory_gb=config.get('max_memory_gb', 8.0),
                chunk_size=config.get('chunk_size', 1000),
                enable_monitoring=config.get('enable_monitoring', True),
                batch_size=config.get('batch_size', 10000),
                enable_batch_processing=True,
                rolling_optimization_threshold=config.get('rolling_optimization_threshold', 1000),
                enable_rolling_optimization=True
            )
            
            self.vectorization_manager = get_unified_vectorization_manager(vectorization_config)
            tprint_success("✅ Unified vectorization manager initialized")
            
            # Initialize VectorBT rolling optimizer
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                enable_gpu=config.get('enable_gpu', False),
                enable_parallel=config.get('enable_parallel', True),
                memory_efficient=config.get('memory_efficient', True),
                chunk_size=config.get('chunk_size', 1000),
                fast_fail=config.get('fast_fail', True),
                enable_logging=config.get('enable_logging', True)
            )
            tprint_success("✅ VectorBT rolling optimizer initialized")
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to initialize optimization components: {e}")
            self.optimization_enabled = False
            tprint_warning("⚠️ Continuing without VectorBT optimizations")

    async def _initialize_hardware_optimization_components(self, config: Dict[str, Any]) -> None:
        """Initialize hardware optimization components."""
        try:
            tprint_info("🚀 Initializing hardware optimization components...")
            
            # Initialize unified hardware manager
            hardware_config = HardwareConfig(
                cpu_optimization_level=OptimizationLevel.BALANCED,
                gpu_optimization_level=OptimizationLevel.BALANCED,
                memory_optimization_level=OptimizationLevel.BALANCED,
                enable_adaptive_optimization=True,
                enable_learning=True,
                auto_tuning_enabled=True,
                performance_monitoring_enabled=True,
                memory_limit_gb=config.get('memory_limit_gb', 8.0),
                enable_memory_pooling=True,
                enable_predictive_allocation=True,
                enable_compression=True
            )
            
            self.hardware_manager = UnifiedHardwareManager(hardware_config)
            init_result = self.hardware_manager.initialize()
            if init_result:
                tprint_success("✅ Unified hardware manager initialized")
            else:
                tprint_warning("⚠️ Unified hardware manager initialization failed")
            
            # Initialize adaptive optimization engine
            self.adaptive_engine = AdaptiveOptimizationEngine(
                database_path="optimization_performance.db"
            )
            # Initialize hardware managers for the adaptive engine
            self.adaptive_engine.initialize_hardware_managers()
            tprint_success("✅ Adaptive optimization engine initialized")
            
            # Initialize CPU optimizer with warning suppression
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*CoreAffinityManager.*")
                warnings.filterwarnings("ignore", message=".*core affinity.*")
                self.cpu_optimizer = AdvancedM1CPUOptimizer()
                # Add custom workload profile for feature engineering
                feature_engineering_profile = WorkloadProfile(
                    name='feature_engineering',
                    cpu_intensity=0.7,
                    memory_intensity=0.8,
                    thermal_sensitivity=0.4,
                    power_sensitivity=0.5,
                    preferred_cores=CoreType.PERFORMANCE,
                    max_threads=6
                )
                self.cpu_optimizer.add_workload_profile(feature_engineering_profile)
                # Optimize for feature engineering workload
                self.cpu_optimizer.optimize_for_workload_profile('feature_engineering')
            tprint_success("✅ Advanced CPU optimizer initialized")
            
            # Initialize GPU manager
            self.gpu_manager = EnhancedM1GPUManager()
            # EnhancedM1GPUManager doesn't have an initialize method
            tprint_success("✅ Enhanced GPU manager initialized")
            
            # Initialize memory optimizer
            self.memory_optimizer = AdvancedM1MemoryOptimizer(
                memory_limit_gb=config.get('memory_limit_gb', 8.0),
                strategy=MemoryStrategy.ADAPTIVE
            )
            # AdvancedM1MemoryOptimizer doesn't have an initialize method
            tprint_success("✅ Advanced memory optimizer initialized")
            
            tprint_success("✅ All hardware optimization components initialized")
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to initialize hardware optimization components: {e}")
            self.hardware_optimization_enabled = False
            tprint_warning("⚠️ Continuing without hardware optimizations")

    def _combine_features(self, features_data: Dict[str, Any], labeled_df: pd.DataFrame) -> pd.DataFrame:
        """Combine features from different sources into a single DataFrame with VectorBT optimizations."""
        tprint_info("🔄 Combining features with VectorBT optimizations...")
        
        # PRIORITY 1: Start with labeled dataframe to preserve target column
        base_features = labeled_df.copy()
        tprint_info(f"📊 Using labeled dataframe as base: {base_features.shape}")
        tprint_info(f"📊 Target column in base: {'price_target_vol_normalized' in base_features.columns}")
        
        # PRIORITY 2: Add main generated features if available
        if 'generated_features' in features_data and features_data['generated_features'] is not None:
            generated_features = features_data['generated_features']
            tprint_info(f"📊 Adding main generated features: {generated_features.shape}")
            
            # Check data alignment
            if generated_features.shape[0] != base_features.shape[0]:
                tprint_warning(f"⚠️ Shape mismatch: base_features {base_features.shape} vs generated_features {generated_features.shape}")
                # Try to align by index if possible
                if hasattr(generated_features.index, 'intersection') and hasattr(base_features.index, 'intersection'):
                    common_index = base_features.index.intersection(generated_features.index)
                    if len(common_index) > 0:
                        tprint_info(f"📊 Aligning dataframes using {len(common_index)} common indices")
                        generated_features = generated_features.loc[common_index]
                        base_features = base_features.loc[common_index]
                    else:
                        tprint_warning("⚠️ No common indices found, skipping generated features")
                        generated_features = None
                else:
                    tprint_warning("⚠️ Cannot align dataframes, skipping generated features")
                    generated_features = None
            
            if generated_features is not None:
                # Add generated features (excluding any duplicate columns and target columns)
                target_cols = ['target', 'label', 'return', 'price_target_vol_normalized']
                generated_cols = [col for col in generated_features.columns
                               if col not in base_features.columns and col not in target_cols]
                if generated_cols:
                    base_features = pd.concat([base_features, generated_features[generated_cols]], axis=1)
                    tprint_info(f"📊 Added {len(generated_cols)} generated features")

        # Use vectorization manager for optimized operations if available
        if self.vectorization_manager and self.optimization_enabled:
            try:
                # Optimize the base dataframe for memory efficiency
                base_features = self.vectorization_manager.optimize_dataframe(base_features)
                tprint_info("✅ Base features optimized for memory efficiency")
            except Exception as e:
                tprint_warning(f"⚠️ Memory optimization failed: {e}")

        # PRIORITY 2: Add lookback optimization features (most sophisticated engineered features)
        if 'lookback_optimization' in features_data and features_data['lookback_optimization'] is not None:
            lookback_data = features_data['lookback_optimization']
            if isinstance(lookback_data, pd.DataFrame):
                # Check if lookback data has the correct shape (samples, features)
                if lookback_data.shape[0] > 1:  # Multiple samples
                    # Optimize lookback dataframe if available
                    if self.vectorization_manager and self.optimization_enabled:
                        try:
                            lookback_data = self.vectorization_manager.optimize_dataframe(lookback_data)
                        except Exception as e:
                            tprint_warning(f"⚠️ Lookback dataframe optimization failed: {e}")
                    
                    # Add lookback features (excluding any duplicate columns)
                    lookback_cols = [col for col in lookback_data.columns
                                   if col not in base_features.columns]
                    if lookback_cols:
                        base_features = pd.concat([base_features, lookback_data[lookback_cols]], axis=1)
                        tprint_info(f"📊 Added {len(lookback_cols)} lookback optimization features (PRIORITY 2)")
                else:
                    tprint_warning(f"⚠️ Lookback optimization data has wrong shape {lookback_data.shape}, skipping")
            elif isinstance(lookback_data, dict):
                # Lookback optimization produces metadata, not feature data
                tprint_info(f"📊 Lookback optimization metadata available: {len(lookback_data)} categories")
                tprint_info(f"📊 Lookback optimization categories: {list(lookback_data.keys())}")
                # TODO: Use this metadata to generate features with optimized lookback periods
                tprint_info("ℹ️ Note: Lookback optimization metadata should be used to generate features with optimized lookback periods")

        # PRIORITY 3: Add interaction features (complex feature interactions)
        for interaction_type in ['analyst_interactions', 'tactician_interactions']:
            if interaction_type in features_data and features_data[interaction_type] is not None:
                interaction_df = features_data[interaction_type]
                if isinstance(interaction_df, pd.DataFrame):
                    # Optimize interaction dataframe if available
                    if self.vectorization_manager and self.optimization_enabled:
                        try:
                            interaction_df = self.vectorization_manager.optimize_dataframe(interaction_df)
                        except Exception as e:
                            tprint_warning(f"⚠️ Interaction dataframe optimization failed: {e}")
                    
                    # Check data alignment and handle shape mismatches
                    if interaction_df.shape[0] != base_features.shape[0]:
                        tprint_warning(f"⚠️ Shape mismatch: base_features {base_features.shape} vs {interaction_type} {interaction_df.shape}")
                        # Align dataframes by index if possible
                        if hasattr(interaction_df.index, 'intersection') and hasattr(base_features.index, 'intersection'):
                            common_index = base_features.index.intersection(interaction_df.index)
                            if len(common_index) > 0:
                                tprint_info(f"📊 Aligning dataframes using {len(common_index)} common indices")
                                interaction_df = interaction_df.loc[common_index]
                                base_features = base_features.loc[common_index]
                            else:
                                tprint_warning(f"⚠️ No common indices found, skipping {interaction_type}")
                                continue
                        else:
                            tprint_warning(f"⚠️ Cannot align dataframes, skipping {interaction_type}")
                            continue
                    
                    # Add interaction features (excluding any duplicate columns)
                    interaction_cols = [col for col in interaction_df.columns
                                     if col not in base_features.columns]
                    if interaction_cols:
                        base_features = pd.concat([base_features, interaction_df[interaction_cols]], axis=1)
                        tprint_info(f"📊 Added {len(interaction_cols)} {interaction_type} features (PRIORITY 3)")

        # PRIORITY 4: Add features from feature dataframe if available (with proper alignment)
        if 'feature_dataframe' in features_data and features_data['feature_dataframe'] is not None:
            feature_df = features_data['feature_dataframe']
            
            # Check data alignment first
            if feature_df.shape[0] != base_features.shape[0]:
                tprint_warning(f"⚠️ Feature dataframe shape mismatch: base_features {base_features.shape} vs feature_df {feature_df.shape}")
                # Try to align by index if possible
                if hasattr(feature_df.index, 'intersection') and hasattr(base_features.index, 'intersection'):
                    common_index = base_features.index.intersection(feature_df.index)
                    if len(common_index) > 0:
                        tprint_info(f"📊 Aligning feature dataframe using {len(common_index)} common indices")
                        feature_df = feature_df.loc[common_index]
                        base_features = base_features.loc[common_index]
                    else:
                        tprint_warning("⚠️ No common indices found, skipping feature dataframe")
                        feature_df = None
                else:
                    tprint_warning("⚠️ Cannot align feature dataframe, skipping")
                    feature_df = None
            
            if feature_df is not None:
                # Optimize feature dataframe if vectorization manager is available
                if self.vectorization_manager and self.optimization_enabled:
                    try:
                        feature_df = self.vectorization_manager.optimize_dataframe(feature_df)
                    except Exception as e:
                        tprint_warning(f"⚠️ Feature dataframe optimization failed: {e}")
            
                # Find common columns (excluding OHLCV, basic time features, and target columns)
                ohlcv_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
                basic_time_cols = ['hour', 'day_of_week', 'base_threshold']
                target_cols = ['target', 'label', 'return', 'price_target_vol_normalized']  # Common target column name

            feature_cols = [col for col in feature_df.columns
                                  if col not in ohlcv_cols and col not in basic_time_cols and col not in target_cols]

            if feature_cols:
                # Use optimized concatenation if available
                if self.vectorization_manager and self.optimization_enabled:
                    try:
                        # Use batch processing for large feature sets
                        if len(feature_cols) > 1000:
                            tprint_info(f"🔄 Processing {len(feature_cols)} features in batches...")
                            # Process in chunks to avoid memory issues
                            chunk_size = self.vectorization_manager.config.chunk_size
                            for i in range(0, len(feature_cols), chunk_size):
                                chunk_cols = feature_cols[i:i + chunk_size]
                                chunk_df = feature_df[chunk_cols]
                                base_features = pd.concat([base_features, chunk_df], axis=1)
                        else:
                            base_features = pd.concat([base_features, feature_df[feature_cols]], axis=1)
                    except Exception as e:
                        tprint_warning(f"⚠️ Optimized concatenation failed, using standard method: {e}")
                        base_features = pd.concat([base_features, feature_df[feature_cols]], axis=1)
                else:
                    base_features = pd.concat([base_features, feature_df[feature_cols]], axis=1)
                
                    tprint_info(f"📊 Added {len(feature_cols)} feature dataframe columns")
                    tprint_info(f"📊 Added {len(feature_cols)} features from feature dataframe (PRIORITY 4)")

        # Remove any non-numeric columns except timestamp and target columns
        numeric_cols = []
        target_cols = ['target', 'label', 'return', 'price_target_vol_normalized']
        
        for col in base_features.columns:
            if col == 'timestamp' or col in target_cols or pd.api.types.is_numeric_dtype(base_features[col]):
                numeric_cols.append(col)

        tprint_info(f"🔍 DEBUG: Base features columns after combination: {list(base_features.columns)}")
        tprint_info(f"🔍 DEBUG: Numeric columns found: {len(numeric_cols)}")
        tprint_info(f"🔍 DEBUG: Numeric columns: {numeric_cols[:10]}...")  # Show first 10

        result_df = base_features[numeric_cols].copy()
        
        # Debug: Check if target column is present
        available_targets = [col for col in target_cols if col in result_df.columns]
        tprint_info(f"📊 Combined feature matrix: {len(numeric_cols)} features, {len(result_df)} samples")
        tprint_info(f"📊 Available target columns: {available_targets}")
        
        if not available_targets:
            tprint_warning("⚠️ No target columns found in combined features!")
            tprint_info(f"📊 All columns in result_df: {list(result_df.columns)[:20]}...")

        # Handle NaN values with optimized operations
        if self.vectorization_manager and self.optimization_enabled:
            try:
                # Use vectorized operations for NaN handling
                tprint_info("🔄 Optimizing NaN handling...")
                
                # Drop columns with too many NaN values (more lenient for sophisticated features)
                nan_threshold = int(0.5 * len(result_df))  # More lenient threshold
                valid_cols = []
                for col in result_df.columns:
                    if result_df[col].count() >= nan_threshold:
                        valid_cols.append(col)
                    else:
                        # Check if it's a sophisticated feature and be more lenient
                        if any(keyword in col.lower() for keyword in ['vectorbt', 'interaction', 'enhanced', 'optimized', 'advanced', 'statistical', 'wavelet', 'entropy', 'ad_line', 'obv', 'volatility', 'order_flow']):
                            if result_df[col].count() >= int(0.3 * len(result_df)):  # Even more lenient for sophisticated features
                                valid_cols.append(col)
                                tprint_info(f"📊 Keeping sophisticated feature with low data coverage: {col}")
                
                result_df = result_df[valid_cols]
                
                # Fill remaining NaN with median using vectorized operations
                numeric_cols_only = result_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols_only) > 0:
                    medians = result_df[numeric_cols_only].median()
                    result_df[numeric_cols_only] = result_df[numeric_cols_only].fillna(medians)
                
                tprint_success("✅ NaN handling optimized with sophisticated feature protection")
            except Exception as e:
                tprint_warning(f"⚠️ Optimized NaN handling failed, using standard method: {e}")
                result_df = result_df.dropna(axis=1, thresh=int(0.7 * len(result_df)))
                result_df = result_df.fillna(result_df.median())
        else:
            # Standard NaN handling
            result_df = result_df.dropna(axis=1, thresh=int(0.7 * len(result_df)))
            result_df = result_df.fillna(result_df.median())

        # Final optimization if vectorization manager is available
        if self.vectorization_manager and self.optimization_enabled:
            try:
                result_df = self.vectorization_manager.optimize_dataframe(result_df)
                tprint_success("✅ Final feature matrix optimized")
            except Exception as e:
                tprint_warning(f"⚠️ Final optimization failed: {e}")

        tprint_info(f"📊 Combined feature matrix: {result_df.shape[1]} features, {result_df.shape[0]} samples")
        return result_df

    def _setup_selection_config(self, config: Dict[str, Any]) -> FinalFeatureSelectionConfig:
        """Setup feature selection configuration."""
        # Add hardware optimization configuration to the config
        selection_config = config.copy()

        # Add default hardware optimization parameters if not present
        selection_config.setdefault('enable_hardware_optimization', True)
        selection_config.setdefault('memory_limit_gb', 8.0)
        selection_config.setdefault('max_memory_mb', 2048.0)
        selection_config.setdefault('streaming_chunk_size', 10000)
        selection_config.setdefault('memory_pressure_threshold', 0.8)
        selection_config.setdefault('enable_caching', True)
        selection_config.setdefault('cache_memory_mb', 1024)
        selection_config.setdefault('cache_memory_limit_gb', 4.0)
        selection_config.setdefault('enable_compression', True)

        return FinalFeatureSelectionConfig(
            max_features=config.get('max_features', 100),
            min_features=config.get('min_features', 10),
            selection_method=config.get('selection_method', 'mutual_info'),
            scoring_threshold=config.get('scoring_threshold', 0.01),
            use_tree_based=config.get('use_tree_based', True)
        )

    def _perform_multi_size_selection(self, features_df: pd.DataFrame, targets: pd.Series, config: Dict[str, Any]) -> Dict[str, List[str]]:
        """Perform feature selection for multiple feature set sizes with CMI-aware Tactician mode support."""
        # Define feature set sizes
        feature_set_sizes = config.get('feature_set_sizes', [60, 50, 40])

        feature_sets = {}

        # Detect Tactician mode and check for CMI availability
        is_tactician_mode = self._detect_tactician_mode(features_df, config)
        cmi_available = CMI_COMPLEMENTARITY_AVAILABLE and self.cmi_scorer is not None
        
        if is_tactician_mode and cmi_available:
            tprint_info("🎯 Tactician mode detected with CMI support - using CMI-based feature selection")
            return self._perform_cmi_aware_selection(features_df, targets, config, feature_set_sizes)
        elif is_tactician_mode and not cmi_available:
            tprint_warning("⚠️ Tactician mode detected but CMI not available - using standard selection")
        else:
            tprint_info("📊 Standard mode - using regular mutual information selection")

        # Separate features from targets and exclude raw data columns
        raw_data_columns = ['open', 'high', 'low', 'close', 'volume', 'hour', 'day_of_week', 'base_threshold']
        basic_features = ['open_time', 'close_time', 'body_size', 'close_return', 'price_range_pct', 
                         'volume_return', 'close_log_return', 'volume_log_return', 'price_range', 
                         'body_size_pct', 'trades', 'quote_volume', 'day', 'lookahead_periods', 'is_weekend']
        
        # Debug: Show all available columns
        tprint_info(f"🔍 DEBUG: All columns in features_df: {list(features_df.columns)}")
        
        # Prioritize sophisticated engineered features over basic ones
        sophisticated_features = [col for col in features_df.columns
                                if col not in ['target', 'label', 'return', 'timestamp', 'price_target_vol_normalized'] + raw_data_columns
                                and any(keyword in col.lower() for keyword in ['vectorbt', 'interaction', 'enhanced', 'optimized', 'advanced', 'statistical', 'wavelet', 'entropy', 'ad_line', 'obv', 'volatility', 'order_flow'])]
        
        basic_engineered_features = [col for col in features_df.columns
                                   if col not in ['target', 'label', 'return', 'timestamp', 'price_target_vol_normalized'] + raw_data_columns
                                   and col not in sophisticated_features]
        
        # Prioritize sophisticated features first
        feature_cols = sophisticated_features + basic_engineered_features
        target_cols = [col for col in ['target', 'label', 'return', 'price_target_vol_normalized']
                      if col in features_df.columns]

        tprint_info(f"🔍 Sophisticated features: {len(sophisticated_features)}")
        tprint_info(f"🔍 Basic engineered features: {len(basic_engineered_features)}")
        tprint_info(f"🔍 Total available features: {len(feature_cols)}")
        tprint_info(f"🔍 Available targets: {len(target_cols)}")
        tprint_info(f"🔍 Sophisticated features: {sophisticated_features[:5]}...")  # Show first 5 sophisticated features
        tprint_info(f"🔍 Basic engineered features: {basic_engineered_features[:5]}...")  # Show first 5 basic features
        tprint_info(f"🔍 Target columns: {target_cols}")

        if not target_cols:
            raise ValueError("No target column found in features dataframe")

        if not feature_cols:
            raise ValueError("No feature columns found in features dataframe")

        X = features_df[feature_cols]
        y = features_df[target_cols[0]]

        tprint_info(f"🔍 Performing feature selection on {len(feature_cols)} features...")

        # Use batch processing if vectorization manager is available
        if self.vectorization_manager and self.optimization_enabled and len(feature_cols) > 1000:
            try:
                tprint_info("🚀 Using VectorBT batch processing for feature selection...")
                
                # Create feature configurations for batch processing
                feature_configs = []
                for size in feature_set_sizes:
                    feature_configs.append({
                        'name': f'selected_features_{size}',
                        'type': 'selection',
                        'params': {
                            'max_features': size,
                            'min_features': max(5, size // 2),
                            'selection_method': config.get('selection_method', 'mutual_info'),
                            'scoring_threshold': config.get('scoring_threshold', 0.01),
                            'use_tree_based': config.get('use_tree_based', True),
                            'X': X,
                            'y': y,
                            'feature_names': feature_cols
                        }
                    })
                
                # Process features in batch
                batch_results = self.vectorization_manager.batch_process_features(
                    features_df, feature_configs
                )
                
                # Extract results
                for size in feature_set_sizes:
                    result_key = f'selected_features_{size}'
                    if result_key in batch_results.columns:
                        selected_features = batch_results[result_key].dropna().tolist()
                        feature_sets[result_key] = selected_features
                        feature_sets[f'selected_feature_dataframe_{size}'] = features_df[selected_features + target_cols].copy()
                
                tprint_success("✅ Batch feature selection completed")
                return feature_sets
                
            except Exception as e:
                tprint_warning(f"⚠️ Batch processing failed, falling back to sequential: {e}")

        # Sequential processing (fallback or for smaller datasets)
        tprint_info("🔄 Using sequential feature selection...")
        
        # Create selection configs for different sizes
        for size in feature_set_sizes:
            tprint_info(f"🎯 Selecting top {size} features...")

            # Create config for this size
            size_config = FinalFeatureSelectionConfig(
                max_features=size,
                min_features=max(5, size // 2),  # Minimum is half the size or 5, whichever is larger
                selection_method=config.get('selection_method', 'mutual_info'),
                scoring_threshold=config.get('scoring_threshold', 0.01),
                use_tree_based=config.get('use_tree_based', True)
            )

            # Create temporary component for this selection
            temp_component = FinalFeatureSelectionComponent(size_config)
            selected_features = temp_component.select_features(X, y, feature_cols)

            feature_sets[f'selected_features_{size}'] = selected_features

            # Also create the corresponding dataframes
            feature_sets[f'selected_feature_dataframe_{size}'] = features_df[selected_features + target_cols].copy()

        tprint_success(f"✅ Created {len(feature_sets)} feature sets")
        return feature_sets

    def _detect_tactician_mode(self, features_df: pd.DataFrame, config: Dict[str, Any]) -> bool:
        """
        Detect if we're in Tactician mode based on launcher commands and available features.
        
        Args:
            features_df: Combined features dataframe
            config: Configuration dictionary
            
        Returns:
            True if in Tactician mode, False otherwise
        """
        # Primary detection: Check current step name for Tactician training steps
        # This is the most reliable method since it comes directly from ares_launcher.py
        current_step_name = getattr(self, 'step_name', '')
        is_tactician_training_step = (
            'tactician_base_training' in current_step_name or
            'tactician_ensemble_training' in current_step_name or
            'tactician' in current_step_name.lower()
        )
        
        # Also check if we're in a Tactician execution context
        # This could be set by upstream steps or the launcher
        tactician_execution_context = config.get('execution_context', '').lower()
        is_tactician_context = 'tactician' in tactician_execution_context
        
        # Secondary detection: Check for Tactician-specific features
        tactician_features = [col for col in features_df.columns if 'tactician' in col.lower()]
        
        # Tertiary detection: Check for CMI-based Tactician features
        cmi_tactician_features = [col for col in features_df.columns if 'cmi' in col.lower()]
        
        # Quaternary detection: Check configuration for explicit Tactician mode
        explicit_tactician_mode = config.get('tactician_mode', False)
        
        # Quinary detection: Check for Analyst features (if present, we might be in complementarity mode)
        analyst_features = [col for col in features_df.columns if 'analyst' in col.lower()]
        
        # Determine mode based on step name (primary) or feature analysis (secondary)
        is_tactician_mode = (
            is_tactician_training_step or
            is_tactician_context or
            len(tactician_features) > 0 or 
            len(cmi_tactician_features) > 0 or 
            explicit_tactician_mode or
            (len(analyst_features) > 0 and config.get('enable_cmi_complementarity', False))
        )
        
        tprint_info(f"🔍 Tactician mode detection:")
        tprint_info(f"  - Current step name: {current_step_name}")
        tprint_info(f"  - Is Tactician training step: {is_tactician_training_step}")
        tprint_info(f"  - Execution context: {config.get('execution_context', 'N/A')}")
        tprint_info(f"  - Is Tactician context: {is_tactician_context}")
        tprint_info(f"  - Tactician features: {len(tactician_features)}")
        tprint_info(f"  - CMI Tactician features: {len(cmi_tactician_features)}")
        tprint_info(f"  - Analyst features: {len(analyst_features)}")
        tprint_info(f"  - Explicit Tactician mode: {explicit_tactician_mode}")
        tprint_info(f"  - CMI complementarity enabled: {config.get('enable_cmi_complementarity', False)}")
        tprint_info(f"  - Detected Tactician mode: {is_tactician_mode}")
        
        return is_tactician_mode

    def _perform_cmi_aware_selection(self, features_df: pd.DataFrame, targets: pd.Series, 
                                   config: Dict[str, Any], feature_set_sizes: List[int]) -> Dict[str, List[str]]:
        """
        Perform CMI-aware feature selection for Tactician mode.
        
        Args:
            features_df: Combined features dataframe
            targets: Target variables
            config: Configuration dictionary
            feature_set_sizes: List of feature set sizes to create
            
        Returns:
            Dictionary of feature sets
        """
        tprint_info("🎯 Performing CMI-aware feature selection for Tactician mode...")
        
        try:
            # Extract Analyst side information for CMI conditioning
            analyst_side_info = self._extract_analyst_side_info_for_cmi(features_df, config)
            
            if not analyst_side_info.get('cmi_enabled', False):
                tprint_warning("⚠️ CMI not available, falling back to standard selection")
                return self._perform_standard_selection(features_df, targets, config, feature_set_sizes)
            
            # Separate Tactician and Analyst features
            tactician_features = [col for col in features_df.columns 
                                if 'tactician' in col.lower() or 'cmi' in col.lower()]
            analyst_features = [col for col in features_df.columns 
                              if 'analyst' in col.lower()]
            other_features = [col for col in features_df.columns 
                            if col not in tactician_features + analyst_features 
                            and col not in ['target', 'label', 'return', 'timestamp', 'price_target_vol_normalized']]
            
            tprint_info(f"🔍 Feature separation:")
            tprint_info(f"  - Tactician features: {len(tactician_features)}")
            tprint_info(f"  - Analyst features: {len(analyst_features)}")
            tprint_info(f"  - Other features: {len(other_features)}")
            
            # Prepare features for CMI selection
            all_features = tactician_features + other_features
            if not all_features:
                tprint_warning("⚠️ No features available for CMI selection")
                return self._perform_standard_selection(features_df, targets, config, feature_set_sizes)
            
            X = features_df[all_features]
            y = features_df[targets.name] if hasattr(targets, 'name') else targets
            
            # Perform CMI-based selection for each size
            feature_sets = {}
            for size in feature_set_sizes:
                tprint_info(f"🎯 CMI-based selection for {size} features...")
                
                # Use CMI scorer for feature selection
                selected_features = self.cmi_scorer.select_features(
                    features=X,
                    targets=y,
                    analyst_side_info=analyst_side_info['side_info']
                )
                
                # Limit to requested size
                selected_features = selected_features[:size]
                
                feature_sets[f'selected_features_{size}'] = selected_features
                feature_sets[f'selected_feature_dataframe_{size}'] = features_df[selected_features + [targets.name if hasattr(targets, 'name') else 'target']].copy()
                
                tprint_success(f"✅ CMI-based selection completed: {len(selected_features)} features selected")
            
            return feature_sets
            
        except Exception as e:
            tprint_error(f"❌ CMI-aware selection failed: {e}")
            tprint_warning("⚠️ Falling back to standard selection")
            return self._perform_standard_selection(features_df, targets, config, feature_set_sizes)

    def _extract_analyst_side_info_for_cmi(self, features_df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract Analyst side information for CMI conditioning.
        
        Args:
            features_df: Combined features dataframe
            config: Configuration dictionary
            
        Returns:
            Dictionary containing Analyst side information and CMI configuration
        """
        if not CMI_COMPLEMENTARITY_AVAILABLE or self.analyst_handler is None:
            return {
                'cmi_enabled': False,
                'reason': 'CMI complementarity not available'
            }
        
        try:
            # Extract Analyst features
            analyst_features = [col for col in features_df.columns if 'analyst' in col.lower()]
            
            if not analyst_features:
                return {
                    'cmi_enabled': False,
                    'reason': 'No Analyst features found'
                }
            
            # Create Analyst features dataframe
            analyst_df = features_df[analyst_features]
            
            # Extract Analyst side information
            analyst_side_info = self.analyst_handler.extract_side_info(
                {'analyst_features': analyst_df},
                targets=None,  # Will be provided later
                data_index=analyst_df.index
            )
            
            if analyst_side_info.is_valid:
                return {
                    'cmi_enabled': True,
                    'analyst_features': analyst_df,
                    'side_info': analyst_side_info
                }
            else:
                return {
                    'cmi_enabled': False,
                    'reason': 'Analyst side information invalid'
                }
                
        except Exception as e:
            tprint_warning(f"⚠️ Failed to extract Analyst side information: {e}")
            return {
                'cmi_enabled': False,
                'reason': f'Extraction failed: {e}'
            }

    def _perform_standard_selection(self, features_df: pd.DataFrame, targets: pd.Series, 
                                  config: Dict[str, Any], feature_set_sizes: List[int]) -> Dict[str, List[str]]:
        """
        Perform standard feature selection (fallback method).
        
        Args:
            features_df: Combined features dataframe
            targets: Target variables
            config: Configuration dictionary
            feature_set_sizes: List of feature set sizes to create
            
        Returns:
            Dictionary of feature sets
        """
        tprint_info("📊 Performing standard feature selection...")
        
        # Use the original selection logic
        feature_sets = {}
        
        # Separate features from targets and exclude raw data columns
        raw_data_columns = ['open', 'high', 'low', 'close', 'volume', 'hour', 'day_of_week', 'base_threshold']
        
        # Prioritize sophisticated engineered features over basic ones
        sophisticated_features = [col for col in features_df.columns
                                if col not in ['target', 'label', 'return', 'timestamp', 'price_target_vol_normalized'] + raw_data_columns
                                and any(keyword in col.lower() for keyword in ['vectorbt', 'interaction', 'enhanced', 'optimized', 'advanced', 'statistical', 'wavelet', 'entropy', 'ad_line', 'obv', 'volatility', 'order_flow'])]
        
        basic_engineered_features = [col for col in features_df.columns
                                   if col not in ['target', 'label', 'return', 'timestamp', 'price_target_vol_normalized'] + raw_data_columns
                                   and col not in sophisticated_features]
        
        # Prioritize sophisticated features first
        feature_cols = sophisticated_features + basic_engineered_features
        target_cols = [col for col in ['target', 'label', 'return', 'price_target_vol_normalized']
                      if col in features_df.columns]

        if not target_cols:
            raise ValueError("No target column found in features dataframe")

        if not feature_cols:
            raise ValueError("No feature columns found in features dataframe")

        X = features_df[feature_cols]
        y = features_df[target_cols[0]]

        # Create selection configs for different sizes
        for size in feature_set_sizes:
            tprint_info(f"🎯 Selecting top {size} features...")

            # Create config for this size
            size_config = FinalFeatureSelectionConfig(
                max_features=size,
                min_features=max(5, size // 2),  # Minimum is half the size or 5, whichever is larger
                selection_method=config.get('selection_method', 'mutual_info'),
                scoring_threshold=config.get('scoring_threshold', 0.01),
                use_tree_based=config.get('use_tree_based', True)
            )

            # Create temporary component for this selection
            temp_component = FinalFeatureSelectionComponent(size_config)
            selected_features = temp_component.select_features(X, y, feature_cols)

            feature_sets[f'selected_features_{size}'] = selected_features

            # Also create the corresponding dataframes
            feature_sets[f'selected_feature_dataframe_{size}'] = features_df[selected_features + target_cols].copy()

        tprint_success(f"✅ Created {len(feature_sets)} feature sets")
        return feature_sets

    def _generate_shap_values(self, feature_sets: Dict[str, List[str]], features_df: pd.DataFrame, targets: pd.Series, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate SHAP values for interpretability."""
        shap_values = {}

        try:
            # Import SHAP (optional import)
            import shap
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            import warnings
            
            # Suppress NumPy deprecation warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*np\.bool.*")
                warnings.filterwarnings("ignore", message=".*np\.int.*")
                warnings.filterwarnings("ignore", message=".*np\.float.*")
                warnings.filterwarnings("ignore", message=".*np\.complex.*")

            # Get target column
            target_cols = [col for col in ['target', 'label', 'return', 'price_target_vol_normalized']
                          if col in features_df.columns]
            if not target_cols:
                tprint_warning("⚠️ No target column found for SHAP analysis")
                return shap_values

            target_col = target_cols[0]
            feature_cols = [col for col in features_df.columns if col != target_col]

            X = features_df[feature_cols]
            y = features_df[target_col]

            # Split data for training
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train a simple model for SHAP analysis
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)

                # Create SHAP explainer with additivity check disabled
            explainer = shap.TreeExplainer(rf_model)

            # Calculate SHAP values for each feature set
            for set_name, feature_list in feature_sets.items():
                if set_name.startswith('selected_features_'):
                    size = set_name.split('_')[-1]

                    if len(feature_list) > 0:
                        # Get SHAP values for this feature set with additivity check disabled
                        shap_test = explainer.shap_values(X_test[feature_list], check_additivity=False)

                        # Store SHAP summary
                        shap_values[f'shap_values_{size}'] = {
                            'shap_values': shap_test.tolist() if hasattr(shap_test, 'tolist') else shap_test,
                            'feature_names': feature_list,
                            'mean_abs_shap': np.mean(np.abs(shap_test), axis=0).tolist(),
                            'feature_importance': dict(zip(feature_list, np.mean(np.abs(shap_test), axis=0)))
                        }

                        tprint_info(f"📊 Generated SHAP values for {size} features")

            tprint_success("✅ SHAP value generation completed")

        except ImportError:
            tprint_warning("⚠️ SHAP not available, skipping SHAP value generation")
        except Exception as e:
            tprint_warning(f"⚠️ Error generating SHAP values: {e}")

        return shap_values

    def _generate_artifacts(self, feature_sets: Dict[str, List[str]], shap_values: Dict[str, Any], config: Dict[str, Any], combined_features_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate artifacts from feature selection results."""
        artifacts = {}

        # Feature sets
        for set_name, feature_list in feature_sets.items():
            if set_name.startswith('selected_features_'):
                artifacts[set_name] = feature_list
            elif set_name.startswith('selected_feature_dataframe_'):
                artifacts[set_name] = feature_sets[set_name]

        # Feature scores from selection component
        if self.selection_component:
            artifacts['feature_scores'] = self.selection_component.get_feature_scores()

        # SHAP values
        for shap_name, shap_data in shap_values.items():
            artifacts[shap_name] = shap_data

        # Selection metadata
        selection_metadata = {
            'total_features_available': len([col for col in combined_features_df.columns
                                           if col not in ['target', 'label', 'return', 'timestamp', 'price_target_vol_normalized']]),
            'feature_set_sizes': config.get('feature_set_sizes', [60, 50, 40]),
            'selection_method': config.get('selection_method', 'mutual_info'),
            'scoring_threshold': config.get('scoring_threshold', 0.01),
            'use_tree_based': config.get('use_tree_based', True),
            'timestamp': datetime.now().isoformat(),
            'symbol': config.get('symbol', 'unknown'),
            'execution_mode': config.get('execution_mode', 'light')
        }
        artifacts['selection_metadata'] = selection_metadata

        return artifacts

    def _calculate_metrics(self, feature_sets: Dict[str, List[str]], shap_values: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate metrics for the feature selection."""
        total_features_selected = sum(len(features) for name, features in feature_sets.items()
                                    if name.startswith('selected_features_'))

        metrics = {
            'total_features_selected': total_features_selected,
            'feature_sets_created': len([name for name in feature_sets.keys() if name.startswith('selected_features_')]),
            'shap_values_generated': len(shap_values),
            'execution_timestamp': datetime.now().isoformat(),
            'symbol': config.get('symbol', 'unknown'),
            'exchange': config.get('exchange', 'binance'),
            'timeframe': config.get('timeframe', '15m'),
            'execution_mode': config.get('execution_mode', 'light')
        }

        # Feature set details
        for set_name, feature_list in feature_sets.items():
            if set_name.startswith('selected_features_'):
                size = set_name.split('_')[-1]
                metrics[f'features_{size}'] = len(feature_list)

        return metrics

    def _summarize_shap_values(self, shap_values: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of SHAP values."""
        summary = {}

        for shap_name, shap_data in shap_values.items():
            if shap_name.startswith('shap_values_'):
                size = shap_name.split('_')[-1]
                if isinstance(shap_data, dict) and 'feature_importance' in shap_data:
                    top_features = sorted(shap_data['feature_importance'].items(),
                                        key=lambda x: x[1], reverse=True)[:10]
                    summary[f'top_10_features_{size}'] = top_features

        return summary

    def _get_optimization_metrics(self) -> Dict[str, Any]:
        """Get optimization performance metrics."""
        metrics = {
            'optimization_enabled': self.optimization_enabled,
            'vectorization_manager_available': self.vectorization_manager is not None,
            'rolling_optimizer_available': self.rolling_optimizer is not None,
            'hardware_optimization_enabled': self.hardware_optimization_enabled,
            'hardware_manager_available': self.hardware_manager is not None,
            'adaptive_engine_available': self.adaptive_engine is not None,
            'cpu_optimizer_available': self.cpu_optimizer is not None,
            'gpu_manager_available': self.gpu_manager is not None,
            'memory_optimizer_available': self.memory_optimizer is not None
        }
        
        if self.vectorization_manager and self.optimization_enabled:
            try:
                vectorization_stats = self.vectorization_manager.get_performance_stats()
                metrics.update({
                    'vectorization_operations': vectorization_stats.get('total_operations', 0),
                    'vectorbt_operations': vectorization_stats.get('vectorbt_operations', 0),
                    'memory_optimizations': vectorization_stats.get('memory_optimizations', 0),
                    'cache_hit_rate': vectorization_stats.get('cache_hit_rate', 0),
                    'batch_operations': vectorization_stats.get('batch_operations', 0)
                })
            except Exception as e:
                tprint_warning(f"⚠️ Failed to get vectorization stats: {e}")
        
        if self.rolling_optimizer and self.optimization_enabled:
            try:
                rolling_stats = self.rolling_optimizer.get_performance_stats()
                metrics.update({
                    'rolling_operations': rolling_stats.get('total_operations', 0),
                    'vectorbt_rolling_operations': rolling_stats.get('vectorbt_operations', 0),
                    'rolling_optimization_rate': rolling_stats.get('vectorbt_usage_rate', 0)
                })
            except Exception as e:
                tprint_warning(f"⚠️ Failed to get rolling optimizer stats: {e}")
        
        # Add hardware optimization metrics
        if self.hardware_manager and self.hardware_optimization_enabled:
            try:
                # Check if the method exists before calling it
                if hasattr(self.hardware_manager, 'get_performance_metrics'):
                    hardware_stats = self.hardware_manager.get_performance_metrics()
                    metrics.update({
                        'hardware_optimization_operations': hardware_stats.get('total_operations', 0),
                        'cpu_optimization_operations': hardware_stats.get('cpu_optimizations', 0),
                        'gpu_optimization_operations': hardware_stats.get('gpu_optimizations', 0),
                        'memory_optimization_operations': hardware_stats.get('memory_optimizations', 0),
                        'adaptive_optimization_operations': hardware_stats.get('adaptive_optimizations', 0)
                })
                else:
                    # Use default values if method doesn't exist
                    metrics.update({
                        'hardware_optimization_operations': 0,
                        'cpu_optimization_operations': 0,
                        'gpu_optimization_operations': 0,
                        'memory_optimization_operations': 0,
                        'adaptive_optimization_operations': 0
                })
            except Exception as e:
                tprint_warning(f"⚠️ Failed to get hardware optimization stats: {e}")
        
        return metrics

    def _get_vectorization_stats(self) -> Dict[str, Any]:
        """Get detailed vectorization statistics."""
        if not self.vectorization_manager or not self.optimization_enabled:
            return {'enabled': False}
        
        try:
            stats = self.vectorization_manager.get_performance_stats()
            analytics = self.vectorization_manager.get_performance_analytics()
            
            return {
                'enabled': True,
                'performance_stats': stats,
                'analytics': analytics,
                'memory_profiling': self.vectorization_manager.get_memory_profiling(),
                'cache_statistics': self.vectorization_manager.get_cache_statistics()
            }
        except Exception as e:
            tprint_warning(f"⚠️ Failed to get detailed vectorization stats: {e}")
            return {'enabled': True, 'error': str(e)}

    def _create_outcome_report(self, feature_sets: Dict[str, List[str]], shap_values: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Create comprehensive outcome report."""
        try:
            report = f"""# Final Feature Selection Outcome Report

**Execution Details:**
- **Symbol:** {config.get('symbol', 'unknown')}
- **Exchange:** {config.get('exchange', 'binance')}
- **Timeframe:** {config.get('timeframe', '15m')}
- **Execution Mode:** {config.get('execution_mode', 'light')}
- **Timestamp:** {datetime.now().isoformat()}

## Feature Selection Summary

**Feature Set Sizes:** {config.get('feature_set_sizes', [60, 50, 40])}

**Results:**
"""

            # Feature set details
            for set_name, feature_list in feature_sets.items():
                if set_name.startswith('selected_features_'):
                    size = set_name.split('_')[-1]
                    report += f"\n### Top {size} Features\n"
                    report += f"- **Count:** {len(feature_list)}\n"
                    if feature_list:
                        report += f"- **Top 5 Features:** {', '.join(feature_list[:5])}\n"

            # Feature scores if available
            if self.selection_component and self.selection_component.get_feature_scores():
                scores = self.selection_component.get_feature_scores()
                if scores:
                    report += "\n## Feature Importance Scores\n"
                    top_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:20]
                    for feature, score in top_scores:
                        report += f"- **{feature}:** {score:.6f}\n"

            # SHAP summary
            if shap_values:
                report += "\n## SHAP Analysis\n"
                report += f"- **SHAP sets generated:** {len(shap_values)}\n"

                for shap_name, shap_data in shap_values.items():
                    if shap_name.startswith('shap_values_'):
                        size = shap_name.split('_')[-1]
                        if isinstance(shap_data, dict) and 'feature_importance' in shap_data:
                            top_shap = sorted(shap_data['feature_importance'].items(),
                                            key=lambda x: x[1], reverse=True)[:5]
                            report += f"\n**Top SHAP Features ({size}):**\n"
                            for feature, importance in top_shap:
                                report += f"- {feature}: {importance:.6f}\n"

            # Configuration
            report += "\n## Configuration\n"
            report += f"- **Selection Method:** {config.get('selection_method', 'mutual_info')}\n"
            report += f"- **Scoring Threshold:** {config.get('scoring_threshold', 0.01)}\n"
            report += f"- **Tree-based Selection:** {config.get('use_tree_based', True)}\n"
            
            # Optimization Information
            report += "\n## Optimization Status\n"
            report += f"- **VectorBT Optimization:** {'Enabled' if self.optimization_enabled else 'Disabled'}\n"
            report += f"- **Hardware Optimization:** {'Enabled' if self.hardware_optimization_enabled else 'Disabled'}\n"
            report += f"- **Vectorization Manager:** {'Available' if self.vectorization_manager else 'Not Available'}\n"
            report += f"- **Rolling Optimizer:** {'Available' if self.rolling_optimizer else 'Not Available'}\n"
            report += f"- **Hardware Manager:** {'Available' if self.hardware_manager else 'Not Available'}\n"
            report += f"- **Adaptive Engine:** {'Available' if self.adaptive_engine else 'Not Available'}\n"
            report += f"- **CPU Optimizer:** {'Available' if self.cpu_optimizer else 'Not Available'}\n"
            report += f"- **GPU Manager:** {'Available' if self.gpu_manager else 'Not Available'}\n"
            report += f"- **Memory Optimizer:** {'Available' if self.memory_optimizer else 'Not Available'}\n"

            # Generated artifacts
            report += "\n## Generated Artifacts\n"
            artifact_count = len([name for name in feature_sets.keys() if name.startswith('selected_features_')]) * 2  # features + dataframes
            artifact_count += len(shap_values)
            artifact_count += 2  # feature_scores + selection_metadata

            report += f"- Feature sets: {len([name for name in feature_sets.keys() if name.startswith('selected_features_')])}\n"
            report += f"- Feature dataframes: {len([name for name in feature_sets.keys() if name.startswith('selected_feature_dataframe_')])}\n"
            report += f"- SHAP analyses: {len(shap_values)}\n"
            report += f"- Metadata and scores: 2\n"
            report += f"- **Total artifacts:** {artifact_count + 1}\n"  # +1 for the report

            report += f"""

---
*Generated by Feature Generation Final Feature Selection Step at {datetime.now().isoformat()}*
"""

            return report

        except Exception as e:
            tprint_error(f"⚠️ Failed to create outcome report: {e}")
            return f"# Final Feature Selection Outcome Report\n\nError creating report: {str(e)}"

    def _generate_markdown_report(self, outcome_report: Dict[str, Any], 
                                 feature_sets: Dict[str, List[str]], 
                                 shap_values: Dict[str, Any], 
                                 config: FinalFeatureSelectionConfig) -> str:
        """
        Generate a comprehensive markdown report for the final feature selection step.
        
        Args:
            outcome_report: The outcome report dictionary
            feature_sets: Dictionary of feature sets
            shap_values: SHAP values dictionary
            config: Configuration object
            
        Returns:
            Markdown formatted report string
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            report = f"""# Final Feature Selection Report

**Generated:** {timestamp}
**Step:** feature_generation_final_feature_selection_step

## Configuration

- **Symbol:** {getattr(self, 'symbol', 'N/A')}
- **Exchange:** {getattr(self, 'exchange', 'N/A')}
- **Timeframe:** {getattr(self, 'timeframe', 'N/A')}
- **Execution Mode:** {getattr(self, 'execution_mode', 'N/A')}
- **Feature Count Targets:** {config.get('feature_count_targets', 'N/A')}
- **Selection Method:** {config.get('selection_method', 'N/A')}
- **Optimization Enabled:** {self.optimization_enabled}

## Feature Selection Results

"""
            
            # Add feature set summaries
            for set_name, features in feature_sets.items():
                if set_name.startswith('selected_features_'):
                    count = set_name.split('_')[-1]
                    report += f"- **{count} Features Set:** {len(features)} features selected\n"
            
            report += f"\n- **Total Feature Sets:** {len([k for k in feature_sets.keys() if k.startswith('selected_features_')])}\n"
            
            # Add SHAP analysis summary
            if shap_values:
                report += f"\n## SHAP Analysis Summary\n\n"
                report += f"- **SHAP Analyses Generated:** {len(shap_values)}\n"
                for shap_name, shap_data in shap_values.items():
                    if isinstance(shap_data, dict) and 'top_features' in shap_data:
                        report += f"- **{shap_name}:** {len(shap_data['top_features'])} top features analyzed\n"
            
            # Add detailed feature lists with SHAP metrics
            report += f"\n## Selected Features by Set\n\n"
            
            for set_name, features in feature_sets.items():
                if set_name.startswith('selected_features_'):
                    count = set_name.split('_')[-1]
                    report += f"### {count} Features Set ({len(features)} features)\n\n"
                    
                    # Get SHAP values for this feature set if available
                    shap_key = f'shap_values_{count}'
                    feature_importance = {}
                    if shap_key in shap_values and isinstance(shap_values[shap_key], dict):
                        feature_importance = shap_values[shap_key].get('feature_importance', {})
                    
                    for i, feature in enumerate(features[:20], 1):  # Show first 20 features
                        shap_score = feature_importance.get(feature, 0.0)
                        report += f"{i}. {feature}"
                        if shap_score > 0:
                            report += f" (SHAP: {shap_score:.4f})"
                        report += "\n"
                    
                    if len(features) > 20:
                        report += f"... and {len(features) - 20} more features\n"
                    
                    # Add SHAP summary for this set
                    if shap_key in shap_values and isinstance(shap_values[shap_key], dict):
                        mean_abs_shap = shap_values[shap_key].get('mean_abs_shap', [])
                        if mean_abs_shap:
                            avg_shap = sum(mean_abs_shap) / len(mean_abs_shap)
                            report += f"\n**Average SHAP Importance:** {avg_shap:.4f}\n"
                    
                    report += "\n"
            
            # Add performance metrics
            report += f"## Performance Metrics\n\n"
            if isinstance(outcome_report, dict):
                report += f"- **Execution Time:** {outcome_report.get('execution_time', 'N/A')} seconds\n"
            else:
                report += f"- **Execution Time:** N/A seconds\n"
            report += f"- **Optimization Enabled:** {'Yes' if self.optimization_enabled else 'No'}\n"
            report += f"- **Hardware Optimization:** {'Yes' if self.hardware_optimization_enabled else 'No'}\n"
            
            # Add optimization details
            if self.optimization_enabled:
                report += f"\n## Optimization Details\n\n"
                report += f"- **VectorBT Optimization:** {'Enabled' if self.vectorization_manager else 'Disabled'}\n"
                report += f"- **Rolling Optimizer:** {'Available' if self.rolling_optimizer else 'Not Available'}\n"
                report += f"- **Hardware Manager:** {'Available' if self.hardware_manager else 'Not Available'}\n"
            
            # Add artifacts summary
            report += f"\n## Generated Artifacts\n\n"
            artifact_count = len([name for name in feature_sets.keys() if name.startswith('selected_features_')]) * 2
            artifact_count += len(shap_values) if shap_values else 0
            artifact_count += 2  # feature_scores + selection_metadata
            
            report += f"- **Feature Sets:** {len([name for name in feature_sets.keys() if name.startswith('selected_features_')])}\n"
            report += f"- **Feature DataFrames:** {len([name for name in feature_sets.keys() if name.startswith('selected_feature_dataframe_')])}\n"
            report += f"- **SHAP Analyses:** {len(shap_values) if shap_values else 0}\n"
            report += f"- **Metadata Files:** 2\n"
            report += f"- **Total Artifacts:** {artifact_count + 2}\n"  # +2 for pickle and markdown reports
            
            report += f"\n## Summary\n\n"
            report += f"Final feature selection completed successfully. Generated {len([k for k in feature_sets.keys() if k.startswith('selected_features_')])} optimized feature sets "
            report += f"with comprehensive SHAP analysis and metadata. All artifacts saved in both pickle and markdown formats.\n"
            
            report += f"\n---\n"
            report += f"*Generated by Feature Generation Final Feature Selection Step at {timestamp}*\n"
            
            return report
            
        except Exception as e:
            tprint_error(f"⚠️ Failed to generate markdown report: {e}")
            return f"# Final Feature Selection Report\n\nError generating report: {str(e)}"

    def _save_markdown_report(self, markdown_content: str, base_name: str) -> str:
        """
        Save a markdown report to the outcomes directory.
        
        Args:
            markdown_content: The markdown content to save
            base_name: Base name for the file
            
        Returns:
            Path where the markdown file was saved
        """
        try:
            # Create outcomes directory if it doesn't exist
            outcomes_dir = Path("outcomes")
            outcomes_dir.mkdir(exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{base_name}_report_{timestamp}.md"
            file_path = outcomes_dir / filename
            
            # Write markdown content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            tprint_success(f"✅ Markdown report saved: {file_path}")
            return str(file_path)
            
        except Exception as e:
            tprint_error(f"⚠️ Failed to save markdown report: {e}")
            raise


# Register the step
def register_feature_generation_final_feature_selection_step():
    """Register the feature generation final feature selection step."""
    from src.training.steps.base_step import step_registry

    step_registry.register("feature_generation_final_feature_selection_step", FeatureGenerationFinalFeatureSelectionStep)
    tprint("✅ Feature generation final feature selection step registered", "SUCCESS")


# Auto-register when module is imported
register_feature_generation_final_feature_selection_step()
