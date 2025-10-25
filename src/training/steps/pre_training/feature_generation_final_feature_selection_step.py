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
import pandas as pd
import numpy as np
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
    LearningAlgorithm,
    OptimizationStrategy
)

from src.utils.hardware.advanced_cpu_optimizer import (
    AdvancedM1CPUOptimizer,
    WorkloadProfile
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
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning, tprint_error
from src.utils.artifact_manager import ArtifactManager

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
            labeled_df = self._get_artifact('labeled_dataframe')
            targets = self._get_artifact('targets')

            if labeled_df is None or targets is None:
                raise ValueError("Required artifacts 'labeled_dataframe' and 'targets' not found")

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
            artifacts = self._generate_artifacts(feature_sets, shap_values, config)

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

            # Save outcome report
            report_path = self._save_artifact(
                outcome_report,
                "final_feature_selection_outcome_report",
                artifact_type="report"
            )

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
            logger.error(error_msg, exc_info=True)

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

        # Get features from interaction generation steps
        try:
            analyst_interactions = self._get_artifact('interaction_features')
            if analyst_interactions is not None:
                features_data['analyst_interactions'] = analyst_interactions
        except Exception as e:
            tprint_warning(f"⚠️ Could not get analyst interaction features: {e}")

        try:
            tactician_interactions = self._get_artifact('interaction_features')  # Same artifact name for both
            if tactician_interactions is not None:
                features_data['tactician_interactions'] = tactician_interactions
        except Exception as e:
            tprint_warning(f"⚠️ Could not get tactician interaction features: {e}")

        # Get features from feature selection step
        try:
            selected_features = self._get_artifact('selected_features')
            if selected_features is not None:
                features_data['selected_features'] = selected_features
        except Exception as e:
            tprint_warning(f"⚠️ Could not get selected features: {e}")

        # Get feature dataframe from feature generation step
        try:
            feature_df = self._get_artifact('feature_dataframe')
            if feature_df is not None:
                features_data['feature_dataframe'] = feature_df
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
            await self.hardware_manager.initialize()
            tprint_success("✅ Unified hardware manager initialized")
            
            # Initialize adaptive optimization engine
            self.adaptive_engine = AdaptiveOptimizationEngine(
                learning_algorithm=LearningAlgorithm.DECISION_TREE,
                optimization_strategy=OptimizationStrategy.BALANCED,
                enable_learning=True,
                auto_tuning_enabled=True
            )
            await self.adaptive_engine.initialize()
            tprint_success("✅ Adaptive optimization engine initialized")
            
            # Initialize CPU optimizer
            self.cpu_optimizer = AdvancedM1CPUOptimizer(
                workload_profile=WorkloadProfile.FEATURE_ENGINEERING,
                optimization_level=OptimizationLevel.BALANCED,
                enable_thermal_monitoring=True,
                enable_power_management=True
            )
            await self.cpu_optimizer.initialize()
            tprint_success("✅ Advanced CPU optimizer initialized")
            
            # Initialize GPU manager
            self.gpu_manager = EnhancedM1GPUManager(
                operation_type=GPUOperationType.MATRIX_OPERATIONS,
                optimization_level=OptimizationLevel.BALANCED,
                enable_mps_acceleration=True,
                enable_gpu_memory_pooling=True,
                enable_batch_operations=True
            )
            await self.gpu_manager.initialize()
            tprint_success("✅ Enhanced GPU manager initialized")
            
            # Initialize memory optimizer
            self.memory_optimizer = AdvancedM1MemoryOptimizer(
                memory_strategy=MemoryStrategy.ADAPTIVE,
                optimization_level=OptimizationLevel.BALANCED,
                memory_limit_gb=config.get('memory_limit_gb', 8.0),
                enable_memory_pooling=True,
                enable_predictive_allocation=True,
                enable_compression=True
            )
            await self.memory_optimizer.initialize()
            tprint_success("✅ Advanced memory optimizer initialized")
            
            tprint_success("✅ All hardware optimization components initialized")
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to initialize hardware optimization components: {e}")
            self.hardware_optimization_enabled = False
            tprint_warning("⚠️ Continuing without hardware optimizations")

    def _combine_features(self, features_data: Dict[str, Any], labeled_df: pd.DataFrame) -> pd.DataFrame:
        """Combine features from different sources into a single DataFrame with VectorBT optimizations."""
        tprint_info("🔄 Combining features with VectorBT optimizations...")
        
        # Start with the labeled dataframe (OHLCV + labels)
        base_features = labeled_df.copy()

        # Use vectorization manager for optimized operations if available
        if self.vectorization_manager and self.optimization_enabled:
            try:
                # Optimize the base dataframe for memory efficiency
                base_features = self.vectorization_manager.optimize_dataframe(base_features)
                tprint_info("✅ Base features optimized for memory efficiency")
            except Exception as e:
                tprint_warning(f"⚠️ Memory optimization failed: {e}")

        # Add features from feature dataframe if available
        if 'feature_dataframe' in features_data and features_data['feature_dataframe'] is not None:
            feature_df = features_data['feature_dataframe']
            
            # Optimize feature dataframe if vectorization manager is available
            if self.vectorization_manager and self.optimization_enabled:
                try:
                    feature_df = self.vectorization_manager.optimize_dataframe(feature_df)
                except Exception as e:
                    tprint_warning(f"⚠️ Feature dataframe optimization failed: {e}")
            
            # Find common columns (excluding OHLCV and target columns)
            ohlcv_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
            target_cols = ['target', 'label', 'return']  # Common target column names

            feature_cols = [col for col in feature_df.columns
                          if col not in ohlcv_cols and col not in target_cols]

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
                
                tprint_info(f"📊 Added {len(feature_cols)} features from feature dataframe")

        # Add interaction features if available
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
                    
                    # Add interaction features (excluding any duplicate columns)
                    interaction_cols = [col for col in interaction_df.columns
                                      if col not in base_features.columns]
                    if interaction_cols:
                        base_features = pd.concat([base_features, interaction_df[interaction_cols]], axis=1)
                        tprint_info(f"📊 Added {len(interaction_cols)} {interaction_type} features")

        # Remove any non-numeric columns except timestamp
        numeric_cols = []
        for col in base_features.columns:
            if col == 'timestamp' or pd.api.types.is_numeric_dtype(base_features[col]):
                numeric_cols.append(col)

        result_df = base_features[numeric_cols].copy()

        # Handle NaN values with optimized operations
        if self.vectorization_manager and self.optimization_enabled:
            try:
                # Use vectorized operations for NaN handling
                tprint_info("🔄 Optimizing NaN handling...")
                
                # Drop columns with too many NaN values
                nan_threshold = int(0.7 * len(result_df))
                valid_cols = []
                for col in result_df.columns:
                    if result_df[col].count() >= nan_threshold:
                        valid_cols.append(col)
                
                result_df = result_df[valid_cols]
                
                # Fill remaining NaN with median using vectorized operations
                numeric_cols_only = result_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols_only) > 0:
                    medians = result_df[numeric_cols_only].median()
                    result_df[numeric_cols_only] = result_df[numeric_cols_only].fillna(medians)
                
                tprint_success("✅ NaN handling optimized")
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
        """Perform feature selection for multiple feature set sizes with VectorBT optimizations."""
        # Define feature set sizes
        feature_set_sizes = config.get('feature_set_sizes', [60, 50, 40])

        feature_sets = {}

        # Separate features from targets
        feature_cols = [col for col in features_df.columns
                       if col not in ['target', 'label', 'return', 'timestamp']]
        target_cols = [col for col in ['target', 'label', 'return']
                      if col in features_df.columns]

        if not target_cols:
            raise ValueError("No target column found in features dataframe")

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

    def _generate_shap_values(self, feature_sets: Dict[str, List[str]], features_df: pd.DataFrame, targets: pd.Series, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate SHAP values for interpretability."""
        shap_values = {}

        try:
            # Import SHAP (optional import)
            import shap
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split

            # Get target column
            target_cols = [col for col in ['target', 'label', 'return']
                          if col in features_df.columns]
            if not target_cols:
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

            # Create SHAP explainer
            explainer = shap.TreeExplainer(rf_model)

            # Calculate SHAP values for each feature set
            for set_name, feature_list in feature_sets.items():
                if set_name.startswith('selected_features_'):
                    size = set_name.split('_')[-1]

                    if len(feature_list) > 0:
                        # Get SHAP values for this feature set
                        shap_test = explainer.shap_values(X_test[feature_list])

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

    def _generate_artifacts(self, feature_sets: Dict[str, List[str]], shap_values: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
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
            'total_features_available': len([col for col in self._get_artifact('labeled_dataframe', pd.DataFrame()).columns
                                           if col not in ['target', 'label', 'return', 'timestamp']]),
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
                hardware_stats = self.hardware_manager.get_performance_metrics()
                metrics.update({
                    'hardware_optimization_operations': hardware_stats.get('total_operations', 0),
                    'cpu_optimization_operations': hardware_stats.get('cpu_optimizations', 0),
                    'gpu_optimization_operations': hardware_stats.get('gpu_optimizations', 0),
                    'memory_optimization_operations': hardware_stats.get('memory_optimizations', 0),
                    'adaptive_optimization_operations': hardware_stats.get('adaptive_optimizations', 0)
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




# Register the step
def register_feature_generation_final_feature_selection_step():
    """Register the feature generation final feature selection step."""
    from src.training.steps.base_step import step_registry

    step_registry.register("feature_generation_final_feature_selection_step", FeatureGenerationFinalFeatureSelectionStep)
    tprint("✅ Feature generation final feature selection step registered", "SUCCESS")


# Auto-register when module is imported
register_feature_generation_final_feature_selection_step()
