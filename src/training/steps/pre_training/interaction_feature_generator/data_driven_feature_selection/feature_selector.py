"""
Data-Driven Feature Selector

This module implements the main orchestrator for the data-driven feature selection
system, coordinating Phase 1, Phase 2, budgeted selection, and final model selection.

Key Features:
- Two-phase gating process (cheap → expensive)
- Budgeted selection with knapsack optimization
- Interaction feature generation
- Final model-level selection
- Comprehensive monitoring and reporting
"""

import logging
import time
import traceback
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

# Import phase modules
from .phase1_cheap_probes import Phase1CheapProbes, Phase1Result
from .phase2_rich_probes import Phase2RichProbes, Phase2Result
from .budgeted_selection import BudgetedFeatureSelection, BudgetedSelectionResult
from .interaction_generator import InteractionFeatureGenerator, InteractionResult
from .final_model_selection import FinalModelSelection, FinalSelectionResult
from .config import DataDrivenFeatureSelectionConfig, create_development_config, create_production_config
from .utils import create_feature_generator_wrappers, filter_wrappers_by_availability

# Import matrix operations and hardware optimizations
try:
    from src.utils.matrix_operations.unified_operations import get_unified_matrix_operations
    from src.utils.matrix_operations.hardware_integration import HardwareOptimizedMatrixProcessor
    from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager
    MATRIX_OPS_AVAILABLE = True
    HARDWARE_AVAILABLE = True
except ImportError:
    MATRIX_OPS_AVAILABLE = False
    HARDWARE_AVAILABLE = False

# Import utilities
try:
    from src.utils.tprint import tprint, tprint_info, tprint_error, tprint_warning, tprint_success, tprint_performance
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print(*args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_performance(*args, **kwargs): print("PERFORMANCE:", *args, **kwargs)

# Set up logging
logger = logging.getLogger(__name__)


def _log_info(message: str, active_logger: Optional[logging.Logger] = None) -> None:
    """Log an informational message and mirror it through tprint."""
    (active_logger or logger).info(message)
    tprint_info(message)


def _log_warning(message: str, active_logger: Optional[logging.Logger] = None) -> None:
    """Log a warning message and mirror it through tprint."""
    (active_logger or logger).warning(message)
    tprint_warning(message)


def _log_error(message: str, active_logger: Optional[logging.Logger] = None) -> None:
    """Log an error message and mirror it through tprint."""
    (active_logger or logger).error(message)
    tprint_error(message)


def _log_success(message: str, active_logger: Optional[logging.Logger] = None) -> None:
    """Log a success message and mirror it through tprint."""
    (active_logger or logger).info(message)
    tprint_success(message)


def _log_performance(message: str, active_logger: Optional[logging.Logger] = None) -> None:
    """Log a performance-related message and mirror it through tprint."""
    (active_logger or logger).info(message)
    tprint_performance(message)


def _log_debug(message: str, active_logger: Optional[logging.Logger] = None) -> None:
    """Log a debug message and mirror it through tprint."""
    (active_logger or logger).debug(message)
    tprint(message)


@dataclass
class FeatureSelectionResult:
    """Complete result of the data-driven feature selection process."""
    # Phase results
    phase1_result: Optional[Phase1Result] = None
    phase2_result: Optional[Phase2Result] = None
    budgeted_result: Optional[BudgetedSelectionResult] = None
    interaction_result: Optional[InteractionResult] = None
    final_result: Optional[FinalSelectionResult] = None
    split_metadata: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Final selections
    selected_features: List[str] = field(default_factory=list)
    selected_interactions: List[str] = field(default_factory=list)
    final_feature_matrix: Optional[np.ndarray] = None
    final_feature_names: List[str] = field(default_factory=list)
    
    # Performance metrics
    total_execution_time: float = 0.0
    total_features_evaluated: int = 0
    total_features_selected: int = 0
    budget_utilization: float = 0.0
    coverage_achieved: Dict[str, bool] = field(default_factory=dict)
    
    # System metrics
    matrix_ops_used: int = 0
    hardware_accelerated_ops: int = 0
    memory_efficient_ops: int = 0
    bayesian_optimizations: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        _log_debug("📝 Serializing FeatureSelectionResult to dictionary")
        return {
            'phase1_result': self.phase1_result.to_dict() if self.phase1_result else None,
            'phase2_result': self.phase2_result.to_dict() if self.phase2_result else None,
            'budgeted_result': self.budgeted_result.to_dict() if self.budgeted_result else None,
            'interaction_result': self.interaction_result.to_dict() if self.interaction_result else None,
            'final_result': self.final_result.to_dict() if self.final_result else None,
            'selected_features': self.selected_features,
            'selected_interactions': self.selected_interactions,
            'final_feature_matrix_shape': self.final_feature_matrix.shape if self.final_feature_matrix is not None else None,
            'final_feature_names': self.final_feature_names,
            'total_execution_time': self.total_execution_time,
            'total_features_evaluated': self.total_features_evaluated,
            'total_features_selected': self.total_features_selected,
            'budget_utilization': self.budget_utilization,
            'coverage_achieved': self.coverage_achieved,
            'matrix_ops_used': self.matrix_ops_used,
            'hardware_accelerated_ops': self.hardware_accelerated_ops,
            'memory_efficient_ops': self.memory_efficient_ops,
            'bayesian_optimizations': self.bayesian_optimizations,
            'split_metadata': {
                key: indices.tolist()
                for key, indices in self.split_metadata.items()
            }
        }


class DataDrivenFeatureSelector:
    """Main orchestrator for data-driven feature selection."""
    
    def __init__(self, config: Optional[DataDrivenFeatureSelectionConfig] = None):
        self.config = config or create_production_config()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        config_type = getattr(self.config, 'name', 'custom')
        _log_info(f"🚀 Initializing DataDrivenFeatureSelector with {config_type} configuration", self.logger)

        # Initialize hardware optimizations
        self._initialize_hardware_optimizations()

        # Initialize matrix operations
        self._initialize_matrix_operations()
        
        # Initialize phase components
        self._initialize_phase_components()
        
        # Performance tracking
        self.performance_metrics = {
            'matrix_ops_used': 0,
            'hardware_accelerated_ops': 0,
            'memory_efficient_ops': 0,
            'bayesian_optimizations': 0,
            'total_execution_time': 0.0
        }
    
    def _initialize_hardware_optimizations(self):
        """Initialize hardware optimization components."""
        if not HARDWARE_AVAILABLE:
            _log_warning("Hardware optimizations not available, using CPU-only mode", self.logger)
            return

        try:
            # Initialize unified hardware manager
            self.hardware_manager = UnifiedHardwareManager()
            
            # Initialize hardware-optimized matrix processor
            if MATRIX_OPS_AVAILABLE:
                from src.utils.matrix_operations.hardware_integration import HardwareConfig
                hardware_config = HardwareConfig(
                    max_memory_gb=self.config.memory_limit_gb,
                    enable_gpu=self.config.enable_parallel_processing,
                    max_cpu_cores=self.config.max_workers,
                    auto_optimize_dtypes=True,
                    auto_chunk_large_data=True
                )
                self.hardware_processor = HardwareOptimizedMatrixProcessor(hardware_config)
            else:
                self.hardware_processor = None
            
            _log_success("✅ Hardware optimizations initialized", self.logger)

        except Exception as e:
            _log_warning(f"⚠️ Failed to initialize hardware optimizations: {e}", self.logger)
            self.hardware_manager = None
            self.hardware_processor = None

    def _initialize_matrix_operations(self):
        """Initialize matrix operations for vectorized computations."""
        if not MATRIX_OPS_AVAILABLE:
            _log_warning("Matrix operations not available, using basic numpy operations", self.logger)
            self.matrix_ops = None
            return
        
        try:
            # Initialize unified matrix operations
            self.matrix_ops = get_unified_matrix_operations(
                enable_gpu=self.config.enable_parallel_processing,
                enable_memory_optimization=True,
                enable_parallel=self.config.enable_parallel_processing
            )
            
            _log_success("✅ Matrix operations initialized", self.logger)

        except Exception as e:
            _log_warning(f"⚠️ Failed to initialize matrix operations: {e}", self.logger)
            self.matrix_ops = None
    
    def _initialize_phase_components(self):
        """Initialize phase components."""
        try:
            # Phase 1: Cheap probes
            self.phase1 = Phase1CheapProbes(self.config.phase1, self.matrix_ops)
            
            # Phase 2: Rich probes
            self.phase2 = Phase2RichProbes(self.config.phase2, self.matrix_ops, self.hardware_processor)
            
            # Budgeted selection
            self.budgeted_selection = BudgetedFeatureSelection(self.config.budget, self.matrix_ops)
            
            # Interaction generation
            self.interaction_generator = InteractionFeatureGenerator(self.config.interaction, self.matrix_ops)
            
            # Final model selection
            self.final_selection = FinalModelSelection(self.config.final_selection, self.matrix_ops)
            
            _log_success("✅ Phase components initialized", self.logger)

        except Exception as e:
            _log_error(f"❌ Failed to initialize phase components: {e}", self.logger)
            raise
    
    async def select_features(self, data: pd.DataFrame, target: np.ndarray, 
                            data_availability: Optional[Dict[str, float]] = None) -> FeatureSelectionResult:
        """Run the complete data-driven feature selection process."""
        start_time = time.time()

        try:
            _log_info("🚀 Starting Data-Driven Feature Selection", self.logger)
            _log_info(f"📊 Data shape: {data.shape}, Target length: {len(target)}", self.logger)

            # Initialize result
            result = FeatureSelectionResult()

            try:
                split_metadata = FinalModelSelection.build_default_split_metadata(len(data))
            except ValueError as exc:
                _log_error(f"❌ Unable to derive split metadata: {exc}", self.logger)
                raise

            result.split_metadata = {key: value.copy() for key, value in split_metadata.items()}
            _log_info(
                "🧮 Derived default train/val/test splits for feature selection",
                self.logger
            )

            # Step 1: Create feature generator wrappers
            _log_info("🔧 Step 1: Creating feature generator wrappers...", self.logger)
            wrappers = await self._create_feature_wrappers()

            if not wrappers:
                _log_error("❌ No feature generators available", self.logger)
                return result
            
            # Filter by data availability
            if data_availability:
                wrappers = filter_wrappers_by_availability(wrappers, data_availability)
                _log_info(f"📊 Filtered to {len(wrappers)} wrappers based on data availability", self.logger)
            
            result.total_features_evaluated = len(wrappers)
            
            # Step 2: Phase 1 - Cheap Probes
            _log_info("🔍 Step 2: Phase 1 - Cheap Probes", self.logger)
            phase1_result = self.phase1.run_phase1(wrappers, data, target)
            result.phase1_result = phase1_result
            
            if not phase1_result.selected_wrappers:
                _log_warning("⚠️ No features selected in Phase 1", self.logger)
                return result

            _log_success(f"✅ Phase 1 completed: {len(phase1_result.selected_wrappers)} features selected", self.logger)

            # Step 3: Phase 2 - Rich Probes
            _log_info("🔧 Step 3: Phase 2 - Rich Probes with Bayesian Optimization", self.logger)
            phase2_result = self.phase2.run_phase2(phase1_result.selected_wrappers, data, target)
            result.phase2_result = phase2_result
            
            if not phase2_result.selected_wrappers:
                _log_warning("⚠️ No features selected in Phase 2", self.logger)
                return result

            _log_success(f"✅ Phase 2 completed: {len(phase2_result.selected_wrappers)} features selected", self.logger)

            # Step 4: Budgeted Selection
            _log_info("💰 Step 4: Budgeted Selection", self.logger)
            budgeted_result = self.budgeted_selection.select_features(
                phase2_result.selected_wrappers, data, target
            )
            result.budgeted_result = budgeted_result
            
            if not budgeted_result.selected_wrappers:
                _log_warning("⚠️ No features selected in budgeted selection", self.logger)
                return result

            _log_success(f"✅ Budgeted selection completed: {len(budgeted_result.selected_wrappers)} features selected", self.logger)

            # Step 5: Interaction Generation
            _log_info("🔗 Step 5: Interaction Feature Generation", self.logger)
            interaction_result = self.interaction_generator.generate_interactions(
                budgeted_result.selected_wrappers, data, target
            )
            result.interaction_result = interaction_result
            
            _log_success(f"✅ Interaction generation completed: {len(interaction_result.selected_interactions)} interactions selected", self.logger)

            # Step 6: Final Model Selection
            _log_info("🎯 Step 6: Final Model Selection", self.logger)
            final_result = self.final_selection.select_final_features(
                budgeted_result.selected_wrappers,
                interaction_result.selected_interactions,
                data,
                target,
                split_metadata=split_metadata
            )
            result.final_result = final_result
            if final_result.split_metadata:
                result.split_metadata = {
                    key: value.copy()
                    for key, value in final_result.split_metadata.items()
                }

            # Compile final results
            result.selected_features = [w.name for w in budgeted_result.selected_wrappers]
            result.selected_interactions = interaction_result.selected_interactions
            result.final_feature_names = final_result.final_feature_names
            result.final_feature_matrix = final_result.final_feature_matrix
            result.total_features_selected = len(result.final_feature_names)
            result.budget_utilization = budgeted_result.budget_utilization
            result.coverage_achieved = budgeted_result.coverage_achieved
            
            # Update performance metrics
            execution_time = time.time() - start_time
            result.total_execution_time = execution_time
            self.performance_metrics['total_execution_time'] = execution_time
            
            # Aggregate performance metrics
            result.matrix_ops_used = sum([
                phase1_result.matrix_ops_used,
                phase2_result.matrix_ops_used,
                budgeted_result.matrix_ops_used
            ])
            result.hardware_accelerated_ops = sum([
                phase1_result.hardware_accelerated_ops,
                phase2_result.hardware_accelerated_ops,
                budgeted_result.hardware_accelerated_ops
            ])
            result.memory_efficient_ops = sum([
                phase1_result.memory_efficient_ops,
                phase2_result.memory_efficient_ops,
                budgeted_result.memory_efficient_ops
            ])
            result.bayesian_optimizations = phase2_result.bayesian_optimizations

            self.performance_metrics.update({
                'matrix_ops_used': result.matrix_ops_used,
                'hardware_accelerated_ops': result.hardware_accelerated_ops,
                'memory_efficient_ops': result.memory_efficient_ops,
                'bayesian_optimizations': result.bayesian_optimizations
            })
            _log_performance("⚡ Updated performance metrics after feature selection run", self.logger)

            _log_success(f"🎉 Data-driven feature selection completed in {execution_time:.3f}s", self.logger)
            _log_success(f"📊 Final selection: {result.total_features_selected} features", self.logger)
            _log_success(f"💰 Budget utilization: {result.budget_utilization:.1%}", self.logger)
            _log_success(f"📈 Coverage achieved: {sum(result.coverage_achieved.values())}/{len(result.coverage_achieved)} families", self.logger)

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            _log_error(f"Data-driven feature selection failed: {e}", self.logger)
            _log_debug(f"Error details: {traceback.format_exc()}", self.logger)
            self.performance_metrics['total_execution_time'] = execution_time

            # Return partial result
            result = FeatureSelectionResult()
            result.total_execution_time = execution_time
            return result
    
    async def _create_feature_wrappers(self) -> List:
        """Create feature generator wrappers from the feature bank."""
        try:
            # Create wrappers from feature bank
            wrappers = create_feature_generator_wrappers()

            if not wrappers:
                _log_warning("⚠️ No feature generators found in feature bank", self.logger)
                return []
            
            # Filter out excluded categories and generators
            filtered_wrappers = []
            for wrapper in wrappers:
                if (wrapper.category not in self.config.exclude_categories and 
                    not any(excluded in wrapper.name for excluded in self.config.exclude_generators)):
                    filtered_wrappers.append(wrapper)
            
            _log_info(f"📊 Created {len(filtered_wrappers)} feature generator wrappers", self.logger)
            return filtered_wrappers

        except Exception as e:
            _log_error(f"❌ Failed to create feature wrappers: {e}", self.logger)
            return []
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {
            'total_execution_time': self.performance_metrics['total_execution_time'],
            'matrix_ops_used': self.performance_metrics['matrix_ops_used'],
            'hardware_accelerated_ops': self.performance_metrics['hardware_accelerated_ops'],
            'memory_efficient_ops': self.performance_metrics['memory_efficient_ops'],
            'bayesian_optimizations': self.performance_metrics['bayesian_optimizations'],
            'hardware_available': HARDWARE_AVAILABLE,
            'matrix_ops_available': MATRIX_OPS_AVAILABLE,
            'config': self.config.to_dict()
        }
        _log_performance("📈 Generated performance summary for DataDrivenFeatureSelector", self.logger)
        return summary
    
    def save_results(self, result: FeatureSelectionResult, filepath: str) -> bool:
        """Save selection results to file."""
        try:
            import json
            
            with open(filepath, 'w') as f:
                json.dump(result.to_dict(), f, indent=2, default=str)

            _log_success(f"✅ Results saved to {filepath}", self.logger)
            return True

        except Exception as e:
            _log_error(f"❌ Failed to save results: {e}", self.logger)
            return False
    
    def load_results(self, filepath: str) -> Optional[FeatureSelectionResult]:
        """Load selection results from file."""
        try:
            import json
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Reconstruct result object
            result = FeatureSelectionResult()
            result.selected_features = data.get('selected_features', [])
            result.selected_interactions = data.get('selected_interactions', [])
            result.final_feature_names = data.get('final_feature_names', [])
            result.total_execution_time = data.get('total_execution_time', 0.0)
            result.total_features_evaluated = data.get('total_features_evaluated', 0)
            result.total_features_selected = data.get('total_features_selected', 0)
            result.budget_utilization = data.get('budget_utilization', 0.0)
            result.coverage_achieved = data.get('coverage_achieved', {})
            
            _log_success(f"✅ Results loaded from {filepath}", self.logger)
            return result

        except Exception as e:
            _log_error(f"❌ Failed to load results: {e}", self.logger)
            return None


# Convenience functions
async def select_features_development(data: pd.DataFrame, target: np.ndarray,
                                   data_availability: Optional[Dict[str, float]] = None) -> FeatureSelectionResult:
    """Select features using development configuration (fast, less thorough)."""
    config = create_development_config()
    selector = DataDrivenFeatureSelector(config)
    _log_info("🛠️ Executing development feature selection workflow")
    return await selector.select_features(data, target, data_availability)


async def select_features_production(data: pd.DataFrame, target: np.ndarray, 
                                   data_availability: Optional[Dict[str, float]] = None) -> FeatureSelectionResult:
    """Select features using production configuration (thorough, robust)."""
    config = create_production_config()
    selector = DataDrivenFeatureSelector(config)
    _log_info("🏭 Executing production feature selection workflow")
    return await selector.select_features(data, target, data_availability)


async def select_features_custom(data: pd.DataFrame, target: np.ndarray, 
                               config: DataDrivenFeatureSelectionConfig,
                               data_availability: Optional[Dict[str, float]] = None) -> FeatureSelectionResult:
    """Select features using custom configuration."""
    selector = DataDrivenFeatureSelector(config)
    _log_info("⚙️ Executing custom feature selection workflow")
    return await selector.select_features(data, target, data_availability)

