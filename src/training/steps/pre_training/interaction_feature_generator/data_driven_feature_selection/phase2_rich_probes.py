"""
Phase 2: Rich Probes for Feature Selection

This module implements Phase 2 of the data-driven feature selection system,
which performs rigorous data-driven lookback optimization for the promising
features that survived Phase 1.

Key Features:
- Bayesian lookback optimization (spline/GP IC surface)
- Hierarchical shrinkage across families and symbols
- Discrete/blend decision with penalties
- Stability-under-shift testing
- Data availability requirements
- HDI width requirements
"""

import logging
import time
import traceback
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy import stats

# Import the lookback optimization system
try:
    from ..feature_interaction_generation.orchestrator import LookbackOptimizationOrchestrator
    from ..feature_interaction_generation.config import create_production_config, FamilyType
    LOOKBACK_OPTIMIZATION_AVAILABLE = True
except ImportError:
    LOOKBACK_OPTIMIZATION_AVAILABLE = False
    FamilyType = None

# Import utilities
from .utils import FeatureGeneratorWrapper, UtilityEstimator, CostEstimator
from .config import Phase2Config, DataDrivenFeatureSelectionConfig

# Import matrix operations for efficient computation
try:
    from src.utils.matrix_operations.unified_operations import get_unified_matrix_operations
    from src.utils.matrix_operations.hardware_integration import HardwareOptimizedMatrixProcessor
    MATRIX_OPS_AVAILABLE = True
except ImportError:
    MATRIX_OPS_AVAILABLE = False

# Import utilities
try:
    from src.utils.tprint import (
        tprint,
        tprint_info,
        tprint_error,
        tprint_warning,
        tprint_success,
        tprint_performance,
        tprint_debug,
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print(*args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_performance(*args, **kwargs): print("PERFORMANCE:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class Phase2Result:
    """Result of Phase 2 rich probes."""
    selected_wrappers: List[FeatureGeneratorWrapper]
    rejected_wrappers: List[FeatureGeneratorWrapper]
    optimization_results: Dict[str, Any]
    execution_time: float
    n_optimized: int
    n_stability_failed: int
    n_availability_failed: int
    n_hdi_failed: int
    
    # Performance metrics
    matrix_ops_used: int = 0
    vectorized_ops: int = 0
    memory_efficient_ops: int = 0
    bayesian_optimizations: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        tprint_debug(
            "[Phase2Result] Serializing results",
            {
                'selected': len(self.selected_wrappers),
                'rejected': len(self.rejected_wrappers),
                'optimizations': self.n_optimized,
            }
        )

        return {
            'selected_wrappers': [w.to_dict() for w in self.selected_wrappers],
            'rejected_wrappers': [w.to_dict() for w in self.rejected_wrappers],
            'optimization_results': self.optimization_results,
            'execution_time': self.execution_time,
            'n_optimized': self.n_optimized,
            'n_stability_failed': self.n_stability_failed,
            'n_availability_failed': self.n_availability_failed,
            'n_hdi_failed': self.n_hdi_failed,
            'matrix_ops_used': self.matrix_ops_used,
            'vectorized_ops': self.vectorized_ops,
            'memory_efficient_ops': self.memory_efficient_ops,
            'bayesian_optimizations': self.bayesian_optimizations
        }


class Phase2RichProbes:
    """Phase 2: Rich probes with Bayesian lookback optimization."""
    
    def __init__(self, config: Phase2Config, matrix_ops=None, hardware_processor=None):
        self.config = config
        self.matrix_ops = matrix_ops
        self.hardware_processor = hardware_processor
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize lookback optimization system
        self.lookback_optimizer = None
        if LOOKBACK_OPTIMIZATION_AVAILABLE and config.enable_bayesian_optimization:
            try:
                optimization_config = create_production_config()
                self.lookback_optimizer = LookbackOptimizationOrchestrator(optimization_config)
                tprint_success("✅ Lookback optimization system initialized")
            except Exception as e:
                tprint_warning(f"⚠️ Failed to initialize lookback optimization: {e}")
                self.lookback_optimizer = None
        
        # Initialize estimators
        self.utility_estimator = UtilityEstimator(matrix_ops)
        self.cost_estimator = CostEstimator(matrix_ops)
        
        # Performance tracking
        self.performance_metrics = {
            'matrix_ops_used': 0,
            'vectorized_ops': 0,
            'memory_efficient_ops': 0,
            'bayesian_optimizations': 0,
            'optimizations_completed': 0,
            'stability_tests_failed': 0,
            'availability_tests_failed': 0,
            'hdi_tests_failed': 0
        }

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def _log_info(self, message: str) -> None:
        """Log informational message to both logger and tprint."""
        self.logger.info(message)
        tprint_info(f"[Phase2RichProbes] {message}")

    def _log_debug(self, message: str) -> None:
        """Log debug message to both logger and tprint."""
        self.logger.debug(message)
        tprint_debug(f"[Phase2RichProbes] {message}")

    def _log_warning(self, message: str) -> None:
        """Log warning message to both logger and tprint."""
        self.logger.warning(message)
        tprint_warning(f"[Phase2RichProbes] {message}")

    def _log_error(self, message: str) -> None:
        """Log error message to both logger and tprint."""
        self.logger.error(message)
        tprint_error(f"[Phase2RichProbes] {message}")

    def _log_success(self, message: str) -> None:
        """Log success message to both logger and tprint."""
        self.logger.info(message)
        tprint_success(f"[Phase2RichProbes] {message}")
    
    def run_phase2(self, wrappers: List[FeatureGeneratorWrapper], 
                  data: pd.DataFrame, target: np.ndarray) -> Phase2Result:
        """Run Phase 2 rich probes with Bayesian optimization."""
        start_time = time.time()
        
        try:
            tprint_info("🚀 Starting Phase 2: Rich Probes with Bayesian Optimization")
            tprint_info(f"📊 Optimizing {len(wrappers)} promising generators")
            
            # Group wrappers by family for hierarchical optimization
            family_groups = self._group_wrappers_by_family(wrappers)
            
            # Run Bayesian optimization for each family
            optimization_results = {}
            optimized_wrappers = []
            
            for family, family_wrappers in family_groups.items():
                tprint_info(f"🔧 Optimizing {family} family ({len(family_wrappers)} generators)")
                
                try:
                    family_result = self._optimize_family(family, family_wrappers, data, target)
                    optimization_results[family] = family_result
                    optimized_wrappers.extend(family_result['optimized_wrappers'])
                    
                except Exception as e:
                    self.logger.warning(f"Failed to optimize {family} family: {e}")
                    # Add wrappers with default values
                    for wrapper in family_wrappers:
                        wrapper.phase2_utility = 0.0
                        wrapper.phase2_uncertainty = 1.0
                        wrapper.phase2_stability = 0.0
                        optimized_wrappers.append(wrapper)
            
            # Apply Phase 2 gating decisions
            tprint_info("🚪 Applying Phase 2 gating decisions...")
            selected_wrappers, rejected_wrappers = self._apply_phase2_gating(optimized_wrappers)
            
            execution_time = time.time() - start_time
            
            # Create result
            result = Phase2Result(
                selected_wrappers=selected_wrappers,
                rejected_wrappers=rejected_wrappers,
                optimization_results=optimization_results,
                execution_time=execution_time,
                n_optimized=self.performance_metrics['optimizations_completed'],
                n_stability_failed=self.performance_metrics['stability_tests_failed'],
                n_availability_failed=self.performance_metrics['availability_tests_failed'],
                n_hdi_failed=self.performance_metrics['hdi_tests_failed'],
                matrix_ops_used=self.performance_metrics['matrix_ops_used'],
                vectorized_ops=self.performance_metrics['vectorized_ops'],
                memory_efficient_ops=self.performance_metrics['memory_efficient_ops'],
                bayesian_optimizations=self.performance_metrics['bayesian_optimizations']
            )
            
            tprint_success(f"✅ Phase 2 completed in {execution_time:.3f}s")
            tprint_success(f"📊 Selected {len(selected_wrappers)} generators from {len(wrappers)} total")
            tprint_success(f"🔧 Completed {result.n_optimized} optimizations")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Phase 2 failed: {e}")
            self.logger.error(f"Error details: {traceback.format_exc()}")
            
            # Return empty result
            return Phase2Result(
                selected_wrappers=[],
                rejected_wrappers=wrappers,
                optimization_results={},
                execution_time=execution_time,
                n_optimized=0,
                n_stability_failed=0,
                n_availability_failed=0,
                n_hdi_failed=0
            )
    
    def _group_wrappers_by_family(self, wrappers: List[FeatureGeneratorWrapper]) -> Dict[str, List[FeatureGeneratorWrapper]]:
        """Group wrappers by feature family."""
        self._log_debug(f"Grouping {len(wrappers)} wrappers by family")
        families = {}

        for wrapper in wrappers:
            family = wrapper.family
            if family not in families:
                families[family] = []
            families[family].append(wrapper)

        summary = ", ".join(f"{family}({len(members)})" for family, members in families.items())
        self._log_debug(f"Grouped wrappers into families: {summary if summary else 'none'}")

        return families
    
    def _optimize_family(self, family: str, wrappers: List[FeatureGeneratorWrapper],
                        data: pd.DataFrame, target: np.ndarray) -> Dict[str, Any]:
        """Optimize a family of feature generators using Bayesian lookback optimization."""
        try:
            self._log_info(f"Optimizing family '{family}' with {len(wrappers)} wrappers")
            if self.lookback_optimizer is None:
                # Fallback to simple optimization
                self._log_warning(f"Lookback optimizer unavailable; using simple optimization for {family}")
                return self._simple_family_optimization(family, wrappers, data, target)

            # Prepare data for optimization
            market_data = {family: data}
            targets = {family: target}
            feature_names = {family: [w.name for w in wrappers]}

            # Run Bayesian optimization
            start_time = time.time()
            optimization_result = self.lookback_optimizer.optimize_lookbacks(
                market_data, targets, feature_names
            )
            duration = time.time() - start_time
            tprint_performance(f"[Phase2RichProbes] Bayesian optimization for {family} took {duration:.3f}s")

            if not optimization_result.success:
                self._log_warning(f"Bayesian optimization reported failure for {family}; using simple fallback")
                return self._simple_family_optimization(family, wrappers, data, target)

            # Extract results for each wrapper
            optimized_wrappers = []

            for wrapper in wrappers:
                try:
                    # Get optimization result for this wrapper
                    wrapper_result = self._extract_wrapper_optimization_result(
                        wrapper, optimization_result, family
                    )

                    if wrapper_result:
                        self._log_debug(f"Using Bayesian optimization result for wrapper {wrapper.name}")
                        optimized_wrappers.append(wrapper_result)
                    else:
                        # Fallback to simple optimization
                        self._log_warning(f"No Bayesian result for {wrapper.name}; falling back to simple optimization")
                        simple_result = self._simple_wrapper_optimization(wrapper, data, target)
                        optimized_wrappers.append(simple_result)

                except Exception as e:
                    self._log_warning(f"Failed to extract optimization result for {wrapper.name}: {e}")
                    simple_result = self._simple_wrapper_optimization(wrapper, data, target)
                    optimized_wrappers.append(simple_result)

            self.performance_metrics['bayesian_optimizations'] += 1
            self.performance_metrics['optimizations_completed'] += len(optimized_wrappers)

            self._log_success(
                f"Completed Bayesian optimization for {family}; optimized {len(optimized_wrappers)} wrappers"
            )

            return {
                'family': family,
                'optimization_success': True,
                'optimized_wrappers': optimized_wrappers,
                'optimization_result': optimization_result.to_dict()
            }

        except Exception as e:
            self._log_warning(f"Family optimization failed for {family}: {e}")
            return self._simple_family_optimization(family, wrappers, data, target)
    
    def _extract_wrapper_optimization_result(self, wrapper: FeatureGeneratorWrapper,
                                           optimization_result: Any, family: str) -> Optional[FeatureGeneratorWrapper]:
        """Extract optimization result for a specific wrapper."""
        try:
            self._log_debug(f"Extracting optimization result for {wrapper.name} in family {family}")
            # This would extract the specific optimization result for the wrapper
            # from the overall optimization result

            # For now, use a simplified approach
            if hasattr(optimization_result, 'decisions') and family in optimization_result.decisions:
                family_decisions = optimization_result.decisions[family]
                
                # Find decision for this wrapper (simplified)
                for decision in family_decisions.values():
                    if hasattr(decision, 'lookback_spec'):
                        # Extract utility and uncertainty from decision
                        wrapper.phase2_utility = getattr(decision, 'utility', 0.0)
                        wrapper.phase2_uncertainty = getattr(decision, 'uncertainty', 1.0)
                        wrapper.phase2_stability = getattr(decision, 'stability', 0.0)
                        
                        # Apply stability test
                        if self.config.enable_stability_test:
                            stability_passed = self._test_stability(wrapper, family_decisions)
                            if not stability_passed:
                                wrapper.phase2_utility = 0.0
                                self.performance_metrics['stability_tests_failed'] += 1
                                self._log_warning(
                                    f"Wrapper {wrapper.name} failed stability test; utility reset"
                                )

                        # Check data availability
                        if self._check_data_availability(wrapper):
                            self.performance_metrics['availability_tests_failed'] += 1
                            wrapper.phase2_utility = 0.0
                            self._log_warning(
                                f"Wrapper {wrapper.name} failed data availability check; utility reset"
                            )

                        # Check HDI width
                        if self._check_hdi_width(wrapper):
                            self.performance_metrics['hdi_tests_failed'] += 1
                            wrapper.phase2_utility = 0.0
                            self._log_warning(
                                f"Wrapper {wrapper.name} failed HDI width check; utility reset"
                            )

                        self._log_debug(
                            f"Extracted Bayesian optimization metrics for {wrapper.name}: "
                            f"utility={wrapper.phase2_utility:.4f}, uncertainty={wrapper.phase2_uncertainty:.4f}, "
                            f"stability={wrapper.phase2_stability:.4f}"
                        )
                        return wrapper

            return None

        except Exception as e:
            self._log_debug(f"Failed to extract optimization result for {wrapper.name}: {e}")
            return None
    
    def _simple_family_optimization(self, family: str, wrappers: List[FeatureGeneratorWrapper],
                                  data: pd.DataFrame, target: np.ndarray) -> Dict[str, Any]:
        """Simple fallback optimization for a family."""
        self._log_info(f"Running simple optimization for family '{family}' with {len(wrappers)} wrappers")
        optimized_wrappers = []

        for wrapper in wrappers:
            try:
                simple_result = self._simple_wrapper_optimization(wrapper, data, target)
                optimized_wrappers.append(simple_result)
            except Exception as e:
                self._log_warning(f"Simple optimization failed for {wrapper.name}: {e}")
                # Set default values
                wrapper.phase2_utility = 0.0
                wrapper.phase2_uncertainty = 1.0
                wrapper.phase2_stability = 0.0
                optimized_wrappers.append(wrapper)

        self.performance_metrics['optimizations_completed'] += len(optimized_wrappers)
        self._log_success(
            f"Simple optimization for {family} produced {len(optimized_wrappers)} wrappers"
        )

        return {
            'family': family,
            'optimization_success': False,
            'optimized_wrappers': optimized_wrappers,
            'optimization_result': None
        }
    
    def _simple_wrapper_optimization(self, wrapper: FeatureGeneratorWrapper,
                                   data: pd.DataFrame, target: np.ndarray) -> FeatureGeneratorWrapper:
        """Simple optimization for a single wrapper."""
        try:
            self._log_debug(f"Simple optimization for wrapper {wrapper.name} (family {wrapper.family})")
            # Generate multiple proxy features with different lookbacks
            lookbacks = self._get_family_lookbacks(wrapper.family)
            utilities = []
            uncertainties = []
            stabilities = []

            for lookback in lookbacks:
                try:
                    # Generate feature with this lookback
                    if hasattr(wrapper.generator, 'generate'):
                        self._log_debug(f"Generating feature for {wrapper.name} with lookback {lookback}")
                        result = wrapper.generator.generate(data, lookback=lookback)

                        if hasattr(result, 'data'):
                            feature = result.data.values
                        elif isinstance(result, pd.Series):
                            feature = result.values
                        elif isinstance(result, np.ndarray):
                            feature = result
                        else:
                            continue
                        
                        if len(feature) < 10:
                            continue
                        
                        # Compute IC and stability
                        ic, ic_error = self._compute_ic_with_bootstrap(feature, target)
                        stability = self._compute_stability_score(feature, target)

                        utilities.append(ic)
                        uncertainties.append(ic_error)
                        stabilities.append(stability)
                        self._log_debug(
                            f"Lookback {lookback} for {wrapper.name}: IC={ic:.4f}, IC_err={ic_error:.4f}, stability={stability:.4f}"
                        )

                except Exception as e:
                    self._log_debug(
                        f"Failed to generate feature for {wrapper.name} with lookback {lookback}: {e}"
                    )
                    continue

            if utilities:
                # Use best utility
                best_idx = np.argmax(utilities)
                wrapper.phase2_utility = utilities[best_idx]
                wrapper.phase2_uncertainty = uncertainties[best_idx]
                wrapper.phase2_stability = stabilities[best_idx]
                self._log_success(
                    f"Wrapper {wrapper.name} best lookback produced utility={wrapper.phase2_utility:.4f}, "
                    f"uncertainty={wrapper.phase2_uncertainty:.4f}, stability={wrapper.phase2_stability:.4f}"
                )
            else:
                wrapper.phase2_utility = 0.0
                wrapper.phase2_uncertainty = 1.0
                wrapper.phase2_stability = 0.0
                self._log_warning(f"Wrapper {wrapper.name} produced no valid utilities; defaulting to neutral values")

            return wrapper

        except Exception as e:
            self._log_warning(f"Simple wrapper optimization failed for {wrapper.name}: {e}")
            wrapper.phase2_utility = 0.0
            wrapper.phase2_uncertainty = 1.0
            wrapper.phase2_stability = 0.0
            return wrapper
    
    def _get_family_lookbacks(self, family: str) -> List[int]:
        """Get appropriate lookbacks for a feature family."""
        lookback_map = {
            'momentum': [5, 8, 12, 16, 20, 25],
            'volatility': [6, 10, 14, 18, 22, 26],
            'rsi': [7, 10, 14, 18, 21, 25],
            'vwap': [10, 15, 20, 25, 30, 35],
            'trend': [5, 10, 15, 20, 25, 30],
            'volume': [5, 10, 15, 20, 25, 30],
            'oscillator': [7, 10, 14, 18, 21, 25],
            'returns': [1, 2, 3, 5, 7, 10],
            'support_resistance': [10, 15, 20, 25, 30, 35],
            'order_flow': [5, 8, 12, 16, 20, 25],
            'microstructure': [5, 8, 12, 16, 20, 25],
            'regime': [10, 15, 20, 25, 30, 35],
            'entropy': [10, 15, 20, 25, 30, 35],
            'acceleration': [5, 8, 12, 16, 20, 25],
            'time': [1, 2, 3, 5, 7, 10],
            'normalization': [5, 10, 15, 20, 25, 30],
            'representation_learning': [10, 15, 20, 25, 30, 35]
        }
        
        return lookback_map.get(family, [5, 10, 15, 20, 25, 30])
    
    def _test_stability(self, wrapper: FeatureGeneratorWrapper, family_decisions: Dict) -> bool:
        """Test stability under shift for a wrapper."""
        try:
            if not self.config.enable_stability_test:
                return True

            # This would implement the stability-under-shift test
            # For now, use a simplified version based on the stability score

            if wrapper.phase2_stability is None:
                self._log_warning(f"Wrapper {wrapper.name} lacks stability metric")
                return False

            # Check if stability is above threshold
            passed = wrapper.phase2_stability >= self.config.stability_threshold
            self._log_debug(
                f"Stability check for {wrapper.name}: stability={wrapper.phase2_stability:.4f}, "
                f"threshold={self.config.stability_threshold:.4f}, passed={passed}"
            )
            return passed

        except Exception as e:
            self._log_debug(f"Stability test encountered error for {wrapper.name}: {e}")
            return False
    
    def _check_data_availability(self, wrapper: FeatureGeneratorWrapper) -> bool:
        """Check data availability requirements for a wrapper."""
        try:
            if wrapper.family in self.config.book_dependent_families:
                availability_value = (
                    wrapper.data_availability if wrapper.data_availability is not None else 0.0
                )
                available = availability_value >= self.config.min_data_availability
                self._log_debug(
                    f"Data availability for {wrapper.name}: availability={availability_value:.4f}, "
                    f"threshold={self.config.min_data_availability:.4f}, meets_requirement={available}"
                )
                return available
            self._log_debug(f"Data availability check skipped for {wrapper.name} (family {wrapper.family})")
            return True

        except Exception as e:
            self._log_debug(f"Data availability check failed for {wrapper.name}: {e}")
            return True
    
    def _check_hdi_width(self, wrapper: FeatureGeneratorWrapper) -> bool:
        """Check HDI width requirements for a wrapper."""
        try:
            if wrapper.phase2_uncertainty is None:
                self._log_warning(f"Wrapper {wrapper.name} missing uncertainty estimate for HDI check")
                return True

            # Convert uncertainty to log-space HDI width
            hdi_width = 2 * wrapper.phase2_uncertainty  # Approximate 95% HDI

            passes = hdi_width <= self.config.max_hdi_width
            self._log_debug(
                f"HDI width check for {wrapper.name}: width={hdi_width:.4f}, "
                f"threshold={self.config.max_hdi_width:.4f}, passes={passes}"
            )

            return passes

        except Exception as e:
            self._log_debug(f"HDI width check failed for {wrapper.name}: {e}")
            return True
    
    def _apply_phase2_gating(self, wrappers: List[FeatureGeneratorWrapper]) -> Tuple[List[FeatureGeneratorWrapper], List[FeatureGeneratorWrapper]]:
        """Apply Phase 2 gating decisions."""
        self._log_info(f"Applying gating to {len(wrappers)} optimized wrappers")
        selected = []
        rejected = []

        for wrapper in wrappers:
            # Check utility threshold
            if wrapper.phase2_utility is None or wrapper.phase2_utility <= self.config.min_utility_threshold:
                rejected.append(wrapper)
                self._log_debug(
                    f"Rejecting {wrapper.name} due to utility {wrapper.phase2_utility} <= {self.config.min_utility_threshold}"
                )
                continue

            # Check uncertainty threshold
            if wrapper.phase2_uncertainty is None or wrapper.phase2_uncertainty > 0.8:
                rejected.append(wrapper)
                self._log_debug(
                    f"Rejecting {wrapper.name} due to uncertainty {wrapper.phase2_uncertainty} > 0.8"
                )
                continue

            # Check stability threshold using configured tolerance for sign flips
            if (
                wrapper.phase2_stability is None
                or wrapper.phase2_stability < self.config.stability_threshold
            ):
                rejected.append(wrapper)
                self._log_debug(
                    f"Rejecting {wrapper.name} due to stability {wrapper.phase2_stability} < {self.config.stability_threshold}"
                )
                continue

            selected.append(wrapper)
            self._log_debug(f"Selected {wrapper.name} for Phase 2 output")

        self._log_success(
            f"Gating completed: selected {len(selected)} wrappers, rejected {len(rejected)} wrappers"
        )
        return selected, rejected
    
    def _compute_ic_with_bootstrap(self, feature: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
        """Compute IC with block bootstrap standard errors."""
        try:
            # Remove NaN values
            valid_mask = np.isfinite(feature) & np.isfinite(target)
            if np.sum(valid_mask) < 10:
                self._log_warning("Insufficient valid samples for IC computation; returning defaults")
                return 0.0, 1.0

            feature_clean = feature[valid_mask]
            target_clean = target[valid_mask]

            # Compute correlation (IC)
            ic = np.corrcoef(feature_clean, target_clean)[0, 1]

            if np.isnan(ic):
                self._log_warning("Computed IC is NaN; returning defaults")
                return 0.0, 1.0

            # Block bootstrap for standard error
            n_samples = min(100, len(feature_clean) // 4)
            block_size = max(1, len(feature_clean) // 10)
            bootstrap_ics = []
            self._log_debug(
                f"Starting bootstrap for IC computation with {n_samples} samples and block size {block_size}"
            )

            for _ in range(n_samples):
                # Block bootstrap
                n_blocks = len(feature_clean) // block_size
                if n_blocks < 2:
                    # Fall back to regular bootstrap
                    indices = np.random.choice(len(feature_clean), size=len(feature_clean), replace=True)
                else:
                    # Block bootstrap
                    block_indices = np.random.choice(n_blocks, size=n_blocks, replace=True)
                    indices = []
                    for block_idx in block_indices:
                        start_idx = block_idx * block_size
                        end_idx = min(start_idx + block_size, len(feature_clean))
                        indices.extend(range(start_idx, end_idx))
                    indices = np.array(indices[:len(feature_clean)])
                
                f_boot = feature_clean[indices]
                t_boot = target_clean[indices]
                
                if len(np.unique(f_boot)) > 1 and len(np.unique(t_boot)) > 1:
                    boot_ic = np.corrcoef(f_boot, t_boot)[0, 1]
                    if not np.isnan(boot_ic):
                        bootstrap_ics.append(boot_ic)

            if bootstrap_ics:
                ic_error = np.std(bootstrap_ics)
            else:
                ic_error = np.sqrt((1 - ic**2) / (len(feature_clean) - 2))

            self._log_debug(
                f"Bootstrap IC computation result: IC={ic:.4f}, IC_error={ic_error:.4f}, samples={len(feature_clean)}"
            )
            return float(ic), float(ic_error)

        except Exception as e:
            self._log_debug(f"Failed to compute IC with bootstrap: {e}")
            return 0.0, 1.0
    
    def _compute_stability_score(self, feature: np.ndarray, target: np.ndarray) -> float:
        """Compute stability score for feature."""
        try:
            # Remove NaN values
            valid_mask = np.isfinite(feature) & np.isfinite(target)
            if np.sum(valid_mask) < 20:
                self._log_warning("Insufficient data for stability computation; returning default")
                return 0.0

            feature_clean = feature[valid_mask]
            target_clean = target[valid_mask]

            # Split data into thirds
            n = len(feature_clean)
            third = n // 3

            if third < 5:
                self._log_warning("Too few observations per segment for stability computation; returning default")
                return 0.0

            # Oldest third
            old_feature = feature_clean[:third]
            old_target = target_clean[:third]

            # Newest third
            new_feature = feature_clean[-third:]
            new_target = target_clean[-third:]

            # Compute IC for each third
            if len(np.unique(old_feature)) > 1 and len(np.unique(old_target)) > 1:
                old_ic = np.corrcoef(old_feature, old_target)[0, 1]
            else:
                old_ic = 0.0

            if len(np.unique(new_feature)) > 1 and len(np.unique(new_target)) > 1:
                new_ic = np.corrcoef(new_feature, new_target)[0, 1]
            else:
                new_ic = 0.0

            # Stability score based on consistency
            if np.isnan(old_ic) or np.isnan(new_ic):
                self._log_warning("Stability IC contains NaN; returning default")
                return 0.0

            # Higher stability if both ICs have same sign and similar magnitude
            if old_ic * new_ic > 0:  # Same sign
                stability = 1.0 - abs(old_ic - new_ic) / (abs(old_ic) + abs(new_ic) + 1e-6)
            else:  # Different signs
                stability = 0.0

            stability_score = max(0.0, min(1.0, stability))
            self._log_debug(
                f"Computed stability score: old_ic={old_ic:.4f}, new_ic={new_ic:.4f}, score={stability_score:.4f}"
            )
            return stability_score

        except Exception as e:
            self._log_debug(f"Failed to compute stability score: {e}")
            return 0.0
