"""
Budgeted Feature Selection

This module implements the budgeted selection of features using a knapsack-style
optimization under compute/latency constraints. It selects the optimal subset
of features that maximizes expected utility while respecting budget constraints.

Key Features:
- Knapsack-style optimization with greedy approximation
- Coverage requirements for feature families
- Diversification penalty for correlated features
- Bang-per-buck ranking
- Budget constraint enforcement
"""

import logging
import time
import traceback
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.metrics.pairwise import cosine_similarity

# Import utilities
from .utils import FeatureGeneratorWrapper, compute_feature_correlations, apply_diversification_penalty
from .config import BudgetConfig, DataDrivenFeatureSelectionConfig, FeatureFamily

# Import matrix operations for efficient computation
try:
    from src.utils.matrix_operations.unified_operations import get_unified_matrix_operations
    from src.utils.matrix_operations.batch_operations import batch_correlation_analysis
    MATRIX_OPS_AVAILABLE = True
except ImportError:
    MATRIX_OPS_AVAILABLE = False

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


@dataclass
class BudgetedSelectionResult:
    """Result of budgeted feature selection."""
    selected_wrappers: List[FeatureGeneratorWrapper]
    rejected_wrappers: List[FeatureGeneratorWrapper]
    selection_metrics: Dict[str, Any]
    execution_time: float
    total_utility: float
    total_cost: float
    budget_utilization: float
    coverage_achieved: Dict[str, bool]
    
    # Performance metrics
    matrix_ops_used: int = 0
    vectorized_ops: int = 0
    optimization_iterations: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'selected_wrappers': [w.to_dict() for w in self.selected_wrappers],
            'rejected_wrappers': [w.to_dict() for w in self.rejected_wrappers],
            'selection_metrics': self.selection_metrics,
            'execution_time': self.execution_time,
            'total_utility': self.total_utility,
            'total_cost': self.total_cost,
            'budget_utilization': self.budget_utilization,
            'coverage_achieved': self.coverage_achieved,
            'matrix_ops_used': self.matrix_ops_used,
            'vectorized_ops': self.vectorized_ops,
            'optimization_iterations': self.optimization_iterations
        }


class BudgetedFeatureSelection:
    """Budgeted feature selection using knapsack-style optimization."""
    
    def __init__(self, config: BudgetConfig, matrix_ops=None):
        self.config = config
        self.matrix_ops = matrix_ops
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Performance tracking
        self.performance_metrics = {
            'matrix_ops_used': 0,
            'vectorized_ops': 0,
            'optimization_iterations': 0,
            'greedy_selections': 0,
            'coverage_enforcements': 0
        }
    
    def select_features(self, wrappers: List[FeatureGeneratorWrapper], 
                      data: pd.DataFrame, target: np.ndarray) -> BudgetedSelectionResult:
        """Select features under budget constraints."""
        start_time = time.time()
        
        try:
            tprint_info("🚀 Starting Budgeted Feature Selection")
            tprint_info(f"📊 Selecting from {len(wrappers)} candidates")
            tprint_info(f"💰 Budget: {self.config.feature_compute_p99_budget_ms}ms, Max features: {self.config.max_features_pre_selection}")
            
            # Pre-process wrappers
            processed_wrappers = self._preprocess_wrappers(wrappers)
            
            # Apply diversification penalty
            if self.config.diversification_penalty > 0:
                processed_wrappers = self._apply_diversification_penalty(processed_wrappers, data, target)
            
            # Compute bang-per-buck scores
            scored_wrappers = self._compute_bang_per_buck_scores(processed_wrappers)
            
            # Greedy selection
            selected_wrappers, rejected_wrappers = self._greedy_selection(scored_wrappers)
            
            # Enforce coverage requirements
            selected_wrappers = self._enforce_coverage_requirements(selected_wrappers, scored_wrappers)
            
            # Compute final metrics
            total_utility = sum(w.phase2_utility or 0.0 for w in selected_wrappers)
            total_cost = sum(w.total_cost for w in selected_wrappers)
            budget_utilization = total_cost / self.config.feature_compute_p99_budget_ms if self.config.feature_compute_p99_budget_ms > 0 else 0.0
            
            # Check coverage
            coverage_achieved = self._check_coverage_achievement(selected_wrappers)
            
            execution_time = time.time() - start_time
            
            # Create selection metrics
            selection_metrics = {
                'n_candidates': len(wrappers),
                'n_selected': len(selected_wrappers),
                'n_rejected': len(rejected_wrappers),
                'selection_ratio': len(selected_wrappers) / len(wrappers) if wrappers else 0.0,
                'utility_per_cost': total_utility / total_cost if total_cost > 0 else 0.0,
                'families_covered': len(set(w.family for w in selected_wrappers)),
                'required_families': len(self.config.required_families),
                'coverage_complete': all(coverage_achieved.values())
            }
            
            # Create result
            result = BudgetedSelectionResult(
                selected_wrappers=selected_wrappers,
                rejected_wrappers=rejected_wrappers,
                selection_metrics=selection_metrics,
                execution_time=execution_time,
                total_utility=total_utility,
                total_cost=total_cost,
                budget_utilization=budget_utilization,
                coverage_achieved=coverage_achieved,
                matrix_ops_used=self.performance_metrics['matrix_ops_used'],
                vectorized_ops=self.performance_metrics['vectorized_ops'],
                optimization_iterations=self.performance_metrics['optimization_iterations']
            )
            
            tprint_success(f"✅ Budgeted selection completed in {execution_time:.3f}s")
            tprint_success(f"📊 Selected {len(selected_wrappers)} features")
            tprint_success(f"💰 Budget utilization: {budget_utilization:.1%}")
            tprint_success(f"📈 Total utility: {total_utility:.3f}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Budgeted selection failed: {e}")
            self.logger.error(f"Error details: {traceback.format_exc()}")
            
            # Return empty result
            return BudgetedSelectionResult(
                selected_wrappers=[],
                rejected_wrappers=wrappers,
                selection_metrics={},
                execution_time=execution_time,
                total_utility=0.0,
                total_cost=0.0,
                budget_utilization=0.0,
                coverage_achieved={}
            )
    
    def _preprocess_wrappers(self, wrappers: List[FeatureGeneratorWrapper]) -> List[FeatureGeneratorWrapper]:
        """Pre-process wrappers for selection."""
        processed = []
        
        for wrapper in wrappers:
            try:
                # Ensure we have Phase 2 results
                if wrapper.phase2_utility is None:
                    wrapper.phase2_utility = 0.0
                if wrapper.phase2_uncertainty is None:
                    wrapper.phase2_uncertainty = 1.0
                if wrapper.phase2_stability is None:
                    wrapper.phase2_stability = 0.0
                
                # Apply cost penalties
                cost_penalty = (self.config.lambda_cost * wrapper.total_cost + 
                              self.config.lambda_uncertainty * wrapper.phase2_uncertainty + 
                              self.config.lambda_staleness * wrapper.total_cost)
                
                # Adjust utility
                adjusted_utility = wrapper.phase2_utility - cost_penalty
                wrapper.phase2_utility = max(0.0, adjusted_utility)  # Ensure non-negative
                
                processed.append(wrapper)
                
            except Exception as e:
                self.logger.warning(f"Failed to preprocess wrapper {wrapper.name}: {e}")
                continue
        
        return processed
    
    def _apply_diversification_penalty(self, wrappers: List[FeatureGeneratorWrapper], 
                                     data: pd.DataFrame, target: np.ndarray) -> List[FeatureGeneratorWrapper]:
        """Apply diversification penalty to highly correlated features."""
        try:
            # Generate proxy features for correlation analysis
            proxy_features = {}
            for wrapper in wrappers:
                try:
                    # Generate a simple proxy feature
                    proxy = self._generate_proxy_feature(wrapper, data)
                    if proxy is not None and len(proxy) > 10:
                        proxy_features[wrapper.name] = proxy
                except Exception as e:
                    self.logger.debug(f"Failed to generate proxy for {wrapper.name}: {e}")
                    continue
            
            if len(proxy_features) < 2:
                return wrappers
            
            # Compute correlations
            correlations = compute_feature_correlations(
                pd.DataFrame(proxy_features), 
                threshold=self.config.correlation_threshold
            )
            
            # Apply diversification penalty
            for wrapper in wrappers:
                if wrapper.name in correlations:
                    n_correlated = len(correlations[wrapper.name])
                    penalty_factor = 1.0 - (self.config.diversification_penalty * n_correlated)
                    wrapper.phase2_utility *= penalty_factor
            
            return wrappers
            
        except Exception as e:
            self.logger.warning(f"Failed to apply diversification penalty: {e}")
            return wrappers
    
    def _generate_proxy_feature(self, wrapper: FeatureGeneratorWrapper, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Generate a proxy feature for correlation analysis."""
        try:
            if hasattr(wrapper.generator, 'generate'):
                result = wrapper.generator.generate(data, lookback=10)  # Use default lookback
                
                if hasattr(result, 'data'):
                    return result.data.values
                elif isinstance(result, pd.Series):
                    return result.values
                elif isinstance(result, np.ndarray):
                    return result
                else:
                    return None
            else:
                return None
                
        except Exception as e:
            self.logger.debug(f"Failed to generate proxy for {wrapper.name}: {e}")
            return None
    
    def _compute_bang_per_buck_scores(self, wrappers: List[FeatureGeneratorWrapper]) -> List[FeatureGeneratorWrapper]:
        """Compute bang-per-buck scores for wrappers."""
        for wrapper in wrappers:
            try:
                if wrapper.total_cost > 0:
                    bang_per_buck = wrapper.phase2_utility / wrapper.total_cost
                else:
                    bang_per_buck = wrapper.phase2_utility
                
                # Store as a temporary attribute
                wrapper.bang_per_buck = bang_per_buck
                
            except Exception as e:
                self.logger.warning(f"Failed to compute bang-per-buck for {wrapper.name}: {e}")
                wrapper.bang_per_buck = 0.0
        
        return wrappers
    
    def _greedy_selection(self, wrappers: List[FeatureGeneratorWrapper]) -> Tuple[List[FeatureGeneratorWrapper], List[FeatureGeneratorWrapper]]:
        """Greedy selection based on bang-per-buck scores."""
        # Sort by bang-per-buck (descending)
        sorted_wrappers = sorted(wrappers, key=lambda w: getattr(w, 'bang_per_buck', 0.0), reverse=True)
        
        selected = []
        rejected = []
        total_cost = 0.0
        
        for wrapper in sorted_wrappers:
            # Check if adding this wrapper would exceed budget
            if (total_cost + wrapper.total_cost <= self.config.feature_compute_p99_budget_ms and 
                len(selected) < self.config.max_features_pre_selection):
                
                selected.append(wrapper)
                total_cost += wrapper.total_cost
                self.performance_metrics['greedy_selections'] += 1
            else:
                rejected.append(wrapper)
        
        return selected, rejected
    
    def _enforce_coverage_requirements(self, selected_wrappers: List[FeatureGeneratorWrapper], 
                                     all_wrappers: List[FeatureGeneratorWrapper]) -> List[FeatureGeneratorWrapper]:
        """Enforce coverage requirements for required families."""
        try:
            # Check current coverage
            current_families = set(w.family for w in selected_wrappers)
            required_families = {f.value for f in self.config.required_families}
            
            missing_families = required_families - current_families
            
            if not missing_families:
                return selected_wrappers
            
            tprint_info(f"🔧 Enforcing coverage for missing families: {missing_families}")
            
            # Find best wrappers for missing families
            additional_wrappers = []
            
            for missing_family in missing_families:
                family_wrappers = [w for w in all_wrappers if w.family == missing_family]
                
                if not family_wrappers:
                    self.logger.warning(f"No wrappers found for required family: {missing_family}")
                    continue
                
                # Sort by utility and select best one
                best_wrapper = max(family_wrappers, key=lambda w: w.phase2_utility or 0.0)
                
                # Check if we can add it within budget
                current_total_cost = sum(w.total_cost for w in selected_wrappers)
                if (current_total_cost + best_wrapper.total_cost <= self.config.feature_compute_p99_budget_ms and 
                    len(selected_wrappers) + len(additional_wrappers) < self.config.max_features_pre_selection):
                    
                    additional_wrappers.append(best_wrapper)
                    self.performance_metrics['coverage_enforcements'] += 1
                else:
                    self.logger.warning(f"Cannot add {best_wrapper.name} due to budget constraints")
            
            # Add additional wrappers
            selected_wrappers.extend(additional_wrappers)
            
            return selected_wrappers
            
        except Exception as e:
            self.logger.warning(f"Failed to enforce coverage requirements: {e}")
            return selected_wrappers
    
    def _check_coverage_achievement(self, selected_wrappers: List[FeatureGeneratorWrapper]) -> Dict[str, bool]:
        """Check if coverage requirements are achieved."""
        current_families = set(w.family for w in selected_wrappers)
        required_families = {f.value for f in self.config.required_families}
        
        coverage = {}
        for family in required_families:
            coverage[family] = family in current_families
        
        return coverage
    
    def _optimize_with_linear_programming(self, wrappers: List[FeatureGeneratorWrapper]) -> List[FeatureGeneratorWrapper]:
        """Alternative optimization using linear programming (for comparison)."""
        try:
            if len(wrappers) < 2:
                return wrappers
            
            # Prepare data for linear programming
            n_wrappers = len(wrappers)
            utilities = np.array([w.phase2_utility or 0.0 for w in wrappers])
            costs = np.array([w.total_cost for w in wrappers])
            
            # Objective: maximize utility
            c = -utilities  # Minimize negative utility (maximize utility)
            
            # Constraint: total cost <= budget
            A_ub = costs.reshape(1, -1)
            b_ub = [self.config.feature_compute_p99_budget_ms]
            
            # Constraint: number of features <= max_features
            A_ub = np.vstack([A_ub, np.ones((1, n_wrappers))])
            b_ub.append(self.config.max_features_pre_selection)
            
            # Bounds: 0 <= x <= 1 (binary variables)
            bounds = [(0, 1) for _ in range(n_wrappers)]
            
            # Solve linear program
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
            if result.success:
                # Select wrappers with x > 0.5
                selected_indices = [i for i, x in enumerate(result.x) if x > 0.5]
                selected_wrappers = [wrappers[i] for i in selected_indices]
                
                self.performance_metrics['optimization_iterations'] += 1
                return selected_wrappers
            else:
                self.logger.warning(f"Linear programming failed: {result.message}")
                return self._greedy_selection(wrappers)[0]
                
        except Exception as e:
            self.logger.warning(f"Linear programming optimization failed: {e}")
            return self._greedy_selection(wrappers)[0]
    
    def _compute_selection_quality_metrics(self, selected_wrappers: List[FeatureGeneratorWrapper]) -> Dict[str, float]:
        """Compute quality metrics for the selection."""
        if not selected_wrappers:
            return {}
        
        try:
            utilities = [w.phase2_utility or 0.0 for w in selected_wrappers]
            costs = [w.total_cost for w in selected_wrappers]
            stabilities = [w.phase2_stability or 0.0 for w in selected_wrappers]
            
            metrics = {
                'mean_utility': np.mean(utilities),
                'std_utility': np.std(utilities),
                'mean_cost': np.mean(costs),
                'std_cost': np.std(costs),
                'mean_stability': np.mean(stabilities),
                'std_stability': np.std(stabilities),
                'utility_per_cost': np.sum(utilities) / np.sum(costs) if np.sum(costs) > 0 else 0.0,
                'diversity_score': self._compute_diversity_score(selected_wrappers)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Failed to compute selection quality metrics: {e}")
            return {}
    
    def _compute_diversity_score(self, selected_wrappers: List[FeatureGeneratorWrapper]) -> float:
        """Compute diversity score for selected wrappers."""
        try:
            if len(selected_wrappers) < 2:
                return 1.0
            
            # Count unique families
            families = set(w.family for w in selected_wrappers)
            family_diversity = len(families) / len(selected_wrappers)
            
            # Count unique categories
            categories = set(w.category for w in selected_wrappers)
            category_diversity = len(categories) / len(selected_wrappers)
            
            # Combine diversity scores
            diversity_score = 0.7 * family_diversity + 0.3 * category_diversity
            
            return min(1.0, diversity_score)
            
        except Exception as e:
            self.logger.debug(f"Failed to compute diversity score: {e}")
            return 0.0