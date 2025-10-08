"""
Interaction Feature Generator

This module generates interaction features from the selected parent features,
ensuring that both parents are available and meet utility requirements.

Key Features:
- Parent availability enforcement
- Multiple interaction types (multiplication, division, addition, subtraction)
- Correlation-based parent selection
- Utility-based interaction evaluation
- Budget constraint compliance
"""

import logging
import time
import traceback
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
from itertools import combinations

# Import utilities
from .utils import FeatureGeneratorWrapper
from .config import InteractionConfig

# Import matrix operations for efficient computation
try:
    from src.utils.matrix_operations.unified_operations import get_unified_matrix_operations
    from src.utils.matrix_operations.batch_operations import batch_matrix_multiply
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
class InteractionFeature:
    """Represents an interaction feature."""
    name: str
    parent1: str
    parent2: str
    interaction_type: str
    utility: float = 0.0
    stability: float = 0.0
    cost: float = 0.0

    def __post_init__(self) -> None:
        """Emit creation details for observability."""
        tprint_info(
            f"🧩 InteractionFeature created: {self.name} "
            f"[{self.parent1} {self.interaction_type} {self.parent2}] "
            f"utility={self.utility:.4f}, stability={self.stability:.4f}"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'parent1': self.parent1,
            'parent2': self.parent2,
            'interaction_type': self.interaction_type,
            'utility': self.utility,
            'stability': self.stability,
            'cost': self.cost
        }


@dataclass
class InteractionResult:
    """Result of interaction feature generation."""
    selected_interactions: List[InteractionFeature]
    rejected_interactions: List[InteractionFeature]
    parent_features: List[str]
    execution_time: float
    n_interactions_generated: int
    n_interactions_selected: int
    
    # Performance metrics
    matrix_ops_used: int = 0
    vectorized_ops: int = 0

    def __post_init__(self) -> None:
        """Log summary of the interaction generation run."""
        tprint_success(
            f"📦 InteractionResult prepared | parents={len(self.parent_features)} "
            f"generated={self.n_interactions_generated} selected={self.n_interactions_selected} "
            f"time={self.execution_time:.3f}s"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'selected_interactions': [i.to_dict() for i in self.selected_interactions],
            'rejected_interactions': [i.to_dict() for i in self.rejected_interactions],
            'parent_features': self.parent_features,
            'execution_time': self.execution_time,
            'n_interactions_generated': self.n_interactions_generated,
            'n_interactions_selected': self.n_interactions_selected,
            'matrix_ops_used': self.matrix_ops_used,
            'vectorized_ops': self.vectorized_ops
        }


class InteractionFeatureGenerator:
    """Generates interaction features from selected parent features."""
    
    def __init__(self, config: InteractionConfig, matrix_ops=None):
        self.config = config
        self.matrix_ops = matrix_ops
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Performance tracking
        self.performance_metrics = {
            'matrix_ops_used': 0,
            'vectorized_ops': 0,
            'interactions_generated': 0,
            'interactions_evaluated': 0
        }

        tprint_info(
            f"🛠️ InteractionFeatureGenerator initialized | "
            f"max_interactions={self.config.max_interactions}, "
            f"utility_threshold={self.config.interaction_utility_threshold:.4f}, "
            f"require_both_parents={self.config.require_both_parents}"
        )

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def _log_debug(self, message: str) -> None:
        self.logger.debug(message)
        tprint_info(f"🔍 {message}")

    def _log_warning(self, message: str) -> None:
        self.logger.warning(message)
        tprint_warning(message)

    def _log_error(self, message: str) -> None:
        self.logger.error(message)
        tprint_error(message)

    def _log_performance(self, message: str) -> None:
        self.logger.info(message)
        tprint_performance(message)
    
    def generate_interactions(self, selected_wrappers: List[FeatureGeneratorWrapper], 
                            data: pd.DataFrame, target: np.ndarray) -> InteractionResult:
        """Generate interaction features from selected parent features."""
        start_time = time.time()
        
        try:
            tprint_info("🚀 Starting Interaction Feature Generation")
            tprint_info(f"📊 Generating interactions from {len(selected_wrappers)} parent features")
            
            # Filter parent features by utility requirements
            parent_features = self._filter_parent_features(selected_wrappers)
            
            if len(parent_features) < 2:
                tprint_warning("⚠️ Insufficient parent features for interaction generation")
                return InteractionResult(
                    selected_interactions=[],
                    rejected_interactions=[],
                    parent_features=[w.name for w in parent_features],
                    execution_time=time.time() - start_time,
                    n_interactions_generated=0,
                    n_interactions_selected=0
                )
            
            # Generate all possible interactions
            all_interactions = self._generate_all_interactions(parent_features, data, target)
            
            # Select best interactions
            selected_interactions, rejected_interactions = self._select_best_interactions(all_interactions)
            
            execution_time = time.time() - start_time
            
            # Create result
            result = InteractionResult(
                selected_interactions=selected_interactions,
                rejected_interactions=rejected_interactions,
                parent_features=[w.name for w in parent_features],
                execution_time=execution_time,
                n_interactions_generated=len(all_interactions),
                n_interactions_selected=len(selected_interactions),
                matrix_ops_used=self.performance_metrics['matrix_ops_used'],
                vectorized_ops=self.performance_metrics['vectorized_ops']
            )
            
            tprint_success(f"✅ Interaction generation completed in {execution_time:.3f}s")
            tprint_success(f"📊 Generated {len(all_interactions)} interactions, selected {len(selected_interactions)}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._log_error(f"Interaction generation failed: {e}")
            self._log_error(f"Error details: {traceback.format_exc()}")

            # Return empty result
            return InteractionResult(
                selected_interactions=[],
                rejected_interactions=[],
                parent_features=[],
                execution_time=execution_time,
                n_interactions_generated=0,
                n_interactions_selected=0
            )
    
    def _filter_parent_features(self, selected_wrappers: List[FeatureGeneratorWrapper]) -> List[FeatureGeneratorWrapper]:
        """Filter parent features by utility requirements."""
        filtered = []
        
        for wrapper in selected_wrappers:
            # Check utility threshold
            if wrapper.phase2_utility is None or wrapper.phase2_utility < self.config.min_parent_utility:
                self._log_warning(
                    f"Parent feature {wrapper.name} excluded (utility={wrapper.phase2_utility})"
                )
                continue

            # Check if both parents required
            if self.config.require_both_parents:
                # For now, assume all features can be parents
                # In practice, you might have specific parent requirements
                pass

            filtered.append(wrapper)

        tprint_info(f"📊 Filtered to {len(filtered)} parent features (min utility: {self.config.min_parent_utility})")
        return filtered
    
    def _generate_all_interactions(self, parent_features: List[FeatureGeneratorWrapper], 
                                 data: pd.DataFrame, target: np.ndarray) -> List[InteractionFeature]:
        """Generate all possible interactions between parent features."""
        interactions = []
        
        # Generate parent feature values
        parent_values = {}
        for wrapper in parent_features:
            try:
                feature_values = self._generate_feature_values(wrapper, data)
                if feature_values is not None and len(feature_values) > 10:
                    parent_values[wrapper.name] = feature_values
                    self._log_debug(f"Cached parent values for {wrapper.name} (n={len(feature_values)})")
                else:
                    self._log_warning(f"Insufficient values for parent {wrapper.name}, skipping")
            except Exception as e:
                self._log_debug(f"Failed to generate values for {wrapper.name}: {e}")
                continue

        if len(parent_values) < 2:
            self._log_warning("Not enough parent features with valid values to generate interactions")
            return interactions
        
        # Generate all pairwise combinations
        parent_names = list(parent_values.keys())
        
        for i, (parent1, parent2) in enumerate(combinations(parent_names, 2)):
            try:
                # Check correlation between parents
                if self._check_parent_correlation(parent_values[parent1], parent_values[parent2]):
                    continue
                
                # Generate interactions for all types
                for interaction_type in self.config.interaction_types:
                    try:
                        interaction = self._create_interaction(
                            parent1, parent2, interaction_type,
                            parent_values[parent1], parent_values[parent2],
                            target
                        )
                        
                        if interaction:
                            interactions.append(interaction)
                            self.performance_metrics['interactions_generated'] += 1
                            
                    except Exception as e:
                        self._log_debug(
                            f"Failed to create {interaction_type} interaction between {parent1} and {parent2}: {e}"
                        )
                        continue

            except Exception as e:
                self._log_debug(f"Failed to process interaction between {parent1} and {parent2}: {e}")
                continue
        
        tprint_info(f"📊 Generated {len(interactions)} potential interactions")
        return interactions
    
    def _generate_feature_values(self, wrapper: FeatureGeneratorWrapper, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Generate feature values for a wrapper."""
        try:
            if hasattr(wrapper.generator, 'generate'):
                result = wrapper.generator.generate(data, lookback=20)  # Use default lookback

                if hasattr(result, 'data'):
                    return result.data.values
                elif isinstance(result, pd.Series):
                    return result.values
                elif isinstance(result, np.ndarray):
                    return result
                else:
                    self._log_warning(f"Unsupported return type from generator for {wrapper.name}; skipping")
                    return None
            else:
                self._log_warning(f"Wrapper {wrapper.name} lacks a generate method; skipping")
                return None

        except Exception as e:
            self._log_debug(f"Failed to generate values for {wrapper.name}: {e}")
            return None
    
    def _check_parent_correlation(self, parent1_values: np.ndarray, parent2_values: np.ndarray) -> bool:
        """Check if parents are too highly correlated."""
        try:
            # Align arrays to same length
            min_length = min(len(parent1_values), len(parent2_values))
            if min_length < 10:
                return True  # Too short, skip
            
            p1 = parent1_values[:min_length]
            p2 = parent2_values[:min_length]
            
            # Remove NaN values
            valid_mask = np.isfinite(p1) & np.isfinite(p2)
            if np.sum(valid_mask) < 10:
                return True  # Too few valid values, skip
            
            p1_clean = p1[valid_mask]
            p2_clean = p2[valid_mask]
            
            # Compute correlation
            correlation = np.corrcoef(p1_clean, p2_clean)[0, 1]

            if np.isnan(correlation):
                self._log_warning("Correlation computation produced NaN; skipping pair")
                return True  # Invalid correlation, skip

            correlation_abs = abs(correlation)
            if correlation_abs > self.config.max_correlation:
                self._log_warning(
                    f"Parents correlation {correlation:.4f} exceeds max {self.config.max_correlation}"
                )
                return True

            self._log_debug(f"Parents correlation {correlation:.4f} within acceptable bounds")
            return False

        except Exception as e:
            self._log_debug(f"Failed to check parent correlation: {e}")
            return True  # Skip on error
    
    def _create_interaction(self, parent1: str, parent2: str, interaction_type: str,
                          parent1_values: np.ndarray, parent2_values: np.ndarray,
                          target: np.ndarray) -> Optional[InteractionFeature]:
        """Create an interaction feature."""
        try:
            # Align arrays to same length
            min_length = min(len(parent1_values), len(parent2_values), len(target))
            if min_length < 10:
                return None
            
            p1 = parent1_values[:min_length]
            p2 = parent2_values[:min_length]
            t = target[:min_length]
            
            # Remove NaN values
            valid_mask = np.isfinite(p1) & np.isfinite(p2) & np.isfinite(t)
            if np.sum(valid_mask) < 10:
                return None
            
            p1_clean = p1[valid_mask]
            p2_clean = p2[valid_mask]
            t_clean = t[valid_mask]
            
            # Generate interaction values
            interaction_values = self._compute_interaction_values(p1_clean, p2_clean, interaction_type)
            
            if interaction_values is None or len(interaction_values) < 10:
                return None
            
            # Compute utility and stability
            utility = self._compute_interaction_utility(interaction_values, t_clean)
            stability = self._compute_interaction_stability(interaction_values, t_clean)
            
            # Create interaction feature
            interaction_name = f"{parent1}_{interaction_type}_{parent2}"
            
            interaction = InteractionFeature(
                name=interaction_name,
                parent1=parent1,
                parent2=parent2,
                interaction_type=interaction_type,
                utility=utility,
                stability=stability,
                cost=1.0  # Simple cost model
            )

            self.performance_metrics['interactions_evaluated'] += 1
            self._log_performance(
                f"Evaluated interaction {interaction_name} | utility={utility:.4f}, stability={stability:.4f}"
            )
            return interaction

        except Exception as e:
            self._log_debug(
                f"Failed to create interaction {interaction_type} between {parent1} and {parent2}: {e}"
            )
            return None
    
    def _compute_interaction_values(self, parent1_values: np.ndarray, parent2_values: np.ndarray, 
                                  interaction_type: str) -> Optional[np.ndarray]:
        """Compute interaction values based on type."""
        try:
            if interaction_type == "multiplication":
                return parent1_values * parent2_values
            elif interaction_type == "division":
                # Avoid division by zero
                return np.where(np.abs(parent2_values) > 1e-8, 
                              parent1_values / parent2_values, 
                              np.zeros_like(parent1_values))
            elif interaction_type == "addition":
                return parent1_values + parent2_values
            elif interaction_type == "subtraction":
                return parent1_values - parent2_values
            else:
                return None
                
        except Exception as e:
            self._log_debug(f"Failed to compute {interaction_type} interaction values: {e}")
            return None
    
    def _compute_interaction_utility(self, interaction_values: np.ndarray, target: np.ndarray) -> float:
        """Compute utility (IC) for interaction feature."""
        try:
            if len(interaction_values) < 10 or len(target) < 10:
                self._log_warning("Insufficient data to compute interaction utility")
                return 0.0

            # Compute correlation (IC)
            ic = np.corrcoef(interaction_values, target)[0, 1]

            if np.isnan(ic):
                self._log_warning("Interaction utility computation produced NaN; defaulting to 0.0")
                return 0.0

            ic_value = float(ic)
            self._log_performance(f"Computed interaction utility: {ic_value:.4f}")
            return ic_value

        except Exception as e:
            self._log_debug(f"Failed to compute interaction utility: {e}")
            return 0.0

    def _compute_interaction_stability(self, interaction_values: np.ndarray, target: np.ndarray) -> float:
        """Compute stability score for interaction feature."""
        try:
            if len(interaction_values) < 20 or len(target) < 20:
                self._log_warning("Insufficient data to compute interaction stability")
                return 0.0

            # Split data into thirds
            n = len(interaction_values)
            third = n // 3

            if third < 5:
                self._log_warning("Not enough samples per segment for stability computation")
                return 0.0
            
            # Oldest third
            old_values = interaction_values[:third]
            old_target = target[:third]
            
            # Newest third
            new_values = interaction_values[-third:]
            new_target = target[-third:]
            
            # Compute IC for each third
            if len(np.unique(old_values)) > 1 and len(np.unique(old_target)) > 1:
                old_ic = np.corrcoef(old_values, old_target)[0, 1]
            else:
                old_ic = 0.0
            
            if len(np.unique(new_values)) > 1 and len(np.unique(new_target)) > 1:
                new_ic = np.corrcoef(new_values, new_target)[0, 1]
            else:
                new_ic = 0.0
            
            # Stability score based on consistency
            if np.isnan(old_ic) or np.isnan(new_ic):
                self._log_warning("Stability computation produced NaN values; defaulting to 0.0")
                return 0.0

            # Higher stability if both ICs have same sign and similar magnitude
            if old_ic * new_ic > 0:  # Same sign
                stability = 1.0 - abs(old_ic - new_ic) / (abs(old_ic) + abs(new_ic) + 1e-6)
            else:  # Different signs
                stability = 0.0
                self._log_warning("Stability penalized due to inconsistent IC signs")
            
            stability_score = max(0.0, min(1.0, stability))
            self._log_performance(f"Computed interaction stability: {stability_score:.4f}")
            return stability_score

        except Exception as e:
            self._log_debug(f"Failed to compute interaction stability: {e}")
            return 0.0
    
    def _select_best_interactions(self, interactions: List[InteractionFeature]) -> Tuple[List[InteractionFeature], List[InteractionFeature]]:
        """Select the best interactions based on utility and constraints."""
        if not interactions:
            self._log_warning("No interactions available for selection")
            return [], []
        
        # Filter by utility threshold
        if self.config.evaluate_interactions:
            filtered_interactions = [i for i in interactions if i.utility >= self.config.interaction_utility_threshold]
        else:
            filtered_interactions = interactions
        
        # Sort by utility (descending)
        sorted_interactions = sorted(filtered_interactions, key=lambda x: x.utility, reverse=True)
        
        # Select up to max_interactions
        selected = sorted_interactions[:self.config.max_interactions]
        rejected = sorted_interactions[self.config.max_interactions:]
        
        tprint_info(f"📊 Selected {len(selected)} interactions from {len(interactions)} candidates")

        if rejected:
            self._log_warning(f"Rejected {len(rejected)} interactions due to budget constraints")

        return selected, rejected
