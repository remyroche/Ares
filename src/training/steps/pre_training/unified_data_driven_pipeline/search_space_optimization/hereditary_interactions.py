"""
Hereditary Interaction Generation

Implements hereditary interactions where A×B is only allowed if both A and B
survive pre-selection, preventing search space explosion.

Key Features:
- Pre-selection constraint enforcement
- Interaction type validation
- Computational efficiency optimization
- Memory usage monitoring
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
import logging
import time
from enum import Enum
from itertools import combinations
import warnings

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

logger = logging.getLogger(__name__)


class InteractionType(Enum):
    """Types of interactions that can be generated."""
    MULTIPLICATION = "multiplication"
    DIVISION = "division"
    ADDITION = "addition"
    SUBTRACTION = "subtraction"
    CORRELATION = "correlation"
    RATIO = "ratio"
    DIFFERENCE = "difference"
    PRODUCT = "product"
    CUSTOM = "custom"


@dataclass
class InteractionFeature:
    """Represents a generated interaction feature."""
    name: str
    feature_a: str
    feature_b: str
    interaction_type: InteractionType
    values: pd.Series
    correlation_with_a: float = 0.0
    correlation_with_b: float = 0.0
    information_content: float = 0.0
    stability_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HereditaryInteractionConfig:
    """Configuration for hereditary interaction generation."""
    
    # Pre-selection constraints
    require_pre_selection: bool = True
    pre_selected_features: Optional[Set[str]] = None
    
    # Interaction generation
    interaction_types: List[InteractionType] = field(default_factory=lambda: [
        InteractionType.MULTIPLICATION,
        InteractionType.DIVISION,
        InteractionType.RATIO,
        InteractionType.DIFFERENCE
    ])
    
    # Quality constraints
    min_correlation_threshold: float = 0.1
    max_correlation_threshold: float = 0.95
    min_information_content: float = 0.01
    min_stability_score: float = 0.5
    
    # Computational constraints
    max_interactions: int = 1000
    max_interactions_per_feature: int = 10
    enable_parallel_processing: bool = True
    max_workers: int = 4
    
    # Memory constraints
    memory_limit_mb: float = 1000.0
    chunk_size: int = 100
    
    # Validation
    validate_interactions: bool = True
    check_collinearity: bool = True
    max_collinearity_threshold: float = 0.99


@dataclass
class HereditaryInteractionResult:
    """Result from hereditary interaction generation."""
    
    # Generated interactions
    interactions: List[InteractionFeature]
    interaction_count: int
    
    # Pre-selection statistics
    pre_selected_features: Set[str]
    pre_selection_count: int
    total_candidates: int
    
    # Quality metrics
    average_correlation: float
    average_information_content: float
    average_stability_score: float
    
    # Performance metrics
    generation_time: float
    memory_usage_mb: float
    parallel_operations: int
    
    # Validation results
    valid_interactions: int
    invalid_interactions: int
    collinearity_violations: int
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class HereditaryInteractionGenerator:
    """
    Generates interactions only for pre-selected features.
    
    This class implements hereditary interactions where A×B is only allowed
    if both A and B survive pre-selection, preventing search space explosion.
    """
    
    def __init__(self, config: Optional[HereditaryInteractionConfig] = None):
        """
        Initialize the hereditary interaction generator.
        
        Args:
            config: Configuration for interaction generation
        """
        self.config = config or HereditaryInteractionConfig()
        self.logger = logger
        
        # Initialize pre-selected features
        if self.config.pre_selected_features is None:
            self.config.pre_selected_features = set()
        
        tprint_info("🧬 Hereditary Interaction Generator initialized")
        tprint_debug(f"📊 Pre-selected features: {len(self.config.pre_selected_features)}")
        tprint_debug(f"📊 Interaction types: {len(self.config.interaction_types)}")
        tprint_debug(f"📊 Max interactions: {self.config.max_interactions}")
    
    def generate_interactions(self, 
                            data: pd.DataFrame,
                            pre_selected_features: Optional[Set[str]] = None) -> HereditaryInteractionResult:
        """
        Generate interactions for pre-selected features.
        
        Args:
            data: Input data with features
            pre_selected_features: Features that survived pre-selection
            
        Returns:
            HereditaryInteractionResult with generated interactions
        """
        start_time = time.time()
        
        tprint_info("🧬 Generating hereditary interactions...")
        tprint_debug(f"📊 Data shape: {data.shape}")
        tprint_debug(f"📊 Pre-selected features: {len(pre_selected_features) if pre_selected_features else 0}")
        
        try:
            # Update pre-selected features
            if pre_selected_features is not None:
                self.config.pre_selected_features = pre_selected_features
            
            # Validate pre-selection
            if not self.config.pre_selected_features:
                tprint_warning("⚠️ No pre-selected features provided")
                return self._create_empty_result(start_time, "No pre-selected features")
            
            # Filter pre-selected features that exist in data
            available_features = set(data.columns)
            valid_pre_selected = self.config.pre_selected_features.intersection(available_features)
            
            if not valid_pre_selected:
                tprint_warning("⚠️ No valid pre-selected features found in data")
                return self._create_empty_result(start_time, "No valid pre-selected features")
            
            tprint_success(f"✅ Using {len(valid_pre_selected)} valid pre-selected features")
            
            # Generate interaction candidates
            tprint_debug("Generating interaction candidates...")
            interaction_candidates = self._generate_interaction_candidates(valid_pre_selected)
            
            if not interaction_candidates:
                tprint_warning("⚠️ No interaction candidates generated")
                return self._create_empty_result(start_time, "No interaction candidates")
            
            tprint_success(f"✅ Generated {len(interaction_candidates)} interaction candidates")
            
            # Generate interactions
            tprint_debug("Generating interactions...")
            interactions = self._generate_interactions_from_candidates(
                data, interaction_candidates
            )
            
            if not interactions:
                tprint_warning("⚠️ No valid interactions generated")
                return self._create_empty_result(start_time, "No valid interactions")
            
            tprint_success(f"✅ Generated {len(interactions)} valid interactions")
            
            # Validate interactions
            if self.config.validate_interactions:
                tprint_debug("Validating interactions...")
                interactions = self._validate_interactions(interactions, data)
                tprint_success(f"✅ {len(interactions)} interactions passed validation")
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(interactions)
            
            # Calculate performance metrics
            generation_time = time.time() - start_time
            memory_usage = self._estimate_memory_usage(data, interactions)
            
            result = HereditaryInteractionResult(
                interactions=interactions,
                interaction_count=len(interactions),
                pre_selected_features=valid_pre_selected,
                pre_selection_count=len(valid_pre_selected),
                total_candidates=len(interaction_candidates),
                average_correlation=quality_metrics['average_correlation'],
                average_information_content=quality_metrics['average_information_content'],
                average_stability_score=quality_metrics['average_stability_score'],
                generation_time=generation_time,
                memory_usage_mb=memory_usage,
                parallel_operations=0,  # TODO: Implement parallel processing
                valid_interactions=len(interactions),
                invalid_interactions=len(interaction_candidates) - len(interactions),
                collinearity_violations=quality_metrics['collinearity_violations'],
                metadata={
                    'config': self.config.__dict__,
                    'interaction_types_used': [t.value for t in self.config.interaction_types]
                }
            )
            
            tprint_success(f"✅ Hereditary interaction generation completed in {generation_time:.3f}s")
            tprint_info(f"📊 Generated interactions: {len(interactions)}")
            tprint_info(f"📊 Average correlation: {quality_metrics['average_correlation']:.3f}")
            tprint_info(f"📊 Memory usage: {memory_usage:.1f}MB")
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Hereditary interaction generation failed: {e}")
            return self._create_empty_result(start_time, str(e))
    
    def _generate_interaction_candidates(self, 
                                       pre_selected_features: Set[str]) -> List[Tuple[str, str, InteractionType]]:
        """Generate interaction candidates from pre-selected features."""
        candidates = []
        
        try:
            # Generate all pairwise combinations
            feature_pairs = list(combinations(pre_selected_features, 2))
            
            tprint_debug(f"📊 Generated {len(feature_pairs)} feature pairs")
            
            # Apply constraints
            max_candidates = self.config.max_interactions
            max_per_feature = self.config.max_interactions_per_feature
            
            # Count interactions per feature
            feature_counts = {feature: 0 for feature in pre_selected_features}
            
            for feature_a, feature_b in feature_pairs:
                # Check per-feature limits
                if (feature_counts[feature_a] >= max_per_feature or 
                    feature_counts[feature_b] >= max_per_feature):
                    continue
                
                # Generate candidates for each interaction type
                for interaction_type in self.config.interaction_types:
                    candidates.append((feature_a, feature_b, interaction_type))
                    feature_counts[feature_a] += 1
                    feature_counts[feature_b] += 1
                    
                    # Check total limit
                    if len(candidates) >= max_candidates:
                        break
                
                if len(candidates) >= max_candidates:
                    break
            
            tprint_debug(f"📊 Generated {len(candidates)} interaction candidates")
            
        except Exception as e:
            tprint_error(f"❌ Failed to generate interaction candidates: {e}")
            return []
        
        return candidates
    
    def _generate_interactions_from_candidates(self, 
                                             data: pd.DataFrame,
                                             candidates: List[Tuple[str, str, InteractionType]]) -> List[InteractionFeature]:
        """Generate interactions from candidates."""
        interactions = []
        
        try:
            for i, (feature_a, feature_b, interaction_type) in enumerate(candidates):
                if i % 100 == 0:
                    tprint_debug(f"Processing candidate {i+1}/{len(candidates)}")
                
                try:
                    # Generate interaction
                    interaction = self._create_interaction(
                        data, feature_a, feature_b, interaction_type
                    )
                    
                    if interaction is not None:
                        interactions.append(interaction)
                        
                except Exception as e:
                    tprint_debug(f"⚠️ Failed to create interaction {feature_a}×{feature_b}: {e}")
                    continue
                
                # Check memory limit
                if self._check_memory_limit(data, interactions):
                    tprint_warning("⚠️ Memory limit reached, stopping generation")
                    break
            
        except Exception as e:
            tprint_error(f"❌ Failed to generate interactions: {e}")
            return []
        
        return interactions
    
    def _create_interaction(self, 
                          data: pd.DataFrame,
                          feature_a: str,
                          feature_b: str,
                          interaction_type: InteractionType) -> Optional[InteractionFeature]:
        """Create a single interaction feature."""
        try:
            # Get feature data
            series_a = data[feature_a].dropna()
            series_b = data[feature_b].dropna()
            
            # Align series
            common_index = series_a.index.intersection(series_b.index)
            if len(common_index) < 10:  # Need minimum data points
                return None
            
            series_a = series_a.loc[common_index]
            series_b = series_b.loc[common_index]
            
            # Generate interaction based on type
            if interaction_type == InteractionType.MULTIPLICATION:
                interaction_values = series_a * series_b
            elif interaction_type == InteractionType.DIVISION:
                # Avoid division by zero
                series_b_safe = series_b.replace(0, np.nan)
                interaction_values = series_a / series_b_safe
            elif interaction_type == InteractionType.ADDITION:
                interaction_values = series_a + series_b
            elif interaction_type == InteractionType.SUBTRACTION:
                interaction_values = series_a - series_b
            elif interaction_type == InteractionType.RATIO:
                # Ratio of A to B
                series_b_safe = series_b.replace(0, np.nan)
                interaction_values = series_a / series_b_safe
            elif interaction_type == InteractionType.DIFFERENCE:
                # Absolute difference
                interaction_values = (series_a - series_b).abs()
            elif interaction_type == InteractionType.CORRELATION:
                # Rolling correlation (simplified)
                window = min(20, len(series_a) // 4)
                interaction_values = series_a.rolling(window).corr(series_b)
            else:
                return None
            
            # Check for valid values
            if interaction_values.isna().all() or interaction_values.nunique() <= 1:
                return None
            
            # Calculate quality metrics
            correlation_with_a = interaction_values.corr(series_a)
            correlation_with_b = interaction_values.corr(series_b)
            information_content = self._calculate_information_content(interaction_values)
            stability_score = self._calculate_stability_score(interaction_values)
            
            # Apply quality filters
            if (abs(correlation_with_a) < self.config.min_correlation_threshold and
                abs(correlation_with_b) < self.config.min_correlation_threshold):
                return None
            
            if abs(correlation_with_a) > self.config.max_correlation_threshold or \
               abs(correlation_with_b) > self.config.max_correlation_threshold:
                return None
            
            if information_content < self.config.min_information_content:
                return None
            
            if stability_score < self.config.min_stability_score:
                return None
            
            # Create interaction feature
            interaction_name = f"{feature_a}_{interaction_type.value}_{feature_b}"
            
            return InteractionFeature(
                name=interaction_name,
                feature_a=feature_a,
                feature_b=feature_b,
                interaction_type=interaction_type,
                values=interaction_values,
                correlation_with_a=correlation_with_a if not pd.isna(correlation_with_a) else 0.0,
                correlation_with_b=correlation_with_b if not pd.isna(correlation_with_b) else 0.0,
                information_content=information_content,
                stability_score=stability_score,
                metadata={
                    'data_points': len(interaction_values),
                    'non_null_ratio': interaction_values.notna().mean(),
                    'variance': interaction_values.var()
                }
            )
            
        except Exception as e:
            tprint_debug(f"⚠️ Failed to create interaction {feature_a}×{feature_b}: {e}")
            return None
    
    def _validate_interactions(self, 
                             interactions: List[InteractionFeature],
                             data: pd.DataFrame) -> List[InteractionFeature]:
        """Validate interactions for quality and collinearity."""
        valid_interactions = []
        
        try:
            for interaction in interactions:
                # Check basic validity
                if not self._is_interaction_valid(interaction):
                    continue
                
                # Check collinearity if enabled
                if self.config.check_collinearity:
                    if self._check_collinearity_violation(interaction, valid_interactions, data):
                        continue
                
                valid_interactions.append(interaction)
            
        except Exception as e:
            tprint_error(f"❌ Interaction validation failed: {e}")
            return interactions  # Return original if validation fails
        
        return valid_interactions
    
    def _is_interaction_valid(self, interaction: InteractionFeature) -> bool:
        """Check if an interaction is valid."""
        try:
            # Check for finite values
            if not interaction.values.notna().any():
                return False
            
            # Check minimum data points
            if len(interaction.values.dropna()) < 10:
                return False
            
            # Check variance
            if interaction.values.var() < 1e-10:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _check_collinearity_violation(self, 
                                    interaction: InteractionFeature,
                                    existing_interactions: List[InteractionFeature],
                                    data: pd.DataFrame) -> bool:
        """Check if interaction violates collinearity constraints."""
        try:
            if not existing_interactions:
                return False
            
            # Calculate correlation with existing interactions
            for existing in existing_interactions:
                try:
                    # Align series
                    common_index = interaction.values.index.intersection(existing.values.index)
                    if len(common_index) < 10:
                        continue
                    
                    corr = interaction.values.loc[common_index].corr(existing.values.loc[common_index])
                    
                    if not pd.isna(corr) and abs(corr) > self.config.max_collinearity_threshold:
                        return True
                        
                except Exception:
                    continue
            
            return False
            
        except Exception:
            return False
    
    def _calculate_information_content(self, series: pd.Series) -> float:
        """Calculate information content of a series."""
        try:
            # Simple entropy-based information content
            value_counts = series.value_counts()
            probabilities = value_counts / len(series)
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            
            # Normalize to 0-1 scale
            max_entropy = np.log2(len(value_counts))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            return min(max(normalized_entropy, 0), 1)
            
        except Exception:
            return 0.0
    
    def _calculate_stability_score(self, series: pd.Series) -> float:
        """Calculate stability score of a series."""
        try:
            # Simple stability based on rolling variance
            window = min(20, len(series) // 4)
            if window < 5:
                return 1.0
            
            rolling_var = series.rolling(window).var()
            stability = 1.0 - (rolling_var.std() / rolling_var.mean()) if rolling_var.mean() > 0 else 0.0
            
            return min(max(stability, 0), 1)
            
        except Exception:
            return 0.0
    
    def _calculate_quality_metrics(self, interactions: List[InteractionFeature]) -> Dict[str, float]:
        """Calculate quality metrics for interactions."""
        if not interactions:
            return {
                'average_correlation': 0.0,
                'average_information_content': 0.0,
                'average_stability_score': 0.0,
                'collinearity_violations': 0
            }
        
        try:
            correlations = []
            information_contents = []
            stability_scores = []
            
            for interaction in interactions:
                correlations.append(abs(interaction.correlation_with_a))
                correlations.append(abs(interaction.correlation_with_b))
                information_contents.append(interaction.information_content)
                stability_scores.append(interaction.stability_score)
            
            return {
                'average_correlation': np.mean(correlations) if correlations else 0.0,
                'average_information_content': np.mean(information_contents) if information_contents else 0.0,
                'average_stability_score': np.mean(stability_scores) if stability_scores else 0.0,
                'collinearity_violations': 0  # TODO: Count actual violations
            }
            
        except Exception:
            return {
                'average_correlation': 0.0,
                'average_information_content': 0.0,
                'average_stability_score': 0.0,
                'collinearity_violations': 0
            }
    
    def _check_memory_limit(self, data: pd.DataFrame, interactions: List[InteractionFeature]) -> bool:
        """Check if memory limit is exceeded."""
        try:
            # Estimate memory usage
            data_memory = data.memory_usage(deep=True).sum() / 1024 / 1024
            interaction_memory = sum(len(interaction.values) * 8 / 1024 / 1024 for interaction in interactions)
            
            total_memory = data_memory + interaction_memory
            
            return total_memory > self.config.memory_limit_mb
            
        except Exception:
            return False
    
    def _estimate_memory_usage(self, data: pd.DataFrame, interactions: List[InteractionFeature]) -> float:
        """Estimate memory usage in MB."""
        try:
            data_memory = data.memory_usage(deep=True).sum() / 1024 / 1024
            interaction_memory = sum(len(interaction.values) * 8 / 1024 / 1024 for interaction in interactions)
            
            return data_memory + interaction_memory
            
        except Exception:
            return 0.0
    
    def _create_empty_result(self, start_time: float, error_message: str) -> HereditaryInteractionResult:
        """Create empty result for failed generation."""
        return HereditaryInteractionResult(
            interactions=[],
            interaction_count=0,
            pre_selected_features=set(),
            pre_selection_count=0,
            total_candidates=0,
            average_correlation=0.0,
            average_information_content=0.0,
            average_stability_score=0.0,
            generation_time=time.time() - start_time,
            memory_usage_mb=0.0,
            parallel_operations=0,
            valid_interactions=0,
            invalid_interactions=0,
            collinearity_violations=0,
            metadata={'error': True, 'error_message': error_message}
        )


# Convenience functions
def generate_hereditary_interactions(data: pd.DataFrame,
                                   pre_selected_features: Set[str],
                                   config: Optional[HereditaryInteractionConfig] = None) -> HereditaryInteractionResult:
    """
    Convenience function to generate hereditary interactions.
    
    Args:
        data: Input data with features
        pre_selected_features: Features that survived pre-selection
        config: Configuration for interaction generation
        
    Returns:
        HereditaryInteractionResult with generated interactions
    """
    generator = HereditaryInteractionGenerator(config)
    return generator.generate_interactions(data, pre_selected_features)


# Export main classes and functions
__all__ = [
    'HereditaryInteractionGenerator',
    'HereditaryInteractionConfig',
    'HereditaryInteractionResult',
    'InteractionFeature',
    'InteractionType',
    'generate_hereditary_interactions'
]