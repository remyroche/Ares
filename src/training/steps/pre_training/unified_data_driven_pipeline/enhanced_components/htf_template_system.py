"""
HTF Template System Component for UnifiedDataDrivenPipeline

This module provides the complete HTF-aware interaction template system
integrated from HTFInteractionTemplates with VectorBT optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import logging
import time
import warnings
from collections import defaultdict
from itertools import combinations, product

# VectorBT imports for HTF template system
try:
    import vectorbt as vbt
    from vectorbt.generic import (
        rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, 
        rolling_sum, rolling_apply, rolling_corr, rolling_cov,
        scale, rank, zscore, winsorize, clip, quantile
    )
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available for HTF template system")

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


@dataclass
class InteractionTemplate:
    """Template for generating interactions."""
    name: str
    template_type: str  # 'core', 'htf_aware'
    formula: str
    required_features: List[str]
    optional_features: List[str]
    max_instances: int
    priority: int
    metadata: Dict[str, Any]


@dataclass
class GeneratedInteraction:
    """Generated interaction feature."""
    name: str
    formula: str
    parent_features: List[str]
    interaction_type: str
    feature_series: pd.Series
    utility_score: float
    metadata: Dict[str, Any]


@dataclass
class HTFTemplateConfig:
    """Configuration for HTF template system."""
    enable_vectorbt: bool = True
    enable_parallel: bool = True
    memory_efficient: bool = True
    max_workers: int = 4
    max_interactions: int = 100
    utility_threshold: float = 0.1
    correlation_threshold: float = 0.95
    enable_htf_aware: bool = True
    enable_core_templates: bool = True
    budget_allocation: Dict[str, int] = None


class CoreInteractionTemplates:
    """Core 15 interaction templates (theory-first)."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.templates = self._create_core_templates()
        tprint_info(f"Initialized core interaction templates | count={len(self.templates)}")
    
    def _create_core_templates(self) -> List[InteractionTemplate]:
        """Create core interaction templates."""
        templates = [
            # Price-Volatility interactions
            InteractionTemplate(
                name="price_vol_interaction",
                template_type="core",
                formula="price_feature * volatility_feature",
                required_features=["price_feature", "volatility_feature"],
                optional_features=[],
                max_instances=5,
                priority=1,
                metadata={"description": "Price-volatility interaction"}
            ),
            
            # Momentum-Mean Reversion interactions
            InteractionTemplate(
                name="momentum_meanrev_interaction",
                template_type="core",
                formula="momentum_feature * mean_reversion_feature",
                required_features=["momentum_feature", "mean_reversion_feature"],
                optional_features=[],
                max_instances=5,
                priority=1,
                metadata={"description": "Momentum-mean reversion interaction"}
            ),
            
            # Liquidity-Price interactions
            InteractionTemplate(
                name="liquidity_price_interaction",
                template_type="core",
                formula="liquidity_feature * price_feature",
                required_features=["liquidity_feature", "price_feature"],
                optional_features=[],
                max_instances=5,
                priority=1,
                metadata={"description": "Liquidity-price interaction"}
            ),
            
            # Volatility-Volume interactions
            InteractionTemplate(
                name="vol_volume_interaction",
                template_type="core",
                formula="volatility_feature * volume_feature",
                required_features=["volatility_feature", "volume_feature"],
                optional_features=[],
                max_instances=5,
                priority=1,
                metadata={"description": "Volatility-volume interaction"}
            ),
            
            # Time-of-day interactions
            InteractionTemplate(
                name="tod_interaction",
                template_type="core",
                formula="feature * tod_indicator",
                required_features=["feature", "tod_indicator"],
                optional_features=[],
                max_instances=3,
                priority=2,
                metadata={"description": "Time-of-day interaction"}
            ),
            
            # Cross-sectional interactions
            InteractionTemplate(
                name="cross_sectional_interaction",
                template_type="core",
                formula="feature - market_feature",
                required_features=["feature", "market_feature"],
                optional_features=[],
                max_instances=3,
                priority=2,
                metadata={"description": "Cross-sectional interaction"}
            ),
            
            # Regime interactions
            InteractionTemplate(
                name="regime_interaction",
                template_type="core",
                formula="feature * regime_indicator",
                required_features=["feature", "regime_indicator"],
                optional_features=[],
                max_instances=3,
                priority=2,
                metadata={"description": "Regime interaction"}
            ),
            
            # Lag interactions
            InteractionTemplate(
                name="lag_interaction",
                template_type="core",
                formula="feature * feature_lag",
                required_features=["feature", "feature_lag"],
                optional_features=[],
                max_instances=5,
                priority=3,
                metadata={"description": "Lag interaction"}
            ),
            
            # Polynomial interactions
            InteractionTemplate(
                name="polynomial_interaction",
                template_type="core",
                formula="feature ** 2",
                required_features=["feature"],
                optional_features=[],
                max_instances=3,
                priority=3,
                metadata={"description": "Polynomial interaction"}
            ),
            
            # Ratio interactions
            InteractionTemplate(
                name="ratio_interaction",
                template_type="core",
                formula="feature1 / (feature2 + epsilon)",
                required_features=["feature1", "feature2"],
                optional_features=[],
                max_instances=5,
                priority=3,
                metadata={"description": "Ratio interaction"}
            ),
            
            # Difference interactions
            InteractionTemplate(
                name="difference_interaction",
                template_type="core",
                formula="feature1 - feature2",
                required_features=["feature1", "feature2"],
                optional_features=[],
                max_instances=5,
                priority=3,
                metadata={"description": "Difference interaction"}
            ),
            
            # Product interactions
            InteractionTemplate(
                name="product_interaction",
                template_type="core",
                formula="feature1 * feature2",
                required_features=["feature1", "feature2"],
                optional_features=[],
                max_instances=5,
                priority=3,
                metadata={"description": "Product interaction"}
            ),
            
            # Conditional interactions
            InteractionTemplate(
                name="conditional_interaction",
                template_type="core",
                formula="feature * (condition > threshold)",
                required_features=["feature", "condition"],
                optional_features=["threshold"],
                max_instances=3,
                priority=3,
                metadata={"description": "Conditional interaction"}
            ),
            
            # Rolling interactions
            InteractionTemplate(
                name="rolling_interaction",
                template_type="core",
                formula="feature.rolling(window).mean()",
                required_features=["feature"],
                optional_features=["window"],
                max_instances=3,
                priority=3,
                metadata={"description": "Rolling interaction"}
            ),
            
            # Z-score interactions
            InteractionTemplate(
                name="zscore_interaction",
                template_type="core",
                formula="(feature - feature.mean()) / feature.std()",
                required_features=["feature"],
                optional_features=[],
                max_instances=3,
                priority=3,
                metadata={"description": "Z-score interaction"}
            ),
            
            # Log interactions
            InteractionTemplate(
                name="log_interaction",
                template_type="core",
                formula="np.log(feature + epsilon)",
                required_features=["feature"],
                optional_features=[],
                max_instances=3,
                priority=2,
                metadata={"description": "Log transformation interaction"}
            ),
            
            # Log ratio interactions
            InteractionTemplate(
                name="log_ratio_interaction",
                template_type="core",
                formula="np.log(feature1 + epsilon) - np.log(feature2 + epsilon)",
                required_features=["feature1", "feature2"],
                optional_features=[],
                max_instances=4,
                priority=2,
                metadata={"description": "Log ratio interaction (log(f1) - log(f2))"}
            ),
            
            # Log product interactions
            InteractionTemplate(
                name="log_product_interaction",
                template_type="core",
                formula="np.log(feature1 + epsilon) + np.log(feature2 + epsilon)",
                required_features=["feature1", "feature2"],
                optional_features=[],
                max_instances=4,
                priority=2,
                metadata={"description": "Log product interaction (log(f1) + log(f2))"}
            ),
            
            # Log volatility interactions
            InteractionTemplate(
                name="log_vol_interaction",
                template_type="core",
                formula="np.log(volatility_feature + epsilon) * price_feature",
                required_features=["volatility_feature", "price_feature"],
                optional_features=[],
                max_instances=3,
                priority=1,
                metadata={"description": "Log volatility × price interaction"}
            ),
            
            # Log momentum interactions
            InteractionTemplate(
                name="log_momentum_interaction",
                template_type="core",
                formula="np.log(momentum_feature + epsilon) * volume_feature",
                required_features=["momentum_feature", "volume_feature"],
                optional_features=[],
                max_instances=3,
                priority=2,
                metadata={"description": "Log momentum × volume interaction"}
            )
        ]
        
        tprint_debug(f"Constructed core template catalogue | total={len(templates)}")
        return templates


class HTFAwareTemplates:
    """HTF-aware interaction templates."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.templates = self._create_htf_aware_templates()
        tprint_info(f"Initialized HTF-aware templates | count={len(self.templates)}")
    
    def _create_htf_aware_templates(self) -> List[InteractionTemplate]:
        """Create HTF-aware interaction templates."""
        templates = [
            # HTF trend × base-TF liquidity
            InteractionTemplate(
                name="htf_trend_liquidity_interaction",
                template_type="htf_aware",
                formula="htf_trend_feature * base_liquidity_feature",
                required_features=["htf_trend_feature", "base_liquidity_feature"],
                optional_features=[],
                max_instances=3,
                priority=1,
                metadata={"description": "HTF trend × base liquidity interaction"}
            ),
            
            # HTF vol × base signal strength
            InteractionTemplate(
                name="htf_vol_signal_interaction",
                template_type="htf_aware",
                formula="htf_volatility_feature * base_signal_feature",
                required_features=["htf_volatility_feature", "base_signal_feature"],
                optional_features=[],
                max_instances=3,
                priority=1,
                metadata={"description": "HTF volatility × base signal interaction"}
            ),
            
            # HTF momentum conflict
            InteractionTemplate(
                name="htf_momentum_conflict_interaction",
                template_type="htf_aware",
                formula="htf_momentum_feature * (-base_momentum_feature)",
                required_features=["htf_momentum_feature", "base_momentum_feature"],
                optional_features=[],
                max_instances=3,
                priority=1,
                metadata={"description": "HTF momentum conflict interaction"}
            ),
            
            # HTF regime × base feature
            InteractionTemplate(
                name="htf_regime_base_interaction",
                template_type="htf_aware",
                formula="htf_regime_feature * base_feature",
                required_features=["htf_regime_feature", "base_feature"],
                optional_features=[],
                max_instances=3,
                priority=2,
                metadata={"description": "HTF regime × base feature interaction"}
            ),
            
            # HTF anchor × base deviation
            InteractionTemplate(
                name="htf_anchor_deviation_interaction",
                template_type="htf_aware",
                formula="htf_anchor_feature * base_deviation_feature",
                required_features=["htf_anchor_feature", "base_deviation_feature"],
                optional_features=[],
                max_instances=3,
                priority=2,
                metadata={"description": "HTF anchor × base deviation interaction"}
            ),
            
            # HTF log trend × base feature
            InteractionTemplate(
                name="htf_log_trend_interaction",
                template_type="htf_aware",
                formula="np.log(htf_trend_feature + epsilon) * base_feature",
                required_features=["htf_trend_feature", "base_feature"],
                optional_features=[],
                max_instances=3,
                priority=1,
                metadata={"description": "HTF log trend × base feature interaction"}
            ),
            
            # HTF log volatility × base signal
            InteractionTemplate(
                name="htf_log_vol_signal_interaction",
                template_type="htf_aware",
                formula="np.log(htf_volatility_feature + epsilon) * base_signal_feature",
                required_features=["htf_volatility_feature", "base_signal_feature"],
                optional_features=[],
                max_instances=3,
                priority=1,
                metadata={"description": "HTF log volatility × base signal interaction"}
            ),
            
            # HTF log momentum × base momentum
            InteractionTemplate(
                name="htf_log_momentum_interaction",
                template_type="htf_aware",
                formula="np.log(htf_momentum_feature + epsilon) * np.log(base_momentum_feature + epsilon)",
                required_features=["htf_momentum_feature", "base_momentum_feature"],
                optional_features=[],
                max_instances=3,
                priority=2,
                metadata={"description": "HTF log momentum × base log momentum interaction"}
            ),
            
            # HTF log regime × base feature
            InteractionTemplate(
                name="htf_log_regime_interaction",
                template_type="htf_aware",
                formula="np.log(htf_regime_feature + epsilon) * base_feature",
                required_features=["htf_regime_feature", "base_feature"],
                optional_features=[],
                max_instances=3,
                priority=2,
                metadata={"description": "HTF log regime × base feature interaction"}
            )
        ]
        
        tprint_debug(f"Constructed HTF-aware template set | total={len(templates)}")
        return templates


class HTFInteractionGenerator:
    """Generates interactions from templates using VectorBT optimization."""
    
    def __init__(self, config: Optional[HTFTemplateConfig] = None):
        self.config = config or HTFTemplateConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize template systems
        self.core_templates = CoreInteractionTemplates() if self.config.enable_core_templates else None
        self.htf_aware_templates = HTFAwareTemplates() if self.config.enable_htf_aware else None
        
        # Performance tracking
        self.performance_stats = {
            'total_generations': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'total_execution_time': 0.0,
            'core_interactions_generated': 0,
            'htf_interactions_generated': 0,
            'vectorbt_operations': 0
        }
        
        tprint_info("⚡ HTF Interaction Generator initialized")
        tprint_debug(f"📊 Configuration: {self.config}")
    
    def generate_interactions(self, materialized_htfs: Dict[str, Any], 
                            base_features: Union[pd.DataFrame, Dict[str, pd.Series], None],
                            targets: Optional[pd.Series] = None) -> List[GeneratedInteraction]:
        """
        Generate interactions from templates using VectorBT optimization.
        
        Args:
            materialized_htfs: Materialized HTF features
            base_features: Base features
            targets: Target variables
            
        Returns:
            List of generated interactions
        """
        tprint_info("⚡ Starting HTF interaction generation")
        
        start_time = time.time()
        
        try:
            # Validate inputs
            if not self._validate_inputs(materialized_htfs, base_features):
                return self._create_empty_result(start_time, "Invalid inputs")
            
            # Normalize base features
            normalized_base_features = self._normalize_base_features(base_features)
            
            # Determine budget allocation
            budget_allocation = self._determine_budget_allocation(materialized_htfs)
            
            # Generate core interactions
            core_interactions = []
            if self.core_templates and budget_allocation['core'] > 0:
                tprint_debug("Generating core interactions")
                core_interactions = self._generate_core_interactions_vectorbt(
                    normalized_base_features, targets, budget_allocation['core']
                )
            
            # Generate HTF-aware interactions
            htf_interactions = []
            if self.htf_aware_templates and budget_allocation['htf_aware'] > 0:
                tprint_debug("Generating HTF-aware interactions")
                htf_interactions = self._generate_htf_interactions_vectorbt(
                    materialized_htfs, normalized_base_features, targets, budget_allocation['htf_aware']
                )
            
            # Combine all interactions
            all_interactions = core_interactions + htf_interactions
            
            # Apply VectorBT-based feature selection
            selected_interactions = self._apply_vectorbt_feature_selection(all_interactions, targets)
            
            execution_time = time.time() - start_time
            
            # Update performance stats
            self.performance_stats.update({
                'total_generations': 1,
                'successful_generations': 1,
                'total_execution_time': execution_time,
                'core_interactions_generated': len(core_interactions),
                'htf_interactions_generated': len(htf_interactions)
            })
            
            tprint_success(f"✅ HTF interaction generation completed in {execution_time:.3f}s")
            tprint_info(f"🏆 Generated {len(selected_interactions)} interactions")
            
            return selected_interactions
            
        except Exception as e:
            tprint_error(f"❌ HTF interaction generation failed: {e}")
            return self._create_empty_result(start_time, str(e))
    
    def _validate_inputs(self, materialized_htfs: Dict[str, Any], 
                        base_features: Union[pd.DataFrame, Dict[str, pd.Series], None]) -> bool:
        """Validate input data and parameters."""
        try:
            if not materialized_htfs:
                tprint_error("No materialized HTF features provided")
                return False
            
            if base_features is None:
                tprint_error("No base features provided")
                return False
            
            return True
            
        except Exception as e:
            tprint_error(f"Input validation failed: {e}")
            return False
    
    def _normalize_base_features(self, base_features: Union[pd.DataFrame, Dict[str, pd.Series], None]) -> Dict[str, pd.Series]:
        """Convert supported base feature structures into a column-keyed mapping."""
        normalized: Dict[str, pd.Series] = {}
        
        try:
            if base_features is None:
                return normalized
            
            if isinstance(base_features, pd.DataFrame):
                for column in base_features.columns:
                    series = base_features[column]
                    if isinstance(series, pd.Series):
                        normalized[column] = series
            elif isinstance(base_features, dict):
                for name, series in base_features.items():
                    if isinstance(series, pd.Series):
                        normalized[name] = series
            
            tprint_debug(f"Normalized base feature mapping | total={len(normalized)}")
            return normalized
            
        except Exception as e:
            tprint_error(f"Base feature normalization failed: {e}")
            return {}
    
    def _determine_budget_allocation(self, materialized_htfs: Dict[str, Any]) -> Dict[str, int]:
        """Determine budget allocation for different interaction types."""
        if self.config.budget_allocation:
            return self.config.budget_allocation
        
        # Base budget
        total_budget = self.config.max_interactions
        
        # Calculate HTF performance
        htf_utilities = []
        for feature_name, feature in materialized_htfs.items():
            if hasattr(feature, 'utility_score'):
                htf_utilities.append(feature.utility_score)
        
        avg_htf_utility = np.mean(htf_utilities) if htf_utilities else 0.0
        
        # Allocate budget based on HTF performance
        if avg_htf_utility > 0.1:  # Top-quartile performance
            core_budget = total_budget // 2
            htf_aware_budget = total_budget - core_budget
        else:
            core_budget = int(total_budget * 0.7)
            htf_aware_budget = total_budget - core_budget
        
        allocation = {
            'core': core_budget,
            'htf_aware': htf_aware_budget
        }
        
        tprint_debug(f"Budget allocation: {allocation}")
        return allocation
    
    def _generate_core_interactions_vectorbt(self, base_features: Dict[str, pd.Series], 
                                           targets: Optional[pd.Series], 
                                           budget: int) -> List[GeneratedInteraction]:
        """Generate core interactions using VectorBT optimization."""
        interactions = []
        
        try:
            if not VECTORBT_AVAILABLE:
                tprint_warning("VectorBT not available, using fallback method")
                return self._generate_core_interactions_fallback(base_features, targets, budget)
            
            # Group features by type
            feature_groups = self._group_features_by_type(base_features)
            
            # Generate interactions for each template
            for template in self.core_templates.templates[:budget]:
                template_interactions = self._generate_template_interactions_vectorbt(
                    template, feature_groups, base_features, targets
                )
                interactions.extend(template_interactions)
                
                if len(interactions) >= budget:
                    break
            
            tprint_success(f"Generated {len(interactions)} core interactions")
            return interactions
            
        except Exception as e:
            tprint_error(f"Core interaction generation failed: {e}")
            return self._generate_core_interactions_fallback(base_features, targets, budget)
    
    def _generate_htf_interactions_vectorbt(self, materialized_htfs: Dict[str, Any],
                                          base_features: Dict[str, pd.Series],
                                          targets: Optional[pd.Series],
                                          budget: int) -> List[GeneratedInteraction]:
        """Generate HTF-aware interactions using VectorBT optimization."""
        interactions = []
        
        try:
            if not VECTORBT_AVAILABLE:
                tprint_warning("VectorBT not available, using fallback method")
                return self._generate_htf_interactions_fallback(materialized_htfs, base_features, targets, budget)
            
            # Group HTF features by type
            htf_groups = self._group_htf_features_by_type(materialized_htfs)
            base_groups = self._group_features_by_type(base_features)
            
            # Generate interactions for each template
            for template in self.htf_aware_templates.templates[:budget]:
                template_interactions = self._generate_htf_template_interactions_vectorbt(
                    template, htf_groups, base_groups, base_features, targets
                )
                interactions.extend(template_interactions)
                
                if len(interactions) >= budget:
                    break
            
            tprint_success(f"Generated {len(interactions)} HTF-aware interactions")
            return interactions
            
        except Exception as e:
            tprint_error(f"HTF interaction generation failed: {e}")
            return self._generate_htf_interactions_fallback(materialized_htfs, base_features, targets, budget)
    
    def _group_features_by_type(self, features: Dict[str, pd.Series]) -> Dict[str, List[str]]:
        """Group features by type."""
        groups = {
            'price_feature': [],
            'volatility_feature': [],
            'momentum_feature': [],
            'mean_reversion_feature': [],
            'liquidity_feature': [],
            'volume_feature': [],
            'tod_indicator': [],
            'regime_indicator': [],
            'feature': [],
            'feature1': [],
            'feature2': [],
            'condition': [],
            'threshold': []
        }
        
        for name, series in features.items():
            name_lower = name.lower()
            
            # Categorize based on feature name
            if any(x in name_lower for x in ['price', 'close', 'open', 'high', 'low']):
                groups['price_feature'].append(name)
            elif any(x in name_lower for x in ['vol', 'sigma', 'rv', 'gk']):
                groups['volatility_feature'].append(name)
            elif any(x in name_lower for x in ['mom', 'momentum', 'signal', 'alpha']):
                groups['momentum_feature'].append(name)
            elif any(x in name_lower for x in ['rsi', 'stoch', 'mean_rev', 'osc']):
                groups['mean_reversion_feature'].append(name)
            elif any(x in name_lower for x in ['liquidity', 'depth', 'book']):
                groups['liquidity_feature'].append(name)
            elif 'volume' in name_lower:
                groups['volume_feature'].append(name)
            elif any(x in name_lower for x in ['tod', 'time_of_day', 'session']):
                groups['tod_indicator'].append(name)
            elif any(x in name_lower for x in ['regime', 'vol_regime']):
                groups['regime_indicator'].append(name)
            
            # Add to general feature groups
            groups['feature'].append(name)
            groups['feature1'].append(name)
            groups['feature2'].append(name)
            groups['condition'].append(name)
        
        return groups
    
    def _group_htf_features_by_type(self, materialized_htfs: Dict[str, Any]) -> Dict[str, List[str]]:
        """Group HTF features by type."""
        groups = {
            'htf_trend_feature': [],
            'htf_volatility_feature': [],
            'htf_momentum_feature': [],
            'htf_anchor_feature': [],
            'htf_regime_feature': []
        }
        
        for name, feature in materialized_htfs.items():
            family = getattr(feature, 'family', 'unknown') or 'unknown'
            metadata = getattr(feature, 'metadata', {}) or {}
            base_feature = str(metadata.get('base_feature', '')).lower()
            name_lower = name.lower()
            
            if (
                family in ['trend_level_vol']
                or any(key in base_feature for key in ['ema', 'trend'])
                or any(key in name_lower for key in ['trend', 'ema'])
            ):
                groups['htf_trend_feature'].append(name)
            
            if (
                (family in ['trend_level_vol'] and any(
                    key in name_lower for key in ['vol', 'sigma', 'rv', 'var']
                ))
                or any(key in base_feature for key in ['vol', 'sigma', 'rv', 'var'])
            ):
                groups['htf_volatility_feature'].append(name)
            
            if (
                family == 'oscillators'
                or any(key in base_feature for key in ['rsi', 'stoch', 'momentum', 'mom', 'osc'])
                or any(key in name_lower for key in ['rsi', 'stoch', 'momentum', 'osc'])
            ):
                groups['htf_momentum_feature'].append(name)
            
            if (
                family == 'anchors'
                or any(key in base_feature for key in ['vwap', 'anchor'])
                or any(key in name_lower for key in ['vwap', 'anchor'])
            ):
                groups['htf_anchor_feature'].append(name)
            
            if (
                any(key in metadata for key in ['regime', 'regime_type', 'dominant_regime', 'regime_label'])
                or metadata.get('is_regime_feature')
            ):
                groups['htf_regime_feature'].append(name)
        
        return groups
    
    def _generate_template_interactions_vectorbt(self, template: InteractionTemplate,
                                              feature_groups: Dict[str, List[str]],
                                              base_features: Dict[str, pd.Series],
                                              targets: Optional[pd.Series]) -> List[GeneratedInteraction]:
        """Generate interactions for a specific template using VectorBT optimization."""
        interactions = []
        
        try:
            # Get feature combinations for this template
            feature_combinations = self._get_feature_combinations_vectorbt(
                template, feature_groups, base_features
            )
            
            for combination in feature_combinations:
                try:
                    # Generate interaction using VectorBT
                    interaction_series = self._calculate_interaction_vectorbt(
                        template, combination, base_features
                    )
                    
                    if interaction_series is not None and self._is_valid_interaction(interaction_series):
                        # Calculate utility score
                        utility_score = self._calculate_utility_score_vectorbt(
                            interaction_series, targets
                        )
                        
                        # Create interaction object
                        interaction = GeneratedInteraction(
                            name=f"{template.name}_{combination['name']}",
                            formula=template.formula,
                            parent_features=combination['features'],
                            interaction_type=template.template_type,
                            feature_series=interaction_series,
                            utility_score=utility_score,
                            metadata={
                                'template': template.name,
                                'combination': combination,
                                'vectorbt_optimized': True
                            }
                        )
                        
                        interactions.append(interaction)
                        
                except Exception as e:
                    tprint_debug(f"Template interaction generation failed: {e}")
                    continue
            
            return interactions
            
        except Exception as e:
            tprint_warning(f"Template {template.name} failed: {e}")
            return []
    
    def _generate_htf_template_interactions_vectorbt(self, template: InteractionTemplate,
                                                   htf_groups: Dict[str, List[str]],
                                                   base_groups: Dict[str, List[str]],
                                                   base_features: Dict[str, pd.Series],
                                                   targets: Optional[pd.Series]) -> List[GeneratedInteraction]:
        """Generate HTF-aware interactions for a specific template using VectorBT optimization."""
        interactions = []
        
        try:
            # Get feature combinations for this template
            feature_combinations = self._get_htf_feature_combinations_vectorbt(
                template, htf_groups, base_groups, base_features
            )
            
            for combination in feature_combinations:
                try:
                    # Generate interaction using VectorBT
                    interaction_series = self._calculate_htf_interaction_vectorbt(
                        template, combination, base_features
                    )
                    
                    if interaction_series is not None and self._is_valid_interaction(interaction_series):
                        # Calculate utility score
                        utility_score = self._calculate_utility_score_vectorbt(
                            interaction_series, targets
                        )
                        
                        # Create interaction object
                        interaction = GeneratedInteraction(
                            name=f"{template.name}_{combination['name']}",
                            formula=template.formula,
                            parent_features=combination['features'],
                            interaction_type=template.template_type,
                            feature_series=interaction_series,
                            utility_score=utility_score,
                            metadata={
                                'template': template.name,
                                'combination': combination,
                                'vectorbt_optimized': True
                            }
                        )
                        
                        interactions.append(interaction)
                        
                except Exception as e:
                    tprint_debug(f"HTF template interaction generation failed: {e}")
                    continue
            
            return interactions
            
        except Exception as e:
            tprint_warning(f"HTF template {template.name} failed: {e}")
            return []
    
    def _get_feature_combinations_vectorbt(self, template: InteractionTemplate,
                                         feature_groups: Dict[str, List[str]],
                                         base_features: Dict[str, pd.Series]) -> List[Dict[str, Any]]:
        """Get feature combinations for a template using VectorBT optimization."""
        combinations = []
        
        try:
            required_features = template.required_features
            optional_features = template.optional_features
            
            # Get required feature lists
            required_lists = [feature_groups.get(req, []) for req in required_features]
            
            # Generate Cartesian product
            for combo in product(*required_lists):
                combination = dict(zip(required_features, combo))
                
                # Add optional features if available
                for opt in optional_features:
                    if opt in feature_groups and feature_groups[opt]:
                        combination[opt] = feature_groups[opt][0]
                
                # Create combination name
                combination_name = "_".join(combination.values())
                
                combinations.append({
                    'name': combination_name,
                    'features': list(combination.values()),
                    'data': [base_features.get(feat) for feat in combination.values() if feat in base_features]
                })
            
            return combinations[:template.max_instances]
            
        except Exception as e:
            tprint_warning(f"Feature combination generation failed: {e}")
            return []
    
    def _get_htf_feature_combinations_vectorbt(self, template: InteractionTemplate,
                                             htf_groups: Dict[str, List[str]],
                                             base_groups: Dict[str, List[str]],
                                             base_features: Dict[str, pd.Series]) -> List[Dict[str, Any]]:
        """Get HTF feature combinations for a template using VectorBT optimization."""
        combinations = []
        
        try:
            required_features = template.required_features
            
            # Map required features to appropriate groups
            required_lists = []
            for req in required_features:
                if req.startswith('htf_'):
                    required_lists.append(htf_groups.get(req, []))
                else:
                    required_lists.append(base_groups.get(req, []))
            
            # Generate Cartesian product
            for combo in product(*required_lists):
                combination = dict(zip(required_features, combo))
                
                # Create combination name
                combination_name = "_".join(combination.values())
                
                # Get data for the combination
                combination_data = []
                for feat in combination.values():
                    if feat in base_features:
                        combination_data.append(base_features[feat])
                    else:
                        # This would be an HTF feature - simplified for now
                        combination_data.append(pd.Series([0] * len(list(base_features.values())[0])))
                
                combinations.append({
                    'name': combination_name,
                    'features': list(combination.values()),
                    'data': combination_data
                })
            
            return combinations[:template.max_instances]
            
        except Exception as e:
            tprint_warning(f"HTF feature combination generation failed: {e}")
            return []
    
    def _calculate_interaction_vectorbt(self, template: InteractionTemplate,
                                      combination: Dict[str, Any],
                                      base_features: Dict[str, pd.Series]) -> Optional[pd.Series]:
        """Calculate interaction using VectorBT optimization."""
        try:
            if not VECTORBT_AVAILABLE:
                return self._calculate_interaction_fallback(template, combination, base_features)
            
            # VectorBT-optimized interaction calculation
            if template.name == "price_vol_interaction":
                if len(combination['data']) >= 2:
                    return combination['data'][0] * combination['data'][1]
            elif template.name == "momentum_meanrev_interaction":
                if len(combination['data']) >= 2:
                    return combination['data'][0] * combination['data'][1]
            elif template.name == "vol_volume_interaction":
                if len(combination['data']) >= 2:
                    return combination['data'][0] * combination['data'][1]
            elif template.name == "ratio_interaction":
                if len(combination['data']) >= 2:
                    return combination['data'][0] / (combination['data'][1] + 1e-8)
            elif template.name == "difference_interaction":
                if len(combination['data']) >= 2:
                    return combination['data'][0] - combination['data'][1]
            elif template.name == "product_interaction":
                if len(combination['data']) >= 2:
                    return combination['data'][0] * combination['data'][1]
            elif template.name == "polynomial_interaction":
                if len(combination['data']) >= 1:
                    return combination['data'][0] ** 2
            elif template.name == "rolling_interaction":
                if len(combination['data']) >= 1:
                    return rolling_mean(combination['data'][0], window=20)
            elif template.name == "zscore_interaction":
                if len(combination['data']) >= 1:
                    return zscore(combination['data'][0])
            elif template.name == "log_interaction":
                if len(combination['data']) >= 1:
                    return np.log(combination['data'][0] + 1e-8)
            elif template.name == "log_ratio_interaction":
                if len(combination['data']) >= 2:
                    return np.log(combination['data'][0] + 1e-8) - np.log(combination['data'][1] + 1e-8)
            elif template.name == "log_product_interaction":
                if len(combination['data']) >= 2:
                    return np.log(combination['data'][0] + 1e-8) + np.log(combination['data'][1] + 1e-8)
            
            # Default fallback
            if len(combination['data']) >= 2:
                return combination['data'][0] * combination['data'][1]
            elif len(combination['data']) >= 1:
                return combination['data'][0]
            
        except Exception as e:
            tprint_debug(f"VectorBT interaction calculation failed: {e}")
        
        return None
    
    def _calculate_htf_interaction_vectorbt(self, template: InteractionTemplate,
                                         combination: Dict[str, Any],
                                         base_features: Dict[str, pd.Series]) -> Optional[pd.Series]:
        """Calculate HTF interaction using VectorBT optimization."""
        try:
            if not VECTORBT_AVAILABLE:
                return self._calculate_htf_interaction_fallback(template, combination, base_features)
            
            # VectorBT-optimized HTF interaction calculation
            if template.name == "htf_trend_liquidity_interaction":
                if len(combination['data']) >= 2:
                    return combination['data'][0] * combination['data'][1]
            elif template.name == "htf_vol_signal_interaction":
                if len(combination['data']) >= 2:
                    return combination['data'][0] * combination['data'][1]
            elif template.name == "htf_momentum_conflict_interaction":
                if len(combination['data']) >= 2:
                    return combination['data'][0] * (-combination['data'][1])
            elif template.name == "htf_regime_base_interaction":
                if len(combination['data']) >= 2:
                    return combination['data'][0] * combination['data'][1]
            elif template.name == "htf_anchor_deviation_interaction":
                if len(combination['data']) >= 2:
                    return combination['data'][0] * combination['data'][1]
            elif template.name == "htf_log_trend_interaction":
                if len(combination['data']) >= 2:
                    return np.log(combination['data'][0] + 1e-8) * combination['data'][1]
            elif template.name == "htf_log_vol_signal_interaction":
                if len(combination['data']) >= 2:
                    return np.log(combination['data'][0] + 1e-8) * combination['data'][1]
            elif template.name == "htf_log_momentum_interaction":
                if len(combination['data']) >= 2:
                    return np.log(combination['data'][0] + 1e-8) * np.log(combination['data'][1] + 1e-8)
            elif template.name == "htf_log_regime_interaction":
                if len(combination['data']) >= 2:
                    return np.log(combination['data'][0] + 1e-8) * combination['data'][1]
            
            # Default fallback
            if len(combination['data']) >= 2:
                return combination['data'][0] * combination['data'][1]
            elif len(combination['data']) >= 1:
                return combination['data'][0]
            
        except Exception as e:
            tprint_debug(f"VectorBT HTF interaction calculation failed: {e}")
        
        return None
    
    def _calculate_utility_score_vectorbt(self, interaction_series: pd.Series, 
                                        targets: Optional[pd.Series]) -> float:
        """Calculate utility score using VectorBT optimization."""
        try:
            if targets is None:
                # Use variance as utility score
                return float(interaction_series.var())
            
            # Calculate correlation with targets
            correlation = interaction_series.corr(targets)
            if pd.isna(correlation):
                return 0.0
            
            # Use absolute correlation as utility score
            return abs(correlation)
            
        except Exception as e:
            tprint_debug(f"Utility score calculation failed: {e}")
            return 0.0
    
    def _is_valid_interaction(self, series: pd.Series) -> bool:
        """Check if an interaction series is valid."""
        if series is None or series.empty:
            return False
        
        # Check for all NaN values
        if series.isna().all():
            return False
        
        # Check for infinite values
        if np.isinf(series).any():
            return False
        
        # Check for constant values (no variance)
        if series.nunique() <= 1:
            return False
        
        return True
    
    def _apply_vectorbt_feature_selection(self, interactions: List[GeneratedInteraction],
                                        targets: Optional[pd.Series]) -> List[GeneratedInteraction]:
        """Apply VectorBT-based feature selection."""
        if not interactions:
            return interactions
        
        try:
            # Sort by utility score
            interactions.sort(key=lambda x: x.utility_score, reverse=True)
            
            # Select top interactions
            max_interactions = min(len(interactions), self.config.max_interactions)
            selected = interactions[:max_interactions]
            
            # Apply additional VectorBT-based filtering
            if VECTORBT_AVAILABLE and targets is not None:
                selected = self._filter_correlated_interactions_vectorbt(selected, targets)
            
            return selected
            
        except Exception as e:
            tprint_warning(f"VectorBT feature selection failed: {e}, returning all interactions")
            return interactions
    
    def _filter_correlated_interactions_vectorbt(self, interactions: List[GeneratedInteraction],
                                               targets: pd.Series) -> List[GeneratedInteraction]:
        """Filter highly correlated interactions using VectorBT."""
        if len(interactions) <= 1:
            return interactions
        
        try:
            # Create DataFrame of interaction features
            interaction_df = pd.DataFrame({
                interaction.name: interaction.feature_series 
                for interaction in interactions
            })
            
            # Calculate correlation matrix
            corr_matrix = interaction_df.corr()
            
            # Find highly correlated pairs
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > self.config.correlation_threshold:
                        high_corr_pairs.append((i, j))
            
            # Remove one from each highly correlated pair (keep the one with higher utility)
            to_remove = set()
            for i, j in high_corr_pairs:
                if interactions[i].utility_score >= interactions[j].utility_score:
                    to_remove.add(j)
                else:
                    to_remove.add(i)
            
            # Filter out highly correlated interactions
            filtered_interactions = [
                interaction for i, interaction in enumerate(interactions)
                if i not in to_remove
            ]
            
            return filtered_interactions
            
        except Exception as e:
            tprint_warning(f"VectorBT correlation filtering failed: {e}")
            return interactions
    
    def _create_empty_result(self, start_time: float, error_message: str) -> List[GeneratedInteraction]:
        """Create empty result for failed generation."""
        return []
    
    # Fallback methods for when VectorBT is not available
    def _generate_core_interactions_fallback(self, base_features: Dict[str, pd.Series], 
                                           targets: Optional[pd.Series], 
                                           budget: int) -> List[GeneratedInteraction]:
        """Fallback method for core interactions when VectorBT is not available."""
        return []
    
    def _generate_htf_interactions_fallback(self, materialized_htfs: Dict[str, Any],
                                          base_features: Dict[str, pd.Series],
                                          targets: Optional[pd.Series],
                                          budget: int) -> List[GeneratedInteraction]:
        """Fallback method for HTF interactions when VectorBT is not available."""
        return []
    
    def _calculate_interaction_fallback(self, template: InteractionTemplate,
                                      combination: Dict[str, Any],
                                      base_features: Dict[str, pd.Series]) -> Optional[pd.Series]:
        """Fallback interaction calculation when VectorBT is not available."""
        try:
            if len(combination['data']) >= 2:
                return combination['data'][0] * combination['data'][1]
            elif len(combination['data']) >= 1:
                return combination['data'][0]
        except:
            pass
        return None
    
    def _calculate_htf_interaction_fallback(self, template: InteractionTemplate,
                                         combination: Dict[str, Any],
                                         base_features: Dict[str, pd.Series]) -> Optional[pd.Series]:
        """Fallback HTF interaction calculation when VectorBT is not available."""
        try:
            if len(combination['data']) >= 2:
                return combination['data'][0] * combination['data'][1]
            elif len(combination['data']) >= 1:
                return combination['data'][0]
        except:
            pass
        return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'total_generations': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'total_execution_time': 0.0,
            'core_interactions_generated': 0,
            'htf_interactions_generated': 0,
            'vectorbt_operations': 0
        }


def create_htf_interaction_generator(config: Optional[HTFTemplateConfig] = None) -> HTFInteractionGenerator:
    """Create an HTF interaction generator with default configuration."""
    return HTFInteractionGenerator(config)