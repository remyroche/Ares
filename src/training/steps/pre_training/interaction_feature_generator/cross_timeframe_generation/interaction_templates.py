"""
HTF-Aware Interaction Templates

Implements HTF-aware interaction generation with:
- Core 15 interactions (theory-first)
- HTF-aware templates (2-3 additional)
- Dynamic budget allocation based on HTF performance
- Cross-asset HTF interactions (optional)
- Interaction heredity enforcement
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from itertools import combinations, product
import warnings
warnings.filterwarnings('ignore')

try:
    from src.utils.tprint import (

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
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
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
        tprint,
        tprint_info,
        tprint_warning,
        tprint_error,
        tprint_success,
        tprint_debug,
        tprint_progress,
        tprint_performance,
    )
    TPRINT_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback for standalone usage
    TPRINT_AVAILABLE = False

    def tprint(*args, **kwargs):
        print(*args, **kwargs)

    def tprint_info(*args, **kwargs):
        print("INFO:", *args, **kwargs)

    def tprint_warning(*args, **kwargs):
        print("WARNING:", *args, **kwargs)

    def tprint_error(*args, **kwargs):
        print("ERROR:", *args, **kwargs)

    def tprint_success(*args, **kwargs):
        print("SUCCESS:", *args, **kwargs)

    def tprint_debug(*args, **kwargs):
        print("DEBUG:", *args, **kwargs)

    def tprint_progress(*args, **kwargs):
        print("PROGRESS:", *args, **kwargs)

    def tprint_performance(*args, **kwargs):
        print("PERFORMANCE:", *args, **kwargs)


@dataclass
class InteractionTemplate:
    """Template for generating interactions."""
    name: str
    template_type: str  # 'core', 'htf_aware', 'cross_asset'
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


class CoreInteractionTemplates:
    """Core 15 interaction templates (theory-first)."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.templates = self._create_core_templates()
        tprint_info(
            "Initialized core interaction templates | count=%d | types=%s"
            % (
                len(self.templates),
                sorted({template.template_type for template in self.templates}),
            )
        )
    
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
            )
        ]
        
        tprint_debug(
            "Constructed core template catalogue | total=%d | names=%s"
            % (len(templates), [template.name for template in templates])
        )
        return templates


class HTFAwareTemplates:
    """HTF-aware interaction templates."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.templates = self._create_htf_aware_templates()
        tprint_info(
            "Initialized HTF-aware templates | count=%d | names=%s"
            % (len(self.templates), [template.name for template in self.templates])
        )
    
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
            )
        ]
        
        tprint_debug(
            "Constructed HTF-aware template set | total=%d | names=%s"
            % (len(templates), [template.name for template in templates])
        )
        return templates


class CrossAssetTemplates:
    """Cross-asset HTF interaction templates."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.templates = self._create_cross_asset_templates()
        tprint_info(
            "Initialized cross-asset templates | count=%d | supports_lag=%s"
            % (
                len(self.templates),
                any('lag' in template.optional_features for template in self.templates),
            )
        )
    
    def _create_cross_asset_templates(self) -> List[InteractionTemplate]:
        """Create cross-asset HTF interaction templates."""
        templates = [
            # Lead-lag interactions
            InteractionTemplate(
                name="lead_lag_interaction",
                template_type="cross_asset",
                formula="asset1_htf_feature * asset2_htf_feature.shift(lag)",
                required_features=["asset1_htf_feature", "asset2_htf_feature"],
                optional_features=["lag"],
                max_instances=2,
                priority=1,
                metadata={"description": "Cross-asset lead-lag interaction"}
            ),
            
            # Correlation interactions
            InteractionTemplate(
                name="correlation_interaction",
                template_type="cross_asset",
                formula="asset1_htf_feature * asset2_htf_feature",
                required_features=["asset1_htf_feature", "asset2_htf_feature"],
                optional_features=[],
                max_instances=2,
                priority=1,
                metadata={"description": "Cross-asset correlation interaction"}
            ),
            
            # Relative strength interactions
            InteractionTemplate(
                name="relative_strength_interaction",
                template_type="cross_asset",
                formula="asset_htf_feature / market_htf_feature",
                required_features=["asset_htf_feature", "market_htf_feature"],
                optional_features=[],
                max_instances=2,
                priority=2,
                metadata={"description": "Cross-asset relative strength interaction"}
            )
        ]
        
        tprint_debug(
            "Constructed cross-asset template catalogue | total=%d | names=%s"
            % (len(templates), [template.name for template in templates])
        )
        return templates


class InteractionGenerator:
    """Generates interactions from templates."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.core_templates = CoreInteractionTemplates()
        self.htf_aware_templates = HTFAwareTemplates()
        self.cross_asset_templates = CrossAssetTemplates()
        tprint_debug(
            "InteractionGenerator initialized | core=%d | htf=%d | cross_asset=%d"
            % (
                len(self.core_templates.templates),
                len(self.htf_aware_templates.templates),
                len(self.cross_asset_templates.templates),
            )
        )
    
    def generate_interactions(self,
                            materialized_htfs: Dict[str, Any],
                            base_features: Union[pd.DataFrame, Mapping[str, pd.Series], None],
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
        self.logger.info("Starting VectorBT-optimized interaction generation")
        htf_feature_count = len(materialized_htfs) if hasattr(materialized_htfs, '__len__') else 0
        if isinstance(base_features, pd.DataFrame):
            base_feature_count = len(base_features.columns)
        elif hasattr(base_features, '__len__') and not isinstance(base_features, (pd.Series, pd.Index)):
            try:
                base_feature_count = len(base_features)
            except TypeError:
                base_feature_count = 0
        else:
            base_feature_count = 0

        tprint_info(
            "VectorBT-optimized interaction generation started | htf_features=%d | base_features=%d | targets=%s"
            % (htf_feature_count, base_feature_count, targets is not None)
        )

        normalized_base_features = self._normalize_base_features(base_features)
        tprint_debug(
            "Normalized base features | columns=%d"
            % (len(normalized_base_features))
        )

        # Determine budget allocation
        budget_allocation = self._determine_budget_allocation(materialized_htfs)
        tprint_debug(
            "Budget allocation computed | core=%d | htf=%d | cross_asset=%d"
            % (
                budget_allocation['core'],
                budget_allocation['htf_aware'],
                budget_allocation['cross_asset'],
            )
        )

        # Generate core interactions with VectorBT optimization
        core_interactions = self._generate_core_interactions_vectorbt(
            normalized_base_features, targets, budget_allocation['core']
        )

        # Generate HTF-aware interactions with VectorBT optimization
        htf_interactions = self._generate_htf_interactions_vectorbt(
            materialized_htfs, normalized_base_features, targets, budget_allocation['htf_aware']
        )

        # Generate cross-asset interactions with VectorBT optimization
        cross_asset_interactions = self._generate_cross_asset_interactions_vectorbt(
            materialized_htfs, normalized_base_features, targets, budget_allocation['cross_asset']
        )

        # Combine all interactions
        all_interactions = core_interactions + htf_interactions + cross_asset_interactions

        # Apply VectorBT-based feature selection
        selected_interactions = self._apply_vectorbt_feature_selection(all_interactions, targets)

        tprint_success(
            "VectorBT-optimized interaction generation completed | total=%d | selected=%d"
            % (len(all_interactions), len(selected_interactions))
        )

        return selected_interactions

    def _generate_core_interactions_vectorbt(self, base_features: pd.DataFrame, 
                                           targets: Optional[pd.Series], 
                                           budget: int) -> List[GeneratedInteraction]:
        """Generate core interactions using VectorBT optimization."""
        interactions = []
        
        if not VECTORBT_AVAILABLE:
            tprint_warning("VectorBT not available, using fallback method")
            return self._generate_core_interactions_fallback(base_features, targets, budget)
        
        try:
            # VectorBT-optimized core interaction generation
            for template in self.core_templates.templates[:budget]:
                template_interactions = self._generate_template_interactions_vectorbt(
                    template, base_features, targets
                )
                interactions.extend(template_interactions)
                
        except Exception as e:
            tprint_error(f"VectorBT core interactions failed: {e}, using fallback")
            return self._generate_core_interactions_fallback(base_features, targets, budget)
        
        return interactions

    def _generate_htf_interactions_vectorbt(self, materialized_htfs: Dict[str, Any],
                                          base_features: pd.DataFrame,
                                          targets: Optional[pd.Series],
                                          budget: int) -> List[GeneratedInteraction]:
        """Generate HTF-aware interactions using VectorBT optimization."""
        interactions = []
        
        if not VECTORBT_AVAILABLE:
            tprint_warning("VectorBT not available, using fallback method")
            return self._generate_htf_interactions_fallback(materialized_htfs, base_features, targets, budget)
        
        try:
            # VectorBT-optimized HTF interaction generation
            for template in self.htf_aware_templates.templates[:budget]:
                template_interactions = self._generate_template_interactions_vectorbt(
                    template, base_features, targets, materialized_htfs
                )
                interactions.extend(template_interactions)
                
        except Exception as e:
            tprint_error(f"VectorBT HTF interactions failed: {e}, using fallback")
            return self._generate_htf_interactions_fallback(materialized_htfs, base_features, targets, budget)
        
        return interactions

    def _generate_cross_asset_interactions_vectorbt(self, materialized_htfs: Dict[str, Any],
                                                  base_features: pd.DataFrame,
                                                  targets: Optional[pd.Series],
                                                  budget: int) -> List[GeneratedInteraction]:
        """Generate cross-asset interactions using VectorBT optimization."""
        interactions = []
        
        if not VECTORBT_AVAILABLE:
            tprint_warning("VectorBT not available, using fallback method")
            return self._generate_cross_asset_interactions_fallback(materialized_htfs, base_features, targets, budget)
        
        try:
            # VectorBT-optimized cross-asset interaction generation
            for template in self.cross_asset_templates.templates[:budget]:
                template_interactions = self._generate_template_interactions_vectorbt(
                    template, base_features, targets, materialized_htfs
                )
                interactions.extend(template_interactions)
                
        except Exception as e:
            tprint_error(f"VectorBT cross-asset interactions failed: {e}, using fallback")
            return self._generate_cross_asset_interactions_fallback(materialized_htfs, base_features, targets, budget)
        
        return interactions

    def _generate_template_interactions_vectorbt(self, template: InteractionTemplate,
                                              base_features: pd.DataFrame,
                                              targets: Optional[pd.Series],
                                              materialized_htfs: Optional[Dict[str, Any]] = None) -> List[GeneratedInteraction]:
        """Generate interactions from a template using VectorBT optimization."""
        interactions = []
        
        try:
            # Get feature combinations for this template
            feature_combinations = self._get_feature_combinations_vectorbt(
                template, base_features, materialized_htfs
            )
            
            for combination in feature_combinations:
                try:
                    # Generate interaction using VectorBT
                    interaction_series = self._calculate_interaction_vectorbt(
                        template, combination, base_features, materialized_htfs
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
                    
        except Exception as e:
            tprint_warning(f"Template {template.name} failed: {e}")
        
        return interactions

    def _get_feature_combinations_vectorbt(self, template: InteractionTemplate,
                                         base_features: pd.DataFrame,
                                         materialized_htfs: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get feature combinations for a template using VectorBT optimization."""
        combinations = []
        
        # Get available features
        available_features = {}
        
        # Add base features
        for col in base_features.columns:
            available_features[col] = base_features[col]
        
        # Add HTF features if available
        if materialized_htfs:
            for name, series in materialized_htfs.items():
                if isinstance(series, pd.Series):
                    available_features[f"htf_{name}"] = series
        
        # Generate combinations based on template requirements
        required_features = template.required_features
        optional_features = template.optional_features
        
        # Simple combination generation (can be enhanced)
        if len(required_features) == 2:
            # Two-feature interactions
            for feat1 in available_features:
                for feat2 in available_features:
                    if feat1 != feat2:
                        combinations.append({
                            'name': f"{feat1}_{feat2}",
                            'features': [feat1, feat2],
                            'data': [available_features[feat1], available_features[feat2]]
                        })
        elif len(required_features) == 1:
            # Single-feature interactions
            for feat in available_features:
                combinations.append({
                    'name': f"{feat}_single",
                    'features': [feat],
                    'data': [available_features[feat]]
                })
        
        return combinations[:template.max_instances]

    def _calculate_interaction_vectorbt(self, template: InteractionTemplate,
                                      combination: Dict[str, Any],
                                      base_features: pd.DataFrame,
                                      materialized_htfs: Optional[Dict[str, Any]] = None) -> Optional[pd.Series]:
        """Calculate interaction using VectorBT optimization."""
        try:
            if template.name == "price_vol_interaction":
                # Price-volatility interaction
                if len(combination['data']) >= 2:
                    price_feature = combination['data'][0]
                    vol_feature = combination['data'][1]
                    return price_feature * vol_feature
                    
            elif template.name == "momentum_meanrev_interaction":
                # Momentum-mean reversion interaction
                if len(combination['data']) >= 2:
                    momentum_feature = combination['data'][0]
                    meanrev_feature = combination['data'][1]
                    return momentum_feature * meanrev_feature
                    
            elif template.name == "vol_volume_interaction":
                # Volatility-volume interaction
                if len(combination['data']) >= 2:
                    vol_feature = combination['data'][0]
                    volume_feature = combination['data'][1]
                    return vol_feature * volume_feature
                    
            elif template.name == "ratio_interaction":
                # Ratio interaction
                if len(combination['data']) >= 2:
                    feat1 = combination['data'][0]
                    feat2 = combination['data'][1]
                    return feat1 / (feat2 + 1e-08)
                    
            elif template.name == "difference_interaction":
                # Difference interaction
                if len(combination['data']) >= 2:
                    feat1 = combination['data'][0]
                    feat2 = combination['data'][1]
                    return feat1 - feat2
                    
            elif template.name == "product_interaction":
                # Product interaction
                if len(combination['data']) >= 2:
                    feat1 = combination['data'][0]
                    feat2 = combination['data'][1]
                    return feat1 * feat2
                    
            elif template.name == "polynomial_interaction":
                # Polynomial interaction
                if len(combination['data']) >= 1:
                    feat = combination['data'][0]
                    return feat ** 2
                    
            elif template.name == "rolling_interaction":
                # Rolling interaction
                if len(combination['data']) >= 1:
                    feat = combination['data'][0]
                    window = 20  # Default window
                    if VECTORBT_AVAILABLE:
                        return rolling_mean(feat, window=window)
                    else:
                        return feat.rolling(window=window).mean()
                        
            elif template.name == "zscore_interaction":
                # Z-score interaction
                if len(combination['data']) >= 1:
                    feat = combination['data'][0]
                    if VECTORBT_AVAILABLE:
                        return zscore(feat)
                    else:
                        return (feat - feat.mean()) / feat.std()
            
            # Default fallback
            if len(combination['data']) >= 2:
                return combination['data'][0] * combination['data'][1]
            elif len(combination['data']) >= 1:
                return combination['data'][0]
                
        except Exception as e:
            tprint_debug(f"VectorBT interaction calculation failed: {e}")
        
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

    def _apply_vectorbt_feature_selection(self, interactions: List[GeneratedInteraction],
                                        targets: Optional[pd.Series]) -> List[GeneratedInteraction]:
        """Apply VectorBT-based feature selection."""
        if not interactions:
            return interactions
        
        try:
            # Sort by utility score
            interactions.sort(key=lambda x: x.utility_score, reverse=True)
            
            # Select top interactions
            max_interactions = min(len(interactions), 100)  # Limit to top 100
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
                    if abs(corr_matrix.iloc[i, j]) > 0.95:  # High correlation threshold
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

    # Fallback methods for when VectorBT is not available
    def _generate_core_interactions_fallback(self, base_features: pd.DataFrame, 
                                           targets: Optional[pd.Series], 
                                           budget: int) -> List[GeneratedInteraction]:
        """Fallback method for core interactions when VectorBT is not available."""
        # Implementation would go here
        return []

    def _generate_htf_interactions_fallback(self, materialized_htfs: Dict[str, Any],
                                          base_features: pd.DataFrame,
                                          targets: Optional[pd.Series],
                                          budget: int) -> List[GeneratedInteraction]:
        """Fallback method for HTF interactions when VectorBT is not available."""
        # Implementation would go here
        return []

    def _generate_cross_asset_interactions_fallback(self, materialized_htfs: Dict[str, Any],
                                                  base_features: pd.DataFrame,
                                                  targets: Optional[pd.Series],
                                                  budget: int) -> List[GeneratedInteraction]:
        """Fallback method for cross-asset interactions when VectorBT is not available."""
        # Implementation would go here
        return []enerate_htf_aware_interactions(
            materialized_htfs, normalized_base_features, targets, budget_allocation['htf_aware']
        )
        
        # Generate cross-asset interactions (if enabled)
        cross_asset_interactions = []
        if self.config.get('enable_cross_asset', False):
            cross_asset_interactions = self._generate_cross_asset_interactions(
                materialized_htfs, targets, budget_allocation['cross_asset']
            )
        
        # Combine all interactions
        all_interactions = core_interactions + htf_interactions + cross_asset_interactions
        
        # Apply interaction heredity
        filtered_interactions = self._apply_interaction_heredity(all_interactions)
        
        self.logger.info(f"Interaction generation completed: {len(filtered_interactions)} interactions generated")
        tprint_success(
            "Interaction generation completed | total=%d | core=%d | htf=%d | cross_asset=%d"
            % (
                len(filtered_interactions),
                len(core_interactions),
                len(htf_interactions),
                len(cross_asset_interactions),
            )
        )
        return filtered_interactions

    def _determine_budget_allocation(self, materialized_htfs: Dict[str, Any]) -> Dict[str, int]:
        """Determine budget allocation for different interaction types."""
        # Base budget
        total_budget = 30  # Maximum interactions

        # Calculate HTF performance
        htf_utilities = []
        for feature_name, feature in materialized_htfs.items():
            if hasattr(feature, 'utility_score'):
                htf_utilities.append(feature.utility_score)

        avg_htf_utility = np.mean(htf_utilities) if htf_utilities else 0.0
        tprint_debug(
            "Average HTF utility computed | count=%d | avg=%.4f"
            % (len(htf_utilities), avg_htf_utility)
        )

        # Allocate budget based on HTF performance
        if avg_htf_utility > 0.1:  # Top-quartile performance
            # Allow more HTF-aware interactions
            core_budget = 15
            htf_aware_budget = 10
            cross_asset_budget = 5
        else:
            # Standard allocation
            core_budget = 20
            htf_aware_budget = 7
            cross_asset_budget = 3
        
        allocation = {
            'core': core_budget,
            'htf_aware': htf_aware_budget,
            'cross_asset': cross_asset_budget
        }
        tprint_info(
            "Budget allocation finalized | total=%d | breakdown=%s"
            % (sum(allocation.values()), allocation)
        )
        return allocation
    
    def _generate_core_interactions(self,
                                  base_features: Dict[str, pd.Series],
                                  targets: Optional[pd.Series],
                                  budget: int) -> List[GeneratedInteraction]:
        """Generate core interactions."""
        interactions = []

        # Group features by type
        feature_groups = self._group_features_by_type(base_features)
        tprint_debug(
            "Generating core interactions | budget=%d | templates=%d"
            % (budget, len(self.core_templates.templates))
        )

        # Generate interactions for each template
        for template in self.core_templates.templates:
            if len(interactions) >= budget:
                break
            
            template_interactions = self._generate_template_interactions(
                template, feature_groups, targets
            )
            tprint_debug(
                "Template processed | name=%s | generated=%d"
                % (template.name, len(template_interactions))
            )

            # Limit by template max_instances
            template_interactions = template_interactions[:template.max_instances]
            interactions.extend(template_interactions)

        return interactions[:budget]
    
    def _generate_htf_aware_interactions(self,
                                       materialized_htfs: Dict[str, Any],
                                       base_features: Dict[str, pd.Series],
                                       targets: Optional[pd.Series],
                                       budget: int) -> List[GeneratedInteraction]:
        """Generate HTF-aware interactions."""
        interactions = []
        
        # Group HTF features by type
        htf_groups = self._group_htf_features_by_type(materialized_htfs)
        base_groups = self._group_features_by_type(base_features)
        tprint_debug(
            "Generating HTF-aware interactions | budget=%d | templates=%d | htf_feature_count=%d"
            % (
                budget,
                len(self.htf_aware_templates.templates),
                sum(len(v) for v in htf_groups.values()),
            )
        )

        # Generate interactions for each template
        for template in self.htf_aware_templates.templates:
            if len(interactions) >= budget:
                break

            template_interactions = self._generate_htf_template_interactions(
                template, htf_groups, base_groups, targets
            )
            tprint_debug(
                "HTF template processed | name=%s | generated=%d"
                % (template.name, len(template_interactions))
            )

            # Limit by template max_instances
            template_interactions = template_interactions[:template.max_instances]
            interactions.extend(template_interactions)

        return interactions[:budget]
    
    def _generate_cross_asset_interactions(self, 
                                         materialized_htfs: Dict[str, Any],
                                         targets: Optional[pd.Series],
                                         budget: int) -> List[GeneratedInteraction]:
        """Generate cross-asset interactions."""
        interactions = []
        
        # Group HTF features by asset
        asset_groups = self._group_htf_features_by_asset(materialized_htfs)
        tprint_debug(
            "Generating cross-asset interactions | budget=%d | templates=%d | assets=%d"
            % (
                budget,
                len(self.cross_asset_templates.templates),
                len(asset_groups),
            )
        )

        # Generate interactions for each template
        for template in self.cross_asset_templates.templates:
            if len(interactions) >= budget:
                break

            template_interactions = self._generate_cross_asset_template_interactions(
                template, asset_groups, targets
            )
            tprint_debug(
                "Cross-asset template processed | name=%s | generated=%d"
                % (template.name, len(template_interactions))
            )

            # Limit by template max_instances
            template_interactions = template_interactions[:template.max_instances]
            interactions.extend(template_interactions)

        return interactions[:budget]
    
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
            'regime_indicator': []
        }

        for name, series in features.items():
            # Categorize based on feature name
            name_lower = name.lower()
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

        # Provide aliases expected by interaction templates
        def _copy_list(values: List[str]) -> List[str]:
            return list(dict.fromkeys(values))

        liquidity_alias = _copy_list(groups['liquidity_feature'])
        signal_alias = _copy_list(groups['momentum_feature'] + groups['mean_reversion_feature'])
        momentum_alias = _copy_list(groups['momentum_feature'])

        deviation_candidates = []
        for name in features.keys():
            name_lower = name.lower()
            if any(x in name_lower for x in ['dev', 'dist', 'zscore', 'boll', 'spread']):
                deviation_candidates.append(name)

        base_feature_pool = _copy_list([
            name
            for key, names in groups.items()
            if key in {
                'price_feature',
                'volatility_feature',
                'momentum_feature',
                'mean_reversion_feature',
                'liquidity_feature',
                'volume_feature'
            }
            for name in names
        ])

        groups.update({
            'base_liquidity_feature': liquidity_alias,
            'base_signal_feature': signal_alias,
            'base_momentum_feature': momentum_alias,
            'base_deviation_feature': _copy_list(deviation_candidates),
            'base_feature': base_feature_pool
        })

        tprint_debug(
            "Grouped base features | counts=%s"
            % {key: len(value) if isinstance(value, list) else len(value) for key, value in groups.items()}
        )
        return groups

    def _normalize_base_features(self,
                                 base_features: Union[pd.DataFrame, Mapping[str, pd.Series], None]
                                 ) -> Dict[str, pd.Series]:
        """Convert supported base feature structures into a column-keyed mapping."""
        normalized: Dict[str, pd.Series] = {}

        if base_features is None:
            tprint_warning("No base features provided; normalization skipped")
            return normalized

        if isinstance(base_features, pd.DataFrame):
            for column in base_features.columns:
                series = base_features[column]
                if isinstance(series, pd.Series):
                    normalized[column] = series
        elif isinstance(base_features, Mapping):
            for name, series in base_features.items():
                if isinstance(series, pd.Series):
                    normalized[name] = series
        else:
            self.logger.warning(
                "Unsupported base feature structure %s; expected DataFrame or mapping of Series.",
                type(base_features)
            )

        tprint_debug(
            "Normalized base feature mapping | total=%d"
            % len(normalized)
        )
        return normalized
    
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
                if name not in groups['htf_trend_feature']:
                    groups['htf_trend_feature'].append(name)

            if (
                (family in ['trend_level_vol'] and any(
                    key in name_lower for key in ['vol', 'sigma', 'rv', 'var']
                ))
                or any(key in base_feature for key in ['vol', 'sigma', 'rv', 'var'])
            ):
                if name not in groups['htf_volatility_feature']:
                    groups['htf_volatility_feature'].append(name)

            if (
                family == 'oscillators'
                or any(key in base_feature for key in ['rsi', 'stoch', 'momentum', 'mom', 'osc'])
                or any(key in name_lower for key in ['rsi', 'stoch', 'momentum', 'osc'])
            ):
                if name not in groups['htf_momentum_feature']:
                    groups['htf_momentum_feature'].append(name)

            if (
                family == 'anchors'
                or any(key in base_feature for key in ['vwap', 'anchor'])
                or any(key in name_lower for key in ['vwap', 'anchor'])
            ):
                if name not in groups['htf_anchor_feature']:
                    groups['htf_anchor_feature'].append(name)

            regime_hint = None
            for key in ['regime', 'regime_type', 'dominant_regime', 'regime_label']:
                if key in metadata and metadata[key]:
                    regime_hint = metadata[key]
                    break

            if regime_hint is None:
                state_metadata = getattr(getattr(feature, 'state', None), 'metadata', {}) or {}
                for key in ['regime', 'regime_type', 'dominant_regime', 'regime_label']:
                    if key in state_metadata and state_metadata[key]:
                        regime_hint = state_metadata[key]
                        break

            if regime_hint is not None or metadata.get('is_regime_feature'):
                if name not in groups['htf_regime_feature']:
                    groups['htf_regime_feature'].append(name)

        tprint_debug(
            "Grouped HTF features by type | counts=%s"
            % {key: len(value) for key, value in groups.items()}
        )
        return groups
    
    def _group_htf_features_by_asset(self, materialized_htfs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Group HTF features by asset."""
        groups: Dict[str, Dict[str, Any]] = defaultdict(dict)

        for feature_name, feature in materialized_htfs.items():
            asset_name: Optional[str] = None

            metadata = getattr(feature, 'metadata', {}) or {}
            if isinstance(metadata, dict):
                asset_name = metadata.get('asset') or metadata.get('symbol')

            if not asset_name:
                separators = ['__', '::', '|']
                for sep in separators:
                    if sep in feature_name:
                        asset_name = feature_name.split(sep)[0]
                        break

            if not asset_name and '_' in feature_name:
                asset_name = feature_name.split('_')[0]

            asset_key = asset_name or 'asset1'
            groups[asset_key][feature_name] = feature

        grouped = dict(groups)
        tprint_debug(
            "Grouped HTF features by asset | assets=%d"
            % len(grouped)
        )
        return grouped
    
    def _generate_template_interactions(self, 
                                     template: InteractionTemplate,
                                     feature_groups: Dict[str, List[str]],
                                     targets: Optional[pd.Series]) -> List[GeneratedInteraction]:
        """Generate interactions for a specific template."""
        interactions = []
        
        # Find matching feature combinations
        required_features = template.required_features
        optional_features = template.optional_features
        
        # Generate combinations
        feature_combinations = self._generate_feature_combinations(
            required_features, optional_features, feature_groups
        )
        tprint_debug(
            "Evaluating template combinations | template=%s | candidates=%d"
            % (template.name, len(feature_combinations))
        )

        for combination in feature_combinations:
            if len(interactions) >= template.max_instances:
                break
            
            try:
                # Generate interaction
                interaction = self._create_interaction(
                    template, combination, targets
                )
                
                if interaction:
                    interactions.append(interaction)
                    
            except Exception as e:
                self.logger.warning(f"Failed to generate interaction {template.name}: {e}")
                continue

        tprint_debug(
            "Template evaluation complete | template=%s | accepted=%d"
            % (template.name, len(interactions))
        )
        return interactions
    
    def _generate_htf_template_interactions(self, 
                                          template: InteractionTemplate,
                                          htf_groups: Dict[str, List[str]],
                                          base_groups: Dict[str, List[str]],
                                          targets: Optional[pd.Series]) -> List[GeneratedInteraction]:
        """Generate HTF-aware interactions for a specific template."""
        interactions = []
        
        # Find matching feature combinations
        required_features = template.required_features
        
        # Generate combinations
        feature_combinations = self._generate_htf_feature_combinations(
            required_features, htf_groups, base_groups
        )
        tprint_debug(
            "Evaluating HTF template combinations | template=%s | candidates=%d"
            % (template.name, len(feature_combinations))
        )

        for combination in feature_combinations:
            if len(interactions) >= template.max_instances:
                break
            
            try:
                # Generate interaction
                interaction = self._create_interaction(
                    template, combination, targets
                )
                
                if interaction:
                    interactions.append(interaction)
                    
            except Exception as e:
                self.logger.warning(f"Failed to generate HTF interaction {template.name}: {e}")
                continue

        tprint_debug(
            "HTF template evaluation complete | template=%s | accepted=%d"
            % (template.name, len(interactions))
        )
        return interactions
    
    def _generate_cross_asset_template_interactions(self,
                                                  template: InteractionTemplate,
                                                  asset_groups: Dict[str, Dict[str, Any]],
                                                  targets: Optional[pd.Series]) -> List[GeneratedInteraction]:
        """Generate cross-asset interactions for a specific template."""
        interactions = []

        asset_items = [
            (asset, features)
            for asset, features in asset_groups.items()
            if features
        ]

        if len(asset_items) < 2:
            tprint_warning(
                "Cross-asset template skipped due to insufficient assets | available=%d"
                % len(asset_items)
            )
            return interactions

        lag_values: List[Optional[int]] = [None]
        if 'lag' in template.optional_features:
            configured_lags: Any = None
            if isinstance(self.config, dict):
                configured_lags = self.config.get('cross_asset_lags')
            else:
                configured_lags = getattr(self.config, 'cross_asset_lags', None)

            if configured_lags is None:
                configured_lags = template.metadata.get('lag_values') or template.metadata.get('lag')

            if configured_lags is None:
                lag_values = [1]
            elif isinstance(configured_lags, (list, tuple, set)):
                lag_values = [int(l) for l in configured_lags if l is not None]
            else:
                lag_values = [int(configured_lags)]

            if not lag_values:
                lag_values = [1]

        def _extract_series(feature_obj: Any) -> Optional[pd.Series]:
            if isinstance(feature_obj, pd.Series):
                return feature_obj
            return getattr(feature_obj, 'feature_series', None)

        seen_names = set()

        for (asset1, features1), (asset2, features2) in combinations(asset_items, 2):
            if len(interactions) >= template.max_instances:
                break

            for feature1_name, feature1 in features1.items():
                series1 = _extract_series(feature1)
                if series1 is None:
                    continue

                for feature2_name, feature2 in features2.items():
                    series2 = _extract_series(feature2)
                    if series2 is None:
                        continue

                    for lag_value in lag_values:
                        if len(interactions) >= template.max_instances:
                            break

                        placeholder_series: Dict[str, pd.Series] = {}
                        placeholder_names: Dict[str, str] = {}

                        for placeholder in template.required_features:
                            lower_placeholder = placeholder.lower()
                            if 'asset1' in lower_placeholder:
                                placeholder_series[placeholder] = series1
                                placeholder_names[placeholder] = feature1_name
                            elif 'asset2' in lower_placeholder:
                                placeholder_series[placeholder] = series2
                                placeholder_names[placeholder] = feature2_name
                            elif 'market' in lower_placeholder:
                                if 'market' in asset2.lower() or 'index' in asset2.lower():
                                    placeholder_series[placeholder] = series2
                                    placeholder_names[placeholder] = feature2_name
                                elif 'market' in asset1.lower() or 'index' in asset1.lower():
                                    placeholder_series[placeholder] = series1
                                    placeholder_names[placeholder] = feature1_name
                                else:
                                    placeholder_series[placeholder] = series2
                                    placeholder_names[placeholder] = feature2_name
                            else:
                                placeholder_series[placeholder] = series1
                                placeholder_names[placeholder] = feature1_name

                        if 'lag' in template.optional_features:
                            lag_int = int(lag_value)
                            lag_value = lag_int
                            placeholder_series['lag'] = lag_int
                            placeholder_names['lag'] = str(lag_int)

                        try:
                            evaluated_series = eval(
                                template.formula,
                                {'np': np, 'pd': pd},
                                placeholder_series
                            )
                        except Exception as exc:
                            self.logger.debug(
                                "Failed to evaluate cross-asset formula %s: %s",
                                template.name,
                                exc
                            )
                            continue

                        if isinstance(evaluated_series, pd.DataFrame):
                            evaluated_series = evaluated_series.iloc[:, 0]

                        if not isinstance(evaluated_series, pd.Series):
                            evaluated_series = pd.Series(evaluated_series)

                        interaction_series = evaluated_series.dropna()
                        if interaction_series.empty:
                            continue

                        name_parts = [template.name, feature1_name, feature2_name]
                        if 'lag' in template.optional_features:
                            name_parts.append(f"lag{lag_value}")

                        interaction_name = f"int_{'_'.join(name_parts)}"
                        if interaction_name in seen_names:
                            continue

                        seen_names.add(interaction_name)

                        formatted_formula = template.formula
                        for placeholder, replacement in placeholder_names.items():
                            formatted_formula = formatted_formula.replace(placeholder, replacement)

                        interaction_series = interaction_series.sort_index()
                        interaction_series.name = interaction_name

                        utility_score = 0.0
                        if targets is not None:
                            aligned = pd.concat(
                                [interaction_series, targets], axis=1, join='inner'
                            ).dropna()
                            if not aligned.empty:
                                corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
                                if not pd.isna(corr):
                                    utility_score = float(corr)

                        parent_features: List[str] = []
                        for placeholder in template.required_features:
                            feature_name = placeholder_names.get(placeholder)
                            if feature_name and feature_name not in parent_features:
                                parent_features.append(feature_name)

                        interaction_metadata = {
                            'template_name': template.name,
                            'template_type': template.template_type,
                            'priority': template.priority,
                            'asset1': asset1,
                            'asset2': asset2
                        }

                        if 'lag' in template.optional_features:
                            interaction_metadata['lag'] = lag_value

                        interactions.append(
                            GeneratedInteraction(
                                name=interaction_name,
                                formula=formatted_formula,
                                parent_features=parent_features,
                                interaction_type=template.template_type,
                                feature_series=interaction_series,
                                utility_score=utility_score,
                                metadata=interaction_metadata
                            )
                        )

                        if len(interactions) >= template.max_instances:
                            break

                    if len(interactions) >= template.max_instances:
                        break

                if len(interactions) >= template.max_instances:
                    break

        tprint_debug(
            "Cross-asset template evaluation complete | template=%s | accepted=%d"
            % (template.name, len(interactions))
        )
        return interactions
    
    def _generate_feature_combinations(self, 
                                     required_features: List[str],
                                     optional_features: List[str],
                                     feature_groups: Dict[str, List[str]]) -> List[Dict[str, str]]:
        """Generate feature combinations for a template."""
        combinations = []
        
        # Get required feature lists
        required_lists = [feature_groups.get(req, []) for req in required_features]
        
        # Generate Cartesian product
        for combo in product(*required_lists):
            combination = dict(zip(required_features, combo))

            # Add optional features if available
            for opt in optional_features:
                if opt in feature_groups and feature_groups[opt]:
                    combination[opt] = feature_groups[opt][0]  # Take first available
            
            combinations.append(combination)

        tprint_debug(
            "Feature combinations built | required=%s | optional=%s | total=%d"
            % (required_features, optional_features, len(combinations))
        )
        return combinations
    
    def _generate_htf_feature_combinations(self, 
                                         required_features: List[str],
                                         htf_groups: Dict[str, List[str]],
                                         base_groups: Dict[str, List[str]]) -> List[Dict[str, str]]:
        """Generate HTF feature combinations for a template."""
        combinations = []
        
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
            combinations.append(combination)

        tprint_debug(
            "HTF feature combinations built | required=%s | total=%d"
            % (required_features, len(combinations))
        )
        return combinations
    
    def _create_interaction(self, 
                          template: InteractionTemplate,
                          feature_combination: Dict[str, str],
                          targets: Optional[pd.Series]) -> Optional[GeneratedInteraction]:
        """Create a specific interaction from template and combination."""
        success = False
        try:
            # This is a simplified implementation
            # In practice, you'd evaluate the formula with actual feature data

            # Create interaction name
            interaction_name = f"int_{template.name}_{'_'.join(feature_combination.values())}"
            
            # Create formula string
            formula = template.formula
            for placeholder, feature_name in feature_combination.items():
                formula = formula.replace(placeholder, feature_name)
            
            # Create dummy feature series (in practice, you'd compute this)
            dummy_series = pd.Series(np.random.randn(1000), name=interaction_name)
            
            # Calculate utility score (simplified)
            utility_score = 0.0
            if targets is not None and len(dummy_series) == len(targets):
                utility_score = dummy_series.corr(targets)
                if pd.isna(utility_score):
                    utility_score = 0.0
            
            success = True
            return GeneratedInteraction(
                name=interaction_name,
                formula=formula,
                parent_features=list(feature_combination.values()),
                interaction_type=template.template_type,
                feature_series=dummy_series,
                utility_score=utility_score,
                metadata={
                    'template_name': template.name,
                    'template_type': template.template_type,
                    'priority': template.priority
                }
            )

        except Exception as e:
            self.logger.warning(f"Failed to create interaction: {e}")
            return None
        finally:
            tprint_debug(
                "Interaction instantiation attempted | template=%s | name=%s | success=%s"
                % (
                    template.name,
                    locals().get('interaction_name', 'unknown'),
                    success,
                )
            )
    
    def _apply_interaction_heredity(self, interactions: List[GeneratedInteraction]) -> List[GeneratedInteraction]:
        """Apply interaction heredity (keep ≥1 parent if interaction survives)."""
        # For now, return all interactions
        # In practice, you'd implement heredity rules
        tprint_info(
            "Interaction heredity applied | retained=%d"
            % len(interactions)
        )
        return interactions


class HTFInteractionTemplates:
    """Main HTF interaction templates system."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.interaction_generator = InteractionGenerator(config)
        tprint_info("HTFInteractionTemplates initialized")

    def generate_interactions(self,
                            materialized_htfs: Dict[str, Any],
                            base_features: Union[pd.DataFrame, Mapping[str, pd.Series], None],
                            targets: Optional[pd.Series] = None) -> List[GeneratedInteraction]:
        """
        Generate HTF-aware interactions.
        
        Args:
            materialized_htfs: Materialized HTF features
            base_features: Base features as a DataFrame or mapping
            targets: Target variables
            
        Returns:
            List of generated interactions
        """
        htf_feature_count = len(materialized_htfs) if hasattr(materialized_htfs, '__len__') else 0
        if isinstance(base_features, pd.DataFrame):
            base_feature_count = len(base_features.columns)
        elif hasattr(base_features, '__len__') and not isinstance(base_features, (pd.Series, pd.Index)):
            try:
                base_feature_count = len(base_features)
            except TypeError:
                base_feature_count = 0
        else:
            base_feature_count = 0

        tprint_info(
            "HTF interaction generation requested | htf=%d | base=%d"
            % (htf_feature_count, base_feature_count)
        )

        interactions = self.interaction_generator.generate_interactions(
            materialized_htfs, base_features, targets
        )
        tprint_success(
            "HTF interaction generation finished | produced=%d"
            % len(interactions)
        )
        return interactions

    def get_interaction_summary(self, interactions: List[GeneratedInteraction]) -> Dict[str, Any]:
        """Get summary of generated interactions."""
        type_counts = {}
        for interaction in interactions:
            interaction_type = interaction.interaction_type
            type_counts[interaction_type] = type_counts.get(interaction_type, 0) + 1

        avg_utility = np.mean([i.utility_score for i in interactions]) if interactions else 0.0

        summary = {
            'total_interactions': len(interactions),
            'type_counts': type_counts,
            'avg_utility': avg_utility,
            'interaction_names': [i.name for i in interactions]
        }
        tprint_info(
            "Generated interaction summary | total=%d | avg_utility=%.4f"
            % (summary['total_interactions'], summary['avg_utility'])
        )
        return summary


    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _vectorbt_apply_operation(self, data: pd.Series, func, 
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)
        
        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
