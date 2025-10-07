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
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from itertools import combinations, product
import warnings
warnings.filterwarnings('ignore')


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
        
        return templates


class HTFAwareTemplates:
    """HTF-aware interaction templates."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.templates = self._create_htf_aware_templates()
    
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
        
        return templates


class CrossAssetTemplates:
    """Cross-asset HTF interaction templates."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.templates = self._create_cross_asset_templates()
    
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
        
        return templates


class InteractionGenerator:
    """Generates interactions from templates."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.core_templates = CoreInteractionTemplates()
        self.htf_aware_templates = HTFAwareTemplates()
        self.cross_asset_templates = CrossAssetTemplates()
    
    def generate_interactions(self, 
                            materialized_htfs: Dict[str, Any],
                            base_features: Dict[str, pd.Series],
                            targets: Optional[pd.Series] = None) -> List[GeneratedInteraction]:
        """
        Generate interactions from templates.
        
        Args:
            materialized_htfs: Materialized HTF features
            base_features: Base features
            targets: Target variables
            
        Returns:
            List of generated interactions
        """
        self.logger.info("Starting interaction generation")
        
        # Determine budget allocation
        budget_allocation = self._determine_budget_allocation(materialized_htfs)
        
        # Generate core interactions
        core_interactions = self._generate_core_interactions(
            base_features, targets, budget_allocation['core']
        )
        
        # Generate HTF-aware interactions
        htf_interactions = self._generate_htf_aware_interactions(
            materialized_htfs, base_features, targets, budget_allocation['htf_aware']
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
        
        return {
            'core': core_budget,
            'htf_aware': htf_aware_budget,
            'cross_asset': cross_asset_budget
        }
    
    def _generate_core_interactions(self, 
                                  base_features: Dict[str, pd.Series],
                                  targets: Optional[pd.Series],
                                  budget: int) -> List[GeneratedInteraction]:
        """Generate core interactions."""
        interactions = []
        
        # Group features by type
        feature_groups = self._group_features_by_type(base_features)
        
        # Generate interactions for each template
        for template in self.core_templates.templates:
            if len(interactions) >= budget:
                break
            
            template_interactions = self._generate_template_interactions(
                template, feature_groups, targets
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
        
        # Generate interactions for each template
        for template in self.htf_aware_templates.templates:
            if len(interactions) >= budget:
                break
            
            template_interactions = self._generate_htf_template_interactions(
                template, htf_groups, base_groups, targets
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
        
        # Generate interactions for each template
        for template in self.cross_asset_templates.templates:
            if len(interactions) >= budget:
                break
            
            template_interactions = self._generate_cross_asset_template_interactions(
                template, asset_groups, targets
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
            if any(x in name.lower() for x in ['price', 'close', 'open', 'high', 'low']):
                groups['price_feature'].append(name)
            elif any(x in name.lower() for x in ['vol', 'sigma', 'rv', 'gk']):
                groups['volatility_feature'].append(name)
            elif any(x in name.lower() for x in ['mom', 'momentum']):
                groups['momentum_feature'].append(name)
            elif any(x in name.lower() for x in ['rsi', 'stoch', 'mean_rev']):
                groups['mean_reversion_feature'].append(name)
            elif any(x in name.lower() for x in ['volume', 'liquidity']):
                groups['liquidity_feature'].append(name)
            elif 'volume' in name.lower():
                groups['volume_feature'].append(name)
            elif any(x in name.lower() for x in ['tod', 'time']):
                groups['tod_indicator'].append(name)
            elif any(x in name.lower() for x in ['regime', 'vol_regime']):
                groups['regime_indicator'].append(name)
        
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
            family = feature.family if hasattr(feature, 'family') else 'unknown'
            
            if family in ['trend_level_vol']:
                if 'trend' in name.lower() or 'ema' in name.lower():
                    groups['htf_trend_feature'].append(name)
                elif 'vol' in name.lower() or 'sigma' in name.lower():
                    groups['htf_volatility_feature'].append(name)
            elif family == 'oscillators':
                groups['htf_momentum_feature'].append(name)
            elif family == 'anchors':
                groups['htf_anchor_feature'].append(name)
        
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

        return dict(groups)
    
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
        
        return combinations
    
    def _create_interaction(self, 
                          template: InteractionTemplate,
                          feature_combination: Dict[str, str],
                          targets: Optional[pd.Series]) -> Optional[GeneratedInteraction]:
        """Create a specific interaction from template and combination."""
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
    
    def _apply_interaction_heredity(self, interactions: List[GeneratedInteraction]) -> List[GeneratedInteraction]:
        """Apply interaction heredity (keep ≥1 parent if interaction survives)."""
        # For now, return all interactions
        # In practice, you'd implement heredity rules
        return interactions


class HTFInteractionTemplates:
    """Main HTF interaction templates system."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.interaction_generator = InteractionGenerator(config)
    
    def generate_interactions(self, 
                            materialized_htfs: Dict[str, Any],
                            base_features: Dict[str, pd.Series],
                            targets: Optional[pd.Series] = None) -> List[GeneratedInteraction]:
        """
        Generate HTF-aware interactions.
        
        Args:
            materialized_htfs: Materialized HTF features
            base_features: Base features
            targets: Target variables
            
        Returns:
            List of generated interactions
        """
        return self.interaction_generator.generate_interactions(
            materialized_htfs, base_features, targets
        )
    
    def get_interaction_summary(self, interactions: List[GeneratedInteraction]) -> Dict[str, Any]:
        """Get summary of generated interactions."""
        type_counts = {}
        for interaction in interactions:
            interaction_type = interaction.interaction_type
            type_counts[interaction_type] = type_counts.get(interaction_type, 0) + 1
        
        avg_utility = np.mean([i.utility_score for i in interactions]) if interactions else 0.0
        
        return {
            'total_interactions': len(interactions),
            'type_counts': type_counts,
            'avg_utility': avg_utility,
            'interaction_names': [i.name for i in interactions]
        }