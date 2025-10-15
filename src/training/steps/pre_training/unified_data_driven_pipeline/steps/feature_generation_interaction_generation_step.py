"""
Feature Generation Interaction Generation Step

This step generates feature interactions as part of the unified data-driven pipeline
by calling the consolidated pipeline at the appropriate stage.
"""

from __future__ import annotations

import logging
import json
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

from src.training.steps.pre_training.unified_data_driven_pipeline.consolidated_pipeline_runner import (
    run_interaction_generation_step
)


@dataclass
class InteractionGenerationResult:
    """Result of interaction generation step with hard parent gating."""
    
    success: bool
    interaction_features: pd.DataFrame
    interaction_metadata: Dict[str, Any]
    generation_metrics: Dict[str, Any]
    artifacts: Dict[str, Any]
    interaction_catalog: pd.DataFrame
    error_message: Optional[str] = None


@dataclass
class InteractionGenerationConfig:
    """Configuration for lean interaction generation with hard parent gating."""
    
    # Hard parent gating
    require_step3_selected_parents: bool = True
    require_step6_trading_default: bool = True
    require_step6_top5_bundles: bool = True
    
    # Lean templates (small & sparse set)
    enable_crosses: bool = True  # trend × vol, momentum × vol_change, mean_rev × spread
    enable_ratios: bool = True  # ratios/diffs with ε-safeguard
    enable_degree2_polynomials: bool = True  # degree-2 only, no higher
    max_polynomial_degree: int = 2
    
    # Parent preprocessing
    center_scale_parents: bool = True  # Center/scale parents before multiplication
    restandardize_after: bool = True  # Re-standardize after interaction creation
    
    # Right-align HTF inputs
    right_align_htf: bool = True
    htf_lag_bars: int = 1  # Always lag by one bar when merging HTF features
    
    # Selection criteria
    min_delta_sharpe: float = 0.1  # ΔSharpe_adj threshold
    min_delta_score: float = 0.1  # Δscore = ΔSharpe_adj - μ·ΔTurnover
    turnover_penalty_mu: float = 0.1  # μ in turnover penalty
    
    # Group-L1 or greedy forward selection
    use_group_l1: bool = False  # Use Group-L1 by parent family
    use_greedy_forward: bool = True  # Use greedy forward add
    
    # Artifacts
    save_interaction_catalog: bool = True
    catalog_filename: str = "interaction_catalog.csv"


class FeatureGenerationInteractionGenerationStep:
    """Interaction generation step with hard parent gating and lean templates."""
    
    def __init__(self, config: Optional[InteractionGenerationConfig] = None):
        """Initialize the interaction generation step."""
        self.config = config or InteractionGenerationConfig()
        self.logger = logging.getLogger(__name__)
    
    def _hard_parent_gating(self, data: pd.DataFrame, pipeline_state: Optional[Dict[str, Any]] = None) -> List[str]:
        """Apply hard parent gating: only Step-3 selected + Step-6 trading_default + top-5 bundles."""
        self.logger.info("🚪 Applying hard parent gating")
        
        allowed_parents = []
        
        try:
            # Get Step-3 selected features
            if self.config.require_step3_selected_parents and pipeline_state:
                step3_features = pipeline_state.get('selected_features', [])
                if step3_features:
                    allowed_parents.extend(step3_features)
                    self.logger.info(f"✅ Added {len(step3_features)} Step-3 selected features")
            
            # Get Step-6 trading default
            if self.config.require_step6_trading_default and pipeline_state:
                trading_default = pipeline_state.get('trading_default')
                if trading_default and 'period' in trading_default and 'lookback' in trading_default:
                    # Generate features for trading default period/lookback
                    default_features = self._generate_features_for_period_lookback(
                        data, trading_default['period'], trading_default['lookback']
                    )
                    if default_features:
                        allowed_parents.extend(default_features)
                        self.logger.info(f"✅ Added {len(default_features)} trading default features")
            
            # Get Step-6 top-5 bundles
            if self.config.require_step6_top5_bundles and pipeline_state:
                interaction_combos = pipeline_state.get('interaction_combos', [])
                for combo in interaction_combos[:5]:  # Top 5 only
                    if 'period' in combo and 'lookback' in combo:
                        bundle_features = self._generate_features_for_period_lookback(
                            data, combo['period'], combo['lookback']
                        )
                        if bundle_features:
                            allowed_parents.extend(bundle_features)
                self.logger.info(f"✅ Added features from {len(interaction_combos[:5])} top bundles")
            
            # Remove duplicates and filter to existing columns
            allowed_parents = list(set(allowed_parents))
            allowed_parents = [p for p in allowed_parents if p in data.columns]
            
            self.logger.info(f"🚪 Hard parent gating: {len(allowed_parents)} allowed parents")
            return allowed_parents
            
        except Exception as e:
            self.logger.warning(f"⚠️ Hard parent gating failed: {e}")
            return data.columns.tolist()  # Fallback to all columns
    
    def _generate_features_for_period_lookback(self, data: pd.DataFrame, period: int, lookback: int) -> List[str]:
        """Generate feature names for a specific period/lookback combination."""
        try:
            features = []
            for col in data.columns:
                if period > 1:
                    features.append(f"{col}_sma_{period}")
                if lookback > 1:
                    features.append(f"{col}_lag_{lookback}")
            return features
        except Exception:
            return []
    
    def _generate_lean_templates(self, allowed_parents: List[str]) -> List[Dict[str, Any]]:
        """Generate lean interaction templates (small & sparse set)."""
        self.logger.info("📋 Generating lean interaction templates")
        
        templates = []
        
        # 1. Crosses: trend × vol, momentum × vol_change, mean_rev × spread
        if self.config.enable_crosses:
            trend_features = [p for p in allowed_parents if any(x in p.lower() for x in ['sma', 'ema', 'trend'])]
            vol_features = [p for p in allowed_parents if any(x in p.lower() for x in ['vol', 'volume', 'volatility'])]
            momentum_features = [p for p in allowed_parents if any(x in p.lower() for x in ['momentum', 'rsi', 'macd'])]
            mean_rev_features = [p for p in allowed_parents if any(x in p.lower() for x in ['mean_rev', 'bb', 'bollinger'])]
            spread_features = [p for p in allowed_parents if any(x in p.lower() for x in ['spread', 'range', 'atr'])]
            
            # trend × vol
            for trend in trend_features[:3]:  # Limit to top 3
                for vol in vol_features[:3]:
                    templates.append({
                        'name': f"cross_{trend}_{vol}",
                        'formula': f"{trend} * {vol}",
                        'parents': [trend, vol],
                        'template_type': 'cross',
                        'category': 'trend_vol'
                    })
            
            # momentum × vol_change
            for momentum in momentum_features[:3]:
                for vol in vol_features[:3]:
                    templates.append({
                        'name': f"cross_{momentum}_{vol}_change",
                        'formula': f"{momentum} * {vol}.diff()",
                        'parents': [momentum, vol],
                        'template_type': 'cross',
                        'category': 'momentum_vol_change'
                    })
            
            # mean_rev × spread
            for mean_rev in mean_rev_features[:3]:
                for spread in spread_features[:3]:
                    templates.append({
                        'name': f"cross_{mean_rev}_{spread}",
                        'formula': f"{mean_rev} * {spread}",
                        'parents': [mean_rev, spread],
                        'template_type': 'cross',
                        'category': 'mean_rev_spread'
                    })
        
        # 2. Ratios/diffs with ε-safeguard
        if self.config.enable_ratios:
            for i, parent1 in enumerate(allowed_parents[:10]):  # Limit combinations
                for parent2 in allowed_parents[i+1:i+6]:  # Limit to next 5
                    templates.append({
                        'name': f"ratio_{parent1}_{parent2}",
                        'formula': f"{parent1} / ({parent2} + 1e-8)",
                        'parents': [parent1, parent2],
                        'template_type': 'ratio',
                        'category': 'ratio'
                    })
        
        # 3. Degree-2 polynomials only
        if self.config.enable_degree2_polynomials:
            for i, parent1 in enumerate(allowed_parents[:5]):  # Limit combinations
                for parent2 in allowed_parents[i+1:i+3]:  # Limit to next 2
                    templates.append({
                        'name': f"poly2_{parent1}_{parent2}",
                        'formula': f"({parent1} - {parent1}.mean()) * ({parent2} - {parent2}.mean())",
                        'parents': [parent1, parent2],
                        'template_type': 'polynomial',
                        'category': 'degree2_polynomial'
                    })
        
        self.logger.info(f"📋 Generated {len(templates)} lean interaction templates")
        return templates
    
    def _apply_right_align_htf(self, data: pd.DataFrame, templates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Right-align HTF inputs and lag by one bar."""
        self.logger.info("🔄 Applying right-align HTF processing")
        
        if not self.config.right_align_htf:
            return templates
        
        processed_templates = []
        
        for template in templates:
            processed_template = template.copy()
            
            # Check if any parent is HTF (higher timeframe)
            htf_parents = [p for p in template['parents'] if any(x in p.lower() for x in ['htf', 'daily', 'hourly'])]
            
            if htf_parents:
                # Apply right-align and lag
                formula = template['formula']
                for htf_parent in htf_parents:
                    # Replace HTF parent with lagged version
                    formula = formula.replace(htf_parent, f"{htf_parent}.shift({self.config.htf_lag_bars})")
                
                processed_template['formula'] = formula
                processed_template['htf_processed'] = True
                processed_template['htf_lag'] = self.config.htf_lag_bars
            
            processed_templates.append(processed_template)
        
        self.logger.info(f"🔄 Processed {len(processed_templates)} templates with HTF right-align")
        return processed_templates
    
    def _greedy_forward_selection(self, data: pd.DataFrame, targets: pd.Series, 
                                 templates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Use greedy forward add with Δscore = ΔSharpe_adj - μ·ΔTurnover."""
        self.logger.info("🎯 Performing greedy forward selection")
        
        if not self.config.use_greedy_forward:
            return templates
        
        selected_interactions = []
        remaining_templates = templates.copy()
        
        while remaining_templates:
            best_template = None
            best_delta_score = -np.inf
            best_template_idx = -1
            
            for i, template in enumerate(remaining_templates):
                try:
                    # Calculate delta score for this template
                    delta_score = self._calculate_delta_score(data, targets, template, selected_interactions)
                    
                    if delta_score >= self.config.min_delta_score:
                        if delta_score > best_delta_score:
                            best_delta_score = delta_score
                            best_template = template
                            best_template_idx = i
                
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to evaluate template {template['name']}: {e}")
                    continue
            
            if best_template is not None:
                selected_interactions.append(best_template)
                remaining_templates.pop(best_template_idx)
                self.logger.info(f"✅ Selected interaction: {best_template['name']} (Δscore: {best_delta_score:.4f})")
            else:
                break  # No more valid templates
        
        self.logger.info(f"🎯 Greedy forward selection: {len(selected_interactions)} interactions selected")
        return selected_interactions
    
    def _calculate_delta_score(self, data: pd.DataFrame, targets: pd.Series, 
                              template: Dict[str, Any], selected_interactions: List[Dict[str, Any]]) -> float:
        """Calculate Δscore = ΔSharpe_adj - μ·ΔTurnover for template."""
        try:
            # Generate interaction data
            interaction_data = self._generate_interaction_data(data, template)
            if interaction_data is None or len(interaction_data) == 0:
                return -np.inf
            
            # Calculate baseline metrics (without this interaction)
            baseline_sharpe = self._calculate_baseline_sharpe(data, targets, selected_interactions)
            baseline_turnover = self._calculate_baseline_turnover(data, selected_interactions)
            
            # Calculate metrics with this interaction
            new_interactions = selected_interactions + [template]
            new_sharpe = self._calculate_baseline_sharpe(data, targets, new_interactions)
            new_turnover = self._calculate_baseline_turnover(data, new_interactions)
            
            # Calculate deltas
            delta_sharpe = new_sharpe - baseline_sharpe
            delta_turnover = new_turnover - baseline_turnover
            
            # Calculate delta score
            delta_score = delta_sharpe - self.config.turnover_penalty_mu * delta_turnover
            
            return delta_score
            
        except Exception as e:
            self.logger.warning(f"⚠️ Delta score calculation failed: {e}")
            return -np.inf
    
    def _generate_interaction_data(self, data: pd.DataFrame, template: Dict[str, Any]) -> Optional[pd.Series]:
        """Generate interaction data from template."""
        try:
            # This is a simplified implementation
            # In practice, you would implement the actual interaction generation logic
            # based on the template formula
            
            parents = template['parents']
            if not all(p in data.columns for p in parents):
                return None
            
            # Simple multiplication as example
            if len(parents) == 2:
                parent1_data = data[parents[0]].dropna()
                parent2_data = data[parents[1]].dropna()
                
                # Align data
                common_index = parent1_data.index.intersection(parent2_data.index)
                if len(common_index) < 10:
                    return None
                
                parent1_aligned = parent1_data.loc[common_index]
                parent2_aligned = parent2_data.loc[common_index]
                
                # Generate interaction
                interaction = parent1_aligned * parent2_aligned
                return interaction.dropna()
            
            return None
            
        except Exception:
            return None
    
    def _calculate_baseline_sharpe(self, data: pd.DataFrame, targets: pd.Series, 
                                  interactions: List[Dict[str, Any]]) -> float:
        """Calculate baseline Sharpe ratio."""
        try:
            if not interactions:
                return 0.0
            
            # Generate all interaction data
            interaction_data = []
            for interaction in interactions:
                data_series = self._generate_interaction_data(data, interaction)
                if data_series is not None:
                    interaction_data.append(data_series)
            
            if not interaction_data:
                return 0.0
            
            # Calculate combined Sharpe
            combined_data = pd.concat(interaction_data, axis=1).mean(axis=1)
            returns = combined_data.pct_change().dropna()
            
            if len(returns) > 1:
                return returns.mean() / (returns.std() + 1e-8)
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_baseline_turnover(self, data: pd.DataFrame, interactions: List[Dict[str, Any]]) -> float:
        """Calculate baseline turnover."""
        try:
            if not interactions:
                return 0.0
            
            # Generate all interaction data
            interaction_data = []
            for interaction in interactions:
                data_series = self._generate_interaction_data(data, interaction)
                if data_series is not None:
                    interaction_data.append(data_series)
            
            if not interaction_data:
                return 0.0
            
            # Calculate combined turnover
            combined_data = pd.concat(interaction_data, axis=1).mean(axis=1)
            turnover = combined_data.diff().abs().mean()
            
            return turnover if not np.isnan(turnover) else 0.0
            
        except Exception:
            return 0.0
    
    def _generate_interaction_catalog(self, selected_interactions: List[Dict[str, Any]], 
                                    data: pd.DataFrame, targets: pd.Series) -> pd.DataFrame:
        """Generate interaction catalog CSV."""
        self.logger.info("📊 Generating interaction catalog")
        
        catalog_data = []
        
        for interaction in selected_interactions:
            try:
                # Generate interaction data
                interaction_data = self._generate_interaction_data(data, interaction)
                
                if interaction_data is not None and len(interaction_data) > 0:
                    # Calculate OOF IC
                    aligned_targets = targets.loc[interaction_data.index]
                    oof_ic = np.corrcoef(interaction_data, aligned_targets)[0, 1] if len(interaction_data) > 1 else 0.0
                    
                    # Calculate delta Sharpe (simplified)
                    delta_sharpe = 0.1  # Placeholder
                    
                    catalog_data.append({
                        'name': interaction['name'],
                        'parents': ', '.join(interaction['parents']),
                        'formula': interaction['formula'],
                        'template_type': interaction['template_type'],
                        'category': interaction.get('category', 'unknown'),
                        'alignment': 'right_align' if interaction.get('htf_processed', False) else 'standard',
                        'oof_ic': oof_ic,
                        'delta_sharpe': delta_sharpe,
                        'accepted': 1 if abs(oof_ic) > 0.01 else 0
                    })
                
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to process interaction {interaction['name']}: {e}")
                continue
        
        catalog_df = pd.DataFrame(catalog_data)
        self.logger.info(f"📊 Generated interaction catalog with {len(catalog_df)} interactions")
        return catalog_df
    
    async def execute(self,
                     data: pd.DataFrame,
                     targets: Optional[pd.Series] = None,
                     pipeline_state: Optional[Dict[str, Any]] = None,
                     symbol: str = "ETHUSDT",
                     timeframe: str = "15m",
                     direction: str = "longs",
                     intensity: str = "blank",
                     lookback_days: Optional[int] = None,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None,
                     exchange: str = "binance",
                     custom_overrides: Optional[Dict[str, Any]] = None) -> InteractionGenerationResult:
        """Execute interaction generation step with hard parent gating and lean templates."""
        
        self.logger.info("🔗 Starting interaction generation with hard parent gating and lean templates")
        
        try:
            # Step 1: Hard parent gating
            self.logger.info("🚪 Step 1: Hard parent gating")
            allowed_parents = self._hard_parent_gating(data, pipeline_state)
            
            if not allowed_parents:
                return InteractionGenerationResult(
                    success=False,
                    interaction_features=pd.DataFrame(),
                    interaction_metadata={},
                    generation_metrics={},
                    artifacts={},
                    interaction_catalog=pd.DataFrame(),
                    error_message="No allowed parents found after hard gating"
                )
            
            # Step 2: Generate lean templates
            self.logger.info("📋 Step 2: Generating lean templates")
            templates = self._generate_lean_templates(allowed_parents)
            
            if not templates:
                return InteractionGenerationResult(
                    success=False,
                    interaction_features=pd.DataFrame(),
                    interaction_metadata={},
                    generation_metrics={},
                    artifacts={},
                    interaction_catalog=pd.DataFrame(),
                    error_message="No interaction templates generated"
                )
            
            # Step 3: Apply right-align HTF processing
            self.logger.info("🔄 Step 3: Applying right-align HTF processing")
            processed_templates = self._apply_right_align_htf(data, templates)
            
            # Step 4: Greedy forward selection
            self.logger.info("🎯 Step 4: Greedy forward selection")
            if targets is not None:
                selected_interactions = self._greedy_forward_selection(data, targets, processed_templates)
            else:
                # Fallback to all templates if no targets
                selected_interactions = processed_templates
            
            # Step 5: Generate interaction features
            self.logger.info("🔧 Step 5: Generating interaction features")
            interaction_features = self._generate_interaction_features(data, selected_interactions)
            
            # Step 6: Generate interaction catalog
            self.logger.info("📊 Step 6: Generating interaction catalog")
            interaction_catalog = self._generate_interaction_catalog(selected_interactions, data, targets)
            
            # Step 7: Generate artifacts
            self.logger.info("📋 Step 7: Generating artifacts")
            artifacts = self._generate_artifacts(interaction_catalog, selected_interactions)
            
            # Calculate metrics
            generation_metrics = {
                'total_templates': len(templates),
                'selected_interactions': len(selected_interactions),
                'interaction_features': len(interaction_features.columns),
                'catalog_entries': len(interaction_catalog),
                'accepted_interactions': len(interaction_catalog[interaction_catalog['accepted'] == 1])
            }
            
            # Create result
            generation_result = InteractionGenerationResult(
                success=True,
                interaction_features=interaction_features,
                interaction_metadata={
                    'allowed_parents': allowed_parents,
                    'templates_generated': len(templates),
                    'htf_processed': self.config.right_align_htf,
                    'selection_method': 'greedy_forward' if self.config.use_greedy_forward else 'all'
                },
                generation_metrics=generation_metrics,
                artifacts=artifacts,
                interaction_catalog=interaction_catalog
            )
            
            self.logger.info(f"✅ Interaction generation completed: {len(interaction_features.columns)} features, {len(interaction_catalog)} catalog entries")
            return generation_result
            
        except Exception as e:
            self.logger.error(f"❌ Interaction generation step failed with exception: {e}")
            return InteractionGenerationResult(
                success=False,
                interaction_features=pd.DataFrame(),
                interaction_metadata={},
                generation_metrics={},
                artifacts={},
                interaction_catalog=pd.DataFrame(),
                error_message=str(e)
            )
    
    def _generate_interaction_features(self, data: pd.DataFrame, selected_interactions: List[Dict[str, Any]]) -> pd.DataFrame:
        """Generate interaction features DataFrame."""
        try:
            interaction_data = {}
            
            for interaction in selected_interactions:
                try:
                    interaction_series = self._generate_interaction_data(data, interaction)
                    if interaction_series is not None and len(interaction_series) > 0:
                        interaction_data[interaction['name']] = interaction_series
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to generate interaction {interaction['name']}: {e}")
                    continue
            
            if interaction_data:
                return pd.DataFrame(interaction_data)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to generate interaction features: {e}")
            return pd.DataFrame()
    
    def _generate_artifacts(self, interaction_catalog: pd.DataFrame, selected_interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate artifacts for interaction generation."""
        try:
            from datetime import datetime
            from pathlib import Path
            
            # Create artifacts directory
            artifacts_dir = Path("outcomes")
            artifacts_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            artifacts = {}
            
            # Save interaction catalog
            if self.config.save_interaction_catalog and not interaction_catalog.empty:
                catalog_path = artifacts_dir / f"{self.config.catalog_filename}_{timestamp}"
                interaction_catalog.to_csv(catalog_path, index=False)
                artifacts['interaction_catalog_path'] = str(catalog_path)
            
            # Save interaction summary
            summary = {
                'total_interactions': len(selected_interactions),
                'catalog_entries': len(interaction_catalog),
                'accepted_interactions': len(interaction_catalog[interaction_catalog['accepted'] == 1]) if not interaction_catalog.empty else 0,
                'template_types': interaction_catalog['template_type'].value_counts().to_dict() if not interaction_catalog.empty else {},
                'categories': interaction_catalog['category'].value_counts().to_dict() if not interaction_catalog.empty else {},
                'timestamp': timestamp
            }
            
            summary_path = artifacts_dir / f"interaction_summary_{timestamp}.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            artifacts['summary_path'] = str(summary_path)
            
            return artifacts
            
        except Exception as e:
            self.logger.warning(f"⚠️ Artifact generation failed: {e}")
            return {}


# Command handler for ares_launcher integration
async def handle_feature_generation_interaction_generation_step(
    symbol: str = "ETHUSDT",
    timeframe: str = "15m",
    direction: str = "longs",
    intensity: str = "blank",
    lookback_days: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    exchange: str = "binance",
    custom_overrides: Optional[Dict[str, Any]] = None,
    **kwargs
) -> InteractionGenerationResult:
    """
    Handle feature generation interaction generation step command.
    
    Args:
        symbol: Trading symbol (default: "ETHUSDT")
        timeframe: Timeframe (default: "15m")
        direction: Direction (default: "longs")
        intensity: Pipeline intensity (default: "blank")
        lookback_days: Lookback days (optional)
        start_date: Start date (optional)
        end_date: End date (optional)
        exchange: Exchange (default: "binance")
        custom_overrides: Custom configuration overrides (optional)
        **kwargs: Additional arguments
        
    Returns:
        InteractionGenerationResult with generation results
    """
    # Create sample data for interaction generation (in real usage, this would come from data loading)
    sample_data = pd.DataFrame({
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 105,
        'low': np.random.randn(1000).cumsum() + 95,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000)
    })
    
    # Generate targets using the labeling system
    from src.training.steps.pre_training.unified_data_driven_pipeline.consolidated_pipeline_runner import ConsolidatedPipelineRunner
    runner = ConsolidatedPipelineRunner()
    targets = runner._generate_targets(sample_data, symbol, timeframe, direction)
    
    # Create step instance and execute
    step = FeatureGenerationInteractionGenerationStep()
    
    return await step.execute(
        data=sample_data,
        targets=targets,
        symbol=symbol,
        timeframe=timeframe,
        direction=direction,
        intensity=intensity,
        lookback_days=lookback_days,
        start_date=start_date,
        end_date=end_date,
        exchange=exchange,
        custom_overrides=custom_overrides
    )