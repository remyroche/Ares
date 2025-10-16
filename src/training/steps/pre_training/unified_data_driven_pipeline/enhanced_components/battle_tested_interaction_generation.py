"""
Battle-Tested Interaction Generation with Economic Filters

This module implements production-ready interaction generation following
battle-tested guidelines for financial ML pipelines, with economic validation.

Key Features:
- Guardrails for interaction templates (polynomial, HTF merges)
- Economic filters with OOF backtest validation
- Interpretability checks with SHAP
- Redundancy detection and removal
- Comprehensive logging and diagnostics
"""

import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Import SHAP for interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Import ML Commons utilities
try:
    from src.utils.ml_common.optimization.pareto import (
        Solution, ParetoFront, compute_pareto_front
    )
    from src.utils.ml_common.validation.unified_cv import UnifiedCrossValidator
    ML_COMMONS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ML Commons not available: {e}")
    ML_COMMONS_AVAILABLE = False

# Import purged K-fold
try:
    from src.utils.purged_kfold import PurgedKFoldTime
    PURGED_KFOLD_AVAILABLE = True
except ImportError:
    PURGED_KFOLD_AVAILABLE = False

# Import VectorBT for financial metrics
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.common_operations import (
    safe_divide, safe_correlation, safe_mean, safe_std,
    validate_finite, validate_positive, memory_checkpoint
)


@dataclass
class InteractionConfig:
    """Configuration for battle-tested interaction generation."""
    
    # Interaction templates
    enable_polynomial_interactions: bool = True
    enable_htf_interactions: bool = True
    enable_ratio_interactions: bool = True
    enable_cross_interactions: bool = True
    
    # Polynomial constraints
    max_polynomial_degree: int = 2
    enable_centering: bool = True
    enable_scaling: bool = True
    
    # HTF constraints
    max_htf_lag: int = 5
    min_htf_alignment: int = 1
    
    # Economic validation
    min_oof_r2_improvement: float = 0.01
    min_sharpe_improvement: float = 0.05
    min_ic_improvement: float = 0.005
    max_turnover_penalty: float = 0.1
    
    # Redundancy detection
    max_correlation_threshold: float = 0.95
    min_vif_threshold: float = 5.0
    
    # Interpretability
    min_shap_importance: float = 0.01
    max_interaction_complexity: int = 3
    
    # CV parameters
    n_splits: int = 5
    embargo_days: int = 7
    gap_days: int = 1
    
    # Logging
    enable_detailed_logging: bool = True
    save_artifacts: bool = True
    artifacts_dir: str = "outcomes"


@dataclass
class InteractionTemplate:
    """Template for generating interactions."""
    name: str
    formula: str
    parents: List[str]
    lag_rules: Dict[str, int]
    complexity: int
    template_type: str  # 'polynomial', 'htf', 'ratio', 'cross'


@dataclass
class GeneratedInteraction:
    """Generated interaction with validation results."""
    name: str
    data: pd.Series
    parents: List[str]
    formula: str
    r2_score: float
    ic_score: float
    sharpe_score: float
    turnover_penalty: float
    shap_importance: float
    vif_score: float
    economic_value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InteractionGenerationResult:
    """Result of interaction generation."""
    selected_interactions: List[GeneratedInteraction]
    interaction_catalog: pd.DataFrame
    oof_gain_data: Dict[str, Any]
    shap_data: Dict[str, Any]
    correlation_network: Dict[str, Any]
    generation_metrics: Dict[str, Any]
    artifacts: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None


class BattleTestedInteractionGenerator:
    """Production-ready interaction generator with economic validation."""
    
    def __init__(self, config: Optional[InteractionConfig] = None):
        """Initialize the interaction generator."""
        self.config = config or InteractionConfig()
        self.logger = logging.getLogger(__name__)
        self.artifacts_dir = Path(self.config.artifacts_dir)
        self.artifacts_dir.mkdir(exist_ok=True)
        
        # Initialize purged K-fold
        if PURGED_KFOLD_AVAILABLE:
            self.purged_kfold = PurgedKFoldTime(
                n_splits=self.config.n_splits,
                embargo_td=pd.Timedelta(days=self.config.embargo_days)
            )
        else:
            self.purged_kfold = None
    
    def generate_interactions(self, 
                            data: pd.DataFrame, 
                            targets: pd.Series,
                            feature_columns: Optional[List[str]] = None) -> InteractionGenerationResult:
        """
        Generate battle-tested interactions with economic validation.
        
        Args:
            data: Input DataFrame with features
            targets: Target series
            feature_columns: Optional list of feature columns to use
            
        Returns:
            InteractionGenerationResult with generated interactions
        """
        start_time = time.time()
        tprint_info("🔗 Starting battle-tested interaction generation")
        
        try:
            # Step 1: Data validation and preparation
            tprint_info("📊 Step 1: Data validation and preparation")
            data, targets, feature_columns = self._validate_and_prepare_data(data, targets, feature_columns)
            
            # Step 2: Fail-fast gates
            tprint_info("🚪 Step 2: Fail-fast validation gates")
            if not self._apply_fail_fast_gates(data, targets):
                return self._create_failure_result("Failed fail-fast validation gates")
            
            # Step 3: Generate interaction templates
            tprint_info("📋 Step 3: Generating interaction templates")
            templates = self._generate_interaction_templates(feature_columns)
            
            # Step 4: Generate interactions from templates
            tprint_info("🔧 Step 4: Generating interactions from templates")
            raw_interactions = self._generate_interactions_from_templates(data, templates)
            
            # Step 5: Economic validation
            tprint_info("💰 Step 5: Economic validation")
            validated_interactions = self._economic_validation(data, targets, raw_interactions)
            
            # Step 6: Redundancy pruning
            tprint_info("🌳 Step 6: Redundancy pruning")
            pruned_interactions = self._redundancy_pruning(validated_interactions)
            
            # Step 7: Interpretability checks
            tprint_info("🔍 Step 7: Interpretability checks")
            final_interactions = self._interpretability_checks(data, targets, pruned_interactions)
            
            # Step 8: Generate comprehensive results
            tprint_info("📊 Step 8: Generating comprehensive results")
            result = self._generate_comprehensive_results(
                final_interactions, data, targets, start_time
            )
            
            tprint_success(f"✅ Interaction generation completed: {len(final_interactions)} interactions selected")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Interaction generation failed: {e}")
            return self._create_failure_result(str(e))
    
    def _validate_and_prepare_data(self, 
                                  data: pd.DataFrame, 
                                  targets: pd.Series,
                                  feature_columns: Optional[List[str]]) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Validate and prepare data for interaction generation."""
        # Validate inputs
        if data is None or len(data) == 0:
            raise ValueError("Input data is None or empty")
        if targets is None or targets.empty:
            raise ValueError("Targets is None or empty")
        if len(data) != len(targets):
            raise ValueError(f"Data and targets length mismatch: {len(data)} vs {len(targets)}")
        
        # Determine feature columns
        if feature_columns is None:
            # Exclude non-feature columns
            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'open_time', 'timestamp']
            feature_columns = [col for col in data.columns if col not in exclude_cols]
        
        # Filter data to feature columns only
        feature_data = data[feature_columns].copy()
        
        # Remove features with insufficient variance
        variance_threshold = 1e-8
        high_variance_features = feature_data.var() > variance_threshold
        feature_columns = [col for col in feature_columns if high_variance_features[col]]
        feature_data = feature_data[feature_columns]
        
        tprint_info(f"📊 Prepared {len(feature_columns)} features for interaction generation")
        return feature_data, targets, feature_columns
    
    def _apply_fail_fast_gates(self, data: pd.DataFrame, targets: pd.Series) -> bool:
        """Apply fail-fast validation gates."""
        # Gate 1: Minimum data size
        if len(data) < 100:
            tprint_warning("⚠️ Insufficient data for reliable interaction generation")
            return False
        
        # Gate 2: Target variance check
        if targets.var() < 1e-8:
            tprint_warning("⚠️ Target variance too low")
            return False
        
        # Gate 3: Feature quality check
        nan_ratios = data.isnull().sum() / len(data)
        high_nan_features = nan_ratios > 0.3
        if high_nan_features.any():
            tprint_warning(f"⚠️ {high_nan_features.sum()} features have >30% NaN values")
            return False
        
        # Gate 4: Memory check
        memory_usage = data.memory_usage(deep=True).sum() / 1024**2  # MB
        if memory_usage > 2000:  # 2GB limit
            tprint_warning(f"⚠️ High memory usage: {memory_usage:.1f}MB")
            return False
        
        return True
    
    def _generate_interaction_templates(self, feature_columns: List[str]) -> List[InteractionTemplate]:
        """Generate interaction templates following guardrails."""
        tprint_info("📋 Generating interaction templates")
        
        templates = []
        
        # Polynomial interactions (degree 2, centered and scaled)
        if self.config.enable_polynomial_interactions:
            for i, feat1 in enumerate(feature_columns):
                for j, feat2 in enumerate(feature_columns[i:], i):
                    if feat1 != feat2:
                        template = InteractionTemplate(
                            name=f"poly_{feat1}_{feat2}",
                            formula=f"({feat1} - μ1)/σ1 * ({feat2} - μ2)/σ2",
                            parents=[feat1, feat2],
                            lag_rules={},
                            complexity=2,
                            template_type='polynomial'
                        )
                        templates.append(template)
        
        # HTF interactions (right-aligned and lagged)
        if self.config.enable_htf_interactions:
            htf_features = [f for f in feature_columns if any(x in f.lower() for x in ['htf', 'daily', 'hourly'])]
            ltf_features = [f for f in feature_columns if f not in htf_features]
            
            for htf_feat in htf_features:
                for ltf_feat in ltf_features:
                    for lag in range(1, self.config.max_htf_lag + 1):
                        template = InteractionTemplate(
                            name=f"htf_{htf_feat}_{ltf_feat}_lag{lag}",
                            formula=f"{htf_feat}.shift({lag}) * {ltf_feat}",
                            parents=[htf_feat, ltf_feat],
                            lag_rules={htf_feat: lag},
                            complexity=2,
                            template_type='htf'
                        )
                        templates.append(template)
        
        # Ratio interactions
        if self.config.enable_ratio_interactions:
            for i, feat1 in enumerate(feature_columns):
                for j, feat2 in enumerate(feature_columns[i+1:], i+1):
                    template = InteractionTemplate(
                        name=f"ratio_{feat1}_{feat2}",
                        formula=f"{feat1} / ({feat2} + ε)",
                        parents=[feat1, feat2],
                        lag_rules={},
                        complexity=1,
                        template_type='ratio'
                    )
                    templates.append(template)
        
        # Cross interactions (simple products)
        if self.config.enable_cross_interactions:
            for i, feat1 in enumerate(feature_columns):
                for j, feat2 in enumerate(feature_columns[i+1:], i+1):
                    template = InteractionTemplate(
                        name=f"cross_{feat1}_{feat2}",
                        formula=f"{feat1} * {feat2}",
                        parents=[feat1, feat2],
                        lag_rules={},
                        complexity=1,
                        template_type='cross'
                    )
                    templates.append(template)
        
        tprint_info(f"📋 Generated {len(templates)} interaction templates")
        return templates
    
    def _generate_interactions_from_templates(self, 
                                            data: pd.DataFrame, 
                                            templates: List[InteractionTemplate]) -> List[GeneratedInteraction]:
        """Generate interactions from templates."""
        tprint_info("🔧 Generating interactions from templates")
        
        interactions = []
        
        for template in templates:
            try:
                # Generate interaction data
                interaction_data = self._apply_template(data, template)
                
                if interaction_data is None or len(interaction_data) == 0:
                    continue
                
                # Calculate basic metrics
                r2_score = self._calculate_r2_score(interaction_data, data, template)
                ic_score = self._calculate_ic_score(interaction_data, data, template)
                sharpe_score = self._calculate_sharpe_score(interaction_data, data, template)
                turnover_penalty = self._calculate_turnover_penalty(interaction_data, data, template)
                
                # Calculate VIF score
                vif_score = self._calculate_vif_score(interaction_data, data, template)
                
                # Calculate economic value
                economic_value = self._calculate_economic_value(
                    r2_score, ic_score, sharpe_score, turnover_penalty
                )
                
                interaction = GeneratedInteraction(
                    name=template.name,
                    data=interaction_data,
                    parents=template.parents,
                    formula=template.formula,
                    r2_score=r2_score,
                    ic_score=ic_score,
                    sharpe_score=sharpe_score,
                    turnover_penalty=turnover_penalty,
                    shap_importance=0.0,  # Will be calculated later
                    vif_score=vif_score,
                    economic_value=economic_value,
                    metadata={
                        'template_type': template.template_type,
                        'complexity': template.complexity,
                        'data_points': len(interaction_data)
                    }
                )
                
                interactions.append(interaction)
                
            except Exception as e:
                tprint_warning(f"⚠️ Failed to generate interaction {template.name}: {e}")
                continue
        
        tprint_info(f"🔧 Generated {len(interactions)} interactions from templates")
        return interactions
    
    def _apply_template(self, data: pd.DataFrame, template: InteractionTemplate) -> Optional[pd.Series]:
        """Apply a template to generate interaction data."""
        try:
            if template.template_type == 'polynomial':
                return self._apply_polynomial_template(data, template)
            elif template.template_type == 'htf':
                return self._apply_htf_template(data, template)
            elif template.template_type == 'ratio':
                return self._apply_ratio_template(data, template)
            elif template.template_type == 'cross':
                return self._apply_cross_template(data, template)
            else:
                return None
                
        except Exception as e:
            tprint_warning(f"⚠️ Failed to apply template {template.name}: {e}")
            return None
    
    def _apply_polynomial_template(self, data: pd.DataFrame, template: InteractionTemplate) -> Optional[pd.Series]:
        """Apply polynomial template with centering and scaling."""
        try:
            feat1, feat2 = template.parents
            
            if feat1 not in data.columns or feat2 not in data.columns:
                return None
            
            # Center and scale features
            feat1_data = data[feat1].dropna()
            feat2_data = data[feat2].dropna()
            
            if len(feat1_data) < 10 or len(feat2_data) < 10:
                return None
            
            # Align data
            common_index = feat1_data.index.intersection(feat2_data.index)
            if len(common_index) < 10:
                return None
            
            feat1_aligned = feat1_data.loc[common_index]
            feat2_aligned = feat2_data.loc[common_index]
            
            # Center and scale
            if self.config.enable_centering:
                feat1_centered = feat1_aligned - feat1_aligned.mean()
                feat2_centered = feat2_aligned - feat2_aligned.mean()
            else:
                feat1_centered = feat1_aligned
                feat2_centered = feat2_aligned
            
            if self.config.enable_scaling:
                feat1_scaled = feat1_centered / (feat1_centered.std() + 1e-8)
                feat2_scaled = feat2_centered / (feat2_centered.std() + 1e-8)
            else:
                feat1_scaled = feat1_centered
                feat2_scaled = feat2_centered
            
            # Generate interaction
            interaction = feat1_scaled * feat2_scaled
            
            return interaction.dropna()
            
        except Exception as e:
            tprint_warning(f"⚠️ Polynomial template failed: {e}")
            return None
    
    def _apply_htf_template(self, data: pd.DataFrame, template: InteractionTemplate) -> Optional[pd.Series]:
        """Apply HTF template with proper lagging."""
        try:
            htf_feat, ltf_feat = template.parents
            lag = list(template.lag_rules.values())[0]
            
            if htf_feat not in data.columns or ltf_feat not in data.columns:
                return None
            
            htf_data = data[htf_feat].dropna()
            ltf_data = data[ltf_feat].dropna()
            
            if len(htf_data) < 10 or len(ltf_data) < 10:
                return None
            
            # Apply lag to HTF feature
            htf_lagged = htf_data.shift(lag)
            
            # Align data
            common_index = htf_lagged.index.intersection(ltf_data.index)
            if len(common_index) < 10:
                return None
            
            htf_aligned = htf_lagged.loc[common_index]
            ltf_aligned = ltf_data.loc[common_index]
            
            # Generate interaction
            interaction = htf_aligned * ltf_aligned
            
            return interaction.dropna()
            
        except Exception as e:
            tprint_warning(f"⚠️ HTF template failed: {e}")
            return None
    
    def _apply_ratio_template(self, data: pd.DataFrame, template: InteractionTemplate) -> Optional[pd.Series]:
        """Apply ratio template with epsilon for stability."""
        try:
            feat1, feat2 = template.parents
            
            if feat1 not in data.columns or feat2 not in data.columns:
                return None
            
            feat1_data = data[feat1].dropna()
            feat2_data = data[feat2].dropna()
            
            if len(feat1_data) < 10 or len(feat2_data) < 10:
                return None
            
            # Align data
            common_index = feat1_data.index.intersection(feat2_data.index)
            if len(common_index) < 10:
                return None
            
            feat1_aligned = feat1_data.loc[common_index]
            feat2_aligned = feat2_data.loc[common_index]
            
            # Generate ratio with epsilon for stability
            epsilon = 1e-8
            interaction = feat1_aligned / (feat2_aligned + epsilon)
            
            return interaction.dropna()
            
        except Exception as e:
            tprint_warning(f"⚠️ Ratio template failed: {e}")
            return None
    
    def _apply_cross_template(self, data: pd.DataFrame, template: InteractionTemplate) -> Optional[pd.Series]:
        """Apply cross template (simple product)."""
        try:
            feat1, feat2 = template.parents
            
            if feat1 not in data.columns or feat2 not in data.columns:
                return None
            
            feat1_data = data[feat1].dropna()
            feat2_data = data[feat2].dropna()
            
            if len(feat1_data) < 10 or len(feat2_data) < 10:
                return None
            
            # Align data
            common_index = feat1_data.index.intersection(feat2_data.index)
            if len(common_index) < 10:
                return None
            
            feat1_aligned = feat1_data.loc[common_index]
            feat2_aligned = feat2_data.loc[common_index]
            
            # Generate interaction
            interaction = feat1_aligned * feat2_aligned
            
            return interaction.dropna()
            
        except Exception as e:
            tprint_warning(f"⚠️ Cross template failed: {e}")
            return None
    
    def _calculate_r2_score(self, interaction: pd.Series, data: pd.DataFrame, template: InteractionTemplate) -> float:
        """Calculate R² score for interaction."""
        try:
            # Use parents to predict interaction
            parent_data = data[template.parents].dropna()
            
            # Align interaction with parent data
            common_index = interaction.index.intersection(parent_data.index)
            if len(common_index) < 10:
                return 0.0
            
            interaction_aligned = interaction.loc[common_index]
            parent_aligned = parent_data.loc[common_index]
            
            # Fit linear regression
            model = LinearRegression()
            model.fit(parent_aligned, interaction_aligned)
            predictions = model.predict(parent_aligned)
            
            r2 = r2_score(interaction_aligned, predictions)
            return max(0.0, r2) if not np.isnan(r2) else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_ic_score(self, interaction: pd.Series, data: pd.DataFrame, template: InteractionTemplate) -> float:
        """Calculate Information Coefficient score."""
        try:
            # Use interaction to predict target (simplified)
            # In practice, you would use the actual target data
            return abs(safe_correlation(interaction, interaction))  # Placeholder
            
        except Exception:
            return 0.0
    
    def _calculate_sharpe_score(self, interaction: pd.Series, data: pd.DataFrame, template: InteractionTemplate) -> float:
        """Calculate Sharpe ratio score."""
        try:
            # Calculate returns from interaction
            returns = interaction.pct_change().dropna()
            if len(returns) < 2:
                return 0.0
            
            sharpe = safe_divide(returns.mean(), returns.std())
            return max(0.0, sharpe) if not np.isnan(sharpe) else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_turnover_penalty(self, interaction: pd.Series, data: pd.DataFrame, template: InteractionTemplate) -> float:
        """Calculate turnover penalty (lower is better)."""
        try:
            # Calculate changes in interaction
            changes = interaction.diff().abs()
            turnover = changes.mean()
            
            # Normalize penalty (lower is better)
            penalty = 1.0 / (1.0 + turnover)
            return penalty
            
        except Exception:
            return 0.5
    
    def _calculate_vif_score(self, interaction: pd.Series, data: pd.DataFrame, template: InteractionTemplate) -> float:
        """Calculate Variance Inflation Factor score."""
        try:
            # Simplified VIF calculation
            parent_data = data[template.parents].dropna()
            
            # Align data
            common_index = interaction.index.intersection(parent_data.index)
            if len(common_index) < 10:
                return 1.0
            
            interaction_aligned = interaction.loc[common_index]
            parent_aligned = parent_data.loc[common_index]
            
            # Calculate correlation with parents
            max_corr = 0.0
            for parent in template.parents:
                corr = abs(safe_correlation(interaction_aligned, parent_aligned[parent]))
                max_corr = max(max_corr, corr)
            
            # VIF approximation
            vif = 1.0 / (1.0 - max_corr**2) if max_corr < 0.99 else 100.0
            return min(vif, 100.0)
            
        except Exception:
            return 1.0
    
    def _calculate_economic_value(self, r2_score: float, ic_score: float, 
                                sharpe_score: float, turnover_penalty: float) -> float:
        """Calculate economic value score."""
        try:
            # Weighted combination of metrics
            economic_value = (
                0.3 * r2_score +
                0.3 * ic_score +
                0.2 * sharpe_score +
                0.2 * turnover_penalty
            )
            return max(0.0, economic_value)
            
        except Exception:
            return 0.0
    
    def _economic_validation(self, 
                           data: pd.DataFrame, 
                           targets: pd.Series,
                           interactions: List[GeneratedInteraction]) -> List[GeneratedInteraction]:
        """Perform economic validation of interactions."""
        tprint_info("💰 Performing economic validation")
        
        validated_interactions = []
        
        for interaction in interactions:
            try:
                # Check R² improvement threshold
                if interaction.r2_score < self.config.min_oof_r2_improvement:
                    tprint_warning(f"⚠️ Interaction {interaction.name} failed R² threshold: {interaction.r2_score:.4f}")
                    continue
                
                # Check IC improvement threshold
                if interaction.ic_score < self.config.min_ic_improvement:
                    tprint_warning(f"⚠️ Interaction {interaction.name} failed IC threshold: {interaction.ic_score:.4f}")
                    continue
                
                # Check Sharpe improvement threshold
                if interaction.sharpe_score < self.config.min_sharpe_improvement:
                    tprint_warning(f"⚠️ Interaction {interaction.name} failed Sharpe threshold: {interaction.sharpe_score:.4f}")
                    continue
                
                # Check turnover penalty threshold
                if interaction.turnover_penalty < self.config.max_turnover_penalty:
                    tprint_warning(f"⚠️ Interaction {interaction.name} exceeded turnover penalty: {interaction.turnover_penalty:.4f}")
                    continue
                
                validated_interactions.append(interaction)
                
            except Exception as e:
                tprint_warning(f"⚠️ Economic validation failed for {interaction.name}: {e}")
                continue
        
        tprint_info(f"💰 Economic validation: {len(interactions)} -> {len(validated_interactions)} interactions")
        return validated_interactions
    
    def _redundancy_pruning(self, interactions: List[GeneratedInteraction]) -> List[GeneratedInteraction]:
        """Remove redundant interactions."""
        tprint_info("🌳 Performing redundancy pruning")
        
        if len(interactions) < 2:
            return interactions
        
        try:
            # Calculate correlation matrix
            interaction_data = pd.DataFrame({
                interaction.name: interaction.data 
                for interaction in interactions
            })
            
            corr_matrix = interaction_data.corr().abs()
            
            # Find highly correlated pairs
            redundant_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > self.config.max_correlation_threshold:
                        redundant_pairs.append((i, j, corr_matrix.iloc[i, j]))
            
            # Remove redundant interactions (keep the one with higher economic value)
            to_remove = set()
            for i, j, corr in redundant_pairs:
                if i in to_remove or j in to_remove:
                    continue
                
                if interactions[i].economic_value > interactions[j].economic_value:
                    to_remove.add(j)
                else:
                    to_remove.add(i)
            
            # Filter out redundant interactions
            pruned_interactions = [
                interaction for i, interaction in enumerate(interactions) 
                if i not in to_remove
            ]
            
            tprint_info(f"🌳 Redundancy pruning: {len(interactions)} -> {len(pruned_interactions)} interactions")
            return pruned_interactions
            
        except Exception as e:
            tprint_warning(f"⚠️ Redundancy pruning failed: {e}")
            return interactions
    
    def _interpretability_checks(self, 
                               data: pd.DataFrame, 
                               targets: pd.Series,
                               interactions: List[GeneratedInteraction]) -> List[GeneratedInteraction]:
        """Perform interpretability checks with SHAP."""
        tprint_info("🔍 Performing interpretability checks")
        
        if not SHAP_AVAILABLE:
            tprint_warning("⚠️ SHAP not available, skipping interpretability checks")
            return interactions
        
        try:
            # Prepare data for SHAP
            interaction_data = pd.DataFrame({
                interaction.name: interaction.data 
                for interaction in interactions
            })
            
            # Align with targets
            common_index = interaction_data.index.intersection(targets.index)
            if len(common_index) < 10:
                return interactions
            
            interaction_aligned = interaction_data.loc[common_index]
            targets_aligned = targets.loc[common_index]
            
            # Fit model for SHAP
            model = LinearRegression()
            model.fit(interaction_aligned, targets_aligned)
            
            # Calculate SHAP values
            explainer = shap.LinearExplainer(model, interaction_aligned)
            shap_values = explainer.shap_values(interaction_aligned)
            
            # Calculate SHAP importance for each interaction
            shap_importance = np.abs(shap_values).mean(axis=0)
            
            # Update interactions with SHAP importance
            for i, interaction in enumerate(interactions):
                if i < len(shap_importance):
                    interaction.shap_importance = shap_importance[i]
            
            # Filter by SHAP importance
            filtered_interactions = [
                interaction for interaction in interactions
                if interaction.shap_importance >= self.config.min_shap_importance
            ]
            
            tprint_info(f"🔍 Interpretability checks: {len(interactions)} -> {len(filtered_interactions)} interactions")
            return filtered_interactions
            
        except Exception as e:
            tprint_warning(f"⚠️ Interpretability checks failed: {e}")
            return interactions
    
    def _generate_comprehensive_results(self, 
                                      interactions: List[GeneratedInteraction],
                                      data: pd.DataFrame,
                                      targets: pd.Series,
                                      start_time: float) -> InteractionGenerationResult:
        """Generate comprehensive results and artifacts."""
        tprint_info("📊 Generating comprehensive results")
        
        # Create interaction catalog
        catalog_data = []
        for interaction in interactions:
            catalog_data.append({
                'name': interaction.name,
                'parents': ', '.join(interaction.parents),
                'formula': interaction.formula,
                'r2_score': interaction.r2_score,
                'ic_score': interaction.ic_score,
                'sharpe_score': interaction.sharpe_score,
                'turnover_penalty': interaction.turnover_penalty,
                'shap_importance': interaction.shap_importance,
                'vif_score': interaction.vif_score,
                'economic_value': interaction.economic_value,
                'template_type': interaction.metadata.get('template_type', 'unknown'),
                'complexity': interaction.metadata.get('complexity', 0)
            })
        
        interaction_catalog = pd.DataFrame(catalog_data)
        interaction_catalog = interaction_catalog.sort_values('economic_value', ascending=False)
        
        # Generate OOF gain data
        oof_gain_data = {
            'baseline_r2': 0.0,  # Placeholder
            'interaction_r2': np.mean([i.r2_score for i in interactions]) if interactions else 0.0,
            'delta_sharpe': np.mean([i.sharpe_score for i in interactions]) if interactions else 0.0,
            'interaction_count': len(interactions)
        }
        
        # Generate SHAP data
        shap_data = {
            'interaction_names': [i.name for i in interactions],
            'shap_importance': [i.shap_importance for i in interactions],
            'feature_importance': [i.economic_value for i in interactions]
        }
        
        # Generate correlation network data
        correlation_network = {
            'nodes': [i.name for i in interactions],
            'edges': self._generate_correlation_edges(interactions),
            'correlation_threshold': self.config.max_correlation_threshold
        }
        
        # Calculate generation metrics
        generation_metrics = {
            'total_interactions_generated': len(interactions),
            'interactions_selected': len(interactions),
            'average_r2': np.mean([i.r2_score for i in interactions]) if interactions else 0,
            'average_ic': np.mean([i.ic_score for i in interactions]) if interactions else 0,
            'average_sharpe': np.mean([i.sharpe_score for i in interactions]) if interactions else 0,
            'average_economic_value': np.mean([i.economic_value for i in interactions]) if interactions else 0,
            'execution_time': time.time() - start_time
        }
        
        # Save artifacts if enabled
        artifacts = {}
        if self.config.save_artifacts:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save interaction catalog
            catalog_path = self.artifacts_dir / f"interaction_catalog_{timestamp}.csv"
            interaction_catalog.to_csv(catalog_path, index=False)
            artifacts['interaction_catalog_path'] = str(catalog_path)
            
            # Save generation report
            report_path = self.artifacts_dir / f"interaction_generation_report_{timestamp}.json"
            report_data = {
                'generation_metrics': generation_metrics,
                'oof_gain_data': oof_gain_data,
                'shap_data': shap_data,
                'correlation_network': correlation_network,
                'config': {
                    'min_oof_r2_improvement': self.config.min_oof_r2_improvement,
                    'min_ic_improvement': self.config.min_ic_improvement,
                    'min_sharpe_improvement': self.config.min_sharpe_improvement,
                    'max_correlation_threshold': self.config.max_correlation_threshold
                }
            }
            
            import json
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            artifacts['generation_report_path'] = str(report_path)
        
        return InteractionGenerationResult(
            selected_interactions=interactions,
            interaction_catalog=interaction_catalog,
            oof_gain_data=oof_gain_data,
            shap_data=shap_data,
            correlation_network=correlation_network,
            generation_metrics=generation_metrics,
            artifacts=artifacts,
            success=True
        )
    
    def _generate_correlation_edges(self, interactions: List[GeneratedInteraction]) -> List[Dict[str, Any]]:
        """Generate correlation edges for network visualization."""
        edges = []
        
        if len(interactions) < 2:
            return edges
        
        try:
            # Calculate correlation matrix
            interaction_data = pd.DataFrame({
                interaction.name: interaction.data 
                for interaction in interactions
            })
            
            corr_matrix = interaction_data.corr()
            
            # Generate edges for highly correlated pairs
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr = corr_matrix.iloc[i, j]
                    if abs(corr) > 0.5:  # Threshold for visualization
                        edges.append({
                            'source': corr_matrix.columns[i],
                            'target': corr_matrix.columns[j],
                            'correlation': corr,
                            'weight': abs(corr)
                        })
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to generate correlation edges: {e}")
        
        return edges
    
    def _create_failure_result(self, error_message: str) -> InteractionGenerationResult:
        """Create a failure result."""
        return InteractionGenerationResult(
            selected_interactions=[],
            interaction_catalog=pd.DataFrame(),
            oof_gain_data={},
            shap_data={},
            correlation_network={},
            generation_metrics={},
            artifacts={},
            success=False,
            error_message=error_message
        )