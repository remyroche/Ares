"""
Interaction Pruning for Interactive Feature Generation

This module implements sophisticated interaction pruning that:
- Generates a whitelist of domain-plausible pairs
- Scores pairs with out-of-fold metrics (ΔIC vs parents, conditional ΔIC)
- Applies stability checks (mean/σ across folds, sign flips)
- Optional HSIC screening for nonlinear dependence
- Keeps ~3-6 best interactions per domain

Key Features:
- Domain-aware interaction generation
- Out-of-fold validation for robustness
- Stability-based pruning
- Nonlinear dependence detection
- Controlled interaction count
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
import logging
import time  # FIXED: Added missing time import
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import itertools
import warnings

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance
)

logger = logging.getLogger(__name__)


@dataclass
class InteractionPruningConfig:
    """Configuration for interaction pruning."""
    # Domain whitelist settings
    enable_domain_whitelist: bool = True
    max_interactions_per_domain: int = 6
    min_interactions_per_domain: int = 3
    
    # Scoring thresholds
    min_delta_ic: float = 0.01  # Minimum ΔIC vs parents
    min_conditional_delta_ic: float = 0.005  # Minimum conditional ΔIC
    min_stability_score: float = 0.7  # Minimum stability score
    
    # Stability checks
    max_sign_flips: int = 1  # Maximum sign flips across folds
    max_coefficient_variation: float = 0.5  # Max CV of coefficients
    
    # Nonlinear dependence
    enable_hsic_screening: bool = True
    hsic_threshold: float = 0.1  # Minimum HSIC score
    
    # Out-of-fold validation
    n_folds: int = 5
    purging_period: int = 5
    embargo_period: int = 2


@dataclass
class InteractionCandidate:
    """A candidate interaction feature."""
    feature1: str
    feature2: str
    interaction_type: str  # 'ratio', 'spread', 'product', 'min', 'max', etc.
    delta_ic: float
    conditional_delta_ic: float
    stability_score: float
    hsic_score: Optional[float] = None
    fold_scores: List[float] = field(default_factory=list)
    fold_coefficients: List[float] = field(default_factory=list)


@dataclass
class PruningResult:
    """Result of interaction pruning."""
    selected_interactions: List[InteractionCandidate]
    rejected_interactions: Dict[str, str]  # interaction_key -> reason
    domain_breakdown: Dict[str, List[InteractionCandidate]]
    pruning_stats: Dict[str, Any]
    performance_metrics: Dict[str, float]


class InteractionPruningSystem:
    """
    Interaction pruning system for interactive feature generation.
    
    Generates domain-plausible interactions and prunes them based on
    out-of-fold performance and stability metrics.
    """
    
    def __init__(self, config: Optional[InteractionPruningConfig] = None):
        """Initialize the interaction pruning system."""
        self.config = config or InteractionPruningConfig()
        self.scaler = StandardScaler()
        
        # Domain whitelist for plausible interactions
        self.domain_whitelist = self._create_domain_whitelist()
        
        tprint_info(f"🚀 Interaction pruning system initialized")
        tprint_info(f"📊 Max interactions per domain: {self.config.max_interactions_per_domain}")
        tprint_info(f"📊 Min ΔIC threshold: {self.config.min_delta_ic}")
        tprint_info(f"📊 Stability threshold: {self.config.min_stability_score}")
    
    def _create_domain_whitelist(self) -> Dict[str, List[Tuple[str, str, str]]]:
        """Create domain whitelist for plausible interaction types."""
        return {
            'price_volume': [
                ('price', 'volume', 'ratio'),
                ('price', 'volume', 'spread'),
                ('price', 'volume', 'product'),
            ],
            'momentum_volatility': [
                ('momentum', 'volatility', 'ratio'),
                ('momentum', 'volatility', 'spread'),
                ('momentum', 'volatility', 'product'),
            ],
            'trend_oscillator': [
                ('trend', 'oscillator', 'ratio'),
                ('trend', 'oscillator', 'spread'),
                ('trend', 'oscillator', 'product'),
            ],
            'time_series': [
                ('series1', 'series2', 'min'),
                ('series1', 'series2', 'max'),
                ('series1', 'series2', 'clamp'),
            ],
            'volatility_volume': [
                ('volatility', 'volume', 'ratio'),
                ('volatility', 'volume', 'spread'),
                ('volatility', 'volume', 'product'),
            ]
        }
    
    def _categorize_feature(self, feature_name: str) -> str:
        """Categorize a feature into a domain based on its name."""
        feature_lower = feature_name.lower()
        
        if any(term in feature_lower for term in ['price', 'close', 'open', 'high', 'low']):
            return 'price'
        elif any(term in feature_lower for term in ['volume', 'vol']):
            return 'volume'
        elif any(term in feature_lower for term in ['momentum', 'rsi', 'roc', 'stoch']):
            return 'momentum'
        elif any(term in feature_lower for term in ['volatility', 'atr', 'std', 'var']):
            return 'volatility'
        elif any(term in feature_lower for term in ['trend', 'sma', 'ema', 'macd']):
            return 'trend'
        elif any(term in feature_lower for term in ['oscillator', 'williams', 'cci']):
            return 'oscillator'
        else:
            return 'other'
    
    def generate_interaction_candidates(self, features: List[str]) -> List[InteractionCandidate]:
        """Generate interaction candidates based on domain whitelist."""
        tprint_debug("🔍 Generating interaction candidates...")
        
        candidates = []
        feature_categories = {f: self._categorize_feature(f) for f in features}
        
        # Generate candidates based on domain whitelist
        for domain, interaction_types in self.domain_whitelist.items():
            domain_features = [f for f, cat in feature_categories.items() 
                             if cat in [it[0] for it in interaction_types] or 
                                cat in [it[1] for it in interaction_types]]
            
            if len(domain_features) < 2:
                continue
            
            # Generate all pairs within domain
            for f1, f2 in itertools.combinations(domain_features, 2):
                for interaction_type in ['ratio', 'spread', 'product', 'min', 'max']:
                    candidate = InteractionCandidate(
                        feature1=f1,
                        feature2=f2,
                        interaction_type=interaction_type,
                        delta_ic=0.0,
                        conditional_delta_ic=0.0,
                        stability_score=0.0
                    )
                    candidates.append(candidate)
        
        # Also generate some cross-domain interactions
        all_features = list(feature_categories.keys())
        for f1, f2 in itertools.combinations(all_features, 2):
            cat1, cat2 = feature_categories[f1], feature_categories[f2]
            if cat1 != cat2:  # Cross-domain
                for interaction_type in ['ratio', 'spread']:
                    candidate = InteractionCandidate(
                        feature1=f1,
                        feature2=f2,
                        interaction_type=interaction_type,
                        delta_ic=0.0,
                        conditional_delta_ic=0.0,
                        stability_score=0.0
                    )
                    candidates.append(candidate)
        
        tprint_info(f"📊 Generated {len(candidates)} interaction candidates")
        return candidates
    
    def create_purged_folds(self, data: pd.DataFrame, n_folds: int = None) -> List[Tuple[int, int]]:
        """Create purged time-series folds for out-of-fold validation."""
        n_folds = n_folds or self.config.n_folds
        n_samples = len(data)
        fold_size = n_samples // n_folds
        
        folds = []
        for i in range(n_folds):
            start = i * fold_size
            end = min((i + 1) * fold_size, n_samples)
            
            # Apply purging and embargo
            purged_start = start + self.config.purging_period
            purged_end = end - self.config.embargo_period
            
            if purged_end > purged_start:
                folds.append((purged_start, purged_end))
        
        tprint_debug(f"📊 Created {len(folds)} purged folds")
        return folds
    
    def create_interaction_feature(self, data: pd.DataFrame, candidate: InteractionCandidate) -> pd.Series:
        """Create an interaction feature from a candidate."""
        f1_data = data[candidate.feature1].dropna()
        f2_data = data[candidate.feature2].dropna()
        
        # Align data
        common_index = f1_data.index.intersection(f2_data.index)
        if len(common_index) < 10:
            return pd.Series(dtype=float)
        
        f1_aligned = f1_data.loc[common_index]
        f2_aligned = f2_data.loc[common_index]
        
        # Create interaction based on type
        if candidate.interaction_type == 'ratio':
            # Avoid division by zero
            interaction = f1_aligned / (f2_aligned + 1e-8)
        elif candidate.interaction_type == 'spread':
            interaction = f1_aligned - f2_aligned
        elif candidate.interaction_type == 'product':
            interaction = f1_aligned * f2_aligned
        elif candidate.interaction_type == 'min':
            interaction = np.minimum(f1_aligned, f2_aligned)
        elif candidate.interaction_type == 'max':
            interaction = np.maximum(f1_aligned, f2_aligned)
        elif candidate.interaction_type == 'clamp':
            # Clamp f1 between f2 and f2*1.1
            interaction = np.clip(f1_aligned, f2_aligned, f2_aligned * 1.1)
        else:
            interaction = f1_aligned  # Fallback
        
        return interaction
    
    def calculate_delta_ic(self, data: pd.DataFrame, target: pd.Series, 
                          candidate: InteractionCandidate, purged_folds: List[Tuple[int, int]]) -> float:
        """Calculate ΔIC (improvement over parents) for an interaction."""
        # Create interaction feature
        interaction = self.create_interaction_feature(data, candidate)
        if len(interaction) < 10:
            return 0.0
        
        # Calculate IC for parents
        parent1_ic = self._calculate_ic(data[candidate.feature1], target, purged_folds)
        parent2_ic = self._calculate_ic(data[candidate.feature2], target, purged_folds)
        parent_ic = max(parent1_ic, parent2_ic)
        
        # Calculate IC for interaction
        interaction_ic = self._calculate_ic(interaction, target, purged_folds)
        
        # Return improvement
        return max(0.0, interaction_ic - parent_ic)
    
    def calculate_conditional_delta_ic(self, data: pd.DataFrame, target: pd.Series,
                                     candidate: InteractionCandidate, 
                                     purged_folds: List[Tuple[int, int]]) -> float:
        """Calculate conditional ΔIC (improvement given parents) for an interaction."""
        # Create interaction feature
        interaction = self.create_interaction_feature(data, candidate)
        if len(interaction) < 10:
            return 0.0
        
        # Align all data
        common_index = interaction.index.intersection(data[candidate.feature1].index)
        common_index = common_index.intersection(data[candidate.feature2].index)
        common_index = common_index.intersection(target.index)
        
        if len(common_index) < 10:
            return 0.0
        
        interaction_aligned = interaction.loc[common_index]
        parent1_aligned = data[candidate.feature1].loc[common_index]
        parent2_aligned = data[candidate.feature2].loc[common_index]
        target_aligned = target.loc[common_index]
        
        # Calculate conditional IC using residualization
        try:
            # Create feature matrix
            X = np.column_stack([parent1_aligned.values, parent2_aligned.values])
            y = target_aligned.values
            
            # Fit linear model to get residuals
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            residuals = y - y_pred
            
            # Calculate IC between interaction and residuals
            interaction_ic = np.corrcoef(interaction_aligned.values, residuals)[0, 1]
            
            return abs(interaction_ic) if not np.isnan(interaction_ic) else 0.0
            
        except Exception as e:
            tprint_debug(f"⚠️ Conditional ΔIC calculation failed: {e}")
            return 0.0
    
    def _calculate_ic(self, feature: pd.Series, target: pd.Series, 
                     purged_folds: List[Tuple[int, int]]) -> float:
        """Calculate Information Coefficient across purged folds."""
        fold_ics = []
        
        for start, end in purged_folds:
            if end <= len(feature):
                fold_feature = feature.iloc[start:end]
                fold_target = target.iloc[start:end]
                
                # Align data
                valid_idx = ~(fold_feature.isna() | fold_target.isna())
                if valid_idx.sum() > 5:
                    fold_feature_clean = fold_feature[valid_idx]
                    fold_target_clean = fold_target[valid_idx]
                    
                    try:
                        ic = np.corrcoef(fold_feature_clean, fold_target_clean)[0, 1]
                        if not np.isnan(ic):
                            fold_ics.append(abs(ic))
                    except:
                        continue
        
        return np.mean(fold_ics) if fold_ics else 0.0
    
    def calculate_stability_score(self, candidate: InteractionCandidate) -> float:
        """Calculate stability score based on fold consistency."""
        if not candidate.fold_scores:
            return 0.0
        
        scores = candidate.fold_scores
        
        # Check for sign flips
        sign_changes = sum(1 for i in range(1, len(scores)) 
                          if (scores[i] > 0) != (scores[i-1] > 0))
        sign_flip_penalty = min(1.0, sign_changes / len(scores))
        
        # Check coefficient of variation
        if len(scores) > 1:
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            cv = std_score / abs(mean_score) if mean_score != 0 else 1.0
            cv_penalty = min(1.0, cv / self.config.max_coefficient_variation)
        else:
            cv_penalty = 0.0
        
        # Calculate stability score
        stability = 1.0 - (sign_flip_penalty + cv_penalty) / 2.0
        return max(0.0, stability)
    
    def calculate_hsic_score(self, data: pd.DataFrame, candidate: InteractionCandidate) -> float:
        """Calculate HSIC score for nonlinear dependence detection."""
        if not self.config.enable_hsic_screening:
            return 1.0
        
        try:
            # Create interaction feature
            interaction = self.create_interaction_feature(data, candidate)
            if len(interaction) < 10:
                return 0.0
            
            # Align with parents
            common_index = interaction.index.intersection(data[candidate.feature1].index)
            common_index = common_index.intersection(data[candidate.feature2].index)
            
            if len(common_index) < 10:
                return 0.0
            
            interaction_aligned = interaction.loc[common_index]
            parent1_aligned = data[candidate.feature1].loc[common_index]
            parent2_aligned = data[candidate.feature2].loc[common_index]
            
            # Calculate HSIC (simplified version)
            # This is a placeholder - full HSIC implementation would be more complex
            from sklearn.metrics.pairwise import rbf_kernel
            
            # Create feature matrix
            X = np.column_stack([parent1_aligned.values, parent2_aligned.values])
            y = interaction_aligned.values.reshape(-1, 1)
            
            # Calculate kernel matrices
            Kx = rbf_kernel(X, gamma=1.0)
            Ky = rbf_kernel(y, gamma=1.0)
            
            # Simplified HSIC calculation
            n = len(X)
            H = np.eye(n) - np.ones((n, n)) / n
            hsic = np.trace(Kx @ H @ Ky @ H) / (n - 1) ** 2
            
            return hsic
            
        except Exception as e:
            tprint_debug(f"⚠️ HSIC calculation failed: {e}")
            return 0.0
    
    def evaluate_interaction_candidates(self, data: pd.DataFrame, target: pd.Series,
                                      candidates: List[InteractionCandidate]) -> List[InteractionCandidate]:
        """Evaluate interaction candidates with out-of-fold metrics."""
        tprint_debug("🔍 Evaluating interaction candidates...")
        
        purged_folds = self.create_purged_folds(data)
        
        for candidate in candidates:
            # Calculate ΔIC
            candidate.delta_ic = self.calculate_delta_ic(data, target, candidate, purged_folds)
            
            # Calculate conditional ΔIC
            candidate.conditional_delta_ic = self.calculate_conditional_delta_ic(
                data, target, candidate, purged_folds
            )
            
            # Calculate fold scores for stability
            candidate.fold_scores = []
            for start, end in purged_folds:
                if end <= len(data):
                    fold_data = data.iloc[start:end]
                    fold_target = target.iloc[start:end]
                    
                    fold_interaction = self.create_interaction_feature(fold_data, candidate)
                    if len(fold_interaction) > 5:
                        fold_ic = self._calculate_ic(fold_interaction, fold_target, [(0, len(fold_interaction))])
                        candidate.fold_scores.append(fold_ic)
            
            # Calculate stability score
            candidate.stability_score = self.calculate_stability_score(candidate)
            
            # Calculate HSIC score
            candidate.hsic_score = self.calculate_hsic_score(data, candidate)
        
        tprint_info(f"📊 Evaluated {len(candidates)} interaction candidates")
        return candidates
    
    def prune_interactions(self, candidates: List[InteractionCandidate]) -> PruningResult:
        """Prune interaction candidates based on performance and stability."""
        tprint_debug("🔍 Pruning interaction candidates...")
        
        rejected_interactions = {}
        selected_interactions = []
        
        # Group candidates by domain
        domain_candidates = defaultdict(list)
        for candidate in candidates:
            domain = f"{candidate.feature1}_{candidate.feature2}"
            domain_candidates[domain].append(candidate)
        
        # Prune within each domain
        for domain, domain_cands in domain_candidates.items():
            # Filter by thresholds
            valid_candidates = []
            for candidate in domain_cands:
                if (candidate.delta_ic >= self.config.min_delta_ic and
                    candidate.conditional_delta_ic >= self.config.min_conditional_delta_ic and
                    candidate.stability_score >= self.config.min_stability_score and
                    (candidate.hsic_score is None or candidate.hsic_score >= self.config.hsic_threshold)):
                    valid_candidates.append(candidate)
                else:
                    reason = []
                    if candidate.delta_ic < self.config.min_delta_ic:
                        reason.append(f"low_delta_ic_{candidate.delta_ic:.3f}")
                    if candidate.conditional_delta_ic < self.config.min_conditional_delta_ic:
                        reason.append(f"low_conditional_delta_ic_{candidate.conditional_delta_ic:.3f}")
                    if candidate.stability_score < self.config.min_stability_score:
                        reason.append(f"low_stability_{candidate.stability_score:.3f}")
                    if candidate.hsic_score is not None and candidate.hsic_score < self.config.hsic_threshold:
                        reason.append(f"low_hsic_{candidate.hsic_score:.3f}")
                    
                    key = f"{candidate.feature1}_{candidate.feature2}_{candidate.interaction_type}"
                    rejected_interactions[key] = "_".join(reason)
            
            # Sort by combined score and select top-k
            if valid_candidates:
                # Calculate combined score
                for candidate in valid_candidates:
                    combined_score = (candidate.delta_ic + 
                                    candidate.conditional_delta_ic + 
                                    candidate.stability_score) / 3.0
                    candidate.combined_score = combined_score
                
                # Sort by combined score
                valid_candidates.sort(key=lambda x: x.combined_score, reverse=True)
                
                # Select top-k
                top_k = valid_candidates[:self.config.max_interactions_per_domain]
                selected_interactions.extend(top_k)
                
                # Reject the rest
                for candidate in valid_candidates[self.config.max_interactions_per_domain:]:
                    key = f"{candidate.feature1}_{candidate.feature2}_{candidate.interaction_type}"
                    rejected_interactions[key] = "not_top_k"
        
        # Group selected interactions by domain
        domain_breakdown = defaultdict(list)
        for interaction in selected_interactions:
            domain = f"{interaction.feature1}_{interaction.feature2}"
            domain_breakdown[domain].append(interaction)
        
        # Calculate statistics
        pruning_stats = {
            'total_candidates': len(candidates),
            'selected_interactions': len(selected_interactions),
            'rejected_interactions': len(rejected_interactions),
            'domains': len(domain_breakdown),
            'average_delta_ic': np.mean([c.delta_ic for c in selected_interactions]) if selected_interactions else 0.0,
            'average_stability': np.mean([c.stability_score for c in selected_interactions]) if selected_interactions else 0.0
        }
        
        # Performance metrics
        performance_metrics = {
            'selection_rate': len(selected_interactions) / len(candidates) if candidates else 0.0,
            'domain_diversity': len(domain_breakdown),
            'average_quality': np.mean([c.combined_score for c in selected_interactions]) if selected_interactions else 0.0
        }
        
        result = PruningResult(
            selected_interactions=selected_interactions,
            rejected_interactions=rejected_interactions,
            domain_breakdown=dict(domain_breakdown),
            pruning_stats=pruning_stats,
            performance_metrics=performance_metrics
        )
        
        tprint_info(f"📊 Pruning completed: {len(selected_interactions)} selected, {len(rejected_interactions)} rejected")
        return result
    
    def prune_interactions_for_data(self, data: pd.DataFrame, target: pd.Series,
                                  features: List[str]) -> PruningResult:
        """Complete interaction pruning pipeline for given data."""
        tprint_success("🚀 Starting interaction pruning pipeline")
        start_time = time.time()
        
        # Generate candidates
        candidates = self.generate_interaction_candidates(features)
        
        # Evaluate candidates
        evaluated_candidates = self.evaluate_interaction_candidates(data, target, candidates)
        
        # Prune interactions
        result = self.prune_interactions(evaluated_candidates)
        
        execution_time = time.time() - start_time
        result.performance_metrics['execution_time'] = execution_time
        
        tprint_success(f"✅ Interaction pruning completed in {execution_time:.3f}s")
        tprint_info(f"📊 Selected {len(result.selected_interactions)} interactions")
        tprint_info(f"📊 Selection rate: {result.performance_metrics['selection_rate']:.1%}")
        
        return result


# Convenience functions

def create_interaction_pruning_system(config: Optional[InteractionPruningConfig] = None) -> InteractionPruningSystem:
    """Create an interaction pruning system with the given configuration."""
    return InteractionPruningSystem(config)


def prune_interactions(data: pd.DataFrame, target: pd.Series, features: List[str],
                      config: Optional[InteractionPruningConfig] = None) -> PruningResult:
    """Convenience function for interaction pruning."""
    system = create_interaction_pruning_system(config)
    return system.prune_interactions_for_data(data, target, features)


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 5000
    
    data = pd.DataFrame({
        'target': np.random.randn(n_samples).cumsum(),
        'price': np.random.randn(n_samples).cumsum(),
        'volume': np.random.randint(1000, 10000, n_samples),
        'momentum': np.random.randn(n_samples),
        'volatility': np.random.randn(n_samples).abs(),
        'trend': np.random.randn(n_samples).cumsum(),
        'oscillator': np.random.randn(n_samples),
    })
    
    # Test interaction pruning
    config = InteractionPruningConfig(
        max_interactions_per_domain=4,
        min_delta_ic=0.005,
        min_stability_score=0.6
    )
    
    features = ['price', 'volume', 'momentum', 'volatility', 'trend', 'oscillator']
    result = prune_interactions(data, data['target'], features, config)
    
    print(f"Selected interactions: {len(result.selected_interactions)}")
    for interaction in result.selected_interactions:
        print(f"  {interaction.feature1} {interaction.interaction_type} {interaction.feature2} "
              f"(ΔIC: {interaction.delta_ic:.3f}, Stability: {interaction.stability_score:.3f})")
    
    print(f"Pruning stats: {result.pruning_stats}")
    print(f"Performance metrics: {result.performance_metrics}")