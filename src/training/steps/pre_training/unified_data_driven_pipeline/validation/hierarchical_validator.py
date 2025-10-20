"""
Hierarchical Validation Strategy

This module implements a hierarchical validation approach to prevent
objective function collapse from repeated economic validation overuse.

Validation Hierarchy:
- Early steps → Statistical (signal/noise, entropy, IC)
- Mid steps → Hybrid (IC + correlation structure)  
- Late steps → Economic (Sharpe, turnover, stability)
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import mutual_info_regression, mutual_info_classif

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.common_operations import (
    safe_divide, safe_correlation, safe_mean, safe_std,
    validate_finite, validate_positive, memory_checkpoint
)


class ValidationStage(Enum):
    """Validation stages in the hierarchical approach."""
    EARLY = "early"      # Statistical validation
    MID = "mid"          # Hybrid validation
    LATE = "late"        # Economic validation


@dataclass
class HierarchicalValidationConfig:
    """Configuration for hierarchical validation."""
    
    # Stage-specific parameters
    early_stage_metrics: List[str] = field(default_factory=lambda: [
        'signal_noise_ratio', 'entropy', 'ic', 'correlation_structure'
    ])
    mid_stage_metrics: List[str] = field(default_factory=lambda: [
        'ic', 'correlation_structure', 'diversity', 'stability'
    ])
    late_stage_metrics: List[str] = field(default_factory=lambda: [
        'sharpe_ratio', 'turnover', 'stability', 'economic_validation'
    ])
    
    # Metric weights by stage
    early_weights: Dict[str, float] = field(default_factory=lambda: {
        'signal_noise_ratio': 0.3,
        'entropy': 0.2,
        'ic': 0.3,
        'correlation_structure': 0.2
    })
    mid_weights: Dict[str, float] = field(default_factory=lambda: {
        'ic': 0.4,
        'correlation_structure': 0.3,
        'diversity': 0.2,
        'stability': 0.1
    })
    late_weights: Dict[str, float] = field(default_factory=lambda: {
        'sharpe_ratio': 0.4,
        'turnover': 0.2,
        'stability': 0.2,
        'economic_validation': 0.2
    })
    
    # Thresholds
    min_signal_noise_ratio: float = 0.1
    min_entropy: float = 0.5
    min_ic: float = 0.01
    max_correlation: float = 0.85
    min_diversity: float = 0.3
    min_stability: float = 0.6
    min_sharpe: float = 0.1
    max_turnover: float = 2.0
    
    # Logging
    verbose: bool = True


@dataclass
class ValidationResult:
    """Result of hierarchical validation."""
    
    stage: ValidationStage
    metrics: Dict[str, float] = field(default_factory=dict)
    weighted_score: float = 0.0
    passed_thresholds: bool = True
    recommendations: List[str] = field(default_factory=list)
    
    # Stage-specific results
    early_results: Optional[Dict[str, float]] = None
    mid_results: Optional[Dict[str, float]] = None
    late_results: Optional[Dict[str, float]] = None


class HierarchicalValidator:
    """
    Hierarchical validation to prevent objective function collapse.
    
    Implements stage-specific validation:
    - Early: Statistical metrics (signal/noise, entropy, IC)
    - Mid: Hybrid metrics (IC + correlation structure)
    - Late: Economic metrics (Sharpe, turnover, stability)
    """
    
    def __init__(self, config: Optional[HierarchicalValidationConfig] = None):
        """Initialize the hierarchical validator."""
        self.config = config or HierarchicalValidationConfig()
        self.logger = logging.getLogger(__name__)
        
        if self.config.verbose:
            tprint("📊 Initializing HierarchicalValidator")
    
    def validate_early_stage(self, 
                           features: pd.DataFrame,
                           targets: pd.Series,
                           context: Dict[str, Any] = None) -> ValidationResult:
        """
        Early stage validation: Statistical metrics.
        
        Args:
            features: Input features
            targets: Target labels
            context: Additional context
            
        Returns:
            ValidationResult for early stage
        """
        if self.config.verbose:
            tprint("🔍 Early stage validation: Statistical metrics")
        
        result = ValidationResult(stage=ValidationStage.EARLY)
        
        # Signal-to-noise ratio
        signal_noise = self._calculate_signal_noise_ratio(features, targets)
        result.metrics['signal_noise_ratio'] = signal_noise
        
        # Entropy analysis
        entropy_score = self._calculate_entropy_score(features)
        result.metrics['entropy'] = entropy_score
        
        # Information Coefficient
        ic_score = self._calculate_ic_score(features, targets)
        result.metrics['ic'] = ic_score
        
        # Correlation structure
        corr_structure = self._calculate_correlation_structure(features)
        result.metrics['correlation_structure'] = corr_structure
        
        # Calculate weighted score
        result.weighted_score = self._calculate_weighted_score(
            result.metrics, self.config.early_weights
        )
        
        # Check thresholds
        result.passed_thresholds = self._check_early_thresholds(result.metrics)
        
        # Generate recommendations
        result.recommendations = self._generate_early_recommendations(result.metrics)
        
        result.early_results = result.metrics.copy()
        
        if self.config.verbose:
            tprint(f"📊 Early stage score: {result.weighted_score:.4f}")
            tprint(f"✅ Passed thresholds: {result.passed_thresholds}")
        
        return result
    
    def validate_mid_stage(self, 
                         features: pd.DataFrame,
                         targets: pd.Series,
                         early_result: ValidationResult,
                         context: Dict[str, Any] = None) -> ValidationResult:
        """
        Mid stage validation: Hybrid metrics.
        
        Args:
            features: Input features
            targets: Target labels
            early_result: Results from early stage
            context: Additional context
            
        Returns:
            ValidationResult for mid stage
        """
        if self.config.verbose:
            tprint("🔍 Mid stage validation: Hybrid metrics")
        
        result = ValidationResult(stage=ValidationStage.MID)
        
        # IC (inherited from early stage)
        result.metrics['ic'] = early_result.metrics.get('ic', 0.0)
        
        # Correlation structure (inherited from early stage)
        result.metrics['correlation_structure'] = early_result.metrics.get('correlation_structure', 0.0)
        
        # Diversity analysis
        diversity_score = self._calculate_diversity_score(features)
        result.metrics['diversity'] = diversity_score
        
        # Stability analysis
        stability_score = self._calculate_stability_score(features, targets)
        result.metrics['stability'] = stability_score
        
        # Calculate weighted score
        result.weighted_score = self._calculate_weighted_score(
            result.metrics, self.config.mid_weights
        )
        
        # Check thresholds
        result.passed_thresholds = self._check_mid_thresholds(result.metrics)
        
        # Generate recommendations
        result.recommendations = self._generate_mid_recommendations(result.metrics)
        
        result.mid_results = result.metrics.copy()
        
        if self.config.verbose:
            tprint(f"📊 Mid stage score: {result.weighted_score:.4f}")
            tprint(f"✅ Passed thresholds: {result.passed_thresholds}")
        
        return result
    
    def validate_late_stage(self, 
                          features: pd.DataFrame,
                          targets: pd.Series,
                          mid_result: ValidationResult,
                          context: Dict[str, Any] = None) -> ValidationResult:
        """
        Late stage validation: Economic metrics.
        
        Args:
            features: Input features
            targets: Target labels
            mid_result: Results from mid stage
            context: Additional context
            
        Returns:
            ValidationResult for late stage
        """
        if self.config.verbose:
            tprint("🔍 Late stage validation: Economic metrics")
        
        result = ValidationResult(stage=ValidationStage.LATE)
        
        # Sharpe ratio
        sharpe_score = self._calculate_sharpe_ratio(features, targets)
        result.metrics['sharpe_ratio'] = sharpe_score
        
        # Turnover analysis
        turnover_score = self._calculate_turnover_score(features)
        result.metrics['turnover'] = turnover_score
        
        # Stability (inherited from mid stage)
        result.metrics['stability'] = mid_result.metrics.get('stability', 0.0)
        
        # Economic validation
        economic_score = self._calculate_economic_validation(features, targets)
        result.metrics['economic_validation'] = economic_score
        
        # Calculate weighted score
        result.weighted_score = self._calculate_weighted_score(
            result.metrics, self.config.late_weights
        )
        
        # Check thresholds
        result.passed_thresholds = self._check_late_thresholds(result.metrics)
        
        # Generate recommendations
        result.recommendations = self._generate_late_recommendations(result.metrics)
        
        result.late_results = result.metrics.copy()
        
        if self.config.verbose:
            tprint(f"📊 Late stage score: {result.weighted_score:.4f}")
            tprint(f"✅ Passed thresholds: {result.passed_thresholds}")
        
        return result
    
    def _calculate_signal_noise_ratio(self, features: pd.DataFrame, targets: pd.Series) -> float:
        """Calculate signal-to-noise ratio."""
        try:
            # Calculate mutual information between features and targets
            mi_scores = []
            for col in features.columns:
                mi = mutual_info_regression(
                    features[[col]], targets, random_state=42
                )[0]
                mi_scores.append(mi)
            
            # Signal-to-noise ratio as mean MI / std MI
            mean_mi = np.mean(mi_scores)
            std_mi = np.std(mi_scores)
            return mean_mi / std_mi if std_mi > 0 else 0.0
        except:
            return 0.0
    
    def _calculate_entropy_score(self, features: pd.DataFrame) -> float:
        """Calculate entropy score for feature diversity."""
        try:
            # Calculate entropy for each feature
            entropy_scores = []
            for col in features.columns:
                # Discretize continuous values
                discretized = pd.cut(features[col], bins=10, labels=False)
                # Calculate entropy
                value_counts = discretized.value_counts()
                probabilities = value_counts / len(discretized)
                entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
                entropy_scores.append(entropy)
            
            return np.mean(entropy_scores)
        except:
            return 0.0
    
    def _calculate_ic_score(self, features: pd.DataFrame, targets: pd.Series) -> float:
        """Calculate Information Coefficient."""
        try:
            # Calculate IC for each feature
            ic_scores = []
            for col in features.columns:
                correlation = features[col].corr(targets)
                ic_scores.append(abs(correlation) if not np.isnan(correlation) else 0.0)
            
            return np.mean(ic_scores)
        except:
            return 0.0
    
    def _calculate_correlation_structure(self, features: pd.DataFrame) -> float:
        """Calculate correlation structure score."""
        try:
            # Calculate correlation matrix
            corr_matrix = features.corr()
            
            # Remove diagonal
            corr_matrix = corr_matrix.where(
                ~np.eye(corr_matrix.shape[0], dtype=bool), 0
            )
            
            # Calculate structure score (1 - mean absolute correlation)
            mean_corr = corr_matrix.abs().mean().mean()
            return 1.0 - mean_corr
        except:
            return 0.0
    
    def _calculate_diversity_score(self, features: pd.DataFrame) -> float:
        """Calculate diversity score."""
        try:
            # Calculate pairwise distances
            distances = pdist(features.T, metric='correlation')
            
            # Diversity as mean distance
            return np.mean(distances)
        except:
            return 0.0
    
    def _calculate_stability_score(self, features: pd.DataFrame, targets: pd.Series) -> float:
        """Calculate stability score."""
        try:
            # Rolling correlation stability
            window = min(20, len(features) // 4)
            if window < 5:
                return 0.0
            
            stability_scores = []
            for col in features.columns:
                rolling_corr = features[col].rolling(window).corr(targets)
                stability = 1.0 - rolling_corr.std() if not rolling_corr.std() is np.nan else 0.0
                stability_scores.append(stability)
            
            return np.mean(stability_scores)
        except:
            return 0.0
    
    def _calculate_sharpe_ratio(self, features: pd.DataFrame, targets: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        try:
            # Calculate returns
            returns = targets.pct_change().dropna()
            if len(returns) == 0:
                return 0.0
            
            # Sharpe ratio
            return returns.mean() / returns.std() if returns.std() > 0 else 0.0
        except:
            return 0.0
    
    def _calculate_turnover_score(self, features: pd.DataFrame) -> float:
        """Calculate turnover score."""
        try:
            # Calculate feature turnover (change in feature values)
            turnover_scores = []
            for col in features.columns:
                changes = features[col].pct_change().abs()
                turnover = changes.mean()
                turnover_scores.append(turnover)
            
            return np.mean(turnover_scores)
        except:
            return 0.0
    
    def _calculate_economic_validation(self, features: pd.DataFrame, targets: pd.Series) -> float:
        """Calculate economic validation score."""
        try:
            # Combine multiple economic metrics
            sharpe = self._calculate_sharpe_ratio(features, targets)
            turnover = self._calculate_turnover_score(features)
            stability = self._calculate_stability_score(features, targets)
            
            # Economic score as weighted combination
            economic_score = (sharpe * 0.5 + (1.0 - turnover) * 0.3 + stability * 0.2)
            return max(0.0, min(1.0, economic_score))
        except:
            return 0.0
    
    def _calculate_weighted_score(self, metrics: Dict[str, float], weights: Dict[str, float]) -> float:
        """Calculate weighted score."""
        try:
            weighted_sum = 0.0
            total_weight = 0.0
            
            for metric, value in metrics.items():
                weight = weights.get(metric, 0.0)
                weighted_sum += value * weight
                total_weight += weight
            
            return weighted_sum / total_weight if total_weight > 0 else 0.0
        except:
            return 0.0
    
    def _check_early_thresholds(self, metrics: Dict[str, float]) -> bool:
        """Check early stage thresholds."""
        return (
            metrics.get('signal_noise_ratio', 0.0) >= self.config.min_signal_noise_ratio and
            metrics.get('entropy', 0.0) >= self.config.min_entropy and
            metrics.get('ic', 0.0) >= self.config.min_ic and
            metrics.get('correlation_structure', 0.0) >= (1.0 - self.config.max_correlation)
        )
    
    def _check_mid_thresholds(self, metrics: Dict[str, float]) -> bool:
        """Check mid stage thresholds."""
        return (
            metrics.get('ic', 0.0) >= self.config.min_ic and
            metrics.get('correlation_structure', 0.0) >= (1.0 - self.config.max_correlation) and
            metrics.get('diversity', 0.0) >= self.config.min_diversity and
            metrics.get('stability', 0.0) >= self.config.min_stability
        )
    
    def _check_late_thresholds(self, metrics: Dict[str, float]) -> bool:
        """Check late stage thresholds."""
        return (
            metrics.get('sharpe_ratio', 0.0) >= self.config.min_sharpe and
            metrics.get('turnover', 0.0) <= self.config.max_turnover and
            metrics.get('stability', 0.0) >= self.config.min_stability and
            metrics.get('economic_validation', 0.0) >= 0.5
        )
    
    def _generate_early_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """Generate early stage recommendations."""
        recommendations = []
        
        if metrics.get('signal_noise_ratio', 0.0) < self.config.min_signal_noise_ratio:
            recommendations.append("Improve signal-to-noise ratio through feature selection")
        
        if metrics.get('entropy', 0.0) < self.config.min_entropy:
            recommendations.append("Increase feature entropy through diversification")
        
        if metrics.get('ic', 0.0) < self.config.min_ic:
            recommendations.append("Enhance Information Coefficient through better feature engineering")
        
        if metrics.get('correlation_structure', 0.0) < (1.0 - self.config.max_correlation):
            recommendations.append("Reduce feature correlation through decorrelation")
        
        return recommendations
    
    def _generate_mid_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """Generate mid stage recommendations."""
        recommendations = []
        
        if metrics.get('diversity', 0.0) < self.config.min_diversity:
            recommendations.append("Increase feature diversity through interaction generation")
        
        if metrics.get('stability', 0.0) < self.config.min_stability:
            recommendations.append("Improve stability through regularization")
        
        return recommendations
    
    def _generate_late_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """Generate late stage recommendations."""
        recommendations = []
        
        if metrics.get('sharpe_ratio', 0.0) < self.config.min_sharpe:
            recommendations.append("Improve Sharpe ratio through risk management")
        
        if metrics.get('turnover', 0.0) > self.config.max_turnover:
            recommendations.append("Reduce turnover through position sizing optimization")
        
        return recommendations
