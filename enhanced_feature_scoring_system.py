#!/usr/bin/env python3
"""
Enhanced Feature Scoring System for Market Analysis Pipeline

This module provides an improved feature scoring system that addresses negative scores
by implementing multiple complementary scoring methods, regime-aware evaluation,
and robust score aggregation techniques.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
from datetime import datetime
import warnings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScoringMethod(Enum):
    """Available scoring methods for features."""
    MUTUAL_INFORMATION = "mutual_information"
    CORRELATION_ABS = "correlation_abs"
    VARIANCE_RATIO = "variance_ratio"
    IMPORTANCE_RANK = "importance_rank"
    STABILITY_SCORE = "stability_score"
    REGIME_CONSISTENCY = "regime_consistency"
    RISK_ADJUSTED_RETURN = "risk_adjusted_return"
    ENSEMBLE_VOTING = "ensemble_voting"

@dataclass
class EnhancedScoringConfig:
    """Configuration for enhanced feature scoring system."""
    # Primary scoring methods (weights sum to 1.0)
    scoring_methods: Dict[ScoringMethod, float] = field(default_factory=lambda: {
        ScoringMethod.MUTUAL_INFORMATION: 0.25,
        ScoringMethod.CORRELATION_ABS: 0.20,
        ScoringMethod.VARIANCE_RATIO: 0.15,
        ScoringMethod.IMPORTANCE_RANK: 0.20,
        ScoringMethod.STABILITY_SCORE: 0.10,
        ScoringMethod.REGIME_CONSISTENCY: 0.10
    })
    
    # Score normalization
    normalize_scores: bool = True
    normalization_method: str = "min_max"  # "min_max", "z_score", "rank", "sigmoid"
    score_floor: float = 0.0
    score_ceiling: float = 1.0
    
    # Ensemble configuration
    enable_ensemble: bool = True
    ensemble_methods: List[str] = field(default_factory=lambda: ["voting", "weighted_average", "rank_fusion"])
    
    # Regime-aware scoring
    enable_regime_scoring: bool = True
    regime_weight: float = 0.3
    
    # Stability analysis
    stability_window: int = 100  # Number of samples for stability calculation
    min_stability_threshold: float = 0.3
    
    # Risk adjustment
    enable_risk_adjustment: bool = True
    risk_free_rate: float = 0.02  # 2% annual risk-free rate
    
    # Quality filters
    min_sample_size: int = 50
    max_correlation_threshold: float = 0.95  # Remove highly correlated features
    
    # Output configuration
    top_k_features: int = 50
    save_intermediate_results: bool = True

class EnhancedFeatureScorer:
    """Enhanced feature scoring system with multiple methods and robust aggregation."""
    
    def __init__(self, config: Optional[EnhancedScoringConfig] = None):
        """Initialize the enhanced feature scorer."""
        self.config = config or EnhancedScoringConfig()
        self.logger = logger
        
        # Validate configuration
        self._validate_config()
        
        # Storage for intermediate results
        self.method_scores = {}
        self.ensemble_scores = {}
        self.final_scores = {}
        self.feature_metadata = {}
        
        self.logger.info("🚀 Enhanced Feature Scorer initialized")
        self.logger.info(f"   Scoring methods: {len(self.config.scoring_methods)}")
        self.logger.info(f"   Ensemble enabled: {self.config.enable_ensemble}")
        self.logger.info(f"   Regime-aware: {self.config.enable_regime_scoring}")
    
    def _validate_config(self):
        """Validate configuration parameters."""
        # Check scoring method weights sum to 1.0
        total_weight = sum(self.config.scoring_methods.values())
        if abs(total_weight - 1.0) > 0.01:
            self.logger.warning(f"Scoring method weights sum to {total_weight:.3f}, normalizing to 1.0")
            # Normalize weights
            for method in self.config.scoring_methods:
                self.config.scoring_methods[method] /= total_weight
        
        # Validate score bounds
        if self.config.score_floor >= self.config.score_ceiling:
            raise ValueError("Score floor must be less than score ceiling")
        
        self.logger.info("✅ Configuration validation passed")
    
    def calculate_mutual_information_score(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """Calculate mutual information scores for features."""
        self.logger.info("📊 Calculating mutual information scores")
        
        try:
            from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
            
            # Determine if classification or regression
            unique_targets = len(np.unique(y))
            is_classification = unique_targets < 20  # Heuristic
            
            if is_classification:
                mi_scores = mutual_info_classif(X, y, random_state=42)
            else:
                mi_scores = mutual_info_regression(X, y, random_state=42)
            
            # Convert to dictionary
            scores = dict(zip(feature_names, mi_scores))
            
            self.logger.info(f"   Calculated MI scores for {len(scores)} features")
            self.logger.info(f"   Score range: [{np.min(mi_scores):.4f}, {np.max(mi_scores):.4f}]")
            
            return scores
            
        except ImportError:
            self.logger.warning("Scikit-learn not available, using correlation fallback")
            return self.calculate_correlation_score(X, y, feature_names)
    
    def calculate_correlation_score(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """Calculate absolute correlation scores for features."""
        self.logger.info("📊 Calculating correlation scores")
        
        correlations = []
        for i in range(X.shape[1]):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            correlations.append(abs(corr) if not np.isnan(corr) else 0.0)
        
        scores = dict(zip(feature_names, correlations))
        
        self.logger.info(f"   Calculated correlation scores for {len(scores)} features")
        self.logger.info(f"   Score range: [{np.min(correlations):.4f}, {np.max(correlations):.4f}]")
        
        return scores
    
    def calculate_variance_ratio_score(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """Calculate variance ratio scores (signal-to-noise ratio)."""
        self.logger.info("📊 Calculating variance ratio scores")
        
        scores = {}
        
        for i, feature_name in enumerate(feature_names):
            feature_data = X[:, i]
            
            # Calculate variance of feature
            feature_var = np.var(feature_data)
            
            # Calculate noise estimate (residual variance after linear fit)
            if feature_var > 0:
                # Simple linear relationship with target
                correlation = np.corrcoef(feature_data, y)[0, 1]
                if not np.isnan(correlation):
                    explained_var = correlation**2 * feature_var
                    noise_var = feature_var - explained_var
                    signal_to_noise = explained_var / (noise_var + 1e-8)
                else:
                    signal_to_noise = 0.0
            else:
                signal_to_noise = 0.0
            
            scores[feature_name] = signal_to_noise
        
        # Normalize scores
        max_score = max(scores.values()) if scores else 1.0
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}
        
        self.logger.info(f"   Calculated variance ratio scores for {len(scores)} features")
        
        return scores
    
    def calculate_importance_rank_score(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """Calculate feature importance using tree-based models."""
        self.logger.info("📊 Calculating importance rank scores")
        
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            
            # Determine model type
            unique_targets = len(np.unique(y))
            is_classification = unique_targets < 20
            
            if is_classification:
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            
            # Fit model
            model.fit(X, y)
            
            # Get feature importances
            importances = model.feature_importances_
            scores = dict(zip(feature_names, importances))
            
            self.logger.info(f"   Calculated importance scores for {len(scores)} features")
            self.logger.info(f"   Score range: [{np.min(importances):.4f}, {np.max(importances):.4f}]")
            
            return scores
            
        except ImportError:
            self.logger.warning("Scikit-learn not available, using variance fallback")
            return self.calculate_variance_ratio_score(X, y, feature_names)
    
    def calculate_stability_score(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """Calculate stability scores for features across different time windows."""
        self.logger.info("📊 Calculating stability scores")
        
        scores = {}
        window_size = min(self.config.stability_window, len(X) // 3)
        
        if window_size < 20:
            # Not enough data for stability analysis
            return {name: 0.5 for name in feature_names}
        
        for i, feature_name in enumerate(feature_names):
            feature_data = X[:, i]
            
            # Calculate correlation stability across different windows
            correlations = []
            
            for start in range(0, len(X) - window_size, window_size // 2):
                end = start + window_size
                window_feature = feature_data[start:end]
                window_target = y[start:end]
                
                if len(window_feature) > 10:
                    corr = np.corrcoef(window_feature, window_target)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
            
            if correlations:
                # Stability is inverse of standard deviation of correlations
                stability = 1.0 / (1.0 + np.std(correlations))
            else:
                stability = 0.0
            
            scores[feature_name] = stability
        
        self.logger.info(f"   Calculated stability scores for {len(scores)} features")
        
        return scores
    
    def calculate_regime_consistency_score(self, X: np.ndarray, y: np.ndarray, feature_names: List[str], 
                                         regime_data: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate regime consistency scores for features."""
        self.logger.info("📊 Calculating regime consistency scores")
        
        if regime_data is None or not self.config.enable_regime_scoring:
            # Return neutral scores if no regime data
            return {name: 0.5 for name in feature_names}
        
        scores = {}
        unique_regimes = np.unique(regime_data)
        
        for i, feature_name in enumerate(feature_names):
            feature_data = X[:, i]
            regime_correlations = []
            
            # Calculate correlation in each regime
            for regime in unique_regimes:
                regime_mask = regime_data == regime
                if np.sum(regime_mask) > 10:  # Minimum samples per regime
                    regime_feature = feature_data[regime_mask]
                    regime_target = y[regime_mask]
                    
                    corr = np.corrcoef(regime_feature, regime_target)[0, 1]
                    if not np.isnan(corr):
                        regime_correlations.append(abs(corr))
            
            if regime_correlations:
                # Consistency is the mean correlation across regimes
                # weighted by the inverse of standard deviation
                mean_corr = np.mean(regime_correlations)
                std_corr = np.std(regime_correlations)
                consistency = mean_corr / (1.0 + std_corr)
            else:
                consistency = 0.0
            
            scores[feature_name] = consistency
        
        self.logger.info(f"   Calculated regime consistency scores for {len(scores)} features")
        
        return scores
    
    def normalize_scores(self, scores: Dict[str, float], method: str = "min_max") -> Dict[str, float]:
        """Normalize scores using the specified method."""
        if not scores:
            return scores
        
        score_values = np.array(list(scores.values()))
        feature_names = list(scores.keys())
        
        if method == "min_max":
            min_score = np.min(score_values)
            max_score = np.max(score_values)
            if max_score > min_score:
                normalized = (score_values - min_score) / (max_score - min_score)
            else:
                normalized = np.ones_like(score_values) * 0.5
        
        elif method == "z_score":
            if np.std(score_values) > 0:
                z_scores = (score_values - np.mean(score_values)) / np.std(score_values)
                # Shift to positive range
                normalized = (z_scores - np.min(z_scores)) / (np.max(z_scores) - np.min(z_scores))
            else:
                normalized = np.ones_like(score_values) * 0.5
        
        elif method == "rank":
            ranks = np.argsort(np.argsort(score_values)) + 1
            normalized = ranks / len(ranks)
        
        elif method == "sigmoid":
            normalized = 1 / (1 + np.exp(-score_values))
        
        else:
            self.logger.warning(f"Unknown normalization method: {method}, using min_max")
            return self.normalize_scores(scores, "min_max")
        
        # Apply floor and ceiling
        normalized = np.clip(normalized, self.config.score_floor, self.config.score_ceiling)
        
        return dict(zip(feature_names, normalized))
    
    def calculate_ensemble_scores(self, method_scores: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate ensemble scores from multiple scoring methods."""
        self.logger.info("🔀 Calculating ensemble scores")
        
        if not method_scores:
            return {}
        
        # Get all feature names
        all_features = set()
        for scores in method_scores.values():
            all_features.update(scores.keys())
        
        ensemble_scores = {}
        
        for feature in all_features:
            # Weighted average of normalized scores
            weighted_sum = 0.0
            total_weight = 0.0
            
            for method_name, scores in method_scores.items():
                if feature in scores:
                    # Convert method name string to enum for weight lookup
                    method_enum = None
                    for enum_method in self.config.scoring_methods:
                        if enum_method.value == method_name:
                            method_enum = enum_method
                            break
                    
                    if method_enum and method_enum in self.config.scoring_methods:
                        weight = self.config.scoring_methods[method_enum]
                        weighted_sum += scores[feature] * weight
                        total_weight += weight
            
            if total_weight > 0:
                ensemble_scores[feature] = weighted_sum / total_weight
            else:
                ensemble_scores[feature] = 0.0
        
        self.logger.info(f"   Calculated ensemble scores for {len(ensemble_scores)} features")
        
        return ensemble_scores
    
    def score_features(self, X: np.ndarray, y: np.ndarray, feature_names: List[str], 
                      regime_data: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Score features using multiple methods and ensemble aggregation.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target variable (n_samples,)
            feature_names: List of feature names
            regime_data: Optional regime labels
            
        Returns:
            Dictionary of feature scores
        """
        self.logger.info("🎯 Starting comprehensive feature scoring")
        self.logger.info(f"   Features: {len(feature_names)}")
        self.logger.info(f"   Samples: {X.shape[0]}")
        
        # Calculate scores using different methods
        method_scores = {}
        
        # Mutual Information
        if ScoringMethod.MUTUAL_INFORMATION in self.config.scoring_methods:
            mi_scores = self.calculate_mutual_information_score(X, y, feature_names)
            if self.config.normalize_scores:
                mi_scores = self.normalize_scores(mi_scores, self.config.normalization_method)
            method_scores["mutual_information"] = mi_scores
        
        # Correlation
        if ScoringMethod.CORRELATION_ABS in self.config.scoring_methods:
            corr_scores = self.calculate_correlation_score(X, y, feature_names)
            if self.config.normalize_scores:
                corr_scores = self.normalize_scores(corr_scores, self.config.normalization_method)
            method_scores["correlation_abs"] = corr_scores
        
        # Variance Ratio
        if ScoringMethod.VARIANCE_RATIO in self.config.scoring_methods:
            var_scores = self.calculate_variance_ratio_score(X, y, feature_names)
            if self.config.normalize_scores:
                var_scores = self.normalize_scores(var_scores, self.config.normalization_method)
            method_scores["variance_ratio"] = var_scores
        
        # Importance Rank
        if ScoringMethod.IMPORTANCE_RANK in self.config.scoring_methods:
            imp_scores = self.calculate_importance_rank_score(X, y, feature_names)
            if self.config.normalize_scores:
                imp_scores = self.normalize_scores(imp_scores, self.config.normalization_method)
            method_scores["importance_rank"] = imp_scores
        
        # Stability Score
        if ScoringMethod.STABILITY_SCORE in self.config.scoring_methods:
            stab_scores = self.calculate_stability_score(X, y, feature_names)
            if self.config.normalize_scores:
                stab_scores = self.normalize_scores(stab_scores, self.config.normalization_method)
            method_scores["stability_score"] = stab_scores
        
        # Regime Consistency
        if ScoringMethod.REGIME_CONSISTENCY in self.config.scoring_methods:
            regime_scores = self.calculate_regime_consistency_score(X, y, feature_names, regime_data)
            if self.config.normalize_scores:
                regime_scores = self.normalize_scores(regime_scores, self.config.normalization_method)
            method_scores["regime_consistency"] = regime_scores
        
        # Store intermediate results
        self.method_scores = method_scores
        
        # Calculate ensemble scores
        if self.config.enable_ensemble and len(method_scores) > 1:
            final_scores = self.calculate_ensemble_scores(method_scores)
        else:
            # Use the first available method
            final_scores = list(method_scores.values())[0] if method_scores else {}
        
        # Final normalization
        if final_scores and self.config.normalize_scores:
            final_scores = self.normalize_scores(final_scores, self.config.normalization_method)
        
        self.final_scores = final_scores
        
        # Log results
        if final_scores:
            score_values = list(final_scores.values())
            negative_count = sum(1 for score in score_values if score < 0)
            
            self.logger.info(f"✅ Feature scoring completed")
            self.logger.info(f"   Final scores: {len(final_scores)} features")
            self.logger.info(f"   Score range: [{np.min(score_values):.4f}, {np.max(score_values):.4f}]")
            self.logger.info(f"   Negative scores: {negative_count}/{len(final_scores)} ({negative_count/len(final_scores)*100:.1f}%)")
        
        return final_scores
    
    def get_top_features(self, scores: Optional[Dict[str, float]] = None, k: Optional[int] = None) -> List[Tuple[str, float]]:
        """Get top k features by score."""
        scores = scores or self.final_scores
        k = k or self.config.top_k_features
        
        if not scores:
            return []
        
        # Sort by score (descending)
        sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_features[:k]
    
    def generate_scoring_report(self) -> Dict[str, Any]:
        """Generate comprehensive scoring report."""
        report = {
            'summary': {
                'total_features': len(self.final_scores) if self.final_scores else 0,
                'scoring_methods_used': list(self.method_scores.keys()),
                'ensemble_enabled': self.config.enable_ensemble,
                'normalization_method': self.config.normalization_method
            },
            'score_statistics': {},
            'method_comparison': {},
            'top_features': self.get_top_features(),
            'recommendations': []
        }
        
        if self.final_scores:
            score_values = np.array(list(self.final_scores.values()))
            
            report['score_statistics'] = {
                'min': float(np.min(score_values)),
                'max': float(np.max(score_values)),
                'mean': float(np.mean(score_values)),
                'std': float(np.std(score_values)),
                'negative_count': int(np.sum(score_values < 0)),
                'negative_percentage': float(np.mean(score_values < 0) * 100)
            }
        
        # Method comparison
        if len(self.method_scores) > 1:
            method_stats = {}
            for method, scores in self.method_scores.items():
                if scores:
                    values = np.array(list(scores.values()))
                    method_stats[method] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'negative_count': int(np.sum(values < 0))
                    }
            report['method_comparison'] = method_stats
        
        # Generate recommendations
        recommendations = []
        
        if report['score_statistics'].get('negative_percentage', 0) == 0:
            recommendations.append("✅ All features have positive scores - scoring system working well")
        elif report['score_statistics'].get('negative_percentage', 0) < 5:
            recommendations.append("✅ Very few negative scores - minor adjustments may be beneficial")
        else:
            recommendations.append("⚠️ Consider additional normalization or different scoring methods")
        
        if report['score_statistics'].get('std', 0) < 0.1:
            recommendations.append("📊 Low score variance - consider rank-based selection")
        
        recommendations.extend([
            "🔄 Regularly retrain scoring models to adapt to market changes",
            "📈 Monitor feature stability across different market regimes",
            "🎯 Consider feature engineering for low-scoring features"
        ])
        
        report['recommendations'] = recommendations
        
        return report

def demonstrate_enhanced_scoring():
    """Demonstrate the enhanced feature scoring system."""
    logger.info("🚀 Demonstrating Enhanced Feature Scoring System")
    logger.info("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    n_samples, n_features = 1000, 20
    
    # Create features with different relationships to target
    X = np.random.randn(n_samples, n_features)
    
    # Create target with known relationships
    y = (0.5 * X[:, 0] +           # Strong positive
         -0.3 * X[:, 1] +          # Moderate negative
         0.8 * X[:, 2] +           # Very strong positive
         0.1 * X[:, 3] +           # Weak positive
         np.random.randn(n_samples) * 0.1)  # Noise
    
    feature_names = [f"feature_{i:02d}" for i in range(n_features)]
    
    # Create regime data
    regime_data = np.random.choice([0, 1, 2], size=n_samples)
    
    logger.info(f"📊 Generated sample data: {n_samples} samples, {n_features} features")
    
    # Initialize enhanced scorer
    config = EnhancedScoringConfig(
        normalize_scores=True,
        normalization_method="min_max",
        enable_ensemble=True,
        enable_regime_scoring=True,
        top_k_features=10
    )
    
    scorer = EnhancedFeatureScorer(config)
    
    # Score features
    scores = scorer.score_features(X, y, feature_names, regime_data)
    
    # Generate report
    report = scorer.generate_scoring_report()
    
    # Print results
    logger.info("\n📈 SCORING RESULTS")
    logger.info("=" * 40)
    
    logger.info(f"Total features scored: {report['summary']['total_features']}")
    logger.info(f"Methods used: {', '.join(report['summary']['scoring_methods_used'])}")
    logger.info(f"Negative scores: {report['score_statistics']['negative_count']}/{report['summary']['total_features']}")
    
    logger.info("\n🏆 Top 10 Features:")
    for i, (feature, score) in enumerate(report['top_features'][:10], 1):
        logger.info(f"   {i:2d}. {feature}: {score:.4f}")
    
    logger.info("\n💡 Recommendations:")
    for rec in report['recommendations']:
        logger.info(f"   {rec}")
    
    # Save results
    output_path = Path("/workspace/enhanced_feature_scoring_results.json")
    import json
    with open(output_path, 'w') as f:
        json.dump({
            'scores': scores,
            'report': report,
            'config': {
                'scoring_methods': {k.value: v for k, v in config.scoring_methods.items()},
                'normalization_method': config.normalization_method,
                'enable_ensemble': config.enable_ensemble
            }
        }, f, indent=2, default=str)
    
    logger.info(f"\n💾 Results saved to: {output_path}")

if __name__ == "__main__":
    demonstrate_enhanced_scoring()