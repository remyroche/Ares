#!/usr/bin/env python3
"""
Automated Feature Importance Analysis System

This module provides comprehensive feature importance analysis capabilities
for the trading system, including:
- Multiple importance calculation methods
- Regime-specific importance analysis
- Temporal stability analysis
- Automated feature ranking and selection
- Integration with existing feature selection tools
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from pathlib import Path
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

# Import system utilities
from ..logger import get_logger
from .matrix_operations import get_enhanced_matrix_operations

class ImportanceMethod(Enum):
    """Available feature importance methods."""
    RANDOM_FOREST = "random_forest"
    LASSO = "lasso"
    ELASTIC_NET = "elastic_net"
    RIDGE = "ridge"
    MUTUAL_INFO = "mutual_information"
    F_SCORE = "f_score"
    PERMUTATION = "permutation"
    SHAP = "shap"
    CORRELATION = "correlation"
    VARIANCE = "variance"

@dataclass
class FeatureImportanceConfig:
    """Configuration for feature importance analysis."""
    # Methods to use
    methods: List[ImportanceMethod] = field(default_factory=lambda: [
        ImportanceMethod.RANDOM_FOREST,
        ImportanceMethod.LASSO,
        ImportanceMethod.MUTUAL_INFO,
        ImportanceMethod.PERMUTATION
    ])
    
    # Model parameters
    random_forest_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42
    })
    
    lasso_params: Dict[str, Any] = field(default_factory=lambda: {
        'alpha': 0.01,
        'random_state': 42
    })
    
    elastic_net_params: Dict[str, Any] = field(default_factory=lambda: {
        'alpha': 0.01,
        'l1_ratio': 0.5,
        'random_state': 42
    })
    
    # Analysis parameters
    top_k_features: int = 20
    min_importance_threshold: float = 0.01
    stability_threshold: float = 0.7
    temporal_window: int = 1000
    
    # Performance settings
    n_jobs: int = -1
    chunk_size: int = 10000
    enable_parallel: bool = True
    
    # Output settings
    save_results: bool = True
    generate_plots: bool = True
    output_directory: Optional[str] = None

@dataclass
class FeatureImportanceResult:
    """Result of feature importance analysis."""
    feature_names: List[str]
    importance_scores: Dict[str, np.ndarray]
    method_scores: Dict[str, Dict[str, float]]
    stability_scores: Dict[str, float]
    temporal_stability: Dict[str, np.ndarray]
    rankings: Dict[str, List[str]]
    meta_info: Dict[str, Any]
    
    def get_top_features(self, method: str = "ensemble", k: int = 10) -> List[str]:
        """Get top k features for a specific method."""
        if method == "ensemble":
            # Average rankings across methods
            ensemble_scores = {}
            for method_name, scores in self.method_scores.items():
                for feature, score in scores.items():
                    if feature not in ensemble_scores:
                        ensemble_scores[feature] = []
                    ensemble_scores[feature].append(score)
            
            # Calculate average scores
            avg_scores = {feature: np.mean(scores) for feature, scores in ensemble_scores.items()}
            sorted_features = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
            return [feature for feature, _ in sorted_features[:k]]
        else:
            return self.rankings.get(method, [])[:k]

class FeatureImportanceAnalyzer:
    """Automated feature importance analyzer."""
    
    def __init__(self, config: Optional[FeatureImportanceConfig] = None):
        self.config = config or FeatureImportanceConfig()
        self.logger = get_logger("FeatureImportanceAnalyzer")
        
        # Initialize matrix operations for performance
        self.matrix_ops = get_enhanced_matrix_operations()
        
        # Results storage
        self.results: Dict[str, FeatureImportanceResult] = {}
        
        self.logger.info("🚀 FeatureImportanceAnalyzer initialized")
    
    def analyze_features(self, 
                        X: pd.DataFrame, 
                        y: pd.Series,
                        regime_labels: Optional[pd.Series] = None,
                        feature_names: Optional[List[str]] = None) -> FeatureImportanceResult:
        """Perform comprehensive feature importance analysis."""
        
        start_time = time.time()
        self.logger.info(f"🔍 Starting feature importance analysis on {X.shape[1]} features")
        
        if feature_names is None:
            feature_names = X.columns.tolist()
        
        # Initialize results structure
        importance_scores = {}
        method_scores = {}
        temporal_stability = {}
        
        # Perform analysis for each method
        for method in self.config.methods:
            self.logger.info(f"📊 Computing importance using {method.value}")
            method_start = time.time()
            
            try:
                scores = self._compute_importance(X, y, method)
                importance_scores[method.value] = scores
                
                # Convert to feature-scores dictionary
                feature_scores = dict(zip(feature_names, scores))
                method_scores[method.value] = feature_scores
                
                method_time = time.time() - method_start
                self.logger.info(f"✅ {method.value} completed in {method_time:.3f}s")
                
            except Exception as e:
                self.logger.error(f"❌ Error computing {method.value}: {e}")
                continue
        
        # Compute stability scores
        stability_scores = self._compute_stability_scores(method_scores)
        
        # Compute temporal stability if regime labels available
        if regime_labels is not None:
            temporal_stability = self._compute_temporal_stability(X, y, regime_labels, feature_names)
        
        # Generate rankings
        rankings = self._generate_rankings(method_scores)
        
        # Create result object
        result = FeatureImportanceResult(
            feature_names=feature_names,
            importance_scores=importance_scores,
            method_scores=method_scores,
            stability_scores=stability_scores,
            temporal_stability=temporal_stability,
            rankings=rankings,
            meta_info={
                'analysis_time': time.time() - start_time,
                'n_features': len(feature_names),
                'n_samples': len(X),
                'methods_used': [m.value for m in self.config.methods],
                'config': self.config.__dict__
            }
        )
        
        # Save results if configured
        if self.config.save_results:
            self._save_results(result)
        
        # Generate plots if configured
        if self.config.generate_plots:
            self._generate_plots(result)
        
        total_time = time.time() - start_time
        self.logger.info(f"✅ Feature importance analysis completed in {total_time:.3f}s")
        
        return result
    
    def _compute_importance(self, X: pd.DataFrame, y: pd.Series, method: ImportanceMethod) -> np.ndarray:
        """Compute feature importance using specified method."""
        
        if method == ImportanceMethod.RANDOM_FOREST:
            return self._random_forest_importance(X, y)
        
        elif method == ImportanceMethod.LASSO:
            return self._lasso_importance(X, y)
        
        elif method == ImportanceMethod.ELASTIC_NET:
            return self._elastic_net_importance(X, y)
        
        elif method == ImportanceMethod.RIDGE:
            return self._ridge_importance(X, y)
        
        elif method == ImportanceMethod.MUTUAL_INFO:
            return self._mutual_info_importance(X, y)
        
        elif method == ImportanceMethod.F_SCORE:
            return self._f_score_importance(X, y)
        
        elif method == ImportanceMethod.PERMUTATION:
            return self._permutation_importance(X, y)
        
        elif method == ImportanceMethod.CORRELATION:
            return self._correlation_importance(X, y)
        
        elif method == ImportanceMethod.VARIANCE:
            return self._variance_importance(X)
        
        else:
            raise ValueError(f"Unknown importance method: {method}")
    
    def _random_forest_importance(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Compute Random Forest feature importance."""
        # Determine if classification or regression
        is_classification = len(y.unique()) < 20 and y.dtype in ['object', 'category', 'int64']
        
        if is_classification:
            model = RandomForestClassifier(**self.config.random_forest_params)
        else:
            model = RandomForestRegressor(**self.config.random_forest_params)
        
        model.fit(X, y)
        return model.feature_importances_
    
    def _lasso_importance(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Compute Lasso feature importance."""
        model = Lasso(**self.config.lasso_params)
        model.fit(X, y)
        return np.abs(model.coef_)
    
    def _elastic_net_importance(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Compute Elastic Net feature importance."""
        model = ElasticNet(**self.config.elastic_net_params)
        model.fit(X, y)
        return np.abs(model.coef_)
    
    def _ridge_importance(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Compute Ridge feature importance."""
        model = Ridge(**self.config.ridge_params)
        model.fit(X, y)
        return np.abs(model.coef_)
    
    def _mutual_info_importance(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Compute mutual information importance."""
        return mutual_info_regression(X, y, random_state=42)
    
    def _f_score_importance(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Compute F-score importance."""
        selector = SelectKBest(f_regression, k='all')
        selector.fit(X, y)
        return selector.scores_
    
    def _permutation_importance(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Compute permutation importance."""
        # Use Random Forest as base model
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        # Compute permutation importance
        perm_importance = permutation_importance(model, X, y, n_repeats=5, random_state=42)
        return perm_importance.importances_mean
    
    def _correlation_importance(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Compute correlation-based importance."""
        correlations = X.corrwith(y).abs()
        return correlations.fillna(0).values
    
    def _variance_importance(self, X: pd.DataFrame) -> np.ndarray:
        """Compute variance-based importance."""
        return X.var().values
    
    def _compute_stability_scores(self, method_scores: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Compute stability scores across methods."""
        if len(method_scores) < 2:
            return {}
        
        stability_scores = {}
        features = list(next(iter(method_scores.values())).keys())
        
        for feature in features:
            scores = []
            for method, scores_dict in method_scores.items():
                if feature in scores_dict:
                    scores.append(scores_dict[feature])
            
            if len(scores) > 1:
                # Compute coefficient of variation (lower is more stable)
                cv = np.std(scores) / (np.mean(scores) + 1e-8)
                stability_scores[feature] = 1 / (1 + cv)  # Convert to stability score (higher is better)
            else:
                stability_scores[feature] = 0.0
        
        return stability_scores
    
    def _compute_temporal_stability(self, X: pd.DataFrame, y: pd.Series, 
                                  regime_labels: pd.Series, feature_names: List[str]) -> Dict[str, np.ndarray]:
        """Compute temporal stability of feature importance across regimes."""
        temporal_stability = {}
        
        for feature in feature_names:
            if feature in X.columns:
                feature_values = X[feature].values
                regime_correlations = []
                
                for regime in regime_labels.unique():
                    regime_mask = regime_labels == regime
                    if regime_mask.sum() > 10:  # Minimum samples
                        regime_corr = np.corrcoef(feature_values[regime_mask], y[regime_mask])[0, 1]
                        if not np.isnan(regime_corr):
                            regime_correlations.append(regime_corr)
                
                if len(regime_correlations) > 1:
                    temporal_stability[feature] = np.array(regime_correlations)
        
        return temporal_stability
    
    def _generate_rankings(self, method_scores: Dict[str, Dict[str, float]]) -> Dict[str, List[str]]:
        """Generate feature rankings for each method."""
        rankings = {}
        
        for method, scores in method_scores.items():
            sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            rankings[method] = [feature for feature, _ in sorted_features]
        
        # Create ensemble ranking
        if len(method_scores) > 1:
            ensemble_scores = {}
            for method, scores in method_scores.items():
                for feature, score in scores.items():
                    if feature not in ensemble_scores:
                        ensemble_scores[feature] = []
                    ensemble_scores[feature].append(score)
            
            # Average scores
            avg_scores = {feature: np.mean(scores) for feature, scores in ensemble_scores.items()}
            sorted_features = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
            rankings['ensemble'] = [feature for feature, _ in sorted_features]
        
        return rankings
    
    def _save_results(self, result: FeatureImportanceResult):
        """Save analysis results."""
        if self.config.output_directory:
            output_dir = Path(self.config.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save detailed results
            results_file = output_dir / f"feature_importance_{int(time.time())}.json"
            
            # Convert numpy arrays to lists for JSON serialization
            serializable_result = {
                'feature_names': result.feature_names,
                'method_scores': result.method_scores,
                'stability_scores': result.stability_scores,
                'rankings': result.rankings,
                'meta_info': result.meta_info
            }
            
            import json
            with open(results_file, 'w') as f:
                json.dump(serializable_result, f, indent=2)
            
            self.logger.info(f"💾 Results saved to {results_file}")
    
    def _generate_plots(self, result: FeatureImportanceResult):
        """Generate visualization plots."""
        if self.config.output_directory:
            output_dir = Path(self.config.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Plot 1: Feature importance comparison
            self._plot_importance_comparison(result, output_dir)
            
            # Plot 2: Stability analysis
            self._plot_stability_analysis(result, output_dir)
            
            # Plot 3: Top features
            self._plot_top_features(result, output_dir)
    
    def _plot_importance_comparison(self, result: FeatureImportanceResult, output_dir: Path):
        """Plot feature importance comparison across methods."""
        if len(result.method_scores) < 2:
            return
        
        # Create comparison plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        methods = list(result.method_scores.keys())
        top_features = result.get_top_features("ensemble", self.config.top_k_features)
        
        # Normalize scores for comparison
        normalized_scores = {}
        for method in methods:
            scores = result.method_scores[method]
            max_score = max(scores.values()) if scores else 1
            normalized_scores[method] = {f: scores.get(f, 0) / max_score for f in top_features}
        
        # Create heatmap
        data_matrix = []
        for feature in top_features:
            row = [normalized_scores[method].get(feature, 0) for method in methods]
            data_matrix.append(row)
        
        sns.heatmap(data_matrix, 
                   xticklabels=methods, 
                   yticklabels=top_features,
                   annot=True, 
                   fmt='.3f',
                   cmap='viridis',
                   ax=ax)
        
        ax.set_title('Feature Importance Comparison Across Methods')
        ax.set_xlabel('Methods')
        ax.set_ylabel('Features')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'importance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_stability_analysis(self, result: FeatureImportanceResult, output_dir: Path):
        """Plot stability analysis."""
        if not result.stability_scores:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        features = list(result.stability_scores.keys())
        stability_values = list(result.stability_scores.values())
        
        # Sort by stability
        sorted_data = sorted(zip(features, stability_values), key=lambda x: x[1], reverse=True)
        features, stability_values = zip(*sorted_data)
        
        bars = ax.bar(range(len(features)), stability_values)
        ax.set_xlabel('Features')
        ax.set_ylabel('Stability Score')
        ax.set_title('Feature Importance Stability Across Methods')
        ax.set_xticks(range(len(features)))
        ax.set_xticklabels(features, rotation=45, ha='right')
        
        # Color bars by stability
        for i, (bar, value) in enumerate(zip(bars, stability_values)):
            if value >= self.config.stability_threshold:
                bar.set_color('green')
            elif value >= self.config.stability_threshold * 0.7:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'stability_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_top_features(self, result: FeatureImportanceResult, output_dir: Path):
        """Plot top features."""
        top_features = result.get_top_features("ensemble", self.config.top_k_features)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get ensemble scores
        ensemble_scores = {}
        if 'ensemble' in result.rankings:
            for method, scores in result.method_scores.items():
                for feature in top_features:
                    if feature in scores:
                        if feature not in ensemble_scores:
                            ensemble_scores[feature] = []
                        ensemble_scores[feature].append(scores[feature])
        
        # Average scores
        avg_scores = {feature: np.mean(scores) for feature, scores in ensemble_scores.items()}
        
        # Sort features by score
        sorted_features = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        features, scores = zip(*sorted_features)
        
        bars = ax.barh(range(len(features)), scores)
        ax.set_xlabel('Importance Score')
        ax.set_ylabel('Features')
        ax.set_title(f'Top {len(features)} Most Important Features')
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'top_features.png', dpi=300, bbox_inches='tight')
        plt.close()

# Convenience functions
def analyze_feature_importance(X: pd.DataFrame, 
                             y: pd.Series,
                             regime_labels: Optional[pd.Series] = None,
                             config: Optional[FeatureImportanceConfig] = None) -> FeatureImportanceResult:
    """Convenience function for feature importance analysis."""
    analyzer = FeatureImportanceAnalyzer(config)
    return analyzer.analyze_features(X, y, regime_labels)

def get_important_features(X: pd.DataFrame, 
                          y: pd.Series,
                          regime_labels: Optional[pd.Series] = None,
                          k: int = 20,
                          methods: Optional[List[ImportanceMethod]] = None) -> List[str]:
    """Get top k important features using automated analysis."""
    if methods is None:
        methods = [ImportanceMethod.RANDOM_FOREST, ImportanceMethod.LASSO, ImportanceMethod.MUTUAL_INFO]
    
    config = FeatureImportanceConfig(
        methods=methods,
        top_k_features=k,
        save_results=False,
        generate_plots=False
    )
    
    analyzer = FeatureImportanceAnalyzer(config)
    result = analyzer.analyze_features(X, y, regime_labels)
    
    return result.get_top_features("ensemble", k)