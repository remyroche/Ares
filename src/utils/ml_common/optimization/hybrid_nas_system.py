"""
Hybrid NAS System - Combining Tree-Based and Neural Architecture Search

This module provides a comprehensive hybrid NAS system that combines
tree-based and neural architecture search approaches to leverage
the strengths of both methodologies.

Key Features:
- Tree-based NAS for fast feature selection and regime detection
- Neural NAS for complex pattern recognition and sequential modeling
- Ensemble methods combining both approaches
- Intelligent routing based on data characteristics
- Complementary optimization strategies
- Integration with existing neural NAS pipeline
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime
from abc import ABC, abstractmethod
import json
from pathlib import Path
from tprint import tprint

# Import existing neural NAS (if available)
try:
    from .neural_architecture_search import NeuralArchitectureSearch, ArchitectureConfig, ArchitectureCandidate
    NEURAL_NAS_AVAILABLE = True
except ImportError:
    tprint_warning("⚠️ [HYBRID_NAS] Neural architecture search module not available")
    NEURAL_NAS_AVAILABLE = False
    # Create placeholder classes
    class NeuralArchitectureSearch:
        def __init__(self, config): pass
        def search(self, *args, **kwargs): raise NotImplementedError("Neural NAS not available")
    class ArchitectureConfig: pass
    class ArchitectureCandidate: pass

# Import tree-based NAS (if available)
try:
    from .tree_based_architecture_search import TreeBasedArchitectureSearch, TreeArchitectureConfig, TreeArchitectureCandidate
    TREE_NAS_AVAILABLE = True
except ImportError:
    tprint_warning("⚠️ [HYBRID_NAS] Tree-based architecture search module not available")
    TREE_NAS_AVAILABLE = False
    # Create placeholder classes
    class TreeBasedArchitectureSearch:
        def __init__(self, config): pass
        def search(self, *args, **kwargs): raise NotImplementedError("Tree NAS not available")
    class TreeArchitectureConfig: pass
    class TreeArchitectureCandidate: pass

# Ensemble imports
try:
    from sklearn.ensemble import VotingRegressor, VotingClassifier, StackingRegressor, StackingClassifier
    from sklearn.model_selection import cross_val_score
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class HybridNASConfig:
    """Configuration for hybrid NAS system."""
    
    # Neural NAS configuration
    neural_config: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    
    # Tree-based NAS configuration
    tree_config: TreeArchitectureConfig = field(default_factory=TreeArchitectureConfig)
    
    # Hybrid strategy
    hybrid_strategy: str = 'complementary'  # 'complementary', 'ensemble', 'routing', 'sequential'
    
    # Data routing rules
    routing_rules: Dict[str, Any] = field(default_factory=lambda: {
        'use_tree_for_tabular': True,
        'use_neural_for_sequential': True,
        'use_tree_for_feature_selection': True,
        'use_neural_for_complex_patterns': True,
        'tabular_threshold': 0.7,  # If >70% tabular features, use tree
        'sequential_threshold': 0.5,  # If >50% sequential patterns, use neural
        'complexity_threshold': 0.8   # If >80% complex patterns, use neural
    })
    
    # Ensemble configuration
    ensemble_methods: List[str] = field(default_factory=lambda: ['voting', 'stacking', 'blending'])
    ensemble_weights: List[float] = field(default_factory=lambda: [0.5, 0.5])  # [tree_weight, neural_weight]
    
    # Performance thresholds
    performance_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'min_accuracy': 0.7,
        'min_efficiency': 0.5,
        'max_training_time': 3600,  # 1 hour
        'min_interpretability': 0.3
    })
    
    # Integration settings
    enable_feature_transfer: bool = True
    enable_architecture_transfer: bool = True
    enable_performance_transfer: bool = True
    
    # Optimization settings
    n_trials: int = 100
    timeout_seconds: int = 7200  # 2 hours
    early_stopping_patience: int = 10


@dataclass
class HybridArchitectureCandidate:
    """A candidate hybrid architecture combining tree and neural approaches."""
    
    # Individual architectures
    tree_architecture: Optional[TreeArchitectureCandidate] = None
    neural_architecture: Optional[ArchitectureCandidate] = None
    
    # Hybrid configuration
    hybrid_method: str = 'complementary'  # 'complementary', 'ensemble', 'routing', 'sequential'
    routing_strategy: Optional[Dict[str, Any]] = None
    ensemble_config: Optional[Dict[str, Any]] = None
    
    # Performance metrics
    combined_accuracy: float = 0.0
    combined_efficiency: float = 0.0
    combined_interpretability: float = 0.0
    combined_robustness: float = 0.0
    overall_score: float = 0.0
    
    # Training info
    total_training_time: float = 0.0
    tree_training_time: float = 0.0
    neural_training_time: float = 0.0
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    trial_number: int = 0


class HybridNASSystem:
    """Main hybrid NAS system combining tree-based and neural approaches."""
    
    def __init__(self, config: HybridNASConfig):
        """Initialize hybrid NAS system."""
        tprint("🚀 [HYBRID_NAS] Initializing Hybrid NAS System", color="cyan", bold=True)
        tprint(f"📊 [HYBRID_NAS] Strategy: {config.hybrid_strategy}", color="blue")
        self.config = config
        self.logger = logger.getChild('HybridNASSystem')
        
        # Initialize individual NAS systems
        if NEURAL_NAS_AVAILABLE:
            tprint("🧠 [HYBRID_NAS] Initializing neural NAS system", color="yellow")
            self.neural_nas = NeuralArchitectureSearch(config.neural_config)
        else:
            tprint_warning("⚠️ [HYBRID_NAS] Neural NAS not available, using placeholder")
            self.neural_nas = NeuralArchitectureSearch(config.neural_config)
            
        if TREE_NAS_AVAILABLE:
            tprint("🌳 [HYBRID_NAS] Initializing tree-based NAS system", color="yellow")
            self.tree_nas = TreeBasedArchitectureSearch(config.tree_config)
        else:
            tprint_warning("⚠️ [HYBRID_NAS] Tree NAS not available, using placeholder")
            self.tree_nas = TreeBasedArchitectureSearch(config.tree_config)
        
        # Hybrid components
        tprint("🔧 [HYBRID_NAS] Initializing hybrid components", color="blue")
        self.candidates = []
        self.best_candidate = None
        
        tprint(f"✅ [HYBRID_NAS] Hybrid NAS System initialized with strategy: {config.hybrid_strategy}", color="green", bold=True)
        self.logger.info(f"✅ Hybrid NAS System initialized with strategy: {config.hybrid_strategy}")
    
    def search(self, 
               X_train: np.ndarray, 
               y_train: np.ndarray,
               X_val: Optional[np.ndarray] = None,
               y_val: Optional[np.ndarray] = None,
               regime_labels: Optional[np.ndarray] = None,
               data_characteristics: Optional[Dict[str, Any]] = None) -> HybridArchitectureCandidate:
        """
        Perform hybrid architecture search.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            regime_labels: Regime labels for regime-aware search (optional)
            data_characteristics: Characteristics of the data to guide routing (optional)
            
        Returns:
            Best hybrid architecture candidate
        """
        tprint("🚀 [HYBRID_NAS] Starting Hybrid NAS Search", color="cyan", bold=True)
        tprint(f"📊 [HYBRID_NAS] Training data shape: {X_train.shape}, labels: {y_train.shape}", color="blue")
        self.logger.info("🚀 Starting Hybrid NAS Search...")
        start_time = time.time()
        
        try:
            # Prepare validation data
            if X_val is None or y_val is None:
                tprint("🔧 [HYBRID_NAS] Splitting training data for validation", color="yellow")
                from sklearn.model_selection import train_test_split
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42
                )
                tprint(f"📊 [HYBRID_NAS] Validation data shape: {X_val.shape}, labels: {y_val.shape}", color="blue")
            
            # Analyze data characteristics
            if data_characteristics is None:
                tprint("🔍 [HYBRID_NAS] Analyzing data characteristics", color="yellow")
                data_characteristics = self._analyze_data_characteristics(X_train, y_train)
                tprint(f"📊 [HYBRID_NAS] Data characteristics: {data_characteristics}", color="cyan")
            
            # Choose search strategy based on data characteristics
            tprint("🎯 [HYBRID_NAS] Choosing search strategy", color="yellow")
            search_strategy = self._choose_search_strategy(data_characteristics)
            tprint(f"📊 [HYBRID_NAS] Selected search strategy: {search_strategy}", color="green")
            self.logger.info(f"📊 Selected search strategy: {search_strategy}")
            
            # Perform hybrid search
            if search_strategy == 'complementary':
                best_candidate = self._complementary_search(X_train, y_train, X_val, y_val, regime_labels)
            elif search_strategy == 'ensemble':
                best_candidate = self._ensemble_search(X_train, y_train, X_val, y_val, regime_labels)
            elif search_strategy == 'routing':
                best_candidate = self._routing_search(X_train, y_train, X_val, y_val, regime_labels, data_characteristics)
            elif search_strategy == 'sequential':
                best_candidate = self._sequential_search(X_train, y_train, X_val, y_val, regime_labels)
            else:
                raise ValueError(f"Unknown search strategy: {search_strategy}")
            
            search_time = time.time() - start_time
            self.logger.info(f"✅ Hybrid NAS completed in {search_time:.2f}s")
            self.logger.info(f"📊 Best hybrid architecture: {best_candidate.hybrid_method}, score: {best_candidate.overall_score:.4f}")
            
            return best_candidate
            
        except Exception as e:
            self.logger.error(f"Hybrid NAS Search failed: {e}")
            raise
    
    def _analyze_data_characteristics(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Analyze data characteristics to guide routing decisions with comprehensive analysis."""
        try:
            # Import utilities for advanced analysis
            from src.utils.math_validation import safe_mean, safe_std, safe_correlation, validate_numeric_array
            from src.utils.common_operations import safe_divide, safe_weighted_average
            from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer
            
            # Validate inputs
            X_train = validate_numeric_array(X_train, "X_train")
            y_train = validate_numeric_array(y_train, "y_train")
            
            n_samples, n_features = X_train.shape
            
            # Basic characteristics
            basic_characteristics = {
                'n_samples': n_samples,
                'n_features': n_features,
                'data_density': n_samples / n_features,
                'feature_ratio': n_features / n_samples
            }
            
            # Calculate tabular vs sequential ratio with advanced analysis
            tabular_analysis = self._calculate_advanced_tabular_ratio(X_train)
            sequential_analysis = self._calculate_advanced_sequential_ratio(X_train)
            complexity_analysis = self._calculate_advanced_complexity_ratio(X_train)
            
            # Calculate data sparsity with detailed analysis
            sparsity_analysis = self._calculate_advanced_sparsity(X_train)
            
            # Calculate feature importance and variance
            feature_analysis = self._calculate_feature_analysis(X_train, y_train)
            
            # Calculate data distribution characteristics
            distribution_analysis = self._calculate_distribution_analysis(X_train, y_train)
            
            # Calculate correlation structure
            correlation_analysis = self._calculate_correlation_analysis(X_train)
            
            # Calculate data quality metrics
            quality_analysis = self._calculate_data_quality_metrics(X_train, y_train)
            
            # Calculate regime characteristics (if applicable)
            regime_analysis = self._calculate_regime_characteristics(X_train, y_train)
            
            # Calculate optimization complexity
            optimization_complexity = self._calculate_optimization_complexity(X_train, y_train)
            
            # Combine all characteristics
            characteristics = {
                **basic_characteristics,
                'tabular_analysis': tabular_analysis,
                'sequential_analysis': sequential_analysis,
                'complexity_analysis': complexity_analysis,
                'sparsity_analysis': sparsity_analysis,
                'feature_analysis': feature_analysis,
                'distribution_analysis': distribution_analysis,
                'correlation_analysis': correlation_analysis,
                'quality_analysis': quality_analysis,
                'regime_analysis': regime_analysis,
                'optimization_complexity': optimization_complexity,
                # Legacy compatibility
                'tabular_ratio': tabular_analysis['tabular_ratio'],
                'sequential_ratio': sequential_analysis['sequential_ratio'],
                'complexity_ratio': complexity_analysis['complexity_ratio'],
                'sparsity': sparsity_analysis['sparsity'],
                'feature_importance_variance': feature_analysis['feature_importance_variance'],
                'is_tabular_dominant': tabular_analysis['tabular_ratio'] > self.config.routing_rules['tabular_threshold'],
                'is_sequential_dominant': sequential_analysis['sequential_ratio'] > self.config.routing_rules['sequential_threshold'],
                'is_complex_dominant': complexity_analysis['complexity_ratio'] > self.config.routing_rules['complexity_threshold']
            }
            
            self.logger.debug(f"Advanced data characteristics analysis completed")
            self.logger.debug(f"Data characteristics: {characteristics}")
            return characteristics
            
        except Exception as e:
            self.logger.warning(f"Advanced data analysis failed: {e}")
            # Fallback to basic characteristics
            return {
                'n_samples': X_train.shape[0], 
                'n_features': X_train.shape[1],
                'tabular_ratio': 0.5,
                'sequential_ratio': 0.3,
                'complexity_ratio': 0.5,
                'sparsity': 0.0,
                'feature_importance_variance': 0.0,
                'is_tabular_dominant': False,
                'is_sequential_dominant': False,
                'is_complex_dominant': False,
                'error': str(e)
            }
    
    def _calculate_advanced_tabular_ratio(self, X: np.ndarray) -> Dict[str, Any]:
        """Calculate advanced tabular ratio analysis."""
        try:
            from src.utils.math_validation import safe_correlation, safe_mean, safe_std
            
            n_features = X.shape[1]
            position = np.arange(len(X))
            
            # Calculate correlation with position for each feature
            correlations = []
            for i in range(n_features):
                corr = safe_correlation(X[:, i], position, 0.0)
                correlations.append(corr)
            
            # Calculate tabular features (low correlation with position)
            tabular_threshold = 0.3
            tabular_features = sum(1 for corr in correlations if abs(corr) < tabular_threshold)
            tabular_ratio = tabular_features / n_features
            
            # Calculate feature independence
            feature_independence = []
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    corr = safe_correlation(X[:, i], X[:, j], 0.0)
                    feature_independence.append(abs(corr))
            
            independence_score = safe_mean(feature_independence) if feature_independence else 0.0
            
            # Calculate feature stability (low variance across samples)
            feature_stability = []
            for i in range(n_features):
                stability = 1.0 / (1.0 + safe_std(X[:, i]))
                feature_stability.append(stability)
            
            stability_score = safe_mean(feature_stability)
            
            return {
                'tabular_ratio': tabular_ratio,
                'n_tabular_features': tabular_features,
                'n_sequential_features': n_features - tabular_features,
                'correlations_with_position': correlations,
                'independence_score': independence_score,
                'stability_score': stability_score,
                'tabular_quality': (tabular_ratio + independence_score + stability_score) / 3.0
            }
            
        except Exception as e:
            self.logger.warning(f"Advanced tabular ratio calculation failed: {e}")
            return {'tabular_ratio': 0.5, 'error': str(e)}
    
    def _calculate_advanced_sequential_ratio(self, X: np.ndarray) -> Dict[str, Any]:
        """Calculate advanced sequential ratio analysis."""
        try:
            from src.utils.math_validation import safe_correlation, safe_mean, safe_std
            
            n_features = X.shape[1]
            sequential_features = 0
            autocorrelations = []
            
            for i in range(n_features):
                feature = X[:, i]
                if len(feature) > 1:
                    # Calculate autocorrelation
                    autocorr = safe_correlation(feature[:-1], feature[1:], 0.0)
                    autocorrelations.append(abs(autocorr))
                    
                    if abs(autocorr) > 0.3:  # Sequential threshold
                        sequential_features += 1
            
            sequential_ratio = sequential_features / n_features if n_features > 0 else 0.0
            
            # Calculate sequential patterns
            sequential_patterns = self._detect_sequential_patterns(X)
            
            # Calculate trend strength
            trend_strength = safe_mean(autocorrelations) if autocorrelations else 0.0
            
            return {
                'sequential_ratio': sequential_ratio,
                'n_sequential_features': sequential_features,
                'autocorrelations': autocorrelations,
                'sequential_patterns': sequential_patterns,
                'trend_strength': trend_strength,
                'sequential_quality': (sequential_ratio + trend_strength) / 2.0
            }
            
        except Exception as e:
            self.logger.warning(f"Advanced sequential ratio calculation failed: {e}")
            return {'sequential_ratio': 0.3, 'error': str(e)}
    
    def _calculate_advanced_complexity_ratio(self, X: np.ndarray) -> Dict[str, Any]:
        """Calculate advanced complexity ratio analysis."""
        try:
            from src.utils.math_validation import safe_mean, safe_std, safe_divide
            
            n_features = X.shape[1]
            complexities = []
            
            for i in range(n_features):
                feature = X[:, i]
                
                # Calculate feature complexity based on multiple factors
                variance = np.var(feature)
                
                # Non-linearity measure
                sorted_feature = np.sort(feature)
                non_linearity = np.var(np.diff(sorted_feature))
                
                # Entropy-based complexity
                from scipy.stats import entropy
                hist, _ = np.histogram(feature, bins=10)
                hist = hist / np.sum(hist)  # Normalize
                hist = hist[hist > 0]  # Remove zeros
                entropy_complexity = entropy(hist) if len(hist) > 0 else 0.0
                
                # Combined complexity score
                complexity = variance * non_linearity * entropy_complexity
                complexities.append(complexity)
            
            # Calculate complexity ratio
            max_complexity = max(complexities) if complexities else 1.0
            complex_features = sum(1 for c in complexities if c > 0.5 * max_complexity)
            complexity_ratio = complex_features / n_features if n_features > 0 else 0.0
            
            # Calculate complexity distribution
            complexity_stats = {
                'mean_complexity': safe_mean(complexities),
                'std_complexity': safe_std(complexities),
                'max_complexity': max(complexities) if complexities else 0.0,
                'min_complexity': min(complexities) if complexities else 0.0
            }
            
            return {
                'complexity_ratio': complexity_ratio,
                'n_complex_features': complex_features,
                'complexity_scores': complexities,
                'complexity_stats': complexity_stats,
                'complexity_quality': safe_divide(safe_mean(complexities), safe_std(complexities), 0.0)
            }
            
        except Exception as e:
            self.logger.warning(f"Advanced complexity ratio calculation failed: {e}")
            return {'complexity_ratio': 0.5, 'error': str(e)}
    
    def _calculate_advanced_sparsity(self, X: np.ndarray) -> Dict[str, Any]:
        """Calculate advanced sparsity analysis."""
        try:
            from src.utils.math_validation import safe_mean, safe_std
            
            # Basic sparsity
            zero_count = np.sum(X == 0)
            total_elements = X.size
            sparsity = zero_count / total_elements if total_elements > 0 else 0.0
            
            # Calculate sparsity by feature
            feature_sparsity = []
            for i in range(X.shape[1]):
                feature_zeros = np.sum(X[:, i] == 0)
                feature_sparsity.append(feature_zeros / len(X[:, i]))
            
            # Calculate sparsity patterns
            sparsity_patterns = {
                'uniform_sparsity': safe_std(feature_sparsity) < 0.1,
                'feature_sparsity_variance': safe_std(feature_sparsity),
                'mean_feature_sparsity': safe_mean(feature_sparsity)
            }
            
            # Calculate effective dimensionality
            effective_features = sum(1 for sp in feature_sparsity if sp < 0.9)
            effective_ratio = effective_features / X.shape[1] if X.shape[1] > 0 else 0.0
            
            return {
                'sparsity': sparsity,
                'feature_sparsity': feature_sparsity,
                'sparsity_patterns': sparsity_patterns,
                'effective_features': effective_features,
                'effective_ratio': effective_ratio,
                'sparsity_quality': 1.0 - sparsity  # Lower sparsity is better
            }
            
        except Exception as e:
            self.logger.warning(f"Advanced sparsity calculation failed: {e}")
            return {'sparsity': 0.0, 'error': str(e)}
    
    def _calculate_feature_analysis(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Calculate feature analysis including importance and variance."""
        try:
            from src.utils.math_validation import safe_mean, safe_std, safe_correlation, validate_numeric_array
            
            n_features = X.shape[1]
            
            # Calculate feature variance
            feature_variance = np.var(X, axis=0)
            feature_importance_variance = np.var(feature_variance)
            
            # Calculate feature-target correlations
            feature_target_correlations = []
            for i in range(n_features):
                corr = safe_correlation(X[:, i], y, 0.0)
                feature_target_correlations.append(abs(corr))
            
            # Calculate feature importance (simplified)
            feature_importance = np.array(feature_target_correlations)
            
            # Calculate feature redundancy
            feature_redundancy = []
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    corr = safe_correlation(X[:, i], X[:, j], 0.0)
                    feature_redundancy.append(abs(corr))
            
            redundancy_score = safe_mean(feature_redundancy) if feature_redundancy else 0.0
            
            # Calculate feature stability
            feature_stability = []
            for i in range(n_features):
                stability = 1.0 / (1.0 + safe_std(X[:, i]))
                feature_stability.append(stability)
            
            return {
                'feature_variance': feature_variance.tolist(),
                'feature_importance_variance': feature_importance_variance,
                'feature_target_correlations': feature_target_correlations,
                'feature_importance': feature_importance.tolist(),
                'redundancy_score': redundancy_score,
                'feature_stability': feature_stability,
                'mean_importance': safe_mean(feature_importance),
                'importance_std': safe_std(feature_importance)
            }
            
        except Exception as e:
            self.logger.warning(f"Feature analysis calculation failed: {e}")
            return {'feature_importance_variance': 0.0, 'error': str(e)}
    
    def _calculate_distribution_analysis(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Calculate data distribution characteristics."""
        try:
            from src.utils.math_validation import safe_mean, safe_std
            from scipy import stats
            
            # Calculate distribution characteristics for features
            feature_distributions = []
            for i in range(X.shape[1]):
                feature = X[:, i]
                
                # Basic statistics
                mean_val = safe_mean(feature)
                std_val = safe_std(feature)
                skewness = stats.skew(feature)
                kurtosis = stats.kurtosis(feature)
                
                # Distribution type detection
                if abs(skewness) < 0.5 and abs(kurtosis) < 0.5:
                    dist_type = 'normal'
                elif abs(skewness) > 1.0:
                    dist_type = 'skewed'
                elif abs(kurtosis) > 1.0:
                    dist_type = 'heavy_tailed'
                else:
                    dist_type = 'mixed'
                
                feature_distributions.append({
                    'mean': mean_val,
                    'std': std_val,
                    'skewness': skewness,
                    'kurtosis': kurtosis,
                    'distribution_type': dist_type
                })
            
            # Calculate overall distribution characteristics
            all_skewness = [fd['skewness'] for fd in feature_distributions]
            all_kurtosis = [fd['kurtosis'] for fd in feature_distributions]
            
            return {
                'feature_distributions': feature_distributions,
                'overall_skewness': safe_mean(all_skewness),
                'overall_kurtosis': safe_mean(all_kurtosis),
                'distribution_diversity': len(set(fd['distribution_type'] for fd in feature_distributions))
            }
            
        except Exception as e:
            self.logger.warning(f"Distribution analysis calculation failed: {e}")
            return {'error': str(e)}
    
    def _calculate_correlation_analysis(self, X: np.ndarray) -> Dict[str, Any]:
        """Calculate correlation structure analysis."""
        try:
            from src.utils.math_validation import safe_correlation
            
            n_features = X.shape[1]
            correlation_matrix = np.zeros((n_features, n_features))
            
            # Calculate pairwise correlations
            for i in range(n_features):
                for j in range(n_features):
                    if i != j:
                        corr = safe_correlation(X[:, i], X[:, j], 0.0)
                        correlation_matrix[i, j] = corr
            
            # Calculate correlation statistics
            upper_triangle = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
            correlation_stats = {
                'mean_correlation': np.mean(np.abs(upper_triangle)),
                'max_correlation': np.max(np.abs(upper_triangle)),
                'min_correlation': np.min(np.abs(upper_triangle)),
                'correlation_std': np.std(upper_triangle)
            }
            
            # Calculate correlation clusters
            high_correlation_pairs = np.sum(np.abs(upper_triangle) > 0.7)
            correlation_density = high_correlation_pairs / len(upper_triangle) if len(upper_triangle) > 0 else 0.0
            
            return {
                'correlation_matrix': correlation_matrix.tolist(),
                'correlation_stats': correlation_stats,
                'correlation_density': correlation_density,
                'high_correlation_pairs': high_correlation_pairs
            }
            
        except Exception as e:
            self.logger.warning(f"Correlation analysis calculation failed: {e}")
            return {'error': str(e)}
    
    def _calculate_data_quality_metrics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Calculate data quality metrics."""
        try:
            from src.utils.math_validation import safe_mean, safe_std
            
            # Calculate missing values
            missing_values = np.isnan(X).sum()
            missing_ratio = missing_values / X.size if X.size > 0 else 0.0
            
            # Calculate outliers (using IQR method)
            outlier_counts = []
            for i in range(X.shape[1]):
                feature = X[:, i]
                Q1 = np.percentile(feature, 25)
                Q3 = np.percentile(feature, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = np.sum((feature < lower_bound) | (feature > upper_bound))
                outlier_counts.append(outliers)
            
            outlier_ratio = sum(outlier_counts) / X.size if X.size > 0 else 0.0
            
            # Calculate data consistency
            feature_consistency = []
            for i in range(X.shape[1]):
                feature = X[:, i]
                consistency = 1.0 / (1.0 + safe_std(feature))
                feature_consistency.append(consistency)
            
            consistency_score = safe_mean(feature_consistency)
            
            return {
                'missing_values': missing_values,
                'missing_ratio': missing_ratio,
                'outlier_counts': outlier_counts,
                'outlier_ratio': outlier_ratio,
                'consistency_score': consistency_score,
                'data_quality_score': (1.0 - missing_ratio) * (1.0 - outlier_ratio) * consistency_score
            }
            
        except Exception as e:
            self.logger.warning(f"Data quality metrics calculation failed: {e}")
            return {'error': str(e)}
    
    def _calculate_regime_characteristics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Calculate regime characteristics if applicable."""
        try:
            from sklearn.cluster import KMeans
            from src.utils.math_validation import safe_mean, safe_std
            
            # Simple regime detection using clustering
            n_regimes = min(5, X.shape[0] // 10)  # Adaptive number of regimes
            if n_regimes < 2:
                return {'n_regimes': 1, 'regime_quality': 0.0}
            
            kmeans = KMeans(n_clusters=n_regimes, random_state=42)
            regime_labels = kmeans.fit_predict(X)
            
            # Calculate regime characteristics
            unique_regimes = np.unique(regime_labels)
            regime_sizes = [np.sum(regime_labels == regime) for regime in unique_regimes]
            regime_balance = 1.0 - (np.std(regime_sizes) / np.mean(regime_sizes)) if np.mean(regime_sizes) > 0 else 0.0
            
            # Calculate regime separation
            regime_centers = kmeans.cluster_centers_
            regime_separation = np.mean([np.linalg.norm(regime_centers[i] - regime_centers[j]) 
                                       for i in range(len(regime_centers)) 
                                       for j in range(i + 1, len(regime_centers))])
            
            return {
                'n_regimes': len(unique_regimes),
                'regime_labels': regime_labels.tolist(),
                'regime_centers': regime_centers.tolist(),
                'regime_balance': regime_balance,
                'regime_separation': regime_separation,
                'regime_quality': (regime_balance + min(1.0, regime_separation / 10.0)) / 2.0
            }
            
        except Exception as e:
            self.logger.warning(f"Regime characteristics calculation failed: {e}")
            return {'n_regimes': 1, 'regime_quality': 0.0, 'error': str(e)}
    
    def _calculate_optimization_complexity(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Calculate optimization complexity metrics."""
        try:
            from src.utils.math_validation import safe_mean, safe_std
            
            n_samples, n_features = X.shape
            
            # Calculate search space complexity
            search_space_size = n_features ** 2  # Simplified estimate
            
            # Calculate data complexity
            data_complexity = n_samples * n_features
            
            # Calculate feature interaction complexity
            feature_interactions = n_features * (n_features - 1) // 2
            
            # Calculate optimization difficulty
            optimization_difficulty = (data_complexity * feature_interactions) / (n_samples * n_features)
            
            return {
                'search_space_size': search_space_size,
                'data_complexity': data_complexity,
                'feature_interactions': feature_interactions,
                'optimization_difficulty': optimization_difficulty,
                'complexity_score': min(1.0, optimization_difficulty / 1000.0)
            }
            
        except Exception as e:
            self.logger.warning(f"Optimization complexity calculation failed: {e}")
            return {'error': str(e)}
    
    def _detect_sequential_patterns(self, X: np.ndarray) -> Dict[str, Any]:
        """Detect sequential patterns in the data."""
        try:
            from src.utils.math_validation import safe_correlation
            
            n_features = X.shape[1]
            patterns = {
                'trending_features': 0,
                'cyclical_features': 0,
                'random_features': 0,
                'pattern_details': []
            }
            
            for i in range(n_features):
                feature = X[:, i]
                
                # Calculate autocorrelation at different lags
                autocorr_1 = safe_correlation(feature[:-1], feature[1:], 0.0)
                autocorr_2 = safe_correlation(feature[:-2], feature[2:], 0.0) if len(feature) > 2 else 0.0
                
                # Classify pattern type
                if abs(autocorr_1) > 0.5:
                    if autocorr_1 > 0:
                        pattern_type = 'trending'
                        patterns['trending_features'] += 1
                    else:
                        pattern_type = 'cyclical'
                        patterns['cyclical_features'] += 1
                else:
                    pattern_type = 'random'
                    patterns['random_features'] += 1
                
                patterns['pattern_details'].append({
                    'feature_idx': i,
                    'pattern_type': pattern_type,
                    'autocorr_1': autocorr_1,
                    'autocorr_2': autocorr_2
                })
            
            return patterns
            
        except Exception as e:
            self.logger.warning(f"Sequential pattern detection failed: {e}")
            return {'error': str(e)}
    
    def _calculate_tabular_ratio(self, X: np.ndarray) -> float:
        """Calculate ratio of tabular features (legacy method)."""
        try:
            # Calculate correlation with position (proxy for time)
            position = np.arange(len(X))
            correlations = [np.corrcoef(X[:, i], position)[0, 1] for i in range(X.shape[1])]
            tabular_features = sum(1 for corr in correlations if abs(corr) < 0.3)
            return tabular_features / X.shape[1]
        except:
            return 0.5  # Default assumption
    
    def _calculate_sequential_ratio(self, X: np.ndarray) -> float:
        """Calculate ratio of sequential features."""
        try:
            # Calculate autocorrelation for each feature
            autocorrelations = []
            for i in range(X.shape[1]):
                feature = X[:, i]
                if len(feature) > 1:
                    autocorr = np.corrcoef(feature[:-1], feature[1:])[0, 1]
                    autocorrelations.append(abs(autocorr))
            
            sequential_features = sum(1 for ac in autocorrelations if ac > 0.3)
            return sequential_features / len(autocorrelations) if autocorrelations else 0.0
        except:
            return 0.3  # Default assumption
    
    def _calculate_complexity_ratio(self, X: np.ndarray) -> float:
        """Calculate ratio of complex features."""
        try:
            # Calculate feature complexity based on variance and non-linearity
            complexities = []
            for i in range(X.shape[1]):
                feature = X[:, i]
                variance = np.var(feature)
                # Simple non-linearity measure
                sorted_feature = np.sort(feature)
                non_linearity = np.var(np.diff(sorted_feature))
                complexity = variance * non_linearity
                complexities.append(complexity)
            
            # Normalize and calculate ratio
            max_complexity = max(complexities) if complexities else 1.0
            complex_features = sum(1 for c in complexities if c > 0.5 * max_complexity)
            return complex_features / len(complexities) if complexities else 0.5
        except:
            return 0.5  # Default assumption
    
    def _calculate_sparsity(self, X: np.ndarray) -> float:
        """Calculate data sparsity."""
        try:
            zero_count = np.sum(X == 0)
            total_elements = X.size
            return zero_count / total_elements
        except:
            return 0.0
    
    def _choose_search_strategy(self, data_characteristics: Dict[str, Any]) -> str:
        """Choose the best search strategy based on data characteristics."""
        try:
            # Use routing rules to determine strategy
            if data_characteristics.get('is_tabular_dominant', False):
                return 'complementary'  # Tree for tabular, neural for complex patterns
            elif data_characteristics.get('is_sequential_dominant', False):
                return 'sequential'  # Sequential processing
            elif data_characteristics.get('is_complex_dominant', False):
                return 'ensemble'  # Combine both approaches
            else:
                return self.config.hybrid_strategy  # Use configured strategy
        except Exception as e:
            self.logger.warning(f"Strategy selection failed: {e}")
            return self.config.hybrid_strategy
    
    def _complementary_search(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray,
                            regime_labels: Optional[np.ndarray] = None) -> HybridArchitectureCandidate:
        """Perform complementary search using both tree and neural NAS."""
        self.logger.info("🔍 Starting complementary search...")
        
        best_candidate = None
        best_score = -np.inf
        
        for trial in range(self.config.n_trials):
            try:
                # Search tree-based architecture for feature selection and regime detection
                tree_architecture = self.tree_nas.search(X_train, y_train, X_val, y_val, regime_labels)
                
                # Use tree results to guide neural architecture search
                selected_features = self._get_selected_features(tree_architecture)
                X_train_selected = X_train[:, selected_features] if selected_features else X_train
                X_val_selected = X_val[:, selected_features] if selected_features else X_val
                
                # Search neural architecture for complex patterns
                neural_architecture = self.neural_nas.search(X_train_selected, y_train, X_val_selected, y_val, regime_labels)
                
                # Create hybrid candidate
                hybrid_candidate = HybridArchitectureCandidate(
                    tree_architecture=tree_architecture,
                    neural_architecture=neural_architecture,
                    hybrid_method='complementary',
                    trial_number=trial
                )
                
                # Evaluate hybrid performance
                performance = self._evaluate_hybrid_architecture(hybrid_candidate, X_train, y_train, X_val, y_val)
                
                # Update best candidate
                if performance['overall_score'] > best_score:
                    best_score = performance['overall_score']
                    best_candidate = hybrid_candidate
                    best_candidate.combined_accuracy = performance['accuracy']
                    best_candidate.combined_efficiency = performance['efficiency']
                    best_candidate.combined_interpretability = performance['interpretability']
                    best_candidate.combined_robustness = performance['robustness']
                    best_candidate.overall_score = performance['overall_score']
                
                self.logger.debug(f"Trial {trial}: Hybrid score {performance['overall_score']:.4f}")
                
            except Exception as e:
                tprint_error(f"❌ [HYBRID_NAS] Trial {trial} failed: {e}")
                self.logger.error(f"Trial {trial} failed: {e}")
                # Continue with next trial but log the error properly
                continue
        
        if best_candidate is None:
            raise RuntimeError("No successful hybrid architecture found")
        
        return best_candidate
    
    def _ensemble_search(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray,
                        regime_labels: Optional[np.ndarray] = None) -> HybridArchitectureCandidate:
        """Perform ensemble search combining tree and neural approaches."""
        self.logger.info("🔍 Starting ensemble search...")
        
        # Search both architectures independently
        tree_architecture = self.tree_nas.search(X_train, y_train, X_val, y_val, regime_labels)
        neural_architecture = self.neural_nas.search(X_train, y_train, X_val, y_val, regime_labels)
        
        # Create ensemble configuration
        ensemble_config = {
            'method': 'voting',
            'weights': self.config.ensemble_weights,
            'tree_weight': self.config.ensemble_weights[0],
            'neural_weight': self.config.ensemble_weights[1]
        }
        
        # Create hybrid candidate
        hybrid_candidate = HybridArchitectureCandidate(
            tree_architecture=tree_architecture,
            neural_architecture=neural_architecture,
            hybrid_method='ensemble',
            ensemble_config=ensemble_config
        )
        
        # Evaluate ensemble performance
        performance = self._evaluate_hybrid_architecture(hybrid_candidate, X_train, y_train, X_val, y_val)
        
        hybrid_candidate.combined_accuracy = performance['accuracy']
        hybrid_candidate.combined_efficiency = performance['efficiency']
        hybrid_candidate.combined_interpretability = performance['interpretability']
        hybrid_candidate.combined_robustness = performance['robustness']
        hybrid_candidate.overall_score = performance['overall_score']
        
        return hybrid_candidate
    
    def _routing_search(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       regime_labels: Optional[np.ndarray] = None,
                       data_characteristics: Optional[Dict[str, Any]] = None) -> HybridArchitectureCandidate:
        """Perform routing search based on data characteristics."""
        self.logger.info("🔍 Starting routing search...")
        
        # Determine which approach to use based on data characteristics
        if data_characteristics and data_characteristics.get('is_tabular_dominant', False):
            # Use tree-based approach for tabular data
            tree_architecture = self.tree_nas.search(X_train, y_train, X_val, y_val, regime_labels)
            hybrid_candidate = HybridArchitectureCandidate(
                tree_architecture=tree_architecture,
                hybrid_method='routing',
                routing_strategy={'primary': 'tree', 'reason': 'tabular_dominant'}
            )
        else:
            # Use neural approach for complex/sequential data
            neural_architecture = self.neural_nas.search(X_train, y_train, X_val, y_val, regime_labels)
            hybrid_candidate = HybridArchitectureCandidate(
                neural_architecture=neural_architecture,
                hybrid_method='routing',
                routing_strategy={'primary': 'neural', 'reason': 'complex_dominant'}
            )
        
        # Evaluate performance
        performance = self._evaluate_hybrid_architecture(hybrid_candidate, X_train, y_train, X_val, y_val)
        
        hybrid_candidate.combined_accuracy = performance['accuracy']
        hybrid_candidate.combined_efficiency = performance['efficiency']
        hybrid_candidate.combined_interpretability = performance['interpretability']
        hybrid_candidate.combined_robustness = performance['robustness']
        hybrid_candidate.overall_score = performance['overall_score']
        
        return hybrid_candidate
    
    def _sequential_search(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray,
                          regime_labels: Optional[np.ndarray] = None) -> HybridArchitectureCandidate:
        """Perform sequential search using tree first, then neural."""
        self.logger.info("🔍 Starting sequential search...")
        
        # Step 1: Use tree-based NAS for feature selection and regime detection
        tree_architecture = self.tree_nas.search(X_train, y_train, X_val, y_val, regime_labels)
        
        # Step 2: Use tree results to guide neural architecture search
        selected_features = self._get_selected_features(tree_architecture)
        X_train_selected = X_train[:, selected_features] if selected_features else X_train
        X_val_selected = X_val[:, selected_features] if selected_features else X_val
        
        # Step 3: Use neural NAS for complex pattern recognition
        neural_architecture = self.neural_nas.search(X_train_selected, y_train, X_val_selected, y_val, regime_labels)
        
        # Create hybrid candidate
        hybrid_candidate = HybridArchitectureCandidate(
            tree_architecture=tree_architecture,
            neural_architecture=neural_architecture,
            hybrid_method='sequential'
        )
        
        # Evaluate performance
        performance = self._evaluate_hybrid_architecture(hybrid_candidate, X_train, y_train, X_val, y_val)
        
        hybrid_candidate.combined_accuracy = performance['accuracy']
        hybrid_candidate.combined_efficiency = performance['efficiency']
        hybrid_candidate.combined_interpretability = performance['interpretability']
        hybrid_candidate.combined_robustness = performance['robustness']
        hybrid_candidate.overall_score = performance['overall_score']
        
        return hybrid_candidate
    
    def _get_selected_features(self, tree_architecture: TreeArchitectureCandidate) -> List[int]:
        """Extract selected features from tree architecture with intelligent feature selection."""
        try:
            from src.utils.math_validation import safe_mean, safe_std
            from src.utils.common_operations import safe_divide
            
            # Initialize feature selection
            selected_features = []
            
            # Method 1: Extract from tree architecture if available
            if hasattr(tree_architecture, 'feature_importance') and tree_architecture.feature_importance is not None:
                feature_importance = tree_architecture.feature_importance
                
                # Calculate importance threshold
                mean_importance = safe_mean(feature_importance)
                std_importance = safe_std(feature_importance)
                threshold = mean_importance + 0.5 * std_importance
                
                # Select features above threshold
                selected_features = [i for i, imp in enumerate(feature_importance) if imp > threshold]
                
                self.logger.debug(f"Selected {len(selected_features)} features based on importance threshold")
            
            # Method 2: Use n_features if available
            elif hasattr(tree_architecture, 'n_features') and tree_architecture.n_features:
                n_features = min(tree_architecture.n_features, 50)  # Limit to reasonable number
                
                # If we have feature importance, use it to rank features
                if hasattr(tree_architecture, 'feature_importance') and tree_architecture.feature_importance is not None:
                    feature_importance = tree_architecture.feature_importance
                    # Sort by importance and take top n_features
                    importance_indices = np.argsort(feature_importance)[::-1]
                    selected_features = importance_indices[:n_features].tolist()
                else:
                    # Fallback to first n_features
                    selected_features = list(range(n_features))
                
                self.logger.debug(f"Selected {len(selected_features)} features based on n_features")
            
            # Method 3: Use model-specific feature selection
            elif hasattr(tree_architecture, 'model') and tree_architecture.model is not None:
                model = tree_architecture.model
                
                # Try to extract feature importance from the model
                if hasattr(model, 'feature_importances_'):
                    feature_importance = model.feature_importances_
                    
                    # Calculate dynamic threshold
                    mean_importance = safe_mean(feature_importance)
                    std_importance = safe_std(feature_importance)
                    threshold = max(mean_importance, 0.01)  # Minimum threshold
                    
                    # Select features above threshold
                    selected_features = [i for i, imp in enumerate(feature_importance) if imp > threshold]
                    
                    self.logger.debug(f"Selected {len(selected_features)} features from model feature_importances_")
                
                elif hasattr(model, 'coef_'):
                    # For linear models, use coefficient magnitude
                    coef = model.coef_
                    if coef.ndim > 1:
                        coef = coef[0]  # Take first class for multi-class
                    
                    coef_magnitude = np.abs(coef)
                    mean_coef = safe_mean(coef_magnitude)
                    threshold = mean_coef
                    
                    selected_features = [i for i, coef_val in enumerate(coef_magnitude) if coef_val > threshold]
                    
                    self.logger.debug(f"Selected {len(selected_features)} features from model coefficients")
            
            # Method 4: Use architecture-specific feature selection
            elif hasattr(tree_architecture, 'architecture_config'):
                config = tree_architecture.architecture_config
                
                # Extract feature selection from config
                if 'selected_features' in config:
                    selected_features = config['selected_features']
                elif 'feature_mask' in config:
                    feature_mask = config['feature_mask']
                    selected_features = [i for i, mask in enumerate(feature_mask) if mask]
                elif 'n_features' in config:
                    n_features = min(config['n_features'], 50)
                    selected_features = list(range(n_features))
                
                self.logger.debug(f"Selected {len(selected_features)} features from architecture config")
            
            # Method 5: Intelligent feature selection based on data characteristics
            else:
                # Use data characteristics to select features
                if hasattr(self, '_data_characteristics') and self._data_characteristics:
                    characteristics = self._data_characteristics
                    
                    # Use feature analysis if available
                    if 'feature_analysis' in characteristics:
                        feature_analysis = characteristics['feature_analysis']
                        
                        if 'feature_importance' in feature_analysis:
                            feature_importance = feature_analysis['feature_importance']
                            
                            # Calculate threshold based on importance distribution
                            mean_importance = safe_mean(feature_importance)
                            std_importance = safe_std(feature_importance)
                            threshold = mean_importance + 0.3 * std_importance
                            
                            selected_features = [i for i, imp in enumerate(feature_importance) if imp > threshold]
                            
                            self.logger.debug(f"Selected {len(selected_features)} features based on data characteristics")
            
            # Validate and limit selected features
            if not selected_features:
                self.logger.warning("No features selected, using fallback selection")
                # Fallback: select first 10 features
                selected_features = list(range(min(10, 50)))
            
            # Limit to reasonable number of features
            max_features = 100
            if len(selected_features) > max_features:
                # If we have too many features, select the most important ones
                if hasattr(tree_architecture, 'feature_importance') and tree_architecture.feature_importance is not None:
                    feature_importance = tree_architecture.feature_importance
                    importance_indices = np.argsort(feature_importance)[::-1]
                    selected_features = importance_indices[:max_features].tolist()
                else:
                    selected_features = selected_features[:max_features]
            
            # Ensure features are within valid range
            max_feature_index = 1000  # Reasonable upper bound
            selected_features = [f for f in selected_features if 0 <= f < max_feature_index]
            
            self.logger.info(f"Feature selection completed: {len(selected_features)} features selected")
            return selected_features
            
        except Exception as e:
            self.logger.error(f"Feature selection failed: {e}")
            # Fallback: return empty list or first few features
            return []
    
    def _evaluate_hybrid_architecture(self, hybrid_candidate: HybridArchitectureCandidate,
                                     X_train: np.ndarray, y_train: np.ndarray,
                                     X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """Evaluate hybrid architecture performance with comprehensive evaluation."""
        try:
            from src.utils.math_validation import safe_mean, safe_std, safe_divide
            from src.utils.common_operations import safe_weighted_average
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            from sklearn.model_selection import cross_val_score
            import time
            
            # Initialize performance tracking
            start_time = time.time()
            evaluation_results = {}
            
            # Extract individual architecture performances
            tree_performance = self._evaluate_individual_architecture(
                hybrid_candidate.tree_architecture, 'tree', X_train, y_train, X_val, y_val
            )
            
            neural_performance = self._evaluate_individual_architecture(
                hybrid_candidate.neural_architecture, 'neural', X_train, y_train, X_val, y_val
            )
            
            # Calculate hybrid-specific metrics
            hybrid_metrics = self._calculate_hybrid_metrics(
                hybrid_candidate, tree_performance, neural_performance, X_train, y_train, X_val, y_val
            )
            
            # Calculate ensemble performance if applicable
            ensemble_performance = {}
            if hybrid_candidate.hybrid_method == 'ensemble':
                ensemble_performance = self._calculate_ensemble_performance(
                    hybrid_candidate, tree_performance, neural_performance, X_train, y_train, X_val, y_val
                )
            
            # Calculate complementary performance
            complementary_performance = self._calculate_complementary_performance(
                hybrid_candidate, tree_performance, neural_performance
            )
            
            # Calculate routing performance
            routing_performance = self._calculate_routing_performance(
                hybrid_candidate, tree_performance, neural_performance
            )
            
            # Combine performances based on hybrid method
            if hybrid_candidate.hybrid_method == 'ensemble':
                # Use ensemble performance
                combined_accuracy = ensemble_performance.get('accuracy', 0.0)
                combined_efficiency = ensemble_performance.get('efficiency', 0.0)
                combined_interpretability = ensemble_performance.get('interpretability', 0.0)
                combined_robustness = ensemble_performance.get('robustness', 0.0)
            elif hybrid_candidate.hybrid_method == 'complementary':
                # Use complementary performance
                combined_accuracy = complementary_performance.get('accuracy', 0.0)
                combined_efficiency = complementary_performance.get('efficiency', 0.0)
                combined_interpretability = complementary_performance.get('interpretability', 0.0)
                combined_robustness = complementary_performance.get('robustness', 0.0)
            elif hybrid_candidate.hybrid_method == 'routing':
                # Use routing performance
                combined_accuracy = routing_performance.get('accuracy', 0.0)
                combined_efficiency = routing_performance.get('efficiency', 0.0)
                combined_interpretability = routing_performance.get('interpretability', 0.0)
                combined_robustness = routing_performance.get('robustness', 0.0)
            else:
                # Use the best performing approach
                combined_accuracy = max(tree_performance.get('accuracy', 0.0), neural_performance.get('accuracy', 0.0))
                combined_efficiency = max(tree_performance.get('efficiency', 0.0), neural_performance.get('efficiency', 0.0))
                combined_interpretability = max(tree_performance.get('interpretability', 0.0), neural_performance.get('interpretability', 0.0))
                combined_robustness = max(tree_performance.get('robustness', 0.0), neural_performance.get('robustness', 0.0))
            
            # Calculate overall score with hybrid-specific weighting
            overall_score = self._calculate_overall_score(
                combined_accuracy, combined_efficiency, combined_interpretability, combined_robustness,
                hybrid_metrics, hybrid_candidate.hybrid_method
            )
            
            # Calculate evaluation time
            evaluation_time = time.time() - start_time
            
            # Create comprehensive results
            results = {
                'accuracy': combined_accuracy,
                'efficiency': combined_efficiency,
                'interpretability': combined_interpretability,
                'robustness': combined_robustness,
                'overall_score': overall_score,
                'tree_performance': tree_performance,
                'neural_performance': neural_performance,
                'hybrid_metrics': hybrid_metrics,
                'ensemble_performance': ensemble_performance,
                'complementary_performance': complementary_performance,
                'routing_performance': routing_performance,
                'evaluation_time': evaluation_time,
                'hybrid_method': hybrid_candidate.hybrid_method
            }
            
            self.logger.info(f"Hybrid architecture evaluation completed: {overall_score:.4f} overall score")
            return results
            
        except Exception as e:
            self.logger.error(f"Hybrid evaluation failed: {e}")
            raise RuntimeError(f"Hybrid evaluation failed: {e}") from e
    
    def _evaluate_individual_architecture(self, architecture, arch_type: str, 
                                        X_train: np.ndarray, y_train: np.ndarray,
                                        X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """Evaluate individual architecture performance."""
        try:
            if architecture is None:
                return {'accuracy': 0.0, 'efficiency': 0.0, 'interpretability': 0.0, 'robustness': 0.0}
            
            # Extract basic performance metrics
            performance = {
                'accuracy': getattr(architecture, 'accuracy', 0.0),
                'efficiency': getattr(architecture, 'efficiency_score', 0.0),
                'interpretability': getattr(architecture, 'interpretability_score', 0.0),
                'robustness': getattr(architecture, 'robustness_score', 0.0)
            }
            
            # Calculate additional metrics if model is available
            if hasattr(architecture, 'model') and architecture.model is not None:
                model = architecture.model
                
                try:
                    # Calculate validation performance
                    if X_val is not None and y_val is not None:
                        y_pred = model.predict(X_val)
                        val_accuracy = accuracy_score(y_val, y_pred)
                        performance['validation_accuracy'] = val_accuracy
                        
                        # Calculate additional metrics
                        if len(np.unique(y_val)) > 2:  # Multi-class
                            performance['precision'] = precision_score(y_val, y_pred, average='weighted')
                            performance['recall'] = recall_score(y_val, y_pred, average='weighted')
                            performance['f1'] = f1_score(y_val, y_pred, average='weighted')
                        else:  # Binary
                            performance['precision'] = precision_score(y_val, y_pred)
                            performance['recall'] = recall_score(y_val, y_pred)
                            performance['f1'] = f1_score(y_val, y_pred)
                    
                    # Calculate cross-validation performance
                    if X_train is not None and y_train is not None:
                        cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
                        performance['cv_accuracy'] = safe_mean(cv_scores)
                        performance['cv_std'] = safe_std(cv_scores)
                    
                except Exception as e:
                    self.logger.warning(f"Additional metrics calculation failed for {arch_type}: {e}")
            
            return performance
            
        except Exception as e:
            self.logger.warning(f"Individual architecture evaluation failed for {arch_type}: {e}")
            return {'accuracy': 0.0, 'efficiency': 0.0, 'interpretability': 0.0, 'robustness': 0.0}
    
    def _calculate_hybrid_metrics(self, hybrid_candidate, tree_performance, neural_performance,
                                 X_train, y_train, X_val, y_val) -> Dict[str, float]:
        """Calculate hybrid-specific metrics."""
        try:
            from src.utils.math_validation import safe_mean, safe_std, safe_divide
            
            # Calculate synergy metrics
            synergy_accuracy = (tree_performance.get('accuracy', 0.0) + neural_performance.get('accuracy', 0.0)) / 2.0
            synergy_efficiency = (tree_performance.get('efficiency', 0.0) + neural_performance.get('efficiency', 0.0)) / 2.0
            
            # Calculate diversity metrics
            accuracy_diversity = abs(tree_performance.get('accuracy', 0.0) - neural_performance.get('accuracy', 0.0))
            efficiency_diversity = abs(tree_performance.get('efficiency', 0.0) - neural_performance.get('efficiency', 0.0))
            
            # Calculate complementarity
            complementarity_score = 1.0 - (accuracy_diversity + efficiency_diversity) / 2.0
            
            # Calculate hybrid stability
            stability_score = 1.0 - safe_std([tree_performance.get('accuracy', 0.0), neural_performance.get('accuracy', 0.0)])
            
            return {
                'synergy_accuracy': synergy_accuracy,
                'synergy_efficiency': synergy_efficiency,
                'accuracy_diversity': accuracy_diversity,
                'efficiency_diversity': efficiency_diversity,
                'complementarity_score': complementarity_score,
                'stability_score': stability_score,
                'hybrid_quality': (synergy_accuracy + complementarity_score + stability_score) / 3.0
            }
            
        except Exception as e:
            self.logger.warning(f"Hybrid metrics calculation failed: {e}")
            return {'hybrid_quality': 0.0}
    
    def _calculate_ensemble_performance(self, hybrid_candidate, tree_performance, neural_performance,
                                      X_train, y_train, X_val, y_val) -> Dict[str, float]:
        """Calculate ensemble performance."""
        try:
            # Use configured ensemble weights
            weights = self.config.ensemble_weights
            tree_weight = weights[0]
            neural_weight = weights[1]
            
            # Calculate weighted ensemble performance
            ensemble_accuracy = (tree_weight * tree_performance.get('accuracy', 0.0) + 
                               neural_weight * neural_performance.get('accuracy', 0.0))
            ensemble_efficiency = (tree_weight * tree_performance.get('efficiency', 0.0) + 
                                 neural_weight * neural_performance.get('efficiency', 0.0))
            ensemble_interpretability = (tree_weight * tree_performance.get('interpretability', 0.0) + 
                                       neural_weight * neural_performance.get('interpretability', 0.0))
            ensemble_robustness = (tree_weight * tree_performance.get('robustness', 0.0) + 
                                neural_weight * neural_performance.get('robustness', 0.0))
            
            return {
                'accuracy': ensemble_accuracy,
                'efficiency': ensemble_efficiency,
                'interpretability': ensemble_interpretability,
                'robustness': ensemble_robustness,
                'ensemble_quality': (ensemble_accuracy + ensemble_efficiency + ensemble_interpretability + ensemble_robustness) / 4.0
            }
            
        except Exception as e:
            self.logger.warning(f"Ensemble performance calculation failed: {e}")
            return {'accuracy': 0.0, 'efficiency': 0.0, 'interpretability': 0.0, 'robustness': 0.0}
    
    def _calculate_complementary_performance(self, hybrid_candidate, tree_performance, neural_performance) -> Dict[str, float]:
        """Calculate complementary performance."""
        try:
            # Use the best performing approach for each metric
            complementary_accuracy = max(tree_performance.get('accuracy', 0.0), neural_performance.get('accuracy', 0.0))
            complementary_efficiency = max(tree_performance.get('efficiency', 0.0), neural_performance.get('efficiency', 0.0))
            complementary_interpretability = max(tree_performance.get('interpretability', 0.0), neural_performance.get('interpretability', 0.0))
            complementary_robustness = max(tree_performance.get('robustness', 0.0), neural_performance.get('robustness', 0.0))
            
            return {
                'accuracy': complementary_accuracy,
                'efficiency': complementary_efficiency,
                'interpretability': complementary_interpretability,
                'robustness': complementary_robustness,
                'complementary_quality': (complementary_accuracy + complementary_efficiency + 
                                       complementary_interpretability + complementary_robustness) / 4.0
            }
            
        except Exception as e:
            self.logger.warning(f"Complementary performance calculation failed: {e}")
            return {'accuracy': 0.0, 'efficiency': 0.0, 'interpretability': 0.0, 'robustness': 0.0}
    
    def _calculate_routing_performance(self, hybrid_candidate, tree_performance, neural_performance) -> Dict[str, float]:
        """Calculate routing performance."""
        try:
            # Use routing strategy to determine which approach to use
            routing_strategy = getattr(hybrid_candidate, 'routing_strategy', {})
            primary_approach = routing_strategy.get('primary', 'tree')
            
            if primary_approach == 'tree':
                return tree_performance
            elif primary_approach == 'neural':
                return neural_performance
            else:
                # Fallback to best performing approach
                return self._calculate_complementary_performance(hybrid_candidate, tree_performance, neural_performance)
                
        except Exception as e:
            self.logger.warning(f"Routing performance calculation failed: {e}")
            return {'accuracy': 0.0, 'efficiency': 0.0, 'interpretability': 0.0, 'robustness': 0.0}
    
    def _calculate_overall_score(self, accuracy, efficiency, interpretability, robustness,
                                hybrid_metrics, hybrid_method) -> float:
        """Calculate overall score with hybrid-specific weighting."""
        try:
            # Base score calculation
            base_score = (0.4 * accuracy + 0.2 * efficiency + 0.2 * interpretability + 0.2 * robustness)
            
            # Add hybrid-specific bonuses
            hybrid_bonus = 0.0
            if hybrid_metrics:
                hybrid_quality = hybrid_metrics.get('hybrid_quality', 0.0)
                complementarity = hybrid_metrics.get('complementarity_score', 0.0)
                stability = hybrid_metrics.get('stability_score', 0.0)
                
                hybrid_bonus = (hybrid_quality + complementarity + stability) / 3.0 * 0.1
            
            # Method-specific adjustments
            method_bonus = 0.0
            if hybrid_method == 'ensemble':
                method_bonus = 0.05  # Ensemble gets slight bonus for complexity
            elif hybrid_method == 'complementary':
                method_bonus = 0.03  # Complementary gets bonus for efficiency
            elif hybrid_method == 'routing':
                method_bonus = 0.02  # Routing gets bonus for intelligence
            
            overall_score = base_score + hybrid_bonus + method_bonus
            return min(1.0, overall_score)  # Cap at 1.0
            
        except Exception as e:
            self.logger.warning(f"Overall score calculation failed: {e}")
            return 0.0
    
    def get_search_summary(self) -> Dict[str, Any]:
        """Get summary of hybrid search results."""
        if not self.candidates:
            return {'message': 'No search results available'}
        
        try:
            return {
                'total_candidates': len(self.candidates),
                'best_hybrid_method': self.best_candidate.hybrid_method if self.best_candidate else None,
                'best_overall_score': self.best_candidate.overall_score if self.best_candidate else 0.0,
                'tree_performance': {
                    'accuracy': self.best_candidate.tree_architecture.accuracy if self.best_candidate and self.best_candidate.tree_architecture else 0.0,
                    'efficiency': self.best_candidate.tree_architecture.efficiency_score if self.best_candidate and self.best_candidate.tree_architecture else 0.0
                },
                'neural_performance': {
                    'accuracy': self.best_candidate.neural_architecture.accuracy if self.best_candidate and self.best_candidate.neural_architecture else 0.0,
                    'efficiency': self.best_candidate.neural_architecture.efficiency_score if self.best_candidate and self.best_candidate.neural_architecture else 0.0
                }
            }
            
        except Exception as e:
            self.logger.error(f"Search summary generation failed: {e}")
            return {'error': str(e)}


# Convenience function
def search_hybrid_architecture(X_train: np.ndarray, 
                              y_train: np.ndarray,
                              X_val: Optional[np.ndarray] = None,
                              y_val: Optional[np.ndarray] = None,
                              config: Optional[HybridNASConfig] = None,
                              regime_labels: Optional[np.ndarray] = None,
                              data_characteristics: Optional[Dict[str, Any]] = None) -> HybridArchitectureCandidate:
    """
    Convenience function to perform hybrid architecture search.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        config: Hybrid NAS configuration
        regime_labels: Regime labels for regime-aware search (optional)
        data_characteristics: Data characteristics for routing (optional)
        
    Returns:
        Best hybrid architecture candidate
    """
    if config is None:
        config = HybridNASConfig()
    
    hybrid_nas = HybridNASSystem(config)
    return hybrid_nas.search(X_train, y_train, X_val, y_val, regime_labels, data_characteristics)