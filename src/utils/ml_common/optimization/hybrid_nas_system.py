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

# Import existing neural NAS
from .neural_architecture_search import NeuralArchitectureSearch, ArchitectureConfig, ArchitectureCandidate

# Import tree-based NAS
from .tree_based_architecture_search import TreeBasedArchitectureSearch, TreeArchitectureConfig, TreeArchitectureCandidate

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
        tprint("🧠 [HYBRID_NAS] Initializing neural NAS system", color="yellow")
        self.neural_nas = NeuralArchitectureSearch(config.neural_config)
        tprint("🌳 [HYBRID_NAS] Initializing tree-based NAS system", color="yellow")
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
        """Analyze data characteristics to guide routing decisions."""
        try:
            n_samples, n_features = X_train.shape

            # Calculate tabular vs sequential ratio
            tabular_ratio = self._calculate_tabular_ratio(X_train)
            sequential_ratio = self._calculate_sequential_ratio(X_train)
            complexity_ratio = self._calculate_complexity_ratio(X_train)

            # Calculate data sparsity
            sparsity = self._calculate_sparsity(X_train)

            # Calculate feature importance variance
            feature_variance = np.var(X_train, axis=0)
            feature_importance_variance = np.var(feature_variance)

            characteristics = {
                'n_samples': n_samples,
                'n_features': n_features,
                'tabular_ratio': tabular_ratio,
                'sequential_ratio': sequential_ratio,
                'complexity_ratio': complexity_ratio,
                'sparsity': sparsity,
                'feature_importance_variance': feature_importance_variance,
                'is_tabular_dominant': tabular_ratio > self.config.routing_rules['tabular_threshold'],
                'is_sequential_dominant': sequential_ratio > self.config.routing_rules['sequential_threshold'],
                'is_complex_dominant': complexity_ratio > self.config.routing_rules['complexity_threshold']
            }

            self.logger.debug(f"Data characteristics: {characteristics}")
            return characteristics

        except Exception as e:
            self.logger.warning(f"Data analysis failed: {e}")
            return {'n_samples': X_train.shape[0], 'n_features': X_train.shape[1]}

    def _calculate_tabular_ratio(self, X: np.ndarray) -> float:
        """Calculate ratio of tabular features."""
        # Simple heuristic: ratio of features that are not highly correlated with time
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
                self.logger.warning(f"Trial {trial} failed: {e}")
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
        """Extract selected features from tree architecture."""
        try:
            if hasattr(tree_architecture, 'n_features') and tree_architecture.n_features:
                # Return first n_features indices (simplified)
                return list(range(min(tree_architecture.n_features, 50)))
            else:
                return []
        except:
            return []

    def _evaluate_hybrid_architecture(self, hybrid_candidate: HybridArchitectureCandidate,
                                     X_train: np.ndarray, y_train: np.ndarray,
                                     X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """Evaluate hybrid architecture performance."""
        try:
            # Get individual performances
            tree_performance = {
                'accuracy': hybrid_candidate.tree_architecture.accuracy if hybrid_candidate.tree_architecture else 0.0,
                'efficiency': hybrid_candidate.tree_architecture.efficiency_score if hybrid_candidate.tree_architecture else 0.0,
                'interpretability': hybrid_candidate.tree_architecture.interpretability_score if hybrid_candidate.tree_architecture else 0.0,
                'robustness': hybrid_candidate.tree_architecture.robustness_score if hybrid_candidate.tree_architecture else 0.0
            }

            neural_performance = {
                'accuracy': hybrid_candidate.neural_architecture.accuracy if hybrid_candidate.neural_architecture else 0.0,
                'efficiency': hybrid_candidate.neural_architecture.efficiency_score if hybrid_candidate.neural_architecture else 0.0,
                'interpretability': 0.3,  # Neural networks are less interpretable
                'robustness': hybrid_candidate.neural_architecture.robustness_score if hybrid_candidate.neural_architecture else 0.0
            }

            # Combine performances based on hybrid method
            if hybrid_candidate.hybrid_method == 'ensemble':
                # Weighted average
                weights = self.config.ensemble_weights
                combined_accuracy = weights[0] * tree_performance['accuracy'] + weights[1] * neural_performance['accuracy']
                combined_efficiency = weights[0] * tree_performance['efficiency'] + weights[1] * neural_performance['efficiency']
                combined_interpretability = weights[0] * tree_performance['interpretability'] + weights[1] * neural_performance['interpretability']
                combined_robustness = weights[0] * tree_performance['robustness'] + weights[1] * neural_performance['robustness']
            else:
                # Use the best performing approach
                combined_accuracy = max(tree_performance['accuracy'], neural_performance['accuracy'])
                combined_efficiency = max(tree_performance['efficiency'], neural_performance['efficiency'])
                combined_interpretability = max(tree_performance['interpretability'], neural_performance['interpretability'])
                combined_robustness = max(tree_performance['robustness'], neural_performance['robustness'])

            # Calculate overall score
            overall_score = (
                0.4 * combined_accuracy +
                0.2 * combined_efficiency +
                0.2 * combined_interpretability +
                0.2 * combined_robustness
            )

            return {
                'accuracy': combined_accuracy,
                'efficiency': combined_efficiency,
                'interpretability': combined_interpretability,
                'robustness': combined_robustness,
                'overall_score': overall_score
            }

        except Exception as e:
            self.logger.warning(f"Hybrid evaluation failed: {e}")
            return {
                'accuracy': 0.0,
                'efficiency': 0.0,
                'interpretability': 0.0,
                'robustness': 0.0,
                'overall_score': 0.0
            }

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
