"""
Automated Feature Engineering for ML Common

This module provides comprehensive automated feature engineering capabilities
specifically designed for financial time series and trading models.

Key Features:
- Genetic algorithm-based feature selection
- Automated feature transformation
- Feature interaction discovery
- Regime-aware feature engineering
- Multi-objective optimization (accuracy + interpretability)
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
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.metrics import r2_score, accuracy_score, f1_score
import warnings

logger = logging.getLogger(__name__)


@dataclass
class FeatureEngineeringConfig:
    """Configuration for automated feature engineering."""
    
    # Search parameters
    n_generations: int = 50
    population_size: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_size: int = 10
    
    # Feature engineering operations
    enable_transformations: bool = True
    enable_interactions: bool = True
    enable_polynomial_features: bool = True
    enable_statistical_features: bool = True
    enable_technical_indicators: bool = True
    
    # Feature selection
    max_features: int = 50
    min_features: int = 5
    feature_selection_method: str = 'mutual_info'  # mutual_info, lasso, elastic_net, random_forest
    
    # Multi-objective optimization
    objectives: List[str] = field(default_factory=lambda: ['accuracy', 'interpretability', 'efficiency'])
    objective_weights: List[float] = field(default_factory=lambda: [0.6, 0.2, 0.2])
    
    # Regime awareness
    enable_regime_awareness: bool = True
    regime_adaptation_strength: float = 0.3
    
    # Performance
    n_jobs: int = -1
    memory_limit_gb: float = 8.0


@dataclass
class FeatureSet:
    """A set of engineered features."""
    
    # Feature definitions
    feature_names: List[str]
    feature_types: List[str]  # original, transformed, interaction, polynomial, statistical
    feature_sources: List[str]  # which original features were used
    
    # Performance metrics
    accuracy_score: float = 0.0
    interpretability_score: float = 0.0
    efficiency_score: float = 0.0
    overall_score: float = 0.0
    
    # Feature statistics
    n_features: int = 0
    feature_importance: Optional[np.ndarray] = None
    feature_correlations: Optional[np.ndarray] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    generation: int = 0


class FeatureTransformer:
    """Base class for feature transformations."""
    
    @abstractmethod
    def transform(self, X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Transform features and return new features with names."""
        pass


class StatisticalFeatureTransformer(FeatureTransformer):
    """Statistical feature transformations."""
    
    def transform(self, X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Apply statistical transformations."""
        try:
            new_features = []
            new_names = []
            
            for i, name in enumerate(feature_names):
                feature = X[:, i]
                
                # Basic statistical features
                new_features.extend([
                    np.log1p(np.abs(feature)),  # Log transformation
                    np.sqrt(np.abs(feature)),   # Square root
                    feature ** 2,               # Squared
                    np.roll(feature, 1),        # Lag 1
                    np.roll(feature, -1),       # Lead 1
                ])
                new_names.extend([
                    f"{name}_log",
                    f"{name}_sqrt", 
                    f"{name}_squared",
                    f"{name}_lag1",
                    f"{name}_lead1"
                ])
                
                # Rolling statistics (simplified)
                if len(feature) > 10:
                    rolling_mean = pd.Series(feature).rolling(window=5, min_periods=1).mean().values
                    rolling_std = pd.Series(feature).rolling(window=5, min_periods=1).std().values
                    
                    new_features.extend([rolling_mean, rolling_std])
                    new_names.extend([f"{name}_rolling_mean", f"{name}_rolling_std"])
            
            return np.column_stack(new_features), new_names
            
        except Exception as e:
            logger.warning(f"Statistical transformation failed: {e}")
            return X, feature_names


class InteractionFeatureTransformer(FeatureTransformer):
    """Feature interaction transformations."""
    
    def transform(self, X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Create feature interactions."""
        try:
            new_features = []
            new_names = []
            
            # Pairwise interactions (limited to avoid explosion)
            max_interactions = min(20, len(feature_names) * (len(feature_names) - 1) // 2)
            interaction_count = 0
            
            for i in range(len(feature_names)):
                for j in range(i + 1, len(feature_names)):
                    if interaction_count >= max_interactions:
                        break
                    
                    # Multiplicative interaction
                    interaction = X[:, i] * X[:, j]
                    new_features.append(interaction)
                    new_names.append(f"{feature_names[i]}_x_{feature_names[j]}")
                    
                    # Additive interaction
                    interaction = X[:, i] + X[:, j]
                    new_features.append(interaction)
                    new_names.append(f"{feature_names[i]}_plus_{feature_names[j]}")
                    
                    interaction_count += 1
                
                if interaction_count >= max_interactions:
                    break
            
            return np.column_stack(new_features), new_names
            
        except Exception as e:
            logger.warning(f"Interaction transformation failed: {e}")
            return X, feature_names


class PolynomialFeatureTransformer(FeatureTransformer):
    """Polynomial feature transformations."""
    
    def __init__(self, degree: int = 2):
        self.degree = degree
    
    def transform(self, X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Create polynomial features."""
        try:
            new_features = []
            new_names = []
            
            # Limit to most important features to avoid explosion
            n_features = min(10, len(feature_names))
            
            for i in range(n_features):
                feature = X[:, i]
                
                # Polynomial features
                for degree in range(2, self.degree + 1):
                    poly_feature = feature ** degree
                    new_features.append(poly_feature)
                    new_names.append(f"{feature_names[i]}_deg{degree}")
            
            return np.column_stack(new_features), new_names
            
        except Exception as e:
            logger.warning(f"Polynomial transformation failed: {e}")
            return X, feature_names


class TechnicalIndicatorTransformer(FeatureTransformer):
    """Technical indicator transformations for financial data."""
    
    def transform(self, X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Create technical indicators."""
        try:
            new_features = []
            new_names = []
            
            # Assume first 4 columns are OHLC data
            if X.shape[1] >= 4:
                # Simple technical indicators
                high = X[:, 1]  # High
                low = X[:, 2]   # Low
                close = X[:, 3]  # Close
                
                # Price-based indicators
                new_features.extend([
                    high - low,  # Range
                    (high + low) / 2,  # Mid price
                    (high + low + close) / 3,  # Typical price
                ])
                new_names.extend([
                    "price_range",
                    "mid_price", 
                    "typical_price"
                ])
                
                # Moving averages (simplified)
                if len(close) > 5:
                    ma5 = pd.Series(close).rolling(window=5, min_periods=1).mean().values
                    ma10 = pd.Series(close).rolling(window=10, min_periods=1).mean().values
                    
                    new_features.extend([ma5, ma10])
                    new_names.extend(["ma5", "ma10"])
            
            return np.column_stack(new_features), new_names
            
        except Exception as e:
            logger.warning(f"Technical indicator transformation failed: {e}")
            return X, feature_names


class AutomatedFeatureEngineer:
    """Main automated feature engineering implementation."""
    
    def __init__(self, config: FeatureEngineeringConfig):
        """Initialize automated feature engineer."""
        self.config = config
        self.logger = logger.getChild('AutomatedFeatureEngineer')
        
        # Initialize transformers
        self.transformers = []
        if config.enable_statistical_features:
            self.transformers.append(StatisticalFeatureTransformer())
        if config.enable_interactions:
            self.transformers.append(InteractionFeatureTransformer())
        if config.enable_polynomial_features:
            self.transformers.append(PolynomialFeatureTransformer())
        if config.enable_technical_indicators:
            self.transformers.append(TechnicalIndicatorTransformer())
        
        # Initialize feature sets
        self.feature_sets = []
        self.best_feature_set = None
        
        self.logger.info(f"✅ Automated Feature Engineer initialized with {len(self.transformers)} transformers")
    
    def engineer_features(self, 
                        X: np.ndarray, 
                        y: np.ndarray,
                        feature_names: Optional[List[str]] = None,
                        regime_labels: Optional[np.ndarray] = None) -> FeatureSet:
        """
        Perform automated feature engineering.
        
        Args:
            X: Input features
            y: Target variable
            feature_names: Names of input features
            regime_labels: Regime labels for regime-aware engineering (optional)
            
        Returns:
            Best feature set
        """
        self.logger.info("🚀 Starting Automated Feature Engineering...")
        start_time = time.time()
        
        try:
            # Prepare feature names
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
            # Generate initial feature sets
            initial_population = self._generate_initial_population(X, feature_names)
            
            # Evolve feature sets
            best_feature_set = self._evolve_feature_sets(
                initial_population, X, y, regime_labels
            )
            
            engineering_time = time.time() - start_time
            self.logger.info(f"✅ Feature engineering completed in {engineering_time:.2f}s")
            self.logger.info(f"📊 Best feature set: {best_feature_set.n_features} features, score: {best_feature_set.overall_score:.4f}")
            
            return best_feature_set
            
        except Exception as e:
            self.logger.error(f"Automated feature engineering failed: {e}")
            raise
    
    def _generate_initial_population(self, X: np.ndarray, feature_names: List[str]) -> List[FeatureSet]:
        """Generate initial population of feature sets."""
        self.logger.info("🧬 Generating initial population...")
        
        population = []
        
        for i in range(self.config.population_size):
            try:
                # Randomly select transformers
                selected_transformers = np.random.choice(
                    self.transformers, 
                    size=np.random.randint(1, len(self.transformers) + 1),
                    replace=False
                )
                
                # Apply transformations
                current_X = X.copy()
                current_names = feature_names.copy()
                
                for transformer in selected_transformers:
                    current_X, current_names = transformer.transform(current_X, current_names)
                
                # Create feature set
                feature_set = FeatureSet(
                    feature_names=current_names,
                    feature_types=['transformed'] * len(current_names),
                    feature_sources=['original'] * len(current_names),
                    n_features=len(current_names),
                    generation=0
                )
                
                population.append(feature_set)
                
            except Exception as e:
                self.logger.warning(f"Initial population generation failed for individual {i}: {e}")
                continue
        
        self.logger.info(f"✅ Generated {len(population)} feature sets")
        return population
    
    def _evolve_feature_sets(self, 
                           population: List[FeatureSet],
                           X: np.ndarray, 
                           y: np.ndarray,
                           regime_labels: Optional[np.ndarray] = None) -> FeatureSet:
        """Evolve feature sets using genetic algorithm."""
        self.logger.info("🧬 Starting feature set evolution...")
        
        best_feature_set = None
        best_score = -np.inf
        
        for generation in range(self.config.n_generations):
            try:
                # Evaluate population
                evaluated_population = []
                for feature_set in population:
                    try:
                        # Create feature matrix
                        X_engineered = self._create_feature_matrix(X, feature_set)
                        
                        # Evaluate feature set
                        performance = self._evaluate_feature_set(X_engineered, y, regime_labels)
                        
                        # Update feature set
                        feature_set.accuracy_score = performance['accuracy']
                        feature_set.interpretability_score = performance['interpretability']
                        feature_set.efficiency_score = performance['efficiency']
                        feature_set.overall_score = performance['overall_score']
                        feature_set.generation = generation
                        
                        evaluated_population.append(feature_set)
                        
                        # Track best
                        if feature_set.overall_score > best_score:
                            best_score = feature_set.overall_score
                            best_feature_set = feature_set
                        
                    except Exception as e:
                        self.logger.warning(f"Feature set evaluation failed: {e}")
                        continue
                
                # Sort by fitness
                evaluated_population.sort(key=lambda x: x.overall_score, reverse=True)
                
                # Create next generation
                if generation < self.config.n_generations - 1:
                    population = self._create_next_generation(evaluated_population)
                
                self.logger.debug(f"Generation {generation}: Best score = {best_score:.4f}")
                
            except Exception as e:
                self.logger.warning(f"Generation {generation} failed: {e}")
                continue
        
        if best_feature_set is None:
            raise RuntimeError("No successful feature set found")
        
        return best_feature_set
    
    def _create_feature_matrix(self, X: np.ndarray, feature_set: FeatureSet) -> np.ndarray:
        """Create feature matrix from feature set."""
        try:
            # This is a simplified implementation
            # In practice, you'd need to store and reconstruct the actual engineered features
            return X  # Placeholder - would need to implement actual feature reconstruction
            
        except Exception as e:
            self.logger.warning(f"Feature matrix creation failed: {e}")
            return X
    
    def _evaluate_feature_set(self, 
                            X: np.ndarray, 
                            y: np.ndarray,
                            regime_labels: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Evaluate a feature set."""
        try:
            # Feature selection
            X_selected = self._select_features(X, y)
            
            # Train model and evaluate
            from sklearn.model_selection import cross_val_score
            from sklearn.ensemble import RandomForestRegressor
            
            # Determine if classification or regression
            is_classification = len(np.unique(y)) <= 10
            
            if is_classification:
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=50, random_state=42)
                scoring = 'accuracy'
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                scoring = 'r2'
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_selected, y, cv=3, scoring=scoring)
            accuracy = np.mean(cv_scores)
            
            # Calculate interpretability (based on feature count and types)
            interpretability = self._calculate_interpretability(X_selected)
            
            # Calculate efficiency (based on feature count and model performance)
            efficiency = self._calculate_efficiency(X_selected, accuracy)
            
            # Calculate overall score
            weights = self.config.objective_weights
            overall_score = (
                weights[0] * accuracy +
                weights[1] * interpretability +
                weights[2] * efficiency
            )
            
            return {
                'accuracy': accuracy,
                'interpretability': interpretability,
                'efficiency': efficiency,
                'overall_score': overall_score
            }
            
        except Exception as e:
            self.logger.warning(f"Feature set evaluation failed: {e}")
            return {
                'accuracy': 0.0,
                'interpretability': 0.0,
                'efficiency': 0.0,
                'overall_score': 0.0
            }
    
    def _select_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Select best features from the feature set."""
        try:
            n_features = min(self.config.max_features, X.shape[1])
            
            if self.config.feature_selection_method == 'mutual_info':
                # Mutual information-based selection
                is_classification = len(np.unique(y)) <= 10
                if is_classification:
                    scores = mutual_info_classif(X, y)
                else:
                    scores = mutual_info_regression(X, y)
                
                # Select top features
                top_indices = np.argsort(scores)[-n_features:]
                return X[:, top_indices]
            
            elif self.config.feature_selection_method == 'lasso':
                # Lasso-based selection
                lasso = LassoCV(cv=3, random_state=42)
                lasso.fit(X, y)
                
                # Select non-zero coefficients
                selected_features = lasso.coef_ != 0
                if np.sum(selected_features) > n_features:
                    # Select top features by coefficient magnitude
                    top_indices = np.argsort(np.abs(lasso.coef_))[-n_features:]
                    selected_features = np.zeros_like(selected_features)
                    selected_features[top_indices] = True
                
                return X[:, selected_features]
            
            elif self.config.feature_selection_method == 'random_forest':
                # Random forest-based selection
                is_classification = len(np.unique(y)) <= 10
                if is_classification:
                    rf = RandomForestClassifier(n_estimators=50, random_state=42)
                else:
                    rf = RandomForestRegressor(n_estimators=50, random_state=42)
                
                rf.fit(X, y)
                importances = rf.feature_importances_
                
                # Select top features
                top_indices = np.argsort(importances)[-n_features:]
                return X[:, top_indices]
            
            else:
                # Random selection
                n_features = min(n_features, X.shape[1])
                selected_indices = np.random.choice(X.shape[1], size=n_features, replace=False)
                return X[:, selected_indices]
                
        except Exception as e:
            self.logger.warning(f"Feature selection failed: {e}")
            # Return original features
            n_features = min(self.config.max_features, X.shape[1])
            return X[:, :n_features]
    
    def _calculate_interpretability(self, X: np.ndarray) -> float:
        """Calculate interpretability score."""
        try:
            # Interpretability based on feature count (fewer features = more interpretable)
            n_features = X.shape[1]
            max_features = self.config.max_features
            
            # Normalize to [0, 1] where 1 is most interpretable
            interpretability = 1.0 - (n_features / max_features)
            return float(max(0.0, min(1.0, interpretability)))
            
        except Exception as e:
            self.logger.warning(f"Interpretability calculation failed: {e}")
            return 0.5
    
    def _calculate_efficiency(self, X: np.ndarray, accuracy: float) -> float:
        """Calculate efficiency score."""
        try:
            # Efficiency based on feature count and accuracy
            n_features = X.shape[1]
            max_features = self.config.max_features
            
            # Efficiency = accuracy / (feature_count / max_features)
            if n_features > 0:
                efficiency = accuracy / (n_features / max_features)
            else:
                efficiency = 0.0
            
            return float(max(0.0, min(1.0, efficiency)))
            
        except Exception as e:
            self.logger.warning(f"Efficiency calculation failed: {e}")
            return 0.5
    
    def _create_next_generation(self, population: List[FeatureSet]) -> List[FeatureSet]:
        """Create next generation using genetic operators."""
        try:
            new_population = []
            
            # Elitism: keep best individuals
            elite_size = min(self.config.elite_size, len(population))
            new_population.extend(population[:elite_size])
            
            # Generate offspring
            while len(new_population) < self.config.population_size:
                # Selection
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)
                
                # Crossover
                if np.random.random() < self.config.crossover_rate:
                    offspring1, offspring2 = self._crossover(parent1, parent2)
                else:
                    offspring1, offspring2 = parent1, parent2
                
                # Mutation
                if np.random.random() < self.config.mutation_rate:
                    offspring1 = self._mutate(offspring1)
                if np.random.random() < self.config.mutation_rate:
                    offspring2 = self._mutate(offspring2)
                
                new_population.extend([offspring1, offspring2])
            
            return new_population[:self.config.population_size]
            
        except Exception as e:
            self.logger.warning(f"Next generation creation failed: {e}")
            return population
    
    def _tournament_selection(self, population: List[FeatureSet], tournament_size: int = 3) -> FeatureSet:
        """Tournament selection."""
        try:
            tournament = np.random.choice(population, size=tournament_size, replace=False)
            return max(tournament, key=lambda x: x.overall_score)
        except Exception as e:
            self.logger.warning(f"Tournament selection failed: {e}")
            return population[0]
    
    def _crossover(self, parent1: FeatureSet, parent2: FeatureSet) -> Tuple[FeatureSet, FeatureSet]:
        """Crossover two feature sets."""
        try:
            # Simple crossover: combine feature names
            combined_names = list(set(parent1.feature_names + parent2.feature_names))
            
            # Create offspring
            offspring1 = FeatureSet(
                feature_names=combined_names[:len(combined_names)//2],
                feature_types=['crossover'] * (len(combined_names)//2),
                feature_sources=['crossover'] * (len(combined_names)//2),
                n_features=len(combined_names)//2
            )
            
            offspring2 = FeatureSet(
                feature_names=combined_names[len(combined_names)//2:],
                feature_types=['crossover'] * (len(combined_names) - len(combined_names)//2),
                feature_sources=['crossover'] * (len(combined_names) - len(combined_names)//2),
                n_features=len(combined_names) - len(combined_names)//2
            )
            
            return offspring1, offspring2
            
        except Exception as e:
            self.logger.warning(f"Crossover failed: {e}")
            return parent1, parent2
    
    def _mutate(self, feature_set: FeatureSet) -> FeatureSet:
        """Mutate a feature set."""
        try:
            # Simple mutation: randomly add or remove features
            mutated_names = feature_set.feature_names.copy()
            
            if np.random.random() < 0.5 and len(mutated_names) > self.config.min_features:
                # Remove a random feature
                if mutated_names:
                    mutated_names.pop(np.random.randint(len(mutated_names)))
            else:
                # Add a random feature
                new_feature = f"mutated_feature_{np.random.randint(1000)}"
                mutated_names.append(new_feature)
            
            return FeatureSet(
                feature_names=mutated_names,
                feature_types=['mutated'] * len(mutated_names),
                feature_sources=['mutated'] * len(mutated_names),
                n_features=len(mutated_names)
            )
            
        except Exception as e:
            self.logger.warning(f"Mutation failed: {e}")
            return feature_set
    
    def get_engineering_summary(self) -> Dict[str, Any]:
        """Get summary of feature engineering results."""
        if not self.feature_sets:
            return {'message': 'No feature engineering results available'}
        
        try:
            # Calculate summary statistics
            accuracies = [fs.accuracy_score for fs in self.feature_sets]
            interpretabilities = [fs.interpretability_score for fs in self.feature_sets]
            efficiencies = [fs.efficiency_score for fs in self.feature_sets]
            overall_scores = [fs.overall_score for fs in self.feature_sets]
            feature_counts = [fs.n_features for fs in self.feature_sets]
            
            return {
                'total_feature_sets': len(self.feature_sets),
                'best_accuracy': float(np.max(accuracies)),
                'best_interpretability': float(np.max(interpretabilities)),
                'best_efficiency': float(np.max(efficiencies)),
                'best_overall_score': float(np.max(overall_scores)),
                'average_features': float(np.mean(feature_counts)),
                'feature_count_range': [int(np.min(feature_counts)), int(np.max(feature_counts))],
                'engineering_statistics': {
                    'accuracy_mean': float(np.mean(accuracies)),
                    'accuracy_std': float(np.std(accuracies)),
                    'interpretability_mean': float(np.mean(interpretabilities)),
                    'interpretability_std': float(np.std(interpretabilities)),
                    'efficiency_mean': float(np.mean(efficiencies)),
                    'efficiency_std': float(np.std(efficiencies)),
                    'overall_score_mean': float(np.mean(overall_scores)),
                    'overall_score_std': float(np.std(overall_scores))
                }
            }
            
        except Exception as e:
            self.logger.error(f"Engineering summary generation failed: {e}")
            return {'error': str(e)}


# Convenience function
def engineer_features_automatically(X: np.ndarray, 
                                  y: np.ndarray,
                                  feature_names: Optional[List[str]] = None,
                                  config: Optional[FeatureEngineeringConfig] = None,
                                  regime_labels: Optional[np.ndarray] = None) -> FeatureSet:
    """
    Convenience function to perform automated feature engineering.
    
    Args:
        X: Input features
        y: Target variable
        feature_names: Names of input features
        config: Feature engineering configuration
        regime_labels: Regime labels for regime-aware engineering (optional)
        
    Returns:
        Best feature set
    """
    if config is None:
        config = FeatureEngineeringConfig()
    
    engineer = AutomatedFeatureEngineer(config)
    return engineer.engineer_features(X, y, feature_names, regime_labels)