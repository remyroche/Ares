"""
Regularized Ensemble Methods for SR ML Prediction with Small Datasets
Implements specialized regularization and ensemble techniques for 91 data points
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    VotingRegressor, VotingClassifier
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
import warnings
warnings.filterwarnings('ignore')

@dataclass
class RegularizedEnsembleConfig:
    """Configuration for regularized ensemble methods."""
    n_estimators: int = 50  # Reduced for small datasets
    max_depth: int = 3      # Strong regularization
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    learning_rate: float = 0.05  # Conservative learning rate
    regularization_strength: float = 1.0
    feature_selection_ratio: float = 0.3
    ensemble_methods: List[str] = None  # ['voting', 'stacking', 'bagging']

class RegularizedEnsembleSR:
    """Regularized ensemble for SR prediction with small datasets."""
    
    def __init__(self, config: RegularizedEnsembleConfig):
        self.config = config
        self.ensemble_models = {}
        self.feature_selector = None
        self.scaler = RobustScaler()  # More robust to outliers
        self.feature_names = []
        
    def create_regularized_ensemble(self, X: np.ndarray, y: np.ndarray, 
                                  feature_names: List[str], 
                                  task_type: str = 'regression') -> Dict[str, Any]:
        """
        Create regularized ensemble for small SR datasets.
        
        Args:
            X: Feature matrix
            y: Target values
            feature_names: List of feature names
            task_type: 'regression' or 'classification'
        """
        self.feature_names = feature_names
        
        results = {
            'ensemble_models': {},
            'performance_scores': {},
            'feature_importance': {},
            'recommendations': []
        }
        
        # 1. Feature Selection for Small Datasets
        X_selected, selected_features = self._select_features_for_small_dataset(X, y, task_type)
        results['selected_features'] = [feature_names[i] for i in selected_features]
        
        # 2. Scale Features
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # 3. Create Individual Regularized Models
        individual_models = self._create_individual_models(X_scaled, y, task_type)
        results['individual_models'] = individual_models
        
        # 4. Create Ensemble Models
        ensemble_models = self._create_ensemble_models(X_scaled, y, task_type)
        results['ensemble_models'] = ensemble_models
        
        # 5. Evaluate Performance
        performance = self._evaluate_models(X_scaled, y, individual_models, ensemble_models, task_type)
        results['performance_scores'] = performance
        
        # 6. Generate Recommendations
        results['recommendations'] = self._generate_ensemble_recommendations(
            len(X), performance
        )
        
        return results
    
    def _select_features_for_small_dataset(self, X: np.ndarray, y: np.ndarray, 
                                         task_type: str) -> Tuple[np.ndarray, List[int]]:
        """Select most informative features for small datasets."""
        n_features = X.shape[1]
        n_features_to_select = max(10, int(n_features * self.config.feature_selection_ratio))
        
        # Use statistical tests for feature selection
        if task_type == 'regression':
            selector = SelectKBest(score_func=f_regression, k=n_features_to_select)
        else:
            selector = SelectKBest(score_func=f_classif, k=n_features_to_select)
        
        X_selected = selector.fit_transform(X, y)
        selected_indices = selector.get_support(indices=True)
        
        self.feature_selector = selector
        
        return X_selected, selected_indices.tolist()
    
    def _create_individual_models(self, X: np.ndarray, y: np.ndarray, 
                                task_type: str) -> Dict[str, Any]:
        """Create individual regularized models."""
        models = {}
        
        if task_type == 'regression':
            # Ridge Regression (L2 regularization)
            models['ridge'] = Ridge(
                alpha=self.config.regularization_strength,
                random_state=42
            )
            
            # Lasso Regression (L1 regularization)
            models['lasso'] = Lasso(
                alpha=self.config.regularization_strength * 0.1,
                max_iter=1000,
                random_state=42
            )
            
            # ElasticNet (L1 + L2 regularization)
            models['elastic_net'] = ElasticNet(
                alpha=self.config.regularization_strength * 0.1,
                l1_ratio=0.5,
                max_iter=1000,
                random_state=42
            )
            
            # Regularized Random Forest
            models['random_forest'] = RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_split=self.config.min_samples_split,
                min_samples_leaf=self.config.min_samples_leaf,
                max_features='sqrt',  # Feature subsampling
                bootstrap=True,
                random_state=42
            )
            
            # Regularized Gradient Boosting
            models['gradient_boosting'] = GradientBoostingRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_split=self.config.min_samples_split,
                min_samples_leaf=self.config.min_samples_leaf,
                learning_rate=self.config.learning_rate,
                subsample=0.8,  # Stochastic gradient boosting
                max_features='sqrt',
                random_state=42
            )
            
            # Support Vector Regression with regularization
            models['svr'] = SVR(
                kernel='rbf',
                C=1.0,  # Regularization parameter
                gamma='scale',
                epsilon=0.1
            )
        
        else:  # Classification
            # Logistic Regression with regularization
            models['logistic'] = LogisticRegression(
                C=1.0 / self.config.regularization_strength,
                max_iter=1000,
                random_state=42
            )
            
            # Regularized Random Forest
            models['random_forest'] = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_split=self.config.min_samples_split,
                min_samples_leaf=self.config.min_samples_leaf,
                max_features='sqrt',
                bootstrap=True,
                class_weight='balanced',  # Handle class imbalance
                random_state=42
            )
            
            # Regularized Gradient Boosting
            models['gradient_boosting'] = GradientBoostingClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_split=self.config.min_samples_split,
                min_samples_leaf=self.config.min_samples_leaf,
                learning_rate=self.config.learning_rate,
                subsample=0.8,
                max_features='sqrt',
                random_state=42
            )
            
            # Support Vector Classifier
            models['svc'] = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
        
        return models
    
    def _create_ensemble_models(self, X: np.ndarray, y: np.ndarray, 
                              task_type: str) -> Dict[str, Any]:
        """Create ensemble models."""
        ensembles = {}
        individual_models = self._create_individual_models(X, y, task_type)
        
        if task_type == 'regression':
            # Voting Regressor
            estimators = [(name, model) for name, model in individual_models.items()]
            ensembles['voting_regressor'] = VotingRegressor(estimators)
            
            # Stacking Regressor (simplified for small datasets)
            ensembles['stacking_regressor'] = self._create_stacking_regressor(X, y)
            
        else:  # Classification
            # Voting Classifier
            estimators = [(name, model) for name, model in individual_models.items()]
            ensembles['voting_classifier'] = VotingClassifier(estimators, voting='soft')
            
            # Stacking Classifier
            ensembles['stacking_classifier'] = self._create_stacking_classifier(X, y)
        
        return ensembles
    
    def _create_stacking_regressor(self, X: np.ndarray, y: np.ndarray):
        """Create stacking regressor optimized for small datasets."""
        from sklearn.ensemble import StackingRegressor
        from sklearn.linear_model import LinearRegression
        
        # Base models
        base_models = [
            ('ridge', Ridge(alpha=1.0)),
            ('lasso', Lasso(alpha=0.1, max_iter=1000)),
            ('rf', RandomForestRegressor(n_estimators=30, max_depth=3, random_state=42))
        ]
        
        # Meta-learner (simple linear model to avoid overfitting)
        meta_learner = Ridge(alpha=1.0)
        
        # Use smaller CV for small datasets
        cv_folds = min(3, len(X) // 3)
        
        return StackingRegressor(
            estimators=base_models,
            final_estimator=meta_learner,
            cv=cv_folds,
            stack_method='predict'
        )
    
    def _create_stacking_classifier(self, X: np.ndarray, y: np.ndarray):
        """Create stacking classifier optimized for small datasets."""
        from sklearn.ensemble import StackingClassifier
        from sklearn.linear_model import LogisticRegression
        
        # Base models
        base_models = [
            ('logistic', LogisticRegression(C=1.0, max_iter=1000)),
            ('rf', RandomForestClassifier(n_estimators=30, max_depth=3, random_state=42)),
            ('svc', SVC(kernel='rbf', C=1.0, probability=True, random_state=42))
        ]
        
        # Meta-learner
        meta_learner = LogisticRegression(C=1.0, max_iter=1000)
        
        # Use smaller CV for small datasets
        cv_folds = min(3, len(X) // 3)
        
        return StackingClassifier(
            estimators=base_models,
            final_estimator=meta_learner,
            cv=cv_folds,
            stack_method='predict_proba'
        )
    
    def _evaluate_models(self, X: np.ndarray, y: np.ndarray, 
                        individual_models: Dict[str, Any], 
                        ensemble_models: Dict[str, Any],
                        task_type: str) -> Dict[str, float]:
        """Evaluate all models with cross-validation."""
        performance = {}
        
        # Determine scoring metric and CV strategy
        scoring = 'r2' if task_type == 'regression' else 'accuracy'
        cv_folds = min(3, len(X) // 3)  # Small CV for small datasets
        
        # Evaluate individual models
        for name, model in individual_models.items():
            try:
                scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring)
                performance[name] = {
                    'mean_score': scores.mean(),
                    'std_score': scores.std(),
                    'scores': scores.tolist()
                }
            except Exception as e:
                performance[name] = {'error': str(e)}
        
        # Evaluate ensemble models
        for name, model in ensemble_models.items():
            try:
                scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring)
                performance[name] = {
                    'mean_score': scores.mean(),
                    'std_score': scores.std(),
                    'scores': scores.tolist()
                }
            except Exception as e:
                performance[name] = {'error': str(e)}
        
        return performance
    
    def _generate_ensemble_recommendations(self, n_samples: int, 
                                         performance: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on performance."""
        recommendations = []
        
        if n_samples < 50:
            recommendations.append("⚠️ Very small dataset - ensemble methods may not provide significant benefit")
            recommendations.append("💡 Focus on single best regularized model")
        
        # Find best performing model
        best_model = None
        best_score = -np.inf
        
        for name, scores in performance.items():
            if 'mean_score' in scores and scores['mean_score'] > best_score:
                best_score = scores['mean_score']
                best_model = name
        
        if best_model:
            recommendations.append(f"🏆 Best performing model: {best_model} (score: {best_score:.3f})")
        
        # Check for overfitting
        for name, scores in performance.items():
            if 'std_score' in scores and scores['std_score'] > 0.2:
                recommendations.append(f"⚠️ High variance in {name} - may be overfitting")
        
        recommendations.append("🎯 Consider ensemble only if individual models have similar performance")
        recommendations.append("📊 Monitor model performance on new SR levels")
        
        return recommendations
    
    def predict_ensemble(self, X: np.ndarray, ensemble_name: str = 'voting_regressor') -> np.ndarray:
        """Make prediction using ensemble model."""
        if ensemble_name not in self.ensemble_models:
            raise ValueError(f"Ensemble {ensemble_name} not found")
        
        # Select and scale features
        X_selected = self.feature_selector.transform(X)
        X_scaled = self.scaler.transform(X_selected)
        
        # Make prediction
        prediction = self.ensemble_models[ensemble_name].predict(X_scaled)
        
        return prediction
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from ensemble models."""
        importance = {}
        
        for name, model in self.ensemble_models.items():
            if hasattr(model, 'feature_importances_'):
                importance[name] = dict(zip(
                    self.feature_names, 
                    model.feature_importances_
                ))
            elif hasattr(model, 'estimators_'):
                # For ensemble models, get importance from individual estimators
                estimator_importance = {}
                for i, estimator in enumerate(model.estimators_):
                    if hasattr(estimator, 'feature_importances_'):
                        estimator_importance[f'estimator_{i}'] = dict(zip(
                            self.feature_names,
                            estimator.feature_importances_
                        ))
                importance[name] = estimator_importance
        
        return importance

# Integration function for existing SR ML Enhancer
def integrate_regularized_ensemble_with_sr_enhancer():
    """Integration function for existing SR ML Enhancer."""
    
    config = RegularizedEnsembleConfig(
        n_estimators=50,
        max_depth=3,
        regularization_strength=1.0,
        feature_selection_ratio=0.3
    )
    
    ensemble_manager = RegularizedEnsembleSR(config)
    
    return ensemble_manager

if __name__ == "__main__":
    # Test regularized ensemble
    config = RegularizedEnsembleConfig()
    ensemble = RegularizedEnsembleSR(config)
    
    # Create dummy SR data
    np.random.seed(42)
    X = np.random.randn(91, 50)
    y = np.random.uniform(0, 1, 91)
    feature_names = [f"feature_{i}" for i in range(50)]
    
    # Test ensemble creation
    results = ensemble.create_regularized_ensemble(X, y, feature_names)
    
    print("Regularized Ensemble Results:")
    print(f"Selected features: {len(results['selected_features'])}")
    print(f"Individual models: {len(results['individual_models'])}")
    print(f"Ensemble models: {len(results['ensemble_models'])}")
    
    print("\nPerformance Scores:")
    for name, scores in results['performance_scores'].items():
        if 'mean_score' in scores:
            print(f"  {name}: {scores['mean_score']:.3f} ± {scores['std_score']:.3f}")
    
    print("\nRecommendations:")
    for rec in results['recommendations']:
        print(f"  {rec}")