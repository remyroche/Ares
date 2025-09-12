"""
Small Dataset Management Utilities for SR ML Prediction
Handles the challenge of training ML models with limited SR level data (91 samples)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR, SVC
import warnings
warnings.filterwarnings('ignore')

@dataclass
class SmallDatasetConfig:
    """Configuration for small dataset management."""
    min_samples_for_training: int = 50
    augmentation_factor: float = 2.0  # How much to increase dataset size
    noise_level: float = 0.05  # Gaussian noise for augmentation
    synthetic_ratio: float = 0.3  # Ratio of synthetic to real data
    cross_validation_folds: int = 5
    regularization_strength: float = 1.0
    feature_selection_ratio: float = 0.3  # Keep top 30% of features

class SRAugmentationEngine:
    """Data augmentation engine specifically for SR level data."""
    
    def __init__(self, config: SmallDatasetConfig):
        self.config = config
        
    def augment_sr_data(self, X: np.ndarray, y: np.ndarray, 
                       feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment SR data using multiple strategies:
        1. Gaussian noise injection
        2. Feature interpolation
        3. Bootstrap sampling with replacement
        4. Synthetic SR level generation
        """
        n_samples, n_features = X.shape
        
        if n_samples < self.config.min_samples_for_training:
            return self._create_synthetic_sr_data(X, y, feature_names)
        
        augmented_X = [X]
        augmented_y = [y]
        
        # 1. Gaussian noise augmentation
        noise_augmented = self._add_gaussian_noise(X, y)
        augmented_X.append(noise_augmented[0])
        augmented_y.append(noise_augmented[1])
        
        # 2. Bootstrap sampling
        bootstrap_samples = self._bootstrap_sampling(X, y)
        augmented_X.append(bootstrap_samples[0])
        augmented_y.append(bootstrap_samples[1])
        
        # 3. Feature interpolation for similar SR levels
        interpolated = self._interpolate_similar_levels(X, y)
        if interpolated is not None:
            augmented_X.append(interpolated[0])
            augmented_y.append(interpolated[1])
        
        # 4. Synthetic SR level generation
        synthetic = self._generate_synthetic_sr_levels(X, y, feature_names)
        augmented_X.append(synthetic[0])
        augmented_y.append(synthetic[1])
        
        # Combine all augmented data
        final_X = np.vstack(augmented_X)
        final_y = np.concatenate(augmented_y)
        
        return final_X, final_y
    
    def _add_gaussian_noise(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Add controlled Gaussian noise to existing data."""
        noise_std = np.std(X, axis=0) * self.config.noise_level
        noise = np.random.normal(0, noise_std, X.shape)
        noisy_X = X + noise
        
        # Add smaller noise to targets to maintain realism
        y_noise = np.random.normal(0, np.std(y) * 0.02, y.shape)
        noisy_y = np.clip(y + y_noise, 0, 1)  # Keep quality scores in [0,1]
        
        return noisy_X, noisy_y
    
    def _bootstrap_sampling(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Bootstrap sampling with replacement."""
        n_samples = X.shape[0]
        bootstrap_size = int(n_samples * 0.8)  # 80% of original size
        
        indices = np.random.choice(n_samples, size=bootstrap_size, replace=True)
        return X[indices], y[indices]
    
    def _interpolate_similar_levels(self, X: np.ndarray, y: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Interpolate between similar SR levels."""
        if X.shape[0] < 4:  # Need at least 4 samples for meaningful interpolation
            return None
        
        # Find similar levels using price proximity (assuming first few features are price-related)
        interpolated_X = []
        interpolated_y = []
        
        for i in range(0, X.shape[0] - 1, 2):
            if i + 1 < X.shape[0]:
                # Linear interpolation between two levels
                alpha = 0.5  # Midpoint interpolation
                new_X = alpha * X[i] + (1 - alpha) * X[i + 1]
                new_y = alpha * y[i] + (1 - alpha) * y[i + 1]
                
                interpolated_X.append(new_X)
                interpolated_y.append(new_y)
        
        if interpolated_X:
            return np.array(interpolated_X), np.array(interpolated_y)
        return None
    
    def _generate_synthetic_sr_levels(self, X: np.ndarray, y: np.ndarray, 
                                    feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic SR levels based on learned patterns."""
        n_synthetic = int(len(y) * self.config.synthetic_ratio)
        
        synthetic_X = []
        synthetic_y = []
        
        # Use statistical properties to generate realistic synthetic data
        feature_means = np.mean(X, axis=0)
        feature_stds = np.std(X, axis=0)
        
        for _ in range(n_synthetic):
            # Generate features based on statistical distribution
            synthetic_features = np.random.normal(feature_means, feature_stds * 0.3)
            
            # Apply SR-specific constraints
            synthetic_features = self._apply_sr_constraints(synthetic_features, feature_names)
            
            # Generate realistic quality score based on feature patterns
            quality_score = self._estimate_quality_from_features(synthetic_features, X, y)
            
            synthetic_X.append(synthetic_features)
            synthetic_y.append(quality_score)
        
        return np.array(synthetic_X), np.array(synthetic_y)
    
    def _apply_sr_constraints(self, features: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """Apply SR-specific constraints to synthetic features."""
        constrained_features = features.copy()
        
        for i, feature_name in enumerate(feature_names):
            if 'touch_count' in feature_name.lower():
                constrained_features[i] = max(0, constrained_features[i])  # Non-negative
            elif 'strength' in feature_name.lower() or 'quality' in feature_name.lower():
                constrained_features[i] = np.clip(constrained_features[i], 0, 1)  # [0,1]
            elif 'bounce_rate' in feature_name.lower():
                constrained_features[i] = np.clip(constrained_features[i], 0, 1)  # [0,1]
            elif 'age' in feature_name.lower():
                constrained_features[i] = max(0, constrained_features[i])  # Non-negative
        
        return constrained_features
    
    def _estimate_quality_from_features(self, features: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """Estimate quality score for synthetic features using nearest neighbors."""
        # Simple nearest neighbor estimation
        distances = np.linalg.norm(X - features, axis=1)
        nearest_indices = np.argsort(distances)[:3]  # Top 3 nearest neighbors
        
        # Weighted average of nearest neighbors
        weights = 1.0 / (distances[nearest_indices] + 1e-8)
        weights = weights / np.sum(weights)
        
        estimated_quality = np.sum(weights * y[nearest_indices])
        return np.clip(estimated_quality, 0, 1)
    
    def _create_synthetic_sr_data(self, X: np.ndarray, y: np.ndarray, 
                                feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Create entirely synthetic data when original dataset is too small."""
        n_synthetic = self.config.min_samples_for_training
        
        # Use statistical properties from available data
        if len(X) > 0:
            feature_means = np.mean(X, axis=0)
            feature_stds = np.std(X, axis=0)
            quality_mean = np.mean(y)
            quality_std = np.std(y)
        else:
            # Default values for SR features
            feature_means = np.random.uniform(0.3, 0.7, len(feature_names))
            feature_stds = np.random.uniform(0.1, 0.3, len(feature_names))
            quality_mean = 0.5
            quality_std = 0.2
        
        synthetic_X = []
        synthetic_y = []
        
        for _ in range(n_synthetic):
            features = np.random.normal(feature_means, feature_stds)
            features = self._apply_sr_constraints(features, feature_names)
            
            quality = np.random.normal(quality_mean, quality_std)
            quality = np.clip(quality, 0, 1)
            
            synthetic_X.append(features)
            synthetic_y.append(quality)
        
        return np.array(synthetic_X), np.array(synthetic_y)

class SmallDatasetMLManager:
    """Manages ML training for small datasets with specialized techniques."""
    
    def __init__(self, config: SmallDatasetConfig):
        self.config = config
        self.augmentation_engine = SRAugmentationEngine(config)
        
    def train_with_small_dataset(self, X: np.ndarray, y: np.ndarray, 
                               feature_names: List[str], 
                               task_type: str = 'regression') -> Dict[str, Any]:
        """
        Train ML model with small dataset using specialized techniques.
        
        Args:
            X: Feature matrix
            y: Target values
            feature_names: List of feature names
            task_type: 'regression' or 'classification'
        
        Returns:
            Dictionary with trained models and performance metrics
        """
        results = {
            'original_samples': len(X),
            'augmented_samples': 0,
            'models': {},
            'performance': {},
            'recommendations': []
        }
        
        # 1. Data Augmentation
        if len(X) < self.config.min_samples_for_training:
            X_aug, y_aug = self.augmentation_engine.augment_sr_data(X, y, feature_names)
            results['augmented_samples'] = len(X_aug)
            results['recommendations'].append(f"Applied data augmentation: {len(X)} -> {len(X_aug)} samples")
        else:
            X_aug, y_aug = X, y
        
        # 2. Feature Selection for Small Datasets
        selected_features, feature_importance = self._select_features_for_small_dataset(
            X_aug, y_aug, feature_names
        )
        
        X_selected = X_aug[:, selected_features]
        selected_feature_names = [feature_names[i] for i in selected_features]
        
        # 3. Model Training with Regularization
        models = self._train_regularized_models(X_selected, y_aug, task_type)
        results['models'] = models
        
        # 4. Cross-Validation for Small Datasets
        cv_scores = self._cross_validate_small_dataset(X_selected, y_aug, task_type)
        results['performance'] = cv_scores
        
        # 5. Generate Recommendations
        results['recommendations'].extend(
            self._generate_recommendations(len(X), len(X_aug), cv_scores)
        )
        
        return results
    
    def _select_features_for_small_dataset(self, X: np.ndarray, y: np.ndarray, 
                                         feature_names: List[str]) -> Tuple[List[int], Dict[str, float]]:
        """Select most important features for small dataset."""
        n_features_to_keep = max(10, int(len(feature_names) * self.config.feature_selection_ratio))
        
        # Use correlation and simple statistical measures
        correlations = []
        for i in range(X.shape[1]):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            correlations.append(abs(corr) if not np.isnan(corr) else 0)
        
        # Select top features by correlation
        top_indices = np.argsort(correlations)[-n_features_to_keep:]
        
        feature_importance = {
            feature_names[i]: correlations[i] 
            for i in top_indices
        }
        
        return top_indices.tolist(), feature_importance
    
    def _train_regularized_models(self, X: np.ndarray, y: np.ndarray, 
                                task_type: str) -> Dict[str, Any]:
        """Train multiple regularized models."""
        models = {}
        
        if task_type == 'regression':
            # Ridge Regression (L2 regularization)
            ridge = Ridge(alpha=self.config.regularization_strength)
            ridge.fit(X, y)
            models['ridge'] = ridge
            
            # Lasso Regression (L1 regularization)
            lasso = Lasso(alpha=self.config.regularization_strength * 0.1)
            lasso.fit(X, y)
            models['lasso'] = lasso
            
            # ElasticNet (L1 + L2 regularization)
            elastic = ElasticNet(alpha=self.config.regularization_strength * 0.1, l1_ratio=0.5)
            elastic.fit(X, y)
            models['elastic_net'] = elastic
            
            # Regularized Random Forest
            rf = RandomForestRegressor(
                n_estimators=50,  # Reduced for small dataset
                max_depth=3,      # Strong regularization
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            rf.fit(X, y)
            models['random_forest'] = rf
            
        else:  # Classification
            # SVM with regularization
            svm = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
            svm.fit(X, y)
            models['svm'] = svm
            
            # Regularized Random Forest
            rf = RandomForestClassifier(
                n_estimators=50,
                max_depth=3,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            rf.fit(X, y)
            models['random_forest'] = rf
        
        return models
    
    def _cross_validate_small_dataset(self, X: np.ndarray, y: np.ndarray, 
                                    task_type: str) -> Dict[str, float]:
        """Perform cross-validation optimized for small datasets."""
        cv_scores = {}
        
        if len(X) < 10:
            # Use Leave-One-Out CV for very small datasets
            from sklearn.model_selection import cross_val_score
            cv = LeaveOneOut()
        else:
            # Use Stratified K-Fold with fewer folds
            n_folds = min(3, len(X) // 3)  # At least 3 samples per fold
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Test each model
        models = self._train_regularized_models(X, y, task_type)
        
        for name, model in models.items():
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring='r2' if task_type == 'regression' else 'accuracy')
                cv_scores[name] = {
                    'mean_score': scores.mean(),
                    'std_score': scores.std(),
                    'scores': scores.tolist()
                }
            except Exception as e:
                cv_scores[name] = {'error': str(e)}
        
        return cv_scores
    
    def _generate_recommendations(self, original_size: int, augmented_size: int, 
                                cv_scores: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on dataset size and performance."""
        recommendations = []
        
        if original_size < 50:
            recommendations.append("⚠️ Very small dataset - consider collecting more SR level data")
            recommendations.append("💡 Use ensemble methods with strong regularization")
            recommendations.append("📊 Focus on feature engineering over model complexity")
        
        if augmented_size > original_size * 2:
            recommendations.append("🔧 Data augmentation applied - monitor for overfitting")
        
        # Check for overfitting
        for model_name, scores in cv_scores.items():
            if 'mean_score' in scores and 'std_score' in scores:
                if scores['std_score'] > 0.3:
                    recommendations.append(f"⚠️ High variance in {model_name} - model may be unstable")
        
        recommendations.append("🎯 Consider transfer learning from similar market data")
        recommendations.append("📈 Implement online learning to update model with new SR levels")
        
        return recommendations

# Example usage and integration
def integrate_with_sr_ml_enhancer():
    """Integration example with existing SR ML Enhancer."""
    config = SmallDatasetConfig(
        min_samples_for_training=50,
        augmentation_factor=2.0,
        regularization_strength=1.0
    )
    
    manager = SmallDatasetMLManager(config)
    
    # This would be called from SRMLEnhancer when training with small dataset
    return manager

if __name__ == "__main__":
    # Test the small dataset management
    config = SmallDatasetConfig()
    manager = SmallDatasetMLManager(config)
    
    # Create dummy SR data
    np.random.seed(42)
    X = np.random.randn(91, 50)  # 91 samples, 50 features
    y = np.random.uniform(0, 1, 91)  # Quality scores
    feature_names = [f"feature_{i}" for i in range(50)]
    
    results = manager.train_with_small_dataset(X, y, feature_names)
    
    print("Small Dataset ML Training Results:")
    print(f"Original samples: {results['original_samples']}")
    print(f"Augmented samples: {results['augmented_samples']}")
    print("\nRecommendations:")
    for rec in results['recommendations']:
        print(f"  {rec}")