"""
Tree Utilities for TAS Tree Architecture

Utility functions and classes for tree-based models and architectures.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
# DecisionTreeClassifier removed - only advanced tree models supported
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


@dataclass
class TreeConfig:
    """Configuration for tree models."""
    max_depth: int = 10
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: Union[str, int, float] = 'auto'
    random_state: int = 42


class TreeUtils:
    """Utility functions for tree models."""
    
    def __init__(self, config: TreeConfig = None):
        self.config = config or TreeConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def create_tree_classifier(self, config: TreeConfig = None):
        """Create a tree classifier - DecisionTreeClassifier removed, use RandomForest instead."""
        tprint_debug("Creating tree classifier (using RandomForest as DecisionTree replacement)")
        tprint_debug(f"Config: {config}")
        
        config = config or self.config
        
        tprint_debug(f"Max depth: {config.max_depth}")
        tprint_debug(f"Min samples split: {config.min_samples_split}")
        tprint_debug(f"Min samples leaf: {config.min_samples_leaf}")
        tprint_debug(f"Max features: {config.max_features}")
        tprint_debug(f"Random state: {config.random_state}")
        
        # Use RandomForest as replacement for DecisionTree
        classifier = RandomForestClassifier(
            n_estimators=1,  # Single tree equivalent
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            max_features=config.max_features,
            random_state=config.random_state
        )
        
        tprint_debug(f"Tree classifier created successfully: {type(classifier)}")
        
        return classifier
    
    def create_tree_regressor(self, config: TreeConfig = None):
        """Create a tree regressor - DecisionTreeRegressor removed, use RandomForest instead."""
        tprint_debug("Creating tree regressor (using RandomForest as DecisionTree replacement)")
        tprint_debug(f"Config: {config}")
        
        config = config or self.config
        
        tprint_debug(f"Max depth: {config.max_depth}")
        tprint_debug(f"Min samples split: {config.min_samples_split}")
        tprint_debug(f"Min samples leaf: {config.min_samples_leaf}")
        tprint_debug(f"Max features: {config.max_features}")
        tprint_debug(f"Random state: {config.random_state}")
        
        # Use RandomForest as replacement for DecisionTree
        regressor = RandomForestRegressor(
            n_estimators=1,  # Single tree equivalent
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            max_features=config.max_features,
            random_state=config.random_state
        )
        
        tprint_debug(f"Tree regressor created successfully: {type(regressor)}")
        
        return regressor
    
    def create_random_forest_classifier(self, n_estimators: int = 100, config: TreeConfig = None) -> RandomForestClassifier:
        """Create a random forest classifier."""
        config = config or self.config
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            max_features=config.max_features,
            random_state=config.random_state
        )
    
    def create_random_forest_regressor(self, n_estimators: int = 100, config: TreeConfig = None) -> RandomForestRegressor:
        """Create a random forest regressor."""
        config = config or self.config
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            max_features=config.max_features,
            random_state=config.random_state
        )
    
    def evaluate_tree_model(self, model, X_test: np.ndarray, y_test: np.ndarray, task_type: str = 'classification') -> Dict[str, float]:
        """Evaluate a tree model."""
        try:
            y_pred = model.predict(X_test)
            
            if task_type == 'classification':
                accuracy = accuracy_score(y_test, y_pred)
                return {'accuracy': accuracy}
            else:
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                return {'mse': mse, 'r2': r2}
                
        except Exception as e:
            self.logger.error(f"Tree model evaluation failed: {e}")
            return {}
    
    def get_tree_depth(self, model) -> int:
        """Get the depth of a tree model."""
        try:
            if hasattr(model, 'tree_'):
                return model.tree_.max_depth
            elif hasattr(model, 'estimators_'):
                # For ensemble models, return average depth
                depths = [tree.tree_.max_depth for tree in model.estimators_]
                return int(np.mean(depths))
            else:
                return 0
        except Exception as e:
            self.logger.warning(f"Could not get tree depth: {e}")
            return 0
    
    def get_feature_importance(self, model) -> np.ndarray:
        """Get feature importance from a tree model."""
        try:
            if hasattr(model, 'feature_importances_'):
                return model.feature_importances_
            else:
                return np.array([])
        except Exception as e:
            self.logger.warning(f"Could not get feature importance: {e}")
            return np.array([])


class TreeArchitectureUtils:
    """Utility functions for tree architectures."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def create_architecture_candidate(self, 
                                   model_type: str = 'random_forest',
                                   n_estimators: int = 100,
                                   max_depth: int = 10,
                                   min_samples_split: int = 2,
                                   min_samples_leaf: int = 1,
                                   max_features: Union[str, int, float] = 'auto') -> Dict[str, Any]:
        """Create a tree architecture candidate."""
        return {
            'model_type': model_type,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features
        }
    
    def validate_architecture(self, architecture: Dict[str, Any]) -> bool:
        """Validate a tree architecture."""
        try:
            # Check required parameters
            required_params = ['model_type', 'max_depth', 'min_samples_split', 'min_samples_leaf']
            for param in required_params:
                if param not in architecture:
                    return False
            
            # Check parameter ranges
            if architecture['max_depth'] < 1:
                return False
            if architecture['min_samples_split'] < 2:
                return False
            if architecture['min_samples_leaf'] < 1:
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Architecture validation failed: {e}")
            return False
    
    def compare_architectures(self, arch1: Dict[str, Any], arch2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two tree architectures."""
        try:
            comparison = {}
            
            for key in arch1.keys():
                if key in arch2:
                    comparison[key] = {
                        'arch1': arch1[key],
                        'arch2': arch2[key],
                        'different': arch1[key] != arch2[key]
                    }
            
            return comparison
            
        except Exception as e:
            self.logger.warning(f"Architecture comparison failed: {e}")
            return {}
    
    def optimize_architecture(self, 
                             base_architecture: Dict[str, Any],
                             optimization_params: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate optimized architecture variants."""
        try:
            variants = []
            
            for param, values in optimization_params.items():
                if param in base_architecture:
                    for value in values:
                        variant = base_architecture.copy()
                        variant[param] = value
                        variants.append(variant)
            
            return variants
            
        except Exception as e:
            self.logger.warning(f"Architecture optimization failed: {e}")
            return []


class TreeModelUtils:
    """Utility functions for tree model operations."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def train_tree_model(self, 
                        model, 
                        X_train: np.ndarray, 
                        y_train: np.ndarray,
                        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, Any]:
        """Train a tree model with optional validation."""
        try:
            # Train the model
            model.fit(X_train, y_train)
            
            # Get training metrics
            train_score = model.score(X_train, y_train)
            
            result = {
                'model': model,
                'train_score': train_score,
                'training_successful': True
            }
            
            # Add validation metrics if provided
            if validation_data is not None:
                X_val, y_val = validation_data
                val_score = model.score(X_val, y_val)
                result['val_score'] = val_score
            
            return result
            
        except Exception as e:
            self.logger.error(f"Tree model training failed: {e}")
            return {
                'model': model,
                'training_successful': False,
                'error': str(e)
            }
    
    def predict_with_confidence(self, 
                               model, 
                               X: np.ndarray, 
                               return_proba: bool = True) -> Dict[str, Any]:
        """Make predictions with confidence scores."""
        try:
            predictions = model.predict(X)
            
            result = {
                'predictions': predictions
            }
            
            # Add probability scores if available
            if return_proba and hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)
                result['probabilities'] = probabilities
                
                # Calculate confidence as max probability
                if len(probabilities.shape) > 1:
                    confidence = np.max(probabilities, axis=1)
                    result['confidence'] = confidence
            
            return result
            
        except Exception as e:
            self.logger.error(f"Tree model prediction failed: {e}")
            return {
                'predictions': None,
                'error': str(e)
            }
    
    def get_model_complexity(self, model) -> Dict[str, Any]:
        """Get model complexity metrics."""
        try:
            complexity = {}
            
            # Tree depth
            if hasattr(model, 'tree_'):
                complexity['depth'] = model.tree_.max_depth
                complexity['n_leaves'] = model.tree_.n_leaves
                complexity['n_nodes'] = model.tree_.node_count
            elif hasattr(model, 'estimators_'):
                # For ensemble models
                depths = [tree.tree_.max_depth for tree in model.estimators_]
                leaves = [tree.tree_.n_leaves for tree in model.estimators_]
                nodes = [tree.tree_.node_count for tree in model.estimators_]
                
                complexity['avg_depth'] = np.mean(depths)
                complexity['avg_leaves'] = np.mean(leaves)
                complexity['avg_nodes'] = np.mean(nodes)
                complexity['n_estimators'] = len(model.estimators_)
            
            return complexity
            
        except Exception as e:
            self.logger.warning(f"Could not get model complexity: {e}")
            return {}
    
    def prune_tree(self, model, X_val: np.ndarray, y_val: np.ndarray) -> Any:
        """Prune a tree model using validation data."""
        try:
            # This is a simplified pruning - in practice would use cost complexity pruning
            if hasattr(model, 'tree_'):
                # For single trees, we can't easily prune after training
                # This would typically be done during training
                return model
            else:
                # For ensemble models, return as-is
                return model
                
        except Exception as e:
            self.logger.warning(f"Tree pruning failed: {e}")
            return model


# Convenience functions
def create_tree_utils(config: TreeConfig = None) -> TreeUtils:
    """Create tree utilities with default configuration."""
    return TreeUtils(config)


def create_architecture_utils() -> TreeArchitectureUtils:
    """Create architecture utilities."""
    return TreeArchitectureUtils()


def create_model_utils() -> TreeModelUtils:
    """Create model utilities."""
    return TreeModelUtils()
