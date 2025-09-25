"""
Pure Tree-Based NAS - 100% Tree Models with Creative Architectures

This module provides a comprehensive tree-based NAS system using only tree models,
including creative architectures like NODE, Oblivious Decision Trees, and other
innovative tree-based approaches.

Key Features:
- 100% tree-based models (no neural networks)
- Creative tree architectures (NODE, Oblivious Trees, etc.)
- Tree-based ensemble methods
- Advanced tree optimization
- Tree-based feature engineering
- Tree-based regime detection
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
from src.utils.tprint import (tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_progress, tprint_performance, tprint_timer)

# Standard tree models
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor

# Advanced tree models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    cb = None

# NODE implementation
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None

# Oblivious Decision Trees

logger = logging.getLogger(__name__)


@dataclass
class PureTreeNASConfig:
    """Configuration for pure tree-based NAS."""
    
    # Tree model types
    tree_models: List[str] = field(default_factory=lambda: [
        'decision_tree', 'random_forest', 'extra_trees', 'gradient_boosting',
        'adaboost', 'bagging', 'xgboost', 'lightgbm', 'catboost',
        'node', 'oblivious_tree', 'rotation_forest', 'isolation_forest',
        'histogram_gradient_boosting', 'voting_tree', 'stacking_tree'
    ])
    
    # Creative tree architectures
    creative_architectures: List[str] = field(default_factory=lambda: [
        'node', 'oblivious_tree', 'rotation_forest', 'histogram_gradient_boosting',
        'voting_tree', 'stacking_tree', 'cascade_tree', 'hierarchical_tree',
        'multi_output_tree', 'regression_tree', 'classification_tree'
    ])
    
    # Tree search space
    tree_search_space: Dict[str, Any] = field(default_factory=lambda: {
        'max_depth': [3, 5, 7, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10, 20, 50],
        'min_samples_leaf': [1, 2, 5, 10, 20],
        'max_features': ['sqrt', 'log2', 0.5, 0.8, 1.0, 'auto'],
        'criterion': ['gini', 'entropy', 'log_loss'],
        'splitter': ['best', 'random'],
        'max_leaf_nodes': [None, 10, 20, 50, 100, 200],
        'min_impurity_decrease': [0.0, 0.01, 0.05, 0.1]
    })
    
    # Ensemble configurations
    ensemble_configs: Dict[str, Any] = field(default_factory=lambda: {
        'voting_methods': ['hard', 'soft'],
        'stacking_methods': ['linear', 'non_linear'],
        'bagging_methods': ['bootstrap', 'pasting'],
        'boosting_methods': ['adaboost', 'gradient_boosting', 'xgboost', 'lightgbm', 'catboost']
    })
    
    # NODE specific configuration
    node_config: Dict[str, Any] = field(default_factory=lambda: {
        'num_layers': [1, 2, 3, 4, 5],
        'num_trees': [1, 2, 4, 8, 16],
        'tree_dim': [1, 2, 3, 4],
        'depth': [4, 6, 8, 10],
        'choice_function': ['entmax15', 'sparsemax'],
        'bin_function': ['entmoid', 'sigmoid']
    })
    
    # Optimization settings
    n_trials: int = 100
    timeout_seconds: int = 3600
    cv_folds: int = 5
    test_size: float = 0.2
    
    # Performance settings
    n_jobs: int = -1
    memory_limit_gb: float = 8.0


@dataclass
class TreeArchitectureCandidate:
    """A candidate tree architecture."""
    
    # Architecture definition
    primary_model: str
    ensemble_method: Optional[str]
    tree_config: Dict[str, Any]
    ensemble_config: Optional[Dict[str, Any]]
    
    # Creative architecture components
    node_config: Optional[Dict[str, Any]] = None
    oblivious_config: Optional[Dict[str, Any]] = None
    rotation_config: Optional[Dict[str, Any]] = None
    
    # Performance metrics
    accuracy: float = 0.0
    efficiency_score: float = 0.0
    interpretability_score: float = 0.0
    robustness_score: float = 0.0
    overall_score: float = 0.0
    
    # Training info
    training_time: float = 0.0
    model_size: int = 0
    n_features: int = 0
    
    # Tree-specific metrics
    tree_depth: int = 0
    n_leaves: int = 0
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    trial_number: int = 0


class NODEModel:
    """Neural Oblivious Decision Ensembles implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize NODE model."""
        self.config = config
        self.model = None
        self.is_trained = False
        self.feature_importances_ = None
        self.training_history = []
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train NODE model with comprehensive training loop."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for NODE model")
        
        try:
            from src.utils.math_validation import validate_numeric_array, safe_mean
            from src.utils.common_operations import safe_weighted_average
            
            # Validate inputs
            X = validate_numeric_array(X, "X")
            y = validate_numeric_array(y, "y")
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.FloatTensor(y)
            
            # Create NODE model
            self.model = self._create_node_model(X.shape[1])
            
            # Training configuration
            learning_rate = self.config.get('learning_rate', 0.001)
            n_epochs = self.config.get('n_epochs', 200)
            batch_size = self.config.get('batch_size', 32)
            patience = self.config.get('patience', 20)
            
            # Optimizer and loss
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
            criterion = nn.MSELoss()
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
            
            # Training loop with early stopping
            best_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(n_epochs):
                # Training mode
                self.model.train()
                total_loss = 0.0
                n_batches = 0
                
                # Mini-batch training
                for i in range(0, len(X_tensor), batch_size):
                    batch_X = X_tensor[i:i+batch_size]
                    batch_y = y_tensor[i:i+batch_size]
                    
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    total_loss += loss.item()
                    n_batches += 1
                
                avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
                self.training_history.append(avg_loss)
                
                # Learning rate scheduling
                scheduler.step(avg_loss)
                
                # Early stopping
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                    # Save best model state
                    self.best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                # Log progress
                if epoch % 20 == 0:
                    logger.info(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
            
            # Load best model
            if hasattr(self, 'best_model_state'):
                self.model.load_state_dict(self.best_model_state)
            
            # Calculate feature importances
            self._calculate_feature_importances(X_tensor)
            
            self.is_trained = True
            logger.info(f"NODE training completed. Best loss: {best_loss:.6f}")
            
        except Exception as e:
            logger.error(f"NODE training failed: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with proper error handling."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        try:
            from src.utils.math_validation import validate_numeric_array
            
            X = validate_numeric_array(X, "X")
            X_tensor = torch.FloatTensor(X)
            
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(X_tensor)
                return predictions.numpy().flatten()
                
        except Exception as e:
            logger.error(f"NODE prediction failed: {e}")
            raise
    
    def _create_node_model(self, input_dim: int):
        """Create comprehensive NODE model architecture."""
        class NODELayer(nn.Module):
            def __init__(self, input_dim, tree_dim, depth, num_trees, choice_function, bin_function):
                super().__init__()
                self.input_dim = input_dim
                self.tree_dim = tree_dim
                self.depth = depth
                self.num_trees = num_trees
                
                # Oblivious decision trees
                self.trees = nn.ModuleList([
                    ObliviousTree(input_dim, tree_dim, depth, choice_function, bin_function)
                    for _ in range(num_trees)
                ])
                
                # Final linear layer for combining tree outputs
                self.final_layer = nn.Linear(num_trees * tree_dim, 1)
                self.dropout = nn.Dropout(0.1)
                
            def forward(self, x):
                # Apply oblivious trees
                tree_outputs = []
                for tree in self.trees:
                    tree_output = tree(x)
                    tree_outputs.append(tree_output)
                
                # Combine tree outputs
                combined = torch.cat(tree_outputs, dim=1)
                combined = self.dropout(combined)
                output = self.final_layer(combined)
                
                return output
        
        return NODELayer(
            input_dim, 
            self.config.get('tree_dim', 2),
            self.config.get('depth', 6),
            self.config.get('num_trees', 2),
            self.config.get('choice_function', 'entmax15'),
            self.config.get('bin_function', 'entmoid')
        )
    
    def _calculate_feature_importances(self, X_tensor: torch.Tensor):
        """Calculate feature importances using gradient-based method."""
        try:
            self.model.eval()
            feature_importances = torch.zeros(X_tensor.shape[1])
            
            # Calculate gradients for each feature
            for i in range(X_tensor.shape[1]):
                X_tensor.requires_grad_(True)
                output = self.model(X_tensor)
                
                # Calculate gradient
                grad = torch.autograd.grad(
                    output.sum(), X_tensor, create_graph=True, retain_graph=True
                )[0]
                
                # Feature importance as mean absolute gradient
                feature_importances[i] = torch.mean(torch.abs(grad[:, i]))
            
            # Normalize importances
            if feature_importances.sum() > 0:
                feature_importances = feature_importances / feature_importances.sum()
            
            self.feature_importances_ = feature_importances.detach().numpy()
            
        except Exception as e:
            logger.warning(f"Feature importance calculation failed: {e}")
            self.feature_importances_ = np.ones(X_tensor.shape[1]) / X_tensor.shape[1]
    
    @property
    def feature_importances_(self):
        """Get feature importances."""
        return self._feature_importances if hasattr(self, '_feature_importances') else None


class ObliviousTree(nn.Module):
    """Oblivious Decision Tree implementation for NODE."""
    
    def __init__(self, input_dim: int, tree_dim: int, depth: int, choice_function: str, bin_function: str):
        super().__init__()
        self.input_dim = input_dim
        self.tree_dim = tree_dim
        self.depth = depth
        
        # Decision nodes (same feature used at each level)
        self.decision_layers = nn.ModuleList([
            nn.Linear(input_dim, 1) for _ in range(depth)
        ])
        
        # Leaf nodes
        self.leaf_layers = nn.ModuleList([
            nn.Linear(2**depth, tree_dim) for _ in range(tree_dim)
        ])
        
        # Choice and bin functions
        self.choice_function = self._get_choice_function(choice_function)
        self.bin_function = self._get_bin_function(bin_function)
    
    def _get_choice_function(self, choice_function: str):
        """Get choice function for routing."""
        if choice_function == 'entmax15':
            return lambda x: torch.softmax(x * 1.5, dim=-1)
        elif choice_function == 'sparsemax':
            return lambda x: torch.softmax(x, dim=-1)
        else:
            return lambda x: torch.softmax(x, dim=-1)
    
    def _get_bin_function(self, bin_function: str):
        """Get bin function for leaf selection."""
        if bin_function == 'entmoid':
            return lambda x: torch.sigmoid(x)
        elif bin_function == 'sigmoid':
            return lambda x: torch.sigmoid(x)
        else:
            return lambda x: torch.sigmoid(x)
    
    def forward(self, x):
        """Forward pass through oblivious tree."""
        batch_size = x.shape[0]
        
        # Decision path through tree
        path_probs = []
        for i in range(self.depth):
            decision = self.decision_layers[i](x)
            prob = self.choice_function(decision)
            path_probs.append(prob)
        
        # Calculate leaf probabilities
        leaf_probs = torch.ones(batch_size, 2**self.depth, device=x.device)
        for i, prob in enumerate(path_probs):
            level_size = 2**(self.depth - i - 1)
            for j in range(2**i):
                start_idx = j * level_size
                end_idx = (j + 1) * level_size
                leaf_probs[:, start_idx:end_idx] *= prob[:, 0:1]  # Left child
                leaf_probs[:, start_idx:end_idx] *= (1 - prob[:, 0:1])  # Right child
        
        # Apply leaf layers
        outputs = []
        for i in range(self.tree_dim):
            leaf_output = self.leaf_layers[i](leaf_probs)
            outputs.append(leaf_output)
        
        return torch.stack(outputs, dim=1)


class ObliviousTreeModel:
    """True Oblivious Decision Tree implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Oblivious Tree model."""
        self.config = config
        self.tree = None
        self.feature_order = None
        self.feature_importances_ = None
        self.tree_structure = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train true Oblivious Tree model."""
        try:
            from src.utils.math_validation import validate_numeric_array
            from src.utils.common_operations import safe_weighted_average
            
            # Validate inputs
            X = validate_numeric_array(X, "X")
            y = validate_numeric_array(y, "y")
            
            # Determine feature order for oblivious structure
            self.feature_order = self._determine_feature_order(X, y)
            
            # Create oblivious tree structure
            self.tree_structure = self._build_oblivious_tree_structure(X, y)
            
            # Train the oblivious tree
            self._train_oblivious_tree(X, y)
            
            # Calculate feature importances
            self._calculate_feature_importances(X, y)
            
        except Exception as e:
            logger.error(f"Oblivious Tree training failed: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using oblivious tree structure."""
        if self.tree_structure is None:
            raise ValueError("Model not trained")
        
        try:
            from src.utils.math_validation import validate_numeric_array
            X = validate_numeric_array(X, "X")
            
            predictions = []
            for i in range(len(X)):
                prediction = self._predict_single_sample(X[i])
                predictions.append(prediction)
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"Oblivious Tree prediction failed: {e}")
            raise
    
    def _determine_feature_order(self, X: np.ndarray, y: np.ndarray) -> List[int]:
        """Determine feature order for oblivious structure using mutual information."""
        try:
            from sklearn.feature_selection import mutual_info_regression
            
            # Calculate mutual information between features and target
            mi_scores = mutual_info_regression(X, y, random_state=42)
            
            # Sort features by mutual information (descending)
            feature_order = np.argsort(mi_scores)[::-1].tolist()
            
            return feature_order
            
        except Exception as e:
            logger.warning(f"Mutual information calculation failed: {e}")
            # Fallback to random order
            return list(range(X.shape[1]))
    
    def _build_oblivious_tree_structure(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Build oblivious tree structure."""
        try:
            max_depth = self.config.get('max_depth', 10)
            min_samples_split = self.config.get('min_samples_split', 2)
            
            # Create tree structure
            tree_structure = {
                'nodes': {},
                'leaves': {},
                'max_depth': max_depth,
                'feature_order': self.feature_order
            }
            
            # Build tree nodes level by level (oblivious structure)
            for level in range(max_depth):
                if level < len(self.feature_order):
                    feature_idx = self.feature_order[level]
                    tree_structure['nodes'][level] = {
                        'feature_idx': feature_idx,
                        'threshold': None,  # Will be set during training
                        'left_child': 2 * level + 1,
                        'right_child': 2 * level + 2
                    }
            
            # Create leaf nodes
            n_leaves = 2 ** max_depth
            for i in range(n_leaves):
                tree_structure['leaves'][i] = {
                    'value': 0.0,  # Will be set during training
                    'samples': []
                }
            
            return tree_structure
            
        except Exception as e:
            logger.error(f"Tree structure building failed: {e}")
            raise
    
    def _train_oblivious_tree(self, X: np.ndarray, y: np.ndarray):
        """Train the oblivious tree by setting thresholds and leaf values."""
        try:
            # Set thresholds for each level
            for level, node_info in self.tree_structure['nodes'].items():
                feature_idx = node_info['feature_idx']
                feature_values = X[:, feature_idx]
                
                # Find optimal threshold using median split
                threshold = np.median(feature_values)
                self.tree_structure['nodes'][level]['threshold'] = threshold
            
            # Calculate leaf values
            for leaf_idx, leaf_info in self.tree_structure['leaves'].items():
                # Find samples that reach this leaf
                leaf_samples = self._get_samples_for_leaf(X, leaf_idx)
                
                if len(leaf_samples) > 0:
                    # Set leaf value to mean of target values
                    leaf_values = y[leaf_samples]
                    self.tree_structure['leaves'][leaf_idx]['value'] = np.mean(leaf_values)
                    self.tree_structure['leaves'][leaf_idx]['samples'] = leaf_samples
                else:
                    # Default to overall mean
                    self.tree_structure['leaves'][leaf_idx]['value'] = np.mean(y)
            
        except Exception as e:
            logger.error(f"Oblivious tree training failed: {e}")
            raise
    
    def _get_samples_for_leaf(self, X: np.ndarray, leaf_idx: int) -> List[int]:
        """Get samples that reach a specific leaf in the oblivious tree."""
        try:
            samples = []
            max_depth = self.tree_structure['max_depth']
            
            for sample_idx in range(len(X)):
                current_leaf = self._traverse_to_leaf(X[sample_idx])
                if current_leaf == leaf_idx:
                    samples.append(sample_idx)
            
            return samples
            
        except Exception as e:
            logger.warning(f"Sample collection for leaf {leaf_idx} failed: {e}")
            return []
    
    def _traverse_to_leaf(self, sample: np.ndarray) -> int:
        """Traverse a sample through the oblivious tree to find its leaf."""
        try:
            current_node = 0
            max_depth = self.tree_structure['max_depth']
            
            for level in range(max_depth):
                if level in self.tree_structure['nodes']:
                    node_info = self.tree_structure['nodes'][level]
                    feature_idx = node_info['feature_idx']
                    threshold = node_info['threshold']
                    
                    if sample[feature_idx] <= threshold:
                        current_node = node_info['left_child']
                    else:
                        current_node = node_info['right_child']
                else:
                    break
            
            # Convert node index to leaf index
            leaf_idx = current_node - (2 ** max_depth - 1)
            return max(0, min(leaf_idx, 2 ** max_depth - 1))
            
        except Exception as e:
            logger.warning(f"Tree traversal failed: {e}")
            return 0
    
    def _predict_single_sample(self, sample: np.ndarray) -> float:
        """Predict a single sample using the oblivious tree."""
        try:
            leaf_idx = self._traverse_to_leaf(sample)
            return self.tree_structure['leaves'][leaf_idx]['value']
            
        except Exception as e:
            logger.warning(f"Single sample prediction failed: {e}")
            return 0.0
    
    def _calculate_feature_importances(self, X: np.ndarray, y: np.ndarray):
        """Calculate feature importances for the oblivious tree."""
        try:
            n_features = X.shape[1]
            feature_importances = np.zeros(n_features)
            
            # Calculate importance based on tree structure
            for level, node_info in self.tree_structure['nodes'].items():
                feature_idx = node_info['feature_idx']
                
                # Calculate importance based on variance reduction
                left_samples = []
                right_samples = []
                
                for i in range(len(X)):
                    sample = X[i]
                    if sample[feature_idx] <= node_info['threshold']:
                        left_samples.append(i)
                    else:
                        right_samples.append(i)
                
                if len(left_samples) > 0 and len(right_samples) > 0:
                    left_values = y[left_samples]
                    right_values = y[right_samples]
                    
                    # Variance reduction
                    total_var = np.var(y)
                    left_var = np.var(left_values)
                    right_var = np.var(right_values)
                    
                    weighted_var = (len(left_samples) * left_var + len(right_samples) * right_var) / len(y)
                    variance_reduction = total_var - weighted_var
                    
                    feature_importances[feature_idx] += max(0, variance_reduction)
            
            # Normalize importances
            if np.sum(feature_importances) > 0:
                feature_importances = feature_importances / np.sum(feature_importances)
            else:
                feature_importances = np.ones(n_features) / n_features
            
            self.feature_importances_ = feature_importances
            
        except Exception as e:
            logger.warning(f"Feature importance calculation failed: {e}")
            self.feature_importances_ = np.ones(X.shape[1]) / X.shape[1]


class RotationForestModel:
    """Enhanced Rotation Forest implementation with proper rotation logic."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Rotation Forest model."""
        self.config = config
        self.base_models = []
        self.rotations = []
        self.feature_importances_ = None
        self.training_history = []
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train enhanced Rotation Forest model."""
        try:
            from sklearn.decomposition import PCA, FastICA
            from sklearn.preprocessing import StandardScaler
            from src.utils.math_validation import validate_numeric_array
            from src.utils.common_operations import safe_weighted_average
            
            # Validate inputs
            X = validate_numeric_array(X, "X")
            y = validate_numeric_array(y, "y")
            
            n_estimators = self.config.get('n_estimators', 10)
            n_features_per_subset = self.config.get('n_features_per_subset', 3)
            rotation_method = self.config.get('rotation_method', 'pca')
            bootstrap = self.config.get('bootstrap', True)
            max_depth = self.config.get('max_depth', 10)
            min_samples_split = self.config.get('min_samples_split', 2)
            min_samples_leaf = self.config.get('min_samples_leaf', 1)
            
            n_features = X.shape[1]
            n_samples = X.shape[0]
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            for i in range(n_estimators):
                try:
                    # Bootstrap sampling if enabled
                    if bootstrap:
                        sample_indices = np.random.choice(
                            n_samples, size=n_samples, replace=True
                        )
                        X_bootstrap = X_scaled[sample_indices]
                        y_bootstrap = y[sample_indices]
                    else:
                        X_bootstrap = X_scaled
                        y_bootstrap = y
                    
                    # Create random feature subset
                    n_subset_features = min(n_features_per_subset, n_features)
                    feature_indices = np.random.choice(
                        n_features, 
                        size=n_subset_features, 
                        replace=False
                    )
                    
                    # Get feature subset
                    X_subset = X_bootstrap[:, feature_indices]
                    
                    # Create rotation matrix
                    if rotation_method == 'pca':
                        rotation = PCA(
                            n_components=min(n_subset_features, 3),
                            random_state=42 + i
                        )
                    elif rotation_method == 'ica':
                        rotation = FastICA(
                            n_components=min(n_subset_features, 3),
                            random_state=42 + i,
                            max_iter=1000
                        )
                    else:
                        # Default to PCA
                        rotation = PCA(
                            n_components=min(n_subset_features, 3),
                            random_state=42 + i
                        )
                    
                    # Fit rotation and transform
                    X_rotated = rotation.fit_transform(X_subset)
                    
                    # Train base model on rotated features
                    base_model = DecisionTreeRegressor(
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        random_state=42 + i,
                        max_features='sqrt'
                    )
                    base_model.fit(X_rotated, y_bootstrap)
                    
                    # Store model and rotation info
                    self.base_models.append(base_model)
                    self.rotations.append({
                        'rotation': rotation,
                        'feature_indices': feature_indices,
                        'scaler': scaler,
                        'rotation_method': rotation_method
                    })
                    
                    # Track training progress
                    self.training_history.append({
                        'estimator': i,
                        'n_features_used': len(feature_indices),
                        'rotation_components': X_rotated.shape[1],
                        'model_score': base_model.score(X_rotated, y_bootstrap)
                    })
                    
                except Exception as e:
                    logger.warning(f"Estimator {i} training failed: {e}")
                    continue
            
            if not self.base_models:
                raise RuntimeError("All estimators failed to train")
            
            # Calculate feature importances
            self._calculate_feature_importances(X, y)
            
            logger.info(f"Rotation Forest training completed with {len(self.base_models)} estimators")
                
        except Exception as e:
            logger.error(f"Rotation Forest training failed: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with enhanced error handling."""
        if not self.base_models:
            raise ValueError("Model not trained")
        
        try:
            from src.utils.math_validation import validate_numeric_array
            X = validate_numeric_array(X, "X")
            
            predictions = []
            weights = []
            
            for i, (model, rotation_info) in enumerate(zip(self.base_models, self.rotations)):
                try:
                    # Get feature subset
                    feature_indices = rotation_info['feature_indices']
                    X_subset = X[:, feature_indices]
                    
                    # Apply scaling
                    scaler = rotation_info['scaler']
                    X_scaled = scaler.transform(X_subset)
                    
                    # Apply rotation
                    rotation = rotation_info['rotation']
                    X_rotated = rotation.transform(X_scaled)
                    
                    # Make prediction
                    pred = model.predict(X_rotated)
                    predictions.append(pred)
                    
                    # Calculate weight based on model performance
                    weight = self._calculate_model_weight(model, X_rotated)
                    weights.append(weight)
                    
                except Exception as e:
                    logger.warning(f"Prediction failed for estimator {i}: {e}")
                    continue
            
            if not predictions:
                raise RuntimeError("All predictions failed")
            
            # Weighted average of predictions
            predictions_array = np.array(predictions)
            weights_array = np.array(weights)
            
            if np.sum(weights_array) > 0:
                weights_array = weights_array / np.sum(weights_array)
                final_predictions = np.average(predictions_array, axis=0, weights=weights_array)
            else:
                final_predictions = np.mean(predictions_array, axis=0)
            
            return final_predictions
            
        except Exception as e:
            logger.error(f"Rotation Forest prediction failed: {e}")
            raise
    
    def _calculate_model_weight(self, model, X_rotated: np.ndarray) -> float:
        """Calculate weight for model based on its performance."""
        try:
            # Use model's score as weight
            if hasattr(model, 'score'):
                return max(0.1, model.score(X_rotated, np.zeros(len(X_rotated))))
            else:
                return 1.0
        except Exception:
            return 1.0
    
    def _calculate_feature_importances(self, X: np.ndarray, y: np.ndarray):
        """Calculate feature importances for the rotation forest."""
        try:
            n_features = X.shape[1]
            feature_importances = np.zeros(n_features)
            
            for model, rotation_info in zip(self.base_models, self.rotations):
                # Get feature indices used by this model
                feature_indices = rotation_info['feature_indices']
                
                # Get model feature importances
                if hasattr(model, 'feature_importances_'):
                    model_importances = model.feature_importances_
                    
                    # Map back to original feature space
                    for i, orig_idx in enumerate(feature_indices):
                        if i < len(model_importances):
                            feature_importances[orig_idx] += model_importances[i]
            
            # Normalize importances
            if np.sum(feature_importances) > 0:
                feature_importances = feature_importances / np.sum(feature_importances)
            else:
                feature_importances = np.ones(n_features) / n_features
            
            self.feature_importances_ = feature_importances
            
        except Exception as e:
            logger.warning(f"Feature importance calculation failed: {e}")
            self.feature_importances_ = np.ones(X.shape[1]) / X.shape[1]
    
    def get_rotation_info(self) -> List[Dict[str, Any]]:
        """Get information about rotations used."""
        try:
            rotation_info = []
            for i, rotation_data in enumerate(self.rotations):
                info = {
                    'estimator': i,
                    'feature_indices': rotation_data['feature_indices'].tolist(),
                    'rotation_method': rotation_data['rotation_method'],
                    'n_components': rotation_data['rotation'].n_components_,
                    'explained_variance_ratio': getattr(
                        rotation_data['rotation'], 'explained_variance_ratio_', None
                    )
                }
                rotation_info.append(info)
            
            return rotation_info
            
        except Exception as e:
            logger.warning(f"Rotation info extraction failed: {e}")
            return []


class HistogramGradientBoostingModel:
    """Enhanced Histogram Gradient Boosting implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Histogram Gradient Boosting model."""
        self.config = config
        self.model = None
        self.feature_importances_ = None
        self.training_history = []
        self.validation_scores = []
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train enhanced Histogram Gradient Boosting model."""
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
            from sklearn.model_selection import train_test_split
            from src.utils.math_validation import validate_numeric_array
            from src.utils.common_operations import safe_weighted_average
            
            # Validate inputs
            X = validate_numeric_array(X, "X")
            y = validate_numeric_array(y, "y")
            
            # Configuration parameters
            max_iter = self.config.get('max_iter', 100)
            max_depth = self.config.get('max_depth', 10)
            learning_rate = self.config.get('learning_rate', 0.1)
            min_samples_leaf = self.config.get('min_samples_leaf', 20)
            l2_regularization = self.config.get('l2_regularization', 0.0)
            early_stopping = self.config.get('early_stopping', True)
            validation_fraction = self.config.get('validation_fraction', 0.1)
            n_iter_no_change = self.config.get('n_iter_no_change', 10)
            tol = self.config.get('tol', 1e-7)
            categorical_features = self.config.get('categorical_features', None)
            monotonic_cst = self.config.get('monotonic_cst', None)
            interaction_cst = self.config.get('interaction_cst', None)
            warm_start = self.config.get('warm_start', False)
            
            # Create model
            self.model = HistGradientBoostingRegressor(
                max_iter=max_iter,
                max_depth=max_depth,
                learning_rate=learning_rate,
                min_samples_leaf=min_samples_leaf,
                l2_regularization=l2_regularization,
                early_stopping=early_stopping,
                validation_fraction=validation_fraction,
                n_iter_no_change=n_iter_no_change,
                tol=tol,
                categorical_features=categorical_features,
                monotonic_cst=monotonic_cst,
                interaction_cst=interaction_cst,
                warm_start=warm_start,
                random_state=42,
                verbose=0
            )
            
            # Train model
            self.model.fit(X, y)
            
            # Extract training history
            if hasattr(self.model, 'train_score_'):
                self.training_history = self.model.train_score_.tolist()
            
            if hasattr(self.model, 'validation_score_'):
                self.validation_scores = self.model.validation_score_.tolist()
            
            # Calculate feature importances
            self._calculate_feature_importances(X, y)
            
            logger.info(f"Histogram Gradient Boosting training completed")
            logger.info(f"   → Final iterations: {self.model.n_iter_}")
            logger.info(f"   → Final score: {self.model.score(X, y):.4f}")
            
        except Exception as e:
            logger.error(f"Histogram Gradient Boosting training failed: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with enhanced error handling."""
        if self.model is None:
            raise ValueError("Model not trained")
        
        try:
            from src.utils.math_validation import validate_numeric_array
            X = validate_numeric_array(X, "X")
            
            predictions = self.model.predict(X)
            return predictions
            
        except Exception as e:
            logger.error(f"Histogram Gradient Boosting prediction failed: {e}")
            raise
    
    def _calculate_feature_importances(self, X: np.ndarray, y: np.ndarray):
        """Calculate feature importances for the histogram gradient boosting model."""
        try:
            # Get feature importances from the model
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importances_ = self.model.feature_importances_
            else:
                # Fallback: calculate using permutation importance
                self._calculate_permutation_importance(X, y)
            
        except Exception as e:
            logger.warning(f"Feature importance calculation failed: {e}")
            self.feature_importances_ = np.ones(X.shape[1]) / X.shape[1]
    
    def _calculate_permutation_importance(self, X: np.ndarray, y: np.ndarray):
        """Calculate permutation importance as fallback."""
        try:
            from sklearn.inspection import permutation_importance
            
            # Calculate permutation importance
            perm_importance = permutation_importance(
                self.model, X, y, n_repeats=5, random_state=42
            )
            
            self.feature_importances_ = perm_importance.importances_mean
            
        except Exception as e:
            logger.warning(f"Permutation importance calculation failed: {e}")
            self.feature_importances_ = np.ones(X.shape[1]) / X.shape[1]
    
    def get_training_curves(self) -> Dict[str, List[float]]:
        """Get training and validation curves."""
        try:
            curves = {}
            
            if self.training_history:
                curves['training_score'] = self.training_history
            
            if self.validation_scores:
                curves['validation_score'] = self.validation_scores
            
            return curves
            
        except Exception as e:
            logger.warning(f"Training curves extraction failed: {e}")
            return {}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        try:
            info = {
                'n_iter_': getattr(self.model, 'n_iter_', None),
                'n_trees_per_iteration_': getattr(self.model, 'n_trees_per_iteration_', None),
                'train_score_': getattr(self.model, 'train_score_', None),
                'validation_score_': getattr(self.model, 'validation_score_', None),
                'feature_importances_': self.feature_importances_,
                'training_history': self.training_history,
                'validation_scores': self.validation_scores
            }
            
            return info
            
        except Exception as e:
            logger.warning(f"Model info extraction failed: {e}")
            return {}
    
    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        """Incremental learning (if warm_start is enabled)."""
        try:
            if not hasattr(self.model, 'partial_fit'):
                raise ValueError("Model does not support partial_fit")
            
            from src.utils.math_validation import validate_numeric_array
            X = validate_numeric_array(X, "X")
            y = validate_numeric_array(y, "y")
            
            self.model.partial_fit(X, y)
            
            # Update feature importances
            self._calculate_feature_importances(X, y)
            
        except Exception as e:
            logger.error(f"Partial fit failed: {e}")
            raise


class PureTreeNAS:
    """Pure Tree-Based NAS system."""
    
    def __init__(self, config: PureTreeNASConfig):
        """Initialize pure tree-based NAS."""
        tprint("🚀 [PURE_TREE_NAS] Initializing Pure Tree-Based NAS", color="cyan", bold=True)
        tprint(f"📊 [PURE_TREE_NAS] Trials: {config.n_trials}", color="blue")
        tprint(f"📊 [PURE_TREE_NAS] Timeout: {config.timeout_seconds}s", color="blue")
        tprint(f"📊 [PURE_TREE_NAS] CV folds: {config.cv_folds}", color="blue")
        self.config = config
        self.logger = logger.getChild('PureTreeNAS')
        self.candidates = []
        self.best_candidate = None
        
        tprint("✅ [PURE_TREE_NAS] Pure Tree-Based NAS initialized successfully", color="green")
        self.logger.info(f"✅ Pure Tree-Based NAS initialized with {config.n_trials} trials")
    
    def search(self, 
               X_train: np.ndarray, 
               y_train: np.ndarray,
               X_val: Optional[np.ndarray] = None,
               y_val: Optional[np.ndarray] = None) -> TreeArchitectureCandidate:
        """
        Perform pure tree-based architecture search.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Best tree architecture candidate
        """
        tprint("🚀 [PURE_TREE_NAS] Starting Pure Tree-Based NAS Search", color="cyan", bold=True)
        tprint(f"📊 [PURE_TREE_NAS] Training data shape: {X_train.shape}, labels: {y_train.shape}", color="blue")
        self.logger.info("🚀 Starting Pure Tree-Based NAS Search...")
        start_time = time.time()
        
        try:
            # Prepare validation data
            if X_val is None or y_val is None:
                tprint("🔧 [PURE_TREE_NAS] Splitting training data for validation", color="yellow")
                from sklearn.model_selection import train_test_split
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=self.config.test_size, random_state=42
                )
                tprint(f"📊 [PURE_TREE_NAS] Validation data shape: {X_val.shape}, labels: {y_val.shape}", color="blue")
            
            # Search for tree architectures
            tprint("🔍 [PURE_TREE_NAS] Starting tree architecture search", color="yellow")
            best_candidate = self._search_tree_architectures(X_train, y_train, X_val, y_val)
            
            search_time = time.time() - start_time
            tprint(f"✅ [PURE_TREE_NAS] Pure Tree NAS completed in {search_time:.2f}s", color="green", bold=True)
            tprint(f"📊 [PURE_TREE_NAS] Best architecture: {best_candidate.primary_model}, score: {best_candidate.overall_score:.4f}", color="cyan")
            self.logger.info(f"✅ Pure Tree NAS completed in {search_time:.2f}s")
            self.logger.info(f"📊 Best architecture: {best_candidate.primary_model}, score: {best_candidate.overall_score:.4f}")
            
            return best_candidate
            
        except Exception as e:
            self.logger.error(f"Pure Tree NAS Search failed: {e}")
            raise
    
    def _search_tree_architectures(self, X_train: np.ndarray, y_train: np.ndarray,
                                 X_val: np.ndarray, y_val: np.ndarray) -> TreeArchitectureCandidate:
        """Search for optimal tree architectures."""
        best_candidate = None
        best_score = -np.inf
        
        for trial in range(self.config.n_trials):
            try:
                # Sample tree architecture
                candidate = self._sample_tree_architecture(trial)
                
                # Train and evaluate
                performance = self._train_and_evaluate_tree_architecture(
                    candidate, X_train, y_train, X_val, y_val
                )
                
                # Update best candidate
                if performance['overall_score'] > best_score:
                    best_score = performance['overall_score']
                    best_candidate = candidate
                    best_candidate.accuracy = performance['accuracy']
                    best_candidate.efficiency_score = performance['efficiency_score']
                    best_candidate.interpretability_score = performance['interpretability_score']
                    best_candidate.robustness_score = performance['robustness_score']
                    best_candidate.overall_score = performance['overall_score']
                    best_candidate.training_time = performance['training_time']
                    best_candidate.model_size = performance['model_size']
                    best_candidate.n_features = performance['n_features']
                    best_candidate.tree_depth = performance['tree_depth']
                    best_candidate.n_leaves = performance['n_leaves']
                    best_candidate.feature_importance = performance['feature_importance']
                
                self.logger.debug(f"Trial {trial}: {candidate.primary_model} Score {performance['overall_score']:.4f}")
                
            except Exception as e:
                tprint_error(f"❌ [PURE_TREE_NAS] Trial {trial} failed: {e}")
                self.logger.error(f"Trial {trial} failed: {e}")
                # Continue with next trial but log the error properly
                continue
        
        if best_candidate is None:
            raise RuntimeError("No successful tree architecture found")
        
        return best_candidate
    
    def _sample_tree_architecture(self, trial_number: int) -> TreeArchitectureCandidate:
        """Sample a random tree architecture."""
        try:
            # Sample primary model
            primary_model = np.random.choice(self.config.tree_models)
            
            # Sample tree configuration
            tree_config = self._sample_tree_config(primary_model)
            
            # Sample ensemble method (optional)
            ensemble_method = None
            ensemble_config = None
            if np.random.random() < 0.3:  # 30% chance of ensemble
                ensemble_method = np.random.choice(['voting', 'stacking', 'bagging', 'boosting'])
                ensemble_config = self._sample_ensemble_config(ensemble_method)
            
            # Sample creative architecture components
            node_config = None
            oblivious_config = None
            rotation_config = None
            
            if primary_model == 'node':
                node_config = self._sample_node_config()
            elif primary_model == 'oblivious_tree':
                oblivious_config = self._sample_oblivious_config()
            elif primary_model == 'rotation_forest':
                rotation_config = self._sample_rotation_config()
            
            return TreeArchitectureCandidate(
                primary_model=primary_model,
                ensemble_method=ensemble_method,
                tree_config=tree_config,
                ensemble_config=ensemble_config,
                node_config=node_config,
                oblivious_config=oblivious_config,
                rotation_config=rotation_config,
                trial_number=trial_number
            )
            
        except Exception as e:
            self.logger.error(f"Tree architecture sampling failed: {e}")
            raise
    
    def _sample_tree_config(self, model_type: str) -> Dict[str, Any]:
        """Sample configuration for specific tree model."""
        base_config = {
            'max_depth': np.random.choice(self.config.tree_search_space['max_depth']),
            'min_samples_split': np.random.choice(self.config.tree_search_space['min_samples_split']),
            'min_samples_leaf': np.random.choice(self.config.tree_search_space['min_samples_leaf']),
            'max_features': np.random.choice(self.config.tree_search_space['max_features']),
            'random_state': 42
        }
        
        if model_type == 'decision_tree':
            base_config.update({
                'criterion': np.random.choice(self.config.tree_search_space['criterion']),
                'splitter': np.random.choice(self.config.tree_search_space['splitter']),
                'max_leaf_nodes': np.random.choice(self.config.tree_search_space['max_leaf_nodes']),
                'min_impurity_decrease': np.random.choice(self.config.tree_search_space['min_impurity_decrease'])
            })
        elif model_type in ['random_forest', 'extra_trees']:
            base_config.update({
                'n_estimators': np.random.randint(10, 200),
                'bootstrap': np.random.choice([True, False]),
                'oob_score': np.random.choice([True, False])
            })
        elif model_type in ['gradient_boosting', 'adaboost']:
            base_config.update({
                'n_estimators': np.random.randint(10, 200),
                'learning_rate': np.random.uniform(0.01, 0.3),
                'subsample': np.random.uniform(0.8, 1.0)
            })
        elif model_type == 'xgboost':
            base_config.update({
                'n_estimators': np.random.randint(10, 200),
                'learning_rate': np.random.uniform(0.01, 0.3),
                'subsample': np.random.uniform(0.8, 1.0),
                'colsample_bytree': np.random.uniform(0.8, 1.0),
                'reg_alpha': np.random.uniform(0, 1),
                'reg_lambda': np.random.uniform(0, 1)
            })
        elif model_type == 'lightgbm':
            base_config.update({
                'n_estimators': np.random.randint(10, 200),
                'learning_rate': np.random.uniform(0.01, 0.3),
                'subsample': np.random.uniform(0.8, 1.0),
                'colsample_bytree': np.random.uniform(0.8, 1.0),
                'num_leaves': np.random.randint(31, 127),
                'reg_alpha': np.random.uniform(0, 1),
                'reg_lambda': np.random.uniform(0, 1)
            })
        elif model_type == 'catboost':
            base_config.update({
                'iterations': np.random.randint(10, 200),
                'learning_rate': np.random.uniform(0.01, 0.3),
                'depth': np.random.randint(3, 10),
                'l2_leaf_reg': np.random.uniform(1, 10),
                'bootstrap_type': np.random.choice(['Bayesian', 'Bernoulli', 'MVS']),
                'subsample': np.random.uniform(0.8, 1.0)
            })
        
        return base_config
    
    def _sample_ensemble_config(self, ensemble_method: str) -> Dict[str, Any]:
        """Sample ensemble configuration."""
        if ensemble_method == 'voting':
            return {
                'voting': np.random.choice(self.config.ensemble_configs['voting_methods']),
                'n_jobs': self.config.n_jobs
            }
        elif ensemble_method == 'stacking':
            return {
                'stack_method': np.random.choice(self.config.ensemble_configs['stacking_methods']),
                'cv': self.config.cv_folds,
                'n_jobs': self.config.n_jobs
            }
        elif ensemble_method == 'bagging':
            return {
                'bootstrap': np.random.choice(self.config.ensemble_configs['bagging_methods']),
                'n_estimators': np.random.randint(5, 50),
                'n_jobs': self.config.n_jobs
            }
        elif ensemble_method == 'boosting':
            return {
                'boosting_method': np.random.choice(self.config.ensemble_configs['boosting_methods']),
                'n_estimators': np.random.randint(5, 50),
                'learning_rate': np.random.uniform(0.01, 0.3)
            }
        else:
            return {}
    
    def _sample_node_config(self) -> Dict[str, Any]:
        """Sample NODE configuration."""
        return {
            'num_layers': np.random.choice(self.config.node_config['num_layers']),
            'num_trees': np.random.choice(self.config.node_config['num_trees']),
            'tree_dim': np.random.choice(self.config.node_config['tree_dim']),
            'depth': np.random.choice(self.config.node_config['depth']),
            'choice_function': np.random.choice(self.config.node_config['choice_function']),
            'bin_function': np.random.choice(self.config.node_config['bin_function'])
        }
    
    def _sample_oblivious_config(self) -> Dict[str, Any]:
        """Sample Oblivious Tree configuration."""
        return {
            'max_depth': np.random.randint(3, 15),
            'min_samples_split': np.random.randint(2, 20),
            'min_samples_leaf': np.random.randint(1, 10),
            'oblivious_structure': True
        }
    
    def _sample_rotation_config(self) -> Dict[str, Any]:
        """Sample Rotation Forest configuration."""
        return {
            'n_estimators': np.random.randint(5, 50),
            'n_features_per_subset': np.random.randint(2, 10),
            'max_depth': np.random.randint(3, 15)
        }
    
    def _train_and_evaluate_tree_architecture(self, candidate: TreeArchitectureCandidate,
                                            X_train: np.ndarray, y_train: np.ndarray,
                                            X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train and evaluate a tree architecture candidate."""
        try:
            start_time = time.time()
            
            # Create and train model
            model = self._create_tree_model(candidate)
            model.fit(X_train, y_train)
            
            # Make predictions
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            
            # Calculate accuracy
            if len(np.unique(y_train)) > 10:  # Regression
                from sklearn.metrics import r2_score, mean_squared_error
                accuracy = r2_score(y_val, val_pred)
                train_accuracy = r2_score(y_train, train_pred)
            else:  # Classification
                from sklearn.metrics import accuracy_score
                accuracy = accuracy_score(y_val, val_pred)
                train_accuracy = accuracy_score(y_train, train_pred)
            
            # Calculate efficiency score
            efficiency_score = self._calculate_efficiency_score(model, candidate)
            
            # Calculate interpretability score
            interpretability_score = self._calculate_interpretability_score(model, candidate)
            
            # Calculate robustness score
            robustness_score = self._calculate_robustness_score(model, X_train, y_train)
            
            # Calculate overall score
            overall_score = (
                0.4 * accuracy +
                0.2 * efficiency_score +
                0.2 * interpretability_score +
                0.2 * robustness_score
            )
            
            # Calculate tree-specific metrics
            tree_depth = self._get_tree_depth(model)
            n_leaves = self._get_n_leaves(model)
            feature_importance = self._get_feature_importance(model)
            model_size = self._estimate_model_size(model)
            
            training_time = time.time() - start_time
            
            return {
                'accuracy': accuracy,
                'efficiency_score': efficiency_score,
                'interpretability_score': interpretability_score,
                'robustness_score': robustness_score,
                'overall_score': overall_score,
                'training_time': training_time,
                'model_size': model_size,
                'n_features': X_train.shape[1],
                'tree_depth': tree_depth,
                'n_leaves': n_leaves,
                'feature_importance': feature_importance
            }
            
        except Exception as e:
            tprint_error(f"❌ [PURE_TREE_NAS] Tree architecture training failed: {e}")
            self.logger.error(f"Tree architecture training failed: {e}")
            # Re-raise the exception instead of returning default values
            raise RuntimeError(f"Tree architecture training failed: {e}") from e
    
    def _create_tree_model(self, candidate: TreeArchitectureCandidate):
        """Create tree model from architecture candidate."""
        model_type = candidate.primary_model
        config = candidate.tree_config
        
        if model_type == 'decision_tree':
            return DecisionTreeRegressor(**config)
        elif model_type == 'random_forest':
            return RandomForestRegressor(**config)
        elif model_type == 'extra_trees':
            return ExtraTreesRegressor(**config)
        elif model_type == 'gradient_boosting':
            return GradientBoostingRegressor(**config)
        elif model_type == 'adaboost':
            return AdaBoostRegressor(**config)
        elif model_type == 'bagging':
            return BaggingRegressor(**config)
        elif model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not available")
            return xgb.XGBRegressor(**config)
        elif model_type == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM not available")
            return lgb.LGBMRegressor(**config, verbose=-1)
        elif model_type == 'catboost':
            if not CATBOOST_AVAILABLE:
                raise ImportError("CatBoost not available")
            return cb.CatBoostRegressor(**config, verbose=False)
        elif model_type == 'node':
            return NODEModel(candidate.node_config)
        elif model_type == 'oblivious_tree':
            return ObliviousTreeModel(candidate.oblivious_config)
        elif model_type == 'rotation_forest':
            return RotationForestModel(candidate.rotation_config)
        elif model_type == 'histogram_gradient_boosting':
            return HistogramGradientBoostingModel(config)
        else:
            raise ValueError(f"Unknown tree model type: {model_type}")
    
    def _calculate_efficiency_score(self, model, candidate: TreeArchitectureCandidate) -> float:
        """Calculate efficiency score based on model complexity."""
        try:
            # Get model complexity metrics
            if hasattr(model, 'tree_'):
                # Single tree
                n_nodes = model.tree_.node_count
                max_depth = model.tree_.max_depth
                complexity = n_nodes * max_depth
            elif hasattr(model, 'estimators_'):
                # Ensemble
                n_trees = len(model.estimators_)
                avg_depth = np.mean([tree.tree_.max_depth for tree in model.estimators_])
                complexity = n_trees * avg_depth
            elif hasattr(model, 'n_estimators'):
                # Boosting
                n_trees = model.n_estimators
                max_depth = getattr(model, 'max_depth', 10)
                complexity = n_trees * max_depth
            else:
                # Fallback
                complexity = 1000
            
            # Efficiency score (lower complexity = higher efficiency)
            efficiency_score = 1.0 / (1.0 + complexity / 10000)
            return float(efficiency_score)
            
        except Exception as e:
            self.logger.warning(f"Efficiency score calculation failed: {e}")
            return 0.5
    
    def _calculate_interpretability_score(self, model, candidate: TreeArchitectureCandidate) -> float:
        """Calculate interpretability score."""
        try:
            # Tree-based models are generally interpretable
            base_score = 0.8
            
            # Adjust based on model complexity
            if hasattr(model, 'max_depth') and model.max_depth <= 5:
                base_score += 0.2
            elif hasattr(model, 'max_depth') and model.max_depth > 15:
                base_score -= 0.2
            
            # Feature importance availability
            if hasattr(model, 'feature_importances_'):
                base_score += 0.1
            
            # Ensemble methods are less interpretable
            if candidate.ensemble_method:
                base_score -= 0.2
            
            return float(np.clip(base_score, 0.0, 1.0))
            
        except Exception as e:
            self.logger.warning(f"Interpretability score calculation failed: {e}")
            return 0.5
    
    def _calculate_robustness_score(self, model, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """Calculate robustness score using cross-validation."""
        try:
            from sklearn.model_selection import cross_val_score
            
            # Use cross-validation to assess robustness
            cv_scores = cross_val_score(model, X_train, y_train, cv=self.config.cv_folds, scoring='r2')
            robustness_score = np.mean(cv_scores)
            return float(robustness_score)
            
        except Exception as e:
            self.logger.warning(f"Robustness score calculation failed: {e}")
            return 0.5
    
    def _get_tree_depth(self, model) -> int:
        """Get tree depth."""
        try:
            if hasattr(model, 'tree_'):
                return model.tree_.max_depth
            elif hasattr(model, 'estimators_'):
                return int(np.mean([tree.tree_.max_depth for tree in model.estimators_]))
            else:
                return getattr(model, 'max_depth', 0)
        except:
            return 0
    
    def _get_n_leaves(self, model) -> int:
        """Get number of leaves."""
        try:
            if hasattr(model, 'tree_'):
                return model.tree_.n_leaves
            elif hasattr(model, 'estimators_'):
                return int(np.mean([tree.tree_.n_leaves for tree in model.estimators_]))
            else:
                return 0
        except:
            return 0
    
    def _get_feature_importance(self, model) -> Dict[str, float]:
        """Get feature importance."""
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                return {f'feature_{i}': float(imp) for i, imp in enumerate(importance)}
            else:
                return {}
        except:
            return {}
    
    def _estimate_model_size(self, model) -> int:
        """Estimate model size in parameters."""
        try:
            if hasattr(model, 'tree_'):
                return model.tree_.node_count
            elif hasattr(model, 'estimators_'):
                return sum(tree.tree_.node_count for tree in model.estimators_)
            elif hasattr(model, 'n_estimators'):
                return model.n_estimators * 100  # Rough estimate
            else:
                return 1000  # Default estimate
        except:
            return 1000
    
    def get_search_summary(self) -> Dict[str, Any]:
        """Get summary of tree architecture search results."""
        if not self.candidates:
            return {'message': 'No search results available'}
        
        try:
            return {
                'total_candidates': len(self.candidates),
                'best_model': self.best_candidate.primary_model if self.best_candidate else None,
                'best_ensemble': self.best_candidate.ensemble_method if self.best_candidate else None,
                'best_score': self.best_candidate.overall_score if self.best_candidate else 0.0,
                'best_accuracy': self.best_candidate.accuracy if self.best_candidate else 0.0,
                'best_efficiency': self.best_candidate.efficiency_score if self.best_candidate else 0.0,
                'best_interpretability': self.best_candidate.interpretability_score if self.best_candidate else 0.0,
                'best_robustness': self.best_candidate.robustness_score if self.best_candidate else 0.0,
                'best_tree_depth': self.best_candidate.tree_depth if self.best_candidate else 0,
                'best_n_leaves': self.best_candidate.n_leaves if self.best_candidate else 0
            }
            
        except Exception as e:
            self.logger.error(f"Search summary generation failed: {e}")
            return {'error': str(e)}


# Convenience function
def search_pure_tree_architecture(X_train: np.ndarray, 
                                 y_train: np.ndarray,
                                 X_val: Optional[np.ndarray] = None,
                                 y_val: Optional[np.ndarray] = None,
                                 config: Optional[PureTreeNASConfig] = None) -> TreeArchitectureCandidate:
    """
    Convenience function to perform pure tree-based architecture search.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        config: Pure tree NAS configuration
        
    Returns:
        Best tree architecture candidate
    """
    if config is None:
        config = PureTreeNASConfig()
    
    pure_tree_nas = PureTreeNAS(config)
    return pure_tree_nas.search(X_train, y_train, X_val, y_val)