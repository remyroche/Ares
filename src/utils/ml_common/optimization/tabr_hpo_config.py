"""
Enhanced HPO Configuration for TabR (Tabular Regression) Model

This module defines hyperparameter optimization configurations for TabR model,
which replaces DepthwiseSeparableCNN in Ares trading system.
Integrates with existing hierarchical optimization tools including:
- Parameter grouping and hierarchical optimization
- Successive halving for efficient search
- TPE (Tree-structured Parzen Estimator) optimization
- Pruning and early stopping
- Adaptive parameter importance weighting
"""

from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np

from src.models import ModelType
from src.utils.ml_common.optimization.hierarchical_parameter_optimizer import (
    ParameterGroup,
    OptimizationStage,
    StageConfig,
    HierarchicalParameterOptimizer,
    create_param_group
)
from src.utils.tprint import tprint_info, tprint_warning, tprint_success

class HPOMode(Enum):
    """HPO optimization modes."""
    LIGHT = "light"      # Quick exploration
    STANDARD = "standard"  # Balanced exploration/exploitation
    THOROUGH = "thorough" # Comprehensive search

@dataclass
class TabRConfig:
    """Configuration for TabR model hyperparameters."""
    
    # Core TabR parameters
    k_neighbors: List[int] = None
    learning_rate: List[float] = None
    weight_decay: List[float] = None
    n_encoder_layers: List[int] = None
    n_predictor_layers: List[int] = None
    d_embedding: List[int] = None
    
    # Training parameters
    batch_size: List[int] = None
    max_epochs: List[int] = None
    early_stopping_patience: List[int] = None
    lr_scheduler_patience: List[int] = None
    dropout: List[float] = None
    
    def __post_init__(self):
        """Initialize default values if not provided."""
        if self.k_neighbors is None:
            self.k_neighbors = [32, 64, 96, 128]
        if self.learning_rate is None:
            self.learning_rate = [1e-5, 1e-4, 1e-3]
        if self.weight_decay is None:
            self.weight_decay = [1e-7, 1e-6, 1e-5]
        if self.n_encoder_layers is None:
            self.n_encoder_layers = [0, 1, 2]  # 0 = TabR-S style
        if self.n_predictor_layers is None:
            self.n_predictor_layers = [1, 2, 3]
        if self.d_embedding is None:
            self.d_embedding = [32, 64, 128, 256]
        if self.batch_size is None:
            self.batch_size = [128, 256, 512]
        if self.max_epochs is None:
            self.max_epochs = [100, 200, 300]
        if self.early_stopping_patience is None:
            self.early_stopping_patience = [10, 15, 20]
        if self.lr_scheduler_patience is None:
            self.lr_scheduler_patience = [5, 10, 15]
        if self.dropout is None:
            self.dropout = [0.0, 0.1, 0.2, 0.3]

class TabRHPOManager:
    """Enhanced HPO manager for TabR models with hierarchical optimization integration."""
    
    def __init__(self):
        self.config = TabRConfig()
    
    def create_parameter_groups(self, mode: HPOMode = HPOMode.STANDARD) -> List[ParameterGroup]:
        """
        Create parameter groups for hierarchical optimization.
        
        Organizes TabR parameters into logical groups:
        1. Core Architecture (k_neighbors, embedding dimensions)
        2. Model Structure (encoder/predictor layers)
        3. Training Optimization (learning rate, batch size, epochs)
        4. Regularization (dropout, weight decay, early stopping)
        
        Args:
            mode: HPO mode affecting parameter ranges
            
        Returns:
            List of ParameterGroup objects for hierarchical optimization
        """
        config = self.config
        
        # Adjust parameter ranges based on mode
        if mode == HPOMode.LIGHT:
            k_neighbors_range = [64, 96]
            learning_rate_range = [1e-4]
            d_embedding_range = [64]
            n_layers_range = [0, 1]  # TabR-S preferred
            batch_size_range = [256]
            max_epochs_range = [100]
        elif mode == HPOMode.STANDARD:
            k_neighbors_range = config.k_neighbors
            learning_rate_range = config.learning_rate
            d_embedding_range = config.d_embedding
            n_layers_range = config.n_encoder_layers[:2] + config.n_predictor_layers[:2]
            batch_size_range = config.batch_size
            max_epochs_range = config.max_epochs
        else:  # THOROUGH
            k_neighbors_range = config.k_neighbors + [256, 512]
            learning_rate_range = config.learning_rate + [5e-6, 5e-3]
            d_embedding_range = config.d_embedding + [512]
            n_layers_range = config.n_encoder_layers + config.n_predictor_layers
            batch_size_range = config.batch_size + [64, 1024]
            max_epochs_range = config.max_epochs + [500]
        
        # Group 1: Core Architecture (highest priority)
        core_group = create_param_group(
            name="tabr_core_architecture",
            params={
                "k_neighbors": {
                    "type": "int",
                    "low": min(k_neighbors_range),
                    "high": max(k_neighbors_range),
                    "log": False
                },
                "d_embedding": {
                    "type": "int", 
                    "low": min(d_embedding_range),
                    "high": max(d_embedding_range),
                    "log": True  # Log scale for embedding dimensions
                }
            },
            priority=1,
            description="TabR core architecture: k-neighbors and embedding dimensions"
        )
        
        # Group 2: Model Structure (depends on core)
        structure_group = create_param_group(
            name="tabr_model_structure",
            params={
                "n_encoder_layers": {
                    "type": "int",
                    "low": min(n_layers_range),
                    "high": max(n_layers_range),
                    "log": False
                },
                "n_predictor_layers": {
                    "type": "int",
                    "low": min(config.n_predictor_layers),
                    "high": max(config.n_predictor_layers),
                    "log": False
                }
            },
            priority=2,
            depends_on=["tabr_core_architecture"],
            description="TabR model structure: encoder and predictor layers"
        )
        
        # Group 3: Training Optimization (depends on structure)
        training_group = create_param_group(
            name="tabr_training_optimization",
            params={
                "learning_rate": {
                    "type": "float",
                    "low": min(learning_rate_range),
                    "high": max(learning_rate_range),
                    "log": True  # Log scale for learning rates
                },
                "batch_size": {
                    "type": "int",
                    "low": min(batch_size_range),
                    "high": max(batch_size_range),
                    "log": True  # Log scale for batch sizes
                },
                "max_epochs": {
                    "type": "int",
                    "low": min(max_epochs_range),
                    "high": max(max_epochs_range),
                    "log": False
                }
            },
            priority=3,
            depends_on=["tabr_model_structure"],
            description="TabR training optimization: learning rate, batch size, epochs"
        )
        
        # Group 4: Regularization (depends on training)
        regularization_group = create_param_group(
            name="tabr_regularization",
            params={
                "dropout": {
                    "type": "float",
                    "low": min(config.dropout),
                    "high": max(config.dropout),
                    "log": False
                },
                "weight_decay": {
                    "type": "float",
                    "low": min(config.weight_decay),
                    "high": max(config.weight_decay),
                    "log": True  # Log scale for weight decay
                },
                "early_stopping_patience": {
                    "type": "int",
                    "low": min(config.early_stopping_patience),
                    "high": max(config.early_stopping_patience),
                    "log": False
                },
                "lr_scheduler_patience": {
                    "type": "int",
                    "low": min(config.lr_scheduler_patience),
                    "high": max(config.lr_scheduler_patience),
                    "log": False
                }
            },
            priority=4,
            depends_on=["tabr_training_optimization"],
            description="TabR regularization: dropout, weight decay, early stopping"
        )
        
        return [core_group, structure_group, training_group, regularization_group]
    
    def create_stage_configs(self, mode: HPOMode = HPOMode.STANDARD) -> Dict[OptimizationStage, StageConfig]:
        """
        Create stage configurations for TabR optimization.
        
        Args:
            mode: HPO mode affecting stage configurations
            
        Returns:
            Dictionary mapping stages to configurations
        """
        if mode == HPOMode.LIGHT:
            return {
                OptimizationStage.COARSE_GRID: StageConfig(
                    stage=OptimizationStage.COARSE_GRID,
                    n_trials=20,  # Reduced for light mode
                    grid_points=2,  # Fewer points
                    enable_pruning=False
                ),
                OptimizationStage.FINE_GRID: StageConfig(
                    stage=OptimizationStage.FINE_GRID,
                    n_trials=15,  # Reduced
                    grid_points=3,  # Fewer points
                    enable_pruning=False
                ),
                OptimizationStage.TPE: StageConfig(
                    stage=OptimizationStage.TPE,
                    n_trials=30,  # Reduced
                    n_startup_trials=5,
                    n_ei_candidates=12,
                    enable_pruning=True
                )
            }
        elif mode == HPOMode.STANDARD:
            return {
                OptimizationStage.COARSE_GRID: StageConfig(
                    stage=OptimizationStage.COARSE_GRID,
                    n_trials=50,
                    grid_points=3,
                    enable_pruning=False
                ),
                OptimizationStage.FINE_GRID: StageConfig(
                    stage=OptimizationStage.FINE_GRID,
                    n_trials=50,
                    grid_points=5,
                    enable_pruning=False
                ),
                OptimizationStage.TPE: StageConfig(
                    stage=OptimizationStage.TPE,
                    n_trials=100,
                    n_startup_trials=10,
                    n_ei_candidates=24,
                    enable_pruning=True
                )
            }
        else:  # THOROUGH
            return {
                OptimizationStage.COARSE_GRID: StageConfig(
                    stage=OptimizationStage.COARSE_GRID,
                    n_trials=80,
                    grid_points=4,
                    enable_pruning=False
                ),
                OptimizationStage.FINE_GRID: StageConfig(
                    stage=OptimizationStage.FINE_GRID,
                    n_trials=80,
                    grid_points=7,
                    enable_pruning=False
                ),
                OptimizationStage.TPE: StageConfig(
                    stage=OptimizationStage.TPE,
                    n_trials=200,
                    n_startup_trials=15,
                    n_ei_candidates=32,
                    enable_pruning=True
                ),
                OptimizationStage.BOHB: StageConfig(
                    stage=OptimizationStage.BOHB,
                    n_trials=150,
                    min_budget=1.0,
                    max_budget=27.0,
                    eta=3,
                    enable_pruning=True
                )
            }
    
    def create_hierarchical_optimizer(
        self,
        objective_func: Callable,
        mode: HPOMode = HPOMode.STANDARD,
        cv_folds: int = 5,
        n_rounds: int = 2,
        enable_final_refinement: bool = True,
        random_state: int = 42,
        verbose: bool = True
    ) -> HierarchicalParameterOptimizer:
        """
        Create a fully configured hierarchical optimizer for TabR.
        
        Args:
            objective_func: Objective function for evaluation
            mode: HPO optimization mode
            cv_folds: Cross-validation folds
            n_rounds: Number of optimization rounds
            enable_final_refinement: Whether to enable final refinement
            random_state: Random seed
            verbose: Verbosity level
            
        Returns:
            Configured HierarchicalParameterOptimizer
        """
        tprint_info(f"🔧 Creating TabR hierarchical optimizer (mode: {mode.value})")
        
        # Create parameter groups
        param_groups = self.create_parameter_groups(mode)
        
        # Create stage configurations
        stage_configs = self.create_stage_configs(mode)
        
        # Create stages list
        stages = list(stage_configs.keys())
        
        # Create optimizer
        optimizer = HierarchicalParameterOptimizer(
            param_groups=param_groups,
            objective_func=objective_func,
            stages=stages,
            stage_configs=stage_configs,
            cv_folds=cv_folds,
            scoring_metric='custom_balanced_score',  # Use financial scoring
            direction='maximize',
            n_rounds=n_rounds,
            enable_final_refinement=enable_final_refinement,
            final_refinement_trials=50 if mode == HPOMode.STANDARD else 100,
            random_state=random_state,
            verbose=verbose,
            use_custom_balanced_score=True
        )
        
        tprint_success(f"✅ TabR hierarchical optimizer created with {len(param_groups)} parameter groups")
        return optimizer
    
    def get_search_space(self, mode: HPOMode = HPOMode.STANDARD) -> Dict[str, Any]:
        """
        Get traditional search space for TabR (backward compatibility).
        
        Args:
            mode: HPO mode
            
        Returns:
            Dictionary defining search space
        """
        config = self.config
        
        if mode == HPOMode.LIGHT:
            return {
                'k_neighbors': [64, 96],
                'learning_rate': [1e-4],
                'weight_decay': [1e-6],
                'n_encoder_layers': [0],  # TabR-S
                'n_predictor_layers': [1],
                'd_embedding': [64],
                'batch_size': [256],
                'max_epochs': [100],
                'early_stopping_patience': [10],
                'lr_scheduler_patience': [5],
                'dropout': [0.0, 0.1]
            }
        elif mode == HPOMode.STANDARD:
            return {
                'k_neighbors': config.k_neighbors,
                'learning_rate': config.learning_rate,
                'weight_decay': config.weight_decay,
                'n_encoder_layers': config.n_encoder_layers[:2],  # Limit to smaller values
                'n_predictor_layers': config.n_predictor_layers[:2],
                'd_embedding': config.d_embedding,
                'batch_size': config.batch_size,
                'max_epochs': config.max_epochs[:2],
                'early_stopping_patience': config.early_stopping_patience[:2],
                'lr_scheduler_patience': config.lr_scheduler_patience[:2],
                'dropout': config.dropout
            }
        else:  # THOROUGH
            return {
                'k_neighbors': config.k_neighbors + [256, 512],
                'learning_rate': config.learning_rate + [5e-6, 5e-3],
                'weight_decay': config.weight_decay + [1e-8, 1e-4],
                'n_encoder_layers': config.n_encoder_layers + [3, 4],
                'n_predictor_layers': config.n_predictor_layers + [4, 5],
                'd_embedding': config.d_embedding + [512],
                'batch_size': config.batch_size + [64, 1024],
                'max_epochs': config.max_epochs + [500],
                'early_stopping_patience': config.early_stopping_patience + [30],
                'lr_scheduler_patience': config.lr_scheduler_patience + [20],
                'dropout': config.dropout + [0.4, 0.5]
            }
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        Get default TabR configuration.
        
        Returns:
            Dictionary with default parameters
        """
        return {
            'k_neighbors': 96,
            'learning_rate': 1e-4,
            'weight_decay': 1e-6,
            'n_encoder_layers': 0,  # TabR-S style
            'n_predictor_layers': 1,
            'd_embedding': 64,
            'batch_size': 256,
            'max_epochs': 200,
            'early_stopping_patience': 15,
            'lr_scheduler_patience': 10,
            'dropout': 0.0,
            'use_embeddings': False,
            'verbose': 0,
            'random_state': 42
        }
    
    def get_config_for_data_size(self, n_samples: int, n_features: int) -> Dict[str, Any]:
        """
        Get TabR configuration adapted to data characteristics.
        
        Args:
            n_samples: Number of training samples
            n_features: Number of features
            
        Returns:
            Dictionary with adapted parameters
        """
        config = self.get_default_config()
        
        # Adapt k_neighbors based on data size
        if n_samples < 1000:
            config['k_neighbors'] = min(32, n_samples // 4)
        elif n_samples < 10000:
            config['k_neighbors'] = min(64, n_samples // 10)
        else:
            config['k_neighbors'] = min(128, n_samples // 20)
        
        # Adapt embedding dimension based on features
        if n_features < 10:
            config['d_embedding'] = 32
        elif n_features < 50:
            config['d_embedding'] = 64
        elif n_features < 100:
            config['d_embedding'] = 128
        else:
            config['d_embedding'] = 256
        
        # Adapt batch size based on data size
        if n_samples < 500:
            config['batch_size'] = 32
        elif n_samples < 5000:
            config['batch_size'] = 128
        else:
            config['batch_size'] = 256
        
        # Adapt epochs based on data size
        if n_samples < 1000:
            config['max_epochs'] = 100
        elif n_samples < 10000:
            config['max_epochs'] = 200
        else:
            config['max_epochs'] = 300
        
        return config

# Global HPO manager instance
tabr_hpo_manager = TabRHPOManager()

# Convenience functions
def get_tabr_search_space(mode: Union[str, HPOMode] = HPOMode.STANDARD) -> Dict[str, Any]:
    """Get TabR search space (backward compatibility)."""
    if isinstance(mode, str):
        mode = HPOMode(mode)
    return tabr_hpo_manager.get_search_space(mode)

def get_tabr_default_config() -> Dict[str, Any]:
    """Get default TabR configuration."""
    return tabr_hpo_manager.get_default_config()

def get_tabr_config_for_data_size(n_samples: int, n_features: int) -> Dict[str, Any]:
    """Get TabR configuration adapted to data characteristics."""
    return tabr_hpo_manager.get_config_for_data_size(n_samples, n_features)

def create_tabr_hierarchical_optimizer(
    objective_func: Callable,
    mode: Union[str, HPOMode] = HPOMode.STANDARD,
    cv_folds: int = 5,
    n_rounds: int = 2,
    enable_final_refinement: bool = True,
    random_state: int = 42,
    verbose: bool = True
) -> HierarchicalParameterOptimizer:
    """
    Create a hierarchical optimizer for TabR models.
    
    This is the recommended approach for TabR hyperparameter optimization,
    providing efficient search through parameter grouping and successive halving.
    
    Args:
        objective_func: Objective function for evaluation
        mode: HPO optimization mode
        cv_folds: Cross-validation folds
        n_rounds: Number of optimization rounds
        enable_final_refinement: Whether to enable final refinement
        random_state: Random seed
        verbose: Verbosity level
        
    Returns:
        Configured HierarchicalParameterOptimizer for TabR
    """
    if isinstance(mode, str):
        mode = HPOMode(mode)
    
    return tabr_hpo_manager.create_hierarchical_optimizer(
        objective_func=objective_func,
        mode=mode,
        cv_folds=cv_folds,
        n_rounds=n_rounds,
        enable_final_refinement=enable_final_refinement,
        random_state=random_state,
        verbose=verbose
    )

# Legacy compatibility for DepthWiseCNN
def get_depthwise_cnn_search_space(mode: HPOMode = HPOMode.STANDARD) -> Dict[str, Any]:
    """Get search space for DepthwiseSeparableCNN (deprecated - returns TabR space)."""
    tprint_warning("⚠️ DepthwiseSeparableCNN is deprecated, using TabR search space instead")
    return get_tabr_search_space(mode)

def get_depthwise_cnn_default_config() -> Dict[str, Any]:
    """Get default config for DepthwiseSeparableCNN (deprecated - returns TabR config)."""
    tprint_warning("⚠️ DepthwiseSeparableCNN is deprecated, using TabR config instead")
    return get_tabr_default_config()

__all__ = [
    'HPOMode',
    'TabRConfig',
    'TabRHPOManager',
    'tabr_hpo_manager',
    'get_tabr_search_space',
    'get_tabr_default_config',
    'get_tabr_config_for_data_size',
    'create_tabr_hierarchical_optimizer',
    'get_depthwise_cnn_search_space',
    'get_depthwise_cnn_default_config',
]
