"""
HPO Configuration Module for Model Training.

This module provides:
1. Parameter group definitions for all model types (LGBM, CatBoost, TCN, GRU, ExtraTrees)
2. Integration with custom_balanced_score from evaluation_metrics.py
3. YAML configuration updater to save optimal parameters back to config files
4. Hierarchical optimization support for base and ensemble models
"""

import yaml
from typing import Dict, Any, List, Optional, cast
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

from src.utils.ml_common.optimization.hierarchical_parameter_optimizer import (
    HierarchicalParameterOptimizer,
    ParameterGroup,
    OptimizationStage,
    HierarchicalOptimizationResult
)
from src.utils.ml_common.optimization.shared_utils.evaluation_metrics import (
    UnifiedEvaluator,
    create_unified_evaluator
)
from src.utils.logger import system_logger

logger = system_logger.getChild('HPOConfig')


# ============================================================================
# Parameter Derivation Functions
# ============================================================================

def derive_dependent_parameters(params: Dict[str, Any], model_type: str) -> Dict[str, Any]:
    """
    Derive dependent parameters based on model type and optimized parameters.
    
    This function implements the parameter tying rules:
    1. LGBM: num_leaves = 2^max_depth ± 2
    2. TCN: batch_size = num_filters
    3. GRU: batch_size = 2 * hidden_units
    4. ExtraTrees: min_samples_split = 2 * min_samples_leaf
    5. All tree models: subsample = colsample_* = sampling_rate
    
    Args:
        params: Dictionary of optimized parameters
        model_type: Type of model ('lgbm', 'catboost', 'tcn', 'gru', 'extratrees', etc.)
    
    Returns:
        Complete parameter dictionary with derived parameters added
    """
    import random
    
    params = params.copy()  # Don't modify original
    model_type_lower = model_type.lower()
    
    # 1. LGBM: Derive num_leaves from max_depth
    if 'lgbm' in model_type_lower or 'meta' in model_type_lower or 'stacker' in model_type_lower:
        if 'max_depth' in params and 'num_leaves' not in params:
            max_depth = params['max_depth']
            # num_leaves = 2^max_depth ± 2
            base_leaves = 2 ** max_depth
            # Randomly choose -2, -1, 0, +1, or +2
            offset = random.choice([-2, -1, 0, 1, 2])
            params['num_leaves'] = max(2, base_leaves + offset)  # Ensure at least 2 leaves
            logger.debug(f"Derived num_leaves={params['num_leaves']} from max_depth={max_depth} (2^{max_depth}{offset:+d})")
        
        # Tie subsample and colsample_bytree if sampling_rate is present
        if 'sampling_rate' in params:
            params['subsample'] = params['sampling_rate']
            params['colsample_bytree'] = params['sampling_rate']
            logger.debug(f"Tied subsample=colsample_bytree={params['sampling_rate']}")
    
    # 2. CatBoost: Tie subsample and colsample_bylevel (only with appropriate bootstrap)
    if 'catboost' in model_type_lower:
        if 'sampling_rate' in params:
            sampling_rate = params.pop('sampling_rate')
            # CatBoost's Bayesian bootstrap doesn't support subsample
            # Use Bernoulli bootstrap which does support subsample
            params.setdefault('bootstrap_type', 'Bernoulli')
            params['subsample'] = sampling_rate
            params['colsample_bylevel'] = sampling_rate
            logger.debug(
                "Set bootstrap_type=%s and tied subsample=colsample_bylevel=%s",
                params['bootstrap_type'],
                sampling_rate,
            )
    
    
    # 4. GRU/LSTM: Derive batch_size from hidden_units
    if 'gru' in model_type_lower or 'lstm' in model_type_lower:
        if 'hidden_units' in params and 'batch_size' not in params:
            params['batch_size'] = params['hidden_units'] * 2
            logger.debug(f"Derived batch_size={params['batch_size']} from hidden_units (2x)")
    
    # 5. ExtraTrees: Derive min_samples_split from min_samples_leaf
    if 'extratrees' in model_type_lower or 'extra' in model_type_lower:
        if 'min_samples_leaf' in params and 'min_samples_split' not in params:
            params['min_samples_split'] = params['min_samples_leaf'] * 2
            logger.debug(f"Derived min_samples_split={params['min_samples_split']} from min_samples_leaf (2x)")
    
    return params


# ============================================================================
# Parameter Group Definitions for Each Model Type
# ============================================================================

class ModelParameterGroups:
    """Defines parameter groups for hierarchical optimization by model type."""
    
    @staticmethod
    def get_lgbm_groups() -> List[ParameterGroup]:
        """
        Get parameter groups for LightGBM models.
        
        Groups:
        1. Structure & Learning Rate (priority 1)
        2. Regularization & Subsampling (priority 2)
        
        Note: num_leaves is auto-derived as 2^max_depth ± 2
              subsample and colsample_bytree are tied together
        """
        return [
            ParameterGroup(
                name="structure_learning_rate",
                params={
                    "max_depth": {"type": "int", "low": 3, "high": 10},
                    "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True}
                },
                priority=1,
                description="Core structure and learning rate parameters"
            ),
            ParameterGroup(
                name="regularization_subsampling",
                params={
                    # num_leaves removed - auto-derived from max_depth
                    "reg_lambda": {"type": "float", "low": 0.0, "high": 5.0},
                    "sampling_rate": {"type": "float", "low": 0.6, "high": 1.0},  # Replaces subsample + colsample_bytree
                    "min_child_samples": {"type": "int", "low": 10, "high": 100}
                },
                priority=2,
                depends_on=["structure_learning_rate"],
                description="Regularization and subsampling parameters"
            )
        ]
    
    @staticmethod
    def get_catboost_groups() -> List[ParameterGroup]:
        """
        Get parameter groups for CatBoost models.
        
        Groups:
        1. Structure & Learning (priority 1)
        2. Regularization (priority 2)
        
        Note: subsample and colsample_bylevel are tied together
        """
        return [
            ParameterGroup(
                name="structure_learning",
                params={
                    "depth": {"type": "int", "low": 4, "high": 10},
                    "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
                    "iterations": {"type": "int", "low": 300, "high": 1500}
                },
                priority=1,
                description="Core structure and learning parameters"
            ),
            ParameterGroup(
                name="regularization",
                params={
                    "l2_leaf_reg": {"type": "float", "low": 1.0, "high": 10.0},
                    "sampling_rate": {"type": "float", "low": 0.6, "high": 1.0}  # Replaces subsample + colsample_bylevel
                },
                priority=2,
                depends_on=["structure_learning"],
                description="Regularization parameters"
            )
        ]
    
    @staticmethod
    def get_tcn_groups() -> List[ParameterGroup]:
        """
        Get parameter groups for TCN (Temporal Convolutional Network) models.
        
        Groups:
        1. Architecture (priority 1)
        2. Training (priority 2)
        
        Note: batch_size is auto-derived as batch_size = num_filters (for training stability)
        """
        return [
            ParameterGroup(
                name="architecture",
                params={
                    "num_filters": {"type": "categorical", "choices": [32, 64, 128, 256]},
                    "num_layers": {"type": "int", "low": 2, "high": 6},
                    "kernel_size": {"type": "int", "low": 2, "high": 5},
                    "dilation_base": {"type": "int", "low": 2, "high": 4}
                },
                priority=1,
                description="TCN architecture parameters"
            ),
            ParameterGroup(
                name="training",
                params={
                    "dropout": {"type": "float", "low": 0.1, "high": 0.5},
                    "learning_rate": {"type": "float", "low": 0.0001, "high": 0.01, "log": True}
                    # batch_size removed - auto-derived from num_filters
                },
                priority=2,
                depends_on=["architecture"],
                description="Training parameters"
            )
        ]
    
    @staticmethod
    def get_tcn_groups_deprecated() -> List[ParameterGroup]:
        """
        [DEPRECATED] Get parameter groups for TCN (Temporal Convolutional Network) models.
        
        Groups:
        1. Architecture (priority 1)
        2. Training (priority 2)
        
        Note: batch_size is auto-derived as batch_size = num_filters (for training stability)
        """
        return [
            ParameterGroup(
                name="architecture",
                params={
                    "num_filters": {"type": "categorical", "choices": [32, 64, 128, 256]},
                    "num_layers": {"type": "int", "low": 2, "high": 6},
                    "kernel_size": {"type": "int", "low": 2, "high": 5},
                    "dilation_base": {"type": "int", "low": 2, "high": 4}
                },
                priority=1,
                description="TCN architecture parameters"
            ),
            ParameterGroup(
                name="training",
                params={
                    "dropout": {"type": "float", "low": 0.1, "high": 0.5},
                    "learning_rate": {"type": "float", "low": 0.0001, "high": 0.01, "log": True}
                    # batch_size removed - auto-derived from num_filters
                },
                priority=2,
                depends_on=["architecture"],
                description="Training parameters"
            )
        ]

    @staticmethod
    def get_depthwise_cnn_groups() -> List[ParameterGroup]:
        """
        Get parameter groups for DepthwiseSeparableCNNRegressor.
        
        Groups:
        1. Architecture (priority 1)
        2. Training (priority 2)
        
        Note: batch_size is optimized directly.
        """
        return [
            ParameterGroup(
                name="architecture",
                params={
                    "filters": {"type": "categorical", "choices": [32, 64, 128]},
                    "kernel_size": {"type": "categorical", "choices": [2, 3, 5]},
                },
                priority=1,
                description="Depthwise CNN architecture parameters"
            ),
            ParameterGroup(
                name="training_regularization",
                params={
                    "dropout": {"type": "float", "low": 0.1, "high": 0.5},
                    "learning_rate": {"type": "float", "low": 0.0001, "high": 0.01, "log": True},
                    "batch_size": {"type": "categorical", "choices": [32, 64, 128, 256]}
                },
                priority=2,
                depends_on=["architecture"],
                description="Training and regularization parameters"
            )
        ]
    
    @staticmethod
    def get_gru_groups() -> List[ParameterGroup]:
        """
        Get parameter groups for GRU models.
        
        Groups:
        1. Architecture (priority 1)
        2. Training (priority 2)
        
        Note: batch_size is auto-derived as batch_size = 2 * hidden_units (for training stability)
        """
        return [
            ParameterGroup(
                name="architecture",
                params={
                    "hidden_units": {"type": "categorical", "choices": [32, 64, 128, 256]},
                    "num_layers": {"type": "int", "low": 1, "high": 4},
                    "sequence_length": {"type": "int", "low": 6, "high": 24}
                },
                priority=1,
                description="GRU architecture parameters"
            ),
            ParameterGroup(
                name="training",
                params={
                    "dropout": {"type": "float", "low": 0.1, "high": 0.5},
                    "learning_rate": {"type": "float", "low": 0.0001, "high": 0.01, "log": True}
                    # batch_size removed - auto-derived from hidden_units
                },
                priority=2,
                depends_on=["architecture"],
                description="Training parameters"
            )
        ]
    
    @staticmethod
    def get_extratrees_groups() -> List[ParameterGroup]:
        """
        Get parameter groups for Extra Trees models.
        
        Groups:
        1. Structure (priority 1)
        2. Sampling (priority 2)
        
        Note: min_samples_split is auto-derived as 2 * min_samples_leaf (constraint satisfaction)
        """
        return [
            ParameterGroup(
                name="structure",
                params={
                    "n_estimators": {"type": "int", "low": 200, "high": 1000},
                    "max_depth": {"type": "int", "low": 5, "high": 20},
                    "max_features": {"type": "categorical", "choices": ["sqrt", "log2", 0.5, 0.7, 0.9]}
                },
                priority=1,
                description="Tree structure parameters"
            ),
            ParameterGroup(
                name="sampling",
                params={
                    "min_samples_leaf": {"type": "int", "low": 1, "high": 10}
                    # min_samples_split removed - auto-derived from min_samples_leaf
                },
                priority=2,
                depends_on=["structure"],
                description="Sampling parameters"
            )
        ]
    
    @staticmethod
    def get_meta_learner_groups() -> List[ParameterGroup]:
        """
        Get parameter groups for ensemble meta-learners (LGBM stacker).
        Uses smaller ranges for faster optimization on meta-learning level.
        
        Note: num_leaves is auto-derived as 2^max_depth ± 2
              subsample and colsample_bytree are tied together
        """
        return [
            ParameterGroup(
                name="meta_structure",
                params={
                    "max_depth": {"type": "int", "low": 3, "high": 8},
                    "learning_rate": {"type": "float", "low": 0.01, "high": 0.2, "log": True}
                },
                priority=1,
                description="Meta-learner structure"
            ),
            ParameterGroup(
                name="meta_regularization",
                params={
                    # num_leaves removed - auto-derived from max_depth
                    "reg_lambda": {"type": "float", "low": 0.0, "high": 3.0},
                    "sampling_rate": {"type": "float", "low": 0.7, "high": 1.0},  # Replaces subsample + colsample_bytree
                    "min_child_samples": {"type": "int", "low": 10, "high": 50}
                },
                priority=2,
                depends_on=["meta_structure"],
                description="Meta-learner regularization"
            )
        ]


# ============================================================================
# Custom Objective Function Using custom_balanced_score
# ============================================================================

class CustomBalancedScoreObjective:
    """
    Objective function that uses custom_balanced_score from evaluation_metrics.py.
    
    This integrates the UnifiedEvaluator's custom_balanced_score which combines:
    - Financial metrics: Sharpe, Max Drawdown, Profit Factor, Total Return
    - Statistical metrics: F1 Score, Accuracy, R² Score
    """
    
    def __init__(
        self,
        is_classification: bool = False,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize objective function.
        
        Args:
            is_classification: Whether this is a classification task
            weights: Custom weights for metrics (if None, uses defaults)
        """
        self.is_classification = is_classification
        self.weights = weights
        self.evaluator = create_unified_evaluator()
    
    def __call__(
        self,
        params: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_class: Any,
        model_type: str = 'lgbm',
        **kwargs
    ) -> float:
        """
        Evaluate parameters using custom_balanced_score.
        
        Args:
            params: Parameters to evaluate
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            model_class: Model class to instantiate
            model_type: Type of model (for parameter derivation)
            **kwargs: Additional arguments
        
        Returns:
            Custom balanced score (higher is better)
        """
        try:
            # Derive dependent parameters based on model type
            complete_params = derive_dependent_parameters(params, model_type)
            
            # Add fixed parameters for stability
            if 'lgbm' in model_type.lower():
                complete_params.setdefault('verbose', -1)
                complete_params.setdefault('n_jobs', -1)
                complete_params.setdefault('random_state', 42)
            elif 'catboost' in model_type.lower():
                complete_params.setdefault('verbose', False)
                complete_params.setdefault('random_seed', 42)
                complete_params.setdefault('thread_count', -1)
            
            # Create and train model
            try:
                model = model_class(**complete_params)
            except Exception as e:
                logger.warning(f"Model creation failed with params {complete_params}: {e}")
                return 0.0
            
            if model is None:
                logger.warning("Model is None after instantiation")
                return 0.0
            
            # Fit model with error handling
            try:
                model.fit(X_train, y_train)
            except Exception as e:
                logger.warning(f"Model fit failed: {e}")
                return 0.0
            
            # Validate data before prediction
            if X_val is None or len(X_val) == 0:
                logger.warning(f"⚠️ X_val is empty (shape: {getattr(X_val, 'shape', 'None')}), skipping prediction")
                return 0.0
            
            if y_val is None or len(y_val) == 0:
                logger.warning(f"⚠️ y_val is empty (shape: {getattr(y_val, 'shape', 'None')}), skipping prediction")
                return 0.0
            
            # Make predictions
            try:
                y_pred = model.predict(X_val)
            except Exception as e:
                logger.warning(f"⚠️ Model predict failed: {e}")
                return 0.0
            
            # Calculate metrics needed for custom_balanced_score
            from sklearn.metrics import (
                mean_squared_error, mean_absolute_error, r2_score,
                f1_score, accuracy_score
            )
            
            # Create mock financial metrics (in real use, these would come from trading simulation)
            # For now, we use prediction quality as proxy
            mse = mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            
            # Construct simple financial metrics from prediction error
            # Better predictions → better "financial" performance
            sharpe_proxy = max(0, 3.0 * (1.0 - min(mse, 1.0)))  # 0-3 range
            max_drawdown_proxy = min(0.5, mse / 2.0)  # Lower is better
            profit_factor_proxy = max(1.0, 3.0 * (1.0 - min(mae, 1.0)))
            
            # Statistical metrics
            if self.is_classification:
                zero_division_safe = cast(Any, 0)
                f1 = f1_score(
                    y_val,
                    y_pred,
                    average='weighted',
                    zero_division=zero_division_safe,
                )
                acc = accuracy_score(y_val, y_pred)
                r2 = 0.5  # Not applicable for classification
            else:
                r2_raw = float(r2_score(y_val, y_pred))
                r2 = max(0.0, r2_raw)
                # For regression, create pseudo-classification metrics
                f1 = float(max(0.0, 1.0 - min(mae, 1.0)))
                acc = float(max(0.0, 1.0 - min(mse, 1.0)))
            
            # Create mock metric objects for evaluator
            class FinancialMetrics:
                def __init__(self, sharpe, drawdown, pf, ret):
                    self.sharpe_ratio = sharpe
                    self.max_drawdown = drawdown
                    self.profit_factor = pf
                    self.total_return = ret
            
            class StatisticalMetrics:
                def __init__(self, f1, acc, r2):
                    self.f1_score = f1
                    self.accuracy = acc
                    self.r2_score = r2
            
            financial_metrics = FinancialMetrics(
                sharpe=sharpe_proxy,
                drawdown=max_drawdown_proxy,
                pf=profit_factor_proxy,
                ret=sharpe_proxy * 0.1  # Simple return proxy
            )
            
            statistical_metrics = StatisticalMetrics(f1=f1, acc=acc, r2=r2)
            
            # Calculate custom balanced score
            score = self.evaluator._calculate_custom_balanced_score(
                financial_metrics=financial_metrics,
                statistical_metrics=statistical_metrics,
                weights=self.weights,
                sample_count=len(y_val),
                apply_sample_penalty=True
            )
            
            return score
            
        except Exception as e:
            logger.warning(f"Objective evaluation failed: {e}")
            return 0.0  # Return worst score


# ============================================================================
# YAML Configuration Updater
# ============================================================================

class YAMLConfigUpdater:
    """
    Updates YAML configuration files with optimal parameters found by HPO.
    
    Preserves original file structure, comments, and only updates the
    relevant sections with optimal parameters.
    """
    
    def __init__(self, config_file: str):
        """
        Initialize YAML updater.
        
        Args:
            config_file: Path to YAML configuration file
        """
        self.config_file = Path(config_file)
        self.backup_dir = self.config_file.parent / 'hpo_backups'
        self.backup_dir.mkdir(exist_ok=True)
    
    def backup_config(self) -> Path:
        """
        Create a backup of the current configuration.
        
        Returns:
            Path to backup file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f"{self.config_file.stem}_backup_{timestamp}.yaml"
        
        with open(self.config_file, 'r') as f:
            content = f.read()
        
        with open(backup_file, 'w') as f:
            f.write(content)
        
        logger.info(f"📁 Config backed up to: {backup_file}")
        return backup_file
    
    def update_model_params(
        self,
        model_name: str,
        optimal_params: Dict[str, Any],
        hpo_result: HierarchicalOptimizationResult,
        model_path: Optional[str] = None
    ) -> bool:
        """
        Update model parameters in YAML config with optimal values from HPO.
        
        Args:
            model_name: Name of the model (e.g., 'lgbm', 'tcn', 'StandaloneGRU')
            optimal_params: Optimal parameters found by HPO
            hpo_result: Full HPO result object
            model_path: Optional path within config (e.g., 'analyst_config.base_models.lgbm')
        
        Returns:
            True if update succeeded, False otherwise
        """
        try:
            # Create backup first
            backup_file = self.backup_config()
            
            # Load current config with numpy scalar handling
            with open(self.config_file, 'r') as f:
                try:
                    config = yaml.safe_load(f)
                except yaml.constructor.ConstructorError as e:
                    logger.warning(f"Failed to load YAML with safe_load due to numpy objects: {e}")
                    # Try with unsafe loading to handle existing numpy scalars
                    f.seek(0)
                    config = yaml.load(f, Loader=yaml.UnsafeLoader)
                    # Clean the loaded config immediately
                    config = self._clean_numpy_scalars(config)
            
            # Find the model section to update
            if model_path:
                # Navigate to specific path
                path_parts = model_path.split('.')
                section = config
                for part in path_parts[:-1]:
                    section = section[part]
                model_section = section[path_parts[-1]]
            else:
                # Try to find the model automatically
                model_section = self._find_model_section(config, model_name)
            
            if model_section is None:
                logger.error(f"Could not find model section for {model_name}")
                return False
            
            # Update params section with optimal parameters
            if 'params' not in model_section:
                model_section['params'] = {}
            
            model_section['params'].update(optimal_params)
            
            # Update HPO section with results
            if 'hpo' not in model_section:
                model_section['hpo'] = {}
            
            model_section['hpo']['optimal_params'] = optimal_params
            model_section['hpo']['last_optimization'] = {
                'timestamp': datetime.now().isoformat(),
                'best_score': float(hpo_result.best_score),
                'total_trials': hpo_result.total_trials,
                'total_time_seconds': hpo_result.total_time,
                'n_rounds': len([r for r in hpo_result.group_results if r.group_name != 'final_refinement'])
            }
            
            # Write updated config with numpy scalar cleanup
            with open(self.config_file, 'w') as f:
                # Clean numpy scalars before dumping
                cleaned_config = self._clean_numpy_scalars(config)
                yaml.dump(cleaned_config, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"✅ Updated {self.config_file} with optimal parameters for {model_name}")
            logger.info(f"   Best score: {hpo_result.best_score:.6f}")
            logger.info(f"   Total trials: {hpo_result.total_trials}")
            logger.info(f"   Optimization time: {hpo_result.total_time:.2f}s")
            logger.info(f"   Backup saved to: {backup_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update YAML config: {e}")
            return False
    
    def _find_model_section(self, config: Dict[str, Any], model_name: str) -> Optional[Dict[str, Any]]:
        """
        Recursively find model section in config by model name.
        
        Args:
            config: Configuration dictionary
            model_name: Name of model to find
        
        Returns:
            Model section dictionary or None if not found
        """
        def search(d: Any, target: str) -> Optional[Dict[str, Any]]:
            if isinstance(d, dict):
                # Check if this dict represents the target model
                if d.get('model_name') == target or d.get('model_type', '').lower().find(target.lower()) >= 0:
                    return d
                
                # Check keys
                if target in d:
                    return d[target]
                
                # Recursively search values
                for v in d.values():
                    result = search(v, target)
                    if result is not None:
                        return result
            
            elif isinstance(d, list):
                for item in d:
                    result = search(item, target)
                    if result is not None:
                        return result
            
            return None
        
        return search(config, model_name)
    
    def _clean_numpy_scalars(self, obj):
        """
        Recursively clean numpy scalars from configuration to prevent YAML serialization errors.
        
        Converts numpy scalars to native Python types that can be safely serialized.
        
        Args:
            obj: Object to clean (dict, list, or scalar)
            
        Returns:
            Cleaned object with numpy scalars converted to native types
        """
        import numpy as np
        
        if isinstance(obj, dict):
            return {k: self._clean_numpy_scalars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_numpy_scalars(item) for item in obj]
        elif isinstance(obj, np.generic):
            # Convert numpy scalars to native Python types
            try:
                if np.issubdtype(obj.dtype, np.floating):
                    return float(obj)
                elif np.issubdtype(obj.dtype, np.integer):
                    return int(obj)
                elif np.issubdtype(obj.dtype, np.bool_):
                    return bool(obj)
                else:
                    return str(obj)
            except (AttributeError, TypeError):
                # Fallback for complex numpy objects
                return str(obj)
        elif hasattr(obj, '__dict__'):
            # Handle numpy objects with attributes
            try:
                if hasattr(obj, 'item'):
                    return obj.item()
                else:
                    return str(obj)
            except (AttributeError, TypeError):
                return str(obj)
        else:
            return obj


# ============================================================================
# HPO Orchestrator
# ============================================================================

class HPOOrchestrator:
    """
    Orchestrates HPO for all models defined in a configuration.
    
    Handles:
    - Reading HPO config from YAML
    - Creating appropriate parameter groups
    - Running hierarchical optimization
    - Saving results back to YAML
    """
    
    def __init__(self, config_file: str, execution_mode: str = 'full'):
        """
        Initialize HPO orchestrator.
        
        Args:
            config_file: Path to YAML configuration file
            execution_mode: Execution mode ('full', 'light')
        """
        self.config_file = config_file
        self.execution_mode = execution_mode
        self.yaml_updater = YAMLConfigUpdater(config_file)
        self.param_groups_factory = ModelParameterGroups()
    
    def get_parameter_groups(self, model_type: str) -> List[ParameterGroup]:
        """
        Get parameter groups for a model type.
        
        Args:
            model_type: Type of model ('lgbm', 'catboost', 'tcn', 'gru', 'extratrees', 'meta_learner')
        
        Returns:
            List of parameter groups for hierarchical optimization
        """
        model_type_lower = model_type.lower()
        
        if 'lgbm' in model_type_lower:
            return self.param_groups_factory.get_lgbm_groups()
        elif 'catboost' in model_type_lower:
            return self.param_groups_factory.get_catboost_groups()
        elif 'depthwise_cnn' in model_type_lower or 'temporal' in model_type_lower:
            logger.info("Using HPO groups for DepthwiseSeparableCNNRegressor.")
            return self.param_groups_factory.get_depthwise_cnn_groups()
        elif 'gru' in model_type_lower or 'lstm' in model_type_lower:
            return self.param_groups_factory.get_gru_groups()
        elif 'extratrees' in model_type_lower or 'extra' in model_type_lower:
            return self.param_groups_factory.get_extratrees_groups()
        elif 'meta' in model_type_lower or 'stacker' in model_type_lower:
            return self.param_groups_factory.get_meta_learner_groups()
        else:
            logger.warning(f"Unknown model type: {model_type}, using LGBM groups as default")
            return self.param_groups_factory.get_lgbm_groups()
    
    def get_optimization_stages(self) -> List[OptimizationStage]:
        """
        Get optimization stages based on execution mode.
        
        Returns:
            List of optimization stages
        """
        if self.execution_mode == 'light':
            return [OptimizationStage.COARSE_GRID]
        elif self.execution_mode == 'small_dataset':
            # Additional optimizations for very small datasets (< 500 samples)
            return [
                OptimizationStage.COARSE_GRID,
                OptimizationStage.FINE_GRID,  # Limited fine grid
                # Skip TPE for very small datasets to avoid overfitting
            ]
        else:
            return [
                OptimizationStage.COARSE_GRID,
                OptimizationStage.FINE_GRID,
                OptimizationStage.TPE
            ]
    
    def run_hpo(
        self,
        model_name: str,
        model_type: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        model_class: Any,
        is_classification: bool = False
    ) -> Optional[HierarchicalOptimizationResult]:
        """
        Run HPO for a specific model.
        
        Args:
            model_name: Name of the model
            model_type: Type of model
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            model_class: Model class to optimize
            is_classification: Whether this is a classification task
        
        Returns:
            HPO result or None if failed
        """
        try:
            logger.info(f"🔍 Starting HPO for {model_name} ({model_type})")
            
            # Get parameter groups
            param_groups = self.get_parameter_groups(model_type)
            
            # Create objective function with custom_balanced_score
            objective_func = CustomBalancedScoreObjective(
                is_classification=is_classification
            )
            
            # Create optimizer
            # Note: X_train, y_train, X_val, y_val will be passed by HierarchicalParameterOptimizer
            # through kwargs, so we don't capture them in the lambda closure
            optimizer = HierarchicalParameterOptimizer(
                param_groups=param_groups,
                objective_func=lambda params, X_train, y_train, X_val, y_val, model=None, **kwargs: objective_func(
                    params=params,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    model_class=model_class,
                    model_type=model_type,  # Pass model_type for parameter derivation
                    **kwargs
                ),
                stages=self.get_optimization_stages(),
                direction='maximize',  # custom_balanced_score is maximized
                n_rounds=2 if self.execution_mode == 'full' else 1,
                enable_final_refinement=self.execution_mode == 'full',
                final_refinement_trials=50,
                verbose=True
            )
            
            # Run optimization
            result = optimizer.optimize(
                X_train=X_train.values if isinstance(X_train, pd.DataFrame) else X_train,
                y_train=y_train.values if isinstance(y_train, pd.Series) else y_train,
                X_val=X_val.values if isinstance(X_val, pd.DataFrame) else X_val,
                y_val=y_val.values if isinstance(y_val, pd.Series) else y_val,
                model=None  # Model created in objective function
            )
            
            # Derive complete parameters before saving
            complete_params = derive_dependent_parameters(result.best_params, model_type)
            
            # Update YAML config with results (including derived parameters)
            self.yaml_updater.update_model_params(
                model_name=model_name,
                optimal_params=complete_params,
                hpo_result=result
            )
            
            return result
            
        except Exception as e:
            logger.error(f"HPO failed for {model_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None


__all__ = [
    'derive_dependent_parameters',
    'ModelParameterGroups',
    'CustomBalancedScoreObjective',
    'YAMLConfigUpdater',
    'HPOOrchestrator'
]

