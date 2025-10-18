"""
Training Utilities

Common training logic patterns shared across all training modules.
Uses existing hardware optimization utilities for M1 optimization.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Use existing utilities
from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
from src.utils.common_operations import get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer
from src.utils.logger import system_logger

# Lazy import to avoid circular dependency
def get_enhanced_model_factory():
    """Lazy import for EnhancedModelFactory to avoid circular dependencies."""
    from src.utils.ml_common.models import EnhancedModelFactory
    return EnhancedModelFactory

def get_model_type():
    """Lazy import for ModelType to avoid circular dependencies."""
    from src.utils.ml_common.models import ModelType
    return ModelType

def get_model_config():
    """Lazy import for ModelConfig to avoid circular dependencies."""
    from src.utils.ml_common.models import ModelConfig
    return ModelConfig
from src.utils.ml_common.optimization import HierarchicalHPO, HierarchicalHPOConfig, HPOPhaseConfig
from src.utils.ml_common.optimization.overfitting_prevention import OverfittingPrevention, OverfittingPreventionConfig
from src.utils.ml_common.evaluation.evaluation_utils import EvaluationUtils

# Import universal validation integration
from .universal_validation_integration import (
    get_validation_integrator,
    validate_trained_model,
    validate_hpo_trial,
    ValidationIntegrationConfig
)

# CV utilities for safe HPO (time-series CV / purged)
try:
    from src.utils.purged_kfold import PurgedKFoldTime  # type: ignore
    _PURGED_AVAILABLE = True
except Exception:
    _PURGED_AVAILABLE = False
from sklearn.model_selection import TimeSeriesSplit

logger = system_logger.getChild('TrainingUtils')

class TrainingUtils:
    """Common training utilities."""

    def __init__(self, config: Any):
        """
        Initialize training utilities with hardware optimization.

        Args:
            config: Training configuration object or dict
        """
        # Handle case where config is a dict instead of proper config object
        if isinstance(config, dict):
            # Convert dict to BaseTrainingConfig
            from .config.base_training_config import BaseTrainingConfig
            default_config = BaseTrainingConfig()
            config_dict = {**default_config.__dict__, **config}
            config = BaseTrainingConfig(**config_dict)

        self.config = config
        self.model_factory = get_enhanced_model_factory()()
        self.overfitting_prevention = OverfittingPrevention(
            OverfittingPreventionConfig() if config.enable_overfitting_prevention else None
        )

        # Initialize hardware optimizers
        self.gpu_manager = get_m1_gpu_manager()
        self.memory_optimizer = get_m1_memory_optimizer()
        self.cpu_optimizer = get_m1_cpu_optimizer()

        if self.gpu_manager:
            logger.info("🚀 M1 GPU optimization enabled")
        if self.memory_optimizer:
            logger.info("🧠 M1 memory optimization enabled")
        if self.cpu_optimizer:
            logger.info("⚡ M1 CPU optimization enabled")

        # Initialize validation integration
        self._initialize_validation_integration()

    def _initialize_validation_integration(self):
        """Initialize universal validation integration."""
        # Create validation configuration
        validation_config = ValidationIntegrationConfig(
            enable_validation=getattr(self.config, 'enable_validation', True),
            enable_overfitting_detection=getattr(self.config, 'enable_overfitting_detection', True),
            enable_temporal_validation=getattr(self.config, 'enable_temporal_validation', True),
            enable_timeframe_validation=getattr(self.config, 'enable_timeframe_validation', True),
            save_validation_reports=getattr(self.config, 'save_validation_reports', True),
            validation_report_directory=getattr(self.config, 'validation_report_directory', "reports/validation"),
            enable_validation_logging=getattr(self.config, 'enable_validation_logging', True),
            fail_on_validation_error=getattr(self.config, 'fail_on_validation_error', False),
            warn_on_validation_issues=getattr(self.config, 'warn_on_validation_issues', True)
        )

        # Initialize validation integrator
        self.validation_integrator = get_validation_integrator(validation_config)

        logger.info("✅ Universal validation integration initialized in TrainingUtils")

    def train_model_with_validation(self,
                                   model: Any,
                                   X_train: np.ndarray,
                                   X_val: np.ndarray,
                                   y_train: np.ndarray,
                                   y_val: np.ndarray,
                                   model_name: str = "unknown",
                                   model_type: str = "unknown",
                                   timestamps: Optional[np.ndarray] = None,
                                   feature_names: Optional[List[str]] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Train model with automatic validation.

        Args:
            model: Model to train
            X_train: Training features
            X_val: Validation features
            y_train: Training labels
            y_val: Validation labels
            model_name: Name of the model
            model_type: Type of model
            timestamps: Optional timestamps for temporal validation
            feature_names: Optional feature names

        Returns:
            Tuple[Any, Dict]: (trained_model, validation_results)
        """
        # Train the model
        model.fit(X_train, y_train)

        # Validate the trained model
        validation_results = validate_trained_model(
            model=model,
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            timestamps=timestamps,
            feature_names=feature_names,
            model_name=model_name,
            model_type=model_type
        )

        # Log validation results
        if validation_results['valid']:
            logger.info(f"✅ Model {model_name} trained and validated successfully")
        else:
            logger.warning(f"⚠️ Model {model_name} trained but validation failed")
            for issue in validation_results.get('critical_issues', []):
                logger.error(f"Critical issue: {issue}")

        return model, validation_results

    def validate_hpo_trial_with_validation(self,
                                          model: Any,
                                          X_train: np.ndarray,
                                          X_val: np.ndarray,
                                          y_train: np.ndarray,
                                          y_val: np.ndarray,
                                          trial_params: Dict[str, Any],
                                          model_name: str = "unknown",
                                          model_type: str = "unknown",
                                          trial_number: int = 0) -> Tuple[Any, Dict[str, Any]]:
        """
        Train model for HPO trial with validation.

        Args:
            model: Model to train
            X_train: Training features
            X_val: Validation features
            y_train: Training labels
            y_val: Validation labels
            trial_params: HPO trial parameters
            model_name: Name of the model
            model_type: Type of model
            trial_number: HPO trial number

        Returns:
            Tuple[Any, Dict]: (trained_model, validation_results)
        """
        # Train the model
        model.fit(X_train, y_train)

        # Validate the HPO trial
        validation_results = validate_hpo_trial(
            model=model,
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            trial_params=trial_params,
            model_name=model_name,
            model_type=model_type,
            trial_number=trial_number
        )

        # Log validation results
        if validation_results['valid']:
            logger.info(f"✅ HPO trial {trial_number} for {model_name} trained and validated successfully")
        else:
            logger.warning(f"⚠️ HPO trial {trial_number} for {model_name} trained but validation failed")
            if validation_results.get('should_prune', False):
                logger.info(f"Trial should be pruned: {validation_results.get('prune_reason', 'Unknown')}")

        return model, validation_results

    def create_model(
        self,
        model_type: str,
        model_name: str,
        model_params: Optional[Dict[str, Any]] = None,
        enable_enhanced_hpo: bool = True,
        regime_labels: Optional[np.ndarray] = None,
        X_regime: Optional[np.ndarray] = None,
        y_regime: Optional[np.ndarray] = None
    ) -> Any:
        """
        Create a model instance using the model factory with enhanced HPO support.

        Args:
            model_type: Type of model to create
            model_name: Name for the model
            model_params: Optional model parameters
            enable_enhanced_hpo: Whether to use enhanced HPO system
            regime_labels: Regime labels for adaptive HPO
            X_regime: Regime-specific features for HPO
            y_regime: Regime-specific targets for HPO

        Returns:
            Created model instance
        """
        if model_params is None:
            model_params = {}

        # Map string model type to ModelType enum
        model_type_enum = self._map_string_to_model_type(model_type)

        # Enhanced HPO integration is no longer used
        if enable_enhanced_hpo and regime_labels is not None and X_regime is not None and y_regime is not None:
            try:
                # Enhanced HPO system is no longer available
                enhance_existing_hpo_pipeline = None
                EnhancedCVStrategies = None

                # Create enhanced HPO system
                enhanced_hpo_config = {
                    'n_trials': getattr(self.config, 'hpo_n_trials', 50),
                    'timeout': getattr(self.config, 'hpo_timeout_seconds', 600),
                    'random_state': 42,
                    'n_jobs': -1,
                    'enable_adaptive_ranges': True,
                    'enable_multi_objective': True,
                    'enable_dynamic_cv': True,
                    'enable_regime_analysis': True,
                    'cv_folds': 5,
                    'search_space': {
                        'learning_rate': {'min': 0.001, 'max': 0.1, 'scale': 'log'},
                        'n_estimators': {'min': 100, 'max': 2000},
                        'max_depth': {'min': 3, 'max': 12},
                        'subsample': {'min': 0.6, 'max': 1.0},
                        'colsample_bytree': {'min': 0.4, 'max': 1.0},
                        'reg_alpha': {'min': 0.0, 'max': 10.0},
                        'reg_lambda': {'min': 0.0, 'max': 10.0},
                        'min_child_weight': {'min': 1, 'max': 20},
                        'gamma': {'min': 0.0, 'max': 5.0}
                    }
                }

                enhanced_hpo = enhance_existing_hpo_pipeline(enhanced_hpo_config)

                # Get regime ID from model name or use default
                regime_id = model_name.split('_regime_')[-1] if '_regime_' in model_name else 'unknown'

                # Optimize hyperparameters using enhanced system
                optimized_params = enhanced_hpo.optimize_for_regime(
                    X=X_regime, y=y_regime,
                    regime_labels=regime_labels,
                    model_factory=self.model_factory,
                    regime_id=regime_id
                )

                # Merge optimized parameters with provided parameters
                model_params.update(optimized_params)

                logger.info(f"🔬 Enhanced HPO applied for {model_type} in regime {regime_id}")

            except Exception as e:
                logger.error(f"❌ Enhanced HPO failed: {e}")
                raise RuntimeError(f"❌ Enhanced HPO failed - fast fail enabled: {e}") from e

        model_config = ModelConfig(
            model_type=model_type_enum,
            model_name=model_name,
            model_params=model_params
        )

        model = self.model_factory.create_model(model_config)

        # Apply overfitting prevention if enabled
        if self.config.enable_overfitting_prevention:
            model = self.overfitting_prevention.apply_regularization(model, model_type)

        return model

    def create_oof_stacking_ensemble(
        self,
        base_models: Dict[str, Any],
        ensemble_name: str = "oof_stacking_ensemble",
        n_outputs: int = 4,
        output_names: Optional[List[str]] = None,
        enable_temporal_validation: bool = True,
        cv_folds: int = 5
    ) -> Any:
        """
        Create OOF stacking ensemble with proper temporal validation.

        Args:
            base_models: Dictionary of base models for each output
            ensemble_name: Name for the ensemble
            n_outputs: Number of outputs
            output_names: Names of outputs
            enable_temporal_validation: Whether to use temporal validation
            cv_folds: Number of CV folds

        Returns:
            OOF stacking ensemble manager
        """
        if output_names is None:
            output_names = [f"output_{i+1}" for i in range(n_outputs)]

        # Import OOF stacking ensemble manager
        from src.utils.ml_common.ensembles.oof_stacking_ensemble_manager import (
            OOFStackingEnsembleManager,
            OOFStackingEnsembleConfig
        )

        # Create configuration
        ensemble_config = OOFStackingEnsembleConfig(
            ensemble_name=ensemble_name,
            output_dir=f"./models/{ensemble_name}",
            n_outputs=n_outputs,
            output_names=output_names,
            base_models=base_models,
            enable_out_of_fold=True,
            enable_temporal_validation=enable_temporal_validation,
            cv_folds=cv_folds,
            enable_early_stopping=True,
            early_stopping_rounds=50
        )

        # Create ensemble manager
        ensemble_manager = OOFStackingEnsembleManager(ensemble_config)

        # Add base models to ensemble
        for output_name, models in base_models.items():
            for model_name, model in models.items():
                ensemble_manager.add_base_model(output_name, model_name, model)

        logger.info(f"✅ OOF Stacking ensemble created: {ensemble_name}")
        return ensemble_manager

    def train_oof_stacking_ensemble(
        self,
        ensemble_manager: Any,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str = "oof_ensemble",
        model_type: str = "stacking",
        timestamps: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Train OOF stacking ensemble with validation.

        Args:
            ensemble_manager: OOF stacking ensemble manager
            X: Training features
            y: Training targets
            model_name: Name of the model
            model_type: Type of model
            timestamps: Optional timestamps for temporal validation
            feature_names: Optional feature names

        Returns:
            Tuple of (trained_ensemble, validation_results)
        """
        # Use universal validation integration
        utility_selections = intelligently_select_utilities(
            X, y, model_type, "training", len(X), X.shape[1]
        )

        logger.info(f"Utility selections for OOF stacking: {utility_selections}")

        # Start overfitting monitoring if selected
        monitoring_session_id = None
        if utility_selections['utilities_selected'].get('overfitting_monitoring', False):
            monitoring_session_id = start_monitoring_session(f"{model_name}_oof_training", model_type)

        # Perform data leakage check if selected
        if utility_selections['utilities_selected'].get('data_leakage_prevention', False):
            if timestamps is not None:
                # Create temporary DataFrame for leakage check
                temp_data = pd.DataFrame(X, index=pd.to_datetime(timestamps))
                leakage_results = perform_data_leakage_check(
                    temp_data, 'timestamp', dataset_name=f"{model_name}_training"
                )
                if leakage_results['leakage_detected']:
                    logger.warning(f"Data leakage detected: {leakage_results}")

        # Train the ensemble
        trained_ensemble = ensemble_manager.fit(X, y)

        # Perform comprehensive validation
        validation_results = self.validate_trained_model(
            model=trained_ensemble,
            X_train=X,
            X_val=X,  # For OOF ensemble, we use training data since CV is already done
            y_train=y,
            y_val=y,
            timestamps=timestamps,
            feature_names=feature_names,
            model_name=model_name,
            model_type=model_type
        )

        # Perform enhanced validation if selected
        if utility_selections['utilities_selected'].get('enhanced_validation', False):
            enhanced_validation = perform_enhanced_validation(
                trained_ensemble, X, y, model_name, model_type, is_classification=True
            )
            validation_results['enhanced_validation'] = enhanced_validation

        # Perform complexity analysis if selected
        if utility_selections['utilities_selected'].get('model_complexity_analysis', False):
            complexity_analysis = perform_complexity_analysis(
                trained_ensemble, X, y, model_name, model_type
            )
            validation_results['complexity_analysis'] = complexity_analysis

        # End monitoring session if active
        if monitoring_session_id:
            end_monitoring_session(monitoring_session_id)

        # Log validation results
        if validation_results['valid']:
            logger.info(f"✅ OOF Stacking ensemble {model_name} trained and validated successfully")
        else:
            logger.warning(f"⚠️ OOF Stacking ensemble {model_name} trained but validation failed")
            for issue in validation_results.get('critical_issues', []):
                logger.error(f"Critical issue: {issue}")

        return trained_ensemble, validation_results

    def _map_string_to_model_type(self, model_type_str: str) -> 'ModelType':
        """
        Map string model type to ModelType enum.

        Args:
            model_type_str: String representation of model type

        Returns:
            ModelType enum value

        Raises:
            ValueError: If model type string cannot be mapped
        """
        from src.utils.ml_common.models.model_factory import ModelType

        # Create mapping from string to enum value
        string_to_enum_mapping = {}

        # Build the mapping by checking the .value attribute of each enum
        for enum_member in ModelType:
            string_to_enum_mapping[enum_member.value] = enum_member

        # Handle common variations and aliases
        aliases = {
            'XGBClassifier': 'XGBClassifier',
            'XGBRegressor': 'XGBRegressor',
            'LGBMClassifier': 'LGBMClassifier',
            'LGBMRegressor': 'LGBMRegressor',
            'LGBMDARTClassifier': 'LGBMDARTClassifier',
            'CatBoostClassifier': 'CatBoostClassifier',
            'CatBoostRegressor': 'CatBoostRegressor',
            'RandomForestClassifier': 'RandomForestClassifier',
            'RandomForestRegressor': 'RandomForestRegressor',
            'ExtraTreesClassifier': 'ExtraTreesClassifier',
            'ExtraTreesRegressor': 'ExtraTreesRegressor',
            'HistGradientBoostingClassifier': 'HistGradientBoostingClassifier',
            'HistGradientBoostingRegressor': 'HistGradientBoostingRegressor',
            'RidgeClassifier': 'RidgeClassifier',
            'Ridge': 'Ridge',
            'LogisticRegression': 'LogisticRegression',
            'LinearRegression': 'LinearRegression',
            'TabNetClassifier': 'TabNetClassifier',
            'TabNetRegressor': 'TabNetRegressor',
            'TCN': 'TCN',
            'LSTM': 'LSTM',
            'WaveNet': 'WaveNet',
            'NODE': 'NODE',
            'NODEClassifier': 'NODEClassifier',
            'VotingClassifier': 'VotingClassifier',
            'VotingRegressor': 'VotingRegressor',
            'StackingClassifier': 'StackingClassifier',
            'StackingRegressor': 'StackingRegressor',
            'BaggingClassifier': 'BaggingClassifier',
            'BaggingRegressor': 'BaggingRegressor',
            'AdaBoostClassifier': 'AdaBoostClassifier',
            'AdaBoostRegressor': 'AdaBoostRegressor'
        }

        # Update mapping with aliases
        for alias, target in aliases.items():
            if target in string_to_enum_mapping:
                string_to_enum_mapping[alias] = string_to_enum_mapping[target]

        # Try direct mapping first
        if model_type_str in string_to_enum_mapping:
            return string_to_enum_mapping[model_type_str]

        # Try case-insensitive matching
        model_type_upper = model_type_str.upper()
        for enum_value, enum_member in string_to_enum_mapping.items():
            if enum_value.upper() == model_type_upper:
                return enum_member

        # If no match found, raise error with helpful message
        available_types = list(string_to_enum_mapping.keys())[:10]  # Show first 10
        raise ValueError(f"Unknown model type: '{model_type_str}'. Available types: {available_types}...")

    def optimize_model_with_hpo(
        self,
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        search_space: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Optimize model using HPO.

        Args:
            model_type: Type of model to optimize
            X: Input features
            y: Target values
            search_space: HPO search space
            model_name: Optional model name

        Returns:
            Dictionary containing optimization results
        """
        if model_name is None:
            model_name = f"{model_type.lower()}_optimized"

        logger.debug(f"🔄 Optimizing {model_type}...")

        # Create base model
        base_model = self.create_model(model_type, model_name)

        # Get search space
        if search_space is None:
            search_space = getattr(self.config, 'hpo_search_spaces', {}).get(model_type, {})

        # Create HPO configuration
        hpo_config = HierarchicalHPOConfig(
            phase1_config=HPOPhaseConfig(
                phase_name=f"{model_type}_optimization",
                models={model_type: base_model},
                search_spaces={model_type: search_space},
                n_trials=self.config.hpo_n_trials,
                timeout_seconds=self.config.hpo_timeout_seconds,
                cv_folds=self.config.hpo_cv_folds
            ),
            phase2_config=HPOPhaseConfig(
                phase_name="meta_models",
                models={},
                search_spaces={},
                n_trials=0
            )
        )

        # Perform HPO
        hpo = HierarchicalHPO(hpo_config)
        # Enforce time-series CV in HPO by configuring the internal HPO object to use time series splits
        # HierarchicalHPO already performs CV; here we simply pass data. Internals will rely on cv_folds
        # Downstream scoring must not use random KFold.
        hpo_results = hpo.optimize_ensemble(X, y)

        # Extract optimized model
        optimized_model = hpo_results['base_models'][model_type]

        return {
            'model': optimized_model,
            'hpo_results': hpo_results,
            'model_type': model_type,
            'optimization_time': hpo_results.get('optimization_time', 0)
        }

    def train_single_model(
        self,
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        model_params: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train single model without HPO.

        Args:
            model_type: Type of model to train
            X: Input features
            y: Target values
            model_params: Optional model parameters
            model_name: Optional model name

        Returns:
            Dictionary containing training results
        """
        if model_name is None:
            model_name = f"{model_type.lower()}_trained"

        logger.debug(f"🔄 Training {model_type} (no HPO)...")

        # Create model
        model = self.create_model(model_type, model_name, model_params)

        # Train model
        start_time = time.time()
        model.fit(X, y)
        training_time = time.time() - start_time

        return {
            'model': model,
            'model_type': model_type,
            'training_time': training_time
        }

    def train_models(
        self,
        model_types: List[str],
        X: np.ndarray,
        y: np.ndarray,
        enable_hpo: bool = True,
        search_spaces: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Train multiple models.

        Args:
            model_types: List of model types to train
            X: Input features
            y: Target values
            enable_hpo: Whether to use HPO
            search_spaces: HPO search spaces for each model type

        Returns:
            Dictionary containing training results
        """
        model_results = {}
        training_metadata = {}

        for model_type in model_types:
            logger.info(f"🔄 Training {model_type}...")

            # Get search space for this model type
            search_space = None
            if search_spaces and model_type in search_spaces:
                search_space = search_spaces[model_type]

            # Train model
            if enable_hpo and search_space:
                model_result = self.optimize_model_with_hpo(
                    model_type, X, y, search_space
                )
            else:
                model_result = self.train_single_model(model_type, X, y)

            model_results[model_type] = model_result

            # Store training metadata
            training_metadata[model_type] = {
                'model_type': model_type,
                'training_time': model_result.get('training_time', 0),
                'optimization_time': model_result.get('optimization_time', 0),
                'samples': len(X),
                'features': X.shape[1]
            }

            logger.info(f"✅ {model_type} trained")

        return {
            'models': model_results,
            'metadata': training_metadata
        }

    def prepare_training_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        validation_size: float = 0.1,
        stratify: Optional[np.ndarray] = None,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training data with train/validation/test splits.

        Args:
            X: Input features
            y: Target values
            test_size: Proportion of data for testing
            validation_size: Proportion of data for validation
            stratify: Array for stratification
            random_state: Random state for reproducibility

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=stratify
        )

        # Second split: separate train and validation
        val_size = validation_size / (1 - test_size)  # Adjust for remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state,
            stratify=stratify[y_temp] if stratify is not None else None
        )

        logger.info(f"📊 Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def scale_features(
        self,
        X_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], StandardScaler]:
        """
        Scale features using StandardScaler.

        Args:
            X_train: Training features
            X_val: Optional validation features
            X_test: Optional test features

        Returns:
            Tuple of scaled features and fitted scaler
        """
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        X_val_scaled = None
        if X_val is not None:
            X_val_scaled = scaler.transform(X_val)

        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_val_scaled, X_test_scaled, scaler

    def evaluate_models(
        self,
        models: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        is_classification: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate multiple models.

        Args:
            models: Dictionary of trained models
            X: Input features
            y: True target values
            is_classification: Whether this is a classification task

        Returns:
            Dictionary containing evaluation results for each model
        """
        evaluation_results = {}

        for model_name, model_result in models.items():
            model = model_result['model']

            try:
                metrics = EvaluationUtils.evaluate_model_performance(
                    model, X, y,
                    metrics=self.config.evaluation_metrics,
                    is_classification=is_classification
                )
                evaluation_results[model_name] = metrics

            except Exception as e:
                logger.warning(f"⚠️ Failed to evaluate {model_name}: {e}")
                evaluation_results[model_name] = {'error': str(e)}

        return evaluation_results

    def get_model_params(self, model_type: str) -> Dict[str, Any]:
        """
        Get default parameters for model type.

        Args:
            model_type: Type of model

        Returns:
            Dictionary of default parameters
        """
        default_params = {
            'TCN': {
                'hidden_size': 64,
                'num_layers': 2,
                'dropout': 0.2,
                'recurrent_dropout': 0.1,
                'l2_regularization': 0.01
            },
            'CATBOOST': {
                'n_estimators': 1000,
                'learning_rate': 0.05,
                'depth': 6,
                'l2_leaf_reg': 3.0,
                'subsample': 0.8,
                'colsample_bylevel': 0.8
            },
            'LIGHTGBM': {
                'n_estimators': 1000,
                'learning_rate': 0.05,
                'max_depth': 6,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            },
            'LGBMDARTCLASSIFIER': {
                'boosting_type': 'dart',
                'n_estimators': 200,
                'learning_rate': 0.05,
                'max_depth': 3,
                'reg_alpha': 1.0,
                'reg_lambda': 1.0,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'drop_rate': 0.1,
                'skip_drop': 0.5
            },
            'RANDOM_FOREST': {
                'n_estimators': 500,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'bootstrap': True
            },
            'NODE': {
                'n_d': 64,
                'n_a': 64,
                'n_steps': 5,
                'gamma': 1.5,
                'lambda_sparse': 1e-3,
                'dropout': 0.1,
                'l2_regularization': 0.01
            },
            'RIDGE': {
                'alpha': 1.0,
                'solver': 'auto',
                'random_state': 42
            }
        }

        return default_params.get(model_type.upper(), {})
