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

from src.utils.ml_common.models import EnhancedModelFactory, ModelType, ModelConfig
from src.utils.ml_common.optimization import HierarchicalHPO, HierarchicalHPOConfig, HPOPhaseConfig
from src.utils.ml_common.optimization.overfitting_prevention import OverfittingPrevention, OverfittingPreventionConfig
from src.utils.ml_common.evaluation.evaluation_utils import EvaluationUtils

# Import new comprehensive utilities
from src.utils.ml_common.data_leakage_prevention import DataLeakagePrevention, DataLeakagePreventionConfig
from src.utils.ml_common.overfitting_monitoring import OverfittingMonitoring, OverfittingMonitoringConfig
from src.utils.ml_common.enhanced_validation import EnhancedValidation, EnhancedValidationConfig
from src.utils.ml_common.hpo_overfitting_prevention import HPOOverfittingPrevention, HPOOverfittingPreventionConfig
from src.utils.ml_common.model_complexity_analysis import ModelComplexityAnalyzer, ModelComplexityAnalysisConfig

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
        self.model_factory = EnhancedModelFactory()
        self.overfitting_prevention = OverfittingPrevention(
            OverfittingPreventionConfig() if config.enable_overfitting_prevention else None
        )

        # Initialize comprehensive ML utilities
        self.data_leakage_prevention = DataLeakagePrevention(DataLeakagePreventionConfig())
        self.overfitting_monitoring = OverfittingMonitoring(OverfittingMonitoringConfig())
        self.enhanced_validation = EnhancedValidation(EnhancedValidationConfig())
        self.hpo_overfitting_prevention = HPOOverfittingPrevention(HPOOverfittingPreventionConfig())
        self.model_complexity_analyzer = ModelComplexityAnalyzer(ModelComplexityAnalysisConfig())

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

        logger.info("✅ Comprehensive ML utilities initialized")
    
    def create_model(
        self, 
        model_type: str, 
        model_name: str,
        model_params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Create a model instance using the model factory.
        
        Args:
            model_type: Type of model to create
            model_name: Name for the model
            model_params: Optional model parameters
            
        Returns:
            Created model instance
        """
        if model_params is None:
            model_params = {}
        
        # Map string model type to ModelType enum
        model_type_enum = self._map_string_to_model_type(model_type)

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

    def train_model_with_comprehensive_validation(
        self,
        model_class: Any,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Union[pd.DataFrame, np.ndarray],
        y_val: Union[pd.Series, np.ndarray],
        X_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_test: Optional[Union[pd.Series, np.ndarray]] = None,
        model_name: str = "comprehensive_model",
        model_params: Optional[Dict[str, Any]] = None,
        feature_names: Optional[List[str]] = None,
        timestamps: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Train a model with comprehensive validation and monitoring.

        This method provides a complete training pipeline with:
        1. Data leakage prevention
        2. Model complexity analysis
        3. Overfitting monitoring
        4. Enhanced validation
        5. Performance tracking

        Args:
            model_class: Model class to train
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            X_test: Optional test features
            y_test: Optional test targets
            model_name: Name of the model
            model_params: Optional model parameters
            feature_names: Optional feature names
            timestamps: Optional timestamp series

        Returns:
            Dictionary containing comprehensive training results
        """
        self.logger.info(f"🚀 Starting comprehensive training for {model_name}")

        results = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'training_successful': False,
            'data_leakage_analysis': {},
            'model_complexity_analysis': {},
            'overfitting_monitoring': {},
            'enhanced_validation': {},
            'performance_metrics': {},
            'recommendations': [],
            'warnings': []
        }

        try:
            # Step 1: Data Leakage Prevention
            self.logger.info("🔍 Step 1: Data Leakage Prevention")
            leakage_results = self.data_leakage_prevention.validate_data_integrity(
                X_train, y_train, timestamps
            )

            if not leakage_results.get('overall_valid', True):
                results['warnings'].append("Data leakage detected - proceeding with caution")

            results['data_leakage_analysis'] = leakage_results

            # Step 2: Model Complexity Analysis
            self.logger.info("🔍 Step 2: Model Complexity Analysis")
            complexity_results = self.model_complexity_analyzer.analyze_model_complexity(
                model_class(**model_params) if model_params else model_class(),
                X_train, y_train, X_val, y_val, model_name, feature_names
            )
            results['model_complexity_analysis'] = complexity_results

            # Step 3: Create and train model with regularization
            self.logger.info("🔍 Step 3: Model Training with Regularization")
            if model_params is None:
                model_params = self.get_default_model_params(model_class.__name__)

            # Apply overfitting prevention
            model = self.create_model(model_class.__name__, model_name, model_params)

            # Train model
            model.fit(X_train, y_train)

            # Step 4: Comprehensive Monitoring
            self.logger.info("🔍 Step 4: Comprehensive Performance Monitoring")
            monitoring_results = self.overfitting_monitoring.monitor_model_performance(
                model, X_train, y_train, X_val, y_val, X_test, y_test, model_name
            )
            results['overfitting_monitoring'] = monitoring_results

            # Step 5: Enhanced Validation
            self.logger.info("🔍 Step 5: Enhanced Validation")
            validation_results = self.enhanced_validation.perform_comprehensive_validation(
                model, X_train, y_train, X_val, y_val, X_test, y_test, model_name, timestamps
            )
            results['enhanced_validation'] = validation_results

            # Step 6: Performance Metrics
            self.logger.info("🔍 Step 6: Performance Metrics")
            performance_metrics = self.evaluate_models(
                {model_name: model}, X_val, y_val,
                is_classification=len(np.unique(y_train)) <= 10
            )
            results['performance_metrics'] = performance_metrics

            # Step 7: Generate Recommendations
            self.logger.info("🔍 Step 7: Generating Recommendations")
            all_recommendations = []

            # Collect recommendations from all analyses
            all_recommendations.extend(leakage_results.get('prevention_report', {}).get('recommendations', []))
            all_recommendations.extend(complexity_results.get('simplification_recommendations', []))
            all_recommendations.extend(monitoring_results.get('recommendations', []))
            all_recommendations.extend(validation_results.get('recommendations', []))

            results['recommendations'] = list(set(all_recommendations))  # Remove duplicates

            # Step 8: Overall Assessment
            results['training_successful'] = self._assess_training_success(results)

            if results['training_successful']:
                self.logger.info(f"✅ Comprehensive training completed successfully for {model_name}")
            else:
                self.logger.warning(f"⚠️ Comprehensive training completed with warnings for {model_name}")

        except Exception as e:
            error_msg = f"Comprehensive training failed for {model_name}: {e}"
            results['error'] = error_msg
            results['training_successful'] = False
            results['recommendations'].append("Review training setup and data quality")
            self.logger.error(f"❌ {error_msg}")

        return results

    def _assess_training_success(self, results: Dict[str, Any]) -> bool:
        """Assess overall training success based on all validation results."""
        try:
            # Check data leakage
            leakage_analysis = results.get('data_leakage_analysis', {})
            if not leakage_analysis.get('overall_valid', True):
                return False

            # Check model complexity
            complexity_analysis = results.get('model_complexity_analysis', {})
            risk_level = complexity_analysis.get('overfitting_risk', 'low')
            if risk_level in ['very_high', 'high']:
                return False

            # Check overfitting monitoring
            monitoring_results = results.get('overfitting_monitoring', {})
            if monitoring_results.get('overfitting_detected', False):
                return False

            # Check enhanced validation
            validation_results = results.get('enhanced_validation', {})
            validation_summary = validation_results.get('validation_summary', {})
            if not validation_summary.get('overall_pass', True):
                return False

            return True

        except Exception as e:
            self.logger.warning(f"Training success assessment failed: {e}")
            return False

    def train_ensemble_with_comprehensive_validation(
        self,
        base_models: Dict[str, Any],
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Union[pd.DataFrame, np.ndarray],
        y_val: Union[pd.Series, np.ndarray],
        ensemble_name: str = "comprehensive_ensemble",
        ensemble_method: str = "voting"
    ) -> Dict[str, Any]:
        """
        Train an ensemble with comprehensive validation and monitoring.

        Args:
            base_models: Dictionary of trained base models
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            ensemble_name: Name of the ensemble
            ensemble_method: Ensemble method ('voting', 'stacking', 'bagging')

        Returns:
            Dictionary containing comprehensive ensemble training results
        """
        self.logger.info(f"🚀 Starting comprehensive ensemble training for {ensemble_name}")

        results = {
            'ensemble_name': ensemble_name,
            'timestamp': datetime.now().isoformat(),
            'training_successful': False,
            'base_model_analyses': {},
            'ensemble_diversity_analysis': {},
            'overfitting_monitoring': {},
            'enhanced_validation': {},
            'performance_metrics': {},
            'recommendations': []
        }

        try:
            # Step 1: Analyze each base model
            self.logger.info("🔍 Step 1: Base Model Analysis")
            for model_name, model in base_models.items():
                model_analysis = self.analyze_model_comprehensive(
                    model, X_train, y_train, X_val, y_val, model_name
                )
                results['base_model_analyses'][model_name] = model_analysis

            # Step 2: Ensemble Diversity Analysis
            self.logger.info("🔍 Step 2: Ensemble Diversity Analysis")
            diversity_results = self.overfitting_monitoring.analyze_ensemble_diversity(
                base_models, X_val, y_val, ensemble_name
            )
            results['ensemble_diversity_analysis'] = diversity_results

            # Step 3: Create and train ensemble
            self.logger.info("🔍 Step 3: Ensemble Creation and Training")
            if ensemble_method == 'voting':
                from sklearn.ensemble import VotingRegressor, VotingClassifier
                is_regression = len(np.unique(y_train)) > 10

                if is_regression:
                    ensemble_model = VotingRegressor(list(base_models.items()))
                else:
                    ensemble_model = VotingClassifier(list(base_models.items()))

            elif ensemble_method == 'stacking':
                from sklearn.ensemble import StackingRegressor, StackingClassifier
                is_regression = len(np.unique(y_train)) > 10

                if is_regression:
                    ensemble_model = StackingRegressor(list(base_models.items()))
                else:
                    ensemble_model = StackingClassifier(list(base_models.items()))

            else:
                raise ValueError(f"Unsupported ensemble method: {ensemble_method}")

            # Train ensemble
            ensemble_model.fit(X_train, y_train)

            # Step 4: Comprehensive Monitoring
            self.logger.info("🔍 Step 4: Ensemble Performance Monitoring")
            monitoring_results = self.overfitting_monitoring.monitor_model_performance(
                ensemble_model, X_train, y_train, X_val, y_val, None, None, ensemble_name
            )
            results['overfitting_monitoring'] = monitoring_results

            # Step 5: Enhanced Validation
            self.logger.info("🔍 Step 5: Ensemble Validation")
            validation_results = self.enhanced_validation.perform_comprehensive_validation(
                ensemble_model, X_train, y_train, X_val, y_val, None, None, ensemble_name
            )
            results['enhanced_validation'] = validation_results

            # Step 6: Performance Comparison
            self.logger.info("🔍 Step 6: Performance Comparison")
            ensemble_metrics = self.evaluate_models(
                {ensemble_name: ensemble_model}, X_val, y_val,
                is_classification=len(np.unique(y_train)) <= 10
            )
            results['performance_metrics'] = ensemble_metrics

            # Compare with base models
            base_metrics = {}
            for model_name, model in base_models.items():
                base_metrics[model_name] = self.evaluate_models(
                    {model_name: model}, X_val, y_val,
                    is_classification=len(np.unique(y_train)) <= 10
                )[model_name]

            results['base_model_metrics'] = base_metrics

            # Step 7: Generate Recommendations
            self.logger.info("🔍 Step 7: Generating Ensemble Recommendations")
            ensemble_recommendations = []

            # Diversity recommendations
            if diversity_results.get('overfitting_risk') in ['high', 'medium']:
                ensemble_recommendations.extend([
                    "Consider adding more diverse base models",
                    "Implement ensemble diversity regularization",
                    "Monitor ensemble performance for overfitting"
                ])

            # Base model recommendations
            for model_name, analysis in results['base_model_analyses'].items():
                model_recommendations = analysis.get('recommendations', [])
                ensemble_recommendations.extend([f"{model_name}: {rec}" for rec in model_recommendations])

            results['recommendations'] = ensemble_recommendations

            # Step 8: Overall Assessment
            results['training_successful'] = self._assess_ensemble_training_success(results)

            if results['training_successful']:
                self.logger.info(f"✅ Comprehensive ensemble training completed for {ensemble_name}")
            else:
                self.logger.warning(f"⚠️ Comprehensive ensemble training completed with warnings for {ensemble_name}")

        except Exception as e:
            error_msg = f"Comprehensive ensemble training failed for {ensemble_name}: {e}"
            results['error'] = error_msg
            results['training_successful'] = False
            results['recommendations'].append("Review ensemble setup and base model compatibility")
            self.logger.error(f"❌ {error_msg}")

        return results

    def _assess_ensemble_training_success(self, results: Dict[str, Any]) -> bool:
        """Assess overall ensemble training success."""
        try:
            # Check ensemble diversity
            diversity_analysis = results.get('ensemble_diversity_analysis', {})
            if diversity_analysis.get('overfitting_risk') == 'very_high':
                return False

            # Check ensemble performance
            ensemble_metrics = results.get('performance_metrics', {})
            if not ensemble_metrics:
                return False

            # Check base model performance
            base_analyses = results.get('base_model_analyses', {})
            failed_base_models = 0

            for model_name, analysis in base_analyses.items():
                if not analysis.get('training_successful', True):
                    failed_base_models += 1

            if failed_base_models > len(base_analyses) * 0.5:  # More than half failed
                return False

            return True

        except Exception as e:
            self.logger.warning(f"Ensemble training success assessment failed: {e}")
            return False

    def optimize_hyperparameters_with_comprehensive_validation(
        self,
        model_class: Any,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        model_name: str = "optimized_model",
        search_space: Optional[Dict[str, Any]] = None,
        custom_objective: Optional[Callable] = None,
        timestamps: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters with comprehensive validation and overfitting prevention.

        Args:
            model_class: Model class to optimize
            X: Feature matrix
            y: Target values
            model_name: Name of the model
            search_space: Parameter search space
            custom_objective: Custom objective function
            timestamps: Optional timestamp series

        Returns:
            Dictionary containing comprehensive optimization results
        """
        self.logger.info(f"🚀 Starting comprehensive HPO for {model_name}")

        results = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'optimization_successful': False,
            'hpo_results': {},
            'best_model_analysis': {},
            'recommendations': []
        }

        try:
            # Step 1: Perform HPO with overfitting prevention
            self.logger.info("🔍 Step 1: Hyperparameter Optimization")
            hpo_results = self.hpo_overfitting_prevention.optimize_hyperparameters(
                model_class, X, y, model_name, search_space, custom_objective, timestamps
            )
            results['hpo_results'] = hpo_results

            # Step 2: Analyze best model
            self.logger.info("🔍 Step 2: Best Model Analysis")
            best_params = hpo_results.get('best_params', {})
            best_model = model_class(**best_params)
            best_model.fit(X, y)

            # Comprehensive analysis of best model
            best_model_analysis = self.train_model_with_comprehensive_validation(
                model_class, X, y, X, y, None, None, f"{model_name}_best", best_params
            )
            results['best_model_analysis'] = best_model_analysis

            # Step 3: Generate recommendations
            self.logger.info("🔍 Step 3: Generating HPO Recommendations")
            hpo_recommendations = hpo_results.get('recommendations', [])
            model_recommendations = best_model_analysis.get('recommendations', [])

            results['recommendations'] = list(set(hpo_recommendations + model_recommendations))

            # Step 4: Overall Assessment
            results['optimization_successful'] = self._assess_hpo_success(results)

            if results['optimization_successful']:
                self.logger.info(f"✅ Comprehensive HPO completed successfully for {model_name}")
            else:
                self.logger.warning(f"⚠️ Comprehensive HPO completed with warnings for {model_name}")

        except Exception as e:
            error_msg = f"Comprehensive HPO failed for {model_name}: {e}"
            results['error'] = error_msg
            results['optimization_successful'] = False
            results['recommendations'].append("Review HPO setup and optimization constraints")
            self.logger.error(f"❌ {error_msg}")

        return results

    def _assess_hpo_success(self, results: Dict[str, Any]) -> bool:
        """Assess overall HPO success."""
        try:
            # Check HPO results
            hpo_results = results.get('hpo_results', {})
            if not hpo_results.get('best_params'):
                return False

            # Check best model analysis
            best_model_analysis = results.get('best_model_analysis', {})
            if not best_model_analysis.get('training_successful', False):
                return False

            # Check overfitting risk
            model_complexity = best_model_analysis.get('model_complexity_analysis', {})
            risk_level = model_complexity.get('overfitting_risk', 'low')

            if risk_level in ['very_high', 'high']:
                return False

            return True

        except Exception as e:
            self.logger.warning(f"HPO success assessment failed: {e}")
            return False

    def analyze_model_comprehensive(
        self,
        model: Any,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Union[pd.DataFrame, np.ndarray],
        y_val: Union[pd.Series, np.ndarray],
        model_name: str = "analyzed_model",
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive model analysis.

        Args:
            model: Trained model
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            model_name: Name of the model
            feature_names: Optional feature names

        Returns:
            Dictionary containing comprehensive analysis results
        """
        self.logger.info(f"🔍 Starting comprehensive analysis for {model_name}")

        results = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'analysis_complete': False,
            'complexity_analysis': {},
            'performance_analysis': {},
            'validation_analysis': {},
            'recommendations': []
        }

        try:
            # 1. Model Complexity Analysis
            self.logger.debug("Analyzing model complexity...")
            complexity_results = self.model_complexity_analyzer.analyze_model_complexity(
                model, X_train, y_train, X_val, y_val, model_name, feature_names
            )
            results['complexity_analysis'] = complexity_results

            # 2. Performance Analysis
            self.logger.debug("Analyzing model performance...")
            performance_results = self.overfitting_monitoring.monitor_model_performance(
                model, X_train, y_train, X_val, y_val, None, None, model_name
            )
            results['performance_analysis'] = performance_results

            # 3. Validation Analysis
            self.logger.debug("Performing validation analysis...")
            validation_results = self.enhanced_validation.perform_comprehensive_validation(
                model, X_train, y_train, X_val, y_val, None, None, model_name
            )
            results['validation_analysis'] = validation_results

            # 4. Generate Comprehensive Recommendations
            self.logger.debug("Generating comprehensive recommendations...")
            all_recommendations = []

            all_recommendations.extend(complexity_results.get('simplification_recommendations', []))
            all_recommendations.extend(performance_results.get('recommendations', []))
            all_recommendations.extend(validation_results.get('recommendations', []))

            results['recommendations'] = list(set(all_recommendations))
            results['analysis_complete'] = True

            self.logger.info(f"✅ Comprehensive analysis completed for {model_name}")

        except Exception as e:
            error_msg = f"Comprehensive analysis failed for {model_name}: {e}"
            results['error'] = error_msg
            results['recommendations'].append("Review analysis setup and data quality")
            self.logger.error(f"❌ {error_msg}")

        return results