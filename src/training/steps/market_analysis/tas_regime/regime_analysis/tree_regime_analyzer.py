"""
Tree Regime Analyzer

Advanced regime analysis using tree-based models with comprehensive evaluation
and optimization capabilities. Integrates with shared utilities for enhanced performance.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import warnings

# Import shared utilities
from src.utils.common_operations import (
    safe_dataframe_operation, validate_dataframe_columns, 
    safe_convert_dtypes, calculate_data_quality_metrics,
    safe_merge_dataframes, create_summary_statistics,
    safe_drop_columns, safe_rename_columns,
    validate_timestamp_column, safe_timestamp_conversion,
    get_dataframe_info, create_data_quality_report,
    CommonUtilities
)
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power,
    validate_finite, validate_positive, validate_range,
    safe_correlation, safe_covariance, safe_mean, safe_std,
    safe_percentile, validate_correlation_matrix,
    safe_matrix_inverse, MathValidation
)
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error
from src.utils.serialization_utils import JSONSerializer, PickleSerializer, UniversalSerializer
from src.utils.data.klines_parquet import KlinesParquetManager, get_klines_manager
from src.utils.nas_tas.bayesian_tpe_optimizer import (
    BayesianTPEOptimizer, BayesianTPEConfig, OptimizationResult
)
from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager, is_m1_available, is_mps_available
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer, optimize_dataframe_memory
from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer

# Optional ML imports with graceful fallback
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available, some functionality will be limited")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class RegimeAnalysisConfig:
    """Configuration for regime analysis."""
    
    # Model configuration
    model_type: str = 'random_forest'  # 'random_forest', 'xgboost', 'lightgbm', 'decision_tree'
    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    random_state: int = 42
    
    # Feature engineering
    feature_selection: bool = True
    feature_importance_threshold: float = 0.01
    correlation_threshold: float = 0.95
    
    # Cross-validation
    cv_folds: int = 5
    test_size: float = 0.2
    
    # Optimization
    enable_hyperparameter_optimization: bool = True
    optimization_trials: int = 50
    
    # Performance
    enable_parallel_processing: bool = True
    n_jobs: int = -1
    
    # M1 optimization
    enable_m1_optimization: bool = True
    memory_limit_gb: Optional[float] = None
    
    # Logging
    verbose: bool = True
    log_level: str = 'INFO'

@dataclass
class RegimeAnalysisResult:
    """Result of regime analysis."""
    
    # Model performance
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    # Feature importance
    feature_importance: Dict[str, float]
    selected_features: List[str]
    
    # Model details
    model_type: str
    model_params: Dict[str, Any]
    training_time: float
    
    # Cross-validation results
    cv_scores: List[float]
    cv_mean: float
    cv_std: float
    
    # Optimization results
    optimization_result: Optional[OptimizationResult] = None
    
    # Data quality
    data_quality_report: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    error_message: Optional[str] = None

class TreeRegimeAnalyzer:
    """
    Advanced tree-based regime analyzer with comprehensive evaluation capabilities.
    
    Features:
    - Multiple tree-based algorithms (Random Forest, XGBoost, LightGBM)
    - Automatic feature selection and engineering
    - Hyperparameter optimization with Bayesian TPE
    - M1 hardware optimization
    - Comprehensive evaluation metrics
    - Data quality assessment
    """
    
    def __init__(self, config: Optional[RegimeAnalysisConfig] = None):
        """Initialize tree regime analyzer."""
        self.config = config or RegimeAnalysisConfig()
        self.logger = logger.getChild('TreeRegimeAnalyzer')
        
        # Initialize utilities
        self.common_utils = CommonUtilities()
        self.math_validator = MathValidation()
        self.serializer = UniversalSerializer()
        
        # Initialize M1 optimizers if available
        self.m1_gpu_manager = None
        self.m1_memory_optimizer = None
        self.m1_cpu_optimizer = None
        
        if self.config.enable_m1_optimization:
            try:
                self.m1_gpu_manager = get_m1_gpu_manager()
                self.m1_memory_optimizer = get_m1_memory_optimizer(self.config.memory_limit_gb)
                self.m1_cpu_optimizer = get_m1_cpu_optimizer()
                
                if is_m1_available():
                    tprint_info("🧠 M1 optimization enabled")
                else:
                    tprint_warning("⚠️ M1 hardware not detected, using standard optimization")
            except Exception as e:
                tprint_warning(f"⚠️ M1 optimization setup failed: {e}")
        
        # Initialize model
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = []
        self.is_trained = False
        
        tprint_info(f"🌳 Tree Regime Analyzer initialized (model: {self.config.model_type})")
    
    def _validate_input_data(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Validate and preprocess input data."""
        try:
            # Convert to numpy arrays if needed
            if isinstance(X, pd.DataFrame):
                X_array = X.values
                self.feature_names = list(X.columns)
            else:
                X_array = np.array(X)
                self.feature_names = [f"feature_{i}" for i in range(X_array.shape[1])]
            
            # Validate X
            X_array = self.math_validator.validate_numeric_array(X_array, "X")
            
            # Validate y if provided
            y_array = None
            if y is not None:
                y_array = np.array(y)
                if y_array.ndim > 1:
                    y_array = y_array.flatten()
                y_array = self.math_validator.validate_numeric_array(y_array, "y")
            
            # M1 optimization
            if self.m1_memory_optimizer and isinstance(X, pd.DataFrame):
                X = self.m1_memory_optimizer.optimize_dataframe_memory(X)
            
            tprint_info(f"✅ Data validated: X shape {X_array.shape}, y shape {y_array.shape if y_array is not None else 'None'}")
            return X_array, y_array
            
        except Exception as e:
            tprint_error(f"❌ Data validation failed: {e}")
            raise
    
    def _create_model(self) -> Any:
        """Create the specified model."""
        try:
            if not SKLEARN_AVAILABLE:
                raise ImportError("scikit-learn is required for tree-based models")
            
            model_params = {
                'random_state': self.config.random_state,
                'n_jobs': self.config.n_jobs if self.config.enable_parallel_processing else 1
            }
            
            if self.config.model_type == 'random_forest':
                if hasattr(self, '_is_regression') and self._is_regression:
                    model = RandomForestRegressor(
                        n_estimators=self.config.n_estimators,
                        max_depth=self.config.max_depth,
                        min_samples_split=self.config.min_samples_split,
                        min_samples_leaf=self.config.min_samples_leaf,
                        **model_params
                    )
                else:
                    model = RandomForestClassifier(
                        n_estimators=self.config.n_estimators,
                        max_depth=self.config.max_depth,
                        min_samples_split=self.config.min_samples_split,
                        min_samples_leaf=self.config.min_samples_leaf,
                        **model_params
                    )
            
            elif self.config.model_type == 'decision_tree':
                if hasattr(self, '_is_regression') and self._is_regression:
                    model = DecisionTreeRegressor(
                        max_depth=self.config.max_depth,
                        min_samples_split=self.config.min_samples_split,
                        min_samples_leaf=self.config.min_samples_leaf,
                        random_state=self.config.random_state
                    )
                else:
                    model = DecisionTreeClassifier(
                        max_depth=self.config.max_depth,
                        min_samples_split=self.config.min_samples_split,
                        min_samples_leaf=self.config.min_samples_leaf,
                        random_state=self.config.random_state
                    )
            
            elif self.config.model_type == 'xgboost' and XGBOOST_AVAILABLE:
                if hasattr(self, '_is_regression') and self._is_regression:
                    model = xgb.XGBRegressor(
                        n_estimators=self.config.n_estimators,
                        max_depth=self.config.max_depth,
                        random_state=self.config.random_state,
                        n_jobs=self.config.n_jobs if self.config.enable_parallel_processing else 1
                    )
                else:
                    model = xgb.XGBClassifier(
                        n_estimators=self.config.n_estimators,
                        max_depth=self.config.max_depth,
                        random_state=self.config.random_state,
                        n_jobs=self.config.n_jobs if self.config.enable_parallel_processing else 1
                    )
            
            elif self.config.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
                if hasattr(self, '_is_regression') and self._is_regression:
                    model = lgb.LGBMRegressor(
                        n_estimators=self.config.n_estimators,
                        max_depth=self.config.max_depth,
                        random_state=self.config.random_state,
                        n_jobs=self.config.n_jobs if self.config.enable_parallel_processing else 1,
                        verbose=-1
                    )
                else:
                    model = lgb.LGBMClassifier(
                        n_estimators=self.config.n_estimators,
                        max_depth=self.config.max_depth,
                        random_state=self.config.random_state,
                        n_jobs=self.config.n_jobs if self.config.enable_parallel_processing else 1,
                        verbose=-1
                    )
            
            else:
                raise ValueError(f"Unsupported model type: {self.config.model_type}")
            
            tprint_info(f"🌳 Created {self.config.model_type} model")
            return model
            
        except Exception as e:
            tprint_error(f"❌ Model creation failed: {e}")
            raise
    
    def _select_features(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Select important features."""
        try:
            if not self.config.feature_selection:
                return X, self.feature_names
            
            # Create a temporary model for feature selection
            temp_model = self._create_model()
            temp_model.fit(X, y)
            
            # Get feature importance
            if hasattr(temp_model, 'feature_importances_'):
                importances = temp_model.feature_importances_
            else:
                # Fallback: use all features
                return X, self.feature_names
            
            # Select features above threshold
            selected_indices = np.where(importances >= self.config.feature_importance_threshold)[0]
            selected_features = [self.feature_names[i] for i in selected_indices]
            
            if len(selected_indices) == 0:
                tprint_warning("⚠️ No features selected, using all features")
                return X, self.feature_names
            
            X_selected = X[:, selected_indices]
            tprint_info(f"🔍 Selected {len(selected_features)} features from {len(self.feature_names)}")
            
            return X_selected, selected_features
            
        except Exception as e:
            tprint_warning(f"⚠️ Feature selection failed: {e}")
            return X, self.feature_names
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray) -> 'TreeRegimeAnalyzer':
        """Train the regime analyzer."""
        try:
            tprint_info("🌳 Training Tree Regime Analyzer...")
            start_time = datetime.now()
            
            # Validate input data
            X_array, y_array = self._validate_input_data(X, y)
            
            # Determine if this is regression or classification
            unique_labels = np.unique(y_array)
            self._is_regression = len(unique_labels) > 10 or np.issubdtype(y_array.dtype, np.floating)
            
            # Feature selection
            X_selected, selected_features = self._select_features(X_array, y_array)
            self.selected_features = selected_features
            
            # Create final model
            self.model = self._create_model()
            
            # Train model
            self.model.fit(X_selected, y_array)
            
            # Store training metadata
            self.training_time = (datetime.now() - start_time).total_seconds()
            self.is_trained = True
            
            tprint_info(f"✅ Training completed in {self.training_time:.2f}s")
            return self
            
        except Exception as e:
            tprint_error(f"❌ Training failed: {e}")
            raise
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions."""
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            # Validate input
            X_array, _ = self._validate_input_data(X)
            
            # Select same features as training
            if hasattr(self, 'selected_features') and len(self.selected_features) < len(self.feature_names):
                # Find indices of selected features
                selected_indices = [i for i, name in enumerate(self.feature_names) if name in self.selected_features]
                X_selected = X_array[:, selected_indices]
            else:
                X_selected = X_array
            
            # Make predictions
            predictions = self.model.predict(X_selected)
            
            return predictions
            
        except Exception as e:
            tprint_error(f"❌ Prediction failed: {e}")
            raise
    
    def analyze_regimes(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None) -> RegimeAnalysisResult:
        """Comprehensive regime analysis."""
        try:
            tprint_info("🔍 Starting comprehensive regime analysis...")
            
            # Validate input data
            X_array, y_array = self._validate_input_data(X, y)
            
            # Data quality assessment
            if isinstance(X, pd.DataFrame):
                data_quality_report = create_data_quality_report(X)
            else:
                data_quality_report = {
                    'shape': X_array.shape,
                    'dtype': str(X_array.dtype),
                    'memory_usage_mb': X_array.nbytes / (1024 * 1024)
                }
            
            # Train model if not already trained
            if not self.is_trained:
                if y_array is None:
                    raise ValueError("Target values (y) are required for training")
                self.fit(X, y_array)
            
            # Cross-validation
            cv_scores = cross_val_score(
                self.model, X_array, y_array, 
                cv=self.config.cv_folds, 
                scoring='accuracy'
            )
            
            # Make predictions
            predictions = self.predict(X_array)
            
            # Calculate metrics
            accuracy = accuracy_score(y_array, predictions)
            
            # Get feature importance
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = dict(zip(
                    getattr(self, 'selected_features', self.feature_names),
                    self.model.feature_importances_
                ))
            else:
                feature_importance = {}
            
            # Create result
            result = RegimeAnalysisResult(
                accuracy=accuracy,
                precision=0.0,  # Would need classification_report for detailed metrics
                recall=0.0,
                f1_score=0.0,
                feature_importance=feature_importance,
                selected_features=getattr(self, 'selected_features', self.feature_names),
                model_type=self.config.model_type,
                model_params=self.model.get_params() if hasattr(self.model, 'get_params') else {},
                training_time=getattr(self, 'training_time', 0.0),
                cv_scores=cv_scores.tolist(),
                cv_mean=float(np.mean(cv_scores)),
                cv_std=float(np.std(cv_scores)),
                data_quality_report=data_quality_report
            )
            
            tprint_info(f"✅ Regime analysis completed: accuracy {accuracy:.4f}")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Regime analysis failed: {e}")
            return RegimeAnalysisResult(
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                feature_importance={},
                selected_features=[],
                model_type=self.config.model_type,
                model_params={},
                training_time=0.0,
                cv_scores=[],
                cv_mean=0.0,
                cv_std=0.0,
                success=False,
                error_message=str(e)
            )


class TreeRegimeDetector:
    """
    Tree-based regime detector for real-time regime identification.
    
    Features:
    - Real-time regime detection
    - Confidence scoring
    - Regime transition detection
    - Performance monitoring
    """
    
    def __init__(self, config: Optional[RegimeAnalysisConfig] = None):
        """Initialize tree regime detector."""
        self.config = config or RegimeAnalysisConfig()
        self.analyzer = TreeRegimeAnalyzer(config)
        self.logger = logger.getChild('TreeRegimeDetector')
        
        # Regime detection state
        self.current_regime = None
        self.regime_history = []
        self.confidence_history = []
        
        tprint_info("🔍 Tree Regime Detector initialized")
    
    def detect_regimes(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Detect regimes in the data."""
        try:
            # Use analyzer for comprehensive analysis
            result = self.analyzer.analyze_regimes(X, y)
            
            # Extract regime information
            if result.success:
                # Get predictions
                predictions = self.analyzer.predict(X)
                
                # Calculate regime statistics
                unique_regimes, regime_counts = np.unique(predictions, return_counts=True)
                regime_distribution = dict(zip(unique_regimes.tolist(), regime_counts.tolist()))
                
                # Update state
                if len(predictions) > 0:
                    self.current_regime = predictions[-1]
                    self.regime_history.append(self.current_regime)
                    self.confidence_history.append(result.accuracy)
                
                return {
                    'regimes': predictions,
                    'current_regime': self.current_regime,
                    'regime_distribution': regime_distribution,
                    'confidence': result.accuracy,
                    'feature_importance': result.feature_importance,
                    'success': True
                }
            else:
                return {
                    'regimes': [],
                    'current_regime': None,
                    'regime_distribution': {},
                    'confidence': 0.0,
                    'feature_importance': {},
                    'success': False,
                    'error': result.error_message
                }
                
        except Exception as e:
            tprint_error(f"❌ Regime detection failed: {e}")
            return {
                'regimes': [],
                'current_regime': None,
                'regime_distribution': {},
                'confidence': 0.0,
                'feature_importance': {},
                'success': False,
                'error': str(e)
            }
    
    def predict_regimes(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict regimes for new data."""
        try:
            if not self.analyzer.is_trained:
                raise ValueError("Detector must be trained before making predictions")
            
            return self.analyzer.predict(X)
            
        except Exception as e:
            tprint_error(f"❌ Regime prediction failed: {e}")
            raise


class TreeRegimeClassifier:
    """
    Tree-based regime classifier with advanced classification capabilities.
    
    Features:
    - Multi-class regime classification
    - Class probability estimation
    - Classification confidence scoring
    - Performance evaluation
    """
    
    def __init__(self, config: Optional[RegimeAnalysisConfig] = None):
        """Initialize tree regime classifier."""
        self.config = config or RegimeAnalysisConfig()
        self.analyzer = TreeRegimeAnalyzer(config)
        self.logger = logger.getChild('TreeRegimeClassifier')
        
        # Classification state
        self.class_labels = None
        self.class_probabilities = None
        
        tprint_info("🏷️ Tree Regime Classifier initialized")
    
    def classify_regimes(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray) -> Dict[str, Any]:
        """Classify regimes with comprehensive evaluation."""
        try:
            # Use analyzer for training and analysis
            result = self.analyzer.analyze_regimes(X, y)
            
            if result.success:
                # Get predictions and probabilities
                predictions = self.analyzer.predict(X)
                
                # Calculate classification metrics
                from sklearn.metrics import classification_report, confusion_matrix
                
                report = classification_report(y, predictions, output_dict=True)
                cm = confusion_matrix(y, predictions)
                
                # Get class probabilities if available
                if hasattr(self.analyzer.model, 'predict_proba'):
                    probabilities = self.analyzer.model.predict_proba(X)
                    self.class_probabilities = probabilities
                else:
                    probabilities = None
                
                # Store class labels
                self.class_labels = np.unique(y)
                
                return {
                    'predictions': predictions,
                    'probabilities': probabilities,
                    'classification_report': report,
                    'confusion_matrix': cm.tolist(),
                    'accuracy': result.accuracy,
                    'feature_importance': result.feature_importance,
                    'success': True
                }
            else:
                return {
                    'predictions': [],
                    'probabilities': None,
                    'classification_report': {},
                    'confusion_matrix': [],
                    'accuracy': 0.0,
                    'feature_importance': {},
                    'success': False,
                    'error': result.error_message
                }
                
        except Exception as e:
            tprint_error(f"❌ Regime classification failed: {e}")
            return {
                'predictions': [],
                'probabilities': None,
                'classification_report': {},
                'confusion_matrix': [],
                'accuracy': 0.0,
                'feature_importance': {},
                'success': False,
                'error': str(e)
            }
    
    def predict_regime_classes(self, X: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict regime classes and probabilities."""
        try:
            if not self.analyzer.is_trained:
                raise ValueError("Classifier must be trained before making predictions")
            
            # Get predictions
            predictions = self.analyzer.predict(X)
            
            # Get probabilities if available
            probabilities = None
            if hasattr(self.analyzer.model, 'predict_proba'):
                probabilities = self.analyzer.model.predict_proba(X)
            
            return predictions, probabilities
            
        except Exception as e:
            tprint_error(f"❌ Regime class prediction failed: {e}")
            raise


# Convenience functions
def create_tree_regime_analyzer(
    model_type: str = 'random_forest',
    enable_optimization: bool = True,
    **kwargs
) -> TreeRegimeAnalyzer:
    """Create a tree regime analyzer with specified configuration."""
    config = RegimeAnalysisConfig(
        model_type=model_type,
        enable_hyperparameter_optimization=enable_optimization,
        **kwargs
    )
    return TreeRegimeAnalyzer(config)


def analyze_regimes_with_trees(
    X: Union[np.ndarray, pd.DataFrame],
    y: np.ndarray,
    model_type: str = 'random_forest',
    **kwargs
) -> RegimeAnalysisResult:
    """Convenience function for regime analysis."""
    analyzer = create_tree_regime_analyzer(model_type=model_type, **kwargs)
    return analyzer.analyze_regimes(X, y)


# Export main classes and functions
__all__ = [
    'TreeRegimeAnalyzer',
    'TreeRegimeDetector', 
    'TreeRegimeClassifier',
    'RegimeAnalysisConfig',
    'RegimeAnalysisResult',
    'create_tree_regime_analyzer',
    'analyze_regimes_with_trees'
]
