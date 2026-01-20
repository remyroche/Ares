"""
Layer 2.5 Integration: Chaser System Integration

Integrates the Layer 2.5 Chaser between Layer 2 (Base Models) and Layer 3 (Meta-Learner).
Provides the complete pipeline for residual learning and conflict detection.

Integration Flow:
Layer 2 (Base Models) → Layer 2.5 (Chaser) → Layer 3 (Meta-Learner)
     ↓                    ↓                      ↓
Causal Parents      Residual Learner         Conflict Detection
Structural Baseline  Non-Linear Alpha        Final Decisions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
import logging
import time
import json
from datetime import datetime
from sklearn.metrics import mean_squared_error

# Import Layer 2.5 components
from .layer2_5_chaser import Layer25Chaser, create_chaser
from .causal_residual_computation import compute_causal_residuals, analyze_residual_quality
from .non_causal_feature_selector import NonCausalFeatureSelector
from .conflict_detection import ConflictDetector

# Import tprint functions
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    # Fallback print functions
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_success(msg): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")
    def tprint_error(msg): print(f"[ERROR] {msg}")

class Layer25Logger:
    """
    Comprehensive logging and performance monitoring for Layer 2.5.
    """

    def __init__(self, log_level=logging.INFO, log_file=None):
        tprint_info(f"Starting __init__")
        self.logger = logging.getLogger('Layer25Integration')
        self.logger.setLevel(log_level)

        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(simple_formatter)
        self.logger.addHandler(console_handler)

        # File handler (if specified)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(detailed_formatter)
            self.logger.addHandler(file_handler)

        # Performance tracking
        self.performance_metrics = {
            'training_times': [],
            'prediction_times': [],
            'memory_usage': [],
            'component_performance': {}
        }

        self.start_time = time.time()

        tprint_info(f"Completed __init__")
    def log_pipeline_start(self, config_summary):
        """Log pipeline initialization."""
        tprint_info(f"Starting log_pipeline_start")
        self.logger.info("🚀 Layer 2.5 Pipeline Started")
        self.logger.info(f"Configuration: {json.dumps(config_summary, indent=2)}")
        self._log_system_info()

        tprint_info(f"Completed log_pipeline_start")
    def log_training_start(self, n_samples, n_features):
        """Log training initialization."""
        tprint_info(f"Starting log_training_start")
        self.logger.info(f"🏋️ Training Started - Samples: {n_samples}, Features: {n_features}")
        self.training_start = time.time()

        tprint_info(f"Completed log_training_start")
    def log_training_complete(self, training_metrics, feature_selection_results):
        """Log training completion with detailed metrics."""
        tprint_info(f"Starting log_training_complete")
        training_time = time.time() - self.training_start

        self.logger.info("✅ Training Completed")
        self.logger.info(f"⏱️ Training Time: {training_time:.2f} seconds")

        # Log training metrics
        self.logger.info("📊 Training Metrics:")
        for metric, value in training_metrics.items():
            if isinstance(value, dict):
                self.logger.info(f"  {metric}:")
                for sub_metric, sub_value in value.items():
                    self.logger.info(f"    {sub_metric}: {sub_value:.6f}")
            else:
                self.logger.info(f"  {metric}: {value:.6f}")

        # Log feature selection results
        if feature_selection_results:
            self.logger.info("🎯 Feature Selection Results:")
            self.logger.info(f"  Selected Features: {feature_selection_results.get('selected_count', 0)}")
            self.logger.info(f"  Excluded Features: {feature_selection_results.get('excluded_count', 0)}")
            self.logger.info(f"  Selection Ratio: {feature_selection_results.get('selection_ratio', 0):.2%}")

        # Store performance metrics
        self.performance_metrics['training_times'].append({
            'timestamp': datetime.now().isoformat(),
            'duration': training_time,
            'metrics': training_metrics
        })

        tprint_info(f"Completed log_training_complete")
    def log_prediction_start(self, n_samples):
        """Log prediction start."""
        tprint_info(f"Starting log_prediction_start")
        self.logger.info(f"🔮 Prediction Started - Samples: {n_samples}")
        self.prediction_start = time.time()

        tprint_info(f"Completed log_prediction_start")
    def log_prediction_complete(self, prediction_results):
        """Log prediction completion with performance metrics."""
        tprint_info(f"Starting log_prediction_complete")
        prediction_time = time.time() - self.prediction_start
        n_samples = len(prediction_results.get('chaser_prediction', []))

        self.logger.info("✅ Prediction Completed")
        self.logger.info(f"⏱️ Prediction Time: {prediction_time:.4f} seconds")
        self.logger.info(f"⚡ Throughput: {n_samples/prediction_time:.2f} samples/second")

        # Log prediction statistics
        if 'chaser_prediction' in prediction_results:
            chaser_pred = prediction_results['chaser_prediction']
            self.logger.info("📈 Prediction Statistics:")
            self.logger.info(f"  Chaser Mean: {np.mean(chaser_pred):.6f}")
            self.logger.info(f"  Chaser Std: {np.std(chaser_pred):.6f}")
            self.logger.info(f"  Chaser Min: {np.min(chaser_pred):.6f}")
            self.logger.info(f"  Chaser Max: {np.max(chaser_pred):.6f}")

        # Log confidence statistics
        if 'chaser_confidence' in prediction_results:
            confidence = prediction_results['chaser_confidence']
            self.logger.info("🎯 Confidence Statistics:")
            self.logger.info(f"  Mean Confidence: {np.mean(confidence):.4f}")
            self.logger.info(f"  High Confidence Ratio: {np.mean(confidence > 0.6):.2%}")

        # Log conflict statistics
        if 'high_conflict' in prediction_results:
            high_conflicts = prediction_results['high_conflict']
            conflict_rate = np.mean(high_conflicts)
            self.logger.info("⚠️ Conflict Statistics:")
            self.logger.info(f"  High Conflict Rate: {conflict_rate:.2%}")
            self.logger.info(f"  Total Conflicts: {np.sum(high_conflicts)}")

        # Store performance metrics
        self.performance_metrics['prediction_times'].append({
            'timestamp': datetime.now().isoformat(),
            'duration': prediction_time,
            'throughput': n_samples/prediction_time,
            'samples': n_samples
        })

        tprint_info(f"Completed log_prediction_complete")
    def log_error(self, operation, error, context=None):
        """Log errors with context."""
        tprint_info(f"Starting log_error")
        error_msg = f"❌ {operation} Failed: {str(error)}"
        if context:
            error_msg += f" | Context: {context}"
        self.logger.error(error_msg)

        tprint_info(f"Completed log_error")
    def log_warning(self, message, context=None):
        """Log warnings with context."""
        tprint_info(f"Starting log_warning")
        warning_msg = f"⚠️ {message}"
        if context:
            warning_msg += f" | Context: {context}"
        self.logger.warning(warning_msg)

        tprint_info(f"Completed log_warning")
    def log_performance_summary(self):
        """Log overall performance summary."""
        tprint_info(f"Starting log_performance_summary")
        total_runtime = time.time() - self.start_time

        self.logger.info("📋 Performance Summary")
        self.logger.info(f"Total Runtime: {total_runtime:.2f} seconds")

        # Training performance
        if self.performance_metrics['training_times']:
            avg_training_time = np.mean([t['duration'] for t in self.performance_metrics['training_times']])
            self.logger.info(f"Average Training Time: {avg_training_time:.2f} seconds")

        # Prediction performance
        if self.performance_metrics['prediction_times']:
            avg_prediction_time = np.mean([t['duration'] for t in self.performance_metrics['prediction_times']])
            avg_throughput = np.mean([t['throughput'] for t in self.performance_metrics['prediction_times']])
            total_predictions = sum([t['samples'] for t in self.performance_metrics['prediction_times']])

            self.logger.info(f"Average Prediction Time: {avg_prediction_time:.4f} seconds")
            self.logger.info(f"Average Throughput: {avg_throughput:.2f} samples/second")
            self.logger.info(f"Total Predictions: {total_predictions}")

        tprint_info(f"Completed log_performance_summary")
    def _log_system_info(self):
        """Log system information."""
        tprint_info(f"Starting _log_system_info")
        try:
            import psutil
            import os

            self.logger.info("🖥️ System Information:")
            self.logger.info(f"  CPU Count: {psutil.cpu_count()}")
            self.logger.info(f"  Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
            self.logger.info(f"  Python Version: {os.sys.version}")

            memory_mb = psutil.virtual_memory().used / (1024**2)
            self.logger.info(f"  Current Memory Usage: {memory_mb:.1f} MB")
        except ImportError:
            self.logger.info("  System monitoring not available (psutil not installed)")

        tprint_info(f"Completed _log_system_info")
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        tprint_info(f"Starting get_performance_report")
        import psutil
        import os

        tprint_info(f"Completed get_performance_report")
        return {
            'total_runtime': time.time() - self.start_time,
            'training_sessions': len(self.performance_metrics['training_times']),
            'prediction_sessions': len(self.performance_metrics['prediction_times']),
            'performance_metrics': self.performance_metrics,
            'system_info': {
                'cpu_count': psutil.cpu_count() if 'psutil' in globals() else 'unknown',
                'memory_gb': psutil.virtual_memory().total / (1024**3) if 'psutil' in globals() else 'unknown'
            }
        }

class HyperparameterOptimizer:
    """
    Joint hyperparameter optimization for Layer 2.5 components.
    """

    def __init__(self):
        tprint_info(f"Starting __init__")
        self.best_params = None
        self.optimization_history = []

        tprint_info(f"Completed __init__")
    def create_param_space(self):
        """Define parameter space for Bayesian optimization."""
        tprint_info(f"Starting create_param_space")
        from skopt.space import Real, Integer

        tprint_info(f"Completed create_param_space")
        return [
            # XGBoost parameters
            Real(0.01, 0.3, name='xgb_learning_rate'),
            Integer(3, 8, name='xgb_max_depth'),
            Real(0.5, 1.0, name='xgb_subsample'),
            Real(0.5, 1.0, name='xgb_colsample_bytree'),
            Real(0.1, 1.0, name='xgb_reg_alpha'),
            Real(0.1, 1.0, name='xgb_reg_lambda'),

            # CatBoost parameters
            Real(0.01, 0.3, name='cat_learning_rate'),
            Integer(3, 8, name='cat_depth'),
            Real(1.0, 5.0, name='cat_l2_leaf_reg'),

            # Ensemble weights
            Real(0.3, 0.8, name='xgb_weight'),

            # Conflict detector thresholds
            Real(0.0, 0.5, name='direction_threshold'),
            Real(0.5, 3.0, name='magnitude_threshold'),
            Real(0.4, 0.8, name='confidence_threshold'),
            Real(0.3, 0.7, name='conflict_intensity_threshold')
        ]

    def objective_function(self, X_train, y_train, X_val, y_val, causal_anchor_val, params):
        """Objective function for optimization."""
        tprint_info(f"Starting objective_function")
        try:
            # Unpack parameters
            xgb_lr, xgb_depth, xgb_sub, xgb_col, xgb_alpha, xgb_lambda, \
            cat_lr, cat_depth, cat_l2, xgb_weight, \
            dir_thresh, mag_thresh, conf_thresh, conf_int_thresh = params

            # Create component parameters
            xgb_params = {
                'n_estimators': 200,
                'max_depth': int(xgb_depth),
                'learning_rate': xgb_lr,
                'subsample': xgb_sub,
                'colsample_bytree': xgb_col,
                'reg_alpha': xgb_alpha,
                'reg_lambda': xgb_lambda,
                'random_state': 42,
                'n_jobs': -1
            }

            cat_params = {
                'iterations': 200,
                'depth': int(cat_depth),
                'learning_rate': cat_lr,
                'l2_leaf_reg': cat_l2,
                'random_seed': 42,
                'verbose': False
            }

            ensemble_weights = [xgb_weight, 1.0 - xgb_weight]

            conflict_params = {
                'direction_threshold': dir_thresh,
                'magnitude_threshold': mag_thresh,
                'confidence_threshold': conf_thresh,
                'conflict_intensity_threshold': conf_int_thresh
            }

            # Create and train pipeline
            integration = Layer25Integration()
            integration.setup_chaser(
                xgb_params=xgb_params,
                cat_params=cat_params,
                ensemble_weights=ensemble_weights
            )
            integration.setup_conflict_detector(**conflict_params)

            # Train
            training_metrics = integration.train_chaser(X_train, y_train)

            # Evaluate on validation
            val_predictions = integration.predict_with_conflict_detection(
                X_val, causal_anchor_val
            )

            # Calculate validation score
            val_score = self.calculate_validation_score(
                val_predictions, y_val, causal_anchor_val
            )

            tprint_info(f"Completed objective_function")
            return -val_score  # Minimize negative score

        except Exception as e:
            print(f"Optimization iteration failed: {e}")
            return 1000.0  # High penalty for failures

        tprint_info(f"Completed objective_function")
    def calculate_validation_score(self, predictions, y_val, causal_anchor_val):
        """Calculate comprehensive validation score."""
        tprint_info(f"Starting calculate_validation_score")
        chaser_pred = predictions['chaser_prediction']
        total_pred = predictions['total_prediction']

        # Individual model scores
        chaser_mse = mean_squared_error(y_val, chaser_pred)
        total_mse = mean_squared_error(y_val, total_pred)
        anchor_mse = mean_squared_error(y_val, causal_anchor_val)

        # Improvement over anchor
        improvement = anchor_mse - total_mse

        # Conflict penalty (reward conflicts that improve performance)
        if 'high_conflict' in predictions:
            conflict_rate = np.mean(predictions['high_conflict'])
            conflict_penalty = conflict_rate * 0.1  # Small penalty for conflicts
        else:
            conflict_penalty = 0.0

        # Combined score (lower is better)
        score = total_mse + conflict_penalty - improvement

        tprint_info(f"Completed calculate_validation_score")
        return score

    def optimize(self, X_train, y_train, X_val, y_val, causal_anchor_val, n_calls=50):
        """Run Bayesian optimization."""
        tprint_info(f"Starting optimize")
        from skopt import gp_minimize
        from skopt.utils import use_named_args

        param_space = self.create_param_space()

        @use_named_args(param_space)
        def objective(**params):
            tprint_info(f"Starting objective")
            param_values = [params[name] for name in
                          ['xgb_learning_rate', 'xgb_max_depth', 'xgb_subsample',
                           'xgb_colsample_bytree', 'xgb_reg_alpha', 'xgb_reg_lambda',
                           'cat_learning_rate', 'cat_depth', 'cat_l2_leaf_reg',
                           'xgb_weight', 'direction_threshold', 'magnitude_threshold',
                           'confidence_threshold', 'conflict_intensity_threshold']]

            tprint_info(f"Completed objective")
            return self.objective_function(
                X_train, y_train, X_val, y_val, causal_anchor_val, param_values
            )

        result = gp_minimize(
            objective,
            param_space,
            n_calls=n_calls,
            n_random_starts=10,
            random_state=42,
            verbose=True
        )

        # Store best parameters
        self.best_params = {
            'xgb_params': {
                'n_estimators': 200,
                'max_depth': int(result.x[1]),
                'learning_rate': result.x[0],
                'subsample': result.x[2],
                'colsample_bytree': result.x[3],
                'reg_alpha': result.x[4],
                'reg_lambda': result.x[5],
                'random_state': 42,
                'n_jobs': -1
            },
            'cat_params': {
                'iterations': 200,
                'depth': int(result.x[7]),
                'learning_rate': result.x[6],
                'l2_leaf_reg': result.x[8],
                'random_seed': 42,
                'verbose': False
            },
            'ensemble_weights': [result.x[9], 1.0 - result.x[9]],
            'conflict_params': {
                'direction_threshold': result.x[10],
                'magnitude_threshold': result.x[11],
                'confidence_threshold': result.x[12],
                'conflict_intensity_threshold': result.x[13]
            }
        }

        tprint_info(f"Completed optimize")
        return self.best_params, result

class Layer25Integration:
    """
    Complete Layer 2.5 Chaser integration system.
    
    Manages the entire pipeline from causal residual computation
    through Chaser training to conflict detection and meta-learner integration.
    """
    
    def __init__(
        self,
        chaser_params: Optional[Dict] = None,
        feature_selector_params: Optional[Dict] = None,
        conflict_detector_params: Optional[Dict] = None,
        enable_residual_analysis: bool = True,
        enable_conflict_detection: bool = True,
        enable_logging: bool = True,
        log_file: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize Layer 2.5 Integration.

        Args:
            chaser_params: Parameters for Chaser model
            feature_selector_params: Parameters for feature selector
            conflict_detector_params: Parameters for conflict detector
            enable_residual_analysis: Whether to analyze residual quality
            enable_conflict_detection: Whether to enable conflict detection
            enable_logging: Whether to enable comprehensive logging
            log_file: Optional file path for logging output
            verbose: Whether to print progress information
        """
        tprint_info(f"Starting __init__")
        self.verbose = verbose
        self.enable_residual_analysis = enable_residual_analysis
        self.enable_conflict_detection = enable_conflict_detection
        self.enable_logging = enable_logging

        # Initialize logger
        self.logger = Layer25Logger(log_file=log_file) if enable_logging else None

        # Initialize components
        self.chaser = None
        self.feature_selector = None
        self.conflict_detector = None

        # Component parameters
        self.chaser_params = chaser_params or {}
        self.feature_selector_params = feature_selector_params or {}
        self.conflict_detector_params = conflict_detector_params or {}

        # Training data
        self.training_features = None
        self.training_residuals = None
        self.causal_graph = None

        # Results storage
        self.chaser_metrics = None
        self.residual_analysis = None
        self.feature_selection_results = None
        self.conflict_statistics = None

        # Log pipeline start
        self._log_pipeline_start()

        tprint_info(f"Completed __init__")
    def _log_pipeline_start(self):
        """Log pipeline start."""
        tprint_info(f"Starting _log_pipeline_start")
        if self.logger:
            config_summary = {
                'chaser_configured': self.chaser is not None,
                'feature_selector_configured': self.feature_selector is not None,
                'conflict_detector_configured': self.conflict_detector is not None,
                'residual_analysis_enabled': self.enable_residual_analysis,
                'conflict_detection_enabled': self.enable_conflict_detection,
                'logging_enabled': self.enable_logging
            }
            self.logger.log_pipeline_start(config_summary)

        tprint_info(f"Completed _log_pipeline_start")
    def setup_chaser(
        self,
        xgb_params: Optional[Dict] = None,
        cat_params: Optional[Dict] = None,
        ensemble_weights: Optional[List[float]] = None,
        **kwargs
    ):
        """Setup the Chaser model with custom parameters."""
        tprint_info(f"Starting setup_chaser")
        chaser_config = {
            'xgb_params': xgb_params,
            'cat_params': cat_params,
            'ensemble_weights': ensemble_weights,
            **kwargs
        }
        chaser_config.update(self.chaser_params)
        
        self.chaser = create_chaser(**chaser_config)
        
        if self.verbose:
            tprint_success("✅ Chaser model initialized")

        tprint_info(f"Completed setup_chaser")
    def setup_feature_selector(
        self,
        causal_graph: Optional[Dict[str, List[str]]] = None,
        technical_patterns: Optional[List[str]] = None,
        **kwargs
    ):
        """Setup the non-causal feature selector."""
        tprint_info(f"Starting setup_feature_selector")
        selector_config = {
            'causal_graph': causal_graph,
            'technical_feature_patterns': technical_patterns,
            **kwargs
        }
        selector_config.update(self.feature_selector_params)
        
        self.feature_selector = NonCausalFeatureSelector(**selector_config)
        self.causal_graph = causal_graph
        
        if self.verbose:
            tprint_success("✅ Feature selector initialized")

        tprint_info(f"Completed setup_feature_selector")
    def setup_conflict_detector(self, **kwargs):
        """Setup the conflict detector."""
        tprint_info(f"Starting setup_conflict_detector")
        detector_config = {**kwargs}
        detector_config.update(self.conflict_detector_params)
        
        self.conflict_detector = ConflictDetector(**detector_config)
        
        if self.verbose:
            tprint_success("✅ Conflict detector initialized")

        tprint_info(f"Completed setup_conflict_detector")
    def prepare_training_data(
        self,
        df: pd.DataFrame,
        target_col: str,
        causal_anchor_prediction: Union[pd.Series, np.ndarray],
        all_feature_cols: List[str],
        causal_parent_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data for the Chaser.
        
        Args:
            df: Training DataFrame
            target_col: Target column name
            causal_anchor_prediction: Causal Anchor predictions
            all_feature_cols: All available feature columns
            causal_parent_cols: Known causal parent columns
            
        Returns:
            Tuple of (X_non_causal, y_residuals)
        """
        tprint_info(f"Starting prepare_training_data")
        try:
            if self.verbose:
                tprint_info("🔧 Preparing Layer 2.5 training data...")
            
            # Step 1: Compute causal residuals
            y_actual = df[target_col]
            y_residuals = compute_causal_residuals(
                y_actual, causal_anchor_prediction, verbose=self.verbose
            )
            
            # Step 2: Select non-causal features
            if self.feature_selector is None:
                # Default feature selection
                if causal_parent_cols is None:
                    causal_parent_cols = ['volume', 'volatility', 'liquidity', 'inventory']
                
                non_causal_features = [col for col in all_feature_cols if col not in causal_parent_cols]
                non_causal_features = [col for col in non_causal_features if col in df.columns]
                
                if self.verbose:
                    tprint_info(f"   - Using default feature selection: {len(non_causal_features)} features")
            else:
                # Use feature selector
                selection_results = self.feature_selector.select_non_causal_features(
                    all_feature_cols, target_col=target_col
                )
                non_causal_features = selection_results['selected_features']
                self.feature_selection_results = selection_results
                
                if self.verbose:
                    tprint_info(f"   - Feature selector: {len(non_causal_features)} features selected")
            
            # Step 3: Prepare feature matrix
            X_non_causal = df[non_causal_features].copy()
            
            # Step 4: Align data
            valid_mask = ~(X_non_causal.isna().any(axis=1) | y_residuals.isna())
            X_clean = X_non_causal[valid_mask]
            y_clean = y_residuals[valid_mask]
            
            # Store for reference
            self.training_features = non_causal_features
            self.training_residuals = y_clean
            
            if self.verbose:
                tprint_success("✅ Training data prepared:")
                tprint_info(f"   - Samples: {len(X_clean)}")
                tprint_info(f"   - Features: {len(X_clean.columns)}")
                tprint_info(f"   - Target mean: {y_clean.mean():.6f}")
                tprint_info(f"   - Target std: {y_clean.std():.6f}")
            
            tprint_info(f"Completed prepare_training_data")
            return X_clean, y_clean
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Training data preparation failed: {e}")
            raise

        tprint_info(f"Completed prepare_training_data")
    def train_chaser(
        self,
        X_non_causal: pd.DataFrame,
        y_residuals: pd.Series,
        cv_folds: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the Chaser model on causal residuals.
        
        Args:
            X_non_causal: Non-causal features
            y_residuals: Causal residuals
            cv_folds: Cross-validation folds
            **kwargs: Additional training parameters
            
        Returns:
            Training metrics
        """
        tprint_info(f"Starting train_chaser")
        try:
            if self.logger:
                self.logger.log_training_start(len(X_non_causal), len(X_non_causal.columns))

            if self.verbose:
                tprint_info("🚀 Training Layer 2.5 Chaser...")

            if self.chaser is None:
                self.setup_chaser()

            # Train the Chaser
            training_metrics = self.chaser.fit(
                X_non_causal, y_residuals, cv_folds=cv_folds, **kwargs
            )

            self.chaser_metrics = training_metrics

            # Analyze residuals if enabled
            if self.enable_residual_analysis:
                if self.verbose:
                    tprint_info("📊 Analyzing residual quality...")

                # This would need the original y_actual and y_anchor
                # For now, store basic residual stats
                self.residual_analysis = {
                    'residual_mean': y_residuals.mean(),
                    'residual_std': y_residuals.std(),
                    'residual_skew': y_residuals.skew(),
                    'sample_count': len(y_residuals)
                }

            if self.logger:
                self.logger.log_training_complete(training_metrics, self.feature_selection_results)

            if self.verbose:
                tprint_success("✅ Chaser training complete!")
                if isinstance(training_metrics, dict):
                    # Log some high-level metrics if available
                    for k, v in training_metrics.items():
                        if isinstance(v, (int, float)):
                            tprint_info(f"   📊 {k}: {v:.4f}")

            tprint_info(f"Completed train_chaser")
            return training_metrics

        except Exception as e:
            if self.logger:
                self.logger.log_error("Chaser Training", e, {"samples": len(X_non_causal), "features": len(X_non_causal.columns)})
            if self.verbose:
                tprint_error(f"❌ Chaser training failed: {e}")
            raise

        tprint_info(f"Completed train_chaser")
    def predict_with_conflict_detection(
        self,
        X_non_causal: pd.DataFrame,
        causal_anchor_prediction: Union[pd.Series, np.ndarray],
        return_conflicts: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Generate Chaser predictions with conflict detection.
        
        Args:
            X_non_causal: Non-causal features
            causal_anchor_prediction: Causal Anchor predictions
            return_conflicts: Whether to perform conflict detection
            
        Returns:
            Dictionary with predictions and conflict information
        """
        tprint_info(f"Starting predict_with_conflict_detection")
        try:
            if self.logger:
                self.logger.log_prediction_start(len(X_non_causal))

            if self.verbose:
                tprint_info("🔮 Generating Chaser predictions...")

            if self.chaser is None:
                raise ValueError("Chaser not trained. Call train_chaser() first.")

            # Get Chaser predictions (including individual models)
            individual_preds, chaser_confidence = self.chaser.predict(
                X_non_causal, return_individual=True, return_confidence=True
            )

            # Extract ensemble mean for backward compatibility
            chaser_prediction = individual_preds.get("ensemble_mean")
            if chaser_prediction is None:
                # Fallback if only one model or ensemble_mean not returned
                chaser_prediction = next(iter(individual_preds.values()))

            results = {
                'chaser_prediction': chaser_prediction,
                'chaser_confidence': chaser_confidence,
                'anchor_prediction': causal_anchor_prediction,
                'total_prediction': causal_anchor_prediction + chaser_prediction,
                'individual_predictions': individual_preds
            }

            # Conflict detection
            if return_conflicts and self.enable_conflict_detection:
                if self.conflict_detector is None:
                    self.setup_conflict_detector()

                conflict_results = self.conflict_detector.detect_conflicts(
                    chaser_prediction, causal_anchor_prediction, chaser_confidence
                )

                results.update(conflict_results)

                # Update conflict statistics
                self.conflict_statistics = self.conflict_detector.get_meta_learner_signals()

            if self.logger:
                self.logger.log_prediction_complete(results)

            if self.verbose:
                n_samples = len(chaser_prediction)
                if 'high_conflict' in results:
                    n_conflicts = np.sum(results['high_conflict'])
                    conflict_rate = n_conflicts / n_samples
                    tprint_info(f"   - High conflicts: {n_conflicts}/{n_samples} ({conflict_rate:.2%})")

                tprint_info(f"   - Mean chaser prediction: {np.mean(chaser_prediction):.6f}")
                tprint_info(f"   - Mean chaser confidence: {np.mean(chaser_confidence):.3f}")

            tprint_info(f"Completed predict_with_conflict_detection")
            return results

        except Exception as e:
            if self.logger:
                self.logger.log_error("Prediction", e, {"samples": len(X_non_causal)})
            if self.verbose:
                tprint_error(f"❌ Prediction with conflict detection failed: {e}")
            raise

        tprint_info(f"Completed predict_with_conflict_detection")
    def get_meta_learner_features(
        self,
        prediction_results: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """
        Prepare advanced features for the Meta-Learner.

        Args:
            prediction_results: Results from predict_with_conflict_detection

        Returns:
            DataFrame with advanced meta-learner features
        """
        tprint_info(f"Starting get_meta_learner_features")
        try:
            n_samples = len(prediction_results['chaser_prediction'])
            meta_features = pd.DataFrame(index=pd.RangeIndex(n_samples))

            # Basic predictions
            meta_features['chaser_prediction'] = prediction_results['chaser_prediction']
            meta_features['chaser_confidence'] = prediction_results['chaser_confidence']
            meta_features['anchor_prediction'] = prediction_results['anchor_prediction']
            meta_features['total_prediction'] = prediction_results['total_prediction']

            # Individual Predictions (Feed all surviving models to downstream users)
            if 'individual_predictions' in prediction_results:
                for name, pred in prediction_results['individual_predictions'].items():
                    if name != "ensemble_mean":
                        meta_features[f"chaser_ind_{name}"] = pred

            # Prediction uncertainty measures
            if hasattr(self.chaser, 'xgb_model') and hasattr(self.chaser, 'cat_model'):
                try:
                    # Get individual model predictions
                    xgb_pred = self.chaser.xgb_model.predict(
                        prediction_results.get('X_features',
                        pd.DataFrame(index=meta_features.index))
                    )
                    cat_pred = self.chaser.cat_model.predict(
                        prediction_results.get('X_features',
                        pd.DataFrame(index=meta_features.index))
                    )

                    # Ensemble disagreement as uncertainty
                    meta_features['model_disagreement'] = np.abs(xgb_pred - cat_pred)
                    meta_features['prediction_variance'] = np.var([xgb_pred, cat_pred], axis=0)
                    meta_features['prediction_std'] = np.std([xgb_pred, cat_pred], axis=0)

                    # Model confidence correlation
                    meta_features['model_agreement'] = 1.0 - (meta_features['model_disagreement'] /
                                                            (np.abs(meta_features['chaser_prediction']) + 1e-8))
                except:
                    # Fallback if individual predictions fail
                    meta_features['model_disagreement'] = np.zeros(n_samples)
                    meta_features['prediction_variance'] = np.zeros(n_samples)
                    meta_features['prediction_std'] = np.zeros(n_samples)
                    meta_features['model_agreement'] = np.ones(n_samples)

            # Conflict features
            if 'conflict_intensity' in prediction_results:
                meta_features['conflict_intensity'] = prediction_results['conflict_intensity']
                meta_features['high_conflict'] = prediction_results['high_conflict'].astype(int)
                meta_features['direction_conflict'] = prediction_results['direction_conflict'].astype(int)
                meta_features['magnitude_conflict'] = prediction_results['magnitude_conflict'].astype(int)

                # Conflict network features
                meta_features['conflict_degree'] = self._calculate_conflict_degree(prediction_results)
                meta_features['conflict_isolation'] = self._calculate_conflict_isolation(prediction_results)

                # Conflict momentum (rolling conflict rate)
                window_size = min(20, n_samples)
                if n_samples >= window_size:
                    meta_features['conflict_momentum'] = (
                        prediction_results['high_conflict'].rolling(window=window_size).mean()
                    )
                    meta_features['conflict_volatility'] = (
                        prediction_results['high_conflict'].rolling(window=window_size).std()
                    )
                else:
                    meta_features['conflict_momentum'] = np.zeros(n_samples)
                    meta_features['conflict_volatility'] = np.zeros(n_samples)

            # Prediction stability features
            meta_features['prediction_stability'] = self._calculate_prediction_stability(
                prediction_results
            )

            # Relative strength features
            meta_features['chaser_relative_strength'] = (
                np.abs(meta_features['chaser_prediction']) /
                (np.abs(meta_features['anchor_prediction']) + 1e-8)
            )

            # Confidence-weighted predictions
            meta_features['confidence_weighted_prediction'] = (
                meta_features['chaser_prediction'] * meta_features['chaser_confidence']
            )

            # Interaction features
            if 'conflict_intensity' in meta_features.columns:
                meta_features['confidence_conflict_interaction'] = (
                    meta_features['chaser_confidence'] * meta_features['conflict_intensity']
                )

            # Temporal features (if we have time series data)
            if hasattr(prediction_results, 'index') and hasattr(prediction_results['index'], 'is_monotonic_increasing'):
                meta_features['prediction_momentum'] = (
                    meta_features['chaser_prediction'] - meta_features['chaser_prediction'].shift(1)
                ).fillna(0)

                meta_features['confidence_momentum'] = (
                    meta_features['chaser_confidence'] - meta_features['chaser_confidence'].shift(1)
                ).fillna(0)

            tprint_info(f"Completed get_meta_learner_features")
            return meta_features

        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Advanced meta-learner feature preparation failed: {e}")
            raise

        tprint_info(f"Completed get_meta_learner_features")
    def _calculate_conflict_degree(self, prediction_results):
        """Calculate conflict centrality/connectivity."""
        tprint_info(f"Starting _calculate_conflict_degree")
        if 'high_conflict' not in prediction_results:
            tprint_info(f"Completed _calculate_conflict_degree")
            return np.zeros(len(prediction_results['chaser_prediction']))

        high_conflicts = prediction_results['high_conflict']
        # Simple conflict degree - how many neighboring points are also in conflict
        window_size = min(5, len(high_conflicts))
        conflict_degree = pd.Series(high_conflicts).rolling(
            window=window_size, center=True, min_periods=1
        ).mean()

        tprint_info(f"Completed _calculate_conflict_degree")
        return conflict_degree.values

    def _calculate_conflict_isolation(self, prediction_results):
        """Calculate how isolated a conflict is from other conflicts."""
        tprint_info(f"Starting _calculate_conflict_isolation")
        if 'high_conflict' not in prediction_results:
            tprint_info(f"Completed _calculate_conflict_isolation")
            return np.zeros(len(prediction_results['chaser_prediction']))

        high_conflicts = prediction_results['high_conflict']
        conflict_mask = high_conflicts.astype(bool)

        # Isolation score: distance to nearest other conflict
        isolation_scores = np.zeros(len(high_conflicts))
        conflict_indices = np.where(conflict_mask)[0]

        for i, idx in enumerate(conflict_indices):
            distances = np.abs(conflict_indices - idx)
            distances[i] = np.inf  # Don't count distance to self
            min_distance = np.min(distances) if len(distances) > 1 else len(high_conflicts)
            isolation_scores[idx] = min_distance

        # Normalize by sequence length
        isolation_scores = isolation_scores / len(high_conflicts)

        tprint_info(f"Completed _calculate_conflict_isolation")
        return isolation_scores

    def _calculate_prediction_stability(self, prediction_results):
        """Calculate prediction stability over time."""
        tprint_info(f"Starting _calculate_prediction_stability")
        chaser_pred = prediction_results['chaser_prediction']

        if len(chaser_pred) < 10:
            tprint_info(f"Completed _calculate_prediction_stability")
            return np.ones(len(chaser_pred)) * 0.5  # Neutral stability

        # Rolling standard deviation as instability measure
        window_size = min(20, len(chaser_pred))
        rolling_std = pd.Series(chaser_pred).rolling(window=window_size).std()

        # Convert to stability score (1 = stable, 0 = unstable)
        stability = 1.0 - (rolling_std / (rolling_std.max() + 1e-8))
        stability = stability.fillna(0.5)  # Neutral for early windows

        tprint_info(f"Completed _calculate_prediction_stability")
        return stability.values
    
    def get_integration_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of the Layer 2.5 integration.
        
        Returns:
            Dictionary with integration summary
        """
        tprint_info(f"Starting get_integration_summary")
        summary = {
            'components_initialized': {
                'chaser': self.chaser is not None,
                'feature_selector': self.feature_selector is not None,
                'conflict_detector': self.conflict_detector is not None
            },
            'training_data': {
                'feature_count': len(self.training_features) if self.training_features is not None else 0,
                'sample_count': len(self.training_residuals) if self.training_residuals is not None else 0
            },
            'performance_metrics': self.chaser_metrics or {},
            'conflict_statistics': self.conflict_statistics or {},
            'residual_analysis': self.residual_analysis or {},
            'feature_selection': self.feature_selection_results or {}
        }
        
        # Save report automatically
        try:
            self._save_chaser_report(summary)
        except Exception as e:
            tprint_error(f"❌ Failed to save Chaser report: {e}")

        tprint_info(f"Completed get_integration_summary")
        return summary

    def _save_chaser_report(self, summary: Dict[str, Any]):
        """Save detailed Layer 2.5 Chaser report."""
        tprint_info(f"Starting _save_chaser_report")
        from pathlib import Path
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = outcomes_dir / f"layer2_5_chaser_report_{ts}.md"

        report = [
            f"# Layer 2.5 Chaser Report ({ts})",
            "",
            "## 1. System Status",
            f"- **Chaser Initialized:** {'✅' if summary['components_initialized']['chaser'] else '❌'}",
            f"- **Feature Selector:** {'✅' if summary['components_initialized']['feature_selector'] else '❌'}",
            f"- **Conflict Detector:** {'✅' if summary['components_initialized']['conflict_detector'] else '❌'}",
            "",
            "## 2. Data Profile",
            f"- **Training Samples:** {summary['training_data']['sample_count']}",
            f"- **Features Used:** {summary['training_data']['feature_count']}",
            "",
            "## 3. Residual Analysis",
        ]

        res = summary.get('residual_analysis', {})
        if res:
            report.append(f"- **Mean Residual:** {res.get('residual_mean', 0.0):.6f}")
            report.append(f"- **Std Residual:** {res.get('residual_std', 0.0):.6f}")
            report.append(f"- **Skewness:** {res.get('residual_skew', 0.0):.4f}")
        else:
            report.append("- No residual analysis available.")

        report.append("")
        report.append("## 4. Conflict Statistics")
        conf = summary.get('conflict_statistics', {})
        if conf:
            # Conflict stats might be complex, extract high level
            report.append("Detailed conflict stats available in integration summary.")
        else:
            report.append("- No conflict statistics available.")

        # Write report
        try:
            report_path.write_text("\n".join(report), encoding="utf-8")
            if self.verbose:
                tprint_success(f"✅ Saved Layer 2.5 Chaser report to {report_path}")
        except Exception as e:
            tprint_error(f"Failed to write report file: {e}")

        tprint_info(f"Completed _save_chaser_report")
    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val,
                               causal_anchor_val, n_calls=50):
        """Optimize hyperparameters for the entire pipeline."""
        tprint_info(f"Starting optimize_hyperparameters")
        optimizer = HyperparameterOptimizer()
        best_params, optimization_result = optimizer.optimize(
            X_train, y_train, X_val, y_val, causal_anchor_val, n_calls
        )

        # Apply best parameters to this instance
        self.setup_chaser(
            xgb_params=best_params['xgb_params'],
            cat_params=best_params['cat_params'],
            ensemble_weights=best_params['ensemble_weights']
        )
        self.setup_conflict_detector(**best_params['conflict_params'])

        tprint_info(f"Completed optimize_hyperparameters")
        return best_params, optimization_result

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        tprint_info(f"Starting get_performance_report")
        if self.logger:
            tprint_info(f"Completed get_performance_report")
            return self.logger.get_performance_report()
        else:
            tprint_info(f"Completed get_performance_report")
            return {"logging": "disabled"}

        tprint_info(f"Completed get_performance_report")
# Convenience functions
def quick_layer25_setup(
    causal_graph: Optional[Dict[str, List[str]]] = None,
    **kwargs
) -> Layer25Integration:
    """
    Quick Layer 2.5 setup with default parameters.
    
    Args:
        causal_graph: Causal graph for feature selection
        **kwargs: Additional parameters
        
    Returns:
        Configured Layer25Integration instance
    """
    tprint_info(f"Starting quick_layer25_setup")
    integration = Layer25Integration(**kwargs)
    integration.setup_chaser()
    integration.setup_feature_selector(causal_graph=causal_graph)
    integration.setup_conflict_detector()
    tprint_info(f"Completed quick_layer25_setup")
    return integration

def end_to_end_layer25(
    df: pd.DataFrame,
    target_col: str,
    causal_anchor_prediction: Union[pd.Series, np.ndarray],
    all_feature_cols: List[str],
    causal_graph: Optional[Dict[str, List[str]]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    End-to-end Layer 2.5 pipeline.
    
    Args:
        df: Training DataFrame
        target_col: Target column
        causal_anchor_prediction: Causal Anchor predictions
        all_feature_cols: All available features
        causal_graph: Causal graph
        **kwargs: Additional parameters
        
    Returns:
        Complete Layer 2.5 results
    """
    # Setup integration
    tprint_info(f"Starting end_to_end_layer25")
    integration = quick_layer25_setup(causal_graph=causal_graph, **kwargs)
    
    # Prepare training data
    X_train, y_train = integration.prepare_training_data(
        df, target_col, causal_anchor_prediction, all_feature_cols
    )
    
    # Train Chaser
    training_metrics = integration.train_chaser(X_train, y_train)
    
    # Generate predictions
    prediction_results = integration.predict_with_conflict_detection(X_train, causal_anchor_prediction)
    
    # Prepare meta-learner features
    meta_features = integration.get_meta_learner_features(prediction_results)
    
    tprint_info(f"Completed end_to_end_layer25")
    return {
        'training_metrics': training_metrics,
        'prediction_results': prediction_results,
        'meta_learner_features': meta_features,
        'integration_summary': integration.get_integration_summary()
    }
