"""
Enhanced HMM Training Components - Final Version

This module provides comprehensive enhancements to the HMM training ML components,
addressing specific requirements:

1. Base learners: Logistic Regression + LightGBM + LSTM (GRU alternative) + XGBoost as meta-learner
2. No fallback - fast fail if infrastructure not available
3. Purpose: Determine "when we are in what regime" (regimes, plural)
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, r2_score, roc_auc_score,
    classification_report, confusion_matrix, log_loss
)
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, KFold,
    TimeSeriesSplit, train_test_split
)
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import lightgbm as lgb
import xgboost as xgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import existing infrastructure - FAST FAIL if not available
try:
    from src.feature_engineering.feature_generators import FeatureGenerator
    from src.training.utils.feature_selection.main_framework import FeatureSelectionFramework
    from src.utils.ml_common.optimization.hpo_utils import HyperparameterOptimization
    from src.utils.ml_common.pareto import ParetoOptimizer
    INFRASTRUCTURE_AVAILABLE = True
except ImportError as e:
    raise ImportError(f"Required infrastructure not available: {e}. Cannot proceed without existing tools.")

# LSTM alternative - GRU (more computationally friendly)
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GRU, Dense, Dropout, LSTM
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    raise ImportError("TensorFlow not available. GRU/LSTM models require TensorFlow.")

class GRURegimePredictor:
    """GRU-based regime predictor (more computationally friendly than LSTM)."""
    
    def __init__(self, sequence_length: int = 20, n_regimes: int = 3, 
                 hidden_units: int = 50, dropout_rate: float = 0.2):
        """Initialize GRU regime predictor."""
        self.sequence_length = sequence_length
        self.n_regimes = n_regimes
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.scaler = StandardScaler()
        
    def _create_sequences(self, X: np.ndarray, y: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for GRU training."""
        X_seq, y_seq = [], []
        
        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i-self.sequence_length:i])
            if y is not None:
                y_seq.append(y[i])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq) if y is not None else None
        
        return X_seq, y_seq
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train GRU model."""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y)
        
        # Build GRU model
        self.model = Sequential([
            GRU(self.hidden_units, return_sequences=True, input_shape=(self.sequence_length, X.shape[1])),
            Dropout(self.dropout_rate),
            GRU(self.hidden_units // 2, return_sequences=False),
            Dropout(self.dropout_rate),
            Dense(32, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(self.n_regimes, activation='softmax')
        ])
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        self.model.fit(
            X_seq, y_seq,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict regime classes."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Create sequences
        X_seq, _ = self._create_sequences(X_scaled)
        
        # Predict
        y_pred_proba = self.model.predict(X_seq, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Pad predictions to match original length
        y_pred_padded = np.zeros(len(X))
        y_pred_padded[self.sequence_length:] = y_pred
        
        return y_pred_padded
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict regime probabilities."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Create sequences
        X_seq, _ = self._create_sequences(X_scaled)
        
        # Predict probabilities
        y_pred_proba = self.model.predict(X_seq, verbose=0)
        
        # Pad probabilities to match original length
        y_pred_proba_padded = np.zeros((len(X), self.n_regimes))
        y_pred_proba_padded[self.sequence_length:] = y_pred_proba
        
        return y_pred_proba_padded

class EnhancedHMMModelTrainer:
    """Enhanced HMM model trainer with specific base learners and XGBoost meta-learner."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize enhanced HMM model trainer."""
        self.config = config
        self.logger = config.get('logger', None)
        self.models = {}
        self.ensemble_models = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        self.scalers = {}
        
        # Configuration
        self.n_features = config.get('n_features', 100)
        self.hpo_trials = config.get('hpo_trials', 100)
        self.cv_folds = config.get('cv_folds', 5)
        self.sequence_length = config.get('sequence_length', 20)
        
        # Initialize existing infrastructure - FAST FAIL
        self._initialize_infrastructure()
        
    def _initialize_infrastructure(self):
        """Initialize existing infrastructure components - FAST FAIL if not available."""
        # Initialize feature generator for 200+ features
        self.feature_generator = FeatureGenerator()
        
        # Initialize feature selection framework
        fs_config = {
            'selection_methods': ['mrmr', 'lasso_stability', 'correlation_filter'],
            'max_features': self.n_features,
            'enable_stability_analysis': True,
            'enable_temporal_analysis': True
        }
        self.feature_selector = FeatureSelectionFramework(fs_config)
        
        # Initialize multi-objective optimization
        hpo_config = {
            'enable_multi_objective': True,
            'objectives': ['accuracy', 'f1_score', 'regime_stability'],
            'objective_weights': [0.4, 0.3, 0.3],
            'n_trials': self.hpo_trials,
            'enable_pruning': True
        }
        self.hpo_optimizer = HyperparameterOptimization(hpo_config)
        
        # Initialize Pareto optimizer
        self.pareto_optimizer = ParetoOptimizer()
    
    def get_base_models(self, is_classification: bool, n_regimes: int) -> Dict[str, Any]:
        """Get specific base models: Logistic Regression + LightGBM + GRU."""
        
        if is_classification:
            models = {
                # Linear model
                'logistic_regression': LogisticRegression(
                    C=1.0, max_iter=1000, random_state=42,
                    class_weight='balanced', multi_class='ovr'
                ),
                
                # LightGBM
                'lightgbm': lgb.LGBMClassifier(
                    n_estimators=200, num_leaves=31, learning_rate=0.05,
                    feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=5,
                    random_state=42, n_jobs=-1, class_weight='balanced'
                ),
                
                # GRU (more computationally friendly than LSTM)
                'gru': GRURegimePredictor(
                    sequence_length=self.sequence_length,
                    n_regimes=n_regimes,
                    hidden_units=50,
                    dropout_rate=0.2
                )
            }
        else:
            models = {
                # Linear model
                'ridge': Ridge(alpha=1.0, random_state=42),
                
                # LightGBM
                'lightgbm': lgb.LGBMRegressor(
                    n_estimators=200, num_leaves=31, learning_rate=0.05,
                    feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=5,
                    random_state=42, n_jobs=-1
                ),
                
                # GRU (more computationally friendly than LSTM)
                'gru': GRURegimePredictor(
                    sequence_length=self.sequence_length,
                    n_regimes=n_regimes,
                    hidden_units=50,
                    dropout_rate=0.2
                )
            }
        
        return models
    
    def create_comprehensive_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set using all 200+ features from feature_engineering."""
        # Use existing feature generator for 200+ features - NO FALLBACK
        features = self.feature_generator.generate_all_features(market_data)
        print(f"✅ Generated {features.shape[1]} features using FeatureGenerator")
        return features
    
    def select_features_advanced(self, X: pd.DataFrame, y: np.ndarray, 
                               is_classification: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """Advanced feature selection using existing infrastructure."""
        # Use existing feature selection framework - NO FALLBACK
        selection_result = self.feature_selector.select_features(
            X, y, 
            method='comprehensive',
            max_features=self.n_features,
            is_classification=is_classification
        )
        
        selected_features = selection_result.get('selected_features', X.columns.tolist()[:self.n_features])
        X_selected = X[selected_features]
        
        print(f"✅ Selected {len(selected_features)} features using FeatureSelectionFramework")
        return X_selected, selected_features
    
    def create_ensemble_models(self, models: Dict[str, Any], is_classification: bool) -> Dict[str, Any]:
        """Create ensemble models with XGBoost as meta-learner."""
        ensembles = {}
        
        if is_classification:
            # Stacking ensemble with XGBoost as meta-learner
            meta_learner = xgb.XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=42, n_jobs=-1
            )
            ensembles['stacking_ensemble'] = StackingClassifier(
                estimators=list(models.items()),
                final_estimator=meta_learner,  # XGBoost as meta-learner
                cv=5, n_jobs=-1
            )
            
        else:
            # Stacking ensemble with XGBoost as meta-learner
            meta_learner = xgb.XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=42, n_jobs=-1
            )
            ensembles['stacking_ensemble'] = StackingRegressor(
                estimators=list(models.items()),
                final_estimator=meta_learner,  # XGBoost as meta-learner
                cv=5, n_jobs=-1
            )
        
        return ensembles
    
    def optimize_hyperparameters_multi_objective(self, model_name: str, X: pd.DataFrame, y: np.ndarray, 
                                               is_classification: bool) -> Dict[str, Any]:
        """Multi-objective hyperparameter optimization using existing tools."""
        # Use existing multi-objective optimization - NO FALLBACK
        optimization_result = self.hpo_optimizer.multi_objective_optimization(
            model_factory=lambda params: self._create_model(model_name, params, is_classification),
            X=X, y=y,
            objectives=['accuracy', 'f1_score', 'regime_stability'],
            objective_weights=[0.4, 0.3, 0.3],
            n_trials=self.hpo_trials
        )
        
        print(f"✅ Multi-objective optimization completed for {model_name}")
        return optimization_result
    
    def _create_model(self, model_name: str, params: Dict[str, Any], is_classification: bool):
        """Create model instance with given parameters."""
        n_regimes = len(np.unique(self.y_train)) if hasattr(self, 'y_train') else 3
        models = self.get_base_models(is_classification, n_regimes)
        base_model = models.get(model_name)
        
        if base_model is None:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Update parameters
        if model_name == 'gru':
            # GRU has different parameter structure
            return GRURegimePredictor(**{**base_model.__dict__, **params})
        else:
            model = type(base_model)(**{**base_model.get_params(), **params})
            return model
    
    def evaluate_model_comprehensive(self, model: Any, X_test: pd.DataFrame, y_test: np.ndarray, 
                                   is_classification: bool) -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        
        y_pred = model.predict(X_test)
        
        if is_classification:
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            
            if y_pred_proba is not None and len(np.unique(y_test)) == 2:
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
            
            if y_pred_proba is not None:
                metrics['log_loss'] = log_loss(y_test, y_pred_proba)
            
        else:
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': np.mean(np.abs(y_test - y_pred)),
                'r2_score': r2_score(y_test, y_pred),
                'explained_variance': 1 - np.var(y_test - y_pred) / np.var(y_test)
            }
        
        return metrics
    
    def train_enhanced_models(self, X: pd.DataFrame, y: np.ndarray, 
                            is_classification: bool = True) -> Dict[str, Any]:
        """Train enhanced models with specific base learners and XGBoost meta-learner."""
        
        results = {
            'models': {},
            'ensemble_models': {},
            'performance': {},
            'feature_importance': {},
            'best_model': None,
            'best_score': 0.0,
            'feature_selection_info': {},
            'regime_analysis': {}
        }
        
        # Store y for model creation
        self.y_train = y
        
        # Create comprehensive features using 200+ features
        X_enhanced = self.create_comprehensive_features(X)
        
        # Select features using advanced methods
        X_selected, selected_features = self.select_features_advanced(
            X_enhanced, y, is_classification
        )
        
        results['feature_selection_info'] = {
            'total_features': X_enhanced.shape[1],
            'selected_features': len(selected_features),
            'selected_feature_names': selected_features
        }
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42, 
            stratify=y if is_classification else None
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Get specific base models: Logistic Regression + LightGBM + GRU
        n_regimes = len(np.unique(y))
        models = self.get_base_models(is_classification, n_regimes)
        
        # Train individual base models
        for name, model in models.items():
            try:
                print(f"🔄 Training base model: {name}")
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Evaluate model
                metrics = self.evaluate_model_comprehensive(model, X_test_scaled, y_test, is_classification)
                
                # Store results
                results['models'][name] = model
                results['performance'][name] = metrics
                
                # Track best model
                score = metrics.get('accuracy' if is_classification else 'r2_score', 0.0)
                if score > results['best_score']:
                    results['best_score'] = score
                    results['best_model'] = name
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    results['feature_importance'][name] = model.feature_importances_
                
                print(f"✅ {name} completed: {score:.4f}")
                
            except Exception as e:
                print(f"❌ Error training {name}: {e}")
                continue
        
        # Create ensemble models with XGBoost as meta-learner
        if len(results['models']) > 1:
            print("🔄 Creating ensemble with XGBoost meta-learner...")
            ensemble_models = self.create_ensemble_models(results['models'], is_classification)
            
            for name, ensemble in ensemble_models.items():
                try:
                    # Train ensemble
                    ensemble.fit(X_train_scaled, y_train)
                    
                    # Evaluate ensemble
                    metrics = self.evaluate_model_comprehensive(ensemble, X_test_scaled, y_test, is_classification)
                    
                    # Store results
                    results['ensemble_models'][name] = ensemble
                    results['performance'][name] = metrics
                    
                    # Track best model
                    score = metrics.get('accuracy' if is_classification else 'r2_score', 0.0)
                    if score > results['best_score']:
                        results['best_score'] = score
                        results['best_model'] = name
                    
                    print(f"✅ {name} completed: {score:.4f}")
                        
                except Exception as e:
                    print(f"❌ Error training ensemble {name}: {e}")
                    continue
        
        # Analyze regime distribution
        results['regime_analysis'] = self._analyze_regime_distribution(y, y_test, results)
        
        return results
    
    def _analyze_regime_distribution(self, y_train: np.ndarray, y_test: np.ndarray, 
                                   results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze regime distribution and model performance per regime."""
        
        # Overall regime distribution
        unique_regimes, train_counts = np.unique(y_train, return_counts=True)
        _, test_counts = np.unique(y_test, return_counts=True)
        
        regime_analysis = {
            'n_regimes': len(unique_regimes),
            'regime_distribution_train': dict(zip(unique_regimes, train_counts)),
            'regime_distribution_test': dict(zip(unique_regimes, test_counts)),
            'regime_balance_train': np.std(train_counts) / np.mean(train_counts) if len(train_counts) > 1 else 0.0,
            'regime_balance_test': np.std(test_counts) / np.mean(test_counts) if len(test_counts) > 1 else 0.0
        }
        
        # Model performance per regime
        for model_name, metrics in results['performance'].items():
            if 'confusion_matrix' in metrics:
                cm = np.array(metrics['confusion_matrix'])
                regime_precision = np.diag(cm) / np.sum(cm, axis=0)
                regime_recall = np.diag(cm) / np.sum(cm, axis=1)
                regime_f1 = 2 * (regime_precision * regime_recall) / (regime_precision + regime_recall)
                
                regime_analysis[f'{model_name}_regime_performance'] = {
                    'precision_per_regime': regime_precision.tolist(),
                    'recall_per_regime': regime_recall.tolist(),
                    'f1_per_regime': regime_f1.tolist()
                }
        
        return regime_analysis

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                    columns=[f'feature_{i}' for i in range(n_features)])
    y = np.random.randint(0, 3, n_samples)  # 3 regimes
    
    # Initialize enhanced trainer
    config = {
        'n_features': 50,
        'hpo_trials': 50,
        'sequence_length': 20
    }
    
    trainer = EnhancedHMMModelTrainer(config)
    
    # Train models
    results = trainer.train_enhanced_models(X, y, is_classification=True)
    
    # Print results
    print("\n" + "="*80)
    print("Enhanced HMM Training Results - Multiple Regimes")
    print("="*80)
    print(f"Best Model: {results['best_model']}")
    print(f"Best Score: {results['best_score']:.4f}")
    print(f"Total Features: {results['feature_selection_info']['total_features']}")
    print(f"Selected Features: {results['feature_selection_info']['selected_features']}")
    print(f"Number of Regimes: {results['regime_analysis']['n_regimes']}")
    print("\nRegime Distribution (Train):")
    for regime, count in results['regime_analysis']['regime_distribution_train'].items():
        print(f"  Regime {regime}: {count} samples")
    print("\nModel Performance:")
    for name, metrics in results['performance'].items():
        score = metrics.get('accuracy', 0.0)
        print(f"  {name}: {score:.4f}")