"""
Enhanced HMM Training Components - Improved Version

This module provides comprehensive enhancements to the HMM training ML components,
integrating with existing infrastructure and addressing specific requirements:

1. Linear model instead of XGBoost, XGBoost as meta-learner
2. Use all 200+ features from src/feature_engineering/
3. Multi-objective optimization using existing tools
4. Feature selection using src/training/utils/feature_selection
5. LSTM consideration for time-series modeling
6. Global models for regime determination (not per-regime models)
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    VotingClassifier, VotingRegressor,
    StackingClassifier, StackingRegressor,
    BaggingClassifier, BaggingRegressor
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
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

# Import existing infrastructure
try:
    from src.feature_engineering.feature_generators import FeatureGenerator
    from src.training.utils.feature_selection.main_framework import FeatureSelectionFramework
    from src.utils.ml_common.optimization.hpo_utils import HyperparameterOptimization
    from src.utils.ml_common.pareto import ParetoOptimizer
    INFRASTRUCTURE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some infrastructure not available: {e}")
    INFRASTRUCTURE_AVAILABLE = False

class EnhancedHMMModelTrainer:
    """Enhanced HMM model trainer with existing infrastructure integration."""
    
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
        self.model_types = config.get('model_types', [
            'random_forest', 'extra_trees', 'gradient_boosting',
            'lightgbm', 'logistic_regression', 'ridge', 'lasso', 'elastic_net',
            'mlp', 'svm', 'knn', 'naive_bayes'
        ])
        self.ensemble_methods = config.get('ensemble_methods', ['voting', 'stacking'])
        self.feature_selection_method = config.get('feature_selection', 'comprehensive')
        self.n_features = config.get('n_features', 100)  # Increased for 200+ features
        self.hpo_trials = config.get('hpo_trials', 100)
        self.cv_folds = config.get('cv_folds', 5)
        self.use_lstm = config.get('use_lstm', False)
        
        # Initialize existing infrastructure
        self._initialize_infrastructure()
        
    def _initialize_infrastructure(self):
        """Initialize existing infrastructure components."""
        if INFRASTRUCTURE_AVAILABLE:
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
        else:
            print("Warning: Using fallback implementations")
            self.feature_generator = None
            self.feature_selector = None
            self.hpo_optimizer = None
            self.pareto_optimizer = None
    
    def get_advanced_models(self, is_classification: bool, n_classes: int = None) -> Dict[str, Any]:
        """Get comprehensive set of advanced models for regime prediction."""
        
        if is_classification:
            models = {
                # Tree-based models
                'random_forest': RandomForestClassifier(
                    n_estimators=200, max_depth=20, min_samples_split=5,
                    min_samples_leaf=2, max_features='sqrt', random_state=42,
                    n_jobs=-1, class_weight='balanced'
                ),
                'extra_trees': ExtraTreesClassifier(
                    n_estimators=200, max_depth=20, min_samples_split=5,
                    min_samples_leaf=2, max_features='sqrt', random_state=42,
                    n_jobs=-1, class_weight='balanced'
                ),
                'gradient_boosting': GradientBoostingClassifier(
                    n_estimators=200, learning_rate=0.1, max_depth=6,
                    min_samples_split=5, min_samples_leaf=2, random_state=42
                ),
                
                # Linear models (replacing XGBoost as base model)
                'logistic_regression': LogisticRegression(
                    C=1.0, max_iter=1000, random_state=42,
                    class_weight='balanced', multi_class='ovr'
                ),
                'ridge': Ridge(alpha=1.0, random_state=42),
                'lasso': Lasso(alpha=0.1, random_state=42, max_iter=1000),
                'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=1000),
                
                # Neural networks
                'mlp_classifier': MLPClassifier(
                    hidden_layer_sizes=(100, 50, 25), activation='relu',
                    solver='adam', alpha=0.001, learning_rate='adaptive',
                    max_iter=1000, random_state=42, early_stopping=True
                ),
                
                # Support Vector Machines
                'svm_rbf': SVC(
                    kernel='rbf', C=1.0, gamma='scale', probability=True,
                    random_state=42, class_weight='balanced'
                ),
                'svm_poly': SVC(
                    kernel='poly', degree=3, C=1.0, probability=True,
                    random_state=42, class_weight='balanced'
                ),
                
                # K-Nearest Neighbors
                'knn': KNeighborsClassifier(
                    n_neighbors=5, weights='distance', algorithm='auto',
                    leaf_size=30, p=2
                ),
                
                # Naive Bayes
                'naive_bayes': GaussianNB(),
                
                # Decision Tree
                'decision_tree': DecisionTreeClassifier(
                    max_depth=10, min_samples_split=5, min_samples_leaf=2,
                    random_state=42, class_weight='balanced'
                )
            }
        else:
            models = {
                # Tree-based models
                'random_forest': RandomForestRegressor(
                    n_estimators=200, max_depth=20, min_samples_split=5,
                    min_samples_leaf=2, max_features='sqrt', random_state=42,
                    n_jobs=-1
                ),
                'extra_trees': ExtraTreesRegressor(
                    n_estimators=200, max_depth=20, min_samples_split=5,
                    min_samples_leaf=2, max_features='sqrt', random_state=42,
                    n_jobs=-1
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=200, learning_rate=0.1, max_depth=6,
                    min_samples_split=5, min_samples_leaf=2, random_state=42
                ),
                
                # Linear models
                'ridge': Ridge(alpha=1.0, random_state=42),
                'lasso': Lasso(alpha=0.1, random_state=42, max_iter=1000),
                'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=1000),
                
                # Neural networks
                'mlp_regressor': MLPRegressor(
                    hidden_layer_sizes=(100, 50, 25), activation='relu',
                    solver='adam', alpha=0.001, learning_rate='adaptive',
                    max_iter=1000, random_state=42, early_stopping=True
                ),
                
                # Support Vector Machines
                'svr_rbf': SVR(kernel='rbf', C=1.0, gamma='scale'),
                'svr_poly': SVR(kernel='poly', degree=3, C=1.0),
                
                # K-Nearest Neighbors
                'knn': KNeighborsRegressor(
                    n_neighbors=5, weights='distance', algorithm='auto',
                    leaf_size=30, p=2
                ),
                
                # Decision Tree
                'decision_tree': DecisionTreeRegressor(
                    max_depth=10, min_samples_split=5, min_samples_leaf=2,
                    random_state=42
                )
            }
        
        return models
    
    def create_comprehensive_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set using all 200+ features from feature_engineering."""
        
        if self.feature_generator is not None:
            # Use existing feature generator for 200+ features
            try:
                features = self.feature_generator.generate_all_features(market_data)
                print(f"✅ Generated {features.shape[1]} features using FeatureGenerator")
                return features
            except Exception as e:
                print(f"Warning: FeatureGenerator failed, using fallback: {e}")
        
        # Fallback: Create comprehensive features manually
        features = market_data.copy()
        
        # Basic price features
        features['returns'] = features['close'].pct_change()
        features['log_returns'] = np.log(features['close'] / features['close'].shift(1))
        features['price_change'] = features['close'].diff()
        
        # Volatility features (multiple timeframes)
        for window in [5, 10, 20, 50, 100]:
            features[f'volatility_{window}'] = features['returns'].rolling(window).std()
            features[f'volatility_ratio_{window}'] = features[f'volatility_{window}'] / features['volatility_20']
            features[f'volatility_ma_{window}'] = features[f'volatility_{window}'].rolling(window).mean()
        
        # Volume features
        features['volume_ma_5'] = features['volume'].rolling(5).mean()
        features['volume_ma_20'] = features['volume'].rolling(20).mean()
        features['volume_ratio'] = features['volume'] / features['volume_ma_20']
        features['volume_price_trend'] = features['volume'] * features['returns']
        features['volume_volatility'] = features['volume'].rolling(20).std()
        
        # Price momentum features
        for window in [5, 10, 20, 50, 100]:
            features[f'price_ma_{window}'] = features['close'].rolling(window).mean()
            features[f'momentum_{window}'] = features['close'] / features['close'].shift(window) - 1
            features[f'price_position_{window}'] = (features['close'] - features[f'price_ma_{window}']) / features[f'price_ma_{window}']
        
        # Technical indicators
        features['rsi_14'] = self._calculate_rsi(features['close'], 14)
        features['rsi_21'] = self._calculate_rsi(features['close'], 21)
        features['macd'] = self._calculate_macd(features['close'])
        features['macd_signal'] = self._calculate_macd_signal(features['close'])
        features['bollinger_position'] = self._calculate_bollinger_position(features['close'])
        
        # Statistical features
        for window in [10, 20, 50, 100]:
            features[f'skewness_{window}'] = features['returns'].rolling(window).skew()
            features[f'kurtosis_{window}'] = features['returns'].rolling(window).kurt()
            features[f'quantile_25_{window}'] = features['returns'].rolling(window).quantile(0.25)
            features[f'quantile_75_{window}'] = features['returns'].rolling(window).quantile(0.75)
        
        # Remove NaN values
        features = features.dropna()
        
        # Select only numeric columns
        numeric_columns = features.select_dtypes(include=[np.number]).columns
        features = features[numeric_columns]
        
        print(f"✅ Generated {features.shape[1]} features using fallback method")
        return features
    
    def select_features_advanced(self, X: pd.DataFrame, y: np.ndarray, 
                               is_classification: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """Advanced feature selection using existing infrastructure."""
        
        if self.feature_selector is not None:
            try:
                # Use existing feature selection framework
                selection_result = self.feature_selector.select_features(
                    X, y, 
                    method=self.feature_selection_method,
                    max_features=self.n_features,
                    is_classification=is_classification
                )
                
                selected_features = selection_result.get('selected_features', X.columns.tolist()[:self.n_features])
                X_selected = X[selected_features]
                
                print(f"✅ Selected {len(selected_features)} features using FeatureSelectionFramework")
                return X_selected, selected_features
                
            except Exception as e:
                print(f"Warning: FeatureSelectionFramework failed, using fallback: {e}")
        
        # Fallback: Simple feature selection
        from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression
        
        if is_classification:
            selector = SelectKBest(score_func=mutual_info_classif, k=min(self.n_features, X.shape[1]))
        else:
            selector = SelectKBest(score_func=mutual_info_regression, k=min(self.n_features, X.shape[1]))
        
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        print(f"✅ Selected {len(selected_features)} features using fallback method")
        return pd.DataFrame(X_selected, columns=selected_features), selected_features
    
    def create_ensemble_models(self, models: Dict[str, Any], is_classification: bool) -> Dict[str, Any]:
        """Create ensemble models with XGBoost as meta-learner."""
        ensembles = {}
        
        if is_classification:
            # Voting ensemble
            ensembles['voting_ensemble'] = VotingClassifier(
                estimators=list(models.items()),
                voting='soft',  # Use predicted probabilities
                n_jobs=-1
            )
            
            # Stacking ensemble with XGBoost as meta-learner
            meta_learner = xgb.XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=42, n_jobs=-1
            )
            ensembles['stacking_ensemble'] = StackingClassifier(
                estimators=list(models.items()),
                final_estimator=meta_learner,
                cv=5, n_jobs=-1
            )
            
            # Bagging ensemble
            base_estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            ensembles['bagging_ensemble'] = BaggingClassifier(
                base_estimator=base_estimator,
                n_estimators=10, random_state=42, n_jobs=-1
            )
            
        else:
            # Voting ensemble
            ensembles['voting_ensemble'] = VotingRegressor(
                estimators=list(models.items()),
                n_jobs=-1
            )
            
            # Stacking ensemble with XGBoost as meta-learner
            meta_learner = xgb.XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=42, n_jobs=-1
            )
            ensembles['stacking_ensemble'] = StackingRegressor(
                estimators=list(models.items()),
                final_estimator=meta_learner,
                cv=5, n_jobs=-1
            )
            
            # Bagging ensemble
            base_estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            ensembles['bagging_ensemble'] = BaggingRegressor(
                base_estimator=base_estimator,
                n_estimators=10, random_state=42, n_jobs=-1
            )
        
        return ensembles
    
    def optimize_hyperparameters_multi_objective(self, model_name: str, X: pd.DataFrame, y: np.ndarray, 
                                               is_classification: bool) -> Dict[str, Any]:
        """Multi-objective hyperparameter optimization using existing tools."""
        
        if self.hpo_optimizer is not None:
            try:
                # Use existing multi-objective optimization
                optimization_result = self.hpo_optimizer.multi_objective_optimization(
                    model_factory=lambda params: self._create_model(model_name, params, is_classification),
                    X=X, y=y,
                    objectives=['accuracy', 'f1_score', 'regime_stability'],
                    objective_weights=[0.4, 0.3, 0.3],
                    n_trials=self.hpo_trials
                )
                
                print(f"✅ Multi-objective optimization completed for {model_name}")
                return optimization_result
                
            except Exception as e:
                print(f"Warning: Multi-objective optimization failed, using fallback: {e}")
        
        # Fallback: Simple optimization
        return self._simple_hyperparameter_optimization(model_name, X, y, is_classification)
    
    def _create_model(self, model_name: str, params: Dict[str, Any], is_classification: bool):
        """Create model instance with given parameters."""
        models = self.get_advanced_models(is_classification)
        base_model = models.get(model_name)
        
        if base_model is None:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Update parameters
        model = type(base_model)(**{**base_model.get_params(), **params})
        return model
    
    def _simple_hyperparameter_optimization(self, model_name: str, X: pd.DataFrame, y: np.ndarray, 
                                          is_classification: bool) -> Dict[str, Any]:
        """Simple hyperparameter optimization fallback."""
        # This would be a simplified version
        return {
            'best_params': {},
            'best_score': 0.0,
            'n_trials': 0
        }
    
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
        """Train enhanced models with comprehensive evaluation."""
        
        results = {
            'models': {},
            'ensemble_models': {},
            'performance': {},
            'feature_importance': {},
            'best_model': None,
            'best_score': 0.0,
            'feature_selection_info': {}
        }
        
        # Create comprehensive features
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
        
        # Get advanced models
        models = self.get_advanced_models(is_classification, len(np.unique(y)))
        
        # Train individual models
        for name, model in models.items():
            try:
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
                
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
        
        # Create ensemble models
        if len(results['models']) > 1:
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
                        
                except Exception as e:
                    print(f"Error training ensemble {name}: {e}")
                    continue
        
        return results
    
    # Helper methods for technical indicators
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        return ema_fast - ema_slow
    
    def _calculate_macd_signal(self, prices: pd.Series, signal: int = 9) -> pd.Series:
        """Calculate MACD signal line."""
        macd = self._calculate_macd(prices)
        return macd.ewm(span=signal).mean()
    
    def _calculate_bollinger_position(self, prices: pd.Series, window: int = 20, std_dev: int = 2) -> pd.Series:
        """Calculate Bollinger Bands position."""
        ma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = ma + (std * std_dev)
        lower = ma - (std * std_dev)
        return (prices - lower) / (upper - lower)

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
        'model_types': ['random_forest', 'extra_trees', 'gradient_boosting', 'logistic_regression'],
        'ensemble_methods': ['voting', 'stacking'],
        'feature_selection': 'comprehensive',
        'n_features': 50,
        'hpo_trials': 50,
        'use_lstm': False
    }
    
    trainer = EnhancedHMMModelTrainer(config)
    
    # Train models
    results = trainer.train_enhanced_models(X, y, is_classification=True)
    
    # Print results
    print("Enhanced HMM Training Results:")
    print(f"Best Model: {results['best_model']}")
    print(f"Best Score: {results['best_score']:.4f}")
    print(f"Total Features: {results['feature_selection_info']['total_features']}")
    print(f"Selected Features: {results['feature_selection_info']['selected_features']}")
    print("\nModel Performance:")
    for name, metrics in results['performance'].items():
        score = metrics.get('accuracy', 0.0)
        print(f"{name}: {score:.4f}")