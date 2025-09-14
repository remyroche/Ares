"""
Enhanced HMM Training Components

This module provides comprehensive enhancements to the HMM training ML components,
including advanced models, ensemble methods, feature engineering, and optimization.
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
    BaggingClassifier, BaggingRegressor,
    AdaBoostClassifier, AdaBoostRegressor
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
from sklearn.feature_selection import (
    SelectKBest, SelectFromModel, RFE, RFECV,
    mutual_info_classif, mutual_info_regression
)
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV
import optuna
import lightgbm as lgb
import xgboost as xgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class EnhancedHMMModelTrainer:
    """Enhanced HMM model trainer with advanced ML techniques."""
    
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
            'lightgbm', 'xgboost', 'mlp', 'svm', 'logistic_regression'
        ])
        self.ensemble_methods = config.get('ensemble_methods', ['voting', 'stacking'])
        self.feature_selection_method = config.get('feature_selection', 'mutual_info')
        self.n_features = config.get('n_features', 50)
        self.hpo_trials = config.get('hpo_trials', 100)
        self.cv_folds = config.get('cv_folds', 5)
        
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
                
                # Linear models
                'logistic_regression': LogisticRegression(
                    C=1.0, max_iter=1000, random_state=42,
                    class_weight='balanced', multi_class='ovr'
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
                
                # Neural networks
                'mlp_regressor': MLPRegressor(
                    hidden_layer_sizes=(100, 50, 25), activation='relu',
                    solver='adam', alpha=0.001, learning_rate='adaptive',
                    max_iter=1000, random_state=42, early_stopping=True
                ),
                
                # Support Vector Machines
                'svr_rbf': SVR(kernel='rbf', C=1.0, gamma='scale'),
                'svr_poly': SVR(kernel='poly', degree=3, C=1.0),
                
                # Linear models
                'ridge': Ridge(alpha=1.0, random_state=42),
                'lasso': Lasso(alpha=0.1, random_state=42, max_iter=1000),
                'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=1000),
                
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
    
    def create_advanced_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set for regime prediction."""
        features = market_data.copy()
        
        # Basic price features
        features['returns'] = features['close'].pct_change()
        features['log_returns'] = np.log(features['close'] / features['close'].shift(1))
        features['price_change'] = features['close'].diff()
        features['price_change_pct'] = features['price_change'] / features['close'].shift(1)
        
        # Volatility features (multiple timeframes)
        for window in [5, 10, 20, 50]:
            features[f'volatility_{window}'] = features['returns'].rolling(window).std()
            features[f'volatility_ratio_{window}'] = features[f'volatility_{window}'] / features['volatility_20']
            features[f'volatility_ma_{window}'] = features[f'volatility_{window}'].rolling(window).mean()
        
        # Volume features
        features['volume_ma_5'] = features['volume'].rolling(5).mean()
        features['volume_ma_20'] = features['volume'].rolling(20).mean()
        features['volume_ratio'] = features['volume'] / features['volume_ma_20']
        features['volume_price_trend'] = features['volume'] * features['returns']
        features['volume_volatility'] = features['volume'].rolling(20).std()
        features['volume_momentum'] = features['volume'] / features['volume'].shift(5) - 1
        
        # Price momentum features
        for window in [5, 10, 20, 50]:
            features[f'price_ma_{window}'] = features['close'].rolling(window).mean()
            features[f'momentum_{window}'] = features['close'] / features['close'].shift(window) - 1
            features[f'price_position_{window}'] = (features['close'] - features[f'price_ma_{window}']) / features[f'price_ma_{window}']
            features[f'momentum_ratio_{window}'] = features[f'momentum_{window}'] / features['momentum_20']
        
        # Technical indicators
        features['rsi_14'] = self._calculate_rsi(features['close'], 14)
        features['rsi_21'] = self._calculate_rsi(features['close'], 21)
        features['rsi_ratio'] = features['rsi_14'] / features['rsi_21']
        
        features['macd'] = self._calculate_macd(features['close'])
        features['macd_signal'] = self._calculate_macd_signal(features['close'])
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        features['bollinger_position'] = self._calculate_bollinger_position(features['close'])
        features['bollinger_width'] = self._calculate_bollinger_width(features['close'])
        features['bollinger_squeeze'] = self._calculate_bollinger_squeeze(features['close'])
        
        # Stochastic oscillator
        features['stoch_k'] = self._calculate_stochastic_k(features)
        features['stoch_d'] = features['stoch_k'].rolling(3).mean()
        
        # High-frequency features
        features['high_low_ratio'] = features['high'] / features['low']
        features['close_position'] = (features['close'] - features['low']) / (features['high'] - features['low'])
        features['body_size'] = abs(features['close'] - features['open']) / features['close']
        features['upper_shadow'] = (features['high'] - np.maximum(features['open'], features['close'])) / features['close']
        features['lower_shadow'] = (np.minimum(features['open'], features['close']) - features['low']) / features['close']
        features['shadow_ratio'] = features['upper_shadow'] / (features['lower_shadow'] + 1e-8)
        
        # Regime-specific features
        features['regime_persistence'] = self._calculate_regime_persistence(features['close'])
        features['trend_strength'] = self._calculate_trend_strength(features['close'])
        features['mean_reversion_signal'] = self._calculate_mean_reversion_signal(features['close'])
        features['volatility_regime'] = self._calculate_volatility_regime(features['returns'])
        
        # Cross-timeframe features
        if 'high' in features.columns and 'low' in features.columns:
            features['daily_range'] = features['high'] - features['low']
            features['daily_range_ratio'] = features['daily_range'] / features['close']
            features['gap_up'] = (features['open'] - features['close'].shift(1)) / features['close'].shift(1)
            features['gap_down'] = (features['close'].shift(1) - features['open']) / features['close'].shift(1)
            features['gap_size'] = abs(features['gap_up']) + abs(features['gap_down'])
        
        # Statistical features
        for window in [10, 20, 50]:
            features[f'skewness_{window}'] = features['returns'].rolling(window).skew()
            features[f'kurtosis_{window}'] = features['returns'].rolling(window).kurt()
            features[f'quantile_25_{window}'] = features['returns'].rolling(window).quantile(0.25)
            features[f'quantile_75_{window}'] = features['returns'].rolling(window).quantile(0.75)
            features[f'quantile_range_{window}'] = features[f'quantile_75_{window}'] - features[f'quantile_25_{window}']
        
        # Remove NaN values
        features = features.dropna()
        
        # Select only numeric columns
        numeric_columns = features.select_dtypes(include=[np.number]).columns
        features = features[numeric_columns]
        
        return features
    
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
    
    def _calculate_bollinger_width(self, prices: pd.Series, window: int = 20, std_dev: int = 2) -> pd.Series:
        """Calculate Bollinger Bands width."""
        ma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = ma + (std * std_dev)
        lower = ma - (std * std_dev)
        return (upper - lower) / ma
    
    def _calculate_bollinger_squeeze(self, prices: pd.Series, window: int = 20, std_dev: int = 2) -> pd.Series:
        """Calculate Bollinger Bands squeeze indicator."""
        ma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = ma + (std * std_dev)
        lower = ma - (std * std_dev)
        return (upper - lower) / ma
    
    def _calculate_stochastic_k(self, data: pd.DataFrame, k_period: int = 14) -> pd.Series:
        """Calculate Stochastic %K."""
        lowest_low = data['low'].rolling(window=k_period).min()
        highest_high = data['high'].rolling(window=k_period).max()
        return 100 * (data['close'] - lowest_low) / (highest_high - lowest_low)
    
    def _calculate_regime_persistence(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate regime persistence indicator."""
        returns = prices.pct_change()
        positive_returns = (returns > 0).astype(int)
        return positive_returns.rolling(window=window).mean()
    
    def _calculate_trend_strength(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate trend strength indicator."""
        returns = prices.pct_change()
        return returns.rolling(window=window).apply(lambda x: np.corrcoef(x, np.arange(len(x)))[0, 1])
    
    def _calculate_mean_reversion_signal(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate mean reversion signal."""
        ma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        return (prices - ma) / std
    
    def _calculate_volatility_regime(self, returns: pd.Series, window: int = 20) -> pd.Series:
        """Calculate volatility regime indicator."""
        volatility = returns.rolling(window=window).std()
        volatility_ma = volatility.rolling(window=window).mean()
        return volatility / volatility_ma
    
    def select_features(self, X: pd.DataFrame, y: np.ndarray, method: str = 'mutual_info',
                       k: int = 50, is_classification: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """Advanced feature selection methods."""

        # Preprocess data to handle infinity and large values
        X_processed = X.copy()

        # Handle infinity values
        inf_mask = np.isinf(X_processed.values)
        if np.any(inf_mask):
            print(f"⚠️ Found {np.sum(inf_mask)} infinity values in data for enhanced HMM feature selection, replacing with finite values")

            # Replace positive infinity
            pos_inf_mask = np.isposinf(X_processed.values)
            if np.any(pos_inf_mask):
                finite_mask = np.isfinite(X_processed.values)
                if np.any(finite_mask):
                    max_finite = np.max(X_processed.values[finite_mask])
                    X_processed.values[pos_inf_mask] = max(max_finite * 10, 1e10)
                else:
                    X_processed.values[pos_inf_mask] = 1e10

            # Replace negative infinity
            neg_inf_mask = np.isneginf(X_processed.values)
            if np.any(neg_inf_mask):
                finite_mask = np.isfinite(X_processed.values)
                if np.any(finite_mask):
                    min_finite = np.min(X_processed.values[finite_mask])
                    X_processed.values[neg_inf_mask] = min(min_finite * 10, -1e10)
                else:
                    X_processed.values[neg_inf_mask] = -1e10

        # Clip extremely large values
        max_float64 = 1e308
        min_float64 = -1e308
        X_processed = X_processed.clip(min_float64, max_float64)

        # Use processed data
        X = X_processed

        if method == 'mutual_info':
            if is_classification:
                selector = SelectKBest(score_func=mutual_info_classif, k=k)
            else:
                selector = SelectKBest(score_func=mutual_info_regression, k=k)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
        elif method == 'rfe':
            # Use Random Forest as base estimator
            base_estimator = RandomForestClassifier(n_estimators=50, random_state=42) if is_classification else RandomForestRegressor(n_estimators=50, random_state=42)
            selector = RFE(estimator=base_estimator, n_features_to_select=k)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
        elif method == 'model_based':
            # Use model-based feature selection
            if is_classification:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            model.fit(X, y)
            selector = SelectFromModel(model, threshold='median')
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
        elif method == 'pca':
            # PCA dimensionality reduction
            pca = PCA(n_components=0.95)  # Keep 95% of variance
            X_selected = pca.fit_transform(X)
            selected_features = [f'PC_{i}' for i in range(X_selected.shape[1])]
            
        else:
            # No selection
            X_selected = X
            selected_features = X.columns.tolist()
        
        return pd.DataFrame(X_selected, columns=selected_features), selected_features
    
    def create_ensemble_models(self, models: Dict[str, Any], is_classification: bool) -> Dict[str, Any]:
        """Create ensemble models from individual models."""
        ensembles = {}
        
        if is_classification:
            # Voting ensemble
            ensembles['voting_ensemble'] = VotingClassifier(
                estimators=list(models.items()),
                voting='soft',  # Use predicted probabilities
                n_jobs=-1
            )
            
            # Stacking ensemble
            meta_learner = LogisticRegression(random_state=42, max_iter=1000)
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
            
            # Boosting ensemble
            ensembles['boosting_ensemble'] = AdaBoostClassifier(
                base_estimator=DecisionTreeClassifier(max_depth=3),
                n_estimators=100, random_state=42
            )
            
        else:
            # Voting ensemble
            ensembles['voting_ensemble'] = VotingRegressor(
                estimators=list(models.items()),
                n_jobs=-1
            )
            
            # Stacking ensemble
            meta_learner = Ridge(random_state=42)
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
            
            # Boosting ensemble
            ensembles['boosting_ensemble'] = AdaBoostRegressor(
                base_estimator=DecisionTreeRegressor(max_depth=3),
                n_estimators=100, random_state=42
            )
        
        return ensembles
    
    def optimize_hyperparameters_advanced(self, model_name: str, X: pd.DataFrame, y: np.ndarray, 
                                        is_classification: bool, n_trials: int = 100) -> Dict[str, Any]:
        """Advanced hyperparameter optimization using Optuna."""
        
        def objective(trial):
            if model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 5, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
                }
                model = RandomForestClassifier(**params) if is_classification else RandomForestRegressor(**params)
                
            elif model_name == 'lightgbm':
                params = {
                    'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)
                }
                if is_classification:
                    params['objective'] = 'multiclass'
                    params['num_class'] = len(np.unique(y))
                else:
                    params['objective'] = 'regression'
                model = lgb.LGBMClassifier(**params) if is_classification else lgb.LGBMRegressor(**params)
                
            elif model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)
                }
                model = xgb.XGBClassifier(**params) if is_classification else xgb.XGBRegressor(**params)
                
            elif model_name == 'mlp':
                params = {
                    'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', 
                        [(50,), (100,), (50, 25), (100, 50), (100, 50, 25)]),
                    'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic']),
                    'solver': trial.suggest_categorical('solver', ['adam', 'lbfgs']),
                    'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
                    'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'adaptive'])
                }
                model = MLPClassifier(**params) if is_classification else MLPRegressor(**params)
            
            # Cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) if is_classification else KFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy' if is_classification else 'r2')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': n_trials
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
            'best_score': 0.0
        }
        
        # Get advanced models
        models = self.get_advanced_models(is_classification, len(np.unique(y)))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, 
            stratify=y if is_classification else None
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
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
        'model_types': ['random_forest', 'extra_trees', 'gradient_boosting', 'mlp'],
        'ensemble_methods': ['voting', 'stacking'],
        'feature_selection': 'mutual_info',
        'n_features': 50,
        'hpo_trials': 50
    }
    
    trainer = EnhancedHMMModelTrainer(config)
    
    # Train models
    results = trainer.train_enhanced_models(X, y, is_classification=True)
    
    # Print results
    print("Enhanced HMM Training Results:")
    print(f"Best Model: {results['best_model']}")
    print(f"Best Score: {results['best_score']:.4f}")
    print("\nModel Performance:")
    for name, metrics in results['performance'].items():
        score = metrics.get('accuracy', 0.0)
        print(f"{name}: {score:.4f}")