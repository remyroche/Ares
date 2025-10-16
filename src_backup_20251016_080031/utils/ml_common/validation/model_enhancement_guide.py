"""
import warnings
Model Enhancement Guide for Overfitting and Underfitting

Comprehensive guide for addressing overfitting and underfitting issues
with specific actions and implementation strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class EnhancementAction:
    """Single enhancement action with implementation details."""
    action_type: str  # 'regularization', 'complexity', 'feature', 'ensemble', 'hyperparameter'
    action_name: str
    description: str
    implementation: str
    priority: int  # 1-5, 1 being highest priority
    expected_improvement: str  # 'low', 'medium', 'high'
    implementation_difficulty: str  # 'easy', 'medium', 'hard'
    code_example: str = ""

@dataclass
class ModelEnhancementPlan:
    """Comprehensive enhancement plan for a model."""
    model_name: str
    model_type: str
    current_issues: List[str]
    enhancement_actions: List[EnhancementAction]
    implementation_order: List[str]
    expected_outcomes: Dict[str, str]
    risk_assessment: str
    timeline_estimate: str

class ModelEnhancementGuide:
    """Comprehensive guide for model enhancement based on overfitting/underfitting detection."""
    
    def __init__(self):
        """Initialize the enhancement guide."""
        self.enhancement_actions = self._initialize_enhancement_actions()
        logger.info("✅ Model Enhancement Guide initialized")
    
    def create_enhancement_plan(self,
                               model_name: str,
                               model_type: str,
                               overfitting_report: Optional[Dict[str, Any]] = None,
                               underfitting_report: Optional[Dict[str, Any]] = None) -> ModelEnhancementPlan:
        """
        Create a comprehensive enhancement plan based on detection reports.
        
        Args:
            model_name: Name of the model
            model_type: Type of model
            overfitting_report: Overfitting detection report
            underfitting_report: Underfitting detection report
            
        Returns:
            ModelEnhancementPlan with specific actions
        """
        current_issues = []
        enhancement_actions = []
        
        # Analyze overfitting issues
        if overfitting_report and overfitting_report.get('is_overfitting', False):
            overfitting_actions = self._get_overfitting_actions(
                overfitting_report, model_type
            )
            enhancement_actions.extend(overfitting_actions)
            current_issues.append("Overfitting detected")
        
        # Analyze underfitting issues
        if underfitting_report and underfitting_report.get('is_underfitting', False):
            underfitting_actions = self._get_underfitting_actions(
                underfitting_report, model_type
            )
            enhancement_actions.extend(underfitting_actions)
            current_issues.append("Underfitting detected")
        
        # If no specific issues detected, suggest general improvements
        if not enhancement_actions:
            enhancement_actions = self._get_general_improvements(model_type)
            current_issues.append("General model improvement")
        
        # Sort actions by priority
        enhancement_actions.sort(key=lambda x: x.priority)
        
        # Create implementation order
        implementation_order = [action.action_name for action in enhancement_actions]
        
        # Generate expected outcomes
        expected_outcomes = self._generate_expected_outcomes(enhancement_actions)
        
        # Assess risks
        risk_assessment = self._assess_implementation_risks(enhancement_actions)
        
        # Estimate timeline
        timeline_estimate = self._estimate_implementation_timeline(enhancement_actions)
        
        return ModelEnhancementPlan(
            model_name=model_name,
            model_type=model_type,
            current_issues=current_issues,
            enhancement_actions=enhancement_actions,
            implementation_order=implementation_order,
            expected_outcomes=expected_outcomes,
            risk_assessment=risk_assessment,
            timeline_estimate=timeline_estimate
        )
    
    def _get_overfitting_actions(self, overfitting_report: Dict[str, Any], model_type: str) -> List[EnhancementAction]:
        """Get enhancement actions for overfitting issues."""
        actions = []
        severity = overfitting_report.get('severity', 'none')
        
        if severity in ['moderate', 'high', 'severe']:
            # High priority: Regularization
            actions.append(EnhancementAction(
                action_type="regularization",
                action_name="Increase Regularization",
                description="Add L1/L2 regularization to prevent overfitting",
                implementation="Increase regularization parameters (alpha, lambda, C)",
                priority=1,
                expected_improvement="high",
                implementation_difficulty="easy",
                code_example=self._get_regularization_code(model_type)
            ))
            
            # High priority: Early Stopping
            actions.append(EnhancementAction(
                action_type="regularization",
                action_name="Implement Early Stopping",
                description="Stop training when validation performance stops improving",
                implementation="Add early stopping with patience parameter",
                priority=1,
                expected_improvement="high",
                implementation_difficulty="easy",
                code_example=self._get_early_stopping_code(model_type)
            ))
        
        if severity in ['high', 'severe']:
            # Medium priority: Cross-validation
            actions.append(EnhancementAction(
                action_type="hyperparameter",
                action_name="Improve Cross-Validation",
                description="Use more robust cross-validation strategy",
                implementation="Implement time series CV or purged CV",
                priority=2,
                expected_improvement="medium",
                implementation_difficulty="medium",
                code_example=self._get_cv_improvement_code()
            ))
            
            # Medium priority: Feature Selection
            actions.append(EnhancementAction(
                action_type="feature",
                action_name="Feature Selection",
                description="Remove redundant or noisy features",
                implementation="Use feature importance, correlation analysis, or automated selection",
                priority=2,
                expected_improvement="medium",
                implementation_difficulty="medium",
                code_example=self._get_feature_selection_code()
            ))
        
        # Low priority: Ensemble methods
        actions.append(EnhancementAction(
            action_type="ensemble",
            action_name="Ensemble Methods",
            description="Use ensemble methods to reduce overfitting",
            implementation="Implement bagging, boosting, or stacking",
            priority=3,
            expected_improvement="medium",
            implementation_difficulty="medium",
            code_example=self._get_ensemble_code(model_type)
        ))
        
        return actions
    
    def _get_underfitting_actions(self, underfitting_report: Dict[str, Any], model_type: str) -> List[EnhancementAction]:
        """Get enhancement actions for underfitting issues."""
        actions = []
        severity = underfitting_report.get('severity', 'none')
        
        if severity in ['mild', 'moderate', 'severe']:
            # High priority: Increase Complexity
            actions.append(EnhancementAction(
                action_type="complexity",
                action_name="Increase Model Complexity",
                description="Increase model capacity to learn more patterns",
                implementation="Increase model parameters (layers, estimators, depth)",
                priority=1,
                expected_improvement="high",
                implementation_difficulty="easy",
                code_example=self._get_complexity_increase_code(model_type)
            ))
            
            # High priority: Feature Engineering
            actions.append(EnhancementAction(
                action_type="feature",
                action_name="Feature Engineering",
                description="Create more informative features",
                implementation="Add polynomial features, interactions, domain-specific features",
                priority=1,
                expected_improvement="high",
                implementation_difficulty="medium",
                code_example=self._get_feature_engineering_code()
            ))
        
        if severity in ['moderate', 'severe']:
            # Medium priority: Hyperparameter Tuning
            actions.append(EnhancementAction(
                action_type="hyperparameter",
                action_name="Advanced Hyperparameter Tuning",
                description="Use sophisticated HPO to find optimal parameters",
                implementation="Implement Bayesian optimization or grid search",
                priority=2,
                expected_improvement="medium",
                implementation_difficulty="medium",
                code_example=self._get_hpo_improvement_code()
            ))
            
            # Medium priority: Algorithm Change
            actions.append(EnhancementAction(
                action_type="complexity",
                action_name="Try Different Algorithm",
                description="Switch to more powerful algorithm",
                implementation="Try ensemble methods, neural networks, or advanced algorithms",
                priority=2,
                expected_improvement="high",
                implementation_difficulty="medium",
                code_example=self._get_algorithm_change_code(model_type)
            ))
        
        # Low priority: Data Augmentation
        actions.append(EnhancementAction(
            action_type="feature",
            action_name="Data Augmentation",
            description="Increase training data through augmentation",
            implementation="Use SMOTE, noise injection, or synthetic data generation",
            priority=3,
            expected_improvement="medium",
            implementation_difficulty="hard",
            code_example=self._get_data_augmentation_code()
        ))
        
        return actions
    
    def _get_general_improvements(self, model_type: str) -> List[EnhancementAction]:
        """Get general improvement actions when no specific issues are detected."""
        actions = [
            EnhancementAction(
                action_type="hyperparameter",
                action_name="Hyperparameter Optimization",
                description="Optimize hyperparameters for better performance",
                implementation="Use grid search or Bayesian optimization",
                priority=1,
                expected_improvement="medium",
                implementation_difficulty="easy",
                code_example=self._get_hpo_improvement_code()
            ),
            EnhancementAction(
                action_type="ensemble",
                action_name="Ensemble Methods",
                description="Use ensemble methods for better performance",
                implementation="Implement voting, bagging, or stacking",
                priority=2,
                expected_improvement="medium",
                implementation_difficulty="medium",
                code_example=self._get_ensemble_code(model_type)
            ),
            EnhancementAction(
                action_type="feature",
                action_name="Feature Engineering",
                description="Create more informative features",
                implementation="Add domain-specific features and interactions",
                priority=3,
                expected_improvement="medium",
                implementation_difficulty="medium",
                code_example=self._get_feature_engineering_code()
            )
        ]
        
        return actions
    
    def _get_regularization_code(self, model_type: str) -> str:
        """Get regularization code example for model type."""
        if 'tree' in model_type or 'forest' in model_type:
            return """
# For Random Forest / XGBoost / LightGBM
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,  # Reduce depth
    min_samples_split=10,  # Increase minimum samples
    min_samples_leaf=5,  # Increase minimum leaf samples
    max_features='sqrt',  # Limit features
    random_state=42
)
"""
        elif 'neural' in model_type or 'mlp' in model_type:
            return """
# For Neural Networks
model = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    alpha=0.01,  # L2 regularization
    learning_rate='adaptive',
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42
)
"""
        else:
            return """
# For Linear Models
model = Ridge(alpha=1.0)  # L2 regularization
# or
model = Lasso(alpha=0.1)  # L1 regularization
# or
model = ElasticNet(alpha=0.1, l1_ratio=0.5)  # Combined L1+L2
"""
    
    def _get_early_stopping_code(self, model_type: str) -> str:
        """Get early stopping code example for model type."""
        if 'tree' in model_type or 'forest' in model_type:
            return """
# For XGBoost / LightGBM
model = XGBRegressor(
    n_estimators=1000,
    early_stopping_rounds=10,
    eval_metric='rmse',
    random_state=42
)

# Train with early stopping
model.fit(X_train, y_train, 
          eval_set=[(X_val, y_val)],
          verbose=False)
"""
        elif 'neural' in model_type or 'mlp' in model_type:
            return """
# For Neural Networks
model = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    max_iter=1000,
    random_state=42
)
"""
        else:
            return """
# For Linear Models with SGD
model = SGDRegressor(
    learning_rate='adaptive',
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42
)
"""
    
    def _get_cv_improvement_code(self) -> str:
        """Get cross-validation improvement code example."""
        return """
# Time Series Cross-Validation
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate
    score = model.score(X_val, y_val)
    print(f"CV Score: {score}")

# Purged Cross-Validation for Financial Data
from mlxtend.evaluate import PurgedKFold

pkf = PurgedKFold(n_splits=5, t1=pd.Series(timestamps))
for train_idx, val_idx in pkf.split(X):
    # Similar to above
    pass
"""
    
    def _get_feature_selection_code(self) -> str:
        """Get feature selection code example."""
        return """
# Feature Importance Selection
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor

# Get feature importance
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Select features with importance > threshold
selector = SelectFromModel(rf, threshold=0.01)
X_selected = selector.fit_transform(X_train, y_train)

# Correlation-based selection
correlation_matrix = X_train.corr()
high_corr_pairs = np.where(np.triu(correlation_matrix, 1) > 0.95)
# Remove highly correlated features
"""
    
    def _get_ensemble_code(self, model_type: str) -> str:
        """Get ensemble code example for model type."""
        if 'classification' in model_type:
            return """
# Voting Classifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

ensemble = VotingClassifier([
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('lr', LogisticRegression(random_state=42)),
    ('svc', SVC(probability=True, random_state=42))
], voting='soft')

ensemble.fit(X_train, y_train)
"""
        else:
            return """
# Voting Regressor
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

ensemble = VotingRegressor([
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('lr', LinearRegression()),
    ('svr', SVR())
])

ensemble.fit(X_train, y_train)
"""
    
    def _get_complexity_increase_code(self, model_type: str) -> str:
        """Get complexity increase code example for model type."""
        if 'tree' in model_type or 'forest' in model_type:
            return """
# Increase Random Forest complexity
model = RandomForestRegressor(
    n_estimators=500,  # Increase from 100
    max_depth=15,  # Increase from 5
    min_samples_split=2,  # Decrease from 10
    min_samples_leaf=1,  # Decrease from 5
    max_features='sqrt',
    random_state=42
)
"""
        elif 'neural' in model_type or 'mlp' in model_type:
            return """
# Increase Neural Network complexity
model = MLPRegressor(
    hidden_layer_sizes=(200, 100, 50),  # More layers
    alpha=0.001,  # Reduce regularization
    learning_rate='adaptive',
    max_iter=2000,  # More iterations
    random_state=42
)
"""
        else:
            return """
# Increase Linear Model complexity
from sklearn.preprocessing import PolynomialFeatures

# Add polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_train)

# Use with regularized model
model = Ridge(alpha=0.1)
model.fit(X_poly, y_train)
"""
    
    def _get_feature_engineering_code(self) -> str:
        """Get feature engineering code example."""
        return """
# Polynomial Features
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_train)

# Interaction Terms
from sklearn.preprocessing import PolynomialFeatures

interaction = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_interaction = interaction.fit_transform(X_train)

# Domain-specific features for financial data
def create_financial_features(df):
    # Technical indicators
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['rsi'] = calculate_rsi(df['close'])
    df['bollinger_upper'] = df['sma_20'] + 2 * df['close'].rolling(20).std()
    df['bollinger_lower'] = df['sma_20'] - 2 * df['close'].rolling(20).std()
    
    # Lag features
    for lag in [1, 2, 3, 5, 10]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
    
    # Statistical features
    df['close_std_20'] = df['close'].rolling(20).std()
    df['close_skew_20'] = df['close'].rolling(20).skew()
    df['close_kurt_20'] = df['close'].rolling(20).kurt()
    
    return df
"""
    
    def _get_hpo_improvement_code(self) -> str:
        """Get HPO improvement code example."""
        return """
# Bayesian Optimization with Optuna
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
    }
    
    model = RandomForestRegressor(**params, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Get best parameters
best_params = study.best_params
model = RandomForestRegressor(**best_params, random_state=42)
"""
    
    def _get_algorithm_change_code(self, model_type: str) -> str:
        """Get algorithm change code example for model type."""
        if 'linear' in model_type:
            return """
# Change from Linear to Tree-based
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Try Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Try XGBoost
model = XGBRegressor(n_estimators=100, random_state=42)

# Try LightGBM
model = LGBMRegressor(n_estimators=100, random_state=42)
"""
        elif 'tree' in model_type:
            return """
# Change from Tree to Neural Network
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Try Neural Network
model = MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42)

# Try Gradient Boosting
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
"""
        else:
            return """
# Try Ensemble Methods
from sklearn.ensemble import VotingRegressor, BaggingRegressor, AdaBoostRegressor

# Voting Ensemble
ensemble = VotingRegressor([
    ('rf', RandomForestRegressor(n_estimators=100)),
    ('xgb', XGBRegressor(n_estimators=100)),
    ('lgb', LGBMRegressor(n_estimators=100))
])

# Bagging
model = BaggingRegressor(
    base_estimator=RandomForestRegressor(n_estimators=50),
    n_estimators=10,
    random_state=42
)
"""
    
    def _get_data_augmentation_code(self) -> str:
        """Get data augmentation code example."""
        return """
# SMOTE for imbalanced data
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Noise injection
def add_noise(X, noise_factor=0.01):
    noise = np.random.normal(0, noise_factor, X.shape)
    return X + noise

X_noisy = add_noise(X_train, noise_factor=0.01)

# Synthetic data generation
from sklearn.datasets import make_regression

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

except ImportError:
    
    cp = None

X_synthetic, y_synthetic = make_regression(
    n_samples=1000,
    n_features=X_train.shape[1],
    noise=0.1,
    random_state=42
)
"""
    
    def _generate_expected_outcomes(self, actions: List[EnhancementAction]) -> Dict[str, str]:
        """Generate expected outcomes for enhancement actions."""
        outcomes = {}
        
        for action in actions:
            if action.expected_improvement == "high":
                outcomes[action.action_name] = "Significant improvement expected (10-30%)"
            elif action.expected_improvement == "medium":
                outcomes[action.action_name] = "Moderate improvement expected (5-15%)"
            else:
                outcomes[action.action_name] = "Small improvement expected (1-5%)"
        
        return outcomes
    
    def _assess_implementation_risks(self, actions: List[EnhancementAction]) -> str:
        """Assess implementation risks for enhancement actions."""
        risk_factors = []
        
        for action in actions:
            if action.implementation_difficulty == "hard":
                risk_factors.append("High complexity implementation")
            elif action.implementation_difficulty == "medium":
                risk_factors.append("Medium complexity implementation")
        
        if len(risk_factors) > 3:
            return "High risk - Multiple complex implementations required"
        elif len(risk_factors) > 1:
            return "Medium risk - Some complex implementations required"
        else:
            return "Low risk - Mostly straightforward implementations"
    
    def _estimate_implementation_timeline(self, actions: List[EnhancementAction]) -> str:
        """Estimate implementation timeline for enhancement actions."""
        total_days = 0
        
        for action in actions:
            if action.implementation_difficulty == "easy":
                total_days += 1
            elif action.implementation_difficulty == "medium":
                total_days += 3
            else:  # hard
                total_days += 7
        
        if total_days <= 3:
            return f"{total_days} days - Quick implementation"
        elif total_days <= 10:
            return f"{total_days} days - Moderate implementation time"
        else:
            return f"{total_days} days - Extended implementation time"
    
    def _initialize_enhancement_actions(self) -> Dict[str, EnhancementAction]:
        """Initialize all available enhancement actions."""
        actions = {}
        
        # Regularization actions
        actions["l1_regularization"] = EnhancementAction(
            action_type="regularization",
            action_name="L1 Regularization",
            description="Add L1 regularization to prevent overfitting",
            implementation="Increase alpha parameter in Lasso/ElasticNet",
            priority=1,
            expected_improvement="high",
            implementation_difficulty="easy"
        )
        
        actions["l2_regularization"] = EnhancementAction(
            action_type="regularization",
            action_name="L2 Regularization",
            description="Add L2 regularization to prevent overfitting",
            implementation="Increase alpha parameter in Ridge/ElasticNet",
            priority=1,
            expected_improvement="high",
            implementation_difficulty="easy"
        )
        
        # Complexity actions
        actions["increase_complexity"] = EnhancementAction(
            action_type="complexity",
            action_name="Increase Model Complexity",
            description="Increase model capacity to learn more patterns",
            implementation="Increase parameters (layers, estimators, depth)",
            priority=1,
            expected_improvement="high",
            implementation_difficulty="easy"
        )
        
        # Feature actions
        actions["feature_engineering"] = EnhancementAction(
            action_type="feature",
            action_name="Feature Engineering",
            description="Create more informative features",
            implementation="Add polynomial features, interactions, domain features",
            priority=1,
            expected_improvement="high",
            implementation_difficulty="medium"
        )
        
        return actions

# Global instance
DEFAULT_ENHANCEMENT_GUIDE = ModelEnhancementGuide()

def get_enhancement_guide() -> ModelEnhancementGuide:
    """Get model enhancement guide instance."""
    return DEFAULT_ENHANCEMENT_GUIDE

def create_enhancement_plan(model_name: str,
                          model_type: str,
                          overfitting_report: Optional[Dict[str, Any]] = None,
                          underfitting_report: Optional[Dict[str, Any]] = None) -> ModelEnhancementPlan:
    """Convenience function to create enhancement plan."""
    guide = get_enhancement_guide()
    return guide.create_enhancement_plan(model_name, model_type, overfitting_report, underfitting_report)

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _vectorbt_apply_operation(self, data: pd.Series, func, 
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)
        
        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
