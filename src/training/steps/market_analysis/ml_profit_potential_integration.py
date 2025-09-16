"""
ML Model Integration with Enhanced Profit Potential Labels

This module demonstrates how machine learning models can benefit from the enhanced
triple barrier method with meaningful profit potential labels. It provides:

1. Multi-target ML models that predict both direction and profit magnitude
2. Profit-aware loss functions that optimize for actual profit rather than accuracy
3. Confidence-weighted predictions that account for prediction uncertainty
4. Regime-specific model training and prediction
5. Profit potential feature importance analysis
6. Model performance evaluation with profit-focused metrics

Key Benefits for ML Models:
- Richer training signals from profit magnitude and confidence
- Better generalization through regime-specific training
- Profit-optimized loss functions instead of simple accuracy
- Uncertainty quantification through confidence scores
- Feature importance based on profit contribution
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime

import pandas as pd
import numpy as np

from src.utils.tprint import tprint
from src.utils.logger import get_logger

# Import ML libraries
try:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

@dataclass
class MLProfitIntegrationConfig:
    """Configuration for ML profit potential integration."""
    
    # Model types
    enable_direction_model: bool = True
    enable_magnitude_model: bool = True
    enable_confidence_model: bool = True
    enable_regime_models: bool = True
    
    # Model parameters
    model_type: str = "lightgbm"  # "lightgbm", "random_forest", "neural_network", "linear"
    test_size: float = 0.2
    random_state: int = 42
    
    # Profit-aware training
    use_profit_weighted_loss: bool = True
    use_confidence_weighted_loss: bool = True
    use_regime_specific_training: bool = True
    
    # Feature selection
    enable_feature_selection: bool = True
    max_features: Optional[int] = None
    feature_selection_method: str = "mutual_info"
    
    # Evaluation metrics
    enable_profit_metrics: bool = True
    enable_confidence_metrics: bool = True
    enable_regime_metrics: bool = True
    
    # Neural network parameters
    hidden_layers: List[int] = field(default_factory=lambda: [64, 32, 16])
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32

class ProfitAwareLossFunction:
    """Custom loss function that optimizes for actual profit rather than accuracy."""
    
    def __init__(self, transaction_cost: float = 0.0008):
        self.transaction_cost = transaction_cost
    
    def calculate_profit_loss(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            confidence: Optional[np.ndarray] = None) -> float:
        """Calculate loss based on actual profit potential."""
        
        # Convert predictions to profit expectations
        predicted_profits = self._predictions_to_profits(y_pred, confidence)
        actual_profits = y_true
        
        # Calculate profit-weighted loss
        profit_diff = predicted_profits - actual_profits
        
        # Weight by confidence if available
        if confidence is not None:
            weights = confidence
        else:
            weights = np.ones_like(profit_diff)
        
        # Calculate weighted MSE
        weighted_loss = np.mean(weights * (profit_diff ** 2))
        
        return weighted_loss
    
    def _predictions_to_profits(self, predictions: np.ndarray, 
                              confidence: Optional[np.ndarray] = None) -> np.ndarray:
        """Convert model predictions to expected profit values."""
        
        # This is a simplified version - in practice, you'd have a more sophisticated mapping
        # For now, we'll assume predictions are already in profit space
        
        if confidence is not None:
            # Adjust predictions by confidence
            return predictions * confidence
        else:
            return predictions

class MLProfitPotentialIntegration:
    """ML model integration with enhanced profit potential labels."""
    
    def __init__(self, config: Optional[MLProfitIntegrationConfig] = None):
        """Initialize the ML profit potential integration system."""
        self.config = config or MLProfitIntegrationConfig()
        self.logger = get_logger('MLProfitPotentialIntegration')
        
        # Initialize models
        self.direction_model = None
        self.magnitude_model = None
        self.confidence_model = None
        self.regime_models = {}
        
        # Initialize scaler
        self.scaler = StandardScaler()
        
        # Initialize loss function
        self.profit_loss = ProfitAwareLossFunction()
        
        # Model performance tracking
        self.model_performance = {}
        self.feature_importance = {}
        
        self.logger.info("🤖 ML Profit Potential Integration initialized")
        tprint("🤖 ML Profit Potential Integration initialized")
    
    def train_models(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train ML models with enhanced profit potential labels."""
        start_time = time.time()
        
        tprint("🚀 Starting ML Model Training with Profit Potential Labels")
        self.logger.info("🚀 Starting ML Model Training with Profit Potential Labels")
        
        # Validate input data
        required_columns = ['profit_category', 'profit_magnitude_score', 'confidence_score']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Prepare features and targets
        X, targets = self._prepare_training_data(data)
        
        # Split data
        X_train, X_test, y_train, y_test = self._split_data(X, targets)
        
        # Train models
        training_results = {}
        
        if self.config.enable_direction_model:
            tprint("📊 Training direction prediction model...")
            direction_result = self._train_direction_model(X_train, X_test, y_train, y_test)
            training_results['direction_model'] = direction_result
            tprint("✅ Direction model trained")
        
        if self.config.enable_magnitude_model:
            tprint("📊 Training profit magnitude prediction model...")
            magnitude_result = self._train_magnitude_model(X_train, X_test, y_train, y_test)
            training_results['magnitude_model'] = magnitude_result
            tprint("✅ Magnitude model trained")
        
        if self.config.enable_confidence_model:
            tprint("📊 Training confidence prediction model...")
            confidence_result = self._train_confidence_model(X_train, X_test, y_train, y_test)
            training_results['confidence_model'] = confidence_result
            tprint("✅ Confidence model trained")
        
        if self.config.enable_regime_models and 'hmm_regime' in data.columns:
            tprint("📊 Training regime-specific models...")
            regime_result = self._train_regime_models(X_train, X_test, y_train, y_test, data)
            training_results['regime_models'] = regime_result
            tprint("✅ Regime models trained")
        
        # Calculate overall performance
        training_time = time.time() - start_time
        training_results['training_time'] = training_time
        training_results['total_samples'] = len(data)
        training_results['feature_count'] = X.shape[1]
        
        tprint(f"✅ ML model training completed in {training_time:.2f}s")
        self.logger.info(f"✅ ML model training completed in {training_time:.2f}s")
        
        return training_results
    
    def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """Prepare features and targets for training."""
        
        # Identify feature columns (exclude target columns)
        exclude_columns = [
            'profit_category', 'profit_magnitude_score', 'confidence_score', 
            'potential_profit_pct', 'label', 'hmm_regime', 'close', 'open', 'high', 'low'
        ]
        
        feature_columns = [col for col in data.columns if col not in exclude_columns]
        X = data[feature_columns].fillna(0)
        
        # Prepare targets
        targets = {
            'direction': (data['profit_magnitude_score'] > 5).astype(int),  # Binary: high profit vs not
            'magnitude': data['profit_magnitude_score'],
            'confidence': data['confidence_score'],
            'profit_pct': data.get('potential_profit_pct', pd.Series(0, index=data.index))
        }
        
        return X, targets
    
    def _split_data(self, X: pd.DataFrame, targets: Dict[str, pd.Series]) -> Tuple:
        """Split data into train and test sets."""
        
        # Use the first target for splitting (they should all have the same index)
        first_target = list(targets.values())[0]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, first_target, 
            test_size=self.config.test_size, 
            random_state=self.config.random_state,
            stratify=first_target if first_target.dtype == 'int' else None
        )
        
        # Split all targets
        y_train_dict = {}
        y_test_dict = {}
        
        for target_name, target_values in targets.items():
            y_train_dict[target_name] = target_values.loc[y_train.index]
            y_test_dict[target_name] = target_values.loc[y_test.index]
        
        return X_train, X_test, y_train_dict, y_test_dict
    
    def _train_direction_model(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                             y_train: Dict[str, pd.Series], y_test: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Train direction prediction model."""
        
        model = self._create_model(task='classification')
        
        # Train model
        model.fit(X_train, y_train['direction'])
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # Calculate metrics
        accuracy = accuracy_score(y_test['direction'], y_pred)
        
        # Calculate profit-aware metrics
        profit_metrics = self._calculate_profit_metrics(
            y_test['direction'], y_pred, y_test['profit_pct']
        )
        
        # Store model
        self.direction_model = model
        
        return {
            'model': model,
            'accuracy': accuracy,
            'profit_metrics': profit_metrics,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def _train_magnitude_model(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                             y_train: Dict[str, pd.Series], y_test: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Train profit magnitude prediction model."""
        
        model = self._create_model(task='regression')
        
        # Train model
        model.fit(X_train, y_train['magnitude'])
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test['magnitude'], y_pred)
        mae = mean_absolute_error(y_test['magnitude'], y_pred)
        r2 = r2_score(y_test['magnitude'], y_pred)
        
        # Calculate profit-aware metrics
        profit_metrics = self._calculate_profit_metrics(
            y_test['magnitude'], y_pred, y_test['profit_pct']
        )
        
        # Store model
        self.magnitude_model = model
        
        return {
            'model': model,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'profit_metrics': profit_metrics,
            'predictions': y_pred
        }
    
    def _train_confidence_model(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                              y_train: Dict[str, pd.Series], y_test: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Train confidence prediction model."""
        
        model = self._create_model(task='regression')
        
        # Train model
        model.fit(X_train, y_train['confidence'])
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test['confidence'], y_pred)
        mae = mean_absolute_error(y_test['confidence'], y_pred)
        r2 = r2_score(y_test['confidence'], y_pred)
        
        # Calculate confidence calibration metrics
        calibration_metrics = self._calculate_calibration_metrics(
            y_test['confidence'], y_pred
        )
        
        # Store model
        self.confidence_model = model
        
        return {
            'model': model,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'calibration_metrics': calibration_metrics,
            'predictions': y_pred
        }
    
    def _train_regime_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                           y_train: Dict[str, pd.Series], y_test: Dict[str, pd.Series], 
                           data: pd.DataFrame) -> Dict[str, Any]:
        """Train regime-specific models."""
        
        regime_results = {}
        
        # Get regime information
        train_regimes = data.loc[y_train['direction'].index, 'hmm_regime']
        test_regimes = data.loc[y_test['direction'].index, 'hmm_regime']
        
        unique_regimes = train_regimes.dropna().unique()
        
        for regime in unique_regimes:
            # Filter data for this regime
            train_mask = train_regimes == regime
            test_mask = test_regimes == regime
            
            if train_mask.sum() < 10:  # Need minimum samples
                continue
            
            X_train_regime = X_train[train_mask]
            X_test_regime = X_test[test_mask]
            y_train_regime = {k: v[train_mask] for k, v in y_train.items()}
            y_test_regime = {k: v[test_mask] for k, v in y_test.items()}
            
            # Train magnitude model for this regime
            model = self._create_model(task='regression')
            model.fit(X_train_regime, y_train_regime['magnitude'])
            
            # Make predictions
            y_pred = model.predict(X_test_regime)
            
            # Calculate metrics
            mse = mean_squared_error(y_test_regime['magnitude'], y_pred)
            r2 = r2_score(y_test_regime['magnitude'], y_pred)
            
            regime_results[f'regime_{int(regime)}'] = {
                'model': model,
                'mse': mse,
                'r2': r2,
                'train_samples': train_mask.sum(),
                'test_samples': test_mask.sum(),
                'predictions': y_pred
            }
        
        # Store regime models
        self.regime_models = regime_results
        
        return regime_results
    
    def _create_model(self, task: str):
        """Create model based on configuration."""
        
        if self.config.model_type == "lightgbm" and LIGHTGBM_AVAILABLE:
            if task == 'classification':
                return lgb.LGBMClassifier(random_state=self.config.random_state, verbose=-1)
            else:
                return lgb.LGBMRegressor(random_state=self.config.random_state, verbose=-1)
        
        elif self.config.model_type == "random_forest" and SKLEARN_AVAILABLE:
            if task == 'classification':
                return RandomForestClassifier(random_state=self.config.random_state, n_jobs=-1)
            else:
                return RandomForestRegressor(random_state=self.config.random_state, n_jobs=-1)
        
        elif self.config.model_type == "linear" and SKLEARN_AVAILABLE:
            if task == 'classification':
                return LogisticRegression(random_state=self.config.random_state, max_iter=1000)
            else:
                return LinearRegression()
        
        elif self.config.model_type == "neural_network" and TORCH_AVAILABLE:
            # This would require a more complex implementation
            # For now, fall back to linear regression
            if task == 'classification':
                return LogisticRegression(random_state=self.config.random_state, max_iter=1000)
            else:
                return LinearRegression()
        
        else:
            # Default fallback
            if task == 'classification':
                return LogisticRegression(random_state=self.config.random_state, max_iter=1000)
            else:
                return LinearRegression()
    
    def _calculate_profit_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                profit_pct: pd.Series) -> Dict[str, float]:
        """Calculate profit-focused metrics."""
        
        # Convert predictions to profit expectations
        predicted_profits = self._predictions_to_profit_expectations(y_pred)
        actual_profits = profit_pct.values
        
        # Calculate profit-weighted accuracy
        profit_weighted_accuracy = self._calculate_profit_weighted_accuracy(
            y_true, y_pred, actual_profits
        )
        
        # Calculate expected profit
        expected_profit = np.mean(predicted_profits)
        actual_expected_profit = np.mean(actual_profits)
        
        # Calculate profit correlation
        profit_correlation = np.corrcoef(predicted_profits, actual_profits)[0, 1]
        
        # Calculate profit Sharpe ratio
        profit_sharpe = self._calculate_profit_sharpe(predicted_profits, actual_profits)
        
        return {
            'profit_weighted_accuracy': profit_weighted_accuracy,
            'expected_profit': expected_profit,
            'actual_expected_profit': actual_expected_profit,
            'profit_correlation': profit_correlation,
            'profit_sharpe': profit_sharpe
        }
    
    def _calculate_calibration_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate confidence calibration metrics."""
        
        # Brier score
        brier_score = np.mean((y_pred - y_true) ** 2)
        
        # Calibration error (simplified)
        calibration_error = np.mean(np.abs(y_pred - y_true))
        
        # Reliability
        reliability = 1.0 - brier_score
        
        return {
            'brier_score': brier_score,
            'calibration_error': calibration_error,
            'reliability': reliability
        }
    
    def _predictions_to_profit_expectations(self, predictions: np.ndarray) -> np.ndarray:
        """Convert model predictions to profit expectations."""
        # This is a simplified mapping - in practice, you'd have a more sophisticated approach
        return predictions
    
    def _calculate_profit_weighted_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                          profits: np.ndarray) -> float:
        """Calculate accuracy weighted by profit potential."""
        
        # Weight correct predictions by their profit potential
        correct_predictions = (y_true == y_pred)
        weighted_correct = np.sum(correct_predictions * np.abs(profits))
        total_weight = np.sum(np.abs(profits))
        
        if total_weight > 0:
            return weighted_correct / total_weight
        else:
            return 0.0
    
    def _calculate_profit_sharpe(self, predicted_profits: np.ndarray, actual_profits: np.ndarray) -> float:
        """Calculate profit Sharpe ratio."""
        
        profit_diff = predicted_profits - actual_profits
        mean_diff = np.mean(profit_diff)
        std_diff = np.std(profit_diff)
        
        if std_diff > 0:
            return mean_diff / std_diff
        else:
            return 0.0
    
    def predict_profit_potential(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions using trained models."""
        
        if not self._models_trained():
            raise ValueError("Models must be trained before making predictions")
        
        # Prepare features
        exclude_columns = [
            'profit_category', 'profit_magnitude_score', 'confidence_score', 
            'potential_profit_pct', 'label', 'hmm_regime', 'close', 'open', 'high', 'low'
        ]
        
        feature_columns = [col for col in data.columns if col not in exclude_columns]
        X = data[feature_columns].fillna(0)
        
        predictions = {}
        
        # Direction predictions
        if self.direction_model is not None:
            direction_pred = self.direction_model.predict(X)
            direction_proba = self.direction_model.predict_proba(X)[:, 1] if hasattr(self.direction_model, 'predict_proba') else direction_pred
            predictions['direction'] = direction_pred
            predictions['direction_probability'] = direction_proba
        
        # Magnitude predictions
        if self.magnitude_model is not None:
            magnitude_pred = self.magnitude_model.predict(X)
            predictions['magnitude'] = magnitude_pred
        
        # Confidence predictions
        if self.confidence_model is not None:
            confidence_pred = self.confidence_model.predict(X)
            predictions['confidence'] = confidence_pred
        
        # Regime-specific predictions
        if self.regime_models and 'hmm_regime' in data.columns:
            regime_predictions = {}
            for regime, regime_data in self.regime_models.items():
                regime_mask = data['hmm_regime'] == int(regime.split('_')[1])
                if regime_mask.any():
                    regime_pred = regime_data['model'].predict(X[regime_mask])
                    regime_predictions[regime] = regime_pred
            predictions['regime_specific'] = regime_predictions
        
        return predictions
    
    def _models_trained(self) -> bool:
        """Check if models are trained."""
        return (self.direction_model is not None or 
                self.magnitude_model is not None or 
                self.confidence_model is not None or 
                len(self.regime_models) > 0)
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """Get feature importance from trained models."""
        
        importance_results = {}
        
        if self.direction_model is not None and hasattr(self.direction_model, 'feature_importances_'):
            importance_results['direction'] = self.direction_model.feature_importances_
        
        if self.magnitude_model is not None and hasattr(self.magnitude_model, 'feature_importances_'):
            importance_results['magnitude'] = self.magnitude_model.feature_importances_
        
        if self.confidence_model is not None and hasattr(self.confidence_model, 'feature_importances_'):
            importance_results['confidence'] = self.confidence_model.feature_importances_
        
        return importance_results
    
    def evaluate_model_performance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate model performance with profit-focused metrics."""
        
        if not self._models_trained():
            raise ValueError("Models must be trained before evaluation")
        
        # Make predictions
        predictions = self.predict_profit_potential(data)
        
        # Calculate performance metrics
        performance = {}
        
        # Direction performance
        if 'direction' in predictions and 'profit_magnitude_score' in data.columns:
            actual_direction = (data['profit_magnitude_score'] > 5).astype(int)
            accuracy = accuracy_score(actual_direction, predictions['direction'])
            performance['direction_accuracy'] = accuracy
        
        # Magnitude performance
        if 'magnitude' in predictions and 'profit_magnitude_score' in data.columns:
            mse = mean_squared_error(data['profit_magnitude_score'], predictions['magnitude'])
            mae = mean_absolute_error(data['profit_magnitude_score'], predictions['magnitude'])
            r2 = r2_score(data['profit_magnitude_score'], predictions['magnitude'])
            performance['magnitude_mse'] = mse
            performance['magnitude_mae'] = mae
            performance['magnitude_r2'] = r2
        
        # Confidence performance
        if 'confidence' in predictions and 'confidence_score' in data.columns:
            mse = mean_squared_error(data['confidence_score'], predictions['confidence'])
            performance['confidence_mse'] = mse
        
        # Profit-focused metrics
        if 'potential_profit_pct' in data.columns:
            profit_metrics = self._calculate_profit_metrics(
                data['profit_magnitude_score'].values,
                predictions.get('magnitude', np.zeros(len(data))),
                data['potential_profit_pct']
            )
            performance['profit_metrics'] = profit_metrics
        
        return performance

# Convenience functions
def create_ml_profit_integration(
    model_type: str = "lightgbm",
    enable_direction_model: bool = True,
    enable_magnitude_model: bool = True,
    enable_confidence_model: bool = True,
    enable_regime_models: bool = True,
    use_profit_weighted_loss: bool = True,
    use_confidence_weighted_loss: bool = True
) -> MLProfitPotentialIntegration:
    """Create ML profit potential integration with specified parameters."""
    config = MLProfitIntegrationConfig(
        model_type=model_type,
        enable_direction_model=enable_direction_model,
        enable_magnitude_model=enable_magnitude_model,
        enable_confidence_model=enable_confidence_model,
        enable_regime_models=enable_regime_models,
        use_profit_weighted_loss=use_profit_weighted_loss,
        use_confidence_weighted_loss=use_confidence_weighted_loss
    )
    
    return MLProfitPotentialIntegration(config)

def train_ml_models_with_profit_potential(
    data: pd.DataFrame,
    model_type: str = "lightgbm",
    enable_direction_model: bool = True,
    enable_magnitude_model: bool = True,
    enable_confidence_model: bool = True,
    enable_regime_models: bool = True
) -> Dict[str, Any]:
    """Train ML models with enhanced profit potential labels."""
    ml_integration = create_ml_profit_integration(
        model_type=model_type,
        enable_direction_model=enable_direction_model,
        enable_magnitude_model=enable_magnitude_model,
        enable_confidence_model=enable_confidence_model,
        enable_regime_models=enable_regime_models
    )
    
    return ml_integration.train_models(data)

if __name__ == '__main__':
    # Test the ML profit potential integration
    tprint('🧪 Testing ML Profit Potential Integration')
    
    # Create test data with enhanced profit labels
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    data = pd.DataFrame({
        'open': np.random.uniform(100, 110, 1000),
        'high': np.random.uniform(105, 115, 1000),
        'low': np.random.uniform(95, 105, 1000),
        'close': np.random.uniform(100, 110, 1000),
        'volume': np.random.uniform(1000, 10000, 1000),
        'hmm_regime': np.random.choice([0, 1, 2, 3], 1000),
        'profit_category': np.random.choice([
            'extreme_loss', 'large_loss', 'medium_loss', 'small_loss', 'break_even',
            'low_profit', 'medium_profit', 'high_profit', 'extreme_profit'
        ], 1000),
        'profit_magnitude_score': np.random.uniform(0, 10, 1000),
        'confidence_score': np.random.uniform(0, 1, 1000),
        'potential_profit_pct': np.random.uniform(-0.05, 0.05, 1000),
        # Add some feature columns
        'feature_1': np.random.normal(0, 1, 1000),
        'feature_2': np.random.normal(0, 1, 1000),
        'feature_3': np.random.normal(0, 1, 1000),
        'feature_4': np.random.normal(0, 1, 1000),
        'feature_5': np.random.normal(0, 1, 1000)
    }, index=dates)
    
    # Test ML model training
    tprint('\n📊 Testing ML model training with profit potential labels...')
    training_results = train_ml_models_with_profit_potential(data)
    
    tprint(f'✅ ML model training completed')
    tprint(f'   Training time: {training_results["training_time"]:.2f}s')
    tprint(f'   Total samples: {training_results["total_samples"]}')
    tprint(f'   Feature count: {training_results["feature_count"]}')
    
    # Show model performance
    for model_name, model_result in training_results.items():
        if isinstance(model_result, dict) and 'model' in model_result:
            tprint(f'\n📋 {model_name} performance:')
            for metric, value in model_result.items():
                if metric != 'model' and metric != 'predictions':
                    if isinstance(value, dict):
                        tprint(f'   {metric}:')
                        for sub_metric, sub_value in value.items():
                            tprint(f'     {sub_metric}: {sub_value:.4f}')
                    else:
                        tprint(f'   {metric}: {value:.4f}')
    
    tprint('✅ ML Profit Potential Integration test completed!')