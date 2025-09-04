#!/usr/bin/env python3
"""ML-Based Regime Transition Detection for Step 3.

This module implements machine learning models to detect regime transitions,
replacing hardcoded logic with trained models that learn transition patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import joblib
import warnings
warnings.filterwarnings('ignore')

class MLRegimeTransitionDetector:
    """Machine learning-based regime transition detector."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_importance = {}
        self.model_performance = {}
        
        # Model configuration
        self.model_types = self.config.get('model_types', [
            'random_forest', 'gradient_boosting', 'logistic_regression', 'neural_network'
        ])
        
        self.transition_window = self.config.get('transition_window', 5)
        self.prediction_horizon = self.config.get('prediction_horizon', 3)
        self.min_transition_samples = self.config.get('min_transition_samples', 50)
        
        # Feature engineering parameters
        self.feature_lags = self.config.get('feature_lags', [1, 2, 3, 5, 10])
        self.volatility_windows = self.config.get('volatility_windows', [5, 10, 20])
        self.momentum_windows = self.config.get('momentum_windows', [5, 10, 20])
        
    def train_transition_models(self, data: pd.DataFrame, regimes: np.ndarray) -> Dict[str, Any]:
        """
        Train ML models to detect regime transitions.
        
        Args:
            data: Market data with OHLCV columns
            regimes: Regime labels for each data point
            
        Returns:
            Training results and model performance
        """
        training_results = {
            'models_trained': [],
            'model_performance': {},
            'feature_importance': {},
            'training_summary': {}
        }
        
        # 1. Prepare training data
        X, y = self._prepare_transition_training_data(data, regimes)
        
        if len(X) == 0 or len(np.unique(y)) < 2:
            return {'error': 'Insufficient data for training transition models'}
        
        # 2. Split data for training and validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 3. Train each model type
        for model_type in self.model_types:
            try:
                model_results = self._train_single_model(
                    model_type, X_train, X_test, y_train, y_test
                )
                
                if model_results['success']:
                    training_results['models_trained'].append(model_type)
                    training_results['model_performance'][model_type] = model_results['performance']
                    training_results['feature_importance'][model_type] = model_results['feature_importance']
                    
                    # Store trained model
                    self.models[model_type] = model_results['model']
                    self.scalers[model_type] = model_results['scaler']
                    self.model_performance[model_type] = model_results['performance']
                    self.feature_importance[model_type] = model_results['feature_importance']
            
            except Exception as e:
                print(f"Failed to train {model_type}: {e}")
                continue
        
        # 4. Select best model
        best_model = self._select_best_model(training_results['model_performance'])
        training_results['best_model'] = best_model
        
        # 5. Generate training summary
        training_results['training_summary'] = self._generate_training_summary(training_results)
        
        return training_results
    
    def _prepare_transition_training_data(self, data: pd.DataFrame, regimes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for regime transition detection."""
        # 1. Create transition labels
        transition_labels = self._create_transition_labels(regimes)
        
        # 2. Create features for transition prediction
        features = self._create_transition_features(data, regimes)
        
        # 3. Align features and labels
        min_length = min(len(features), len(transition_labels))
        features = features[:min_length]
        transition_labels = transition_labels[:min_length]
        
        # 4. Remove samples with insufficient history
        max_lag = max(self.feature_lags) if self.feature_lags else 10
        valid_mask = np.arange(len(features)) >= max_lag
        
        features = features[valid_mask]
        transition_labels = transition_labels[valid_mask]
        
        # 5. Handle class imbalance
        transition_labels = self._handle_class_imbalance(transition_labels)
        
        return features, transition_labels
    
    def _create_transition_labels(self, regimes: np.ndarray) -> np.ndarray:
        """Create labels for regime transitions."""
        transition_labels = np.zeros(len(regimes), dtype=int)
        
        # Look ahead to detect transitions
        for i in range(len(regimes) - self.prediction_horizon):
            current_regime = regimes[i]
            future_regimes = regimes[i + 1:i + self.prediction_horizon + 1]
            
            # Check if regime changes in the prediction horizon
            if np.any(future_regimes != current_regime):
                transition_labels[i] = 1  # Transition will occur
        
        return transition_labels
    
    def _create_transition_features(self, data: pd.DataFrame, regimes: np.ndarray) -> np.ndarray:
        """Create features for transition prediction."""
        feature_list = []
        
        # 1. Price-based features
        price_features = self._create_price_transition_features(data)
        feature_list.append(price_features)
        
        # 2. Volume-based features
        volume_features = self._create_volume_transition_features(data)
        feature_list.append(volume_features)
        
        # 3. Volatility-based features
        volatility_features = self._create_volatility_transition_features(data)
        feature_list.append(volatility_features)
        
        # 4. Regime-based features
        regime_features = self._create_regime_transition_features(regimes)
        feature_list.append(regime_features)
        
        # 5. Technical indicator features
        technical_features = self._create_technical_transition_features(data)
        feature_list.append(technical_features)
        
        # 6. Cross-feature interactions
        interaction_features = self._create_interaction_transition_features(data, regimes)
        feature_list.append(interaction_features)
        
        # Combine all features
        all_features = np.concatenate([f for f in feature_list if f is not None], axis=1)
        
        return all_features
    
    def _create_price_transition_features(self, data: pd.DataFrame) -> np.ndarray:
        """Create price-based transition features."""
        features = []
        
        # Price momentum features
        for window in self.momentum_windows:
            momentum = data['close'].pct_change(window)
            features.append(momentum.values.reshape(-1, 1))
            
            # Momentum acceleration
            momentum_acc = momentum.diff()
            features.append(momentum_acc.values.reshape(-1, 1))
        
        # Price position features
        for window in [10, 20, 50]:
            rolling_high = data['high'].rolling(window).max()
            rolling_low = data['low'].rolling(window).min()
            price_position = (data['close'] - rolling_low) / (rolling_high - rolling_low)
            features.append(price_position.values.reshape(-1, 1))
        
        # Price range features
        price_range = (data['high'] - data['low']) / data['close']
        features.append(price_range.values.reshape(-1, 1))
        
        # Price gap features
        price_gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
        features.append(price_gap.values.reshape(-1, 1))
        
        return np.concatenate(features, axis=1) if features else np.array([]).reshape(len(data), 0)
    
    def _create_volume_transition_features(self, data: pd.DataFrame) -> np.ndarray:
        """Create volume-based transition features."""
        features = []
        
        # Volume momentum features
        for window in self.momentum_windows:
            volume_momentum = data['volume'].pct_change(window)
            features.append(volume_momentum.values.reshape(-1, 1))
        
        # Volume ratio features
        for window in [5, 10, 20]:
            volume_ratio = data['volume'] / data['volume'].rolling(window).mean()
            features.append(volume_ratio.values.reshape(-1, 1))
        
        # Volume-price relationship
        volume_price_trend = (data['close'].pct_change() * data['volume']).rolling(10).sum()
        features.append(volume_price_trend.values.reshape(-1, 1))
        
        # Volume volatility
        volume_volatility = data['volume'].rolling(20).std() / data['volume'].rolling(20).mean()
        features.append(volume_volatility.values.reshape(-1, 1))
        
        return np.concatenate(features, axis=1) if features else np.array([]).reshape(len(data), 0)
    
    def _create_volatility_transition_features(self, data: pd.DataFrame) -> np.ndarray:
        """Create volatility-based transition features."""
        features = []
        
        # Multi-timeframe volatility
        for window in self.volatility_windows:
            volatility = data['close'].pct_change().rolling(window).std()
            features.append(volatility.values.reshape(-1, 1))
            
            # Volatility momentum
            vol_momentum = volatility.pct_change()
            features.append(vol_momentum.values.reshape(-1, 1))
        
        # Volatility of volatility
        returns = data['close'].pct_change()
        vol_of_vol = returns.rolling(20).std().rolling(10).std()
        features.append(vol_of_vol.values.reshape(-1, 1))
        
        # Volatility regime features
        vol_regime = self._classify_volatility_regime(data['close'].pct_change().rolling(20).std())
        features.append(vol_regime.values.reshape(-1, 1))
        
        return np.concatenate(features, axis=1) if features else np.array([]).reshape(len(data), 0)
    
    def _create_regime_transition_features(self, regimes: np.ndarray) -> np.ndarray:
        """Create regime-based transition features."""
        features = []
        
        # Regime persistence
        regime_persistence = self._calculate_regime_persistence(regimes)
        features.append(regime_persistence.reshape(-1, 1))
        
        # Regime stability
        regime_stability = self._calculate_regime_stability(regimes)
        features.append(regime_stability.reshape(-1, 1))
        
        # Regime transition probability
        transition_prob = self._calculate_transition_probability(regimes)
        features.append(transition_prob.reshape(-1, 1))
        
        # Regime duration
        regime_duration = self._calculate_regime_duration(regimes)
        features.append(regime_duration.reshape(-1, 1))
        
        # Regime change frequency
        change_frequency = self._calculate_regime_change_frequency(regimes)
        features.append(change_frequency.reshape(-1, 1))
        
        return np.concatenate(features, axis=1) if features else np.array([]).reshape(len(regimes), 0)
    
    def _create_technical_transition_features(self, data: pd.DataFrame) -> np.ndarray:
        """Create technical indicator transition features."""
        features = []
        
        # RSI features
        rsi = self._calculate_rsi(data['close'])
        features.append(rsi.values.reshape(-1, 1))
        features.append(rsi.diff().values.reshape(-1, 1))  # RSI momentum
        
        # MACD features
        macd = self._calculate_macd(data['close'])
        features.append(macd.values.reshape(-1, 1))
        features.append(macd.diff().values.reshape(-1, 1))  # MACD momentum
        
        # Bollinger Bands features
        bb_position, bb_width = self._calculate_bollinger_bands(data['close'])
        features.append(bb_position.values.reshape(-1, 1))
        features.append(bb_width.values.reshape(-1, 1))
        
        # ATR features
        atr = self._calculate_atr(data)
        features.append(atr.values.reshape(-1, 1))
        features.append((atr / data['close']).values.reshape(-1, 1))  # Normalized ATR
        
        return np.concatenate(features, axis=1) if features else np.array([]).reshape(len(data), 0)
    
    def _create_interaction_transition_features(self, data: pd.DataFrame, regimes: np.ndarray) -> np.ndarray:
        """Create interaction features for transition prediction."""
        features = []
        
        # Price-volume interaction
        price_change = data['close'].pct_change()
        volume_change = data['volume'].pct_change()
        price_volume_interaction = price_change * volume_change
        features.append(price_volume_interaction.values.reshape(-1, 1))
        
        # Volatility-volume interaction
        volatility = data['close'].pct_change().rolling(20).std()
        vol_vol_interaction = volatility * data['volume']
        features.append(vol_vol_interaction.values.reshape(-1, 1))
        
        # Regime-momentum interaction
        momentum = data['close'].pct_change(10)
        regime_momentum_interaction = regimes.astype(float) * momentum
        features.append(regime_momentum_interaction.values.reshape(-1, 1))
        
        # Regime-volatility interaction
        regime_vol_interaction = regimes.astype(float) * volatility
        features.append(regime_vol_interaction.values.reshape(-1, 1))
        
        return np.concatenate(features, axis=1) if features else np.array([]).reshape(len(data), 0)
    
    def _handle_class_imbalance(self, labels: np.ndarray) -> np.ndarray:
        """Handle class imbalance in transition labels."""
        # Calculate class weights
        unique_classes = np.unique(labels)
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=labels)
        
        # Store class weights for model training
        self.class_weights = dict(zip(unique_classes, class_weights))
        
        return labels
    
    def _train_single_model(self, model_type: str, X_train: np.ndarray, X_test: np.ndarray,
                          y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Train a single model type."""
        try:
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Initialize model
            model = self._initialize_model(model_type)
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate performance metrics
            performance = self._calculate_model_performance(y_test, y_pred, y_pred_proba)
            
            # Calculate feature importance
            feature_importance = self._calculate_feature_importance(model, X_train_scaled.shape[1])
            
            return {
                'success': True,
                'model': model,
                'scaler': scaler,
                'performance': performance,
                'feature_importance': feature_importance
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'model': None,
                'scaler': None,
                'performance': {},
                'feature_importance': {}
            }
    
    def _initialize_model(self, model_type: str):
        """Initialize a model based on type."""
        if model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        elif model_type == 'logistic_regression':
            return LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            )
        elif model_type == 'neural_network':
            return MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _calculate_model_performance(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate model performance metrics."""
        performance = {}
        
        # Basic metrics
        performance['accuracy'] = np.mean(y_true == y_pred)
        
        # Confusion matrix metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        performance['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        performance['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        performance['f1_score'] = 2 * (performance['precision'] * performance['recall']) / (performance['precision'] + performance['recall']) if (performance['precision'] + performance['recall']) > 0 else 0
        
        # ROC AUC if probabilities available
        if y_pred_proba is not None:
            try:
                performance['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            except:
                performance['roc_auc'] = 0.0
        
        return performance
    
    def _calculate_feature_importance(self, model, n_features: int) -> Dict[str, float]:
        """Calculate feature importance."""
        if hasattr(model, 'feature_importances_'):
            return {f'feature_{i}': importance for i, importance in enumerate(model.feature_importances_)}
        elif hasattr(model, 'coef_'):
            # For linear models, use absolute coefficients
            coef_abs = np.abs(model.coef_[0])
            return {f'feature_{i}': coef for i, coef in enumerate(coef_abs)}
        else:
            return {f'feature_{i}': 0.0 for i in range(n_features)}
    
    def _select_best_model(self, model_performance: Dict[str, Dict[str, float]]) -> str:
        """Select the best model based on performance."""
        if not model_performance:
            return None
        
        # Score models based on F1 score and ROC AUC
        model_scores = {}
        for model_type, performance in model_performance.items():
            f1_score = performance.get('f1_score', 0)
            roc_auc = performance.get('roc_auc', 0)
            # Combined score
            model_scores[model_type] = 0.6 * f1_score + 0.4 * roc_auc
        
        return max(model_scores, key=model_scores.get)
    
    def _generate_training_summary(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate training summary."""
        summary = {
            'models_trained': len(training_results['models_trained']),
            'best_model': training_results['best_model'],
            'best_performance': training_results['model_performance'].get(training_results['best_model'], {}),
            'feature_importance_top_10': {}
        }
        
        # Get top 10 features from best model
        if training_results['best_model']:
            best_model_importance = training_results['feature_importance'].get(training_results['best_model'], {})
            sorted_features = sorted(best_model_importance.items(), key=lambda x: x[1], reverse=True)
            summary['feature_importance_top_10'] = dict(sorted_features[:10])
        
        return summary
    
    def predict_transitions(self, data: pd.DataFrame, regimes: np.ndarray, 
                          model_type: Optional[str] = None) -> Dict[str, Any]:
        """Predict regime transitions using trained models."""
        if not self.models:
            return {'error': 'No trained models available'}
        
        if model_type is None:
            model_type = self.config.get('best_model', list(self.models.keys())[0])
        
        if model_type not in self.models:
            return {'error': f'Model {model_type} not found'}
        
        # Prepare features
        features = self._create_transition_features(data, regimes)
        
        # Scale features
        scaler = self.scalers[model_type]
        features_scaled = scaler.transform(features)
        
        # Make predictions
        model = self.models[model_type]
        transition_predictions = model.predict(features_scaled)
        transition_probabilities = model.predict_proba(features_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
        
        return {
            'transition_predictions': transition_predictions,
            'transition_probabilities': transition_probabilities,
            'model_used': model_type,
            'confidence_scores': transition_probabilities if transition_probabilities is not None else transition_predictions
        }
    
    def save_models(self, filepath: str) -> bool:
        """Save trained models to file."""
        try:
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'model_performance': self.model_performance,
                'feature_importance': self.feature_importance,
                'config': self.config
            }
            joblib.dump(model_data, filepath)
            return True
        except Exception as e:
            print(f"Failed to save models: {e}")
            return False
    
    def load_models(self, filepath: str) -> bool:
        """Load trained models from file."""
        try:
            model_data = joblib.load(filepath)
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.model_performance = model_data['model_performance']
            self.feature_importance = model_data['feature_importance']
            self.config.update(model_data['config'])
            return True
        except Exception as e:
            print(f"Failed to load models: {e}")
            return False
    
    # Helper methods for feature calculation
    
    def _classify_volatility_regime(self, volatility: pd.Series) -> pd.Series:
        """Classify volatility regime."""
        low_threshold = volatility.rolling(100).quantile(0.33)
        high_threshold = volatility.rolling(100).quantile(0.67)
        
        regime = pd.Series(1, index=volatility.index)
        regime[volatility > high_threshold] = 3
        regime[(volatility > low_threshold) & (volatility <= high_threshold)] = 2
        
        return regime.fillna(1)
    
    def _calculate_regime_persistence(self, regimes: np.ndarray) -> np.ndarray:
        """Calculate regime persistence."""
        persistence = np.zeros(len(regimes))
        current_regime = regimes[0]
        current_count = 0
        
        for i in range(len(regimes)):
            if regimes[i] == current_regime:
                current_count += 1
            else:
                current_count = 1
                current_regime = regimes[i]
            persistence[i] = current_count
        
        return persistence
    
    def _calculate_regime_stability(self, regimes: np.ndarray) -> np.ndarray:
        """Calculate regime stability."""
        stability = np.zeros(len(regimes))
        window = 20
        
        for i in range(len(regimes)):
            start_idx = max(0, i - window + 1)
            recent_regimes = regimes[start_idx:i+1]
            stability[i] = 1 / (1 + np.std(recent_regimes))
        
        return stability
    
    def _calculate_transition_probability(self, regimes: np.ndarray) -> np.ndarray:
        """Calculate transition probability."""
        unique_regimes = np.unique(regimes)
        n_regimes = len(unique_regimes)
        
        if n_regimes < 2:
            return np.zeros(len(regimes))
        
        # Calculate transition matrix
        transition_matrix = np.zeros((n_regimes, n_regimes))
        regime_map = {regime: i for i, regime in enumerate(unique_regimes)}
        
        for i in range(len(regimes) - 1):
            current_idx = regime_map[regimes[i]]
            next_idx = regime_map[regimes[i + 1]]
            transition_matrix[current_idx, next_idx] += 1
        
        # Normalize to probabilities
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums > 0)
        
        # Calculate transition probabilities
        transition_probs = np.zeros(len(regimes))
        for i in range(len(regimes)):
            current_idx = regime_map[regimes[i]]
            other_probs = transition_matrix[current_idx, :]
            other_probs[current_idx] = 0  # Exclude staying in same regime
            transition_probs[i] = np.sum(other_probs)
        
        return transition_probs
    
    def _calculate_regime_duration(self, regimes: np.ndarray) -> np.ndarray:
        """Calculate regime duration."""
        duration = np.zeros(len(regimes))
        current_regime = regimes[0]
        current_duration = 0
        
        for i in range(len(regimes)):
            if regimes[i] == current_regime:
                current_duration += 1
            else:
                current_duration = 1
                current_regime = regimes[i]
            duration[i] = current_duration
        
        return duration
    
    def _calculate_regime_change_frequency(self, regimes: np.ndarray) -> np.ndarray:
        """Calculate regime change frequency."""
        change_freq = np.zeros(len(regimes))
        window = 50
        
        for i in range(len(regimes)):
            start_idx = max(0, i - window + 1)
            recent_regimes = regimes[start_idx:i+1]
            changes = np.sum(np.diff(recent_regimes) != 0)
            change_freq[i] = changes / len(recent_regimes)
        
        return change_freq
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - 100 / (1 + rs)
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate MACD."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        bb_upper = sma + std * num_std
        bb_lower = sma - std * num_std
        bb_position = (prices - bb_lower) / (bb_upper - bb_lower)
        bb_width = (bb_upper - bb_lower) / sma
        return bb_position, bb_width
    
    def _calculate_atr(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate ATR."""
        high = data['high']
        low = data['low']
        close = data['close']
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        return atr


# Example usage and testing
if __name__ == "__main__":
    # Create sample data with regime transitions
    np.random.seed(42)
    n_samples = 2000
    
    # Create regime sequence with transitions
    regimes = np.zeros(n_samples, dtype=int)
    regimes[500:1000] = 1
    regimes[1000:1500] = 2
    regimes[1500:] = 1
    
    # Create market data
    data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(n_samples) * 0.01),
        'high': 100 + np.cumsum(np.random.randn(n_samples) * 0.01) + np.abs(np.random.randn(n_samples) * 0.5),
        'low': 100 + np.cumsum(np.random.randn(n_samples) * 0.01) - np.abs(np.random.randn(n_samples) * 0.5),
        'close': 100 + np.cumsum(np.random.randn(n_samples) * 0.01),
        'volume': np.random.lognormal(10, 1, n_samples)
    })
    
    # Initialize ML transition detector
    config = {
        'model_types': ['random_forest', 'gradient_boosting', 'logistic_regression'],
        'transition_window': 5,
        'prediction_horizon': 3
    }
    
    detector = MLRegimeTransitionDetector(config)
    
    # Train models
    training_results = detector.train_transition_models(data, regimes)
    
    print("ML Transition Detection Training Results:")
    print(f"Models trained: {training_results['models_trained']}")
    print(f"Best model: {training_results['best_model']}")
    print(f"Best performance: {training_results['best_performance']}")
    
    # Test predictions
    predictions = detector.predict_transitions(data, regimes)
    
    print(f"\nPrediction Results:")
    print(f"Model used: {predictions['model_used']}")
    print(f"Transition predictions: {np.sum(predictions['transition_predictions'])} transitions predicted")
    print(f"Mean confidence: {np.mean(predictions['confidence_scores']):.4f}")
    
    # Save models
    detector.save_models('transition_models.joblib')
    print("Models saved successfully")