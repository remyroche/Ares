"""
ML-Based Entry Timing Labeler for Tactician

This module implements machine learning-based entry timing labeling that:
1. Uses initial rule-based labeling as training data
2. Trains ML models to predict entry quality
3. Generates refined labels based on ML predictions
4. Iteratively improves labeling quality

The approach follows this workflow:
Initial Rule-Based Labels → ML Model Training → Refined Labels → Model Retraining → Final Labels
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, field
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings

try:
    from src.utils.logger import system_logger
    from src.utils.tprint import (
        tprint, tprint_info, tprint_warning, tprint_error, tprint_success,
        tprint_debug, tprint_progress, tprint_timer
    )
    UTILS_AVAILABLE = True
except ImportError as e:
    print(f"❌ CRITICAL: Failed to import utilities: {e}")

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

# Import VectorBT Rolling Optimizer for enhanced performance
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import (
        VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer,
        optimized_rolling_mean, optimized_rolling_std, optimized_rolling_var,
        optimized_rolling_min, optimized_rolling_max, optimized_rolling_sum,
        optimized_rolling_apply, optimized_rolling_corr, optimized_rolling_cov
    )
    VECTORBT_ROLLING_OPTIMIZER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ WARNING: VectorBT Rolling Optimizer not available: {e}")
    VECTORBT_ROLLING_OPTIMIZER_AVAILABLE = False

except ImportError:
    
    cp = None
    UTILS_AVAILABLE = False

@dataclass
class MLEntryTimingConfig:
    """Configuration for ML-based entry timing labeling."""
    # Model configuration
    models: List[str] = field(default_factory=lambda: ['random_forest', 'gradient_boosting', 'ridge'])
    test_size: float = 0.2
    random_state: int = 42
    
    # Feature engineering
    feature_windows: List[int] = field(default_factory=lambda: [3, 5, 10, 20, 50])
    technical_indicators: bool = True
    price_action_features: bool = True
    volume_features: bool = True
    volatility_features: bool = True
    
    # Training configuration
    max_iterations: int = 3
    min_improvement_threshold: float = 0.01
    cross_validation_folds: int = 5
    
    # Quality thresholds
    min_r2_score: float = 0.3
    min_correlation: float = 0.5

class MLEntryTimingLabeler:
    """ML-based entry timing labeler for Tactician."""
    
    def __init__(self, config: MLEntryTimingConfig):
        self.config = config
        self.logger = system_logger.getChild('MLEntryTimingLabeler')
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.training_history = []
        
        # Initialize VectorBT Rolling Optimizer for enhanced performance
        if VECTORBT_ROLLING_OPTIMIZER_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(
                enable_gpu=False,  # Conservative for ML labeling
                enable_parallel=True,
                memory_efficient=True,
                chunk_size=500,  # Smaller chunks for ML operations
                fast_fail=False,  # Use fallbacks for robustness
                enable_logging=True
            )
            tprint_success("✅ VectorBT Rolling Optimizer initialized for ML Entry Timing Labeler")
        else:
            self.vectorbt_optimizer = None
            tprint_warning("⚠️ VectorBT Rolling Optimizer not available for ML Entry Timing Labeler")
        
    def create_ml_based_labels(
        self,
        data: pd.DataFrame,
        initial_labels: pd.Series,
        analyst_signals: pd.Series,
        regime_assignments: Optional[pd.Series] = None
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Create ML-based entry timing labels.
        
        Args:
            data: Market data with OHLCV columns
            initial_labels: Initial rule-based labels
            analyst_signals: Analyst green light signals
            regime_assignments: Optional regime assignments
            
        Returns:
            Tuple of (ml_labels, training_metrics)
        """
        tprint_info("🤖 Creating ML-based entry timing labels...")
        
        # Step 1: Generate features for ML training
        features = self._generate_ml_features(data, analyst_signals, regime_assignments)
        tprint_info(f"📊 Generated {len(features.columns)} features for ML training")
        
        # Step 2: Prepare training data
        X, y, valid_indices = self._prepare_training_data(features, initial_labels)
        tprint_info(f"📊 Training data: {X.shape[0]} samples, {X.shape[1]} features")
        
        if len(X) < 100:
            tprint_warning("⚠️ Insufficient training data for ML labeling")
            return initial_labels, {'error': 'Insufficient training data'}
        
        # Step 3: Train ML models
        training_metrics = self._train_models(X, y)
        tprint_info(f"📊 Model training completed. Best R²: {training_metrics.get('best_r2', 0):.3f}")
        
        # Step 4: Generate ML-based labels
        ml_labels = self._generate_ml_labels(features, valid_indices)
        
        # Step 5: Calculate quality metrics
        quality_metrics = self._calculate_ml_quality_metrics(
            initial_labels, ml_labels, training_metrics
        )
        
        tprint_success(f"✅ ML-based labeling completed")
        tprint_info(f"📊 ML label quality: {quality_metrics.get('overall_quality', 0):.3f}")
        
        return ml_labels, {**training_metrics, **quality_metrics}
    
    def _generate_ml_features(
        self,
        data: pd.DataFrame,
        analyst_signals: pd.Series,
        regime_assignments: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Generate comprehensive features for ML training."""
        features = pd.DataFrame(index=data.index)
        
        # Price action features
        if self.config.price_action_features:
            price_features = self._generate_price_action_features(data)
            features = pd.concat([features, price_features], axis=1)
        
        # Technical indicators
        if self.config.technical_indicators:
            tech_features = self._generate_technical_indicator_features(data)
            features = pd.concat([features, tech_features], axis=1)
        
        # Volume features
        if self.config.volume_features:
            volume_features = self._generate_volume_features(data)
            features = pd.concat([features, volume_features], axis=1)
        
        # Volatility features
        if self.config.volatility_features:
            vol_features = self._generate_volatility_features(data)
            features = pd.concat([features, vol_features], axis=1)
        
        # Analyst signal features
        analyst_features = self._generate_analyst_signal_features(analyst_signals)
        features = pd.concat([features, analyst_features], axis=1)
        
        # Regime features
        if regime_assignments is not None:
            regime_features = self._generate_regime_features(regime_assignments)
            features = pd.concat([features, regime_features], axis=1)
        
        # Time-based features
        time_features = self._generate_time_features(data.index)
        features = pd.concat([features, time_features], axis=1)
        
        return features
    
    def _generate_price_action_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate price action features."""
        features = pd.DataFrame(index=data.index)
        
        # Basic price features
        features['open'] = data['open']
        features['high'] = data['high']
        features['low'] = data['low']
        features['close'] = data['close']
        
        # Price ratios
        features['hl_ratio'] = data['high'] / data['low']
        features['oc_ratio'] = data['open'] / data['close']
        features['body_ratio'] = abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
        
        # Price changes
        for window in self.config.feature_windows:
            features[f'price_change_{window}'] = data['close'].pct_change(window)
            features[f'price_volatility_{window}'] = data['close'].pct_change().rolling(window).std()
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            ma = data['close'].rolling(window).mean()
            features[f'ma_{window}'] = ma
            features[f'price_vs_ma_{window}'] = (data['close'] - ma) / ma
        
        return features
    
    def _generate_technical_indicator_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate technical indicator features."""
        features = pd.DataFrame(index=data.index)
        
        # RSI
        for window in [14, 21]:
            rsi = self._calculate_rsi(data['close'], window)
            features[f'rsi_{window}'] = rsi
        
        # MACD
        macd_line, signal_line, histogram = self._calculate_macd(data['close'])
        features['macd'] = macd_line
        features['macd_signal'] = signal_line
        features['macd_histogram'] = histogram
        
        # Bollinger Bands
        for window in [20, 50]:
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(data['close'], window)
            features[f'bb_upper_{window}'] = bb_upper
            features[f'bb_middle_{window}'] = bb_middle
            features[f'bb_lower_{window}'] = bb_lower
            features[f'bb_width_{window}'] = (bb_upper - bb_lower) / bb_middle
            features[f'bb_position_{window}'] = (data['close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)
        
        return features
    
    def _generate_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volume-based features."""
        features = pd.DataFrame(index=data.index)
        
        # Volume ratios
        for window in self.config.feature_windows:
            avg_volume = data['volume'].rolling(window).mean()
            features[f'volume_ratio_{window}'] = data['volume'] / (avg_volume + 1e-8)
            features[f'volume_change_{window}'] = data['volume'].pct_change(window)
        
        # Volume-price relationship
        features['volume_price_trend'] = (data['volume'] * data['close'].pct_change()).rolling(20).sum()
        
        # VWAP
        vwap = (data['volume'] * data['close']).rolling(20).sum() / data['volume'].rolling(20).sum()
        features['vwap'] = vwap
        features['price_vs_vwap'] = (data['close'] - vwap) / vwap
        
        return features
    
    def _generate_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility-based features."""
        features = pd.DataFrame(index=data.index)
        
        # Rolling volatility
        for window in self.config.feature_windows:
            returns = data['close'].pct_change()
            vol = returns.rolling(window).std()
            features[f'volatility_{window}'] = vol
            
            # Volatility of volatility
            vol_of_vol = vol.rolling(window).std()
            features[f'vol_of_vol_{window}'] = vol_of_vol
        
        # GARCH-like features
        returns = data['close'].pct_change()
        features['returns'] = returns
        features['abs_returns'] = abs(returns)
        features['squared_returns'] = returns ** 2
        
        return features
    
    def _generate_analyst_signal_features(self, analyst_signals: pd.Series) -> pd.DataFrame:
        """Generate analyst signal features."""
        features = pd.DataFrame(index=analyst_signals.index)
        
        features['analyst_signal'] = analyst_signals
        
        # Signal strength over time
        for window in [3, 5, 10]:
            features[f'analyst_signal_strength_{window}'] = analyst_signals.rolling(window).mean()
            features[f'analyst_signal_consistency_{window}'] = analyst_signals.rolling(window).std()
        
        # Time since last signal
        last_signal = analyst_signals[analyst_signals > 0].index
        if len(last_signal) > 0:
            time_since_signal = pd.Series(index=analyst_signals.index, dtype=float)
            for i, idx in enumerate(analyst_signals.index):
                prev_signals = last_signal[last_signal <= idx]
                if len(prev_signals) > 0:
                    time_since_signal.iloc[i] = (idx - prev_signals[-1]).total_seconds() / 3600  # hours
                else:
                    time_since_signal.iloc[i] = np.nan
            features['time_since_analyst_signal'] = time_since_signal
        
        return features
    
    def _generate_regime_features(self, regime_assignments: pd.Series) -> pd.DataFrame:
        """Generate regime-based features."""
        features = pd.DataFrame(index=regime_assignments.index)
        
        # One-hot encoding of regimes
        regime_dummies = pd.get_dummies(regime_assignments, prefix='regime')
        features = pd.concat([features, regime_dummies], axis=1)
        
        # Regime duration
        regime_changes = regime_assignments.diff() != 0
        regime_duration = pd.Series(index=regime_assignments.index, dtype=float)
        current_duration = 0
        for i, changed in enumerate(regime_changes):
            if changed:
                current_duration = 1
            else:
                current_duration += 1
            regime_duration.iloc[i] = current_duration
        features['regime_duration'] = regime_duration
        
        return features
    
    def _generate_time_features(self, index: pd.Index) -> pd.DataFrame:
        """Generate time-based features."""
        features = pd.DataFrame(index=index)
        
        # Time components
        features['hour'] = index.hour
        features['day_of_week'] = index.dayofweek
        features['day_of_month'] = index.day
        features['month'] = index.month
        
        # Cyclical encoding
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        
        return features
    
    def _prepare_training_data(
        self, 
        features: pd.DataFrame, 
        labels: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
        """Prepare training data for ML models."""
        # Align features and labels
        common_index = features.index.intersection(labels.index)
        X = features.loc[common_index]
        y = labels.loc[common_index]
        
        # Remove rows with NaN values
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        valid_indices = X_clean.index
        
        # Handle infinite values
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        X_clean = X_clean.fillna(X_clean.median())
        
        return X_clean.values, y_clean.values, valid_indices
    
    def _train_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train ML models for entry timing prediction."""
        training_metrics = {}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['main'] = scaler
        
        # Train models
        model_configs = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=self.config.random_state),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=self.config.random_state),
            'ridge': Ridge(alpha=1.0)
        }
        
        best_model = None
        best_score = -np.inf
        
        for model_name in self.config.models:
            if model_name not in model_configs:
                continue
                
            tprint_info(f"🤖 Training {model_name}...")
            
            model = model_configs[model_name]
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=self.config.cross_validation_folds)
            
            training_metrics[model_name] = {
                'r2_score': r2,
                'mse': mse,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            # Store model if it's the best
            if r2 > best_score:
                best_score = r2
                best_model = model_name
                self.models['best'] = model
            
            # Store feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[model_name] = model.feature_importances_
            
            tprint_info(f"   R²: {r2:.3f}, CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        training_metrics['best_model'] = best_model
        training_metrics['best_r2'] = best_score
        
        return training_metrics
    
    def _generate_ml_labels(
        self, 
        features: pd.DataFrame, 
        valid_indices: pd.Index
    ) -> pd.Series:
        """Generate ML-based labels using trained models."""
        if 'best' not in self.models:
            tprint_error("❌ No trained model available for label generation")
            return pd.Series(0, index=features.index)
        
        # Prepare features
        X = features.loc[valid_indices]
        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
        
        # Scale features
        X_scaled = self.scalers['main'].transform(X_clean)
        
        # Generate predictions
        predictions = self.models['best'].predict(X_scaled)
        
        # Create labels series
        ml_labels = pd.Series(0, index=features.index, dtype=float)
        ml_labels.loc[valid_indices] = predictions
        
        # Apply quality threshold
        quality_threshold = np.percentile(predictions[predictions > 0], 70) if (predictions > 0).any() else 0.5
        ml_labels = ml_labels.where(ml_labels >= quality_threshold, 0)
        
        return ml_labels
    
    def _calculate_ml_quality_metrics(
        self,
        initial_labels: pd.Series,
        ml_labels: pd.Series,
        training_metrics: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate quality metrics for ML-based labeling."""
        metrics = {}
        
        # Basic metrics
        initial_positive = (initial_labels > 0).sum()
        ml_positive = (ml_labels > 0).sum()
        
        metrics['initial_positive_count'] = initial_positive
        metrics['ml_positive_count'] = ml_positive
        metrics['label_change_ratio'] = ml_positive / initial_positive if initial_positive > 0 else 0
        
        # Correlation with initial labels
        common_index = initial_labels.index.intersection(ml_labels.index)
        if len(common_index) > 0:
            correlation = initial_labels.loc[common_index].corr(ml_labels.loc[common_index])
            metrics['correlation_with_initial'] = correlation if not np.isnan(correlation) else 0
        
        # Model performance
        metrics['best_r2_score'] = training_metrics.get('best_r2', 0)
        metrics['best_model'] = training_metrics.get('best_model', 'unknown')
        
        # Overall quality
        metrics['overall_quality'] = (
            metrics.get('correlation_with_initial', 0) * 0.4 +
            metrics.get('best_r2_score', 0) * 0.6
        )
        
        return metrics
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        rolling_mean = prices.rolling(window).mean()
        rolling_std = prices.rolling(window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, rolling_mean, lower_band
    
    def save_models(self, filepath: str) -> None:
        """Save trained models and scalers."""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_importance': self.feature_importance,
            'training_history': self.training_history
        }
        joblib.dump(model_data, filepath)
        tprint_success(f"✅ Models saved to {filepath}")
    
    def load_models(self, filepath: str) -> None:
        """Load trained models and scalers."""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.feature_importance = model_data['feature_importance']
        self.training_history = model_data['training_history']
        tprint_success(f"✅ Models loaded from {filepath}")

    def _optimized_rolling_operation(self, data: pd.Series, operation: str, 
                                   window: int, **kwargs) -> pd.Series:
        """Perform optimized rolling operation using VectorBT Rolling Optimizer."""
        if self.vectorbt_optimizer is not None:
            try:
                if operation == 'mean':
                    return self.vectorbt_optimizer.rolling_mean(data, window=window, **kwargs)
                elif operation == 'std':
                    return self.vectorbt_optimizer.rolling_std(data, window=window, **kwargs)
                elif operation == 'var':
                    return self.vectorbt_optimizer.rolling_var(data, window=window, **kwargs)
                elif operation == 'min':
                    return self.vectorbt_optimizer.rolling_min(data, window=window, **kwargs)
                elif operation == 'max':
                    return self.vectorbt_optimizer.rolling_max(data, window=window, **kwargs)
                elif operation == 'sum':
                    return self.vectorbt_optimizer.rolling_sum(data, window=window, **kwargs)
                elif operation == 'apply':
                    func = kwargs.get('func')
                    return self.vectorbt_optimizer.rolling_apply(data, func, window=window, **kwargs)
                else:
                    raise ValueError(f"Unsupported operation: {operation}")
            except Exception as e:
                tprint_warning(f"⚠️ VectorBT Rolling Optimizer failed for {operation}: {e}, using fallback")
                return self._fallback_rolling_operation(data, operation, window, **kwargs)
        else:
            return self._fallback_rolling_operation(data, operation, window, **kwargs)
    
    def _fallback_rolling_operation(self, data: pd.Series, operation: str, 
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
        elif operation == 'apply':
            func = kwargs.get('func')
            return data.rolling(window=window).apply(func, **kwargs)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Legacy method - now uses optimized rolling operations."""
        return self._optimized_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Legacy method - now uses fallback rolling operations."""
        return self._fallback_rolling_operation(data, operation, window, **kwargs)
    
    def _vectorbt_apply_operation(self, data: pd.Series, func, 
                                 window: int, **kwargs) -> pd.Series:
        """Legacy method - now uses optimized rolling operations."""
        return self._optimized_rolling_operation(data, 'apply', window, func=func, **kwargs)
