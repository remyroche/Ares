#!/usr/bin/env python3
"""Regime Persistence & Forecasting.

This module implements regime persistence modeling and forecasting capabilities,
integrating with existing analyst forecasting logic for regime transition prediction.
"""

import ast
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import asyncio
from datetime import datetime, timedelta
from scipy.stats import expon, gamma, weibull_min
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

class RegimePersistenceForecaster:
    """Regime persistence modeling and forecasting system."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Persistence modeling parameters
        self.min_regime_duration = self.config.get('min_regime_duration', 5)
        self.max_regime_duration = self.config.get('max_regime_duration', 1000)
        self.persistence_models = {}
        
        # Forecasting parameters (max 1h)
        self.forecast_horizon = self.config.get('forecast_horizon', 1)  # hours (max 1h)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.prediction_window = self.config.get('prediction_window', 10)
        
        # Model storage
        self.transition_models = {}
        self.duration_models = {}
        self.forecast_models = {}
        
        # Integration with existing analyst logic
        self.use_analyst_integration = self.config.get('use_analyst_integration', True)
        
    def build_persistence_models(self, data: pd.DataFrame, regimes: np.ndarray) -> Dict[str, Any]:
        """Build persistence models for regime forecasting."""
        print("🔮 Building regime persistence and forecasting models...")
        
        # Step 1: Analyze regime durations
        print("  📊 Analyzing regime durations...")
        duration_analysis = self._analyze_regime_durations(regimes)
        
        # Step 2: Build persistence models
        print("  🏗️ Building persistence models...")
        persistence_models = self._build_persistence_models(duration_analysis)
        
        # Step 3: Build transition models
        print("  🔄 Building transition models...")
        transition_models = self._build_transition_models(regimes)
        
        # Step 4: Build forecasting models
        print("  🎯 Building forecasting models...")
        forecast_models = self._build_forecasting_models(data, regimes)
        
        # Step 5: Integrate with existing analyst logic
        if self.use_analyst_integration:
            print("  🔗 Integrating with existing analyst forecasting...")
            analyst_integration = self._integrate_analyst_forecasting(data, regimes)
        else:
            analyst_integration = {}
        
        return {
            'duration_analysis': duration_analysis,
            'persistence_models': persistence_models,
            'transition_models': transition_models,
            'forecast_models': forecast_models,
            'analyst_integration': analyst_integration,
            'model_quality': self._assess_model_quality(persistence_models, transition_models, forecast_models)
        }
    
    def _analyze_regime_durations(self, regimes: np.ndarray) -> Dict[str, Any]:
        """Analyze regime durations and persistence patterns."""
        if len(regimes) == 0:
            return {'regime_durations': [], 'duration_stats': {}}
        
        # Calculate regime durations
        regime_durations = []
        current_regime = regimes[0]
        current_duration = 1
        
        for i in range(1, len(regimes)):
            if regimes[i] == current_regime:
                current_duration += 1
            else:
                regime_durations.append({
                    'regime': current_regime,
                    'duration': current_duration,
                    'start_index': i - current_duration,
                    'end_index': i - 1
                })
                current_regime = regimes[i]
                current_duration = 1
        
        # Add the last regime
        regime_durations.append({
            'regime': current_regime,
            'duration': current_duration,
            'start_index': len(regimes) - current_duration,
            'end_index': len(regimes) - 1
        })
        
        # Calculate duration statistics
        durations_by_regime = {}
        for duration_info in regime_durations:
            regime = duration_info['regime']
            duration = duration_info['duration']
            
            if regime not in durations_by_regime:
                durations_by_regime[regime] = []
            durations_by_regime[regime].append(duration)
        
        # Calculate statistics for each regime
        duration_stats = {}
        for regime, durations in durations_by_regime.items():
            if len(durations) > 0:
                duration_stats[regime] = {
                    'mean_duration': np.mean(durations),
                    'median_duration': np.median(durations),
                    'std_duration': np.std(durations),
                    'min_duration': np.min(durations),
                    'max_duration': np.max(durations),
                    'count': len(durations),
                    'durations': durations
                }
        
        return {
            'regime_durations': regime_durations,
            'duration_stats': duration_stats,
            'total_regimes': len(np.unique(regimes)),
            'total_durations': len(regime_durations)
        }
    
    def _build_persistence_models(self, duration_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Build persistence models for each regime."""
        persistence_models = {}
        duration_stats = duration_analysis['duration_stats']
        
        for regime, stats in duration_stats.items():
            durations = stats['durations']
            
            if len(durations) < 5:  # Need minimum samples
                continue
            
            # Fit different distribution models
            models = {}
            
            try:
                # Exponential distribution
                exp_params = expon.fit(durations)
                models['exponential'] = {
                    'type': 'exponential',
                    'params': exp_params,
                    'aic': self._calculate_aic(durations, expon, exp_params)
                }
            except:
                pass
            
            try:
                # Gamma distribution
                gamma_params = gamma.fit(durations)
                models['gamma'] = {
                    'type': 'gamma',
                    'params': gamma_params,
                    'aic': self._calculate_aic(durations, gamma, gamma_params)
                }
            except:
                pass
            
            try:
                # Weibull distribution
                weibull_params = weibull_min.fit(durations)
                models['weibull'] = {
                    'type': 'weibull',
                    'params': weibull_params,
                    'aic': self._calculate_aic(durations, weibull_min, weibull_params)
                }
            except:
                pass
            
            # Select best model based on AIC
            if models:
                best_model = min(models.values(), key=lambda x: x['aic'])
                persistence_models[regime] = best_model
                
                # Create survival function
                persistence_models[regime]['survival_function'] = self._create_survival_function(
                    best_model['type'], best_model['params']
                )
        
        return persistence_models
    
    def _build_transition_models(self, regimes: np.ndarray) -> Dict[str, Any]:
        """Build transition probability models."""
        if len(regimes) < 2:
            return {}
        
        # Calculate transition matrix
        unique_regimes = np.unique(regimes)
        n_regimes = len(unique_regimes)
        transition_matrix = np.zeros((n_regimes, n_regimes))
        
        # Count transitions
        for i in range(len(regimes) - 1):
            from_regime = regimes[i]
            to_regime = regimes[i + 1]
            
            from_idx = np.where(unique_regimes == from_regime)[0][0]
            to_idx = np.where(unique_regimes == to_regime)[0][0]
            
            transition_matrix[from_idx, to_idx] += 1
        
        # Normalize to probabilities
        row_sums = transition_matrix.sum(axis=1)
        for i in range(n_regimes):
            if row_sums[i] > 0:
                transition_matrix[i, :] /= row_sums[i]
        
        # Build transition models
        transition_models = {
            'transition_matrix': transition_matrix,
            'regime_mapping': {i: regime for i, regime in enumerate(unique_regimes)},
            'reverse_mapping': {regime: i for i, regime in enumerate(unique_regimes)}
        }
        
        # Calculate transition statistics
        transition_stats = {}
        for i, from_regime in enumerate(unique_regimes):
            transition_stats[from_regime] = {
                'most_likely_transition': unique_regimes[np.argmax(transition_matrix[i, :])],
                'transition_probability': np.max(transition_matrix[i, :]),
                'transition_entropy': -np.sum(transition_matrix[i, :] * np.log(transition_matrix[i, :] + 1e-10))
            }
        
        transition_models['transition_stats'] = transition_stats
        
        return transition_models
    
    def _build_forecasting_models(self, data: pd.DataFrame, regimes: np.ndarray) -> Dict[str, Any]:
        """Build forecasting models for regime transitions."""
        if len(regimes) < 100:  # Need sufficient data
            return {}
        
        # Prepare features for forecasting
        features = self._prepare_forecasting_features(data, regimes)
        
        if features is None or len(features) == 0:
            return {}
        
        # Build models for different forecasting horizons (max 1h)
        forecast_models = {}
        
        # Convert timeframes to periods based on data frequency
        # Assuming 1m data: 1h = 60 periods, 30m = 2 periods, 15m = 4 periods, 5m = 12 periods
        data_frequency_minutes = self._estimate_data_frequency(data)
        
        if data_frequency_minutes <= 1:  # 1m or less data
            horizons = [5, 15, 30, 60]  # 5m, 15m, 30m, 1h
        elif data_frequency_minutes <= 5:  # 5m data
            horizons = [3, 6, 12]  # 15m, 30m, 1h
        elif data_frequency_minutes <= 15:  # 15m data
            horizons = [2, 4]  # 30m, 1h
        elif data_frequency_minutes <= 30:  # 30m data
            horizons = [2]  # 1h
        else:  # 1h or higher data
            horizons = [1]  # 1h only
        
        for horizon in horizons:
            try:
                model = self._train_forecast_model(features, regimes, horizon)
                if model is not None:
                    forecast_models[f'horizon_{horizon}h'] = model
            except Exception as e:
                print(f"Error building forecast model for horizon {horizon}h: {e}")
        
        return forecast_models
    
    def _prepare_forecasting_features(self, data: pd.DataFrame, regimes: np.ndarray) -> Optional[pd.DataFrame]:
        """Prepare features for regime forecasting."""
        try:
            features = []
            
            # Price-based features
            if 'close' in data.columns:
                returns = data['close'].pct_change()
                features.extend([
                    returns.rolling(5).mean(),
                    returns.rolling(10).mean(),
                    returns.rolling(20).mean(),
                    returns.rolling(5).std(),
                    returns.rolling(10).std(),
                    returns.rolling(20).std()
                ])
            
            # Volume-based features
            if 'volume' in data.columns:
                volume = data['volume']
                features.extend([
                    volume.rolling(5).mean(),
                    volume.rolling(10).mean(),
                    volume.rolling(20).mean(),
                    (volume / volume.rolling(20).mean()).rolling(5).mean()
                ])
            
            # Volatility features
            if 'high' in data.columns and 'low' in data.columns:
                volatility = (data['high'] - data['low']) / data['close']
                features.extend([
                    volatility.rolling(5).mean(),
                    volatility.rolling(10).mean(),
                    volatility.rolling(20).mean()
                ])
            
            # Regime-based features
            regime_features = self._create_regime_features(regimes)
            features.extend(regime_features)
            
            # Combine features
            if features:
                feature_df = pd.concat(features, axis=1)
                feature_df.columns = [f'feature_{i}' for i in range(len(features))]
                return feature_df.dropna()
            
            return None
            
        except Exception as e:
            print(f"Error preparing forecasting features: {e}")
            return None
    
    def _create_regime_features(self, regimes: np.ndarray) -> List[pd.Series]:
        """Create regime-based features for forecasting."""
        features = []
        
        # Current regime
        features.append(pd.Series(regimes, name='current_regime'))
        
        # Regime duration
        regime_duration = self._calculate_regime_duration_series(regimes)
        features.append(regime_duration)
        
        # Regime change indicators
        regime_changes = np.diff(regimes, prepend=regimes[0]) != 0
        features.append(pd.Series(regime_changes.astype(int), name='regime_change'))
        
        # Regime stability (rolling window)
        stability = self._calculate_regime_stability_series(regimes)
        features.append(stability)
        
        return features
    
    def _calculate_regime_duration_series(self, regimes: np.ndarray) -> pd.Series:
        """Calculate regime duration for each time point."""
        durations = np.zeros(len(regimes))
        current_duration = 1
        
        for i in range(1, len(regimes)):
            if regimes[i] == regimes[i-1]:
                current_duration += 1
            else:
                current_duration = 1
            durations[i] = current_duration
        
        return pd.Series(durations, name='regime_duration')
    
    def _calculate_regime_stability_series(self, regimes: np.ndarray, window: int = 20) -> pd.Series:
        """Calculate regime stability over rolling window."""
        stability = np.zeros(len(regimes))
        
        for i in range(window, len(regimes)):
            window_regimes = regimes[i-window:i]
            # Stability as inverse of regime changes in window
            changes = np.sum(np.diff(window_regimes) != 0)
            stability[i] = 1.0 - (changes / (window - 1))
        
        return pd.Series(stability, name='regime_stability')
    
    def _train_forecast_model(self, features: pd.DataFrame, regimes: np.ndarray, horizon: int) -> Optional[Dict[str, Any]]:
        """Train forecasting model for specific horizon."""
        try:
            # Create target variable (regime changes in future)
            target = np.zeros(len(regimes))
            for i in range(len(regimes) - horizon):
                if regimes[i] != regimes[i + horizon]:
                    target[i] = 1
            
            # Align features and target
            min_length = min(len(features), len(target))
            X = features.iloc[:min_length].values
            y = target[:min_length]
            
            if len(X) < 50:  # Need minimum samples
                return None
            
            # Train Random Forest model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X, y)
            
            # Calculate feature importance
            feature_importance = dict(zip(features.columns, model.feature_importances_))
            
            return {
                'model': model,
                'horizon': horizon,
                'feature_importance': feature_importance,
                'training_samples': len(X),
                'model_type': 'random_forest'
            }
            
        except Exception as e:
            print(f"Error training forecast model for horizon {horizon}: {e}")
            return None
    
    def _integrate_analyst_forecasting(self, data: pd.DataFrame, regimes: np.ndarray) -> Dict[str, Any]:
        """Integrate with existing analyst forecasting logic."""
        try:
            # Import analyst forecasting components
            from src.analyst.enhanced_regime_predictor import EnhancedRegimePredictor
            
            # Initialize analyst predictor
            analyst_config = {
                'stability_threshold': 0.1,
                'min_persistence': 3,
                'entropy_percentile': 75,
                'confidence_threshold': 0.7
            }
            
            analyst_predictor = EnhancedRegimePredictor(analyst_config)
            
            # Prepare data for analyst predictor
            features = self._prepare_analyst_features(data)
            hmm_probs = self._create_hmm_probabilities(regimes)
            
            # Get analyst predictions
            analyst_predictions = analyst_predictor.predict_regime_changes(
                features, hmm_probs, regimes
            )
            
            return {
                'analyst_predictions': analyst_predictions,
                'integration_success': True,
                'prediction_count': len(analyst_predictions.get('predictions', [])),
                'confidence_scores': analyst_predictions.get('confidence_scores', [])
            }
            
        except Exception as e:
            print(f"Error integrating with analyst forecasting: {e}")
            return {
                'integration_success': False,
                'error': str(e)
            }
    
    def _prepare_analyst_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features in format expected by analyst predictor."""
        features = pd.DataFrame(index=data.index)
        
        # Basic price features
        if 'close' in data.columns:
            features['returns'] = data['close'].pct_change()
            features['volatility'] = features['returns'].rolling(20).std()
            features['momentum'] = features['returns'].rolling(10).mean()
        
        # Volume features
        if 'volume' in data.columns:
            features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        
        return features.fillna(0)
    
    def _create_hmm_probabilities(self, regimes: np.ndarray) -> np.ndarray:
        """Create HMM probabilities from regime sequence."""
        # Create simple probability matrix
        n_regimes = len(np.unique(regimes))
        n_samples = len(regimes)
        
        probs = np.zeros((n_samples, n_regimes))
        
        for i, regime in enumerate(regimes):
            regime_idx = int(regime) % n_regimes
            probs[i, regime_idx] = 1.0
        
        return probs
    
    def forecast_regime_transitions(self, current_data: pd.DataFrame, current_regime: int, 
                                  models: Dict[str, Any]) -> Dict[str, Any]:
        """Forecast regime transitions using built models."""
        print("🔮 Forecasting regime transitions...")
        
        forecasts = {}
        
        # Persistence-based forecasting
        if 'persistence_models' in models:
            persistence_forecast = self._forecast_persistence(current_regime, models['persistence_models'])
            forecasts['persistence'] = persistence_forecast
        
        # Transition-based forecasting
        if 'transition_models' in models:
            transition_forecast = self._forecast_transitions(current_regime, models['transition_models'])
            forecasts['transitions'] = transition_forecast
        
        # ML-based forecasting
        if 'forecast_models' in models:
            ml_forecast = self._forecast_ml(current_data, current_regime, models['forecast_models'])
            forecasts['ml'] = ml_forecast
        
        # Combine forecasts
        combined_forecast = self._combine_forecasts(forecasts)
        
        return {
            'individual_forecasts': forecasts,
            'combined_forecast': combined_forecast,
            'forecast_confidence': self._calculate_forecast_confidence(forecasts),
            'forecast_horizon': self.forecast_horizon
        }
    
    def _forecast_persistence(self, current_regime: int, persistence_models: Dict[str, Any]) -> Dict[str, Any]:
        """Forecast using persistence models."""
        if current_regime not in persistence_models:
            return {'forecast': 'unknown', 'confidence': 0.0}
        
        model = persistence_models[current_regime]
        survival_func = model.get('survival_function')
        
        if survival_func is None:
            return {'forecast': 'unknown', 'confidence': 0.0}
        
        # Calculate survival probability for forecast horizon
        survival_prob = survival_func(self.forecast_horizon)
        regime_change_prob = 1 - survival_prob
        
        return {
            'forecast': 'regime_change' if regime_change_prob > 0.5 else 'regime_persistence',
            'regime_change_probability': regime_change_prob,
            'survival_probability': survival_prob,
            'confidence': abs(regime_change_prob - 0.5) * 2,  # Distance from 0.5
            'model_type': model['type']
        }
    
    def _forecast_transitions(self, current_regime: int, transition_models: Dict[str, Any]) -> Dict[str, Any]:
        """Forecast using transition models."""
        if 'transition_matrix' not in transition_models:
            return {'forecast': 'unknown', 'confidence': 0.0}
        
        transition_matrix = transition_models['transition_matrix']
        regime_mapping = transition_models['regime_mapping']
        reverse_mapping = transition_models['reverse_mapping']
        
        if current_regime not in reverse_mapping:
            return {'forecast': 'unknown', 'confidence': 0.0}
        
        current_idx = reverse_mapping[current_regime]
        transition_probs = transition_matrix[current_idx, :]
        
        # Find most likely next regime
        most_likely_idx = np.argmax(transition_probs)
        most_likely_regime = regime_mapping[most_likely_idx]
        transition_prob = transition_probs[most_likely_idx]
        
        return {
            'forecast': 'regime_change' if most_likely_regime != current_regime else 'regime_persistence',
            'next_regime': most_likely_regime,
            'transition_probability': transition_prob,
            'confidence': transition_prob,
            'all_transitions': {regime_mapping[i]: prob for i, prob in enumerate(transition_probs)}
        }
    
    def _forecast_ml(self, current_data: pd.DataFrame, current_regime: int, 
                    forecast_models: Dict[str, Any]) -> Dict[str, Any]:
        """Forecast using ML models."""
        if not forecast_models:
            return {'forecast': 'unknown', 'confidence': 0.0}
        
        # Prepare current features
        features = self._prepare_forecasting_features(current_data, np.array([current_regime]))
        
        if features is None or len(features) == 0:
            return {'forecast': 'unknown', 'confidence': 0.0}
        
        # Get latest features
        latest_features = features.iloc[-1].values.reshape(1, -1)
        
        # Make predictions with different horizons
        predictions = {}
        for horizon_name, model_info in forecast_models.items():
            try:
                model = model_info['model']
                prediction = model.predict(latest_features)[0]
                predictions[horizon_name] = {
                    'regime_change_probability': prediction,
                    'confidence': abs(prediction - 0.5) * 2
                }
            except Exception as e:
                print(f"Error in ML forecast for {horizon_name}: {e}")
                predictions[horizon_name] = {'regime_change_probability': 0.5, 'confidence': 0.0}
        
        # Average predictions
        avg_change_prob = np.mean([p['regime_change_probability'] for p in predictions.values()])
        avg_confidence = np.mean([p['confidence'] for p in predictions.values()])
        
        return {
            'forecast': 'regime_change' if avg_change_prob > 0.5 else 'regime_persistence',
            'regime_change_probability': avg_change_prob,
            'confidence': avg_confidence,
            'individual_predictions': predictions
        }
    
    def _combine_forecasts(self, forecasts: Dict[str, Any]) -> Dict[str, Any]:
        """Combine different forecasting approaches."""
        if not forecasts:
            return {'forecast': 'unknown', 'confidence': 0.0}
        
        # Weight different approaches
        weights = {
            'persistence': 0.3,
            'transitions': 0.3,
            'ml': 0.4
        }
        
        combined_change_prob = 0.0
        combined_confidence = 0.0
        total_weight = 0.0
        
        for approach, forecast in forecasts.items():
            if approach in weights and 'regime_change_probability' in forecast:
                weight = weights[approach]
                combined_change_prob += weight * forecast['regime_change_probability']
                combined_confidence += weight * forecast.get('confidence', 0.0)
                total_weight += weight
        
        if total_weight > 0:
            combined_change_prob /= total_weight
            combined_confidence /= total_weight
        
        return {
            'forecast': 'regime_change' if combined_change_prob > 0.5 else 'regime_persistence',
            'regime_change_probability': combined_change_prob,
            'confidence': combined_confidence,
            'approach_weights': weights
        }
    
    def _calculate_forecast_confidence(self, forecasts: Dict[str, Any]) -> float:
        """Calculate overall forecast confidence."""
        if not forecasts:
            return 0.0
        
        confidences = []
        for forecast in forecasts.values():
            if 'confidence' in forecast:
                confidences.append(forecast['confidence'])
        
        if confidences:
            return np.mean(confidences)
        return 0.0
    
    def _assess_model_quality(self, persistence_models: Dict, transition_models: Dict, 
                            forecast_models: Dict) -> Dict[str, Any]:
        """Assess quality of built models."""
        quality_scores = {}
        
        # Persistence model quality
        if persistence_models:
            quality_scores['persistence'] = len(persistence_models) / 5.0  # Normalize by expected regimes
        else:
            quality_scores['persistence'] = 0.0
        
        # Transition model quality
        if transition_models and 'transition_matrix' in transition_models:
            transition_matrix = transition_models['transition_matrix']
            # Quality based on matrix sparsity and balance
            sparsity = np.sum(transition_matrix > 0) / transition_matrix.size
            balance = 1.0 - np.std(transition_matrix.flatten())
            quality_scores['transitions'] = (sparsity + balance) / 2.0
        else:
            quality_scores['transitions'] = 0.0
        
        # Forecast model quality
        if forecast_models:
            quality_scores['forecasting'] = len(forecast_models) / 5.0  # Normalize by expected horizons
        else:
            quality_scores['forecasting'] = 0.0
        
        # Overall quality
        overall_quality = np.mean(list(quality_scores.values()))
        
        return {
            'individual_scores': quality_scores,
            'overall_quality': overall_quality,
            'model_count': {
                'persistence': len(persistence_models),
                'transitions': 1 if transition_models else 0,
                'forecasting': len(forecast_models)
            }
        }
    
    # Helper methods
    
    def _estimate_data_frequency(self, data: pd.DataFrame) -> int:
        """Estimate data frequency in minutes."""
        if len(data) < 2:
            return 1  # Default to 1 minute
        
        # Calculate time differences
        if hasattr(data.index, 'to_pydatetime'):
            time_diffs = data.index.to_series().diff().dropna()
        else:
            # If index is not datetime, assume 1 minute intervals
            return 1
        
        # Get median time difference in minutes
        if len(time_diffs) > 0:
            median_diff = time_diffs.median()
            if hasattr(median_diff, 'total_seconds'):
                frequency_minutes = int(median_diff.total_seconds() / 60)
            else:
                frequency_minutes = 1
        else:
            frequency_minutes = 1
        
        return max(1, frequency_minutes)  # Ensure at least 1 minute
    
    def _calculate_aic(self, data: np.ndarray, distribution, params: Tuple) -> float:
        """Calculate AIC for distribution fit."""
        try:
            log_likelihood = np.sum(distribution.logpdf(data, *params))
            n_params = len(params)
            aic = -2 * log_likelihood + 2 * n_params
            return aic
        except:
            return np.inf
    
    def _create_survival_function(self, dist_type: str, params: Tuple):
        """Create survival function for distribution."""
        if dist_type == 'exponential':
            return lambda t: expon.sf(t, *params)
        elif dist_type == 'gamma':
            return lambda t: gamma.sf(t, *params)
        elif dist_type == 'weibull':
            return lambda t: weibull_min.sf(t, *params)
        else:
            return lambda t: 0.5  # Default