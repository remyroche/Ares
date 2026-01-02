"""
Spectral Chaser for Adaptive Event-Driven Labeling (AEDL)

Enhanced Layer 2.5 Chaser with spectral vision and resonance detection.
Replaces traditional technical analysis with frequency-dependent analysis
using wavelet decomposition and cross-scale resonance.

Key Features:
- Spectral vision with 5-scale wavelet decomposition
- Cross-scale resonance detection for harmonic entries
- Causal compression (20 → 4 alpha features)
- Phase synchronization for breakout vs reversion
- RSV-based position sizing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import catboost as cb
import time

# Import AEDL components
from .adaptive_event_driven_labeling import AdaptiveEventDrivenLabeling
from .causal_compression import CausalCompression

# Import tprint functions
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    # Fallback print functions
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_success(msg): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")
    def tprint_error(msg): print(f"[ERROR] {msg}")


class SpectralChaser:
    """
    Enhanced Layer 2.5 Chaser with spectral vision and resonance detection.
    
    Uses frequency-dependent analysis instead of traditional technical indicators
    to hunt for non-linear alpha in market gaps.
    """
    
    def __init__(
        self,
        causal_graph: Dict[str, List[str]] = None,
        model_types: List[str] = None,
        aedl_params: Dict[str, Any] = None,
        model_params: Dict[str, Any] = None,
        verbose: bool = True
    ):
        """
        Initialize Spectral Chaser.
        
        Args:
            causal_graph: DAG for causal parent filtering
            model_types: Types of models to train
            aedl_params: Parameters for AEDL framework
            model_params: Parameters for ML models
            verbose: Whether to print progress information
        """
        self.verbose = verbose
        self.causal_graph = causal_graph or {}
        self.model_types = model_types or ['xgb', 'catboost', 'rf', 'linear']
        self.aedl_params = aedl_params or {}
        self.model_params = model_params or {}
        
        # Initialize AEDL framework
        self.aedl = AdaptiveEventDrivenLabeling(
            causal_graph=self.causal_graph,
            **self.aedl_params,
            verbose=verbose
        )
        
        # Initialize models
        self.models = {}
        self.model_metrics = {}
        self.feature_names = []
        
        # Training metadata
        self.training_history = {}
        self.prediction_history = {}
        
        if self.verbose:
            tprint_info("🔬 Spectral Chaser: Initializing...")
            tprint_info(f"   ⚙️ Model types: {self.model_types}")
            tprint_info(f"   ⚙️ AEDL framework: Frequency-dependent analysis")
            tprint_info(f"   ⚙️ Causal graph: {'Provided' if self.causal_graph else 'None'}")
            tprint_success("   ✅ Spectral Chaser: Initialization complete")
    
    def fit(
        self,
        df: pd.DataFrame,
        y_residuals: pd.Series,
        causal_anchor_predictions: pd.Series = None,
        specialist_configs: Dict[str, Dict[str, Any]] = None,
        sample_weight: pd.Series = None,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Train Spectral Chaser with frequency-dependent features.
        
        Args:
            df: Market data with OHLCV and derived features
            y_residuals: Target residuals (y_actual - y_causal_anchor)
            causal_anchor_predictions: Causal anchor model predictions
            specialist_configs: Configuration for specialist extraction
            sample_weight: Optional sample weights for training
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with training metrics
        """
        try:
            if self.verbose:
                tprint_info("🚀 Spectral Chaser: Starting training with spectral vision...")
            
            training_start_time = time.time()
            
            # Validate inputs
            if len(df) != len(y_residuals):
                raise ValueError("df and y_residuals length mismatch")
            
            if causal_anchor_predictions is not None and len(causal_anchor_predictions) != len(df):
                raise ValueError("df and causal_anchor_predictions length mismatch")
            
            # Step 1: Process market data through AEDL pipeline
            if self.verbose:
                tprint_info("   🎯 Step 1: AEDL pipeline processing...")
            
            aedl_results = self.aedl.process_market_data(
                df, causal_anchor_predictions, specialist_configs
            )
            
            if 'error' in aedl_results:
                raise ValueError(f"AEDL processing failed: {aedl_results['error']}")
            
            # Step 2: Extract alpha features
            alpha_features = aedl_results.get('alpha_features', {})
            
            if not alpha_features:
                if self.verbose:
                    tprint_warning("   ⚠️ No alpha features available, using spectral components")
                alpha_features = aedl_results.get('spectral_components', {})
            
            if not alpha_features:
                raise ValueError("No features available for training")
            
            # Create feature matrix
            X_alpha = pd.DataFrame(alpha_features)
            
            # Inject Continuous ZoneScore features from input df (passed from Layer 2)
            # This follows the 'Continuous framework' where these are state variables.
            zone_score_cols = [c for c in df.columns if c.startswith('surprise_') or 'zone_score' in c]
            if zone_score_cols:
                if self.verbose:
                    tprint_info(f"      🧬 Injecting {len(zone_score_cols)} continuous ZoneScore features...")
                for col in zone_score_cols:
                    X_alpha[col] = df[col].reindex(X_alpha.index).fillna(0)

            X_alpha = X_alpha.fillna(0)  # Handle NaN values
            
            # Store feature names
            self.feature_names = X_alpha.columns.tolist()
            
            if self.verbose:
                tprint_info(f"      ✅ Feature matrix: {X_alpha.shape}")
                tprint_info(f"         - Samples: {X_alpha.shape[0]}")
                tprint_info(f"         - Features: {X_alpha.shape[1]}")
            
            # Step 3: Train models with cross-validation
            if self.verbose:
                tprint_info("   🧠 Step 2: Training models with cross-validation...")
            
            cv_results = self._train_models_with_cv(X_alpha, y_residuals, cv_folds, sample_weight=sample_weight)
            
            # Step 4: Compile training metrics
            training_time = time.time() - training_start_time
            
            training_metrics = {
                'aedl_results': aedl_results,
                'feature_matrix_shape': X_alpha.shape,
                'feature_names': self.feature_names,
                'cv_results': cv_results,
                'training_time': training_time,
                'model_count': len(self.models),
                'resonance_analysis': aedl_results.get('resonance_analysis', {}),
                'compression_metrics': aedl_results.get('compression_metrics', {}),
                'position_guidance': aedl_results.get('position_sizing_guidance', {})
            }
            
            # Store training history
            self.training_history = training_metrics
            
            if self.verbose:
                tprint_success("✅ Spectral Chaser training complete:")
                tprint_info(f"   - Models trained: {len(self.models)}")
                tprint_info(f"   - Features used: {len(self.feature_names)}")
                tprint_info(f"   - Training time: {training_time:.3f}s")
                
                # Show best model
                if cv_results:
                    best_model = min(cv_results.items(), key=lambda x: x[1]['cv_mse'])
                    tprint_info(f"   - Best model: {best_model[0]} (CV MSE: {best_model[1]['cv_mse']:.6f})")
            
            return training_metrics
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Spectral Chaser training failed: {e}")
            return {'error': str(e)}
    
    def _train_models_with_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int,
        sample_weight: pd.Series = None
    ) -> Dict[str, Dict[str, float]]:
        """Train multiple models with time series cross-validation."""
        try:
            cv_results = {}
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            
            for model_type in self.model_types:
                try:
                    if self.verbose:
                        tprint_info(f"      📊 Training {model_type} model...")
                    
                    # Initialize model
                    model = self._create_model(model_type)
                    
                    # Cross-validation
                    # Note: cross_val_score doesn't directly support sample_weight with scoring='neg_mean_squared_error'
                    # unless passing fit_params.
                    fit_params = {}
                    if sample_weight is not None:
                        # Align weights
                        w_aligned = sample_weight.reindex(X.index).fillna(1.0).values
                        fit_params = {'sample_weight': w_aligned}

                    cv_scores = cross_val_score(
                        model, X, y, cv=tscv, scoring='neg_mean_squared_error',
                        fit_params=fit_params
                    )
                    
                    # Train on full dataset
                    if sample_weight is not None:
                        w_full = sample_weight.reindex(X.index).fillna(1.0).values
                        model.fit(X, y, sample_weight=w_full)
                    else:
                        model.fit(X, y)
                    
                    # Calculate metrics
                    train_pred = model.predict(X)
                    train_mse = mean_squared_error(y, train_pred)
                    train_mae = mean_absolute_error(y, train_pred)
                    
                    # Store model and metrics
                    self.models[model_type] = model
                    cv_results[model_type] = {
                        'cv_mse': -np.mean(cv_scores),
                        'cv_std': np.std(cv_scores),
                        'train_mse': train_mse,
                        'train_mae': train_mae,
                        'cv_scores': cv_scores.tolist()
                    }
                    
                    if self.verbose:
                        tprint_info(f"         ✅ {model_type}: CV MSE = {-np.mean(cv_scores):.6f}")
                    
                except Exception as e:
                    if self.verbose:
                        tprint_warning(f"      ⚠️ {model_type} training failed: {e}")
                    continue
            
            return cv_results
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Model training failed: {e}")
            return {}
    
    def _create_model(self, model_type: str):
        """Create model instance based on type."""
        if self.verbose:
            tprint_info(f"🏗️ Creating {model_type} model")
        params = self.model_params.get(model_type, {})
        
        if model_type == 'xgb':
            default_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'n_jobs': -1
            }
            return xgb.XGBRegressor(**{**default_params, **params})
        
        elif model_type == 'catboost':
            default_params = {
                'iterations': 100,
                'depth': 6,
                'learning_rate': 0.1,
                'random_seed': 42,
                'verbose': False
            }
            return cb.CatBoostRegressor(**{**default_params, **params})
        
        elif model_type == 'rf':
            default_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42,
                'n_jobs': -1
            }
            return RandomForestRegressor(**{**default_params, **params})
        
        elif model_type == 'linear':
            default_params = {}
            return LinearRegression(**{**default_params, **params})
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def predict(
        self,
        df: pd.DataFrame,
        causal_anchor_predictions: pd.Series = None,
        return_resonance: bool = True,
        return_confidence: bool = True
    ) -> Dict[str, Any]:
        """
        Make predictions with spectral context and resonance analysis.
        
        Args:
            df: Market data for prediction
            causal_anchor_predictions: Causal anchor predictions
            return_resonance: Whether to return resonance analysis
            return_confidence: Whether to return confidence scores
            
        Returns:
            Dictionary with predictions and spectral context
        """
        try:
            if self.verbose:
                tprint_info("🔮 Spectral Chaser: Making predictions with spectral vision...")
            
            prediction_start_time = time.time()
            
            # Validate models
            if not self.models:
                raise ValueError("Models not trained. Call fit() first.")
            
            # Step 1: Process market data through AEDL pipeline
            if self.verbose:
                tprint_info("   🎯 Step 1: AEDL pipeline processing...")
            
            aedl_results = self.aedl.process_market_data(df, causal_anchor_predictions)
            
            if 'error' in aedl_results:
                raise ValueError(f"AEDL processing failed: {aedl_results['error']}")
            
            # Step 2: Extract features
            alpha_features = aedl_results.get('alpha_features', {})
            if not alpha_features:
                alpha_features = aedl_results.get('spectral_components', {})
            
            if not alpha_features:
                raise ValueError("No features available for prediction")
            
            # Create feature matrix
            X_alpha = pd.DataFrame(alpha_features)
            X_alpha = X_alpha.fillna(0)
            
            # Align with training features
            if self.feature_names:
                missing_features = set(self.feature_names) - set(X_alpha.columns)
                for feature in missing_features:
                    X_alpha[feature] = 0
                X_alpha = X_alpha[self.feature_names]
            
            # Step 3: Make predictions
            if self.verbose:
                tprint_info("   🧠 Step 2: Generating model predictions...")
            
            predictions = {}
            prediction_metrics = {}
            
            for model_name, model in self.models.items():
                try:
                    pred = model.predict(X_alpha)
                    predictions[model_name] = pred
                    
                    # Calculate confidence (prediction variance across models)
                    if return_confidence:
                        confidence = self._calculate_prediction_confidence(pred, X_alpha, model)
                        prediction_metrics[f'{model_name}_confidence'] = confidence
                    
                except Exception as e:
                    if self.verbose:
                        tprint_warning(f"      ⚠️ {model_name} prediction failed: {e}")
                    continue
            
            # Step 4: Ensemble predictions
            if predictions:
                ensemble_pred = np.mean(list(predictions.values()), axis=0)
                predictions['ensemble'] = ensemble_pred
                
                if return_confidence:
                    pred_std = np.std(list(predictions.values()), axis=0)
                    prediction_metrics['ensemble_confidence'] = pred_std
            
            # Step 5: Compile results
            prediction_time = time.time() - prediction_start_time
            
            results = {
                'predictions': predictions,
                'prediction_metrics': prediction_metrics,
                'aedl_context': aedl_results,
                'prediction_time': prediction_time,
                'n_predictions': len(X_alpha),
                'n_features': len(X_alpha.columns)
            }
            
            # Store prediction history
            self.prediction_history = results
            
            if self.verbose:
                tprint_success("✅ Spectral predictions complete:")
                tprint_info(f"   - Predictions: {len(predictions)} models")
                tprint_info(f"   - Samples: {len(X_alpha)}")
                tprint_info(f"   - Features: {len(X_alpha.columns)}")
                tprint_info(f"   - Prediction time: {prediction_time:.3f}s")
                
                if 'ensemble' in predictions:
                    tprint_info(f"   - Ensemble mean: {np.mean(predictions['ensemble']):.6f}")
                    tprint_info(f"   - Ensemble std: {np.std(predictions['ensemble']):.6f}")
            
            return results
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Spectral prediction failed: {e}")
            return {'error': str(e)}
    
    def _calculate_prediction_confidence(
        self,
        predictions: np.ndarray,
        X: pd.DataFrame,
        model
    ) -> np.ndarray:
        """Calculate prediction confidence based on model uncertainty."""
        if self.verbose:
            tprint_info("📊 Calculating prediction confidence")
        try:
            # Simple confidence based on prediction magnitude and feature variance
            pred_std = np.std(predictions)
            feature_std = np.mean([X[col].std() for col in X.columns])
            
            # Normalize confidence (higher std = lower confidence)
            confidence = 1.0 / (1.0 + pred_std / (feature_std + 1e-9))
            
            return np.full_like(predictions, confidence)
            
        except Exception:
            return np.full_like(predictions, 0.5)
    
    def get_spectral_insights(self) -> Dict[str, Any]:
        """Get spectral insights from latest analysis."""
        if self.verbose:
            tprint_info("🔍 Getting spectral insights")
        try:
            if not self.training_history:
                return {'error': 'No training history available'}
            
            aedl_results = self.training_history.get('aedl_results', {})
            
            insights = {
                'resonance_regime': aedl_results.get('position_sizing_guidance', {}).get('resonance_regime', 'UNKNOWN'),
                'rsv_eigenvalue': aedl_results.get('rsv_eigenvalue', 0.0),
                'compression_ratio': aedl_results.get('compression_metrics', {}).get('total_compression_ratio', 1.0),
                'specialist_count': len(aedl_results.get('specialist_signals', {})),
                'spectral_components': len(aedl_results.get('spectral_components', {})),
                'alpha_features': len(aedl_results.get('alpha_features', {})),
                'harmonic_entries': self.aedl.get_harmonic_entries(),
                'structural_breakouts': self.aedl.get_structural_breakouts()
            }
            
            return insights
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_feature_importance(self, model_type: str = 'ensemble') -> Dict[str, float]:
        """Get feature importance for trained models."""
        if self.verbose:
            tprint_info("📈 Getting feature importance")
        try:
            if model_type == 'ensemble':
                # Average importance across all models
                importance_scores = {}
                
                for model_name, model in self.models.items():
                    if hasattr(model, 'feature_importances_'):
                        importance = model.feature_importances_
                        for i, feature in enumerate(self.feature_names):
                            if i < len(importance):
                                importance_scores[feature] = importance_scores.get(feature, 0) + importance[i]
                
                # Average by number of models
                n_models = len(self.models)
                if n_models > 0:
                    importance_scores = {k: v/n_models for k, v in importance_scores.items()}
                
                return importance_scores
            
            elif model_type in self.models:
                model = self.models[model_type]
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                    return dict(zip(self.feature_names, importance))
            
            return {}
            
        except Exception as e:
            return {}


# Convenience functions for quick usage
def quick_spectral_chaser(
    df_train: pd.DataFrame,
    y_train: pd.Series,
    df_test: pd.DataFrame,
    causal_anchor_train: pd.Series = None,
    causal_anchor_test: pd.Series = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """Quick spectral chaser training and prediction."""
    if verbose:
        tprint_info("🚀 Quick spectral chaser")
    chaser = SpectralChaser(verbose=verbose)
    
    # Train
    training_metrics = chaser.fit(df_train, y_train, causal_anchor_train)
    
    # Predict
    prediction_results = chaser.predict(df_test, causal_anchor_test)
    
    return {
        'training_metrics': training_metrics,
        'prediction_results': prediction_results,
        'spectral_insights': chaser.get_spectral_insights()
    }


if __name__ == "__main__":
    # Example usage
    print("Spectral Chaser for AEDL")
    print("Use quick_spectral_chaser() for quick usage")
    
    print("\nSpectral Chaser Pipeline:")
    print("1. AEDL processing (specialist extraction → wavelet decomposition → resonance)")
    print("2. Causal compression (20 → 4 alpha features)")
    print("3. Model training with cross-validation")
    print("4. Spectral predictions with resonance context")
    print("5. Position sizing guidance based on RSV eigenvalue")
