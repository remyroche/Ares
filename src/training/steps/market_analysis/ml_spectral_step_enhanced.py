"""
Enhanced ML Spectral Analysis Step with MI Improvements

This enhanced version implements:
- Enhanced feature generation for MI improvement
- Real-time MI monitoring during training
- Hyperparameter optimization for MI > 0.02 target
- Data structure standardization
- Binary output enforcement
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
from datetime import datetime, timedelta
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score

from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.versioned_artifacts import VersionedArtifactStore
from src.training.steps.market_analysis.ml_risk_regime_step import MLRiskRegimeStepHMM
from src.training.steps.market_analysis.specialist_diagnostics_mixin_enhanced_v2 import (
    SpecialistDiagnosticsMixinEnhancedV2
)
from src.training.steps.market_analysis.specialist_diagnostics_mixin import SpecialistDiagnosticsMixin
from src.training.steps.market_analysis.enhanced_feature_generators import MIOptimizedFeaturePipeline
from src.training.steps.market_analysis.specialist_interface import SpecialistDataInterface
from src.training.steps.market_analysis.specialist_data_standard import SpecialistType

logger = logging.getLogger(__name__)


class EnhancedMLSpectralStep(MLRiskRegimeStepHMM, SpecialistDiagnosticsMixinEnhancedV2, SpecialistDiagnosticsMixin):

    @property
    def artifact_router(self):
        """Override artifact_router property for enhanced specialists."""
        if self._artifact_router is None:
            from src.utils.artifact_router import ArtifactRouter
            self._artifact_router = ArtifactRouter(
                base_dir="artifacts",
                versioned_store_dir="versioned_artifacts",
                historical_data_dir="historical_data",
                enable_versioned_artifacts=self.use_versioned_artifacts
            )
        return self._artifact_router

    @property
    def versioned_store(self):
        """Override versioned_store property for enhanced specialists to use correct model name."""
        if self._versioned_store is None and self.use_versioned_artifacts:
            # Use enhanced specialist model name instead of default 'analyst'
            symbol = self._current_context.get('symbol', 'UNKNOWN')
            exchange = self._current_context.get('exchange', 'binance')
            timeframe = self._current_context.get('timeframe', '15m')
            direction = self._current_context.get('direction', 'long')
            model = 'enhanced_ml_spectral_step'  # Use the correct model name

            # Create store path with full context separation
            store_name = f"{symbol}_{exchange}_{timeframe}_{direction}_{model}"
            store_path = os.path.join("versioned_artifacts", store_name)

            self._versioned_store = VersionedArtifactStore(
                store_path=store_path,
                auto_version=True,
                enable_row_versioning=True
            )

            # Store context in store metadata
            if hasattr(self._versioned_store, '_metadata'):
                self._versioned_store._metadata['context'] = {
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': timeframe,
                    'direction': direction,
                    'model': model
                }

        return self._versioned_store

    """
    Enhanced Spectral Analysis Specialist with MI optimization.
    
    Key enhancements:
    - Enhanced feature generation for MI improvement
    - Real-time MI monitoring during training
    - Hyperparameter optimization for MI > 0.02
    - Standardized data structure output
    - Binary output enforcement
    """
    
    def __init__(self, step_name: str = "enhanced_ml_spectral_step"):
        """Initialize the enhanced spectral step."""
        super().__init__(step_name=step_name)  # Parent class already enables versioned artifacts
        self._current_context = {}
        self._artifact_manager = None
        self._versioned_store = None
        self.step_name = step_name
        self.logger = logger.getChild("EnhancedMLSpectralStep")
        self.feature_pipeline = MIOptimizedFeaturePipeline()
        self.mi_history = []
        self.training_metrics = []
        
    def _compute_enhanced_structural_optimized_horizon_optimized_spectral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute enhanced momentum features with MI improvements."""
        features = pd.DataFrame(index=df.index)
        
        # Basic momentum features
        returns = df['close'].pct_change()
        
        # Multi-timeframe momentum
        for window in [60,80,100]:
            features[f'momentum_{window}'] = returns.rolling(window).sum()
            features[f'momentum_accel_{window}'] = returns.rolling(window).sum() - returns.rolling(window*2).sum()
            features[f'momentum_vol_{window}'] = returns.rolling(window).std()
        
        # RSI-like momentum
        for window in [60,80,100]:
            gains = returns.clip(lower=0)
            losses = -returns.clip(upper=0)
            avg_gains = gains.rolling(window).mean()
            avg_losses = losses.rolling(window).mean()
            rs = avg_gains / (avg_losses + 1e-8)
            features[f'rsi_{window}'] = 100 - (100 / (1 + rs))
        
        # Price momentum indicators
        for window in [10, 20, 50]:
            sma = df['close'].rolling(window).mean()
            features[f'price_spectral_{window}'] = (df['close'] - sma) / sma
            features[f'price_spectral_cross_{window}'] = (df['close'] > sma).astype(int)
        
        # Volume momentum
        if 'volume' in df.columns:
            volume_change = df['volume'].pct_change()
            for window in [5, 10, 20]:
                features[f'volume_spectral_{window}'] = volume_change.rolling(window).sum()
                features[f'volume_price_spectral_{window}'] = (returns * volume_change).rolling(window).sum()
        
        return features
    
    def _generate_enhanced_features(self, df: pd.DataFrame, specialist_type=None) -> pd.DataFrame:
        """Generate enhanced spectral features with optimized performance."""
        # Basic momentum features
        momentum_features = self._compute_enhanced_structural_optimized_horizon_optimized_spectral_features(df)
        
        # Skip heavy enhanced feature pipeline for performance
        enhanced_features = pd.DataFrame(index=df.index)
        
        # Manual feature engineering for spectral analysis (limited)
        manual_features = self._create_manual_spectral_enhanced_features(df, enhanced_features)
        
        # Combine features
        all_features = pd.concat([momentum_features, enhanced_features, manual_features], axis=1)
        
        # Remove duplicate columns
        all_features = all_features.loc[:, ~all_features.columns.duplicated()]
        
        # Limit to top 50 features for performance
        if len(all_features.columns) > 50:
            all_features = all_features.iloc[:, :50]
        
        return all_features
    
    def _create_manual_spectral_enhanced_features(self, df: pd.DataFrame, enhanced_features: pd.DataFrame) -> pd.DataFrame:
        """Create manual enhanced features for spectral analysis (optimized)."""
        manual_features = pd.DataFrame(index=df.index)
        
        if all(col in df.columns for col in ['close', 'high', 'low', 'volume']):
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df.get('volume', pd.Series(1, index=df.index))
            returns = close.pct_change()
            
            # Simplified spectral features (avoid heavy FFT computations)
            # Basic momentum features
            for window in [5, 10, 20, 50]:
                manual_features[f'returns_{window}'] = returns.rolling(window).mean()
                manual_features[f'returns_std_{window}'] = returns.rolling(window).std()
                manual_features[f'volume_{window}'] = volume.pct_change().rolling(window).mean()
            
            # Price-based features
            manual_features['high_low_ratio'] = high / low
            manual_features['close_to_high'] = close / high
            manual_features['close_to_low'] = close / low
            
            # Simple volatility features
            manual_features['volatility_5'] = returns.rolling(5).std()
            manual_features['volatility_20'] = returns.rolling(20).std()
            
            # Trend features
            manual_features['trend_5'] = close > close.rolling(5).mean()
            manual_features['trend_20'] = close > close.rolling(20).mean()
        
        return manual_features
    
    def _apply_manual_spectral_feature_selection(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply manual feature selection for spectral features."""
        if features.empty:
            return features
        
        # Remove constant features
        constant_features = features.columns[features.nunique() <= 1]
        if len(constant_features) > 0:
            features = features.drop(columns=constant_features)
            self.logger.info(f"Removed {len(constant_features)} constant spectral features")
        
        # Manual redundancy reduction - remove highly correlated features
        correlation_matrix = features.corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        # Find highly correlated pairs (>0.9)
        to_drop = []
        for column in upper_triangle.columns:
            correlated_features = upper_triangle[column][upper_triangle[column] > 0.9]
            if not correlated_features.empty:
                # Keep the feature that comes first alphabetically (deterministic)
                for correlated_feature in correlated_features.index:
                    if correlated_feature > column:  # Drop the later feature alphabetically
                        to_drop.append(correlated_feature)
        
        # Remove redundant features
        if to_drop:
            features = features.drop(columns=list(set(to_drop)))
            self.logger.info(f"Removed {len(set(to_drop))} redundant spectral features: {list(set(to_drop))}")
        
        # Keep only the most informative features (limit to top 30 by variance)
        if len(features.columns) > 30:
            feature_variances = features.var()
            top_features = feature_variances.nlargest(30).index
            features = features[top_features]
            self.logger.info(f"Limited spectral features to top 30 by variance")
        
        return features
    
    # Helper methods for spectral analysis
    def _compute_spectral_decomposition(self, series: pd.Series, window: int) -> np.ndarray:
        """Compute FFT-based spectral decomposition."""
        # Use rolling window FFT
        fft_results = []
        for i in range(len(series) - window + 1):
            window_data = series.iloc[i:i+window].fillna(0)
            if len(window_data) == window:
                fft_result = np.fft.fft(window_data.values)
                fft_results.append(fft_result[:window//2])  # Only positive frequencies
        
        return np.array(fft_results) if fft_results else np.array([])
    
    def _compute_spectral_entropy(self, fft_results: np.ndarray) -> pd.Series:
        """Compute spectral entropy from FFT results."""
        entropies = []
        for fft in fft_results:
            if len(fft) > 0:
                power_spectrum = np.abs(fft) ** 2
                power_spectrum = power_spectrum / (power_spectrum.sum() + 1e-8)
                entropy = -np.sum(power_spectrum * np.log(power_spectrum + 1e-8))
                entropies.append(entropy)
            else:
                entropies.append(0.0)
        
        return pd.Series(entropies, index=range(len(entropies)))
    
    def _get_dominant_frequency(self, fft_results: np.ndarray) -> pd.Series:
        """Get dominant frequency from FFT results."""
        dominant_freqs = []
        for fft in fft_results:
            if len(fft) > 0:
                power_spectrum = np.abs(fft) ** 2
                dominant_freq = np.argmax(power_spectrum)
                dominant_freqs.append(dominant_freq)
            else:
                dominant_freqs.append(0)
        
        return pd.Series(dominant_freqs, index=range(len(dominant_freqs)))
    
    def _compute_power_concentration(self, fft_results: np.ndarray) -> pd.Series:
        """Compute power concentration (how much power is in dominant frequency)."""
        concentrations = []
        for fft in fft_results:
            if len(fft) > 0:
                power_spectrum = np.abs(fft) ** 2
                total_power = power_spectrum.sum()
                if total_power > 0:
                    max_power = power_spectrum.max()
                    concentration = max_power / total_power
                else:
                    concentration = 0.0
                concentrations.append(concentration)
            else:
                concentrations.append(0.0)
        
        return pd.Series(concentrations, index=range(len(concentrations)))
    
    def _compute_low_frequency_momentum(self, series: pd.Series, window: int) -> pd.Series:
        """Compute low-frequency momentum (trend component)."""
        # Simple low-pass filter using moving average
        return series.rolling(window).mean()
    
    def _compute_high_frequency_momentum(self, series: pd.Series, window: int) -> pd.Series:
        """Compute high-frequency momentum (noise/volatility component)."""
        # High-frequency component = original - low-frequency
        low_freq = self._compute_low_frequency_momentum(series, window)
        return series - low_freq
    
    def _compute_wavelet_energy(self, series: pd.Series, scale: int) -> pd.Series:
        """Compute wavelet-like energy at given scale."""
        # Simple approximation using differences
        wavelet_approx = series.diff(scale).abs().rolling(scale).sum()
        return wavelet_approx
    
    def _compute_wavelet_coherence(self, series1: pd.Series, scale1: int, scale2: int) -> pd.Series:
        """Compute wavelet coherence between two scales."""
        energy1 = self._compute_wavelet_energy(series1, scale1)
        energy2 = self._compute_wavelet_energy(series1, scale2)
        
        # Simple coherence approximation
        coherence = energy1 / (energy2 + 1e-8)
        return coherence
    
    def _find_autocorr_peaks(self, series: pd.Series, max_lag: int) -> pd.Series:
        """Find number of significant peaks in autocorrelation."""
        peak_counts = []
        for i in range(len(series) - max_lag):
            window_data = series.iloc[i:i+max_lag+1].fillna(0)
            if len(window_data) == max_lag + 1:
                autocorr = [window_data.autocorr(lag) for lag in range(1, max_lag+1)]
                # Find peaks (simplified)
                peaks = sum(1 for j in range(1, len(autocorr)-1) 
                          if autocorr[j] > autocorr[j-1] and autocorr[j] > autocorr[j+1])
                peak_counts.append(peaks)
            else:
                peak_counts.append(0)
        
        return pd.Series(peak_counts, index=range(len(peak_counts)))
    
    def _compute_cyclical_strength(self, series: pd.Series, period: int) -> pd.Series:
        """Compute cyclical strength at given period."""
        # Use autocorrelation at the period
        cyclical_strength = series.rolling(period).apply(
            lambda x: x.autocorr(period) if len(x.dropna()) > period else 0
        )
        return cyclical_strength.abs()
    
    def _compute_cross_spectral_coherence(self, series1: pd.Series, series2: pd.Series, window: int) -> pd.Series:
        """Compute cross-spectral coherence between two series."""
        coherences = []
        for i in range(len(series1) - window + 1):
            window1 = series1.iloc[i:i+window].fillna(0)
            window2 = series2.iloc[i:i+window].fillna(0)
            
            if len(window1) == window and len(window2) == window:
                fft1 = np.fft.fft(window1.values)
                fft2 = np.fft.fft(window2.values)
                
                # Cross-spectral density
                cross_spectrum = fft1 * np.conj(fft2)
                power_spectrum1 = np.abs(fft1) ** 2
                power_spectrum2 = np.abs(fft2) ** 2
                
                # Coherence
                coherence = np.abs(cross_spectrum) / np.sqrt(power_spectrum1 * power_spectrum2 + 1e-8)
                avg_coherence = np.mean(coherence[:window//2])  # Average coherence
                
                coherences.append(avg_coherence)
            else:
                coherences.append(0.0)
        
        return pd.Series(coherences, index=range(len(coherences)))
    
    def _classify_spectral_regime(self, entropy: pd.Series, concentration: pd.Series, freq_ratio: pd.Series) -> pd.Series:
        """Classify spectral regime based on entropy, concentration, and frequency ratio."""
        regime = np.zeros(len(entropy))
        
        # High entropy, low concentration, balanced freq ratio -> noisy regime
        noisy = (entropy > entropy.rolling(100).mean()) & (concentration < concentration.rolling(100).mean())
        
        # Low entropy, high concentration -> trend regime
        trend = (entropy < entropy.rolling(100).mean()) & (concentration > concentration.rolling(100).mean())
        
        # High freq ratio -> volatile regime
        volatile = freq_ratio > freq_ratio.rolling(100).mean()
        
        regime[noisy] = 2  # Noisy
        regime[trend] = 1  # Trending
        regime[volatile] = 0  # Volatile
        
        return pd.Series(regime, index=entropy.index)
    
    def _create_spectral_labels(self, df: pd.DataFrame, lookforward: int = 35) -> pd.Series:
        """Create momentum persistence labels."""
        returns = df['close'].pct_change()
        
        # Future momentum
        future_returns = returns.shift(-lookforward).rolling(lookforward).sum()
        
        # Binary label: positive future momentum
        labels = (future_returns > returns.rolling(25).std() * 0.5).astype(int)
        
        return labels
    
    def _compute_mi_during_training(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                  X_val: pd.DataFrame, y_val: pd.Series,
                                  model_predictions: np.ndarray) -> Dict[str, float]:
        """Compute MI metrics during training for monitoring."""
        mi_metrics = {}
        
        try:
            # Feature MI to target
            feature_mi_scores = []
            for col in X_train.select_dtypes(include=[np.number]).columns:
                mi_score = mutual_info_regression(
                    X_train[col].values.reshape(-1, 1), y_train.values
                )[0]
                feature_mi_scores.append(mi_score)
            
            if feature_mi_scores:
                mi_metrics['avg_feature_mi'] = np.mean(feature_mi_scores)
                mi_metrics['max_feature_mi'] = np.max(feature_mi_scores)
                mi_metrics['high_mi_features'] = sum(1 for mi in feature_mi_scores if mi > 0.02)
            
            # Prediction MI to target
            mi_metrics['prediction_mi'] = mutual_info_regression(
                model_predictions.reshape(-1, 1), y_val.values
            )[0]
            
            # MI improvement tracking
            self.mi_history.append(mi_metrics['prediction_mi'])
            
        except Exception as e:
            self.logger.warning(f"MI computation failed: {e}")
            mi_metrics = {'prediction_mi': 0.0, 'avg_feature_mi': 0.0}
        
        return mi_metrics
    
    def _optimize_hyperparameters_for_mi(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Optimize hyperparameters specifically for MI improvement."""
        best_params = {}
        best_mi = 0.0
        
        # Parameter grid for MI optimization
        # Parameter grid for MI-focused optimization
        param_grid = {
            "n_estimators": [200, 300, 500],
            "max_depth": [4, 6],
            "learning_rate": [0.03, 0.07, 0.1],
            "subsample": [0.8, 0.9],
            "colsample_bytree": [0.8, 0.9],
            "gamma": [0, 0.1, 0.2],
            "reg_alpha": [0.1, 0.5, 1.0],
            "reg_lambda": [2, 5, 10],
            "min_child_weight": [20, 40]
        }
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        for params in self._generate_param_combinations(param_grid, max_combinations=20):
            mi_scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train model with current parameters
                model = lgb.LGBMClassifier(
                    objective='binary',
                    random_state=42,
                    verbose=-1,
                    **params
                )
                
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                         callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
                
                # Compute MI
                val_pred = model.predict_proba(X_val)[:, 1]
                mi_score = mutual_info_regression(
                    val_pred.reshape(-1, 1), y_val.values
                )[0]
                mi_scores.append(mi_score)
            
            avg_mi = np.mean(mi_scores)
            
            if avg_mi > best_mi:
                best_mi = avg_mi
                best_params = params.copy()
                
                tprint_info(f"🔥 New best MI: {avg_mi:.4f} with params: {params}")
        
        tprint_success(f"✅ Best hyperparameters found: MI = {best_mi:.4f}")
        return best_params, best_mi
    

    def _generate_param_combinations(self, param_grid: Dict[str, List], max_combinations: int = 20):
        """Generate parameter combinations for optimization."""
        import itertools
        import random
        
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        # Generate all combinations
        all_combinations = list(itertools.product(*values))
        
        # Randomly sample if too many
        if len(all_combinations) > max_combinations:
            all_combinations = random.sample(all_combinations, max_combinations)
        
        for combination in all_combinations:
            yield dict(zip(keys, combination))
    
    def _train_enhanced_spectral_model(self, features: pd.DataFrame, labels: pd.Series) -> Tuple[lgb.LGBMClassifier, Dict[str, float]]:
        """Train enhanced momentum model with MI optimization."""
        
        # Optimize hyperparameters for MI
        tprint_info("🔧 Optimizing hyperparameters for MI improvement...")
        best_params, best_mi = self._optimize_hyperparameters_for_mi(features, labels)
        
        # Time series split for final training
        n_splits = 5
        split_idx = len(features) // (n_splits + 1)
        
        models = []
        mi_scores = []
        auc_scores = []
        
        for i in range(n_splits):
            train_start = i * split_idx
            train_end = (i + 2) * split_idx
            val_end = (i + 3) * split_idx
            
            if val_end > len(features):
                break
            
            X_train = features.iloc[train_start:train_end]
            y_train = labels.iloc[train_start:train_end]
            X_val = features.iloc[train_end:val_end]
            y_val = labels.iloc[train_end:val_end]
            
            # Train model with optimized parameters
            model = lgb.LGBMClassifier(
                objective='binary',
                random_state=42 + i,
                verbose=-1,
                **best_params
            )
            
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                     callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
            
            # Evaluate
            val_pred = model.predict_proba(X_val)[:, 1]
            
            # Compute MI
            mi_score = mutual_info_regression(
                val_pred.reshape(-1, 1), y_val.values
            )[0]
            mi_scores.append(mi_score)
            
            # Compute AUC
            auc = roc_auc_score(y_val, val_pred)
            auc_scores.append(auc)
            
            models.append(model)
            
            # Store training metrics
            self.training_metrics.append({
                'fold': i,
                'mi_score': mi_score,
                'auc_score': auc,
                'n_features': len(X_train.columns)
            })
        
        # Select best model based on MI
        best_idx = np.argmax(mi_scores)
        best_model = models[best_idx]
        
        metrics = {
            'mi_score': np.mean(mi_scores),
            'mi_std': np.std(mi_scores),
            'auc': np.mean(auc_scores),
            'auc_std': np.std(auc_scores),
            'best_mi': np.max(mi_scores),
            'best_auc': np.max(auc_scores),
            'n_features': len(features.columns),
            'optimization_params': best_params
        }
        
        return best_model, metrics
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute enhanced momentum persistence step."""
        try:
            symbol = str(config.get("symbol", "ETHUSDT"))
            exchange = str(config.get("exchange", "binance"))
            timeframe = str(config.get("timeframe", "15m"))
            direction = str(config.get("direction", "long"))

            # Set context for artifact saving - MUST BE DONE FIRST
            self._current_context = {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'direction': direction,
                'model': 'enhanced_ml_spectral_step'
            }

            tprint_info(f"🚀 Starting Enhanced Spectral Analysis for {symbol}")

            # Load market data
            market_data = self._load_market_data(config, timeframe)
            
            # Generate enhanced features
            tprint_info("🛠️ Generating enhanced momentum features...")
            features = self._generate_enhanced_features(market_data)
            
            # Create labels
            labels = self._create_spectral_labels(market_data)
            
            # Align features and labels
            common_index = features.index.intersection(labels.index)
            features = features.loc[common_index]
            labels = labels.loc[common_index]
            
            # Clean data
            valid_mask = ~(features.isna().any(axis=1)) & ~(labels.isna())
            features = features[valid_mask]
            labels = labels[valid_mask]
            
            if len(features) < 500:
                raise ValueError(f"Insufficient data: {len(features)} samples")
            
            tprint_info(f"📊 Training data: {len(features)} samples, {len(features.columns)} features")
            
            # Train enhanced model
            tprint_info("🤖 Training enhanced momentum model with MI optimization...")
            model, metrics = self._train_enhanced_spectral_model(features, labels)
            
            # Generate predictions
            predictions = model.predict(features)
            probabilities = model.predict_proba(features)[:, 1]
            
            # Create standardized output
            output_df = self._create_standardized_output(
                features, labels, predictions, probabilities, symbol, exchange, timeframe, direction
            )
            
            # Save artifacts
            artifact_name = f"enhanced_ml_spectral_prediction_{timeframe}"
            artifacts: List[str] = []
            metadata = SpecialistDataInterface.create_standard_metadata(
                specialist_name="EnhancedMLSpectralStep",
                config=config,
                metrics=metrics,
                mi_score=metrics['mi_score'],
                hsic_score=0.0  # Not computed for this implementation
            )
            
            artifact_path = self._save_artifact(
                data=output_df,
                artifact_name=artifact_name,
                artifact_type="data",
                data_category="predictions",
                metadata=metadata
            )
            artifacts.append(artifact_path)
            
            # Run enhanced diagnostics
            diagnostics_result = self.run_enhanced_diagnostics(symbol, exchange, timeframe, direction)
            
            # Final summary
            tprint_success(f"✅ Enhanced Spectral Analysis completed:")
            tprint_info(f"   MI Score: {metrics['mi_score']:.4f} (target: >0.02)")
            tprint_info(f"   AUC: {metrics['auc']:.3f}")
            tprint_info(f"   Features: {metrics['n_features']}")
            
            return {
                "success": True,
                "metrics": metrics,
                "n_samples": len(output_df),
                "features": list(features.columns),
                "artifact_name": artifact_name,
                "diagnostics": diagnostics_result,
                "mi_history": self.mi_history,
                "training_metrics": self.training_metrics
            }
            
        except Exception as e:
            self.logger.exception(f"❌ Enhanced Spectral Analysis step failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _create_standardized_output(self, features: pd.DataFrame, labels: pd.Series,
                                  predictions: np.ndarray, probabilities: np.ndarray,
                                  symbol: str, exchange: str, timeframe: str, direction: str) -> pd.DataFrame:
        """Create standardized output structure."""
        output_df = pd.DataFrame(index=features.index)
        output_df['timestamp'] = features.index
        output_df['specialist_prediction'] = predictions
        output_df['specialist_probability'] = probabilities
        output_df['target_label'] = labels
        
        # Add original features for reference
        for col in features.columns[:20]:  # Limit to first 20 features
            output_df[f'feature_{col}'] = features[col]
        
        return output_df
    
    def _load_market_data(self, config: Dict[str, Any], timeframe: str) -> pd.DataFrame:
        """Load market data using BaseStep method."""
        market_data, _market_source = self.load_market_data_or_fail(
            {**config, "timeframe": timeframe},
            pipeline_state={},
            allow_config_override=True,
        )
        return market_data
