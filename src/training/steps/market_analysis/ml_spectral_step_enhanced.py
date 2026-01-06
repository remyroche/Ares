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
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, roc_auc_score

from src.utils.tprint import (
    tprint,
    tprint_info,
    tprint_warning,
    tprint_error,
    tprint_success,
)
from src.utils.ml_common.afml_utils import (
    get_daily_vol, get_t_events, get_vertical_barrier, 
    apply_triple_barrier, get_bins, get_weights_by_uniqueness,
    frac_diff_fixed, compute_spectral_energy, get_sample_weights
)
from src.training.steps.market_analysis.specialist_data_standard import SpecialistType
from src.utils.ml_common.specialist_xgb import train_specialist_xgb_with_oof
from src.utils.versioned_artifacts import VersionedArtifactStore
from src.training.steps.base_step import BaseStep
from src.training.steps.market_analysis.specialist_diagnostics_mixin_enhanced_v2 import (
    SpecialistDiagnosticsMixinEnhancedV2
)
from src.training.steps.market_analysis.afml_specialist_mixin import AFMLSpecialistMixin
from src.training.steps.market_analysis.enhanced_feature_generators import MIOptimizedFeaturePipeline
from src.training.steps.market_analysis.specialist_interface import SpecialistDataInterface
from src.training.steps.market_analysis.specialist_data_standard import SpecialistType

logger = logging.getLogger(__name__)


class EnhancedMLSpectralStep(SpecialistDiagnosticsMixinEnhancedV2, AFMLSpecialistMixin, BaseStep):

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
        """Initialize the enhanced specialist."""
        BaseStep.__init__(self, step_name, use_versioned_artifacts=True)
        SpecialistDiagnosticsMixinEnhancedV2.__init__(self, step_name=step_name)
        self._current_context = {}
        self._artifact_manager = None
        self._versioned_store = None
        self.step_name = step_name
        self.logger = logger.getChild("EnhancedMLSpectralStep")
        self.feature_pipeline = MIOptimizedFeaturePipeline()
        self.mi_history = []
        self.training_metrics = []
        self._market_data_cache = {}
        tprint(f"✅ Initialized Enhanced {step_name} (MI-Optimized)", "SUCCESS")
    
    def _compute_frequency_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Frequency Domain Energy features:
        1. Hilbert Transform (Phase, Instantaneous Frequency)
        2. Dominant Cycle Period
        """
        features = pd.DataFrame(index=df.index)
        
        # Spectral focus: Multiple windows for Hilbert Transform
        for window in [50, 100, 200]:
            spectral_data = compute_spectral_energy(df['close'], window=window)
            features[f'dominant_freq_{window}'] = spectral_data['dominant_freq']
            features[f'phase_{window}'] = spectral_data['phase']
            
            # Phase change (oscillation velocity)
            features[f'phase_velocity_{window}'] = features[f'phase_{window}'].diff()
            
        return features

    def _generate_enhanced_features(self, df: pd.DataFrame, specialist_type=None) -> pd.DataFrame:
        """Generate Frequency Domain features for Spectral Specialist."""
        # 1. Frequency Domain focus
        freq_features = self._compute_frequency_domain_features(df)
        
        # 2. Enhanced features from pipeline (if useful)
        enhanced_features = self.feature_pipeline.generate_enhanced_features(
            df, 'spectral', {'enhanced_features': True}
        )
        
        # Combine all features
        all_features = pd.concat([freq_features, enhanced_features], axis=1)
        all_features = all_features.loc[:, ~all_features.columns.duplicated()]
        all_features = all_features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
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
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute enhanced spectral analysis specialist with AFML hardening."""
        start_time = time.time()
        try:
            symbol = str(config.get("symbol", "ETHUSDT"))
            exchange = str(config.get("exchange", "binance"))
            timeframe = str(config.get("timeframe", "15m"))
            direction = str(config.get("direction", "long"))

            # Set context for artifact saving
            self.set_context(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                direction=direction,
                model=self.step_name
            )
            
            self._versioned_store = None
            _ = self.versioned_store
            
            market_data = self._load_market_data(config, timeframe)
            if market_data is None or len(market_data) < 1000:
                return {"success": False, "error": "Insufficient data"}
            
            # 1. Feature Generation
            tprint_info("🛠️ Generating enhanced spectral features...")
            feature_df = self._generate_enhanced_features(market_data)
            
            # 2-4. AFML: Sampling, Labeling, Weighting, Alignment via Helper
            X, y, weights = self.prepare_specialist_data(
                market_data=market_data,
                feature_df=feature_df,
                config=config,
                filter_type='volatility',
                pt_sl_config_key='spectral_pt_sl',
                default_pt_sl=[2.5, 1.0]
            )
            
            # 5. Centralized purged-CV training
            tprint_info("🤖 Training enhanced spectral model with centralized XGB helper (purged CV & AFML weights)...")
            training_result = train_specialist_xgb_with_oof(
                X.fillna(0.0),
                y.fillna(0.0),
                sample_weight=weights,
                n_splits=5,
            )

            oof_probs = training_result.oof_predictions
            last_model = training_result.model
            metrics = training_result.metrics
            
            # AFML Audit: Update metrics using full OOF set
            valid_oof = oof_probs.dropna()
            if len(valid_oof) > 0:
                y_full_true = y.loc[valid_oof.index]
                y_full_pred_prob = valid_oof.values
                y_full_pred = (y_full_pred_prob >= 0.5).astype(int)
                
                if 'auc' not in metrics:
                    try:
                        metrics['auc'] = float(roc_auc_score(y_full_true, y_full_pred_prob))
                    except Exception:
                        metrics['auc'] = 0.5
                if 'mi_score' not in metrics:
                    try:
                        metrics['mi_score'] = float(self.compute_binned_mi(y_full_pred_prob, y_full_true.values))
                    except Exception as e:
                        self.logger.warning(f"Failed to calculate full OOF metrics: {e}")
                        metrics['mi_score'] = 0.0
            else:
                metrics = {'auc': 0.5, 'mi_score': 0.0}
                y_full_pred_prob = np.array([])
                y_full_pred = np.array([])

            metrics.update({
                'n_features': len(X.columns),
                'n_samples': len(X),
            })
            
            # Align results back to full market index
            # AFML FIX: Initialize with NaN instead of 0.5 to allow proper ffilling downstream
            final_probs = pd.Series(np.nan, index=market_data.index)
            if len(valid_oof) > 0:
                final_probs.loc[valid_oof.index] = y_full_pred_prob
            
            # Ffill probabilities so the signal is persistent between events
            final_probs = final_probs.ffill().fillna(0.5)
            final_preds = (final_probs >= 0.5).astype(int)
            
            full_labels = pd.Series(0, index=market_data.index)
            full_labels.loc[y.index] = y
            
            # 6. Standardized Output and Artifacts
            result = self.save_specialist_results(
                config=config,
                feature_df=feature_df,
                labels=full_labels,
                predictions=final_preds.values,
                probabilities=final_probs.values,
                model=last_model,
                metrics=metrics,
                specialist_name="EnhancedMLSpectralStep"
            )
            
            execution_time = time.time() - start_time
            result["execution_time"] = execution_time
            tprint_success(f"✅ Enhanced Spectral Analysis completed in {execution_time:.2f}s")

            return result
            
        except Exception as e:
            self.logger.exception(f"❌ Enhanced Spectral Analysis failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _load_market_data(self, config: Dict[str, Any], timeframe: str) -> pd.DataFrame:
        """Load market data using BaseStep method."""
        market_data, _market_source = self.load_market_data_or_fail(
            {**config, "timeframe": timeframe},
            pipeline_state={},
            allow_config_override=True,
        )
        return market_data
