"""
Enhanced ML Momentum Persistence Step with MI Improvements

This enhanced version implements:
- AFML hardening: CUSUM filtering (Price), Triple Barrier Method, Uniqueness weighting
- Enhanced feature generation for MI improvement
- Real-time MI monitoring during training
- Hyperparameter optimization for MI > 0.02 target
"""

import os
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple, List
from pathlib import Path
import logging
import time
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
    frac_diff_fixed, compute_structural_inertia, get_sample_weights
)
from src.utils.versioned_artifacts import VersionedArtifactStore
from src.training.steps.base_step import BaseStep
from src.training.steps.market_analysis.specialist_diagnostics_mixin_enhanced_v2 import (
    SpecialistDiagnosticsMixinEnhancedV2
)
from src.training.steps.market_analysis.afml_specialist_mixin import AFMLSpecialistMixin
from src.training.steps.market_analysis.enhanced_feature_generators import MIOptimizedFeaturePipeline
from src.training.steps.market_analysis.specialist_interface import SpecialistDataInterface
from src.training.steps.market_analysis.specialist_data_standard import SpecialistType
from src.utils.ml_common.specialist_xgb import train_specialist_model_with_oof

logger = logging.getLogger(__name__)


class EnhancedMLMomentumPersistenceStep(SpecialistDiagnosticsMixinEnhancedV2, AFMLSpecialistMixin, BaseStep):

    @property
    def artifact_router(self):
        """Override artifact_router property for enhanced specialists."""
        if self._artifact_router is None:
            from src.utils.artifact_router import ArtifactRouter
            # Discover repo root dynamically
            repo_root = Path(__file__).resolve().parents[4]
            self._artifact_router = ArtifactRouter(
                base_dir=str(repo_root / "artifacts"),
                versioned_store_dir=str(repo_root / "versioned_artifacts"),
                historical_data_dir=str(repo_root / "historical_data"),
                enable_versioned_artifacts=self.use_versioned_artifacts
            )
        return self._artifact_router

    def __init__(self, step_name: str = "enhanced_ml_momentum_persistence_step"):
        """Initialize the enhanced specialist."""
        BaseStep.__init__(self, step_name, use_versioned_artifacts=True)
        SpecialistDiagnosticsMixinEnhancedV2.__init__(self, step_name=step_name)
        self._current_context = {}
        self._artifact_manager = None
        self._versioned_store = None
        self.step_name = step_name
        self.logger = logger.getChild("EnhancedMLMomentumPersistenceStep")
        self.feature_pipeline = MIOptimizedFeaturePipeline()
        self.mi_history = []
        self.training_metrics = []
        self._market_data_cache = {}
        tprint(f"✅ Initialized Enhanced {step_name} (MI-Optimized)", "SUCCESS")
    
    @property
    def versioned_store(self):
        """Use a specialist-specific versioned store path."""
        if self._versioned_store is None and self.use_versioned_artifacts:
            symbol = self._current_context.get('symbol', 'UNKNOWN')
            exchange = self._current_context.get('exchange', 'binance')
            timeframe = self._current_context.get('timeframe', '15m')
            direction = self._current_context.get('direction', 'long')
            model = self.step_name

            store_name = f"{symbol}_{exchange}_{timeframe}_{direction}_{model}"
            # Discover repo root dynamically
            repo_root = Path(__file__).resolve().parents[4]
            store_root = repo_root / "versioned_artifacts"
            store_root.mkdir(parents=True, exist_ok=True)
            store_path = store_root / store_name

            self._versioned_store = VersionedArtifactStore(
                store_path=str(store_path),
                auto_version=True,
                enable_row_versioning=True
            )

            if hasattr(self._versioned_store, '_metadata'):
                self._versioned_store._metadata['context'] = {
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': timeframe,
                    'direction': direction,
                    'model': model
                }
                self._versioned_store._save_metadata()

        return self._versioned_store
        
    def _compute_structural_inertia_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Structural Inertia features:
        1. Fractional Differentiation (d=0.3 to 0.7)
        2. Normalized Regression Slope
        """
        features = pd.DataFrame(index=df.index)
        
        # Apply Fractional Differentiation to preserve memory
        d_val = 0.4
        close_fd = frac_diff_fixed(df['close'], d=d_val)
        # Re-index to match original df
        close_fd = close_fd.reindex(df.index).fillna(method='ffill').fillna(0)
        
        # Calculate Structural Inertia (Slope / SE) on FD series
        # Optimization: use step=5 to speed up rolling regressions
        for window in [20, 40, 60]:
            features[f'structural_inertia_{window}'] = compute_structural_inertia(close_fd, window=window, step=5)
            # Acceleration of inertia
            features[f'inertia_accel_{window}'] = features[f'structural_inertia_{window}'].diff()
            
        return features

    def _get_momentum_combined_manual_features(self, df: pd.DataFrame, pipeline_features: pd.DataFrame) -> pd.DataFrame:
        """Generate combined manual features for Momentum Persistence."""
        inertia_features = self._compute_structural_inertia_features(df)
        manual_features = self._create_manual_enhanced_features(df, pipeline_features)
        
        return pd.concat([inertia_features, manual_features], axis=1)

    def _create_manual_enhanced_features(self, df: pd.DataFrame, enhanced_features: pd.DataFrame) -> pd.DataFrame:
        """Create manual enhanced features to address redundancy and improve poor performers."""
        manual_features = pd.DataFrame(index=df.index)
        
        if all(col in df.columns for col in ['close', 'high', 'low', 'volume']):
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df.get('volume', pd.Series(1, index=df.index))
            returns = close.pct_change()
            
            # 1. Address smc|volume_force_breakout redundancy (0.632)
            # Create orthogonal regime features
            if 'smc_predicted' in enhanced_features.columns and 'vol_force_breakout' in enhanced_features.columns:
                smc = enhanced_features['smc_predicted']
                vol = enhanced_features['vol_force_breakout']
                
                # Standardize for orthogonal decomposition
                smc_std = (smc - smc.mean()) / (smc.std() + 1e-8)
                vol_std = (vol - vol.mean()) / (vol.std() + 1e-8)
                
                # Create orthogonal volume signal (remove smc component)
                if len(smc_std) > 1:
                    cov_matrix = np.cov(vol_std, smc_std)
                    if cov_matrix.shape == (2, 2):
                        orthogonal_vol = vol_std - (cov_matrix[0,1] / (np.var(smc_std) + 1e-8)) * smc_std
                        orthogonal_vol = orthogonal_vol / (orthogonal_vol.std() + 1e-8)
                        manual_features['orthogonal_volume_regime'] = orthogonal_vol
                
                # Regime divergence (captures disagreement)
                regime_divergence = np.abs(smc_std - vol_std)
                manual_features['regime_divergence'] = regime_divergence
                
                # Regime consensus (captures agreement)
                regime_consensus = (smc_std + vol_std) / 2
                manual_features['regime_consensus'] = regime_consensus
            
            # 2. Improve risk_score
            medium_vol = returns.rolling(20).std()
            
            # Volume-adjusted risk
            volume_ma = volume.rolling(20).mean()
            volume_ratio = volume / (volume_ma + 1e-8)
            volume_adjusted_risk = medium_vol * (1 + np.log(volume_ratio + 1))
            manual_features['enhanced_volume_adjusted_risk'] = volume_adjusted_risk
            
            # Range-based risk
            range_ratio = (high - low) / (close + 1e-8)
            range_vol = range_ratio.rolling(20).std()
            manual_features['enhanced_range_based_risk'] = range_vol
            
            # Downside risk
            downside_returns = returns.copy()
            downside_returns[downside_returns > 0] = 0
            downside_vol = downside_returns.rolling(20).std()
            manual_features['enhanced_downside_risk'] = downside_vol
            
            # Risk regime classification
            risk_zscore = (medium_vol - medium_vol.rolling(100).mean()) / (medium_vol.rolling(100).std() + 1e-8)
            manual_features['enhanced_risk_regime'] = np.where(risk_zscore > 1, 2, np.where(risk_zscore < -1, 0, 1))
            
            # 3. Improve path_risk_score
            price_path = close.rolling(10).mean()
            path_smoothness = np.abs(price_path.diff().diff())
            manual_features['enhanced_path_smoothness'] = path_smoothness
            
            path_velocity = close.rolling(5).mean().diff()
            path_acceleration = path_velocity.diff()
            manual_features['enhanced_path_acceleration'] = path_acceleration
            
            path_vol = path_velocity.rolling(20).std()
            manual_features['enhanced_path_volatility'] = path_vol
            
            path_range = price_path.rolling(20).max() - price_path.rolling(20).min()
            path_breakout = np.abs(close - price_path) / (path_range + 1e-8)
            manual_features['enhanced_path_breakout'] = path_breakout
            
            # 4. Additional orthogonal momentum features
            momentum_regime = (returns.rolling(20).mean() > 0).astype(int)
            manual_features['momentum_regime'] = momentum_regime
            
            vol_adjusted_momentum = returns.rolling(10).mean() / (returns.rolling(10).std() + 1e-8)
            manual_features['vol_adjusted_momentum'] = vol_adjusted_momentum
            
            momentum_persistence = (returns.rolling(5).mean() * returns.rolling(10).mean()).rolling(5).sum()
            manual_features['momentum_persistence'] = momentum_persistence
            
        return manual_features
    
    def generate_pipeline_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Generate base pipeline features."""
        # For momentum persistence, we use the combined manual features as pipeline features
        # when called from external consumers (like GMM pipeline)
        return self._get_momentum_combined_manual_features(market_data, pd.DataFrame(index=market_data.index))

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute enhanced momentum persistence specialist training with AFML hardening."""
        return await self.execute_standard_specialist_logic(
            config=config,
            specialist_type=SpecialistType.MOMENTUM_PERSISTENCE,
            manual_feature_func=self._get_momentum_combined_manual_features,
            filter_type='price',
            pt_sl_config_key='momentum_pt_sl',
            default_pt_sl=[2.5, 1.0],
            suffix="enhanced_momentum_persistence_features"
        )

    def _load_market_data(self, symbol: str, exchange: str, timeframe: str) -> pd.DataFrame:
        """Load market data using BaseStep method."""
        market_data, _ = self.load_market_data_or_fail(
            {"symbol": symbol, "exchange": exchange, "timeframe": timeframe},
            pipeline_state={},
            allow_config_override=True,
        )
        return market_data
