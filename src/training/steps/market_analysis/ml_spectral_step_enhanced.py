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
from src.utils.ml_common.specialist_xgb import train_specialist_model_with_oof
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
            features[f'dominant_freq_{window}'] = spectral_data.get('dominant_freq', 0.0)
            features[f'phase_{window}'] = spectral_data.get('phase', 0.0)
            
            # Phase change (oscillation velocity)
            features[f'phase_velocity_{window}'] = features[f'phase_{window}'].diff()
            
        return features

    def _get_spectral_combined_manual_features(self, df: pd.DataFrame, pipeline_features: pd.DataFrame) -> pd.DataFrame:
        """Combine frequency domain features and manual enhancements."""
        # 1. Frequency Domain features
        freq_features = self._compute_frequency_domain_features(df)
        
        # 2. Manual Features
        manual_features = self._create_manual_spectral_enhanced_features(df, pipeline_features)
        
        return pd.concat([freq_features, manual_features], axis=1)

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
    
    def generate_pipeline_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Generate base pipeline features."""
        # For spectral regime, we use the combined manual features as pipeline features
        # when called from external consumers (like GMM pipeline)
        return self._get_spectral_combined_manual_features(market_data, pd.DataFrame(index=market_data.index))

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute enhanced spectral analysis specialist with AFML hardening."""
        return await self.execute_standard_specialist_logic(
            config=config,
            specialist_type=SpecialistType.SPECTRAL,
            manual_feature_func=self._get_spectral_combined_manual_features,
            filter_type='volatility',
            pt_sl_config_key='spectral_pt_sl',
            default_pt_sl=[2.5, 1.0],
            suffix="enhanced_spectral_features"
        )
    
    def _load_market_data_with_cache(self, config: Dict[str, Any], timeframe: str) -> Tuple[pd.DataFrame, str]:
        """Load market data using BaseStep method."""
        market_data, market_source = self.load_market_data_or_fail(
            {**config, "timeframe": timeframe},
            pipeline_state={},
            allow_config_override=True,
        )
        return market_data, market_source
