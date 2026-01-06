"""
Enhanced ML Volatility Burst Step with MI Improvements

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
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
import time
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, roc_auc_score

from src.utils.ml_common.afml_utils import (
    get_daily_vol, get_t_events, get_vertical_barrier, 
    apply_triple_barrier, get_bins, get_weights_by_uniqueness,
    frac_diff_fixed, get_sample_weights
)
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
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


class EnhancedMLVolatilityBurstStep(SpecialistDiagnosticsMixinEnhancedV2, AFMLSpecialistMixin, BaseStep):

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

    """
    Enhanced Momentum Persistence Specialist with MI optimization.
    
    Key enhancements:
    - Enhanced feature generation for MI improvement
    - Real-time MI monitoring during training
    - Hyperparameter optimization for MI > 0.02
    - Standardized data structure output
    - Binary output enforcement
    """
    
    def __init__(self, step_name: str = "enhanced_ml_volatility_burst_step"):
        """Initialize the enhanced specialist."""
        BaseStep.__init__(self, step_name, use_versioned_artifacts=True)
        SpecialistDiagnosticsMixinEnhancedV2.__init__(self, step_name=step_name)
        self._current_context = {}
        self._artifact_manager = None
        self._versioned_store = None
        self.step_name = step_name
        self.logger = logger.getChild("EnhancedMLVolatilityBurstStep")
        self.feature_pipeline = MIOptimizedFeaturePipeline()
        self.mi_history = []
        self.training_metrics = []
        self._market_data_cache = {}
        tprint(f"✅ Initialized Enhanced {step_name} (MI-Optimized)", "SUCCESS")
    
    def versioned_store(self):
        """Use a specialist-specific versioned store path."""
        if self._versioned_store is None and self.use_versioned_artifacts:
            symbol = self._current_context.get('symbol', 'UNKNOWN')
            exchange = self._current_context.get('exchange', 'binance')
            timeframe = self._current_context.get('timeframe', '15m')
            direction = self._current_context.get('direction', 'long')
            model = 'enhanced_ml_volatility_burst_step'

            store_name = f"{symbol}_{exchange}_{timeframe}_{direction}_{model}"
            store_path = os.path.join("versioned_artifacts", store_name)

            self._versioned_store = VersionedArtifactStore(
                store_path=store_path,
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
        
    def _compute_enhanced_volatility_optimized_horizon_optimized_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute enhanced momentum features with MI improvements."""
        features = pd.DataFrame(index=df.index)
        
        # Basic momentum features
        returns = df['close'].pct_change()
        
        # Multi-timeframe momentum
        for window in [10,20,40]:
            features[f'momentum_{window}'] = returns.rolling(window).sum()
            features[f'momentum_accel_{window}'] = returns.rolling(window).sum() - returns.rolling(window*2).sum()
            features[f'momentum_vol_{window}'] = returns.rolling(window).std()
        
        # RSI-like momentum
        for window in [10,20,40]:
            gains = returns.clip(lower=0)
            losses = -returns.clip(upper=0)
            avg_gains = gains.rolling(window).mean()
            avg_losses = losses.rolling(window).mean()
            rs = avg_gains / (avg_losses + 1e-8)
            features[f'rsi_{window}'] = 100 - (100 / (1 + rs))
        
        # Price momentum indicators
        for window in [10, 20, 50]:
            sma = df['close'].rolling(window).mean()
            features[f'price_momentum_{window}'] = (df['close'] - sma) / sma
            features[f'price_momentum_cross_{window}'] = (df['close'] > sma).astype(int)
        
        # Volume momentum
        if 'volume' in df.columns:
            volume_change = df['volume'].pct_change()
            for window in [5, 10, 20]:
                features[f'volume_momentum_{window}'] = volume_change.rolling(window).sum()
                features[f'volume_price_momentum_{window}'] = (returns * volume_change).rolling(window).sum()
        
        return features
    
    def _compute_enhanced_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compatibility shim for legacy enhanced momentum helpers."""
        return self._compute_enhanced_volatility_optimized_horizon_optimized_momentum_features(df)

    def _get_volatility_combined_manual_features(self, df: pd.DataFrame, pipeline_features: pd.DataFrame) -> pd.DataFrame:
        """Generate combined manual features for Volatility Burst."""
        momentum_features = self._compute_enhanced_momentum_features(df)
        manual_features = self._create_manual_volatility_burst_enhanced_features(df, pipeline_features)
        
        return pd.concat([momentum_features, manual_features], axis=1)

    def _create_manual_volatility_burst_enhanced_features(self, df: pd.DataFrame, enhanced_features: pd.DataFrame) -> pd.DataFrame:
        """Create manual enhanced features for volatility burst detection."""
        manual_features = pd.DataFrame(index=df.index)
        
        if all(col in df.columns for col in ['close', 'high', 'low', 'volume']):
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df.get('volume', pd.Series(1, index=df.index))
            returns = close.pct_change()
            
            # 1. Enhanced volatility burst features
            # Multi-timeframe volatility signals
            volatility_short = returns.rolling(10).std()
            volatility_medium = returns.rolling(20).std()
            volatility_long = returns.rolling(50).std()
            
            # 1b. Relative Volume (RVOL)
            rvol = volume / (volume.rolling(50).mean() + 1e-8)
            manual_features['enhanced_rvol'] = rvol
            
            # 1c. ATR/Price ratio
            atr = (high - low).rolling(14).mean()
            manual_features['enhanced_atr_price_ratio'] = atr / (close + 1e-8)
            
            # 1d. Bollinger Band Width expansion
            sma_20 = close.rolling(20).mean()
            std_20 = close.rolling(20).std()
            bb_width = (std_20 * 4) / (sma_20 + 1e-8)
            manual_features['enhanced_bb_width'] = bb_width
            manual_features['enhanced_bb_width_expansion'] = bb_width / (bb_width.rolling(50).mean() + 1e-8)
            
            manual_features['volatility_short'] = volatility_short
            manual_features['volatility_medium'] = volatility_medium
            manual_features['volatility_long'] = volatility_long
            
            # Volatility burst detection
            vol_burst_short = volatility_short > (volatility_short.rolling(100).mean() * 2)
            vol_burst_medium = volatility_medium > (volatility_medium.rolling(100).mean() * 2)
            vol_burst_long = volatility_long > (volatility_long.rolling(100).mean() * 2)
            
            manual_features['vol_burst_short'] = vol_burst_short.astype(int)
            manual_features['vol_burst_medium'] = vol_burst_medium.astype(int)
            manual_features['vol_burst_long'] = vol_burst_long.astype(int)
            
            # Volatility regime consistency
            vol_consistency = (volatility_medium > volatility_medium.rolling(100).mean()).rolling(20).mean()
            manual_features['volatility_consistency'] = vol_consistency
            
            # Volatility regime transitions
            vol_transitions = (volatility_medium > volatility_medium.rolling(100).mean()).astype(int).diff().abs()
            manual_features['volatility_transitions'] = vol_transitions
            
            # 2. Volume-adjusted volatility features
            volume_ma = volume.rolling(20).mean()
            volume_ratio = volume / (volume_ma + 1e-8)
            
            volume_adjusted_vol = volatility_medium * (1 + np.log(volume_ratio + 1))
            manual_features['volume_adjusted_volatility'] = volume_adjusted_vol
            
            # Volume-volatility divergence
            volume_regime = (volume_ratio > 1).astype(int)
            volatility_regime = (volatility_medium > volatility_medium.rolling(100).mean()).astype(int)
            volume_vol_divergence = np.abs(volume_regime - volatility_regime)
            manual_features['volume_volatility_divergence'] = volume_vol_divergence
            
            # 3. Range-based volatility features
            range_ratio = (high - low) / close
            range_volatility = volatility_medium * range_ratio
            manual_features['range_volatility'] = range_volatility
            
            # Range-volatility regime
            range_regime = (range_ratio > range_ratio.rolling(50).mean()).astype(int)
            manual_features['range_vol_regime'] = range_regime
            
            # 4. Volatility persistence features
            vol_persistence_short = (volatility_short > volatility_short.rolling(100).mean()).rolling(5).sum()
            vol_persistence_medium = (volatility_medium > volatility_medium.rolling(100).mean()).rolling(10).sum()
            manual_features['vol_persistence_short'] = vol_persistence_short
            manual_features['vol_persistence_medium'] = vol_persistence_medium
            
            # Volatility momentum
            vol_momentum = volatility_medium.diff().rolling(5).mean()
            manual_features['volatility_momentum'] = vol_momentum
            
            # 5. Enhanced volatility price interaction
            # Price-volatility correlation
            price_vol_corr = returns.rolling(20).corr(volatility_medium)
            manual_features['price_volatility_correlation'] = price_vol_corr
            
            # Volatility-adjusted returns
            vol_adjusted_returns = returns / (volatility_medium + 1e-8)
            manual_features['vol_adjusted_returns'] = vol_adjusted_returns
            
            # Volatility regime strength
            vol_regime_strength = abs(volatility_medium - volatility_medium.rolling(100).mean()) / (volatility_medium.rolling(100).std() + 1e-8)
            manual_features['volatility_regime_strength'] = vol_regime_strength
            
            # 6. Volatility burst intensity
            burst_intensity_short = volatility_short / (volatility_short.rolling(100).mean() + 1e-8)
            burst_intensity_medium = volatility_medium / (volatility_medium.rolling(100).mean() + 1e-8)
            manual_features['burst_intensity_short'] = burst_intensity_short
            manual_features['burst_intensity_medium'] = burst_intensity_medium
            
            # Volatility acceleration
            vol_acceleration = volatility_medium.diff().diff()
            manual_features['volatility_acceleration'] = vol_acceleration
            
            # 7. Microstructure volatility features
            # Volatility of volatility
            vol_of_vol = volatility_medium.rolling(20).std()
            manual_features['volatility_of_volatility'] = vol_of_vol
            
            # Volatility depth
            vol_depth = volume * volatility_medium
            manual_features['volatility_depth'] = vol_depth
            
            # Market efficiency indicator
            efficiency = abs(returns.rolling(10).mean()) / (volatility_medium + 1e-8)
            manual_features['market_efficiency'] = efficiency
            
            # 8. Volatility regime classification
            # High volatility regime
            high_vol = (volatility_medium > volatility_medium.rolling(100).quantile(0.75)).astype(int)
            manual_features['high_volatility_regime'] = high_vol
            
            # Low volatility regime
            low_vol = (volatility_medium < volatility_medium.rolling(100).quantile(0.25)).astype(int)
            manual_features['low_volatility_regime'] = low_vol
            
            # Volatility stress indicator
            vol_stress = np.where(volatility_medium > volatility_medium.rolling(100).quantile(0.9), 2, 
                                 np.where(volatility_medium < volatility_medium.rolling(100).quantile(0.1), 0, 1))
            manual_features['volatility_stress'] = vol_stress
            
        return manual_features
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute enhanced volatility burst step."""
        return await self.execute_standard_specialist_logic(
            config=config,
            specialist_type=SpecialistType.VOLATILITY_BURST,
            manual_feature_func=self._get_volatility_combined_manual_features,
            filter_type='volatility',
            pt_sl_config_key='volatility_burst_pt_sl',
            default_pt_sl=[3.0, 1.0],
            suffix="enhanced_volatility_burst_features"
        )
    
    def _load_market_data(self, symbol: str, exchange: str, timeframe: str) -> pd.DataFrame:
        """Load market data using BaseStep method."""
        market_data, _ = self.load_market_data_or_fail(
            {"symbol": symbol, "exchange": exchange, "timeframe": timeframe},
            pipeline_state={},
            allow_config_override=True,
        )
        return market_data
