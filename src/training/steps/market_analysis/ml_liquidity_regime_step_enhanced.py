"""
Enhanced ML Liquidity Regime Step with MI Improvements

This enhanced version implements:
- Enhanced feature generation for MI improvement
- Real-time MI monitoring during training
- Hyperparameter optimization for MI > 0.02 target
- Data structure standardization
- Binary output enforcement
"""

import logging
import time
from typing import Any, Dict, Optional, Tuple, List
from datetime import datetime
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.metrics import accuracy_score, roc_auc_score

from src.training.steps.base_step import BaseStep
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
    frac_diff_fixed, get_sample_weights
)
from src.utils.versioned_artifacts.temporal_splits import (
    create_temporal_split_config_for_pipeline,
    TemporalSplitConfig,
)
from src.training.steps.market_analysis.specialist_diagnostics_mixin_enhanced_v2 import (
    SpecialistDiagnosticsMixinEnhancedV2
)
from src.training.steps.market_analysis.afml_specialist_mixin import AFMLSpecialistMixin
from src.training.steps.market_analysis.enhanced_feature_generators import MIOptimizedFeaturePipeline
from src.training.steps.market_analysis.specialist_interface import SpecialistDataInterface
from src.training.steps.market_analysis.specialist_data_standard import SpecialistType
from src.utils.ml_common.specialist_xgb import train_specialist_model_with_oof
from src.utils.data_loader import DataLoader

logger = logging.getLogger(__name__)


class EnhancedMLLiquidityRegimeStep(SpecialistDiagnosticsMixinEnhancedV2, AFMLSpecialistMixin, BaseStep):

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
    Enhanced Liquidity Regime Specialist with MI optimization.
    
    Key enhancements:
    - Enhanced feature generation for MI improvement
    - Real-time MI monitoring during training
    - Hyperparameter optimization for MI > 0.02
    - Standardized data structure output
    - Binary output enforcement
    """
    
    def __init__(self, step_name: str = "enhanced_ml_liquidity_regime_step"):
        """Initialize the enhanced specialist."""
        BaseStep.__init__(self, step_name, use_versioned_artifacts=True)
        SpecialistDiagnosticsMixinEnhancedV2.__init__(self, step_name=step_name)
        self._current_context = {}
        self._artifact_manager = None
        self._versioned_store = None
        self.step_name = step_name
        self.logger = logger.getChild("EnhancedMLLiquidityRegimeStep")
        self.feature_pipeline = MIOptimizedFeaturePipeline()
        self.mi_history = []
        self.training_metrics = []
        self._market_data_cache = {}
        tprint(f"✅ Initialized Enhanced {step_name} (MI-Optimized)", "SUCCESS")
    
    def _get_liquidity_combined_manual_features(self, df: pd.DataFrame, pipeline_features: pd.DataFrame) -> pd.DataFrame:
        """Combine liquidity features, enhanced features, and specific liquidity enhancements."""
        # Import original liquidity features
        try:
             # Reconstruct basic config from context
            config = {
                'symbol': self._current_context.get('symbol'),
                'exchange': self._current_context.get('exchange'),
                'timeframe': self._current_context.get('timeframe'),
                'direction': self._current_context.get('direction')
            }
            from src.feature_generation.categories.liquidity_regime_features import generate_liquidity_regime_features
            base_liquidity_features = generate_liquidity_regime_features(df, config)
        except ImportError:
            base_liquidity_features = pd.DataFrame(index=df.index)

        # Manual feature engineering for liquidity regime
        manual_features = self._create_manual_liquidity_enhanced_features(df, pipeline_features)
        
        # Combine all features
        all_features = pd.concat([base_liquidity_features, manual_features], axis=1)
        
        return all_features
    
    def _create_manual_liquidity_enhanced_features(self, df: pd.DataFrame, enhanced_features: pd.DataFrame) -> pd.DataFrame:
        """Create manual enhanced features for liquidity regime detection."""
        manual_features = pd.DataFrame(index=df.index)
        
        if all(col in df.columns for col in ['close', 'high', 'low', 'volume']):
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df.get('volume', pd.Series(1, index=df.index))
            returns = close.pct_change()
            
            # 1. Enhanced liquidity depth features (State-focused)
            # Use multi-timeframe median volume for more robust capacity proxy
            volume_median_50 = volume.rolling(50).median()
            volume_median_200 = volume.rolling(200).median()
            
            # Re-define volume ratios for compatibility with later features
            volume_ma_short = volume.rolling(10).mean()
            volume_ma_medium = volume.rolling(20).mean()
            volume_ma_long = volume.rolling(50).mean()
            
            volume_ratio_short = volume / (volume_ma_short + 1e-8)
            volume_ratio_medium = volume / (volume_ma_medium + 1e-8)
            volume_ratio_long = volume / (volume_ma_long + 1e-8)
            
            manual_features['liquidity_capacity_ratio'] = volume / (volume_median_200 + 1e-8)
            manual_features['liquidity_depth_stability'] = volume_median_50 / (volume_median_200 + 1e-8)
            
            # AMIHUD-style Illiquidity Proxy (Price Impact)
            manual_features['amihud_illiquidity'] = abs(returns) / (volume * close + 1e-8)
            
            # Liquidity stress based on tails of volume distribution
            vol_std_200 = volume.rolling(200).std()
            manual_features['liquidity_stress_low'] = (volume < (volume_median_200 - 1.5 * vol_std_200)).astype(int)
            manual_features['liquidity_stress_high'] = (volume > (volume_median_200 + 2.5 * vol_std_200)).astype(int)
            
            # 2. Price-impact and Efficiency Interaction
            # Market Efficiency Coefficient (MEC) - variation of price move relative to sum of returns
            # Detects if liquidity is "absorbing" moves or if price is "slipping"
            abs_sum_rets = returns.abs().rolling(20).sum()
            range_rets = (close.rolling(20).max() - close.rolling(20).min()) / close
            manual_features['market_absorption_ratio'] = range_rets / (abs_sum_rets + 1e-8)
            
            # Liquidity-adjusted volatility
            volatility = returns.rolling(20).std()
            liquidity_adjusted_vol = volatility / (volume_ratio_medium + 1e-8)
            manual_features['liquidity_adjusted_volatility'] = liquidity_adjusted_vol
            
            # Liquidity-adjusted slippage proxy
            manual_features['slippage_proxy'] = (high - low) / (volume_ratio_medium + 1e-8)
            
            # 3. Range-based liquidity features
            range_ratio = (high - low) / close
            range_volume = range_ratio * volume_ratio_medium
            manual_features['range_volume_liquidity'] = range_volume
            
            # Range-liquidity regime
            range_regime = (range_ratio > range_ratio.rolling(50).mean()).astype(int)
            manual_features['range_liquidity_regime'] = range_regime
            
            # 4. Liquidity persistence features
            liquidity_persistence_short = (volume_ratio_short > 1).rolling(5).sum()
            liquidity_persistence_medium = (volume_ratio_medium > 1).rolling(10).sum()
            manual_features['liquidity_persistence_short'] = liquidity_persistence_short
            manual_features['liquidity_persistence_medium'] = liquidity_persistence_medium
            
            # Liquidity momentum
            liquidity_momentum = volume_ratio_medium.diff().rolling(5).mean()
            manual_features['liquidity_momentum'] = liquidity_momentum
            
            # 5. Enhanced liquidity volatility interaction
            vol_liquidity_regime = (liquidity_adjusted_vol > liquidity_adjusted_vol.rolling(100).mean()).astype(int)
            manual_features['vol_liquidity_regime'] = vol_liquidity_regime
            
            # Liquidity volatility strength
            liq_vol_strength = abs(liquidity_adjusted_vol)
            manual_features['liquidity_vol_strength'] = liq_vol_strength
            
            # 6. Microstructure liquidity features
            # Price impact estimation
            price_impact = abs(returns) / (volume_ratio_medium + 1e-8)
            manual_features['price_impact'] = price_impact
            
            # Liquidity depth proxy
            depth_proxy = volume / (range_ratio + 1e-8)
            manual_features['liquidity_depth'] = depth_proxy
            
            # Market efficiency indicator
            efficiency = abs(returns.rolling(10).mean()) / (volume_ratio_medium + 1e-8)
            manual_features['market_efficiency'] = efficiency
            
            # 7. Liquidity regime classification
            # High liquidity regime
            high_liquidity = (volume_ratio_medium > 1.5).astype(int)
            manual_features['high_liquidity_regime'] = high_liquidity
            
            # Low liquidity regime
            low_liquidity = (volume_ratio_medium < 0.5).astype(int)
            manual_features['low_liquidity_regime'] = low_liquidity
            
            # Liquidity stress indicator
            liquidity_stress = np.where(volume_ratio_medium < 0.3, 2, np.where(volume_ratio_medium > 2, 0, 1))
            manual_features['liquidity_stress'] = liquidity_stress
            
        return manual_features
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute enhanced liquidity regime step with AFML hardening."""
        return await self.execute_standard_specialist_logic(
            config=config,
            specialist_type=SpecialistType.LIQUIDITY_REGIME, # Assuming this exists or falls back
            manual_feature_func=self._get_liquidity_combined_manual_features,
            filter_type='spread',
            pt_sl_config_key='liquidity_pt_sl',
            default_pt_sl=[1.5, 1.5],
            suffix="enhanced_liquidity_regime_features"
        )
    
    def _load_market_data_with_cache(self, config: Dict[str, Any], timeframe: str) -> Tuple[pd.DataFrame, str]:
        """Load market data with caching."""
        market_data, market_source = self.load_market_data_or_fail(
            {**config, "timeframe": timeframe},
            pipeline_state={},
            allow_config_override=True,
        )
        return market_data, market_source
