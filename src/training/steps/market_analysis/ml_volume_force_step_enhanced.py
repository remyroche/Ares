"""
Enhanced ML Volume Force Step with MI Improvements & Standardization

This enhanced version implements:
- Enhanced feature generation for MI improvement
- Data structure standardization
- Binary output enforcement
- MI monitoring and optimization
- Ensemble compatibility
"""

import logging
import time
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
)

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
from src.feature_generation.categories.enhanced_volume_force_features import (
    generate_enhanced_volume_force_features,
)
from src.training.steps.market_analysis.specialist_diagnostics_mixin_enhanced_v2 import (
    SpecialistDiagnosticsMixinEnhancedV2,
)
from src.training.steps.market_analysis.afml_specialist_mixin import AFMLSpecialistMixin
from src.training.steps.market_analysis.enhanced_feature_generators import MIOptimizedFeaturePipeline
from src.training.steps.market_analysis.specialist_interface import SpecialistDataInterface
from src.training.steps.market_analysis.specialist_data_standard import SpecialistType
from src.utils.ml_common.specialist_xgb import train_specialist_model_with_oof
from src.utils.versioned_artifacts import VersionedArtifactStore

logger = logging.getLogger(__name__)


class EnhancedMLVolumeForceStep(SpecialistDiagnosticsMixinEnhancedV2, AFMLSpecialistMixin, BaseStep):

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

    """Enhanced Volume Force step with MI improvements and standardization."""

    def __init__(self, step_name: str = "enhanced_ml_volume_force_step"):
        """Initialize the enhanced specialist."""
        BaseStep.__init__(self, step_name, use_versioned_artifacts=True)
        SpecialistDiagnosticsMixinEnhancedV2.__init__(self, step_name=step_name)
        self._current_context = {}
        self._artifact_manager = None
        self._versioned_store = None
        self.step_name = step_name
        self.logger = logger.getChild("EnhancedMLVolumeForceStep")
        self.feature_pipeline = MIOptimizedFeaturePipeline()
        self.mi_history = []
        self.training_metrics = []
        self._market_data_cache = {}
        self._feature_cache = {}
        tprint(f"✅ Initialized Enhanced {step_name} (MI-Optimized)", "SUCCESS")
    
    @property
    def versioned_store(self):
        """Specialized versioned store aligned to repo-root versioned_artifacts."""
        if self._versioned_store is None and self.use_versioned_artifacts:
            symbol = self._current_context.get("symbol", "UNKNOWN")
            exchange = self._current_context.get("exchange", "binance")
            timeframe = self._current_context.get("timeframe", "15m")
            direction = self._current_context.get("direction", "long")
            model = self.step_name

            store_name = f"{symbol}_{exchange}_{timeframe}_{direction}_{model}"
            repo_root = Path(__file__).resolve().parents[4]
            store_root = repo_root / "versioned_artifacts"
            store_root.mkdir(parents=True, exist_ok=True)
            store_path = store_root / store_name

            self._versioned_store = VersionedArtifactStore(
                store_path=str(store_path),
                auto_version=True,
                enable_row_versioning=True,
            )

            if hasattr(self._versioned_store, "_metadata"):
                self._versioned_store._metadata["context"] = {
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe,
                    "direction": direction,
                    "model": model,
                }
                self._versioned_store._save_metadata()

        return self._versioned_store

    def _get_volume_combined_manual_features(self, df: pd.DataFrame, pipeline_features: pd.DataFrame) -> pd.DataFrame:
        """Combine base, pipeline, and manual volume force features."""
        # 1. Base Volume Force Features
        base_features = generate_enhanced_volume_force_features(df, {})
        
        # 2. Manual Features
        manual_features = self._create_manual_volume_force_enhanced_features(df, pipeline_features)
        
        return pd.concat([base_features, manual_features], axis=1)

    def generate_pipeline_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Generate base pipeline features."""
        # For volume force, we use the combined manual features as pipeline features
        # when called from external consumers (like GMM pipeline)
        return self._get_volume_combined_manual_features(market_data, pd.DataFrame(index=market_data.index))

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute enhanced volume force step with MI optimization."""
        return await self.execute_standard_specialist_logic(
            config=config,
            specialist_type=SpecialistType.VOLUME_FORCE,
            manual_feature_func=self._get_volume_combined_manual_features,
            filter_type='volume',
            pt_sl_config_key='volume_force_pt_sl',
            default_pt_sl=[1.5, 1.5],
            suffix="enhanced_volume_force_features"
        )

    def _create_manual_volume_force_enhanced_features(self, df: pd.DataFrame, enhanced_features: pd.DataFrame) -> pd.DataFrame:
        """Create manual enhanced features for volume force detection."""
        manual_features = pd.DataFrame(index=df.index)
        
        if all(col in df.columns for col in ['close', 'high', 'low', 'volume']):
            close = df['close']
            volume = df.get('volume', pd.Series(1, index=df.index))
            returns = close.pct_change()
            
            log_volume = np.log1p(volume)
            try:
                from src.utils.ml_common.afml_utils import frac_diff_fixed
                fd_vol = frac_diff_fixed(log_volume, d=0.45)
                manual_features['frac_diff_log_volume_norm'] = (fd_vol - fd_vol.rolling(100).mean()) / (fd_vol.rolling(100).std() + 1e-8)
            except Exception as e:
                self.logger.warning(f"Fractional differentiation failed, using price returns: {e}")
                # Use price returns as default fallback per user preference
                manual_features['frac_diff_log_volume_norm'] = returns.rolling(20).mean()
            
            volume_ma_long = volume.rolling(200).mean()
            manual_features['volume_impulse_ratio'] = (volume - volume.rolling(5).mean()) / (volume_ma_long + 1e-8)
            manual_features['order_flow_pressure'] = (returns * volume).rolling(10).sum() / (volume.rolling(10).sum() + 1e-8)
            
        return manual_features

    def _load_market_data_with_cache(self, config: Dict[str, Any], timeframe: str) -> Tuple[pd.DataFrame, str]:
        """Load market data with caching."""
        market_data, market_source = self.load_market_data_or_fail(
            {**config, "timeframe": timeframe},
            pipeline_state={},
            allow_config_override=True,
        )
        return market_data, market_source
