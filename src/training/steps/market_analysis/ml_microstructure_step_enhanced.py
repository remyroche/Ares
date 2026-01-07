"""
Enhanced ML Microstructure Step with MI Improvements

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

from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.versioned_artifacts import VersionedArtifactStore
from src.training.steps.market_analysis.ml_risk_regime_step import MLRiskRegimeStepHMM
from src.training.steps.market_analysis.specialist_diagnostics_mixin_enhanced_v2 import (
    SpecialistDiagnosticsMixinEnhancedV2
)
from src.training.steps.market_analysis.specialist_diagnostics_mixin import SpecialistDiagnosticsMixin
from src.training.steps.market_analysis.afml_specialist_mixin import AFMLSpecialistMixin
from src.training.steps.market_analysis.enhanced_feature_generators import MIOptimizedFeaturePipeline
from src.training.steps.market_analysis.specialist_interface import SpecialistDataInterface
from src.training.steps.market_analysis.specialist_data_standard import SpecialistType
from src.utils.ml_common.specialist_xgb import train_specialist_model_with_oof

logger = logging.getLogger(__name__)


class EnhancedMLMicrostructureStep(MLRiskRegimeStepHMM, SpecialistDiagnosticsMixinEnhancedV2, AFMLSpecialistMixin, SpecialistDiagnosticsMixin):

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
            model = 'enhanced_ml_microstructure_step'  # Use the correct model name

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
    Enhanced Microstructure Specialist with MI optimization.
    
    Key enhancements:
    - Enhanced feature generation for MI improvement
    - Real-time MI monitoring during training
    - Hyperparameter optimization for MI > 0.02
    - Standardized data structure output
    - Binary output enforcement
    """
    
    def __init__(self, step_name: str = "enhanced_ml_microstructure_step"):
        """Initialize the enhanced microstructure step."""
        super().__init__(step_name=step_name)  # Parent class already enables versioned artifacts
        self._current_context = {}
        self._artifact_manager = None
        self._versioned_store = None
        self.step_name = step_name
        self.logger = logger.getChild("EnhancedMLMicrostructureStep")
        self.feature_pipeline = MIOptimizedFeaturePipeline()
        self.mi_history = []
        self.training_metrics = []
        
    def _compute_enhanced_structural_optimized_horizon_optimized_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
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
            features[f'price_microstructure_{window}'] = (df['close'] - sma) / sma
            features[f'price_microstructure_cross_{window}'] = (df['close'] > sma).astype(int)
        
        # Volume momentum
        if 'volume' in df.columns:
            volume_change = df['volume'].pct_change()
            for window in [5, 10, 20]:
                features[f'volume_microstructure_{window}'] = volume_change.rolling(window).sum()
                features[f'volume_price_microstructure_{window}'] = (returns * volume_change).rolling(window).sum()
        
        return features
    
    def _get_micro_combined_manual_features(self, df: pd.DataFrame, pipeline_features: pd.DataFrame) -> pd.DataFrame:
        """Combine microstructure features and manual enhancements."""
        # 1. Base Microstructure Features
        micro_features = self._compute_enhanced_structural_optimized_horizon_optimized_microstructure_features(df)
        
        # 2. Manual Features
        manual_features = self._create_manual_microstructure_enhanced_features(df, pipeline_features)
        
        return pd.concat([micro_features, manual_features], axis=1)

    def _create_manual_microstructure_enhanced_features(self, df: pd.DataFrame, enhanced_features: pd.DataFrame) -> pd.DataFrame:
        """Create manual enhanced features for microstructure analysis (optimized)."""
        manual_features = pd.DataFrame(index=df.index)
        
        if all(col in df.columns for col in ['close', 'high', 'low', 'volume']):
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df.get('volume', pd.Series(1, index=df.index))
            returns = close.pct_change()
            
            # Simplified microstructure features (avoid heavy computations)
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
            
            # Microstructure-specific features
            # Price efficiency
            manual_features['price_efficiency'] = abs(returns.rolling(10).sum())
            
            # Volume efficiency
            manual_features['volume_efficiency'] = abs(volume.pct_change().rolling(10).sum())
            
            # Spread features
            manual_features['spread'] = (high - low) / close
            manual_features['spread_ma'] = manual_features['spread'].rolling(20).mean()
        
        return manual_features
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute enhanced microstructure step."""
        return await self.execute_standard_specialist_logic(
            config=config,
            specialist_type=SpecialistType.MICROSTRUCTURE,
            manual_feature_func=self._get_micro_combined_manual_features,
            filter_type='spread',
            pt_sl_config_key='microstructure_pt_sl',
            default_pt_sl=[2.0, 1.0],
            suffix="enhanced_microstructure_features"
        )
    
    def _load_market_data(self, symbol: str, exchange: str, timeframe: str) -> pd.DataFrame:
        """Load market data using BaseStep method."""
        market_data, _ = self.load_market_data_or_fail(
            {"symbol": symbol, "exchange": exchange, "timeframe": timeframe},
            pipeline_state={},
            allow_config_override=True,
        )
        return market_data
