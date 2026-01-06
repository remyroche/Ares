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

    def _generate_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate enhanced volatility burst features with manual feature engineering."""
        # Basic momentum features
        momentum_features = self._compute_enhanced_momentum_features(df)
        
        # Enhanced features from pipeline
        enhanced_features = self.feature_pipeline.generate_enhanced_features(
            df, 'volatility_burst', {'enhanced_features': True}
        )
        
        # Manual feature engineering for volatility burst
        manual_features = self._create_manual_volatility_burst_enhanced_features(df, enhanced_features)
        
        # Combine all features
        all_features = [momentum_features, enhanced_features, manual_features]
        
        # Combine all features with manual redundancy reduction
        if all_features:
            combined_features = pd.concat(all_features, axis=1)
            
            # Manual redundancy reduction and feature selection
            combined_features = self._apply_manual_volatility_burst_feature_selection(combined_features)
            
            # Remove duplicates and clean
            combined_features = combined_features.loc[:, ~combined_features.columns.duplicated()]
            combined_features = combined_features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            
            return combined_features
        
        return pd.DataFrame(index=df.index)
    
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
    
    def _apply_manual_volatility_burst_feature_selection(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply manual feature selection for volatility burst features."""
        if features.empty:
            return features
        
        # Remove constant features
        constant_features = features.columns[features.nunique() <= 1]
        if len(constant_features) > 0:
            features = features.drop(columns=constant_features)
            self.logger.info(f"Removed {len(constant_features)} constant volatility burst features")
        
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
            self.logger.info(f"Removed {len(set(to_drop))} redundant volatility burst features: {list(set(to_drop))}")
        
        # Keep only the most informative features (limit to top 30 by variance)
        if len(features.columns) > 30:
            feature_variances = features.var()
            top_features = feature_variances.nlargest(30).index
            features = features[top_features]
            self.logger.info(f"Limited volatility burst features to top 30 by variance")
        
        return features
    
    def _create_volatility_burst_labels(self, df: pd.DataFrame, lookforward: int = 35) -> pd.Series:
        """Create true volatility burst labels targeting volatility expansion."""
        returns = df['close'].pct_change()
        
        # 1. Realized volatility (rolling std)
        current_vol = returns.rolling(25).std()
        
        # 2. Future realized volatility over the lookforward window
        # We look at the standard deviation of returns in the next 'lookforward' periods
        future_vol = returns.shift(-lookforward).rolling(lookforward).std()
        
        # 3. Future absolute returns (max price move)
        future_abs_return = returns.shift(-lookforward).rolling(lookforward).apply(lambda x: np.abs(x).max(), raw=True)
        
        # Binary label: Volatility expansion OR price shock
        # Expansion: Future vol is 50% higher than current vol
        # Shock: Future move is > 2 standard deviations
        vol_expansion = future_vol > current_vol * 1.5
        price_shock = future_abs_return > current_vol * 2.5
        
        labels = (vol_expansion | price_shock).astype(int)
        
        return labels
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute enhanced volatility burst step."""
        start_time = time.time()
        try:
            symbol = str(config.get("symbol", "ETHUSDT"))
            exchange = str(config.get("exchange", "binance"))
            timeframe = str(config.get("timeframe", "15m"))
            direction = str(config.get("direction", "long"))

            self.set_context(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                direction=direction,
                model=self.step_name
            )
            
            self._versioned_store = None
            _ = self.versioned_store
            
            # 1. Load market data
            market_data, market_source = self.load_market_data_or_fail(
                {**config, "timeframe": timeframe},
                pipeline_state={},
                allow_config_override=True,
            )
            
            # 2. Generate Enhanced Features
            tprint_info("🛠️ Generating enhanced volatility features...")
            feature_df = self._generate_enhanced_features(market_data)
            
            # 3-4. AFML: Sampling, Labeling, Weighting, Alignment via Helper
            X, y, weights = self.prepare_specialist_data(
                market_data=market_data,
                feature_df=feature_df,
                config=config,
                filter_type='volatility',
                pt_sl_config_key='volatility_burst_pt_sl',
                default_pt_sl=[3.0, 1.0]
            )

            # 4. Train Enhanced Model with Centralized XGB Trainer
            tprint_info("🤖 Training Enhanced Volatility Burst model with centralized XGB helper (purged CV & AFML weights)...")
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

            metrics.update({
                'n_features': len(X.columns),
                'n_samples': len(X)
            })

            # 5. Generate Final Standardized Output
            final_probs = pd.Series(np.nan, index=market_data.index)
            if len(valid_oof) > 0:
                final_probs.loc[valid_oof.index] = y_full_pred_prob
            
            final_probs = final_probs.ffill().fillna(0.5)
            final_preds = (final_probs >= 0.5).astype(int)
            
            full_labels = pd.Series(0, index=market_data.index)
            full_labels.loc[y.index] = y

            output_df = self._create_standardized_output(
                feature_df, full_labels, final_preds.values, final_probs.values, symbol, exchange, timeframe, direction
            )
            
            artifact_name = f"enhanced_volatility_burst_prediction_{timeframe}"
            metadata = SpecialistDataInterface.create_standard_metadata(
                specialist_name="EnhancedMLVolatilityBurstStep",
                config=config,
                metrics=metrics,
                mi_score=metrics.get('mi_score', 0.0),
                hsic_score=0.0
            )
            
            self._save_artifact(data=output_df, artifact_name=artifact_name, artifact_type="data", data_category="predictions", metadata=metadata)
            
            try:
                if self.versioned_store:
                    self.versioned_store.add_data(output_df, version_name=artifact_name)
                    tprint_success(f"💾 Saved predictions to versioned store as '{artifact_name}'")
            except Exception as ve:
                tprint_warning(f"Versioned store save failed: {ve}")
            
            self._save_artifact(data=last_model, artifact_name=f"enhanced_volatility_model_{timeframe}", artifact_type="model", data_category="models", metadata=metadata)
            
            # 6. Run Enhanced Diagnostics
            tprint_info("🔍 Running Enhanced Diagnostics...")
            diagnostics_result = self.run_enhanced_diagnostics(symbol, exchange, timeframe, direction)
            
            tprint_success(f"✅ Enhanced Volatility Burst completed in {time.time()-start_time:.2f}s")
            return {
                "success": True,
                "metrics": metrics,
                "n_samples": len(X),
                "artifact_name": artifact_name,
                "diagnostics": diagnostics_result
            }
            
        except Exception as e:
            self.logger.exception(f"❌ Enhanced volatility burst step failed: {e}")
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
        """Load market data - placeholder implementation."""
        # This would be implemented based on the actual data loading mechanism
        # Using alternative data loading approach
            # market_data = self._load_alternative_market_data(config, timeframe)
        market_data, _market_source = self.load_market_data_or_fail(
            {**config, "timeframe": timeframe},
            pipeline_state={},
            allow_config_override=True,
        )
        return market_data
