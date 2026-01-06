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
from src.utils.ml_common.standardized_xgb_trainer import (
    StandardizedXGBTrainer,
    XGBTrainingConfig,
)
from src.training.steps.market_analysis.specialist_diagnostics_mixin_enhanced_v2 import (
    SpecialistDiagnosticsMixinEnhancedV2
)
from src.training.steps.market_analysis.afml_specialist_mixin import AFMLSpecialistMixin
from src.training.steps.market_analysis.enhanced_feature_generators import MIOptimizedFeaturePipeline
from src.training.steps.market_analysis.specialist_interface import SpecialistDataInterface
from src.training.steps.market_analysis.specialist_data_standard import SpecialistType
from src.utils.ml_common.specialist_xgb import train_specialist_xgb_with_oof
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
    
    def _generate_enhanced_liquidity_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Generate enhanced liquidity features with manual feature engineering."""
        # Import original liquidity features
        from src.feature_generation.categories.liquidity_regime_features import generate_liquidity_regime_features
        base_liquidity_features = generate_liquidity_regime_features(df, config)
        
        # Enhanced features from pipeline
        enhanced_features = self.feature_pipeline.generate_enhanced_features(
            df, 'liquidity_regime', {'enhanced_features': True}
        )
        
        # Manual feature engineering for liquidity regime
        manual_features = self._create_manual_liquidity_enhanced_features(df, enhanced_features)
        
        # Combine all features
        all_features = [base_liquidity_features, enhanced_features, manual_features]
        
        # Combine all features with manual redundancy reduction
        if all_features:
            combined_features = pd.concat(all_features, axis=1)
            
            # Manual redundancy reduction and feature selection
            combined_features = self._apply_manual_liquidity_feature_selection(combined_features)
            
            # Remove duplicates and clean
            combined_features = combined_features.loc[:, ~combined_features.columns.duplicated()]
            combined_features = combined_features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            
            return combined_features
        
        return pd.DataFrame(index=df.index)
    
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
    
    def _apply_manual_liquidity_feature_selection(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply manual feature selection for liquidity regime features."""
        if features.empty:
            return features
        
        # Remove constant features
        constant_features = features.columns[features.nunique() <= 1]
        if len(constant_features) > 0:
            features = features.drop(columns=constant_features)
            self.logger.info(f"Removed {len(constant_features)} constant liquidity features")
        
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
            self.logger.info(f"Removed {len(set(to_drop))} redundant liquidity features: {list(set(to_drop))}")
        
        # Keep only the most informative features (limit to top 30 by variance)
        if len(features.columns) > 30:
            feature_variances = features.var()
            top_features = feature_variances.nlargest(30).index
            features = features[top_features]
            self.logger.info(f"Limited liquidity features to top 30 by variance")
        
        return features
    
    def _add_liquidity_specific_features(self, df: pd.DataFrame, liquidity_features: pd.DataFrame) -> pd.DataFrame:
        """Add liquidity-specific enhanced features."""
        features = pd.DataFrame(index=df.index)
        
        # Enhanced volume analysis
        if 'volume' in df.columns and 'close' in df.columns:
            volume = df['volume']
            price_change = df['close'].pct_change()
            
            # Volume-price relationship enhancements
            features['volume_price_correlation_10'] = price_change.rolling(15).corr(volume)
            features['volume_price_correlation_20'] = price_change.rolling(25).corr(volume)
            features['volume_price_correlation_50'] = price_change.rolling(60).corr(volume)
            
            # Volume pattern recognition
            volume_ma = volume.rolling(25).mean()
            features['volume_pattern_accumulation'] = (volume > volume_ma * 1.5).astype(int)
            features['volume_pattern_distribution'] = (volume < volume_ma * 0.5).astype(int)
            features['volume_pattern_churning'] = ((volume >= volume_ma * 0.5) & (volume <= volume_ma * 1.5)).astype(int)
            
            # Volume efficiency metrics
            features['volume_efficiency_ratio'] = price_change.abs() / (volume + 1e-8)
            features['volume_efficiency_ma'] = features['volume_efficiency_ratio'].rolling(25).mean()
            
            # Volume momentum
            volume_change = volume.pct_change()
            features['volume_momentum_10'] = volume_change.rolling(15).sum()
            features['volume_momentum_20'] = volume_change.rolling(25).sum()
            features['volume_acceleration'] = volume_change.rolling(15).sum() - volume_change.rolling(25).sum()
        
        # Enhanced price analysis
        if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
            high_low_range = df['high'] - df['low']
            close_price = df['close']
            
            # Range analysis
            range_ma = high_low_range.rolling(25).mean()
            features['range_expansion'] = high_low_range / range_ma
            features['range_contraction'] = (high_low_range < range_ma * 0.7).astype(int)
            features['range_breakout_up'] = (high_low_range > high_low_range.rolling(25).max().shift(1)).astype(int)
            features['range_breakout_down'] = (high_low_range < high_low_range.rolling(25).min().shift(1)).astype(int)
            
            # Price efficiency
            mid_price = (df['high'] + df['low']) / 2
            features['price_efficiency'] = (close_price - mid_price) / mid_price
            features['price_efficiency_ma'] = features['price_efficiency'].rolling(25).mean()
            
            # Support/resistance levels
            for window in [20, 50]:
                rolling_max = close_price.rolling(window).max()
                rolling_min = close_price.rolling(window).min()
                
                features[f'distance_to_resistance_{window}'] = (rolling_max - close_price) / rolling_max
                features[f'distance_to_support_{window}'] = (close_price - rolling_min) / rolling_max
                features[f'sr_strength_{window}'] = (rolling_max - rolling_min) / close_price
        
        # Market microstructure features
        if 'volume' in df.columns and 'close' in df.columns:
            # Volume profile analysis
            volume_ma = df['volume'].rolling(25).mean()
            price_change = df['close'].pct_change()
            
            features['volume_anomaly'] = df['volume'] / volume_ma
            features['volume_price_trend'] = (price_change * df['volume']).rolling(15).sum()
            
            # Order flow imbalance proxy
            features['order_flow_proxy'] = (price_change * df['volume']).rolling(15).sum()
            features['order_flow_persistence'] = (features['order_flow_proxy'] > 0).rolling(25).sum()
        
        return features
    
    def _create_liquidity_labels(self, df: pd.DataFrame, lookforward: int = 35) -> pd.Series:
        """Create liquidity regime labels based on volume and price patterns."""
        if 'volume' not in df.columns or 'close' not in df.columns:
            # Fallback to simple return-based labels
            returns = df['close'].pct_change()
            volume_change = df['volume'].pct_change()
            
            # Liquidity stress indicator
            liquidity_stress = returns.rolling(25).std() * volume_change.rolling(25).std()
            future_stress = liquidity_stress.shift(-lookforward)
            
            labels = (future_stress > liquidity_stress.quantile(0.8)).astype(int)
            return labels
        
        # Liquidity-specific labeling
        volume = df['volume']
        close_price = df['close']
        
        # Volume patterns
        volume_ma = volume.rolling(25).mean()
        volume_anomaly = volume / volume_ma
        
        # Price patterns
        price_change = close_price.pct_change()
        price_volatility = price_change.rolling(25).std()
        
        # Liquidity stress indicator
        liquidity_stress = price_volatility * volume_anomaly
        
        # Future liquidity stress
        future_stress = liquidity_stress.shift(-lookforward)
        
        # Label: positive if liquidity stress increases (potential regime change)
        labels = (future_stress > liquidity_stress.quantile(0.75)).astype(int)
        
        return labels
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute enhanced liquidity regime step with AFML hardening."""
        start_time = time.time()
        metrics: Dict[str, Any] = {}
        artifacts: List[str] = []

        try:
            symbol = str(config.get("symbol", "ETHUSDT"))
            exchange = str(config.get("exchange", "binance"))
            timeframe = str(config.get("timeframe", "15m"))
            direction = str(config.get("direction", "long"))

            if not symbol or not exchange:
                raise ValueError("Config must include 'symbol' and 'exchange'")

            self.set_context(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                direction=direction,
                model=self.step_name,
            )
            # Reset versioned store per run to ensure correct context/path
            self._versioned_store = None
            # Ensure versioned store is initialized with correct context
            _ = self.versioned_store

            # 1. Load Market Data
            market_data, market_source = self._load_market_data_with_cache(config, timeframe)

            # 2. Generate Enhanced Features
            tprint_info("🛠️ Generating Enhanced Liquidity features...")
            feature_df = self._generate_enhanced_liquidity_features(market_data, config)
            
            # 3-5. AFML: Sampling, Labeling, Weighting, Alignment via Helper
            X, y, weights = self.prepare_specialist_data(
                market_data=market_data,
                feature_df=feature_df,
                config=config,
                filter_type='spread',
                pt_sl_config_key='liquidity_pt_sl',
                default_pt_sl=[1.5, 1.5]
            )

            # 6. Centralized purged-CV training
            tprint_info("🤖 Training Enhanced Liquidity model with centralized XGB helper (purged CV & AFML weights)...")
            training_result = train_specialist_xgb_with_oof(
                X.fillna(0.0),
                y.fillna(0.0),
                sample_weight=weights,
                n_splits=5,
            )

            oof_probs = training_result.oof_predictions
            last_model = training_result.model
            
            metrics = training_result.metrics.copy()
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
                'n_samples': len(X)
            })

            # 5. Generate Final Standardized Output (Aligned to full market_data index)
            # AFML FIX: Initialize with NaN instead of 0.5 to allow proper ffilling downstream
            final_probs = pd.Series(np.nan, index=market_data.index if 'market_data' in locals() else (df.index if 'df' in locals() else X.index))
            if len(valid_oof) > 0:
                final_probs.loc[valid_oof.index] = y_full_pred_prob
            
            # Ffill probabilities so the signal is persistent between events
            final_probs = final_probs.ffill().fillna(0.5)
            final_preds = (final_probs >= 0.5).astype(int)
            
            full_labels = pd.Series(0, index=market_data.index if 'market_data' in locals() else (df.index if 'df' in locals() else X.index))
            full_labels.loc[y.index] = y

            standardized_output = self._create_standardized_output(
                feature_df if 'feature_df' in locals() else (features_df if 'features_df' in locals() else X), 
                full_labels, final_preds.values, final_probs.values, symbol, exchange, timeframe, direction
            )

            # 9. Save Artifacts
            artifact_name = f"enhanced_ml_liquidity_predictions_{timeframe}"
            metadata = SpecialistDataInterface.create_standard_metadata(
                specialist_name="EnhancedMLLiquidityRegimeStep",
                config=config,
                metrics=metrics,
                mi_score=metrics['mi_score'],
                hsic_score=0.0
            )
            
            artifact_path = self._save_artifact(
                data=standardized_output,
                artifact_name=artifact_name,
                artifact_type="data",
                data_category="predictions",
                metadata=metadata
            )
            artifacts.append(artifact_path)

            # 10. Run Enhanced Diagnostics
            tprint_info("🔍 Running Enhanced Diagnostics...")
            diagnostics_result = self.run_enhanced_diagnostics(symbol, exchange, timeframe, direction)
            
            if diagnostics_result.get('success', False):
                compliance_report = diagnostics_result['compliance_report']
                ensemble_compatibility = diagnostics_result['ensemble_compatibility']
                
                tprint_success(f"✅ Enhanced Diagnostics Complete:")
                tprint_info(f"   MI Score: {compliance_report['metrics']['mi_score']:.4f}")
                tprint_info(f"   Requirements Met: {compliance_report['requirements_met']}/3")
                tprint_info(f"   Ensemble Ready: {ensemble_compatibility['ensemble_ready']}")
                
                metrics.update({
                    'enhanced_mi_score': compliance_report['metrics']['mi_score'],
                    'enhanced_requirements_met': compliance_report['requirements_met'],
                    'enhanced_ensemble_ready': ensemble_compatibility['ensemble_ready']
                })

            # 11. Final Summary
            execution_time = time.time() - start_time
            metrics["execution_time"] = execution_time
            metrics["n_samples"] = len(X)

            tprint_success(f"✅ Enhanced Liquidity Regime completed in {execution_time:.2f}s")
            tprint_info(f"📊 Final Metrics: MI={metrics.get('mi_score', 0):.4f}, AUC={metrics.get('auc', 0):.3f}")

            return {
                "success": True,
                "metrics": metrics,
                "n_samples": len(X),
                "features": list(X.columns),
                "artifacts": artifacts,
                "diagnostics": diagnostics_result,
                "mi_history": self.mi_history,
                "training_metrics": self.training_metrics,
                "execution_time": execution_time
            }

        except Exception as e:
            self.logger.exception(f"❌ Enhanced Liquidity Regime step failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _create_standardized_output(self, features: pd.DataFrame, labels: pd.Series,
                                  predictions: np.ndarray, probabilities: np.ndarray,
                                  symbol: str, exchange: str, timeframe: str, direction: str) -> pd.DataFrame:
        """Create standardized output structure."""
        standardized = pd.DataFrame(index=features.index)
        standardized['timestamp'] = features.index
        standardized['specialist_prediction'] = predictions
        standardized['specialist_probability'] = probabilities
        standardized['target_label'] = labels
        
        # Add original features for reference
        for col in features.columns[:20]:  # Limit to first 20 features
            standardized[f'feature_{col}'] = features[col]
        
        return standardized
    
    def _load_market_data_with_cache(self, config: Dict[str, Any], timeframe: str) -> Tuple[pd.DataFrame, str]:
        """Load market data with caching."""
        market_data, market_source = self.load_market_data_or_fail(
            {**config, "timeframe": timeframe},
            pipeline_state={},
            allow_config_override=True,
        )
        return market_data, market_source
