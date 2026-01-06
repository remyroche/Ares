"""
Enhanced ML Risk Regime Step with MI Improvements

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
from src.utils.ml_common.afml_utils import (
    get_daily_vol, get_t_events, get_vertical_barrier, 
    apply_triple_barrier, get_bins, get_weights_by_uniqueness,
    frac_diff_fixed, get_sample_weights
)
from src.utils.tprint import (
    tprint,
    tprint_info,
    tprint_warning,
    tprint_error,
    tprint_success,
)
from src.training.steps.market_analysis.specialist_diagnostics_mixin_enhanced_v2 import (
    SpecialistDiagnosticsMixinEnhancedV2
)
from src.training.steps.market_analysis.afml_specialist_mixin import AFMLSpecialistMixin
from src.training.steps.market_analysis.enhanced_feature_generators import MIOptimizedFeaturePipeline
from src.training.steps.market_analysis.specialist_interface import SpecialistDataInterface
from src.training.steps.market_analysis.specialist_data_standard import SpecialistType
from src.utils.ml_common.specialist_xgb import train_specialist_xgb_with_oof

logger = logging.getLogger(__name__)


class EnhancedMLRiskRegimeStep(SpecialistDiagnosticsMixinEnhancedV2, AFMLSpecialistMixin, BaseStep):

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
    Enhanced Risk Regime Specialist with MI optimization.
    
    Key enhancements:
    - Enhanced feature generation for MI improvement
    - Real-time MI monitoring during training
    - Risk-specific feature engineering
    - Hyperparameter optimization for MI > 0.02
    - Data structure standardization
    - Binary output enforcement
    """
    
    def __init__(self, step_name: str = "enhanced_ml_risk_regime_step"):
        """Initialize the enhanced specialist."""
        BaseStep.__init__(self, step_name, use_versioned_artifacts=True)
        SpecialistDiagnosticsMixinEnhancedV2.__init__(self, step_name=step_name)
        self._current_context = {}
        self._artifact_manager = None
        self._versioned_store = None
        self.step_name = step_name
        self.logger = logger.getChild("EnhancedMLRiskRegimeStep")
        self.feature_pipeline = MIOptimizedFeaturePipeline()
        self.mi_history = []
        self.training_metrics = []
        self._market_data_cache = {}
        tprint(f"✅ Initialized Enhanced {step_name} (MI-Optimized)", "SUCCESS")
    
    def _generate_enhanced_risk_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Generate enhanced risk features with manual feature engineering."""
        # Import original risk features
        from src.feature_generation.categories.risk_regime_features import generate_risk_regime_features
        base_risk_features = generate_risk_regime_features(df, config)
        
        # Enhanced features from pipeline
        enhanced_features = self.feature_pipeline.generate_enhanced_features(
            df, 'risk_regime', {'enhanced_features': True}
        )
        
        # Manual feature engineering for risk regime
        manual_features = self._create_manual_risk_enhanced_features(df, enhanced_features)
        
        # Combine all features
        all_features = [base_risk_features, enhanced_features, manual_features]
        
        # Combine all features with manual redundancy reduction
        if all_features:
            combined_features = pd.concat(all_features, axis=1)
            
            # Manual redundancy reduction and feature selection
            combined_features = self._apply_manual_risk_feature_selection(combined_features)
            
            # Remove duplicates and clean
            combined_features = combined_features.loc[:, ~combined_features.columns.duplicated()]
            combined_features = combined_features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            
            return combined_features
        
        return pd.DataFrame(index=df.index)
    
    def _apply_manual_risk_feature_selection(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply manual feature selection for risk regime features."""
        if features.empty:
            return features
        
        # Remove constant features
        constant_features = features.columns[features.nunique() <= 1]
        if len(constant_features) > 0:
            features = features.drop(columns=constant_features)
            self.logger.info(f"Removed {len(constant_features)} constant risk features")
        
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
            self.logger.info(f"Removed {len(set(to_drop))} redundant risk features: {list(set(to_drop))}")
        
        # Keep only the most informative features (limit to top 30 by variance)
        if len(features.columns) > 30:
            feature_variances = features.var()
            top_features = feature_variances.nlargest(30).index
            features = features[top_features]
            self.logger.info(f"Limited risk features to top 30 by variance")
        
        return features
    
    def _create_manual_risk_enhanced_features(self, df: pd.DataFrame, enhanced_features: pd.DataFrame) -> pd.DataFrame:
        """Create advanced manual enhanced features for risk regime detection."""
        manual_features = pd.DataFrame(index=df.index)
        
        if all(col in df.columns for col in ['close', 'high', 'low', 'volume']):
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df.get('volume', pd.Series(1, index=df.index))
            returns = close.pct_change()
            
            # 1. Tail Risk Measures: VaR and CVaR at multiple confidence intervals
            for window in [20, 50]:
                # VaR at 95% and 99% confidence intervals
                var_95 = returns.rolling(window).quantile(0.05)
                var_99 = returns.rolling(window).quantile(0.01)
                manual_features[f'var_95_{window}d'] = var_95
                manual_features[f'var_99_{window}d'] = var_99
                
                # CVaR (Expected Shortfall) at 95% and 99%
                cvar_95 = returns[returns <= var_95].rolling(window).mean()
                cvar_99 = returns[returns <= var_99].rolling(window).mean()
                manual_features[f'cvar_95_{window}d'] = cvar_95
                manual_features[f'cvar_99_{window}d'] = cvar_99
                
                # Tail risk ratios
                manual_features[f'tail_risk_ratio_95_{window}d'] = cvar_95 / (var_95 + 1e-8)
                manual_features[f'tail_risk_ratio_99_{window}d'] = cvar_99 / (var_99 + 1e-8)
            
            # 2. Enhanced Drawdown Risk Features
            cum_returns = (1 + returns).cumprod()
            for window in [20, 50, 100]:
                rolling_max = cum_returns.rolling(window).max()
                drawdown = (cum_returns - rolling_max) / rolling_max
                
                # Maximum drawdown
                max_drawdown = drawdown.rolling(window).min()
                manual_features[f'max_drawdown_{window}d'] = max_drawdown
                
                # Current drawdown percentage
                manual_features[f'current_drawdown_{window}d'] = drawdown
                
                # Drawdown velocity (speed of decline)
                drawdown_velocity = drawdown.diff().rolling(5).mean()
                manual_features[f'drawdown_velocity_{window}d'] = drawdown_velocity
                
                # Drawdown duration (consecutive periods in drawdown)
                drawdown_duration = (drawdown < 0).astype(int).rolling(window).sum()
                manual_features[f'drawdown_duration_{window}d'] = drawdown_duration
                
                # Drawdown recovery time
                drawdown_recovery = (drawdown >= 0).astype(int).rolling(window).sum()
                manual_features[f'drawdown_recovery_{window}d'] = drawdown_recovery
            
            # 3. Volatility-Adjusted Risk Features
            for window in [20, 50]:
                volatility = returns.rolling(window).std()
                
                # Risk-adjusted returns
                risk_adj_returns = returns.rolling(window).mean() / (volatility + 1e-8)
                manual_features[f'risk_adj_returns_{window}d'] = risk_adj_returns
                
                # Sharpe ratio volatility
                sharpe_vol = volatility * np.sqrt(252)
                manual_features[f'sharpe_volatility_{window}d'] = sharpe_vol
                
                # Sortino ratio (downside risk adjusted)
                downside_returns = returns.copy()
                downside_returns[downside_returns > 0] = 0
                downside_vol = downside_returns.rolling(window).std()
                sortino_ratio = returns.rolling(window).mean() / (downside_vol + 1e-8)
                manual_features[f'sortino_ratio_{window}d'] = sortino_ratio
                
                # Volatility-adjusted VaR
                var_95_adj = var_95 / (volatility + 1e-8)
                manual_features[f'var_95_adj_{window}d'] = var_95_adj
            
            # 4. Risk Regime Classification
            for window in [20, 50]:
                volatility = returns.rolling(window).std()
                vol_ma = volatility.rolling(window*2).mean()
                vol_std = volatility.rolling(window*2).std()
                
                # High/low volatility regimes
                high_vol_regime = (volatility > vol_ma * 1.5).astype(int)
                low_vol_regime = (volatility < vol_ma * 0.5).astype(int)
                normal_vol_regime = ((volatility >= vol_ma * 0.5) & (volatility <= vol_ma * 1.5)).astype(int)
                
                manual_features[f'high_vol_regime_{window}d'] = high_vol_regime
                manual_features[f'low_vol_regime_{window}d'] = low_vol_regime
                manual_features[f'normal_vol_regime_{window}d'] = normal_vol_regime
                
                # Risk regime transitions
                vol_regime = np.where(volatility > vol_ma * 1.5, 2, np.where(volatility < vol_ma * 0.5, 0, 1))
                regime_changes = np.diff(vol_regime, prepend=vol_regime[0])
                manual_features[f'risk_regime_changes_{window}d'] = np.abs(regime_changes)
                
                # Risk regime persistence
                vol_regime_series = pd.Series(vol_regime, index=df.index)
                regime_persistence = (vol_regime_series == 1).rolling(10).sum()
                manual_features[f'risk_regime_persistence_{window}d'] = regime_persistence
            
            # 5. Enhanced multi-dimensional volatility features
            # Realized volatility across multiple timeframes
            vol_5 = returns.rolling(5).std()
            vol_10 = returns.rolling(10).std()
            vol_20 = returns.rolling(20).std()
            vol_50 = returns.rolling(50).std()
            
            # Volatility term structure
            vol_term_structure = vol_20 / (vol_50 + 1e-8)
            manual_features['vol_term_structure'] = vol_term_structure
            
            # Volatility momentum
            vol_momentum = vol_20.diff().rolling(5).mean()
            manual_features['vol_momentum'] = vol_momentum
            
            # Volatility acceleration
            vol_acceleration = vol_20.diff().diff()
            manual_features['vol_acceleration'] = vol_acceleration
            
            # 6. Advanced tail risk features
            manual_features['skewness_20'] = returns.rolling(20).skew()
            manual_features['kurtosis_20'] = returns.rolling(20).kurt()
            
            # Downside risk features
            downside_returns = returns.copy()
            downside_returns[downside_returns > 0] = 0
            manual_features['downside_deviation_20'] = downside_returns.rolling(20).std()
            
            # Semi-variance (downside risk focus)
            semi_variance = ((downside_returns) ** 2).rolling(20).mean()
            manual_features['semi_variance_20'] = semi_variance
            
            # 7. Volume-adjusted risk features
            volume_ma = volume.rolling(20).mean()
            volume_ratio = volume / (volume_ma + 1e-8)
            volume_weighted_vol = vol_20 * (1 + np.log(volume_ratio + 1))
            manual_features['volume_weighted_volatility'] = volume_weighted_vol
            
            # Volume-volatility divergence
            vol_regime = (vol_20 > vol_20.rolling(100).mean()).astype(int)
            volume_regime = (volume_ratio > 1).astype(int)
            volume_vol_divergence = np.abs(vol_regime - volume_regime)
            manual_features['volume_vol_divergence'] = volume_vol_divergence
            
            # 8. Price-based risk features
            range_ratio = (high - low) / close
            range_vol = range_ratio.rolling(20).std()
            manual_features['range_volatility'] = range_vol
            
            # Price efficiency
            price_efficiency = abs(returns.rolling(10).mean()) / (vol_10 + 1e-8)
            manual_features['price_efficiency'] = price_efficiency
            
            # Trend strength
            trend_strength = abs(returns.rolling(20).mean()) / (returns.rolling(20).std() + 1e-8)
            manual_features['trend_strength'] = trend_strength
            
            # 9. Market microstructure risk features
            spread_proxy = range_ratio
            spread_volatility = spread_proxy.rolling(20).std()
            manual_features['spread_volatility'] = spread_volatility
            
            # Order flow imbalance proxy
            price_volume_imbalance = (returns * volume).rolling(10).sum()
            manual_features['order_flow_imbalance'] = price_volume_imbalance
            
            # Market depth proxy
            market_depth = volume / (range_ratio + 1e-8)
            manual_features['market_depth'] = market_depth
            
            # 10. Composite risk indicators
            # Risk stress index
            vol_zscore_20 = (vol_20 - vol_20.rolling(100).mean()) / (vol_20.rolling(100).std() + 1e-8)
            risk_stress_index = (
                0.3 * (vol_zscore_20 > 1).astype(int) +
                0.3 * (max_drawdown < -0.05).astype(int) +
                0.2 * (volume_vol_divergence > 0).astype(int) +
                0.2 * (drawdown_velocity < -0.01).astype(int)
            )
            manual_features['risk_stress_index'] = risk_stress_index
            
            # Risk appetite indicator
            risk_appetite = 1 - risk_stress_index
            manual_features['risk_appetite'] = risk_appetite
            
        return manual_features
    def _add_risk_specific_features(self, df: pd.DataFrame, risk_features: pd.DataFrame) -> pd.DataFrame:
        """Add risk-specific enhanced features."""
        features = pd.DataFrame(index=df.index)
        
        # Enhanced risk analysis
        if 'close' in df.columns:
            close = df['close']
            returns = close.pct_change()
            
            # Multi-timeframe risk analysis
            for window in [60,80,100]:
                # Volatility risk
                volatility = returns.rolling(window).std()
                features[f'volatility_risk_{window}'] = volatility
                
                # Downside risk
                downside_returns = returns[returns < 0]
                downside_volatility = downside_returns.rolling(window).std()
                features[f'downside_volatility_{window}'] = downside_volatility
                
                # Upside volatility
                upside_returns = returns[returns > 0]
                upside_volatility = upside_returns.rolling(window).std()
                features[f'upside_volatility_{window}'] = upside_volatility
                
                # Volatility skewness
                volatility_skew = upside_volatility / (downside_volatility + 1e-8)
                features[f'volatility_skew_{window}'] = volatility_skew
                
                # Maximum drawdown risk
                cumulative_returns = (1 + returns).cumprod()
                rolling_max = cumulative_returns.rolling(window).max()
                drawdown = (cumulative_returns - rolling_max) / rolling_max
                max_drawdown = drawdown.rolling(window).min()
                features[f'max_drawdown_{window}'] = max_drawdown
                
                # Drawdown duration
                drawdown_duration = (drawdown < 0).rolling(window).sum()
                features[f'drawdown_duration_{window}'] = drawdown_duration
                
                # Risk-adjusted returns
                risk_adjusted_returns = returns.rolling(window).mean() / volatility
                features[f'risk_adjusted_returns_{window}'] = risk_adjusted_returns
                
                # Value at Risk (VaR)
                var_95 = returns.rolling(window).quantile(0.05)
                var_99 = returns.rolling(window).quantile(0.01)
                features[f'var_95_{window}'] = var_95
                features[f'var_99_{window}'] = var_99
                
                # Conditional Value at Risk (CVaR)
                cvar_95 = returns[returns <= var_95].rolling(window).mean()
                features[f'cvar_95_{window}'] = cvar_95
            
            # Risk regime indicators
            for window in [20, 50]:
                # High volatility regime
                volatility_ma = returns.rolling(window).std().rolling(window*2).mean()
                high_volatility = (returns.rolling(window).std() > volatility_ma * 1.5)
                features[f'high_volatility_regime_{window}'] = high_volatility.astype(int)
                
                # Low volatility regime
                low_volatility = (returns.rolling(window).std() < volatility_ma * 0.5)
                features[f'low_volatility_regime_{window}'] = low_volatility.astype(int)
                
                # Risk escalation
                risk_escalation = volatility.rolling(window).diff()
                features[f'risk_escalation_{window}'] = risk_escalation
                
                # Risk persistence
                risk_persistence = (high_volatility.rolling(window).mean())
                features[f'risk_persistence_{window}'] = risk_persistence
            
            # Tail risk analysis
            for window in [20, 50]:
                # Tail risk indicator
                tail_risk = returns.rolling(window).apply(lambda x: (x < x.quantile(0.05)).mean())
                features[f'tail_risk_{window}'] = tail_risk
                
                # Extreme events
                extreme_events = (returns.abs() > returns.rolling(window).std() * 2)
                features[f'extreme_events_{window}'] = extreme_events.rolling(window).sum()
                
                # Tail risk persistence
                tail_risk_persistence = tail_risk.rolling(window).mean()
                features[f'tail_risk_persistence_{window}'] = tail_risk_persistence
        
        # Volume-risk relationship
        if 'volume' in df.columns and 'close' in df.columns:
            volume = df['volume']
            returns = df['close'].pct_change()
            
            # Volume-adjusted risk
            volume_ma = volume.rolling(25).mean()
            volume_anomaly = volume / volume_ma
            volume_adjusted_volatility = returns.rolling(25).std() * volume_anomaly
            features['volume_adjusted_volatility'] = volume_adjusted_volatility
            
            # Volume-risk correlation
            for window in [10, 20, 50]:
                volume_risk_corr = returns.rolling(window).corr(volume)
                features[f'volume_risk_corr_{window}'] = volume_risk_corr
                
                # Volume confirmation of risk
                features[f'volume_risk_confirmation_{window}'] = (
                    (volume_anomaly > 1.5) & (returns.rolling(window).std() > returns.rolling(window*2).std() * 1.2)
                ).astype(int)
                
                # Volume-weighted risk
                volume_weighted_risk = (returns.abs() * volume).rolling(window).sum()
                features[f'volume_weighted_risk_{window}'] = volume_weighted_risk
        
        # Support/resistance risk analysis
        if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
            close = df['close']
            high = df['high']
            low = df['low']
            
            # Range-based risk
            for window in [20, 50]:
                range_volatility = (high - low).rolling(window).std()
                features[f'range_volatility_{window}'] = range_volatility
                
                # Range expansion risk
                range_ma = (high - low).rolling(window).mean()
                range_expansion = (high - low) / range_ma
                features[f'range_expansion_{window}'] = range_expansion
                
                # Range contraction risk
                range_contraction = (range_expansion < 0.7).astype(int)
                features[f'range_contraction_{window}'] = range_contraction
                
                # Breakout risk
                rolling_max = close.rolling(window).max()
                rolling_min = close.rolling(window).min()
                breakout_risk = (close > rolling_max.shift(1)) | (close < rolling_min.shift(1))
                features[f'breakout_risk_{window}'] = breakout_risk.astype(int)
        
        # Time-based risk patterns
        if isinstance(df.index, pd.DatetimeIndex):
            features['hour_of_day'] = df.index.hour
            features['day_of_week'] = df.index.dayofweek
            features['is_london_session'] = ((df.index.hour >= 8) & (df.index.hour <= 16)).astype(int)
            features['is_ny_session'] = ((df.index.hour >= 13) & (df.index.hour <= 21)).astype(int)
            features['is_asia_session'] = ((df.index.hour >= 0) & (df.index.hour <= 8)).astype(int)
            
            # Session overlaps
            features['is_london_ny_overlap'] = ((df.index.hour >= 13) & (df.index.hour <= 16)).astype(int)
            
            # Weekend effects on risk
            features['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
            
            # Time-based risk escalation
            features['is_end_of_day'] = (df.index.hour >= 20).astype(int)
            features['is_start_of_day'] = (df.index.hour <= 8).astype(int)
            
            # Month-end risk
            features['is_month_end'] = (df.index.day >= 28).astype(int)
        
        return features
    
    def _create_risk_labels(self, df: pd.DataFrame, lookforward: int = 35) -> pd.Series:
        """Create enhanced risk labels based on multiple risk factors."""
        if 'close' in df.columns:
            returns = df['close'].pct_change()
            
            # Multi-timeframe volatility analysis
            vol_10 = returns.rolling(10).std()
            vol_20 = returns.rolling(25).std()
            vol_50 = returns.rolling(60).std()
            
            # Volatility regime detection
            vol_regime_current = vol_20 / vol_50
            vol_regime_future = returns.shift(-lookforward).rolling(25).std() / vol_50
            
            # Drawdown-based risk
            rolling_max = df['close'].rolling(20).max()
            drawdown = (df['close'] - rolling_max) / rolling_max
            future_drawdown = drawdown.shift(-lookforward)
            
            # Downside risk focus
            downside_returns = returns.copy()
            downside_returns[downside_returns > 0] = 0
            downside_vol = downside_returns.rolling(20).std()
            future_downside_vol = downside_returns.shift(-lookforward).rolling(20).std()
            
            # Enhanced risk labeling with balanced thresholds
            # Condition 1: Volatility escalation (adjusted for balance)
            vol_escalation = vol_regime_future > vol_regime_current * 1.15  # Increased from 1.05
            
            # Condition 2: Significant drawdown ahead (adjusted for balance)
            drawdown_risk = future_drawdown < -0.025  # Increased from 2% to 2.5%
            
            # Condition 3: Downside volatility increase (adjusted for balance)
            downside_risk = future_downside_vol > downside_vol * 1.3  # Increased from 1.1
            
            # Condition 4: Price shock risk (adjusted for balance)
            price_shock_threshold = vol_20 * 1.8  # Increased from 1.5-sigma to 1.8-sigma
            future_shock = abs(returns.shift(-lookforward)) > price_shock_threshold
            
            # Condition 5: Volatility regime shift detection (adjusted for balance)
            vol_regime_shift = (
                (vol_regime_current < 0.7) & (vol_regime_future > 1.3) |  # Tighter thresholds
                (vol_regime_current > 1.3) & (vol_regime_future < 0.7)
            )
            
            # Condition 6: Sudden volatility spike detection (adjusted for balance)
            vol_spike_threshold = vol_20.rolling(50).mean() * 2.2  # Increased from 1.8
            vol_spike = vol_regime_future > vol_spike_threshold
            
            # Condition 7: Volatility persistence detection (adjusted for balance)
            vol_persistence = (vol_regime_future > 1.5) & (vol_regime_current > 1.5)  # Increased from 1.3
            
            # Combine risk signals with higher threshold for balance
            risk_signal = (
                vol_escalation.astype(int) + 
                drawdown_risk.astype(int) + 
                downside_risk.astype(int) + 
                future_shock.astype(int) +
                vol_regime_shift.astype(int) +
                vol_spike.astype(int) +
                vol_persistence.astype(int)
            )
            
            # Label: 1 for high risk conditions, 0 for normal/low risk
            # Use higher threshold to reduce positive labels
            labels = (risk_signal >= 2).astype(int)  # Increased from 1 to 2
            
            # Add regime-aware adjustment
            # In high volatility regimes, be more selective
            high_vol_regime = vol_regime_current > 1.5
            labels.loc[high_vol_regime] = labels.loc[high_vol_regime] & (
                (vol_escalation.loc[high_vol_regime]) | 
                (future_shock.loc[high_vol_regime]) |
                (vol_spike.loc[high_vol_regime])
            )
            
            # In low volatility regimes, be more sensitive to escalation
            low_vol_regime = vol_regime_current < 0.7
            labels.loc[low_vol_regime] = labels.loc[low_vol_regime] | (
                (vol_escalation.loc[low_vol_regime]) |
                (vol_regime_shift.loc[low_vol_regime])
            )
            
            return labels.astype(int)
        else:
            # Enhanced fallback labels
            returns = df['close'].pct_change()
            vol_20 = returns.rolling(25).std()
            future_vol = returns.shift(-lookforward).rolling(25).std()
            
            # Use lower threshold for better sensitivity
            labels = (future_vol > vol_20 * 1.1).astype(int)
            return labels
    

    def _optimize_xgb_hyperparameters_for_mi(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, Any], float]:
        """Optimize XGBoost hyperparameters specifically for MI improvement."""
        best_params = {}
        best_mi = 0.0
        
        # Parameter grid for XGBoost MI optimization
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
        
        for params in self._generate_param_combinations(param_grid, max_combinations=15):
            mi_scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train XGBoost model
                import xgboost as xgb
                model = xgb.XGBClassifier(
                    objective='binary:logistic',
                    random_state=42,
                    eval_metric='logloss',
                    early_stopping_rounds=20,
                    **params
                )
                
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                         verbose=False)
                
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
                
                tprint_info(f"🔥 New best XGB MI: {avg_mi:.4f} with params: {params}")
        
        tprint_success(f"✅ Best XGBoost hyperparameters found: MI = {best_mi:.4f}")
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
    
    def _train_enhanced_risk_model(self, features: pd.DataFrame, labels: pd.Series, 
                                 config: Dict[str, Any], sample_weight: Optional[pd.Series] = None) -> Tuple[Any, Dict[str, float]]:
        """Train enhanced risk model with MI optimization and optional AFML weights, optimized for Meta-Labeling."""
        import xgboost as xgb
        
        # Optimize hyperparameters for MI
        tprint_info("🔧 Optimizing XGBoost hyperparameters for Meta-Labeling (Tail-Loss probability)...")
        best_params, best_mi = self._optimize_xgb_hyperparameters_for_mi(features, labels)
        
        tprint_success(f"✅ Best hyperparameters found with MI = {best_mi:.4f}")
        
        # Train final model with best params using time series CV
        tscv = TimeSeriesSplit(n_splits=5)
        
        final_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            early_stopping_rounds=20,
            n_estimators=best_params.get('n_estimators', 300),
            max_depth=best_params.get('max_depth', 6),
            learning_rate=best_params.get('learning_rate', 0.05),
            subsample=best_params.get('subsample', 0.8),
            colsample_bytree=best_params.get('colsample_bytree', 0.8),
            gamma=best_params.get('gamma', 0.1),
            reg_alpha=best_params.get('reg_alpha', 0.5),
            reg_lambda=best_params.get('reg_lambda', 2.0),
            min_child_weight=best_params.get('min_child_weight', 20),
            # Risk focus: scale_pos_weight if labels are imbalanced (tail losses are rare)
            scale_pos_weight=(len(labels) - labels.sum()) / (labels.sum() + 1e-8) if labels.sum() > 0 else 1.0
        )
        
        # Get last fold for final training with validation
        train_indices, val_indices = list(tscv.split(features))[-1]
        X_train, X_val = features.iloc[train_indices], features.iloc[val_indices]
        y_train, y_val = labels.iloc[train_indices], labels.iloc[val_indices]
        w_train = sample_weight.iloc[train_indices] if sample_weight is not None else None
        
        final_model.fit(X_train, y_train, sample_weight=w_train, eval_set=[(X_val, y_val)], verbose=False)
        
        # Compute MI score on validation set
        val_pred = final_model.predict_proba(X_val)[:, 1]
        mi_score = self.compute_binned_mi(val_pred, y_val.values)
        
        # Store training metrics
        self.training_metrics.append({
            'mi_score': mi_score,
            'n_features': len(features.columns),
            'best_params': best_params
        })
        
        # Compute AUC on validation set
        from sklearn.metrics import roc_auc_score, log_loss
        try:
            auc_score = roc_auc_score(y_val, val_pred)
            logloss = log_loss(y_val, val_pred)
        except Exception:
            auc_score = 0.5
            logloss = 0.0
        
        metrics = {
            'mi_score': mi_score,
            'auc': auc_score,
            'log_loss': logloss,
            'n_features': len(features.columns),
            'optimization_params': best_params,
        }
        
        return final_model, metrics
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute enhanced risk regime step."""
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
                model="enhanced_risk_regime",
            )

            tprint_info(f"🚀 Starting Enhanced Risk Regime for {symbol} on {exchange}")

            # 1. Load Market Data
            market_data, market_source = self._load_market_data_with_cache(config, timeframe)

            # 2. Generate Enhanced Features
            tprint_info("🛠️ Generating Enhanced Risk Regime features...")
            features_df = self._generate_enhanced_risk_features(market_data, config)
            
            # AFML UPDATE: Apply CUSUM Sampling
            # Risk regime should focus on periods of significant volatility activity to capture tail-risk
            # 3. AFML: Sampling, Labeling, Weighting, Alignment via Helper
            tprint_info("🎯 Generating Enhanced Risk Regime labels with Triple Barrier Method...")
            X, y, weights = self.prepare_specialist_data(
                market_data=market_data,
                feature_df=features_df,
                config=config,
                filter_type='volatility',
                pt_sl_config_key='risk_regime_pt_sl',
                default_pt_sl=[2.0, 1.0]
            )

            # 4. Train Enhanced Model with MI Optimization using OOF and weights
            tprint_info("🤖 Training Enhanced Risk Regime model (Meta-Label) with OOF & AFML weights...")
            
            # Use TimeSeriesSplit for OOF predictions
            n_splits = 5
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            oof_probs = pd.Series(np.nan, index=X.index)
            last_model = None
            
            # Optimize hyperparameters once on the sampled set
            best_params, best_mi = self._optimize_xgb_hyperparameters_for_mi(X, y)

            for fold, (train_indices, val_indices) in enumerate(tscv.split(X)):
                X_train_fold, X_val_fold = X.iloc[train_indices], X.iloc[val_indices]
                y_train_fold, y_val_fold = y.iloc[train_indices], y.iloc[val_indices]
                w_train_fold = weights.iloc[train_indices]
                
                # Train model for this fold
                model = xgb.XGBClassifier(
                    objective='binary:logistic',
                    eval_metric='logloss',
                    early_stopping_rounds=20,
                    random_state=42 + fold,
                    **best_params
                )
                
                model.fit(
                    X_train_fold.fillna(0), 
                    y_train_fold.fillna(0), 
                    sample_weight=w_train_fold,
                    eval_set=[(X_val_fold.fillna(0), y_val_fold.fillna(0))], 
                    verbose=False
                )
                
                # Store OOF prediction
                fold_probs = model.predict_proba(X_val_fold.fillna(0))[:, 1]
                oof_probs.iloc[val_indices] = fold_probs
                last_model = model

            # AFML Audit: Update metrics using full OOF set
            valid_oof = oof_probs.dropna()
            if len(valid_oof) > 0:
                y_full_true = y.loc[valid_oof.index]
                y_full_pred_prob = valid_oof.values
                y_full_pred = (y_full_pred_prob >= 0.5).astype(int)
                
                metrics = {}
                try:
                    # Use fast binned MI proxy for binary targets
                    metrics['auc'] = float(roc_auc_score(y_full_true, y_full_pred_prob))
                    metrics['mi_score'] = float(self.compute_binned_mi(y_full_pred_prob, y_full_true.values))
                except Exception as e:
                    self.logger.warning(f"Failed to calculate full OOF metrics: {e}")
                    metrics['auc'] = 0.5
                    metrics['mi_score'] = 0.0
            else:
                metrics = {'auc': 0.5, 'mi_score': 0.0}
                y_full_pred_prob = np.array([])
                y_full_pred = np.array([])

            metrics.update({
                'n_features': len(X.columns),
                'optimization_params': best_params if 'best_params' in locals() else {},
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

            result = self.save_specialist_results(
                config=config,
                feature_df=feature_df if 'feature_df' in locals() else (features_df if 'features_df' in locals() else X),
                labels=full_labels,
                predictions=final_preds.values,
                probabilities=final_probs.values,
                model=last_model,
                metrics=metrics,
                specialist_name="EnhancedMLRiskRegimeStep"
            )

            # 9. Final Summary
            execution_time = time.time() - start_time
            metrics["execution_time"] = execution_time
            metrics["n_samples"] = len(X)

            result["execution_time"] = execution_time
            result["mi_history"] = self.mi_history
            result["training_metrics"] = self.training_metrics

            tprint_success(f"✅ Enhanced Risk Regime completed in {execution_time:.2f}s")
            tprint_info(f"📊 Final Metrics: MI={metrics.get('mi_score', 0):.4f}, AUC={metrics.get('auc', 0):.3f}")

            return result

        except Exception as e:
            self.logger.exception(f"❌ Enhanced Risk Regime step failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _load_market_data_with_cache(self, config: Dict[str, Any], timeframe: str) -> Tuple[pd.DataFrame, str]:
        """Load market data with caching using BaseStep method."""
        symbol = config.get("symbol", "ETHUSDT")
        exchange = config.get("exchange", "binance")
        
        # Generate cache key
        cache_key = (symbol, exchange, timeframe)
        
        # Check cache first
        if cache_key in self._market_data_cache:
            self.logger.info(f"📦 Using cached market data for {symbol}")
            return self._market_data_cache[cache_key], "cache"
        
        # Use standard BaseStep method
        market_data, market_source = self.load_market_data_or_fail(
            {**config, "timeframe": timeframe},
            pipeline_state={},
            allow_config_override=True,
        )
        
        # Cache the data
        self._market_data_cache[cache_key] = market_data
        
        self.logger.info(f"✅ Loaded {len(market_data)} rows of market data for {symbol} from {market_source}")
        return market_data, market_source
    
    def _generate_synthetic_market_data(self, symbol: str, timeframe: str, end_date: datetime) -> pd.DataFrame:
        """Generate synthetic market data with realistic characteristics."""
        import numpy as np
        
        # Determine number of periods based on timeframe
        periods_per_day = {
            '1m': 1440,
            '5m': 288,
            '15m': 96,
            '1h': 24,
            '4h': 6,
            '1d': 1
        }
        
        periods = periods_per_day.get(timeframe, 96)  # Default to 15m
        
        # Generate 3 years of data
        total_periods = periods * 365 * 3
        
        # Create date range
        dates = pd.date_range(end=end_date, periods=total_periods, freq=f'{timeframe}')
        
        # Generate realistic price series with trend and volatility
        np.random.seed(42)
        
        # Base price (starting around $2000 for ETH)
        base_price = 2000.0
        
        # Generate returns with realistic characteristics
        # Geometric Brownian Motion
        drift = 0.0001  # Slight upward trend
        volatility = 0.02  # 2% per period
        
        returns = []
        current_return = 0.0
        
        for i in range(total_periods):
            # Random shock
            random_shock = np.random.normal(0, volatility)
            
            # Total return
            total_return = drift + random_shock
            
            # Add some volatility clustering
            if i > 20:
                recent_vol = np.std(returns[-20:]) if len(returns) > 20 else volatility
                volatility_factor = 0.7 + 0.6 * (recent_vol / volatility)
                total_return *= volatility_factor
            
            returns.append(total_return)
            current_return = total_return
        
        # Convert to prices
        prices = [base_price]
        for ret in returns:
            new_price = prices[-1] * (1 + ret)
            prices.append(new_price)
        
        prices = prices[1:]  # Remove initial price
        
        # Generate OHLC from returns
        high_prices = []
        low_prices = []
        close_prices = prices
        open_prices = []
        
        for i in range(len(prices)):
            if i == 0:
                open_price = base_price
            else:
                open_price = close_prices[i-1]
            
            # Generate realistic intrabar movement
            intrabar_vol = volatility * 0.5
            high = open_price * (1 + abs(np.random.normal(0, intrabar_vol)))
            low = open_price * (1 - abs(np.random.normal(0, intrabar_vol)))
            
            # Ensure OHLC relationships
            high = max(high, open_price, close_prices[i])
            low = min(low, open_price, close_prices[i])
            
            high_prices.append(high)
            low_prices.append(low)
            open_prices.append(open_price)
        
        # Generate volume with correlation to price movement
        base_volume = 1000000
        volume_multipliers = []
        
        for i in range(len(prices)):
            # Volume tends to be higher with larger price movements
            price_move = abs(returns[i])
            volume_multiplier = 1.0 + price_move * 10  # Volume increases with price movement
            
            # Add some randomness
            volume_multiplier *= np.random.uniform(0.5, 2.0)
            
            volume_multipliers.append(volume_multiplier)
        
        volumes = [base_volume * vm for vm in volume_multipliers]
        
        # Create DataFrame
        synthetic_data = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        }, index=dates)
        
        return synthetic_data
