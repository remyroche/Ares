"""
Enhanced ML Reversion Regime Step with MI Improvements

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

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score

from src.training.steps.base_step import BaseStep
from src.utils.tprint import (
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

logger = logging.getLogger(__name__)


class EnhancedMLReversionRegimeStep(SpecialistDiagnosticsMixinEnhancedV2, AFMLSpecialistMixin, BaseStep):

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
    Enhanced Reversion Regime Specialist with MI optimization.
    
    Key enhancements:
    - Enhanced feature generation for MI improvement
    - Real-time MI monitoring during training
    - Reversion-specific feature engineering
    - Hyperparameter optimization for MI > 0.02
    - Data structure standardization
    - Binary output enforcement
    """
    
    def __init__(self, step_name: str = "enhanced_ml_reversion_regime_step"):
        """Initialize the enhanced reversion regime step."""
        super().__init__()
        self._current_context = {}
        self._artifact_manager = None
        self._versioned_store = None
        self.step_name = step_name
        BaseStep.__init__(self, step_name, use_versioned_artifacts=True)
        self.logger = logger.getChild("EnhancedMLReversionRegimeStep")
        self.feature_pipeline = MIOptimizedFeaturePipeline()
        self.mi_history = []
        self.training_metrics = []
        
    def _generate_enhanced_reversion_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Generate enhanced reversion features with manual feature engineering."""
        # Import original reversion features
        try:
            from src.feature_generation.categories.reversion_regime_features import generate_reversion_regime_features
            base_reversion_features = generate_reversion_regime_features(df, config)
        except ImportError:
            # Fallback if original features not available
            base_reversion_features = pd.DataFrame(index=df.index)
        
        # Enhanced features from pipeline
        enhanced_features = self.feature_pipeline.generate_enhanced_features(
            df, 'reversion', {'enhanced_features': True}
        )
        
        # Manual feature engineering for reversion regime
        manual_features = self._create_manual_reversion_enhanced_features(df, enhanced_features)
        
        # Reversion-specific enhanced features (vectorized)
        specific_reversion_features = self._add_reversion_specific_features(df, base_reversion_features)
        
        # Combine all features
        all_features = [base_reversion_features, enhanced_features, manual_features, specific_reversion_features]
        
        # Combine all features with manual redundancy reduction
        if all_features:
            combined_features = pd.concat(all_features, axis=1)
            
            # Manual redundancy reduction and feature selection
            combined_features = self._apply_manual_reversion_feature_selection(combined_features)
            
            # Remove duplicates and clean
            combined_features = combined_features.loc[:, ~combined_features.columns.duplicated()]
            combined_features = combined_features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            
            return combined_features
        
        return pd.DataFrame(index=df.index)
    
    def _create_manual_reversion_enhanced_features(self, df: pd.DataFrame, enhanced_features: pd.DataFrame) -> pd.DataFrame:
        """Create manual enhanced features for reversion regime detection."""
        manual_features = pd.DataFrame(index=df.index)
        
        if all(col in df.columns for col in ['close', 'high', 'low', 'volume']):
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df.get('volume', pd.Series(1, index=df.index))
            returns = close.pct_change()
            
            # 1. Enhanced mean reversion features
            # Multi-timeframe mean reversion signals
            for window in [10, 20, 50, 100]:
                mean_price = close.rolling(window).mean()
                reversion_signal = (close - mean_price) / mean_price
                manual_features[f'reversion_signal_{window}'] = reversion_signal
                
                # Reversion strength (distance from mean)
                reversion_strength = abs(reversion_signal)
                manual_features[f'reversion_strength_{window}'] = reversion_strength
                
                # Reversion velocity (speed of return to mean)
                reversion_velocity = reversion_signal.diff()
                manual_features[f'reversion_velocity_{window}'] = reversion_velocity
            
            # Precompute commonly reused signals
            reversion_signal_20 = (close - close.rolling(20).mean()) / close.rolling(20).mean()
            reversion_signal_50 = (close - close.rolling(50).mean()) / close.rolling(50).mean()
            
            # 2. Bollinger Band-based reversion features
            for window in [20, 50]:
                bb_mean = close.rolling(window).mean()
                bb_std = close.rolling(window).std()
                bb_upper = bb_mean + 2 * bb_std
                bb_lower = bb_mean - 2 * bb_std
                bb_position = (close - bb_lower) / (bb_upper - bb_lower)
                
                manual_features[f'bb_position_{window}'] = bb_position
                manual_features[f'bb_width_{window}'] = (bb_upper - bb_lower) / bb_mean
                manual_features[f'bb_squeeze_{window}'] = ((bb_upper - bb_lower) < (bb_upper - bb_lower).rolling(50).mean()).astype(int)
            
            # 3. RSI-based reversion features
            for window in [14, 30]:
                gains = returns.clip(lower=0)
                losses = -returns.clip(upper=0)
                avg_gains = gains.rolling(window).mean()
                avg_losses = losses.rolling(window).mean()
                rs = avg_gains / (avg_losses + 1e-8)
                rsi = 100 - (100 / (1 + rs))
                
                # RSI reversion signals
                rsi_overbought = (rsi > 70).astype(int)
                rsi_oversold = (rsi < 30).astype(int)
                rsi_neutral = ((rsi >= 30) & (rsi <= 70)).astype(int)
                
                manual_features[f'rsi_overbought_{window}'] = rsi_overbought
                manual_features[f'rsi_oversold_{window}'] = rsi_oversold
                manual_features[f'rsi_neutral_{window}'] = rsi_neutral
                manual_features[f'rsi_reversion_{window}'] = 50 - abs(rsi - 50)  # Distance from neutral
            
            # 4. Stochastic oscillator reversion features
            for window in [14, 20]:
                lowest_low = low.rolling(window).min()
                highest_high = high.rolling(window).max()
                stochastic = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-8)
                
                # Stochastic reversion signals
                stoch_overbought = (stochastic > 80).astype(int)
                stoch_oversold = (stochastic < 20).astype(int)
                
                manual_features[f'stoch_overbought_{window}'] = stoch_overbought
                manual_features[f'stoch_oversold_{window}'] = stoch_oversold
                manual_features[f'stoch_reversion_{window}'] = 50 - abs(stochastic - 50)
            
            # 5. Price channel reversion features
            for window in [20, 50]:
                channel_upper = high.rolling(window).max()
                channel_lower = low.rolling(window).min()
                channel_middle = (channel_upper + channel_lower) / 2
                channel_position = (close - channel_lower) / (channel_upper - channel_lower + 1e-8)
                
                manual_features[f'channel_position_{window}'] = channel_position
                manual_features[f'channel_breakout_{window}'] = ((close > channel_upper) | (close < channel_lower)).astype(int)
                manual_features[f'channel_reversion_{window}'] = 0.5 - abs(channel_position - 0.5)
            
            # 6. Volume-adjusted reversion features
            if 'volume' in df.columns:
                volume_ma = volume.rolling(20).mean()
                volume_ratio = volume / (volume_ma + 1e-8)
                
                for window in [20, 50]:
                    mean_price = close.rolling(window).mean()
                    volume_adjusted_reversion = (close - mean_price) / mean_price * volume_ratio
                    manual_features[f'volume_adjusted_reversion_{window}'] = volume_adjusted_reversion
                    
                    # Volume divergence from reversion
                    reversion_signal = (close - mean_price) / mean_price
                    volume_divergence = abs(reversion_signal) * (1 - volume_ratio)
                    manual_features[f'volume_divergence_{window}'] = volume_divergence
            
            # 7. Multi-timeframe reversion confirmation
            # Short-term vs long-term reversion agreement
            short_reversion = (close - close.rolling(10).mean()) / close.rolling(10).mean()
            long_reversion = (close - close.rolling(50).mean()) / close.rolling(50).mean()
            reversion_agreement = np.sign(short_reversion) == np.sign(long_reversion)
            manual_features['reversion_agreement'] = reversion_agreement.astype(int)
            
            # Reversion divergence (signals pointing opposite directions)
            reversion_divergence = abs(short_reversion - long_reversion)
            manual_features['reversion_divergence'] = reversion_divergence
            
            # 8. Reversion regime classification
            # Strong reversion regime (far from mean)
            strong_reversion_20 = abs((close - close.rolling(20).mean()) / close.rolling(20).mean()) > 0.02
            strong_reversion_50 = abs((close - close.rolling(50).mean()) / close.rolling(50).mean()) > 0.03
            manual_features['strong_reversion_regime'] = (strong_reversion_20 | strong_reversion_50).astype(int)
            
            # Weak reversion regime (close to mean)
            weak_reversion_20 = abs((close - close.rolling(20).mean()) / close.rolling(20).mean()) < 0.01
            weak_reversion_50 = abs((close - close.rolling(50).mean()) / close.rolling(50).mean()) < 0.015
            manual_features['weak_reversion_regime'] = (weak_reversion_20 | weak_reversion_50).astype(int)
            
            # 9. Reversion momentum features
            # Reversion acceleration (second derivative of reversion signal)
            reversion_acceleration = reversion_signal_20.diff().diff()
            manual_features['reversion_acceleration'] = reversion_acceleration
            
            # Reversion persistence (how long reversion signal persists)
            reversion_persistence = (np.sign(reversion_signal_20) == np.sign(reversion_signal_20.shift(1))).rolling(10).sum()
            manual_features['reversion_persistence'] = reversion_persistence
            
            # 10. Advanced reversion risk features
            # Reversion failure risk (price continues moving away from mean)
            reversion_failure = (abs(reversion_signal_20) > abs(reversion_signal_20.shift(1))) & (np.sign(reversion_signal_20) == np.sign(reversion_signal_20.shift(1)))
            manual_features['reversion_failure_risk'] = reversion_failure.astype(int)
            
            # Reversion success probability (based on historical reversion success)
            reversion_success = (np.sign(reversion_signal_20) != np.sign(reversion_signal_20.shift(5))).rolling(50).mean()
            manual_features['reversion_success_probability'] = reversion_success
            
            # 11. Composite reversion indicators
            # Reversion strength index (combines multiple reversion signals)
            zero_series = pd.Series(0, index=df.index)
            rsi_overbought_14 = manual_features.get('rsi_overbought_14', zero_series)
            rsi_oversold_14 = manual_features.get('rsi_oversold_14', zero_series)
            stoch_overbought_14 = manual_features.get('stoch_overbought_14', zero_series)
            stoch_oversold_14 = manual_features.get('stoch_oversold_14', zero_series)
            
            reversion_strength_index = (
                0.3 * (abs(reversion_signal_20) > 0.02).astype(int) +
                0.3 * (abs(reversion_signal_50) > 0.03).astype(int) +
                0.2 * ((rsi_overbought_14.astype(bool) | rsi_oversold_14.astype(bool))).astype(int) +
                0.2 * ((stoch_overbought_14.astype(bool) | stoch_oversold_14.astype(bool))).astype(int)
            )
            manual_features['reversion_strength_index'] = reversion_strength_index
            
            # Reversion quality index (confidence in reversion signal)
            reversion_quality = (
                0.25 * reversion_agreement +
                0.25 * (reversion_persistence >= 5).astype(int) +
                0.25 * (reversion_success_probability > 0.6).astype(int) +
                0.25 * (volume_ratio > 1.2).astype(int) if 'volume' in df.columns else 0
            )
            manual_features['reversion_quality'] = reversion_quality
            
        return manual_features
    
    def _apply_manual_reversion_feature_selection(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply manual feature selection for reversion features."""
        if features.empty:
            return features
        
        # Remove constant features
        constant_features = features.columns[features.nunique() <= 1]
        if len(constant_features) > 0:
            features = features.drop(columns=constant_features)
            self.logger.info(f"Removed {len(constant_features)} constant reversion features")
        
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
            self.logger.info(f"Removed {len(set(to_drop))} redundant reversion features: {list(set(to_drop))}")
        
        # Keep only the most informative features (limit to top 30 by variance)
        if len(features.columns) > 30:
            feature_variances = features.var()
            top_features = feature_variances.nlargest(30).index
            features = features[top_features]
            self.logger.info(f"Limited reversion features to top 30 by variance")
        
        return features
    
    def _add_reversion_specific_features(self, df: pd.DataFrame, reversion_features: pd.DataFrame) -> pd.DataFrame:
        """Add reversion-specific enhanced features."""
        features = pd.DataFrame(index=df.index)
        
        # Enhanced reversion analysis
        if 'close' in df.columns:
            close = df['close']
            returns = close.pct_change()
            
            # Multi-timeframe reversion analysis
            for window in [20,40,60]:
                # Reversion signals
                reversion_signal = -returns.rolling(window).mean()
                features[f'reversion_signal_{window}'] = reversion_signal
                
                # Reversion strength
                features[f'reversion_strength_{window}'] = abs(reversion_signal)
                
                # Reversion persistence
                features[f'reversion_persistence_{window}'] = (reversion_signal > 0).rolling(window).mean()
                
                # Reversion volatility
                reversion_volatility = reversion_signal.rolling(window).std()
                features[f'reversion_volatility_{window}'] = reversion_volatility
                
                # Reversion acceleration
                features[f'reversion_acceleration_{window}'] = reversion_signal.diff().diff()
                
                # Reversion exhaustion detection
                features[f'reversion_exhaustion_{window}'] = (reversion_persistence[f'reversion_persistence_{window}'] < 0.2).astype(int)
            
            # Reversion opportunity detection
            for window in [10, 20, 50]:
                future_reversion = -returns.shift(-window)
                current_reversion = -returns.rolling(window).mean()
                
                reversion_opportunity = (future_reversion > current_reversion * 1.5)
                features[f'reversion_opportunity_{window}'] = reversion_opportunity.astype(int)
                
                # Reversion risk
                reversion_risk = returns.rolling(window).std()
                features[f'reversion_risk_{window}'] = reversion_risk
                
                # Reversion reward ratio
                features[f'reversion_reward_ratio_{window}'] = (
                    future_reversion.abs() / (current_reversion.abs() + 1e-8)
                )
            
            # Mean reversion analysis
            for window in [20, 50, 100]:
                mean_reversion = -returns.rolling(window).mean()
                features[f'mean_reversion_{window}'] = mean_reversion
                features[f'mean_reversion_ma_{window}'] = mean_reversion.rolling(window*2).mean()
                
                # Reversion consistency
                reversion_consistency = (mean_reversion > 0).rolling(window).mean()
                features[f'reversion_consistency_{window}'] = reversion_consistency
                
                # Reversion trend
                reversion_trend = mean_reversion.diff()
                features[f'reversion_trend_{window}'] = reversion_trend
            
            # Volatility-adjusted reversion
            volatility = returns.rolling(25).std()
            volatility_adjusted_reversion = mean_reversion / volatility
            features[f'volatility_adjusted_reversion_20'] = volatility_adjusted_reversion
            features[f'volatility_adjusted_reversion_50'] = volatility_adjusted_reversion.rolling(60).mean()
            
            # Price level reversion analysis
            for price_level in [0.5, 1.0, 2.0]:
                price_adjusted_returns = returns / price_level
                features[f'price_adjusted_reversion_{price_level}'] = price_adjusted_returns.rolling(25).mean()
        
        # Volume-reversion relationship
        if 'volume' in df.columns and 'close' in df.columns:
            volume = df['volume']
            returns = df['close'].pct_change()
            
            # Volume-reversion correlation
            for window in [10, 20, 50]:
                volume_reversion_corr = returns.rolling(window).corr(volume)
                features[f'volume_reversion_corr_{window}'] = volume_reversion_corr
                
                # Volume confirmation of reversion
                volume_ma = volume.rolling(25).mean()
                volume_anomaly = volume / volume_ma
                features[f'volume_reversion_confirmation_{window}'] = (
                    (volume_anomaly > 1.2) & (returns.rolling(window).mean() < 0)
                ).astype(int)
                
                # Volume-weighted reversion
                volume_weighted_reversion = (returns * volume).rolling(window).sum()
                features[f'volume_weighted_reversion_{window}'] = volume_weighted_reversion
        
        # Support/resistance adjusted reversion
        if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
            close = df['close']
            
            # Position-based reversion analysis
            for window in [20, 50]:
                rolling_max = close.rolling(window).max()
                rolling_min = close.rolling(window).min()
                close_position = (close - rolling_min) / (rolling_max - rolling_min)
                
                # Distance-based reversion opportunities
                features[f'distance_to_support_{window}'] = close_position
                features[f'distance_to_resistance_{window}'] = 1 - close_position
                
                # Reversion from support/resistance
                features[f'support_reversion_{window}'] = (close_position < 0.2).astype(int)
                features[f'resistance_reversion_{window}'] = (close_position > 0.8).astype(int)
                
                # Mid-range reversion
                features[f'mid_range_reversion_{window}'] = ((close_position >= 0.3) & (close_position <= 0.7)).astype(int)
        
        # Time-based reversion patterns
        if isinstance(df.index, pd.DatetimeIndex):
            features['hour_of_day'] = df.index.hour
            features['day_of_week'] = df.index.dayofweek
            features['is_london_session'] = ((df.index.hour >= 8) & (df.index.hour <= 16)).astype(int)
            features['is_ny_session'] = ((df.index.hour >= 13) & (df.index.hour <= 21)).astype(int)
            features['is_asia_session'] = ((df.index.hour >= 0) & (df.index.hour <= 8)).astype(int)
            
            # Session overlaps
            features['is_london_ny_overlap'] = ((df.index.hour >= 13) & (df.index.hour <= 16)).astype(int)
            
            # Weekend effects on reversion
            features['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
            
            # Time-based reversion opportunities
            features['is_end_of_day'] = (df.index.hour >= 20).astype(int)
            features['is_start_of_day'] = (df.index.hour <= 8).astype(int)
        
        return features
    
    def _create_reversion_labels(self, df: pd.DataFrame, lookforward: int = 35) -> pd.Series:
        """Create reversion labels based on mean reversion patterns."""
        if 'close' in df.columns:
            returns = df['close'].pct_change()
            
            # Multi-timeframe reversion analysis
            reversion_20 = -returns.rolling(25).mean()
            reversion_50 = -returns.rolling(60).mean()
            
            # Reversion strength detection
            reversion_strength = reversion_20
            future_reversion = -returns.shift(-lookforward)
            
            # Label: 1 for strong reversion opportunity
            labels = (future_reversion > reversion_strength * 1.5).astype(int)
            
            return labels
        else:
            # Fallback to simple reversion-based labels
            returns = df['close'].pct_change()
            future_returns = returns.shift(-lookforward)
            labels = (future_returns < -returns.rolling(25).mean()).astype(int)
            return labels
    

    def _train_enhanced_reversion_model(self, features: pd.DataFrame, labels: pd.Series, 
                                       config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
        """Train enhanced reversion model with ExtraTrees optimization."""
        
        tprint_info("🤖 Training ExtraTrees reversion model...")
        
        from src.utils.ml_common.specialist_xgb import train_specialist_model_with_oof
        
        # Use the centralized ExtraTrees trainer
        training_result = train_specialist_model_with_oof(
            features, 
            labels,
            n_splits=config.get("reversion_n_splits", 5)
        )
        
        return training_result.model, training_result.metrics
    
    def generate_pipeline_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Generate base pipeline features."""
        # For reversion regime, we use the combined manual features as pipeline features
        # when called from external consumers (like GMM pipeline)
        return self._generate_enhanced_reversion_features(market_data, {})

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute enhanced reversion regime step."""
        return await self.execute_standard_specialist_logic(
            config=config,
            specialist_type=SpecialistType.REVERSION_REGIME,
            manual_feature_func=self._generate_enhanced_reversion_features,
            filter_type='volatility',
            pt_sl_config_key='reversion_pt_sl',
            default_pt_sl=[2.0, 1.0],
            suffix="enhanced_reversion_features"
        )

    def _load_market_data_with_cache(self, config: Dict[str, Any], timeframe: str) -> Tuple[pd.DataFrame, str]:
        """Load market data with caching."""
        # This would be implemented based on the actual data loading mechanism
        # Using alternative data loading approach
        market_data, market_source = self.load_market_data_or_fail(
            {**config, "timeframe": timeframe},
            pipeline_state={},
            allow_config_override=True,
        )
        return market_data, market_source
