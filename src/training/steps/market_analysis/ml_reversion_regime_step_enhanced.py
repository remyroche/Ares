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
from src.training.steps.market_analysis.enhanced_feature_generators import MIOptimizedFeaturePipeline
from src.training.steps.market_analysis.specialist_interface import SpecialistDataInterface
from src.training.steps.market_analysis.specialist_data_standard import SpecialistType

logger = logging.getLogger(__name__)


class EnhancedMLReversionRegimeStep(SpecialistDiagnosticsMixinEnhancedV2, BaseStep):

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
        
        # Combine all features
        all_features = [base_reversion_features, enhanced_features, manual_features]
        
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
            reversion_signal_20 = (close - close.rolling(20).mean()) / close.rolling(20).mean()
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
            reversion_strength_index = (
                0.3 * (abs(reversion_signal_20) > 0.02).astype(int) +
                0.3 * (abs(reversion_signal_50) > 0.03).astype(int) +
                0.2 * (rsi_overbought_14 | rsi_oversold_14).astype(int) +
                0.2 * (stoch_overbought_14 | stoch_oversold_14).astype(int)
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
    

    def save(self, artifact_name: str, data, artifact_type: str = "data", data_category: str = "predictions"):
        """Custom save method for enhanced specialists."""
        try:
            # Use versioned store directly if available
            if hasattr(self, '_versioned_store') and self._versioned_store is not None:
                context = {
                    'symbol': self._current_context.get('symbol', 'UNKNOWN'),
                    'exchange': self._current_context.get('exchange', 'binance'),
                    'timeframe': self._current_context.get('timeframe', '15m'),
                    'direction': self._current_context.get('direction', 'long'),
                    'model': self._current_context.get('model', 'analyst'),
                    'step_name': self.step_name,
                }
                self._versioned_store.save(
                    artifact_name=artifact_name,
                    data=data,
                    artifact_type=artifact_type,
                    data_category=data_category,
                    context=context
                )
                self.logger.info(f"✅ Saved {artifact_name} to versioned store")
            else:
                self.logger.warning(f"⚠️ Cannot save {artifact_name}: no versioned store available")
        except Exception as e:
            self.logger.error(f"❌ Failed to save {artifact_name}: {e}")

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
                    use_label_encoder=False,
                    **params
                )
                
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                         early_stopping_rounds=20, verbose=False)
                
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
        values = list(param_grid.values)
        
        # Generate all combinations
        all_combinations = list(itertools.product(*values))
        
        # Randomly sample if too many
        if len(all_combinations) > max_combinations:
            all_combinations = random.sample(all_combinations, max_combinations)
        
        for combination in all_combinations:
            yield dict(zip(keys, combination))
    
    def _train_enhanced_reversion_model(self, features: pd.DataFrame, labels: pd.Series, 
                                       config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
        """Train enhanced reversion model with MI optimization."""
        
        # Optimize hyperparameters for MI
        tprint_info("🔧 Optimizing XGBoost hyperparameters for MI improvement...")
        best_params, best_mi = self._optimize_xgb_hyperparameters_for_mi(features, labels)
        
        # Create temporal split config
        temporal_config = create_temporal_split_config_for_pipeline(
            symbol=config.get("symbol", "ETHUSDT"),
            exchange=config.get("exchange", "binance"),
            timeframe=config.get("timeframe", "15m"),
            direction=config.get("direction", "long"),
            n_splits=config.get("reversion_n_splits", 5),
            walk_forward_type="rolling",
            test_size_ratio=config.get("reversion_test_size_ratio", 0.2),
            min_train_samples=config.get("reversion_min_train_samples", 500),
        )
        
        # Create training config
        training_config = XGBTrainingConfig(
            objective="binary:logistic",
            random_state=42,
            **best_params
        )
        
        # Train with standardized trainer
        trainer = StandardizedXGBTrainer(training_config)
        train_result = trainer.train_time_series_cv(features, labels, temporal_config)
        
        # Extract best model
        best_model = train_result.models[-1] if train_result.models else None
        
        # Compute MI metrics
        oof_preds = train_result.oof_predictions
        if 'probability' in oof_preds.columns:
            mi_score = mutual_info_regression(
                oof_preds['probability'].values.reshape(-1, 1), 
                labels.loc[oof_preds.index].values
            )[0]
        else:
            mi_score = 0.0
        
        # Store training metrics
        self.training_metrics.append({
            'mi_score': mi_score,
            'n_features': len(features.columns),
            'best_params': best_params
        })
        
        metrics = {
            'mi_score': mi_score,
            'auc': train_result.metrics.get('oof_auc', 0.0),
            'log_loss': train_result.metrics.get('oof_log_loss', 0.0),
            'n_features': len(features.columns),
            'optimization_params': best_params,
            'training_time': train_result.metrics.get('training_time', 0.0)
        }
        
        return best_model, metrics
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute enhanced reversion regime step."""
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
                model="enhanced_reversion_regime",
            )

            tprint_info(f"🚀 Starting Enhanced Reversion Regime for {symbol} on {exchange}")

            # 1. Load Market Data
            market_data, market_source = self._load_market_data_with_cache(config, timeframe)

            # 2. Generate Enhanced Features
            tprint_info("🛠️ Generating Enhanced Reversion Regime features...")
            feature_df = self._generate_enhanced_reversion_features(market_data, config)
            
            tprint_info(f"✅ Enhanced Reversion Regime features: {len(feature_df.columns)} columns")

            if not config.get("is_batch_run", False):
                feature_df_reset = feature_df.reset_index().rename(columns={feature_df.index.name or "index": "timestamp"})
                features_path = self._save_artifact(
                    data=feature_df_reset,
                    artifact_name="enhanced_ml_reversion_features",
                    artifact_type="data",
                    data_category="features",
                    metadata={"source": market_source, "enhanced": True}
                )
                artifacts.append(features_path)

            # 3. Generate Labels
            tprint_info("🎯 Generating Enhanced Reversion Regime labels...")
            labels = self._create_reversion_labels(market_data)

            # Align features and labels
            common_index = feature_df.index.intersection(labels.index)
            X = feature_df.loc[common_index]
            y = labels.loc[common_index]

            # Clean data
            valid_mask = X.notna().all(axis=1) & y.notna()
            X = X.loc[valid_mask]
            y = y.loc[valid_mask]

            if len(X) < 500:
                raise RuntimeError(f"Insufficient valid samples: {len(X)} < 500")

            tprint_info(f"📊 Training Data: {len(X)} samples, {len(X.columns)} features")

            # 4. Train Enhanced Model with MI Optimization
            tprint_info("🤖 Training Enhanced Reversion Regime model with MI optimization...")
            model, model_metrics = self._train_enhanced_reversion_model(X, y, config)
            
            metrics.update(model_metrics)

            # 5. Generate Predictions
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)[:, 1]

            # 6. Create Standardized Output
            standardized_output = self._create_standardized_output(
                X, y, predictions, probabilities, symbol, exchange, timeframe, direction
            )

            # 7. Save Artifacts
            artifact_name = f"enhanced_ml_reversion_predictions_{timeframe}"
            metadata = SpecialistDataInterface.create_standard_metadata(
                specialist_name="EnhancedMLReversionRegimeStep",
                config=config,
                metrics=metrics,
                mi_score=metrics['mi_score'],
                hsic_score=0.0
            )
            
            
            # DEBUG: Check artifact saving setup
            print(f"🐛 DEBUG: About to save artifact: {artifact_name}")
            print(f"🐛 DEBUG: Output df shape: {output_df.shape}")
            print(f"🐛 DEBUG: Artifact router type: {type(self.artifact_router)}")
            print(f"🐛 DEBUG: Versioned store available: {hasattr(self, '_versioned_store') and self._versioned_store is not None}")
            if hasattr(self, '_versioned_store') and self._versioned_store is not None:
                print(f"🐛 DEBUG: Versioned store type: {type(self._versioned_store)}")
            
            self.artifact_router.save(
                artifact_name=artifact_name,
                data=standardized_output,
                metadata=metadata
            )
            artifacts.append(artifact_name)

            # 8. Run Enhanced Diagnostics
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

            # 9. Final Summary
            execution_time = time.time() - start_time
            metrics["execution_time"] = execution_time
            metrics["n_samples"] = len(standardized_output)

            tprint_success(f"✅ Enhanced Reversion Regime completed in {execution_time:.2f}s")
            tprint_info(f"📊 Final Metrics: MI={metrics.get('mi_score', 0):.4f}, AUC={metrics.get('auc', 0):.3f}")

            return {
                "success": True,
                "metrics": metrics,
                "n_samples": len(standardized_output),
                "features": list(X.columns),
                "artifacts": artifacts,
                "diagnostics": diagnostics_result,
                "mi_history": self.mi_history,
                "training_metrics": self.training_metrics,
                "execution_time": execution_time
            }

        except Exception as e:
            self.logger.exception(f"❌ Enhanced Reversion Regime step failed: {e}")
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
        # This would be implemented based on the actual data loading mechanism
        # Using alternative data loading approach
        market_data, market_source = self.load_market_data_or_fail(
            {**config, "timeframe": timeframe},
            pipeline_state={},
            allow_config_override=True,
        )
        return market_data, market_source
