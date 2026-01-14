"""
Enhanced ML Candlestick Pattern Step with MI Improvements

This enhanced version implements:
- Enhanced feature generation for MI improvement
- Real-time MI monitoring during training
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import TimeSeriesSplit
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


class EnhancedMLCandlestickStep(MLRiskRegimeStepHMM, SpecialistDiagnosticsMixinEnhancedV2, AFMLSpecialistMixin, SpecialistDiagnosticsMixin):

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
            model = 'enhanced_ml_candlestick_step'  # Use the correct model name

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
    Enhanced Candlestick Pattern Specialist with MI optimization.
    
    Key enhancements:
    - Enhanced feature generation for MI improvement
    - Real-time MI monitoring during training
    - Hyperparameter optimization for MI > 0.02
    - Standardized data structure output
    - Binary output enforcement
    """
    
    def __init__(self, step_name: str = "enhanced_ml_candlestick_step"):
        """Initialize the enhanced candlestick step."""
        super().__init__(step_name=step_name)  # Parent class already enables versioned artifacts
        self._current_context = {}
        self._artifact_manager = None
        self._versioned_store = None
        self.step_name = step_name
        self.logger = logger.getChild("EnhancedMLCandlestickStep")
        self.feature_pipeline = MIOptimizedFeaturePipeline()
        self.mi_history = []
        self.training_metrics = []
        
    def _compute_enhanced_structural_optimized_horizon_optimized_candlestick_features(self, df: pd.DataFrame) -> pd.DataFrame:
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
            features[f'price_candlestick_{window}'] = (df['close'] - sma) / sma
            features[f'price_candlestick_cross_{window}'] = (df['close'] > sma).astype(int)
        
        # Volume momentum
        if 'volume' in df.columns:
            volume_change = df['volume'].pct_change()
            for window in [5, 10, 20]:
                features[f'volume_candlestick_{window}'] = volume_change.rolling(window).sum()
                features[f'volume_price_candlestick_{window}'] = (returns * volume_change).rolling(window).sum()
        
        return features
    
    def _generate_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate enhanced candlestick features with optimized performance."""
        # Basic candlestick features
        candlestick_features = self._compute_enhanced_structural_optimized_horizon_optimized_candlestick_features(df)
        
        # Combine features
        all_features = pd.concat([candlestick_features], axis=1)
        
        # Remove duplicate columns
        all_features = all_features.loc[:, ~all_features.columns.duplicated()]
        
        # Limit to top 50 features for performance
        if len(all_features.columns) > 50:
            all_features = all_features.iloc[:, :50]
        
        return all_features
    
    def _create_manual_candlestick_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create manual enhanced features for candlestick pattern analysis."""
        manual_features = pd.DataFrame(index=df.index)
        
        if all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            open_price = df['open']
            high = df['high']
            low = df['low']
            close = df['close']
            volume = df.get('volume', pd.Series(1, index=df.index))
            
            # 1. Enhanced candlestick body features
            # Body size and position
            body_size = abs(close - open_price)
            body_position = (close + open_price) / 2
            upper_shadow = high - np.maximum(close, open_price)
            lower_shadow = np.minimum(close, open_price) - low
            total_range = high - low
            
            manual_features['body_size'] = body_size
            manual_features['body_position'] = body_position
            manual_features['upper_shadow'] = upper_shadow
            manual_features['lower_shadow'] = lower_shadow
            manual_features['total_range'] = total_range
            
            # Body-to-range ratio
            body_ratio = body_size / (total_range + 1e-8)
            manual_features['body_ratio'] = body_ratio
            
            # Upper shadow ratio
            upper_shadow_ratio = upper_shadow / (total_range + 1e-8)
            manual_features['upper_shadow_ratio'] = upper_shadow_ratio
            
            # Lower shadow ratio
            lower_shadow_ratio = lower_shadow / (total_range + 1e-8)
            manual_features['lower_shadow_ratio'] = lower_shadow_ratio
            
            # 2. Candlestick color and direction features
            # Candle color (1 for bullish, 0 for bearish)
            candle_color = (close > open_price).astype(int)
            manual_features['candle_color'] = candle_color
            
            # Color persistence
            color_persistence = candle_color.rolling(5).sum()
            manual_features['color_persistence'] = color_persistence
            
            # Color transitions
            color_transitions = candle_color.diff().abs()
            manual_features['color_transitions'] = color_transitions
            
            # 3. Doji and spinning top detection
            # Doji detection (small body relative to range)
            doji_threshold = total_range * 0.1
            is_doji = body_size < doji_threshold
            manual_features['is_doji'] = is_doji.astype(int)
            
            # Spinning top detection (small body, long shadows)
            spinning_top = (body_size < doji_threshold) & (upper_shadow > body_size) & (lower_shadow > body_size)
            manual_features['is_spinning_top'] = spinning_top.astype(int)
            
            # Hammer detection (small body, long lower shadow, short upper shadow)
            hammer = (body_size < doji_threshold) & (lower_shadow > 2 * body_size) & (upper_shadow < body_size)
            manual_features['is_hammer'] = hammer.astype(int)
            
            # Hanging man detection (small body, long upper shadow, short lower shadow)
            hanging_man = (body_size < doji_threshold) & (upper_shadow > 2 * body_size) & (lower_shadow < body_size)
            manual_features['is_hanging_man'] = hanging_man.astype(int)
            
            # 4. Engulfing pattern detection
            # Bullish engulfing
            bullish_engulfing = (
                (candle_color.shift(1) == 0) &  # Previous candle bearish
                (candle_color == 1) &  # Current candle bullish
                (close > open_price.shift(1)) &  # Current close > previous open
                (open_price < close.shift(1))  # Current open < previous close
            )
            manual_features['bullish_engulfing'] = bullish_engulfing.astype(int)
            
            # Bearish engulfing
            bearish_engulfing = (
                (candle_color.shift(1) == 1) &  # Previous candle bullish
                (candle_color == 0) &  # Current candle bearish
                (close < open_price.shift(1)) &  # Current close < previous open
                (open_price > close.shift(1))  # Current open > previous close
            )
            manual_features['bearish_engulfing'] = bearish_engulfing.astype(int)
            
            # 5. Harami pattern detection
            # Bullish harami
            bullish_harami = (
                (candle_color.shift(1) == 0) &  # Previous candle bearish
                (candle_color == 1) &  # Current candle bullish
                (body_size < body_size.shift(1)) &  # Current body smaller
                (open_price > close.shift(1)) &  # Current open > previous close
                (close < open_price.shift(1))  # Current close < previous open
            )
            manual_features['bullish_harami'] = bullish_harami.astype(int)
            
            # Bearish harami
            bearish_harami = (
                (candle_color.shift(1) == 1) &  # Previous candle bullish
                (candle_color == 0) &  # Current candle bearish
                (body_size < body_size.shift(1)) &  # Current body smaller
                (open_price < close.shift(1)) &  # Current open < previous close
                (close > open_price.shift(1))  # Current close > previous open
            )
            manual_features['bearish_harami'] = bearish_harami.astype(int)
            
            # 6. Morning star and evening star patterns
            # Morning star (3-candle pattern)
            morning_star = (
                (candle_color.shift(2) == 0) &  # First candle bearish
                (body_size.shift(1) < body_size.shift(2) * 0.3) &  # Second candle small body
                (candle_color == 1) &  # Third candle bullish
                (close > (open_price.shift(2) + close.shift(2)) / 2)  # Third close > midpoint of first candle
            )
            manual_features['morning_star'] = morning_star.astype(int)
            
            # Evening star (3-candle pattern)
            evening_star = (
                (candle_color.shift(2) == 1) &  # First candle bullish
                (body_size.shift(1) < body_size.shift(2) * 0.3) &  # Second candle small body
                (candle_color == 0) &  # Third candle bearish
                (close < (open_price.shift(2) + close.shift(2)) / 2)  # Third close < midpoint of first candle
            )
            manual_features['evening_star'] = evening_star.astype(int)
            
            # 7. Tweezer bottom and top patterns
            # Tweezer bottom (2-candle pattern with same lows)
            tweezer_bottom = (
                (candle_color.shift(1) == 0) &  # First candle bearish
                (candle_color == 1) &  # Second candle bullish
                (abs(low - low.shift(1)) < total_range * 0.05)  # Lows almost equal
            )
            manual_features['tweezer_bottom'] = tweezer_bottom.astype(int)
            
            # Tweezer top (2-candle pattern with same highs)
            tweezer_top = (
                (candle_color.shift(1) == 1) &  # First candle bullish
                (candle_color == 0) &  # Second candle bearish
                (abs(high - high.shift(1)) < total_range * 0.05)  # Highs almost equal
            )
            manual_features['tweezer_top'] = tweezer_top.astype(int)
            
            # 8. Volume-adjusted candlestick features
            if 'volume' in df.columns:
                # Volume-weighted body size
                volume_weighted_body = body_size * volume
                manual_features['volume_weighted_body'] = volume_weighted_body
                
                # Volume-weighted range
                volume_weighted_range = total_range * volume
                manual_features['volume_weighted_range'] = volume_weighted_range
                
                # Volume confirmation (high volume confirms pattern)
                volume_ma = volume.rolling(20).mean()
                volume_confirmation = volume > volume_ma * 1.5
                manual_features['volume_confirmation'] = volume_confirmation.astype(int)
                
                # Volume divergence (price moves without volume)
                price_move = abs(close - open_price)
                volume_divergence = (price_move > price_move.rolling(20).mean()) & (volume < volume_ma)
                manual_features['volume_divergence'] = volume_divergence.astype(int)
            
            # 9. Multi-timeframe candlestick features
            # 3-candle trend
            three_candle_trend = ((close > close.shift(1)) & (close.shift(1) > close.shift(2))).astype(int) - \
                               ((close < close.shift(1)) & (close.shift(1) < close.shift(2))).astype(int)
            manual_features['three_candle_trend'] = three_candle_trend
            
            # 5-candle trend
            five_candle_trend = ((close > close.shift(1)) & (close.shift(1) > close.shift(2)) & 
                                (close.shift(2) > close.shift(3)) & (close.shift(3) > close.shift(4))).astype(int) - \
                              ((close < close.shift(1)) & (close.shift(1) < close.shift(2)) & 
                               (close.shift(2) < close.shift(3)) & (close.shift(3) < close.shift(4))).astype(int)
            manual_features['five_candle_trend'] = five_candle_trend
            
            # 10. Composite candlestick indicators
            # Pattern strength index
            pattern_strength = (
                bullish_engulfing.astype(int) + bearish_engulfing.astype(int) +
                morning_star.astype(int) + evening_star.astype(int) +
                hammer.astype(int) + hanging_man.astype(int)
            )
            manual_features['pattern_strength'] = pattern_strength
            
            # Candlestick quality index
            quality_index = (
                0.3 * (body_ratio > 0.6).astype(int) +  # Strong body
                0.3 * (volume_confirmation.astype(int) if 'volume' in df.columns else 0) +  # Volume confirmation
                0.2 * (color_persistence >= 3).astype(int) +  # Consistent direction
                0.2 * (total_range > total_range.rolling(20).mean()).astype(int)  # Significant range
            )
            manual_features['candlestick_quality'] = quality_index
            
        return manual_features
    
    def _apply_manual_candlestick_feature_selection(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply manual feature selection for candlestick features."""
        if features.empty:
            return features
        
        # Remove constant features
        constant_features = features.columns[features.nunique() <= 1]
        if len(constant_features) > 0:
            features = features.drop(columns=constant_features)
            self.logger.info(f"Removed {len(constant_features)} constant candlestick features")
        
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
            self.logger.info(f"Removed {len(set(to_drop))} redundant candlestick features: {list(set(to_drop))}")
        
        # Keep only the most informative features (limit to top 30 by variance)
        if len(features.columns) > 30:
            feature_variances = features.var()
            top_features = feature_variances.nlargest(30).index
            features = features[top_features]
            self.logger.info(f"Limited candlestick features to top 30 by variance")
        
        return features
    
    def _create_candlestick_labels(self, df: pd.DataFrame, lookforward: int = 35) -> pd.Series:
        """Create momentum persistence labels."""
        returns = df['close'].pct_change()
        
        # Future momentum
        future_returns = returns.shift(-lookforward).rolling(lookforward).sum()
        
        # Binary label: positive future momentum
        labels = (future_returns > returns.rolling(25).std() * 0.5).astype(int)
        
        return labels
    
    def _compute_mi_during_training(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                  X_val: pd.DataFrame, y_val: pd.Series,
                                  model_predictions: np.ndarray) -> Dict[str, float]:
        """Compute MI metrics during training for monitoring."""
        mi_metrics = {}
        
        try:
            # Feature MI to target
            feature_mi_scores = []
            for col in X_train.select_dtypes(include=[np.number]).columns:
                mi_score = mutual_info_regression(
                    X_train[col].values.reshape(-1, 1), y_train.values
                )[0]
                feature_mi_scores.append(mi_score)
            
            if feature_mi_scores:
                mi_metrics['avg_feature_mi'] = np.mean(feature_mi_scores)
                mi_metrics['max_feature_mi'] = np.max(feature_mi_scores)
                mi_metrics['high_mi_features'] = sum(1 for mi in feature_mi_scores if mi > 0.02)
            
            # Prediction MI to target
            mi_metrics['prediction_mi'] = mutual_info_regression(
                model_predictions.reshape(-1, 1), y_val.values
            )[0]
            
            # MI improvement tracking
            self.mi_history.append(mi_metrics['prediction_mi'])
            
        except Exception as e:
            self.logger.warning(f"MI computation failed: {e}")
            mi_metrics = {'prediction_mi': 0.0, 'avg_feature_mi': 0.0}
        
        return mi_metrics
    
    def _train_enhanced_candlestick_model(self, features: pd.DataFrame, labels: pd.Series) -> Tuple[ExtraTreesClassifier, Dict[str, float]]:
        """Train enhanced candlestick model with ExtraTrees and specified parameters."""
        
        tprint_info("🤖 Training ExtraTrees specialist model...")
        
        # User-specified parameters for ExtraTrees
        n_features = features.shape[1]
        max_features = int(np.log2(n_features)) if n_features > 1 else 1
        
        params = {
            "n_estimators": 1000,
            "max_features": max_features,
            "min_samples_leaf": 0.02,
            "max_depth": None, # controlled by min_samples_leaf
            "class_weight": "balanced_subsample",
            "criterion": "entropy",
            "n_jobs": -1,
            "random_state": 42
        }
        
        # Time series split for evaluation
        n_splits = 5
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        mi_scores = []
        auc_scores = []
        
        for train_idx, val_idx in tscv.split(features):
            X_train, X_val = features.iloc[train_idx], features.iloc[val_idx]
            y_train, y_val = labels.iloc[train_idx], labels.iloc[val_idx]
            
            model = ExtraTreesClassifier(**params)
            model.fit(X_train, y_train)
            
            # Evaluate
            val_pred = model.predict_proba(X_val)[:, 1]
            
            # Compute MI
            mi_score = mutual_info_regression(
                val_pred.reshape(-1, 1), y_val.values
            )[0]
            mi_scores.append(mi_score)
            
            # Compute AUC
            try:
                auc = roc_auc_score(y_val, val_pred)
            except Exception:
                auc = 0.5
            auc_scores.append(auc)
            
            # Store training metrics
            self.training_metrics.append({
                'mi_score': mi_score,
                'auc_score': auc,
                'n_features': n_features
            })
        
        # Final model on full data
        final_model = ExtraTreesClassifier(**params)
        final_model.fit(features, labels)
        
        metrics = {
            'mi_score': np.mean(mi_scores),
            'mi_std': np.std(mi_scores),
            'auc': np.mean(auc_scores),
            'auc_std': np.std(auc_scores),
            'best_mi': np.max(mi_scores),
            'best_auc': np.max(auc_scores),
            'n_features': n_features,
            'optimization_params': params
        }
        
        return final_model, metrics
    
    def _get_candlestick_combined_manual_features(self, df: pd.DataFrame, pipeline_features: pd.DataFrame) -> pd.DataFrame:
        """Combine candlestick features and manual enhancements."""
        # 1. Base Candlestick Features
        candlestick_features = self._generate_enhanced_features(df)
        
        # 2. Manual Features
        manual_features = self._create_manual_candlestick_enhanced_features(df)
        
        # Apply manual feature selection
        combined = pd.concat([candlestick_features, manual_features], axis=1)
        return self._apply_manual_candlestick_feature_selection(combined)

    def generate_pipeline_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Generate base pipeline features."""
        # For candlestick, we use the combined manual features as pipeline features
        # when called from external consumers (like GMM pipeline)
        return self._get_candlestick_combined_manual_features(market_data, pd.DataFrame(index=market_data.index))

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute enhanced candlestick step."""
        return await self.execute_standard_specialist_logic(
            config=config,
            specialist_type=SpecialistType.CANDLESTICK,
            manual_feature_func=self._get_candlestick_combined_manual_features,
            filter_type='price',
            pt_sl_config_key='candlestick_pt_sl',
            default_pt_sl=[2.0, 1.0],
            suffix="enhanced_candlestick_features"
        )
    
    def _load_market_data(self, symbol: str, exchange: str, timeframe: str) -> pd.DataFrame:
        """Load market data using BaseStep method."""
        config = {"symbol": symbol, "exchange": exchange, "timeframe": timeframe}
        market_data, _market_source = self.load_market_data_or_fail(
            config,
            pipeline_state={},
            allow_config_override=True,
        )
        return market_data
