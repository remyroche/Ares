"""
Analyst Layer 3 Gate Training

Trains ExtraTrees gate model on meta model OOF predictions + regime/performance features.
The gate model filters out toxic trades and preserves capital during adverse regimes.

Input Features:
- Meta model OOF predictions (confidence >= 0.4 only)
- Regime features (volatility, trend, volume)
- Model performance features (rolling win rate, consecutive losses)
- Basic disagreement features

Layer 3 Success Criteria:
- PnL increases overall
- Delta Max Drawdown: Reduces MDD by > 20%
- Delta Sortino: Increases Sortino by > 0.5
- Rejection Balance: Loss avoided by rejected losers > 1.2 * gain missed by rejected winners
- Gating Frequency: Active 5-30% of time
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from scipy import stats

try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    def tprint_info(*args, **kwargs): print(*args)
    def tprint_success(*args, **kwargs): print(*args)
    def tprint_warning(*args, **kwargs): print(*args)
    def tprint_error(*args, **kwargs): print(*args)

from src.training.steps.model_training.analyst_multi_layer_metrics import (
    LayerMetrics, CalibrationMetrics, TradingMetrics, RiskMetrics,
    PredictiveMetrics, DiversityMetrics, StabilityMetrics, ActivityMetrics, GateMetrics,
    MultiLayerMetricsReporter,
    compute_calibration_metrics, compute_predictive_metrics,
    compute_trading_metrics, compute_risk_metrics, compute_gate_metrics,
    generate_layer_markdown_report
)


@dataclass
class GateModelConfig:
    """Configuration for gate model."""
    name: str = "gate_extratrees"
    n_estimators: int = 500
    max_depth: int = 5
    min_samples_leaf: int = 50
    max_features: str = "sqrt"
    bootstrap: bool = True
    class_weight: str = "balanced"
    random_state: int = 42
    min_win_probability: float = 0.55  # Minimum confidence to allow trade
    meta_confidence_threshold: float = 0.4  # Only train on samples with meta confidence >= this


class RegimeFeatureGenerator:
    """
    Generates regime features for gate model.
    
    Features include:
    - Volatility regime (high/mid/low based on rolling volatility)
    - Trend regime (up/down/sideways based on moving averages)
    - Volume regime (high/low based on rolling volume)
    """
    
    def __init__(
        self,
        vol_short_window: int = 12,
        vol_med_window: int = 48,
        vol_long_window: int = 200,
        trend_short_window: int = 20,
        trend_long_window: int = 50,
        volume_window: int = 20
    ):
        """
        Initialize the generator.
        
        Args:
            vol_short_window: Short-term volatility window
            vol_med_window: Medium-term volatility window
            vol_long_window: Long-term volatility window
            trend_short_window: Short-term trend window
            trend_long_window: Long-term trend window
            volume_window: Volume lookback window
        """
        self.vol_short_window = vol_short_window
        self.vol_med_window = vol_med_window
        self.vol_long_window = vol_long_window
        self.trend_short_window = trend_short_window
        self.trend_long_window = trend_long_window
        self.volume_window = volume_window
    
    def generate(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Generate regime features from OHLCV data.
        
        Args:
            ohlcv: DataFrame with close, high, low, volume columns
            
        Returns:
            DataFrame with regime features
        """
        features = pd.DataFrame(index=ohlcv.index)
        
        if 'close' not in ohlcv.columns:
            tprint_warning("⚠️ 'close' column not found, skipping regime features")
            return features
        
        close = ohlcv['close']
        
        # Log returns
        log_ret = np.log(close / close.shift(1))
        
        # Volatility features
        rv_short = log_ret.rolling(self.vol_short_window).std() * np.sqrt(252)
        rv_med = log_ret.rolling(self.vol_med_window).std() * np.sqrt(252)
        rv_long = log_ret.rolling(self.vol_long_window).std() * np.sqrt(252)
        
        features['rv_short'] = rv_short
        features['rv_short_over_med'] = rv_short / (rv_med + 1e-8)
        features['rv_z_score'] = (rv_short - rv_long) / (rv_long + 1e-8)
        
        # Volatility regime buckets
        rv_quantiles = rv_short.rolling(200, min_periods=50).quantile(0.5)
        features['vol_regime_high'] = (rv_short > rv_quantiles * 1.5).astype(int)
        features['vol_regime_low'] = (rv_short < rv_quantiles * 0.5).astype(int)
        features['vol_regime_mid'] = 1 - features['vol_regime_high'] - features['vol_regime_low']
        
        # Trend features
        sma_short = close.rolling(self.trend_short_window).mean()
        sma_long = close.rolling(self.trend_long_window).mean()
        
        features['trend_strength'] = (sma_short - sma_long) / (sma_long + 1e-8)
        features['trend_direction'] = np.sign(sma_short - sma_long)
        
        # Trend regime buckets
        features['trend_up'] = (features['trend_strength'] > 0.01).astype(int)
        features['trend_down'] = (features['trend_strength'] < -0.01).astype(int)
        features['trend_sideways'] = 1 - features['trend_up'] - features['trend_down']
        
        # Momentum features
        features['momentum_12'] = close.pct_change(12)
        features['momentum_24'] = close.pct_change(24)
        
        # Volume features (if available)
        if 'volume' in ohlcv.columns:
            volume = ohlcv['volume']
            vol_ma = volume.rolling(self.volume_window).mean()
            vol_std = volume.rolling(self.volume_window).std()
            
            features['volume_ratio'] = volume / (vol_ma + 1e-8)
            features['volume_z'] = (volume - vol_ma) / (vol_std + 1e-8)
            
            # Volume regime
            features['volume_high'] = (volume > vol_ma * 1.5).astype(int)
            features['volume_low'] = (volume < vol_ma * 0.5).astype(int)
        
        # ATR-based features (if high/low available)
        if 'high' in ohlcv.columns and 'low' in ohlcv.columns:
            high = ohlcv['high']
            low = ohlcv['low']
            tr = (high - low) / close
            features['atr_pct'] = tr.rolling(14).mean()
            features['atr_ratio'] = tr / (tr.rolling(100, min_periods=20).mean() + 1e-8)
        
        return features


class ModelPerformanceFeatureGenerator:
    """
    Generates model performance features for gate model.
    
    Features include:
    - Rolling win rate
    - Rolling average PnL
    - Consecutive wins/losses
    - Time since last trade
    """
    
    def __init__(
        self,
        short_window: int = 5,
        med_window: int = 20,
        long_window: int = 50
    ):
        """
        Initialize the generator.
        
        Args:
            short_window: Short-term lookback
            med_window: Medium-term lookback
            long_window: Long-term lookback
        """
        self.short_window = short_window
        self.med_window = med_window
        self.long_window = long_window
    
    def generate(
        self,
        predictions: pd.Series,
        returns: pd.Series,
        threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Generate model performance features.
        
        Args:
            predictions: Model predictions
            returns: Actual returns
            threshold: Prediction threshold for trade decision
            
        Returns:
            DataFrame with performance features
        """
        features = pd.DataFrame(index=predictions.index)
        
        # Trade decision
        trade = (predictions >= threshold).astype(int)
        
        # Trade returns (0 if no trade)
        trade_returns = returns * trade
        
        # Win/loss labels
        win = (trade_returns > 0).astype(int)
        loss = (trade_returns < 0).astype(int)
        
        # Rolling win rate
        features['rolling_winrate_5'] = win.rolling(self.short_window, min_periods=1).mean()
        features['rolling_winrate_20'] = win.rolling(self.med_window, min_periods=1).mean()
        features['rolling_winrate_50'] = win.rolling(self.long_window, min_periods=1).mean()
        
        # Rolling average PnL
        features['rolling_pnl_5'] = trade_returns.rolling(self.short_window, min_periods=1).mean()
        features['rolling_pnl_20'] = trade_returns.rolling(self.med_window, min_periods=1).mean()
        
        # Consecutive losses
        loss_streak = loss.groupby((loss != loss.shift()).cumsum()).cumcount() + 1
        features['consecutive_losses'] = loss_streak * loss
        
        # Consecutive wins  
        win_streak = win.groupby((win != win.shift()).cumsum()).cumcount() + 1
        features['consecutive_wins'] = win_streak * win
        
        # Time since last trade
        trade_positions = trade.to_numpy()
        time_since_trade = np.zeros(len(trade_positions))
        last_trade_idx = -1
        
        for i in range(len(trade_positions)):
            if trade_positions[i] == 1:
                last_trade_idx = i
            if last_trade_idx >= 0:
                time_since_trade[i] = i - last_trade_idx
            else:
                time_since_trade[i] = i + 1
        
        features['time_since_last_trade'] = time_since_trade
        
        # Rolling max drawdown (simple version)
        cumulative = (1 + trade_returns).cumprod()
        rolling_max = cumulative.rolling(self.long_window, min_periods=1).max()
        features['rolling_drawdown'] = (cumulative - rolling_max) / (rolling_max + 1e-8)
        
        return features


class GateModelTrainer:
    """
    Trains ExtraTrees gate model.
    """
    
    def __init__(
        self,
        config: GateModelConfig,
        burn_in_periods: int = 100
    ):
        """
        Initialize the trainer.
        
        Args:
            config: Gate model configuration
            burn_in_periods: Number of periods to exclude as burn-in
        """
        self.config = config
        self.burn_in_periods = burn_in_periods
        
        self.model = None
        self.calibrator = None
        self.feature_names: List[str] = []
        self.oof_predictions: Optional[pd.Series] = None
        self.oof_gate_decisions: Optional[pd.Series] = None
        self.training_metrics: Dict[str, Any] = {}
    
    def _create_model(self) -> ExtraTreesClassifier:
        """Create the ExtraTrees classifier."""
        return ExtraTreesClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            max_features=self.config.max_features,
            bootstrap=self.config.bootstrap,
            class_weight=self.config.class_weight,
            random_state=self.config.random_state,
            n_jobs=-1
        )
    
    def _prepare_features(
        self,
        meta_predictions: pd.Series,
        regime_features: pd.DataFrame,
        performance_features: pd.DataFrame,
        disagreement_features: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Prepare features for gate model.
        
        Args:
            meta_predictions: Meta model OOF predictions
            regime_features: Regime features
            performance_features: Model performance features
            disagreement_features: Basic disagreement features
            
        Returns:
            Combined feature DataFrame
        """
        features = pd.DataFrame(index=meta_predictions.index)
        
        # Meta prediction as feature
        features['meta_prediction'] = meta_predictions
        features['meta_confidence'] = np.abs(meta_predictions - 0.5) * 2  # Normalize to 0-1
        
        # Add regime features
        for col in regime_features.columns:
            features[f'regime_{col}'] = regime_features[col]
        
        # Add performance features
        for col in performance_features.columns:
            features[f'perf_{col}'] = performance_features[col]
        
        # Add disagreement features (basic ones for gate)
        disagree_cols = ['disagree_mean', 'disagree_std', 'disagree_max', 'disagree_min']
        for col in disagree_cols:
            if col in disagreement_features.columns:
                features[col] = disagreement_features[col]
        
        return features
    
    def _create_target(
        self,
        returns: pd.Series,
        meta_predictions: pd.Series,
        threshold: float = 0.5
    ) -> pd.Series:
        """
        Create binary target for gate model.
        
        Target = 1 if trade was profitable, 0 otherwise.
        
        Args:
            returns: Actual returns
            meta_predictions: Meta model predictions
            threshold: Prediction threshold
            
        Returns:
            Binary target series
        """
        # Trade decision based on meta predictions
        trade = (meta_predictions >= threshold).astype(int)
        
        # Target: was the trade profitable?
        trade_returns = returns * trade
        target = (trade_returns > 0).astype(int)
        
        # For non-trades, we use the underlying return direction
        non_trade_mask = trade == 0
        target[non_trade_mask] = (returns[non_trade_mask] > 0).astype(int)
        
        return target
    
    def train_walk_forward(
        self,
        meta_predictions: pd.Series,
        regime_features: pd.DataFrame,
        performance_features: pd.DataFrame,
        disagreement_features: pd.DataFrame,
        returns: pd.Series,
        n_splits: int = 5,
        embargo_periods: int = 10
    ) -> Tuple[pd.Series, pd.Series, Dict[str, Any]]:
        """
        Train using walk-forward validation with OOF predictions.
        
        Args:
            meta_predictions: Meta model OOF predictions
            regime_features: Regime features
            performance_features: Model performance features
            disagreement_features: Basic disagreement features
            returns: Actual returns
            n_splits: Number of walk-forward splits
            embargo_periods: Embargo periods between train and val
            
        Returns:
            (OOF gate probabilities, OOF gate decisions, training metrics)
        """
        start_time = time.time()
        
        # Prepare features
        X = self._prepare_features(
            meta_predictions,
            regime_features,
            performance_features,
            disagreement_features
        )
        self.feature_names = list(X.columns)
        
        # Create target
        y = self._create_target(returns, meta_predictions)
        
        n_samples = len(X)
        
        # Filter to samples with high enough meta confidence
        confidence = np.abs(meta_predictions - 0.5) * 2
        confidence_mask = confidence >= (self.config.meta_confidence_threshold * 2 - 1)
        
        # Valid samples
        valid_mask = ~X.isna().any(axis=1) & ~y.isna() & confidence_mask
        valid_mask[:self.burn_in_periods] = False
        valid_indices = np.where(valid_mask)[0]
        
        tprint_info(f"[{self.config.name}] Valid samples (confidence >= {self.config.meta_confidence_threshold}): "
                   f"{len(valid_indices)}/{n_samples}")
        
        # Initialize OOF predictions
        oof_probs = np.full(n_samples, np.nan)
        fold_metrics: List[Dict[str, float]] = []
        
        # Walk-forward splits
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(valid_indices)):
            train_idx = valid_indices[train_idx]
            val_idx = valid_indices[val_idx]
            
            # Apply embargo
            if embargo_periods > 0 and len(train_idx) > embargo_periods:
                train_idx = train_idx[:-embargo_periods]
            
            tprint_info(f"[{self.config.name}] Fold {fold_idx + 1}/{n_splits}: "
                       f"Train={len(train_idx)}, Val={len(val_idx)}")
            
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            # Train model
            model = self._create_model()
            model.fit(X_train, y_train)
            
            # Predict
            fold_probs = model.predict_proba(X_val)[:, 1]
            oof_probs[val_idx] = fold_probs
            
            # Save model from last fold
            if fold_idx == n_splits - 1:
                self.model = model
            
            # Compute fold metrics
            if len(np.unique(y_val)) > 1:
                from sklearn.metrics import roc_auc_score
                fold_auc = roc_auc_score(y_val, fold_probs)
            else:
                fold_auc = 0.5
            
            fold_metrics.append({
                "fold": fold_idx + 1,
                "auc": fold_auc,
                "n_train": len(train_idx),
                "n_val": len(val_idx)
            })
        
        training_time = time.time() - start_time
        
        # Create OOF Series
        self.oof_predictions = pd.Series(
            oof_probs, index=X.index,
            name=f"{self.config.name}_prob"
        )
        
        # Gate decisions
        self.oof_gate_decisions = pd.Series(
            (oof_probs >= self.config.min_win_probability).astype(int),
            index=X.index,
            name=f"{self.config.name}_decision"
        )
        
        # Aggregate training metrics
        self.training_metrics = {
            "model_name": self.config.name,
            "n_splits": n_splits,
            "mean_auc": np.mean([m["auc"] for m in fold_metrics]),
            "std_auc": np.std([m["auc"] for m in fold_metrics]),
            "fold_metrics": fold_metrics,
            "training_time_sec": training_time,
            "n_features": len(self.feature_names),
            "n_valid_samples": len(valid_indices),
            "confidence_threshold": self.config.meta_confidence_threshold
        }
        
        tprint_success(f"[{self.config.name}] Training complete: "
                      f"AUC={self.training_metrics['mean_auc']:.4f} ± {self.training_metrics['std_auc']:.4f}")
        
        return self.oof_predictions, self.oof_gate_decisions, self.training_metrics
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict using trained gate model.
        
        Args:
            X: Features DataFrame
            
        Returns:
            (probabilities, gate decisions)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        probs = self.model.predict_proba(X)[:, 1]
        decisions = (probs >= self.config.min_win_probability).astype(int)
        
        return probs, decisions
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        if self.model is None:
            return pd.DataFrame()
        
        importances = self.model.feature_importances_
        
        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances
        })
        
        return df.sort_values("importance", ascending=False).reset_index(drop=True)


class Layer3Orchestrator:
    """
    Orchestrates training of Layer 3 gate model.
    
    Manages:
    - Regime feature generation
    - Performance feature generation
    - Gate model training
    - Before/after comparison
    - Metrics reporting
    """
    
    def __init__(
        self,
        gate_config: Optional[GateModelConfig] = None,
        reporter: Optional[MultiLayerMetricsReporter] = None,
        symbol: str = "UNKNOWN",
        exchange: str = "binance",
        timeframe: str = "15m",
        direction: str = "long"
    ):
        """
        Initialize the orchestrator.
        
        Args:
            gate_config: Gate model configuration
            reporter: Metrics reporter instance
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            direction: Trading direction
        """
        self.gate_config = gate_config or GateModelConfig()
        self.reporter = reporter or MultiLayerMetricsReporter()
        
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        self.direction = direction
        
        self.regime_generator = RegimeFeatureGenerator()
        self.performance_generator = ModelPerformanceFeatureGenerator()
        
        self.trainer: Optional[GateModelTrainer] = None
        self.gate_metrics: Optional[LayerMetrics] = None
        
        # For comparison
        self.pre_gate_metrics: Dict[str, float] = {}
        self.post_gate_metrics: Dict[str, float] = {}
    
    def train_gate_model(
        self,
        meta_predictions: pd.Series,
        disagreement_features: pd.DataFrame,
        ohlcv: pd.DataFrame,
        returns: pd.Series,
        n_splits: int = 5
    ) -> Tuple[pd.Series, pd.Series, Dict[str, Any]]:
        """
        Train gate model and collect OOF predictions.
        
        Args:
            meta_predictions: Meta model OOF predictions
            disagreement_features: Disagreement features from Layer 2
            ohlcv: OHLCV data for regime features
            returns: Actual returns
            n_splits: Number of walk-forward splits
            
        Returns:
            (OOF gate probabilities, OOF gate decisions, training metrics)
        """
        tprint_info("=" * 80)
        tprint_info("LAYER 3: TRAINING GATE MODEL")
        tprint_info("=" * 80)
        
        # Generate regime features
        tprint_info("\n🔧 Generating regime features...")
        regime_features = self.regime_generator.generate(ohlcv)
        tprint_success(f"✅ Generated {regime_features.shape[1]} regime features")
        
        # Generate performance features
        tprint_info("🔧 Generating model performance features...")
        performance_features = self.performance_generator.generate(
            meta_predictions,
            returns,
            threshold=0.5
        )
        tprint_success(f"✅ Generated {performance_features.shape[1]} performance features")
        
        # Store pre-gate metrics
        self._compute_pre_gate_metrics(meta_predictions, returns)
        
        # Create trainer
        self.trainer = GateModelTrainer(
            config=self.gate_config,
            burn_in_periods=100
        )
        
        # Train gate model
        gate_probs, gate_decisions, training_metrics = self.trainer.train_walk_forward(
            meta_predictions,
            regime_features,
            performance_features,
            disagreement_features,
            returns,
            n_splits=n_splits
        )
        
        # Compute comprehensive metrics
        self.gate_metrics = self._compute_layer_metrics(
            gate_probs,
            gate_decisions,
            meta_predictions,
            returns,
            training_metrics
        )
        
        self.reporter.record_metrics(self.gate_metrics)
        
        # Generate markdown report
        report_path = self.reporter.output_dir / f"L3_{self.gate_config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        generate_layer_markdown_report(self.gate_metrics, str(report_path))
        
        # Print comparison
        self._print_gate_comparison(gate_decisions, returns)
        
        return gate_probs, gate_decisions, training_metrics
    
    def _compute_pre_gate_metrics(
        self,
        meta_predictions: pd.Series,
        returns: pd.Series
    ) -> None:
        """Compute metrics before gating."""
        valid_mask = ~meta_predictions.isna() & ~returns.isna()
        preds = meta_predictions[valid_mask].values
        rets = returns[valid_mask].values
        
        # Simulate trading without gate
        trade_mask = preds >= 0.5
        trade_returns = rets[trade_mask]
        
        if len(trade_returns) > 0:
            risk = compute_risk_metrics(trade_returns)
            trading = compute_trading_metrics(preds[trade_mask], trade_returns)
            
            self.pre_gate_metrics = {
                "sortino": risk.sortino_ratio,
                "max_drawdown": risk.max_drawdown,
                "profit_factor": trading.profit_factor,
                "win_rate": trading.win_rate,
                "total_pnl": np.sum(trade_returns),
                "n_trades": len(trade_returns)
            }
        else:
            self.pre_gate_metrics = {
                "sortino": 0,
                "max_drawdown": 0,
                "profit_factor": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "n_trades": 0
            }
    
    def _compute_layer_metrics(
        self,
        gate_probs: pd.Series,
        gate_decisions: pd.Series,
        meta_predictions: pd.Series,
        returns: pd.Series,
        training_metrics: Dict[str, Any]
    ) -> LayerMetrics:
        """Compute comprehensive Layer 3 metrics."""
        # Align data
        valid_mask = ~gate_probs.isna() & ~meta_predictions.isna() & ~returns.isna()
        gate_valid = gate_probs[valid_mask].values
        decisions_valid = gate_decisions[valid_mask].values
        meta_valid = meta_predictions[valid_mask].values
        returns_valid = returns[valid_mask].values
        
        # Create binary target (was trade profitable?)
        trade_mask = meta_valid >= 0.5
        actual_profit = (returns_valid > 0).astype(int)
        
        # Calibration on gate predictions
        calibration = compute_calibration_metrics(
            actual_profit[trade_mask] if trade_mask.sum() > 0 else actual_profit,
            gate_valid[trade_mask] if trade_mask.sum() > 0 else gate_valid
        )
        
        # Predictive metrics
        predictive = compute_predictive_metrics(
            actual_profit[trade_mask] if trade_mask.sum() > 0 else actual_profit,
            gate_valid[trade_mask] if trade_mask.sum() > 0 else gate_valid
        )
        
        # Trading metrics after gating
        final_trade_mask = (meta_valid >= 0.5) & (decisions_valid == 1)
        gated_returns = returns_valid[final_trade_mask]
        
        if len(gated_returns) > 0:
            trading = compute_trading_metrics(
                meta_valid[final_trade_mask],
                gated_returns
            )
            risk = compute_risk_metrics(gated_returns)
        else:
            trading = TradingMetrics()
            risk = RiskMetrics()
        
        # Compute gate-specific metrics
        gate = compute_gate_metrics(
            meta_valid,
            meta_valid * decisions_valid,
            returns_valid,
            decisions_valid
        )
        
        # Also compute from pre-gate comparison
        gate.delta_max_drawdown = self.pre_gate_metrics.get("max_drawdown", 0) - risk.max_drawdown
        gate.delta_sortino = risk.sortino_ratio - self.pre_gate_metrics.get("sortino", 0)
        gate.gating_frequency = 1 - np.mean(decisions_valid[trade_mask]) if trade_mask.sum() > 0 else 0
        
        # Store post-gate metrics
        self.post_gate_metrics = {
            "sortino": risk.sortino_ratio,
            "max_drawdown": risk.max_drawdown,
            "profit_factor": trading.profit_factor,
            "win_rate": trading.win_rate,
            "total_pnl": np.sum(gated_returns) if len(gated_returns) > 0 else 0,
            "n_trades": len(gated_returns)
        }
        
        # Create LayerMetrics
        metrics = LayerMetrics(
            model_name=self.gate_config.name,
            layer="L3_gate",
            timestamp=datetime.now().isoformat(),
            symbol=self.symbol,
            exchange=self.exchange,
            timeframe=self.timeframe,
            direction=self.direction,
            model_type="extratrees",
            calibration=calibration,
            stability=StabilityMetrics(),
            trading=trading,
            risk=risk,
            predictive=predictive,
            activity=ActivityMetrics(),
            diversity=DiversityMetrics(),
            gate=gate,
            n_samples=int(valid_mask.sum()),
            n_features=training_metrics.get("n_features", 0),
            training_duration_sec=training_metrics.get("training_time_sec", 0),
            notes=f"Gating freq: {gate.gating_frequency:.2%}, Delta MDD: {gate.delta_max_drawdown:+.4f}"
        )
        
        return metrics
    
    def _print_gate_comparison(
        self,
        gate_decisions: pd.Series,
        returns: pd.Series
    ) -> None:
        """Print before/after gate comparison."""
        tprint_info("\n" + "=" * 80)
        tprint_info("GATE MODEL IMPACT ANALYSIS")
        tprint_info("=" * 80)
        
        pre = self.pre_gate_metrics
        post = self.post_gate_metrics
        
        tprint_info(f"\n{'Metric':<25} {'Before Gate':<15} {'After Gate':<15} {'Change':<15}")
        tprint_info("-" * 70)
        
        # Sortino
        delta = post['sortino'] - pre['sortino']
        tprint_info(f"{'Sortino Ratio':<25} {pre['sortino']:<15.4f} {post['sortino']:<15.4f} {delta:+.4f}")
        
        # Max Drawdown
        delta = pre['max_drawdown'] - post['max_drawdown']  # Positive is improvement
        tprint_info(f"{'Max Drawdown':<25} {pre['max_drawdown']:<15.4f} {post['max_drawdown']:<15.4f} {delta:+.4f}")
        
        # Profit Factor
        delta = post['profit_factor'] - pre['profit_factor']
        tprint_info(f"{'Profit Factor':<25} {pre['profit_factor']:<15.4f} {post['profit_factor']:<15.4f} {delta:+.4f}")
        
        # Win Rate
        delta = post['win_rate'] - pre['win_rate']
        tprint_info(f"{'Win Rate':<25} {pre['win_rate']:<15.4f} {post['win_rate']:<15.4f} {delta:+.4f}")
        
        # Total PnL
        delta = post['total_pnl'] - pre['total_pnl']
        tprint_info(f"{'Total PnL':<25} {pre['total_pnl']:<15.6f} {post['total_pnl']:<15.6f} {delta:+.6f}")
        
        # Trade count
        delta = post['n_trades'] - pre['n_trades']
        tprint_info(f"{'Trade Count':<25} {pre['n_trades']:<15} {post['n_trades']:<15} {delta:+}")
        
        # Success criteria check
        tprint_info("\n📋 Layer 3 Success Criteria Check:")
        
        mdd_improvement = (pre['max_drawdown'] - post['max_drawdown']) / max(0.01, pre['max_drawdown'])
        mdd_pass = mdd_improvement >= 0.20
        tprint_info(f"   MDD Reduction: {mdd_improvement*100:.1f}% (target: >20%) {'✅' if mdd_pass else '❌'}")
        
        sortino_delta = post['sortino'] - pre['sortino']
        sortino_pass = sortino_delta >= 0.5
        tprint_info(f"   Sortino Increase: {sortino_delta:+.4f} (target: >0.5) {'✅' if sortino_pass else '❌'}")
        
        gate_freq = 1 - (post['n_trades'] / max(1, pre['n_trades']))
        freq_pass = 0.05 <= gate_freq <= 0.30
        tprint_info(f"   Gating Frequency: {gate_freq*100:.1f}% (target: 5-30%) {'✅' if freq_pass else '❌'}")
        
        pnl_increase = post['total_pnl'] > pre['total_pnl']
        tprint_info(f"   PnL Increase: {pnl_increase} {'✅' if pnl_increase else '❌'}")
    
    def get_gate_trainer(self) -> GateModelTrainer:
        """Get the trained gate model trainer."""
        return self.trainer
    
    def get_comparison_results(self) -> Dict[str, Dict[str, float]]:
        """Get before/after comparison results."""
        return {
            "pre_gate": self.pre_gate_metrics,
            "post_gate": self.post_gate_metrics
        }
