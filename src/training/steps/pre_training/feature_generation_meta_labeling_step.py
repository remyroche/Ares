"""
Feature Generation Meta-Labeling Step.

This step implements meta-labeling as an alternative to the triple barrier method.
Instead of labeling based on price barriers, meta-labels predict whether primary signals
(from technical indicators) will be profitable.

Key differences from triple barrier method:
- Uses primary signals (RSI, MA crossovers, momentum) as the basis
- Meta-labels are binary: 1 = profitable signal, 0 = unprofitable signal
- Trains a meta-model to filter primary signals
- Uses proper time-series CV with purging to avoid lookahead bias
"""

import asyncio
import logging
import json
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import gc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.calibration import CalibratedClassifierCV

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning, tprint_error

logger = logging.getLogger(__name__)


def purge_training_idxs(
    train_idxs: np.ndarray,
    val_start_idx: int,
    val_end_idx: int,
    horizon: int
) -> np.ndarray:
    """
    Remove training indices that would create lookahead bias.

    A training sample at position i uses data up to i and predicts i+horizon.
    We must remove training samples where:
    1. The prediction horizon (i + horizon) reaches into validation period
    2. The entry time falls within validation period

    Args:
        train_idxs: Array of training indices (positions in DataFrame)
        val_start_idx: Start of validation period (inclusive)
        val_end_idx: End of validation period (exclusive)
        horizon: Number of periods the label looks ahead

    Returns:
        Filtered training indices without lookahead bias
    """
    filtered = []
    for i in train_idxs:
        # Drop if prediction horizon reaches into validation
        if (i + horizon) >= val_start_idx and i < val_end_idx:
            continue
        # Drop if entry is in validation window
        if (i >= val_start_idx) and (i < val_end_idx):
            continue
        # Drop if lookahead overlaps validation start
        if (i + horizon) >= val_start_idx and i < val_start_idx:
            continue
        filtered.append(i)

    return np.array(filtered, dtype=int)


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute RSI (Relative Strength Index).

    Args:
        prices: Series of prices (typically close)
        period: RSI period (default 14)

    Returns:
        RSI values (0-100)
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def generate_primary_signals(
    df: pd.DataFrame,
    rsi_period: int = 14,
    sma_fast: int = 10,
    sma_slow: int = 30,
    momentum_period: int = 10,
    rsi_oversold: float = 30,
    rsi_overbought: float = 70,
    momentum_threshold: float = 0.005
) -> pd.DataFrame:
    """
    Generate primary trading signals from technical indicators.

    These signals are NOT the final trading decisions. They are inputs to the
    meta-labeling model which will learn which signals to act on.

    Args:
        df: DataFrame with OHLCV data (must have 'close' column)
        rsi_period: Period for RSI calculation
        sma_fast: Fast moving average period
        sma_slow: Slow moving average period
        momentum_period: Momentum lookback period
        rsi_oversold: RSI level for oversold signal
        rsi_overbought: RSI level for overbought signal
        momentum_threshold: Momentum threshold for signal

    Returns:
        DataFrame with signal columns (rsi, ma, mom, consensus)
        Each signal is in {-1, 0, 1} where:
        - 1 = bullish signal
        - 0 = neutral
        - -1 = bearish signal
    """
    signals = pd.DataFrame(index=df.index)

    # RSI signals
    df_local = df.copy()
    df_local['rsi'] = compute_rsi(df_local['close'], period=rsi_period)
    signals['rsi'] = 0
    signals.loc[df_local['rsi'] < rsi_oversold, 'rsi'] = 1  # Oversold -> bullish
    signals.loc[df_local['rsi'] > rsi_overbought, 'rsi'] = -1  # Overbought -> bearish

    # Moving average crossover signals
    df_local['sma_fast'] = df_local['close'].rolling(sma_fast).mean()
    df_local['sma_slow'] = df_local['close'].rolling(sma_slow).mean()
    signals['ma'] = 0
    signals.loc[df_local['sma_fast'] > df_local['sma_slow'], 'ma'] = 1
    signals.loc[df_local['sma_fast'] < df_local['sma_slow'], 'ma'] = -1

    # Momentum signals
    df_local['momentum'] = df_local['close'].pct_change(momentum_period)
    signals['mom'] = 0
    signals.loc[df_local['momentum'] > momentum_threshold, 'mom'] = 1
    signals.loc[df_local['momentum'] < -momentum_threshold, 'mom'] = -1

    # Consensus signal: majority vote with sign
    signals['consensus'] = signals[['rsi', 'ma', 'mom']].sum(axis=1).apply(np.sign)

    return signals


def create_meta_labels(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    profit_threshold: float = 0.015,
    stop_threshold: float = 0.010,
    horizon: int = 16
) -> pd.Series:
    """
    Create meta-labels: binary labels indicating if a signal leads to profit.

    CRITICAL: Meta-labels must be computed from FIXED primary signals only,
    never from features that are being optimized in CV. This prevents leakage.

    For each bar where we have a consensus signal (non-zero):
    - Look ahead up to 'horizon' bars
    - Check if profit_threshold is hit before stop_threshold
    - Label = 1 if profitable, 0 if stopped out or inconclusive

    Args:
        df: DataFrame with price data (must have 'close')
        signals: DataFrame with signal columns (must have 'consensus')
        profit_threshold: Profit target as fraction (e.g., 0.015 = 1.5%)
        stop_threshold: Stop loss as fraction (e.g., 0.010 = 1.0%)
        horizon: Maximum bars to look ahead

    Returns:
        Series of meta-labels (1/0) with NaN where no signal exists
    """
    meta_labels = pd.Series(index=df.index, dtype=float)
    meta_labels[:] = np.nan

    close_prices = df['close'].values
    consensus_signals = signals['consensus'].values

    for i in range(len(df) - horizon):
        signal = consensus_signals[i]

        # Only create labels where we have a signal
        if signal == 0:
            continue

        entry_price = close_prices[i]

        # Look ahead up to horizon bars
        for j in range(1, horizon + 1):
            if i + j >= len(df):
                break

            future_price = close_prices[i + j]

            if signal > 0:  # Long signal
                pnl = (future_price - entry_price) / entry_price

                # Hit profit target
                if pnl >= profit_threshold:
                    meta_labels.iloc[i] = 1.0
                    break
                # Hit stop loss
                elif pnl <= -stop_threshold:
                    meta_labels.iloc[i] = 0.0
                    break

            elif signal < 0:  # Short signal
                pnl = (entry_price - future_price) / entry_price

                # Hit profit target
                if pnl >= profit_threshold:
                    meta_labels.iloc[i] = 1.0
                    break
                # Hit stop loss
                elif pnl <= -stop_threshold:
                    meta_labels.iloc[i] = 0.0
                    break

        # If we didn't hit either threshold, label as unsuccessful
        if pd.isna(meta_labels.iloc[i]):
            meta_labels.iloc[i] = 0.0

    return meta_labels


def create_meta_features(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    volume_available: bool = True
) -> pd.DataFrame:
    """
    Create features for the meta-model.

    These features help the model decide which primary signals to act on.
    Features should capture:
    - Signal strength/confluence
    - Market context (volatility, trend, volume)
    - Signal quality indicators

    Args:
        df: DataFrame with OHLCV data
        signals: DataFrame with primary signals
        volume_available: Whether volume data is available

    Returns:
        DataFrame of features for meta-model
    """
    features = pd.DataFrame(index=df.index)

    # Signal features
    features['signal_strength'] = signals[['rsi', 'ma', 'mom']].abs().sum(axis=1)
    features['signal_consensus'] = signals['consensus'].abs()
    features['signal_direction'] = signals['consensus']

    # Volatility features
    returns = df['close'].pct_change()
    features['volatility_5'] = returns.rolling(5).std()
    features['volatility_20'] = returns.rolling(20).std()
    features['volatility_ratio'] = features['volatility_5'] / (features['volatility_20'] + 1e-8)

    # Trend features
    features['sma_slope'] = df['close'].rolling(10).mean().pct_change(5)
    features['price_vs_sma20'] = (df['close'] - df['close'].rolling(20).mean()) / (df['close'].rolling(20).mean() + 1e-8)

    # Volume features (if available)
    if volume_available and 'volume' in df.columns:
        features['volume_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-8)
        features['volume_trend'] = df['volume'].rolling(5).mean() / (df['volume'].rolling(20).mean() + 1e-8)
    else:
        features['volume_ratio'] = 1.0
        features['volume_trend'] = 1.0

    # Price momentum features
    features['momentum_5'] = df['close'].pct_change(5)
    features['momentum_10'] = df['close'].pct_change(10)
    features['momentum_20'] = df['close'].pct_change(20)

    # Recent performance
    features['recent_high_distance'] = (df['high'].rolling(20).max() - df['close']) / (df['close'] + 1e-8)
    features['recent_low_distance'] = (df['close'] - df['low'].rolling(20).min()) / (df['close'] + 1e-8)

    return features


def translate_metalabels_to_targets(
    meta_labels: pd.Series,
    signals: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float = 0.5
) -> Tuple[pd.Series, pd.Series]:
    """
    Translate meta-labels and probabilities to target format compatible with
    downstream optimization steps.

    Downstream steps (feature_generation_period_lookback_optimization_step,
    feature_generation_final_feature_selection_step) expect continuous targets.

    We map:
    - Meta-probability * signal_direction -> continuous target
    - High probability + long signal -> positive target
    - High probability + short signal -> negative target
    - Low probability or no signal -> zero target

    Args:
        meta_labels: Binary labels (1/0) for signal quality
        signals: DataFrame with 'consensus' column
        probabilities: Predicted probabilities from meta-model
        threshold: Probability threshold for generating targets

    Returns:
        Tuple of (target_long, target_short) Series compatible with downstream steps
    """
    target_long = pd.Series(0.0, index=meta_labels.index)
    target_short = pd.Series(0.0, index=meta_labels.index)

    consensus = signals['consensus'].values

    for i in range(len(meta_labels)):
        if pd.isna(meta_labels.iloc[i]):
            continue

        prob = probabilities[i] if i < len(probabilities) else 0.5
        signal = consensus[i]

        # Scale probability to target magnitude
        # Higher probability -> stronger target signal
        if prob >= threshold and signal != 0:
            target_magnitude = (prob - threshold) / (1.0 - threshold)

            if signal > 0:  # Long signal
                target_long.iloc[i] = target_magnitude
            elif signal < 0:  # Short signal
                target_short.iloc[i] = target_magnitude

    return target_long, target_short


class FeatureGenerationMetaLabelingStep(BaseStep):
    """
    Feature Generation Meta-Labeling Step.

    Alternative to triple barrier labeling that uses meta-labels.
    Uses the same base class for data loading/saving but implements
    a different labeling methodology.
    """

    def __init__(self, step_name: str = "feature_generation_meta_labeling_step"):
        """Initialize the meta-labeling step."""
        super().__init__(step_name)
        self.logger = system_logger.getChild('FeatureGenerationMetaLabeling')

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute feature generation meta-labeling.

        Args:
            config: Configuration dictionary containing:
                - symbol: Trading symbol (e.g., 'ETHUSDT')
                - exchange: Exchange name (e.g., 'binance')
                - timeframe: Timeframe (e.g., '15m')
                - execution_mode: 'full', 'light', or 'blank'
                - profit_threshold: Profit target (default 0.015)
                - stop_threshold: Stop loss (default 0.010)
                - horizon: Lookahead periods (default 16)

        Returns:
            Dict containing:
            - 'success': bool indicating if step completed successfully
            - 'artifacts': dict of created artifacts
            - 'metrics': dict of performance metrics
            - 'error': error message if step failed (optional)
        """
        start_time = datetime.now()

        # Validate required config
        required = ('symbol', 'exchange', 'timeframe')
        missing = [k for k in required if k not in config or not config[k]]
        if missing:
            error_msg = f"Missing required config keys: {', '.join(missing)}"
            tprint(f"❌ {error_msg}", "ERROR")
            return {'success': False, 'error': error_msg}

        tprint(f"🏷️ Starting meta-labeling for {config['symbol']}", "INFO")

        try:
            # Extract config parameters
            profit_threshold = config.get('profit_threshold', 0.015)
            stop_threshold = config.get('stop_threshold', 0.010)
            horizon = config.get('horizon', 16)

            # Load market data
            tprint("📊 Loading market data...", "INFO")
            from src.utils.data.klines_parquet import get_klines_manager
            klines_manager = get_klines_manager(data_dir=config.get('data_dir', 'historical_data'))

            market_data = klines_manager.read_data(
                symbol=config['symbol'],
                interval=config['timeframe'],
                data_type="processed"
            )

            if market_data is None or market_data.empty:
                raise ValueError(f"No market data available for {config['symbol']} {config['timeframe']}")

            tprint(f"✅ Loaded {len(market_data)} samples", "SUCCESS")

            # Check for required columns
            if 'close' not in market_data.columns:
                raise ValueError("Missing required 'close' column in market data")

            volume_available = 'volume' in market_data.columns

            # STEP 1: Generate primary signals (FIXED - not optimized in CV)
            tprint("🎯 Generating primary signals...", "INFO")
            primary_signals = generate_primary_signals(market_data)

            # STEP 2: Create meta-labels from FIXED signals
            tprint("🏷️ Creating meta-labels...", "INFO")
            meta_labels = create_meta_labels(
                market_data,
                primary_signals,
                profit_threshold=profit_threshold,
                stop_threshold=stop_threshold,
                horizon=horizon
            )

            # Count labeled samples
            labeled_mask = ~meta_labels.isna()
            n_labeled = labeled_mask.sum()
            n_positive = (meta_labels == 1.0).sum()
            n_negative = (meta_labels == 0.0).sum()

            tprint(f"📊 Meta-labels: {n_labeled} total ({n_positive} profitable, {n_negative} unprofitable)", "INFO")

            if n_labeled < 100:
                tprint("⚠️ Warning: Very few labeled samples, results may be unreliable", "WARNING")

            # STEP 3: Create features for meta-model
            tprint("🔧 Creating meta-features...", "INFO")
            meta_features = create_meta_features(market_data, primary_signals, volume_available)

            # STEP 4: Train meta-model with proper time-series CV and purging
            tprint("🎓 Training meta-model with time-series CV...", "INFO")

            outer_cv = TimeSeriesSplit(n_splits=5)
            cv_results = []

            for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(market_data)):
                # Purge training set
                train_idx_purged = purge_training_idxs(
                    train_idx,
                    test_idx[0],
                    test_idx[-1] + 1,
                    horizon=horizon
                )

                if len(train_idx_purged) == 0:
                    tprint(f"⚠️ Fold {fold_idx}: All training samples purged, skipping", "WARNING")
                    continue

                # Get training and test data
                X_train = meta_features.iloc[train_idx_purged]
                y_train = meta_labels.iloc[train_idx_purged]
                X_test = meta_features.iloc[test_idx]
                y_test = meta_labels.iloc[test_idx]

                # Filter out NaN labels
                train_mask = ~y_train.isna()
                test_mask = ~y_test.isna()

                if train_mask.sum() < 10 or test_mask.sum() < 10:
                    tprint(f"⚠️ Fold {fold_idx}: Too few samples, skipping", "WARNING")
                    continue

                X_train_clean = X_train[train_mask].fillna(0)
                y_train_clean = y_train[train_mask]
                X_test_clean = X_test[test_mask].fillna(0)
                y_test_clean = y_test[test_mask]

                # Train model
                model = LogisticRegression(max_iter=1000, random_state=42)
                model.fit(X_train_clean, y_train_clean)

                # Evaluate
                y_pred_proba = model.predict_proba(X_test_clean)[:, 1]
                y_pred = model.predict(X_test_clean)

                try:
                    auc = roc_auc_score(y_test_clean, y_pred_proba)
                except:
                    auc = 0.5

                precision = precision_score(y_test_clean, y_pred, zero_division=0)
                recall = recall_score(y_test_clean, y_pred, zero_division=0)
                f1 = f1_score(y_test_clean, y_pred, zero_division=0)

                cv_results.append({
                    'fold': fold_idx,
                    'auc': auc,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'n_train': len(y_train_clean),
                    'n_test': len(y_test_clean)
                })

                tprint(f"✅ Fold {fold_idx}: AUC={auc:.3f}, Prec={precision:.3f}, Rec={recall:.3f}, F1={f1:.3f}", "INFO")

            # Train final model on all data
            tprint("🎓 Training final meta-model on full dataset...", "INFO")

            full_mask = ~meta_labels.isna()
            X_full = meta_features[full_mask].fillna(0)
            y_full = meta_labels[full_mask]

            final_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            final_model.fit(X_full, y_full)

            # Generate predictions for all data
            X_all = meta_features.fillna(0)
            probabilities = final_model.predict_proba(X_all)[:, 1]

            # STEP 5: Translate to targets for downstream steps
            tprint("🔄 Translating meta-labels to targets...", "INFO")

            target_long, target_short = translate_metalabels_to_targets(
                meta_labels,
                primary_signals,
                probabilities,
                threshold=0.6  # Higher threshold for quality
            )

            # Create output DataFrame with targets
            labeled_data = market_data.copy()
            labeled_data['meta_label'] = meta_labels
            labeled_data['meta_probability'] = probabilities
            labeled_data['target_long'] = target_long
            labeled_data['target_short'] = target_short
            labeled_data['primary_signal'] = primary_signals['consensus']

            # Add signal columns for analysis
            for col in primary_signals.columns:
                labeled_data[f'signal_{col}'] = primary_signals[col]

            # Save labeled data
            tprint("💾 Saving labeled data...", "INFO")

            labeled_data_path = self._save_artifact(
                data=labeled_data,
                artifact_name=f"{config['symbol']}_{config['timeframe']}_meta_labeled_data",
                artifact_type="data",
                metadata={
                    'symbol': config['symbol'],
                    'exchange': config['exchange'],
                    'timeframe': config['timeframe'],
                    'profit_threshold': profit_threshold,
                    'stop_threshold': stop_threshold,
                    'horizon': horizon,
                    'n_samples': len(labeled_data),
                    'n_labeled': int(n_labeled),
                    'n_positive': int(n_positive),
                    'n_negative': int(n_negative),
                    'positive_rate': float(n_positive / n_labeled) if n_labeled > 0 else 0.0
                }
            )

            # Calculate metrics
            avg_auc = np.mean([r['auc'] for r in cv_results]) if cv_results else 0.5
            avg_precision = np.mean([r['precision'] for r in cv_results]) if cv_results else 0.0
            avg_recall = np.mean([r['recall'] for r in cv_results]) if cv_results else 0.0
            avg_f1 = np.mean([r['f1'] for r in cv_results]) if cv_results else 0.0

            # Feature importances
            feature_importances = dict(zip(meta_features.columns, final_model.feature_importances_))
            top_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:10]

            tprint("🎯 Top 10 features:", "INFO")
            for feat, imp in top_features:
                tprint(f"  {feat}: {imp:.4f}", "INFO")

            elapsed_time = (datetime.now() - start_time).total_seconds()

            result = {
                'success': True,
                'artifacts': {
                    'labeled_data_path': labeled_data_path
                },
                'metrics': {
                    'n_samples': len(labeled_data),
                    'n_labeled': int(n_labeled),
                    'n_positive': int(n_positive),
                    'n_negative': int(n_negative),
                    'positive_rate': float(n_positive / n_labeled) if n_labeled > 0 else 0.0,
                    'cv_mean_auc': float(avg_auc),
                    'cv_mean_precision': float(avg_precision),
                    'cv_mean_recall': float(avg_recall),
                    'cv_mean_f1': float(avg_f1),
                    'n_cv_folds': len(cv_results),
                    'elapsed_seconds': elapsed_time,
                    'top_features': dict(top_features),
                    'config': {
                        'profit_threshold': profit_threshold,
                        'stop_threshold': stop_threshold,
                        'horizon': horizon
                    }
                },
                'cv_results': cv_results
            }

            tprint(f"✅ Meta-labeling completed in {elapsed_time:.1f}s", "SUCCESS")
            tprint(f"📊 CV Performance: AUC={avg_auc:.3f}, Precision={avg_precision:.3f}, Recall={avg_recall:.3f}", "SUCCESS")

            return result

        except Exception as e:
            elapsed_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Meta-labeling failed: {str(e)}"
            tprint(f"❌ {error_msg}", "ERROR")
            self.logger.exception("Meta-labeling error")

            return {
                'success': False,
                'error': error_msg,
                'elapsed_seconds': elapsed_time
            }
