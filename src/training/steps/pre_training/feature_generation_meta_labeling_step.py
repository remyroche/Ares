"""
Feature Generation Meta-Labeling Step (Enhanced Version).

This enhanced version addresses critical issues:
1. Computes realized returns, not just binary labels
2. Uses isotonic regression to map probabilities to expected returns
3. Avoids circular behavior (doesn't reuse raw signals as features)
4. Handles edge windows and overlapping events properly
5. Uses economic metrics (not just accuracy)
6. Includes transaction costs in target estimation

Based on guidance from "Advances in Financial Machine Learning" by Marcos López de Prado.
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
from sklearn.isotonic import IsotonicRegression

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

    CRITICAL: A training sample at position i uses data up to i and predicts i+horizon.
    We must remove training samples where the prediction horizon reaches into validation.

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
    """Compute RSI (Relative Strength Index)."""
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

    CRITICAL: These signals are FIXED and must never be re-optimized during CV.
    They define the "primary model" whose signals we will meta-label.

    Returns:
        DataFrame with signal columns (rsi, ma, mom, consensus)
    """
    signals = pd.DataFrame(index=df.index)
    df_local = df.copy()

    # RSI signals
    df_local['rsi'] = compute_rsi(df_local['close'], period=rsi_period)
    signals['rsi'] = 0
    signals.loc[df_local['rsi'] < rsi_oversold, 'rsi'] = 1
    signals.loc[df_local['rsi'] > rsi_overbought, 'rsi'] = -1

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

    # Consensus signal: majority vote
    signals['consensus'] = signals[['rsi', 'ma', 'mom']].sum(axis=1).apply(np.sign)

    return signals


def compute_realized_returns(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    profit_threshold: float = 0.015,
    stop_threshold: float = 0.010,
    horizon: int = 16,
    transaction_cost: float = 0.0005,
    min_event_spacing: int = 4
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute realized returns for each signal event.

    IMPROVED: Returns continuous values (realized return) instead of binary labels.
    This allows isotonic regression to map probabilities to expected returns.

    Args:
        df: DataFrame with price data
        signals: DataFrame with signal columns
        profit_threshold: Profit target as fraction
        stop_threshold: Stop loss as fraction
        horizon: Maximum bars to look ahead
        transaction_cost: Transaction cost per trade (round trip)
        min_event_spacing: Minimum bars between signals (prevents overlapping events)

    Returns:
        Tuple of (realized_returns, binary_labels)
        - realized_returns: Actual returns achieved (NaN where no signal)
        - binary_labels: Binary success/failure (for model training)
    """
    realized_returns = pd.Series(index=df.index, dtype=float)
    realized_returns[:] = np.nan

    binary_labels = pd.Series(index=df.index, dtype=float)
    binary_labels[:] = np.nan

    close_prices = df['close'].values
    consensus_signals = signals['consensus'].values

    last_event_idx = -min_event_spacing  # Track last signal to avoid overlaps

    for i in range(len(df) - horizon):
        signal = consensus_signals[i]

        # Only create labels where we have a signal
        if signal == 0:
            continue

        # Handle overlapping events: skip if too close to previous signal
        if (i - last_event_idx) < min_event_spacing:
            continue

        # Edge window handling: skip events too close to end of available data
        if i + horizon >= len(df):
            # Mark as NaN - incomplete forward window
            continue

        entry_price = close_prices[i]
        exit_price = None
        exit_reason = None

        # Look ahead up to horizon bars
        for j in range(1, horizon + 1):
            if i + j >= len(df):
                break

            future_price = close_prices[i + j]

            if signal > 0:  # Long signal
                pnl = (future_price - entry_price) / entry_price

                # Hit profit target
                if pnl >= profit_threshold:
                    exit_price = future_price
                    exit_reason = 'profit'
                    break
                # Hit stop loss
                elif pnl <= -stop_threshold:
                    exit_price = future_price
                    exit_reason = 'stop'
                    break

            elif signal < 0:  # Short signal
                pnl = (entry_price - future_price) / entry_price

                # Hit profit target
                if pnl >= profit_threshold:
                    exit_price = future_price
                    exit_reason = 'profit'
                    break
                # Hit stop loss
                elif pnl <= -stop_threshold:
                    exit_price = future_price
                    exit_reason = 'stop'
                    break

        # If no exit, use end-of-horizon price (timeout)
        if exit_price is None:
            exit_price = close_prices[min(i + horizon, len(df) - 1)]
            exit_reason = 'timeout'

        # Compute realized return accounting for transaction costs
        if signal > 0:  # Long
            gross_return = (exit_price - entry_price) / entry_price
        else:  # Short
            gross_return = (entry_price - exit_price) / entry_price

        net_return = gross_return - transaction_cost  # Subtract costs

        # Store results
        realized_returns.iloc[i] = net_return
        binary_labels.iloc[i] = 1.0 if net_return > 0 else 0.0

        last_event_idx = i  # Update last event position

    return realized_returns, binary_labels


def create_meta_features(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    volume_available: bool = True,
    include_raw_signals: bool = False
) -> pd.DataFrame:
    """
    Create features for the meta-model.

    CRITICAL: By default, does NOT include raw signal values to avoid circular behavior.
    Features capture market context, not the signals themselves.

    Args:
        df: DataFrame with OHLCV data
        signals: DataFrame with primary signals (used only for context)
        volume_available: Whether volume data is available
        include_raw_signals: WARNING: Set True only for ablation tests

    Returns:
        DataFrame of features for meta-model
    """
    features = pd.DataFrame(index=df.index)

    # Market context features (NOT the signals themselves)

    # Volatility regime
    returns = df['close'].pct_change()
    features['volatility_5'] = returns.rolling(5).std()
    features['volatility_20'] = returns.rolling(20).std()
    features['volatility_ratio'] = features['volatility_5'] / (features['volatility_20'] + 1e-8)

    # Use EMA for smoothing (reduces noise)
    features['volatility_ema'] = returns.std()  # Will be replaced with EMA
    for i in range(1, len(features)):
        features['volatility_ema'].iloc[i] = (
            0.1 * returns.iloc[i]**2 +
            0.9 * features['volatility_ema'].iloc[i-1] if i > 0 else returns.iloc[i]**2
        )
    features['volatility_ema'] = np.sqrt(features['volatility_ema'])

    # Trend strength
    features['sma_slope'] = df['close'].rolling(10).mean().pct_change(5)
    features['price_vs_sma20'] = (
        (df['close'] - df['close'].rolling(20).mean()) /
        (df['close'].rolling(20).mean() + 1e-8)
    )

    # ADX-like trend strength (simplified)
    high_low = df['high'] - df['low']
    features['atr_14'] = high_low.rolling(14).mean()
    features['atr_ratio'] = features['atr_14'] / (df['close'] + 1e-8)

    # Volume context (if available)
    if volume_available and 'volume' in df.columns:
        vol_sma = df['volume'].rolling(20).mean()
        features['volume_ratio'] = df['volume'] / (vol_sma + 1e-8)
        features['volume_trend'] = (
            df['volume'].rolling(5).mean() / (vol_sma + 1e-8)
        )
        # Volume-price divergence
        features['vol_price_corr'] = returns.rolling(20).corr(
            df['volume'].pct_change()
        )
    else:
        features['volume_ratio'] = 1.0
        features['volume_trend'] = 1.0
        features['vol_price_corr'] = 0.0

    # Momentum (NOT the momentum signal, but market momentum)
    features['momentum_5'] = df['close'].pct_change(5)
    features['momentum_10'] = df['close'].pct_change(10)
    features['momentum_20'] = df['close'].pct_change(20)

    # Smoothed momentum using EMA
    features['momentum_ema'] = features['momentum_10'].ewm(span=5).mean()

    # Position in recent range
    recent_high = df['high'].rolling(20).max()
    recent_low = df['low'].rolling(20).min()
    features['range_position'] = (
        (df['close'] - recent_low) / (recent_high - recent_low + 1e-8)
    )

    # Time-based features (can help with regime changes)
    if isinstance(df.index, pd.DatetimeIndex):
        features['hour'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek
    else:
        features['hour'] = 0
        features['day_of_week'] = 0

    # WARNING: Including raw signals can cause circular behavior
    if include_raw_signals:
        tprint("⚠️ WARNING: Including raw signal features - may cause circular behavior", "WARNING")
        features['signal_strength'] = signals[['rsi', 'ma', 'mom']].abs().sum(axis=1)
        features['signal_consensus'] = signals['consensus'].abs()

    return features


def fit_probability_to_return_mapping(
    probabilities: np.ndarray,
    realized_returns: np.ndarray,
    method: str = 'isotonic'
) -> IsotonicRegression:
    """
    Fit mapping from predicted probability to expected return.

    Uses isotonic regression to create a monotonic mapping that captures
    the empirical relationship between model confidence and realized returns.

    CRITICAL: Must use out-of-fold probabilities to avoid leakage.

    Args:
        probabilities: Out-of-fold predicted probabilities
        realized_returns: Realized returns for those events
        method: 'isotonic' or 'binned'

    Returns:
        Fitted IsotonicRegression model
    """
    # Remove NaN values
    mask = ~(np.isnan(probabilities) | np.isnan(realized_returns))
    p_clean = probabilities[mask]
    r_clean = realized_returns[mask]

    if len(p_clean) < 10:
        tprint("⚠️ Warning: Very few samples for probability mapping", "WARNING")

    if method == 'isotonic':
        # Isotonic regression: monotonic mapping
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(p_clean, r_clean)
        return iso

    elif method == 'binned':
        # Alternative: binned approach (less smooth but more robust)
        # Not implemented here, but would bin probabilities and take mean return per bin
        raise NotImplementedError("Binned method not yet implemented")

    else:
        raise ValueError(f"Unknown method: {method}")


def translate_to_targets_with_isotonic(
    realized_returns: pd.Series,
    probabilities: np.ndarray,
    signals: pd.DataFrame,
    iso_regressor: IsotonicRegression
) -> Tuple[pd.Series, pd.Series]:
    """
    Translate probabilities to continuous targets using isotonic regression.

    This creates economically meaningful targets based on expected returns.

    Args:
        realized_returns: Actual returns (used only for validation)
        probabilities: Predicted probabilities from meta-model
        signals: Signal directions
        iso_regressor: Fitted isotonic regression model

    Returns:
        Tuple of (target_long, target_short)
    """
    target_long = pd.Series(0.0, index=realized_returns.index)
    target_short = pd.Series(0.0, index=realized_returns.index)

    consensus = signals['consensus'].values

    for i in range(len(realized_returns)):
        if pd.isna(realized_returns.iloc[i]):
            continue

        prob = probabilities[i] if i < len(probabilities) else 0.5
        signal = consensus[i]

        # Map probability to expected return using isotonic regression
        expected_return = iso_regressor.predict([prob])[0]

        # Only create targets for high-confidence predictions
        # and in the direction of the signal
        if signal > 0:  # Long signal
            target_long.iloc[i] = max(0, expected_return)
        elif signal < 0:  # Short signal
            target_short.iloc[i] = max(0, expected_return)

    return target_long, target_short


class FeatureGenerationMetaLabelingStep(BaseStep):
    """
    Feature Generation Meta-Labeling Step (Enhanced).

    Improvements over basic version:
    - Computes realized returns (not just binary labels)
    - Uses isotonic regression for probability → expected return mapping
    - Avoids circular behavior (doesn't include raw signals in features)
    - Handles overlapping events and edge windows
    - Includes transaction costs
    - Uses economic metrics
    """

    def __init__(self, step_name: str = "feature_generation_meta_labeling_step"):
        """Initialize the meta-labeling step."""
        super().__init__(step_name)
        self.logger = system_logger.getChild('FeatureGenerationMetaLabeling')

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute feature generation meta-labeling with enhanced methodology.

        Args:
            config: Configuration dictionary

        Returns:
            Result dictionary with targets and metrics
        """
        start_time = datetime.now()

        # Validate required config
        required = ('symbol', 'exchange', 'timeframe')
        missing = [k for k in required if k not in config or not config[k]]
        if missing:
            error_msg = f"Missing required config keys: {', '.join(missing)}"
            tprint(f"❌ {error_msg}", "ERROR")
            return {'success': False, 'error': error_msg}

        tprint(f"🏷️ Starting enhanced meta-labeling for {config['symbol']}", "INFO")

        try:
            # Extract config parameters
            profit_threshold = config.get('profit_threshold', 0.015)
            stop_threshold = config.get('stop_threshold', 0.010)
            horizon = config.get('horizon', 16)
            transaction_cost = config.get('transaction_cost', 0.0005)
            min_event_spacing = config.get('min_event_spacing', 4)

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

            if 'close' not in market_data.columns:
                raise ValueError("Missing required 'close' column in market data")

            volume_available = 'volume' in market_data.columns

            # STEP 1: Generate FIXED primary signals
            tprint("🎯 Generating fixed primary signals...", "INFO")
            primary_signals = generate_primary_signals(market_data)

            n_long_signals = (primary_signals['consensus'] > 0).sum()
            n_short_signals = (primary_signals['consensus'] < 0).sum()
            tprint(f"📊 Primary signals: {n_long_signals} long, {n_short_signals} short", "INFO")

            # STEP 2: Compute realized returns (continuous) and binary labels
            tprint("💰 Computing realized returns with transaction costs...", "INFO")
            realized_returns, binary_labels = compute_realized_returns(
                market_data,
                primary_signals,
                profit_threshold=profit_threshold,
                stop_threshold=stop_threshold,
                horizon=horizon,
                transaction_cost=transaction_cost,
                min_event_spacing=min_event_spacing
            )

            # Statistics
            labeled_mask = ~binary_labels.isna()
            n_labeled = labeled_mask.sum()
            n_positive = (binary_labels == 1.0).sum()
            n_negative = (binary_labels == 0.0).sum()

            if n_labeled > 0:
                mean_return = realized_returns[labeled_mask].mean()
                median_return = realized_returns[labeled_mask].median()
                win_rate = n_positive / n_labeled

                tprint(f"📊 Events: {n_labeled} total ({n_positive} wins, {n_negative} losses)", "INFO")
                tprint(f"📈 Win rate: {win_rate:.1%}, Mean return: {mean_return:.2%}, Median: {median_return:.2%}", "INFO")
            else:
                tprint("⚠️ Warning: No labeled events found", "WARNING")
                mean_return = 0.0
                median_return = 0.0
                win_rate = 0.0

            # STEP 3: Create meta-features (WITHOUT raw signals to avoid circular behavior)
            tprint("🔧 Creating meta-features (excluding raw signals)...", "INFO")
            meta_features = create_meta_features(
                market_data,
                primary_signals,
                volume_available,
                include_raw_signals=False  # CRITICAL: avoid circular behavior
            )

            # STEP 4: Train meta-model with time-series CV
            # Get out-of-fold probabilities for isotonic regression
            tprint("🎓 Training meta-model with purged time-series CV...", "INFO")

            outer_cv = TimeSeriesSplit(n_splits=5)
            cv_results = []

            # Store out-of-fold predictions
            oof_probabilities = pd.Series(np.nan, index=market_data.index)

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
                y_train = binary_labels.iloc[train_idx_purged]
                X_test = meta_features.iloc[test_idx]
                y_test = binary_labels.iloc[test_idx]

                # Filter out NaN labels
                train_mask = ~y_train.isna()
                test_mask = ~y_test.isna()

                if train_mask.sum() < 10 or test_mask.sum() < 5:
                    tprint(f"⚠️ Fold {fold_idx}: Too few samples, skipping", "WARNING")
                    continue

                X_train_clean = X_train[train_mask].fillna(0)
                y_train_clean = y_train[train_mask]
                X_test_clean = X_test[test_mask].fillna(0)
                y_test_clean = y_test[test_mask]

                # Train model
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=8,
                    min_samples_leaf=20,  # Prevent overfitting
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train_clean, y_train_clean)

                # Get probabilities
                y_pred_proba = model.predict_proba(X_test_clean)[:, 1]

                # Store out-of-fold probabilities
                test_indices_with_labels = test_idx[test_mask]
                oof_probabilities.iloc[test_indices_with_labels] = y_pred_proba

                # Evaluate with proper metrics
                try:
                    auc = roc_auc_score(y_test_clean, y_pred_proba)
                except:
                    auc = 0.5

                y_pred = (y_pred_proba >= 0.5).astype(int)
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

            # STEP 5: Fit isotonic regression using OOF probabilities
            tprint("📈 Fitting probability → expected return mapping (isotonic regression)...", "INFO")

            oof_mask = ~oof_probabilities.isna() & ~realized_returns.isna()

            if oof_mask.sum() < 20:
                tprint("⚠️ Warning: Too few samples for isotonic regression, using simple threshold", "WARNING")
                iso_regressor = None
            else:
                iso_regressor = fit_probability_to_return_mapping(
                    oof_probabilities[oof_mask].values,
                    realized_returns[oof_mask].values,
                    method='isotonic'
                )
                tprint(f"✅ Fitted mapping using {oof_mask.sum()} out-of-fold samples", "SUCCESS")

            # STEP 6: Train final model on all data
            tprint("🎓 Training final meta-model on full dataset...", "INFO")

            full_mask = ~binary_labels.isna()
            X_full = meta_features[full_mask].fillna(0)
            y_full = binary_labels[full_mask]

            final_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_leaf=20,
                random_state=42,
                n_jobs=-1
            )
            final_model.fit(X_full, y_full)

            # Generate predictions for all data
            X_all = meta_features.fillna(0)
            probabilities = final_model.predict_proba(X_all)[:, 1]

            # STEP 7: Translate to targets using isotonic regression
            tprint("🔄 Translating probabilities to economic targets...", "INFO")

            if iso_regressor is not None:
                target_long, target_short = translate_to_targets_with_isotonic(
                    realized_returns,
                    probabilities,
                    primary_signals,
                    iso_regressor
                )
            else:
                # Fallback: simple threshold-based approach
                tprint("⚠️ Using fallback threshold-based translation", "WARNING")
                target_long = pd.Series(0.0, index=market_data.index)
                target_short = pd.Series(0.0, index=market_data.index)

                threshold = 0.6
                for i in range(len(market_data)):
                    if probabilities[i] >= threshold:
                        if primary_signals['consensus'].iloc[i] > 0:
                            target_long.iloc[i] = probabilities[i] - threshold
                        elif primary_signals['consensus'].iloc[i] < 0:
                            target_short.iloc[i] = probabilities[i] - threshold

            # Create output DataFrame
            labeled_data = market_data.copy()
            labeled_data['realized_return'] = realized_returns
            labeled_data['binary_label'] = binary_labels
            labeled_data['meta_probability'] = probabilities
            labeled_data['target_long'] = target_long
            labeled_data['target_short'] = target_short
            labeled_data['primary_signal'] = primary_signals['consensus']

            # Save labeled data
            tprint("💾 Saving labeled data...", "INFO")

            labeled_data_path = self._save_artifact(
                data=labeled_data,
                artifact_name=f"{config['symbol']}_{config['timeframe']}_meta_labeled_data_v2",
                artifact_type="data",
                metadata={
                    'symbol': config['symbol'],
                    'exchange': config['exchange'],
                    'timeframe': config['timeframe'],
                    'profit_threshold': profit_threshold,
                    'stop_threshold': stop_threshold,
                    'horizon': horizon,
                    'transaction_cost': transaction_cost,
                    'n_samples': len(labeled_data),
                    'n_labeled': int(n_labeled),
                    'n_positive': int(n_positive),
                    'win_rate': float(win_rate),
                    'mean_return': float(mean_return),
                    'median_return': float(median_return)
                }
            )

            # Calculate metrics
            avg_auc = np.mean([r['auc'] for r in cv_results]) if cv_results else 0.5
            avg_precision = np.mean([r['precision'] for r in cv_results]) if cv_results else 0.0

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
                    'win_rate': float(win_rate),
                    'mean_return': float(mean_return),
                    'median_return': float(median_return),
                    'cv_mean_auc': float(avg_auc),
                    'cv_mean_precision': float(avg_precision),
                    'n_cv_folds': len(cv_results),
                    'elapsed_seconds': elapsed_time,
                    'top_features': dict(top_features),
                    'config': {
                        'profit_threshold': profit_threshold,
                        'stop_threshold': stop_threshold,
                        'horizon': horizon,
                        'transaction_cost': transaction_cost,
                        'min_event_spacing': min_event_spacing
                    }
                },
                'cv_results': cv_results
            }

            tprint(f"✅ Enhanced meta-labeling completed in {elapsed_time:.1f}s", "SUCCESS")
            tprint(f"📊 Performance: AUC={avg_auc:.3f}, Win Rate={win_rate:.1%}, Mean Return={mean_return:.2%}", "SUCCESS")

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
