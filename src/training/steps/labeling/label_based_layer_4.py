"""Layer 4 — Triple Barrier Trailing Profit & Sizing with Entropy Bars.

Layer2 is about learnability, layer3 about relation to target (IC, calibration),
layer4 is about position sizing. I want to trade it with a triple barrier method
that includes trailing profit.

This module implements:
1.  Triple Barrier Trailing Logic (Exit Strategy).
2.  Inverse Volatility Sizing (Position Sizing).
3.  Integration with Layer 5 via `layer4_prob` proxy generation.
4.  Entropy Bars integration for improved information-based sampling.

REFACTORED: Now uses 4-horizon ORF estimates (CATE) and Standard Errors (SE) from Layer 3.
Enhanced with entropy bars for better market microstructure analysis.
Estimates: ORF 12/48 Reg/Class.
Sizing uses Risk-Adjusted CATE (Point Estimate / Standard Error).
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
from datetime import datetime
import json

from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
from scipy.stats import spearmanr, norm
import statsmodels.api as sm
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.calibration import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.base import clone

# Import entropy bars functionality
try:
    from src.utils.entropy_bars import (
        fetch_1min_data_for_entropy_bars,
        generate_entropy_bars_from_ohlcv,
        calculate_specialized_entropy_features
    )
    ENTROPY_BARS_AVAILABLE = True
except ImportError as e:
    ENTROPY_BARS_AVAILABLE = False
    print(f"⚠️ Entropy bars not available in Layer 4: {e}")

# Configuration Constants
STOP_LOSS_FLOOR = 0.004  # 0.3% Fees + 0.1% Spread Buffer
TARGET_VOLATILITY = 0.01  # 1% target volatility for sizing
VOLATILITY_SAFETY_FLOOR = 1e-4  # Prevent division by zero
HOME_RUN_MULTIPLIER = 3.0  # Multiplier for home run detection
WEIGHT_CLIP_MIN = 0.5  # Minimum weight clip
WEIGHT_CLIP_MAX = 2.0  # Maximum weight clip

class SimpleMultiModelRiskEngine:
    """
    Simple Multi-Model Risk Engine updated for 4-Horizon ORF inputs with Entropy Bars.
    
    Consumes:
    - CATE (Point Estimates) for 12/48 Reg/Class
    - SE (Standard Errors) for 12/48 Reg/Class
    - Disagreement features (e.g. 12 Reg vs 48 Reg)
    - Entropy bar features for market microstructure analysis
    """
    
    def __init__(self, 
                 n_estimators: int = 1000, 
                 max_features: str = 'log2',
                 consensus_weights: Optional[Dict[str, float]] = None):
        
        self.consensus_weights = consensus_weights or {
            'extratrees': 0.66,
            'ridge': 0.34
        }
        
        self.extratrees = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_features=max_features,
            bootstrap=True,
            n_jobs=-1,
            random_state=42
        )
        
        self.ridge = Ridge(alpha=1.0, random_state=42)
        
        self.calibrators = {
            'extratrees': IsotonicRegression(out_of_bounds='clip'),
            'ridge': IsotonicRegression(out_of_bounds='clip')
        }
        
        self.consensus_calibrator = IsotonicRegression(out_of_bounds='clip')
        self.feature_names = None
        self.is_fitted = False
    
    def _compute_financial_weights(self, abs_returns: pd.Series) -> pd.Series:
        weights = abs_returns / (abs_returns.sum() + 1e-9) * len(abs_returns)
        weights = weights.clip(weights.quantile(0.01), weights.quantile(0.99))
        weights = weights / (weights.sum() + 1e-9) * len(weights)
        return weights
    
    def _extract_orf_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derives features from ORF outputs:
        1. Risk-Adjusted CATE (Lift / Uncertainty)
        2. Horizon Disagreement (12 vs 48)
        3. Raw Point Estimates
        """
        feats = pd.DataFrame(index=df.index)
        
        # Horizons & Types
        horizons = ['12', '48']
        types = ['reg', 'cls']
        
        for h in horizons:
            for t in types:
                cate_col = f'orf_cate_{h}_{t}'
                se_col = f'orf_se_{h}_{t}'
                
                if cate_col in df.columns:
                    feats[f'cate_{h}_{t}'] = df[cate_col]
                    
                    if se_col in df.columns:
                        # Risk-Adjusted CATE: Higher lift with lower uncertainty = more conviction
                        feats[f'ra_cate_{h}_{t}'] = df[cate_col] / (df[se_col] + 1e-9)
        
        # Disagreement Features (Horizon Spread)
        if 'orf_cate_12_reg' in df.columns and 'orf_cate_48_reg' in df.columns:
            feats['reg_horizon_disagreement'] = df['orf_cate_12_reg'] - df['orf_cate_48_reg']
            
        if 'orf_cate_12_cls' in df.columns and 'orf_cate_48_cls' in df.columns:
            feats['cls_horizon_disagreement'] = df['orf_cate_12_cls'] - df['orf_cate_48_cls']
            
        return feats.fillna(0)

    def _extract_entropy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract entropy bar features for enhanced market microstructure analysis.
        """
        feats = pd.DataFrame(index=df.index)
        
        # Entropy bar specific features
        entropy_feature_cols = [
            'staleness_seconds', 'staleness_minutes', 'drift_proxy', 
            'lz_complexity', 'trend_conviction_index', 'staleness_adjusted_drift',
            'entropy_ma', 'entropy_std', 'entropy_zscore'
        ]
        
        for col in entropy_feature_cols:
            if col in df.columns:
                feats[col] = df[col]
        
        # Entropy OHLCV features
        entropy_ohlcv_cols = ['entropy_open', 'entropy_high', 'entropy_low', 'entropy_close', 'entropy_volume']
        for col in entropy_ohlcv_cols:
            if col in df.columns:
                feats[col] = df[col]
        
        return feats.fillna(0)

    def train(self, df: pd.DataFrame, market_features: pd.DataFrame,
              y_true: pd.Series, abs_returns: pd.Series) -> Dict[str, Any]:
        """
        Train risk engine using 4-horizon ORF features, entropy features, and market context.
        """
        tprint_info("🚀 Training Layer 4 Risk Engine (Multi-Horizon ORF + Entropy Bars)...")
        
        orf_feats = self._extract_orf_features(df)
        entropy_feats = self._extract_entropy_features(df) if ENTROPY_BARS_AVAILABLE else pd.DataFrame(index=df.index)
        
        # Combine all features
        X = pd.concat([orf_feats, entropy_feats, market_features], axis=1).fillna(0)
        self.feature_names = X.columns.tolist()
        weights = self._compute_financial_weights(abs_returns)
        
        tprint_info(f"📊 Training on {len(self.feature_names)} features ({len(orf_feats.columns)} ORF, {len(entropy_feats.columns)} entropy, {len(market_features.columns)} market)")
        
        base_predictions = {}
        
        # Train ExtraTrees
        tprint_info(f"📊 Training ExtraTrees...")
        self.extratrees.fit(X, y_true, sample_weight=weights)
        et_preds = self.extratrees.predict(X)
        base_predictions['extratrees'] = et_preds
        self.calibrators['extratrees'].fit(et_preds, y_true)
        
        # Train Ridge
        tprint_info("📊 Training Ridge...")
        self.ridge.fit(X, y_true, sample_weight=weights)
        ridge_preds = self.ridge.predict(X)
        base_predictions['ridge'] = ridge_preds
        self.calibrators['ridge'].fit(ridge_preds, y_true)
        
        # Create Weighted Consensus
        consensus_raw = (self.consensus_weights['extratrees'] * et_preds + 
                         self.consensus_weights['ridge'] * ridge_preds)
        
        # Calibrate Consensus
        self.consensus_calibrator.fit(consensus_raw, y_true)
        consensus_calibrated = self.consensus_calibrator.transform(consensus_raw)
        
        self.is_fitted = True
        self.final_predictions_ = consensus_calibrated
        
        metrics = {
            'consensus_weighted_logloss': log_loss(y_true, consensus_calibrated, sample_weight=weights),
            'n_features': len(self.feature_names),
            'n_orf_features': len(orf_feats.columns),
            'n_entropy_features': len(entropy_feats.columns),
            'n_market_features': len(market_features.columns),
            'mean_conviction': consensus_calibrated.mean()
        }
        
        tprint_success(f"✅ Layer 4 Engine trained: WL={metrics['consensus_weighted_logloss']:.4f}, Features={metrics['n_features']}")
        return metrics

    def predict_bet_size(self, df: pd.DataFrame, market_features: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("RiskEngine must be fitted")
            
        orf_feats = self._extract_orf_features(df)
        entropy_feats = self._extract_entropy_features(df) if ENTROPY_BARS_AVAILABLE else pd.DataFrame(index=df.index)
        
        X = pd.concat([orf_feats, entropy_feats, market_features], axis=1).fillna(0)
        X = X[self.feature_names]
        
        et_preds = self.extratrees.predict(X)
        et_cal = self.calibrators['extratrees'].transform(et_preds)
        
        ridge_preds = self.ridge.predict(X)
        ridge_cal = self.calibrators['ridge'].transform(ridge_preds)
        
        consensus = (self.consensus_weights['extratrees'] * et_cal + 
                     self.consensus_weights['ridge'] * ridge_cal)
        
        return self.consensus_calibrator.transform(consensus)


def integrate_entropy_bars_into_layer4(
    df: pd.DataFrame,
    symbol: str = "ETHUSDT",
    exchange: str = "binance",
    config: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Integrate entropy bars into Layer 4 processing.
    
    Args:
        df: Original DataFrame with market data
        symbol: Trading symbol
        exchange: Exchange name
        config: Configuration dictionary
        
    Returns:
        Tuple of (enhanced_df, entropy_bars_df)
    """
    if not ENTROPY_BARS_AVAILABLE:
        tprint_warning("⚠️ Entropy bars not available in Layer 4, using original data")
        return df, pd.DataFrame()
    
    cfg = config or {}
    
    try:
        # Fetch 1-minute data for entropy bar generation
        tprint_info("🔧 Layer 4: Fetching 1-minute data for entropy bar generation")
        
        # Determine date range from existing data
        if not df.empty and hasattr(df.index, 'min') and hasattr(df.index, 'max'):
            start_date = df.index.min().strftime('%Y-%m-%d')
            end_date = df.index.max().strftime('%Y-%m-%d')
        else:
            # Default to last 30 days if no date range available
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
        
        min_data = fetch_1min_data_for_entropy_bars(
            symbol=symbol,
            exchange=exchange,
            start_date=start_date,
            end_date=end_date,
            data_dir=cfg.get('data_dir', 'historical_data')
        )
        
        if min_data is None or min_data.empty:
            tprint_warning("⚠️ Layer 4: No 1-minute data available, skipping entropy bars")
            return df, pd.DataFrame()
        
        # Generate entropy bars
        tprint_info("🔄 Layer 4: Generating entropy bars from 1-minute data")
        entropy_bars = generate_entropy_bars_from_ohlcv(
            ohlcv_data=min_data,
            n_bins=cfg.get('entropy_bins', 10),
            window_size=cfg.get('entropy_window', 100),
            target_minutes=cfg.get('entropy_target_minutes', 15),
            symbol=symbol,
            exchange=exchange
        )
        
        if entropy_bars.empty:
            tprint_warning("⚠️ Layer 4: Failed to generate entropy bars")
            return df, pd.DataFrame()
        
        # Calculate specialized entropy features
        tprint_info("🎯 Layer 4: Calculating specialized entropy features")
        entropy_features = calculate_specialized_entropy_features(
            entropy_bars=entropy_bars,
            base_model_updates=df,  # Use df as proxy for base model updates
            specialist_prices=df['close'] if 'close' in df.columns else None,
            volatility_window=cfg.get('volatility_window', 20)
        )
        
        # Merge entropy features back to main dataframe
        enhanced_df = df.copy()
        
        # Forward-fill entropy features to match main dataframe timestamps
        for col in entropy_features.columns:
            enhanced_df[col] = entropy_features[col].reindex(enhanced_df.index, method='ffill').fillna(0)
        
        # Add entropy bar OHLCV data as additional columns
        entropy_ohlcv_cols = ['open', 'high', 'low', 'close', 'volume', 'n_minutes', 'entropy_contribution']
        for col in entropy_ohlcv_cols:
            if col in entropy_bars.columns:
                enhanced_df[f'entropy_{col}'] = entropy_bars[col].reindex(enhanced_df.index, method='ffill').fillna(
                    enhanced_df[col] if col in enhanced_df.columns else 0
                )
        
        tprint_success(f"✅ Layer 4: Integrated entropy bars: {len(entropy_bars)} bars, {len(entropy_features.columns)} features")
        
        return enhanced_df, entropy_bars_df
        
    except Exception as e:
        tprint_error(f"❌ Layer 4: Error integrating entropy bars: {e}")
        return df, pd.DataFrame()


def train_layer4_simple_multimodel(
    oof_df: pd.DataFrame,
    market_data: pd.DataFrame,
    base_model_predictions: Optional[pd.DataFrame] = None,
    target_col: str = 'realized_return',
    n_folds: int = 5,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Train Layer 4 using 4-horizon ORF estimates, entropy bars, and market features.
    """
    cfg = config or {}
    tprint_info("🚀 Starting Layer 4 Training (Multi-Horizon ORF + Entropy Bars)...")
    
    # Integrate entropy bars if enabled
    symbol = cfg.get('symbol', 'ETHUSDT')
    exchange = cfg.get('exchange', 'binance')
    
    if cfg.get('use_entropy_bars', True):
        oof_df, entropy_bars_df = integrate_entropy_bars_into_layer4(oof_df, symbol, exchange, cfg)
        cfg['entropy_bars_df'] = entropy_bars_df
    else:
        tprint_info("⏭️ Layer 4: Skipping entropy bars (disabled in config)")
        entropy_bars_df = pd.DataFrame()
    
    # Generate market features
    try:
        from .layer4_extratrees_pnl import MetaLearnerFeatures
        generator = MetaLearnerFeatures(config=config)
        market_features = generator.generate(
            df=oof_df.join(market_data, how='left', rsuffix='_mkt'),
            raw_price_col='close',
            denoised_price_col='denoised_price'
        )
    except ImportError:
        tprint_warning("⚠️ MetaLearnerFeatures not available, using empty market features")
        market_features = pd.DataFrame(index=oof_df.index)
    
    y_binary = (oof_df[target_col] > 0).astype(int)
    abs_returns = oof_df[target_col].abs()
    
    kf = KFold(n_splits=n_folds, shuffle=False)
    oof_bet_sizes = np.zeros(len(oof_df))
    
    engine = SimpleMultiModelRiskEngine()
    
    for train_idx, val_idx in kf.split(oof_df):
        engine.train(
            df=oof_df.iloc[train_idx],
            market_features=market_features.iloc[train_idx],
            y_true=y_binary.iloc[train_idx],
            abs_returns=abs_returns.iloc[train_idx]
        )
        oof_bet_sizes[val_idx] = engine.predict_bet_size(
            df=oof_df.iloc[val_idx],
            market_features=market_features.iloc[val_idx]
        )
    
    # Final fit
    final_metrics = engine.train(oof_df, market_features, y_binary, abs_returns)
    
    oof_df_out = oof_df.copy()
    oof_df_out['layer4_prob'] = oof_bet_sizes
    
    # Add entropy bars information to metrics
    if not entropy_bars_df.empty:
        final_metrics['entropy_bars_count'] = len(entropy_bars_df)
        final_metrics['entropy_features_count'] = len([col for col in oof_df_out.columns if col.startswith(('staleness_', 'drift_', 'lz_', 'trend_', 'entropy_'))])
    
    return oof_df_out, final_metrics

class MetaLearnerFeatures:
    def __init__(self, **kwargs):
        pass
    def generate(self, df, **kwargs):
        return pd.DataFrame(index=df.index)
