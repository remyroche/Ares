"""Layer 4 — Triple Barrier Trailing Profit & Sizing with Entropy Bars.

Layer2 is about learnability, layer3 about relation to target (IC, calibration),
layer4 is about position sizing. I want to trade it with a triple barrier method
that includes trailing profit.

This module implements:
1.  Triple Barrier Trailing Logic (Exit Strategy).
2.  Inverse Volatility Sizing (Position Sizing).
3.  Integration with Layer 5 via `layer4_prob` proxy generation.
4.  Entropy Bars integration for improved information-based sampling.

REFACTORED: Now uses Layer 3 probability outputs (12/48 horizons).
Enhanced with entropy bars for better market microstructure analysis.
Sizing uses calibrated probability consensus.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
from datetime import datetime
import json
import os
import hashlib

from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
from scipy.stats import spearmanr, norm
import statsmodels.api as sm
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.calibration import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.base import clone

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import RobustScaler

from src.utils.huber_regressor_for_trees import prepare_huber_teacher_outputs
from src.training.steps.labeling.layer4_dual_chaser import (
    StructuralRegimeGMM,
    train_dual_chaser_audit,
    generate_dual_chaser_features
)
from src.utils.layer4_optimized import (
    compute_financial_weights_numba,
    extract_prob_features_numba,
    compute_prob_stats_numba,
    rolling_sadf_score_numba,
    rolling_cusum_scores_numba,
    compute_proxy_entropy_numba
)
from src.training.steps.labeling.feature_engineering_utils import apply_layer2_price_processing

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

def downcast_float(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast float64 columns to float32 to save memory and speed up processing."""
    float_cols = df.select_dtypes(include=['float64']).columns
    if len(float_cols) > 0:
        # Check for infs before casting (only if necessary to avoid copy if not needed)
        # Using a more efficient check
        vals = df[float_cols].values
        if np.isinf(vals).any():
             df[float_cols] = df[float_cols].replace([np.inf, -np.inf], np.nan)

        # Use simple astype
        df[float_cols] = df[float_cols].astype(np.float32)
    return df

class SimpleMultiModelRiskEngine:
    """
    Simple Multi-Model Risk Engine updated for Layer 3 probability inputs with Entropy Bars.

    Consumes:
    - Layer 3 probability outputs (12/48 horizons, ensemble variants)
    - Disagreement/dispersion features across horizons
    - Entropy bar features for market microstructure analysis

    Updated to use Huber Regressor for Teacher logic, pruning, and constraints.
    """
    
    def __init__(self, 
                 n_estimators: int = 1000, 
                 max_features: str = 'log2',
                 consensus_weights: Optional[Dict[str, float]] = None):
        
        # Default weights if not provided
        self.consensus_weights = consensus_weights or {
            'extratrees': 0.25,
            'lgbm': 0.25,
            'xgboost': 0.25,
            'catboost': 0.25
        }
        
        # Models configuration
        self.extratrees = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_features=max_features,
            bootstrap=True,
            n_jobs=-1,
            random_state=42
        )
        
        # LGBM params (implicit defaults + warm start handling in train)
        self.lgbm_params = {
            'n_estimators': 1000,
            'n_jobs': 2,
            'random_state': 42,
            'verbosity': -1,
            'reg_alpha': 5.0,
            'reg_lambda': 50.0
        }
        self.lgbm_model = None

        # XGB params
        self.xgb_params = {
            'n_estimators': 1000,
            'n_jobs': 2,
            'random_state': 42,
            'num_parallel_tree': 7,
            'colsample_bynode': 0.4,
            'subsample': 0.6,
            'reg_lambda': 50, # "22 regularisation 50" -> l2 regularization
            'min_child_weight': 10,
            'gamma': 1.1,
            'learning_rate': 0.03
        }
        self.xgb_model = None

        # CatBoost params
        self.catboost_params = {
            'n_estimators': 1000,
            'random_state': 42,
            'thread_count': -1,
            'subsample': 0.6,
            'colsample_bylevel': 0.5,
            'leaf_estimation_iterations': 10,
            'l2_leaf_reg': 20, # overriding 7 with 20 per instructions
            'random_strength': 5,
            'bootstrap_type': 'MVS',
            'allow_writing_files': False,
            'verbose': 0
        }
        self.catboost_model = None
        
        self.calibrators = {
            'extratrees': IsotonicRegression(out_of_bounds='clip'),
            'lgbm': IsotonicRegression(out_of_bounds='clip'),
            'xgboost': IsotonicRegression(out_of_bounds='clip'),
            'catboost': IsotonicRegression(out_of_bounds='clip')
        }
        
        self.consensus_calibrator = IsotonicRegression(out_of_bounds='clip')
        self.feature_names = None
        self.selected_features = None
        self.huber_model = None
        self.huber_scaler = None
        self.huber_feature_columns = None
        self.is_fitted = False

        # Dual Chaser components
        self.stable_chaser = None
        self.aggressive_chaser = None
        self.dual_chaser_scaler = None
    
    def _compute_financial_weights(self, abs_returns: pd.Series, volatility: pd.Series) -> pd.Series:
        # Use Numba-optimized implementation
        weights_array = compute_financial_weights_numba(
            abs_returns.values.astype(np.float64),
            volatility.values.astype(np.float64)
        )
        return pd.Series(weights_array, index=abs_returns.index)
    
    def _extract_layer3_prob_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derives features from Layer 3 probability outputs:
        1. Raw probabilities per horizon/model
        2. Confidence (distance from 0.5)
        3. Horizon disagreement (12 vs 48)
        4. Cross-model dispersion (mean/std/range)
        """
        feats = pd.DataFrame(index=df.index)

        base_prob_cols = []
        if 'meta_prob' in df.columns:
            base_prob_cols.append('meta_prob')
        if 'meta_prob_48' in df.columns:
            base_prob_cols.append('meta_prob_48')

        extra_prob_cols = [c for c in df.columns if c.startswith('meta_prob_') and c not in base_prob_cols]
        prob_cols = base_prob_cols + extra_prob_cols

        if not prob_cols:
            return feats

        # Vectorized feature extraction using Numba
        # Optimization: Avoid astype copy if already float32
        probs_vals = df[prob_cols].values
        if probs_vals.dtype == np.float32:
            probs_array = probs_vals
        else:
            probs_array = probs_vals.astype(np.float32)

        logits, confidences = extract_prob_features_numba(probs_array)

        for idx, col in enumerate(prob_cols):
            feats[f'{col}_raw'] = probs_array[:, idx]
            feats[f'{col}_logit'] = logits[:, idx]
            feats[f'{col}_confidence'] = confidences[:, idx]

        if 'meta_prob' in df.columns and 'meta_prob_48' in df.columns:
            # Vectorized horizon features
            p12 = df['meta_prob'].values
            p48 = df['meta_prob_48'].values
            diff = p12 - p48
            feats['horizon_disagreement'] = diff
            feats['horizon_agreement'] = 1.0 - np.abs(diff)

        # Numba-optimized statistics
        means, stds, mins, maxs, ranges = compute_prob_stats_numba(probs_array)
        feats['prob_mean'] = means
        feats['prob_std'] = stds
        feats['prob_min'] = mins
        feats['prob_max'] = maxs
        feats['prob_range'] = ranges

        return feats.fillna(0.0)

    def _extract_entropy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract entropy bar features for enhanced market microstructure analysis.
        """
        feats = pd.DataFrame(index=df.index)
        
        # Entropy bar specific features
        entropy_feature_cols = [
            'staleness_seconds', 'staleness_minutes', 'drift_proxy', 
            'lz_complexity', 'trend_conviction_index', 'staleness_adjusted_drift',
            'entropy_ma', 'entropy_std', 'entropy_zscore', 'proxy_entropy'
        ]
        
        for col in entropy_feature_cols:
            if col in df.columns:
                feats[col] = df[col]
        
        # Entropy OHLCV features
        entropy_ohlcv_cols = ['entropy_close', 'entropy_volume']
        for col in entropy_ohlcv_cols:
            if col in df.columns:
                feats[col] = df[col]
        
        return feats.fillna(0)

    def train(self, df: pd.DataFrame, market_features: pd.DataFrame,
              y_true: pd.Series, abs_returns: pd.Series) -> Dict[str, Any]:
        """
        Train risk engine using Layer 3 probability features, entropy features, and market context.
        Uses Huber Regressor for Teacher output generation (constraints, pruning, warm start).
        """
        tprint_info("🚀 Training Layer 4 Risk Engine (Layer 3 Probabilities + Entropy Bars)...")

        prob_feats = self._extract_layer3_prob_features(df)
        entropy_feats = self._extract_entropy_features(df) if ENTROPY_BARS_AVAILABLE else pd.DataFrame(index=df.index)
        
        # Combine all features
        X_full = pd.concat([prob_feats, entropy_feats, market_features], axis=1).fillna(0)

        # --- Dual Chaser Audit & Feature Generation ---
        # 1. Fit GMM to identify regimes (if data available)
        try:
            gmm = StructuralRegimeGMM(n_regimes=4)
            env_indices, _ = gmm.fit_predict(df)

            if env_indices and len(env_indices) >= 2:
                tprint_info("🏃 Training Dual Chaser Audit (IRM vs Ridge) with OOF...")

                # Filter out performance features for Chasers to avoid leakage
                chaser_cols = [c for c in X_full.columns if not c.startswith('perf_') and not c.startswith('meta_')]
                X_for_chaser = X_full[chaser_cols]

                # 2. Prepare Scaled Features for Chasers
                self.dual_chaser_scaler = RobustScaler()
                X_chaser_scaled = self.dual_chaser_scaler.fit_transform(X_for_chaser)

                # Construct simple CV splits for OOF generation
                n_samples = len(X_full)
                n_splits = 5
                fold_size = n_samples // n_splits
                cv_splits = []
                for k in range(n_splits):
                    val_start = k * fold_size
                    val_end = (k + 1) * fold_size if k < n_splits - 1 else n_samples
                    val_idx = np.arange(val_start, val_end)
                    train_idx = np.concatenate([np.arange(0, val_start), np.arange(val_end, n_samples)])
                    cv_splits.append((train_idx, val_idx))

                # 3. Train Chasers & Get OOF
                self.stable_chaser, self.aggressive_chaser, oof_stable, oof_agg = train_dual_chaser_audit(
                    X_chaser_scaled,
                    y_true.values,
                    env_indices,
                    cv_splits=cv_splits
                )

                # 4. Generate Dual Chaser Features using OOF predictions
                if oof_stable is not None and oof_agg is not None:
                    oof_stable.index = X_full.index
                    oof_agg.index = X_full.index

                    # Prepare trade_returns proxy for meta-health features
                    trade_ret_proxy = (2 * y_true - 1) * abs_returns

                    chaser_feats = generate_dual_chaser_features(
                        df=df,
                        p_stable=oof_stable,
                        p_agg=oof_agg,
                        trade_returns=trade_ret_proxy
                    )

                    # 5. Add to X_full
                    X_full = pd.concat([X_full, chaser_feats], axis=1)
                    tprint_success(f"✅ Added {chaser_feats.shape[1]} Dual Chaser features (OOF)")
                else:
                    tprint_warning("⚠️ Dual Chaser OOF generation failed.")
                    self.stable_chaser = None
            else:
                tprint_warning("⚠️ Dual Chaser: GMM failed or insufficient regimes. Skipping.")
                self.stable_chaser = None
        except Exception as e:
            tprint_error(f"❌ Dual Chaser failed: {e}")
            import traceback
            traceback.print_exc()
            self.stable_chaser = None

        # Downcast to float32
        X_full = downcast_float(X_full)

        # Extract volatility for weighting
        if 'volatility' in market_features.columns:
            volatility = market_features['volatility']
        else:
            tprint_warning("⚠️ Volatility not found in market_features. Using unweighted fallback (1.0).")
            volatility = pd.Series(np.ones(len(abs_returns)), index=abs_returns.index)

        weights = self._compute_financial_weights(abs_returns, volatility)
        
        tprint_info(f"📊 Processing Huber Teacher on {len(X_full.columns)} potential features...")

        # --- Prepare Huber Teacher ---
        huber_outputs = prepare_huber_teacher_outputs(
            X_train=X_full,
            y_train=y_true,
            pruning_percentile=15,
            corr_threshold=0.7
        )

        self.selected_features = huber_outputs['selected_features']
        self.huber_model = huber_outputs.get('huber_teacher', huber_outputs.get('huber_model'))
        self.huber_scaler = huber_outputs['scaler']
        self.huber_feature_columns = X_full.columns

        X_pruned = X_full[self.selected_features]
        warm_start_train = huber_outputs['warm_start']['train']

        # Extract monotonic constraints (handle dict return from updated Huber)
        if isinstance(huber_outputs['monotonic_constraints'], dict):
             if 'monotonic_constraints_details' in huber_outputs:
                 # Use the pre-calculated selected list (aligned with selected_features)
                 monotonic_cst_tuple = huber_outputs['monotonic_constraints_details']['lightgbm_selected']
             else:
                 # Fallback: reconstruct
                 monotonic_cst_tuple = [huber_outputs['monotonic_constraints'][f] for f in self.selected_features]
        else:
             monotonic_cst_tuple = huber_outputs['monotonic_constraints']

        # Ensure tuple format
        monotonic_cst_tuple = tuple(int(x) for x in monotonic_cst_tuple)

        interaction_cst = huber_outputs['interaction_constraints']

        # DEBUG: Inspect constraints for XGBoost issue
        if interaction_cst:
            tprint_info(f"DEBUG: Interaction Constraints Sample: {interaction_cst[:2]}")
            tprint_info(f"DEBUG: Interaction Type: {type(interaction_cst)}")
            if len(interaction_cst) > 0:
                tprint_info(f"DEBUG: Interaction Inner Type: {type(interaction_cst[0])}, Element Type: {type(interaction_cst[0][0])}")
            tprint_info(f"DEBUG: Pruned Features: {self.selected_features[:5]}")

        tprint_info(f"✨ Feature pruning: {len(self.selected_features)}/{len(X_full.columns)} features selected")
        
        base_predictions = {}
        
        # --- 1. ExtraTrees (with monotonic constraints) ---
        tprint_info(f"📊 Training ExtraTrees (with monotonic constraints)...")
        # Check if monotonic_cst is supported (sklearn 1.4+)
        try:
            self.extratrees.set_params(monotonic_cst=monotonic_cst_tuple)
        except ValueError:
            tprint_warning("ExtraTrees monotonic_cst parameter not supported or invalid. Skipping constraints.")
            self.extratrees.set_params(monotonic_cst=None)
        except Exception:
             # In case of older sklearn versions that don't accept monotonic_cst in set_params yet
            pass

        self.extratrees.fit(X_pruned, y_true, sample_weight=weights)
        et_preds = self.extratrees.predict(X_pruned)
        base_predictions['extratrees'] = et_preds
        self.calibrators['extratrees'].fit(et_preds, y_true)
        
        # --- 2. LGBM Regressor ---
        tprint_info("📊 Training LGBM (with constraints & warm start)...")
        self.lgbm_model = LGBMRegressor(**self.lgbm_params)
        self.lgbm_model.set_params(
            monotone_constraints=list(monotonic_cst_tuple),
            interaction_constraints=interaction_cst if interaction_cst else None
        )
        self.lgbm_model.fit(
            X_pruned, y_true,
            sample_weight=weights,
            init_score=warm_start_train
        )
        lgbm_raw_preds = self.lgbm_model.predict(X_pruned)
        # For LGBMRegressor, predict() returns the raw prediction (sum of trees).
        # If trained with init_score, the trees model the residual.
        # We must add the init_score (warm_start) manually to get the full prediction.
        lgbm_preds = lgbm_raw_preds + warm_start_train
        base_predictions['lgbm'] = lgbm_preds
        self.calibrators['lgbm'].fit(lgbm_preds, y_true)

        # --- 3. XGB Regressor ---
        tprint_info("📊 Training XGBoost (with constraints & warm start)...")
        self.xgb_model = XGBRegressor(**self.xgb_params)

        # Try-catch for XGBoost constraints failure
        try:
            self.xgb_model.set_params(
                monotone_constraints=monotonic_cst_tuple,
                interaction_constraints=interaction_cst if interaction_cst else None
            )

            self.xgb_model.fit(
                X_pruned, y_true,
                sample_weight=weights,
                base_margin=warm_start_train
            )
        except (ValueError, KeyError) as e:
            tprint_error(f"XGBoost constraints failed: {e}. Retrying without interaction constraints.")
            self.xgb_model.set_params(interaction_constraints=None)
            self.xgb_model.fit(
                X_pruned, y_true,
                sample_weight=weights,
                base_margin=warm_start_train
            )

        xgb_preds = self.xgb_model.predict(X_pruned, base_margin=warm_start_train)
        base_predictions['xgboost'] = xgb_preds
        self.calibrators['xgboost'].fit(xgb_preds, y_true)

        # --- 4. CatBoost Regressor ---
        tprint_info("📊 Training CatBoost (with constraints & warm start)...")
        self.catboost_model = CatBoostRegressor(**self.catboost_params)
        # Monotonic constraints string format for CatBoost: "1:1,2:-1,3:0" or list
        # It accepts list.
        self.catboost_model.set_params(monotone_constraints=list(monotonic_cst_tuple))

        self.catboost_model.fit(
            X_pruned, y_true,
            sample_weight=weights,
            baseline=warm_start_train
        )

        cb_raw_preds = self.catboost_model.predict(X_pruned)
        catboost_preds = warm_start_train + cb_raw_preds
        base_predictions['catboost'] = catboost_preds
        self.calibrators['catboost'].fit(catboost_preds, y_true)
        
        # --- Consensus ---
        consensus_raw = (
            self.consensus_weights['extratrees'] * base_predictions['extratrees'] +
            self.consensus_weights['lgbm'] * base_predictions['lgbm'] +
            self.consensus_weights['xgboost'] * base_predictions['xgboost'] +
            self.consensus_weights['catboost'] * base_predictions['catboost']
        )
        
        # Calibrate Consensus
        self.consensus_calibrator.fit(consensus_raw, y_true)
        consensus_calibrated = self.consensus_calibrator.transform(consensus_raw)
        
        self.is_fitted = True
        self.final_predictions_ = consensus_calibrated
        self.feature_names = self.selected_features # Save for consistency check
        
        metrics = {
            'consensus_weighted_logloss': log_loss(y_true, consensus_calibrated, sample_weight=weights),
            'n_features_total': len(X_full.columns),
            'n_features_pruned': len(self.selected_features),
            'mean_conviction': consensus_calibrated.mean()
        }
        
        tprint_success(f"✅ Layer 4 Engine trained: WL={metrics['consensus_weighted_logloss']:.4f}, Features={metrics['n_features_pruned']}")
        return metrics

    def predict_bet_size(self, df: pd.DataFrame, market_features: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("RiskEngine must be fitted")
            
        prob_feats = self._extract_layer3_prob_features(df)
        entropy_feats = self._extract_entropy_features(df) if ENTROPY_BARS_AVAILABLE else pd.DataFrame(index=df.index)
        
        X_full = pd.concat([prob_feats, entropy_feats, market_features], axis=1).fillna(0)

        # --- Dual Chaser Feature Generation ---
        if self.stable_chaser is not None and self.dual_chaser_scaler is not None:
            try:
                # Filter out performance features
                chaser_cols = [c for c in X_full.columns if not c.startswith('perf_') and not c.startswith('meta_')]
                X_for_chaser = X_full[chaser_cols]

                if isinstance(X_for_chaser, pd.DataFrame):
                    X_chaser_scaled = pd.DataFrame(
                        self.dual_chaser_scaler.transform(X_for_chaser),
                        columns=X_for_chaser.columns,
                        index=X_for_chaser.index
                    )
                else:
                    X_chaser_scaled = self.dual_chaser_scaler.transform(X_for_chaser)

                # Predict on test set using full models
                p_stable = pd.Series(self.stable_chaser.predict(X_chaser_scaled.values), index=X_full.index)
                p_agg = pd.Series(self.aggressive_chaser.predict(X_chaser_scaled.values), index=X_full.index)

                # Generate features (meta-health skipped or using placeholder as trade returns unknown for test)
                chaser_feats = generate_dual_chaser_features(
                    df=df,
                    p_stable=p_stable,
                    p_agg=p_agg,
                    trade_returns=None
                )

                X_full = pd.concat([X_full, chaser_feats], axis=1)
            except Exception as e:
                tprint_warning(f"⚠️ Dual Chaser prediction failed: {e}")
                import traceback
                traceback.print_exc()
                pass

        # Downcast to float32
        X_full = downcast_float(X_full)

        if self.huber_feature_columns is not None:
            X_full = X_full.reindex(columns=self.huber_feature_columns, fill_value=0.0)

        # Prepare Warm Start for this new data
        X_scaled = pd.DataFrame(self.huber_scaler.transform(X_full), columns=X_full.columns)
        warm_start = self.huber_model.predict(X_scaled)
        
        # Select features
        X_pruned = X_full[self.selected_features]

        # 1. ExtraTrees
        et_preds = self.extratrees.predict(X_pruned)
        et_cal = self.calibrators['extratrees'].transform(et_preds)
        
        # 2. LGBM
        lgbm_raw = self.lgbm_model.predict(X_pruned)
        lgbm_preds = lgbm_raw + warm_start
        lgbm_cal = self.calibrators['lgbm'].transform(lgbm_preds)

        # 3. XGBoost
        xgb_preds = self.xgb_model.predict(X_pruned, base_margin=warm_start)
        xgb_cal = self.calibrators['xgboost'].transform(xgb_preds)
        
        # 4. CatBoost
        cb_raw = self.catboost_model.predict(X_pruned)
        catboost_preds = cb_raw + warm_start
        cb_cal = self.calibrators['catboost'].transform(catboost_preds)

        # Consensus
        consensus = (
            self.consensus_weights['extratrees'] * et_cal +
            self.consensus_weights['lgbm'] * lgbm_cal +
            self.consensus_weights['xgboost'] * xgb_cal +
            self.consensus_weights['catboost'] * cb_cal
        )
        
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
    cfg = config or {}
    
    try:
        # Determine date range from existing data
        if not df.empty and hasattr(df.index, 'min') and hasattr(df.index, 'max'):
            start_date = df.index.min().strftime('%Y-%m-%d')
            end_date = df.index.max().strftime('%Y-%m-%d')
        else:
            # Default to last 30 days if no date range available
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
        
        # --- Caching Mechanism ---
        cache_dir = Path("cache/entropy_bars")
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Create hash of parameters to uniquely identify cache
        cache_key = f"{symbol}_{exchange}_{start_date}_{end_date}_{cfg.get('entropy_bins', 10)}_{cfg.get('entropy_window', 100)}_{cfg.get('entropy_target_minutes', 15)}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        cache_file = cache_dir / f"entropy_features_{cache_hash}.parquet"

        if cache_file.exists():
            tprint_info(f"⚡ Layer 4: Loading entropy features from cache: {cache_file}")
            try:
                entropy_features_all = pd.read_parquet(cache_file)

                # Filter to match current df index
                # Ensure index is datetime
                if not isinstance(entropy_features_all.index, pd.DatetimeIndex):
                    entropy_features_all.index = pd.to_datetime(entropy_features_all.index)

                # Reindex to match current df
                # This is fast
                entropy_features = entropy_features_all.reindex(df.index, method='ffill').fillna(0)

                enhanced_df = df.join(entropy_features, rsuffix='_entropy')
                return enhanced_df, pd.DataFrame() # We don't return raw bars from cache to save space
            except Exception as e:
                tprint_warning(f"⚠️ Failed to load cache, regenerating: {e}")

        # Try to fetch 1-min data if allowed
        use_proxy = False
        min_data = None
        
        if ENTROPY_BARS_AVAILABLE:
            tprint_info("🔧 Layer 4: Fetching 1-minute data for entropy bar generation")
            try:
                min_data = fetch_1min_data_for_entropy_bars(
                    symbol=symbol,
                    exchange=exchange,
                    start_date=start_date,
                    end_date=end_date,
                    data_dir=cfg.get('data_dir', 'historical_data')
                )
            except Exception as e:
                tprint_warning(f"⚠️ Layer 4: Failed to fetch 1-min data: {e}. Switching to Proxy Entropy.")
                use_proxy = True
        else:
            tprint_info("ℹ️ Layer 4: Entropy Bars module not available. Switching to Proxy Entropy.")
            use_proxy = True

        if use_proxy or min_data is None or min_data.empty:
            if not use_proxy:
                tprint_warning("⚠️ Layer 4: No 1-minute data available. Switching to Proxy Entropy.")

            # --- PROXY ENTROPY CALCULATION ---
            tprint_info("🔄 Layer 4: Calculating Proxy Entropy (Numba Optimized) on base timeframe...")

            # Calculate returns on the base dataframe
            if 'close' in df.columns:
                price_col = 'close'
            elif 'Close' in df.columns:
                price_col = 'Close'
            else:
                tprint_error("❌ Layer 4: No price column found for proxy entropy.")
                return df, pd.DataFrame()

            # Calculate log returns
            prices = df[price_col].values.astype(np.float64)
            returns = np.zeros_like(prices)
            returns[1:] = np.diff(np.log(prices + 1e-9))

            # Use Numba function
            proxy_entropy = compute_proxy_entropy_numba(
                returns,
                window=cfg.get('entropy_window', 100),
                n_bins=cfg.get('entropy_bins', 10)
            )

            entropy_features = pd.DataFrame(index=df.index)
            entropy_features['proxy_entropy'] = proxy_entropy

            # Cache the proxy features too
            try:
                entropy_features.to_parquet(cache_file)
                tprint_info(f"💾 Layer 4: Cached proxy entropy features to {cache_file}")
            except Exception as e:
                tprint_warning(f"⚠️ Failed to cache proxy entropy features: {e}")

            enhanced_df = df.join(entropy_features, rsuffix='_entropy')
            return enhanced_df, pd.DataFrame()
        
        # Generate entropy bars (Normal Path)
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
            tprint_warning("⚠️ Layer 4: Failed to generate entropy bars. Using empty features.")
            return df, pd.DataFrame()
        
        # Calculate specialized entropy features
        tprint_info("🎯 Layer 4: Calculating specialized entropy features")
        entropy_features = calculate_specialized_entropy_features(
            entropy_bars=entropy_bars,
            base_model_updates=df,  # Use df as proxy for base model updates
            specialist_prices=df['close'] if 'close' in df.columns else None,
            volatility_window=cfg.get('volatility_window', 20)
        )
        
        # Prepare feature set to merge
        features_to_merge = entropy_features.copy()
        
        # Add entropy bar OHLCV data as additional columns
        entropy_ohlcv_cols = ['close', 'volume', 'n_minutes', 'entropy_contribution']
        for col in entropy_ohlcv_cols:
            if col in entropy_bars.columns:
                 features_to_merge[f'entropy_{col}'] = entropy_bars[col]

        # Cache the raw features (before reindexing to specific df)
        try:
            features_to_merge.to_parquet(cache_file)
            tprint_info(f"💾 Layer 4: Cached entropy features to {cache_file}")
        except Exception as e:
            tprint_warning(f"⚠️ Failed to cache entropy features: {e}")

        # Merge entropy features back to main dataframe
        # Vectorized reindex and ffill using join/reindex
        # reindex with method='ffill' is efficient

        aligned_features = features_to_merge.reindex(df.index, method='ffill').fillna(0)
        enhanced_df = df.join(aligned_features)
        
        tprint_success(f"✅ Layer 4: Integrated entropy bars: {len(entropy_bars)} bars, {len(entropy_features.columns)} features")
        
        return enhanced_df, entropy_bars
        
    except Exception as e:
        tprint_error(f"❌ Layer 4: Error integrating entropy bars: {e}")
        return df, pd.DataFrame()


class MetaLearnerFeatures:
    """
    Generates structural market features for Layer 4 meta-learning.
    Uses Numba-optimized SADF and CUSUM scores to detect structural breaks and regimes.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.window_sadf = self.config.get('window_sadf', 100)

    def generate(self, df: pd.DataFrame, raw_price_col: str = 'close', **kwargs) -> pd.DataFrame:
        """
        Generate structural features.
        """
        if raw_price_col not in df.columns:
            return pd.DataFrame(index=df.index)

        # Calculate returns
        prices = df[raw_price_col].values.astype(np.float64)
        returns = np.zeros_like(prices)
        returns[1:] = np.diff(np.log(prices + 1e-9))

        # Volatility (for weighting)
        vol_window = self.config.get('volatility_window', 20)
        volatility = pd.Series(returns).rolling(window=vol_window, min_periods=1).std().fillna(0.0).values

        # SADF Scores (Numba Optimized)
        sadf_scores = rolling_sadf_score_numba(returns, self.window_sadf)

        # CUSUM Scores (Numba Optimized)
        mean_ret = np.mean(returns)
        cusum_scores = rolling_cusum_scores_numba(returns, mean_ret)

        # Normalize
        max_sadf = np.max(sadf_scores)
        sadf_norm = sadf_scores / (max_sadf + 1e-9) if max_sadf > 0 else sadf_scores

        max_cusum = np.max(cusum_scores)
        cusum_norm = cusum_scores / (max_cusum + 1e-9) if max_cusum > 0 else cusum_scores

        features = pd.DataFrame(index=df.index)
        features['volatility'] = volatility
        features['sadf_score_norm'] = sadf_norm
        features['cusum_score_norm'] = cusum_norm

        # Integrate Anti-Explosion Features
        try:
            processed = apply_layer2_price_processing(df, price_col=raw_price_col, enable_price_features=True)
            # Select new features
            new_cols = [c for c in processed.columns if c not in df.columns]
            if new_cols:
                features = pd.concat([features, processed[new_cols]], axis=1)
                # tprint_info(f"   ✨ Layer 4: Added {len(new_cols)} Anti-Explosion features")
        except Exception as e:
            tprint_warning(f"   ⚠️ Layer 4: Anti-Explosion feature generation failed: {e}")

        return features


class ModelPerformanceFeatures:
    """
    Generates features based on the historical performance of the base models.
    Includes 'Skill' metrics and 'Risk-off' pressure indicators.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        # Configuration for Skill Feature
        self.skill_window = self.config.get('perf_skill_window', 100)  # Heavy smoothing
        self.skill_embargo = self.config.get('perf_skill_embargo', 2)  # Strict embargo

        # Configuration for Risk-off Feature
        self.risk_window = self.config.get('perf_risk_window', 50)
        self.risk_embargo = self.config.get('perf_risk_embargo', 2)
        self.dd_scaling = self.config.get('perf_dd_scaling', 0.1) # 'c' in tanh(dd/c)

    def generate(self, df: pd.DataFrame, pred_col: str, target_col: str) -> pd.DataFrame:
        if pred_col not in df.columns or target_col not in df.columns:
            return pd.DataFrame(index=df.index)

        # Removed current_drawdown, recent_sharpe, hit_rate features as requested.
        # Returning empty DataFrame with correct index to minimize impact on downstream concatenation
        return pd.DataFrame(index=df.index)


def train_layer4_simple_multimodel(
    oof_df: pd.DataFrame,
    market_data: pd.DataFrame,
    base_model_predictions: Optional[pd.DataFrame] = None,
    target_col: str = 'realized_return',
    n_folds: int = 5,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Train Layer 4 using Layer 3 probability outputs, entropy bars, and market features.
    """
    cfg = config or {}
    tprint_info("🚀 Starting Layer 4 Training (Layer 3 Probabilities + Entropy Bars)...")
    
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
    # Use the local MetaLearnerFeatures class which is now properly implemented
    generator = MetaLearnerFeatures(config=config)
    market_features = generator.generate(
        df=oof_df.join(market_data, how='left', rsuffix='_mkt'),
        raw_price_col='close'
    )
    
    # Generate model performance features
    perf_generator = ModelPerformanceFeatures(config=config)
    # Prefer 'meta_prob' (L3 output) for performance tracking
    pred_col = 'meta_prob' if 'meta_prob' in oof_df.columns else None

    if pred_col:
        perf_features = perf_generator.generate(
            df=oof_df,
            pred_col=pred_col,
            target_col=target_col
        )
        market_features = pd.concat([market_features, perf_features], axis=1)
    else:
        tprint_warning("⚠️ Layer 4: 'meta_prob' not found, skipping performance features.")

    y_binary = (oof_df[target_col] > 0).astype(int)
    abs_returns = oof_df[target_col].abs()
    
    kf = KFold(n_splits=n_folds, shuffle=False)
    oof_bet_sizes = np.zeros(len(oof_df))
    
    # Instantiate engine once, but models will be re-trained per fold
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
    
    # Final fit on full data
    final_metrics = engine.train(oof_df, market_features, y_binary, abs_returns)
    
    oof_df_out = oof_df.copy()
    oof_df_out['layer4_prob'] = oof_bet_sizes
    
    # Add entropy bars information to metrics
    if not entropy_bars_df.empty:
        final_metrics['entropy_bars_count'] = len(entropy_bars_df)
        final_metrics['entropy_features_count'] = len([col for col in oof_df_out.columns if col.startswith(('staleness_', 'drift_', 'lz_', 'trend_', 'entropy_', 'proxy_entropy'))])
    
    return oof_df_out, final_metrics
