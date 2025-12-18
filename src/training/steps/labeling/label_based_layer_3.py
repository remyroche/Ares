import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.metrics import (
    log_loss,
    brier_score_loss,
    roc_auc_score,
    accuracy_score
)
from typing import List, Tuple, Optional, Any, Dict
import copy
import shap

from src.training.steps.labeling.feature_generation_meta_labeling_step import (
    create_meta_features,
)
from src.training.steps.labeling.label_based_pipeline import (
    select_features_with_quality,
)
from src.training.steps.labeling.mda_shap_feature_selection import (
    run_mda_shap_feature_selection,
)
from src.feature_generation.categories.ensemble_disagreement import (
    calculate_ensemble_disagreement_features,
)

from src.utils.purged_kfold import PurgedKFoldTime


def _fast_expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true_arr = np.asarray(y_true, dtype=float).reshape(-1)
    y_prob_arr = np.asarray(y_prob, dtype=float).reshape(-1)

    mask = np.isfinite(y_true_arr) & np.isfinite(y_prob_arr)
    if not np.any(mask):
        return 0.0

    y_true_arr = y_true_arr[mask]
    y_prob_arr = y_prob_arr[mask]
    n = int(y_prob_arr.size)
    if n <= 0:
        return 0.0

    y_prob_arr = np.clip(y_prob_arr, 0.0, 1.0)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(y_prob_arr, bin_edges, right=True) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    counts = np.bincount(bin_idx, minlength=n_bins).astype(float)
    sum_prob = np.bincount(bin_idx, weights=y_prob_arr, minlength=n_bins).astype(float)
    sum_true = np.bincount(bin_idx, weights=y_true_arr, minlength=n_bins).astype(float)

    nonzero = counts > 0
    if not np.any(nonzero):
        return 0.0

    mean_prob = np.zeros(n_bins, dtype=float)
    mean_true = np.zeros(n_bins, dtype=float)
    mean_prob[nonzero] = sum_prob[nonzero] / counts[nonzero]
    mean_true[nonzero] = sum_true[nonzero] / counts[nonzero]

    return float(np.sum((counts[nonzero] / n) * np.abs(mean_prob[nonzero] - mean_true[nonzero])))

def layer3_analyst_lgbm(
    oof_df: pd.DataFrame,
    base_model_cols: List[str],
    target_col: str,
    train_split_date: Optional[str] = None,
    sample_weight: Optional[np.ndarray] = None,
    # New arguments for Scheme comparison
    layer1_weight: Optional[np.ndarray] = None,
    layer2_weight: Optional[np.ndarray] = None,
    net_returns: Optional[np.ndarray] = None,
    market_data: Optional[pd.DataFrame] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Any]:
    """
    Transforms diverse Base Model scores into a single Calibrated Probability using LGBM.

    Performs a comparison of 7 specified weighting schemes using ScoreL3 logic:
      ScoreL3 = 100*(AUC-0.5) + 50*(0.693-LogLoss) - 200*ECE

    Selects the best scheme and trains the final production model.
    """
    print(f"\n{'='*60}")
    print("LAYER 3: ANALYST META-MODEL (LGBM + CALIBRATION) - COMPARATIVE MODE")
    print(f"{'='*60}")

    df = oof_df.copy()

    cfg = config if isinstance(config, dict) else {}
    enable_timing = False
    try:
        enable_timing = bool(cfg.get('layer3_timing', False))
    except Exception:
        enable_timing = False
    t0_all = time.perf_counter() if enable_timing else None

    # ---------------------------------------------------------
    # 1. Feature Engineering: Ensemble Disagreement
    # ---------------------------------------------------------
    print("<< Generating Disagreement Features...")

    meta_features = []
    if not base_model_cols:
          print("⚠️ No base models provided for Layer 3 feature engineering!")
    else:
        df[base_model_cols] = df[base_model_cols].fillna(0.5)

        prob_dict = {str(c): df[c].astype(float).values for c in base_model_cols}
        pred_dict = {str(c): (df[c].astype(float).values - 0.5) for c in base_model_cols}

        try:
            disagree = calculate_ensemble_disagreement_features(
                model_predictions=pred_dict,
                model_probabilities=prob_dict,
                model_confidences=None,
                feature_names=None,
                logger=None,
            )
        except Exception:
            disagree = {}

        disagree_feature_names = [
            "prediction_dispersion",
            "confidence_gap",
            "uncertainty",
            "prediction_range",
            "avg_divergence",
            "max_confidence",
            "disagreement_rate",
        ]
        disagree_cols = []
        for k in disagree_feature_names:
            col = f"ens_{k}"
            disagree_cols.append(col)
            try:
                v = disagree.get(k)
                if isinstance(v, pd.Series):
                    df[col] = pd.to_numeric(v.values, errors="coerce")
                else:
                    df[col] = 0.0
            except Exception:
                df[col] = 0.0

        df[disagree_cols] = df[disagree_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        meta_features = list(dict.fromkeys(base_model_cols + disagree_cols))

    additional_feature_df = pd.DataFrame(index=df.index)
    if market_data is not None and isinstance(market_data, pd.DataFrame) and not market_data.empty:
        try:
            volume_available = bool('volume' in market_data.columns)

            signals_dummy = pd.DataFrame(index=market_data.index)
            signals_dummy['consensus'] = 0.0

            meta_full = create_meta_features(
                df=market_data,
                signals=signals_dummy,
                volume_available=volume_available,
                include_raw_signals=False,
                use_kalman=True,
            )

            try:
                enable_nn = bool(cfg.get('enable_nn_sequence_embeddings_layer3', cfg.get('enable_nn_sequence_embeddings', False)))
            except Exception:
                enable_nn = False

            if enable_nn:
                try:
                    from src.training.steps.labeling.short_nn_sequence_template import (
                        generate_nn_sequence_embeddings,
                        load_cached_nn_embeddings,
                    )

                    nn_cfg = cfg.get('nn_sequence_encoder_layer3')
                    if not isinstance(nn_cfg, dict):
                        nn_cfg = cfg.get('nn_sequence_encoder')
                    if not isinstance(nn_cfg, dict):
                        nn_cfg = {}

                    cache_path = nn_cfg.get('cache_path')
                    seq_len = int(nn_cfg.get('seq_len', 24))
                    embed_dim = int(nn_cfg.get('embed_dim', 8))
                    device = str(nn_cfg.get('device', 'cpu'))
                    encoder_type = str(nn_cfg.get('encoder_type', 'stacked'))
                    use_conv = bool(nn_cfg.get('use_conv', True))
                    use_lstm = bool(nn_cfg.get('use_lstm', True))
                    use_attention = bool(nn_cfg.get('use_attention', False))
                    pretrained_path = nn_cfg.get('pretrained_path', None)

                    nn_embed = None
                    if cache_path:
                        try:
                            nn_embed = load_cached_nn_embeddings(
                                cache_path=str(cache_path),
                                target_index=market_data.index,
                            )
                        except Exception:
                            nn_embed = None

                    if nn_embed is None:
                        nn_embed = generate_nn_sequence_embeddings(
                            market_data=market_data,
                            encoder_type=encoder_type,
                            seq_len=seq_len,
                            embed_dim=embed_dim,
                            use_conv=use_conv,
                            use_lstm=use_lstm,
                            use_attention=use_attention,
                            device=device,
                            pretrained_path=pretrained_path,
                        )

                    if nn_embed is not None and hasattr(nn_embed, 'columns') and len(nn_embed) > 0:
                        nn_embed = nn_embed.reindex(meta_full.index)
                        new_cols = [c for c in nn_embed.columns if c not in meta_full.columns]
                        if new_cols:
                            meta_full = pd.concat([meta_full, nn_embed[new_cols]], axis=1)
                except Exception:
                    pass

            meta_selected = meta_full
            try:
                enable_de_prado = bool(cfg.get('enable_feature_selection_layer3', True))
            except Exception:
                enable_de_prado = True

            if enable_de_prado and meta_full is not None and not meta_full.empty:
                try:
                    target_n = int(cfg.get('target_feature_count_layer3', cfg.get('target_feature_count', 70)))
                except Exception:
                    target_n = 70
                try:
                    corr_thr = float(cfg.get('feature_correlation_threshold_layer3', cfg.get('feature_correlation_threshold', 0.85)))
                except Exception:
                    corr_thr = 0.85

                try:
                    meta_selected, _ = select_features_with_quality(
                        df_features=meta_full,
                        target_n=target_n,
                        correlation_threshold=corr_thr,
                        generate_horizons=bool(cfg.get('enable_multi_horizon_features', True)),
                        horizon_config=cfg.get('feature_horizon_config', {
                            'Short': 5,
                            'Medium': 20,
                            'Long': 60,
                        }),
                        enable_cross_features=bool(cfg.get('enable_cross_features', True)),
                        market_data=market_data,
                        config=cfg,
                        use_hierarchical=bool(cfg.get('use_hierarchical_selection', True)),
                        use_lgbm_sweep=bool(cfg.get('use_lgbm_sweep', True)),
                        lgbm_lookahead=int(cfg.get('lgbm_sweep_lookahead', 4)),
                        lgbm_max_features=int(cfg.get('lgbm_max_features', 300)),
                        quality_drop_percentile=float(cfg.get('quality_drop_percentile', 20.0)),
                        symbol=str(cfg.get('symbol', '')),
                        exchange=str(cfg.get('exchange', '')),
                        timeframe=str(cfg.get('timeframe', '')),
                        use_cache=bool(cfg.get('use_feature_selection_cache', True)),
                        force_recompute=bool(cfg.get('force_recompute_features', False)),
                    )
                except Exception:
                    meta_selected = meta_full

            additional_feature_df = meta_selected.reindex(df.index).fillna(0.0)
        except Exception:
            additional_feature_df = pd.DataFrame(index=df.index)

    if additional_feature_df is not None and not additional_feature_df.empty:
        new_cols = [c for c in additional_feature_df.columns if c not in df.columns]
        if new_cols:
            df = pd.concat([df, additional_feature_df[new_cols]], axis=1)

    try:
        enable_mda_shap = bool(cfg.get('enable_mda_shap_selection_layer3', cfg.get('enable_mda_shap_selection', True)))
    except Exception:
        enable_mda_shap = True

    selected_additional_features = list(additional_feature_df.columns)
    if enable_mda_shap and additional_feature_df is not None and not additional_feature_df.empty:
        try:
            geo_df_for_target = df[base_model_cols] if base_model_cols else pd.DataFrame(index=df.index)
            geo_mean = geo_df_for_target.mean(axis=1, skipna=True)
            geo_count = geo_df_for_target.notna().sum(axis=1)
            y_joint = (geo_mean >= 0.5).astype(float)
            y_joint.loc[geo_count <= 0] = np.nan

            w_sel = None
            try:
                if layer1_weight is not None and layer2_weight is not None and net_returns is not None:
                    w1 = np.asarray(layer1_weight, dtype=float).reshape(-1)
                    w2 = np.asarray(layer2_weight, dtype=float).reshape(-1)
                    rr = np.asarray(net_returns, dtype=float).reshape(-1)

                    n = int(len(df))
                    if w1.size != n:
                        w1 = w1[:n] if w1.size >= n else np.pad(w1, (n - w1.size, 0), constant_values=1.0)
                    if w2.size != n:
                        w2 = w2[:n] if w2.size >= n else np.pad(w2, (n - w2.size, 0), constant_values=1.0)
                    if rr.size != n:
                        rr = rr[:n] if rr.size >= n else np.pad(rr, (n - rr.size, 0), constant_values=0.0)

                    w_sel_arr = (w1 * w2 * np.log1p(np.abs(rr))).astype(float)
                    w_sel_arr = np.where(np.isfinite(w_sel_arr) & (w_sel_arr > 0.0), w_sel_arr, 1.0)
                    w_sel = pd.Series(w_sel_arr, index=df.index)
                    if float(w_sel.mean()) > 1e-9:
                        w_sel = w_sel / float(w_sel.mean())
            except Exception:
                w_sel = None

            mda_shap_cfg = cfg.get('mda_shap_config_layer3')
            if not isinstance(mda_shap_cfg, dict):
                mda_shap_cfg = {
                    'model_type': 'rf',
                    'n_folds': int(cfg.get('cv_splits', 5)),
                    'pre_filters': {
                        'enable_lgbm_mdi_filter': True,
                        'enable_correlation_filter': True,
                        'enable_variance_filter': True,
                        'enable_anova_filter': True,
                    },
                    'corr_threshold': 0.85,
                    'top_clusters': 8,
                    'shap_sample_size': min(1000, int(np.sum(y_joint.notna()))),
                    'elbow_min_features': int(cfg.get('layer3_elbow_min_features', 40)),
                    'max_selected_features': int(cfg.get('layer3_max_selected_features', 70)),
                    'verbose': True,
                }

            extractor_cfg = cfg.get('regime_leaf_extractor_config')
            if not isinstance(extractor_cfg, dict):
                extractor_cfg = {}
            
            # Enforce defaults if empty to ensure we get features
            if not extractor_cfg.get('enabled_targets'):
                 # Default to Volatility and Trend Efficiency if not specified
                 extractor_cfg['enabled_targets'] = ['regime_volatility', 'regime_trend_efficiency']

            mda_shap_cfg.setdefault(
                'regime_leaf_config',
                {
                    'enabled': bool(cfg.get('enable_regime_leaf_features', True)),
                    'market_data': market_data,
                    'X_base': None,
                    'extractor_config': extractor_cfg,
                    'random_state': int(cfg.get('random_state', 42)),
                    'verbose': True,
                },
            )

            # NEW: Generate Regime Features BEFORE selection so they are available as candidates
            try:
                rl_cfg = mda_shap_cfg.get('regime_leaf_config') if isinstance(mda_shap_cfg, dict) else None
                if isinstance(rl_cfg, dict) and bool(rl_cfg.get('enabled', False)) and market_data is not None:
                    from src.training.steps.labeling.regime_leaf_feature_extractor import extract_regime_leaf_onehot_features
                    
                    print("<< Extracting Regime Leaf Features...")
                    rl_df = extract_regime_leaf_onehot_features(
                        X=additional_feature_df,
                        market_data=market_data,
                        config=extractor_cfg,
                        random_state=int(rl_cfg.get('random_state', 42)),
                        verbose=bool(rl_cfg.get('verbose', True)),
                    )
                    
                    if rl_df is not None and not getattr(rl_df, 'empty', True):
                        # Align and merge
                        rl_df = rl_df.reindex(additional_feature_df.index).fillna(0.0)
                        # Drop columns that already exist to avoid duplicates
                        new_rl_cols = [c for c in rl_df.columns if c not in additional_feature_df.columns]
                        if new_rl_cols:
                            additional_feature_df = pd.concat([additional_feature_df, rl_df[new_rl_cols]], axis=1)
                            print(f"   Added {len(new_rl_cols)} regime features for selection.")
            except Exception as e:
                print(f"⚠️ Failed to generate Regime Features: {e}")

            selected_feats, selection_results = run_mda_shap_feature_selection(
                X=additional_feature_df,
                y=y_joint,
                target_sample_weight=w_sel,
                config=mda_shap_cfg,
                artifact_router=None,
                pipeline_context={
                    'symbol': cfg.get('symbol'),
                    'exchange': cfg.get('exchange'),
                    'timeframe': cfg.get('timeframe'),
                    'direction': cfg.get('direction'),
                },
            )

            # Update features based on selection
            final_sel = [f for f in list(selected_feats or []) if isinstance(f, str) and f in additional_feature_df.columns]
            if final_sel:
                selected_additional_features = final_sel

            # Ensure selected features are in df
            for col in selected_additional_features:
                if col not in df.columns and col in additional_feature_df.columns:
                    df[col] = additional_feature_df[col]

        except Exception as e:
            print(f"⚠️ Feature Selection Failed: {e}")
            selected_additional_features = list(additional_feature_df.columns)

    additional_features = [c for c in df.columns if c.startswith('vol_') or c.startswith('trend_') or c in ['volatility_1d']]
    meta_features += [c for c in additional_features if c not in meta_features]

    # Clean target
    try:
        neutral_value = cfg.get('layer3_neutral_target_value', 0.5) if isinstance(cfg, dict) else 0.5
        neutral_value = float(neutral_value)
    except Exception:
        neutral_value = 0.5
    try:
        y_num = pd.to_numeric(df[target_col], errors='coerce').astype(float)
        df[target_col] = y_num
        if np.isfinite(neutral_value):
            df.loc[np.isclose(df[target_col].astype(float), neutral_value, atol=1e-12), target_col] = np.nan
    except Exception:
        pass
    df = df.dropna(subset=[target_col])

    # Tight alignment: require Series aligned to oof_df.index.
    # Avoid silent truncation/padding because it invalidates OOF + scheme selection.
    def _require_series_aligned(vec, name: str) -> pd.Series:
        if vec is None:
            raise ValueError(f"{name} is required for Layer3 scheme comparison and must be a pd.Series aligned to oof_df.index")
        if not isinstance(vec, pd.Series):
            raise TypeError(f"{name} must be a pd.Series aligned to oof_df.index (got {type(vec)})")
        if not vec.index.equals(oof_df.index):
            raise ValueError(
                f"{name} index mismatch vs oof_df.index. "
                "Pass a pd.Series with exactly the same DatetimeIndex as oof_df."
            )
        s = pd.to_numeric(vec, errors="coerce").astype(float)
        s = s.replace([np.inf, -np.inf], np.nan)
        return s

    w_l1_s = _require_series_aligned(layer1_weight, "layer1_weight")
    w_l2_s = _require_series_aligned(layer2_weight, "layer2_weight")
    ret_s = _require_series_aligned(net_returns, "net_returns")

    # After target dropna, align by index (no reordering)
    w_l1 = w_l1_s.reindex(df.index).values
    w_l2 = w_l2_s.reindex(df.index).values
    ret_vec = ret_s.reindex(df.index).values

    if len(w_l1) != len(df) or len(w_l2) != len(df) or len(ret_vec) != len(df):
        raise ValueError("Layer3 internal alignment error: weight/return lengths do not match df after target filtering")

    # Calculate Magnitude Factor: log(1 + max(NetReturns, 0))
    # Only positive returns contribute to magnitude - losses should not boost weight
    magnitude_log = np.log1p(np.clip(ret_vec, 0, None))

    # ---------------------------------------------------------
    # 2. Define Weighting Schemes
    # ---------------------------------------------------------
    schemes = {}

    # Scheme 1: target_sample_weight (layer1)
    schemes["S1_L1"] = w_l1

    # Scheme 2: target_sample_weight * final composite weight (layer2)
    schemes["S2_L1_L2"] = w_l1 * w_l2

    # Scheme 3: final composite weight (layer2)
    schemes["S3_L2"] = w_l2

    # Scheme 4: log(1+NetReturns) for magnitude integration
    schemes["S4_Mag"] = magnitude_log

    # Scheme 5: target_sample_weight * log(1+NetReturns)
    schemes["S5_L1_Mag"] = w_l1 * magnitude_log

    # Scheme 6: final composite weight * log(1+NetReturns)
    schemes["S6_L2_Mag"] = w_l2 * magnitude_log

    # Scheme 7: target_sample_weight * final composite weight * log(1+NetReturns)
    schemes["S7_All"] = w_l1 * w_l2 * magnitude_log

    # Scheme 8: Asymmetric weighting - downweight losing trades (loss aversion)
    loss_mask = ret_vec < 0
    schemes["S8_Asymmetric"] = np.where(
        loss_mask,
        w_l2 * 0.9,  # Downweight losing trades
        w_l2 * 1.1   # Boost winning trades
    )

    # Scheme 9: Class Balanced weighting - compensate for low base rate
    # Ensures winners and losers have equal aggregate weight in training
    try:
        y_bin = (y_values > 0.5).astype(int)
        pos_count = np.sum(y_bin == 1)
        neg_count = np.sum(y_bin == 0)
        if pos_count > 0 and neg_count > 0:
            scale_pos = neg_count / pos_count
            schemes["S9_ClassBalanced"] = np.where(y_bin == 1, w_l2 * scale_pos, w_l2)
        else:
            schemes["S9_ClassBalanced"] = w_l2
    except Exception:
        schemes["S9_ClassBalanced"] = w_l2

    # Normalize all weights to mean=1.0 for stability
    for k in schemes:
        if schemes[k].mean() > 1e-9:
            schemes[k] = schemes[k] / schemes[k].mean()
        else:
            schemes[k] = np.ones_like(schemes[k]) # Fallback for zero weights

    # ---------------------------------------------------------
    # 3. Comparative Evaluation (2-Phase Scheme Pruning)
    # ---------------------------------------------------------
    # Phase 1: Quick screening on fold 1 only for all schemes
    # Phase 2: Full 5-fold evaluation for top 3 schemes
    # This reduces training calls from 105+ to ~66 (37% reduction)
    print("\n>> Phase 1: Quick Screening (8 Schemes, Fold 1 Only)...")

    results = []

    best_score = -float('inf')
    best_scheme_name = None
    best_model_artifacts = None # To store OOF preds and Final Model

    lgbm_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'n_estimators': 200,
        'learning_rate': 0.03,
        'max_depth': 4,
        'num_leaves': 16,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 1.0,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': 1,
        'verbose': -1
    }

    disagree_cols_all = [
        "ens_prediction_dispersion",
        "ens_confidence_gap",
        "ens_uncertainty",
        "ens_prediction_range",
        "ens_avg_divergence",
        "ens_max_confidence",
        "ens_disagreement_rate",
    ]
    pinned = list(dict.fromkeys(base_model_cols + [c for c in disagree_cols_all if c in df.columns]))
    selected_additional_features = [c for c in selected_additional_features if c in df.columns]
    meta_features = list(dict.fromkeys(pinned + selected_additional_features))

    X = df[meta_features]
    y = df[target_col]

    X_values = X.to_numpy(copy=False)
    y_values = y.to_numpy(copy=False)

    # De Prado-style: purged + embargoed sequential folds.
    # Default purge/embargo are derived from bar time delta and lookahead horizon.
    try:
        n_splits = int(cfg.get('cv_splits', 5))
    except Exception:
        n_splits = 5
    n_splits = int(max(2, n_splits))

    # Infer bar duration
    bar_td = None
    try:
        if isinstance(df.index, pd.DatetimeIndex) and len(df.index) >= 3:
            deltas = df.index.to_series().diff().dropna()
            if len(deltas) > 0:
                bar_td = deltas.median()
    except Exception:
        bar_td = None

    try:
        purge_bars = int(cfg.get('layer3_purge_bars', 0))
    except Exception:
        purge_bars = 0
    try:
        embargo_bars = int(cfg.get('layer3_embargo_bars', 0))
    except Exception:
        embargo_bars = 0

    if purge_bars <= 0:
        try:
            # Match triple-barrier max lookahead default (conservative).
            purge_bars = int(cfg.get('layer3_max_lookahead_bars', 100))
        except Exception:
            purge_bars = 100
    if embargo_bars <= 0:
        embargo_bars = int(max(1, int(purge_bars // 2)))

    if bar_td is not None and isinstance(bar_td, pd.Timedelta) and pd.notna(bar_td):
        purge = bar_td * int(max(0, purge_bars))
        embargo = bar_td * int(max(0, embargo_bars))
    else:
        purge = int(max(0, purge_bars))
        embargo = int(max(0, embargo_bars))

    cv = PurgedKFoldTime(n_splits=n_splits, purge=purge, embargo=embargo)
    fold_indices = list(cv.split(df))

    # Helper function to evaluate a scheme on specific folds
    def evaluate_scheme(name, w_vec, fold_list, calibration_method: str = 'isotonic'):
        oof_probs = np.full(len(df), np.nan)
        fold_metrics = []
        try:
            # Determine if target is continuous (soft labels)
            unique_y = np.unique(y_values)
            is_continuous = len(unique_y) > 2 or (len(unique_y) > 0 and not np.array_equal(unique_y, [0.0, 1.0]) and not np.array_equal(unique_y, [0, 1]))
            
            for fold_idx in fold_list:
                train_idx, test_idx = fold_indices[fold_idx]
                X_train, X_test = X_values[train_idx], X_values[test_idx]
                y_train = y_values[train_idx]
                w_train = w_vec[train_idx]

                if is_continuous:
                    # Use Regressor for soft labels (minimizing MSE/Brier score or similar)
                    # objective='regression' (l2) or 'binary' (logloss)? 
                    # If soft labels are probs, regression (MSE) is robust.
                    reg = lgb.LGBMRegressor(**lgbm_params)
                    reg.fit(X_train, y_train, sample_weight=w_train)
                    probs = reg.predict(X_test)
                    probs = np.clip(probs, 0.0, 1.0) # Ensure valid prob range
                else:
                    # Discrete labels: Use Classifier + Calibration
                    # Dynamic Calibration Selection: Sigmoid vs Isotonic
                    # Split training data to evaluate which method calibrates better
                    
                    calib_method_to_use = 'isotonic' # Default
                    
                    try:
                         # Use last 20% of training data for calibration validation
                         n_tr = len(X_train)
                         if n_tr > 200:
                             split_idx = int(n_tr * 0.8)
                             X_cal_tr, X_cal_val = X_train[:split_idx], X_train[split_idx:]
                             y_cal_tr, y_cal_val = y_train[:split_idx], y_train[split_idx:]
                             w_cal_tr = w_train[:split_idx] if w_train is not None else None
                             
                             base_cal = lgb.LGBMClassifier(**lgbm_params)
                             base_cal.fit(X_cal_tr, y_cal_tr, sample_weight=w_cal_tr)
                             
                             # Test Isotonic
                             iso = CalibratedClassifierCV(base_cal, method='isotonic', cv='prefit')
                             iso.fit(X_cal_val, y_cal_val) # Actually CalibratedClassifierCV with prefit expects validation data in fit
                             p_iso = iso.predict_proba(X_cal_val)[:, 1]
                             ece_iso = _fast_expected_calibration_error(y_cal_val, p_iso)
                             
                             # Test Sigmoid
                             sig = CalibratedClassifierCV(base_cal, method='sigmoid', cv='prefit')
                             sig.fit(X_cal_val, y_cal_val)
                             p_sig = sig.predict_proba(X_cal_val)[:, 1]
                             ece_sig = _fast_expected_calibration_error(y_cal_val, p_sig)
                             
                             if ece_sig < ece_iso:
                                 calib_method_to_use = 'sigmoid'
                                 
                             # print(f"   [Calib] Fold {fold_idx}: Iso ECE={ece_iso:.4f}, Sig ECE={ece_sig:.4f} -> Used {calib_method_to_use}")
                    except Exception as e:
                         # Fallback to config default or isotonic
                         calib_method_to_use = str(calibration_method)

                    # Final Fit with selected method using internal CV (more robust than prefit split)
                    tscv_inner = TimeSeriesSplit(n_splits=3)
                    base_est = lgb.LGBMClassifier(**lgbm_params)
                    calib_clf = CalibratedClassifierCV(
                        estimator=base_est,
                        method=calib_method_to_use,
                        cv=tscv_inner
                    )
                    calib_clf.fit(X_train, y_train, sample_weight=w_train)
                    probs = calib_clf.predict_proba(X_test)[:, 1]
                
                oof_probs[test_idx] = probs

                try:
                    y_fold_true = y_values[test_idx]
                    y_fold_prob = np.asarray(probs, dtype=float)
                    mask_f = np.isfinite(y_fold_true) & np.isfinite(y_fold_prob)
                    if bool(np.any(mask_f)):
                        y_fold_true = y_fold_true[mask_f]
                        y_fold_prob = y_fold_prob[mask_f]
                        if is_continuous:
                            y_fold_bin = (y_fold_true > 0.5).astype(int)
                        else:
                            y_fold_bin = y_fold_true
                        try:
                            auc_f = float(roc_auc_score(y_fold_bin, y_fold_prob)) if int(np.unique(y_fold_bin).size) >= 2 else float('nan')
                        except Exception:
                            auc_f = float('nan')
                        try:
                            ll_f = float(log_loss(y_fold_bin, y_fold_prob))
                        except Exception:
                            ll_f = float('nan')
                        try:
                            ece_f = float(_fast_expected_calibration_error(y_fold_bin, y_fold_prob, n_bins=10))
                        except Exception:
                            ece_f = float('nan')
                        fold_metrics.append({"fold": int(fold_idx), "auc": auc_f, "logloss": ll_f, "ece": ece_f})
                except Exception:
                    pass

            mask = ~np.isnan(oof_probs)
            y_true_eval = y_values[mask]
            y_prob_eval = oof_probs[mask]

            if len(y_true_eval) == 0:
                raise ValueError("No valid predictions generated.")
            
            # For soft labels, we might want to bin y_true for AUC? 
            # Or just calculate AUC treating y_true as continuous (ranking).
            # roc_auc_score supports continuous y_true? Yes, it treats them as probabilistic reference?
            # Actually, standard AUC needs binary y_true.
            # If y_true is continuous, we might need to threshold it for binary metrics or rely on LogLoss/IC.
            
            if is_continuous:
                 # Threshold for binary metrics
                 y_true_binary = (y_true_eval > 0.5).astype(int)
            else:
                 y_true_binary = y_true_eval

            try:
                auc = roc_auc_score(y_true_binary, y_prob_eval)
            except ValueError:
                auc = 0.5 # Handle single class edge case
                
            ll = log_loss(y_true_binary, y_prob_eval) # Log loss against binary truth or soft? sklearn log_loss supports soft y_true? Yes.
            # But let's use binary target for standard metrics for now to avoid confusion
            
            ece = _fast_expected_calibration_error(y_true_binary, y_prob_eval, n_bins=10)
            score = 100 * (auc - 0.5) + 50 * (0.693 - ll) - 200 * ece

            # Interpretability Rating (raised thresholds for meaningful classification)
            if score < 0: rating = "Toxic"
            elif score < 0.2: rating = "Weak"
            elif score < 0.4: rating = "Good"
            else: rating = "Excellent"

            def _fold_stats(key: str):
                vals = [m.get(key) for m in fold_metrics if isinstance(m, dict)]
                vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
                if len(vals) == 0:
                    return {
                        "mean": float('nan'),
                        "std": float('nan'),
                        "min": float('nan'),
                        "max": float('nan'),
                        "n": 0,
                    }
                arr = np.asarray(vals, dtype=float)
                return {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "n": int(arr.size),
                }

            auc_stats = _fold_stats("auc")
            ll_stats = _fold_stats("logloss")
            ece_stats = _fold_stats("ece")

            return {
                "Scheme": name,
                "Score": score,
                "AUC": auc,
                "LogLoss": ll,
                "ECE": ece,
                "FoldAUC_mean": auc_stats["mean"],
                "FoldAUC_std": auc_stats["std"],
                "FoldAUC_min": auc_stats["min"],
                "FoldAUC_max": auc_stats["max"],
                "FoldAUC_n": auc_stats["n"],
                "FoldLogLoss_mean": ll_stats["mean"],
                "FoldLogLoss_std": ll_stats["std"],
                "FoldLogLoss_min": ll_stats["min"],
                "FoldLogLoss_max": ll_stats["max"],
                "FoldLogLoss_n": ll_stats["n"],
                "FoldECE_mean": ece_stats["mean"],
                "FoldECE_std": ece_stats["std"],
                "FoldECE_min": ece_stats["min"],
                "FoldECE_max": ece_stats["max"],
                "FoldECE_n": ece_stats["n"],
                "Rating": rating,
                "oof_probs": oof_probs,
                "w_vec": w_vec
            }
        except Exception as e:
            print(f"⚠️ Scheme {name} failed: {e}")
            return {
                "Scheme": name,
                "Score": -999,
                "AUC": 0, "LogLoss": 99, "ECE": 99, "Rating": "Failed",
                "FoldAUC_mean": float('nan'),
                "FoldAUC_std": float('nan'),
                "FoldAUC_min": float('nan'),
                "FoldAUC_max": float('nan'),
                "FoldAUC_n": 0,
                "FoldLogLoss_mean": float('nan'),
                "FoldLogLoss_std": float('nan'),
                "FoldLogLoss_min": float('nan'),
                "FoldLogLoss_max": float('nan'),
                "FoldLogLoss_n": 0,
                "FoldECE_mean": float('nan'),
                "FoldECE_std": float('nan'),
                "FoldECE_min": float('nan'),
                "FoldECE_max": float('nan'),
                "FoldECE_n": 0,
                "oof_probs": None, "w_vec": w_vec
            }

    # Phase 1: Quick screening on fold 0 only
    phase1_results = []
    for name, w_vec in schemes.items():
        print(f"   Screening {name}...")
        result = evaluate_scheme(name, w_vec, [0], calibration_method='sigmoid')  # Only fold 0
        phase1_results.append(result)

    # Sort by score and take top 3 for full evaluation
    phase1_results.sort(key=lambda x: x["Score"], reverse=True)
    top_schemes = phase1_results[:3]
    top_scheme_names = [r["Scheme"] for r in top_schemes]

    print(f"\n>> Phase 2: Full Evaluation (Top 3: {top_scheme_names})...")

    # Phase 2: Full 5-fold evaluation for top 3 schemes
    for name in top_scheme_names:
        print(f"   Full evaluation: {name}...")
        result = evaluate_scheme(name, schemes[name], list(range(len(fold_indices))), calibration_method='isotonic')  # All folds
        results.append(result)

        if result["Score"] > best_score:
            best_score = result["Score"]
            best_scheme_name = name
            best_model_artifacts = {
                "oof_probs": result["oof_probs"],
                "w_vec": result["w_vec"]
            }

    # ---------------------------------------------------------
    # 4. Reporting & Selection
    # ---------------------------------------------------------
    results_df = pd.DataFrame(results).sort_values("Score", ascending=False)

    try:
        ts = None
        try:
            if isinstance(cfg, dict):
                ts = cfg.get('run_timestamp')
        except Exception:
            ts = None
        ts = str(ts or datetime.utcnow().strftime('%Y%m%d_%H%M%S'))
    except Exception:
        ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')

    try:
        symbol = str(cfg.get('symbol', '')) if isinstance(cfg, dict) else ''
    except Exception:
        symbol = ''
    try:
        timeframe = str(cfg.get('timeframe', '')) if isinstance(cfg, dict) else ''
    except Exception:
        timeframe = ''

    try:
        outcomes_dir = Path('outcomes')
        outcomes_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        outcomes_dir = Path('outcomes')

    try:
        scheme_csv = outcomes_dir / f"layer3_scheme_comparison_{symbol}_{timeframe}_{ts}.csv"
        export_cols = [
            c
            for c in [
                'Scheme', 'Score', 'AUC', 'LogLoss', 'ECE', 'Rating',
                'FoldAUC_mean', 'FoldAUC_std', 'FoldAUC_min', 'FoldAUC_max', 'FoldAUC_n',
                'FoldLogLoss_mean', 'FoldLogLoss_std', 'FoldLogLoss_min', 'FoldLogLoss_max', 'FoldLogLoss_n',
                'FoldECE_mean', 'FoldECE_std', 'FoldECE_min', 'FoldECE_max', 'FoldECE_n',
            ]
            if c in results_df.columns
        ]
        results_df[export_cols].to_csv(scheme_csv, index=False)
    except Exception:
        pass

    print("\n" + "="*60)
    print("   LAYER 3 WEIGHTING SCHEME COMPARISON")
    print("="*60)
    print(f"{'Scheme':<15} | {'Score':<8} | {'AUC':<6} | {'LogLoss':<8} | {'ECE':<6} | {'Rating'}")
    print("-" * 75)
    for row in results_df.itertuples(index=False):
        print(f"{row.Scheme:<15} | {row.Score:>8.4f} | {row.AUC:>6.4f} | {row.LogLoss:>8.4f} | {row.ECE:>6.4f} | {row.Rating}")
    print("-" * 75)

    print(f"\n🏆 WINNER: {best_scheme_name} (Score: {best_score:.4f})")

    if best_model_artifacts is None:
        print("❌ Critical Failure: No schemes succeeded.")
        # Fallback to simple unweighted
        return df, None

    # ---------------------------------------------------------
    # 5. Final Model Training (Production) using WINNER
    # ---------------------------------------------------------
    print(f">> Training Final Production Model using {best_scheme_name}...")

    df['meta_prob'] = best_model_artifacts['oof_probs']
    w_best = best_model_artifacts['w_vec']

    honest_auc = float('nan')
    honest_logloss = float('nan')
    honest_ece = float('nan')
    honest_brier = float('nan')
    honest_n_train = 0
    honest_n_test = 0
    honest_holdout_start = None

    try:
        holdout_n = cfg.get('layer3_honest_holdout_n') if isinstance(cfg, dict) else None
        holdout_n = int(holdout_n) if holdout_n is not None else 0
    except Exception:
        holdout_n = 0

    try:
        holdout_frac = cfg.get('layer3_honest_holdout_frac', 0.15) if isinstance(cfg, dict) else 0.15
        holdout_frac = float(holdout_frac)
    except Exception:
        holdout_frac = 0.15
    if (not np.isfinite(holdout_frac)) or holdout_frac <= 0.0 or holdout_frac >= 0.5:
        holdout_frac = 0.15

    try:
        n_total = int(len(df))
        if n_total >= 200:
            if holdout_n > 0:
                holdout_n = int(min(max(50, holdout_n), max(50, n_total // 2)))
                holdout_start = int(max(0, n_total - holdout_n))
            else:
                holdout_start = int(max(0, int(np.floor(n_total * (1.0 - holdout_frac)))))
                holdout_start = int(min(max(50, holdout_start), max(50, n_total - 50)))

            honest_holdout_start = holdout_start
            honest_n_test = int(n_total - holdout_start)

            # Respect purge around holdout boundary (avoid adjacent label overlap)
            try:
                purge_bars_int = int(purge_bars)
            except Exception:
                purge_bars_int = 0
            purge_bars_int = int(max(0, purge_bars_int))

            train_end = int(max(0, holdout_start - purge_bars_int))
            honest_n_train = int(train_end)

            if honest_n_train >= 50 and honest_n_test >= 50:
                X_arr = X.to_numpy(copy=False)
                y_arr = pd.to_numeric(df[target_col], errors='coerce').astype(float).to_numpy(copy=False)
                w_arr = np.asarray(w_best, dtype=float).reshape(-1)

                X_train = X_arr[:train_end]
                y_train = y_arr[:train_end]
                w_train = w_arr[:train_end]
                X_test = X_arr[holdout_start:]
                y_test = y_arr[holdout_start:]

                mask_tr = np.isfinite(y_train) & np.all(np.isfinite(X_train), axis=1) & np.isfinite(w_train)
                mask_te = np.isfinite(y_test) & np.all(np.isfinite(X_test), axis=1)

                X_train = X_train[mask_tr]
                y_train = y_train[mask_tr]
                w_train = w_train[mask_tr]
                X_test = X_test[mask_te]
                y_test = y_test[mask_te]

                if len(y_train) >= 50 and len(y_test) >= 50:
                    base_est = lgb.LGBMClassifier(**lgbm_params)
                    tscv_inner = TimeSeriesSplit(n_splits=3)
                    calib_clf = CalibratedClassifierCV(
                        estimator=base_est,
                        method='isotonic',
                        cv=tscv_inner,
                    )
                    calib_clf.fit(X_train, y_train.astype(int), sample_weight=w_train)
                    p_test = calib_clf.predict_proba(X_test)[:, 1]

                    y_bin = y_test.astype(int)
                    if int(np.unique(y_bin).size) >= 2:
                        honest_auc = float(roc_auc_score(y_bin, p_test))
                    else:
                        honest_auc = float('nan')
                    honest_logloss = float(log_loss(y_bin, p_test))
                    honest_ece = float(_fast_expected_calibration_error(y_bin, p_test, n_bins=10))
                    try:
                        honest_brier = float(brier_score_loss(y_bin, p_test))
                    except Exception:
                        honest_brier = float('nan')
    except Exception:
        pass

    # Detect continuous again for final training (should match above)
    unique_y = np.unique(y_values)
    is_continuous = len(unique_y) > 2 or (len(unique_y) > 0 and not np.array_equal(unique_y, [0.0, 1.0]) and not np.array_equal(unique_y, [0, 1]))

    try:
        if is_continuous:
             final_model = lgb.LGBMRegressor(**lgbm_params)
             final_model.fit(X, y, sample_weight=w_best)
        else:
            final_base = lgb.LGBMClassifier(**lgbm_params)
            final_tscv = TimeSeriesSplit(n_splits=3)
            final_model = CalibratedClassifierCV(
                estimator=final_base,
                method='isotonic',
                cv=final_tscv
            )
            final_model.fit(X, y, sample_weight=w_best)
            
    except Exception as e:
        print(f"⚠️ Final model training failed: {e}")
        final_model = None

    # ---------------------------------------------------------
    # 6. Final Diagnostics (on Best OOF)
    # ---------------------------------------------------------
    # Just reusing the print layout from before for consistency
    mask = ~np.isnan(df['meta_prob'])
    y_true = y[mask]
    y_prob = df.loc[mask, 'meta_prob']

    score_logloss = float('nan')
    score_auc = float('nan')
    score_ic = float('nan')
    score_mce = float('nan')
    score_brier = float('nan')
    score_ece = float('nan')

    if len(y_true) > 0:
        # Handle continuous targets for metrics
        if is_continuous:
            y_true_metrics = (y_true > 0.5).astype(int)
        else:
            y_true_metrics = y_true

        score_logloss = log_loss(y_true_metrics, y_prob)
        try: score_auc = roc_auc_score(y_true_metrics, y_prob)
        except: score_auc = 0.5
        score_ic, _ = spearmanr(y_prob, y_true) # Spearman works fine with continuous
        if np.isnan(score_ic): score_ic = 0.0

        prob_true, prob_pred = calibration_curve(y_true_metrics, y_prob, n_bins=10)
        score_mce = np.max(np.abs(prob_true - prob_pred)) if len(prob_true) > 0 else 0.0

        score_ece = _fast_expected_calibration_error(
            np.asarray(y_true_metrics, dtype=float),
            np.asarray(y_prob, dtype=float),
            n_bins=10,
        )

        if is_continuous:
             try:
                 y_true_arr = np.asarray(y_true, dtype=float)
                 y_prob_arr = np.asarray(y_prob, dtype=float)
                 m = np.isfinite(y_true_arr) & np.isfinite(y_prob_arr)
                 score_brier = float(np.mean((y_true_arr[m] - y_prob_arr[m]) ** 2)) if bool(np.any(m)) else float('nan')
             except Exception:
                 score_brier = float('nan')
        else:
             try:
                 score_brier = brier_score_loss(y_true, y_prob)
             except ValueError:
                 try:
                     y_true_arr = np.asarray(y_true, dtype=float)
                     y_prob_arr = np.asarray(y_prob, dtype=float)
                     m = np.isfinite(y_true_arr) & np.isfinite(y_prob_arr)
                     score_brier = float(np.mean((y_true_arr[m] - y_prob_arr[m]) ** 2)) if bool(np.any(m)) else float('nan')
                 except Exception:
                     score_brier = float('nan')

    metrics = {
        "Log Loss": f"{score_logloss:.5f}",
        "AUC":      f"{score_auc:.5f}",
        "IC":       f"{score_ic:.5f}",
        "ECE":      f"{score_ece:.5f}",
        "MCE":      f"{score_mce:.5f}",
        "Brier":    f"{score_brier:.5f}"
    }

    try:
        md_path = outcomes_dir / f"layer3_report_{symbol}_{timeframe}_{ts}.md"
        lines = [
            "# Layer3 Report\n",
            f"- timestamp: {ts}\n",
            f"- symbol: {symbol}\n",
            f"- timeframe: {timeframe}\n",
            f"- n_rows_input: {int(len(oof_df))}\n",
            f"- n_rows_after_target_dropna: {int(len(df))}\n",
            f"- n_base_models: {int(len(base_model_cols or []))}\n",
            f"- winner_scheme: {best_scheme_name}\n",
            f"- winner_score: {float(best_score) if best_score is not None else float('nan')}\n",
            "\n## Winner Metrics (OOF)\n",
        ]
        for k in ['AUC', 'Log Loss', 'ECE', 'IC', 'MCE', 'Brier']:
            if k in metrics:
                lines.append(f"- {k}: {metrics[k]}\n")
        lines.append("\n## Honest Holdout Metrics (Forward Tail)\n")
        lines.append(f"- n_train: {int(honest_n_train)}\n")
        lines.append(f"- n_holdout: {int(honest_n_test)}\n")
        lines.append(f"- honest_auc: {float(honest_auc) if np.isfinite(honest_auc) else float('nan')}\n")
        lines.append(f"- honest_logloss: {float(honest_logloss) if np.isfinite(honest_logloss) else float('nan')}\n")
        lines.append(f"- honest_ece: {float(honest_ece) if np.isfinite(honest_ece) else float('nan')}\n")
        lines.append(f"- honest_brier: {float(honest_brier) if np.isfinite(honest_brier) else float('nan')}\n")

        # Add Weighting Scheme Comparison Table
        lines.append("\n## Weighting Scheme Comparison\n")

        # Markdown table header
        table_cols = ['Scheme', 'Score', 'AUC', 'LogLoss', 'ECE', 'Rating']
        header = "| " + " | ".join(table_cols) + " |"
        separator = "| " + " | ".join(["---"] * len(table_cols)) + " |"
        lines.append(header + "\n")
        lines.append(separator + "\n")

        for _, row in results_df.iterrows():
            row_str = "|"
            for col in table_cols:
                val = row[col]
                if isinstance(val, float):
                    row_str += f" {val:.4f} |"
                else:
                    row_str += f" {val} |"
            lines.append(row_str + "\n")

        md_path.write_text(''.join(lines))
    except Exception:
        pass

    try:
        summary_row = {
            'timestamp': ts,
            'symbol': symbol,
            'timeframe': timeframe,
            'n_rows_input': int(len(oof_df)),
            'n_rows_after_target_dropna': int(len(df)),
            'n_base_models': int(len(base_model_cols or [])),
            'winner_scheme': str(best_scheme_name),
            'winner_score': float(best_score) if best_score is not None else float('nan'),
            'auc': float(score_auc),
            'logloss': float(score_logloss),
            'ece': float(score_ece),
            'ic': float(score_ic),
            'mce': float(score_mce),
            'brier': float(score_brier),
            'honest_auc': float(honest_auc) if np.isfinite(honest_auc) else float('nan'),
            'honest_logloss': float(honest_logloss) if np.isfinite(honest_logloss) else float('nan'),
            'honest_ece': float(honest_ece) if np.isfinite(honest_ece) else float('nan'),
            'honest_brier': float(honest_brier) if np.isfinite(honest_brier) else float('nan'),
            'honest_n_train': int(honest_n_train),
            'honest_n_holdout': int(honest_n_test),
        }
        pd.DataFrame([summary_row]).to_csv(
            outcomes_dir / f"layer3_summary_{symbol}_{timeframe}_{ts}.csv",
            index=False,
        )
    except Exception:
        pass

    print("\n   WINNER PERFORMANCE (OOF)")
    for k, v in metrics.items():
        print(f"   {k:<10} : {v}")
    print("")

    try:
        if honest_holdout_start is not None and int(honest_n_train) > 0 and int(honest_n_test) > 0:
            print("   HONEST HOLDOUT (Forward Tail)")
            print(f"   n_train   : {int(honest_n_train)}")
            print(f"   n_holdout : {int(honest_n_test)}")
            print(f"   AUC       : {float(honest_auc):.5f}" if np.isfinite(honest_auc) else "   AUC       : nan")
            print(f"   Log Loss  : {float(honest_logloss):.5f}" if np.isfinite(honest_logloss) else "   Log Loss  : nan")
            print(f"   ECE       : {float(honest_ece):.5f}" if np.isfinite(honest_ece) else "   ECE       : nan")
            print(f"   Brier     : {float(honest_brier):.5f}" if np.isfinite(honest_brier) else "   Brier     : nan")
            print("")
    except Exception:
        pass

    if enable_timing and t0_all is not None:
        dt = time.perf_counter() - t0_all
        print(f"Layer3 timing: total_seconds={dt:.3f}")

    # ---------------------------------------------------------
    # 7. SHAP Analysis
    # ---------------------------------------------------------
    _run_shap_analysis(final_model, X, outcomes_dir, symbol, timeframe, ts, md_path)

    # Return full dataframe with predictions + final model
    return df, final_model

def _run_shap_analysis(model, X, output_dir, symbol, timeframe, ts, md_path):
    """
    Computes SHAP values for the final model and saves a summary plot.
    Appends results to the markdown report.
    """
    print("\n>> Running SHAP Analysis on Production Model...")
    try:
        if model is None:
            return

        # Sample data for SHAP (max 1000 rows)
        n_sample = min(1000, len(X))
        if n_sample <= 0:
            return

        # Use random sampling for representativeness (or could use tail)
        # Using tail is better for "current regime" explanation, random for global.
        # Let's use random with fixed seed.
        X_sample = X.sample(n=n_sample, random_state=42)

        shap_values_list = []
        estimators = []

        # Extract estimators
        if isinstance(model, CalibratedClassifierCV):
            if hasattr(model, 'calibrated_classifiers_'):
                for cc in model.calibrated_classifiers_:
                    est = getattr(cc, 'estimator', None) or getattr(cc, 'base_estimator', None)
                    if est:
                        estimators.append(est)
        else:
            # Assume it's a direct LGBMRegressor or Classifier
            estimators.append(model)

        if not estimators:
            print("⚠️ SHAP: Could not extract base estimators from model.")
            return

        print(f"   Aggregating SHAP values from {len(estimators)} estimators...")

        # Calculate SHAP values
        for est in estimators:
            try:
                explainer = shap.TreeExplainer(est)
                vals = explainer.shap_values(X_sample)

                # Handle binary classification output (list of arrays)
                if isinstance(vals, list):
                    # Usually index 1 is positive class
                    if len(vals) == 2:
                        vals = vals[1]
                    else:
                        vals = vals[0] # Fallback

                shap_values_list.append(vals)
            except Exception as e:
                print(f"   ⚠️ Estimator SHAP failed: {e}")

        if not shap_values_list:
            return

        # Average SHAP values
        avg_shap_values = np.mean(shap_values_list, axis=0)

        # 1. Summary Plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(avg_shap_values, X_sample, show=False, plot_size=(10, 8))

        plot_filename = f"layer3_shap_{symbol}_{timeframe}_{ts}.png"
        plot_path = output_dir / plot_filename
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"   SHAP plot saved to: {plot_path}")

        # 2. Text Summary (Top features)
        # Calculate mean absolute SHAP value per feature
        feature_importance = pd.DataFrame(
            list(zip(X_sample.columns, np.abs(avg_shap_values).mean(0))),
            columns=['feature', 'importance']
        )
        feature_importance.sort_values(by='importance', ascending=False, inplace=True)
        top_20 = feature_importance.head(20)

        print("\n   TOP 20 FEATURES BY SHAP IMPORTANCE:")
        print(top_20.to_string(index=False))

        # 3. Append to Markdown Report
        if md_path and md_path.exists():
            with open(md_path, 'a') as f:
                f.write("\n\n## SHAP Feature Importance (Global)\n")
                f.write(f"![SHAP Summary]({plot_filename})\n\n")

                f.write("### Top 20 Features\n")
                f.write("| Feature | Mean |SHAP| |\n")
                f.write("| --- | --- |\n")
                for _, row in top_20.iterrows():
                    f.write(f"| {row['feature']} | {row['importance']:.6f} |\n")
                f.write("\n")

    except Exception as e:
        print(f"⚠️ SHAP Analysis failed: {e}")
        import traceback
        traceback.print_exc()

# ---------------------------------------------------------
# Helper: Advanced Diagnostic Plot (Unchanged)
# ---------------------------------------------------------
def plot_diagnostics(y_true, y_prob, output_path=None):
    """
    Plots Reliability Diagram (Calibration) AND Probability Density (Resolution).
    """
    try:
        # Remove NaNs
        mask = ~np.isnan(y_prob) & ~np.isnan(y_true)
        y_true = y_true[mask]
        y_prob = y_prob[mask]

        if len(y_true) == 0:
            return

        fig, ax = plt.subplots(1, 2, figsize=(14, 6))

        # 1. Reliability Diagram
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        ax[0].plot(prob_pred, prob_true, marker='o', linewidth=2, label='Meta-Model')
        ax[0].plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.5, label='Perfect')
        ax[0].set_xlabel('Predicted Probability')
        ax[0].set_ylabel('Actual Win Rate')
        ax[0].set_title('Calibration (Reliability)')
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)

        # 2. Probability Density (Histogram)
        sns.histplot(y_prob, bins=20, kde=True, ax=ax[1], color='purple', alpha=0.6)
        ax[1].set_xlim(0, 1)
        ax[1].set_xlabel('Predicted Probability')
        ax[1].set_title('Resolution (Confidence Distribution)')
        ax[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path)
            print(f"Diagnostics plot saved to {output_path}")
        else:
            pass
        plt.close(fig)
    except Exception as e:
        print(f"Failed to generate plots: {e}")
