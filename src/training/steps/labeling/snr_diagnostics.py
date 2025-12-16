#!/usr/bin/env python3
"""SNR & Label Diagnostics for Meta-Labeling Outputs.

Usage examples (from project root):

  python scripts/snr_diagnostics.py label-quality \
      --symbol ETHUSDT --exchange binance --timeframe 15m

  python scripts/snr_diagnostics.py label-learnability \
      --symbol ETHUSDT --exchange binance --timeframe 15m

  python scripts/snr_diagnostics.py model-robustness \
      --symbol ETHUSDT --exchange binance --timeframe 15m

  python scripts/snr_diagnostics.py trading-simulation \
      --symbol ETHUSDT --exchange binance --timeframe 15m \
      --prob-thresholds 0.55 0.60 0.65

  python scripts/snr_diagnostics.py full \
      --symbol ETHUSDT --exchange binance --timeframe 15m

Subcommands:
- label-quality:      Label distribution, coverage, economic SNR, retention.
- label-learnability: Learnability (AUC-based) and entropy/balance of labels.
- model-robustness:   Probe model CV stability (AUC mean/std across folds).
- trading-simulation: Model calibration, trades/day, PnL/day, equity curves,
                      consecutive losses, win-rate stability at different
                      probability thresholds.
- full:               Run all diagnostics and aggregate results.

This script is designed to be run *after* the
`feature_generation_meta_labeling_step` has been executed via the launcher,
so that the `labeled_data_{symbol}_{timeframe}` artifact exists.
"""

import argparse
import logging
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import system_logger  # type: ignore
from src.training.steps.labeling.feature_generation_meta_labeling_step import (  # type: ignore
    FeatureGenerationMetaLabelingStep,
    compute_learnability_score,
    compute_label_entropy_score,
    combined_label_quality_objective,
    DEFAULT_TRANSACTION_COST,
    compute_label_quality_score_from_components,
)
from src.training.steps.labeling.labeled_data_schema import (
    LABELED_DATA_SCHEMA_VERSION,
    get_required_labeled_data_columns,
    validate_labeled_data_schema,
)

# Import calibration quality utilities for enhanced diagnostics
try:
    from src.training.steps.labeling.probability_calibration import (
        calibration_quality_report,
        validate_monotonicity,
        select_brier_optimal_threshold,
    )
    CALIBRATION_QUALITY_AVAILABLE = True
except ImportError:
    CALIBRATION_QUALITY_AVAILABLE = False

# Import signal spacing utilities for density diagnostics
try:
    from src.training.steps.labeling.signal_spacing_utils import (
        compute_signal_spacing_stats,
        recommend_signal_spacing,
    )
    SIGNAL_SPACING_AVAILABLE = True
except ImportError:
    SIGNAL_SPACING_AVAILABLE = False


try:
    import lightgbm as lgb  # type: ignore
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import calibration_curve
    import matplotlib.pyplot as plt  # type: ignore
    import matplotlib  # type: ignore
    matplotlib.use('Agg')  # Use non-interactive backend
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "snr_diagnostics requires lightgbm and scikit-learn to be installed. "
        "Install them in your environment before running this script."
    ) from exc


logger = system_logger.getChild("snr_diagnostics")


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


OUTCOMES_DIR = Path("outcomes")


_LAST_EXPORTS: dict[str, dict] = {}


def _ensure_outcomes_dir() -> Path:
    """Ensure outcomes directory exists and return it."""
    OUTCOMES_DIR.mkdir(exist_ok=True)
    return OUTCOMES_DIR


def _export_report(
    prefix: str,
    symbol: str,
    exchange: str,
    timeframe: str,
    direction: str,
    model: str,
    payload: dict,
    markdown_lines: list[str],
) -> tuple[Path, Path]:
    """Export diagnostics payload as JSON and Markdown into outcomes/.

    Filenames are of the form:
        outcomes/{prefix}_{symbol}_{timeframe}_{YYYYMMDD_HHMMSS}.json/md
    """
    out_dir = _ensure_outcomes_dir()
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base_name = f"{prefix}_{symbol}_{timeframe}_{ts}"

    # Enrich payload with common metadata
    meta = {
        "symbol": symbol,
        "exchange": exchange,
        "timeframe": timeframe,
        "direction": direction,
        "model": model,
        "prefix": prefix,
        "timestamp_utc": ts,
    }
    full_payload = {"metadata": meta, **payload}

    json_path = out_dir / f"{base_name}.json"
    md_path = out_dir / f"{base_name}.md"

    with json_path.open("w") as f_json:
        json.dump(full_payload, f_json, indent=2, default=str)

    with md_path.open("w") as f_md:
        f_md.write("\n".join(markdown_lines))

    _LAST_EXPORTS[prefix] = {
        "json_path": json_path,
        "md_path": md_path,
        "payload": full_payload,
        "markdown_lines": markdown_lines,
    }

    logger.info("Saved %s diagnostics to %s and %s", prefix, json_path, md_path)
    return json_path, md_path


def _load_labeled_data(
    symbol: str,
    exchange: str,
    timeframe: str,
    direction: str = "long",
    model: str = "analyst",
) -> pd.DataFrame:
    """Load labeled_data artifact produced by FeatureGenerationMetaLabelingStep.

    Tries both versioned HDF5 and legacy artifacts via the same BaseStep
    `_get_artifact` mechanism, so it remains compatible with older runs.
    """
    step = FeatureGenerationMetaLabelingStep()
    step.set_context(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        direction=direction,
        model=model,
    )

    # Primary artifact name used by the meta-labeling step
    primary_name = f"labeled_data_{symbol}_{timeframe}"

    # Legacy/candidate names for compatibility
    candidate_names = [
        primary_name,
        f"labeled_data_{symbol}_{exchange}_{timeframe}",
        f"labeled_data_{symbol}_{timeframe}_{direction}",
    ]

    for name in candidate_names:
        try:
            df = step._get_artifact(  # type: ignore[attr-defined]
                artifact_name=name,
                artifact_type="data",
                data_category="features",
            )
        except Exception:
            df = None

        if isinstance(df, pd.DataFrame) and not df.empty:
            # Validate basic labeled_data schema for downstream diagnostics
            try:
                validate_labeled_data_schema(
                    df,
                    required_cols=get_required_labeled_data_columns(
                        [
                            "meta_probability",
                            "event_duration_bars",
                        ]
                    ),
                    context="snr_diagnostics._load_labeled_data",
                )
            except Exception:
                # Fallback to minimal schema for backward compatibility
                validate_labeled_data_schema(
                    df,
                    context="snr_diagnostics._load_labeled_data",
                )
            # Helpful debug info: these columns drive the bucket diagnostics.
            try:
                cols = set(map(str, df.columns))
                logger.info(
                    "labeled_data columns present: meta_probability=%s, volatility_1d=%s, targets=%s",
                    "meta_probability" in cols,
                    "volatility_1d" in cols,
                    any(c in cols for c in ("target_long", "target_short")),
                )
            except Exception:
                pass
            logger.info(
                "Loaded labeled data from artifact '%s' with shape %s",
                name,
                df.shape,
            )
            return df

    raise FileNotFoundError(
        f"Could not locate labeled_data artifact for {symbol} {exchange} {timeframe}. "
        f"Tried names: {candidate_names}. Run feature_generation_meta_labeling_step first."
    )


def _build_feature_matrix_from_labeled(labeled_df: pd.DataFrame, direction: str = "long") -> Tuple[pd.DataFrame, pd.Series]:
    """Construct (X, y) for learnability / robustness diagnostics from labeled_data.

    - y: uses directional binary_label (binary_label_long/short) if available, otherwise falls back to binary_label.
    - X: all numeric columns except obvious target/label/return-related ones.
    """
    # Prefer directional binary labels
    if direction == "long" and "binary_label_long" in labeled_df.columns:
        y = labeled_df["binary_label_long"].copy()
    elif direction == "short" and "binary_label_short" in labeled_df.columns:
        y = labeled_df["binary_label_short"].copy()
    elif "binary_label_long" in labeled_df.columns:
        y = labeled_df["binary_label_long"].copy()
    elif "binary_label_short" in labeled_df.columns:
        y = labeled_df["binary_label_short"].copy()
    elif "binary_label" in labeled_df.columns:
        y = labeled_df["binary_label"].copy()
    else:
        raise ValueError(
            "labeled_data is missing required binary label column. "
            "Expected: binary_label_long, binary_label_short, or binary_label"
        )

    # Numeric feature candidates
    numeric = labeled_df.select_dtypes(include=[np.number]).copy()

    # Drop columns that are clearly targets/labels/returns or sample weights,
    # plus obvious post-event fields (exit_* and close_time) that can encode
    # realized outcome information.
    drop_patterns = [
        "target",
        "label",
        "return",
        "meta_probability",
        "r_multiple",
        "sample_weight",
        "event_duration",
        "adaptive_profit_threshold",
        "adaptive_stop_threshold",
        "exit_",
        "close_time",
    ]
    drop_cols = []
    for col in numeric.columns:
        lower = col.lower()
        if any(pat in lower for pat in drop_patterns):
            drop_cols.append(col)

    X = numeric.drop(columns=drop_cols, errors="ignore")

    # Align X and y on common index and drop NaNs in y
    valid_mask = ~y.isna()
    y_clean = y[valid_mask]
    X_clean = X.loc[y_clean.index].fillna(0)

    if len(y_clean) < 50:
        logger.warning("Only %d valid samples after cleaning; diagnostics may be noisy", len(y_clean))

    return X_clean, y_clean


# --------------------------------------------------------------------------------------
# New Diagnostic Helper Functions
# --------------------------------------------------------------------------------------

def _compute_regime_auc_breakdown(
    labeled_df: pd.DataFrame,
    y_proba: np.ndarray,
    y_true: np.ndarray,
) -> dict:
    """Compute AUC breakdown by volatility, HMM, and liquidity regimes if available.

    Returns:
        Dict with per-regime AUC values and summary statistics.
    """
    regime_aucs = {}

    # Volatility regime breakdown
    if "volatility_regime" in labeled_df.columns:
        try:
            vol_regimes = labeled_df["volatility_regime"].dropna().unique()
            for regime in vol_regimes:
                regime_mask = (labeled_df["volatility_regime"] == regime).values
                if regime_mask.sum() >= 20 and len(np.unique(y_true[regime_mask])) >= 2:
                    auc = roc_auc_score(y_true[regime_mask], y_proba[regime_mask])
                    regime_aucs[f"vol_{regime}"] = float(auc)
        except Exception:
            pass

    # HMM regime breakdown if available
    if "hmm_regime_label_1h" in labeled_df.columns:
        try:
            hmm_regimes = labeled_df["hmm_regime_label_1h"].dropna().unique()
            for regime in hmm_regimes:
                regime_mask = (labeled_df["hmm_regime_label_1h"] == regime).values
                if regime_mask.sum() >= 20 and len(np.unique(y_true[regime_mask])) >= 2:
                    auc = roc_auc_score(y_true[regime_mask], y_proba[regime_mask])
                    regime_aucs[f"hmm_{regime}"] = float(auc)
        except Exception:
            pass

    # Liquidity regime breakdown if available
    liquidity_regime_cols = [
        c for c in labeled_df.columns 
        if c.startswith('liquidity_liquidity_regime_') and 'prob_' in c
    ]
    
    if liquidity_regime_cols:
        try:
            # For each liquidity regime probability column, compute AUC for high-probability regime periods
            for col in liquidity_regime_cols:
                # Extract regime number from column name (e.g., 'liquidity_liquidity_regime_0_prob_')
                regime_num = col.split('_')[2] if '_' in col else 'unknown'
                
                # Consider periods where probability > 0.6 as "in regime"
                regime_mask = labeled_df[col].fillna(0) > 0.6
                
                if regime_mask.sum() >= 20 and len(np.unique(y_true[regime_mask])) >= 2:
                    auc = roc_auc_score(y_true[regime_mask], y_proba[regime_mask])
                    regime_aucs[f"liquidity_{regime_num}"] = float(auc)
        except Exception:
            pass

    return regime_aucs


def _compute_temporal_auc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    window_size: int = 50,
) -> dict:
    """Compute rolling AUC across time with specified window size.

    Returns:
        Dict with temporal_aucs (list of AUC values) and corresponding indices.
    """
    temporal_aucs = []
    temporal_indices = []

    for i in range(0, len(y_true) - window_size, max(1, window_size // 4)):
        window_end = min(i + window_size, len(y_true))
        window_true = y_true[i:window_end]
        window_proba = y_proba[i:window_end]

        if len(np.unique(window_true)) >= 2:
            try:
                auc = roc_auc_score(window_true, window_proba)
                temporal_aucs.append(float(auc))
                temporal_indices.append(i + window_size // 2)
            except Exception:
                pass

    return {
        "temporal_aucs": temporal_aucs,
        "temporal_indices": temporal_indices,
    }


def _compute_feature_importance_stability(
    labeled_df: pd.DataFrame,
    cv_splits: int = 5,
    n_features_top: int = 20,
) -> dict:
    """Compute feature importance variance across CV folds.

    Returns:
        Dict with feature_importance_std, mean_importance, and concentration metrics.
    """
    X, y = _build_feature_matrix_from_labeled(labeled_df, direction="long")  # Default direction for this internal function
    X_array = X.values.astype(float)
    y_array = y.values.astype(float)

    tscv = TimeSeriesSplit(n_splits=cv_splits)
    fold_importances = []

    for tr_idx, te_idx in tscv.split(X_array):
        X_tr = X_array[tr_idx]
        y_tr = y_array[tr_idx]

        # Clean NaNs
        mask = ~np.isnan(y_tr)
        X_tr_clean = X_tr[mask]
        y_tr_clean = y_tr[mask]

        if len(y_tr_clean) < 50 or len(np.unique(y_tr_clean)) < 2:
            continue

        try:
            clf = lgb.LGBMClassifier(
                boosting_type="gbdt",
                objective="binary",
                max_depth=3,
                n_estimators=50,
                learning_rate=0.1,
                verbose=-1,
                random_state=42,
            )
            clf.fit(X_tr_clean, y_tr_clean)

            # Get feature importance
            importances = clf.feature_importances_
            fold_importances.append(importances)
        except Exception:
            continue

    if not fold_importances:
        return {
            "feature_importance_std": 0.0,
            "importance_concentration": 0.0,
            "top_features": [],
        }

    # Compute stats across folds
    importances_array = np.array(fold_importances)
    mean_importance = importances_array.mean(axis=0)
    std_importance = importances_array.std(axis=0)

    # Concentration: fraction of importance in top N features
    top_k_importance = np.sum(np.sort(mean_importance)[-n_features_top:])
    total_importance = np.sum(mean_importance)
    concentration = float(top_k_importance / (total_importance + 1e-9))

    # Top features
    top_indices = np.argsort(mean_importance)[-n_features_top:][::-1]
    top_features = [
        {
            "feature_idx": int(idx),
            "mean_importance": float(mean_importance[idx]),
            "std_importance": float(std_importance[idx]),
        }
        for idx in top_indices if mean_importance[idx] > 0
    ]

    return {
        "feature_importance_std": float(np.mean(std_importance)),
        "importance_concentration": concentration,
        "top_features": top_features,
    }


def _apply_confident_learning_noise_filter(
    df: pd.DataFrame,
    y_true_col: str = "binary_label",
    y_proba_col: str = "meta_probability",
    threshold_confident: float = 0.9,
    min_samples_required: int = 100,
    verbose: bool = True
) -> dict:
    """Apply confident learning noise filter to remove suspected mislabeled rows.
    
    This function identifies samples where the model is highly confident but the label
    disagrees with the prediction, indicating potential mislabeling. These samples are
    removed from the dataset and SNR metrics are recomputed.
    
    Args:
        df: Input DataFrame with labels and probabilities
        y_true_col: Column name for true labels
        y_proba_col: Column name for predicted probabilities
        threshold_confident: Confidence threshold for identifying confident predictions
        min_samples_required: Minimum samples required for filtering
        verbose: Whether to print filtering statistics
        
    Returns:
        Dict with filtered DataFrame, noise statistics, and SNR recomputation
    """
    # Validate inputs
    if y_true_col not in df.columns or y_proba_col not in df.columns:
        if verbose:
            print(f"Warning: Missing required columns {y_true_col} or {y_proba_col}")
        return {
            "filtered_df": df,
            "noise_stats": {},
            "snr_before": {},
            "snr_after": {},
            "applied_filter": False
        }
    
    # Get labeled samples only
    labeled_mask = df[y_true_col].notna() & df[y_proba_col].notna()
    labeled_df = df[labeled_mask].copy()
    
    if len(labeled_df) < min_samples_required:
        if verbose:
            print(f"Warning: Insufficient labeled samples ({len(labeled_df)} < {min_samples_required})")
        return {
            "filtered_df": df,
            "noise_stats": {},
            "snr_before": {},
            "snr_after": {},
            "applied_filter": False
        }
    
    # Extract labels and probabilities
    y_true = labeled_df[y_true_col].values
    y_proba = labeled_df[y_proba_col].values
    
    # Estimate noise using confident learning
    noise_stats = _estimate_label_noise_confident_learning(
        y_true, y_proba, threshold_confident=threshold_confident
    )
    
    # Get indices of potential mislabeled samples
    mislabeled_indices = noise_stats["mislabeled_indices"]
    
    if len(mislabeled_indices) == 0:
        if verbose:
            print("No mislabeled candidates detected")
        return {
            "filtered_df": df,
            "noise_stats": noise_stats,
            "snr_before": {},
            "snr_after": {},
            "applied_filter": False
        }
    
    # Get actual DataFrame indices to remove
    labeled_indices = labeled_df.index
    indices_to_remove = [labeled_indices[i] for i in mislabeled_indices if i < len(labeled_indices)]
    
    # Compute SNR before filtering
    snr_before = {}
    if "realized_return" in labeled_df.columns:
        returns = labeled_df["realized_return"]
        pos_mask = labeled_df[y_true_col] == 1
        neg_mask = labeled_df[y_true_col] == 0
        
        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
            pos_returns = returns[pos_mask].dropna()
            neg_returns = returns[neg_mask].dropna()
            
            if len(pos_returns) > 1 and len(neg_returns) > 1:
                pos_mean, pos_std = pos_returns.mean(), pos_returns.std()
                neg_mean = neg_returns.mean()
                snr_before = {
                    "pos_mean": float(pos_mean),
                    "pos_std": float(pos_std),
                    "neg_mean": float(neg_mean),
                    "snr": float(pos_mean / (pos_std + 1e-8)),
                    "cohens_d": float((pos_mean - neg_mean) / np.sqrt(((pos_returns.var() + neg_returns.var()) / 2) + 1e-8))
                }
    
    # Apply filter - remove mislabeled samples
    filtered_df = df.drop(indices_to_remove).copy()
    
    # Compute SNR after filtering
    snr_after = {}
    if "realized_return" in filtered_df.columns:
        filtered_labeled_mask = filtered_df[y_true_col].notna() & filtered_df[y_proba_col].notna()
        filtered_labeled = filtered_df[filtered_labeled_mask]
        
        if len(filtered_labeled) > 0:
            returns = filtered_labeled["realized_return"]
            pos_mask = filtered_labeled[y_true_col] == 1
            neg_mask = filtered_labeled[y_true_col] == 0
            
            if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                pos_returns = returns[pos_mask].dropna()
                neg_returns = returns[neg_mask].dropna()
                
                if len(pos_returns) > 1 and len(neg_returns) > 1:
                    pos_mean, pos_std = pos_returns.mean(), pos_returns.std()
                    neg_mean = neg_returns.mean()
                    snr_after = {
                        "pos_mean": float(pos_mean),
                        "pos_std": float(pos_std),
                        "neg_mean": float(neg_mean),
                        "snr": float(pos_mean / (pos_std + 1e-8)),
                        "cohens_d": float((pos_mean - neg_mean) / np.sqrt(((pos_returns.var() + neg_returns.var()) / 2) + 1e-8))
                    }
    
    # Calculate improvement metrics
    snr_improvement = {}
    if snr_before and snr_after:
        snr_improvement = {
            "snr_delta": float(snr_after["snr"] - snr_before["snr"]),
            "snr_pct_change": float((snr_after["snr"] - snr_before["snr"]) / abs(snr_before["snr"]) * 100) if snr_before["snr"] != 0 else 0,
            "cohens_d_delta": float(snr_after["cohens_d"] - snr_before["cohens_d"]),
            "samples_removed": len(indices_to_remove),
            "samples_remaining": len(filtered_labeled_mask),
            "removal_rate": float(len(indices_to_remove) / len(labeled_df) * 100)
        }
    
    if verbose:
        print(f"\n=== Confident Learning Noise Filter Results ===")
        print(f"Original labeled samples: {len(labeled_df)}")
        print(f"Mislabeled candidates: {len(indices_to_remove)} ({noise_stats['estimated_noise_rate']:.1%})")
        print(f"Samples remaining: {len(filtered_labeled_mask)}")
        
        if snr_improvement:
            print(f"SNR change: {snr_before['snr']:.3f} → {snr_after['snr']:.3f} ({snr_improvement['snr_pct_change']:+.1f}%)")
            print(f"Cohen's d change: {snr_before['cohens_d']:.3f} → {snr_after['cohens_d']:.3f} ({snr_improvement['cohens_d_delta']:+.3f})")
    
    return {
        "filtered_df": filtered_df,
        "noise_stats": noise_stats,
        "snr_before": snr_before,
        "snr_after": snr_after,
        "snr_improvement": snr_improvement,
        "applied_filter": len(indices_to_remove) > 0,
        "indices_removed": indices_to_remove
    }


def _apply_confident_learning_noise_filter(
    df: pd.DataFrame,
    y_true_col: str = "binary_label",
    y_proba_col: str = "meta_probability",
    threshold_confident: float = 0.9,
    min_samples_required: int = 100,
    verbose: bool = True
) -> dict:
    """Apply confident learning noise filter to remove suspected mislabeled rows.
    
    This function identifies samples where the model is highly confident but the label
    disagrees with the prediction, indicating potential mislabeling. These samples are
    removed from the dataset and SNR metrics are recomputed.
    
    Args:
        df: Input DataFrame with labels and probabilities
        y_true_col: Column name for true labels
        y_proba_col: Column name for predicted probabilities
        threshold_confident: Confidence threshold for identifying confident predictions
        min_samples_required: Minimum samples required for filtering
        verbose: Whether to print progress
        
    Returns:
        Dict with filtered DataFrame and noise statistics
    """
    # Check if probability column exists
    if y_proba_col not in df.columns:
        if verbose:
            print(f"Probability column '{y_proba_col}' not found - skipping noise filter")
        return {
            "filtered_df": df,
            "noise_stats": {},
            "snr_before": {},
            "snr_after": {},
            "snr_improvement": {},
            "applied_filter": False,
            "indices_removed": []
        }
    
    # Filter labeled samples
    labeled_df = df[df[y_true_col].notna()].copy()
    if len(labeled_df) < min_samples_required:
        if verbose:
            print(f"Insufficient labeled samples ({len(labeled_df)} < {min_samples_required}) - skipping")
        return {
            "filtered_df": df,
            "noise_stats": {},
            "snr_before": {},
            "snr_after": {},
            "snr_improvement": {},
            "applied_filter": False,
            "indices_removed": []
        }
    
    # Extract labels and probabilities
    y_true = labeled_df[y_true_col].values
    y_proba = labeled_df[y_proba_col].values
    
    # Estimate noise using confident learning
    noise_stats = _estimate_label_noise_confident_learning(
        y_true, y_proba, threshold_confident=threshold_confident
    )
    
    # Get indices of potential mislabeled samples
    mislabeled_indices = noise_stats["mislabeled_indices"]
    
    if len(mislabeled_indices) == 0:
        if verbose:
            print("No mislabeled candidates detected")
        return {
            "filtered_df": df,
            "noise_stats": noise_stats,
            "snr_before": {},
            "snr_after": {},
            "snr_improvement": {},
            "applied_filter": False,
            "indices_removed": []
        }
    
    # Get original indices
    original_indices = labeled_df.index[mislabeled_indices]
    
    # Create filtered dataset
    filtered_df = df.drop(original_indices)
    
    # Compute SNR before and after filtering
    snr_before = _compute_snr_metrics(labeled_df[y_true_col], labeled_df.get('realized_return', pd.Series()))
    snr_after = _compute_snr_metrics(
        filtered_df[filtered_df[y_true_col].notna()][y_true_col],
        filtered_df[filtered_df[y_true_col].notna()].get('realized_return', pd.Series())
    )
    
    # Compute improvement metrics
    snr_improvement = {}
    if snr_before and snr_after:
        snr_improvement = {
            "snr_delta": snr_after.get("snr", 0) - snr_before.get("snr", 0),
            "snr_pct_change": ((snr_after.get("snr", 0) - snr_before.get("snr", 0)) / max(abs(snr_before.get("snr", 1e-8)), 1e-8)) * 100,
            "cohens_d_delta": snr_after.get("cohens_d", 0) - snr_before.get("cohens_d", 0)
        }
    
    # Additional statistics
    filtered_labeled_mask = filtered_df[y_true_col].notna()
    indices_to_remove = list(original_indices)
    
    noise_stats.update({
        "original_labeled_samples": len(labeled_df),
        "samples_remaining": len(filtered_labeled_mask),
        "removal_rate": float(len(indices_to_remove) / len(labeled_df) * 100)
    })
    
    if verbose:
        print(f"\n=== Confident Learning Noise Filter Results ===")
        print(f"Original labeled samples: {len(labeled_df)}")
        print(f"Mislabeled candidates: {len(indices_to_remove)} ({noise_stats['estimated_noise_rate']:.1%})")
        print(f"Samples remaining: {len(filtered_labeled_mask)}")
        
        if snr_improvement:
            print(f"SNR change: {snr_before['snr']:.3f} → {snr_after['snr']:.3f} ({snr_improvement['snr_pct_change']:+.1f}%)")
            print(f"Cohen's d change: {snr_before['cohens_d']:.3f} → {snr_after['cohens_d']:.3f} ({snr_improvement['cohens_d_delta']:+.3f})")
    
    return {
        "filtered_df": filtered_df,
        "noise_stats": noise_stats,
        "snr_before": snr_before,
        "snr_after": snr_after,
        "snr_improvement": snr_improvement,
        "applied_filter": len(indices_to_remove) > 0,
        "indices_removed": indices_to_remove
    }


def _compute_snr_metrics(y_true: pd.Series, y_returns: Optional[pd.Series] = None) -> dict:
    """Compute SNR metrics for label quality assessment."""
    if len(y_true) == 0:
        return {}
    
    # Basic metrics
    n_positive = int((y_true == 1).sum())
    n_negative = int((y_true == 0).sum())
    n_total = len(y_true)
    
    if n_positive == 0 or n_negative == 0:
        return {
            "n_positive": n_positive,
            "n_negative": n_negative,
            "n_total": n_total,
            "positive_rate": n_positive / n_total,
            "snr": 0.0,
            "cohens_d": 0.0
        }
    
    # Compute effect size and SNR if returns available
    snr = 0.0
    cohens_d = 0.0
    
    if y_returns is not None and len(y_returns) == len(y_true):
        try:
            pos_returns = y_returns[y_true == 1].dropna()
            neg_returns = y_returns[y_true == 0].dropna()
            
            if len(pos_returns) > 1 and len(neg_returns) > 1:
                pos_mean = float(pos_returns.mean())
                neg_mean = float(neg_returns.mean())
                pos_std = float(pos_returns.std())
                neg_std = float(neg_returns.std())
                
                # Cohen's d
                pooled_std = np.sqrt(((len(pos_returns) - 1) * pos_std**2 + (len(neg_returns) - 1) * neg_std**2) / 
                                   (len(pos_returns) + len(neg_returns) - 2))
                cohens_d = (pos_mean - neg_mean) / max(pooled_std, 1e-8)
                
                # SNR (signal-to-noise ratio)
                signal = abs(pos_mean - neg_mean)
                noise = np.sqrt((pos_std**2 + neg_std**2) / 2)
                snr = signal / max(noise, 1e-8)
        except Exception:
            pass
    
    return {
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_total": n_total,
        "positive_rate": n_positive / n_total,
        "snr": snr,
        "cohens_d": cohens_d
    }

    # Find confident predictions (high confidence in one direction)
    confident_mask = confidence >= threshold_confident

    # Find disagreements: where confident prediction differs from true label
    disagreement_mask = (y_pred != y_true) & confident_mask

    # Potential mislabeled indices
    mislabeled_indices = np.where(disagreement_mask)[0]

    # Statistics
    n_confident = int(np.sum(confident_mask))
    n_disagreements = int(np.sum(disagreement_mask))
    noise_rate = n_disagreements / max(n_confident, 1)

    # Per-class analysis
    pos_mask = y_true == 1
    neg_mask = y_true == 0

    false_neg_rate = np.sum((y_pred[pos_mask] != y_true[pos_mask]) & confident_mask[pos_mask]) / max(np.sum(pos_mask), 1)
    false_pos_rate = np.sum((y_pred[neg_mask] != y_true[neg_mask]) & confident_mask[neg_mask]) / max(np.sum(neg_mask), 1)

    return {
        "n_confident_predictions": int(n_confident),
        "n_mislabeled_candidates": int(n_disagreements),
        "estimated_noise_rate": float(noise_rate),
        "false_neg_rate_confident": float(false_neg_rate),
        "false_pos_rate_confident": float(false_pos_rate),
        "mislabeled_indices": mislabeled_indices.tolist()[:100],  # Limit to first 100
    }


def _run_label_shuffle_cv(
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: int = 5,
) -> dict:
    """Run a label-shuffled time-series CV sanity check.

    Uses the same TimeSeriesSplit and probe LightGBM as the main robustness
    routine, but with labels randomly permuted along the time axis. AUC and
    Brier should collapse towards random (≈0.5 / ≈0.25) if the pipeline is
    free of structural leakage.
    """

    try:
        X_array = X.values.astype(float)
        y_array = y.values.astype(float)
    except Exception:
        return {"n_folds": 0}

    if X_array.size == 0 or y_array.size == 0:
        return {"n_folds": 0}

    rng = np.random.default_rng(12345)
    y_shuffled = y_array.copy()
    rng.shuffle(y_shuffled)

    tscv = TimeSeriesSplit(n_splits=cv_splits)
    aucs: list[float] = []
    briers: list[float] = []
    aps: list[float] = []

    for tr_idx, te_idx in tscv.split(X_array):
        X_tr = X_array[tr_idx]
        X_te = X_array[te_idx]
        y_tr = y_shuffled[tr_idx]
        y_te = y_shuffled[te_idx]

        mask_tr = ~np.isnan(y_tr)
        mask_te = ~np.isnan(y_te)
        y_tr_clean = y_tr[mask_tr]
        X_tr_clean = X_tr[mask_tr]
        y_te_clean = y_te[mask_te]
        X_te_clean = X_te[mask_te]

        if len(y_tr_clean) < 50 or len(y_te_clean) < 20:
            continue
        if len(np.unique(y_tr_clean)) < 2 or len(np.unique(y_te_clean)) < 2:
            continue

        try:
            clf = lgb.LGBMClassifier(
                boosting_type="gbdt",
                objective="binary",
                max_depth=3,
                n_estimators=50,
                learning_rate=0.1,
                subsample=0.7,
                colsample_bytree=0.7,
                min_child_samples=20,
                n_jobs=-1,
                verbose=-1,
                random_state=1337,
            )
            clf.fit(X_tr_clean, y_tr_clean)
            prob = clf.predict_proba(X_te_clean)[:, 1]

            auc = roc_auc_score(y_te_clean, prob)
            brier = brier_score_loss(y_te_clean, prob)
            ap = average_precision_score(y_te_clean, prob)
        except Exception:
            continue

        aucs.append(float(auc))
        briers.append(float(brier))
        aps.append(float(ap))

    if not aucs:
        return {"n_folds": 0}

    aucs_arr = np.array(aucs, dtype=float)
    briers_arr = np.array(briers, dtype=float)
    aps_arr = np.array(aps, dtype=float)

    return {
        "n_folds": int(len(aucs_arr)),
        "mean_auc": float(np.nanmean(aucs_arr)),
        "std_auc": float(np.nanstd(aucs_arr)),
        "mean_brier": float(np.nanmean(briers_arr)),
        "mean_ap": float(np.nanmean(aps_arr)),
    }


def _compute_strict_holdout_metrics(
    X: pd.DataFrame,
    y: pd.Series,
    holdout_fraction: float = 0.3,
) -> dict:
    """Compute a single forward holdout split (train early, test late).

    This approximates a more realistic "train-then-freeze" evaluation by
    using the earliest (1 - holdout_fraction) of samples for training and the
    last holdout_fraction for testing.
    """

    try:
        X_array = X.values.astype(float)
        y_array = y.values.astype(float)
    except Exception:
        return {}

    mask = ~np.isnan(y_array)
    X_clean = X_array[mask]
    y_clean = y_array[mask]

    n_total = len(y_clean)
    if n_total < 100:
        return {
            "n_total": int(n_total),
            "n_train": 0,
            "n_test": 0,
        }

    split_idx = int(n_total * (1.0 - holdout_fraction))
    if split_idx <= 50 or n_total - split_idx <= 20:
        return {
            "n_total": int(n_total),
            "n_train": 0,
            "n_test": 0,
        }

    X_tr = X_clean[:split_idx]
    y_tr = y_clean[:split_idx]
    X_te = X_clean[split_idx:]
    y_te = y_clean[split_idx:]

    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return {
            "n_total": int(n_total),
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
        }

    try:
        clf = lgb.LGBMClassifier(
            boosting_type="gbdt",
            objective="binary",
            max_depth=3,
            n_estimators=50,
            learning_rate=0.1,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_samples=20,
            n_jobs=-1,
            verbose=-1,
            random_state=4242,
        )
        clf.fit(X_tr, y_tr)
        prob = clf.predict_proba(X_te)[:, 1]

        auc = roc_auc_score(y_te, prob)
        brier = brier_score_loss(y_te, prob)
        ap = average_precision_score(y_te, prob)
    except Exception:
        return {
            "n_total": int(n_total),
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
        }

    return {
        "n_total": int(n_total),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "holdout_fraction": float(holdout_fraction),
        "auc": float(auc),
        "brier": float(brier),
        "ap": float(ap),
    }


def _scan_single_feature_leakage(
    X: pd.DataFrame,
    y: pd.Series,
    auc_threshold: float = 0.9,
    top_k: int = 10,
) -> dict:
    """Scan single features for unusually high AUC vs labels.

    This is a coarse leakage detector: if any individual feature can almost
    perfectly separate the labels on its own, it is likely carrying target-
    like information (e.g. realized return derivatives or hidden label codes).
    """

    results: list[dict] = []
    try:
        y_array = y.values.astype(float)
    except Exception:
        return {
            "features": [],
            "suspicious_features": [],
            "max_auc": None,
            "auc_threshold": float(auc_threshold),
        }

    for idx, col in enumerate(X.columns):
        try:
            x_col = X[col].values.astype(float)
        except Exception:
            continue

        mask = ~np.isnan(x_col) & ~np.isnan(y_array)
        if mask.sum() < 50:
            continue

        y_valid = y_array[mask]
        x_valid = x_col[mask]

        if len(np.unique(y_valid)) < 2:
            continue

        try:
            auc = roc_auc_score(y_valid, x_valid)
        except Exception:
            continue

        results.append(
            {
                "feature_name": str(col),
                "feature_idx": int(idx),
                "auc": float(auc),
            }
        )

    if not results:
        return {
            "features": [],
            "suspicious_features": [],
            "max_auc": None,
            "auc_threshold": float(auc_threshold),
        }

    # Sort by distance from random (0.5)
    results_sorted = sorted(results, key=lambda r: abs(r["auc"] - 0.5), reverse=True)
    suspicious = [
        r
        for r in results_sorted
        if r["auc"] >= auc_threshold or r["auc"] <= 1.0 - auc_threshold
    ][:top_k]

    max_auc = results_sorted[0]["auc"] if results_sorted else None

    return {
        "features": results_sorted[:top_k],
        "suspicious_features": suspicious,
        "max_auc": float(max_auc) if max_auc is not None and np.isfinite(max_auc) else None,
        "auc_threshold": float(auc_threshold),
    }


def _plot_temporal_auc(
    temporal_indices: list,
    temporal_aucs: list,
    symbol: str,
    timeframe: str,
) -> Path:
    """Generate and save temporal AUC plot."""
    try:
        fig, ax = plt.subplots(figsize=(12, 6))

        if temporal_indices and temporal_aucs:
            ax.plot(temporal_indices, temporal_aucs, marker='o', linewidth=2, markersize=6, label='Rolling AUC')
            ax.axhline(y=0.5, color='r', linestyle='--', label='Random (0.5)')
            ax.axhline(y=np.mean(temporal_aucs), color='g', linestyle='--', label=f'Mean ({np.mean(temporal_aucs):.3f})')

            ax.set_xlabel('Sample Index')
            ax.set_ylabel('AUC')
            ax.set_title(f'Temporal AUC Evolution: {symbol} {timeframe}')
            ax.set_ylim([0.4, 0.8])
            ax.grid(True, alpha=0.3)
            ax.legend()

        out_dir = _ensure_outcomes_dir()
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        plot_path = out_dir / f"temporal_auc_{symbol}_{timeframe}_{ts}.png"

        fig.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Saved temporal AUC plot to {plot_path}")
        return plot_path
    except Exception as e:
        logger.warning(f"Failed to generate temporal AUC plot: {e}")
        return Path()


# --------------------------------------------------------------------------------------
# Label-quality diagnostics
# --------------------------------------------------------------------------------------


def run_label_quality(
    symbol: str,
    exchange: str,
    timeframe: str,
    direction: str = "long",
    model: str = "analyst",
) -> None:
    """Compute label-quality and economic SNR diagnostics from labeled_data."""
    df = _load_labeled_data(symbol, exchange, timeframe, direction=direction, model=model)

    # Prefer directional binary labels
    if direction == "long" and "binary_label_long" in df.columns:
        binary_labels = df["binary_label_long"]
    elif direction == "short" and "binary_label_short" in df.columns:
        binary_labels = df["binary_label_short"]
    elif "binary_label_long" in df.columns:
        binary_labels = df["binary_label_long"]
    elif "binary_label_short" in df.columns:
        binary_labels = df["binary_label_short"]
    elif "binary_label" in df.columns:
        binary_labels = df["binary_label"]
    else:
        raise ValueError(
            "labeled_data must contain a binary label column "
            "(binary_label_long, binary_label_short, or binary_label)"
        )
    
    if "realized_return" not in df.columns:
        raise ValueError("labeled_data must contain 'realized_return' column")
    realized_returns = df["realized_return"]

    n_samples = len(df)
    labeled_mask = ~binary_labels.isna()
    n_labeled = int(labeled_mask.sum())

    n_positive = int((binary_labels == 1.0).sum())
    n_negative = int((binary_labels == 0.0).sum())
    positive_rate = n_positive / n_labeled if n_labeled > 0 else 0.0
    coverage = n_labeled / n_samples if n_samples > 0 else 0.0

    # Pre-filter: all events with realized returns
    pre_mask = ~realized_returns.isna()
    n_pre_total = int(pre_mask.sum())

    tx_cost = float(DEFAULT_TRANSACTION_COST)

    if n_pre_total > 0:
        pre_returns = realized_returns[pre_mask]
        raw_label_pre = (pre_returns > tx_cost).astype(int)
        n_pre_pos = int((raw_label_pre == 1).sum())
        n_pre_neg = int((raw_label_pre == 0).sum())
    else:
        pre_returns = pd.Series(dtype=float)
        raw_label_pre = pd.Series(dtype=float)
        n_pre_pos = n_pre_neg = 0

    # Retention metrics
    n_post_total = n_labeled
    n_post_pos = n_positive
    n_post_neg = n_negative

    retention_total = n_post_total / max(n_pre_total, 1)
    retention_pos = n_post_pos / max(n_pre_pos, 1) if n_pre_pos > 0 else 0.0
    retention_neg = n_post_neg / max(n_pre_neg, 1) if n_pre_neg > 0 else 0.0

    def _safe_stats(x: pd.Series) -> Tuple[float, float, int]:
        x_clean = x.dropna()
        if len(x_clean) == 0:
            return 0.0, 0.0, 0
        return float(x_clean.mean()), float(x_clean.std()), len(x_clean)

    # Pre-filter stats (raw economic labels)
    pre_pos_ret = pre_returns[raw_label_pre == 1]
    pre_neg_ret = pre_returns[raw_label_pre == 0]

    pre_pos_mean, pre_pos_std, n_pre_pos_eff = _safe_stats(pre_pos_ret)
    pre_neg_mean, pre_neg_std, n_pre_neg_eff = _safe_stats(pre_neg_ret)

    # Post-filter stats on labeled events
    returns_labeled = realized_returns[labeled_mask]
    labels_clean = binary_labels[labeled_mask]

    post_pos_ret = returns_labeled[labels_clean == 1]
    post_neg_ret = returns_labeled[labels_clean == 0]

    post_pos_mean, post_pos_std, n_post_pos_eff = _safe_stats(post_pos_ret)
    post_neg_mean, post_neg_std, n_post_neg_eff = _safe_stats(post_neg_ret)

    def _cohens_d(m1, s1, n1, m2, s2, n2) -> float:
        if n1 <= 1 or n2 <= 1:
            return float("nan")
        pooled = ((n1 - 1) * (s1 ** 2) + (n2 - 1) * (s2 ** 2)) / max(n1 + n2 - 2, 1)
        if pooled <= 0:
            return float("nan")
        return (m1 - m2) / np.sqrt(pooled)

    d_pre = _cohens_d(pre_pos_mean, pre_pos_std, n_pre_pos_eff, pre_neg_mean, pre_neg_std, n_pre_neg_eff)
    d_post = _cohens_d(post_pos_mean, post_pos_std, n_post_pos_eff, post_neg_mean, post_neg_std, n_post_neg_eff)

    snr_pre = pre_pos_mean / (pre_pos_std + 1e-8) if pre_pos_std > 0 else 0.0
    snr_post = post_pos_mean / (post_pos_std + 1e-8) if post_pos_std > 0 else 0.0

    # Label overlap diagnostic
    overlap_pos_in_neg = int((post_pos_ret < 0).sum())
    overlap_neg_in_pos = int((post_neg_ret > 0).sum())
    total_events_for_overlap = len(post_pos_ret) + len(post_neg_ret)
    if total_events_for_overlap > 0:
        pct_overlap = (overlap_pos_in_neg + overlap_neg_in_pos) / total_events_for_overlap
    else:
        pct_overlap = 0.0

    # Cost-aware event quality
    if len(returns_labeled.dropna()) > 0:
        unconditional_mean = float(returns_labeled.mean())
        frac_small = float((returns_labeled.abs() < tx_cost).mean())
    else:
        unconditional_mean = 0.0
        frac_small = 0.0

    aleatoric_fraction = float(frac_small)
    if aleatoric_fraction < 0.40:
        aleatoric_comment = "Aleatoric fraction < 40%: most error is model/feature-driven; improvement is possible."
    elif aleatoric_fraction < 0.60:
        aleatoric_comment = "Aleatoric fraction 40–60%: mixed noise and model limitations."
    else:
        aleatoric_comment = "Aleatoric fraction > 60%: most unpredictability is intrinsic to the target."

    if len(post_pos_ret.dropna()) > 0:
        mean_pos_ret = float(post_pos_ret.mean())
    else:
        mean_pos_ret = 0.0

    # Prepare isotonic expected returns for bucket diagnostics
    expected_ret = None
    if "target_long" in df.columns or "target_short" in df.columns:
        try:
            if direction == "long" and "target_long" in df.columns:
                expected_ret = df["target_long"].astype(float)
            elif direction == "short" and "target_short" in df.columns:
                expected_ret = df["target_short"].astype(float)
            else:
                # Fallback: combine long/short targets into a single expected return
                tl = df["target_long"].astype(float) if "target_long" in df.columns else pd.Series(0.0, index=df.index)
                ts = df["target_short"].astype(float) if "target_short" in df.columns else pd.Series(0.0, index=df.index)
                expected_ret = tl.where(tl > 0, 0.0) - ts.where(ts > 0, 0.0)
        except Exception:
            expected_ret = None

    # High-probability bucket diagnostics (top-k% by meta_probability),
    # using isotonic expected returns instead of raw realized returns.
    bucket_stats = {}
    prob_series = None
    prob_variance_warning = None
    if "meta_probability" in df.columns and expected_ret is not None:
        prob_series = df["meta_probability"].astype(float)
        valid_bucket_mask = labeled_mask & prob_series.notna() & expected_ret.notna()
        if valid_bucket_mask.any():
            probs_valid = prob_series[valid_bucket_mask]
            rets_valid = expected_ret[valid_bucket_mask]
            labels_valid = binary_labels[valid_bucket_mask]
            
            # Check for constant or near-constant probabilities (indicates model not learning)
            prob_std = float(probs_valid.std())
            prob_range = float(probs_valid.max() - probs_valid.min())
            if prob_std < 0.01 or prob_range < 0.05:
                prob_variance_warning = (
                    f"⚠️ Low probability variance (std={prob_std:.4f}, range={prob_range:.4f}). "
                    "Model may be outputting constant probabilities - bucket analysis unreliable."
                )

            bucket_fracs = [0.05, 0.10, 0.20, 0.30, 0.40]
            for frac in bucket_fracs:
                if len(probs_valid) < max(int(1.0 / frac), 50):
                    continue
                try:
                    q = probs_valid.quantile(1.0 - frac)
                    bucket_mask = (probs_valid >= q)
                    if bucket_mask.sum() < 20:
                        continue

                    rets_bucket = rets_valid[bucket_mask]
                    labels_bucket = labels_valid[bucket_mask]
                    win_rate_bucket = float((labels_bucket == 1.0).mean())
                    mean_ret_bucket = float(rets_bucket.mean()) if len(rets_bucket) > 0 else 0.0
                    std_ret_bucket = float(rets_bucket.std()) if len(rets_bucket) > 1 else 0.0
                    sharpe_bucket = mean_ret_bucket / (std_ret_bucket + 1e-8) if std_ret_bucket > 0 else 0.0

                    key = f"top_{int(frac * 100)}"
                    bucket_stats[key] = {
                        "frac": float(frac),
                        "threshold": float(q),
                        "n_events": int(bucket_mask.sum()),
                        "win_rate": float(win_rate_bucket),
                        "mean_expected_return": float(mean_ret_bucket),
                        "sharpe_expected": float(sharpe_bucket),
                    }
                except Exception:
                    continue

    # Enhanced volatility-bucket diagnostics with adaptive thresholds
    vol_bucket_stats = {}
    if "volatility_1d" in df.columns:
        try:
            vol = df["volatility_1d"].astype(float)
            vol_mask = labeled_mask & vol.notna()
            if vol_mask.sum() >= 60:
                vol_valid = vol[vol_mask]
                low_thr = float(vol_valid.quantile(1.0 / 3.0))
                high_thr = float(vol_valid.quantile(2.0 / 3.0))

                regimes = {
                    "low": vol < low_thr,
                    "mid": (vol >= low_thr) & (vol < high_thr),
                    "high": vol >= high_thr,
                }

                # Get adaptive threshold information if available
                adaptive_profit = df.get("adaptive_profit_threshold")
                adaptive_stop = df.get("adaptive_stop_threshold")

                for name, regime_mask in regimes.items():
                    seg_mask = vol_mask & regime_mask
                    if seg_mask.sum() < 30:
                        continue

                    seg_returns = realized_returns[seg_mask]
                    seg_labels = binary_labels[seg_mask]

                    seg_ret_clean = seg_returns.dropna()
                    seg_labels_clean = seg_labels[~seg_labels.isna()]

                    if len(seg_ret_clean) == 0 or len(seg_labels_clean) == 0:
                        continue

                    seg_mean = float(seg_ret_clean.mean())
                    seg_std = float(seg_ret_clean.std()) if len(seg_ret_clean) > 1 else 0.0
                    seg_sharpe = seg_mean / (seg_std + 1e-8) if seg_std > 0 else 0.0

                    seg_pos = int((seg_labels == 1.0).sum())
                    seg_neg = int((seg_labels == 0.0).sum())
                    seg_total = int(seg_mask.sum())
                    seg_pos_rate = seg_pos / max(seg_pos + seg_neg, 1)

                    bucket_stats = {
                        "n_events": seg_total,
                        "n_positive": seg_pos,
                        "n_negative": seg_neg,
                        "positive_rate": float(seg_pos_rate),
                        "mean_return": float(seg_mean),
                        "sharpe": float(seg_sharpe),
                        "low_threshold": float(low_thr),
                        "high_threshold": float(high_thr),
                    }

                    # Add adaptive threshold statistics if available
                    if adaptive_profit is not None and adaptive_stop is not None:
                        seg_adaptive_profit = adaptive_profit[seg_mask].dropna()
                        seg_adaptive_stop = adaptive_stop[seg_mask].dropna()
                        
                        if len(seg_adaptive_profit) > 0:
                            bucket_stats.update({
                                "adaptive_threshold_mean": float(seg_adaptive_profit.mean()),
                                "adaptive_threshold_std": float(seg_adaptive_profit.std()),
                                "adaptive_stop_mean": float(seg_adaptive_stop.mean()),
                                "adaptive_stop_std": float(seg_adaptive_stop.std()),
                            })

                    vol_bucket_stats[name] = bucket_stats
        except Exception as e:
            print(f"Warning: Volatility bucket analysis failed: {e}")
            vol_bucket_stats = {}

    # Simple interpretation helpers for coverage, effect size, SNR and retention
    if coverage < 0.05:
        coverage_comment = "Low coverage (<5%): labels are very sparse; probe models may struggle."
    elif coverage < 0.2:
        coverage_comment = "Moderate coverage (5–20%): typical for event-driven labeling."
    else:
        coverage_comment = "High coverage (>20%): many labeled events; check for redundancy or label noise."

    def _effect_comment(d_val: float) -> str:
        if not np.isfinite(d_val):
            return "Effect size not available (insufficient data)."
        ad = abs(d_val)
        if ad < 0.2:
            return "Very weak separation between positive and negative returns."
        if ad < 0.5:
            return "Small separation between positive and negative returns."
        if ad < 0.8:
            return "Moderate separation between positive and negative returns."
        if ad < 1.5:
            return "Large separation; labels correlate well with economic outcomes."
        return "Very large separation; labels are strongly aligned with economic outcomes."

    effect_post_comment = _effect_comment(d_post)

    if snr_post < 0.5:
        snr_comment = "Low SNR: positive-label returns are noisy relative to their mean."
    elif snr_post < 1.0:
        snr_comment = "Moderate SNR: some signal, but still fairly noisy."
    else:
        snr_comment = "High SNR: positive-label returns are well separated from noise."

    if retention_total < 0.1:
        retention_comment = "Filters are extremely aggressive; only a small fraction of events are kept."
    elif retention_total < 0.3:
        retention_comment = "Filters are moderately aggressive; many events are discarded."
    else:
        retention_comment = "Filters keep a substantial share of events; label density is relatively high."

    noise_ceiling = None
    noise_ceiling_comment = (
        "Noise ceiling requires multiple labelers or repeated labels; "
        "not available in current artifacts."
    )

    def _score_component_lq(value: float, low: float, high: float) -> float:
        if value is None or not np.isfinite(value):
            return 0.0
        if value <= low:
            return 0.0
        if value >= high:
            return 1.0
        return float((value - low) / (high - low))

    coverage_score = _score_component_lq(coverage, 0.05, 0.2)
    retention_score = _score_component_lq(retention_total, 0.1, 0.3)
    snr_score = _score_component_lq(snr_post, 0.5, 1.0)
    d_score = _score_component_lq(abs(d_post) if np.isfinite(d_post) else float("nan"), 0.2, 1.5)
    econ_margin = mean_pos_ret - tx_cost
    econ_score = _score_component_lq(econ_margin, 0.0, 0.02)

    label_quality_score_components = [coverage_score, retention_score, snr_score, d_score, econ_score]
    label_quality_score = float(np.mean(label_quality_score_components))

    if label_quality_score < 0.4:
        label_quality_rating = "Bad"
        label_quality_comment = "Low coverage/SNR or weak economic separation; labels are likely noisy or too sparse."
    elif label_quality_score < 0.7:
        label_quality_rating = "Pass"
        label_quality_comment = "Mixed label quality; some usable signal but economic separation or coverage may be modest."
    else:
        label_quality_rating = "Great"
        label_quality_comment = "Strong label quality with good coverage, separation and economic margins."

    # Console output
    print("""
=== Label-Quality Diagnostics ===
""".strip())

    print(f"Symbol: {symbol} | Exchange: {exchange} | Timeframe: {timeframe}")
    print(f"Total samples: {n_samples}")
    print(f"Labeled samples: {n_labeled} (coverage={coverage:.1%})")
    print(f"Positive labels: {n_positive} ({positive_rate:.1%}), Negative labels: {n_negative}")
    print()

    print("-- Pre vs Post Filter Retention --")
    print(f"Pre-filter events (realized_return not NaN): {n_pre_total}")
    print(f"Pre-filter positive/negative (raw econ > cost): {n_pre_pos} / {n_pre_neg}")
    print(f"Post-filter labeled events: {n_post_total}")
    print(f"Post-filter positive/negative (binary_label): {n_post_pos} / {n_post_neg}")
    print(f"Total retention (post / pre): {retention_total:.1%}")
    print(f"Positive retention: {retention_pos:.1%}")
    print(f"Negative retention: {retention_neg:.1%}")
    print()

    print("-- Economic Separation and SNR --")
    print(f"Pre-filter mean return (label=1/0): {pre_pos_mean:.2%} / {pre_neg_mean:.2%}")
    print(f"Post-filter mean return (label=1/0): {post_pos_mean:.2%} / {post_neg_mean:.2%}")
    print(f"Pre-filter Cohen's d (1 vs 0): {d_pre:.3f}")
    print(f"Post-filter Cohen's d (1 vs 0): {d_post:.3f}")
    print(f"Pre-filter SNR (mean/std, label=1): {snr_pre:.3f}")
    print(f"Post-filter SNR (mean/std, label=1): {snr_post:.3f}")
    print()

    print("-- Label Overlap and Cost-Aware Quality --")
    print(f"Label overlap (mis-signed P&L share): {pct_overlap:.1%}")
    print(f"Transaction cost (approx per event): {tx_cost:.3%}")
    print(f"Unconditional mean event return: {unconditional_mean:.2%}")
    print(f"Mean return (label=1) minus cost: {(mean_pos_ret - tx_cost):.2%}")
    print(f"Fraction of labeled events with |return| < cost: {frac_small:.1%}")

    print()
    print("-- Aleatoric Uncertainty --")
    print(f"Aleatoric uncertainty fraction (|return| < cost): {aleatoric_fraction:.1%}")
    print(f"Interpretation: {aleatoric_comment}")

    if bucket_stats:
        print()
        print("-- High-Probability Buckets (by meta_probability, isotonic expected returns) --")
        if prob_variance_warning:
            print(prob_variance_warning)
        for key in sorted(bucket_stats.keys(), key=lambda k: bucket_stats[k]["frac"]):
            stats = bucket_stats[key]
            print(
                f"Top {int(stats['frac']*100):2d}%: n={stats['n_events']}, "
                f"win_rate={stats['win_rate']:.1%}, "
                f"mean_exp_ret={stats['mean_expected_return']:.2%}, "
                f"Sharpe_exp={stats['sharpe_expected']:.2f}"
            )

    if vol_bucket_stats:
        print()
        print("-- Enhanced Volatility Buckets (by volatility_1d) --")
        for name in ["low", "mid", "high"]:
            if name not in vol_bucket_stats:
                continue
            stats = vol_bucket_stats[name]
            print(
                f"Vol {name:>4}: n={stats['n_events']}, "
                f"pos_rate={stats['positive_rate']:.1%}, "
                f"mean_ret={stats['mean_return']:.2%}, "
                f"Sharpe={stats['sharpe']:.2f}, "
                f"vol_range=[{stats.get('low_threshold', 'nan'):.4f}, {stats.get('high_threshold', 'nan'):.4f}]"
            )
            
            # Add volatility-aware threshold diagnostics if available
            if 'adaptive_threshold_mean' in stats:
                print(f"         Adaptive profit threshold: {stats['adaptive_threshold_mean']:.4f} ± {stats.get('adaptive_threshold_std', 0):.4f}")
                print(f"         Adaptive stop threshold: {stats['adaptive_stop_mean']:.4f} ± {stats.get('adaptive_stop_std', 0):.4f}")

    print()
    print("-- Interpretation Hints --")
    print(f"Coverage: {coverage:.1%} → {coverage_comment}")
    print(f"Post-filter effect size (Cohen's d={d_post:.3f}) → {effect_post_comment}")
    print(f"Post-filter SNR (label=1: {snr_post:.3f}) → {snr_comment}")
    print(f"Retention (total={retention_total:.1%}) → {retention_comment}")

    # Export payload
    payload = {
        "section": "label_quality",
        "n_samples": int(n_samples),
        "n_labeled": int(n_labeled),
        "coverage": float(coverage),
        "n_positive": int(n_positive),
        "n_negative": int(n_negative),
        "positive_rate": float(positive_rate),
        "pre": {
            "n_total": int(n_pre_total),
            "n_positive": int(n_pre_pos),
            "n_negative": int(n_pre_neg),
            "mean_pos_return": float(pre_pos_mean),
            "mean_neg_return": float(pre_neg_mean),
            "cohens_d": float(d_pre) if np.isfinite(d_pre) else None,
            "snr_pos": float(snr_pre),
        },
        "post": {
            "n_total": int(n_post_total),
            "n_positive": int(n_post_pos),
            "n_negative": int(n_post_neg),
            "mean_pos_return": float(post_pos_mean),
            "mean_neg_return": float(post_neg_mean),
            "cohens_d": float(d_post) if np.isfinite(d_post) else None,
            "snr_pos": float(snr_post),
        },
        "retention": {
            "total": float(retention_total),
            "positive": float(retention_pos),
            "negative": float(retention_neg),
        },
        "overlap": {
            "pct_overlap": float(pct_overlap),
        },
        "cost_metrics": {
            "tx_cost": float(tx_cost),
            "unconditional_mean_return": float(unconditional_mean),
            "mean_pos_minus_cost": float(mean_pos_ret - tx_cost),
            "frac_small_vs_cost": float(frac_small),
        },
        "probability_buckets": bucket_stats,
        "probability_variance_warning": prob_variance_warning,
        "volatility_buckets": vol_bucket_stats,
        "advanced": {
            "aleatoric_uncertainty_fraction": float(aleatoric_fraction),
            "aleatoric_comment": aleatoric_comment,
            "noise_ceiling": noise_ceiling,
            "noise_ceiling_comment": noise_ceiling_comment,
        },
        "summary_score": {
            "score": float(label_quality_score),
            "rating": label_quality_rating,
            "comment": label_quality_comment,
        },
    }

    md_lines = [
        "# SNR Label-Quality Diagnostics",
        "",
        f"**Symbol**: {symbol}",
        f"**Exchange**: {exchange}",
        f"**Timeframe**: {timeframe}",
        "",
        "## Summary",
        f"- Total samples: {n_samples}",
        f"- Labeled samples: {n_labeled} (coverage={coverage:.1%})",
        f"- Positive labels: {n_positive} ({positive_rate:.1%})",
        f"- Negative labels: {n_negative}",
        "",
        "## Retention",
        f"- Pre-filter events (realized_return not NaN): {n_pre_total}",
        f"- Pre-filter pos/neg (raw econ > cost): {n_pre_pos} / {n_pre_neg}",
        f"- Post-filter labeled events: {n_post_total}",
        f"- Post-filter pos/neg (binary_label): {n_post_pos} / {n_post_neg}",
        f"- Total retention: {retention_total:.1%}",
        f"- Positive retention: {retention_pos:.1%}",
        f"- Negative retention: {retention_neg:.1%}",
        "",
        "## Economic Separation and SNR",
        f"- Pre-filter mean return (label=1/0): {pre_pos_mean:.2%} / {pre_neg_mean:.2%}",
        f"- Post-filter mean return (label=1/0): {post_pos_mean:.2%} / {post_neg_mean:.2%}",
        f"- Pre-filter Cohen's d: {d_pre:.3f}",
        f"- Post-filter Cohen's d: {d_post:.3f}",
        f"- Pre-filter SNR (label=1): {snr_pre:.3f}",
        f"- Post-filter SNR (label=1): {snr_post:.3f}",
        "",
        "## Label Overlap and Cost Metrics",
        f"- Label overlap (mis-signed P&L share): {pct_overlap:.1%}",
        f"- Transaction cost (approx per event): {tx_cost:.3%}",
        f"- Unconditional mean event return: {unconditional_mean:.2%}",
        f"- Mean return (label=1) minus cost: {(mean_pos_ret - tx_cost):.2%}",
        f"- Fraction of labeled events with |return| < cost: {frac_small:.1%}",
        f"- Aleatoric uncertainty fraction (|return| < cost): {aleatoric_fraction:.1%}",
        "",
        "## High-Probability Buckets (by meta_probability, isotonic expected returns)",
    ]

    if bucket_stats:
        if prob_variance_warning:
            md_lines.append(f"\n{prob_variance_warning}\n")
        for key in sorted(bucket_stats.keys(), key=lambda k: bucket_stats[k]["frac"]):
            stats = bucket_stats[key]
            md_lines.append(
                f"- Top {int(stats['frac']*100):2d}%: n={stats['n_events']}, "
                f"win_rate={stats['win_rate']:.1%}, "
                f"mean_exp_ret={stats['mean_expected_return']:.2%}, "
                f"Sharpe_exp={stats['sharpe_expected']:.2f}"
            )
    else:
        md_lines.append("- meta_probability not available or insufficient data for bucket diagnostics.")

    md_lines.extend([
        "",
        "## Enhanced Volatility Buckets (by volatility_1d)",
    ])

    if vol_bucket_stats:
        for name in ["low", "mid", "high"]:
            if name not in vol_bucket_stats:
                continue
            stats = vol_bucket_stats[name]
            md_lines.append(
                f"- Vol {name}: n={stats['n_events']}, "
                f"pos_rate={stats['positive_rate']:.1%}, "
                f"mean_ret={stats['mean_return']:.2%}, "
                f"Sharpe={stats['sharpe']:.2f}, "
                f"vol_range=[{stats.get('low_threshold', 'nan'):.4f}, {stats.get('high_threshold', 'nan'):.4f}]"
            )
            
            # Add adaptive threshold information to markdown
            if 'adaptive_threshold_mean' in stats:
                md_lines.append(
                    f"  - Adaptive profit threshold: {stats['adaptive_threshold_mean']:.4f} ± {stats.get('adaptive_threshold_std', 0):.4f}"
                )
                md_lines.append(
                    f"  - Adaptive stop threshold: {stats['adaptive_stop_mean']:.4f} ± {stats.get('adaptive_stop_std', 0):.4f}"
                )
    else:
        md_lines.append("- volatility_1d not available or insufficient data for volatility buckets.")

    md_lines.extend([
        "",
        "## Interpretation Hints",
        f"- Coverage ({coverage:.1%}): {coverage_comment}",
        f"- Post-filter effect size (Cohen's d={d_post:.3f}): {effect_post_comment}",
        f"- Post-filter SNR (label=1): {snr_post:.3f} → {snr_comment}",
        f"- Retention (total={retention_total:.1%}): {retention_comment}",
        "",
        "## Overall Label-Quality Score",
        f"- Score (0-1): {label_quality_score:.3f}",
        f"- Rating: {label_quality_rating}",
        f"- Summary: {label_quality_comment}",
    ])

    json_path, md_path = _export_report(
        prefix="snr_label_quality",
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        direction=direction,
        model=model,
        payload=payload,
        markdown_lines=md_lines,
    )

    print(f"\nReports saved to: {json_path} and {md_path}")


# --------------------------------------------------------------------------------------
# Label-learnability diagnostics
# --------------------------------------------------------------------------------------


def run_label_learnability(
    symbol: str,
    exchange: str,
    timeframe: str,
    direction: str = "long",
    model: str = "analyst",
    cv_splits: int = 3,
) -> None:
    """Compute learnability & entropy-based label-quality scores."""
    df = _load_labeled_data(symbol, exchange, timeframe, direction=direction, model=model)
    X, y = _build_feature_matrix_from_labeled(df, direction=direction)

    learnability, mean_auc = compute_learnability_score(X, y, cv_splits=cv_splits)
    balance = compute_label_entropy_score(y)
    combined, diagnostics = combined_label_quality_objective(
        X,
        y,
        learnability_weight=0.7,
        balance_weight=0.3,
        cv_splits=cv_splits,
    )

    n_valid = int((~y.isna()).sum())
    pos_rate = float(y.mean()) if n_valid > 0 else 0.0

    # Interpretation helpers for learnability and balance
    if mean_auc < 0.55:
        auc_comment = "Mean CV AUC < 0.55 → very weak learnability; labels are close to random."
    elif mean_auc < 0.6:
        auc_comment = "Mean CV AUC 0.55–0.60 → weak but potentially usable signal."
    elif mean_auc < 0.7:
        auc_comment = "Mean CV AUC 0.60–0.70 → moderate learnability."
    else:
        auc_comment = "Mean CV AUC ≥ 0.70 → strong learnability; labels are easy to learn."

    if balance < 0.5:
        balance_comment = "Entropy score < 0.5 → labels are highly imbalanced or dominated by one class."
    elif balance < 0.8:
        balance_comment = "Entropy score 0.5–0.8 → some imbalance but usually acceptable."
    else:
        balance_comment = "Entropy score ≥ 0.8 → labels are well balanced."

    if combined < 0.4:
        combined_comment = "Combined score < 0.4 → overall label quality is weak; consider revisiting thresholds."
    elif combined < 0.6:
        combined_comment = "Combined score 0.4–0.6 → mixed quality; may be adequate for robust models."
    else:
        combined_comment = "Combined score ≥ 0.6 → good overall label quality."

    # Map combined score into [0, 1] summary with rating
    learnability_score = float(max(0.0, min(1.0, combined)))
    if learnability_score < 0.4:
        learnability_rating = "Bad"
    elif learnability_score < 0.6:
        learnability_rating = "Pass"
    else:
        learnability_rating = "Great"

    # Console output
    print("""
=== Label-Learnability Diagnostics ===
""".strip())
    print(f"Symbol: {symbol} | Exchange: {exchange} | Timeframe: {timeframe}")
    print(f"Valid labeled samples: {n_valid}")
    print(f"Positive label rate: {pos_rate:.1%}")
    print()

    print("-- Learnability (Probe Model AUC) --")
    print(f"Mean CV AUC: {mean_auc:.4f}")
    print(f"Learnability score (AUC - 0.5 * std): {learnability:.4f}")
    print()

    print("-- Entropy / Balance --")
    print(f"Entropy-based balance score: {balance:.4f}")
    print()

    print("-- Combined Label-Quality Objective (0.7 * learnability + 0.3 * balance) --")
    print(f"Combined score: {combined:.4f}")
    print()

    print("Diagnostics snapshot:")
    for k in sorted(diagnostics.keys()):
        v = diagnostics[k]
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print()
    print("-- Interpretation Hints --")
    print(f"Learnability (mean AUC={mean_auc:.4f}) → {auc_comment}")
    print(f"Balance (entropy score={balance:.4f}) → {balance_comment}")
    print(f"Combined score ({combined:.4f}) → {combined_comment}")

    # Export payload
    payload = {
        "section": "label_learnability",
        "n_valid": int(n_valid),
        "positive_rate": float(pos_rate),
        "learnability": float(learnability),
        "mean_auc": float(mean_auc),
        "balance": float(balance),
        "combined": float(combined),
        "diagnostics": diagnostics,
        "summary_score": {
            "score": learnability_score,
            "rating": learnability_rating,
            "comment": combined_comment,
        },
    }

    md_lines = [
        "# Label-Learnability Diagnostics",
        "",
        f"**Symbol**: {symbol}",
        f"**Exchange**: {exchange}",
        f"**Timeframe**: {timeframe}",
        "",
        "## Summary",
        f"- Valid labeled samples: {n_valid}",
        f"- Positive label rate: {pos_rate:.1%}",
        "",
        "## Learnability",
        f"- Mean CV AUC: {mean_auc:.4f}",
        f"- Learnability score (AUC - 0.5 * std): {learnability:.4f}",
        "",
        "## Entropy / Balance",
        f"- Balance score: {balance:.4f}",
        "",
        "## Combined Label-Quality Objective",
        f"- Combined score: {combined:.4f}",
        "",
        "## Interpretation Hints",
        f"- Learnability (mean AUC={mean_auc:.4f}): {auc_comment}",
        f"- Balance (entropy score={balance:.4f}): {balance_comment}",
        f"- Combined score ({combined:.4f}): {combined_comment}",
        "",
        "## Overall Learnability Score",
        f"- Score (0-1): {learnability_score:.3f}",
        f"- Rating: {learnability_rating}",
        f"- Summary: {combined_comment}",
    ]

    json_path, md_path = _export_report(
        prefix="snr_label_learnability",
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        direction=direction,
        model=model,
        payload=payload,
        markdown_lines=md_lines,
    )

    print(f"\nReports saved to: {json_path} and {md_path}")


# --------------------------------------------------------------------------------------
# Model-robustness diagnostics
# --------------------------------------------------------------------------------------


def run_model_robustness(
    symbol: str,
    exchange: str,
    timeframe: str,
    direction: str = "long",
    model: str = "analyst",
    cv_splits: int = 5,
) -> None:
    """Run a probe LightGBM model with time-series CV to assess robustness.

    Reports per-fold AUC, Brier score, PR-AUC, and summary statistics.
    """
    df = _load_labeled_data(symbol, exchange, timeframe, direction=direction, model=model)
    X, y = _build_feature_matrix_from_labeled(df, direction=direction)

    y_array = y.values.astype(float)
    X_array = X.values.astype(float)

    tscv = TimeSeriesSplit(n_splits=cv_splits)

    fold_metrics = []
    all_y_true = []
    all_p_pred = []
    all_p_baseline = []
    for fold_idx, (tr_idx, te_idx) in enumerate(tscv.split(X_array), start=1):
        X_tr, X_te = X_array[tr_idx], X_array[te_idx]
        y_tr, y_te = y_array[tr_idx], y_array[te_idx]

        # Require both classes in train and test for meaningful AUC
        if len(np.unique(y_tr[~np.isnan(y_tr)])) < 2 or len(np.unique(y_te[~np.isnan(y_te)])) < 2:
            continue

        # Clean NaNs in labels consistently between X and y
        mask_tr = ~np.isnan(y_tr)
        y_tr_clean = y_tr[mask_tr]
        X_tr_clean = X_tr[mask_tr]
        mask_te = ~np.isnan(y_te)
        y_te_clean = y_te[mask_te]
        X_te_clean = X_te[mask_te]

        if len(y_tr_clean) < 50 or len(y_te_clean) < 20:
            continue

        clf = lgb.LGBMClassifier(
            boosting_type="gbdt",
            objective="binary",
            max_depth=3,
            n_estimators=50,
            learning_rate=0.1,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_samples=20,
            n_jobs=-1,
            verbose=-1,
            random_state=42,
        )

        clf.fit(X_tr_clean, y_tr_clean)
        prob = clf.predict_proba(X_te_clean)[:, 1]

        # Logistic regression model for model-family comparison
        log_clf = LogisticRegression(solver="lbfgs", max_iter=200)
        log_clf.fit(X_tr_clean, y_tr_clean)
        log_prob = log_clf.predict_proba(X_te_clean)[:, 1]

        # Naive baseline: constant probability equal to training positive rate
        pos_rate_tr = float(np.nanmean(y_tr_clean)) if len(y_tr_clean) > 0 else 0.5
        baseline_prob = np.full_like(prob, fill_value=pos_rate_tr, dtype=float)

        all_y_true.append(y_te_clean)
        all_p_pred.append(prob)
        all_p_baseline.append(baseline_prob)

        try:
            auc = roc_auc_score(y_te_clean, prob)
        except Exception:
            auc = float("nan")

        try:
            brier = brier_score_loss(y_te_clean, prob)
        except Exception:
            brier = float("nan")

        try:
            ap = average_precision_score(y_te_clean, prob)
        except Exception:
            ap = float("nan")

        # Logistic regression metrics
        try:
            auc_log = roc_auc_score(y_te_clean, log_prob)
        except Exception:
            auc_log = float("nan")

        try:
            brier_log = brier_score_loss(y_te_clean, log_prob)
        except Exception:
            brier_log = float("nan")

        try:
            ap_log = average_precision_score(y_te_clean, log_prob)
        except Exception:
            ap_log = float("nan")

        fold_metrics.append({
            "fold": fold_idx,
            "n_train": int(len(y_tr_clean)),
            "n_test": int(len(y_te_clean)),
            "auc": float(auc) if np.isfinite(auc) else float("nan"),
            "brier": float(brier) if np.isfinite(brier) else float("nan"),
            "ap": float(ap) if np.isfinite(ap) else float("nan"),
            "auc_logistic": float(auc_log) if np.isfinite(auc_log) else float("nan"),
            "brier_logistic": float(brier_log) if np.isfinite(brier_log) else float("nan"),
            "ap_logistic": float(ap_log) if np.isfinite(ap_log) else float("nan"),
        })

    if not fold_metrics:
        print("No valid CV folds for robustness diagnostics (insufficient data or degenerate labels).")
        return

    aucs = np.array([m["auc"] for m in fold_metrics], dtype=float)
    briers = np.array([m["brier"] for m in fold_metrics], dtype=float)
    aps = np.array([m["ap"] for m in fold_metrics], dtype=float)

    # Logistic regression metrics per fold
    aucs_log = np.array([m.get("auc_logistic", float("nan")) for m in fold_metrics], dtype=float)
    briers_log = np.array([m.get("brier_logistic", float("nan")) for m in fold_metrics], dtype=float)
    aps_log = np.array([m.get("ap_logistic", float("nan")) for m in fold_metrics], dtype=float)

    mean_auc = float(np.nanmean(aucs))
    std_auc = float(np.nanstd(aucs))

    mean_brier = float(np.nanmean(briers))
    std_brier = float(np.nanstd(briers))

    mean_ap = float(np.nanmean(aps))
    std_ap = float(np.nanstd(aps))

    mean_auc_log = float(np.nanmean(aucs_log)) if np.isfinite(aucs_log).any() else float("nan")
    mean_brier_log = float(np.nanmean(briers_log)) if np.isfinite(briers_log).any() else float("nan")
    mean_ap_log = float(np.nanmean(aps_log)) if np.isfinite(aps_log).any() else float("nan")

    stability_score = 1.0 - (std_auc / (mean_auc + 1e-9)) if np.isfinite(mean_auc) else 0.0

    # Aggregate predictions across folds for advanced diagnostics
    if all_y_true:
        y_all = np.concatenate(all_y_true)
        p_all = np.concatenate(all_p_pred)
        p_base_all = np.concatenate(all_p_baseline)
    else:
        y_all = np.array([])
        p_all = np.array([])
        p_base_all = np.array([])

    pseudo_r2 = float("nan")
    model_snr = float("nan")
    auc_global = float("nan")
    perm_pvalue = float("nan")
    baseline_auc = float("nan")
    baseline_brier = float("nan")
    baseline_ap = float("nan")
    delta_auc = float("nan")
    delta_brier = float("nan")
    delta_ap = float("nan")
    pseudo_r2_ci_low = float("nan")
    pseudo_r2_ci_high = float("nan")
    residual_pattern_strength = float("nan")
    residual_lag1_autocorr = float("nan")

    if y_all.size > 0:
        # Pseudo-R^2 on probabilities: 1 - SSE/SST
        try:
            y_mean = float(np.mean(y_all))
            sse = float(np.sum((y_all - p_all) ** 2))
            sst = float(np.sum((y_all - y_mean) ** 2))
            if sst > 0:
                pseudo_r2 = 1.0 - sse / sst
        except Exception:
            pseudo_r2 = float("nan")

        # Residual diagnostics (pattern strength and autocorrelation)
        try:
            residuals = y_all - p_all
            if residuals.size > 1:
                # Pattern strength: max - min mean residual across probability deciles
                try:
                    quantiles = np.quantile(p_all, np.linspace(0.0, 1.0, 11))
                    bucket_means: list[float] = []
                    for i in range(10):
                        lo, hi = quantiles[i], quantiles[i + 1]
                        mask = (p_all >= lo) & (p_all <= hi)
                        if np.any(mask):
                            bucket_means.append(float(np.mean(residuals[mask])))
                    if bucket_means:
                        residual_pattern_strength = float(max(bucket_means) - min(bucket_means))
                except Exception:
                    residual_pattern_strength = float("nan")

                # Lag-1 autocorrelation of residuals (time-ordered across folds)
                try:
                    r0 = residuals[:-1]
                    r1 = residuals[1:]
                    if r0.size > 1 and np.std(r0) > 0 and np.std(r1) > 0:
                        corr_matrix = np.corrcoef(r0, r1)
                        residual_lag1_autocorr = float(corr_matrix[0, 1])
                except Exception:
                    residual_lag1_autocorr = float("nan")
        except Exception:
            residual_pattern_strength = float("nan")
            residual_lag1_autocorr = float("nan")

        # Model-level SNR: separation of predicted probabilities for pos vs neg labels
        try:
            pos_mask = y_all == 1.0
            neg_mask = y_all == 0.0
            p_pos = p_all[pos_mask]
            p_neg = p_all[neg_mask]
            if len(p_pos) > 1 and len(p_neg) > 1:
                mean_pos = float(np.mean(p_pos))
                mean_neg = float(np.mean(p_neg))
                std_pos = float(np.std(p_pos))
                std_neg = float(np.std(p_neg))
                denom = len(p_pos) + len(p_neg) - 2
                if denom > 0:
                    pooled_var = ((len(p_pos) - 1) * std_pos ** 2 + (len(p_neg) - 1) * std_neg ** 2) / denom
                    pooled_std = float(np.sqrt(pooled_var))
                    if pooled_std > 0 and np.isfinite(pooled_std):
                        model_snr = (mean_pos - mean_neg) / pooled_std
        except Exception:
            model_snr = float("nan")

        # Global AUC across all folds
        try:
            auc_global = float(roc_auc_score(y_all, p_all))
        except Exception:
            auc_global = float("nan")

        # Permutation p-value for global AUC
        if np.isfinite(auc_global) and y_all.size >= 100:
            rng = np.random.default_rng(42)
            perm_aucs: list[float] = []
            for _ in range(200):
                y_perm = rng.permutation(y_all)
                try:
                    perm_auc = roc_auc_score(y_perm, p_all)
                except Exception:
                    continue
                if np.isfinite(perm_auc):
                    perm_aucs.append(float(perm_auc))
            if perm_aucs:
                perm_arr = np.array(perm_aucs, dtype=float)
                perm_pvalue = float((np.sum(perm_arr >= auc_global) + 1) / (len(perm_arr) + 1))

        # Baseline metrics (constant probability) aggregated across folds
        try:
            baseline_auc = float(roc_auc_score(y_all, p_base_all))
        except Exception:
            baseline_auc = float("nan")
        try:
            baseline_brier = float(brier_score_loss(y_all, p_base_all))
        except Exception:
            baseline_brier = float("nan")
        try:
            baseline_ap = float(average_precision_score(y_all, p_base_all))
        except Exception:
            baseline_ap = float("nan")

        if np.isfinite(baseline_auc) and np.isfinite(mean_auc):
            delta_auc = mean_auc - baseline_auc
        if np.isfinite(baseline_brier) and np.isfinite(mean_brier):
            # Lower Brier is better; positive delta means improvement vs baseline
            delta_brier = baseline_brier - mean_brier
        if np.isfinite(baseline_ap) and np.isfinite(mean_ap):
            delta_ap = mean_ap - baseline_ap

        # Bootstrap CI for pseudo-R^2
        if np.isfinite(pseudo_r2) and y_all.size >= 100:
            rng_ci = np.random.default_rng(123)
            boot_stats: list[float] = []
            for _ in range(200):
                idx = rng_ci.integers(0, y_all.size, size=y_all.size)
                y_boot = y_all[idx]
                p_boot = p_all[idx]
                try:
                    y_mean_boot = float(np.mean(y_boot))
                    sse_boot = float(np.sum((y_boot - p_boot) ** 2))
                    sst_boot = float(np.sum((y_boot - y_mean_boot) ** 2))
                    if sst_boot > 0:
                        boot_r2 = 1.0 - sse_boot / sst_boot
                        if np.isfinite(boot_r2):
                            boot_stats.append(float(boot_r2))
                except Exception:
                    continue
            if boot_stats:
                boot_arr = np.array(boot_stats, dtype=float)
                pseudo_r2_ci_low = float(np.percentile(boot_arr, 2.5))
                pseudo_r2_ci_high = float(np.percentile(boot_arr, 97.5))

    # Label-shuffle CV, strict holdout, and single-feature leakage scan
    label_shuffle_metrics = _run_label_shuffle_cv(X, y, cv_splits=cv_splits)
    strict_holdout_metrics = _compute_strict_holdout_metrics(X, y)
    single_feature_leakage = _scan_single_feature_leakage(X, y)

    # NEW: Compute enhanced diagnostics
    regime_aucs = {}
    temporal_auc_data = {"temporal_aucs": [], "temporal_indices": []}
    feature_importance_data = {
        "feature_importance_std": 0.0,
        "importance_concentration": 0.0,
        "top_features": [],
    }
    label_noise_data = {
        "n_confident_predictions": 0,
        "n_mislabeled_candidates": 0,
        "estimated_noise_rate": 0.0,
        "false_neg_rate_confident": 0.0,
        "false_pos_rate_confident": 0.0,
        "mislabeled_indices": [],
    }
    temporal_auc_plot_path = None

    try:
        if y_all.size > 0:
            # Regime-specific AUC breakdown
            regime_aucs = _compute_regime_auc_breakdown(df, p_all, y_all.astype(int))

            # Temporal AUC evolution
            temporal_auc_data = _compute_temporal_auc(y_all, p_all, window_size=min(50, len(y_all) // 5))

            # Plot temporal AUC
            if temporal_auc_data["temporal_aucs"]:
                temporal_auc_plot_path = _plot_temporal_auc(
                    temporal_auc_data["temporal_indices"],
                    temporal_auc_data["temporal_aucs"],
                    symbol,
                    timeframe,
                )
    except Exception as e:
        logger.warning(f"Failed to compute regime AUC breakdown: {e}")

    try:
        # Feature importance stability analysis
        feature_importance_data = _compute_feature_importance_stability(df, cv_splits=cv_splits, n_features_top=20)
    except Exception as e:
        logger.warning(f"Failed to compute feature importance stability: {e}")

    try:
        if y_all.size > 0:
            # Label noise estimation via confident learning
            label_noise_data = _estimate_label_noise_confident_learning(y_all.astype(int), p_all, threshold_confident=0.9)
    except Exception as e:
        logger.warning(f"Failed to estimate label noise: {e}")

    # Model family comparison comment (LightGBM vs LogisticRegression)
    model_family_comment = "N/A"
    if np.isfinite(mean_auc) and np.isfinite(mean_auc_log):
        diff = mean_auc - mean_auc_log
        if diff > 0.02:
            model_family_comment = "Nonlinear (LightGBM) >> linear (Logistic); real nonlinear structure present."
        elif diff < -0.02:
            model_family_comment = "Linear >> nonlinear; tree model may be overfitting or mis-specified."
        else:
            if mean_auc >= 0.6 and mean_auc_log >= 0.6:
                model_family_comment = "All models perform similarly well; problem is stable and well-posed."
            else:
                model_family_comment = "All models perform similarly poorly; target has low intrinsic predictability."

    # Interpretation helpers for robustness
    if mean_auc < 0.55:
        auc_comment = "Mean CV AUC < 0.55 → robust models may still struggle; signal is weak."
    elif mean_auc < 0.6:
        auc_comment = "Mean CV AUC 0.55–0.60 → weak but potentially exploitable signal."
    elif mean_auc < 0.7:
        auc_comment = "Mean CV AUC 0.60–0.70 → moderate predictive power."
    else:
        auc_comment = "Mean CV AUC ≥ 0.70 → strong predictive power for the probe model."

    if stability_score < 0.8:
        stability_comment = "Stability score < 0.8 → performance is quite unstable across time splits."
    elif stability_score < 0.9:
        stability_comment = "Stability score 0.8–0.9 → moderate stability; some variation across folds."
    else:
        stability_comment = "Stability score ≥ 0.9 → highly stable performance across folds."

    if mean_brier > 0.25:
        brier_comment = "Mean Brier > 0.25 → probabilities are poorly calibrated or close to random."
    elif mean_brier > 0.18:
        brier_comment = "Mean Brier 0.18–0.25 → moderate calibration; room for improvement."
    else:
        brier_comment = "Mean Brier ≤ 0.18 → reasonably well-calibrated probabilities."

    # Map robustness into [0, 1] summary score with rating
    def _score_component_mr(value: float, low: float, high: float, invert: bool = False) -> float:
        if value is None or not np.isfinite(value):
            return 0.0
        if invert:
            # Lower is better
            if value >= high:
                return 0.0
            if value <= low:
                return 1.0
            return float((high - value) / (high - low))
        # Higher is better
        if value <= low:
            return 0.0
        if value >= high:
            return 1.0
        return float((value - low) / (high - low))

    auc_score = _score_component_mr(mean_auc, 0.55, 0.70)
    stability_score_norm = _score_component_mr(stability_score, 0.80, 0.90)
    brier_score_norm = _score_component_mr(mean_brier, 0.18, 0.25, invert=True)

    robustness_score_components = [auc_score, stability_score_norm, brier_score_norm]
    robustness_score = float(np.mean(robustness_score_components))

    if robustness_score < 0.4:
        robustness_rating = "Bad"
        robustness_comment = "Probe model is weak or unstable across folds."
    elif robustness_score < 0.7:
        robustness_rating = "Pass"
        robustness_comment = "Moderate robustness; some time variation or calibration issues."
    else:
        robustness_rating = "Great"
        robustness_comment = "Strong, stable probe model with consistent performance."

    def _fmt(value, digits: int = 4) -> str:
        if value is None or not np.isfinite(float(value)):
            return "N/A"
        return f"{float(value):.{digits}f}"

    # Console output
    print("""
=== Model-Robustness Diagnostics (Probe LightGBM) ===
""".strip())
    print(f"Symbol: {symbol} | Exchange: {exchange} | Timeframe: {timeframe}")
    print(f"Folds evaluated: {len(fold_metrics)} (requested: {cv_splits})")
    print()

    print("Per-fold metrics:")
    for m in fold_metrics:
        print(
            f"  Fold {m['fold']}: n_train={m['n_train']}, n_test={m['n_test']}, "
            f"AUC={m['auc']:.4f}, Brier={m['brier']:.4f}, AP={m['ap']:.4f}"
        )

    print()
    print("Summary:")
    print(f"  Mean AUC: {mean_auc:.4f} (std={std_auc:.4f})")
    print(f"  Mean Brier: {mean_brier:.4f} (std={std_brier:.4f})")
    print(f"  Mean AP: {mean_ap:.4f} (std={std_ap:.4f})")
    print(f"  Stability score (1 - std(AUC)/mean(AUC)): {stability_score:.4f}")

    print()
    print("-- Interpretation Hints --")
    print(f"Mean AUC ({mean_auc:.4f}) → {auc_comment}")
    print(f"Stability score ({stability_score:.4f}) → {stability_comment}")
    print(f"Mean Brier ({mean_brier:.4f}) → {brier_comment}")

    print()
    print("-- Advanced Robustness Diagnostics --")
    print(f"Pseudo-R^2 (y vs predicted prob): {_fmt(pseudo_r2)}")
    print(
        f"Pseudo-R^2 95% CI: "
        f"[{_fmt(pseudo_r2_ci_low)}, {_fmt(pseudo_r2_ci_high)}]"
    )
    print(f"Global AUC (all folds combined): {_fmt(auc_global)}")
    print(f"Permutation p-value for global AUC: {_fmt(perm_pvalue)}")
    print(f"Model-level SNR (p_hat pos vs neg): {_fmt(model_snr)}")

    print()
    print("-- Label-Shuffle CV Sanity Check --")
    if label_shuffle_metrics.get("n_folds", 0) > 0:
        print(
            "  Shuffled mean AUC: "
            f"{_fmt(label_shuffle_metrics.get('mean_auc'))} "
            f"(std={_fmt(label_shuffle_metrics.get('std_auc'))}), "
            f"folds={int(label_shuffle_metrics.get('n_folds', 0))}"
        )
    else:
        print("  No valid folds for label-shuffle CV (insufficient data).")

    print()
    print("-- Strict Forward Holdout --")
    if strict_holdout_metrics.get("n_test", 0) > 0:
        print(
            "  Holdout AUC: "
            f"{_fmt(strict_holdout_metrics.get('auc'))}, "
            f"Brier: {_fmt(strict_holdout_metrics.get('brier'))}, "
            f"AP: {_fmt(strict_holdout_metrics.get('ap'))}, "
            f"train={strict_holdout_metrics.get('n_train', 0)}, "
            f"test={strict_holdout_metrics.get('n_test', 0)}"
        )
    else:
        print("  Holdout metrics not available (insufficient data).")

    print()
    print("-- Naive Baseline Comparison (constant probability) --")
    print(f"Baseline AUC: {_fmt(baseline_auc)} | Probe AUC: {_fmt(mean_auc)} | Delta: {_fmt(delta_auc)}")
    print(
        f"Baseline Brier: {_fmt(baseline_brier)} | Probe Brier: {_fmt(mean_brier)} "
        f"| Delta (baseline - probe): {_fmt(delta_brier)}"
    )
    print(f"Baseline AP: {_fmt(baseline_ap)} | Probe AP: {_fmt(mean_ap)} | Delta: {_fmt(delta_ap)}")

    print()
    print("-- Residual Diagnostics --")
    print(
        "Residual pattern strength (max - min mean residual across probability deciles): "
        f"{_fmt(residual_pattern_strength)}"
    )
    print(f"Residual lag-1 autocorrelation: {_fmt(residual_lag1_autocorr)}")

    print()
    print("-- Single-Feature Leakage Scan --")
    if single_feature_leakage.get("max_auc") is not None:
        print(
            "  Max single-feature AUC: "
            f"{_fmt(single_feature_leakage.get('max_auc'))} "
            f"(threshold={_fmt(single_feature_leakage.get('auc_threshold'))})"
        )
    else:
        print("  No valid single-feature AUCs computed.")

    print()
    print("-- Model Family Comparison (LightGBM vs LogisticRegression) --")
    print(
        f"Mean AUC LightGBM: {_fmt(mean_auc)} | "
        f"LogisticRegression: {_fmt(mean_auc_log)}"
    )
    print(f"Model-family comment: {model_family_comment}")

    print()
    print("-- Regime-Specific AUC Breakdown --")
    if regime_aucs:
        for regime_name, auc_val in regime_aucs.items():
            print(f"  {regime_name}: {auc_val:.4f}")
    else:
        print("  (No regime-specific breakdown available)")

    print()
    print("-- Temporal AUC Evolution --")
    if temporal_auc_data["temporal_aucs"]:
        print(f"  Mean rolling AUC: {np.mean(temporal_auc_data['temporal_aucs']):.4f}")
        print(f"  Min rolling AUC: {np.min(temporal_auc_data['temporal_aucs']):.4f}")
        print(f"  Max rolling AUC: {np.max(temporal_auc_data['temporal_aucs']):.4f}")
        if temporal_auc_plot_path:
            print(f"  Plot saved: {temporal_auc_plot_path}")
    else:
        print("  (Insufficient data for temporal AUC analysis)")

    print()
    print("-- Feature Importance Stability Analysis --")
    if feature_importance_data.get("top_features"):
        print(f"  Feature importance std: {feature_importance_data['feature_importance_std']:.4f}")
        print(f"  Importance concentration: {feature_importance_data['importance_concentration']:.1%}")
        print("  Top 5 features:")
        for feat in feature_importance_data["top_features"][:5]:
            feat_idx = feat.get("feature_idx", "?")
            mean_imp = feat.get("mean_importance", 0.0)
            std_imp = feat.get("std_importance", 0.0)
            print(f"    Feature {feat_idx}: mean={mean_imp:.4f}, std={std_imp:.4f}")
    else:
        print("  (No feature importance data available)")

    print()
    print("-- Label Noise Estimation (Confident Learning) --")
    if label_noise_data.get("n_confident_predictions", 0) > 0:
        print(f"  N confident predictions: {label_noise_data['n_confident_predictions']}")
        print(f"  N mislabeled candidates: {label_noise_data['n_mislabeled_candidates']}")
        print(f"  Estimated noise rate: {label_noise_data['estimated_noise_rate']:.1%}")
        print(f"  False neg rate (confident): {label_noise_data['false_neg_rate_confident']:.1%}")
        print(f"  False pos rate (confident): {label_noise_data['false_pos_rate_confident']:.1%}")
    else:
        print("  (Insufficient data for label noise analysis)")

    # Export payload
    payload = {
        "section": "model_robustness",
        "cv_splits": int(cv_splits),
        "fold_metrics": fold_metrics,
        "summary": {
            "mean_auc": float(mean_auc),
            "std_auc": float(std_auc),
            "mean_brier": float(mean_brier),
            "std_brier": float(std_brier),
            "mean_ap": float(mean_ap),
            "std_ap": float(std_ap),
            "stability_score": float(stability_score),
            "n_folds": int(len(fold_metrics)),
        },
        "advanced": {
            "global_auc": float(auc_global) if np.isfinite(auc_global) else None,
            "pseudo_r2": float(pseudo_r2) if np.isfinite(pseudo_r2) else None,
            "pseudo_r2_ci_low": float(pseudo_r2_ci_low) if np.isfinite(pseudo_r2_ci_low) else None,
            "pseudo_r2_ci_high": float(pseudo_r2_ci_high) if np.isfinite(pseudo_r2_ci_high) else None,
            "model_snr": float(model_snr) if np.isfinite(model_snr) else None,
            "perm_pvalue_auc": float(perm_pvalue) if np.isfinite(perm_pvalue) else None,
            "residual_pattern_strength": float(residual_pattern_strength)
            if np.isfinite(residual_pattern_strength)
            else None,
            "residual_lag1_autocorr": float(residual_lag1_autocorr)
            if np.isfinite(residual_lag1_autocorr)
            else None,
            "model_family": {
                "mean_auc_lightgbm": float(mean_auc) if np.isfinite(mean_auc) else None,
                "mean_auc_logistic": float(mean_auc_log) if np.isfinite(mean_auc_log) else None,
                "comment": model_family_comment,
            },
            "baseline": {
                "auc": float(baseline_auc) if np.isfinite(baseline_auc) else None,
                "brier": float(baseline_brier) if np.isfinite(baseline_brier) else None,
                "ap": float(baseline_ap) if np.isfinite(baseline_ap) else None,
                "delta_auc": float(delta_auc) if np.isfinite(delta_auc) else None,
                "delta_brier": float(delta_brier) if np.isfinite(delta_brier) else None,
                "delta_ap": float(delta_ap) if np.isfinite(delta_ap) else None,
            },
            "label_shuffle_cv": label_shuffle_metrics,
            "strict_holdout": strict_holdout_metrics,
            "single_feature_leakage": single_feature_leakage,
        },
        "regime_analysis": {
            "regime_aucs": regime_aucs,
            "temporal_auc": temporal_auc_data,
            "temporal_auc_plot": str(temporal_auc_plot_path) if temporal_auc_plot_path else None,
        },
        "feature_importance_stability": feature_importance_data,
        "label_noise_estimation": label_noise_data,
        "summary_score": {
            "score": float(robustness_score),
            "rating": robustness_rating,
            "comment": robustness_comment,
        },
    }

    md_lines = [
        "# Model-Robustness Diagnostics (Probe LightGBM)",
        "",
        f"**Symbol**: {symbol}",
        f"**Exchange**: {exchange}",
        f"**Timeframe**: {timeframe}",
        "",
        "## Fold Metrics",
    ]

    for m in fold_metrics:
        md_lines.append(
            f"- Fold {m['fold']}: n_train={m['n_train']}, n_test={m['n_test']}, "
            f"AUC={m['auc']:.4f}, Brier={m['brier']:.4f}, AP={m['ap']:.4f}"
        )

    md_lines.extend(
        [
            "",
            "## Summary",
            f"- Mean AUC: {mean_auc:.4f} (std={std_auc:.4f})",
            f"- Mean Brier: {mean_brier:.4f} (std={std_brier:.4f})",
            f"- Mean AP: {mean_ap:.4f} (std={std_ap:.4f})",
            f"- Stability score (1 - std(AUC)/mean(AUC)): {stability_score:.4f}",
            "",
            "## Interpretation Hints",
            f"- Mean AUC ({mean_auc:.4f}): {auc_comment}",
            f"- Stability score ({stability_score:.4f}): {stability_comment}",
            f"- Mean Brier ({mean_brier:.4f}): {brier_comment}",
            "",
            "## Advanced Robustness Diagnostics",
            f"- Global AUC (all folds combined): {_fmt(auc_global)}",
            f"- Pseudo-R^2 (y vs predicted prob): {_fmt(pseudo_r2)}",
            f"- Pseudo-R^2 95% CI: [{_fmt(pseudo_r2_ci_low)}, {_fmt(pseudo_r2_ci_high)}]",
            f"- Permutation p-value for global AUC: {_fmt(perm_pvalue)}",
            f"- Model-level SNR (p_hat pos vs neg): {_fmt(model_snr)}",
            "",
            "## Label-Shuffle CV Sanity Check",
            f"- Shuffled mean AUC: {_fmt(label_shuffle_metrics.get('mean_auc'))}",
            f"- Shuffled std AUC: {_fmt(label_shuffle_metrics.get('std_auc'))}",
            f"- Shuffled folds: {int(label_shuffle_metrics.get('n_folds', 0))}",
            "",
            "## Strict Forward Holdout",
            f"- Holdout AUC: {_fmt(strict_holdout_metrics.get('auc'))}",
            f"- Holdout Brier: {_fmt(strict_holdout_metrics.get('brier'))}",
            f"- Holdout AP: {_fmt(strict_holdout_metrics.get('ap'))}",
            f"- Holdout train / test: {strict_holdout_metrics.get('n_train', 0)} / {strict_holdout_metrics.get('n_test', 0)}",
            "",
            "## Single-Feature Leakage Scan",
            f"- Max single-feature AUC: {_fmt(single_feature_leakage.get('max_auc'))}",
            f"- AUC threshold for suspicion: {_fmt(single_feature_leakage.get('auc_threshold'))}",
            "",
            "## Naive Baseline Comparison (constant probability)",
            f"- Baseline AUC: {_fmt(baseline_auc)} | Probe AUC: {_fmt(mean_auc)} | Delta: {_fmt(delta_auc)}",
            f"- Baseline Brier: {_fmt(baseline_brier)} | Probe Brier: {_fmt(mean_brier)} | Delta (baseline - probe): {_fmt(delta_brier)}",
            f"- Baseline AP: {_fmt(baseline_ap)} | Probe AP: {_fmt(mean_ap)} | Delta: {_fmt(delta_ap)}",
            "",
            "## Residual Diagnostics",
            "- Residual pattern strength (max - min mean residual across probability deciles): "
            f"{_fmt(residual_pattern_strength)}",
            f"- Residual lag-1 autocorrelation: {_fmt(residual_lag1_autocorr)}",
            "",
            "## Model Family Comparison (LightGBM vs LogisticRegression)",
            f"- Mean AUC LightGBM: {_fmt(mean_auc)} | LogisticRegression: {_fmt(mean_auc_log)}",
            f"- Comment: {model_family_comment}",
            "",
            "## Regime-Specific AUC Breakdown",
        ]
    )

    # Add regime AUC data
    if regime_aucs:
        for regime_name, auc_val in regime_aucs.items():
            md_lines.append(f"- {regime_name}: {auc_val:.4f}")
    else:
        md_lines.append("- No regime-specific breakdown available (volatility or HMM regimes not found)")

    md_lines.extend(
        [
            "",
            "## Temporal AUC Evolution",
        ]
    )

    # Add temporal AUC data
    if temporal_auc_data["temporal_aucs"]:
        md_lines.append(f"- Mean rolling AUC: {np.mean(temporal_auc_data['temporal_aucs']):.4f}")
        md_lines.append(f"- Min rolling AUC: {np.min(temporal_auc_data['temporal_aucs']):.4f}")
        md_lines.append(f"- Max rolling AUC: {np.max(temporal_auc_data['temporal_aucs']):.4f}")
        md_lines.append(f"- AUC at start: {temporal_auc_data['temporal_aucs'][0]:.4f}")
        md_lines.append(f"- AUC at end: {temporal_auc_data['temporal_aucs'][-1]:.4f}")
        if temporal_auc_plot_path:
            md_lines.append(f"- Plot saved: `{temporal_auc_plot_path}`")
        md_lines.append("**Interpretation**: If rolling AUC declines over time, model performance degrades on recent data.")
    else:
        md_lines.append("- Insufficient data for temporal AUC analysis")

    md_lines.extend(
        [
            "",
            "## Feature Importance Stability Analysis",
        ]
    )

    # Add feature importance stability
    if feature_importance_data.get("top_features"):
        md_lines.append(f"- Feature importance std (across CV folds): {feature_importance_data['feature_importance_std']:.4f}")
        md_lines.append(f"- Importance concentration (top 20 features): {feature_importance_data['importance_concentration']:.3%}")
        md_lines.append("- Top features (with stability):")
        for feat in feature_importance_data["top_features"][:10]:
            feat_idx = feat.get("feature_idx", "?")
            mean_imp = feat.get("mean_importance", 0.0)
            std_imp = feat.get("std_importance", 0.0)
            md_lines.append(f"  - Feature {feat_idx}: mean={mean_imp:.4f}, std={std_imp:.4f}")
        md_lines.append("**Interpretation**: High std_importance across folds suggests unstable features (overfitting risk).")
    else:
        md_lines.append("- No feature importance data available")

    md_lines.extend(
        [
            "",
            "## Label Noise Estimation (Confident Learning)",
        ]
    )

    # Add label noise data
    if label_noise_data.get("n_confident_predictions", 0) > 0:
        md_lines.append(f"- N confident predictions (confidence ≥ 0.9): {label_noise_data['n_confident_predictions']}")
        md_lines.append(f"- N mislabeled candidates (confident but wrong): {label_noise_data['n_mislabeled_candidates']}")
        md_lines.append(f"- Estimated label noise rate: {label_noise_data['estimated_noise_rate']:.3%}")
        md_lines.append(f"- False negative rate (confident): {label_noise_data['false_neg_rate_confident']:.3%}")
        md_lines.append(f"- False positive rate (confident): {label_noise_data['false_pos_rate_confident']:.3%}")
        md_lines.append("**Interpretation**: High noise rate (>5%) suggests labels may be mislabeled; consider tightening TPSL geometry.")
    else:
        md_lines.append("- Insufficient data for label noise analysis")

    md_lines.extend(
        [
            "",
            "## Overall Model-Robustness Score",
            f"- Score (0-1): {robustness_score:.3f}",
            f"- Rating: {robustness_rating}",
            f"- Summary: {robustness_comment}",
        ]
    )

    json_path, md_path = _export_report(
        prefix="snr_model_robustness",
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        direction=direction,
        model=model,
        payload=payload,
        markdown_lines=md_lines,
    )

    print(f"\nReports saved to: {json_path} and {md_path}")


# --------------------------------------------------------------------------------------
# Trading simulation diagnostics
# --------------------------------------------------------------------------------------


def run_trading_simulation(
    symbol: str,
    exchange: str,
    timeframe: str,
    direction: str = "long",
    model: str = "analyst",
    prob_column: str = "meta_probability",
    prob_thresholds: Optional[List[float]] = None,
    cv_splits: int = 5,
) -> None:
    """Run trading simulation diagnostics with calibration and threshold analysis.

    Computes:
    - Model calibration (Brier score, calibration curves)
    - Trading metrics at different probability thresholds (0.60, 0.65, 0.70, 0.75):
      - Trades per day
      - PnL per day (average, percentage)
      - Simplified equity curve simulation
      - Consecutive loss metric
      - Win-rate stability
    """
    if prob_thresholds is None:
        prob_thresholds = [0.60, 0.65, 0.70, 0.75]

    df = _load_labeled_data(symbol, exchange, timeframe, direction=direction, model=model)
    X, y = _build_feature_matrix_from_labeled(df, direction=direction)

    if prob_column not in df.columns:
        # Auto-fallback: prefer ensemble / bagged outputs if meta_probability is absent.
        fallback_cols = [
            "meta_probability",
            "meta_probability_ensemble",
            "meta_probability_lgbm_bag_mean",
            "meta_probability_lgbm_bag_lower",
        ]
        chosen = None
        for c in fallback_cols:
            if c in df.columns:
                chosen = c
                break
        if chosen is None:
            raise ValueError(
                f"labeled_data must contain '{prob_column}' (or one of {fallback_cols}) column for trading simulation"
            )
        logger.warning(
            "prob_column '%s' not found; falling back to '%s' for trading simulation",
            prob_column,
            chosen,
        )
        prob_column = chosen
    if "realized_return" not in df.columns:
        raise ValueError("labeled_data must contain 'realized_return' column for trading simulation")

    # Get date range for trades per day calculation
    if hasattr(df.index, 'to_pydatetime'):
        dates = pd.to_datetime(df.index)
    else:
        dates = pd.to_datetime(df.index)

    date_range_days = (dates.max() - dates.min()).days
    if date_range_days <= 0:
        date_range_days = 1

    meta_prob = df[prob_column].astype(float)
    realized_returns = df["realized_return"].astype(float)
    
    # Prefer directional binary labels
    if direction == "long" and "binary_label_long" in df.columns:
        binary_labels = df["binary_label_long"]
    elif direction == "short" and "binary_label_short" in df.columns:
        binary_labels = df["binary_label_short"]
    elif "binary_label_long" in df.columns:
        binary_labels = df["binary_label_long"]
    elif "binary_label_short" in df.columns:
        binary_labels = df["binary_label_short"]
    elif "binary_label" in df.columns:
        binary_labels = df["binary_label"]
    else:
        binary_labels = None

    # Valid mask: events with both meta_probability and realized_return
    valid_mask = meta_prob.notna() & realized_returns.notna()
    if binary_labels is not None:
        valid_mask = valid_mask & binary_labels.notna()

    n_valid = int(valid_mask.sum())
    if n_valid < 50:
        print(f"Insufficient valid samples ({n_valid}) for trading simulation diagnostics.")
        return

    prob_valid = meta_prob[valid_mask].values
    returns_valid = realized_returns[valid_mask].values
    labels_valid = binary_labels[valid_mask].values if binary_labels is not None else None

    # -------------------------------------------------------------------------
    # 1. Model Calibration Diagnostics
    # -------------------------------------------------------------------------

    # Train a probe model and get predictions for calibration analysis
    y_array = y.values.astype(float)
    X_array = X.values.astype(float)

    tscv = TimeSeriesSplit(n_splits=cv_splits)

    all_y_true = []
    all_p_pred = []

    for fold_idx, (tr_idx, te_idx) in enumerate(tscv.split(X_array), start=1):
        X_tr, X_te = X_array[tr_idx], X_array[te_idx]
        y_tr, y_te = y_array[tr_idx], y_array[te_idx]

        # Require both classes in train and test
        if len(np.unique(y_tr[~np.isnan(y_tr)])) < 2 or len(np.unique(y_te[~np.isnan(y_te)])) < 2:
            continue

        mask_tr = ~np.isnan(y_tr)
        y_tr_clean = y_tr[mask_tr]
        X_tr_clean = X_tr[mask_tr]
        mask_te = ~np.isnan(y_te)
        y_te_clean = y_te[mask_te]
        X_te_clean = X_te[mask_te]

        if len(y_tr_clean) < 50 or len(y_te_clean) < 20:
            continue

        clf = lgb.LGBMClassifier(
            boosting_type="gbdt",
            objective="binary",
            max_depth=3,
            n_estimators=50,
            learning_rate=0.1,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_samples=20,
            n_jobs=-1,
            verbose=-1,
            random_state=42,
        )

        clf.fit(X_tr_clean, y_tr_clean)
        prob = clf.predict_proba(X_te_clean)[:, 1]

        all_y_true.append(y_te_clean)
        all_p_pred.append(prob)

    # Aggregate predictions for calibration
    calibration_metrics = {}
    calibration_curve_data = {}
    iso_calibrator = None  # Will hold fitted isotonic calibrator

    if all_y_true:
        y_all = np.concatenate(all_y_true)
        p_all = np.concatenate(all_p_pred)

        # Brier score (uncalibrated)
        try:
            brier = float(brier_score_loss(y_all, p_all))
        except Exception:
            brier = float("nan")

        # -----------------------------------------------------------------
        # Fit isotonic calibration on OOS predictions
        # -----------------------------------------------------------------
        try:
            from sklearn.isotonic import IsotonicRegression
            iso_calibrator = IsotonicRegression(out_of_bounds='clip')
            iso_calibrator.fit(p_all, y_all)
            
            # Compute calibrated Brier for comparison
            p_calibrated = iso_calibrator.predict(p_all)
            brier_calibrated = float(brier_score_loss(y_all, p_calibrated))
            try:
                abs_improvement = float(brier - brier_calibrated)
                rel_improvement_pct = float(100.0 * abs_improvement / max(float(brier), 1e-12))
                abs_improvement_pct_points = float(100.0 * abs_improvement)
            except Exception:
                rel_improvement_pct = float("nan")
                abs_improvement_pct_points = float("nan")
            logger.info(
                f"Isotonic calibration fitted: Brier {brier:.4f} -> {brier_calibrated:.4f} "
                f"(improvement: {abs_improvement_pct_points:.2f} pct-pts, {rel_improvement_pct:.2f}% relative)"
            )
        except Exception as e:
            logger.warning(f"Isotonic calibration failed: {e}")
            iso_calibrator = None
            brier_calibrated = float("nan")

        # Calibration curve (reliability diagram data)
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_all, p_all, n_bins=10, strategy='uniform'
            )
            calibration_curve_data = {
                "fraction_of_positives": [float(x) for x in fraction_of_positives],
                "mean_predicted_value": [float(x) for x in mean_predicted_value],
            }

            # Expected Calibration Error (ECE)
            bin_counts = np.histogram(p_all, bins=10, range=(0, 1))[0]
            ece = 0.0
            for i in range(len(fraction_of_positives)):
                if i < len(bin_counts) and bin_counts[i] > 0:
                    ece += (bin_counts[i] / len(p_all)) * abs(fraction_of_positives[i] - mean_predicted_value[i])

            # Maximum Calibration Error (MCE)
            mce = float(np.max(np.abs(fraction_of_positives - mean_predicted_value)))

        except Exception:
            ece = float("nan")
            mce = float("nan")
            brier_calibrated = float("nan")

        calibration_metrics = {
            "brier_score": brier,
            "brier_score_calibrated": brier_calibrated if 'brier_calibrated' in dir() else float("nan"),
            "expected_calibration_error": float(ece) if np.isfinite(ece) else float("nan"),
            "max_calibration_error": float(mce) if np.isfinite(mce) else float("nan"),
            "n_samples": int(len(y_all)),
            "isotonic_calibration_applied": iso_calibrator is not None,
        }
    else:
        calibration_metrics = {
            "brier_score": float("nan"),
            "brier_score_calibrated": float("nan"),
            "expected_calibration_error": float("nan"),
            "max_calibration_error": float("nan"),
            "n_samples": 0,
            "isotonic_calibration_applied": False,
        }

    # -------------------------------------------------------------------------
    # 2. Trading Simulation at Different Probability Thresholds
    # -------------------------------------------------------------------------
    
    # Apply isotonic calibration to trading probabilities if available
    if iso_calibrator is not None:
        prob_trading = iso_calibrator.predict(prob_valid)
        logger.info(f"Applied isotonic calibration to {len(prob_trading)} trading probabilities")
    else:
        prob_trading = prob_valid
        logger.info("Using uncalibrated probabilities for trading simulation")

    threshold_results = {}
    best_gate_info: Optional[dict] = None
    regime_threshold_results: dict = {}
    regime_gating_cfg: dict = {}

    for threshold in prob_thresholds:
        # Filter events passing the threshold (using calibrated probabilities)
        trade_mask = prob_trading >= threshold
        n_trades = int(trade_mask.sum())

        if n_trades < 10:
            threshold_results[f"threshold_{threshold:.2f}"] = {
                "threshold": float(threshold),
                "n_trades": n_trades,
                "trades_per_day": 0.0,
                "mean_return_per_trade": float("nan"),
                "pnl_per_day_pct": float("nan"),
                "win_rate": float("nan"),
                "max_consecutive_losses": 0,
                "avg_consecutive_losses": float("nan"),
                "win_rate_stability": float("nan"),
                "sharpe_ratio": float("nan"),
                "max_drawdown": float("nan"),
                "final_equity": float("nan"),
                "insufficient_data": True,
            }
            continue

        trade_returns = returns_valid[trade_mask]
        trade_labels = labels_valid[trade_mask] if labels_valid is not None else None

        # Trades per day
        trades_per_day = n_trades / date_range_days

        # Mean return per trade
        mean_return = float(np.mean(trade_returns))
        std_return = float(np.std(trade_returns)) if n_trades > 1 else 0.0

        # PnL per day (percentage)
        total_pnl = float(np.sum(trade_returns))
        pnl_per_day = total_pnl / date_range_days

        # Win rate
        if trade_labels is not None:
            win_rate = float(np.mean(trade_labels == 1.0))
        else:
            win_rate = float(np.mean(trade_returns > 0))

        # Sharpe ratio (trade-level)
        if std_return > 0:
            sharpe = mean_return / std_return * np.sqrt(n_trades)
        else:
            sharpe = float("nan")

        # Equity curve simulation
        equity_curve = np.cumprod(1 + trade_returns)
        final_equity = float(equity_curve[-1]) if len(equity_curve) > 0 else 1.0

        # Max drawdown
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max
        max_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

        # Consecutive losses metric
        if trade_labels is not None:
            losses = (trade_labels == 0.0).astype(int)
        else:
            losses = (trade_returns <= 0).astype(int)

        consecutive_losses = []
        current_streak = 0
        for loss in losses:
            if loss:
                current_streak += 1
            else:
                if current_streak > 0:
                    consecutive_losses.append(current_streak)
                current_streak = 0
        if current_streak > 0:
            consecutive_losses.append(current_streak)

        max_consecutive_losses = max(consecutive_losses) if consecutive_losses else 0
        avg_consecutive_losses = float(np.mean(consecutive_losses)) if consecutive_losses else 0.0

        # Win-rate stability (rolling window standard deviation)
        if n_trades >= 20:
            window_size = min(50, n_trades // 4)
            if trade_labels is not None:
                wins = (trade_labels == 1.0).astype(float)
            else:
                wins = (trade_returns > 0).astype(float)

            rolling_win_rates = []
            for i in range(0, n_trades - window_size + 1, window_size // 2):
                window_wins = wins[i:i + window_size]
                rolling_win_rates.append(float(np.mean(window_wins)))

            if len(rolling_win_rates) > 1:
                win_rate_stability = 1.0 - float(np.std(rolling_win_rates))
            else:
                win_rate_stability = float("nan")
        else:
            win_rate_stability = float("nan")

        threshold_results[f"threshold_{threshold:.2f}"] = {
            "threshold": float(threshold),
            "n_trades": n_trades,
            "trades_per_day": float(trades_per_day),
            "mean_return_per_trade": float(mean_return),
            "std_return_per_trade": float(std_return),
            "pnl_per_day_pct": float(pnl_per_day * 100),
            "win_rate": float(win_rate),
            "sharpe_ratio": float(sharpe) if np.isfinite(sharpe) else float("nan"),
            "max_drawdown": float(max_drawdown * 100),
            "final_equity": float(final_equity),
            "max_consecutive_losses": int(max_consecutive_losses),
            "avg_consecutive_losses": float(avg_consecutive_losses),
            "win_rate_stability": float(win_rate_stability) if np.isfinite(win_rate_stability) else float("nan"),
            "insufficient_data": False,
        }

    regime_masks: dict = {}
    try:
        if "hmm_regime_label_1h" in df.columns:
            regimes_series = df.loc[valid_mask, "hmm_regime_label_1h"]
            unique_regs = regimes_series.dropna().unique()
            for reg_val in unique_regs:
                mask_reg = (regimes_series == reg_val).to_numpy()
                if mask_reg.sum() >= 20:
                    regime_masks[f"hmm_{reg_val}"] = mask_reg
        elif "volatility_1d" in df.columns:
            vol_series = df.loc[valid_mask, "volatility_1d"].astype(float)
            vol_clean = vol_series.dropna()
            if len(vol_clean) >= 60:
                low_thr = float(vol_clean.quantile(1.0 / 3.0))
                high_thr = float(vol_clean.quantile(2.0 / 3.0))
                vol_arr = vol_series.to_numpy()
                low_mask = vol_arr < low_thr
                mid_mask = (vol_arr >= low_thr) & (vol_arr < high_thr)
                high_mask = vol_arr >= high_thr
                if low_mask.sum() >= 20:
                    regime_masks["vol_low"] = low_mask
                if mid_mask.sum() >= 20:
                    regime_masks["vol_mid"] = mid_mask
                if high_mask.sum() >= 20:
                    regime_masks["vol_high"] = high_mask
    except Exception:
        regime_masks = {}

    if regime_masks:
        dates_valid = dates[valid_mask.values]
        for regime_name, regime_mask in regime_masks.items():
            try:
                regime_results: dict = {}
                regime_dates = dates_valid[regime_mask]
                if regime_dates.size > 0:
                    regime_days = (regime_dates.max() - regime_dates.min()).days
                    if regime_days <= 0:
                        regime_days = 1
                else:
                    regime_days = date_range_days

                best_reg_score = float("-inf")

                for threshold in prob_thresholds:
                    key = f"threshold_{threshold:.2f}"
                    trade_mask_reg = (prob_valid >= threshold) & regime_mask
                    n_trades_reg = int(trade_mask_reg.sum())

                    if n_trades_reg < 10:
                        regime_results[key] = {
                            "threshold": float(threshold),
                            "n_trades": n_trades_reg,
                            "trades_per_day": 0.0,
                            "mean_return_per_trade": float("nan"),
                            "std_return_per_trade": float("nan"),
                            "pnl_per_day_pct": float("nan"),
                            "win_rate": float("nan"),
                            "sharpe_ratio": float("nan"),
                            "max_drawdown": float("nan"),
                            "final_equity": float("nan"),
                            "max_consecutive_losses": 0,
                            "avg_consecutive_losses": float("nan"),
                            "win_rate_stability": float("nan"),
                            "insufficient_data": True,
                        }
                        continue

                    trade_returns_reg = returns_valid[trade_mask_reg]
                    if labels_valid is not None:
                        trade_labels_reg = labels_valid[trade_mask_reg]
                    else:
                        trade_labels_reg = None

                    trades_per_day_reg = n_trades_reg / max(regime_days, 1)
                    mean_return_reg = float(np.mean(trade_returns_reg))
                    std_return_reg = float(np.std(trade_returns_reg)) if n_trades_reg > 1 else 0.0
                    total_pnl_reg = float(np.sum(trade_returns_reg))
                    pnl_per_day_reg = total_pnl_reg / max(regime_days, 1)

                    if trade_labels_reg is not None:
                        win_rate_reg = float(np.mean(trade_labels_reg == 1.0))
                    else:
                        win_rate_reg = float(np.mean(trade_returns_reg > 0))

                    if std_return_reg > 0:
                        sharpe_reg = mean_return_reg / std_return_reg * np.sqrt(n_trades_reg)
                    else:
                        sharpe_reg = float("nan")

                    equity_reg = np.cumprod(1 + trade_returns_reg)
                    final_equity_reg = float(equity_reg[-1]) if len(equity_reg) > 0 else 1.0
                    running_max_reg = np.maximum.accumulate(equity_reg)
                    drawdowns_reg = (equity_reg - running_max_reg) / running_max_reg
                    max_drawdown_reg = float(np.min(drawdowns_reg)) if len(drawdowns_reg) > 0 else 0.0

                    if trade_labels_reg is not None:
                        losses_reg = (trade_labels_reg == 0.0).astype(int)
                    else:
                        losses_reg = (trade_returns_reg <= 0).astype(int)

                    consecutive_losses_reg = []
                    current_streak_reg = 0
                    for loss in losses_reg:
                        if loss:
                            current_streak_reg += 1
                        else:
                            if current_streak_reg > 0:
                                consecutive_losses_reg.append(current_streak_reg)
                            current_streak_reg = 0
                    if current_streak_reg > 0:
                        consecutive_losses_reg.append(current_streak_reg)

                    max_consecutive_losses_reg = max(consecutive_losses_reg) if consecutive_losses_reg else 0
                    avg_consecutive_losses_reg = float(np.mean(consecutive_losses_reg)) if consecutive_losses_reg else 0.0

                    if n_trades_reg >= 20:
                        window_size_reg = min(50, n_trades_reg // 4)
                        if trade_labels_reg is not None:
                            wins_reg = (trade_labels_reg == 1.0).astype(float)
                        else:
                            wins_reg = (trade_returns_reg > 0).astype(float)
                        rolling_win_rates_reg = []
                        for i in range(0, n_trades_reg - window_size_reg + 1, max(1, window_size_reg // 2)):
                            window_wins_reg = wins_reg[i:i + window_size_reg]
                            rolling_win_rates_reg.append(float(np.mean(window_wins_reg)))
                        if len(rolling_win_rates_reg) > 1:
                            win_rate_stability_reg = 1.0 - float(np.std(rolling_win_rates_reg))
                        else:
                            win_rate_stability_reg = float("nan")
                    else:
                        win_rate_stability_reg = float("nan")

                    regime_results[key] = {
                        "threshold": float(threshold),
                        "n_trades": n_trades_reg,
                        "trades_per_day": float(trades_per_day_reg),
                        "mean_return_per_trade": float(mean_return_reg),
                        "std_return_per_trade": float(std_return_reg),
                        "pnl_per_day_pct": float(pnl_per_day_reg * 100),
                        "win_rate": float(win_rate_reg),
                        "sharpe_ratio": float(sharpe_reg) if np.isfinite(sharpe_reg) else float("nan"),
                        "max_drawdown": float(max_drawdown_reg * 100),
                        "final_equity": float(final_equity_reg),
                        "max_consecutive_losses": int(max_consecutive_losses_reg),
                        "avg_consecutive_losses": float(avg_consecutive_losses_reg),
                        "win_rate_stability": float(win_rate_stability_reg) if np.isfinite(win_rate_stability_reg) else float("nan"),
                        "insufficient_data": False,
                    }

                    score_reg = 0.0
                    if np.isfinite(sharpe_reg):
                        score_reg = float(sharpe_reg * np.sqrt(max(n_trades_reg, 1)))

                    if score_reg > best_reg_score and pnl_per_day_reg > 0.0:
                        best_reg_score = score_reg
                        regime_gating_cfg[regime_name] = {
                            "prob_threshold": float(threshold),
                            "expected_return_threshold": 0.0,
                            "mean_return": float(mean_return_reg),
                            "sharpe": float(sharpe_reg) if np.isfinite(sharpe_reg) else 0.0,
                            "n_trades": int(n_trades_reg),
                            "trades_per_day": float(trades_per_day_reg),
                            "score": float(score_reg),
                        }

                if regime_results:
                    regime_threshold_results[regime_name] = regime_results
            except Exception:
                continue

    # -------------------------------------------------------------------------
    # 2B. Select best gate by PnL/day and update meta_gating_config
    # -------------------------------------------------------------------------

    # Select the threshold with the highest positive PnL per day.
    best_key: Optional[str] = None
    best_score: float = float("-inf")
    for key, res in threshold_results.items():
        if res.get("insufficient_data", False):
            continue
        score = res.get("pnl_per_day_pct", float("nan"))
        if not np.isfinite(score):
            continue
        if score > best_score:
            best_score = float(score)
            best_key = key

    if best_key is not None and best_score > 0.0:
        res = threshold_results[best_key]
        try:
            best_gate_info = {
                "threshold": float(res["threshold"]),
                "n_trades": int(res["n_trades"]),
                "trades_per_day": float(res["trades_per_day"]),
                "mean_return_per_trade": float(res["mean_return_per_trade"]),
                "pnl_per_day_pct": float(res["pnl_per_day_pct"]),
                "win_rate": float(res["win_rate"]),
                "sharpe_ratio": float(res["sharpe_ratio"]),
                "max_drawdown": float(res["max_drawdown"]),
                "final_equity": float(res["final_equity"]),
            }

            va_dir = Path("versioned_artifacts") / f"{symbol}_{exchange}_{timeframe}_{direction}_analyst"
            gating_path = va_dir / "meta_gating_config.json"
            try:
                va_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

            if gating_path.exists():
                try:
                    with open(gating_path, "r") as f:
                        meta_gating_config = json.load(f)
                except Exception:
                    meta_gating_config = {}
            else:
                meta_gating_config = {}

            if not isinstance(meta_gating_config, dict):
                meta_gating_config = {}

            meta_gating_config.setdefault("symbol", symbol)
            meta_gating_config.setdefault("exchange", exchange)
            meta_gating_config.setdefault("timeframe", timeframe)
            meta_gating_config.setdefault("direction", direction)
            meta_gating_config.setdefault("model_family", f"{model}_meta")

            mg = meta_gating_config.get("meta_gating")
            if not isinstance(mg, dict):
                mg = {}

            mg["version"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

            entry = mg.get("entry")
            if not isinstance(entry, dict):
                entry = {}
            entry.update(
                {
                    "prob_threshold": float(best_gate_info["threshold"]),
                    "use_expected_return": False,
                    "expected_return_threshold": 0.0,
                    "expected_return_unit": "fraction",
                    "min_trades": int(best_gate_info["n_trades"]),
                }
            )
            mg["entry"] = entry

            existing_bt = mg.get("backtest_metrics")
            if not isinstance(existing_bt, dict):
                existing_bt = {}
            backtest_metrics = {
                "mean_return_gated": float(best_gate_info["mean_return_per_trade"]),
                "sharpe_gated": float(best_gate_info["sharpe_ratio"]),
                "trades_gated": int(best_gate_info["n_trades"]),
                "trades_per_day_gated": float(best_gate_info["trades_per_day"]),
                "pnl_per_day_pct": float(best_gate_info["pnl_per_day_pct"]),
                "max_drawdown_pct": float(best_gate_info["max_drawdown"]),
                "final_equity": float(best_gate_info["final_equity"]),
            }
            if "auc_oof" in existing_bt:
                backtest_metrics["auc_oof"] = existing_bt["auc_oof"]
            mg["backtest_metrics"] = backtest_metrics

            mg.setdefault("calibration", {})
            mg.setdefault("triple_barrier", {})
            regime_specific = mg.get("regime_specific")
            if not isinstance(regime_specific, dict):
                regime_specific = {}
            if regime_gating_cfg:
                regime_specific.update(regime_gating_cfg)
            mg["regime_specific"] = regime_specific

            meta_gating_config["meta_gating"] = mg

            try:
                with open(gating_path, "w") as f:
                    json.dump(meta_gating_config, f, indent=2)
                logger.info(
                    "Updated meta_gating_config at %s using SNR trading-simulation best threshold %.2f "
                    "(PnL/day=%.4f%%, trades/day=%.3f)",
                    gating_path,
                    best_gate_info["threshold"],
                    best_gate_info["pnl_per_day_pct"],
                    best_gate_info["trades_per_day"],
                )
            except Exception as e:
                logger.warning(
                    "Failed to write meta_gating_config at %s from trading-simulation: %s",
                    gating_path,
                    e,
                )
        except Exception as e:
            logger.warning(
                "Failed to derive meta-gating configuration from trading-simulation results: %s",
                e,
            )

    # -------------------------------------------------------------------------
    # Interpretation helpers
    # -------------------------------------------------------------------------

    brier = calibration_metrics.get("brier_score", float("nan"))
    if not np.isfinite(brier):
        brier_comment = "Brier score not available."
    elif brier > 0.25:
        brier_comment = "Brier > 0.25 → Poorly calibrated probabilities."
    elif brier > 0.18:
        brier_comment = "Brier 0.18-0.25 → Moderate calibration."
    else:
        brier_comment = "Brier ≤ 0.18 → Well-calibrated probabilities."

    ece = calibration_metrics.get("expected_calibration_error", float("nan"))
    if not np.isfinite(ece):
        ece_comment = "ECE not available."
    elif ece > 0.15:
        ece_comment = "ECE > 0.15 → Significant calibration error."
    elif ece > 0.05:
        ece_comment = "ECE 0.05-0.15 → Moderate calibration error."
    else:
        ece_comment = "ECE ≤ 0.05 → Well-calibrated model."

    # -------------------------------------------------------------------------
    # Console output
    # -------------------------------------------------------------------------

    print("""
=== Trading Simulation Diagnostics ===
""".strip())
    print(f"Symbol: {symbol} | Exchange: {exchange} | Timeframe: {timeframe}")
    print(f"Date range: {date_range_days} days | Valid samples: {n_valid}")
    print()

    print("-- Model Calibration --")
    print(f"Brier Score: {calibration_metrics.get('brier_score', float('nan')):.4f}")
    print(f"Expected Calibration Error (ECE): {calibration_metrics.get('expected_calibration_error', float('nan')):.4f}")
    print(f"Max Calibration Error (MCE): {calibration_metrics.get('max_calibration_error', float('nan')):.4f}")
    print()

    print("-- Trading Metrics by Probability Threshold --")
    print()

    for key in sorted(threshold_results.keys()):
        result = threshold_results[key]
        thr = result["threshold"]

        if result.get("insufficient_data", False):
            print(f"Threshold {thr:.2f}: Insufficient data ({result['n_trades']} trades)")
            continue

        print(f"Threshold {thr:.2f}:")
        print(f"  Trades: {result['n_trades']} ({result['trades_per_day']:.2f}/day)")
        print(f"  Mean Return/Trade: {result['mean_return_per_trade']*100:.4f}%")
        print(f"  PnL/Day: {result['pnl_per_day_pct']:.4f}%")
        print(f"  Win Rate: {result['win_rate']*100:.1f}%")
        print(f"  Sharpe Ratio: {result['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown: {result['max_drawdown']:.2f}%")
        print(f"  Final Equity: {result['final_equity']:.4f}")
        print(f"  Max Consecutive Losses: {result['max_consecutive_losses']}")
        print(f"  Avg Consecutive Losses: {result['avg_consecutive_losses']:.2f}")
        print(f"  Win-Rate Stability: {result['win_rate_stability']:.3f}")
        print()

    print("-- Interpretation --")
    print(f"Calibration ({brier:.4f}): {brier_comment}")
    print(f"ECE ({ece:.4f}): {ece_comment}")

    # -------------------------------------------------------------------------
    # Export payload
    # -------------------------------------------------------------------------

    payload = {
        "section": "trading_simulation",
        "date_range_days": int(date_range_days),
        "n_valid_samples": int(n_valid),
        "calibration": calibration_metrics,
        "calibration_curve": calibration_curve_data,
        "threshold_results": threshold_results,
    }
    if regime_threshold_results:
        payload["regime_threshold_results"] = regime_threshold_results
    if regime_gating_cfg:
        payload["regime_best_gates"] = regime_gating_cfg
    if best_gate_info is not None:
        payload["best_gate"] = best_gate_info

    md_lines = [
        "# Trading Simulation Diagnostics",
        "",
        f"**Symbol**: {symbol}",
        f"**Exchange**: {exchange}",
        f"**Timeframe**: {timeframe}",
        f"**Direction**: {direction}",
        "",
        "## Overview",
        f"- Date range: {date_range_days} days",
        f"- Valid samples: {n_valid}",
        "",
        "## Model Calibration",
        f"- Brier Score: {calibration_metrics.get('brier_score', float('nan')):.4f}",
        f"- Expected Calibration Error (ECE): {calibration_metrics.get('expected_calibration_error', float('nan')):.4f}",
        f"- Max Calibration Error (MCE): {calibration_metrics.get('max_calibration_error', float('nan')):.4f}",
        "",
        "### Calibration Interpretation",
        f"- {brier_comment}",
        f"- {ece_comment}",
        "",
        "## Trading Metrics by Probability Threshold",
        "",
    ]

    # Add threshold results to markdown
    for key in sorted(threshold_results.keys()):
        result = threshold_results[key]
        thr = result["threshold"]

        md_lines.append(f"### Threshold {thr:.2f}")

        if result.get("insufficient_data", False):
            md_lines.append(f"- Insufficient data ({result['n_trades']} trades)")
        else:
            md_lines.extend([
                f"- **Trades**: {result['n_trades']} ({result['trades_per_day']:.2f}/day)",
                f"- **Mean Return/Trade**: {result['mean_return_per_trade']*100:.4f}%",
                f"- **PnL/Day**: {result['pnl_per_day_pct']:.4f}%",
                f"- **Win Rate**: {result['win_rate']*100:.1f}%",
                f"- **Sharpe Ratio**: {result['sharpe_ratio']:.3f}",
                f"- **Max Drawdown**: {result['max_drawdown']:.2f}%",
                f"- **Final Equity**: {result['final_equity']:.4f}",
                f"- **Max Consecutive Losses**: {result['max_consecutive_losses']}",
                f"- **Avg Consecutive Losses**: {result['avg_consecutive_losses']:.2f}",
                f"- **Win-Rate Stability**: {result['win_rate_stability']:.3f}",
            ])
        md_lines.append("")

    # Add summary table
    md_lines.extend([
        "## Summary Table",
        "",
        "| Threshold | Trades | Trades/Day | Mean Return | PnL/Day | Win Rate | Sharpe | Max DD | Consec Losses |",
        "|-----------|--------|------------|-------------|---------|----------|--------|--------|---------------|",
    ])

    for key in sorted(threshold_results.keys()):
        result = threshold_results[key]
        if result.get("insufficient_data", False):
            continue

        md_lines.append(
            f"| {result['threshold']:.2f} | {result['n_trades']} | {result['trades_per_day']:.2f} | "
            f"{result['mean_return_per_trade']*100:.3f}% | {result['pnl_per_day_pct']:.3f}% | "
            f"{result['win_rate']*100:.1f}% | {result['sharpe_ratio']:.2f} | "
            f"{result['max_drawdown']:.1f}% | {result['max_consecutive_losses']} |"
        )

    if best_gate_info is not None:
        md_lines.extend(
            [
                "",
                "## Recommended Gating Threshold (from Trading Simulation)",
                "",
                f"- **Probability threshold**: {best_gate_info['threshold']:.2f}",
                f"- **Trades**: {best_gate_info['n_trades']} ({best_gate_info['trades_per_day']:.3f}/day)",
                f"- **Mean return/trade**: {best_gate_info['mean_return_per_trade']*100:.4f}%",
                f"- **PnL/day**: {best_gate_info['pnl_per_day_pct']:.4f}%",
                f"- **Sharpe (trades)**: {best_gate_info['sharpe_ratio']:.3f}",
                f"- **Max drawdown**: {best_gate_info['max_drawdown']:.2f}%",
                f"- **Final equity**: {best_gate_info['final_equity']:.4f}",
            ]
        )

    if regime_gating_cfg:
        md_lines.extend([
            "",
            "## Regime-Specific Recommended Thresholds",
            "",
        ])
        for regime_name, cfg in sorted(regime_gating_cfg.items()):
            md_lines.extend(
                [
                    f"- **Regime** `{regime_name}`:",
                    f"  - prob_threshold = {cfg.get('prob_threshold', float('nan')):.2f}",
                    f"  - trades/day ≈ {cfg.get('trades_per_day', float('nan')):.3f}",
                    f"  - mean_return ≈ {cfg.get('mean_return', float('nan'))*100:.4f}%",
                    f"  - Sharpe ≈ {cfg.get('sharpe', float('nan')):.3f}",
                    f"  - n_trades = {cfg.get('n_trades', 0)}",
                ]
            )

    json_path, md_path = _export_report(
        prefix="snr_trading_simulation",
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        direction=direction,
        model=model,
        payload=payload,
        markdown_lines=md_lines,
    )

    print(f"\nReports saved to: {json_path} and {md_path}")


def run_full(
    symbol: str,
    exchange: str,
    timeframe: str,
    direction: str = "long",
    model: str = "analyst",
    cv_splits_learn: int = 3,
    cv_splits_robust: int = 5,
    prob_column: str = "meta_probability",
    prob_thresholds: Optional[List[float]] = None,
) -> None:
    _LAST_EXPORTS.clear()

    run_label_quality(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        direction=direction,
        model=model,
    )

    run_label_learnability(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        direction=direction,
        model=model,
        cv_splits=cv_splits_learn,
    )

    run_model_robustness(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        direction=direction,
        model=model,
        cv_splits=cv_splits_robust,
    )

    run_trading_simulation(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        direction=direction,
        model=model,
        prob_column=prob_column,
        prob_thresholds=prob_thresholds,
        cv_splits=cv_splits_robust,
    )

    required_prefixes = [
        "snr_label_quality",
        "snr_label_learnability",
        "snr_model_robustness",
        "snr_trading_simulation",
    ]
    missing = [p for p in required_prefixes if p not in _LAST_EXPORTS]
    if missing:
        logger.warning("Missing diagnostics for prefixes: %s", ", ".join(missing))
        return

    lq = _LAST_EXPORTS["snr_label_quality"]
    ll = _LAST_EXPORTS["snr_label_learnability"]
    mr = _LAST_EXPORTS["snr_model_robustness"]
    ts = _LAST_EXPORTS["snr_trading_simulation"]

    lq_payload = lq["payload"]
    ll_payload = ll["payload"]
    mr_payload = mr["payload"]
    ts_payload = ts["payload"]

    lq_coverage = lq_payload.get("coverage")
    lq_positive_rate = lq_payload.get("positive_rate")
    lq_post = lq_payload.get("post", {})
    lq_snr_post = lq_post.get("snr_pos")
    lq_cohens_d = lq_post.get("cohens_d")

    learnability = ll_payload.get("learnability")
    learn_mean_auc = ll_payload.get("mean_auc")
    balance = ll_payload.get("balance")
    combined = ll_payload.get("combined")

    mr_summary = mr_payload.get("summary", {})
    mr_mean_auc = mr_summary.get("mean_auc")
    mr_stability = mr_summary.get("stability_score")
    mr_mean_brier = mr_summary.get("mean_brier")

    mr_advanced = mr_payload.get("advanced", {}) if isinstance(mr_payload, dict) else {}
    mr_global_auc = mr_advanced.get("global_auc") if isinstance(mr_advanced, dict) else None
    mr_pseudo_r2 = mr_advanced.get("pseudo_r2") if isinstance(mr_advanced, dict) else None
    mr_perm_p = mr_advanced.get("perm_pvalue_auc") if isinstance(mr_advanced, dict) else None
    mr_baseline = mr_advanced.get("baseline", {}) if isinstance(mr_advanced, dict) else {}
    mr_delta_auc = mr_baseline.get("delta_auc") if isinstance(mr_baseline, dict) else None
    mr_delta_brier = mr_baseline.get("delta_brier") if isinstance(mr_baseline, dict) else None
    mr_delta_ap = mr_baseline.get("delta_ap") if isinstance(mr_baseline, dict) else None

    lq_summary_score = lq_payload.get("summary_score", {})
    ll_summary_score = ll_payload.get("summary_score", {})
    mr_summary_score = mr_payload.get("summary_score", {})

    lq_score = lq_summary_score.get("score")
    lq_rating = lq_summary_score.get("rating")
    ll_score = ll_summary_score.get("score")
    ll_rating = ll_summary_score.get("rating")
    mr_score = mr_summary_score.get("score")
    mr_rating = mr_summary_score.get("rating")

    lq_advanced = lq_payload.get("advanced", {})
    aleatoric_fraction = lq_advanced.get("aleatoric_uncertainty_fraction")

    def _fmt_pct(value) -> str:
        if value is None or not np.isfinite(float(value)):
            return "N/A"
        return f"{float(value):.1%}"

    def _fmt_float(value, digits: int = 4) -> str:
        if value is None or not np.isfinite(float(value)):
            return "N/A"
        return f"{float(value):.{digits}f}"

    md_lines: list[str] = [
        "# Full SNR Diagnostics Report",
        "",
        f"**Symbol**: {symbol}",
        f"**Exchange**: {exchange}",
        f"**Timeframe**: {timeframe}",
        f"**Direction**: {direction}",
        f"**Model**: {model}",
        "",
        "## High-Level Summary",
        f"- Label coverage: {_fmt_pct(lq_coverage)} (labeled / total samples)",
        f"- Label positive rate: {_fmt_pct(lq_positive_rate)}",
        f"- Label economic SNR (post-filter, label=1): {_fmt_float(lq_snr_post, digits=3)}",
        f"- Label effect size (post-filter Cohen's d): {_fmt_float(lq_cohens_d, digits=3)}",
        f"- Aleatoric uncertainty fraction (|return| < cost): {_fmt_pct(aleatoric_fraction)}",
        "",
        f"- Learnability mean CV AUC: {_fmt_float(learn_mean_auc, digits=4)}",
        f"- Learnability score (AUC - 0.5 * std): {_fmt_float(learnability, digits=4)}",
        f"- Label balance (entropy score): {_fmt_float(balance, digits=4)}",
        f"- Combined label-quality score: {_fmt_float(combined, digits=4)}",
        "",
        f"- Probe model mean AUC: {_fmt_float(mr_mean_auc, digits=4)}",
        f"- Probe model stability score: {_fmt_float(mr_stability, digits=4)}",
        f"- Probe model mean Brier score: {_fmt_float(mr_mean_brier, digits=4)}",
        f"- Probe global AUC (all folds combined): {_fmt_float(mr_global_auc, digits=4)}",
        f"- Probe pseudo-R^2 (y vs predicted prob): {_fmt_float(mr_pseudo_r2, digits=4)}",
        f"- Probe permutation p-value (AUC): {_fmt_float(mr_perm_p, digits=3)}",
        f"- Probe vs baseline ΔAUC: {_fmt_float(mr_delta_auc, digits=4)}, ΔBrier (baseline - probe): {_fmt_float(mr_delta_brier, digits=4)}, ΔAP: {_fmt_float(mr_delta_ap, digits=4)}",
        "",
        f"- Label-quality summary score: {_fmt_float(lq_score, digits=3)} (Rating: {lq_rating or 'N/A'})",
        f"- Learnability summary score: {_fmt_float(ll_score, digits=3)} (Rating: {ll_rating or 'N/A'})",
        f"- Model-robustness summary score: {_fmt_float(mr_score, digits=3)} (Rating: {mr_rating or 'N/A'})",
        "",
        "## Metric Definitions (brief)",
        "- **Coverage**: share of events that receive a binary label.",
        "- **Positive rate**: fraction of labeled events with label=1.",
        "- **Cohen's d**: standardized difference in mean returns between positive and negative labels.",
        "- **SNR (mean/std)**: mean positive-label return divided by its standard deviation.",
        "- **Learnability AUC**: mean cross-validated ROC AUC from a shallow probe model.",
        "- **Learnability score**: AUC penalized by instability (AUC - 0.5 * std).",
        "- **Entropy balance**: how balanced labels are between 0 and 1; 1.0 is 50/50.",
        "- **Combined score**: weighted average of learnability and balance.",
        "- **Brier score**: mean squared error between predicted probabilities and true labels; lower is better.",
        "- **Stability score**: 1 - std(AUC)/mean(AUC); higher indicates more stable performance across folds.",
        "",
        "## Detailed Diagnostics",
        "",
        "### Label-Quality",
    ]

    lq_md = lq["markdown_lines"]
    ll_md = ll["markdown_lines"]
    mr_md = mr["markdown_lines"]
    ts_md = ts["markdown_lines"]

    md_lines.extend(lq_md[2:] if len(lq_md) > 2 else lq_md)
    md_lines.extend([
        "",
        "### Label-Learnability",
    ])
    md_lines.extend(ll_md[2:] if len(ll_md) > 2 else ll_md)
    md_lines.extend([
        "",
        "### Model-Robustness",
    ])
    md_lines.extend(mr_md[2:] if len(mr_md) > 2 else mr_md)
    md_lines.extend([
        "",
        "### Trading-Simulation",
    ])
    md_lines.extend(ts_md[2:] if len(ts_md) > 2 else ts_md)

    md_lines.extend([
        "",
        "## Label Quality, Learnability and Robustness Reference",
        "",
        "### Label quality",
        "1. Noise Ceiling (if multiple labelers / repeated labels). If you have multiple labelers, this can be combined with inter-rater reliability metrics (ICC, Cohen09s kappa).",
        "> 0.6 b Labels are internally consistent; high R00 is achievable.",
        "0.40.6 b Labels moderately noisy; realistic ceilings are limited.",
        "< 0.4 b Labels are extremely noisy; even perfect models cannot perform well.",
        "",
        "2. Aleatoric Uncertainty Fraction. Could link it to expected max R00; i.e., intrinsic unpredictability sets a ceiling for achievable performance",
        "< 40% b Most error is model/feature-driven; improvement is possible.",
        "4060% b Mixed noise and model limitations.",
        "> 60% b Most unpredictability is intrinsic to the target.",
        "",
        "### Label learnability vs noise",
        "1. R00. Low R00 could be due to missing features or poor model choice, not just label noise",
        "R00 > 0.40 b The target has a strong predictable signal; meaningful modeling gains are possible.",
        "0.10 < R00 0.40 b The target has a weakbmoderate signal; features matter more than model choice.",
        "R00 0.10 b The target is barely predictable; noise likely dominates.",
        "",
        "2. SNR",
        "SNR > 1 b Signal is stronger than noise; the target is learnable.",
        "0.3 < SNR 1 b Weak but real signal exists; more features or nonlinear models may help.",
        "SNR 0.3 b Noise overwhelms signal; predictability is fundamentally low.",
        "",
        "3. Permutation p-value. If p is high, it may indicate noisy labels, but it could also reflect poor features or an underpowered model.",
        "p < 0.01 b The model captures a real, statistically robust pattern.",
        "0.01 c p 0.20 b There might be signal, but itb s weak or unstable.",
        "p > 0.20 b The model performs no better than chance; label likely noisy.",
        "",
        "4. Naive Baselines. A very simple predictive model used as a reference point. Establishes a floor for model performance & distinguish real signal from noise:",
        "Model 4 baseline b low predictability, focus on labels or features",
        "Model >> baseline b real signal exists, worth improving features/model (doesn't say we haven't reached the ceiling)",
        "",
        "### Model & features robustness",
        "1. Bootstrap R00 Confidence Interval. Helps assess stability and reliability of model performance, helps detect overfitting if the CI is very wide or unstable across bootstraps",
        "CI does NOT include 0 b Performance is reliably above noise level.",
        "CI barely clears 0 (lower bound < 0.05) b Signal is present but fragile.",
        "CI spans below 0 b Model performance might be indistinguishable from noise.",
        "",
        "2. Residual Structure. Residual structure tells you what signal your model/features are missing (and if there is a pattern), not directly about label noise.",
        "Residuals look random b The model extracted essentially all available signal.",
        "Residuals show patterns b There is remaining structure the model/features are missing.",
        "Residuals differ strongly across subgroups b Predictability varies by segment (not globally noisy).",
        "",
        "3. Residual Autocorrelation. Measures whether residuals are temporally or sequentially correlated (often lag-1 autocorrelation). Even if R00 looks okay, autocorrelated residuals indicate hidden structure your features/model missed.",
        "Lag-1 autocorr < 0.10 b No missing temporal/ordered structure.",
        "0.100.20 b Some time dependence is not modeled.",
        "> 0.20 b Strong sequential structure missing; target not fully explained.",
        "",
        "4. Model Family Comparison. Helps diagnose whether your model class is adequate and whether there09s remaining learnable signal",
        "Nonlinear >> linear b There is real nonlinear structure not captured by simple models.",
        "Linear >> nonlinear b Tree model overfitting.",
        "All models perform similarly well b The problem is stable and well-posed.",
        "All models perform similarly poorly b The target has low intrinsic predictability.",
        "Ensembles significantly better b High model uncertainty; more data helps",
    ])

    combined_payload = {
        "cv_splits_learn": int(cv_splits_learn),
        "cv_splits_robust": int(cv_splits_robust),
        "label_quality": lq_payload,
        "label_learnability": ll_payload,
        "model_robustness": mr_payload,
        "trading_simulation": ts_payload,
    }

    json_path, md_path = _export_report(
        prefix="snr_full_diagnostics",
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        direction=direction,
        model=model,
        payload=combined_payload,
        markdown_lines=md_lines,
    )

    print(f"\nFull diagnostics report saved to: {json_path} and {md_path}")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _add_common_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--symbol", type=str, default="ETHUSDT")
    sub.add_argument("--exchange", type=str, default="binance")
    sub.add_argument("--timeframe", type=str, default="15m")
    sub.add_argument("--direction", type=str, default="long", choices=["long", "short", "both"])
    sub.add_argument("--model", type=str, default="analyst")


def main() -> None:
    parser = argparse.ArgumentParser(description="SNR and label diagnostics for meta-labeling outputs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # label-quality
    p_quality = subparsers.add_parser("label-quality", help="Label distribution and economic SNR diagnostics")
    _add_common_args(p_quality)

    # label-learnability
    p_learn = subparsers.add_parser("label-learnability", help="Learnability and entropy-based label quality")
    _add_common_args(p_learn)
    p_learn.add_argument("--cv-splits", type=int, default=3)

    # model-robustness
    p_robust = subparsers.add_parser("model-robustness", help="Probe model CV robustness diagnostics")
    _add_common_args(p_robust)
    p_robust.add_argument("--cv-splits", type=int, default=5)

    # trading-simulation
    p_trading = subparsers.add_parser("trading-simulation", help="Trading simulation with calibration and threshold analysis")
    _add_common_args(p_trading)
    p_trading.add_argument("--cv-splits", type=int, default=5)
    p_trading.add_argument("--prob-column", type=str, default="meta_probability",
                           help="Name of probability column to use (default: meta_probability)")
    p_trading.add_argument("--prob-thresholds", type=float, nargs="+", default=[0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
                          help="Probability thresholds to analyze (default: 0.55 0.60 0.65 0.70 0.75 0.80)")

    # full
    p_full = subparsers.add_parser("full", help="Run all diagnostics and aggregate results")
    _add_common_args(p_full)
    p_full.add_argument("--cv-splits-learn", type=int, default=3)
    p_full.add_argument("--cv-splits-robust", type=int, default=5)
    p_full.add_argument("--prob-column", type=str, default="meta_probability",
                        help="Name of probability column to use for trading simulation (default: meta_probability)")
    p_full.add_argument("--prob-thresholds", type=float, nargs="+", default=[0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
                        help="Probability thresholds to analyze (default: 0.55 0.60 0.65 0.70 0.75 0.80)")

    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    if args.command == "label-quality":
        run_label_quality(
            symbol=args.symbol,
            exchange=args.exchange,
            timeframe=args.timeframe,
            direction=args.direction,
            model=args.model,
        )

    elif args.command == "label-learnability":
        run_label_learnability(
            symbol=args.symbol,
            exchange=args.exchange,
            timeframe=args.timeframe,
            direction=args.direction,
            model=args.model,
            cv_splits=args.cv_splits,
        )

    elif args.command == "model-robustness":
        run_model_robustness(
            symbol=args.symbol,
            exchange=args.exchange,
            timeframe=args.timeframe,
            direction=args.direction,
            model=args.model,
            cv_splits=args.cv_splits,
        )

    elif args.command == "trading-simulation":
        run_trading_simulation(
            symbol=args.symbol,
            exchange=args.exchange,
            timeframe=args.timeframe,
            direction=args.direction,
            model=args.model,
            prob_column=args.prob_column,
            prob_thresholds=args.prob_thresholds,
            cv_splits=args.cv_splits,
        )

    elif args.command == "full":
        run_full(
            symbol=args.symbol,
            exchange=args.exchange,
            timeframe=args.timeframe,
            direction=args.direction,
            model=args.model,
            cv_splits_learn=args.cv_splits_learn,
            cv_splits_robust=args.cv_splits_robust,
            prob_column=args.prob_column,
            prob_thresholds=args.prob_thresholds,
        )


if __name__ == "__main__":
    main()
