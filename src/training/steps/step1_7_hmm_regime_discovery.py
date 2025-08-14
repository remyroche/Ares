# src/training/steps/step1_7_hmm_regime_discovery.py

import os
import json
import math
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.logger import system_logger
from src.utils.error_handler import (
    handle_errors,
    handle_data_processing_errors,
    handle_type_conversions,
    safe_division,
    clean_dataframe,
)

# FE (vectorized only)
from src.training.steps.vectorized_advanced_feature_engineering import (
    VectorizedAdvancedFeatureEngineering,
)

# Data loading
from src.training.steps.unified_data_loader import UnifiedDataLoader

# HMM
from hmmlearn.hmm import GMMHMM

# Selection
from sklearn.preprocessing import StandardScaler

# Optional clustering
try:
    import hdbscan  # type: ignore
    _HDBSCAN_AVAILABLE = True
except Exception:
    _HDBSCAN_AVAILABLE = False

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances


@dataclass
class BlockConfig:
    """Configuration for HMM block analysis."""
    name: str
    n_states: int
    max_features: int = 3


# Default block setup
BLOCKS: List[BlockConfig] = [
    BlockConfig("momentum", 4, 3),
    BlockConfig("volatility", 4, 3),
    BlockConfig("liquidity", 3, 3),
    BlockConfig("microstructure", 5, 3),
]

# Timeframes to train on
TIMEFRAMES: List[str] = ["1m", "5m", "15m"]


def _is_candlestick_feature(name: str) -> bool:
    """Check if a feature name represents a candlestick pattern."""
    name_l = name.lower()
    candles = [
        "engulf", "hammer", "shooting_star", "tweezer", "marubozu",
        "three_methods", "doji", "spinning_top",
    ]
    return any(c in name_l for c in candles)


def _assign_block(name: str) -> Optional[str]:
    """Assign a feature to a specific block based on its name."""
    name_l = name.lower()
    if _is_candlestick_feature(name_l):
        return None
    if any(k in name_l for k in ["rsi", "macd", "bb_position", "momentum", "trend_"]):
        return "momentum"
    if any(k in name_l for k in ["volatility", "bb_width", "atr", "ewma_volatility"]):
        return "volatility"
    if any(k in name_l for k in ["volume", "trade_count", "trade_volume", "liquidity", "vwap"]):
        return "liquidity"
    if any(k in name_l for k in ["spread", "imbalance", "impact", "order_book", "orderflow", "micro"]):
        return "microstructure"
    # default: None (ignored)
    return None


@handle_type_conversions(default_return=np.array([]))
def _winsorize(col: np.ndarray, lower: float = 0.01, upper: float = 0.99) -> np.ndarray:
    """Winsorize array to handle outliers."""
    if col.size == 0:
        return col
    lo = np.nanquantile(col, lower)
    hi = np.nanquantile(col, upper)
    return np.clip(col, lo, hi)


@handle_data_processing_errors(default_return=pd.DataFrame())
def _robust_scale(df: pd.DataFrame) -> pd.DataFrame:
    """Robust scaling using IQR with fallback to standard deviation."""
    df_out = pd.DataFrame(index=df.index)
    for c in df.columns:
        arr = df[c].astype(float).values
        arr = _winsorize(arr)
        q1 = np.nanquantile(arr, 0.25)
        q3 = np.nanquantile(arr, 0.75)
        iqr = (q3 - q1) if (q3 - q1) != 0 else np.nan
        if not np.isnan(iqr) and iqr > 1e-12:
            scaled = (arr - np.nanmedian(arr)) / (iqr + 1e-12)
        else:
            std = np.nanstd(arr)
            scaled = (arr - np.nanmean(arr)) / (std + 1e-12)
        df_out[c] = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
    return df_out


@handle_data_processing_errors(default_return=[])
def _corr_prune(df: pd.DataFrame, thr: float = 0.95) -> List[str]:
    """Remove highly correlated columns."""
    if df.empty:
        return []
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    return [c for c in upper.columns if any(upper[c] >= thr)]


@handle_data_processing_errors(default_return=pd.DataFrame())
def _select_block_features(full_df: pd.DataFrame, block: str, max_features: int) -> pd.DataFrame:
    """Select and prepare features for a specific block."""
    cols = [c for c in full_df.columns if _assign_block(c) == block]
    if not cols:
        return pd.DataFrame(index=full_df.index)
    X = full_df[cols].copy()
    # Drop constant columns
    nunique = X.nunique(dropna=True)
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        X = X.drop(columns=const_cols, errors="ignore")
    if X.shape[1] <= max_features:
        return X.fillna(0)
    # Correlation prune
    drop_cols = _corr_prune(X, thr=0.95)
    X = X.drop(columns=drop_cols, errors="ignore")
    if X.shape[1] <= max_features:
        return X.fillna(0)
    # Unsupervised heuristic: choose features with highest variance (post robust scale)
    Xr = _robust_scale(X)
    var = Xr.var().sort_values(ascending=False)
    keep = list(var.head(max_features).index)
    return Xr[keep].fillna(0)


@handle_errors(
    exceptions=(Exception,),
    default_return=(None, None),
    context="step1_7_hmm_regime_discovery._fit_block_hmm",
)
def _fit_block_hmm(X: pd.DataFrame, n_states: int, random_state: int = 42) -> Tuple[Optional[GMMHMM], Optional[StandardScaler]]:
    """Fit HMM model for a specific block with enhanced error handling."""
    try:
        # Use GMMHMM with diagonal covariances, 2 mixtures per state to approximate heavy tails
        model = GMMHMM(
            n_components=n_states,
            n_mix=2,
            covariance_type="diag",
            n_iter=200,
            tol=1e-3,
            random_state=random_state,
        )
        # hmmlearn expects 2D array
        arr = X.values.astype(float)
        # Scale for stability using StandardScaler (after robust scaling)
        scaler = StandardScaler()
        arr_scaled = scaler.fit_transform(arr)
        model.fit(arr_scaled)
        return model, scaler
    except Exception as e:
        system_logger.error(f"Error fitting HMM for block: {e}")
        return None, None


@handle_errors(
    exceptions=(Exception,),
    default_return=(np.array([]), np.array([])),
    context="step1_7_hmm_regime_discovery._posteriors",
)
def _posteriors(model: GMMHMM, scaler: StandardScaler, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Get posterior probabilities and state predictions."""
    try:
        arr = X.values.astype(float)
        arr_scaled = scaler.transform(arr)
        # score_samples returns (logprob, posteriors)
        logprob, gamma = model.score_samples(arr_scaled)
        states = model.predict(arr_scaled)
        return states.astype(int), gamma
    except Exception as e:
        system_logger.error(f"Error computing posteriors: {e}")
        return np.array([]), np.array([])


@handle_data_processing_errors(default_return=(pd.Series(dtype=str), pd.DataFrame()))
def _build_combination_profiles(block_states: Dict[str, np.ndarray], block_posteriors: Dict[str, np.ndarray]) -> Tuple[pd.Series, pd.DataFrame]:
    """Build combination profiles from block states and posteriors."""
    # combination key per row (efficient join of key parts)
    if not block_states:
        combination_keys = pd.Series(dtype=str)
    else:
        key_parts = [[f"{b}:{int(v)}" for v in s] for b, s in block_states.items()]
        # Transpose and join
        joined_keys = ["|".join(map(str, row)) for row in zip(*key_parts)]
        combination_keys = pd.Series(joined_keys)
    # profile vector: concatenated mean posteriors per block across occurrences
    profiles = {}
    for combo, idx in combination_keys.groupby(combination_keys).groups.items():
        vecs: List[np.ndarray] = []
        for b, gamma in block_posteriors.items():
            if len(idx) == 0:
                continue
            # mean posterior for this block at these indices
            vecs.append(np.nanmean(gamma[idx, :], axis=0))
        profiles[combo] = np.concatenate(vecs, axis=0)
    profile_df = pd.DataFrame.from_dict(profiles, orient="index")
    return combination_keys, profile_df


@handle_data_processing_errors(default_return=pd.Series([-1] * 1000))
def _cluster_combinations(profile_df: pd.DataFrame, min_cluster_size: int = 5) -> pd.Series:
    """Cluster combinations using HDBSCAN or fallback to AgglomerativeClustering."""
    if profile_df.empty or profile_df.shape[0] < 2:
        return pd.Series([-1] * profile_df.shape[0], index=profile_df.index)
    X = profile_df.values.astype(float)
    # Normalize rows to unit norm to emphasize cosine
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    Xn = X / norms
    if _HDBSCAN_AVAILABLE and profile_df.shape[0] >= max(10, min_cluster_size):
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
        labels = clusterer.fit_predict(Xn)
        return pd.Series(labels, index=profile_df.index)
    # Fallback: Agglomerative on cosine distance, choose a number of clusters heuristically
    dist = cosine_distances(Xn)
    # Heuristic: ~1 cluster per 40 combos, capped [2,12]
    n_clusters = int(max(2, min(12, max(2, profile_df.shape[0] // 40))))
    agg = AgglomerativeClustering(n_clusters=n_clusters, affinity="precomputed", linkage="average")
    labels = agg.fit_predict(dist)
    return pd.Series(labels, index=profile_df.index)


@handle_data_processing_errors(default_return={})
def _state_feature_medians(X_block: pd.DataFrame, states: np.ndarray) -> Dict[int, Dict[str, float]]:
    """Calculate median feature values for each state."""
    med = {}
    for s in np.unique(states):
        mask = states == s
        if mask.sum() == 0:
            continue
        med[int(s)] = {c: float(np.nanmedian(X_block.loc[mask, c])) for c in X_block.columns}
    return med


@handle_data_processing_errors(default_return={})
def _name_states(block: str, medians: Dict[int, Dict[str, float]]) -> Dict[int, str]:
    """Generate human-readable names for states based on feature medians."""
    # Generate simple, human-readable names per state using feature medians
    names: Dict[int, str] = {}
    if not medians:
        return names
    # Compute a scalar score per state depending on block
    scores: Dict[int, float] = {}
    for s, feat in medians.items():
        vals = list(feat.values()) if feat else [0.0]
        if block == "volatility":
            # intensity via absolute values
            score = float(np.nanmean(np.abs(vals)))
        elif block == "momentum":
            score = float(np.nanmean(vals))
        elif block == "liquidity":
            score = float(np.nanmean(vals))
        else:  # microstructure
            score = float(np.nanmean(vals))
        scores[int(s)] = score
    # Rank into tertiles
    sorted_states = sorted(scores.items(), key=lambda kv: kv[1])
    n = max(1, len(sorted_states))
    for rank, (s, sc) in enumerate(sorted_states):
        q = rank / max(1, n - 1)
        if block == "momentum":
            if q < 0.33:
                names[s] = "Bearish Momentum"
            elif q > 0.66:
                names[s] = "Bullish Momentum"
            else:
                names[s] = "Neutral Momentum"
        elif block == "volatility":
            if q < 0.33:
                names[s] = "Low & Stable Vol"
            elif q > 0.66:
                names[s] = "High & Choppy Vol"
            else:
                names[s] = "Expanding Vol"
        elif block == "liquidity":
            if q < 0.33:
                names[s] = "Low Liquidity"
            elif q > 0.66:
                names[s] = "High Liquidity"
            else:
                names[s] = "Medium Liquidity"
        else:  # microstructure
            if q < 0.33:
                names[s] = "Tight Microstructure"
            elif q > 0.66:
                names[s] = "Stressed Microstructure"
            else:
                names[s] = "Neutral Microstructure"
    return names


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="step1_7_hmm_regime_discovery._persist_dataframe",
)
def _persist_dataframe(df: pd.DataFrame, path: str) -> None:
    """Persist DataFrame to parquet file with enhanced error handling."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.reset_index().to_parquet(path, index=False)
        system_logger.info(f"✅ Saved DataFrame to {path} (shape: {df.shape})")
    except Exception as e:
        system_logger.error(f"❌ Failed to save DataFrame to {path}: {e}")
        raise


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="step1_7_hmm_regime_discovery._persist_json",
)
def _persist_json(obj: Dict[str, Any], path: str) -> None:
    """Persist JSON object to file with enhanced error handling."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
        system_logger.info(f"✅ Saved JSON to {path}")
    except Exception as e:
        system_logger.error(f"❌ Failed to save JSON to {path}: {e}")
        raise


@handle_errors(exceptions=(Exception,), default_return=False, context="step1_7_hmm_regime_discovery")
async def run_step(
    symbol: str,
    exchange: str = "BINANCE",
    data_dir: str = "data/training",
    timeframe: str = "1m",
    lookback_days: Optional[int] = None,
    **kwargs: Any,
) -> bool:
    """
    Step 1_7: HMM regime discovery via block HMMs and composite clustering.
    Uses vectorized advanced features (excluding candlestick pattern features).
    Outputs per-timeframe block states/posteriors, combination IDs, and composite cluster IDs.
    
    Enhanced with:
    - Comprehensive logging for troubleshooting and efficiency monitoring
    - Thorough error handling using decorators
    - Proper data usage (scaling, normalization, returns vs prices)
    - Complete type hints throughout
    """
    logger = system_logger.getChild("Step1_7.HMMRegimeDiscovery")
    logger.info("🚀 Step 1_7: HMM Regime Discovery — vectorized features only (no candles)")

    # Load data per timeframe
    loader = UnifiedDataLoader({})
    fe = VectorizedAdvancedFeatureEngineering({})
    await fe.initialize()

    any_success = False
    for tf in TIMEFRAMES:
        logger.info(f"🔄 Processing timeframe: {tf}")
        
        try:
            df = await loader.load_unified_data(
                symbol=symbol,
                exchange=exchange,
                timeframe=tf,
                lookback_days=lookback_days or 180,
                use_streaming=False,
            )
            if df is None or df.empty:
                logger.warning(f"⚠️ No unified data for {exchange}_{symbol}_{tf}; skipping")
                continue
                
            # Ensure datetime index
            if "timestamp" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df = df.set_index("timestamp").sort_index()

            # Extract OHLCV for FE
            price_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
            if len(price_cols) < 5:
                logger.warning(f"⚠️ Missing OHLCV columns for {tf}; found {price_cols}")
                continue
            price_df = df[["open", "high", "low", "close", "volume"]].copy()
            vol_df = price_df[["volume"]].copy()

            # Engineer vectorized features
            features_dict = await fe.engineer_features(price_df, vol_df)
            if not features_dict:
                logger.warning(f"⚠️ No features produced for {tf}; skipping")
                continue
            features_df = pd.DataFrame(features_dict, index=price_df.index)

            # Filter out candlestick features only
            filtered_cols = [c for c in features_df.columns if not _is_candlestick_feature(c)]
            features_df = features_df[filtered_cols]

            logger.info(
                {
                    "msg": f"Feature engineering completed for {tf}",
                    "original_features": len(features_dict),
                    "filtered_features": len(filtered_cols),
                    "data_shape": features_df.shape,
                }
            )

            # Build per-block feature matrices with robust scaling and selection
            block_features: Dict[str, pd.DataFrame] = {}
            for blk in BLOCKS:
                X_blk = _select_block_features(features_df, blk.name, blk.max_features)
                if X_blk.empty:
                    logger.warning(f"Block '{blk.name}' has no features after selection — skipping")
                    continue
                # No extra robust scaling here to avoid duplication
                block_features[blk.name] = X_blk

            if not block_features:
                logger.warning(f"⚠️ No blocks populated for {tf}; skipping")
                continue

            logger.info(
                {
                    "msg": f"Block features prepared for {tf}",
                    "blocks": list(block_features.keys()),
                    "block_shapes": {k: v.shape for k, v in block_features.items()},
                }
            )

            # Fit HMMs and infer posteriors
            block_models: Dict[str, Any] = {}
            block_scalers: Dict[str, Any] = {}
            block_states: Dict[str, np.ndarray] = {}
            block_posteriors: Dict[str, np.ndarray] = {}
            state_feature_medians: Dict[str, Dict[int, Dict[str, float]]] = {}

            for blk in BLOCKS:
                X_blk = block_features.get(blk.name)
                if X_blk is None or X_blk.empty:
                    continue
                    
                logger.info(f"🧩 Training HMM for block='{blk.name}' n_states={blk.n_states} features={list(X_blk.columns)}")
                
                model, scaler = _fit_block_hmm(X_blk, blk.n_states)
                if model is None or scaler is None:
                    logger.error(f"❌ Failed to fit HMM for block '{blk.name}'")
                    continue
                    
                states, gamma = _posteriors(model, scaler, X_blk)
                if len(states) == 0 or len(gamma) == 0:
                    logger.error(f"❌ Failed to compute posteriors for block '{blk.name}'")
                    continue
                    
                block_models[blk.name] = model
                block_scalers[blk.name] = scaler
                block_states[blk.name] = states
                block_posteriors[blk.name] = gamma
                state_feature_medians[blk.name] = _state_feature_medians(X_blk, states)

                logger.info(
                    {
                        "msg": f"HMM training completed for block '{blk.name}'",
                        "n_states": blk.n_states,
                        "unique_states_found": len(np.unique(states)),
                        "state_counts": {int(s): int((states == s).sum()) for s in np.unique(states)},
                    }
                )

            # Persist per-block states and posteriors per timeframe
            out_idx = price_df.index
            block_cols: Dict[str, Any] = {}
            for blk in BLOCKS:
                if blk.name not in block_states:
                    continue
                block_cols[f"{blk.name}_state_id"] = block_states[blk.name]
                gamma = block_posteriors[blk.name]
                for i in range(gamma.shape[1]):
                    block_cols[f"{blk.name}_p_state_{i}"] = gamma[:, i]
            block_df = pd.DataFrame(block_cols, index=out_idx)
            block_out_path = os.path.join(
                data_dir,
                f"{exchange}_{symbol}_hmm_block_states_{tf}.parquet",
            )
            _persist_dataframe(block_df, block_out_path)
            logger.info(f"💾 Saved block states/posteriors -> {block_out_path} ({len(block_df)} rows)")

            # Build combinations and composite clusters
            combo_keys, profile_df = _build_combination_profiles(block_states, block_posteriors)
            if profile_df.empty:
                logger.warning(f"⚠️ Empty combination profiles for {tf}; skipping clustering")
                continue
            # Filter rare combinations (< 0.5% of bars)
            counts = combo_keys.value_counts()
            min_count = max(5, int(0.005 * len(combo_keys)))
            keep_combos = counts[counts >= min_count].index
            profile_df = profile_df.loc[profile_df.index.intersection(keep_combos)]
            labels = _cluster_combinations(profile_df, min_cluster_size=max(5, min_count))

            logger.info(
                {
                    "msg": f"Clustering completed for {tf}",
                    "total_combinations": len(combo_keys),
                    "kept_combinations": len(keep_combos),
                    "unique_clusters": len(np.unique(labels)),
                    "cluster_counts": labels.value_counts().to_dict(),
                }
            )

            # Map timestamps to combination id and composite id
            combo_to_id = {k: i for i, k in enumerate(profile_df.index)}
            combo_series = combo_keys.map(combo_to_id).fillna(-1).astype(int)
            # Only mapped combos get cluster label; others set to -1 (noise)
            cluster_map = labels.to_dict()
            cluster_series = combo_keys.map(cluster_map).fillna(-1).astype(int)

            composite_df = pd.DataFrame(
                {
                    "combination_id": combo_series.values,
                    "composite_cluster_id": cluster_series.values,
                },
                index=out_idx,
            )
            composite_out_path = os.path.join(
                data_dir,
                f"{exchange}_{symbol}_hmm_composite_clusters_{tf}.parquet",
            )
            _persist_dataframe(composite_df, composite_out_path)
            logger.info(f"💾 Saved composite clusters -> {composite_out_path} ({len(composite_df)} rows)")

            # Compute cluster centroids and per-timestamp intensity scores in concatenated posterior space
            # Build gamma concat matrix in a stable column order
            gamma_cols: List[str] = []
            for blk in BLOCKS:
                if blk.name not in block_posteriors:
                    continue
                # columns were written as {blk}_p_state_{i}; ensure order by state index
                n_states_blk = block_posteriors[blk.name].shape[1]
                for i in range(n_states_blk):
                    col = f"{blk.name}_p_state_{i}"
                    if col in block_df.columns:
                        gamma_cols.append(col)
            gamma_concat = block_df[gamma_cols].values.astype(float)
            # Normalize rows
            norms = np.linalg.norm(gamma_concat, axis=1, keepdims=True) + 1e-12
            gamma_norm = gamma_concat / norms
            # Centroids per cluster (exclude noise -1)
            unique_clusters = sorted(int(c) for c in np.unique(cluster_series.values) if int(c) >= 0)
            cluster_centroids: Dict[int, List[float]] = {}
            centroid_norms: Dict[int, np.ndarray] = {}
            for cid in unique_clusters:
                mask = (cluster_series.values == cid)
                if not np.any(mask):
                    continue
                centroid = gamma_norm[mask].mean(axis=0)
                c_norm = np.linalg.norm(centroid) + 1e-12
                centroid = centroid / c_norm
                cluster_centroids[int(cid)] = centroid.tolist()
                centroid_norms[int(cid)] = centroid
            # Intensities: cosine similarity to each centroid
            intensity_data: Dict[str, np.ndarray] = {}
            for cid, cvec in centroid_norms.items():
                vals = np.dot(gamma_norm, cvec.reshape(-1,))
                intensity_data[f"intensity_cluster_{cid}"] = vals
            intensity_df = pd.DataFrame(intensity_data, index=out_idx)
            intensity_out_path = os.path.join(
                data_dir,
                f"{exchange}_{symbol}_hmm_composite_intensity_{tf}.parquet",
            )
            _persist_dataframe(intensity_df, intensity_out_path)
            logger.info(f"💾 Saved composite intensities -> {intensity_out_path} ({len(intensity_df)} rows, {len(intensity_df.columns)} clusters)")

            # Persist meta (state medians, frequencies, profiles)
            # Derive human-readable names per block
            state_names: Dict[str, Dict[int, str]] = {}
            for blk in BLOCKS:
                med = state_feature_medians.get(blk.name, {})
                state_names[blk.name] = _name_states(blk.name, med)

            meta: Dict[str, Any] = {
                "exchange": exchange,
                "symbol": symbol,
                "timeframe": tf,
                "blocks": [{"name": b.name, "n_states": b.n_states} for b in BLOCKS],
                "state_feature_medians": state_feature_medians,
                "state_names": state_names,
                "cluster_centroids": cluster_centroids,
                "combination_counts": counts.to_dict(),
                "kept_combinations": list(map(str, keep_combos)),
                "cluster_labels": labels.astype(int).to_dict(),
            }
            meta_out_path = os.path.join(
                data_dir,
                f"{exchange}_{symbol}_hmm_composite_meta_{tf}.json",
            )
            _persist_json(meta, meta_out_path)
            logger.info(f"💾 Saved meta -> {meta_out_path}")

            any_success = True

        except Exception as e:
            logger.error(f"❌ Error processing timeframe {tf}: {e}")
            continue

    logger.info("✅ Step 1_7: HMM Regime Discovery completed" if any_success else "⚠️ Step 1_7 produced no outputs")
    return any_success