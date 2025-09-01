# src/analyst/regime_runtime.py

from src.utils.hmm_composite_manager import get_hmm_composite_manager
from src.utils.logger import system_logger
from typing import Any
import os

import joblib
import numpy as np
import pandas as pd

def _load_parquet(path: str) -> pd.DataFrame | None:
    try:
        if os.path.exists(path):
            return pd.read_parquet(path)
        return None
    except Exception as e:
        system_logger.warning(f"Failed to read parquet {path}: {e}")
        return None

def _align_last(df: pd.DataFrame, ts: pd.Timestamp | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = (
            df.dropna(subset=["timestamp"])
            .sort_values("timestamp")
            .set_index("timestamp")
        )
    if ts is None:
        return df.tail(1)
    return df.loc[df.index <= ts].tail(1)

def _ewm_prob(ind: pd.Series, span: int = 3) -> pd.Series:
    return ind.astype(float).ewm(span=span, adjust=False).mean().clip(0.0, 1.0)

def _entropy(arr_df: pd.DataFrame) -> pd.Series:
    p = arr_df.clip(1e-9, 1.0)
    return -np.sum(p * np.log(p), axis=1)

def _compute_transition_matrix(cluster_ids: np.ndarray) -> np.ndarray:
    vals = cluster_ids.astype(int)
    K = int(np.max(vals[vals >= 0]) + 1) if np.any(vals >= 0) else 0
    T = np.zeros((K, K), dtype=float)
    for i in range(len(vals) - 1):
        c, n = vals[i], vals[i + 1]
        if c >= 0 and n >= 0:
            T[c, n] += 1
    rowsum = T.sum(axis=1, keepdims=True) + 1e-9
    return T / rowsum

def _build_p_k_matrix(cluster_ids: pd.Series) -> pd.DataFrame:
    labels = sorted([int(x) for x in np.unique(cluster_ids.values) if int(x) >= 0])
    p_cols: dict[str, pd.Series] = {}
    for k in labels:
        ind = (cluster_ids == k).astype(float)
        p_cols[f"p_k_{k}"] = _ewm_prob(ind, span=3)
    p_df = pd.DataFrame(p_cols, index=cluster_ids.index)
    if p_df.empty:
        return p_df
    s = p_df.sum(axis=1).replace(0, 1.0)
    return p_df.div(s, axis=0)

def _mk_features(block_df: pd.DataFrame, comp_df: pd.DataFrame) -> pd.DataFrame:
    cluster_ids = comp_df["composite_cluster_id"].astype(int)
    p_df = _build_p_k_matrix(cluster_ids)
    dp_df = p_df.diff().fillna(0.0).add_prefix("dp_")
    d2p_df = dp_df.diff().fillna(0.0).add_prefix("d2p_")
    features = pd.concat([p_df, dp_df, d2p_df], axis=1)
    features["entropy"] = _entropy(
        p_df if not p_df.empty else pd.DataFrame(index=features.index),
    )
    for blk in ["momentum", "volatility", "liquidity", "microstructure"]:
        cols = [c for c in block_df.columns if c.startswith(f"{blk}_p_state_")]
        if cols:
            features[f"{blk}_entropy"] = _entropy(block_df[cols])
    T = _compute_transition_matrix(cluster_ids.values)
    K = T.shape[0]
    if K > 0:
        cur = cluster_ids.values
        Pnext = np.zeros((len(cur), K), dtype=float)
        for i in range(len(cur)):
            c = cur[i]
            if 0 <= c < K:
                Pnext[i, :] = T[c, :]
        for j in range(K):
            features[f"p_next_{j}"] = Pnext[:, j]
        features["most_likely_next"] = np.argmax(Pnext, axis=1)
    return features

def _build_keep_cols(X_all: pd.DataFrame, k: int) -> list[str]:
    return [
        c
        for c in X_all.columns
        if (
            c.startswith((f"p_k_{k}", f"dp_p_k_{k}", f"d2p_p_k_{k}", "p_next_"))
            or c == "entropy"
            or c
            in (
                "momentum_entropy",
                "volatility_entropy",
                "liquidity_entropy",
                "microstructure_entropy",
            )
        )
    ]
