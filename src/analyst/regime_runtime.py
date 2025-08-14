# src/analyst/regime_runtime.py

import os
import json
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import joblib

from src.utils.logger import system_logger


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
		df = df.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
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
	T = T / rowsum
	return T


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
	features["entropy"] = _entropy(p_df if not p_df.empty else pd.DataFrame(index=features.index))
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
		c for c in X_all.columns if (
			c.startswith(f"p_k_{k}") or c.startswith(f"dp_p_k_{k}") or c.startswith(f"d2p_p_k_{k}")
			or c == "entropy" or c.startswith("p_next_") or c in (
				"momentum_entropy", "volatility_entropy", "liquidity_entropy", "microstructure_entropy"
			)
		)
	]


def get_current_regime_info(
	exchange: str,
	symbol: str,
	timeframe: str,
	data_dir: str = "data/training",
	checkpoints_dir: str = "checkpoints",
) -> Dict[str, Any]:
	logger = system_logger.getChild("RegimeRuntime")
	# Load composite clusters & intensities
	comp_path = os.path.join(data_dir, f"{exchange}_{symbol}_hmm_composite_clusters_{timeframe}.parquet")
	int_path = os.path.join(data_dir, f"{exchange}_{symbol}_hmm_composite_intensity_{timeframe}.parquet")
	block_path = os.path.join(data_dir, f"{exchange}_{symbol}_hmm_block_states_{timeframe}.parquet")
	comp_df = _load_parquet(comp_path)
	int_df = _load_parquet(int_path)
	blk_df = _load_parquet(block_path)
	if comp_df is None or comp_df.empty:
		return {"cluster_id": -1, "intensities": {}, "p_emerge": {}, "exit_hazard": None}
	# Align to latest timestamp present in comp_df
	ts = None
	if "timestamp" in comp_df.columns:
		comp_df["timestamp"] = pd.to_datetime(comp_df["timestamp"], errors="coerce", utc=True)
		comp_df = comp_df.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
		last_row = comp_df.tail(1)
		ts = last_row.index[-1]
	else:
		last_row = comp_df.tail(1)
	# Cluster id
	cid = int(last_row["composite_cluster_id"].iloc[0]) if not last_row.empty else -1
	# Intensities (optional)
	intensities: Dict[int, float] = {}
	if int_df is not None and not int_df.empty:
		row_int = _align_last(int_df, ts)
		if not row_int.empty:
			for c in row_int.columns:
				if c.startswith("intensity_cluster_"):
					try:
						kid = int(c.split("_")[-1])
						intensities[kid] = float(row_int[c].iloc[0])
					except Exception:
						pass
	# Forecasting features
	p_emerge: Dict[int, float] = {}
	exit_hazard: float | None = None
	try:
		if blk_df is not None and not blk_df.empty:
			blk_row = _align_last(blk_df, ts)
			comp_row = last_row
			if not blk_row.empty and not comp_row.empty:
				X_all = _mk_features(blk_df, comp_df)
				X_last = X_all.loc[X_all.index <= ts].tail(1)
				# Per-cluster calibrated emergence
				models_dir = os.path.join(checkpoints_dir, "regime_forecasting", exchange, symbol, timeframe)
				if os.path.isdir(models_dir):
					for fname in os.listdir(models_dir):
						if fname.startswith("emergence_cluster_") and fname.endswith("_calibrator.joblib"):
							try:
								k = int(fname.split("_")[2])
								cal = joblib.load(os.path.join(models_dir, fname))
								keep_cols = _build_keep_cols(X_all, k)
								Xi = X_last[keep_cols].fillna(0.0) if keep_cols else X_last.fillna(0.0)
								p = float(cal.predict_proba(Xi.values)[:, 1][0])
								p_emerge[k] = p
							except Exception as e:
								logger.warning(f"Emergence inference failed for {fname}: {e}")
					# Exit hazard for current cluster
					hcal_path = os.path.join(models_dir, f"hazard_cluster_{cid}_calibrator.joblib")
					if cid >= 0 and os.path.exists(hcal_path):
						try:
							cal_h = joblib.load(hcal_path)
							keep_cols_h = _build_keep_cols(X_all, cid)
							Xh = X_last[keep_cols_h].fillna(0.0) if keep_cols_h else X_last.fillna(0.0)
							exit_hazard = float(cal_h.predict_proba(Xh.values)[:, 1][0])
						except Exception as e:
							logger.warning(f"Hazard inference failed for cluster {cid}: {e}")
	except Exception as e:
		logger.warning(f"Forecasting inference failed: {e}")
	return {
		"cluster_id": cid,
		"intensities": intensities,
		"p_emerge": p_emerge,
		"exit_hazard": exit_hazard,
		"timestamp": ts.isoformat() if hasattr(ts, "isoformat") else None,
	}