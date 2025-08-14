# src/training/steps/step1_8_regime_forecasting.py

import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
import joblib

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors
from src.training.steps.unified_data_loader import UnifiedDataLoader

# Timeframes to process
TIMEFRAMES: list[str] = ["1m", "5m", "15m"]


@dataclass
class EmergenceConfig:
	H: int = 5  # horizon bars
	persist_d: int = 3  # min sustained bars
	tau: float = 0.6  # activation threshold for p_k


@dataclass
class CVConfig:
	val_ratio: float = 0.2
	purge_bars: int = 10
	embargo_bars: int = 10


def _ewm_prob(ind: pd.Series, span: int = 3) -> pd.Series:
	return ind.astype(float).ewm(span=span, adjust=False).mean().clip(0.0, 1.0)


def _entropy(p_vec: pd.DataFrame) -> pd.Series:
	p = p_vec.clip(1e-9, 1.0)
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
	# Normalize to sum ~1
	s = p_df.sum(axis=1).replace(0, 1.0)
	p_df = p_df.div(s, axis=0)
	return p_df


def _mk_features(block_df: pd.DataFrame, composite_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[int]]:
	# Build p_k, slopes, accelerations, entropy; add block-level entropies
	cluster_ids = composite_df["composite_cluster_id"].astype(int)
	p_df = _build_p_k_matrix(cluster_ids)
	# Derivatives
	dp_df = p_df.diff().fillna(0.0).add_prefix("dp_")
	d2p_df = dp_df.diff().fillna(0.0).add_prefix("d2p_")
	# Entropy across clusters
	ent = _entropy(p_df)
	features = pd.concat([p_df, dp_df, d2p_df], axis=1)
	features["entropy"] = ent
	# Block entropies from posteriors if present
	for blk in ["momentum", "volatility", "liquidity", "microstructure"]:
		cols = [c for c in block_df.columns if c.startswith(f"{blk}_p_state_")]
		if cols:
			features[f"{blk}_entropy"] = _entropy(block_df[cols])
	# Transition matrix-derived next cluster probs (static per current state)
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
	return features, sorted([int(k) for k in p_df.columns.str.extract(r"p_k_(\d+)").dropna()[0].astype(int).unique()])


def _label_emergence(p_k: pd.Series, cfg: EmergenceConfig) -> pd.Series:
	# Label 1 if within next H bars p_k crosses >= tau and sustains for >= d bars
	n = len(p_k)
	label = np.zeros(n, dtype=int)
	vals = p_k.values
	for i in range(n):
		end = min(n, i + cfg.H + 1)
		segment = vals[i + 1 : end]
		if segment.size == 0:
			continue
		# find first index where sustained >= d above tau
		cnt = 0
		hit = False
		for v in segment:
			if v >= cfg.tau:
				cnt += 1
				if cnt >= cfg.persist_d:
					hit = True
					break
			else:
				cnt = 0
		label[i] = 1 if hit else 0
	return pd.Series(label, index=p_k.index)


def _label_exit_hazard(cluster_ids: pd.Series, k: int) -> pd.Series:
	vals = cluster_ids.values.astype(int)
	n = len(vals)
	haz = np.zeros(n, dtype=int)
	for i in range(n - 1):
		if vals[i] == k:
			haz[i] = 1 if vals[i + 1] != k else 0
	return pd.Series(haz, index=cluster_ids.index)


def _walkforward_split(df: pd.DataFrame, val_ratio: float) -> Tuple[pd.Index, pd.Index]:
	cut = int(len(df) * (1.0 - val_ratio))
	train_idx = df.index[:cut]
	val_idx = df.index[cut:]
	return train_idx, val_idx


def _train_calibrated_lgbm(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> Tuple[Any, Any, float]:
	clf = LGBMClassifier(
		n_estimators=400,
		learning_rate=0.05,
		max_depth=-1,
		n_jobs=4,
		colsample_bytree=0.8,
		subsample=0.8,
		reg_alpha=0.0,
		reg_lambda=0.0,
		objective="binary",
	)
	clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="auc", verbose=False)
	# Platt calibration (sigmoid)
	proba_val = clf.predict_proba(X_val)[:, 1]
	cal = CalibratedClassifierCV(base_estimator=clf, cv="prefit", method="sigmoid")
	cal.fit(X_val, y_val)
	auc = float(roc_auc_score(y_val, cal.predict_proba(X_val)[:, 1]))
	return clf, cal, auc


def _persist_model(path_dir: str, name: str, model: Any, calibrator: Any, meta: dict[str, Any]) -> None:
	os.makedirs(path_dir, exist_ok=True)
	joblib.dump(model, os.path.join(path_dir, f"{name}_model.joblib"))
	joblib.dump(calibrator, os.path.join(path_dir, f"{name}_calibrator.joblib"))
	with open(os.path.join(path_dir, f"{name}_meta.json"), "w") as f:
		json.dump(meta, f, indent=2)


@handle_errors(exceptions=(Exception,), default_return=False, context="step1_8_regime_forecasting")
async def run_step(
	symbol: str,
	exchange: str = "BINANCE",
	data_dir: str = "data/training",
	timeframe: str = "1m",
	lookback_days: int | None = None,
	H: int = 5,
	persist_d: int = 3,
	tau: float = 0.6,
	**kwargs: Any,
) -> bool:
	"""
	Step 1_8: Train emergence and exit-risk models for composite regimes using HMM-derived features.
	Produces calibrated LightGBM models per cluster per timeframe.
	"""
	logger = system_logger.getChild("Step1_8.RegimeForecasting")
	logger.info("🚀 Step 1_8: Regime Forecasting (emergence & exit)")

	cfg = EmergenceConfig(H=H, persist_d=persist_d, tau=tau)
	loader = UnifiedDataLoader({})
	any_success = False

	for tf in TIMEFRAMES:
		logger.info(f"🔄 Timeframe: {tf}")
		block_path = os.path.join(data_dir, f"{exchange}_{symbol}_hmm_block_states_{tf}.parquet")
		comp_path = os.path.join(data_dir, f"{exchange}_{symbol}_hmm_composite_clusters_{tf}.parquet")
		if not (os.path.exists(block_path) and os.path.exists(comp_path)):
			logger.warning(f"Artifacts missing for {tf}; skipping")
			continue
		block_df = pd.read_parquet(block_path)
		comp_df = pd.read_parquet(comp_path)
		# Ensure timestamp index
		for d in (block_df, comp_df):
			if "timestamp" in d.columns:
				d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce", utc=True)
				d.set_index("timestamp", inplace=True)
				d.sort_index(inplace=True)
		# Align
		idx = block_df.index.intersection(comp_df.index)
		block_df = block_df.reindex(idx)
		comp_df = comp_df.reindex(idx)
		# Build features
		X_all, clusters = _mk_features(block_df, comp_df)
		cluster_ids = comp_df["composite_cluster_id"].astype(int)
		# Train per cluster models
		models_dir = os.path.join("checkpoints", "regime_forecasting", exchange, symbol, tf)
		for k in clusters:
			p_k = X_all.get(f"p_k_{k}")
			if p_k is None:
				continue
			# Emergence dataset (all times)
			y_e = _label_emergence(p_k, cfg)
			X_e = X_all.copy()
			# Basic selection: keep p_k and its diffs, entropy and p_next
			keep_cols = [c for c in X_e.columns if c.startswith((f"p_k_{k}", f"dp_p_k_{k}", f"d2p_p_k_{k}", "entropy", "p_next_", "momentum_entropy", "volatility_entropy", "liquidity_entropy", "microstructure_entropy"))]
			if not keep_cols:
				keep_cols = list(X_e.columns)
			X_e = X_e[keep_cols].fillna(0.0)
			# Split
			tr_idx, vl_idx = _walkforward_split(X_e, val_ratio=0.2)
			clf_e, cal_e, auc_e = _train_calibrated_lgbm(X_e.loc[tr_idx], y_e.loc[tr_idx], X_e.loc[vl_idx], y_e.loc[vl_idx])
			_persist_model(models_dir, f"emergence_cluster_{k}", clf_e, cal_e, {"auc_val": auc_e, "cluster": k, "timeframe": tf})
			logger.info(f"✅ Emergence model saved for cluster {k} (AUC={auc_e:.3f})")
			# Exit hazard dataset (only while in k)
			mask_in_k = (cluster_ids == k)
			if mask_in_k.sum() < 50:
				logger.info(f"Insufficient samples for hazard model cluster {k}")
				continue
			y_h = _label_exit_hazard(cluster_ids, k)
			X_h = X_all.loc[mask_in_k].copy()
			y_h = y_h.loc[mask_in_k]
			keep_cols_h = [c for c in X_h.columns if c.startswith((f"p_k_{k}", f"dp_p_k_{k}", f"d2p_p_k_{k}", "entropy", "p_next_", "momentum_entropy", "volatility_entropy", "liquidity_entropy", "microstructure_entropy"))]
			if not keep_cols_h:
				keep_cols_h = list(X_h.columns)
			X_h = X_h[keep_cols_h].fillna(0.0)
			tr_idx_h, vl_idx_h = _walkforward_split(X_h, val_ratio=0.2)
			clf_h, cal_h, auc_h = _train_calibrated_lgbm(X_h.loc[tr_idx_h], y_h.loc[tr_idx_h], X_h.loc[vl_idx_h], y_h.loc[vl_idx_h])
			_persist_model(models_dir, f"hazard_cluster_{k}", clf_h, cal_h, {"auc_val": auc_h, "cluster": k, "timeframe": tf})
			logger.info(f"✅ Hazard model saved for cluster {k} (AUC={auc_h:.3f})")
		any_success = True

	logger.info("✅ Step 1_8 completed" if any_success else "⚠️ Step 1_8 produced no models")
	return any_success