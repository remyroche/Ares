# src/training/steps/step1_7_hmm_regime_discovery_validator.py

import os
import pandas as pd
from typing import Any

from src.utils.logger import system_logger


def _exists(path: str) -> bool:
	try:
		return os.path.exists(path)
	except Exception:
		return False


def run_validator(training_input: dict[str, Any], pipeline_state: dict[str, Any]) -> dict[str, Any]:
	logger = system_logger.getChild("Step1_7.Validator")
	symbol = training_input.get("symbol", "ETHUSDT")
	exchange = training_input.get("exchange", "BINANCE")
	data_dir = training_input.get("data_dir", "data/training")

	timeframes = ["1m", "5m", "15m", "30m"]
	found_any = False
	messages: list[str] = []
	for tf in timeframes:
		block_path = os.path.join(data_dir, f"{exchange}_{symbol}_hmm_block_states_{tf}.parquet")
		comp_path = os.path.join(data_dir, f"{exchange}_{symbol}_hmm_composite_clusters_{tf}.parquet")
		if _exists(block_path) and _exists(comp_path):
			try:
				df_b = pd.read_parquet(block_path)
				df_c = pd.read_parquet(comp_path)
				req_cols = [c for c in df_b.columns if c.endswith("_state_id")] + ["combination_id", "composite_cluster_id"]
				if df_c is not None:
					req_cols = [c for c in req_cols if c in (list(df_b.columns) + list(df_c.columns))]
				if req_cols:
					found_any = True
					messages.append(f"OK {tf}: artifacts present, rows={len(df_b)}")
				else:
					messages.append(f"WARN {tf}: missing expected columns in outputs")
			except Exception as e:
				messages.append(f"WARN {tf}: failed to read outputs: {e}")
		else:
			messages.append(f"WARN {tf}: artifacts missing")

	passed = found_any
	status = {
		"validation_passed": bool(passed),
		"messages": messages,
	}
	if passed:
		logger.info("✅ Step 1_7 validation passed for at least one timeframe")
	else:
		logger.warning("⚠️ Step 1_7 validation did not find valid artifacts")
	return status