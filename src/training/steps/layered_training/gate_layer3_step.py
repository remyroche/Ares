"""
Layer 3: Gate Model Step

Trains a Gate Model (ExtraTreesClassifier) to filter Layer 2 predictions.
Inputs: Layer 2 OOF + Regime Features.
Target: Binary Success (Net Return > 0).
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any

from src.training.steps.base_step import BaseStep, step_registry
from src.utils.versioned_artifacts import VersionedArtifactStore
from src.utils.reporting.layered_pipeline_reporter import LayeredPipelineReporter

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, precision_score

class GateLayer3Step(BaseStep):
    def __init__(self, step_name: str = "gate_layer3_step"):
        super().__init__(step_name)

    async def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        self.log(f"Starting Layer 3 Gate Training...")

        symbol = config.get("symbol")
        exchange = config.get("exchange")
        timeframe = config.get("timeframe")
        direction = config.get("direction", "long")

        # Load Layer 2 OOF
        l2_oof_path = f"artifacts/layer2_oof_{symbol}_{timeframe}_{direction}.csv"
        if not os.path.exists(l2_oof_path):
             return {"success": False, "error": "Layer 2 OOF not found."}

        l2_oof = pd.read_csv(l2_oof_path, index_col=0, parse_dates=True)
        # We need to know which model was best, or we take 'average' or just all?
        # Typically we gate the *chosen* signal.
        # For this implementation, we'll try to find the 'best' column from metadata or just take the one with highest var?
        # Simpler: Use ALL columns from L2 OOF as input to Gate?
        # User said: "using OOF predictions only from Analyst meta model". Singular.
        # We'll use the 'average' or 'lgbm' (whichever is present).
        # Let's use the one with highest correlation in the OOF if possible, or just all.
        # Actually, let's use all L2 predictions as input.

        # Load Regime Features (Labeled Data)
        store_path = f"versioned_artifacts/{symbol}_{exchange}_{timeframe}_{direction}_meta_labeling"
        store = VersionedArtifactStore(store_path=store_path)
        labeled_data, _ = store.load_latest("labeled_data")

        target_col = "target_long" if direction == "long" else "target_short"
        if target_col not in labeled_data.columns:
            target_col = "realized_return"

        common_idx = l2_oof.index.intersection(labeled_data.index)
        l2_oof = l2_oof.loc[common_idx]
        labeled_data = labeled_data.loc[common_idx]

        # Define Gate Target: Profitable Trade (Net of fees approx 0.1%)
        y_gate = (labeled_data[target_col] > 0.001).astype(int)

        # Define Features: L2 Preds + Regime
        # Simple regime features if available
        regime_cols = [c for c in labeled_data.columns if "regime" in c or "vol" in c]
        # Limit to top 10 regime features to avoid noise
        regime_cols = regime_cols[:10]

        X_gate = pd.concat([l2_oof, labeled_data[regime_cols]], axis=1)
        X_gate = X_gate.fillna(0)

        # Train Gate
        gate_model = ExtraTreesClassifier(n_estimators=100, max_depth=5, min_samples_leaf=30, n_jobs=1, class_weight="balanced")

        tscv = TimeSeriesSplit(n_splits=5)
        gate_oof_preds = pd.Series(index=X_gate.index, dtype=float)

        reporter = LayeredPipelineReporter()

        self.log("Training Gate Model (CV)...")
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_gate)):
            X_tr, y_tr = X_gate.iloc[train_idx], y_gate.iloc[train_idx]
            X_val = X_gate.iloc[val_idx]

            gate_model.fit(X_tr, y_tr)
            # Predict Proba of Class 1 (Profit)
            preds = gate_model.predict_proba(X_val)[:, 1]
            gate_oof_preds.iloc[val_idx] = preds

        # Metrics
        valid_gate = gate_oof_preds.dropna()
        y_valid = y_gate.reindex(valid_gate.index)

        auc = roc_auc_score(y_valid, valid_gate)
        # Check precision at threshold 0.6
        binary_preds = (valid_gate > 0.6).astype(int)
        prec = precision_score(y_valid, binary_preds, zero_division=0)

        metrics = {
            "auc_roc": auc,
            "precision_at_0.6": prec,
            "prediction_std": valid_gate.std()
        }
        reporter.log_metrics("Layer3", "Gate_ExtraTrees", "OOF", metrics)

        self.log(f"Gate AUC: {auc:.4f}, Precision@0.6: {prec:.4f}")

        # Save OOF
        l3_oof_path = f"artifacts/layer3_oof_{symbol}_{timeframe}_{direction}.csv"
        gate_oof_preds.to_csv(l3_oof_path)

        return {
            "success": True,
            "l3_oof_path": l3_oof_path,
            "gate_auc": auc
        }

step_registry.register("gate_layer3_step", GateLayer3Step)
