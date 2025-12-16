"""
Layer 2: Analyst Meta Model Step

Trains a meta-model on Layer 1 OOF predictions + disagreement features.
Compares modalities: Average, LGBM, ExtraTrees, Linear.
Generates Layer 2 OOF predictions.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from src.training.steps.base_step import BaseStep, step_registry
from src.utils.versioned_artifacts import VersionedArtifactStore
from src.utils.reporting.layered_pipeline_reporter import LayeredPipelineReporter

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, roc_auc_score
import lightgbm as lgb

class AnalystMetaLayer2Step(BaseStep):
    def __init__(self, step_name: str = "analyst_meta_layer2_step"):
        super().__init__(step_name)

    async def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        self.log(f"Starting Layer 2 Analyst Meta Training...")

        symbol = config.get("symbol")
        exchange = config.get("exchange")
        timeframe = config.get("timeframe")
        direction = config.get("direction", "long")

        # Load Layer 1 OOF
        l1_oof_path = f"artifacts/layer1_oof_{symbol}_{timeframe}_{direction}.csv"
        if not os.path.exists(l1_oof_path):
            return {"success": False, "error": f"Layer 1 OOF not found at {l1_oof_path}"}

        l1_oof = pd.read_csv(l1_oof_path, index_col=0, parse_dates=True)

        # Load Targets (from Labeled Data)
        store_path = Path(f"versioned_artifacts/{symbol}_{exchange}_{timeframe}_{direction}_meta_labeling")
        store = VersionedArtifactStore(store_path=store_path)
        labeled_data, _ = store.load_latest("labeled_data")

        target_col = "target_long" if direction == "long" else "target_short"
        if target_col not in labeled_data.columns:
            target_col = "realized_return" # Fallback

        # Align
        common_idx = l1_oof.index.intersection(labeled_data.index)
        l1_oof = l1_oof.loc[common_idx]
        y = labeled_data.loc[common_idx, target_col]

        # Generate Disagreement Features
        X_meta = l1_oof.copy()
        X_meta["mean_pred"] = l1_oof.mean(axis=1)
        X_meta["std_pred"] = l1_oof.std(axis=1)
        X_meta["min_pred"] = l1_oof.min(axis=1)
        X_meta["max_pred"] = l1_oof.max(axis=1)
        X_meta["range_pred"] = X_meta["max_pred"] - X_meta["min_pred"]

        # Train Meta Models
        models = {
            "average": None, # Simple average
            "linear": Ridge(alpha=1.0),
            "extratrees": ExtraTreesRegressor(n_estimators=100, max_depth=5, min_samples_leaf=20, n_jobs=1),
            "lgbm": lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, n_jobs=1)
        }

        tscv = TimeSeriesSplit(n_splits=5)
        oof_meta = pd.DataFrame(index=X_meta.index, columns=models.keys())

        reporter = LayeredPipelineReporter()

        self.log("Training Meta Models (CV)...")
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_meta)):
            X_tr, y_tr = X_meta.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X_meta.iloc[val_idx], y.iloc[val_idx]

            # 1. Average (No training)
            oof_meta.loc[X_val.index, "average"] = X_val["mean_pred"]

            # 2. Others
            for name, model in models.items():
                if name == "average": continue

                model.fit(X_tr, y_tr)
                pred = model.predict(X_val)
                oof_meta.loc[X_val.index, name] = pred

        # Evaluate and Select Best
        best_ic = -1
        best_model_name = "average"

        valid_idx = oof_meta.dropna().index
        y_valid = y.loc[valid_idx]

        for name in models.keys():
            preds = oof_meta.loc[valid_idx, name].astype(float)
            ic = preds.corr(y_valid, method='spearman')

            metrics = {"ic": ic, "prediction_std": preds.std()}
            reporter.log_metrics("Layer2", name, "OOF", metrics)

            if ic > best_ic:
                best_ic = ic
                best_model_name = name

        self.log(f"Best Meta Model: {best_model_name} (IC: {best_ic:.4f})")

        # Final Retrain on Full Data
        final_model = models[best_model_name]
        if best_model_name != "average":
            final_model.fit(X_meta, y)

        # Save OOF
        l2_oof_path = f"artifacts/layer2_oof_{symbol}_{timeframe}_{direction}.csv"
        oof_meta.to_csv(l2_oof_path)
        self.log(f"Saved Layer 2 OOF predictions to {l2_oof_path}")

        # Save Model (pickle or dedicated save)
        # For simplicity, we assume retraining in final step, but we save best config

        return {
            "success": True,
            "best_model": best_model_name,
            "l2_oof_path": l2_oof_path
        }

step_registry.register("analyst_meta_layer2_step", AnalystMetaLayer2Step)
