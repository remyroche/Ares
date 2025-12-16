"""
Layer 3: Gate Model Step

Trains a Gate Model to filter Layer 2 predictions.
Compares ExtraTreesClassifier and RidgeClassifier.
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
from sklearn.linear_model import RidgeClassifier
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

        # Define Models
        models = {
            "ExtraTrees": ExtraTreesClassifier(n_estimators=100, max_depth=5, min_samples_leaf=30, n_jobs=1, class_weight="balanced"),
            "Ridge": RidgeClassifier(class_weight="balanced", alpha=1.0)
        }

        tscv = TimeSeriesSplit(n_splits=5)
        # DataFrame to store OOF preds for each model
        gate_oof_df = pd.DataFrame(index=X_gate.index, columns=models.keys())

        reporter = LayeredPipelineReporter()

        self.log("Training Gate Models (CV)...")
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_gate)):
            X_tr, y_tr = X_gate.iloc[train_idx], y_gate.iloc[train_idx]
            X_val = X_gate.iloc[val_idx]

            for name, model in models.items():
                model.fit(X_tr, y_tr)

                # Predict scores
                if hasattr(model, "predict_proba"):
                    preds = model.predict_proba(X_val)[:, 1]
                else:
                    # RidgeClassifier has decision_function
                    preds = model.decision_function(X_val)
                    # Normalize to 0-1 sigmoid-like for comparability if needed,
                    # but for ranking (AUC) raw score is fine.
                    # For consistency with ExtraTrees (0-1), we might want to sigmoid it?
                    # But AUC doesn't care. We'll store raw scores.

                gate_oof_df.loc[X_val.index, name] = preds

        # Evaluation and Selection
        best_auc = -1
        best_model_name = "ExtraTrees" # Default

        valid_idx = gate_oof_df.dropna().index
        y_valid = y_gate.reindex(valid_idx)

        for name in models.keys():
            preds = gate_oof_df.loc[valid_idx, name].astype(float)

            # Calculate AUC
            try:
                auc = roc_auc_score(y_valid, preds)
            except ValueError:
                auc = 0.5

            # Calculate Precision @ Top 40% (proxy for confidence > threshold)
            # Dynamic thresholding based on quantile
            threshold = preds.quantile(0.60)
            binary_preds = (preds > threshold).astype(int)
            prec = precision_score(y_valid, binary_preds, zero_division=0)

            metrics = {
                "auc_roc": auc,
                "precision_at_40pct": prec,
                "prediction_std": preds.std()
            }
            reporter.log_metrics("Layer3", f"Gate_{name}", "OOF", metrics)
            self.log(f"Model {name}: AUC={auc:.4f}, Prec@40%={prec:.4f}")

            if auc > best_auc:
                best_auc = auc
                best_model_name = name

        self.log(f"Best Gate Model: {best_model_name} (AUC: {best_auc:.4f})")

        # Save Winner OOF
        l3_oof_path = f"artifacts/layer3_oof_{symbol}_{timeframe}_{direction}.csv"
        # Save as a Series (single column) for compatibility with downstream
        gate_oof_df[best_model_name].to_csv(l3_oof_path)

        return {
            "success": True,
            "l3_oof_path": l3_oof_path,
            "gate_auc": best_auc,
            "best_model": best_model_name
        }

step_registry.register("gate_layer3_step", GateLayer3Step)
