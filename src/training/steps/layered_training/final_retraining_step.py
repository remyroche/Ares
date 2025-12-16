"""
Final Retraining Step

Retrains the entire hierarchy for production:
1. Layer 1 (Base): Trained on 100% Raw Data.
2. Layer 2 (Meta): Trained on 100% Layer 1 OOF Predictions.
3. Layer 3 (Gate): Trained on 100% Layer 2 OOF Predictions.

Crucially, Meta and Gate models are trained on the noisy OOF predictions
generated during the cross-validation steps, NOT on the predictions of
the fully trained Layer 1 models (which would look too perfect).
"""

import os
import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, Any

from src.training.steps.base_step import BaseStep, step_registry
from src.utils.versioned_artifacts import VersionedArtifactStore
from src.utils.ml_common.diversity_defense_trainer import DiversityDefenseTrainer

# Meta/Gate Models
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
import lightgbm as lgb

class FinalRetrainingStep(BaseStep):
    def __init__(self, step_name: str = "final_retraining_step"):
        super().__init__(step_name)

    async def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        self.log("Starting Final Retraining...")

        symbol = config.get("symbol")
        exchange = config.get("exchange")
        timeframe = config.get("timeframe")
        direction = config.get("direction", "long")

        # 1. Retrain Layer 1 (Base) on 100% Data
        # --------------------------------------
        self.log("Retraining Layer 1 (Base) on 100% Data...")
        # Load Data (Same as Layer 1 step)
        store_path = f"versioned_artifacts/{symbol}_{exchange}_{timeframe}_{direction}_meta_labeling"
        store = VersionedArtifactStore(store_path=store_path)
        labeled_data, _ = store.load_latest("labeled_data")

        # Load feature list (assume saved by L1 step or standard loc)
        # For robustness, we re-derive or load.
        # Ideally, passed via context, but we'll re-load from reports.
        from src.utils.pipeline_standards import PipelineStandards
        base_dir = PipelineStandards.build_path('reports', exchange=exchange, asset=symbol)
        std_features_path = Path(base_dir) / "post_hpo_evaluation" / "selected_features_latest.json"

        import json
        if std_features_path.exists():
            with open(std_features_path, "r") as f:
                features = json.load(f)
        else:
            return {"success": False, "error": "Selected features not found for final retraining."}

        target_col = "target_long" if direction == "long" else "target_short"
        if target_col not in labeled_data.columns:
            target_col = "realized_return"

        df_train = labeled_data.dropna(subset=features + [target_col])
        X = df_train[features]
        y = df_train[target_col]

        # Instantiate Trainer (Params should be loaded from best_params, using defaults for now)
        base_params = {"learning_rate": 0.05, "num_leaves": 31}
        trainer = DiversityDefenseTrainer(base_params=base_params, n_estimators=150)

        final_base_models = trainer.train_ensemble(X, y)

        # Save Base Models
        prod_dir = Path(f"production_models/{symbol}_{timeframe}_{direction}")
        prod_dir.mkdir(parents=True, exist_ok=True)

        for name, model in final_base_models.items():
            model.save_model(str(prod_dir / f"layer1_{name}.txt"))
        self.log(f"Saved {len(final_base_models)} Layer 1 models to {prod_dir}")

        # 2. Retrain Layer 2 (Meta) on 100% OOF Data
        # -----------------------------------------
        self.log("Retraining Layer 2 (Meta) on 100% OOF Data...")
        l1_oof_path = f"artifacts/layer1_oof_{symbol}_{timeframe}_{direction}.csv"
        if not os.path.exists(l1_oof_path):
             return {"success": False, "error": "Layer 1 OOF missing for final training."}

        l1_oof = pd.read_csv(l1_oof_path, index_col=0, parse_dates=True)
        common_idx = l1_oof.index.intersection(y.index)

        X_meta = l1_oof.loc[common_idx].copy()
        # Add Disagreement Features
        X_meta["mean_pred"] = X_meta.mean(axis=1)
        X_meta["std_pred"] = X_meta.std(axis=1)
        X_meta["min_pred"] = X_meta.min(axis=1)
        X_meta["max_pred"] = X_meta.max(axis=1)
        X_meta["range_pred"] = X_meta["max_pred"] - X_meta["min_pred"]

        y_meta = y.loc[common_idx]

        # Train Final Meta Model (e.g. LGBM)
        # In Layer 2 step, we might have selected the best.
        # Here we assume LGBM or Average. If Average, no training needed.
        # We'll train an LGBM Regressor as the default "smart" meta model.
        meta_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, n_jobs=1)
        meta_model.fit(X_meta, y_meta)

        joblib.dump(meta_model, prod_dir / "layer2_meta_lgbm.pkl")
        self.log("Saved Layer 2 Meta Model.")

        # 3. Retrain Layer 3 (Gate) on 100% OOF Data
        # -----------------------------------------
        self.log("Retraining Layer 3 (Gate) on 100% OOF Data...")
        l2_oof_path = f"artifacts/layer2_oof_{symbol}_{timeframe}_{direction}.csv"
        if not os.path.exists(l2_oof_path):
             return {"success": False, "error": "Layer 2 OOF missing for final training."}

        l2_oof = pd.read_csv(l2_oof_path, index_col=0, parse_dates=True)
        common_idx_l3 = l2_oof.index.intersection(labeled_data.index)

        l2_oof_subset = l2_oof.loc[common_idx_l3]

        # Regime Features
        regime_cols = [c for c in labeled_data.columns if "regime" in c or "vol" in c][:10]
        X_gate = pd.concat([l2_oof_subset, labeled_data.loc[common_idx_l3, regime_cols]], axis=1).fillna(0)

        y_gate_target = (labeled_data.loc[common_idx_l3, target_col] > 0.001).astype(int)

        gate_model = ExtraTreesClassifier(n_estimators=100, max_depth=5, min_samples_leaf=30, n_jobs=1, class_weight="balanced")
        gate_model.fit(X_gate, y_gate_target)

        joblib.dump(gate_model, prod_dir / "layer3_gate_et.pkl")
        self.log("Saved Layer 3 Gate Model.")

        return {
            "success": True,
            "prod_dir": str(prod_dir),
            "models_count": len(final_base_models) + 2
        }

step_registry.register("final_retraining_step", FinalRetrainingStep)
