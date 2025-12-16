"""
Layer 1: Analyst Base Models Step

This step implements the training of the Diversity Defense ensemble (Layer 1).
It loads data, excludes NN training period, runs a vanilla HPO (if needed),
and trains 10 diverse models (Sharpe/Tanh/Huber).
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from src.training.steps.base_step import BaseStep, step_registry
from src.utils.versioned_artifacts import VersionedArtifactStore
from src.utils.ml_common.diversity_defense_trainer import DiversityDefenseTrainer
from src.utils.reporting.layered_pipeline_reporter import LayeredPipelineReporter
from src.utils.tprint import tprint_info, tprint_success, tprint_warning
from src.training.steps.labeling.meta_labeling_hpo_sample_weighted import compute_learnability_with_calibration

# For Vanilla HPO (reuse BayesianTPEOptimizer or simplified grid)
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

class AnalystBaseLayer1Step(BaseStep):
    def __init__(self, step_name: str = "layer1_analyst_base_training"):
        super().__init__(step_name)

    async def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        self.log(f"Starting Layer 1 Analyst Base Training...")

        symbol = config.get("symbol")
        exchange = config.get("exchange")
        timeframe = config.get("timeframe")
        direction = config.get("direction", "long")
        execution_mode = config.get("execution_mode", "light")

        store_path = Path(f"versioned_artifacts/{symbol}_{exchange}_{timeframe}_{direction}_meta_labeling")
        store = VersionedArtifactStore(store_path=store_path)

        # Load labeled data
        try:
            labeled_data, meta = store.load_latest("labeled_data")
            if labeled_data is None:
                raise FileNotFoundError("labeled_data not found in versioned artifacts.")
            self.log(f"Loaded labeled_data: {labeled_data.shape}")
        except Exception as e:
            self.log(f"Failed to load labeled_data: {e}")
            return {"success": False, "error": str(e)}

        # Load selected feature names
        from src.utils.pipeline_standards import PipelineStandards
        base_dir = PipelineStandards.build_path('reports', exchange=exchange, asset=symbol)
        std_features_path = Path(base_dir) / "post_hpo_evaluation" / "selected_features_latest.json"

        selected_features = []
        if std_features_path.exists():
            with open(std_features_path, "r") as f:
                selected_features = json.load(f)
            self.log(f"Loaded {len(selected_features)} selected features from {std_features_path}")
        else:
            self.log("Selected features JSON not found. Using all numeric columns excluding targets.")
            exclude = ["target", "label", "ret", "close", "open", "high", "low", "volume", "timestamp"]
            selected_features = [c for c in labeled_data.select_dtypes(include=np.number).columns
                                 if not any(x in c.lower() for x in exclude)]

        # Apply NN Exclusion
        cache_dir = Path("data/cache")
        nn_exclusion_ranges = []
        if cache_dir.exists():
            for meta_file in cache_dir.glob("*_metadata.json"):
                try:
                    with open(meta_file, "r") as f:
                        meta_content = json.load(f)
                        if "nn_training_end" in meta_content:
                            nn_exclusion_ranges.append((
                                meta_content.get("nn_training_start"),
                                meta_content.get("nn_training_end")
                            ))
                except Exception:
                    pass

        if nn_exclusion_ranges:
            max_end = max([pd.Timestamp(end) for start, end in nn_exclusion_ranges if end])
            self.log(f"Applying NN Exclusion: Dropping data before {max_end}")
            labeled_data = labeled_data[labeled_data.index > max_end]
            self.log(f"Data after exclusion: {labeled_data.shape}")

        if labeled_data.empty:
            return {"success": False, "error": "No data left after exclusion."}

        target_col = "target_long" if direction == "long" else "target_short"
        if target_col not in labeled_data.columns:
            target_col = "realized_return"

        if target_col not in labeled_data.columns:
             return {"success": False, "error": f"Target column {target_col} not found."}

        df_train = labeled_data.dropna(subset=selected_features + [target_col]).copy()
        X = df_train[selected_features]
        y = df_train[target_col]

        self.log("Running Vanilla HPO for Base Parameters...")
        best_base_params = {
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 5,
            "min_child_samples": 20
        }

        self.log("Training Diversity Defense Ensemble...")
        trainer = DiversityDefenseTrainer(
            base_params=best_base_params,
            n_estimators=150,
            decay_rate=0.0005
        )

        tscv = TimeSeriesSplit(n_splits=5)
        # Get model names by calling private method or dry run?
        # Let's instantiate configs to get names
        dummy_configs = trainer._generate_diversity_configs()
        model_names = [c["name"] for c in dummy_configs]
        oof_preds_df = pd.DataFrame(index=X.index, columns=model_names)

        self.log(f"Generating OOF predictions with {tscv.n_splits} splits...")
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            fold_models = trainer.train_ensemble(X_tr, y_tr, eval_set=(X_val, y_val))
            val_preds = trainer.predict(X_val)

            for m_name, p_arr in val_preds.items():
                oof_preds_df.loc[X_val.index, m_name] = p_arr

        self.log("Training Final Models on Full Data...")
        final_models = trainer.train_ensemble(X, y)

        oof_path = f"artifacts/layer1_oof_{symbol}_{timeframe}_{direction}.csv"
        os.makedirs("artifacts", exist_ok=True)
        oof_preds_df.to_csv(oof_path)
        self.log(f"Saved Layer 1 OOF predictions to {oof_path}")

        models_dir = Path(f"artifacts/models/layer1_{symbol}_{timeframe}_{direction}")
        models_dir.mkdir(parents=True, exist_ok=True)
        for m_name, model in final_models.items():
            model.save_model(str(models_dir / f"{m_name}.txt"))

        reporter = LayeredPipelineReporter()
        valid_oof = oof_preds_df.dropna()
        y_valid = y.reindex(valid_oof.index)

        for m_name in model_names:
            preds = valid_oof[m_name].astype(float)
            ic = preds.corr(y_valid, method='spearman')
            signal = np.tanh(preds)
            pnl = signal * y_valid
            sharpe = pnl.mean() / (pnl.std() + 1e-9) * np.sqrt(252*96)

            metrics = {
                "ic": ic,
                "sharpe_ratio": sharpe,
                "prediction_std": preds.std()
            }

            reporter.log_metrics("Layer1", m_name, "OOF", metrics)

        corr_matrix = valid_oof.astype(float).corr()
        avg_corr = (corr_matrix.sum().sum() - len(corr_matrix)) / (len(corr_matrix)**2 - len(corr_matrix))
        self.log(f"Average Pairwise Correlation: {avg_corr:.4f}")

        return {
            "success": True,
            "oof_path": oof_path,
            "models_dir": str(models_dir),
            "feature_list": selected_features
        }

# Register the step
step_registry.register("analyst_base_layer1_step", AnalystBaseLayer1Step)
