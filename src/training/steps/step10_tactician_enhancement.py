# src/training/steps/step10_tactician_enhancement.py

import asyncio
import json
import os
import pickle
from datetime import datetime
from typing import Any

import numpy as np
import optuna
import pandas as pd

from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error,
)
from src.training.steps.unified_data_loader import get_unified_data_loader


class TacticianEnhancementStep:
    """Step 10: Tactician Models Enhancement (1m timeframe).

    - Robust Feature Selection: mutual information, permutation importance, model-based importance
    - Hyperparameter Optimization (Optuna) for LightGBM, XGBoost, CatBoost (lightweight trials)
    - Final Retraining with best feature subset and hyperparameters
    - Advanced Optimization: probability threshold tuning on validation split
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger

    @handle_errors(exceptions=(Exception,), default_return=False, context="tactician enhancement step initialization")
    async def initialize(self) -> bool:
        self.logger.info("Initializing Tactician Enhancement Step (Step 10)...")
        return True

    @handle_errors(exceptions=(Exception,), default_return={"status": "FAILED"}, context="tactician enhancement execute")
    async def execute(self, training_input: dict[str, Any], pipeline_state: dict[str, Any]) -> dict[str, Any]:
        start = datetime.now()
        symbol = training_input.get("symbol", "ETHUSDT")
        exchange = training_input.get("exchange", "BINANCE")
        data_dir = training_input.get("data_dir", "data/training")
        timeframe = "1m"

        # 1) Load tactician labeled data (1m)
        X_train, X_val, y_train, y_val, feature_names = await self._load_training_validation(symbol, exchange, data_dir, timeframe)
        if X_train is None:
            msg = "No tactician training data available for enhancement"
            self.logger.error(msg)
            return {"status": "FAILED", "error": msg}

        # 2) Feature Selection (robust, SHAP-free)
        selected_features = await self._select_features(X_train, y_train, feature_names)
        Xtr = X_train[selected_features].astype(np.float32)
        Xva = X_val[selected_features].astype(np.float32)

        # 3) HPO for core models (lightweight)
        models = {}
        try:
            models["lightgbm"] = await self._hpo_lightgbm(Xtr, y_train, Xva, y_val)
        except Exception as e:
            self.logger.warning(f"LGBM HPO skipped: {e}")
        try:
            models["xgboost"] = await self._hpo_xgboost(Xtr, y_train, Xva, y_val)
        except Exception as e:
            self.logger.warning(f"XGB HPO skipped: {e}")
        try:
            models["catboost"] = await self._hpo_catboost(Xtr, y_train, Xva, y_val)
        except Exception as e:
            self.logger.warning(f"CatBoost HPO skipped: {e}")

        # 4) Final Retraining (fit on full training features, evaluate on validation)
        enhanced_dir = os.path.join(data_dir, "tactician_enhanced_models")
        os.makedirs(enhanced_dir, exist_ok=True)
        summary = {"timeframe": timeframe, "models": {}, "selected_features": selected_features}

        for name, pkg in models.items():
            if not pkg:
                continue
            model = pkg["model"]
            threshold = pkg.get("best_threshold", 0.5)
            # Evaluate
            y_proba = model.predict_proba(Xva)[:, -1] if hasattr(model, "predict_proba") else model.decision_function(Xva)
            y_hat = (y_proba >= threshold).astype(int)
            acc = float((y_hat == y_val).mean())
            pkg["validation_accuracy"] = acc
            pkg["selected_threshold"] = threshold
            # Persist
            out_path = os.path.join(enhanced_dir, f"{exchange}_{symbol}_{timeframe}_{name}.pkl")
            with open(out_path, "wb") as f:
                pickle.dump(pkg, f)
            summary["models"][name] = {
                "path": out_path,
                "accuracy": acc,
                "hyperparameters": pkg.get("hyperparameters", {}),
                "threshold": threshold,
            }

        # 5) Write summary
        sum_file = os.path.join(enhanced_dir, f"{exchange}_{symbol}_{timeframe}_enhancement_summary.json")
        with open(sum_file, "w") as f:
            json.dump(summary, f, indent=2)

        duration = (datetime.now() - start).total_seconds()
        self.logger.info(f"✅ Step 10 Tactician Enhancement complete in {duration:.2f}s; saved to {enhanced_dir}")
        pipeline_state["tactician_enhanced_models"] = summary
        return {"status": "SUCCESS", "enhanced_models_dir": enhanced_dir, "duration": duration}

    async def _load_training_validation(self, symbol: str, exchange: str, data_dir: str, timeframe: str):
        try:
            # Prefer Parquet dataset scan
            part_base = os.path.join(data_dir, "parquet", "labeled")
            if os.path.isdir(part_base):
                from src.training.enhanced_training_manager_optimized import ParquetDatasetManager

                pdm = ParquetDatasetManager(logger=self.logger)
                filters_tr = [("exchange", "==", exchange), ("symbol", "==", symbol), ("timeframe", "==", timeframe), ("split", "==", "train")]
                filters_va = [("exchange", "==", exchange), ("symbol", "==", symbol), ("timeframe", "==", timeframe), ("split", "==", "validation")]
                label_col = self.config.get("label_column", "label")
                feat_cols = self.config.get("feature_columns")
                cols = ["timestamp", label_col]
                if isinstance(feat_cols, list) and len(feat_cols) > 0:
                    cols = ["timestamp", *feat_cols, label_col]
                Xy_tr = pdm.cached_projection(base_dir=part_base, filters=filters_tr, columns=cols, cache_dir="data_cache/projections", cache_key_prefix=f"tact_tr_{exchange}_{symbol}")
                Xy_va = pdm.cached_projection(base_dir=part_base, filters=filters_va, columns=cols, cache_dir="data_cache/projections", cache_key_prefix=f"tact_va_{exchange}_{symbol}")
                df_tr = pd.DataFrame(Xy_tr)
                df_va = pd.DataFrame(Xy_va)
            else:
                # Fallback to labeled pickle produced by Step 8
                pkl = os.path.join(data_dir, "tactician_labeled_data", f"{exchange}_{symbol}_tactician_labeled.pkl")
                if not os.path.exists(pkl):
                    return None, None, None, None, []
                df = pickle.load(open(pkl, "rb"))
                # Simple split
                split = int(len(df) * 0.8)
                df_tr, df_va = df.iloc[:split].copy(), df.iloc[split:].copy()
            # Build features
            label_col = self.config.get("label_column", "label")
            context_cols = {"timestamp", "exchange", "symbol", "timeframe", "year", "month", "day"}
            use_cols = [c for c in df_tr.columns if c not in context_cols and c != label_col]
            X_train = df_tr[use_cols].select_dtypes(include=[np.number]).fillna(0)
            X_val = df_va[use_cols].select_dtypes(include=[np.number]).fillna(0)
            y_train = df_tr[label_col].astype(int)
            y_val = df_va[label_col].astype(int)
            return X_train, X_val, y_train, y_val, list(X_train.columns)
        except Exception as e:
            self.logger.error(f"Failed to load training/validation: {e}")
            return None, None, None, None, []

    async def _select_features(self, X: pd.DataFrame, y: pd.Series, feature_names: list[str]) -> list[str]:
        try:
            # 1) Mutual Information top-k
            from sklearn.feature_selection import mutual_info_classif
            mi = mutual_info_classif(X.values, y.values, random_state=42)
            k = min(len(feature_names), 150)
            top_mi_idx = np.argsort(mi)[-k:]
            cand = [feature_names[i] for i in top_mi_idx]
            X_cand = X[cand]

            # 2) Permutation importance with a small RF
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.inspection import permutation_importance

            rf = RandomForestClassifier(n_estimators=150, max_depth=10, n_jobs=-1, random_state=42)
            rf.fit(X_cand, y)
            perm = permutation_importance(rf, X_cand, y, n_repeats=5, random_state=42, n_jobs=-1)
            pi = perm.importances_mean
            top_pi_idx = np.argsort(pi)[-min(len(cand), 120):]
            cand2 = [cand[i] for i in top_pi_idx]
            X_cand2 = X[cand2]

            # 3) Model-based importance via LightGBM
            import lightgbm as lgb

            lgbm = lgb.LGBMClassifier(n_estimators=300, num_leaves=64, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
            lgbm.fit(X_cand2, y)
            fi = getattr(lgbm, "feature_importances_", np.zeros(len(cand2)))
            top_fi_idx = np.argsort(fi)[-min(len(cand2), 100):]
            selected = [cand2[i] for i in top_fi_idx]
            return selected or feature_names[: min(len(feature_names), 60)]
        except Exception as e:
            self.logger.warning(f"Feature selection failed: {e}; using fallback")
            return feature_names[: min(len(feature_names), 80)]

    async def _hpo_lightgbm(self, Xtr: pd.DataFrame, ytr: pd.Series, Xva: pd.DataFrame, yva: pd.Series) -> dict[str, Any] | None:
        try:
            import lightgbm as lgb

            def objective(trial: optuna.Trial) -> float:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=200),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 31, 255),
                    "max_depth": trial.suggest_int("max_depth", 4, 10),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                }
                model = lgb.LGBMClassifier(random_state=42, verbose=-1, **params)
                model.fit(Xtr, ytr)
                pred = model.predict(Xva)
                return float((pred == yva).mean())

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=20)
            best = study.best_params
            model = lgb.LGBMClassifier(random_state=42, verbose=-1, **best)
            model.fit(Xtr, ytr)
            best_thr = self._tune_threshold(model, Xva, yva)
            return {"model": model, "hyperparameters": best, "best_threshold": best_thr}
        except Exception as e:
            self.logger.warning(f"LGBM HPO failed: {e}")
            return None

    async def _hpo_xgboost(self, Xtr: pd.DataFrame, ytr: pd.Series, Xva: pd.DataFrame, yva: pd.Series) -> dict[str, Any] | None:
        try:
            import xgboost as xgb

            def objective(trial: optuna.Trial) -> float:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=200),
                    "max_depth": trial.suggest_int("max_depth", 3, 8),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                }
                model = xgb.XGBClassifier(random_state=42, tree_method="hist", eval_metric="logloss", verbosity=0, **params)
                model.fit(Xtr, ytr)
                pred = model.predict(Xva)
                return float((pred == yva).mean())

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=20)
            best = study.best_params
            model = xgb.XGBClassifier(random_state=42, tree_method="hist", eval_metric="logloss", verbosity=0, **best)
            model.fit(Xtr, ytr)
            best_thr = self._tune_threshold(model, Xva, yva)
            return {"model": model, "hyperparameters": best, "best_threshold": best_thr}
        except Exception as e:
            self.logger.warning(f"XGB HPO failed: {e}")
            return None

    async def _hpo_catboost(self, Xtr: pd.DataFrame, ytr: pd.Series, Xva: pd.DataFrame, yva: pd.Series) -> dict[str, Any] | None:
        try:
            from catboost import CatBoostClassifier

            def objective(trial: optuna.Trial) -> float:
                params = {
                    "iterations": trial.suggest_int("iterations", 200, 1000, step=200),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "depth": trial.suggest_int("depth", 4, 10),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                }
                model = CatBoostClassifier(random_seed=42, verbose=False, **params)
                model.fit(Xtr, ytr)
                pred = model.predict(Xva)
                return float((pred == yva).mean())

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=20)
            best = study.best_params
            model = CatBoostClassifier(random_seed=42, verbose=False, **best)
            model.fit(Xtr, ytr)
            best_thr = self._tune_threshold(model, Xva, yva)
            return {"model": model, "hyperparameters": best, "best_threshold": best_thr}
        except Exception as e:
            self.logger.warning(f"CatBoost HPO failed: {e}")
            return None

    def _tune_threshold(self, model: Any, X: pd.DataFrame, y: pd.Series) -> float:
        # Simple threshold sweep to maximize accuracy (can switch to F1 or custom metric)
        try:
            if not hasattr(model, "predict_proba"):
                return 0.5
            proba = model.predict_proba(X)[:, -1]
            best, best_acc = 0.5, -1.0
            for thr in np.linspace(0.3, 0.7, 41):
                pred = (proba >= thr).astype(int)
                acc = float((pred == y).mean())
                if acc > best_acc:
                    best_acc, best = acc, thr
            return float(best)
        except Exception:
            return 0.5


async def run_step(symbol: str, exchange: str = "BINANCE", data_dir: str = "data/training", force_rerun: bool = False, **kwargs) -> bool:
    config = {"symbol": symbol, "exchange": exchange, "data_dir": data_dir}
    step = TacticianEnhancementStep(config)
    await step.initialize()
    training_input = {"symbol": symbol, "exchange": exchange, "data_dir": data_dir, **kwargs}
    pipeline_state: dict[str, Any] = {}
    result = await step.execute(training_input, pipeline_state)
    return result.get("status") == "SUCCESS"