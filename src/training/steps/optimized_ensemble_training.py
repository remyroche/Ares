# src/training/steps/optimized_ensemble_training.py

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from typing import Any, Dict, List, Optional
import logging
import time
import ray
from joblib import Memory

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Caching Setup ---
# Use joblib.Memory to cache the results of our training function.
# This prevents re-computation if the same model is trained on the same data.
CACHE_DIR = "./cachedir"
memory = Memory(CACHE_DIR, verbose=0)

# --- Ray Worker Function for Parallel Execution ---
@ray.remote
def train_and_evaluate_worker_ray(
    model_config: Dict[str, Any],
    X_train_ref: ray.ObjectRef,
    y_train_ref: ray.ObjectRef,
    X_val_ref: Optional[ray.ObjectRef] = None,
    y_val_ref: Optional[ray.ObjectRef] = None,
    early_stopping_patience: int = 10
) -> Dict[str, Any]:
    """
    A self-contained, cacheable Ray worker function to train a single model.
    It receives references to data in Ray's shared object store to avoid serialization costs.
    """
    # De-reference objects from Ray's object store
    X_train, y_train = ray.get(X_train_ref), ray.get(y_train_ref)
    X_val, y_val = (ray.get(X_val_ref), ray.get(y_val_ref)) if X_val_ref else (None, None)

    model_name = model_config["name"]
    model_class = model_config["class"]
    params = model_config["params"]
    params['n_jobs'] = 1 # Ensure model runs on a single core within the process
    if 'random_state' not in params:
        params['random_state'] = 42
    
    model = model_class(**params)

    # --- Caching Wrapper ---
    # We define the core logic inside a function that joblib can cache.
    # The hash is computed based on the model, its parameters, and the data's hash.
    @memory.cache
    def _cached_train(model_instance, X_tr, y_tr, X_va, y_va):
        try:
            if model_name in ['lightgbm', 'xgboost'] and X_va is not None and y_va is not None:
                if model_name == 'lightgbm':
                    model_instance.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                                       callbacks=[lgb.early_stopping(early_stopping_patience, verbose=False)])
                else:
                    model_instance.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            else:
                model_instance.fit(X_tr, y_tr)
            
            score = accuracy_score(y_va, model_instance.predict(X_va)) if X_va is not None else None
            return {"model": model_instance, "validation_accuracy": score, "error": None}
        except Exception as e:
            return {"model": None, "validation_accuracy": None, "error": str(e)}

    # Execute the cached function
    result = _cached_train(model, X_train, y_train, X_val, y_val)
    result["model_name"] = model_name
    return result


class HighPerformanceEnsembleManager:
    """
    Manages ensemble training using Ray for parallelization, Joblib for caching,
    and a sophisticated meta-learner for improved outcomes.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        if not ray.is_initialized():
            ray.init(logging_level=logging.ERROR)
            self.logger.info(f"Ray initialized successfully. Dashboard URL: {ray.dashboard_url}")
        
        self.logger.info(f"Caching enabled. Cache directory: {CACHE_DIR}")

    def train_base_models(
        self,
        model_configs: List[Dict[str, Any]],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> List[Dict[str, Any]]:
        """
        Trains multiple models in parallel using Ray for optimal performance.
        """
        self.logger.info(f"Starting parallel training for {len(model_configs)} models using Ray...")
        
        # Put data into Ray's object store once to be shared across all workers
        X_train_ref = ray.put(X_train)
        y_train_ref = ray.put(y_train)
        X_val_ref = ray.put(X_val)
        y_val_ref = ray.put(y_val)

        tasks = [
            train_and_evaluate_worker_ray.remote(config, X_train_ref, y_train_ref, X_val_ref, y_val_ref)
            for config in model_configs
        ]
        
        results = ray.get(tasks)
        
        trained_models = []
        for res in results:
            if res["error"]:
                self.logger.error(f"Training failed for {res['model_name']}: {res['error']}")
            else:
                self.logger.info(f"Successfully trained {res['model_name']}. Validation Accuracy: {res['validation_accuracy']:.4f}")
                trained_models.append(res)
        
        return trained_models

    def create_stacking_ensemble(
        self,
        trained_models: List[Dict[str, Any]],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv_folds: int = 5
    ) -> StackingClassifier:
        """
        Creates a StackingClassifier with a sophisticated XGBoost meta-learner.
        """
        self.logger.info("Creating stacking ensemble with XGBoost meta-learner...")
        
        estimators = [(res["model_name"], res["model"]) for res in trained_models if res["model"]]

        if len(estimators) < 2:
            self.logger.warning("Cannot create an ensemble with fewer than 2 valid base models.")
            return None

        # --- Sophisticated Meta-Learner ---
        # Use a well-regularized XGBoost classifier to capture complex interactions.
        meta_learner = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_jobs=1
        )
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        stacking_classifier = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=cv,
            n_jobs=-1 # CV for the meta-learner can be parallelized
        )

        self.logger.info("Fitting the stacking ensemble...")
        stacking_classifier.fit(X_train, y_train)
        self.logger.info("Stacking ensemble fitted successfully.")
        
        return stacking_classifier

    def shutdown(self):
        """Shuts down the Ray instance."""
        if ray.is_initialized():
            ray.shutdown()
            self.logger.info("Ray has been shut down.")


if __name__ == "__main__":
    # --- Example Usage ---
    MODEL_CONFIGS = [
        {"name": "rf", "class": RandomForestClassifier, "params": {"n_estimators": 100, "max_depth": 15}},
        {"name": "lgbm", "class": lgb.LGBMClassifier, "params": {"n_estimators": 200, "learning_rate": 0.1}},
        {"name": "xgb", "class": xgb.XGBClassifier, "params": {"n_estimators": 150, "learning_rate": 0.1}}
    ]

    X, y = pd.DataFrame(np.random.randn(2000, 25)), pd.Series(np.random.randint(0, 2, 2000))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    manager = HighPerformanceEnsembleManager()
    
    try:
        # First run: will execute and cache results
        print("\n--- First Run: Training and Caching ---")
        start_time = time.time()
        base_models = manager.train_base_models(MODEL_CONFIGS, X_train, y_train, X_val, y_val)
        print(f"First run took: {time.time() - start_time:.2f}s")
        
        # Second run: should be much faster due to caching
        print("\n--- Second Run: Loading from Cache ---")
        start_time = time.time()
        cached_models = manager.train_base_models(MODEL_CONFIGS, X_train, y_train, X_val, y_val)
        print(f"Second run took: {time.time() - start_time:.2f}s")

        if base_models:
            stacking_model = manager.create_stacking_ensemble(base_models, X_train, y_train)
            if stacking_model:
                y_pred_ensemble = stacking_model.predict(X_val)
                accuracy = accuracy_score(y_val, y_pred_ensemble)
                print(f"\n--- Final Ensemble Performance ---")
                print(f"Validation Accuracy: {accuracy:.4f}")
    finally:
        manager.shutdown()

