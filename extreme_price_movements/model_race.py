import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.stats import rankdata
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import joblib
from extreme_price_movements.utils import tprint
from extreme_price_movements.purged_cv import PurgedKFold

def calculate_selection_score(y_true, y_prob, y_returns):
    """
    S_model = (AUC - 0.5) * BSS * IC
    Optimized:
    - Brier reference calculated analytically (O(1) from mean).
    - Spearman correlation via rank transformation + Pearson (O(N)).
    """
    # 1. AUC (Ranking)
    try:
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_prob)
        else:
            auc = 0.5
    except ValueError:
        auc = 0.5

    # 2. Brier Skill Score (Calibration)
    # BS = mean((p - y)^2)
    # Reference: p = mean(y) constant
    # BS_ref = mean((mean(y) - y)^2) = var(y)
    bs_actual = brier_score_loss(y_true, y_prob)
    mean_val = np.mean(y_true)
    bs_ref = np.mean((y_true - mean_val) ** 2)

    bss = 1.0 - (bs_actual / (bs_ref + 1e-12))

    # 3. Information Coefficient (Magnitude)
    # Spearman correlation between predicted prob and ACTUAL CONTINUOUS RETURN
    # Optimized: rankdata + corrcoef
    # Handle constant inputs to avoid NaNs
    if np.std(y_prob) < 1e-9 or np.std(y_returns) < 1e-9:
        ic = 0.0
    else:
        rank_prob = rankdata(y_prob)
        rank_ret = rankdata(y_returns)
        # np.corrcoef returns 2x2 matrix
        ic = np.corrcoef(rank_prob, rank_ret)[0, 1]
        if np.isnan(ic):
            ic = 0.0

    # Combined Score
    selection_score = (auc - 0.5) * max(0, bss) * max(0, ic)

    return {
        "Selection_Score": selection_score,
        "AUC": auc,
        "BSS": bss,
        "IC": ic,
        "Brier": bs_actual
    }

class ModelRace(BaseEstimator, ClassifierMixin):
    def __init__(self, kind="mr", n_splits=3):
        tprint(f"Entering function: __init__ in model_race.py")
        self.kind = kind
        self.n_splits = n_splits
        self.best_model = None
        self.best_model_name = None
        self.metrics = {}

    def _get_candidates(self, race_mode=True):
        tprint(f"Entering function: _get_candidates in model_race.py (race_mode={race_mode})")
        candidates = {}

        # Scaling factors for race vs final
        n_est_et = 200 if race_mode else 1000
        n_iter_cb = 300 if race_mode else 1000
        n_est_xgb = 2 if race_mode else 10 # 2*150=300 vs 10*150=1500 trees

        # 1. Baseline
        # LogisticRegression (ElasticNet)
        candidates["elasticnet"] = LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.5, C=1.0, max_iter=2000, random_state=42
        )

        # 2. ExtraTrees
        candidates["extratrees"] = ExtraTreesClassifier(
            n_estimators=n_est_et,
            max_depth=7,
            min_samples_leaf=50,
            max_features=0.5,
            bootstrap=True,
            criterion='entropy',
            n_jobs=-1,
            random_state=42
        )

        # 3. CatBoost
        candidates["catboost"] = CatBoostClassifier(
            iterations=n_iter_cb,
            learning_rate=0.02,
            l2_leaf_reg=20,
            depth=5,
            subsample=0.6,
            colsample_bylevel=0.6,
            random_strength=10,
            min_data_in_leaf=50,
            verbose=0,
            loss_function='Logloss',
            allow_writing_files=False,
            random_state=42
        )

        # 4. XGBoost
        candidates["xgboost"] = XGBClassifier(
            n_estimators=n_est_xgb,
            num_parallel_tree=150,
            max_depth=6,
            gamma=4,
            reg_lambda=5,
            reg_alpha=1,
            subsample=0.6,
            colsample_bylevel=0.6,
            max_delta_step=3,
            n_jobs=-1,
            random_state=42
        )

        return candidates

    def fit(self, X, y, sample_weight=None, returns=None):
        """
        X: features
        y: binary target
        sample_weight: weights for training
        returns: continuous returns for IC calculation (validation)
        """
        tprint(f"Entering function: fit in model_race.py")

        # 0. Preparation
        if returns is None:
            returns = y

        # Optimize: Convert to numpy once if possible (and suitable for all models)
        # ExtraTrees/XGBoost prefer numpy. CatBoost handles both but numpy is fine if no categorical features.
        # We assume numeric features here.
        X_np = X
        use_numpy = False
        if hasattr(X, "iloc"):
            try:
                # Use float32 to save memory and match FastFuncs usage
                X_np = X.to_numpy(dtype=np.float32, copy=False)
                use_numpy = True
            except (ValueError, TypeError):
                # Fallback if conversion fails (e.g. mixed types)
                use_numpy = False

        # Cache CV splits
        tscv = PurgedKFold(n_splits=self.n_splits, purge=5, embargo=2)
        cached_splits = list(tscv.split(X))

        # 1. The Race
        candidates = self._get_candidates(race_mode=True)
        results = {}

        for name, model in candidates.items():
            tprint(f"Race: Training {name}...")
            scores = []

            try:
                for train_idx, val_idx in cached_splits:
                    # Slicing optimization
                    if use_numpy:
                        X_tr, X_val = X_np[train_idx], X_np[val_idx]
                    else:
                        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]

                    # Handle y, weight, returns (numpy arrays assumed if passed from training.py)
                    # If they are series, convert or use iloc.
                    # Assuming they are numpy arrays as per training.py usage, but let's be safe.
                    def safe_slice(arr, idx):
                        if hasattr(arr, "iloc"): return arr.iloc[idx]
                        return arr[idx]

                    y_tr = safe_slice(y, train_idx)
                    y_val = safe_slice(y, val_idx)
                    w_tr = safe_slice(sample_weight, train_idx) if sample_weight is not None else None
                    ret_val = safe_slice(returns, val_idx)

                    # SKIP CALIBRATION during race
                    # Fit base model directly
                    model.fit(X_tr, y_tr, sample_weight=w_tr)

                    # Predict proba
                    # Some models (like LogReg) might not have predict_proba?
                    # All our candidates do.
                    probs = model.predict_proba(X_val)[:, 1]

                    metrics = calculate_selection_score(y_val, probs, ret_val)
                    scores.append(metrics["Selection_Score"])

                avg_score = np.nanmean(scores)
                results[name] = avg_score
                tprint(f"  {name} Score: {avg_score:.4f}")

            except Exception as e:
                tprint(f"  {name} Failed: {e}")
                results[name] = -float("inf")

        if not results:
            raise ValueError("All models failed in race")

        best_name = max(results, key=results.get)
        self.best_model_name = best_name
        self.metrics = results
        tprint(f"Race Winner: {best_name} (Score={results[best_name]:.4f})")

        # 2. Final Retraining & Calibration
        tprint(f"Retraining {best_name} on full data (full config)...")
        # Get FULL config
        final_candidates = self._get_candidates(race_mode=False)
        final_base = final_candidates[best_name]

        # Calibrate on full data
        # We wrap the full model in CalibratedClassifierCV(cv=3)
        # This will internally perform 3-fold CV on the full dataset to train 3 calibrated classifiers
        # and average them.
        self.best_model = CalibratedClassifierCV(
            estimator=final_base,
            method='isotonic',
            cv=3
        )
        self.best_model.fit(X, y, sample_weight=sample_weight)

        return self

    def predict_proba(self, X):
        tprint(f"Entering function: predict_proba in model_race.py")
        if self.best_model is None:
            raise ValueError("ModelRace not fitted")
        return self.best_model.predict_proba(X)

    def predict(self, X):
        # Return probability class 1
        tprint(f"Entering function: predict in model_race.py")
        return self.predict_proba(X)[:, 1]
