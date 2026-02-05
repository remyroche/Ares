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

class ScaledLogisticRegression(LogisticRegression):
    """
    Wrapper to apply StandardScaler internally, ensuring sample_weight 
    is correctly passed to fit (bypassing Pipeline limitations with CalibratedClassifierCV).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scaler = StandardScaler()

    def fit(self, X, y, sample_weight=None):
        X_scaled = self.scaler.fit_transform(X)
        return super().fit(X_scaled, y, sample_weight=sample_weight)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return super().predict(X_scaled)
        
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return super().predict_proba(X_scaled)

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

    # Combined Score: weighted sum (more forgiving than triple product)
    # AUC is primary (0-1 range centered at 0.5), IC adds magnitude signal
    auc_contrib = max(0, auc - 0.5)  # 0 to 0.5
    ic_contrib = max(0, ic)           # 0 to 1
    bss_contrib = max(0, bss)         # 0 to 1
    selection_score = 0.6 * auc_contrib + 0.3 * ic_contrib + 0.1 * bss_contrib

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
        # ScaledLogisticRegression (Solves Pipeline+SampleWeight issue)
        candidates["elasticnet"] = ScaledLogisticRegression(
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
        self.oof_probs = None  # Will store OOF predictions from best model

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

        def safe_slice(arr, idx):
            if hasattr(arr, "iloc"): return arr.iloc[idx]
            return arr[idx]

        # Store per-model detailed metrics for reporting
        detailed_metrics = {}

        for name, model in candidates.items():
            tprint(f"Race: Training {name}...")
            fold_scores = []
            fold_aucs = []
            fold_ics = []
            fold_bss = []

            try:
                for fold_i, (train_idx, val_idx) in enumerate(cached_splits):
                    if use_numpy:
                        X_tr, X_val = X_np[train_idx], X_np[val_idx]
                    else:
                        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]

                    y_tr = safe_slice(y, train_idx)
                    y_val = safe_slice(y, val_idx)
                    w_tr = safe_slice(sample_weight, train_idx) if sample_weight is not None else None
                    ret_val = safe_slice(returns, val_idx)

                    model.fit(X_tr, y_tr, sample_weight=w_tr)
                    probs = model.predict_proba(X_val)[:, 1]

                    metrics = calculate_selection_score(y_val, probs, ret_val)
                    fold_scores.append(metrics["Selection_Score"])
                    fold_aucs.append(metrics["AUC"])
                    fold_ics.append(metrics["IC"])
                    fold_bss.append(metrics["BSS"])

                avg_score = np.nanmean(fold_scores)
                avg_auc = np.nanmean(fold_aucs)
                avg_ic = np.nanmean(fold_ics)
                avg_bss_val = np.nanmean(fold_bss)
                results[name] = avg_score
                detailed_metrics[name] = {"score": avg_score, "AUC": avg_auc, "IC": avg_ic, "BSS": avg_bss_val}
                tprint(f"  {name}: Score={avg_score:.4f}  AUC={avg_auc:.4f}  IC={avg_ic:.4f}  BSS={avg_bss_val:.4f}")

            except Exception as e:
                tprint(f"  {name} Failed: {e}")
                results[name] = -float("inf")

        self.detailed_metrics = detailed_metrics

        if not results:
            raise ValueError("All models failed in race")

        best_name = max(results, key=results.get)
        self.best_model_name = best_name
        self.metrics = results
        if best_name in detailed_metrics:
            dm = detailed_metrics[best_name]
            tprint(f"Race Winner: {best_name} (Score={dm['score']:.4f}, AUC={dm['AUC']:.4f}, IC={dm['IC']:.4f}, BSS={dm['BSS']:.4f})")
        else:
            tprint(f"Race Winner: {best_name} (Score={results[best_name]:.4f})")

        # 2. Generate OOF predictions with best model (for meta model)
        tprint(f"Generating OOF predictions with {best_name}...")
        oof_probs = np.full(len(y), np.nan, dtype=np.float32)
        oof_candidates = self._get_candidates(race_mode=True)
        oof_model = oof_candidates[best_name]
        for train_idx, val_idx in cached_splits:
            if use_numpy:
                X_tr, X_val = X_np[train_idx], X_np[val_idx]
            else:
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr = safe_slice(y, train_idx)
            w_tr = safe_slice(sample_weight, train_idx) if sample_weight is not None else None
            oof_model.fit(X_tr, y_tr, sample_weight=w_tr)
            oof_probs[val_idx] = oof_model.predict_proba(X_val)[:, 1]
        # Fill any remaining NaN with 0.5 (neutral)
        oof_probs = np.nan_to_num(oof_probs, nan=0.5)
        self.oof_probs = oof_probs
        tprint(f"OOF predictions: mean={np.mean(oof_probs):.4f}, std={np.std(oof_probs):.4f}")

        # 3. Final Retraining & Calibration
        tprint(f"Retraining {best_name} on full data (full config)...")
        final_candidates = self._get_candidates(race_mode=False)
        final_base = final_candidates[best_name]

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
