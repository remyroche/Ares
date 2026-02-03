import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.stats import spearmanr
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import joblib
from extreme_price_movements.utils import tprint

def calculate_selection_score(y_true, y_prob, y_returns):
    """
    S_model = (AUC - 0.5) * BSS * IC
    """
    # 1. AUC (Ranking)
    # y_true must be binary.
    try:
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_prob)
        else:
            auc = 0.5
    except ValueError:
        auc = 0.5

    # 2. Brier Skill Score (Calibration)
    bs_actual = brier_score_loss(y_true, y_prob)
    # Reference: predicting the mean hit rate of the validation set
    mean_val = np.mean(y_true)
    bs_ref = brier_score_loss(y_true, np.full_like(y_true, mean_val))
    bss = 1 - (bs_actual / (bs_ref + 1e-12))

    # 3. Information Coefficient (Magnitude)
    # Spearman correlation between predicted prob and ACTUAL CONTINUOUS RETURN
    # y_returns should be the realized return (signed or not depending on context)
    # y_prob correlates with strength of move?
    ic, _ = spearmanr(y_prob, y_returns)

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
        self.kind = kind
        self.n_splits = n_splits
        self.best_model = None
        self.best_model_name = None
        self.metrics = {}

    def _get_candidates(self):
        candidates = {}

        # 1. Baseline
        # LogisticRegression (ElasticNet)
        candidates["elasticnet"] = LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.5, C=1.0, max_iter=2000, random_state=42
        )

        # 2. ExtraTrees
        candidates["extratrees"] = ExtraTreesClassifier(
            n_estimators=1000,
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
            iterations=1000,
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
            n_estimators=10,
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
        candidates = self._get_candidates()
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        results = {}

        # We need returns for validation scoring.
        # If returns is None, we can't calc IC properly.
        # Assuming caller passes returns aligned with y.
        if returns is None:
            # Fallback? IC using y (binary)? No, IC needs continuous.
            # We'll use y as proxy if needed, but warning.
            returns = y

        for name, model in candidates.items():
            tprint(f"Race: Training {name}...")
            scores = []

            try:
                for train_idx, val_idx in tscv.split(X):
                    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_tr, y_val = y[train_idx], y[val_idx]
                    w_tr = sample_weight[train_idx] if sample_weight is not None else None
                    ret_val = returns[val_idx]

                    # Calibrated Wrapper
                    # method='isotonic' needs samples. 'sigmoid' is safer for small data.
                    # User code used 'isotonic' with cv=3.
                    # We use inner CV for calibration.
                    calibrated_model = CalibratedClassifierCV(
                        estimator=model,
                        method='isotonic',
                        cv=3
                    )

                    # Handling sample_weight in CalibratedClassifierCV.fit
                    # It supports sample_weight if the underlying estimator does.
                    # All our candidates do.
                    calibrated_model.fit(X_tr, y_tr, sample_weight=w_tr)

                    probs = calibrated_model.predict_proba(X_val)[:, 1]

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

        # Retrain best on full data
        tprint(f"Retraining {best_name} on full data...")
        final_base = candidates[best_name]

        # Final Calibration on full data?
        # Use CV=3 again?
        self.best_model = CalibratedClassifierCV(
            estimator=final_base,
            method='isotonic',
            cv=3
        )
        self.best_model.fit(X, y, sample_weight=sample_weight)

        return self

    def predict_proba(self, X):
        if self.best_model is None:
            raise ValueError("ModelRace not fitted")
        return self.best_model.predict_proba(X)

    def predict(self, X):
        # Return probability class 1
        return self.predict_proba(X)[:, 1]
