import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import HuberRegressor, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import joblib
from extreme_price_movements.utils import tprint

class ModelRace(BaseEstimator, RegressorMixin):
    def __init__(self, kind="mr", n_splits=3):
        self.kind = kind
        self.n_splits = n_splits
        self.best_model = None
        self.best_model_name = None
        self.metrics = {}

    def _get_candidates(self):
        candidates = {}

        # 1. Baseline (Huber or ElasticNet)
        if self.kind == "mr":
            candidates["huber"] = Pipeline([
                ("scaler", StandardScaler()),
                ("reg", HuberRegressor(epsilon=1.35, max_iter=200))
            ])
        else: # tf
            candidates["elasticnet"] = Pipeline([
                ("scaler", StandardScaler()),
                ("reg", ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=2000))
            ])

        # 2. ExtraTrees
        # Objective: absolute_error
        candidates["extratrees"] = ExtraTreesRegressor(
            n_estimators=1000,
            criterion="absolute_error",
            max_depth=7,
            min_samples_leaf=50,
            max_features=0.5,
            n_jobs=-1,
            random_state=42
        )

        # 3. CatBoost
        # loss_function=Quantile:alpha=0.5
        candidates["catboost"] = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.02,
            loss_function="Quantile:alpha=0.5",
            l2_leaf_reg=20,
            depth=5,
            subsample=0.6,
            colsample_bylevel=0.6,
            random_strength=10,
            min_data_in_leaf=50,
            verbose=False,
            allow_writing_files=False,
            random_state=42
        )

        # 4. XGBoost
        # absoluteerror -> reg:absoluteerror (since xgboost 1.7?) or just mae?
        # XGBoost supports "reg:absoluteerror".
        candidates["xgboost"] = XGBRegressor(
            n_estimators=10,
            num_parallel_tree=150, # Random Forest style?
            max_depth=6,
            gamma=4,
            reg_lambda=5, # lambda is keyword
            alpha=1,
            subsample=0.6,
            colsample_bylevel=0.6,
            max_delta_step=3,
            objective="reg:absoluteerror",
            n_jobs=-1,
            random_state=42
        )

        return candidates

    def fit(self, X, y, sample_weight=None):
        candidates = self._get_candidates()
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        results = {}

        # We need to handle sample_weight for all models?
        # Huber: yes. ET: yes. CatBoost: yes. XGB: yes.

        for name, model in candidates.items():
            tprint(f"Race: Training {name}...")
            scores = []
            try:
                # CV evaluation (IC or MAE?)
                # User says: "Apply calibration... compare AUC, Brier, IC".
                # For Regression, IC (Correlation) is standard. AUC/Brier is for classification.
                # Assuming IC for ranking regressors.

                for train_idx, val_idx in tscv.split(X):
                    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_tr, y_val = y[train_idx], y[val_idx]
                    w_tr = sample_weight[train_idx] if sample_weight is not None else None

                    # Clone model?
                    m = joblib.clone(model) # sklearn models
                    # CatBoost/XGBoost compatibility with clone?
                    # Generally yes if scikit-learn wrappers.

                    if name in ["catboost", "xgboost", "extratrees"]:
                        m.fit(X_tr, y_tr, sample_weight=w_tr)
                    elif name == "huber":
                        m.fit(X_tr, y_tr, reg__sample_weight=w_tr) # Pipeline step name
                    else:
                        m.fit(X_tr, y_tr, reg__sample_weight=w_tr) # ElasticNet pipeline

                    preds = m.predict(X_val)

                    # Metric: IC (Spearman Correlation)
                    ic = pd.Series(preds).corr(pd.Series(y_val), method="spearman")
                    scores.append(ic)

                avg_score = np.nanmean(scores)
                results[name] = avg_score
                tprint(f"  {name} IC: {avg_score:.4f}")

            except Exception as e:
                tprint(f"  {name} Failed: {e}")
                results[name] = -1.0

        # Select best
        if not results:
            raise ValueError("All models failed in race")

        best_name = max(results, key=results.get)
        self.best_model_name = best_name
        self.metrics = results
        tprint(f"Race Winner: {best_name} (IC={results[best_name]:.4f})")

        # Retrain best on full data
        tprint(f"Retraining {best_name} on full data...")
        final_model = candidates[best_name]

        if best_name in ["catboost", "xgboost", "extratrees"]:
            final_model.fit(X, y, sample_weight=sample_weight)
        elif best_name == "huber":
            final_model.fit(X, y, reg__sample_weight=sample_weight)
        else:
            final_model.fit(X, y, reg__sample_weight=sample_weight)

        self.best_model = final_model
        return self

    def predict(self, X):
        if self.best_model is None:
            raise ValueError("ModelRace not fitted")
        return self.best_model.predict(X)
