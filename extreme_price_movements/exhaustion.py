import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from extreme_price_movements.utils import tprint
from extreme_price_movements.feature_selection_extreme_events import mdi_feature_selection_leakage_safe

class ExhaustionModel:
    def __init__(self, C=1.0, l1_ratio=0.3, cv_splits=5):
        tprint(f"Entering function: __init__ in exhaustion.py")
        self.C = C
        self.l1_ratio = l1_ratio
        self.cv_splits = cv_splits
        self.model = None
        self.metrics = {}
        self.selected_features = None

    def _make_base_estimator(self):
        tprint(f"Entering function: _make_base_estimator in exhaustion.py")
        return LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=self.l1_ratio,
            C=self.C,
            max_iter=2000,
            random_state=42,
            class_weight="balanced" # Helpful for imbalance? User didn't specify but good practice.
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """
        Fits the model with Platt Scaling calibration using TimeSeriesSplit.
        Includes MDI Feature Selection (Leakage Safe).
        """
        tprint(f"Entering function: fit in exhaustion.py")

        # 1. Feature Selection
        n_samples = len(X)
        # Max cap 40, or n/100
        n_select = min(40, max(1, n_samples // 100))

        tprint(f"ExhaustionModel: Running feature selection. Target features={n_select}")

        base_selector = ExtraTreesClassifier(
            n_estimators=100,
            max_depth=4,
            min_samples_leaf=50,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42,
            class_weight="balanced"
        )

        sel_res = mdi_feature_selection_leakage_safe(
            X=X,
            y=y,
            base_model=base_selector,
            n_splits=self.cv_splits, # Use same splits as CV
            top_n_precluster=n_select,
            keep_top_per_cluster=1,
            use_quantile_transform_for_corr=True
        )

        self.selected_features = sel_res.selected_features
        tprint(f"ExhaustionModel: Selected {len(self.selected_features)} features.")

        X_sel = X[self.selected_features]

        # 2. Calibration
        base_clf = self._make_base_estimator()

        # Use CalibratedClassifierCV with TimeSeriesSplit
        # method='sigmoid' is Platt scaling.
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)

        self.model = CalibratedClassifierCV(
            estimator=base_clf,
            method='sigmoid',
            cv=tscv
        )

        self.model.fit(X_sel, y)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        tprint(f"Entering function: predict_proba in exhaustion.py")
        if self.model is None:
            raise ValueError("Model not fitted")

        if self.selected_features is not None:
             # Ensure we have the columns (handle missing safely? or assume caller provides same schema)
             # Usually schema is consistent.
             X = X[self.selected_features]

        return self.model.predict_proba(X)[:, 1]

    def compute_oof_predictions(self, X: pd.DataFrame, y: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Computes OOF probabilities manually to get accurate performance metrics.
        Returns: (oof_probs, metrics)
        """
        tprint(f"Entering function: compute_oof_predictions in exhaustion.py")

        # Apply selection if available
        if self.selected_features is not None:
            X = X[self.selected_features]

        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        oof_preds = np.full(len(y), np.nan)

        briers = []
        aucs = []

        # We need to ensure X is numpy
        X_arr = X.to_numpy(dtype=np.float32) if isinstance(X, pd.DataFrame) else X

        for train_idx, test_idx in tscv.split(X_arr):
            X_train, X_test = X_arr[train_idx], X_arr[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Inner CV for calibration on the training set
            inner_cv = TimeSeriesSplit(n_splits=3)
            clf = CalibratedClassifierCV(
                estimator=self._make_base_estimator(),
                method='sigmoid',
                cv=inner_cv
            )
            clf.fit(X_train, y_train)

            p_test = clf.predict_proba(X_test)[:, 1]
            oof_preds[test_idx] = p_test

            try:
                briers.append(brier_score_loss(y_test, p_test))
                if len(np.unique(y_test)) > 1:
                    aucs.append(roc_auc_score(y_test, p_test))
            except Exception:
                pass

        # Overall metrics
        valid_mask = ~np.isnan(oof_preds)
        if valid_mask.sum() > 0:
            final_brier = brier_score_loss(y[valid_mask], oof_preds[valid_mask])
            try:
                final_auc = roc_auc_score(y[valid_mask], oof_preds[valid_mask])
            except:
                final_auc = np.nan
        else:
            final_brier = np.nan
            final_auc = np.nan

        metrics = {
            "oof_brier": final_brier,
            "oof_auc": final_auc,
            "fold_briers": briers,
            "fold_aucs": aucs
        }
        self.metrics = metrics
        return oof_preds, metrics
