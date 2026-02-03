import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss, roc_auc_score
from extreme_price_movements.utils import tprint

class ExhaustionModel:
    def __init__(self, C=1.0, l1_ratio=0.3, cv_splits=5):
        self.C = C
        self.l1_ratio = l1_ratio
        self.cv_splits = cv_splits
        self.model = None
        self.metrics = {}

    def _make_base_estimator(self):
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
        """
        base_clf = self._make_base_estimator()

        # Use CalibratedClassifierCV with TimeSeriesSplit
        # cv=TimeSeriesSplit(n_splits=self.cv_splits)
        # method='sigmoid' is Platt scaling.
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)

        self.model = CalibratedClassifierCV(
            estimator=base_clf,
            method='sigmoid',
            cv=tscv
        )

        self.model.fit(X, y)

        # Compute metrics on the "latest" fold approximation or just fit metrics?
        # User wants "Brier, AUC".
        # Usually we want OOF metrics.
        # But CalibratedClassifierCV refits on the whole data at the end (ensemble of calibrated models).

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not fitted")
        return self.model.predict_proba(X)[:, 1]

    def compute_oof_predictions(self, X: pd.DataFrame, y: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Computes OOF probabilities manually to get accurate performance metrics.
        Returns: (oof_probs, metrics)
        """
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        oof_preds = np.full(len(y), np.nan)

        briers = []
        aucs = []

        # We need to ensure X is numpy
        X_arr = X.to_numpy(dtype=np.float32) if isinstance(X, pd.DataFrame) else X

        for train_idx, test_idx in tscv.split(X_arr):
            X_train, X_test = X_arr[train_idx], X_arr[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Use base estimator inside calibration for this fold?
            # Or just train base estimator and calibrator on train?
            # CalibratedClassifierCV does internal CV if we pass an integer,
            # but if we passed 'prefit' we need to split manually.
            # Here we want to train a calibrated model on X_train and predict on X_test.

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
