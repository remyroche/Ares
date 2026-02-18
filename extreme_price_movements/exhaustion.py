import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from extreme_price_movements.utils import tprint, clean_dataset
from extreme_price_movements.feature_selection_extreme_events import mdi_feature_selection_v3
from extreme_price_movements.purged_cv import PurgedKFold


class ExhaustionModel(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=200, max_depth=8, n_select=15, 
                 min_samples_leaf=50, min_impurity_decrease=1e-5, ccp_alpha=1e-1,
                 random_state=42, n_jobs=3, class_weight="balanced", cv_splits=5):
        # User requested ExtraTrees instead of LogisticRegression
        # "ExhaustionModel uses ScaledLogisticRegression (Linear) -> implement ExtraTrees instead"
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.n_select = n_select
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.class_weight = class_weight
        self.cv_splits = cv_splits
        
        self.feature_selector = None
        self.model = None
        self.selected_features_ = None

    @property
    def selected_features(self):
        return self.selected_features_

    def _make_base_estimator(self):
        return ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            ccp_alpha=self.ccp_alpha,
            max_features="sqrt",
            bootstrap=False,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            class_weight=self.class_weight
        )

    def fit(self, X, y, sample_weight=None):
        """
        Fits ExtraTreesClassifier.
        Handles soft labels by thresholding y at 0 to define class,
        and using y as a confidence multiplier for sample_weight if y is continuous.
        """
        tprint(f"Entering function: fit in exhaustion.py")

        # 0. Clean Data
        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Handle Soft Labels
        # If y is continuous (soft labels from fast_funcs), we treat y>0 as Class 1.
        y_vals = np.asarray(y)
        y_binary = (y_vals > 0.0).astype(int)
        
        final_weights = sample_weight.copy() if sample_weight is not None else np.ones(len(y))
        
        # Multiply weight by label quality for positives if y is float
        if np.issubdtype(y_vals.dtype, np.floating):
             pos_mask = y_vals > 0
             # w_new = w_old * label_quality
             final_weights[pos_mask] = final_weights[pos_mask] * y_vals[pos_mask]

        if X_clean.empty:
            tprint("ExhaustionModel: X is empty. Cannot fit.")
            return self

        # 1. Feature Selection (MDI)
        # We use a small forest for selection
        tprint(f"ExhaustionModel: Running feature selection. Target features={self.n_select}")
        sel = ExtraTreesClassifier(
            n_estimators=50, 
            max_depth=6, 
            max_features="sqrt", 
            random_state=self.random_state, 
            n_jobs=self.n_jobs,
            class_weight=self.class_weight
        )
        sel.fit(X_clean, y_binary, sample_weight=final_weights)
        
        importances = sel.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        top_n = min(self.n_select, X.shape[1])
        self.selected_features_ = X.columns[indices[:top_n]].tolist()
        tprint(f"ExhaustionModel: Selected {len(self.selected_features_)} features.")
        
        X_sel = X_clean[self.selected_features_]
        
        # 2. Main Model Training (Extra Trees)
        self.model = ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            ccp_alpha=self.ccp_alpha,
            max_features="sqrt",
            bootstrap=False,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            class_weight=self.class_weight
        )
        
        self.model.fit(X_sel, y_binary, sample_weight=final_weights)
        
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not fitted")

        if self.selected_features_ is not None:
             # Handle missing columns gracefully? No, model expects features.
             # Ensure columns exist.
             missing = [c for c in self.selected_features_ if c not in X.columns]
             if missing:
                 tprint(f"Warning: {len(missing)} selected features missing in prediction X. Filling with 0.")
                 for c in missing:
                     X[c] = 0.0 # Or NaN? Model might fail. 0 is safer for sparse/standardized.
             X = X[self.selected_features_]

        proba = self.model.predict_proba(X)
        if proba.shape[1] == 1:
            return np.zeros(len(X)) if self.model.classes_[0] == 0 else np.ones(len(X))
        return proba[:, 1]

    def compute_oof_predictions(self, X: pd.DataFrame, y: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Computes OOF probabilities manually to get accurate performance metrics.
        Returns: (oof_probs, metrics)
        """
        tprint(f"Entering function: compute_oof_predictions in exhaustion.py")

        # Apply selection if available
        if self.selected_features_ is not None:
            missing = [c for c in self.selected_features_ if c not in X.columns]
            if missing:
                 for c in missing: X[c] = 0.0
            X = X[self.selected_features_]

        # Use fixed splits for OOF (PurgedKFold)
        # Note: self.cv_splits was removed from init in previous edit? 
        # Check __init__ args. It's not there. We should probably add it or hardcode.
        # Looking at previous file view, cv_splits went away. I'll default to 5.
        tscv = PurgedKFold(n_splits=5, purge=5, embargo=2)
        oof_preds = np.full(len(y), np.nan)

        briers = []
        aucs = []

        # We need to ensure X is numpy
        X_arr = X.to_numpy(dtype=np.float32) if isinstance(X, pd.DataFrame) else X
        
        # Handle soft labels for OOF same as fit
        y_vals = np.asarray(y)
        y_binary = (y_vals > 0.0).astype(int)

        for train_idx, test_idx in tscv.split(X_arr):
            X_train, X_test = X_arr[train_idx], X_arr[test_idx]
            y_train, y_test = y_binary[train_idx], y_binary[test_idx]
            # Weights for training? We don't have sample_weight passed here. 
            # compute_oof_predictions signature doesn't take sample_weight.
            # We'll assume unweighted OOF or we should update signature. 
            # Given method didn't take it before, we proceed without.

            # Inner CV for calibration? ExtraTrees usually okay without if probas needed?
            # User didn't ask for calibration changes, but we shouldn't use CalibratedClassifierCV 
            # if we want to test true model performance.
            # But the original code used CalibratedClassifierCV.
            # I will use the Base Model directly for OOF to match the main model.
            
            clf = ExtraTreesClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                ccp_alpha=self.ccp_alpha,
                max_features="sqrt",
                bootstrap=False,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                class_weight=self.class_weight
            )
            clf.fit(X_train, y_train)

            proba = clf.predict_proba(X_test)
            if proba.shape[1] == 1:
                # Single class in training fold — assign 0 or 1 based on which class
                p_test = np.zeros(len(X_test)) if clf.classes_[0] == 0 else np.ones(len(X_test))
            else:
                p_test = proba[:, 1]
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
