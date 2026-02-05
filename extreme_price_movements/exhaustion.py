import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from extreme_price_movements.utils import tprint, clean_dataset
from extreme_price_movements.feature_selection_extreme_events import mdi_feature_selection_v3
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
        return ScaledLogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=self.l1_ratio,
            C=self.C,
            max_iter=2000,
            random_state=42,
            class_weight="balanced" 
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray = None):
        """
        Fits the model with Platt Scaling calibration using TimeSeriesSplit.
        Includes MDI Feature Selection (Leakage Safe).
        
        Args:
            X: Feature matrix
            y: Binary labels
            sample_weight: Optional sample weights for training
        """
        tprint(f"Entering function: fit in exhaustion.py")

        # Clean Dataset BEFORE feature selection to ensure consistency
        # This prevents mdi from selecting features on a cleaned set, while this model fits on dirty set.
        X, y, sample_weight = clean_dataset(X, y, sample_weight, name="X_exh")

        if X.empty:
            tprint("ExhaustionModel: X is empty after cleaning. Cannot fit.")
            # self.model remains None.
            return self

        # 1. Feature Selection
        n_samples = len(X)
        # Max cap 40, or n/100
        n_select = min(40, max(1, n_samples // 100))

        tprint(f"ExhaustionModel: Running feature selection. Target features={n_select}")

        base_selector = ExtraTreesClassifier(
            n_estimators=500, # Increased per v3 request
            max_depth=None,   # Let v3 suggest depth
            min_samples_leaf=50,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42,
            class_weight="balanced"
        )

        # Note: mdi_feature_selection_v3 also calls clean_dataset, but since we already cleaned X,
        # it should be a no-op (or catch anything we missed).
        sel_res = mdi_feature_selection_v3(
            X=X,
            y=y,
            base_model=base_selector,
            n_splits=self.cv_splits,
            analysis_n_estimators=500,
            sample_weight=sample_weight,
            end_features=n_select,
            cumulative_cap=0.98,
            min_share=0.001,
            min_features=5,
            max_features_pct=0.5
        )

        # Check if selection returned empty features?
        if not sel_res.selected_features:
            tprint("ExhaustionModel: No features selected. Cannot fit.")
            return self

        self.selected_features = sel_res.selected_features[:n_select]
        tprint(f"ExhaustionModel: Selected {len(self.selected_features)} features.")

        # Re-slice X based on selection.
        # Since X is already cleaned (same rows as mdi saw), this is safe.
        X_sel = X[self.selected_features]

        # 2. Calibration — ensure float64 for sklearn compatibility
        if sample_weight is not None:
            sample_weight = sample_weight.astype(np.float64)
        X_sel = X_sel.astype(np.float64)

        base_clf = self._make_base_estimator()

        # Adapt cv_splits to minority class size to avoid sklearn ValueError
        # CalibratedClassifierCV needs at least n_splits examples per class
        min_class_count = min(np.bincount(y.astype(int)))
        if min_class_count < 2:
            tprint(f"ExhaustionModel: Minority class has only {min_class_count} samples. Fitting base model without calibration.")
            self.model = base_clf
            self.model.fit(X_sel, y, sample_weight=sample_weight)
        else:
            effective_splits = min(self.cv_splits, min_class_count)
            if effective_splits < self.cv_splits:
                tprint(f"ExhaustionModel: Reducing CV splits {self.cv_splits} -> {effective_splits} (minority class={min_class_count})")
            tscv = PurgedKFold(n_splits=effective_splits, purge=5, embargo=2)
            self.model = CalibratedClassifierCV(
                estimator=base_clf,
                method='sigmoid',
                cv=tscv
            )
            self.model.fit(X_sel, y, sample_weight=sample_weight)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        tprint(f"Entering function: predict_proba in exhaustion.py")
        if self.model is None:
            raise ValueError("Model not fitted")

        if self.selected_features is not None:
             # Handle missing columns gracefully? No, model expects features.
             # Ensure columns exist.
             missing = [c for c in self.selected_features if c not in X.columns]
             if missing:
                 tprint(f"Warning: {len(missing)} selected features missing in prediction X. Filling with 0.")
                 for c in missing:
                     X[c] = 0.0 # Or NaN? Model might fail. 0 is safer for sparse/standardized.
             X = X[self.selected_features]

        return self.model.predict_proba(X)[:, 1]

    def compute_oof_predictions(self, X: pd.DataFrame, y: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Computes OOF probabilities manually to get accurate performance metrics.
        Returns: (oof_probs, metrics)
        """
        tprint(f"Entering function: compute_oof_predictions in exhaustion.py")

        # Should we clean X here too? Ideally yes, but this is usually called after fit with same data?
        # If passed new data, it might need cleaning.
        # Let's assume caller handles it or just select features.

        # Apply selection if available
        if self.selected_features is not None:
            X = X[self.selected_features]

        tscv = PurgedKFold(n_splits=self.cv_splits, purge=5, embargo=2)
        oof_preds = np.full(len(y), np.nan)

        briers = []
        aucs = []

        # We need to ensure X is numpy
        X_arr = X.to_numpy(dtype=np.float32) if isinstance(X, pd.DataFrame) else X

        for train_idx, test_idx in tscv.split(X_arr):
            X_train, X_test = X_arr[train_idx], X_arr[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Inner CV for calibration on the training set (Purged)
            inner_cv = PurgedKFold(n_splits=3, purge=3, embargo=1)
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
