# Layer 2 Teacher-Student Architecture Review

## Overview
The `Label_based_layer_2.py` module uses a **Teacher-Student** (or "Law & Fashion") architecture implemented in the `HuberResidualStack` class. This architecture combines a robust linear baseline (Teacher) with a non-linear residual learner (Student).

## Components
1.  **Teacher (Law)**: `IRMLinearRegressor` (Huber Loss).
    *   Role: Captures the linear trend and provides a robust baseline prediction.
2.  **Student (Fashion)**: Tree-based model (`ExtraTrees`, `LGBM`, `XGB`, `CatBoost`).
    *   Role: Learns to correct the Teacher's errors (residuals).

## Combination Logic

### 1. OOF Generation (Training Phase)
The goal is to generate unbiased out-of-fold predictions for the combined stack to train the calibrator.

1.  **Teacher OOF**: Generated via K-Fold Cross-Validation of the `HuberRegressor`.
2.  **Student Training**: The Student is trained on the training folds using the Teacher's OOF predictions as a baseline/offset.
    *   **LGBM**: Uses `init_score = law_oof`.
    *   **XGB**: Uses `base_margin = law_oof`.
    *   **CatBoost**: Uses `baseline = law_oof`.
    *   **ExtraTrees**: Trains on explicit residuals `y - law_oof`.
3.  **Student OOF Extraction**: The Student predicts on the validation fold. The code ensures only the **tree contribution** (residual) is extracted:
    *   **LGBM**: `predict(..., raw_score=True)` (Returns raw logits/trees).
    *   **XGB**: `predict(..., base_margin=law_oof, output_margin=True)` then subtracts `law_oof`.
    *   **CatBoost**: `predict(..., prediction_type='RawFormulaVal')` (Returns raw formula value/trees).
4.  **Total OOF**: `Total = Teacher OOF + Student OOF`.
5.  **Calibration**: An `IsotonicRegression` model is fitted on `Total OOF` vs `y`.

### 2. Inference (Prediction Phase)
1.  **Teacher Prediction**: `law_pred = law_model.predict(X)`.
2.  **Student Prediction**:
    *   **LGBM**: `fashion_pred = fashion_model.predict(X, raw_score=True)`.
        *   *Combination*: `Total = law_pred + fashion_pred`.
    *   **CatBoost**: `fashion_pred = fashion_model.predict(X, prediction_type='RawFormulaVal')`.
        *   *Combination*: `Total = law_pred + fashion_pred`.
    *   **XGB**: `Total = fashion_model.predict(X, base_margin=law_pred, output_margin=True)`.
        *   *Note*: XGBoost internally sums the base margin and tree outputs.
    *   **ExtraTrees**: `fashion_pred = fashion_model.predict(X)`.
        *   *Combination*: `Total = law_pred + fashion_pred`.
3.  **Final Probability**: `calibrator.predict(Total)`.

## Verification
Reproduction scripts confirmed that for LGBM and CatBoost, `predict` without the baseline argument (when trained with one) returns only the tree contribution, validating the manual summation logic used in `HuberResidualStack`.
