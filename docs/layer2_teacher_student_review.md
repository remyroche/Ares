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

1.  **Teacher OOF Generation**:
    *   Using K-Fold Cross-Validation, `oof_preds` (Teacher OOF) is fully populated.
    *   For each fold `k`: `oof_preds[val_idx]` comes from a Teacher trained on `train_idx`.
    *   **Crucial**: This is done *before* Student training to ensure independence.

2.  **Student OOF Generation (Nested)**:
    *   Iterate through the same K-Fold splits.
    *   **Training the Student**:
        *   The Student is trained on `train_idx`.
        *   **Baseline/Init Score**: It uses `oof_preds[train_idx]`.
        *   **Why**: `oof_preds[train_idx]` are the "unseen" predictions for the training data (generated when they were validation sets in other folds). This simulates the noise profile of OOF predictions, matching standard stacking practices.
    *   **Predicting with Student**:
        *   Predict on `val_idx`.
        *   **Baseline/Init Score**: It uses `oof_preds[val_idx]` (Law OOF).
        *   **Extraction**: We extract only the **tree contribution**:
            *   **LGBM**: `predict(..., raw_score=True)` (Returns raw logits/trees).
            *   **XGB**: `predict(..., base_margin=law_oof, output_margin=True)` then subtracts `law_oof`.
            *   **CatBoost**: `predict(..., prediction_type='RawFormulaVal')` (Returns raw formula value/trees).
3.  **Total OOF**: `Total = Teacher OOF + Student OOF`.
4.  **Calibration**: An `IsotonicRegression` model is fitted on `Total OOF` vs `y`.

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

## Verification Findings
*   **Score Space Consistency**: Confirmed that `HuberRegressor` outputs (linear scores) are treated as raw margins/logits throughout. LGBM/CatBoost/XGB are correctly configured to add their tree outputs to this baseline.
*   **Teacher Handling**: Confirmed that `law_oof` is computed via K-Fold first. Student training uses the training-split portion of this OOF vector (`oof_preds[train_idx]`), ensuring the baseline is effectively "out-of-sample" relative to the Teacher, preventing leakage and simulating the inference scenario correctly.
