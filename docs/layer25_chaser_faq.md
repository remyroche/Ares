# Layer 2.5 Chaser FAQ

## 1. Where do the sample weights come from?

The final sample weights (`w_final`) used to train the Chaser students are a combination of three sources:

1.  **Teacher Uncertainty (Internal Upweighting):**
    The Chaser specifically upweights samples where the Teacher (BayesianRidge) is uncertain.
    *   **Logic:** `w_chase = sqrt(std_oof / median_std)`
    *   **Implementation:** `uncertainty_to_chaser_weight` function in `src/training/steps/labeling/layer2_5_chaser.py`.
    *   **Goal:** Focus the non-linear students on areas where the linear teacher struggles. These weights are clipped between 0.5 and 2.0.

2.  **Regime Probabilities (Regime-Awareness):**
    If `regime_split=True`, the weights are modulated by the probability of the current regime.
    *   **Logic:** `final_w = w_final * P(Regime=k)`
    *   This ensures that the specialist for Regime K focuses primarily on data points belonging to that regime.

3.  **Base Weights (External):**
    The model accepts a `sample_weight` argument in its `fit` method. These are the standard weights from the labeling pipeline (e.g., uniqueness weights, event weights) passed down from the caller.

**Formula:**
`Total_Weight = Base_Weight * Regime_Probability * Teacher_Uncertainty_Factor`

## 2. How do we denoise the target?

The Chaser denoises the target primarily through **Winsorization (Robust Clipping)** based on the Median Absolute Deviation (MAD). This happens in two stages:

1.  **Teacher Level (Baseline Denoising):**
    Before training the linear Teacher (BayesianRidge), the raw target `y` is winsorized to prevent outliers from skewing the baseline trend.
    *   `y_teacher = winsorize(y, k=4.0)`

2.  **Student Level (Residual Denoising):**
    The students learn the *residuals* (`y - teacher_pred`). These residuals are also winsorized before being used as training targets.
    *   **Logic:** `target = winsorize(y - teacher_mu, k=3.0)`
    *   **Method:** It calculates the **Robust Sigma** (approx 1.4826 * MAD) and clips values that are more than `k` sigmas away from the median.

This ensures that the Chaser learns systematic non-linear patterns rather than chasing noise or extreme outliers.
