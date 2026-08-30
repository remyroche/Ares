# Causal Gates Analysis and Implementation Plan (Rev 2)

## Executive Summary

This document provides a detailed technical plan for integrating a "Two Gates" causal architecture into `LabelBasedLayer2`. This architecture enhances signal quality by filtering unstable regimes early (Gate A) and dynamically routing predictions to the best experts later (Gate B).

### The Two Gates Architecture

1.  **Gate A (Pre-Geometry, "The Bouncer"):** A lightweight, causal filter applied *before* geometry optimization. It uses raw state features and OOF specialist signals to identify and discard "unstable" regimes where no reliable expert exists.
2.  **Gate B (Post-Race, "The Router"):** An expressive Mixture-of-Experts (MoE) router applied *after* model training. It uses the Out-Of-Fold (OOF) predictions of the winning models to dynamically weight and combine them based on the current market regime.

---

## 1. Component Enhancements: `tree_based_causal_gates.py`

The `StabilityRegimeTree` class requires enhancements to support validity filtering (Gate A) and robust routing (Gate B).

### Required Methods

1.  **`predict_stability(Z)`**: Returns the stability score of the assigned leaf (max stability across experts in that leaf).
    ```python
    def predict_stability(self, Z: pd.DataFrame) -> pd.Series:
        """Returns the 'score_best' of the assigned leaf for each sample."""
        # ...
    ```

2.  **`predict_leaf_valid(Z)`**: Returns a boolean mask indicating if the assigned leaf is structurally valid (sufficient samples, consistent across folds).
    ```python
    def predict_leaf_valid(self, Z: pd.DataFrame, min_valid_frac: float = 0.8) -> pd.Series:
        """Returns True if the assigned leaf meets validity criteria."""
        # ...
    ```

3.  **`prune_experts_in_leaves()`**: Post-fit optimization to remove weak experts from leaves to ensure stability.

4.  **`merge_similar_leaves()`**: Post-fit optimization to collapse adjacent leaves with similar expert weights, reducing fragmentation.

---

## 2. Gate A: Pre-Geometry Filtering ("The Bouncer")

**Objective:** Filter unstable timestamps before geometry generation to prevent overfitting to noise.

*   **Integration Point:** `LabelBasedLayer2.execute`, *after* `_run_cross_asset_pipeline` (specialists) and *before* `orthogonal_label_generation`.
*   **Safety Requirement:** Training must avoid circularity. Inputs (`Z` and `S`) must be **Out-Of-Fold (OOF)** or generated via a **rolling window**. Folds must be **purged and time-blocked**.

### Inputs & Target
*   **Inputs (`Z`):** State features (e.g., `vol_regime`, `ms__*`).
*   **Inputs (`S`):** OOF Specialist Signals (`_raw_causal_specialist_predictions`).
*   **Target (`y`):** **"Best-Expert Utility Stability"**.
    *   Calculate utility for each expert $i$ at time $t$: $u_{i,t} = r_{t+1} \cdot \tanh(s_{i,t})$.
    *   Select best utility: $u^*_{t} = \max_i(u_{i,t})$.
    *   Target is the stability of this best utility over a rolling window:
        $$y_{target} = \frac{\text{RollingMean}(u^*)}{\text{RollingStd}(u^*)}$$
    *   *Goal:* Train the tree to identify states where *at least one* expert is consistently profitable.

### Implementation Steps
1.  **Data Prep:** Align `Z` and `S` (OOF). Compute `y_target` causally.
2.  **Training:** Fit `StabilityRegimeTree` using purged K-folds (`make_purged_kfold_folds`).
3.  **Optimization:** Call `prune_experts_in_leaves()` and `merge_similar_leaves()`.
4.  **Prediction:**
    *   Get validity mask: `valid_mask = gate_a.predict_leaf_valid(Z)`.
    *   Get stability scores: `scores = gate_a.predict_stability(Z)`.
    *   Final Filter: `keep_mask = valid_mask & (scores > threshold)`.
5.  **Application:** Pass `keep_mask` to `orthogonal_label_generation` to restrict event generation.

---

## 3. Gate B: Post-Race Routing ("The Router")

**Objective:** Dynamic Mixture-of-Experts routing for final prediction.

*   **Integration Point:** `LabelBasedLayer2.execute`, *after* `_train_geometry_batch` (OOF aggregation).
*   **Constraint:** Must handle sparse OOF predictions (different geometries cover different events).

### Alignment Strategy: Intersection Index
*   **Strict Alignment:** Build the training matrix only on timestamps where **all** candidate experts (or the top K) have predictions.
    *   `idx = intersection(expert_1.index, expert_2.index, ...)`
*   **Folds:** Construct **new** purged K-folds based on this aligned event index, *not* the global timeline. Purge size must match event horizon.

### Inputs & Target
*   **Inputs (`Z`):** State features aligned to the intersection index.
*   **Inputs (`P`):** OOF predictions of trained models (winners).
*   **Target (`y`):** Realized returns (aligned).

### Implementation Steps
1.  **Alignment:** Compute intersection index of `individual_geos`. Align `Z`, `P`, and `y`.
2.  **Training:** Fit `StabilityRegimeTree` (max_depth=3) on the aligned set using event-specific purged folds.
3.  **Optimization:** Prune and merge leaves.
4.  **Routing:** Call `gate_b.route(Z_full, P_full)`.
    *   Note: For inference (routing), we apply to the full set where data exists. The intersection constraint applies to *training* to ensure valid weight learning.
5.  **Output:** Use the routed signal as the final Layer 2 score. Attach metrics (`entropy`, `disagreement`).

---

## 4. Summary of Code Changes

### `src/training/steps/labeling/tree_based_causal_gates.py`
*   Implement `predict_stability`, `predict_leaf_valid`, `prune_experts_in_leaves`, `merge_similar_leaves`.

### `src/training/steps/labeling/label_based_layer_2.py`
*   **Gate A:** Inject training/filtering logic before event generation. Ensure `_raw_causal_specialist_predictions` are OOF/Rolling. Construct stability target.
*   **Gate B:** Inject alignment and training logic after OOF generation. Construct aligned folds. Replace simple averaging with `gate_b.route()`.
