# Causal Gates Analysis and Implementation Plan

## Executive Summary

This document provides a review and implementation plan for integrating a "Two Gates" causal architecture into `LabelBasedLayer2`. The goal is to enhance signal quality and stability by filtering unstable regimes early (Gate A) and dynamically routing predictions to the best experts later (Gate B).

### The Two Gates Architecture

1.  **Gate A (Pre-Geometry, "The Bouncer"):** A lightweight, causal filter applied *before* expensive geometry optimization. It uses raw state features and specialist signals to identify and discard "untradeable" or "unstable" regimes, saving compute and improving label quality.
2.  **Gate B (Post-Race, "The Router"):** An expressive Mixture-of-Experts (MoE) router applied *after* model training. It uses the Out-Of-Fold (OOF) predictions of the winning models to dynamically weight and combine them based on the current market regime.

---

## 1. Component Review: `tree_based_causal_gates.py`

The `StabilityRegimeTree` class is well-suited for both gates but requires minor enhancement to fully support the "filtering" use case of Gate A.

### Suitability Analysis
*   **Gate B (Routing):** `StabilityRegimeTree` is natively designed for this. The `route(Z, preds)` method accepts state features (`Z`) and expert predictions (`preds`), returning a weighted signal, disagreement, and entropy. **Status: Ready.**
*   **Gate A (Filtering):** The current implementation focuses on *routing* (assigning the best expert). To work as a filter, it needs to output a scalar "stability/suitability score" for the regime itself, rather than just choosing an expert. We can derive this from `LeafAssignment.score_best` (the stability score of the best expert in that leaf). **Status: Needs Enhancement.**

### Required Changes
*   **Add `predict_stability(Z)` method:**
    ```python
    def predict_stability(self, Z: pd.DataFrame) -> pd.Series:
        """
        Predict the stability score of the assigned regime (leaf) for each sample.
        Used for Gate A filtering.
        """
        leaf_ids = self.predict_leaf_ids(Z)
        scores = np.zeros(len(leaf_ids))
        for i, leaf_id in enumerate(leaf_ids):
            scores[i] = self.leaves_[leaf_id].score_best
        return pd.Series(scores, index=Z.index)
    ```

---

## 2. Integration Plan: `label_based_layer_2.py`

### Gate A: Pre-Geometry Filtering

**Objective:** Filter unstable timestamps before Triple Barrier Method (TBM) label generation.

*   **Integration Point:** In `LabelBasedLayer2.execute`, *after* `_run_cross_asset_pipeline` (where specialists are run) and *before* `orthogonal_label_generation`.
*   **Inputs:**
    *   `Z` (State Features): `ms__`, `vol_regime`, and other macro features constructed in `_run_cross_asset_pipeline`.
    *   `S` (Specialist Signals): `_raw_causal_specialist_predictions` (Step 2a outputs).
*   **Target (`y`):** Proxy for regime stability. A good candidate is **forward volatility-adjusted returns** (e.g., `returns / volatility`) or a binary "tradeable" label (e.g., `abs(ret) > threshold`).
*   **Mechanism:**
    1.  Construct `Z` and `S`.
    2.  Train `StabilityRegimeTree` (Gate A) on a historical window (or load pre-trained).
    3.  Call `gate_a.predict_stability(Z)`.
    4.  Create a boolean mask: `stability_mask = scores > threshold`.
    5.  Pass this mask (or filtered index) to `orthogonal_label_generation` to restrict event generation to stable regimes.

**Code Hook:**
```python
# Inside LabelBasedLayer2.execute
# ... after _run_cross_asset_pipeline ...

if self.causal_gate_enabled:
    tprint_info("🔒 Running Gate A (Pre-Geometry Filtering)...")
    # 1. Build State Z
    Z_gate = self._build_causal_gate_state_features(df)

    # 2. Get Specialist Signals S
    S_gate = self._raw_causal_specialist_predictions

    # 3. Train/Predict Gate A
    # (Assuming we use a simplified target like 1-day forward Sharpe proxy)
    y_target = (df['close'].pct_change().shift(-1) / df['volatility_1d']).fillna(0)

    gate_a = StabilityRegimeTree(max_depth=2, ...)
    gate_a.fit(Z_gate, S_gate, y_target, folds)

    stability_scores = gate_a.predict_stability(Z_gate)
    valid_regime_mask = stability_scores > self.gate_a_threshold

    tprint_info(f"   📉 Gate A filtered {len(df) - valid_regime_mask.sum()} unstable timestamps.")

    # 4. Apply to Event Generation
    # Pass valid_regime_mask to orthogonal_label_generation
```

### Gate B: Post-Race Routing

**Objective:** Final Mixture-of-Experts routing across trained predictors.

*   **Integration Point:** In `LabelBasedLayer2.execute`, *after* `_train_geometry_batch` (where OOF predictions `individual_geos` are collected) and *before* the final return.
*   **Inputs:**
    *   `Z` (State Features): Same `Z` as Gate A (or enhanced).
    *   `P` (OOF Predictions): `individual_geos` (dict of `uuid` -> `pd.Series`).
*   **Target (`y`):** Realized returns (or OOF returns computed from consensus).
*   **Mechanism:**
    1.  Align `Z` and `P` (OOFs are sparse/event-based, `Z` is continuous; align `Z` to events).
    2.  Train `StabilityRegimeTree` (Gate B) to maximize stability of the *routed* signal.
    3.  Call `gate_b.route(Z, P)`.
    4.  The output `routed_signal` becomes the final `l2_score`.
    5.  Metrics (`entropy`, `disagreement`) are added to the result payload.

**Code Hook:**
```python
# Inside LabelBasedLayer2.execute
# ... after OOF generation ...

if self.causal_gate_enabled:
    tprint_info("🔀 Running Gate B (Post-Race Routing)...")

    # 1. Align Z to OOF events
    Z_routing = Z_gate.reindex(oof_returns.index).fillna(0)

    # 2. Train Gate B
    gate_b = StabilityRegimeTree(max_depth=3, ...)
    gate_b.fit(Z_routing, individual_geos, oof_returns, folds)

    # 3. Route
    routing_results = gate_b.route(Z_routing, individual_geos)

    # 4. Override Score
    oof_scores = routing_results['signal']

    # 5. Attach Metrics
    result['gate_metrics'] = routing_results[['entropy', 'disagreement', 'leaf_id']]
```

---

## 3. Summary of Recommendations

1.  **Enhance `StabilityRegimeTree`:** Add `predict_stability(Z)` to expose the best-expert score for filtering.
2.  **Update `LabelBasedLayer2.execute`:**
    *   Insert **Gate A** logic before `orthogonal_label_generation` to filter the timeline based on regime stability.
    *   Insert **Gate B** logic after `_train_geometry_batch` to replace simple averaging with regime-conditional routing.
3.  **Data Flow:** Ensure `_raw_causal_specialist_predictions` is correctly populated and accessible before Gate A runs.

This design transforms Layer 2 from a static "bagging" ensemble into a dynamic, regime-aware causal pipeline.
