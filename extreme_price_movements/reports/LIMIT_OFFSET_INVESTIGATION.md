# Limit Offset and Execution Entry Modeling Review

This document summarizes the findings from the review of limit offset target economics and operational mode status in `position_sizer_v2.py` and `limit_order_pricer.py`.

## 1. Offset Contract Definition

- **Units:** Basis points (bps) exclusively. (e.g. 5.0 bps, 50.0 bps).
- **Bounds:** Globally unified. Default limits are explicitly clamped to `[5.0, 50.0]` bps.
- **Sign Convention:** Positive means an improved price relative to the signal price (e.g. buying lower or selling higher).
- **Behavior:**
    - A wider limit offset means greater execution price improvement, but lower fill probability.
    - A tighter offset represents greater market urgency, higher participation, but more slippage/fees relative to pure mid-point.

## 2. Heuristic vs ML Path Readiness

- **Current Status:** The machine-learning implementation (`LayerCExecutionOptimizer.fit_limit_offset` in `position_sizer_v2.py`) was structurally disconnected and operated as a stub. It was receiving an unspecified target without clear construction rules, applying silent bounds of `[0.0, 5.0]`, and using suboptimal shared features.
- **Operational Mode:** The configuration variable `limit_offset_mode` now explicitly controls the routing. We default to `limit_offset_mode="heuristic"` and map all operational limit offsets to `predict_offset()` in `limit_order_pricer.py`.
- **Target Constraints:** The ML path (`offset_mode="ml"`) is locked behind the requirement to configure a valid, forward-looking `limit_offset_target_mode` (e.g., `utility_grid_search`). This formally prevents using max-excursion hindsight proxies which would cause backtest survivorship bias.
- **Recommendation:** Do not use `offset_mode="ml"` until a robust grid-search or simulated-fill tradeoff objective is written to produce `y_offset`.

## 3. Economic Alignment and Urgency Trade-off

The heuristic path in `limit_order_pricer.py` (`estimate_entry_limit_offset`) successfully captures operational execution assumptions:
- **Urgency / Expected MFE:** Larger expected favorable excursions (`mfe_hat`) linearly reduce the requested offset (increasing aggressiveness/urgency).
- **Adverse Moves / Expected MAE:** High `mae_hat` causes wider limits to be placed (seeking safety and better entries before the price resolves).
- **Confidence:** Prediction uncertainty tightens offsets (`confidence_penalty`), preferring execution certainty when the edge is less sharp.
- **Microstructure Additions:** The current logic is solid, but the `limit_order_pricer.py` logic does not yet take direct advantage of real-time depth imbalances or book-decay (`impact_z`, `rng_z`) at a tick level. If ML mode is reactivated, prioritizing these order-book features (now correctly wired via `limit_offset_sizer`) would materialistically improve predictive participation modeling.
