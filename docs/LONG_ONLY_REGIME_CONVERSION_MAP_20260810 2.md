# Long-only causal regime-conditioned conversion-map ablation

## Pipeline used

### 1. Frozen upstream stack

The input score was not retrained or retuned. It came from the frozen ATR2/q4h specialist-residual contract:

- seven frozen, side-specific specialist views;
- 68 causal fields per view;
- specialist target: ATR-spacing 2.0 economic grade;
- specialist query: 4-hour bucket × side;
- residual target: per-row net residual in bps around the prequential base expected-net map;
- residual learner: native LightGBM LambdaRank;
- ranking: pooled global ranking, not per-timestamp top-k.

The input population was the strict-OOS primary history (September 2023–February 2024) plus transport rows (July–November 2024). Only `side_name == long` rows were retained. Shorts did not enter the join, calibration, fitting, or metrics.

### 2. Causal context join

Each frozen prediction was joined one-to-one to the existing causal regime ledger. The mapping used only decision-time fields:

- `regime_p_calm`, `regime_p_trend`, `regime_p_stress`, `regime_p_transition`;
- `regime_entropy`;
- `regime_transition_onset_proxy`;
- `regime_state_duration_hours`.

The four regime probabilities were checked to be finite, non-negative, and to sum to one on every row.

### 3. Economic target and lineage

The calibration target was the existing exact H12 `net_bps` outcome after the declared cost treatment. A label became usable at:

`outcome_resolved_at = decision_timestamp + 13 hours`.

At every daily anchor, calibration fitting was restricted to rows satisfying:

`outcome_resolved_at < anchor_timestamp`.

Thus no same-day unresolved outcome, future row, or current test label entered the map. The frozen score itself was never changed during fitting.

### 4. Mapping arms

The following fixed, non-HPO arms were compared:

- **C0 global:** prior-resolved global additive correction;
- **C1 side:** side correction; it is intentionally degenerate with C0 in this long-only run;
- **C2 side × soft regime:** strongly shrunk expectation of regime corrections;
- **C3 hierarchical affine soft regime:** C2 plus a heavily shrunk score slope/intercept correction.

Shrinkage constants were fixed at 5,000 global rows, 1,500 side rows, 3,000 effective regime rows, with a 0.50 regime-weight cap. This is a calibration map, not a collection of local experts.

### 5. Evaluation

For each arm, the mapped score was ranked globally and evaluated at top 0.5%, 1%, 2%, 5%, and 10%. Reported metrics are gross bps/trade, net bps/trade, and rank IC. Results were also computed by era and month.

## Results

All-era long-only top-5 net EV:

| Arm | Net bps/trade | Rank IC |
|---|---:|---:|
| Raw control | **−50.55** | **0.0234** |
| C2 side × soft regime | −59.60 | 0.0093 |
| C0 global | −69.27 | 0.0107 |
| C1 side | −69.27 | 0.0107 |
| C3 affine soft regime | −94.01 | 0.0182 |

The mapping is not an advance. It reduces the global top-5 score quality and does not repair the top-10 tail. C2 is the least damaging mapped arm, but remains worse than the raw control.

Top-5 net EV by era:

| Era | Raw | C2 |
|---|---:|---:|
| Sep–Oct 2023 | −73.11 | −84.49 |
| Nov–Dec 2023 | −69.13 | −82.55 |
| Jan–Feb 2024 | −10.37 | −9.41 |
| Jul–Oct 2024 | −63.79 | −86.77 |
| Nov 2024 | −13.71 | −1.55 |
| All eras | **−50.55** | **−59.60** |

C2 helps slightly in January–February and November, but worsens the older era and July–October. It therefore confirms, rather than solves, the regime-transport problem.

## Decision

Reject C0–C3 for the long-only production ranking. Keep the raw frozen score as the control. The negative result is useful: the existing causal regime probabilities do not provide enough stable information to convert this score into a portable common-bps ranking through a simple prior map.

The next step should not be another mapping-constant sweep. It should be an information audit:

1. Measure whether the regime fields distinguish profitable versus unprofitable score tails within each month, conditional on the frozen score.
2. Add only causal, varying trust/conversion fields that are available at decision time and belong to the meta contract.
3. Train a small long-only reliability model on the residual after a prior map, with selection on an earlier era and a fully untouched later era.
4. Require improvement in top-5 **and** top-10 net EV, positive rank IC, and no catastrophic worst-month result before promotion.

If those fields remain non-incremental, the bottleneck is the upstream candidate/score information contract rather than calibration.

## Artifacts

- Runner: `scripts/run_long_only_regime_conversion_map.py`
- Predictions: `data_perp/artifacts/long_only_regime_conversion_map_20260810_v1/predictions.parquet`
- Metrics: `data_perp/artifacts/long_only_regime_conversion_map_20260810_v1/metrics.parquet`
- Manifest: `data_perp/artifacts/long_only_regime_conversion_map_20260810_v1/manifest.json`
