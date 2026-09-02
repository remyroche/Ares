# H4 trailing-activation HPO — 2026-09-01

## Decision

**Do not promote.**  The selected 2025 activation controller improved the
strict-OOF selection period materially, but gave back 417 bps of total net
outcome on the single, frozen 2026 confirmation.  The current retained H4
continuation policy remains unchanged.  No live, admission, MC1, C1 S/R,
geometry, portfolio, or execution artifact was changed.

This is a focused successor to
`H4_ACTUATOR_COUNTERFACTUAL_ABLATION_20260901.md`.  It tests only trailing
activation because giveback had weak headroom and stop distance had zero exact
counterfactual variation in the preceding screen.

## Receipt

- Runner: `scripts/run_causal_sr_h4_activation_hpo.py`
  (`09ede699a0f28d7cac1c2d0508f7f23eaecf39786f3f0ea7e2a8db71fc72d182`)
- Output: `data_perp/artifacts/causal_sr_h4_activation_hpo_2025oof_2026confirm_20260901_v1`
- Run manifest SHA-256:
  `d278c0fd952dc3801f084154372316b582506bd90bd019a103b63115ea799b2d`
- 2025 exact summary SHA-256:
  `feb3743ef2605a19498969aebaabc9f0de72f42f32200fc3e335c97b00f981c0`
- 2026 frozen summary SHA-256:
  `5644c6862dafd088329a63ef9556384eb60fb8b28f02f95144c2d84c8881491d`

## Causal / population contract

- Long-only causal S/R plus paired BCF/current-v5 MC1 route; exact resolved
  rich-parent one-minute paths only.
- Labels: paired MC1 at least +40 bps, without portfolio constraints, using
  the existing first/middle/last target-free state sample.  Each row becomes
  available only after its own H12 parent outcome resolves.
- Economic assessment: paired MC1 at least +50 bps and the unchanged normal
  global chronological constrained auction.
- Selection: June–December 2025 monthly strict-prior OOF.  A held month sees
  only earlier, fully-resolved labels.
- Confirmation: models, authority threshold, and multiplier are selected in
  2025 then frozen.  June–August 2026 is used once for confirmation only.
- Inputs: all existing numeric, target-free H4 state fields.  No later
  feature-selection receipt or future outcome field is used.
- Controller: only trailing activation; all actions are tightening-only and
  take effect from the next completed 15-minute interval.
- No exchange calls were made.

## HPO funnel

Six shallow LightGBM regressors were screened by strict-OOF label ranking:

- targets: max advantage of 0.65/0.80 tightening, or direct 0.80 advantage;
- losses: L2 and Huber;
- geometries: depth 2/3/4, 4/7/15 leaves, 5%/10% support floors;
- fixed causal weighting: inverse number of sampled states per candidate.

The three strongest 2025 label models were then assessed in the exact
portfolio replay across a predeclared authority grid:

| Authority | Threshold | Activation multiplier |
|---|---:|---:|
| q15_m080 | +15 bps | 0.80 |
| q25_m080 | +25 bps | 0.80 |
| q35_m080 | +35 bps | 0.80 |
| q25_m070 | +25 bps | 0.70 |
| q25_m090 | +25 bps | 0.90 |

The winner was selected by 2025 total net bps / absolute max drawdown,
with net bps per trade and worst week as secondary ordering fields.

## Exact constrained results

### 2025 strict-prior OOF (Jun–Dec)

| Arm | Trades | Net bps/trade | Total net bps | Sortino | Max DD | Worst week | CVaR10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Parent | 4,424 | +114.24 | +505,405 | 0.495 | −0.516 | +0.194 | −430.23 |
| **Winner: tight-max L2 d3/l7, q25, ×0.70** | **4,482** | **+117.39** | **+526,143** | **0.548** | **−0.494** | **+0.393** | **−405.06** |
| Direct-0.80 L2 d3/l7, q25, ×0.70 | 4,469 | +117.70 | +525,981 | 0.544 | −0.494 | +0.393 | −408.44 |
| Direct-0.80 L2 d4/l15, q25, ×0.70 | 4,466 | +117.60 | +525,193 | 0.544 | −0.494 | +0.393 | −410.01 |

The selected model was active at 30,185 observable states.  Its improvement
is not a trade-count reduction: it increased accepted entries by 58 while
improving per-trade EV, Sortino, drawdown, worst week, and tail loss in the
strict-OOF period.

### Frozen 2026 confirmation (Jun–Aug)

| Arm | Trades | Net bps/trade | Total net bps | Sortino | Max DD | Worst week | CVaR10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Parent | 558 | +136.97 | +76,430 | 0.686 | −0.098 | +0.026 | −267.46 |
| Frozen 2025 winner | 558 | 136.22 | 76,013 | 0.690 | −0.098 | +0.026 | −267.46 |

The frozen controller is lower in each available month: June −264.6 bps,
July −54.3 bps, and August −98.6 bps; total delta −417.4 bps.  Its small
Sortino increase occurs without a downside or economic improvement and is
not sufficient evidence for promotion.

## Conclusion

The activation response is learnable in 2025 but currently lacks portable
economic benefit.  Further activation work should not tune against this 2026
block.  It should either await a new untouched period or use a separately
predeclared older-window cross-era validation before another 2026 test.
