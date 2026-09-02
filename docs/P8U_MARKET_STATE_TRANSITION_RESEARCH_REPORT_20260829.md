# P8U Market-State / Transition Research — 2026-08-29

## Decision

**Do not change the P8U routed F72 / Under-F120 research-canonical stack.**

The market-state work created a causal, reusable representation and found
useful 2025 timestamp-level structure.  However, neither candidate-level
Meta rank nor the frozen state-quality gate produced a sufficiently stable
2026 improvement.  The state representation remains research infrastructure,
not an active score, admission, MC1, portfolio, or live-trading input.

This report is long-only, offline research.  It makes no exchange, live,
admission, or portfolio-production change.

## Contract and causality

The target-free hourly market-state lattice spans December 2024 through July
2026.  It has 3,202 derived fields, including predeclared fast/slow Kalman
pairs `(2,14)`, `(3,14)`, `(3,21)`, `(5,21)`, `(5,42)`, and `(7,42)` days.
Each pair exposes level, innovation, innovation z-score, gain,
posterior/prior variance, change, and normalized fast/slow differences.

The raw drivers are market-wide return dispersion/tails, breadth/downside
breadth, volatility level/dispersion, depth/spread, OI/funding, dependence,
spectral structure, and BTC-relative structure.  The state is joined to the
candidate panel only by decision timestamp, before candidate filtering and
before any policy/path outcome join.

All screens fit on labels whose availability timestamp precedes the held
period.  Correlation blocks are frozen from target-free December 2024--April
2025 values.  The state is evaluated as a **timestamp quality coordinate**:
it estimates the realised policy quality of the two candidates already chosen
by the target-free Base score.  It is never misused as a within-timestamp
candidate ranker.

## Screen and model funnel

1. Conditional CMI/IC screen, conditioned on Base-rank bands, retained 30
   state candidates: breadth (14), volatility (6), cross-return (4),
   dependence (3), liquidity (2), and spectral (1).
2. Random 3--8-field subspaces, pair synergy, and a beam search ran on strict
   OOF May--December 2025 timestamps.
3. A shallow Huber State-Meta model used Base top-two context plus candidate
   state fields.  Its initial six-field beam was tested independently in 2026.
4. A grouped permutation MDA froze correlation blocks/sub-blocks on
   pre-May-2025 target-free data and permuted only held state values.
5. Candidate-level residual Meta tests used the same nine target-free state
   and interaction inputs on strict-OOF 2026 folds.

## Results

### Timestamp-level State-Meta probe

| Period | Contract | Mean probe score | Mean residual IC | Top-20 timestamp spread | Positive months | Worst spread |
|---|---|---:|---:|---:|---:|---:|
| May--Dec 2025 strict OOF | Base context only | +0.9779 | +0.3404 | +161.54 bps | 8/8 | +91.99 bps |
| May--Dec 2025 strict OOF | Six-field State Meta | **+1.1011** | +0.3353 | **+186.68 bps** | 8/8 | **+102.83 bps** |
| Jan--Jul 2026 frozen confirmation | Base context only | **+0.8330** | **+0.3598** | +130.63 bps | **7/7** | **+62.50 bps** |
| Jan--Jul 2026 frozen confirmation | Six-field State Meta | +0.8366 | +0.3501 | +132.22 bps | 7/7 | +73.82 bps |

The broad six-field block had a positive but very small 2026 average effect.
It was therefore subjected to the higher-bar grouped MDA rather than being
promoted.

### Grouped MDA and frozen minimal contract

Only one sub-block met the predeclared MDA criterion: positive score and
Top-20-spread importance in at least 60% of May--December 2025 folds.

`ms_transition_volatility_level_fast3d_slow21d_kalman_level`

Its 2025 permutation impact was +0.0193 mean probe-score and +4.24 bps mean
Top-20 spread, in 5/8 folds.  That is real but modest.

| Jan--Jul 2026 frozen contract | Mean probe score | Mean residual IC | Top-20 timestamp spread | Positive months | Worst spread |
|---|---:|---:|---:|---:|---:|
| Base context only | **+0.8330** | **+0.3598** | +130.63 bps | **7/7** | **+62.50 bps** |
| One-field volatility state | +0.8183 | +0.3237 | +131.29 bps | 6/7 | **-12.61 bps** |
| Delta | -0.0148 | -0.0361 | +0.66 bps | -1 month | -75.10 bps |

The one-field gate fails portability: it is effectively unhelpful on average,
has no rank information in June, and makes its selected Top-20 state negative.
It is rejected.

### Candidate-level Meta integration

The same target-free state block was also evaluated as a normal residual Meta
input on January--July 2026.  It did not add usable cross-sectional power.

| Target arm | SStableMeta | Residual IC | CMI conditional on Base | Mean Top-2 substitution | Worst weekly Top-2 |
|---|---:|---:|---:|---:|---:|
| Magnitude residual | +0.00984 | +0.06110 | +0.03005 | -2.49 bps | -12.32 bps |
| Signed calibration state | -0.12252 | +0.04802 | +0.02682 | -21.12 bps | -47.58 bps |

Neither target advances to MC1, admission, or constrained-portfolio replay.

## Related lower-standard SHAP audit

The requested lower-standard assessment was already completed on the strict
F72 SHAP ledger.  In addition to the three 12/12 core fields (signed balance,
positive total, and `mark_perp_dislocation` contribution), recurrent but
diagnostic-only candidates were entropy, `mark_trigger_risk_5h`,
`leverage_build`, and `prog_eff_24`.  Conditional-only candidates were
absolute-total, top-1/top-3 shares, 5/10-hour mark-gap volatility, and
distance from the prior-day low.

The only directionally useful standalone test was SHAP entropy: about
+2.74 / +2.58 / +2.56 / +0.94 bps at Top-1/2/5/10, with positive monthly
deltas in 5/7, 5/7, 4/7, and 6/7 months.  It is too small to promote.
All remaining relaxed candidates were mixed or tip-only.

## Final status

- B0/E/T are not an active path in this work.
- No Base, Meta, MC1, admission, portfolio, exit-policy, or live contract is
  changed.
- The materialised lattice is retained because it is causal, reusable, and
  may support a later predeclared risk/episode hypothesis.
- Any new use requires a newly frozen training hypothesis and a later,
  untouched confirmation—not a re-selection from these results.

## Reproducibility artifacts

- State lattice: `data_perp/artifacts/strict_r3_p8u_market_state_transition_dec24_jul26_20260829_v1`
- Conditional screen: `data_perp/artifacts/strict_r3_p8u_market_state_transition_screen_dec24_jul26_20260829_v2`
- 2025 timestamp probe: `data_perp/artifacts/strict_r3_p8u_market_state_timestamp_probe_maydec25_20260829_v1`
- Grouped MDA: `data_perp/artifacts/strict_r3_p8u_market_state_timestamp_group_mda_maydec25_20260829_v1`
- Frozen one-field 2026 confirmation: `data_perp/artifacts/strict_r3_p8u_market_state_timestamp_mda1_janjul26_confirmation_20260829_v2`
- Candidate-level Meta grid: `data_perp/artifacts/strict_r3_p8u_market_state_meta9_target_grid_janjul26_20260829_v1`
- Existing SHAP audit: `docs/P8U_F72_BASE_STATE_META_INPUT_AUDIT_20260829.md`

The `strict_r3_p8u_market_state_timestamp_group_mda_janjul26_confirmation_20260829_v1`
directory is a non-result from a startup invocation that applied the 2025
selector again to 2026.  It must not be used.  The `_mda1_..._v2` artifact is
the valid frozen-contract confirmation.
