# Frozen stack: hierarchical side/regime conversion ablation

Date: 2026-08-10  
Input: frozen ATR2 specialists → q4h×side ordinal residual LambdaRank  
Rows: 388,494 strict-OOS candidates, July–November 2024

## Purpose

The causal 21-day admission audit showed that the current score-to-net mapping
does not transport: it admitted only 1.17% of rows and realised −244.21 net
bps/trade. This ablation tests whether a conservative common-bps calibration
using side and causal soft-regime information can repair the score without
refitting the base, specialist, or residual model.

## Causal contract

For each UTC decision day, calibration uses only rows with
`outcome_resolved_at = decision_ts + 13h` strictly before that day's first
decision timestamp. The calibration is one shared mapping, not a set of local
experts. The soft regime inputs are the four causal state probabilities:

`regime_p_calm`, `regime_p_trend`, `regime_p_stress`, `regime_p_transition`.

The hierarchy is:

- C0: global additive correction;
- C1: global + strongly shrunk side correction;
- C2: global + side + strongly shrunk side×soft-regime correction;
- C3: C2 plus strongly shrunk affine slope deviations.

All calibration corrections are fit prequentially. Final evaluation remains a
single pooled-global ranking; there are no per-timestamp quotas or side quotas.

Implementation: `scripts/run_frozen_hierarchical_conversion_ablation.py`  
Artifacts: `data_perp/artifacts/frozen_hierarchical_conversion_ablation_20260810_v1/`

## Pooled global tails

Net bps/trade, with the existing fixed 100-bps cost applied once:

| Arm | Top 0.5% | Top 1% | Top 2% | Top 5% | Top 10% |
|---|---:|---:|---:|---:|---:|
| Frozen raw control | −23.86 | **−7.30** | **+16.01** | **+8.89** | **−37.63** |
| C0 global | −19.88 | −9.98 | +4.32 | −6.53 | −45.85 |
| C1 side | −299.67 | −323.59 | −103.59 | −150.53 | −85.90 |
| C2 side×soft regime | −385.02 | −280.64 | −104.00 | −151.11 | −89.34 |
| C3 affine side×soft regime | −297.96 | −193.20 | −86.47 | −116.59 | −104.68 |

No calibrated arm improves the frozen control at top 1%, top 5%, or top 10%.
C0 should preserve ordering only if its correction is constant, but the
prequential daily correction is time-varying and therefore changes pooled
cross-day ordering; it still loses 15.42 bps at top 5%.

## Monthly global top-5 net

| Arm | Jul | Aug | Sep | Oct | Nov | Mean | Worst |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen raw control | −51.55 | −171.07 | −58.25 | −81.07 | **+11.00** | −70.19 | −171.07 |
| C0 global | −60.81 | −151.99 | −57.05 | −81.33 | **+11.68** | −67.90 | −151.99 |
| C1 side | −314.28 | −315.45 | −125.46 | −114.04 | −3.10 | −174.47 | −315.45 |
| C2 side×soft regime | −276.59 | −329.80 | −121.66 | −115.66 | −5.66 | −169.87 | −329.80 |
| C3 affine side×soft regime | −183.50 | **+68.05** | −73.37 | −65.26 | −231.89 | −97.19 | −231.89 |

C3 improves August in isolation but catastrophically reverses November. This is
regime calibration instability, not a portable improvement.

## Per-side top-5 net

| Arm | Long | Short |
|---|---:|---:|
| Frozen raw control | **+17.25** | −195.71 |
| C0 global | +7.16 | −200.15 |
| C1 side | +14.36 | −122.28 |
| C2 side×soft regime | −1.34 | −116.06 |
| C3 affine side×soft regime | −20.98 | −84.16 |

C1–C3 reduce the short-side loss only by moving the global ranking toward
shorts; they simultaneously damage long performance and remain deeply negative
on short. This is not a valid pooled-global repair.

## Decision

`NO_HIERARCHICAL_CONVERSION_ADVANCE`.

The current causal regime fields are available and leakage-safe, but their
economic meaning shifts across transport months and sides. A shared additive or
affine calibration layer cannot recover a stable common-bps ranking. Keep the
unmodified frozen raw residual score as the reference control; do not add C0–C3
to production or to the next model-selection baseline.

The next justified experiment is not another calibration-parameter sweep. It
should test a conversion target whose training population is explicitly
side/regime-support aware, with a fail-closed OOD rule and equal-month/worst-
month selection. Any candidate must beat the raw control at pooled top-5 while
not worsening the −171.07 bps worst month and must separately clear the short
side.
