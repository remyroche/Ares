# Strict-R3 Long-Only EV Admission Cold-Start Audit — 2026-08-12

## Scope

This audit keeps the schema-v2 strict-R3 scoring stack, frozen geometry/K9
representation, current SimplePolicyOptimiser-selected policy outcome, and
portfolio rules fixed.  It tests only causal admission-map handling after a
28-day producer refit, then a deterministic ordering correction inside equal
mapped-EV bins.

Policy outcome contract:

- signal decision at close; entry one hour later;
- H12 timeout;
- 100 bps cost exactly once;
- frozen pre-2025 SimplePolicyOptimiser winner: SL `4.1520006 ATR`, trailing
  activation `2.3262249 ATR`, giveback `0.1023720 ATR`.

All rows are long-only, target-free when scored, strict-prequential, and use
the single frozen October--December 2024 geometry/K9 bundle.

## Diagnosis

The producer score still ranks opportunities after refits.  The failure is
the level map, not a reversal of score information:

- the exact-producer reserve bridge has 16 zero-admission weeks in Jan--Jul
  2026 and no 50-bps admissions from June onward;
- its May 21 producer has a maximum mapped EV of only `48.20 bps` in early
  June; the June 18 producer has a maximum of `-7.60 bps`;
- raw top-score tails remain positive in July, but the prior reserve and
  causal residual correction both remain depressed after the June shock.

This is a causal calibration/regime-level problem.  It cannot be honestly
repaired at the start of July by using July outcomes.

## Admission-map ablations, 2026 Jan--Jul

`E050` means mapped expected policy net at least `+50 bps`.

| Arm | Raw admissions | Raw net bps/trade | Constrained trades | Constrained net bps/trade | Sortino | Max drawdown | Worst week |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exact-producer reserve bridge | 11,400 | +83.13 | 1,347 | +128.74 | 0.647 | -24.8% | -5.5% |
| Same-model 42-day reserve seed | 14,002 | +86.99 | 2,265 | +120.81 | 0.366 | -79.8% | -66.7% |
| 50% exact / 50% seeded expected-EV blend | 11,155 | +110.31 | 2,016 | +134.31 | 0.391 | -64.0% | -14.8% |
| 75% seeded expected-EV blend | 11,455 | +113.17 | 2,111 | +129.29 | 0.413 | -66.5% | -54.7% |
| Exact primary + seeded fallback at CDF >= .990 | 15,385 | +98.60 | 2,263 | +130.19 | 0.388 | -68.4% | -32.3% |
| Exact primary + seeded fallback at CDF >= .995 | 13,839 | +97.93 | 2,029 | +126.80 | 0.341 | -48.3% | -47.8% |

The seeded methods restore post-refit activity, including July, but do so by
admitting more aggressively through the June shock.  None improves the
exact-reserve control on both quality and risk.  They remain research-only.

## Accepted correction: score tie-break inside equal-EV bins

The monotone 20-bin EV map necessarily makes many admissible candidates have
identical expected bps.  In the exact-reserve 2026 E050 arm:

- 37.4% of admitted candidates lay in an equal-EV bin;
- 57.6% of decision timestamps contained at least one such tie;
- prior behavior ordered those ties by candidate identity.

The repaired auction changes only this secondary ordering:

```text
primary: mapped expected policy net bps
secondary: same fitted producer's final_score
tertiary: fixed friction then deterministic identity
```

`final_score` is already the same-producer prior-42-day CDF.  It is causal at
the decision, and it does not alter admission, the expected-EV value,
rank-based position size, model score, map, or exit policy.

### Matched constrained portfolio result

| Period | Arm | Trades | Net bps/trade | Sortino | Max drawdown |
|---|---|---:|---:|---:|---:|
| 2025 Apr--Jul | Exact E050 control | 1,678 | +131.27 | 0.392 | -44.8% |
| 2025 Apr--Jul | Exact E050 + CDF tie-break | 1,684 | +152.62 | 0.416 | -48.8% |
| 2026 Jan--Jul | Exact E050 control | 1,347 | +128.74 | 0.647 | -24.8% |
| 2026 Jan--Jul | Exact E050 + CDF tie-break | 1,374 | +138.46 | 0.635 | -33.6% |

The tie-break is portable across both evaluated eras and is the only change
from this audit eligible for a future canonical-stack update.  It needs an
untouched forward confirmation before promotion.  It does **not** resolve the
June shock or exact-reserve admission drought; this is intentional.

## Artifacts

- Full strict 2026 score ledger:
  `data_perp/artifacts/strict_r3_lockstep_exactreserve_monthstore_strictfull_long_2026_janjul_20260812_v7/`
- Exact-reserve maps and threshold control:
  `data_perp/artifacts/strict_r3_lockstep_exactreserve_calibrated_long_2026_janjul_20260812_v1/`
  and `data_perp/artifacts/strict_r3_lockstep_exactreserve_thresholds_long_2026_janjul_20260812_v1/`.
- Same-model reserve-seeded map under the optimized policy:
  `data_perp/artifacts/strict_r3_lockstep_exactreserve_monthstore_strictfull_long_2026_janjul_reserve_seeded_optimised_policy_20260812_v1/`.
- Map blend and extreme-fallback research ablations:
  `data_perp/artifacts/strict_r3_lockstep_causal_admission_blend_long_2026_janjul_20260812_v1/`
  and `data_perp/artifacts/strict_r3_lockstep_reserve_seeded_extreme_fallback_long_2026_janjul_20260812_v1/`.
- 2026 tie-break replay:
  `data_perp/artifacts/strict_r3_lockstep_tiebreak_portfolio_exact_long_2026_janjul_20260812_v1/`.
- 2025 Apr--Jul independent control and tie-break replays:
  `data_perp/artifacts/strict_r3_lockstep_tiebreak_portfolio_exact_long_2025_aprjul_control_20260812_v2/`
  and `data_perp/artifacts/strict_r3_lockstep_tiebreak_portfolio_exact_long_2025_aprjul_tiebreak_20260812_v2/`.

## Code changes

- `scripts/ablate_strict_r3_exact_reserve_thresholds.py` now accepts an
  explicit, immutable causal map mode and reconstructs provenance from the
  lockstep hashes where necessary.
- `scripts/replay_strict_r3_tail_health_portfolio.py` correctly preserves
  fail-closed unmapped rejections and optionally enables the score-only
  secondary tie-break.
- `scripts/replay_strict_r3_forward_portfolio.py` consumes the optional
  tie-break field only after primary mapped-EV rank has tied.
- The new map blend and extreme-fallback scripts are ablation-only and have
  not been wired into the canonical inference path.
