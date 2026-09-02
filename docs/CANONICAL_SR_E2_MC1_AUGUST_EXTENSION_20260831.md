# Canonical S/R + E2 MC1 input study — August extension

## Scope

This is an offline, portfolio-constrained extension of the June–July input study. It covers **2026-08-01 00:00 through 2026-08-18 21:00 UTC** only: the archived paired BCF/current-v5 score source ends there. It is not a full calendar-month result, a live-execution reconstruction, or a promotion decision.

The policy outcome is the retained source-aligned 15-minute parent policy: long-only, entry one hour after the signal close, 4.1520 ATR stop, 2.3262 ATR trailing activation, 0.10237 ATR giveback, 12-hour timeout, and 100 bps cost applied once. The later rich/smooth live-exit policy is intentionally out of scope.

All arms use family-specific prequential absolute policy-net-EV MC1 maps (HGB depth 2, 80 iterations, learning rate 0.04, L2 20, minimum leaf 100, seed 1729), a 21-day 10%-trimmed prior-resolved score-band shift, dual BCF/current mapped-EV admission at +50 bps, BCF-EV auction priority, and the controlled long-only portfolio (7x, 10% slots, two new entries, eight concurrent positions, 80% wallet cap).

## Results

| Arm | Portfolio trades | Net EV/trade | Total net bps | Worst week | Max drawdown |
|---|---:|---:|---:|---:|---:|
| Frozen retained historical control | 151 | +164.34 | +24,815.16 | +14.96 | -36.97% |
| C0 refit core | 23 | +555.70 | +12,781.01 | +365.69 | -8.14% |
| C1 core + causal S/R | 107 | +225.48 | +24,126.41 | +129.81 | -16.80% |
| C2 core + 15m E2 | 20 | +377.45 | +7,548.90 | +59.06 | -0.87% |
| C3 core + S/R + 15m E2 | 84 | +221.20 | +18,581.08 | +148.19 | -3.20% |

Relative to the valid post-February refit baseline C0, C1 adds 84 portfolio trades and +11,345.40 total bps, but lowers EV/trade by 330.22 bps. C2 and C3 likewise trade more sparsely/less sparsely than C0 respectively, but neither is a promotion on this partial block alone. The frozen retained control provides the broadest participation and the highest total August contribution in this extension.

## Availability and causal controls

- 51,369 target-free archived candidate rows were retained before outcome joining.
- 35,075 rows had both score families' complete decision-time MC1 core inputs; 16,294 current-v5 rows were explicitly unavailable because base/consensus input was incomplete, not because of a future outcome.
- The parent-policy label materialisation yielded 34,419 resolved paths. The remaining 656 rows were invalid due to five unreadable local 15-minute source files; the materialiser used existing local one-minute archives only for those files and marked that provenance. Invalid rows were excluded before both fitting and portfolio capacity.
- 15-minute E2 was available on 31,640 of 35,075 held August rows (90.2%) using only earlier resolved months. S/R snapshots were present for 1,382 rows (3.9%) and otherwise represented as unavailable rather than imputed.
- The run made no exchange calls or order submissions. Target-free availability and prediction panels are persisted separately from the joined outcomes.

## Interpretation

August is supportive but not decisive. Every arm has positive realised EV and positive weekly means across the available slice, yet the refit arms use a very small number of trades in their core-only variants and the source stops before the end of the month. Keep the study challenger-only; do not infer live-stack performance or replace the canonical mapping from this result.

## Reproduction

- Runner: `scripts/run_canonical_sr_e2_mc1_august_extension.py`
- Input preparation: `scripts/prepare_canonical_sr_e2_august_extension_inputs.py`
- Policy-label materialisation: `scripts/materialize_strict_r3_frozen_policy_labels_v2.py`
- Focused contract test: `tests/test_canonical_sr_e2_mc1_august_extension_contract.py` (5 passed)
- Result directory: `data_perp/artifacts/canonical_sr_e2_mc1_input_ablation_august_20260831_v3`
