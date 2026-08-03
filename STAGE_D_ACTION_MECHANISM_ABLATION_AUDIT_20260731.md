# Stage-D action mechanism ablation audit — 2026-07-31

## Decision

`RESEARCH_ONLY_NO_INCREMENTAL_MECHANISM_PROMOTION`

Canonical run: `data_perp/artifacts/stage_d_action_mechanism_ablation_20260731_v4/`.
Independent deterministic rerun: `data_perp/artifacts/stage_d_action_mechanism_ablation_20260731_v5/`.
All 11 files are byte-identical between v4 and v5. Runs v1-v3 are superseded.

The fixed population contains 108,139 unique candidates. Development OOF covers April-July 2024
(24,267 evaluated candidates); final frozen OOS covers August-November 2024 (31,258 candidates).
The candidate-set SHA-256 is `2088db5b78152be60a5b1b6a5f69500d6010e3e1a01b174dff0df5186d7b78b5`.

## Correctness repair

The v1 bootstrap used a sequential RNG arm by arm, so identical blocked arms received different
resamples. The repaired runner precomputes one deterministic UTC-day block-index matrix for every
split and side scope and reuses it for all arms and baselines. D2/D3 and D5/D6/D7/D8 now have exactly
identical bootstrap rows. The bootstrap estimand remains pooled bps per trade: sampled day sums divided
by sampled row counts, not an equal-day average.

The first repaired reruns also exposed tiny multithreaded LightGBM drift (mapped predictions up to
1.03e-12 bps; summary Huber loss up to 3.19e-11). Both selection and final models are now deterministic,
single-threaded, and column-wise. The focused suite passes 8/8 tests.

## Development admission

Only A1 path geometry was admitted into D9. Its paired development increment over A0 was +0.4364
bps/trade: long +0.4980, short +0.3797, two positive months, 108 symbols, concentration 0.1495, with
prediction improvement and preserved calibration. A2 failed prediction improvement and the long-side
gate; A4 had no positive month and failed the long-side gate; A5 was negative on both sides; A9 failed
the long-side gate. A0 and A1 retained selected features in every side/fold.

## Exact economics and prediction metrics

| Split | Arm | Policy net bps/trade | vs continue | vs exit | MAE bps | Spearman | ROC-AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Development OOF | D0 | 67.1838 | 48.4370 | 71.3989 | 178.5553 | 0.6148 | 0.8471 | 0.1540 |
| Development OOF | D9=A0+A1 | 67.6202 | 48.8734 | 71.8353 | 173.8668 | 0.6395 | 0.8679 | 0.1513 |
| Final OOS | D0 | 104.8948 | 80.1684 | 98.6619 | 130.2280 | 0.8238 | 0.9549 | 0.0766 |
| Final OOS | D9=A0+A1 | 104.8718 | 80.1455 | 98.6390 | 131.3164 | 0.8171 | 0.9559 | 0.0734 |

D9's paired final increment over D0 is -0.0229 bps/trade. By side it is +0.0050 long and -0.0544
short. By final month: August +0.0337, September -0.0757, October -0.0044, November -0.0466.
Thus it fails final aggregate, short-side, latest-month, and three-of-four-month incremental gates.

D9 itself remains strongly better than both fixed baselines. Its final paired UTC-day bootstrap versus
always-continue is +80.1490 bps/trade, 95% CI [75.5666, 85.0306], and versus always-exit is +98.5302,
95% CI [91.1354, 106.8569]. These numbers establish action-head value relative to naive policies; they
do not establish incremental value for A1 over the A0 action control.

## Conclusion

A0 is the frozen mechanism winner. A1 improved development OOF but did not transport incrementally to
the final period, so D9 must not be promoted. A2, A4, A5, and A9 are rejected as incremental groups.
A3, A6, A7, and A8 remain explicit `NOT_RUN` arms for source or lineage reasons. The decision threshold
is exactly zero predicted delta bps, with no top-k rule.
