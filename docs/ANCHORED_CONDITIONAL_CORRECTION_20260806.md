# Anchored conditional correction audit (2026-08-06)

## Scope and control

This is a matched audit on the exact frozen incumbent. It does not retrain or
substitute the current ranker.

- 388,494 candidates, July--November 2024.
- Global pooled top-k ranking after the existing common-bps mapping.
- The incumbent score is `incumbent_score` from
  `frozen_residual_query_hpo_20260810_v1`.
- Frozen score parity is exact: maximum absolute score difference is 0.0 over
  all 388,494 candidate IDs.
- The H12 labels already include the single declared 100-bps cost. No second
  cost is subtracted.
- July has no matured training history in this run and is therefore an
  anchor-only control; August--November use chronological prior history.

## 1. Family/path discovery

The portable condition layer contains three long and three short causal pair
families. A pair is a soft path activation: each leg uses a causal q25/q75
sigmoid membership and the pair membership is raised to exponent 1.5 for
specialist weighting. The names (low/high) are descriptive, not hard-coded
trade rules.

| Side | Pair path | Effective rows | Supported months | Hard joint share | Standalone rank IC | Standalone top-1 net | Standalone top-5 net | Standalone top-10 net |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Long | OI expansion/compression 96h high × RV peer residual low | 2,696 | 9 | 6.7% | 0.087 | -30.6 | -31.9 | -37.1 |
| Long | Return per OI change 1h low × order-book spread residual low | 2,648 | 9 | 5.8% | 0.047 | -46.7 | -48.1 | -51.6 |
| Long | Downside breadth intensity low × RV peer residual low | 2,837 | 8 | 6.1% | 0.080 | -20.9 | -35.5 | -38.8 |
| Short | Cross-asset lower tail low × OI expansion/compression 24h high | 2,898 | 5 | 6.4% | 0.050 | -77.8 | -84.5 | -87.0 |
| Short | Funding z low × tail asymmetry/order-book spread z low | 3,166 | 9 | 5.4% | 0.078 | -59.7 | -83.5 | -85.2 |
| Short | RV peer residual low × impact per OI intensity high | 4,015 | 9 | 4.1% | 0.072 | -86.8 | -86.3 | -89.2 |

The selected paths therefore have discovery recurrence and non-trivial
support, but none is a standalone economic signal. The path layer is useful as
context about which local geometry is active, not as a direct trade score.

## 2. Specialist outputs and consensus controls

Each family emits raw score, within-query rank, membership, gated rank,
innovation rank, uncertainty and OOD. The six families are side-local and use
4-hour × side LambdaRank queries. Their target is the canonical per-row net
residual, ordinalized at -150/-50/+50/+150 bps, followed by a prior-resolved
side-local monotone EV map.

Current global H12 net controls (top 1% / 5% / 10%):

| Control | Top 1% | Top 5% | Top 10% |
|---|---:|---:|---:|
| Anchor-only control | -12.71 | +6.94 | -50.25 |
| Frozen incumbent stack | -7.30 | +8.89 | -37.63 |
| Equal pair average | +40.19 | -72.00 | -67.61 |
| Regularized linear blend | +60.82 | -47.74 | -83.03 |
| Full-context GMM | +45.41 | -70.24 | -83.71 |

This is the sense in which the family layer can appear useful only as
consensus weighting: averaging or weighting can improve a very small top-1
tail, while destroying the broader top-5/top-10 ranking. It is not a portable
positive-EV specialist score.

## 3. Conditional feature contract

Before selection the MLP sees 167 causal fields:

- 3 anchor/base scalars (`base_score`, `base_ev_bps`, `incumbent_score`);
- 12 frozen head scores plus percentile geometry and five consensus summaries;
- 42 family/path fields (six families × seven outputs);
- 6 family × anchor interactions;
- 58 causal regime/context fields (regime probabilities, entropy, transition,
  liquidity, breadth, funding, OI, cross-asset and deleveraging context);
- 40 prior-only reliability fields (anchor residual/failure and family
  residuals over 1/3/7/14/28-day windows).

The fold-local fit-only screen retains 96 fields. Selected-field counts by
category are:

| Fold | Anchor/base | Consensus geometry | Family/path | Family-anchor | Causal context | Recent anchor reliability | Recent family reliability |
|---|---:|---:|---:|---:|---:|---:|---:|
| Aug | 3 | 18 | 26 | 5 | 10 | 6 | 28 |
| Sep | 3 | 18 | 27 | 6 | 9 | 9 | 24 |
| Oct | 3 | 18 | 27 | 6 | 11 | 8 | 23 |
| Nov | 3 | 15 | 27 | 4 | 13 | 8 | 26 |

All 58 ledger context fields had 100% coverage in the joined contract. Recent
reliability features use only outcomes matured before each row; they are not
inference-time outcome features.

## 4. Latent correctness and MLP

The latent target is the six family memberships multiplied by the prior
anchor residual. Five continuous SVD factors are fit on the prior training
rows. The MLP predicts residual EV, soft demotion/promotion probabilities and
the five latent factors. Its correction authority is bounded and multiplied by
a causal support factor in [0.25, 1].

| Test month | Fit/cal/test rows | Features before/after | MLP iterations | Loss | Test residual MAE | Calibration slope | Mean support | Latent test IC range |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Aug | 62,529 / 15,633 / 79,448 | 167 / 96 | 100 | 0.246 | 243.9 bps | 0.000 | 0.468 | -0.009 to +0.036 |
| Sep | 126,305 / 31,577 / 76,912 | 167 / 96 | 100 | 0.252 | 180.3 bps | 0.000 | 0.440 | -0.038 to +0.027 |
| Oct | 187,712 / 46,928 / 78,716 | 167 / 96 | 96 | 0.243 | 181.0 bps | 0.000 | 0.493 | -0.043 to +0.068 |
| Nov | 250,553 / 62,639 / 73,898 | 167 / 96 | 100 | 0.289 | 323.7 bps | 0.046 | 0.509 | +0.004 to +0.042 |

The zero calibration slope in August--October means the residual output was
effectively collapsed by calibration; November has a small positive slope but
the test residual error worsens sharply. The latent factors have near-zero
transport rank IC. This is not evidence for a reliable residual predictor.

## 5. Conditional correction arms

Pooled global H12 results, with the frozen incumbent as the matched anchor:

| Arm | Top 1% net | Top 5% net | Top 10% net |
|---|---:|---:|---:|
| Frozen incumbent / anchor | -7.30 | **+8.89** | -37.63 |
| Bounded residual (50 down / 25 up) | -36.32 | -19.44 | -59.32 |
| Bounded residual (100 down / 50 up) | -36.32 | -19.44 | -59.32 |
| Demotion-only | -27.50 | +0.54 | -55.90 |
| Promotion-only | -20.93 | -12.73 | -29.82 |
| Residual + heads | -20.62 | -11.65 | -37.32 |
| Dynamic head weighting | **+18.72** | -31.54 | -43.96 |
| Dynamic + residual | -25.52 | -19.71 | -60.36 |

Dynamic head weighting is the only arm with a pooled top-1 improvement
(+26.02 bps versus the incumbent), but it loses 40.42 bps at top-5 and is not
stable by month. Top-5 net by month for the incumbent versus dynamic weighting
was:

| Month | Incumbent | Dynamic weighting |
|---|---:|---:|
| Jul | -51.55 | -51.55 |
| Aug | -171.07 | -168.12 |
| Sep | -58.25 | -61.86 |
| Oct | -81.07 | -84.19 |
| Nov | +11.00 | +37.01 |

The per-side top-5 diagnostic reinforces the conversion problem (these are
side-local tails, not the production global tail): incumbent long +17.25 bps
versus short -195.71 bps; residual+heads long +28.32 versus short -97.41.

## Decision

The six family paths should remain available as causal context and reliability
inputs. They should not be promoted as standalone specialists. The conditional
MLP should not replace or directly correct the incumbent yet: its only visible
benefit is a narrow, unstable top-1 consensus-weighting effect. The next useful
test is a side-local reliability gate trained specifically for the global
top-5 objective, with strict shrinkage and a positive worst-month requirement;
the current residual MLP fails that gate.

Artifacts:

- `data_perp/artifacts/anchored_conditional_correction_20260806_v1/conditional_metrics.parquet`
- `data_perp/artifacts/anchored_conditional_correction_20260806_v1/fold_diagnostics.json`
- `data_perp/artifacts/anchored_conditional_correction_20260806_v1/feature_contract.json`
- `data_perp/artifacts/anchored_conditional_correction_20260806_v1/correctness_test_report.json`
- `data_perp/artifacts/pair_condition_specialists_20260806_v12_recurrence/PAIR_CONDITION_SPECIALIST_REPORT.md`
