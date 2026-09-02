# Strict-R3 top-tail reliability and K9 geometry ablation

**Date:** 2026-08-10  
**Side:** long only  
**Status:** completed matched development/confirmation funnel; top-30 curriculum plus K9 temperature 0.25 promoted to the schema-v4 canonical executable research contract; not production-approved  
**Outcome:** pre-2025 SimplePolicyOptimiser winner, including 100 bps cost exactly once

## 1. Question and causal contract

The experiment tested whether the current +100-bps policy-residual reliability
head should learn from every historical candidate or focus on candidates near
the upstream score tail that can actually be traded. The upstream D2
strict-R3 base and conditional-usefulness ten-head consensus were held fixed.

For every conversion cutoff:

- the retained fraction is computed once from the pooled-global **training**
  score distribution;
- that scalar cutoff is frozen and applied to reference/held rows;
- no held-window percentile or per-timestamp rank defines the training gate;
- only labels resolved before the cutoff are consumed;
- the reliability target is `policy_net_bps - base_anchor_bps > +100`;
- the reliability model remains the frozen 4-hour UTC LambdaRank head;
- candidates outside the fitted tail domain receive multiplier 1.0;
- candidates inside it receive `1 - 0.75 * (1 - correctness_rank)`;
- final normalization uses the same-model prior-42-day reference;
- evaluation uses one pooled global ranking, never a per-timestamp ranking.

Screen runs use 80,000 supervised rows and 30,000 geometry rows. Finalists use
the full 240,000/100,000 caps and are replayed through causal EV admission and
the portfolio auction.

## 2. Retained-fraction funnel

Net bps/trade are under the selected optimized policy.

| Year | Training fraction | Top 1% | Top 2% | Top 5% | Top-2 portability | Worst Top-2 month | Positive months |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2025 | 5% | +71.81 | +51.77 | +28.22 | -0.63 | -33.70 | 6/7 |
| 2025 | 10% | +145.37 | +91.43 | +41.64 | +78.89 | +4.16 | 7/7 |
| 2025 | 15% | +182.31 | +146.53 | +68.92 | +114.51 | +71.94 | 7/7 |
| 2025 | 20% | +177.43 | +158.46 | +85.73 | +141.87 | +98.46 | 7/7 |
| 2025 | 25% | +174.94 | +154.78 | +101.80 | +129.08 | +90.28 | 7/7 |
| 2025 | 30% | +172.63 | +154.62 | +110.95 | +129.33 | +94.29 | 7/7 |
| 2026 | 5% | +40.78 | +15.82 | -5.15 | -13.62 | -17.16 | 5/7 |
| 2026 | 10% | +110.45 | +48.75 | -2.40 | +21.47 | +0.55 | 7/7 |
| 2026 | 15% | +132.71 | +81.00 | +12.23 | +88.97 | +7.09 | 7/7 |
| 2026 | 20% | +148.48 | +104.14 | +37.12 | +109.31 | +17.98 | 7/7 |
| 2026 | 25% | +153.65 | +107.40 | +48.73 | +104.81 | +20.11 | 7/7 |
| 2026 | 30% | +141.01 | +101.59 | +47.97 | +118.15 | +22.31 | 7/7 |

The literal development-period stopping rule triggers: 30% does not beat the
20% gate's 2025 Top-2 portability (+129.33 versus +141.87). Therefore no 2026
35% or 40% arm was run, and any already-produced 2025 35/40 rows are excluded
from selection. Among the eligible 20/25/30 arms, however, 30% has the best
worst-year portability after confirmation: `min(2025, 2026) = +118.15`, versus
+109.31 for 20% and +104.81 for 25%. That is why 30%, not a wider gate, was
carried into the focused feature/K9 work.

The narrow gates fail for a simple reason: they give the reliability learner
too little support and make it extrapolate immediately outside the extreme
tip. The useful operating region begins around 15-20%; 30% best preserves the
cross-era floor while retaining much more Top-5 breadth.

## 3. Full-cap effect relative to the all-row canonical reliability head

| Year | Stack | Top 0.5% | Top 1% | Top 2% | Top 5% | Top 10% |
|---|---|---:|---:|---:|---:|---:|
| 2025 | Current all-row head | +177.56 | +147.35 | +123.77 | +88.02 | +52.06 |
| 2025 | Top-30 head | +207.44 | +172.28 | +162.60 | +119.15 | +56.11 |
| 2025 | Top-30 + K9 temp 0.25 | **+213.55** | **+179.05** | **+164.92** | **+120.26** | +55.32 |
| 2026 | Current all-row head | +168.65 | +133.73 | +93.72 | +42.11 | **+11.37** |
| 2026 | Top-30 head | +186.33 | **+160.90** | +110.60 | +51.42 | +6.47 |
| 2026 | Top-30 + K9 temp 0.25 | **+189.33** | +160.25 | **+112.14** | **+51.71** | +5.66 |

The top-30 curriculum is a real improvement at the traded tail, not merely a
screen-cap artifact. It sacrifices some 2026 Top-10 breadth, which is outside
the fitted reliability domain and should not be used to justify admission.

## 4. Reliability-input ablation

The additional inputs were constructed strictly as-of from prior-resolved
policy outcomes:

- cross-model state: base/consensus level, gap and absolute disagreement;
- recent state: 3/7/14-day support, residual mean, positive rate,
  approximately-correct rate and adverse-100/adverse-200 rates;
- covariance breaks: 7-day minus 28-day covariance/correlation of upstream
  score or disagreement with policy residual.

### 2025 screen

| Input block | Top 0.5% | Top 1% | Top 2% | Top 5% | Top-2 portability | Worst month |
|---|---:|---:|---:|---:|---:|---:|
| Existing state | +197.59 | +172.63 | +154.62 | +110.95 | +129.33 | +94.29 |
| + recent | +203.69 | +174.15 | +157.81 | +114.62 | +136.98 | +98.64 |
| + cross-model | +200.55 | +181.76 | **+157.89** | **+116.61** | +132.68 | **+99.10** |
| + covariance only | +201.14 | +174.14 | +154.08 | +112.25 | +128.95 | +94.68 |
| + all three | +202.15 | +176.75 | +157.82 | +115.80 | **+139.90** | +92.60 |

### 2026 confirmation of the 2025-selected combined block

| Input block | Top 0.5% | Top 1% | Top 2% | Top 5% | Top 10% | Top-2 portability | Worst month |
|---|---:|---:|---:|---:|---:|---:|---:|
| Existing state | +168.22 | +141.01 | **+101.59** | **+47.97** | **+10.78** | **+118.15** | +22.31 |
| + all three | **+179.98** | **+145.65** | +101.38 | +46.10 | +7.91 | +117.56 | **+27.35** |

The richer fields are informative at the extreme tip and improve the worst
month, but they do not improve the predeclared cross-era Top-2 portability
criterion. They remain available as risk diagnostics and are not promoted as
ordinary score inputs. Covariance breaks alone are not incremental.

## 5. Alternative downside-risk use

The same state was also tested as a shallow binary one-way demoter with targets
`policy residual <= -100 bps` and `<= -200 bps`, alphas 0.25/0.50/0.75. This
does not change the canonical TP6/SL4 Severe-200 head; it is a separate policy-
residual reliability overlay.

The 2025 winner was the -200-bps target at alpha 0.75:

| Year | Arm | Top 0.5% | Top 1% | Top 2% | Top 5% | Top 10% | Portability | Worst month |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2025 | Top-30 positive control | +197.59 | +172.63 | +154.62 | +110.95 | +49.60 | +129.33 | +94.29 |
| 2025 | -200 / alpha 0.75 | **+254.90** | **+240.71** | **+185.34** | +75.04 | +21.23 | **+190.03** | **+127.45** |
| 2026 | Top-30 positive control | +168.22 | +141.01 | +101.59 | +47.97 | +10.78 | **+118.15** | **+22.31** |
| 2026 | -200 / alpha 0.75 | **+216.24** | **+185.37** | **+133.43** | **+66.73** | **+20.86** | +110.53 | +13.43 |

The demoter substantially improves pooled economics in both eras, but the
2026 worst month and portability deteriorate. Under the selection rule it does
not replace the positive-residual head. It is a promising aggressive tail arm
for a later untouched-period or blend-weight test, not a canonical replacement.

## 6. K9 representation repair

The current K9 posterior was nearly uniform: median entropy was about 2.177
against `log(9) = 2.197`, with a median top-two margin near 0.0167. This made
the aggregate state weak even though raw memberships had already failed the
consensus ablation.

The repair keeps centers, input fields, cluster ordering, fit rows and the
training-derived base temperature fixed. It multiplies only that frozen
temperature by 0.75, 0.50 or 0.25. Smaller values sharpen the posterior. The
scale and effective temperature are hashed and persisted. No held row can
choose or rescale its own temperature.

2025 screen:

| Temperature scale | Top 0.5% | Top 1% | Top 2% | Top 5% | Top-2 portability | Worst month |
|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | +197.59 | +172.63 | +154.62 | +110.95 | +129.33 | +94.29 |
| 0.75 | +200.68 | +171.59 | +152.92 | +110.76 | +126.00 | +96.85 |
| 0.50 | +203.67 | **+174.18** | **+156.84** | **+111.83** | +130.93 | +93.54 |
| 0.25 | **+200.17** | +170.89 | +155.24 | +110.05 | **+132.65** | **+98.14** |

Scale 0.25 won by portability and was the only scale confirmed/full-capped.

## 7. Full-cap finalist and executable portfolio

| Year | Version | Top-2 EV | Portability | Worst month | Portfolio trades | Trades/day | Portfolio net EV | Max drawdown |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2025 | Top-30 control | +162.60 | **+154.01** | **+95.07** | 3,728 | 17.58 | +151.76 | -75.9% |
| 2025 | Top-30 + K9 temp 0.25 | **+164.92** | +151.28 | +92.07 | 3,713 | 17.51 | **+158.08** | **-72.9%** |
| 2026 | Top-30 control | +110.60 | +109.56 | **+34.23** | 2,764 | 13.04 | +147.09 | -66.9% |
| 2026 | Top-30 + K9 temp 0.25 | **+112.14** | **+112.32** | +33.20 | 2,782 | 13.12 | **+151.30** | **-59.5%** |

The 2025 portability reduction is small and is offset by better pooled tails,
portfolio EV and drawdown. The cross-era minimum portability improves from
+109.56 to +112.32. Scale 0.25 is therefore paired with the top-30 curriculum
in the schema-v4 canonical executable research contract. This promotion does
not constitute production approval; it still requires later untouched
validation and separate drawdown repair.

### Full-cap Top-2 month contributions

| Month | Top-30 control | Top-30 + K9 temp 0.25 |
|---|---:|---:|
| 2025-01 | +323.70 | +322.91 |
| 2025-02 | +188.75 | +188.67 |
| 2025-03 | +258.24 | +263.46 |
| 2025-04 | +243.82 | +241.03 |
| 2025-05 | +95.07 | +92.07 |
| 2025-06 | +150.98 | +146.97 |
| 2025-07 | +104.16 | +111.44 |
| 2026-01 | +139.81 | +142.08 |
| 2026-02 | +212.93 | +218.57 |
| 2026-03 | +153.27 | +159.77 |
| 2026-04 | +180.08 | +175.81 |
| 2026-05 | +34.23 | +33.20 |
| 2026-06 | +79.31 | +82.55 |
| 2026-07 | +55.43 | +65.75 |

Every month remains positive. May 2026 remains the binding regime and is not
repaired materially by temperature sharpening.

## 8. Decision

1. Promote the top-30 training curriculum over the former all-row reliability
   head in schema v4.
2. Freeze K9 temperature scale 0.25 in the same schema-v4 contract because it
   improves the cross-era portability floor and both portfolio replays.
3. Do not add the larger recent/covariance/cross-model feature block to the
   ordinary scorer yet.
4. Do not promote the -200/0.75 downside demoter despite its strong pooled
   economics; its 2026 worst-month transport is weaker.
5. Do not run 35% or 40% confirmation arms.
6. Keep schema v4 research-canonical but require a later frozen period before
   production approval.

## 9. Artifacts

- `data_perp/artifacts/strict_r3_highrank_correctness_screen_long_2025_janjul_20260810_v2`
- `data_perp/artifacts/strict_r3_highrank_correctness_screen_long_2026_janjul_20260810_v1`
- `data_perp/artifacts/strict_r3_highrank_correctness_extension_long_2025_janjul_20260810_v1`
- `data_perp/artifacts/strict_r3_highrank_correctness_extension_long_2026_janjul_20260810_v2`
- `data_perp/artifacts/strict_r3_highrank_correctness_top30_fullcap_long_2025_janjul_20260810_v1`
- `data_perp/artifacts/strict_r3_highrank_correctness_top30_fullcap_long_2026_janjul_20260810_v1`
- `data_perp/artifacts/strict_r3_top30_reliability_feature_screen_long_2025_janjul_20260810_v1`
- `data_perp/artifacts/strict_r3_top30_reliability_feature_confirm_long_2026_janjul_20260810_v1`
- `data_perp/artifacts/strict_r3_top30_downside_demoter_screen_long_2025_janjul_20260810_v1`
- `data_perp/artifacts/strict_r3_top30_downside_demoter_confirm_long_2026_janjul_20260810_v2`
- `data_perp/artifacts/strict_r3_top30_k9_temperature_screen_long_2025_janjul_20260810_v1`
- `data_perp/artifacts/strict_r3_top30_k9_temperature_confirm_long_2026_janjul_20260810_v1`
- `data_perp/artifacts/strict_r3_top30_k9_temperature_fullcap_long_2025_janjul_20260810_v1`
- `data_perp/artifacts/strict_r3_top30_k9_temperature_fullcap_long_2026_janjul_20260810_v1`

The incomplete first threshold experiment and the interrupted 2026 extension
v1 are preserved for lineage but excluded from all decisions.
