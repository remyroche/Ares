# Same-Base In-Sample Geometry/K9 Ablation

## Decision

The deliberately riskier, same-base in-sample geometry/K9 challenger is
economically competitive with—and at Top-5% stronger than—the strict C3
rolling representation on the matched April--July 2026 long-only replay.
It is **not promoted to the canonical C3 contract**: its K9 representation is
derived from a labelled R3 reference model and the recent rows used to fit K9
remain in the downstream Severe/Correctness fit. It is useful evidence that
base-leaf geometry contains incremental conversion information, not yet
deployment evidence for a causal state representation.

## Matched contract

- Held periods: April--July 2026; 235,567 long candidates per arm.
- Upstream handoff: fixed monthly strict-R3 base plus ten consensus heads.
- Exit: first 15-minute open one hour after decision; SL 3 ATR; trailing
  activation 0.5 ATR; giveback 0.25 ATR; 12-hour timeout; 100 bps cost once.
- Parent and child: Severe-200 one-way alpha-0.5 demotion followed by
  `correct_100` one-way alpha-0.75 demotion; 4-hour LambdaRank queries.
- Evaluation: one pooled global ranking, not per-timestamp selection.

For each held month, a single 64-round R3 robust-clear reference model is fit
only on label-matured prior rows. That exact model provides both the eight
active-leaf support/OOD fields and the leaf paths used by K9. K9 is a
`MiniBatchKMeans(K=9)` over one-hot leaf paths from the latest 3, 6, or 9
available months. The K9 fit rows are deliberately retained in meta fitting;
this is the controlled in-sample element. Raw leaf IDs never enter the
downstream models.

## Pooled global policy-net results

All figures are net bps/trade after the 100-bps cost.

| Arm | Top 0.5% | Top 1% | Top 2% | Top 5% | Top 10% | Policy rank IC |
|---|---:|---:|---:|---:|---:|---:|
| Fixed upstream control | +84.08 | +48.32 | +26.72 | +3.19 | -18.44 | 0.1515 |
| Strict C3 rolling | **+149.13** | **+130.27** | **+101.20** | +48.88 | +15.35 | 0.1580 |
| Same-base K9, 3 months | +130.30 | +122.93 | +99.69 | **+53.74** | +17.09 | 0.1672 |
| Same-base K9, 6 months | +124.00 | +113.72 | +97.16 | +52.41 | +18.93 | 0.1662 |
| Same-base K9, 9 months | +136.21 | +115.95 | +96.10 | +53.55 | **+18.95** | **0.1675** |

The 3-month arm is the Top-5 winner: +4.86 bps/trade above C3 rolling and
+50.55 above the exact matched upstream control. Its 11,779 Top-5 trades
produce +632,954 net bps in aggregate over 101 traded days (116.62
trades/day). Its gross Top-5 EV is +153.74 bps/trade, so cost is reconciled as
one 100-bps deduction.

## Monthly Top-5 stability

| Arm | Apr | May | Jun | Jul | Mean | Median | Worst | MAD | Positive months |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Strict C3 rolling | +105.39 | +47.57 | +36.04 | +74.97 | +65.99 | +61.27 | +36.04 | 19.47 | 4/4 |
| Same-base K9, 3 months | +100.47 | +45.64 | +51.10 | +82.68 | +69.97 | +66.89 | +45.64 | 18.52 | 4/4 |
| Same-base K9, 6 months | +103.08 | +49.84 | +50.64 | +81.47 | **+71.26** | +66.05 | +49.84 | 15.81 | 4/4 |
| Same-base K9, 9 months | +96.85 | **+53.52** | +50.34 | +72.42 | +68.28 | +62.97 | **+50.34** | **11.04** | 4/4 |

The in-sample leaf geometry does not win the sharpest Top-0.5/1/2 tails:
strict C3 rolling remains best there. It does improve the operating Top-5 tail
and the all-row rank IC. The 9-month arm is the most stable Top-5 variant; the
3-month arm has the highest pooled Top-5 net EV.

## Interpretation and next gate

This result supports the hypothesis that the base model's leaf topology carries
conversion information that raw-market K9 only partly captures. It does not
establish that the benefit is portable, because three effects are coupled:

1. K9 is based on a label-trained R3 representation rather than raw market
   geometry.
2. K9 sees the same recent rows that downstream training later consumes.
3. The base-coupled meta fit may have a different usable historical-support
   span from strict C3.

Therefore retain **C3 rolling** as the canonical research representation and
retain the same-base 3-month arm as the economic challenger. The next decisive
test is a three-way support control: hold the current R3 reference fixed, fit
K9 on the most recent X months, then compare (a) all mature meta rows, (b) only
the K9-fit rows, and (c) only rows after a disjoint K9 burn-in. That separates
the value of in-sample K9, recent support, and a causally clean downstream
training boundary.

## Artifacts

- `data_perp/artifacts/basecoupled_geometry_k9_matched_20260810_v1/`
- `data_perp/artifacts/basecoupled_geometry_k9_matched_20260810_v1/metrics_global.parquet`
- `data_perp/artifacts/basecoupled_geometry_k9_matched_20260810_v1/metrics_monthly.parquet`
- `data_perp/artifacts/basecoupled_geometry_k9_matched_20260810_v1/fold_audit.parquet`
- `data_perp/artifacts/basecoupled_geometry_k9_matched_20260810_v1/geometry_bundle_audit.parquet`
