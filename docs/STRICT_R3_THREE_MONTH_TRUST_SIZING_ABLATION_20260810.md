# Strict-R3 three-month trust-sizing ablations — 2025 OOF and frozen 2026 confirmation

## Decision

The Local Distribution Forest Proxy (LDF) support-shrinkage arm (legacy artifact ID `N5_drf_support_l110_meanrisk`) is the best tail-EV sizing overlay: it improves 2025 development and 2026 confirmation at top 1%, 2%, and 5%. The Bayesian GAM arm `G3_bayes_gam_cmi_l110_meanrisk` is the best constrained-portfolio risk overlay: it lowers matched 2026 maximum drawdown from about 82.1% to 71.4% while slightly improving portfolio net EV/trade. Neither is production-approved because drawdown remains severe and the worst 2026 top-5 month remains negative.

The Bayesian arms are small positive sizing refinements. The GAM family overfits the 2025 sizing gain and loses it in 2026. NGBoost improves only modestly. The MLP arms fail the development selection score and do not advance.

## Fixed experiment contract

- Long side only; frozen strict-R3 final score and pooled-global candidate ranking.
- Trust models change relative position size only; they cannot rerank or admit candidates.
- Three-month train blocks and three-month held blocks; labels must be resolved before the held-block boundary.
- 2025 is development OOF model selection. The top three arm names are frozen before 2026 is opened.
- 'Frozen 2026 confirmation' is untouched for this trust-sizing funnel only. The upstream canonical strict-R3 stack predates this funnel and has separate 2026 research history.
- Canonical SimplePolicyOptimiser outcome, including 100 bps cost exactly once.
- Causal side-local hierarchical 21/42/84-day EV mapping; admission requires mapped net EV >= +50 bps.
- Portfolio: eight concurrent, two new entries per bar, one position per asset, 80% margin, 7x leverage.
- Raw Geometry/K9 cluster memberships are excluded because bundle meanings change. Only bundle-invariant entropy, top-two margin, OOD, drift, and support summaries are pooled.
- Implementation fidelity: NGBoost is the actual NGBoost categorical classifier; LDF is a declared random-forest predictive-distribution proxy with local-support/parent shrinkage, not a specialized external distributional-random-forest package; the MLP is optimized directly under a bounded Student-t negative-log-likelihood; Bayesian GAM uncertainty uses Bayesian-ridge and conditional-scale components.

## 2025 development selection — every arm

| pipeline | development_shard | arm | weighted_tail_score | mean_portability_top1_2_5 | worst_month_top1_2_5 | selection_score |
| --- | --- | --- | --- | --- | --- | --- |
| bayesian | bayesian | B1_raw_singleton_l100_mean | 294.14 | 124.45 | 60.33 | 325.25 |
| bayesian | bayesian | B5_stable_ranklossfp_l125_predictive | 293.94 | 124.38 | 60.29 | 325.04 |
| bayesian | bayesian | B4_stable_rankfp_l125_predictive | 293.93 | 124.42 | 60.31 | 325.04 |
| bayesian | bayesian | B3_stable_rankloss_l110_meanrisk | 293.82 | 124.37 | 60.26 | 324.91 |
| bayesian | bayesian | B2_stable_rank_l100_meanrisk | 293.59 | 124.42 | 60.18 | 324.69 |
| bayesian | bayesian | B0_equal_control | 291.10 | 122.65 | 59.42 | 321.76 |
| gam | gam | G5_bayes_dist_gam_cmi_l125_predictive | 299.44 | 124.84 | 61.25 | 330.65 |
| gam | gam | G3_bayes_gam_cmi_l110_meanrisk | 298.78 | 125.20 | 63.11 | 330.08 |
| gam | gam | G4_dist_gam_cmi_l110_meanrisk | 296.48 | 126.20 | 60.37 | 328.03 |
| gam | gam | G1_gam_singleton_l100_mean | 295.00 | 124.42 | 60.14 | 326.11 |
| gam | gam | G2_bayes_gam_singleton_l110_mean | 294.77 | 124.27 | 60.42 | 325.84 |
| gam | gam | G0_equal_control | 291.10 | 122.65 | 59.42 | 321.76 |
| nonlinear | distributional_forest | N5_drf_support_l110_meanrisk | 303.38 | 130.69 | 62.34 | 336.05 |
| nonlinear | distributional_forest | N6_drf_parent_l125_predictive | 303.08 | 129.60 | 62.07 | 335.48 |
| nonlinear | distributional_forest | N4_drf_raw_l125_mean | 297.39 | 125.45 | 60.83 | 328.75 |
| nonlinear | ngboost | N3_ngboost_shrunk_l125_predictive | 295.11 | 124.34 | 59.89 | 326.19 |
| nonlinear | ngboost | N1_ngboost_raw_l100_mean | 294.94 | 124.60 | 59.96 | 326.09 |
| nonlinear | ngboost | N2_ngboost_cal_l110_meanrisk | 294.88 | 124.41 | 59.67 | 325.98 |
| nonlinear | ngboost | N0_equal_control | 291.10 | 122.65 | 59.42 | 321.76 |
| nonlinear | distributional_mlp | N8_mlp_l110_predictive | 287.16 | 129.66 | 62.41 | 319.57 |
| nonlinear | distributional_mlp | N7_mlp_l100_meanrisk | 288.53 | 123.33 | 61.31 | 319.36 |
| nonlinear | distributional_mlp | N9_mlp_l125_predictive | 284.26 | 126.91 | 63.14 | 315.98 |

## Global tail metrics for each pipeline's control and top three

Values are exposure-weighted net bps/trade. Candidate membership is the frozen global ranking; therefore improvements are sizing improvements, not selection improvements.

### bayesian — 2025

| arm | top_0.5% | top_1% | top_2% | top_5% | top_10% |
| --- | --- | --- | --- | --- | --- |
| B0_equal_control | 213.55 | 179.05 | 164.92 | 120.26 | 55.32 |
| B1_raw_singleton_l100_mean | 214.19 | 179.99 | 166.60 | 123.16 | 62.15 |
| B4_stable_rankfp_l125_predictive | 214.09 | 179.91 | 166.52 | 122.99 | 61.67 |
| B5_stable_ranklossfp_l125_predictive | 214.11 | 179.92 | 166.53 | 122.99 | 61.64 |

### bayesian — 2026

| arm | top_0.5% | top_1% | top_2% | top_5% | top_10% |
| --- | --- | --- | --- | --- | --- |
| B0_equal_control | 189.33 | 160.25 | 112.14 | 51.71 | 5.66 |
| B1_raw_singleton_l100_mean | 191.37 | 163.03 | 115.69 | 55.82 | 9.34 |
| B4_stable_rankfp_l125_predictive | 191.48 | 163.13 | 115.76 | 55.75 | 9.18 |
| B5_stable_ranklossfp_l125_predictive | 191.46 | 163.10 | 115.72 | 55.69 | 9.13 |

### gam — 2025

| arm | top_0.5% | top_1% | top_2% | top_5% | top_10% |
| --- | --- | --- | --- | --- | --- |
| G0_equal_control | 213.55 | 179.05 | 164.92 | 120.26 | 55.32 |
| G3_bayes_gam_cmi_l110_meanrisk | 220.37 | 183.93 | 168.09 | 123.37 | 61.23 |
| G4_dist_gam_cmi_l110_meanrisk | 217.72 | 182.14 | 167.30 | 123.02 | 60.86 |
| G5_bayes_dist_gam_cmi_l125_predictive | 220.43 | 184.08 | 168.86 | 123.88 | 61.49 |

### gam — 2026

| arm | top_0.5% | top_1% | top_2% | top_5% | top_10% |
| --- | --- | --- | --- | --- | --- |
| G0_equal_control | 189.33 | 160.25 | 112.14 | 51.71 | 5.66 |
| G3_bayes_gam_cmi_l110_meanrisk | 187.64 | 159.06 | 112.20 | 52.33 | 6.27 |
| G4_dist_gam_cmi_l110_meanrisk | 187.79 | 159.28 | 112.36 | 52.47 | 6.49 |
| G5_bayes_dist_gam_cmi_l125_predictive | 187.95 | 159.17 | 112.09 | 52.17 | 6.19 |

### nonlinear — 2025

| arm | top_0.5% | top_1% | top_2% | top_5% | top_10% |
| --- | --- | --- | --- | --- | --- |
| N0_equal_control | 213.55 | 179.05 | 164.92 | 120.26 | 55.32 |
| N4_drf_raw_l125_mean | 215.57 | 181.63 | 168.42 | 125.31 | 64.83 |
| N5_drf_support_l110_meanrisk | 219.51 | 185.61 | 172.07 | 126.77 | 63.79 |
| N6_drf_parent_l125_predictive | 219.04 | 185.60 | 171.78 | 126.35 | 63.22 |

### nonlinear — 2026

| arm | top_0.5% | top_1% | top_2% | top_5% | top_10% |
| --- | --- | --- | --- | --- | --- |
| N0_equal_control | 189.33 | 160.25 | 112.14 | 51.71 | 5.66 |
| N4_drf_raw_l125_mean | 191.20 | 162.56 | 114.86 | 54.15 | 8.79 |
| N5_drf_support_l110_meanrisk | 194.21 | 165.40 | 116.86 | 55.40 | 8.57 |
| N6_drf_parent_l125_predictive | 193.00 | 164.76 | 116.46 | 55.14 | 8.28 |

## Monthly top-1/2/5 metrics

### bayesian — 2025 (development OOF)

| month | arm | top_1% | top_2% | top_5% |
| --- | --- | --- | --- | --- |
| 2025-01 | B0_equal_control | 321.76 | 264.21 | 191.38 |
| 2025-01 | B1_raw_singleton_l100_mean | 321.44 | 264.14 | 192.39 |
| 2025-01 | B4_stable_rankfp_l125_predictive | 321.43 | 264.18 | 192.55 |
| 2025-01 | B5_stable_ranklossfp_l125_predictive | 321.42 | 264.17 | 192.54 |
| 2025-02 | B0_equal_control | 174.66 | 192.22 | 176.80 |
| 2025-02 | B1_raw_singleton_l100_mean | 173.77 | 191.70 | 175.87 |
| 2025-02 | B4_stable_rankfp_l125_predictive | 173.73 | 191.61 | 175.21 |
| 2025-02 | B5_stable_ranklossfp_l125_predictive | 173.73 | 191.61 | 175.20 |
| 2025-03 | B0_equal_control | 246.79 | 144.66 | 68.25 |
| 2025-03 | B1_raw_singleton_l100_mean | 247.81 | 147.66 | 76.86 |
| 2025-03 | B4_stable_rankfp_l125_predictive | 247.80 | 147.77 | 76.47 |
| 2025-03 | B5_stable_ranklossfp_l125_predictive | 247.80 | 147.71 | 76.44 |
| 2025-04 | B0_equal_control | 283.18 | 239.18 | 175.90 |
| 2025-04 | B1_raw_singleton_l100_mean | 284.15 | 239.89 | 178.25 |
| 2025-04 | B4_stable_rankfp_l125_predictive | 283.20 | 239.16 | 177.75 |
| 2025-04 | B5_stable_ranklossfp_l125_predictive | 283.19 | 239.14 | 177.74 |
| 2025-05 | B0_equal_control | 133.21 | 88.99 | 59.42 |
| 2025-05 | B1_raw_singleton_l100_mean | 133.04 | 89.82 | 60.33 |
| 2025-05 | B4_stable_rankfp_l125_predictive | 132.49 | 89.52 | 60.31 |
| 2025-05 | B5_stable_ranklossfp_l125_predictive | 132.46 | 89.50 | 60.29 |
| 2025-06 | B0_equal_control | 189.48 | 141.94 | 81.48 |
| 2025-06 | B1_raw_singleton_l100_mean | 190.25 | 143.38 | 83.81 |
| 2025-06 | B4_stable_rankfp_l125_predictive | 190.04 | 143.19 | 83.57 |
| 2025-06 | B5_stable_ranklossfp_l125_predictive | 190.02 | 143.17 | 83.56 |
| 2025-07 | B0_equal_control | 169.13 | 134.61 | 112.03 |
| 2025-07 | B1_raw_singleton_l100_mean | 168.85 | 134.51 | 111.81 |
| 2025-07 | B4_stable_rankfp_l125_predictive | 168.94 | 134.60 | 111.98 |
| 2025-07 | B5_stable_ranklossfp_l125_predictive | 168.95 | 134.62 | 112.00 |

### bayesian — 2026 (frozen confirmation)

| month | arm | top_1% | top_2% | top_5% |
| --- | --- | --- | --- | --- |
| 2026-01 | B0_equal_control | 206.81 | 142.19 | 72.66 |
| 2026-01 | B1_raw_singleton_l100_mean | 207.09 | 142.52 | 74.19 |
| 2026-01 | B4_stable_rankfp_l125_predictive | 207.06 | 142.50 | 74.21 |
| 2026-01 | B5_stable_ranklossfp_l125_predictive | 207.04 | 142.51 | 74.22 |
| 2026-02 | B0_equal_control | 301.80 | 212.95 | 110.93 |
| 2026-02 | B1_raw_singleton_l100_mean | 301.26 | 215.77 | 117.80 |
| 2026-02 | B4_stable_rankfp_l125_predictive | 301.23 | 215.84 | 117.22 |
| 2026-02 | B5_stable_ranklossfp_l125_predictive | 301.21 | 215.84 | 117.18 |
| 2026-03 | B0_equal_control | 161.85 | 145.55 | 93.21 |
| 2026-03 | B1_raw_singleton_l100_mean | 161.86 | 145.67 | 94.59 |
| 2026-03 | B4_stable_rankfp_l125_predictive | 161.85 | 145.66 | 94.44 |
| 2026-03 | B5_stable_ranklossfp_l125_predictive | 161.85 | 145.67 | 94.46 |
| 2026-04 | B0_equal_control | 223.65 | 176.41 | 114.25 |
| 2026-04 | B1_raw_singleton_l100_mean | 223.46 | 176.65 | 116.02 |
| 2026-04 | B4_stable_rankfp_l125_predictive | 223.61 | 176.85 | 116.14 |
| 2026-04 | B5_stable_ranklossfp_l125_predictive | 223.61 | 176.85 | 116.13 |
| 2026-05 | B0_equal_control | 93.22 | 71.65 | 26.81 |
| 2026-05 | B1_raw_singleton_l100_mean | 95.52 | 75.13 | 31.14 |
| 2026-05 | B4_stable_rankfp_l125_predictive | 95.75 | 75.43 | 31.31 |
| 2026-05 | B5_stable_ranklossfp_l125_predictive | 95.76 | 75.43 | 31.27 |
| 2026-06 | B0_equal_control | 89.00 | 38.33 | -19.36 |
| 2026-06 | B1_raw_singleton_l100_mean | 92.09 | 40.31 | -18.37 |
| 2026-06 | B4_stable_rankfp_l125_predictive | 91.82 | 40.13 | -18.46 |
| 2026-06 | B5_stable_ranklossfp_l125_predictive | 91.75 | 40.08 | -18.49 |
| 2026-07 | B0_equal_control | 81.69 | 37.53 | 1.44 |
| 2026-07 | B1_raw_singleton_l100_mean | 81.07 | 37.83 | 2.02 |
| 2026-07 | B4_stable_rankfp_l125_predictive | 80.99 | 37.74 | 1.95 |
| 2026-07 | B5_stable_ranklossfp_l125_predictive | 80.99 | 37.75 | 1.95 |

### gam — 2025 (development OOF)

| month | arm | top_1% | top_2% | top_5% |
| --- | --- | --- | --- | --- |
| 2025-01 | G0_equal_control | 321.76 | 264.21 | 191.38 |
| 2025-01 | G3_bayes_gam_cmi_l110_meanrisk | 324.78 | 263.54 | 187.32 |
| 2025-01 | G4_dist_gam_cmi_l110_meanrisk | 320.25 | 261.49 | 188.18 |
| 2025-01 | G5_bayes_dist_gam_cmi_l125_predictive | 330.02 | 268.50 | 191.28 |
| 2025-02 | G0_equal_control | 174.66 | 192.22 | 176.80 |
| 2025-02 | G3_bayes_gam_cmi_l110_meanrisk | 184.90 | 197.53 | 179.73 |
| 2025-02 | G4_dist_gam_cmi_l110_meanrisk | 183.43 | 196.86 | 178.70 |
| 2025-02 | G5_bayes_dist_gam_cmi_l125_predictive | 179.53 | 195.52 | 179.07 |
| 2025-03 | G0_equal_control | 246.79 | 144.66 | 68.25 |
| 2025-03 | G3_bayes_gam_cmi_l110_meanrisk | 242.09 | 138.35 | 72.79 |
| 2025-03 | G4_dist_gam_cmi_l110_meanrisk | 243.64 | 140.78 | 74.58 |
| 2025-03 | G5_bayes_dist_gam_cmi_l125_predictive | 245.91 | 143.91 | 76.26 |
| 2025-04 | G0_equal_control | 283.18 | 239.18 | 175.90 |
| 2025-04 | G3_bayes_gam_cmi_l110_meanrisk | 283.10 | 237.48 | 174.87 |
| 2025-04 | G4_dist_gam_cmi_l110_meanrisk | 285.31 | 240.61 | 178.64 |
| 2025-04 | G5_bayes_dist_gam_cmi_l125_predictive | 284.47 | 238.76 | 175.98 |
| 2025-05 | G0_equal_control | 133.21 | 88.99 | 59.42 |
| 2025-05 | G3_bayes_gam_cmi_l110_meanrisk | 135.27 | 92.55 | 63.11 |
| 2025-05 | G4_dist_gam_cmi_l110_meanrisk | 133.36 | 90.07 | 60.37 |
| 2025-05 | G5_bayes_dist_gam_cmi_l125_predictive | 133.63 | 90.68 | 61.25 |
| 2025-06 | G0_equal_control | 189.48 | 141.94 | 81.48 |
| 2025-06 | G3_bayes_gam_cmi_l110_meanrisk | 192.47 | 143.37 | 82.62 |
| 2025-06 | G4_dist_gam_cmi_l110_meanrisk | 195.40 | 146.49 | 85.87 |
| 2025-06 | G5_bayes_dist_gam_cmi_l125_predictive | 194.09 | 144.15 | 83.07 |
| 2025-07 | G0_equal_control | 169.13 | 134.61 | 112.03 |
| 2025-07 | G3_bayes_gam_cmi_l110_meanrisk | 173.69 | 138.71 | 116.85 |
| 2025-07 | G4_dist_gam_cmi_l110_meanrisk | 166.08 | 132.09 | 111.66 |
| 2025-07 | G5_bayes_dist_gam_cmi_l125_predictive | 171.60 | 136.40 | 114.49 |

### gam — 2026 (frozen confirmation)

| month | arm | top_1% | top_2% | top_5% |
| --- | --- | --- | --- | --- |
| 2026-01 | G0_equal_control | 206.81 | 142.19 | 72.66 |
| 2026-01 | G3_bayes_gam_cmi_l110_meanrisk | 207.05 | 143.03 | 73.59 |
| 2026-01 | G4_dist_gam_cmi_l110_meanrisk | 206.26 | 142.42 | 73.26 |
| 2026-01 | G5_bayes_dist_gam_cmi_l125_predictive | 207.09 | 143.07 | 73.59 |
| 2026-02 | G0_equal_control | 301.80 | 212.95 | 110.93 |
| 2026-02 | G3_bayes_gam_cmi_l110_meanrisk | 301.34 | 212.80 | 110.99 |
| 2026-02 | G4_dist_gam_cmi_l110_meanrisk | 301.24 | 213.00 | 111.31 |
| 2026-02 | G5_bayes_dist_gam_cmi_l125_predictive | 301.51 | 212.86 | 110.97 |
| 2026-03 | G0_equal_control | 161.85 | 145.55 | 93.21 |
| 2026-03 | G3_bayes_gam_cmi_l110_meanrisk | 161.82 | 145.54 | 93.29 |
| 2026-03 | G4_dist_gam_cmi_l110_meanrisk | 161.81 | 145.53 | 93.27 |
| 2026-03 | G5_bayes_dist_gam_cmi_l125_predictive | 161.83 | 145.54 | 93.25 |
| 2026-04 | G0_equal_control | 223.65 | 176.41 | 114.25 |
| 2026-04 | G3_bayes_gam_cmi_l110_meanrisk | 223.65 | 176.22 | 114.78 |
| 2026-04 | G4_dist_gam_cmi_l110_meanrisk | 222.88 | 175.63 | 114.84 |
| 2026-04 | G5_bayes_dist_gam_cmi_l125_predictive | 223.80 | 176.14 | 114.60 |
| 2026-05 | G0_equal_control | 93.22 | 71.65 | 26.81 |
| 2026-05 | G3_bayes_gam_cmi_l110_meanrisk | 93.04 | 70.29 | 27.00 |
| 2026-05 | G4_dist_gam_cmi_l110_meanrisk | 94.59 | 71.79 | 28.01 |
| 2026-05 | G5_bayes_dist_gam_cmi_l125_predictive | 92.33 | 69.91 | 26.64 |
| 2026-06 | G0_equal_control | 89.00 | 38.33 | -19.36 |
| 2026-06 | G3_bayes_gam_cmi_l110_meanrisk | 89.00 | 38.33 | -19.36 |
| 2026-06 | G4_dist_gam_cmi_l110_meanrisk | 89.00 | 38.33 | -19.36 |
| 2026-06 | G5_bayes_dist_gam_cmi_l125_predictive | 89.00 | 38.33 | -19.36 |
| 2026-07 | G0_equal_control | 81.69 | 37.53 | 1.44 |
| 2026-07 | G3_bayes_gam_cmi_l110_meanrisk | 83.11 | 38.86 | 2.12 |
| 2026-07 | G4_dist_gam_cmi_l110_meanrisk | 82.65 | 38.37 | 1.91 |
| 2026-07 | G5_bayes_dist_gam_cmi_l125_predictive | 83.77 | 39.44 | 2.33 |

### nonlinear — 2025 (development OOF)

| month | arm | top_1% | top_2% | top_5% |
| --- | --- | --- | --- | --- |
| 2025-01 | N0_equal_control | 321.76 | 264.21 | 191.38 |
| 2025-01 | N4_drf_raw_l125_mean | 322.31 | 265.36 | 192.99 |
| 2025-01 | N5_drf_support_l110_meanrisk | 324.24 | 268.75 | 196.95 |
| 2025-01 | N6_drf_parent_l125_predictive | 323.72 | 268.39 | 196.74 |
| 2025-02 | N0_equal_control | 174.66 | 192.22 | 176.80 |
| 2025-02 | N4_drf_raw_l125_mean | 174.33 | 192.29 | 176.80 |
| 2025-02 | N5_drf_support_l110_meanrisk | 176.92 | 195.59 | 178.26 |
| 2025-02 | N6_drf_parent_l125_predictive | 178.77 | 196.41 | 178.96 |
| 2025-03 | N0_equal_control | 246.79 | 144.66 | 68.25 |
| 2025-03 | N4_drf_raw_l125_mean | 248.30 | 146.95 | 74.51 |
| 2025-03 | N5_drf_support_l110_meanrisk | 258.38 | 155.19 | 79.52 |
| 2025-03 | N6_drf_parent_l125_predictive | 256.80 | 153.77 | 78.93 |
| 2025-04 | N0_equal_control | 283.18 | 239.18 | 175.90 |
| 2025-04 | N4_drf_raw_l125_mean | 286.75 | 242.47 | 179.89 |
| 2025-04 | N5_drf_support_l110_meanrisk | 289.68 | 244.61 | 180.70 |
| 2025-04 | N6_drf_parent_l125_predictive | 289.29 | 244.05 | 179.93 |
| 2025-05 | N0_equal_control | 133.21 | 88.99 | 59.42 |
| 2025-05 | N4_drf_raw_l125_mean | 134.94 | 91.91 | 60.83 |
| 2025-05 | N5_drf_support_l110_meanrisk | 137.43 | 93.50 | 62.34 |
| 2025-05 | N6_drf_parent_l125_predictive | 137.15 | 92.86 | 62.07 |
| 2025-06 | N0_equal_control | 189.48 | 141.94 | 81.48 |
| 2025-06 | N4_drf_raw_l125_mean | 192.38 | 147.03 | 89.29 |
| 2025-06 | N5_drf_support_l110_meanrisk | 196.32 | 148.56 | 87.81 |
| 2025-06 | N6_drf_parent_l125_predictive | 195.91 | 148.22 | 87.45 |
| 2025-07 | N0_equal_control | 169.13 | 134.61 | 112.03 |
| 2025-07 | N4_drf_raw_l125_mean | 170.36 | 136.39 | 114.20 |
| 2025-07 | N5_drf_support_l110_meanrisk | 172.25 | 139.33 | 118.05 |
| 2025-07 | N6_drf_parent_l125_predictive | 171.94 | 138.87 | 117.56 |

### nonlinear — 2026 (frozen confirmation)

| month | arm | top_1% | top_2% | top_5% |
| --- | --- | --- | --- | --- |
| 2026-01 | N0_equal_control | 206.81 | 142.19 | 72.66 |
| 2026-01 | N4_drf_raw_l125_mean | 209.93 | 144.55 | 75.95 |
| 2026-01 | N5_drf_support_l110_meanrisk | 210.24 | 145.44 | 75.45 |
| 2026-01 | N6_drf_parent_l125_predictive | 210.03 | 145.12 | 75.02 |
| 2026-02 | N0_equal_control | 301.80 | 212.95 | 110.93 |
| 2026-02 | N4_drf_raw_l125_mean | 304.17 | 218.13 | 121.86 |
| 2026-02 | N5_drf_support_l110_meanrisk | 307.66 | 219.06 | 118.30 |
| 2026-02 | N6_drf_parent_l125_predictive | 304.27 | 217.06 | 117.36 |
| 2026-03 | N0_equal_control | 161.85 | 145.55 | 93.21 |
| 2026-03 | N4_drf_raw_l125_mean | 163.56 | 147.08 | 95.43 |
| 2026-03 | N5_drf_support_l110_meanrisk | 163.85 | 146.96 | 94.51 |
| 2026-03 | N6_drf_parent_l125_predictive | 164.53 | 147.73 | 95.02 |
| 2026-04 | N0_equal_control | 223.65 | 176.41 | 114.25 |
| 2026-04 | N4_drf_raw_l125_mean | 224.99 | 178.66 | 117.98 |
| 2026-04 | N5_drf_support_l110_meanrisk | 226.68 | 180.18 | 118.09 |
| 2026-04 | N6_drf_parent_l125_predictive | 226.88 | 180.27 | 118.05 |
| 2026-05 | N0_equal_control | 93.22 | 71.65 | 26.81 |
| 2026-05 | N4_drf_raw_l125_mean | 95.15 | 73.98 | 29.56 |
| 2026-05 | N5_drf_support_l110_meanrisk | 98.77 | 77.76 | 31.86 |
| 2026-05 | N6_drf_parent_l125_predictive | 97.49 | 77.40 | 31.57 |
| 2026-06 | N0_equal_control | 89.00 | 38.33 | -19.36 |
| 2026-06 | N4_drf_raw_l125_mean | 89.07 | 38.45 | -19.27 |
| 2026-06 | N5_drf_support_l110_meanrisk | 91.40 | 40.46 | -17.94 |
| 2026-06 | N6_drf_parent_l125_predictive | 91.02 | 40.29 | -17.89 |
| 2026-07 | N0_equal_control | 81.69 | 37.53 | 1.44 |
| 2026-07 | N4_drf_raw_l125_mean | 82.90 | 39.02 | 2.35 |
| 2026-07 | N5_drf_support_l110_meanrisk | 82.66 | 38.64 | 2.13 |
| 2026-07 | N6_drf_parent_l125_predictive | 82.65 | 38.55 | 2.05 |

## Stability by month

| pipeline | year | arm | tail | portability | month_median_bps | month_mad_bps | worst_month_bps | positive_months | months |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bayesian | 2025 | B0_equal_control | 0.01 | 161.35 | 189.48 | 56.27 | 133.21 | 7 | 7 |
| bayesian | 2025 | B0_equal_control | 0.02 | 120.88 | 144.66 | 47.56 | 88.99 | 7 | 7 |
| bayesian | 2025 | B0_equal_control | 0.05 | 85.72 | 112.03 | 52.61 | 59.42 | 7 | 7 |
| bayesian | 2025 | B1_raw_singleton_l100_mean | 0.01 | 161.64 | 190.25 | 57.21 | 133.04 | 7 | 7 |
| bayesian | 2025 | B1_raw_singleton_l100_mean | 0.02 | 125.64 | 147.66 | 44.04 | 89.82 | 7 | 7 |
| bayesian | 2025 | B1_raw_singleton_l100_mean | 0.05 | 86.07 | 111.81 | 51.48 | 60.33 | 7 | 7 |
| bayesian | 2025 | B4_stable_rankfp_l125_predictive | 0.01 | 161.26 | 190.04 | 57.54 | 132.49 | 7 | 7 |
| bayesian | 2025 | B4_stable_rankfp_l125_predictive | 0.02 | 125.86 | 147.77 | 43.84 | 89.52 | 7 | 7 |
| bayesian | 2025 | B4_stable_rankfp_l125_predictive | 0.05 | 86.14 | 111.98 | 51.67 | 60.31 | 7 | 7 |
| bayesian | 2025 | B5_stable_ranklossfp_l125_predictive | 0.01 | 161.24 | 190.02 | 57.56 | 132.46 | 7 | 7 |
| bayesian | 2025 | B5_stable_ranklossfp_l125_predictive | 0.02 | 125.77 | 147.71 | 43.89 | 89.50 | 7 | 7 |
| bayesian | 2025 | B5_stable_ranklossfp_l125_predictive | 0.05 | 86.15 | 112.00 | 51.71 | 60.29 | 7 | 7 |
| bayesian | 2026 | B0_equal_control | 0.01 | 127.54 | 161.85 | 68.64 | 81.69 | 7 | 7 |
| bayesian | 2026 | B0_equal_control | 0.02 | 106.92 | 142.19 | 70.54 | 37.53 | 7 | 7 |
| bayesian | 2026 | B0_equal_control | 0.05 | 32.51 | 72.66 | 41.59 | -19.36 | 6 | 7 |
| bayesian | 2026 | B1_raw_singleton_l100_mean | 0.01 | 128.69 | 161.86 | 66.34 | 81.07 | 7 | 7 |
| bayesian | 2026 | B1_raw_singleton_l100_mean | 0.02 | 108.83 | 142.52 | 67.39 | 37.83 | 7 | 7 |
| bayesian | 2026 | B1_raw_singleton_l100_mean | 0.05 | 34.30 | 74.19 | 43.05 | -18.37 | 6 | 7 |
| bayesian | 2026 | B4_stable_rankfp_l125_predictive | 0.01 | 128.80 | 161.85 | 66.10 | 80.99 | 7 | 7 |
| bayesian | 2026 | B4_stable_rankfp_l125_predictive | 0.02 | 108.97 | 142.50 | 67.07 | 37.74 | 7 | 7 |
| bayesian | 2026 | B4_stable_rankfp_l125_predictive | 0.05 | 34.30 | 74.21 | 42.89 | -18.46 | 6 | 7 |
| bayesian | 2026 | B5_stable_ranklossfp_l125_predictive | 0.01 | 128.80 | 161.85 | 66.09 | 80.99 | 7 | 7 |
| bayesian | 2026 | B5_stable_ranklossfp_l125_predictive | 0.02 | 108.97 | 142.51 | 67.09 | 37.75 | 7 | 7 |
| bayesian | 2026 | B5_stable_ranklossfp_l125_predictive | 0.05 | 34.26 | 74.22 | 42.94 | -18.49 | 6 | 7 |
| gam | 2025 | G0_equal_control | 0.01 | 161.35 | 189.48 | 56.27 | 133.21 | 7 | 7 |
| gam | 2025 | G0_equal_control | 0.02 | 120.88 | 144.66 | 47.56 | 88.99 | 7 | 7 |
| gam | 2025 | G0_equal_control | 0.05 | 85.72 | 112.03 | 52.61 | 59.42 | 7 | 7 |
| gam | 2025 | G3_bayes_gam_cmi_l110_meanrisk | 0.01 | 167.66 | 192.47 | 49.62 | 135.27 | 7 | 7 |
| gam | 2025 | G3_bayes_gam_cmi_l110_meanrisk | 0.02 | 117.96 | 143.37 | 50.83 | 92.55 | 7 | 7 |
| gam | 2025 | G3_bayes_gam_cmi_l110_meanrisk | 0.05 | 89.98 | 116.85 | 53.74 | 63.11 | 7 | 7 |
| gam | 2025 | G4_dist_gam_cmi_l110_meanrisk | 0.01 | 171.28 | 195.40 | 48.24 | 133.36 | 7 | 7 |
| gam | 2025 | G4_dist_gam_cmi_l110_meanrisk | 0.02 | 121.30 | 146.49 | 50.37 | 90.07 | 7 | 7 |
| gam | 2025 | G4_dist_gam_cmi_l110_meanrisk | 0.05 | 86.01 | 111.66 | 51.29 | 60.37 | 7 | 7 |
| gam | 2025 | G5_bayes_dist_gam_cmi_l125_predictive | 0.01 | 168.18 | 194.09 | 51.82 | 133.63 | 7 | 7 |
| gam | 2025 | G5_bayes_dist_gam_cmi_l125_predictive | 0.02 | 118.46 | 144.15 | 51.38 | 90.68 | 7 | 7 |
| gam | 2025 | G5_bayes_dist_gam_cmi_l125_predictive | 0.05 | 87.87 | 114.49 | 53.25 | 61.25 | 7 | 7 |
| gam | 2026 | G0_equal_control | 0.01 | 127.54 | 161.85 | 68.64 | 81.69 | 7 | 7 |
| gam | 2026 | G0_equal_control | 0.02 | 106.92 | 142.19 | 70.54 | 37.53 | 7 | 7 |
| gam | 2026 | G0_equal_control | 0.05 | 32.51 | 72.66 | 41.59 | -19.36 | 6 | 7 |
| gam | 2026 | G3_bayes_gam_cmi_l110_meanrisk | 0.01 | 127.43 | 161.82 | 68.78 | 83.11 | 7 | 7 |
| gam | 2026 | G3_bayes_gam_cmi_l110_meanrisk | 0.02 | 108.15 | 143.03 | 69.77 | 38.33 | 7 | 7 |
| gam | 2026 | G3_bayes_gam_cmi_l110_meanrisk | 0.05 | 33.63 | 73.59 | 41.19 | -19.36 | 6 | 7 |
| gam | 2026 | G4_dist_gam_cmi_l110_meanrisk | 0.01 | 128.20 | 161.81 | 67.23 | 82.65 | 7 | 7 |
| gam | 2026 | G4_dist_gam_cmi_l110_meanrisk | 0.02 | 107.13 | 142.42 | 70.57 | 38.33 | 7 | 7 |
| gam | 2026 | G4_dist_gam_cmi_l110_meanrisk | 0.05 | 33.10 | 73.26 | 41.58 | -19.36 | 6 | 7 |
| gam | 2026 | G5_bayes_dist_gam_cmi_l125_predictive | 0.01 | 127.08 | 161.83 | 69.50 | 83.77 | 7 | 7 |
| gam | 2026 | G5_bayes_dist_gam_cmi_l125_predictive | 0.02 | 108.18 | 143.07 | 69.79 | 38.33 | 7 | 7 |
| gam | 2026 | G5_bayes_dist_gam_cmi_l125_predictive | 0.05 | 33.72 | 73.59 | 41.01 | -19.36 | 6 | 7 |
| nonlinear | 2025 | N0_equal_control | 0.01 | 161.35 | 189.48 | 56.27 | 133.21 | 7 | 7 |
| nonlinear | 2025 | N0_equal_control | 0.02 | 120.88 | 144.66 | 47.56 | 88.99 | 7 | 7 |
| nonlinear | 2025 | N0_equal_control | 0.05 | 85.72 | 112.03 | 52.61 | 59.42 | 7 | 7 |
| nonlinear | 2025 | N4_drf_raw_l125_mean | 0.01 | 164.43 | 192.38 | 55.92 | 134.94 | 7 | 7 |
| nonlinear | 2025 | N4_drf_raw_l125_mean | 0.02 | 124.40 | 147.03 | 45.26 | 91.91 | 7 | 7 |
| nonlinear | 2025 | N4_drf_raw_l125_mean | 0.05 | 87.52 | 114.20 | 53.37 | 60.83 | 7 | 7 |
| nonlinear | 2025 | N5_drf_support_l110_meanrisk | 0.01 | 166.87 | 196.32 | 58.89 | 137.43 | 7 | 7 |
| nonlinear | 2025 | N5_drf_support_l110_meanrisk | 0.02 | 134.99 | 155.19 | 40.40 | 93.50 | 7 | 7 |
| nonlinear | 2025 | N5_drf_support_l110_meanrisk | 0.05 | 90.19 | 118.05 | 55.71 | 62.34 | 7 | 7 |
| nonlinear | 2025 | N6_drf_parent_l125_predictive | 0.01 | 166.53 | 195.91 | 58.76 | 137.15 | 7 | 7 |
| nonlinear | 2025 | N6_drf_parent_l125_predictive | 0.02 | 132.44 | 153.77 | 42.65 | 92.86 | 7 | 7 |
| nonlinear | 2025 | N6_drf_parent_l125_predictive | 0.05 | 89.81 | 117.56 | 55.49 | 62.07 | 7 | 7 |
| nonlinear | 2026 | N0_equal_control | 0.01 | 127.54 | 161.85 | 68.64 | 81.69 | 7 | 7 |
| nonlinear | 2026 | N0_equal_control | 0.02 | 106.92 | 142.19 | 70.54 | 37.53 | 7 | 7 |
| nonlinear | 2026 | N0_equal_control | 0.05 | 32.51 | 72.66 | 41.59 | -19.36 | 6 | 7 |
| nonlinear | 2026 | N4_drf_raw_l125_mean | 0.01 | 129.35 | 163.56 | 68.42 | 82.90 | 7 | 7 |
| nonlinear | 2026 | N4_drf_raw_l125_mean | 0.02 | 109.26 | 144.55 | 70.57 | 38.45 | 7 | 7 |
| nonlinear | 2026 | N4_drf_raw_l125_mean | 0.05 | 33.72 | 75.95 | 45.91 | -19.27 | 6 | 7 |
| nonlinear | 2026 | N5_drf_support_l110_meanrisk | 0.01 | 131.31 | 163.85 | 65.07 | 82.66 | 7 | 7 |
| nonlinear | 2026 | N5_drf_support_l110_meanrisk | 0.02 | 111.60 | 145.44 | 67.68 | 38.64 | 7 | 7 |
| nonlinear | 2026 | N5_drf_support_l110_meanrisk | 0.05 | 36.09 | 75.45 | 42.85 | -17.94 | 6 | 7 |
| nonlinear | 2026 | N6_drf_parent_l125_predictive | 0.01 | 131.01 | 164.53 | 67.05 | 82.65 | 7 | 7 |
| nonlinear | 2026 | N6_drf_parent_l125_predictive | 0.02 | 111.26 | 145.12 | 67.73 | 38.55 | 7 | 7 |
| nonlinear | 2026 | N6_drf_parent_l125_predictive | 0.05 | 35.62 | 75.02 | 43.03 | -17.89 | 6 | 7 |

## Stability by week

| pipeline | year | arm | tail | week_median_bps | worst_week_bps | best_week_bps | positive_weeks | weeks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bayesian | 2025 | B0_equal_control | 0.01 | 179.90 | 22.49 | 567.62 | 31 | 31 |
| bayesian | 2025 | B0_equal_control | 0.02 | 146.48 | -7.75 | 447.60 | 30 | 31 |
| bayesian | 2025 | B0_equal_control | 0.05 | 103.40 | -60.80 | 332.62 | 28 | 31 |
| bayesian | 2025 | B1_raw_singleton_l100_mean | 0.01 | 179.59 | 21.76 | 567.58 | 31 | 31 |
| bayesian | 2025 | B1_raw_singleton_l100_mean | 0.02 | 146.27 | -6.48 | 449.27 | 30 | 31 |
| bayesian | 2025 | B1_raw_singleton_l100_mean | 0.05 | 103.48 | -57.79 | 335.09 | 28 | 31 |
| bayesian | 2025 | B4_stable_rankfp_l125_predictive | 0.01 | 179.71 | 22.13 | 567.75 | 31 | 31 |
| bayesian | 2025 | B4_stable_rankfp_l125_predictive | 0.02 | 146.33 | -5.51 | 448.48 | 30 | 31 |
| bayesian | 2025 | B4_stable_rankfp_l125_predictive | 0.05 | 103.39 | -56.96 | 333.76 | 28 | 31 |
| bayesian | 2025 | B5_stable_ranklossfp_l125_predictive | 0.01 | 179.73 | 22.12 | 567.75 | 31 | 31 |
| bayesian | 2025 | B5_stable_ranklossfp_l125_predictive | 0.02 | 146.33 | -5.50 | 448.44 | 30 | 31 |
| bayesian | 2025 | B5_stable_ranklossfp_l125_predictive | 0.05 | 103.38 | -56.95 | 333.73 | 28 | 31 |
| bayesian | 2026 | B0_equal_control | 0.01 | 159.15 | 26.73 | 358.83 | 31 | 31 |
| bayesian | 2026 | B0_equal_control | 0.02 | 116.28 | -3.99 | 359.54 | 29 | 31 |
| bayesian | 2026 | B0_equal_control | 0.05 | 68.62 | -52.44 | 259.55 | 23 | 31 |
| bayesian | 2026 | B1_raw_singleton_l100_mean | 0.01 | 159.30 | 26.43 | 358.67 | 31 | 31 |
| bayesian | 2026 | B1_raw_singleton_l100_mean | 0.02 | 116.18 | -2.58 | 359.41 | 29 | 31 |
| bayesian | 2026 | B1_raw_singleton_l100_mean | 0.05 | 69.93 | -51.62 | 263.84 | 23 | 31 |
| bayesian | 2026 | B4_stable_rankfp_l125_predictive | 0.01 | 159.46 | 26.31 | 358.81 | 31 | 31 |
| bayesian | 2026 | B4_stable_rankfp_l125_predictive | 0.02 | 116.26 | -2.68 | 359.52 | 29 | 31 |
| bayesian | 2026 | B4_stable_rankfp_l125_predictive | 0.05 | 69.78 | -51.74 | 264.03 | 23 | 31 |
| bayesian | 2026 | B5_stable_ranklossfp_l125_predictive | 0.01 | 159.46 | 26.33 | 358.79 | 31 | 31 |
| bayesian | 2026 | B5_stable_ranklossfp_l125_predictive | 0.02 | 116.27 | -2.68 | 359.49 | 29 | 31 |
| bayesian | 2026 | B5_stable_ranklossfp_l125_predictive | 0.05 | 69.80 | -51.75 | 264.02 | 23 | 31 |
| gam | 2025 | G0_equal_control | 0.01 | 179.90 | 22.49 | 567.62 | 31 | 31 |
| gam | 2025 | G0_equal_control | 0.02 | 146.48 | -7.75 | 447.60 | 30 | 31 |
| gam | 2025 | G0_equal_control | 0.05 | 103.40 | -60.80 | 332.62 | 28 | 31 |
| gam | 2025 | G3_bayes_gam_cmi_l110_meanrisk | 0.01 | 184.90 | 18.91 | 566.27 | 31 | 31 |
| gam | 2025 | G3_bayes_gam_cmi_l110_meanrisk | 0.02 | 146.36 | -9.98 | 444.93 | 30 | 31 |
| gam | 2025 | G3_bayes_gam_cmi_l110_meanrisk | 0.05 | 105.90 | -57.14 | 337.14 | 28 | 31 |
| gam | 2025 | G4_dist_gam_cmi_l110_meanrisk | 0.01 | 178.77 | 23.49 | 566.52 | 31 | 31 |
| gam | 2025 | G4_dist_gam_cmi_l110_meanrisk | 0.02 | 144.37 | -4.08 | 449.68 | 30 | 31 |
| gam | 2025 | G4_dist_gam_cmi_l110_meanrisk | 0.05 | 106.37 | -53.65 | 335.02 | 28 | 31 |
| gam | 2025 | G5_bayes_dist_gam_cmi_l125_predictive | 0.01 | 183.38 | 21.90 | 566.82 | 31 | 31 |
| gam | 2025 | G5_bayes_dist_gam_cmi_l125_predictive | 0.02 | 140.73 | -8.28 | 446.62 | 30 | 31 |
| gam | 2025 | G5_bayes_dist_gam_cmi_l125_predictive | 0.05 | 105.75 | -56.45 | 333.10 | 28 | 31 |
| gam | 2026 | G0_equal_control | 0.01 | 159.15 | 26.73 | 358.83 | 31 | 31 |
| gam | 2026 | G0_equal_control | 0.02 | 116.28 | -3.99 | 359.54 | 29 | 31 |
| gam | 2026 | G0_equal_control | 0.05 | 68.62 | -52.44 | 259.55 | 23 | 31 |
| gam | 2026 | G3_bayes_gam_cmi_l110_meanrisk | 0.01 | 159.08 | 26.61 | 358.20 | 31 | 31 |
| gam | 2026 | G3_bayes_gam_cmi_l110_meanrisk | 0.02 | 116.29 | -3.32 | 359.16 | 29 | 31 |
| gam | 2026 | G3_bayes_gam_cmi_l110_meanrisk | 0.05 | 69.91 | -51.64 | 259.44 | 23 | 31 |
| gam | 2026 | G4_dist_gam_cmi_l110_meanrisk | 0.01 | 159.12 | 26.69 | 358.65 | 31 | 31 |
| gam | 2026 | G4_dist_gam_cmi_l110_meanrisk | 0.02 | 116.30 | -2.15 | 359.00 | 29 | 31 |
| gam | 2026 | G4_dist_gam_cmi_l110_meanrisk | 0.05 | 69.58 | -52.08 | 259.71 | 23 | 31 |
| gam | 2026 | G5_bayes_dist_gam_cmi_l125_predictive | 0.01 | 159.05 | 26.03 | 358.39 | 31 | 31 |
| gam | 2026 | G5_bayes_dist_gam_cmi_l125_predictive | 0.02 | 116.28 | -4.30 | 359.27 | 29 | 31 |
| gam | 2026 | G5_bayes_dist_gam_cmi_l125_predictive | 0.05 | 69.51 | -51.36 | 259.48 | 23 | 31 |
| nonlinear | 2025 | N0_equal_control | 0.01 | 179.90 | 22.49 | 567.62 | 31 | 31 |
| nonlinear | 2025 | N0_equal_control | 0.02 | 146.48 | -7.75 | 447.60 | 30 | 31 |
| nonlinear | 2025 | N0_equal_control | 0.05 | 103.40 | -60.80 | 332.62 | 28 | 31 |
| nonlinear | 2025 | N4_drf_raw_l125_mean | 0.01 | 180.59 | 22.74 | 567.71 | 31 | 31 |
| nonlinear | 2025 | N4_drf_raw_l125_mean | 0.02 | 150.18 | -4.97 | 450.82 | 30 | 31 |
| nonlinear | 2025 | N4_drf_raw_l125_mean | 0.05 | 103.23 | -54.88 | 336.11 | 28 | 31 |
| nonlinear | 2025 | N5_drf_support_l110_meanrisk | 0.01 | 183.01 | 26.45 | 572.49 | 31 | 31 |
| nonlinear | 2025 | N5_drf_support_l110_meanrisk | 0.02 | 155.14 | 0.45 | 453.26 | 31 | 31 |
| nonlinear | 2025 | N5_drf_support_l110_meanrisk | 0.05 | 105.23 | -52.81 | 338.06 | 28 | 31 |
| nonlinear | 2025 | N6_drf_parent_l125_predictive | 0.01 | 181.93 | 31.72 | 571.22 | 31 | 31 |
| nonlinear | 2025 | N6_drf_parent_l125_predictive | 0.02 | 153.55 | -0.50 | 454.64 | 30 | 31 |
| nonlinear | 2025 | N6_drf_parent_l125_predictive | 0.05 | 104.98 | -53.67 | 338.56 | 28 | 31 |
| nonlinear | 2026 | N0_equal_control | 0.01 | 159.15 | 26.73 | 358.83 | 31 | 31 |
| nonlinear | 2026 | N0_equal_control | 0.02 | 116.28 | -3.99 | 359.54 | 29 | 31 |
| nonlinear | 2026 | N0_equal_control | 0.05 | 68.62 | -52.44 | 259.55 | 23 | 31 |
| nonlinear | 2026 | N4_drf_raw_l125_mean | 0.01 | 160.48 | 26.96 | 361.87 | 31 | 31 |
| nonlinear | 2026 | N4_drf_raw_l125_mean | 0.02 | 118.83 | -2.63 | 361.45 | 30 | 31 |
| nonlinear | 2026 | N4_drf_raw_l125_mean | 0.05 | 71.63 | -50.66 | 267.56 | 23 | 31 |
| nonlinear | 2026 | N5_drf_support_l110_meanrisk | 0.01 | 161.40 | 26.94 | 368.29 | 31 | 31 |
| nonlinear | 2026 | N5_drf_support_l110_meanrisk | 0.02 | 118.87 | -0.76 | 366.22 | 29 | 31 |
| nonlinear | 2026 | N5_drf_support_l110_meanrisk | 0.05 | 72.23 | -51.20 | 270.31 | 23 | 31 |
| nonlinear | 2026 | N6_drf_parent_l125_predictive | 0.01 | 162.20 | 27.44 | 366.27 | 31 | 31 |
| nonlinear | 2026 | N6_drf_parent_l125_predictive | 0.02 | 119.26 | -1.91 | 361.96 | 29 | 31 |
| nonlinear | 2026 | N6_drf_parent_l125_predictive | 0.05 | 72.58 | -51.49 | 267.56 | 23 | 31 |

The exact per-week, per-arm, per-tail rows are retained in `top3_weekly.parquet`; the table above is their compact stability summary.

## 2026 causal admission plus portfolio constraints

| pipeline | arm | accepted_trades | trades_per_day | gross_bps_per_trade | net_bps_per_trade | positive_rate | max_drawdown |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bayesian | B0_equal_control | 2470 | 11.65 | 249.06 | 149.06 | 0.61 | -0.82 |
| bayesian | B1_raw_singleton_l100_mean | 2466 | 11.63 | 250.48 | 150.48 | 0.61 | -0.81 |
| bayesian | B5_stable_ranklossfp_l125_predictive | 2465 | 11.63 | 249.94 | 149.94 | 0.61 | -0.81 |
| bayesian | B4_stable_rankfp_l125_predictive | 2465 | 11.63 | 249.94 | 149.94 | 0.61 | -0.81 |
| gam | G0_equal_control | 2470 | 11.65 | 249.06 | 149.06 | 0.61 | -0.82 |
| gam | G5_bayes_dist_gam_cmi_l125_predictive | 2576 | 12.15 | 249.88 | 149.88 | 0.61 | -0.72 |
| gam | G3_bayes_gam_cmi_l110_meanrisk | 2576 | 12.15 | 250.25 | 150.25 | 0.61 | -0.71 |
| gam | G4_dist_gam_cmi_l110_meanrisk | 2565 | 12.10 | 250.47 | 150.47 | 0.61 | -0.72 |
| nonlinear | N0_equal_control | 2470 | 11.65 | 249.06 | 149.06 | 0.61 | -0.82 |
| nonlinear | N5_drf_support_l110_meanrisk | 2517 | 11.87 | 249.94 | 149.94 | 0.61 | -0.79 |
| nonlinear | N6_drf_parent_l125_predictive | 2512 | 11.85 | 244.47 | 144.47 | 0.61 | -0.79 |
| nonlinear | N4_drf_raw_l125_mean | 2479 | 11.69 | 249.15 | 149.15 | 0.62 | -0.81 |

Wallet endpoints are intentionally not promoted: compounding 80% margin at 7x over thousands of overlapping research trades produces numerically explosive wallet paths. Net bps/trade, trades/day, and drawdown are the interpretable portfolio diagnostics here.

## Interpretation

1. `N5_drf_support_l110_meanrisk` transports best for global tails: it raises top-1/2/5 weighted EV in both eras and improves the 2026 top-5 worst month slightly.
2. `G3_bayes_gam_cmi_l110_meanrisk` is the portfolio-risk challenger: versus equal sizing in 2026 it raises constrained net EV from about 149.1 to 150.3 bps/trade and improves maximum drawdown from about 82.1% to 71.4%, despite losing about 1.2 bps at the globally ranked top 1%.
3. These gains are still small relative to the underlying alpha and do not change candidate order. They are sizing overlays, not new alpha layers.
4. The 2026 top-5 tail still has one negative month and only 23/31 positive weeks. The overlays are reliable at top 1%, less so deeper in the tail.
5. Portfolio drawdown remains about 71-79% for the best challengers. This fails production risk acceptance despite positive per-trade EV.
6. No raw K9/archetype ID crosses bundle boundaries. A future cluster-conditioned ablation must either freeze one geometry bundle or train/evaluate only inside identical bundle hashes.

## Artifact map

The comparison bundle contains full global, monthly, weekly, stability, fold, CMI-edge, causal-admission portfolio, and manifest tables. The nonlinear NGBoost and MLP development shards are kept separately so negative results remain auditable.
