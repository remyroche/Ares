# P8U F72 SHAP Top-10 Contributor Residual-State Trial — 2026-08-29

## Decision

Do **not** promote either SHAP challenger. The F72/Under-F120 canonical research contract is unchanged.

The trial verifies that current SHAP contributor geometry contains modest, causal information about future Base residuals. However, its downstream effect is either negative (entropy) or economically immaterial (top-10 state) after the actual dual-MC1 admission and constrained portfolio path.

This is offline long-only research only. No live, exchange, canonical, admission-production, or portfolio-production component was changed.

## Method

The source is the immutable August 2025–July 2026 strict-OOF F72 SHAP ledger. For each candidate it contains all 72 CatBoost SHAP contributions, and its receipts prove that the SHAP reconstruction matches the frozen F72 score.

For each candidate the producer:

1. selects its ten largest absolute SHAP contributors, retaining contributor identity, signed contribution, and current SHAP mass;
2. assigns the candidate to the same 20-bin Base-rank band used by the calibration research;
3. forms strict-prequential residual events from policy outcomes, which only become visible after `policy_label_available_ts`;
4. for each current top contributor/sign and Base band, queries 7-day and 21-day prior-resolved residual histories; and
5. aggregates those histories using the current candidate's top-ten SHAP mass.

The target-free output includes residual mean, absolute residual mean, residual standard deviation, +50-bps rate, effective support, support fraction, and signed residual alignment for 7d and 21d, plus a 21d top-1 variant. It also writes a compact timestamp-level table containing exactly ten aggregate F72 contributors for all 8,716 decision timestamps.

## Same-band residual diagnostic — January–July 2026

All figures are conditional on the candidate's Base-rank band. Positive IC means the causal state increases with later realised strict-prequential Base residual.

| State | Mean conditional residual IC | Positive months | Same-band upper–lower residual |
|---|---:|---:|---:|
| Top-10, 21d absolute-residual mean | +0.0278 | 6/7 | −15.46 bps |
| Top-10, 21d residual std | +0.0242 | 6/7 | −14.73 bps |
| Top-10, 21d signed alignment | +0.0177 | 6/7 | **+8.75 bps** |
| Top-1, 21d signed alignment | +0.0138 | 6/7 | **+9.51 bps** |
| Top-10, 7d +50-bps rate | +0.0186 | 5/7 | −5.63 bps |

The reliable directional candidate is therefore **signed alignment**, not raw support. Support is negatively related to later residuals in this period, and is not a safe promotion signal.

The most recurrent 2026 timestamp-level contributors were `mark_perp_dislocation` (top-1 in 5,063 hours), `mark_vs_perp_bps`, `mark_trigger_risk_5h`, `leverage_build`, `donchian_zone_1d_atr`, and `fund_rate`.

## Strict OOF Meta screen — January–July 2026

The frozen Under `rank_xendcg` objective is unchanged. Every held Meta score was persisted target-free before its held outcomes were opened.

| Arm | Added Meta inputs | SStableMeta | Residual IC | CMI given Base | Admission utility |
|---|---:|---:|---:|---:|---:|
| F120 control | — | -0.09056 | 0.11638 | 0.17876 | +5.10 bps |
| Entropy-only | `shap_f72_entropy` | **-0.06859** | 0.11698 | 0.17775 | +5.35 bps |
| Top-10 residual state | 17 state fields | -0.07877 | 0.11647 | **0.18026** | **+5.68 bps** |
| Top-10 state + entropy | 18 fields | -0.08713 | 0.11616 | 0.17842 | +5.51 bps |

Entropy improves the Meta-only stability measure; top-10 residual state improves conditional information and the admission-substitution utility. The combination is worse than either, so it was not sent downstream.

## Matched dual-MC1 constrained portfolio — April–July 2026

The downstream comparison uses the exact same F72 Base receipts, independent strict-prequential BCF and Current MC1 maps, both maps at the +50-bps gate, one shared chronological portfolio state, and the frozen normal auction constraints. April is the first evaluable month after the three-month MC1 warm-up.

| Arm | Accepted trades | Net EV/trade | Total net bps | Worst month | Worst week | Max DD |
|---|---:|---:|---:|---:|---:|---:|
| F120 control | 3,634 | **+119.21** | +433,201 | **+68.28** | +50.44 | -21.86% |
| Entropy-only | 3,565 | +118.26 | 421,590 | +66.22 | +43.40 | -22.45% |
| Top-10 residual state | 3,676 | +118.14 | **+434,272** | +67.82 | **+50.46** | -21.86% |

Top-10 state adds 42 entries and +1,071 bps total, but loses 1.07 bps/trade and does not materially improve downside. Entropy loses 69 entries, 0.95 bps/trade, 11.6k total bps, and worst-week performance. Neither clears the promotion bar.

## Causality and completeness

- Strict-OOF SHAP source re-used; parent score reconstruction receipt checked.
- Candidate-local top-ten contributor ranks are deterministic by absolute contribution and frozen F72 field order.
- Every timestamp has exactly ten persisted aggregate contributors.
- Residual events use a strict-prequential Base anchor and enter only after their declared label-availability timestamp.
- State at decision time excludes same-time and future outcomes.
- Target-free F120+state panels are persisted before Meta outcome metrics; dual-MC1 maps are separately strict-prequential.

## Artifacts

- Target-free top-10 residual-state panel:
  `data_perp/artifacts/strict_r3_p8u_f72_shap_top10_residual_state_aug25_jul26_20260829_v1`
- Same-band diagnostic and contributor-frequency audit:
  `data_perp/artifacts/strict_r3_p8u_f72_shap_top10_residual_state_audit_janjul26_20260829_v1`
- Objective screens:
  `data_perp/artifacts/strict_r3_p8u_f72_shap_top10_{control,entropy,top10,top10_entropy}_under_janjul26_20260829_v1`
- Downstream comparisons:
  `data_perp/artifacts/strict_r3_p8u_f72_shap_top10_{control,entropy,top10}_downstream_dual_mc1_janjul26_20260829_v1`
