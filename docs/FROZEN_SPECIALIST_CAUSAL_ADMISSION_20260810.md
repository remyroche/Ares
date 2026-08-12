# Frozen ATR2 specialist/residual: causal 21-day admission audit

Date: 2026-08-10  
Stack: frozen seven-view ATR2 specialists → q4h×side ordinal residual LambdaRank  
Transport population: July–November 2024 (388,494 strict-OOS rows)

## Contract

The score is the frozen residual stack score; no model, feature, specialist,
query, or exit-policy change was made. The admission map is the canonical
pooled-parent / side-shrunk 21-calendar-day map:

- score-rank bins are built from prior resolved rows only;
- `label_available_ts = decision_ts + 13h`;
- a reference is usable only when `label_available_ts < snapshot decision_ts`;
- the common parent map is fitted before the side-local isotonic curve;
- side curves are shrunk toward the parent with 500-row shrinkage;
- admission threshold is mapped expected net >= +50 bps;
- final trading ranking remains pooled-global, never per timestamp.

The prediction-to-ledger join is one-to-one and the fold-qualified admission
identity is `fold::candidate_id`. The warm-up is fail-closed until a 21-day
reference window has accumulated.

Artifacts:

- Input: `data_perp/artifacts/frozen_specialist_admission_input_20260810_v1/`
- Admission output: `data_perp/artifacts/frozen_specialist_causal_21d_admission_20260810_v1/`
- Materializer: `scripts/materialize_frozen_specialist_admission_input.py`

## Pooled result

| Quantity | Result |
|---|---:|
| Strict-OOS rows | 388,494 |
| Rows with a mapped score | 385,810 (99.31%) |
| Rows admitted at mapped >= +50 bps | 4,561 (1.17%) |
| Realised net on all admitted rows | **−244.21 bps/trade** |
| Realised gross on all admitted rows | −144.21 bps/trade |

The raw frozen stack, before admission, is −7.30 / +8.89 / −37.63 net bps at
the global top 1% / 5% / 10% tails. Admission does not improve the stack:

| Selection | Top 1% | Top 5% | Top 10% |
|---|---:|---:|---:|
| Raw frozen score, all rows | −7.30 | **+8.89** | −37.63 |
| Admitted, rank by raw score (fraction of admitted) | +40.02 | −70.18 | −57.42 |
| Admitted, rank by mapped expected net (fraction of admitted) | **+25.47** | **−0.89** | **−8.16** |

The admitted-tail figures use 1%/5%/10% of the admitted population. If the
original full-population denominator is retained, the eligible set is too small
to fill the requested 1%–10% tails; selecting all eligible rows gives −244.21
bps/trade and is not a meaningful top-k comparison.

## Monthly transport behavior

| Month | Rows | Mapped | Admitted | Admission rate | Admitted net | Mapped mean |
|---|---:|---:|---:|---:|---:|---:|
| Jul-2024 | 79,520 | 76,836 | 4,022 | 5.06% | −263.85 | +81.15 |
| Aug-2024 | 79,448 | 79,448 | 0 | 0.00% | — | −99.79 |
| Sep-2024 | 76,912 | 76,912 | 0 | 0.00% | — | −103.61 |
| Oct-2024 | 78,716 | 78,716 | 0 | 0.00% | — | −100.93 |
| Nov-2024 | 73,898 | 73,898 | 539 | 0.73% | −97.62 | −97.21 |

The map’s semantics move sharply by month. July has a positive mapped tail but
negative realised admitted outcomes; August–October have no score mapped above
the +50-bps floor; November has a small positive mapped tail but negative
realised admitted outcomes. This is a regime/calibration transport failure, not
an ordinary threshold-tuning issue.

## Per-side behavior

| Side | Rows | Admitted | Admission rate | Admitted net | Mapped mean | Admitted top-5 mapped net |
|---|---:|---:|---:|---:|---:|---:|
| Long | 194,247 | 1,916 | 0.99% | −97.39 | +113.48 | +39.22 |
| Short | 194,247 | 2,645 | 1.36% | **−350.57** | +53.68 | **−359.80** |

Short-side admission is especially harmful: it admits more rows than long while
its realised net is −350.57 bps/trade. The global admission failure is therefore
not caused solely by the known short-side residual weakness; the long side also
fails in aggregate, although much less severely.

## Interpretation

This is a clean negative result for the current frozen stack plus the canonical
21-day admission layer:

1. The causal map is not merely too conservative. Its positive mapped regions do
   not transport to positive realised net.
2. The score-to-net relationship shifts across months. A map that looks positive
   in July still selects negative outcomes, and the map collapses below the floor
   through August–October.
3. The side-local shrinkage does not repair the short side. A common parent map
   plus side shrinkage is insufficient when the score semantics change by regime.
4. The result is consistent with the existing frozen-stack diagnosis: pooled
   top-5 profitability is long-driven and has a materially negative worst month.

## Decision

Do **not** promote this admission rule or use it as a production gate. Keep the
unadmitted frozen q4h residual control as the reference, with its known
execution-readiness failure, and treat causal admission as a diagnostic proving
that current score calibration is not portable.

The next repair should be a side- and regime-aware conversion model trained on
prior-resolved data, with explicit support/OOD and calibration-shift features,
and a fail-closed rule for unsupported regimes. It must be evaluated on the same
July–November rows with a positive worst-month and short-side gate; changing the
threshold alone is not justified by this audit.
