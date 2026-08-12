# R3 target-repair audit — interim 15-minute contract (2026-08-07)

## Decision

15-minute OHLCV is approved as the interim coarse source for target and ranking
diagnostics. It is explicitly labelled `proxy_15m`; it is not silently mixed
with the exact-minute execution contract. The repaired B25 target remains the
control. The +50 bps hurdle does not advance.

Terminal decision: `CURRENT_TARGET_VALID_BUT_PROXY_ECONOMICS_LEVEL_SHIFT`.

The target ordering is usable for development: the proxy's first-touch events
agree with the available exact 2025 labels on 99.989% of valid overlap rows
after normalising the legacy event-code convention, and robust-clear agreement
is 96.676%. Absolute execution economics still shift: exact/proxy gross means
are +3.75/-30.32 bps, respectively. Therefore proxy net bps are diagnostic,
not execution-readiness evidence.

## Frozen interim contract

- Signal timestamp: `decision_ts`.
- Entry: first 15-minute bar at `decision_ts + 1h`.
- Horizon: 48 contiguous 15-minute bars (12 hours).
- Geometry: TP `+6 ATR`, SL `-4 ATR`, ATR from the decision-time panel.
- Same-bar precedence: adverse/SL before favourable/TP.
- Cost: 100 bps, applied exactly once (`net = gross - 100`).
- R3 clear: pre-adverse MFE exceeds cost + 25 bps; timeout is distinct from
  adverse; invalid/incomplete paths are null and excluded from fitting.
- Features: existing side-local F0 feature lists; no path or outcome fields.
- OOF: 12 strict chronological folds with
  `label_available_ts < held-out fold start`, separately trained by side.

## Population and substrate

| Surface | Rows | Valid | Resolution |
|---|---:|---:|---|
| Historical training surface | 1,608,298 | 1,608,298 | exact minute |
| Current 2025 proxy source | 1,090,558 | 1,090,244 | 15m proxy |
| Combined supervised surface | 2,698,542 | 2,698,542 | mixed, explicit per row |

The 314 incomplete current proxy paths remain in the source coverage audit but
are not encoded as economic failures.

Corrected 2025 proxy first-touch composition:

| Side | Adverse | Timeout | Upper |
|---|---:|---:|---:|
| Long | 19.87% | 71.08% | 9.05% |
| Short | 18.15% | 71.89% | 9.96% |

The initial run incorrectly classified the equal horizon sentinels as adverse;
the materializer now guards `has_lower`/`has_upper`, and unit tests cover no
touch, first touch, and cost-once behavior.

## Strict OOF target comparison

Scores are `P(clear) - 0.5 P(adverse)`. Net uplift is the selected tail's mean
net minus the global pool mean; it is not the absolute mean net EV.

| Target | Pooled IC | Top-5 net uplift | Top-30 net uplift | Top-40 net uplift | Top-40 recall | Long top-5 | Short top-5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B25 (control) | 0.4576 | **−7.93 bps** | **−2.10 bps** | **−1.35 bps** | 48.85% | +10.49 bps | −26.34 bps |
| B50 | 0.4546 | −8.20 bps | −2.38 bps | −1.66 bps | 50.11% | +10.24 bps | −26.64 bps |

B50 slightly raises own-target top-40 recall, but loses rank IC and canonical
economic uplift. Its apparent recall gain is not stable enough to compensate:

- B25 top-30 recall range across 31 months: 15.53 percentage points; top-40
  range: 15.50 pp.
- B50 ranges: 15.61 pp and 15.52 pp.
- Both have positive IC in all 31 months and positive top-5 clear uplift in all
  31 months.
- B25 top-5 net uplift is positive in 15/31 months (mean −6.72 bps); B50 is
  positive in 15/31 (mean −6.66 bps).

The side split is the key unresolved issue: long is modestly positive, while
short remains materially negative. This is a side-local conversion/admission
problem, not evidence that the R3 opportunity target has no signal.

## Exact-overlap validation

On 1,090,244 valid 2025 overlap rows:

- First-touch agreement: 99.989% after mapping legacy exact codes
  (`TP=0, SL=1, timeout=2`) to the proxy convention
  (`adverse=0, timeout=1, upper=2`).
- Robust-clear event agreement: 96.676%.
- Exact gross mean: +3.75 bps; proxy gross mean: −30.32 bps.
- Exact net mean: −96.25 bps; proxy net mean: −130.32 bps.

The agreement supports using the proxy for target ordering, class prevalence,
and feature/recall diagnostics. The level shift prevents using proxy absolute
net EV to certify an execution policy or to compare against exact historical
policy economics.

## Substrate audit highlights

The proxy substrate is almost complete: the only current-year invalid rows are
157 long and 157 short rows in January, all caused by a missing entry bar. No
other month has an incomplete path or invalid ATR in the materialized source.

The fixed-cost geometry is not binding in the current proxy population: the
median `6*ATR - 100 bps` margin is positive in every month (roughly 362–666
bps), and the share with TP net margin ≤50 bps ranges from effectively zero to
about 8%. The dominant economic mass is therefore timeout, not a zero-margin
TP event. This supports retaining an explicit timeout class rather than
collapsing it into failure.

The oracle soft-target tails are strongly profitable before any causal model is
fit on the long side; short-side oracle top-5% net is positive in most but not
all months. The strict OOF model cannot convert that ordering into pooled
positive net uplift. This confirms the remaining bottleneck is feature/model
conversion and side-local calibration, not absence of target economics.

The generated descriptive files are under
`data_perp/artifacts/current_r3_target_substrate_proxy_15m_20260807_v1/`.
They are grouped by side and month; no future-derived regime is used as an
inference input.

## Causal side-local conversion diagnostic

I also applied a prior-resolved, side-local 20-bin monotone score-to-net map to
the strict OOF score (labels before each test month only). Rows without at least
100 prior-resolved labels are excluded; no future-label fallback is used. This
is a mapping diagnostic, not a retrained meta layer or an execution policy.

On 2025 proxy rows:

| Global tail | Raw score net | Side-local mapped net |
|---|---:|---:|
| Top 1% | −218.03 bps | **−57.89 bps** |
| Top 5% | −166.77 bps | −152.35 bps |
| Top 10% | −157.33 bps | −151.15 bps |
| Top 20% | −148.79 bps | −146.95 bps |

The map materially improves the top-1% selection and changes side composition,
but remains negative at every tail. This supports a conversion/admission repair
focused on side-local calibration and regime transport; it does not justify
claiming that the base target is execution-ready.

## Active scope: long-only causal 21-day admission

Shorts are excluded from the active workstream. The long-only admission audit
uses the existing strict-OOF score and a prior-resolved 21-calendar-day,
20-bin side-local map with pooled-parent shrinkage, 5% tail trimming, and a
50-bps expected-net admission floor. No future labels or short-side rows enter
this audit.

| Selection | Rows | Gross bps/trade | Net bps/trade | Months represented |
|---|---:|---:|---:|---:|
| Raw score global top 1% | 13,387 | +120.53 | +20.53 | 31 |
| Admitted long rows (thresholded mapped EV) | 11,125 | +121.09 | +21.09 | 8 |

The admitted set is only 0.831% of the long population, so it contains fewer
rows than the requested 1%, 5%, 10%, and 20% global quotas. Consequently,
the current comparison returns the same 11,125 admitted rows for each quota;
it is an absolute-threshold admission test, not a quota-preserving top-k
test. Admitted rows occur only in 2023-04, 2024-03/04/05/07/08, and
2025-01/02. Monthly admitted net is positive in 2023-04 (+36.4 bps),
2024-03 (+50.4), and 2024-05 (+56.6), but negative in 2024-04 (-146.7),
2024-07 (-48.0), 2024-08 (-32.9), and is only +5.1 bps in 2025-01.

The map therefore improves the long raw top-1% average by only +0.56 bps and
does not establish portable positive net EV. Its main effect is abstention:
when the prior 21-day long map does not support a 50-bps expected-net floor,
the model emits no admitted trade. The zero-admission months are a genuine
coverage consequence of the declared causal rule, not an imputation or a
short-side fallback. Artifacts:
`data_perp/artifacts/current_r3_21d_admission_20260807_v5/`.

The gate must not replace the live score as the ranking variable. In the
quota-preserving diagnostic, ranking the admitted rows by the mapped expected
EV produces a negative top-1% conditional tail (-183.2 bps net; one month of
support), whereas ranking admitted rows by the original OOF score gives:

| Conditional admitted tail | Rows | Net bps/trade | Months | Worst month |
|---|---:|---:|---:|---:|
| Top 1% | 112 | +425.9 | 5 | −627.8 |
| Top 5% | 557 | +209.6 | 6 | −363.7 |
| Top 10% | 1,113 | +108.3 | 6 | −192.2 |
| Top 20% | 2,225 | +63.0 | 6 | −89.8 |

These conditional tails are highly unstable and therefore not evidence of a
deployable edge. They do establish the correct policy semantics: use the
21-day mapped EV only as an admission gate, then retain the raw long score for
global ranking. The map-ranked tails are a diagnostic failure, not a reason to
replace the base score with the mapped value. The quota-preserving outputs are
in `admission_tail_metrics.parquet` and `admission_month_tail_metrics.parquet`.

## Files

- Corrected proxy labels:
  `data_perp/artifacts/current_2025_r3_proxy_labels_15m_20260807_v3/`
- Refreshed mixed training surface:
  `data_perp/artifacts/current_r3_repair_proxy_training_surface_20260807_v4/`
- B25 OOF:
  `data_perp/artifacts/current_r3_proxy_target_repair_b25_contrast05_20260807_v4/`
- B50 OOF:
  `data_perp/artifacts/current_r3_proxy_target_repair_b50_contrast05_20260807_v1/`
- Materializer: `scripts/materialize_15m_r3_proxy_labels.py`
- Surface refresh: `scripts/refresh_r3_proxy_training_surface.py`
- Correctness tests: `tests/test_materialize_15m_r3_proxy_labels.py`

## Next work

1. Keep B25 as the long-side base opportunity target for the interim stack.
2. Improve long-side conversion/admission coverage and portability; do not
   promote B50 on pooled recall alone.
3. Use existing exact substrate audits for cost/ATR, invalid-row, compression,
   and nearby-contract diagnostics; append 2025 proxy rows with an explicit
   resolution field rather than pooling levels.
4. Treat any 15m exit-grid result as a coarse sensitivity study. Final policy
   selection still requires the exact-minute path contract.
