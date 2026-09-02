# Timestamp-top-30 score-correction ablation (long only)

## Question

The support/OOD/K9 reliability fields had not improved the incumbent bounded
LDF sizing overlay.  That does **not** establish that the fields carry no
conditional information: the accepted tail was concentrated at the 1.75x
size cap, leaving a sizing-only overlay little authority.  This ablation asks
whether the high-score candidates can instead be ranked more usefully.

## Strict contract

- Candidate universe is unchanged and target-free.
- A candidate enters the focused training domain only when its frozen upstream
  `final_score` is in the top 30% **within its own decision timestamp**.  The
  deterministic tie-break is `candidate_id`; no held-month percentile is used.
- At each monthly fold the LambdaRank model is fit on the preceding three
  months of rows with `policy_label_available_ts < fold_start`, capped at
  60,000 equal-month sampled rows.
- The learned target is a deliberately coarse policy-net grade:
  `<= -200`, `(-200,-50]`, `(-50,50)`, `[50,150)`, `>=150` bps.  It reduces
  supervision sensitivity to small path/exit noise.
- Inputs are the recovered 84-field, target-free meta-context contract under
  frozen Geometry/K9 bundle `5ed9e795…ab5c5`; raw K9 membership slots remain
  excluded.
- The ranker is a regularised depth-4 LambdaRank model (220 trees, 15 leaves,
  350 min-child rows, 0.03 learning rate, 0.75 feature fraction, 0.80 row
  fraction, L1=0.15, L2=12, truncation 3).
- Position sizing and its 0.25–1.75 bounds are unchanged.

The learned raw score is converted only to a **current-timestamp local rank**.
The final challenger is:

```text
corrected_score = final_score
                + 0.20 * (timestamp_base_rank - 0.5)
                + 0.05 * (focused_model_rank - timestamp_base_rank)
```

Both rank terms use only scores for candidates actionable at the same decision
timestamp.  The second term is deliberately residual: the learned model is
not allowed to discard the stronger timestamp-local base-rank correction.

## What the ablation established

### 2025 development global-tail diagnostic, April–July

Net bps/trade, scored-candidate global tails; policy outcomes are joined only
after scoring.

| Arm | Top 0.5% | Top 1% | Top 2% | Top 5% |
|---|---:|---:|---:|---:|
| Frozen `final_score` | +14.56 | +6.83 | -3.13 | -16.25 |
| Timestamp base-rank correction | +55.32 | +44.95 | +42.51 | +11.51 |
| Direct focused model correction | +39.01 | +29.42 | +26.43 | -0.24 |
| **Base-rank + focused residual** | **+53.82** | **+56.76** | **+41.66** | **+9.97** |

The timestamp-local normalisation is the dominant repair.  The focused model
adds Top-1/month portability but gives back 1.54 bps at Top-5 versus the
no-learning local-rank control.  It is therefore not evidence that all
support/OOD context is useless; rather, the model must be used conditionally
on local base rank.

### Frozen 2026 global-tail diagnostic, January–June

| Arm | Top 0.5% | Top 1% | Top 2% | Top 5% |
|---|---:|---:|---:|---:|
| Frozen `final_score` | -12.78 | -22.83 | -28.05 | -39.34 |
| Timestamp base-rank correction | +48.49 | +31.61 | +18.74 | -9.76 |
| **Base-rank + focused residual** | **+53.81** | **+33.36** | **+17.35** | **-13.62** |

The January–May subset (excluding thin June) is still +32.57 bps at Top-1
and +16.97 at Top-2 for the blended challenger.  It remains negative at
Top-5 (−13.67), so this is a high-confidence-tail improvement, not broad
candidate conversion.

## Causal 21-day EV-admission replay

The global-tail outputs above are not admission claims.  A separate replay
rebuilds `Causal21dAdmissionSpec(hierarchical_tail_side_shrinkage_v2)` on
`corrected_score`, using only the preceding 21 calendar days of fully resolved
policy labels, common-bps rank maps, mapped EV >= +50 bps, and fail-closed
support.  December 2025 was rescored with a strict September–November model
to supply January 2026's same-contract reference.

| Period | Arm | Admitted | Valid outcomes | Coverage | Net bps / valid admitted trade |
|---|---|---:|---:|---:|---:|
| 2025 May–Jul | Frozen score | 1,432 | 1,138 | 79.5% | -26.30 |
| 2025 May–Jul | **Corrected blend** | **3,788** | **2,903** | 76.6% | **+54.20** |
| 2026 Jan–Jun | Frozen score | 899 | 659 | 73.3% | +13.14 |
| 2026 Jan–Jun | **Corrected blend** | **2,386** | **1,523** | 63.8% | **+36.52** |

The blend has no admissions in February 2026 and only one in the thin June
surface.  It is not production-approved from these results alone.

## Coverage limitation

The lower coverage is a replay-data issue, not a feature available to the
score:

- the blend's 2026 Top-2 tail has 34.4% `incomplete_15m_path` rows;
- the frozen-score control has 27.8%; and
- both scores are constructed before `policy_path_valid`, outcome source, or
  future bars are consulted.

The unavailable rows require a complete downloaded 15-minute path for the
frozen next-hour-entry/H12 policy label.  They must never be converted to
economic failures or removed from inference candidates.  More complete
historical OHLC is required before treating the apparent EV uplift as a fully
covered execution result.

## Decision

Keep this as the **timestamp-local score-correction challenger**, not the
canonical stack yet.  It passes a meaningful strict-prequential score and
admission check, but promotion requires:

1. repair/replay the incomplete 15-minute label coverage on the selected
   candidate population;
2. rerun causal admission with portfolio/concurrency/exposure constraints;
3. verify the corrected score over a further frozen period; and
4. promote only if the positive result survives outcome coverage and portfolio
   accounting.

## Reproducible producers

- `scripts/run_strict_r3_timestamp_top30_reliability_rank_ablation.py`
- `scripts/run_strict_r3_timestamp_rank_control.py`
- `scripts/run_strict_r3_corrected_score_admission.py`

The source manifests explicitly label the earlier global-tail outputs as
diagnostics; they do not claim unchanged causal admission after a score change.
