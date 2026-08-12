# Historical training-history compatibility audit — 2026-08-07

## Purpose

The 2025 residual walk-forward is chronologically live-like, but its residual
fit intentionally uses only prior 2025 months. This audit checks whether older
rows can be added without silently mixing incompatible base-score or feature
contracts.

Clarification: allowing a later fold to use matured outcomes from earlier 2025
months is not a validation leak; it is the defining behavior of an expanding
walk-forward retraining schedule. The v2 run therefore matches a policy that
re-fits the residual monthly at the first signal of each month, with a 13-hour
signal-to-label-maturity cutoff. What it does *not* yet match is a deployment
that seeds the January 2025 fit with pre-2025 residual outcomes, retrains at a
different cadence, or also re-fits the upstream base model and its value map.

## Compatible historical substrate found

The following two stores are row-aligned candidate populations:

- `data_perp/artifacts/full_universe_t2_t4_panel_20260801_v3/`
- `data_perp/artifacts/tp6_sl4_robust_clear_labels_20260802_v1/`

They contain 1,850,552 rows from 2023-04-01 through 2024-11-30, with 780
causal panel fields and exact TP6/SL4/H12 R3 labels. The label store has an
explicit `__label_available_at__` equal to 12 hours after the decision
timestamp (the decision timestamp is signal + 1 hour), so the effective signal
to label-maturity latency is 13 hours. It reports 1,703,668 valid rows and
146,884 invalid rows; invalid rows must remain excluded from supervised fitting.

The current repaired 120-field structural contract is represented in this
historical panel. This is sufficient substrate for a future all-history base
OOF regeneration and residual run.

## Incompatible score streams that must not be spliced in

The existing historical score artifacts do not provide the required current
base contract across this whole period:

| Artifact | Coverage | Score/feature contract | Use |
|---|---|---|---|
| `historical_2023_2024_r3_m6_rolling_20260809_v1` | 533,654 rows, Sep 2023–Feb 2024 | M6 meta probability, 16-field R5 context, older stack | Historical diagnostic only |
| `reconstructed_base_residual_stack_2022_2024_20260730_v4` | 360,012 rows, Jan 2022–Dec 2024 | frozen base backcast plus held-block residual/EV scores | Research diagnostic; not current-base OOF |
| current 2025 expanding arm | Feb–Dec 2025 held folds | repaired 120-field contract and frozen pre-2025 base outputs | Valid monthly adaptive 2025 diagnostic |

The older M6 and reconstructed scores may be useful for regime/economic
diagnostics, but they cannot be appended as if they were predictions from the
current base model. Doing so would make the residual target and score-to-EV map
non-stationary by construction.

## Required all-history implementation

Before claiming that older data are used in the production-equivalent arm:

1. Regenerate current-contract, side-local base OOF predictions on the
   historical panel, with the same R3 target, feature contract, entry convention,
   and 13-hour label-availability rule.
2. Join those base predictions one-to-one with the historical exact labels and
   current residual/context fields.
3. Train the residual monthly (or at the declared production cadence) using
   only labels with `label_available_at < held_start`, and choose explicitly
   between all-history expanding and rolling 12/24-month windows.
4. Keep 2025 as a later untouched evaluation segment for the all-history arm;
   do not use its outcomes to choose the history window or HPO.
5. Compare against the restricted prior-2025 arm on identical 2025 held rows,
   with pooled global top-k, monthly/worst-month, side, and clustered intervals.

Until step 1 is complete, the honest status is:

`HISTORICAL_LABEL_AND_FEATURE_SUBSTRATE_AVAILABLE_BUT_CURRENT_BASE_OOF_MISSING`

The current 2025 result must therefore remain labeled as a prior-2025-history
walk-forward diagnostic, not an all-history production simulation.

## 2026-08-07 — handoff-compatible all-history residual diagnostic

The first history-window comparison is now materialised at
`/Users/remyroche/Documents/Codex/artifacts/policy_correction_all_history_expanding_20260807_v2/`.
It uses the compatible structural 120-field long-side contract and the same
frozen 15-minute execution policy as the restricted arm. The run has 12 held
2025 folds (January--December), 1,463,365 held rows, and a 13-hour
signal-to-label-maturity cutoff. Training history contains the 15 available
structural months from July 2023 through November 2024, then adds each
matured earlier 2025 month.

This is explicitly a **handoff-compatible proxy**, not the final raw-base-OOF
arm: historical rows use `prequential_base_expected_net_bps`, and 2025 rows
use `base_expected_net_bps_proxy`. Older M6 and reconstructed diagnostic
scores are not included.

| Global 2025 tail | All-history expanding | Prior-2025-only expanding |
|---|---:|---:|
| Top 1% net bps/trade | +9.22 | +44.40 |
| Top 2% net bps/trade | +1.37 | +33.59 |
| Top 5% net bps/trade | −3.06 | +18.59 |
| Top 10% net bps/trade | −8.26 | +7.34 |

On the matched February--December rows, all-history is +14.11 / +7.00 /
−1.68 / −7.20 bps at top 1/2/5/10%, versus +44.40 / +33.59 / +18.59 /
 +7.34 for the restricted arm. Top-5 is positive in 4/12 all-history months
(4/11 common months), versus 8/11 for the restricted arm. The result is
diagnostic evidence that naively adding older history harms this residual
conversion contract; it is not evidence against chronological walk-forward.

The comparison tables and methodology are in
`/Users/remyroche/Documents/Codex/artifacts/policy_correction_history_comparison_20260807_v1/HISTORY_WINDOW_COMPARISON.md`.

### Rolling-window sensitivity

Explicit 12- and 24-compatible-month windows were also run from the same
cache. Their global 2025 top-5 net results are −0.20 and −1.70 bps/trade;
matched February--December results are −0.30 and +0.48 bps/trade. Top-5 is
positive in 6/12 and 5/12 months respectively. Neither rolling window
matches the restricted prior-2025-only arm (+18.59 bps/trade, positive in
8/11 months), so the current evidence favors recent-history adaptation over
blindly adding older rows. This remains a proxy comparison until raw current-
base OOF regeneration is complete.

The full ablation is at
`/Users/remyroche/Documents/Codex/artifacts/policy_correction_history_window_ablation_20260807_v1/HISTORY_WINDOW_ABLATION.md`.

### Current-contract R3 coverage audit (2026-08-07)

The current 2025 structural panel has 3,171,120 side-expanded candidate rows,
with 120 declared long fields and 120 declared short fields; the panel
manifest reports all 120 fields per side at or above the 90% coverage gate.
The exact H12 path pack is contract-correct (signal +1h decision, exact-minute
entry, `[entry, entry+12h)`, and label availability at the endpoint), but it
contains only 20,448 rows in 2025. That is at most 0.64% of the complete
side-expanded population before identity matching.

The complete `tp6_sl4_robust_clear_labels_20260802_v1` store contains
1,850,552 rows from 2023-04-01 through 2024-11-30 and has no 2025 rows. Thus,
the existing path pack can support a matched diagnostic, but it cannot support
a production-equivalent 2025 current-base OOF claim. The full 2025 candidate
path population must be materialised and converted to the R3
robust-clear/adverse/weak labels before that claim is made.

Audit artifact:
`/Users/remyroche/Documents/Codex/artifacts/current_r3_contract_coverage_audit_20260807_v1/CURRENT_R3_CONTRACT_COVERAGE_AUDIT.md`.

### Current-contract full-population relabel and coarse-source recovery (2026-08-07)

The full current 2025 side-expanded population has now been relabelled with
the exact one-minute materialiser at
`/Users/remyroche/Documents/Codex/artifacts/full_current_2025_r3_labels_20260807_v2/`.
All 24 month×side cells are present: 3,171,120 candidate rows, of which
1,407,030 (44.4%) have a valid exact-minute H12 path. The remaining
1,764,090 rows are retained with null economic/R3 targets and are not eligible
for supervised fitting. Their dominant reason is missing/non-contiguous
decision-time ATR/path source coverage, not an economic loss.

Because the requested source policy allows a coarse proxy when minute data is
unavailable, a separate 15-minute contract was materialised at
`/Users/remyroche/Documents/Codex/artifacts/full_current_2025_r3_proxy_labels_20260807_v1/`.
It recovers 2,847,884 valid rows (89.8%). The proxy keeps the same signal-close
`+1h` decision, TP6/SL4, H12, adverse same-bar precedence, 100-bps cost and
R3 cost+25-bps hurdle, but uses the 15-minute open and 48 coarse bars and is
marked `label_resolution=proxy_15m`. It must not be silently merged with the
exact-minute contract.

The proxy is empirically close to the exact contract on the 1,378,396 rows
where both are valid: first-touch event agreement is 99.77%, robust-clear
agreement is 99.97%, net-sign agreement is 99.94%, soft-R3 Spearman is 0.9993,
and the median gross difference is 0 bps. This supports a declared proxy
training arm, but does not turn the proxy into an exact execution evaluation.

Artifacts:

- exact coverage: `/Users/remyroche/Documents/Codex/artifacts/full_current_2025_r3_labels_20260807_v2/coverage.parquet`
- proxy coverage: `/Users/remyroche/Documents/Codex/artifacts/full_current_2025_r3_proxy_labels_20260807_v1/coverage.parquet`
- exact/proxy comparison: `/Users/remyroche/Documents/Codex/artifacts/exact_proxy_r3_comparison_20260807_v1/exact_proxy_comparison.parquet`
- coarse-source audit: `/Users/remyroche/Documents/Codex/artifacts/coarse_ohlcv_coverage_20260807_v1/coarse_source_symbol_coverage.parquet`

The remaining blocker is now narrower: regenerate current-base OOF predictions
against this label substrate, keeping exact and proxy rows explicitly
separate in the manifest and evaluation tables. A pooled 2025 result must not
claim exact-minute execution semantics for proxy-labelled rows.

**Interim resolution:** 15-minute OHLCV is approved as the coarse source for
development and target-repair work. It is a declared `proxy_15m` contract, not
an implicit substitute for exact-minute execution. New expanding folds may use
matured 15-minute labels, provided that the fold manifest records the proxy
resolution, label maturity, entry convention, and cost-once rule. Exact and
proxy rows must remain separately reported; a final execution-readiness claim
still requires a uniform production-resolution contract.

### Current-contract expanding R3 base OOF (2026-08-07)

The current-contract OOF is now complete at
`/Users/remyroche/Documents/Codex/artifacts/current_r3_base_oof_20260807_v1/`.
It uses the frozen 120-field long/short feature lists, fixed three-class R3
base parameters (220 trees, learning rate 0.035, depth 5, 24 leaves,
minimum child count 1% of the fold fit, feature fraction 0.85, L2 20), and no
2025 HPO or feature selection. Each monthly fold trains on all available
historical exact rows plus matured prior-2025 15m-proxy rows, enforcing
`label_available_ts < fold_start`.

The run contains 2,847,884 valid proxy-held rows across 12 monthly folds and
1,378,396 exact-minute-overlap rows. The raw base score is
`P(clear) - 0.5 P(adverse)`.

| Pooled 2025 tail | Gross bps/trade (proxy) | Net bps/trade (proxy) |
|---|---:|---:|
| Top 1% | +74.89 | −25.11 |
| Top 2% | +53.86 | −46.14 |
| Top 5% | +38.01 | −61.99 |
| Top 10% | +27.50 | −72.50 |

On the exact-minute-overlap subset, top-1 is +100.65 gross / +0.65 net and
top-5 is +49.75 gross / −50.25 net. These are overlap diagnostics, not a
complete exact-minute 2025 evaluation.

The target is learnable but not yet economically sufficient: pooled clear-class
AUC is 0.671, clear rank IC is 0.296, net rank IC is 0.026, and top-40% clear
recall is 50.05%. Net rank IC is positive in 11/12 months, but proxy top-5 net
is positive in only 2/12 months (mean −64.19 bps, worst −126.27 bps). Side
top-5 proxy net is −25.04 bps long and −93.97 bps short.

This is the expected target/conversion separation: the base learns the R3
robust-clear event, but a clear event is not equivalent to a cost-clearing
realised TP6/SL4 execution. Gross top tails remain positive while the fixed
100-bps cost floor makes net tails negative. The next repair should therefore
be cost-aware target/value conversion or a residual reliability layer, not a
claim that the base has no predictive information.

Artifacts:

- predictions: `/Users/remyroche/Documents/Codex/artifacts/current_r3_base_oof_20260807_v1/base_oof_predictions.parquet`
- tail metrics: `/Users/remyroche/Documents/Codex/artifacts/current_r3_base_oof_20260807_v1/base_metrics.parquet`
- learning/stability audit: `/Users/remyroche/Documents/Codex/artifacts/current_r3_base_oof_learning_20260807_v1/learning_metrics.parquet`
- fold lineage: `/Users/remyroche/Documents/Codex/artifacts/current_r3_base_oof_20260807_v1/fold_provenance.json`

### Current R3 residual conversion test (2026-08-07)

A matched expanding residual arm was run at
`/Users/remyroche/Documents/Codex/artifacts/current_r3_residual_oof_20260807_v1/`.
It uses the OOF same-side base probabilities, 60 config-owned meta/context
fields, a per-row net residual around the training-fold base value map,
ordinalized at −100/−25/+25/+100 bps, and native LambdaRank q4h×side queries.
The same maturity rule (`label_available_ts < fold_start`) and historical
exact/current proxy split were enforced.

| Pooled proxy tail | Raw base net | Base value-map net | Base + residual net |
|---|---:|---:|---:|
| Top 1% | −25.11 | −72.36 | −54.42 |
| Top 2% | −46.14 | −63.28 | −80.03 |
| Top 5% | −61.99 | −65.73 | −81.94 |
| Top 10% | −72.50 | −84.65 | −88.33 |

On exact overlap, the residual arm is −72.39 bps at top-1 and −74.61 bps at
top-5, versus raw-base +0.65 and −50.25 bps. Only 3/24 side-month cells are
positive at residual top-5. The residual layer therefore does not currently
add reliable conversion information; its failure is consistent with the
unstable base-score-to-net mapping and target/economic mismatch, not evidence
that the R3 classifier has no signal.

Residual artifacts:

- predictions: `/Users/remyroche/Documents/Codex/artifacts/current_r3_residual_oof_20260807_v1/residual_oof_predictions.parquet`
- metrics: `/Users/remyroche/Documents/Codex/artifacts/current_r3_residual_oof_20260807_v1/residual_metrics.parquet`
- lineage: `/Users/remyroche/Documents/Codex/artifacts/current_r3_residual_oof_20260807_v1/fold_provenance.json`
