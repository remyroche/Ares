# Stage-C output audit — 2026-08-01

## Scope and canonical evidence

This audit applies the attached Stage-C specification: continuation-versus-exhaustion information research for `P(retain_h0_given_clear)`, using causal OHLCV-derived fields and admitting OI, funding, and regime-transition fields only when their point-in-time lineage is proven.

The canonical completed Stage-1 result is:

`data_perp/artifacts/stage_c_conditional_retention_ablation_20260731_v4/`

The Stage-C v3 report predates v4. Where they differ, v4's sealed manifest and the read-only readiness audit are authoritative.

## Target and population

- Target: `retain_h0_given_clear`.
- Support: exact H0 clear-first rows only; non-clear rows are null and were not used as negative examples.
- Label on support: `exact_h12_net_bps > 0`.
- Frozen Stage-0 panel: 272,686 rows; 252,702 compatible with all admitted groups; 103,681 clear-first support.
- v4 Stage-1 window: 252,677 compatible rows and 103,667 clear-first rows after excluding 25 rows (14 clear-first) at 2024-12-01.
- The source panel covers 2023-03-30 through 2024-11-30, 128 linear USD-perpetual symbols. Inverse-perpetual rows are excluded.
- Target audit independently verifies: 149,021 non-clear null labels; 64,590 positive and 39,091 negative support labels; every support label equals `exact_h12_net_bps > 0`; side and support-month fields match.

## Feature/source audit

Admitted and tested groups:

| Arm | Group | Fields | Disposition |
|---|---|---:|---|
| C0 | Frozen E15 control | 91-field union | Reused byte-stable control |
| C1 | Price continuation/exhaustion | 28 | Tested; diagnostic only |
| C2 | Volume/liquidity OHLCV proxies | 9 | Tested; diagnostic only |
| C3 | Volatility/state transition | 12 | Tested; diagnostic only |
| C6 | Cross-sectional confirmation | 10 | Tested; diagnostic only |
| C8 | Fixed transparent composites | 5 | Tested; diagnostic only |

All new admitted fields are trailing, vectorised OHLCV/cross-sectional transforms. Liquidity-like fields are explicitly named as OHLCV proxies; no factual L2, spread, depth, aggressor, or liquidation data is claimed. Source/feature lineage, coverage, exclusions, availability, units, side normalization, missingness, and live-parity status are persisted.

Blocked groups:

- **F4/OI:** 314 archived files / 5,600,580 rows have only nominal hourly `ts`/index labels, not source observation or availability clocks. The existing ingestion path can unbounded-forward-fill them.
- **F5/funding:** 313 files / 5,302,785 rows have the same availability problem, plus no proven funding-value kind or settlement timestamp.
- **F7/regime transitions:** no candidate-level strict OOF/prequential sidecar was available.

No OI, funding, future settlement, or unproven learned transition value entered the admitted panel. No adapter was invented.

## OOF/validation audit

- Side-local frozen v11 LightGBM classifier and fixed hyperparameters; no new HPO or target search.
- Development OOF: expanding monthly folds April–July 2024.
- Final OOS: frozen August–November 2024, descriptive only.
- Training rule: `decision_ts < fold_start - 12h` and `label_available_ts < fold_start`.
- Availability filtering, clipping, correlation reduction, gain selection, and feature freezing are training-fold-only; final OOS labels were not used for selection.
- Six tested arms use exactly the same candidate IDs in every fold: C0/C1/C2/C3/C6/C8. There are 53,736 unique evaluated IDs and 322,416 arm-prediction rows.
- 37 focused tests pass, including 29 Stage-C readiness/materialization/runner/lineage checks and the target-support invariants.
- v4's 17 output hashes and the readiness audit's 5 output hashes verify.

## Results

Development aggregate OOF:

| Arm | ROC-AUC | PR-AUC | Brier | Spearman with exact H12 net | Top-decile exact H12 net |
|---|---:|---:|---:|---:|---:|
| C0 | 0.5819 | 0.6815 | 0.23253 | 0.1026 | +26.2 bps |
| C1/F1 | 0.5833 | 0.6845 | 0.23226 | 0.1075 | +26.7 bps |
| C2/F2 | 0.5820 | 0.6802 | 0.23243 | 0.1015 | +22.6 bps |
| C3/F3 | 0.5853 | 0.6848 | 0.23216 | 0.1107 | +26.9 bps |
| C6/F6 | 0.5785 | 0.6719 | 0.23518 | 0.0796 | +4.9 bps |
| C8/F8 | 0.5795 | 0.6800 | 0.23288 | 0.0996 | +23.4 bps |

Final OOS is not selection evidence. Its aggregate top-decile exact-H12 net values were C0 +63.1, C1 +70.5, C2 +60.6, C3 +61.0, C6 +57.1, and C8 +66.0 bps. These are conditional retention-ranking diagnostics, not complete-candidate or deployable entry EV.

The strongest development diagnostic was F3: aggregate ROC delta +0.00240, bootstrap probability of improvement 0.865, Brier delta −0.00040, and positive-month fraction 0.75. It still failed the predeclared admission gate because one side was materially negative (non-material side fraction 0.875, required 1.0). F1 was smaller and also failed transport/stability. F2, F6, and F8 were weaker or adverse on key measures.

## Gated stages and disposition

No arm passed the full development-only Stage-1 gate. Therefore:

- no compact combination or leave-one-group-out test was fabricated;
- no frozen Stage-B hierarchy substitution was run;
- no entry threshold, global ranking, quota, sizing, action layer, or portfolio policy changed;
- no Stage-B economic result is available or implied.

The single supported terminal disposition is:

`CURRENT_OHLCV_OI_FUNDING_CONTRACT_INSUFFICIENT_FOR_ENTRY_RETENTION`

This means the currently admitted OHLCV information did not provide sufficiently stable conditional-retention evidence under the predeclared gate. It does not mean every feature is uninformative: F3 is a useful diagnostic lead, but not a promoted trading input.

## Remaining gaps and next admissible work

1. Acquire or reconstruct raw OI/funding records with provider/exchange/market/product identity, source-event and observed timestamps, `available_ts`, ingestion/revision hashes, units, funding semantics, and finite source-specific staleness; then use an as-of join and stale-row rejection.
2. Materialize candidate-level regime/transition predictions with strict prequential/OOF lineage before admitting F7.
3. Re-audit the inherited E15 control separately: its byte-stable reuse is required for comparison, but it is not evidence that every inherited field is newly admissible under this Stage-C source contract.
4. Reconcile the older v3 narrative report with v4 in future handovers; do not reopen Stage-B until a new Stage-1 group passes all gates.
