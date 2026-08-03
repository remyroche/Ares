# Stage-D final report — 2026-07-31

## Verdict and scope

`CLEAR_EVENT_CONTINUE_EXIT_ACTION_RESEARCH_PASSES`

Optional data-lineage disposition: `OI_FUNDING_CAUSAL_LINEAGE_UNRESOLVED`.

This is a research-only pass for the binary action taken after an exact clear event: `EXIT_NOW` versus `CONTINUE_FROZEN_POLICY`. It does not change or validate candidate entry, the frozen Stage-B hierarchy, entry thresholds, sizing, exposure, concurrency, or portfolio policy. The frozen decisions `STAGE_B_NO_EXECUTION_TARGET_ADVANCES` and `CURRENT_OHLCV_OI_FUNDING_CONTRACT_INSUFFICIENT_FOR_ENTRY_RETENTION` remain in force. Final-OOS action results are an assessment of the development-frozen rule, not a new selection surface and not evidence that the underlying entry system is deployable.

Canonical evidence is D0-v2, corrected action features v5, deterministic baselines v4, mechanism ablation v4, remediated compact action model v9, and OI/funding lineage v4. Compact v10 is the byte-identical same-code rerun of v9. Their declared input, code, output, and external manifest seals verify against current bytes.

## Answers to the thirteen required questions

1. **Is always exiting at first clear better than always continuing?** No. Across 108,139 actionable clear-first rows, always continue averages 27.746 net bps/trade and always exit 7.094 bps/trade. Signed `EXIT_NOW − CONTINUE` is −20.652 bps/trade; the paired 2,000-replicate UTC-day interval is [−25.224, −16.469].

2. **How large is giveback under the frozen policy?** Exiting is better on 40.764% of rows. The positive loss that mechanical exits could avoid sums to 8,306,201 bps, or 76.810 unconditional bps/trade (p90 258.791). This is not the signed policy effect: mechanical exit also sacrifices 10,539,465 bps of retained upside, 97.462 unconditional bps/trade (p90 277.807), which is why always exit loses overall.

3. **Does observed path-to-clear improve continuation prediction?** The broader A1 path group improves the cumulative D2 development diagnostics (MAE 178.555→173.867 bps, Spearman IC .615→.640, ROC-AUC .847→.868, Brier .1540→.1513), but it does **not** survive the stricter compact readmission on identical development rows. The final compact model is A0-only; no unsupported A1 claim is carried forward.

4. **Which mechanisms add information?** A1 passed the cumulative D2 screen but was dropped at compact readmission because A0-only performed better. Therefore no add-on group enters the final compact model. A2 failed prediction improvement; A4 had a negative policy increment; A5 failed prediction and policy value; A9 had a negative long-side effect. A3 was unavailable. A6/A7/A8 were correctly not run.

5. **Are improvements stable by side, month, symbol, and time-to-clear?** The A0-only compact final assessment is positive versus both baselines on long (+90.449/+109.223 bps) and short (+68.487/+86.664 bps), covers 126 symbols, and has maximum absolute symbol-uplift concentration .0158. Every final month is positive versus both baselines. Time-to-clear and symbol slices are reported without reranking and do not alter the fixed rule.

6. **Were OI and funding admitted?** No. A6 and A7 are `REJECTED_LINEAGE`; 13 source classes were audited with zero admissions. Missing availability timestamps, unbounded fill, product ambiguity, mixed units, and absent funding settlement semantics prevent causal use.

7. **Does the learned action policy improve net versus both deterministic baselines?** Yes for the frozen 0-bps margin. Development net is 94.310 bps/trade, +75.563 versus continue and +98.525 versus exit. Final-OOS net is 104.849 bps/trade, +80.123 versus continue and +98.616 versus exit. This is conditional action-layer economics on already-clear rows.

8. **How much loss is avoided by correct exits?** Final OOS avoids 81.659 unconditional bps/trade through correct exits and exits 97.649% of giveback cases.

9. **How much retained upside is sacrificed by false exits?** Final OOS false-exit opportunity cost is 1.536 unconditional bps/trade; 14.005% of retained cases are incorrectly exited.

10. **Does the latest period pass?** Yes. The development-selected 0-bps rule yields November 2024 uplift of +88.971 bps versus always continue and +121.843 bps versus always exit. No final month was used to choose the rule or margin.

11. **What is paired day-block uncertainty?** With 1,000 fixed UTC-day bootstrap replicates, final policy uplift has 95% intervals [75.845, 84.671] bps versus always continue and [90.664, 106.279] bps versus always exit; both positive probabilities are 1.0.

12. **What is the terminal decision?** Exactly one model decision: `CLEAR_EVENT_CONTINUE_EXIT_ACTION_RESEARCH_PASSES`. `OI_FUNDING_CAUSAL_LINEAGE_UNRESOLVED` is separately recorded as an optional data-lineage disposition, not a second model decision.

13. **What remains blocked because no entry model passed?** Entry-target promotion, a deployable entry score, candidate entry thresholds, Stage-B substitution, global entry ranking, sizing, stops, exposure, concurrency, and portfolio promotion all remain blocked. A Stage-D action pass cannot establish profitability or deployability of the upstream candidate population.

## Compact model and validation

The compact model contains A0 only after development-only re-admission dropped A1, with training-only clipping, correlation reduction and side-local feature selection capped at 32. The action margin is 0 bps, selected on 24,267 development-OOF rows and frozen before 31,258 final-OOS rows. Final diagnostics are MAE 130.206 bps, Spearman IC .839, ROC-AUC .961, PR-AUC .978, Brier .0668, log loss .2277, calibration slope 1.119 and intercept −9.831 bps. The continue/exit rates are 54.018%/45.982%. Final and development leave-group-out/symbol slices use identical candidate hashes and rows.

All 21 specification-named correctness tests pass exactly. They cover population identity, first-clear timing, causal execution, feature cutoffs, future-path exclusion, one-time costs, frozen continuation, exact delta arithmetic, arm identity, resolved-label folds, training-only preprocessing/selection, eligible cross sections, OI/funding rejection and bounded age, transition OOF lineage, incremental-bps mapping, development-only thresholding, and unchanged entry/portfolio policy.

## Protocol, gates, blocked stages, and limitations

Development-only selection uses four monthly OOF folds from 2024-04 through 2024-07; the untouched descriptive final-OOS period is 2024-08 through 2024-11. The model seed and both paired bootstrap seeds are 20260731; D1 uses 2,000 UTC-day replicates across 611 days and the final compact replay uses 1,000 replicates. Counts are 108,139 source clear-first rows, 24,267 development rows, 31,258 final rows, and 126 final symbols.

All eight frozen research gates pass: causal/lineage integrity, both paired baseline uplifts, side stability, latest-period uplift, calibration, action support, and symbol breadth/concentration. D3/A3, D6/A6, D7/A7, and D8/A8 remain blocked by source or lineage constraints. The evidence is candidate-conditioned, research-only, and final-OOS descriptive; it does not validate entry quality, remove upstream selection bias, authorize policy changes, or establish live trading performance.
