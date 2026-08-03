# Stage-E falsification and complete-candidate overlay

Date: 2026-07-31
Scope: frozen Stage-D v9 first-clear continue/exit overlay, with v10 used only as its byte-identical reproducibility companion. Stage B, Stage C, entry selection/ranking, sizing, portfolio constraints, stops, continuation logic, costs, and the zero-bps action margin were not changed.

## Terminal decision

`STAGE_D_PASS_REVOKED_TARGET_PROXY_OR_CAUSAL_DEFECT`

The Stage-D action result is not causally valid. A selected A0 feature called `known_row_cost_bps` is sourced from the future-resolved `label_execution_cost_return`. It is unavailable at the action decision and cannot be reconstructed from the entry row plus the completed price prefix. The pre-existing target-purity contract explicitly classified realised row cost and exit-time spread as outcome-bound and forbidden as model inputs.

This is decisive under the Stage-E rule that revokes Stage D when a selected feature uses future execution information, cannot be independently reconstructed at decision time, or makes cost/target arithmetic circular. No downstream sensitivity or overlay result can rescue that causal failure.

## E1 — causal sufficiency and target proximity

- All 61 A0 fields and every side/fold selection were inventoried.
- The completed-prefix audit reconstructs causal action state exactly on a fixed 1,000-row sample drawn from the 108,139-row population, without exposing the future suffix.
- `time_to_clear_minutes` and the clear-bar gross mark reconstruct exactly.
- `known_row_cost_bps` fails causal reconstruction and was selected by both side models.
- `estimated_net_if_exit_now_bps` is consequently not a live-reproducible net estimate because it subtracts that outcome-derived cost.
- The cost field has Spearman 0.89945 with `delta_continue_bps` and Pearson -0.32222 with future exit hour. Its mean varies by future exit reason: 96.13 bps for full-stop, 100.19 for timeout, and 101.41 for trailing exits.
- The cost cancels algebraically between the two target arms, but remains a powerful proxy for the future-resolved exit path. Algebraic cancellation therefore does not make it an admissible feature.
- Canonical v3 and companion v4 artifacts are byte-identical.

Evidence: `data_perp/artifacts/stage_e_a0_causal_sufficiency_20260731_v3/`.

## E2/E3 — minimal information, deletion, and conditional permutation

M0 reproduces the frozen Stage-D development replay exactly over 24,267 OOF rows: candidate rows, folds, raw scores, mapped scores, probabilities, and actions all match.

| Arm or intervention | Uplift vs always continue | M0 uplift retained |
|---|---:|---:|
| M0 full A0 | +75.56 bps | 100.00% |
| M1 action state only | +1.87 bps | 2.48% |
| M2 action state + cost + geometry | +75.51 bps | 99.93% |
| M3 entry-static only | +2.28 bps | 3.02% |
| M5 estimated-exit only | -0.98 bps | negative |
| M0 without cost family | +1.70 bps | 2.25% |

A Ridge M2 reaches +76.92 bps, MAE 13.74 bps, and IC 0.985. Conditional cost permutation destroys 68.22–71.52 bps of uplift, while action-state and entry-static interventions add little. This establishes that the inadmissible cost family, not a robust causal interaction among live fields, explains virtually the entire Stage-D result.

Canonical v1 and companion v2 are byte-identical. No second-OOS data was accessed.

Evidence: `data_perp/artifacts/stage_e_minimal_information_diagnostics_20260731_v1/` and `STAGE_E_E2_E3_AUDIT_20260731.md`.

## E4 — frozen execution sensitivity

The already-frozen v9 decisions were replayed without refitting, rescoring, or changing actions across 60 combinations of 0/1/2/5/10-minute latency, 0/10/25/50-bps incremental slippage, and -25/0/+25-bps exit-estimator stress. The unperturbed replay matches v9 exactly. On 31,157 common-support rows, even 10 minutes plus 50 bps retains a nominal +52.20-bps uplift versus always continue.

This is a mechanically correct sensitivity replay of a causally invalid model. It is retained as diagnostic evidence only and is not promotion evidence. Canonical v4 and companion v5 are byte-identical.

Evidence: `data_perp/artifacts/stage_e_execution_sensitivity_20260731_v4/`.

## E5 — second sealed chronological validation

Status: `NOT_RUN_FROZEN_MODEL_ARTIFACT_UNAVAILABLE`.

The earliest compatible later material was sealed as January–April 2025 before results were opened. The v9/v10 packs contain predictions, preprocessing state, selected features, and calibrators, but no serialized LightGBM booster/tree. Reconstructing the scorer would require prohibited retraining. The later context also contains only 7 of 27 required static selected features. Empty results and bootstrap tables record the blocked run; no model was refit and no second-OOS result was opened.

Evidence: `data_perp/artifacts/stage_e_second_oos_readiness_20260731_v1/`.

## E6 — complete-candidate frozen-policy overlay

P0 and P1 use the identical 132,248-candidate population from April through November 2024. Non-clear candidates remain on the frozen policy; only eligible first-clear actions are overlaid. No ranking, selection, sizing, portfolio, exposure, concurrency, or stop logic is introduced.

| Metric | P0 frozen policy | P1 action overlay | Change |
|---|---:|---:|---:|
| Gross EV per original candidate | -85.54 bps | -53.00 bps | +32.54 bps |
| Cost per original candidate | 99.57 bps | 99.57 bps | 0.00 bps |
| Net EV per original candidate | -185.11 bps | -152.57 bps | +32.54 bps |
| Win rate | 31.27% | 33.72% | +2.45 pp |
| Profit factor | 0.220 | 0.260 | +0.040 |

The paired UTC-day bootstrap interval for the nominal increment is [30.97, 34.10] bps with probability positive 1.0. Nominal increments are positive for long (+38.82), short (+26.25), and November (+39.39), but absolute P1 net EV remains negative for both sides and every month. Because P1 actions come from the causally defective model, this is only a historical arithmetic overlay—not evidence that the action layer adds live causal value.

Canonical v1 and companion v2 are byte-identical.

Evidence: `data_perp/artifacts/stage_e_full_candidate_overlay_20260731_v1/`.

## Reconciliation with the earlier positive headline

The earlier +90.45-bps long, +68.49-bps short, and +88.97-bps latest-month figures were conditional uplift versus always continue on already-clear rows. They were neither absolute entry EV nor complete-population EV. The complete-population replay confirms the upstream candidate policy is strongly negative. Stage E further shows that virtually all conditional uplift was learned from an unavailable outcome-derived cost proxy.

## What remains valid and what must change

Valid engineering evidence:

- frozen replay identity and deterministic reproduction;
- the exact full-candidate P0 economics;
- the mechanics of the latency/slippage and population overlay evaluators;
- the finding that the present action model is dominated by an inadmissible field.

Invalidated claims:

- Stage-D causal action value;
- the learned continue/exit model's promotion status;
- any inference that positive conditional uplift makes the entry stream profitable.

Before retraining an action model:

1. Replace `known_row_cost_bps` and all exit-time/outcome cost inputs with a genuinely decision-time cost estimate derived only from entry-known fees, contemporaneous spread/order-book state, and a predeclared causal fill model.
2. Rebuild `estimated_net_if_exit_now_bps` from the completed prefix and that causal cost estimate; prove independent live reconstruction before fitting.
3. Add an automated provenance gate that rejects label-, exit-, suffix-, and future-resolution-derived columns even if renamed.
4. Serialize the complete frozen model, preprocessing, feature order, calibrators, and schema so a true untouched chronological replay is possible without retraining.
5. Repeat E2/E3 first. Require material uplift after deleting every cost/geometry shortcut before spending another sealed OOS period.
6. Only then rerun execution sensitivity and the identical-population overlay. Absolute net EV, both sides, latest period, and global candidate economics must be reported separately from conditional uplift.

## Verification

All 16 specification-named correctness tests are present using valid Python identifiers, and the full Stage-E suite contains 28 passing tests. The specification's final test name contained spaces; it is implemented as `test_no_portfolio_or_sizing_logic_is_invoked` without changing its meaning. Required artifacts, hashes, blocked-stage disposition, limitations, and terminal decision are recorded in `correctness_test_report.json` and `run_manifest.json`.
