# Independent audit — Stage-D compact action model v9

## Verdict

**PASS.** Canonical D3/D4 artifact:
`data_perp/artifacts/stage_d_compact_action_model_20260731_v9/`.

Independent same-code rerun:
`data_perp/artifacts/stage_d_compact_action_model_20260731_v10/`.

The prior v3 failures are remediated. This is a research pass for the conditional clear-event
continue-versus-exit action only. It is not an entry-system, portfolio, or deployment pass.

## Reproducibility and seals

- All 11 v9/v10 files are byte-identical, including Parquets, manifests and the manifest seal.
- Runner SHA-256: `2d0e4005f7bbbb5fc7316ec6dd6a08b7c24956a2087208838bf078483e511ba3`.
- Test SHA-256: `41100ccddc51995ee23fd7e5e320c9a42e8ed749e8b971f0ca33fd47ffec2638`.
- The current runner and test files match those seals.
- Every one of nine declared output hashes and seven input hashes matches. `manifest.sha256` matches
  `run_manifest.json`.
- Focused compact suite: 10 passed.
- Both LightGBM selection and prediction models use one thread with deterministic column-wise mode.
  Calibration parameters are canonicalized below economically meaningful precision.

## Development-only selection and compactness

- D2-v4 supplied A0+A1 as the initial development-approved group set.
- Full and leave-A1-out models use the same 24,267 April-July development OOF candidate IDs.
- The predeclared development-only rule drops a group only when policy net, MAE and IC all improve and
  calibration is preserved. A0-only satisfied every condition:

| Development OOF | Full A0+A1 | A0-only | A0 change |
|---|---:|---:|---:|
| Policy net bps/trade | 94.1678 | 94.3102 | +0.1424 |
| MAE bps | 135.0736 | 134.0906 | -0.9831 |
| Spearman IC | 0.7952 | 0.8025 | +0.0073 |

- The frozen compact group decision is therefore `DROPPED_A1_path_geometry_to_clear`; the final model
  contains A0 only. No final-OOS result participates in this decision.
- Margin selection is also development-only. Among 0/25/50 bps, 0 wins with +75.5634 bps/trade
  versus always continue, compared with +75.2601 and +74.3780.
- Final side-local models each contain exactly 32 selected features, satisfying the <=32 cap.
- All 16 stored development fold states have resolved labels strictly before the held-out start.
  Filtering, clipping, imputation, correlation reduction and feature selection are stored per training
  fold. Final features and side-local calibrators are frozen before August; calibration uses 11,625
  long and 12,642 short development OOF rows only.

## Leave-group-out and identical rows

The leave-group-out artifact now contains both development and final results for:

- the full D2-approved A0+A1 model; and
- leave-A1-out A0-only.

All four arm/split combinations carry matching candidate hashes and `identical_rows_to_full_compact =
True`: 24,267 development rows and 31,258 final rows. Final leave-out results are descriptive only.

## Lineage and policy invariants

- The compact runner binds to corrected Stage-D feature pack v5, not v3/v4.
- The lineage gate is derived from sealed feature/group/lineage files. All 61 requested A0 fields are
  admitted causal, point-in-time safe and live reproducible; no unadmitted field is included.
- A6/A7/A8 are excluded with the required `REJECTED_LINEAGE`, `REJECTED_LINEAGE`, and
  `REJECTED_OOF_LINEAGE` dispositions. Feature availability is no later than the action decision.
- Candidate IDs are unique; continue/exit cost arithmetic and delta arithmetic all pass. The research
  gate consumes this evidence-derived result rather than a hard-coded `True`.
- Independent replay checks show that each row's gross, cost, net and delta values exactly match the
  sealed counterfactual pack. Actions are exactly `predicted_delta_continue_bps > margin`; reordering
  candidates cannot affect them. There is no top-k rule.
- Only `EXIT_NOW` and `CONTINUE_FROZEN_POLICY` are selected. The runner does not invoke an entry model,
  policy optimiser, sizing, concurrency, exposure, allocation, quota, or portfolio constraint and does
  not change the frozen counterfactual economics.

## Final replay and gates

At the frozen 0-bps margin on 31,258 August-November final-OOS rows:

| Metric | Result |
|---|---:|
| Learned policy net | 104.8492 bps/trade |
| Uplift vs always continue | +80.1228 bps/trade |
| Uplift vs always exit | +98.6163 bps/trade |
| Continue rate | 54.02% |
| Long uplift vs continue | +90.4486 bps/trade |
| Short uplift vs continue | +68.4874 bps/trade |
| Latest month (November) uplift | +88.9712 bps/trade |
| Worst month (October) uplift | +68.8512 bps/trade |
| MAE | 130.2063 bps |
| Spearman IC | 0.8394 |

Calibration passes overall and for both sides: slopes are 1.1189 overall, 1.1448 long and 1.2182
short; intercepts are -9.83, +27.07 and -57.36 bps, within the declared [0.5, 1.5] and +/-75-bps
limits. Both action rates exceed 2%.

All 126 final symbols have explicit result slices; their rows sum exactly to the final population.
Absolute symbol-uplift concentration is 0.01583, below the 0.35 gate, with support above the minimum 10.

Every final gate independently recomputes to true: uplift versus both baselines, both sides, latest
period, calibration, action support, symbol breadth/concentration and causal lineage.

## Paired day bootstrap

Independent reconstruction resamples all 122 final UTC days with replacement and recomputes pooled
sum-bps divided by sampled row count for each draw. With 1,000 seed-20260731 draws it exactly reproduces:

- versus always continue: 95% CI [75.8450, 84.6709], probability positive 1.0;
- versus always exit: 95% CI [90.6638, 106.2792], probability positive 1.0.

## Terminal interpretation

The sole compact D3/D4 decision `CLEAR_EVENT_CONTINUE_EXIT_ACTION_RESEARCH_PASSES` is supported. The
passing compact mechanism is A0 action state; A1 is removed. This result establishes useful causal
information after a candidate has already reached its first exact H0 clear. It does not establish that
candidate entry is profitable or that the full strategy should be promoted.
