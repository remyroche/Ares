# Independent audit — Stage-D compact action model v3

## Verdict

**FAIL — supersede; do not treat v3 as the canonical D3/D4 completion artifact.**

Audited root: `data_perp/artifacts/stage_d_compact_action_model_20260731_v3/`.
The final action policy is economically positive against both deterministic baselines, but the compact
mechanism decision and the evidence package do not satisfy the full Stage-D specification.

## Verified PASS evidence

- All nine output hashes in `run_manifest.json` match their files. All five input hashes match current
  sealed inputs. `manifest.sha256` matches `run_manifest.json`. The sealed D2-v4 manifest hash matches.
- D2 group selection is development-only and admits A1 only. The compact model requests A0+A1.
- Development OOF contains 24,267 unique rows from April-July 2024. Final OOS contains 31,258 unique
  rows from August-November 2024. Every margin uses the same candidate IDs.
- Every stored chronological fold has `train_max_label_available_ts < heldout_start`. Stored
  preprocessing states select 32 features per side/fold or fewer. Final frozen models select exactly
  32 features per side. Clipping, median imputation, correlation reduction and feature selection are
  represented as training-derived state.
- Side-local calibration for the final replay uses development OOF only: 11,625 long rows and 12,642
  short rows. No final row is used for calibration or margin selection.
- The three predeclared margins are exactly 0/25/50 bps. Development selects 0 bps with +75.4210
  bps/trade versus always continue, ahead of 25 (+75.1285) and 50 (+74.3230).
- Replay actions exactly implement the absolute rule `predicted_delta_continue_bps > margin`; there is
  no top-k selection in the materialized policy.
- Final 0-bps replay produces +80.0963 bps/trade versus always continue and +98.5898 versus always
  exit. Long uplift versus continue is +90.4217; short is +68.4614; latest November is +88.8830.
  Continue rate is 53.78%. Overall and side calibration remain within the declared slope/intercept gate.
- The paired bootstrap is correct whole-UTC-day resampling. Independent reconstruction over 122 days,
  with 1,000 seed-20260731 draws and the pooled sum-bps/sum-rows estimand, exactly reproduces the
  intervals: [75.7788, 84.6562] versus continue and [90.6085, 106.2705] versus exit.
- Replay contains only `EXIT_NOW` and `CONTINUE_FROZEN_POLICY` outcomes and fixed counterfactual
  economics. It does not materialize an entry, sizing, concurrency, exposure or portfolio decision.

## Required FAIL findings

### 1. Symbol stability is absent

`stage_d_compact_model_results.parquet` has no `dimension = symbol` rows. Its dimensions are overall,
side, month, time-to-clear, volatility, latest period and worst month only. Consequently the compact
model cannot answer required final-report question 5 or establish symbol breadth/concentration and
symbol-stability evidence on development and final OOS.

### 2. v3 leave-A1-out is incomplete

`stage_d_leave_group_out_results.parquet` contains only development OOF. There is no final-OOS
leave-A1-out replay in v3, despite the D3 requirement to run leave-one-group-out tests on the compact
model. Candidate identity is correctly identical for the available development comparison.

### 3. The available leave-A1-out evidence rejects A1 in the compact architecture

On the identical 24,267 development rows, A0-only is better than compact A0+A1:

| Development OOF | A0+A1 compact | Leave A1 out (A0) | A0 change |
|---|---:|---:|---:|
| Net policy bps/trade | 94.1678 | 94.3102 | +0.1424 |
| Uplift vs continue | 75.4210 | 75.5634 | +0.1424 |
| MAE bps | 135.0736 | 134.0906 | -0.9831 |
| Spearman IC | 0.7952 | 0.8025 | +0.0073 |

The later v4 diagnostic, although not part of v3, confirms the same direction on final OOS: A0-only
net policy is 104.8492 versus 104.8227 for A0+A1 (+0.0265), with lower MAE (130.2063 versus 131.7381)
and higher IC (0.8394 versus 0.8298). Importing D2's A1 admission is not sufficient after compact
re-fitting; the compact leave-group-out result must drive a development-only re-admission/drop rule.

### 4. The causal/lineage gate is asserted, not derived

The runner hard-codes `no_causal_or_lineage_violation: True`. It is not computed from input seals,
feature availability checks, fold chronology, group dispositions and source-lineage evidence. A terminal
gate must be fail-closed and evidence-derived; an unconditional Boolean cannot support the claimed pass.

### 5. Same-code determinism is not proven for v3

The current runner SHA-256 does not match v3's sealed runner hash, and v3 does not archive a matching
runner source or a test-suite hash. v3 and v4 share byte-identical common data outputs, but their runner
hashes and schemas differ, so v4 is not an exact same-code deterministic rerun of v3. This is useful
semantic evidence, not the required reproducibility proof.

## Required remediation

Generate a fresh version that:

1. adds development and final symbol slices, symbol breadth/concentration and explicit stability evidence;
2. includes development and final identical-row leave-A1-out replay;
3. applies a predeclared development-only compact group-retention rule, which currently selects A0-only;
4. derives the causal/lineage gate fail-closed from sealed, machine-checkable evidence;
5. archives or hashes the exact runner and tests and produces a byte-identical same-code rerun;
6. preserves the 32-feature side cap, frozen development calibration, 0/25/50 development-only margin
   selection, absolute-bps action rule, fixed counterfactual policy and paired whole-day bootstrap.
