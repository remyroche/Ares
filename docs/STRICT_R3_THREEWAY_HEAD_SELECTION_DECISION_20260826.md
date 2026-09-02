# Strict-R3 three-head selection decision — 2026-08-26

## Decision

Retain the existing frozen B0, E, and T upstream heads for the active
Strict-R3 research/live contract.  Do **not** promote the 2026-08-26 HPO
three-head blend to consensus, MC1, or inference.

The challenger has a real timestamp-local ranking improvement, but it fails
the required downstream test on the same routed candidate identities: after
the unchanged residual/consensus, dual-MC1 admission, and chronological
portfolio auction it has lower net EV per trade, lower total net EV, weaker
worst-period results, and deeper drawdown at both predeclared thresholds.

This document is the canonical decision record for this feature-selection and
head-HPO cycle.  It supersedes the decision status in
[ROUTED_BASE_HEAD_REPLACEMENT_RESEARCH_20260826.md](ROUTED_BASE_HEAD_REPLACEMENT_RESEARCH_20260826.md),
which remains the detailed upstream-research receipt.

## Common validation contract

- Long only; strict chronological OOF scoring.
- Frozen timestamp-local router: top 50% for upstream research.
- Training rows must have labels resolved before the 28-day reserve preceding
  a held month.  Invalid policy rows are excluded, never encoded as losses.
- HPO uses query-safe subsampling, 30-round early stopping and
  `MedianPruner(startup=8, warmup_fold=1)`.
- Upstream diagnostic: select Top-1/2/5/10 **inside each timestamp**, compute
  realised canonical rich-policy net bps, then equal-weight timestamps.
- Advancement test: strict-prequential residual heads, dual BCF/current MC1
  maps, dual admission gate, one global chronological portfolio, and identical
  rich-policy labels/costs/constraints.

The HPO/grid period is development evidence.  It is not an untouched
promotion period.

## Head contracts and HPO results

| Head | Role and target | Objective | Feature contract | HPO result | Decision |
|---|---|---|---:|---|---|
| B | `policy_ordinal_base_grade` from canonical policy net: `<=0`, `0–50`, `50–100`, `100–200`, `200–400`, `>400` bps | LambdaRank, direction `+1` | 72 selected causal fields | improves standalone Top-1/2/5/10 from `177.79/155.78/116.54/90.55` to `189.21/162.75/122.26/93.77` bps | Research-only; not promoted downstream |
| E | `supportive_path_efficiency_h12` | Huber, direction `+1` | incumbent frozen 120 causal fields | `177.79/155.78/116.54/90.55` to `179.74/155.87/116.79/89.39` bps; stability flat/slightly lower | retain incumbent |
| T | `supportive_time_to_meaningful_mfe_h12` | Huber, direction `−1` | incumbent frozen 120 causal fields | stronger Top-1/2 (`186.70/157.88`) but weaker Top-5/10 (`113.76/88.46`) and stability | retain incumbent |

All values are timestamp-local realised net bps per trade.  The E/T full-
universe feature-selection challengers were separately rejected because they
reduced the incumbent B0+E+T conditional score:

| Conditional replacement arm | Top-1 | Top-5 | Top-10 | Stable Top-10 | Decision |
|---|---:|---:|---:|---:|---|
| Incumbent B0+E+T | +89.77 | +76.84 | +57.16 | +51.23 | retain |
| Replace E, best F25 | +64.25 | +54.11 | +40.04 | +35.15 | reject |
| Replace T, best F120 | +37.88 | +36.01 | +28.62 | +23.31 | reject |

### Frozen HPO parameters

| Head | Depth | Leaves | Learning rate | Minimum leaf fraction | Feature / bagging fraction | L1 / L2 | Min gain | Additional rank parameters |
|---|---:|---:|---:|---:|---|---|---:|---|
| B | 4 | 27 | 0.06959 | 0.005335 | 0.85784 / 0.71416 | 0.15466 / 0.11575 | 0.000946 | sigmoid 0.84051; truncation 5 |
| E | 4 | 39 | 0.01946 | 0.005441 | 0.83626 / 0.88730 | 0.001177 / 0.18609 | 0.006897 | Huber alpha 0.85235 |
| T | 3 | 23 | 0.02447 | 0.007664 | 0.88313 / 0.79440 | 0.008156 / 0.13721 | 0.005133 | Huber alpha 0.90856 |

The full ordered feature contracts and the machine-readable parameters are
immutable in:

- B: [winner.json](../data_perp/artifacts/strict_r3_direct_head_crossyear_hpo_20260826_b_v1/winner.json), feature SHA `977228b53c99a1984ffcbe061c1ba234f07abfe0d47231e51cb6b2b9c5a44d86`.
- E: [winner.json](../data_perp/artifacts/strict_r3_direct_head_crossyear_hpo_20260826_e_v1/winner.json), feature SHA `184dab8a60fde4918c18de0f4f88d8b3cd2750f432e6a355ce76ca3242565b78`.
- T: [winner.json](../data_perp/artifacts/strict_r3_direct_head_crossyear_hpo_20260826_t_v1/winner.json), same frozen 120-field SHA as E.

The B selection sequence was `1,407 → 1,199` hygiene-valid → `1,094`
after the near-duplicate veto → Screen120 → OOF MDA ladder → F70 → F72
structure/location add-back.  The exact F72 selection is in
[selection.json](../data_perp/artifacts/strict_r3_b0_family_addback_20260826_v1_policy_ordinal_base_g3/selection.json).
E and T used the same broader causal universe and the strengthened
conditional-selection test, but neither replacement survived it.

## Upstream blend grid: encouraging but insufficient

The strict-OOF grid chose `B=40%`, `E=55%`, `T=5%` over equal rank blending
on the upstream development objective.

| Upstream rank blend | Top-1 | Top-2 | Top-5 | Top-10 | Weekly q10 Top-2 | Worst-month Top-10 |
|---|---:|---:|---:|---:|---:|---:|
| Equal B/E/T | +178.85 | +159.53 | +116.31 | +90.08 | +96.35 | +49.81 |
| B40 / E55 / T05 | +188.78 | +176.65 | +135.49 | +106.74 | +105.26 | +53.25 |
| Delta | +9.93 | +17.12 | +19.18 | +16.66 | +8.91 | +3.44 |

This shows why an upstream-only conclusion would have been misleading: the
blend is clearly better at timestamp-local selection but not sufficiently
calibrated for the existing downstream stack.

## Required downstream evidence and result

The following comparison uses common source populations for the new blend and
the matched incumbent control.  Both are scored Jan–Jul 2026 and replayed
Apr–Jul 2026 under identical policy labels, residual/consensus fitting,
prequential BCF/current MC1 mappings, dual admission and global portfolio
constraints.  `Live baseline` is separately reported because its historical
candidate universe is narrower; it is not the direct identity-matched control.

| Arm / dual-MC1 gate | Entries | Net EV / trade | Total net bps | Worst month | Worst week | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|
| New B40/E55/T05, 30 bps | 3,738 | +105.80 | +395,464 | +70.05 | +42.50 | −32.16% |
| Matched incumbent, 30 bps | 3,117 | +136.93 | +426,805 | +115.41 | +84.48 | −25.24% |
| **Challenger minus matched**, 30 bps | **+621** | **−31.13** | **−31,342** | **−45.36** | **−41.98** | **−6.92pp** |
| New B40/E55/T05, 50 bps | 3,249 | +125.15 | +406,612 | +81.96 | +39.37 | −32.20% |
| Matched incumbent, 50 bps | 2,597 | +159.81 | +415,035 | +130.22 | +93.53 | −25.35% |
| **Challenger minus matched**, 50 bps | **+652** | **−34.66** | **−8,423** | **−48.27** | **−54.16** | **−6.85pp** |

The higher entry count is not enough compensation: the new blend admits a
materially weaker marginal population.  It fails the predeclared economic,
worst-period and drawdown gates, so its frozen HPO and blend weights must not
replace the retained B/E/T upstream contract.

For scale, the separate historical-live source replay records `+139.50` bps
per trade at 30 bps and `+154.50` at 50 bps.  The matched incumbent’s different
routed population explains its greater coverage; it does not licence a direct
promotion comparison against that narrower live control.

## Reproducibility map

| Stage | Evidence |
|---|---|
| B/E/T HPO | `strict_r3_direct_head_crossyear_hpo_20260826_{b,e,t}_v1` |
| Strict OOF B/E/T scores | `strict_r3_frozen_head_oof_20260826_{b,e,t}_v3` |
| Blend grid | `strict_r3_threeway_rank_blend_grid_20260826_v1` |
| New target-free score source | `strict_r3_frozen_threeway_targetfree_20260826_v2` |
| Exact matched incumbent source | `strict_r3_frozen_threeway_matched_control_targetfree_20260826_v2` |
| Downstream scoring / maps | `strict_r3_hpo_threeway_downstream_{challenger,matched_control}_20260826_v1` |
| Sealed 30/50 bps replays | `strict_r3_threeway_admission_threshold_{challenger,control}_{30,50}_20260826_v3` |
| Historical-live reference | `strict_r3_threeway_admission_threshold_live_baseline_{30,50}_20260826_v1` |

The relevant runners are:

- `scripts/run_strict_r3_direct_head_crossyear_hpo_v1.py`
- `scripts/score_strict_r3_frozen_head_oof_v1.py`
- `scripts/run_strict_r3_threeway_rank_blend_grid_v1.py`
- `scripts/materialize_strict_r3_frozen_threeway_targetfree_v1.py`
- `scripts/run_strict_r3_hpo_threeway_downstream_validation_v1.py`
- `scripts/replay_strict_r3_threeway_admission_threshold_v1.py`

## Next allowed research step

Do not retune this rejected blend on the same period.  Any future three-head
challenger must first demonstrate a mechanism that improves calibration of the
marginal population before entering the same untouched downstream/MC1
evaluation path.
