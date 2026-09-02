# Short P0 → O → C → K0: next ablation funnel

## Scope and decision

This is a short-only, research-only strict-prequential funnel. It does not
change long-side code, canonical artifacts, live inference, policy, portfolio
rules, or the fixed admission floor.

The frozen architecture evaluated throughout is:

```text
P0/F90 relative ranker
  → O45: P(short MFE within 6h > 250 bps)
  → C59: conditional five-state normalized-regret conversion score
  → K0 = p(O) × μ1(C) + (1 − p(O)) × μ0(P0 anchor)
  → admit only when expected policy net is at least +75 bps
```

Every held candidate is scored target-free. Exact policy paths and rich MFE/
MAE paths are used only after score generation, for fitting on prior-resolved
rows and for outcome measurement. Invalid/incomplete paths are never encoded
as economic failures.

The selected research configuration remains **frozen O45 + frozen C59 +
G1 p(O)-quintile residual correction (k=250) + H0 five-bin P0-anchor μ0
shrinkage (k=500)**. It is a short-research challenger only. It has not been
promoted to a canonical or live stack.

## Era coverage

| Era | Status | Use in decisions |
|---|---|---|
| 2024 | Causal warm-up and diagnostic only | Not pooled into selection. The stored P0 population starts in May; frozen O45 starts in October; a strict C59/K0 mapping needs three purged inner C-OOF slices and first has full support in April 2025. |
| 2025 | Strict monthly outer OOF | Primary selection era. Full C/K0 scorecard is Apr–Dec. |
| 2026 | Strict monthly outer OOF | Primary selection era, Jan–Jul. |

The sparse 2024 K0 examples are deliberately not called performance evidence.
The original frozen O45/C59 K0 control has only **5 known admitted outcomes**
and the strongest competing-risk diagnostic has **14**. The later C-SP
coverage-repair control does provide three diagnostic 2024 months (320 known
admissions, +59.84 bps/trade), but its feature contract and development HPO
were selected during 2024; it is therefore development evidence, not an
independent third selection era. A proper independently scored 2024 K0 era
requires a pre-2024 selected feature contract plus older compatible
target-free P0/features history, so O/C/K0 can warm up before 2024.

### Comparable full-stack evidence

The selected G1/H0 contract is first fully supported in **July 2025**: the
October--December 2024 C59 outputs are needed as prior resolved outer-OOF
history for the K0 map. The following is therefore the only comparable
full-stack scorecard; it is strict prequential and excludes the 2024 warm-up
from every selection statistic.

| Period | Months | Known admitted trades | Net bps/trade | Total net bps | Worst month | CVaR10 | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| 2024-10--12 C59 warm-up | 3 | 305 | +74.22 | +22,637.55 | +55.69 | -487.70 | Causal upstream diagnostic only; not a G1/H0 evaluation and not independent of 2024 development. |
| 2025-07--12 G1/H0 | 6 | 516 | +160.50 | +82,819.50 | +32.06 | -542.79 | First independent full-stack era. |
| 2026-01--07 G1/H0 | 7 | 340 | +127.27 | +43,273.24 | +10.86 | -645.18 | Second independent full-stack era. |
| 2025-07--2026-07 G1/H0 | 13 | 856 | +147.30 | +126,092.74 | +10.86 | -583.46 | Pooled strict-OOS evidence; the selection basis. |

The closest earlier MC1-equivalent diagnostic is also inadequate to turn 2024
into a third era: it has only December 2024 support, with 14 known MC1
admissions. It is retained as a thin warm-up diagnostic only and has no
selection or promotion authority.

## Causality contract

- Outer model fitting uses only `label_available_at < held-month start`.
- Inner OOF calibration uses purged chronological slices; labels resolve before
  each inner fit cut-off.
- O hard-negative weights, where tested, use only OOF O scores.
- C fits only true O-positive training rows; that condition is never requested
  at inference.
- K0 mapping uses prior *outer-OOF* scores and prior-resolved outcomes only.
- `μ0` receives only `prequential_base_anchor_bps`; it receives no path,
  outcome, regime, trust, consensus, or future field.
- Admission is fixed at `K0_expected_policy_net_bps >= 75` throughout.

## Phase outcomes

### A — opportunity timing

The static O45 binary classifier remains the portable opportunity choice.

| Arm | 2025 net bps/trade | 2026 net bps/trade | Worst month | Result |
|---|---:|---:|---:|---|
| A0 static binary O45 | 169.04 | **217.07** | -60.33 | Retained |
| A2a 1h hazard, uniform intervals | **183.53** | 171.23 | +9.61 | Reject: harms 2026 materially |
| A2a 1h hazard, early weighting | **188.38** | 171.01 | -110.42 | Reject: harms 2026 and concentration |
| A2c 2h hazard, uniform intervals | 175.14 | 159.47 | +51.55 | Reject: loses 2026 EV |

The hazard models learn timing, but their timing information does not produce a
better two-era K0 stack. Therefore the `p_hit_2h` residual in G3 is not run.

### B — competing risks

Multinomial favorable-before-adverse with the 3 ATR adverse definition is the
best competing-risk arm, but it does not advance over static O45.

| Arm | 2025 net bps/trade | 2026 net bps/trade | Worst month | O AUC | Top-20 precision | Result |
|---|---:|---:|---:|---:|---:|---|
| A0 static binary O45 | 169.04 | **217.07** | -60.33 | 0.635 | 0.506 | Retained |
| B multinomial, adverse 3 ATR | **177.10** | 173.49 | +27.05 | 0.617 | 0.468 | Reject: lower 2026 EV and O quality |
| B cause-specific, adverse 2 ATR | 161.46 | 151.39 | +1.95 | 0.539 | 0.341 | Reject |

The available 2024 competing-risk diagnostics cover only Nov–Dec. For the
multinomial 3-ATR arm they comprise 14 known admissions at +213.26 bps/trade;
this is too little support to influence model choice.

### C — feature gap review

The required inventory is in `docs/SHORT_OC_FEATURE_GAP_REVIEW.md`. It
approved only four bounded target-free blocks; every field passed the later-era
90% coverage/variance gate:

| Block | Layer | Causal fields added | Head-level result, 2025–2026 strict OOF | Full-stack result | Decision |
|---|---|---|---|---|---|
| SF | O or C | three existing short-form recovery/relative-return fields | O precision@20 0.514 vs 0.510 control; C Spearman 0.419 vs 0.424 | loses 2026 in both O and C tests | Reject |
| TF | O or C | `false_clean_short`, price/OI recovery, market OI breadth | O AUC 0.657 vs 0.656 and precision@20 0.515; C Spearman 0.420 | no two-era K0 improvement | Reject |
| XS | O | seven pre-existing complete-universe cross-sectional fields | O AUC 0.659, PR-AUC 0.487, precision@20 0.516—best O-block target metrics | +136.24 / +140.60 bps, but worse worst month (−17.93) | Diagnostic only; does not displace frozen O45 |
| SP | C | four prior-only 30-day self-percentiles | C Spearman 0.427 vs 0.424 frozen C59 | +134.57 / +144.19 bps, CVaR10 −378.31 vs −394.89 | Carried to matched C-SP K0/HPO challenger; not promoted without a common-period comparison |

`O_XS + C_SP` was the only permitted combination after the individual screen.
It regressed 2026 to +135.49 bps and produced a −42.60-bps worst month, so it
is rejected. No generic liquidity, spectral, session, or post-entry/path block
was added. The C-SP HPO below tests the same frozen C-SP fields; it does not
repeat feature selection.

### D — false-positive opportunity weighting

Uniform O weighting remains selected. The matched C59 D0 ledger begins in
April 2025, so it is compared only with D arms, not with the earlier A0
headline.

| Arm | 2025 net bps/trade | 2026 net bps/trade | Worst month | O AUC | Top-20 precision | Result |
|---|---:|---:|---:|---:|---:|---|
| D0 uniform | **187.94** | **149.02** | -23.39 | 0.642 | 0.508 | Retained |
| D1 OOF top-10% hard negatives ×1.5 | 177.43 | 145.70 | +2.00 | **0.648** | **0.512** | Reject: lower EV in both eras; worse CVaR |
| D2 graded OOF hard negatives | 162.02 | 126.11 | -28.31 | 0.621 | 0.470 | Reject |

### E/F — conditional conversion target

The frozen five-state C3 normalized-regret target remains selected. Quantile
targets increase conditional rank metrics in places, but do not improve K0
economics across both selection eras.

| Arm | 2025 net bps/trade | 2026 net bps/trade | C rank IC | C net Spearman | Worst month | Result |
|---|---:|---:|---:|---:|---:|---|
| E0 C3 five-state control | 170.46 | **173.85** | 0.457 | 0.405 | -9.16 | Retained |
| E1 cumulative ordinal | **177.43** | 137.58 | 0.438 | 0.390 | -53.26 | Reject: 2026 degradation |
| E2 continuation ratio | 172.40 | 145.56 | 0.438 | 0.395 | -60.27 | Reject |
| F2 q75 normalized regret | 152.81 | 144.32 | **0.485** | **0.417** | -20.11 | Reject: target fit does not transfer to K0 EV |
| F3 q25/q50/q75, λ=.50 | 146.80 | 145.85 | **0.491** | 0.416 | -6.90 | Reject: lower EV in both eras |
| F4 Huber normalized regret | 142.31 | 135.28 | 0.489 | 0.408 | -26.46 | Reject |

### Final limited O/C HPO

The final HPOs were intentionally separate. Each used chronological 2024
development folds, whole-day-preserving subsampling where applicable, early
stopping, MedianPruner only after two folds, and a hard stop at **20
consecutive completed/pruned non-improvements**. Neither changed the other
head, K0, the candidate population, or the +75-bps admission rule.

| Head | Control | HPO result | Decision |
|---|---|---|---|
| C, C-SP feature contract | +134.57 bps (2025), +144.19 (2026), CVaR10 −378.31 | 43 trials; best single seed +137.80 / +128.96, CVaR10 −403.86; seed-3 +138.44 / +123.74, CVaR10 −399.83 | Reject: lower 2026 EV, lower total EV, worse CVaR. Keep the uniform control. |
| O, frozen O45 contract | +187.94 bps (2025), +149.02 (2026), CVaR10 −399.22 | 43 trials; +180.47 / +140.35, CVaR10 −433.68. Worst month improves −23.39 → +1.21. | Reject: both era EVs and CVaR deteriorate; the isolated worst-month improvement is insufficient. Keep frozen O45. |

The selected O45 HPO development parameters were depth 5, 19 leaves, 142
trees, learning rate 0.0334, minimum child 85, subsample 0.889, feature
fraction 0.851, L2 24.49, and `extra_trees=True`. They are recorded for
reproducibility but are **not** adopted.

The selected C-SP HPO development parameters were depth 2, 26 leaves, 432
trees, learning rate 0.0156, minimum child 102, subsample 0.954, feature
fraction 0.911, L2 2.86, and no extra trees. They are likewise rejected.

### G/H — analytic K0 calibration

This phase corrects the fallback to the required P0-anchor contract. Earlier
timing/conversion scripts used an inherited scalar `μ0`; those scalar figures
are not a valid H0 baseline and must not be presented as the final P0-anchor
K0 result.

The matched corrected control is:

```text
G0: μ1(C) isotonic
H0: μ0(P0 anchor) in five training-quantile bins, k=500 shrinkage
```

The only advancing mapping is a small one-dimensional μ1 residual correction:

```text
G1: p(O) quintile → residual(policy_net − μ1(C)), k=250
H0: unchanged five-bin P0-anchor μ0, k=500
```

| K0 arm | 2025 net bps/trade | 2026 net bps/trade | Mean | Known admissions | Worst month | CVaR10 | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| Corrected G0/H0 control | 159.32 | 109.13 | 140.12 | 818 | -6.72 | -603.21 | Control |
| **G1 p(O) quintile k250 + H0** | **160.50** | **127.27** | **147.30** | **856** | **+10.86** | **-583.46** | **Selected** |
| G2 p(O) tercile k250 + H0 | 160.50 | 126.82 | 147.10 | 857 | +10.86 | -584.03 | Near tie; lower 2026 EV |
| G1 + H3 isotonic μ0/support k500 | 152.77 | 105.61 | 130.81 | 1,089 | -29.64 | -555.11 | Reject: lower both eras |
| G1 + H2 ten-bin μ0 k500 | 150.23 | 96.51 | 124.07 | 1,031 | -29.68 | -558.81 | Reject |

The μ1 correction is bounded and conservative: it can adjust the conditional
value by the shrinkage-weighted residual observed in a `p(O)` quintile; it
cannot create a separate multivariate mapper or change the probability model.
All H1/H2/H3 alternatives are rejected by the predeclared gate: retain at
least 80% participation, neither 2025 nor 2026 EV worse by more than 10 bps,
and CVaR10 not worse by more than 25 bps.

## Required next action for meaningful 2024 evidence

Do not relax purging or calibration support to manufacture a 2024 K0 score.
Instead materialise pre-May-2024 short P0 target-free population, causal
features, and exact policy/rich labels from an older compatible history. Then
rebuild O45 and C59 strictly prequential so 2024 receives an independent
outer-OOF period with the same K0 contract. Only then can 2024 be included as
a third selection era.

## Artifacts and scripts

| Component | Script | Immutable result |
|---|---|---|
| Exact timing/competing-risk labels | `scripts/materialize_strict_r3_short_p0_oc_k0_event_timing_labels.py` | `data_perp/artifacts/strict_r3_short_p0_oc_k0_event_timing_labels_202405_202607_20260822_v8` |
| A: timing O | `scripts/run_strict_r3_short_p0_oc_k0_phase_a_timing.py` | `data_perp/artifacts/strict_r3_short_p0_oc_k0_phase_a_timing_202408_202607_20260822_v3` |
| B: competing-risk O | `scripts/run_strict_r3_short_p0_oc_k0_phase_b_competing_risk.py` | `data_perp/artifacts/strict_r3_short_p0_oc_k0_phase_b_competing_risk_202408_202607_20260822_v1` |
| D: false-positive O weights | `scripts/run_strict_r3_short_p0_oc_k0_phase_d_false_positive.py` | `data_perp/artifacts/strict_r3_short_p0_oc_k0_phase_d_false_positive_202408_202607_20260822_v1` |
| E/F: C target/objective | `scripts/run_strict_r3_short_p0_oc_k0_phase_ef_conversion.py` | `data_perp/artifacts/strict_r3_short_p0_oc_k0_phase_ef_conversion_202408_202607_20260822_v1` |
| Final C-only HPO | `scripts/run_strict_r3_short_p0_oc_k0_round3_c_hpo.py` | `data_perp/artifacts/strict_r3_short_p0_oc_k0_c_sp_hpo_202408_202607_20260822_v2` |
| Final O45-only HPO | `scripts/run_strict_r3_short_p0_oc_k0_o45_hpo.py` | `data_perp/artifacts/strict_r3_short_p0_oc_k0_o45_hpo_202408_202607_20260822_v1` |
| G/H: analytic K0 mapping | `scripts/run_strict_r3_short_p0_oc_k0_phase_gh_mapping.py` | `data_perp/artifacts/strict_r3_short_p0_oc_k0_phase_gh_mapping_202408_202607_20260822_v1` |

The Phase G/H artifact is deterministic: the final artifact and an independent
smoke replay have identical Phase H summary tables.
