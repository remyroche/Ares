# Short P0/F90 Side-Aware Pipeline Audit

Status: **completed research iteration — P0/F90 base canonical; no short live authority**  
Scope: canonical P0/F90 short base followed by the long-stack-equivalent,
strict-prequential downstream pipeline.

## Frozen short base

P0/F90 is the canonical short **relative ranker** for this research iteration.
It is not an absolute-EV admission model and must not be re-targeted merely to
make it one.

| Contract | Evidence |
|---|---|
| Side | short only |
| Feature-selection artifact | `data_perp/artifacts/strict_r3_short_policy_conversion_p1_k32_chronological_mda_20260820_v1/selected_features.json` |
| Frozen selected fields | 90 target-free causal fields (from 115 fields passing the `>=90%` population coverage gate) |
| Base configuration | `config/strict_r3_short_p0_f90_base_v1.json` |
| October–December 2024 confirmation rank IC | 0.1522 |
| Positive-ranking hours | 87.59% |
| Top-1 / Top-2 / Top-4 uplift vs query mean | +101.19 / +86.53 / +77.98 bps |
| Strict 2025 three-block OOS IC | 0.1221 / 0.1469 / 0.1178 |
| Strict 2025 OOS positive-ranking hours | 79.76% / 81.52% / 76.22% |
| Strict 2025 top-1 global uplift | +91.09 / +81.41 / +127.90 bps |

These results establish relative opportunity ordering only. Admission is the
responsibility of the downstream trust/EV-map stack.

## Frozen short Geometry/K9 representation

The geometry/K9 definition is frozen exactly once, not refit per month or
fold. This preserves feature semantics for every downstream consumer.

| Item | Value |
|---|---|
| Definition window | 2024-10-01 through 2025-01-01 UTC |
| Complete warm-up population | 196,340 rows |
| Encoder | 64-tree geometry encoder |
| Encoder fit month coverage | 61,949 October / 70,525 November / 63,866 December rows |
| K9 fit | 100,000 equal-month sampled warm-up rows |
| K9 fit month coverage | 33,333 October / 33,334 November / 33,333 December rows |
| Membership temperature | 6.13564157 |
| Bundle | `data_perp/artifacts/strict_r3_short_p0_f90_geometry_k9_octdec2024_20260821_v3` |
| Bundle SHA-256 | `2c93f1690ef36a780f2aa7b5c32b125937827f2e760913ddc3d68b65e24c9262` |
| Stable downstream representation | 33 aggregate geometry/K9 fields; raw `k09__cluster_*` membership fields excluded |

## Consensus process

The new consensus contract follows the long-side topology but is short-only:

- ten CMI-selected LambdaRank residual heads, with feature caps from 40 to
  120 and a 3:1 base-to-Geometry/K9 diversity constraint;
- six exact-timestamp-side and four 4-hour-side query candidates;
- residual ordinalisation, query geometry, weighting, and LightGBM parameters
  selected by **incremental 75% P0 base + 25% median-of-accepted-heads
  ensemble performance**, not standalone head performance. Each trial is
  scored against the prior accepted ensemble on the same chronological folds;
  promotion also requires all-tail and worst-month improvement;
- chronological 2025-Q1 development only; April 2025 onward remains untouched
  for promotion evidence;
- every head HPO stops after 20 consecutive non-improving completed **or
  pruned** trials (with 40 as its hard trial ceiling).
- the HPO sampler now precomputes the immutable month key once and reuses it
  in both sampling layers. This is a runtime-only repair: ordinary and
  equal-month sampled query identities, weights, labels, chronology and model
  parameters are unchanged (regression-tested).
- the downstream Top-1/2/5 and monthly portability evaluator likewise reuses
  the held fold's precomputed month key. Its complete conditional utility is
  regression-tested equal to the timestamp-derived implementation.

| Artifact | Status |
|---|---|
| Frozen candidate head contract | `data_perp/artifacts/strict_r3_short_p0_cmi_consensus_contract_20260821_v2/short_consensus_contract.json` |
| Ensemble-only HPO result | complete: `data_perp/artifacts/strict_r3_short_p0_conditional_consensus_hpo_20260821_v2`; zero of ten heads promoted |
| Strict OOS replay | base-only conversion in progress; retained history starts 2024-04-01 and evaluation starts 2025-04-01 |

The earlier v1 short-consensus experiment is explicitly rejected: its all-ten
blend reduced 2025 top-1 / top-2 / top-5 results from P0's +82.02 / +45.02 /
+1.52 bps to +18.27 / -7.80 / -32.68 bps. It is retained as negative
evidence, not as a production or research champion.

The v2 HPO independently reaches the same conclusion under a stronger,
ensemble-first selection rule. Its incumbent was P0/F90 alone; each candidate
was evaluated only as `75% base rank + 25% median(accepted heads plus this
candidate)` on the same January--March 2025 chronological development folds.
No standalone-head metric could promote a head. The candidate must improve the
complete conditional utility and the all-tail/worst-month gate.

| Candidate head | Trials | Stop | Best ensemble utility delta (bps) | Δ Top-1 | Δ Top-2 | Δ Top-5 | Decision |
|---|---:|---|---:|---:|---:|---:|---|
| cmi_cap40_equal_month | 33 | 20 non-improving | +0.44 | +9.21 | −0.20 | −10.40 | reject: Top-5 regresses |
| cmi_cap60_ordinary | 40 | hard ceiling | −5.42 | +0.39 | −9.30 | −11.05 | reject |
| cmi_cap120_ordinary | 40 | hard ceiling | −16.51 | −13.05 | −14.65 | −14.47 | reject |
| cmi_cap80_equal_month | 40 | hard ceiling | −22.62 | −20.99 | −21.33 | −14.89 | reject |
| cmi_cap120_equal_month | 25 | 20 non-improving | −23.76 | −24.51 | −21.98 | −20.60 | reject |
| cmi_cap80_ordinary | 40 | hard ceiling | −23.89 | −18.21 | −17.89 | −19.49 | reject |
| cmi_cap40_ordinary | 22 | 20 non-improving | −23.99 | −23.66 | −21.25 | −18.58 | reject |
| cmi_cap100_ordinary | 40 | hard ceiling | −24.73 | −10.36 | −22.86 | −20.24 | reject |
| cmi_cap100_equal_month | 24 | 20 non-improving | −31.54 | −24.56 | −30.17 | −27.76 | reject |
| cmi_cap60_equal_month | 21 | 20 non-improving | −34.52 | −19.25 | −25.27 | −25.62 | reject |

This is a fail-closed result, not a partial consensus. The short downstream
pipeline continues only through the explicit `base_only_no_consensus_head_passed_development_gate` contract: upstream equals P0/F90 rank, all residual
and BCF fields remain unavailable, and any later trust/MC1 evidence is
reported as a base-only challenger.

## Side-aware downstream contracts

The common long pipeline is now guarded for side-local use rather than relying
on convention:

- `strict_r3_cell_day_trust.py` persists and validates `side`; mixed-side
  trust ledgers fail before fit or scoring. Its short R5 model and
  posterior-admission contracts are now separate paths and fail if their
  declared side does not match the fit/scoring side; the legacy unscoped R5
  overlay is deliberately long-only.
- The long R5 field list is **not** reused for short P0/F90: 40 of its 66
  fields are absent under the short score schema. The new explicit short R5
  contract instead contains 13 score/conversion fields, 33 stable frozen
  Geometry/K9 aggregate fields, and 20 already coverage-gated P0 context
  fields. Raw `k09__cluster_*` memberships remain prohibited. Before score
  fields are added, all 53 physical non-score fields have at least 93.38%
  coverage on the P0/F90-eligible training population; the lower apparent
  coverage on the full target-free universe consists of rows intentionally
  ineligible for base scoring.
- `strict_r3_mc1_mapper.py` and `strict_r3_bcf_mc1_mapper.py` persist and
  validate bundle/row side.
- `materialize_strict_r3_bcf_mc1_ledger.py` now requires an explicit
  side-local score source and side-local parent-policy source, and records
  `side` in its immutable manifest. A long policy ledger cannot silently
  populate a short BCF mapper.
- policy materialisation and `simple_policy_optimiser` use short directional
  entries, stops, trailing logic, bid/ask orientation, and portfolio exposure.
- The sequential exact-one-minute rich-extension HPO now takes a sealed
  `--side`, requires a matching side on its frozen parent policy and exact
  path data set, and preserves that side through portfolio candidates,
  tournament, frozen evaluation, manifests and correctness receipts. The
  parent rich-policy HPO now writes the side into `frozen_challenger.json`.

Short trust, MC1, BCF, and policy bundles were then fit from the completed
short base-only ledger. Long artifacts and long feature lists were never used
as score, map, or feature substitutes.

Focused side-segregation coverage was re-run on 2026-08-21. The current
focused suite completed **19/19 tests**: strict prequential residuals,
Geometry/K9 input lineage, consensus sampling, trust, MC1, BCF,
rich-policy side handling, native BCF features, and the strict-history/
dual-admission mapper regression. The only warnings are expected pandas
timezone-to-period conversion warnings in the sampling-equivalence tests.

```text
test_short_bcf_ledger_side_contract,
test_short_rich_extensions_side_contract,
test_short_trust_side_contract,
test_short_mc1_side_contract,
test_short_consensus_hpo_sampling,
test_side_aware_contract, and test_strict_r3_side_local_pipeline.
```

## Same-model conversion and causal admission substrate

The base-only continuation keeps `upstream = base_rank42`; it never invents a
neutral residual score. Each held fold uses a strict-prequential P0/F90 model
and its same-model preceding 42-day rank reference. The 20-bin monotonic
policy-net map is fitted from earlier resolved predictions only.

| Artifact | Scope | Result |
|---|---|---|
| `strict_r3_short_p0_baseonly_conversion_oof_2025jan_2026jul_20260821_v1` | Jan-2025--Jul-2026 strict conversion OOF | complete; base-only and frozen Geometry/K9 hash preserved |
| `strict_r3_short_p0_baseonly_immediate_calibration_2025jan_2026jul_20260821_v1` | exact same-model 28-day producer reserve | 18 fitted reserve calibrators; January 2025 correctly fails closed for insufficient earlier resolved support |
| `strict_r3_short_p0_baseonly_cell_day28_admission_2025jan_2026jul_20260821_v1` | 28-day side-local Cell-day EV map | 1,696,069 scored candidates; 83,603 earliest reserve rows fail closed; 2,389 raw mapped admissions; no held outcome used for its own decision |

No path-validity, H12 completeness, or outcome is used to create a candidate,
score it, rank it, or admit it. Those data are joined only after the fixed map
decision for outcome evaluation.

## R5 trust and MC1 mapping

### R5 9-month posterior trust

The side-aware R5 contract was repaired before fit: the long-only
`ob_trade_size_to_l1_depth_z_24h` field covered only 84.65% of the actual
short training population. It was replaced, through a fixed pre-cutoff
residual-MI screen among frozen P0 causal fields, with
`oi_drawdown_from_peak_168h` (100% coverage; MI proxy 0.2970). The exact
66-field short contract is recorded in
`config/strict_r3_short_cell_day_residual_trust_model_r5_9m_v1.json`.

The target is clipped Cell-day policy residual:

```text
clip(policy_net_bps - causal_28d_cell_day_expected_net_bps, -500, +500)
```

Monthly R5 bundles use an equal-month top-30% training population and nine
months of prior history. The first valid short fit is December 2025; the
November cutoff is deliberately rejected because it falls one hour short of a
full nine months. The output
`strict_r3_short_p0_r5_monthly_oof_2025dec_2026jul_20260821_v1` has 735,278
rows, 222,704 mapped rows and 467 source-level posterior admissions. Its
direct mapped-row comparison is constructive (+4.66 bps/trade versus
−45.25 for the raw map), but the portfolio sample is far too small to
promote.

### MC1-d2

The frozen short MC1 bundle is
`strict_r3_short_p0_baseonly_current_mc1_bundle_20250701_extended_20260821_v1`
(`bundle_id=9f58dd1ebb1b784c2c1a`). It is a depth-2 histogram gradient
boosting expected-EV mapper, with 80 iterations, learning rate 0.04, L2 20,
minimum leaf 100 and seed 1729. Its six causal inputs are:

```text
final_score, base_rank42, conditional_consensus_rank, upstream,
ordinary_shadow_consensus_rank, correctness_rank
```

For base-only short, `conditional_consensus_rank` and `upstream` are the
base-ranked values under the explicit no-consensus contract; this is recorded
rather than silently substituting a long consensus. The mapper fit only rows
strictly before 2025-07-01. Its daily recent-global shift consumes only policy
labels resolved before the UTC decision day. The scorer's incremental 21-day
history was tested against a repeated full-history calculation on 11,400
July-2025 rows and produced identical identities and outputs.

Source-level MC1 results at +50 bps are 2,097 outcome-valid admissions at
+72.25 bps/trade. This is aggregate evidence only. The constrained rich-policy
replay below is the advancement evidence.

BCF is intentionally **unavailable**: its upstream residual-head ensemble
contains zero promoted head. Fitting a BCF mapper or borrowing long BCF fields
would violate the short contract.

## Side-aware rich SimplePolicyOptimiser

`scripts/run_strict_r3_rich_policy_hpo.py` was repaired to accept partitioned
side-local prequential ledgers and choose
`prequential_base_rank42` when the explicit ensemble `prequential_upstream`
field does not exist. The loader verifies the side, strict-prequential flag,
unique candidate identity, source score field and target-free entry
executability before loading a path.

The frozen short policy selection uses 2024 only: April--June form the
distributional/adverse calibration slice and July--December are the policy
selection slice. It uses a deterministic per-month cap of 3,500 rows from the
top 5% of the short base score. The broad 48-trial search, including the
canonical smooth-capital-protection branch, executed **38** trials: trial 17
was the final improvement and trials 18--37 were the required 20 consecutive
non-improvements. It stopped with reason
`no_improvement_patience`, rather than consuming the ceiling.

The earlier `strict_r3_short_p0_rich_policy_hpo_2024select_2025_2026_20260821_v1`
receipt is retained as a diagnostic only: it was generated before the smooth
capital-protection branch was included. The authoritative frozen policy is
`strict_r3_short_p0_rich_policy_hpo_smooth_2024select_2025_2026oos_20260821_v1/frozen_challenger.json`.

| Parameter | Frozen short rich-policy value |
|---|---:|
| Stop multiplier | 2.6046 ATR |
| Stop ATR transform | multiplier 1.25, power 1.00, 0.8% absolute floor |
| Trailing activation | 1.4220 ATR; floor 0.5%, cap 1.0%; decay starts immediately, 8-bar half-life, minimum 0.85x |
| Trailing gap | fixed 0.18658 ATR |
| Fast adverse exit | enabled; MAE 1.25 ATR, speed 0.30, maximum prior MFE 0.35 ATR, 85th-percentile severity; calibrated theta 3.59361 |
| Smooth capital protection | explicitly searched (activation 1.0/1.25/1.5/1.75/2.0/2.5; strength 0.25/0.5/0.75; power 0.75--2.0) but disabled by the 2024-only winner |
| Entry / horizon / cost | decision-time first 15-minute open / 48 bars (H12) / 100 bps exactly once |

The broad short-score control versus the frozen rich policy, both through the
same chronological auction, is:

| Year | Arm | Trades | Net bps/trade | Total net bps | Worst month | Worst week | Max drawdown (bps) |
|---|---|---:|---:|---:|---:|---:|---:|
| 2025 | parent control | 1,867 | +67.94 | +126,847 | −10.86 | −105.95 | −6,871 |
| 2025 | frozen rich policy | 2,277 | +55.10 | +125,456 | −5.57 | −78.63 | −7,663 |
| 2026 | parent control | 985 | +12.14 | +11,961 | −83.14 | −417.44 | −17,783 |
| 2026 | frozen rich policy | 1,177 | +7.10 | +8,352 | −29.96 | −148.12 | −11,235 |

This validates the **side-aware policy mechanism**, not a short trading stack:
those source-score rows are a development-tail control, not a causal
mapper-admitted population.

## Final 2025--2026 full-stack constrained replay

`scripts/replay_strict_r3_short_p0_mapped_rich_policy.py` is the final
full-stack evaluator. For each arm it performs:

```text
target-free P0/F90 candidate identity
→ frozen strict OOS mapper decision, expected EV >= +50 bps
→ only then exact 15-minute H12 path materialisation
→ frozen short rich policy
→ chronological auction: two concurrent / two new per timestamp / one asset
   (priority: the arm's mapped expected net bps)
```

It does not refit a mapper, score a policy path before admission, or reserve
portfolio capacity for an unlabelled row. All mapper identities joined to the
short prequential ledger exactly (zero missing identities). The full machine
receipts are in
`data_perp/artifacts/strict_r3_short_p0_mapped_rich_policy_smooth_oos_2025_2026_20260821_v1`.

| Mapping arm | Mapper admissions before path | Portfolio trades | Net bps/trade | Total net bps | Max DD (bps) | Worst month | Worst week |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw Cell-day 28d | 2,389 | 306 | **+74.36** | +22,756 | −3,924 | −7.64 | −256.53 |
| R5 posterior trust | 311 | 39 | +74.32 | +2,899 | −480 | −100.02 | −120.00 |
| MC1-d2 | 3,613 | 734 | +41.90 | +30,751 | −8,846 | −179.68 | −344.03 |

| Arm | 2025 constrained result | 2026 constrained result | Portability verdict |
|---|---|---|---|
| Raw Cell-day 28d | 252 trades, +77.82 bps/trade | 54 trades, +58.24 | positive but very sparse/intermittent |
| R5 posterior trust | no eligible portfolio rows before Dec-2025 | 39 trades, +74.32; February −100.02 | reject: inadequate support and negative month |
| MC1-d2 | 460 trades, +69.38 | 274 trades, **−4.26**; March--May and July negative | reject: aggregate-positive overall but negative 2026 |

MC1 exit attribution confirms that its weakness is not hidden by a mean:
300 trailing exits contribute +195.91 bps/trade, while 82 stop losses cost
−358.46 bps/trade and 330 timeouts average only +11.09 bps/trade. Its 2026
portfolio result cannot support short-side promotion.

## Final advancement decision

1. **Promote P0/F90 as the canonical short relative base model.** Its feature
   contract, geometry/K9 state and chronological MDA process are frozen.
2. **Do not promote residual consensus or BCF.** No short consensus head
   passed the ensemble gate.
3. **Do not promote R5 or MC1 as short admission/live mapping.** Neither
   passes the monthly portability requirement after the exact rich-policy
   exit and constrained portfolio are applied.
4. **Do not enable short Kraken execution.** The canonical executable stack
   remains long-only.
5. Further short work must be predeclared and target the conversion/admission
   failure, not re-target P0/F90's portable relative ranker. Any new HPO must
   stop after 20 consecutive non-improving trials.
