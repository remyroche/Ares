# Strict-R3 D2 + Conditional Consensus + Top-30 Correctness + 28-Day Cell-Day EV + A5 Bounded-10 Trust — Long-Only Canonical Handover

> **A5 bounded-10 promotion — 2026-08-12.** By explicit decision, the final
> trust layer is now `F2_blend_a10_fixed_A0_top15`, not ungated R5 posterior.
> R5 remains the causal A0 anchor used inside A5; it is no longer the final
> auction/admission value on its own. The executable formula is
> `A0 + 0.10 * (causally calibrated A4 - A0)`. A4 is the independently
> supported, neutral-mean residual forest. Its calibration uses only earlier
> OOS A4 predictions whose policy labels resolved before the new bundle
> cutoff. Admission is fixed before the A5 rerank: `A0 >= +50 bps` and the
> candidate must be in the timestamp-local top 15% by the pre-trust
> `final_score`. A5 cannot add or remove a candidate relative to that fixed
> A0-top-15 gate; missing A0, A4, or calibration fails closed. There is no
> second EV-map top-20 gate.

> **28-day admission and A0 anchor — 2026-08-12.** The canonical
> admission map is now `strict_oof_exact_producer_cell_day_trim15_28d_v1`.
> Each exact producer's excluded 28-day OOS reserve freezes twenty score cells.
> Policy outcomes are
> reduced to one mean per UTC day and cell; every retained day has equal weight;
> the highest and lowest 15% of days are removed symmetrically; the resulting
> curve is monotone in score; and admission still requires at least +50 net bps.
> The R5 Cell-day residual trust model remains the frozen nine-month A0 anchor.
> A0 owns the +50-bps economic floor, but final admission additionally applies
> the timestamp-local top-15 domain and final auction ordering uses A5
> bounded-10. Historical 42-day maps are
> archived comparison artifacts only; every newly materialised admission map
> and trust target uses the explicit 28-day contract.

> **Historical demotion-only correction.** The prior three-month R5 handoff reported uplift
> from an evaluator that calculated the declared three-way corroboration flag
> but did not apply it to authority. The production implementation now follows
> the prose contract exactly: authority is zero unless `P(overestimate >= 100
> bps) >= 0.65`, residual q25 is at most -50 bps, and effective support is at
> least 120. On the regenerated 28-day May--July 2026 replay, the probability
> gate never fired. R5 was therefore a safe no-op: both control and overlay
> accepted 859 identical trades at +66.13 net bps/trade. Historical R5 uplift
> remains ablation evidence only and is not a canonical performance claim.

> **28-day continuation result — 2026-08-12.** The missing August 2025--March
> 2026 latest-model-fit trust substrate has now been materialised, and R5 was
> compared with 3/6/9/12-month training windows on matched May--July 2026
> candidates. The promoted posterior integration uses R5's posterior expected
> policy net directly with the same +50-bps hurdle. The 9-month arm improves
> constrained EV from +72.30
> to +114.36 bps/trade, restores June admission, has three positive active
> months, and reduces max drawdown from -56.63% to -40.79%. Because this period
> selected both the window and integration mode, it is frozen as a research
> configuration. It is now the canonical executable-research configuration by
> explicit decision, but remains shadow-only pending later untouched evidence.
> A schema-v5 inference bundle is now sealed for hourly shadow use; the old
> schema-v3 bundle remains a historical demotion-only bundle. Full lineage is in
> `docs/CELL_DAY_RESIDUAL_TRUST_OVERLAY_20260812.md`.

> **Backward walk-forward confirmation.** Seven additional frozen 9-month R5
> folds over October 2025--April 2026 confirm the posterior-admission behavior.
> On the contiguous October 2025--July 2026 constrained replay, R5 produces
> +161.22 versus +115.68 net bps/trade, 41/42 positive active weeks, Sortino
> 0.490 versus 0.370, and max drawdown -43.81% versus -60.55%. This is strong
> OOS-by-fold research confirmation, but not untouched evidence; the selected arm
> is now canonical research evidence, not production approval.

> **Status correction — 2026-08-12.** The historical N5 results below were
> produced against an earlier score producer.  A source-aligned strict-lockstep
> replay found the old N5 sidecar's `final_score` correlation to the repaired
> producer was only 0.654, so it is not valid canonical evidence.  A repaired
> block-OOF refit on the exact score/admission contract underperformed the
> matched unit-size control (−0.80 net bps/trade; Sortino −0.085; max drawdown
> worsened by 16.84 percentage points) for January–July 2026.  Accordingly,
> N5 is a research challenger pending a matched 2025 replay and is not an
> active canonical sizing layer.  The base + consensus + correctness score and
> causal EV-admission stack remain the executable research control.

> **Schema-v5 correction — 2026-08-11.** The schema-v4 description below is
> retained only as historical research lineage.  It used a rolling C3/K9
> geometry representation and must not be used for inference, new OOS claims,
> or deployment.  The authoritative current contract is the frozen-geometry
> schema-v5 stack in this section.  Its October–December 2024 geometry/K9
> definition is fitted once, then retained unchanged for every 2025–26 fold;
> raw K9 memberships are never inference inputs.

## Schema-v5 — current executable research contract

`schema-v5` names the model architecture below. The sealed file format used by
the hourly scorer is independently versioned as inference-bundle `schema-v5`.
Those two version numbers describe different contracts and are not expected to
match.

```text
target-free point-in-time long candidates
→ strict-R3 D2 base, fitted only on resolved prior labels
→ same-base prior-28-day rank and prequential policy-net map
→ ten conditional-usefulness policy-residual LambdaRank heads
→ 75% base-rank + 25% conditional-consensus rank
→ one frozen Oct–Dec 2024 geometry/K9 representation
→ top-30% policy-residual correctness ranker
→ same-model prior-28-day CDF
→ exact-producer 28-day equal-day expected-net map with symmetric 15% day trim
  (twenty fixed reserve score cells; >= +50 bps)
→ A0/R5 residual trust anchor, trained on the preceding 9 months
→ A4 independent-local residual forest, trained on 75% timestamp-top-30 plus
  25% lower-reference rows from the same causal nine-month history
→ causal Huber calibration of A4 from prior resolved OOS A4 predictions
→ fixed admission: A0 >= +50 bps AND timestamp-local pre-trust top 15%
→ auction value: A0 + 10% × (calibrated A4 − A0)
→ unit relative sizing (the 45-field LDF is shadow-only)
→ frozen SimplePolicyOptimiser policy and constrained long-only auction
```

The strict-R3 historical producer is
[run_strict_r3_canonical_walkforward.py](/Users/remyroche/Documents/Ares/scripts/run_strict_r3_canonical_walkforward.py).
The forward scorer is
[score_strict_r3_forward.py](/Users/remyroche/Documents/Ares/scripts/score_strict_r3_forward.py),
followed by
[admit_strict_r3_forward.py](/Users/remyroche/Documents/Ares/scripts/admit_strict_r3_forward.py).
They do not consume outcomes while scoring or make held-window percentile
calculations.  Every forward score now carries a semantic score-family ID,
conversion-bundle hash, monthly upstream base/meta hash, and frozen geometry
hash.  [admit_strict_r3_forward.py](/Users/remyroche/Documents/Ares/scripts/admit_strict_r3_forward.py)
accepts only prior-resolved strict-OOF outcomes from that **exact producer**.
Every newly fitted producer has an excluded 28-day reserve scored by that same
bundle, so its fixed score cells and equal-day seed are available from its first
live hour.  It does not pool raw scores or outcomes across independently fitted
base/meta or conversion bundles.  The calibrator persists the complete sorted
reserve score reference and compact day × cell seed; inference therefore uses
the exact same cell assignments as replay.
The reusable prequential base/map/residual ledger is generated by
[materialize_strict_r3_prequential_ledger_v2.py](/Users/remyroche/Documents/Ares/scripts/materialize_strict_r3_prequential_ledger_v2.py)
with `--reference-days 28`; its fold audit and manifest persist the physical
window. The active lock-step scorer also derives its reference start from the
canonical `REFERENCE_DAYS = 28` constant. This is the required contract for
all newly fitted bundles.

The completed homogeneous ledger is
`data_perp/artifacts/strict_r3_schema_v2_prequential_ledger_targetfree_long_2024_2026_raw15m_strictfull_prior28_20260812_v1`.
It contains 2,820,951 unique long OOF rows from February 2024 through July
2026. All 31 base folds use a same-model physical 28-day reference; 30 folds
have complete causal map/residual handoffs (the first is the expected
insufficient-prior-OOF warm-up). Every emitted base/rank/upstream score is
finite and every row is marked prequential.

The already sealed August-1 shadow bundle predates that final ledger patch. Its
held/reference scoring reserve and Cell-day admission reserve are both exactly
28 days, but its historical OOF teacher ledger was normalized against 42-day
same-model references. That teacher ledger remains causal and does not create
lookahead; it is nevertheless a calibration-horizon mismatch. The August bundle
is therefore valid **transitional shadow evidence**, not proof of full 28-day
training/replay parity. The next bundle must be rebuilt from a manifest carrying
`reference_window_days = 28` before production promotion.

Persisted column and file identifiers such as `base_rank42`,
`prequential_base_rank42`, and `same_model_prior42_reference_scores.parquet`
remain **legacy wire aliases**. They are not evidence of a 42-day runtime
window. Renaming them would break frozen model feature order and sealed artifact
hashes; the authoritative physical lineage is the persisted
`reference_window_days = 28` plus the audited half-open timestamps
`[activation - 28 days, activation)`.
An explicit audit of all eight 2026 calibrators verifies that their maximum
`policy_label_available_ts` is exactly one hour before activation in every
case. No activation-time or later outcome enters a promoted map.
Post-score policy outcome attachment is isolated in
[attach_strict_r3_policy_outcomes.py](/Users/remyroche/Documents/Ares/scripts/attach_strict_r3_policy_outcomes.py).
The original 45-field LDF is produced by
[run_strict_r3_ldf_v3_frozen_walkforward.py](/Users/remyroche/Documents/Ares/scripts/run_strict_r3_ldf_v3_frozen_walkforward.py)
and is strictly a bounded 0.25–1.75 post-admission sizing overlay: it does not
change ranking or admission.  The completed matched replay demotes it to a
shadow comparator: unit sizing is no worse in 2025 and is materially safer in
2026.  The LDF artifact remains versioned for research and inference-parity
testing, but must not change live size without a new frozen validation.

The frozen policy is: 12-hour timeout, 100-bps cost charged once, stop
4.1520006 ATR, trailing activation 2.3262249 ATR, and giveback 0.1023720 ATR.

The residual, conversion, admission, A0/A5-target and replay outcome contract is
now materialised on the strict target-free source population in
`data_perp/artifacts/strict_r3_source_aligned_optimized_policy_outcomes_long_2024jul_jul2026_20260812_v1`.
It contains all 2,564,827 July-2024 through July-2026 source identities and
2,531,560 valid selected-policy outcomes (98.70%). July--December 2024 causal
15-minute coverage is 90.37--93.32%; January 2025--July 2026 is complete.
The assembler ignored and audited 48,326 labels belonging to the broader
pre-feature-screen outcome population; those rows cannot expand the scoring
universe. Missing/incomplete paths remain `policy_path_valid=false` and never
become zero-return or negative supervision.

The frozen portfolio contract is explicit and machine-readable in
`config/strict_r3_cell_day_trim15_portfolio_28d_a5_b10_v1.json`: long-only, 7x leverage,
eight concurrent positions, at most two new entries per executable 15-minute
bar, one open position per asset, 10% initial-margin slots, and an 80% total
margin cap. The causal auction admits only the fixed A0-top-15 population and
sorts rows by bounded-A5 expected net, then uses same-producer `final_score`
only to resolve ties. Rows without valid A0, A4, and A5 calibration values are
rejected. The raw 28-day Cell-day estimate remains the causal target anchor.
The auction does not fit a second EV-priority curve on held outcomes. A matched
January–July 2026 comparison against the former implicit curve returned the
identical 2,875 accepted candidate IDs, identical rejection reasons, and
identical +119.204486 net bps/trade; the explicit causal curve therefore
removes a latent outcome dependency without changing the historical result.

### Canonical A5 bounded-10 trust integration

R5 is fitted separately for each producer cutoff on the preceding nine
months of resolved, strict-prequential rows. Training is restricted to the
top 30% within each decision timestamp by `final_score`, sampled equally by
month, and capped at 60,000 rows. Its target is:

```text
clip(policy_net_bps - 28_day_cell_day_expected_net_bps, -500, +500)
```

The persisted Local Distribution Forest Proxy has 64 trees, depth 8, minimum
leaf support 120, feature fraction 0.70, bootstrap fraction 0.75, and seed
20260810. It consumes the frozen ordered 66-field score, agreement, support,
OOD, drift, leaf/path, aggregate K9, and active-rule reliability contract in
`config/strict_r3_cell_day_residual_trust_model_r5_9m_v1.json`, which inherits
the fields and model parameters from
`config/strict_r3_cell_day_residual_trust_overlay_v1.json` but freezes the
validated 9-month field order. Raw K9 membership
IDs are prohibited. Up to eight stable train-only CMI interactions may be
added; the August bundle retained four.

The active schema-v5 conversion trainer also excludes every
October--December 2024 geometry-definition row from supervised Severe-200
fitting. Severe remains a shadow diagnostic and cannot change `final_score`,
but this exclusion is enforced so even diagnostic consumers cannot train on
the representation-definition sample. The correctness learner may use the
frozen aggregate transform because its target is separate and every fit/held
application uses the already-frozen representation; raw K9 identities remain
prohibited.

The A0 anchor is:

```text
posterior_expected_policy_net_bps
    = 28_day_cell_day_expected_net_bps
    + locally_shrunk_expected_residual_bps

A0 = posterior_expected_policy_net_bps
```

The A4 component uses the same ordered 66 target-free fields and forest
parameters, but changes the training contract to the validated independent
local variant: 75% timestamp-local top-30 rows plus 25% lower-reference rows,
uniform mean weights, independent-experience leaf support, and local-leaf
uncertainty. Its target remains the clipped 28-day-map residual. A causal
Huber calibrator is then fitted on earlier OOS A4 predictions only; every
calibration label must resolve strictly before the new bundle cutoff.

The final contract is:

```text
domain = timestamp-local top 15% by pre-trust final_score
admit = finite(A0, calibrated_A4) AND A0 >= +50 bps AND domain
A5_bounded10 = A0 + 0.10 * (calibrated_A4 - A0)
auction order = A5_bounded10 descending, then final_score descending
```

A5 may rerank only; it cannot add or remove rows relative to the fixed
A0-top-15 admission population.

The posterior is estimated only from prior-resolved, strict-prequential rows.
The Cell-day map fields anchor the target but do not enter the model feature
matrix. Missing posterior values fail closed. The older three-gate, 10%-cap
demotion output remains persisted for historical diagnostics only and cannot
silently replace the posterior integration.

The A0 design handoff is
[CELL_DAY_RESIDUAL_TRUST_OVERLAY_20260812.md](/Users/remyroche/Documents/Ares/docs/CELL_DAY_RESIDUAL_TRUST_OVERLAY_20260812.md).
The selected integration contract is
`config/strict_r3_cell_day_residual_trust_posterior_28d_challenger_v1.json`.
Despite the retained filename, its status is
`canonical_executable_research_pending_untouched_forward_validation`.
The August-1 reusable bundle is
`data_perp/artifacts/strict_r3_cell_day_residual_trust_bundle_long_20260801_28d_r5_9m_posterior_v2`.
It trains on 60,000 equal-month rows from 1,049,296 eligible prior-resolved
rows over 2025-11-01 through 2026-08-01, retains the exact ordered 66-field
contract from `config/strict_r3_cell_day_residual_trust_model_r5_9m_v1.json`
and four train-only stable-CMI interactions, and declares posterior
admission ownership in its immutable manifest.

The bounded-A5 integration contract is
`config/strict_r3_a5_bounded_10pct_canonical_v1.json`. The sealed August A4
and A5 calibration bundle is
`data_perp/artifacts/strict_r3_a5_bounded10_trust_bundle_long_20260801_28d_9m_v1`.
It trains A4 on 60,000 equal-month rows from 1,049,296 eligible prior-resolved
rows over 2025-11-01 through 2026-08-01 and calibrates from 120,000 prior OOS
A4 rows. The fitted calibration is slope 0.6316908, intercept -27.2265 bps,
and predictive-SD scale 0.9676419. The bundle uses the same frozen geometry
hash and prohibits raw K9 memberships.

The longer matched validation is documented in
[A5_BOUNDED_LONGER_VALIDATION_20260812.md](/Users/remyroche/Documents/Ares/docs/A5_BOUNDED_LONGER_VALIDATION_20260812.md).
Its earlier recommendation not to promote A5 is superseded by the explicit
promotion decision here. On April 2025--July 2026, bounded A5-10 produced
6,652 constrained trades (13.66/day), +152.35 net bps/trade, Sortino 0.471,
max drawdown -48.53%, and worst week -31.51%. The ungated R5 control produced
6,704 trades (13.77/day) at +149.84 bps/trade. On the matched full-nine-month
October 2025--July 2026 slice, A5-10 was effectively tied with R5 (+162.31
versus +162.41 bps/trade). All ten months were positive and 41/42 weeks were
active; the two drought weeks were shared with A0.

Historical R5 replay must use
`scripts/run_strict_r3_cell_day_residual_trust_walkforward.py`. It fits and
persists one independent nine-month R5 bundle for every upstream/conversion
producer activation, then scores only that producer's held block. The runner
requires the physical 28-day, equal-UTC-day, symmetric-15%-trim map manifest,
one frozen geometry/K9 hash, strict-prequential upstream rows, and exact
candidate/timestamp joins. Held scoring receives only the 66 target-free
fields plus the causal Cell-day anchor; policy outcomes are unavailable to the
score call. An early cutoff without 2,000 resolved prior map rows, or without
a complete nine-calendar-month mapped-history window, is retained with
identical candidate IDs but no posterior, so admission fails closed. A
single static R5 bundle is prohibited for multi-cutoff performance claims.

Canonical A5 replay then uses
`scripts/run_strict_r3_a5_bounded_walkforward.py` on matched A0 and A4 OOS
folds. It refits only the A4-to-policy-net calibrator for each held month from
earlier resolved OOS A4 predictions, applies the fixed A0-top-15 admission,
and emits the bounded-10 auction value. It never consumes the held month's
outcome while constructing score or admission.

After the lock-step score ledger completes, the canonical downstream funnel is
`scripts/run_strict_r3_r5_canonical_postprocessing.py`. It materialises the
authoritative 28-day Cell-day provenance, independently refits R5 at every
producer activation, invokes the current-v5 frozen-geometry portfolio replay,
and reports the layer waterfall under both optimized-policy and exact TP6/SL4
outcomes. The funnel explicitly disables the unmatched historical N5 sizing
sidecar; unit relative sizing remains active until a source-aligned LDF replay
passes its own promotion gate.

`scripts/run_strict_r3_hourly_shadow.py` is the canonical inference-parity
entry point. It defaults to the immutable schema-v5 inference bundle and
requires an explicit
point-in-time wallet/open-position snapshot, composes the
frozen score, 28-day Cell-day map, nine-month A0, A4, and causal A5 calibration producers, and applies the same portfolio
contract through `extreme_price_movements/strict_r3_shadow_portfolio.py`.
The active shadow seal is
`config/strict_r3_inference_bundle_long_20260801_v5_a5_b10.json`.
The historical schema-v3 seal implements the superseded three-month
demotion-only integration; schema v4 is the superseded R5-only posterior
integration. Schema v5 additionally seals A4, the causal A5 calibration, and
the bounded-A5 contract. It verifies 23 immutable inputs. The seal covers upstream and conversion models,
frozen geometry/K9, feature and universe contracts, same-model reference
features, resolved ledger, EV bridge/index, exit policy, portfolio policy, A0,
A4, and A5 model/calibration contracts. It fails closed outside the producer's
declared 2026-08-01 through 2026-08-29-exclusive window. The shadow CLI no longer accepts those component
paths individually, so model/map/policy vintages cannot be mixed at invocation.
There is deliberately no exchange client or order-capable mode. The historical
schema-v3 August 12 snapshot scored and mapped 170 long candidates and accepted
zero under the old Cell-day gate; it is not posterior-admission evidence.

The schema-v5 runner additionally fails closed unless the held snapshot meets
the bundle's frozen base-feature completeness fraction. An attempted reuse of
an August-12 feature materialization with zero complete 120-field rows was
correctly rejected. The canonical hourly orchestrator subsequently rebuilt the
complete 170-asset point-in-time universe and all features from their declared
sources. Seven assets failed the contemporaneous actionability gate; 162 of the
remaining 163 rows were complete on all 120 frozen fields (99.39%, above the
sealed 90% cycle gate), and every individual field also had at least 99.39%
finite coverage. The sealed bundle then completed score, Cell-day mapping, A0,
A4, bounded A5, and portfolio logic in shadow-only mode with all 23 hashes
verified, no held percentiles, no held outcomes, and zero exchange calls. The
09:00 UTC observation admitted no trades: its causal expected-net estimates
were below the +50-bps hurdle. This is valid fail-closed inference-parity
evidence, not economic performance evidence. The upstream orchestration evidence is
`data_perp/artifacts/strict_r3_hourly_shadow_r5_9m_posterior_20260812T090000Z_v3_featurefixed/run_manifest.json`;
the current bounded-A5 downstream receipt is
`data_perp/artifacts/strict_r3_shadow_cycle_28d_a5_b10_20260812T090000Z_v1/run_manifest.json`.

The persisted-bundle parity gate rebuilt the complete prior-28-day reference
and August 1--7 held surface from target-free inputs. It matched 120,252 rows
and all nine score components with maximum absolute delta **0.0**. The held
8,542-row feature surface had all 120 fields on 99.64% of rows, every row met
the 90% row-coverage gate, the weakest field was finite on 99.68%, and no
deployed field was constant. That parity artifact predates posterior promotion:
it validates the shared upstream producer and feature contract, while the
current schema-v5 inference seal independently verifies 23 immutable inputs,
including A0, A4, and the causal A5 calibration.

Schema-v2 additionally enforces two score-time contracts that were only
audited externally in the first snapshot.  First, all 170 frozen-universe
members enter feature generation before any actionability filter, preserving
the complete contemporaneous cross-section.  Second, executable candidates
must have an official Kraken best-bid/best-ask observation at the signal hour
with spread no greater than 100 bps; the historical spread registry defines
membership only and is never substituted for the current spread. Missing
current spreads fail closed.

The retired demotion-only bundle was also exercised for five consecutive hours
and admitted nineteen rows. Those counts do **not** describe the current
pipeline. The authoritative bounded-A5 snapshot has 163 actionable rows, 162
rows complete on all 120 fields, zero A5 admissions and zero portfolio entries
at 09:00 UTC. This is consistent with the fixed A0-top-15 gate: A5 cannot
create an admission when A0 does not clear +50 bps.

The feature-source repair shared by both shadow generations made two changes
without changing a model, geometry/K9 state, EV map, policy, or portfolio rule:

- official public Kraken one-hour trade candles now fill only unavailable
  coarse OHLCV cells; complete downloaded 15-minute bars retain precedence and
  one-minute data remains prohibited;
- the selected `post_liquidation_rebound_score` now declares all causal parent
  fields as generation dependencies, and official funding updates retain the
  existing observation-time + one-hour availability shift.

These runs are valid **shadow evidence**, not production promotion. They sent
zero exchange requests and created no orders. The current bounded-A5 bundle
fails closed outside its declared 2026-08-01 through 2026-08-29-exclusive
window. It uses the fully excluded prior-28-day reserve, nine-month A0/A4
training, and prior-OOS A5 calibration. Production
execution remains disabled; a later untouched period and an explicit
promotion decision are still required.

The historical fail-closed successor preflight made the timing explicit. For an August
13 00:00 UTC cutoff it requires target-free source features through August 12
23:00. Because H12 labels are gated by `label_available_ts < activation`, the
latest usable reserve outcome is the August 12 11:00 decision, available at
23:00; the final twelve decision hours remain scored but do not enter the map.
The existing immutable source and policy stores stop at July 31 23:00 (last
label available August 1 11:00), so the preflight is correctly `ready=false`.
The historical prequential ledger already reaches beyond the July 2 reserve
boundary and was not the blocker. That preflight is retained as historical
evidence; the schema-v3 August-1 successor was built and sealed separately.

### Archived pre-R5 schema-v5 causal replay

The tables in this subsection describe the unit-size Cell-day stack before R5
posterior admission was promoted. They remain useful upstream/control evidence
but are not current R5 performance claims. The independently refitted,
full-nine-month R5 matched waterfall is the only artifact allowed to supersede
them; while that regeneration is pending, use the R5 handover's October
2025–July 2026 figures as selection/backward-confirmation evidence only.

| Period | Scored candidates | Policy-valid rows | Global diagnostic Top 0.5% | Top 1% | Top 2% | Top 3% | Top 5% |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2025 Jan–Jul | 757,733 | 757,733 | +229.93 | +204.20 | +168.56 | +146.95 | +117.44 |
| 2026 Jan–Jul | 843,935 | 843,869 | +165.24 | +120.38 | +84.06 | +69.40 | +45.19 |

Values are policy-net bps/trade after cost.  Global tails remain an
**offline ranking diagnostic**, not a live entry rule.  Live admission uses
the causal exact-producer Cell-day trim 15% expected-net map above.

| Period | Mapped candidates | Admitted candidates | Portfolio trades | Trades/day | Portfolio net bps/trade | Positive weeks | Negative weeks |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2025 Jan–Jul | 757,733 | — | 4,020 | 18.96 | +151.87 | 30 / 32 | 2 / 32 |
| 2026 Jan–Jul | 843,935 | — | 2,875 | 13.56 | +119.20 | 22 / 24 | 2 / 24 |

These were the authoritative pre-R5 Cell-day admission portfolio metrics as of
2026-08-12.  The 2025 figure combines the matched January–March and April–July
blocks.  They are long-only, unit-relative-size constrained replays using the
frozen policy, not live capital forecasts.  Exact-reserve rollback metrics over
the same blocks are 3,028 trades at +158.20 bps/trade for 2025 and 1,374 trades
at +138.46 bps/trade for 2026.  Cell-day admission deliberately accepts more
trades and lower EV/trade in exchange for greater total opportunity capture.

The 2026 positive per-trade result is not by itself sufficient for live
approval.  Absolute drawdown is **not** a rejection criterion: it must be
assessed jointly with return-to-drawdown, Sortino, worst-period loss, and the
chosen leverage/margin policy. At a fixed policy, per-trade EV is unchanged by
wallet scale; leverage and margin can increase profit and drawdown together.
The decision must therefore use a predeclared risk objective rather than an
absolute-drawdown cutoff. A matched 2026 unit-sizing diagnostic gives:

| Margin slot | Compounded return / max DD | Sortino | Worst week | Mean margin use |
|---:|---:|---:|---:|---:|
| 10% | 336,267 | 0.281 | -24.6% | 78.0% |
| 5% | 3,114 | 0.334 | -3.1% | 40.1% |
| 2.5% | 193 | 0.353 | -1.5% | 20.1% |

The high-leverage arm earns more and has the strongest compounded
return-to-drawdown ratio; lower sizing improves scale-independent downside
stability.  These compounded figures are not a live capital forecast.  Final
live approval requires a declared risk objective and an inference-parity dry
run, not a fixed maximum-drawdown cutoff.

Matched LDF-sizing ablation (identical score, admission, policy, and
portfolio constraints):

| Period | Relative sizing | Net bps/trade | Max drawdown |
|---|---|---:|---:|
| 2025 Jan–Jul | unit **(canonical)** | +164.31 | -66.7% |
| 2025 Jan–Jul | LDF 45-field overlay | +164.18 | -79.8% |
| 2026 Jan–Jul | unit **(canonical)** | +106.31 | -70.4% |
| 2026 Jan–Jul | LDF 45-field overlay | +99.08 | -93.7% |

Immutable schema-v5 evidence:

- 2025 score/replay: `data_perp/artifacts/strict_r3_current_v6_frozen45_statealias_long_2025_janjul_20260811_v1`, `data_perp/artifacts/strict_r3_current_v6_frozen45_ldf45_portfolio_long_2025_janjul_20260811_v1`
- 2026 score/replay: `data_perp/artifacts/strict_r3_current_v6_frozen45_long_2026_janjul_20260811_v1`, `data_perp/artifacts/strict_r3_current_v6_frozen45_ldf45_portfolio_long_2026_janjul_20260811_v1`
- frozen geometry: `data_perp/artifacts/strict_r3_schema_v2_geometry_k9_long_octdec2024_k9weighted_20260811_v1`
- exact-producer lineage and admission replay: `data_perp/artifacts/strict_r3_current_v6_frozen45_full_policy_long_{2025,2026}_janjul_20260811_v1/{producer_lineage_v1,full_producer_vintage_unit_portfolio_replay_v1}`
- cross-producer transport audits: `data_perp/artifacts/strict_r3_current_v6_frozen45_full_policy_long_{2025,2026}_janjul_20260811_v1/ev_map_transport_audit_v1`
- promoted admission config: `config/strict_r3_cell_day_trim15_admission_28d_v1.json`
- promoted inference implementation: `extreme_price_movements/strict_r3_cell_day_admission.py`
- trust contract and implementation: `config/strict_r3_cell_day_residual_trust_overlay_v1.json`, `extreme_price_movements/strict_r3_cell_day_trust.py`, `scripts/fit_strict_r3_cell_day_residual_trust.py`
- canonical per-refit posterior replay producer: `scripts/run_strict_r3_cell_day_residual_trust_walkforward.py`
- canonical downstream replay/report funnel: `scripts/run_strict_r3_r5_canonical_postprocessing.py`, `scripts/report_strict_r3_r5_canonical_waterfall.py`
- canonical posterior-integration config: `config/strict_r3_cell_day_residual_trust_posterior_28d_challenger_v1.json`
- canonical posterior portfolio config: `config/strict_r3_cell_day_trim15_portfolio_28d_r5_9m_posterior_v1.json`
- canonical bounded-A5 contract and portfolio: `config/strict_r3_a5_bounded_10pct_canonical_v1.json`, `config/strict_r3_cell_day_trim15_portfolio_28d_a5_b10_v1.json`
- executable A5 implementation and bundle fitter: `extreme_price_movements/strict_r3_a5_trust.py`, `scripts/fit_strict_r3_a5_bounded_trust.py`
- sealed bounded-A5 inference bundle: `config/strict_r3_inference_bundle_long_20260801_v5_a5_b10.json`
- exchange-free shadow auction: `extreme_price_movements/strict_r3_shadow_portfolio.py`, `scripts/run_strict_r3_shadow_cycle.py`
- canonical one-hour orchestrator: `scripts/run_strict_r3_hourly_shadow.py`
- consecutive-hour deterministic auditor: `scripts/audit_strict_r3_multi_hour_shadow.py`
- successor-bundle preflight: `scripts/audit_strict_r3_successor_bundle_readiness.py`, `data_perp/artifacts/strict_r3_successor_bundle_readiness_20260813_v2.json`
- machine-readable R5 9-month posterior readiness receipt: `config/strict_r3_live_readiness_receipt_28d_r5_20260812.json` (receipt schema v3; execution remains disabled)
- historical demotion-only schema-v3 bundle: `config/strict_r3_inference_bundle_long_20260801_v3.json`
- schema-v4 posterior validator and shadow runtime: `extreme_price_movements/strict_r3_inference_bundle.py`, `scripts/validate_strict_r3_inference_bundle.py`, `scripts/run_strict_r3_shadow_cycle.py`
- exact 28-day successor/parity: `data_perp/artifacts/strict_r3_lockstep_successor28_long_aug1_7_current_spread_20260812_v1`, `data_perp/artifacts/strict_r3_lockstep_successor28_parity_long_aug1_7_20260812_v1`
- immediate 28-day calibrator and canonical R5 bundle: `data_perp/artifacts/strict_r3_immediate_calibration_successor28_long_aug1_20260812_v2`, `data_perp/artifacts/strict_r3_cell_day_residual_trust_bundle_long_20260801_28d_r5_9m_posterior_v2`
- current schema-v4 posterior end-to-end shadow dry run: `data_perp/artifacts/strict_r3_hourly_shadow_r5_9m_posterior_20260812T090000Z_v3_featurefixed`
- current schema-v5 bounded-A5 end-to-end shadow dry run: `data_perp/artifacts/strict_r3_shadow_cycle_28d_a5_b10_20260812T090000Z_v1`
- verified score/admission/portfolio snapshot: `data_perp/artifacts/strict_r3_cell_day_trim15_sealed_shadow_cycle_20260801T000000Z_20260812_v1`
- current-spread-gated one-hour snapshots: `data_perp/artifacts/strict_r3_hourly_shadow_20260812T090000Z_v1`, final fail-before-score code in `data_perp/artifacts/strict_r3_hourly_shadow_20260812T090000Z_v2`
- deterministic five-hour snapshots: `data_perp/artifacts/strict_r3_multi_hour_shadow_20260812T0500_1000Z_v1`, repeated identically in `data_perp/artifacts/strict_r3_multi_hour_shadow_20260812T0500_1000Z_v2`
- untouched target-free candidates and final causal features: `data_perp/artifacts/strict_r3_untouched_shadow_20260812T090000Z_v4_candidate_grid`, `data_perp/artifacts/strict_r3_untouched_shadow_20260812T090000Z_v5_features`
- exact-universe funding refresh audit: `data_perp/artifacts/strict_r3_untouched_funding_refresh_exact170_20260812T090000Z_v1`
- promoted 2026 calibrators: `data_perp/artifacts/strict_r3_cell_day_trim15_canonical_maps_long_2026_janjul_20260812_v1`
- calibrator causality audit: `data_perp/artifacts/strict_r3_cell_day_trim15_canonical_map_causality_audit_20260812_v1.json`
- matched map/portfolio evidence: `data_perp/artifacts/strict_r3_cell_day_bayesian_ev_map_ablation_long_{2025_janmar,2025_aprjul,2026_janjul}_20260812_v2`, `data_perp/artifacts/strict_r3_cell_day_bayesian_portfolio_long_{2025_janmar,2025_aprjul,2026_janjul}_cell_day_trim_15pct_20260812_v1`

### EV-map producer-homogeneity decision

The score's 0–1 CDF range is not sufficient to pool realised policy outcomes
across fitted producers.  Strict OOF transport tests rejected simple
cross-producer 42-day pooling in both 2025 (7/13 transitions passed) and 2026
(8/13 passed).  In the 2026 failures, prior top-5 net economics of about
+51 to +66 bps became −39, −12, +0, or −2 bps under the new producer.
The current executable admission is therefore
`strict_oof_exact_producer_cell_day_trim15_28d_v1`: each producer remains
isolated, but its own excluded reserve makes the map immediately usable.

Future work may introduce a cross-producer bridge only after a
separately frozen causal standardisation layer passes these transport gates;
raw ledger concatenation is prohibited.

### July 2026 drought diagnosis

The July drought is not missing data, score failure, or an auction bottleneck:

- all 123,225 July candidates have finite scores and finite Cell-day maps;
- the map admits only one raw candidate, on July 31; the auction accepts it;
- July uses two exact producers: the June-18 producer through July 15 and the
  July-16 producer thereafter.  There is no cross-producer score or outcome
  pooling;
- the first-half raw ranking was strong: Top-0.5/1/2/5% policy-net was
  +166.56/+143.43/+116.70/+51.98 bps.  Its highest fixed score cell averaged
  +120.74 bps per equal-weight July day, with 12 of 14 days positive.  The
  causal map nevertheless started from strongly negative May–June evidence:
  its frozen reserve's highest cell was only +11.57 trimmed bps, while cells
  17 and 18 were −75.24 and −49.71 bps.  Maximum mapped EV therefore recovered
  only from −20.04 bps on July 1 to +27.88 on July 15, below the unchanged
  +50-bps hurdle. This is legacy 42-day diagnostic evidence; the 28-day
  contract supersedes it for future producers;
- the July-16 refit did not create a cold start: its complete same-model reserve
  covered June 4–July 15.  However, the new reserve's highest cell was still
  only +23.60 trimmed bps and the two adjacent high cells were −66.63 and
  −36.42 bps.  The producer was immediately calibrated, but not economically
  admissible;
- the second-half raw ranking then weakened materially outside the extreme
  tip: Top-0.5/1/2/5% was +10.54/−16.24/−10.75/−55.65 bps.  Top-cell daily
  outcomes of −454.41 bps on July 21 and −287.39 bps on July 23 delayed the
  recovery despite several later positive days;
- maximum mapped EV reached +45.15 bps on July 29 and +51.77 on July 31,
  crossing the hurdle only then.  The single admitted REZ trade returned
  +321.38 net bps under the frozen policy;
- over the complete month, which combines both producers, raw Top-0.5/1/2/5%
  was +130.04/+88.08/+68.64/+4.74 bps.  These pooled hindsight diagnostics do
  not imply that a causal July-1 map could know the subsequent rebound.

The legacy drought was therefore an economically conservative calibration lag after
the June shock, followed by a genuine second-half ranking deterioration.  It
is not a technical cold start, missing-score problem, or portfolio-capacity
failure.  The map's equal-day construction behaved as declared, but the legacy 42-day
window plus a +50-bps hurdle necessarily adapts slowly when a profitable
rebound immediately follows a broad negative regime.  Bayesian k7/p90 provides
266 July portfolio trades at +33.46 bps, but its materially weaker aggregate
Sortino and worst-week behavior prevent its promotion.  Do not lower the
canonical floor solely to force July activity; treat July opportunity recall
as an explicit shadow diagnostic for the next untouched period.

---

**Archived status (rolling-geometry schema-v4):** the entire section between
this notice and “Part B” is retained for research lineage only. It is
superseded by the schema-v5 frozen-geometry + 28-day Cell-day + R5 9-month
posterior contract above. Nothing in this archived section is an active
admission, sizing, geometry, or inference default.

**Scope:** long side only.

**Current matched evaluation:** January–July 2025 development and January–July 2026 confirmation.

**Updated:** 2026-08-12.
**Promotion status:** not yet production-approved. August and the preceding periods influenced development; promotion requires a later frozen period and an end-to-end inference/execution dry run. This is not an absolute-drawdown rejection.

The archived stack was the repaired C3 pipeline in this section with the
**conditional-usefulness ten-head consensus promoted to the primary upstream
score, the correctness curriculum restricted to the pooled-global upstream
top 30%, and the K9 membership temperature frozen at 0.25 times its
training-derived value, followed after causal admission by N5 single-forest
support-aware relative sizing**.  The former ordinary/equal cap consensus remains a shadow rollback
score and cannot drive admission.  Severe-200 is also shadow-only, raw K9
memberships are excluded from consensus and correctness, and live admission
uses the causal 21/42/84-day hierarchical tail map.  The older schema-v2
frozen-geometry stack is retained in Part B solely for historical replay and
lineage.  Where the two parts conflict, Part A is authoritative.

# Archived Part A — rolling-C3 research stack (not canonical)

## A1. Architecture

```text
target-free point-in-time long candidates
→ strict-R3 base with D2 top-20 robust-clear curriculum at 1.5x
→ same-model prior-42-day base rank
→ prior-prequential selected-policy net map
→ ten conditional-usefulness policy-residual LambdaRank heads
→ 75% base rank + 25% consensus rank
→ one C3 rolling three-month geometry/K9 bundle per downstream fit
→ frozen K9 temperature scale 0.25 for aggregate state
→ current-base leaf/path state
→ TP6/SL4 Severe-200 probability as a shadow diagnostic only
→ +100-bps policy-residual correctness ranker trained on the pooled-global
  training top 30%, without raw K9 memberships
→ same-model prior-42-day CDF
→ causal prior-resolved 21/42/84-day hierarchical tail admission at +50 bps
→ N5 single-forest expected-net/support/risk shrinkage
→ bounded 0.25–1.75 relative size multiplier
→ constrained long-only portfolio auction
```

The upstream and conversion cadences are deliberately different:

- the D2 base, policy map, and conditional consensus are refit on UTC calendar-month cutoffs;
- the C3 geometry/leaf/correctness bundle uses a six-month supervised window and is refit every four weeks;
- the rolling raw-market K9 uses the three months immediately before the supervised window, is aligned to the preceding C3 bundle, and freezes `effective_temperature = training_temperature × 0.25`;
- one conversion fit uses one geometry bundle for its training, prior-42-day reference, and held rows;
- prior-42-day normalization applies the same conversion model to causal monthly upstream scores; it never ranks within the held window.

A two-week conversion cadence won the 2025 screen but failed the 2026
worst-month transport gate (−157.20 bps and −137.61 portability).  The
four-week cadence remained positive in all seven 2026 Top-2 months and is
canonical.

The correctness and Severe models are both refit at every four-week cutoff
using only labels resolved before that cutoff.  The matched overlay ablation
showed that periodic refitting did not repair Severe transport: the 2025
alpha-0.5 winner fell to five positive months and a negative 2026 worst month.
Raw K9 memberships also reduced Top-2 EV in both years.  The live score
therefore uses the periodically refit correctness ranker with aggregate
geometry/leaf state but no cluster membership vector.  Severe-200 remains
frozen to the exact H12 TP6/SL4 net-loss event at or below −200 bps and is
persisted only for monitoring.

The base curriculum changes only training weights: R3 robust-clear samples in
the prior strict-prequential teacher's global top 20% receive a 1.5x boost
before weights are projected to mean one and capped to `[0.25, 4]`.

The monthly base is a 120-field LightGBM multiclass classifier with 220 trees,
learning rate 0.035, depth 5, 24 leaves, minimum child support 2,400, feature
fraction 0.85, and L2 20. It fits at most the latest 240,000 rows whose R3
labels have resolved before the monthly cutoff. Its live score is
`P(clear) - 0.5 × P(adverse)`; `P(weak)` is retained in the decomposition.

All ten canonical residual heads use the same ordinal selected-policy residual
target with boundaries `[-150, -50, +50, +150]` bps.  Their frozen diversity
comes from query geometry, month weighting, and train-only MDA subsets:

| Head | Query | Weighting | Fields |
|---|---|---|---:|
| cap40 ordinary | exact timestamp × side | ordinary | 40 |
| cap40 equal-month | exact timestamp × side | equal month | 40 |
| cap60 ordinary | 4-hour × side | ordinary | 15 |
| cap60 equal-month | exact timestamp × side | equal month | 30 |
| cap80 ordinary | exact timestamp × side | ordinary | 80 |
| cap80 equal-month | 4-hour × side | equal month | 80 |
| cap100 ordinary | exact timestamp × side | ordinary | 100 |
| cap100 equal-month | 4-hour × side | equal month | 100 |
| cap120 ordinary | 4-hour × side | ordinary | 120 |
| cap120 equal-month | exact timestamp × side | equal month | 51 |

Each head is capped at 240,000 rows by sampling **complete query groups**. The
cap never splits a timestamp-side or four-hour-side query. Equal-month heads
allocate the cap across months before sampling whole groups; ordinary heads
sample whole groups over the pooled training population. Each head's causal
rank reference is its sampled, resolved, prequential training-score
distribution.

All ten heads retain the frozen LightGBM LambdaRank defaults because no HPO
challenger passed the conditional downstream gate: 120 trees, learning rate
0.035, depth 5, 31 leaves, minimum child support 300, feature and bagging
fractions 0.82, L1 0.02, L2 2.0, max-bin 127, truncation 10, and gains
`[0, 0.25, 1, 3, 7]`.

The median of these ten train-distribution ranks is the canonical consensus.
The old ten ordinary/equal cap heads are still fitted and emitted as
`ordinary_shadow_*`, but they are excluded from the canonical blend,
correctness score, admission map, and portfolio auction.

The four-week correctness head asks whether selected-policy realised net minus
the causal base anchor exceeds +100 bps. It is a 4-hour × long-side LambdaRank
model trained on six resolved months and capped at 240,000 equal-month sampled
rows: 120 trees, learning rate 0.035, depth 4, 15 leaves, minimum child support
`max(120, 3% of fit rows)`, feature fraction 0.80, bagging fraction 0.82, L1
0.05, L2 5, max-bin 127, binary gains `[0, 1]`, and truncation 10. Inputs are
the base score/rank/anchor, conditional consensus, 75/25 upstream blend,
current-base leaf support/OOD summaries, and aggregate C3 state. Raw K9
membership, negative-distance, and confidence vectors are explicitly filtered
out.

After equal-month capping, the trainer computes the 70th percentile of the
**training** upstream score and fits correctness only on rows at or above that
frozen scalar. Reference and held rows never redefine the gate. At scoring,
the correctness multiplier applies only when `upstream >= training_score_floor`;
all other rows receive multiplier 1.0. The fitted floor, retained fraction,
K9 scale and effective temperature are persisted and validated by schema v4.

## A2. Selected execution policy

The SimplePolicyOptimiser winner was selected only on strict-prequential pre-2025 data:

| Parameter | Value |
|---|---:|
| Stop | 4.1520006 ATR |
| Trailing activation | 2.3262249 ATR |
| Giveback | 0.1023720 ATR |
| Timeout | 12 hours |
| Cost | 100 bps exactly once |

This selected policy—not the older hard-coded SL3/activation0.5/giveback0.25 comparator—defines the current policy map, residual target, causal admission outcomes, and portfolio replay.

Immutable policy contract:
`data_perp/artifacts/strict_r3_schema_v2_simple_policy_targetfree_long_pre2025_20260809_v3/winner.json`.

## A3. Current repaired full-cap results

The table below is the current matched long-only replay.  Every scored
candidate enters the global-tail denominator before outcome availability is
examined.  January–July 2025 optimized-policy coverage is repaired from 82.7%
to 100% by preserving exact/15-minute outcomes first and using an explicitly
labelled hourly OHLC proxy only for missing paths.  The same repair yields
effectively complete 2026 coverage.

| Year | Top 0.5% | Top 1% | Top 2% | Top 5% | Top 10% | Top-2 portability | Top-2 worst month | Positive Top-2 months |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2025 | **+213.55** | **+179.05** | **+164.92** | **+120.26** | **+55.32** | **+151.28** | **+92.07** | 7/7 |
| 2026 | **+189.33** | **+160.25** | **+112.14** | **+51.71** | **+5.66** | **+112.32** | **+33.20** | 7/7 |

All values are optimized-policy net bps/trade after the fixed 100-bps cost.
The schema-v4 numbers use the top-30 correctness curriculum and K9 temperature
scale 0.25 at full caps. The complete scored population enters each pooled
tail before outcomes are inspected. The selection evidence and exact immutable
artifacts are in
`docs/STRICT_R3_TOPTAIL_RELIABILITY_GEOMETRY_ABLATION_20260810.md`.

### Schema-v4 promotion versus the former all-row correctness contract

| Year | Contract | Top 1% | Top 2% | Top 5% | Top-2 portability | Worst month |
|---:|---|---:|---:|---:|---:|---:|
| 2025 | Former schema-v3 all-row correctness | +147.35 | +123.77 | +88.02 | +118.18 | +66.93 |
| 2025 | **Schema-v4 top-30 + K9 0.25** | **+179.05** | **+164.92** | **+120.26** | **+151.28** | **+92.07** |
| 2026 | Former schema-v3 all-row correctness | +133.73 | +93.72 | +42.11 | +101.94 | +32.30 |
| 2026 | **Schema-v4 top-30 + K9 0.25** | **+160.25** | **+112.14** | **+51.71** | **+112.32** | **+33.20** |

The promoted contract improves average net EV per trade at Top-1/2/5 in both
years, improves portability in both years, improves each confirmation-era
worst month, and retains 7/7 positive Top-2 months. These were the promotion
criteria. The 2026 Top-10 result falls by 5.71 bps because the reliability
head is deliberately optimized for the tradable upstream tail; Top-10 is
retained as a breadth diagnostic rather than a promotion objective.

Selection was lexicographic rather than based on the largest pooled headline:
first retain positive cross-era transport and month coverage, then maximize
the minimum year-level Top-2 portability and worst-month protection, and only
then compare average per-trade EV at Top-1/2/5. On that contract, top-30 plus
K9 0.25 beats the former all-row head in every promoted metric. The much more
aggressive downside demoter was rejected despite larger pooled tails because
its 2026 portability and worst month deteriorated.

Two stronger-looking tip-only arms remain shadow research outputs. Adding the
full recent/covariance/cross-model reliability block raised 2026 Top-1 from
+141.01 to +145.65 bps in the screen but reduced Top-2 portability from
+118.15 to +117.56. The policy-residual loss `<= -200 bps`, alpha-0.75 demoter
raised 2026 Top-1/2/5 to +185.37/+133.43/+66.73 bps, but reduced portability
from +118.15 to +110.53 and the worst month from +22.31 to +13.43. Neither is
canonical because confirmation-era portability and worst-month protection are
explicit gates, not tie-breakers that pooled tail EV may override.

### Historical upstream consensus selection versus ordinary shadow

| Year | Stack | Top 1% | Top 2% | Top 5% | Top 10% | Top-2 portability | Worst Top-2 month |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2025 | Ordinary shadow | +127.38 | +108.38 | +74.96 | +40.44 | +107.44 | +56.38 |
| 2025 | **Conditional canonical** | **+147.35** | **+123.77** | **+88.02** | **+52.06** | **+118.18** | **+66.93** |
| 2026 | Ordinary shadow | +129.38 | +88.99 | +39.84 | +6.59 | +76.98 | **+44.08** |
| 2026 | **Conditional canonical** | **+133.73** | **+93.72** | **+42.11** | **+11.37** | **+101.94** | +32.30 |

This matched upstream ablation used the former all-row conversion so that only
the consensus source changed. The conditional contract remains canonical
inside schema v4 because it improves every declared
Top-1/2/5/10 pooled tail, improves portability in both years, and retains 7/7
positive Top-2 months.  The 2026 ordinary shadow still has the better worst
Top-2 month and slightly better portfolio EV/drawdown, so it remains a
versioned rollback comparator rather than being deleted.

A matched K9 input screen found that feeding entropy/margin/OOD summaries, all nine
soft memberships, and a train-only conditional-MI three-membership subset all
reduced Top-2 portability when added directly to the consensus heads. K9 is
therefore not fed directly to those heads. Schema v4 instead sharpens the
aggregate K9 state used downstream by correctness; it does not reintroduce
the raw membership vector. Details are in
`docs/TEN_HEAD_K9_CONSENSUS_INTEGRATION_20260810.md`.

### Earlier matched D2 versus D0 evidence

All tails are selected once from one pooled global long-side ranking. They are diagnostic, not live admission thresholds.

| Year | Arm | Top 0.5% | Top 1% | Top 2% | Top 5% | Top 10% |
|---:|---|---:|---:|---:|---:|---:|
| 2025 | Matched D0 | **+192.06** | +160.46 | +131.61 | +78.43 | +35.25 |
| 2025 | D2 base + D0 residual | +170.21 | **+164.27** | **+137.97** | **+88.92** | **+45.05** |
| 2026 | Matched D0 | **+116.45** | +85.60 | +68.77 | +26.50 | −0.02 |
| 2026 | D2 base + D0 residual | +111.66 | **+91.10** | **+78.11** | **+35.61** | **+2.76** |

These values use the causal score denominator: the pooled global tail is selected from every finite-score candidate before outcome coverage is inspected. EV is then computed only on selected rows with valid outcomes. Outcome coverage is 60.6–75.2% for D0 and 60.7–74.3% for D2 across the reported 2025 tails; in 2026 it is 86.4–99.0% for D0 and 85.6–98.7% for D2. The lower 2025 coverage is a reliability limitation, not an admission filter.

Top-2 stability also improves on the matched producer:

| Year | Arm | Portability | Worst month | Positive months |
|---:|---|---:|---:|---:|
| 2025 | Matched D0 | **+108.71** | +65.66 | 7/7 |
| 2025 | D2 base | +107.89 | **+68.06** | 7/7 |
| 2026 | Matched D0 | +61.97 | +20.72 | 7/7 |
| 2026 | D2 base | **+78.37** | **+33.45** | 7/7 |

The paired day bootstrap supports the broader tails. D2-minus-D0 top-2 mean is +6.42 bps in 2025 with 95% interval `[−10.72, +21.70]`; the later 2026 result is +9.58 bps with `[+0.32, +20.78]`. Top-1 intervals still cross zero.

## A4. Causal admission and portfolio

| Year | Conversion contract | Trades | Trades/day | Net bps/trade | Positive rate | Max drawdown |
|---:|---|---:|---:|---:|---:|---:|
| 2025 | Former schema-v3 all-row | 3,336 | 15.74 | +145.80 | 64.66% | −79.88% |
| 2025 | **Schema-v4 top-30 + K9 0.25** | **3,713** | **17.51** | **+158.08** | **65.36%** | **−72.91%** |
| 2026 | Former schema-v3 all-row | 2,185 | 10.31 | +135.16 | 59.77% | −68.01% |
| 2026 | **Schema-v4 top-30 + K9 0.25** | **2,782** | **13.12** | **+151.30** | **61.04%** | **−59.47%** |

The repaired live map keeps the 21-day causal response but uses uneven,
fine-grained rank bins in the upper tail and shrinks their conditional net
estimates toward 42- and 84-day side-local parents.  Every evaluation starts
with at least 84 prior scored days, so January no longer has an artificial
empty reference.  It admits only mapped net EV at or above +50 bps and fails
closed without support.  In the matched ordinary-control map-selection audit,
this design raised Top-2 admission recall from 76.1% to 78.4% in 2025 and from
43.2% to 57.6% in 2026; the table above is the separate conditional-consensus
end-to-end replay using that frozen map design.

The hierarchical map still admits only mapped net EV at or above +50 bps and
fails closed without support. Hourly-proxy rows are reported separately and
never determine candidate identity, score, or admission. Schema v4 improves
portfolio EV, participation and drawdown in both matched years; nevertheless,
the absolute drawdowns remain too large for production approval.

Portfolio selection no longer filters on future path availability.  An
admitted candidate without a realised path reserves a conservative H12 slot
and is reported as outcome-unavailable rather than replaced using future
knowledge.  No such unavailable row was accepted in these final replays.

The research stack still does not advance to live trading. The schema-v4 2025
and 2026 maximum drawdowns remain −72.9% and −59.5%, respectively. The ranking
and admission repairs advance; leverage/allocation and correlated-exposure
control must be repaired separately.

### A4.1 Canonical Local Distribution Forest Proxy (LDF) sizing

LDF is now the canonical research sizing overlay. It is intentionally placed
**after** the causal EV admission map. It cannot change candidate identity,
`final_score`, pooled-global ranking, or the +50-bps admission decision. It may
only change the relative notional of an already admitted trade.

The executable contract is again the exact original arm with legacy artifact
ID `N5_drf_support_l110_meanrisk`; its public model name is **Local
Distribution Forest Proxy (LDF)**. Trial 8 from the later two-forest HPO is a
challenger, not the incumbent, because it did not beat this arm on the full
matched replay:

- three resolved months, equal-month subsampled to at most 60,000 rows;
- training restricted by a train-only scalar to the upstream Top 30%;
- direct selected-policy `policy_net_bps` mean target;
- one 64-tree `RandomForestRegressor`, depth 8, minimum leaf 120, 70% features,
  75% bootstrap row sample;
- stable conditional-MI interactions with rank-and-loss weighting;
- local leaf-support shrinkage toward a 10-bin train-only parent map with
  support prior 300;
- authority clipped to `[0, 1.10]` around the causal mapped EV;
- risk from cross-tree dispersion plus the frozen training residual scale;
- sizing quality `max(mean,0)/(mean² + predictive_sd²)`, converted through the
  training quality CDF to a bounded `[0.25, 1.75]` relative multiplier.

Its 45-field contract contains base/consensus/correctness agreement, causal
3/7/14-day recent residual and failure-rate state, covariance/correlation
breaks, leaf support/OOD, and bundle-invariant K9 entropy/margin/OOD/support.
Raw `k09__cluster_*` memberships are not passed to this pooled forest because
the slot meanings change between rolling Geometry/K9 bundles.

#### Matched tail evidence

| Year | Arm | Top 1% | Top 2% | Top 5% | Top 10% | Top-1/2 positive months | Top-5 positive months |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2025 | Unit-size score control | +179.05 | +164.92 | +120.26 | +55.32 | 7/7 | 7/7 |
| 2025 | **Canonical LDF / original N5** | **+185.61** | **+172.07** | **+126.77** | **+63.79** | **7/7** | **7/7** |
| 2026 | Unit-size score control | +160.25 | +112.14 | +51.71 | +5.66 | 7/7 | 6/7 |
| 2026 | **Canonical LDF / original N5** | **+165.40** | **+116.86** | **+55.40** | **+8.57** | **7/7** | **6/7** |

Worst-month exposure-weighted net EV is also stronger at the operational
Top-1/2/5 tails:

| Year | Arm | Worst Top 1% | Worst Top 2% | Worst Top 5% |
|---:|---|---:|---:|---:|
| 2025 | Unit-size score control | +133.21 | +88.99 | +59.42 |
| 2025 | N4 raw-forest baseline | +134.94 | +91.91 | +60.83 |
| 2025 | Trial-8 HPO challenger | +133.04 | +89.28 | +59.67 |
| 2025 | **Canonical LDF / original N5** | **+137.43** | **+93.50** | **+62.34** |
| 2026 | Unit-size score control | +81.69 | +37.53 | −19.36 |
| 2026 | N4 raw-forest baseline | **+82.90** | +38.45 | −19.27 |
| 2026 | Trial-8 HPO challenger | **+82.66** | **+38.71** | −18.12 |
| 2026 | **Canonical LDF / original N5** | +82.66 | +38.64 | **−17.94** |

For canonical LDF, the 2025 Top-1/2/5 worst month is May. In 2026, July is
worst at Top-1/2 and June is worst at Top-5.

Values are exposure-weighted selected-policy net bps/trade. Candidate ranking
is unchanged, so these differences measure sizing quality rather than a new
trade selector. N5 transports positively to 2026 at every reported tail. Its
2026 Top-5 worst month is June at −17.94 bps; this remains a known gap.

#### Portfolio risk and the −78.9% drawdown

The earlier nonlinear-funnel equal arm mapped a constant quality to the upper
1.75 size bound, so its reported −82.1% drawdown was **not** a true unit-size
control. The matched audit below replays both arms with the same admission,
auction, 80% margin cap and 7x leverage, while the control uses exactly 1.0
relative size.

| Year | Arm | Trades/day | Net bps/trade | Compounded return | MaxDD | Daily Sharpe | Daily Sortino | Omega | Log-Calmar | Profit factor | Ulcer index |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2025 | Unit size | 17.63 | +158.08 | 2.24e14× | −72.91% | 13.28 | 21.10 | 8.74 | 78.39 | 1.57 | 13.22 |
| 2025 | **N5** | 16.09 | +150.11 | **2.89e16×** | −83.59% | 12.91 | **23.81** | **9.53** | **78.43** | **1.59** | 14.91 |
| 2026 | Unit size | 13.11 | +151.30 | 1.05e11× | −59.47% | **12.85** | **21.55** | **11.44** | **73.18** | **1.55** | **7.65** |
| 2026 | **N5** | 11.87 | +149.94 | **2.13e12×** | −78.94% | 11.01 | 17.73 | 9.21 | 61.53 | 1.31 | 20.31 |

The raw compounded-return/MaxDD ratio is extremely high for N5—approximately
`3.46e16` in 2025 and `2.69e12` in 2026—because the replay repeatedly compounds
7x leveraged exposure. It confirms that N5 converts additional drawdown into
much greater terminal PnL, but it is not a sufficient risk statistic by itself.
The scale-stable diagnostics are mixed: 2025 Sortino, Omega, log-Calmar and
profit factor are strong and broadly equal to or better than control; in 2026
they remain high in absolute terms but are all lower than unit size, while the
ulcer index and worst week are worse. N5 was therefore selected for research
sizing inside this archived experiment, but the evidence did **not** justify production approval at the
current leverage/allocation settings.

#### Portable MDA and joint target/parameter HPO

A custom N5 MDA was run on the longest compatible causal span, October 2024
through July 2025, using three chronological blocks, equal-month subsampling,
and held permutations within month × frozen-score decile. Its loss is the
degradation in exposure-weighted Top-1/2/5 policy EV with median, MAD,
worst-fold and recurrence penalties.

The 12-field compact contract selected by MDA was rejected by its mandatory
unreduced control. On full 2025 replay it produced +180.87/+166.24/+121.10 bps
at Top-1/2/5, below the original N5's +185.61/+172.07/+126.77. The full
45-field contract therefore remained selected inside this archived arm; feature selection is allowed
to select no reduction.

The HPO jointly varied the mean target (`policy_net`, parent residual,
winsorized net), the OOB risk target (squared, downside, absolute error), and
trees/depth/leaf support/feature and row subsampling/support prior/authority.
The subsampled winner was 128 trees, depth 6, leaf 240, 55% features, 65% rows,
support prior 450, direct policy-net mean and OOB squared risk. It did not pass
the full development promotion gate: full 2025 Top-1/2/5 was
+182.63/+167.97/+122.62, again below original N5. Its frozen 2026 result was
+165.22/+116.98/+55.47, effectively tied with original N5 rather than a clear
transport improvement. The HPO arm remained a challenger; the exact original
single-forest N5 was this archived arm's selected model.

#### Bundle-local semantic roles and activation-weighted aggregates

The raw-archetype recurrence lane has now been implemented and tested. Raw
memberships are used only inside one identical Geometry/K9 bundle. Clusters are
ordered using **training-only expected policy residual**, then exposed as
activation-scaled semantic contributions rather than `cluster_00`-style slots:

- expected residual;
- downside risk;
- effective support;
- confidence.

A second arm normalizes all nine memberships for each candidate timestamp and
sums the four semantic contributions across clusters. It therefore answers,
for the state that is active *now*, the membership-weighted expected residual,
downside risk, effective support, and confidence. Raw cluster identities never
enter the pooled forest.

The diagnostic uses the first half of each exact bundle for fitting and the
second half for evaluation. Eight 2025 bundles select the representation; the
same frozen choice is then reported on eight 2026 bundles without using 2026
for selection. The objective is `Top5 + 0.5×Top10 + 0.2×Top20`, with all values
being mean exposure-weighted selected-policy net bps/trade across bundles.

| Year | Arm | Objective | Top 5% | Top 10% | Top 20% | Mean uplift vs stable | Positive-uplift bundles | Worst uplift |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 2025 | Stable 45-field N5 | 204.760 | 157.888 | 82.584 | 27.900 | — | — | — |
| 2025 | **Activation-weighted semantic aggregate** | **205.104** | **158.084** | **82.807** | **28.083** | **+0.344** | 5/8 | −1.234 |
| 2025 | Semantic roles only | 204.375 | 157.602 | 82.398 | 27.872 | −0.385 | 2/8 | −1.809 |
| 2025 | Roles + activation aggregate | 204.819 | 157.903 | 82.654 | 27.941 | +0.059 | 4/8 | −0.756 |
| 2026 | Stable 45-field N5 | 59.390 | 58.534 | 10.870 | −22.896 | — | — | — |
| 2026 | **2025-selected activation aggregate** | **59.397** | **58.573** | 10.823 | −22.935 | **+0.008** | 5/8 | −0.988 |
| 2026 | Semantic roles only | 59.717 | 58.814 | 10.951 | −22.865 | +0.327 | 6/8 | −1.231 |
| 2026 | Roles + activation aggregate | 59.905 | 58.922 | 11.069 | −22.760 | +0.515 | 4/8 | −1.348 |

The activation aggregate is the legitimate 2025 winner, but its frozen 2026
uplift is effectively zero and Top-10/20 are marginally worse. The apparently
better 2026 roles-plus-aggregate arm cannot be retro-promoted because it did
not win in 2025 and improves only four of eight 2026 bundles. Accordingly,
these four semantic aggregates are retained as shadow diagnostics, not added
to the archived 45-field N5 contract. Production promotion also requires a
cold-start-safe replay of the current bundle encoder over pre-cutoff history;
the half-bundle experiment cannot score a bundle from its first timestamp.

Evidence is immutable under
`data_perp/artifacts/strict_r3_n5_bundle_local_recurrence_20260810_v2`.

## A5. Archived handover references

The exact ablation funnel, parameters, monthly support, bootstrap, calibration, K9-regime results, feature drift, implementation files, and immutable artifact paths are documented in:

- `docs/STRICT_R3_C3_SELF_DISTILLATION_LONG_ONLY_HANDOVER_20260810.md`
- `docs/STRICT_R3_OUTCOME_ADMISSION_CONVERSION_REPAIR_20260810.md`
- `docs/TEN_HEAD_CONDITIONAL_USEFULNESS_FUNNEL_20260810.md`
- `docs/TEN_HEAD_K9_CONSENSUS_INTEGRATION_20260810.md`
- `docs/STRICT_R3_TOPTAIL_RELIABILITY_GEOMETRY_ABLATION_20260810.md`
- `data_perp/artifacts/strict_r3_c3_self_distillation_long_only_report_20260810_v3`

### Archived schema-v4 implementation

| File | Canonical role |
|---|---|
| `config/strict_r3_conditional_consensus_v1.json` | Immutable target/query/field contract for all ten canonical residual heads |
| `extreme_price_movements/strict_r3_canonical_current.py` | Monthly upstream and four-week C3 conversion bundles, scoring, persistence, and admission |
| `scripts/train_strict_r3_canonical.py` | Explicit `upstream` or `conversion` bundle trainer |
| `scripts/run_strict_r3_canonical_walkforward.py` | Exact monthly-upstream → four-week-conversion walk-forward producer |
| `scripts/score_strict_r3_forward.py` | Outcome-free schema-v4 scorer; schema-v2 is opt-in reconciliation only |
| `scripts/replay_strict_r3_forward_portfolio.py` | Hierarchical EV admission and constrained portfolio auction |
| `tests/test_strict_r3_canonical_current.py` | Contract, score decomposition, K9 veto, and Severe-shadow tests |
| `config/strict_r3_ldf_support_v3.json` | Canonical 45-field LDF contract for exact original `N5_drf_support_l110_meanrisk` |
| `config/strict_r3_ldf_support_v2.json` | Demoted two-forest HPO trial-8 challenger contract |
| `config/strict_r3_n5_forest_support_v1.json` | Superseded legacy single-forest N5 contract |
| `extreme_price_movements/n5_forest_support_sizing.py` | LDF implementation, legacy bundle compatibility, and HPO surface |
| `extreme_price_movements/strict_r3_n5_canonical.py` | Canonical LDF train/score/persist/load contract |
| `scripts/train_strict_r3_n5_sizing.py` | Three-month resolved-history LDF bundle trainer |
| `scripts/materialize_strict_r3_n5_causal_features.py` | Target-free prior-resolved N5 feature sidecar and coverage audit |
| `scripts/run_strict_r3_n5_canonical_selection.py` | Portable MDA, target/model HPO, and frozen 2026 replay |
| `scripts/replay_strict_r3_n5_existing_risk.py` | True unit-size matched Sharpe/Sortino/Calmar/ulcer audit |
| `scripts/run_strict_r3_n5_bundle_local_recurrence.py` | Same-bundle raw-archetype role and activation-weighted recurrence diagnostic |

`score_strict_r3_forward.py` and `replay_strict_r3_forward_portfolio.py` both
default to `current-v4`; `legacy-v2` must be requested explicitly. Schema-v3
conversion bundles fail the schema-v4 loader rather than being silently
reinterpreted. Monthly upstream schema-v3 bundles remain reusable because the
base/map/consensus contract did not change. The current
portfolio command rejects active Severe demotion, verifies that every
four-week conversion bundle maps to one rolling C3 geometry identity, and
requires and hashes the frozen pre-2025 SimplePolicyOptimiser winner artifact,
rejecting any parameter mismatch. A different policy requires separately named
map/residual/correctness retraining.

This paragraph describes the superseded schema-v4 experiment only.  In the
current schema-v5 contract, N5 is shadow-only, unit relative sizing is active,
and Cell-day trim 15% drives admission.  Neither this historical N5 arm nor the
current stack is live-trading approved.  A later untouched period and explicit
end-to-end service wiring/parity dry run remain required.

# Part B — Historical frozen-geometry schema-v2 control

The sections below preserve the previous one-time October–December 2024 geometry contract, its older fixed-policy comparisons, and extended 2025–August 2026 evidence. They are retained to explain lineage and are no longer the selected downstream architecture.

## 1. Authoritative architecture

    target-free point-in-time long candidates
        → 120-field strict-R3 base
        → same-model prior-42-day base rank
        → prior-OOF 20-bin policy-net map
        → ten policy-net residual LambdaRank heads
        → 75% base rank + 25% consensus rank
        → one frozen October–December 2024 geometry/K9 representation
        → Severe-200 one-way demotion
        → same-model prior-42-day raw-Severe CDF
        → causal prior-resolved 21-day expected-net admission at +50 bps
        → global long-only portfolio auction

There is no held-window percentile operation at any stage. The candidate, feature, and prediction surfaces are generated without future paths, outcome validity, or label completeness. Outcomes are joined only after scoring for evaluation.

The globally pooled top 1%, 2%, and 3% results are retrospective ranking diagnostics. They are not executable admission rules. The executable path is the prior-resolved 21-day expected-net map followed by the portfolio auction.

## 2. Candidate and feature contract

The candidate universe is every frozen-universe symbol × signal-hour row that is available and executable at decision time. Entry is the first bar after the signal close: signal close + one hour.

Cross-sectional fields are built from the complete contemporaneously available market universe before candidate filtering. Future path completeness cannot affect candidate identity or input features.

Long-only target-free population:

| Item | Rows |
|---|---:|
| Raw frozen-universe hourly rows | 2,382,720 |
| Decision-time eligible rows | 2,217,364 |
| Decision-time rejected rows | 165,356 |
| Final 2023–2026 source-panel rows after the 90% feature gate | 3,440,876 |

All 120 model fields vary and have at least 90% finite coverage on the source panel. The lowest observed individual-field coverage is 94.66%. Remaining missing values are imputed from training-only medians.

The authorized 15-minute coarse proxy applies only to ob_trade_size_to_l1_depth_z_24h. It is a same-timestamp market-median fallback and is labelled as such. No other missing strict contract field is silently synthesized.

The ordered feature contract is:

- config/strict_r3_canonical_v2_feature_contract.json

## 3. Strict-R3 base

### Target

The base is a three-class R3 classifier:

| Class | Meaning |
|---:|---|
| 0 | adverse-first |
| 1 | weak, timeout, or valid unresolved path |
| 2 | robust clear before adverse movement |

The base score is:

    base_score = P(clear) - 0.5 × P(adverse)

The class meanings were checked against exact economics. On the audited sample, mean exact net by class was approximately −407.5, −197.1, and +83.9 bps for adverse, weak, and clear respectively.

### Model

LightGBM multiclass classifier:

| Parameter | Value |
|---|---:|
| Trees | 220 |
| Learning rate | 0.035 |
| Max depth | 5 |
| Leaves | 24 |
| Minimum child rows | 2,400 |
| Feature fraction | 0.85 |
| L2 | 20 |
| Training cap | 240,000 rows |

Each held month is scored by a model trained only on rows whose 12-hour labels resolve before that month.

## 4. Prequential policy-net map

The held-month base and its preceding 42-day reference window are scored by the same monthly bundle. The base rank is the empirical CDF of the held score against that prior reference, never against the held month.

A 20-bin monotonic map converts base rank to expected fixed-policy net bps. It is fitted only on earlier strict-prequential base predictions with already resolved policy outcomes.

    base_anchor_bps = prior-OOF policy-net map(base_rank42)

No in-sample or post-date base prediction enters this map.

## 5. Ten residual consensus heads

### Target

    policy_residual_bps =
        realised fixed-policy net bps
        - prequential base_anchor_bps

Residual grades are ordinal bands:

| Grade | Residual |
|---:|---|
| 0 | at or below −150 bps |
| 1 | −150 to −50 bps |
| 2 | −50 to +50 bps |
| 3 | +50 to +150 bps |
| 4 | above +150 bps |

### Model and queries

The ten heads are LightGBM LambdaRank models:

- feature caps 40, 60, 80, 100, and 120;
- ordinary and equal-month weighting at every cap;
- four-hour UTC × long-side query groups;
- training-fold score references, never held-window ranks.

Frozen parameters:

| Parameter | Value |
|---|---:|
| Trees | 120 |
| Learning rate | 0.035 |
| Max depth | 5 |
| Leaves | 31 |
| Minimum child rows | 300 |
| Feature fraction | 0.82 |
| Bagging fraction | 0.82 |
| L1 | 0.02 |
| L2 | 2.0 |
| Max bin | 127 |
| LambdaRank truncation | 10 |
| Label gains | 0, 0.25, 1, 3, 7 |

The consensus rank is the median of the ten head ranks. The upstream score is:

    upstream = 0.75 × base_rank42 + 0.25 × consensus_rank

## 6. Frozen geometry/K9 and Severe-200

Geometry/K9 is fitted exactly once and is never refitted monthly.

Definition window:

    2024-10-01 00:00 UTC
    ≤ decision timestamp
    < 2025-01-01 00:00 UTC

Valid warm-up support:

| Month | Rows |
|---|---:|
| October 2024 | 45,019 |
| November 2024 | 45,525 |
| December 2024 | 44,706 |
| Total | 135,250 |

The 64-tree geometry encoder uses the complete available warm-up under the 240,000-row cap. The K9 MiniBatchKMeans fit uses 100,000 equal-month sampled rows. Leaf support and membership temperature use the complete warm-up in batches. Geometry-definition rows are excluded from supervised Severe-200 training.

Frozen geometry hash:

    7a602dfb5f10bef3791fd869b17dcfaeb53f96264fa8983c01ef5fd79681191c

All 20 monthly bundles and all 2,201,410 held predictions carry that single hash. Monthly-bundle validation fails closed if the definition dates, hash, field order, or representation state differ.

### Severe target and score

The Severe classifier receives exactly 123 ordered fields: five core upstream fields, 73 causal context fields, and 45 frozen geometry/K9 fields.

    severe_target = 1[exact TP6/SL4 H12 net bps ≤ −200]

Frozen classifier:

| Parameter | Value |
|---|---:|
| Trees | 35 |
| Learning rate | 0.044477 |
| Max depth | 5 |
| Leaves | 15 |
| Minimum child rows | 103 |
| Feature fraction | 0.73933 |
| Bagging fraction | 0.78535 |
| L1 | 0.02534 |
| L2 | 16.5789 |
| Max bin | 127 |

One-way demotion:

    raw_severe =
        upstream × (1 - 0.5 × P(severe loss))

Final score:

    final_score =
        CDF(raw_severe, current-model prior-42-day raw_severe)

For January 2026, the January bundle re-scores the full 2025-11-20 through 2025-12-31 reference. No score from another monthly model is reused.

## 7. Exit and label contracts

### Model-aligned frozen trailing policy

The base economic map and residual heads use:

| Component | Contract |
|---|---|
| Entry | signal close + one hour |
| Stop | 3 ATR |
| Trailing activation | 0.5 ATR |
| Giveback | 0.25 ATR |
| Timeout | 12 hours |
| Cost | 100 bps exactly once |

The diagnostic replay uses 15-minute bars and exact minute data where available for selected rows.

### Exact TP6/SL4 comparator

| Component | Contract |
|---|---|
| Entry | exact decision-minute open |
| Take profit | +6 ATR |
| Stop loss | −4 ATR |
| Timeout | 12 hours |
| Same-minute conflict | adverse/stop precedence |
| ATR | Wilder ATR(14) from completed pre-entry hourly candles |
| Cost | 100 bps exactly once |

Exact TP6/SL4 labels were materialized after scoring for the outcome-free globally selected top 3%. Coverage is 98.1–98.2%. Invalid rows remain null and are excluded, not encoded as losses.

## 8. Causal walk-forward ranking results

The ranking population contains 2,201,252 finite final scores. Selection is one pooled global ranking across the complete evaluation window, then decomposed by month and week.

### Global tails

| Tail | Exit | Selected score rows | Valid outcomes | Coverage | Gross bps/trade | Net bps/trade | Positive rate | Trades/calendar day |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1% | Exact TP6/SL4 | 22,013 | 21,601 | 98.13% | +123.29 | **+23.29** | 47.80% | 36.99 |
| 1% | Frozen trailing | 22,013 | 21,940 | 99.67% | +225.19 | **+125.19** | 59.48% | 37.57 |
| 2% | Exact TP6/SL4 | 44,026 | 43,209 | 98.14% | +98.28 | **−1.72** | 46.03% | 73.99 |
| 2% | Frozen trailing | 44,026 | 43,862 | 99.63% | +184.60 | **+84.60** | 56.11% | 75.11 |
| 3% | Exact TP6/SL4 | 66,038 | 64,859 | 98.21% | +84.59 | **−15.41** | 44.84% | 111.06 |
| 3% | Frozen trailing | 66,038 | 65,782 | 99.61% | +162.33 | **+62.33** | 53.68% | 112.64 |

The score is policy-aligned: it ranks the trailing-policy outcome much better than the alternative TP6/SL4 exit. TP6/SL4 is positive only at the narrowest global tail.

Trailing-policy stage attribution confirms that the complete stack is active:

| Score stage | Top 1% net | Top 2% net | Top 3% net |
|---|---:|---:|---:|
| Base rank42 | −8.03 | −16.90 | −22.83 |
| Base + residual consensus | +79.27 | +47.13 | +31.06 |
| Plus Severe-200 demotion | +123.13 | +84.41 | +60.54 |
| Plus same-model prior-42-day CDF | **+125.19** | **+84.60** | **+62.33** |

The residual consensus is the main conversion repair, the Severe overlay adds
substantial tail protection, and the final prior-only CDF preserves that
ordering without using the held distribution.

### Monthly net bps/trade

| Month | TP6 1% | TP6 2% | TP6 3% | Trailing 1% | Trailing 2% | Trailing 3% |
|---|---:|---:|---:|---:|---:|---:|
| 2025-01 | +102.52 | +112.17 | +106.78 | +162.16 | +177.26 | +168.48 |
| 2025-02 | −82.21 | −113.75 | −83.36 | +64.96 | +30.93 | +55.88 |
| 2025-03 | +127.35 | −77.49 | −118.00 | +278.94 | +115.40 | +61.06 |
| 2025-04 | +141.15 | +98.28 | +82.50 | +282.42 | +206.32 | +167.54 |
| 2025-05 | +81.20 | +26.78 | −7.11 | +134.79 | +86.41 | +64.65 |
| 2025-06 | +133.58 | +97.49 | +76.43 | +202.99 | +165.92 | +139.40 |
| 2025-07 | +0.09 | −8.76 | −16.68 | +104.77 | +61.98 | +42.59 |
| 2025-08 | −210.06 | −169.70 | −156.13 | +55.14 | +36.21 | +3.04 |
| 2025-09 | +115.87 | +72.14 | +53.90 | +197.21 | +126.37 | +102.19 |
| 2025-10 | +79.64 | +38.38 | +10.63 | +147.93 | +85.75 | +51.23 |
| 2025-11 | −0.52 | −16.11 | −31.51 | +133.89 | +80.58 | +61.98 |
| 2025-12 | −9.87 | −14.82 | −18.74 | +111.28 | +81.23 | +68.56 |
| 2026-01 | +57.60 | +23.89 | −3.53 | +150.71 | +92.52 | +60.69 |
| 2026-02 | +122.39 | +28.00 | −3.39 | +338.35 | +207.61 | +141.40 |
| 2026-03 | −17.55 | −28.39 | −21.46 | +74.48 | +73.83 | +75.41 |
| 2026-04 | +7.39 | −0.73 | −20.38 | +169.78 | +148.78 | +127.31 |
| 2026-05 | −83.44 | −76.75 | −69.04 | +32.72 | +18.74 | +7.41 |
| 2026-06 | −396.93 | −214.86 | −61.11 | −59.06 | +43.80 | −16.99 |
| 2026-07 | −53.14 | −46.79 | −51.31 | +20.32 | +15.37 | +5.14 |
| 2026-08 | −402.45 | −122.25 | −94.35 | +196.17 | +78.58 | −18.12 |

Monthly stability:

| Exit | Tail | Positive months | Negative months | Equal-month average | Worst month | Best month |
|---|---:|---:|---:|---:|---:|---:|
| TP6/SL4 | 1% | 11 | 9 | −14.37 | −402.45 | +141.15 |
| TP6/SL4 | 2% | 8 | 12 | −19.66 | −214.86 | +112.17 |
| TP6/SL4 | 3% | 5 | 15 | −21.29 | −156.13 | +106.78 |
| Frozen trailing | 1% | 19 | 1 | +140.00 | −59.06 | +338.35 |
| Frozen trailing | 2% | 20 | 0 | +96.68 | +15.37 | +207.61 |
| Frozen trailing | 3% | 18 | 2 | +68.44 | −18.12 | +168.48 |

Weekly stability:

| Exit | Tail | Covered weeks | Positive | Negative | Equal-week average | Worst | Best |
|---|---:|---:|---:|---:|---:|---:|---:|
| TP6/SL4 | 1% | 79 | 44 | 35 | +9.82 | −1,513.65 | +450.58 |
| TP6/SL4 | 2% | 82 | 45 | 37 | −11.21 | −1,513.65 | +351.99 |
| TP6/SL4 | 3% | 83 | 42 | 41 | −0.57 | −705.36 | +512.56 |
| Frozen trailing | 1% | 79 | 74 | 5 | +158.34 | −100.00 | +827.02 |
| Frozen trailing | 2% | 82 | 74 | 8 | +117.56 | −65.50 | +345.17 |
| Frozen trailing | 3% | 83 | 72 | 11 | +92.64 | −128.58 | +444.73 |

Week-level extremes can have very small support. The authoritative week-by-week rows, including row counts and coverage, are in weekly_tail_metrics.parquet rather than being inferred from the summary.

## 9. SimplePolicyOptimiser and executable portfolio

The optimizer was selected only on strict-prequential 2024 rows. It did not use the 2025–2026 evaluation period.

Development selection:

- one pooled-global upstream top-5% population;
- deterministic cap of 3,500 rows per month;
- 40 trials;
- objective: median monthly net bps/trade minus 0.5 × monthly net MAD;
- fixed 12-hour timeout, signal-close + one-hour entry, and 100-bps cost.

Winner:

| Parameter | Value |
|---|---:|
| Stop | 4.1520 ATR |
| Trailing activation | 2.3262 ATR |
| Giveback | 0.10237 ATR |
| Development objective | +122.33 bps |
| Frozen-policy control objective | +54.58 bps |

This is a separately named execution-layer policy. The residual stack remains trained on the frozen SL3/activation0.5/giveback0.25 policy target. A fully target-aligned optimized-policy model requires separate retraining.

### Causal admission

For every decision:

1. use final Severe CDF42;
2. use only fully resolved outcomes in the preceding 21 calendar days;
3. fit 20 score bins with 5% trimming;
4. use pooled-parent, long-side-shrunk common-bps mapping;
5. require parent support of 500 and side support of 20;
6. admit only mapped expected net of at least +50 bps;
7. fail closed when support is insufficient.

Currently actionable admitted candidates are ordered by mapped expected net, then passed to:

- maximum eight concurrent positions;
- maximum two new positions per 15-minute bar;
- maximum one position per asset;
- 80% entry-margin cap;
- 7× leverage;
- 10% wallet margin slot;
- initial wallet $1,000.

### Aggregate executable replay

| Metric | Result |
|---|---:|
| Admission-passing candidates | 37,759 |
| Accepted trades | 2,356 |
| Trades per calendar day | 4.03 |
| Gross bps/trade | +161.01 |
| Net bps/trade | **+61.01** |
| Positive-trade rate | 58.45% |
| Net sum | +143,745.78 bps |
| Trade weeks positive / negative / zero | 27 / 9 / 48 |
| Trade months positive / negative / zero | 11 / 3 / 6 |
| Maximum observed entry margin | 80.00% |
| Maximum concurrent positions | 8 |
| Maximum new entries per bar | 2 |
| Mark-to-market maximum drawdown | −86.06% |
| Realized-wallet maximum drawdown | −86.93% |

The wallet path compounds aggressively and is not a stable headline. Per-trade economics, admission coverage, and drawdown are more informative. The high drawdown prevents production promotion despite positive mean net EV.

Monthly executable replay:

| Month | Trades | Net bps/trade | Positive rate | Wallet return |
|---|---:|---:|---:|---:|
| 2025-01 | 417 | +75.83 | 58.75% | +304.41% |
| 2025-02 | 116 | +83.53 | 56.90% | −40.17% |
| 2025-03 | 92 | −51.11 | 46.74% | −50.36% |
| 2025-04 | 436 | +141.09 | 66.28% | +3,673.31% |
| 2025-05 | 296 | +33.14 | 56.42% | +25.65% |
| 2025-06 | 17 | +104.08 | 64.71% | +9.16% |
| 2025-07 | 166 | +99.92 | 63.25% | +152.70% |
| 2025-08 | 29 | +37.71 | 68.97% | +52.30% |
| 2025-09 | 0 | — | — | 0.00% |
| 2025-10 | 335 | +53.43 | 60.60% | +44.11% |
| 2025-11 | 127 | +70.93 | 58.27% | +84.39% |
| 2025-12 | 0 | — | — | 0.00% |
| 2026-01 | 230 | −54.13 | 46.09% | −75.51% |
| 2026-02 | 21 | +45.05 | 52.38% | +26.40% |
| 2026-03 | 45 | −63.69 | 44.44% | −16.35% |
| 2026-04 | 29 | +131.88 | 58.62% | +21.56% |
| 2026-05 | 0 | — | — | 0.00% |
| 2026-06 | 0 | — | — | 0.00% |
| 2026-07 | 0 | — | — | 0.00% |
| 2026-08 | 0 | — | — | 0.00% |

The May–August 2026 zeros are causal fail-closed behavior: the prior-21-day map had no bin above +50 bps. They are not missing model scores. This is safer than forcing trades, but it exposes an admission-transport gap that must be addressed before deployment.

## 10. Causality and wiring audit

The final audit passes:

- long side only;
- 20 monthly bundles and 2,201,410 predictions;
- target-free candidate and scoring surfaces;
- unique candidate identities;
- zero held-window percentiles;
- no outcome columns consumed during scoring;
- same bundle for held rows and prior-42-day reference;
- all references strictly precede their cutoff;
- strict-prequential downstream ledger;
- policy-net rather than TP6 residual target;
- one frozen geometry hash across every month;
- no geometry-definition row in supervised Severe training;
- exact ordered 123-field Severe contract;
- all 120 source fields varying with at least 90% coverage;
- cost applied exactly once;
- prior-resolved causal 21-day admission;
- no retrospective live threshold;
- observed 80% entry-margin, eight-position, and two-entry-per-bar caps.

The January 2026 reference mismatch is repaired: January uses a same-model replay over its full prior 42-day window.

The audit file is:

- data_perp/artifacts/strict_r3_schema_v2_optimised_policy_portfolio_targetfree_long_2025_aug7_2026_20260809_v3/correctness_test_report.json

## 11. Authoritative implementation files

| File | Role |
|---|---|
| extreme_price_movements/strict_r3_canonical_v2.py | Model contracts, immutable geometry state, prequential ledger, monthly bundles, hashes |
| config/strict_r3_canonical_v2_feature_contract.json | Ordered 120-field base and 73-field Severe context contracts |
| config/strict_r3_frozen_15m_policy.json | Frozen model-aligned execution policy |
| scripts/materialize_strict_r3_target_free_hourly_grid_v2.py | Target-free point-in-time long population |
| scripts/materialize_strict_r3_forward_features.py | Vectorized causal feature materialization |
| scripts/assemble_strict_r3_canonical_source_panel_v2.py | Feature/label source panel with coverage gate |
| scripts/materialize_strict_r3_frozen_policy_labels_v2.py | Model-aligned fixed-policy labels |
| scripts/materialize_strict_r3_prequential_ledger_v2.py | Strict prequential base/map/residual ledger |
| scripts/fit_strict_r3_geometry_k9_v2.py | One-time October–December geometry/K9 fit |
| scripts/train_strict_r3_canonical_v2.py | Monthly bundle fit |
| scripts/run_strict_r3_canonical_walkforward_v2.py | 2025–2026 monthly walk-forward orchestration |
| scripts/score_strict_r3_forward.py | Thin outcome-free scorer |
| scripts/replay_strict_r3_forward_simple_policy_15m.py | Frozen trailing-policy outcome replay |
| scripts/materialize_packb_tp6_sl4_h12_labels.py | Exact TP6/SL4 comparator labels |
| scripts/evaluate_strict_r3_walkforward_long_v2.py | Pooled-global long-tail and period metrics |
| scripts/optimise_strict_r3_schema_v2_policy_pre2025.py | Pre-2025 SimplePolicyOptimiser selection |
| scripts/replay_strict_r3_forward_portfolio.py | Causal 21-day admission and portfolio auction |
| scripts/audit_strict_r3_schema_v2_long_wiring.py | End-to-end correctness audit |

No production logic or artifact is imported from /Users/remyroche/Documents/Codex.

## 12. Authoritative artifacts

| Artifact | Path |
|---|---|
| Target-free long population | data_perp/artifacts/strict_r3_schema_v2_target_free_long_2025_aug7_2026_20260809_v1 |
| Final source panel | data_perp/artifacts/strict_r3_schema_v2_source_panel_targetfree_long_2023_aug7_2026_20260809_v2 |
| Prequential ledger | data_perp/artifacts/strict_r3_schema_v2_prequential_ledger_targetfree_long_2024_2026_20260809_v1 |
| Frozen geometry/K9 | data_perp/artifacts/strict_r3_schema_v2_geometry_k9_targetfree_long_octdec2024_20260809_v3 |
| Monthly walk-forward bundles and scores | data_perp/artifacts/strict_r3_schema_v2_walkforward_targetfree_long_2025_aug7_2026_20260809_v1 |
| Fixed trailing replay | data_perp/artifacts/strict_r3_schema_v2_fixed_policy_replay_targetfree_long_2025_aug7_2026_20260810_v2 |
| Exact selected-tail TP6 labels | data_perp/artifacts/strict_r3_schema_v2_exact_tp6_tail_long_2025_aug7_2026_20260810_v1 |
| Final global/month/week diagnostics | data_perp/artifacts/strict_r3_schema_v2_tail_metrics_exactbackfill_long_2025_aug7_2026_20260810_v1 |
| Pre-2025 optimized policy | data_perp/artifacts/strict_r3_schema_v2_simple_policy_targetfree_long_pre2025_20260809_v3 |
| Executable admission/portfolio replay | data_perp/artifacts/strict_r3_schema_v2_optimised_policy_portfolio_targetfree_long_2025_aug7_2026_20260809_v3 |

## 13. Historical-control boundary

This closing section belongs to the archived schema-v2 control. The active
architecture is the schema-v5 contract at the top of this document: frozen
October–December 2024 Geometry/K9, 28-day Cell-day trim-15 expected net, A0
nine-month posterior anchor plus bounded A5-10 top-15 reranking/admission, and unit sizing. The rolling-C3 and N5 paths
are historical/shadow comparators only.

Still prohibited in every version:

- held-month base, head, or final percentiles;
- future-path-qualified candidates;
- residual fitting on in-sample or post-date base predictions;
- mixing geometry bundles inside one downstream fit;
- base-consensus-only August artifacts;
- retrospective pooled top-k as a live threshold;
- substituting an exit policy without refitting its policy map and residual target.

## 14. Decision and next gate

The A5 bounded-10 over A0-top-15 arm is the canonical long-only executable-research
and hourly-shadow contract. It is not an exchange-authorized production policy.
The remaining promotion gate is a later untouched forward period using the
same hashes, 28-day map, nine-month A0/A4 targets, fields and parameters,
causal A4 calibration, A0 +50-bps hurdle, timestamp-local top-15 domain,
bounded alpha 0.10, unit sizing, optimized exit policy, and portfolio limits.
Until that gate passes, the hourly runner remains shadow-only and makes zero
exchange calls.
