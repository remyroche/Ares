# P8u Meta-Layer Family Selection — 2026-08-28

## Decision

This is offline, long-only research.  It does **not** change the live trader,
the deployed canonical stack, exchange execution, or a production admission
threshold.

The retained base is the frozen Router50 → 72-field Raw-bps CatBoost
QueryRMSE (`tail_linear_125`) contract described in
[`P8U_BASE_PRECISION_PRESERVATION_HANDOVER_20260828.md`](P8U_BASE_PRECISION_PRESERVATION_HANDOVER_20260828.md).

The single **F120 unexpected-under-confidence XENDCG** head is selected as the
canonical *research* Meta coordinate. On the Apr--Jul 2026
strict-prequential portfolio block it is effectively flat in total net EV
versus Base-only (`-0.03%`) while increasing unit EV and materially reducing
drawdown. It now supersedes the F72 Under-XENDCG research control for the P8U
Router50/F72 Base research stack, but has **not** earned live authority: its
feature/model/family choice and this evaluation share the same development
period, and it reduces entries.

The exact frozen composition is documented in
[`P8U_ROUTED_F72_UNDERF120_RESEARCH_CANONICAL_20260828.md`](P8U_ROUTED_F72_UNDERF120_RESEARCH_CANONICAL_20260828.md).

## Fixed evaluation contract

```text
Frozen P8u Router50 identities
  → Raw-bps CatBoost QueryRMSE Base timestamp rank
  → optional Meta rank (75% Base + 25% equal-weight Meta blend)
  → independent strict-prequential Current and BCF MC1 maps
  → both MC1 expected values >= +50 bps
  → one chronological, portfolio-constrained auction
  → canonical reconciled rich-policy net outcome
```

All Base and Meta scores are written as target-free receipts before the policy
label join.  Meta and MC1 training only use labels resolved before each held
month.  The evaluated months are Apr--Jul 2026, after a Jan--Mar MC1 warm-up.
No pooled future rank, outcome-qualified candidate filter, or exchange path is
used.

## Target/query family screen

One winner per family was selected by strict-OOF `SStableMeta`, then sent to
the exact same dual-MC1 and portfolio replay.  The screen is a diagnostic only;
the portfolio table determines advancement.

| Family | Frozen winner | Query | SStableMeta | Conditional MI | MC1/portfolio outcome |
|---|---|---|---:|---:|---|
| Magnitude residual | `magnitude_bps__base_band_block28` | Base-band × 28-day block | -0.0615 | 0.1177 | Small +1.49k bps total, but worse drawdown and worst week |
| Under-confidence | `under_bps100__timestamp` | Exact timestamp | -0.0007 | 0.1613 | Qualified only after the F120/XENDCG feature/model refinement |
| Over-confidence | `over_atr1__timestamp` | Exact timestamp | -0.1724 | 0.0444 | Reject: -10.66k total bps |
| Signed calibration state | `state_bps__base_band_block28` | Base-band × 28-day block | -0.0772 | 0.1134 | Reject: -20.15k total bps |

## Full-universe feature selection and model-family screen

The Under target was re-run from the full causal source universe: 1,407 raw
numeric fields, 1,098 coverage/variance-clean fields, then a 120-field
strict-OOF MDA/CMI/stability contract.  The intended 30--70 field reductions
lost `SStableMeta`; F120 is retained as the best *candidate*, not as a claim
that 120 is universally optimal.

All six model families used the same target, exact timestamp query, 120-field
contract, outer folds, and external BaseStable scoring metric.

| Family | SStableMeta | CMI conditional on Base | Residual IC | Conditional IC | Utility spread | Top-2 substitution EV |
|---|---:|---:|---:|---:|---:|---:|
| **CatBoost YetiRank** | **-0.0259** | 0.1800 | 0.1074 | 0.1010 | 41.92 | -31.89 |
| XGBoost pairwise | -0.0282 | 0.1765 | 0.1114 | 0.1153 | 43.24 | -29.39 |
| LightGBM LambdaRank | -0.0626 | 0.1703 | 0.1098 | 0.1098 | 43.53 | -36.90 |
| XGBoost NDCG | -0.0671 | 0.1763 | 0.1105 | 0.1089 | 41.65 | -39.82 |
| CatBoost QueryRMSE | -0.0824 | 0.1808 | 0.1167 | 0.1301 | 45.13 | -39.91 |
| LightGBM Rank-XENDCG | -0.0862 | 0.1801 | 0.1165 | 0.1338 | 45.59 | -43.36 |

CatBoost YetiRank won the standalone screen, so it alone received full,
chronological HPO.  Its selected configuration was depth 4, learning rate
0.0258825, L2 24.0070, random strength 0.199782, RSM 0.716737, and subsample
0.883348.  Its screen `SStableMeta` was +0.0563, but its isolated Apr--Jul
confirmation was **-0.1060**.  It was rejected before any canonical change.

## Single-head, constrained downstream comparison

All figures are realised rich-policy net bps for Apr--Jul 2026 under the
shared dual-MC1 +50-bps gate and one chronological constrained portfolio.

| Meta coordinate | Entries | MC1-admitted rows | Net EV/trade | Total net bps | Worst month | Worst week | Max DD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base-only control | 3,763 | 17,422 | +115.16 | +433,350 | +65.95 | +47.39 | -28.32% |
| Magnitude | 3,728 | 15,480 | +116.64 | **+434,836** | +66.48 | +45.64 | -30.69% |
| Under, F72 XENDCG control | 3,512 | 13,022 | +120.29 | +422,448 | +68.69 | +49.23 | -31.96% |
| **Under, F120 XENDCG challenger** | **3,634** | **14,074** | **+119.21** | **+433,201** | **+68.28** | **+50.44** | **-21.96%** |
| Under, F120 CatBoost YetiRank | 3,705 | 15,093 | +115.30 | +427,179 | +67.23 | +48.45 | -28.32% |
| Over | 3,549 | 14,059 | +119.10 | +422,688 | +69.30 | +47.32 | -28.54% |
| State | 3,526 | 13,002 | +117.19 | +413,200 | +65.42 | +38.92 | -23.55% |

F120 Under versus Base-only: -129 entries, +4.05 bps/trade, -150 total bps,
+2.33 bps worst month, +3.05 bps worst week, and **+6.35pp** less maximum
drawdown.  The trade-level Sortino proxy is 0.643 versus 0.628 for Base-only.
This is a promising risk trade-off but not a clean net-EV/participation
improvement.

## Meta authority audit

The retained F120 Under signal was then audited separately from Meta-head
selection.  This is important because its useful information must be given an
appropriate downstream role rather than mechanically blended into the
portfolio score.

The existing architecture uses the current (Base + Meta) MC1 map as a
**confirmation gate** and retains the BCF MC1 map as the auction priority:

```text
BCF MC1 expected EV >= 50 bps
AND Current (Base + Meta) MC1 expected EV >= 50 bps
-> auction priority = BCF MC1 expected EV
```

On the Apr--Jul 2026 P8u block, BCF-only candidates averaged +42.56 realised
net bps/trade while rows passing both maps averaged +117.58 before portfolio
constraints.  The Meta coordinate therefore provides useful conservative
confirmation.  It does not safely create a broad second source of entries.

The following constrained-portfolio probes all used the same frozen scores,
canonical rich-policy labels, +50-bps BCF gate, and one shared chronological
auction.  They are diagnostic only; no parameter from this table has been
promoted.

| Current-MC1 gate | Entries | Net EV/trade | Total net bps | Worst month | Worst week | Max DD |
|---:|---:|---:|---:|---:|---:|---:|
| 30 bps | 3,731 | +116.39 | +434,257 | +66.19 | +48.45 | -28.32% |
| 40 bps | 3,696 | +116.81 | +431,727 | +66.44 | +48.45 | -28.32% |
| **50 bps (frozen)** | **3,634** | **+119.21** | **+433,201** | **+68.28** | **+50.44** | **-21.96%** |
| 60 bps | 3,476 | +117.68 | +409,043 | +64.24 | +41.48 | -22.45% |
| 70 bps | 3,306 | +121.90 | +403,014 | +70.83 | +41.87 | -22.45% |
| 80 bps | 2,996 | +121.55 | +364,168 | +67.78 | +48.50 | -30.86% |

Auction-priority ablations also reject giving this Meta-derived map direct
ranking authority.  The frozen BCF priority retains +119.21 bps/trade,
+433,201 total bps, +68.28 worst month, +50.44 worst week, and -21.96% max
drawdown.  A 50/50 BCF/Current priority falls to +115.18 bps/trade,
+425,474 total bps, +62.05 worst month, +34.15 worst week, and -23.13% max
drawdown; Current-only priority is materially worse again (+112.91 bps/trade,
+418,202 total bps, -39.83% max drawdown).

A three-month, day-balanced, hierarchical monotone correction of BCF mapped
EV by Base-rank band x Meta-rank band was also rejected.  It increased raw
near-threshold admissions but reduced candidate-level realised EV from
+117.58 to +108.39 bps/trade; using that correction only to re-order the
already dual-admitted portfolio also degraded all portfolio risk metrics.

The next distinct research direction should therefore be a *bounded local
demoter*, not another global confirmation ranker: train only within high-BCF
score bands to identify severe policy-loss risk, calibrate its posterior
strictly prequentially, and permit it only to reduce mapped EV.  A second,
separate experiment may train a within-timestamp substitution head whose
target is policy outcome relative to the BCF-selected alternative; it must be
allowed to act only where two already-admitted candidates have comparable BCF
expected EV.  Neither head may manufacture an admission without passing the
existing BCF gate.

### Follow-up authority probes

The proposed local-demoter and replacement mechanisms were subsequently
tested on the same Apr--Jul 2026 strict-prequential block.  They remain
research diagnostics, not a second selection round for the retained F120
head.

1. A depth-2 severe-loss classifier was fitted only on earlier, label-resolved
   rows with BCF expected EV of at least +30 bps.  It consumed BCF and Current
   mapped EV, Base/Under ranks, their signed disagreement, and agreement
   state.  Its only possible effect was a subtraction from Current EV before
   the existing dual +50-bps gate.  A 50-bps probability-proportional
   demotion yielded +120.79 bps/trade, but only +405,002 total bps, versus
   +119.21 and +433,201 for the retained F120 confirmation gate.  Stronger
   demotion further traded contribution for smaller participation and did not
   improve drawdown.
2. Adding the other frozen Meta ranks to that demoter was rejected.  State was
   the least harmful addition (+122.17 bps/trade, +408,176 total bps at the
   same 50-bps authority); Magnitude, Over, and the combined M/O/S set all
   reduced total contribution and/or downside quality.  The retained F120
   Under head remains the only useful Meta input in this role.
3. Limiting the demoter to its within-timestamp top 5/10/20% risk slice and
   targeting either <= -100 or <= -200 realised policy bps did not advance.
   The best near-control case (+120.16 bps/trade) still reduced total bps and
   worsened or preserved drawdown.
4. A timestamp-query replacement ranker was trained on earlier policy residual
   relative to BCF expected EV and could re-order only already dual-admitted
   candidates.  Both the Core-only and Core+M/O/S variants lowered net EV,
   worst-month/week economics, and drawdown quality at every tested authority.
5. The Current MC1 map was augmented with causal 3/7/14-day Meta-versus-Base
   rank IC and top-tail excess-EV state.  January target-free score history
   was restored so its no-new-feature control reproduced the frozen result
   exactly.  The added state reduced total bps to +428,481 and did not improve
   any risk metric.

The actionable conclusion is more specific than simply “Meta does not work”:
the F120 Under output is valuable as a **single conservative confirmation
coordinate**.  The available Magnitude, Over, State, disagreement, and recent
correctness variants do not yet contain portable incremental information once
that gate and BCF priority are fixed.  Further Meta work should use a new,
predeclared target/data source or a later untouched period—not further blends,
thresholds, or authority tuning on this block.

### Conditional score-band audit

The four Meta heads were also assessed within fixed BCF mapped-EV bands, using
their timestamp-local bottom 20% versus top 20% ranks.  Positive delta means
the high-rank slice realised more canonical rich-policy net bps/trade; negative
severe delta means its <= -100-bps loss rate was lower.  The table uses every
valid Apr--Jul candidate, before admission and portfolio selection.

| BCF expected-EV band | Under F120 ΔEV / Δ severe-loss pp | Magnitude ΔEV / Δ severe-loss pp | Over ΔEV / Δ severe-loss pp | State ΔEV / Δ severe-loss pp |
|---|---:|---:|---:|---:|
| 30–50 bps | -9.24 / -9.35 | +2.97 / -9.41 | +85.72 / +0.79 | +46.69 / -9.93 |
| 50–75 bps | +113.37 / -17.54 | +14.21 / -7.71 | +6.69 / +11.38 | +46.30 / -8.82 |
| 75–100 bps | +128.60 / -12.61 | +122.53 / -13.50 | +111.56 / -2.77 | +120.56 / -11.35 |
| 100–150 bps | +226.14 / -21.53 | +68.16 / -14.43 | +144.96 / -0.95 | +76.77 / -21.44 |
| 150+ bps | +196.37 / -20.18 | +235.99 / -21.42 | +194.45 / -6.23 | +152.52 / -17.45 |

This explains the retained Under head's useful role: at BCF EV >= +50 bps its
high-rank slice has a large realised-EV advantage and materially lower severe
loss rate.  Below the BCF floor it is **not** a robust rescue signal: the
30–50-bps Under high slice is slightly worse in EV, despite less severe loss.
Over is particularly unsafe as a demoter around the 50–75-bps BCF band because
its high score is associated with a higher severe-loss rate.

The obvious score-band implementation was tested and rejected: subtracting
25/50/100 bps from Current EV for low-Under candidates in BCF 50–75, 50–100,
or all 50+ bands always reduced total bps and worsened at least one of
worst-month, worst-week, or drawdown.  Conditional association therefore does
not by itself justify a policy rule after the existing Current MC1 gate and
BCF-priority auction have already acted.

## Full equal-weight output grid and parsimony

`U` = F120 Under-XENDCG; `M` = magnitude; `O` = over-confidence; `S` = signed
state.  Each selected Meta coordinate has equal weight within the fixed 25%
Meta portion of Current.  No weight fitting is used.

| Combination | Entries | Net EV/trade | Total net bps | Worst month | Worst week | Max DD |
|---|---:|---:|---:|---:|---:|---:|
| Base-only | 3,763 | +115.16 | +433,350 | +65.95 | +47.39 | -28.32% |
| M | 3,728 | +116.64 | **+434,836** | +66.48 | +45.64 | -30.69% |
| U | 3,634 | +119.21 | +433,201 | +68.28 | +50.44 | **-21.96%** |
| U + M | 3,641 | +118.36 | +430,963 | +68.61 | +48.66 | -35.09% |
| M + S | 3,586 | +118.89 | +426,348 | +66.44 | +41.15 | -28.32% |
| U + M + O + S | 3,235 | +121.18 | +392,021 | +62.43 | +43.36 | -30.04% |

All remaining pair/triple combinations also reduce total net EV by at least
7.0k bps relative to Base-only.  The full grid therefore fails the parsimony
test: additional coordinates lower participation and total economic
contribution despite sometimes increasing per-trade EV.

For the required leave-one-out diagnostic of the all-head blend, no omitted
head shows durable additive authority.  Removing `M` actually improves the
full blend by +9.40k bps total; removing `S` improves it by +23.28k; removing
`U` improves it by +16.83k; and removing `O` improves it by +17.38k.  The
all-head blend is rejected rather than rationalised with post-hoc weights.

## Reproducibility and artifacts

- Target/query config: `config/strict_r3_p8u_meta_target_query_grid_20260828_v1.json`
- Full feature-selection winner: `data_perp/artifacts/strict_r3_p8u_meta_under_fullfeatures_selection_20260828_v2/`
- Cross-family audit: `data_perp/artifacts/strict_r3_p8u_meta_under_f120_crossmodel_janjul2026_20260828_v1/`
- Rejected CatBoost HPO: `data_perp/artifacts/strict_r3_p8u_meta_under_f120_catboost_yetirank_hpo_20260828_v1/`
- F120 challenger MC1 replay: `data_perp/artifacts/strict_r3_p8u_meta_fullfeatures_mc1_f120_janjul2026_20260828_v2/`
- Combination grid: `data_perp/artifacts/strict_r3_p8u_meta_full_combo_*_janjul2026_20260828_v1/`
- Model-family runner: `scripts/run_strict_r3_p8u_meta_crossmodel_v1.py`
- CatBoost HPO runner: `scripts/run_strict_r3_p8u_meta_catboost_hpo_v1.py`

Focused tests passed: `tests/test_run_strict_r3_p8u_meta_crossmodel_v1.py` and
`tests/test_run_strict_r3_p8u_meta_catboost_hpo_v1.py` (4 tests).

## Bounded BCF-MC1 × Meta interaction demotion — rejected

The score-band associations above motivated one further strictly bounded test.
It is more conservative than a score blend: a shallow, prequential adverse-path
classifier consumed the frozen BCF/Current MC1 expected-EV coordinates and the
four target-free Meta ranks (`U`, `M`, `O`, `S`).  It could apply only a
non-positive BCF correction, retained the unchanged Current-MC1 >= +50 bps
gate, and retained the chronological portfolio adapter.  The target-free input
panel was written before the canonical policy labels were joined.

The declared authority ranges were BCF MC1 EV 30--100, 30--150, and 30--200
bps.  For each, the model was fitted using only the prior four calendar months
with resolved labels.  It tested a severe-loss (`policy net <= -100 bps`) and
a non-positive-policy target, each at 50% and 100% authority.  Corrections
were shrunk by BCF-band training support, capped at -100 bps, and verified to
be zero outside their declared interval and never positive.

The May--July 2026 constrained replay rejects this family.  The closest result
was the 30--150 severe-loss model at 100% authority; its tiny +1.37k-bps total
gain is accompanied by a worse drawdown and weaker worst-period metrics, so it
does not meet the portfolio advance gate.

| Arm | Admitted | Entries | Net EV/trade | Total net bps | Worst month | Worst week | Max DD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Control | 10,531 | 2,780 | +103.64 | +288,125 | +68.28 | +50.44 | -21.96% |
| 30--100 severe-loss, 100% | 10,356 | 2,783 | +103.81 | +288,915 | +69.00 | +46.67 | -21.96% |
| **30--150 severe-loss, 100%** | 10,380 | 2,794 | +103.61 | **+289,492** | +66.40 | +47.20 | **-26.09%** |
| 30--200 severe-loss, 100% | 10,362 | 2,798 | +103.30 | +289,034 | +67.36 | +46.21 | -26.09% |
| Best non-positive-policy arm (30--100, 50%) | 10,490 | 2,794 | +101.84 | +284,535 | +64.35 | +43.68 | -21.96% |

This result reinforces the design diagnosis: the heads do contain conditional
information, but a generic post-map demoter mostly duplicates the dual mapper's
selection and then perturbs the constrained auction.  Further work should use
a distinct target or an explicitly different decision role, not another
demotion-range or authority sweep.

Artifact and focused test:

- `data_perp/artifacts/strict_r3_p8u_meta_bounded_bcf_demotion_janjul2026_20260828_v3/`
- `scripts/run_strict_r3_p8u_meta_bounded_bcf_demotion_v1.py`
- `tests/test_strict_r3_p8u_meta_bounded_bcf_demotion_v1.py` (2 passed)

## Banded CatBoost expected-EV mapper — rejected

A distinct calibration architecture was then tested: five frozen BCF-MC1 EV
bands (30--50, 50--75, 75--100, 100--150, 150+ bps) each received its own
shallow CatBoost regressor.  Its inputs were only target-free Current/BCF MC1
coordinates, their final scores, and the four target-free Meta ranks.  Its
target was canonical policy net clipped to [-300, +600] bps.

For every held month, the CatBoost model was fitted before a 28-day calibration
reserve.  An isotonic mapper was fitted only on that reserve's out-of-model
CatBoost predictions and fully resolved policy outcomes.  May--July values
were then mapped before any held outcomes were read.  Sparse isolated bands
used a depth-1 rather than depth-2 model; bands were never pooled.  The mapper
was evaluated both as a BCF-coordinate replacement (`Current >= 50` and
mapped CatBoost EV >= 50) and as an additional BCF confirmation.

| Arm | Admitted | Entries | Net EV/trade | Total net bps | Worst month | Worst week | Max DD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dual-MC1 control | 10,531 | 2,780 | +103.64 | +288,125 | +68.28 | +50.44 | -21.96% |
| CatBoost replaces BCF coordinate | 9,342 | 2,762 | +93.55 | +258,381 | +51.43 | +26.56 | -34.50% |
| CatBoost confirms BCF coordinate | 8,350 | 2,676 | +97.81 | +261,743 | +55.86 | +27.14 | -41.70% |

The mapper loses 10.09 bps/trade and 29.7k total bps when used as the BCF
coordinate; the confirmation version is also worse on every material economic
and risk metric.  This rejects direct clipped-EV regression plus isotonic
mapping on the present MC1/Meta geometry.  It should not be HPO-tuned on this
same May--July selection block.

One narrow role was also tested because the 150+ BCF band had the only
directionally useful CatBoost within-band relationship: keep the *identical*
dual-MC1 admission set and substitute CatBoost mapped EV for BCF auction
priority only for candidates whose raw BCF MC1 EV was at least +150 bps.  That
also fails: it keeps all 10,531 admissions but changes the constrained outcome
from 2,780 entries at +103.64 bps/trade / +288,125 total bps to 2,781 at
+102.39 bps/trade / +284,737 total bps; drawdown worsens from -21.96% to
-25.32%.  Therefore even its apparent 150+ conditional association is not
strong enough to improve the actual portfolio auction.

Artifact and focused test:

- `data_perp/artifacts/strict_r3_p8u_meta_banded_catboost_mapper_janjul2026_20260828_v4/`
- `scripts/run_strict_r3_p8u_meta_banded_catboost_mapper_v1.py`
- `tests/test_strict_r3_p8u_meta_banded_catboost_mapper_v1.py` (3 passed)

## Next valid gate

Do not use the Apr--Jul 2026 selected F120 result as production evidence.
Materialise a later, target-free panel; freeze both the F72 control and F120
candidate without retuning; rerun the same Base → Meta → dual-MC1 → constrained
portfolio chain.  Advance only if F120 preserves the drawdown/Sortino benefit
without a material loss of total net EV, trade participation, or worst-period
economics.
