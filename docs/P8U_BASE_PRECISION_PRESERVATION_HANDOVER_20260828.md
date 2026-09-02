# P8u Base Precision/Preservation Research — 2026-08-28

## Status

The Router50 gate is fixed upstream and is **not** a comparison arm in this work.
The canonical P8u **research** Base contract is the 72-field Raw-bps CatBoost
QueryRMSE head with tail-linear-125 timestamp-normalised sample weights. It is
canonical for subsequent P8u research, but has **not** been promoted to the
live stack.

The directly comparable control is the frozen 72-field P8u policy-ordinal
LightGBM Rank-XENDCG Base head. Both heads score identical Router50 candidate
identities. Neither receives the Router score as a numeric input, and neither
has a post-Router Base top-30% cutoff in the downstream diagnostic.

## Base objective

```text
BaseScore = 0.30 DTP2 + 0.30 DTP5 + 0.20 DTP10 + 0.20 ResidualUR10→30
ScoreStable = weekly robust mean(Q20–Q80) + 0.5 × mean(Q15, Q10, Q5)
```

All components are normalised to the fixed policy-ordinal Base control. DTP is
timestamp-local policy-net precision. ResidualUR10→30 measures utility retained
between ranks 10 and 30; `UR20` is a diagnostic, not an unstated fifth score term.

## Matched standalone Base results

Strict OOF protocol: frozen P8u Router50 candidate IDs; three resolved calendar
months of training; a 28-day label reserve; 60,000 complete-query cap; held
months November 2025 through July 2026. Held score panels are target-free until
the evaluation join.

| Base head | ScoreStable | DTP2 | DTP5 | DTP10 | Residual UR10→30 |
|---|---:|---:|---:|---:|---:|
| Policy-ordinal control | 1.174 | +151.75 bps | +92.05 bps | +44.97 bps | 0.416 |
| **Raw-bps CatBoost, tail-125** | **1.714** | **+190.30 bps** | **+131.26 bps** | **+86.14 bps** | 0.304 |
| Delta | **+0.540** | **+38.55 bps** | **+39.21 bps** | **+41.17 bps** | −0.113 |

The candidate creates a materially stronger tip but retains less incremental
rank-10→30 utility. Promotion therefore depends on downstream economics, not
this table alone.

### Candidate Base contract

| Item | Frozen value |
|---|---|
| Router | P8u, top 50% by timestamp; identity gate only |
| Features | Existing 72-field P8u causal feature contract |
| Target | Raw canonical rich-policy net bps, clipped and ordinalised into six equal-width bins |
| Model | CatBoost QueryRMSE, exact timestamp × long-side query |
| Target-search provenance | G3 clipped-economic; CatBoost QueryRMSE consumes the six ordinal labels directly and has no `label_gain` parameter |
| Weights | Tail-linear-125, renormalised inside timestamp and clipped to `[0.5, 2.0]` |
| Parameters | depth 5; learning rate 0.0650994; feature fraction 0.800651; bagging fraction 0.709605; L2 2.235726; random strength 0.942890 |
| Base cutoff | None after Router50: every routed row remains available to Meta |

The full HPO refresh was rejected: it scored 1.677 rather than the frozen
tail-125 candidate's 1.714 on the five-fold confirmation.

## Feature-set selection through downstream economics

The three frozen 130-field beam finalists were all sent through the same
target-free Base ledger, strict-OOF Meta fit, 50-bps MC1 diagnostic, and
constrained portfolio as F72. This satisfies the feature-selection downstream
test; it does **not** substitute for the later normal six-month promotion gate.

| Base contract, with under-Meta conditioning | Entries | MC1 admissions | Net EV/trade | Total net bps | Worst month | Worst week | MaxDD |
|---|---:|---:|---:|---:|---:|---:|---:|
| **F72 Raw-bps, frozen tail-125** | **1,282** | 5,449 | +158.96 | **+203.79k** | +132.71 | +98.68 | −14.78% |
| F130 MDA, frozen parameters | 1,025 | 4,373 | **+183.58** | +188.17k | **+182.15** | **+137.49** | −17.85% |
| F130 inclusion/swap | 1,288 | 5,660 | +150.92 | +194.39k | +128.68 | +109.20 | −15.16% |
| F130 gain/swap | 1,501 | 6,077 | +128.07 | +192.23k | +97.69 | +77.56 | −20.22% |
| F130 MDA, its full-HPO refresh | 1,339 | 5,489 | +148.10 | +198.31k | +111.16 | +87.44 | **−14.32%** |

The frozen F130 MDA set improves unit EV by +24.62 bps/trade, but loses 257
portfolio entries, 1,076 admissions, 15.62k total bps, and 3.08 percentage
points of drawdown versus F72. Its HPO refresh increases participation but
loses the unit-EV, total-bps, and worst-week advantages. The other two 130-field
sets do not improve the retained F72 economics. Therefore **F72 remains the
feature contract**: it best fulfils the joint precision-and-preservation role.

The 130-field HPO result is nevertheless preserved for research: depth 3,
learning rate 0.082539, feature fraction 0.771803, bagging fraction 0.849006,
L2 0.145854, random strength 1.908544. It improved the MDA set's own
five-fold ScoreStable to 1.666, but still did not beat F72's 1.714.

## Matched Base → Meta → MC1 diagnostic

This is a **two-month short-warm-up diagnostic**, not a replacement for the
required six-month MC1 evaluation. Both heads use the same Router50 identities,
same canonical policy labels, same 50-bps admission floor, same inherited
portfolio constraints, and a two-month strictly earlier MC1 history. The Meta
input is the same `under_atr1__timestamp` head and has demotion/mapping authority
only; auctions remain ordered by Base rank.

| Variant | Policy-ordinal entries / EV | Raw-bps entries / EV | Raw-bps delta |
|---|---:|---:|---:|
| Base only | 871 / +147.39 bps/trade | 1,314 / **+156.09** | +443 entries; **+8.70 bps/trade**; +76.73k net bps |
| Base + under Meta | 1,451 / +117.98 bps/trade | 1,282 / **+158.96** | −169 entries; **+40.99 bps/trade**; +32.61k net bps |

| Variant | Policy control: worst month / week / MaxDD | Raw-bps: worst month / week / MaxDD |
|---|---:|---:|
| Base only | +140.77 / +20.68 bps / −38.36% | +130.45 / **+97.43 bps** / **−21.23%** |
| Base + under Meta | +91.87 / +61.51 bps / −35.71% | **+132.71** / **+98.68 bps** / **−14.78%** |

The Raw-bps Base wins this matched two-month downstream test, particularly after
conservative Meta conditioning. It remains insufficient for promotion because
only June–July 2026 have both strict-OOF Base and four-month strict-OOF Meta
support. There is not yet a six-month common new-Base Meta ledger for a normal
MC1/admission/portfolio test.

## Causality and identity checks

Both diagnostics passed:

- held Base and Meta scores are target-free before the policy join;
- all labels used to train Base, Meta, and MC1 resolve before their applicable
  reserve or held month;
- every Base/Meta candidate ID matches exactly and is inside the frozen Router50
  population;
- daily MC1 shift uses only prior resolved outcomes;
- the Meta head supplies mapping/demotion context only and does not take auction
  rank authority;
- no canonical, live, or exchange artifact was mutated.

## Key artifacts

- Policy-ordinal Base: `data_perp/artifacts/strict_r3_p8u_policyordinal_base_history_20260828_v1/`
- Raw-bps Base: `data_perp/artifacts/strict_r3_p8u_tail125_base_history_20260828_v2/`
- Policy Base adapter: `data_perp/artifacts/strict_r3_p8u_policyordinal_singlehead_downstream_source_20260828_v2/`
- Raw Base adapter: `data_perp/artifacts/strict_r3_p8u_tail125_singlehead_downstream_source_20260828_v1/`
- Policy Meta / diagnostic: `data_perp/artifacts/strict_r3_p8u_policyordinal_meta_selected_aprjul_20260828_v1/`, `data_perp/artifacts/strict_r3_p8u_policyordinal_shortwarm_meta_mc1_diagnostic_20260828_v1/`
- Raw Meta / diagnostic: `data_perp/artifacts/strict_r3_p8u_tail125_meta_selected_aprjul_20260828_v1/`, `data_perp/artifacts/strict_r3_p8u_tail125_shortwarm_meta_mc1_diagnostic_20260828_v1/`

## Extended matched Base → Meta → MC1 result

The repaired label ledger makes a longer comparison possible. This is the
first direct **Base-versus-Base** downstream comparison: it does not compare a
Base model with the Router. Both arms use the exact same P8u Router50
identities, 72 causal Base fields, target-free Base and Meta score receipts,
four-month OOF Meta training, a three-month strictly earlier MC1 mapper,
canonical policy labels, a +50-bps mapper gate, and the same chronological
portfolio constraints.

The first valid Base score is August 2025: the frozen three-month Base fit
requires Router50 scores from April onward. Consequently, the first valid
four-month Meta score is January 2026 and the earliest fully supported mapper
evaluation is April--July 2026. This is a four-month causal evaluation, not a
six-month promotion test.

| Base contract | Mapper input | Accepted entries | Mapper-admitted rows | Net EV/trade | Total net bps | Worst month | Worst week | MaxDD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen policy-ordinal LightGBM | Base only | 2,447 | 7,246 | +142.05 | +347.58k | +114.72 | +40.86 | -35.46% |
| Frozen policy-ordinal LightGBM | Base + under Meta | 2,372 | 7,703 | **+160.66** | +381.09k | **+149.36** | **+78.46** | -30.47% |
| Raw-bps CatBoost tail-125 | Base only | 2,833 | 11,534 | +136.02 | +385.35k | +70.10 | +52.92 | -20.87% |
| Raw-bps CatBoost tail-125 | Base + under Meta | **2,750** | **11,208** | +140.94 | **+387.59k** | +80.48 | +50.84 | **-18.14%** |

For the comparable Base + under-Meta rows, Raw-bps CatBoost adds 378 accepted
trades and +6.50k total bps, with a 12.33 percentage-point lower maximum
drawdown. The policy-ordinal Base is nevertheless **+19.72 bps/trade** better,
and has materially stronger worst-month and worst-week economics. Therefore
the result is split: the raw candidate has a participation/drawdown advantage;
the policy-ordinal control retains the unit-EV and tail-stability advantage.
No Base replacement advances from this result.

### Exact new artifacts

- Raw strict-OOF Base history: `data_perp/artifacts/strict_r3_p8u_tail125_base_history_aug25_jul26_successorlabels_20260828_v1/`
- Raw target-free Base adapter: `data_perp/artifacts/strict_r3_p8u_tail125_singlehead_downstream_source_aug25_jul26_20260828_v1/`
- Raw OOF under-Meta history: `data_perp/artifacts/strict_r3_p8u_tail125_meta_under_jan26_jul26_successorlabels_20260828_v2/`
- Raw four-month mapper/portfolio replay: `data_perp/artifacts/strict_r3_p8u_tail125_normal4m_meta3m_mc1_aprjul_successorlabels_20260828_v1/`
- Policy target-free Base adapter: `data_perp/artifacts/strict_r3_p8u_policyordinal_singlehead_downstream_source_aug25_jul26_20260828_v3/`
- Policy OOF under-Meta history: `data_perp/artifacts/strict_r3_p8u_policyordinal_meta_under_jan26_jul26_successorlabels_20260828_v3/`
- Policy four-month mapper/portfolio replay: `data_perp/artifacts/strict_r3_p8u_policyordinal_normal4m_meta3m_mc1_aprjul_successorlabels_20260828_v1/`

## Superseding independent dual-MC1 Base comparison

The preceding ``normal4m`` table used a single mapper coordinate copied into
the inherited dual-gate columns. It was useful for a preliminary Base-versus-
Base screen, but it was not a valid dual-family architecture. It is therefore
**superseded** by the following target-free reconstruction:

```text
Frozen P8u Router50 identities
  └─ Base-only / BCF family: Base timestamp rank
  └─ Base-plus-Meta / Current family:
       0.75 × Base timestamp rank
     + 0.25 × strict-OOF unexpected-trailing Meta rank
  └─ independent three-month MC1 map for each family
  └─ both maps must be >= +50 bps
  └─ one chronological constrained portfolio, prioritised by BCF MC1 EV
```

The Meta output is persisted target-free before labels join. The Raw-bps Meta
screen was also extended to three predeclared heads: only
``under_atr1__timestamp`` had positive conditional information
(``residual IC = 0.0460``, ``CMI = 0.0448``). The magnitude and adverse-path
heads were negative on the same seven strict-OOF months, so they have no score
authority in this comparison.

| Base contract | Portfolio entries | Dual-MC1 admitted rows | Net EV/trade | Total net bps | Worst month | Worst week | MaxDD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen policy-ordinal LightGBM + under Meta | 2,576 | 6,219 | +128.86 | +331.95k | +101.18 | +70.09 | −36.05% |
| **Raw-bps CatBoost tail-125 + under Meta** | 2,549 | **7,721** | **+160.48** | **+409.07k** | **+122.45** | **+78.50** | **−15.92%** |
| **Raw-bps delta** | −27 | +1,502 | **+31.62** | **+77.12k** | **+21.27** | **+8.41** | **+20.13 pp** |

This is a true Base-versus-Base comparison: both arms use the same frozen
Router50 candidate IDs, source-aligned policy labels, `under_atr1` Meta target,
three-month MC1 fit, +50-bps dual gate, and portfolio constraints. Each MC1
family is fitted separately; no family score or map is numerically reused.

| Month | Policy entries / net bps per trade | Raw-bps entries / net bps per trade | Raw-bps total net bps |
|---|---:|---:|---:|
| 2026-04 | 645 / +150.27 | 676 / **+208.03** | +140.63k |
| 2026-05 | 846 / +101.18 | 722 / **+140.27** | +101.28k |
| 2026-06 | 587 / +149.27 | 552 / **+169.95** | +93.81k |
| 2026-07 | 498 / **+124.10** | 599 / +122.45 | +73.35k |

Raw-bps leads unit EV in three of the four months, and both Base contracts are
positive in every evaluated month. The result is therefore not a single-month
or one-trade effect, though the sample remains too short for promotion.

The earliest valid Base score is August 2025; after the four-month Meta
warm-up and three-month MC1 warm-up, the fully supported evaluation remains
April--July 2026. It is consequently a four-month strict causal diagnostic,
not a production promotion test or a claim of byte-identical live parity.

## Decision and next gate

**Freeze Raw-bps CatBoost tail-125 as the retained P8u Base research winner;
do not change the live stack.** It is the only selected Base contract that
wins the corrected independent dual-MC1 comparison on unit EV, total EV,
worst period, and drawdown.

Promotion still requires six-or-more common post-Meta months under this exact
dual-family construction, then an untouched later forward period. It must
retain the same candidate IDs, source-aligned policy labels, admission rule,
and portfolio state.

### Superseding artifacts

- Raw three-arm Meta OOF ledger:
  `data_perp/artifacts/strict_r3_p8u_tail125_meta_three_arms_jan_jul_successorlabels_20260828_v1/`
- Raw independent dual-MC1 replay:
  `data_perp/artifacts/strict_r3_p8u_tail125_true_dual_mc1_aprjul_successorlabels_20260828_v1/`
- Policy control independent dual-MC1 replay:
  `data_perp/artifacts/strict_r3_p8u_policyordinal_true_dual_mc1_aprjul_successorlabels_20260828_v1/`
- Reusable diagnostic producer:
  `scripts/run_strict_r3_p8u_singlebase_true_dual_mc1_v1.py`
- Canonical research contract:
  `config/strict_r3_p8u_raw_catboost_base_research_canonical_20260828_v1.json`
