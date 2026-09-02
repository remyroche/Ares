# Routed-only Router-50 — Frozen Winning Research Stack Metrics

**Status:** frozen research challenger as of 2026-08-28; no live promotion.
**Evaluation:** the matched layer audit is April–July 2026; the separately
sealed target-free extension now adds signal hours through 2026-08-27.  All
results are long-only and rich-policy net is joined only after target-free
scores have been persisted.

## 2026-08-28 matched layer audit — current frozen contracts

**Status:** research evidence only; no live configuration, model, admission
floor, execution policy, or exchange process was changed.  This section is
the current comparison after the P8U/F72/Under feature-selection work.  It
uses immutable target-free score receipts, joins rich-policy outcomes only
after scoring, and uses the frozen 15-minute SimplePolicyOptimiser policy with
smooth capital protection and the fixed 100-bps **round-trip** cost exactly
once.

The supporting receipt is
`data_perp/artifacts/strict_r3_p8u_full_layer_audit_20260828_v8/`.  It
replaces the earlier exploratory v1–v7 layer-audit outputs; those directories
are not evidence.

### Coverage and causal boundary

| Layer | Strict OOF score interval | Rows | Evaluation use |
|---|---|---:|---|
| P8U Router | Apr-2025–Jul-2026 | 1,824,286 | Exact candidate-population recall audit |
| F72 Base | Mar-2025–Aug-2026 | 1,024,779 | Full Base tail diagnostics through July; August extension is target-free scored |
| F72 + Under F120 | Aug-2025–Aug-2026 | 749,840 | Full Meta tail diagnostics through July; August extension is target-free scored |
| Dual MC1 maps | Nov-2025–Aug-2026 | 581,992 | Three prior resolved months are the strict warm-up |

### August 2026 incremental extension — signals through 27 August

The historical 15-minute cache was refreshed append-only, then a 170-symbol,
target-free candidate grid was materialised before any outcome source was
opened.  P8U retained exactly 55,080 Router50 rows (648 decision timestamps),
including the terminal 28-August 00:00 decision.  The declared August-27
evaluation cutoff removes that final decision, leaving 54,995 Router50 rows
over 647 timestamps.  Rich-policy labels were materialised afterwards: all
55,080 retained identities have a valid H12 15-minute path and the 100-bps
cost exactly once.

The F72 model's fit-time imputation contract remains active.  Its jointly
observed 72-field coverage is 82.38% across August; the main residual
historical-state gap is `bars_in_high_vol_state_log_norm` (80.17% for
August 1--14, 91.09% after the refresh).  Under F120 has 96.34% joint
coverage and no missing selected field.  Thus the extension is valid under
the frozen scorer, but should not be oversold as a fully observed-input
promotion period.

The extension is retrospective reconciliation evidence: it was generated
after this research configuration existed and is not untouched promotion
evidence.  The complete receipt is
`data_perp/artifacts/strict_r3_p8u_f72_underf120_dual_mc1_nov25_aug27_20260828_v1/`.

### August-27 cutoff-aware admission and portfolio replay

The terminal replay now explicitly excludes decision timestamps at or after
`2026-08-28T00:00:00Z`.  It uses the existing rich policy, fixed 100-bps
round-trip cost once, independent strict-prequential Current/BCF MC1 maps,
and one global chronological portfolio state.  The frozen operational
research control is **dual MC1 >= +50 bps / two new entries per timestamp**;
the other rows are sensitivity diagnostics, not threshold selection.

| Gate / cap | Raw dual admits / EV | Constrained entries / EV | Total bps | Worst month / week | Max DD |
|---|---:|---:|---:|---:|---:|
| 30 / 1 | 45,136 / +106.56 | 6,751 / +148.47 | +1,002,350 | +89.01 / +26.25 | −31.81% |
| 30 / 2 | 45,136 / +106.56 | 9,126 / +119.06 | +1,086,505 | +64.65 / +32.52 | −42.54% |
| 40 / 2 | 35,837 / +123.61 | 8,916 / +123.20 | +1,098,447 | +65.08 / +36.96 | −43.12% |
| **50 / 2 frozen** | **29,474 / +137.08** | **8,461 / +129.11** | **+1,092,386** | **+68.28 / +39.42** | **−31.56%** |
| 50 / 1 | 29,474 / +137.08 | 6,342 / +153.41 | +972,948 | +88.98 / +26.25 | −31.81% |

At the frozen 50/2 control, every calendar day has at least one entry; one
day has fewer than five entries, nine fewer than ten, and the maximum is 44.
No negative calendar-week wallet close occurs, so weekly Sortino is undefined
rather than infinite; the intraperiod −31.56% maximum drawdown remains the
relevant risk warning.  Full monthly, weekly, daily, and lower-tail metrics
are in `data_perp/artifacts/strict_r3_p8u_f72_underf120_extended_quality_aug27_20260828_v3/`.

### August 1–27 target-free layer extension

The exact cutoff target-free score receipt reports the following
timestamp-local average net bps / `>+50` precision. It is evaluation only:
policy labels were joined after Router, F72 Base, and Under score persistence.

| Tail | F72 Base | Current 75/25 Base/Under |
|---|---:|---:|
| Top 1% | +210.83 / 69.24% | +109.33 / 69.24% |
| Top 2% | +163.63 / 65.53% | +85.81 / 67.31% |
| Top 5% | +93.02 / 57.19% | +52.54 / 62.29% |
| Top 10% | +53.37 / 51.40% | +29.59 / 57.46% |
| Top 15% | +33.89 / 47.87% | +20.50 / 53.79% |

Router50 economic recall is 66.62% / 72.60% / 76.62% / 80.15% above
+50/+100/+150/+200 bps. Under F120 conditional MI given F72 is 0.08081 nats.
Receipt: `data_perp/artifacts/strict_r3_p8u_august01_27_layer_extension_20260828_v1/`.

### Router: Top-50% economic recall — exact Jul-2025–Jul-2026 intersection

| Policy-net opportunity | Prior P8U | Current successor P8U | Delta |
|---:|---:|---:|---:|
| > +50 bps | 73.10% | 71.98% | −1.13 pp |
| > +100 bps | 80.87% | 79.92% | −0.95 pp |
| > +150 bps | 84.91% | 84.19% | −0.72 pp |
| > +200 bps | 87.13% | 86.54% | −0.59 pp |

The successor router is not an economic-recall advance: every matched global
threshold is lower.  Its month-level delta standard deviation is 1.50–1.93
pp, and it is particularly weaker in August 2025 and June 2026.  F72/Base
gains below must not be misdescribed as Router gains.

### Base: F72 precision — exact Apr–Jul-2026 candidate intersection

All cuts are timestamp-local, then averaged over timestamps.

| Tail | Prior routed three-way Base: net bps / >+50 hit rate | F72 Base: net bps / >+50 hit rate | Delta net / hit rate |
|---:|---:|---:|---:|
| Top 1% | +182.05 / 64.02% | +226.76 / 67.34% | **+44.72 / +3.32 pp** |
| Top 2% | +147.10 / 59.41% | +183.43 / 63.21% | **+36.32 / +3.80 pp** |
| Top 5% | +103.67 / 54.84% | +125.44 / 56.83% | **+21.76 / +1.99 pp** |
| Top 10% | +70.73 / 50.94% | +82.73 / 51.56% | **+12.00 / +0.63 pp** |
| Top 15% | +46.83 / 48.26% | +56.37 / 48.17% | +9.55 / −0.08 pp |

F72 is a genuine tip-ranking improvement through Top-10%; it is not a broad
hit-rate improvement past that boundary.

### Meta: Under F120 incremental information and trade-off

Under F120 conditional MI given F72 is **0.1384 nats** on the exact matched
panel, versus **0.0330 nats** for the previous T6/T9 coordinate given its
three-way Base.  It therefore has substantially more non-redundant
information.  Its 75/25 Current blend is principally a *reliability* change:
it raises hit likelihood while reducing raw rich-policy net bps at the
extreme tail.

| Tail | F72 Base bps | F72+Under bps | Δ bps | Δ >+50 hit rate | Δ positive hit rate |
|---:|---:|---:|---:|---:|---:|
| Top 1% | +226.76 | +156.69 | −70.08 | +4.55 pp | +3.12 pp |
| Top 2% | +183.43 | +132.37 | −51.06 | +6.47 pp | +4.25 pp |
| Top 5% | +125.44 | +95.66 | −29.78 | +8.22 pp | +5.51 pp |
| Top 10% | +82.73 | +65.20 | −17.53 | +8.55 pp | +5.94 pp |
| Top 15% | +56.37 | +46.25 | −10.12 | +8.01 pp | +5.95 pp |

Against the previous T6/T9 Current coordinate, F72+Under is weaker at
Top-1/2 (−32.40/−13.95 bps) but stronger from Top-5 onward (+10.03/+18.31
bps at Top-5/10) and has +3.59 to +5.81 pp higher `>+50` hit rate across
Top-1–5.

### MC1 conversion — independent Current and BCF maps

The current map was retrained from the new Base/Meta output panel and
reproduced its frozen dual-prediction ledger byte-for-byte.  Below are the
unconstrained, dual-admitted candidates; the portfolio has not yet imposed
cross-time capacity/concurrency.

| Dual MC1 floor | Current: rows / EV / >+50 hit | Legacy-live: rows / EV / >+50 hit |
|---:|---:|---:|
| 30 bps | 42,210 / +107.09 / 54.21% | 28,811 / +119.70 / 57.21% |
| 40 bps | 33,248 / +125.25 / 55.99% | 24,581 / +131.52 / 58.90% |
| 50 bps | 27,382 / +137.91 / 57.25% | 20,966 / +144.96 / 60.76% |

At Top-2/5/10% per timestamp, current BCF-MC1 gives
**+193.04 / +134.91 / +91.48 bps** and **60.73% / 54.97% / 50.72%** `>+50`
precision.  The current map is more active but lower-unit-quality than
legacy before portfolio constraints.

### Portfolio conversion — one chronological global portfolio

All arms use the same rich policy and normal 7× / 10%-margin-slot / eight
concurrent-position limits.  This is the requested capacity sweep, not an
ex-post global-tail table.

| +50-bps dual gate, max new entries/timestamp | Current entries / EV / total bps / DD | Legacy entries / EV / total bps / DD |
|---:|---:|---:|
| 1 | 5,733 / +156.44 / +896,894 / −25.42% | 4,445 / +156.78 / +696,907 / −26.20% |
| 2 | 7,600 / +131.31 / +997,959 / −30.06% | 5,919 / +138.12 / +817,520 / −28.27% |
| 3 | 7,852 / +127.58 / +1,001,721 / −31.65% | 6,280 / +136.57 / +857,677 / −25.11% |
| 4 | 7,882 / +126.68 / +998,479 / −32.24% | 6,356 / +135.81 / +863,180 / −26.68% |

The current stack has a credible total-contribution improvement, especially
at one entry/timestamp (+200.0k bps with essentially unchanged unit EV), but
the selected two-entry configuration loses 6.81 bps/trade, 30.20 bps in the
worst month, 27.34 bps in the worst week, and 1.79 percentage points of
drawdown versus the floor-matched legacy control.  It is therefore **not a
promotion** on this audit alone.

For current `50 bps / two entries`, the 40-week replay has no negative weekly
net-return observation, so Sortino is explicitly **undefined**, rather than
reported as infinity.  Weekly Q5 wallet return is +118.56%, median absolute
deviation 186.22%, and standard deviation 483.43%; these are highly
leveraged simulated-wallet diagnostics, not a claim of low economic risk.

### Month-by-month constrained replay — +50 bps / two new entries

`EV` is net rich-policy bps/trade. `DD` is the maximum drawdown inside that
calendar month’s global-portfolio equity segment.

| Month | Current: trades/day / EV / DD | Legacy-live: trades/day / EV / DD |
|---|---:|---:|
| Nov-25 | 30.87 / +147.61 / −30.06% | 24.63 / +159.09 / −15.92% |
| Dec-25 | 26.29 / +95.20 / −22.30% | 22.39 / +98.48 / −18.09% |
| Jan-26 | 27.45 / +140.49 / −23.83% | 25.45 / +113.38 / −24.82% |
| Feb-26 | 16.50 / +170.74 / −17.99% | 19.25 / +161.97 / −25.21% |
| Mar-26 | 29.48 / +167.71 / −19.40% | 28.03 / +159.01 / −13.84% |
| Apr-26 | 28.50 / +167.66 / −21.15% | 26.23 / +155.51 / −28.27% |
| May-26 | 31.58 / +104.26 / −21.96% | 23.45 / +116.27 / −9.26% |
| Jun-26 | 25.07 / +153.08 / −15.16% | 13.03 / +150.73 / −13.57% |
| Jul-26 | 33.74 / +68.28 / −16.13% | 12.39 / +132.34 / −9.43% |

The current stack is economically stronger than legacy in Jan–Apr and Jun,
but its expanded participation is low quality in July and has worse local
drawdown in Nov/Dec/May.  This reinforces the non-promotion decision.

### August 1–27 constrained extension — +50 bps dual gate / two entries

This is the same F72 + Under, independent Current/BCF MC1, dual-admission,
and one chronological portfolio replay.  The table is keyed by decision day;
the final 00:00 UTC decision generated from the August 27 signal hour is in
the full-month receipt but excluded here so every row is a complete UTC day.

| Metric | August 1–27 |
|---|---:|
| Scored Router50 rows | 54,995 |
| Dual-MC1 admitted rows | 2,092 |
| Shared-portfolio entries | 861 |
| Net EV/trade | +109.67 bps |
| Total net bps | +94,427 |
| Days with no entry | 0 / 27 |
| Best / worst active-day EV | +470.54 / −81.15 bps |
| August-only max drawdown (including the final Aug-28 00:00 decision) | −28.12% |

| Day | Dual admitted | Portfolio entries | Net EV/trade (bps) | Total net bps |
|---|---:|---:|---:|---:|
| Aug 01 | 57 | 30 | −23.14 | −694 |
| Aug 02 | 51 | 28 | +136.86 | +3,832 |
| Aug 03 | 60 | 25 | +96.13 | +2,403 |
| Aug 04 | 59 | 29 | +11.17 | +324 |
| Aug 05 | 53 | 24 | +49.72 | +1,193 |
| Aug 06 | 58 | 34 | +73.38 | +2,495 |
| Aug 07 | 64 | 37 | +91.57 | +3,388 |
| Aug 08 | 77 | 30 | +153.52 | +4,606 |
| Aug 09 | 72 | 30 | +143.10 | +4,293 |
| Aug 10 | 67 | 36 | −6.61 | −238 |
| Aug 11 | 65 | 28 | +102.49 | +2,870 |
| Aug 12 | 53 | 30 | +21.64 | +649 |
| Aug 13 | 59 | 31 | +53.93 | +1,672 |
| Aug 14 | 80 | 28 | +57.17 | +1,601 |
| Aug 15 | 78 | 29 | +40.08 | +1,162 |
| Aug 16 | 78 | 31 | +21.93 | +680 |
| Aug 17 | 73 | 38 | −81.15 | −3,084 |
| Aug 18 | 78 | 32 | +70.36 | +2,251 |
| Aug 19 | 77 | 33 | +183.75 | +6,064 |
| Aug 20 | 72 | 34 | +290.67 | +9,883 |
| Aug 21 | 85 | 38 | +466.77 | +17,737 |
| Aug 22 | 76 | 38 | +470.54 | +17,881 |
| Aug 23 | 120 | 37 | +145.56 | +5,386 |
| Aug 24 | 120 | 36 | +255.04 | +9,182 |
| Aug 25 | 120 | 35 | −62.78 | −2,197 |
| Aug 26 | 120 | 28 | −45.99 | −1,288 |
| Aug 27 | 120 | 32 | +74.27 | +2,377 |

The extension carries the full evaluated period (Nov-2025 through the
August-27 signal) to 8,463 shared-portfolio entries at +129.11 bps/trade and
+1,092,636 total net bps.  Relative to the pre-August receipt it adds 863
entries and +94,677 net bps, but reduces unit EV from +131.31 to +129.11 bps,
and lowers worst-week EV from +50.44 to +39.42 bps.  That is confirmation of
positive contribution, not a basis to relax the frozen +50-bps gate.

### Remaining oracle opportunity

The post-hoc top-2 oracle on the matched current candidate panel remains
large: **+652.27 bps** at top 2%, **+453.42 bps** at top 5%, and **+338.88
bps** at top 10% per timestamp.  This is a non-tradeable ceiling, but it
shows that the central bottleneck is conversion/selection after a good F72
tip signal—not the absence of favorable paths.

### Reproduction

`scripts/report_strict_r3_p8u_layer_comparison_v1.py` produces the immutable
v8 receipt.  It is target-free until the canonical rich-policy label ledger
is joined, uses only prior-resolved labels for MC1 fitting, and does not train
models, access the exchange, or mutate live state.

## Decision status

This is the retained research control for all new challenger work.  The
complete frozen contract is Router50 -> routed-only three-way Base -> T6/T9
consensus -> router-aware dual Current/BCF MC1 -> dual +50-bps admission ->
BCF mapped-EV priority -> one chronological constrained portfolio -> frozen
rich 15-minute policy.

The post-freeze 150-field full-universe add-back challenger is also rejected.
It passed its full-support HPO screen, but its independent Apr--Jul
target-free forward score had `S_stable=0.81211` versus `0.81223` for the
HPO-tuned frozen 30-field control. It improved only the Q25/worst fold and
lost mean routing utility, R50, and R100, so it did not advance to consensus,
MC1, admission, or portfolio replay. The exact causal receipts and the
single-cap HPO repair are recorded in
`STRICT_R3_ROUTED_ONLY_RECALL_STACK_HANDOVER_20260826.md`.

The newer single-Base XENDCG finalists are not promotions.  Their strongest
result, F3, reached +103.81 bps/trade in a June–July fixed-R/U mini-MC1 screen;
the original full ET50 contract on the identical candidate/label period was
+154.49 bps/trade.  Their full ET50 compatibility result also trails the
matched ET50 control (+132.94 versus +139.73 bps/trade).  These experiments
remain forward challengers only; see
`ROUTER_SINGLE_BASE_XENDCG_DOWNSTREAM_20260827.md` for the qualification.

The frozen designation is an offline research decision, not a statement of
live equivalence.  The April–July period was used for selection and any
replacement requires later untouched evidence under the same candidate,
policy, admission, and portfolio contract.

## Frozen selected contract

```text
full causal universe
→ exact timestamp-local P8u top-50% router
→ routed-only enhanced three-way base (no numeric router input)
→ routed-only T6/T9 consensus heads (no numeric router input)
→ Current + BCF MC1 maps (one router primary rank input)
→ both MC1 expected EVs >= +50 bps
→ BCF mapped-EV priority; one chronological global portfolio
→ maximum two new entries per timestamp
→ frozen rich 15-minute policy
```

The router, base, T6/T9 heads, and MC1 maps are all strict prequential. The router only reaches MC1 as the distinct `router_primary_rank`; the previous three router aliases were byte-identical and are explicitly excluded.

## Layer waterfall

`Top-2` is local per decision timestamp; it is a score-quality diagnostic. Only the final row applies the causal dual-MC1 gate and shared chronological portfolio constraints.

| Stage | Rows | Top-2 EV/trade | Top-2 total bps | Precision > +50 bps | Portfolio entries | Portfolio EV/trade | Portfolio total bps |
|---|---:|---:|---:|---:|---:|---:|---:|
| Enhanced base, full universe — diagnostic | 483,772 | +137.76 | +804,785 | 58.59% | — | — | — |
| Selected routed-only enhanced base | 242,739 | +106.62 | +622,877 | 51.52% | — | — | — |
| Oracle within routed set — post-hoc ceiling only | 242,739 | +655.30 | +3,828,244 | 99.97% | — | — | — |
| Selected router-free T6/T9 BCF score | 242,739 | +146.88 | +858,083 | 59.29% | — | — | — |
| Dual MC1 >= +50, before cross-time constraints | 9,008 | +217.08 | +1,004,054 | 67.57% | — | — | — |
| **Dual MC1 >= +50 plus shared portfolio** | **9,005 admitted** | — | — | — | **2,648** | **+163.98** | **+434,216** |

At the selected setting, T6/T9 recover +40.26 bps of local top-2 EV over the routed-only base. Dual MC1 raises the pre-portfolio local top-2 EV by +70.20 bps. Capacity and concurrency reduce it to +163.98 bps/trade in the executable research replay.

## MC1 input and threshold comparison

Both columns use the exact same routed-only base and router-free T6/T9 heads. `Router rank` means one primary router-rank feature in the two MC1 maps only.

| Dual-MC1 floor | MC1 input | Entries | EV/trade | Total bps | Worst month | Worst week | Max DD | Sortino |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 30 | no router rank | 3,236 | +133.35 | +431,506 | +97.20 | +70.98 | −31.26% | 463.08 |
| 30 | router rank | 3,266 | +135.86 | +443,713 | +97.87 | +70.97 | −22.63% | 383.07 |
| 40 | no router rank | 2,994 | +141.09 | +422,409 | +102.22 | +75.31 | −22.12% | 416.90 |
| 40 | router rank | 2,950 | +149.92 | +442,261 | +106.90 | +82.97 | −21.83% | 631.97 |
| 50 | no router rank | 2,688 | +155.30 | +417,457 | +109.34 | +75.89 | −22.34% | 383.92 |
| **50** | **router rank — selected** | **2,648** | **+163.98** | **+434,216** | **+109.54** | **+78.30** | **−16.83%** | **2,102.93** |

The selected router-rank MC1 variant trades 40 fewer entries than the no-rank 50-bps control, but adds +8.67 bps/trade and +16.8k total bps while improving drawdown by 5.51 points. Neither exact-MC1 variant has an entry-free day in this evaluation window.

## Portfolio conversion receipt

This makes the requested no-cross-time-portfolio versus shared-portfolio comparison explicit. The first entry column applies the dual gate and permits at most two selections independently at every timestamp. The second applies the single chronological portfolio state and all normal capacity constraints.

| Floor | MC1 input | Admitted rows | Timestamp-local entries | Local EV/trade | Local total bps | Shared-portfolio entries | Portfolio EV/trade | Portfolio total bps |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 30 | no router rank | 12,020 | 5,015 | +188.11 | +943,378 | 3,236 | +133.35 | +431,506 |
| 30 | router rank | 12,550 | 5,110 | +202.95 | +1,037,052 | 3,266 | +135.86 | +443,713 |
| 40 | no router rank | 10,249 | 4,755 | +196.54 | +934,525 | 2,994 | +141.09 | +422,409 |
| 40 | router rank | 10,558 | 4,794 | +214.35 | +1,027,596 | 2,950 | +149.92 | +442,261 |
| 50 | no router rank | 8,856 | 4,428 | +207.17 | +917,351 | 2,688 | +155.30 | +417,457 |
| **50** | **router rank — selected** | **9,005** | **4,498** | **+223.22** | **+1,004,054** | **2,648** | **+163.98** | **+434,216** |

The portfolio state is the material execution constraint: at the selected 50-bps gate it converts +223.22 local EV/trade into +163.98 bps/trade while preserving a positive result in every held month.

## Remaining oracle opportunity after admission

This is a post-hoc ceiling only: within each causally admitted timestamp, it selects the two rows with the best eventual rich-policy outcome. It measures opportunity left after the admission gate, not a tradeable rule.

| Floor | Admitted timestamps | Oracle top-two rows | Timestamp-local oracle top-two EV | Oracle total bps | Oracle worst-month EV |
|---:|---:|---:|---:|---:|---:|
| 30 | 2,752 | 5,110 | +316.53 | +1,690,297 | +202.40 |
| 40 | 2,652 | 4,794 | +313.24 | +1,575,823 | +203.55 |
| **50** | **2,564** | **4,498** | **+303.92** | **+1,449,353** | **+199.74** |

Thus, after the selected +50-bps gate, the auction/order stage captures +223.22 bps/trade of a +303.92-bps timestamp-local post-admission ceiling; the shared chronological portfolio realizes +163.98 bps/trade. The remaining research headroom is selection/conversion inside an already admitted set, not broader admission relaxation.

## Input-ablation decisions

| Layer | Selected input treatment | Reason |
|---|---|---|
| Base | Router-gated population; no router rank feature | Router rank improved raw base ranking, but did not improve final downstream portfolio results. |
| T6/T9 consensus | Router-gated population; no router rank feature | Adding router rank lost 182 entries, 7.04 bps/trade, 45.9k total bps, and risk-adjusted performance at the 50-bps final replay. |
| MC1 | One `router_primary_rank` feature | At the 50-bps gate it improves EV/trade, total bps, worst-month EV, worst-week EV, and drawdown. |

## Nearest historical comparator

The nearest older matched B0/T6/T9 control recorded 1,847 entries, +166.97 bps/trade, +308,398 total bps, +150.99 worst-month EV, and +106.96 worst-week EV. The selected routed-only stack has 801 more entries and +125,818 total bps but 2.99 bps/trade less and weaker worst-period means. It is not live-equivalent and is retained as research evidence only.

## Matched legacy-live score-family comparison

This is the valid live-baseline comparison for the same April–July 2026 rich-policy outcome ledger. It starts from the sealed legacy current-v5 and BCF score panels, reads only their target-free MC1 inputs, then refits each MC1 map strictly prequentially on the reconciled rich-policy labels. The common dual gate uses BCF mapped EV as priority and the same 7x / 10%-slot / two-new-entries / eight-position portfolio engine.

It is **not byte-identical to the currently deployed map state**: re-fitting the maps on the reconciled rich-policy ledger is necessary to compare outcomes under one execution contract. The frozen legacy score families themselves remain unchanged.

| Arm | Dual floor | Shared scored rows | Portfolio entries | Net EV/trade | Total net bps | Worst month | Worst week | Max DD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Legacy BCF/current score families | 30 bps — live operating floor | 99,017 | 2,805 | +122.68 | +344,110 | +104.25 | +72.89 | −19.82% |
| Legacy BCF/current score families | 50 bps — floor-matched control | 99,017 | 2,289 | +137.56 | +314,871 | **+112.89** | +76.71 | −24.86% |
| **Selected Router-50 routed-only stack** | **50 bps** | **242,739 routed rows** | **2,648** | **+163.98** | **+434,216** | +109.54 | **+78.30** | **−16.83%** |

Against the floor-matched legacy control, the selected stack adds 359 entries, +26.42 bps/trade, and +119,345 total bps. It improves worst-week EV by +1.60 bps and max drawdown by 8.03 percentage points; worst-month EV is 3.35 bps lower. Against the live 30-bps floor, it trades 157 fewer entries but adds +41.30 bps/trade, +90,106 total bps, +5.29 bps worst-month EV, +5.42 bps worst-week EV, and 2.99 points of drawdown improvement.

## Receipts

| Purpose | Artifact |
|---|---|
| Final routed-only MC1-rank stack | `data_perp/artifacts/strict_r3_router50_baseN_metaN_mc1R_routedonly_20260826_v1/` |
| Router-free MC1 control | `data_perp/artifacts/strict_r3_router50_baseN_metaN_mc1N_routedonly_20260826_v1/` |
| MC1 report | `data_perp/artifacts/strict_r3_router50_routedonly_mc1_input_metrics_20260826_v2/` |
| Meta input report | `data_perp/artifacts/strict_r3_router50_routedonly_meta_input_metrics_20260826_v5/` |
| Base input report | `data_perp/artifacts/strict_r3_router50_base_variant_metrics_distinct_router_20260826_v7/` |
| Full waterfall | `data_perp/artifacts/strict_r3_router50_routedonly_final_waterfall_20260826_v5/` |
| Matched legacy score-family replay | `data_perp/artifacts/strict_r3_legacy_live_dual_reconciled_rich_portfolio_aprjul_20260826_v1/` |

See `docs/STRICT_R3_ROUTED_ONLY_RECALL_STACK_HANDOVER_20260826.md` for targets, HPO, feature-selection lineage, policy parameters, causality evidence, and reproduction paths.
