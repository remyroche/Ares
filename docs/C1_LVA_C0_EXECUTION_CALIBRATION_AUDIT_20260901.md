# C1-LVA / C0 execution and cohort audit

Status: research and operational audit. This document does **not** promote C1
to the exchange-writing contract.

## Exit monitoring

The active v181 monitor invokes the rich parent policy every 30 seconds when
there is a tracked position. It advances policy state on completed one-minute
bars and submits real reduce-only exits. At entry it installs a Kraken
protective stop whose trigger is adjusted so the expected size-aware
liquidation VWAP corresponds to the policy stop. A normal trailing,
smooth-protection, or timeout exit remains a separate reduce-only market
action; it is not reported as a protective-stop fill.

The 2026-09-01 runtime audit found three exchange-writing 30-second monitor
processes, each attached to a different historical state contract. All three
states are currently empty or absent, so no position is presently managed by
more than one process. There is, however, no proven account-wide singleton
between those state contracts. In addition, no live hourly entry producer was
running at the audit time. Therefore the exit implementation is verified, but
the operational claim that there is one coherent *entry-and-exit* live system
is not yet proven; the old monitors must be consolidated or their ownership
made mutually exclusive before a new live release.

The updated execution engine stores the following terminal entry prediction
telemetry on every future close:

```text
execution-adjusted expected EV
raw mapped gross EV
adverse-only delay gap
full spread
entry VWAP impact
microstructure friction and buffer
```

The already-running pre-update monitor retains its imported code; it must be
replaced only by a separately sealed successor before it can emit these new
fields. The exit telemetry records policy threshold, protective-stop VWAP target,
actual exchange fill, and exit-to-policy gap. The implementation is covered by
the focused P8U live parity, asset-limit, terminal telemetry, fee-sidecar, and
calibration tests (24 passing on 2026-09-01).

### 2026-09-01 live-readiness recheck

A read-only Kraken Futures query at `2026-09-01T17:37:25Z` returned zero open
positions. Therefore no actual position is currently unprotected. The active
v181 process continues to run in its already-imported runtime, but a fresh
read-only v181 monitor load correctly fails closed before exchange access:

```text
sealed runtime code hash mismatch: extreme_price_movements/config.py
```

The active v181 execution contract is also stale relative to the checked-out
runtime: `features.py`, `strict_r3_live_execution.py`,
`strict_r3_shadow_portfolio.py`, `run_tp6_sl4_exact170_canonical_consensus.py`,
and `strict_r3_inference_bundle.py` do not match its sealed hashes. The local
Git object pack is corrupt, so the old sealed source cannot be independently
reconstructed from repository history. Do not restart, reseal, or promote this
monitor until those runtime changes are reviewed as one explicit successor and
the account-wide monitor ownership is consolidated. This is fail-closed by
design; it is not evidence that the current source has equivalent semantics.

## Actual predicted-versus-realised EV audit

The immutable close ledger originally omitted confirmed fees and the
execution-adjusted prediction. Two immutable sidecars repair the audit without
rewriting history:

1. `strict_r3_fee_confirmed_execution_sidecar_20260901_v1` matches exact
   contract/time windows to Kraken's account log and books realised PnL, fees,
   and funding. Seven historical trades are fee-confirmed; ambiguous paths are
   excluded.
2. `strict_r3_execution_prediction_recovery_20260901_v1` matches candidate id
   and actual fill time to immutable entry producer receipts. It recovers four
   exact legacy adjusted-EV values. It does not rerun scores or consult exits.

The resulting bucket audit is:

| Adjusted-EV bucket | Confirmed trades | Mean predicted bps | Mean realised net bps |
|---|---:|---:|---:|
| <0 through 50–75 | 0 | — | — |
| 75–100 | 2 | +88.52 | −257.90 |
| 100–150 | 2 | +118.62 | +39.11 |
| >150 | 0 | — | — |

The two populated means happen to be ordered, but this is **not strong
monotonicity evidence**: there are only four observations, two populated
buckets, and no C1-LVA live entry. The audit explicitly returns
`insufficient_sample_for_monotonicity_assessment` until at least 20
fee-confirmed observations span at least three buckets. It must be rerun on
each future close using the terminal telemetry rather than recovered legacy
receipts.

## C0 versus C1-LVA: matched May–July 2026 evidence

This comparison uses exactly 65,656 common candidate IDs, dual BCF/current
MC1 admission at +50 bps, BCF-priority constrained auction, and the historical
15-minute parent-policy outcome panel. It is useful for selection geometry, but
it is **not** the exact-one-minute/+5-minute executable outcome contract.

| Arm | Admitted | Portfolio entries | Net EV/trade | Total net bps | Worst month | Worst week | Max DD |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 frozen control | 3,524 | 1,000 | +157.18 | +157,181.74 | +155.48 | +0.37 | −42.94% |
| C1-LVA | 3,901 | 1,063 | +159.03 | +169,044.64 | +140.63 | +75.42 | −32.79% |

The immutable receipt for this table is
`data_perp/artifacts/p8u_c1_full_coverage_matched_oos_portfolio_mayjul_20260901_v2`.
Its outcome source is explicitly `hourly_ohlc_proxy`, including
`hourly_proxy_trailing` exits. It must not be compared directly with the
exact-one-minute/live-entry contract below.

### Proxy-to-exact reconciliation

The apparent C1 decline is primarily an outcome-materialisation correction,
not a score-model regression. On the same 1,063 proxy-selected C1 candidate
IDs, 1,038 have a valid exact-one-minute outcome:

| Evaluation on the same C1-selected IDs | Trades | Mean net EV/trade |
|---|---:|---:|
| Historical hourly proxy outcome | 1,038 | +159.90 bps |
| Exact 1-minute rich policy; entry at decision +5 minutes | 1,038 | +111.12 bps |
| Exact minus proxy | 1,038 | -48.78 bps |

For the 937 IDs selected by both the proxy and exact portfolio replays, the
same difference is -48.35 bps/trade (+159.92 proxy versus +111.57 exact).
The median same-ID difference is approximately zero, so the mean correction is
concentrated in path-sensitive tail cases--precisely where a coarse OHLC proxy
cannot determine the order of MFE, trailing arming, giveback, protection, and
stop events. The remaining change to the latest exact C1 result (+105.61 bps)
comes from different capacity occupancy/selection after one-minute exits and
the five-minute entry delay. Costs are charged once in both contracts.

C1 is not a pure replacement:

| Cohort, using its own normal constrained portfolio selection | Entries | Realised net EV/trade |
|---|---:|---:|
| C0 selected and also C1-admitted | 784 | +171.66 bps |
| C0 selected but C1-not-admitted | 216 | +104.63 bps |
| C1 selected but C0-not-admitted | 275 | +123.93 bps |

Thus C1 finds a meaningful profitable extension, but it also drops a
materially positive C0-only population. The bounded successor to test under
the exact one-minute/+5-minute contract is:

```text
C0 retains primary admission and auction authority.
C1 may fill only a timestamp at which C0 has no admitted candidate.
The ordinary global portfolio constraints remain unchanged.
```

This prevents C1 from replacing strong C0 choices while preserving its recall
extension. The matching exact-policy replay is now available below. C1 remains
a challenger because this May--July period is selection evidence and there is
still no separately sealed current C1 inference bundle.

## Exact-one-minute / five-minute-entry C0/C1 integration test

The following tests reuse one immutable exact-one-minute rich-parent outcome
panel. Candidates are admitted target-free at +50 bps by both BCF/current MC1
maps before any path is joined; the normal global 7x/10%-slot, two-new,
eight-concurrent, 80%-wallet auction is then replayed. The rich-policy cost is
charged once. This is the appropriate comparison for C0/C1 selection, rather
than the preceding 15-minute proxy.

| Route | Entries | Net EV/trade | Total net bps | Max DD | Sortino |
|---|---:|---:|---:|---:|---:|
| C0 refit core | 959 | **+121.23** | +116,261.66 | **-8.73%** | **0.698** |
| C1 LVA | 1,174 | +105.61 | +123,987.74 | -12.59% | 0.612 |
| C0 primary, C1 fills C0-empty timestamps; C1 dual floor 50 | 1,138 | +112.87 | **+128,442.89** | -10.15% | 0.632 |
| Same, C1 dual floor 75 | 1,003 | +118.93 | +119,288.02 | -8.73% | 0.674 |
| Same, C1 dual floor 100 | 967 | +121.04 | +117,044.31 | -8.73% | 0.696 |
| Hard tier: both-admitted, then C1-only, then C0-only | 1,211 | +105.08 | +127,252.12 | -11.46% | 0.605 |

The hard agreement tier is rejected: its exact selected C1-only and C0-only
trades realise only +54.05 and +55.42 bps/trade respectively, and both are
negative in May. The 15-minute proxy cohort means are therefore not a valid
reason to force C1-only ahead of C0-only in a live auction.

The only defensible C1 combination currently is a *bounded recall extension*:
C0 remains primary, and C1 may fill a C0-empty decision timestamp at a fixed
dual-MC1 floor. A 75-bps C1 floor is the closest compromise here, adding
44 entries and +3,026.36 total bps versus C0 with only -2.30 bps/trade, but
it is a selected May--July result and must not be promoted without later
untouched evidence. Pure C0 remains the control for EV/trade and drawdown.

## August exact-one-minute C0/C1 sanity replay

An offline August 1--18 replay used C0/C1 target-free direct score panels,
dual +50-bps admission before outcome access, exact one-minute rich-parent
exits with five-minute entry delay, and the ordinary chronological portfolio.
The payload-verified minimal-cover reader was explicitly labelled research-only;
it avoids redundant historical conflict scans but does not substitute for a
full source conflict audit.

| Arm | Target-free admitted | Exact path valid | Portfolio entries | Net EV/trade | Total net bps | Max DD |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 42 | 25 | 18 | +92.35 | +1,662.25 | -4.20% |
| C1-LVA | 183 | 86 | 51 | +90.40 | +4,610.40 | -4.20% |

This has the expected *recall* direction for C1, but only 94 of 192 union
rows have valid exact paths; it is neither a fully covered holdout nor
promotion evidence. It does not overturn the May--July result that C0 is the
quality/risk control and C1 should be limited, if used at all, to a bounded
gap-fill challenger.

The predeclared 75-bps C0-primary/C1-gap-fill rule was also applied to this
same partial August panel:

| Route | Portfolio entries | Net EV/trade | Total net bps | Delta EV/trade vs C0 |
|---|---:|---:|---:|---:|
| C0 | 18 | +92.35 | +1,662.25 | — |
| C0 primary + C1 gap-fill >=75 | 36 | +75.97 | +2,734.81 | -16.38 |

The additional 18 August gap-fill entries are therefore materially weaker than
C0. This does not prove that C1 is harmful overall--the exact source coverage
is thin--but it fails the intended later-period confirmation for the selected
May--July gap-fill rule. Do not promote that combination.

## C1-LVA assembly parity

The canonical C1-LVA assembler was rerun without outcomes or exchange access
against its sealed May--July package. It reproduced the BCF and Current mapped
EVs with a maximum absolute delta of `0.0` and reproduced every dual-admission
decision exactly:

| Month | Target-free rows | BCF max delta | Current max delta | Admission equality |
|---|---:|---:|---:|---|
| 2026-05 | 28,092 | 0.0 | 0.0 | exact |
| 2026-06 | 13,300 | 0.0 | 0.0 | exact |
| 2026-07 | 24,264 | 0.0 | 0.0 | exact |

Receipt: `data_perp/artifacts/c1_lva_canonical_parity_20260901_v2_runtime_recheck/receipt.json`.

## Immutable execution-EV calibration refresh

The observed execution-EV bucket report is now refreshable without touching
the trader.  [`refresh_strict_r3_execution_adjusted_ev_calibration.py`](../scripts/refresh_strict_r3_execution_adjusted_ev_calibration.py)
reads only the append-only close ledger and immutable fee-confirmed and
entry-prediction sidecars.  It performs no exchange I/O, state mutation,
feature scoring, admission, portfolio action, or order submission; each run
writes a new immutable receipt.

Latest receipt:
[`refresh_20260901T181715Z`](../data_perp/artifacts/strict_r3_execution_adjusted_ev_calibration_refreshes/refresh_20260901T181715Z/receipt.json).

| Execution-adjusted EV bucket | Fee-confirmed trades | Mean predicted bps | Mean realised net bps |
|---|---:|---:|---:|
| <0 | 0 | — | — |
| 0–25 | 0 | — | — |
| 25–50 | 0 | — | — |
| 50–75 | 0 | — | — |
| 75–100 | 2 | +88.52 | -257.90 |
| 100–150 | 2 | +118.62 | +39.11 |
| >150 | 0 | — | — |

The two populated bucket means are directionally ordered, and the four-row
Pearson correlation is +0.415.  This is **not** strong monotonicity evidence:
the predeclared minimum is 20 fee-confirmed prediction/outcome pairs across at
least three populated buckets.  The receipt correctly reports
`insufficient_sample_for_monotonicity_assessment`; fee-pending, gross-only,
and missing-entry-prediction rows remain excluded.

## User-directed C0/C1 agreement-tier route

The canonical no-order combined inference contract is now
[`C0_C1_AGREEMENT_TIER_CANONICAL_INFERENCE_20260901.md`](C0_C1_AGREEMENT_TIER_CANONICAL_INFERENCE_20260901.md):
`both-admitted -> C1-only/C0-only by selected raw BCF EV`.  Its target-free
assembler preserves both family values and uses the agreement offset only as a
separate portfolio-order key.  It does not contaminate raw expected EV or
execution adjustment.

This is an explicit participation/total-contribution choice, not evidence that
C1 is a pure quality improvement: in the exact May path, C1-only accepted
trades averaged -12.75 bps and C0-only accepted trades -7.21 bps, while the
both-admitted cohort averaged +103.66 bps.  The route remains no-order until
current C0/C1 package parity and a separately sealed exchange release exist.
This proves historical C1 source/mapper parity, not a current-month C1
producer or exchange-writing activation.

## Relevant immutable evidence

- `data_perp/artifacts/strict_r3_execution_adjusted_ev_calibration_20260901_v5/calibration.json`
- `data_perp/artifacts/strict_r3_fee_confirmed_execution_sidecar_20260901_v1/sidecar.json`
- `data_perp/artifacts/strict_r3_execution_prediction_recovery_20260901_v1/sidecar.json`
- `data_perp/artifacts/p8u_c1_full_coverage_matched_oos_portfolio_mayjul_20260901_v2/portfolio_summary.parquet`
- `data_perp/artifacts/p8u_c1_full_coverage_matched_oos_portfolio_mayjul_20260901_v2/monthly_metrics.parquet`
- `data_perp/artifacts/p8u_c1_lva_vs_core_exact1m_parent_mayjul_20260901_v9_all_active_sources_clean/portfolio_summary.parquet`
- `data_perp/artifacts/c0_primary_c1_gapfill_exact1m_mayjul_20260901_v2_floor50/portfolio_summary.parquet`
- `data_perp/artifacts/c0_primary_c1_gapfill_exact1m_mayjul_20260901_v2_floor75/portfolio_summary.parquet`
- `data_perp/artifacts/c0_primary_c1_gapfill_exact1m_mayjul_20260901_v2_floor100/portfolio_summary.parquet`
- `data_perp/artifacts/c0_c1_agreement_tier_exact1m_mayjul_20260901_v1/portfolio_summary.parquet`
- `data_perp/artifacts/causal_sr_c1_lva_august_exact1m_parent_20260901_v2/run_manifest.json`
- `data_perp/artifacts/causal_sr_c1_lva_august_exact1m_parent_20260901_v2/portfolio_summary.parquet`
- `data_perp/artifacts/c0_primary_c1_gapfill_exact1m_august_20260901_v1_floor75/portfolio_summary.parquet`
