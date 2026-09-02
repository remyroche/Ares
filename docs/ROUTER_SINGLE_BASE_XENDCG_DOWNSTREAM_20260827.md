# Router50 single-Base XENDCG downstream assessment — 2026-08-27

## Scope and decision

This is an offline, long-only development comparison.  It assesses the two
feature-selected XENDCG Base variants downstream because both retained useful
Top-10/20 recall despite failing the original extreme-tip guard.  It does not
change live inference, exchange execution, or the frozen live bundle.

The matched executable-research chain is identical for every arm:

```text
target-free frozen Router50 identities
→ one strict-OOF Base scores every routed candidate
→ fixed R residual-magnitude head + fixed U unexpected-upside head
→ separate strict prequential Current and BCF mini-MC1 maps
→ both mapped EVs >= +50 bps
→ one chronological constrained portfolio
```

The R/U and MC1 probes use March–May 2026 for warm-up and June–July 2026 for
the held portfolio.  The policy label has complete coverage for selected rows;
no invalid outcome is converted into an economic failure.

**Current decision:** no candidate is promoted to the live stack.  F2 is
weaker in the intended fixed-R/U mini-MC1 funnel; F1's HPO refresh is also
weaker there.  F3 is the only remaining intended-target challenger with a
material mini-MC1 improvement.  It still requires the exact final
`Base -> R/U -> full dual-MC1 -> portfolio` replay before it can be rejected
or advanced.  The full-ET50 table below is a separately useful compatibility
diagnostic, not that final R/U architecture.

## Recall condition for the exception

| Base feature contract | Recall of top-50 opportunity @ Top-10 | @ Top-20 | Recall of top-100 opportunity @ Top-10 | @ Top-20 |
|---|---:|---:|---:|---:|
| F1/MDA90 | 17.97% | 31.50% | 22.16% | 36.43% |
| F1/F72 comparison | 17.98% | 31.67% | 22.07% | 36.67% |
| F3/MDA25 | 18.74% | 32.66% | 23.28% | 38.16% |
| F3/F72 comparison | 19.62% | 34.69% | 24.21% | 40.54% |

F1 preserves essentially all broad recall.  F3 gives up 0.9–2.4 percentage
points but remains sufficiently broad to test downstream, where it later
shows a risk-efficiency trade-off.

## Matched intended-target R/U dual-MC1 selection screen, June–July 2026

All economics are canonical rich-policy net bps/trade after the same dual
admission and one shared chronological portfolio; selected-row coverage is
100%.  This is the target-correct R/U stage: F1 uses `T3 ATR`, while F2 and
F3 use `T2 sqrt-ATR`.  R and U are strictly-OOF, fixed probes; the Current and
BCF MC1_d2 maps are fitted prequentially with the same frozen +50-bps dual
gate.  The held period is necessarily June--July because the F1/F2/F3 Base
ledgers begin in November 2025, leaving March--May as their first three
complete R/U/MC1 warm-up months.

| Arm / actual Base target | Entries | Dual-MC1 admitted | Net EV/trade | Total net bps | Worst month | Worst week | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|
| ET50 50/50 Base control | 1,652 | 5,098 | +96.63 | +159,636 | +73.55 | **+54.11** | -18.61% |
| F1/MDA90 / `T3 ATR`, rank-XENDCG | 1,816 | 6,587 | +96.82 | +175,825 | +64.65 | +45.65 | -16.23% |
| F2/MDA25 / `T2 sqrt-ATR`, LambdaRank | 1,858 | 7,476 | +81.37 | +151,186 | +52.46 | +27.09 | -21.24% |
| F3/MDA25 / `T2 sqrt-ATR`, rank-XENDCG | 1,618 | 6,028 | **+103.81** | +167,957 | **+75.23** | +50.01 | **-11.87%** |
| F1/MDA90 HPO / `T3 ATR`, rank-XENDCG | 1,668 | 5,861 | +93.13 | +155,333 | +43.63 | +27.64 | -17.84% |

### Deltas against the matched ET50 Base/R/U diagnostic

| Arm | Entries | Admitted | EV/trade | Total net bps | Worst month | Worst week | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|
| F1/MDA90 | +164 | +1,489 | +0.19 | +16,189 | -8.90 | -8.46 | +2.39pp |
| F2/MDA25 | +206 | +2,378 | -15.26 | -8,450 | -21.09 | -27.02 | -2.63pp |
| F3/MDA25 | -34 | +930 | +7.17 | +8,321 | +1.68 | -4.10 | +6.74pp |
| F1/MDA90 + local HPO | +16 | +763 | -3.51 | -4,304 | -29.92 | -26.47 | +0.77pp |

F3 is the only intended-target candidate that materially improves both
EV/trade and total economics in this screen while also improving drawdown. It
does give up 4.10 bps in the worst week; it is therefore a forward challenger,
not a promotion. F1 is effectively flat per trade and weakens both worst
month and week; F2 and the F1-HPO refresh are rejected before any new full
architecture work.

### Benchmark reconciliation

The diagnostic above is **not** the earlier full ET50 stack.  It preserves the
same Router50 identities and 50/50 efficiency/timing Base, but substitutes the
experimental R/U heads and mini-MC1 mapper for the existing cap80/cap120
consensus, generic-correctness layer, and their full Current/BCF MC1 contract.
It is therefore valid only for comparing the single-Base experiments to each
other.

On the identical 121,594 June--July candidate IDs and rich-policy labels, the
original full ET50 contract accepted 1,251 portfolio entries at **+154.49
bps/trade** (+193,271 total bps); this R/U diagnostic accepted 1,652 at
**+96.63 bps/trade** (+159,636 total bps).  The diagnostic’s lower result is
an architecture/admission change, not a period or label-coverage change.  In
particular, it admits 2,315 candidates the full contract rejects (+23.49
realised bps on average) and rejects 1,727 that the full contract admits
(+116.69 realised bps on average).

The earlier stored headline of **+155.36 bps/trade** is also not an
identical-window number: despite its stale `2026_marjul` label, its decision
ledger runs from 2026-04-01 through 2026-07-31.  The fresh full-contract
control below deliberately begins on 2026-05-01, because that is the first
common held window available to the single-Base adapters.  The excluded April
slice contains 707 accepted entries at approximately **+201.04 bps/trade**.
Consequently the fresh May--July result of +139.73 bps/trade is the expected
period-specific control, not a regression in the ET50 implementation.

No single-Base arm in this table can be promoted against the production-like
ET50 benchmark until it is replayed through the same full downstream contract.

## Full ET50-contract policy-ordinal companion diagnostic, May--July 2026

This is the required follow-up to the diagnostic table.  It uses one fresh,
identical-window replay for every arm:

```text
Router50 identities
→ frozen Base coordinate (ET50, F1, or F3)
→ cap80 ordinary + cap120 equal-month consensus
→ generic correctness
→ full strict-prequential Current and BCF MC1 maps
→ dual >= +50 bps admission
→ one chronological constrained portfolio
```

Each row here deliberately uses its **policy-ordinal `T0` companion score**
because that is the common score coordinate accepted by the historical ET50
consumer.  F1, F2, and F3 were actually selected on `T3`, `T2`, and `T2`
respectively.  They are supplied through a target-free one-head adapter: all
historical E/T/B0 score slots equal that `T0` companion and all historical
disagreement slots are zero.  It has no E/T/R3 numerical authority and every
Router50 row is retained.  The 120 causal market fields, policy labels,
auction, and dual-MC1 contract are otherwise identical.  All selected outcomes
have 100% coverage.

| Arm | Entries | Dual admitted | Net EV/trade | Total net bps | Worst month | Worst week | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full ET50 control | 2,066 | 7,458 | **+139.73** | **+288,686** | +117.07 | **+79.26** | **-13.48%** |
| F1/MDA90 `T0` companion | 1,657 | 4,550 | +138.31 | +229,185 | **+121.79** | +39.94 | -28.17% |
| F2/MDA25 `T0` companion | 1,693 | 4,874 | +135.81 | +229,920 | **+126.32** | +48.21 | -41.14% |
| F3/MDA25 `T0` companion | 1,786 | 5,302 | +132.94 | +237,433 | +117.58 | +30.98 | -30.51% |

Relative to the full ET50 control, F1 loses 409 entries, -1.42 bps/trade,
-59,501 total bps, -39.32 bps on the worst week, and 14.60 percentage points
of drawdown.  F2—the residual-utility finalist—loses 373 entries, -3.93
bps/trade, -58,766 total bps, -31.05 bps on the worst week, and 27.59
drawdown points, despite a +9.25-bps improvement in its best single-month
floor.  F3 loses 280 entries, -6.79 bps/trade, -51,253 total bps, -48.28 bps
on the worst week, and 17.02 drawdown points.  This establishes that merely
substituting the finalists' `T0` companion into the old ET50 schema is not an
improvement.  It **does not** test the intended `T3/T2/T2 -> R/U -> full
dual-MC1` candidate pipelines, so it cannot be used as their final rejection.

## HPO falsification

F1's HPO winner was depth 3, 7 leaves, learning rate 0.05262, 0.61% minimum
data fraction, 0.784 feature fraction, 0.820 bagging fraction, L1 0.00037,
L2 0.417, and minimum gain 0.00035.  It passed its local Base gate, but the
full OOF replay shows why it cannot be adopted:

| F1 comparison | Entries | EV/trade | Total net bps | Worst month | Worst week | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|
| F1/MDA90, fixed R/U | 1,816 | +96.82 | +175,825 | +64.65 | +45.65 | -16.23% |
| F1/MDA90 + HPO, fixed R/U | 1,668 | +93.13 | +155,333 | +43.63 | +27.64 | -17.84% |
| HPO delta | -148 | -3.69 | -20,492 | -21.02 | -18.01 | -1.62pp |

The HPO arm's Base-only stage is stronger (+95.95 bps/trade and +180,192 total
bps), but this does not survive the residual correction.  The downstream
criterion therefore correctly rejects it rather than optimizing the Base in
isolation.

F3's local-HPO candidate also failed its nine-month full-OOF Base screen, so
it was not allowed into the downstream chain.  A historical receipt issue
(the requested rather than depth-valid leaf count) was corrected and the
effective 7-leaf value is recorded in the Base manifest; it did not affect any
prior score or conclusion.

## Lineage and causality checks

- Base held scores are written target-free before policy outcomes join.
- All Base training labels resolve before the 28-day fold reserve.
- Router50 candidate identity is exact; the router is never a numeric Base
  input and there is no post-Base cutoff.
- Base medians are trained only on each training fold.
- R/U target-free receipts preserve Base score/rank identity exactly.
- Current and BCF MC1 maps fit only on March–May before the June–July held
  months, and both need expected EV of at least +50 bps.
- Portfolio evaluation is chronological with one global state and complete
  policy-outcome coverage for accepted trades.

Primary immutable artifacts:

- `data_perp/artifacts/strict_r3_router_single_base_f1_mda90_full_oof_20260827_v3`
- `data_perp/artifacts/strict_r3_router_single_base_f2_mda25_full_oof_20260827_v1`
- `data_perp/artifacts/strict_r3_router_single_base_f3_mda25_full_oof_20260827_v3`
- `data_perp/artifacts/strict_r3_router_single_base_dual_mini_mc1_et50_control_20260827_v1`
- `data_perp/artifacts/strict_r3_router_single_base_dual_mini_mc1_f1_mda90_20260827_v1`
- `data_perp/artifacts/strict_r3_router_single_base_dual_mini_mc1_f3_mda25_20260827_v1`
- `data_perp/artifacts/strict_r3_router_single_base_full_et50_f2_mda25_20260827_v2`
- `data_perp/artifacts/strict_r3_router_single_base_dual_mini_mc1_f1_mda90_hpo_20260827_v1`
