# Causal Exotic Dynamics MC1 Ablation — 31 August 2026

## Decision

**Retain M0, the frozen paired BCF/current MC1 control.** No causal dynamics
family or predeclared combination is promoted to the canonical or live stack.

The screen is research-only.  It made no exchange calls and changed no live,
canonical, exit, admission, execution, or Geometry/K9 artifact.

## Contract

The feature source is
[`extreme_price_movements/causal_exotic_dynamics.py`](../extreme_price_movements/causal_exotic_dynamics.py).
For a decision at `T`, it consumes only completed 15-minute OHLCV bars strictly
before `T`; the long-horizon features require a contiguous 288-bar (72-hour)
local history.  There is no outcome, model, policy, portfolio, future bar, or
exchange input.  Missing/corrupt local symbol files are explicit unavailable
states, never imputed.

Spectral summaries are canonicalised to 15 decimal places at their producer
boundary.  This removes harmless FFT alignment-level differences (about
1e-16) without changing economic precision, so an untouched prior slice has
one exact persisted representation.

The target-free matrix contains 466,856 exact paired BCF/current candidate
identities, 170 symbols, and 116 fields over 2025-04-01 through 2026-07-31.
The only permanently unreadable local sources are TON and USDT.  All other
availability gaps are row-local warmup/contiguity gaps.

The mapper contract is unchanged from the source-aligned parent policy:

* separate BCF and current-v5 depth-2 HGB maps;
* target: resolved parent-policy net bps;
* specialist target: parent-policy net bps minus the paired frozen-MC1 mean;
* only labels resolved before the held decision month are used;
* dual BCF/current expected EV admission at +50 bps;
* BCF expected EV ranks the fixed global 7x / 10%-slot auction;
* invalid outcomes are excluded before portfolio capacity is consumed.

The initial 2025 discovery period is July–December.  Each held month has a
three-calendar-month prior-only training window.  The recurring field contract
was frozen from those folds before any 2026 result was read.  January–July
2026 is confirmation only.

## Families and frozen 2025 field contracts

| Family | Stable 2025 fields |
|---|---:|
| CP / change-point-Kalman | 7 |
| SP / spectral | 6 |
| WV / wavelet | 8 |
| EN / entropy | 5 |
| DS / distribution | 8 |

The individual field and month-level quality record is
`data_perp/artifacts/causal_exotic_dynamics_assessment_2025_20260831_v2_expanded/`.
Only fields selected in at least half of the prior-only folds entered the
frozen contract.  This retains recurrence rather than one-period IC.

## 2025 strict-OOS decision evidence

The relevant comparator for every head arm is its same-timestamp M0 control;
the head cannot exist in its first prequential month, so it starts in August.

| Arm | Entries | Net EV/trade | Total net bps | Worst month | Worst week | Max DD | Ulcer |
|---|---:|---:|---:|---:|---:|---:|---:|
| M0 control, DS-head support | 2,346 | +191.40 | +449,015 | +162.62 | +110.73 | -37.36% | 6.03% |
| DS specialist head | 2,499 | +194.93 | +487,131 | +138.86 | +71.27 | -37.88% | 5.62% |
| WV raw+head | 2,450 | +190.87 | +467,638 | +143.54 | +68.16 | -37.79% | 6.09% |
| DS + SP heads | 2,481 | +184.40 | +457,485 | +139.92 | +58.15 | -40.86% | 6.11% |
| DS + WV raw+head | 2,426 | +188.67 | +457,712 | +144.12 | +77.77 | -40.37% | 5.81% |
| DS + SP + WV | 2,415 | +189.84 | +458,465 | +145.83 | +77.77 | -40.33% | 5.92% |

DS head is the sole individual arm with a positive EV/trade and total-PnL
increment in the strict-OOS screen (+3.53 bps/trade; +38,116 bps).  Its
downside result is mixed: Ulcer is lower, but maximum drawdown, worst month,
worst week, and time underwater are worse.  It is therefore not a robust
promotion.  Every predeclared addition to DS lowers EV/trade and increases
maximum drawdown in 2025.

## Bounded DS-head geometry falsification

The DS result was also screened across four predeclared, shallow
high-support HGB geometries.  The DS specialist itself, field contract,
labels, admission, auction and replay policy remained fixed; only the mapper
geometry changed.  Geometry selection was restricted to the same strict
2025 OOS period.  This was a fragility test, not further broad HPO.

| Geometry | 2025 entries | 2025 EV/trade | Δ EV/trade vs matched M0 | Δ total bps | 2025 Max DD | 2026 EV/trade | 2026 Max DD |
|---|---:|---:|---:|---:|---:|---:|---:|
| H0 frozen depth-2 | 2,499 | +194.93 | +3.53 | +38,116 | -37.88% | +198.38 | -26.97% |
| H1 depth-1 | 2,416 | +188.45 | -2.94 | +6,291 | -39.62% | +212.51 | -35.11% |
| H2 smoother depth-2 | 2,522 | +190.37 | -1.02 | +31,110 | -37.88% | +198.49 | -26.97% |
| H3 more regularized depth-2 | 2,516 | +195.73 | +4.33 | +43,435 | -37.88% | +193.92 | -40.50% |

H3 is the narrow 2025 winner on the predeclared primary metric, but it does
not establish a robust, risk-improving plateau: its 2025 worst month and week
are lower than the matched control, and its frozen 2026 maximum drawdown is
-40.50% versus the M0 control's -42.94% but materially worse than H0/H2's
-26.97%.  Conversely H0/H2 provide encouraging 2026 downside outcomes but
did not improve 2025 EV/trade.  There is therefore no geometry that dominates
across the selection and confirmation criteria.  The conclusion remains
**retain M0; do not promote a dynamics mapper**.

## Frozen 2026 confirmation (not selection)

| Arm | Entries | Net EV/trade | Total net bps | Max DD | Ulcer | Daily CVaR5 | Time underwater |
|---|---:|---:|---:|---:|---:|---:|---:|
| M0 control | 3,245 | +189.76 | +615,783 | -42.94% | 8.22% | -15.82% | 46.58% |
| DS specialist head | 2,820 | +198.38 | +559,443 | -26.97% | 4.49% | -8.97% | 41.35% |
| WV raw+head | 2,746 | +199.94 | +549,027 | -30.02% | 4.02% | -9.11% | 39.98% |
| DS + SP + WV | 2,720 | +205.07 | +557,794 | -28.49% | 3.69% | -8.17% | 39.88% |

The confirmation period is encouraging for the DS/WV shape, particularly on
drawdown and downside tail risk, but it reduces total contribution by roughly
58k bps for the joint arm.  More importantly, it cannot repair the weak and
mixed 2025 selection evidence.  It is confirmation, not permission to select
against the discovery record.

For June–July alone, the joint arm is +174.42 bps/trade and +77,267 total bps
versus M0 +158.71 and +76,183, with max DD -12.25% versus -42.94%.  This is a
useful future hypothesis, not a promoted result.

## August boundary

`canonical_sr_e2_mc1_august_extension_inputs_20260831_v1` supplies target-free
August BCF/current score cores only; it deliberately has no matching complete
source-aligned parent-policy outcome ledger for every candidate.  Partial
portfolio decision files cannot substitute for those labels without changing
the candidate/capacity contract.  This study therefore ends at 2026-07-31 and
does not claim an August portfolio confirmation.

## Reproducibility

1. Target-free materialisation:
   `scripts/materialize_causal_exotic_dynamics.py`
   → `data_perp/artifacts/causal_exotic_dynamics_2025train_2026confirm_20260831_v3_expanded/`
2. Strict 2025 quality, nested selection, and specialist probes:
   `scripts/assess_causal_exotic_dynamics.py`
   → `data_perp/artifacts/causal_exotic_dynamics_assessment_2025_20260831_v2_expanded/`
3. Individual family raw/head/raw+head maps and constrained replay:
   `scripts/run_causal_exotic_dynamics_mc1_ablation.py`
   → `data_perp/artifacts/causal_exotic_dynamics_mc1_ablation_2025oof_2026confirm_20260831_v1/`
4. Small predeclared addition funnel:
   `scripts/run_causal_exotic_dynamics_mc1_combo_ablation.py`
   → `data_perp/artifacts/causal_exotic_dynamics_mc1_combo_ablation_2025oof_2026confirm_20260831_v1/`
5. Bounded DS-head geometry falsification:
   `scripts/run_causal_exotic_dynamics_ds_head_geometry_hpo.py`
   → `data_perp/artifacts/causal_exotic_dynamics_ds_head_geometry_hpo_2025oof_2026confirm_20260831_v1/`

Both run manifests attest `no_exchange_calls: true`.  The individual and
combo audit files record every trained fold and every intentionally
unavailable early prequential head fold.
