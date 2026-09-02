# Causal anchor / forward-filter study — rejected as an MC1 input

## Decision

The retained seven-field **levels/value-area C1** contract remains the only
research challenger.  The causal anchor and forward Kalman/filter state is a
valid source-research artifact, but it is **rejected as an additional MC1
input**.  It must not be added to the canonical or live stack.

The reason is economic rather than mechanical: the selected anchor state
improves source prediction substantially, but it reduces the constrained
portfolio contribution of the already stronger C1 levels/value-area challenger
in the main June--July confirmation window.  Its small August gain does not
reverse that result and August was confirmation-only.

## Contract and causality

The source materialisation is target-free at decision time.  It creates
anchors from confirmed 1h/4h swings, break/reclaim/failed-break structure,
directional volatility shocks, and strictly-prior OI events.  Each anchor has
multiple causal reference prices (event open/close/mid/VWAP/extreme,
structural level, and OI-WAP where available), lifecycle, path, location and
transition state.

The five source variants are:

| Variant | State supplied | 2025 selection score |
|---|---|---:|
| M0 | Market controls only | 0.04479 |
| M1 | Anchor/path state | 0.08211 |
| **M2** | **M1 + transition state** | **0.08369** |
| M3 | M2 + causal forward Kalman state | 0.08294 |
| M4 | M3 + Kalman innovation/transition terms | 0.08208 |

M2 was selected solely on July--December 2025, using a deterministic
source-identity 1-in-20 sample (274,877 rows) fixed independently of labels.
M0 cannot win.  June--August 2026 were never used to choose a variant.

The Kalman representation is a forward fixed-parameter filter only; no
smoother, backward state update, future observation, or imputation is used.
An unavailable local source becomes explicit missing anchor state, not a
candidate exclusion.  The materialisation recorded 9,438,725 events and
53,147 snapshots across 160 symbols; 13 corrupted local OHLCV sources stayed
explicitly unavailable.  There were no exchange calls.

The resolved source labels are next-eight-hour revisit, rejection,
accepted-cross, continuation, and directional utility (away-MFE minus
toward-MAE in ATR).  They train source heads only after resolution and are
never supplied to a scored candidate or the downstream mapper.

## Source evidence

M2 is a real conditional state signal.  Across the three 2026 confirmation
months its utility Spearman is +0.1076, +0.0073 and +0.0320 respectively
(June, July, August), compared with M0's +0.0550, +0.0132 and +0.0211.
The binary structural heads remain much more stable: M2 rejection AUC is
0.896/0.885/0.884 and continuation AUC is 0.681/0.669/0.676.

Conditional-information diagnostics are calculated after conditioning on M0,
price controls and source family.  The highest M2 fields are consistently
distance, prior MFE/giveback, and approach/transition state; this supports the
claim that the source state is not merely a price-location restatement.  The
useful high-support age bands are 0--1h, 1--2h, 2--4h and 4--8h.  Later bands
are too sparse to interpret.

## Downstream economic test

Every downstream row uses the same family-specific absolute parent-policy
MC1 architecture, 21-day 10%-trimmed prior-resolved score-band shift, dual
BCF/current +50-bps admission, BCF-MC1 auction priority, source-aligned
parent-policy labels, and controlled global long-only portfolio auction
(7x/10%-slot, two new entries, eight concurrent, 80% wallet).  Invalid
outcomes are excluded before capacity.  Thus the comparison is an actual
admission-and-portfolio test, not a top-tail diagnostic.

| Window | Arm | Accepted | Net EV / trade | Net contribution | Worst month | Worst week | Max drawdown |
|---|---|---:|---:|---:|---:|---:|---:|
| Jun--Jul 2026 | C1 levels/value-area | 465 | +197.74 bps | +91,947.72 bps | +167.77 bps | +120.76 bps | -31.48% |
| Jun--Jul 2026 | Anchor M2 only | 189 | +260.66 bps | +49,263.88 bps | +220.04 bps | +82.72 bps | -37.27% |
| Jun--Jul 2026 | C1 + Anchor M2 | 404 | +211.52 bps | +85,452.87 bps | +184.24 bps | +130.97 bps | -33.67% |
| Aug. 1--18 2026 | C1 levels/value-area | 98 | +277.61 bps | +27,205.70 bps | +277.61 bps | +148.19 bps | -4.71% |
| Aug. 1--18 2026 | Anchor M2 only | 53 | +336.67 bps | +17,843.57 bps | +336.67 bps | +260.27 bps | -5.47% |
| Aug. 1--18 2026 | C1 + Anchor M2 | 96 | +287.00 bps | +27,551.61 bps | +287.00 bps | +148.19 bps | -4.71% |

Joined arithmetically over the measured windows, C1 has 563 accepted trades,
+211.64 bps/trade and +119,153.42 bps contribution.  C1+M2 has 500 trades,
+226.01 bps/trade and +113,004.47 bps contribution: **-6,148.94 bps total**
with 63 fewer entries.  M2 is more selective and raises per-trade EV, but it
does not satisfy the predeclared total-contribution and risk-portability gate.
In June--July it also worsens drawdown by 2.19 percentage points versus C1.

The partial-August improvement (+345.91 bps total versus C1) is too small and
too short to overturn the main-window loss.  It is confirmation evidence,
not a licence to select post hoc.

## Retained and removed feature contracts

The retained C1 levels/value-area contract is unchanged:

- `profile_poc_distance_atr`
- `profile_vah_distance_atr`
- `profile_val_distance_atr`
- `profile_hvn_distance_atr`
- `profile_lvn_distance_atr`
- `profile_inside_value_area`
- `profile_value_area_width_atr`

Backward deletions of POC, VAH/VAL, HVN/LVN, or value-area geometry each lost
between 6,411 and 9,692 bps in joined June--August contribution.  Therefore
all seven remain together in the research challenger.

The following are removed from the candidate contract: anchor M2, M3 and M4
Kalman/filter fields; time-at-price/balance; OI-at-price; the all-feature
profile bundle; and Bollinger, Keltner, and Donchian state.  Channel features
were individually tested on top of levels/value-area and all lowered joined
portfolio contribution, from -8,635 bps (Donchian) to -15,076 bps
(Bollinger).  They are diagnostic research features only.

## Reproducibility

- Source geometry: `extreme_price_movements/causal_anchor_geometry.py`
- Target-free materialisation:
  `scripts/materialize_causal_anchor_geometry.py`
- Source-head selection and confirmation: `scripts/run_causal_anchor_heads.py`
- Downstream mapper runner:
  `scripts/run_canonical_sr_e2_mc1_input_ablation.py`
- Partial-August extension:
  `scripts/run_canonical_sr_e2_mc1_august_extension.py`
- Source artifact:
  `data_perp/artifacts/causal_anchor_geometry_2025_train_2026_score_20260831_v3`
- Source-head receipt:
  `data_perp/artifacts/causal_anchor_heads_2025select_2026confirm_20260831_v2`
- June--July portfolio receipt:
  `data_perp/artifacts/canonical_anchor_mc1_junjul_20260831_v1`
- August portfolio receipt:
  `data_perp/artifacts/canonical_anchor_mc1_august_20260831_v1`

Focused causal source/merge/extension tests pass (10/10).  This receipt makes
no live, canonical, policy, execution, or exchange-state change.
