# P8U E2 q50 Agreement + H4 Giveback-20 Research-Canonical Handover

> **Superseded for any live or release decision (2026-09-01).** This document
> records the original 15-minute, source-valid research result. Its `+23.21`
> bps/trade H4 headline and `+25.31` bps/trade H4 + Giveback-20 headline are
> not exact-one-minute, decision-plus-five-minute live-equivalent estimates.
> The action-aligned exact-path release test found H4 + Giveback-20 below the
> unchanged rich parent policy, so neither component may be enabled in the
> exchange-writing stack from this handover. The authoritative decision record
> is [`E2_H4_GIVEBACK20_EXACT_RELEASE_GATE_20260901.md`](E2_H4_GIVEBACK20_EXACT_RELEASE_GATE_20260901.md),
> with the population/policy reconciliation in
> [`CAUSAL_SR_E2_C1_H4_MATCHED_2X2_20260831.md`](CAUSAL_SR_E2_C1_H4_MATCHED_2X2_20260831.md).

## Status and decision

This is the research-canonical long-only P8U entry/continuation extension as of
2026-08-30:

```text
target-free BCF/current-MC1 candidate universe
→ ordinary BCF top-two incumbents
→ E2 q50 agreement entry authority
→ normal portfolio constraints
→ rich parent policy
→ H4 continuation state model every completed 15-minute bar
→ 50% earlier trailing activation plus 20% tighter giveback when H4 >= 0
```

`E2_q50_agreement` and `H4_giveback20` are frozen research choices. They do
not retroactively change historical live results. The separately sealed live
successor below was activated after a current-refit, target-free parity check;
it processes only new decisions and never replays a stale entry.

The machine-readable source of truth is
[`strict_r3_p8u_e2_h4_giveback20_research_canonical_20260830_v1.json`](../config/strict_r3_p8u_e2_h4_giveback20_research_canonical_20260830_v1.json).
Its deterministic implementation is
[`p8u_e2_h4_giveback20_contract.py`](../extreme_price_movements/p8u_e2_h4_giveback20_contract.py).
They enforce the fixed E2 intersection and the fixed H4 next-interval action:
50% earlier trailing activation plus 20% tighter giveback when H4 is
non-negative. The gradual controller is explicitly excluded from this
canonical contract.

The selected components have completed bounded model HPO. No additional model
HPO run was needed for this handover. The E2 intersection itself is
deterministic--it has no learned merger, calibration, or free authority weight.

### Attribution clarification

The reported **+2.09 bps/trade** is deliberately narrow: it is the incremental
effect of Giveback-20 *after H4 activation authority is already enabled*.  It
must not be read as H4's total value against the unchanged rich parent policy.
On this handover's direct composed Jun--Aug comparison, H4 activation-only is
the parent-policy comparison and adds **+23.21 bps/trade**; Giveback-20 adds
the further +2.09 bps/trade.  The total fixed H4 + Giveback-20 difference from
the rich parent is therefore **+25.31 bps/trade**.

An exact-one-minute, +5-minute-entry v58 replay has also been run as a
separate **execution-transfer diagnostic**.  It is not an H4 replacement
measurement: it uses a different source-valid Router50/dual-MC1 route,
execution timing, policy materialisation, and common portfolio history.  Its
smaller uplift must not be compared with or substituted for the E2-selected
Jun--Aug H4 result above.

### August matched controller confirmation

The later August-only controller comparison uses the exact E2-selected,
source-valid population and the normal constrained auction. The neutral
adapter reproduces the rich parent exactly on every retained path: zero net
bps, exit-bar, and exit-reason differences. Seven `CRCLX/USD:USD` paths are
currently unreadable from the archived 15-minute source, so every arm below is
restricted identically to 700 candidates; this is a matched
source-availability comparison, not an imputation.

| August matched arm | Accepted trades | Net EV/trade | Total net EV | Max DD | Worst week | Sortino |
|---|---:|---:|---:|---:|---:|---:|
| Rich parent | 492 | +33.44 bps | +16,452.2 bps | -35.18% | -13.05% | 0.1223 |
| H4 activation-only | 532 | +52.03 bps | +27,680.9 bps | -19.54% | +24.43% | 0.2196 |
| **H4 + Giveback-20 (canonical)** | **532** | **+54.98 bps** | **+29,251.4 bps** | **-19.21%** | **+28.16%** | **0.2289** |
| Gradual activation + giveback | 498 | +52.76 bps | +26,272.7 bps | -28.11% | +12.07% | 0.2060 |

The gradual controller improves the parent but loses to fixed H4 +
Giveback-20 by 34 accepted trades, 2.23 bps/trade, 2,978.7 total bps, 8.90
percentage points of drawdown, and 0.0230 Sortino. It is therefore a rejected
research comparator, not a canonical alternative. The full receipt is
[`strict_r3_p8u_15m_e2_h4_gradual_august_holdout_20260830_v4`](../data_perp/artifacts/strict_r3_p8u_15m_e2_h4_gradual_august_holdout_20260830_v4).

## Causal contract

- All entry candidates and all E2 selections are target-free at decision time.
- Each monthly fit uses up to four preceding complete calendar months. Training
  labels must have resolved before the held-month boundary.
- June--July 2026 are the selection period. August 2026 remains an untouched
  holdout in every cited selection decision.
- Rich-policy costs are embedded exactly once in the outcome label/replay.
- E2 never expands entry capacity: it only retains a replacement when both
  component selectors independently chose it under their frozen policy.
- H4 observes only the completed 15-minute state of an already-open position.
  Its decision changes the *next* interval only; it cannot arm and exit on the
  same bar, cannot loosen a threshold, and does not alter sizing.
- The ordinary constrained portfolio auction remains unchanged.

## Entry model: E2 q50 agreement

### What it does

H0 and H3 independently score the same reserve/incumbent pair. A pair is a
20--30-bps dual-MC1 reserve candidate versus the marginal ordinary BCF-priority
30-bps incumbent. The supervised target is:

```text
pair_advantage_bps = rich-policy net bps(reserve) - rich-policy net bps(incumbent)
```

Each head can replace the incumbent only when its predicted q50 pair advantage
is at least +50 bps. E2 takes the intersection of their target-free selections.
It therefore has demotion and replacement authority but no promotion/capacity
authority.

### Frozen HPO-selected component heads

| Head | Loss / target | Depth | Leaves | Min child fraction | L2 | Learning rate | Trees | Seed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| H0 | LightGBM quantile q50 on `pair_advantage_bps` | 3 | 7 | 0.03 | 8 | 0.03 | 350 | 1729 |
| H3 | LightGBM quantile q50 on `pair_advantage_bps` | 2 | 3 | 0.04 | 12 | 0.03 | 350 | 1729 |

Both use `subsample=0.80`, `colsample_bytree=0.80`, and reserve-position
sample weight `1 + clip((reserve_dual_mc1_min_bps - 20) / 10, 0, 1)`.

The HPO evaluated six support-first geometries after causal feature selection.
H0 ranked first and H3 second on June--July portfolio-constrained selection
evidence. E2 was subsequently tested as their parameter-free selection
intersection and passed the August holdout.

### Features

The entry contract is exactly the position-ordered, fold-specific 30-field
`E3_vwap_fs` selection from:

`data_perp/artifacts/strict_r3_p8u_15m_entry_feature_contract_20260830_v2/stable_selected_features.parquet`

SHA-256: `60423f6c83043565de3fd0b5d06d8e9ea391c9d945f589cd5800f9da3ef7c580`.

It always includes the score/margin context:

- `margin__bcf_final_score`
- `margin__bcf_mc1_expected_bps`
- `margin__current_mc1_expected_bps`
- `margin__dual_mc1_min_bps`
- `incumbent_bcf_mc1_expected_bps`

The remaining causally selected fields are 15-minute impulse, pullback,
efficiency, volatility, volume/participation, level-structure, and VWAP state
features. Examples include `pullback_severity_trend_15m`,
`failed_favorable_break_15m`, `dist_to_vwap_atr`,
`delta_dist_to_vwap_15m`, `vwap_slope_aligned`,
`price_minus_vwap_slope`, `rv_15m_1h`, and
`volume_acceleration_15m_1h`.

The exact field *order* must be read from the hashed feature-selection receipt
for the current refit; it must not be replaced by a hand-maintained static
list. This preserves the causal fold-specific feature-selection contract.

### E2 evidence

| Scope | BCF top-two control | E2 q50 agreement | E2 delta |
|---|---:|---:|---:|
| Jun--Aug: trades | 1,527 | 1,512 | -15 |
| Jun--Aug: net EV/trade | +11.39 bps | +13.75 bps | +2.37 bps |
| Jun--Aug: total net EV | +17,386.5 bps | +20,795.3 bps | +3,408.8 bps |
| Jun--Aug: max drawdown | -89.0% | -84.8% | +4.2 pp |
| Jun--Aug: Sortino | 0.0335 | 0.0421 | +0.0086 |
| August holdout: net EV/trade | +5.14 bps | +10.06 bps | +4.92 bps |
| August holdout: total net EV | +2,741.9 bps | +5,252.9 bps | +2,510.9 bps |
| August holdout: max drawdown | -60.6% | -53.8% | +6.8 pp |

The residual-MC1 and residual-base alternatives did not port to August and are
explicitly non-canonical. The demotion-only alternatives were positive but
inferior to E2.

## Continuation model: H4 with Giveback-20

### Target and authority

H4 is a LightGBM L1 mean regressor of `activation50_advantage_bps`:

```text
U(continue with the existing 50%-earlier trailing action)
  - U(continue under the unchanged rich parent policy)
```

The prediction is recomputed at each completed 15-minute state. If prediction
is non-negative, it enables the already-tested 50% earlier trailing activation
and reduces the parent-policy trailing giveback by 20% for the following
interval. It does not tighten the hard stop and does not relax smooth capital
protection or any other parent-policy threshold.

### Frozen HPO-selected model

| Loss / target | Depth | Leaves | Min child fraction | L2 | Learning rate | Trees | Seed |
|---|---:|---:|---:|---:|---:|---:|---:|
| LightGBM L1 mean on `activation50_advantage_bps` | 4 | 15 | 0.05 | 20 | 0.025 | 420 | 1729 |

The HPO evaluated five high-support LightGBM geometries on April--July with
August held out. H4 was selected by total net EV per absolute drawdown. The
subsequent bounded authority grid tested giveback tightening {0%, 10%, 20%},
hard-stop tightening {0%, 10%, 20%}, and their matched combinations. The 20%
giveback-only variant was the clear portable winner; tighter hard stops were
inferior.

### Features

The continuation contract is exactly the position-ordered, fold-specific
45-field `C4_normalized_vwap_fs` selection from:

`data_perp/artifacts/strict_r3_p8u_15m_continuation_feature_contract_20260830_v2/stable_selected_features.parquet`

SHA-256: `5d0fdb50855070a60cbfd785153dc2f7b905820a01bb5ab6f8d6375cc9ccbf8e`.

Mandatory state is:

- `time_in_trade`, `current_pnl_atr`, `current_MFE_ATR`, `current_MAE_ATR`
- `giveback_from_MFE_ATR`, `distance_to_current_SL_ATR`
- `is_trailing_active`, `current_protection_state`, `MC1_expected_bps`

The remaining selected fields are causal state-normalised measures of expected
MFE/MAE/giveback at age, VWAP distance/cross/slope, favourable/adverse force,
momentum, efficiency, volume, resistance distance, and volatility. As with E2,
the exact fold-specific feature order comes only from the hashed receipt.

### Composed E2 + H4 evidence

This is the required direct composition replay--not an inference from separate
entry and exit studies. It uses E2's exact target-free selections, normal
portfolio constraints, H4's fixed model, and a common source-valid universe.
It has three distinct layers of comparison:

1. `C0_parent_rich_policy`: no continuation-head authority;
2. `H4_control_activation50`: H4's full value, i.e. activation-only authority;
3. `H4_giveback20`: H4 plus the separately selected 20% giveback tightening.

| Scope | Arm | Trades | Net EV/trade | Total net EV | Max DD | Sortino | Worst week |
|---|---|---:|---:|---:|---:|---:|---:|
| Jun--Jul selection | Rich parent | 949 | +32.19 bps | +30,545.0 bps | -36.67% | 0.1323 | -18.11% |
| Jun--Jul selection | H4 activation-only | 991 | +58.04 bps | +57,515.0 bps | -16.72% | 0.2631 | +11.65% |
| Jun--Jul selection | H4 + Giveback-20 | 991 | +59.67 bps | +59,134.0 bps | -16.55% | 0.2687 | +12.74% |
| August holdout | Rich parent | 499 | +33.08 bps | +16,506.8 bps | -35.33% | 0.1218 | -13.24% |
| August holdout | H4 activation-only | 539 | +51.43 bps | +27,718.1 bps | -19.55% | 0.2192 | +24.00% |
| August holdout | H4 + Giveback-20 | 539 | +54.36 bps | +29,300.6 bps | -19.22% | 0.2287 | +27.77% |
| Jun--Aug | Rich parent | 1,448 | +32.49 bps | +47,051.8 bps | -43.11% | 0.1285 | -20.34% |
| Jun--Aug | H4 activation-only | 1,530 | +55.71 bps | +85,233.1 bps | -19.55% | 0.2471 | +6.96% |
| Jun--Aug | H4 + Giveback-20 | 1,530 | +57.80 bps | +88,434.6 bps | -19.22% | 0.2540 | +8.39% |

Thus, versus the rich parent policy, **H4 activation-only adds +23.21
bps/trade and +38,181.4 bps total** across Jun--Aug.  Giveback-20 contributes
an additional **+2.09 bps/trade and +3,201.4 bps total** on top of H4, for a
total canonical uplift of **+25.31 bps/trade and +41,382.8 bps**.  The result
also holds in the untouched August holdout: H4 adds +18.35 bps/trade, and the
full H4 + Giveback-20 stack adds +21.28 bps/trade.

One historical archive issue was treated fail-closed: five June `TON/USD:USD`
candidate paths were excluded identically from every composed H4 arm because
their local 15-minute Parquet source was an unhydrated/corrupt placeholder. No
bar was imputed or reconstructed. The receipt is
`source_coverage.parquet` (SHA-256
`138a6dab165b60941298a1d72bd4c0b4fa9f0251ef8bdb582c81dd73ce78acfc`).

## Exact v58 gradual-controller execution-transfer diagnostic (research only)

This later experiment tests a calibrated *continuous* authority under a
different execution-transfer contract.  It is not promoted into the live
contract, does not replace the fixed H4 research handover, and cannot be used
to lower its stated parent-to-H4 uplift.  A gradual successor for canonical H4
must instead be trained and replayed on the exact E2-selected Jun--Aug route,
with August retained as its untouched holdout.

### Matched contract

```text
target-free Router50 route
→ dual BCF/current MC1 >= +50 bps
→ BCF priority and one chronological portfolio state
→ actual +5-minute entry
→ full exact one-minute rich-policy paths
→ completed 15-minute H4 state only governs the following interval
```

The null gradual adapter reproduces all 7,701 parent paths exactly: zero net,
gross, exit-minute, exit-time, and exit-reason differences.  H4 is fitted only
on earlier resolved states; each held month keeps its final preceding 28 days
as an isotonic calibration reserve for `P(activation50_advantage_bps > 0)`.
No held-path state or outcome reaches the controller.

The 432-cell coarse grid screens May--June, then nominates three variants per
control for a normal constrained July holdout.  Each independent winner is
replayed across May--July only after selection.  The controller never changes
sizing, a previously ratcheted floor, or a stop beyond the hard 5% entry-loss
cap.

| Exact v58 May--Jul constrained arm | Trades | Net EV/trade | Total net EV | Max DD | Worst week | Sortino |
|---|---:|---:|---:|---:|---:|---:|
| Rich parent | 2,480 | +59.49 bps | +147,542.7 bps | -36.86% | +12.25% | 0.2919 |
| Fixed H4 activation + Giveback-20 | 2,587 | +61.93 bps | +160,220.5 bps | -31.51% | +3.36% | 0.3156 |
| Gradual activation | 2,507 | +65.39 bps | +163,933.1 bps | -31.76% | +29.50% | 0.3274 |
| **Gradual activation + giveback** | **2,504** | **+65.73 bps** | **+164,580.1 bps** | **-31.70%** | **+30.29%** | **0.3286** |
| Gradual activation + giveback + stop | 2,504 | +65.31 bps | +163,527.3 bps | -31.69% | +30.34% | 0.3257 |

The selected gradual activation/giveback research controller is:

- calibrated probability threshold: `p = 0.20`;
- activation authority: at high `p`, up to 75% earlier with power 2; at low
  `p`, asymmetrically postpone up to 75% with the same power;
- giveback authority: at high `p`, tighten up to 30% with square-root response;
  at low `p`, extend up to 30% symmetrically;
- no stop authority in the retained composition.  The independently selected
  stop shrinker is positive alone but reduces the activation/giveback result.

Relative to the exact v58 rich parent, the retained composition adds **+6.23
bps/trade**, **+17,037.4 total bps**, 24 entries, improves max drawdown by
**5.17 percentage points**, and improves Sortino by **0.0366**.  July was a
selection holdout for the component grid, but May--June screen evidence is not
independent of the full-period summary; the result remains a research
challenger pending untouched validation.

New receipts and scripts:

- parent-parity receipt:
  `data_perp/artifacts/strict_r3_p8u_v58_gradual_exit_parent_parity_20260830_v1/receipt.json`;
- 432-cell screen blocks:
  `data_perp/artifacts/strict_r3_p8u_v58_gradual_exit_grid_screen_*_20260830_v1/`;
- July finalist selection and individual full controls:
  `data_perp/artifacts/strict_r3_p8u_v58_gradual_exit_finalists_20260830_v5/`;
- selected-control interaction replay:
  `data_perp/artifacts/strict_r3_p8u_v58_gradual_exit_composition_20260830_v1/`;
- generic exact adapter:
  `extreme_price_movements/exact_1m_h4_overlay_research.py`
  (`2d94cac3cc8f8beb70b4b2b3e8017e88c1f18dc802e8eb7a78730cb37fc8e923`);
- grid runner:
  `scripts/run_strict_r3_p8u_v58_gradual_exit_grid.py`
  (`df885cfec99e30c6c156c4f455b306375dff23b355ef66b731a03a69cc7b3727`).

## Required scripts and immutable receipts

| Purpose | Path | SHA-256 |
|---|---|---|
| Entry feature selection | `scripts/run_strict_r3_p8u_15m_entry_feature_contract_ablation.py` | feature contract above |
| Entry HPO | `scripts/run_strict_r3_p8u_15m_entry_postfs_hpo.py` | `7e3d791830d65cbbfb421f02f28f264ebc41b10e37db04793d6d620d1fb8c61a` |
| E2 / demotion / residual comparison | `scripts/run_strict_r3_p8u_15m_entry_e2_demotion_residual_ablation.py` | `7dc15254cdf89ef2932cbac6d87e6854359793db91eb61fa376b76301c7a710b` |
| Continuation feature selection | `scripts/run_strict_r3_p8u_15m_continuation_feature_contract_ablation.py` | feature contract above |
| H4 model HPO | `scripts/run_strict_r3_p8u_15m_continuation_postfs_hpo.py` | `55a4c36c1be869e52f726b482efcaaf0a12e64b482f7a4549448fc77c2af9cd7` |
| H4 authority replay | `scripts/run_strict_r3_p8u_15m_h4_exit_modulation_ablation.py` | `2675bb7c04aba9d200719370bf67939dd94749ca9dcf54f3312dd5fdaf15b3ac` |
| Parent replay ordering | `extreme_price_movements/p8u_continuation_state.py` | `c1918cc906c79c1b56416597a581d70721bd2ba95d56c7d152baeec91b0e60b5` |

Primary receipts:

- E2 comparison: `data_perp/artifacts/strict_r3_p8u_15m_entry_e2_demotion_residual_20260830_v1/portfolio_summary.parquet`
  SHA-256 `d14451b7575611b0a746c8dfdf5f97e1b6ca3217dcdb5f61781291b00f82e55e`
- E2 selection: `data_perp/artifacts/strict_r3_p8u_15m_entry_e2_demotion_residual_20260830_v1/E2_q50_agreement_selection_target_free.parquet`
  SHA-256 `c25fcfeb23952d80437d2009afc6c754711829d13d9125ee2720a20acfe038ec`
- H4 authority comparison: `data_perp/artifacts/strict_r3_p8u_15m_h4_exit_modulation_sourcevalid_20260830_v1/portfolio_summary.parquet`
  SHA-256 `67a475f96b3cc824daf2ebe976efd8a8d364ba09a421c06ac11c14f2bde8bc08`
- Direct E2 + H4 composition: `data_perp/artifacts/strict_r3_p8u_15m_e2_h4_giveback20_composed_20260830_v1/portfolio_summary.parquet`
  SHA-256 `5ff1f105b25722c57551d0484d6ad26e9ad83f9fb8d370aa482c948bda780180`
- Exact rich-parent / H4 / Giveback-20 decomposition: `data_perp/artifacts/strict_r3_p8u_15m_h4_parent_vs_giveback20_e2_20260830_v1/portfolio_summary.parquet`
  Manifest SHA-256 `aa1963695a3e0f9852874747ec2f0422d3cb2a04e542b35cbdca0d688b6a6f46`

## Implementation guardrails

1. Refit both E2 component models and H4 only from prior-resolved data.
2. Materialise and hash the current fold's E3/C4 selected feature contracts
   before scoring; reject a missing or reordered contract.
3. Form E2 strictly as `selection(H0) intersection selection(H3)`; do not
   average model scores, widen the reserve range, or add capacity.
4. Apply H4 only after an entry has passed E2 and the normal portfolio auction.
5. At each completed 15-minute state, persist the decision and apply any H4
   action only to the following interval.
6. When H4 prediction is non-negative: `activation_earlier=0.50`,
   `giveback_tighten=0.20`, `sl_tighten=0.00`. Otherwise use the unchanged
   rich parent policy.
7. A source-unavailable path is ineligible for every compared arm; never fill
   it from a later bar or a different source.

## Separately sealed live-parity candidate (no order authority)

### Activated exchange-writing successor — 2026-08-30 22:30 UTC

The research contract is now connected to the separately sealed Kraken
Futures long-only successor. This is an operational activation, not a new
model selection or parameter change. It is bound to:

- release candidate
  [`strict_r3_p8u_e2_h4_live_parity_release_candidate_20260831_v2.json`](../config/strict_r3_p8u_e2_h4_live_parity_release_candidate_20260831_v2.json)
  — SHA-256 `502768ad252b9e8f5a83e79c898f701dc5a133781589fbdb2c8c7e1db9839d7f`;
- exchange-writing gateway
  [`strict_r3_p8u_e2_h4_kraken_live_gateway_20260831_v1.json`](../config/strict_r3_p8u_e2_h4_kraken_live_gateway_20260831_v1.json)
  — SHA-256 `5d025952c2554e0567662d539a35eda06c9eb42ba04c1f6a3bdab54f4a8976fe`;
- persistent fresh-hour session
  [`strict_r3_p8u_e2_h4_kraken_live_session_20260831_v1.json`](../config/strict_r3_p8u_e2_h4_kraken_live_session_20260831_v1.json)
  — SHA-256 `e0297041736ca39ee0d2198cff1dd5beb9cbe5c44f6d3b219bea87206972bf4d`;
- explicit user-authorized activation
  [`strict_r3_p8u_e2_h4_kraken_live_activation_20260831_v1.json`](../config/strict_r3_p8u_e2_h4_kraken_live_activation_20260831_v1.json)
  — SHA-256 `e17ba85d62b81df73d11f3aec896536b7d350ec54262d782294ddc9817014bcc`.

The legacy overlap entry session and minute monitor were stopped only after
the new target-free E2 selection, auction, entry-blueprint, H4-reference, and
empty-state monitor preflights succeeded. The successor starts flat and waits
for the next fresh UTC hour; its 900-second decision-age guard rejects all
earlier score commits. At run time it preserves the upstream Router/base/meta/
dual-MC1 contract, applies the capacity-preserving E2 replacement before the
ordinary auction, and runs H4 only on completed 15-minute position state. If
H4 state is unavailable, the unmodified rich parent policy remains active.

The research-canonical contract now has a **current-refit, no-order successor
candidate** at
`config/strict_r3_p8u_e2_h4_live_parity_release_candidate_20260830_v1.json`
(SHA-256 `63d6b0fe5a1060396f481d41e553a97628796100d70b630d73dc0d2007bd5bef`).

It seals a 2026-08-29 UTC resolved-label cutoff and the preceding four-month
refit:

- E2 H0/H3: 1,021 prior-resolved reserve/incumbent pairs, with the exact
  ordered 30-field E3 receipt;
- H4: 59,131 prior-resolved completed states, with the exact ordered 45-field
  C4 receipt;
- hash-bound model files and input selection receipts in
  `data_perp/artifacts/strict_r3_p8u_e2_h4_live_parity_bundle_20260830_v1`.

The no-order implementation consists of:

- `extreme_price_movements/inference/p8u_e2_h4_live_features.py` for
  target-free, strict completed-15-minute E2 features;
- `extreme_price_movements/inference/p8u_e2_h4_live_parity.py` for the
  capacity-preserving E2 pairing/replacement and H4 prediction authority;
- `extreme_price_movements/inference/p8u_e2_h4_continuation.py` for pure,
  completed-15-minute H4 state construction with only prior state reference;
- `scripts/score_strict_r3_p8u_e2_live_parity_noorder.py` for an immutable
  target-free E2 selection receipt.

This is deliberately **not** connected to
`config/strict_r3_p8u_kraken_live_gateway_20260830_v3_overlap.json` or the
active minute monitor.  The active gateway still has the old dual-50 adapter;
it cannot represent the canonical E2 20--30 bps reserve replacement without
silently discarding the reserve candidate.  Promotion therefore still requires
the successor auction/monitor integration, a same-bundle feature/score/admission
and exit parity receipt, and a distinct explicit exchange-writing activation.

### No-order successor wiring and parity update

The earlier candidate section is now extended by the sealed v4 no-order
successor receipt (v2/v3 remain immutable predecessor records):
`agents/receipts/20260830_strict_r3_p8u_e2_h4_live_parity_candidate_v4.json`.
Its release-candidate configuration is
`config/strict_r3_p8u_e2_h4_live_parity_release_candidate_20260830_v1.json`
(SHA-256 `2f3286a0074c524d196bf1fe21ed77946378184460b07bda9ea63744596300c9`).

It adds three sealed successor components without modifying either active
exchange process:

1. `p8u_e2_h4_auction.py` applies the existing portfolio constraints only
   **after** `e2_entry_selected`; BCF-MC1 expected bps remains the auction
   priority.  E2 can replace a marginal slot but cannot manufacture capacity.
2. `p8u_e2_h4_continuation.py` persists the 45-field H4 decision together
   with the bundle manifest, state-input hash, decision boundary, and the
   half-open next-interval authority.  `p8u_e2_h4_rich_policy.py` advances
   the proven rich parent minute engine one completed minute at a time, using
   the 50%-earlier activation / 20%-tighter giveback only after that decision
   boundary.  A minute ending exactly at the state boundary cannot be changed.
   `run_strict_r3_p8u_e2_h4_noorder_position_monitor.py` is the separately
   named no-order monitor core that writes this successor state into a new
   immutable snapshot before any future exchange integration.
3. `audit_strict_r3_p8u_e2_h4_inference_replay_parity.py` independently
   replayed the entry reserve/marginal pairing and H4 prediction on only
   target-free source columns for 2026-08-20.  The receipt at
   `data_perp/artifacts/strict_r3_p8u_e2_h4_inference_replay_parity_20260830_v1/receipt.json`
   reports 0.0 max absolute delta for H0, H3, and H4; 168 entry rows,
   17 E2 pair rows, and 603 completed H4 states.  It read no outcome columns
   and made no exchange/order call.

The focused E2/H4 contract tests now cover eight conditions, including E2
before the auction, hash-bound H4 persistence, and the no-same-bar interval
rule.  These components are release-ready only as a no-order successor.  The
remaining explicit activation work is to attach them to separately named
gateway/monitor processes, produce current-decision feature and exact-minute
exit parity receipts, then obtain a separate exchange-writing authorization.
