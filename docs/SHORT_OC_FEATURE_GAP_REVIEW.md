# Short P0 → O → C: feature-gap review

Status: completed before any new inference feature is added for the next O/C/K0 ablation funnel.

## Scope and evidence

The review covers the frozen short P0 substrate, O45, C59, the frozen F115
selection pool, and the union of causal `canonical120` panel schemas available
for 2024–26.  The panel union has 225 named fields; 172 are common to every
era panel.  O45 and C59 remain frozen controls in all formulation rounds.

The path/MFE/MAE quantities are labels only.  They are not eligible feature
candidates.  Any approved new feature must be target-free at decision time,
pass the 90% coverage and non-zero-variance gates by chronological training
fold, and carry source-lineage evidence before it enters O or C.

## Existing representation by conceptual family

| Family | Existing causal representation | Assessment |
|---|---|---|
| Price trend / momentum | `mkt_ret_15m`, `mkt_ret_4h`, `mkt_ret_24h`, `ret4h_peer_resid`, recovery fractions, `efficiency_ratio_20`, range/location fields | **PARTIALLY REPRESENTED** — market and peer-relative momentum are present; direct asset return ladders and acceleration are not. |
| Rally geometry | `price_recovery_fraction_{24h,72h}`, `loc_range_pos_{24,48}`, `loc_swing_range_pos_24`, `log_bars_since_above_{1,2,3}atr` | **PARTIALLY REPRESENTED** — there is no direct 6h/12h trough run-up, high distance, or time-since-high coordinate. |
| Reversal / exhaustion | `exh_qual_surprise`, `leveraged_long_breakout_risk`, `false_clean_short`, `spike_score_surprise`, liquidation/rebound fields | **PARTIALLY REPRESENTED** — useful composite proxies exist but price acceleration × volume/OI non-confirmation is not explicit. |
| Support / resistance | daily Donchian/VWAP support/resistance distances, barrier pressure, range locations | **ALREADY REPRESENTED** for the current horizon; no first addition proposed. |
| Volatility / transition | realised/relative volatility, semivol ratio, high-vol-state age, `mkt_rv_4h`, `mkt_return_accel_1h` | **PARTIALLY REPRESENTED** — strong market-state representation, weak asset-level volatility-transition representation. |
| Volume | volume z-scores, trend, entropy, price-volume residual/correlation | **ALREADY REPRESENTED** at level/relative-state level; only use in explicit exhaustion interactions if needed. |
| Liquidity / order book | spread/depth/residual/stress, Amihud and trade-size-to-depth features | **ALREADY REPRESENTED**; adding generic liquidity transforms would be redundant. |
| OI / funding | OI drawdown/recovery, market OI changes/acceleration, OI-to-volume/funding interactions, `leverage_build` | **PARTIALLY REPRESENTED** — direct short-horizon asset OI state changes, duration, and state transitions are absent. |
| Leverage / crowding | `leverage_build`, breakout-risk, OI/funding interactions | **PARTIALLY REPRESENTED** — level is present; price/OI mechanism state duration and changes are missing. |
| Market breadth / cross-sectional state | breadth, breadth dispersion/recovery/change, many `q_*` tails and `xs_dispersion_*` fields | **PARTIALLY REPRESENTED** — distributions are represented, but direct timestamp-local percentile/robust-z coordinates for the strongest raw asset fields are not. |
| Session / time | session positions, hour/day/session-open features | **ALREADY REPRESENTED**. |
| Spectral / regime | spectral eigensystem and entropy fields | **ALREADY REPRESENTED**; do not add another generic regime transform. |

## Candidate-concept disposition

| Proposed concept | Disposition | Rationale / action |
|---|---|---|
| Asset returns at 15m, 30m, 1h, 2h, 4h, 8h | **GENUINELY MISSING** | Existing fields are market/peer residuals or recovery proxies. Materialise a bounded raw-return ladder only. |
| Run-up from 6h/12h/24h trough | **PARTIALLY REPRESENTED** | 24h/72h recovery fractions exist. Add only 6h and 12h ATR-normalised run-up; retain the existing 24h proxies. |
| Distance from and time since 6h/12h/24h high | **GENUINELY MISSING** | `log_bars_since_above_natr` is not a high-location or time-since-new-high measure. |
| Consecutive positive bars / positive-bar fraction / upside path efficiency | **GENUINELY MISSING** | These are not target-free fields in the O/C contracts. Add only 1h/4h versions initially. |
| Asset return slope/acceleration | **GENUINELY MISSING** | `mkt_return_accel_1h` is market-only. Add 1h and 4h asset acceleration, not a broad derivative factory. |
| Price acceleration × volume deceleration | **GENUINELY MISSING** | Existing `exh_qual_surprise` is related but not an explicit causal non-confirmation interaction. Add 1h/4h only if primitive coverage passes. |
| Price acceleration × OI deceleration | **GENUINELY MISSING** | `price_up_oi_down_4h_rz` is a single-state proxy. Add 1h/4h interactions only. |
| New high without breadth, volume, or OI confirmation | **PARTIALLY REPRESENTED** | Breadth/volume/OI primitives and breakout-risk composite exist. Use the direct new-high indicator only with a predeclared market-confirmation interaction. |
| Price/OI mechanism states at 1h/4h/12h | **PARTIALLY REPRESENTED** | Market states and `price_up_oi_down_4h_rz` exist; direct asset states at 1h/12h do not. Add the three other direction pairs at 1h/4h/12h. |
| Mechanism-state duration and transitions | **GENUINELY MISSING** | No direct causal duration or transition coordinates exist. Add only the two predeclared short-relevant transitions: up/OI-up → up/OI-down and up/OI-down → down/OI-down. |
| Deltas of every O/C field | **REDUNDANT** | Prohibited: would create a broad, correlated transform expansion. |
| Deltas/accelerations of price recovery, OI drawdown/recovery, leverage, funding, liquidity/spread stress, breadth | **PARTIALLY REPRESENTED** | Market return/OI/breadth acceleration exists; asset-level causal deltas for the listed high-value fields are absent. Evaluate only these bounded variants. |
| Timestamp-local raw-field percentile / robust-z | **PARTIALLY REPRESENTED** | Tail/dispersion features are not equivalent to a direct asset coordinate. Add for a deduplicated top-20 O/C raw-field subset only, calculated before P0/candidate filtering. |
| 7d/30d/90d self-percentile | **GENUINELY MISSING** | Current rolling z/residual fields do not give a true own-history percentile. Start with 30d for the approved raw subset; add 7d/90d only if the block advances. |
| Generic liquidity, spectral, session, or extra funding transforms | **REDUNDANT** | Adequately represented already; no addition approved. |
| Post-entry path/MFE/MAE/correctness feature | **NOT CAUSALLY AVAILABLE** | Keep as supervised labels only. |

## Approved bounded feature blocks for Phase 2 only

No field is added by this review.  The following are the only approved blocks
for later OF/CF ablations after the O formulation round:

1. **SF — short rally mechanics:** asset return ladder; 6h/12h trough run-up;
   high distance/age; 1h/4h positive-bar count/fraction/efficiency; 1h/4h
   return acceleration; bounded price-acceleration × volume/OI-deceleration.
2. **TF — transition mechanics:** 1h/4h/12h asset price×OI direction states,
   their durations, two specified state transitions, and deltas only for
   asset OI drawdown/recovery, leverage build, funding, breadth, price
   recovery, liquidity/spread stress, and support/resistance distance.
3. **XS — cross-sectional coordinates:** timestamp-local percentile and robust
   z for the deduplicated raw O/C stability-core subset.  The complete
   decision-time universe is required before P0 filtering.
4. **SP — self-percentiles:** prior-only 30d own-history percentile for the
   same approved subset.  7d/90d are conditional extensions, not initial
   fields.

The proposed blocks are independent.  They must first improve the changed
head's own target metrics; final K0 economics alone may not select them.

## Causal/source gates before Phase 2

- All calculations must use completed information strictly before the
  decision timestamp; timestamp-local XS values use the complete target-free
  universe at that timestamp.
- All own-history ranks exclude the current observation.
- Each block needs chronological per-fold coverage >=90%, non-zero variance,
  no post-entry lineage, and a source audit in the generated feature artifact.
- O and C may receive different advancing blocks.  The long stack, live
  contracts, and frozen A0 control are untouched.

## Recommendation

Proceed with formulation work on frozen O45/C59 first.  If an O formulation
advances, test SF, TF, XS, and SP independently in that order.  Do not add
generic regime, liquidity, session, spectral, or path-outcome fields.
