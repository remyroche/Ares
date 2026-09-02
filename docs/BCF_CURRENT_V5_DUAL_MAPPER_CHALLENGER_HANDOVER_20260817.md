# BCF/current-v5 dual-MC1 agreement challenger

**Status:** offline sealed challenger; not deployed or order-capable.  
**Scope:** long-only Kraken perpetual candidate population.  
**Selection:** frozen on the 2026 grid; one frozen winner-only 2025 test.  
**Receipt:** `agents/receipts/20260817_strict_r3_bcf_current_v5_agreement_blend_v1.json`.

## Purpose

The experiment asks whether the BCF score family can act as a conservative
confirmation of the current-v5 score family.  It does not train a third alpha
model and it does not blend raw scores.  Each family first receives its own
causal absolute-EV map; a candidate must pass both maps.  BCF then provides
the auction priority.

```text
point-in-time candidate
 ├─ current-v5 final_score → current-v5 MC1 expected policy-net EV
 └─ BCF final_score        → BCF MC1 expected policy-net EV

admit only if both mapped EVs >= 30 bps
→ global constrained auction, priority = BCF MC1 expected EV
→ frozen rich SimplePolicyOptimiser exit, with Adaptive Exit V1 activation modulation
```

The `30 bps` threshold and BCF-only priority were chosen by the predeclared
2026 balance screen: increase total bps, preserve at least 97% of the control
EV per trade, and avoid an excessive zero-trade-day increase.  The 2025 run
uses that exact choice; it is not another threshold search.

## Shared candidate and outcome contract

- Point-in-time long candidate identity: `candidate_id`, symbol and UTC
  decision timestamp.
- Future outcome fields are joined only after scoring.  Invalid policy paths
  are excluded from mapper fitting and from portfolio capacity allocation.
- Parent-policy outcome source:
  `data_perp/artifacts/strict_r3_source_aligned_optimized_policy_outcomes_long_2024jan_jul2026_20260812_v1/candidate_policy_outcomes.parquet`
  (SHA-256 `d0d03430111437723b8207746ea05bb5c9b6c0e9795dae99fa0bbf9f7c970600`).
- The matched score-family comparison verified every shared policy field on
  522,070 candidate IDs before portfolio replay.
- The outcome used by the two MC1 maps and the constrained replay is the
  source-aligned **optimized** 15-minute parent policy: entry at the first
  executable 15-minute decision open, 12-hour timeout, 100 bps cost once,
  SL `4.15200064 ATR`, trailing activation `2.32622492 ATR`, giveback
  `0.10237199 ATR`.

The generic `FrozenPolicyContract` defaults in
`strict_r3_canonical_v2.py` are not the policy-label source for these
comparison metrics.  The `winner.json` above is authoritative.

## BCF score family

BCF means the schema-v2 base/consensus/final-score pipeline implemented in
`extreme_price_movements/strict_r3_canonical_v2.py`.

1. **Strict-R3 base.** A three-class LightGBM model uses the frozen 120 causal
   base fields.  It predicts adverse, weak and robust-clear path states, with
   `base_score = P(clear) - 0.5 * P(adverse)`.  Each historical bundle fits at
   most 240,000 pre-resolved rows.
2. **Same-model base reference.** The fitted base scores its own preceding
   42-day target-free reserve.  `base_rank42` is the reserve empirical CDF;
   held rows never enter that reference.
3. **Policy anchor.** A 20-bin monotonic map trained on earlier prequential
   base predictions and resolved parent-policy outcomes gives
   `base_anchor_bps`.  The residual target is
   `policy_net_bps - base_anchor_bps`.
4. **Ten residual rankers.** Five feature caps (`40, 60, 80, 100, 120`) times
   ordinary/equal-month weighting train ten LambdaRank heads with 4-hour UTC
   × side queries.  Residual bands are `[-150, -50, +50, +150]` bps.  Every
   head is converted through its own training-score CDF; their median is
   `consensus_rank`.
5. **Upstream score.**
   `upstream = 0.75 * base_rank42 + 0.25 * consensus_rank`.
6. **Frozen Geometry/K9.** The 64-tree encoder, leaf contract and K9 state are
   defined once on October–December 2024 and are never refit monthly.  The
   geometry bundle is part of the Severe feature contract; state semantics are
   therefore persistent across historical folds.
7. **Severe demotion.** A 123-field classifier estimates
   `P(H12 TP6/SL4 net <= -200 bps)`.  It receives five upstream fields,
   73 causal context fields and 45 geometry/K9 structural fields.  It is a
   demoter only:
   `raw_severe = upstream * (1 - 0.5 * P(severe))`.
8. **Final BCF score.** The same fitted bundle scores its preceding 42-day
   reference on `raw_severe`; `BCF_final_score = CDF42(raw_severe)`.  No
   held-window rank is used.

Historical BCF model parameters are fixed in the module:

| Component | Main parameters |
|---|---|
| Base | multiclass LightGBM; 220 trees; LR .035; depth 5; 24 leaves; min-child 2,400; 85% feature fraction; L2 20 |
| Residual head | LambdaRank; 120 trees; LR .035; depth 5; 31 leaves; min-child 300; 82% feature/bag fraction; L1 .02; L2 2; gains `[0,.25,1,3,7]`; truncation 10 |
| Geometry encoder | binary LightGBM; 64 trees; LR .04; depth 5; 31 leaves; K9 MiniBatchKMeans on 100k equal-month rows |
| Severe demoter | binary LightGBM; 35 trees; LR .04448; depth 5; 15 leaves; min-child 103; L2 16.579 |

## current-v5 score family

The current-v5 score is the active inference family, implemented by
`extreme_price_movements/strict_r3_canonical_current.py` and sealed by the
v59 inference bundle.  It uses the same Strict-R3 opportunity idea and
top-30% timestamp-local compute routing, but it has a different conversion
stack: conditional residual consensus, correctness demotion and its own
prior-reserve final-score CDF.  Its score and calibration state are not
assumed interchangeable with BCF.

## Family-specific MC1 maps

The comparison reruns the frozen MC1_d2 recipe independently per score family
on the repaired policy substrate.  It uses six causal inputs:

1. `final_score`
2. `base_rank42`
3. `conditional_consensus_rank`
4. `upstream`
5. `ordinary_shadow_consensus_rank`
6. `correctness_rank`

For each historical monthly fold, the static map is a deterministic,
day-balanced HistGradientBoostingRegressor (depth 2, 80 iterations, LR .04,
L2 20, min leaf 100, seed 1729, maximum 50,000 rows).  The target is clipped
canonical parent-policy net bps.  A causal 21-day robust residual location
shift is then added using only labels available before the decision.  The
result is `mc1_expected_bps` for that score family.

This is why a BCF score is never fed through the current-v5 MC1 map: the
mapper is family-specific and its input geometry is score-family-specific.

## Portfolio contract

Both maps must clear the same 30-bps threshold.  The auction uses one common
state with:

- long side only;
- 7× leverage;
- 10% wallet-margin slots;
- two new entries per decision hour;
- eight concurrent positions;
- 80% wallet margin cap;
- one asset position at a time;
- the optimized parent-policy exit and the source-aligned policy outcome.

All reported entries are portfolio-accepted entries, not retrospective top-k
tails.

## Results

### 2026, Jan–Jul (selection period)

| Stack | Entries | Trades/day | Net bps/trade | Total net bps | Worst month | Worst week | Max DD |
|---|---:|---:|---:|---:|---:|---:|---:|
| current-v5 MC1 >=50, priority=current MC1 | 3,760 | 17.74 | +146.86 | +552,209 | +8.87 | −35.40 | −59.38% |
| **Dual 30, BCF priority** | **3,792** | **17.89** | **+168.13** | **+637,568** | **+133.34** | **+37.13** | **−49.42%** |

The challenger adds 32 entries, +21.27 bps/trade and +85,359 total bps.  It
also has zero zero-trade days versus one for the control and five days below
five trades versus ten.

### 2025, Feb–Dec (frozen winner-only test)

| Stack | Entries | Trades/day | Net bps/trade | Total net bps | Worst month | Worst week | Max DD |
|---|---:|---:|---:|---:|---:|---:|---:|
| current-v5 MC1 >=50, priority=current MC1 | 5,822 | 17.43 | +179.58 | +1,045,540 | +111.63 | −440.05 | −94.74% |
| **Dual 30, BCF priority** | **6,106** | **18.28** | **+172.40** | **+1,052,671** | **+118.34** | **−367.23** | **−88.80%** |

The 2025 test trades more and improves total bps, worst month, worst week and
drawdown-adjusted total.  Its EV/trade is 7.19 bps lower (−4.0%), which is the
explicit participation trade-off.

Artifacts:

- 2026 grid: `data_perp/artifacts/strict_r3_bcf_current_v5_agreement_blend_2026_20260817_v1`
- 2025 frozen-winner test: `data_perp/artifacts/strict_r3_bcf_current_v5_agreement_winner_2025_20260817_v1`
- BCF MC1 predictions: `data_perp/artifacts/strict_r3_score_family_matched_mc1_canonical_policy_20260817_v7_bcf/predictions_bcf_mc1_d2.parquet`
- current-v5 MC1 predictions: `data_perp/artifacts/strict_r3_score_family_matched_mc1_canonical_policy_20260817_v7_current/predictions_current_v5_mc1_d2.parquet`
- grid runner: `scripts/ablate_strict_r3_bcf_current_v5_agreement_blend.py`
- family MC1 runner: `scripts/replay_strict_r3_score_family_mc1_canonical_policy.py`

## Five-minute live-execution validation: not yet passed

The research replay assumes the declared 15-minute executable decision open.
For a five-minute live-delay test, a complete Kraken 1-minute bar must be
present at `decision + 5 minutes` for each selected entry.  On the dual-30
2026 accepted population, exact local Kraken support is:

| Period | Entries | exact +5m Kraken bars | Coverage |
|---|---:|---:|---:|
| Jan | 658 | 3 | 0.46% |
| Feb | 425 | 1 | 0.24% |
| Mar | 712 | 3 | 0.42% |
| Apr | 691 | 1 | 0.14% |
| May | 621 | 1 | 0.16% |
| Jun | 323 | 0 | 0.00% |
| Jul | 362 | 0 | 0.00% |
| **Total** | **3,792** | **9** | **0.24%** |

This fails the evidence requirement.  The existing sparse execution cache
cannot establish that the reported full-year metrics survive a five-minute
fill delay.  An attempted 15-minute stress materialisation is also not a
replacement: the raw direct-15m source covers only part of the required
symbol universe, while the parent-policy ledger includes separately labelled
proxy support.

The immutable coverage receipt is
`data_perp/artifacts/strict_r3_bcf_current_v5_agreement_15m_latency_stress_2026_20260817_v1/five_minute_coverage_manifest.json`.

## Requirements before activation

1. Materialise a current, sealed BCF bundle from the same point-in-time
   candidate/features contract as the live v59 producer, including its 42-day
   reference scores and its BCF-specific MC1 calibration state.
2. Implement a dual-map admission module that requires both expected EVs >=30
   bps, ranks the auction by BCF MC1 EV, and preserves all current live
   spread, impact, delay, price-drift and portfolio gates.
3. Validate the implementation on a later, untouched forward period with
   complete exact Kraken 1-minute execution data at `decision + 5 minutes`.
   The current source is insufficient for a retrospective 2026 confirmation.
4. Seal the successor inference/execution/state contracts and run the existing
   deterministic feature/score/admission parity suite before allowing orders.

## 7. 2026-08-17 live-materialisation receipt

The BCF scorer and a separate BCF-native MC1 calibration artifact are now
materialised as an offline challenger. The scorer uses the immutable August
BCF bundle and its own preceding 42-day same-model reserve. Its MC1 inputs are
generated from BCF's own ten residual-head geometry; it does not feed BCF
scores through the current-v5 map.

Receipt:
`agents/receipts/20260817_strict_r3_bcf_current_dual_mc1_live_materialisation_v1.json`.

The dual +30-bps gate and BCF-MC1 priority auction passed exchange-free smoke
tests. It remains deliberately non-order-capable until a source-aligned rich
exit replay and an append-only post-cutoff BCF resolved-score ledger prove the
dynamic shift and full live chain together.
