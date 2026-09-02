# Causal S/R Heads — Entry and Continuation Research Handover

## Decision

Do **not** add the causal support/resistance (S/R) feature bundle to the
research-canonical `E2_q50_agreement + H4 + 20% tighter giveback` stack.

> **Superseded for promotion decisions (2026-08-30):**
> [`CAUSAL_SR_ORACLE_CEILING_DECISION_20260830.md`](CAUSAL_SR_ORACLE_CEILING_DECISION_20260830.md)
> records the final oracle-ceiling falsification.  The causal S/R features
> remain research-only; no live or canonical stack changed.

The standalone S/R engine and its four shallow prequential heads are valid,
causal, and useful interaction predictors.  But their incremental use did not
port across the unchanged portfolio-constrained E2/H4 studies:

- adding S/R features to E2 reduced all-OOS net EV/trade by **3.13 bps** and
  total net EV by **4,727.7 bps**;
- adding them to H4 improved June–July selection performance, but degraded the
  untouched August holdout by **7.83 bps/trade**, **4,357.5 bps** total, and
  **3.50 pp** max drawdown.

Therefore the components remain an offline research challenger only.  No live
model, live feature contract, entry authority, or exit policy was changed.

## Causal S/R contract

The engine is a separate, source-local 15-minute OHLCV consumer.  It creates
levels and state snapshots before any interaction outcome is known.

1. Confirmed structure: 1-hour pivots are delayed three completed 1-hour bars;
   4-hour pivots are delayed two completed 4-hour bars.
2. Candidate levels: 1h/4h swings, rolling extrema, prior day/week, VWAP,
   range boundaries, and role reversals.
3. Zones: same-side candidates are merged only inside a deterministic
   0.20-ATR radius.  Touches reset only after two completed 15-minute bars
   move at least 0.75 ATR away.
4. Labels: an 8-hour multi-barrier reaction-before-penetration outcome.  A
   label is available only after that path resolves.
5. Strict OOS folds: each held 2026 month fits exclusively on interactions
   whose `label_available_ts` precedes its first timestamp.  Historical 2025
   interactions are used as training history; no future data or outcome joins
   enter a snapshot.
6. Source failures are candidate-local.  GAS and TON were unavailable in the
   initial materialisation because their archived Parquet files were corrupt;
   the continuation replay additionally encountered TON as unavailable.  No
   fill, cross-symbol substitute, or future bar was used.

The role-reversal counter is capped at six in the implementation.  This means
“many historical role reversals”; it prevents an unbounded numerical chain
from becoming a model feature and does not alter a zone identity or use future
information.

## Four frozen S/R heads

All heads use deterministic LightGBM with seed 1729, depth 3, seven leaves,
`min_child_samples=160`, `subsample=0.80`, `colsample_bytree=0.85`, and
`reg_lambda=12`.

| Head | Target | Loss | Trees / learning rate | OOS diagnostic, Feb–Aug 2026 |
|---|---|---|---|---|
| Prior strength | `y_reaction_strength` | L1 | 280 / .03 | mean Spearman 0.068 |
| Conditional strength | `y_reaction_strength` | L1 | 280 / .03 | mean Spearman **0.247**, minimum 0.235 |
| Accepted-break probability | `y_accepted_break` | binary log-loss | 300 / .03 | mean AUC **0.634**, minimum 0.611; Brier 0.196 |
| Reaction magnitude q50 | `reaction_MFE_atr` | q50 quantile | 320 / .03 | mean Spearman **0.203**, minimum 0.186 |

Inputs are pre-touch zone attributes (type, timeframe, confluence, historical
support, reaction/penetration history, source provenance) plus causal approach
state (distance, return/velocity/acceleration, efficiency, consistency,
impulse/pullback, compression and prior volume).  Touch-bar diagnostics and
every realised interaction field are excluded.

For the long side, the downstream bundle exposes:

- `sr_long_support_hold_strength`
- `sr_long_resistance_break_probability`
- `sr_long_downside_break_probability`
- `sr_long_resistance_rejection_strength`
- `sr_long_structure_balance`
- support/resistance distances in ATR
- support/resistance prior strength and q50 reaction magnitude

The entry arm additionally uses reserve-minus-incumbent margins for these
fields, matching the existing E2 pairwise feature topology.

## Portfolio-constrained entry evidence

The entry test preserves the E2 target-free universe, BCF/current dual-MC1
filter, ordinary portfolio auction, two H0/H3 q50 component models, and their
intersection/replacement authority.  June–July is selection; August is an
untouched holdout.

| Scope | Arm | Accepted trades | Net EV/trade | Total net EV | Max DD | Sortino | Worst week |
|---|---|---:|---:|---:|---:|---:|---:|
| Jun–Aug | Frozen E2 control | 1,512 | +13.75 bps | +20,795.3 bps | -84.80% | 0.042 | -52.52% |
| Jun–Aug | E2 + S/R | 1,513 | +10.62 bps | +16,067.6 bps | -88.87% | 0.031 | -49.35% |
| August | Frozen E2 control | 522 | +10.06 bps | +5,252.9 bps | -53.81% | 0.025 | -34.55% |
| August | E2 + S/R | 529 | +7.19 bps | +3,801.6 bps | -58.53% | 0.014 | -40.36% |

The S/R arm increased August participation by seven accepted trades, but the
economic and risk outcomes were worse.  It is rejected for entry authority.

## Portfolio-constrained continuation evidence

The continuation test retains the frozen E2 entry selection, rich parent
policy, H4 state model, 50%-earlier activation rule, and 20%-tighter giveback.
It changes only the H4 feature inputs.  The valid entry population begins in
June, so June–July is selection and August is untouched.

| Scope | Arm | Accepted trades | Net EV/trade | Total net EV | Max DD | Sortino | Worst week |
|---|---|---:|---:|---:|---:|---:|---:|
| Jun–Jul | H4 giveback-20 control | 991 | +59.67 bps | +59,134.0 bps | -16.55% | 0.269 | +12.74% |
| Jun–Jul | H4 + S/R giveback-20 | 992 | +62.20 bps | +61,698.9 bps | -15.91% | 0.282 | +17.98% |
| August | H4 giveback-20 control | 539 | +54.36 bps | +29,300.6 bps | -19.22% | 0.229 | +27.77% |
| August | H4 + S/R giveback-20 | 536 | +46.54 bps | +24,943.1 bps | -22.72% | 0.187 | +18.18% |
| Jun–Aug | H4 giveback-20 control | 1,530 | +57.80 bps | +88,434.6 bps | -19.22% | 0.254 | +8.39% |
| Jun–Aug | H4 + S/R giveback-20 | 1,528 | +56.70 bps | +86,642.0 bps | -22.72% | 0.246 | +14.00% |

The in-sample selection improvement is encouraging but the August reversal is
material.  No continuation authority is granted.

## Receipts and relevant scripts

| Purpose | File / artifact | SHA-256 |
|---|---|---|
| Causal level engine | `extreme_price_movements/causal_sr_engine.py` | `887c2488fc9ba9765056bc8427a36e38cbaa98b919e0c4156fe073b8857d9dc4` |
| Local materialiser | `scripts/materialize_causal_sr_engine.py` | `931f1b531477cdaf1605b92c1ee097435f19a17ba01e15874ecd11a615af2d2c` |
| Prequential head runner | `scripts/run_causal_sr_heads.py` | `4d8b4a4b1aaea959302878006bc8bed97f06872793626d260fe843ad975fd33c` |
| Entry challenger | `scripts/run_causal_sr_entry_e2_ablation.py` | `d78b4cd223cb7967f88f2526bd099ad4b655e97f84066bfd8a68617f7fde6b79` |
| Continuation challenger | `scripts/run_causal_sr_continuation_ablation.py` | `d1a47a6fac74c5834bcf6b800fc37c3b8028a825755e70ad0c7966cecb4848f2` |
| Materialised S/R ledger | `data_perp/artifacts/causal_sr_engine_2025_train_2026_score_20260830_v1` | manifest `a838a05c5be131d47cb8eb65d175df038d3c93714ea5496818aafd967c2dce54` |
| Corrected OOF head export | `data_perp/artifacts/causal_sr_heads_oof_20260830_v3_entrypivotfix` | manifest `a9066ef2669632211dac76ee5af51bcc38f8595441ec34f9c154049927ad75dc` |
| Entry test | `data_perp/artifacts/causal_sr_entry_e2_agreement_20260830_v4_pairmargin_fixed` | manifest `6d43e7fe5a0fc6b7a58a954d77f553d525591c3c53dad22b0c8662b27cea8a00` |
| Continuation test | `data_perp/artifacts/causal_sr_continuation_h4_giveback20_20260830_v5_named` | manifest `b52732da23d580b7617fa925c84a9de96c85bfc8485b515363bd6da6b009da13` |

## Next research options

Treat this as evidence that the S/R representation predicts local interaction
outcomes but is not yet stable enough for broad policy authority.  Reasonable
future challenger paths are narrow and predeclared: use S/R only as a
no-promotion risk demoter near a nearby resistance/support zone; apply a
monotonic, support-gated shrinkage overlay; or extend the validation period
before selecting a new interaction-to-policy mapping.  None should modify the
current canonical stack without a new untouched test.
