# Strict-R3 O3-v2 S11 Research Canonical

**Published:** 2026-08-25  
**Status:** `CANONICAL_RESEARCH_CONTRACT_NOT_LIVE`  
**Scope:** long-only, offline research. This replaces the prior T6/T9 challenger specification as the canonical research contract. It does **not** alter the deployed trader, live admission bundle, or any open-position logic. Promotion requires an untouched forward period.

## Canonical decision

S11 is the selected score contract:

```text
S11 final score = 0.75 × B + 0.20 × T6 + 0.05 × T9
```

All three inputs are timestamp-local percentile ranks on the same target-free routed candidate population:

- `B`: enhanced-base rank;
- `T6`: rank-error ordinal correction-head rank;
- `T9`: exit-quality ordinal correction-head rank.

The candidate must first pass the enhanced base's timestamp-local top-30% route. S11 is then mapped independently by the **Current** and **BCF** MC1 expected-EV mappers. Admission requires **both** mapped values to be at least `+50 bps`; the unchanged chronological constrained auction is applied only afterwards.

The machine-readable canonical receipt is [strict_r3_o3v2_t6t9_s11_research_canonical_20260825_v1.json](../config/strict_r3_o3v2_t6t9_s11_research_canonical_20260825_v1.json).

## Why S11 is canonical

The exact same-family control is S1, which used the older `75% B / 12.5% T6 / 12.5% T9` mix. Across the strict-prequential May–July 2026 replay, S11 increases participation and total economics while modestly improving every selected stability statistic.

| Contract | Base / T6 / T9 | Entries | Trades/day | MC1-admitted candidates | Net EV/trade | Total net bps | Worst month | Worst week | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S1 exact score-family control | 75 / 12.5 / 12.5 | 2,191 | 23.82 | 9,893 | +139.37 | +305,366 | +127.81 | +82.02 | −14.51% |
| **S11 research canonical** | **75 / 20 / 5** | **2,244** | **24.39** | **10,562** | **+140.50** | **+315,279** | **+128.09** | **+82.23** | **−13.75%** |
| **S11 minus S1** | — | **+53** | **+0.58** | **+669** | **+1.13** | **+9,913** | **+0.28** | **+0.21** | **+0.76 pp** |

All figures use the reconciled rich-policy net outcome and a single constrained chronological portfolio. The replay had 100% policy-outcome coverage for selected entries.

## Role-separated T9 evidence

T9 has little standalone ranking authority but has modest value as an MC1 conditioning coordinate. That is why its direct S11 authority is deliberately limited to 5% while it remains visible to both MC1 mappers.

| Role-separated comparator | Entries | Net EV/trade | Total net bps | Worst month | Worst week | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|
| S5: `85% B + 15% T6`; T9 hidden from MC1 | 2,237 | +139.10 | +311,156 | +132.33 | +80.09 | −13.80% |
| S5: `85% B + 15% T6`; T9 visible only to MC1 | 2,228 | +140.50 | +313,023 | +131.42 | +80.01 | −15.11% |
| **S11: 75% B + 20% T6 + 5% T9; T9 visible to MC1** | **2,244** | **+140.50** | **+315,279** | **+128.09** | **+82.23** | **−13.75%** |

Compared with the visible-only S5 control, S11 adds 16 entries and +2,255 total bps at effectively identical EV/trade, while improving drawdown by 1.36 percentage points. Its aggregate advantage is driven by useful additional admissions, **not** by proven same-timestamp replacement superiority: in 253 direct substitutions, mean S11-minus-S1 realised EV is −26.65 bps and only 45.1% are positive. This is the main reason S11 remains research-only until it passes untouched forward validation.

## Layer contract

### 1. Enhanced base (`B`)

The base is the equal common-bps mean of three strict-OOF, 120-causal-field outputs:

| Component | Target / purpose |
|---|---|
| B0 | Strict-R3 opportunity score: `P(clear) − 0.5 × P(adverse)` |
| Direct efficiency | Direct policy-conversion efficiency |
| Direct timing | Direct timing / conversion |
| Enhanced base | `(B0 + direct efficiency + direct timing) / 3` |

The enhanced base is routed independently at each decision timestamp: retain `max(1, ceil(30% × available candidates))`, breaking ties by candidate ID. It has clear raw ordering uplift versus B0 in the same May–July block: timestamp-local top-1/2/5/10% realised rich-policy net is `+144.2 / +124.7 / +110.4 / +81.0 bps`, respectively, versus B0 `+99.9 / +80.5 / +70.8 / +49.2 bps`.

### 2. Correction heads (`T6`, `T9`)

The two heads are the entire consensus layer. The other exploratory heads are not part of this contract.

| Head | Frozen physical slot | Target | Query / model | Causal features | Training support |
|---|---|---|---|---:|---|
| T6 | `cap80_ordinary` | Five bins of `rank(realised rich-policy net) − base_rank`; cut points `−.20, −.05, .05, .20` | 4-hour UTC cycle × side; LightGBM L2, 120 trees, depth 5, 31 leaves, min child 300, LR .035, feature/bagging .82, L1 .02, L2 2.0 | 102 | Uniform (`S0_uniform`) |
| T9 | `cap120_equal_month` | Five exit states: stop, timeout, smooth protection, regular trailing, large trailing | Same query and fixed tree geometry | 73 | Mild coarse triple-barrier balance (`S5_tbm_coarse`) |

The frozen physical-slot receipt is:

```text
data_perp/artifacts/strict_r3_o3v2_t6t9_consensus_contract_20260825_v1/
  selected_physical_slots.json
SHA-256: bdcd87049184f586e3a64e9a6fe5cf74907be5785132123f406bbefaed5e41bc
```

Neither correction rank is stand-alone alpha. Their purpose is conditional correction of the enhanced-base rank. On routed candidates, the S11 score gives timestamp-local top-1/2/5/10% rich-policy net of `+180.68 / +175.75 / +125.20 / +88.90 bps`; its score-to-realised-net rank IC is `0.1869`.

### 3. Current and BCF MC1 admission mappers

`Current` and `BCF` are **two MC1 expected-EV mappers**, not correction heads. Each is fitted separately, strictly prequentially, on six complete preceding calendar months of target-free S11 predictions joined only afterwards to resolved rich-policy outcomes.

Ordered mapper inputs:

```text
final_score
base_rank42
base_anchor_bps
correctness_rank
t6_consensus_rank
t6_combined_rank = 0.75 × B + 0.25 × T6
t9_consensus_rank
t9_combined_rank = 0.75 × B + 0.25 × T9
```

Both add their causal recent-shift component. The fixed MC1 capacity is HistGradientBoostingRegressor depth 2, at most 4 leaves, 80 iterations, learning rate .04, L2 20, minimum leaf 100, seed 1729. The `C1 depth-2 / four-leaf` confirmation is bit-identical to the original C0 result; larger capacity did not advance.

The two maps have near-identical but independently fitted predictions (Pearson `0.9964 / 0.9927 / 0.9962` and Spearman `0.9995 / 0.9991 / 0.9990` in May/June/July). They are therefore a conservative corroboration gate, not independent ranking alpha. Do not lower the frozen `+50 bps` threshold: the dual 50–60 bps band realised only `+38.53 bps` against a `+54.42 bps` mean prediction.

## Strict-OOS evidence and causality

**Ledger:** November 2025–July 2026 homogeneous target-free T6/T9 receipts.  
**Evaluation:** May–July 2026.  
**Outcome:** canonical reconciled rich-policy net, joined only after target-free scores.  
**Portfolio:** one shared chronological constrained auction, unchanged from the existing canonical research policy.

The final audit passed:

| Causality invariant | Result |
|---|---:|
| Target-free Current score rows | 307,384; no prohibited outcome columns |
| Target-free BCF score rows | 307,384; no prohibited outcome columns |
| Current/BCF candidate IDs | Exact match |
| Valid prediction rows | 106,260 |
| Label availability | Every valid label resolves after its decision timestamp |

Source audit: [MC1_ABLATION_REPORT.md](../data_perp/artifacts/strict_r3_t6t9_mc1_ablation_audit_20260825_v7/MC1_ABLATION_REPORT.md) and [correctness_report.json](../data_perp/artifacts/strict_r3_t6t9_mc1_ablation_audit_20260825_v7/correctness_report.json).

## Selection boundary

The following arms were rejected:

- all tested supplemental geometry/CMI feature blocks: none provided repeatable downstream improvement under the advancement gate;
- deeper MC1 models: depth-3 variants weakened drawdown or total economics;
- more direct T9 authority: T9 is useful as conditioning, not material raw score authority;
- threshold relaxation: the 50–60 bps admission band is overconfident.

`G1` had the largest total-bps number (`+319,118`) but lowered EV/trade and failed July robustness. It is not canonical. S11 is selected for the best combined participation, total net, per-trade EV, month/week stability, and drawdown result among the predeclared main score contracts.

The legacy matched-live row is an **external** comparator only because it does not share the enhanced-base candidate population:

| External legacy comparator | Entries | Net EV/trade | Total net bps | Worst month | Worst week | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|
| Legacy matched-live control | 1,896 | +151.31 | +286,875 | +127.85 | +97.11 | −17.87% |
| S11 minus legacy | +348 | −10.81 | +28,404 | +0.24 | −14.88 | +4.12 pp |

Do not interpret this as a like-for-like promotion result. S1 is the valid within-family control.

## Required next validation

Before any live replacement, freeze this exact file, source receipts, policy, route, MC1 features, capacity, authority, and threshold. Then run a later untouched forward period and require:

1. target-free identity and causal-label audit to pass;
2. no material deterioration in worst week, drawdown, concentration, or CVaR;
3. S11's incremental admissions to remain positive and economically useful;
4. no reliance on a single month, asset, or delayed-policy outcome convention;
5. a separately authorized inference-parity and execution review.

No live configuration is superseded by this document.

## Reproduction references

| Purpose | Path |
|---|---|
| Canonical research config | `config/strict_r3_o3v2_t6t9_s11_research_canonical_20260825_v1.json` |
| Frozen head selection | `data_perp/artifacts/strict_r3_o3v2_t6t9_consensus_contract_20260825_v1/selected_physical_slots.json` |
| S11 scorer / MC1 runner | `scripts/run_strict_r3_t6t9_mc1_ablation.py` |
| Independent causality audit | `scripts/audit_strict_r3_t6t9_mc1_ablation.py` |
| Final audited artifacts | `data_perp/artifacts/strict_r3_t6t9_mc1_ablation_audit_20260825_v7/` |
| Rich-policy ledger | `data_perp/artifacts/strict_r3_enhanced_base_rich_policy_labels_reconciled_20260823_v1/canonical_reconciled_policy_labels.parquet` |
| Homogeneous T6 ledger | `data_perp/artifacts/strict_r3_o3v2_t6_uniform_homogeneous_202511_202607_20260825_v3/` |
| Homogeneous T9 ledger | `data_perp/artifacts/strict_r3_o3v2_t9_tbm_homogeneous_202511_202607_20260825_v3/` |
