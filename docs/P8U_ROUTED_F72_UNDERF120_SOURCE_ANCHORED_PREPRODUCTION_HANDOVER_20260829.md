# P8U Router50 + F72 + Under F120 — Source-Anchored Preproduction Handover

## Status

This is the current **long-only research/preproduction** contract. It is
strictly offline: exchange I/O and order submission are disabled. It
supersedes the warm-state claims in the prior v6 handover, while retaining
that document as historical research evidence.

The machine-readable contract is
[`strict_r3_p8u_routed_f72_underf120_research_canonical_20260829_v8_sourceanchored.json`](../config/strict_r3_p8u_routed_f72_underf120_research_canonical_20260829_v8_sourceanchored.json).
The complete hash-bound bundle is
[`bundle.json`](../data_perp/artifacts/strict_r3_p8u_preproduction_bundle_20260829_v18_sourceanchored/bundle.json),
SHA-256 `cdb4f15ab49e9f1a9f473debf8fa66f57858076ba263a3cffa5117aebcf09486`.

### Canonical entry/exit extension

This document remains the upstream Router50/F72/Under/dual-MC1 preproduction
contract. Its canonical research extension is now
[`P8U E2 q50 Agreement + H4 Giveback-20`](P8U_E2_Q50_AGREEMENT_H4_GIVEBACK20_CANONICAL_HANDOVER_20260830.md):
E2 is a replacement-only entry intersection after ordinary BCF top-two
selection, and H4 applies 50% earlier trailing activation plus a 20% tighter
giveback after a non-negative completed 15-minute continuation prediction.
That extension is hash-bound and research-canonical, but is **not** bound into
this older preproduction bundle or any exchange-writing gateway.

## Frozen scoring chain

```text
complete target-free point-in-time universe
→ P8U Router timestamp-local top 50% identity gate
→ F72 Raw-bps CatBoost Base
→ Under F120 Rank-XENDCG confirmation head
→ BCF = Base rank
→ Current = 0.75 × Base rank + 0.25 × Under rank
→ independently selected six-complete-month BCF and Current MC1 maps
→ admit only when both expected policy net values are ≥ +50 bps
→ BCF MC1 expected-net auction priority
→ rich 15-minute policy proxy and portfolio constraints
```

The Router numeric value has no feature authority downstream. Base, Under,
both score coordinates, and both MC1 maps receive only the exact Router50
identities.

| Component | Frozen contract |
|---|---|
| Router | 30 causal fields; LightGBM Rank-XENDCG; timestamp-local top 50% identity gate |
| Base | 72 causal fields; Raw-bps CatBoost QueryRMSE, `tail_linear_125` |
| Under | 120 causal fields plus nine deterministic Base-query geometry fields; LightGBM Rank-XENDCG confirmation head |
| MC1 | Separate BCF and Current packages fitted on exactly six adjacent complete calendar months; both require ≥ +50 policy-net bps |
| Policy | Frozen rich 15-minute policy proxy; 100-bps round-trip cost embedded exactly once |

The prior v6 handover contains the frozen model targets, HPO, feature-selection
receipts, historical layer metrics, and portfolio evidence. This document is
the authoritative operational-contract update.

## Source-anchored feature contract

All 175 unique Router/Base/Under fields are materialised once over the full
160-symbol contemporaneous universe before Router50 selection. Feature state
is persisted across: raw rolling operators, causal transforms, derived/nested
history, OI-IQR, fixed-FFD, spectral, grouped, EWMA, and regime-transition
operators.

The exactness baseline is the immutable source panel, not the older unanchored
August feature parquet:

| Artifact | Identity |
|---|---|
| Primitive source panel | [`source_panel_state.joblib`](../data_perp/artifacts/strict_r3_p8u_canonical_source_state_20260828_v1/source_panel_state.joblib), SHA-256 `be21442c191e6aaf47ba2f922d416f37eb795a879405f37784749f022bb6945d` |
| Source-anchor manifest | [`source_anchor_manifest.json`](../data_perp/artifacts/strict_r3_p8u_source_anchored_reference_20260829_v2/source_anchor_manifest.json), SHA-256 `ad9c07ff9b769ae7f9c72aa4e3e5a9c7cbc9ab981ef4145a7eba4106ce001600` |
| Sealed warm state | [`state_bundle_manifest.json`](../data_perp/artifacts/strict_r3_p8u_canonical_warm_state_bundle_20260829_v4_sourceanchored/state_bundle_manifest.json), SHA-256 `5aac11efc8aeff250a144488865d782ba54a08d593bd2b58747a4beabc73a3b7` |
| Required feature plan | [`required_feature_plan.json`](../data_perp/artifacts/strict_r3_p8u_preproduction_bundle_20260828_v8/audit/required_feature_plan.json), 175 fields, SHA-256 `540fec0dda092b0974bafd8b61da1000b1e4096a907725db76cd309742d5db3c` |

The bounded history remains **1,536 hours**. It is not lengthened at runtime.
The repaired state uses exact rolling sufficient statistics and per-feature
transform state. A stale historical LDO OI input could not be reconstructed
from the unanchored legacy panel, so source hash binding prevents an invalid
exact-parity claim instead of guessing a value.

## Parity and security evidence

Three sequential, source-anchored checks passed after one cold bootstrap:

| Checkpoint | Audited matrix | Result |
|---|---:|---|
| t10 | 85 symbols × 175 fields | zero mismatches |
| t11 | 85 symbols × 175 fields | zero mismatches |
| t12 | 85 symbols × 175 fields | zero mismatches; max observed delta `2.384185791015625e-07` |

The final receipt is
[`parity_summary.json`](../data_perp/artifacts/strict_r3_p8u_canonical_stateful_tail1536_sourceanchored_t12_20260829_v1/parity_summary.json).
The source-anchor sealer rejects a reference that lacks the matching primitive
source-panel hash. The preproduction bundle verifies all 38 bound artifacts
before it loads a model; any changed file fails closed.

Target-free assembled scoring was verified on the t12 anchored panel:

| Stage | Rows |
|---|---:|
| Input candidate universe | 85 |
| Router50-eligible | 43 |
| Base / Under / dual-MC1 scored | 43 |
| Dual-MC1 admitted | 2 |

The Base/Under/MC1 identity set was exactly the Router50-eligible set.

## Efficient inference requirements

1. Restore the sealed state once into a single long-lived worker.
2. Append only the newly completed primitive timestamp and atomically commit
   state plus target-free feature receipt after success.
3. Derive the 175-field union automatically from the hash-bound contracts;
   reject missing contemporaneous fields before Router scoring.
4. Run full-universe feature generation once; apply Router50 only after
   cross-sectional features are complete; reuse `float32` matrices and warm
   Router/Base/Under/MC1 models in the same process.
5. Re-hash artifacts on load and whenever their stat fingerprint changes.
   There is no fallback to a latest MC1 model or a reselected feature list.

Do not split Router and downstream feature computation into separate graphs
unless a future two-stage implementation proves byte-parity. Do not rebuild
1,536 hours on each decision; that interval is retained only as the verified
bootstrap/reference boundary.

## Remaining promotion blockers

- Run target-free assembled inference/replay parity across a new appended
  decision sequence, using this exact bundle.
- Validate the rich 15-minute policy against a separately named exact
  one-minute execution contract.
- Collect untouched forward evidence without retuning.
- Obtain separate explicit authorization for any exchange-writing activity.

No artifact here authorizes live trading.
