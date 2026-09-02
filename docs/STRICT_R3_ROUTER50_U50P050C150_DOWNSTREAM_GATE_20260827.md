# Router50 U50 p=.50/cap=150 — downstream gate

**Date:** 2026-08-27
**Status:** `REJECTED_AT_DOWNSTREAM_GATE`
**Scope:** offline, long-only research; no live, canonical, exchange, or model-bundle mutation.

## Decision

`U50_p050_c150` improved strict-OOF Router50 utility recall, but did **not** improve the full routed downstream stack. It is therefore not eligible for Router weight search, feature selection, or HPO. The frozen `P8u_floor100_cap250` Router50 control remains retained.

The comparison refit every downstream stage separately for each router:

```text
timestamp-local top-50% route
-> routed-only E/T base models
-> cap80-ordinary + cap120-equal-month consensus
-> independent Current and BCF MC1 maps
-> dual MC1 expected EV >= +50 bps
-> one chronological constrained portfolio
```

The router has no numerical authority downstream. Both candidates use the same full-universe source, policy ledger, route fraction, T6/T9 contract, admission threshold, and portfolio contract.

## Strict-OOF Router-only screen, Jul 2025–Jul 2026

| Metric | P8u control | U50 p=.50/cap=150 | Delta |
|---|---:|---:|---:|
| R50 utility recall | 76.84% | 78.99% | +2.14 pp |
| R50 count recall | 73.14% | 77.19% | +4.04 pp |
| R100 count recall | 80.91% | 82.59% | +1.68 pp |
| Router composite | 0.7690 | 0.7926 | +0.0236 |
| Fold stability | 0.7543 | 0.7789 | +0.0246 |
| Worst fold | 0.7055 | 0.7399 | +0.0343 |

## Matched downstream result, Apr–Jul 2026

| Metric | P8u control | U50 p=.50/cap=150 | Delta |
|---|---:|---:|---:|
| Dual-MC1 admitted candidates | 10,806 | 12,097 | +1,291 |
| Portfolio entries | 2,943 | 3,048 | +105 |
| Net EV / trade | +151.58 bps | +145.98 bps | -5.60 bps |
| Total net EV | +446,101.67 bps | +444,958.84 bps | -1,142.83 bps |
| Worst month | +120.81 bps | +119.72 bps | -1.09 bps |
| Worst week | +74.04 bps | +71.74 bps | -2.30 bps |
| Max drawdown | -20.21% | -18.38% | +1.83 pp |
| Positive-month fraction | 100% | 100% | unchanged |

The candidate’s improved drawdown and participation do not offset the loss of capital efficiency and total realized policy net. It fails the hard downstream-improvement condition.

## Causality/identity checks

Both completed receipts confirm:

- target-free score panels contain no outcomes;
- router-to-source and Current-to-BCF identities are exact;
- all routed base training rows are router selected;
- base, consensus, and MC1 training use only prior resolved labels;
- router outputs have no numeric authority in base, consensus, or MC1 features.

## Immutable receipts

- Control: `data_perp/artifacts/strict_r3_router50_p8u_control_routed_et_t6t9_mc1_gate_20260827_v1/`
- Challenger: `data_perp/artifacts/strict_r3_router50_u50p050c150_routed_et_t6t9_mc1_gate_20260827_v1/`

No live or canonical contract changed.
