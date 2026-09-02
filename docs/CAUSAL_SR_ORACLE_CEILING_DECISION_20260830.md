# Causal S/R Oracle Ceiling — Decision Record

## Decision

Do **not** promote or further broaden the current causal S/R heads.  They
remain offline diagnostic/research components only.  No live feature,
inference bundle, entry authority, continuation policy, or execution path was
changed.

This decision follows the requested falsification sequence: a strict
out-of-sample prediction audit followed by deliberately **non-causal oracle**
ceiling tests.  The oracle receives the first resolved future S/R interaction
for the relevant pre-decision zone, so it is forbidden from causal replay,
calibration, inference, or live trading.  If even that information does not
materially improve the incumbent portfolio outcome, training additional S/R
specialists is not justified.

The established H4 result is separate from this S/R study: the rich-parent
policy produced +32.49 bps/trade, H4 activation +55.71 (+23.21), and H4 with
20% tighter giveback +57.80 (+25.31), with maximum drawdown improving from
-43.11% to -19.22%.  That is the approximately +20–25 bps uplift previously
reported; it is not an S/R uplift.

## Contracts and safeguards

- Frozen causal engine: `causal_sr_engine_2025_train_2026_score_20260830_v1`.
  It uses only completed local 15-minute OHLCV, with a 45-day warm-up and
  labels resolving over at most eight hours.
- Frozen prequential heads:
  `causal_sr_heads_oof_20260830_v3_entrypivotfix`; each held month fits only
  interactions resolving before that month.
- Existing entry and H4 continuation feature/policy/auction contracts remain
  unchanged.  The only studied variation is the addition of causal S/R fields
  or non-causal oracle fields to the relevant model input.
- Unreadable archival rich-policy label parts were explicitly excluded from
  **every** comparison arm, never imputed or replaced: CELO, CRCLX, EIGEN,
  ETC, GAS, GMT, JTO, KAITO, LDO, MEW, ONG, PORTAL, RENDER, REZ, SAGA, SSV,
  TON, XMR, and ZRO.  This makes the entry comparison matched but does not
  retroactively claim full-universe coverage.

## Diagnostic repair

`sr_reaction_magnitude_q50` has always been trained on `reaction_MFE_atr`,
but its reporting metric incorrectly compared it with generic reaction
strength.  The reporting code now evaluates that head against
`reaction_MFE_atr`; existing fitted values and OOF predictions are unchanged.

Focused regression coverage:

```text
python3 -m pytest -q tests/test_causal_sr_oracle_diagnostic_contract.py
# 2 passed
```

## Predictive potency

The heads are real interaction predictors.  In particular, their extreme
tail calibration is strong across every held month (February–August 2026):

| OOF head | p90 lift | p95 lift | p98 lift | Minimum monthly p90/p95/p98 |
|---|---:|---:|---:|---:|
| Conditional reaction strength | +0.223 | +0.338 | +0.491 | +0.208 / +0.322 / +0.472 |
| Accepted-break probability | +0.218 | +0.273 | +0.284 | +0.207 / +0.254 / +0.273 |
| Reaction-MFE q50 | +2.51 ATR | +4.42 ATR | +9.15 ATR | positive in every month |
| Prior strength | +0.018 | +0.023 | — | near-inert |

Conditional on H4, the MFE heads retain some novelty (resistance residual IC
0.191, minimum 0.116; support residual IC 0.146, minimum 0.071).  The
accepted-break head does not: its residual IC is negative, consistent with H4
already representing that information.  Predictive potency alone therefore
does not establish a profitable portfolio use.

## Portfolio-constrained ceiling tests

### Continuation: H4 + 20% tighter giveback

All arms use the same frozen E2 entries, H4 training schedule, rich policy and
global portfolio auction.  June–July is selection; August remains held out.

| Arm | Scope | Trades | Net EV/trade | Total net EV | Max DD | Sortino |
|---|---|---:|---:|---:|---:|---:|
| H4 control | Jun–Jul | 986 | +59.53 | +58,700 | -16.55% | 0.268 |
| H4 + causal S/R | Jun–Jul | 987 | +62.13 | +61,322 | -15.91% | 0.280 |
| H4 + oracle S/R | Jun–Jul | 987 | +60.88 | +60,085 | -16.58% | 0.272 |
| H4 control | August | 532 | +54.98 | +29,251 | -19.21% | 0.229 |
| H4 + causal S/R | August | 529 | +47.04 | +24,883 | -22.54% | 0.187 |
| H4 + oracle S/R | August | 535 | +46.99 | +25,142 | -23.08% | 0.192 |
| H4 control | Jun–Aug | 1,518 | **+57.94** | **+87,951** | **-19.21%** | **0.253** |
| H4 + causal S/R | Jun–Aug | 1,516 | +56.86 | +86,205 | -22.54% | 0.245 |
| H4 + oracle S/R | Jun–Aug | 1,522 | +56.00 | +85,227 | -23.08% | 0.242 |

The perfect oracle cannot improve full OOS continuation performance; it is
worse in August and on all-OOS risk-adjusted terms.  Further continuation S/R
head development is therefore falsified.

### Entry: E2 q50 agreement replacement

Each challenger has a four-month prior-only pair fit, preserves the 20–30 bps
reserve / marginal ordinary-core replacement topology, the H0/H3 intersection
and the global portfolio auction.  The causal and oracle rows below are
compared primarily to their matched retrained control; the frozen incumbent is
shown because it is the deployable reference.

| Arm | Trades | Net EV/trade | Total net EV | Max DD | Worst week | Sortino |
|---|---:|---:|---:|---:|---:|---:|
| Frozen E2 incumbent | 1,155 | **+14.96** | **+17,278** | **-68.89%** | -41.93% | **0.0418** |
| Matched retrained control | 1,156 | +10.27 | +11,876 | -75.57% | -44.59% | 0.0267 |
| + causal OOF S/R fields | 1,147 | +14.27 | +16,373 | -70.51% | -40.39% | 0.0402 |
| + non-causal future S/R oracle | 1,155 | +14.62 | +16,890 | -72.72% | -40.75% | 0.0399 |

The oracle adds only +4.35 bps/trade versus the weaker matched retrained
control, and still fails to beat the frozen deployable E2 on EV/trade, total
EV, drawdown, or Sortino.  August has a small oracle gain (+17.02 versus
+13.50 bps/trade) but it is not enough to overcome the complete OOS result.

## Result

The S/R heads capture real local interaction behaviour, but their ceiling is
too small and unstable once the already-selected E2/H4 population and global
portfolio constraints are respected.  Do not proceed to source-specialist,
distributional, or policy-target S/R heads.  Revisit only if a new independent
entry/exit architecture leaves a clearly demonstrated S/R residual opportunity.

## Reproducibility

- Oracle and conditional-predictive audit:
  `scripts/run_causal_sr_oracle_audit.py`
  → `data_perp/artifacts/causal_sr_oracle_audit_20260830_v1`
- Continuation ceiling:
  `scripts/run_causal_sr_oracle_continuation_ablation.py`
  → `data_perp/artifacts/causal_sr_oracle_continuation_ceiling_20260830_v1`
- Entry ceiling:
  `scripts/run_causal_sr_oracle_entry_ablation.py`
  → `data_perp/artifacts/causal_sr_oracle_entry_ceiling_20260830_v1`
- Corrected head diagnostics:
  `scripts/run_causal_sr_heads.py`
- Regression tests:
  `tests/test_causal_sr_oracle_diagnostic_contract.py`

The original causal-head handover remains at
`docs/CAUSAL_SR_HEADS_ENTRY_CONTINUATION_RESEARCH_HANDOVER_20260830.md` and is
superseded for promotion decisions by this record.
