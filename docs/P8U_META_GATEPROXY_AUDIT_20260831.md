# P8U Meta GateProxy retrospective audit — 2026-08-31

## Scope and decision

This is a **receipt-only retrospective audit**, not a new HPO or a selection
event.  It uses the frozen P0 Ridge GateProxy scores and the already-published
strict-MC1 constrained-portfolio summaries for the confirmed final-HPO trials.
No Meta score, MC1 map, policy label, live bundle, or exchange state was
fitted, changed, or promoted.

The audit answers whether GateProxy was sufficiently useful as a cheap proxy
for downstream MC1 utility.  It does **not** justify a new GateProxy V2: the
new HPO-bank sample is only five confirmed trials per family, four trials per
family remain unconfirmed, and the retained Under incumbent is ranked too low.
Any GateProxy V2 work must be predeclared on a later, separate bank.

## Inputs

- Frozen GateProxy P0 Ridge model:
  `strict_r3_p8u_meta_downstream_proxy_allcontracts88_20260830_v1`
  (SHA-256 `3d8d07385d72ac6002e5b0c7e5f3800b6a6cef363dfbfd7251bfedbfcb03adb3`).
- Under and State final-HPO GateProxy score receipts.
- Exact published strict-MC1 constrained-portfolio outcomes for February–July
  2026.  The outcome is realised constrained portfolio net EV per realised
  trade, not an HPO training proxy.
- Retained Under control (`lgbm_hpo_05_depth4_sparse`) as a retrospective
  reference only.

## Results

| Family | Confirmed / bank | Spearman: proxy → constrained EV | Actual winner | Proxy rank of winner | Regret @1 | Regret @3 | Regret @5 |
|---|---:|---:|---|---:|---:|---:|---:|
| Under | 5 / 9 | -0.50 | `lgbm_final_08_depth6_guarded` | 3 | 2.19 bps | 0.00 bps | 0.00 bps |
| State | 5 / 9 | +0.50 | `lgbm_final_06_depth5_capacity` | 2 | 1.74 bps | 0.00 bps | 0.00 bps |

`Regret@k` is the actual winner’s constrained net EV per trade minus the best
actual constrained EV among the proxy top-k.  It is a **lower-bound diagnostic**:
the four unconfirmed bank candidates in each family are excluded, rather than
being treated as losses.

### Containment

| Family | Actual top-3 contained in proxy top-3 | Actual top-3 contained in proxy top-5 | Winner in proxy top-3 / top-5 |
|---|---:|---:|---|
| Under | 1 / 3 | 2 / 3 | yes / yes |
| State | 2 / 3 | 3 / 3 | yes / yes |

### Retained incumbent diagnostic

The retained Under control realised **+135.73 bps/trade**, above every newly
confirmed Under HPO trial.  Yet its frozen GateProxy score placed it **4th of
10** when added retrospectively to the nine-trial Under bank.  Three lower-EV
new trials were ranked above it.

That is the key falsification result: GateProxy was useful enough to keep the
best *new* candidate inside its top-three, but it is not reliable enough to
replace the established incumbent or to be further tuned from this small bank.

### Uncertainty diagnostic

The four-surrogate spread is an acquisition uncertainty in GateProxy target
units, **not** a calibrated interval for portfolio bps.  As a deliberately
descriptive leave-one-out affine conversion to portfolio bps, apparent 1σ / 2σ
coverage was 60% / 100% for Under and 100% / 100% for State (five trials each).
These figures are far too small and indirect to support a calibration claim.

## Conclusion

1. For the confirmed new HPO trials, proxy top-3 and top-5 had zero observed
   regret in both families; a top-three confirmation funnel would have retained
   each new-family winner.
2. The proxy does not have stable enough *global* ordering: Under’s negative
   rank correlation and the incumbent’s fourth-place proxy rank are material
   counter-evidence.
3. Therefore retain the existing Meta contract.  Do not run GateProxy V2
   optimisation, change the incumbent, or promote any final-HPO challenger from
   this audit.
4. If future work is warranted, predeclare a later independent HPO bank and
   test only the frozen proxy/uncertainty diagnostic before using it for any
   candidate funnel.

## Frozen operating rule for future independent HPO banks

GateProxy is a **top-three challenger funnel**, never an incumbent-versus-
challenger arbiter and never a promotion mechanism.

```text
new independent HPO bank
→ cheap frozen GateProxy descriptors
→ rank challengers only
→ strict downstream confirmation:
     incumbent, always
     + proxy top-3 challengers
     + one predeclared uncertainty/diversity challenger when compute permits
→ strict MC1 and constrained portfolio replay
→ select only from the confirmed downstream results
```

The retained incumbent is an external anchor: it bypasses GateProxy ranking.
This is necessary because the audit showed it to be the best realised Under
contract while GateProxy placed it fourth among the bank-plus-reference set.

The future success criteria for GateProxy are therefore:

- `Recall@3(best challenger)`;
- `Regret@3` among confirmed challengers;
- repeatability of `P(best challenger ∈ proxy top-3)` across independent
  HPO banks.

Spearman and exact global ordering remain diagnostics only.  Do not retune
the proxy coefficients using any downstream period used to assess this rule.

## Receipts

- [Audit trial table](../data_perp/artifacts/strict_r3_p8u_meta_gateproxy_portfolio_audit_finalhpo_20260831_v1/gateproxy_audit_trials.parquet)
- [Family summary](../data_perp/artifacts/strict_r3_p8u_meta_gateproxy_portfolio_audit_finalhpo_20260831_v1/gateproxy_audit_family_summary.parquet)
- [Regret table](../data_perp/artifacts/strict_r3_p8u_meta_gateproxy_portfolio_audit_finalhpo_20260831_v1/gateproxy_audit_regret.parquet)
- [Correctness receipt](../data_perp/artifacts/strict_r3_p8u_meta_gateproxy_portfolio_audit_finalhpo_20260831_v1/correctness_report.json)
