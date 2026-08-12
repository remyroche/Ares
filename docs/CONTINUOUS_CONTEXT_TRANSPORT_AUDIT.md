# Continuous context transport audit

`scripts/audit_regime_transport_selection.py` screens continuous regime and
market-context fields before they enter a residual/conversion model. It is not
a generator search and it never uses cluster IDs, GMM posteriors, state
probabilities, memberships, or state labels.

The audit applies four compact checks to each candidate field, on top of an
optional frozen reference score contract:

1. A 90% coverage and non-constant gate.
2. Chronological within-era and cross-era economic MDA. Each model is fitted
   before the evaluation period minus the declared label-horizon embargo, using
   a thresholded net-opportunity label. Only the held-out candidate column is
   deterministically rotated for MDA. Economic reporting remains realised net
   bps in the globally ranked top fraction.
3. Standardised effect-direction consistency across those chronological fits.
4. A diagnostic-only era-separation proxy. It uses no outcomes, but is a
   representation-selection diagnostic (early part of each historical era vs
   its later part), not untouched OOS evidence and never an inference feature.

The output classifications are conservative:

- `INVARIANT_CORE`: positive portable contribution, stable direction, and low
  era-separation proxy importance.
- `SMOOTHLY_CONDITIONED`: no strong portability failure, but context is better
  retained as a continuous conditional field than as a universal core signal.
- `ERA_SHORTCUT`: poor cross-era economics together with high era-separation.
- `CONTROLLER_DIAGNOSTIC`: trust/admission/controller fields; diagnostic only.
- `REJECTED`: failed coverage, membership exclusion, or economic transport.

Example:

```bash
python scripts/audit_regime_transport_selection.py \
  --input data_perp/artifacts/candidates.parquet \
  --features-file data_perp/contracts/continuous_regime_context.txt \
  --reference-feature score_residual_expected_ev \
  --era-column evaluation_month \
  --threshold-bps 50 --embargo-hours 12 --top-fraction 0.10 \
  --output-dir data_perp/artifacts/continuous_context_transport_v1
```

The resulting `selection_manifest.json` explicitly records the thresholded
label, embargo, global ranking convention, coverage gate, and excluded
membership/cluster-field contract. Do not promote a field solely because its
within-era MDA is positive; it must pass its cross-era and direction checks.
