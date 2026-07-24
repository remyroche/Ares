# R2 Deterministic Contract Test Log

Date: 2026-07-24
Storage/model timezone: UTC
Source revision under test: `0cf4e1f3bbde7e890609697432446a1b6b3a4149`
Python: 3.12.2
DuckDB restored from the project lock: 1.5.4

## Focused suite

Command:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m pytest
  -p no:cacheprovider -q
  tests/test_path_auxiliary_targets.py
  tests/test_path_auxiliary_lgbm.py
  tests/test_path_archetype_labels.py
  tests/test_path_archetype_support.py
  tests/test_catboost_archetype_classifier.py
  tests/test_execution_ev_labels.py
  tests/test_execution_ev_meta.py
  tests/test_execution_ev_model_ablation.py
  tests/test_materialize_execution_ev_joined_handoff.py
  tests/test_run_execution_ev_meta.py
  tests/test_pipeline_stage_manifest_schema.py
```

Result: **PASS — 138 passed in 9.53 seconds.**

## Broader suite

Command:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m pytest
  -p no:cacheprovider -q
  tests/test_materialize_path_archetype_candidates.py
  tests/test_run_path_auxiliary_lgbm_models.py
  tests/test_run_catboost_path_archetype_classifier.py
  tests/test_path_archetype_geometry_search.py
  tests/test_materialize_execution_ev_auxiliary_oof.py
  tests/test_materialize_execution_ev_catboost_refinement_oof.py
  tests/test_materialize_execution_ev_alpha_oof.py
  tests/test_materialize_execution_ev_joined_handoff.py
```

Initial result after restoring DuckDB: **FAIL — 143 passed, 1 failed.**
The remaining failure showed that DuckDB persisted `path_cost_return` as
single-precision `FLOAT`, which did not reproduce the declared Python cost
value exactly.

Correction: `scripts/materialize_path_archetype_candidates.py` now persists
`path_cost_return` as `DOUBLE`.

Final result: **PASS — 144 passed, 6 warnings in 16.00 seconds.** The warnings
are existing PyArrow conversion warnings about discarding non-zero nanoseconds;
they do not alter the UTC join keys exercised by these tests.

After adding the canonical Pack-B per-side lineage guard to the alpha OOF
adapter, the same broader suite was rerun with five additional lineage tests.
Result: **PASS — 149 passed, 6 warnings in 17.70 seconds.**

## Final targeted verification

The stage-manifest and corrected candidate materializer tests were rerun
together.

Result: **PASS — 6 passed in 0.43 seconds.**

## Gate interpretation

This log proves the synthetic deterministic test suites pass in the recorded
environment. It does not by itself prove R2 complete: the required
artifact-level class-schema, horizon, cost, row/fold, and frozen-state hash
reconciliation must still be generated against the P0 artifacts.
