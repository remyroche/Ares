# Root-cause diagnostic acceptance ledger

Authoritative specification: `/Users/remyroche/.codex/attachments/465b0473-9c18-4cde-8a8a-6709c73bd999/pasted-text-1.txt`.

Status: `[x]` means proven by current artifacts and tests; `[ ]` means incomplete or unproven.

## Architecture boundary

- [ ] Keep exactly two trainable pipeline layers: the directional/alpha base head and its stopped-gradient residual learner.
- [ ] Do not train or integrate auxiliary, CatBoost, execution-EV meta, timing, action, sizing, or portfolio heads.
- [ ] Report base directional metrics separately from residual/economic metrics.
- [ ] Keep every diagnostic research-only; no promotion, sizing, concurrency, exposure, or portfolio optimization.

## Stage 0 — canonical substrate

- [ ] Freeze row identity, product/side, UTC timing, entry, H12 horizon, policy, path, feature cutoff, and label availability.
- [ ] Materialize gross, fee, spread, slippage, total cost, and net as separate fields with exact reconciliation.
- [ ] Prove candidate identity and ordered-row parity across comparison arms.
- [ ] Prevent inverse/linear perpetual population mixing.
- [ ] Build a feature-target algebra/proximity scanner covering targets, arithmetic components, realised costs, future fills/path summaries, and target-derived mappings.
- [ ] Produce all four Stage-0 deliverables and their manifests/seals.

## Stage 1 — target/population oracle ladder

- [ ] Run null/prior and UTC-day-grouped shuffled controls.
- [ ] Run realised gross and realised net H12 oracle rankings.
- [ ] Run hindsight event-state and best-permitted-action oracles, or explicit `NOT_RUN` with direct evidence.
- [ ] Report pooled top 1/5/10/20%, side, month, support, gross/net, bootstrap, and regret.
- [ ] Run 0/1/5/10-minute delay, path-resolution, barrier, timeout, and entry-price sensitivity, or explicit `NOT_RUN` where immutable inputs are absent.
- [ ] Report rank correlation, label agreement, top-tail Jaccard, and economic sensitivity.

## Stage 2 — feature-information audit

- [ ] Verify every candidate feature's timestamp, staleness, live reproducibility, target overlap, future dependency, and OOF/prequential dependency.
- [ ] Run side-local chronological transported univariate IC/AUC/PR-AUC, decile spread, sign stability, missingness, and concentration.
- [ ] Run nested chronological mechanism-group diagnostics on identical rows without pooled full-period feature selection.
- [ ] Run residual probes using same causal, excluded causal, and future-oracle features and classify the residual-information source.
- [ ] Run adversarial/PSI/JS/Wasserstein, missingness, prediction, and calibration drift by required slices.

## Stage 3 — learning-efficiency ladder

- [ ] Compare null, linear, additive, shallow tree, production-like, high-capacity causal, and future-feature oracle capacities on identical folds/rows.
- [ ] Preserve only the base directional head plus residual learner architecture.
- [ ] Report train/OOF gaps, seed dispersion, sample/history/feature/capacity curves, calibration, and gross/net economics.
- [ ] Run predeclared semi-synthetic alpha recovery and diagnose pipeline failure if the injected component is not recovered.
- [ ] Calculate null-to-causal, production-to-causal, causal-to-future gaps and economic regret; use ratios only with safe denominators.

## Stage 4 — metric concordance

- [ ] Build the declared statistical/calibration/economic metric matrix across all valid arms.
- [ ] Measure development-metric association with later OOS gross/net, worst-month/side, and paired-bootstrap economics.
- [ ] Recommend an evidence-based lexicographic selection hierarchy with correctness and gross economics first.

## Stage 5 — execution waterfall

- [ ] Materialize reference/ideal-entry, executable-entry, delayed-entry, frozen-policy gross, and post-cost net on identical candidates.
- [ ] Reconcile candidate-selection, entry-transfer, delay/slippage, policy-geometry, and cost losses.
- [ ] Run the controlled entry/resolution/delay/geometry/cost factorial with fixed rows/evaluator, or explicit `NOT_RUN` for unavailable immutable inputs.
- [ ] Identify the economic contribution of each failure layer in bps.

## Stage 6 — policy audit

- [ ] Keep policy targets incremental and compare always-baseline, always-alternative, oracle, and learned action where admissible.
- [ ] Reject future costs/fills/path suffixes, target components, and in-sample upstream outputs.
- [ ] Require prefix recomputation, proximity ablation, leave-target-adjacent-out replay, execution sensitivity, later sealed OOS, and identical-entry full overlay—or record explicit `NOT_RUN` under the two-head scope.
- [ ] Separate conditional-policy uplift from complete-population gross/net EV.

## Final evidence

- [ ] Produce the eight named parquet/report deliverables, correctness report, and run manifest.
- [ ] Rank all supported terminal failure classes by economic contribution in bps.
- [ ] All new targeted tests pass, relevant compatibility tests pass, and deterministic reruns are byte-identical.
- [ ] An independent requirement-by-requirement completion audit passes before reporting.
