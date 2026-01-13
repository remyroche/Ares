0) Scope and priority

This file defines mandatory engineering and research practices for any agent (and humans) modifying this repository. When rules conflict, precedence is:

Safety & correctness (no leakage, no lookahead, reproducibility)

Step completion criteria (metrics + report + logs)

Performance requirements (Numba/JIT/caching)

Style and maintainability

1) Good practices (code quality)
1.1 Structure and readability

Prefer small, composable functions with single responsibility.

No “magic” constants: use named constants or config.

All public functions and classes must include docstrings describing:

inputs/outputs, shapes and dtypes (where relevant)

time index semantics (timezone, bar close/open convention)

leakage constraints (what the function is allowed to “see”)

1.2 Types, validation, and errors

Add type hints for all non-trivial functions.

Validate assumptions early:

index monotonicity, missing data policy, dtype constraints, expected columns

Fail loudly with actionable exceptions (include context: symbol, timeframe, step id).

1.3 Testing discipline

Every bug fix must come with a regression test.

Critical invariants must be covered by tests:

no lookahead leakage checks

deterministic results given fixed seeds and fixed data manifests

stable feature generation schema (column naming, ordering)

1.4 Reproducibility and traceability

Every run must record:

git commit hash

config hash / name

dataset manifest/version id

random seeds

environment summary (python version, key libs)

Results must be written into outcomes/ with run identifiers and timestamps.

2) Artifact management & raw data fetching
2.1 Mandatory base abstractions

All pipeline steps must inherit from BaseStep (or BaseClass if that is your canonical abstraction).

No direct ad-hoc fetching in notebooks or random scripts. Raw data fetching must occur only through the designated Base abstraction and its connectors.

2.2 Artifact lifecycle

Every step must:

declare inputs (artifact ids + versions/manifests)

declare outputs (artifact ids + versions)

store metadata: schema, date range, exchange, symbol universe, sampling

Each artifact must be:

content-addressed where feasible (hash-based)

cached on disk (and optionally remote) with explicit invalidation rules

2.3 Data provenance rules

Raw data must be immutable once written (new versions are new manifests).

All transformations must be attributable (step name, parameters, timestamp).

3) Performance: Numba/Numpy/JIT + extensive caching
3.1 Default performance posture

Vectorize with NumPy first.

Use Numba (or equivalent JIT) for hot loops and path-dependent computations.

Avoid pandas loops and per-row apply in performance-critical code.

3.2 Caching requirements

Any step taking more than a “short run” must implement caching keyed by:

input artifact ids + versions

config parameters

code version (commit hash) if needed for safety

Cache must be:

transparent (clear logs on cache hit/miss)

invalidation-safe (never reuse if inputs/config changed)

Prefer multi-level caching:

in-memory for intra-run

disk for inter-run

optional remote cache for CI/team use

3.3 Performance acceptance checks

Each major step should emit:

runtime

peak memory (where feasible)

cache hit ratio (if applicable)

If performance regresses materially, it is treated as a bug.

4) ML-specific code: de Prado (AFML) and causal frameworks
4.1 AFML alignment (mandatory patterns)

When implementing ML components, follow AFML principles, including:

Proper labeling and event definition (avoid naive next-bar labels unless justified).

Appropriate cross-validation for financial time series:

purging and embargo to reduce leakage

walk-forward evaluation for model selection

Thoughtful sample weighting and handling of overlapping outcomes.

Emphasis on meta-labeling / decision-layer separation where appropriate.

4.2 Causal and robustness posture

Treat signals as hypotheses; require tests for:

stability across regimes

sensitivity to costs/slippage

adversarial/noise perturbations

Prefer causal-minded feature design:

avoid post-treatment variables

document plausible mechanism

use controlled comparisons / ablations

4.3 Baselines are mandatory

Every new model must be compared against:

buy-and-hold (or equivalent passive baseline for the instrument)

a simple momentum/MA baseline

a turnover-matched random baseline (when applicable)

5) Mandatory logging: tprint on call and completion
5.1 Logging rule

Every significant function (and all step entrypoints) must:

tprint at start: function name, key parameters, input shapes/date ranges

tprint at end: success, output shapes, key metrics, runtime, cache status

Errors must be logged with enough context to reproduce (step id, artifact ids, config).

5.2 Logging content standards

Include, where relevant:

symbol universe, timeframe, exchange, date range

number of rows, missingness rate, feature count

labels distribution and sample weights summary

seed values and config identifiers

6) Step completion = metrics and financial quality assessment
6.1 Definition of “step complete”

A step is not complete until it produces:

Output artifacts (with manifests)

A metrics payload (stored and written to outcomes/)

A short step report snippet (also in outcomes/)

6.2 Required metrics by step type

Feature generation / selection

Mutual Information (MI) and/or conditional MI where applicable

Robust/Rank MI (RMI) or equivalent stability-aware variants

Feature stability across time splits (drift indicators)

Redundancy checks (correlation clusters, VIF-style diagnostics where applicable)

Regime discovery

Regime separability/stability metrics (e.g., transition stability, persistence)

Out-of-sample regime predictability checks (avoid hindsight regimes)

Performance conditioning by regime (PnL/risk metrics per regime)

Model training / selection

PR-AUC (especially for imbalanced events)

Calibration metrics (reliability curves / Brier score where relevant)

Out-of-sample performance with purged/embargoed CV

Turnover, capacity proxy, cost-adjusted returns

Tail risk: max drawdown, CVaR (or drawdown distribution)

Backtest / policy evaluation

Sharpe/Sortino (with caveats and consistent sampling)

Max drawdown, Calmar

Hit rate, profit factor

Turnover and slippage sensitivity (stress test)

6.3 Step-by-step reporting
Each step must append to a run-level report in outcomes/:
what changed
what improved/regressed (with metrics)
what is the next risk/unknown

7) Debugging: logs + latest report in outcomes/ are the source of truth
7.1 Troubleshooting workflow (mandatory order)
When something fails or metrics regress:
Read latest run report in outcomes/
Inspect logs (most recent first)
Reproduce with the minimal step command/config that generated the artifact
Fix + add regression test
Re-run the step and confirm metrics + report updated

7.2 No “silent fixes”
Do not “fix forward” by changing many components at once.
Fix the minimal root cause, then re-run to confirm.

8) Trading safety guardrails (always on)
No live trading by default.
Paper/backtest only unless explicitly enabled by configuration and environment gating.
Always include transaction costs and slippage assumptions in evaluation; default pessimistic.

9) Definition of done (for any PR/change)
A change is “done” only when:
tests pass
the relevant step(s) run successfully
outcomes/ contains an updated report + metrics payload
logs show tprint start/end for the step(s)
performance constraints (JIT/caching) are respected for hot paths

10) Versioning
Use Python 3.11
Use Poetry for dependency management and update it when adding new dependencies
