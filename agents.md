# agents.md — Engineering Rules for Crypto ML (Troubleshooting + Root-Cause Fixing First)

## 0) Scope and enforcement
These rules are mandatory for any agent modifying this repository. If a rule conflicts with “make it run,” the rule wins. “Green” is defined by: tests + the designated pipeline command + non-regressing metrics in `outcomes/`.

---

## 1) Troubleshooting is a structured investigation (not a vague task)

### 1.1 Single source of truth
- Always start from:
  1) **latest report** in `outcomes/`
  2) **logs** for the failing run
- Do not infer causes without reading both.

### 1.2 Standard investigation sequence (must follow in order)
1. **Locate the failing run**: identify run id / timestamp in `outcomes/`.
2. **Reproduce**: rerun the minimal failing command/config (no extra steps).
3. **Find the first assumption violation** (not the last stack trace).
4. **Capture state** (inputs + data context) at failure.
5. **Form a hypothesis** (one sentence).
6. **Prove** the hypothesis with a minimal reproduction using real pipeline data.
7. **Fix the root cause** with the smallest change that preserves intended semantics.
8. **Verify**: regression test symmetry + rerun the workflow + check metrics.
9. **Document**: update the `outcomes/` report snippet with root cause + fix + verification.

---

## 2) Rule 7.3 — The Traceability Chain (mandatory in every error report)
Every error report from the agent must include:

### 2.1 Input State
- Exact **artifact IDs and versions** (and/or manifests)
- Exact **step ID** (if applicable)
- A **config snippet** sufficient to reproduce (not the full file)
- Environment fingerprint:
  - git commit hash
  - Python version
  - Poetry lock hash (or lockfile checksum)

### 2.2 First Point of Failure
- The first log line where an assumption was violated, e.g.:
  - first NaN introduced (with column name and row index/time)
  - first shape mismatch (expected vs observed)
  - first schema deviation (missing/extra columns)
- Include the log line reference (timestamp + line excerpt), not only the stack trace.

### 2.3 Data Context (at the moment of failure)
Provide a concise snapshot:
- shape (rows/cols)
- missingness summary (overall + key columns)
- head/tail
- key stats (min/max/mean/quantiles) as relevant
- time range and timezone assumptions

**Non-negotiable:** Error reports without this chain are incomplete and must be treated as “not investigated.”

---

## 3) Root-cause fixing: no silent workarounds

### 3.1 Rule 7.4 — Hypothesis-First Fixing (mandatory)
Before modifying any code to fix a bug, the agent must:

1. **State a hypothesis (one sentence)**  
   Example: “The data fetcher returned an empty dataframe because the exchange had no candles for that symbol on 2024-12-25.”

2. **Prove the hypothesis with a minimal reproduction**  
   - Must reproduce the error *by triggering the identified root cause*.
   - Must use **real pipeline data and configs**.
   - **Do not** introduce mock/synthetic data to “make the test pass.”

3. **Workaround check (explicit justification required)**  
   If the proposed fix includes any of the following, the agent must justify why it is structural (not symptom patching):
   - conditionals: `if`, `try/except`, fallback returns
   - data transforms: `fillna`, `clip`, winsorization, dropping rows
   - threshold changes (including model thresholds, filters, min periods)
   - “ignore errors” flags
   - broad coercions (casting types, forcing timezone) without diagnosis

**Default assumption:** adding `fillna(0)` or early returns is a workaround unless proven otherwise.

### 3.2 “Structural fix” definition
A fix is structural only if it:
- addresses the generating mechanism of the failure (upstream cause), and
- preserves intended semantics, and
- increases observability (stronger invariants/validation), and
- does not hide data quality issues.

Acceptable structural outcomes include:
- correcting data fetch parameters / calendars / symbol mapping
- fixing schema contracts between steps
- adding explicit invariants and failing fast with actionable errors
- correcting time alignment / timezone / indexing logic
- adjusting feature/label logic to remove leakage or undefined behavior

---

## 4) Rule 7.5 — Regression Test Symmetry (fix verification is objective)
A bug is only “fixed” if all are true:

1. **A new test exists that fails on the old code.**
2. **The same test passes on the new code.**
3. **No existing metrics in `outcomes/` regressed unexpectedly.**
   - If a metric changes, the agent must explain whether it is:
     - expected (due to correcting a bug), or
     - a regression (unintended behavior change)
   - The explanation must be added to the run report in `outcomes/`.

---

## 5) Logging requirements that enable root-cause tracing (uses `tprint`)

### 5.1 Mandatory Logging Standards (Entry / Intermediate / Exit)
Every significant function and all step entrypoints must use `tprint`:

**Entry**
- intent (“what this function/step is trying to do”)
- input fingerprint:
  - artifact IDs (or dataset identifiers)
  - config hash/snippet id
  - shapes/dtypes/time range
  - input hash (stable fingerprint of the input state)

**Intermediate**
- after each major transformation, log:
  - shape
  - NaN ratios (overall + critical columns)
  - key invariant checks (schema present, monotonic index)
  - example: “After merge: 10,000 rows, 42 cols, 0.0% NaNs”

**Exit**
- output artifacts produced (IDs/paths)
- runtime + cache hit/miss
- “confidence score” style statement:
  - schema matches expected
  - invariants hold
  - leakage checks passed (when relevant)

### 5.2 Fail-fast invariants (preferred)
- Prefer explicit invariant checks that raise actionable errors over silent coercions.
- When failing, include the traceability chain identifiers (artifact id, step id, config hash).

---

## 6) Where to look first when something breaks (agent behavior)
When a workflow fails, the agent must:
1. Read latest report in `outcomes/` for the failing run.
2. Search logs for the **first point of failure**.
3. Produce a Traceability Chain report.
4. State and prove a hypothesis.
5. Implement a structural fix with regression test symmetry.

---

## 7) Agent Task Completion Checklist (must be output at end of every task)
- [ ] Traceability Chain provided for any investigated failure.
- [ ] Hypothesis stated and proven with a minimal reproduction using real pipeline data.
- [ ] Workaround check completed (if applicable) with justification.
- [ ] No unexpected metric regressions in `outcomes/` (or explained in the report).


## 8) Mode-Aware Troubleshooting and Failure Semantics

The agent must adapt its troubleshooting scope based on the active execution mode.  
Mode awareness is mandatory and must be explicitly stated at the beginning of any investigation or report.

### 8.1 Execution modes (explicit declaration required)
At the start of any troubleshooting or analysis, the agent must declare one of:

- **Light Mode**
- **Blank Mode**
- **Full Mode**

If the mode is not explicitly declared in the task context, default to **Light Mode**.

---

### 8.2 Light Mode / Blank Mode: Logic-first troubleshooting
**Objective:** Ensure correctness of code, data flow, and assumptions.

In Light or Blank Mode, troubleshooting must focus exclusively on:
- bugs, exceptions, and crashes
- logical inconsistencies
- schema, shape, or alignment errors
- leakage, lookahead, or indexing violations
- broken invariants or invalid assumptions

**Rules**
- Financial performance metrics are **out of scope** unless they directly reveal a logic flaw (e.g., NaN PnL due to upstream bug).
- A step is considered “fixed” once:
  - the Traceability Chain is complete,
  - the root cause is addressed structurally,
  - regression test symmetry holds.

**Anti-pattern**
- Do not rationalize poor financial performance in Light/Blank Mode.
- Do not optimize metrics; fix correctness only.

---

### 8.3 Full Mode: Financial quality is part of correctness
**Objective:** Ensure the system is both *correct* and *financially meaningful*.

In Full Mode, **financial metrics are first-class debugging signals**, not optional diagnostics.

A troubleshooting task is considered **failed** if:
- code runs without errors **but**
- financial metrics violate AFML or causal expectations.

Examples of failures in Full Mode:
- MI/RMI collapses after a refactor with no explained causal reason.
- PR-AUC degrades materially out-of-sample while in-sample improves (overfitting signal).
- Strategy Sharpe improves but drawdown or tail risk worsens unexpectedly.
- Regime-conditioned performance breaks causal consistency (e.g., signal works only in hindsight regimes).

---

### 8.4 AFML / causal failure semantics (Full Mode)
In Full Mode, the following are treated as **troubleshooting failures**, not “model outcomes”:

- Metric instability across time splits or regimes
- Performance that disappears after purging/embargo
- Improvements explainable only by leakage, overlap, or threshold tuning
- Excessive sensitivity to costs, slippage, or minor perturbations
- Features with high MI but no plausible causal mechanism

**Interpretation rule**
> Poor or unstable financial metrics imply a *hidden bug, leakage, or invalid assumption* until proven otherwise.

The agent must:
1. Treat the metric degradation as a signal of a root cause.
2. Re-enter the troubleshooting loop (Section 7).
3. Produce a Traceability Chain that includes **metric context**.

---

### 8.5 Mode-dependent Definition of “Done”
A task is only “done” if:

- **Light / Blank Mode**
  - All bugs and logic flaws are resolved
  - Tests pass
  - No new invariant violations appear

- **Full Mode**
  - All of the above **and**
  - Financial metrics are:
    - stable across splits/regimes, or
    - any degradation is explicitly explained and justified
  - Metrics align with AFML and causal expectations

Unexplained poor financial performance in Full Mode is equivalent to:
> “Troubleshooting incomplete.”

---

### 8.6 Mandatory reporting addition
In Full Mode, the final report in `outcomes/` must include:
- a short **financial diagnostics section**
- key metrics before vs after the fix
- a statement answering:
  > “Why do these metrics make sense under AFML / causal reasoning?”

Absence of this section means the task is not complete.

---

### 8.7 Checklist extension (mode-aware)
When outputting the Agent Task Completion Checklist, the agent must also state:

- **Mode used:** Light / Blank / Full
- **If Full Mode:**  
  - [ ] Financial metrics reviewed and deemed causally consistent  
  - [ ] No unexplained metric degradation relative to prior runs
