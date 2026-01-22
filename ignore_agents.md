# agents.md — Engineering Rules for Crypto ML (Troubleshooting + Root-Cause Fixing First)

## 0) Scope and enforcement
These rules are mandatory for any agent modifying this repository. If a rule conflicts with “make it run,” the rule wins. “Green” is defined by: tests + the designated pipeline command + non-regressing metrics in `outcomes/`.
Should the logs or report in outcomes/ be unavailable, do with the available log excerpts and a code review.
Prefer explicit log file paths (e.g., ETH run logs) when available; if not, proceed with the best available log excerpts.

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

## Agent Debug Protocol (mandatory, flowchart-style)

This protocol governs **all debugging and troubleshooting behavior**.  
Deviation is not allowed. If a step fails, the agent must return to the last valid checkpoint.

---

### STEP 0 — Declare Mode (hard gate)
Before any investigation, explicitly declare:

- **Light Mode** — logic, data, and code correctness only
- **Blank Mode** — same as Light Mode, minimal assumptions
- **Full Mode** — includes financial metrics as correctness signals (AFML / causal)

If no mode is specified, default to **Light Mode**.

---

### STEP 1 — Identify the Failing Run (input gate)
- Locate the failing run ID / timestamp in `outcomes/`.
- Confirm:
  - git commit hash
  - config hash/snippet
  - artifact IDs and versions
- If any are missing → **STOP** and report “non-reproducible run.”

---


### STEP 2 — Locate the First Assumption Violation (root-cause gate)
- Scan logs **chronologically**, not from the stack trace.
- Identify the **first log line** where an assumption breaks:
  - first NaN introduction
  - first empty dataframe
  - first shape/schema mismatch
  - first invariant failure

If only a final stack trace is available → **STOP** and add missing logging.

---

### STEP 3 — Capture the Traceability Chain (mandatory artifact)
Produce a Traceability Chain containing:

**3.1 Input State**
- step ID

**3.2 First Point of Failure**
- exact log line (timestamp + excerpt)
- expected vs observed state

**3.3 Data Context**
- shape (rows/cols)
- missingness summary
- head/tail
- relevant statistics
- time range + timezone

If any element is missing → **STOP** (investigation incomplete).

---

### STEP 4 — Form a Hypothesis (single sentence, mandatory)
State exactly **one** hypothesis explaining the failure.

Example:
> “The feature step produced NaNs because the raw candle data is missing for this symbol on exchange holidays.”

If multiple hypotheses exist → choose the most upstream one.

---

### STEP 5 — Prove the Hypothesis (proof gate)
- Create a **minimal reproduction** that fails for the same reason:
  - must use real pipeline data
  - must use real configs
  - no mocks, no synthetic data

If the hypothesis cannot be proven → **STOP** and revise hypothesis.

---

### STEP 6 — Workaround Check (anti-patching gate)
Before implementing a fix, answer:

- Does the fix introduce:
  - conditionals (`if`, `try/except`)
  - data coercions (`fillna`, `clip`, drop rows)
  - threshold changes
  - silent fallbacks?

If **yes**, explicitly justify:
- why this removes the generating cause
- why it does not hide bad data
- why it preserves intended semantics

Unjustified workarounds are **forbidden**.

---

### STEP 7 — Implement Structural Fix (minimal-change rule)
- Fix the **root cause**, not the symptom.
- Prefer:
  - correcting upstream data assumptions
  - enforcing stronger invariants
  - failing fast with actionable errors
- Do not refactor unrelated code.

---

### STEP 8 — Regression Test Symmetry (verification gate)
A fix is invalid unless:

- a new test fails on old code
- the same test passes on new code
- existing tests still pass

If this gate fails → return to STEP 8.

---

### STEP 9 — Mode-Specific Validation

**Light / Blank Mode**
- Confirm:
  - no exceptions
  - invariants hold
  - pipeline completes

**Full Mode**
- Evaluate financial metrics:
  - MI / RMI stability
  - PR-AUC (OOS)
  - cost-adjusted performance
  - regime robustness

If metrics degrade unexpectedly:
- treat as a **hidden bug**
- return to STEP 3

---

### STEP 10 — Metrics & Report Update (documentation gate)
- Write updated metrics payload to `outcomes/`.
- Append a report section including:
  - root cause
  - fix summary
  - test added
  - metric deltas (if Full Mode)
  - etc

Missing report → **STOP** (task incomplete).

---

### STEP 11 — Final Checklist (exit gate)
Output the **Agent Task Completion Checklist**, including:
- execution mode used
- confirmation that all protocol steps were satisfied

Unchecked items mean the task is **not complete**.

---

### FAILURE SEMANTICS (global)
- Poor or unstable financial metrics in Full Mode  
  ⇒ troubleshooting failure, not a “model result.”
- Silent workarounds  
  ⇒ invalid fix.
- Missing traceability  
  ⇒ investigation rejected.

