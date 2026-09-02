# Strict-R3 live operations protocol

For every UTC candle, the live trader must establish target-free source
coverage, causal feature generation, frozen-stack scoring, EV admission,
portfolio/execution results, and one-minute exit monitoring.  A terminal
per-candle report is written by the read-only `xx:03` observer only after the
hourly producer completes the pipeline or rejects every opportunity. An
incident attempt receives its own immutable report but is not terminal: the
observer remains active until a tested successor attempt is complete.

The current v53 recovery contract permits a decision to remain executable for
its current UTC hour (0–3,600 seconds), not indefinitely. It preserves the
same feature, model, EV-admission, price-gap, book-spread/impact,
execution-adjusted-EV, portfolio, protective-stop and state-lineage gates.
The next hour invalidates an unexecuted decision.

The exit-policy clock is the verified Kraken fill timestamp, never the model
decision timestamp. The first evaluated one-minute bar must therefore be
wholly post-fill. The runtime closure includes the full policy-label
materialisation dependency chain. The v53 execution boundary also validates
the complete hash-sealed inference bundle before a live monitor or executor
receives authority, rather than checking only its direct modules. A missing
or changed transitive module therefore fails before source refresh, scoring,
or exchange I/O.

The one-minute execution store remains append-only, but the v53 monitor holds
a signature-validated in-process row index. Its first access fully validates
the immutable parts; later polls only revalidate the filesystem signature and
read newly appended timestamps. This prevents accumulated micro-parts from
turning a minute-monitor pass into a multi-minute one while retaining conflict
detection after another process writes or after a restart.

The hourly producer is idempotent by decision and sealed runtime lineage. If
the service restarts during an hour which already has a successful immutable
receipt, it records `already_completed` and remains alive for the next UTC
candle; it must never exit or resubmit the completed hour. A separate
read-only supervisor consumes every immutable report attempt from `xx:03`,
writing an operational receipt with either `observe_next_candle` or
`investigate_root_cause_before_any_same-hour_successor`. It has no source,
score, exchange, live-state, restart, or patch authority.

The process PID files—not a detached `screen` session—are the authoritative
service control plane. A restart must explicitly terminate each recorded
worker PID, verify that no superseded producer or monitor child remains, and
only then launch the successor. The producer treats temporary pre-execution
receipt contention as a bounded same-hour retry; a failure after the exchange
stage begins is never retried automatically and requires an incident review.

Any irregularity is an incident. The required iterative response is:

```text
preserve receipts and state
→ trace the producing data/code lineage to the root cause
→ patch the narrowest causal defect
→ add a deterministic regression test
→ run unit tests and no-submit same-candle replay
→ reseal versioned runtime contracts
→ restart exactly one entry producer and one exit monitor
→ reconcile Kraken and validate the repaired candle
→ restore live authority
```

Do not label a retry, a local fill, or a threshold change as a repair. A
missing source/feature fails closed for the affected symbol and still requires
investigation. A global source, lineage, state-chain, or exit-monitor failure
blocks new entries until repaired. The reporter itself never patches blindly,
but the operating agent is authorized to patch, test, reseal, restart, and
return the validated stack to live execution.

The controlled restart order is: keep the exchange protective stop active,
stop the existing entry writer and minute monitor by their verified PID (and
wait for both child and wrapper to exit), migrate the state
with a hash-bound receipt that preserves positions and processed decisions,
start exactly one new minute monitor and verify its first receipt, then start
exactly one entry producer, the xx:03 observer, and the read-only
report-consuming supervisor. No patch may be promoted without the focused
test and a no-submit receipt; once validated, the successor is allowed to
change the live services.
