# Strict-R3 live operations: detection, investigation, repair, and restart

## Objective

The long-only Kraken trader is not a fire-and-forget scheduler.  For every UTC
hour it must establish that the entire executable chain worked:

```text
completed hourly candle
→ authoritative point-in-time source refresh
→ causal feature-state advance
→ frozen base / K9 / consensus / MC1 scoring
→ causal EV admission and portfolio auction
→ execution preflight and protected entry
→ one-minute position monitoring and 15-minute policy updates
```

An irregularity is an incident, not an ordinary rejected trade.  The mandatory
response is: preserve evidence → find the root cause → patch the root cause →
run focused tests → reseal/restart the one writer → validate the repaired chain
→ return the corrected live stack to entry authority.  Do not treat an extra
retry, a local fill, or a changed threshold as a repair.

## Per-candle operational cadence

### Entry chain

At each fresh UTC hour, the singleton live entry producer must:

1. append raw Kraken 15-minute OHLCV, official book fields, mark/OI, and
   funding; never replace a prior observed bar with a local fill;
2. retry only incomplete symbols at 30, 60, 120, and 180 seconds;
3. classify every one of the 170 frozen symbols as exchange-observed,
   locally-filled-flat, missing 15-minute bar, or missing decision open;
4. build all target-free candidates and causal features from the complete
   decision-time universe;
5. fail closed only for the affected candidate when a frozen feature is not
   available, and record the exact field(s); investigate every such case;
6. run base routing, consensus/geometry, Robust-21 + frozen MC1 mapping,
   admission, and the portfolio auction; and
7. preflight only portfolio-accepted entries against live executable price,
   delay gap, spread, and impact, then submit only if adjusted EV remains at
   least +50 bps and install the protective stop immediately.

At `xx:10` UTC, the independent read-only candle reporter writes one immutable
report.  It must state the count and reason at every funnel stage, all source
coverage states, audit/checkpoint results, order results, and the current
one-minute monitor receipt.

### Exit chain

The separate persistent monitor wakes every minute.  It must reconcile live
positions, update MFE/trailing state only from fully post-fill one-minute bars,
evaluate the ML exit modulator only on completed 15-minute bars, retain the
SimplePolicyOptimiser parent policy as fallback, and submit only reduce-only
orders.  A missed, stale, unprotected, or unexplained exit is an incident.

## Incident protocol

The `action_required` report status or any unexpected funnel/exit condition
starts this exact procedure.

1. **Freeze the evidence.** Keep all immutable receipts, raw source snapshots,
   current feature-state bundle, decision input frame, execution response,
   and Kraken read-only reconciliation.  Never overwrite a failed receipt or
   delete cache rows to make the symptom disappear.
2. **Classify the failure.** The investigation must identify one of:
   source availability/provenance; candidate eligibility; feature-state or
   feature-contract parity; model/map/calibration lineage; portfolio/execution
   preflight; or position-monitor/exit state.
3. **Trace the root cause.** Follow the producing code and data lineage to the
   first invalid transformation.  Examples: upstream exchange omission versus
   an over-aggressive cache filter; absent decision open versus stale refresh;
   a missing feature input versus a generator state bug; rejected order versus
   wrong priority metric; a delayed exit versus pre-fill bar misuse.  Reporting
   only the final rejection reason is insufficient.
4. **Apply the narrowest causal patch.** Preserve target-free candidate
   semantics, frozen model contracts, costs, and outcome availability rules.
   Do not impute from future bars or change an admission threshold merely to
   restore participation.
5. **Test the patch before live relaunch.** Add or extend a deterministic test
   for the root cause; run the affected unit tests; then replay the same
   decision in no-submit mode from its immutable point-in-time inputs.  Verify
   candidate identities, feature values, model outputs, admission, policy,
   and exits agree with the corrected contract.
6. **Reseal and restart safely.** If runtime code or any hash-bound contract
   changes, create a versioned successor bundle/authorization and validate all
   hashes.  Stop the old singleton only after the successor is validated; start
   exactly one hourly writer and one minute monitor.  Reconcile Kraken before
   re-enabling new entries, so no duplicate order or untracked position is
   possible.  A passed successor validation restores normal live entry
   authority; the corrected stack is expected to trade rather than remain in
   permanent shadow or diagnostic mode.
7. **Validate after restart.** Re-run the current candle only while it is
   still inside the authorized entry window; otherwise validate it as a
   no-submit replay and wait for the next candle.  Confirm the report is
   `pass`, required symbols are source-complete, the audit/checkpoint is clean,
   and the minute monitor produces a fresh receipt.  Document cause, patch,
   tests, runtime hashes, and residual limitations in the incident report.

## Fail-closed authority

- Global source, lineage, calibration, or state-chain failures prevent all new
  entries for the affected cycle.
- A missing source/feature for one symbol prevents that symbol only, but still
  requires investigation; it is not silently normalized as ordinary rejection.
- Any entry-execution discrepancy is reconciled against Kraken before a retry;
  never submit a replacement order without determining whether the first order
  filled.
- Any exit-monitor discrepancy stops new entries until live positions are
  reconciled and a working protective exchange order is confirmed.  Existing
  positions retain their parent protective policy; do not leave them unmanaged
  while repairing adaptive logic.
- The read-only reporter does not patch live code or restart services itself.
  It generates the immutable, per-candle evidence that directs the agent's
  investigation.  The agent is authorized to patch, test, reseal, restart, and
  restore live entry authority once the protocol's validation gates pass.

## Required artifacts

For every candle retain:

- producer, source-refresh, scorer, primary-audit, runtime-checkpoint, and
  executor receipts;
- `data_perp/reports/strict_r3_live_candle_<runtime>_<timestamp>.md/.json`;
- position-monitor receipts and any exit/entry email context; and
- for incidents, a root-cause note, test receipt, successor bundle hashes, and
  post-restart validation report.

This protocol governs the persistent live services; it is separate from the
offline full-reconstruction/parity audits so diagnostic work never delays a
fresh decision or duplicates exchange-writing behavior.
