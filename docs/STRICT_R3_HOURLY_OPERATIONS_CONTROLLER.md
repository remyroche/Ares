# Strict-R3 hourly operations controller

## Purpose

`scripts/run_strict_r3_live_operations_controller.py` makes the operational
loop actively recover only known, reversible service failures. It consumes the
immutable live-candle report and an independently hash-bound controller
configuration. It is invoked by
`scripts/run_strict_r3_live_operations_supervisor_loop.sh` both when the
hourly watch begins and when a terminal candle report is written.

It is not a general code-patching system. That boundary is intentional: a
controller that could alter features, models, calibration, state lineage, or
execution rules in response to an incident could create an unreviewed live
strategy or duplicate a trade.

## Automatically recoverable conditions

Provided the controller configuration is explicitly enabled and each launcher
and its sealed inference/execution/activation contracts match their pinned
SHA-256 values, the controller may:

1. Restart a missing minute-position monitor immediately.
2. Restart a missing hourly producer only through its sealed launcher, which
   must wait for the next fresh decision boundary. It never resubmits the
   already observed decision.

Each restart is subject to a 24-hour budget and a post-start PID/command
identity check. A failed identity check is a fail-closed incident, not a
successful restart.

## Conditions that remain fail-closed

- Any receipt that crossed the exchange-execution-intent boundary but lacks a
  terminal success receipt.
- Source coverage, direct-15m price consistency, feature parity, model hash,
  calibration, Geometry/K9, state-lineage, or runtime-seal failure.
- A conflicting/unknown process identity, duplicate service, or an exhausted
  restart budget.
- A stale decision. The controller never runs a historical signal with order
  authority.
- Any unclassified failure.

For these cases it writes an immutable controller receipt with the root-cause
class and leaves entry trading blocked until a focused diagnosis, test, and
resealed successor are available.

## Promotion use

The checked-in controller config is deliberately disabled while the BCF
same-lineage challenger is still unpromoted:

`config/strict_r3_live_operations_controller_v1.json`

When a live successor is approved, produce a fresh versioned controller config
that binds the exact producer and monitor launchers, their SHA-256 values, PID
paths, expected command fragments, restart budget, and activation status.
Validate it with the controller tests and a no-order service-restart drill
before enabling `auto_restart_authorized`.

## Evidence

Each classification writes:

`data_perp/artifacts/strict_r3_live_operations_controller_<runtime>_<decision>_<source-hash>/run_manifest.json`

The receipt records input hashes, root-cause category, allowed action, service
identity checks, restart budget, post-restart verification, and guarantees
that no stale execution, model/policy mutation, or state-lineage mutation was
performed.
