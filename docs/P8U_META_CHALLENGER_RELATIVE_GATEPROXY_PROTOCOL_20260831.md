# P8U Meta — challenger-relative HPO shortlist protocol

## Decision

Yes: Meta HPO needs a better **research decision objective**, but not a new
universal downstream-EV ranker.  The frozen `GateProxy_P0_Ridge` may remain a
cheap transition shortlist for already-defined work.  For the next genuinely
independent HPO bank, the proxy question becomes:

> Is this challenger likely to beat the fixed incumbent after the same strict
> MC1, dual-admission, and constrained chronological portfolio replay?

This fixes the precise failure found in the GateProxy audit: P0 contained the
best *new* challenger in its top three, but ranked the realised incumbent only
fourth.  An absolute trial-quality target cannot safely arbitrate between an
established incumbent and nearby challengers.

## What changes

The future proxy will learn `P(BeatIncumbent)` from cheap strict-OOF
diagnostics.  It will also carry a conditional margin-of-victory estimate for
shortlist ordering.  Its inputs are both challenger diagnostics and the
same diagnostics expressed relative to the fixed incumbent.

The new proxy will be trained and validated by **independent HPO bank**, never
random trials.  The primary tests are Recall@3 and Regret@3, not Spearman.

## What does not change

Nothing in the current score, MC1, admission, portfolio, execution, or live
contracts changes.  The incumbent always receives full downstream
confirmation; it is not a proxy-ranked trial.  No new challenger can advance
from this proxy alone.

## Confirmation set

```text
fixed incumbent
  + challenger-relative proxy Top-3
  + one uncertainty/diversity challenger when compute permits
  → fresh strict MC1
  → one common constrained chronological portfolio
  → actual promotion decision
```

The linked predeclared contract is
[`strict_r3_p8u_meta_challenger_relative_gateproxy_protocol_20260831_v1.json`](../config/strict_r3_p8u_meta_challenger_relative_gateproxy_protocol_20260831_v1.json).

## Status

This is a **future-bank protocol**, intentionally not a refit of P0.  The
completed Under/State bank and its February–July 2026 downstream period remain
an audit set; using them to fit this successor and claim the same results would
be circular.  The next independent, structurally diverse challenger bank is
the first valid fit-and-falsification opportunity.

## Evidence behind the decision

- Under: P0's proxy-to-constrained-EV Spearman was `-0.50`, yet its top-three
  contained the best confirmed challenger (`Regret@3 = 0`).
- State: the equivalent Spearman was `+0.50`; its top-three also had
  `Regret@3 = 0`.
- The retained Under incumbent realised `+135.73 bps/trade` but ranked fourth
  among the bank-plus-reference set by P0.  This rejects P0 as an
  incumbent-versus-challenger selector, not as a challenger shortlist funnel.

See also [the frozen audit](P8U_META_GATEPROXY_AUDIT_20260831.md) and
[the strict top-three rerun](P8U_META_GATEPROXY_TOP3_RERUN_20260831.md).
