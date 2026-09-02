# P8U Meta path-label lineage recovery audit

## Status

**Do not promote a recovered path-label source into the current Meta pipeline.**

The original strict-R3 supportive-path label contract remains authoritative.
Three required parts are macOS offloaded/dataless and cannot currently be read:

- `2025-05`;
- `2026-01`;
- `2026-02`.

The historical parent candidate ledger cited by the original manifest was also
archived.  A source-aligned recovery producer was added at
`scripts/recover_strict_r3_p8u_supportive_path_labels_v1.py`, but its output
is a **new successor contract**, not a bit-identical replacement.

## Recovery inputs and safeguards

The recovery producer uses only:

1. immutable P8U target-free candidate identities;
2. the canonical policy ledger as policy provenance;
3. frozen Kraken 15-minute bars for future-path supervision.

It derives the signal timestamp only from the frozen candidate ID and verifies
the one-hour decision delay.  It emits target-only fields and never opens an
inference, MC1, admission, portfolio, live, or exchange-writing source.

The only unavailable bar needed by the affected candidate population was
`GAS/USD:USD`.  A public Kraken recovery cache was downloaded into the new,
separate artifact:

`data_perp/artifacts/strict_r3_p8u_recovered_gas15m_history_20260830_v1`.

It has 29,569 contiguous exchange-observed 15-minute rows over
2025-04-29 through 2026-03-03.  The original bar cache is unchanged.

## April 2025 control result

The recovery was tested against an intact original April part on the exact
target-free candidate intersection.  The current P8U bridge contains 53,973
April candidates; 53,898 overlap the historical part.  The original part has
107,309 April decisions because it was generated from a broader old candidate
panel.

On the overlapping IDs, the fields consumed by the Meta target reader do not
meet exact-contract parity:

| Check | Result |
|---|---:|
| `supportive_path_valid` disagreements | 684 |
| exact ATR-fraction equality | 75.70% |
| exact peak-MFE-ATR equality | 82.31% |
| ATR absolute delta, p90 / p99 / max | 0.000006 / 0.002301 / 0.034317 |
| peak-MFE-ATR absolute delta, p90 / p99 / max | 0.000075 / 0.334929 / 8.174706 |

The mismatch is broader than the recovered GAS series, indicating historical
15-minute source revisions after the original August-23 label materialisation.
This makes a mixed old/new label source scientifically invalid for strict MC1
confirmation.

## Consequence for the Meta HPO pipeline

- State, Under, and Over completed confirmations retain their original
  immutable label lineage.
- The remaining Magnitude full confirmation is halted rather than being run
  on a mixed lineage.
- The recovered labels are usable only as an explicitly versioned **new**
  path-label substrate, after rerunning target/query selection, feature
  selection, weights, model-family choice, HPO, and MC1 confirmation under
  that one common contract.

## Valid next actions

1. Restore the original offloaded files and original parent ledger from a
   durable external backup, then resume the frozen Magnitude confirmation with
   exact historical lineage; or
2. Declare a new complete source-aligned path-label contract, rematerialise it
   for the entire P8U research population, and restart every path-dependent
   stage.  It must not be pooled with prior results.

No silent fallback, candidate dropping, label imputation, or partial
replacement is allowed.
