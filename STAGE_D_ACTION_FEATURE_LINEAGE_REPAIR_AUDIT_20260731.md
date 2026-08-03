# Stage-D A0 action-state lineage repair audit — 2026-07-31

## Verdict

**PASS.** Canonical corrected pack:
`data_perp/artifacts/stage_d_action_features_20260731_v5/`.

Independent same-code rerun:
`data_perp/artifacts/stage_d_action_features_20260731_v6/`.

The intermediate v4 pack predates the complete staleness-rule correction and is superseded. v3 is
superseded for the original early-availability metadata defect.

## Corrected lineage

The following A0 fields are now explicitly action-state fields rather than entry-static fields:

- `time_to_clear_minutes`
- `gross_return_at_action_bps`
- `estimated_net_if_exit_now_bps`

For all three, the dictionary and lineage table now declare:

- availability: `action_decision_ts`;
- lookback: `entry_ts..action_decision_ts`;
- path stop: inclusive through `first_clear_bar_index`, with the future suffix never decoded;
- staleness: exact completed path prefix through the action decision;
- source: exact completed 1m OHLC action path plus frozen entry geometry.

Genuinely entry-static A0 fields remain `entry_ts (persisted frozen row)` with frozen-entry lookback,
no path-stop rule, and frozen-row staleness. Tests cover row cost, barrier, spread/half-spreads,
entry-price log and side identity explicitly.

Existing A1/A2/A3/A4/A9 path fields retain their action-decision availability, entry-to-action lookback
and prefix stop semantics; their staleness description was normalized to the equivalent explicit phrase
`exact path prefix through action decision`.

## Determinism and parity

- v5 and v6 are byte-identical for all 11 files, including both manifests.
- Both contain 108,139 rows, 134 admitted features and 152 total feature-panel columns.
- Ordered candidate hash: `fdfe4bec81ae0f34c8e58c982c4847ae408d38e4c198d98c9b9e753658bcd571`.
- Candidate-set hash: `2088db5b78152be60a5b1b6a5f69500d6010e3e1a01b174dff0df5186d7b78b5`.
- The v5 feature panel is byte-identical to v3 (`a953a255f233ae210039fc8fb56a5c2f78a713f3b3b466cca81a4746c20cacd5`)
  and dataframe-equal with identical dtypes. Group definitions, coverage, membership, market ledger,
  rejected controls and dependency ledger are also byte-identical.
- Only metadata-bearing lineage/dictionary/manifests changed, plus the semantically equivalent path
  staleness wording described above.
- Corrected dictionary SHA-256: `efa5c208849f208a4e7a7b4cfbfb8de0315d90458aaa5a25e9100a36a24a9356`.
- Corrected lineage SHA-256: `178e211acce4fe0620732e3797a3ed350e6544a6720950474345f4c01f97c151`.
- v5/v6 run-manifest SHA-256: `e9c32a5f2da845b9b635701ec60360e8d08371de3f1eea2856c962f5f244dd46`.
- Materializer SHA-256: `654ceee01a6b53b1857ea1bc4f034871922f966173c83132978371fe252b07b8`.
- Focused feature suite: 16 passed. The two new tests prevent action-state fields from claiming entry
  availability and prevent truly entry-static A0 fields from drifting to path-dependent semantics.

Downstream Stage-D modeling must bind to feature v5 (or its byte-identical v6 seal), never v3/v4.
