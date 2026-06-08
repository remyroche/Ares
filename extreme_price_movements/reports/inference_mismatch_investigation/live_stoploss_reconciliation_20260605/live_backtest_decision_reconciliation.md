# Live Backtest Decision Reconciliation

This audit uses current live ledger candidates, deployed rank-reference artifacts, and the training feature store.

## Verdicts

- Same-signal-bar would-open/actually-open replay: FAIL
- Feature parity: PASS
- Prediction parity: PASS
- Rank-normalization parity: PASS
- Policy symbol availability at inference: PASS
- Available non-traded symbol OOS performance: PARTIAL

## Interpretation

- A strict replay row is one with persisted signal timestamp, rank threshold, rank score, and stale-signal gate state.
- Legacy rows are not used to prove bidirectional equivalence because they predate the full diagnostic fields.
- Legacy traded rows are separately classified against the current rank and stale-entry gates to show whether they would still be admissible today.
- Rank parity is marked UNPROVEN, not FAIL, when the current rank-reference file was modified after the live decision.
- Feature/prediction verdicts are based on fresh/current strict rows when available; legacy rows remain in the JSON for diagnosis.
- Feature mismatches in live-sensitive orderbook-derived model inputs are failures for strict parity unless the training-selected cache value is used.
- Symbol availability PASS means policy-OOS symbols have live OHLCV and training feature files; it does not mean they passed masks on the sampled live bars.

## Summary
### decision_replay
```json
{
  "ledger_rows": 447,
  "strict_replay_rows": 344,
  "legacy_or_incomplete_rows": 103,
  "live_opened_strict": 5,
  "replay_opened_strict": 7,
  "strict_decision_matches": 338,
  "strict_decision_mismatches": 6,
  "strict_gap_classes": {
    "match": 338,
    "replay_opened_live_rejected": 4,
    "live_opened_replay_rejected": 2
  },
  "legacy_decision_counts": {
    "rank_rejected": 96,
    "traded": 7
  },
  "legacy_traded_rows": 7,
  "legacy_traded_rank_gate_pass": 7,
  "legacy_traded_current_stale_gate_fail": 7,
  "legacy_traded_max_signal_to_entry_seconds": 5102.368352
}
```

### feature_parity
```json
{
  "rows": 41620,
  "decisions": 344,
  "mismatches": 2,
  "live_sensitive_orderbook_mismatches": 2,
  "max_abs_diff": 1.0673351883888245,
  "features_with_mismatch": [
    "xasset_ob_liquidity_peer_resid"
  ],
  "fresh_current": {
    "rows": 41200,
    "decisions": 341,
    "mismatches": 0,
    "live_sensitive_orderbook_mismatches": 0,
    "max_abs_diff": 0.0,
    "features_with_mismatch": []
  }
}
```

### prediction_and_rank_parity
```json
{
  "rows": 344,
  "base_mismatches": 1,
  "meta_mismatches": 0,
  "policy_rank_rows_with_current_reference_after_decision": 3,
  "auction_rank_rows_with_current_reference_after_decision": 3,
  "policy_rank_mismatches_on_fresh_reference": 0,
  "auction_rank_mismatches_on_fresh_reference": 0,
  "max_base_abs_diff": 0.0002737375182086943,
  "max_meta_abs_diff": 0.0,
  "max_policy_rank_live_score_abs_diff": 0.002883044195947204,
  "max_auction_rank_live_score_abs_diff": 0.0003499785666656319,
  "fresh_current_rows": 341,
  "fresh_current_base_mismatches": 0,
  "fresh_current_meta_mismatches": 0,
  "fresh_current_policy_rank_mismatches": 0,
  "fresh_current_auction_rank_mismatches": 0,
  "fresh_current": {
    "rows": 341,
    "base_mismatches": 0,
    "meta_mismatches": 0,
    "policy_rank_mismatches": 0,
    "auction_rank_mismatches": 0,
    "max_base_abs_diff": 0.0,
    "max_meta_abs_diff": 0.0,
    "max_policy_rank_live_score_abs_diff": 0.0,
    "max_auction_rank_live_score_abs_diff": 0.0
  }
}
```

### symbol_universe
```json
{
  "policy_symbols": 100,
  "live_candidate_symbols": 79,
  "live_traded_symbols": 11,
  "policy_available_symbols": 100,
  "available_not_seen_in_live_candidates": 39,
  "missing_live_ohlcv": 0,
  "missing_training_features": 0
}
```

### non_traded_symbol_oos
```json
{
  "rows": 32,
  "live_traded_symbols": 11,
  "comparable_confidence_bands": 16,
  "non_traded_bands_within_5pct": 13,
  "min_non_traded_minus_traded_gross_hit_delta": -0.22015069020513667,
  "non_traded_within_5pct_at_similar_confidence": false
}
```

### cross_strategy_symbol_oos_eligibility
```json
{
  "symbols_evaluated": 100,
  "deployable_symbols": 68,
  "rejected_symbols": 32,
  "min_gross_hit_delta": -0.05,
  "min_mean_net_return": 0.0
}
```
