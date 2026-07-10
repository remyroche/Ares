"""Append-safe prediction ledger for traded and untraded live candidates."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


PREDICTION_LEDGER_DIAGNOSTIC_COLUMNS = [
    "decision_ts",
    "signal_bar_ts",
    "feature_source_max_ts",
    "feature_available_ts",
    "feature_contract_hash",
    "model_artifact_run_id",
    "policy_artifact_run_id",
    "rank_score_source",
    "policy_rank_pct",
    "policy_rank_reference_n",
    "policy_rank_reference_source",
    "policy_rank_reference_hash",
    "policy_rank_reference_mtime",
    "auction_rank_pct",
    "auction_rank_reference_n",
    "auction_rank_reference_source",
    "auction_rank_reference_hash",
    "auction_rank_reference_mtime",
    "auction_rank_score_source",
    "threshold_rank_score",
    "threshold_rank_score_source",
    "threshold_basis_policy_id",
    "threshold_basis_family",
    "threshold_basis_window_days",
    "threshold_basis_rank_score",
    "threshold_basis_rank_score_source",
    "threshold_basis_selected",
    "threshold_basis_reason",
    "threshold_basis_dynamic_ev_target",
    "threshold_basis_dynamic_score_threshold",
    "threshold_basis_recent_reference_rows",
    "threshold_basis_reference_rows",
    "threshold_basis_baseline_activity_count",
    "threshold_basis_global_dynamic_ev_target",
    "threshold_basis_global_dynamic_score_threshold",
    "normalized_rank_score",
    "model_feature_audit_schema",
    "model_feature_snapshot_hash",
    "base_model_key",
    "meta_model_feature_key",
    "base_model_feature_count",
    "meta_model_feature_count",
    "base_model_features_json",
    "meta_model_features_json",
    "base_model_feature_values_json",
    "meta_model_feature_values_json",
    "model_feature_value_sources_json",
    "model_feature_missing_json",
    "source_strategy_id",
    "canonical_strategy_id",
    "decision_strategy_id",
    "strategy_side",
    "auction_policy_version",
    "auction_candidate_count",
    "auction_rank_number",
    "auction_rank_index",
    "auction_entry_cap",
    "auction_max_new_entries_per_bar",
    "auction_ordering_key_json",
    "auction_sorted_at",
    "auction_selected_before_capacity",
    "was_traded",
    "portfolio_decision",
    "portfolio_reject_reason",
    "liquidity_reject_reason",
    "prescore_market_mask_enabled",
    "prescore_market_mask_allowed",
    "prescore_market_mask_reason",
    "prescore_signal_price",
    "prescore_raw_signal_close",
    "prescore_raw_signal_close_ts",
    "prescore_raw_signal_volume",
    "prescore_raw_signal_volume_ts",
    "prescore_raw_signal_close_reference_gap_bps",
    "prescore_raw_signal_close_reference_source",
    "prescore_signal_bar_close_ts",
    "prescore_signal_close_to_decision_seconds",
    "prescore_max_signal_close_to_entry_seconds",
    "prescore_stale_signal_age_gate_exceeded",
    "prescore_oi_key",
    "prescore_oi_value",
    "prescore_oi_ts",
    "prescore_oi_age_hours",
    "prescore_ticker_bid",
    "prescore_ticker_ask",
    "prescore_ticker_mid",
    "prescore_ticker_last",
    "prescore_ticker_spread_bps",
    "prescore_max_spread_bps",
    "prescore_ticker_spread_weight",
    "prescore_ticker_age_seconds",
    "prescore_ticker_fetch_latency_seconds",
    "prescore_ticker_reject_reason",
    "prescore_orderbook_side",
    "prescore_orderbook_capacity_quote_within_slippage",
    "prescore_orderbook_intended_quote_size",
    "prescore_orderbook_depth_weight",
    "prescore_liquidity_capacity_weight",
    "prescore_orderbook_slippage_bps",
    "prescore_orderbook_reject_reason",
    "signal_price",
    "decision_mid",
    "theoretical_entry_price",
    "policy_entry_price",
    "expected_entry_price",
    "expected_fill_price",
    "realized_entry_price",
    "realized_exit_price",
    "ticker_bid",
    "ticker_ask",
    "ticker_mid",
    "ticker_last",
    "ticker_request_started_at",
    "ticker_received_at",
    "ticker_fetch_latency_seconds",
    "exchange_ticker_timestamp",
    "exchange_ticker_age_seconds",
    "orderbook_side",
    "best_touch",
    "max_walk_price",
    "intended_quote_size",
    "orderbook_capacity_quote_within_slippage",
    "spread_weight",
    "depth_weight",
    "liquidity_capacity_weight",
    "half_spread_bps",
    "effective_orderbook_slippage_cap_bps",
    "max_entry_friction_bps",
    "max_orderbook_slippage_bps",
    "max_chase_bps",
    "entry_limit_price",
    "limit_price",
    "entry_friction_formula",
    "entry_friction_gate",
    "signal_gap_bps",
    "expected_total_entry_friction_bps",
    "expected_friction_drag_bps",
    "expected_fill_slippage_bps",
    "orderbook_slippage_bps",
    "slippage_bps",
    "entry_gap_bps",
    "entry_slippage_proxy_bps",
    "adverse_signal_gap_bps",
    "ev_haircut_bps",
    "ev_haircut_raw_live_entry_friction_bps",
    "estimated_ev_historical_net_return",
    "estimated_ev_historical_cost_bps",
    "ev_haircut_observed_spread_bps",
    "ev_haircut_observed_half_spread_bps",
    "ev_haircut_spread_baseline_bps",
    "ev_haircut_spread_baseline_source",
    "ev_haircut_half_spread_baseline_bps",
    "ev_haircut_spread_excess_bps",
    "ev_haircut_orderbook_slippage_bps",
    "ev_haircut_adverse_signal_gap_bps",
    "ev_haircut_observed_delay_slippage_bps",
    "ev_haircut_delay_slippage_baseline_bps",
    "ev_haircut_delay_slippage_excess_bps",
    "ev_haircut_expected_stop_exit_friction_bps",
    "ev_haircut_stop_exit_baseline_bps",
    "ev_haircut_stop_exit_excess_bps",
    "ev_haircut_stop_exit_source",
    "ev_haircut_contract",
    "ev_inference_cost_rebase_enabled",
    "ev_inference_cost_rebase_applied",
    "ev_inference_fixed_round_trip_cost_bps",
    "ev_inference_spread_multiplier",
    "ev_inference_spread_observed_or_baseline_bps",
    "ev_inference_spread_model_bps",
    "ev_inference_total_cost_bps",
    "ev_inference_cost_model_contract",
    "ev_adjusted_entry_friction_bps",
    "ev_adjusted_gross_return_before_cost",
    "ev_adjusted_historical_net_return_before_rebase",
    "ev_adjusted_net_return_before_friction",
    "ev_adjusted_net_return_after_friction",
    "ev_adjusted_calibrated_score",
    "ev_adjusted_rank_score",
    "ev_adjusted_source",
    "entry_delay_effect_bps",
    "entry_delay_adverse_bps",
    "entry_delay_abs_bps",
    "decision_to_entry_seconds",
    "signal_close_to_entry_seconds",
    "signal_to_entry_seconds",
    "gross_to_net_friction_drag_bps",
    "entry_notional_quote",
    "base_amount",
    "requested_entry_leverage",
    "configured_entry_leverage",
    "actual_entry_leverage",
    "exchange_entry_leverage",
    "max_entry_leverage",
    "perp_default_leverage",
    "perp_rank_leverage",
    "perp_legacy_risk_cap_leverage",
    "perp_liquidation_risk_cap_leverage",
    "perp_risk_cap_leverage",
    "perp_effective_leverage",
    "perp_stop_loss_pct",
    "perp_liquidation_guard_enabled",
    "perp_liquidation_guard_reason",
    "perp_liquidation_requested_leverage",
    "perp_liquidation_guarded_leverage",
    "perp_liquidation_safe_max_leverage",
    "perp_liquidation_leverage_capped",
    "perp_liquidation_guard_reject",
    "perp_liquidation_stop_distance_pct",
    "perp_liquidation_stop_distance_bps",
    "perp_liquidation_required_distance_pct",
    "perp_liquidation_distance_at_requested_pct",
    "perp_liquidation_distance_at_guarded_pct",
    "perp_liquidation_maintenance_margin_pct",
    "perp_liquidation_fee_buffer_pct",
    "perp_liquidation_safety_buffer_pct",
    "entry_fee_quote",
    "entry_fee_cost",
    "entry_fee_currency",
    "entry_fee_source",
    "entry_fee_bps",
    "spread_bps",
    "ticker_spread_bps",
    "realized_fee_bps",
    "realized_funding_bps",
    "realized_borrow_bps",
    "dynamic_hr_surprise_threshold",
    "dynamic_hr_surprise_applied",
    "dynamic_hr_surprise_reason",
    "dynamic_hr_surprise_head",
    "dynamic_hr_surprise_z_eff",
    "dynamic_hr_surprise_guarded_y",
    "dynamic_hr_surprise_w_lower",
    "dynamic_hr_surprise_w_raise",
    "dynamic_hr_surprise_state_age_days",
    "policy_archetype",
    "policy_archetype_source",
    "archetype_hit_surprise_threshold",
    "archetype_hit_surprise_mode",
    "archetype_hit_surprise_applied",
    "archetype_hit_surprise_reason",
    "archetype_hit_surprise_matched_key",
    "archetype_hit_surprise_threshold_delta",
    "archetype_hit_surprise_quality_adjustment",
    "archetype_hit_surprise_priority_multiplier",
    "archetype_hit_surprise_priority_adjustment",
    "archetype_hit_surprise_rank_adjustment",
    "feature_drift_psi_core",
    "feature_drift_ks_core",
    "feature_drift_cov_shift",
    "regime_centroid_similarity_train",
    "rare_leaf_fraction",
    "leaf_count_p10",
    "leaf_count_min",
    "leaf_weight_p10",
    "contrib_top1_abs_share",
    "contrib_top3_abs_share",
    "contrib_entropy",
    "contrib_balance",
    "num_material_contrib_features",
    "prob_uncertainty",
    "portfolio_state_after_snapshot_json",
    "portfolio_state_after_snapshot_hash",
    "open_positions_after_json",
    "active_positions_after_json",
    "open_positions_after",
    "open_positions_after_count",
    "wallet_after",
    "open_notional_after",
    "available_wallet_after",
]

_LGBM_DIAGNOSTIC_SUFFIXES = [
    "feature_drift_psi_core",
    "feature_drift_ks_core",
    "feature_drift_cov_shift",
    "regime_centroid_similarity_train",
    "rare_leaf_fraction",
    "leaf_count_p10",
    "leaf_count_min",
    "leaf_weight_p10",
    "contrib_top1_abs_share",
    "contrib_top3_abs_share",
    "contrib_entropy",
    "contrib_balance",
    "num_material_contrib_features",
    "prob_uncertainty",
]

PREDICTION_LEDGER_DIAGNOSTIC_COLUMNS.extend(
    f"{prefix}_{suffix}"
    for prefix in ("base_lgbm", "meta_lgbm")
    for suffix in _LGBM_DIAGNOSTIC_SUFFIXES
)


class PredictionLedger:
    """Persist top live predictions and later outcome resolution state."""

    def __init__(self, path: str | Path = "data/live_state/prediction_ledger.parquet"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame()
        return pd.read_parquet(self.path)

    def _write_atomic(self, df: pd.DataFrame) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(self.path)

    @staticmethod
    def _identity_columns(df: pd.DataFrame) -> List[str]:
        """Return the stable candidate identity used for append/resolve upserts."""
        if df is None or df.empty:
            return []
        signal_ts = df.get("signal_bar_ts")
        has_signal_ts = False
        if signal_ts is not None:
            try:
                has_signal_ts = bool(
                    pd.to_datetime(signal_ts, utc=True, errors="coerce").notna().any()
                )
            except Exception:
                has_signal_ts = bool(signal_ts.notna().any())
        decision_ts = df.get("decision_ts")
        has_decision_ts = False
        if decision_ts is not None:
            try:
                has_decision_ts = bool(
                    pd.to_datetime(decision_ts, utc=True, errors="coerce").notna().any()
                )
            except Exception:
                has_decision_ts = bool(decision_ts.notna().any())
        # A live signal can be reconsidered more than once before the next
        # feature bar arrives.  Those attempts have different portfolio state
        # and must remain separate for decision replay.  Legacy rows without a
        # decision timestamp keep the older signal-bar upsert behavior.
        time_cols: list[str] = []
        if has_signal_ts:
            time_cols.append("signal_bar_ts")
        if has_decision_ts:
            time_cols.append("decision_ts")
        elif not has_signal_ts:
            time_cols.append("timestamp")
        preferred = tuple(
            time_cols
            + [
                "symbol",
                "side",
                "strategy_id",
                "meta_head_hash",
                "model_artifact_run_id",
                "policy_artifact_run_id",
            ]
        )
        subset = [c for c in preferred if c in df.columns]
        if any(c in subset for c in time_cols) and len(subset) >= 2:
            return subset
        fallback = (
            "timestamp",
            "symbol",
            "side",
            "strategy_id",
            "meta_head_hash",
        )
        return [c for c in fallback if c in df.columns]

    def append_rows(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        new = pd.DataFrame(rows)
        ordered_cols = list(
            dict.fromkeys(list(new.columns) + PREDICTION_LEDGER_DIAGNOSTIC_COLUMNS)
        )
        new = new.reindex(columns=ordered_cols)
        for ts_col in ("timestamp", "signal_bar_ts", "decision_ts"):
            if ts_col in new.columns:
                new[ts_col] = pd.to_datetime(new[ts_col], utc=True, errors="coerce")
        old = self._read()
        out = new if old.empty else pd.concat([old, new], ignore_index=True, sort=False)
        subset = self._identity_columns(out)
        if subset:
            out = out.drop_duplicates(subset=subset, keep="last")
        self._write_atomic(out)

    def load_unresolved(self, *, max_age_hours: int = 168) -> pd.DataFrame:
        df = self._read()
        if df.empty:
            return df
        ts = pd.to_datetime(df.get("timestamp"), utc=True, errors="coerce")
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=int(max_age_hours))
        status = df.get("outcome_status", pd.Series(index=df.index, dtype=object))
        unresolved = status.isna() | status.astype(str).isin({"", "pending", "open"})
        return df.loc[unresolved & (ts >= cutoff)].copy()

    def mark_resolved(self, updates: pd.DataFrame) -> None:
        if updates is None or updates.empty:
            return
        old = self._read()
        if old.empty:
            self._write_atomic(updates.copy())
            return
        key_cols = [
            c
            for c in self._identity_columns(old)
            if c in old.columns and c in updates.columns
        ]
        if not key_cols:
            self._write_atomic(pd.concat([old, updates], ignore_index=True, sort=False))
            return
        old_idx = old.set_index(key_cols, drop=False)
        upd_idx = updates.set_index(key_cols, drop=False)

        # Deduplicate updates, keeping the last one
        upd_idx = upd_idx[~upd_idx.index.duplicated(keep="last")]

        # Safely handle schema evolution without causing dtype incompatibilities
        for col in upd_idx.columns:
            if col not in old_idx.columns:
                old_idx[col] = pd.Series(dtype=upd_idx[col].dtype)

        # Bulk update matching indices
        common_idx = upd_idx.index.intersection(old_idx.index)
        if not common_idx.empty:
            old_idx.loc[common_idx, upd_idx.columns] = upd_idx.loc[common_idx]

        # Append new rows
        new_idx = upd_idx.index.difference(old_idx.index)
        if not new_idx.empty:
            old_idx = pd.concat([old_idx, upd_idx.loc[new_idx]], axis=0, sort=False)

        self._write_atomic(old_idx.reset_index(drop=True))


def top_fraction_rows(
    rows: Iterable[Dict[str, Any]],
    *,
    rank_key: str = "normalized_rank_score",
    pct: float = 0.15,
) -> List[Dict[str, Any]]:
    """Return the top fraction per strategy for ledger persistence."""
    df = pd.DataFrame(list(rows))
    if df.empty or rank_key not in df.columns or "strategy_id" not in df.columns:
        return []
    keep_parts = []
    for _strategy, grp in df.groupby("strategy_id", sort=False):
        n_keep = max(1, int(len(grp) * float(pct)))
        keep_parts.append(grp.sort_values(rank_key, ascending=False).head(n_keep))
    return pd.concat(keep_parts, ignore_index=True).to_dict("records")
