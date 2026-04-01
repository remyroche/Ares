import re

with open("extreme_price_movements/lgbm_based_mask_generation.py", "r") as f:
    source = f.read()

# I will replace the learnability section and Final Regime Score section with the new metric calculations.

old_code = """
            # 7. Learnability (Efficiency Frontier) - expensive section
            subset_oof_coverage = 0.0
            mask_auc = np.nan
            lift = np.nan
            entropy_red = np.nan
            if not rejected:
                run_ridge = False
                if (
                    hasattr(self, "bucket_ridge_keys")
                    and group_bucket_key in self.bucket_ridge_keys
                ):
                    if canonical_key in self.bucket_ridge_keys[group_bucket_key]:
                        run_ridge = True

                if run_ridge:
                    mask_auc, subset_oof_coverage, oof_preds = self._compute_subset_auc(
                        X, target_ret, mask, folds
                    )
                    if np.isfinite(mask_auc) and np.isfinite(global_auc):
                        lift = mask_auc - global_auc
                else:
                    subset_oof_coverage = float(np.mean(mask))
                    lift = 0.0  # Neutral lift
                    rejected, rejection_reason = True, "not_in_top_ridge_candidates"

                mask_entropy = self._compute_entropy(target_ret[mask])
                entropy_red = 1.0 - (mask_entropy / (global_entropy + 1e-9))
                if subset_oof_coverage < min_oof_coverage:
                    rejected, rejection_reason = (
                        True,
                        "insufficient_subset_oof_coverage",
                    )
                elif not np.isfinite(lift):
                    rejected, rejection_reason = True, "missing_learnability"
                elif run_ridge and lift < 0.01:  # Lower threshold for lift (was 1.10)
                    rejected, rejection_reason = True, "low_lift"

            # 8. Event-based Expected Value
            tp_payoff = tp_atr  # TP payoff in ATR units
            sl_payoff = sl_atr  # SL payoff in ATR units
            timeout_payoff = mean_ret_mask  # Average return for timeouts

            ev_per_event = (
                tbm_metrics["tp_rate"] * tp_payoff
                - tbm_metrics["sl_rate"] * sl_payoff
                + tbm_metrics["timeout_rate"] * timeout_payoff
            )

            # Fetch cheap_rank for Final Regime Score
            cheap_rank = bucket_cheap_ranks.get(group_bucket_key, {}).get(
                canonical_key, -np.inf
            )
            if not np.isfinite(cheap_rank):
                cheap_rank = 0.0

            # 9. Final Regime Score
            regime_score = (
                0.30 * cheap_rank
                + 0.20 * lift
                + 0.20 * ret_uplift
                + 0.20 * ev_per_event
                + 0.10 * (mask_auc if np.isfinite(mask_auc) else 0.0)
            )
"""

new_code = """
            # 7. Learnability (Efficiency Frontier) & Trade Realization
            subset_oof_coverage = 0.0
            mask_auc = np.nan
            lift = np.nan
            entropy_red = np.nan

            # New Metrics
            ridge_pnl_raw = 0.0
            ridge_trade_sortino_raw = 0.0
            ridge_trade_sortino = 0.0
            threshold_star = np.nan
            trades_per_symbol_day_above_t = 0.0
            selected_trades = []

            if not rejected:
                run_ridge = False
                if (
                    hasattr(self, "bucket_ridge_keys")
                    and group_bucket_key in self.bucket_ridge_keys
                ):
                    if canonical_key in self.bucket_ridge_keys[group_bucket_key]:
                        run_ridge = True

                if run_ridge:
                    mask_auc, subset_oof_coverage, oof_preds = self._compute_subset_auc(
                        X, target_ret, mask, folds
                    )
                    if np.isfinite(mask_auc) and np.isfinite(global_auc):
                        lift = mask_auc - global_auc

                    # Find threshold_star and generate trades
                    t_star, sel_trades, reject_info = self._find_threshold_star(
                        oof_preds=oof_preds,
                        fwd_ret=target_ret,
                        data=data,
                        horizon=horizon,
                        round_fee=0.0015,
                        forbid_concurrent=True
                    )

                    if t_star is None:
                        rejected = True
                        rejection_reason = "no positive post-fee expectancy threshold"
                        rejection_data = reject_info
                    else:
                        threshold_star = t_star
                        selected_trades = sel_trades
                        trades_per_symbol_day_above_t = len(selected_trades) / total_symbol_days

                        if trades_per_symbol_day_above_t < 0.1:
                            rejected = True
                            rejection_reason = "low_trade_rate_above_threshold_star"
                        else:
                            # Compute Ridge PNL
                            pnl_info = self.compute_ridge_pnl(
                                trades=selected_trades,
                                threshold_star=threshold_star,
                                round_fee=0.0015,
                                min_weight=0.10,
                                max_weight=0.30,
                                convex_power=2.0,
                                starting_capital=1.0,
                                forbid_concurrent=True
                            )
                            ridge_pnl_raw = pnl_info["ridge_pnl_raw"]

                            # Compute Ridge Sortino
                            gross_returns = np.array([tr["gross_trade_return"] for tr in selected_trades])
                            scores = np.array([tr["confidence_score"] for tr in selected_trades])

                            sortino_info = self.compute_ridge_trade_sortino(
                                gross_trade_returns=gross_returns,
                                confidence_scores=scores,
                                threshold_star=threshold_star,
                                round_fee=0.0015,
                                min_weight=0.10,
                                max_weight=0.30,
                                convex_power=2.0,
                                sortino_scale=2.0,
                                eps=1e-9
                            )
                            ridge_trade_sortino_raw = sortino_info["ridge_trade_sortino_raw"]
                            ridge_trade_sortino = sortino_info["ridge_trade_sortino"]

                else:
                    subset_oof_coverage = float(np.mean(mask))
                    lift = 0.0  # Neutral lift
                    rejected, rejection_reason = True, "not_in_top_ridge_candidates"

                mask_entropy = self._compute_entropy(target_ret[mask])
                entropy_red = 1.0 - (mask_entropy / (global_entropy + 1e-9))
                if subset_oof_coverage < min_oof_coverage:
                    rejected, rejection_reason = (
                        True,
                        "insufficient_subset_oof_coverage",
                    )
                elif not np.isfinite(lift):
                    rejected, rejection_reason = True, "missing_learnability"
                elif run_ridge and lift <= 0.0:  # Hard gate on lift <= 0
                    rejected, rejection_reason = True, "low_lift"

            # 8. Event-based Expected Value
            tp_payoff = tp_atr  # TP payoff in ATR units
            sl_payoff = sl_atr  # SL payoff in ATR units
            timeout_payoff = mean_ret_mask  # Average return for timeouts

            ev_per_event = (
                tbm_metrics["tp_rate"] * tp_payoff
                - tbm_metrics["sl_rate"] * sl_payoff
                + tbm_metrics["timeout_rate"] * timeout_payoff
            )

            if ev_per_event <= 0:
                rejected, rejection_reason = True, "ev_per_event <= 0"

            # base_regime_score is computed after all rules are assessed (due to PNL scaling)
            regime_score = np.nan
"""

# Perform replacement
if old_code in source:
    source = source.replace(old_code, new_code)
    with open("extreme_price_movements/lgbm_based_mask_generation.py", "w") as f:
        f.write(source)
    print("Patched correctly!")
else:
    print("Could not find the target code to replace. Trying regex...")
    # fall back to more robust search
    pattern = re.compile(re.escape(old_code), re.DOTALL)
    match = pattern.search(source)
    if match:
        source = source[:match.start()] + new_code + source[match.end():]
        with open("extreme_price_movements/lgbm_based_mask_generation.py", "w") as f:
            f.write(source)
        print("Patched via regex.")
    else:
        print("Still could not find code block.")
