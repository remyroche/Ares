import re

with open("extreme_price_movements/policy_optimiser.py", "r") as f:
    content = f.read()

# Replace the Z-score logic back to median and IQR computed over the candidate set being compared
# within the current optimisation family/stage, as explicitly requested by the reviewer.
new_z_score_logic = """        def _calculate_custom_score(metrics: Dict[str, float], all_metrics_history: List[Dict[str, float]]) -> float:
            if not all_metrics_history:
                return 0.0

            def z(val: float, key: str) -> float:
                arr = np.array([m[key] for m in all_metrics_history])
                if len(arr) == 0:
                    return 0.0
                med = float(np.median(arr))
                q1 = float(np.percentile(arr, 25))
                q3 = float(np.percentile(arr, 75))
                iqr = max(q3 - q1, 1e-6)
                return (val - med) / iqr

            return (
                0.15 * z(metrics.get("net_pnl", 0.0), "net_pnl") +
                0.10 * z(metrics.get("net_pnl", 0.0) / np.sqrt(metrics.get("robust_downside_semi_variance", 1e-6)), "robust_downside_ratio") +
                0.10 * z(metrics.get("pnl_top_25pct_taken_trades", 0.0), "pnl_top_25pct_taken_trades") +
                0.14 * z(metrics.get("weekly_sortino", 0.0), "weekly_sortino") +
                0.11 * z(metrics.get("monthly_sortino", 0.0), "monthly_sortino") +
                0.06 * z(metrics.get("median_pnl_per_winning_trade", 0.0), "median_pnl_per_winning_trade") +
                0.05 * z(metrics.get("weekly_gtp", 0.0), "weekly_gtp") +
                0.05 * z(metrics.get("trade_return_skew", 0.0), "trade_return_skew") -
                0.16 * z(metrics.get("ulcer", 0.0), "ulcer") -
                0.08 * z(metrics.get("tuw", 0.0), "tuw") -
                0.05 * z(metrics.get("pct_negative_trades", 0.0), "pct_negative_trades")
            )"""

content = re.sub(
    r"        # Precompute baseline reference for static Z-score normalization.*?"
    r"                0\.05 \* z\(metrics\.get\(\"pct_negative_trades\", 0\.0\), \"pct_negative_trades\"\)\n"
    r"            \)",
    new_z_score_logic,
    content,
    flags=re.DOTALL
)

# Now we have to fix all the call sites of _calculate_custom_score that we changed
content = content.replace("f_score = _calculate_custom_score(f_metrics)", "f_score = _calculate_custom_score(f_metrics, family_train_scores_all_metrics)")
content = content.replace("return _calculate_custom_score(metrics)", "return _calculate_custom_score(metrics, family_train_scores_all_metrics)")
content = content.replace("baseline_custom_score = _calculate_custom_score(baseline_metrics)", "baseline_custom_score = _calculate_custom_score(family_train_scores_all_metrics[0], family_train_scores_all_metrics)")
content = content.replace("family_val_metric = _calculate_custom_score(family_val_metrics)", "validation_history = [baseline_val_metrics, family_val_metrics]\n            family_val_metric = _calculate_custom_score(family_val_metrics, validation_history)")
content = content.replace("best_overall_val_metric_current = _calculate_custom_score(baseline_val_metrics)", "best_overall_val_metric_current = _calculate_custom_score(baseline_val_metrics, validation_history)")

with open("extreme_price_movements/policy_optimiser.py", "w") as f:
    f.write(content)
