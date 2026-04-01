import re

with open("extreme_price_movements/lgbm_based_mask_generation.py", "r") as f:
    source = f.read()

# Add the new metrics to the output dict
old_dict = """                    "mask_auc": mask_auc,
                    "global_auc": global_auc,
                    "global_entropy": global_entropy,
                    "entropy_reduction": entropy_red,
                    "tp_rate": tbm_metrics["tp_rate"],"""

new_dict = """                    "mask_auc": mask_auc,
                    "global_auc": global_auc,
                    "global_entropy": global_entropy,
                    "entropy_reduction": entropy_red,
                    "ridge_pnl_raw": ridge_pnl_raw,
                    "ridge_trade_sortino": ridge_trade_sortino,
                    "ridge_trade_sortino_raw": ridge_trade_sortino_raw,
                    "threshold_star": threshold_star,
                    "trades_per_symbol_day_above_t": trades_per_symbol_day_above_t,
                    "tp_rate": tbm_metrics["tp_rate"],"""

source = source.replace(old_dict, new_dict)
with open("extreme_price_movements/lgbm_based_mask_generation.py", "w") as f:
    f.write(source)
print("Added metrics to assessment_results")
