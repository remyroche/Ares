import re

with open("extreme_price_movements/run_pipeline.py", "r") as f:
    code = f.read()

# Update policy_optimiser subview extraction
policy_re = r'if "utility_policy_optimisation" in slice_plan\.get\("materialized_views", \{\}\):.*?cfg\["_active_stage_view"\] = apply_stage_usage_limits\(\s*stage_view, max_assets=cfg\.get\("planned_max_assets"\), max_months=cfg\.get\("planned_max_months"\)\s*\)'

# Actually wait, policy_optimiser corresponds to holdout_strategy_eval["sub_views"]["policy_optimiser"]
# What about 'utility_policy_optimisation'? That's for the 'optimise' step (sizer tuning).
# Oh wait, let's trace this carefully:
# "holdout_strategy_eval" contains "policy_optimiser" and "backtest_eval".
# 'optimise' uses 'utility_policy_optimisation' which is distinct.
# The user instruction states:
# policy_optimiser mode must extract the policy_optimiser sub-view from holdout_strategy_eval.
# backtest, inference_backtest, and oos_eval modes must extract the backtest_eval sub-view from holdout_strategy_eval.
# optimise mode extracts utility_policy_optimisation.

policy_inject = """
                if "holdout_strategy_eval" in slice_plan.get("materialized_views", {}):
                    stage_view = slice_plan["materialized_views"]["holdout_strategy_eval"].get("sub_views", {}).get("policy_optimiser")
                    if stage_view:
                        cfg["_active_stage_view"] = apply_stage_usage_limits(
                            stage_view, max_assets=cfg.get("planned_max_assets"), max_months=cfg.get("planned_max_months")
                        )
                    else:
                        tprint("Warning: policy_optimiser sub_view not found")
"""
code = re.sub(
    r'if "utility_policy_optimisation" in slice_plan\.get\("materialized_views", \{\}\):.*?cfg\["_active_stage_view"\].*?\)',
    policy_inject.strip("\n"),
    code,
    count=1, # Replace the one inside policy_optimiser block
    flags=re.DOTALL
)

# Replace the one for oos_eval, backtest, inference_backtest

oos_inject = """
                if "holdout_strategy_eval" in slice_plan.get("materialized_views", {}):
                    stage_view = slice_plan["materialized_views"]["holdout_strategy_eval"].get("sub_views", {}).get("backtest_eval")
                    if stage_view:
                        cfg["_active_stage_view"] = apply_stage_usage_limits(
                            stage_view, max_assets=cfg.get("planned_max_assets"), max_months=cfg.get("planned_max_months")
                        )
                    else:
                        tprint("Warning: backtest_eval sub_view not found")
"""

# Let's just do a string replacement for backtest, inference_backtest and oos_eval.
# We had injected them earlier using `modify_handler("run_backtest", code, "holdout_strategy_eval")` etc.

backtest_search = r'if "holdout_strategy_eval" in slice_plan\.get\("materialized_views", \{\}\):\n\s*stage_view = slice_plan\["materialized_views"\]\["holdout_strategy_eval"\]\n\s*stage_view = apply_stage_usage_limits\(\n\s*stage_view, \n\s*max_assets=cfg\.get\("planned_max_assets"\), \n\s*max_months=cfg\.get\("planned_max_months"\)\n\s*\)\n\s*cfg\["_active_stage_view"\] = stage_view'

# My inject strings might differ slightly due to formatting. Let's find exactly what I injected.
