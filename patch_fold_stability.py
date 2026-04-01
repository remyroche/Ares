import re

with open("extreme_price_movements/lgbm_based_mask_generation.py", "r") as f:
    source = f.read()

old_compute_pnl = """        capital = starting_capital
        for wr in weighted_net_returns:
            capital = capital * (1.0 + wr)

        ridge_pnl_raw = capital - starting_capital

        return {
            "ridge_pnl_raw": ridge_pnl_raw,
            "selected_trades": selected_trades,
            "weighted_net_returns": weighted_net_returns,
            "ending_capital": capital,
        }"""

new_compute_pnl = """        import collections
        capital = starting_capital
        fold_pnls = collections.defaultdict(lambda: starting_capital)
        for i, t in enumerate(selected_trades):
            wr = weighted_net_returns[i]
            capital = capital * (1.0 + wr)
            if "fold_id" in t:
                fold_pnls[t["fold_id"]] = fold_pnls[t["fold_id"]] * (1.0 + wr)

        ridge_pnl_raw = capital - starting_capital

        fold_pnl_raws = {k: v - starting_capital for k, v in fold_pnls.items()}

        return {
            "ridge_pnl_raw": ridge_pnl_raw,
            "fold_pnl_raws": fold_pnl_raws,
            "selected_trades": selected_trades,
            "weighted_net_returns": weighted_net_returns,
            "ending_capital": capital,
        }"""

source = source.replace(old_compute_pnl, new_compute_pnl)

# Also fix the empty return
empty_ret_old = """        if len(eligible_trades) == 0:
            return {
                "ridge_pnl_raw": 0.0,
                "selected_trades": [],
                "weighted_net_returns": [],
                "ending_capital": starting_capital,
            }"""

empty_ret_new = """        if len(eligible_trades) == 0:
            return {
                "ridge_pnl_raw": 0.0,
                "fold_pnl_raws": {},
                "selected_trades": [],
                "weighted_net_returns": [],
                "ending_capital": starting_capital,
            }"""

source = source.replace(empty_ret_old, empty_ret_new)

empty_ret_old2 = """        if len(selected_trades) == 0:
            return {
                "ridge_pnl_raw": 0.0,
                "selected_trades": [],
                "weighted_net_returns": [],
                "ending_capital": starting_capital,
            }"""

empty_ret_new2 = """        if len(selected_trades) == 0:
            return {
                "ridge_pnl_raw": 0.0,
                "fold_pnl_raws": {},
                "selected_trades": [],
                "weighted_net_returns": [],
                "ending_capital": starting_capital,
            }"""

source = source.replace(empty_ret_old2, empty_ret_new2)

with open("extreme_price_movements/lgbm_based_mask_generation.py", "w") as f:
    f.write(source)
print("Patched compute_pnl to return fold-level PNL.")
