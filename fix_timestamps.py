with open("extreme_price_movements/lgbm_based_mask_generation.py", "r") as f:
    source = f.read()

import re
# We need to change the _find_threshold_star method so that instead of adding a timedelta,
# it simply uses the numeric indices or safe array offsets.
# The user prompt:
# if forbid_concurrent:
#    selected_trades = []
#    current_open_until = None
#    for t in eligible_trades:
#        if current_open_until is None or t.entry_time >= current_open_until:
#            selected_trades.append(t)
#            current_open_until = t.exit_time

# If entry_time and exit_time are just integer indices, it solves all timestamp frequency problems natively!
# So entry_time = idx, exit_time = idx + horizon.

old_trade_func = r"""        all_trades = \[\]
        if np\.any\(valid_mask\):
            valid_indices = np\.where\(valid_mask\)\[0\]

            has_timestamp = "timestamp" in data\.columns
            if has_timestamp:
                timestamps = pd\.to_datetime\(data\["timestamp"\]\)\.to_numpy\(\)
                exit_offsets = pd\.to_timedelta\(horizon, unit='h'\)\.to_numpy\(\)
            else:
                timestamps = np\.arange\(len\(data\)\)
                exit_offsets = horizon

            has_symbol = "symbol" in data\.columns
            if has_symbol:
                symbols = data\["symbol"\]\.to_numpy\(\)
            else:
                symbols = np\.zeros\(len\(data\)\)

            # Optimize trade creation
            selected_oof = oof_preds\[valid_indices\]
            selected_ret = fwd_ret\[valid_indices\]
            selected_ts = timestamps\[valid_indices\]
            selected_sym = symbols\[valid_indices\]

            for i in range\(len\(valid_indices\)\):
                all_trades\.append\(\{
                    "confidence_score": float\(selected_oof\[i\]\),
                    "gross_trade_return": float\(selected_ret\[i\]\),
                    "entry_time": selected_ts\[i\],
                    "exit_time": selected_ts\[i\] \+ exit_offsets if has_timestamp else selected_ts\[i\] \+ horizon,
                    "symbol": selected_sym\[i\]
                \}\)"""

new_trade_func = """        all_trades = []
        if np.any(valid_mask):
            valid_indices = np.where(valid_mask)[0]

            has_symbol = "symbol" in data.columns
            if has_symbol:
                symbols = data["symbol"].to_numpy()
            else:
                symbols = np.zeros(len(data))

            # Optimize trade creation
            selected_oof = oof_preds[valid_indices]
            selected_ret = fwd_ret[valid_indices]
            selected_sym = symbols[valid_indices]

            for i in range(len(valid_indices)):
                idx = int(valid_indices[i])
                all_trades.append({
                    "confidence_score": float(selected_oof[i]),
                    "gross_trade_return": float(selected_ret[i]),
                    "entry_time": idx,
                    "exit_time": idx + horizon,
                    "symbol": selected_sym[i],
                    "fold_idx": idx # We will map fold_id later, or we can just keep idx and map it.
                })"""

source = re.sub(old_trade_func, new_trade_func, source, flags=re.DOTALL)

with open("extreme_price_movements/lgbm_based_mask_generation.py", "w") as f:
    f.write(source)
