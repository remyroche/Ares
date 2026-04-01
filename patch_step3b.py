import re

with open("extreme_price_movements/lgbm_based_mask_generation.py", "r") as f:
    source = f.read()

# I need to properly get timestamps. Usually the data passed to assess_rules has a `timestamp` column.
# Let's write a function that extracts trades with proper timestamps.

trade_func = """
    @staticmethod
    def _find_threshold_star(
        oof_preds: np.ndarray,
        fwd_ret: np.ndarray,
        data: pd.DataFrame,
        horizon: int,
        round_fee: float = 0.0015,
        forbid_concurrent: bool = True
    ) -> Tuple[Optional[float], List[Dict[str, Any]], Dict[str, Any]]:
        valid_mask = np.isfinite(oof_preds) & np.isfinite(fwd_ret)

        # Build all valid trades
        all_trades = []
        if np.any(valid_mask):
            valid_indices = np.where(valid_mask)[0]

            if "timestamp" in data.columns:
                timestamps = pd.to_datetime(data["timestamp"]).to_numpy()
                exit_offsets = pd.to_timedelta(horizon, unit='h').to_numpy() # assumption of hour, wait, let's just use index if no timestamp
            else:
                timestamps = np.arange(len(data))
                exit_offsets = horizon

            symbols = data["symbol"].to_numpy() if "symbol" in data.columns else np.zeros(len(data))

            for idx in valid_indices:
                if "timestamp" in data.columns:
                    exit_time = timestamps[idx] + np.timedelta64(horizon, 'h')
                else:
                    exit_time = timestamps[idx] + horizon

                all_trades.append({
                    "confidence_score": float(oof_preds[idx]),
                    "gross_trade_return": float(fwd_ret[idx]),
                    "entry_time": timestamps[idx],
                    "exit_time": exit_time,
                    "symbol": symbols[idx]
                })

        if not all_trades:
            return None, [], {"reason": "no valid trades"}

        best_t = None
        max_net_expectancy = -np.inf
        best_trade_rate = 0.0
        best_selected = []

        for t in np.arange(0.60, 0.95, 0.05):
            t_float = float(t)
            eligible = [tr for tr in all_trades if tr["confidence_score"] >= t_float]

            if not eligible:
                continue

            eligible.sort(key=lambda x: x["entry_time"])

            if forbid_concurrent:
                selected = []
                # Concurrency is symbol-specific
                current_open_until = {}
                for tr in eligible:
                    sym = tr["symbol"]
                    if sym not in current_open_until or tr["entry_time"] >= current_open_until[sym]:
                        selected.append(tr)
                        current_open_until[sym] = tr["exit_time"]
            else:
                selected = eligible

            if not selected:
                continue

            returns = np.array([tr["gross_trade_return"] for tr in selected])
            net_expectancy_t = float(np.mean(returns - round_fee))

            if net_expectancy_t > max_net_expectancy:
                max_net_expectancy = net_expectancy_t
                best_t = t_float
                best_trade_rate = len(selected) # Temporary, will be divided by symbol days later
                best_selected = selected

            if net_expectancy_t > 0:
                return t_float, selected, {}

        return None, best_selected, {
            "reason": "no positive post-fee expectancy threshold",
            "max_net_expectancy": max_net_expectancy,
            "best_threshold_candidate": best_t,
            "trades_per_symbol_day_at_best_t": best_trade_rate # This needs adjustment for symbol_days
        }
"""

# Replace the previous _find_threshold_star
pattern = r"    @staticmethod\n    def _find_threshold_star.*?return None, \[\], \{.*?\}\n"
match = re.search(pattern, source, re.DOTALL)
if match:
    source = source[:match.start()] + trade_func + source[match.end():]
    with open("extreme_price_movements/lgbm_based_mask_generation.py", "w") as f:
        f.write(source)
    print("Patched successfully")
else:
    print("Could not find _find_threshold_star to replace")
