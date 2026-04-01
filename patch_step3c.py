import re

with open("extreme_price_movements/lgbm_based_mask_generation.py", "r") as f:
    source = f.read()

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

        all_trades = []
        if np.any(valid_mask):
            valid_indices = np.where(valid_mask)[0]

            has_timestamp = "timestamp" in data.columns
            if has_timestamp:
                timestamps = pd.to_datetime(data["timestamp"]).to_numpy()
                exit_offsets = pd.to_timedelta(horizon, unit='h').to_numpy()
            else:
                timestamps = np.arange(len(data))
                exit_offsets = horizon

            has_symbol = "symbol" in data.columns
            if has_symbol:
                symbols = data["symbol"].to_numpy()
            else:
                symbols = np.zeros(len(data))

            # Optimize trade creation
            selected_oof = oof_preds[valid_indices]
            selected_ret = fwd_ret[valid_indices]
            selected_ts = timestamps[valid_indices]
            selected_sym = symbols[valid_indices]

            for i in range(len(valid_indices)):
                all_trades.append({
                    "confidence_score": float(selected_oof[i]),
                    "gross_trade_return": float(selected_ret[i]),
                    "entry_time": selected_ts[i],
                    "exit_time": selected_ts[i] + exit_offsets if has_timestamp else selected_ts[i] + horizon,
                    "symbol": selected_sym[i]
                })

        if not all_trades:
            return None, [], {"reason": "no valid trades"}

        best_t = None
        max_net_expectancy = -np.inf
        best_trade_rate = 0.0
        best_selected = []

        # Try thresholds on a 0-1 confidence scale from 0.60 to 0.90 inclusive, step size 0.05
        # Add a small epsilon to 0.90 to include it
        for t in np.arange(0.60, 0.95, 0.05):
            t_float = float(t)
            eligible = [tr for tr in all_trades if tr["confidence_score"] >= t_float]

            if not eligible:
                continue

            eligible.sort(key=lambda x: x["entry_time"])

            if forbid_concurrent:
                selected = []
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

            if net_expectancy_t > 0:
                # threshold_star is the smallest confidence threshold t such that net_expectancy_t > 0
                return t_float, selected, {}

            if net_expectancy_t > max_net_expectancy:
                max_net_expectancy = net_expectancy_t
                best_t = t_float
                best_trade_rate = len(selected) # Temporary, will be divided by symbol days later
                best_selected = selected

        return None, best_selected, {
            "reason": "no positive post-fee expectancy threshold",
            "max_net_expectancy": max_net_expectancy,
            "best_threshold_candidate": best_t,
            "trades_per_symbol_day_at_best_t": best_trade_rate # This needs adjustment for symbol_days
        }
"""

pattern = r"    @staticmethod\n    def _find_threshold_star.*?return None, best_selected, \{\n.*?\}\n"
match = re.search(pattern, source, re.DOTALL)
if match:
    source = source[:match.start()] + trade_func + source[match.end():]
    with open("extreme_price_movements/lgbm_based_mask_generation.py", "w") as f:
        f.write(source)
    print("Patched successfully")
else:
    print("Could not find _find_threshold_star to replace")
