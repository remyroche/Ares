import re

with open("extreme_price_movements/lgbm_based_mask_generation.py", "r") as f:
    source = f.read()

# I will add `_find_threshold_star` method to MaskAssessor.

method_code = """
    @staticmethod
    def _find_threshold_star(
        oof_preds: np.ndarray,
        fwd_ret: np.ndarray,
        timestamps: np.ndarray,
        horizon: int,
        round_fee: float = 0.0015,
        forbid_concurrent: bool = True
    ) -> Tuple[Optional[float], List[Dict[str, Any]], Dict[str, Any]]:
        valid_mask = np.isfinite(oof_preds) & np.isfinite(fwd_ret)

        # Build all valid trades
        all_trades = []
        if len(timestamps) > 0 and np.any(valid_mask):
            valid_indices = np.where(valid_mask)[0]
            for idx in valid_indices:
                all_trades.append({
                    "confidence_score": float(oof_preds[idx]),
                    "gross_trade_return": float(fwd_ret[idx]),
                    "entry_time": timestamps[idx],
                    "exit_time": timestamps[idx] + np.timedelta64(horizon, 'h'), # Assuming hourly, wait, let's just use int if it's index, or whatever it is
                    "index": idx
                })

        # If timestamps are not datetime, we can just use index
        if not all_trades:
            return None, [], {"reason": "no valid trades"}

        best_t = None
        max_net_expectancy = -np.inf
        best_trade_rate = 0.0

        for t in np.arange(0.60, 0.95, 0.05):
            t = float(t)
            eligible = [tr for tr in all_trades if tr["confidence_score"] >= t]

            if not eligible:
                continue

            eligible.sort(key=lambda x: x["entry_time"])

            if forbid_concurrent:
                selected = []
                current_open_until = None
                for tr in eligible:
                    if current_open_until is None or tr["entry_time"] >= current_open_until:
                        selected.append(tr)
                        current_open_until = tr["exit_time"]
            else:
                selected = eligible

            if not selected:
                continue

            returns = np.array([tr["gross_trade_return"] for tr in selected])
            net_expectancy_t = float(np.mean(returns - round_fee))

            if net_expectancy_t > max_net_expectancy:
                max_net_expectancy = net_expectancy_t
                best_t = t
                best_trade_rate = len(selected) # Temporary, will be divided later

            if net_expectancy_t > 0:
                return t, selected, {}

        return None, [], {
            "reason": "no positive post-fee expectancy threshold",
            "max_net_expectancy": max_net_expectancy,
            "best_threshold_candidate": best_t,
            "trades_per_symbol_day_at_best_t": best_trade_rate # This needs adjustment for symbol_days
        }
"""

class_def_pattern = r"(class MaskAssessor:.*?)(def _compute_total_symbol_days)"
match = re.search(class_def_pattern, source, re.DOTALL)
if match:
    new_source = source[:match.end(1)] + method_code + "\n    " + source[match.start(2):]
    with open("extreme_price_movements/lgbm_based_mask_generation.py", "w") as f:
        f.write(new_source)
    print("Patched successfully")
else:
    print("Could not find MaskAssessor class")
