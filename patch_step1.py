import re

with open("extreme_price_movements/lgbm_based_mask_generation.py", "r") as f:
    source = f.read()

pnl_func = """
    @staticmethod
    def compute_ridge_pnl(
        trades: List[Dict[str, Any]],
        threshold_star: float,
        round_fee: float = 0.0015,
        min_weight: float = 0.10,
        max_weight: float = 0.30,
        convex_power: float = 2.0,
        starting_capital: float = 1.0,
        forbid_concurrent: bool = True
    ) -> Dict[str, Any]:
        eligible_trades = [t for t in trades if t["confidence_score"] >= threshold_star]
        if len(eligible_trades) == 0:
            return {
                "ridge_pnl_raw": 0.0,
                "selected_trades": [],
                "weighted_net_returns": [],
                "ending_capital": starting_capital,
            }

        eligible_trades.sort(key=lambda x: x["entry_time"])

        if forbid_concurrent:
            selected_trades = []
            current_open_until = None
            for t in eligible_trades:
                if current_open_until is None or t["entry_time"] >= current_open_until:
                    selected_trades.append(t)
                    current_open_until = t["exit_time"]
        else:
            selected_trades = eligible_trades

        if len(selected_trades) == 0:
            return {
                "ridge_pnl_raw": 0.0,
                "selected_trades": [],
                "weighted_net_returns": [],
                "ending_capital": starting_capital,
            }

        weighted_net_returns = []
        for t in selected_trades:
            conf = t["confidence_score"]
            denom = max(1.0 - threshold_star, 1e-9)
            normalized_conf = min(max((conf - threshold_star) / denom, 0.0), 1.0)

            position_weight = (
                min_weight + (max_weight - min_weight) * (normalized_conf ** convex_power)
            )

            net_trade_return = t["gross_trade_return"] - round_fee
            weighted_net_return = position_weight * net_trade_return
            weighted_net_returns.append(weighted_net_return)

        capital = starting_capital
        for wr in weighted_net_returns:
            capital = capital * (1.0 + wr)

        ridge_pnl_raw = capital - starting_capital

        return {
            "ridge_pnl_raw": ridge_pnl_raw,
            "selected_trades": selected_trades,
            "weighted_net_returns": weighted_net_returns,
            "ending_capital": capital,
        }

    @staticmethod
    def compute_ridge_trade_sortino(
        gross_trade_returns: np.ndarray,
        confidence_scores: np.ndarray,
        threshold_star: float,
        round_fee: float = 0.0015,
        min_weight: float = 0.10,
        max_weight: float = 0.30,
        convex_power: float = 2.0,
        sortino_scale: float = 2.0,
        eps: float = 1e-9,
    ) -> Dict[str, Any]:
        r = np.asarray(gross_trade_returns, dtype=float)
        s = np.asarray(confidence_scores, dtype=float)

        if r.shape != s.shape:
            raise ValueError("gross_trade_returns and confidence_scores must have the same shape")

        n = r.size
        if n == 0:
            return {
                "selected_mask": np.zeros(0, dtype=bool),
                "sizing_weights": np.zeros(0, dtype=float),
                "net_weighted_returns": np.zeros(0, dtype=float),
                "ridge_trade_sortino_raw": 0.0,
                "ridge_trade_sortino": 0.0,
            }

        selected_mask = s >= threshold_star

        if not np.any(selected_mask):
            return {
                "selected_mask": selected_mask,
                "sizing_weights": np.zeros(n, dtype=float),
                "net_weighted_returns": np.zeros(n, dtype=float),
                "ridge_trade_sortino_raw": 0.0,
                "ridge_trade_sortino": 0.0,
            }

        kept_scores = s[selected_mask]

        denom = max(1.0 - threshold_star, eps)
        normalized_scores = np.clip((kept_scores - threshold_star) / denom, 0.0, 1.0)

        kept_weights = min_weight + (max_weight - min_weight) * (normalized_scores ** convex_power)

        sizing_weights = np.zeros(n, dtype=float)
        sizing_weights[selected_mask] = kept_weights

        net_weighted_returns = np.zeros(n, dtype=float)
        net_weighted_returns[selected_mask] = kept_weights * (r[selected_mask] - round_fee)

        realized = net_weighted_returns[selected_mask]

        mean_ret = float(np.mean(realized)) if realized.size else 0.0
        downside = np.minimum(realized, 0.0)
        downside_dev = float(np.sqrt(np.mean(downside ** 2))) if realized.size else 0.0

        ridge_trade_sortino_raw = mean_ret / (downside_dev + eps)

        ridge_trade_sortino = float(
            np.tanh(max(ridge_trade_sortino_raw, 0.0) / sortino_scale)
        )

        return {
            "selected_mask": selected_mask,
            "sizing_weights": sizing_weights,
            "net_weighted_returns": net_weighted_returns,
            "ridge_trade_sortino_raw": ridge_trade_sortino_raw,
            "ridge_trade_sortino": ridge_trade_sortino,
        }
"""

class_def_pattern = r"(class MaskAssessor:.*?)(def _compute_total_symbol_days)"

# Insert before _compute_total_symbol_days
match = re.search(class_def_pattern, source, re.DOTALL)
if match:
    new_source = source[:match.end(1)] + pnl_func + "\n    " + source[match.start(2):]
    with open("extreme_price_movements/lgbm_based_mask_generation.py", "w") as f:
        f.write(new_source)
    print("Patched successfully")
else:
    print("Could not find MaskAssessor class")
