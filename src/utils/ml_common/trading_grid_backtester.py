import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional


def _bars_per_year_from_timeframe(timeframe: str) -> float:
    tf = str(timeframe).lower().strip()
    try:
        if tf.endswith("m") and tf[:-1].isdigit():
            minutes = int(tf[:-1])
            if minutes <= 0:
                return 365.0
            bars_per_day = (24 * 60) / minutes
            return bars_per_day * 365.0
        if tf.endswith("h") and tf[:-1].isdigit():
            hours = int(tf[:-1])
            if hours <= 0:
                return 365.0
            bars_per_day = 24 / hours
            return bars_per_day * 365.0
        if tf.endswith("d") and tf[:-1].isdigit():
            days = int(tf[:-1])
            if days <= 0:
                return 365.0
            bars_per_day = 1.0 / days
            return bars_per_day * 365.0
        if tf.endswith("w") and tf[:-1].isdigit():
            weeks = int(tf[:-1])
            if weeks <= 0:
                return 52.0
            bars_per_week = 1.0 / weeks
            return bars_per_week * 52.0
    except Exception:
        pass
    return 365.0


def _compute_basic_performance_metrics(returns: pd.Series, timeframe: str) -> Dict[str, float]:
    n_bars = int(len(returns))
    if n_bars == 0:
        return {
            "bars": 0,
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
        }

    total_return = float((1.0 + returns).prod() - 1.0)

    # ------------------------------------------------------------------
    # Use approximate *daily* returns for annualization instead of
    # per-bar scaling. This keeps Sharpe/Sortino/Calmar in a realistic
    # numeric range for high-frequency strategies while preserving
    # ordering between configurations.
    # ------------------------------------------------------------------
    bars_per_year = _bars_per_year_from_timeframe(timeframe)
    periods_per_year = 365.0

    daily_returns: List[float] = []
    if n_bars > 0 and bars_per_year > 0.0:
        approx_bars_per_day = bars_per_year / periods_per_year
        bars_per_day = int(round(approx_bars_per_day)) if approx_bars_per_day > 0.0 else 0
        if bars_per_day <= 1:
            # Treat each bar as one period when timeframe is already daily+
            arr = returns.to_numpy(dtype=float)
            for val in arr:
                if np.isfinite(val):
                    daily_returns.append(float(val))
        else:
            # Chunk contiguous bars into approximate "days" based on
            # the inferred bars_per_day for this timeframe.
            arr = returns.to_numpy(dtype=float)
            for start in range(0, n_bars, bars_per_day):
                segment = arr[start : start + bars_per_day]
                if segment.size == 0:
                    continue
                day_ret = float((1.0 + segment).prod() - 1.0)
                if np.isfinite(day_ret):
                    daily_returns.append(day_ret)

    daily_arr = np.array(daily_returns, dtype=float)
    if daily_arr.size >= 2:
        mean_daily = float(daily_arr.mean())
        std_daily = float(daily_arr.std())
        annualized_return = float(mean_daily * periods_per_year)
        annualized_vol = float(std_daily * np.sqrt(periods_per_year)) if std_daily > 0.0 else 0.0
        risk_free = 0.0
        sharpe = float((annualized_return - risk_free) / annualized_vol) if annualized_vol > 0.0 else 0.0

        downside_daily = daily_arr[daily_arr < 0.0]
        if downside_daily.size > 1:
            downside_std = float(downside_daily.std() * np.sqrt(periods_per_year))
            sortino = float((annualized_return - risk_free) / downside_std) if downside_std > 0.0 else sharpe
        else:
            sortino = sharpe
    else:
        annualized_return = 0.0
        annualized_vol = 0.0
        sharpe = 0.0
        sortino = 0.0

    equity = (1.0 + returns).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_drawdown = float(drawdown.min()) if len(drawdown) > 0 else 0.0

    positive = returns[returns > 0]
    negative = returns[returns < 0]
    n_pos = int(len(positive))
    n_neg = int(len(negative))
    n_nonzero = n_pos + n_neg
    win_rate = float(n_pos / n_nonzero) if n_nonzero > 0 else 0.0
    avg_win = float(positive.mean()) if n_pos > 0 else 0.0
    avg_loss = float(negative.mean()) if n_neg > 0 else 0.0
    gross_profit = float(positive.sum()) if n_pos > 0 else 0.0
    gross_loss = float(-negative.sum()) if n_neg > 0 else 0.0
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else 0.0

    return {
        "bars": n_bars,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }


def _compute_period_returns(returns: pd.Series, periods: int) -> List[float]:
    n = len(returns)
    if n == 0 or periods <= 0:
        return []
    bounds = [int(round(i * n / periods)) for i in range(periods + 1)]
    result: List[float] = []
    for i in range(periods):
        start = bounds[i]
        end = bounds[i + 1]
        if end <= start:
            result.append(0.0)
        else:
            segment = returns.iloc[start:end]
            segment_ret = float((1.0 + segment).prod() - 1.0)
            result.append(segment_ret)
    return result


def run_simple_long_grid_backtest(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    raw_returns: pd.Series,
    predictions: pd.Series,
    confidence: pd.Series,
    ml_df: pd.DataFrame,
    timeframe: str,
    fee_rate: float = 0.003,
    regime_col: Optional[str] = None,
    max_holding_bars: int = 6,
    tp_values: Optional[List[float]] = None,
    sl_values: Optional[List[float]] = None,
    trail_distance_atr_mult: Optional[float] = None,
    trail_atr_lookback: int = 14,
    gate_mask: Optional[pd.Series] = None,
    gate_prob: Optional[pd.Series] = None,
    gate_prob_threshold: float = 0.5,
    trail_distance_atr_mult_values: Optional[List[float]] = None,
) -> pd.DataFrame:
    index = close.index
    close = close.reindex(index).astype(float)
    high = high.reindex(index).astype(float)
    low = low.reindex(index).astype(float)
    raw_returns = raw_returns.reindex(index).fillna(0.0).astype(float)
    predictions = predictions.reindex(index).astype(float)
    confidence = confidence.reindex(index).fillna(0.0).astype(float)
    ml_df = ml_df.reindex(index)

    # Align optional gating series
    if gate_mask is not None:
        gate_mask = gate_mask.reindex(index).fillna(0.0).astype(float)
    if gate_prob is not None:
        gate_prob = gate_prob.reindex(index).fillna(0.0).astype(float)

    # Determine trailing distance grid (in ATR multiples). When no explicit grid
    # is provided, fall back to a single value from trail_distance_atr_mult (or
    # 0.0 meaning no trailing).
    if trail_distance_atr_mult_values is not None:
        trail_grid = [float(v) for v in trail_distance_atr_mult_values] if len(trail_distance_atr_mult_values) > 0 else [0.0]
    else:
        if trail_distance_atr_mult is not None:
            trail_grid = [float(trail_distance_atr_mult)]
        else:
            # Default trailing grid aligned with meta_labeling_hpo_experiment search
            # space for trail_distance ~0.6–1.2x ATR, plus an explicit no-trailing
            # configuration.
            trail_grid = [0.0, 0.6, 0.9, 1.2]

    use_trailing_global = False
    atr_series = None
    if any((v is not None) and (v > 0.0) for v in trail_grid):
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        lookback = int(trail_atr_lookback) if trail_atr_lookback is not None else 14
        if lookback < 2:
            lookback = 2
        atr_series = true_range.rolling(window=lookback, min_periods=1).mean()
        atr_series = atr_series.reindex(index).astype(float)
        use_trailing_global = True

    regimes = None
    unique_regimes: Optional[List[Any]] = None
    if regime_col is not None and regime_col in ml_df.columns:
        regimes = ml_df[regime_col]
        try:
            non_null_regimes = regimes.dropna()
            if len(non_null_regimes) > 0:
                unique_vals = np.unique(non_null_regimes.to_numpy())
                unique_regimes = list(unique_vals)
        except Exception:
            regimes = None
            unique_regimes = None

    # Profit-take grid: 0.7%–2.2% in 0.1% steps. This extends the previous
    # 0.7%–1.2% range so that grid backtests can cover the higher profit
    # levels explored by meta_labeling_hpo_experiment (profit_thr_base
    # search space ~0.8%–2.2%).
    if tp_values is None:
        tp_values = [v / 10000.0 for v in range(70, 230, 10)]
    if sl_values is None:
        sl_values = [v / 10000.0 for v in range(30, 70, 10)]
    # Use quantile-based confidence thresholds so that we always work with the
    # top X% of analyst confidence rather than fixed absolute cutoffs. We map:
    #   0.5 -> top 50%, 0.6 -> top 40%, 0.7 -> top 30%, 0.8 -> top 20%,
    #   0.9 -> top 10%, 0.95 -> top 5%.
    long_mask = predictions > 0.0
    if bool(long_mask.any()):
        conf_for_quantiles = confidence[long_mask]
    else:
        conf_for_quantiles = confidence
    quantile_levels = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    conf_quantiles = {q: float(conf_for_quantiles.quantile(q)) for q in quantile_levels}

    rows: List[Dict[str, Any]] = []

    for tp in tp_values:
        for sl in sl_values:
            # Allow risk-reward ratios starting at 1:1 (TP >= SL) instead of
            # enforcing a minimum of 2:1. Only skip configurations where
            # TP < SL (reward smaller than risk).
            if tp < sl:
                continue
            for trail_mult in trail_grid:
                for q in quantile_levels:
                    th_value = conf_quantiles[q]
                    top_share = (1.0 - q) * 100.0
                    # ------------------------------------------------------------------
                    # Entry logic: go long when analyst confidence is above the
                    # quantile-based threshold (top X% of confidence values) AND the
                    # Analyst prediction is bullish (prediction > 0). Optionally
                    # require gate_mask/gate_prob to allow entries. Gating only
                    # affects new trade openings.
                    # ------------------------------------------------------------------
                    long_signal = (confidence >= th_value) & (predictions > 0.0)
                    if gate_mask is not None:
                        long_signal = long_signal & (gate_mask > 0.0)
                    if gate_prob is not None:
                        try:
                            prob_th = float(gate_prob_threshold)
                        except Exception:
                            prob_th = 0.5
                        long_signal = long_signal & (gate_prob >= prob_th)

                    n = len(index)
                    position = pd.Series(0.0, index=index, dtype=float)
                    strategy_returns_wo_fees = pd.Series(0.0, index=index, dtype=float)

                    in_position = False
                    cum_factor = 1.0  # cumulative (1+return) from current trade entry
                    entry_price = 0.0
                    bars_in_trade = 0  # number of bars spent in the current trade (including entry bar)
                    trailing_active = False
                    event_use_trailing = False
                    event_trail_dist = 0.0
                    peak_price = 0.0

                    # Exit-reason counters (per completed trade)
                    exit_profit = 0
                    exit_stop = 0
                    exit_trailing = 0
                    exit_max_hold = 0
                    exit_conflict = 0
                    exit_end_of_series = 0

                    # We use t-1's signal to decide entry at bar t to avoid lookahead.
                    for i in range(1, n):
                        if (not in_position) and bool(long_signal.iloc[i - 1]):
                            # Check previous bar's signal for entry
                            in_position = True
                            cum_factor = 1.0
                            # Use the previous close as the effective entry price so that
                            # PnL starts from the entry bar using raw_returns[i].
                            entry_price = float(close.iloc[i - 1])
                            bars_in_trade = 0
                            trailing_active = False
                            event_use_trailing = False
                            event_trail_dist = 0.0
                            peak_price = entry_price
                            if use_trailing_global and atr_series is not None and trail_mult is not None and trail_mult > 0.0:
                                atr_value = float(atr_series.iloc[i - 1])
                                if np.isfinite(atr_value) and atr_value > 0.0:
                                    event_trail_dist = atr_value * float(trail_mult)
                                    event_use_trailing = True

                        if in_position:
                            # We start this bar in a long position
                            position.iloc[i] = 1.0
                            bars_in_trade += 1
                            bar_ret = float(raw_returns.iloc[i])

                            # Proposed cumulative factor using close-to-close returns
                            proposed_factor = cum_factor * (1.0 + bar_ret)

                            if entry_price != 0.0:
                                high_price = float(high.iloc[i])
                                low_price = float(low.iloc[i])
                                high_ret = float(high_price / entry_price - 1.0)
                                low_ret = float(low_price / entry_price - 1.0)
                            else:
                                high_price = float(high.iloc[i])
                                low_price = float(low.iloc[i])
                                high_ret = 0.0
                                low_ret = 0.0

                            exit_now = False
                            target_ret = 0.0
                            exit_reason = ""

                            if event_use_trailing:
                                fixed_stop_price = entry_price * (1.0 - sl)
                                effective_stop = fixed_stop_price
                                if trailing_active:
                                    if high_price > peak_price:
                                        peak_price = high_price
                                    current_trailing_stop = peak_price - event_trail_dist
                                    if current_trailing_stop > effective_stop:
                                        effective_stop = current_trailing_stop
                                    min_profit_price_long = entry_price * (1.0 + tp)
                                    if min_profit_price_long > effective_stop:
                                        effective_stop = min_profit_price_long

                                if trailing_active and low_price <= effective_stop:
                                    exit_now = True
                                    if entry_price > 0.0:
                                        target_ret = float(effective_stop / entry_price - 1.0)
                                    else:
                                        target_ret = 0.0
                                    if effective_stop > fixed_stop_price + 1e-8:
                                        exit_reason = "trailing"
                                    else:
                                        exit_reason = "stop"

                                if (not trailing_active) and (not exit_now):
                                    activation_price = entry_price * (1.0 + tp)
                                    if high_price >= activation_price:
                                        peak_price = high_price
                                        intra_bar_stop = peak_price - event_trail_dist
                                        eff_intra_stop = fixed_stop_price
                                        if intra_bar_stop > eff_intra_stop:
                                            eff_intra_stop = intra_bar_stop
                                        if low_price <= eff_intra_stop:
                                            raw_exit_price = 0.5 * (high_price + low_price)
                                            min_profit_price_long = entry_price * (1.0 + tp)
                                            exit_price = raw_exit_price
                                            if exit_price < min_profit_price_long:
                                                exit_price = min_profit_price_long
                                            exit_now = True
                                            if entry_price > 0.0:
                                                target_ret = float(exit_price / entry_price - 1.0)
                                            else:
                                                target_ret = 0.0
                                            exit_reason = "trailing"
                                        else:
                                            trailing_active = True
                            else:
                                # TP/SL detection based on high/low relative to entry close
                                # If both TP and SL are hit in the same bar, treat it as a
                                # neutral outcome (flat PnL) rather than forcing SL or TP.
                                if low_ret <= -sl and high_ret >= tp:
                                    exit_now = True
                                    target_ret = 0.0
                                    exit_reason = "conflict"
                                elif high_ret >= tp:
                                    exit_now = True
                                    target_ret = tp
                                    exit_reason = "profit"
                                elif low_ret <= -sl:
                                    exit_now = True
                                    target_ret = -sl
                                    exit_reason = "stop"

                        # Enforce a maximum holding period measured in bars.
                        if (not exit_now) and max_holding_bars > 0 and bars_in_trade >= max_holding_bars:
                            exit_now = True

                            # Horizon exit semantics: close at this bar's close, but if the
                            # close has moved beyond the fixed stop level, use the midpoint
                            # between the close and the stop. This mirrors the labeling
                            # logic which avoids synthetic losses far beyond the nominal
                            # stop level when trades dwell until the horizon.
                            if entry_price > 0.0:
                                final_close = float(close.iloc[i])
                                fixed_stop_price = entry_price * (1.0 - sl)
                                if final_close < fixed_stop_price:
                                    exit_price = 0.5 * (final_close + fixed_stop_price)
                                else:
                                    exit_price = final_close
                                target_ret = float(exit_price / entry_price - 1.0)
                            else:
                                # Fallback: preserve legacy behavior when entry_price is invalid.
                                target_ret = proposed_factor - 1.0

                            exit_reason = "max_hold"

                        if exit_now:
                            # Adjust this bar's return so that the cumulative
                            # trade return equals exactly the TP/SL (or max-hold)
                            # threshold.
                            if cum_factor > 0.0:
                                adjusted_bar_ret = (1.0 + target_ret) / cum_factor - 1.0
                            else:
                                adjusted_bar_ret = target_ret
                            strategy_returns_wo_fees.iloc[i] = adjusted_bar_ret

                            # Mark flat position on this bar so that the
                            # position series reflects the actual exit.
                            position.iloc[i] = 0.0

                            # Update exit-reason counters at trade close.
                            if exit_reason == "profit":
                                exit_profit += 1
                            elif exit_reason == "stop":
                                exit_stop += 1
                            elif exit_reason == "trailing":
                                exit_trailing += 1
                            elif exit_reason == "max_hold":
                                exit_max_hold += 1
                            elif exit_reason == "conflict":
                                exit_conflict += 1

                            # Close the position; do not re-open on the same bar.
                            in_position = False
                            cum_factor = 1.0
                            entry_price = 0.0
                            bars_in_trade = 0
                            trailing_active = False
                            event_use_trailing = False
                            event_trail_dist = 0.0
                            peak_price = 0.0
                        else:
                            # Continue the trade with the raw close-to-close return
                            strategy_returns_wo_fees.iloc[i] = bar_ret
                            cum_factor = proposed_factor
                    else:
                        position.iloc[i] = 0.0

                # Calculate turnover only for actual position changes (entries/exits)
                position_changes = position.diff().abs().fillna(0.0)
                turnover = position_changes * ((position != 0) | (position.shift(1) != 0)).astype(float)
                per_side_fee_rate = fee_rate * 0.5
                fee_returns = turnover * per_side_fee_rate
                strategy_returns_with_fees = strategy_returns_wo_fees - fee_returns

                metrics_wo = _compute_basic_performance_metrics(strategy_returns_wo_fees, timeframe)
                metrics_with = _compute_basic_performance_metrics(strategy_returns_with_fees, timeframe)

                equity_wo = (1.0 + strategy_returns_wo_fees).cumprod()
                equity_with = (1.0 + strategy_returns_with_fees).cumprod()
                equity_final_wo = float(equity_wo.iloc[-1]) if metrics_wo["bars"] > 0 else 1.0
                equity_final_with = float(equity_with.iloc[-1]) if metrics_with["bars"] > 0 else 1.0

                period_returns_with = _compute_period_returns(strategy_returns_with_fees, periods=5)
                period_std_with = float(np.std(period_returns_with)) if len(period_returns_with) > 0 else 0.0

                # Approximate duration in days based on timeframe
                bars = metrics_wo["bars"]
                bars_per_year = _bars_per_year_from_timeframe(timeframe)
                bars_per_day = bars_per_year / 365.0 if bars_per_year > 0 else 0.0
                duration_days = float(bars / bars_per_day) if bars_per_day > 0 else 0.0

                # Coefficient of variation: lower is more stable
                mean_period_return = float(np.mean(period_returns_with)) if len(period_returns_with) > 0 else 0.0
                if abs(mean_period_return) > 1e-9 and period_std_with > 0:
                    coefficient_of_variation = abs(period_std_with / mean_period_return)
                else:
                    coefficient_of_variation = 0.0

                # Profit stability metric: higher is better. Defined as the
                # inverse of the coefficient of variation when available.
                if coefficient_of_variation > 0.0:
                    profit_stability = float(1.0 / coefficient_of_variation)
                else:
                    profit_stability = 0.0

                # Average per-day return (geometric), based on total_return_with_fees and duration_days
                if duration_days > 0:
                    try:
                        total_ret_raw = metrics_with["total_return"]
                        avg_daily_return_raw = (1.0 + total_ret_raw) ** (1.0 / duration_days) - 1.0
                    except Exception:
                        avg_daily_return_raw = 0.0
                else:
                    avg_daily_return_raw = 0.0
                # ------------------------------------------------------------------
                # Trade-level statistics (completed trades only)
                # ------------------------------------------------------------------
                in_position = (position != 0).astype(int)
                pos_changes_int = in_position.diff().fillna(0)
                entries = pos_changes_int == 1
                exits = pos_changes_int == -1

                trade_durations: List[int] = []
                trade_returns_wo: List[float] = []
                trade_returns_with: List[float] = []

                entry_idx: Optional[int] = None
                for idx in range(len(in_position)):
                    if entries.iloc[idx]:
                        entry_idx = idx
                    elif exits.iloc[idx] and entry_idx is not None:
                        trade_slice = slice(entry_idx, idx + 1)
                        seg_wo = strategy_returns_wo_fees.iloc[trade_slice]
                        seg_with = strategy_returns_with_fees.iloc[trade_slice]
                        tr_wo = float((1.0 + seg_wo).prod() - 1.0)
                        tr_with = float((1.0 + seg_with).prod() - 1.0)
                        trade_durations.append(idx - entry_idx + 1)
                        trade_returns_wo.append(tr_wo)
                        trade_returns_with.append(tr_with)
                        entry_idx = None

                # If a trade is still open at the end of the series, treat it as
                # closed at the final bar (mark-to-market) so that trade-level
                # statistics remain consistent with bar-level PnL.
                if entry_idx is not None:
                    trade_slice = slice(entry_idx, len(in_position))
                    seg_wo = strategy_returns_wo_fees.iloc[trade_slice]
                    seg_with = strategy_returns_with_fees.iloc[trade_slice]
                    tr_wo = float((1.0 + seg_wo).prod() - 1.0)
                    tr_with = float((1.0 + seg_with).prod() - 1.0)
                    trade_durations.append(len(in_position) - entry_idx)
                    trade_returns_wo.append(tr_wo)
                    trade_returns_with.append(tr_with)
                    exit_end_of_series += 1

                n_trades = len(trade_durations)
                if n_trades > 0:
                    avg_trade_duration_bars = float(np.mean(trade_durations))
                else:
                    avg_trade_duration_bars = 0.0

                # Convert average trade duration to days
                bars_per_day = bars_per_year / 365.0 if bars_per_year > 0 else 0.0
                avg_trade_duration_days = float(avg_trade_duration_bars / bars_per_day) if bars_per_day > 0 else 0.0

                # Trade-level win rate and risk/reward (based on trade PnL)
                if n_trades > 0:
                    trade_returns_wo_arr = np.array(trade_returns_wo)
                    wins_mask = trade_returns_wo_arr > 0.0
                    n_wins = int(wins_mask.sum())
                    n_losses = n_trades - n_wins
                    win_rate_trades = float(n_wins / n_trades) if n_trades > 0 else 0.0

                    if n_wins > 0:
                        avg_win_trade = float(trade_returns_wo_arr[wins_mask].mean())
                    else:
                        avg_win_trade = 0.0
                    if n_losses > 0:
                        avg_loss_trade = float(trade_returns_wo_arr[~wins_mask].mean())
                    else:
                        avg_loss_trade = 0.0

                    trade_returns_with_arr = np.array(trade_returns_with)
                    wins_with_mask = trade_returns_with_arr > 0.0
                    n_wins_with = int(wins_with_mask.sum())
                    n_losses_with = n_trades - n_wins_with
                    win_rate_trades_with = float(n_wins_with / n_trades) if n_trades > 0 else 0.0
                    if n_wins_with > 0:
                        avg_win_trade_with = float(trade_returns_with_arr[wins_with_mask].mean())
                    else:
                        avg_win_trade_with = 0.0
                    if n_losses_with > 0:
                        avg_loss_trade_with = float(trade_returns_with_arr[~wins_with_mask].mean())
                    else:
                        avg_loss_trade_with = 0.0

                    # Mean trade return (with/without fees) so we can inspect
                    # average PnL per completed trade for this configuration
                    avg_trade_return_wo = float(trade_returns_wo_arr.mean())
                    avg_trade_return_with = float(trade_returns_with_arr.mean())
                else:
                    win_rate_trades = 0.0
                    win_rate_trades_with = 0.0
                    avg_win_trade = 0.0
                    avg_loss_trade = 0.0
                    avg_win_trade_with = 0.0
                    avg_loss_trade_with = 0.0
                    avg_trade_return_wo = 0.0
                    avg_trade_return_with = 0.0

                rr_wo = 0.0
                if avg_loss_trade < 0.0 and abs(avg_loss_trade) > 0.0:
                    rr_wo = float(avg_win_trade / abs(avg_loss_trade))

                rr_with = 0.0
                if avg_loss_trade_with < 0.0 and abs(avg_loss_trade_with) > 0.0:
                    rr_with = float(avg_win_trade_with / abs(avg_loss_trade_with))

                # Trade-level accuracy: same as trade win rate
                acc = win_rate_trades

                # Calculate Calmar ratio (annualized return / abs(max drawdown))
                calmar_wo = 0.0
                if abs(metrics_wo["max_drawdown"]) > 1e-9:
                    calmar_wo = float(metrics_wo["annualized_return"] / abs(metrics_wo["max_drawdown"]))

                calmar_with = 0.0
                if abs(metrics_with["max_drawdown"]) > 1e-9:
                    calmar_with = float(metrics_with["annualized_return"] / abs(metrics_with["max_drawdown"]))

                # Average trade opportunities per day for this configuration
                avg_trades_per_day = float(n_trades / duration_days) if duration_days > 0 else 0.0

                # Directional accuracy on raw returns while in position:
                # - 1-bar: share of bars where raw return > 0 with position != 0
                # - 2/3/4-bar: share of starting bars where the cumulative
                #   raw return over the next h bars is positive.
                dir_acc_1 = 0.0
                dir_acc_2 = 0.0
                dir_acc_3 = 0.0
                dir_acc_4 = 0.0
                try:
                    in_pos_mask = position != 0.0
                    if bool(in_pos_mask.any()):
                        rr = raw_returns.reindex(position.index).to_numpy()
                        idx_arr = np.where(in_pos_mask.to_numpy())[0]
                        n_rr = len(rr)

                        # 1-bar directional accuracy
                        dir_acc_1 = float((rr[idx_arr] > 0.0).mean()) if idx_arr.size > 0 else 0.0

                        # Precompute equity curve for sliding multi-bar horizons
                        equity_curve = np.cumprod(1.0 + rr)

                        def _horizon_dir_acc(h: int) -> float:
                            if h <= 0 or n_rr == 0:
                                return 0.0
                            valid = idx_arr[idx_arr + h - 1 < n_rr]
                            if valid.size == 0:
                                return 0.0
                            end_idx = valid + h - 1
                            start_equity = np.where(valid > 0, equity_curve[valid - 1], 1.0)
                            end_equity = equity_curve[end_idx]
                            cum_ret = end_equity / start_equity - 1.0
                            return float((cum_ret > 0.0).mean())

                        dir_acc_2 = _horizon_dir_acc(2)
                        dir_acc_3 = _horizon_dir_acc(3)
                        dir_acc_4 = _horizon_dir_acc(4)
                except Exception:
                    dir_acc_1 = 0.0
                    dir_acc_2 = 0.0
                    dir_acc_3 = 0.0
                    dir_acc_4 = 0.0

                trail_label = "off"
                trailing_enabled = bool(use_trailing_global and trail_mult is not None and trail_mult > 0.0)
                if trailing_enabled:
                    trail_label = f"{trail_mult:.2f}xATR"

                row: Dict[str, Any] = {
                    "grid_config": f"tp={tp * 100:.3f}%,sl={sl * 100:.3f}%,conf_top={top_share:.0f}%,trail={trail_label}",
                    "take_profit_pct": tp,
                    "stop_loss_pct": sl,
                    "trail_distance_atr_mult": float(trail_mult) if trail_mult is not None else 0.0,
                    "trailing_enabled": trailing_enabled,
                    "exit_trades_profit": exit_profit,
                    "exit_trades_stop": exit_stop,
                    "exit_trades_trailing": exit_trailing,
                    "exit_trades_max_hold": exit_max_hold,
                    "exit_trades_conflict": exit_conflict,
                    "exit_trades_end_of_series": exit_end_of_series,
                    "confidence_threshold": th_value,
                    "confidence_quantile": q,
                    "bars": metrics_wo["bars"],
                    "duration_days": duration_days,
                    # Express total returns and final equity as percentages
                    "strategy_total_return_without_fees_%": metrics_wo["total_return"] * 100.0,
                    "strategy_total_return_with_fees_%": metrics_with["total_return"] * 100.0,
                    "equity_final_without_fees_%": equity_final_wo * 100.0,
                    "equity_final_with_fees_%": equity_final_with * 100.0,
                    # Geometric average daily return (percentage per day)
                    "avg_daily_return_with_fees_%": avg_daily_return_raw * 100.0,
                    "max_drawdown_with_fees": metrics_with["max_drawdown"],
                    "accuracy": acc,
                    "directional_accuracy": dir_acc_1,
                    "directional_accuracy_2bar": dir_acc_2,
                    "directional_accuracy_3bar": dir_acc_3,
                    "directional_accuracy_4bar": dir_acc_4,
                    "risk_reward_ratio_without_fees": rr_wo,
                    "risk_reward_ratio_with_fees": rr_with,
                    "sharpe_ratio_without_fees": metrics_wo["sharpe_ratio"],
                    "sharpe_ratio_with_fees": metrics_with["sharpe_ratio"],
                    "sortino_ratio_without_fees": metrics_wo["sortino_ratio"],
                    "sortino_ratio_with_fees": metrics_with["sortino_ratio"],
                    "calmar_ratio_without_fees": calmar_wo,
                    "calmar_ratio_with_fees": calmar_with,
                    # Trade-level win rate (separate for with/without fees)
                    "win_rate_without_fees": win_rate_trades,
                    "win_rate_with_fees": win_rate_trades_with,
                    "profit_factor_without_fees": metrics_wo["profit_factor"],
                    "profit_factor_with_fees": metrics_with["profit_factor"],
                    "coefficient_of_variation_with_fees": coefficient_of_variation,
                    "profit_stability_with_fees": profit_stability,
                    "number_of_trades": n_trades,
                    "avg_trade_duration_bars": avg_trade_duration_bars,
                    "avg_trade_duration_days": avg_trade_duration_days,
                    "avg_trades_per_day": avg_trades_per_day,
                    # Mean completed-trade return for this bucket/config
                    "avg_trade_return_without_fees_%": avg_trade_return_wo * 100.0,
                    "avg_trade_return_with_fees_%": avg_trade_return_with * 100.0,
                }

                if regimes is not None and unique_regimes is not None and metrics_wo["bars"] > 0:
                    try:
                        nonzero_mask = position != 0.0
                        if bool(nonzero_mask.any()):
                            for reg in unique_regimes:
                                reg_mask = nonzero_mask & (regimes == reg)
                                if bool(reg_mask.any()):
                                    # Use raw returns for regime accuracy, not clipped returns
                                    acc_reg = float((raw_returns[reg_mask] > 0.0).mean())
                                else:
                                    acc_reg = 0.0
                                row[f"accuracy_regime_{reg}"] = acc_reg
                    except Exception:
                        pass
                rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["grid_config"])

    return pd.DataFrame(rows)


def run_simple_short_grid_backtest(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    raw_returns: pd.Series,
    predictions: pd.Series,
    confidence: pd.Series,
    ml_df: pd.DataFrame,
    timeframe: str,
    fee_rate: float = 0.0015,
    regime_col: Optional[str] = None,
    max_holding_bars: int = 6,
    tp_values: Optional[List[float]] = None,
    sl_values: Optional[List[float]] = None,
    trail_distance_atr_mult: Optional[float] = None,
    trail_atr_lookback: int = 14,
    gate_mask: Optional[pd.Series] = None,
    gate_prob: Optional[pd.Series] = None,
    gate_prob_threshold: float = 0.5,
) -> pd.DataFrame:
    index = close.index
    close = close.reindex(index).astype(float)
    high = high.reindex(index).astype(float)
    low = low.reindex(index).astype(float)
    raw_returns = raw_returns.reindex(index).fillna(0.0).astype(float)
    predictions = predictions.reindex(index).astype(float)
    confidence = confidence.reindex(index).fillna(0.0).astype(float)
    ml_df = ml_df.reindex(index)

    close_inv = 1.0 / close.replace(0.0, np.nan)
    high_inv = 1.0 / low.replace(0.0, np.nan)
    low_inv = 1.0 / high.replace(0.0, np.nan)

    close_inv = close_inv.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    high_inv = high_inv.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    low_inv = low_inv.replace([np.inf, -np.inf], np.nan).ffill().bfill()

    raw_returns_inv = close_inv.pct_change().fillna(0.0)

    predictions_long = -predictions

    return run_simple_long_grid_backtest(
        close=close_inv,
        high=high_inv,
        low=low_inv,
        raw_returns=raw_returns_inv,
        predictions=predictions_long,
        confidence=confidence,
        ml_df=ml_df,
        timeframe=timeframe,
        fee_rate=fee_rate,
        regime_col=regime_col,
        max_holding_bars=max_holding_bars,
        tp_values=tp_values,
        sl_values=sl_values,
        trail_distance_atr_mult=trail_distance_atr_mult,
        trail_atr_lookback=trail_atr_lookback,
        gate_mask=gate_mask,
        gate_prob=gate_prob,
        gate_prob_threshold=gate_prob_threshold,
    )
