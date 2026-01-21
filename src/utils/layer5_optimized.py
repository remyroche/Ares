import numpy as np
from numba import njit, prange

@njit(cache=True)
def apply_position_sizing_numba(
    probs,
    dampening,
    threshold,
    kelly_fraction,
    steepness,
    dampening_mult
):
    """
    Vectorized position sizing calculation.
    """
    n = len(probs)
    sizes = np.zeros(n, dtype=np.float64)

    for i in range(n):
        p = probs[i]
        d = dampening[i]

        if p < threshold:
            sizes[i] = 0.0
            continue

        # 1. Ramp
        denom = 1.0 - threshold
        if denom < 1e-9:
            denom = 1e-9

        relative_conf = (p - threshold) / denom
        if relative_conf < 0: relative_conf = 0.0
        if relative_conf > 1: relative_conf = 1.0

        curve = relative_conf ** steepness

        # 2. Dampening
        # final_dampening = 1 means model is certain (d=0); 0 means uncertain (d=1)
        # The user said: "confidence_shield = 1.0 - (dampening * dampening_mult)"
        # Assuming 'dampening' input is 0..1 (1=High Uncertainty)
        shield = 1.0 - (d * dampening_mult)
        if shield < 0.0:
            shield = 0.0

        # 3. Final Size
        raw_size = curve * kelly_fraction * shield

        # Clip
        if raw_size > kelly_fraction:
            raw_size = kelly_fraction
        if raw_size < 0.0:
            raw_size = 0.0

        sizes[i] = raw_size

    return sizes

@njit(cache=True)
def run_atr_backtest_numba(
    prices_close,
    prices_high,
    prices_low,
    atr,
    sizes,
    sl_atr_mult,
    trail_trigger_mult,
    trail_dist_mult,
    initial_balance=100000.0,
    fee_rate=0.001
):
    """
    Simulates trades using ATR-based SL/Trailing Profit.
    Returns:
        equity_curve: array of balance over time
        trades: (entry_idx, exit_idx, entry_price, exit_price, pnl, size)
    """
    n = len(prices_close)
    equity_curve = np.zeros(n, dtype=np.float64)

    # We'll store trades in a fixed size array and slice later, or use list
    # Numba lists are a bit slower, let's use a large array
    max_trades = n # worst case
    trades = np.zeros((max_trades, 6), dtype=np.float64) # entry_idx, exit_idx, entry_price, exit_price, pnl, size
    trade_count = 0

    balance = initial_balance

    in_position = False
    entry_price = 0.0
    pos_size_units = 0.0
    stop_loss = 0.0
    high_seen = 0.0
    entry_idx = -1

    for i in range(n):
        # Update balance for non-trade steps (just carry over)
        # We only update balance on close

        current_close = prices_close[i]
        current_high = prices_high[i]
        current_low = prices_low[i]
        current_atr = atr[i]

        if not in_position:
            # Check for entry
            size_alloc = sizes[i]

            if size_alloc > 1e-6:
                # Enter
                in_position = True
                entry_idx = i
                entry_price = current_close

                # Calculate position size in units
                # Cost = balance * size_alloc
                cost = balance * size_alloc
                pos_size_units = cost / entry_price

                # Deduct Principal AND Fee
                balance -= cost
                balance -= cost * fee_rate

                # Initial Stop Loss
                stop_loss = entry_price - (current_atr * sl_atr_mult)
                high_seen = entry_price

        else:
            # Update High Seen
            if current_high > high_seen:
                high_seen = current_high

            # Trailing Logic
            # If price moved up beyond trigger
            if (high_seen - entry_price) > (current_atr * trail_trigger_mult):
                # Activate trail
                potential_sl = high_seen - (current_atr * trail_dist_mult)
                if potential_sl > stop_loss:
                    stop_loss = potential_sl

            # Check Exit
            # Assume exit at stop_loss if low <= stop_loss
            if current_low <= stop_loss:
                # Exited
                exit_price = stop_loss
                # Slippage? Ignoring for now or covered by fee/stop logic

                gross_pnl = (exit_price - entry_price) * pos_size_units
                sale_value = exit_price * pos_size_units

                # Apply fee on exit
                fee = sale_value * fee_rate

                # Credit balance (Sale proceeds - Fee)
                balance += (sale_value - fee)

                # Record trade
                # entry_idx, exit_idx, entry_price, exit_price, pnl, size_alloc
                trades[trade_count, 0] = entry_idx
                trades[trade_count, 1] = i
                trades[trade_count, 2] = entry_price
                trades[trade_count, 3] = exit_price

                # Net PnL = (Sale - Fee) - (Cost + Fee)
                # But Cost + Fee was deducted from balance earlier.
                # Sale - Fee is added now.
                # So PnL = Balance_post - Balance_pre
                # But we don't track Balance_pre easily here for the trade row calculation.
                # Calculate explicitly:
                # CostBasis = Units * EntryPrice
                # EntryFee = CostBasis * FeeRate
                # ExitValue = Units * ExitPrice
                # ExitFee = ExitValue * FeeRate
                # PnL = ExitValue - ExitFee - CostBasis - EntryFee

                cost_basis = pos_size_units * entry_price
                entry_fee = cost_basis * fee_rate
                exit_value = pos_size_units * exit_price
                exit_fee = exit_value * fee_rate

                net_pnl = exit_value - exit_fee - cost_basis - entry_fee

                trades[trade_count, 4] = net_pnl
                trades[trade_count, 5] = sizes[entry_idx]

                trade_count += 1
                in_position = False
                pos_size_units = 0.0

            else:
                # Still in position
                pass

        # Calculate Equity
        if in_position:
            # Equity = Balance (Cash) + Position Value
            # Position Value = Units * Close
            equity_curve[i] = balance + (pos_size_units * current_close)
        else:
            equity_curve[i] = balance

    return equity_curve, trades[:trade_count]
