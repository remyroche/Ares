import enum
from typing import Tuple

# =============================================================================
# Canonical Outcome Semantics
# =============================================================================

class ExitReason(str, enum.Enum):
    STOP_LOSS = "stop_loss"                 # Static or Break-Even Stop Hit
    TRAILING_STOP = "trailing_stop"         # Trailing Stop Hit (Profit locked)
    TAKE_PROFIT = "take_profit"             # Hard TP hit (rarely used in engine, used in Numba)
    TIME_EXIT = "time_exit"                 # Reached max_hold_bars/hours without hitting barriers
    GIVEBACK_EXIT = "giveback_exit"         # Retracement from MFE exceeded giveback_pct
    EARLY_INVALIDATION = "early_invalidation" # Adversely drifted over time without MFE
    NO_ENTRY = "no_entry"                   # Could not evaluate entry (e.g., NaN price)
    NO_PATH = "no_path"                     # No forward path data available
    LIMIT_NOT_FILLED = "limit_not_filled"   # Limit order entry was never filled

def map_numba_exit_code_to_canonical(code: int) -> str:
    """Map Numba simulate_trade_exit integer code to canonical string."""
    mapping = {
        0: ExitReason.TAKE_PROFIT.value,
        1: ExitReason.STOP_LOSS.value,
        2: ExitReason.TRAILING_STOP.value,
        3: ExitReason.TIME_EXIT.value
    }
    return mapping.get(code, "unknown")

def map_engine_exit_reason_to_canonical(reason: str) -> str:
    """Ensure engine string maps to the canonical enum value."""
    try:
        return ExitReason(reason).value
    except ValueError:
        return reason

def map_tbm_label_to_canonical_training_semantic(label: int) -> str:
    """
    Map TBM label integer to training semantic concept.

    TBM Labels vs Execution:
    - 0 (Adverse): The path hit the adverse MAE barrier before the MFE barrier.
                   Maps cleanly to STOP_LOSS.
    - 1 (Neutral): The path hit neither barrier before the horizon.
                   Maps cleanly to TIME_EXIT.
    - 2 (Favorable): The path hit the favorable MFE barrier before the MAE barrier.
                     IMPORTANT: In execution, this usually means "Trail Activation Reached",
                     not a realized "Take Profit Exit". The engine ratchets stops rather than
                     exiting at a hard TP.
    """
    mapping = {
        0: "ADVERSE_BARRIER_FIRST",
        1: "NEITHER_BARRIER_HIT",
        2: "FAVORABLE_BARRIER_FIRST_OR_TRAIL_ACTIVATION"
    }
    return mapping.get(label, "unknown")


# =============================================================================
# Execution Pricing Helpers
# =============================================================================

def resolve_limit_fill(
    open_price: float,
    high_price: float,
    low_price: float,
    limit_price: float,
    is_long: bool
) -> Tuple[bool, float]:
    """
    Resolve a limit order fill correctly, without optimistic extreme-pricing.

    A limit order guarantees the limit price or better.
    - If the bar gaps favorably past the limit (open is better than limit),
      it fills at the open price.
    - If the bar does not gap, but the intrabar extreme reaches the limit,
      it fills EXACTLY at the limit price. It DOES NOT fill at the bar extreme.

    Args:
        open_price: Bar open price
        high_price: Bar high price
        low_price: Bar low price
        limit_price: The requested limit price
        is_long: True if buying (entry long / exit short), False if selling (entry short / exit long)

    Returns:
        (did_fill, fill_price)
    """
    if is_long:
        # Buying limit order: Fills at or below limit price
        if open_price <= limit_price:
            # Gapped down below our limit price (or opened exactly at it)
            return True, float(open_price)
        elif low_price <= limit_price:
            # Reached our limit price intraday, but didn't gap. Fill at exactly the limit.
            return True, float(limit_price)
        else:
            return False, 0.0
    else:
        # Selling limit order: Fills at or above limit price
        if open_price >= limit_price:
            # Gapped up above our limit price
            return True, float(open_price)
        elif high_price >= limit_price:
            # Reached our limit price intraday
            return True, float(limit_price)
        else:
            return False, 0.0


def resolve_stop_fill(
    open_price: float,
    high_price: float,
    low_price: float,
    stop_price: float,
    is_long: bool
) -> Tuple[bool, float]:
    """
    Resolve a stop loss fill correctly, handling gap risk.

    A stop order converts to a market order when triggered.
    - If the bar gaps adversely past the stop (open is worse than stop),
      it fills at the open price (experiencing gap slippage).
    - If the bar does not gap, but the intrabar extreme hits the stop,
      it fills EXACTLY at the stop price.

    Args:
        open_price: Bar open price
        high_price: Bar high price
        low_price: Bar low price
        stop_price: The requested stop trigger price
        is_long: True if the base position is long (selling to exit), False if short (buying to exit)

    Returns:
        (did_fill, fill_price)
    """
    if is_long:
        # Exiting a long position: Stop triggered if price drops to or below stop_price
        if open_price <= stop_price:
            # Gapped down below stop price. Fill at the open (take the gap loss).
            return True, float(open_price)
        elif low_price <= stop_price:
            # Hit the stop intraday. Fill at exactly the stop price.
            return True, float(stop_price)
        else:
            return False, 0.0
    else:
        # Exiting a short position: Stop triggered if price rises to or above stop_price
        if open_price >= stop_price:
            # Gapped up above stop price. Fill at the open (take the gap loss).
            return True, float(open_price)
        elif high_price >= stop_price:
            # Hit the stop intraday. Fill at exactly the stop price.
            return True, float(stop_price)
        else:
            return False, 0.0


# =============================================================================
# Tie-Breaker Precedence
# =============================================================================
# For same-bar multi-barrier hits in Numba/Fast simulation:
# 1. Compare absolute distance from bar Open to each triggered barrier.
# 2. Shortest distance wins (proxy for reaching it first intraday).
# 3. If distances are exactly equal, precedence is:
#    STOP_LOSS (1) > TRAILING_STOP (2) > TAKE_PROFIT (0).
# This is a deterministic proxy. It ensures worst-case outcomes take precedence
# to penalize ambiguity.
