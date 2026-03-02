import re

with open("extreme_price_movements/training.py", "r") as f:
    content = f.read()

search_15m_false = """    # 15m precision is NOT used in the optimizer — the build_event_cache_15m
    # requires a contiguous 15m array aligned 4:1 with the full 1h panel,
    # which would require downloading 90 days of 15m data per symbol (~400 symbols).
    # 1h resolution is sufficient for grid search; 15m is used in backtest engine per-trade.
    use_15m = False
    exchange = None"""

replace_15m_true = """    # We enable 15m precision for the optimizer as requested.
    use_15m = True
    exchange = None
    try:
        import ccxt
        exchange = ccxt.binance({
            'enableRateLimit': True,
        })
    except ImportError:
        tprint("WARNING: ccxt not installed. Cannot download 15m data for optimizer.")
        use_15m = False"""

if search_15m_false in content:
    content = content.replace(search_15m_false, replace_15m_true)
    print("Replaced use_15m = False")
else:
    print("Could not find use_15m = False")

with open("extreme_price_movements/training.py", "w") as f:
    f.write(content)
