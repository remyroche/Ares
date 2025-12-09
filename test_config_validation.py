from live_trading.config import TradingConfig

def test_validation():
    # Test default valid config
    try:
        config = TradingConfig()
        config.validate()
        print("PASS: Default config valid")
    except Exception as e:
        print(f"FAIL: Default config invalid: {e}")

    # Test invalid values (e.g. negative leverage) - expect clamping
    try:
        config = TradingConfig(max_leverage=-1)
        config.validate()
        if config.max_leverage == 5.0:
            print("PASS: Auto-corrected negative leverage to min (5.0)")
        else:
            print(f"FAIL: Leverage handling unexpected: {config.max_leverage}")
    except ValueError as e:
        print(f"FAIL: Caught unexpected error: {e}")

if __name__ == "__main__":
    test_validation()
