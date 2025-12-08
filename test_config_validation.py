from live_trading.config import TradingConfig

def test_validation():
    try:
        config = TradingConfig(direction="invalid")
        config.validate()
        print("FAIL: Accepted invalid direction")
    except ValueError as e:
        print(f"PASS: Caught invalid direction: {e}")

    try:
        config = TradingConfig(direction="long")
        config.validate()
        print("PASS: Accepted 'long'")
    except ValueError as e:
        print(f"FAIL: Rejected 'long': {e}")

    try:
        config = TradingConfig(direction="both")
        config.validate()
        print("PASS: Accepted 'both'")
    except ValueError as e:
        print(f"FAIL: Rejected 'both': {e}")

if __name__ == "__main__":
    test_validation()
