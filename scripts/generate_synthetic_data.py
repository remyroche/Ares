import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

def generate_synthetic_data(symbol="ETHUSDT", exchange="binance", interval="15m", days=1095):
    print(f"Generating synthetic data for {symbol} {interval}...")

    # Create time range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    # Frequency mapping
    freq_map = {"1m": "1min", "15m": "15min", "1h": "1H", "4h": "4H", "1d": "D"}
    freq = freq_map.get(interval, "15min")

    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    n = len(dates)

    # Generate random walk price
    price = 2000.0
    prices = []
    for _ in range(n):
        change = np.random.normal(0, 0.001)  # 0.1% vol per bar
        price = price * (1 + change)
        prices.append(price)

    closes = np.array(prices)
    opens = closes * (1 + np.random.normal(0, 0.0005, n))
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.001, n)))
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.001, n)))
    volumes = np.abs(np.random.normal(1000, 500, n))

    df = pd.DataFrame({
        'timestamp': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes,
        'quote_volume': volumes * closes,
        'trades': np.random.randint(10, 100, n),
        'taker_buy_base': volumes * 0.5,
        'taker_buy_quote': (volumes * closes) * 0.5,
        'exchange': exchange,
        'timeframe': interval
    })

    # Set index
    # df.set_index('timestamp', inplace=True)
    # Don't set index if parquet expects column, but ensure timestamp column exists

    # Save path
    base_dir = Path("historical_data")
    save_dir = base_dir / exchange.lower() / symbol.lower() / "processed" / f"{symbol.lower()}_{interval}"
    save_dir.mkdir(parents=True, exist_ok=True)

    file_path = save_dir / f"synthetic_{symbol}_{interval}.parquet"
    df.to_parquet(file_path, index=False)

    print(f"Saved {n} rows to {file_path}")

if __name__ == "__main__":
    generate_synthetic_data()
