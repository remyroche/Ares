from datetime import datetime
import numpy as np
import pandas as pd
from src.utils.warning_symbols import (
    error, warning, critical, problem, failed, invalid,
    missing, timeout, connection_error, validation_error,
    initialization_error, execution_error
)

def debug_triple_barrier():
    """Debug the triple barrier logic with a simple example"""

    # Create a simple test dataset with known price movements
    test_data = pd.DataFrame({
        'timestamp': pd.date_range('2023-07-28 10:00:00', periods=10, freq='1min'),
        'price': [1850, 1855, 1860, 1865, 1870, 1875, 1880, 1885, 1890, 1895]
    })

    print("Test data:")
    print(test_data)
    print(f"Price range: ${test_data.price.min():.2f} to ${test_data.price.max():.2f}")
    print(f"Total movement: {((test_data.price.max() - test_data.price.min()) / test_data.price.min() * 100):.2f}%")

    # Test parameters
    target_pct = 0.4  # 0.4%
    stop_pct = 0.1    # 0.1%

    print(f"\nTesting: Target {target_pct}%, Stop {stop_pct}%")

    prices = test_data['price'].values
    timestamps = test_data['timestamp'].values

    occurrences = 0
    total_attempts = 0

    for i in range(len(test_data) - 1):
        start_price = prices[i]
        start_time = timestamps[i]

        print(f"\nStarting from price ${start_price:.2f} at {start_time}")

        # Calculate target and stop prices
        up_target = start_price * (1 + target_pct / 100)
        down_target = start_price * (1 - target_pct / 100)
        up_stop = start_price * (1 + stop_pct / 100)
        down_stop = start_price * (1 - stop_pct / 100)

        print(f"  Up target: ${up_target:.2f} (${start_price:.2f} + {target_pct}%)")
        print(f"  Down target: ${down_target:.2f} (${start_price:.2f} - {target_pct}%)")
        print(f"  Up stop: ${up_stop:.2f} (${start_price:.2f} + {stop_pct}%)")
        print(f"  Down stop: ${down_stop:.2f} (${start_price:.2f} - {stop_pct}%)")

        # Look ahead
        for j in range(i + 1, len(test_data)):
            current_price = prices[j]
            current_time = timestamps[j]

            print(f"    Checking price ${current_price:.2f} at {current_time}")

            # Check time barrier (24 hours)
            time_diff = (current_time - start_time).astype('timedelta64[s]').astype(float)
            if time_diff > 24 * 3600:
                print(f"      Time barrier hit ({time_diff/3600:.1f}h)")
                break

            # Check stop loss first
            if current_price >= up_stop:
                print(f"      STOP LOSS HIT: Price ${current_price:.2f} >= Up stop ${up_stop:.2f}")
                break
            elif current_price <= down_stop:
                print(f"      STOP LOSS HIT: Price ${current_price:.2f} <= Down stop ${down_stop:.2f}")
                break

            # Check targets
            if current_price >= up_target:
                print(f"      SUCCESS: Price ${current_price:.2f} >= Up target ${up_target:.2f}")
                occurrences += 1
                break
            elif current_price <= down_target:
                print(f"      SUCCESS: Price ${current_price:.2f} <= Down target ${down_target:.2f}")
                occurrences += 1
                break

        total_attempts += 1

    print(f"\nResults: {occurrences} successes out of {total_attempts} attempts")

if __name__ == "__main__":
    debug_triple_barrier()
