import pandas as pd
import pandas_ta as ta

# Create a simple DataFrame

df = pd.DataFrame(
    {
        "open": [1, 2, 3, 4, 5],
        "high": [2, 3, 4, 5, 6],
        "low": [0, 1, 2, 3, 4],
        "close": [1, 2, 3, 4, 5],
        "volume": [100, 110, 120, 130, 140],
    },
)

print("DataFrame columns:", df.columns)

# Try to use the .ta accessor
try:
    print("Testing df.ta.sma...")
    sma = df.ta.sma(length=3)
    print("SMA result:")
    print(sma)
except Exception as e:
    print("Error using df.ta.sma:", e)

# Try to use ta.sma directly
try:
    print("Testing ta.sma...")
    sma_direct = ta.sma(df["close"], length=3)
    print("Direct SMA result:")
    print(sma_direct)
except Exception as e:
    print("Error using ta.sma:", e)
