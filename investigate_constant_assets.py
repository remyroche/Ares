import pandas as pd
import numpy as np
from extreme_price_movements.universe import refresh_margin_universe_daily, build_fetch_universe
from extreme_price_movements.config import CFG
from extreme_price_movements.data_store import PartitionedOHLCVStore as DataStore
from extreme_price_movements.utils import tprint

def main():
    tprint("Investigating constant assets...")
    
    # 1. Initialize Store and Universe
    store = DataStore(CFG["data_root"])
    mu = refresh_margin_universe_daily(None, quote="USDT")
    syms_all = build_fetch_universe(mu.symbols, CFG["market_basket"], CFG["fetch_symbols_M"])
    
    tprint(f"Total Universe: {len(syms_all)} symbols")
    
    # 2. Replicate Variance Logic
    asof = pd.Timestamp.utcnow()
    cutoff = asof - pd.Timedelta(days=30)
    
    investigation = []
    
    for s in syms_all:
        try:
            df = store.load(s, columns=["close"], start_ts=cutoff, end_ts=asof)
            if df.empty or len(df) < 10:
                investigation.append({"symbol": s, "variance": -1.0, "reason": "Empty/Too Short"})
                continue

            r = df["close"].resample("12h").last().to_numpy()
            if r.size < 3:
                investigation.append({"symbol": s, "variance": -1.0, "reason": "Insuff. samples after resample"})
                continue

            rets = r[1:] / r[:-1] - 1.0
            if rets.size <= 1:
                investigation.append({"symbol": s, "variance": 0.0, "reason": "Insuff. rets"})
                continue
                
            var = float(np.var(rets, ddof=1))
            investigation.append({"symbol": s, "variance": var, "reason": "OK"})

        except Exception as e:
            investigation.append({"symbol": s, "variance": -1.0, "reason": f"Error: {e}"})

    # 3. Analyze Results
    constant_assets = [x for x in investigation if x["variance"] <= 1e-18 and x["variance"] >= 0]
    error_assets = [x for x in investigation if x["variance"] < 0]
    
    print("\n--- CONSTANT ASSETS (Variance <= 1e-18) ---")
    for x in constant_assets:
        print(f"{x['symbol']}: Var={x['variance']} ({x['reason']})")
        
    print("\n--- ERROR/EMPTY ASSETS ---")
    for x in error_assets:
        print(f"{x['symbol']}: {x['reason']}")
        
    print(f"\nSummary: {len(constant_assets)} constant, {len(error_assets)} errors out of {len(syms_all)} total.")

if __name__ == "__main__":
    main()
