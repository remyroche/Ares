import os
import sys
import pandas as pd
import numpy as np

# Add Ares to path
sys.path.append("/Users/remyroche/Documents/Ares")

from extreme_price_movements.config import CFG
from extreme_price_movements.run_pipeline import run_backtest, _normalize_cfg_paths, _apply_fee_model, BASE_ROUND_TRIP_FEE_PCT

def run_limit_order_analysis():
    ts = "20260214_190000"
    
    # Offsets to test: Market (0%), Limit (0.1%), Limit (0.2%), Limit (0.3%)
    offsets_bps = [0.0, 10.0, 20.0, 30.0]
    
    results = []
    
    for offset in offsets_bps:
        print(f"\n--- Running Backtest with Limit Offset {offset} bps ---")
        cfg_override = CFG.copy()
        _apply_fee_model(cfg_override, BASE_ROUND_TRIP_FEE_PCT)
        _normalize_cfg_paths(cfg_override)
        if offset > 0:
            cfg_override["use_limit_orders"] = True
            cfg_override["limit_offset_bps"] = offset
            # Optional: Assume lower fee for maker orders? Let's use 10 bps for limit, 25 for market
            cfg_override["fee_bps_limit_entry"] = 10.0
        else:
            cfg_override["use_limit_orders"] = False
            
        # Run backtest end-to-end for this config and read resulting trades.
        run_backtest(cfg_override, ts_override=ts)
        run_id = ts
        bt_path = os.path.join(cfg_override["data_root"], "artifacts", run_id, "backtest_results.csv")
        if os.path.exists(bt_path):
            df_trades = pd.read_csv(bt_path)
        else:
            df_trades = pd.DataFrame()
        
        if df_trades is not None and not df_trades.empty:
            filled_trades = len(df_trades)
            win_rate = (df_trades["pnl"] > 0).mean() * 100
            total_pnl = df_trades["pnl"].sum()
            avg_pnl = df_trades["pnl"].mean()
            
            # Median MAE/MFE of the actual filled trades
            median_mae = df_trades["mae_pct"].median() if "mae_pct" in df_trades.columns else np.nan
            median_mfe = df_trades["mfe_pct"].median() if "mfe_pct" in df_trades.columns else np.nan
        else:
            filled_trades = 0
            win_rate = 0.0
            total_pnl = 0.0
            avg_pnl = 0.0
            median_mae = np.nan
            median_mfe = np.nan
            
        results.append({
            "Offset (bps)": offset,
            "Total Trades Filled": filled_trades,
            "Win Rate (%)": win_rate,
            "Total PnL": total_pnl,
            "Avg PnL": avg_pnl,
            "Median MAE (%)": median_mae * 100 if not np.isnan(median_mae) else np.nan,
            "Median MFE (%)": median_mfe * 100 if not np.isnan(median_mfe) else np.nan
        })
        
    df_results = pd.DataFrame(results)
    print("\n\n=== Limit Order Execution Analysis ===")
    print(df_results.to_markdown(index=False))
    
    out_dir = f"/Users/remyroche/Documents/Ares/reports/{ts}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/limit_order_analysis.md", "w") as f:
        f.write("# Limit Order Execution Analysis\n\n")
        f.write(df_results.to_markdown(index=False))
        f.write("\n\n*Note: Offset 0.0 represents Market Orders (25 bps taker fee). Limits use maker fee assumption if filled.*")

if __name__ == "__main__":
    run_limit_order_analysis()
