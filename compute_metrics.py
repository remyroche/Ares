import pandas as pd
import numpy as np

for direction in ['long', 'short']:
    print(f"\\n=== Direction: {direction.upper()} ===")
    df = pd.read_parquet(f'data/artifacts/20260214_190000/ridge_sizer/ridge_sizer_oof_{direction}.parquet')
    
    # df has columns: 'bucket', 'score', 'fwd_ret_H4', 'ts', 'asset'
    
    for bucket, group in df.groupby('bucket'):
        pred = group['score'].values
        # the fwd_ret_H4 is already the net return? Or gross? 
        # run_ridge_sizer says it uses trade_outcomes['return'] which is gross or net depends. 
        # But we know cost is 0.005
        # The Ridge predictions are scores that we rank.
        true_raw = group['fwd_ret_H4'].values
        
        # Sizing logic from _evaluate_params:
        k = max(1, int(0.30 * len(pred)))
        selected_indices = np.argpartition(pred, -k)[-k:]
        
        sel_pred = pred[selected_indices]
        order = np.argsort(sel_pred)
        rank_local = np.empty(len(sel_pred), dtype=float)
        rank_local[order] = (np.arange(len(sel_pred), dtype=float) + 0.5) / max(len(sel_pred), 1)
        pos_frac = 0.05 + 0.10 * rank_local
        
        selected_returns = true_raw[selected_indices]
        ts_masked = group['ts'].values[selected_indices]
        
        net_returns = (selected_returns - 0.005) * pos_frac
        
        total_pnl = np.sum(net_returns)
        
        ts_conv = pd.to_datetime(ts_masked)
        n_days = (ts_conv.max() - ts_conv.min()).total_seconds() / 86400.0
        n_days = max(1.0/24.0, n_days)
        pnl_per_day = total_pnl / n_days
        trades_per_day = len(selected_returns) / n_days
        
        daily_df = pd.DataFrame({'return': net_returns, 'date': ts_conv.date})
        daily_sum = daily_df.groupby('date')['return'].sum()
        
        unique_dates = np.unique(pd.to_datetime(group['ts'].values).date)
        full_daily = pd.Series(0.0, index=unique_dates)
        full_daily.update(daily_sum)
        daily_returns = full_daily.values
        
        mean_daily = np.mean(daily_returns)
        std_daily = np.std(daily_returns)
        downside = daily_returns[daily_returns < 0]
        downside_dev = np.sqrt(np.mean(downside**2)) if len(downside) > 0 else 0.0
        sortino = mean_daily / downside_dev * np.sqrt(252) if downside_dev > 0 else 0.0
        
        cum_ret = np.cumsum(daily_returns)
        running_max = np.maximum.accumulate(cum_ret)
        drawdowns = running_max - cum_ret
        max_dd = np.max(drawdowns)
        
        print(f"Bucket: {bucket}")
        print(f"  PnL/Day: {pnl_per_day:.6f}")
        print(f"  Trades/Day: {trades_per_day:.4f}")
        print(f"  Sortino: {sortino:.3f}")
        print(f"  MaxDD: {max_dd:.6f}")
        print(f"  Total Trades: {len(selected_indices)}")
        print(f"  Win Rate (Net): {np.mean(net_returns > 0):.3f}")
