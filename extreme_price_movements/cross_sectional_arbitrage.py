import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
import os
import glob

# Global Strategy Constants
FEE_PCT = 0.005
FORECAST_HORIZON_H = 6
BETA_WINDOW_H = 30 * 24
RETRAIN_FREQ_H = 30 * 24
REGIME_WINDOW_D = 60
LIQUIDITY_FILTER_PCT = 0.20
KILL_SWITCH_WINDOW_D = 7

class DataLoader:
    def __init__(self, data_dir="extreme_price_movements", n_assets=20):
        self.data_dir = data_dir
        self.n_assets = n_assets

    def load(self):
        # Attempt to find data
        print(f"Attempting to load data from {self.data_dir}...")
        dfs = []
        # Check for parquet files in subdirs (e.g. data_store format)
        # or simple CSVs
        for root, dirs, files in os.walk(self.data_dir):
            for f in files:
                if f.endswith(".parquet") or f.endswith(".csv"):
                    # heuristic: try to identify asset name from filename
                    # e.g. "ETHUSDT.csv" or "symbol=ETH_USDT"
                    path = os.path.join(root, f)
                    try:
                        if f.endswith(".parquet"):
                            df = pd.read_parquet(path)
                        else:
                            df = pd.read_csv(path)

                        # Normalize columns
                        df.columns = [c.lower() for c in df.columns]
                        if 'timestamp' not in df.columns and 'ts' in df.columns:
                            df.rename(columns={'ts': 'timestamp'}, inplace=True)

                        # If index is timestamp
                        if not 'timestamp' in df.columns and isinstance(df.index, pd.DatetimeIndex):
                            df.reset_index(inplace=True)
                            df.rename(columns={'index': 'timestamp'}, inplace=True)

                        if 'timestamp' in df.columns and 'close' in df.columns:
                            # Assume filename is asset
                            asset_name = f.split('.')[0].replace('symbol=', '').replace('_', '/')
                            df['asset'] = asset_name

                            # Ensure numeric
                            for col in ['open','high','low','close','volume']:
                                if col in df.columns:
                                    df[col] = pd.to_numeric(df[col], errors='coerce')

                            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                            df.dropna(subset=['close'], inplace=True)
                            if len(df) > 100:
                                dfs.append(df[['timestamp', 'asset', 'open', 'high', 'low', 'close', 'volume']])
                    except Exception as e:
                        print(f"Failed to load {path}: {e}")

        if len(dfs) >= 2: # Need at least 2 for cross-sectional
            print(f"Loaded {len(dfs)} assets from disk.")
            data = pd.concat(dfs).sort_values(['timestamp', 'asset'])
            data.set_index(['timestamp', 'asset'], inplace=True)
            # Forward fill missing hours?
            # Reindex to common hourly grid
            # For simplicity, just return as is
            return data
        else:
            print("Insufficient data found. Falling back to Synthetic Data.")
            return self.generate_synthetic()

    def generate_synthetic(self):
        print(f"Generating synthetic data for {self.n_assets} assets + Market...")
        np.random.seed(42)
        periods = 24 * 365
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='h')

        # Market
        vol_mkt = np.abs(np.random.normal(0.002, 0.0005, periods)).cumsum()
        vol_mkt = np.abs(vol_mkt - vol_mkt.mean()) * 0.0001 + 0.002
        ret_mkt = np.random.normal(0, 1, periods) * vol_mkt
        price_mkt = 10000 * np.cumprod(1 + ret_mkt)

        asset_data = []
        df_mkt = pd.DataFrame({
            'timestamp': dates, 'asset': "BTC/USDT", 'close': price_mkt,
            'open': price_mkt, 'high': price_mkt * 1.005, 'low': price_mkt * 0.995,
            'volume': np.random.lognormal(10, 1, periods)
        })
        asset_data.append(df_mkt)

        betas = np.random.uniform(0.5, 1.5, self.n_assets)
        for i in range(self.n_assets):
            asset = f"ASSET_{i}"
            ret = betas[i] * ret_mkt + np.random.normal(0, 0.002, periods)
            price = 100 * np.cumprod(1 + ret)
            vol = np.random.lognormal(8, 0.5, periods) * (1 + 10 * np.abs(ret))

            df = pd.DataFrame({
                'timestamp': dates, 'asset': asset, 'close': price,
                'open': price, 'high': price * 1.01, 'low': price * 0.99,
                'volume': vol
            })
            asset_data.append(df)

        data = pd.concat(asset_data).sort_values(['timestamp', 'asset'])
        data.set_index(['timestamp', 'asset'], inplace=True)
        return data

def run_barrier_simulation(entry_closes, highs, lows, closes, tp, sl, hold_hours=6):
    """
    Simulate barrier exit for a set of entries.
    entry_closes: Series of entry prices (indexed by timestamp)
    Returns: Series of Returns
    """
    results = []
    # This function expects 'entry_closes' to be a Series of prices for ONE asset, indexed by time.
    # But we iterate over time in the caller usually.
    # Let's keep the iterative logic in the main class for flexibility.
    return results

class StrategyRunner:
    def __init__(self, data):
        self.data = data.copy()
        self.market_asset = "BTC/USDT"
        if self.market_asset not in self.data.index.get_level_values('asset').unique():
             self.market_asset = self.data.index.get_level_values('asset')[0]

    def run_dispersion(self):
        print("\n--- Dispersion Strategy (Baseline) ---")
        closes = self.data['close'].unstack()
        highs = self.data['high'].unstack()
        lows = self.data['low'].unstack()

        rets = closes.pct_change()
        z = rets.sub(rets.mean(axis=1), axis=0).div(rets.std(axis=1), axis=0)

        longs_mask = (z.rank(axis=1) == 1)
        shorts_mask = (z.rank(axis=1, ascending=False) == 1)

        self._simulate_trades(longs_mask, shorts_mask, closes, highs, lows, "Dispersion")

    def run_ml(self):
        print("\n--- ML Strategy ---")
        closes = self.data['close'].unstack()
        volumes = self.data['volume'].unstack()
        highs = self.data['high'].unstack()
        lows = self.data['low'].unstack()

        # Features
        ret1h = closes.pct_change()
        ret6h = closes / closes.shift(6) - 1
        ret3h = closes / closes.shift(3) - 1

        # Market
        mkt = closes[self.market_asset]
        mkt_ret = ret1h[self.market_asset]

        # Beta
        beta = ret1h.rolling(720).cov(mkt_ret).div(mkt_ret.rolling(720).var(), axis=0).fillna(1.0)

        # Residuals
        ret_res = ret1h - beta.multiply(mkt_ret, axis=0)
        mom6_res = ret6h - beta.multiply(mkt.pct_change(6), axis=0)

        # Additional Features
        # 3h Accel
        mom3 = ret3h
        mom3_lag = ret3h.shift(3)
        accel3 = mom3 - mom3_lag

        # Time
        hod = closes.index.hour
        dow = closes.index.dayofweek
        sin_hod = np.sin(2 * np.pi * hod / 24.0)
        cos_hod = np.cos(2 * np.pi * hod / 24.0)
        weekend = (dow >= 5).astype(float)

        # Target
        y_raw = closes.shift(-6) / closes - 1
        sigma = ret1h.rolling(24).std()
        y_norm = y_raw.div(sigma, axis=0).clip(-3, 3)

        # Prepare Data
        # Stack
        def stack(df, name): return df.stack().rename(name)

        feats = pd.concat([
            stack(mom6_res, 'mom6_res'),
            stack(ret_res, 'mom1_res'),
            stack(accel3, 'accel3'),
            stack(y_norm, 'target'),
            stack(beta, 'beta')
        ], axis=1)

        # Add Time Feats
        feats['sin_hod'] = feats.index.get_level_values('timestamp').hour.map(lambda h: np.sin(2*np.pi*h/24))
        feats['cos_hod'] = feats.index.get_level_values('timestamp').hour.map(lambda h: np.cos(2*np.pi*h/24))
        feats['weekend'] = (feats.index.get_level_values('timestamp').dayofweek >= 5).astype(float)

        feats.dropna(inplace=True)

        # Train/Predict Loop
        print("  Training ML Model...")
        timestamps = feats.index.get_level_values('timestamp').unique().sort_values()
        preds = []

        step = 30 * 24
        start = 30 * 24

        for i in range(start, len(timestamps), step):
            t = timestamps[i]
            # Predict up to (but not including) the next step start
            # If next step exceeds array, go to end inclusive
            next_idx = min(i + step, len(timestamps))

            if next_idx >= len(timestamps):
                # Last chunk
                pred_mask = (feats.index.get_level_values('timestamp') >= t)
            else:
                end_t = timestamps[next_idx] # Start of next chunk
                pred_mask = (feats.index.get_level_values('timestamp') >= t) & (feats.index.get_level_values('timestamp') < end_t)

            # Train on [0, t - 6h]
            train_mask = feats.index.get_level_values('timestamp') < (t - pd.Timedelta(hours=6))
            train = feats[train_mask]

            if len(train) > 1000 and pred_mask.any():
                model = ExtraTreesRegressor(n_estimators=20, max_depth=5, n_jobs=-1, random_state=42)
                cols = ['mom6_res', 'mom1_res', 'accel3', 'sin_hod', 'cos_hod', 'weekend']
                model.fit(train[cols], train['target'])

                test = feats[pred_mask]
                p = model.predict(test[cols])
                preds.append(pd.Series(p, index=test.index))

        if not preds:
            print("  No predictions.")
            return

        y_pred = pd.concat(preds)
        y_pred.name = 'score'

        # Rank
        # Reconstruct DataFrame
        scores = y_pred.unstack()

        # Signals
        # Long Top 10%, Short Bottom 10%
        ranks = scores.rank(axis=1, pct=True)
        longs = ranks >= 0.9
        shorts = ranks <= 0.1

        self._simulate_trades(longs, shorts, closes, highs, lows, "ML Strategy")

    def _simulate_trades(self, longs, shorts, closes, highs, lows, label):
        print(f"\n  Simulating Trades for {label}...")
        timestamps = longs.index

        for config, tp, sl in [("A", 0.02, -0.01), ("B", 0.03, -0.02)]:
            trades = []

            # Vectorized-ish loop? No, iteration is safer for path dependency
            # Limiting to last 180 days for speed if needed, but doing full 1y is fine for synthetic

            for t in timestamps[:-24]:
                # Longs
                for asset in longs.loc[t][longs.loc[t]].index:
                    if asset not in closes.columns: continue
                    entry = closes.loc[t, asset]
                    exit_price = entry # Default

                    # Check next 6 hours
                    for h_idx in range(1, 7):
                        if t not in timestamps: break # weird edge case
                        # Get future timestamp (approx index lookup)
                        # Assume sorted
                        try:
                            curr_t = timestamps[timestamps.get_loc(t) + h_idx]
                        except:
                            break

                        hi = highs.loc[curr_t, asset]
                        lo = lows.loc[curr_t, asset]

                        if lo <= entry * (1 + sl):
                            exit_price = entry * (1 + sl)
                            break
                        if hi >= entry * (1 + tp):
                            exit_price = entry * (1 + tp)
                            break

                        if h_idx == 6:
                            exit_price = closes.loc[curr_t, asset]

                    ret = (exit_price / entry - 1) - FEE_PCT
                    trades.append(ret)

                # Shorts
                for asset in shorts.loc[t][shorts.loc[t]].index:
                    if asset not in closes.columns: continue
                    entry = closes.loc[t, asset]
                    exit_price = entry

                    for h_idx in range(1, 7):
                        try:
                            curr_t = timestamps[timestamps.get_loc(t) + h_idx]
                        except:
                            break

                        hi = highs.loc[curr_t, asset]
                        lo = lows.loc[curr_t, asset]

                        if hi >= entry * (1 - sl): # Short SL (Price goes up)
                            exit_price = entry * (1 - sl)
                            break
                        if lo <= entry * (1 - tp): # Short TP (Price goes down)
                            exit_price = entry * (1 - tp)
                            break

                        if h_idx == 6:
                            exit_price = closes.loc[curr_t, asset]

                    ret = (entry - exit_price) / entry - FEE_PCT
                    trades.append(ret)

            if trades:
                arr = np.array(trades)
                print(f"    Config {config}: Count={len(arr)}, Mean PnL={arr.mean():.4f}, Sharpe={(arr.mean()/arr.std())*np.sqrt(24*365):.2f}")
            else:
                print(f"    Config {config}: No trades")

if __name__ == "__main__":
    loader = DataLoader()
    data = loader.load()
    runner = StrategyRunner(data)
    runner.run_dispersion()
    runner.run_ml()
