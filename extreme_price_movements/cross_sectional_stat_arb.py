import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
import ccxt
import time
import os
from pathlib import Path

# Local imports
from extreme_price_movements.data_store import PartitionedOHLCVStore
from extreme_price_movements.universe import get_training_universe, refresh_margin_universe_daily
from extreme_price_movements.fast_funcs import (
    numba_rolling_std,
    numba_rolling_mean,
    numba_rolling_corr,
    numba_pct_change,
    numba_rolling_median,
    numba_rolling_sum
)
from extreme_price_movements.utils import tprint, clean_dataset

# Suppress warnings
warnings.filterwarnings("ignore")

class CrossSectionalStatArb:
    def __init__(self, data_dir="data", config=None):
        self.data_dir = data_dir
        self.config = config or {
            "timeframe": "1h",
            "lookback_beta": 30 * 24,  # 30 days
            "lookback_regime": 60 * 24, # 60 days
            "horizon": 6,  # 6 hours
            "vol_norm_window": 24, # 24 hours
            "min_history": 60 * 24, # Min history to start
            "train_window_start": 90 * 24, # Start training after 90 days
            "retrain_freq": 30 * 24, # Retrain every 30 days
            "fees": 0.005, # 0.5% round trip
            "n_deciles": 10,
            "liquidity_filter_pct": 0.20, # Bottom 20% excluded
            "market_symbol": "BTC/USDT", # Or fetch dynamically
            "mock_data": False, # For testing
            "universe_size": 50, # Top 50 liquid assets
            "max_weight_per_asset": 0.15, # Risk Control
        }
        self.store = PartitionedOHLCVStore(root_dir=data_dir, timeframe=self.config["timeframe"])
        self.models = {}
        self.features = None
        self.targets = None
        self.regimes = None
        self.predictions = pd.DataFrame()
        self.positions = pd.DataFrame()
        self.trade_logs = pd.DataFrame()

    def load_data(self):
        """Loads data from store or fetches if missing."""
        tprint("Loading data...")

        if self.config.get("mock_data"):
            self._generate_mock_data()
            return

        # Universe Selection
        try:
            # Try to get top liquid assets
            # We use a simple heuristic: list existing directories in data/ohlcv
            # If empty, we might need to fetch.
            existing_symbols = self._list_existing_symbols()

            if not existing_symbols:
                tprint("No existing data found. Attempting to fetch top liquid assets...")
                self._fetch_initial_data()
                existing_symbols = self._list_existing_symbols()

            if not existing_symbols:
                # Fallback to mock data if fetch fails (e.g. no API keys or blocked)
                tprint("WARNING: No data available and fetch failed. Falling back to Mock Data.")
                self._generate_mock_data()
                return

            self.symbols = existing_symbols
            # Ensure BTC is in symbols
            if self.config["market_symbol"] not in self.symbols:
                 # Try to find a BTC equivalent
                 btc_candidates = [s for s in self.symbols if "BTC" in s and "USDT" in s]
                 if btc_candidates:
                     self.config["market_symbol"] = btc_candidates[0]
                     tprint(f"Using {self.config['market_symbol']} as market proxy.")
                 else:
                     tprint("WARNING: BTC/USDT not found. Market neutralization might fail.")

        except Exception as e:
            tprint(f"Error initializing universe: {e}")
            self._generate_mock_data()
            return

        # Load Data
        dfs = {}
        for sym in self.symbols:
            df = self.store.load(sym)
            if not df.empty:
                dfs[sym] = df

        if not dfs:
            tprint("No data loaded.")
            return

        # Align Data
        # We need Close, Volume at least.
        # Create Panel
        self.close = pd.DataFrame({s: dfs[s]['close'] for s in dfs}).sort_index()
        self.volume = pd.DataFrame({s: dfs[s]['volume'] for s in dfs}).sort_index()

        # Forward fill small gaps (up to 3 hours), then drop
        self.close = self.close.ffill(limit=3)
        self.volume = self.volume.fillna(0) # Volume 0 if missing

        # Filter to common start? No, allow ragged start.
        tprint(f"Loaded {len(self.symbols)} assets. Range: {self.close.index.min()} to {self.close.index.max()}")

    def _list_existing_symbols(self):
        """Lists symbols from data store directory."""
        ohlcv_dir = Path(self.data_dir) / "ohlcv"
        if not ohlcv_dir.exists():
            return []

        symbols = []
        for p in ohlcv_dir.glob("symbol=*"):
            s = p.name.replace("symbol=", "").replace("_", "/")
            symbols.append(s)
        return symbols

    def _fetch_initial_data(self):
        """Fetches initial batch of data for top assets."""
        tprint("Fetching initial data for universe...")
        try:
            # 1. Fetch Universe
            univ = get_training_universe(None, {
                "market_basket": ["BTC/USDT", "ETH/USDT"],
                "fetch_symbols_M": self.config["universe_size"],
                "variance_filter_pct": 0.5 # Dummy
            }, self.store)

            # 2. Fetch Data
            # We use a temporary exchange instance
            exchange = ccxt.binance()
            # Start from 1 year ago?
            start_ts = int((pd.Timestamp.utcnow() - pd.DateOffset(months=6)).timestamp() * 1000)

            for sym in univ:
                self.store.update_symbol(exchange, sym, start_ts)

        except Exception as e:
            tprint(f"Error fetching data: {e}")

    def _generate_mock_data(self):
        """Generates mock data for testing."""
        tprint("Generating mock data...")
        dates = pd.date_range(end=pd.Timestamp.utcnow(), periods=24*100, freq='h') # 100 days
        self.symbols = [f"ASSET_{i}" for i in range(20)] + ["BTC/USDT"]

        # Random Walk
        close_data = {}
        vol_data = {}
        np.random.seed(42)

        # Generate Market (BTC)
        mkt_rets = np.random.normal(0, 0.002, size=len(dates)) # Low vol
        mkt_price = 10000 * (1 + mkt_rets).cumprod()
        close_data["BTC/USDT"] = mkt_price
        vol_data["BTC/USDT"] = np.random.uniform(1e6, 1e7, size=len(dates))

        for s in self.symbols:
            if s == "BTC/USDT": continue

            # Beta + Idiosyncratic
            beta = np.random.uniform(0.5, 1.5)
            idio_rets = np.random.normal(0, 0.005, size=len(dates))

            rets = beta * mkt_rets + idio_rets

            price = 100 * (1 + rets).cumprod()
            close_data[s] = price
            vol_data[s] = np.random.uniform(1000, 10000, size=len(dates))

        self.close = pd.DataFrame(close_data, index=dates)
        self.volume = pd.DataFrame(vol_data, index=dates)
        self.config["market_symbol"] = "BTC/USDT"
        tprint(f"Mock data generated: {len(self.symbols)} symbols, {len(dates)} hours.")

    def compute_features(self):
        """Computes features, targets, and regime indicators."""
        tprint("Computing features...")

        # 1. Returns
        # Simple returns: close / close.shift(1) - 1
        # numba_pct_change uses shift
        self.ret1h = numba_pct_change(self.close, 1)

        # Market Return
        mkt_sym = self.config["market_symbol"]
        if mkt_sym not in self.ret1h.columns:
            # Maybe fallback to first column?
            mkt_sym = self.ret1h.columns[0]
            tprint(f"Market symbol not found, using {mkt_sym}")
            self.config["market_symbol"] = mkt_sym

        self.r_mkt = self.ret1h[mkt_sym]

        # 2. Beta (Rolling 30d = 720h)
        # Beta = Corr(r, r_mkt) * Std(r) / Std(r_mkt)

        # Broadcast Market Return
        r_mkt_df = pd.DataFrame(
            np.tile(self.r_mkt.values[:, None], (1, len(self.close.columns))),
            index=self.r_mkt.index,
            columns=self.close.columns
        )

        window_beta = self.config["lookback_beta"]

        tprint("Computing Beta...")
        # Rolling Corr
        corr_beta = numba_rolling_corr(self.ret1h, r_mkt_df, window_beta)

        # Rolling Std
        std_asset = numba_rolling_std(self.ret1h, window_beta)
        std_mkt = numba_rolling_std(r_mkt_df, window_beta)

        self.beta = corr_beta * (std_asset / (std_mkt + 1e-9))
        self.beta = self.beta.fillna(1.0)

        # 3. Residual Returns
        # r_res = r - beta * r_mkt
        self.r_res = self.ret1h - self.beta * r_mkt_df

        # 4. Target Generation (6h Forward)
        # y_raw = (close_{t+6} - close_t) / close_t
        # This is exactly return 6h shifted back 6h? No, forward looking.
        # We can compute 6h return then shift back 6.
        ret6h = numba_pct_change(self.close, self.config["horizon"])
        # Shift back to align with time t (target for time t is return from t to t+6)
        # numba_pct_change(x, 6) at index i computes (x[i] - x[i-6])/x[i-6]
        # So ret6h[t+6] is return from t to t+6.
        # y_raw[t] = ret6h[t+6]
        self.y_raw = ret6h.shift(-self.config["horizon"])

        # Market Return Forward 6h
        r_mkt_6h = numba_pct_change(self.close[mkt_sym], self.config["horizon"]).shift(-self.config["horizon"])
        r_mkt_6h_df = pd.DataFrame(
            np.tile(r_mkt_6h.values[:, None], (1, len(self.close.columns))),
            index=r_mkt_6h.index,
            columns=self.close.columns
        )

        # Residual Forward Return
        # y_res = y_raw - beta * r_mkt_6h
        self.y_res = self.y_raw - self.beta * r_mkt_6h_df

        # Vol Normalize (24h realized vol)
        self.sigma_24h = numba_rolling_std(self.ret1h, self.config["vol_norm_window"])

        # y_norm
        self.y_norm = self.y_res / (self.sigma_24h + 1e-9)

        # Clip Target
        self.y_clip = self.y_norm.clip(-3, 3)

        # 5. Features
        # A) Residual 6h momentum
        # mom6 = 6h return (Backward looking)
        self.mom6 = ret6h

        # Market 6h return (Backward)
        r_mkt_6h_back = numba_pct_change(self.close[mkt_sym], self.config["horizon"])
        r_mkt_6h_back_df = pd.DataFrame(
            np.tile(r_mkt_6h_back.values[:, None], (1, len(self.close.columns))),
            index=r_mkt_6h_back.index,
            columns=self.close.columns
        )

        self.mom6_res = self.mom6 - self.beta * r_mkt_6h_back_df

        # B) Residual 1h reversal
        # mom1_res = r_res (already computed)
        self.mom1_res = self.r_res

        # C) Relative Volume 6h
        # rvol6 = volume_6h / rolling 7d avg volume
        vol6 = numba_rolling_sum(self.volume, self.config["horizon"])

        vol_1h_avg_7d = numba_rolling_mean(self.volume, 7 * 24)
        self.rvol6 = vol6 / (vol_1h_avg_7d * 6 + 1e-9)

        # D) Cross-sectional rank of mom6_res
        # rank_res = rank(mom6_res) scaled [0,1]
        self.rank_res = self.mom6_res.rank(axis=1, pct=True)

        # E) Daily Volatility -> self.sigma_24h (Already computed)

        # F) Market Regime Indicators
        # BTC 24h realized vol
        btc_vol_24h = numba_rolling_std(self.ret1h[mkt_sym], 24)
        # Cross-sectional dispersion: std_i(r_res_{i,t})
        # Axis 1 std
        self.disp_t = self.r_res.std(axis=1)
        # BTC 24h momentum
        btc_mom_24h = numba_pct_change(self.close[mkt_sym], 24)

        # Broadcast Regime Features
        self.btc_vol_24h = pd.DataFrame(
            np.tile(btc_vol_24h.values[:, None], (1, len(self.close.columns))),
            index=btc_vol_24h.index,
            columns=self.close.columns
        )
        self.disp_t_expanded = pd.DataFrame(
            np.tile(self.disp_t.values[:, None], (1, len(self.close.columns))),
            index=self.disp_t.index,
            columns=self.close.columns
        )
        self.btc_mom_24h = pd.DataFrame(
            np.tile(btc_mom_24h.values[:, None], (1, len(self.close.columns))),
            index=btc_mom_24h.index,
            columns=self.close.columns
        )

        # Stack Features for Training
        self.feature_names = [
            "mom6_res", "mom1_res", "rvol6", "rank_res", "sigma_24h",
            "btc_vol_24h", "disp_t", "btc_mom_24h"
        ]

        self.features_dict = {
            "mom6_res": self.mom6_res,
            "mom1_res": self.mom1_res,
            "rvol6": self.rvol6,
            "rank_res": self.rank_res,
            "sigma_24h": self.sigma_24h,
            "btc_vol_24h": self.btc_vol_24h,
            "disp_t": self.disp_t_expanded,
            "btc_mom_24h": self.btc_mom_24h
        }

        # 6. Regime Detection
        # Rolling 60d median
        disp_median_60d = numba_rolling_median(self.disp_t, self.config["lookback_regime"])
        btc_vol_median_60d = numba_rolling_median(btc_vol_24h, self.config["lookback_regime"])

        # Flags
        # High Dispersion: disp_t > median
        # High Market Vol: BTC_vol > median

        is_high_disp = self.disp_t > disp_median_60d
        is_high_vol = btc_vol_24h > btc_vol_median_60d

        self.regime_flags = pd.Series(index=self.disp_t.index, data="R3")
        self.regime_flags[is_high_disp & is_high_vol] = "R1"
        self.regime_flags[is_high_disp & ~is_high_vol] = "R2"
        # R3 is default (Low Disp)

        tprint("Features and Regimes computed.")

    def _prepare_training_data(self, start_idx, end_idx):
        """Prepares X, y for training window, filtering for R1/R2."""
        valid_regimes = ["R1", "R2"]

        # Use integer slicing for time window
        # We need to intersect this slice with regime mask

        # Get regime mask for the whole series first, then slice?
        # Or slice first? Slice first is faster.

        regime_slice = self.regime_flags.iloc[start_idx:end_idx]
        mask_in_slice = regime_slice.isin(valid_regimes)

        if not mask_in_slice.any():
            return None, None

        X_list = []
        for fname in self.feature_names:
            df = self.features_dict[fname]
            # Slice time window
            subset_window = df.iloc[start_idx:end_idx]
            # Filter by regime
            subset = subset_window.loc[mask_in_slice]
            X_list.append(subset.values.flatten())

        X = np.column_stack(X_list)

        y_window = self.y_clip.iloc[start_idx:end_idx]
        y_subset = y_window.loc[mask_in_slice]
        y = y_subset.values.flatten()

        valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)

        return X[valid_mask], y[valid_mask]

    def _predict(self, model, start_idx, end_idx):
        """Predicts for the window (all regimes, we deploy in R1/R2 later)."""

        # Use integer slicing
        # Check if slice is valid
        if start_idx >= len(self.close):
            return None

        rows = end_idx - start_idx
        if rows <= 0:
            return None

        # Adjust end_idx if it exceeds length (iloc handles this but for reshape we need exact rows)
        real_end_idx = min(end_idx, len(self.close))
        rows = real_end_idx - start_idx
        cols = len(self.close.columns)

        X_list = []
        for fname in self.feature_names:
            df = self.features_dict[fname]
            subset = df.iloc[start_idx:real_end_idx]
            X_list.append(subset.values.flatten())

        X = np.column_stack(X_list)

        # Handle NaNs in prediction input
        X = np.nan_to_num(X, nan=0.0)

        preds = model.predict(X)
        preds_reshaped = preds.reshape(rows, cols)

        preds_df = pd.DataFrame(
            preds_reshaped,
            index=self.close.index[start_idx:real_end_idx],
            columns=self.close.columns
        )
        return preds_df

    def train_and_simulate(self):
        """Walk-forward training and simulation."""
        tprint("Starting Walk-Forward Simulation...")

        n_bars = len(self.close)
        train_start_idx = self.config["train_window_start"]
        step_size = self.config["retrain_freq"]

        current_idx = train_start_idx

        all_preds = []

        model = ExtraTreesRegressor(n_estimators=50, n_jobs=-1, max_depth=5, random_state=42)

        while current_idx < n_bars:
            tprint(f"Step: {current_idx}/{n_bars} ({self.close.index[current_idx]})")

            # Embargo logic
            train_end_idx = current_idx - self.config["horizon"]
            if train_end_idx < self.config["min_history"]:
                current_idx += step_size
                continue

            X_train, y_train = self._prepare_training_data(0, train_end_idx)

            if X_train is not None and len(X_train) > 100:
                model.fit(X_train, y_train)

                pred_end_idx = min(current_idx + step_size, n_bars)
                # tprint(f"  Predicting {self.close.index[current_idx]} to {self.close.index[pred_end_idx-1]}")

                preds = self._predict(model, current_idx, pred_end_idx)
                if preds is not None:
                    all_preds.append(preds)
            else:
                tprint("  Insufficient training data. Skipping step.")

            current_idx += step_size

        if not all_preds:
            tprint("No predictions generated.")
            return

        self.predictions = pd.concat(all_preds).sort_index()
        tprint("Simulation Complete. Constructing Portfolio...")

        self._construct_portfolio()

    def _construct_portfolio(self):
        """Constructs portfolio from predictions."""
        # Pre-compute dollar volume for liquidity filter
        dollar_vol = self.close * self.volume
        adv7d = numba_rolling_mean(dollar_vol, 7 * 24)

        metrics = []

        # Tracking for Risk Controls & Turnover
        pnl_history = [] # Hourly PnL
        tranche_weights = {} # Key: timestamp, Value: Series(weights)
        horizon = self.config["horizon"]

        valid_idx = self.predictions.index

        # We iterate over ALL timestamps in prediction range to handle Kill Switch correctly
        # Even if R3, we might need to update PnL history (0 PnL)
        if len(valid_idx) == 0:
            return

        start_t = valid_idx[0]
        end_t = valid_idx[-1]
        full_idx = self.close.loc[start_t:end_t].index

        for t in full_idx:
            # 1. Kill Switch Check (Rolling 7d Sharpe)
            kill_switch_active = False
            if len(pnl_history) >= 7 * 24:
                # Compute rolling sharpe on last 7 days
                window_pnl = np.array(pnl_history[-168:])
                std_pnl = np.std(window_pnl)
                if std_pnl > 1e-9:
                    sharpe = np.mean(window_pnl) / std_pnl * np.sqrt(24 * 365) # Annualized
                    if sharpe < 0:
                        kill_switch_active = True

            # Initialize weights for new tranche at t
            weights_t = pd.Series(0.0, index=self.close.columns)
            regime = "R3"

            should_trade = False
            if t in valid_idx and t in self.regime_flags.index:
                regime = self.regime_flags.loc[t]
                if regime in ["R1", "R2"] and not kill_switch_active:
                    should_trade = True

            if should_trade:
                preds_t = self.predictions.loc[t]

                # Liquidity Filter
                adv_t = adv7d.loc[t]
                liq_thresh = adv_t.quantile(self.config["liquidity_filter_pct"])
                liquid_mask = adv_t >= liq_thresh

                valid_preds = preds_t[liquid_mask]

                if len(valid_preds) >= 10:
                    # Rank
                    ranks = valid_preds.rank(pct=True)

                    # Deciles
                    long_mask = ranks >= (1.0 - 1.0/self.config["n_deciles"])
                    short_mask = ranks <= (1.0/self.config["n_deciles"])

                    longs = valid_preds.index[long_mask]
                    shorts = valid_preds.index[short_mask]

                    if len(longs) > 0 and len(shorts) > 0:
                        w_long = 1.0 / len(longs)
                        w_short = -1.0 / len(shorts)

                        weights_t[longs] = w_long
                        weights_t[shorts] = w_short

                        # Regime Sizing
                        size_mult = 0.5 if regime == "R2" else 1.0
                        weights_t *= size_mult

                        # Max Weight Cap
                        max_w = self.config.get("max_weight_per_asset", 0.15)
                        weights_t = weights_t.clip(-max_w, max_w)

                        # Beta Hedge
                        beta_t = self.beta.loc[t].fillna(1.0)
                        port_beta = (weights_t * beta_t).sum()

                        # Hedge with BTC
                        mkt_sym = self.config["market_symbol"]
                        if mkt_sym in weights_t.index:
                            weights_t[mkt_sym] -= port_beta
                        else:
                            pass

            # Store Tranche
            # Only store if non-zero
            if (weights_t != 0).any():
                tranche_weights[t] = weights_t

            # Calculate Turnover
            # Turnover = sum(abs(w_new - w_old))
            # w_old is the tranche expiring at t (initiated at t - horizon)
            # w_new is weights_t
            t_expire = t - pd.Timedelta(hours=horizon)
            w_expire = tranche_weights.get(t_expire, pd.Series(0.0, index=self.close.columns))

            # Net turnover at this step
            turnover = (weights_t - w_expire).abs().sum()

            # Calculate PnL for *this specific tranche* expiring at t
            step_pnl = 0.0
            gross_pnl_val = 0.0
            cost_val = 0.0
            exposure_val = 0.0

            # Check if we have an expiring tranche
            # Use w_expire directly (if it has non-zero exposure)
            if w_expire.abs().sum() > 0 and t_expire in self.y_raw.index:
                ret_vec = self.y_raw.loc[t_expire]

                # Check NaNs in relevant assets
                valid_assets = w_expire[w_expire != 0].index
                if not ret_vec[valid_assets].isna().any():
                    gross_pnl_val = (w_expire * ret_vec).sum()
                    exposure_val = w_expire.abs().sum()
                    # Fees: "Subtract 0.5% fee per round trade."
                    # We assume fee covers round trip.
                    cost_val = exposure_val * self.config["fees"]
                    step_pnl = gross_pnl_val - cost_val

            # Clean up old tranche from storage
            if t_expire in tranche_weights:
                del tranche_weights[t_expire]

            # Update History for Kill Switch
            pnl_history.append(step_pnl)

            # Record Metrics (if trade was active or expiring)
            if should_trade or step_pnl != 0:
                 metrics.append({
                    "timestamp": t,
                    "regime": regime,
                    "pnl": step_pnl,
                    "gross_pnl": gross_pnl_val,
                    "cost": cost_val,
                    "exposure": exposure_val,
                    "turnover": turnover,
                    "kill_switch": kill_switch_active,
                    "n_trades": (weights_t != 0).sum()
                })

        self.trade_logs = pd.DataFrame(metrics)
        if not self.trade_logs.empty:
            self.trade_logs = self.trade_logs.set_index("timestamp")
            tprint(f"Backtest Complete. Total PnL: {self.trade_logs['pnl'].sum():.4f}")
        else:
            tprint("Backtest Complete. No trades executed.")

    def evaluate(self):
        """Computes and prints metrics."""
        if self.trade_logs.empty:
            tprint("No trades to evaluate.")
            return

        df = self.trade_logs

        total_pnl = df['pnl'].sum()
        avg_pnl = df['pnl'].mean()
        # Approx annualized sharpe based on hourly pnl stream
        # df contains rows for every hour where trade happened OR PnL was realized.
        # This approximates the time series.
        if df['pnl'].std() > 1e-9:
             sharpe = df['pnl'].mean() / df['pnl'].std() * np.sqrt(365 * 24)
        else:
             sharpe = 0.0

        avg_turnover = df['turnover'].mean()

        tprint("="*30)
        tprint("STRATEGY PERFORMANCE")
        tprint("="*30)
        tprint(f"Total PnL: {total_pnl:.4f}")
        tprint(f"Avg PnL per Hour: {avg_pnl:.4f}")
        tprint(f"Sharpe (Annualized): {sharpe:.2f}")
        tprint(f"Avg Turnover: {avg_turnover:.4f}")
        tprint(f"Trades Count: {len(df)}")
        tprint(f"Regime Breakdown (PnL):")
        try:
            print(df.groupby("regime")['pnl'].sum())
        except: pass

        # Removed CSV export to avoid clutter
        # df.to_csv("strategy_trade_log.csv")

if __name__ == "__main__":
    arb = CrossSectionalStatArb(config={
        "timeframe": "1h",
        "lookback_beta": 720,
        "lookback_regime": 1440,
        "horizon": 6,
        "vol_norm_window": 24,
        "min_history": 24*60,
        "train_window_start": 24*90,
        "retrain_freq": 24*30,
        "fees": 0.005,
        "n_deciles": 10,
        "liquidity_filter_pct": 0.20,
        "market_symbol": "BTC/USDT",
        "mock_data": True,
        "universe_size": 20
    })

    arb.load_data()
    arb.compute_features()
    arb.train_and_simulate()
    arb.evaluate()
