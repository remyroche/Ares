import numpy as np
import pandas as pd

from config import CFG
from utils import tprint, Timer

from universe import refresh_margin_universe_daily, build_fetch_universe
from data_store import make_spot_exchange, OHLCVStore, to_panel, downcast_panel_float32
from features import compute_market_features, add_regime_gates, compute_features_hourly
from engine import hourly_engine_backtest

def assert_basket_present(panel_close: pd.DataFrame, basket: list[str]):
    missing = [s for s in basket if s not in panel_close.columns]
    if missing:
        raise ValueError(f"Market basket symbols missing from fetched data: {missing}")

if __name__ == "__main__":
    cfg = CFG
    tprint("BOOT")

    ex = make_spot_exchange()

    with Timer("Margin universe refresh"):
        mu = refresh_margin_universe_daily(None, quote="USDT")
    margin_symbols = mu.symbols
    tprint(f"Margin symbols: {len(margin_symbols)}")

    # build fetch universe
    syms = build_fetch_universe(margin_symbols, cfg["market_basket"], cfg["fetch_symbols_M"])
    tprint(f"Fetch universe: {len(syms)} (includes basket={len(cfg['market_basket'])})")

    store = OHLCVStore(root_dir=cfg["data_root"], timeframe=cfg["timeframe"])

    since = (pd.Timestamp.utcnow().tz_localize("UTC") - pd.Timedelta(days=365*cfg["fetch_years"])).floor("D")
    since_ms = int(since.value // 10**6)

    dfs = {}
    with Timer("Fetch/Update symbols (Parquet)"):
        for i, s in enumerate(syms, 1):
            if i % 25 == 0:
                tprint(f"Progress: {i}/{len(syms)}")
            try:
                df = store.update_symbol(ex, s, since_ms)
                if len(df) >= 24 * 365 * 2:
                    dfs[s] = df
            except Exception as e:
                tprint(f"Fetch error {s}: {e}")

    with Timer("Build panel"):
        panel = to_panel(dfs)

        # align common columns
        common = set(panel["close"].columns)
        for k in panel:
            common &= set(panel[k].columns)
        common = sorted(common)
        for k in panel:
            panel[k] = panel[k][common].dropna(how="all")

        assert_basket_present(panel["close"], cfg["market_basket"])
        panel = downcast_panel_float32(panel)

        tprint(f"Panel: ts={len(panel['close'])}  syms={panel['close'].shape[1]}")

    with Timer("Market features + gates"):
        mkt_df = compute_market_features(panel, cfg["market_basket"], trend_sma_hours=24*14)
        mkt_gates = add_regime_gates(
            mkt_df,
            gate_vol_lookback_hours=cfg["gate_vol_lookback_hours"],
            gate_trend_thr=cfg["gate_trend_thr"]
        )

    with Timer("Feature engineering"):
        feats = compute_features_hourly(panel, mkt_gates, cfg)

    symbols_all = [s for s in panel["close"].columns if s in margin_symbols]
    tprint(f"Trade universe (margin-overlap): {len(symbols_all)}")

    with Timer("Hourly engine backtest"):
        eq, trades, stats = hourly_engine_backtest(panel, feats, mkt_gates, cfg, symbols_all)

    tprint(f"STATS: {stats}")
    if not trades.empty:
        tprint("TRADES sample:")
        print(trades.head(15).to_string(index=False))
    else:
        tprint("(no trades)")
