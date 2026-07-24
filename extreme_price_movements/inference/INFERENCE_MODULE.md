# Extreme Price Movements - Inference Module

## Overview

The `extreme_price_movements/inference/` directory contains the complete inference pipeline for the trading system. It handles data fetching, feature generation, candidate selection, model inference, and trade execution.

## Directory Structure

```
extreme_price_movements/inference/
├── __init__.py                 # Package initialization
├── run_inference.py            # Main entry point
├── feature_generator.py        # Feature generation
├── candidate_selector.py       # Candidate selection logic
├── data_fetcher.py            # OHLCV data fetching
├── model_orchestrator.py      # ML model orchestration
├── trade_executor.py          # Trade execution
└── trade_logger.py            # Trade logging
```

## Components

### 1. run_inference.py

Main entry point for the inference pipeline.

**Key Functions:**
- `main()` - Entry point with CLI arguments
- `run_shadow_mode()` - Shadow trading mode (paper trading)
- `run_live_mode()` - Live trading mode
- `run_challenger_monitor()` - Challenger monitoring every 5 minutes

**CLI Usage:**
```bash
# Shadow mode (paper trading)
python3 -m extreme_price_movements.inference.run_inference --shadow --lookback-hours 4 --inference-interval 300

# Live trading
python3 -m extreme_price_movements.inference.run_inference --live --lookback-hours 4 --inference-interval 300
```

**Arguments:**
- `--live` - Run live trading mode
- `--shadow` - Run shadow trading mode (default)
- `--symbols` - Symbols to trade
- `--inference-interval` - Inference interval in seconds (default: 3600 = 1 hour)
- `--challenger-interval` - Position-monitor check interval in seconds (default: 60s = 1 minute)
- `--lookback-hours` - Lookback hours for features (default: 48)

### 2. feature_generator.py

Generates market features from price data.

**Key Functions:**
- `generate_features(panel, basket_syms, ...)` - Main feature generation function
- `_compute_per_symbol_features(panel, basket_syms)` - Per-symbol feature computation
- `get_market_data(panel, symbol)` - Get market data for a symbol

**Features Generated:**
- **Per-symbol features:** `ret24h`, `ret6h`, `ret1h`, `range_12h_pct`, `volatility_zscore`, `chop_score`, `mkt_rv_24h`
- **Market features (prefixed with `mkt_`):** `mkt_close`, `mkt_high`, `mkt_low`, `mkt_volume`, `mkt_ret24h`, `mkt_ret6h`, `mkt_trend`, `mkt_rv`
- **Regime features (prefixed with `reg_`):** `reg_mkt_rv_med`, `reg_G_VOL`, `reg_G_TREND`, `reg_mkt_rv_ratio`, `reg_mkt_rv_pct`, `reg_abs_mkt_ret24h_z`, `reg_trend_bin3`

**Returns:** `Dict[str, pd.DataFrame]` - Dictionary of feature DataFrames

### 3. candidate_selector.py

Selects trade candidates based on thresholds.

**Key Functions:**
- `select_candidates(panel, feats, extreme_pct, min_range_pct, min_vol_zscore, ...)` - Select trade candidates

**Parameters:**
- `extreme_pct` - Percentage of top/bottom performers (default: 0.05)
- `min_range_pct` - Minimum 12h high/low range percentage (default: 0.06)
- `min_vol_zscore` - Minimum volatility z-score threshold (default: 1.5)
- `metric` - Performance metric to rank by (default: "ret24h")
- `chop_thr` - Maximum choppiness score threshold (default: 0.5)

**Returns:** `Tuple[List[str], List[str]]` - (long_candidates, short_candidates)

### 4. data_fetcher.py

Fetches OHLCV data from the exchange.

**Key Functions:**
- `get_panel(symbols)` - Get OHLCV panel for symbols
- `fetch_incremental(symbol)` - Incrementally fetch new data

### 5. model_orchestrator.py

Orchestrates ML model inference.

**Key Functions:**
- `run_full_chain(symbol, side, features, panel)` - Run complete inference chain

### 6. trade_executor.py

Executes trades (live or shadow mode).

**Key Functions:**
- `execute_trade(symbol, side, size, bucket_key)` - Execute a trade
- `close_position(symbol)` - Close a position

## Process Flow

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  DataFetcher    │────▶│ FeatureGenerator │────▶│ CandidateSelector  │
│  (OHLCV data)  │     │  (30 features)   │     │ (filtered symbols) │
└─────────────────┘     └──────────────────┘     └────────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  TradeExecutor  │◀────│ ModelOrchestrator│◀────│ run_inference_step │
│  (orders)       │     │  (ML predictions)│     │                   │
└─────────────────┘     └──────────────────┘     └────────────────────┘
```

## Debugging Notes

### Common Errors

1. **`'str' object has no attribute 'empty'`**
   - **Root cause:** Shape mismatch between DataFrame and Series in candidate selection
   - **Location:** `candidates.py` line 142
   - **Fix:** Align shapes before comparison using `.align()` or broadcast Series to DataFrame

2. **Feature generation returns empty dictionary**
   - **Check:** Verify panel data is loaded correctly
   - **Check:** Verify symbols are in the panel

3. **Candidate selection returns empty lists**
   - **Check:** Verify feature thresholds are appropriate
   - **Check:** Verify features are computed correctly

### Testing Individual Components

```python
# Test feature generation
from extreme_price_movements.inference.feature_generator import generate_features
from extreme_price_movements.inference.data_fetcher import DataFetcher

fetcher = DataFetcher()
symbols = ['BTCUSDT', 'ETHUSDT']
panel = fetcher.get_panel(symbols)
feats = generate_features(panel, basket_syms=symbols)

# Test candidate selection
from extreme_price_movements.inference.candidate_selector import select_candidates

long_cands, short_cands = select_candidates(
    panel=panel,
    feats=feats,
    extreme_pct=0.05,
    min_range_pct=0.06,
    min_vol_zscore=1.5,
)
```

## Configuration

Configuration is loaded from:
- `config/inference_config.yaml` - Main inference configuration
- `extreme_price_movements/offline_optimisers/reports/tbm_best_params.csv` - Optimized parameters
