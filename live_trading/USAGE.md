# Live Trading Infrastructure - Usage Guide

## Overview

This live trading infrastructure provides a complete pipeline for production trading:

1. **BinanceDataFeed** - Fetches OHLCV data from Binance for margin-enabled assets passing universe filters
2. **OrderManagerV2** - Order execution with PortfolioManager integration and OCO support
3. **LiveTradingOrchestrator** - Main orchestration logic wiring everything together

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LiveTradingOrchestrator                          │
├─────────────────────────────────────────────────────────────────────┤
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐ │
│  │ BinanceDataFeed│  │ ModelOrchestrator│  │  OrderManagerV2      │ │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────────────┘ │
│          │                   │                   │                  │
│  ┌───────▼────────┐  ┌──────▼─────────┐  ┌──────▼────────────────┐ │
│  │ API Client     │  │ PortfolioManager│  │ Position Tracking    │ │
│  └────────────────┘  └────────────────┘  └───────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Environment Setup

```bash
export BINANCE_API_KEY="your_api_key"
export BINANCE_API_SECRET="your_api_secret"
```

### 2. Basic Usage

```python
import asyncio
from extreme_price_movements.live_trading import (
    LiveTradingOrchestrator,
    LiveTradingConfig,
    APIClient,
    TradingConfig,
)
from extreme_price_movements.portfolio_manager import PortfolioManager

# Load model bundle from artifacts
model_bundle = load_model_bundle("path/to/bundle.pkl")

# Load strategy parameters
strategy_exit_params = load_strategy_params("path/to/params.json")

# Configure
config = LiveTradingConfig(
    data_root=".",
    run_id="20260405_104105",  # Your run ID
    max_positions=4,
    max_portfolio_pct=0.30,
    max_position_usdt=5000.0,
    confidence_threshold=0.5,
)

# Create orchestrator
orchestrator = LiveTradingOrchestrator(
    config=config,
    model_bundle=model_bundle,
    strategy_exit_params=strategy_exit_params,
)

# Start trading
async def main():
    await orchestrator.initialize()
    await orchestrator.start()
    
    # Run for specified duration
    await asyncio.sleep(3600)  # 1 hour
    
    await orchestrator.stop()

asyncio.run(main())
```

## Components

### BinanceDataFeed

Fetches real-time OHLCV data from Binance:

- **Margin-enabled only**: Only trades symbols available on cross margin
- **Universe filtering**: Applies universe.py hardcoded exclusions
- **Quote deduplication**: Prefers USDT over other quote currencies
- **Automatic refresh**: Updates data every 60 seconds (configurable)

```python
from extreme_price_movements.live_trading import BinanceDataFeed, DataFeedConfig

config = DataFeedConfig(
    timeframe="15m",
    lookback_bars=200,
    update_interval_seconds=60.0,
    quotes=("USDT",),
)

data_feed = BinanceDataFeed(api_client, config)
await data_feed.initialize()
await data_feed.start()

# Get current data
panel = data_feed.get_panel()
btc_data = data_feed.get_symbol_data("BTC/USDT")
```

### OrderManagerV2

Executes trades with PortfolioManager integration:

- **Portfolio constraints**: Checks max positions, side limits, strategy limits
- **OCO orders**: Places entry + SL + TP orders together
- **Dynamic sizing**: Uses PortfolioManager's calculated position size cap
- **SL updates**: Monitors positions and updates stops (giveback/trailing)
- **Wallet monitoring**: Fetches real-time balance for position sizing

```python
from extreme_price_movements.live_trading import OrderManagerV2

order_mgr = OrderManagerV2(
    api_client=api_client,
    portfolio_manager=portfolio_mgr,
    config={
        "sl_mult": 1.0,
        "tp_mult": 3.0,
        "trail_mult": 0.25,
    }
)

# Place OCO order
result = await order_mgr.place_oco_order(
    symbol="BTC/USDT",
    side="long",
    strategy_id="long_tf",
    entry_price=65000.0,
    confidence_score=0.85,
    initial_threshold=0.5,
    params={"sl_mult": 1.0, "tp_mult": 3.0},
)
```

### LiveTradingOrchestrator

Main orchestration logic:

```python
# Two-stage feature generation
# 1. Mask features (lightweight) -> filter symbols
# 2. Full ML features (expensive) -> only for passing symbols

# Run inference cycle
await orchestrator._run_inference_cycle()

# Get status
status = orchestrator.get_status()
print(f"Portfolio: {status['portfolio_state']}")
print(f"Open positions: {status['open_positions']}")
```

## Integration with Existing Components

### PortfolioManager

The PortfolioManager from previous implementation is fully integrated:

```python
# Constraints enforced automatically:
- Max 4 positions
- Max 30% portfolio invested
- Dynamic threshold: initial + (n_positions * (1 - initial)) / 4
- 24h cooldown after losses
- Max 75% same side (3 long OR 3 short)
- Max 50% same strategy (2 per strategy, different assets)

# Position sizing formula:
position_size = min(requested, 30% - current_invested, 5000 USDT)
```

### Confidence Calibration

Uses isotonic regression calibration from `simple_position_sizer.py`:

```python
# Load calibration
from extreme_price_movements.simple_position_sizer import load_calibration_curves

calibration_data = load_calibration_curves(".", "run_id")

# In orchestrator:
# - Raw confidence scores are calibrated using learned curves
# - Only top 75% (by calibrated score) are considered for trading
# - Conflict resolution uses calibrated confidence ratios
```

### Strategy Acceptance

Only runs strategies accepted by `policy_optimiser.py`:

```python
# Filter loaded from:
{data_root}/artifacts/{run_id}/strategy_final_acceptation.json

# In orchestrator:
# - Only accepted strategy_ids are used for inference
# - Applied upstream before expensive feature generation
```

## Two-Stage Feature Generation

```python
# Stage 1: Mask features (fast, for all symbols)
mask_features = await orchestrator._generate_mask_features(panel, symbols)

# Apply masks -> filter to passing symbols only
passing_symbols = orchestrator._apply_masks(panel, mask_features, symbols)

# Stage 2: Full ML features (expensive, only for passing symbols)
full_features = await orchestrator._generate_full_features(panel, passing_symbols)

# Run inference only on passing symbols
for symbol in passing_symbols:
    result = orchestrator._run_inference(symbol, full_features[symbol])
```

This saves compute by:
- Not generating expensive features for symbols that won't pass masks
- Not running models on low-quality candidates

## Order Flow

```
1. Get data from Binance
   └── Fetch OHLCV for margin-enabled universe

2. Apply masks (two-stage features)
   ├── Generate mask features
   ├── Apply mask rules
   └── Filter to passing symbols

3. Run inference
   ├── Generate full ML features (only passing symbols)
   ├── Run ModelOrchestrator
   └── Get confidence scores

4. Apply calibration
   └── Calibrate raw scores using isotonic regression

5. Portfolio management
   ├── Check constraints (max positions, side limits, etc.)
   ├── Calculate dynamic threshold
   └── Determine position size

6. Execute trade
   ├── Place entry order (LIMIT)
   ├── Place OCO (SL + TP)
   └── Start position monitoring

7. Monitor until close
   ├── Check position every 60s
   ├── Update SL (giveback/trailing)
   └── Handle fills (SL/TP/entry)
```

## Safety Features

### Emergency Stop

```python
# Close all positions immediately
await orchestrator.emergency_stop()
```

### Daily Limits

```python
config = LiveTradingConfig(
    max_daily_trades=20,  # Max 20 trades per day
)
```

### Position Monitoring

- **Stop Loss Updates**: Automatic giveback and trailing logic
- **MFE Tracking**: Monitors max favorable excursion
- **Order Status**: Real-time order status updates

## Configuration

### LiveTradingConfig

```python
LiveTradingConfig(
    # Data
    timeframe="15m",              # Bar timeframe
    lookback_bars=200,            # Historical bars to fetch
    data_update_interval=60.0,    # Seconds between updates
    
    # Strategy
    data_root=".",               # Root for artifacts
    run_id=None,                  # Run ID for loading artifacts
    
    # Portfolio
    max_positions=4,              # Max simultaneous positions
    max_portfolio_pct=0.30,       # Max % of portfolio invested
    max_position_usdt=5000.0,     # Max absolute position size
    cooldown_hours=24.0,          # Cooldown after losses
    
    # Inference
    confidence_threshold=0.5,       # Minimum raw confidence
    use_calibration=True,          # Use isotonic calibration
    min_calibrated_confidence=0.75, # Top 75% threshold
    
    # Execution
    initial_threshold=0.5,         # Base entry threshold
    monitor_interval=60.0,         # Position monitoring interval
    
    # Safety
    max_daily_trades=20,           # Max trades per day
)
```

## Monitoring

### Status Check

```python
status = orchestrator.get_status()

{
    "running": True,
    "daily_trade_count": 3,
    "max_daily_trades": 20,
    "portfolio_state": {
        "n_positions": 2,
        "invested_pct": 0.15,
        "long_count": 1,
        "short_count": 1,
    },
    "open_positions": 2,
    "data_feed_symbols": 150,
    "calibration_loaded": 4,
    "strategies_accepted": 8,
}
```

### Callbacks

```python
# Register for signal events
orchestrator.register_signal_callback(
    lambda signal: print(f"Signal: {signal}")
)

# Register for trade events
orchestrator.register_trade_callback(
    lambda trade: print(f"Trade executed: {trade}")
)
```

## API Compatibility

The implementation is compatible with:

- **PortfolioManager** (previously created): Full integration for position constraints
- **inference_backtest.py**: Uses same calibration and strategy filtering logic
- **simple_position_sizer.py**: Loads calibration curves from OOF predictions
- **policy_optimiser.py**: Uses strategy_final_acceptation.json for filtering
- **model_orchestrator.py**: Uses for full inference chain

## Testing

```python
# Shadow mode (paper trading)
config = LiveTradingConfig(
    # ... same config
)

# Run without actual order placement
# (Modify OrderManagerV2 to skip actual API calls)
```

## Troubleshooting

### API Connection Issues

```python
# Check API credentials
import os
print(os.environ.get("BINANCE_API_KEY"))
print(os.environ.get("BINANCE_API_SECRET"))
```

### No Symbols Passing Masks

```python
# Check universe filtering
symbols = data_feed.get_trading_symbols()
print(f"Trading universe: {len(symbols)} symbols")

# Check margin availability
margin_pairs = fetch_binance_cross_margin_pairs()
print(f"Margin pairs: {len(margin_pairs)}")
```

### Low Confidence Scores

```python
# Check calibration loaded
calibration = load_calibration_curves(".", "run_id")
print(f"Calibration strategies: {len(calibration)}")

# Check strategy acceptance
acceptance_path = Path("artifacts/run_id/strategy_final_acceptation.json")
print(f"Acceptance file exists: {acceptance_path.exists()}")
```
