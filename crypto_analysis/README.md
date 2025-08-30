# Cryptocurrency Trading Analysis Tool

A comprehensive tool for downloading and analyzing cryptocurrency data to compare scalping vs swing trading strategies across multiple assets.

## Features

### Data Downloader (`data_downloader.py`)
- Downloads 15-minute klines from Binance API for multiple cryptocurrencies
- Covers 2 years of historical data
- Supports 15 popular assets: ETH, ADA, ALGO, BTC, BNB, SOL, DOT, LINK, MATIC, AVAX, ATOM, UNI, LTC, XRP, BCH
- **Stores data in efficient Parquet format** with optimal compression (3-5x smaller than CSV)
- Includes data validation and integrity checks
- Rate limiting and error handling with fallback options
- Comprehensive logging and progress tracking

### Data Analyzer (`data_analyzer.py`)
- **Basic Metrics Analysis:**
  - Total returns and volatility
  - Volume analysis and patterns
  - Price range analysis
  - Daily high/low patterns

- **Intraday Pattern Analysis:**
  - Hourly volume patterns
  - Daily volume patterns
  - Peak trading hours identification
  - Volatility patterns by time of day

- **Strategy Simulation:**
  - **Scalping Strategy:** Quick trades with 0.5% take profit and 0.3% stop loss
  - **Swing Trading Strategy:** Longer-term trades with 5% take profit and 3% stop loss
  - Performance metrics: win rate, total return, Sharpe ratio, max drawdown

- **Comprehensive Reporting:**
  - Detailed performance comparison
  - Risk analysis
  - Top performers identification
  - CSV export of results

- **Visualizations:**
  - Strategy performance comparison charts
  - Intraday pattern analysis
  - Risk vs return scatter plots
  - Volume and volatility patterns

## Installation

1. Clone or download the project files
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Download Data
```bash
python data_downloader.py
```

This will:
- Download 2 years of 15-minute klines for all supported assets
- Save data to `data/crypto_15m_data_YYYYMMDD_HHMMSS.parquet`
- Display download progress and summary

### Step 2: Analyze Data
```bash
python data_analyzer.py
```

This will:
- Load the most recent data file
- Perform comprehensive analysis on all assets
- Generate detailed reports in console
- Create visualizations in `plots/` directory
- Save results to CSV files in `results/` directory

## Output Files

### Data Files
- `data/crypto_15m_data_*.parquet` - Raw OHLCV data in Parquet format (optimized for storage and performance)

### Analysis Results
- `results/basic_metrics.csv` - Basic asset metrics
- `results/scalping_results.csv` - Scalping strategy results
- `results/swing_results.csv` - Swing trading results

### Visualizations
- `plots/strategy_comparison.png` - Strategy performance comparison
- `plots/intraday_patterns.png` - Intraday trading patterns

### Logs
- `data_download.log` - Download process logs
- `analysis.log` - Analysis process logs

## Analysis Metrics

### Basic Metrics
- **Total Return:** Overall price change over the period
- **Volatility:** Annualized standard deviation of returns
- **Average Volume:** Mean trading volume
- **Volume Volatility:** Coefficient of variation of volume
- **Daily Range:** Average daily high-low range
- **Price Range:** Total price range over the period

### Strategy Metrics
- **Total Trades:** Number of completed trades
- **Win Rate:** Percentage of profitable trades
- **Average Profit:** Mean profit per trade
- **Total Return:** Cumulative strategy return
- **Max Drawdown:** Maximum peak-to-trough decline
- **Sharpe Ratio:** Risk-adjusted return measure

### Intraday Patterns
- **Peak Hours:** Hours with highest trading volume
- **Hourly Patterns:** Volume and volatility by hour
- **Daily Patterns:** Volume and volatility by day of week

## Strategy Details

### Scalping Strategy
- **Entry Signal:** Price increase > 0.5% in one period
- **Take Profit:** 0.5% profit target
- **Stop Loss:** 0.3% stop loss
- **Hold Time:** Until profit target or stop loss hit

### Swing Trading Strategy
- **Entry Signal:** Price increase > 2% over 4 periods (1 hour)
- **Take Profit:** 5% profit target
- **Stop Loss:** 3% stop loss
- **Hold Time:** Maximum 96 periods (24 hours) or until exit condition

## Supported Assets

The tool analyzes the following cryptocurrency pairs:
- ETHUSDT (Ethereum)
- ADAUSDT (Cardano)
- ALGOUSDT (Algorand)
- BTCUSDT (Bitcoin)
- BNBUSDT (Binance Coin)
- SOLUSDT (Solana)
- DOTUSDT (Polkadot)
- LINKUSDT (Chainlink)
- MATICUSDT (Polygon)
- AVAXUSDT (Avalanche)
- ATOMUSDT (Cosmos)
- UNIUSDT (Uniswap)
- LTCUSDT (Litecoin)
- XRPUSDT (Ripple)
- BCHUSDT (Bitcoin Cash)

## Customization

### Adding New Assets
Edit the `assets` list in `data_downloader.py`:
```python
assets = [
    'ETHUSDT', 'ADAUSDT', 'ALGOUSDT', 'BTCUSDT', 'BNBUSDT',
    'SOLUSDT', 'DOTUSDT', 'LINKUSDT', 'MATICUSDT', 'AVAXUSDT',
    'ATOMUSDT', 'UNIUSDT', 'LTCUSDT', 'XRPUSDT', 'BCHUSDT',
    'YOUR_NEW_ASSET'  # Add new assets here
]
```

### Modifying Strategy Parameters
Edit the strategy parameters in `data_analyzer.py`:

**Scalping Strategy:**
```python
scalping_results = self.simulate_scalping_strategy(
    symbol_data, 
    take_profit_pct=0.005,  # 0.5% take profit
    stop_loss_pct=0.003     # 0.3% stop loss
)
```

**Swing Strategy:**
```python
swing_results = self.simulate_swing_strategy(
    symbol_data,
    take_profit_pct=0.05,   # 5% take profit
    stop_loss_pct=0.03,     # 3% stop loss
    hold_periods=96         # 24 hours maximum
)
```

## Data Storage

### Parquet Format Benefits
- **Compression:** 3-5x smaller file sizes compared to CSV
- **Performance:** Faster read/write operations
- **Data Types:** Preserves exact data types (float64, int64, datetime)
- **Indexing:** Maintains datetime index for efficient time-based queries
- **Compatibility:** Works with pandas, Apache Arrow, and other data tools

### Data Structure
The Parquet files contain the following columns:
- `open_time` (datetime index): Timestamp of the kline
- `open`, `high`, `low`, `close` (float64): OHLC prices
- `volume` (float64): Trading volume
- `quote_asset_volume` (float64): Volume in quote currency
- `number_of_trades` (int64): Number of trades in the period
- `symbol` (string): Trading pair symbol

## Notes

- The tool uses Binance's public API with rate limiting to avoid being blocked
- Data is downloaded in chunks to handle large datasets efficiently
- **Raw data is stored in Parquet format for optimal storage and performance**
- Analysis includes realistic trading simulation with proper entry/exit logic
- Results are based on historical data and may not predict future performance
- Consider transaction costs and slippage in real trading scenarios

## Troubleshooting

### Common Issues

1. **API Rate Limiting:** The tool includes built-in rate limiting. If you encounter issues, increase the sleep time in `data_downloader.py`

2. **Missing Data:** Some assets might have gaps in data. The tool handles this gracefully and logs warnings

3. **Memory Issues:** For very large datasets, consider processing assets individually

4. **Visualization Errors:** Ensure matplotlib and seaborn are properly installed

### Log Files
Check the log files for detailed error information:
- `data_download.log` for download issues
- `analysis.log` for analysis issues

## License

This tool is provided for educational and research purposes. Use at your own risk for actual trading decisions.