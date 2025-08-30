# Cryptocurrency Price Movement Analysis Tool

A comprehensive tool for downloading and analyzing cryptocurrency data to calculate potential profits from different triple barrier methods and price movement patterns.

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
- **Price Movement Analysis:**
  - Total returns and volatility
  - Average daily and intraday price movements
  - Price change frequency and distribution
  - Movement size percentiles and patterns

- **Triple Barrier Profit Analysis:**
  - **Multiple barrier levels:** 0.3% to 1.5% in 0.1% increments (13 levels)
  - **Up/Down barriers:** Down barrier is half of up barrier (e.g., 1% up / 0.5% down)
  - **No time limit:** Trades continue until barrier is hit
  - **Entry signals:** Enter on price direction change
  - **Exit conditions:** Take profit at up barrier, stop loss at down barrier

- **Intraday Pattern Analysis:**
  - Hourly price movement patterns
  - Daily price movement patterns
  - Best trading hours identification
  - Movement size distribution by time

- **Movement Statistics:**
  - Price movement size distribution
  - Consecutive positive/negative runs
  - Small/medium/large movement frequencies
  - Movement percentiles (10th, 25th, 50th, 75th, 90th, 95th, 99th)

- **Comprehensive Reporting:**
  - Detailed barrier level comparisons
  - Top performers by barrier level
  - Average performance across all barriers
  - CSV export of results

- **Visualizations:**
  - Triple barrier profit analysis charts
  - Intraday price movement patterns
  - Movement distribution analysis
  - Volatility vs profit scatter plots

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
- `results/price_movement_metrics.csv` - Basic price movement metrics
- `results/triple_barrier_profits.csv` - Triple barrier profit analysis

### Visualizations
- `plots/triple_barrier_analysis.png` - Triple barrier profit analysis
- `plots/intraday_patterns.png` - Intraday price movement patterns

### Logs
- `data_download.log` - Download process logs
- `analysis.log` - Analysis process logs

## Analysis Metrics

### Price Movement Metrics
- **Total Return:** Overall price change over the period
- **Volatility:** Annualized standard deviation of returns
- **Average Daily Range:** Average daily high-low range
- **Average Intraday Movement:** Average daily price movement
- **Average Price Change:** Mean absolute price change per period
- **Price Change Standard Deviation:** Variability of price changes

### Triple Barrier Profit Metrics
- **Total Trades:** Number of completed trades
- **Average Profit:** Mean profit per trade
- **Win Rate:** Percentage of profitable trades
- **Profit Frequency:** Trades per period
- **Max Profit:** Maximum single trade profit
- **Total Potential Profit:** Cumulative profit across all trades
- **Average Trade Duration:** Mean time per trade (in 15-minute periods)
- **Take Profit Rate:** Percentage of trades that hit take profit
- **Stop Loss Rate:** Percentage of trades that hit stop loss

### Movement Statistics
- **Movement Percentiles:** 10th, 25th, 50th, 75th, 90th, 95th, 99th percentiles
- **Small Movements:** Percentage of movements ≤ 0.1%
- **Medium Movements:** Percentage of movements 0.1-1%
- **Large Movements:** Percentage of movements > 1%
- **Consecutive Runs:** Average length of positive/negative price runs

### Intraday Patterns
- **Best Trading Hours:** Hours with highest price movements
- **Peak Hours:** Hours with highest trading volume
- **Hourly Patterns:** Price movements and volatility by hour
- **Daily Patterns:** Price movements and volatility by day of week

## Triple Barrier Analysis Details

### Barrier Levels Tested
- **0.3% to 1.5%:** 13 barrier levels in 0.1% increments
- **Up Barriers:** 0.3%, 0.4%, 0.5%, 0.6%, 0.7%, 0.8%, 0.9%, 1.0%, 1.1%, 1.2%, 1.3%, 1.4%, 1.5%
- **Down Barriers:** Half of up barriers (0.15%, 0.2%, 0.25%, etc.)

### Trading Logic
- **Entry Signal:** Enter position when price direction changes (crosses previous close)
- **Long Position:** Buy when price goes up, take profit at up barrier, stop loss at down barrier
- **Short Position:** Sell when price goes down, take profit at down barrier, stop loss at up barrier
- **No Time Limit:** Trades continue until either barrier is hit
- **Perfect Execution:** Assumes no slippage, no fees, immediate execution

### Key Insights Provided
- **Profit Potential:** Average profit per trade for each barrier level
- **Win Rate Analysis:** Success rate for each barrier configuration
- **Trade Frequency:** How often trades occur for each barrier level
- **Asset Comparison:** Which assets offer the best profit opportunities
- **Risk Assessment:** Relationship between volatility and profit potential
- **Barrier Performance:** Which barrier levels work best for each asset

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

### Modifying Barrier Levels
Edit the barrier levels in `data_analyzer.py`:

```python
triple_barrier_profits = self.calculate_triple_barrier_profits(
    symbol_data, 
    barrier_levels=[0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015]  # 0.3% to 1.5%
)
```

### Customizing Analysis
You can modify the analysis parameters:

```python
# Change barrier levels
barrier_levels=[0.003, 0.007, 0.015, 0.025, 0.04]  # 0.3%, 0.7%, 1.5%, 2.5%, 4%

# Add more detailed movement analysis
movement_stats = self.calculate_movement_statistics(symbol_data)
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