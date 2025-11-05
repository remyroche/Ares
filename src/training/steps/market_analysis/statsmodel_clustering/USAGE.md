# Statsmodel Clustering Usage Guide

This guide provides comprehensive instructions for using the statsmodel clustering module.

## 🚀 Quick Start

### Installation

The required packages have been installed:
```bash
pip install numpy pandas scipy scikit-learn statsmodels
pip install optuna scikit-optimize
pip install networkx python-louvain
```

### Basic Usage

#### 1. Download Data

```bash
# Download data for ETHUSDT
python3 cli.py download --symbol ETHUSDT --exchange BINANCE --timeframe 1h --years 2

# Download with custom output
python3 cli.py download --symbol BTCUSDT --exchange BINANCE --timeframe 4h --years 1 --output btc_data.parquet
```

#### 2. Run Clustering Analysis

```bash
# Run clustering with 3 regimes
python3 cli.py cluster --symbol ETHUSDT --data-file data_cache/ETHUSDT_BINANCE_1h_clustering_data.parquet --regimes 3

# Run with custom PCA components
python3 cli.py cluster --symbol ETHUSDT --data-file data.parquet --regimes 4 --pca-components 15
```

#### 3. Run Complete Pipeline

```bash
# Run full pipeline (download + clustering)
python3 cli.py pipeline --symbol ETHUSDT --exchange BINANCE --timeframe 1h --years 2 --regimes 3

# Run with custom directories
python3 cli.py pipeline --symbol BTCUSDT --exchange BINANCE --timeframe 4h --years 1 --regimes 4 --data-dir custom_data --output-dir custom_outcomes
```

#### 4. Optimize Parameters

```bash
# Run parameter optimization
python3 cli.py optimize --symbol ETHUSDT --data-file data.parquet --trials 50

# Run with custom output directory
python3 cli.py optimize --symbol ETHUSDT --data-file data.parquet --trials 100 --output-dir optimization_results
```

## 📋 Command Reference

### Global Options

- `--verbose, -v`: Enable verbose output
- `--log-level`: Set log level (DEBUG, INFO, WARNING, ERROR)

### Download Command

```bash
python3 cli.py download [OPTIONS]
```

**Options:**
- `--symbol SYMBOL`: Trading symbol (default: ETHUSDT)
- `--exchange EXCHANGE`: Exchange name (default: BINANCE)
- `--timeframe TIMEFRAME`: Timeframe (default: 1h)
- `--years YEARS`: Years of historical data (default: 2)
- `--data-dir DATA_DIR`: Data directory (default: data_cache)
- `--force`: Force re-download even if data exists
- `--output OUTPUT`: Output file path (optional)

### Cluster Command

```bash
python3 cli.py cluster [OPTIONS]
```

**Options:**
- `--symbol SYMBOL`: Trading symbol (default: ETHUSDT)
- `--data-file DATA_FILE`: Input data file path (required)
- `--regimes REGIMES`: Number of regimes (default: 3)
- `--pca-components PCA_COMPONENTS`: PCA components (default: 12)
- `--output-dir OUTPUT_DIR`: Output directory (default: outcomes)
- `--config CONFIG`: Configuration file path (optional)

### Pipeline Command

```bash
python3 cli.py pipeline [OPTIONS]
```

**Options:**
- `--symbol SYMBOL`: Trading symbol (default: ETHUSDT)
- `--exchange EXCHANGE`: Exchange name (default: BINANCE)
- `--timeframe TIMEFRAME`: Timeframe (default: 1h)
- `--years YEARS`: Years of historical data (default: 2)
- `--regimes REGIMES`: Number of regimes (default: 3)
- `--data-dir DATA_DIR`: Data directory (default: data_cache)
- `--output-dir OUTPUT_DIR`: Output directory (default: outcomes)
- `--force-download`: Force re-download even if data exists
- `--config CONFIG`: Configuration file path (optional)

### Optimize Command

```bash
python3 cli.py optimize [OPTIONS]
```

**Options:**
- `--symbol SYMBOL`: Trading symbol (default: ETHUSDT)
- `--data-file DATA_FILE`: Input data file path (required)
- `--trials TRIALS`: Number of optimization trials (default: 50)
- `--output-dir OUTPUT_DIR`: Output directory (default: outcomes)
- `--config CONFIG`: Configuration file path (optional)

## 🔧 Configuration Files

You can use JSON configuration files to customize the clustering parameters:

```json
{
  "k_regimes": 3,
  "trend": "c",
  "order": 0,
  "switching_variance": true,
  "switching_trend": true,
  "maxiter": 100,
  "enable_pca": true,
  "pca_components": 12,
  "enable_scaling": true,
  "enable_diagnostics": true,
  "enable_hardware_optimization": true
}
```

Use with:
```bash
python3 cli.py cluster --symbol ETHUSDT --data-file data.parquet --config config.json
```

## 📊 Output Files

The module generates several output files:

### Data Files
- `{SYMBOL}_{EXCHANGE}_{TIMEFRAME}_clustering_data.parquet`: Downloaded market data

### Results Files
- `{SYMBOL}_clustering_results_{TIMESTAMP}.json`: Clustering results and metrics
- `{SYMBOL}_regime_labels_{TIMESTAMP}.csv`: Regime labels for each time point

### Example Output Structure
```
outcomes/
├── ETHUSDT_clustering_results_20251105_180000.json
├── ETHUSDT_regime_labels_20251105_180000.csv
└── ...

data_cache/
└── ETHUSDT_BINANCE_1h_clustering_data.parquet
```

## 🧪 Testing

Run the test suite to verify installation:

```bash
# Run basic tests
python3 simple_test.py

# Run comprehensive tests (requires full environment)
python3 test_implementation.py
```

## 📈 Features

### Data Downloading
- **Multiple Exchanges**: Support for BINANCE, BYBIT, OKX, KRAKEN
- **Flexible Timeframes**: 1m, 5m, 15m, 30m, 1h, 4h, 1d
- **Data Validation**: Comprehensive quality checks and validation
- **Caching**: Intelligent caching to avoid re-downloads
- **Gap Detection**: Automatic detection and reporting of data gaps

### Clustering Analysis
- **Markov Regression**: Advanced regime switching models
- **PCA Dimensionality Reduction**: Automatic feature reduction
- **Hardware Optimization**: Automatic hardware acceleration
- **Comprehensive Diagnostics**: Model validation and analysis
- **Multiple Regimes**: Support for 2-10 regimes

### Feature Engineering
- **Price Features**: Returns, log returns, price ratios
- **Volume Features**: Volume ratios and normalized volume
- **Volatility Features**: Rolling volatility and volatility ratios
- **Trend Features**: Moving averages and trend indicators

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'src'
   ```
   **Solution**: Run from the project root directory or ensure PYTHONPATH is set correctly.

2. **Data Download Fails**
   ```
   ⚠️ Core imports not available - cannot download data
   ```
   **Solution**: Check API keys and network connectivity.

3. **Memory Issues**
   ```
   MemoryError: Unable to allocate array
   ```
   **Solution**: Reduce `--years` parameter or use larger `--timeframe`.

### Debug Mode

Enable verbose logging for debugging:
```bash
python3 cli.py --verbose --log-level DEBUG cluster --symbol ETHUSDT --data-file data.parquet
```

## 🔗 Integration

### Python API

You can also use the module directly in Python:

```python
from src.training.steps.market_analysis.statsmodel_clustering.core import (
    create_data_downloader,
    create_enhanced_markov_regression_adapter
)

# Download data
config = {
    'symbol': 'ETHUSDT',
    'exchange': 'BINANCE',
    'timeframe': '1h',
    'lookback_years': 2
}
downloader = create_data_downloader(config)
success, data, error = await downloader.download_data()

# Run clustering
adapter = create_enhanced_markov_regression_adapter(k_regimes=3)
result = adapter.fit(data)
```

## 📚 Advanced Usage

### Custom Data Downloader

Create a custom data downloader by extending `BaseDataDownloader`:

```python
from src.training.steps.market_analysis.statsmodel_clustering.core import BaseDataDownloader

class CustomDataDownloader(BaseDataDownloader):
    async def download_data(self):
        # Implement custom download logic
        pass
    
    def validate_data(self, data):
        # Implement custom validation
        pass
```

### Custom Configuration

Use configuration files for complex setups:

```json
{
  "data_config": {
    "symbol": "ETHUSDT",
    "exchange": "BINANCE",
    "timeframe": "1h",
    "lookback_years": 2
  },
  "clustering_config": {
    "k_regimes": 4,
    "pca_components": 15,
    "enable_hardware_optimization": true
  }
}
```

## 📞 Support

For issues and questions:

1. Check the troubleshooting section above
2. Run the test suite to identify issues
3. Enable verbose logging for detailed error information
4. Check the main README.md for additional context

---

**Note**: This module requires proper API credentials for data downloading. Ensure you have valid exchange API keys configured.