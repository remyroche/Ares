# Exchange Interface Testing Suite

A comprehensive testing suite for validating ExchangeInterface functionality across different cryptocurrency exchanges.

## Features

- ✅ **Comprehensive Testing**: Tests all major ExchangeInterface operations
- 🔍 **Detailed Logging**: Uses tprint for rich, structured logging
- 🖥️ **CLI Interface**: Easy-to-use command-line interface
- 📊 **Performance Metrics**: Tracks response times and success rates
- 🔧 **Configurable**: Customizable test parameters and operations
- 📄 **JSON Output**: Export test results for analysis
- 🛡️ **Error Handling**: Robust error handling and reporting

## Test Operations

The testing suite validates the following operations:

1. **Connection Management**
   - Exchange connection establishment
   - Connection status monitoring
   - Disconnection handling

2. **Market Data Access**
   - Klines (candlestick data) download
   - Ticker data fetching
   - Order book retrieval
   - Recent trades data

3. **Account Management**
   - Balance fetching
   - Account information retrieval

4. **Order Management**
   - Open orders listing
   - Order status checking
   - Order creation (simulated)

5. **Position Management**
   - Position status checking
   - Position information retrieval

## Quick Start

### 1. Simple Test
```bash
# Run a quick test with default settings
python test_exchange.py
```

### 2. Full Test Suite
```bash
# Run the complete test suite
python exchange_interface_test_suite.py

# Run with specific exchange
python exchange_interface_test_suite.py --exchange binance

# Run specific operations only
python exchange_interface_test_suite.py --operations klines,balance,ticker

# Run with custom symbol and interval
python exchange_interface_test_suite.py --symbol ETHUSDT --interval 5m
```

## Command Line Options

### Basic Options
- `--exchange, -e`: Exchange type (simulated, binance, coinbase, kraken, bybit)
- `--symbol, -s`: Trading symbol to test (default: BTCUSDT)
- `--interval, -i`: Kline interval (default: 1m)
- `--quantity, -q`: Test quantity for orders (default: 0.001)

### API Configuration
- `--api-key`: API key for live exchange testing
- `--api-secret`: API secret for live exchange testing
- `--live, -l`: Use live exchange (default: testnet)

### Test Configuration
- `--operations, -o`: Comma-separated list of operations to test
- `--timeout, -t`: Timeout for operations in seconds (default: 30)
- `--verbose, -v`: Enable verbose output
- `--output, -f`: Output results to JSON file

## Usage Examples

### 1. Test Simulated Exchange
```bash
python exchange_interface_test_suite.py --exchange simulated --verbose
```

### 2. Test Binance Testnet
```bash
python exchange_interface_test_suite.py \
  --exchange binance \
  --symbol BTCUSDT \
  --operations klines,balance,ticker,orderbook \
  --verbose
```

### 3. Test Live Exchange (with API credentials)
```bash
python exchange_interface_test_suite.py \
  --exchange binance \
  --live \
  --api-key YOUR_API_KEY \
  --api-secret YOUR_API_SECRET \
  --symbol ETHUSDT \
  --interval 5m \
  --verbose
```

### 4. Export Results to JSON
```bash
python exchange_interface_test_suite.py \
  --exchange binance \
  --operations klines,balance,ticker \
  --output test_results.json \
  --verbose
```

### 5. Test Specific Operations
```bash
# Test only market data operations
python exchange_interface_test_suite.py --operations klines,ticker,orderbook

# Test only account operations
python exchange_interface_test_suite.py --operations balance,orders

# Test everything except positions
python exchange_interface_test_suite.py --operations connection,klines,balance,ticker,orderbook,trades,orders
```

## Test Results

The testing suite provides detailed results including:

- **Success/Failure Counts**: Total tests, passed, failed
- **Performance Metrics**: Response times for each operation
- **Error Details**: Specific error messages for failed tests
- **Data Samples**: Sample data returned from successful operations
- **Summary Statistics**: Overall success rate and duration

### Sample Output
```
🚀 Exchange Interface Test Suite Initialized
🔌 Testing Exchange Connection
✅ Connection successful (0.45s)
📊 Testing Klines Download
✅ Klines download successful (1.23s)
💰 Testing Balance Fetch
✅ Balance fetch successful (0.67s)
📈 Testing Ticker Data
✅ Ticker fetch successful (0.34s)

📊 Test Suite Summary
Total Tests: 4
Passed: 4
Failed: 0
Success Rate: 100.0%
Total Duration: 2.69s
```

## Configuration

### TestConfig Class
The testing suite uses a `TestConfig` class for configuration:

```python
@dataclass
class TestConfig:
    exchange_type: str = "simulated"
    test_symbol: str = "BTCUSDT"
    test_interval: str = "1m"
    test_quantity: float = 0.001
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    testnet: bool = True
    verbose: bool = True
    timeout: int = 30
    test_operations: List[str] = ["connection", "klines", "balance", ...]
```

### Exchange Configuration
The ExchangeInterface is configured with:

```python
exchange_config = {
    'exchange_type': config.exchange_type,
    'api_key': config.api_key,
    'api_secret': config.api_secret,
    'testnet': config.testnet,
    'rate_limits': {
        'requests_per_minute': 1200,
        'weight_per_minute': 6000
    }
}
```

## Error Handling

The testing suite includes comprehensive error handling:

- **Connection Errors**: Handles exchange connection failures
- **API Errors**: Manages API rate limits and errors
- **Data Validation**: Validates returned data structure and content
- **Timeout Handling**: Manages operation timeouts
- **Graceful Degradation**: Continues testing even if some operations fail

## Logging

The testing suite uses the `tprint` utility for rich logging:

- `tprint_info()`: General information messages
- `tprint_success()`: Success indicators
- `tprint_warning()`: Warning messages
- `tprint_error()`: Error messages
- `tprint_structured()`: Structured data display

## Integration

### Using in CI/CD
```bash
# Run tests in CI pipeline
python exchange_interface_test_suite.py \
  --exchange binance \
  --operations klines,balance \
  --output ci_results.json
```

### Using in Development
```python
from exchange_interface_test_suite import ExchangeInterfaceTestSuite, TestConfig

# Create custom test configuration
config = TestConfig(
    exchange_type="binance",
    test_symbol="ETHUSDT",
    test_operations=["klines", "balance"]
)

# Run tests programmatically
test_suite = ExchangeInterfaceTestSuite(config)
results = await test_suite.run_all_tests()
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure you're running from the project root directory
   - Check that all dependencies are installed

2. **Connection Failures**
   - Verify exchange type is correct
   - Check API credentials for live exchanges
   - Ensure network connectivity

3. **Data Validation Errors**
   - Some exchanges may return different data formats
   - Check symbol availability on the exchange
   - Verify interval is supported

4. **Timeout Errors**
   - Increase timeout value with `--timeout`
   - Check network latency
   - Verify exchange API status

### Debug Mode
Enable verbose output for detailed debugging:

```bash
python exchange_interface_test_suite.py --verbose
```

## Contributing

To add new test operations:

1. Add the operation name to the `test_operations` list
2. Implement the test method following the pattern `_test_{operation}()`
3. Add the operation to the argument parser choices
4. Update this README with the new operation

## License

This testing suite is part of the main project and follows the same license terms.