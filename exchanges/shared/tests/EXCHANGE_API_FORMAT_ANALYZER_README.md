# Exchange API Format Analyzer

## Overview

The `exchange_api_format_analyzer.py` script complements `enhanced_position_test_suite.py` by providing comprehensive testing and analysis of exchange API response formats. While the position test suite focuses on functional testing of trading operations, this analyzer focuses on:

1. **Systematic API Testing**: Tests all exchange API endpoints consistently across all supported exchanges
2. **Format Analysis**: Collects and analyzes response formats to identify differences
3. **Adapter Generation**: Provides recommendations and code suggestions for creating unified adapters

## Key Features

- **Multi-Exchange Testing**: Tests Binance, OKX, BingX, Gate.io, MEXC, Phemex
- **Response Collection**: Saves raw API responses for analysis
- **Format Comparison**: Identifies common fields, exchange-specific fields, and inconsistencies
- **Recommendations**: Generates actionable recommendations for standardization
- **Adapter Suggestions**: Provides code templates for adapters

## Usage

### Basic Usage (Mock Mode)

```bash
# Run analysis in mock mode (no API credentials needed)
python exchanges/shared/tests/exchange_api_format_analyzer.py --mode mock
```

### Real Exchange Testing

```bash
# Set environment variables for API credentials
export BINANCE_API_KEY=your_key
export BINANCE_API_SECRET=your_secret
export OKX_API_KEY=your_key
export OKX_API_SECRET=your_secret

# Run analysis with real exchange calls (uses testnet)
python exchanges/shared/tests/exchange_api_format_analyzer.py --mode real
```

### Custom Configuration

```bash
# Test specific exchanges
python exchanges/shared/tests/exchange_api_format_analyzer.py \
    --mode mock \
    --exchanges binance okx bingx

# Test with custom symbols
python exchanges/shared/tests/exchange_api_format_analyzer.py \
    --mode mock \
    --symbols BTCUSDT ETHUSDT ADAUSDT

# Custom output directory
python exchanges/shared/tests/exchange_api_format_analyzer.py \
    --mode mock \
    --output-dir my_analysis_results
```

## Output

The script generates several output files in the `exchange_format_analysis/` directory:

### 1. Response Samples (`samples/`)
- `{exchange}_{response_type}_samples.json`: Raw API responses organized by exchange and response type
- Contains timestamp, response time, and full response data

### 2. Analysis Report (`format_analysis_report.txt`)
- Human-readable summary of format differences
- Field-by-field analysis
- Standardization recommendations

### 3. JSON Report (`format_analysis_report.json`)
- Machine-readable format analysis
- Can be used programmatically for adapter generation
- Contains field mappings and recommendations

## Integration with enhanced_position_test_suite.py

### Complementary Roles

| Feature | enhanced_position_test_suite.py | exchange_api_format_analyzer.py |
|---------|-------------------------------|--------------------------------|
| **Focus** | Functional testing | Format analysis |
| **Tests** | Trading operations, positions, orders | API response formats |
| **Output** | Test results, pass/fail status | Format differences, adapter recommendations |
| **Use Case** | Verify exchange integration works | Ensure data format consistency |

### Combined Workflow

1. **Run Format Analyzer First**:
   ```bash
   python exchanges/shared/tests/exchange_api_format_analyzer.py --mode mock
   ```
   - Identifies format differences
   - Generates adapter recommendations

2. **Create/Update Adapters**:
   - Use recommendations from analysis report
   - Implement adapters based on field mappings

3. **Run Position Test Suite**:
   ```bash
   python exchanges/shared/tests/enhanced_position_test_suite.py --mode mock
   ```
   - Validates adapters work correctly
   - Tests end-to-end trading operations

## Analyzing Results

### Example Analysis Report Structure

```
TICKER
  Common fields (8): last_price, bid_price, ask_price, volume_24h, ...
  BINANCE:
    Total fields: 12
    Exchange-specific fields: 2
    Specific: base_volume, quote_volume
  OKX:
    Total fields: 10
    Exchange-specific fields: 1
    Specific: funding_rate
```

### Key Insights

1. **Common Fields**: Fields present in all exchanges - these are already standardized
2. **Exchange-Specific Fields**: Fields unique to an exchange - may need mapping
3. **Missing Fields**: Common fields missing from an exchange - need adapter logic
4. **Type Inconsistencies**: Same field with different types across exchanges

## Creating Adapters

Based on the analysis report, you can create adapters:

### Example: Ticker Adapter

```python
def adapt_ticker_response(raw_response: Dict[str, Any], exchange: str) -> Dict[str, Any]:
    """Adapt ticker response to unified format."""
    standardized = {
        'symbol': raw_response.get('symbol', ''),
        'last_price': _extract_price(raw_response, exchange),
        'bid_price': _extract_bid(raw_response, exchange),
        'ask_price': _extract_ask(raw_response, exchange),
        'volume_24h': _extract_volume(raw_response, exchange),
        'timestamp': datetime.now(timezone.utc)
    }
    
    # Exchange-specific mappings
    if exchange == 'binance':
        standardized['base_volume'] = raw_response.get('baseVolume')
        standardized['quote_volume'] = raw_response.get('quoteVolume')
    elif exchange == 'okx':
        standardized['funding_rate'] = raw_response.get('fundingRate')
    
    return standardized
```

## API Endpoints Tested

The analyzer tests the following endpoints:

1. **Market Data**:
   - `get_ticker()` - Current price information
   - `get_klines()` / `get_ohlcv()` - Historical candlestick data
   - `get_order_book()` - Order book depth

2. **Account Data**:
   - `get_balance()` - Account balance
   - `get_account_info()` - Account information

3. **Trading Data** (mock mode only):
   - `get_open_orders()` - Open orders
   - `get_positions()` - Current positions
   - `get_order_status()` - Order status

## Best Practices

1. **Run Regularly**: Run format analysis when:
   - Adding new exchanges
   - Exchange API updates
   - Before major releases

2. **Compare Over Time**: Track format changes to detect API breaking changes

3. **Use Real Mode Periodically**: While mock mode is faster, real mode provides actual API response analysis

4. **Review Recommendations**: Prioritize standardization recommendations based on:
   - Data criticality
   - Usage frequency
   - Impact on trading logic

## Troubleshooting

### No Exchanges Initialized
- Check API credentials are set correctly
- Verify exchange names match supported exchanges
- In mock mode, credentials are optional

### Missing Response Samples
- Check network connectivity
- Verify exchange API endpoints are accessible
- Review error messages in sample files

### Incomplete Analysis
- Ensure sufficient response samples collected
- Verify all exchanges responded successfully
- Check for API rate limiting

## Next Steps

After running the analyzer:

1. **Review the Report**: Understand format differences
2. **Prioritize Standardization**: Focus on frequently used endpoints
3. **Create Adapters**: Implement unified adapters based on recommendations
4. **Test with Position Suite**: Verify adapters work correctly
5. **Document Mappings**: Keep adapter documentation updated

## Integration with Existing Code

The analyzer integrates with:
- `ExchangeDispatcher`: Uses dispatcher for API calls
- `ExchangeInterface`: Follows interface definitions
- `UnifiedExchangeAdapter`: Compatible with existing adapters

## Contributing

When adding new exchanges:
1. Add exchange to `ExchangeType` enum
2. Ensure `ExchangeDispatcher` supports it
3. Run analyzer to generate format analysis
4. Create adapters based on recommendations
5. Test with position test suite
