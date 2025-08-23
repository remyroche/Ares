# Step2 Command Usage Guide

## Quick Start

Use the new `step2` command to start the enhanced_training_pipeline from step2 with existing data:

```bash
python ares_launcher.py step2 --symbol ETHUSDT --exchange BINANCE
```

## What It Does

✅ **Validates existing data** from step1 and step1_5  
✅ **Shows detailed report** of data completeness  
✅ **Proceeds with existing data** - no new downloads  
✅ **Uses smart fallbacks** - comprehensive validator when available, simple file checker otherwise  
✅ **Warns about missing optional files** but continues if required files exist  
❌ **Fails gracefully** if essential data is missing  

## Prerequisites

Before using the `step2` command, ensure you have:

1. **Step1 data** in `data_cache/`:
   - `klines_BINANCE_ETHUSDT_1m_consolidated.parquet`
   - `klines_BINANCE_ETHUSDT_5m_consolidated.parquet` 
   - `aggtrades_BINANCE_ETHUSDT_consolidated.parquet`

2. **Step1_5 data** in `data_cache/`:
   - `processed_BINANCE_ETHUSDT_train.parquet`
   - `processed_BINANCE_ETHUSDT_validation.parquet`
   - `processed_BINANCE_ETHUSDT_test.parquet`

## Usage Examples

### Basic Usage
```bash
# Start from step2 with existing data
python ares_launcher.py step2 --symbol ETHUSDT --exchange BINANCE
```

### Specific Step2
```bash
# Start from specific step2
python ares_launcher.py step2 --symbol ETHUSDT --exchange BINANCE --step step2_feature_engineering
```

### Force Rerun
```bash
# Force rerun from step2 (clears progress)
python ares_launcher.py step2 --symbol ETHUSDT --exchange BINANCE --force-rerun
```

### With GUI
```bash
# Start step2 with GUI
python ares_launcher.py step2 --symbol ETHUSDT --exchange BINANCE --gui
```

## Expected Output

### Success Case
```
================================================================================
📊 DATA COMPLETENESS VALIDATION REPORT
🎯 Symbol: ETHUSDT
🏢 Exchange: BINANCE
================================================================================
📁 Step1 Data Collection: ✅ COMPLETE
   📄 Found 3 step1 files
🔄 Step1_5 Data Converter: ✅ COMPLETE
   📄 Found 3 step1_5 files

✅ READY TO START FROM STEP2
   Proceeding with existing data...
================================================================================
```

### Warning Case
```
⚠️  WARNINGS:
   • Data gaps detected: Missing 1m klines data

🕳️  DATA GAPS:
   • Missing 1m klines data

💡 RECOMMENDATIONS:
   • Consider running data collection to fill gaps, but proceeding with existing data
```

### Failure Case
```
❌ Cannot start from step2 - missing required data
Please run step1 and step1_5 first to collect and process data
```

## Troubleshooting

### Missing Data
If you get "missing required data" errors:

1. **Run step1 first**:
   ```bash
   python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step1_data_collection
   ```

2. **Run step1_5 second**:
   ```bash
   python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step1_5_data_converter
   ```

3. **Then try step2**:
   ```bash
   python ares_launcher.py step2 --symbol ETHUSDT --exchange BINANCE
   ```

### Data Gaps
If you see warnings about data gaps:

- **Review the gaps** in the validation report
- **Consider running data collection** to fill missing data
- **Proceed anyway** if the gaps are acceptable for your use case

### Validation Failures
If the validator fails to import:

- The system will **fall back to basic file checks**
- **Check that data_cache/ directory exists**
- **Verify file permissions** on data files

## Benefits

### Time Savings
- **No re-downloading** of existing data
- **Faster iteration** for development and testing
- **Efficient resource usage**

### Better Workflow
- **Clear feedback** on data status
- **Flexible execution** with partial data
- **Detailed reporting** for debugging

### Reliability
- **Validation before execution** prevents failures
- **Graceful error handling** with clear messages
- **Fallback mechanisms** for edge cases

## Comparison with Other Commands

| Command | Purpose | Data Source | Validation |
|---------|---------|-------------|------------|
| `step2` | Start from step2 | Existing data only | Comprehensive |
| `blank` | Quick training | Downloads if needed | Basic |
| `full` | Full training | Downloads if needed | Basic |
| `load` | Data collection only | Downloads missing | None |

## Tips

1. **Use `step2` for development** - faster iteration with existing data
2. **Use `blank` for testing** - quick training with minimal data
3. **Use `full` for production** - complete training with all data
4. **Use `load` for data collection** - download and prepare data

## Support

If you encounter issues:

1. **Check the validation report** for specific data issues
2. **Review the logs** in the `log/` directory
3. **Verify data files** exist in `data_cache/`
4. **Run data collection** if essential files are missing