# Centralized Logging System Guide

## Overview

The Ares project now uses a centralized logging system that ensures:
1. **All log outputs are in the `log/` directory**
2. **Each run has a centralized log file with datetime in filename**
3. **Consistent logging format across all modules**
4. **Proper log rotation and management**

## Key Features

### ✅ Centralized Configuration
- Single point of configuration for all logging
- Consistent formatting across all modules
- Thread-safe logging operations

### ✅ Organized Log Files
- All logs go to the `log/` directory
- Main log file: `ares_run_YYYYMMDD_HHMMSS.log`
- Module-specific logs: `ares_<module>_<suffix>_YYYYMMDD_HHMMSS.log`
- Automatic log rotation (10MB main, 5MB module-specific)

### ✅ Run-based Logging
- Each application run gets a unique timestamp-based ID
- All logs for a run are grouped by timestamp
- Easy to track and correlate logs from the same execution

## Usage

### Basic Usage

```python
from centralized_logging import get_logger

# Get a logger for your module
logger = get_logger(__name__)

# Use the logger
logger.info("This is an info message")
logger.warning("This is a warning")
logger.error("This is an error")
logger.debug("This is a debug message")
```

### Module-Specific Logging

For modules that need their own dedicated log files:

```python
from centralized_logging import add_module_logger

# Create a module-specific logger with dedicated log file
module_logger = add_module_logger("my_module", "validation")
module_logger.info("This goes to a separate log file")
```

### Getting Run Information

```python
from centralized_logging import get_run_id, get_log_file_path

# Get current run ID
run_id = get_run_id()  # e.g., "20250904_151526"

# Get main log file path
log_path = get_log_file_path()  # e.g., "log/ares_run_20250904_151526.log"
```

### Setting Log Level

```python
from centralized_logging import set_log_level
import logging

# Set to debug level for more verbose logging
set_log_level(logging.DEBUG)
```

## Migration from Old Logging

### Before (Old Way)
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("my_module.log"),  # ❌ Scattered files
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("MyModule")
```

### After (New Way)
```python
from centralized_logging import get_logger

logger = get_logger(__name__)  # ✅ Centralized, organized
```

## Log File Structure

```
log/
├── ares_run_20250904_151526.log              # Main centralized log
├── ares_analysis_validation_20250904_151526.log  # Module-specific log
├── ares_data_quality_20250904_151526.log     # Another module log
└── ... (other historical logs)
```

## Log Format

All logs use a consistent format:
```
YYYY-MM-DD HH:MM:SS - <logger_name> - <level> - <message>
```

Example:
```
2025-09-04 15:15:26 - __main__ - INFO - Testing centralized logging system
2025-09-04 15:15:26 - module1 - WARNING - This is a warning message
```

## Benefits

1. **Organization**: All logs in one place (`log/` directory)
2. **Traceability**: Each run has a unique timestamp-based ID
3. **Consistency**: Same format across all modules
4. **Maintenance**: Centralized configuration makes updates easy
5. **Performance**: Automatic log rotation prevents disk space issues
6. **Debugging**: Easy to correlate logs from the same execution

## Files Updated

The following files have been updated to use the centralized logging system:

- `comprehensive_analysis_demo.py`
- `comprehensive_analysis_core.py`
- `comprehensive_quality_auditor.py`
- `comprehensive_analysis_simplified.py`
- `comprehensive_professional_analysis.py`
- `data_quality/simple_quality_orchestrator.py`
- `crypto_analysis/data_analyzer.py`
- `crypto_analysis/data_downloader.py`

## Implementation Details

The centralized logging system is implemented in `centralized_logging.py` and provides:

- **Singleton pattern**: Ensures only one logging configuration per run
- **Thread safety**: Safe for concurrent logging operations
- **Automatic initialization**: Logging is set up on first import
- **Flexible configuration**: Supports both basic and module-specific logging
- **Log rotation**: Prevents log files from growing too large

## Best Practices

1. **Always use `get_logger(__name__)`** for module-specific loggers
2. **Use module-specific loggers** only when you need separate log files
3. **Don't configure logging manually** - let the centralized system handle it
4. **Use appropriate log levels** (DEBUG, INFO, WARNING, ERROR)
5. **Include context in log messages** for better debugging

## Troubleshooting

### Logs not appearing in log/ directory
- Ensure you're using `get_logger(__name__)` from `centralized_logging`
- Check that the `log/` directory exists and is writable

### Multiple log files for same run
- This is normal for module-specific loggers
- The main log file contains all messages
- Module-specific files contain only messages from that module

### Log rotation issues
- Logs are automatically rotated when they reach size limits
- Old logs are kept as `.1`, `.2`, etc. files
- Check disk space if rotation seems to be failing