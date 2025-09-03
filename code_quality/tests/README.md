# Code Quality Tests

This directory contains tests for various code quality tools and utilities.

## Test Files

### test_common_operations.py
Comprehensive unit tests for the `src/utils/common_operations.py` module.

- **Test Classes**: 14
- **Test Methods**: 90+
- **Coverage**: All functions in common_operations module

## Running Tests

From the project root:
```bash
# Run all common operations tests
python code_quality/run_common_operations_tests.py

# Run with coverage report
python code_quality/run_common_operations_tests.py --coverage

# Run directly with unittest
python -m unittest code_quality.tests.test_common_operations -v
```

From the code_quality directory:
```bash
# Run the test runner
python run_common_operations_tests.py

# Run specific test class
python -m unittest tests.test_common_operations.TestDateTimeOperations -v
```

## Test Categories

1. **DateTime Operations** - Date/time formatting and parsing
2. **DataFrame Operations** - Pandas DataFrame utilities
3. **File Operations** - File I/O and directory management
4. **Parquet Operations** - Parquet file handling
5. **Async Operations** - Asynchronous utilities
6. **Validation Operations** - Data validation functions
7. **And more...**

## Adding New Tests

When adding tests for new utilities:
1. Create a new test file in this directory
2. Follow the naming convention: `test_<module_name>.py`
3. Use the existing test structure as a template
4. Update this README with the new test information