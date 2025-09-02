# Testing Framework Setup Guide

## Overview
This guide provides a comprehensive approach to establishing a robust testing framework for the project.

## Current State Analysis

### Existing Test Files
- `test_analysis_dir/` contains only 3 test files
- Multiple test files scattered in root directory
- No clear test organization or coverage metrics
- Missing test configuration files

## Recommended Testing Stack

### Core Tools
1. **pytest** - Modern testing framework
2. **pytest-cov** - Coverage reporting
3. **pytest-mock** - Mocking support
4. **pytest-asyncio** - Async testing
5. **pytest-xdist** - Parallel test execution
6. **tox** - Testing across Python versions

### Additional Tools
- **hypothesis** - Property-based testing
- **pytest-benchmark** - Performance testing
- **pytest-timeout** - Test timeout management
- **pytest-env** - Environment variable management

## Project Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── pytest.ini               # Pytest configuration
├── .coveragerc              # Coverage configuration
├── unit/                    # Unit tests
│   ├── __init__.py
│   ├── test_utils/
│   ├── test_models/
│   ├── test_data_management/
│   └── test_monitoring/
├── integration/             # Integration tests
│   ├── __init__.py
│   ├── test_pipeline/
│   ├── test_api/
│   └── test_database/
├── e2e/                     # End-to-end tests
│   ├── __init__.py
│   └── test_workflows/
├── performance/             # Performance tests
│   ├── __init__.py
│   └── benchmarks/
└── fixtures/                # Test data and fixtures
    ├── __init__.py
    ├── data/
    └── mocks/
```

## Configuration Files

### pytest.ini
```ini
[pytest]
# Test discovery
python_files = test_*.py *_test.py
python_classes = Test* *Tests
python_functions = test_*

# Test paths
testpaths = tests
norecursedirs = .git .tox dist build *.egg

# Output options
addopts = 
    -v
    --strict-markers
    --tb=short
    --cov=src
    --cov-report=term-missing:skip-covered
    --cov-report=html
    --cov-report=xml
    --cov-fail-under=80

# Custom markers
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
    e2e: marks tests as end-to-end tests
    
# Logging
log_cli = true
log_cli_level = INFO
```

### .coveragerc
```ini
[run]
source = src
omit = 
    */tests/*
    */test_*
    */__init__.py
    */config/*
    */migrations/*

[report]
precision = 2
show_missing = True
skip_covered = False

[html]
directory = htmlcov

[xml]
output = coverage.xml
```

### conftest.py
```python
"""Shared test fixtures and configuration."""
import pytest
import logging
from pathlib import Path
from unittest.mock import Mock

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)

@pytest.fixture
def test_data_dir():
    """Return path to test data directory."""
    return Path(__file__).parent / "fixtures" / "data"

@pytest.fixture
def mock_logger():
    """Provide a mock logger for testing."""
    logger = Mock()
    logger.debug = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.critical = Mock()
    return logger

@pytest.fixture
def sample_trade_data():
    """Provide sample trade data for testing."""
    return {
        "symbol": "BTCUSDT",
        "price": 50000.0,
        "quantity": 0.1,
        "timestamp": 1234567890
    }

@pytest.fixture
def temp_workspace(tmp_path):
    """Create a temporary workspace for testing."""
    workspace = tmp_path / "test_workspace"
    workspace.mkdir()
    (workspace / "src").mkdir()
    (workspace / "data").mkdir()
    (workspace / "logs").mkdir()
    return workspace

@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests."""
    # Add singleton reset logic here
    pass

# Async fixtures
@pytest.fixture
async def async_client():
    """Provide an async test client."""
    # Initialize async client
    pass
```

## Test Templates

### Unit Test Template
```python
"""Unit tests for module_name."""
import pytest
from unittest.mock import Mock, patch

from src.module_name import ClassName


class TestClassName:
    """Test cases for ClassName."""
    
    @pytest.fixture
    def instance(self):
        """Create instance for testing."""
        return ClassName()
    
    def test_initialization(self, instance):
        """Test proper initialization."""
        assert instance is not None
        assert instance.attribute == expected_value
    
    def test_method_success(self, instance):
        """Test method with valid input."""
        result = instance.method(valid_input)
        assert result == expected_output
    
    def test_method_validation(self, instance):
        """Test method input validation."""
        with pytest.raises(ValueError):
            instance.method(invalid_input)
    
    @patch('src.module_name.external_dependency')
    def test_with_mock(self, mock_dep, instance):
        """Test with mocked dependency."""
        mock_dep.return_value = "mocked_value"
        result = instance.method_using_dep()
        assert result == "expected_with_mock"
        mock_dep.assert_called_once()
    
    @pytest.mark.parametrize("input,expected", [
        (1, 2),
        (2, 4),
        (3, 6),
    ])
    def test_parametrized(self, instance, input, expected):
        """Test with multiple input scenarios."""
        assert instance.double(input) == expected
```

### Integration Test Template
```python
"""Integration tests for feature_name."""
import pytest
from pathlib import Path

from src.pipeline import Pipeline
from src.data_loader import DataLoader


@pytest.mark.integration
class TestFeatureIntegration:
    """Integration tests for feature."""
    
    @pytest.fixture
    def pipeline(self, temp_workspace):
        """Create pipeline instance."""
        return Pipeline(workspace=temp_workspace)
    
    def test_full_pipeline_flow(self, pipeline, sample_data):
        """Test complete pipeline execution."""
        # Setup
        pipeline.load_data(sample_data)
        
        # Execute
        result = pipeline.run()
        
        # Verify
        assert result.status == "success"
        assert len(result.outputs) > 0
    
    def test_error_handling(self, pipeline):
        """Test pipeline error handling."""
        with pytest.raises(PipelineError):
            pipeline.run_with_invalid_config()
```

### Performance Test Template
```python
"""Performance benchmarks."""
import pytest
import numpy as np


@pytest.mark.benchmark
class TestPerformance:
    """Performance test cases."""
    
    def test_processing_speed(self, benchmark):
        """Benchmark data processing speed."""
        data = np.random.rand(10000, 100)
        
        def process():
            return np.mean(data, axis=0)
        
        result = benchmark(process)
        assert result.shape == (100,)
    
    @pytest.mark.slow
    def test_large_dataset(self, benchmark):
        """Test with large dataset."""
        # Large dataset test
        pass
```

## Testing Best Practices

### 1. Test Organization
- One test file per source file
- Mirror source directory structure
- Group related tests in classes
- Use descriptive test names

### 2. Test Coverage Goals
- Minimum 80% overall coverage
- 100% coverage for critical paths
- Focus on edge cases
- Test error conditions

### 3. Fixture Best Practices
- Keep fixtures focused and reusable
- Use fixture scope appropriately
- Avoid fixture interdependencies
- Document fixture purposes

### 4. Mocking Guidelines
- Mock external dependencies
- Don't mock what you're testing
- Verify mock interactions
- Use spec=True for interface checking

### 5. Assertion Best Practices
- One logical assertion per test
- Use descriptive assertion messages
- Test positive and negative cases
- Verify state changes

## Running Tests

### Basic Commands
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/unit/test_utils.py

# Run tests matching pattern
pytest -k "test_validation"

# Run marked tests
pytest -m unit
pytest -m "not slow"

# Run in parallel
pytest -n auto

# Run with verbose output
pytest -vv

# Generate HTML coverage report
pytest --cov=src --cov-report=html
```

### Continuous Integration
```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10]
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    
    - name: Run tests
      run: |
        pytest --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

## Test Data Management

### 1. Fixtures Directory
```
tests/fixtures/
├── data/
│   ├── sample_trades.json
│   ├── test_config.yaml
│   └── mock_responses.json
├── mocks/
│   ├── api_responses.py
│   └── database_mocks.py
└── factories/
    ├── trade_factory.py
    └── user_factory.py
```

### 2. Data Factory Example
```python
"""Factory for creating test data."""
from datetime import datetime
from typing import Dict, Any


class TradeFactory:
    """Factory for creating trade objects."""
    
    @staticmethod
    def create_trade(**kwargs) -> Dict[str, Any]:
        """Create a trade with defaults."""
        defaults = {
            "id": "test_trade_001",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "price": 50000.0,
            "quantity": 0.1,
            "timestamp": datetime.now().isoformat(),
            "status": "FILLED"
        }
        defaults.update(kwargs)
        return defaults
    
    @staticmethod
    def create_batch(count: int, **kwargs) -> List[Dict[str, Any]]:
        """Create multiple trades."""
        return [
            TradeFactory.create_trade(id=f"trade_{i}", **kwargs)
            for i in range(count)
        ]
```

## Migration Plan

### Phase 1: Setup (Week 1)
1. Create test directory structure
2. Install testing dependencies
3. Configure pytest and coverage
4. Create initial fixtures

### Phase 2: Migration (Week 2)
1. Move existing tests to new structure
2. Add missing unit tests
3. Create integration test suite
4. Set up CI pipeline

### Phase 3: Enhancement (Week 3)
1. Add performance tests
2. Implement property-based testing
3. Create end-to-end tests
4. Add mutation testing

### Phase 4: Maintenance
1. Monitor coverage metrics
2. Regular test review
3. Update test documentation
4. Performance baseline tracking

## Metrics and Reporting

### Coverage Goals
- Overall: 80%+
- Critical paths: 95%+
- New code: 90%+

### Test Performance
- Unit tests: < 10 seconds
- Integration tests: < 60 seconds
- E2E tests: < 5 minutes

### Quality Metrics
- Test flakiness: < 1%
- Test maintenance time: < 10% of dev time
- Bug escape rate: < 5%

## Resources

### Documentation
- [pytest documentation](https://docs.pytest.org/)
- [coverage.py documentation](https://coverage.readthedocs.io/)
- [Python testing best practices](https://docs.python-guide.org/writing/tests/)

### Tools
- [pytest plugins](https://pytest.org/en/latest/reference/plugin_list.html)
- [mutation testing with mutmut](https://mutmut.readthedocs.io/)
- [property-based testing with hypothesis](https://hypothesis.readthedocs.io/)