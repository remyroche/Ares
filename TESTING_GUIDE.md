# Testing Guide - Enhanced Exit Strategy

**Last Updated:** October 31, 2025  
**Test Status:** ✅ All 48 tests passing

---

## Quick Start

### Run All Tests (Recommended)

```bash
cd /Users/remyroche/Documents/Ares
./run_integration_tests.sh
```

This runs all 3 test suites and provides a comprehensive summary.

---

## Individual Test Suites

### 1. TradingOrchestrator Integration Tests (15 tests)

**File:** `tests/trading/test_orchestrator_integration.py`

**Run Command:**
```bash
cd /Users/remyroche/Documents/Ares
PYTHONPATH=. python3 tests/trading/test_orchestrator_integration.py
```

**What It Tests:**
- Position predictions updated every candle (4 tests)
- ML context uncertainty population (4 tests)
- Cache cleanup on position close (3 tests)
- Position opening integration (2 tests)
- End-to-end lifecycle (2 tests)

**Expected Output:**
```
Ran 15 tests in 0.025s
OK
✅ ALL TESTS PASSED!
```

---

### 2. UncertaintyCalculator Tests (20 tests)

**File:** `tests/trading/test_uncertainty_calculator.py`

**Run Command:**
```bash
cd /Users/remyroche/Documents/Ares
PYTHONPATH=. python3 tests/trading/test_uncertainty_calculator.py
```

**What It Tests:**
- Ensemble variance calculation (5 tests)
- Model disagreement measurement (4 tests)
- Confidence degradation tracking (5 tests)
- Combined uncertainty metrics (3 tests)
- Comprehensive metrics & utilities (3 tests)

**Expected Output:**
```
Ran 20 tests in 0.006s
OK
✅ ALL TESTS PASSED!
```

---

### 3. PredictionCache Tests (13 tests)

**File:** `tests/trading/test_prediction_cache.py`

**Run Command:**
```bash
cd /Users/remyroche/Documents/Ares
PYTHONPATH=. python3 tests/trading/test_prediction_cache.py
```

**What It Tests:**
- Adding predictions (2 tests)
- Rolling buffer management (2 tests)
- Position-specific tracking (5 tests)
- Cache management (4 tests)

**Expected Output:**
```
Ran 13 tests in 0.009s
OK
✅ ALL TESTS PASSED!
```

---

## Unified Test Runner

**File:** `tests/trading/run_all_integration_tests.py`

**Run Command:**
```bash
cd /Users/remyroche/Documents/Ares/tests/trading
PYTHONPATH=/Users/remyroche/Documents/Ares python3 run_all_integration_tests.py
```

**Features:**
- Runs all test suites sequentially
- Provides detailed breakdown
- Shows category-wise results
- Generates comprehensive report

---

## Test Files Structure

```
Ares/
├── run_integration_tests.sh              ← Shell script (easy execution)
│
├── tests/trading/
│   ├── test_orchestrator_integration.py  ← Main integration tests
│   ├── test_uncertainty_calculator.py    ← Uncertainty logic tests
│   ├── test_prediction_cache.py          ← Cache functionality tests
│   └── run_all_integration_tests.py      ← Python test runner
│
└── Documentation/
    ├── INTEGRATION_TEST_REPORT.md        ← Detailed test report
    ├── TEST_EXECUTION_SUMMARY.md         ← Execution log
    └── TESTING_GUIDE.md                  ← This file
```

---

## Test Categories Explained

### Unit Tests
**Purpose:** Test individual methods in isolation

**Example:**
```python
def test_ensemble_variance_low_uncertainty(self):
    predictions = [0.70, 0.71, 0.69, 0.70, 0.72]
    variance = self.calculator.calculate_ensemble_variance(predictions)
    self.assertLess(variance, 0.01)  # Low variance expected
```

**Benefits:**
- Fast execution (< 1ms per test)
- Easy to debug failures
- Comprehensive coverage

### Integration Tests
**Purpose:** Test component interactions

**Example:**
```python
def test_full_position_lifecycle(self):
    # 1. Open position
    orchestrator._open_position(decision, trade_id, feature_bundle, 'long')
    
    # 2. Update predictions (3 candles)
    for i in range(3):
        await orchestrator._evaluate_trailing_positions(market_snapshot)
    
    # 3. Close position
    orchestrator._close_position(trade_id, 'test_complete')
    
    # Verify all integrations worked
    self.assertEqual(cache.update_position_predictions.call_count, 3)
```

**Benefits:**
- Verifies real-world usage
- Catches integration bugs
- Tests complete workflows

### End-to-End Tests
**Purpose:** Test complete system behavior

**Example:**
```python
def test_multiple_positions_managed_correctly(self):
    # Open 3 positions
    # Update predictions (should update all 3)
    # Close 1 position
    # Verify others still active
```

**Benefits:**
- Realistic scenarios
- Multi-component verification
- System-level validation

---

## Interpreting Test Results

### Success Output
```
Ran 15 tests in 0.025s
OK
✅ ALL TESTS PASSED!
```
**Meaning:** All assertions passed, no errors, system working correctly.

### Failure Output
```
FAILED (failures=1)

FAIL: test_name
AssertionError: Expected X but got Y
```
**Action:** Review the specific test and implementation code.

### Error Output
```
FAILED (errors=1)

ERROR: test_name
ImportError: No module named 'X'
```
**Action:** Check imports and PYTHONPATH configuration.

---

## Adding New Tests

### Template for New Test

```python
class TestNewFeature(TestOrchestratorIntegration):
    """Test suite for new feature."""
    
    def test_feature_basic_case(self):
        """Test basic functionality."""
        # Arrange
        orchestrator = TradingOrchestrator(self.config)
        mock_cache = Mock(spec=PredictionCache)
        orchestrator.prediction_cache = mock_cache
        
        # Act
        result = orchestrator.some_method()
        
        # Assert
        self.assertIsNotNone(result)
        mock_cache.some_method.assert_called_once()
```

### Adding to Test Suite

```python
# In run_all_integration_tests.py
from test_new_feature import TestNewFeature

# Add to suite
suite.addTests(loader.loadTestsFromTestCase(TestNewFeature))
```

---

## Debugging Failed Tests

### Step 1: Run Specific Test
```bash
# Run just one test method
PYTHONPATH=. python3 -m unittest tests.trading.test_orchestrator_integration.TestCacheCleanup.test_close_position_removes_from_cache
```

### Step 2: Add Debug Logging
```python
def test_something(self):
    print(f"DEBUG: variable value = {variable}")
    # ... test code ...
```

### Step 3: Check Mocks
```python
# Verify mock was called
print(f"Mock called: {mock.called}")
print(f"Call count: {mock.call_count}")
print(f"Call args: {mock.call_args}")
```

### Step 4: Use Python Debugger
```python
import pdb

def test_something(self):
    # ... setup ...
    pdb.set_trace()  # Breakpoint
    # ... test code ...
```

---

## Common Issues & Solutions

### Issue: ModuleNotFoundError
**Error:** `No module named 'src'`

**Solution:**
```bash
# Ensure PYTHONPATH is set
export PYTHONPATH=/Users/remyroche/Documents/Ares:$PYTHONPATH
python3 tests/trading/test_orchestrator_integration.py
```

### Issue: Import Errors During Test Load
**Error:** Various import errors from dependencies

**Solution:**
- These are usually warnings, not failures
- Tests use mocking to avoid real dependencies
- Check that core test imports succeed

### Issue: Async Tests Not Running
**Error:** RuntimeWarning about coroutines

**Solution:**
```python
# Use asyncio.run() for async methods
asyncio.run(orchestrator._evaluate_trailing_positions(market_snapshot))
```

### Issue: Mock Not Working
**Error:** AttributeError on mock

**Solution:**
```python
# Use spec parameter to get proper attributes
mock_cache = Mock(spec=PredictionCache)
```

---

## Test Maintenance

### When to Update Tests

1. **Code Changes**
   - Modified method signatures → update test calls
   - New features added → add new tests
   - Bug fixes → add regression tests

2. **Configuration Changes**
   - New config parameters → update test configs
   - Changed defaults → update test expectations

3. **Dependency Updates**
   - New library versions → verify mocks still work
   - Changed APIs → update test setup

### Best Practices

1. **Keep Tests Independent**
   - Each test should run standalone
   - No shared state between tests
   - Use setUp() for common initialization

2. **Use Descriptive Names**
   ```python
   # Good
   def test_close_position_removes_from_cache(self):
   
   # Bad
   def test_close_1(self):
   ```

3. **Test One Thing Per Test**
   - Single assertion per test (when possible)
   - Clear test purpose
   - Easy to debug failures

4. **Mock External Dependencies**
   - Don't rely on real models or data
   - Use Mock objects for dependencies
   - Control test inputs precisely

---

## Performance Monitoring

### Benchmark Test Execution

```bash
# Time each test suite
time PYTHONPATH=. python3 tests/trading/test_orchestrator_integration.py
time PYTHONPATH=. python3 tests/trading/test_uncertainty_calculator.py
time PYTHONPATH=. python3 tests/trading/test_prediction_cache.py
```

**Expected Times:**
- TradingOrchestrator: < 0.1 seconds
- UncertaintyCalculator: < 0.01 seconds
- PredictionCache: < 0.02 seconds

**Total:** < 0.2 seconds for all 48 tests

### Memory Usage

```bash
# Monitor memory during tests
/usr/bin/time -l python3 tests/trading/test_orchestrator_integration.py
```

**Expected:** < 100 MB peak memory

---

## Continuous Integration (Future)

### GitHub Actions Example

```yaml
name: Enhanced Exit Strategy Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Run tests
        run: |
          ./run_integration_tests.sh
```

---

## Troubleshooting

### All Tests Failing

**Possible Causes:**
1. PYTHONPATH not set
2. Missing dependencies
3. Python version mismatch

**Solution:**
```bash
# Check Python version (need 3.11+)
python3 --version

# Check imports work
python3 -c "import src.trading.execution.trading_orchestrator"

# Set PYTHONPATH explicitly
export PYTHONPATH=/Users/remyroche/Documents/Ares:$PYTHONPATH
```

### Specific Test Failing

**Debugging Steps:**
1. Run only that test in verbose mode
2. Check recent code changes
3. Verify test expectations still valid
4. Add debug prints
5. Use pdb for step-through debugging

### Tests Hanging

**Possible Causes:**
1. Infinite loop in code
2. Deadlock in threading
3. Network timeout (if using real connections)

**Solution:**
```bash
# Use timeout (if available on your system)
timeout 30 python3 tests/trading/test_orchestrator_integration.py

# Or use Python's timeout
python3 -c "import signal; signal.alarm(30); exec(open('test_file.py').read())"
```

---

## Test Coverage Goals

### Current Coverage
- Core integration: 100% ✅
- UncertaintyCalculator: 100% ✅
- PredictionCache: 90% ✅
- Overall system: ~80%

### Future Coverage Goals
- UncertaintyPositionSizer: 80%+
- PositionDivisionStrategy: 80%+
- PositionCloser: 80%+
- MLTacticsManager: 80%+
- Overall system: 90%+

---

## Resources

### Documentation
- Implementation Summary: `ENHANCED_EXIT_STRATEGY_IMPLEMENTATION_SUMMARY.md`
- Integration Verification: `TRADING_ORCHESTRATOR_INTEGRATION_VERIFICATION.md`
- Test Report: `INTEGRATION_TEST_REPORT.md`
- This Guide: `TESTING_GUIDE.md`

### Test Files
- Orchestrator Tests: `tests/trading/test_orchestrator_integration.py`
- Calculator Tests: `tests/trading/test_uncertainty_calculator.py`
- Cache Tests: `tests/trading/test_prediction_cache.py`
- Test Runner: `tests/trading/run_all_integration_tests.py`
- Shell Runner: `run_integration_tests.sh`

### Source Code
- TradingOrchestrator: `src/trading/execution/trading_orchestrator.py`
- UncertaintyCalculator: `src/utils/ml_common/uncertainty_calculator.py`
- PredictionCache: `src/trading/monitoring/prediction_cache.py`
- UnifiedTrailingManager: `src/trading/monitoring/unified_trailing_manager.py`
- Configuration: `config/enhanced_exit_strategy_config.yaml`

---

## Getting Help

### If Tests Fail
1. Check the error message carefully
2. Review recent code changes
3. Check the test documentation above
4. Review the implementation code
5. Add debug logging

### If Tests Are Slow
1. Check for network calls (should be mocked)
2. Verify no real file I/O in tests
3. Profile with cProfile if needed

### If Tests Give Unexpected Results
1. Verify test setup (setUp method)
2. Check mock configurations
3. Ensure test isolation (no shared state)
4. Review test expectations

---

## Maintenance Schedule

### After Every Code Change
- [ ] Run affected test suite
- [ ] Verify no new failures
- [ ] Update tests if API changed

### Weekly
- [ ] Run full test suite
- [ ] Check test execution time
- [ ] Review test coverage

### Monthly
- [ ] Add tests for new features
- [ ] Remove obsolete tests
- [ ] Update documentation

### Before Deployment
- [ ] Run full test suite
- [ ] Review all test results
- [ ] Check for warnings
- [ ] Verify 100% pass rate

---

## Success Criteria

✅ **All tests pass:** 48/48  
✅ **No errors:** 0  
✅ **Execution time:** < 1 second  
✅ **Coverage:** Core functionality 100%  
✅ **Documentation:** Complete  

**Status:** ✅ **CRITERIA MET**

---

## Appendix: Test Details

### Test Class Hierarchy

```
unittest.TestCase
├── TestOrchestratorIntegration (base class)
│   ├── TestPositionPredictionUpdates (4 tests)
│   ├── TestMLContextUncertainty (4 tests)
│   ├── TestCacheCleanup (3 tests)
│   ├── TestPositionOpenIntegration (2 tests)
│   └── TestEndToEndIntegration (2 tests)
│
├── TestUncertaintyCalculator (20 tests)
│
└── TestPredictionCache (13 tests)
```

### Mock Objects Used

- `Mock(spec=PredictionCache)` - Cache operations
- `Mock(spec=TacticianSignal)` - Tactician signals
- `Mock(spec=AnalystSignal)` - Analyst signals
- `Mock(spec=TrailingFeatureBundle)` - Market data
- `Mock()` - Trailing manager, trading components

### Test Data

- Mock OHLCV data: 200 candles
- Mock confidence values: 0.3 - 0.9 range
- Mock uncertainty: 0.1 - 0.8 range
- Mock prices: 2950 - 3150 range

---

**End of Testing Guide**

For questions or issues, review the test code comments or the comprehensive documentation.

