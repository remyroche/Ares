#!/bin/bash

# Comprehensive Integration Test Runner for Enhanced Exit Strategy
# Runs all unit tests and generates summary report

export PYTHONPATH=/Users/remyroche/Documents/Ares:$PYTHONPATH

echo "================================================================================"
echo "ENHANCED EXIT STRATEGY - COMPREHENSIVE TEST SUITE"
echo "================================================================================"
echo "Start Time: $(date)"
echo ""

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Test 1: TradingOrchestrator Integration
echo "📋 Test Suite 1: TradingOrchestrator Integration"
echo "--------------------------------------------------------------------------------"
if python3 tests/trading/test_orchestrator_integration.py 2>&1 | tail -10 | grep -q "ALL TESTS PASSED"; then
    echo "✅ PASSED"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    echo "❌ FAILED"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))
echo ""

# Test 2: UncertaintyCalculator
echo "📋 Test Suite 2: UncertaintyCalculator"
echo "--------------------------------------------------------------------------------"
if python3 tests/trading/test_uncertainty_calculator.py 2>&1 | tail -10 | grep -q "ALL TESTS PASSED"; then
    echo "✅ PASSED"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    echo "❌ FAILED - Running detailed output..."
    python3 tests/trading/test_uncertainty_calculator.py 2>&1 | tail -50
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))
echo ""

# Test 3: PredictionCache
echo "📋 Test Suite 3: PredictionCache"
echo "--------------------------------------------------------------------------------"
if python3 tests/trading/test_prediction_cache.py 2>&1 | tail -10 | grep -q "ALL TESTS PASSED"; then
    echo "✅ PASSED"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    echo "❌ FAILED"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))
echo ""

# Final Summary
echo "================================================================================"
echo "FINAL TEST SUMMARY"
echo "================================================================================"
echo "Total Test Suites:  $TOTAL_TESTS"
echo "✅ Passed:          $PASSED_TESTS"
echo "❌ Failed:          $FAILED_TESTS"
echo "Success Rate:       $(echo "scale=1; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc)%"
echo ""

if [ $FAILED_TESTS -eq 0 ]; then
    echo "✅ ✅ ✅  ALL TEST SUITES PASSED  ✅ ✅ ✅"
    echo ""
    echo "The Enhanced Exit Strategy integration is VERIFIED and READY FOR DEPLOYMENT!"
    exit 0
else
    echo "❌ SOME TESTS FAILED - Review required"
    exit 1
fi

