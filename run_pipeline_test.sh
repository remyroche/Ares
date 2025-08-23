#!/bin/bash

# Test Script for Step1, Step1_5, and Step2 Pipeline with Ares Launcher
# This script demonstrates how to use ares_launcher to test the pipeline

set -e  # Exit on any error

echo "🚀 Starting Pipeline Test with Ares Launcher"
echo "=============================================="

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data_cache
mkdir -p data/training
mkdir -p log

# Set environment variables for testing
export BLANK_TRAINING_MODE=1
export FULL_TRAINING_MODE=0
export FORCE=1

echo "🔧 Environment setup completed"
echo "   - BLANK_TRAINING_MODE=1"
echo "   - FULL_TRAINING_MODE=0"
echo "   - FORCE=1"

# Test 1: Run step1_data_collection
echo ""
echo "🧪 Test 1: Step1 Data Collection"
echo "--------------------------------"
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --step step1_data_collection --force-rerun

if [ $? -eq 0 ]; then
    echo "✅ Step1 test completed successfully"
else
    echo "❌ Step1 test failed"
    exit 1
fi

# Test 2: Run step1_5_data_converter
echo ""
echo "🧪 Test 2: Step1.5 Data Converter"
echo "---------------------------------"
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --step step1_5_data_converter --force-rerun

if [ $? -eq 0 ]; then
    echo "✅ Step1.5 test completed successfully"
else
    echo "❌ Step1.5 test failed"
    exit 1
fi

# Test 3: Run step2_feature_engineering
echo ""
echo "🧪 Test 3: Step2 Feature Engineering"
echo "-----------------------------------"
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --step step2_feature_engineering --force-rerun

if [ $? -eq 0 ]; then
    echo "✅ Step2 test completed successfully"
else
    echo "❌ Step2 test failed"
    exit 1
fi

# Test 4: Run complete pipeline from step1
echo ""
echo "🧪 Test 4: Complete Pipeline (Step1 -> Step1.5 -> Step2)"
echo "--------------------------------------------------------"
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --step step1_data_collection --force-rerun

if [ $? -eq 0 ]; then
    echo "✅ Complete pipeline test completed successfully"
else
    echo "❌ Complete pipeline test failed"
    exit 1
fi

# Test 5: Run blank training mode (which includes all steps)
echo ""
echo "🧪 Test 5: Blank Training Mode"
echo "-----------------------------"
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --force-rerun

if [ $? -eq 0 ]; then
    echo "✅ Blank training test completed successfully"
else
    echo "❌ Blank training test failed"
    exit 1
fi

# Validate outputs
echo ""
echo "🔍 Validating Pipeline Outputs"
echo "-----------------------------"

# Check step1 outputs
echo "Checking Step1 outputs..."
if [ -f "data_cache/klines_BINANCE_ETHUSDT_1m_consolidated.parquet" ] && [ -f "data_cache/aggtrades_BINANCE_ETHUSDT_consolidated.parquet" ]; then
    echo "✅ Step1 outputs found"
else
    echo "❌ Step1 outputs missing"
    exit 1
fi

# Check step1_5 outputs
echo "Checking Step1.5 outputs..."
if [ -f "data_cache/unified_BINANCE_ETHUSDT_1m.parquet" ] && [ -f "data_cache/unified_BINANCE_ETHUSDT_1m_config.json" ]; then
    echo "✅ Step1.5 outputs found"
else
    echo "❌ Step1.5 outputs missing"
    exit 1
fi

# Check step2 outputs
echo "Checking Step2 outputs..."
if [ -f "data/training/features_BINANCE_ETHUSDT_train.parquet" ] && [ -f "data/training/features_BINANCE_ETHUSDT_val.parquet" ] && [ -f "data/training/features_BINANCE_ETHUSDT_test.parquet" ]; then
    echo "✅ Step2 outputs found"
else
    echo "❌ Step2 outputs missing"
    exit 1
fi

echo ""
echo "=============================================="
echo "🎉 ALL TESTS PASSED! Pipeline is working correctly."
echo "=============================================="

# Show file sizes for verification
echo ""
echo "📊 Generated File Sizes:"
echo "-----------------------"
if [ -f "data_cache/klines_BINANCE_ETHUSDT_1m_consolidated.parquet" ]; then
    echo "Klines data: $(du -h data_cache/klines_BINANCE_ETHUSDT_1m_consolidated.parquet | cut -f1)"
fi
if [ -f "data_cache/aggtrades_BINANCE_ETHUSDT_consolidated.parquet" ]; then
    echo "Aggtrades data: $(du -h data_cache/aggtrades_BINANCE_ETHUSDT_consolidated.parquet | cut -f1)"
fi
if [ -f "data_cache/unified_BINANCE_ETHUSDT_1m.parquet" ]; then
    echo "Unified data: $(du -h data_cache/unified_BINANCE_ETHUSDT_1m.parquet | cut -f1)"
fi
if [ -f "data/training/features_BINANCE_ETHUSDT_train.parquet" ]; then
    echo "Training features: $(du -h data/training/features_BINANCE_ETHUSDT_train.parquet | cut -f1)"
fi

echo ""
echo "✅ Pipeline test completed successfully!"