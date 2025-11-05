#!/bin/bash

# Script to call Sticky Finite HMM Regime Discovery Step directly

echo "🚀 Calling Sticky Finite HMM Regime Discovery Step Directly"
echo "========================================================"

# Set PYTHONPATH to include project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Change to the step directory
cd src/training/steps/market_analysis/sticky_finite_hmm_clustering/

# Examples of different calls:

echo "📊 Example 1: Basic usage (ETHUSDT 1h)"
python3 sticky_finite_hmm_regime_discovery_step.py --symbol ETHUSDT --exchange binance --timeframe 1h --execution-mode light

echo ""
echo "📊 Example 2: BTCUSDT with full mode"
python3 sticky_finite_hmm_regime_discovery_step.py --symbol BTCUSDT --exchange binance --timeframe 4h --execution-mode full

echo ""
echo "📊 Example 3: Fast execution without auto-tuning"
python3 sticky_finite_hmm_regime_discovery_step.py --symbol ETHUSDT --exchange binance --timeframe 1h --disable-auto-tuning
