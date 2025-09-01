#!/bin/bash

# Updated Ares 21-Step Training Pipeline
# Using new step-based commands with validation

echo "🚀 Ares 21-Step Training Pipeline (New Commands)"
echo "================================================="

# Step 1: Data Collection
echo "📥 Step 1: Data Collection"
python ares_launcher.py step1 --symbol ETHUSDT --exchange BINANCE --training-mode blank --force

# Step 1.5: Data Converter  
echo "🔄 Step 1.5: Data Converter"
python ares_launcher.py step1_5 --symbol ETHUSDT --exchange BINANCE --training-mode blank

# Alternative: Load data (combines step 1 and 1.5)
echo "📊 Alternative: Load Data (Step 1 + 1.5)"
python ares_launcher.py load --symbol ETHUSDT --exchange BINANCE

# Step 2: Data Reading
echo "📖 Step 2: Data Reading" 
python ares_launcher.py step2 --symbol ETHUSDT --exchange BINANCE --training-mode blank

# Step 2.5: S/R Optimization
echo "🎯 Step 2.5: Support/Resistance Optimization"
python ares_launcher.py step2_5 --symbol ETHUSDT --exchange BINANCE --training-mode blank

# Step 3: HMM Regime Discovery
echo "🧠 Step 3: HMM Regime Discovery"
python ares_launcher.py step3 --symbol ETHUSDT --exchange BINANCE --training-mode blank

# Step 4: Processing & Labeling (Triple Barrier)
echo "🏷️ Step 4: Processing & Labeling"
python ares_launcher.py step4 --symbol ETHUSDT --exchange BINANCE --training-mode blank

# Step 5: Regime Data Splitting
echo "📊 Step 5: Regime Data Splitting"
python ares_launcher.py step5 --symbol ETHUSDT --exchange BINANCE --training-mode blank

# Step 6: Feature Engineering
echo "⚙️ Step 6: Feature Engineering"
python ares_launcher.py step6 --symbol ETHUSDT --exchange BINANCE --training-mode blank

# Step 6.5: Unified Regime Intelligence
echo "🔗 Step 6.5: Unified Regime Intelligence"
python ares_launcher.py step6_5 --symbol ETHUSDT --exchange BINANCE --training-mode blank

# Step 7: Enhanced Matrix Operations
echo "🔢 Step 7: Enhanced Matrix Operations"
python ares_launcher.py step7 --symbol ETHUSDT --exchange BINANCE --training-mode blank

# Step 8: Regime Data Splitting (Final)
echo "📈 Step 8: Regime Data Splitting (Final)"
python ares_launcher.py step8 --symbol ETHUSDT --exchange BINANCE --training-mode blank

# Step 9: HMM-Based Training
echo "🎓 Step 9: HMM-Based Training"
python ares_launcher.py step9 --symbol ETHUSDT --exchange BINANCE --training-mode blank

# Step 9.5: HMM LM Generalist Training
echo "🎭 Step 9.5: HMM LM Generalist Training"
python ares_launcher.py step9_5 --symbol ETHUSDT --exchange BINANCE --training-mode blank

# Step 10: Unified Regime Intelligence
echo "🧭 Step 10: Unified Regime Intelligence"
python ares_launcher.py step10 --symbol ETHUSDT --exchange BINANCE --training-mode blank

# Step 11: Analyst Creation
echo "👨‍💼 Step 11: Analyst Creation"
python ares_launcher.py step11 --symbol ETHUSDT --exchange BINANCE --training-mode blank

# Step 12: Analyst Enhancement
echo "📈 Step 12: Analyst Enhancement"
python ares_launcher.py step12 --symbol ETHUSDT --exchange BINANCE --training-mode blank

# Step 13: Analyst Ensemble Creation
echo "🎭 Step 13: Analyst Ensemble Creation"
python ares_launcher.py step13 --symbol ETHUSDT --exchange BINANCE --training-mode blank

# Step 14: Tactician Labeling
echo "🏷️ Step 14: Tactician Labeling"
python ares_launcher.py step14 --symbol ETHUSDT --exchange BINANCE --training-mode blank

# Step 15: Tactician Specialist Training
echo "🎯 Step 15: Tactician Specialist Training"
python ares_launcher.py step15 --symbol ETHUSDT --exchange BINANCE --training-mode blank

# Step 16: Confidence Calibration
echo "⚖️ Step 16: Confidence Calibration"
python ares_launcher.py step16 --symbol ETHUSDT --exchange BINANCE --training-mode full

# Step 17: Final Parameters Optimization
echo "🔧 Step 17: Final Parameters Optimization"
python ares_launcher.py step17 --symbol ETHUSDT --exchange BINANCE --training-mode full

# Step 18: Walk Forward Validation
echo "🚶 Step 18: Walk Forward Validation"
python ares_launcher.py step18 --symbol ETHUSDT --exchange BINANCE --training-mode full

# Step 19: Monte Carlo Validation
echo "🎲 Step 19: Monte Carlo Validation"
python ares_launcher.py step19 --symbol ETHUSDT --exchange BINANCE --training-mode full

# Step 20: A/B Testing
echo "🧪 Step 20: A/B Testing"
python ares_launcher.py step20 --symbol ETHUSDT --exchange BINANCE --training-mode full

# Step 21: Saving Results
echo "💾 Step 21: Saving Results"
python ares_launcher.py step21 --symbol ETHUSDT --exchange BINANCE --training-mode full

echo "✅ All 21 steps completed!"
echo ""
echo "🎉 Pipeline Summary:"
echo "   • Data Collection & Processing: Steps 1-2.5"
echo "   • Regime Discovery & Labeling: Steps 3-5"
echo "   • Feature Engineering: Steps 6-7"
echo "   • Training & Intelligence: Steps 8-10"
echo "   • Analyst Development: Steps 11-13"
echo "   • Tactician Training: Steps 14-15"
echo "   • Validation & Testing: Steps 16-20 (full mode)"
echo "   • Final Output: Step 21"
echo ""
echo "📝 Notes:"
echo "   • Steps 1-15 use 'blank' training mode (180 days, faster)"
echo "   • Steps 16-21 use 'full' training mode (730 days, comprehensive)"
echo "   • Each step includes automatic validation of previous steps"
echo "   • Use --force flag to restart from a specific step"