#!/bin/bash

# Updated Ares 21-Step Training Pipeline
# Based on actual step files in src/training/steps/

echo "🚀 Ares 21-Step Training Pipeline"
echo "=================================="

# Step 1: Data Collection
echo "📥 Step 1: Data Collection"
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --step step1_data_collection --force

# Step 1.5: Data Converter
echo "🔄 Step 1.5: Data Converter"
python ares_launcher.py step1_5 --symbol ETHUSDT --exchange BINANCE

# Alternative: Load data (combines step 1 and 1.5)
echo "📊 Alternative: Load Data (Step 1 + 1.5)"
python ares_launcher.py load --symbol ETHUSDT --exchange BINANCE

# Step 2: Data Reading
echo "📖 Step 2: Data Reading"
python ares_launcher.py step2 --symbol ETHUSDT --exchange BINANCE

# Step 2.5: S/R Optimization
echo "🎯 Step 2.5: Support/Resistance Optimization"
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --step step2_5_sr_optimization --force-rerun

# Step 3: HMM Regime Discovery
echo "🧠 Step 3: HMM Regime Discovery"
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --step step3_hmm_regime_discovery --force-rerun

# Step 4: Processing & Labeling (Triple Barrier)
echo "🏷️ Step 4: Processing & Labeling"
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --step step4_processing_labeling --force-rerun

# Step 5: Regime Data Splitting
echo "📊 Step 5: Regime Data Splitting"
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --step step5_regime_data_splitting --force-rerun

# Step 6: Feature Engineering
echo "⚙️ Step 6: Feature Engineering"
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --step step6_feature_engineering --force-rerun

# Step 6.5: Unified Regime Intelligence
echo "🔗 Step 6.5: Unified Regime Intelligence"
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --step step6_5_unified_regime_intelligence --force-rerun

# Step 7: Enhanced Matrix Operations
echo "🔢 Step 7: Enhanced Matrix Operations"
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --step step7_enhanced_matrix_operations --force-rerun

# Step 8: Regime Data Splitting (Final)
echo "📈 Step 8: Regime Data Splitting (Final)"
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --step step8_regime_data_splitting --force-rerun

# Step 9: HMM-Based Training
echo "🎓 Step 9: HMM-Based Training"
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --step step9_hmm_based_training --force-rerun

# Step 9.5: HMM LM Generalist Training
echo "🎭 Step 9.5: HMM LM Generalist Training"
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --step step9_5_hmm_lm_generalist_training --force-rerun

# Step 10: Unified Regime Intelligence
echo "🧭 Step 10: Unified Regime Intelligence"
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --step step10_unified_regime_intelligence --force-rerun

# Step 11: Analyst Creation
echo "👨‍💼 Step 11: Analyst Creation"
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --step step11_analyst_creation --force-rerun

# Step 12: Analyst Enhancement
echo "📈 Step 12: Analyst Enhancement"
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --step step12_analyst_enhancement --force-rerun

# Step 13: Analyst Ensemble Creation
echo "🎭 Step 13: Analyst Ensemble Creation"
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --step step13_analyst_ensemble_creation --force-rerun

# Step 14: Tactician Labeling
echo "🏷️ Step 14: Tactician Labeling"
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --step step14_tactician_labeling --force-rerun

# Step 15: Tactician Specialist Training
echo "🎯 Step 15: Tactician Specialist Training"
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --step step15_tactician_specialist_training --force-rerun

# Step 16: Confidence Calibration
echo "⚖️ Step 16: Confidence Calibration"
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --step step16_confidence_calibration --force-rerun

# Step 17: Final Parameters Optimization
echo "🔧 Step 17: Final Parameters Optimization"
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --step step17_final_parameters_optimization --force-rerun

# Step 18: Walk Forward Validation
echo "🚶 Step 18: Walk Forward Validation"
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --step step18_walk_forward_validation --force-rerun

# Step 19: Monte Carlo Validation
echo "🎲 Step 19: Monte Carlo Validation"
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --step step19_monte_carlo_validation --force-rerun

# Step 20: A/B Testing
echo "🧪 Step 20: A/B Testing"
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --step step20_ab_testing --force-rerun

# Step 21: Saving Results
echo "💾 Step 21: Saving Results"
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --step step21_saving --force-rerun

echo "✅ All 21 steps completed!"
echo ""
echo "🎉 Pipeline Summary:"
echo "   • Data Collection & Processing: Steps 1-2.5"
echo "   • Regime Discovery & Labeling: Steps 3-5"
echo "   • Feature Engineering: Steps 6-7"
echo "   • Training & Intelligence: Steps 8-10"
echo "   • Analyst Development: Steps 11-13"
echo "   • Tactician Training: Steps 14-15"
echo "   • Validation & Testing: Steps 16-20"
echo "   • Final Output: Step 21"