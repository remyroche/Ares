#!/bin/bash

# Ares 20-Step Training Pipeline
# Consolidated script using new step commands with training modes
# Supports: light (30 days), blank (180 days), full (730 days)

# Parse arguments
TRAINING_MODE=${1:-blank}
SYMBOL=${2:-ETHUSDT}
EXCHANGE=${3:-BINANCE}

# Validate training mode
if [[ ! "$TRAINING_MODE" =~ ^(light|blank|full)$ ]]; then
    echo "❌ Invalid training mode: $TRAINING_MODE"
    echo "📋 Usage: $0 [training_mode] [symbol] [exchange]"
    echo ""
    echo "Parameters:"
    echo "   training_mode: light, blank, or full (default: blank)"
    echo "   symbol: Trading symbol (default: ETHUSDT)"
    echo "   exchange: Exchange name (default: BINANCE)"
    echo ""
    echo "Examples:"
    echo "   $0 light BTCUSDT BINANCE"
    echo "   $0 blank ETHUSDT MEXC"
    echo "   $0 full ADAUSDT GATEIO"
    echo ""
    echo "Training Modes:"
    echo "   • light: 30 days, 5% compute, quick testing"
    echo "   • blank: 180 days, 25% compute, moderate testing"
    echo "   • full: 730 days, 100% compute, production"
    exit 1
fi

# Validate symbol and exchange
if [[ -z "$SYMBOL" || -z "$EXCHANGE" ]]; then
    echo "❌ Symbol and exchange cannot be empty"
    echo "📋 Usage: $0 [training_mode] [symbol] [exchange]"
    exit 1
fi

echo "🚀 Ares 20-Step Training Pipeline ($TRAINING_MODE mode)"
echo "======================================================="
echo "🎯 Symbol: $SYMBOL"
echo "🏢 Exchange: $EXCHANGE"
echo ""

case $TRAINING_MODE in
    "light")
        echo "💡 LIGHT MODE: 30 days, 5% compute, ~10 min per step"
        VALIDATION_MODE="blank"  # Use blank for validation steps
        ;;
    "blank")
        echo "🧪 BLANK MODE: 180 days, 25% compute, ~30 min per step"
        VALIDATION_MODE="blank"
        ;;
    "full")
        echo "🚀 FULL MODE: 730 days, 100% compute, ~120 min per step"
        VALIDATION_MODE="full"
        ;;
esac
echo ""

# Step 1: Data Collection
echo "📥 Step 1: Data Collection"
python ares_launcher.py step1 --symbol $SYMBOL --exchange $EXCHANGE --training-mode $TRAINING_MODE --force

# Step 1.5: Data Converter
echo "🔄 Step 1.5: Data Converter"
python ares_launcher.py step1_5 --symbol $SYMBOL --exchange $EXCHANGE --training-mode $TRAINING_MODE

# Alternative: Load data (combines step 1 and 1.5)
echo "📊 Alternative: Load Data (Step 1 + 1.5)"
python ares_launcher.py load --symbol $SYMBOL --exchange $EXCHANGE

# Step 2: Data Reading
echo "📖 Step 2: Data Reading"
python ares_launcher.py step2 --symbol $SYMBOL --exchange $EXCHANGE --training-mode $TRAINING_MODE

# Step 2.5: S/R Optimization
echo "🎯 Step 2.5: Support/Resistance Optimization"
python ares_launcher.py step2_5 --symbol $SYMBOL --exchange $EXCHANGE --training-mode $TRAINING_MODE

# Step 3: HMM Regime Discovery
echo "🧠 Step 3: HMM Regime Discovery"
python ares_launcher.py step3 --symbol $SYMBOL --exchange $EXCHANGE --training-mode $TRAINING_MODE

# Step 4: Processing & Labeling (Triple Barrier)
echo "🏷️ Step 4: Processing & Labeling"
python ares_launcher.py step4 --symbol $SYMBOL --exchange $EXCHANGE --training-mode $TRAINING_MODE

# Step 5: Regime Data Splitting
echo "📊 Step 5: Regime Data Splitting"
python ares_launcher.py step5 --symbol $SYMBOL --exchange $EXCHANGE --training-mode $TRAINING_MODE

# Step 6: Feature Engineering
echo "⚙️ Step 6: Feature Engineering"
python ares_launcher.py step6 --symbol $SYMBOL --exchange $EXCHANGE --training-mode $TRAINING_MODE

# Step 7: Enhanced Matrix Operations
echo "🔢 Step 7: Enhanced Matrix Operations"
python ares_launcher.py step7 --symbol $SYMBOL --exchange $EXCHANGE --training-mode $TRAINING_MODE

# Step 8: Regime Data Splitting (Final)
echo "📈 Step 8: Regime Data Splitting (Final)"
python ares_launcher.py step8 --symbol $SYMBOL --exchange $EXCHANGE --training-mode $TRAINING_MODE

# Step 9: HMM-Based Training
echo "🎓 Step 9: HMM-Based Training"
python ares_launcher.py step9 --symbol $SYMBOL --exchange $EXCHANGE --training-mode $TRAINING_MODE

# Step 9.5: HMM LM Generalist Training
echo "🎭 Step 9.5: HMM LM Generalist Training"
python ares_launcher.py step9_5 --symbol $SYMBOL --exchange $EXCHANGE --training-mode $TRAINING_MODE

# Step 10: Unified Regime Intelligence
echo "🧭 Step 10: Unified Regime Intelligence"
python ares_launcher.py step10 --symbol $SYMBOL --exchange $EXCHANGE --training-mode $TRAINING_MODE

# Step 11: Analyst Creation
echo "👨‍💼 Step 11: Analyst Creation"
python ares_launcher.py step11 --symbol $SYMBOL --exchange $EXCHANGE --training-mode $TRAINING_MODE

# Step 12: Analyst Enhancement
echo "📈 Step 12: Analyst Enhancement"
python ares_launcher.py step12 --symbol $SYMBOL --exchange $EXCHANGE --training-mode $TRAINING_MODE

# Step 13: Analyst Ensemble Creation
echo "🎭 Step 13: Analyst Ensemble Creation"
python ares_launcher.py step13 --symbol $SYMBOL --exchange $EXCHANGE --training-mode $TRAINING_MODE

# Step 14: Tactician Labeling
echo "🏷️ Step 14: Tactician Labeling"
python ares_launcher.py step14 --symbol $SYMBOL --exchange $EXCHANGE --training-mode $TRAINING_MODE

# Step 15: Tactician Specialist Training
echo "🎯 Step 15: Tactician Specialist Training"
python ares_launcher.py step15 --symbol $SYMBOL --exchange $EXCHANGE --training-mode $TRAINING_MODE

# Step 16: Confidence Calibration
echo "⚖️ Step 16: Confidence Calibration"
python ares_launcher.py step16 --symbol $SYMBOL --exchange $EXCHANGE --training-mode $VALIDATION_MODE

# Step 17: Final Parameters Optimization
echo "🔧 Step 17: Final Parameters Optimization"
python ares_launcher.py step17 --symbol $SYMBOL --exchange $EXCHANGE --training-mode $VALIDATION_MODE

# Step 18: Walk Forward Validation
echo "🚶 Step 18: Walk Forward Validation"
python ares_launcher.py step18 --symbol $SYMBOL --exchange $EXCHANGE --training-mode $VALIDATION_MODE

# Step 19: Monte Carlo Validation
echo "🎲 Step 19: Monte Carlo Validation"
python ares_launcher.py step19 --symbol $SYMBOL --exchange $EXCHANGE --training-mode $VALIDATION_MODE

# Step 20: A/B Testing
echo "🧪 Step 20: A/B Testing"
python ares_launcher.py step20 --symbol $SYMBOL --exchange $EXCHANGE --training-mode $VALIDATION_MODE

# Step 21: Saving Results
echo "💾 Step 21: Saving Results"
python ares_launcher.py step21 --symbol $SYMBOL --exchange $EXCHANGE --training-mode $VALIDATION_MODE

echo ""
echo "✅ All 20 steps completed for $SYMBOL on $EXCHANGE in $TRAINING_MODE mode!"
echo ""
echo "🎉 Pipeline Summary:"
echo "   • Symbol: $SYMBOL"
echo "   • Exchange: $EXCHANGE"
echo "   • Training Mode: $TRAINING_MODE"
echo "   • Validation Mode: $VALIDATION_MODE"
echo ""
echo "📋 Step Groups:"
echo "   • Data Collection & Processing: Steps 1-2.5"
echo "   • Regime Discovery & Labeling: Steps 3-5"
echo "   • Feature Engineering: Steps 6-7"
echo "   • Training & Intelligence: Steps 8-10"
echo "   • Analyst Development: Steps 11-13"
echo "   • Tactician Training: Steps 14-15"
echo "   • Validation & Testing: Steps 16-21 ($VALIDATION_MODE mode)"
echo ""
echo "📊 Training Mode Comparison:"
echo "   Mode    | Days | Comp Load | Max Trials | Duration/Step"
echo "   --------|------|-----------|------------|-------------"
echo "   light   |  30  |    5%     |    10      |   ~10 min"
echo "   blank   | 180  |   25%     |    50      |   ~30 min"
echo "   full    | 730  |  100%     |   200      |  ~120 min"
echo ""
echo "📝 Notes:"
echo "   • Training mode: $TRAINING_MODE (used for steps 1-15)"
echo "   • Validation mode: $VALIDATION_MODE (used for steps 16-21)"
echo "   • Data amount scales: 30d → 180d → 730d lookback periods"
echo "   • Computational load scales: trials, jobs, iterations reduced"
echo "   • Same features/algorithms across all modes"
echo "   • Step 10 contains the Unified Regime Intelligence system"
echo ""
echo "🚀 Usage Examples:"
echo "   ./pipeline.sh light                    # ETHUSDT/BINANCE, fast testing"
echo "   ./pipeline.sh blank BTCUSDT BINANCE    # BTCUSDT/BINANCE, medium testing"
echo "   ./pipeline.sh full ADAUSDT MEXC        # ADAUSDT/MEXC, production"
echo "   ./pipeline.sh light ETHUSDT GATEIO     # ETHUSDT/GATEIO, fast testing"