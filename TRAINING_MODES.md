# Training Modes Guide

The Ares pipeline supports three training modes that automatically scale optimization parameters based on the lookback period and computational intensity.

## 🎯 Training Modes

### 💡 Light Mode (30 days)
- **Purpose**: Quick testing and development
- **Lookback**: 30 days (less data for speed)
- **Computational Load**: 5% (10 max trials vs 200)
- **Features**: Same algorithms and features as full mode
- **Duration**: ~10 minutes per step
- **Use Case**: Rapid prototyping, code validation, CI/CD testing

### 🧪 Blank Mode (180 days)  
- **Purpose**: Moderate testing and validation
- **Lookback**: 180 days (medium data amount)
- **Computational Load**: 25% (50 max trials vs 200)
- **Features**: Same algorithms and features as full mode
- **Duration**: ~30 minutes per step
- **Use Case**: Feature validation, model testing, development validation

### 🚀 Full Mode (730 days)
- **Purpose**: Production-ready models
- **Lookback**: 730 days (full data)
- **Computational Load**: 100% (200 max trials)
- **Features**: All algorithms and features enabled
- **Duration**: ~120 minutes per step
- **Use Case**: Production deployment, final model training

## 📊 Parameter Scaling

**Data Scaling**: Different lookback periods (30d → 180d → 730d)
**Computational Scaling**: Only trials, jobs, iterations are reduced (same features/algorithms)

| Parameter | Light (5%) | Blank (25%) | Full (100%) |
|-----------|------------|-------------|-------------|
| **Data Parameters** |
| Lookback Days | 30 | 180 | 730 |
| Min Data Points | 50 | 100 | 500 |
| **Computational Parameters** |
| Max Trials | 10 | 50 | 200 |
| N Trials | 5 | 25 | 100 |
| **Step 17 - Final Parameters Optimization** |
| Confidence Threshold Trials | 3 | 10 | 40 |
| Volatility Trials | 3 | 13 | 50 |
| Position Sizing Trials | 3 | 15 | 60 |
| Risk Management Trials | 3 | 13 | 50 |
| Parallel Jobs | 1 | 1 | 4 |
| **Step 12 - Analyst Enhancement** |
| LightGBM Trials | 3 | 13 | 50 |
| XGBoost Trials | 3 | 13 | 50 |
| Neural Network Trials | 3 | 7 | 25 |
| Parallel Jobs | 1 | 1 | 4 |

## 🚀 Usage

### Pipeline Script
```bash
# Quick testing (30 days, ~2 hours total)
./pipeline.sh light

# Moderate testing (180 days, ~5 hours total)  
./pipeline.sh blank

# Production training (730 days, ~40 hours total)
./pipeline.sh full
```

### Individual Steps
```bash
# Light mode for quick feature testing
python ares_launcher.py step6 --symbol ETHUSDT --exchange BINANCE --training-mode light

# Blank mode for validation
python ares_launcher.py step17 --symbol ETHUSDT --exchange BINANCE --training-mode blank

# Full mode for production
python ares_launcher.py step17 --symbol ETHUSDT --exchange BINANCE --training-mode full
```

## ⚙️ Automatic Features

### Parameter Scaling
- **Minimum Trials**: Always at least 3 for statistical validity
- **Intensity Scaling**: All parameters scale by intensity percentage
- **Timeout Scaling**: Optimization timeouts scale with intensity
- **Memory Usage**: Lower memory requirements for lighter modes

### Mode-Specific Features
- **Light Mode**: Disables advanced model training, ensemble training
- **Blank Mode**: Enables basic ensemble training
- **Full Mode**: Enables all advanced features (multi-timeframe, adaptive training)

### Validation Strategy
- **Light/Blank**: Validation steps use same mode for consistency
- **Full**: All validation steps use full mode for production readiness

## 📝 Best Practices

### Development Workflow
1. **Light Mode**: Initial development and testing
2. **Blank Mode**: Feature validation and integration testing
3. **Full Mode**: Final production training

### Resource Management
- Light mode: ~2GB RAM, 1 CPU core
- Blank mode: ~4GB RAM, 2 CPU cores
- Full mode: ~8GB RAM, 4 CPU cores

### Time Planning
- Light: Perfect for daily development cycles
- Blank: Good for weekly validation runs
- Full: Plan for overnight/weekend runs

## 🔧 Configuration

Training modes are centrally configured in `src/config/training_modes.py`:
- All parameters automatically scale
- Step-specific overrides available
- Easy to add new optimization parameters
- Consistent across all pipeline steps

The system ensures that lighter modes have proportionally lighter optimization parameters while maintaining statistical validity and model quality.