# Training Steps - Organized Structure

This directory contains all the training steps organized into logical categories for better maintainability and modularity.

## Directory Structure

```
src/training/steps/
├── data_collection/           # Data collection and preprocessing
├── market_analysis/          # Market analysis and feature engineering
├── model_training/           # Model training and development
├── optimisation/             # Parameter optimization and calibration
├── backtesting/              # Backtesting and validation
└── run_all_pipelines.py      # Main orchestrator for all pipelines
```

## Categories

### 1. Data Collection (`data_collection/`)
Contains all components related to data collection and preprocessing:
- Raw data collection from exchanges
- Data quality validation and checking
- Unified data loading and preprocessing
- Data conversion and format standardization
- Integrated data quality pipeline

**Main Entry Point:** `step01_data_collection_main.py`

### 2. Market Analysis (`market_analysis/`)
Contains all components related to market analysis and feature engineering:
- HMM regime discovery and clustering (includes the modular `hmm_clustering/` subdirectory)
- Regime data splitting and labeling
- Feature engineering and selection
- Advanced matrix operations
- Fractional differentiation
- Regime continuity management

**Main Entry Point:** `step03_market_analysis_main.py`

### 3. Model Training (`model_training/`)
Contains all components related to model training and development:
- HMM-based training and multi-timeframe ensembles
- Unified regime intelligence
- Analyst creation, enhancement, and ensemble creation
- Tactician labeling and specialist training
- Model persistence and validation components

**Main Entry Point:** `step09_model_training_main.py`

### 4. Optimization (`optimisation/`)
Contains all components related to parameter optimization and calibration:
- Confidence calibration per regime
- Final parameters optimization
- Parameter optimization wrapper

**Main Entry Point:** `step16_optimisation_main.py`

### 5. Backtesting (`backtesting/`)
Contains all components related to backtesting and validation:
- Walk forward validation per regime
- Monte Carlo validation per regime
- A/B testing per regime
- Model saving and persistence

**Main Entry Point:** `step18_backtesting_main.py`

## Usage

### Running Individual Pipelines

Each category has its own main entry point that can be run independently:

```bash
# Data Collection
python src/training/steps/data_collection/step01_data_collection_main.py

# Market Analysis
python src/training/steps/market_analysis/step03_market_analysis_main.py

# Model Training
python src/training/steps/model_training/step09_model_training_main.py

# Optimization
python src/training/steps/optimisation/step16_optimisation_main.py

# Backtesting
python src/training/steps/backtesting/step18_backtesting_main.py
```

### Running All Pipelines

To run all pipelines in sequence:

```bash
python src/training/steps/run_all_pipelines.py
```

## Modular Structure

Each category follows the same modular pattern as the `hmm_clustering` module:

1. **Main Entry Point**: A main Python file that provides a simple interface to run the pipeline
2. **`__init__.py`**: Contains all imports and a main pipeline function
3. **Component Files**: Individual step files and their dependencies
4. **Subdirectories**: For complex components (like `hmm_clustering/`)

## Configuration

Each pipeline accepts configuration parameters that can be customized:

- `symbol`: Trading symbol (default: "ETHUSDT")
- `exchange`: Exchange name (default: "BINANCE")
- `timeframe`: Data timeframe (default: "1m")
- `data_dir`: Data directory (default: "data_cache")
- `force_rerun`: Force re-execution of steps (default: True)
- Category-specific parameters for enabling/disabling specific components

## Benefits of This Structure

1. **Modularity**: Each category can be run independently
2. **Maintainability**: Related components are grouped together
3. **Scalability**: Easy to add new components to existing categories
4. **Reusability**: Components can be imported and used in other contexts
5. **Testing**: Each category can be tested independently
6. **Documentation**: Clear separation of concerns makes documentation easier

## Migration Notes

- All original step files have been moved to their appropriate categories
- Import statements in existing code may need to be updated
- The modular structure maintains backward compatibility through the main entry points
- Configuration files and results are saved in the data directory for persistence