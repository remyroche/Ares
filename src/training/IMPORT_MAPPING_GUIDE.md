# Import Mapping Guide

## Old vs New Import Mappings

### Training Manager
```python
# Old
from src.training.enhanced_training_manager import EnhancedTrainingManager

# New
from src.training.core.training_manager import create_training_manager
```

### Step Imports

#### Data Preparation Steps
```python
# Old
from src.training.steps.step01_data_collection import DataCollectionStep
from src.training.steps.step02_data_reading import DataReadingStep

# New
from src.training.steps.data_preparation.step01_data_collection import DataCollectionStep
from src.training.steps.data_preparation.step02_data_reading import DataReadingStep
```

#### Market Analysis Steps
```python
# Old
from src.training.steps.step03_hmm_regime_discovery import HMMRegimeDiscoveryStep
from src.training.steps.step04_regime_data_splitting import RegimeDataSplittingStep

# New
from src.training.steps.market_analysis.step03_hmm_regime_discovery import HMMRegimeDiscoveryStep
from src.training.steps.market_analysis.step04_regime_data_splitting import RegimeDataSplittingStep
```

#### Model Training Steps
```python
# Old
from src.training.steps.step05_labeling import LabelingStep
from src.training.steps.step06_feature_engineering import FeatureEngineeringStep

# New
from src.training.steps.model_training.step05_labeling import LabelingStep
from src.training.steps.feature_engineering.step06_feature_engineering import FeatureEngineeringStep
```

#### Validation Steps
```python
# Old
from src.training.steps.step16_confidence_calibration import ConfidenceCalibrationStep
from src.training.steps.step17_final_parameters_optimization import FinalParametersOptimizationStep

# New
from src.training.steps.validation.step16_confidence_calibration import ConfidenceCalibrationStep
from src.training.steps.validation.step17_final_parameters_optimization import FinalParametersOptimizationStep
```

## Usage Example

```python
# Old way
from src.training.enhanced_training_manager import EnhancedTrainingManager

manager = EnhancedTrainingManager(config)
await manager.run_training()

# New way
from src.training.core.training_manager import create_training_manager

manager = create_training_manager(config)
await manager.run_pipeline(training_input)
```
