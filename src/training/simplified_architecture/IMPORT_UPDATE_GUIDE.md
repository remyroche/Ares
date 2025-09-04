# Import Update Guide

## Files That Need Import Updates

### 1. **src/training/run_pipeline_with_step08.py**
```python
# OLD
from src.training.enhanced_training_manager import EnhancedTrainingManager

# NEW
from src.training.simplified_architecture.enhanced_pipeline_orchestrator import create_pipeline
```

### 2. **src/core/service_registry.py**
```python
# OLD
from src.training.training_manager import TrainingManager

# NEW
from src.training.simplified_architecture.enhanced_pipeline_orchestrator import EnhancedPipelineOrchestrator
```

## Complete Import Mapping

### **Core Training Manager Replacement**
```python
# OLD
from src.training.enhanced_training_manager import TrainingManager
from src.training.training_manager import TrainingManager
from src.training.step_orchestrator import StepOrchestrator
from src.training.simplified_training_manager import SimplifiedTrainingManager

# NEW
from src.training.simplified_architecture.enhanced_pipeline_orchestrator import (
    EnhancedPipelineOrchestrator, create_pipeline
)
```

### **Step Components**
```python
# OLD
from src.training.steps.step01_data_collection import DataCollectionStep
from src.training.steps.step01_5_data_converter import DataConverterStep
from src.training.steps.step06_feature_engineering import FeatureEngineeringStep
from src.training.steps.step09_hmm_based_training import HMMBasedTrainingStep

# NEW
from src.training.simplified_architecture.migrated_components.data_components import (
    DataCollectionStep, DataConverterStep
)
from src.training.simplified_architecture.enhanced_interfaces import (
    StepConfig, StepFactory, BasePipelineStep
)
```

### **Configuration System**
```python
# OLD
from src.training.step_config import StepConfig

# NEW
from src.training.simplified_architecture.enhanced_config_system import (
    ConfigurationManager, PipelineConfiguration, StepConfiguration
)
from src.training.simplified_architecture.enhanced_interfaces import StepConfig
```

### **Dependency Injection**
```python
# NEW (Add these imports where needed)
from src.training.simplified_architecture.dependency_injection import (
    EnhancedDIContainer, ServiceLifetime, inject, injectable
)
```

## Usage Pattern Changes

### **Old Pattern:**
```python
# OLD
from src.training.enhanced_training_manager import TrainingManager

config = {
    'symbol': 'BTCUSDT',
    'exchange': 'binance',
    'timeframe': '1h',
    'lookback_days': 30
}

manager = TrainingManager(config)
result = await manager.execute_enhanced_training(config)
```

### **New Pattern:**
```python
# NEW
from src.training.simplified_architecture.enhanced_pipeline_orchestrator import create_pipeline

# Create pipeline from configuration file
pipeline = create_pipeline("config/trading_pipeline.yaml")
result = await pipeline.run()

# Or create pipeline from configuration object
from src.training.simplified_architecture.enhanced_config_system import PipelineConfiguration

config = PipelineConfiguration(
    name="Trading_Pipeline",
    version="1.0.0",
    steps=[...]
)
pipeline = EnhancedPipelineOrchestrator(config=config)
result = await pipeline.run()
```

## Files to Update

### **High Priority (Core Files)**
1. `src/training/run_pipeline_with_step08.py`
2. `src/core/service_registry.py`
3. Any files that directly instantiate `TrainingManager`

### **Medium Priority (Integration Files)**
1. Files in `src/training/examples/`
2. Files in `src/training/demo_*.py`
3. Test files that use the old training manager

### **Low Priority (Documentation)**
1. `src/training/MIGRATION_GUIDE.md` (already updated)
2. `src/training/IMPORT_MAPPING_GUIDE.md` (already updated)
3. Other documentation files

## Automated Update Script

Create a script to automatically update imports:

```python
# update_imports.py
import os
import re
from pathlib import Path

def update_imports_in_file(file_path):
    """Update imports in a single file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Define replacement patterns
    replacements = [
        (
            r'from src\.training\.enhanced_training_manager import TrainingManager',
            'from src.training.simplified_architecture.enhanced_pipeline_orchestrator import create_pipeline'
        ),
        (
            r'from src\.training\.training_manager import TrainingManager',
            'from src.training.simplified_architecture.enhanced_pipeline_orchestrator import EnhancedPipelineOrchestrator'
        ),
        (
            r'from src\.training\.step_orchestrator import StepOrchestrator',
            'from src.training.simplified_architecture.enhanced_pipeline_orchestrator import EnhancedPipelineOrchestrator'
        )
    ]
    
    # Apply replacements
    for old_pattern, new_import in replacements:
        content = re.sub(old_pattern, new_import, content)
    
    # Write back if changes were made
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Updated imports in {file_path}")

# Run the update
for py_file in Path('src').rglob('*.py'):
    if py_file.name != 'update_imports.py':  # Don't update this script
        update_imports_in_file(py_file)
```

## Testing After Import Updates

After updating imports, run these tests:

```bash
# Test the new architecture
cd src/training/simplified_architecture
python -m pytest tests/ -v

# Test specific components
python -c "
from src.training.simplified_architecture.enhanced_pipeline_orchestrator import create_pipeline
print('Import successful!')
"

# Test configuration loading
python -c "
from src.training.simplified_architecture.enhanced_config_system import ConfigurationManager
config_manager = ConfigurationManager()
print('Configuration system working!')
"
```