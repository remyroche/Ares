"""Script to update imports from old training system to new BaseStep system."""

import os
from pathlib import Path
import re


def update_imports_in_file(file_path: Path) -> bool:
    """Update imports in a single file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if file was modified
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Update enhanced_training_manager imports
        content = re.sub(
            r'from src\.training\.enhanced_training_manager import .*',
            'from src.training.core.training_manager import create_training_manager',
            content
        )
        
        # Update direct step imports to new structure
        step_import_mappings = {
            # Data preparation steps
            r'from src\.training\.steps\.step01_data_collection import':
                'from src.training.steps.data_preparation.step01_data_collection import',
            r'from src\.training\.steps\.step02_data_reading import':
                'from src.training.steps.data_preparation.step02_data_reading import',
                
            # Market analysis steps
            r'from src\.training\.steps\.step03_hmm_regime_discovery import':
                'from src.training.steps.market_analysis.step03_hmm_regime_discovery import',
            r'from src\.training\.steps\.step04_regime_data_splitting import':
                'from src.training.steps.market_analysis.step04_regime_data_splitting import',
                
            # Model training steps
            r'from src\.training\.steps\.step05_labeling import':
                'from src.training.steps.model_training.step05_labeling import',
            r'from src\.training\.steps\.step06_feature_engineering import':
                'from src.training.steps.feature_engineering.step06_feature_engineering import',
                
            # Validation steps
            r'from src\.training\.steps\.(step16|step17|step18|step19|step20)':
                r'from src.training.steps.validation.\1',
        }
        
        for old_pattern, new_pattern in step_import_mappings.items():
            content = re.sub(old_pattern, new_pattern, content)
        
        # Update class names if they've changed
        class_name_mappings = {
            'TrainingManager': 'TrainingManager',
            # Add more class mappings as needed
        }
        
        for old_name, new_name in class_name_mappings.items():
            # Only replace class names, not parts of other names
            content = re.sub(r'\b' + old_name + r'\b', new_name, content)
        
        if content != original_content:
            with open(file_path, 'w') as f:
                f.write(content)
            return True
            
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        
    return False


def find_python_files(root_dir: Path) -> list:
    """Find all Python files in directory tree.
    
    Args:
        root_dir: Root directory to search
        
    Returns:
        List of Python file paths
    """
    python_files = []
    
    for py_file in root_dir.rglob("*.py"):
        # Skip migration scripts and templates
        if "migration" in str(py_file) or "template" in str(py_file):
            continue
            
        python_files.append(py_file)
        
    return python_files


def main():
    """Main function to update imports."""
    print("🔄 Updating imports to new training system...")
    
    # Find all Python files
    src_dir = Path("src")
    python_files = find_python_files(src_dir)
    
    print(f"Found {len(python_files)} Python files to check")
    
    updated_count = 0
    for file_path in python_files:
        if update_imports_in_file(file_path):
            print(f"  ✅ Updated: {file_path}")
            updated_count += 1
    
    print(f"\n📊 Summary: Updated {updated_count} files")
    
    # Create import mapping documentation
    import_mapping_doc = """# Import Mapping Guide

## Old vs New Import Mappings

### Training Manager
```python
# Old
from src.training.core.training_manager import create_training_manager

# New
from src.training.core.training_manager import create_training_manager
```

### Step Imports

#### Data Preparation Steps
```python
# Old
from src.training.steps.data_preparation.step01_data_collection import DataCollectionStep
from src.training.steps.data_preparation.step02_data_reading import DataReadingStep

# New
from src.training.steps.data_preparation.step01_data_collection import DataCollectionStep
from src.training.steps.data_preparation.step02_data_reading import DataReadingStep
```

#### Market Analysis Steps
```python
# Old
from src.training.steps.market_analysis.step03_hmm_regime_discovery import HMMRegimeDiscoveryStep
from src.training.steps.market_analysis.step04_regime_data_splitting import RegimeDataSplittingStep

# New
from src.training.steps.market_analysis.step03_hmm_regime_discovery import HMMRegimeDiscoveryStep
from src.training.steps.market_analysis.step04_regime_data_splitting import RegimeDataSplittingStep
```

#### Model Training Steps
```python
# Old
from src.training.steps.model_training.step05_labeling import LabelingStep
from src.training.steps.feature_engineering.step06_feature_engineering import FeatureEngineeringStep

# New
from src.training.steps.model_training.step05_labeling import LabelingStep
from src.training.steps.feature_engineering.step06_feature_engineering import FeatureEngineeringStep
```

#### Validation Steps
```python
# Old
from src.training.steps.validation.step16_confidence_calibration import ConfidenceCalibrationStep
from src.training.steps.validation.step17_final_parameters_optimization import FinalParametersOptimizationStep

# New
from src.training.steps.validation.step16_confidence_calibration import ConfidenceCalibrationStep
from src.training.steps.validation.step17_final_parameters_optimization import FinalParametersOptimizationStep
```

## Usage Example

```python
# Old way
from src.training.core.training_manager import create_training_manager

manager = TrainingManager(config)
await manager.run_training()

# New way
from src.training.core.training_manager import create_training_manager

manager = create_training_manager(config)
await manager.run_pipeline(training_input)
```
"""
    
    # Save import mapping guide
    guide_path = Path("src/training/IMPORT_MAPPING_GUIDE.md")
    with open(guide_path, 'w') as f:
        f.write(import_mapping_doc)
    
    print(f"\n📝 Created import mapping guide: {guide_path}")


if __name__ == "__main__":
    main()