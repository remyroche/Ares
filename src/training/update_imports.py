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
        content = re.sub('from src\\.training\\.enhanced_training_manager import .*', 'from src.training.core.training_manager import create_training_manager', content)
        step_import_mappings = {'from src\\.training\\.steps\\.step01_data_collection import': 'from src.training.steps.data_preparation.step01_data_collection import', 'from src\\.training\\.steps\\.step02_data_reading import': 'from src.training.steps.data_preparation.step02_data_reading import', 'from src\\.training\\.steps\\.step03_hmm_regime_discovery import': 'from src.training.steps.market_analysis.step03_hmm_regime_discovery import', 'from src\\.training\\.steps\\.step04_regime_data_splitting import': 'from src.training.steps.market_analysis.step04_regime_data_splitting import', 'from src\\.training\\.steps\\.step05_labeling import': 'from src.training.steps.model_training.step05_labeling import', 'from src\\.training\\.steps\\.step06_feature_engineering import': 'from src.training.steps.feature_engineering.step06_feature_engineering import', 'from src\\.training\\.steps\\.(step16|step17|step18|step19|step20)': 'from src.training.steps.validation.\\1'}
        for old_pattern, new_pattern in step_import_mappings.items():
            content = re.sub(old_pattern, new_pattern, content)
        class_name_mappings = {'TrainingManager': 'TrainingManager'}
        for old_name, new_name in class_name_mappings.items():
            content = re.sub('\\b' + old_name + '\\b', new_name, content)
        if content != original_content:
            with open(file_path, 'w') as f:
                f.write(content)
            return True
    except Exception as e:
        print(f'Error updating {file_path}: {e}')
    return False

def find_python_files(root_dir: Path) -> list:
    """Find all Python files in directory tree.
    
    Args:
        root_dir: Root directory to search
        
    Returns:
        List of Python file paths
    """
    python_files = []
    for py_file in root_dir.rglob('*.py'):
        if 'migration' in str(py_file) or 'template' in str(py_file):
            continue
        python_files.append(py_file)
    return python_files

def main() -> None:
    """Main function to update imports."""
    print('🔄 Updating imports to new training system...')
    src_dir = Path('src')
    python_files = find_python_files(src_dir)
    print(f'Found {len(python_files)} Python files to check')
    updated_count = 0
    for file_path in python_files:
        if update_imports_in_file(file_path):
            print(f'  ✅ Updated: {file_path}')
            updated_count += 1
    print(f'\n📊 Summary: Updated {updated_count} files')
    import_mapping_doc = '# Import Mapping Guide\n\n## Old vs New Import Mappings\n\n### Training Manager\n```python\n# Old\nfrom src.training.core.training_manager import create_training_manager\n\n# New\nfrom src.training.core.training_manager import create_training_manager\n```\n\n### Step Imports\n\n#### Data Preparation Steps\n```python\n# Old\nfrom src.training.steps.data_preparation.step01_data_collection import DataCollectionStep\nfrom src.training.steps.data_preparation.step02_data_reading import DataReadingStep\n\n# New\nfrom src.training.steps.data_preparation.step01_data_collection import DataCollectionStep\nfrom src.training.steps.data_preparation.step02_data_reading import DataReadingStep\n```\n\n#### Market Analysis Steps\n```python\n# Old\nfrom src.training.steps.market_analysis.step03_hmm_regime_discovery import HMMRegimeDiscoveryStep\nfrom src.training.steps.market_analysis.step04_regime_data_splitting import RegimeDataSplittingStep\n\n# New\nfrom src.training.steps.market_analysis.step03_hmm_regime_discovery import HMMRegimeDiscoveryStep\nfrom src.training.steps.market_analysis.step04_regime_data_splitting import RegimeDataSplittingStep\n```\n\n#### Model Training Steps\n```python\n# Old\nfrom src.training.steps.model_training.step05_labeling import LabelingStep\nfrom src.training.steps.feature_engineering.step06_feature_engineering import FeatureEngineeringStep\n\n# New\nfrom src.training.steps.model_training.step05_labeling import LabelingStep\nfrom src.training.steps.feature_engineering.step06_feature_engineering import FeatureEngineeringStep\n```\n\n#### Validation Steps\n```python\n# Old\nfrom src.training.steps.validation.step16_confidence_calibration import ConfidenceCalibrationStep\nfrom src.training.steps.validation.step17_final_parameters_optimization import FinalParametersOptimizationStep\n\n# New\nfrom src.training.steps.validation.step16_confidence_calibration import ConfidenceCalibrationStep\nfrom src.training.steps.validation.step17_final_parameters_optimization import FinalParametersOptimizationStep\n```\n\n## Usage Example\n\n```python\n# Old way\nfrom src.training.core.training_manager import create_training_manager\n\nmanager = TrainingManager(config)\nawait manager.run_training()\n\n# New way\nfrom src.training.core.training_manager import create_training_manager\n\nmanager = create_training_manager(config)\nawait manager.run_pipeline(training_input)\n```\n'
    guide_path = Path('src/training/IMPORT_MAPPING_GUIDE.md')
    with open(guide_path, 'w') as f:
        f.write(import_mapping_doc)
    print(f'\n📝 Created import mapping guide: {guide_path}')
if __name__ == '__main__':
    main()