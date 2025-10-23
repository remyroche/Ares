"""
Import Updater for Base Classes

This module provides utilities to update import statements throughout the codebase
to use the new production-ready base classes. It handles both direct imports and
factory-based imports.

Key Features:
- Automatic import statement detection and updating
- Support for both production and existing base classes
- Factory function integration
- Backward compatibility maintenance
- Comprehensive error handling and validation
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import logging

# Setup logging
logger = logging.getLogger(__name__)

class ImportUpdater:
    """
    Class for updating import statements throughout the codebase.
    
    This class provides methods to automatically update import statements
    to use the new production-ready base classes while maintaining
    backward compatibility.
    """
    
    def __init__(self, workspace_root: str = "/workspace"):
        """
        Initialize the import updater.
        
        Args:
            workspace_root: Root directory of the workspace
        """
        self.workspace_root = Path(workspace_root)
        self.src_dir = self.workspace_root / "src"
        
        # Import mappings for base classes
        self.import_mappings = {
            'BaseValidator': {
                'old_imports': [
                    'from src.utils.base_validator import BaseValidator',
                    'from src.utils.base_validator import BaseValidator as',
                    'import src.utils.base_validator.BaseValidator',
                ],
                'new_imports': [
                    'from src.core.abstract_base_classes import BaseValidator',
                    'from src.core.factory import create_validator',
                ]
            },
            'BaseTrainingStep': {
                'old_imports': [
                    'from src.utils.ml_common.training.base_training_step import BaseTrainingStep',
                    'from src.utils.ml_common.training.base_training_step import BaseTrainingStep as',
                    'import src.utils.ml_common.training.base_training_step.BaseTrainingStep',
                ],
                'new_imports': [
                    'from src.core.abstract_base_classes import BaseTrainingStep',
                    'from src.core.factory import create_training_step',
                ]
            },
            'BaseClusteringAlgorithm': {
                'old_imports': [
                    'from src.training.steps.market_analysis.components.clustering_algorithms import BaseClusteringAlgorithm',
                    'from src.training.steps.market_analysis.components.clustering_algorithms import BaseClusteringAlgorithm as',
                    'import src.training.steps.market_analysis.components.clustering_algorithms.BaseClusteringAlgorithm',
                ],
                'new_imports': [
                    'from src.core.abstract_base_classes import BaseClusteringAlgorithm',
                    'from src.core.factory import create_clustering_algorithm',
                ]
            },
            'MultiOutputModel': {
                'old_imports': [
                    'from src.utils.ml_common.models.multi_output_models import MultiOutputModel',
                    'from src.utils.ml_common.models.multi_output_models import MultiOutputModel as',
                    'import src.utils.ml_common.models.multi_output_models.MultiOutputModel',
                ],
                'new_imports': [
                    'from src.core.abstract_base_classes import MultiOutputModel',
                    'from src.core.factory import create_multi_output_model',
                ]
            },
            'BasePatternDiscoverer': {
                'old_imports': [
                    'from src.research.price_patterns.pattern_discovery_framework import BasePatternDiscoverer',
                    'from src.research.price_patterns.pattern_discovery_framework import BasePatternDiscoverer as',
                    'import src.research.price_patterns.pattern_discovery_framework.BasePatternDiscoverer',
                ],
                'new_imports': [
                    'from src.core.abstract_base_classes import BasePatternDiscoverer',
                    'from src.core.factory import create_pattern_discoverer',
                ]
            },
            'BaseLabelingStrategy': {
                'old_imports': [
                    'from src.research.profit_labeling.ensemble_labeling_system import BaseLabelingStrategy',
                    'from src.research.profit_labeling.ensemble_labeling_system import BaseLabelingStrategy as',
                    'import src.research.profit_labeling.ensemble_labeling_system.BaseLabelingStrategy',
                ],
                'new_imports': [
                    'from src.core.abstract_base_classes import BaseLabelingStrategy',
                    'from src.core.factory import create_labeling_strategy',
                ]
            }
        }
        
        # Files to exclude from updates
        self.exclude_patterns = [
            '__pycache__',
            '.git',
            '.pytest_cache',
            'node_modules',
            '.venv',
            'venv',
            'env',
            '.env',
            '*.pyc',
            '*.pyo',
            '*.pyd',
            '*.so',
            '*.dll',
            '*.dylib',
            '*.exe',
            '*.egg',
            '*.egg-info',
            'dist',
            'build',
            '*.egg-info'
        ]
        
        # Statistics
        self.stats = {
            'files_processed': 0,
            'imports_updated': 0,
            'errors': 0,
            'warnings': 0
        }

    def should_exclude_file(self, file_path: Path) -> bool:
        """
        Check if a file should be excluded from processing.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file should be excluded
        """
        file_str = str(file_path)
        
        for pattern in self.exclude_patterns:
            if pattern in file_str:
                return True
        
        return False

    def find_python_files(self) -> List[Path]:
        """
        Find all Python files in the src directory.
        
        Returns:
            List of Python file paths
        """
        python_files = []
        
        for file_path in self.src_dir.rglob("*.py"):
            if not self.should_exclude_file(file_path):
                python_files.append(file_path)
        
        logger.info(f"Found {len(python_files)} Python files to process")
        return python_files

    def find_imports_in_file(self, file_path: Path) -> List[Tuple[str, int, str]]:
        """
        Find import statements in a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of tuples (import_line, line_number, full_line)
        """
        imports = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines, 1):
                line = line.strip()
                
                # Check for import statements
                if line.startswith('from ') and ' import ' in line:
                    imports.append((line, i, line))
                elif line.startswith('import '):
                    imports.append((line, i, line))
                    
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            self.stats['errors'] += 1
        
        return imports

    def update_imports_in_file(self, file_path: Path, dry_run: bool = True) -> Dict[str, int]:
        """
        Update import statements in a file.
        
        Args:
            file_path: Path to the file
            dry_run: If True, only show what would be changed
            
        Returns:
            Dictionary with update statistics
        """
        file_stats = {
            'imports_found': 0,
            'imports_updated': 0,
            'errors': 0
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            updated_content = content
            
            # Process each import mapping
            for class_name, mapping in self.import_mappings.items():
                for old_import in mapping['old_imports']:
                    if old_import in content:
                        file_stats['imports_found'] += 1
                        
                        # Choose the most appropriate new import
                        new_import = mapping['new_imports'][0]  # Use core import by default
                        
                        # Update the import
                        updated_content = updated_content.replace(old_import, new_import)
                        file_stats['imports_updated'] += 1
                        
                        logger.info(f"Updated import in {file_path}: {old_import} -> {new_import}")
            
            # Write the updated content if not dry run
            if not dry_run and updated_content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(updated_content)
                
                logger.info(f"Updated file: {file_path}")
            
        except Exception as e:
            logger.error(f"Error updating file {file_path}: {e}")
            file_stats['errors'] += 1
            self.stats['errors'] += 1
        
        return file_stats

    def update_all_imports(self, dry_run: bool = True) -> Dict[str, int]:
        """
        Update all import statements in the codebase.
        
        Args:
            dry_run: If True, only show what would be changed
            
        Returns:
            Dictionary with overall statistics
        """
        logger.info("Starting import update process...")
        
        python_files = self.find_python_files()
        
        for file_path in python_files:
            try:
                file_stats = self.update_imports_in_file(file_path, dry_run)
                
                self.stats['files_processed'] += 1
                self.stats['imports_updated'] += file_stats['imports_updated']
                self.stats['errors'] += file_stats['errors']
                
                if file_stats['imports_updated'] > 0:
                    logger.info(f"Updated {file_stats['imports_updated']} imports in {file_path}")
                
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                self.stats['errors'] += 1
        
        logger.info(f"Import update completed. Processed {self.stats['files_processed']} files, "
                   f"updated {self.stats['imports_updated']} imports, "
                   f"encountered {self.stats['errors']} errors")
        
        return self.stats

    def create_factory_usage_examples(self, output_file: str = "factory_usage_examples.py") -> None:
        """
        Create examples of how to use the factory functions.
        
        Args:
            output_file: Path to output file
        """
        examples = '''"""
Factory Usage Examples

This file demonstrates how to use the factory functions to create
instances of the production-ready base classes.

Key Benefits:
- Type-safe instantiation
- Automatic configuration validation
- Consistent interface across all base classes
- Easy switching between production and existing implementations
"""

# Import factory functions
from src.core.factory import (
    create_validator, create_training_step, create_clustering_algorithm,
    create_multi_output_model, create_pattern_discoverer, create_labeling_strategy,
    create_complete_pipeline, ConfigurationPresets
)

# Import enums and types
from src.core.abstract_base_classes import (
    ValidationLevel, TrainingStatus, ClusteringAlgorithm,
    PatternType, LabelingStrategy
)

# Example 1: Create individual components
def create_individual_components():
    """Create individual components using factory functions."""
    
    # Create a validator
    validator = create_validator(
        name="data_validator",
        validator_type="data",
        validation_level=ValidationLevel.PRODUCTION,
        config={
            'required_columns': ['price', 'volume', 'returns'],
            'min_samples': 100,
            'max_missing_ratio': 0.05
        }
    )
    
    # Create a training step
    training_step = create_training_step(
        name="ml_training",
        model_type="random_forest",
        config={
            'n_estimators': 200,
            'max_depth': 10,
            'scale_features': True
        }
    )
    
    # Create a clustering algorithm
    clustering = create_clustering_algorithm(
        name="kmeans_clustering",
        algorithm=ClusteringAlgorithm.KMEANS,
        n_clusters=5,
        config={
            'random_state': 42,
            'n_init': 10
        }
    )
    
    # Create a multi-output model
    model = create_multi_output_model(
        name="multi_output_rf",
        n_outputs=3,
        output_names=['signal_strength', 'confidence', 'risk_score'],
        config={
            'n_estimators': 150,
            'max_depth': 8
        }
    )
    
    # Create a pattern discoverer
    pattern_discoverer = create_pattern_discoverer(
        name="momentum_discoverer",
        pattern_type=PatternType.MOMENTUM,
        config={
            'lookback_period': 20,
            'momentum_threshold': 0.03,
            'confidence_threshold': 0.7
        }
    )
    
    # Create a labeling strategy
    labeling_strategy = create_labeling_strategy(
        name="profit_labeling",
        strategy=LabelingStrategy.PROFIT_BASED,
        config={
            'profit_threshold': 0.02,
            'lookforward_period': 5,
            'min_confidence': 0.6
        }
    )
    
    return {
        'validator': validator,
        'training_step': training_step,
        'clustering': clustering,
        'model': model,
        'pattern_discoverer': pattern_discoverer,
        'labeling_strategy': labeling_strategy
    }

# Example 2: Create complete pipeline
def create_complete_pipeline_example():
    """Create a complete ML pipeline using factory functions."""
    
    # Create production pipeline
    production_pipeline = create_complete_pipeline(
        pipeline_name="production_ml_pipeline",
        config_preset="production",
        use_production=True
    )
    
    # Create development pipeline
    development_pipeline = create_complete_pipeline(
        pipeline_name="development_ml_pipeline",
        config_preset="development",
        use_production=True
    )
    
    # Create testing pipeline
    testing_pipeline = create_complete_pipeline(
        pipeline_name="testing_ml_pipeline",
        config_preset="testing",
        use_production=True
    )
    
    return {
        'production': production_pipeline,
        'development': development_pipeline,
        'testing': testing_pipeline
    }

# Example 3: Use configuration presets
def use_configuration_presets():
    """Demonstrate how to use configuration presets."""
    
    # Get different configuration presets
    production_config = ConfigurationPresets.get_production_config()
    development_config = ConfigurationPresets.get_development_config()
    testing_config = ConfigurationPresets.get_testing_config()
    
    # Get specialized configurations
    ml_config = ConfigurationPresets.get_ml_pipeline_config()
    clustering_config = ConfigurationPresets.get_clustering_config()
    pattern_config = ConfigurationPresets.get_pattern_discovery_config()
    labeling_config = ConfigurationPresets.get_labeling_config()
    
    # Use presets with factory functions
    validator = create_validator(
        name="preset_validator",
        **production_config
    )
    
    training_step = create_training_step(
        name="preset_training",
        **ml_config
    )
    
    clustering = create_clustering_algorithm(
        name="preset_clustering",
        **clustering_config
    )
    
    return {
        'validator': validator,
        'training_step': training_step,
        'clustering': clustering
    }

# Example 4: Backward compatibility
def backward_compatibility_example():
    """Demonstrate backward compatibility with existing code."""
    
    # Create components using existing base classes
    existing_validator = create_validator(
        name="existing_validator",
        use_production=False
    )
    
    existing_training_step = create_training_step(
        name="existing_training",
        use_production=False
    )
    
    existing_clustering = create_clustering_algorithm(
        name="existing_clustering",
        use_production=False
    )
    
    # Create components using production base classes
    production_validator = create_validator(
        name="production_validator",
        use_production=True
    )
    
    production_training_step = create_training_step(
        name="production_training",
        use_production=True
    )
    
    production_clustering = create_clustering_algorithm(
        name="production_clustering",
        use_production=True
    )
    
    return {
        'existing': {
            'validator': existing_validator,
            'training_step': existing_training_step,
            'clustering': existing_clustering
        },
        'production': {
            'validator': production_validator,
            'training_step': production_training_step,
            'clustering': production_clustering
        }
    }

# Example 5: Error handling
def error_handling_example():
    """Demonstrate error handling in factory functions."""
    
    try:
        # This should work
        validator = create_validator(
            name="test_validator",
            validation_level=ValidationLevel.PRODUCTION
        )
        print(f"Created validator: {validator.name}")
        
    except Exception as e:
        print(f"Error creating validator: {e}")
    
    try:
        # This should fail
        invalid_validator = create_validator(
            name="",  # Empty name should cause error
            validation_level=ValidationLevel.PRODUCTION
        )
        
    except Exception as e:
        print(f"Expected error for empty name: {e}")

if __name__ == "__main__":
    # Run examples
    print("Creating individual components...")
    components = create_individual_components()
    print(f"Created {len(components)} components")
    
    print("\\nCreating complete pipelines...")
    pipelines = create_complete_pipeline_example()
    print(f"Created {len(pipelines)} pipelines")
    
    print("\\nUsing configuration presets...")
    preset_components = use_configuration_presets()
    print(f"Created {len(preset_components)} components with presets")
    
    print("\\nTesting backward compatibility...")
    compatibility_components = backward_compatibility_example()
    print(f"Created {len(compatibility_components)} component sets")
    
    print("\\nTesting error handling...")
    error_handling_example()
    
    print("\\nAll examples completed successfully!")
'''
        
        output_path = self.workspace_root / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(examples)
        
        logger.info(f"Created factory usage examples: {output_path}")

    def generate_migration_report(self, output_file: str = "migration_report.md") -> None:
        """
        Generate a migration report.
        
        Args:
            output_file: Path to output file
        """
        report = f'''# Base Classes Migration Report

## Overview
This report summarizes the migration of the codebase to use the new production-ready abstract base classes.

## Statistics
- Files Processed: {self.stats['files_processed']}
- Imports Updated: {self.stats['imports_updated']}
- Errors Encountered: {self.stats['errors']}
- Warnings: {self.stats['warnings']}

## Migration Details

### Base Classes Migrated
1. **BaseValidator** - Enhanced validation framework with production features
2. **BaseTrainingStep** - Complete ML workflow support with hardware optimization
3. **BaseClusteringAlgorithm** - Clustering framework with performance optimization
4. **MultiOutputModel** - Multi-output ML models with ensemble support
5. **BasePatternDiscoverer** - Pattern discovery framework with mathematical definitions
6. **BaseLabelingStrategy** - Labeling strategies with confidence calculation

### Key Improvements
- **Production Readiness**: All base classes now include comprehensive error handling, logging, and monitoring
- **Hardware Optimization**: Automatic M1 chip optimization and memory management
- **Type Safety**: Extensive type hints and protocol definitions
- **Performance Tracking**: Built-in performance metrics and monitoring
- **Validation**: Comprehensive input validation and data integrity checks
- **Documentation**: Extensive documentation and examples

### Backward Compatibility
- All existing classes continue to work without changes
- Factory functions provide easy migration path
- Configuration presets for common use cases
- Gradual migration support

### Usage Examples

#### Using Factory Functions
```python
from src.core.factory import create_validator, create_training_step

# Create validator
validator = create_validator(
    name="data_validator",
    validation_level=ValidationLevel.PRODUCTION
)

# Create training step
training_step = create_training_step(
    name="ml_training",
    model_type="random_forest"
)
```

#### Using Configuration Presets
```python
from src.core.factory import ConfigurationPresets, create_complete_pipeline

# Create complete pipeline with production config
pipeline = create_complete_pipeline(
    pipeline_name="production_pipeline",
    config_preset="production"
)
```

#### Backward Compatibility
```python
from src.core.factory import create_validator

# Use existing base class
existing_validator = create_validator(
    name="legacy_validator",
    use_production=False
)

# Use production base class
production_validator = create_validator(
    name="production_validator",
    use_production=True
)
```

## Next Steps
1. Update existing code to use factory functions
2. Migrate to production base classes gradually
3. Update tests to use new base classes
4. Monitor performance improvements
5. Update documentation and examples

## Support
For questions or issues with the migration, please refer to:
- Factory usage examples: `factory_usage_examples.py`
- Integration tests: `src/tests/test_integration_wiring.py`
- Base class documentation: `src/core/README.md`
'''
        
        output_path = self.workspace_root / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Generated migration report: {output_path}")

def main():
    """Main function to run the import updater."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Update import statements for base classes")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without making changes")
    parser.add_argument("--workspace-root", default="/workspace", help="Workspace root directory")
    parser.add_argument("--generate-examples", action="store_true", help="Generate factory usage examples")
    parser.add_argument("--generate-report", action="store_true", help="Generate migration report")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create updater
    updater = ImportUpdater(args.workspace_root)
    
    # Update imports
    if not args.dry_run:
        stats = updater.update_all_imports(dry_run=False)
        print(f"Updated {stats['imports_updated']} imports in {stats['files_processed']} files")
    else:
        stats = updater.update_all_imports(dry_run=True)
        print(f"Would update {stats['imports_updated']} imports in {stats['files_processed']} files")
    
    # Generate examples
    if args.generate_examples:
        updater.create_factory_usage_examples()
    
    # Generate report
    if args.generate_report:
        updater.generate_migration_report()

if __name__ == "__main__":
    main()