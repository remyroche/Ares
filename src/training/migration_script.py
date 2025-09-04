"""Migration script to update the training system to use the new BaseStep pattern.

This script helps with:
1. Creating template migrations for remaining steps
2. Updating imports in existing code
3. Moving files to the new directory structure
"""
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import asyncio
STEP_MAPPING = {'01': ('data_preparation', 'data_collection'), '01_5': ('data_preparation', 'data_converter'), '02': ('data_preparation', 'data_reading'), '02_5': ('data_preparation', 'sr_optimization'), '03': ('market_analysis', 'hmm_regime_discovery'), '03_5': ('market_analysis', 'final_regime_clustering'), '04': ('market_analysis', 'regime_data_splitting'), '05': ('model_training', 'labeling'), '04_5': ('model_training', 'triple_barrier_method'), '06': ('feature_engineering', 'feature_engineering'), '07': ('model_training', 'enhanced_matrix_operations'), '08': ('model_training', 'feature_selection'), '09': ('model_training', 'hmm_based_training'), '09_5': ('model_training', 'multi_timeframe_hmm_ensemble'), '10': ('model_training', 'unified_regime_intelligence'), '11': ('model_training', 'analyst_creation'), '12': ('model_training', 'analyst_enhancement'), '13': ('model_training', 'analyst_ensemble_creation'), '14': ('model_training', 'tactician_labeling'), '15': ('model_training', 'tactician_specialist_training'), '16': ('validation', 'confidence_calibration'), '17': ('validation', 'final_parameters_optimization'), '18': ('validation', 'walk_forward_validation'), '19': ('validation', 'monte_carlo_validation'), '20': ('validation', 'ab_testing'), '21': ('model_training', 'saving')}

def create_step_template(step_num: str, category: str, step_name: str) -> str:
    """Create a template for migrating a step to BaseStep pattern.
    
    Args:
        step_num: Step number (e.g., "07", "09_5")
        category: Category directory (e.g., "model_training")
        step_name: Step name (e.g., "enhanced_matrix_operations")
        
    Returns:
        Template code for the step
    """
    class_name = ''.join((word.capitalize() for word in step_name.split('_'))) + 'Step'
    template = f'''"""Step {step_num}: {step_name.replace('_', ' ').title()} - Refactored to use BaseStep.\n\nThis module implements {step_name.replace('_', ' ')} functionality.\n"""\n\nfrom typing import Any, Dict, Tuple, Optional\nfrom pathlib import Path\nimport pandas as pd\nimport numpy as np\nimport json\n\nfrom src.training.base_step import BaseStep\nfrom src.utils.logger import system_logger\nfrom src.core.decorators import handles_errors\n\n\nclass {class_name}(BaseStep):\n    """Step {step_num}: {step_name.replace('_', ' ').title()} using standardized base class."""\n    \n    def __init__(self, config: Dict[str, Any]):\n        """Initialize {step_name.replace('_', ' ')} step.\n        \n        Args:\n            config: Configuration dictionary\n        """\n        super().__init__(config, "{step_num}", "{step_name}")\n        \n        # Step-specific configuration\n        # TODO: Add specific configuration parameters\n        \n    def _initialize_step(self) -> None:\n        """Initialize step-specific components."""\n        # TODO: Initialize any step-specific components\n        self.logger.info("✅ {step_name.replace('_', ' ').title()} step initialized")\n    \n    def validate_inputs(\n        self, \n        training_input: Dict[str, Any], \n        pipeline_state: Dict[str, Any]\n    ) -> Tuple[bool, list]:\n        """Validate step inputs.\n        \n        Args:\n            training_input: Training input parameters\n            pipeline_state: Current pipeline state\n            \n        Returns:\n            Tuple of (is_valid, errors)\n        """\n        errors = []\n        \n        # TODO: Add input validation logic\n        \n        return len(errors) == 0, errors\n    \n    @handles_errors(\n        exceptions=(Exception,),\n        default_return={{"success": False}},\n        context="{step_name.replace('_', ' ')} execution"\n    )\n    async def execute_logic(\n        self,\n        training_input: Dict[str, Any],\n        pipeline_state: Dict[str, Any]\n    ) -> Dict[str, Any]:\n        """Execute {step_name.replace('_', ' ')} logic.\n        \n        Args:\n            training_input: Training input parameters\n            pipeline_state: Current pipeline state\n            \n        Returns:\n            Updated pipeline state\n        """\n        self.logger.info("🚀 Starting {step_name.replace('_', ' ')}...")\n        \n        # TODO: Implement step logic\n        \n        return pipeline_state\n    \n    def validate_outputs(self, pipeline_state: Dict[str, Any]) -> Tuple[bool, list]:\n        """Validate step outputs.\n        \n        Args:\n            pipeline_state: Updated pipeline state\n            \n        Returns:\n            Tuple of (is_valid, errors)\n        """\n        errors = []\n        \n        # TODO: Add output validation logic\n        \n        return len(errors) == 0, errors\n    \n    def get_required_inputs(self) -> list:\n        """Get list of required inputs for this step."""\n        # TODO: Update with actual required inputs\n        return []\n    \n    def get_produced_outputs(self) -> list:\n        """Get list of outputs produced by this step."""\n        # TODO: Update with actual outputs\n        return []\n    \n    def get_dependencies(self) -> list:\n        """Get list of step dependencies."""\n        # TODO: Update with actual dependencies\n        return []\n'''
    return template

def update_imports_in_file(file_path: Path, old_imports: List[str], new_imports: List[str]) -> bool:
    """Update imports in a Python file.
    
    Args:
        file_path: Path to the file
        old_imports: List of old import statements
        new_imports: List of new import statements
        
    Returns:
        True if file was updated
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        original_content = content
        for old_import, new_import in zip(old_imports, new_imports):
            content = content.replace(old_import, new_import)
        if content != original_content:
            with open(file_path, 'w') as f:
                f.write(content)
            return True
    except Exception as e:
        print(f'Error updating {file_path}: {e}')
    return False

def find_files_with_old_imports(root_dir: Path) -> List[Path]:
    """Find all Python files using old training manager imports.
    
    Args:
        root_dir: Root directory to search
        
    Returns:
        List of file paths
    """
    old_import_patterns = ['from src.training.enhanced_training_manager import', 'from src.training.training_manager import', 'from src.training.steps.step', 'import src.training.enhanced_training_manager', 'import src.training.training_manager']
    files_to_update = []
    for py_file in root_dir.rglob('*.py'):
        try:
            with open(py_file, 'r') as f:
                content = f.read()
            for pattern in old_import_patterns:
                if pattern in content:
                    files_to_update.append(py_file)
                    break
        except Exception as e:
            print(f'Error reading {py_file}: {e}')
    return files_to_update

def create_migration_summary() -> str:
    """Create a summary of the migration status.
    
    Returns:
        Migration summary report
    """
    summary = ['Training System Migration Summary', '=' * 50, '']
    steps_dir = Path('src/training/steps')
    migrated = []
    pending = []
    for step_num, (category, step_name) in STEP_MAPPING.items():
        new_path = steps_dir / category / f'step{step_num}_{step_name}.py'
        old_path = steps_dir / f'step{step_num}_{step_name}.py'
        if new_path.exists():
            migrated.append(f'Step {step_num}: {step_name} ✅')
        elif old_path.exists():
            pending.append(f'Step {step_num}: {step_name} ⏳')
        else:
            pending.append(f'Step {step_num}: {step_name} ❌ (not found)')
    summary.append(f'Migrated Steps ({len(migrated)}):')
    summary.extend((f'  {item}' for item in migrated))
    summary.append(f'\nPending Steps ({len(pending)}):')
    summary.extend((f'  {item}' for item in pending))
    files_to_update = find_files_with_old_imports(Path('src'))
    summary.append(f'\nFiles needing import updates: {len(files_to_update)}')
    return '\n'.join(summary)

def main() -> None:
    """Main migration function."""
    print('🚀 Training System Migration Tool')
    print('=' * 50)
    print('\n' + create_migration_summary())
    print('\n\nCreating templates for remaining steps...')
    templates_dir = Path('src/training/steps/migration_templates')
    templates_dir.mkdir(exist_ok=True)
    for step_num, (category, step_name) in STEP_MAPPING.items():
        template_path = templates_dir / f'step{step_num}_{step_name}_template.py'
        if not template_path.exists():
            template = create_step_template(step_num, category, step_name)
            with open(template_path, 'w') as f:
                f.write(template)
            print(f'  Created template for Step {step_num}: {step_name}')
    print(f'\nTemplates created in: {templates_dir}')
    print('\n\nFiles that need import updates:')
    files_to_update = find_files_with_old_imports(Path('src'))
    for file_path in files_to_update[:10]:
        print(f'  - {file_path}')
    if len(files_to_update) > 10:
        print(f'  ... and {len(files_to_update) - 10} more files')
    print('\n✅ Migration analysis complete!')
    print('\nNext steps:')
    print('1. Review and complete the step templates in migration_templates/')
    print('2. Move completed steps to their appropriate directories')
    print('3. Update imports in dependent files')
    print('4. Test the migrated pipeline')
if __name__ == '__main__':
    main()