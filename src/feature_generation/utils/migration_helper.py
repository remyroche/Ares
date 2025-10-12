"""
Migration Helper for Feature Generators

This module provides utilities to help migrate existing feature generators
to use the new centralized utilities.
"""

import ast
import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class FeatureGeneratorMigrator:
    """Helper class for migrating feature generators to use centralized utilities."""
    
    def __init__(self):
        self.import_replacements = {
            'VectorizedFeatureGenerator': 'UnifiedFeatureGenerator',
            'FeatureConfig': 'UnifiedFeatureConfig',
            'get_vectorbt_rolling_optimizer': 'get_centralized_rolling_manager',
            'VectorBTScaler': 'get_scaler_factory',
            'create_vectorbt_scaler': 'get_scaler_factory'
        }
        
        self.method_replacements = {
            'rolling_mean': 'rolling_mean',
            'rolling_std': 'rolling_std',
            'rolling_var': 'rolling_var',
            'rolling_min': 'rolling_min',
            'rolling_max': 'rolling_max',
            'rolling_sum': 'rolling_sum',
            'rolling_median': 'rolling_median',
            'rolling_skew': 'rolling_skew',
            'rolling_kurt': 'rolling_kurt'
        }
        
        self.rolling_operation_patterns = [
            r'self\.rolling_optimizer\.rolling_(\w+)\(([^)]+)\)',
            r'rolling_(\w+)\(([^)]+)\)',
            r'data\.rolling\(window=(\w+)\)\.(\w+)\(\)'
        ]
    
    def migrate_file(self, file_path: str, output_path: Optional[str] = None) -> bool:
        """
        Migrate a single feature generator file.
        
        Args:
            file_path: Path to the file to migrate
            output_path: Optional output path (defaults to same file)
            
        Returns:
            True if migration was successful
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Apply migrations
            migrated_content = self._apply_migrations(content)
            
            # Write migrated content
            output_path = output_path or file_path
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(migrated_content)
            
            logger.info(f"Successfully migrated {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to migrate {file_path}: {e}")
            return False
    
    def _apply_migrations(self, content: str) -> str:
        """Apply all migrations to the content."""
        # 1. Update imports
        content = self._update_imports(content)
        
        # 2. Update class definitions
        content = self._update_class_definitions(content)
        
        # 3. Update configuration methods
        content = self._update_configuration_methods(content)
        
        # 4. Update rolling operations
        content = self._update_rolling_operations(content)
        
        # 5. Update scaling operations
        content = self._update_scaling_operations(content)
        
        # 6. Add centralized utility initialization
        content = self._add_centralized_utilities(content)
        
        return content
    
    def _update_imports(self, content: str) -> str:
        """Update import statements."""
        # Add new imports
        new_imports = [
            "from ..core.unified_feature_generator import UnifiedFeatureGenerator, UnifiedFeatureConfig",
            "from ..utils.centralized_rolling_manager import get_centralized_rolling_manager, RollingOperation",
            "from ..utils.scaler_factory import get_scaler_factory, ScalerType",
            "from ..utils.common_operations import get_common_operations"
        ]
        
        # Replace old imports
        for old_import, new_import in self.import_replacements.items():
            content = content.replace(old_import, new_import)
        
        # Add new imports if not already present
        for new_import in new_imports:
            if new_import not in content:
                # Find the last import statement and add after it
                import_pattern = r'(from\s+[^\n]+\n|import\s+[^\n]+\n)'
                matches = list(re.finditer(import_pattern, content))
                if matches:
                    last_import = matches[-1]
                    content = (content[:last_import.end()] + 
                             new_import + '\n' + 
                             content[last_import.end():])
        
        return content
    
    def _update_class_definitions(self, content: str) -> str:
        """Update class definitions to use UnifiedFeatureGenerator."""
        # Replace base class
        content = re.sub(
            r'class\s+(\w+)\(VectorizedFeatureGenerator\)',
            r'class \1(UnifiedFeatureGenerator)',
            content
        )
        
        # Update constructor calls
        content = re.sub(
            r'super\(\)\.__init__\(config\)',
            r'super().__init__(config)',
            content
        )
        
        return content
    
    def _update_configuration_methods(self, content: str) -> str:
        """Update configuration methods to use UnifiedFeatureConfig."""
        # Replace FeatureConfig with UnifiedFeatureConfig
        content = re.sub(
            r'FeatureConfig\(',
            'UnifiedFeatureConfig(',
            content
        )
        
        # Add default configuration parameters
        config_pattern = r'UnifiedFeatureConfig\(\s*([^)]+)\)'
        
        def add_default_params(match):
            config_content = match.group(1)
            
            # Check if auto_normalize is already present
            if 'auto_normalize' not in config_content:
                config_content += ',\n            auto_normalize=True,\n            normalization_method=\'zscore\',\n            normalization_feature_type=\'default\',\n            enable_batch_processing=True'
            
            return f'UnifiedFeatureConfig(\n            {config_content}\n        )'
        
        content = re.sub(config_pattern, add_default_params, content, flags=re.MULTILINE | re.DOTALL)
        
        return content
    
    def _update_rolling_operations(self, content: str) -> str:
        """Update rolling operations to use centralized manager."""
        # Replace manual rolling operations with centralized methods
        for pattern in self.rolling_operation_patterns:
            content = re.sub(pattern, self._replace_rolling_operation, content)
        
        # Replace manual VectorBT optimization blocks
        content = self._replace_vectorbt_optimization_blocks(content)
        
        return content
    
    def _replace_rolling_operation(self, match) -> str:
        """Replace a single rolling operation."""
        operation = match.group(1) if len(match.groups()) > 0 else 'mean'
        args = match.group(2) if len(match.groups()) > 1 else ''
        
        # Map to centralized method
        if operation in self.method_replacements:
            return f'self.rolling_{operation}({args})'
        else:
            return f'self.rolling_operation(RollingOperation.{operation.upper()}, {args})'
    
    def _replace_vectorbt_optimization_blocks(self, content: str) -> str:
        """Replace VectorBT optimization blocks with centralized calls."""
        # Pattern for VectorBT optimization blocks
        pattern = r'if\s+self\.rolling_optimizer.*?except.*?return\s+[^}]+}'
        
        def replace_optimization_block(match):
            block = match.group(0)
            
            # Extract the operation and parameters
            operation_match = re.search(r'rolling_(\w+)\(([^)]+)\)', block)
            if operation_match:
                operation = operation_match.group(1)
                args = operation_match.group(2)
                return f'self.rolling_{operation}({args})'
            
            return block
        
        content = re.sub(pattern, replace_optimization_block, content, flags=re.MULTILINE | re.DOTALL)
        
        return content
    
    def _update_scaling_operations(self, content: str) -> str:
        """Update scaling operations to use centralized factory."""
        # Replace manual scaler creation
        content = re.sub(
            r'VectorBTScaler\([^)]+\)',
            'self.scaler_factory.create_scaler("zscore")',
            content
        )
        
        # Replace manual normalization
        content = re.sub(
            r'scaler\.fit_transform\(([^)]+)\)',
            r'self.normalize_feature(\1)',
            content
        )
        
        return content
    
    def _add_centralized_utilities(self, content: str) -> str:
        """Add centralized utility initialization if not present."""
        if 'self.rolling_manager' not in content:
            # Add to __init__ method
            init_pattern = r'(def __init__\(self[^)]*\):\s*[^}]+)(super\(\)\.__init__\(config\))'
            
            def add_utilities(match):
                init_content = match.group(1)
                super_call = match.group(2)
                
                utilities = '''
        # Initialize centralized utilities
        self.rolling_manager = get_centralized_rolling_manager()
        self.scaler_factory = get_scaler_factory()
        self.common_operations = get_common_operations()
        '''
                
                return init_content + super_call + utilities
            
            content = re.sub(init_pattern, add_utilities, content, flags=re.MULTILINE | re.DOTALL)
        
        return content
    
    def migrate_directory(self, directory_path: str, pattern: str = "*.py") -> Dict[str, bool]:
        """
        Migrate all files in a directory matching the pattern.
        
        Args:
            directory_path: Path to the directory
            pattern: File pattern to match
            
        Returns:
            Dictionary mapping file paths to migration success status
        """
        results = {}
        directory = Path(directory_path)
        
        for file_path in directory.glob(pattern):
            if file_path.is_file():
                results[str(file_path)] = self.migrate_file(str(file_path))
        
        return results
    
    def generate_migration_report(self, results: Dict[str, bool]) -> str:
        """Generate a migration report."""
        total_files = len(results)
        successful = sum(1 for success in results.values() if success)
        failed = total_files - successful
        
        report = f"""
Migration Report
================

Total files processed: {total_files}
Successful migrations: {successful}
Failed migrations: {failed}
Success rate: {(successful/total_files)*100:.1f}%

Files processed:
"""
        
        for file_path, success in results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            report += f"  {file_path}: {status}\n"
        
        return report

def migrate_feature_generators(directory_path: str = "src/feature_generation/categories") -> None:
    """
    Migrate all feature generators in the specified directory.
    
    Args:
        directory_path: Path to the categories directory
    """
    migrator = FeatureGeneratorMigrator()
    
    # Migrate all Python files in the directory
    results = migrator.migrate_directory(directory_path, "*.py")
    
    # Generate and print report
    report = migrator.generate_migration_report(results)
    print(report)
    
    # Save report to file
    with open("migration_report.txt", "w") as f:
        f.write(report)

if __name__ == "__main__":
    migrate_feature_generators()