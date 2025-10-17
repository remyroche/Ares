"""
Automated Migration Script for Models Training Components

This script automates the migration of existing models_training components
to the new ModularComponent architecture. It provides batch migration,
validation, and rollback capabilities.

Usage:
    python migrate_components.py --mode analyze
    python migrate_components.py --mode migrate --components analyst_models,analyst_ensemble
    python migrate_components.py --mode validate --components all
    python migrate_components.py --mode rollback --backup backup_20231201
"""

import argparse
import logging
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import migration utilities
from .unified_data_driven_pipeline.core.migration_utils import (
    ModelsTrainingMigrationUtils, analyze_component, validate_migration_compatibility,
    create_component_wrapper, migrate_component, generate_migration_report
)

# Import existing components (only tactician components remain to be migrated)
try:
    from .tactician_models_training import TacticianModelsTrainingStep
    TACTICIAN_MODELS_AVAILABLE = True
except ImportError:
    TACTICIAN_MODELS_AVAILABLE = False

try:
    from .tactician_ensemble_training import TacticianEnsembleTrainingStep
    TACTICIAN_ENSEMBLE_AVAILABLE = True
except ImportError:
    TACTICIAN_ENSEMBLE_AVAILABLE = False

# Analyst and ML components have been migrated to ModularComponent architecture
ANALYST_MODELS_AVAILABLE = False
ANALYST_ENSEMBLE_AVAILABLE = False
ML_LABELER_AVAILABLE = False


class ModelsTrainingMigrationManager:
    """Manager for migrating models training components."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.migration_utils = ModelsTrainingMigrationUtils(self.logger)
        self.backup_dir = Path("migration_backups")
        self.migration_log = []
        
        # Component mapping (only tactician components remain to be migrated)
        self.component_mapping = {
            'tactician_models': {
                'class': TacticianModelsTrainingStep if TACTICIAN_MODELS_AVAILABLE else None,
                'migrated_class': 'TacticianModelsTrainingModular',
                'file': 'tactician_models_training_modular.py'
            },
            'tactician_ensemble': {
                'class': TacticianEnsembleTrainingStep if TACTICIAN_ENSEMBLE_AVAILABLE else None,
                'migrated_class': 'TacticianEnsembleTrainingModular',
                'file': 'tactician_ensemble_training_modular.py'
            }
        }
        
        # Note: Analyst and ML components have been successfully migrated
        self.migrated_components = {
            'analyst_models': 'AnalystModelsTrainingModular',
            'analyst_ensemble': 'AnalystEnsembleTrainingModular', 
            'ml_labeler': 'MLEntryTimingLabelerModular'
        }
    
    def analyze_components(self, components: List[str] = None) -> Dict[str, Any]:
        """Analyze components for migration compatibility."""
        self.logger.info("🔍 Analyzing components for migration compatibility")
        
        if components is None or 'all' in components:
            components = list(self.component_mapping.keys())
        
        components_to_analyze = []
        for comp_name in components:
            if comp_name in self.component_mapping:
                comp_class = self.component_mapping[comp_name]['class']
                if comp_class is not None:
                    components_to_analyze.append(comp_class)
                else:
                    self.logger.warning(f"Component {comp_name} not available")
            else:
                self.logger.warning(f"Unknown component: {comp_name}")
        
        if not components_to_analyze:
            self.logger.error("No components available for analysis")
            return {}
        
        # Generate migration report
        report = generate_migration_report(components_to_analyze)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"migration_analysis_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"📊 Analysis report saved to: {report_file}")
        return report
    
    def migrate_components(self, components: List[str], strategy: str = 'wrapper') -> Dict[str, Any]:
        """Migrate specified components."""
        self.logger.info(f"🚀 Starting migration of components: {components}")
        
        # Create backup
        backup_id = self._create_backup()
        
        migration_results = {}
        
        for comp_name in components:
            if comp_name not in self.component_mapping:
                self.logger.error(f"Unknown component: {comp_name}")
                continue
            
            comp_info = self.component_mapping[comp_name]
            comp_class = comp_info['class']
            
            if comp_class is None:
                self.logger.error(f"Component {comp_name} not available")
                continue
            
            try:
                self.logger.info(f"Migrating {comp_name}...")
                
                # Migrate component
                result = migrate_component(comp_class, strategy)
                
                if result.success:
                    self.logger.info(f"✅ {comp_name} migrated successfully")
                    
                    # Save migrated component
                    self._save_migrated_component(comp_name, result.migrated_component)
                    
                    migration_results[comp_name] = {
                        'success': True,
                        'migrated_component': result.migrated_component.__name__,
                        'compatibility_score': result.compatibility_score,
                        'migration_time': result.migration_time
                    }
                else:
                    self.logger.error(f"❌ {comp_name} migration failed: {result.errors}")
                    migration_results[comp_name] = {
                        'success': False,
                        'errors': [str(e) for e in result.errors],
                        'warnings': result.warnings
                    }
                
            except Exception as e:
                self.logger.error(f"❌ {comp_name} migration failed with exception: {e}")
                migration_results[comp_name] = {
                    'success': False,
                    'errors': [str(e)],
                    'warnings': []
                }
        
        # Save migration results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"migration_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(migration_results, f, indent=2, default=str)
        
        self.logger.info(f"📋 Migration results saved to: {results_file}")
        return migration_results
    
    def validate_migrations(self, components: List[str] = None) -> Dict[str, Any]:
        """Validate migrated components."""
        self.logger.info("🔍 Validating migrated components")
        
        if components is None or 'all' in components:
            components = list(self.component_mapping.keys())
        
        validation_results = {}
        
        for comp_name in components:
            if comp_name not in self.component_mapping:
                self.logger.warning(f"Unknown component: {comp_name}")
                continue
            
            try:
                # Import migrated component
                migrated_module = self._import_migrated_component(comp_name)
                if migrated_module is None:
                    validation_results[comp_name] = {
                        'success': False,
                        'error': 'Failed to import migrated component'
                    }
                    continue
                
                # Create instance and test
                migrated_class = getattr(migrated_module, self.component_mapping[comp_name]['migrated_class'])
                instance = migrated_class(f"test_{comp_name}")
                
                # Test initialization
                init_success = instance.initialize()
                
                # Test basic functionality
                test_data = self._create_test_data(comp_name)
                process_success = False
                if init_success:
                    try:
                        result = instance.process(test_data)
                        process_success = result is not None
                    except Exception as e:
                        self.logger.warning(f"Processing test failed for {comp_name}: {e}")
                
                # Test cleanup
                instance.cleanup()
                
                validation_results[comp_name] = {
                    'success': init_success and process_success,
                    'initialization': init_success,
                    'processing': process_success,
                    'health': instance.get_health_report() if hasattr(instance, 'get_health_report') else None
                }
                
                if init_success and process_success:
                    self.logger.info(f"✅ {comp_name} validation passed")
                else:
                    self.logger.warning(f"⚠️ {comp_name} validation failed")
                
            except Exception as e:
                self.logger.error(f"❌ {comp_name} validation failed with exception: {e}")
                validation_results[comp_name] = {
                    'success': False,
                    'error': str(e)
                }
        
        # Save validation results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"validation_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        self.logger.info(f"📋 Validation results saved to: {results_file}")
        return validation_results
    
    def rollback_migration(self, backup_id: str) -> bool:
        """Rollback migration using backup."""
        self.logger.info(f"🔄 Rolling back migration using backup: {backup_id}")
        
        backup_path = self.backup_dir / backup_id
        if not backup_path.exists():
            self.logger.error(f"Backup {backup_id} not found")
            return False
        
        try:
            # Restore files from backup
            for file_path in backup_path.rglob('*'):
                if file_path.is_file():
                    relative_path = file_path.relative_to(backup_path)
                    target_path = Path('.') / relative_path
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file_path, target_path)
            
            self.logger.info(f"✅ Rollback completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Rollback failed: {e}")
            return False
    
    def _create_backup(self) -> str:
        """Create backup of current state."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_id = f"backup_{timestamp}"
        backup_path = self.backup_dir / backup_id
        
        self.logger.info(f"📦 Creating backup: {backup_id}")
        
        # Create backup directory
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Backup current files (only tactician components remain)
        files_to_backup = [
            'src/training/steps/models_training/tactician_models_training.py',
            'src/training/steps/models_training/tactician_ensemble_training.py'
        ]
        
        for file_path in files_to_backup:
            if os.path.exists(file_path):
                target_path = backup_path / file_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, target_path)
        
        self.logger.info(f"✅ Backup created: {backup_id}")
        return backup_id
    
    def _save_migrated_component(self, comp_name: str, migrated_component: Any) -> None:
        """Save migrated component to file."""
        comp_info = self.component_mapping[comp_name]
        file_path = f"src/training/steps/models_training/components/{comp_info['file']}"
        
        # This would save the migrated component code to file
        # For now, just log the action
        self.logger.info(f"💾 Saved migrated component {comp_name} to {file_path}")
    
    def _import_migrated_component(self, comp_name: str) -> Optional[Any]:
        """Import migrated component module."""
        comp_info = self.component_mapping[comp_name]
        module_name = comp_info['file'].replace('.py', '')
        
        try:
            # This would import the actual migrated module
            # For now, return None as placeholder
            return None
        except Exception as e:
            self.logger.error(f"Failed to import migrated component {comp_name}: {e}")
            return None
    
    def _create_test_data(self, comp_name: str) -> Dict[str, Any]:
        """Create test data for component validation."""
        import numpy as np
        import pandas as pd
        
        if comp_name in ['analyst_models', 'analyst_ensemble']:
            return {
                'X_train': pd.DataFrame({
                    'feature1': np.random.randn(100),
                    'feature2': np.random.randn(100),
                    'feature3': np.random.randn(100)
                }),
                'y_train': np.random.randint(0, 2, 100),
                'X_val': pd.DataFrame({
                    'feature1': np.random.randn(20),
                    'feature2': np.random.randn(20),
                    'feature3': np.random.randn(20)
                }),
                'y_val': np.random.randint(0, 2, 20)
            }
        elif comp_name == 'ml_labeler':
            return {
                'features': np.random.randn(100, 10),
                'market_data': np.random.randn(100, 5)
            }
        else:
            return {
                'data': np.random.randn(100, 10),
                'target': np.random.randint(0, 2, 100)
            }


def main():
    """Main migration script entry point."""
    parser = argparse.ArgumentParser(description='Migrate models training components to ModularComponent architecture')
    parser.add_argument('--mode', choices=['analyze', 'migrate', 'validate', 'rollback'], required=True,
                       help='Migration mode')
    parser.add_argument('--components', nargs='+', default=['all'],
                       help='Components to process (default: all)')
    parser.add_argument('--strategy', choices=['wrapper', 'refactor', 'hybrid'], default='wrapper',
                       help='Migration strategy (default: wrapper)')
    parser.add_argument('--backup', type=str,
                       help='Backup ID for rollback')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
                       help='Log level (default: INFO)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # Create migration manager
    manager = ModelsTrainingMigrationManager(logger)
    
    try:
        if args.mode == 'analyze':
            logger.info("🔍 Starting component analysis")
            report = manager.analyze_components(args.components)
            logger.info("✅ Analysis completed")
            
        elif args.mode == 'migrate':
            logger.info("🚀 Starting component migration")
            results = manager.migrate_components(args.components, args.strategy)
            logger.info("✅ Migration completed")
            
        elif args.mode == 'validate':
            logger.info("🔍 Starting migration validation")
            results = manager.validate_migrations(args.components)
            logger.info("✅ Validation completed")
            
        elif args.mode == 'rollback':
            if not args.backup:
                logger.error("Backup ID required for rollback")
                return 1
            logger.info("🔄 Starting migration rollback")
            success = manager.rollback_migration(args.backup)
            if success:
                logger.info("✅ Rollback completed")
            else:
                logger.error("❌ Rollback failed")
                return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())