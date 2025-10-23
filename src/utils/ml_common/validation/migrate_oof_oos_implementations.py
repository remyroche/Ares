"""
OOF/OOS Implementation Migration Script

This script helps migrate existing OOF/OOS implementations to use the enhanced
consolidated utilities. It provides automated refactoring suggestions and
can perform basic code transformations.

Usage:
    python migrate_oof_oos_implementations.py --target-file path/to/file.py
    python migrate_oof_oos_implementations.py --target-directory path/to/directory
    python migrate_oof_oos_implementations.py --dry-run --target-file path/to/file.py
"""

import os
import re
import ast
import argparse
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import shutil
from datetime import datetime

logger = logging.getLogger(__name__)


class OOFOOSMigrationAnalyzer:
    """Analyzes code for OOF/OOS patterns and suggests migrations."""
    
    def __init__(self):
        """Initialize the migration analyzer."""
        self.logger = logging.getLogger(f"{__name__}.OOFOOSMigrationAnalyzer")
        
        # Patterns to detect OOF/OOS usage
        self.oof_patterns = [
            r'oof_stacking_ensemble_manager',
            r'enhanced_oof_stacking_with_confidence',
            r'create_oof_stacking_ensemble',
            r'train_oof_stacking_ensemble',
            r'evaluate_oof_performance',
            r'_generate_oof_predictions',
            r'_generate_oof_model_predictions',
            r'OOFStackingEnsembleManager',
            r'EnhancedOOFStackingEnsembleManager'
        ]
        
        self.oos_patterns = [
            r'_compute_oos_sharpe',
            r'_oos_sharpe_nested',
            r'oos_sharpe_nested_vectorized',
            r'validate_oos',
            r'oos_validation',
            r'sharpe_ratio.*nested',
            r'nested.*sharpe'
        ]
        
        self.leakage_patterns = [
            r'leakage_detection_system',
            r'LeakageDetector',
            r'detect_leakage',
            r'leakage.*detection'
        ]
        
        # Migration mappings
        self.import_mappings = {
            'from src.utils.ml_common.ensembles.oof_stacking_ensemble_manager import OOFStackingEnsembleManager': 
                'from src.utils.ml_common.validation.enhanced_consolidated_oof_oos import EnhancedConsolidatedOOFGenerator',
            'from src.utils.ml_common.ensembles.enhanced_oof_stacking_with_confidence import EnhancedOOFStackingEnsembleManager':
                'from src.utils.ml_common.validation.enhanced_consolidated_oof_oos import EnhancedConsolidatedOOFGenerator',
            'from src.utils.ml_common.training.training_utils import create_oof_stacking_ensemble':
                'from src.utils.ml_common.validation.enhanced_consolidated_oof_oos import create_enhanced_oof_generator',
            'from leakage_detection_system import LeakageDetector':
                'from src.utils.ml_common.validation.enhanced_consolidated_oof_oos import create_enhanced_oof_generator'
        }
        
        self.class_mappings = {
            'OOFStackingEnsembleManager': 'EnhancedConsolidatedOOFGenerator',
            'EnhancedOOFStackingEnsembleManager': 'EnhancedConsolidatedOOFGenerator',
            'LeakageDetector': 'create_enhanced_oof_generator'
        }
        
        self.method_mappings = {
            'create_oof_stacking_ensemble': 'create_enhanced_oof_generator',
            'train_oof_stacking_ensemble': 'generate_oof_predictions',
            'evaluate_oof_performance': 'generate_oof_predictions',
            'detect_leakage': 'generate_oof_predictions'
        }
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a file for OOF/OOS patterns."""
        self.logger.info(f"🔍 Analyzing file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            analysis = {
                'file_path': file_path,
                'oof_patterns_found': [],
                'oos_patterns_found': [],
                'leakage_patterns_found': [],
                'suggested_migrations': [],
                'complexity_score': 0,
                'migration_priority': 'low'
            }
            
            # Check for OOF patterns
            for pattern in self.oof_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    analysis['oof_patterns_found'].append(pattern)
                    analysis['complexity_score'] += 1
            
            # Check for OOS patterns
            for pattern in self.oos_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    analysis['oos_patterns_found'].append(pattern)
                    analysis['complexity_score'] += 1
            
            # Check for leakage patterns
            for pattern in self.leakage_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    analysis['leakage_patterns_found'].append(pattern)
                    analysis['complexity_score'] += 1
            
            # Generate migration suggestions
            analysis['suggested_migrations'] = self._generate_migration_suggestions(content)
            
            # Determine migration priority
            if analysis['complexity_score'] >= 5:
                analysis['migration_priority'] = 'high'
            elif analysis['complexity_score'] >= 3:
                analysis['migration_priority'] = 'medium'
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing file {file_path}: {e}")
            return {
                'file_path': file_path,
                'error': str(e),
                'migration_priority': 'low'
            }
    
    def _generate_migration_suggestions(self, content: str) -> List[Dict[str, str]]:
        """Generate migration suggestions for the content."""
        suggestions = []
        
        # Check for import statements to migrate
        for old_import, new_import in self.import_mappings.items():
            if old_import in content:
                suggestions.append({
                    'type': 'import',
                    'old': old_import,
                    'new': new_import,
                    'description': f'Replace import: {old_import} -> {new_import}'
                })
        
        # Check for class usage to migrate
        for old_class, new_class in self.class_mappings.items():
            if old_class in content:
                suggestions.append({
                    'type': 'class',
                    'old': old_class,
                    'new': new_class,
                    'description': f'Replace class: {old_class} -> {new_class}'
                })
        
        # Check for method usage to migrate
        for old_method, new_method in self.method_mappings.items():
            if old_method in content:
                suggestions.append({
                    'type': 'method',
                    'old': old_method,
                    'new': new_method,
                    'description': f'Replace method: {old_method} -> {new_method}'
                })
        
        return suggestions
    
    def analyze_directory(self, directory_path: str) -> Dict[str, Any]:
        """Analyze a directory for OOF/OOS patterns."""
        self.logger.info(f"🔍 Analyzing directory: {directory_path}")
        
        results = {
            'directory_path': directory_path,
            'files_analyzed': 0,
            'files_with_oof_oos': 0,
            'high_priority_files': [],
            'medium_priority_files': [],
            'low_priority_files': [],
            'total_suggestions': 0
        }
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    analysis = self.analyze_file(file_path)
                    
                    results['files_analyzed'] += 1
                    
                    if analysis.get('oof_patterns_found') or analysis.get('oos_patterns_found') or analysis.get('leakage_patterns_found'):
                        results['files_with_oof_oos'] += 1
                        results['total_suggestions'] += len(analysis.get('suggested_migrations', []))
                        
                        if analysis['migration_priority'] == 'high':
                            results['high_priority_files'].append(analysis)
                        elif analysis['migration_priority'] == 'medium':
                            results['medium_priority_files'].append(analysis)
                        else:
                            results['low_priority_files'].append(analysis)
        
        return results


class OOFOOSMigrationTransformer:
    """Transforms code to use enhanced consolidated utilities."""
    
    def __init__(self):
        """Initialize the migration transformer."""
        self.logger = logging.getLogger(f"{__name__}.OOFOOSMigrationTransformer")
    
    def transform_file(self, file_path: str, dry_run: bool = True) -> Dict[str, Any]:
        """Transform a file to use enhanced consolidated utilities."""
        self.logger.info(f"🔄 Transforming file: {file_path} (dry_run={dry_run})")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            transformed_content = original_content
            transformations = []
            
            # Apply import transformations
            for old_import, new_import in self.import_mappings.items():
                if old_import in transformed_content:
                    transformed_content = transformed_content.replace(old_import, new_import)
                    transformations.append({
                        'type': 'import',
                        'old': old_import,
                        'new': new_import
                    })
            
            # Apply class transformations
            for old_class, new_class in self.class_mappings.items():
                if old_class in transformed_content:
                    transformed_content = transformed_content.replace(old_class, new_class)
                    transformations.append({
                        'type': 'class',
                        'old': old_class,
                        'new': new_class
                    })
            
            # Apply method transformations
            for old_method, new_method in self.method_mappings.items():
                if old_method in transformed_content:
                    transformed_content = transformed_content.replace(old_method, new_method)
                    transformations.append({
                        'type': 'method',
                        'old': old_method,
                        'new': new_method
                    })
            
            # Apply specific pattern transformations
            transformed_content = self._apply_pattern_transformations(transformed_content, transformations)
            
            # Write transformed content if not dry run
            if not dry_run and transformed_content != original_content:
                # Create backup
                backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                shutil.copy2(file_path, backup_path)
                self.logger.info(f"📁 Created backup: {backup_path}")
                
                # Write transformed content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(transformed_content)
                self.logger.info(f"✅ Transformed file: {file_path}")
            
            return {
                'file_path': file_path,
                'transformations_applied': len(transformations),
                'transformations': transformations,
                'dry_run': dry_run,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error transforming file {file_path}: {e}")
            return {
                'file_path': file_path,
                'error': str(e),
                'success': False
            }
    
    def _apply_pattern_transformations(self, content: str, transformations: List[Dict]) -> str:
        """Apply specific pattern transformations."""
        # Transform OOFStackingEnsembleConfig usage
        content = re.sub(
            r'OOFStackingEnsembleConfig\(',
            'EnhancedOOFConfig(',
            content
        )
        
        # Transform OOFStackingEnsembleManager instantiation
        content = re.sub(
            r'OOFStackingEnsembleManager\(',
            'create_enhanced_oof_generator(',
            content
        )
        
        # Transform method calls
        content = re.sub(
            r'\.fit\(([^)]+)\)',
            r'.generate_oof_predictions(\1)',
            content
        )
        
        return content


def main():
    """Main function for the migration script."""
    parser = argparse.ArgumentParser(description='Migrate OOF/OOS implementations to enhanced consolidated utilities')
    parser.add_argument('--target-file', type=str, help='Target file to migrate')
    parser.add_argument('--target-directory', type=str, help='Target directory to migrate')
    parser.add_argument('--dry-run', action='store_true', help='Perform dry run without making changes')
    parser.add_argument('--output-report', type=str, help='Output report file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize analyzer and transformer
    analyzer = OOFOOSMigrationAnalyzer()
    transformer = OOFOOSMigrationTransformer()
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'dry_run': args.dry_run,
        'files_processed': 0,
        'transformations_applied': 0,
        'errors': []
    }
    
    try:
        if args.target_file:
            # Analyze and transform single file
            analysis = analyzer.analyze_file(args.target_file)
            print(f"\n📊 Analysis Results for {args.target_file}:")
            print(f"   - OOF patterns found: {len(analysis.get('oof_patterns_found', []))}")
            print(f"   - OOS patterns found: {len(analysis.get('oos_patterns_found', []))}")
            print(f"   - Leakage patterns found: {len(analysis.get('leakage_patterns_found', []))}")
            print(f"   - Migration priority: {analysis.get('migration_priority', 'low')}")
            print(f"   - Suggested migrations: {len(analysis.get('suggested_migrations', []))}")
            
            if analysis.get('suggested_migrations'):
                print(f"\n💡 Migration suggestions:")
                for suggestion in analysis['suggested_migrations']:
                    print(f"   - {suggestion['description']}")
            
            # Transform file
            if not args.dry_run:
                transform_result = transformer.transform_file(args.target_file, dry_run=False)
                if transform_result['success']:
                    print(f"\n✅ File transformed successfully")
                    print(f"   - Transformations applied: {transform_result['transformations_applied']}")
                else:
                    print(f"\n❌ File transformation failed: {transform_result.get('error', 'Unknown error')}")
            
            results['files_processed'] = 1
            results['transformations_applied'] = len(analysis.get('suggested_migrations', []))
        
        elif args.target_directory:
            # Analyze and transform directory
            analysis = analyzer.analyze_directory(args.target_directory)
            print(f"\n📊 Directory Analysis Results for {args.target_directory}:")
            print(f"   - Files analyzed: {analysis['files_analyzed']}")
            print(f"   - Files with OOF/OOS: {analysis['files_with_oof_oos']}")
            print(f"   - High priority files: {len(analysis['high_priority_files'])}")
            print(f"   - Medium priority files: {len(analysis['medium_priority_files'])}")
            print(f"   - Low priority files: {len(analysis['low_priority_files'])}")
            print(f"   - Total suggestions: {analysis['total_suggestions']}")
            
            # Show high priority files
            if analysis['high_priority_files']:
                print(f"\n🚨 High Priority Files:")
                for file_analysis in analysis['high_priority_files']:
                    print(f"   - {file_analysis['file_path']} (complexity: {file_analysis['complexity_score']})")
            
            # Transform files if not dry run
            if not args.dry_run:
                all_files = (analysis['high_priority_files'] + 
                           analysis['medium_priority_files'] + 
                           analysis['low_priority_files'])
                
                for file_analysis in all_files:
                    file_path = file_analysis['file_path']
                    transform_result = transformer.transform_file(file_path, dry_run=False)
                    if transform_result['success']:
                        results['transformations_applied'] += transform_result['transformations_applied']
                    else:
                        results['errors'].append(f"Failed to transform {file_path}: {transform_result.get('error', 'Unknown error')}")
                
                print(f"\n✅ Directory transformation completed")
                print(f"   - Files processed: {len(all_files)}")
                print(f"   - Transformations applied: {results['transformations_applied']}")
            
            results['files_processed'] = analysis['files_analyzed']
        
        else:
            print("❌ Please specify either --target-file or --target-directory")
            return 1
        
        # Generate report
        if args.output_report:
            with open(args.output_report, 'w') as f:
                import json
                json.dump(results, f, indent=2)
            print(f"\n📄 Report saved to: {args.output_report}")
        
        print(f"\n🎉 Migration {'simulation' if args.dry_run else 'completed'} successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())