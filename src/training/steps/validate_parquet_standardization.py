#!/usr/bin/env python3
"""
Validate Parquet Standardization

This script validates that all training steps are properly using the standardized
Parquet handler and that file paths, column names, and data formats are consistent.
"""

import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd


class ParquetStandardizationValidator:
    """Validates Parquet standardization across all training steps."""
    
    def __init__(self):
        self.validation_results = {
            'import_consistency': [],
            'path_consistency': [],
            'filename_consistency': [],
            'schema_consistency': [],
            'errors': []
        }
        
    def test_import_consistency(self) -> bool:
        """Test that all training steps have the standardized_parquet_handler import."""
        print("📦 Testing import consistency...")
        
        try:
            # Find all Python files in the training steps directory
            steps_dir = Path(__file__).parent
            python_files = list(steps_dir.rglob('*.py'))
            
            files_with_import = 0
            files_without_import = 0
            files_with_parquet_ops = 0
            
            for py_file in python_files:
                # Skip the validation script itself and the handler
                if py_file.name in ['validate_parquet_standardization.py', 'standardized_parquet_handler.py', 'update_all_steps_for_standardization.py']:
                    continue
                
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check if file contains parquet operations
                    has_parquet_ops = any(op in content for op in [
                        'pd.read_parquet', 'df.to_parquet', 'read_parquet', 'to_parquet',
                        'pipeline_standards.build_path', 'pipeline_standards.generate_file_name'
                    ])
                    
                    if has_parquet_ops:
                        files_with_parquet_ops += 1
                        # Check if it has the standardized import
                        has_standardized_import = 'standardized_parquet_handler' in content
                        
                        if has_standardized_import:
                            files_with_import += 1
                            self.validation_results['import_consistency'].append({
                                'test': f'{py_file.name}',
                                'status': 'PASSED',
                                'message': f'Has standardized_parquet_handler import'
                            })
                        else:
                            files_without_import += 1
                            self.validation_results['import_consistency'].append({
                                'test': f'{py_file.name}',
                                'status': 'FAILED',
                                'message': f'Missing standardized_parquet_handler import but has parquet operations'
                            })
                    else:
                        # File doesn't have parquet operations, so it's fine
                        self.validation_results['import_consistency'].append({
                            'test': f'{py_file.name}',
                            'status': 'PASSED',
                            'message': f'No parquet operations found, import not needed'
                        })
                
                except Exception as e:
                    self.validation_results['errors'].append(f'Error reading {py_file}: {e}')
            
            print(f"📊 Import analysis: {files_with_parquet_ops} files with parquet operations")
            print(f"   - {files_with_import} files with standardized imports")
            print(f"   - {files_without_import} files missing standardized imports")
            
            return files_without_import == 0
            
        except Exception as e:
            self.validation_results['errors'].append(f'Import consistency test failed: {e}')
            return False
    
    def test_path_consistency(self) -> bool:
        """Test that path generation is consistent."""
        print("🔍 Testing path consistency...")
        
        try:
            # Check pipeline_standards.py for path definitions
            project_root = Path(__file__).parent.parent.parent
            pipeline_standards_file = project_root / "utils" / "pipeline_standards.py"
            
            if not pipeline_standards_file.exists():
                self.validation_results['errors'].append('pipeline_standards.py not found')
                return False
            
            with open(pipeline_standards_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for directory structure definition
            if 'DIRECTORY_STRUCTURE' not in content:
                self.validation_results['path_consistency'].append({
                    'test': 'directory_structure',
                    'status': 'FAILED',
                    'message': 'DIRECTORY_STRUCTURE not found in pipeline_standards.py'
                })
                return False
            
            # Check for required path types
            required_paths = ['raw_data', 'unified_data', 'processed_data', 'reports']
            for path_type in required_paths:
                if f"'{path_type}'" not in content:
                    self.validation_results['path_consistency'].append({
                        'test': f'{path_type}_path',
                        'status': 'FAILED',
                        'message': f'{path_type} path not defined in DIRECTORY_STRUCTURE'
                    })
                else:
                    self.validation_results['path_consistency'].append({
                        'test': f'{path_type}_path',
                        'status': 'PASSED',
                        'message': f'{path_type} path defined in DIRECTORY_STRUCTURE'
                    })
            
            return True
            
        except Exception as e:
            self.validation_results['errors'].append(f'Path consistency test failed: {e}')
            return False
    
    def test_filename_consistency(self) -> bool:
        """Test that filename generation is consistent."""
        print("📝 Testing filename consistency...")
        
        try:
            # Check pipeline_standards.py for filename definitions
            project_root = Path(__file__).parent.parent.parent
            pipeline_standards_file = project_root / "utils" / "pipeline_standards.py"
            
            with open(pipeline_standards_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for file naming definition
            if 'FILE_NAMING' not in content:
                self.validation_results['filename_consistency'].append({
                    'test': 'file_naming',
                    'status': 'FAILED',
                    'message': 'FILE_NAMING not found in pipeline_standards.py'
                })
                return False
            
            # Check for required file types
            required_files = ['klines', 'aggtrades', 'unified', 'validated_data']
            for file_type in required_files:
                if f"'{file_type}'" not in content:
                    self.validation_results['filename_consistency'].append({
                        'test': f'{file_type}_filename',
                        'status': 'FAILED',
                        'message': f'{file_type} filename not defined in FILE_NAMING'
                    })
                else:
                    self.validation_results['filename_consistency'].append({
                        'test': f'{file_type}_filename',
                        'status': 'PASSED',
                        'message': f'{file_type} filename defined in FILE_NAMING'
                    })
            
            return True
            
        except Exception as e:
            self.validation_results['errors'].append(f'Filename consistency test failed: {e}')
            return False
    
    def test_schema_consistency(self) -> bool:
        """Test that schemas are consistent."""
        print("📋 Testing schema consistency...")
        
        try:
            # Check pipeline_standards.py for schema definitions
            project_root = Path(__file__).parent.parent.parent
            pipeline_standards_file = project_root / "utils" / "pipeline_standards.py"
            
            with open(pipeline_standards_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for schemas definition
            if 'SCHEMAS' not in content:
                self.validation_results['schema_consistency'].append({
                    'test': 'schemas_definition',
                    'status': 'FAILED',
                    'message': 'SCHEMAS not found in pipeline_standards.py'
                })
                return False
            
            # Check for required schemas
            required_schemas = ['klines', 'aggtrades', 'unified', 'validated_data']
            for schema_name in required_schemas:
                if f"'{schema_name}'" not in content:
                    self.validation_results['schema_consistency'].append({
                        'test': f'{schema_name}_schema',
                        'status': 'FAILED',
                        'message': f'{schema_name} schema not defined in SCHEMAS'
                    })
                else:
                    self.validation_results['schema_consistency'].append({
                        'test': f'{schema_name}_schema',
                        'status': 'PASSED',
                        'message': f'{schema_name} schema defined in SCHEMAS'
                    })
            
            return True
            
        except Exception as e:
            self.validation_results['errors'].append(f'Schema consistency test failed: {e}')
            return False
    
    def test_standardized_handler_exists(self) -> bool:
        """Test that the standardized handler exists and is properly structured."""
        print("🔧 Testing standardized handler...")
        
        try:
            handler_file = Path(__file__).parent / "standardized_parquet_handler.py"
            
            if not handler_file.exists():
                self.validation_results['errors'].append('standardized_parquet_handler.py not found')
                return False
            
            with open(handler_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for required methods
            required_methods = [
                'get_standardized_path',
                'get_standardized_filename', 
                'read_parquet_standardized',
                'write_parquet_standardized',
                'list_parquet_files',
                'standardize_columns',
                'standardize_dtypes',
                'validate_data_quality'
            ]
            
            for method in required_methods:
                if f'def {method}' not in content:
                    self.validation_results['import_consistency'].append({
                        'test': f'{method}_method',
                        'status': 'FAILED',
                        'message': f'Method {method} not found in standardized_parquet_handler.py'
                    })
                else:
                    self.validation_results['import_consistency'].append({
                        'test': f'{method}_method',
                        'status': 'PASSED',
                        'message': f'Method {method} found in standardized_parquet_handler.py'
                    })
            
            return True
            
        except Exception as e:
            self.validation_results['errors'].append(f'Standardized handler test failed: {e}')
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all validation tests."""
        print("🚀 Starting Parquet Standardization Validation")
        print("=" * 60)
        
        try:
            # Run all tests
            tests = [
                self.test_import_consistency,
                self.test_path_consistency,
                self.test_filename_consistency,
                self.test_schema_consistency,
                self.test_standardized_handler_exists
            ]
            
            for test in tests:
                try:
                    test()
                except Exception as e:
                    self.validation_results['errors'].append(f'Test {test.__name__} failed: {e}')
            
            # Calculate overall results
            total_tests = 0
            passed_tests = 0
            
            for category, results in self.validation_results.items():
                if category == 'errors':
                    continue
                for result in results:
                    total_tests += 1
                    if result['status'] == 'PASSED':
                        passed_tests += 1
            
            success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
            
            return {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': total_tests - passed_tests,
                'success_rate': success_rate,
                'errors': self.validation_results['errors'],
                'detailed_results': self.validation_results,
                'overall_status': 'PASSED' if success_rate >= 90 and len(self.validation_results['errors']) == 0 else 'FAILED'
            }
            
        except Exception as e:
            self.validation_results['errors'].append(f'Test execution failed: {e}')
            return {
                'total_tests': 0,
                'passed_tests': 0,
                'failed_tests': 0,
                'success_rate': 0,
                'errors': self.validation_results['errors'],
                'detailed_results': self.validation_results,
                'overall_status': 'FAILED'
            }
    
    def print_results(self, results: Dict[str, Any]):
        """Print validation results."""
        print("\n📊 Validation Results:")
        print("=" * 60)
        print(f"Total Tests: {results['total_tests']}")
        print(f"Passed: {results['passed_tests']}")
        print(f"Failed: {results['failed_tests']}")
        print(f"Success Rate: {results['success_rate']:.1f}%")
        print(f"Overall Status: {results['overall_status']}")
        
        if results['errors']:
            print(f"\n❌ Errors ({len(results['errors'])}):")
            for error in results['errors']:
                print(f"   - {error}")
        
        # Print detailed results by category
        for category, category_results in results['detailed_results'].items():
            if category == 'errors' or not category_results:
                continue
            
            print(f"\n📋 {category.replace('_', ' ').title()}:")
            for result in category_results:
                status_emoji = "✅" if result['status'] == 'PASSED' else "❌"
                print(f"   {status_emoji} {result['test']}: {result['message']}")


def main():
    """Main validation function."""
    validator = ParquetStandardizationValidator()
    results = validator.run_all_tests()
    validator.print_results(results)
    
    return results['overall_status'] == 'PASSED'


if __name__ == "__main__":
    success = main()
    print(f"\n🎉 Validation {'PASSED' if success else 'FAILED'}!")
    exit(0 if success else 1)