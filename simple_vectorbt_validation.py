#!/usr/bin/env python3
"""
Simple VectorBT Integration Validation Script

This script validates that all feature generators and transformers
are properly using VectorBT natively for maximum performance.

Usage:
    python3 simple_vectorbt_validation.py
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleVectorBTValidator:
    """Validates VectorBT integration across all generators and transformers."""
    
    def __init__(self, workspace_root: str = "/workspace"):
        self.workspace_root = Path(workspace_root)
        self.validation_results = {
            'total_files': 0,
            'vectorbt_imports': 0,
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'missing_vectorbt': 0,
            'errors': []
        }
    
    def validate_file(self, file_path: Path) -> Dict[str, Any]:
        """Validate a single file for VectorBT integration."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            result = {
                'file_path': str(file_path),
                'has_vectorbt_imports': False,
                'has_vectorbt_operations': False,
                'has_pandas_fallbacks': False,
                'vectorbt_operation_count': 0,
                'pandas_operation_count': 0,
                'issues': []
            }
            
            # Check for VectorBT imports
            vectorbt_import_patterns = [
                r'import vectorbt',
                r'from vectorbt',
                r'VECTORBT_AVAILABLE'
            ]
            
            vectorbt_import_found = any(re.search(pattern, content) for pattern in vectorbt_import_patterns)
            result['has_vectorbt_imports'] = vectorbt_import_found
            
            if vectorbt_import_found:
                self.validation_results['vectorbt_imports'] += 1
            
            # Check for VectorBT operations
            vectorbt_operation_patterns = [
                r'_vectorbt_rolling_operation',
                r'rolling_mean\(',
                r'rolling_std\(',
                r'rolling_var\(',
                r'rolling_min\(',
                r'rolling_max\(',
                r'rolling_sum\(',
                r'rolling_apply\(',
                r'rolling_corr\(',
                r'rolling_cov\('
            ]
            
            vectorbt_ops = sum(len(re.findall(pattern, content)) for pattern in vectorbt_operation_patterns)
            result['vectorbt_operation_count'] = vectorbt_ops
            result['has_vectorbt_operations'] = vectorbt_ops > 0
            
            if vectorbt_ops > 0:
                self.validation_results['vectorbt_operations'] += 1
            
            # Check for pandas operations (should be minimal)
            pandas_patterns = [
                r'\.rolling\(window=(\w+)\)\.mean\(\)',
                r'\.rolling\(window=(\w+)\)\.std\(\)',
                r'\.rolling\(window=(\w+)\)\.var\(\)',
                r'\.rolling\(window=(\w+)\)\.min\(\)',
                r'\.rolling\(window=(\w+)\)\.max\(\)',
                r'\.rolling\(window=(\w+)\)\.sum\(\)'
            ]
            
            pandas_ops = sum(len(re.findall(pattern, content)) for pattern in pandas_patterns)
            result['pandas_operation_count'] = pandas_ops
            result['has_pandas_fallbacks'] = pandas_ops > 0
            
            if pandas_ops > 0:
                self.validation_results['pandas_fallbacks'] += 1
            
            # Identify issues
            if not vectorbt_import_found:
                result['issues'].append("Missing VectorBT imports")
            
            if vectorbt_ops == 0 and pandas_ops > 0:
                result['issues'].append("No VectorBT operations found, only pandas operations")
            
            if pandas_ops > vectorbt_ops and vectorbt_ops > 0:
                result['issues'].append("More pandas operations than VectorBT operations")
            
            return result
            
        except Exception as e:
            error_msg = f"Error validating {file_path}: {e}"
            logger.error(error_msg)
            self.validation_results['errors'].append(error_msg)
            return {
                'file_path': str(file_path),
                'has_vectorbt_imports': False,
                'has_vectorbt_operations': False,
                'has_pandas_fallbacks': False,
                'vectorbt_operation_count': 0,
                'pandas_operation_count': 0,
                'issues': [f"Validation error: {e}"]
            }
    
    def validate_all_files(self) -> List[Dict[str, Any]]:
        """Validate all feature generators and transformers."""
        logger.info("🔍 Validating VectorBT integration...")
        
        results = []
        
        # Validate feature generation categories
        categories_dir = self.workspace_root / "src" / "feature_generation" / "categories"
        if categories_dir.exists():
            for py_file in categories_dir.glob("*.py"):
                if py_file.name != "__init__.py":
                    result = self.validate_file(py_file)
                    results.append(result)
                    self.validation_results['total_files'] += 1
        
        # Validate feature engineering roadmap
        roadmap_dir = self.workspace_root / "src" / "feature_engineering_roadmap"
        if roadmap_dir.exists():
            for py_file in roadmap_dir.glob("*.py"):
                if py_file.name != "__init__.py":
                    result = self.validate_file(py_file)
                    results.append(result)
                    self.validation_results['total_files'] += 1
        
        # Validate core feature generation
        core_dir = self.workspace_root / "src" / "feature_generation" / "core"
        if core_dir.exists():
            for py_file in core_dir.glob("*.py"):
                if py_file.name != "__init__.py":
                    result = self.validate_file(py_file)
                    results.append(result)
                    self.validation_results['total_files'] += 1
        
        # Validate transforms
        transforms_dir = self.workspace_root / "src" / "features_common" / "transforms"
        if transforms_dir.exists():
            for py_file in transforms_dir.glob("*.py"):
                if py_file.name != "__init__.py":
                    result = self.validate_file(py_file)
                    results.append(result)
                    self.validation_results['total_files'] += 1
        
        return results
    
    def generate_validation_report(self, results: List[Dict[str, Any]]) -> str:
        """Generate comprehensive validation report."""
        report = []
        report.append("=" * 80)
        report.append("VECTORBT INTEGRATION VALIDATION REPORT")
        report.append("=" * 80)
        
        # Summary statistics
        report.append(f"\n📊 SUMMARY STATISTICS:")
        report.append(f"  Total files validated: {self.validation_results['total_files']}")
        report.append(f"  Files with VectorBT imports: {self.validation_results['vectorbt_imports']}")
        report.append(f"  Files with VectorBT operations: {self.validation_results['vectorbt_operations']}")
        report.append(f"  Files with pandas fallbacks: {self.validation_results['pandas_fallbacks']}")
        report.append(f"  Validation errors: {len(self.validation_results['errors'])}")
        
        # Calculate percentages
        if self.validation_results['total_files'] > 0:
            vectorbt_import_pct = (self.validation_results['vectorbt_imports'] / self.validation_results['total_files']) * 100
            vectorbt_ops_pct = (self.validation_results['vectorbt_operations'] / self.validation_results['total_files']) * 100
            
            report.append(f"\n📈 INTEGRATION RATES:")
            report.append(f"  VectorBT imports: {vectorbt_import_pct:.1f}%")
            report.append(f"  VectorBT operations: {vectorbt_ops_pct:.1f}%")
        
        # Files with issues
        files_with_issues = [r for r in results if r['issues']]
        if files_with_issues:
            report.append(f"\n⚠️  FILES WITH ISSUES ({len(files_with_issues)}):")
            for result in files_with_issues[:10]:  # Show first 10
                report.append(f"\n  📁 {result['file_path']}")
                for issue in result['issues']:
                    report.append(f"    - {issue}")
                report.append(f"    VectorBT operations: {result['vectorbt_operation_count']}")
                report.append(f"    Pandas operations: {result['pandas_operation_count']}")
            
            if len(files_with_issues) > 10:
                report.append(f"    ... and {len(files_with_issues) - 10} more files with issues")
        
        # Files without VectorBT
        files_without_vectorbt = [r for r in results if not r['has_vectorbt_imports']]
        if files_without_vectorbt:
            report.append(f"\n❌ FILES WITHOUT VECTORBT ({len(files_without_vectorbt)}):")
            for result in files_without_vectorbt[:10]:  # Show first 10
                report.append(f"  - {result['file_path']}")
            
            if len(files_without_vectorbt) > 10:
                report.append(f"  ... and {len(files_without_vectorbt) - 10} more files without VectorBT")
        
        # Best practices compliance
        report.append(f"\n✅ BEST PRACTICES COMPLIANCE:")
        
        # Check for proper VectorBT usage patterns
        proper_usage = 0
        for result in results:
            if (result['has_vectorbt_imports'] and 
                result['has_vectorbt_operations'] and 
                result['vectorbt_operation_count'] >= result['pandas_operation_count']):
                proper_usage += 1
        
        if self.validation_results['total_files'] > 0:
            compliance_pct = (proper_usage / self.validation_results['total_files']) * 100
            report.append(f"  Proper VectorBT usage: {compliance_pct:.1f}%")
        
        # Overall assessment
        report.append(f"\n🎯 OVERALL ASSESSMENT:")
        
        if vectorbt_import_pct >= 90 and vectorbt_ops_pct >= 80:
            report.append(f"  🟢 EXCELLENT: VectorBT integration is comprehensive and well-implemented")
        elif vectorbt_import_pct >= 70 and vectorbt_ops_pct >= 60:
            report.append(f"  🟡 GOOD: VectorBT integration is mostly complete with room for improvement")
        elif vectorbt_import_pct >= 50 and vectorbt_ops_pct >= 40:
            report.append(f"  🟠 FAIR: VectorBT integration is partially complete, needs more work")
        else:
            report.append(f"  🔴 POOR: VectorBT integration needs significant improvement")
        
        return "\n".join(report)
    
    def run_validation(self) -> None:
        """Run complete validation and generate report."""
        logger.info("🚀 Starting VectorBT integration validation...")
        
        results = self.validate_all_files()
        report = self.generate_validation_report(results)
        
        print(report)
        
        # Save report to file
        report_file = self.workspace_root / "vectorbt_integration_validation_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"\n📄 Detailed report saved to: {report_file}")


def main():
    """Main execution function."""
    validator = SimpleVectorBTValidator()
    validator.run_validation()


if __name__ == "__main__":
    main()