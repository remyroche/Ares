#!/usr/bin/env python3
"""
Comprehensive Feature Audit Script

This script audits ALL features across the entire codebase to ensure
they are natively using VectorBT for maximum performance.

Usage:
    python3 comprehensive_feature_audit.py
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Set
import ast

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveFeatureAuditor:
    """Audits all features to ensure VectorBT native usage."""
    
    def __init__(self, workspace_root: str = "/workspace"):
        self.workspace_root = Path(workspace_root)
        self.audit_results = {
            'total_files': 0,
            'files_with_vectorbt': 0,
            'files_without_vectorbt': 0,
            'feature_generators': 0,
            'transformers': 0,
            'rolling_operations': 0,
            'vectorbt_operations': 0,
            'pandas_operations': 0,
            'missing_vectorbt_files': [],
            'feature_categories': {},
            'performance_issues': []
        }
        
        # Patterns to identify features and operations
        self.feature_patterns = [
            r'class.*Feature.*Generator',
            r'class.*Feature.*Extractor',
            r'class.*Feature.*Calculator',
            r'class.*Feature.*Engine',
            r'def.*generate.*feature',
            r'def.*extract.*feature',
            r'def.*calculate.*feature',
            r'def.*create.*feature'
        ]
        
        self.rolling_patterns = [
            r'\.rolling\(window=(\w+)\)\.mean\(\)',
            r'\.rolling\(window=(\w+)\)\.std\(\)',
            r'\.rolling\(window=(\w+)\)\.var\(\)',
            r'\.rolling\(window=(\w+)\)\.min\(\)',
            r'\.rolling\(window=(\w+)\)\.max\(\)',
            r'\.rolling\(window=(\w+)\)\.sum\(\)',
            r'\.rolling\(window=(\w+)\)\.apply\(',
            r'\.rolling\(window=(\w+)\)\.corr\(',
            r'\.rolling\(window=(\w+)\)\.cov\('
        ]
        
        self.vectorbt_patterns = [
            r'rolling_mean\(',
            r'rolling_std\(',
            r'rolling_var\(',
            r'rolling_min\(',
            r'rolling_max\(',
            r'rolling_sum\(',
            r'rolling_apply\(',
            r'rolling_corr\(',
            r'rolling_cov\(',
            r'_vectorbt_rolling_operation',
            r'VECTORBT_AVAILABLE'
        ]
    
    def audit_file(self, file_path: Path) -> Dict[str, Any]:
        """Audit a single file for VectorBT usage."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            result = {
                'file_path': str(file_path),
                'has_vectorbt_imports': False,
                'has_vectorbt_operations': False,
                'has_pandas_operations': False,
                'feature_classes': [],
                'rolling_operations': 0,
                'vectorbt_operations': 0,
                'pandas_operations': 0,
                'issues': []
            }
            
            # Check for VectorBT imports
            if 'import vectorbt' in content or 'VECTORBT_AVAILABLE' in content:
                result['has_vectorbt_imports'] = True
                self.audit_results['files_with_vectorbt'] += 1
            else:
                result['has_vectorbt_imports'] = False
                self.audit_results['files_without_vectorbt'] += 1
                self.audit_results['missing_vectorbt_files'].append(str(file_path))
            
            # Check for feature classes
            for pattern in self.feature_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                result['feature_classes'].extend(matches)
                if matches:
                    self.audit_results['feature_generators'] += len(matches)
            
            # Check for rolling operations
            rolling_ops = 0
            for pattern in self.rolling_patterns:
                matches = re.findall(pattern, content)
                rolling_ops += len(matches)
            result['rolling_operations'] = rolling_ops
            result['has_pandas_operations'] = rolling_ops > 0
            self.audit_results['rolling_operations'] += rolling_ops
            self.audit_results['pandas_operations'] += rolling_ops
            
            # Check for VectorBT operations
            vectorbt_ops = 0
            for pattern in self.vectorbt_patterns:
                matches = re.findall(pattern, content)
                vectorbt_ops += len(matches)
            result['vectorbt_operations'] = vectorbt_ops
            result['has_vectorbt_operations'] = vectorbt_ops > 0
            self.audit_results['vectorbt_operations'] += vectorbt_ops
            
            # Identify issues
            if not result['has_vectorbt_imports'] and rolling_ops > 0:
                result['issues'].append("Has rolling operations but no VectorBT imports")
            
            if rolling_ops > vectorbt_ops and vectorbt_ops > 0:
                result['issues'].append("More pandas operations than VectorBT operations")
            
            if rolling_ops > 0 and vectorbt_ops == 0:
                result['issues'].append("Has rolling operations but no VectorBT operations")
            
            return result
            
        except Exception as e:
            logger.error(f"Error auditing {file_path}: {e}")
            return {
                'file_path': str(file_path),
                'has_vectorbt_imports': False,
                'has_vectorbt_operations': False,
                'has_pandas_operations': False,
                'feature_classes': [],
                'rolling_operations': 0,
                'vectorbt_operations': 0,
                'pandas_operations': 0,
                'issues': [f"Audit error: {e}"]
            }
    
    def audit_all_features(self) -> List[Dict[str, Any]]:
        """Audit all features across the entire codebase."""
        logger.info("🔍 Auditing all features for VectorBT usage...")
        
        results = []
        
        # Define search directories
        search_dirs = [
            "src/feature_generation",
            "src/feature_engineering_roadmap", 
            "src/features_common",
            "src/analyst",
            "src/trading",
            "src/training",
            "src/feature_selection",
            "src/monitoring",
            "src/supervisor",
            "src/tactician",
            "src/strategist",
            "src/utils"
        ]
        
        for search_dir in search_dirs:
            dir_path = self.workspace_root / search_dir
            if dir_path.exists():
                logger.info(f"Auditing {search_dir}...")
                for py_file in dir_path.rglob("*.py"):
                    if py_file.name != "__init__.py" and "test" not in str(py_file).lower():
                        result = self.audit_file(py_file)
                        results.append(result)
                        self.audit_results['total_files'] += 1
                        
                        # Categorize by directory
                        category = search_dir.split('/')[-1]
                        if category not in self.audit_results['feature_categories']:
                            self.audit_results['feature_categories'][category] = {
                                'total': 0,
                                'with_vectorbt': 0,
                                'without_vectorbt': 0,
                                'issues': 0
                            }
                        
                        self.audit_results['feature_categories'][category]['total'] += 1
                        if result['has_vectorbt_imports']:
                            self.audit_results['feature_categories'][category]['with_vectorbt'] += 1
                        else:
                            self.audit_results['feature_categories'][category]['without_vectorbt'] += 1
                        if result['issues']:
                            self.audit_results['feature_categories'][category]['issues'] += 1
        
        return results
    
    def generate_comprehensive_report(self, results: List[Dict[str, Any]]) -> str:
        """Generate comprehensive audit report."""
        report = []
        report.append("=" * 100)
        report.append("COMPREHENSIVE FEATURE VECTORBT AUDIT REPORT")
        report.append("=" * 100)
        
        # Overall statistics
        report.append(f"\n📊 OVERALL STATISTICS:")
        report.append(f"  Total files audited: {self.audit_results['total_files']}")
        report.append(f"  Files with VectorBT: {self.audit_results['files_with_vectorbt']}")
        report.append(f"  Files without VectorBT: {self.audit_results['files_without_vectorbt']}")
        report.append(f"  Feature generators found: {self.audit_results['feature_generators']}")
        report.append(f"  Total rolling operations: {self.audit_results['rolling_operations']}")
        report.append(f"  VectorBT operations: {self.audit_results['vectorbt_operations']}")
        report.append(f"  Pandas operations: {self.audit_results['pandas_operations']}")
        
        # Calculate percentages
        if self.audit_results['total_files'] > 0:
            vectorbt_coverage = (self.audit_results['files_with_vectorbt'] / self.audit_results['total_files']) * 100
            vectorbt_ops_ratio = (self.audit_results['vectorbt_operations'] / max(self.audit_results['rolling_operations'], 1)) * 100
            
            report.append(f"\n📈 COVERAGE METRICS:")
            report.append(f"  VectorBT file coverage: {vectorbt_coverage:.1f}%")
            report.append(f"  VectorBT operation ratio: {vectorbt_ops_ratio:.1f}%")
        
        # Category breakdown
        report.append(f"\n📁 CATEGORY BREAKDOWN:")
        for category, stats in self.audit_results['feature_categories'].items():
            coverage = (stats['with_vectorbt'] / stats['total']) * 100 if stats['total'] > 0 else 0
            report.append(f"  {category}:")
            report.append(f"    Total files: {stats['total']}")
            report.append(f"    With VectorBT: {stats['with_vectorbt']} ({coverage:.1f}%)")
            report.append(f"    Without VectorBT: {stats['without_vectorbt']}")
            report.append(f"    Issues: {stats['issues']}")
        
        # Files with issues
        files_with_issues = [r for r in results if r['issues']]
        if files_with_issues:
            report.append(f"\n⚠️  FILES WITH ISSUES ({len(files_with_issues)}):")
            for result in files_with_issues[:20]:  # Show first 20
                report.append(f"\n  📁 {result['file_path']}")
                for issue in result['issues']:
                    report.append(f"    - {issue}")
                report.append(f"    Rolling ops: {result['rolling_operations']}")
                report.append(f"    VectorBT ops: {result['vectorbt_operations']}")
            
            if len(files_with_issues) > 20:
                report.append(f"    ... and {len(files_with_issues) - 20} more files with issues")
        
        # Files without VectorBT
        files_without_vectorbt = [r for r in results if not r['has_vectorbt_imports'] and r['rolling_operations'] > 0]
        if files_without_vectorbt:
            report.append(f"\n❌ FILES WITHOUT VECTORBT ({len(files_without_vectorbt)}):")
            for result in files_without_vectorbt[:15]:  # Show first 15
                report.append(f"  - {result['file_path']} ({result['rolling_operations']} rolling ops)")
            
            if len(files_without_vectorbt) > 15:
                report.append(f"  ... and {len(files_without_vectorbt) - 15} more files without VectorBT")
        
        # Performance analysis
        report.append(f"\n🚀 PERFORMANCE ANALYSIS:")
        
        # Calculate VectorBT adoption rate
        if self.audit_results['rolling_operations'] > 0:
            adoption_rate = (self.audit_results['vectorbt_operations'] / self.audit_results['rolling_operations']) * 100
            report.append(f"  VectorBT adoption rate: {adoption_rate:.1f}%")
        
        # Calculate feature coverage
        if self.audit_results['feature_generators'] > 0:
            feature_coverage = (self.audit_results['files_with_vectorbt'] / self.audit_results['total_files']) * 100
            report.append(f"  Feature coverage: {feature_coverage:.1f}%")
        
        # Overall assessment
        report.append(f"\n🎯 OVERALL ASSESSMENT:")
        
        if vectorbt_coverage >= 95 and vectorbt_ops_ratio >= 80:
            report.append(f"  🟢 EXCELLENT: Comprehensive VectorBT integration achieved")
        elif vectorbt_coverage >= 85 and vectorbt_ops_ratio >= 70:
            report.append(f"  🟡 GOOD: Strong VectorBT integration with room for improvement")
        elif vectorbt_coverage >= 70 and vectorbt_ops_ratio >= 50:
            report.append(f"  🟠 FAIR: Partial VectorBT integration, needs more work")
        else:
            report.append(f"  🔴 POOR: VectorBT integration needs significant improvement")
        
        # Recommendations
        report.append(f"\n💡 RECOMMENDATIONS:")
        
        if files_without_vectorbt:
            report.append(f"  1. Add VectorBT imports to {len(files_without_vectorbt)} files without them")
        
        if files_with_issues:
            report.append(f"  2. Fix issues in {len(files_with_issues)} files")
        
        if self.audit_results['pandas_operations'] > self.audit_results['vectorbt_operations']:
            report.append(f"  3. Replace {self.audit_results['pandas_operations'] - self.audit_results['vectorbt_operations']} pandas operations with VectorBT")
        
        report.append(f"  4. Ensure all feature generators use VectorBT for datasets > 1000 samples")
        report.append(f"  5. Add proper error handling and fallbacks for VectorBT operations")
        
        return "\n".join(report)
    
    def run_comprehensive_audit(self) -> None:
        """Run comprehensive feature audit."""
        logger.info("🚀 Starting comprehensive feature audit...")
        
        results = self.audit_all_features()
        report = self.generate_comprehensive_report(results)
        
        print(report)
        
        # Save report to file
        report_file = self.workspace_root / "comprehensive_feature_audit_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"\n📄 Detailed report saved to: {report_file}")


def main():
    """Main execution function."""
    auditor = ComprehensiveFeatureAuditor()
    auditor.run_comprehensive_audit()


if __name__ == "__main__":
    main()