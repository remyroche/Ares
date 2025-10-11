#!/usr/bin/env python3
"""
Feature Usage VectorBT Audit Script

This script audits whether the actual features (not just files) are using VectorBT
operations in their implementations.

Usage:
    python3 audit_feature_usage_vectorbt.py
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

class FeatureUsageAuditor:
    """Audits actual feature usage of VectorBT operations."""
    
    def __init__(self, workspace_root: str = "/workspace"):
        self.workspace_root = Path(workspace_root)
        self.audit_results = {
            'total_features': 0,
            'features_using_vectorbt': 0,
            'features_using_pandas': 0,
            'features_mixed': 0,
            'vectorbt_operations': 0,
            'pandas_operations': 0,
            'feature_details': []
        }
        
        # Patterns to identify feature methods
        self.feature_method_patterns = [
            r'def.*generate.*feature',
            r'def.*extract.*feature', 
            r'def.*calculate.*feature',
            r'def.*create.*feature',
            r'def.*_generate_feature',
            r'def.*_extract_feature',
            r'def.*_calculate_feature',
            r'def.*_create_feature'
        ]
        
        # VectorBT operation patterns
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
            r'vbt\.',
            r'vectorbt\.'
        ]
        
        # Pandas operation patterns
        self.pandas_patterns = [
            r'\.rolling\(window=(\w+)\)\.mean\(\)',
            r'\.rolling\(window=(\w+)\)\.std\(\)',
            r'\.rolling\(window=(\w+)\)\.var\(\)',
            r'\.rolling\(window=(\w+)\)\.min\(\)',
            r'\.rolling\(window=(\w+)\)\.max\(\)',
            r'\.rolling\(window=(\w+)\)\.sum\(\)',
            r'\.rolling\(window=(\w+)\)\.apply\(',
            r'\.rolling\(window=(\w+)\)\.corr\(',
            r'\.rolling\(window=(\w+)\)\.cov\(',
            r'\.ewm\(',
            r'\.expanding\('
        ]
    
    def audit_file(self, file_path: Path) -> Dict[str, Any]:
        """Audit a single file for actual feature VectorBT usage."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            result = {
                'file_path': str(file_path),
                'features': [],
                'total_features': 0,
                'vectorbt_features': 0,
                'pandas_features': 0,
                'mixed_features': 0
            }
            
            # Find feature methods
            feature_methods = []
            for pattern in self.feature_method_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    # Extract method name and find its body
                    method_name = match.group(0)
                    start_pos = match.start()
                    
                    # Find the method body (simplified - look for next def or end of file)
                    next_def = content.find('\n    def ', start_pos)
                    if next_def == -1:
                        next_def = content.find('\nclass ', start_pos)
                    if next_def == -1:
                        next_def = len(content)
                    
                    method_body = content[start_pos:next_def]
                    feature_methods.append({
                        'name': method_name,
                        'body': method_body
                    })
            
            result['total_features'] = len(feature_methods)
            self.audit_results['total_features'] += len(feature_methods)
            
            # Analyze each feature method
            for method in feature_methods:
                feature_analysis = self._analyze_feature_method(method)
                result['features'].append(feature_analysis)
                
                if feature_analysis['uses_vectorbt'] and feature_analysis['uses_pandas']:
                    result['mixed_features'] += 1
                    self.audit_results['features_mixed'] += 1
                elif feature_analysis['uses_vectorbt']:
                    result['vectorbt_features'] += 1
                    self.audit_results['features_using_vectorbt'] += 1
                elif feature_analysis['uses_pandas']:
                    result['pandas_features'] += 1
                    self.audit_results['features_using_pandas'] += 1
                
                self.audit_results['vectorbt_operations'] += feature_analysis['vectorbt_ops']
                self.audit_results['pandas_operations'] += feature_analysis['pandas_ops']
            
            return result
            
        except Exception as e:
            logger.error(f"Error auditing {file_path}: {e}")
            return {
                'file_path': str(file_path),
                'features': [],
                'total_features': 0,
                'vectorbt_features': 0,
                'pandas_features': 0,
                'mixed_features': 0
            }
    
    def _analyze_feature_method(self, method: Dict[str, str]) -> Dict[str, Any]:
        """Analyze a single feature method for VectorBT usage."""
        method_body = method['body']
        
        # Count VectorBT operations
        vectorbt_ops = 0
        for pattern in self.vectorbt_patterns:
            vectorbt_ops += len(re.findall(pattern, method_body))
        
        # Count pandas operations
        pandas_ops = 0
        for pattern in self.pandas_patterns:
            pandas_ops += len(re.findall(pattern, method_body))
        
        return {
            'name': method['name'],
            'uses_vectorbt': vectorbt_ops > 0,
            'uses_pandas': pandas_ops > 0,
            'vectorbt_ops': vectorbt_ops,
            'pandas_ops': pandas_ops,
            'is_optimized': vectorbt_ops > pandas_ops or (vectorbt_ops > 0 and pandas_ops == 0)
        }
    
    def audit_core_features(self) -> List[Dict[str, Any]]:
        """Audit core feature generation files."""
        logger.info("🔍 Auditing core feature generation for actual VectorBT usage...")
        
        results = []
        
        # Focus on core feature generation files
        core_dirs = [
            "src/feature_generation/categories",
            "src/feature_engineering_roadmap"
        ]
        
        for core_dir in core_dirs:
            dir_path = self.workspace_root / core_dir
            if dir_path.exists():
                logger.info(f"Auditing {core_dir}...")
                for py_file in dir_path.glob("*.py"):
                    if py_file.name != "__init__.py":
                        result = self.audit_file(py_file)
                        results.append(result)
        
        return results
    
    def generate_feature_usage_report(self, results: List[Dict[str, Any]]) -> str:
        """Generate detailed feature usage report."""
        report = []
        report.append("=" * 100)
        report.append("FEATURE USAGE VECTORBT AUDIT REPORT")
        report.append("=" * 100)
        
        # Overall statistics
        report.append(f"\n📊 FEATURE USAGE STATISTICS:")
        report.append(f"  Total features found: {self.audit_results['total_features']}")
        report.append(f"  Features using VectorBT: {self.audit_results['features_using_vectorbt']}")
        report.append(f"  Features using pandas only: {self.audit_results['features_using_pandas']}")
        report.append(f"  Features using both: {self.audit_results['features_mixed']}")
        report.append(f"  Total VectorBT operations: {self.audit_results['vectorbt_operations']}")
        report.append(f"  Total pandas operations: {self.audit_results['pandas_operations']}")
        
        # Calculate percentages
        if self.audit_results['total_features'] > 0:
            vectorbt_pct = (self.audit_results['features_using_vectorbt'] / self.audit_results['total_features']) * 100
            pandas_pct = (self.audit_results['features_using_pandas'] / self.audit_results['total_features']) * 100
            mixed_pct = (self.audit_results['features_mixed'] / self.audit_results['total_features']) * 100
            
            report.append(f"\n📈 FEATURE USAGE PERCENTAGES:")
            report.append(f"  VectorBT features: {vectorbt_pct:.1f}%")
            report.append(f"  Pandas-only features: {pandas_pct:.1f}%")
            report.append(f"  Mixed features: {mixed_pct:.1f}%")
        
        # Operation ratio
        if self.audit_results['pandas_operations'] > 0:
            op_ratio = (self.audit_results['vectorbt_operations'] / self.audit_results['pandas_operations']) * 100
            report.append(f"\n⚡ OPERATION RATIO:")
            report.append(f"  VectorBT to pandas ratio: {op_ratio:.1f}%")
        
        # File-by-file breakdown
        report.append(f"\n📁 FILE-BY-FILE BREAKDOWN:")
        for result in results:
            if result['total_features'] > 0:
                report.append(f"\n  📄 {result['file_path']}")
                report.append(f"    Total features: {result['total_features']}")
                report.append(f"    VectorBT features: {result['vectorbt_features']}")
                report.append(f"    Pandas features: {result['pandas_features']}")
                report.append(f"    Mixed features: {result['mixed_features']}")
                
                # Show individual features
                for feature in result['features']:
                    status = "🟢 VectorBT" if feature['is_optimized'] else "🔴 Pandas" if feature['uses_pandas'] else "⚪ None"
                    report.append(f"      - {feature['name']}: {status} (VB:{feature['vectorbt_ops']}, PD:{feature['pandas_ops']})")
        
        # Features that need attention
        features_needing_attention = []
        for result in results:
            for feature in result['features']:
                if feature['uses_pandas'] and not feature['uses_vectorbt']:
                    features_needing_attention.append({
                        'file': result['file_path'],
                        'feature': feature['name'],
                        'pandas_ops': feature['pandas_ops']
                    })
        
        if features_needing_attention:
            report.append(f"\n⚠️  FEATURES NEEDING VECTORBT CONVERSION ({len(features_needing_attention)}):")
            for item in features_needing_attention[:20]:  # Show first 20
                report.append(f"  - {item['file']}::{item['feature']} ({item['pandas_ops']} pandas ops)")
            
            if len(features_needing_attention) > 20:
                report.append(f"  ... and {len(features_needing_attention) - 20} more features")
        
        # Overall assessment
        report.append(f"\n🎯 OVERALL ASSESSMENT:")
        
        if self.audit_results['total_features'] > 0:
            vectorbt_feature_pct = (self.audit_results['features_using_vectorbt'] / self.audit_results['total_features']) * 100
            
            if vectorbt_feature_pct >= 80:
                report.append(f"  🟢 EXCELLENT: Most features are using VectorBT")
            elif vectorbt_feature_pct >= 60:
                report.append(f"  🟡 GOOD: Many features are using VectorBT")
            elif vectorbt_feature_pct >= 40:
                report.append(f"  🟠 FAIR: Some features are using VectorBT")
            else:
                report.append(f"  🔴 POOR: Few features are using VectorBT")
        
        return "\n".join(report)
    
    def run_feature_usage_audit(self) -> None:
        """Run feature usage audit."""
        logger.info("🚀 Starting feature usage VectorBT audit...")
        
        results = self.audit_core_features()
        report = self.generate_feature_usage_report(results)
        
        print(report)
        
        # Save report to file
        report_file = self.workspace_root / "feature_usage_vectorbt_audit_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"\n📄 Detailed report saved to: {report_file}")


def main():
    """Main execution function."""
    auditor = FeatureUsageAuditor()
    auditor.run_feature_usage_audit()


if __name__ == "__main__":
    main()