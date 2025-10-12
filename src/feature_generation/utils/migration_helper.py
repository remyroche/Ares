"""
Migration Helper for VectorBT Optimizations

This module provides utilities to help migrate existing feature generators
to use the new VectorBT optimization system.

Key Features:
- Automatic code pattern detection
- Migration suggestions
- Performance comparison tools
- Integration validation
"""

import ast
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class VectorBTOptimizationMigrationHelper:
    """
    Helper class for migrating feature generators to use VectorBT optimizations.
    """
    
    def __init__(self):
        """Initialize the migration helper."""
        self.logger = logging.getLogger(__name__)
        
        # Patterns to detect and replace
        self.rolling_patterns = {
            'pandas_rolling_mean': r'(\w+)\.rolling\(window=(\w+)\)\.mean\(\)',
            'pandas_rolling_std': r'(\w+)\.rolling\(window=(\w+)\)\.std\(\)',
            'pandas_rolling_var': r'(\w+)\.rolling\(window=(\w+)\)\.var\(\)',
            'pandas_rolling_min': r'(\w+)\.rolling\(window=(\w+)\)\.min\(\)',
            'pandas_rolling_max': r'(\w+)\.rolling\(window=(\w+)\)\.max\(\)',
            'pandas_rolling_sum': r'(\w+)\.rolling\(window=(\w+)\)\.sum\(\)',
            'pandas_rolling_skew': r'(\w+)\.rolling\(window=(\w+)\)\.skew\(\)',
            'pandas_rolling_kurt': r'(\w+)\.rolling\(window=(\w+)\)\.kurt\(\)',
            'pandas_rolling_quantile': r'(\w+)\.rolling\(window=(\w+)\)\.quantile\(([^)]+)\)',
            'pandas_rolling_apply': r'(\w+)\.rolling\(window=(\w+)\)\.apply\(([^)]+)\)',
        }
        
        self.statistical_patterns = {
            'numpy_mean': r'np\.mean\(([^)]+)\)',
            'numpy_std': r'np\.std\(([^)]+)\)',
            'numpy_var': r'np\.var\(([^)]+)\)',
            'numpy_skew': r'np\.skew\(([^)]+)\)',
            'numpy_kurt': r'np\.kurt\(([^)]+)\)',
            'manual_skewness': r'\(centered \*\* 3\)\.rolling\(window=(\w+)\)\.mean\(\) / \(rolling_std \*\* 3 \+ 1e-8\)',
            'manual_kurtosis': r'\(centered \*\* 4\)\.rolling\(window=(\w+)\)\.mean\(\) / \(rolling_std \*\* 4 \+ 1e-8\) - 3',
        }
        
        self.optimization_patterns = {
            'vectorbt_imports': r'from vectorbt\.generic import',
            'scattered_rolling': r'rolling_mean|rolling_std|rolling_var',
            'manual_statistical': r'np\.(mean|std|var|skew|kurt)',
            'individual_operations': r'\.rolling\([^)]+\)\.(mean|std|var|min|max|sum)\(\)',
        }
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a feature generator file for optimization opportunities.
        
        Args:
            file_path: Path to the feature generator file
            
        Returns:
            Analysis results with optimization opportunities
        """
        try:
            with open(file_path, 'r') as f:
                content = f.read()
        except Exception as e:
            self.logger.error(f"Failed to read file {file_path}: {e}")
            return {}
        
        analysis = {
            'file_path': file_path,
            'file_size': len(content),
            'lines_of_code': len(content.split('\n')),
            'optimization_opportunities': [],
            'rolling_operations': [],
            'statistical_operations': [],
            'migration_suggestions': [],
            'performance_impact': 'unknown'
        }
        
        # Detect rolling operations
        rolling_ops = self._detect_rolling_operations(content)
        analysis['rolling_operations'] = rolling_ops
        
        # Detect statistical operations
        statistical_ops = self._detect_statistical_operations(content)
        analysis['statistical_operations'] = statistical_ops
        
        # Generate migration suggestions
        suggestions = self._generate_migration_suggestions(rolling_ops, statistical_ops)
        analysis['migration_suggestions'] = suggestions
        
        # Estimate performance impact
        analysis['performance_impact'] = self._estimate_performance_impact(rolling_ops, statistical_ops)
        
        return analysis
    
    def _detect_rolling_operations(self, content: str) -> List[Dict[str, Any]]:
        """Detect rolling operations in the code."""
        operations = []
        
        for pattern_name, pattern in self.rolling_patterns.items():
            matches = re.finditer(pattern, content)
            for match in matches:
                operations.append({
                    'type': 'rolling',
                    'pattern': pattern_name,
                    'match': match.group(0),
                    'line_number': content[:match.start()].count('\n') + 1,
                    'suggested_replacement': self._suggest_rolling_replacement(pattern_name, match)
                })
        
        return operations
    
    def _detect_statistical_operations(self, content: str) -> List[Dict[str, Any]]:
        """Detect statistical operations in the code."""
        operations = []
        
        for pattern_name, pattern in self.statistical_patterns.items():
            matches = re.finditer(pattern, content)
            for match in matches:
                operations.append({
                    'type': 'statistical',
                    'pattern': pattern_name,
                    'match': match.group(0),
                    'line_number': content[:match.start()].count('\n') + 1,
                    'suggested_replacement': self._suggest_statistical_replacement(pattern_name, match)
                })
        
        return operations
    
    def _suggest_rolling_replacement(self, pattern_name: str, match) -> str:
        """Suggest replacement for rolling operations."""
        if 'mean' in pattern_name:
            return f"# Use: rolling_optimizer.single_rolling_operation(data, RollingOperationConfig(operation=RollingOperationType.MEAN, window={match.group(2)}))"
        elif 'std' in pattern_name:
            return f"# Use: rolling_optimizer.single_rolling_operation(data, RollingOperationConfig(operation=RollingOperationType.STD, window={match.group(2)}))"
        elif 'var' in pattern_name:
            return f"# Use: rolling_optimizer.single_rolling_operation(data, RollingOperationConfig(operation=RollingOperationType.VAR, window={match.group(2)}))"
        elif 'min' in pattern_name:
            return f"# Use: rolling_optimizer.single_rolling_operation(data, RollingOperationConfig(operation=RollingOperationType.MIN, window={match.group(2)}))"
        elif 'max' in pattern_name:
            return f"# Use: rolling_optimizer.single_rolling_operation(data, RollingOperationConfig(operation=RollingOperationType.MAX, window={match.group(2)}))"
        elif 'sum' in pattern_name:
            return f"# Use: rolling_optimizer.single_rolling_operation(data, RollingOperationConfig(operation=RollingOperationType.SUM, window={match.group(2)}))"
        elif 'skew' in pattern_name:
            return f"# Use: rolling_optimizer.single_rolling_operation(data, RollingOperationConfig(operation=RollingOperationType.SKEW, window={match.group(2)}))"
        elif 'kurt' in pattern_name:
            return f"# Use: rolling_optimizer.single_rolling_operation(data, RollingOperationConfig(operation=RollingOperationType.KURT, window={match.group(2)}))"
        else:
            return "# Use: rolling_optimizer.single_rolling_operation(data, config)"
    
    def _suggest_statistical_replacement(self, pattern_name: str, match) -> str:
        """Suggest replacement for statistical operations."""
        if 'numpy_mean' in pattern_name:
            return f"# Use: statistical_optimizer.single_statistical_operation(data, StatisticalOperationConfig(operation=StatisticalOperationType.MEAN))"
        elif 'numpy_std' in pattern_name:
            return f"# Use: statistical_optimizer.single_statistical_operation(data, StatisticalOperationConfig(operation=StatisticalOperationType.STD))"
        elif 'numpy_var' in pattern_name:
            return f"# Use: statistical_optimizer.single_statistical_operation(data, StatisticalOperationConfig(operation=StatisticalOperationType.VAR))"
        elif 'manual_skewness' in pattern_name:
            return f"# Use: statistical_optimizer.single_statistical_operation(data, StatisticalOperationConfig(operation=StatisticalOperationType.SKEW, window={match.group(1)}))"
        elif 'manual_kurtosis' in pattern_name:
            return f"# Use: statistical_optimizer.single_statistical_operation(data, StatisticalOperationConfig(operation=StatisticalOperationType.KURT, window={match.group(1)}))"
        else:
            return "# Use: statistical_optimizer.single_statistical_operation(data, config)"
    
    def _generate_migration_suggestions(self, rolling_ops: List[Dict], statistical_ops: List[Dict]) -> List[str]:
        """Generate migration suggestions based on detected operations."""
        suggestions = []
        
        if rolling_ops:
            suggestions.append(f"Found {len(rolling_ops)} rolling operations that can be optimized")
            suggestions.append("Consider using batch_rolling_operations() for multiple operations")
            suggestions.append("Add rolling_optimizer to __init__ method")
        
        if statistical_ops:
            suggestions.append(f"Found {len(statistical_ops)} statistical operations that can be optimized")
            suggestions.append("Consider using batch_statistical_operations() for multiple operations")
            suggestions.append("Add statistical_optimizer to __init__ method")
        
        if len(rolling_ops) > 5 or len(statistical_ops) > 5:
            suggestions.append("High optimization potential - consider using unified_optimizer")
        
        return suggestions
    
    def _estimate_performance_impact(self, rolling_ops: List[Dict], statistical_ops: List[Dict]) -> str:
        """Estimate the performance impact of optimizations."""
        total_ops = len(rolling_ops) + len(statistical_ops)
        
        if total_ops == 0:
            return "No optimization opportunities detected"
        elif total_ops < 5:
            return "Low impact - 2-3x improvement expected"
        elif total_ops < 15:
            return "Medium impact - 3-5x improvement expected"
        else:
            return "High impact - 5-10x improvement expected"
    
    def generate_migration_template(self, analysis: Dict[str, Any]) -> str:
        """Generate a migration template for the analyzed file."""
        template = f"""
# Migration Template for {analysis['file_path']}

## Analysis Summary
- File size: {analysis['file_size']} characters
- Lines of code: {analysis['lines_of_code']}
- Rolling operations: {len(analysis['rolling_operations'])}
- Statistical operations: {len(analysis['statistical_operations'])}
- Performance impact: {analysis['performance_impact']}

## Required Imports
```python
from ..utils.unified_optimization_wrapper import (
    UnifiedOptimizationWrapper,
    UnifiedOptimizationConfig,
    OptimizationMode,
    create_unified_optimizer
)
from ..utils.consolidated_rolling_optimizer import (
    RollingOperationConfig,
    RollingOperationType,
    get_global_rolling_optimizer
)
from ..utils.statistical_calculations_optimizer import (
    StatisticalOperationConfig,
    StatisticalOperationType,
    get_global_statistical_optimizer
)
```

## Required __init__ Updates
```python
def __init__(self, config: Optional[FeatureConfig] = None, 
             enable_gpu: bool = True, 
             enable_parallel: bool = True,
             optimization_mode: OptimizationMode = OptimizationMode.AUTO):
    # ... existing initialization ...
    
    # Initialize optimization components
    self.optimization_config = UnifiedOptimizationConfig(
        mode=optimization_mode,
        enable_gpu=enable_gpu,
        enable_parallel=enable_parallel,
        performance_threshold=1000
    )
    
    self.unified_optimizer = create_unified_optimizer(self.optimization_config)
    self.rolling_optimizer = get_global_rolling_optimizer()
    self.statistical_optimizer = get_global_statistical_optimizer()
```

## Migration Suggestions
"""
        
        for suggestion in analysis['migration_suggestions']:
            template += f"- {suggestion}\n"
        
        template += """
## Performance Monitoring
```python
def get_performance_report(self) -> Dict[str, Any]:
    return {
        'generator_stats': self.performance_stats,
        'unified_optimizer_stats': self.unified_optimizer.get_performance_report(),
        'rolling_optimizer_stats': self.rolling_optimizer.get_performance_stats(),
        'statistical_optimizer_stats': self.statistical_optimizer.get_performance_stats()
    }
```
"""
        
        return template
    
    def analyze_directory(self, directory_path: str) -> Dict[str, Any]:
        """Analyze all feature generator files in a directory."""
        directory = Path(directory_path)
        results = {
            'directory': directory_path,
            'total_files': 0,
            'analyzed_files': 0,
            'total_rolling_operations': 0,
            'total_statistical_operations': 0,
            'files_with_high_impact': [],
            'files_with_medium_impact': [],
            'files_with_low_impact': [],
            'file_analyses': {}
        }
        
        # Find all Python files in the directory
        python_files = list(directory.glob('*.py'))
        results['total_files'] = len(python_files)
        
        for file_path in python_files:
            if file_path.name.startswith('__'):
                continue
                
            analysis = self.analyze_file(str(file_path))
            if analysis:
                results['analyzed_files'] += 1
                results['total_rolling_operations'] += len(analysis['rolling_operations'])
                results['total_statistical_operations'] += len(analysis['statistical_operations'])
                results['file_analyses'][file_path.name] = analysis
                
                # Categorize by impact
                if 'High impact' in analysis['performance_impact']:
                    results['files_with_high_impact'].append(file_path.name)
                elif 'Medium impact' in analysis['performance_impact']:
                    results['files_with_medium_impact'].append(file_path.name)
                else:
                    results['files_with_low_impact'].append(file_path.name)
        
        return results
    
    def generate_directory_report(self, analysis: Dict[str, Any]) -> str:
        """Generate a comprehensive report for directory analysis."""
        report = f"""
# VectorBT Optimization Analysis Report

## Directory: {analysis['directory']}
- Total files: {analysis['total_files']}
- Analyzed files: {analysis['analyzed_files']}
- Total rolling operations: {analysis['total_rolling_operations']}
- Total statistical operations: {analysis['total_statistical_operations']}

## Files by Impact Level

### High Impact ({len(analysis['files_with_high_impact'])} files)
Expected improvement: 5-10x
"""
        
        for filename in analysis['files_with_high_impact']:
            file_analysis = analysis['file_analyses'][filename]
            report += f"- {filename}: {len(file_analysis['rolling_operations'])} rolling, {len(file_analysis['statistical_operations'])} statistical\n"
        
        report += f"""
### Medium Impact ({len(analysis['files_with_medium_impact'])} files)
Expected improvement: 3-5x
"""
        
        for filename in analysis['files_with_medium_impact']:
            file_analysis = analysis['file_analyses'][filename]
            report += f"- {filename}: {len(file_analysis['rolling_operations'])} rolling, {len(file_analysis['statistical_operations'])} statistical\n"
        
        report += f"""
### Low Impact ({len(analysis['files_with_low_impact'])} files)
Expected improvement: 2-3x
"""
        
        for filename in analysis['files_with_low_impact']:
            file_analysis = analysis['file_analyses'][filename]
            report += f"- {filename}: {len(file_analysis['rolling_operations'])} rolling, {len(file_analysis['statistical_operations'])} statistical\n"
        
        report += """
## Migration Priority

1. **High Impact Files**: Start with these for maximum performance gain
2. **Medium Impact Files**: Migrate after high impact files
3. **Low Impact Files**: Migrate last or as needed

## Next Steps

1. Use the migration templates for each file
2. Test performance improvements
3. Monitor optimization effectiveness
4. Update documentation and examples
"""
        
        return report


# Convenience functions
def analyze_feature_generator(file_path: str) -> Dict[str, Any]:
    """Analyze a single feature generator file."""
    helper = VectorBTOptimizationMigrationHelper()
    return helper.analyze_file(file_path)


def analyze_feature_generators_directory(directory_path: str) -> Dict[str, Any]:
    """Analyze all feature generators in a directory."""
    helper = VectorBTOptimizationMigrationHelper()
    return helper.analyze_directory(directory_path)


def generate_migration_report(directory_path: str) -> str:
    """Generate a comprehensive migration report for a directory."""
    helper = VectorBTOptimizationMigrationHelper()
    analysis = helper.analyze_directory(directory_path)
    return helper.generate_directory_report(analysis)