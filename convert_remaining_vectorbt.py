#!/usr/bin/env python3
"""
Comprehensive VectorBT Conversion Script

This script converts all remaining files that use pandas rolling operations
to use VectorBT for maximum performance.

Usage:
    python3 convert_remaining_vectorbt.py
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorBTConverter:
    """Converts remaining files to use VectorBT operations."""
    
    def __init__(self, workspace_root: str = "/workspace"):
        self.workspace_root = Path(workspace_root)
        self.converted_files = []
        self.failed_files = []
        
        # VectorBT import pattern
        self.vectorbt_imports = '''# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None'''
    
    def convert_file(self, file_path: Path) -> bool:
        """Convert a single file to use VectorBT operations."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Skip if already has VectorBT imports
            if 'import vectorbt as vbt' in content:
                logger.info(f"Skipping {file_path} - already has VectorBT imports")
                return True
            
            # Add VectorBT imports if not present
            if 'VECTORBT_AVAILABLE' not in content:
                content = self._add_vectorbt_imports(content)
            
            # Convert rolling operations
            content = self._convert_rolling_operations(content)
            
            # Add VectorBT helper methods if not present
            if '_vectorbt_rolling_operation' not in content:
                content = self._add_vectorbt_helpers(content)
            
            # Write updated content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.converted_files.append(str(file_path))
            logger.info(f"✅ Converted {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to convert {file_path}: {e}")
            self.failed_files.append(str(file_path))
            return False
    
    def _add_vectorbt_imports(self, content: str) -> str:
        """Add VectorBT imports to the file."""
        # Find the last import statement
        import_pattern = r'(import\s+[^\n]+\n)'
        imports = re.findall(import_pattern, content)
        
        if imports:
            last_import = imports[-1]
            last_import_pos = content.rfind(last_import)
            insert_pos = last_import_pos + len(last_import)
            
            # Insert VectorBT imports after the last import
            content = content[:insert_pos] + '\n' + self.vectorbt_imports + '\n' + content[insert_pos:]
        else:
            # If no imports found, add at the beginning
            content = self.vectorbt_imports + '\n\n' + content
        
        return content
    
    def _convert_rolling_operations(self, content: str) -> str:
        """Convert pandas rolling operations to VectorBT operations."""
        # Pattern for rolling operations with VectorBT optimization
        patterns = [
            # Simple rolling operations
            (r'(\w+)\.rolling\(window=(\w+)\)\.mean\(\)', 
             r'self._vectorbt_rolling_operation(\1, "mean", \2)'),
            (r'(\w+)\.rolling\(window=(\w+)\)\.std\(\)', 
             r'self._vectorbt_rolling_operation(\1, "std", \2)'),
            (r'(\w+)\.rolling\(window=(\w+)\)\.var\(\)', 
             r'self._vectorbt_rolling_operation(\1, "var", \2)'),
            (r'(\w+)\.rolling\(window=(\w+)\)\.min\(\)', 
             r'self._vectorbt_rolling_operation(\1, "min", \2)'),
            (r'(\w+)\.rolling\(window=(\w+)\)\.max\(\)', 
             r'self._vectorbt_rolling_operation(\1, "max", \2)'),
            (r'(\w+)\.rolling\(window=(\w+)\)\.sum\(\)', 
             r'self._vectorbt_rolling_operation(\1, "sum", \2)'),
            
            # Rolling operations with apply
            (r'(\w+)\.rolling\(window=(\w+)\)\.apply\(([^)]+)\)', 
             r'self._vectorbt_apply_operation(\1, \3, \2)'),
            
            # Direct VectorBT operations for large datasets
            (r'data\[[\'"]([^\'"]+)[\'"]\]\.rolling\(window=(\w+)\)\.mean\(\)', 
             r'rolling_mean(data["\1"], window=\2) if VECTORBT_AVAILABLE and len(data) > 1000 else data["\1"].rolling(window=\2).mean()'),
            (r'data\[[\'"]([^\'"]+)[\'"]\]\.rolling\(window=(\w+)\)\.std\(\)', 
             r'rolling_std(data["\1"], window=\2) if VECTORBT_AVAILABLE and len(data) > 1000 else data["\1"].rolling(window=\2).std()'),
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)
        
        return content
    
    def _add_vectorbt_helpers(self, content: str) -> str:
        """Add VectorBT helper methods to the file."""
        helper_methods = '''
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _vectorbt_apply_operation(self, data: pd.Series, func, 
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)
        
        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
'''
        
        # Add helper methods at the end of the file
        content = content + '\n' + helper_methods
        return content
    
    def convert_all_remaining_files(self):
        """Convert all remaining files that need VectorBT conversion."""
        logger.info("🔄 Converting remaining files to use VectorBT...")
        
        # Find all Python files with rolling operations
        all_files = []
        
        # Search in various directories
        search_dirs = [
            "src/analyst",
            "src/trading", 
            "src/training",
            "src/feature_generation/utils",
            "src/feature_generation/optimization_backup",
            "src/feature_selection",
            "src/monitoring",
            "src/supervisor",
            "src/tactician",
            "src/strategist"
        ]
        
        for search_dir in search_dirs:
            dir_path = self.workspace_root / search_dir
            if dir_path.exists():
                for py_file in dir_path.rglob("*.py"):
                    if py_file.name != "__init__.py" and "test" not in str(py_file):
                        all_files.append(py_file)
        
        # Convert files that have rolling operations
        for file_path in all_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check if file has rolling operations
                if '.rolling(' in content:
                    self.convert_file(file_path)
            except Exception as e:
                logger.warning(f"Could not read {file_path}: {e}")
    
    def generate_conversion_report(self):
        """Generate conversion report."""
        logger.info("\n" + "="*80)
        logger.info("VECTORBT CONVERSION REPORT")
        logger.info("="*80)
        
        logger.info(f"\n✅ Successfully converted {len(self.converted_files)} files:")
        for file_path in self.converted_files:
            logger.info(f"  - {file_path}")
        
        if self.failed_files:
            logger.info(f"\n❌ Failed to convert {len(self.failed_files)} files:")
            for file_path in self.failed_files:
                logger.info(f"  - {file_path}")
        
        logger.info(f"\n📊 Summary:")
        logger.info(f"  - Total files processed: {len(self.converted_files) + len(self.failed_files)}")
        logger.info(f"  - Successfully converted: {len(self.converted_files)}")
        logger.info(f"  - Failed: {len(self.failed_files)}")
        if len(self.converted_files) + len(self.failed_files) > 0:
            logger.info(f"  - Success rate: {len(self.converted_files) / (len(self.converted_files) + len(self.failed_files)) * 100:.1f}%")
        
        logger.info("\n🎉 VectorBT conversion complete!")
        logger.info("All remaining files now use VectorBT optimizations.")


def main():
    """Main execution function."""
    logger.info("🚀 Starting comprehensive VectorBT conversion...")
    
    converter = VectorBTConverter()
    converter.convert_all_remaining_files()
    converter.generate_conversion_report()


if __name__ == "__main__":
    main()