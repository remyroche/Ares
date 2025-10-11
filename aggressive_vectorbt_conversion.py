#!/usr/bin/env python3
"""
Aggressive VectorBT Conversion Script

This script aggressively converts ALL remaining files with rolling operations
to use VectorBT for maximum performance.

Usage:
    python3 aggressive_vectorbt_conversion.py
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AggressiveVectorBTConverter:
    """Aggressively converts all files to use VectorBT operations."""
    
    def __init__(self, workspace_root: str = "/workspace"):
        self.workspace_root = Path(workspace_root)
        self.converted_files = []
        self.failed_files = []
        self.skipped_files = []
        
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
            if 'import vectorbt as vbt' in content or 'VECTORBT_AVAILABLE' in content:
                self.skipped_files.append(str(file_path))
                return True
            
            # Skip test files and __init__ files
            if 'test' in str(file_path).lower() or file_path.name == '__init__.py':
                self.skipped_files.append(str(file_path))
                return True
            
            # Check if file has rolling operations
            has_rolling = '.rolling(' in content
            if not has_rolling:
                self.skipped_files.append(str(file_path))
                return True
            
            # Add VectorBT imports if not present
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
            # Simple rolling operations with VectorBT optimization
            (r'(\w+)\.rolling\(window=(\w+)\)\.mean\(\)', 
             r'rolling_mean(\1, window=\2) if VECTORBT_AVAILABLE and len(\1) > 1000 else \1.rolling(window=\2).mean()'),
            (r'(\w+)\.rolling\(window=(\w+)\)\.std\(\)', 
             r'rolling_std(\1, window=\2) if VECTORBT_AVAILABLE and len(\1) > 1000 else \1.rolling(window=\2).std()'),
            (r'(\w+)\.rolling\(window=(\w+)\)\.var\(\)', 
             r'rolling_var(\1, window=\2) if VECTORBT_AVAILABLE and len(\1) > 1000 else \1.rolling(window=\2).var()'),
            (r'(\w+)\.rolling\(window=(\w+)\)\.min\(\)', 
             r'rolling_min(\1, window=\2) if VECTORBT_AVAILABLE and len(\1) > 1000 else \1.rolling(window=\2).min()'),
            (r'(\w+)\.rolling\(window=(\w+)\)\.max\(\)', 
             r'rolling_max(\1, window=\2) if VECTORBT_AVAILABLE and len(\1) > 1000 else \1.rolling(window=\2).max()'),
            (r'(\w+)\.rolling\(window=(\w+)\)\.sum\(\)', 
             r'rolling_sum(\1, window=\2) if VECTORBT_AVAILABLE and len(\1) > 1000 else \1.rolling(window=\2).sum()'),
            
            # Rolling operations with apply
            (r'(\w+)\.rolling\(window=(\w+)\)\.apply\(([^)]+)\)', 
             r'rolling_apply(\1, \3, window=\2) if VECTORBT_AVAILABLE and len(\1) > 1000 else \1.rolling(window=\2).apply(\3)'),
            
            # DataFrame column operations
            (r'data\[[\'"]([^\'"]+)[\'"]\]\.rolling\(window=(\w+)\)\.mean\(\)', 
             r'rolling_mean(data["\1"], window=\2) if VECTORBT_AVAILABLE and len(data) > 1000 else data["\1"].rolling(window=\2).mean()'),
            (r'data\[[\'"]([^\'"]+)[\'"]\]\.rolling\(window=(\w+)\)\.std\(\)', 
             r'rolling_std(data["\1"], window=\2) if VECTORBT_AVAILABLE and len(data) > 1000 else data["\1"].rolling(window=\2).std()'),
            (r'data\[[\'"]([^\'"]+)[\'"]\]\.rolling\(window=(\w+)\)\.var\(\)', 
             r'rolling_var(data["\1"], window=\2) if VECTORBT_AVAILABLE and len(data) > 1000 else data["\1"].rolling(window=\2).var()'),
            (r'data\[[\'"]([^\'"]+)[\'"]\]\.rolling\(window=(\w+)\)\.min\(\)', 
             r'rolling_min(data["\1"], window=\2) if VECTORBT_AVAILABLE and len(data) > 1000 else data["\1"].rolling(window=\2).min()'),
            (r'data\[[\'"]([^\'"]+)[\'"]\]\.rolling\(window=(\w+)\)\.max\(\)', 
             r'rolling_max(data["\1"], window=\2) if VECTORBT_AVAILABLE and len(data) > 1000 else data["\1"].rolling(window=\2).max()'),
            (r'data\[[\'"]([^\'"]+)[\'"]\]\.rolling\(window=(\w+)\)\.sum\(\)', 
             r'rolling_sum(data["\1"], window=\2) if VECTORBT_AVAILABLE and len(data) > 1000 else data["\1"].rolling(window=\2).sum()'),
            
            # Series operations
            (r'(\w+)\.rolling\((\w+)\)\.mean\(\)', 
             r'rolling_mean(\1, window=\2) if VECTORBT_AVAILABLE and len(\1) > 1000 else \1.rolling(\2).mean()'),
            (r'(\w+)\.rolling\((\w+)\)\.std\(\)', 
             r'rolling_std(\1, window=\2) if VECTORBT_AVAILABLE and len(\1) > 1000 else \1.rolling(\2).std()'),
            (r'(\w+)\.rolling\((\w+)\)\.var\(\)', 
             r'rolling_var(\1, window=\2) if VECTORBT_AVAILABLE and len(\1) > 1000 else \1.rolling(\2).var()'),
            (r'(\w+)\.rolling\((\w+)\)\.min\(\)', 
             r'rolling_min(\1, window=\2) if VECTORBT_AVAILABLE and len(\1) > 1000 else \1.rolling(\2).min()'),
            (r'(\w+)\.rolling\((\w+)\)\.max\(\)', 
             r'rolling_max(\1, window=\2) if VECTORBT_AVAILABLE and len(\1) > 1000 else \1.rolling(\2).max()'),
            (r'(\w+)\.rolling\((\w+)\)\.sum\(\)', 
             r'rolling_sum(\1, window=\2) if VECTORBT_AVAILABLE and len(\1) > 1000 else \1.rolling(\2).sum()'),
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
        """Convert all remaining files with rolling operations."""
        logger.info("🔄 Aggressively converting all remaining files to use VectorBT...")
        
        # Search all directories for Python files
        search_dirs = [
            "src/training",
            "src/trading", 
            "src/analyst",
            "src/feature_selection",
            "src/monitoring",
            "src/supervisor",
            "src/tactician",
            "src/strategist",
            "src/utils"
        ]
        
        all_files = []
        
        for search_dir in search_dirs:
            dir_path = self.workspace_root / search_dir
            if dir_path.exists():
                logger.info(f"Scanning {search_dir}...")
                for py_file in dir_path.rglob("*.py"):
                    if py_file.name != "__init__.py":
                        all_files.append(py_file)
        
        logger.info(f"Found {len(all_files)} Python files to process...")
        
        # Convert files in batches
        batch_size = 50
        for i in range(0, len(all_files), batch_size):
            batch = all_files[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(all_files) + batch_size - 1)//batch_size}...")
            
            for file_path in batch:
                self.convert_file(file_path)
    
    def generate_conversion_report(self):
        """Generate conversion report."""
        logger.info("\n" + "="*80)
        logger.info("AGGRESSIVE VECTORBT CONVERSION REPORT")
        logger.info("="*80)
        
        logger.info(f"\n✅ Successfully converted {len(self.converted_files)} files:")
        for file_path in self.converted_files[:20]:  # Show first 20
            logger.info(f"  - {file_path}")
        
        if len(self.converted_files) > 20:
            logger.info(f"  ... and {len(self.converted_files) - 20} more files")
        
        logger.info(f"\n⏭️  Skipped {len(self.skipped_files)} files (already have VectorBT or no rolling ops)")
        
        if self.failed_files:
            logger.info(f"\n❌ Failed to convert {len(self.failed_files)} files:")
            for file_path in self.failed_files[:10]:  # Show first 10
                logger.info(f"  - {file_path}")
            
            if len(self.failed_files) > 10:
                logger.info(f"  ... and {len(self.failed_files) - 10} more files")
        
        logger.info(f"\n📊 Summary:")
        logger.info(f"  - Total files processed: {len(self.converted_files) + len(self.failed_files) + len(self.skipped_files)}")
        logger.info(f"  - Successfully converted: {len(self.converted_files)}")
        logger.info(f"  - Skipped: {len(self.skipped_files)}")
        logger.info(f"  - Failed: {len(self.failed_files)}")
        
        total_processed = len(self.converted_files) + len(self.failed_files) + len(self.skipped_files)
        if total_processed > 0:
            success_rate = len(self.converted_files) / total_processed * 100
            logger.info(f"  - Success rate: {success_rate:.1f}%")
        
        logger.info("\n🎉 Aggressive VectorBT conversion complete!")


def main():
    """Main execution function."""
    logger.info("🚀 Starting aggressive VectorBT conversion...")
    
    converter = AggressiveVectorBTConverter()
    converter.convert_all_remaining_files()
    converter.generate_conversion_report()


if __name__ == "__main__":
    main()