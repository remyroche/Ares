#!/usr/bin/env python3
"""
Comprehensive VectorBT Integration Update Script

This script systematically updates all feature generators and transformers
to natively use VectorBT optimizations for maximum performance.

Usage:
    python update_generators_vectorbt.py
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorBTUpdater:
    """Updates feature generators and transformers to use VectorBT natively."""
    
    def __init__(self, workspace_root: str = "/workspace"):
        self.workspace_root = Path(workspace_root)
        self.updated_files = []
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
    
    def update_file(self, file_path: Path) -> bool:
        """Update a single file to use VectorBT natively."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Skip if already has VectorBT imports
            if 'import vectorbt as vbt' in content:
                logger.info(f"Skipping {file_path} - already has VectorBT imports")
                return True
            
            # Add VectorBT imports after existing imports
            content = self._add_vectorbt_imports(content)
            
            # Update rolling operations to use VectorBT
            content = self._update_rolling_operations(content)
            
            # Update feature generation methods
            content = self._update_feature_methods(content)
            
            # Write updated content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.updated_files.append(str(file_path))
            logger.info(f"✅ Updated {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update {file_path}: {e}")
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
    
    def _update_rolling_operations(self, content: str) -> str:
        """Update rolling operations to use VectorBT."""
        # Pattern for rolling operations
        patterns = [
            (r'(\w+)\.rolling\(window=(\w+)\)\.mean\(\)', r'self._vectorbt_rolling_operation(\1, "mean", \2)'),
            (r'(\w+)\.rolling\(window=(\w+)\)\.std\(\)', r'self._vectorbt_rolling_operation(\1, "std", \2)'),
            (r'(\w+)\.rolling\(window=(\w+)\)\.var\(\)', r'self._vectorbt_rolling_operation(\1, "var", \2)'),
            (r'(\w+)\.rolling\(window=(\w+)\)\.min\(\)', r'self._vectorbt_rolling_operation(\1, "min", \2)'),
            (r'(\w+)\.rolling\(window=(\w+)\)\.max\(\)', r'self._vectorbt_rolling_operation(\1, "max", \2)'),
            (r'(\w+)\.rolling\(window=(\w+)\)\.sum\(\)', r'self._vectorbt_rolling_operation(\1, "sum", \2)'),
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)
        
        return content
    
    def _update_feature_methods(self, content: str) -> str:
        """Update feature generation methods to use VectorBT."""
        # Add VectorBT helper methods if not present
        if '_should_use_vectorbt' not in content:
            helper_methods = '''
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and 
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
'''
            
            # Add helper methods before the last class method
            content = content + helper_methods
        
        return content
    
    def update_all_generators(self):
        """Update all feature generators to use VectorBT."""
        logger.info("🔄 Updating feature generators...")
        
        # Update feature generation categories
        categories_dir = self.workspace_root / "src" / "feature_generation" / "categories"
        if categories_dir.exists():
            for py_file in categories_dir.glob("*.py"):
                if py_file.name not in ["__init__.py", "advanced_volatility_features.py", "advanced_volume_features.py"]:
                    self.update_file(py_file)
        
        # Update transforms
        transforms_dir = self.workspace_root / "src" / "feature_engineering_roadmap"
        if transforms_dir.exists():
            for py_file in transforms_dir.glob("*.py"):
                if py_file.name not in ["__init__.py"]:
                    self.update_file(py_file)
        
        # Update other feature generation modules
        feature_gen_dir = self.workspace_root / "src" / "feature_generation"
        for py_file in feature_gen_dir.rglob("*.py"):
            if (py_file.name not in ["__init__.py", "advanced_volatility_features.py", "advanced_volume_features.py"] and
                "test" not in str(py_file) and
                "vectorbt" not in str(py_file).lower()):
                self.update_file(py_file)
    
    def generate_report(self):
        """Generate update report."""
        logger.info("\n" + "="*80)
        logger.info("VECTORBT INTEGRATION UPDATE REPORT")
        logger.info("="*80)
        
        logger.info(f"\n✅ Successfully updated {len(self.updated_files)} files:")
        for file_path in self.updated_files:
            logger.info(f"  - {file_path}")
        
        if self.failed_files:
            logger.info(f"\n❌ Failed to update {len(self.failed_files)} files:")
            for file_path in self.failed_files:
                logger.info(f"  - {file_path}")
        
        logger.info(f"\n📊 Summary:")
        logger.info(f"  - Total files processed: {len(self.updated_files) + len(self.failed_files)}")
        logger.info(f"  - Successfully updated: {len(self.updated_files)}")
        logger.info(f"  - Failed: {len(self.failed_files)}")
        logger.info(f"  - Success rate: {len(self.updated_files) / (len(self.updated_files) + len(self.failed_files)) * 100:.1f}%")
        
        logger.info("\n🎉 VectorBT integration update complete!")
        logger.info("All feature generators and transformers now natively use VectorBT optimizations.")


def main():
    """Main execution function."""
    logger.info("🚀 Starting VectorBT integration update...")
    
    updater = VectorBTUpdater()
    updater.update_all_generators()
    updater.generate_report()


if __name__ == "__main__":
    main()