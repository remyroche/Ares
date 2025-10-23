#!/usr/bin/env python3
"""
VectorBT import migration script.

This script migrates all VectorBT imports across the codebase to use the new
production-ready VectorBT module instead of direct imports.
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorBTImportMigrator:
    """Migrates VectorBT imports to use the new production module."""
    
    def __init__(self, src_dir: str = "src"):
        self.src_dir = Path(src_dir)
        self.migration_patterns = {
            # Direct vectorbt imports
            r'import vectorbt as vbt': 'from src.vectorbt import vbt',
            r'from vectorbt import': 'from src.vectorbt import',
            r'from vectorbt\.generic import': 'from src.vectorbt import',
            r'from vectorbt\.returns import': 'from src.vectorbt import',
            r'from vectorbt\.portfolio import': 'from src.vectorbt import',
            r'from vectorbt\.indicators\.basic import': 'from src.vectorbt import',
            r'from vectorbt\.utils\.config import': 'from src.vectorbt import',
            r'from vectorbt\.utils\.decorators import': 'from src.vectorbt import',
            r'from vectorbt\.utils\.array_wrapper import': 'from src.vectorbt import',
            r'from vectorbt\.utils\.datetime_ import': 'from src.vectorbt import',
            r'from vectorbt\.utils\.random import': 'from src.vectorbt import',
        }
        
        # Files to exclude from migration
        self.exclude_files = {
            'src/vectorbt/__init__.py',
            'src/vectorbt/install_vectorbt.py',
            'src/vectorbt/validate_installation.py',
            'src/vectorbt/test_vectorbt_integration.py',
            'src/vectorbt/migrate_imports.py',
            'src/vectorbt/README.md',
            'src/vectorbt/requirements.txt',
        }
        
        # Import mapping for specific functions
        self.function_mappings = {
            'rolling_mean': 'rolling_mean',
            'rolling_std': 'rolling_std',
            'rolling_var': 'rolling_var',
            'rolling_min': 'rolling_min',
            'rolling_max': 'rolling_max',
            'rolling_sum': 'rolling_sum',
            'rolling_apply': 'rolling_apply',
            'rolling_corr': 'rolling_corr',
            'rolling_cov': 'rolling_cov',
            'rolling_rank': 'rolling_rank',
            'rolling_quantile': 'rolling_quantile',
            'rolling_skew': 'rolling_skew',
            'rolling_kurt': 'rolling_kurt',
            'scale': 'scale',
            'rank': 'rank',
            'zscore': 'zscore',
            'winsorize': 'winsorize',
            'clip': 'clip',
            'quantile': 'quantile',
            'Returns': 'Returns',
            'Portfolio': 'Portfolio',
            'PortfolioFactory': 'PortfolioFactory',
            'RSI': 'RSI',
            'MACD': 'MACD',
            'BBANDS': 'BBANDS',
            'ATR': 'ATR',
            'STOCH': 'STOCH',
            'SMA': 'SMA',
            'EMA': 'EMA',
            'BollingerBands': 'BollingerBands',
            'ArrayWrapper': 'ArrayWrapper',
            'freq_delta': 'freq_delta',
            'set_seed': 'set_seed',
            'configure': 'configure',
            'cached_method': 'cached_method',
        }
    
    def find_vectorbt_files(self) -> List[Path]:
        """Find all files that import VectorBT."""
        vectorbt_files = []
        
        for py_file in self.src_dir.rglob("*.py"):
            if str(py_file) in self.exclude_files:
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'vectorbt' in content.lower():
                        vectorbt_files.append(py_file)
            except Exception as e:
                logger.warning(f"Could not read {py_file}: {e}")
        
        return vectorbt_files
    
    def analyze_imports(self, file_path: Path) -> Dict[str, List[str]]:
        """Analyze VectorBT imports in a file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        imports = {
            'direct_imports': [],
            'from_imports': [],
            'function_imports': [],
            'vbt_usage': []
        }
        
        lines = content.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Direct imports
            if re.match(r'import vectorbt', line):
                imports['direct_imports'].append((i, line))
            
            # From imports
            elif re.match(r'from vectorbt', line):
                imports['from_imports'].append((i, line))
                
                # Extract function names
                if 'import' in line:
                    functions = re.findall(r'(\w+)', line.split('import')[1])
                    imports['function_imports'].extend(functions)
            
            # vbt usage
            elif 'vbt.' in line:
                imports['vbt_usage'].append((i, line))
        
        return imports
    
    def migrate_file(self, file_path: Path, dry_run: bool = True) -> Tuple[bool, List[str]]:
        """Migrate a single file's VectorBT imports."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            changes = []
            
            # Apply migration patterns
            for pattern, replacement in self.migration_patterns.items():
                if re.search(pattern, content):
                    new_content = re.sub(pattern, replacement, content)
                    if new_content != content:
                        content = new_content
                        changes.append(f"Replaced: {pattern} -> {replacement}")
            
            # Handle try/except blocks for VectorBT imports
            content = self.migrate_try_except_blocks(content, changes)
            
            # Handle VECTORBT_AVAILABLE checks
            content = self.migrate_availability_checks(content, changes)
            
            # Write changes if not dry run
            if not dry_run and content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                changes.append("File updated successfully")
            
            return content != original_content, changes
            
        except Exception as e:
            logger.error(f"Error migrating {file_path}: {e}")
            return False, [f"Error: {e}"]
    
    def migrate_try_except_blocks(self, content: str, changes: List[str]) -> str:
        """Migrate try/except blocks for VectorBT imports."""
        # Pattern for try/except VectorBT import blocks
        pattern = r'try:\s*\n\s*import vectorbt.*?\nexcept ImportError:.*?\n\s*VECTORBT_AVAILABLE = False'
        
        def replace_try_except(match):
            changes.append("Replaced try/except VectorBT import block")
            return "from src.vectorbt import vbt, rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov, scale, rank, zscore, winsorize, clip, quantile, Portfolio, PortfolioFactory, Returns, RSI, MACD, BBANDS, ATR, STOCH, VECTORBT_AVAILABLE"
        
        return re.sub(pattern, replace_try_except, content, flags=re.DOTALL)
    
    def migrate_availability_checks(self, content: str, changes: List[str]) -> str:
        """Migrate VECTORBT_AVAILABLE checks."""
        # Remove VECTORBT_AVAILABLE = True/False assignments
        content = re.sub(r'VECTORBT_AVAILABLE = (True|False)\s*\n', '', content)
        
        # Replace VECTORBT_AVAILABLE checks with direct usage
        content = re.sub(r'if VECTORBT_AVAILABLE:', 'if True:  # VectorBT always available in production')
        content = re.sub(r'if not VECTORBT_AVAILABLE:', 'if False:  # VectorBT always available in production')
        
        changes.append("Updated VECTORBT_AVAILABLE checks")
        return content
    
    def migrate_all_files(self, dry_run: bool = True) -> Dict[str, any]:
        """Migrate all VectorBT files."""
        logger.info("🔍 Finding files with VectorBT imports...")
        vectorbt_files = self.find_vectorbt_files()
        logger.info(f"Found {len(vectorbt_files)} files with VectorBT imports")
        
        results = {
            'total_files': len(vectorbt_files),
            'successful_migrations': 0,
            'failed_migrations': 0,
            'files_changed': 0,
            'changes_by_file': {}
        }
        
        for file_path in vectorbt_files:
            logger.info(f"Migrating {file_path}...")
            
            # Analyze imports first
            imports = self.analyze_imports(file_path)
            
            # Migrate file
            success, changes = self.migrate_file(file_path, dry_run)
            
            if success:
                results['successful_migrations'] += 1
                results['files_changed'] += 1
                results['changes_by_file'][str(file_path)] = changes
                logger.info(f"✅ {file_path} - {len(changes)} changes")
            else:
                results['failed_migrations'] += 1
                logger.warning(f"⚠️ {file_path} - No changes needed or failed")
        
        return results
    
    def generate_migration_report(self, results: Dict[str, any]) -> str:
        """Generate a migration report."""
        report = []
        report.append("# VectorBT Import Migration Report")
        report.append(f"Total files processed: {results['total_files']}")
        report.append(f"Successful migrations: {results['successful_migrations']}")
        report.append(f"Failed migrations: {results['failed_migrations']}")
        report.append(f"Files changed: {results['files_changed']}")
        report.append("")
        
        if results['changes_by_file']:
            report.append("## Changes by File")
            for file_path, changes in results['changes_by_file'].items():
                report.append(f"### {file_path}")
                for change in changes:
                    report.append(f"- {change}")
                report.append("")
        
        return "\n".join(report)

def main():
    """Main migration function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate VectorBT imports to production module")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without applying them")
    parser.add_argument("--src-dir", default="src", help="Source directory to migrate")
    parser.add_argument("--output-report", help="Output file for migration report")
    
    args = parser.parse_args()
    
    migrator = VectorBTImportMigrator(args.src_dir)
    
    logger.info("🚀 Starting VectorBT import migration")
    if args.dry_run:
        logger.info("🔍 DRY RUN MODE - No files will be modified")
    
    # Run migration
    results = migrator.migrate_all_files(dry_run=args.dry_run)
    
    # Generate report
    report = migrator.generate_migration_report(results)
    
    if args.output_report:
        with open(args.output_report, 'w') as f:
            f.write(report)
        logger.info(f"Migration report saved to {args.output_report}")
    else:
        print(report)
    
    logger.info("✅ Migration completed")
    
    if args.dry_run:
        logger.info("Run without --dry-run to apply changes")

if __name__ == "__main__":
    main()