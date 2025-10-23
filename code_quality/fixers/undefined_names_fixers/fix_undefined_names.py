#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Script to fix common undefined names and variables issues found in the repository.
This script analyzes the undefined names report and applies common fixes.
"""

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


class UndefinedNamesFixer:
    """Fix common undefined names and variables issues."""

    def __init__(self, report_path: str):
        self.report_path = report_path
        self.report_data = self._load_report()
        self.common_imports = {
            'pd': 'import pandas as pd',
            'np': 'import numpy as np',
            'datetime': 'from datetime import datetime',
            'Dict': 'from typing import Dict',
            'List': 'from typing import List',
            'Optional': 'from typing import Optional',
            'Any': 'from typing import Any',
            'Tuple': 'from typing import Tuple',
            'Union': 'from typing import Union',
            'Set': 'from typing import Set',
            'Callable': 'from typing import Callable',
            'dataclass': 'from dataclasses import dataclass',
            'cached': 'from functools import cached_property',
            'error': 'from logging import error',
            'info': 'from logging import info',
            'warning': 'from logging import warning',
            'debug': 'from logging import debug',
            'handles_errors': 'from src.utils.error_handler import handles_errors',
        }
        
        # Common patterns that should be ignored (false positives)
        self.ignore_patterns = {
            'symbol', 'exchange', 'market_data', 'regime_info', 'timeframe',
            'training_mode', 'force_reload', 'use_ensemble', 'sr_results',
            'validation_data', 'additional_data', 'current_price', 'volume',
            'timestamp', 'exit_price', 'duration', 'model_input', 'regime_labels',
            'model_predictions', 'actual_outcomes', 'model_uncertainties',
            'current_regime', 'model_names', 'required_count', 'performance',
            'observations', 'regime1', 'regime2', 'timestamps', 'price_movements',
            'data', 'n_scenarios', 'target_movements', 'rows', 'include_metadata',
            'time_range', 'output_path', 'trade_data', 'transition_data',
            'period_days', 'klines_df', 'agg_trades_df', 'futures_df',
            'analyst_confidence_scores', 'tactician_confidence_scores',
            'analyst_signals', 'ml_profit_predictions', 'initialize_sr_parameters',
            'from_regime', 'to_regime', 'confidence', 'websocket', 'PerformanceLevel',
            'MetricsDashboard', 'AdvancedTracer', 'CorrelationManager', 'MLMonitor',
            'ReportScheduler', 'TrackingSystem', 'Supervisor', 'PerformanceMonitor',
            'ModelInput', 'func', 'args', 'kwargs'
        }

    def _load_report(self) -> Dict:
        """Load the undefined names report."""
        with open(self.report_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def analyze_undefined_names(self) -> Dict[str, int]:
        """Analyze the most common undefined names."""
        name_counts = Counter()
        
        for file_path, file_data in self.report_data.get('files', {}).items():
            if file_data.get('status') == 'success':
                for error in file_data.get('errors', []):
                    if error.get('error_type') == 'undefined_name':
                        name = error.get('name', '')
                        if name not in self.ignore_patterns:
                            name_counts[name] += 1
        
        return dict(name_counts.most_common(50))

    def get_files_needing_imports(self) -> Dict[str, Set[str]]:
        """Get files that need specific imports."""
        files_needing_imports = defaultdict(set)
        
        for file_path, file_data in self.report_data.get('files', {}).items():
            if file_data.get('status') == 'success':
                for error in file_data.get('errors', []):
                    if error.get('error_type') == 'undefined_name':
                        name = error.get('name', '')
                        if name in self.common_imports:
                            files_needing_imports[file_path].add(name)
        
        return dict(files_needing_imports)

    def fix_file_imports(self, file_path: str, needed_imports: Set[str]) -> bool:
        """Fix imports in a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            tprint(f"❌ Could not read {file_path}: {e}")
            return False

        # Check if file has syntax errors first
        if any(error.get('error_type') == 'syntax_error' 
               for error in self.report_data.get('files', {}).get(file_path, {}).get('errors', [])):
            tprint(f"⚠️  Skipping {file_path} due to syntax errors")
            return False

        lines = content.split('\n')
        import_lines = []
        other_lines = []
        in_imports = True
        
        # Separate import lines from other lines
        for line in lines:
            stripped = line.strip()
            if in_imports and (stripped.startswith('import ') or stripped.startswith('from ') or stripped == ''):
                import_lines.append(line)
            else:
                in_imports = False
                other_lines.append(line)
        
        # Add missing imports
        existing_imports = set()
        for line in import_lines:
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                # Extract the imported name
                if ' as ' in stripped:
                    imported_name = stripped.split(' as ')[-1].strip()
                elif 'import ' in stripped:
                    imported_name = stripped.split('import ')[-1].strip()
                else:
                    continue
                existing_imports.add(imported_name)
        
        # Add missing imports
        new_imports = []
        for name in needed_imports:
            if name not in existing_imports:
                import_statement = self.common_imports[name]
                new_imports.append(import_statement)
        
        if new_imports:
            # Add new imports
            import_lines.extend([''] + new_imports)
            
            # Reconstruct file content
            new_content = '\n'.join(import_lines + other_lines)
            
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                tprint(f"✅ Fixed imports in {file_path}: {', '.join(needed_imports)}")
                return True
            except Exception as e:
                tprint(f"❌ Could not write {file_path}: {e}")
                return False
        
        return False

    def fix_syntax_errors(self) -> int:
        """Fix common syntax errors."""
        fixed_count = 0
        
        for file_path, file_data in self.report_data.get('files', {}).items():
            if file_data.get('status') == 'success':
                syntax_errors = [error for error in file_data.get('errors', []) 
                               if error.get('error_type') == 'syntax_error']
                
                if syntax_errors:
                    if self._fix_file_syntax_errors(file_path, syntax_errors):
                        fixed_count += 1
        
        return fixed_count

    def _fix_file_syntax_errors(self, file_path: str, syntax_errors: List[Dict]) -> bool:
        """Fix syntax errors in a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            tprint(f"❌ Could not read {file_path}: {e}")
            return False

        original_content = content
        
        # Fix common syntax errors
        for error in syntax_errors:
            context = error.get('context', '')
            if 'unexpected indent' in context:
                # Try to fix indentation issues
                lines = content.split('\n')
                fixed_lines = []
                
                for i, line in enumerate(lines):
                    # Check for lines that start with spaces but should be at column 0
                    if line.strip() and line.startswith(' ') and not line.startswith('    '):
                        # This might be an indentation error
                        # Try to fix by removing leading spaces
                        fixed_line = line.lstrip()
                        if fixed_line:
                            fixed_lines.append(fixed_line)
                        else:
                            fixed_lines.append(line)
                    else:
                        fixed_lines.append(line)
                
                content = '\n'.join(fixed_lines)
        
        # Only write if content changed
        if content != original_content:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                tprint(f"✅ Fixed syntax errors in {file_path}")
                return True
            except Exception as e:
                tprint(f"❌ Could not write {file_path}: {e}")
                return False
        
        return False

    def run_fixes(self) -> Dict[str, int]:
        """Run all fixes and return statistics."""
        tprint("="*70)
        tprint("FIXING UNDEFINED NAMES AND VARIABLES")
        tprint("="*70)
        
        # Analyze undefined names
        tprint("\n📊 Analyzing undefined names...")
        name_counts = self.analyze_undefined_names()
        tprint(f"Top undefined names:")
        for name, count in list(name_counts.items())[:20]:
            tprint(f"  {name}: {count} occurrences")
        
        # Get files needing imports
        tprint("\n🔍 Finding files needing imports...")
        files_needing_imports = self.get_files_needing_imports()
        tprint(f"Found {len(files_needing_imports)} files needing imports")
        
        # Fix imports
        tprint("\n🔧 Fixing imports...")
        import_fixes = 0
        for file_path, needed_imports in files_needing_imports.items():
            if self.fix_file_imports(file_path, needed_imports):
                import_fixes += 1
        
        # Fix syntax errors
        tprint("\n🔧 Fixing syntax errors...")
        syntax_fixes = self.fix_syntax_errors()
        
        return {
            'import_fixes': import_fixes,
            'syntax_fixes': syntax_fixes,
            'total_files_processed': len(files_needing_imports),
            'top_undefined_names': name_counts
        }


def main():
    """Main function to run the undefined names fixer."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fix common undefined names and variables issues"
    )
    parser.add_argument("--report", "-r", 
                       default="code_quality/reports/undefined_names_report.json",
                       help="Path to undefined names report")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be fixed without making changes")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.report):
        tprint(f"❌ Report file not found: {args.report}")
        tprint("Run the undefined names checker first:")
        tprint("python code_quality/check_undefined_names_standalone.py --target . --output code_quality/reports/undefined_names_report.json")
        return 1
    
    fixer = UndefinedNamesFixer(args.report)
    
    if args.dry_run:
        tprint("🔍 DRY RUN - No changes will be made")
        name_counts = fixer.analyze_undefined_names()
        files_needing_imports = fixer.get_files_needing_imports()
        
        tprint(f"\nTop undefined names:")
        for name, count in list(name_counts.items())[:20]:
            tprint(f"  {name}: {count} occurrences")
        
        tprint(f"\nFiles needing imports: {len(files_needing_imports)}")
        for file_path, imports in list(files_needing_imports.items())[:10]:
            tprint(f"  {file_path}: {', '.join(imports)}")
        
        return 0
    
    # Run fixes
    results = fixer.run_fixes()
    
    tprint("\n" + "="*70)
    tprint("FIXING COMPLETED")
    tprint("="*70)
    tprint(f"✅ Import fixes applied: {results['import_fixes']}")
    tprint(f"✅ Syntax fixes applied: {results['syntax_fixes']}")
    tprint(f"📁 Total files processed: {results['total_files_processed']}")
    
    if results['import_fixes'] > 0 or results['syntax_fixes'] > 0:
        tprint(f"\n🔄 Run the checker again to verify fixes:")
        tprint(f"python code_quality/check_undefined_names_standalone.py --target . --output code_quality/reports/undefined_names_report_after_fixes.json")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
