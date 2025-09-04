#!/usr/bin/env python3
"""
Script to fix undefined names that are likely function parameters or common variables.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Any

def get_parameter_fixes() -> Dict[str, str]:
    """Get fixes for common undefined parameter names."""
    return {
        # Common function parameters
        'symbol': 'symbol: str',
        'exchange': 'exchange: str', 
        'timeframe': 'timeframe: str',
        'request': 'request: Any',
        'features': 'features: List[str]',
        'market_data': 'market_data: pd.DataFrame',
        'data_dir': 'data_dir: str',
        'training_input': 'training_input: Any',
        'training_data': 'training_data: pd.DataFrame',
        'sr_levels': 'sr_levels: Dict[str, Any]',
        'pipeline_state': 'pipeline_state: Dict[str, Any]',
        'model': 'model: Any',
        'kwargs': '**kwargs',
        'decision_id': 'decision_id: str',
        'force_rerun': 'force_rerun: bool = False',
        'failed': 'failed: bool = False',
        'strategy': 'strategy: str',
        'sr_model': 'sr_model: Any',
        'timeframe_name': 'timeframe_name: str',
        'logger': 'logger: Any',
        'trade_context': 'trade_context: Dict[str, Any]',
        'trade_data': 'trade_data: pd.DataFrame',
        'price_data': 'price_data: pd.DataFrame',
        'X_train': 'X_train: np.ndarray',
        'size': 'size: int',
        'traced': 'traced: bool = False',
        'hmm_model': 'hmm_model: Any',
        'gap_info': 'gap_info: Dict[str, Any]',
        'regime_id': 'regime_id: str',
        'centralized_decorators': 'centralized_decorators: Any',
    }

def analyze_file_for_undefined_names(file_path: str, report_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Analyze a file for undefined names that can be fixed."""
    fixes = []
    
    if file_path not in report_data.get('files', {}):
        return fixes
    
    file_data = report_data['files'][file_path]
    if file_data.get('status') != 'success' or file_data.get('total_errors', 0) == 0:
        return fixes
    
    parameter_fixes = get_parameter_fixes()
    
    for error in file_data.get('errors', []):
        if error.get('error_type') == 'undefined_name':
            name = error.get('name', '')
            line_num = error.get('line', 0)
            
            if name in parameter_fixes:
                fixes.append({
                    'name': name,
                    'line': line_num,
                    'fix': parameter_fixes[name],
                    'context': error.get('context', '')
                })
    
    return fixes

def fix_undefined_parameters_in_file(file_path: str, fixes: List[Dict[str, Any]], dry_run: bool = True) -> int:
    """Fix undefined parameters in a file."""
    if not fixes:
        return 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        fixed_count = 0
        
        # Sort fixes by line number (descending) to avoid line number shifts
        fixes.sort(key=lambda x: x['line'], reverse=True)
        
        for fix in fixes:
            line_idx = fix['line'] - 1  # Convert to 0-based index
            if 0 <= line_idx < len(lines):
                line = lines[line_idx]
                name = fix['name']
                fix_text = fix['fix']
                
                # Check if this is likely a function parameter context
                if ('def ' in line or 'lambda ' in line or 
                    line.strip().startswith(name) or 
                    name in line and ('=' in line or ',' in line)):
                    
                    if dry_run:
                        print(f"  Would fix line {fix['line']}: {name} -> {fix_text}")
                    else:
                        # Try to add type annotation or parameter
                        if 'def ' in line:
                            # Function definition - add parameter
                            if name not in line:
                                # Add parameter to function signature
                                if line.strip().endswith(':'):
                                    lines[line_idx] = line.rstrip(':') + f', {fix_text}:'
                                elif line.strip().endswith(')'):
                                    lines[line_idx] = line.rstrip(')') + f', {fix_text})'
                                else:
                                    lines[line_idx] = line + f', {fix_text}'
                        else:
                            # Variable assignment - add type annotation
                            if '=' in line and name in line:
                                lines[line_idx] = line.replace(name, f'{fix_text}')
                    
                    fixed_count += 1
        
        if not dry_run and fixed_count > 0:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
        
        return fixed_count
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0

def main():
    """Main function to fix undefined parameters."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix undefined parameter names')
    parser.add_argument('--report', default='code_quality/reports/undefined_names_current_scan.json',
                       help='Path to undefined names report')
    parser.add_argument('--dry-run', action='store_true', default=True,
                       help='Show what would be fixed without making changes')
    parser.add_argument('--apply', action='store_true',
                       help='Actually apply the fixes')
    
    args = parser.parse_args()
    
    if args.apply:
        args.dry_run = False
    
    # Load report
    try:
        with open(args.report, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
    except FileNotFoundError:
        print(f"Report file not found: {args.report}")
        return
    except json.JSONDecodeError as e:
        print(f"Error parsing report: {e}")
        return
    
    print(f"Analyzing undefined names report: {args.report}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'APPLYING FIXES'}")
    print("=" * 80)
    
    total_fixes = 0
    files_processed = 0
    
    # Process files with most errors first
    files_with_errors = []
    for file_path, file_data in report_data.get('files', {}).items():
        if file_data.get('status') == 'success' and file_data.get('total_errors', 0) > 0:
            files_with_errors.append((file_path, file_data.get('total_errors', 0)))
    
    # Sort by error count (descending)
    files_with_errors.sort(key=lambda x: x[1], reverse=True)
    
    for file_path, error_count in files_with_errors[:20]:  # Process top 20 files
        if not os.path.exists(file_path):
            continue
            
        fixes = analyze_file_for_undefined_names(file_path, report_data)
        if fixes:
            print(f"\n📁 {file_path} ({error_count} errors)")
            print(f"   Found {len(fixes)} parameter fixes:")
            
            fixed_count = fix_undefined_parameters_in_file(file_path, fixes, args.dry_run)
            total_fixes += fixed_count
            files_processed += 1
    
    print("\n" + "=" * 80)
    print(f"Summary:")
    print(f"  Files processed: {files_processed}")
    print(f"  Total fixes: {total_fixes}")
    print(f"  Mode: {'DRY RUN' if args.dry_run else 'APPLIED'}")
    
    if args.dry_run and total_fixes > 0:
        print(f"\nTo apply these fixes, run with --apply flag")

if __name__ == "__main__":
    main()
