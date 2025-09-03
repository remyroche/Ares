#!/usr/bin/env python3
"""
Fix import and docstring order issues in Python files.
This script specifically targets the common pattern where imports appear before module docstrings.
"""

import os
import re
import ast
from pathlib import Path
from datetime import datetime
import shutil

def fix_file_structure(filepath):
    """Fix the structure of a Python file to ensure proper order."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        lines = content.split('\n')
        
        # Categories
        shebang = []
        encoding = []
        file_comment = []  # Comments that indicate file path
        early_imports = []  # Imports that appear before docstring
        docstring_lines = []
        remaining_lines = []
        
        i = 0
        found_docstring = False
        in_docstring = False
        docstring_delim = None
        
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Shebang
            if i == 0 and line.startswith('#!'):
                shebang.append(line)
                i += 1
                continue
                
            # Encoding
            if i < 2 and line.startswith('#') and ('coding' in line or 'encoding' in line):
                encoding.append(line)
                i += 1
                continue
                
            # File path comment
            if line.startswith('# src/') or line.startswith('# /'):
                file_comment.append(line)
                i += 1
                continue
                
            # Empty line
            if not stripped:
                if not found_docstring and not early_imports:
                    i += 1
                    continue
                else:
                    remaining_lines.append(line)
                    i += 1
                    continue
                    
            # Check for docstring start
            if not found_docstring and (stripped.startswith('"""') or stripped.startswith("'''")):
                found_docstring = True
                in_docstring = True
                docstring_delim = stripped[:3]
                docstring_lines.append(line)
                
                # Check if it's a one-liner
                if stripped.count(docstring_delim) >= 2:
                    in_docstring = False
                i += 1
                continue
                
            # Continue collecting docstring
            if in_docstring:
                docstring_lines.append(line)
                if docstring_delim in stripped and len(docstring_lines) > 1:
                    in_docstring = False
                i += 1
                continue
                
            # If we haven't found docstring yet and this is an import, it's an early import
            if not found_docstring and (stripped.startswith('from ') or stripped.startswith('import ')):
                early_imports.append(line)
                i += 1
                continue
                
            # Everything else
            remaining_lines.append(line)
            i += 1
            
        # Reconstruct the file in proper order
        result = []
        
        # 1. Shebang
        if shebang:
            result.extend(shebang)
            
        # 2. Encoding
        if encoding:
            result.extend(encoding)
            
        # 3. File comment
        if file_comment:
            if result:
                result.append('')
            result.extend(file_comment)
            
        # 4. Module docstring
        if docstring_lines:
            if result:
                result.append('')
            result.extend(docstring_lines)
            
        # 5. Early imports (that were before docstring)
        if early_imports:
            if result:
                result.append('')
            result.extend(early_imports)
            
        # 6. Rest of file
        if remaining_lines:
            if result and result[-1].strip():
                result.append('')
            # Remove leading empty lines
            while remaining_lines and not remaining_lines[0].strip():
                remaining_lines.pop(0)
            result.extend(remaining_lines)
            
        # Fix specific import issues
        final_lines = []
        for line in result:
            # Remove problematic import aliases
            if ' as ' in line and '_src_' in line:
                line = re.sub(r'\s+as\s+\w+_src_\w+', '', line)
            final_lines.append(line)
            
        return '\n'.join(final_lines)
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None


def main():
    # List of files with import/docstring issues
    problem_files = [
        "/workspace/src/analyst/di_analyst.py",
        "/workspace/src/monitoring/performance_dashboard.py",
        "/workspace/src/monitoring/performance_monitor.py",
        "/workspace/src/launcher/enhanced_trading_launcher.py",
        "/workspace/src/interfaces/enhanced_event_bus.py",
        "/workspace/src/supervisor/enhanced_prediction_service.py",
        "/workspace/src/analyst/analyst.py",
        "/workspace/src/analyst/advanced_feature_engineering.py",
        "/workspace/src/analyst/autoencoder_feature_generator.py",
        "/workspace/src/analyst/enhanced_prediction_integrator.py",
        "/workspace/src/analyst/feature_engineering_orchestrator.py",
        "/workspace/src/analyst/liquidation_risk_model.py",
        "/workspace/src/analyst/meta_label_relevance.py",
        "/workspace/src/analyst/ml_confidence_predictor.py",
        "/workspace/src/analyst/unified_regime_classifier.py",
        "/workspace/src/database/migration_utils.py",
        "/workspace/src/database/precomputed_features_manager.py",
        "/workspace/src/exchange/binance.py",
        "/workspace/src/integration/paper_trading_integration.py",
    ]
    
    # Create backup directory
    backup_dir = Path("import_fix_backups")
    backup_dir.mkdir(exist_ok=True)
    
    fixed = 0
    failed = 0
    
    print("Fixing import and docstring order issues...")
    print("=" * 60)
    
    for filepath in problem_files:
        if os.path.exists(filepath):
            print(f"\nProcessing: {filepath}")
            
            # Backup
            backup_path = backup_dir / f"{Path(filepath).name}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
            shutil.copy2(filepath, backup_path)
            
            # Fix
            fixed_content = fix_file_structure(filepath)
            
            if fixed_content:
                try:
                    # Test if it compiles
                    compile(fixed_content, filepath, 'exec')
                    
                    # Write back
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(fixed_content)
                    print(f"  ✓ Fixed successfully")
                    fixed += 1
                except SyntaxError as e:
                    print(f"  ✗ Still has syntax error: {e}")
                    failed += 1
            else:
                print(f"  ✗ Failed to process")
                failed += 1
                
    print("\n" + "=" * 60)
    print(f"Summary:")
    print(f"  Files fixed: {fixed}")
    print(f"  Files failed: {failed}")


if __name__ == "__main__":
    main()