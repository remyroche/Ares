#!/usr/bin/env python3
"""
Fix orphaned import lines.
"""

import re
from pathlib import Path

def fix_orphaned_imports(file_path):
    """Fix files with orphaned import lines."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to find orphaned decorator names not part of proper import
    orphaned_pattern = r'\n\s*(handle_specific_errors|performance_monitor|handle_errors|memory_efficient|pipeline_checkpoint|resource_monitor),?\s*\n'
    
    # Check if file has orphaned imports
    if re.search(orphaned_pattern, content):
        # Replace orphaned imports with proper import statement
        # First, remove the orphaned lines
        content = re.sub(orphaned_pattern, '\n', content)
        
        # Check if we already have core.decorators import
        if 'from src.core.decorators import' not in content:
            # Add import after other imports
            lines = content.split('\n')
            import_added = False
            
            for i, line in enumerate(lines):
                if line.startswith('from src.utils.logger import'):
                    # Insert before logger import
                    lines.insert(i, 'from src.core.decorators import handles_errors, log_execution_time')
                    lines.insert(i+1, '')
                    import_added = True
                    break
            
            if not import_added:
                # Find a good place to add the import
                for i, line in enumerate(lines):
                    if line.startswith('from ') or line.startswith('import '):
                        continue
                    elif i > 0 and (lines[i-1].startswith('from ') or lines[i-1].startswith('import ')):
                        lines.insert(i, 'from src.core.decorators import handles_errors, log_execution_time')
                        lines.insert(i+1, '')
                        break
            
            content = '\n'.join(lines)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    
    return False

def main():
    files_with_orphans = [
        'src/pipelines/components/lifecycle_manager.py',
        'src/pipelines/components/data_manager.py',
        'src/pipelines/live_trading_pipeline.py',
        'src/training/core/checkpoint_manager.py',
        'src/training/core/stage_context.py',
        'src/training/core/pipeline_orchestrator.py',
        'src/exchange/binance.py',
        'src/utils/model_manager.py',
        'src/strategist/strategist_backup.py',
    ]
    
    print("Fixing orphaned imports...\n")
    
    fixed_count = 0
    for file_path in files_with_orphans:
        full_path = Path('/workspace') / file_path
        if full_path.exists():
            print(f"Processing {file_path}...")
            if fix_orphaned_imports(full_path):
                fixed_count += 1
                print(f"  ✓ Fixed")
            else:
                print(f"  - No orphaned imports found")
    
    print(f"\nFixed {fixed_count} files.")

if __name__ == '__main__':
    main()