#!/usr/bin/env python3
"""
Final cleanup of merge conflict artifacts.
"""

import re
from pathlib import Path

def clean_file(file_path):
    """Clean up a file by removing orphaned decorator names."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    cleaned_lines = []
    skip_lines = set()
    
    # First pass: identify orphaned decorator names
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Check if this is an orphaned decorator name
        if stripped in ['PerformanceLevel,', 'handle_errors,', 'handle_specific_errors,',
                       'memory_efficient,', 'performance_monitor,', 'pipeline_checkpoint,',
                       'resource_monitor,', ')']:
            # Check context - is this part of a valid import?
            if i > 0:
                prev_line = lines[i-1].strip()
                # If previous line has 'from' or ends with comma, might be valid
                if not ('from' in prev_line or prev_line.endswith(',') or prev_line.endswith('(')):
                    skip_lines.add(i)
                    continue
                    
            # Also check if this is just a standalone line
            if i + 1 < len(lines):
                next_line = lines[i+1].strip()
                if not (next_line.endswith(',') or next_line.endswith(')') or 
                       next_line in ['PerformanceLevel,', 'handle_errors,', 'handle_specific_errors,',
                                    'memory_efficient,', 'performance_monitor,', 'pipeline_checkpoint,',
                                    'resource_monitor,', ')']):
                    # If next line isn't a continuation, skip this
                    if i > 0 and 'from' not in lines[i-1]:
                        skip_lines.add(i)
    
    # Second pass: write cleaned content
    for i, line in enumerate(lines):
        if i not in skip_lines:
            cleaned_lines.append(line)
    
    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(cleaned_lines)
    
    return len(skip_lines) > 0

def fix_duplicate_imports(file_path):
    """Fix duplicate import statements."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all imports from src.core.decorators
    imports_pattern = r'from src\.core\.decorators import ([^;\n]+)'
    matches = list(re.finditer(imports_pattern, content))
    
    if len(matches) > 1:
        # Collect all imported items
        all_imports = set()
        for match in matches:
            import_text = match.group(1)
            # Handle both single line and multi-line imports
            if '(' in import_text:
                # Multi-line - need to extract full block
                start = match.start()
                end = content.find(')', start) + 1
                if end > start:
                    import_block = content[match.start():end]
                    items = re.findall(r'(\w+)(?:,|\s*\))', import_block)
                    all_imports.update(items)
            else:
                # Single line
                items = [item.strip() for item in import_text.split(',')]
                all_imports.update(items)
        
        # Remove duplicates and create single import
        if all_imports:
            # Remove all old imports
            for match in reversed(matches):
                if '(' in match.group(1):
                    # Find the closing parenthesis
                    start = match.start()
                    end = content.find(')', start) + 1
                    if end > start:
                        # Also remove any trailing newline
                        if end < len(content) and content[end] == '\n':
                            end += 1
                        content = content[:start] + content[end:]
                else:
                    # Single line import
                    start = match.start()
                    end = match.end()
                    # Include the newline
                    if end < len(content) and content[end] == '\n':
                        end += 1
                    content = content[:start] + content[end:]
            
            # Add single consolidated import at the first location
            import_items = sorted(list(all_imports))
            if len(import_items) > 3:
                new_import = f"from src.core.decorators import (\n"
                for item in import_items[:-1]:
                    new_import += f"    {item},\n"
                new_import += f"    {import_items[-1]}\n)"
            else:
                new_import = f"from src.core.decorators import {', '.join(import_items)}"
            
            # Insert at first import location
            first_match = matches[0]
            insert_pos = first_match.start()
            content = content[:insert_pos] + new_import + '\n' + content[insert_pos:]
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return True
    
    return False

def main():
    files_to_clean = [
        'src/launcher/enhanced_trading_launcher.py',
        'src/interfaces/enhanced_event_bus.py',
        'src/pipelines/base_pipeline.py',
        'src/pipelines/components/monitoring_manager.py',
        'src/pipelines/components/lifecycle_manager.py',
        'src/pipelines/components/data_manager.py',
        'src/pipelines/live_trading_pipeline.py',
        'src/integration/paper_trading_integration.py',
        'src/training/core/checkpoint_manager.py',
        'src/training/core/stage_context.py',
        'src/training/core/pipeline_orchestrator.py',
        'src/exchange/binance.py',
        'src/utils/model_manager.py',
        'src/strategist/strategist_backup.py',
        'src/analyst/enhanced_prediction_integrator.py',
        'src/analyst/liquidation_risk_model.py',
        'src/training/feature_engineering.py',
    ]
    
    print("Performing final cleanup...\n")
    
    for file_path in files_to_clean:
        full_path = Path('/workspace') / file_path
        if full_path.exists():
            print(f"Cleaning {file_path}...")
            
            # First fix duplicates
            dup_fixed = fix_duplicate_imports(full_path)
            
            # Then clean orphaned lines
            orphan_fixed = clean_file(full_path)
            
            if dup_fixed or orphan_fixed:
                print(f"  ✓ Cleaned")
            else:
                print(f"  - Already clean")
    
    print("\nCleanup complete!")

if __name__ == '__main__':
    main()