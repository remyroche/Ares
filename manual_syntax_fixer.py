#!/usr/bin/env python3
"""
Manual syntax fixer for the most common issues in the codebase.
"""

import os
import re
import sys
import json
from pathlib import Path
from datetime import datetime
import shutil

class ManualSyntaxFixer:
    def __init__(self):
        self.backup_dir = Path("manual_fix_backups")
        self.backup_dir.mkdir(exist_ok=True)
        self.fixes_applied = 0
        self.files_processed = 0
        
    def backup_file(self, filepath):
        """Create backup before modifying."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"{filepath.name}.{timestamp}.bak"
        shutil.copy2(filepath, backup_path)
        return backup_path
        
    def fix_common_patterns(self, content):
        """Fix the most common syntax error patterns."""
        lines = content.split('\n')
        fixed_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Fix pattern: misplaced code in __init__ definition
            if 'def __init__(' in line and i + 1 < len(lines):
                # Check if next line has misplaced code
                next_line = lines[i + 1]
                if 'self.' in next_line and '=' in next_line and not next_line.strip().startswith('self,'):
                    # This is likely misplaced initialization code
                    fixed_lines.append(line)
                    # Skip the misplaced line
                    i += 2
                    continue
                    
            # Fix __future__ imports location
            if 'from __future__ import' in line:
                # This should be at the top - we'll handle it separately
                i += 1
                continue
                
            fixed_lines.append(line)
            i += 1
            
        # Now handle __future__ imports
        future_imports = [l for l in lines if 'from __future__ import' in l]
        if future_imports:
            # Reconstruct with __future__ at top
            result = []
            
            # Find shebang and encoding
            for line in fixed_lines[:3]:
                if line.startswith('#!') or 'coding' in line:
                    result.append(line)
                    
            # Add __future__ imports
            if result:
                result.append('')
            result.extend(future_imports)
            
            # Add the rest
            result.append('')
            for line in fixed_lines:
                if 'from __future__ import' not in line:
                    if not (line.startswith('#!') or ('coding' in line and fixed_lines.index(line) < 3)):
                        result.append(line)
                        
            return '\n'.join(result)
            
        return '\n'.join(fixed_lines)
        
    def fix_specific_file_patterns(self, filepath, content):
        """Apply file-specific fixes based on known patterns."""
        filename = filepath.name
        
        # Fix monitoring files
        if filename in ['performance_dashboard.py', 'performance_monitor.py']:
            # Remove problematic import aliases
            content = re.sub(r'\s+as\s+\w+_src_\w+', '', content)
            
        # Fix supervisor.py
        if filename == 'supervisor.py':
            # Fix specific syntax error at line 39
            content = re.sub(r'(\n\s+)(\w+)\s*=\s*([^,\n]+)(?=\n\s+self,)', r'\1self,\n\1\2 = \3', content)
            
        # Fix files with unterminated strings
        if 'unterminated string literal' in str(filepath):
            # Ensure all triple quotes are closed
            if content.count('"""') % 2 != 0:
                content += '\n"""'
            if content.count("'''") % 2 != 0:
                content += "\n'''"
                
        return content
        
    def fix_file(self, filepath):
        """Fix a single file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Create backup
            self.backup_file(filepath)
            
            # Apply fixes
            original = content
            content = self.fix_common_patterns(content)
            content = self.fix_specific_file_patterns(filepath, content)
            
            # Only write if changed
            if content != original:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.fixes_applied += 1
                print(f"  ✓ Fixed: {filepath.name}")
            else:
                print(f"  → No changes: {filepath.name}")
                
            self.files_processed += 1
            return True
            
        except Exception as e:
            print(f"  ✗ Error: {filepath.name} - {e}")
            return False


def main():
    # Key files that need fixing
    priority_files = [
        "/workspace/src/monitoring/performance_dashboard.py",
        "/workspace/src/monitoring/performance_monitor.py",
        "/workspace/src/supervisor/supervisor.py",
        "/workspace/src/tactician/tactician.py",
        "/workspace/src/analyst/analyst.py",
        "/workspace/src/launcher/enhanced_trading_launcher.py",
        "/workspace/src/interfaces/enhanced_event_bus.py",
        "/workspace/src/pipelines/live_trading_pipeline.py",
        "/workspace/src/training/steps/step1/data_quality_dashboard.py",
        "/workspace/src/training/steps/step1/comprehensive_gap_filler.py",
    ]
    
    fixer = ManualSyntaxFixer()
    
    print("Manual Syntax Fixer")
    print("=" * 60)
    print(f"Processing {len(priority_files)} priority files...")
    
    for filepath in priority_files:
        if os.path.exists(filepath):
            print(f"\nProcessing: {filepath}")
            fixer.fix_file(Path(filepath))
            
    print("\n" + "=" * 60)
    print(f"Summary:")
    print(f"  Files processed: {fixer.files_processed}")
    print(f"  Files fixed: {fixer.fixes_applied}")
    
    # Now run validation
    print("\nValidating fixes...")
    valid_count = 0
    for filepath in priority_files:
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    compile(f.read(), filepath, 'exec')
                valid_count += 1
                print(f"  ✓ Valid: {Path(filepath).name}")
            except SyntaxError as e:
                print(f"  ✗ Still invalid: {Path(filepath).name} - {e}")
                
    print(f"\nValid files: {valid_count}/{len(priority_files)}")


if __name__ == "__main__":
    main()