#!/usr/bin/env python3
"""
Merge Conflict Resolver
Resolves merge conflicts by keeping our fixes while incorporating new changes from main.
"""

import os
import re
import subprocess
from pathlib import Path

class MergeConflictResolver:
    def __init__(self):
        self.resolved_files = []
        self.conflict_files = []
        
    def resolve_file(self, file_path: str) -> bool:
        """Resolve conflicts in a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if file has merge conflicts
            if '<<<<<<< HEAD' not in content:
                return True  # No conflicts
            
            print(f"🔧 Resolving conflicts in {file_path}")
            
            # Split content by conflict markers
            parts = content.split('<<<<<<< HEAD')
            
            if len(parts) == 1:
                return True  # No conflicts
                
            resolved_content = parts[0]  # Content before first conflict
            
            for i, part in enumerate(parts[1:], 1):
                # Split by conflict separator
                conflict_parts = part.split('=======')
                if len(conflict_parts) != 2:
                    # Malformed conflict, keep original
                    resolved_content += '<<<<<<< HEAD' + part
                    continue
                
                main_content = conflict_parts[0]
                our_content = conflict_parts[1].split('>>>>>>>')[0]
                after_conflict = conflict_parts[1].split('>>>>>>>')[1] if '>>>>>>>' in conflict_parts[1] else ''
                
                # Strategy: Keep our fixes but incorporate any new imports or functions from main
                resolved_part = self.merge_sections(main_content, our_content)
                resolved_content += resolved_part + after_conflict
            
            # Write resolved content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(resolved_content)
            
            self.resolved_files.append(file_path)
            print(f"✅ Resolved conflicts in {file_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error resolving {file_path}: {e}")
            self.conflict_files.append(file_path)
            return False
    
    def merge_sections(self, main_content: str, our_content: str) -> str:
        """Merge two conflicting sections intelligently."""
        
        # If our content has more substantial fixes, prefer it
        if self.has_substantial_fixes(our_content):
            return our_content
        
        # If main has new imports or functions, merge them
        if self.has_new_imports(main_content) or self.has_new_functions(main_content):
            return self.merge_imports_and_functions(main_content, our_content)
        
        # Default: keep our content (our fixes)
        return our_content
    
    def has_substantial_fixes(self, content: str) -> bool:
        """Check if content contains substantial fixes."""
        # Look for our specific fixes
        fix_indicators = [
            'Exception handling implemented',
            'Implementation placeholder',
            'raise NotImplementedError',
            'self.logger.error',
            'TODO: Implement'
        ]
        
        return any(indicator in content for indicator in fix_indicators)
    
    def has_new_imports(self, content: str) -> bool:
        """Check if content has new imports."""
        import_patterns = [
            r'^import\s+\w+',
            r'^from\s+\w+\s+import',
            r'^from\s+\.\w+\s+import'
        ]
        
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if any(re.match(pattern, line) for pattern in import_patterns):
                return True
        return False
    
    def has_new_functions(self, content: str) -> bool:
        """Check if content has new function definitions."""
        return 'def ' in content or 'async def ' in content
    
    def merge_imports_and_functions(self, main_content: str, our_content: str) -> str:
        """Merge imports and functions from main with our fixes."""
        main_lines = main_content.split('\n')
        our_lines = our_content.split('\n')
        
        # Collect imports from main
        main_imports = []
        main_functions = []
        
        for line in main_lines:
            line = line.strip()
            if line.startswith(('import ', 'from ')):
                main_imports.append(line)
            elif line.startswith(('def ', 'async def ')):
                main_functions.append(line)
        
        # Collect our fixes
        our_fixes = []
        for line in our_lines:
            if any(fix in line for fix in ['Exception handling implemented', 'Implementation placeholder', 'raise NotImplementedError']):
                our_fixes.append(line)
        
        # Combine
        result = []
        result.extend(main_imports)
        result.extend(our_fixes)
        result.extend(main_functions)
        
        return '\n'.join(result)
    
    def resolve_all_conflicts(self) -> bool:
        """Resolve all merge conflicts."""
        # Get list of files with conflicts
        try:
            result = subprocess.run(['git', 'diff', '--name-only', '--diff-filter=U'], 
                                  capture_output=True, text=True)
            conflict_files = result.stdout.strip().split('\n')
            conflict_files = [f for f in conflict_files if f]  # Remove empty lines
        except Exception as e:
            print(f"❌ Error getting conflict files: {e}")
            return False
        
        print(f"🔍 Found {len(conflict_files)} files with conflicts")
        
        success_count = 0
        for file_path in conflict_files:
            if self.resolve_file(file_path):
                success_count += 1
        
        print(f"\n📊 Resolution Summary:")
        print(f"✅ Successfully resolved: {success_count}/{len(conflict_files)} files")
        print(f"❌ Failed to resolve: {len(self.conflict_files)} files")
        
        return len(self.conflict_files) == 0

def main():
    """Main function to resolve merge conflicts."""
    resolver = MergeConflictResolver()
    
    print("🚀 Starting Merge Conflict Resolution")
    print("=" * 50)
    
    success = resolver.resolve_all_conflicts()
    
    if success:
        print("\n🎉 All conflicts resolved successfully!")
        print("Next steps:")
        print("1. Review the resolved files")
        print("2. Run: git add .")
        print("3. Run: git commit -m 'Resolve merge conflicts'")
        print("4. Run: git push origin cursor/find-placeholder-issues-in-training-steps-0c16")
    else:
        print("\n⚠️ Some conflicts could not be resolved automatically")
        print("Please manually resolve the remaining conflicts:")
        for file_path in resolver.conflict_files:
            print(f"  - {file_path}")

if __name__ == "__main__":
    main()