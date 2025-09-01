#!/usr/bin/env python3
"""
Resolve Merge Conflicts
Automatically resolves merge conflicts by keeping our changes.
"""

import os
import re
import subprocess

def resolve_conflicts():
    """Resolve merge conflicts by keeping our changes."""
    
    # Get list of conflicted files
    result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
    conflicted_files = []
    
    for line in result.stdout.split('\n'):
        if line.startswith('UU ') or line.startswith('UD ') or line.startswith('DU '):
            file_path = line[3:]  # Remove the status prefix
            conflicted_files.append(file_path)
    
    print(f"Found {len(conflicted_files)} conflicted files")
    
    resolved_count = 0
    
    for file_path in conflicted_files:
        try:
            print(f"Resolving conflicts in: {file_path}")
            
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"  File {file_path} doesn't exist, skipping")
                continue
            
            # Read the file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Remove conflict markers and keep our changes (HEAD)
            # Pattern: <<<<<<< HEAD ... ======= ... >>>>>>> branch
            content = re.sub(
                r'<<<<<<< HEAD\s*\n(.*?)\s*\n=======\s*\n(.*?)\s*\n>>>>>>> [^\n]*\s*\n',
                r'\1\n',
                content,
                flags=re.DOTALL
            )
            
            # Remove any remaining conflict markers
            content = re.sub(r'<<<<<<< HEAD\s*\n', '', content)
            content = re.sub(r'=======\s*\n', '', content)
            content = re.sub(r'>>>>>>> [^\n]*\s*\n', '', content)
            
            # Remove any remaining conflict markers without newlines
            content = re.sub(r'<<<<<<< HEAD', '', content)
            content = re.sub(r'=======', '', content)
            content = re.sub(r'>>>>>>> [^\n]*', '', content)
            
            # Write the resolved content back
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                resolved_count += 1
                print(f"  ✅ Resolved conflicts")
            else:
                print(f"  ⚠️  No conflicts found in file")
        
        except Exception as e:
            print(f"  ❌ Error resolving {file_path}: {e}")
    
    print(f"\nResolved conflicts in {resolved_count} files")
    return resolved_count

def add_resolved_files():
    """Add resolved files to staging area."""
    try:
        # Add all resolved files
        result = subprocess.run(['git', 'add', '.'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Added resolved files to staging area")
            return True
        else:
            print(f"❌ Error adding files: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error adding files: {e}")
        return False

def commit_resolution():
    """Commit the conflict resolution."""
    try:
        result = subprocess.run([
            'git', 'commit', '-m', 'Resolve merge conflicts - keep code quality improvements'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Committed conflict resolution")
            return True
        else:
            print(f"❌ Error committing: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error committing: {e}")
        return False

def main():
    """Main function to resolve merge conflicts."""
    print("🔧 Starting Merge Conflict Resolution")
    
    # Resolve conflicts
    resolved_count = resolve_conflicts()
    
    if resolved_count > 0:
        # Add resolved files
        if add_resolved_files():
            # Commit the resolution
            if commit_resolution():
                print("\n🎉 Merge conflicts resolved successfully!")
                print("📊 Summary:")
                print(f"   - Files resolved: {resolved_count}")
                print(f"   - Status: Ready for PR")
            else:
                print("\n❌ Failed to commit resolution")
        else:
            print("\n❌ Failed to add resolved files")
    else:
        print("\n⚠️  No conflicts were resolved")

if __name__ == "__main__":
    main()