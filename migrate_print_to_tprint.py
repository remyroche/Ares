#!/usr/bin/env python3
"""
Print to TPrint Migration Script

This script automatically converts all print statements to tprint statements
and adds the necessary import at the top of Python files.

Features:
- Converts tprint() calls to tprint() calls
- Adds import statement at the top of files
- Handles various print formats (tprint("text"), tprint(var), tprint("text", var), etc.)
- Preserves all arguments and formatting
- Creates backup files before modification
- Supports batch processing of multiple files
- Full backward compatibility

Usage:
    python migrate_print_to_tprint.py <file_or_directory>
    
Examples:
    python migrate_print_to_tprint.py script.py
    python migrate_print_to_tprint.py src/
    python migrate_print_to_tprint.py --dry-run script.py
    python migrate_print_to_tprint.py --backup-dir backups/ src/
"""

import os
import sys
import re
import argparse
import shutil
from pathlib import Path
from typing import List, Tuple, Optional


class PrintToTPrintMigrator:
    """Migrates print statements to tprint statements."""
    
    def __init__(self, backup_dir: Optional[str] = None, dry_run: bool = False):
        self.backup_dir = backup_dir
        self.dry_run = dry_run
        self.files_processed = 0
        self.files_modified = 0
        self.print_statements_converted = 0
        
        # Import statement to add
        self.import_statement = "from src.utils.tprint import tprint"
        
        # Regex patterns for different print statement formats
        self.print_patterns = [
            # Simple tprint("text")
            (r'\bprint\s*\(\s*([^)]+)\s*\)', r'tprint(\1)'),
            # Print with multiple arguments
            (r'\bprint\s*\(\s*([^)]+)\s*\)', r'tprint(\1)'),
        ]
    
    def create_backup(self, file_path: Path) -> Optional[Path]:
        """Create a backup of the file before modification."""
        if not self.backup_dir:
            return None
            
        backup_dir = Path(self.backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Create backup with timestamp
        backup_name = f"{file_path.stem}_backup_{file_path.suffix}"
        backup_path = backup_dir / backup_name
        
        shutil.copy2(file_path, backup_path)
        return backup_path
    
    def has_tprint_import(self, content: str) -> bool:
        """Check if the file already has tprint import."""
        import_patterns = [
            r'from\s+src\.utils\.tprint\s+import\s+tprint',
            r'from\s+\.*utils\.tprint\s+import\s+tprint',
            r'import\s+.*tprint',
        ]
        
        for pattern in import_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        return False
    
    def has_print_statements(self, content: str) -> bool:
        """Check if the file has print statements."""
        return bool(re.search(r'\bprint\s*\(', content))
    
    def add_import_statement(self, content: str) -> str:
        """Add tprint import statement at the top of the file."""
        lines = content.split('\n')
        
        # Find the best place to insert the import
        insert_index = 0
        
        # Skip shebang and encoding declarations
        for i, line in enumerate(lines):
            if line.startswith('#!') or line.startswith('# -*- coding:') or line.startswith('# coding:'):
                insert_index = i + 1
            elif line.strip() == '':
                continue
            elif line.startswith('"""') or line.startswith("'''"):
                # Skip docstrings
                continue
            else:
                break
        
        # Insert the import statement
        lines.insert(insert_index, self.import_statement)
        lines.insert(insert_index + 1, '')  # Add blank line after import
        
        return '\n'.join(lines)
    
    def convert_print_to_tprint(self, content: str) -> Tuple[str, int]:
        """Convert print statements to tprint statements."""
        converted_count = 0
        modified_content = content
        
        # Count print statements first
        print_matches = re.findall(r'\bprint\s*\(', content)
        converted_count = len(print_matches)
        
        # Pattern to match print statements
        # This handles various formats: tprint("text"), tprint(var), tprint("text", var), etc.
        print_pattern = r'\bprint\s*\(([^)]*)\)'
        
        def replace_print(match):
            args = match.group(1)
            return f'tprint({args})'
        
        # Replace all print statements
        modified_content = re.sub(print_pattern, replace_print, modified_content)
        
        return modified_content, converted_count
    
    def process_file(self, file_path: Path) -> bool:
        """Process a single Python file."""
        try:
            # Read the file
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Check if file has print statements
            if not self.has_print_statements(original_content):
                tprint(f"  ⏭️  No print statements found in {file_path}")
                return False
            
            # Check if already has tprint import
            has_import = self.has_tprint_import(original_content)
            
            # Convert print to tprint
            modified_content, converted_count = self.convert_print_to_tprint(original_content)
            
            # Add import if needed
            if not has_import:
                modified_content = self.add_import_statement(modified_content)
            
            # Check if any changes were made
            if modified_content == original_content:
                tprint(f"  ⏭️  No changes needed for {file_path}")
                return False
            
            if self.dry_run:
                tprint(f"  🔍 [DRY RUN] Would modify {file_path}")
                tprint(f"     - Convert {converted_count} print statements to tprint")
                if not has_import:
                    tprint(f"     - Add tprint import statement")
                self.print_statements_converted += converted_count
                return True
            
            # Create backup if requested
            if self.backup_dir:
                backup_path = self.create_backup(file_path)
                tprint(f"  💾 Backup created: {backup_path}")
            
            # Write the modified content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            
            tprint(f"  ✅ Modified {file_path}")
            tprint(f"     - Converted {converted_count} print statements to tprint")
            if not has_import:
                tprint(f"     - Added tprint import statement")
            
            self.print_statements_converted += converted_count
            return True
            
        except Exception as e:
            tprint(f"  ❌ Error processing {file_path}: {e}")
            return False
    
    def process_directory(self, directory_path: Path) -> None:
        """Process all Python files in a directory recursively."""
        python_files = list(directory_path.rglob('*.py'))
        
        if not python_files:
            tprint(f"No Python files found in {directory_path}")
            return
        
        tprint(f"Found {len(python_files)} Python files in {directory_path}")
        
        for file_path in python_files:
            tprint(f"\n📄 Processing {file_path}")
            self.files_processed += 1
            
            if self.process_file(file_path):
                self.files_modified += 1
    
    def process_path(self, path: Path) -> None:
        """Process a file or directory."""
        if path.is_file():
            if path.suffix != '.py':
                tprint(f"❌ {path} is not a Python file")
                return
            
            tprint(f"📄 Processing {path}")
            self.files_processed += 1
            
            if self.process_file(path):
                self.files_modified += 1
                
        elif path.is_dir():
            self.process_directory(path)
        else:
            tprint(f"❌ {path} does not exist")
    
    def print_summary(self) -> None:
        """Print migration summary."""
        tprint("\n" + "="*60)
        tprint("MIGRATION SUMMARY")
        tprint("="*60)
        tprint(f"Files processed: {self.files_processed}")
        tprint(f"Files modified: {self.files_modified}")
        tprint(f"Print statements converted: {self.print_statements_converted}")
        
        if self.dry_run:
            tprint("\n🔍 This was a DRY RUN - no files were actually modified")
            tprint("Run without --dry-run to apply changes")
        else:
            tprint(f"\n✅ Migration completed successfully!")
            if self.backup_dir:
                tprint(f"💾 Backups saved in: {self.backup_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Migrate print statements to tprint statements",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python migrate_print_to_tprint.py script.py
  python migrate_print_to_tprint.py src/
  python migrate_print_to_tprint.py --dry-run script.py
  python migrate_print_to_tprint.py --backup-dir backups/ src/
        """
    )
    
    parser.add_argument(
        'paths',
        nargs='+',
        help='File(s) or directory to process'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without modifying files'
    )
    
    parser.add_argument(
        '--backup-dir',
        help='Directory to store backup files (default: no backups)'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    paths = [Path(p) for p in args.paths]
    for path in paths:
        if not path.exists():
            tprint(f"❌ Path does not exist: {path}")
            sys.exit(1)
    
    # Create migrator
    migrator = PrintToTPrintMigrator(
        backup_dir=args.backup_dir,
        dry_run=args.dry_run
    )
    
    tprint("🔄 Print to TPrint Migration Tool")
    tprint("="*60)
    
    if args.dry_run:
        tprint("🔍 DRY RUN MODE - No files will be modified")
    
    if args.backup_dir:
        tprint(f"💾 Backups will be saved to: {args.backup_dir}")
    
    tprint()
    
    # Process all paths
    for path in paths:
        migrator.process_path(path)
    
    # Print summary
    migrator.print_summary()


if __name__ == "__main__":
    main()