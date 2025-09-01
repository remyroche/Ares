#!/usr/bin/env python3
"""
Dead Code Remover

This script systematically removes dead code based on the analysis results.
It processes the code quality analysis report and removes unused functions,
variables, and other dead code.
"""

import re
import ast
import os
from typing import List, Dict, Set, Tuple


class DeadCodeRemover:
    """Removes dead code from Python files based on analysis results."""
    
    def __init__(self, analysis_file: str):
        self.analysis_file = analysis_file
        self.dead_code_issues = self._parse_analysis_file()
        self.removed_functions = []
        self.removed_variables = []
        
    def _parse_analysis_file(self) -> Dict[str, List[str]]:
        """Parse the analysis file to extract dead code issues."""
        issues = {}
        current_file = None
        
        with open(self.analysis_file, 'r') as f:
            content = f.read()
            
        # Extract file sections
        file_sections = re.split(r'File: (.+?)\n-+\n', content)[1:]
        
        for i in range(0, len(file_sections), 2):
            if i + 1 < len(file_sections):
                file_path = file_sections[i].strip()
                section_content = file_sections[i + 1]
                
                # Extract dead code issues
                dead_code_matches = re.findall(
                    r'- Function \'([^\']+)\' appears to be unused \(line (\d+)\)',
                    section_content
                )
                
                if dead_code_matches:
                    issues[file_path] = dead_code_matches
                    
        return issues
    
    def remove_dead_code_from_file(self, file_path: str, dry_run: bool = True) -> bool:
        """Remove dead code from a single file."""
        if file_path not in self.dead_code_issues:
            return False
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            original_content = content
            modified = False
            
            # Parse the file
            try:
                tree = ast.parse(content)
            except SyntaxError:
                print(f"Skipping {file_path} due to syntax errors")
                return False
                
            # Get line numbers for dead functions
            dead_functions = self.dead_code_issues[file_path]
            
            # Sort by line number (descending) to avoid line number shifts
            dead_functions.sort(key=lambda x: int(x[1]), reverse=True)
            
            lines = content.split('\n')
            
            for func_name, line_num_str in dead_functions:
                line_num = int(line_num_str) - 1  # Convert to 0-based index
                
                if line_num < len(lines):
                    # Find the function definition and remove it
                    removed = self._remove_function(lines, line_num, func_name)
                    if removed:
                        modified = True
                        self.removed_functions.append(f"{file_path}:{func_name}")
                        print(f"{'[DRY RUN] ' if dry_run else ''}Removed function '{func_name}' from {file_path}")
            
            if modified and not dry_run:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
                    
            return modified
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return False
    
    def _remove_function(self, lines: List[str], start_line: int, func_name: str) -> bool:
        """Remove a function starting from the given line."""
        # Find the function definition
        func_pattern = rf'^\s*(?:async\s+)?def\s+{re.escape(func_name)}\s*\('
        
        if start_line >= len(lines) or not re.match(func_pattern, lines[start_line]):
            return False
            
        # Find the end of the function
        indent_level = len(lines[start_line]) - len(lines[start_line].lstrip())
        end_line = start_line
        
        # Look for the end of the function
        for i in range(start_line + 1, len(lines)):
            line = lines[i]
            if not line.strip():  # Empty line
                continue
                
            current_indent = len(line) - len(line.lstrip())
            
            # If we find a line with same or less indentation, we've reached the end
            if current_indent <= indent_level:
                break
                
            end_line = i
        
        # Remove the function (including the line after end_line)
        del lines[start_line:end_line + 1]
        return True
    
    def remove_dead_code_systematically(self, dry_run: bool = True) -> Dict[str, int]:
        """Remove dead code from all files systematically."""
        results = {
            'files_processed': 0,
            'files_modified': 0,
            'functions_removed': 0
        }
        
        print(f"{'[DRY RUN] ' if dry_run else ''}Starting dead code removal...")
        print(f"Found {len(self.dead_code_issues)} files with dead code issues")
        
        for file_path in self.dead_code_issues:
            results['files_processed'] += 1
            
            if self.remove_dead_code_from_file(file_path, dry_run):
                results['files_modified'] += 1
                results['functions_removed'] += len(self.dead_code_issues[file_path])
        
        print(f"\n{'[DRY RUN] ' if dry_run else ''}Dead code removal completed:")
        print(f"- Files processed: {results['files_processed']}")
        print(f"- Files modified: {results['files_modified']}")
        print(f"- Functions removed: {results['functions_removed']}")
        
        if not dry_run and self.removed_functions:
            print(f"\nRemoved functions:")
            for func in self.removed_functions[:10]:  # Show first 10
                print(f"  - {func}")
            if len(self.removed_functions) > 10:
                print(f"  ... and {len(self.removed_functions) - 10} more")
        
        return results


def main():
    """Main function to run dead code removal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Remove dead code from Python files')
    parser.add_argument('analysis_file', help='Path to the code quality analysis file')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be removed without making changes')
    parser.add_argument('--apply', action='store_true', help='Actually apply the changes')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.analysis_file):
        print(f"Analysis file not found: {args.analysis_file}")
        return
    
    remover = DeadCodeRemover(args.analysis_file)
    
    if args.apply:
        print("WARNING: This will permanently remove dead code from your files!")
        response = input("Are you sure you want to continue? (y/N): ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            return
        
        results = remover.remove_dead_code_systematically(dry_run=False)
    else:
        results = remover.remove_dead_code_systematically(dry_run=True)
    
    print(f"\nDead code removal {'completed' if args.apply else 'simulation'} successfully!")


if __name__ == "__main__":
    main()