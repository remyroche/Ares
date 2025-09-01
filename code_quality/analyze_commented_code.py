#!/usr/bin/env python3
"""
Analyze commented code blocks in Python files
Identifies commented code that might need to be implemented or removed
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any


class CommentedCodeAnalyzer:
    """Analyzes Python files for commented code blocks."""
    
    def __init__(self, exclusions_file: str = None):
        self.exclusions = self._load_exclusions(exclusions_file)
        self.commented_code_blocks = []
        
    def _load_exclusions(self, exclusions_file: str) -> set:
        """Load exclusion patterns from file."""
        exclusions = set()
        if exclusions_file and os.path.exists(exclusions_file):
            with open(exclusions_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        exclusions.add(line)
        return exclusions
    
    def _should_exclude(self, filepath: str) -> bool:
        """Check if file should be excluded based on patterns."""
        for pattern in self.exclusions:
            if pattern in filepath or filepath.endswith(pattern.replace('*', '')):
                return True
        return False
    
    def analyze_file(self, filepath: str) -> List[Dict[str, Any]]:
        """Analyze a single Python file for commented code blocks."""
        if self._should_exclude(filepath):
            return []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            commented_blocks = []
            
            # Patterns to identify commented code
            code_patterns = [
                r'^\s*#\s*(def\s+\w+\(.*\):)',  # Commented function definitions
                r'^\s*#\s*(class\s+\w+.*:)',   # Commented class definitions
                r'^\s*#\s*(if\s+.*:)',         # Commented if statements
                r'^\s*#\s*(for\s+.*:)',        # Commented for loops
                r'^\s*#\s*(while\s+.*:)',      # Commented while loops
                r'^\s*#\s*(try\s*:)',          # Commented try blocks
                r'^\s*#\s*(except\s+.*:)',    # Commented except blocks
                r'^\s*#\s*(return\s+.*)',     # Commented return statements
                r'^\s*#\s*(import\s+.*)',     # Commented imports
                r'^\s*#\s*(from\s+.*)',       # Commented from imports
                r'^\s*#\s*([a-zA-Z_]\w*\s*=)', # Commented variable assignments
                r'^\s*#\s*([a-zA-Z_]\w*\(.*\))', # Commented function calls
            ]
            
            for i, line in enumerate(lines, 1):
                for pattern in code_patterns:
                    match = re.search(pattern, line)
                    if match:
                        commented_blocks.append({
                            'file': filepath,
                            'line': i,
                            'code': line.strip(),
                            'type': self._classify_commented_code(match.group(1)),
                            'context': self._get_context(lines, i)
                        })
                        break
            
            # Look for multi-line commented blocks
            multi_line_blocks = self._find_multi_line_comments(lines, filepath)
            commented_blocks.extend(multi_line_blocks)
            
            return commented_blocks
            
        except Exception as e:
            print(f"Error analyzing {filepath}: {e}")
            return []
    
    def _classify_commented_code(self, code: str) -> str:
        """Classify the type of commented code."""
        if re.match(r'def\s+\w+\(.*\):', code):
            return 'function_definition'
        elif re.match(r'class\s+\w+.*:', code):
            return 'class_definition'
        elif re.match(r'if\s+.*:', code):
            return 'conditional_statement'
        elif re.match(r'for\s+.*:', code):
            return 'loop_statement'
        elif re.match(r'while\s+.*:', code):
            return 'loop_statement'
        elif re.match(r'try\s*:', code):
            return 'exception_handling'
        elif re.match(r'except\s+.*:', code):
            return 'exception_handling'
        elif re.match(r'return\s+.*', code):
            return 'return_statement'
        elif re.match(r'import\s+.*', code):
            return 'import_statement'
        elif re.match(r'from\s+.*', code):
            return 'import_statement'
        elif re.match(r'[a-zA-Z_]\w*\s*=', code):
            return 'variable_assignment'
        elif re.match(r'[a-zA-Z_]\w*\(.*\)', code):
            return 'function_call'
        else:
            return 'other'
    
    def _find_multi_line_comments(self, lines: List[str], filepath: str) -> List[Dict[str, Any]]:
        """Find multi-line commented code blocks."""
        multi_line_blocks = []
        in_comment_block = False
        comment_start = 0
        comment_lines = []
        
        for i, line in enumerate(lines, 1):
            # Check for start of comment block
            if line.strip().startswith('"""') or line.strip().startswith("'''"):
                if not in_comment_block:
                    in_comment_block = True
                    comment_start = i
                    comment_lines = [line]
                else:
                    # End of comment block
                    in_comment_block = False
                    comment_lines.append(line)
                    
                    # Check if this looks like commented code
                    if self._looks_like_code_block(comment_lines):
                        multi_line_blocks.append({
                            'file': filepath,
                            'line': comment_start,
                            'code': '\n'.join(comment_lines),
                            'type': 'multi_line_code_block',
                            'context': self._get_context(lines, comment_start)
                        })
            elif in_comment_block:
                comment_lines.append(line)
        
        return multi_line_blocks
    
    def _looks_like_code_block(self, comment_lines: List[str]) -> bool:
        """Determine if a comment block looks like it contains code."""
        code_indicators = [
            'def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except',
            'return ', 'import ', 'from ', '=', '(', ')', '[', ']',
            'self.', 'def ', 'async def ', 'await ', 'yield '
        ]
        
        content = '\n'.join(comment_lines).lower()
        return any(indicator in content for indicator in code_indicators)
    
    def analyze_directory(self, directory: str) -> List[Dict[str, Any]]:
        """Analyze all Python files in a directory."""
        all_commented_blocks = []
        
        for root, dirs, files in os.walk(directory):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if not self._should_exclude(os.path.join(root, d))]
            
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    
                    if self._should_exclude(filepath):
                        continue
                    
                    print(f"Analyzing commented code in: {filepath}")
                    blocks = self.analyze_file(filepath)
                    all_commented_blocks.extend(blocks)
        
        return all_commented_blocks
    
    def generate_report(self, commented_blocks: List[Dict[str, Any]]) -> str:
        """Generate a report of commented code blocks."""
        if not commented_blocks:
            return "No commented code blocks found."
        
        report = []
        report.append("=" * 80)
        report.append("COMMENTED CODE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Total commented code blocks found: {len(commented_blocks)}")
        report.append("")
        
        # Group by type
        by_type = {}
        for block in commented_blocks:
            block_type = block['type']
            if block_type not in by_type:
                by_type[block_type] = []
            by_type[block_type].append(block)
        
        for block_type, blocks in by_type.items():
            report.append(f"\n{block_type.replace('_', ' ').title()} ({len(blocks)} found):")
            report.append("-" * 50)
            
            for block in blocks:
                report.append(f"\nFile: {block['file']}")
                report.append(f"Line: {block['line']}")
                report.append(f"Code: {block['code']}")
                report.append("")
        
        return "\n".join(report)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze commented code blocks')
    parser.add_argument('directory', help='Directory to analyze')
    parser.add_argument('--exclusions', help='Exclusions file path')
    parser.add_argument('--output', help='Output report to file')
    
    args = parser.parse_args()
    
    analyzer = CommentedCodeAnalyzer(args.exclusions)
    commented_blocks = analyzer.analyze_directory(args.directory)
    
    report = analyzer.generate_report(commented_blocks)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == '__main__':
    main()