#!/usr/bin/env python3
"""
Code Cleanup Script for Ares Trading Bot
This script helps identify and clean up various code quality issues.
"""

import os
import re
import ast
from pathlib import Path
from typing import List, Dict, Set, Tuple
import argparse


class CodeAnalyzer:
    """Analyzes Python code for various quality issues."""
    
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.debug_patterns = [
            r'print\(.*DEBUG.*\)',
            r'print\(.*🔍.*\)',
            r'print\(.*debug.*\)',
            r'print\(.*Debug.*\)',
        ]
        self.type_ignore_patterns = [
            r'# type: ignore',
            r'# noqa',
        ]
        self.broad_exception_patterns = [
            r'except Exception:',
            r'except Exception as e:',
            r'except:',
        ]
        
    def find_python_files(self) -> List[Path]:
        """Find all Python files in the project."""
        python_files = []
        for root, dirs, files in os.walk(self.root_dir):
            # Skip certain directories
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.pytest_cache', 'venv', '.venv'}]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
        return python_files
    
    def analyze_file(self, file_path: Path) -> Dict[str, List[Tuple[int, str]]]:
        """Analyze a single Python file for issues."""
        issues = {
            'debug_statements': [],
            'type_ignores': [],
            'broad_exceptions': [],
            'unused_imports': [],
            'todo_comments': [],
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
            # Check for debug statements
            for i, line in enumerate(lines, 1):
                for pattern in self.debug_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        issues['debug_statements'].append((i, line.strip()))
                        break
                
                # Check for type ignore comments
                for pattern in self.type_ignore_patterns:
                    if re.search(pattern, line):
                        issues['type_ignores'].append((i, line.strip()))
                        break
                
                # Check for broad exception handling
                for pattern in self.broad_exception_patterns:
                    if re.search(pattern, line):
                        issues['broad_exceptions'].append((i, line.strip()))
                        break
                
                # Check for TODO comments
                if re.search(r'TODO|FIXME|XXX|HACK|BUG', line, re.IGNORECASE):
                    issues['todo_comments'].append((i, line.strip()))
            
            # Try to parse AST for unused imports (basic check)
            try:
                tree = ast.parse(content)
                issues['unused_imports'] = self._find_potentially_unused_imports(tree, lines)
            except SyntaxError:
                pass  # Skip files with syntax errors
                
        except (IOError, OSError, UnicodeDecodeError) as e:
            print(f"Error analyzing {file_path}: {e}")
            
        return issues
    
    def _find_potentially_unused_imports(self, tree: ast.AST, lines: List[str]) -> List[Tuple[int, str]]:
        """Find potentially unused imports using AST analysis."""
        unused_imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname is None and not self._is_import_used(tree, alias.name):
                        unused_imports.append((node.lineno, f"import {alias.name}"))
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        if alias.name != '*' and not self._is_import_used(tree, alias.name):
                            unused_imports.append((node.lineno, f"from {node.module} import {alias.name}"))
        
        return unused_imports
    
    def _is_import_used(self, tree: ast.AST, import_name: str) -> bool:
        """Check if an import is used in the AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == import_name:
                return True
            elif isinstance(node, ast.Attribute) and hasattr(node, 'value'):
                if isinstance(node.value, ast.Name) and node.value.id == import_name:
                    return True
        return False
    
    def generate_report(self) -> str:
        """Generate a comprehensive analysis report."""
        python_files = self.find_python_files()
        
        total_issues = {
            'debug_statements': 0,
            'type_ignores': 0,
            'broad_exceptions': 0,
            'unused_imports': 0,
            'todo_comments': 0,
        }
        
        detailed_issues = {}
        
        print(f"Analyzing {len(python_files)} Python files...")
        
        for file_path in python_files:
            issues = self.analyze_file(file_path)
            if any(issues.values()):
                detailed_issues[str(file_path)] = issues
                for issue_type, count in total_issues.items():
                    total_issues[issue_type] += len(issues[issue_type])
        
        # Generate report
        report = []
        report.append("# Code Quality Analysis Report")
        report.append("")
        report.append("## Summary")
        report.append("")
        for issue_type, count in total_issues.items():
            report.append(f"- **{issue_type.replace('_', ' ').title()}**: {count}")
        report.append("")
        
        report.append("## Detailed Issues")
        report.append("")
        
        for file_path, issues in detailed_issues.items():
            if any(issues.values()):
                report.append(f"### {file_path}")
                report.append("")
                
                for issue_type, issue_list in issues.items():
                    if issue_list:
                        report.append(f"#### {issue_type.replace('_', ' ').title()}")
                        report.append("")
                        for line_num, line_content in issue_list[:10]:  # Limit to first 10
                            report.append(f"Line {line_num}: `{line_content}`")
                        if len(issue_list) > 10:
                            report.append(f"... and {len(issue_list) - 10} more")
                        report.append("")
        
        return "\n".join(report)
    
    def create_cleanup_script(self, output_file: str = "cleanup_actions.py"):
        """Create a script to automatically fix some issues."""
        python_files = self.find_python_files()
        
        script_content = [
            "#!/usr/bin/env python3",
            '"""',
            "Auto-generated cleanup script for Ares Trading Bot",
            "This script will automatically fix some code quality issues.",
            '"""',
            "",
            "import re",
            "from pathlib import Path",
            "",
            "def cleanup_file(file_path: str):",
            '    """Clean up a single file."""',
            "    try:",
            "        with open(file_path, 'r', encoding='utf-8') as f:",
            "            content = f.read()",
            "",
            "        original_content = content",
            "",
            "        # Remove debug print statements",
            "        debug_patterns = [",
            "            r'print\\(.*DEBUG.*\\)\\s*\\n',",
            "            r'print\\(.*🔍.*\\)\\s*\\n',",
            "            r'print\\(.*debug.*\\)\\s*\\n',",
            "        ]",
            "        for pattern in debug_patterns:",
            "            content = re.sub(pattern, '', content, flags=re.IGNORECASE)",
            "",
            "        # Remove type ignore comments (be careful with this)",
            "        content = re.sub(r'\\s*# type: ignore.*\\n', '\\n', content)",
            "",
            "        # Only write if content changed",
            "        if content != original_content:",
            "            with open(file_path, 'w', encoding='utf-8') as f:",
            "                f.write(content)",
            "            print(f'Cleaned up: {file_path}')",
            "",
            "    except Exception as e:",
            "        print(f'Error cleaning up {file_path}: {e}')",
            "",
            "def main():",
            '    """Main cleanup function."""',
            "    # List of files to clean up (add more as needed)",
            "    files_to_cleanup = [",
        ]
        
        for file_path in python_files:
            script_content.append(f'        "{file_path}",')
        
        script_content.extend([
            "    ]",
            "",
            "    for file_path in files_to_cleanup:",
            "        cleanup_file(file_path)",
            "",
            'if __name__ == "__main__":',
            "    main()",
        ])
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(script_content))
        
        print(f"Cleanup script created: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze code quality issues in Ares Trading Bot")
    parser.add_argument("--root-dir", default=".", help="Root directory to analyze")
    parser.add_argument("--output", default="code_quality_report.md", help="Output report file")
    parser.add_argument("--create-cleanup", action="store_true", help="Create cleanup script")
    
    args = parser.parse_args()
    
    analyzer = CodeAnalyzer(args.root_dir)
    
    # Generate report
    report = analyzer.generate_report()
    
    with open(args.output, 'w') as f:
        f.write(report)
    
    print(f"Report generated: {args.output}")
    
    if args.create_cleanup:
        analyzer.create_cleanup_script()


if __name__ == "__main__":
    main()