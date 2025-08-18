#!/usr/bin/env python3
"""
Exception Handling Fix Script for Ares Trading Bot
This script helps identify and fix overly broad exception handling.
"""

import re
import ast
from pathlib import Path
from typing import List, Dict, Set, Tuple
import argparse


class ExceptionHandlerFixer:
    """Identifies and suggests fixes for overly broad exception handling."""
    
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        
        # Common specific exception types for different contexts
        self.trading_exceptions = {
            'ValueError', 'TypeError', 'KeyError', 'IndexError',
            'AttributeError', 'ZeroDivisionError', 'OverflowError'
        }
        
        self.data_processing_exceptions = {
            'ValueError', 'TypeError', 'KeyError', 'IndexError',
            'pd.errors.EmptyDataError', 'pd.errors.ParserError',
            'pd.errors.DtypeWarning', 'pd.errors.SettingWithCopyWarning'
        }
        
        self.network_exceptions = {
            'ConnectionError', 'TimeoutError', 'requests.exceptions.RequestException',
            'aiohttp.ClientError', 'aiohttp.ServerTimeoutError'
        }
        
        self.file_io_exceptions = {
            'FileNotFoundError', 'PermissionError', 'OSError',
            'IOError', 'UnicodeDecodeError', 'UnicodeEncodeError'
        }
        
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
    
    def analyze_file(self, file_path: Path) -> Dict[str, List[Tuple[int, str, str]]]:
        """Analyze a single Python file for broad exception handling."""
        issues = {
            'broad_exceptions': [],
            'except_exception': [],
            'bare_except': [],
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
            # Check for broad exception handling patterns
            for i, line in enumerate(lines, 1):
                line_stripped = line.strip()
                
                # Check for @handle_errors with broad exceptions
                if '@handle_errors' in line and 'Exception' in line:
                    issues['broad_exceptions'].append((i, line_stripped, 'handle_errors decorator'))
                
                # Check for except Exception:
                elif re.search(r'except\s+Exception\s*:', line):
                    issues['except_exception'].append((i, line_stripped, 'except Exception'))
                
                # Check for bare except:
                elif re.search(r'except\s*:', line):
                    issues['bare_except'].append((i, line_stripped, 'bare except'))
                    
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            
        return issues
    
    def suggest_specific_exceptions(self, context: str, line_content: str) -> str:
        """Suggest specific exception types based on context."""
        if 'trading' in context.lower() or 'position' in context.lower():
            return ', '.join(sorted(self.trading_exceptions))
        elif 'data' in context.lower() or 'pandas' in context.lower():
            return ', '.join(sorted(self.data_processing_exceptions))
        elif 'network' in context.lower() or 'http' in context.lower():
            return ', '.join(sorted(self.network_exceptions))
        elif 'file' in context.lower() or 'io' in context.lower():
            return ', '.join(sorted(self.file_io_exceptions))
        else:
            return ', '.join(sorted(self.trading_exceptions))  # Default to trading exceptions
    
    def generate_fix_suggestions(self, file_path: Path, issues: Dict[str, List[Tuple[int, str, str]]]) -> List[str]:
        """Generate specific fix suggestions for a file."""
        suggestions = []
        
        if not any(issues.values()):
            return suggestions
            
        suggestions.append(f"\n## Fixes for {file_path}")
        
        for issue_type, issue_list in issues.items():
            if issue_list:
                suggestions.append(f"\n### {issue_type.replace('_', ' ').title()}")
                
                for line_num, line_content, context in issue_list:
                    specific_exceptions = self.suggest_specific_exceptions(context, line_content)
                    
                    if issue_type == 'broad_exceptions':
                        # Fix @handle_errors decorator
                        if 'exceptions=(Exception,)' in line_content:
                            fixed_line = line_content.replace(
                                'exceptions=(Exception,)',
                                f'exceptions=({specific_exceptions},)'
                            )
                            suggestions.append(f"Line {line_num}: Replace broad exceptions")
                            suggestions.append(f"  Current: `{line_content}`")
                            suggestions.append(f"  Fixed:   `{fixed_line}`")
                    
                    elif issue_type == 'except_exception':
                        # Fix except Exception:
                        fixed_line = line_content.replace(
                            'except Exception:',
                            f'except ({specific_exceptions}):'
                        )
                        suggestions.append(f"Line {line_num}: Replace with specific exceptions")
                        suggestions.append(f"  Current: `{line_content}`")
                        suggestions.append(f"  Fixed:   `{fixed_line}`")
                    
                    elif issue_type == 'bare_except':
                        # Fix bare except:
                        fixed_line = line_content.replace(
                            'except:',
                            f'except ({specific_exceptions}):'
                        )
                        suggestions.append(f"Line {line_num}: Replace bare except")
                        suggestions.append(f"  Current: `{line_content}`")
                        suggestions.append(f"  Fixed:   `{fixed_line}`")
                    
                    suggestions.append("")
        
        return suggestions
    
    def create_automated_fix_script(self, output_file: str = "fix_exceptions_automated.py"):
        """Create a script to automatically fix some exception handling issues."""
        python_files = self.find_python_files()
        
        script_content = [
            "#!/usr/bin/env python3",
            '"""',
            "Automated exception handling fix script for Ares Trading Bot",
            "This script automatically fixes overly broad exception handling.",
            '"""',
            "",
            "import re",
            "from pathlib import Path",
            "",
            "# Common specific exception types",
            "TRADING_EXCEPTIONS = (",
            "    'ValueError', 'TypeError', 'KeyError', 'IndexError',",
            "    'AttributeError', 'ZeroDivisionError', 'OverflowError'",
            ")",
            "",
            "DATA_PROCESSING_EXCEPTIONS = (",
            "    'ValueError', 'TypeError', 'KeyError', 'IndexError',",
            "    'pd.errors.EmptyDataError', 'pd.errors.ParserError',",
            ")",
            "",
            "NETWORK_EXCEPTIONS = (",
            "    'ConnectionError', 'TimeoutError',",
            "    'requests.exceptions.RequestException',",
            "    'aiohttp.ClientError',",
            ")",
            "",
            "def fix_file_exceptions(file_path: str):",
            '    """Fix exception handling in a single file."""',
            "    try:",
            "        with open(file_path, 'r', encoding='utf-8') as f:",
            "            content = f.read()",
            "",
            "        original_content = content",
            "",
            "        # Fix @handle_errors with broad exceptions",
            "        content = re.sub(",
            "            r'@handle_errors\\(exceptions=\\(Exception,\\)\\)',",
            "            r'@handle_errors(exceptions=TRADING_EXCEPTIONS)',",
            "            content",
            "        )",
            "",
            "        # Fix except Exception:",
            "        content = re.sub(",
            "            r'except\\s+Exception\\s*:',",
            "            r'except TRADING_EXCEPTIONS:',",
            "            content",
            "        )",
            "",
            "        # Fix bare except:",
            "        content = re.sub(",
            "            r'except\\s*:',",
            "            r'except TRADING_EXCEPTIONS:',",
            "            content",
            "        )",
            "",
            "        # Only write if content changed",
            "        if content != original_content:",
            "            with open(file_path, 'w', encoding='utf-8') as f:",
            "                f.write(content)",
            "            print(f'Fixed exceptions in: {file_path}')",
            "",
            "    except Exception as e:",
            "        print(f'Error fixing {file_path}: {e}')",
            "",
            "def main():",
            '    """Main fix function."""',
            "    # List of files to fix (add more as needed)",
            "    files_to_fix = [",
        ]
        
        for file_path in python_files:
            script_content.append(f'        "{file_path}",')
        
        script_content.extend([
            "    ]",
            "",
            "    for file_path in files_to_fix:",
            "        fix_file_exceptions(file_path)",
            "",
            'if __name__ == "__main__":',
            "    main()",
        ])
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(script_content))
        
        print(f"Automated fix script created: {output_file}")
    
    def generate_report(self) -> str:
        """Generate a comprehensive exception handling analysis report."""
        python_files = self.find_python_files()
        
        total_issues = {
            'broad_exceptions': 0,
            'except_exception': 0,
            'bare_except': 0,
        }
        
        all_suggestions = []
        
        print(f"Analyzing {len(python_files)} Python files for exception handling issues...")
        
        for file_path in python_files:
            issues = self.analyze_file(file_path)
            if any(issues.values()):
                suggestions = self.generate_fix_suggestions(file_path, issues)
                all_suggestions.extend(suggestions)
                
                for issue_type, count in total_issues.items():
                    total_issues[issue_type] += len(issues[issue_type])
        
        # Generate report
        report = []
        report.append("# Exception Handling Analysis Report")
        report.append("")
        report.append("## Summary")
        report.append("")
        for issue_type, count in total_issues.items():
            report.append(f"- **{issue_type.replace('_', ' ').title()}**: {count}")
        report.append("")
        
        report.append("## Detailed Fix Suggestions")
        report.extend(all_suggestions)
        
        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Analyze and fix exception handling in Ares Trading Bot")
    parser.add_argument("--root-dir", default=".", help="Root directory to analyze")
    parser.add_argument("--output", default="exception_handling_report.md", help="Output report file")
    parser.add_argument("--create-fix-script", action="store_true", help="Create automated fix script")
    
    args = parser.parse_args()
    
    fixer = ExceptionHandlerFixer(args.root_dir)
    
    # Generate report
    report = fixer.generate_report()
    
    with open(args.output, 'w') as f:
        f.write(report)
    
    print(f"Report generated: {args.output}")
    
    if args.create_fix_script:
        fixer.create_automated_fix_script()


if __name__ == "__main__":
    import os
    main()