#!/usr/bin/env python3
"""
Code Style Analysis Script
Analyzes Python files for style violations and formatting issues.
"""

import os
import re
from collections import Counter, defaultdict
from pathlib import Path


class StyleAnalyzer:
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.style_issues = defaultdict(list)
        self.issue_counts = Counter()

    def analyze_style(self):
        """Analyze code style issues."""
        print("🎨 Analyzing code style issues...")

        python_files = self._find_python_files()

        for file_path in python_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                self._analyze_file_style(file_path, content)

            except Exception:
                # Skip files with read errors
                pass

        self._generate_style_report()

    def _find_python_files(self):
        """Find all Python files."""
        python_files = []
        exclude_patterns = [
            "__pycache__", ".git", "venv", "env", "node_modules",
            ".pytest_cache", "code_quality_env",
        ]

        for root, dirs, files in os.walk(self.repo_path):
            dirs[:] = [d for d in dirs if d not in exclude_patterns]

            for file in files:
                if file.endswith(".py"):
                    file_path = Path(root) / file
                    python_files.append(file_path)

        return python_files

    def _analyze_file_style(self, file_path: Path, content: str):
        """Analyze style issues in a single file."""
        lines = content.splitlines()

        for line_num, line in enumerate(lines, 1):
            # Check line length
            if len(line) > 88:
                self._add_issue(file_path, "line_length", line_num, f"Line too long ({len(line)} characters)")

            # Check trailing whitespace
            if line.rstrip() != line:
                self._add_issue(file_path, "trailing_whitespace", line_num, "Trailing whitespace")

            # Check mixed tabs and spaces
            if "\t" in line and " " in line:
                self._add_issue(file_path, "mixed_tabs_spaces", line_num, "Mixed tabs and spaces")

            # Check for bare except
            if re.match(r"^\s*except\s*:", line):
                self._add_issue(file_path, "bare_except", line_num, "Bare except clause")

            # Check for unused imports (simple heuristic)
            if line.strip().startswith("import ") or line.strip().startswith("from "):
                if "#" in line and "unused" in line.lower():
                    self._add_issue(file_path, "unused_import", line_num, "Potentially unused import")

            # Check for missing docstrings (simple heuristic)
            if re.match(r"^\s*def\s+\w+\s*\(", line):
                # Check if next non-empty line is a docstring
                next_line_idx = line_num
                while next_line_idx < len(lines) and not lines[next_line_idx - 1].strip():
                    next_line_idx += 1

                if next_line_idx < len(lines):
                    next_line = lines[next_line_idx - 1].strip()
                    if not (next_line.startswith(('"""', "'''"))):
                        self._add_issue(file_path, "missing_docstring", line_num, "Function missing docstring")

        # Check file-level issues
        if not content.strip():
            self._add_issue(file_path, "empty_file", 1, "Empty file")

        if not content.endswith("\n"):
            self._add_issue(file_path, "missing_newline", len(lines), "File doesn't end with newline")

    def _add_issue(self, file_path: Path, issue_type: str, line_num: int, message: str):
        """Add a style issue to the collection."""
        self.style_issues[issue_type].append({
            "file": str(file_path),
            "line": line_num,
            "message": message,
        })
        self.issue_counts[issue_type] += 1

    def _generate_style_report(self):
        """Generate a style analysis report."""
        print("\n🎨 Style Analysis Results:")
        print("=" * 50)

        total_issues = sum(self.issue_counts.values())
        print(f"Total style issues found: {total_issues}")
        print()

        # Show issue counts by type
        for issue_type, count in self.issue_counts.most_common():
            print(f"{issue_type}: {count} issues")

        print("\n" + "=" * 50)

        # Show detailed issues for top categories
        top_issues = self.issue_counts.most_common(5)
        for issue_type, count in top_issues:
            if self.style_issues[issue_type]:
                print(f"\n📋 {issue_type.upper()} ({count} issues):")
                for issue in self.style_issues[issue_type][:5]:  # Show first 5
                    print(f"  {issue['file']}:{issue['line']} - {issue['message']}")
                if count > 5:
                    print(f"  ... and {count - 5} more issues")

        # Save detailed report
        self._save_style_report()

    def _save_style_report(self):
        """Save detailed style report to file."""
        report_lines = ["# Code Style Analysis Report", ""]

        report_lines.append(f"**Total Issues**: {sum(self.issue_counts.values())}")
        report_lines.append("")

        for issue_type, count in self.issue_counts.most_common():
            report_lines.append(f"## {issue_type.replace('_', ' ').title()} ({count} issues)")
            report_lines.append("")

            for issue in self.style_issues[issue_type]:
                report_lines.append(f"- `{issue['file']}:{issue['line']}` - {issue['message']}")
            report_lines.append("")

        report_content = "\n".join(report_lines)

        with open("style_analysis_report.md", "w") as f:
            f.write(report_content)

        print("\n📄 Detailed style report saved to: style_analysis_report.md")

def main():
    """Main function."""
    analyzer = StyleAnalyzer()
    analyzer.analyze_style()

if __name__ == "__main__":
    main()
