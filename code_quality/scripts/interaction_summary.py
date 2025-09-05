#!/usr/bin/env python3
"""
Summarize code interactions from the analysis.
"""

import json
from collections import defaultdict
from pathlib import Path

# Load the JSON report
try:
    with open("/workspace/code_quality/interaction_analysis.json") as f:
        data = json.load(f)
except FileNotFoundError:
    print("No interaction analysis data found. Please run the interaction analysis first.")
    data = {}

print("CODE INTERACTION MAPPING SUMMARY")
print("=" * 60)
print()

# Overall statistics
print("OVERALL STATISTICS")
print("-" * 30)
if data and 'summary' in data:
    print(f"Files analyzed: {data['summary']['files_processed']}")
    print(f"Total issues: {data['summary']['total_issues']}")
    print(f"Undefined functions: {data['summary']['undefined_functions']}")
    print(f"Missing await calls: {data['summary']['missing_await']}")
else:
    print("No data available")
print()

# Analyze undefined functions
undefined_funcs = defaultdict(int)
async_issues = defaultdict(int)
files_with_issues = defaultdict(int)
module_issues = defaultdict(int)

for issue in data.get("issues", []):
    file_path = issue["file_path"]
    files_with_issues[file_path] += 1

    # Extract module from file path
    parts = Path(file_path).parts
    if len(parts) > 2:
        module = parts[2]  # e.g., /workspace/src/MODULE/...
        module_issues[module] += 1

    if issue["issue_type"] == "undefined_function":
        msg = issue["message"]
        if "Function '" in msg:
            func_name = msg.split("'")[1]
            undefined_funcs[func_name] += 1

    elif issue["issue_type"] == "missing_await":
        msg = issue["message"]
        if "Async function '" in msg:
            func_name = msg.split("'")[1]
            async_issues[func_name] += 1

# Key Interactions
print("KEY CODE INTERACTIONS")
print("-" * 30)
print()

# Module-level interactions
print("MODULES WITH MOST ISSUES:")
for module, count in sorted(module_issues.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {module}: {count} issues")
print()

# Most undefined functions
print("TOP 15 UNDEFINED FUNCTIONS:")
for func, count in sorted(undefined_funcs.items(), key=lambda x: x[1], reverse=True)[:15]:
    print(f"  {func}: {count} occurrences")
print()

# Async patterns
print("ASYNC FUNCTIONS MISSING AWAIT:")
for func, count in sorted(async_issues.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {func}: {count} occurrences")
print()

# Files with most issues
print("FILES WITH MOST ISSUES:")
for file, count in sorted(files_with_issues.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {Path(file).name}: {count} issues")
print()

# Interaction patterns
print("INTERACTION PATTERNS DETECTED:")
print("-" * 30)

# Common patterns in undefined functions
patterns = defaultdict(int)
for func in undefined_funcs:
    if func.startswith("get_"):
        patterns["Getter functions"] += undefined_funcs[func]
    elif func.startswith("set_"):
        patterns["Setter functions"] += undefined_funcs[func]
    elif func.startswith("_"):
        patterns["Private functions"] += undefined_funcs[func]
    elif func in ["append", "extend", "insert", "remove", "pop"]:
        patterns["List operations"] += undefined_funcs[func]
    elif func in ["keys", "values", "items", "get"]:
        patterns["Dict operations"] += undefined_funcs[func]
    elif func in ["now", "today", "strftime", "isoformat"]:
        patterns["DateTime operations"] += undefined_funcs[func]

for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
    print(f"  {pattern}: {count} occurrences")

print()
print("RECOMMENDATIONS:")
print("-" * 30)
print("1. Add missing imports for common operations (datetime, pandas)")
print("2. Ensure all async functions are properly awaited")
print("3. Review module dependencies and circular imports")
print("4. Add type hints to clarify expected interfaces")
print("5. Consider creating utility modules for common undefined functions")


class InteractionSummary:
    """Class wrapper for interaction summary functionality."""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
    
    def generate_summary(self, data_file: str = None):
        """Generate interaction summary from data file."""
        if data_file:
            try:
                with open(data_file) as f:
                    data = json.load(f)
            except FileNotFoundError:
                print("No interaction analysis data found. Please run the interaction analysis first.")
                data = {}
        else:
            data = {}
        
        # This would contain the summary generation logic
        return data
