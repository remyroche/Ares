#!/usr/bin/env python3
from src.utils.tprint import tprint

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
    tprint("No interaction analysis data found. Please run the interaction analysis first.")
    data = {}

tprint("CODE INTERACTION MAPPING SUMMARY")
tprint("=" * 60)
tprint()

# Overall statistics
tprint("OVERALL STATISTICS")
tprint("-" * 30)
if data and 'summary' in data:
    tprint(f"Files analyzed: {data['summary']['files_processed']}")
    tprint(f"Total issues: {data['summary']['total_issues']}")
    tprint(f"Undefined functions: {data['summary']['undefined_functions']}")
    tprint(f"Missing await calls: {data['summary']['missing_await']}")
else:
    tprint("No data available")
tprint()

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
tprint("KEY CODE INTERACTIONS")
tprint("-" * 30)
tprint()

# Module-level interactions
tprint("MODULES WITH MOST ISSUES:")
for module, count in sorted(module_issues.items(), key=lambda x: x[1], reverse=True)[:10]:
    tprint(f"  {module}: {count} issues")
tprint()

# Most undefined functions
tprint("TOP 15 UNDEFINED FUNCTIONS:")
for func, count in sorted(undefined_funcs.items(), key=lambda x: x[1], reverse=True)[:15]:
    tprint(f"  {func}: {count} occurrences")
tprint()

# Async patterns
tprint("ASYNC FUNCTIONS MISSING AWAIT:")
for func, count in sorted(async_issues.items(), key=lambda x: x[1], reverse=True)[:10]:
    tprint(f"  {func}: {count} occurrences")
tprint()

# Files with most issues
tprint("FILES WITH MOST ISSUES:")
for file, count in sorted(files_with_issues.items(), key=lambda x: x[1], reverse=True)[:10]:
    tprint(f"  {Path(file).name}: {count} issues")
tprint()

# Interaction patterns
tprint("INTERACTION PATTERNS DETECTED:")
tprint("-" * 30)

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
    tprint(f"  {pattern}: {count} occurrences")

tprint()
tprint("RECOMMENDATIONS:")
tprint("-" * 30)
tprint("1. Add missing imports for common operations (datetime, pandas)")
tprint("2. Ensure all async functions are properly awaited")
tprint("3. Review module dependencies and circular imports")
tprint("4. Add type hints to clarify expected interfaces")
tprint("5. Consider creating utility modules for common undefined functions")


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
                tprint("No interaction analysis data found. Please run the interaction analysis first.")
                data = {}
        else:
            data = {}
        
        # This would contain the summary generation logic
        return data
