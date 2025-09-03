#!/usr/bin/env python3
"""
Extract and summarize code interactions from the function validation report.
"""

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def extract_interactions(json_file):
    """Extract interaction patterns from the validation report."""

    with open(json_file) as f:
        data = json.load(f)

    interactions = {
        "undefined_functions": defaultdict(list),
        "missing_await": [],
        "import_conflicts": defaultdict(list),
        "function_calls": defaultdict(set),
        "module_dependencies": defaultdict(set),
        "async_patterns": [],
        "missing_docstrings": [],
    }

    # Process issues
    for issue in data.get("issues", []):
        issue_type = issue["issue_type"]
        file_path = issue["file_path"]

        if issue_type == "undefined_function":
            # Extract function name from message
            msg = issue["message"]
            if "Function '" in msg:
                func_name = msg.split("'")[1]
                interactions["undefined_functions"][func_name].append({
                    "file": file_path,
                    "line": issue["line_number"],
                })

        elif issue_type == "missing_await":
            msg = issue["message"]
            if "Async function '" in msg:
                func_name = msg.split("'")[1]
                interactions["missing_await"].append({
                    "function": func_name,
                    "file": file_path,
                    "line": issue["line_number"],
                })

        elif issue_type == "import_conflict":
            interactions["import_conflicts"][file_path].append(issue["message"])

        elif issue_type == "missing_docstring":
            interactions["missing_docstrings"].append({
                "file": file_path,
                "line": issue["line_number"],
                "message": issue["message"],
            })

    # Analyze module dependencies from file paths
    for issue in data.get("issues", []):
        file_path = Path(issue["file_path"])
        if file_path.parts:
            module = ".".join(file_path.parts[:-1])
            interactions["module_dependencies"][module].add(str(file_path))

    return interactions, data


def generate_interaction_report(interactions, data):
    """Generate a comprehensive interaction report."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"/workspace/code_quality/code_interactions_report_{timestamp}.txt"

    with open(report_file, "w") as f:
        f.write("CODE INTERACTION ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        # Summary statistics
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total files analyzed: {data['summary']['files_processed']}\n")
        f.write(f"Total issues found: {data['summary']['total_issues']}\n")
        f.write(f"Undefined functions: {data['summary']['undefined_functions']}\n")
        f.write(f"Missing await calls: {data['summary']['missing_await']}\n\n")

        # Most common undefined functions
        f.write("TOP 20 UNDEFINED FUNCTIONS\n")
        f.write("-" * 40 + "\n")
        func_counts = Counter(interactions["undefined_functions"].keys())
        for func, count in func_counts.most_common(20):
            f.write(f"{func}: {len(interactions['undefined_functions'][func])} occurrences\n")
            # Show first 3 locations
            f.writelines(f"  - {loc['file']}:{loc['line']}\n" for loc in interactions["undefined_functions"][func][:3])
            if len(interactions["undefined_functions"][func]) > 3:
                f.write(f"  ... and {len(interactions['undefined_functions'][func]) - 3} more\n")
            f.write("\n")

        # Async/await issues
        f.write("\nASYNC/AWAIT ISSUES\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total missing await calls: {len(interactions['missing_await'])}\n\n")

        # Group by function
        async_funcs = defaultdict(list)
        for issue in interactions["missing_await"]:
            async_funcs[issue["function"]].append(issue)

        for func, issues in sorted(async_funcs.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
            f.write(f"{func}: {len(issues)} missing await calls\n")
            for issue in issues[:3]:
                f.write(f"  - {issue['file']}:{issue['line']}\n")
            if len(issues) > 3:
                f.write(f"  ... and {len(issues) - 3} more\n")
            f.write("\n")

        # Module interaction analysis
        f.write("\nMODULE INTERACTION ANALYSIS\n")
        f.write("-" * 40 + "\n")

        # Find modules with most files
        module_sizes = {mod: len(files) for mod, files in interactions["module_dependencies"].items()}
        f.writelines(f"{module}: {size} files\n" for module, size in sorted(module_sizes.items(), key=lambda x: x[1], reverse=True)[:10])

        # Import conflicts
        if interactions["import_conflicts"]:
            f.write("\n\nIMPORT CONFLICTS\n")
            f.write("-" * 40 + "\n")
            for file, conflicts in list(interactions["import_conflicts"].items())[:10]:
                f.write(f"\n{file}:\n")
                f.writelines(f"  - {conflict}\n" for conflict in conflicts[:3])

        # Key insights
        f.write("\n\nKEY INSIGHTS\n")
        f.write("-" * 40 + "\n")

        # Identify potential missing modules
        undefined_patterns = defaultdict(int)
        for func in interactions["undefined_functions"]:
            if "_" in func:
                prefix = func.split("_")[0]
                undefined_patterns[prefix] += 1

        f.write("Common undefined function prefixes (potential missing modules):\n")
        for prefix, count in sorted(undefined_patterns.items(), key=lambda x: x[1], reverse=True)[:10]:
            f.write(f"  - {prefix}_*: {count} functions\n")

        # Identify files with most issues
        file_issues = defaultdict(int)
        for issue in data.get("issues", []):
            file_issues[issue["file_path"]] += 1

        f.write("\nFiles with most issues:\n")
        for file, count in sorted(file_issues.items(), key=lambda x: x[1], reverse=True)[:10]:
            f.write(f"  - {file}: {count} issues\n")

    return report_file


def generate_visualization_script(interactions, output_file):
    """Generate a script to create visual representations of the interactions."""

    script_content = '''#!/usr/bin/env python3
"""
Visualization script for code interactions.
Run this to generate graphical representations.
"""

import json
from pathlib import Path

# Data extracted from the analysis
undefined_functions = {
'''

    # Add top undefined functions
    func_counts = Counter(interactions["undefined_functions"].keys())
    top_funcs = dict(func_counts.most_common(20))
    script_content += f"    {repr(top_funcs)}\n"

    script_content += """
}

missing_await_functions = {
"""

    # Add async functions missing await
    async_funcs = defaultdict(int)
    for issue in interactions["missing_await"]:
        async_funcs[issue["function"]] += 1
    script_content += f"    {repr(dict(async_funcs))}\n"

    script_content += '''
}

print("CODE INTERACTION VISUALIZATION DATA")
print("=" * 50)
print()
print("Top 10 Undefined Functions:")
for func, count in sorted(undefined_functions.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {func}: {count} occurrences")

print()
print("Top 10 Async Functions Missing Await:")
for func, count in sorted(missing_await_functions.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {func}: {count} occurrences")

print()
print("To create visual graphs:")
print("1. Use matplotlib/seaborn for bar charts of function counts")
print("2. Use networkx for dependency graphs")
print("3. Use graphviz for call flow diagrams")
print()
print("Example visualization code:")
print("""
import matplotlib.pyplot as plt

# Bar chart of undefined functions
funcs = list(undefined_functions.keys())[:10]
counts = [undefined_functions[f] for f in funcs]

plt.figure(figsize=(12, 6))
plt.bar(funcs, counts)
plt.xticks(rotation=45, ha='right')
plt.title('Top 10 Undefined Functions')
plt.xlabel('Function Name')
plt.ylabel('Occurrences')
plt.tight_layout()
plt.savefig('undefined_functions.png')
""")
'''

    with open(output_file, "w") as f:
        f.write(script_content)

    import os
    os.chmod(output_file, 0o755)


def main():
    # Load the validation report
    json_file = "/workspace/code_quality/interaction_analysis.json"

    print("EXTRACTING CODE INTERACTIONS")
    print("=" * 50)

    # Extract interactions
    interactions, data = extract_interactions(json_file)

    # Generate report
    report_file = generate_interaction_report(interactions, data)
    print(f"\nGenerated interaction report: {report_file}")

    # Generate visualization script
    viz_script = "/workspace/code_quality/visualize_interactions.py"
    generate_visualization_script(interactions, viz_script)
    print(f"Generated visualization script: {viz_script}")

    # Print summary
    print("\nINTERACTION SUMMARY:")
    print("-" * 30)
    print(f"Unique undefined functions: {len(interactions['undefined_functions'])}")
    print(f"Total undefined function calls: {sum(len(v) for v in interactions['undefined_functions'].values())}")
    print(f"Missing await calls: {len(interactions['missing_await'])}")
    print(f"Modules analyzed: {len(interactions['module_dependencies'])}")

    # Most problematic areas
    print("\nMOST PROBLEMATIC AREAS:")
    func_counts = Counter(interactions["undefined_functions"].keys())
    for func, _ in func_counts.most_common(5):
        count = len(interactions["undefined_functions"][func])
        print(f"  - {func}: {count} undefined references")


if __name__ == "__main__":
    main()
