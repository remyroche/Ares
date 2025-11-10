#!/usr/bin/env python3
"""
Script to add comprehensive tprints to trading execution files.

This script adds tprint statements to:
- Function entry points with key parameters
- All return statements
- Important state changes and decisions
"""

import re
import sys
from pathlib import Path


def add_tprint_import(content: str) -> str:
    """Add tprint import if not present."""
    if "from src.printing import tprint" in content:
        return content

    # Find the tprint utilities import block
    pattern = r"from src\.utils\.tprint import \("
    if re.search(pattern, content):
        # Add after the existing tprint utils import
        replacement = r"from src.utils.tprint import (\n    tprint_info, tprint_warning, tprint_error, tprint_success,\n    tprint_structured, LogLevel\n)\nfrom src.printing import tprint"
        content = re.sub(
            r"from src\.utils\.tprint import \(\s*tprint_info.*?\)",
            replacement,
            content,
            flags=re.DOTALL
        )

    return content


def add_tprint_to_function(content: str, class_name: str = None) -> str:
    """Add tprints to function definitions."""
    lines = content.split('\n')
    new_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]
        new_lines.append(line)

        # Match function definitions
        func_match = re.match(r'(\s*)(async\s+)?def\s+(\w+)\s*\(([^)]*)\)', line)
        if func_match:
            indent = func_match.group(1)
            is_async = func_match.group(2) is not None
            func_name = func_match.group(3)
            params = func_match.group(4)

            # Skip special methods like __repr__, __str__, etc (except __init__)
            if func_name.startswith('_') and func_name != '__init__' and not func_name.startswith('__'):
                i += 1
                continue

            # Find the docstring end
            i += 1
            while i < len(lines) and (lines[i].strip().startswith('"""') or lines[i].strip().startswith("'''")):
                new_lines.append(lines[i])
                if lines[i].strip().endswith('"""') or lines[i].strip().endswith("'''"):
                    break
                i += 1

            # Check if there's already a tprint
            if i + 1 < len(lines) and 'tprint' in lines[i + 1]:
                i += 1
                continue

            # Add tprint after docstring
            method_name = f"{class_name}.{func_name}" if class_name else func_name

            # Extract key parameters (skip self, cls)
            param_list = [p.strip().split(':')[0].split('=')[0].strip()
                         for p in params.split(',') if p.strip()]
            param_list = [p for p in param_list if p not in ['self', 'cls']]

            if param_list:
                param_str = ', '.join([f"{p}={{{p}}}" for p in param_list[:4]])  # First 4 params
                tprint_line = f'{indent}    tprint(f"🚀 {method_name}: {param_str}", "INFO")'
            else:
                tprint_line = f'{indent}    tprint(f"🚀 {method_name}: Starting", "INFO")'

            new_lines.append(tprint_line)
            new_lines.append('')

        i += 1

    return '\n'.join(new_lines)


def add_tprint_to_returns(content: str) -> str:
    """Add tprints before return statements."""
    lines = content.split('\n')
    new_lines = []

    for i, line in enumerate(lines):
        # Check if line contains a return statement
        if re.match(r'\s+return\s+', line):
            indent = re.match(r'(\s*)', line).group(1)

            # Check if previous line already has tprint
            if i > 0 and 'tprint' in lines[i-1]:
                new_lines.append(line)
                continue

            # Extract return value
            return_match = re.search(r'return\s+(.*)', line)
            if return_match:
                return_val = return_match.group(1).strip()
                if return_val in ['True', 'False', 'None', '[]', '{}', '""', "''", '0', '0.0']:
                    # Simple return value
                    tprint_line = f'{indent}tprint(f"✅ Returning {return_val}", "DEBUG")'
                else:
                    # Complex return value
                    tprint_line = f'{indent}tprint(f"✅ Returning result", "DEBUG")'

                new_lines.append(tprint_line)

        new_lines.append(line)

    return '\n'.join(new_lines)


def process_file(file_path: Path) -> bool:
    """Process a single file to add tprints."""
    print(f"Processing {file_path}...")

    try:
        content = file_path.read_text()

        # Extract class name if present
        class_match = re.search(r'^class\s+(\w+)', content, re.MULTILINE)
        class_name = class_match.group(1) if class_match else None

        # Add import
        content = add_tprint_import(content)

        # Add tprints to functions
        content = add_tprint_to_function(content, class_name)

        # Add tprints to returns
        # content = add_tprint_to_returns(content)  # Disabled for now to avoid too many tprints

        # Write back
        file_path.write_text(content)

        print(f"✅ Successfully processed {file_path}")
        return True

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return False


def main():
    """Main function."""
    base_dir = Path("/home/user/Ares/src/trading/execution")

    files_to_process = [
        "live_trader.py",
        "live_trading_scheduler.py",
        "partial_bar_nowcasting.py",
    ]

    success_count = 0
    for filename in files_to_process:
        file_path = base_dir / filename
        if file_path.exists():
            if process_file(file_path):
                success_count += 1
        else:
            print(f"⚠️  File not found: {file_path}")

    print(f"\n📊 Summary: Processed {success_count}/{len(files_to_process)} files successfully")


if __name__ == "__main__":
    main()
