#!/usr/bin/env python3
"""
Enhance existing files with entry point tprints where missing.
"""

import re
from pathlib import Path


def add_entry_tprints(file_path: Path) -> bool:
    """Add entry tprints to functions that don't have them."""
    print(f"Enhancing {file_path.name}...")

    try:
        lines = file_path.read_text().split('\n')
        new_lines = []
        i = 0
        changes = 0

        # Extract class name
        class_name = None
        for line in lines:
            if match := re.match(r'^class\s+(\w+)', line):
                class_name = match.group(1)
                break

        while i < len(lines):
            line = lines[i]
            new_lines.append(line)

            # Match function definitions
            func_match = re.match(r'(\s*)(async\s+)?def\s+(\w+)\s*\(', line)
            if func_match:
                indent = func_match.group(1)
                func_name = func_match.group(3)

                # Skip private methods except __init__
                if func_name.startswith('_') and func_name != '__init__':
                    i += 1
                    continue

                # Skip property getters/setters
                if i > 0 and '@property' in lines[i-1]:
                    i += 1
                    continue

                # Look ahead to see if there's already a tprint
                j = i + 1
                has_tprint = False

                # Skip until we find the function body (after docstring)
                while j < len(lines) and j < i + 20:  # Check next 20 lines
                    if '"""' in lines[j] or "'''" in lines[j]:
                        # Skip docstring
                        j += 1
                        if not (lines[j-1].strip().endswith('"""') or lines[j-1].strip().endswith("'''")):
                            while j < len(lines) and not (lines[j].strip().endswith('"""') or lines[j].strip().endswith("'''")):
                                j += 1
                        j += 1
                        continue

                    # Check if there's already a tprint
                    if 'tprint' in lines[j] and func_name in lines[j]:
                        has_tprint = True
                        break

                    # If we hit actual code (not blank lines or comments), stop
                    stripped = lines[j].strip()
                    if stripped and not stripped.startswith('#') and not stripped.startswith('"""') and not stripped.startswith("'''"):
                        break

                    j += 1

                # If no tprint found, add one
                if not has_tprint:
                    # Find where to insert (after docstring)
                    insert_pos = i + 1

                    # Skip docstring
                    if insert_pos < len(lines) and ('"""' in lines[insert_pos] or "'''" in lines[insert_pos]):
                        if not (lines[insert_pos].strip().endswith('"""') or lines[insert_pos].strip().endswith("'''")):
                            while insert_pos < len(lines) and not (lines[insert_pos].strip().endswith('"""') or lines[insert_pos].strip().endswith("'''")):
                                insert_pos += 1
                        insert_pos += 1

                    method_name = f"{class_name}.{func_name}" if class_name else func_name
                    tprint_line = f'{indent}    tprint(f"🚀 {method_name}: Entered", "INFO")'

                    # Insert the tprint
                    new_lines = new_lines[:insert_pos] + [tprint_line, ''] + new_lines[insert_pos:]
                    i = insert_pos + 2  # Skip past our insertion
                    changes += 1
                    continue

            i += 1

        if changes > 0:
            file_path.write_text('\n'.join(new_lines))
            print(f"✅ Added {changes} entry tprints to {file_path.name}")
            return True
        else:
            print(f"ℹ️  No changes needed for {file_path.name}")
            return False

    except Exception as e:
        print(f"❌ Error enhancing {file_path}: {e}")
        return False


def main():
    """Main function."""
    base_dir = Path("/home/user/Ares/src/trading/execution")

    files_to_enhance = [
        "paper_trader.py",
        "paper_trading_integration.py",
        "trading_orchestrator.py",
    ]

    for filename in files_to_enhance:
        file_path = base_dir / filename
        if file_path.exists():
            add_entry_tprints(file_path)
        else:
            print(f"⚠️  File not found: {file_path}")


if __name__ == "__main__":
    main()
