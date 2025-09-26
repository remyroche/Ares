#!/usr/bin/env python3
"""
Script to fix silent failures in the market_analysis directory by replacing bare except clauses
with proper error handling using tprint logging.
"""

import os
import re
import glob
from pathlib import Path

def fix_bare_except_clauses():
    """Fix all bare except clauses in the market_analysis directory."""

    # Find all Python files in the market_analysis directory
    market_analysis_dir = "/workspace/src/training/steps/market_analysis"

    # Check if tprint import is already present
    tprint_import_pattern = r'from src\.utils\.tprint import.*'

    files_fixed = 0
    exceptions_fixed = 0

    for root, dirs, files in os.walk(market_analysis_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Skip if file already has tprint import
                    if re.search(tprint_import_pattern, content):
                        tprint_available = True
                    else:
                        tprint_available = False

                    # Find all bare except clauses - simplified pattern
                    bare_except_pattern = r'\bexcept:\s*\n'

                    # Find context around bare except clauses
                    matches = list(re.finditer(bare_except_pattern, content))

                    # Filter out non-bare except clauses
                    bare_matches = []
                    for match in matches:
                        start = match.start()
                        # Check if this is a bare except (not followed by specific exception type)
                        context_before = content[max(0, start-50):start]
                        if not re.search(r'except\s+(Exception|ImportError|ValueError|TypeError|RuntimeError|OSError|IOError|KeyError|IndexError|AttributeError|NameError|ZeroDivisionError|FileNotFoundError)\s+as', context_before + 'except:'):
                            bare_matches.append(match)

                    if bare_matches:
                        print(f"🔍 Found {len(bare_matches)} bare except clauses in {file_path}")

                        # Add tprint import if not present
                        if not tprint_available:
                            # Add tprint import after existing imports
                            import_pattern = r'(^import.*|^from.*import.*)'
                            lines = content.split('\n')

                            # Find the last import line
                            last_import_idx = -1
                            for i, line in enumerate(lines):
                                if re.match(import_pattern, line):
                                    last_import_idx = i

                            if last_import_idx >= 0:
                                # Add tprint import after the last import
                                tprint_import = '''

# Import tprint for comprehensive logging
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)'''
                                lines.insert(last_import_idx + 1, tprint_import)
                                content = '\n'.join(lines)
                                print(f"✅ Added tprint import to {file_path}")

                        # Fix each bare except clause
                        for match in matches:
                            start = match.start()
                            end = match.end()

                            # Get context around the except clause
                            context_start = max(0, start - 100)
                            context_end = min(len(content), end + 100)
                            context = content[context_start:context_end]

                            # Find the exact except clause with more context
                            except_start = context.find('except:')
                            if except_start >= 0:
                                # Get the line with the except clause
                                line_start = context[:except_start].rfind('\n') + 1
                                line_end = context.find('\n', except_start)
                                if line_end == -1:
                                    line_end = len(context)

                                except_line = context[line_start:line_end].strip()

                                # Replace bare except with proper exception handling
                                if 'pass' in context[except_start:except_start+50]:
                                    # If it's just pass, use debug logging
                                    replacement = except_line.replace('except:', 'except Exception as e:') + '\n' + ' ' * (except_start - line_start) + '                    tprint_debug(f"🔍 Failed operation: {e}")\n' + ' ' * (except_start - line_start) + '                    pass'
                                else:
                                    # If it has other code, use warning logging
                                    replacement = except_line.replace('except:', 'except Exception as e:') + '\n' + ' ' * (except_start - line_start) + '                    tprint_warning(f"⚠️ Operation failed: {e}")'

                                content = content.replace(except_line, replacement)

                        # Write the fixed content back
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)

                        files_fixed += 1
                        exceptions_fixed += len(bare_matches)
                        print(f"✅ Fixed {len(bare_matches)} bare except clauses in {file_path}")

                except Exception as e:
                    from src.utils.tprint import tprint_error
                    tprint_error(f"❌ Error processing {file_path}: {e}")

    print(f"\n🎉 Fixed {exceptions_fixed} bare except clauses in {files_fixed} files")

if __name__ == "__main__":
    fix_bare_except_clauses()