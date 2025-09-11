# Print to TPrint Migration Guide

## Overview

The `migrate_print_to_tprint.py` script automatically converts all `print` statements to `tprint` statements and adds the necessary import at the top of Python files. This ensures full backward compatibility while upgrading your code to use the enhanced timestamped printing functionality.

## Features

✅ **Automatic Conversion**
- Converts `print()` calls to `tprint()` calls
- Preserves all arguments and formatting
- Handles complex print statements with multiple arguments

✅ **Import Management**
- Automatically adds `from src.utils.tprint import tprint` at the top of files
- Detects existing imports to avoid duplicates
- Places imports in the correct location (after shebang/encoding)

✅ **Safety Features**
- Creates backup files before modification
- Dry-run mode to preview changes
- Comprehensive error handling

✅ **Batch Processing**
- Process single files or entire directories
- Recursive directory processing
- Detailed progress reporting

## Installation

The migration script is located at:
```
/workspace/migrate_print_to_tprint.py
```

Make it executable:
```bash
chmod +x migrate_print_to_tprint.py
```

## Usage

### Basic Syntax
```bash
python3 migrate_print_to_tprint.py [options] <file_or_directory>
```

### Options
- `--dry-run`: Show what would be changed without modifying files
- `--backup-dir <directory>`: Directory to store backup files
- `--help`: Show help message

## Examples

### 1. Migrate a Single File
```bash
# Preview changes (recommended first step)
python3 migrate_print_to_tprint.py --dry-run script.py

# Apply changes with backup
python3 migrate_print_to_tprint.py --backup-dir backups script.py

# Apply changes without backup
python3 migrate_print_to_tprint.py script.py
```

### 2. Migrate a Directory
```bash
# Preview changes for entire src/ directory
python3 migrate_print_to_tprint.py --dry-run src/

# Migrate entire directory with backups
python3 migrate_print_to_tprint.py --backup-dir backups src/
```

### 3. Batch Migration
```bash
# Migrate multiple files
python3 migrate_print_to_tprint.py --backup-dir backups file1.py file2.py file3.py

# Migrate entire project
python3 migrate_print_to_tprint.py --backup-dir project_backups .
```

## Before and After Examples

### Example 1: Simple Print Statements
**Before:**
```python
#!/usr/bin/env python3

def main():
    print("Hello, world!")
    print("Starting application...")
    print("Application completed!")
```

**After:**
```python
#!/usr/bin/env python3
from src.utils.tprint import tprint

def main():
    tprint("Hello, world!")
    tprint("Starting application...")
    tprint("Application completed!")
```

### Example 2: Complex Print Statements
**Before:**
```python
import os

def process_data():
    print("Processing data...")
    print("Debug info:", variable1, 42, [1, 2, 3])
    print(f"Result: {result}")
    print("Status:", "SUCCESS" if success else "FAILED")
```

**After:**
```python
from src.utils.tprint import tprint
import os

def process_data():
    tprint("Processing data...")
    tprint("Debug info:", variable1, 42, [1, 2, 3])
    tprint(f"Result: {result}")
    tprint("Status:", "SUCCESS" if success else "FAILED")
```

### Example 3: Print in Loops and Conditions
**Before:**
```python
def analyze_data():
    for i, item in enumerate(data):
        print(f"Processing item {i}: {item}")
        if item > threshold:
            print("Item exceeds threshold")
        else:
            print("Item within limits")
```

**After:**
```python
from src.utils.tprint import tprint

def analyze_data():
    for i, item in enumerate(data):
        tprint(f"Processing item {i}: {item}")
        if item > threshold:
            tprint("Item exceeds threshold")
        else:
            tprint("Item within limits")
```

## Migration Process

### Step 1: Preview Changes (Recommended)
```bash
python3 migrate_print_to_tprint.py --dry-run your_script.py
```

This shows you exactly what will be changed without modifying any files.

### Step 2: Create Backups
```bash
python3 migrate_print_to_tprint.py --backup-dir backups your_script.py
```

This creates backup files before making changes.

### Step 3: Verify Results
```bash
python3 your_script.py
```

Test that your script works correctly with the new tprint statements.

## Output Examples

### Dry Run Output
```
🔄 Print to TPrint Migration Tool
============================================================
🔍 DRY RUN MODE - No files will be modified

📄 Processing example_script.py
  🔍 [DRY RUN] Would modify example_script.py
     - Convert 11 print statements to tprint
     - Add tprint import statement

============================================================
MIGRATION SUMMARY
============================================================
Files processed: 1
Files modified: 1
Print statements converted: 11

🔍 This was a DRY RUN - no files were actually modified
Run without --dry-run to apply changes
```

### Actual Migration Output
```
🔄 Print to TPrint Migration Tool
============================================================
💾 Backups will be saved to: backups

📄 Processing example_script.py
  💾 Backup created: backups/example_script_backup_.py
  ✅ Modified example_script.py
     - Converted 11 print statements to tprint
     - Added tprint import statement

============================================================
MIGRATION SUMMARY
============================================================
Files processed: 1
Files modified: 1
Print statements converted: 11

✅ Migration completed successfully!
💾 Backups saved in: backups
```

## Benefits After Migration

### Before Migration
```python
print("User logged in")  # 2025-09-11 08:31:15.811 User logged in
```

### After Migration
```python
tprint("User logged in")  # [2025-09-11 08:31:15.811] User logged in
```

**Benefits:**
- ✅ **Consistent timestamps** on all output
- ✅ **Better debugging** with precise timing
- ✅ **Professional logging** appearance
- ✅ **Easy filtering** of timestamped output
- ✅ **Performance monitoring** capabilities

## Advanced Usage

### Custom Import Path
If your project structure is different, you can modify the import statement in the script:

```python
# In migrate_print_to_tprint.py, line 30:
self.import_statement = "from your.custom.path.tprint import tprint"
```

### Selective Migration
To migrate only specific files, use glob patterns:

```bash
# Migrate only test files
python3 migrate_print_to_tprint.py --dry-run tests/*.py

# Migrate only main scripts
python3 migrate_print_to_tprint.py --dry-run src/main*.py
```

### Integration with CI/CD
```bash
# In your build script
python3 migrate_print_to_tprint.py --backup-dir backups src/
if [ $? -eq 0 ]; then
    echo "Migration successful"
    python3 -m pytest tests/
else
    echo "Migration failed"
    exit 1
fi
```

## Troubleshooting

### Common Issues

1. **Import Error After Migration**
   ```bash
   ModuleNotFoundError: No module named 'src.utils.tprint'
   ```
   **Solution:** Ensure the tprint module is in your Python path or adjust the import statement.

2. **Backup Directory Permission Error**
   ```bash
   PermissionError: [Errno 13] Permission denied: 'backups'
   ```
   **Solution:** Create the backup directory manually or use a different location.

3. **No Changes Detected**
   ```bash
   No print statements found in file.py
   ```
   **Solution:** The file doesn't contain any print statements, so no migration is needed.

### Recovery

If something goes wrong, you can restore from backups:

```bash
# Restore from backup
cp backups/your_script_backup_.py your_script.py

# Or use git if you're using version control
git checkout your_script.py
```

## Best Practices

1. **Always run dry-run first** to preview changes
2. **Create backups** before migrating important files
3. **Test thoroughly** after migration
4. **Migrate incrementally** for large projects
5. **Use version control** as an additional safety net

## Integration with Existing Workflows

### Git Hooks
```bash
# Pre-commit hook to ensure all prints are migrated
#!/bin/bash
python3 migrate_print_to_tprint.py --dry-run .
if [ $? -ne 0 ]; then
    echo "Please migrate print statements to tprint"
    exit 1
fi
```

### IDE Integration
Configure your IDE to run the migration script on save or as a custom command.

### Makefile Integration
```makefile
migrate:
	python3 migrate_print_to_tprint.py --backup-dir backups src/

test-migration:
	python3 migrate_print_to_tprint.py --dry-run src/
```

## Conclusion

The migration script provides a safe, automated way to upgrade your codebase from basic `print` statements to the enhanced `tprint` functionality. With its dry-run mode, backup capabilities, and comprehensive error handling, you can confidently migrate your entire project while maintaining full backward compatibility.