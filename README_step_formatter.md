# Step Formatter Script

A Python script that automatically detects and formats step mentions in both file contents and file names by adding leading zeros (e.g., `step01` → `step01`, `step02` → `step02`, etc.).

## Features

- **Content Processing**: Automatically detects and formats step mentions in file contents
- **Filename Processing**: Renames files that contain step mentions in their names
- **Smart Detection**: Only processes single-digit step numbers (1-9), leaving double-digit and higher unchanged
- **Multiple File Types**: Supports various text file formats including Python, Markdown, JSON, YAML, etc.
- **Safety Features**: Dry-run mode, backup creation, and comprehensive logging
- **Recursive Processing**: Can process entire directory trees

## What It Does

The script converts:
- `step01` → `step01`
- `step02` → `step02`
- `step03` → `step03`
- `step04` → `step04`
- `step05` → `step05`
- `step06` → `step06`
- `step07` → `step07`
- `step08` → `step08`
- `step09` → `step09`

**Important**: It does NOT change:
- `step0` (already has a leading zero)
- `step10`, `step11`, `step12`, etc. (already have two digits)
- `step01`, `step02`, etc. (already properly formatted)

## Installation

No additional dependencies required - uses only Python standard library modules:

- `os`
- `re` (regex)
- `shutil`
- `argparse`
- `pathlib`
- `logging`

## Usage

### Basic Usage

```bash
# Process current directory
python step_formatter.py

# Process specific directory
python step_formatter.py /path/to/directory

# Process specific file
python step_formatter.py /path/to/file.py
```

### Command Line Options

```bash
python step_formatter.py [OPTIONS] [PATH]

Options:
  --dry-run     Show what would be changed without making changes
  --backup      Create backup files before making changes
  --recursive   Process subdirectories recursively
  --help        Show help message

PATH: Directory or file to process (default: current directory)
```

### Examples

```bash
# Dry run to see what would be changed
python step_formatter.py --dry-run

# Process current directory with backups
python step_formatter.py --backup

# Process entire directory tree recursively
python step_formatter.py --recursive --backup /path/to/project

# Process specific file with backup
python step_formatter.py --backup my_script.py
```

## Testing

A test script is included to demonstrate the formatter:

```bash
# Create test files with step mentions
python test_step_formatter.py

# Run formatter on test files (dry run)
python step_formatter.py --dry-run

# Apply changes to test files
python step_formatter.py --backup

# Clean up test files
python test_step_formatter.py --cleanup
```

## How It Works

### 1. Content Processing
- Reads text files and searches for step mentions using regex pattern `\bstep([1-9])\b`
- Replaces matches with leading zeros
- Writes updated content back to files
- Creates backups if `--backup` flag is used

### 2. Filename Processing
- Checks if filenames contain step mentions
- Renames files to include leading zeros
- Creates backups if `--backup` flag is used

### 3. File Type Detection
The script processes these file extensions:
- **Code**: `.py`, `.js`, `.ts`, `.jsx`, `.tsx`, `.css`, `.scss`, `.sql`, `.sh`, `.bash`, `.zsh`, `.fish`, `.ps1`, `.bat`, `.cmd`
- **Data**: `.json`, `.yaml`, `.yml`, `.toml`, `.ini`, `.cfg`, `.csv`, `.xml`
- **Documentation**: `.md`, `.txt`, `.rst`, `.html`, `.log`

### 4. Safety Features
- **Size Limit**: Skips files larger than 10MB to prevent memory issues
- **Hidden Files**: Skips files starting with `.`
- **Backup Creation**: Creates `.backup` files before making changes
- **Dry Run Mode**: Shows what would be changed without making changes
- **Error Handling**: Gracefully handles file access errors and continues processing

## Output

The script provides detailed logging of all operations:

```
2024-01-15 10:30:00 - INFO - Starting step formatter in LIVE mode
2024-01-15 10:30:00 - INFO - Backup mode enabled - backup files will be created
2024-01-15 10:30:01 - INFO - Created backup: /path/to/file.py.backup
2024-01-15 10:30:01 - INFO - Updated /path/to/file.py: 3 step mentions formatted
2024-01-15 10:30:01 - INFO - Would rename: test_step1_script.py -> test_step01_script.py
2024-01-15 10:30:01 - INFO - ==================================================
2024-01-15 10:30:01 - INFO - PROCESSING COMPLETE
2024-01-15 10:30:01 - INFO - ==================================================
2024-01-15 10:30:01 - INFO - Files processed: 15
2024-01-15 10:30:01 - INFO - Content changes: 12
2024-01-15 10:30:01 - INFO - Filename changes: 3
2024-01-15 10:30:01 - INFO - Total changes: 15
```

## Use Cases

### 1. Code Standardization
Standardize step naming conventions across a codebase:
```python
# Before
def step1_initialize():
    pass

def step2_process():
    pass

# After
def step01_initialize():
    pass

def step02_process():
    pass
```

### 2. Documentation Updates
Update documentation files to use consistent step numbering:
```markdown
# Before
1. step01: Initialize
2. step02: Process
3. step03: Report

# After
1. step01: Initialize
2. step02: Process
3. step03: Report
```

### 3. Configuration Files
Standardize step IDs in configuration files:
```json
// Before
{
  "steps": ["step01", "step02", "step03"]
}

// After
{
  "steps": ["step01", "step02", "step03"]
}
```

### 4. File Organization
Rename files to use consistent step numbering:
```
# Before
workflow_step1.py
workflow_step2.py
workflow_step3.py

# After
workflow_step01.py
workflow_step02.py
workflow_step03.py
```

## Best Practices

1. **Always use `--dry-run` first** to see what will be changed
2. **Use `--backup` flag** to create backup files before making changes
3. **Test on a small subset** before processing entire projects
4. **Review changes** after processing to ensure accuracy
5. **Version control** your files before running the formatter

## Limitations

- Only processes single-digit step numbers (1-9)
- Skips binary files and very large files (>10MB)
- Requires write permissions for files being modified
- May need to be run multiple times if new files are added

## Troubleshooting

### Common Issues

1. **Permission Denied**: Ensure you have write permissions for the target files
2. **File in Use**: Close any applications that might have the files open
3. **Large Files**: Very large files are automatically skipped to prevent memory issues
4. **Hidden Files**: Files starting with `.` are automatically skipped

### Getting Help

If you encounter issues:
1. Check the log output for error messages
2. Verify file permissions
3. Try running with `--dry-run` first
4. Check if files are locked by other processes

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the script.

## License

This script is provided as-is for educational and practical use.