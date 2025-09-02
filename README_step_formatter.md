# Step Formatter - Automatic Step Number Standardization

This tool automatically detects and formats step mentions in your codebase, converting `step01`, `step02`, etc. to `step01`, `step02`, etc. for consistency.

## 🎯 What It Does

The step formatter automatically:
- **Detects** mentions of `step01`, `step02`, ..., `step09` in file contents and filenames
- **Converts** them to `step01`, `step02`, ..., `step09` 
- **Preserves** existing double-digit steps like `step10`, `step11`, etc.
- **Works** on all text-based files (Python, Markdown, JSON, YAML, etc.)
- **Processes** files recursively through subdirectories

## 🚀 Quick Start

### 1. See What Would Change (Dry Run)
```bash
python3 format_steps.py
```
This shows you exactly what would be changed without making any modifications.

### 2. Apply the Changes
```bash
python3 format_steps.py --apply
```
This applies all the formatting changes to your files.

### 3. Apply with Backup Files
```bash
python3 format_steps.py --apply --backup
```
This creates backup files before making changes, so you can always revert if needed.

## 📁 Files Included

- **`step_formatter.py`** - The main formatter script (comprehensive)
- **`format_steps.py`** - Simple wrapper script (recommended for most users)
- **`demo_step_formatting.py`** - Demonstration of what gets formatted
- **`README_step_formatter.md`** - This documentation

## 🔧 Advanced Usage

### Direct Usage of Main Script
```bash
# Dry run on current directory
python3 step_formatter.py --dry-run --recursive .

# Apply changes with backups
python3 step_formatter.py --backup --recursive .

# Process specific directory
python3 step_formatter.py --recursive /path/to/directory

# Process single file
python3 step_formatter.py filename.py
```

### Command Line Options
- `--dry-run` - Show what would change without making changes
- `--backup` - Create backup files before making changes
- `--recursive` - Process subdirectories recursively
- `path` - Target file or directory (default: current directory)

## 📝 Examples

### What Gets Formatted
```
✅ step01  → step01
✅ step02  → step02
✅ step03  → step03
✅ step04  → step04
✅ step05  → step05
✅ step06  → step06
✅ step07  → step07
✅ step08  → step08
✅ step09  → step09
```

### What Doesn't Get Formatted
```
❌ step10 → step10 (already double digit)
❌ step11 → step11 (already double digit)
❌ step12 → step12 (already double digit)
❌ step0  → step0  (not in range 1-9)
```

## 🎯 Regex Pattern

The formatter uses this regex pattern:
```regex
\bstep([1-9])\b
```

This matches:
- Word boundaries (`\b`)
- Literal 'step'
- Single digit 1-9 (`[1-9]`)
- Word boundaries (`\b`)

## 📊 Supported File Types

The formatter processes these file extensions:
- **Code**: `.py`, `.js`, `.ts`, `.jsx`, `.tsx`, `.css`, `.scss`, `.sql`
- **Config**: `.yaml`, `.yml`, `.json`, `.toml`, `.ini`, `.cfg`
- **Docs**: `.md`, `.txt`, `.rst`, `.html`
- **Data**: `.csv`, `.xml`
- **Shell**: `.sh`, `.bash`, `.zsh`, `.fish`, `.ps1`, `.bat`, `.cmd`
- **Logs**: `.log`

## 🛡️ Safety Features

- **Dry Run Mode**: Always test first with `--dry-run`
- **Backup Option**: Create backup files with `--backup`
- **File Size Limits**: Skips files larger than 10MB
- **Binary File Detection**: Only processes text-based files
- **Error Handling**: Gracefully handles file access issues

## 📈 Performance

- **Fast**: Processes thousands of files quickly
- **Memory Efficient**: Processes files one at a time
- **Recursive**: Handles deep directory structures
- **Selective**: Only processes relevant file types

## 🔍 Testing

Run the demonstration to see examples:
```bash
python3 demo_step_formatting.py
```

## 📋 Current Status

Based on the dry run, the formatter found:
- **Files processed**: 5,503
- **Content changes**: 525 step mentions
- **Filename changes**: 0 (all filenames already properly formatted)

## 🚨 Important Notes

1. **Always test first** with `--dry-run` to see what will change
2. **Use backups** (`--backup`) for production codebases
3. **Review changes** after formatting to ensure nothing unexpected happened
4. **Version control** - commit your changes before running the formatter

## 🆘 Troubleshooting

### Common Issues

**"No step mentions found"**
- The formatter only processes single-digit steps (1-9)
- Double-digit steps (10+) are intentionally left unchanged

**"Permission denied"**
- Ensure you have read/write access to the target files
- Check file permissions and ownership

**"File too large"**
- Files larger than 10MB are skipped for memory safety
- Process large files individually if needed

### Getting Help

```bash
python3 format_steps.py --help
python3 step_formatter.py --help
```

## 🤝 Contributing

The step formatter is designed to be:
- **Safe**: Always testable with dry runs
- **Efficient**: Fast processing of large codebases  
- **Flexible**: Multiple usage patterns and options
- **Reliable**: Comprehensive error handling and validation

## 📄 License

This tool is part of your codebase and follows the same licensing terms.