# Enhanced Code Quality Validation

This directory now includes enhanced validation tools that check for proper function arguments and data access patterns.

## New Validators

### 1. Enhanced Validator (`enhanced_validator.py`)

The enhanced validator provides comprehensive checks for:

#### Function Argument Validation
- **Missing required arguments**: Detects when required parameters are not provided
- **Too many arguments**: Identifies when too many positional arguments are passed
- **Unknown keyword arguments**: Finds invalid keyword arguments
- **Parameter type validation**: Checks argument types against function signatures

#### Data Access Validation
- **Unsafe attribute access**: Detects `obj.attr` without null checks
- **Unsafe dictionary access**: Identifies `dict[key]` without existence checks
- **Unsafe list/array access**: Finds `list[index]` without bounds checks
- **Missing null/None checks**: Highlights potential NoneType errors

### 2. Integrated Validator (`integrated_validator.py`)

Combines the enhanced validator with the existing function validator to provide:
- All checks from `function_validator.py` (undefined functions, missing await, etc.)
- All checks from `enhanced_validator.py` (argument validation, data access)
- Unified reporting in multiple formats (JSON, text, markdown)

## Usage

### Command Line Usage

#### Enhanced Validator Only
```bash
python code_quality/enhanced_validator.py --project-root /path/to/project --output report.txt
```

Options:
- `--project-root`: Project directory to analyze (default: current directory)
- `--output`: Output file for the report
- `--exclude`: Patterns to exclude (e.g., `--exclude "test_*" "__pycache__"`)
- `--json`: Output JSON format instead of text

#### Integrated Validator (Recommended)
```bash
python code_quality/integrated_validator.py --project-root /path/to/project --output-dir reports/
```

Options:
- `--project-root`: Project directory to analyze
- `--output-dir`: Directory for output reports (creates JSON, text, and markdown)
- `--exclude`: Patterns to exclude
- `--verbose`: Enable verbose logging

### Programmatic Usage

```python
from code_quality.enhanced_validator import EnhancedValidator
from code_quality.integrated_validator import IntegratedValidator

# Enhanced validator only
validator = EnhancedValidator(project_root='.')
report = validator.validate_project()

# Integrated validator (recommended)
integrated = IntegratedValidator(project_root='.')
report_files = integrated.generate_report()
```

## Example Issues Detected

### Function Argument Issues
```python
# Missing required argument
result = process_data(100)  # Error: Missing 'threshold' argument

# Too many arguments
result = calculate(1, 2, 3, 4)  # Error: Too many arguments

# Wrong keyword argument
result = process(data, normalized=True)  # Error: Unknown keyword 'normalized'
```

### Data Access Issues
```python
# Unsafe dictionary access
value = config["key"]  # Warning: Use config.get("key") instead

# Unsafe attribute access
data = response.data  # Warning: Check if response is not None first

# Unsafe list access
first = items[0]  # Warning: Check if items is not empty first
```

## Running Examples

To see the validators in action with example code:

```bash
python code_quality/example_validation_usage.py
```

This will:
1. Run the enhanced validator on the current project
2. Run the integrated validator
3. Create test files with various issues
4. Validate the test files to show detected issues

## Integration with Existing Tools

The enhanced validator integrates seamlessly with the existing code quality framework:
- Works with the same file exclusion patterns
- Follows the same reporting structure
- Can be run independently or as part of the integrated validator
- Compatible with the existing CI/CD pipelines

## Best Practices

1. **Use the Integrated Validator**: For comprehensive analysis, use `integrated_validator.py`
2. **Regular Scans**: Run validation as part of your development workflow
3. **Fix High-Priority Issues**: Focus on errors before warnings
4. **Safe Data Access**: Always validate data existence before access
5. **Type Hints**: Use type annotations to improve validation accuracy

## Output Formats

### JSON Report
Structured data suitable for programmatic processing or CI/CD integration.

### Text Summary
Human-readable summary with issue counts and top problems.

### Markdown Report
Detailed report with:
- Summary statistics
- Issues grouped by file
- Severity indicators (🔴 Error, 🟡 Warning, ℹ️ Info)
- Suggestions for fixes

## Extending the Validators

To add new validation rules:

1. **For function arguments**: Modify `ArgumentAndAccessValidator` in `enhanced_validator.py`
2. **For data access**: Add patterns to `_validate_attribute_access` or `_validate_subscript_access`
3. **For new issue types**: Update the `ValidationIssue` dataclass and statistics tracking

## Performance

The validators are optimized for large codebases:
- Single-pass AST analysis where possible
- Efficient pattern matching
- Minimal memory footprint
- Parallel file processing capability (can be added)

## Troubleshooting

### Common Issues

1. **Syntax Errors**: Files with syntax errors are skipped with a warning
2. **Import Errors**: External dependencies are not validated
3. **Dynamic Code**: Runtime-generated code cannot be statically analyzed

### Debug Mode

Enable verbose logging for detailed analysis:
```bash
python code_quality/integrated_validator.py --verbose
```

## Future Enhancements

Planned improvements:
- [ ] Type inference for better validation
- [ ] Custom rule configuration
- [ ] IDE integration
- [ ] Automatic fix suggestions
- [ ] Performance profiling
- [ ] Cross-file data flow analysis