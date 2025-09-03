# Code Quality Validation Tools

This directory contains comprehensive tools for validating code quality in Python projects, with a specific focus on function existence, parameter validation, and async/await usage.

## Tools Overview

### 1. Comprehensive Code Review (`comprehensive_code_review.py`)
A full-featured code quality analyzer that checks:
- **Function existence and import validation**
- **Parameter validation and type checking**
- **Async/await usage verification**
- **Code style and formatting**
- **Security vulnerabilities** (hardcoded secrets, SQL injection)
- **Performance issues** (magic numbers, unused variables)
- **Documentation quality** (missing docstrings)
- **Error handling patterns** (bare except clauses)

### 2. Function Validator (`function_validator.py`)
A focused tool specifically for function-related validation:
- **Function existence verification**
- **Import consistency checking**
- **Async/await pattern validation**
- **Function call analysis**
- **Parameter validation**

### 3. Runner Script (`run_validation.py`)
A convenient wrapper that runs both tools with easy configuration.

## Directory Structure

```
code_quality/
├── pipelines/         # Orchestration pipelines for running multiple tools
├── scripts/           # Individual code quality tools
├── analyzers/         # Code analysis modules
├── fixers/           # Code fixing modules
├── reports/          # Generated reports (timestamped)
├── utils/            # Utility modules (report aggregator, etc.)
└── core/             # Core functionality
```

### Pipelines

The `pipelines/` directory contains orchestration scripts that coordinate multiple tools:
- **Unified Pipelines**: Run all tools comprehensively
- **Category Pipelines**: Focus on specific types of issues (syntax, async, types)
- **Enhanced Pipelines**: Include unified reporting with per-file/directory analysis

See [pipelines/README.md](pipelines/README.md) for detailed pipeline documentation.

## Installation

The tools use only Python standard library modules, so no additional installation is required beyond Python 3.7+.

Optional dependencies (for enhanced functionality):
```bash
pip install astroid mypy bandit
```

## Usage

### Quick Start

Run comprehensive code quality checks using the unified pipeline:
```bash
cd /workspace/code_quality/pipelines
python pipeline_unified_enhanced.py --project-root /workspace/src
```

Or run the basic validation tools:
```bash
python code_quality/run_validation.py
```

### Command Line Options

#### Runner Script
```bash
python code_quality/run_validation.py [OPTIONS]

Options:
  --mode {comprehensive,function,both}  Validation mode to run (default: both)
  --project-root PATH                   Project root directory (default: current)
  --output-dir PATH                     Output directory for reports (default: ./reports)
  --verbose, -v                         Verbose output
```

#### Individual Tools

**Comprehensive Review:**
```bash
python code_quality/comprehensive_code_review.py --project-root /path/to/project --output report.json
```

**Function Validator:**
```bash
python code_quality/function_validator.py --project-root /path/to/project --output validation.json
```

### Examples

1. **Validate current project:**
   ```bash
   cd /path/to/your/project
   python code_quality/run_validation.py
   ```

2. **Run only function validation:**
   ```bash
   python code_quality/run_validation.py --mode function
   ```

3. **Custom output directory:**
   ```bash
   python code_quality/run_validation.py --output-dir ./my_reports
   ```

4. **Exclude specific patterns:**
   ```bash
   python code_quality/comprehensive_code_review.py --exclude "*/tests/*" "*/venv/*"
   ```

## What Gets Checked

### Function Existence
- ✅ Functions defined in the same file
- ✅ Functions imported from other modules
- ✅ Built-in Python functions
- ❌ Undefined functions (reported as errors)

### Async/Await Usage
- ✅ Async functions properly awaited
- ❌ Async functions called without await (reported as errors)
- ✅ Proper async function definitions

### Parameter Validation
- ✅ Function argument counts
- ✅ Default parameter values
- ⚠️ Functions with too many arguments (warnings)

### Import Consistency
- ✅ Import statement validation
- ⚠️ Potential naming conflicts
- ✅ Import path verification

### Code Style
- ✅ Line length limits (120 characters)
- ✅ Trailing whitespace
- ✅ File encoding (UTF-8)
- ✅ File ending with newline
- ✅ Naming conventions (snake_case for functions, PascalCase for classes)

### Security
- ❌ Hardcoded secrets in function calls
- ❌ Potential SQL injection vulnerabilities
- ⚠️ Bare except clauses

### Documentation
- ⚠️ Missing function docstrings
- ⚠️ Missing class docstrings

## Output

Each tool generates two types of reports:

### 1. JSON Report
Detailed machine-readable report with all issues, metadata, and statistics.

### 2. Text Summary
Human-readable summary grouped by issue type and severity.

### Report Structure

```json
{
  "summary": {
    "project_root": "/path/to/project",
    "files_processed": 42,
    "total_issues": 15,
    "errors": 3,
    "warnings": 10,
    "info": 2,
    "processing_time_seconds": 2.34
  },
  "issues": [
    {
      "file_path": "src/main.py",
      "line_number": 25,
      "issue_type": "missing_await",
      "severity": "error",
      "message": "Async function 'fetch_data' is called without await",
      "suggestion": "Add 'await' before the function call: await fetch_data(...)",
      "code_snippet": "result = fetch_data(url)"
    }
  ],
  "function_analysis": {
    "total_calls": 156,
    "total_definitions": 89,
    "async_functions": 23
  }
}
```

## Configuration

### Exclude Patterns
Default patterns that are automatically excluded:
- `*/__pycache__/*`
- `*/.git/*`
- `*/venv/*`
- `*/env/*`
- `*/node_modules/*`
- `*.pyc`, `*.pyo`, `*.pyd`

### Custom Exclusions
Add your own exclusion patterns:
```bash
python code_quality/comprehensive_code_review.py --exclude "*/tests/*" "*/docs/*"
```

## Integration

### CI/CD Pipeline
Add to your CI/CD pipeline:
```yaml
# GitHub Actions example
- name: Run Code Quality Validation
  run: |
    python code_quality/run_validation.py --mode both --output-dir ./reports
    # Fail if there are errors
    if grep -q '"severity": "error"' ./reports/*.json; then
      echo "Code quality validation failed!"
      exit 1
    fi
```

### Pre-commit Hooks
Add to your pre-commit configuration:
```yaml
repos:
  - repo: local
    hooks:
      - id: code-quality-check
        name: Code Quality Validation
        entry: python code_quality/function_validator.py
        language: system
        types: [python]
```

## Customization

### Adding New Checks
Extend the tools by subclassing the visitor classes:

```python
class CustomVisitor(FunctionValidatorVisitor):
    def visit_Call(self, node):
        # Add your custom logic here
        super().visit_Call(node)
```

### Custom Issue Types
Add new issue types by extending the issue classes:

```python
@dataclass
class CustomIssue(FunctionIssue):
    custom_field: str
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're running from the project root
2. **Permission Errors**: Check file permissions for the target directory
3. **Memory Issues**: For very large projects, consider running on subsets

### Performance Tips

- Use `--exclude` to skip unnecessary directories
- Run on specific subdirectories for focused analysis
- Use the focused function validator for quick checks

## Contributing

To add new validation rules:

1. Identify the AST node type to check
2. Add a new visitor method in the appropriate visitor class
3. Implement the validation logic
4. Add appropriate issue reporting
5. Update tests and documentation

## License

This code quality validation tool is part of the Ares Trading Bot project and follows the same licensing terms.