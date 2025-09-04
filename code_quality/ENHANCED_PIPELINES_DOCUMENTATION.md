# Enhanced Pipelines Documentation

## Overview

This document describes the enhancements made to the `sequential_fixer.py` and `pipeline_unified_enhanced.py` pipelines with comprehensive static analysis and AST analysis capabilities.

## New Features

### 1. Static Analysis Integration

The pipelines now include comprehensive static analysis using industry-standard tools:

#### Tools Integrated:
- **Pylint**: Advanced code quality and style analysis
- **Flake8**: Style guide enforcement and error detection
- **MyPy**: Static type checking for Python code
- **Bandit**: Security vulnerability scanning

#### Features:
- Configurable tool settings per tool
- Comprehensive issue categorization (critical, warning, info)
- Security vulnerability detection
- Type checking with detailed error reporting
- Style guide enforcement with customizable rules

### 2. AST Analysis Integration

Advanced AST-based analysis using specialized tools:

#### Tools Integrated:
- **Astroid**: Advanced AST parsing and analysis
- **Jedi**: Code completion and static analysis
- **Custom AST Analysis**: Cyclomatic complexity, nesting levels, unused variables

#### Features:
- Cyclomatic complexity analysis
- Deep nesting detection
- Unused variable identification
- Code completion issue analysis
- Import resolution checking

## Enhanced Pipeline Architecture

### Sequential Fixer Pipeline

The enhanced sequential fixer now includes 8 comprehensive steps:

1. **Auto-fix syntax and style issues**
2. **Linter analysis and error reporting**
3. **AST parsing and compilation validation**
4. **Import analysis for conflicts and circular dependencies**
5. **Function signature analysis for compatibility**
6. **Comprehensive static analysis** (NEW)
7. **Advanced AST analysis** (NEW)
8. **Generate comprehensive report**

### Unified Enhanced Pipeline

The unified pipeline now includes additional analysis categories:

- **Syntax and Imports**: Basic fixes and import analysis
- **Async and Types**: Async/await fixes and type hints
- **Analysis**: Comprehensive code analysis including:
  - Function validation
  - Enhanced validation
  - Comprehensive review
  - Interaction mapping
  - Metrics analysis
  - Test coverage analysis
  - Code smell detection
  - Documentation analysis
  - Performance analysis
  - Configuration analysis
  - Data flow analysis
  - **Static analysis** (NEW)
  - **AST analysis** (NEW)

## Configuration System

### New Configuration Classes

#### StaticAnalysisConfig
```python
@dataclass
class StaticAnalysisConfig:
    enabled: bool = True
    tools: list[str] = ["pylint", "flake8", "mypy", "bandit"]
    pylint_config: dict[str, Any] = {...}
    flake8_config: dict[str, Any] = {...}
    mypy_config: dict[str, Any] = {...}
    bandit_config: dict[str, Any] = {...}
```

#### ASTAnalysisConfig
```python
@dataclass
class ASTAnalysisConfig:
    enabled: bool = True
    tools: list[str] = ["astroid", "jedi", "custom_ast"]
    astroid_config: dict[str, Any] = {...}
    jedi_config: dict[str, Any] = {...}
    custom_ast_config: dict[str, Any] = {...}
```

### Configuration Options

#### Pylint Configuration
- `max_line_length`: Maximum line length (default: 120)
- `disable`: List of disabled message IDs
- `max_args`: Maximum function arguments
- `max_locals`: Maximum local variables
- `max_returns`: Maximum return statements
- `max_branches`: Maximum branches
- `max_statements`: Maximum statements

#### Flake8 Configuration
- `max_line_length`: Maximum line length (default: 120)
- `extend_ignore`: Additional error codes to ignore

#### MyPy Configuration
- `ignore_missing_imports`: Ignore missing import errors
- `show_error_codes`: Show error codes in output
- `no_error_summary`: Disable error summary

#### Bandit Configuration
- `severity_level`: Minimum severity level to report
- `confidence_level`: Minimum confidence level to report

#### AST Analysis Configuration
- `max_function_length`: Maximum function length
- `max_nesting_level`: Maximum nesting level
- `check_unused_variables`: Enable unused variable checking
- `max_cyclomatic_complexity`: Maximum cyclomatic complexity
- `max_parameters`: Maximum function parameters

## Usage Examples

### Sequential Fixer

```python
from code_quality.fixers.sequential_fixer import SequentialFixer
from code_quality.core.config import get_default_config

# Initialize with default configuration
config = get_default_config()
fixer = SequentialFixer(config)

# Run enhanced pipeline
results = fixer.run_pipeline(
    target="/path/to/project",
    output_dir="/path/to/reports",
    create_backups=True,
    run_pre_commit=False
)

# Access new analysis results
static_results = results["step_results"]["static_analysis"]
ast_results = results["step_results"]["ast_analysis"]

print(f"Static analysis issues: {static_results['results']['summary']['total_issues_found']}")
print(f"AST analysis issues: {ast_results['results']['summary']['total_issues_found']}")
```

### Unified Enhanced Pipeline

```python
from code_quality.pipelines.pipeline_unified_enhanced import UnifiedEnhancedPipeline

# Initialize pipeline
pipeline = UnifiedEnhancedPipeline("/path/to/project")

# Run all analyses including new static and AST analysis
results = pipeline.run_all()

# Access new analysis results
static_analysis = results["analysis"]["static_analysis"]
ast_analysis = results["analysis"]["ast_analysis"]

print(f"Static analysis execution time: {static_analysis['execution_time']:.2f}s")
print(f"AST analysis execution time: {ast_analysis['execution_time']:.2f}s")
```

## New Analysis Modules

### StaticAnalysisAnalyzer

Located at: `code_quality/analyzers/static_analysis_analyzer.py`

**Key Methods:**
- `analyze_file(file_path)`: Analyze a single Python file
- `analyze_directory(directory_path)`: Analyze all Python files in a directory
- `_run_pylint(file_path)`: Run Pylint analysis
- `_run_flake8(file_path)`: Run Flake8 analysis
- `_run_mypy(file_path)`: Run MyPy analysis
- `_run_bandit(file_path)`: Run Bandit analysis

**Output Format:**
```python
{
    "file": "path/to/file.py",
    "tools": {
        "pylint": {"status": "success", "issues": [...]},
        "flake8": {"status": "success", "issues": [...]},
        "mypy": {"status": "success", "issues": [...]},
        "bandit": {"status": "success", "issues": [...]}
    },
    "summary": {
        "total_issues": 10,
        "critical_issues": 2,
        "warnings": 5,
        "info": 3,
        "security_issues": 1
    }
}
```

### ASTAnalysisAnalyzer

Located at: `code_quality/analyzers/ast_analysis_analyzer.py`

**Key Methods:**
- `analyze_file(file_path)`: Analyze a single Python file
- `analyze_directory(directory_path)`: Analyze all Python files in a directory
- `_run_astroid_analysis(file_path)`: Run Astroid analysis
- `_run_jedi_analysis(file_path)`: Run Jedi analysis
- `_run_custom_ast_analysis(file_path)`: Run custom AST analysis

**Output Format:**
```python
{
    "file": "path/to/file.py",
    "tools": {
        "astroid": {"status": "success", "issues": [...]},
        "jedi": {"status": "success", "issues": [...]},
        "custom_ast": {"status": "success", "complexity_issues": [...]}
    },
    "summary": {
        "total_issues": 8,
        "complexity_issues": 3,
        "refactoring_opportunities": 2,
        "code_completion_issues": 2,
        "ast_analysis_issues": 1
    }
}
```

## Enhanced Reporting

### New Metrics

The enhanced pipelines now track additional metrics:

#### Static Analysis Metrics
- `static_analysis_issues`: Total issues found by static analysis
- `static_analysis_critical`: Critical issues found
- `static_analysis_security`: Security vulnerabilities found

#### AST Analysis Metrics
- `ast_analysis_issues`: Total issues found by AST analysis
- `ast_analysis_complexity`: Complexity-related issues
- `ast_analysis_refactoring`: Refactoring opportunities

### Enhanced Recommendations

The pipelines now generate recommendations for:

1. **Static Analysis Issues**
   - Critical static analysis issues
   - Security vulnerabilities
   - Code quality improvements

2. **AST Analysis Issues**
   - High complexity functions
   - Refactoring opportunities
   - Code structure improvements

### Report Formats

#### JSON Reports
Comprehensive JSON reports with detailed analysis results for each tool and file.

#### HTML Reports
User-friendly HTML reports with visual indicators and categorized issues.

#### Terminal Output
Enhanced terminal output with color-coded severity levels and detailed summaries.

## Dependencies

### Required Packages

The enhanced pipelines require the following additional packages:

```bash
pip install pylint flake8 mypy bandit astroid jedi
```

### Optional Dependencies

Some tools are optional and the pipelines will gracefully handle their absence:

- **Astroid**: Advanced AST analysis (optional)
- **Jedi**: Code completion analysis (optional)

## Testing

### Test Suite

A comprehensive test suite is available at:
`code_quality/tests/test_enhanced_pipelines.py`

**Test Coverage:**
- StaticAnalysisAnalyzer functionality
- ASTAnalysisAnalyzer functionality
- Enhanced SequentialFixer integration
- Enhanced UnifiedEnhancedPipeline integration
- Configuration system integration

### Running Tests

```bash
cd /workspace/code_quality
python -m pytest tests/test_enhanced_pipelines.py -v
```

## Performance Considerations

### Optimization Features

1. **Parallel Processing**: Tools can be run in parallel where possible
2. **Caching**: Results are cached to avoid redundant analysis
3. **Selective Analysis**: Only analyze changed files when possible
4. **Timeout Protection**: Tools have timeout limits to prevent hanging

### Resource Usage

- **Memory**: Moderate increase due to AST parsing
- **CPU**: Higher usage during analysis phases
- **Disk**: Additional storage for detailed reports
- **Time**: Longer execution time due to comprehensive analysis

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   - Install required packages: `pip install pylint flake8 mypy bandit astroid jedi`
   - Check Python version compatibility

2. **Tool Failures**
   - Check tool-specific configuration
   - Verify file permissions
   - Review timeout settings

3. **Performance Issues**
   - Reduce analysis scope
   - Disable optional tools
   - Increase timeout limits

### Debug Mode

Enable debug mode for detailed logging:

```python
config = get_default_config()
config.reporting.verbose = True
```

## Future Enhancements

### Planned Features

1. **Incremental Analysis**: Only analyze changed files
2. **Custom Rules**: User-defined analysis rules
3. **Integration**: CI/CD pipeline integration
4. **Visualization**: Interactive issue visualization
5. **Auto-fixing**: Automatic issue resolution

### Extension Points

The architecture supports easy extension:

1. **New Analysis Tools**: Add new analyzers by implementing the analyzer interface
2. **Custom Metrics**: Define project-specific metrics
3. **Report Formats**: Add new output formats
4. **Integration**: Integrate with external tools and services

## Conclusion

The enhanced pipelines provide comprehensive code quality analysis with industry-standard tools and advanced AST-based analysis. The modular architecture allows for easy customization and extension while maintaining high performance and reliability.

For more information, refer to the individual analyzer documentation and configuration files.