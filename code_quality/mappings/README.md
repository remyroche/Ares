# Dead Code Analysis Tools

This directory contains comprehensive tools for analyzing dead code in Python repositories, with a focus on accurate dependency detection and safe function removal.

## 🚀 Quick Start

Run the comprehensive analysis on the `src/utils/` directory:

```bash
cd code_quality/mappings
python3 run_comprehensive_analysis.py --target-dir ../../src/utils
```

## 🛠️ Available Tools

### 1. Enhanced Function Usage Analyzer (`enhanced_function_usage_analyzer.py`)

**Purpose**: Analyzes function usage patterns and identifies truly unused functions using multiple parsing strategies.

**Features**:
- Works around syntax errors using regex fallback
- Detects function definitions, calls, and imports
- Categorizes files (test, config, export)
- Provides risk assessment for function removal

**Usage**:
```bash
python3 enhanced_function_usage_analyzer.py src/utils/
```

**Output**: `enhanced_function_usage_report.json`

### 2. Advanced Dependency Analyzer (`dependency_analyzer_v2.py`)

**Purpose**: Provides comprehensive dependency mapping and risk assessment for function removal.

**Features**:
- Builds dependency graphs using NetworkX
- Detects dependency cycles
- Assesses removal risk levels (safe, low, medium, high, critical)
- Works around syntax errors

**Usage**:
```bash
python3 dependency_analyzer_v2.py src/utils/
```

**Output**: `advanced_dependency_analysis.json`

### 3. Function Usage Validator (`function_usage_validator.py`)

**Purpose**: Validates function usage patterns and identifies truly unused functions with high accuracy.

**Features**:
- Multiple validation strategies
- Import/export statement analysis
- Usage pattern classification
- Risk level assessment

**Usage**:
```bash
python3 function_usage_validator.py src/utils/
```

**Output**: `function_usage_validation.json`

### 4. Comprehensive Analysis Runner (`run_comprehensive_analysis.py`)

**Purpose**: Orchestrates all analysis tools and generates consolidated reports.

**Features**:
- Runs all analysis tools in sequence
- Merges results from multiple tools
- Generates comprehensive final report
- Provides actionable recommendations

**Usage**:
```bash
python3 run_comprehensive_analysis.py --target-dir src/utils
```

**Output**: Multiple JSON reports + comprehensive text report

## 📊 Understanding the Results

### Risk Levels

- **Safe to Remove**: Functions with no dependencies, can be safely deleted
- **Low Risk**: Functions with minimal dependencies (e.g., only used in tests)
- **Medium Risk**: Functions with moderate dependencies, requires careful review
- **High Risk**: Functions with many dependencies, high chance of breaking code
- **Critical Risk**: Functions that are exported or have critical dependencies

### Validation Status

- **Truly Unused**: No usage detected anywhere
- **Used Function**: Has meaningful usage patterns
- **Imported Function**: Imported by other modules
- **Exported Function**: Listed in `__all__` or exported from `__init__.py`

## 🔍 Analysis Process

1. **Function Extraction**: Parse Python files to find function definitions
2. **Usage Detection**: Identify where functions are called or referenced
3. **Import Analysis**: Check import statements and module dependencies
4. **Export Analysis**: Identify functions exported from modules
5. **Dependency Mapping**: Build dependency graphs and detect cycles
6. **Risk Assessment**: Calculate removal risk based on dependencies
7. **Validation**: Cross-reference results from multiple tools

## ⚠️ Important Notes

### Syntax Error Handling

The tools are designed to work around syntax errors in Python files:
- Primary analysis uses AST parsing
- Fallback to regex-based extraction when AST fails
- Syntax errors are logged but don't prevent analysis

### False Positives

Some functions may appear unused but are actually used:
- Dynamic imports (`importlib.import_module`)
- String-based function calls
- Decorator-based usage
- Configuration-driven function selection

### Safe Removal Process

1. **Start with "Safe to Remove" functions**
2. **Test thoroughly after each removal**
3. **Remove functions in small batches**
4. **Keep version control backups**
5. **Review high-risk functions manually**

## 📁 Output Files

### Individual Tool Reports

- `enhanced_function_usage_report.json` - Function usage analysis
- `advanced_dependency_analysis.json` - Dependency mapping and risk assessment
- `function_usage_validation.json` - Usage validation results

### Consolidated Reports

- `comprehensive_dead_code_analysis_report.txt` - Human-readable summary
- Tool-specific JSON files with detailed analysis

## 🎯 Best Practices

### Before Running Analysis

1. **Ensure clean working directory**
2. **Have version control in place**
3. **Understand your codebase structure**
4. **Identify critical modules to preserve**

### During Analysis

1. **Review all tool outputs**
2. **Pay attention to risk assessments**
3. **Check for dependency cycles**
4. **Verify export statements**

### After Analysis

1. **Start with lowest risk functions**
2. **Test thoroughly after each change**
3. **Document removed functions**
4. **Monitor for regressions**

## 🔧 Customization

### Skipping Tools

```bash
python3 run_comprehensive_analysis.py --target-dir src/utils --skip-tools enhanced_function_usage advanced_dependency
```

### Custom Output Directory

```bash
python3 run_comprehensive_analysis.py --target-dir src/utils --output-dir ./reports
```

### Tool-Specific Options

Each tool can be run independently with its own options:

```bash
# Enhanced Function Usage Analyzer
python3 enhanced_function_usage_analyzer.py src/utils/

# Advanced Dependency Analyzer  
python3 dependency_analyzer_v2.py src/utils/

# Function Usage Validator
python3 function_usage_validator.py src/utils/
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all required packages are installed
2. **Syntax Errors**: Tools handle these automatically, check logs
3. **Memory Issues**: For large codebases, analyze in smaller chunks
4. **Permission Errors**: Ensure write access to output directory

### Debug Mode

Add debug output by modifying the tools or checking the detailed JSON reports.

## 📈 Performance

- **Small codebases** (< 100 files): Analysis completes in seconds
- **Medium codebases** (100-1000 files): Analysis completes in minutes  
- **Large codebases** (> 1000 files): Consider analyzing in chunks

## 🤝 Contributing

To improve these tools:

1. **Test with different codebases**
2. **Report false positives/negatives**
3. **Suggest new analysis strategies**
4. **Improve regex patterns for syntax error handling**

## 📚 Dependencies

Required Python packages:
- `ast` (built-in)
- `pathlib` (built-in)
- `collections` (built-in)
- `re` (built-in)
- `json` (built-in)
- `networkx` (for dependency graphs)

Install with:
```bash
pip install networkx
```

## 📞 Support

For issues or questions:
1. Check the tool output logs
2. Review the JSON reports for details
3. Verify file permissions and paths
4. Ensure Python version compatibility (3.7+)