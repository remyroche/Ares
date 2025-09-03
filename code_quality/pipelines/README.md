# Code Quality Pipelines

This directory contains orchestration pipelines that coordinate multiple code quality tools to provide comprehensive analysis and fixes.

## Available Pipelines

### Unified Pipelines

These pipelines run all code quality tools in a coordinated manner:

#### 1. **pipeline_unified_enhanced.py** - Enhanced Unified Pipeline
The most comprehensive pipeline with full unified reporting capabilities.

```bash
python pipeline_unified_enhanced.py --project-root /workspace/src
```

**Features:**
- Runs all code quality tools
- Generates unified reports with per-file and per-directory analysis
- Creates both JSON and Markdown reports
- Provides detailed console output
- Uses the ReportAggregator for comprehensive reporting

#### 2. **pipeline_unified_integrated.py** - Integrated Pipeline
Direct import version for better performance.

```bash
python pipeline_unified_integrated.py --project-root /workspace/src
```

**Features:**
- Directly imports code quality modules
- Better performance than subprocess approach
- Tighter integration between tools
- Suitable for development environments

#### 3. **pipeline_unified_standalone.py** - Standalone Pipeline
Subprocess-based pipeline for maximum isolation.

```bash
python pipeline_unified_standalone.py --project-root /workspace/src
```

**Features:**
- Uses subprocess calls (no direct imports)
- Avoids import conflicts
- Configurable timeouts
- Can run individual tools or categories

**Options:**
- `--categories syntax_imports async_types` - Run specific categories only
- `--tool syntax_fixer` - Run a single tool
- `--timeout 600` - Set timeout per tool (default: 300s)

### Category-Based Pipelines

These pipelines focus on specific categories of code quality issues:

#### 1. **pipeline_syntax_imports.py** - Syntax and Import Pipeline
Handles syntax errors, import management, and circular dependencies.

```bash
python pipeline_syntax_imports.py --project-root /workspace/src
```

**Runs:**
- Advanced syntax fixes
- Import fixes and management
- Circular import detection

**Options:**
- `--syntax-only` - Run only syntax fixes
- `--imports-only` - Run only import fixes
- `--circular-only` - Run only circular import detection

#### 2. **pipeline_syntax_imports_enhanced.py** - Enhanced Syntax/Import Pipeline
Same as above but with unified reporting capabilities.

```bash
python pipeline_syntax_imports_enhanced.py --project-root /workspace/src
```

**Additional Features:**
- Unified report generation (JSON + Markdown)
- Per-file and per-directory analysis
- Comprehensive issue tracking

#### 3. **pipeline_async_types.py** - Async and Type Hints Pipeline
Focuses on async/await patterns and type annotations.

```bash
python pipeline_async_types.py --project-root /workspace/src
```

**Runs:**
- Async/await pattern fixes
- Type hint enhancements

**Options:**
- `--async-only` - Run only async fixes
- `--types-only` - Run only type hint enhancements

#### 4. **pipeline_analysis.py** - Code Analysis Pipeline
Performs deep code analysis without making changes.

```bash
python pipeline_analysis.py --project-root /workspace/src
```

**Runs:**
- Function validation
- Code interaction mapping
- Comprehensive code review

**Options:**
- `--validation-only` - Run only function validation
- `--interactions-only` - Run only interaction mapping
- `--review-only` - Run only comprehensive review

## Pipeline Architecture

### Import Structure
All pipelines are configured to import tools from the parent directories:
- Individual tools from `code_quality/scripts/`
- Analyzers from `code_quality/analyzers/`
- Utils from `code_quality/utils/`
- Core modules from `code_quality/`

### Report Generation
Pipelines generate reports in `code_quality/reports/` with timestamps:
- Individual tool reports: `{tool_name}_{YYYYMMDD_HHMMSS}.json`
- Pipeline reports: `{pipeline_name}_{YYYYMMDD_HHMMSS}.json`
- Unified reports: `unified_report_{YYYYMMDD_HHMMSS}.{json|md}`

## Usage Examples

### Full Code Quality Check
```bash
cd /workspace/code_quality/pipelines
python pipeline_unified_enhanced.py --project-root /workspace/src
```

### Quick Syntax and Import Fixes
```bash
python pipeline_syntax_imports_enhanced.py --project-root /workspace/src
```

### Async Pattern Analysis Only
```bash
python pipeline_async_types.py --async-only --project-root /workspace/src
```

### Code Analysis Without Changes
```bash
python pipeline_analysis.py --project-root /workspace/src
```

### Running with Custom Timeout
```bash
python pipeline_unified_standalone.py --timeout 600 --project-root /workspace/src
```

## Best Practices

1. **For Production**: Use `pipeline_unified_standalone.py` to avoid import conflicts
2. **For Development**: Use `pipeline_unified_integrated.py` for faster execution
3. **For Reporting**: Use `pipeline_unified_enhanced.py` for comprehensive reports
4. **Order Matters**: Always run syntax fixes before other tools
5. **Incremental Fixes**: Use category pipelines for targeted improvements

## Pipeline Selection Guide

| Use Case | Recommended Pipeline |
|----------|---------------------|
| Full project scan with detailed reports | `pipeline_unified_enhanced.py` |
| Quick development checks | `pipeline_unified_integrated.py` |
| Production/CI environments | `pipeline_unified_standalone.py` |
| Fix syntax/import issues only | `pipeline_syntax_imports_enhanced.py` |
| Fix async patterns only | `pipeline_async_types.py` |
| Analysis without modifications | `pipeline_analysis.py` |

## Output

All pipelines generate:
1. **Console Output**: Real-time progress and summary
2. **JSON Reports**: Detailed machine-readable results
3. **Markdown Reports**: Human-readable summaries (enhanced pipelines)
4. **Individual Tool Reports**: Specific results from each tool

Reports are saved in `/workspace/code_quality/reports/` with timestamps for tracking improvements over time.