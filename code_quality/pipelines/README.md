# Code Quality Pipelines

This directory contains the main pipeline scripts for comprehensive code quality analysis. The pipelines are organized into focused, specialized tools plus one overall orchestrator.

## Pipeline Overview

### Focused Pipelines

#### 1. Complexity Pipeline (`complexity_pipeline.py`)
**Purpose**: Code complexity analysis with focus on cyclomatic complexity
**Features**:
- Cyclomatic complexity analysis
- Cognitive complexity analysis
- Maintainability metrics
- Code metrics analysis

**Usage**:
```bash
python pipelines/complexity_pipeline.py --analysis-type cyclomatic
python pipelines/complexity_pipeline.py --analysis-type cognitive
python pipelines/complexity_pipeline.py --analysis-type maintainability
python pipelines/complexity_pipeline.py --analysis-type metrics
```

#### 2. Dead Code Pipeline (`dead_code_pipeline.py`)
**Purpose**: Dead code detection and removal
**Features**:
- Enhanced dead code analysis
- Automatic dead code removal
- Unused imports detection
- Unreachable code detection

**Usage**:
```bash
python pipelines/dead_code_pipeline.py --analysis-type enhanced --auto-fix
```

#### 3. Auto Fixer Pipeline (`auto_fixer_pipeline.py`)
**Purpose**: Automatic code fixing with conservative approach
**Features**:
- Import fixes
- Syntax fixes
- Type hint fixes
- Conservative auto-fixing

**Usage**:
```bash
python pipelines/auto_fixer_pipeline.py --fix-type imports --conservative
```

#### 4. Interaction Mapping Pipeline (`interaction_mapping_pipeline.py`)
**Purpose**: Code interaction and dependency analysis
**Features**:
- Call graph analysis
- Dependency mapping
- Data flow analysis
- Architecture analysis

**Usage**:
```bash
python pipelines/interaction_mapping_pipeline.py --analysis-type call_graph
```

#### 5. Import-Free Analysis Pipeline (`import_free_analysis_pipeline.py`)
**Purpose**: Code analysis without import dependencies
**Features**:
- Syntax analysis
- Structure analysis
- Pattern detection
- AST-based analysis

**Usage**:
```bash
python pipelines/import_free_analysis_pipeline.py --analysis-type syntax
```

#### 6. Unified Enhanced Pipeline (`pipeline_unified_enhanced.py`)
**Purpose**: Comprehensive analysis with imports
**Features**:
- Complete code quality assessment
- Import analysis integration
- Advanced reporting
- Plugin system integration

**Usage**:
```bash
python pipelines/pipeline_unified_enhanced.py
```

### Master Orchestrator

#### Overall Pipeline (`overall_pipeline.py`)
**Purpose**: Master orchestrator for all pipelines
**Features**:
- Run all pipelines or specific subsets
- Comprehensive reporting
- Pipeline coordination
- Result aggregation

**Usage**:
```bash
python pipelines/overall_pipeline.py --all
python pipelines/overall_pipeline.py --pipelines complexity,dead_code,auto_fixer
```

## Quick Start

### Run All Pipelines
```bash
# Run all pipelines with default settings
python pipelines/overall_pipeline.py --all

# Run on specific project
python pipelines/overall_pipeline.py --project-root /path/to/project --all
```

### Run Specific Pipelines
```bash
# Run individual pipelines
python pipelines/complexity_pipeline.py --analysis-type cyclomatic
python pipelines/dead_code_pipeline.py --analysis-type enhanced --auto-fix
python pipelines/auto_fixer_pipeline.py --fix-type imports --conservative
python pipelines/interaction_mapping_pipeline.py --analysis-type call_graph
python pipelines/import_free_analysis_pipeline.py --analysis-type syntax
python pipelines/pipeline_unified_enhanced.py

# Run subset of pipelines
python pipelines/overall_pipeline.py --pipelines complexity,dead_code,auto_fixer
```

### List Available Pipelines
```bash
python pipelines/overall_pipeline.py --list
```

## Pipeline Dependencies

All pipelines are designed to be independent and can be run standalone:

```
overall_pipeline (orchestrator)
├── complexity_pipeline (independent)
├── dead_code_pipeline (independent)
├── auto_fixer_pipeline (independent)
├── interaction_mapping_pipeline (independent)
├── import_free_analysis_pipeline (independent)
└── pipeline_unified_enhanced (independent)
```

## Configuration

### Pipeline Configuration
Each pipeline can be configured through command-line arguments:

```bash
# Complexity analysis with custom settings
python pipelines/complexity_pipeline.py --analysis-type cyclomatic --project-root /path/to/project

# Dead code analysis with auto-fix
python pipelines/dead_code_pipeline.py --analysis-type enhanced --auto-fix

# Conservative auto-fixing
python pipelines/auto_fixer_pipeline.py --fix-type imports --conservative
```

### Global Configuration
The overall pipeline supports global configuration:

```bash
# Run all pipelines with custom project root
python pipelines/overall_pipeline.py --all --project-root /path/to/project

# Run specific pipelines with custom arguments
python pipelines/overall_pipeline.py --pipelines complexity,dead_code --custom-args complexity:--analysis-type,metrics
```

## Output and Reports

### Report Locations
- **Overall Pipeline Reports**: `overall_pipeline_results_*.json`
- **Individual Pipeline Reports**: Generated by each pipeline
- **Master Orchestrator Reports**: `master_pipeline_results_*.json`

### Report Formats
- **JSON**: Machine-readable results
- **Console**: Human-readable summaries
- **Logs**: Detailed execution information

## Examples

### Basic Usage
```bash
# Run complexity analysis
python pipelines/complexity_pipeline.py --analysis-type cyclomatic

# Run dead code analysis with auto-fix
python pipelines/dead_code_pipeline.py --analysis-type enhanced --auto-fix

# Run conservative import fixes
python pipelines/auto_fixer_pipeline.py --fix-type imports --conservative
```

### Advanced Usage
```bash
# Run all pipelines on specific project
python pipelines/overall_pipeline.py --all --project-root /path/to/project

# Run subset with custom arguments
python pipelines/overall_pipeline.py --pipelines complexity,dead_code --custom-args complexity:--analysis-type,metrics

# Run individual pipeline with verbose output
python pipelines/complexity_pipeline.py --analysis-type cyclomatic --verbose
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Permission Issues**: Check file permissions for output directories
3. **Pipeline Not Found**: Use `--list` to see available pipelines
4. **Custom Arguments**: Use proper format: `pipeline:arg1,arg2`

### Debug Mode
```bash
# Enable verbose output
python pipelines/overall_pipeline.py --all --verbose

# List available pipelines
python pipelines/overall_pipeline.py --list
```

## Contributing

### Adding New Pipelines
1. Create pipeline script in this directory
2. Add to `overall_pipeline.py` available_pipelines dictionary
3. Update this README
4. Test integration

### Pipeline Development
1. Follow the existing pipeline structure
2. Implement proper argument parsing
3. Add comprehensive error handling
4. Include detailed documentation

## Support

For issues and questions:
1. Check pipeline execution logs
2. Use `--list` to verify available pipelines
3. Review individual pipeline documentation
4. Check configuration settings

## License

This code quality pipeline system is part of the larger code quality framework and follows the same licensing terms.