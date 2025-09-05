# Code Quality Pipelines

This directory contains the main pipeline scripts for comprehensive code quality analysis. The pipelines are organized to provide different levels of analysis and fixing capabilities.

## Pipeline Overview

### Core Pipelines

#### 1. Unified Enhanced Pipeline (`unified_enhanced_pipeline.py`)
**Purpose**: The most comprehensive code quality analysis pipeline
**Features**:
- All analyzers, visualizers, and fix scripts
- Plugin system integration
- Complete code quality assessment
- Advanced reporting and visualization

**Usage**:
```bash
python unified_enhanced_pipeline.py --project-root /path/to/project
```

#### 2. Sequential Code Fixer (`sequential_code_fixer.py`)
**Purpose**: Sequential auto-fix pipeline for code issues
**Features**:
- Syntax fixing and style issues
- Linter analysis and error reporting
- AST parsing and compilation validation
- Import conflict analysis
- Comprehensive static analysis

**Usage**:
```bash
python sequential_code_fixer.py --target /path/to/code
```

#### 3. Code Interaction Mapper (`code_interaction_mapper.py`)
**Purpose**: Maps code interactions and dependencies
**Features**:
- Dependency analysis
- Call graph analysis
- Architecture analysis
- Import analysis
- Enhanced dead code analysis with cross-file checking

**Usage**:
```bash
python code_interaction_mapper.py --project-root /path/to/project
```

#### 4. Dead Code Analyzer (`dead_code_analyzer.py`)
**Purpose**: Detects and analyzes dead code
**Features**:
- Unused code detection
- Dead import identification
- Unreachable code detection
- Deprecated code analysis
- Vulture library integration

**Usage**:
```bash
python dead_code_analyzer.py --target /path/to/code
```

#### 5. Complexity CLI (`complexity_cli.py`)
**Purpose**: Code complexity analysis
**Features**:
- PyExamine, Radon, and Xenon integration
- Complexity metrics calculation
- Detailed complexity reporting
- Multiple output formats

**Usage**:
```bash
python complexity_cli.py analyze /path/to/code
```

#### 6. Enhanced Import Analysis (`enhanced_import_analysis.py`)
**Purpose**: Comprehensive import analysis
**Features**:
- Import dependency mapping
- Circular dependency detection
- Unused import identification
- Import conflict resolution
- Import optimization suggestions

**Usage**:
```bash
python enhanced_import_analysis.py --target /path/to/code
```

### Specialized Components

#### Enhanced Import Analyzer Plugin (`enhanced_import_analyzer_plugin.py`)
**Purpose**: Plugin for enhanced import analysis
**Features**:
- Plugin architecture integration
- Comprehensive import analysis
- Optimization suggestions
- Detailed reporting

#### Script Integration Manager (`script_integration_manager.py`)
**Purpose**: Manages integration of all scripts into pipelines
**Features**:
- Script discovery and categorization
- Integration status checking
- Pipeline organization planning
- Comprehensive reporting

#### Master Pipeline Orchestrator (`master_pipeline_orchestrator.py`)
**Purpose**: Orchestrates all pipelines
**Features**:
- Pipeline discovery and registration
- Dependency management
- Execution scheduling
- Result aggregation
- Configuration management

**Usage**:
```bash
python master_pipeline_orchestrator.py --project-root /path/to/project
```

## Pipeline Dependencies

The pipelines have the following dependency structure:

```
unified_enhanced_pipeline (no dependencies)
├── sequential_code_fixer
├── code_interaction_mapper
│   └── dead_code_analyzer
└── enhanced_import_analysis

complexity_cli (independent)
```

## Configuration

### Pipeline Configuration
Each pipeline can be configured through the master orchestrator:

```json
{
  "pipeline_configs": {
    "unified_enhanced_pipeline": {
      "enabled": true,
      "priority": 1,
      "timeout": 1800
    },
    "sequential_code_fixer": {
      "enabled": true,
      "priority": 2,
      "timeout": 1200
    }
  }
}
```

### Global Configuration
```json
{
  "parallel_execution": false,
  "max_parallel_pipelines": 4,
  "timeout_seconds": 3600,
  "retry_failed": true,
  "max_retries": 2,
  "output_formats": ["json", "html", "markdown"]
}
```

## Usage Examples

### Run All Pipelines
```bash
# Run all pipelines with default settings
python master_pipeline_orchestrator.py

# Run with custom project root
python master_pipeline_orchestrator.py --project-root /path/to/project

# Run with custom configuration
python master_pipeline_orchestrator.py --config custom_config.json
```

### Run Specific Pipelines
```bash
# Run only specific pipelines
python master_pipeline_orchestrator.py --pipelines unified_enhanced_pipeline,sequential_code_fixer

# Run individual pipeline
python unified_enhanced_pipeline.py --project-root /path/to/project
```

### Integration Analysis
```bash
# Analyze script integration status
python script_integration_manager.py --output integration_report.txt

# Generate comprehensive integration report
python script_integration_manager.py --verbose
```

## Output and Reports

### Report Locations
- **Master Reports**: `/workspace/code_quality/reports/master_pipeline_results_*.json`
- **Individual Pipeline Reports**: `/workspace/code_quality/reports/[pipeline_name]_*.json`
- **Integration Reports**: `/workspace/code_quality/reports/integration_report.txt`

### Report Formats
- **JSON**: Machine-readable results
- **HTML**: Interactive web reports
- **Markdown**: Human-readable documentation
- **Text**: Console-friendly summaries

## Plugin System

The pipelines support a comprehensive plugin system:

### Available Plugins
- **Production Plugins**: Syntax fixer, import fixer, dead code fixer, linter runner, security scanner
- **Code Quality Plugins**: Black, isort, autopep8, flake8, ruff, and more
- **Custom Plugins**: Enhanced import analyzer, script integration manager

### Plugin Integration
```python
from enhanced_import_analyzer_plugin import EnhancedImportAnalyzerPlugin

# Register plugin
plugin = EnhancedImportAnalyzerPlugin()
plugin_registry.register_plugin("enhanced_import_analyzer", plugin)
```

## Error Handling and Fallbacks

### Mock Implementations
- **Vulture Library**: Graceful fallback when not available
- **External Tools**: Try-catch blocks around tool imports
- **Configuration**: Default configurations when custom configs unavailable

### Fallback Mechanisms
- **Tool Availability**: Automatic detection and fallback
- **Analysis Degradation**: Simplified analysis when advanced tools unavailable
- **Error Recovery**: Retry mechanisms and error reporting

## Performance Considerations

### Optimization Features
- **Parallel Execution**: Experimental parallel pipeline execution
- **Caching**: Result caching for repeated analyses
- **Timeout Management**: Configurable timeouts for each pipeline
- **Resource Management**: Memory and CPU usage optimization

### Monitoring
- **Execution Time**: Detailed timing information
- **Resource Usage**: Memory and CPU monitoring
- **Progress Tracking**: Real-time progress updates
- **Error Reporting**: Comprehensive error logging

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Permission Issues**: Check file permissions for output directories
3. **Timeout Issues**: Increase timeout values in configuration
4. **Memory Issues**: Reduce parallel execution or increase system memory

### Debug Mode
```bash
# Enable verbose output
python master_pipeline_orchestrator.py --verbose

# Enable debug information
python script_integration_manager.py --verbose
```

## Contributing

### Adding New Pipelines
1. Create pipeline script in this directory
2. Add to pipeline registry in master orchestrator
3. Define dependencies and configuration
4. Update documentation

### Plugin Development
1. Inherit from base plugin class
2. Implement required methods
3. Add to plugin registry
4. Test integration

## Support

For issues and questions:
1. Check the integration report for script status
2. Review pipeline execution logs
3. Consult the mock implementation review
4. Check configuration settings

## License

This code quality pipeline system is part of the larger code quality framework and follows the same licensing terms.