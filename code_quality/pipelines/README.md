# Code Quality Pipelines

This directory contains the main pipeline scripts for comprehensive code quality analysis. The pipelines are organized into focused, specialized tools plus one overall orchestrator.

## 🚀 **New Features (Latest Updates)**

### ✨ **Enhanced Type Safety**
- **Comprehensive type hints** across all pipeline classes
- **Pydantic validation** for configuration management
- **Runtime type checking** with automatic validation

### 🔄 **Intelligent Caching System**
- **Result caching** to avoid redundant analysis
- **File-based cache** with automatic invalidation
- **Memory-efficient caching** with configurable limits

### 🏥 **Health Monitoring**
- **Pipeline health checks** with status monitoring
- **Resource usage tracking** (memory, execution time)
- **Plugin status monitoring** with detailed diagnostics

### 📊 **Advanced Configuration**
- **Pydantic-powered validation** with automatic error messages
- **Environment-specific configs** support
- **Schema validation** with detailed field descriptions

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

### Pipeline Configuration with Pydantic Validation
Each pipeline supports advanced configuration with automatic validation:

```python
from code_quality.pipelines.base_pipeline import PipelineConfig
from pathlib import Path

# Create validated configuration
config = PipelineConfig(
    project_root=Path("/path/to/project"),
    max_workers=8,
    timeout_per_tool=600,
    log_level="DEBUG",
    cache_enabled=True
)

# Configuration is automatically validated
print(f"Project root: {config.project_root}")
print(f"Max workers: {config.max_workers}")
```

### Command Line Configuration
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

## 🔄 **Caching System**

### Cache Configuration
```python
from code_quality.pipelines.complexity_pipeline import ComplexityPipeline

# Enable caching for faster repeated runs
pipeline = ComplexityPipeline(
    project_root="/path/to/project",
    cache_enabled=True
)

# Results are automatically cached
results = pipeline.run_cyclomatic_complexity_analysis()

# Clear cache when needed
pipeline.clear_cache()
```

### Cache Benefits
- **Faster execution** on repeated analysis
- **Memory efficient** with configurable limits
- **Persistent cache** survives pipeline restarts
- **Automatic invalidation** based on file changes

## 🏥 **Health Monitoring**

### Health Check API
```python
from code_quality.pipelines.complexity_pipeline import ComplexityPipeline

pipeline = ComplexityPipeline(project_root="/path/to/project")

# Get comprehensive health status
health = pipeline.health_check()

print(f"Status: {health['status']}")
print(f"Memory usage: {health['memory_usage_mb']} MB")
print(f"Plugins loaded: {health['plugins_loaded']}")
```

### Health Check Output
```json
{
  "status": "healthy",
  "timestamp": "20250106_143022",
  "project_root": "/path/to/project",
  "project_root_valid": true,
  "reports_dir_exists": true,
  "plugins_loaded": 3,
  "memory_usage_mb": 45.2,
  "execution_count": 5,
  "pipeline_type": "ComplexityPipeline"
}
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

## 📚 **API Reference**

### Base Pipeline Class
```python
from code_quality.pipelines.base_pipeline import BasePipeline, PipelineConfig

class BasePipeline:
    """Enhanced base class with comprehensive features."""

    def __init__(self, project_root: Optional[Path] = None,
                 config: Optional[PipelineConfig] = None,
                 enable_plugins: bool = True,
                 pipeline_name: str = "base") -> None:
        """Initialize pipeline with standardized setup."""

    def health_check(self) -> Dict[str, Any]:
        """Return comprehensive health status."""

    def clear_cache(self) -> None:
        """Clear all cached results."""

    def get_metrics(self) -> Dict[str, Any]:
        """Get execution metrics and statistics."""

    def get_execution_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get execution history with optional limit."""
```

### Pipeline Configuration
```python
@dataclass
class PipelineConfig:
    """Validated pipeline configuration."""
    project_root: Path
    max_workers: int = Field(default=4, ge=1, le=32)
    timeout_per_tool: int = Field(default=300, ge=1, le=3600)
    log_level: str = Field(default="INFO", regex=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    cache_enabled: bool = Field(default=True)
    dry_run: bool = Field(default=False)
```

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

### Advanced Usage with New Features
```python
from code_quality.pipelines.complexity_pipeline import ComplexityPipeline

# Initialize with advanced configuration
pipeline = ComplexityPipeline(
    project_root="/path/to/project",
    enable_plugins=True
)

# Check pipeline health
health = pipeline.health_check()
print(f"Pipeline status: {health['status']}")

# Run analysis with caching
results = pipeline.run_cyclomatic_complexity_analysis()

# Get detailed metrics
metrics = pipeline.get_metrics()
print(f"Files processed: {metrics['files_processed']}")
print(f"Execution time: {metrics['total_execution_time']:.2f}s")

# Clear cache if needed
pipeline.clear_cache()
```

### Programmatic Usage
```bash
# Run all pipelines on specific project
python pipelines/overall_pipeline.py --all --project-root /path/to/project

# Run subset with custom arguments
python pipelines/overall_pipeline.py --pipelines complexity,dead_code --custom-args complexity:--analysis-type,metrics

# Run individual pipeline with verbose output
python pipelines/complexity_pipeline.py --analysis-type cyclomatic --verbose
```

### Configuration Examples
```python
# Using Pydantic validation
from code_quality.pipelines.base_pipeline import PipelineConfig

config = PipelineConfig(
    project_root=Path("/path/to/project"),
    max_workers=8,
    cache_enabled=True,
    log_level="DEBUG"
)

# Configuration automatically validated
pipeline = ComplexityPipeline(config=config)
```

### Health Monitoring Examples
```python
# Monitor pipeline health
pipeline = ComplexityPipeline(project_root="/path/to/project")
health = pipeline.health_check()

if health['status'] == 'healthy':
    print("✅ Pipeline is healthy")
    print(f"Memory usage: {health['memory_usage_mb']} MB")
    print(f"Plugins loaded: {health['plugins_loaded']}")
else:
    print("❌ Pipeline has issues")
    print(f"Issue: {health.get('error', 'Unknown error')}")
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Permission Issues**: Check file permissions for output directories
3. **Pipeline Not Found**: Use `--list` to see available pipelines
4. **Custom Arguments**: Use proper format: `pipeline:arg1,arg2`
5. **Pydantic Not Available**: Falls back to dataclass validation automatically
6. **Cache Issues**: Use `pipeline.clear_cache()` to reset cache
7. **Memory Issues**: Monitor with `pipeline.health_check()`

### New Feature Troubleshooting

#### Caching Issues
```python
# Check cache status
pipeline = ComplexityPipeline(project_root="/path/to/project")
health = pipeline.health_check()

if 'cache_dir' in health:
    print(f"Cache directory: {health['cache_dir']}")

# Clear cache if corrupted
pipeline.clear_cache()
```

#### Configuration Validation Errors
```python
from pydantic import ValidationError
from code_quality.pipelines.base_pipeline import PipelineConfig

try:
    config = PipelineConfig(
        project_root="/nonexistent/path",  # This will fail validation
        max_workers=0  # This will also fail
    )
except ValidationError as e:
    print(f"Configuration error: {e}")
```

#### Plugin Loading Issues
```python
# Check plugin status
pipeline = ComplexityPipeline(project_root="/path/to/project")
health = pipeline.health_check()

if health['plugins_loaded'] == 0:
    print("Warning: No plugins loaded")
    print("Check plugin dependencies and imports")
```

### Debug Mode
```bash
# Enable verbose output
python pipelines/overall_pipeline.py --all --verbose

# List available pipelines
python pipelines/overall_pipeline.py --list

# Check pipeline health
python -c "
from code_quality.pipelines.complexity_pipeline import ComplexityPipeline
p = ComplexityPipeline()
import json
print(json.dumps(p.health_check(), indent=2))
"
```

### Performance Optimization
```python
# Enable caching for better performance
pipeline = ComplexityPipeline(
    project_root="/path/to/project",
    cache_enabled=True,
    max_workers=8  # Utilize more CPU cores
)

# Monitor performance
import time
start = time.time()
results = pipeline.run_cyclomatic_complexity_analysis()
duration = time.time() - start

print(f"Analysis completed in {duration:.2f} seconds")
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