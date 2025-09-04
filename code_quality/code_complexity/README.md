# Code Complexity Analysis Pipeline

A comprehensive code complexity analysis pipeline that combines **PyExamine**, **Radon**, and **Xenon** tools to provide detailed complexity metrics for Python codebases.

## Features

- **Multi-tool Analysis**: Combines PyExamine, Radon, and Xenon for comprehensive complexity assessment
- **Per-file Analysis**: Detailed complexity metrics for individual Python files
- **Per-directory Analysis**: Aggregated complexity metrics for directories
- **Multiple Output Formats**: JSON, HTML, Markdown, and summary reports
- **Configurable**: YAML-based configuration for customizing analysis parameters
- **Command Line Interface**: Easy-to-use CLI for running analyses
- **Tool Availability Checking**: Verify which analysis tools are installed

## Installation

### Prerequisites

Install the required analysis tools:

```bash
# Install PyExamine
pip install pyexamine

# Install Radon
pip install radon

# Install Xenon
pip install xenon
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```bash
# Analyze a single file
python cli.py analyze /path/to/file.py

# Analyze a directory
python cli.py analyze /path/to/directory

# Check if tools are available
python cli.py check-tools
```

### Using Configuration

```bash
# Use custom configuration
python cli.py analyze /path/to/code --config custom_config.yaml

# Generate configuration template
python cli.py generate-config --output my_config.yaml
```

### Output Formats

```bash
# Generate multiple output formats
python cli.py analyze /path/to/code --format json --format html --format markdown
```

## Configuration

The pipeline uses YAML configuration files. Here's the default configuration:

```yaml
tools:
  pyexamine:
    enabled: true
    timeout: 30
  radon:
    enabled: true
    timeout: 30
  xenon:
    enabled: true
    timeout: 30

analysis:
  include_tests: false
  include_docs: false
  max_file_size_mb: 10
  max_line_count: 10000

thresholds:
  complexity: 0.5
  cyclomatic_complexity: 10
  maintainability: 50

output:
  json: true
  html: true
  markdown: true
  summary: true
```

## Analysis Tools

### PyExamine
- Provides code complexity scoring
- Analyzes code structure and patterns
- Generates complexity metrics

### Radon
- **Cyclomatic Complexity**: Measures code complexity based on control flow
- **Maintainability Index**: Assesses code maintainability
- **Raw Metrics**: Lines of code, comments, etc.

### Xenon
- Monitors code complexity over time
- Provides complexity scoring
- Tracks complexity trends

## Output Reports

### JSON Report
Detailed machine-readable results with all metrics and metadata.

### HTML Report
Interactive web-based report with color-coded complexity levels and tables.

### Markdown Report
Human-readable report suitable for documentation and version control.

### Summary Report
Concise overview with key statistics and top complex files.

## API Usage

```python
from complexity_pipeline import ComplexityPipeline

# Initialize pipeline
pipeline = ComplexityPipeline('config.yaml')

# Analyze a file
results = pipeline.run_full_analysis('/path/to/file.py')

# Save results
pipeline.save_results(results, 'analysis_results.json')
```

## Directory Structure

```
code_complexity/
├── analyzers/           # Analysis tool wrappers
│   ├── pyexamine_analyzer.py
│   ├── radon_analyzer.py
│   └── xenon_analyzer.py
├── config/              # Configuration files
│   ├── complexity_config.py
│   └── default_config.yaml
├── utils/               # Utility classes
│   ├── file_utils.py
│   └── report_generator.py
├── reports/             # Generated reports
├── logs/                # Analysis logs
├── complexity_pipeline.py  # Main pipeline
├── cli.py               # Command line interface
└── requirements.txt     # Dependencies
```

## Complexity Scoring

The pipeline calculates a combined complexity score (0.0 to 1.0) based on:

- **PyExamine Score**: Direct complexity assessment
- **Radon Cyclomatic Complexity**: Normalized (higher complexity = lower score)
- **Radon Maintainability Index**: Normalized (higher maintainability = higher score)
- **Xenon Score**: Normalized (lower complexity = higher score)

### Score Interpretation

- **0.7 - 1.0**: Low complexity (good)
- **0.4 - 0.7**: Medium complexity (acceptable)
- **0.0 - 0.4**: High complexity (needs attention)

## Examples

### Analyze Current Directory

```bash
python cli.py analyze . --format html --format markdown
```

### Analyze with Custom Thresholds

```yaml
# custom_config.yaml
thresholds:
  complexity: 0.3
  cyclomatic_complexity: 15
  maintainability: 40
```

```bash
python cli.py analyze /path/to/code --config custom_config.yaml
```

### Programmatic Usage

```python
from complexity_pipeline import ComplexityPipeline
from config.complexity_config import ComplexityConfig

# Create custom config
config = ComplexityConfig()
config.complexity_threshold = 0.3

# Initialize pipeline
pipeline = ComplexityPipeline()
pipeline.config = config

# Run analysis
results = pipeline.run_full_analysis('/path/to/code')
```

## Troubleshooting

### Tool Not Found

If a tool is not found, install it:

```bash
pip install pyexamine radon xenon
```

### Permission Errors

Ensure you have read permissions for the target files and write permissions for the output directory.

### Large Codebases

For large codebases, consider:
- Increasing timeout values in configuration
- Excluding test files and documentation
- Setting file size limits

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is part of the code quality analysis suite and follows the same licensing terms.