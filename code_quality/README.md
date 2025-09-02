# Code Quality Tools

A comprehensive suite of Python code quality analysis, auto-fixing, and reporting tools.

## Features

### 🔧 Auto-Fixers
- **Syntax Auto-Fix**: Comprehensive Python syntax correction using multiple tools
- **Code Formatting**: Black, isort, autopep8, and yapf integration
- **Import Organization**: Automatic import sorting and organization
- **Style Consistency**: Enforce consistent coding standards

### 📊 Analysis Tools
- **Linter Integration**: Flake8, pylint, mypy, and pycodestyle
- **Code Complexity**: Radon-based complexity analysis
- **Security Analysis**: Bandit security vulnerability detection
- **Dead Code Detection**: Vulture-based unused code identification
- **Dependency Analysis**: Import and package dependency assessment

### 📈 Reporting
- **Error Reports**: Per-file and per-directory error counts
- **Quality Metrics**: Complexity scores, maintainability indices
- **Visual Reports**: Rich terminal output and HTML reports
- **Trend Analysis**: Historical quality tracking

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Auto-Fix**:
   ```bash
   python -m code_quality.fixers.auto_fixer --path /path/to/your/code
   ```

3. **Generate Quality Report**:
   ```bash
   python -m code_quality.reporters.quality_reporter --path /path/to/your/code
   ```

4. **Analyze Dependencies**:
   ```bash
   python -m code_quality.analyzers.dependency_analyzer --path /path/to/your/code
   ```

## Tool Categories

### Core (`core/`)
- Configuration management
- Common utilities
- Base classes and interfaces

### Fixers (`fixers/`)
- `auto_fixer.py`: Main auto-fixing orchestrator
- `syntax_fixer.py`: Syntax error correction
- `import_fixer.py`: Import organization
- `style_fixer.py`: Code style formatting

### Analyzers (`analyzers/`)
- `linter_analyzer.py`: Linting and syntax analysis
- `complexity_analyzer.py`: Code complexity metrics
- `dead_code_analyzer.py`: Unused code detection
- `dependency_analyzer.py`: Import and package analysis
- `security_analyzer.py`: Security vulnerability detection

### Reporters (`reporters/`)
- `quality_reporter.py`: Comprehensive quality reports
- `error_reporter.py`: Error summary and statistics
- `html_reporter.py`: HTML-formatted reports
- `trend_reporter.py`: Historical quality tracking

### Utils (`utils/`)
- File processing utilities
- Output formatting
- Configuration helpers

## Configuration

Create a `config.yaml` file in your project root:

```yaml
code_quality:
  auto_fix:
    enabled: true
    tools: ["black", "isort", "autopep8"]
  
  analysis:
    linters: ["flake8", "pylint", "mypy"]
    complexity_threshold: 10
    security_checks: true
  
  reporting:
    output_format: ["terminal", "html"]
    include_metrics: true
    save_reports: true
```

## Examples

### Fix All Issues in a Directory
```python
from code_quality.fixers.auto_fixer import AutoFixer

fixer = AutoFixer("/path/to/code")
fixer.fix_all()
```

### Generate Quality Report
```python
from code_quality.reporters.quality_reporter import QualityReporter

reporter = QualityReporter("/path/to/code")
report = reporter.generate_report()
reporter.save_report("quality_report.html")
```

### Analyze Dependencies
```python
from code_quality.analyzers.dependency_analyzer import DependencyAnalyzer

analyzer = DependencyAnalyzer("/path/to/code")
dependencies = analyzer.analyze()
analyzer.generate_report()
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.