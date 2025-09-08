# Attribute Checker Pipeline

## Overview

The Attribute Checker Pipeline is a specialized code quality tool designed to detect missing methods and attributes in Python classes. It uses advanced static analysis techniques to identify potential attribute access issues while minimizing false positives.

## Features

### 🔍 **Advanced Detection**
- **Missing Method Calls**: Identifies calls to methods that don't exist in the class
- **Undefined Attributes**: Detects access to attributes that aren't defined
- **Inheritance Awareness**: Understands parent class relationships
- **Context Analysis**: Provides detailed context for each issue

### 🎯 **Smart Filtering**
- **External Attributes**: Filters out commonly injected attributes (`logger`, `config`, etc.)
- **Instance Attributes**: Recognizes attributes set in `__init__` methods
- **Parent Methods**: Excludes methods defined in parent classes
- **Common Patterns**: Handles typical Python patterns and idioms

### 📊 **Severity Classification**
- **🚨 Critical Errors**: Real issues requiring immediate attention
- **⚠️ Warnings**: Potential issues that may be false positives
- **ℹ️ Info**: Suggestions for code improvement

## Usage

### Command Line Interface

```bash
# Analyze single file
python pipelines/attribute_checker_pipeline.py \
    --target-file src/my_file.py \
    --class-name MyClass

# Analyze entire directory
python pipelines/attribute_checker_pipeline.py \
    --target-dir src/ \
    --exclude-patterns "__pycache__,test_*"

# Generate custom report
python pipelines/attribute_checker_pipeline.py \
    --target-dir src/ \
    --output-file my_report.json
```

### Programmatic Usage

```python
from code_quality.pipelines.attribute_checker_pipeline import AttributeCheckerPipeline

# Initialize pipeline
pipeline = AttributeCheckerPipeline()

# Analyze single file
result = pipeline.analyze_file("src/my_file.py", "MyClass")

# Analyze directory
result = pipeline.analyze_directory("src/")

# Generate report
report_path = pipeline.generate_report("analysis_report.json")
```

## Configuration Options

### Pipeline Configuration

```yaml
# config.yaml
attribute_checker:
  exclude_common_attrs: true
  severity_filter: "all"  # "all", "warnings", "errors"
  max_workers: 4
  timeout: 300
```

### Plugin Configuration

```yaml
# plugin_config.yaml
plugins:
  attribute_checker:
    enabled: true
    severity_filter: "warnings"
    exclude_common_attrs: true
```

## Analysis Output

### Sample Report Structure

```json
{
  "timestamp": "20250908_225718",
  "pipeline_type": "AttributeCheckerPipeline",
  "project_root": "/path/to/project",
  "summary": {
    "total_files_analyzed": 5,
    "total_errors": 0,
    "total_warnings": 23,
    "files_with_issues": 3
  },
  "analysis_results": {
    "file1.py": {
      "status": "success",
      "missing_count": 5,
      "error_count": 0,
      "warning_count": 5,
      "missing_items": [...]
    }
  }
}
```

### Issue Categories

#### Method Call Issues
```python
# ❌ Missing method
self.unknown_method()  # Method doesn't exist in class

# ✅ Valid method call
self.existing_method()  # Method defined in class or parent
```

#### Attribute Access Issues
```python
# ❌ Undefined attribute
self.unknown_attr  # Attribute not defined anywhere

# ✅ Valid attribute access
self.instance_attr  # Defined in __init__
self.config         # External/injected attribute
```

## Integration with Code Quality Tools

### Standalone Usage
```bash
# Direct script usage
python attribute_checker.py src/file.py ClassName

# Pipeline integration
python -m code_quality.cli run-pipeline attribute_checker --target src/
```

### Plugin Integration
```python
from plugins.attribute_checker_analyzer import AttributeCheckerAnalyzer

analyzer = AttributeCheckerAnalyzer()
result = analyzer.analyze("src/my_file.py")
```

## Advanced Features

### Custom Exclusion Patterns

```python
# Exclude specific patterns
exclude_patterns = ["test_*", "temp_*", "__pycache__"]
result = pipeline.analyze_directory("src/", exclude_patterns)
```

### Severity Filtering

```python
# Only show critical errors
pipeline.config.update({"severity_filter": "errors"})
result = pipeline.run_analysis("src/")

# Show all issues
pipeline.config.update({"severity_filter": "all"})
```

### Multi-Class Analysis

The pipeline automatically detects and analyzes all classes in a file:

```python
# File with multiple classes
classes = pipeline._extract_classes_from_file("src/complex.py")
# Returns: ["BaseClass", "DerivedClass", "HelperClass"]

# Analyze all classes
for class_name in classes:
    result = pipeline.analyze_file("src/complex.py", class_name)
```

## Performance Considerations

### Optimization Strategies
- **Parallel Processing**: Multiple files analyzed concurrently
- **Caching**: Results cached to avoid re-analysis
- **Selective Analysis**: Only Python files are processed
- **Memory Efficient**: AST-based analysis with minimal memory footprint

### Performance Tuning

```python
# Configure for performance
config = {
    "parallel_execution": True,
    "max_workers": 8,
    "cache_enabled": True,
    "timeout_per_tool": 60
}

pipeline = AttributeCheckerPipeline(config=config)
```

## Troubleshooting

### Common Issues

#### High False Positive Rate
```python
# Solution: Enable common attribute filtering
pipeline.config.update({"exclude_common_attrs": True})
```

#### Slow Analysis
```python
# Solution: Increase workers and enable caching
config = {
    "max_workers": 8,
    "cache_enabled": True,
    "parallel_execution": True
}
```

#### Missing Dependencies
```bash
# Solution: Ensure attribute_checker.py is in Python path
export PYTHONPATH="$PYTHONPATH:/path/to/code_quality"
```

### Debug Mode

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

pipeline = AttributeCheckerPipeline()
```

## Integration Examples

### CI/CD Integration

```bash
# In your CI pipeline
#!/bin/bash
python code_quality/pipelines/attribute_checker_pipeline.py \
    --target-dir src/ \
    --output-file attribute_report.json

# Check exit code
if [ $? -ne 0 ]; then
    echo "Attribute access issues found!"
    exit 1
fi
```

### Pre-commit Hook

```bash
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: attribute-checker
        name: Attribute Checker
        entry: python code_quality/pipelines/attribute_checker_pipeline.py
        args: [--target-dir, src/, --severity-filter, errors]
        language: system
        files: \.py$
```

### IDE Integration

```python
# VS Code extension integration
import subprocess

def run_attribute_check(file_path, class_name):
    cmd = [
        "python", "code_quality/pipelines/attribute_checker_pipeline.py",
        "--target-file", file_path,
        "--class-name", class_name,
        "--output-file", "temp_report.json"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0
```

## Contributing

### Adding New Filters

```python
# Extend AttributeChecker class
class CustomAttributeChecker(AttributeChecker):
    def __init__(self, class_name: str):
        super().__init__(class_name)

        # Add custom external attributes
        self.known_external_attrs.update({
            "custom_attr": True,
            "project_specific": True
        })
```

### Custom Plugins

```python
# Create custom plugin
from plugins.base_plugin import BasePlugin, PluginCategory

class CustomAttributePlugin(BasePlugin):
    def get_metadata(self):
        return PluginMetadata(
            name="CustomAttributeChecker",
            category=PluginCategory.ANALYSIS,
            # ... other metadata
        )
```

## Support

### Getting Help

1. **Check Logs**: Enable verbose logging for detailed analysis
2. **Review Configuration**: Verify pipeline and plugin settings
3. **Validate Dependencies**: Ensure all required modules are available
4. **Check File Permissions**: Verify read access to target files

### Common Error Messages

- `"Attribute checker not available"`: Missing dependencies
- `"No classes found in file"`: File doesn't contain class definitions
- `"Analysis failed: <error>"`: Syntax or parsing error in target file

## Changelog

### Version 2.0.0
- ✅ Enhanced false positive reduction
- ✅ Plugin system integration
- ✅ Severity classification system
- ✅ Parallel processing support
- ✅ Comprehensive reporting

### Version 1.0.0
- ✅ Basic attribute access detection
- ✅ Simple filtering system
- ✅ Command-line interface
- ✅ JSON report generation
