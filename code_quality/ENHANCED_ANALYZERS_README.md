# Enhanced Dead Code Analyzers

This document describes the new Multi-Modal and Context-Aware dead code analyzers that significantly improve the accuracy and reliability of dead code detection.

## Overview

The enhanced analyzers provide:

1. **Multi-Modal Analysis**: Combines multiple analysis approaches for superior accuracy
2. **Context-Aware Analysis**: Considers framework patterns and usage context to reduce false positives
3. **Framework Detection**: Automatically detects Django, Flask, FastAPI, and other frameworks
4. **Pattern Recognition**: Identifies design patterns, anti-patterns, and usage patterns
5. **Confidence Scoring**: Provides confidence levels for all analysis results

## Components

### 1. Multi-Modal Dead Code Analyzer (`multi_modal_dead_code_analyzer.py`)

Combines four different analysis approaches:

- **Static Analysis**: AST-based dead code detection
- **Dynamic Analysis**: Runtime usage pattern analysis
- **Semantic Analysis**: Code meaning and context analysis
- **Machine Learning**: ML-based dead code prediction

**Key Features:**
- Cross-validation across multiple analyzers
- Consensus-based result combination
- Confidence scoring for each result
- Disagreement analysis between analyzers

**Usage:**
```python
from analyzers.multi_modal_dead_code_analyzer import MultiModalDeadCodeAnalyzer
from core.config import AnalysisConfig

config = AnalysisConfig()
analyzer = MultiModalDeadCodeAnalyzer(config)
result = analyzer.analyze(project_root)
```

### 2. Context-Aware Dead Code Analyzer (`context_aware_dead_code_analyzer.py`)

Provides framework-aware dead code analysis:

- **Framework Detection**: Identifies the primary framework used
- **Context Filtering**: Filters false positives based on framework patterns
- **Usage Pattern Analysis**: Considers how code is actually used
- **Importance Scoring**: Weights code elements by importance

**Key Features:**
- Framework-specific rule sets
- Lifecycle method detection
- Public API identification
- Framework hook recognition

**Usage:**
```python
from analyzers.context_aware_dead_code_analyzer import ContextAwareDeadCodeAnalyzer
from core.config import AnalysisConfig

config = AnalysisConfig()
analyzer = ContextAwareDeadCodeAnalyzer(config)
result = analyzer.analyze(project_root)
```

### 3. Framework Detector (`framework_detector.py`)

Automatically detects frameworks and development patterns:

**Supported Frameworks:**
- Django (models, views, urls, admin)
- Flask (routes, blueprints, extensions)
- FastAPI (routers, dependencies, middleware)
- Pyramid (views, routes, configuration)
- Tornado (handlers, applications)
- Celery (tasks, workers)
- SQLAlchemy (models, sessions)
- Pytest (fixtures, tests)

**Detection Methods:**
- Import statement analysis
- File structure analysis
- Configuration file analysis
- Pattern matching

**Usage:**
```python
from analyzers.framework_detector import FrameworkDetector

detector = FrameworkDetector()
context = detector.detect_frameworks(project_root)
```

### 4. Pattern Analyzer (`pattern_analyzer.py`)

Analyzes usage patterns and design patterns:

**Pattern Types:**
- **Design Patterns**: Singleton, Factory, Observer, Decorator, etc.
- **Framework Patterns**: Django models, Flask routes, FastAPI routers
- **Anti-Patterns**: God Object, Long Parameter List, Deep Nesting
- **Usage Patterns**: Function usage, class usage, import patterns

**Key Features:**
- Pattern recognition and classification
- Anti-pattern detection
- Usage frequency analysis
- Importance scoring

**Usage:**
```python
from analyzers.pattern_analyzer import PatternAnalyzer

analyzer = PatternAnalyzer()
result = analyzer.analyze_patterns(project_root)
```

## Integration with Dead Code Pipeline

The enhanced analyzers are fully integrated with the existing dead code pipeline:

### New Analysis Types

1. **Multi-Modal Analysis**:
   ```bash
   python code_quality/pipelines/dead_code_pipeline.py --analysis-type multi_modal
   ```

2. **Context-Aware Analysis**:
   ```bash
   python code_quality/pipelines/dead_code_pipeline.py --analysis-type context_aware
   ```

3. **Comprehensive Analysis** (includes all new analyzers):
   ```bash
   python code_quality/pipelines/dead_code_pipeline.py --analysis-type all
   ```

### Pipeline Integration

The pipeline now includes:
- `run_multi_modal_dead_code_analysis()`: Runs multi-modal analysis
- `run_context_aware_dead_code_analysis()`: Runs context-aware analysis
- Enhanced reporting with confidence scores and framework information

## Benefits

### Accuracy Improvements

1. **Reduced False Positives**: Context-aware filtering eliminates framework-specific false positives
2. **Cross-Validation**: Multi-modal analysis validates results across multiple approaches
3. **Framework Awareness**: Understands framework conventions and patterns
4. **Usage Context**: Considers how code is actually used in the project

### Performance Benefits

1. **Parallel Analysis**: Multiple analyzers run in parallel where possible
2. **Smart Caching**: Results are cached to avoid redundant analysis
3. **Incremental Updates**: Only re-analyze changed files
4. **Optimized Detection**: Framework-specific optimizations

### Developer Experience

1. **Better Reporting**: Detailed reports with confidence scores and recommendations
2. **Framework Insights**: Understands your project's framework and patterns
3. **Actionable Results**: Clear recommendations for code cleanup
4. **Historical Tracking**: Track improvements over time

## Example Results

### Multi-Modal Analysis Results

```json
{
  "total_analyzers": 4,
  "successful_analyzers": 4,
  "dead_functions": 12,
  "dead_classes": 3,
  "dead_imports": 8,
  "overall_confidence": 0.85,
  "consensus_scores": {
    "functions": 0.82,
    "classes": 0.78,
    "imports": 0.91
  }
}
```

### Context-Aware Analysis Results

```json
{
  "primary_framework": "django",
  "framework_confidence": 0.95,
  "context_aware_dead_functions": 5,
  "context_aware_dead_classes": 1,
  "context_aware_dead_imports": 3,
  "false_positives_filtered": 7,
  "context_awareness_score": 0.88,
  "overall_confidence": 0.87
}
```

## Testing

Run the test suite to verify the analyzers are working correctly:

```bash
python code_quality/test_enhanced_analyzers.py
```

The test suite will:
1. Test framework detection
2. Test pattern analysis
3. Test multi-modal analysis
4. Test context-aware analysis
5. Provide a summary of results

## Configuration

### AnalysisConfig Options

```python
config = AnalysisConfig()
config.enable_framework_detection = True
config.enable_pattern_analysis = True
config.enable_multi_modal_analysis = True
config.enable_context_aware_analysis = True
config.confidence_threshold = 0.7
config.framework_specific_rules = True
```

### Framework-Specific Settings

```python
# Django-specific settings
config.django_settings = {
    "protected_models": ["User", "Group"],
    "protected_views": ["LoginView", "LogoutView"],
    "admin_protection": True
}

# Flask-specific settings
config.flask_settings = {
    "protected_routes": ["/", "/health"],
    "blueprint_protection": True
}
```

## Best Practices

1. **Use Context-Aware Analysis**: Always use context-aware analysis for framework-based projects
2. **Review Confidence Scores**: Pay attention to confidence scores when making decisions
3. **Check Framework Detection**: Verify that the correct framework is detected
4. **Review Recommendations**: Follow the generated recommendations for best results
5. **Regular Analysis**: Run analysis regularly to track improvements

## Troubleshooting

### Common Issues

1. **Framework Not Detected**: Check import statements and file structure
2. **High False Positives**: Enable context-aware analysis
3. **Low Confidence Scores**: Review code documentation and patterns
4. **Performance Issues**: Use incremental analysis for large codebases

### Debug Mode

Enable debug mode for detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

Planned improvements include:

1. **Machine Learning Models**: Train custom models for specific frameworks
2. **Real-Time Analysis**: File system watching for immediate feedback
3. **IDE Integration**: Direct integration with popular IDEs
4. **Custom Rules**: User-defined framework rules and patterns
5. **Performance Optimization**: Further performance improvements for large codebases

## Contributing

To contribute to the enhanced analyzers:

1. Follow the existing code structure
2. Add comprehensive tests
3. Update documentation
4. Ensure backward compatibility
5. Follow the coding standards

## License

The enhanced analyzers are part of the code quality toolkit and follow the same license terms.