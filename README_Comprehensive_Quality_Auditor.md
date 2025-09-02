# Comprehensive Quality Auditor

A single-file solution that provides **exhaustive audit functionality** and **comprehensive, unified report generation** for data quality analysis.

## 🚀 Features

- **Exhaustive Audit**: Comprehensive analysis of files and directories
- **Unified Reporting**: Generate detailed reports in multiple formats
- **Multi-Format Support**: CSV, JSON, TXT, Python, YAML, MD, LOG files
- **No Dependencies**: Uses only standard Python libraries
- **Flexible Analysis**: Single files, directories, or recursive analysis
- **Quality Scoring**: 5-tier quality assessment system
- **Issue Detection**: Automatic identification of data quality problems
- **Smart Recommendations**: Actionable suggestions for improvement

## 📋 Requirements

- Python 3.6+
- No external packages required
- Works on Linux, macOS, and Windows

## 🎯 Quick Start

### Basic Usage

```bash
# Run exhaustive audit on a directory
python3 comprehensive_quality_auditor.py --audit /path/to/directory --recursive

# Run exhaustive audit on a single file
python3 comprehensive_quality_auditor.py --audit /path/to/file.csv

# Generate unified report from existing audit
python3 comprehensive_quality_auditor.py --generate-unified

# Run full audit and generate report
python3 comprehensive_quality_auditor.py --full-audit /path/to/target --output-format both
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--audit PATH` | Run exhaustive audit on specified path | Required (one of three main actions) |
| `--generate-unified` | Generate unified report from existing audit | Required (one of three main actions) |
| `--full-audit PATH` | Run full audit and generate unified report | Required (one of three main actions) |
| `--recursive` | Recursively analyze subdirectories | True |
| `--file-pattern` | File pattern for filtering (e.g., '*.csv') | * |
| `--max-files` | Maximum number of files to analyze | 1000 |
| `--output-format` | Output format: text, json, or both | both |
| `--verbose, -v` | Enable verbose logging | False |

## 🔍 What Gets Audited

### File Properties
- File size and metadata
- Read/write permissions
- Creation and modification times
- File accessibility

### Content Analysis
- **JSON**: Structure, nesting depth, validity
- **CSV**: Row/column consistency, empty cells, headers
- **Python**: Syntax validation, imports, functions, classes
- **Text**: Encoding issues, line analysis, content structure
- **YAML**: Basic structure validation
- **Markdown/Log**: Text content analysis

### Quality Assessment
- **EXCELLENT** (90-100): No issues detected
- **GOOD** (75-89): Minor issues, generally sound
- **ACCEPTABLE** (60-74): Some issues, needs attention
- **POOR** (40-59): Significant problems
- **CRITICAL** (20-39): Severe issues requiring immediate action

## 📊 Report Outputs

### Text Report
- Executive summary with key metrics
- Quality distribution breakdown
- Overall assessment with emojis
- Top recommendations
- Detailed results grouped by quality level
- Human-readable format for stakeholders

### JSON Report
- Machine-readable structured data
- Complete audit results
- Detailed analysis for each file
- Programmatic access to results
- Integration with other tools

### Log File
- Detailed audit process logging
- Progress tracking
- Error reporting
- Performance metrics

## 💡 Use Cases

### Data Quality Teams
- Regular quality assessments
- Compliance reporting
- Issue tracking and prioritization
- Quality improvement planning

### Development Teams
- Code quality checks
- Technical debt assessment
- File organization analysis
- Documentation quality

### Data Scientists
- Dataset quality validation
- Data pipeline monitoring
- Quality metrics tracking
- Automated reporting

### DevOps Engineers
- System health monitoring
- Configuration file validation
- Log file analysis
- Infrastructure quality assessment

## 🔧 Advanced Usage

### Custom File Patterns

```bash
# Audit only CSV files
python3 comprehensive_quality_auditor.py --audit /data --file-pattern "*.csv"

# Audit Python and JSON files
python3 comprehensive_quality_auditor.py --audit /code --file-pattern "*.{py,json}"

# Audit specific file types recursively
python3 comprehensive_quality_auditor.py --audit /project --file-pattern "*.{csv,json,txt}" --recursive
```

### Limited Analysis

```bash
# Quick audit of first 100 files
python3 comprehensive_quality_auditor.py --audit /large-directory --max-files 100

# Non-recursive analysis
python3 comprehensive_quality_auditor.py --audit /directory --recursive=false
```

### Output Control

```bash
# JSON only for API integration
python3 comprehensive_quality_auditor.py --full-audit /path --output-format json

# Text only for human review
python3 comprehensive_quality_auditor.py --full-audit /path --output-format text

# Both formats for complete coverage
python3 comprehensive_quality_auditor.py --full-audit /path --output-format both
```

## 📈 Performance Considerations

- **File Count**: Default limit of 1000 files prevents excessive runtime
- **Recursive Analysis**: Can be disabled for shallow directory structures
- **Progress Tracking**: Built-in progress indicators for large audits
- **Memory Management**: Efficient processing of large files
- **Logging**: Configurable verbosity levels

## 🛠️ Integration Examples

### CI/CD Pipeline

```yaml
# GitHub Actions example
- name: Quality Audit
  run: |
    python3 comprehensive_quality_auditor.py --full-audit . --output-format json
    python3 -c "
    import json
    with open('comprehensive_audit_report_*.json') as f:
        data = json.load(f)
        if data['audit_summary']['critical_issues'] > 0:
            exit(1)
    "
```

### Automated Monitoring

```bash
#!/bin/bash
# Daily quality check script
python3 comprehensive_quality_auditor.py --full-audit /data --output-format both

# Check for critical issues
if grep -q "CRITICAL" comprehensive_audit_report_*.txt; then
    echo "Critical quality issues detected!" | mail -s "Quality Alert" admin@company.com
fi
```

### Python Integration

```python
from comprehensive_quality_auditor import ComprehensiveQualityAuditor

# Initialize auditor
auditor = ComprehensiveQualityAuditor()

# Run audit
results = auditor.run_exhaustive_audit("/path/to/data", recursive=True)

# Access results programmatically
summary = results["audit_summary"]
quality_score = summary["overall_quality"]
critical_issues = summary["critical_issues"]

# Generate report
report_file = auditor.generate_unified_report("text")
```

## 🚨 Troubleshooting

### Common Issues

1. **Permission Denied**: Ensure read access to target directories
2. **Large File Count**: Use `--max-files` to limit analysis scope
3. **Memory Issues**: Process smaller batches or disable recursive analysis
4. **Encoding Errors**: Files with non-UTF-8 encoding may cause issues

### Debug Mode

```bash
# Enable verbose logging
python3 comprehensive_quality_auditor.py --audit /path --verbose

# Check log file for detailed information
tail -f quality_audit.log
```

## 📝 Output Examples

### Sample Text Report Header

```
====================================================================================================
COMPREHENSIVE QUALITY AUDIT REPORT
====================================================================================================

Generated: 2025-09-02 17:50:26

EXECUTIVE SUMMARY
--------------------------------------------------
Target Path: /data/project
Audit Mode: exhaustive
Total Files: 150
Successful Audits: 148
Failed Audits: 2
Success Rate: 98.7%
Overall Quality: GOOD
Total Data Size: 2.45 MB
Critical Issues: 1
Audit Duration: 12.34 seconds
```

### Sample JSON Structure

```json
{
  "audit_summary": {
    "audit_timestamp": "2025-09-02T17:50:26.033242",
    "target_path": "/data/project",
    "overall_quality": "good",
    "total_files": 150,
    "critical_issues": 1
  },
  "analysis_results": {
    "file1.csv": {
      "quality_assessment": {
        "overall_quality": "excellent",
        "quality_score": 95
      }
    }
  }
}
```

## 🤝 Contributing

This tool is designed to be self-contained and extensible. Key areas for enhancement:

- Additional file format support
- Custom quality metrics
- Integration with external quality tools
- Performance optimizations
- Enhanced reporting templates

## 📄 License

This tool is provided as-is for data quality assessment and reporting purposes.

---

**Ready to audit your data quality?** Run the comprehensive quality auditor today and get detailed insights into your files and directories!