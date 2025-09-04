# Two Separate Pipelines - Usage Guide

## 🎯 Overview

You now have **two completely separate, independent pipelines**:

1. **Code Mapping Pipeline** - Dead code analysis and dependency mapping
2. **Code Complexity Pipeline** - Comprehensive complexity analysis

## 📁 Pipeline 1: Code Mapping & Dead Code Analysis

### **File:** `map_code_interactions.py`

### **Purpose:**
- Dead code detection and analysis
- Dependency mapping and relationships
- Call graph analysis
- Architecture analysis
- Import relationship analysis
- Basic complexity analysis (Radon only)

### **Usage:**
```bash
# Navigate to code_quality directory
cd /workspace/code_quality

# Basic analysis
python3 map_code_interactions.py

# Analyze specific project
python3 map_code_interactions.py --project-root /path/to/your/project

# Exclude specific directories
python3 map_code_interactions.py --exclude venv __pycache__ .git
```

### **What It Does:**
1. **Dependencies Analysis** - Maps module relationships
2. **Call Graph Analysis** - Analyzes function call patterns
3. **Architecture Analysis** - Identifies system structure
4. **Import Analysis** - Maps import relationships
5. **Basic Complexity Analysis** - Simple Radon-based complexity
6. **Dead Code Analysis** - Identifies unused/deprecated code

### **Output:**
- Comprehensive HTML reports
- JSON data files
- Dead code analysis results
- Dependency maps
- Architecture diagrams

---

## 📁 Pipeline 2: Code Complexity Analysis

### **File:** `code_complexity/cli.py`

### **Purpose:**
- Comprehensive complexity analysis
- Multi-tool integration (PyExamine, Radon, Xenon)
- Per-file and per-directory analysis
- Advanced reporting and visualizations

### **Usage:**
```bash
# Navigate to complexity analysis directory
cd /workspace/code_quality/code_complexity

# Analyze a single file
python3 cli.py analyze /path/to/file.py

# Analyze a directory
python3 cli.py analyze /path/to/directory

# Check tool availability
python3 cli.py check-tools

# Generate configuration template
python3 cli.py generate-config --output my_config.yaml

# Multiple output formats
python3 cli.py analyze /path/to/code --format json --format html --format markdown
```

### **What It Does:**
1. **Multi-Tool Analysis** - Combines PyExamine, Radon, and Xenon
2. **Per-File Analysis** - Detailed complexity metrics for each file
3. **Per-Directory Analysis** - Aggregated complexity statistics
4. **Combined Scoring** - Unified complexity scores (0.0-1.0)
5. **Advanced Reporting** - JSON, HTML, Markdown, and summary reports

### **Output:**
- JSON reports with detailed metrics
- HTML reports with interactive tables
- Markdown reports for documentation
- Summary reports with key statistics
- Tool comparison data

---

## 🔄 How to Use Both Pipelines

### **Workflow 1: Dead Code Analysis First**
```bash
# Step 1: Run code mapping to find dead code
cd /workspace/code_quality
python3 map_code_interactions.py --project-root /path/to/project

# Step 2: Run complexity analysis on the same project
cd code_complexity
python3 cli.py analyze /path/to/project --format json --format markdown
```

### **Workflow 2: Complexity Analysis First**
```bash
# Step 1: Run complexity analysis to identify complex areas
cd /workspace/code_quality/code_complexity
python3 cli.py analyze /path/to/project --format html

# Step 2: Run code mapping to find dead code in complex areas
cd ..
python3 map_code_interactions.py --project-root /path/to/project
```

### **Workflow 3: Parallel Analysis**
```bash
# Run both pipelines simultaneously on different terminals
# Terminal 1:
cd /workspace/code_quality && python3 map_code_interactions.py --project-root /path/to/project

# Terminal 2:
cd /workspace/code_quality/code_complexity && python3 cli.py analyze /path/to/project
```

---

## 📊 Output Comparison

### **Code Mapping Pipeline Output:**
```
CODE INTERACTION MAPPING COMPLETE!
================================================================================

[1/5] Analyzing module dependencies...
  - Found 45 modules
  - Total dependencies: 234

[2/5] Analyzing function call graph...
  - Found 156 functions
  - Total function calls: 445

[3/5] Analyzing system architecture...
  - Identified 3 architectural layers
  - Found 12 components

[4/5] Analyzing import relationships...
  - Total imports: 89
  - Circular imports: 2

[5/6] Analyzing code complexity...
  - Average cyclomatic complexity: 8.45
  - Files with high complexity: 12
  - Note: For comprehensive complexity analysis, use: python code_complexity/cli.py

[6/6] Analyzing dead code and deprecated patterns...
  - Found 23 potentially dead functions
  - 8 functions flagged as high-risk
  - 15 functions safe to remove

NEXT STEPS
==================================================
For comprehensive complexity analysis, run:
  python code_complexity/cli.py analyze /path/to/your/project
```

### **Code Complexity Pipeline Output:**
```
Starting complexity analysis on: /path/to/project

[5/7] Running enhanced complexity analysis...
  - Files analyzed: 45
  - Average complexity score: 0.623
  - Highest complexity: 0.891
  - Lowest complexity: 0.234
  - PyExamine average: 0.567
  - Radon CC average: 7.23
  - Xenon average: 4.12
  - Complexity distribution:
    * Low (≥0.7): 12 files (26.7%)
    * Medium (0.4-0.7): 28 files (62.2%)
    * High (<0.4): 5 files (11.1%)

==================================================
ANALYSIS SUMMARY
==================================================
Files analyzed: 45
Average complexity: 0.623
Highest complexity: 0.891
Lowest complexity: 0.234

Complexity distribution:
  Low (≥0.7):     12 files ( 26.7%)
  Medium (0.4-0.7): 28 files ( 62.2%)
  High (<0.4):     5 files ( 11.1%)
==================================================
Analysis completed. Results saved to: complexity_analysis_20250904_151739.json
```

---

## 🛠️ Installation Requirements

### **Code Mapping Pipeline:**
- **No additional tools required** - Uses existing analyzers
- **Works out of the box** - All dependencies already available

### **Code Complexity Pipeline:**
- **Optional tools** (for enhanced analysis):
  ```bash
  pip install pyexamine  # For PyExamine analysis
  pip install radon      # For Radon analysis  
  pip install xenon      # For Xenon analysis
  ```
- **Visualization libraries** (optional):
  ```bash
  pip install matplotlib seaborn pandas  # For charts and graphs
  ```
- **Works without tools** - Graceful degradation when tools unavailable

---

## 🎯 When to Use Which Pipeline

### **Use Code Mapping Pipeline When:**
- You want to find dead/unused code
- You need to understand dependency relationships
- You want to map function call patterns
- You need architecture analysis
- You want to identify circular imports

### **Use Code Complexity Pipeline When:**
- You want detailed complexity metrics
- You need to compare different complexity tools
- You want per-file complexity analysis
- You need complexity visualizations
- You want to track complexity trends

### **Use Both Pipelines When:**
- You want comprehensive code analysis
- You need to understand both dead code and complexity
- You're planning a major refactoring
- You want to prioritize refactoring efforts
- You need complete code quality assessment

---

## 📁 File Structure

```
code_quality/
├── map_code_interactions.py          # Pipeline 1: Code Mapping
├── code_complexity/                  # Pipeline 2: Code Complexity
│   ├── cli.py                       # Main complexity CLI
│   ├── complexity_pipeline.py       # Core complexity pipeline
│   ├── analyzers/                   # Complexity analyzers
│   ├── config/                      # Configuration
│   ├── utils/                       # Utilities
│   └── reports/                     # Generated reports
├── analyzers/                       # Shared analyzers
├── reporters/                       # Shared reporters
└── core/                           # Shared core functionality
```

---

## ✅ Benefits of Separate Pipelines

1. **Focused Purpose** - Each pipeline has a clear, specific purpose
2. **Independent Operation** - Can run separately or together
3. **Simplified Maintenance** - Easier to maintain and update
4. **Flexible Usage** - Use only what you need
5. **Clear Output** - Distinct, focused reports
6. **Better Performance** - No unnecessary overhead
7. **Easier Debugging** - Issues are isolated to specific pipelines

You now have two powerful, independent tools that can work together or separately based on your needs!