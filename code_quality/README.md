# Code Quality Analysis Tools

This folder contains comprehensive tools for analyzing code quality, dependencies, and identifying unused code in Python repositories.

## 🛠️ Available Tools

### 1. **Unused Code Analyzer** (`unused_code_analyzer.py`)
**Purpose**: Identifies unused functions, classes, modules, and imports throughout your codebase.

**Usage**:
```bash
# Analyze entire src directory
python3 unused_code_analyzer.py src/

# Analyze specific directory
python3 unused_code_analyzer.py src/training/
```

**Outputs**:
- `unused_code_report.json` - Detailed JSON report
- Console summary with cleanup recommendations

**What it finds**:
- 🗑️ Unused functions and methods
- 🏗️ Unused classes
- 📦 Unused modules
- 📥 Dead imports
- 💡 Cleanup potential percentage

---

### 2. **Enhanced Dependency Analyzer** (`enhanced_dependency_analyzer.py`)
**Purpose**: Maps module dependencies and import relationships with syntax error handling.

**Usage**:
```bash
python3 enhanced_dependency_analyzer.py src/
```

**Outputs**:
- `enhanced_dependencies.dot` - DOT format dependency graph
- `enhanced_dependency_report.json` - Comprehensive dependency report
- Console summary with dependency statistics

**What it shows**:
- 🔗 Import dependencies between modules
- 🏠 Internal vs external imports
- 🔄 Circular dependencies
- 📊 Most/least dependent modules

---

### 3. **Function Call Analyzer** (`function_call_analyzer.py`)
**Purpose**: Maps function-to-function call relationships and execution paths.

**Usage**:
```bash
python3 function_call_analyzer.py src/
```

**Outputs**:
- `function_calls.dot` - Function call graph in DOT format
- `function_calls_report.json` - Detailed call analysis
- Console summary with call statistics

**What it reveals**:
- 📞 Function call relationships
- 🔍 Most called/calling functions
- 🏗️ Class method interactions
- 📏 Call chain analysis

---

### 4. **Function Call Visualizer** (`visualize_function_calls.py`)
**Purpose**: Converts DOT files to visual formats and provides interactive analysis.

**Usage**:
```bash
# Show interactive options
python3 visualize_function_calls.py

# Generate heatmap data
python3 visualize_function_calls.py --heatmap

# Show call chains
python3 visualize_function_calls.py --chains

# Create simplified graph
python3 visualize_function_calls.py --simplified

# Generate all image formats
python3 visualize_function_calls.py --all-formats
```

**Outputs**:
- Various image formats (PNG, SVG, PDF, JPG)
- `function_calls_heatmap.csv` - Data for heatmap visualization
- `simplified_function_calls.dot` - Simplified call graph

---

### 5. **Basic Dependency Analyzer** (`dependency_analyzer.py`)
**Purpose**: Simple dependency analysis without external dependencies.

**Usage**:
```bash
python3 dependency_analyzer.py src/
```

**Outputs**:
- `dependencies.dot` - Basic dependency graph
- `dependency_report.json` - Simple dependency report

---

## 🚀 Quick Start

### **Step 1: Run Unused Code Analysis**
```bash
cd code_quality
python3 unused_code_analyzer.py src/
```

### **Step 2: Analyze Dependencies**
```bash
python3 enhanced_dependency_analyzer.py src/
```

### **Step 3: Map Function Calls**
```bash
python3 function_call_analyzer.py src/
```

### **Step 4: Visualize Results**
```bash
python3 visualize_function_calls.py --simplified
```

---

## 📊 Understanding the Outputs

### **Unused Code Report**
- **Unused functions**: Functions defined but never called
- **Unused classes**: Classes defined but never instantiated
- **Unused modules**: Modules that aren't imported anywhere
- **Dead imports**: Imported items that are never used

### **Dependency Reports**
- **Import relationships**: Which modules depend on which
- **Circular dependencies**: Potential import cycles
- **Dependency depth**: How deep the import chains go

### **Function Call Reports**
- **Call relationships**: Which functions call which
- **Call chains**: Execution paths through your code
- **Function usage**: Most and least used functions

---

## 🎯 Use Cases

### **For Code Cleanup**
1. Run `unused_code_analyzer.py` to find dead code
2. Review unused functions and classes
3. Remove or refactor unused code
4. Clean up dead imports

### **For Architecture Review**
1. Run `enhanced_dependency_analyzer.py` to see module dependencies
2. Identify tightly coupled modules
3. Plan refactoring to reduce coupling
4. Document module relationships

### **For Performance Analysis**
1. Run `function_call_analyzer.py` to see call patterns
2. Identify frequently called functions
3. Find potential optimization targets
4. Understand execution bottlenecks

### **For Team Onboarding**
1. Generate visual dependency graphs
2. Create simplified call flow diagrams
3. Document codebase structure
4. Show new developers the big picture

---

## 🔧 Requirements

### **Basic Requirements**
- Python 3.7+
- No external packages needed (uses built-in `ast` module)

### **For Visualization**
- Graphviz (for converting DOT to images)
- Install with: `sudo apt-get install graphviz`

### **For Advanced Analysis**
- Professional tools like `pycallgraph`, `snakefood` (optional)

---

## 📁 Output Files

All tools generate outputs in the current directory:

- **`.dot` files**: Graph description files (can be converted to images)
- **`.json` files**: Detailed analysis reports
- **`.csv` files**: Data for external visualization tools
- **Image files**: Visual representations (when Graphviz is available)

---

## 💡 Pro Tips

1. **Start with unused code analysis** - easiest wins for cleanup
2. **Use simplified graphs** for presentations and documentation
3. **Combine multiple tools** for comprehensive understanding
4. **Run regularly** to track code quality over time
5. **Export to multiple formats** for different stakeholders

---

## 🆘 Troubleshooting

### **Syntax Errors**
- Many files have syntax errors (common in large codebases)
- Tools handle errors gracefully and continue analysis
- Fix syntax errors for more accurate results

### **Large Codebases**
- Analysis may take several minutes for large repositories
- Progress indicators show processing status
- Consider analyzing subdirectories separately

### **Memory Issues**
- Very large codebases may require more memory
- Tools process files incrementally to minimize memory usage
- Consider running on machines with sufficient RAM

---

## 🔄 Regular Maintenance

### **Recommended Schedule**
- **Weekly**: Run unused code analysis
- **Monthly**: Full dependency analysis
- **Quarterly**: Comprehensive function call mapping
- **Before major releases**: All analyses for documentation

### **Integration**
- Add to CI/CD pipelines for automated analysis
- Generate reports for code review processes
- Include in architecture documentation
- Use for technical debt tracking

---

## 📞 Support

These tools are designed to be self-contained and well-documented. If you encounter issues:

1. Check the console output for error messages
2. Verify Python version compatibility (3.7+)
3. Ensure sufficient permissions to read source files
4. Check that the target directory contains Python files

---

## 🎉 Success Metrics

After using these tools, you should see:

- **Reduced codebase size** through dead code removal
- **Cleaner dependencies** with fewer circular imports
- **Better documentation** with visual architecture diagrams
- **Improved maintainability** through understanding of code relationships
- **Faster onboarding** for new team members

