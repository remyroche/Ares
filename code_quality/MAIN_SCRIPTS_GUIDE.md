# Main Scripts Guide - Enhanced Import Analysis System

## 🎯 **Primary Entry Points**

### **1. Enhanced Import Analysis (Main Analysis Script)**
```bash
# Core analyzer - comprehensive import and undefined variable analysis
python3 analyzers/enhanced_import_analysis.py --target . --stats
```
**Purpose:** Main analysis engine with enhanced accuracy and detailed reporting
**Features:** 
- Enhanced import analysis (duplicates, wildcards, relative imports)
- Advanced undefined variable detection with reduced false positives
- Issue classification and severity levels
- Comprehensive JSON reporting

### **2. Intelligent Import Fixer (Auto-Fixing Script)**
```bash
# Intelligent auto-fixing with confidence-based decisions
python3 analyzers/intelligent_import_fixer.py --target . --interactive
```
**Purpose:** Automatically fix import issues with intelligent confidence assessment
**Features:**
- 95% auto-fix for high-confidence issues
- 4% confirm-fix for medium-confidence issues  
- 1% flag-only for low-confidence issues
- Comprehensive safety validation

### **3. Standalone Runner (Simple Entry Point)**
```bash
# Easy-to-use standalone runner
python3 run_enhanced_import_analysis.py --target . --stats --verbose
```
**Purpose:** Simple entry point without complex pipeline dependencies
**Features:**
- No complex pipeline infrastructure required
- Direct analyzer usage
- Command-line interface
- JSON report generation

### **4. Pipeline Integration (Advanced Usage)**
```bash
# Full pipeline integration (if pipeline system is working)
python3 pipelines/pipeline_enhanced_import_analysis.py --target . --stats
```
**Purpose:** Integration with existing code_quality pipeline infrastructure
**Features:**
- Plugin system integration
- Advanced configuration options
- Pipeline reporting
- **Note:** May have import issues with complex pipeline dependencies

## 📊 **Script Comparison**

| Script | Purpose | Complexity | Auto-Fix | Best For |
|--------|---------|------------|----------|----------|
| `enhanced_import_analysis.py` | Analysis only | Low | ❌ | Understanding issues |
| `intelligent_import_fixer.py` | Analysis + Auto-fix | Medium | ✅ | Production use |
| `run_enhanced_import_analysis.py` | Simple analysis | Low | ❌ | Quick analysis |
| `pipeline_enhanced_import_analysis.py` | Full pipeline | High | ❌ | Enterprise integration |

## 🚀 **Recommended Usage**

### **For Most Users:**
```bash
# Start with analysis to understand issues
python3 analyzers/enhanced_import_analysis.py --target . --stats

# Then use intelligent fixer for auto-fixing
python3 analyzers/intelligent_import_fixer.py --target . --interactive
```

### **For Simple Analysis:**
```bash
# Quick and easy analysis
python3 run_enhanced_import_analysis.py --target . --stats --verbose
```

### **For Production/CI:**
```bash
# Automated fixing with reporting
python3 analyzers/intelligent_import_fixer.py --target . --output report.json
```

## 🔧 **Command Line Options**

### **Enhanced Import Analysis:**
```bash
python3 analyzers/enhanced_import_analysis.py [OPTIONS]

Options:
  --target, -t PATH        Target file or directory
  --output, -o FILE        Output JSON report file
  --min-severity LEVEL     Minimum severity (low/medium/high/critical)
  --max-issues-per-file N  Maximum issues per file
  --ignore-patterns LIST   Directory patterns to ignore
  --stats                  Show detailed statistics
```

### **Intelligent Import Fixer:**
```bash
python3 analyzers/intelligent_import_fixer.py [OPTIONS] TARGET

Options:
  --interactive, -i        Interactive mode for confirmations
  --dry-run               Show what would be fixed
  --no-auto-fix           Disable automatic fixing
  --no-backup             Don't create backups
  --output, -o FILE       Output report file
```

### **Standalone Runner:**
```bash
python3 run_enhanced_import_analysis.py [OPTIONS]

Options:
  --target, -t PATH        Target file or directory
  --output, -o FILE        Output JSON report file
  --stats                  Show detailed statistics
  --verbose, -v            Enable verbose output
  --min-severity LEVEL     Minimum severity level
```

## 📁 **File Structure**

```
code_quality/
├── analyzers/
│   ├── enhanced_import_analysis.py      # 🎯 MAIN ANALYZER
│   ├── intelligent_import_fixer.py      # 🎯 AUTO-FIXER
│   └── duplicate_import_fixer.py        # Specialized duplicate fixer
├── pipelines/
│   └── pipeline_enhanced_import_analysis.py  # Pipeline integration
├── run_enhanced_import_analysis.py      # 🎯 SIMPLE RUNNER
└── test_*.py                           # Test scripts
```

## 🎯 **Quick Start Examples**

### **1. Analyze Current Directory:**
```bash
python3 analyzers/enhanced_import_analysis.py --target . --stats
```

### **2. Auto-Fix with Confirmation:**
```bash
python3 analyzers/intelligent_import_fixer.py --target . --interactive
```

### **3. Dry Run (See What Would Be Fixed):**
```bash
python3 analyzers/intelligent_import_fixer.py --target . --dry-run
```

### **4. Generate Report:**
```bash
python3 run_enhanced_import_analysis.py --target . --output report.json --stats
```

## 🔍 **Which Script to Use When**

### **Use `enhanced_import_analysis.py` when:**
- You want to understand what import issues exist
- You need detailed analysis and reporting
- You want to see statistics and categorization
- You're doing manual review

### **Use `intelligent_import_fixer.py` when:**
- You want to automatically fix import issues
- You're comfortable with confidence-based decisions
- You want maximum automation with safety
- You're doing production cleanup

### **Use `run_enhanced_import_analysis.py` when:**
- You want a simple, straightforward analysis
- You don't need complex pipeline features
- You want minimal dependencies
- You're doing quick checks

### **Use `pipeline_enhanced_import_analysis.py` when:**
- You're integrating with existing pipeline infrastructure
- You need plugin system features
- You want advanced configuration options
- You're doing enterprise-level integration

## 🎯 **Answer to Your Question**

**The main script depends on your use case:**

- **For analysis only:** `analyzers/enhanced_import_analysis.py`
- **For auto-fixing:** `analyzers/intelligent_import_fixer.py` 
- **For simple usage:** `run_enhanced_import_analysis.py`
- **For pipeline integration:** `pipelines/pipeline_enhanced_import_analysis.py`

**Most users should start with:** `analyzers/enhanced_import_analysis.py` for analysis, then `analyzers/intelligent_import_fixer.py` for auto-fixing.