# 📁 Output Locations Guide

## Where to Find Your Analysis Reports and Visualizations

When you run the enhanced code interaction analysis, all outputs are saved to a timestamped directory. Here's exactly where to find everything:

## 🗂️ **Main Output Directory Structure**

```
code_quality/visualizers/reports/report_YYYYMMDD_HHMMSS/
├── 📄 interactions_YYYYMMDD_HHMMSS.json          # Raw analysis data
├── 📄 interactions_summary_YYYYMMDD_HHMMSS.txt   # Text summary report
├── 🌐 interactions_YYYYMMDD_HHMMSS.html          # Basic HTML report
├── 🌐 enhanced_interactions_YYYYMMDD_HHMMSS.html # **Enhanced HTML report**
├── 📊 dead_code_types_YYYYMMDD_HHMMSS.png        # Dead code by type chart
├── 📊 dead_code_severity_YYYYMMDD_HHMMSS.png     # Severity distribution chart
├── 📊 deprecated_code_YYYYMMDD_HHMMSS.png        # Deprecated code analysis
├── 📊 impact_analysis_YYYYMMDD_HHMMSS.png        # Impact analysis chart
├── 📊 removal_plan_YYYYMMDD_HHMMSS.png           # Removal plan timeline
├── 📊 function_usage_map_YYYYMMDD_HHMMSS.png     # **Function usage mapping**
├── 📊 dependencies_YYYYMMDD_HHMMSS.png           # Module dependencies
├── 📊 circular_deps_YYYYMMDD_HHMMSS.png          # Circular dependencies
├── 📊 complexity_heatmap_YYYYMMDD_HHMMSS.png     # Complexity heatmap
├── 📊 function_network_YYYYMMDD_HHMMSS.png       # Function call network
└── 🌐 dashboard_YYYYMMDD_HHMMSS.html             # Interactive dashboard
```

## 🎯 **Key Files to Check**

### **1. Enhanced HTML Report** ⭐ **MOST IMPORTANT**
- **Location**: `code_quality/visualizers/reports/report_YYYYMMDD_HHMMSS/enhanced_interactions_YYYYMMDD_HHMMSS.html`
- **What it contains**: 
  - Interactive dashboard with all analysis results
  - Color-coded sections for different analysis types
  - Summary cards with key metrics
  - Detailed issue listings with code snippets
  - Removal plan with timeline and recommendations
- **How to open**: Double-click the file or open in any web browser

### **2. Function Usage Mapping** ⭐ **NEW FEATURE**
- **Location**: `code_quality/visualizers/reports/report_YYYYMMDD_HHMMSS/function_usage_map_YYYYMMDD_HHMMSS.png`
- **What it contains**:
  - **4-panel visualization**:
    1. **Function Usage Heatmap**: Shows which functions are called most/least
    2. **Dead vs Used Functions**: Pie chart of function usage distribution
    3. **Function Call Network**: Visual network of function relationships
    4. **Usage Statistics**: Bar chart of usage patterns
- **How to view**: Open with any image viewer or web browser

### **3. Dead Code Analysis Charts**
- **Dead Code Types**: `dead_code_types_YYYYMMDD_HHMMSS.png`
- **Severity Distribution**: `dead_code_severity_YYYYMMDD_HHMMSS.png`
- **Deprecated Code**: `deprecated_code_YYYYMMDD_HHMMSS.png`
- **Impact Analysis**: `impact_analysis_YYYYMMDD_HHMMSS.png`
- **Removal Plan**: `removal_plan_YYYYMMDD_HHMMSS.png`

## 🚀 **How to Run and Find Your Reports**

### **Method 1: Using the Main Script**
```bash
# Run from the workspace root
python code_quality/map_code_interactions.py

# The script will output the exact path:
# "All reports saved to: code_quality/visualizers/reports/report_YYYYMMDD_HHMMSS"
```

### **Method 2: Using the Demo Script**
```bash
# Run the enhanced visualization demo
python code_quality/examples/enhanced_visualization_demo.py

# This creates a demo codebase and shows you exactly where files are saved
```

### **Method 3: Using the CLI Tools**
```bash
# Run comprehensive analysis
python data_quality/mapping/cli.py comprehensive /path/to/your/code --output-dir my_analysis

# Files will be in: my_analysis/
```

## 📍 **Finding Your Reports**

### **Step 1: Look for the Console Output**
When you run the analysis, you'll see output like this:
```
CODE INTERACTION MAPPING COMPLETE!
================================================================================

All reports saved to: code_quality/visualizers/reports/report_20241203_143022

Generated files:
  - JSON: interactions_20241203_143022.json
  - SUMMARY: interactions_summary_20241203_143022.txt
  - HTML: interactions_20241203_143022.html
  - ENHANCED_HTML: enhanced_interactions_20241203_143022.html
```

### **Step 2: Navigate to the Directory**
```bash
# From workspace root
cd code_quality/visualizers/reports/report_YYYYMMDD_HHMMSS

# List all files
ls -la
```

### **Step 3: Open the Enhanced HTML Report**
```bash
# Open in default browser (Linux/Mac)
open enhanced_interactions_YYYYMMDD_HHMMSS.html

# Or just double-click the file in your file manager
```

## 🎨 **What You'll See in the Enhanced HTML Report**

### **📊 Analysis Summary Section**
- **4 summary cards** showing:
  - Total Dead Code Issues
  - Deprecated Code Items
  - High Impact Issues
  - Potential Lines Removed

### **💀 Dead Code Analysis Section**
- **High Priority Issues** (red border)
- **Medium Priority Issues** (orange border)
- **Low Priority Issues** (green border)
- **Code snippets** for each issue
- **Confidence scores** and impact levels

### **⚠️ Deprecated Code Section**
- **Deprecation details** with reasons
- **Removal versions** and alternatives
- **Code snippets** showing deprecated functions

### **📈 Impact Analysis Section**
- **Impact distribution** charts
- **Total impact score** visualization
- **Risk categorization**

### **🗓️ Removal Plan Section**
- **Time savings estimates** (hours, days, lines)
- **Phase timeline** with effort estimates
- **Risk assessment** with color-coded indicators
- **Recommended approach**

### **💡 Recommendations Section**
- **Prioritized action items**
- **Best practices** for cleanup
- **Risk mitigation strategies**

## 🔍 **Function Usage Mapping Details**

The new function usage mapping shows:

### **Panel 1: Usage Heatmap**
- **Rows**: Top 20 functions
- **Columns**: Times Called, Calls Made, Is Dead, Is Deprecated, Impact Score
- **Colors**: Red (high usage/impact) to Green (low usage/impact)

### **Panel 2: Dead vs Used Distribution**
- **Pie chart** showing:
  - Used Functions (green)
  - Dead Functions (red)
  - Deprecated Functions (orange)
  - Unused Functions (gray)

### **Panel 3: Call Network**
- **Circular layout** of functions
- **Lines** showing call relationships
- **Thickness** indicates call frequency

### **Panel 4: Usage Statistics**
- **Bar chart** showing:
  - Highly Used (>5 calls)
  - Moderately Used (1-5 calls)
  - Unused (0 calls)

## 🛠️ **Troubleshooting**

### **If you can't find the reports:**
1. **Check the console output** for the exact path
2. **Look in**: `code_quality/visualizers/reports/`
3. **Find the most recent** `report_YYYYMMDD_HHMMSS` folder
4. **Make sure the analysis completed** without errors

### **If visualizations are missing:**
1. **Check if matplotlib is installed**: `pip install matplotlib`
2. **Look for error messages** in the console output
3. **The HTML report will still be generated** even without charts

### **If the enhanced HTML report is empty:**
1. **Check if dead code analysis ran** successfully
2. **Look for error messages** in the console
3. **Try running the demo script** first to test

## 📱 **Mobile-Friendly Reports**

The enhanced HTML reports are **responsive** and work on:
- **Desktop browsers** (Chrome, Firefox, Safari, Edge)
- **Mobile devices** (phones, tablets)
- **Tablet browsers** with touch support

## 🔄 **Updating Reports**

To generate new reports:
1. **Delete old report folders** if needed
2. **Run the analysis again**
3. **New timestamped folder** will be created
4. **All files will be regenerated** with latest data

---

**💡 Pro Tip**: Bookmark the enhanced HTML report in your browser for easy access to your analysis results!