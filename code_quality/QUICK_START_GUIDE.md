# 🚀 Quick Start Guide - Enhanced Dead Code Analysis

## Get Started in 3 Steps

### **Step 1: Run the Analysis**
```bash
# From your workspace root directory
python code_quality/map_code_interactions.py
```

### **Step 2: Find Your Reports**
Look for this output in your console:
```
All reports saved to: code_quality/visualizers/reports/report_YYYYMMDD_HHMMSS
```

### **Step 3: Open the Enhanced Report**
Navigate to the folder and open:
```
enhanced_interactions_YYYYMMDD_HHMMSS.html
```

## 🎯 **What You'll Get**

### **📊 Visual Charts** (PNG files)
- `function_usage_map_YYYYMMDD_HHMMSS.png` - **Function usage mapping**
- `dead_code_types_YYYYMMDD_HHMMSS.png` - Dead code by type
- `dead_code_severity_YYYYMMDD_HHMMSS.png` - Severity distribution
- `deprecated_code_YYYYMMDD_HHMMSS.png` - Deprecated code analysis
- `impact_analysis_YYYYMMDD_HHMMSS.png` - Impact analysis
- `removal_plan_YYYYMMDD_HHMMSS.png` - Removal plan timeline

### **🌐 Interactive Reports** (HTML files)
- `enhanced_interactions_YYYYMMDD_HHMMSS.html` - **Main interactive report**
- `interactions_YYYYMMDD_HHMMSS.html` - Basic report
- `dashboard_YYYYMMDD_HHMMSS.html` - Interactive dashboard

### **📄 Data Files** (JSON/TXT)
- `interactions_YYYYMMDD_HHMMSS.json` - Raw analysis data
- `interactions_summary_YYYYMMDD_HHMMSS.txt` - Text summary

## 🔍 **Function Usage Mapping - What You'll See**

The new function usage mapping shows **4 panels**:

### **Panel 1: Usage Heatmap** 🔥
- **Rows**: Your functions (top 20)
- **Columns**: Usage metrics
- **Colors**: Red (high usage) → Green (low usage)
- **Shows**: Which functions are called most/least

### **Panel 2: Dead vs Used** 🥧
- **Pie chart** showing function distribution
- **Green**: Used functions
- **Red**: Dead functions  
- **Orange**: Deprecated functions
- **Gray**: Unused functions

### **Panel 3: Call Network** 🕸️
- **Circular layout** of functions
- **Lines**: Show which functions call each other
- **Thickness**: How often they're called
- **Visual**: See your code's call relationships

### **Panel 4: Usage Statistics** 📊
- **Bar chart** of usage patterns
- **Highly Used**: Functions called >5 times
- **Moderately Used**: Functions called 1-5 times
- **Unused**: Functions never called

## 🎨 **Enhanced HTML Report Features**

### **📊 Analysis Summary**
- **4 summary cards** with key metrics
- **Color-coded** by importance
- **Gradient backgrounds** for visual appeal

### **💀 Dead Code Analysis**
- **Priority-based** issue listing
- **Code snippets** for each issue
- **Confidence scores** and impact levels
- **Color-coded borders** (red/orange/green)

### **⚠️ Deprecated Code**
- **Deprecation reasons** and versions
- **Alternative suggestions**
- **Removal timelines**

### **📈 Impact Analysis**
- **Impact distribution** charts
- **Total impact score**
- **Risk categorization**

### **🗓️ Removal Plan**
- **Time savings estimates**
- **Phase timeline** with effort estimates
- **Risk assessment** with color indicators
- **Actionable recommendations**

## 🛠️ **Troubleshooting**

### **Missing Visualizations?**
```bash
# Install matplotlib
pip install matplotlib

# Run analysis again
python code_quality/map_code_interactions.py
```

### **Can't Find Reports?**
1. **Check console output** for the exact path
2. **Look in**: `code_quality/visualizers/reports/`
3. **Find the most recent** `report_YYYYMMDD_HHMMSS` folder

### **Empty Reports?**
1. **Make sure you have Python files** in your project
2. **Check for error messages** in console
3. **Try the demo first**: `python code_quality/examples/enhanced_visualization_demo.py`

## 🎯 **Demo Mode**

Want to see it in action first?
```bash
# Run the demo with sample code
python code_quality/examples/enhanced_visualization_demo.py
```

This creates sample code and shows you exactly what the analysis looks like.

## 📱 **Mobile Friendly**

All reports work on:
- **Desktop** (Chrome, Firefox, Safari, Edge)
- **Mobile** (phones, tablets)
- **Touch devices**

## 🔄 **Generate New Reports**

To analyze different code or get updated results:
```bash
# Just run again - new timestamped folder will be created
python code_quality/map_code_interactions.py
```

## 💡 **Pro Tips**

1. **Bookmark** the enhanced HTML report in your browser
2. **Share** the report folder with your team
3. **Use the function usage map** to identify critical functions
4. **Follow the removal plan** for safe code cleanup
5. **Check the recommendations** for best practices

---

**🎉 That's it! You now have comprehensive dead code analysis with beautiful visualizations and actionable insights.**