#!/usr/bin/env python3
"""
Enhanced Visualization Demo

Demonstrates the enhanced visualization capabilities of the code interaction mapper
with dead code analysis integration.
"""

import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from map_code_interactions import CodeInteractionMapper


def create_demo_codebase():
    """Create a demo codebase with various dead code patterns."""
    demo_dir = Path("demo_codebase")
    demo_dir.mkdir(exist_ok=True)
    
    # Main module with dead code
    main_file = demo_dir / "main.py"
    main_file.write_text('''
"""
Main module demonstrating various dead code patterns.
"""

import os
import sys
import json  # Unused import
from typing import List, Dict, Optional  # Dict and Optional unused

# Deprecated function
@deprecated(reason="Use new_calculate instead", version="2.0", alternative="new_calculate")
def old_calculate(x, y):
    """Old calculation method."""
    return x + y

# Function with unused parameters
def process_data(data, unused_param, debug=False):
    """Process data with unused parameter."""
    return data.upper()

# Unused function
def helper_function():
    """This function is never called."""
    return "helper"

# Unused variable
unused_var = "never used"

# Function with dynamic import
def load_module_dynamically(module_name):
    """Load module dynamically."""
    return __import__(module_name)

# Function with importlib
def load_with_importlib(module_name):
    """Load module with importlib."""
    import importlib
    return importlib.import_module(module_name)

# Function with unreachable code
def unreachable_example():
    """Function with unreachable code."""
    return "done"
    print("This will never execute")  # Unreachable

# Function with conditional dead code
def conditional_dead():
    """Function with conditional dead code."""
    if False:  # Always false
        print("This code is dead")
    return "alive"

# Used function
def calculate_sum(a, b):
    """Calculate sum of two numbers."""
    return a + b

# Main function
def main():
    """Main function."""
    result = calculate_sum(5, 3)
    old_calculate(1, 2)  # Using deprecated function
    process_data("test", "unused")
    return result

if __name__ == "__main__":
    main()
''')

    # Utility module
    util_file = demo_dir / "utils.py"
    util_file.write_text('''
"""
Utility module with more dead code patterns.
"""

import os
import sys  # Unused import

# Deprecated class
@deprecated(reason="Use NewProcessor instead", version="2.1")
class OldProcessor:
    """Old processor class."""
    
    def process(self, data):
        """Process data."""
        return data

# Unused class
class UnusedClass:
    """This class is never used."""
    
    def method(self):
        """Unused method."""
        return "unused"

# Function with unused return value
def get_config():
    """Get configuration."""
    return {"setting": "value"}

# Function that doesn't use return value
def setup():
    """Setup function."""
    config = get_config()  # Return value not used
    print("Setup complete")

# Used function
def format_output(data):
    """Format output data."""
    return f"Output: {data}"

# Main function
def main():
    """Main function."""
    setup()
    result = format_output("test")
    return result
''')

    # Module with imports
    import_file = demo_dir / "imports.py"
    import_file.write_text('''
"""
Module demonstrating import patterns.
"""

from main import calculate_sum, old_calculate, helper_function
from utils import format_output, OldProcessor
import os
import sys  # Unused import

def example_usage():
    """Example of using imported functions."""
    result = calculate_sum(10, 20)
    old_calculate(5, 5)  # Using deprecated function
    formatted = format_output(result)
    return formatted

def create_processor():
    """Create a processor instance."""
    processor = OldProcessor()  # Using deprecated class
    return processor
''')

    return demo_dir


def run_enhanced_visualization_demo():
    """Run the enhanced visualization demo."""
    print("ENHANCED VISUALIZATION DEMO")
    print("=" * 50)
    
    # Create demo codebase
    demo_dir = create_demo_codebase()
    print(f"Created demo codebase in: {demo_dir}")
    
    # Initialize mapper
    mapper = CodeInteractionMapper(str(demo_dir))
    
    print(f"\nRunning enhanced code interaction analysis...")
    print("-" * 50)
    
    # Run the complete analysis
    report_files = mapper.run()
    
    print(f"\nANALYSIS COMPLETE!")
    print("=" * 30)
    print(f"Report directory: {report_files['report_dir']}")
    print(f"\nGenerated files:")
    print(f"  📄 JSON Report: {Path(report_files['json']).name}")
    print(f"  📄 Summary Report: {Path(report_files['summary']).name}")
    print(f"  🌐 HTML Report: {Path(report_files['html']).name}")
    print(f"  🌐 Enhanced HTML Report: {Path(report_files['enhanced_html']).name}")
    
    # List visual files
    report_dir = Path(report_files['report_dir'])
    visual_files = list(report_dir.glob("*.png"))
    if visual_files:
        print(f"\n📊 Generated Visualizations:")
        for visual_file in visual_files:
            print(f"  📈 {visual_file.name}")
    
    print(f"\n🎯 Key Features Demonstrated:")
    print(f"  ✅ Dead code detection with Vulture integration")
    print(f"  ✅ Deprecated code detection (@deprecated decorators)")
    print(f"  ✅ Dynamic import analysis (__import__, importlib)")
    print(f"  ✅ Conditional dead code detection")
    print(f"  ✅ Impact analysis and prioritization")
    print(f"  ✅ Dependency-aware removal planning")
    print(f"  ✅ Risk assessment and phased removal")
    print(f"  ✅ Enhanced HTML reports with visualizations")
    print(f"  ✅ Multiple chart types (bar, pie, timeline)")
    print(f"  ✅ Interactive dashboards")
    
    print(f"\n📋 What to Check:")
    print(f"  1. Open the enhanced HTML report for interactive analysis")
    print(f"  2. Review the visual charts for dead code patterns")
    print(f"  3. Check the removal plan and recommendations")
    print(f"  4. Examine the impact analysis and risk assessment")
    
    print(f"\n🔍 Sample Analysis Results:")
    
    # Show some sample results
    if 'dead_code' in mapper.results:
        dead_code = mapper.results['dead_code']
        print(f"  📊 Total Dead Code Issues: {dead_code.total_issues}")
        print(f"  ⚠️  Deprecated Code Items: {len(dead_code.deprecated_issues or [])}")
        print(f"  🔴 High Impact Issues: {len(dead_code.issues_by_severity.get('high', []))}")
        print(f"  📏 Potential Lines Removed: {dead_code.potential_savings.get('total_lines', 0)}")
        
        if dead_code.impact_analysis and "removal_plan" in dead_code.impact_analysis:
            removal_plan = dead_code.impact_analysis["removal_plan"]
            time_savings = removal_plan.get('estimated_time_savings', {})
            print(f"  ⏱️  Estimated Time Savings: {time_savings.get('estimated_hours_saved', 0):.1f} hours")
            print(f"  📅 Removal Phases: {len(removal_plan.get('removal_phases', []))}")
    
    print(f"\n✨ Demo complete! Check the generated reports for detailed analysis.")


if __name__ == "__main__":
    run_enhanced_visualization_demo()