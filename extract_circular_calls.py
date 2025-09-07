#!/usr/bin/env python3
"""
Extract circular calls from the interaction mapping results
"""

import json
import sys
from pathlib import Path

def extract_circular_calls_from_main_file():
    """Extract circular calls from the main interaction mapping file"""
    main_file = "/Users/remyroche/Documents/Ares/code_quality/reports/interaction_mapping/interaction_mapping_20250906_012543.json"
    
    try:
        print("Loading main interaction mapping file...")
        with open(main_file, 'r') as f:
            data = json.load(f)
        
        print("Extracting call graph analysis results...")
        call_graph_results = data.get('call_graph_analysis', {}).get('results', {})
        
        # Extract circular calls
        circular_calls = call_graph_results.get('circular_calls', [])
        total_functions = call_graph_results.get('total_functions', 0)
        max_call_depth = call_graph_results.get('max_call_depth', 0)
        
        print(f"\n📊 Call Graph Analysis Results:")
        print(f"Total Functions: {total_functions}")
        print(f"Max Call Depth: {max_call_depth}")
        print(f"Circular Calls: {len(circular_calls)}")
        
        if circular_calls:
            print(f"\n🔄 Circular Calls Found:")
            for i, call in enumerate(circular_calls[:20]):  # Show first 20
                print(f"{i+1:2d}. {call}")
            
            if len(circular_calls) > 20:
                print(f"... and {len(circular_calls) - 20} more circular calls")
        else:
            print("\n✅ No circular calls found!")
        
        return circular_calls, total_functions, max_call_depth
        
    except Exception as e:
        print(f"Error reading main file: {e}")
        return [], 0, 0

def create_detailed_circular_calls_report(circular_calls, total_functions, max_call_depth):
    """Create a detailed HTML report of circular calls"""
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Circular Calls Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .stats {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        .circular-call {{ 
            background-color: #fff3cd; 
            border: 1px solid #ffeaa7; 
            padding: 15px; 
            margin: 10px 0; 
            border-radius: 5px;
            border-left: 4px solid #f39c12;
        }}
        .call-chain {{ 
            font-family: monospace; 
            background-color: #f8f9fa; 
            padding: 10px; 
            border-radius: 3px;
            margin: 5px 0;
            white-space: pre-wrap;
        }}
        .warning {{ background-color: #f8d7da; border: 1px solid #f5c6cb; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        .success {{ background-color: #d4edda; border: 1px solid #c3e6cb; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        .recommendations {{ background-color: #d1ecf1; border: 1px solid #bee5eb; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        .search-box {{ margin: 20px 0; }}
        .search-box input {{ padding: 10px; width: 300px; border: 1px solid #ddd; border-radius: 3px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔄 Circular Calls Analysis Report</h1>
        <p>Generated on: 2025-09-06</p>
        <p>Project: Ares Trading System</p>
    </div>
    
    <div class="stats">
        <h3>📊 Analysis Summary</h3>
        <p><strong>Total Functions Analyzed:</strong> {total_functions:,}</p>
        <p><strong>Maximum Call Depth:</strong> {max_call_depth}</p>
        <p><strong>Circular Calls Found:</strong> {len(circular_calls)}</p>
        <p><strong>Circular Call Rate:</strong> {(len(circular_calls) / max(total_functions, 1) * 100):.2f}%</p>
    </div>
"""
    
    if circular_calls:
        html_content += f"""
    <div class="warning">
        <h3>⚠️ Warning: {len(circular_calls)} Circular Calls Detected</h3>
        <p>Circular calls can indicate potential issues in your code architecture:</p>
        <ul>
            <li><strong>Infinite recursion risks:</strong> Functions calling each other in a loop</li>
            <li><strong>Complex dependencies:</strong> Hard to maintain and understand</li>
            <li><strong>Performance issues:</strong> Potential for stack overflow</li>
            <li><strong>Testing difficulties:</strong> Hard to unit test in isolation</li>
            <li><strong>Debugging challenges:</strong> Complex call stacks</li>
        </ul>
    </div>
    
    <div class="search-box">
        <input type="text" id="searchInput" placeholder="Search circular calls..." onkeyup="filterCalls()">
    </div>
    
    <h2>🔍 Detailed Circular Calls Analysis</h2>
"""
        
        for i, call in enumerate(circular_calls):
            html_content += f"""
    <div class="circular-call" data-call="{call.lower()}">
        <h4>Circular Call #{i+1}</h4>
        <div class="call-chain">{call}</div>
        <p><strong>Risk Level:</strong> {'High' if 'main' in call.lower() or 'init' in call.lower() else 'Medium'}</p>
    </div>
"""
    else:
        html_content += """
    <div class="success">
        <h3>✅ Excellent! No Circular Calls Found</h3>
        <p>Your codebase has a clean architecture with no circular dependencies detected. This is a sign of good software design!</p>
    </div>
"""
    
    html_content += """
    <div class="recommendations">
        <h3>💡 Recommendations for Circular Calls</h3>
        <h4>If you have circular calls, consider these solutions:</h4>
        <ol>
            <li><strong>Extract Common Functionality:</strong> Move shared logic to a separate utility module</li>
            <li><strong>Use Dependency Injection:</strong> Pass dependencies as parameters instead of importing</li>
            <li><strong>Implement Interfaces:</strong> Use abstract base classes to break direct dependencies</li>
            <li><strong>Event-Driven Architecture:</strong> Use events to decouple components</li>
            <li><strong>Factory Pattern:</strong> Use factories to create objects with their dependencies</li>
            <li><strong>Observer Pattern:</strong> Use observers to notify components of changes</li>
            <li><strong>Refactor Large Functions:</strong> Break down complex functions into smaller, focused ones</li>
        </ol>
        
        <h4>Code Review Checklist:</h4>
        <ul>
            <li>Review each circular call to understand why it exists</li>
            <li>Check if the circular dependency is necessary</li>
            <li>Look for opportunities to break the cycle</li>
            <li>Test the refactored code thoroughly</li>
            <li>Document any remaining necessary circular dependencies</li>
        </ul>
    </div>
    
    <script>
        function filterCalls() {
            const input = document.getElementById('searchInput');
            const filter = input.value.toLowerCase();
            const calls = document.querySelectorAll('.circular-call');
            
            calls.forEach(call => {
                const callText = call.getAttribute('data-call');
                if (callText.includes(filter)) {
                    call.style.display = 'block';
                } else {
                    call.style.display = 'none';
                }
            });
        }
    </script>
</body>
</html>
"""
    
    output_path = "/Users/remyroche/Documents/Ares/code_quality/reports/interaction_mapping/detailed_circular_calls_report.html"
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Created detailed circular calls report: {output_path}")
    return output_path

def main():
    """Main function to extract and report circular calls"""
    print("🔍 Extracting circular calls from interaction mapping results...")
    
    circular_calls, total_functions, max_call_depth = extract_circular_calls_from_main_file()
    
    if circular_calls:
        report_path = create_detailed_circular_calls_report(circular_calls, total_functions, max_call_depth)
        
        # Open the report
        import subprocess
        subprocess.run(["open", report_path])
        
        print(f"\n✅ Circular calls analysis complete!")
        print(f"📄 Detailed report: {report_path}")
    else:
        print("\n✅ No circular calls found - your codebase has clean architecture!")

if __name__ == "__main__":
    main()
