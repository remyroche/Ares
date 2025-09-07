#!/usr/bin/env python3
"""
Extract detailed circular calls with actual function names from the interaction mapping results
"""

import json
import re
from pathlib import Path

def extract_detailed_circular_calls():
    """Extract circular calls with actual function names and file locations"""
    main_file = "/Users/remyroche/Documents/Ares/code_quality/reports/interaction_mapping/interaction_mapping_20250906_012543.json"
    
    try:
        print("Loading main interaction mapping file...")
        with open(main_file, 'r') as f:
            data = json.load(f)
        
        print("Extracting detailed call graph analysis results...")
        call_graph_results = data.get('call_graph_analysis', {}).get('results', {})
        
        # Extract functions data
        functions = call_graph_results.get('functions', {})
        circular_calls = call_graph_results.get('circular_calls', [])
        
        print(f"\n📊 Detailed Call Graph Analysis:")
        print(f"Total Functions: {len(functions)}")
        print(f"Circular Calls: {len(circular_calls)}")
        
        # Process circular calls to get detailed information
        detailed_circular_calls = []
        
        for i, call in enumerate(circular_calls):
            # Parse the circular call to extract function names
            if ' -> ' in call:
                parts = call.split(' -> ')
                if len(parts) >= 2:
                    source_func = parts[0].strip()
                    target_func = parts[-1].strip()
                    
                    # Find source function details
                    source_details = functions.get(source_func, {})
                    target_details = functions.get(target_func, {})
                    
                    detailed_call = {
                        'id': i + 1,
                        'source_function': source_func,
                        'target_function': target_func,
                        'source_file': source_details.get('file_path', 'Unknown'),
                        'target_file': target_details.get('file_path', 'Unknown'),
                        'source_line': source_details.get('line_number', 'Unknown'),
                        'target_line': target_details.get('line_number', 'Unknown'),
                        'source_calls': source_details.get('calls', []),
                        'target_calls': target_details.get('calls', []),
                        'call_chain': call
                    }
                    
                    detailed_circular_calls.append(detailed_call)
        
        return detailed_circular_calls, functions
        
    except Exception as e:
        print(f"Error reading main file: {e}")
        return [], {}

def create_enhanced_circular_calls_report(detailed_calls, functions):
    """Create an enhanced HTML report with detailed function information"""
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Enhanced Circular Calls Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .stats {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        .circular-call {{ 
            background-color: #fff3cd; 
            border: 1px solid #ffeaa7; 
            padding: 20px; 
            margin: 15px 0; 
            border-radius: 5px;
            border-left: 4px solid #f39c12;
        }}
        .function-details {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            border: 1px solid #dee2e6;
        }}
        .function-name {{ 
            font-weight: bold; 
            color: #e74c3c; 
            font-family: monospace;
            font-size: 16px;
        }}
        .file-path {{ 
            color: #6c757d; 
            font-family: monospace; 
            font-size: 12px;
            margin: 5px 0;
        }}
        .call-chain {{ 
            font-family: monospace; 
            background-color: #e9ecef; 
            padding: 10px; 
            border-radius: 3px;
            margin: 10px 0;
            border-left: 3px solid #007bff;
        }}
        .calls-list {{
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 3px;
            margin: 5px 0;
            font-family: monospace;
            font-size: 12px;
        }}
        .warning {{ background-color: #f8d7da; border: 1px solid #f5c6cb; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        .search-box {{ margin: 20px 0; }}
        .search-box input {{ padding: 10px; width: 300px; border: 1px solid #ddd; border-radius: 3px; }}
        .risk-high {{ border-left-color: #dc3545 !important; }}
        .risk-medium {{ border-left-color: #ffc107 !important; }}
        .risk-low {{ border-left-color: #28a745 !important; }}
        .tabs {{ margin: 20px 0; }}
        .tab {{ padding: 10px 20px; background: #e9ecef; border: none; cursor: pointer; margin-right: 5px; }}
        .tab.active {{ background: #007bff; color: white; }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔄 Enhanced Circular Calls Analysis Report</h1>
        <p>Generated on: 2025-09-06</p>
        <p>Project: Ares Trading System</p>
    </div>
    
    <div class="stats">
        <h3>📊 Analysis Summary</h3>
        <p><strong>Total Functions Analyzed:</strong> {len(functions):,}</p>
        <p><strong>Circular Calls Found:</strong> {len(detailed_calls)}</p>
        <p><strong>Circular Call Rate:</strong> {(len(detailed_calls) / max(len(functions), 1) * 100):.2f}%</p>
    </div>
"""
    
    if detailed_calls:
        html_content += f"""
    <div class="warning">
        <h3>⚠️ Warning: {len(detailed_calls)} Circular Calls Detected</h3>
        <p>These functions call each other in a circular pattern, which can lead to:</p>
        <ul>
            <li><strong>Infinite recursion</strong> if not properly handled</li>
            <li><strong>Stack overflow</strong> in recursive scenarios</li>
            <li><strong>Complex debugging</strong> due to circular call stacks</li>
            <li><strong>Testing difficulties</strong> when trying to isolate functions</li>
        </ul>
    </div>
    
    <div class="search-box">
        <input type="text" id="searchInput" placeholder="Search by function name, file, or call chain..." onkeyup="filterCalls()">
    </div>
    
    <div class="tabs">
        <button class="tab active" onclick="showTab('all')">All Circular Calls ({len(detailed_calls)})</button>
        <button class="tab" onclick="showTab('high-risk')">High Risk</button>
        <button class="tab" onclick="showTab('medium-risk')">Medium Risk</button>
        <button class="tab" onclick="showTab('low-risk')">Low Risk</button>
    </div>
    
    <div id="all" class="tab-content active">
        <h2>🔍 All Circular Calls</h2>
"""
        
        for call in detailed_calls:
            # Determine risk level
            risk_level = "low"
            if any(keyword in call['source_function'].lower() for keyword in ['main', 'init', 'start', 'run']):
                risk_level = "high"
            elif any(keyword in call['source_function'].lower() for keyword in ['process', 'handle', 'execute']):
                risk_level = "medium"
            
            risk_class = f"risk-{risk_level}"
            
            html_content += f"""
        <div class="circular-call {risk_class}" data-call="{call['source_function'].lower()} {call['target_function'].lower()}" data-risk="{risk_level}">
            <h3>🔄 Circular Call #{call['id']} - {call['source_function']} ↔ {call['target_function']}</h3>
            
            <div class="call-chain">
                <strong>Call Chain:</strong> {call['call_chain']}
            </div>
            
            <div style="display: flex; gap: 20px; margin: 15px 0;">
                <div class="function-details" style="flex: 1;">
                    <h4>📤 Source Function</h4>
                    <div class="function-name">{call['source_function']}</div>
                    <div class="file-path">📁 {call['source_file']}</div>
                    <div class="file-path">📍 Line: {call['source_line']}</div>
                    <div class="calls-list">
                        <strong>Calls:</strong> {', '.join(call['source_calls'][:10])}
                        {f'... and {len(call["source_calls"]) - 10} more' if len(call['source_calls']) > 10 else ''}
                    </div>
                </div>
                
                <div class="function-details" style="flex: 1;">
                    <h4>📥 Target Function</h4>
                    <div class="function-name">{call['target_function']}</div>
                    <div class="file-path">📁 {call['target_file']}</div>
                    <div class="file-path">📍 Line: {call['target_line']}</div>
                    <div class="calls-list">
                        <strong>Calls:</strong> {', '.join(call['target_calls'][:10])}
                        {f'... and {len(call["target_calls"]) - 10} more' if len(call['target_calls']) > 10 else ''}
                    </div>
                </div>
            </div>
            
            <div style="background-color: #e9ecef; padding: 10px; border-radius: 3px; margin-top: 10px;">
                <strong>Risk Level:</strong> <span style="color: {'#dc3545' if risk_level == 'high' else '#ffc107' if risk_level == 'medium' else '#28a745'}">{risk_level.upper()}</span>
                <br><strong>Issue:</strong> {call['source_function']} and {call['target_function']} call each other, creating a circular dependency
            </div>
        </div>
"""
        
        html_content += """
    </div>
    
    <div id="high-risk" class="tab-content">
        <h2>🔴 High Risk Circular Calls</h2>
        <p>These involve critical functions like main, init, start, or run functions.</p>
    </div>
    
    <div id="medium-risk" class="tab-content">
        <h2>🟡 Medium Risk Circular Calls</h2>
        <p>These involve process, handle, or execute functions.</p>
    </div>
    
    <div id="low-risk" class="tab-content">
        <h2>🟢 Low Risk Circular Calls</h2>
        <p>These involve utility or helper functions.</p>
    </div>
"""
    else:
        html_content += """
    <div class="success">
        <h3>✅ Excellent! No Circular Calls Found</h3>
        <p>Your codebase has a clean architecture with no circular dependencies detected.</p>
    </div>
"""
    
    html_content += """
    <div style="background-color: #d1ecf1; border: 1px solid #bee5eb; padding: 20px; border-radius: 5px; margin: 20px 0;">
        <h3>💡 How to Fix Circular Calls</h3>
        <h4>Common Solutions:</h4>
        <ol>
            <li><strong>Extract Common Logic:</strong> Move shared functionality to a separate utility module</li>
            <li><strong>Dependency Injection:</strong> Pass dependencies as parameters instead of importing</li>
            <li><strong>Interface Segregation:</strong> Use abstract base classes to break direct dependencies</li>
            <li><strong>Event-Driven Architecture:</strong> Use events to decouple components</li>
            <li><strong>Factory Pattern:</strong> Use factories to create objects with their dependencies</li>
            <li><strong>Lazy Loading:</strong> Import modules only when needed</li>
        </ol>
        
        <h4>Example Refactoring:</h4>
        <pre style="background: #f8f9fa; padding: 15px; border-radius: 3px; overflow-x: auto;">
# Before (Circular):
# file_a.py
from file_b import function_b
def function_a():
def function_a():
    pass
    return function_b()

# file_b.py  
from file_a import function_a
def function_b():
def function_b():
    pass
    return function_a()

# After (Fixed):
# common.py
def shared_logic():
def shared_logic():
    pass
    return "shared result"

# file_a.py
from common import shared_logic
def function_a():
def function_a():
    pass
    return shared_logic()

# file_b.py
from common import shared_logic  
def function_b():
def function_b():
    pass
    return shared_logic()
        </pre>
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
        
        function showTab(tabName) {
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            
            // Remove active class from all tabs
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected tab content
            document.getElementById(tabName).classList.add('active');
            
            // Add active class to clicked tab
            event.target.classList.add('active');
            
            // Filter calls based on risk level
            const calls = document.querySelectorAll('.circular-call');
            calls.forEach(call => {
                if (tabName === 'all') {
                    call.style.display = 'block';
                } else {
                    const risk = call.getAttribute('data-risk');
                    if (risk === tabName.replace('-risk', '')) {
                        call.style.display = 'block';
                    } else {
                        call.style.display = 'none';
                    }
                }
            });
        }
    </script>
</body>
</html>
"""
    
    output_path = "/Users/remyroche/Documents/Ares/code_quality/reports/interaction_mapping/enhanced_circular_calls_report.html"
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Created enhanced circular calls report: {output_path}")
    return output_path

def main():
    """Main function to extract and report detailed circular calls"""
    print("🔍 Extracting detailed circular calls with function names...")
    
    detailed_calls, functions = extract_detailed_circular_calls()
    
    if detailed_calls:
        report_path = create_enhanced_circular_calls_report(detailed_calls, functions)
        
        # Open the report
        import subprocess
        subprocess.run(["open", report_path])
        
        print(f"\n✅ Enhanced circular calls analysis complete!")
        print(f"📄 Detailed report: {report_path}")
        
        # Print summary
        print(f"\n📊 Summary:")
        for i, call in enumerate(detailed_calls[:10]):
            print(f"{i+1:2d}. {call['source_function']} ↔ {call['target_function']}")
            print(f"    📁 {call['source_file']}:{call['source_line']} ↔ {call['target_file']}:{call['target_line']}")
        
        if len(detailed_calls) > 10:
            print(f"... and {len(detailed_calls) - 10} more circular calls")
    else:
        print("\n✅ No circular calls found - your codebase has clean architecture!")

if __name__ == "__main__":
    main()
