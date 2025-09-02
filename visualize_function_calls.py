#!/usr/bin/env python3
"""
Function Call Visualization Tool
Converts DOT files to various visual formats and provides interactive analysis.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

class FunctionCallVisualizer:
    def __init__(self, dot_file="function_calls.dot", json_file="function_calls_report.json"):
        self.dot_file = dot_file
        self.json_file = json_file
        self.report_data = None
        
    def load_report(self):
        """Load the function calls report."""
        try:
            with open(self.json_file, 'r') as f:
                self.report_data = json.load(f)
            print(f"✅ Loaded report: {self.json_file}")
            return True
        except FileNotFoundError:
            print(f"❌ Report file not found: {self.json_file}")
            return False
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON in: {self.json_file}")
            return False
    
    def check_graphviz(self):
        """Check if Graphviz is available."""
        try:
            result = subprocess.run(['dot', '-V'], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Graphviz is available")
                return True
            else:
                print("❌ Graphviz not working properly")
                return False
        except FileNotFoundError:
            print("❌ Graphviz not installed. Install with: sudo apt-get install graphviz")
            return False
    
    def convert_dot_to_image(self, output_format="png"):
        """Convert DOT file to image format."""
        if not self.check_graphviz():
            return False
        
        output_file = f"function_calls.{output_format}"
        
        try:
            cmd = ['dot', f'-T{output_format}', self.dot_file, '-o', output_file]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Generated {output_file}")
                return True
            else:
                print(f"❌ Error generating {output_file}: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def generate_all_formats(self):
        """Generate all common image formats."""
        formats = ['png', 'svg', 'pdf', 'jpg']
        
        print("🔄 Generating all image formats...")
        for fmt in formats:
            self.convert_dot_to_image(fmt)
    
    def create_simplified_graph(self, max_functions=50):
        """Create a simplified graph with fewer functions for better visualization."""
        if not self.report_data:
            print("❌ No report data loaded")
            return
        
        # Get top functions by call count
        most_called = self.report_data.get('most_called_functions', [])
        most_calling = self.report_data.get('most_calling_functions', [])
        
        # Create simplified DOT content
        dot_content = ["digraph SimplifiedFunctionCalls {"]
        dot_content.append("  rankdir=LR;")
        dot_content.append("  node [shape=box, fontsize=10];")
        dot_content.append("  edge [fontsize=8];")
        
        # Add nodes for top functions
        top_functions = set()
        for func, _ in most_called[:max_functions//2]:
            top_functions.add(func)
        for func, _ in most_calling[:max_functions//2]:
            top_functions.add(func)
        
        # Add nodes
        for func in top_functions:
            if func in self.report_data.get('function_definitions', {}):
                file_path = self.report_data['function_definitions'][func]
                short_path = Path(file_path).name
                dot_content.append(f'  "{func}" [label="{func}\\n{short_path}"];')
            else:
                dot_content.append(f'  "{func}" [label="{func}\\n(undefined)"];')
        
        # Add edges for call relationships
        call_graph = self.report_data.get('call_graph', {})
        edge_count = 0
        max_edges = 100  # Limit edges for readability
        
        for caller, callees in call_graph.items():
            if caller in top_functions and edge_count < max_edges:
                for callee in callees:
                    if callee in top_functions:
                        dot_content.append(f'  "{caller}" -> "{callee}";')
                        edge_count += 1
                        if edge_count >= max_edges:
                            break
            if edge_count >= max_edges:
                break
        
        dot_content.append("}")
        
        # Save simplified graph
        simplified_file = "simplified_function_calls.dot"
        with open(simplified_file, 'w') as f:
            f.write('\n'.join(dot_content))
        
        print(f"✅ Simplified graph saved to {simplified_file}")
        return simplified_file
    
    def generate_call_chain_report(self):
        """Generate a text report of call chains."""
        if not self.report_data:
            print("❌ No report data loaded")
            return
        
        call_chains = self.report_data.get('call_chains', [])
        
        print(f"\n{'='*60}")
        print("CALL CHAIN ANALYSIS")
        print(f"{'='*60}")
        
        # Group chains by length
        chains_by_length = {}
        for chain in call_chains:
            length = len(chain)
            if length not in chains_by_length:
                chains_by_length[length] = []
            chains_by_length[length].append(chain)
        
        for length in sorted(chains_by_length.keys()):
            print(f"\n📏 Call chains of length {length}:")
            for i, chain in enumerate(chains_by_length[length][:10], 1):  # Show first 10
                print(f"   {i:2d}. {' -> '.join(chain)}")
            if len(chains_by_length[length]) > 10:
                print(f"   ... and {len(chains_by_length[length]) - 10} more")
    
    def generate_function_heatmap_data(self):
        """Generate data for creating a function call heatmap."""
        if not self.report_data:
            print("❌ No report data loaded")
            return
        
        # Create a matrix of function calls
        functions = list(self.report_data.get('function_definitions', {}).keys())
        call_graph = self.report_data.get('call_graph', {})
        
        # Create CSV for heatmap
        csv_content = ["Caller,Function,Calls"]
        
        for caller in functions:
            if caller in call_graph:
                for callee in call_graph[caller]:
                    csv_content.append(f"{caller},{callee},1")
        
        with open("function_calls_heatmap.csv", 'w') as f:
            f.write('\n'.join(csv_content))
        
        print("✅ Heatmap data saved to function_calls_heatmap.csv")
    
    def print_interactive_analysis(self):
        """Print interactive analysis options."""
        if not self.report_data:
            print("❌ No report data loaded")
            return
        
        print(f"\n{'='*60}")
        print("INTERACTIVE ANALYSIS OPTIONS")
        print(f"{'='*60}")
        
        summary = self.report_data.get('summary', {})
        print(f"📊 Current Analysis:")
        print(f"   • Functions: {summary.get('total_functions', 0)}")
        print(f"   • Function calls: {summary.get('total_function_calls', 0)}")
        print(f"   • Classes: {summary.get('total_classes', 0)}")
        
        print(f"\n🔍 Analysis Commands:")
        print(f"   • python3 visualize_function_calls.py --heatmap")
        print(f"   • python3 visualize_function_calls.py --chains")
        print(f"   • python3 visualize_function_calls.py --simplified")
        print(f"   • python3 visualize_function_calls.py --all-formats")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize function call relationships")
    parser.add_argument('--heatmap', action='store_true', help='Generate heatmap data')
    parser.add_argument('--chains', action='store_true', help='Show call chains')
    parser.add_argument('--simplified', action='store_true', help='Create simplified graph')
    parser.add_argument('--all-formats', action='store_true', help='Generate all image formats')
    parser.add_argument('--dot-file', default='function_calls.dot', help='Input DOT file')
    parser.add_argument('--json-file', default='function_calls_report.json', help='Input JSON file')
    
    args = parser.parse_args()
    
    visualizer = FunctionCallVisualizer(args.dot_file, args.json_file)
    
    if not visualizer.load_report():
        return
    
    if args.heatmap:
        visualizer.generate_function_heatmap_data()
    elif args.chains:
        visualizer.generate_call_chain_report()
    elif args.simplified:
        simplified_file = visualizer.create_simplified_graph()
        if simplified_file:
            visualizer.dot_file = simplified_file
            visualizer.convert_dot_to_image('png')
    elif args.all_formats:
        visualizer.generate_all_formats()
    else:
        # Default: show interactive options
        visualizer.print_interactive_analysis()
        
        # Also generate basic PNG
        if visualizer.check_graphviz():
            visualizer.convert_dot_to_image('png')
            print("\n💡 Use --help for more options")

if __name__ == "__main__":
    main()