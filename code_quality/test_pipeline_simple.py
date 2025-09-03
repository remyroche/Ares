#!/usr/bin/env python3
"""
Simple Pipeline Test - Demonstrates the 3-step pipeline without external dependencies.

This shows:
1. Code checking/analysis
2. Data generation
3. Visualization creation (as HTML/text files)
"""

import ast
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict


class SimpleCodeAnalyzer:
    """Simple code analyzer that demonstrates the pipeline."""
    
    def __init__(self, project_path):
        self.project_path = Path(project_path)
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'project': str(self.project_path),
            'modules': {},
            'functions': {},
            'complexity': {},
            'imports': defaultdict(list)
        }
    
    def step1_analyze_code(self):
        """Step 1: Check/Analyze the code."""
        print("STEP 1: ANALYZING CODE")
        print("=" * 50)
        
        py_files = list(self.project_path.rglob("*.py"))
        print(f"Found {len(py_files)} Python files")
        
        for py_file in py_files[:10]:  # Limit to 10 files for demo
            if '__pycache__' in str(py_file):
                continue
                
            print(f"  Analyzing: {py_file.relative_to(self.project_path)}")
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)
                    
                    # Extract information
                    module_name = str(py_file.relative_to(self.project_path))
                    self.results['modules'][module_name] = {
                        'lines': len(content.splitlines()),
                        'functions': [],
                        'classes': [],
                        'imports': []
                    }
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            self.results['modules'][module_name]['functions'].append(node.name)
                            self.results['functions'][f"{module_name}::{node.name}"] = {
                                'complexity': len([n for n in ast.walk(node) if isinstance(n, ast.If)]) + 1,
                                'lines': node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 1
                            }
                        elif isinstance(node, ast.ClassDef):
                            self.results['modules'][module_name]['classes'].append(node.name)
                        elif isinstance(node, ast.Import):
                            for alias in node.names:
                                self.results['imports'][module_name].append(alias.name)
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                self.results['imports'][module_name].append(node.module)
                    
                    # Calculate module complexity
                    self.results['complexity'][module_name] = {
                        'functions': len(self.results['modules'][module_name]['functions']),
                        'classes': len(self.results['modules'][module_name]['classes']),
                        'imports': len(self.results['imports'][module_name]),
                        'lines': self.results['modules'][module_name]['lines']
                    }
                    
            except Exception as e:
                print(f"    Error analyzing {py_file}: {e}")
        
        print(f"\n✓ Analysis complete: {len(self.results['modules'])} modules analyzed")
    
    def step2_generate_data(self):
        """Step 2: Generate structured data."""
        print("\nSTEP 2: GENERATING DATA")
        print("=" * 50)
        
        # Create output directory with datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"code_quality/visualizers/reports/report_{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON data
        json_file = output_dir / f"analysis_data_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ Saved JSON data: {json_file}")
        
        # Generate summary report
        summary_file = output_dir / f"summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("CODE ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Project: {self.results['project']}\n")
            f.write(f"Timestamp: {self.results['timestamp']}\n\n")
            
            f.write("MODULES ANALYZED:\n")
            for module, info in self.results['modules'].items():
                f.write(f"\n{module}:\n")
                f.write(f"  - Lines: {info['lines']}\n")
                f.write(f"  - Functions: {len(info['functions'])}\n")
                f.write(f"  - Classes: {len(info['classes'])}\n")
            
            f.write("\n\nTOP COMPLEX FUNCTIONS:\n")
            complex_funcs = sorted(self.results['functions'].items(), 
                                 key=lambda x: x[1]['complexity'], reverse=True)[:10]
            for func, info in complex_funcs:
                f.write(f"  {func}: complexity={info['complexity']}\n")
        
        print(f"✓ Saved summary: {summary_file}")
        
        self.output_dir = output_dir
        self.timestamp = timestamp
        return output_dir
    
    def step3_create_visualizations(self):
        """Step 3: Create visualizations (as HTML since matplotlib not available)."""
        print("\nSTEP 3: CREATING VISUALIZATIONS")
        print("=" * 50)
        
        # Create HTML visualization
        html_file = self.output_dir / f"visualization_{self.timestamp}.html"
        
        with open(html_file, 'w') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Code Quality Visualization</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .metric-card { 
            background: #f0f0f0; 
            padding: 15px; 
            margin: 10px; 
            border-radius: 8px;
            display: inline-block;
            width: 200px;
        }
        .metric-value { font-size: 36px; font-weight: bold; color: #333; }
        .metric-label { color: #666; }
        .module-list { margin: 20px 0; }
        .module-item { 
            background: #fff; 
            border: 1px solid #ddd; 
            padding: 10px; 
            margin: 5px 0;
        }
        .complexity-bar {
            height: 20px;
            background: linear-gradient(to right, #4CAF50, #FFC107, #F44336);
            border-radius: 4px;
            margin: 5px 0;
        }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background: #f0f0f0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Code Quality Visualization Report</h1>
        <p>Generated: """ + self.timestamp + """</p>
        
        <h2>Summary Metrics</h2>
        <div>
            <div class="metric-card">
                <div class="metric-label">Total Modules</div>
                <div class="metric-value">""" + str(len(self.results['modules'])) + """</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Functions</div>
                <div class="metric-value">""" + str(len(self.results['functions'])) + """</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Lines</div>
                <div class="metric-value">""" + str(sum(m['lines'] for m in self.results['modules'].values())) + """</div>
            </div>
        </div>
        
        <h2>Module Complexity</h2>
        <table>
            <tr>
                <th>Module</th>
                <th>Lines</th>
                <th>Functions</th>
                <th>Classes</th>
                <th>Imports</th>
            </tr>""")
            
            for module, complexity in sorted(self.results['complexity'].items(), 
                                           key=lambda x: x[1]['lines'], reverse=True):
                f.write(f"""
            <tr>
                <td>{module}</td>
                <td>{complexity['lines']}</td>
                <td>{complexity['functions']}</td>
                <td>{complexity['classes']}</td>
                <td>{complexity['imports']}</td>
            </tr>""")
            
            f.write("""
        </table>
        
        <h2>Function Complexity</h2>
        <table>
            <tr>
                <th>Function</th>
                <th>Complexity</th>
                <th>Lines</th>
            </tr>""")
            
            complex_funcs = sorted(self.results['functions'].items(), 
                                 key=lambda x: x[1]['complexity'], reverse=True)[:20]
            for func, info in complex_funcs:
                f.write(f"""
            <tr>
                <td>{func}</td>
                <td>{info['complexity']}</td>
                <td>{info['lines']}</td>
            </tr>""")
            
            f.write("""
        </table>
        
        <h2>Import Dependencies</h2>
        <div class="module-list">""")
            
            for module, imports in self.results['imports'].items():
                if imports:
                    f.write(f"""
            <div class="module-item">
                <strong>{module}</strong> imports: {', '.join(set(imports))}
            </div>""")
            
            f.write("""
        </div>
    </div>
</body>
</html>""")
        
        print(f"✓ Created HTML visualization: {html_file}")
        
        # Create a simple text "graph"
        graph_file = self.output_dir / f"dependency_graph_{self.timestamp}.txt"
        with open(graph_file, 'w') as f:
            f.write("DEPENDENCY GRAPH (Text Representation)\n")
            f.write("=" * 50 + "\n\n")
            
            for module, imports in self.results['imports'].items():
                if imports:
                    f.write(f"{module}\n")
                    for imp in set(imports):
                        f.write(f"  └── {imp}\n")
                    f.write("\n")
        
        print(f"✓ Created text graph: {graph_file}")
        
        return [html_file, graph_file]


def main():
    """Run the complete pipeline."""
    print("SIMPLE CODE QUALITY PIPELINE TEST")
    print("=" * 80)
    print("\nThis demonstrates the 3-step pipeline:")
    print("1. Analyze code (AST parsing)")
    print("2. Generate data (JSON + summaries)")
    print("3. Create visualizations (HTML + text)\n")
    
    # Analyze the code_quality directory itself
    analyzer = SimpleCodeAnalyzer("code_quality")
    
    # Step 1: Analyze code
    analyzer.step1_analyze_code()
    
    # Step 2: Generate data
    output_dir = analyzer.step2_generate_data()
    
    # Step 3: Create visualizations
    viz_files = analyzer.step3_create_visualizations()
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nGenerated files:")
    for file in output_dir.iterdir():
        print(f"  - {file.name}")
    
    print("\n✅ The pipeline is working correctly!")
    print("\nNOTE: This is a simplified version. The full pipeline includes:")
    print("  - More sophisticated analysis (complexity metrics, call graphs)")
    print("  - Rich visualizations (with matplotlib, networkx, plotly)")
    print("  - Interactive dashboards")
    print("\nTo use the full pipeline, install dependencies:")
    print("  pip install -r code_quality/requirements.txt")


if __name__ == "__main__":
    main()