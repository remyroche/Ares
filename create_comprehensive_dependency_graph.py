#!/usr/bin/env python3
"""
Create a comprehensive dependency graph with real data from the interaction mapping results
"""

import json
import re
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np

def extract_dependency_data():
    """Extract real dependency data from the interaction mapping results"""
    main_file = "/Users/remyroche/Documents/Ares/code_quality/reports/interaction_mapping/interaction_mapping_20250906_012543.json"
    
    try:
        print("Loading main interaction mapping file...")
        with open(main_file, 'r') as f:
            data = json.load(f)
        
        print("Extracting dependency analysis results...")
        
        # Get dependency analysis results
        dependency_results = data.get('dependency_analysis', {}).get('results', {})
        modules = dependency_results.get('modules', {})
        
        # Get call graph results for function-level dependencies
        call_graph_results = data.get('call_graph_analysis', {}).get('results', {})
        functions = call_graph_results.get('functions', {})
        
        # Get enhanced interaction mapping for cross-module interactions
        enhanced_results = data.get('enhanced_interaction_mapping', {}).get('results', {})
        interactions = enhanced_results.get('interactions', [])
        
        print(f"Found {len(modules)} modules, {len(functions)} functions, {len(interactions)} interactions")
        
        return modules, functions, interactions
        
    except Exception as e:
        print(f"Error reading main file: {e}")
        return {}, {}, []

def analyze_dependencies(modules, functions, interactions):
    """Analyze dependencies and create a comprehensive graph structure"""
    
    # Create module dependency graph
    module_deps = defaultdict(set)
    module_info = {}
    
    for module_name, module_data in modules.items():
        file_path = module_data.get('file_path', '')
        dependencies = module_data.get('dependencies', [])
        internal_deps = module_data.get('internal_dependencies', [])
        external_deps = module_data.get('external_dependencies', [])
        
        module_info[module_name] = {
            'file_path': file_path,
            'dependencies': dependencies,
            'internal_dependencies': internal_deps,
            'external_dependencies': external_deps,
            'dependency_count': len(dependencies),
            'internal_count': len(internal_deps),
            'external_count': len(external_deps)
        }
        
        # Add dependencies to graph
        for dep in dependencies:
            module_deps[module_name].add(dep)
    
    # Analyze function-level dependencies
    function_deps = defaultdict(set)
    function_info = {}
    
    for func_name, func_data in functions.items():
        file_path = func_data.get('file_path', '')
        calls = func_data.get('calls', [])
        
        # Extract module from file path
        module_name = Path(file_path).stem if file_path else 'unknown'
        
        function_info[func_name] = {
            'file_path': file_path,
            'module': module_name,
            'calls': calls,
            'call_count': len(calls)
        }
        
        # Add function calls as dependencies
        for call in calls:
            function_deps[func_name].add(call)
    
    # Analyze cross-module interactions
    cross_module_interactions = []
    for interaction in interactions:
        if interaction.get('is_cross_module', False):
            cross_module_interactions.append(interaction)
    
    return module_deps, module_info, function_deps, function_info, cross_module_interactions

def create_comprehensive_dependency_graph(module_deps, module_info, function_deps, function_info, cross_module_interactions):
    """Create a comprehensive interactive dependency graph"""
    
    # Create nodes for the graph
    nodes = []
    links = []
    
    # Add module nodes
    for module_name, info in module_info.items():
        # Determine module type based on file path
        file_path = info['file_path']
        module_type = 'unknown'
        
        if 'src/training' in file_path:
            module_type = 'training'
        elif 'src/analyst' in file_path:
            module_type = 'analyst'
        elif 'src/tactician' in file_path:
            module_type = 'tactician'
        elif 'src/monitoring' in file_path:
            module_type = 'monitoring'
        elif 'src/utils' in file_path:
            module_type = 'utils'
        elif 'src/config' in file_path:
            module_type = 'config'
        elif 'code_quality' in file_path:
            module_type = 'code_quality'
        elif any(ext in file_path for ext in ['.py']):
            module_type = 'core'
        
        # Determine if it's external dependency
        is_external = any(ext in module_name.lower() for ext in ['numpy', 'pandas', 'sklearn', 'torch', 'tensorflow', 'matplotlib', 'seaborn'])
        
        nodes.append({
            'id': module_name,
            'name': module_name,
            'type': 'external' if is_external else module_type,
            'file_path': file_path,
            'dependency_count': info['dependency_count'],
            'internal_count': info['internal_count'],
            'external_count': info['external_count'],
            'size': max(info['dependency_count'] * 2, 10),
            'is_external': is_external
        })
    
    # Add function nodes for high-importance functions
    function_counter = Counter()
    for func_name, info in function_info.items():
        if info['call_count'] > 5:  # Only include functions with many calls
            function_counter[func_name] = info['call_count']
    
    # Add top functions as nodes
    for func_name, call_count in function_counter.most_common(50):  # Top 50 functions
        info = function_info[func_name]
        nodes.append({
            'id': f"func_{func_name}",
            'name': func_name,
            'type': 'function',
            'file_path': info['file_path'],
            'module': info['module'],
            'call_count': call_count,
            'size': max(call_count, 5),
            'is_external': False
        })
    
    # Create links between modules
    for module_name, deps in module_deps.items():
        for dep in deps:
            if dep in module_info:  # Only link to modules we have info about
                links.append({
                    'source': module_name,
                    'target': dep,
                    'type': 'module_dependency',
                    'weight': 1
                })
    
    # Create links between functions and their modules
    for func_name, info in function_info.items():
        if f"func_{func_name}" in [node['id'] for node in nodes]:
            module_name = info['module']
            if module_name in module_info:
                links.append({
                    'source': f"func_{func_name}",
                    'target': module_name,
                    'type': 'function_to_module',
                    'weight': 2
                })
    
    # Add cross-module interaction links
    for interaction in cross_module_interactions[:100]:  # Limit to first 100
        source_file = interaction.get('source_file', '')
        target_file = interaction.get('target_file', '')
        
        if source_file and target_file:
            source_module = Path(source_file).stem
            target_module = Path(target_file).stem
            
            # Check if both modules exist in our nodes
            source_exists = any(node['id'] == source_module for node in nodes)
            target_exists = any(node['id'] == target_module for node in nodes)
            
            if source_exists and target_exists:
                links.append({
                    'source': source_module,
                    'target': target_module,
                    'type': 'cross_module_interaction',
                    'weight': 3
                })
    
    return nodes, links

def create_interactive_dependency_graph(nodes, links):
    """Create the HTML for the interactive dependency graph"""
    
    # Convert to JSON for JavaScript
    nodes_json = json.dumps(nodes)
    links_json = json.dumps(links)
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Comprehensive Dependency Graph - Ares Trading System</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f8f9fa; }}
        .header {{ background-color: #ffffff; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .controls {{ background-color: #ffffff; padding: 15px; border-radius: 8px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .stats {{ background-color: #e8f4f8; padding: 15px; border-radius: 8px; margin: 10px 0; }}
        .node {{ stroke: #fff; stroke-width: 2px; cursor: pointer; }}
        .link {{ stroke: #999; stroke-opacity: .6; }}
        .node:hover {{ stroke: #ff0000; stroke-width: 3px; }}
        .tooltip {{ position: absolute; padding: 12px; background: rgba(0,0,0,0.9); color: white; border-radius: 6px; pointer-events: none; font-size: 12px; max-width: 300px; }}
        .controls button {{ margin: 5px; padding: 10px 15px; background: #007acc; color: white; border: none; border-radius: 4px; cursor: pointer; }}
        .controls button:hover {{ background: #005a9e; }}
        .controls button.active {{ background: #28a745; }}
        .legend {{ background-color: #ffffff; padding: 15px; border-radius: 8px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .legend-item {{ display: inline-block; margin: 5px 15px 5px 0; }}
        .legend-color {{ display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 5px; }}
        #graph-container {{ background-color: #ffffff; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📦 Comprehensive Dependency Graph</h1>
        <p><strong>Ares Trading System</strong> - Complete module and function dependency visualization</p>
        <p>Generated on: 2025-09-06</p>
    </div>
    
    <div class="stats">
        <h3>📊 Graph Statistics</h3>
        <p><strong>Total Nodes:</strong> {len(nodes)} (modules and functions)</p>
        <p><strong>Total Links:</strong> {len(links)} (dependencies and interactions)</p>
        <p><strong>Module Types:</strong> Training, Analyst, Tactician, Monitoring, Utils, Config, Code Quality, External</p>
    </div>
    
    <div class="controls">
        <button onclick="resetZoom()">Reset Zoom</button>
        <button onclick="highlightExternal()">Highlight External Dependencies</button>
        <button onclick="showModulesOnly()">Show Modules Only</button>
        <button onclick="showFunctionsOnly()">Show Functions Only</button>
        <button onclick="showHighDependency()">Show High Dependency</button>
        <button onclick="toggleLabels()">Toggle Labels</button>
    </div>
    
    <div class="legend">
        <h4>Legend</h4>
        <div class="legend-item"><span class="legend-color" style="background-color: #3498db;"></span>Training Modules</div>
        <div class="legend-item"><span class="legend-color" style="background-color: #2ecc71;"></span>Analyst Modules</div>
        <div class="legend-item"><span class="legend-color" style="background-color: #e74c3c;"></span>Tactician Modules</div>
        <div class="legend-item"><span class="legend-color" style="background-color: #9b59b6;"></span>Monitoring Modules</div>
        <div class="legend-item"><span class="legend-color" style="background-color: #1abc9c;"></span>Utils Modules</div>
        <div class="legend-item"><span class="legend-color" style="background-color: #34495e;"></span>Config Modules</div>
        <div class="legend-item"><span class="legend-color" style="background-color: #f39c12;"></span>External Dependencies</div>
        <div class="legend-item"><span class="legend-color" style="background-color: #95a5a6;"></span>Functions</div>
        <div class="legend-item"><span class="legend-color" style="background-color: #4ecdc4;"></span>Code Quality</div>
        <br>
        <div style="margin-top: 10px;">
            <strong>Link Types:</strong>
            <span style="color: #4ecdc4;">—</span> Module Dependencies
            <span style="color: #ff6b6b;">—</span> Cross-Module Interactions
            <span style="color: #f39c12;">—</span> Function to Module
        </div>
    </div>
    
    <div id="graph-container">
        <div id="dependency-graph"></div>
        <div class="tooltip" id="tooltip"></div>
    </div>
    
    <script>
        const width = 1400;
        const height = 900;
        
        // Data
        const nodes = {nodes_json};
        const links = {links_json};
        
        // Create SVG
        const svg = d3.select("#dependency-graph")
            .append("svg")
            .attr("width", width)
            .attr("height", height);
        
        // Create zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on("zoom", (event) => {{
                g.attr("transform", event.transform);
            }});
        
        svg.call(zoom);
        
        const g = svg.append("g");
        
        // Color scale for different types
        const colorScale = d3.scaleOrdinal()
            .domain(['training', 'analyst', 'tactician', 'monitoring', 'utils', 'config', 'external', 'function', 'code_quality', 'unknown'])
            .range(['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#1abc9c', '#34495e', '#f39c12', '#95a5a6', '#4ecdc4', '#bdc3c7']);
        
        // Create force simulation
        const simulation = d3.forceSimulation(nodes)
            .force("link", d3.forceLink(links).id(d => d.id).distance(d => d.type === 'function_to_module' ? 100 : 150))
            .force("charge", d3.forceManyBody().strength(d => d.is_external ? -200 : -300))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(d => d.size + 5));
        
        // Create links
        const link = g.append("g")
            .selectAll("line")
            .data(links)
            .enter().append("line")
            .attr("class", "link")
            .attr("stroke", d => {{
                switch(d.type) {{
                    case 'module_dependency': return '#4ecdc4';
                    case 'cross_module_interaction': return '#ff6b6b';
                    case 'function_to_module': return '#f39c12';
                    default: return '#999';
                }}
            }})
            .attr("stroke-width", d => Math.sqrt(d.weight));
        
        // Create nodes
        const node = g.append("g")
            .selectAll("circle")
            .data(nodes)
            .enter().append("circle")
            .attr("class", "node")
            .attr("r", d => Math.sqrt(d.size) + 3)
            .attr("fill", d => colorScale(d.type))
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended))
            .on("mouseover", showTooltip)
            .on("mouseout", hideTooltip)
            .on("click", selectNode);
        
        // Add labels
        const label = g.append("g")
            .selectAll("text")
            .data(nodes)
            .enter().append("text")
            .text(d => d.name.length > 20 ? d.name.substring(0, 17) + '...' : d.name)
            .attr("font-size", "10px")
            .attr("text-anchor", "middle")
            .attr("dy", "0.35em")
            .style("pointer-events", "none");
        
        // Update positions on simulation tick
        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
            
            node
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);
            
            label
                .attr("x", d => d.x)
                .attr("y", d => d.y);
        }});
        
        // Tooltip functions
        function showTooltip(event, d) {{
            const tooltip = d3.select("#tooltip");
            let content = `<strong>${{d.name}}</strong><br/>`;
            content += `Type: ${{d.type}}<br/>`;
            if (d.dependency_count !== undefined) {{
                content += `Dependencies: ${{d.dependency_count}}<br/>`;
                content += `Internal: ${{d.internal_count}}, External: ${{d.external_count}}<br/>`;
            }}
            if (d.call_count !== undefined) {{
                content += `Function Calls: ${{d.call_count}}<br/>`;
            }}
            if (d.file_path) {{
                content += `File: ${{d.file_path}}<br/>`;
            }}
            
            tooltip
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 10) + "px")
                .style("opacity", 1)
                .html(content);
        }}
        
        function hideTooltip() {{
            d3.select("#tooltip").style("opacity", 0);
        }}
        
        function selectNode(event, d) {{
            // Highlight connected nodes
            const connectedNodes = new Set();
            const connectedLinks = links.filter(link => 
                link.source.id === d.id || link.target.id === d.id
            );
            
            connectedLinks.forEach(link => {{
                connectedNodes.add(link.source.id);
                connectedNodes.add(link.target.id);
            }});
            
            node.style("opacity", node => 
                connectedNodes.has(node.id) ? 1 : 0.3
            );
            
            link.style("opacity", link => 
                connectedNodes.has(link.source.id) && connectedNodes.has(link.target.id) ? 1 : 0.1
            );
        }}
        
        // Drag functions
        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}
        
        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}
        
        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}
        
        // Control functions
        function resetZoom() {{
            svg.transition().duration(750).call(zoom.transform, d3.zoomIdentity);
            node.style("opacity", 1);
            link.style("opacity", 1);
        }}
        
        function highlightExternal() {{
            node.style("opacity", d => d.is_external ? 1 : 0.3);
            link.style("opacity", d => 
                d.source.is_external || d.target.is_external ? 1 : 0.1
            );
        }}
        
        function showModulesOnly() {{
            node.style("opacity", d => d.type !== 'function' ? 1 : 0.1);
            link.style("opacity", d => 
                d.source.type !== 'function' && d.target.type !== 'function' ? 1 : 0.1
            );
        }}
        
        function showFunctionsOnly() {{
            node.style("opacity", d => d.type === 'function' ? 1 : 0.1);
            link.style("opacity", d => 
                d.source.type === 'function' || d.target.type === 'function' ? 1 : 0.1
            );
        }}
        
        function showHighDependency() {{
            node.style("opacity", d => 
                (d.dependency_count && d.dependency_count > 10) || 
                (d.call_count && d.call_count > 20) ? 1 : 0.3
            );
        }}
        
        let labelsVisible = true;
        function toggleLabels() {{
            labelsVisible = !labelsVisible;
            label.style("opacity", labelsVisible ? 1 : 0);
        }}
    </script>
</body>
</html>
"""
    
    output_path = "/Users/remyroche/Documents/Ares/code_quality/reports/interaction_mapping/comprehensive_dependency_graph.html"
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Created comprehensive dependency graph: {output_path}")
    return output_path

def main():
    """Main function to create comprehensive dependency graph"""
    print("📦 Creating comprehensive dependency graph with real data...")
    
    modules, functions, interactions = extract_dependency_data()
    module_deps, module_info, function_deps, function_info, cross_module_interactions = analyze_dependencies(modules, functions, interactions)
    nodes, links = create_comprehensive_dependency_graph(module_deps, module_info, function_deps, function_info, cross_module_interactions)
    
    graph_path = create_interactive_dependency_graph(nodes, links)
    
    # Open the graph
    import subprocess
    subprocess.run(["open", graph_path])
    
    print(f"\n✅ Comprehensive dependency graph created!")
    print(f"📊 Graph contains {len(nodes)} nodes and {len(links)} links")
    print(f"📄 Graph file: {graph_path}")

if __name__ == "__main__":
    main()
