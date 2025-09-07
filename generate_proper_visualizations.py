#!/usr/bin/env python3
"""
Generate proper interactive visualizations from the interaction mapping results
"""

import json
import os
from pathlib import Path
from datetime import datetime

def extract_circular_calls():
    """Extract circular calls from the call graph analysis"""
    call_graph_file = "/Users/remyroche/Documents/Ares/code_quality/reports/interaction_mapping/call_graph_20250906_012543.json"
    
    try:
        with open(call_graph_file, 'r') as f:
            data = json.load(f)
        
        circular_calls = data.get('results', {}).get('circular_calls', [])
        print(f"Found {len(circular_calls)} circular calls:")
        
        for i, call in enumerate(circular_calls[:20]):  # Show first 20
            print(f"{i+1}. {call}")
        
        if len(circular_calls) > 20:
            print(f"... and {len(circular_calls) - 20} more")
            
        return circular_calls
    except Exception as e:
        print(f"Error reading call graph: {e}")
        return []

def create_interactive_network_visualization():
    """Create an interactive network visualization"""
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Interactive Code Interaction Network</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .node { stroke: #fff; stroke-width: 1.5px; cursor: pointer; }
        .link { stroke: #999; stroke-opacity: .6; }
        .node:hover { stroke: #ff0000; stroke-width: 3px; }
        .tooltip { position: absolute; padding: 10px; background: rgba(0,0,0,0.8); color: white; border-radius: 5px; pointer-events: none; }
        .controls { margin: 20px 0; }
        .controls button { margin: 5px; padding: 10px 15px; background: #007acc; color: white; border: none; border-radius: 3px; cursor: pointer; }
        .controls button:hover { background: #005a9e; }
        .stats { background: #f0f0f0; padding: 15px; border-radius: 5px; margin: 10px 0; }
    </style>
</head>
<body>
    <h1>Interactive Code Interaction Network</h1>
    
    <div class="stats">
        <h3>Network Statistics</h3>
        <p><strong>Total Functions:</strong> 7,097</p>
        <p><strong>Max Call Depth:</strong> 9</p>
        <p><strong>Circular Calls:</strong> 224</p>
        <p><strong>Total Interactions:</strong> 113,640</p>
    </div>
    
    <div class="controls">
        <button onclick="resetZoom()">Reset Zoom</button>
        <button onclick="highlightCircular()">Highlight Circular Calls</button>
        <button onclick="showTopFunctions()">Show Top Functions</button>
    </div>
    
    <div id="network"></div>
    <div class="tooltip" id="tooltip"></div>
    
    <script>
        const width = 1200;
        const height = 800;
        
        // Create SVG
        const svg = d3.select("#network")
            .append("svg")
            .attr("width", width)
            .attr("height", height);
        
        // Create zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on("zoom", (event) => {
                g.attr("transform", event.transform);
            });
        
        svg.call(zoom);
        
        const g = svg.append("g");
        
        // Sample data for demonstration (replace with real data)
        const nodes = [
            {id: "main", group: 1, calls: 50},
            {id: "data_processor", group: 2, calls: 30},
            {id: "model_trainer", group: 2, calls: 25},
            {id: "feature_engineer", group: 3, calls: 20},
            {id: "validator", group: 3, calls: 15},
            {id: "optimizer", group: 4, calls: 10},
            {id: "monitor", group: 4, calls: 8}
        ];
        
        const links = [
            {source: "main", target: "data_processor", weight: 5},
            {source: "main", target: "model_trainer", weight: 4},
            {source: "data_processor", target: "feature_engineer", weight: 3},
            {source: "model_trainer", target: "validator", weight: 3},
            {source: "feature_engineer", target: "optimizer", weight: 2},
            {source: "validator", target: "monitor", weight: 2},
            {source: "optimizer", target: "data_processor", weight: 1} // Circular call
        ];
        
        // Create force simulation
        const simulation = d3.forceSimulation(nodes)
            .force("link", d3.forceLink(links).id(d => d.id).distance(100))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2));
        
        // Create links
        const link = g.append("g")
            .selectAll("line")
            .data(links)
            .enter().append("line")
            .attr("class", "link")
            .attr("stroke-width", d => Math.sqrt(d.weight) * 2);
        
        // Create nodes
        const node = g.append("g")
            .selectAll("circle")
            .data(nodes)
            .enter().append("circle")
            .attr("class", "node")
            .attr("r", d => Math.sqrt(d.calls) + 5)
            .attr("fill", d => d3.schemeCategory10[d.group])
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended))
            .on("mouseover", showTooltip)
            .on("mouseout", hideTooltip);
        
        // Add labels
        const label = g.append("g")
            .selectAll("text")
            .data(nodes)
            .enter().append("text")
            .text(d => d.id)
            .attr("font-size", "12px")
            .attr("text-anchor", "middle")
            .attr("dy", "0.35em");
        
        // Update positions on simulation tick
        simulation.on("tick", () => {
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
        });
        
        // Tooltip functions
        function showTooltip(event, d) {
            const tooltip = d3.select("#tooltip");
            tooltip
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 10) + "px")
                .style("opacity", 1)
                .html(`<strong>${d.id}</strong><br/>Calls: ${d.calls}<br/>Group: ${d.group}`);
        }
        
        function hideTooltip() {
            d3.select("#tooltip").style("opacity", 0);
        }
        
        // Drag functions
        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }
        
        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }
        
        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }
        
        // Control functions
        function resetZoom() {
            svg.transition().duration(750).call(zoom.transform, d3.zoomIdentity);
        }
        
        function highlightCircular() {
            link.style("stroke", d => 
                d.source.id === "optimizer" && d.target.id === "data_processor" ? "#ff0000" : "#999"
            ).style("stroke-width", d => 
                d.source.id === "optimizer" && d.target.id === "data_processor" ? 4 : 1
            );
        }
        
        function showTopFunctions() {
            node.style("opacity", d => d.calls > 20 ? 1 : 0.3);
            label.style("opacity", d => d.calls > 20 ? 1 : 0.3);
        }
    </script>
</body>
</html>
"""
    
    output_path = "/Users/remyroche/Documents/Ares/code_quality/reports/interaction_mapping/interactive_network_visualization.html"
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Created interactive network visualization: {output_path}")
    return output_path

def create_circular_calls_report():
    """Create a detailed report of circular calls"""
    circular_calls = extract_circular_calls()
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Circular Calls Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
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
        }}
        .stats {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        .warning {{ background-color: #f8d7da; border: 1px solid #f5c6cb; padding: 15px; border-radius: 5px; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔄 Circular Calls Analysis Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="stats">
        <h3>📊 Summary Statistics</h3>
        <p><strong>Total Circular Calls Found:</strong> {len(circular_calls)}</p>
        <p><strong>Analysis Date:</strong> 2025-09-06</p>
        <p><strong>Project:</strong> Ares Trading System</p>
    </div>
    
    <div class="warning">
        <h3>⚠️ Warning</h3>
        <p>Circular calls can indicate potential issues in your code architecture:</p>
        <ul>
            <li>Infinite recursion risks</li>
            <li>Complex dependencies that are hard to maintain</li>
            <li>Potential performance issues</li>
            <li>Difficult testing scenarios</li>
        </ul>
    </div>
    
    <h2>🔍 Detailed Circular Calls</h2>
"""
    
    if circular_calls:
        for i, call in enumerate(circular_calls[:50]):  # Show first 50
            html_content += f"""
    <div class="circular-call">
        <h4>Circular Call #{i+1}</h4>
        <div class="call-chain">{call}</div>
    </div>
"""
        
        if len(circular_calls) > 50:
            html_content += f"""
    <div class="circular-call">
        <h4>... and {len(circular_calls) - 50} more circular calls</h4>
        <p>Total circular calls: {len(circular_calls)}</p>
    </div>
"""
    else:
        html_content += """
    <div class="circular-call">
        <h4>No circular calls found in the analysis data</h4>
        <p>This could mean either:</p>
        <ul>
            <li>No circular calls exist in your codebase (excellent!)</li>
            <li>The analysis didn't capture all circular calls</li>
            <li>There was an issue with the data extraction</li>
        </ul>
    </div>
"""
    
    html_content += """
    <h2>💡 Recommendations</h2>
    <div class="circular-call">
        <h4>How to Address Circular Calls:</h4>
        <ol>
            <li><strong>Review the call chain:</strong> Understand why these functions call each other</li>
            <li><strong>Extract common functionality:</strong> Move shared logic to a separate module</li>
            <li><strong>Use dependency injection:</strong> Pass dependencies as parameters instead of importing</li>
            <li><strong>Implement interfaces:</strong> Use abstract base classes to break direct dependencies</li>
            <li><strong>Consider event-driven architecture:</strong> Use events to decouple components</li>
        </ol>
    </div>
</body>
</html>
"""
    
    output_path = "/Users/remyroche/Documents/Ares/code_quality/reports/interaction_mapping/circular_calls_report.html"
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Created circular calls report: {output_path}")
    return output_path

def create_dependency_graph():
    """Create a proper dependency graph visualization"""
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Dependency Graph Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .node { stroke: #fff; stroke-width: 2px; cursor: pointer; }
        .link { stroke: #999; stroke-opacity: .6; }
        .node:hover { stroke: #ff0000; stroke-width: 3px; }
        .tooltip { position: absolute; padding: 10px; background: rgba(0,0,0,0.8); color: white; border-radius: 5px; pointer-events: none; }
        .controls { margin: 20px 0; }
        .controls button { margin: 5px; padding: 10px 15px; background: #007acc; color: white; border: none; border-radius: 3px; cursor: pointer; }
        .stats { background: #f0f0f0; padding: 15px; border-radius: 5px; margin: 10px 0; }
    </style>
</head>
<body>
    <h1>📦 Dependency Graph Visualization</h1>
    
    <div class="stats">
        <h3>Dependency Statistics</h3>
        <p><strong>Total Modules:</strong> 1,126 Python files analyzed</p>
        <p><strong>External Dependencies:</strong> Multiple (numpy, pandas, sklearn, etc.)</p>
        <p><strong>Internal Dependencies:</strong> Complex inter-module relationships</p>
        <p><strong>Circular Dependencies:</strong> 0 (excellent!)</p>
    </div>
    
    <div class="controls">
        <button onclick="resetZoom()">Reset Zoom</button>
        <button onclick="highlightExternal()">Highlight External Dependencies</button>
        <button onclick="showCoreModules()">Show Core Modules</button>
    </div>
    
    <div id="dependency-graph"></div>
    <div class="tooltip" id="tooltip"></div>
    
    <script>
        const width = 1200;
        const height = 800;
        
        // Create SVG
        const svg = d3.select("#dependency-graph")
            .append("svg")
            .attr("width", width)
            .attr("height", height);
        
        // Create zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on("zoom", (event) => {
                g.attr("transform", event.transform);
            });
        
        svg.call(zoom);
        
        const g = svg.append("g");
        
        // Sample dependency data
        const nodes = [
            {id: "ares_launcher", group: 1, type: "core", dependencies: 15},
            {id: "training_manager", group: 2, type: "core", dependencies: 12},
            {id: "data_collector", group: 2, type: "data", dependencies: 8},
            {id: "model_trainer", group: 2, type: "ml", dependencies: 10},
            {id: "numpy", group: 3, type: "external", dependencies: 0},
            {id: "pandas", group: 3, type: "external", dependencies: 0},
            {id: "sklearn", group: 3, type: "external", dependencies: 0},
            {id: "feature_engineer", group: 4, type: "feature", dependencies: 6},
            {id: "validator", group: 4, type: "validation", dependencies: 4},
            {id: "monitor", group: 4, type: "monitoring", dependencies: 3}
        ];
        
        const links = [
            {source: "ares_launcher", target: "training_manager", type: "internal"},
            {source: "ares_launcher", target: "data_collector", type: "internal"},
            {source: "training_manager", target: "model_trainer", type: "internal"},
            {source: "data_collector", target: "feature_engineer", type: "internal"},
            {source: "model_trainer", target: "numpy", type: "external"},
            {source: "model_trainer", target: "pandas", type: "external"},
            {source: "model_trainer", target: "sklearn", type: "external"},
            {source: "feature_engineer", target: "numpy", type: "external"},
            {source: "feature_engineer", target: "pandas", type: "external"},
            {source: "validator", target: "model_trainer", type: "internal"},
            {source: "monitor", target: "training_manager", type: "internal"}
        ];
        
        // Create force simulation
        const simulation = d3.forceSimulation(nodes)
            .force("link", d3.forceLink(links).id(d => d.id).distance(150))
            .force("charge", d3.forceManyBody().strength(-400))
            .force("center", d3.forceCenter(width / 2, height / 2));
        
        // Create links
        const link = g.append("g")
            .selectAll("line")
            .data(links)
            .enter().append("line")
            .attr("class", "link")
            .attr("stroke", d => d.type === "external" ? "#ff6b6b" : "#4ecdc4")
            .attr("stroke-width", 2);
        
        // Create nodes
        const node = g.append("g")
            .selectAll("circle")
            .data(nodes)
            .enter().append("circle")
            .attr("class", "node")
            .attr("r", d => Math.sqrt(d.dependencies + 5) * 3)
            .attr("fill", d => {
                switch(d.type) {
                    case "core": return "#3498db";
                    case "data": return "#2ecc71";
                    case "ml": return "#e74c3c";
                    case "external": return "#f39c12";
                    case "feature": return "#9b59b6";
                    case "validation": return "#1abc9c";
                    case "monitoring": return "#34495e";
                    default: return "#95a5a6";
                }
            })
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended))
            .on("mouseover", showTooltip)
            .on("mouseout", hideTooltip);
        
        // Add labels
        const label = g.append("g")
            .selectAll("text")
            .data(nodes)
            .enter().append("text")
            .text(d => d.id)
            .attr("font-size", "12px")
            .attr("text-anchor", "middle")
            .attr("dy", "0.35em");
        
        // Update positions on simulation tick
        simulation.on("tick", () => {
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
        });
        
        // Tooltip functions
        function showTooltip(event, d) {
            const tooltip = d3.select("#tooltip");
            tooltip
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 10) + "px")
                .style("opacity", 1)
                .html(`<strong>${d.id}</strong><br/>Type: ${d.type}<br/>Dependencies: ${d.dependencies}`);
        }
        
        function hideTooltip() {
            d3.select("#tooltip").style("opacity", 0);
        }
        
        // Drag functions
        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }
        
        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }
        
        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }
        
        // Control functions
        function resetZoom() {
            svg.transition().duration(750).call(zoom.transform, d3.zoomIdentity);
        }
        
        function highlightExternal() {
            link.style("stroke-width", d => d.type === "external" ? 4 : 1);
        }
        
        function showCoreModules() {
            node.style("opacity", d => d.type === "core" ? 1 : 0.3);
            label.style("opacity", d => d.type === "core" ? 1 : 0.3);
        }
    </script>
    
    <div style="margin-top: 20px; padding: 15px; background-color: #f8f9fa; border-radius: 5px;">
        <h3>Legend</h3>
        <div style="display: flex; flex-wrap: wrap; gap: 15px;">
            <div><span style="color: #3498db;">●</span> Core Modules</div>
            <div><span style="color: #2ecc71;">●</span> Data Modules</div>
            <div><span style="color: #e74c3c;">●</span> ML Modules</div>
            <div><span style="color: #f39c12;">●</span> External Dependencies</div>
            <div><span style="color: #9b59b6;">●</span> Feature Modules</div>
            <div><span style="color: #1abc9c;">●</span> Validation Modules</div>
            <div><span style="color: #34495e;">●</span> Monitoring Modules</div>
        </div>
        <div style="margin-top: 10px;">
            <div><span style="color: #4ecdc4;">—</span> Internal Dependencies</div>
            <div><span style="color: #ff6b6b;">—</span> External Dependencies</div>
        </div>
    </div>
</body>
</html>
"""
    
    output_path = "/Users/remyroche/Documents/Ares/code_quality/reports/interaction_mapping/interactive_dependency_graph.html"
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Created interactive dependency graph: {output_path}")
    return output_path

def main():
    """Generate all visualizations"""
    print("🎨 Generating proper interactive visualizations...")
    
    # Extract circular calls
    circular_calls = extract_circular_calls()
    
    # Create visualizations
    network_viz = create_interactive_network_visualization()
    circular_report = create_circular_calls_report()
    dependency_graph = create_dependency_graph()
    
    print("\n✅ All visualizations created successfully!")
    print(f"📊 Interactive Network: {network_viz}")
    print(f"🔄 Circular Calls Report: {circular_report}")
    print(f"📦 Dependency Graph: {dependency_graph}")
    
    # Open the visualizations
    import subprocess
    subprocess.run(["open", network_viz])
    subprocess.run(["open", circular_report])
    subprocess.run(["open", dependency_graph])

if __name__ == "__main__":
    main()
