#!/usr/bin/env python3
"""
Analyze the interaction mapping pipeline results to identify key issues
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

def analyze_architecture_issues():
    """Analyze architecture issues from the architecture report"""
    print("🏗️  Analyzing Architecture Issues...")
    print("=" * 50)
    
    arch_file = Path("/Users/remyroche/Documents/Ares/code_quality/reports/interaction_mapping/architecture_20250906_012543.json")
    
    if not arch_file.exists():
        print("❌ Architecture report not found")
        return
    
    with open(arch_file, 'r') as f:
        arch_data = json.load(f)
    
    results = arch_data.get("results", {})
    components = results.get("components", {})
    
    # Analyze architecture scores
    scores = []
    low_scores = []
    high_coupling = []
    low_cohesion = []
    
    for file_path, component in components.items():
        score = component.get("architecture_score", 0)
        coupling = component.get("coupling_score", 0)
        cohesion = component.get("cohesion_score", 0)
        violations = component.get("violations", [])
        
        scores.append({
            "file": Path(file_path).name,
            "score": score,
            "coupling": coupling,
            "cohesion": cohesion,
            "violations": len(violations)
        })
        
        if score < 70:
            low_scores.append((Path(file_path).name, score))
        
        if coupling > 0.7:
            high_coupling.append((Path(file_path).name, coupling))
            
        if cohesion < 0.3:
            low_cohesion.append((Path(file_path).name, cohesion))
    
    # Sort by score
    scores.sort(key=lambda x: x["score"])
    
    print(f"📊 Total Components Analyzed: {len(components)}")
    print(f"📊 Average Architecture Score: {sum(s['score'] for s in scores) / len(scores):.1f}")
    print()
    
    if low_scores:
        print("⚠️  Components with Low Architecture Scores (< 70):")
        for file, score in sorted(low_scores, key=lambda x: x[1])[:10]:
            print(f"   • {file}: {score}")
        print()
    
    if high_coupling:
        print("🔗 Components with High Coupling (> 0.7):")
        for file, coupling in sorted(high_coupling, key=lambda x: x[1], reverse=True)[:10]:
            print(f"   • {file}: {coupling:.2f}")
        print()
    
    if low_cohesion:
        print("🧩 Components with Low Cohesion (< 0.3):")
        for file, cohesion in sorted(low_cohesion, key=lambda x: x[1])[:10]:
            print(f"   • {file}: {cohesion:.2f}")
        print()
    
    # Show top 10 worst components
    print("🔴 Top 10 Components Needing Attention:")
    for i, comp in enumerate(scores[:10], 1):
        print(f"   {i:2d}. {comp['file']:<40} Score: {comp['score']:5.1f} Coupling: {comp['coupling']:.2f} Cohesion: {comp['cohesion']:.2f}")
    
    return {
        "total_components": len(components),
        "low_scores": low_scores,
        "high_coupling": high_coupling,
        "low_cohesion": low_cohesion,
        "worst_components": scores[:10]
    }

def analyze_dependency_issues():
    """Analyze dependency issues from the comprehensive dependency graph"""
    print("\n🔗 Analyzing Dependency Issues...")
    print("=" * 50)
    
    # Read the comprehensive dependency graph HTML to extract insights
    dep_file = Path("/Users/remyroche/Documents/Ares/code_quality/reports/interaction_mapping/comprehensive_dependency_graph.html")
    
    if not dep_file.exists():
        print("❌ Comprehensive dependency graph not found")
        return
    
    # Extract key statistics from the HTML
    with open(dep_file, 'r') as f:
        content = f.read()
    
    # Look for statistics in the HTML
    if "Graph contains" in content:
        import re
        stats_match = re.search(r'Graph contains (\d+) nodes and (\d+) links', content)
        if stats_match:
            nodes = int(stats_match.group(1))
            links = int(stats_match.group(2))
            print(f"📊 Dependency Graph Statistics:")
            print(f"   • Total Nodes: {nodes}")
            print(f"   • Total Links: {links}")
            print(f"   • Average Connections per Node: {links/nodes:.1f}")
            
            if links/nodes > 5:
                print("⚠️  High connectivity detected - potential tight coupling")
            elif links/nodes < 1:
                print("ℹ️  Low connectivity - good separation of concerns")
    
    return {"nodes": nodes if 'nodes' in locals() else 0, "links": links if 'links' in locals() else 0}

def analyze_circular_calls_summary():
    """Analyze circular calls summary"""
    print("\n🔄 Analyzing Circular Calls Summary...")
    print("=" * 50)
    
    # Read the enhanced circular calls report
    circ_file = Path("/Users/remyroche/Documents/Ares/code_quality/reports/interaction_mapping/enhanced_circular_calls_report.html")
    
    if not circ_file.exists():
        print("❌ Enhanced circular calls report not found")
        return
    
    with open(circ_file, 'r') as f:
        content = f.read()
    
    # Extract statistics
    import re
    
    # Look for total functions and circular calls
    func_match = re.search(r'Total Functions: (\d+)', content)
    circ_match = re.search(r'Circular Calls: (\d+)', content)
    
    if func_match and circ_match:
        total_funcs = int(func_match.group(1))
        total_circ = int(circ_match.group(1))
        
        print(f"📊 Circular Calls Analysis:")
        print(f"   • Total Functions: {total_funcs}")
        print(f"   • Circular Calls: {total_circ}")
        print(f"   • Circular Call Rate: {total_circ/total_funcs*100:.2f}%")
        
        if total_circ/total_funcs > 0.05:  # More than 5%
            print("⚠️  High circular call rate detected")
        elif total_circ/total_funcs < 0.01:  # Less than 1%
            print("✅ Low circular call rate - good design")
    
    return {"total_functions": total_funcs if 'total_funcs' in locals() else 0, 
            "circular_calls": total_circ if 'total_circ' in locals() else 0}

def generate_issue_summary():
    """Generate a comprehensive issue summary"""
    print("\n📋 COMPREHENSIVE ISSUE SUMMARY")
    print("=" * 60)
    
    arch_issues = analyze_architecture_issues()
    dep_issues = analyze_dependency_issues()
    circ_issues = analyze_circular_calls_summary()
    
    print("\n🎯 KEY ISSUES IDENTIFIED:")
    print("-" * 30)
    
    issues = []
    
    if arch_issues and arch_issues.get("low_scores"):
        issues.append(f"• {len(arch_issues['low_scores'])} components with low architecture scores")
    
    if arch_issues and arch_issues.get("high_coupling"):
        issues.append(f"• {len(arch_issues['high_coupling'])} components with high coupling")
    
    if arch_issues and arch_issues.get("low_cohesion"):
        issues.append(f"• {len(arch_issues['low_cohesion'])} components with low cohesion")
    
    if dep_issues and dep_issues.get("links", 0) > 0:
        avg_conn = dep_issues["links"] / dep_issues["nodes"] if dep_issues["nodes"] > 0 else 0
        if avg_conn > 5:
            issues.append(f"• High dependency connectivity ({avg_conn:.1f} avg connections)")
    
    if circ_issues and circ_issues.get("circular_calls", 0) > 0:
        circ_rate = circ_issues["circular_calls"] / circ_issues["total_functions"] if circ_issues["total_functions"] > 0 else 0
        if circ_rate > 0.05:
            issues.append(f"• High circular call rate ({circ_rate*100:.1f}%)")
    
    if not issues:
        print("✅ No major architectural issues detected!")
    else:
        for issue in issues:
            print(issue)
    
    print("\n💡 RECOMMENDATIONS:")
    print("-" * 20)
    print("1. Focus on components with low architecture scores")
    print("2. Reduce coupling in highly connected components")
    print("3. Improve cohesion in loosely coupled components")
    print("4. Review and refactor circular dependencies")
    print("5. Consider breaking down large, complex modules")

if __name__ == "__main__":
    generate_issue_summary()
