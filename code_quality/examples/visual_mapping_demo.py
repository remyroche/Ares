#!/usr/bin/env python3
"""
Visual Mapping Demo

This script demonstrates the visual mapping capabilities of the code quality tools.
It shows how to generate various types of visualizations for code analysis.
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from visualizers import (
    DependencyGraphVisualizer,
    ComplexityHeatmapVisualizer,
    InteractionNetworkVisualizer,
    DashboardGenerator
)


def demonstrate_visual_mapping():
    """Demonstrate all visual mapping capabilities."""
    print("CODE QUALITY VISUAL MAPPING DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Create output directory
    output_dir = Path("code_quality/demo_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample data representing a typical Python project
    sample_data = {
        "project_name": "Sample Python Project",
        "dependencies": {
            "modules": {
                # Core modules
                "app.main": ["app.config", "app.database", "app.api", "app.utils"],
                "app.config": ["app.utils.validators", "app.utils.loaders"],
                "app.database": ["app.config", "app.models", "app.utils.db"],
                
                # API layer
                "app.api": ["app.database", "app.auth", "app.serializers"],
                "app.api.endpoints": ["app.api", "app.services"],
                "app.auth": ["app.database", "app.utils.crypto"],
                
                # Service layer
                "app.services": ["app.database", "app.models", "app.utils"],
                "app.services.user": ["app.services", "app.models.user"],
                "app.services.product": ["app.services", "app.models.product"],
                
                # Models
                "app.models": ["app.database"],
                "app.models.user": ["app.models"],
                "app.models.product": ["app.models"],
                
                # Utilities
                "app.utils": [],
                "app.utils.validators": ["app.utils"],
                "app.utils.loaders": ["app.utils"],
                "app.utils.db": ["app.utils"],
                "app.utils.crypto": ["app.utils"],
                
                # Circular dependency example
                "app.circular_a": ["app.circular_b"],
                "app.circular_b": ["app.circular_c"],
                "app.circular_c": ["app.circular_a"]
            },
            "circular_imports": [
                ["app.circular_a", "app.circular_b", "app.circular_c"]
            ]
        },
        "complexity": {
            "files": {
                "app/main.py": {
                    "cyclomatic_complexity": 8,
                    "lines_of_code": 150,
                    "maintainability_index": 75,
                    "complexity": 8,
                    "lines": 150,
                    "functions": 5
                },
                "app/database.py": {
                    "cyclomatic_complexity": 25,
                    "lines_of_code": 450,
                    "maintainability_index": 45,
                    "complexity": 25,
                    "lines": 450,
                    "functions": 12
                },
                "app/api/endpoints.py": {
                    "cyclomatic_complexity": 15,
                    "lines_of_code": 300,
                    "maintainability_index": 60,
                    "complexity": 15,
                    "lines": 300,
                    "functions": 10
                },
                "app/services/user.py": {
                    "cyclomatic_complexity": 20,
                    "lines_of_code": 350,
                    "maintainability_index": 55,
                    "complexity": 20,
                    "lines": 350,
                    "functions": 8
                },
                "app/models/user.py": {
                    "cyclomatic_complexity": 5,
                    "lines_of_code": 100,
                    "maintainability_index": 85,
                    "complexity": 5,
                    "lines": 100,
                    "functions": 3
                },
                "app/utils/validators.py": {
                    "cyclomatic_complexity": 12,
                    "lines_of_code": 200,
                    "maintainability_index": 70,
                    "complexity": 12,
                    "lines": 200,
                    "functions": 15
                },
                "app/auth.py": {
                    "cyclomatic_complexity": 18,
                    "lines_of_code": 250,
                    "maintainability_index": 58,
                    "complexity": 18,
                    "lines": 250,
                    "functions": 7
                }
            },
            "average_complexity": 14.6
        },
        "call_graph": {
            "functions": {
                # Main application flow
                "main": ["init_app", "run_server"],
                "init_app": ["load_config", "setup_database", "setup_api", "setup_logging"],
                "run_server": ["create_app", "start_server"],
                
                # Configuration
                "load_config": ["validate_config", "load_env_vars"],
                "validate_config": ["check_required_fields", "validate_types"],
                
                # Database operations
                "setup_database": ["connect_db", "run_migrations", "seed_data"],
                "connect_db": ["create_connection_pool", "test_connection"],
                "run_migrations": ["load_migration_files", "execute_migrations"],
                
                # API setup
                "setup_api": ["register_routes", "setup_middleware", "setup_error_handlers"],
                "register_routes": ["load_endpoints", "validate_routes"],
                
                # Authentication
                "authenticate_user": ["validate_token", "get_user_from_token", "check_permissions"],
                "validate_token": ["decode_jwt", "verify_signature"],
                
                # Business logic
                "create_user": ["validate_user_data", "hash_password", "save_to_db", "send_welcome_email"],
                "update_product": ["validate_product_data", "check_inventory", "update_db", "update_cache"],
                
                # Utilities
                "hash_password": ["generate_salt", "apply_hash"],
                "send_email": ["render_template", "connect_smtp", "send_message"]
            }
        },
        "issues": [
            {
                "file": "app/database.py",
                "message": "Cyclomatic complexity is 25 (threshold is 10)",
                "severity": "High",
                "line": 145,
                "type": "complexity"
            },
            {
                "file": "app/services/user.py",
                "message": "Function 'process_user_data' has 15 parameters",
                "severity": "High",
                "line": 87,
                "type": "complexity"
            },
            {
                "file": "app/api/endpoints.py",
                "message": "Missing error handling in API endpoint",
                "severity": "Medium",
                "line": 234,
                "type": "error_handling"
            },
            {
                "file": "app/auth.py",
                "message": "Potential SQL injection vulnerability",
                "severity": "High",
                "line": 156,
                "type": "security"
            },
            {
                "file": "app/utils/validators.py",
                "message": "Unused import 'datetime'",
                "severity": "Low",
                "line": 5,
                "type": "unused"
            },
            {
                "file": "app/main.py",
                "message": "Global variable usage detected",
                "severity": "Medium",
                "line": 23,
                "type": "code_smell"
            }
        ],
        "architecture": {
            "layers": [
                "Presentation Layer (API)",
                "Business Logic Layer (Services)",
                "Data Access Layer (Models/Database)",
                "Utility Layer"
            ],
            "components": {
                "API Gateway": {
                    "type": "entry_point",
                    "dependencies": ["Auth Service", "Business Services"]
                },
                "Auth Service": {
                    "type": "service",
                    "dependencies": ["User Database", "Token Manager"]
                },
                "Business Services": {
                    "type": "service",
                    "dependencies": ["Data Models", "External APIs"]
                },
                "Data Models": {
                    "type": "data",
                    "dependencies": ["Database Connection"]
                }
            }
        }
    }
    
    # Initialize visualizers
    print("Initializing visualizers...")
    dep_viz = DependencyGraphVisualizer(str(output_dir))
    complexity_viz = ComplexityHeatmapVisualizer(str(output_dir))
    network_viz = InteractionNetworkVisualizer(str(output_dir))
    dashboard_gen = DashboardGenerator(str(output_dir))
    
    generated_files = []
    
    # 1. Dependency Visualizations
    print("\n[1/6] Creating dependency visualizations...")
    
    deps = sample_data["dependencies"]["modules"]
    fig, metadata = dep_viz.create_dependency_graph(deps, "Module Dependencies")
    files = dep_viz.save_figure(fig, "dependency_graph")
    generated_files.extend(files)
    print(f"  ✓ Generated dependency graph: {files[0]}")
    
    # Circular dependencies
    circular = sample_data["dependencies"]["circular_imports"]
    fig = dep_viz.create_circular_dependency_visualization(circular, "Circular Dependencies")
    files = dep_viz.save_figure(fig, "circular_dependencies")
    generated_files.extend(files)
    print(f"  ✓ Generated circular dependency visualization: {files[0]}")
    
    # Module hierarchy
    fig = dep_viz.create_module_hierarchy(deps, "Module Hierarchy")
    files = dep_viz.save_figure(fig, "module_hierarchy")
    generated_files.extend(files)
    print(f"  ✓ Generated module hierarchy: {files[0]}")
    
    # 2. Complexity Visualizations
    print("\n[2/6] Creating complexity visualizations...")
    
    complexity_data = sample_data["complexity"]["files"]
    fig, metadata = complexity_viz.create_complexity_heatmap(complexity_data, "Code Complexity Heatmap")
    files = complexity_viz.save_figure(fig, "complexity_heatmap")
    generated_files.extend(files)
    print(f"  ✓ Generated complexity heatmap: {files[0]}")
    
    # Treemap
    fig = complexity_viz.create_treemap_visualization(
        complexity_data, 
        'cyclomatic_complexity',
        "Complexity Treemap"
    )
    files = complexity_viz.save_figure(fig, "complexity_treemap")
    generated_files.extend(files)
    print(f"  ✓ Generated complexity treemap: {files[0]}")
    
    # Bubble chart
    fig = complexity_viz.create_module_complexity_bubble_chart(
        complexity_data,
        "Module Complexity Overview"
    )
    files = complexity_viz.save_figure(fig, "complexity_bubble")
    generated_files.extend(files)
    print(f"  ✓ Generated complexity bubble chart: {files[0]}")
    
    # 3. Function Call Network
    print("\n[3/6] Creating function call network...")
    
    call_graph = sample_data["call_graph"]["functions"]
    fig, metadata = network_viz.create_function_call_network(call_graph, "Function Call Network")
    files = network_viz.save_figure(fig, "function_network")
    generated_files.extend(files)
    print(f"  ✓ Generated function call network: {files[0]}")
    
    # 4. Interactive Network
    print("\n[4/6] Creating interactive network...")
    
    html_file = network_viz.create_interactive_network(
        call_graph,
        title="Interactive Function Network"
    )
    generated_files.append(html_file)
    print(f"  ✓ Generated interactive network: {html_file}")
    
    # 5. Module Interaction Matrix
    print("\n[5/6] Creating module interaction matrix...")
    
    # Convert dependencies to interactions format
    interactions = {}
    for module, deps in sample_data["dependencies"]["modules"].items():
        interactions[module] = deps
    
    fig = network_viz.create_module_interaction_matrix(interactions, "Module Interaction Matrix")
    files = network_viz.save_figure(fig, "interaction_matrix")
    generated_files.extend(files)
    print(f"  ✓ Generated interaction matrix: {files[0]}")
    
    # 6. Interactive Dashboard
    print("\n[6/6] Creating interactive dashboard...")
    
    dashboard_file = dashboard_gen.generate_quality_dashboard(
        sample_data,
        "Code Quality Dashboard - Demo Project"
    )
    generated_files.append(dashboard_file)
    print(f"  ✓ Generated interactive dashboard: {dashboard_file}")
    
    # Summary
    print("\n" + "=" * 80)
    print("VISUAL MAPPING DEMONSTRATION COMPLETE!")
    print("=" * 80)
    print()
    print(f"Generated {len(generated_files)} visualization files in {output_dir}/")
    print()
    print("The visualizations include:")
    print("  • Dependency graphs showing module relationships")
    print("  • Circular dependency detection and visualization")
    print("  • Module hierarchy views")
    print("  • Code complexity heatmaps")
    print("  • Complexity treemaps for easy identification of complex files")
    print("  • Bubble charts showing multiple complexity dimensions")
    print("  • Function call networks")
    print("  • Interactive network visualizations")
    print("  • Module interaction matrices")
    print("  • Comprehensive HTML dashboards with all metrics")
    print()
    print("Each visualization provides different insights:")
    print("  - Dependency graphs: Understand module coupling")
    print("  - Complexity heatmaps: Identify maintenance hotspots")
    print("  - Function networks: Trace execution flow")
    print("  - Interactive dashboards: Explore metrics dynamically")
    print()
    print("Open the HTML files in a web browser for interactive exploration!")
    
    # Save the sample data for reference
    sample_data_file = output_dir / "sample_data.json"
    with open(sample_data_file, 'w') as f:
        json.dump(sample_data, f, indent=2)
    print(f"\nSample data saved to: {sample_data_file}")
    
    return generated_files


if __name__ == "__main__":
    demonstrate_visual_mapping()