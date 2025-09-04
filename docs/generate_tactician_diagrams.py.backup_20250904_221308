#!/usr/bin/env python3
"""Generate visual diagrams for Tactician architecture."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np


def create_architecture_diagram():
    """Create the component architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define component positions and sizes
    components = {
        # Main components
        'Tactician': {'pos': (5, 9), 'size': (2, 0.8), 'color': '#FFB6C1'},
        'TacticsOrchestrator': {'pos': (5, 7.5), 'size': (2.5, 0.8), 'color': '#87CEEB'},
        'DecisionPolicy': {'pos': (5, 6), 'size': (2, 0.8), 'color': '#98FB98'},
        
        # Sizing components
        'PositionSizer': {'pos': (1.5, 4.5), 'size': (1.8, 0.7), 'color': '#DDA0DD'},
        'LeverageSizer': {'pos': (3.5, 4.5), 'size': (1.8, 0.7), 'color': '#DDA0DD'},
        'PositionDivision': {'pos': (5.5, 4.5), 'size': (1.8, 0.7), 'color': '#DDA0DD'},
        
        # Analysis components
        'SRBreakout': {'pos': (1, 3), 'size': (1.8, 0.7), 'color': '#F0E68C'},
        'ScenarioPredictor': {'pos': (3, 3), 'size': (1.8, 0.7), 'color': '#F0E68C'},
        'MLTactics': {'pos': (5, 3), 'size': (1.8, 0.7), 'color': '#F0E68C'},
        
        # Execution components
        'PositionMonitor': {'pos': (7.5, 4.5), 'size': (1.8, 0.7), 'color': '#FFE4B5'},
        'OrderManager': {'pos': (7.5, 3), 'size': (1.8, 0.7), 'color': '#FFE4B5'},
        
        # SR Modules
        'SRLevelDetector': {'pos': (0.5, 1.5), 'size': (1.5, 0.6), 'color': '#E0E0E0'},
        'SRMetrics': {'pos': (2.2, 1.5), 'size': (1.5, 0.6), 'color': '#E0E0E0'},
        'SRFeatures': {'pos': (3.9, 1.5), 'size': (1.5, 0.6), 'color': '#E0E0E0'},
    }
    
    # Draw components
    for name, props in components.items():
        box = FancyBboxPatch(
            (props['pos'][0] - props['size'][0]/2, props['pos'][1] - props['size'][1]/2),
            props['size'][0], props['size'][1],
            boxstyle="round,pad=0.1",
            facecolor=props['color'],
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(box)
        
        # Add text
        ax.text(props['pos'][0], props['pos'][1], name,
                ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Define connections
    connections = [
        ('Tactician', 'TacticsOrchestrator'),
        ('TacticsOrchestrator', 'DecisionPolicy'),
        ('DecisionPolicy', 'PositionSizer'),
        ('DecisionPolicy', 'LeverageSizer'),
        ('DecisionPolicy', 'SRBreakout'),
        ('DecisionPolicy', 'MLTactics'),
        ('TacticsOrchestrator', 'PositionMonitor'),
        ('TacticsOrchestrator', 'OrderManager'),
        ('TacticsOrchestrator', 'PositionDivision'),
        ('Tactician', 'ScenarioPredictor'),
        ('SRBreakout', 'SRLevelDetector'),
        ('SRBreakout', 'SRMetrics'),
        ('SRBreakout', 'SRFeatures'),
    ]
    
    # Draw connections
    for start, end in connections:
        start_pos = components[start]['pos']
        end_pos = components[end]['pos']
        
        arrow = ConnectionPatch(
            start_pos, end_pos, "data", "data",
            arrowstyle="->", shrinkA=30, shrinkB=30,
            mutation_scale=20, fc="black", linewidth=1.5
        )
        ax.add_patch(arrow)
    
    # Add title
    ax.text(5, 9.8, 'Tactician Component Architecture', 
            ha='center', fontsize=16, fontweight='bold')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='#FFB6C1', label='Main Components'),
        mpatches.Patch(color='#87CEEB', label='Orchestration'),
        mpatches.Patch(color='#DDA0DD', label='Sizing Components'),
        mpatches.Patch(color='#F0E68C', label='Analysis Components'),
        mpatches.Patch(color='#FFE4B5', label='Execution Components'),
        mpatches.Patch(color='#E0E0E0', label='SR Modules'),
    ]
    ax.legend(handles=legend_elements, loc='lower center', ncol=3, frameon=False)
    
    plt.tight_layout()
    plt.savefig('docs/tactician_architecture.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_data_flow_diagram():
    """Create the data flow diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define stages
    stages = [
        {'name': 'Input\nValidation', 'pos': (2, 8), 'color': '#FFE4E1'},
        {'name': 'Component\nInitialization', 'pos': (2, 6.5), 'color': '#F0F8FF'},
        {'name': 'Market Data\nAnalysis', 'pos': (2, 5), 'color': '#F0FFF0'},
        {'name': 'S/R Level\nDetection', 'pos': (5, 5), 'color': '#F0FFF0'},
        {'name': 'ML Feature\nExtraction', 'pos': (8, 5), 'color': '#F0FFF0'},
        {'name': 'Decision\nAggregation', 'pos': (5, 3.5), 'color': '#FFF8DC'},
        {'name': 'Position\nSizing', 'pos': (3, 2), 'color': '#E6E6FA'},
        {'name': 'Risk\nManagement', 'pos': (5, 2), 'color': '#E6E6FA'},
        {'name': 'Order\nExecution', 'pos': (7, 2), 'color': '#E6E6FA'},
        {'name': 'Performance\nTracking', 'pos': (5, 0.5), 'color': '#FFDEAD'},
    ]
    
    # Draw stages
    for stage in stages:
        circle = plt.Circle(stage['pos'], 0.8, facecolor=stage['color'], 
                          edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(stage['pos'][0], stage['pos'][1], stage['name'],
                ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Define flow
    flows = [
        (0, 1),  # Input -> Initialization
        (1, 2),  # Initialization -> Market Analysis
        (1, 3),  # Initialization -> S/R Detection
        (1, 4),  # Initialization -> ML Features
        (2, 5),  # Market Analysis -> Decision
        (3, 5),  # S/R Detection -> Decision
        (4, 5),  # ML Features -> Decision
        (5, 6),  # Decision -> Position Sizing
        (5, 7),  # Decision -> Risk Management
        (5, 8),  # Decision -> Order Execution
        (6, 9),  # Position Sizing -> Performance
        (7, 9),  # Risk Management -> Performance
        (8, 9),  # Order Execution -> Performance
    ]
    
    # Draw flows
    for start_idx, end_idx in flows:
        start = stages[start_idx]['pos']
        end = stages[end_idx]['pos']
        
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))
    
    # Add title
    ax.text(5, 9.5, 'Tactician Data Flow', 
            ha='center', fontsize=16, fontweight='bold')
    
    # Add annotations
    ax.text(0.5, 8, 'User Input', ha='center', fontsize=10, style='italic')
    ax.text(5, -0.5, 'Results & Metrics', ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig('docs/tactician_data_flow.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_sequence_diagram():
    """Create a simplified sequence diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define participants
    participants = [
        'User', 'Tactician', 'Orchestrator', 'DecisionPolicy', 
        'SRPredictor', 'OrderManager'
    ]
    
    # Draw participant boxes
    x_positions = np.linspace(1, 9, len(participants))
    for i, (participant, x) in enumerate(zip(participants, x_positions)):
        box = FancyBboxPatch(
            (x - 0.6, 8.5), 1.2, 0.8,
            boxstyle="round,pad=0.1",
            facecolor='lightblue',
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(box)
        ax.text(x, 8.9, participant, ha='center', va='center', 
                fontsize=10, fontweight='bold')
        
        # Draw lifeline
        ax.plot([x, x], [8.5, 0.5], 'k--', alpha=0.5)
    
    # Define interactions
    interactions = [
        (0, 1, 7.5, 'execute_tactics()'),
        (1, 2, 7.0, 'start_orchestration()'),
        (2, 3, 6.5, 'generate_decision()'),
        (3, 4, 6.0, 'get_sr_levels()'),
        (4, 3, 5.5, 'sr_context'),
        (3, 2, 5.0, 'trade_decision'),
        (2, 5, 4.5, 'execute_order()'),
        (5, 2, 4.0, 'order_result'),
        (2, 1, 3.5, 'execution_complete'),
        (1, 0, 3.0, 'tactics_result'),
    ]
    
    # Draw interactions
    for from_idx, to_idx, y, label in interactions:
        from_x = x_positions[from_idx]
        to_x = x_positions[to_idx]
        
        if from_idx < to_idx:
            ax.annotate('', xy=(to_x - 0.1, y), xytext=(from_x + 0.1, y),
                       arrowprops=dict(arrowstyle='->', lw=1.5))
        else:
            ax.annotate('', xy=(to_x + 0.1, y), xytext=(from_x - 0.1, y),
                       arrowprops=dict(arrowstyle='->', lw=1.5, linestyle='dashed'))
        
        # Add label
        mid_x = (from_x + to_x) / 2
        ax.text(mid_x, y + 0.1, label, ha='center', fontsize=8)
    
    # Add title
    ax.text(5, 9.8, 'Tactician Execution Sequence', 
            ha='center', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('docs/tactician_sequence.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("Generating Tactician architecture diagrams...")
    
    # Create diagrams
    create_architecture_diagram()
    create_data_flow_diagram()
    create_sequence_diagram()
    
    print("Diagrams generated successfully!")
    print("- tactician_architecture.png")
    print("- tactician_data_flow.png")
    print("- tactician_sequence.png")