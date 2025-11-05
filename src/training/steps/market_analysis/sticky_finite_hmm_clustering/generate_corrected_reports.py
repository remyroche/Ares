#!/usr/bin/env python3
"""
Generate corrected reports with proper executive summary and CSV export.
"""

import csv
from datetime import datetime
from pathlib import Path

def generate_corrected_trial_analysis():
    """Generate corrected trial analysis with proper executive summary."""
    
    # Read the existing trial analysis
    existing_file = "outcomes/ETHUSDT_StickyFiniteHMM_Trial_Analysis_20251105_001456.md"
    
    if Path(existing_file).exists():
        with open(existing_file, 'r') as f:
            content = f.read()
        
        # Fix the executive summary with actual best trial data
        # From the table, trial 18 has the best score: 0.2375
        corrected_content = content.replace(
            "- **Best Composite Score**: 0.0000\n- **Optimal Parameters**: {}\n- **Optimal Number of Regimes**: N/A\n- **Best ELBO**: N/A",
            "- **Best Composite Score**: 0.2375\n- **Optimal Parameters**: {'K': 3, 'base_alpha': 0.7, 'kappa': 25.0, 'n_mixtures': 2, 'svi_iterations': 1000, 'learning_rate': 0.05}\n- **Optimal Number of Regimes**: 3\n- **Best ELBO**: -2046.71"
        )
        
        # Write corrected version
        corrected_file = "outcomes/ETHUSDT_StickyFiniteHMM_Trial_Analysis_CORRECTED_20251105_001456.md"
        with open(corrected_file, 'w') as f:
            f.write(corrected_content)
        
        print(f"✅ Corrected trial analysis generated: {corrected_file}")
        return corrected_file
    else:
        print(f"❌ Original file not found: {existing_file}")
        return None

def generate_quality_metrics_csv():
    """Generate CSV with quality metrics from the best trial."""
    
    # Best trial metrics from the analysis
    metrics = {
        'Composite Quality Score': 0.2375,
        'Silhouette Score': -0.0121,
        'Davies-Bouldin Index': 51.2992,
        'Calinski-Harabasz Index': 3.17,
        'Temporal Smoothness': 0.9498,
        'Regime Balance': 0.9383,
        'Number of Regimes': 3,
        'Total Samples': 26279,
        'Regime Changes': 1320
    }
    
    # Create CSV
    csv_file = "outcomes/cluster_quality_metrics_ETHUSDT_StickyFiniteHMM_20251105_001456.csv"
    
    csv_data = []
    csv_data.append(['Metric', 'Value', 'Description'])
    csv_data.append(['Composite Quality Score', metrics['Composite Quality Score'], 'Overall clustering quality (0-1, higher is better)'])
    csv_data.append(['Silhouette Score', metrics['Silhouette Score'], 'Cluster separation and cohesion'])
    csv_data.append(['Davies-Bouldin Index', metrics['Davies-Bouldin Index'], 'Cluster similarity (lower is better)'])
    csv_data.append(['Calinski-Harabasz Index', metrics['Calinski-Harabasz Index'], 'Between-cluster dispersion'])
    csv_data.append(['Temporal Smoothness', metrics['Temporal Smoothness'], 'Regime persistence over time'])
    csv_data.append(['Regime Balance', metrics['Regime Balance'], 'Equitability of regime sizes'])
    csv_data.append(['Number of Regimes', metrics['Number of Regimes'], 'Distinct market regimes discovered'])
    csv_data.append(['Total Samples', metrics['Total Samples'], 'Data points analyzed'])
    csv_data.append(['Regime Changes', metrics['Regime Changes'], 'Number of regime transitions'])
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)
    
    print(f"✅ Quality metrics CSV generated: {csv_file}")
    return csv_file

def main():
    """Main function to generate corrected reports."""
    
    print("🔧 Generating Corrected Reports")
    print("=" * 50)
    
    # Create outcomes directory
    Path("outcomes").mkdir(exist_ok=True)
    
    # Generate corrected trial analysis
    corrected_md = generate_corrected_trial_analysis()
    
    # Generate quality metrics CSV
    csv_file = generate_quality_metrics_csv()
    
    print("\n📊 Summary:")
    print(f"✅ Corrected Markdown Report: {corrected_md}")
    print(f"✅ Quality Metrics CSV: {csv_file}")
    
    if corrected_md and csv_file:
        print("\n🎉 All corrected reports generated successfully!")
        print("\n📋 Fixed Issues:")
        print("✅ Executive Summary now shows actual best trial data")
        print("✅ Optimal Parameters: K=3, α=0.7, κ=25.0, 2 mixtures, 1000 SVI iterations")
        print("✅ Best ELBO: -2046.71")
        print("✅ CSV export with detailed metrics created")
        return True
    else:
        print("\n❌ Some reports failed to generate")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
