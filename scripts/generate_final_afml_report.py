
import pandas as pd
import numpy as np
import os
from pathlib import Path
from src.utils.versioned_artifacts import VersionedArtifactStore
from src.utils.tprint import tprint_info, tprint_success

def generate_cross_specialist_report():
    context = "ETHUSDT_binance_15m_long_"
    probs_dict = {}
    
    tprint_info("📊 Loading specialist predictions for cross-correlation analysis...")
    
    # Dynamically find all enhanced specialist stores
    store_dirs = [d for d in os.listdir("versioned_artifacts") if d.startswith(context) and "enhanced" in d]
    
    for store_dir in store_dirs:
        store_path = os.path.join("versioned_artifacts", store_dir)
        try:
            store = VersionedArtifactStore(store_path)
            versions = store.list_versions()
            if not versions:
                continue
                
            # Find the most recent prediction version
            pred_versions = [v for v in versions if "prediction" in v.lower()]
            if not pred_versions:
                latest = versions[-1]
            else:
                latest = pred_versions[-1]
                
            df = store.get_view(latest).to_pandas()
            if 'specialist_probability' in df.columns:
                # Extract clean name
                name = store_dir.replace(context, '').replace('enhanced_ml_', '').replace('enhanced_xgb_', '').replace('_step', '').replace('enhanced_', '')
                probs_dict[name] = df['specialist_probability']
                tprint_info(f"  - Loaded {name} from {store_dir}")
        except Exception as e:
            print(f"Error loading {store_dir}: {e}")
            
    if not probs_dict:
        print("No predictions found to analyze.")
        return
        
    df_probs = pd.DataFrame(probs_dict)
    corr_matrix = df_probs.corr()
    
    report_path = "artifacts/cross_specialist_afml_report.md"
    with open(report_path, "w") as f:
        f.write("# Final Cross-Specialist AFML Comparison Report\n\n")
        f.write("## Specialist Probability Correlations\n\n")
        f.write("This matrix shows the correlation between the prediction probabilities of all 13 specialists. Low correlation indicates good diversity for the ensemble.\n\n")
        
        # Manual markdown table generation to avoid tabulate dependency
        headers = [""] + list(corr_matrix.columns)
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for idx, row in corr_matrix.iterrows():
            f.write(f"| **{idx}** | " + " | ".join([f"{val:.3f}" for val in row]) + " |\n")
        f.write("\n\n")
        
        f.write("## Diversity Analysis\n\n")
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    high_corr.append(f"- **{corr_matrix.columns[i]}** & **{corr_matrix.columns[j]}**: {corr_matrix.iloc[i, j]:.3f}")
        
        if high_corr:
            f.write("### High Correlation Pairs (> 0.7)\n")
            f.writelines("\n".join(high_corr) + "\n\n")
        else:
            f.write("✅ All specialists show high diversity (no pairs > 0.7 correlation).\n\n")
            
    tprint_success(f"💾 Final report saved to {report_path}")

if __name__ == "__main__":
    generate_cross_specialist_report()
