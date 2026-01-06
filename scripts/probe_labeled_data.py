
import pandas as pd
import numpy as np
from src.training.steps.labeling.feature_generation_meta_labeling_step import FeatureGenerationMetaLabelingStep

def probe():
    step = FeatureGenerationMetaLabelingStep()
    step.set_context(
        symbol="ETHUSDT",
        exchange="binance",
        timeframe="15m",
        direction="long",
        model="analyst",
    )
    
    artifact_name = "labeled_data_ETHUSDT_15m"
    try:
        df = step._get_artifact(
            artifact_name=artifact_name,
            artifact_type="data",
            data_category="features",
        )
        print(f"Artifact {artifact_name} found. Shape: {df.shape}")
        print("\nColumns and non-null counts:")
        print(df.count().to_string())
        
        target_cols = [c for c in df.columns if 'target' in c or 'label' in c]
        print(f"\nTarget columns: {target_cols}")
        for col in target_cols:
            if col in df.columns:
                print(f"{col} non-null: {df[col].notna().sum()}")
                print(f"{col} value counts:\n{df[col].value_counts()}")
                
    except Exception as e:
        print(f"Error loading {artifact_name}: {e}")

if __name__ == "__main__":
    probe()
