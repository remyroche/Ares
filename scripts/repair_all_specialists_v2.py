
import os
import re
import numpy as np

def fix_specialist(file_path):
    print(f"Fixing {file_path}...")
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 1. Standardize __init__
    # Use re.DOTALL to match across multiple lines
    # We target the __init__ method and replace its body or the whole method
    init_pattern = r'def __init__\(self, step_name: str = "([^"]+)"\):.*?self\.feature_pipeline = MIOptimizedFeaturePipeline\(\)'
    
    def init_replacer(match):
        step_name = match.group(1)
        # Try to find the logger child name
        logger_name_match = re.search(r'self\.logger = logger\.getChild\("([^"]+)"\)', match.group(0))
        logger_name = logger_name_match.group(1) if logger_name_match else step_name
        
        return f"""def __init__(self, step_name: str = "{step_name}"):
        \"\"\"Initialize the enhanced specialist.\"\"\"
        BaseStep.__init__(self, step_name, use_versioned_artifacts=True)
        SpecialistDiagnosticsMixinEnhancedV2.__init__(self, step_name=step_name)
        self._current_context = {{}}
        self._artifact_manager = None
        self._versioned_store = None
        self.step_name = step_name
        self.logger = logger.getChild("{logger_name}")
        self.feature_pipeline = MIOptimizedFeaturePipeline()
        self.mi_history = []
        self.training_metrics = []
        self._market_data_cache = {{}}"""

    content = re.sub(init_pattern, init_replacer, content, flags=re.DOTALL)

    # 2. Fix metrics calculation and n_samples reporting
    # Look for the block that calculates metrics after training and before standardized output
    # We'll use a more flexible regex to find where oof_probs is used to calculate auc/mi
    
    metrics_block = """
            # AFML Audit: Update metrics using full OOF set
            valid_oof = oof_probs.dropna()
            if len(valid_oof) > 0:
                y_full_true = y.loc[valid_oof.index]
                y_full_pred_prob = valid_oof.values
                
                metrics = {}
                try:
                    # Use fast binned MI proxy for binary targets
                    metrics['auc'] = float(roc_auc_score(y_full_true, y_full_pred_prob))
                    metrics['mi_score'] = float(self.compute_binned_mi(y_full_pred_prob, y_full_true.values))
                except Exception as e:
                    self.logger.warning(f"Failed to calculate full OOF metrics: {e}")
                    metrics['auc'] = 0.5
                    metrics['mi_score'] = 0.0
            else:
                metrics = {'auc': 0.5, 'mi_score': 0.0}
                y_full_pred_prob = np.array([])

            metrics.update({
                'n_features': len(X.columns),
                'optimization_params': best_params if 'best_params' in locals() else {},
                'n_samples': len(X)
            })

            # 5. Generate Final Standardized Output (Aligned to full market_data index)
            final_probs = pd.Series(0.5, index=market_data.index if 'market_data' in locals() else df.index)
            if len(valid_oof) > 0:
                final_probs.loc[valid_oof.index] = y_full_pred_prob
            final_preds = (final_probs >= 0.5).astype(int)
            
            full_labels = pd.Series(0, index=market_data.index if 'market_data' in locals() else df.index)
            full_labels.loc[y.index] = y

            standardized_output = self._create_standardized_output(
                feature_df if 'feature_df' in locals() else (features_df if 'features_df' in locals() else X), 
                full_labels, final_preds.values, final_probs.values, symbol, exchange, timeframe, direction
            )
    """
    
    # Identify the region to replace
    # It usually starts with something like "valid_oof = oof_probs.dropna()" or "# Update metrics using full OOF set"
    # and ends before "artifact_name ="
    
    # Let's try to find the start of the metrics block
    start_markers = [
        r'# Update metrics using full OOF set',
        r'valid_oof = oof_probs\.dropna\(\)'
    ]
    
    replaced = False
    for marker in start_markers:
        pattern = marker + r'.*?standardized_output = self\._create_standardized_output\(.*?\)'
        if re.search(pattern, content, re.DOTALL):
            content = re.sub(pattern, metrics_block, content, flags=re.DOTALL)
            replaced = True
            break
            
    if not replaced:
        print(f"   ⚠️ Could not find metrics block in {file_path}")

    # 3. Final return statement fix
    content = re.sub(r'"n_samples": len\([^)]+\)', r'"n_samples": len(X)', content)
    content = re.sub(r'metrics\["n_samples"\] = len\([^)]+\)', r'metrics["n_samples"] = len(X)', content)

    # 4. Special fix for Risk 'rolling' error
    if "risk" in file_path.lower():
        content = content.replace("regime_persistence = (vol_regime == 1).rolling(10).sum()", 
                                  "vol_regime_series = pd.Series(vol_regime, index=df.index)\n                regime_persistence = (vol_regime_series == 1).rolling(10).sum()")

    with open(file_path, 'w') as f:
        f.write(content)

specialists = [
    "src/training/steps/market_analysis/ml_path_regime_step_enhanced.py",
    "src/training/steps/market_analysis/ml_volume_force_step_enhanced.py",
    "src/training/steps/market_analysis/ml_risk_regime_step_enhanced.py",
    "src/training/steps/market_analysis/ml_volatility_burst_step_enhanced.py",
    "src/training/steps/market_analysis/ml_reversion_regime_step_enhanced.py",
    "src/training/steps/market_analysis/ml_momentum_persistence_step_enhanced.py",
    "src/training/steps/market_analysis/ml_liquidity_regime_step_enhanced.py",
    "src/training/steps/market_analysis/ml_smc_regime_step_enhanced.py",
    "src/training/steps/market_analysis/ml_spectral_step_enhanced.py",
    "src/training/steps/market_analysis/ml_microstructure_step_enhanced.py",
    "src/training/steps/market_analysis/xgb_macro_regime_step_enhanced.py",
    "src/training/steps/market_analysis/xgb_meso_regime_step_enhanced.py"
]

for f in specialists:
    if os.path.exists(f):
        fix_specialist(f)
    else:
        print(f"File not found: {f}")
