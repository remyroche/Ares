
import os
import re
import numpy as np

def fix_specialist(file_path):
    print(f"Fixing {file_path}...")
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 1. Standardize __init__
    # Target the __init__ method and ensure all necessary variables are initialized once.
    init_match = re.search(r'def __init__\(self, step_name: str = "([^"]+)"\):.*?(?=\n    (?:def|@property))', content, re.DOTALL)
    if init_match:
        step_name = init_match.group(1)
        logger_name_match = re.search(r'self\.logger = logger\.getChild\("([^"]+)"\)', init_match.group(0))
        logger_name = logger_name_match.group(1) if logger_name_match else step_name
        
        new_init = f"""def __init__(self, step_name: str = "{step_name}"):
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
        self._market_data_cache = {{}}
        tprint(f"✅ Initialized Enhanced {{step_name}} (MI-Optimized)", "SUCCESS")
    """
        content = content[:init_match.start()] + new_init + content[init_match.end():]

    # 2. Fix the metrics and output block
    # We'll use a more surgical approach to find the metrics calculation part.
    
    new_metrics_logic = """
            # AFML Audit: Update metrics using full OOF set
            valid_oof = oof_probs.dropna()
            if len(valid_oof) > 0:
                y_full_true = y.loc[valid_oof.index]
                y_full_pred_prob = valid_oof.values
                y_full_pred = (y_full_pred_prob >= 0.5).astype(int)
                
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
                y_full_pred = np.array([])

            metrics.update({
                'n_features': len(X.columns),
                'optimization_params': best_params if 'best_params' in locals() else {},
                'n_samples': len(X)
            })

            # 5. Generate Final Standardized Output (Aligned to full market_data index)
            final_probs = pd.Series(0.5, index=market_data.index if 'market_data' in locals() else (df.index if 'df' in locals() else X.index))
            if len(valid_oof) > 0:
                final_probs.loc[valid_oof.index] = y_full_pred_prob
            final_preds = (final_probs >= 0.5).astype(int)
            
            full_labels = pd.Series(0, index=market_data.index if 'market_data' in locals() else (df.index if 'df' in locals() else X.index))
            full_labels.loc[y.index] = y

            standardized_output = self._create_standardized_output(
                feature_df if 'feature_df' in locals() else (features_df if 'features_df' in locals() else X), 
                full_labels, final_preds.values, final_probs.values, symbol, exchange, timeframe, direction
            )
    """

    # Identify the region to replace
    # We look for where the OOF probabilities are aligned or metrics are calculated.
    # It usually starts with valid_oof or a comment about OOF metrics.
    
    pattern = r'(?:# Update metrics using full OOF set|# Re-calculate metrics on full OOF set|valid_oof = oof_probs\.dropna\(\)).*?standardized_output = self\._create_standardized_output\(.*?\)'
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, new_metrics_logic, content, flags=re.DOTALL)
    else:
        print(f"   ⚠️ Could not find metrics block in {file_path}")

    # 3. Fix the final return statement to use n_samples = len(X)
    # Ensure n_samples is reported consistently
    content = re.sub(r'"n_samples":\s*len\([^)]+\)', r'"n_samples": len(X)', content)
    content = re.sub(r'metrics\["n_samples"\]\s*=\s*len\([^)]+\)', r'metrics["n_samples"] = len(X)', content)

    # 4. Special fix for Risk 'rolling' error
    if "risk" in file_path.lower():
        # Fix all potential numpy rolling calls
        content = content.replace("vol_regime == 1).rolling(10).sum()", 
                                  "pd.Series(vol_regime, index=df.index) == 1).rolling(10).sum()")
        content = content.replace("vol_regime).diff(", 
                                  "pd.Series(vol_regime, index=df.index)).diff(")

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
