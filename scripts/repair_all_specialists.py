
import os
import re

def fix_specialist(file_path):
    print(f"Fixing {file_path}...")
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 1. Fix __init__ duplication and ensure mi_history/training_metrics
    # Find the whole __init__ block and replace it
    init_match = re.search(r'def __init__\(self, step_name: str = "[^"]+"\):.*?(?=def|\Z)', content, re.DOTALL)
    if init_match:
        init_block = init_match.group(0)
        # Extract step_name from the existing block
        step_name_match = re.search(r'step_name: str = "([^"]+)"', init_block)
        step_name = step_name_match.group(1) if step_name_match else "unknown"
        
        # Extract logger name if possible
        logger_match = re.search(r'self.logger = logger.getChild\("([^"]+)"\)', init_block)
        logger_name = logger_match.group(1) if logger_match else "Specialist"
        
        # Determine base classes for super calls
        is_xgb = "xgb" in file_path.lower()
        
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
        content = content.replace(init_block, new_init)

    # 2. Fix execute method metrics and n_samples logic
    # We want to find the part where metrics are calculated after the loop
    # and replace it with a clean version.
    
    # First, ensure valid_oof, y_full_true, y_full_pred_prob are defined correctly
    metrics_logic = """
            # Update metrics using full OOF set
            valid_oof = oof_probs.dropna()
            if len(valid_oof) > 0:
                y_full_true = y.loc[valid_oof.index]
                y_full_pred_prob = valid_oof.values
                
                metrics = {}
                try:
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
                'optimization_params': best_params if 'best_params' in locals() else {}
            })

            # 5. Generate Final Standardized Output (Aligned to full market_data index)
            final_probs = pd.Series(0.5, index=market_data.index)
            if len(valid_oof) > 0:
                final_probs.loc[valid_oof.index] = y_full_pred_prob
            final_preds = (final_probs >= 0.5).astype(int)
            
            full_labels = pd.Series(0, index=market_data.index)
            full_labels.loc[y.index] = y

            standardized_output = self._create_standardized_output(
                feature_df if 'feature_df' in locals() else features_df, 
                full_labels, final_preds.values, final_probs.values, symbol, exchange, timeframe, direction
            )
    """
    
    # This replacement is tricky because the context varies. 
    # I'll look for the comment "# Update metrics using full OOF set" and replace until standardized_output
    pattern = r'# Update metrics using full OOF set.*?standardized_output = self\._create_standardized_output\(.*?\)'
    content = re.sub(pattern, metrics_logic, content, flags=re.DOTALL)

    # 3. Fix the final return statement to use n_samples = len(X)
    content = re.sub(r'"n_samples": len\([^)]+\)', r'"n_samples": len(X)', content)
    content = re.sub(r'metrics\["n_samples"\] = len\([^)]+\)', r'metrics["n_samples"] = len(X)', content)

    # 4. Special fix for Risk 'rolling' error
    if "risk" in file_path.lower():
        content = content.replace("regime_persistence = (vol_regime == 1).rolling(10).sum()", 
                                  "vol_regime_series = pd.Series(vol_regime, index=df.index)\n                regime_persistence = (vol_regime_series == 1).rolling(10).sum()")

    with open(file_path, 'w') as f:
        f.write(content)

specialists = [
    "src/training/steps/market_analysis/ml_volume_force_step_enhanced.py",
    "src/training/steps/market_analysis/ml_risk_regime_step_enhanced.py",
    "src/training/steps/market_analysis/ml_volatility_burst_step_enhanced.py",
    "src/training/steps/market_analysis/ml_momentum_persistence_step_enhanced.py",
    "src/training/steps/market_analysis/ml_smc_regime_step_enhanced.py",
    "src/training/steps/market_analysis/ml_spectral_step_enhanced.py",
    "src/training/steps/market_analysis/xgb_macro_regime_step_enhanced.py",
    "src/training/steps/market_analysis/xgb_meso_regime_step_enhanced.py"
]

for f in specialists:
    if os.path.exists(f):
        fix_specialist(f)
    else:
        print(f"File not found: {f}")
