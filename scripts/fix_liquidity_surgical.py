import pandas as pd
import numpy as np
from pathlib import Path

file_path = "src/training/steps/market_analysis/ml_liquidity_regime_step_enhanced.py"
with open(file_path, 'r') as f:
    lines = f.readlines()

# Find the start of the corrupted block
start_idx = -1
for i, line in enumerate(lines):
    if "standardized_output = self._create_standardized_output(" in line:
        start_idx = i
        break

if start_idx != -1:
    # Keep everything up to the first standardized_output call
    new_lines = lines[:start_idx]
    
    # Add clean tail
    new_lines.extend([
        "            standardized_output = self._create_standardized_output(\n",
        "                feature_df if 'feature_df' in locals() else (features_df if 'features_df' in locals() else X), \n",
        "                full_labels, final_preds.values, final_probs.values, symbol, exchange, timeframe, direction\n",
        "            )\n",
        "\n",
        "            # 9. Save Artifacts\n",
        "            artifact_name = f\"enhanced_ml_liquidity_predictions_{timeframe}\"\n",
        "            metadata = SpecialistDataInterface.create_standard_metadata(\n",
        "                specialist_name=\"EnhancedMLLiquidityRegimeStep\",\n",
        "                config=config,\n",
        "                metrics=metrics,\n",
        "                mi_score=metrics['mi_score'],\n",
        "                hsic_score=0.0\n",
        "            )\n",
        "            \n",
        "            artifact_path = self._save_artifact(\n",
        "                data=standardized_output,\n",
        "                artifact_name=artifact_name,\n",
        "                artifact_type=\"data\",\n",
        "                data_category=\"predictions\",\n",
        "                metadata=metadata\n",
        "            )\n",
        "            artifacts.append(artifact_path)\n",
        "\n",
        "            # 10. Run Enhanced Diagnostics\n",
        "            tprint_info(\"🔍 Running Enhanced Diagnostics...\")\n",
        "            diagnostics_result = self.run_enhanced_diagnostics(symbol, exchange, timeframe, direction)\n",
        "            \n",
        "            if diagnostics_result.get('success', False):\n",
        "                compliance_report = diagnostics_result['compliance_report']\n",
        "                ensemble_compatibility = diagnostics_result['ensemble_compatibility']\n",
        "                \n",
        "                tprint_success(f\"✅ Enhanced Diagnostics Complete:\")\n",
        "                tprint_info(f\"   MI Score: {compliance_report['metrics']['mi_score']:.4f}\")\n",
        "                tprint_info(f\"   Requirements Met: {compliance_report['requirements_met']}/3\")\n",
        "                tprint_info(f\"   Ensemble Ready: {ensemble_compatibility['ensemble_ready']}\")\n",
        "                \n",
        "                metrics.update({\n",
        "                    'enhanced_mi_score': compliance_report['metrics']['mi_score'],\n",
        "                    'enhanced_requirements_met': compliance_report['requirements_met'],\n",
        "                    'enhanced_ensemble_ready': ensemble_compatibility['ensemble_ready']\n",
        "                })\n",
        "\n",
        "            # 11. Final Summary\n",
        "            execution_time = time.time() - start_time\n",
        "            metrics[\"execution_time\"] = execution_time\n",
        "            metrics[\"n_samples\"] = len(X)\n",
        "\n",
        "            tprint_success(f\"✅ Enhanced Liquidity Regime completed in {execution_time:.2f}s\")\n",
        "            tprint_info(f\"📊 Final Metrics: MI={metrics.get('mi_score', 0):.4f}, AUC={metrics.get('auc', 0):.3f}\")\n",
        "\n",
        "            return {\n",
        "                \"success\": True,\n",
        "                \"metrics\": metrics,\n",
        "                \"n_samples\": len(X),\n",
        "                \"features\": list(X.columns),\n",
        "                \"artifacts\": artifacts,\n",
        "                \"diagnostics\": diagnostics_result,\n",
        "                \"mi_history\": self.mi_history,\n",
        "                \"training_metrics\": self.training_metrics,\n",
        "                \"execution_time\": execution_time\n",
        "            }\n",
        "\n",
        "        except Exception as e:\n",
        "            self.logger.exception(f\"❌ Enhanced Liquidity Regime step failed: {e}\")\n",
        "            return {\"success\": False, \"error\": str(e)}\n",
        "    \n",
        "    def _create_standardized_output(self, features: pd.DataFrame, labels: pd.Series,\n",
        "                                  predictions: np.ndarray, probabilities: np.ndarray,\n",
        "                                  symbol: str, exchange: str, timeframe: str, direction: str) -> pd.DataFrame:\n",
        "        \"\"\"Create standardized output structure.\"\"\"\n",
        "        standardized = pd.DataFrame(index=features.index)\n",
        "        standardized['timestamp'] = features.index\n",
        "        standardized['specialist_prediction'] = predictions\n",
        "        standardized['specialist_probability'] = probabilities\n",
        "        standardized['target_label'] = labels\n",
        "        \n",
        "        # Add original features for reference\n",
        "        for col in features.columns[:20]:  # Limit to first 20 features\n",
        "            standardized[f'feature_{col}'] = features[col]\n",
        "        \n",
        "        return standardized\n",
        "    \n",
        "    def _load_market_data_with_cache(self, config: Dict[str, Any], timeframe: str) -> Tuple[pd.DataFrame, str]:\n",
        "        \"\"\"Load market data with caching.\"\"\"\n",
        "        market_data, market_source = self.load_market_data_or_fail(\n",
        "            {**config, \"timeframe\": timeframe},\n",
        "            pipeline_state={},\n",
        "            allow_config_override=True,\n",
        "        )\n",
        "        return market_data, market_source\n"
    ])
    
    with open(file_path, 'w') as f:
        f.writelines(new_lines)
    print(f"Fixed {file_path}")
