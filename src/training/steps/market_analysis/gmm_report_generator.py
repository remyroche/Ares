
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import json
import os
from src.utils.tprint import tprint_info, tprint_success, tprint_warning

class GMMReportGenerator:
    """
    Generates detailed reports for GMM regime detection results.
    Produces summaries of regime characteristics, transition probabilities, and feature importance.
    """
    
    def __init__(self, output_dir: str = "outcomes/gmm_reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def generate_report(self, 
                       gmm_results: Dict[str, Any], 
                       symbol: str, 
                       timeframe: str) -> Dict[str, Any]:
        """
        Generate comprehensive GMM report from engine results.
        
        Args:
            gmm_results: Results dictionary from QuantitativeRegimeEngine
            symbol: Trading symbol
            timeframe: Data timeframe
            
        Returns:
            Dictionary containing the report summary
        """
        tprint_info(f"📊 Generating GMM Report for {symbol} ({timeframe})...")
        
        report = {
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": gmm_results.get("timestamp"),
            "pipelines": {}
        }
        
        # Process Returns Pipeline
        if "returns_pipeline" in gmm_results:
            report["pipelines"]["returns"] = self._analyze_pipeline(
                gmm_results["returns_pipeline"], "Returns"
            )
            
        # Process FracDiff Pipeline
        if "fracdiff_pipeline" in gmm_results:
            report["pipelines"]["fracdiff"] = self._analyze_pipeline(
                gmm_results["fracdiff_pipeline"], "FracDiff"
            )
            
        # Save detailed JSON report
        report_path = os.path.join(self.output_dir, f"gmm_report_{symbol}_{timeframe}.json")
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4, default=str)
            tprint_success(f"✅ GMM JSON Report saved to {report_path}")
        except Exception as e:
            tprint_warning(f"⚠️ Failed to save JSON report: {e}")
            
        # Generate Text Summary
        self._save_text_summary(report, symbol, timeframe)
        
        return report
    
    def _analyze_pipeline(self, pipeline_result: Dict[str, Any], name: str) -> Dict[str, Any]:
        """Analyze a single GMM pipeline result."""
        
        if "error" in pipeline_result:
            return {"status": "failed", "error": pipeline_result["error"]}
            
        model_params = pipeline_result.get("model_params", {})
        features = pipeline_result.get("features", pd.DataFrame())
        
        analysis = {
            "status": "success",
            "n_features": pipeline_result.get("n_features", 0),
            "n_regimes": len(model_params.get("weights", [])),
            "regime_distribution": {},
            "transition_matrix": model_params.get("transition_matrix", []),
            "regime_persistence": {},
            "regime_characteristics": {}
        }
        
        # 1. Regime Distribution
        if not features.empty:
            regime_cols = [c for c in features.columns if c.startswith("REGIME_") and not c.endswith("VELOCITY") and not c.endswith("INTEGRITY")]
            if regime_cols:
                # Calculate dominant regime
                dominant_regimes = features[regime_cols].idxmax(axis=1)
                dist_counts = dominant_regimes.value_counts(normalize=True)
                analysis["regime_distribution"] = dist_counts.to_dict()
                
                # 2. Persistence (Diagonal of transition matrix)
                trans_mat = np.array(model_params.get("transition_matrix", []))
                if trans_mat.size > 0:
                    for i in range(len(trans_mat)):
                        analysis["regime_persistence"][f"REGIME_{i}"] = trans_mat[i, i]
                        
                # 3. Regime Characteristics (if target present)
                target = pipeline_result.get("target_innovations")
                if target is not None and not target.empty:
                    # Align target with features
                    common_idx = features.index.intersection(target.index)
                    if not common_idx.empty:
                        aligned_target = target.loc[common_idx]
                        aligned_regimes = dominant_regimes.loc[common_idx]
                        
                        # Group by regime
                        for regime in analysis["regime_distribution"].keys():
                            mask = aligned_regimes == regime
                            if mask.any():
                                regime_data = aligned_target[mask]
                                analysis["regime_characteristics"][regime] = {
                                    "mean": float(regime_data.mean()),
                                    "std": float(regime_data.std()),
                                    "skew": float(regime_data.skew()),
                                    "kurt": float(regime_data.kurt()),
                                    # Financial Metrics (assuming 15m data for annualization: 4 * 24 * 365 / horizon)
                                    # Here using approx 24192 for 15m annualization (4 * 24 * 252)
                                    "sharpe": float(regime_data.mean() / (regime_data.std() + 1e-9) * np.sqrt(24192)),
                                    "ann_return": float(regime_data.mean() * 24192)
                                }

        return analysis

    def _save_text_summary(self, report: Dict[str, Any], symbol: str, timeframe: str):
        """Save a human-readable text summary."""
        summary_path = os.path.join(self.output_dir, f"gmm_summary_{symbol}_{timeframe}.txt")
        
        lines = []
        lines.append(f"GMM REGIME DETECTION REPORT: {symbol} [{timeframe}]")
        lines.append("=" * 60)
        lines.append("")
        
        for name, pipe in report["pipelines"].items():
            lines.append(f"PIPELINE: {name.upper()}")
            lines.append("-" * 30)
            
            if pipe["status"] == "failed":
                lines.append(f"Status: FAILED ({pipe.get('error')})")
                continue
                
            lines.append(f"Features Used: {pipe['n_features']}")
            lines.append(f"Regimes Detected: {pipe['n_regimes']}")
            lines.append("")
            
            lines.append("Regime Distribution:")
            for regime, freq in pipe.get("regime_distribution", {}).items():
                lines.append(f"  - {regime}: {freq:.1%}")
            lines.append("")
            
            lines.append("Regime Persistence (Self-Transition Prob):")
            for regime, prob in pipe.get("regime_persistence", {}).items():
                lines.append(f"  - {regime}: {prob:.1%}")
            lines.append("")

            if pipe.get("regime_characteristics"):
                lines.append("Regime Characteristics (Target Innovations):")
                # Sort regimes by volatility (std) to identify Quiet vs Volatile
                sorted_regimes = sorted(
                    pipe["regime_characteristics"].items(),
                    key=lambda x: x[1].get("std", 0)
                )
                
                for regime, stats in sorted_regimes:
                    lines.append(f"  - {regime}:")
                    lines.append(f"      Mean: {stats['mean']:.6f}")
                    lines.append(f"      Std:  {stats['std']:.6f}")
                    lines.append(f"      Sharpe: {stats.get('sharpe', 0.0):.2f}")
                    lines.append(f"      Ann. Ret: {stats.get('ann_return', 0.0):.2%}")
                    lines.append(f"      Skew: {stats['skew']:.4f}")
                lines.append("")

            lines.append("Transition Matrix:")
            trans_mat = pipe.get("transition_matrix", [])
            if trans_mat:
                # Format simple matrix
                tm = np.array(trans_mat)
                for i, row in enumerate(tm):
                    row_str = "  ".join([f"{val:.2f}" for val in row])
                    lines.append(f"  R{i}: [{row_str}]")
            lines.append("")
            lines.append("=" * 60)
            lines.append("")
            
        try:
            with open(summary_path, 'w') as f:
                f.write("\n".join(lines))
            tprint_success(f"✅ GMM Text Summary saved to {summary_path}")
        except Exception as e:
            tprint_warning(f"⚠️ Failed to save text summary: {e}")
