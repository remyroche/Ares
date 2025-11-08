"""
Model Training Report Generator

This module provides utilities for generating detailed markdown and JSON reports
for model training metrics including HPO results, accuracy, R2 scores, and other
performance metrics for analyst and tactician models.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List


class ModelTrainingReportGenerator:
    """
    Generates comprehensive markdown and JSON reports for model training results.
    """

    def __init__(self, outcomes_dir: str = "outcomes"):
        """
        Initialize the report generator.

        Args:
            outcomes_dir: Directory to store reports (default: "outcomes")
        """
        self.outcomes_dir = Path(outcomes_dir)
        self.outcomes_dir.mkdir(parents=True, exist_ok=True)

    def generate_training_report(self,
                                training_type: str,
                                symbol: str,
                                exchange: str,
                                timeframe: str,
                                direction: str,
                                models_trained: Dict[str, Any],
                                metrics: Dict[str, Any],
                                hpo_results: Optional[Dict[str, Any]] = None,
                                regime_performance: Optional[Dict[str, Any]] = None,
                                training_config: Optional[Dict[str, Any]] = None,
                                feature_info: Optional[Dict[str, Any]] = None,
                                execution_time: Optional[float] = None) -> tuple[str, str]:
        """
        Generate both markdown and JSON reports for model training.

        Args:
            training_type: Type of training (analyst_base, analyst_ensemble, etc.)
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            direction: Trading direction
            models_trained: Dictionary of trained models
            metrics: Performance metrics
            hpo_results: HPO optimization results
            regime_performance: Performance by regime
            training_config: Training configuration used
            feature_info: Information about features used
            execution_time: Total execution time in seconds

        Returns:
            Tuple of (markdown_path, json_path)
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Generate markdown report
            markdown_content = self._generate_markdown_report(
                training_type, symbol, exchange, timeframe, direction,
                models_trained, metrics, hpo_results, regime_performance,
                training_config, feature_info, execution_time
            )

            markdown_filename = f"{training_type}_{symbol}_{timeframe}_{direction}_report_{timestamp}.md"
            markdown_path = self.outcomes_dir / markdown_filename

            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            # Generate JSON report
            json_data = self._generate_json_report(
                training_type, symbol, exchange, timeframe, direction,
                models_trained, metrics, hpo_results, regime_performance,
                training_config, feature_info, execution_time
            )

            json_filename = f"{training_type}_{symbol}_{timeframe}_{direction}_metrics_{timestamp}.json"
            json_path = self.outcomes_dir / json_filename

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, default=str)

            return str(markdown_path), str(json_path)

        except Exception as e:
            print(f"Error generating training reports: {e}")
            return "", ""

    def _generate_markdown_report(self,
                                  training_type: str,
                                  symbol: str,
                                  exchange: str,
                                  timeframe: str,
                                  direction: str,
                                  models_trained: Dict[str, Any],
                                  metrics: Dict[str, Any],
                                  hpo_results: Optional[Dict[str, Any]],
                                  regime_performance: Optional[Dict[str, Any]],
                                  training_config: Optional[Dict[str, Any]],
                                  feature_info: Optional[Dict[str, Any]],
                                  execution_time: Optional[float]) -> str:
        """Generate the markdown report content."""

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

        report = f"""# {training_type.replace('_', ' ').title()} - Training Report

## 📊 Executive Summary

**Generated:** {timestamp}
**Training Type:** {training_type}
**Symbol:** {symbol}
**Exchange:** {exchange}
**Timeframe:** {timeframe}
**Direction:** {direction}
**Execution Time:** {f'{execution_time:.2f}s' if execution_time else 'N/A'}

---

## 🤖 Models Trained

**Total Models:** {len(models_trained)}
**Model Types:** {', '.join(models_trained.keys())}

"""

        # Per-model details
        report += "### Per-Model Details\n\n"
        for model_name, model_info in models_trained.items():
            report += f"#### {model_name.upper()}\n\n"
            if isinstance(model_info, dict):
                if 'type' in model_info:
                    report += f"- **Type:** {model_info.get('type', 'N/A')}\n"
                if 'algorithm' in model_info:
                    report += f"- **Algorithm:** {model_info.get('algorithm', 'N/A')}\n"
                if 'training_samples' in model_info:
                    report += f"- **Training Samples:** {model_info.get('training_samples', 'N/A'):,}\n"

                # Model-specific parameters
                params_to_show = ['n_estimators', 'max_depth', 'learning_rate', 'num_leaves',
                                 'iterations', 'depth', 'l2_leaf_reg']
                for param in params_to_show:
                    if param in model_info:
                        report += f"- **{param}:** {model_info[param]}\n"
            report += "\n"

        # Overall metrics
        report += "## 📈 Performance Metrics\n\n"
        report += "### Overall Performance\n\n"

        metrics_to_display = {
            'overall_accuracy': 'Overall Accuracy',
            'overall_precision': 'Overall Precision',
            'overall_recall': 'Overall Recall',
            'overall_f1_score': 'Overall F1 Score',
            'overall_r2_score': 'Overall R² Score',
            'overall_mse': 'Overall MSE',
            'overall_mae': 'Overall MAE',
            'best_model': 'Best Model',
            'model_count': 'Model Count'
        }

        for metric_key, metric_label in metrics_to_display.items():
            if metric_key in metrics:
                value = metrics[metric_key]
                if isinstance(value, (int, float)):
                    if metric_key in ['overall_mse', 'overall_mae']:
                        report += f"- **{metric_label}:** {value:.6f}\n"
                    elif metric_key not in ['model_count']:
                        report += f"- **{metric_label}:** {value:.4f}\n"
                    else:
                        report += f"- **{metric_label}:** {value}\n"
                else:
                    report += f"- **{metric_label}:** {value}\n"

        report += "\n### Per-Model Metrics\n\n"

        # Individual model metrics
        model_metric_keys = [k for k in metrics.keys() if any(k.startswith(f"{m}_") for m in models_trained.keys())]

        for model_name in models_trained.keys():
            model_specific_metrics = {k: v for k, v in metrics.items() if k.startswith(f"{model_name}_")}
            if model_specific_metrics:
                report += f"#### {model_name.upper()}\n\n"
                for metric_key, metric_value in model_specific_metrics.items():
                    metric_label = metric_key.replace(f"{model_name}_", "").replace("_", " ").title()
                    if isinstance(metric_value, (int, float)) and not isinstance(metric_value, bool):
                        report += f"- **{metric_label}:** {metric_value:.4f}\n"
                    else:
                        report += f"- **{metric_label}:** {metric_value}\n"
                report += "\n"

        # HPO Results
        if hpo_results:
            report += "## 🔍 Hyperparameter Optimization (HPO) Results\n\n"

            if 'best_params' in hpo_results:
                report += "### Best Parameters Found\n\n"
                for model_name, params in hpo_results['best_params'].items():
                    report += f"#### {model_name.upper()}\n\n"
                    if isinstance(params, dict):
                        for param_name, param_value in params.items():
                            report += f"- **{param_name}:** {param_value}\n"
                    report += "\n"

            if 'optimization_metrics' in hpo_results:
                report += "### Optimization Metrics\n\n"
                opt_metrics = hpo_results['optimization_metrics']
                if isinstance(opt_metrics, dict):
                    for metric_name, metric_value in opt_metrics.items():
                        if isinstance(metric_value, (int, float)):
                            report += f"- **{metric_name.replace('_', ' ').title()}:** {metric_value:.4f}\n"
                        else:
                            report += f"- **{metric_name.replace('_', ' ').title()}:** {metric_value}\n"
                report += "\n"

        # Regime Performance
        if regime_performance:
            report += "## 🌍 Regime-Based Performance\n\n"

            for regime_name, regime_metrics in regime_performance.items():
                report += f"### {regime_name.replace('_', ' ').title()}\n\n"
                if isinstance(regime_metrics, dict):
                    for metric_name, metric_value in regime_metrics.items():
                        if isinstance(metric_value, (int, float)):
                            report += f"- **{metric_name.title()}:** {metric_value:.4f}\n"
                        else:
                            report += f"- **{metric_name.title()}:** {metric_value}\n"
                report += "\n"

        # Feature Information
        if feature_info:
            report += "## 📋 Feature Information\n\n"

            if 'feature_count' in feature_info:
                report += f"**Total Features:** {feature_info['feature_count']}\n\n"

            if 'feature_source' in feature_info:
                report += f"**Feature Source:** {feature_info['feature_source']}\n\n"

            if 'feature_names' in feature_info and feature_info['feature_names']:
                features = feature_info['feature_names']
                if len(features) <= 20:
                    report += "**Features Used:**\n\n"
                    for i, feature in enumerate(features, 1):
                        report += f"{i}. {feature}\n"
                else:
                    report += f"**Features Used:** {len(features)} features\n\n"
                    report += "**First 20 Features:**\n\n"
                    for i, feature in enumerate(features[:20], 1):
                        report += f"{i}. {feature}\n"
                    report += f"\n*... and {len(features) - 20} more features*\n"
                report += "\n"

            if 'regime_features_included' in feature_info:
                report += f"**Regime Features Included:** {feature_info['regime_features_included']}\n\n"

        # Training Configuration
        if training_config:
            report += "## ⚙️ Training Configuration\n\n"

            important_configs = [
                'feature_set_size',
                'use_hpo',
                'hpo_trials',
                'validation_split',
                'test_split',
                'cross_validation_folds',
                'early_stopping',
                'max_epochs'
            ]

            for config_key in important_configs:
                if config_key in training_config:
                    config_label = config_key.replace('_', ' ').title()
                    report += f"- **{config_label}:** {training_config[config_key]}\n"

            report += "\n"

        # Data Sources
        report += "## 📦 Data Sources (HDF5 Artifacts)\n\n"
        report += "**Loaded from Versioned Artifacts:**\n\n"
        report += "1. **Features:** `feature_generation_final_feature_selection_step`\n"
        report += "   - Selected feature set (60 features or configured size)\n"
        report += "   - Stored in HDF5 format via versioned artifacts\n\n"
        report += "2. **Labels/Targets:** `feature_generation_labeling_integration_step`\n"
        report += "   - Direction-specific targets (long/short)\n"
        report += "   - Volume-based confidence adjustments\n\n"
        report += "3. **Regime Probabilities:** `regime_ensemble_training`\n"
        report += "   - Regime ensemble predictions\n"
        report += "   - Market regime classification probabilities\n\n"

        # Model Persistence
        report += "## 💾 Model Persistence\n\n"
        report += "**Format:** Pickle (.pkl)\n\n"
        report += "**Storage Location:** `/artifacts` directory\n\n"
        report += "**Models Saved:**\n\n"
        for i, model_name in enumerate(models_trained.keys(), 1):
            report += f"{i}. `{training_type}_{model_name}.pkl`\n"
        report += "\n"

        # Footer
        report += "---\n\n"
        report += f"*Report generated by ModelTrainingReportGenerator on {timestamp}*\n"

        return report

    def _generate_json_report(self,
                             training_type: str,
                             symbol: str,
                             exchange: str,
                             timeframe: str,
                             direction: str,
                             models_trained: Dict[str, Any],
                             metrics: Dict[str, Any],
                             hpo_results: Optional[Dict[str, Any]],
                             regime_performance: Optional[Dict[str, Any]],
                             training_config: Optional[Dict[str, Any]],
                             feature_info: Optional[Dict[str, Any]],
                             execution_time: Optional[float]) -> Dict[str, Any]:
        """Generate the JSON report data."""

        json_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "training_type": training_type,
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "direction": direction,
                "execution_time_seconds": execution_time
            },
            "models_trained": {
                "count": len(models_trained),
                "model_names": list(models_trained.keys()),
                "details": models_trained
            },
            "metrics": metrics,
            "hpo_results": hpo_results or {},
            "regime_performance": regime_performance or {},
            "feature_info": feature_info or {},
            "training_config": training_config or {},
            "data_sources": {
                "features": "feature_generation_final_feature_selection_step",
                "labels": "feature_generation_labeling_integration_step",
                "regime_probabilities": "regime_ensemble_training"
            },
            "persistence": {
                "format": "pickle",
                "location": "/artifacts",
                "files": [f"{training_type}_{model_name}.pkl" for model_name in models_trained.keys()]
            }
        }

        return json_data


def create_model_training_report(training_type: str,
                                 symbol: str,
                                 exchange: str,
                                 timeframe: str,
                                 direction: str,
                                 models_trained: Dict[str, Any],
                                 metrics: Dict[str, Any],
                                 hpo_results: Optional[Dict[str, Any]] = None,
                                 regime_performance: Optional[Dict[str, Any]] = None,
                                 training_config: Optional[Dict[str, Any]] = None,
                                 feature_info: Optional[Dict[str, Any]] = None,
                                 execution_time: Optional[float] = None,
                                 outcomes_dir: str = "outcomes") -> tuple[str, str]:
    """
    Convenience function to create model training reports.

    Returns:
        Tuple of (markdown_path, json_path)
    """
    generator = ModelTrainingReportGenerator(outcomes_dir)
    return generator.generate_training_report(
        training_type, symbol, exchange, timeframe, direction,
        models_trained, metrics, hpo_results, regime_performance,
        training_config, feature_info, execution_time
    )
