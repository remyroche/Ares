import csv
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

class LayeredPipelineReporter:
    """
    Standardized CSV reporter for the Layered Training Pipeline.
    Logs metrics for each layer/model to a central CSV file.
    """

    def __init__(self, output_dir: str = "outcomes/layered_pipeline"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.output_dir / f"layered_metrics_{self.timestamp}.csv"

        self.fieldnames = [
            "timestamp", "layer", "model_name", "metric_type",
            "ic", "sharpe_ratio", "sortino_ratio", "calmar_ratio", "omega_ratio", "tail_ratio",
            "win_rate", "profit_factor", "avg_trade_expectancy",
            "auc_roc", "information_ratio", "directional_accuracy",
            "ece", "brier_score", "mce", "prob_dist_skew", "log_loss", "prediction_std",
            "feature_importance_shift", "coeff_stability",
            "turnover_daily", "regime_hv_perf", "regime_mv_perf", "regime_lv_perf"
        ]

        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log_metrics(self, layer: str, model_name: str, metric_type: str, metrics: Dict[str, Any]):
        """
        Log a row of metrics.
        """
        row = {
            "timestamp": datetime.now().isoformat(),
            "layer": layer,
            "model_name": model_name,
            "metric_type": metric_type
        }

        # Flatten known metrics
        for field in self.fieldnames:
            if field in metrics:
                row[field] = metrics[field]
            elif field not in row:
                row[field] = ""

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)

    def get_report_path(self) -> str:
        return str(self.csv_path)
