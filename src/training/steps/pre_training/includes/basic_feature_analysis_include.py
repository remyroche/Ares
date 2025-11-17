from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


class BasicFeatureAnalysisInclude:
    def __init__(self, output_dir: str = "outcomes") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def run(self, features: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(features, pd.DataFrame) or features.empty:
            return {}
        analysis = self._build_analysis(features)
        report_path = self._write_report(analysis, config)
        return {
            "metrics": analysis,
            "report_path": str(report_path)
        }

    def _build_analysis(self, features: pd.DataFrame) -> Dict[str, Any]:
        numeric_features = features.select_dtypes(include=[np.number])
        missing_total = int(features.isnull().sum().sum())
        total_cells = max(len(features) * max(len(features.columns), 1), 1)
        missing_ratio = missing_total / total_cells
        constant_features = self._constant_columns(numeric_features)
        high_missing = self._high_missing(features)
        variance = numeric_features.var()
        top_variance = variance.sort_values(ascending=False).head(10).to_dict()
        low_variance = variance.sort_values().head(10).to_dict()
        correlations = self._correlation_pairs(numeric_features)
        return {
            "total_features": len(features.columns),
            "row_count": len(features),
            "missing_values": missing_total,
            "missing_ratio": missing_ratio,
            "numeric_features": len(numeric_features.columns),
            "non_numeric_features": len(features.columns) - len(numeric_features.columns),
            "constant_features": constant_features,
            "high_missing_features": high_missing,
            "top_variance_features": top_variance,
            "low_variance_features": low_variance,
            "variance_mean": float(variance.mean()) if not variance.empty else 0.0,
            "variance_median": float(variance.median()) if not variance.empty else 0.0,
            "variance_std": float(variance.std()) if not variance.empty else 0.0,
            "high_correlation_pairs": correlations[:10],
            "median_correlation": float(np.nanmedian([c[2] for c in correlations])) if correlations else 0.0
        }

    def _constant_columns(self, df: pd.DataFrame) -> List[str]:
        result: List[str] = []
        if df.empty:
            return result
        for column in df.columns:
            series = df[column].dropna()
            if series.nunique() <= 1:
                result.append(column)
        return result

    def _high_missing(self, df: pd.DataFrame) -> Dict[str, float]:
        missing = df.isnull().mean()
        filtered = missing[missing >= 0.4].sort_values(ascending=False)
        return filtered.to_dict()

    def _correlation_pairs(self, df: pd.DataFrame) -> List[Tuple[str, str, float]]:
        pairs: List[Tuple[str, str, float]] = []
        if df.shape[1] < 2:
            return pairs
        corr = df.corr().abs()
        columns = corr.columns
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                value = corr.iloc[i, j]
                if np.isfinite(value):
                    pairs.append((columns[i], columns[j], float(value)))
        pairs.sort(key=lambda item: item[2], reverse=True)
        return pairs

    def _write_report(self, analysis: Dict[str, Any], config: Dict[str, Any]) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"basic_feature_analysis_{timestamp}.md"
        with path.open("w") as handle:
            handle.write("# Basic Feature Analysis\n\n")
            handle.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            handle.write(f"**Symbol:** {config.get('symbol', 'N/A')}\n")
            handle.write(f"**Exchange:** {config.get('exchange', 'N/A')}\n")
            handle.write(f"**Timeframe:** {config.get('timeframe', 'N/A')}\n")
            handle.write(f"**Execution Mode:** {config.get('execution_mode', 'N/A')}\n\n")
            handle.write("## Summary\n\n")
            handle.write(f"- Total Features: {analysis['total_features']}\n")
            handle.write(f"- Numeric Features: {analysis['numeric_features']}\n")
            handle.write(f"- Non-numeric Features: {analysis['non_numeric_features']}\n")
            handle.write(f"- Rows: {analysis['row_count']}\n")
            handle.write(f"- Missing Values: {analysis['missing_values']} ({analysis['missing_ratio']:.2%})\n")
            handle.write(f"- Mean Variance: {analysis['variance_mean']:.6f}\n")
            handle.write(f"- Median Variance: {analysis['variance_median']:.6f}\n")
            handle.write(f"- Variance Std: {analysis['variance_std']:.6f}\n")
            handle.write(f"- Constant Features: {len(analysis['constant_features'])}\n")
            handle.write(f"- High Missing Columns: {len(analysis['high_missing_features'])}\n")
            handle.write(f"- Median Correlation: {analysis['median_correlation']:.3f}\n\n")
            handle.write("## Constant Features\n\n")
            if analysis['constant_features']:
                for name in analysis['constant_features'][:20]:
                    handle.write(f"- {name}\n")
                if len(analysis['constant_features']) > 20:
                    handle.write(f"- ... and {len(analysis['constant_features']) - 20} more\n")
            else:
                handle.write("- None detected\n")
            handle.write("\n## High Missing Features (>= 40%)\n\n")
            if analysis['high_missing_features']:
                for name, ratio in analysis['high_missing_features'].items():
                    handle.write(f"- {name}: {ratio:.2%}\n")
            else:
                handle.write("- None detected\n")
            handle.write("\n## Top Variance Features\n\n")
            if analysis['top_variance_features']:
                for name, value in analysis['top_variance_features'].items():
                    handle.write(f"- {name}: {value:.6f}\n")
            else:
                handle.write("- Not available\n")
            handle.write("\n## Lowest Variance Features\n\n")
            if analysis['low_variance_features']:
                for name, value in analysis['low_variance_features'].items():
                    handle.write(f"- {name}: {value:.6f}\n")
            else:
                handle.write("- Not available\n")
            handle.write("\n## High Correlation Pairs (>= 0.90)\n\n")
            if analysis['high_correlation_pairs']:
                for left, right, score in analysis['high_correlation_pairs']:
                    handle.write(f"- {left} <> {right}: {score:.3f}\n")
            else:
                handle.write("- None detected\n")
        return path
