#!/usr/bin/env python3
"""
Partition Optimization Script

This script analyzes and optimizes parquet partition structures for better performance.
It provides recommendations for partition strategy improvements and can perform
partition maintenance operations.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from utils.data_loader import PartitionedDataLoader
from utils.logger import system_logger
import argparse
import sys


# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))


class PartitionOptimizer:
    """Analyzes and optimizes parquet partition structures."""

    def __init__(self, data_cache_path: str = "data_cache"):
        self.data_cache_path = Path(data_cache_path)
        self.loader = PartitionedDataLoader()
        self.logger = system_logger.getChild("PartitionOptimizer")

    def analyze_all_partitions(self) -> Dict[str, Any]:
        """Analyze all partitioned datasets in the data cache."""
        results: Dict[str, Any] = {
            "analysis_timestamp": datetime.now().isoformat(),
            "datasets": {},
            "summary": {
                "total_datasets": 0,
                "total_size_gb": 0.0,
                "total_files": 0,
                "optimization_opportunities": 0,
            },
        }

        # Find all partitioned datasets
        partitioned_dirs = self._find_partitioned_datasets()

        for dataset_path in partitioned_dirs:
            # Extract dataset info from path
            dataset_info = self._parse_dataset_path(dataset_path)
            if not dataset_info:
                continue

            # Analyze the dataset
            analysis = self.loader.optimize_partition_access(
                str(self.data_cache_path),
                dataset_info["exchange"],
                dataset_info["symbol"],
                dataset_info["data_type"],
            )

            dataset_key = f"{dataset_info['exchange']}_{dataset_info['symbol']}_{dataset_info['data_type']}"
            results["datasets"][dataset_key] = {
                "path": str(dataset_path),
                "info": dataset_info,
                "analysis": analysis,
            }

            # Update summary
            results["summary"]["total_datasets"] += 1
            if (
                "partition_analysis" in analysis
                and "total_size_bytes" in analysis["partition_analysis"]
            ):
                results["summary"]["total_size_gb"] += analysis[
                    "partition_analysis"
                ]["total_size_bytes"] / (1024**3)
                results["summary"]["total_files"] += analysis[
                    "partition_analysis"
                ].get("total_files", 0)

            if "recommendations" in analysis:
                results["summary"]["optimization_opportunities"] += len(
                    analysis["recommendations"]
                )

        return results

    def _find_partitioned_datasets(self) -> List[Path]:
        """Find all partitioned dataset directories."""
        partitioned_dirs: List[Path] = []

        # Look for unified directory structure
        unified_path = self.data_cache_path / "unified"
        if unified_path.exists():
            try:
                for exchange_dir in unified_path.iterdir():
                    if exchange_dir.is_dir():
                        for symbol_dir in exchange_dir.iterdir():
                            if symbol_dir.is_dir():
                                for timeframe_dir in symbol_dir.iterdir():
                                    if timeframe_dir.is_dir():
                                        # Check if this is a partitioned structure
                                        if any(
                                            (timeframe_dir / "exchange=BINANCE").exists()
                                            for _ in range(1)
                                        ):
                                            partitioned_dirs.append(timeframe_dir)
            except Exception as e:  # noqa: BLE001
                self.logger.warning(f"Error scanning unified path: {e}")

        return partitioned_dirs

    def _parse_dataset_path(self, dataset_path: Path) -> Dict[str, str] | None:
        """Parse dataset path to extract exchange, symbol, and data type."""
        # Expected structure: data_cache/unified/{exchange}/{symbol}/{timeframe}
        parts = dataset_path.parts
        if len(parts) >= 4 and parts[-4] == "unified":
            return {
                "exchange": parts[-3],
                "symbol": parts[-2],
                "timeframe": parts[-1],
                "data_type": "klines",  # Default assumption
            }
        return None

    def generate_optimization_report(self, analysis_results: Dict[str, Any], output_file: str | None) -> str:
        """Generate a comprehensive optimization report."""
        report_lines: List[str] = []

        # Header
        report_lines.append("=" * 80)
        report_lines.append("PARQUET PARTITION OPTIMIZATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {analysis_results['analysis_timestamp']}")
        report_lines.append("")

        # Summary
        summary = analysis_results["summary"]
        report_lines.append("SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"Total Datasets: {summary['total_datasets']}")
        report_lines.append(f"Total Size: {summary['total_size_gb']:.2f} GB")
        report_lines.append(f"Total Files: {summary['total_files']:,}")
        report_lines.append(
            f"Optimization Opportunities: {summary['optimization_opportunities']}"
        )
        report_lines.append("")

        # Detailed Analysis
        report_lines.append("DETAILED ANALYSIS")
        report_lines.append("-" * 40)

        for dataset_key, dataset_info in analysis_results["datasets"].items():
            report_lines.append(f"\nDataset: {dataset_key}")
            report_lines.append(f"Path: {dataset_info['path']}")

            analysis: Dict[str, Any] = dataset_info["analysis"]
            if "partition_analysis" in analysis:
                pa = analysis["partition_analysis"]
                report_lines.append(f"  Total Files: {pa.get('total_files', 0):,}")
                report_lines.append(
                    f"  Total Size: {pa.get('total_size_bytes', 0) / (1024**3):.2f} GB"
                )
                report_lines.append(
                    f"  Average File Size: {pa.get('avg_file_size', 0) / (1024**2):.1f} MB"
                )

                if "partition_counts" in pa:
                    report_lines.append("  Partition Distribution:")
                    for partition, values in pa["partition_counts"].items():
                        unique_count = len(values) if isinstance(values, (list, set, tuple)) else int(values)
                        report_lines.append(
                            f"    {partition}: {unique_count} unique values"
                        )

            if "recommendations" in analysis and analysis["recommendations"]:
                report_lines.append("  Recommendations:")
                for rec in analysis["recommendations"]:
                    report_lines.append(f"    ⚠️  {rec['suggestion']}")
            else:
                report_lines.append("  ✅ No optimization recommendations")

        # Optimization Actions
        report_lines.append("\n" + "=" * 80)
        report_lines.append("RECOMMENDED ACTIONS")
        report_lines.append("=" * 80)

        all_recommendations: List[Dict[str, Any]] = []
        for dataset_info in analysis_results["datasets"].values():
            if "recommendations" in dataset_info["analysis"]:
                all_recommendations.extend(dataset_info["analysis"]["recommendations"])

        if all_recommendations:
            # Group recommendations by type
            rec_by_type: Dict[str, List[Dict[str, Any]]] = {}
            for rec in all_recommendations:
                rec_type = rec.get("type", "general")
                if rec_type not in rec_by_type:
                    rec_by_type[rec_type] = []
                rec_by_type[rec_type].append(rec)

            for rec_type, recs in rec_by_type.items():
                report_lines.append(f"\n{rec_type.upper()} ISSUES ({len(recs)} found):")
                for rec in recs:
                    report_lines.append(f"  • {rec['suggestion']}")
        else:
            report_lines.append("✅ No optimization actions required!")

        report = "\n".join(report_lines)

        # Save to file if specified
        if output_file:
            try:
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(report)
                self.logger.info(f"Report saved to: {output_file}")
            except Exception as e:  # noqa: BLE001
                self.logger.error(f"Failed to save report to {output_file}: {e}")

        return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze and optimize parquet partition structures"
    )
    parser.add_argument(
        "--data-cache", default="data_cache", help="Path to data cache directory"
    )
    parser.add_argument("--output", help="Output file for the report")
    parser.add_argument(
        "--action",
        choices=["analyze", "optimize"],
        default="analyze",
        help="Action to perform",
    )

    args = parser.parse_args()

    optimizer = PartitionOptimizer(args.data_cache)

    if args.action == "analyze":
        print("🔍 Analyzing partition structures...")
        analysis_results = optimizer.analyze_all_partitions()

        print("📊 Generating optimization report...")
        report = optimizer.generate_optimization_report(analysis_results, args.output)

        if not args.output:
            print("\n" + report)

        print(
            f"✅ Analysis complete! Found {analysis_results['summary']['optimization_opportunities']} optimization opportunities."
        )

    elif args.action == "optimize":
        print("🚧 Partition optimization not yet implemented")
        print("Use --action analyze to see recommendations first")


if __name__ == "__main__":
    main()
