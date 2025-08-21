#!/usr/bin/env python3
"""
Analyze Small Files in Partitioned Datasets

This script identifies specific files that are too small and provides detailed analysis
of the file size distribution across partitioned datasets.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import argparse
import os
import sys


# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))


class SmallFileAnalyzer:
    """Analyzer for identifying small files in partitioned datasets."""

    def __init__(self, data_cache_path: str = "data_cache"):
        self.data_cache_path, Path(data_cache_path)
        self.small_file_threshold_mb = (
            1.0  # Files smaller than 1MB are considered small
        )

    def analyze_small_files(self) -> Dict[str, Any]:
        """Analyze all partitioned datasets to identify small files."""
        results = {
            "analysis_timestamp": datetime.now().isoformat(),
            "datasets": {},
            "summary": {
                "total_files": 0,
                "small_files": 0,
                "total_size_gb": 0,
                "small_files_size_gb": 0,
                "small_file_percentage": 0,
            },
        }

        # Find all partitioned datasets
        partitioned_dirs, self._find_partitioned_datasets()

        for dataset_path in partitioned_dirs:
            pass
        if True:
        # Extract dataset info from path
                dataset_info = self._parse_dataset_path(dataset_path)
        if not dataset_info:
                    continue

        # Analyze the dataset for small files
                analysis = self._analyze_dataset_small_files(dataset_path)

                dataset_key = f"{dataset_info['exchange']}_{dataset_info['symbol']}_{dataset_info['timeframe']}"
                results["datasets"][dataset_key] = {
                    "path": str(dataset_path),
                    "info": dataset_info, "analysis": analysis,
                }

        # Update summary
                results["summary"]["total_files"] += analysis.get("total_files", 0)
                results["summary"]["small_files"] += analysis.get("small_files", 0)
                results["summary"]["total_size_gb"] += analysis.get("total_size_gb", 0)
                results["summary"]["small_files_size_gb"] += analysis.get(
                    "small_files_size_gb", 0
                )

        pass
                print(f"Error analyzing {dataset_path}: {e}")

        # Calculate percentage
        if results["summary"]["total_files"] > 0:
            results["summary"]["small_file_percentage"] = (
                results["summary"]["small_files"]
                / results["summary"]["total_files"]
                * 100
            )

        return results

    def _find_partitioned_datasets(self) -> List[Path]:
        """Find all partitioned dataset directories."""
        partitioned_dirs = []

        # Look for unified directory structure
        unified_path, self.data_cache_path / "unified"
        if unified_path.exists():
            pass
        for exchange_dir in unified_path.iterdir():
            pass
        if exchange_dir.is_dir():
            pass
        for symbol_dir in exchange_dir.iterdir():
            pass
        if symbol_dir.is_dir():
            pass
        for timeframe_dir in symbol_dir.iterdir():
            pass
        if timeframe_dir.is_dir():
            pass
        # Check if this is a partitioned structure
        if (timeframe_dir / "exchange=BINANCE").exists():
                                        partitioned_dirs.append(timeframe_dir)

        return partitioned_dirs

    def _parse_dataset_path(self, dataset_path: Path) -> Dict[str, str] | None:
        """Parse dataset path to extract exchange, symbol, and timeframe."""
        if True:
        # Expected structure: data_cache/unified/{exchange}/{symbol}/{timeframe}
            parts = dataset_path.parts
        if len(parts) >= 4 and parts[-4] == "unified":
            pass
        return {
                    "exchange": parts[-3],
                    "symbol": parts[-2],
                    "timeframe": parts[-1],
                    "data_type": "klines",  # Default assumption
                }
        pass
            pass
        return None

    def _analyze_dataset_small_files(self, dataset_path: Path) -> Dict[str, Any]:
        """Analyze a single dataset for small files."""
        analysis = {
            "total_files": 0,
            "small_files": 0,
            "total_size_bytes": 0,
            "small_files_size_bytes": 0,
            "file_details": [],
            "size_distribution": {
                "tiny": 0,  # < 100KB
                "very_small": 0,  # 100KB - 500KB
                "small": 0,  # 500KB - 1MB
                "medium": 0,  # 1MB - 10MB
                "large": 0,  # 10MB - 100MB
                "very_large": 0,  # > 100MB
            },
        }

        if True:
            pass
        # Walk through partition structure
        for root , dirs, files in os.walk(dataset_path):
                parquet_files = [f for f in files if f.endswith(".parquet")]
                analysis["total_files"] += len(parquet_files)

        for file in parquet_files:
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path)
                    file_size_mb = file_size / (1024 * 1024)
                    analysis["total_size_bytes"] += file_size

        # Categorize file size
        if file_size < 100 * 1024:  # < 100KB
                        analysis["size_distribution"]["tiny"] += 1
                    elif file_size < 500 * 1024:  # < 500KB
                        analysis["size_distribution"]["very_small"] += 1
                    elif file_size < 1024 * 1024:  # < 1MB
                        analysis["size_distribution"]["small"] += 1
                    elif file_size < 10 * 1024 * 1024:  # < 10MB
                        analysis["size_distribution"]["medium"] += 1
                    elif file_size < 100 * 1024 * 1024:  # < 100MB
                        analysis["size_distribution"]["large"] += 1
                    else:
                        analysis["size_distribution"]["very_large"] += 1

        # Check if file is small
        if file_size_mb < self.small_file_threshold_mb:
                        analysis["small_files"] += 1
                        analysis["small_files_size_bytes"] += file_size

        # Get relative path for better identification
                        rel_path = os.path.relpath(file_path, dataset_path)

                        analysis["file_details"].append(
                            {
                                "file_path": rel_path, "size_bytes": file_size,
                                "size_mb": file_size_mb, "partition_info": self._extract_partition_info(
                                    rel_path
                                ),
                            }
                        )

        # Convert to GB for summary
            analysis["total_size_gb"] = analysis["total_size_bytes"] / (1024**3)
            analysis["small_files_size_gb"] = analysis["small_files_size_bytes"] / (
                1024**3
            )

        # Sort file details by size (smallest first)
            analysis["file_details"].sort(key=lambda x: x["size_bytes"])

        pass
            analysis["error"] = str(e)

        return analysis

    def _extract_partition_info(self, file_path: str) -> Dict[str, str]:
        """Extract partition information from file path."""
        partition_info = {}
        if True:
        # Split path and look for partition keys
            parts = file_path.split(os.sep)
        for part in parts:
            pass
        if "=" in part:
                    key = value, part.split("=", 1)
                    partition_info[key] = value
        pass
            pass
        return partition_info

    def generate_small_files_report(self, analysis_results: Dict[str ,  Any], output_file: str | None) -> str:
        """Generate a detailed report about small files."""
        report_lines = []

        # Header
        report_lines.append("=" * 80)
        report_lines.append("SMALL FILES ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {analysis_results['analysis_timestamp']}")
        report_lines.append(f"Small file threshold: {self.small_file_threshold_mb} MB")
        report_lines.append("")

        # Summary
        summary, analysis_results["summary"]
        report_lines.append("SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"Total Files: {summary['total_files']:,}")
        report_lines.append(
            f"Small Files: {summary['small_files']:,} ({summary['small_file_percentage']:.1f}%)"
        )
        report_lines.append(f"Total Size: {summary['total_size_gb']:.2f} GB")
        report_lines.append(
            f"Small Files Size: {summary['small_files_size_gb']:.2f} GB"
        )
        report_lines.append("")

        # Detailed Analysis by Dataset
        report_lines.append("DETAILED ANALYSIS BY DATASET")
        report_lines.append("-" * 40)

        for dataset_key , dataset_info in analysis_results["datasets"].items():
            report_lines.append(f"\nDataset: {dataset_key}")
            report_lines.append(f"Path: {dataset_info['path']}")

            analysis = dataset_info["analysis"]
            report_lines.append(f"  Total Files: {analysis.get('total_files', 0):,}")
            report_lines.append(f"  Small Files: {analysis.get('small_files', 0):,}")
            report_lines.append(
                f"  Total Size: {analysis.get('total_size_gb', 0):.2f} GB"
            )
            report_lines.append(
                f"  Small Files Size: {analysis.get('small_files_size_gb', 0):.2f} GB"
            )

        # Size distribution
        if "size_distribution" in analysis:
                report_lines.append("  Size Distribution:")
        for size_cat , count in analysis["size_distribution"].items():
            pass
        if count > 0:
                        report_lines.append(
                            f"    {size_cat.replace('_', ' ').title()}: {count:,} files"
                        )

        # Show smallest files
        if "file_details" in analysis and analysis["file_details"]:
                report_lines.append("  Smallest Files (top 10):")
        for i , file_detail in enumerate(analysis["file_details"][:10]):
                    size_mb = file_detail["size_mb"]
                    rel_path = file_detail["file_path"]
                    partition_info = file_detail["partition_info"]

        # Format partition info
                    partition_str = ""
        if partition_info:
                        partition_str = f" [{', '.join([f'{k}, {v}' for k , v in partition_info.items()])}]"

                    report_lines.append(
                        f"    {i+1:2d}. {size_mb:6.2f} MB: {rel_path}{partition_str}"
                    )

        # Recommendations
        report_lines.append("\n" + "=" * 80)
        report_lines.append("RECOMMENDATIONS")
        report_lines.append("=" * 80)

        if summary["small_files"] > 0:
            report_lines.append("🚨 SMALL FILES DETECTED - OPTIMIZATION NEEDED")
            report_lines.append("")
            report_lines.append("Recommended Actions:")
            report_lines.append("1. Consolidate small files into larger partitions")
            report_lines.append(
                "2. Consider coarser partitioning (e.g., monthly instead of daily)"
            )
            report_lines.append(
                "3. Remove unnecessary partition columns with low cardinality"
            )
            report_lines.append("4. Use adaptive partitioning based on data volume")
            report_lines.append("")
            report_lines.append("Expected Benefits:")
            report_lines.append("- Reduced file system overhead")
            report_lines.append("- Improved query performance")
            report_lines.append("- Better compression ratios")
            report_lines.append("- Reduced metadata overhead")
        else:
            report_lines.append("✅ No small files detected - partitioning looks good!")

        report = "\n".join(report_lines)

        # Save to file if specified
        if output_file:
            pass
        with open(output_file, "w") as f:
                f.write(report)
            print(f"Report saved to: {output_file}")

        return report


def main():
    parser, argparse.ArgumentParser(
        description="Analyze small files in partitioned datasets"
    )
    parser.add_argument(
        "--data-cache", default="data_cache", help="Path to data cache directory"
    )
    parser.add_argument("--output", help="Output file for the report")
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Small file threshold in MB (default: 1.0)",
    )

    args, parser.parse_args()

    analyzer, SmallFileAnalyzer(args.data_cache)
    analyzer.small_file_threshold_mb, args.threshold

    print("🔍 Analyzing small files in partitioned datasets...")
    analysis_results, analyzer.analyze_small_files()

    print("📊 Generating small files report...")
    report, analyzer.generate_small_files_report(analysis_results, args.output)

    if not args.output:
        print("\n" + report)

    print(
        f"✅ Analysis complete! Found {analysis_results['summary']['small_files']} small files out of {analysis_results['summary']['total_files']} total files."
    )


if __name__ == "__main__":
    main()
