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

from src.utils.advanced_decorators import performance_monitor, PerformanceLevel


class SmallFileAnalyzer:
    """Analyzer for identifying small files in partitioned datasets."""

    def __init__(self, data_cache_path: str = "data_cache"):
        self.data_cache_path = Path(data_cache_path)
        self.small_file_threshold_mb = 1.0  # Files smaller than 1MB are considered small

    @performance_monitor(level=PerformanceLevel.DETAILED)
    def analyze_small_files(self) -> Dict[str, Any]:
        """Analyze all partitioned datasets to identify small files."""
        results: Dict[str, Any] = {
            "analysis_timestamp": datetime.now().isoformat(),
            "datasets": {},
            "summary": {
                "total_files": 0,
                "small_files": 0,
                "total_size_gb": 0.0,
                "small_files_size_gb": 0.0,
                "small_file_percentage": 0.0,
            },
        }

        partitioned_dirs = self._find_partitioned_datasets()

        for dataset_path in partitioned_dirs:
            # Extract dataset info from path
            dataset_info = self._parse_dataset_path(dataset_path)
            if not dataset_info:
                continue

            # Analyze the dataset for small files
            analysis = self._analyze_dataset_small_files(dataset_path)

            dataset_key = f"{dataset_info['exchange']}_{dataset_info['symbol']}_{dataset_info['timeframe']}"
            results["datasets"][dataset_key] = {
                "path": str(dataset_path),
                "info": dataset_info,
                "analysis": analysis,
            }

            # Update summary
            results["summary"]["total_files"] += analysis.get("total_files", 0)
            results["summary"]["small_files"] += analysis.get("small_files", 0)
            results["summary"]["total_size_gb"] += analysis.get("total_size_gb", 0.0)
            results["summary"]["small_files_size_gb"] += analysis.get(
                "small_files_size_gb", 0.0
            )

        # Calculate percentage
        if results["summary"]["total_files"] > 0:
            results["summary"]["small_file_percentage"] = (
                results["summary"]["small_files"]
                / results["summary"]["total_files"]
                * 100.0
            )

        return results

    def _find_partitioned_datasets(self) -> List[Path]:
        """Find all partitioned dataset directories."""
        partitioned_dirs: List[Path] = []

        # Look for unified directory structure
        unified_path = self.data_cache_path / "unified"
        if unified_path.exists():
            for exchange_dir in unified_path.iterdir():
                if not exchange_dir.is_dir():
                    continue
                for symbol_dir in exchange_dir.iterdir():
                    if not symbol_dir.is_dir():
                        continue
                    for timeframe_dir in symbol_dir.iterdir():
                        if not timeframe_dir.is_dir():
                            continue
                        # Check if this is a partitioned structure
                        if (timeframe_dir / "exchange=BINANCE").exists():
                            partitioned_dirs.append(timeframe_dir)

        return partitioned_dirs

    def _parse_dataset_path(self, dataset_path: Path) -> Dict[str, str] | None:
        """Parse dataset path to extract exchange, symbol, and timeframe."""
        # Expected structure: data_cache/unified/{exchange}/{symbol}/{timeframe}
        parts = dataset_path.parts
        if len(parts) >= 4 and parts[-4] == "unified":
            return {
                "exchange": parts[-3],
                "symbol": parts[-2],
                "timeframe": parts[-1],
            }
        return None

    def _analyze_dataset_small_files(self, dataset_path: Path) -> Dict[str, Any]:
        """Analyze the given dataset directory for small files."""
        total_files = 0
        small_files = 0
        total_size_bytes = 0
        small_files_size_bytes = 0

        for root, _dirs, files in os.walk(dataset_path):
            for fname in files:
                fpath = Path(root) / fname
                try:
                    size = fpath.stat().st_size
                except FileNotFoundError:
                    continue
                total_files += 1
                total_size_bytes += size
                if size < self.small_file_threshold_mb * 1024 * 1024:
                    small_files += 1
                    small_files_size_bytes += size

        return {
            "total_files": total_files,
            "small_files": small_files,
            "total_size_gb": round(total_size_bytes / (1024**3), 6),
            "small_files_size_gb": round(small_files_size_bytes / (1024**3), 6),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze small files in datasets")
    parser.add_argument(
        "--data_cache_path",
        default="data_cache",
        help="Path to the data_cache directory",
    )
    args = parser.parse_args()

    analyzer = SmallFileAnalyzer(data_cache_path=args.data_cache_path)
    results = analyzer.analyze_small_files()

    print("=== Summary ===")
    for key, value in results["summary"].items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
