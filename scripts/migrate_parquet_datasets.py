#!/usr/bin/env python3
"""
Enhanced Parquet Dataset Management Script

This script provides comprehensive parquet dataset management including:
- Migration from flat to partitioned datasets
- Partition analysis and optimization
- Performance recommendations
- Dataset maintenance

Usage:
  # Migrate datasets
  python scripts/migrate_parquet_datasets.py migrate \
    --exchange BINANCE --symbol ETHUSDT --timeframe 1m \
    [--src-base data/training/parquet] [--dst-base data_cache/parquet]

  # Analyze partitions
  python scripts/migrate_parquet_datasets.py analyze \
    --data-cache data_cache [--output analysis_report.txt]

  # Optimize partitions
  python scripts/migrate_parquet_datasets.py optimize \
    --data-cache data_cache [--dry-run]

Notes:
  - Static columns exchange/symbol/timeframe will be added if missing.
  - Existing partitioned data will be appended/overwritten per dataset manager behavior.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils.logger import system_logger  # noqa: E402
from src.utils.common_operations import (  # noqa: E402
    get_current_datetime, format_datetime, ensure_directory,
    safe_read_parquet, safe_to_parquet, list_parquet_files,
    format_bytes
)
from src.training.enhanced_training_manager_optimized import (  # noqa: E402
    ParquetDatasetManager,
)


def _handle_errors(default: Optional[Any] = None):
    """Decorator to log exceptions and return a default value."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            logger=system_logger.getChild(func.__name__)
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error("Error in %s: %s", func.__name__, exc, exc_info=True)
                return default

        return wrapper

    return decorator


class EnhancedParquetManager:
    """Enhanced parquet dataset manager with analysis and optimization capabilities."""

    def __init__(self, data_cache_path: str="data_cache") -> None:
        self.data_cache_path: Path=Path(data_cache_path)
        self.logger=system_logger.getChild("EnhancedParquetManager")

    @_handle_errors(default=None)
    def migrate_dir(
        self,
        src_dir: Path,
        dst_base_dir: Path,
        schema_name: str,
        exchange: str,
        symbol: str,
        timeframe: str,
    ) -> None:
        """Migrate flat parquet directory to partitioned dataset."""
        pdm=ParquetDatasetManager(logger=self.logger)

        static_columns: Dict[str, str] = {
            "exchange": exchange,
            "symbol": symbol,
            "timeframe": timeframe,
        }

        self.logger.info(
            "Migrating %s -> %s (schema=%s)", src_dir, dst_base_dir, schema_name
        )
        compression=(
            "zstd" if schema_name in {"klines", "aggtrades", "futures"} else "snappy"
        )
        pdm.migrate_flat_parquet_dir_to_partitioned(
            src_dir=str(src_dir),
            dst_base_dir=str(dst_base_dir),
            schema_name=schema_name,
            static_columns=static_columns,
            compression=compression,
        )

    @_handle_errors(default={})
    def analyze_partitions(self) -> Dict[str, Any]:
        """Analyze all partitioned datasets in the data cache."""
        results: Dict[str, Any] = {
            "analysis_timestamp": format_datetime(get_current_datetime(), "%Y-%m-%dT%H:%M:%S"),
            "datasets": {},
            "summary": {
                "total_datasets": 0,
                "total_size_gb": 0.0,
                "total_files": 0,
                "optimization_opportunities": 0,
            },
        }

        # Find all partitioned datasets
        partitioned_dirs=self._find_partitioned_datasets()

        for dataset_path in partitioned_dirs:
            try:
                # Extract dataset info from path
                dataset_info=self._parse_dataset_path(dataset_path)
                if not dataset_info:
                    continue

                # Analyze the dataset
                analysis=self._analyze_dataset(dataset_path)

                dataset_key=(
                    f"{dataset_info['exchange']}_{dataset_info['symbol']}_{dataset_info['timeframe']}"
                )
                results["datasets"][dataset_key] = {
                    "path": str(dataset_path),
                    "info": dataset_info,
                    "analysis": analysis,
                }

                # Update summary
                results["summary"]["total_datasets"] += 1
                if "total_size_bytes" in analysis:
                    results["summary"]["total_size_gb"] += analysis["total_size_bytes"] / (
                        1024 ** 3
                    )
                    results["summary"]["total_files"] += analysis.get("total_files", 0)

                if "recommendations" in analysis:
                    results["summary"]["optimization_opportunities"] += len(
                        analysis["recommendations"]
                    )
            except Exception as e:  # pragma: no cover - continue on errors
                self.logger.error("Error analyzing %s: %s", dataset_path, e)

        return results

    @_handle_errors(default=[])
    def _find_partitioned_datasets(self) -> List[Path]:
        """Find all partitioned dataset directories."""
        partitioned_dirs: List[Path] = []

        # Look for unified directory structure
        unified_path=self.data_cache_path / "unified"
        if unified_path.exists():
            for exchange_dir in unified_path.iterdir():
                if exchange_dir.is_dir():
                    for symbol_dir in exchange_dir.iterdir():
                        if symbol_dir.is_dir():
                            for timeframe_dir in symbol_dir.iterdir():
                                if timeframe_dir.is_dir():
                                    # Check if this is a partitioned structure
                                    if any(p.name.startswith("exchange=") for p in timeframe_dir.iterdir()):
                                        partitioned_dirs.append(timeframe_dir)

        # Also look for parquet directory structure
        parquet_path=self.data_cache_path / "parquet"
        if parquet_path.exists():
            for subdir in parquet_path.iterdir():
                if subdir.is_dir() and any(subdir.rglob("*.parquet")):
                    partitioned_dirs.append(subdir)

        return partitioned_dirs

    @_handle_errors(default=None)
    def _parse_dataset_path(self, dataset_path: Path) -> Optional[Dict[str, str]]:
        """Parse dataset path to extract exchange, symbol, and timeframe.

        Attempts to infer from either unified path or partition key=value segments.
        """
        parts = dataset_path.parts

        # Unified structure: data_cache/unified/{exchange}/{symbol}/{timeframe}
        try:
            if len(parts) >= 4 and parts[-4] == "unified":
                return {
                    "exchange": parts[-3],
                    "symbol": parts[-2],
                    "timeframe": parts[-1],
                    "data_type": "klines",
                }
        except Exception:  # pragma: no cover
            pass

        # Fallback: parse key=value segments from path
        kv: Dict[str, str] = {}
        for p in parts:
            if "=" in p:
                key, value=p.split("=", 1)
                kv[key] = value
        if kv:
            return {
                "exchange": kv.get("exchange", "UNKNOWN"),
                "symbol": kv.get("symbol", "UNKNOWN"),
                "timeframe": kv.get("timeframe", "UNKNOWN"),
                "data_type": kv.get("data_type", dataset_path.name),
            }

        # Parquet directory structure (best-effort defaults)
        if len(parts) >= 2 and parts[-2] == "parquet":
            return {
                "exchange": "BINANCE",
                "symbol": "ETHUSDT",
                "timeframe": "1m",
                "data_type": parts[-1],
            }

        return None

    @_handle_errors(default={})
    def _analyze_dataset(self, dataset_path: Path) -> Dict[str, Any]:
        """Analyze a single dataset."""
        analysis: Dict[str, Any] = {
            "total_files": 0,
            "total_size_bytes": 0,
            "partition_counts": {},
            "file_sizes": [],
            "recommendations": [],
        }

        # Walk through partition structure
        for root, _dirs, files in os.walk(dataset_path):
            parquet_files=[f for f in files if f.endswith(".parquet")]
            analysis["total_files"] += len(parquet_files)

            for file in parquet_files:
                file_path=os.path.join(root, file)
                file_size=os.path.getsize(file_path)
                analysis["total_size_bytes"] += file_size
                analysis["file_sizes"].append(file_size)

            # Extract partition information from path
            rel_path=os.path.relpath(root, dataset_path)
            if "=" in rel_path:
                partition_parts=rel_path.split(os.sep)
                for part in partition_parts:
                    if "=" in part:
                        key, value=part.split("=", 1)
                        if key not in analysis["partition_counts"]:
                            analysis["partition_counts"][key] = set()  # type: ignore[assignment]
                        analysis["partition_counts"][key].add(value)  # type: ignore[index]

        # Convert sets to lists for JSON serialization
        for key in list(analysis["partition_counts"].keys()):
            values_set=analysis["partition_counts"][key]
            if isinstance(values_set, set):
                analysis["partition_counts"][key] = list(values_set)

        # Calculate additional statistics
        if analysis["file_sizes"]:
            sizes=analysis["file_sizes"]
            analysis["avg_file_size"] = sum(sizes) / len(sizes)
            analysis["min_file_size"] = min(sizes)
            analysis["max_file_size"] = max(sizes)

        # Generate recommendations
        if analysis.get("avg_file_size", 0) > 100_000_000:  # 100MB
            analysis["recommendations"].append(
                {
                    "type": "large_files",
                    "suggestion": (
                        f"Consider finer partitioning to reduce file sizes (avg: {analysis['avg_file_size'] / 1_000_000:.1f}MB)"
                    ),
                }
            )
        elif analysis.get("avg_file_size", 0) > 0 and analysis["avg_file_size"] < 1_000_000:  # 1MB
            analysis["recommendations"].append(
                {
                    "type": "small_files",
                    "suggestion": (
                        f"Consider coarser partitioning to increase file sizes (avg: {analysis['avg_file_size'] / 1_000_000:.1f}MB)"
                    ),
                }
            )

        # Check partition distribution
        for partition_col, values in analysis["partition_counts"].items():
            if isinstance(values, list):
                if len(values) > 100:
                    analysis["recommendations"].append(
                        {
                            "type": "high_cardinality",
                            "suggestion": (
                                f"Consider coarser partitioning for {partition_col} ({len(values)} unique values)"
                            ),
                        }
                    )
                elif len(values) < 5:
                    analysis["recommendations"].append(
                        {
                            "type": "low_cardinality",
                            "suggestion": (
                                f"Consider removing {partition_col} partitioning ({len(values)} unique values)"
                            ),
                        }
                    )

        return analysis

    @_handle_errors(default="")
    def generate_analysis_report(
        self, analysis_results: Dict[str, Any], output_file: Optional[str]
    ) -> str:
        """Generate a comprehensive analysis report."""
        report_lines: List[str] = []

        # Header
        report_lines.append("=" * 80)
        report_lines.append("ENHANCED PARQUET PARTITION ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {analysis_results['analysis_timestamp']}")
        report_lines.append("")

        # Summary
        summary=analysis_results["summary"]
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

            analysis=dataset_info["analysis"]
            report_lines.append(
                f"  Total Files: {analysis.get('total_files', 0):,}"
            )
            report_lines.append(
                f"  Total Size: {analysis.get('total_size_bytes', 0) / (1024**3):.2f} GB"
            )
            report_lines.append(
                f"  Average File Size: {analysis.get('avg_file_size', 0) / (1024**2):.1f} MB"
            )

            if "partition_counts" in analysis:
                report_lines.append("  Partition Distribution:")
                for partition, values in analysis["partition_counts"].items():
                    if isinstance(values, list):
                        report_lines.append(
                            f"    {partition}: {len(values)} unique values"
                        )

            if analysis.get("recommendations"):
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
                rec_type=rec.get("type", "misc")
                if rec_type not in rec_by_type:
                    rec_by_type[rec_type] = []
                rec_by_type[rec_type].append(rec)

            for rec_type, recs in rec_by_type.items():
                report_lines.append(f"\n{rec_type.upper()} ISSUES ({len(recs)} found):")
                for rec in recs:
                    report_lines.append(f"  • {rec['suggestion']}")
        else:
            report_lines.append("✅ No optimization actions required!")

        report="\n".join(report_lines)

        # Save to file if specified
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(report)
            self.logger.info("Report saved to: %s", output_file)

        return report

    @_handle_errors(default={"status": "not_implemented"})
    def optimize_partitions(self, dry_run: bool=True) -> Dict[str, Any]:
        """Optimize partition structures (placeholder for future implementation)."""
        self.logger.info("Partition optimization not yet implemented")
        return {
            "status": "not_implemented",
            "message": "Partition optimization will be implemented in future versions",
            "dry_run": dry_run,
        }


def migrate_datasets(args: argparse.Namespace) -> int:
    """Migrate flat parquet directories to partitioned datasets."""
    manager=EnhancedParquetManager(args.dst_base)

    src_base=Path(args.src_base)
    dst_base=Path(args.dst_base)

    if not src_base.exists():
        system_logger.warning(f"Source base does not exist: {src_base}")
        return 0

    # Map subdirectory names to schema names
    dataset_map: Dict[str, str] = {
        "klines": "klines",
        "aggtrades": "aggtrades",
        "futures": "futures",
        "features": "split",
        "labeled": "split",
        "regime_data": "split",
        "vectorized_features": "split",
    }

    migrated_any=False
    for subdir_name, schema_name in dataset_map.items():
        src_dir=src_base / subdir_name
        if not src_dir.exists() or not any(src_dir.rglob("*.parquet")):
            continue
        dst_dir=dst_base / subdir_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        manager.migrate_dir(
            src_dir=src_dir,
            dst_base_dir=dst_dir,
            schema_name=schema_name,
            exchange=args.exchange,
            symbol=args.symbol,
            timeframe=args.timeframe,
        )
        migrated_any=True

    if not migrated_any:
        system_logger.info(f"No flat Parquet directories found under {src_base}")
    else:
        system_logger.info(
            f"Migration complete. Partitioned datasets available under {dst_base}",
        )
    return 0


def analyze_partitions(args: argparse.Namespace) -> int:
    """Analyze partition structures and generate report."""
    manager=EnhancedParquetManager(args.data_cache)

    print("🔍 Analyzing partition structures...")
    analysis_results=manager.analyze_partitions()

    print("📊 Generating analysis report...")
    report=manager.generate_analysis_report(analysis_results, args.output)

    if not args.output:
        print("\n" + report)

    print(
        f"✅ Analysis complete! Found {analysis_results['summary']['optimization_opportunities']} optimization opportunities."
    )
    return 0


def optimize_partitions_cli(args: argparse.Namespace) -> int:
    """Optimize partition structures."""
    manager=EnhancedParquetManager(args.data_cache)

    print("🚧 Partition optimization not yet implemented")
    print("Use 'analyze' action to see recommendations first")
    _=manager.optimize_partitions(dry_run=args.dry_run)
    return 0


def main() -> int:
    parser=argparse.ArgumentParser(
        description="Enhanced parquet dataset management with analysis and optimization",
    )
    subparsers=parser.add_subparsers(dest="action", help="Available actions")

    # Migration subcommand
    migrate_parser=subparsers.add_parser(
        "migrate", help="Migrate flat parquet to partitioned"
    )
    migrate_parser.add_argument(
        "--exchange", default=os.environ.get("AresExchange", "BINANCE")
    )
    migrate_parser.add_argument(
        "--symbol", default=os.environ.get("AresSymbol", "ETHUSDT")
    )
    migrate_parser.add_argument(
        "--timeframe", default=os.environ.get("AresTimeframe", "1m")
    )
    migrate_parser.add_argument("--src-base", default="data/training/parquet")
    migrate_parser.add_argument("--dst-base", default="data_cache/parquet")

    # Analysis subcommand
    analyze_parser=subparsers.add_parser(
        "analyze", help="Analyze partition structures"
    )
    analyze_parser.add_argument(
        "--data-cache", default="data_cache", help="Path to data cache directory"
    )
    analyze_parser.add_argument("--output", help="Output file for the report")

    # Optimization subcommand
    optimize_parser=subparsers.add_parser(
        "optimize", help="Optimize partition structures"
    )
    optimize_parser.add_argument(
        "--data-cache", default="data_cache", help="Path to data cache directory"
    )
    optimize_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )

    args=parser.parse_args()

    if args.action== "migrate":
        return migrate_datasets(args)
    elif args.action== "analyze":
        return analyze_partitions(args)
    elif args.action== "optimize":
        return optimize_partitions_cli(args)
    else:
        parser.print_help()
        return 1


if __name__== "__main__":
    raise SystemExit(main())
