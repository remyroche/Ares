"""High-level regime analysis orchestration."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

from src.utils.tprint import tprint, tprint_success

try:  # pragma: no cover - fallback retained for runtime parity
    from src.utils.logging_utils import get_logger, log_warning
except ImportError:  # pragma: no cover - ensures CLI still works without dependency
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)

    def log_warning(message: str) -> None:
        logging.getLogger("RegimeAnalyzer").warning(message)

from .data_access import load_regime_datasets
from .metrics import calculate_regime_distribution, calculate_clustering_metrics
from .reporting import print_detailed_metrics, print_analysis_summary


class RegimeAnalysisService:
    """Coordinates loading, computation, and reporting of regime metrics."""

    def __init__(self, data_cache_path: Path | str = "data_cache"):
        self.data_cache_path = Path(data_cache_path)
        if not self.data_cache_path.exists():
            raise FileNotFoundError(f"Data cache directory not found: {self.data_cache_path}")
        self.logger = get_logger("RegimeAnalyzer")
        tprint("🔍 Regime Analysis service initialized", "INFO")

    def analyze(self, symbol: str = "ETHUSDT") -> Dict[str, Any]:
        """Execute the full regime analysis workflow for a symbol."""
        tprint(f"🚀 Starting comprehensive regime analysis for {symbol}", "INFO")
        nas_features, nas_labels, tas_features, tas_labels = self._load_datasets(symbol)

        self._print_initial_overview(nas_labels, tas_labels)

        nas_distribution = calculate_regime_distribution(nas_labels, "NAS")
        tas_distribution = calculate_regime_distribution(tas_labels, "TAS")

        nas_metrics = calculate_clustering_metrics(nas_features, nas_labels, "NAS")
        tas_metrics = calculate_clustering_metrics(tas_features, tas_labels, "TAS")

        print_detailed_metrics(nas_distribution, nas_metrics, "NAS")
        print_detailed_metrics(tas_distribution, tas_metrics, "TAS")

        analysis = self._compile_analysis(
            symbol,
            nas_distribution,
            tas_distribution,
            nas_metrics,
            tas_metrics,
            nas_labels,
            tas_labels,
        )

        output_path = self._save_analysis(analysis, symbol)
        tprint_success(f"Regime analysis completed and saved to {output_path}")

        print_analysis_summary(analysis)
        return analysis

    def _load_datasets(self, symbol: str) -> Tuple[Any, ...]:
        try:
            return load_regime_datasets(self.data_cache_path, symbol)
        except Exception as exc:  # pragma: no cover - error surface for CLI
            log_warning(f"Failed to load regime datasets: {exc}")
            raise

    def _print_initial_overview(self, nas_labels, tas_labels) -> None:
        tprint("\n" + "=" * 80, "INFO")
        tprint("📊 REGIME ANALYSIS - INITIAL OVERVIEW", "INFO")
        tprint("=" * 80, "INFO")
        tprint(
            f"🔬 NAS regimes: {len(set(nas_labels))} {sorted(set(int(label) for label in nas_labels))}",
            "INFO",
        )
        tprint(
            f"🎯 TAS regimes: {len(set(tas_labels))} {sorted(set(int(label) for label in tas_labels))}",
            "INFO",
        )
        tprint("=" * 80, "INFO")

    def _compile_analysis(
        self,
        symbol: str,
        nas_distribution: Dict[str, Any],
        tas_distribution: Dict[str, Any],
        nas_metrics: Dict[str, Any],
        tas_metrics: Dict[str, Any],
        nas_labels,
        tas_labels,
    ) -> Dict[str, Any]:
        analysis_timestamp = datetime.now().isoformat()
        return {
            "symbol": symbol,
            "analysis_timestamp": analysis_timestamp,
            "nas_analysis": {
                "distribution": nas_distribution,
                "clustering_metrics": nas_metrics,
            },
            "tas_analysis": {
                "distribution": tas_distribution,
                "clustering_metrics": tas_metrics,
            },
            "summary": {
                "nas_regimes": len(set(nas_labels)),
                "tas_regimes": len(set(tas_labels)),
                "nas_samples": len(nas_labels),
                "tas_samples": len(tas_labels),
            },
        }

    def _save_analysis(self, analysis: Dict[str, Any], symbol: str) -> Path:
        output_dir = Path("regime_analysis_results")
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{symbol}_regime_analysis_{timestamp}.json"
        with output_path.open("w") as handle:
            json.dump(analysis, handle, indent=2)
        return output_path
