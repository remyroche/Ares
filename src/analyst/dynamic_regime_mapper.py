# src/analyst/dynamic_regime_mapper.py

from src.utils.logger import system_logger
from typing import Any
import json
import os

from src.utils.error_handler import handle_errors
import pandas as pd


class DynamicRegimeMapper:
    """
    Dynamically maps HMM composite cluster IDs to regime names based on Step 1.7 results.
    Reads actual archetype descriptions and creates regime mappings automatically.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("DynamicRegimeMapper")

        # Cache for regime mappings
        self.regime_mappings: dict[
            str, dict[int, str],
        ] = {}  # timeframe -> cluster_id -> regime_name
        self.archetype_descriptions: dict[
            str, dict[int, str],
        ] = {}  # timeframe -> cluster_id -> description
        self.cluster_centroids: dict[
            str, dict[int, list[float]],
        ] = {}  # timeframe -> cluster_id -> centroid

        # Configuration
        self.data_dir = config.get("data_dir", "data/training")
        self.auto_discover_regimes = config.get("auto_discover_regimes", True)
        self.regime_naming_strategy = config.get(
            "regime_naming_strategy",
            "archetype_based",
        )

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="dynamic regime mapper initialization",
    )
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="regime discovery from step01_7",
    )
    async def _discover_regimes_from_step1_7(self) -> bool:
        """Discover regimes by reading Step 1.7 HMM clustering results."""
        try:
            self.logger.info("Discovering regimes from Step 1.7 results...")

            # Look for meta files that contain archetype descriptions
            meta_files = []
            for filename in os.listdir(self.data_dir):
                if filename.endswith("_hmm_composite_meta_1m.json"):
                    meta_files.append(filename)

            if not meta_files:
                self.logger.warning("No Step 1.7 meta files found for regime discovery")
                return False

            # Process each meta file to extract regime information
            for meta_file in meta_files:
                await self._process_meta_file(meta_file)

            self.logger.info(
                f"Discovered regimes for {len(self.regime_mappings)} timeframes",
            )
            return True

        except Exception as e:
            self.logger.exception(f"Error discovering regimes from Step 1.7: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="meta file processing",
    )
    async def _process_meta_file(self, meta_file: str) -> bool:
        """Process a Step 1.7 meta file to extract regime information."""
        try:
            meta_path = os.path.join(self.data_dir, meta_file)

            with open(meta_path) as f:
                meta_data = json.load(f)

            # Extract timeframe from filename
            # Format: {exchange}_{symbol}_hmm_composite_meta_{timeframe}.json
            parts = meta_file.split("_")
            if len(parts) >= 4:
                timeframe = parts[-1].replace(".json", "")
            else:
                timeframe = "1m"  # Default

            # Extract archetype descriptions
            archetype_descriptions = meta_data.get("archetype_descriptions", {})
            cluster_centroids = meta_data.get("cluster_centroids", {})

            # Convert string keys to integers
            archetype_descriptions_int = {}
            cluster_centroids_int = {}

            for cluster_id_str, description in archetype_descriptions.items():
                try:
                    cluster_id = int(cluster_id_str)
                    archetype_descriptions_int[cluster_id] = description
                except ValueError:
                    continue

            for cluster_id_str, centroid in cluster_centroids.items():
                try:
                    cluster_id = int(cluster_id_str)
                    cluster_centroids_int[cluster_id] = centroid
                except ValueError:
                    continue

            # Generate regime names based on archetype descriptions
            regime_mapping = self._generate_regime_mapping_from_archetypes(
                archetype_descriptions_int, cluster_centroids_int,
            )

            # Store the mappings
            self.regime_mappings[timeframe] = regime_mapping
            self.archetype_descriptions[timeframe] = archetype_descriptions_int
            self.cluster_centroids[timeframe] = cluster_centroids_int

            self.logger.info(
                f"Processed {timeframe}: {len(regime_mapping)} regimes discovered",
            )

            # Log the discovered regimes
            for cluster_id, regime_name in regime_mapping.items():
                description = archetype_descriptions_int.get(
                    cluster_id, "No description",
                )
                self.logger.info(
                    f"  Cluster {cluster_id} -> {regime_name}: {description}",
                )

            return True

        except Exception as e:
            self.logger.exception(f"Error processing meta file {meta_file}: {e}")
            return False

    def _generate_regime_mapping_from_archetypes(
        self, archetype_descriptions: dict[int, str],
        cluster_centroids: dict[int, list[float]],
    ) -> dict[int, str]:
        """Generate regime names from archetype descriptions."""
        regime_mapping = {}

        for cluster_id, description in archetype_descriptions.items():
            regime_name = self._classify_archetype_to_regime(
                cluster_id, description,
                cluster_centroids.get(cluster_id, []),
            )
            regime_mapping[cluster_id] = regime_name

        return regime_mapping

    def _classify_archetype_to_regime(
        self, cluster_id: int,
        description: str, centroid: list[float],
    ) -> str:
        """Classify an archetype description into a regime name."""

        # Handle rare/unclassifiable market conditions (-1)
        if cluster_id == -1:
            return "RARE_MARKET_CONDITIONS"

        description_lower = description.lower()

        # Core trend classification
        if "strong upward" in description_lower or "bullish" in description_lower:
            if "high volatility" in description_lower:
                return "HIGH_VOLATILITY_BULL"
            if "low volatility" in description_lower:
                return "STRONG_BULL_TREND"
            return "MODERATE_BULL_TREND"

        if "strong downward" in description_lower or "bearish" in description_lower:
            if "high volatility" in description_lower:
                return "HIGH_VOLATILITY_BEAR"
            if "low volatility" in description_lower:
                return "STRONG_BEAR_TREND"
            return "MODERATE_BEAR_TREND"

        if "sideways" in description_lower:
            if "high volatility" in description_lower:
                return "VOLATILE_SIDEWAYS"
            if "low volatility" in description_lower:
                return "TIGHT_SIDEWAYS_RANGE"
            return "WIDE_SIDEWAYS_RANGE"

        # Transition states
        if "transition" in description_lower:
            if "bull to bear" in description_lower:
                return "BULL_TO_BEAR_TRANSITION"
            if "bear to bull" in description_lower:
                return "BEAR_TO_BULL_TRANSITION"
            return "TRANSITION_REGIME"

        # Specialized states
        if "accumulation" in description_lower:
            return "ACCUMULATION_PHASE"
        if "distribution" in description_lower:
            return "DISTRIBUTION_PHASE"
        if "breakout" in description_lower:
            return "BREAKOUT_PREPARATION"

        # Volatility-based classification
        if "high volatility" in description_lower:
            if "stressed" in description_lower:
                return "EXTREME_VOLATILITY"
            return "HIGH_VOLATILITY_REGIME"

        if "low volatility" in description_lower:
            return "LOW_VOLATILITY_RANGE"

        # Liquidity-based classification
        if (
            "limited liquidity" in description_lower
            or "low liquidity" in description_lower
        ):
            return "LOW_LIQUIDITY_REGIME"
        if (
            "abundant liquidity" in description_lower
            or "high liquidity" in description_lower
        ):
            return "HIGH_LIQUIDITY_REGIME"

        # Default classification based on cluster characteristics
        return f"REGIME_{cluster_id}"


# Convenience function for easy integration