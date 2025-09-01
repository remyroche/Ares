from src.utils.logger import system_logger
from typing import Any
import numpy as np
import pandas as pd
from src.utils.centralized_decorators_simple import (
    comprehensive_data_validation,
    validate_data_quality,
    with_tracing_span,
)


class OrderBookAnalyzer:
    """Analyze order book snapshots for walls and compute features.

    Assumptions:
    - Input snapshots as DataFrame with columns: ['bid_price','bid_size','ask_price','ask_size'] or aggregated ladders
    - For correlation, S/R zones provided as DataFrame or dict with centers and scores
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.logger = system_logger.getChild("OrderBookAnalyzer")

    @validate_data_quality(validation_level="WARNING")
    @with_tracing_span("wall_identification")
    def identify_walls(
        self,
        book_df: pd.DataFrame,
        price_col: str,
        size_col: str,
        top_k: int = 5,
    ) -> pd.DataFrame:
        """Identify top-K size clusters (walls) on one side of the book."""
        try:
            df = book_df[[price_col, size_col]].dropna().copy()
            if df.empty:
                return pd.DataFrame(columns=["price", "size"])  # empty
            # Group by price level if needed; take max size per price
            grouped = df.groupby(price_col, as_index=False)[size_col].sum()
            grouped = grouped.rename(columns={price_col: "price", size_col: "size"})
            return (
                grouped.sort_values("size", ascending=False)
                .head(top_k)
                .reset_index(drop=True)
            )
        except Exception as e:
            self.logger.warning(f"identify_walls failed: {e}")
            return pd.DataFrame(columns=["price", "size"])  # empty

    @validate_data_quality(validation_level="WARNING")
    @with_tracing_span("wall_features_computation")
    def correlate_walls_with_sr(
        self,
        wall_prices: list[float],
        sr_centers: list[float],
        tol_pct: float = 0.002,
    ) -> dict[str, float]:
        """Compute simple correlation/overlap metrics between wall locations and S/R centers."""
        try:
            if not wall_prices or not sr_centers:
                return {"overlap_ratio": 0.0, "avg_min_dist_to_sr": 1.0}
            wp = np.array(wall_prices)
            sc = np.array(sr_centers)
            # Overlap: fraction of walls within tolerance of any SR center
            overlaps = []
            min_dists = []
            for p in wp:
                dists = np.abs(sc - p) / np.maximum(1e-8, p)
                overlaps.append(float((dists <= tol_pct).any()))
                min_dists.append(float(np.min(dists)))
            return {
                "overlap_ratio": float(np.mean(overlaps)),
                "avg_min_dist_to_sr": float(np.mean(min_dists)),
            }
        except Exception as e:
            self.logger.warning(f"correlate_walls_with_sr failed: {e}")
            return {"overlap_ratio": 0.0, "avg_min_dist_to_sr": 1.0}
