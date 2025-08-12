import numpy as np
import pandas as pd
from typing import Any, Dict, List

from src.utils.logger import system_logger


class OrderBookAnalyzer:
    """Analyze order book snapshots for walls and compute features.

    Assumptions:
    - Input snapshots as DataFrame with columns: ['bid_price','bid_size','ask_price','ask_size'] or aggregated ladders
    - For correlation, S/R zones provided as DataFrame or dict with centers and scores
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.logger = system_logger.getChild("OrderBookAnalyzer")

    def identify_walls(self, book_df: pd.DataFrame, price_col: str, size_col: str, top_k: int = 5) -> pd.DataFrame:
        """Identify top-K size clusters (walls) on one side of the book."""
        try:
            df = book_df[[price_col, size_col]].dropna().copy()
            if df.empty:
                return pd.DataFrame(columns=["price", "size"])  # empty
            # Group by price level if needed; take max size per price
            grouped = df.groupby(price_col, as_index=False)[size_col].sum()
            grouped = grouped.rename(columns={price_col: "price", size_col: "size"})
            walls = grouped.sort_values("size", ascending=False).head(top_k).reset_index(drop=True)
            return walls
        except Exception as e:
            self.logger.warning(f"identify_walls failed: {e}")
            return pd.DataFrame(columns=["price", "size"])  # empty

    def compute_wall_features(
        self,
        mid_price: float,
        bid_walls: pd.DataFrame,
        ask_walls: pd.DataFrame,
    ) -> Dict[str, float]:
        """Compute nearest wall distances/sizes and imbalance features."""
        try:
            features: Dict[str, float] = {
                "nearest_bid_wall_dist_pct": 1.0,
                "nearest_ask_wall_dist_pct": 1.0,
                "nearest_bid_wall_size": 0.0,
                "nearest_ask_wall_size": 0.0,
                "wall_imbalance": 0.0,
            }
            if mid_price <= 0:
                return features

            if bid_walls is not None and not bid_walls.empty:
                below = bid_walls[bid_walls["price"] <= mid_price]
                if not below.empty:
                    nearest_bid = below.iloc[(mid_price - below["price"]).abs().argmin()]
                    features["nearest_bid_wall_size"] = float(nearest_bid["size"]) 
                    features["nearest_bid_wall_dist_pct"] = float((mid_price - nearest_bid["price"]) / mid_price)
            if ask_walls is not None and not ask_walls.empty:
                above = ask_walls[ask_walls["price"] >= mid_price]
                if not above.empty:
                    nearest_ask = above.iloc[(above["price"] - mid_price).abs().argmin()]
                    features["nearest_ask_wall_size"] = float(nearest_ask["size"]) 
                    features["nearest_ask_wall_dist_pct"] = float((nearest_ask["price"] - mid_price) / mid_price)

            total_bid = float(bid_walls["size"].sum()) if bid_walls is not None and not bid_walls.empty else 0.0
            total_ask = float(ask_walls["size"].sum()) if ask_walls is not None and not ask_walls.empty else 0.0
            denom = max(1e-8, total_bid + total_ask)
            features["wall_imbalance"] = (total_bid - total_ask) / denom
            return features
        except Exception as e:
            self.logger.warning(f"compute_wall_features failed: {e}")
            return {
                "nearest_bid_wall_dist_pct": 1.0,
                "nearest_ask_wall_dist_pct": 1.0,
                "nearest_bid_wall_size": 0.0,
                "nearest_ask_wall_size": 0.0,
                "wall_imbalance": 0.0,
            }

    def correlate_walls_with_sr(
        self,
        wall_prices: List[float],
        sr_centers: List[float],
        tol_pct: float = 0.002,
    ) -> Dict[str, float]:
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

    def correlate_from_files(self, sr_zones_file: str, book_file: str) -> Dict[str, float]:
        """Load SR zones and order book walls from files and compute correlation metrics."""
        try:
            sr = pd.read_parquet(sr_zones_file) if sr_zones_file.endswith(".parquet") else pd.read_csv(sr_zones_file)
            book = pd.read_parquet(book_file) if book_file.endswith(".parquet") else pd.read_csv(book_file)
            # Expect sr to have 'center' and 'side', and book to have bid/ask ladders
            centers = sr["center"].dropna().astype(float).tolist() if "center" in sr.columns else []
            # Identify top walls from book snapshot
            bid_walls = self.identify_walls(book, price_col="bid_price", size_col="bid_size", top_k=10)
            ask_walls = self.identify_walls(book, price_col="ask_price", size_col="ask_size", top_k=10)
            wall_prices = []
            wall_prices.extend(bid_walls.get("price", pd.Series([])).astype(float).tolist())
            wall_prices.extend(ask_walls.get("price", pd.Series([])).astype(float).tolist())
            metrics = self.correlate_walls_with_sr(wall_prices, centers)
            self.logger.info(f"Order book vs SR correlation: {metrics}")
            return metrics
        except Exception as e:
            self.logger.warning(f"correlate_from_files failed: {e}")
            return {"overlap_ratio": 0.0, "avg_min_dist_to_sr": 1.0}