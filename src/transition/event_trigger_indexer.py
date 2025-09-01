# src/transition/event_trigger_indexer.py

from collections.abc import Iterable
from src.analyst.meta_labeling_system import CompositeHMMRegimeSystem
from src.utils.logger import system_logger
from typing import TYPE_CHECKING, Any
from dataclasses import dataclass
from src.training.enhanced_training_manager import EnhancedTrainingManager
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass  # TODO: Add proper implementation
@dataclass
class EventConfig:
    pre_window: int
    post_window: int
    label_cooldown_bars: int
    window_iou_threshold: float
    use_reliability_weighting: bool
    use_rising_edge_only: bool
    preserve_secondary_labels: bool

class EventTriggerIndexer:
    """
    Build event triggers (t, 0) from meta-label intensities with safeguards:
    - optional reliability-weighted intensity
    - rising-edge detection against activation thresholds
    - per-label cooldown to avoid clustering
    - global non-maximum suppression on overlapping windows (IoU)
    - preserve secondary co-occurring labels as multi-hot context
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("EventTriggerIndexer")
        tm_cfg = (config or {}).get("TRANSITION_MODELING", {})
        self.event_cfg = EventConfig(
            pre_window=int(tm_cfg.get("pre_window", 60)),
            post_window=int(tm_cfg.get("post_window", 20)),
            label_cooldown_bars=int(tm_cfg.get("label_cooldown_bars", 45)),
            window_iou_threshold=float(tm_cfg.get("window_iou_threshold", 0.5)),
            use_reliability_weighting=bool(
                tm_cfg.get("use_reliability_weighting", True),
            ),
            use_rising_edge_only=bool(tm_cfg.get("use_rising_edge_only", True)),
            preserve_secondary_labels=bool(
                tm_cfg.get("preserve_secondary_labels", True),
            ),
        )

        # Load thresholds and reliability
        self.etm = EnhancedTrainingManager(config)
        self.activation_thresholds = self.etm.get_activation_thresholds()
        self.label_reliability = self.etm.get_label_reliability()

    def _weighted_intensity(self, label: str, intensity: float) -> float:
        if not self.event_cfg.use_reliability_weighting:
            return float(intensity)
        rel = float(self.label_reliability.get(label, 1.0))
        return float(np.clip(intensity * rel, 0.0, 1.0))

    def _rising_edge(self, series: pd.Series, threshold: float) -> pd.Series:
        above = (series >= threshold).astype(int)
        # Rising edge: 0 -> 1 transition
        return (above.diff().fillna(0) > 0).astype(bool)

    def _make_windows(self, indices: np.ndarray) -> np.ndarray:
        pre = self.event_cfg.pre_window
        post = self.event_cfg.post_window
        starts = indices - pre
        ends = indices + post
        return np.stack([starts, ends], axis=1)

    @staticmethod
    def _interval_iou(a: np.ndarray, b: np.ndarray) -> float:
        # a, b: [start, end] inclusive windows
        inter_start = max(a[0], b[0])
        inter_end = min(a[1], b[1])
        inter = max(0, inter_end - inter_start + 1)
        union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter
        return float(inter / union) if union > 0 else 0.0

    def _nms(self, event_rows: list[dict]) -> list[dict]:
        if not event_rows:
            return []
        # Convert to arrays for efficiency
        idx = np.array([r["row_index"] for r in event_rows], dtype=np.int64)
        scores = np.array([r["weighted_intensity"] for r in event_rows], dtype=float)
        windows = self._make_windows(idx)
        order = np.argsort(-scores)  # descending by weighted score
        keep: list[int] = []
        suppressed = np.zeros(len(order), dtype=bool)
        iou_thr = self.event_cfg.window_iou_threshold
        for i, o in enumerate(order):
            if suppressed[i]:
                continue
            keep.append(o)
            win_o = windows[o]
            # suppress overlapping windows with high IoU
            for j in range(i + 1, len(order)):
                if suppressed[j]:
                    continue
                o2 = order[j]
                if self._interval_iou(win_o, windows[o2]) >= iou_thr:
                    suppressed[j] = True
        return [event_rows[k] for k in keep]

    def _compute_intensities_if_missing(
        self, combined_df: pd.DataFrame,
        price_data: pd.DataFrame | None = None,
        volume_data: pd.DataFrame | None = None,
        candidate_labels: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        # If intensity_ columns exist = return as-is
        int_cols = [c for c in combined_df.columns if c.startswith("intensity_")]
        if int_cols:
            return combined_df
        # Try to compute intensities using CompositeHMMRegimeSystem
        try:
            meta = CompositeHMMRegimeSystem(self.config)
            labels = candidate_labels
            if labels is None:
                labels = meta.all_labels
            out = combined_df.copy()
            # Fallback to active_ columns if present (binary intensities)
            act_cols = [c for c in combined_df.columns if c.startswith("active_")]
            if act_cols:
                for ac in act_cols:
                    out[ac.replace("active_", "intensity_")] = combined_df[ac].astype(
                        float,
                    )
                return out
            # Coarse proxy intensities if price/volume provided
            if (
                price_data is not None
                and not price_data.empty
                and volume_data is not None
            ):
                max_labels = 50
                labels = list(labels)[:max_labels]
                for lab in labels:
                    vals: list[float] = []
                    for i in range(len(price_data)):
                        p_slice = price_data.iloc[: i + 1]
                        v_slice = volume_data.iloc[: i + 1]
                        feats = {}
                        try:
                            feats.update(meta._calculate_technical_indicators(p_slice))
                        except Exception as e:
                            self.logger.warning(
                                f"Technical indicators failed at i={i} for {lab}: {e}",
                            )
                        try:
                            feats.update(meta._calculate_volume_features(v_slice))
                        except Exception as e:
                            self.logger.warning(
                                f"Volume features failed at i={i} for {lab}: {e}",
                            )
                        try:
                            feats.update(meta._calculate_price_action_patterns(p_slice))
                        except Exception as e:
                            self.logger.warning(
                                f"Price action patterns failed at i={i} for {lab}: {e}",
                            )
                        try:
                            feats.update(meta._calculate_volatility_patterns(p_slice))
                        except Exception as e:
                            self.logger.warning(
                                f"Volatility patterns failed at i={i} for {lab}: {e}",
                            )
                        try:
                            feats.update(meta._calculate_momentum_patterns(p_slice))
                        except Exception as e:
                            self.logger.warning(
                                f"Momentum patterns failed at i={i} for {lab}: {e}",
                            )
                        try:
                            vals.append(
                                meta._compute_label_intensity(
                                    lab = p_slice,
                                    v_slice = feats,
                                ),
                            )
                        except Exception as e:
                            self.logger.warning(
                                f"Intensity computation failed at i={i} for {lab}: {e}",
                            )
                            vals.append(0.0)
                    out[f"intensity_{lab}"] = (
                        pd.Series(vals, index = price_data.index)
                        .reindex(out.index)
                        .fillna(0.0)
                    )
                return out
        except Exception as e:
            self.logger.warning(f"Intensity backfill failed: {e}")
        return combined_df
