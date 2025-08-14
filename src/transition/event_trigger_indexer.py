# src/transition/event_trigger_indexer.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.training.enhanced_training_manager import EnhancedTrainingManager
from src.utils.logger import system_logger
from src.analyst.meta_labeling_system import MetaLabelingSystem


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
    Build event triggers (t=0) from meta-label intensities with safeguards:
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
            use_reliability_weighting=bool(tm_cfg.get("use_reliability_weighting", True)),
            use_rising_edge_only=bool(tm_cfg.get("use_rising_edge_only", True)),
            preserve_secondary_labels=bool(tm_cfg.get("preserve_secondary_labels", True)),
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
        re = (above.diff().fillna(0) > 0).astype(bool)
        return re

    def _make_windows(self, indices: np.ndarray) -> np.ndarray:
        pre = self.event_cfg.pre_window
        post = self.event_cfg.post_window
        starts = indices - pre
        ends = indices + post
        return np.stack([starts, ends], axis=1)

    @staticmethod
    def _interval_iou(a: np.ndarray, b: np.ndarray) -> float:
        # a,b: [start, end] inclusive windows
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

    def _apply_cooldown(self, sorted_events: list[dict]) -> list[dict]:
        # Enforce per-label cooldown on sorted-by-time events
        cooldown = self.event_cfg.label_cooldown_bars
        last_idx_by_label: dict[str, int] = {}
        out: list[dict] = []
        for ev in sorted_events:
            lab = ev["event_label"]
            last = last_idx_by_label.get(lab, -10**9)
            if ev["row_index"] - last < cooldown:
                continue
            out.append(ev)
            last_idx_by_label[lab] = ev["row_index"]
        return out

    def _compute_intensities_if_missing(
        self,
        combined_df: pd.DataFrame,
        price_data: pd.DataFrame | None,
        volume_data: pd.DataFrame | None,
        candidate_labels: Iterable[str] | None,
    ) -> pd.DataFrame:
        # If intensity_ columns exist, return as-is
        int_cols = [c for c in combined_df.columns if c.startswith("intensity_")]
        if int_cols:
            return combined_df
        # Try to compute intensities using MetaLabelingSystem
        try:
            meta = MetaLabelingSystem(self.config)
            labels = candidate_labels
            if labels is None:
                labels = meta.all_labels
            out = combined_df.copy()
            # Fallback to active_ columns if present (binary intensities)
            act_cols = [c for c in combined_df.columns if c.startswith("active_")]
            if act_cols:
                for ac in act_cols:
                    out[ac.replace("active_", "intensity_")] = combined_df[ac].astype(float)
                return out
            # Coarse proxy intensities if price/volume provided
            if price_data is not None and not price_data.empty and volume_data is not None:
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
                            self.logger.warning(f"Technical indicators failed at i={i} for {lab}: {e}")
                        try:
                            feats.update(meta._calculate_volume_features(v_slice))
                        except Exception as e:
                            self.logger.warning(f"Volume features failed at i={i} for {lab}: {e}")
                        try:
                            feats.update(meta._calculate_price_action_patterns(p_slice))
                        except Exception as e:
                            self.logger.warning(f"Price action patterns failed at i={i} for {lab}: {e}")
                        try:
                            feats.update(meta._calculate_volatility_patterns(p_slice))
                        except Exception as e:
                            self.logger.warning(f"Volatility patterns failed at i={i} for {lab}: {e}")
                        try:
                            feats.update(meta._calculate_momentum_patterns(p_slice))
                        except Exception as e:
                            self.logger.warning(f"Momentum patterns failed at i={i} for {lab}: {e}")
                        try:
                            vals.append(meta._compute_label_intensity(lab, p_slice, v_slice, feats))
                        except Exception as e:
                            self.logger.warning(f"Intensity computation failed at i={i} for {lab}: {e}")
                            vals.append(0.0)
                    out[f"intensity_{lab}"] = pd.Series(vals, index=price_data.index).reindex(out.index).fillna(0.0)
                return out
        except Exception as e:
            self.logger.warning(f"Intensity backfill failed: {e}")
        return combined_df

    def build_event_index(
        self,
        combined_df: pd.DataFrame,
        price_data: pd.DataFrame | None = None,
        volume_data: pd.DataFrame | None = None,
        candidate_labels: Iterable[str] | None = None,
        timeframe: str | None = None,
        instrument_id: str | None = None,
    ) -> pd.DataFrame:
        """
        Input combined_df includes meta-label intensity/active columns (from existing pipeline).
        Returns a DataFrame with:
          - timestamp (index of combined_df)
          - row_index (int index)
          - event_label (primary anchor)
          - intensity, weighted_intensity
          - secondary_labels (list[str]) if preserve_secondary_labels is True
          - timeframe, instrument_id
        """
        if combined_df is None or combined_df.empty:
            return pd.DataFrame()

        # Ensure intensities are available
        combined_df = self._compute_intensities_if_missing(combined_df, price_data, volume_data, candidate_labels)

        # Determine candidate labels from columns
        if candidate_labels is None:
            # Look for intensity_ columns: intensity_<LABEL>
            intensity_cols = [c for c in combined_df.columns if c.startswith("intensity_")]
            candidate_labels = [c.replace("intensity_", "") for c in intensity_cols]
        labels = list(candidate_labels)
        if not labels:
            return pd.DataFrame()

        # Precompute primary activations by rising edge
        events: list[dict] = []
        # base index for windowing
        base_index = combined_df.index
        for lab in labels:
            thr = float(self.activation_thresholds.get(lab, 0.5))
            inten_col = f"intensity_{lab}"
            if inten_col not in combined_df.columns:
                continue
            series = pd.to_numeric(combined_df[inten_col], errors="coerce").fillna(0.0)
            if self.event_cfg.use_rising_edge_only:
                edges = self._rising_edge(series, thr)
            else:
                edges = series >= thr
            trigger_idx = np.where(edges.values)[0]
            # Pre-extract intensity columns for secondary lookup to avoid constructing Series repeatedly
            secondary_cols = {other_lab: f"intensity_{other_lab}" for other_lab in labels if other_lab != lab and f"intensity_{other_lab}" in combined_df.columns}
            for ridx in trigger_idx:
                ts = base_index[ridx]
                intensity = float(series.iat[ridx])
                weighted = self._weighted_intensity(lab, intensity)
                # Collect secondary co-occurring labels above threshold at the same row
                secondary: list[str] = []
                if self.event_cfg.preserve_secondary_labels:
                    for other_lab, colname in secondary_cols.items():
                        inten2 = float(combined_df[colname].iat[ridx])
                        thr2 = float(self.activation_thresholds.get(other_lab, 0.5))
                        if inten2 >= thr2:
                            secondary.append(other_lab)
                events.append({
                    "timestamp": ts,
                    "row_index": int(ridx),
                    "event_label": lab,
                    "intensity": intensity,
                    "weighted_intensity": weighted,
                    "secondary_labels": secondary,
                    "timeframe": timeframe or combined_df.get("timeframe", pd.Series([None]*len(combined_df), index=base_index)).iat[ridx],
                    "instrument_id": instrument_id,
                })

        if not events:
            return pd.DataFrame()

        # Sort by time for cooldown
        events_sorted = sorted(events, key=lambda r: r["row_index"])
        events_cd = self._apply_cooldown(events_sorted)
        # Apply global NMS on windows
        events_nms = self._nms(events_cd)

        # Keep secondary labels info
        out_df = pd.DataFrame(events_nms)
        out_df = out_df.sort_values("row_index").reset_index(drop=True)
        return out_df