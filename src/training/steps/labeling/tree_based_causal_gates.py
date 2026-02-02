from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from numba import njit  # type: ignore
except Exception:
    njit = None

from src.utils.tprint import tprint_info, tprint_success, tprint_warning


if njit is not None:
    @njit
    def _traverse_rows_numba(
        X: np.ndarray,
        feat_idx: np.ndarray,
        thr: np.ndarray,
        left: np.ndarray,
        right: np.ndarray,
        leaf_id: np.ndarray,
        is_leaf: np.ndarray,
    ) -> np.ndarray:
        n = X.shape[0]
        out = np.empty(n, dtype=np.int64)
        for i in range(n):
            node = 0
            while not is_leaf[node]:
                j = feat_idx[node]
                if X[i, j] <= thr[node]:
                    node = left[node]
                else:
                    node = right[node]
            out[i] = leaf_id[node]
        return out
else:
    _traverse_rows_numba = None


# ============================================================
# Purged K-Fold (blocked) utilities — de Prado compatible
# ============================================================
def make_purged_kfold_folds(
    index: pd.Index,
    n_folds: int = 8,
    purge: int = 0,
    embargo: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Blocked (contiguous) K-fold with purge/embargo.
    Training uses all samples except validation block, with purge around val
    and embargo after val. This is generally better than walk-forward for
    estimating stability dispersion across folds because train mass is balanced.

    Args:
        index: time-ordered index (used for length only)
        n_folds: number of folds (recommend 8–12)
        purge: number of samples to purge before val start through val end
        embargo: number of samples to embargo after val end

    Returns:
        list of (train_idx, val_idx)
    """
    n = len(index)
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if n < n_folds * 20:
        raise ValueError("Not enough samples for requested folds.")

    all_idx = np.arange(n)

    # contiguous validation blocks
    fold_sizes = [n // n_folds] * n_folds
    for i in range(n % n_folds):
        fold_sizes[i] += 1

    starts = np.cumsum([0] + fold_sizes[:-1])
    ends = starts + np.array(fold_sizes)

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_folds):
        v0, v1 = int(starts[k]), int(ends[k])  # [v0, v1)
        val_idx = all_idx[v0:v1]

        train_mask = np.ones(n, dtype=bool)
        train_mask[v0:v1] = False  # remove validation block

        # purge: remove samples close to val block (label overlap / leakage)
        if purge > 0:
            train_mask[max(0, v0 - purge) : v1] = False

        # embargo: remove samples immediately after val block
        if embargo > 0:
            train_mask[v0 : min(n, v1 + embargo)] = False

        train_idx = all_idx[train_mask]
        folds.append((train_idx, val_idx))

    return folds


# ============================================================
# Stability metrics (Sharpe + robust dispersion)
# ============================================================
def _safe_sharpe(u: np.ndarray, eps: float = 1e-12) -> float:
    """
    Robust Sharpe proxy: mean(u)/std(u).
    Returns NaN for insufficient samples or near-zero std.
    """
    u = u[np.isfinite(u)]
    if u.size < 10:
        return np.nan
    mu = float(np.mean(u))
    sd = float(np.std(u, ddof=1))
    if sd < eps:
        return np.nan
    return mu / sd


def stability_score_from_fold_metrics(
    fm: np.ndarray,
    mode: str = "min_minus_iqr",
    disp_penalty: float = 0.5,
) -> float:
    """
    Convert per-fold metrics -> stability score.

    Recommended:
      - "min_minus_iqr": min(fm) - lambda * IQR(fm)
        (robust for small K, less noisy than variance)
    Alternatives:
      - "min": minimax robustness (maximize worst fold)
      - "mean_minus_iqr": mean(fm) - lambda*IQR(fm)
      - "neg_iqr": -IQR(fm)

    Returns:
        float stability score; -inf if insufficient metrics.
    """
    fm = np.asarray(fm, dtype=np.float32)
    fm = fm[np.isfinite(fm)]
    if fm.size < 3:
        return -np.inf

    q75, q25 = np.percentile(fm, 75), np.percentile(fm, 25)
    iqr = float(q75 - q25)

    if mode == "min_minus_iqr":
        return float(np.min(fm) - disp_penalty * iqr)
    if mode == "mean_minus_iqr":
        return float(np.mean(fm) - disp_penalty * iqr)
    if mode == "min":
        return float(np.min(fm))
    if mode == "neg_iqr":
        return -iqr

    raise ValueError(f"Unknown stability mode: {mode}")


# ============================================================
# Regime tree data structures
# ============================================================
@dataclass
class LeafAssignment:
    leaf_id: int
    expert_best: str
    expert_weights: Dict[str, float]  # soft weights (sum to 1)
    score_best: float
    scores_by_expert: Dict[str, float]
    n_samples: int
    fold_scores_by_expert: Dict[str, List[float]] = field(default_factory=dict)
    valid_folds_by_expert: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LeafAssignment":
        return cls(**d)


@dataclass
class Node:
    node_id: int
    depth: int
    is_leaf: bool
    leaf_id: Optional[int] = None

    feature: Optional[str] = None
    feature_idx: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "node_id": self.node_id,
            "depth": self.depth,
            "is_leaf": self.is_leaf,
            "leaf_id": self.leaf_id,
            "feature": self.feature,
            "feature_idx": self.feature_idx,
            "threshold": self.threshold,
        }
        if self.left: d["left"] = self.left.to_dict()
        if self.right: d["right"] = self.right.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Node":
        d = dict(d)
        left_dict = d.pop("left", None)
        right_dict = d.pop("right", None)
        node = cls(**d)
        if left_dict: node.left = cls.from_dict(left_dict)
        if right_dict: node.right = cls.from_dict(right_dict)
        return node


@dataclass
class NodeStats:
    """
    Stores stability stats for a region (node) so pruning can be done correctly
    without recomputing expensive stability metrics.

    parent_best_score: stability score of best expert on the node region
    children_weighted_score: weighted avg of children best scores
    n_left/n_right: sample counts
    parent_best_expert: expert with best stability score on parent region
    """
    parent_best_score: float
    children_weighted_score: float
    n_left: int
    n_right: int
    parent_best_expert: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NodeStats":
        return cls(**d)


# ============================================================
# Stability-Optimized Regime Tree
# ============================================================
class StabilityRegimeTree:
    """
    Learns a partition R(x) over *state features* Z that maximizes stability
    (not returns) for routing among specialists.

    Key properties:
      - No look-ahead in Z preprocessing (ffill only; no bfill).
      - Purged K-fold evaluation (balanced training mass).
      - Fast fold intersection via boolean masks (no np.isin).
      - Robust stability objective using IQR across folds.
      - Tracks feature_importances_ via accumulated best_gain.
      - Split requires min_stability_gain to avoid noisy micro-splits.

    Fit inputs:
      - Z: pd.DataFrame state features (time-ordered rows)
      - preds_oof: dict expert -> np.ndarray OOF predictions aligned to Z
      - y: np.ndarray aligned to Z (future return / outcome proxy)
      - folds: list(train_idx, val_idx) from make_purged_kfold_folds()

    Inference:
      - predict_leaf_ids(Z_new): leaf assignment
      - route(Z_new, preds_by_expert): routed expert score (hard or soft)
    """

    def __init__(
        self,
        max_depth: int = 2,
        min_leaf_samples: float | int = 0.005,
        min_leaf_val_per_fold: int = 200,
        n_thresholds: int = 15,
        stability_mode: str = "min_minus_iqr",
        disp_penalty: float = 0.5,
        utility_transform: str = "tanh",  # tanh / clip / identity
        soft_weights: bool = True,
        top_k_weights: int = 3,
        weight_temperature: float = 8.0,
        min_stability_gain: float = 0.02,
        min_valid_fold_frac: float = 0.6,
        min_total_leaf_val_samples: Optional[int] = None,
        expert_prune_gap: float = 0.1,
        expert_prune_cum_weight: float = 0.9,
        expert_prune_worst_fold: float = -0.25,
        expert_prune_corr: float = 0.95,
        merge_leaf_l1_eps: float = 0.2,
        prune_alpha: float = 0.03,
        child_min_leaf_fraction: float = 0.35,
        nan_score_floor: float = -1e6,
        min_gain_relax_parent_abs: float = 0.01,
        min_gain_relaxed: float = 0.0001,
        min_asset_leaf_samples: int = 0,
        zscore_mode: str = "expanding",
        zscore_window: Optional[int] = None,
        split_imbalance_penalty: float = 0.01,
        use_binned_splits: bool = False,
        n_feature_bins: int = 255,
        use_numba_inference: bool = True,
        verbose: bool = False,
    ):
        self.max_depth = int(max_depth)
        self.min_leaf_samples = min_leaf_samples
        self._effective_min_leaf_samples = 0  # Calculated in fit
        self._child_min_leaf_samples = 0
        self.min_leaf_val_per_fold = int(min_leaf_val_per_fold)
        self.n_thresholds = int(n_thresholds)
        self.stability_mode = str(stability_mode)
        self.disp_penalty = float(disp_penalty)
        self.utility_transform = str(utility_transform)
        self.soft_weights = bool(soft_weights)
        self.top_k_weights = int(top_k_weights)
        self.weight_temperature = float(weight_temperature)
        self.min_stability_gain = float(min_stability_gain)
        self.min_valid_fold_frac = float(min_valid_fold_frac)
        self.min_total_leaf_val_samples = (
            int(min_total_leaf_val_samples) if min_total_leaf_val_samples is not None else None
        )
        self.expert_prune_gap = float(expert_prune_gap)
        self.expert_prune_cum_weight = float(expert_prune_cum_weight)
        self.expert_prune_worst_fold = float(expert_prune_worst_fold)
        self.expert_prune_corr = float(expert_prune_corr)
        self.merge_leaf_l1_eps = float(merge_leaf_l1_eps)
        self.prune_alpha = float(prune_alpha)
        self.child_min_leaf_fraction = float(child_min_leaf_fraction)
        self.nan_score_floor = float(nan_score_floor)
        self.min_gain_relax_parent_abs = float(min_gain_relax_parent_abs)
        self.min_gain_relaxed = float(min_gain_relaxed)
        self.min_asset_leaf_samples = int(min_asset_leaf_samples)
        self.zscore_mode = str(zscore_mode)
        self.zscore_window = int(zscore_window) if zscore_window is not None else None
        self.split_imbalance_penalty = float(split_imbalance_penalty)
        self.use_binned_splits = bool(use_binned_splits)
        self.n_feature_bins = int(n_feature_bins)
        self.use_numba_inference = bool(use_numba_inference)
        self.verbose = bool(verbose)

        # DEBUG: Check if _find_best_split exists
        if verbose:
            has_split = '_find_best_split' in dir(self)
            tprint_info(f"🐛 [RegimeTree] Debug: has _find_best_split={has_split}")

        self.root_: Optional[Node] = None
        self.leaves_: Dict[int, LeafAssignment] = {}
        self.features_: List[str] = []
        self._feature_to_index_: Dict[str, int] = {}
        self.experts_: List[str] = []
        self.fold_val_masks_: List[np.ndarray] = []
        self.feature_importances_: Dict[str, float] = {}
        self.node_stats_: Dict[int, NodeStats] = {}
        self._preds_oof_cache_: Dict[str, np.ndarray] = {}
        self._utility_cache_: Dict[str, np.ndarray] = {}
        self._y_cache_: Optional[np.ndarray] = None
        self._leaf_ids_cache_: Optional[np.ndarray] = None
        self._asset_id_codes_: Optional[np.ndarray] = None
        self._asset_id_max_: int = 0
        self._Z_binned_: Optional[np.ndarray] = None
        self._bin_edges_by_feature_: Dict[str, np.ndarray] = {}
        self._flat_tree_cache_: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None
        self.split_diagnostics: Counter = Counter()

    # -------------------------
    # Public API
    # -------------------------
    def fit(
        self,
        Z: pd.DataFrame,
        preds_oof: Dict[str, np.ndarray],
        y: np.ndarray,
        folds: List[Tuple[np.ndarray, np.ndarray]],
        asset_ids: Optional[np.ndarray | pd.Series | List[str]] = None,
    ) -> "StabilityRegimeTree":
        tprint_info(f"🌲 [RegimeTree] Starting fit with {len(Z)} samples and {len(preds_oof)} specialists")
        Z = self._validate_Z(Z)
        y = self._validate_y(y, len(Z))
        preds_oof = self._validate_preds(preds_oof, len(Z))

        # Calculate effective min leaf samples if percentage
        if isinstance(self.min_leaf_samples, float):
            self._effective_min_leaf_samples = max(1, int(len(Z) * self.min_leaf_samples))
        else:
            self._effective_min_leaf_samples = int(self.min_leaf_samples)
        self._child_min_leaf_samples = max(
            5,
            int(max(1, self._effective_min_leaf_samples) * self.child_min_leaf_fraction),
        )

        tprint_info(
            "   🌲 Effective min_leaf_samples: "
            f"{self._effective_min_leaf_samples} ({self.min_leaf_samples}), "
            f"child_min_leaf_samples: {self._child_min_leaf_samples}"
        )

        self.split_diagnostics = Counter()
        self.features_ = list(Z.columns)
        self._feature_to_index_ = {c: j for j, c in enumerate(self.features_)}
        self.experts_ = list(preds_oof.keys())
        self.feature_importances_ = {f: 0.0 for f in self.features_}
        self._preds_oof_cache_ = preds_oof
        self._y_cache_ = y
        self._utility_cache_ = {k: self._utility(y, s) for k, s in preds_oof.items()}

        if asset_ids is not None:
            asset_arr = np.asarray(asset_ids)
            if asset_arr.shape[0] != len(Z):
                raise ValueError("asset_ids must align with Z length.")
            asset_codes = pd.Categorical(asset_arr).codes
            self._asset_id_codes_ = asset_codes
            self._asset_id_max_ = int(asset_codes.max()) if asset_codes.size else 0

        # Precompute fold validation masks once (fast intersections later)
        self.fold_val_masks_ = []
        n = len(Z)
        for _, val_idx in folds:
            m = np.zeros(n, dtype=bool)
            m[val_idx] = True
            self.fold_val_masks_.append(m)

        # Numpy view for fast thresholding
        Z_np = Z.to_numpy(dtype=np.float32, copy=False)
        col_index = self._feature_to_index_

        self._flat_tree_cache_ = None
        self._Z_binned_ = None
        self._bin_edges_by_feature_ = {}
        if self.use_binned_splits:
            n_bins = max(16, min(4096, self.n_feature_bins))
            q = np.linspace(0.0, 100.0, n_bins + 1, dtype=np.float32)[1:-1]
            Z_binned = np.empty(Z_np.shape, dtype=np.uint16)
            for feat, j in col_index.items():
                col = Z_np[:, j]
                edges = np.unique(np.percentile(col, q))
                edges = edges[np.isfinite(edges)]
                self._bin_edges_by_feature_[feat] = edges.astype(np.float32, copy=False)
                if edges.size == 0:
                    Z_binned[:, j] = 0
                else:
                    Z_binned[:, j] = np.searchsorted(edges, col, side="left").astype(np.uint16, copy=False)
            self._Z_binned_ = Z_binned

        # Root mask: all samples
        root_mask = np.ones(n, dtype=bool)

        node_counter = 0
        leaf_counter = 0

        def build_node(idx_mask: np.ndarray, depth: int) -> Node:
            nonlocal node_counter, leaf_counter

            node_id = node_counter
            node_counter += 1

            n_node = int(idx_mask.sum())
            # stopping conditions
            if depth >= self.max_depth:
                self._record_split_diag("max_depth_stop")
            if n_node < self._effective_min_leaf_samples:
                self._record_split_diag("node_min_samples_stop")
            if n_node < 2 * self._child_min_leaf_samples:
                self._record_split_diag("node_child_min_samples_stop")
            if depth >= self.max_depth or n_node < 2 * self._child_min_leaf_samples:
                leaf_id = leaf_counter
                leaf_counter += 1
                node = Node(node_id=node_id, depth=depth, is_leaf=True, leaf_id=leaf_id)
                self.leaves_[leaf_id] = self._assign_leaf(
                    leaf_id=leaf_id,
                    idx_mask=idx_mask,
                    y=y,
                    preds_oof=preds_oof,
                )
                return node

            best = self._find_best_split(
                idx_mask=idx_mask,
                Z_np=Z_np,
                col_index=col_index,
                y=y,
                preds_oof=preds_oof,
            )

            if best is None:
                leaf_id = leaf_counter
                leaf_counter += 1
                node = Node(node_id=node_id, depth=depth, is_leaf=True, leaf_id=leaf_id)
                self.leaves_[leaf_id] = self._assign_leaf(
                    leaf_id=leaf_id,
                    idx_mask=idx_mask,
                    y=y,
                    preds_oof=preds_oof,
                )
                return node

            feat, thr, left_mask, right_mask, gain, parent_score, left_score, right_score, parent_expert = best
            self.feature_importances_[feat] += float(gain)
            n_left = int(left_mask.sum())
            n_right = int(right_mask.sum())
            children_weighted = (n_left * left_score + n_right * right_score) / max(1, n_left + n_right)
            self.node_stats_[node_id] = NodeStats(
                parent_best_score=float(parent_score),
                children_weighted_score=float(children_weighted),
                n_left=n_left,
                n_right=n_right,
                parent_best_expert=parent_expert,
            )

            node = Node(
                node_id=node_id,
                depth=depth,
                is_leaf=False,
                feature=feat,
                feature_idx=int(col_index[feat]),
                threshold=float(thr),
            )
            node.left = build_node(left_mask, depth + 1)
            node.right = build_node(right_mask, depth + 1)
            return node

        self.root_ = build_node(root_mask, 0)
        self._flat_tree_cache_ = None
        self._leaf_ids_cache_ = self.predict_leaf_ids(Z)
        
        # Post-construction refinements (internalized)
        self._prune_leaf_experts()
        if self.prune_alpha > 0:
            self.prune(self.prune_alpha)
        if self.merge_leaf_l1_eps > 0:
            self.merge_similar_leaves(self.merge_leaf_l1_eps)
        
        if self.split_diagnostics:
            tprint_info(
                "   📈 [RegimeTree] Split diagnostics: "
                + ", ".join(f"{k}={v}" for k, v in sorted(self.split_diagnostics.items()))
            )
        self._log_leaf_summaries()

        tprint_success(f"✅ [RegimeTree] Fit complete: {len(self.leaves_)} leaves, {len(self.features_)} features")
        return self

    def predict_leaf_ids(self, Z_new: pd.DataFrame) -> np.ndarray:
        if self.root_ is None:
            raise RuntimeError("Tree is not fitted.")
        Z_new = self._validate_Z(Z_new)
        Z_new = Z_new[self.features_]
        X = Z_new.to_numpy(dtype=np.float32, copy=False)

        if (
            self.use_numba_inference
            and _traverse_rows_numba is not None
            and self.root_ is not None
        ):
            feat_idx, thr, left, right, leaf_id, is_leaf = self._get_flat_tree_arrays()
            return _traverse_rows_numba(X, feat_idx, thr, left, right, leaf_id, is_leaf).astype(int, copy=False)

        out = np.empty(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            out[i] = self._traverse_to_leaf_id(X[i], self.root_)
        return out

    def _get_flat_tree_arrays(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.root_ is None:
            raise RuntimeError("Tree is not fitted.")
        if self._flat_tree_cache_ is not None:
            return self._flat_tree_cache_

        feat_idx_list: List[int] = []
        thr_list: List[float] = []
        left_list: List[int] = []
        right_list: List[int] = []
        leaf_list: List[int] = []
        is_leaf_list: List[bool] = []
        node_to_arr: Dict[int, int] = {}

        def _add(node: Node) -> int:
            if node.node_id in node_to_arr:
                return node_to_arr[node.node_id]
            idx = len(feat_idx_list)
            node_to_arr[node.node_id] = idx

            feat_idx_list.append(int(node.feature_idx) if node.feature_idx is not None else -1)
            thr_list.append(float(node.threshold) if node.threshold is not None else 0.0)
            left_list.append(-1)
            right_list.append(-1)
            leaf_list.append(int(node.leaf_id) if node.leaf_id is not None else -1)
            is_leaf_list.append(bool(node.is_leaf))

            if not node.is_leaf:
                if node.left is None or node.right is None:
                    raise RuntimeError("Malformed tree child pointer.")
                lidx = _add(node.left)
                ridx = _add(node.right)
                left_list[idx] = int(lidx)
                right_list[idx] = int(ridx)

            return idx

        _add(self.root_)

        feat_idx = np.asarray(feat_idx_list, dtype=np.int64)
        thr = np.asarray(thr_list, dtype=np.float32)
        left = np.asarray(left_list, dtype=np.int64)
        right = np.asarray(right_list, dtype=np.int64)
        leaf_id = np.asarray(leaf_list, dtype=np.int64)
        is_leaf = np.asarray(is_leaf_list, dtype=np.bool_)

        self._flat_tree_cache_ = (feat_idx, thr, left, right, leaf_id, is_leaf)
        return self._flat_tree_cache_

    def route(
        self,
        Z_new: pd.DataFrame,
        preds_by_expert: Dict[str, np.ndarray],
    ) -> pd.DataFrame:
        """
        Route and combine expert predictions for new data.
        Returns a DataFrame with:
          - 'signal': The weighted average of experts.
          - 'disagreement': Weighted standard deviation (uncertainty).
          - 'leaf_id': The regime location.
          - 'entropy': Shannon entropy of expert weights.
        """
        if self.root_ is None:
            raise RuntimeError("Tree is not fitted.")

        Z_new = self._validate_Z(Z_new)
        Z_new = Z_new[self.features_]
        n = len(Z_new)

        for e in self.experts_:
            if e not in preds_by_expert:
                raise KeyError(f"Missing expert prediction for: {e}")
            if len(preds_by_expert[e]) != n:
                raise ValueError(f"Expert {e} prediction length mismatch: {len(preds_by_expert[e])} != {n}")

        preds_by_expert_norm: Dict[str, np.ndarray] = {}
        for e in self.experts_:
            preds_by_expert_norm[e] = self._normalize_position_scale(preds_by_expert[e])

        leaf_ids = self.predict_leaf_ids(Z_new)
        signals = np.zeros(n, dtype=np.float32)
        disagreements = np.zeros(n, dtype=np.float32)
        entropies = np.zeros(n, dtype=np.float32)
        best_scores = np.zeros(n, dtype=np.float32)
        best_experts: List[str] = []

        for i, leaf_id in enumerate(leaf_ids):
            leaf = self.leaves_[leaf_id]
            current_signal = 0.0
            active_experts = []

            for e, w in leaf.expert_weights.items():
                if e == "ABSTAIN_SPECIALIST":
                    pred = 0.0
                else:
                    pred = float(preds_by_expert_norm[e][i])
                current_signal += w * pred
                active_experts.append((pred, w))

            signals[i] = current_signal

            weighted_var = 0.0
            for pred, w in active_experts:
                weighted_var += w * (pred - current_signal) ** 2
            disagreements[i] = np.sqrt(max(0.0, weighted_var))

            weights_arr = np.array(list(leaf.expert_weights.values()), dtype=np.float32)
            entropies[i] = float(-np.sum(weights_arr * np.log(weights_arr + 1e-12)))
            best_scores[i] = float(leaf.score_best)
            best_experts.append(leaf.expert_best)

        return pd.DataFrame(
            {
                "signal": signals,
                "disagreement": disagreements,
                "leaf_id": leaf_ids,
                "entropy": entropies,
                "best_score": best_scores,
                "best_expert": best_experts,
            },
            index=Z_new.index,
        )

    # -------------------------
    # Internal helpers
    # -------------------------
    def _validate_Z(self, Z: pd.DataFrame) -> pd.DataFrame:
        if Z is None or Z.empty:
            raise ValueError("Z is empty.")
        Z = Z.replace([np.inf, -np.inf], np.nan)

        # CAUSAL: ffill only (NO bfill)
        Z = Z.ffill()

        # Remaining NaNs are leading NaNs; fill causally with 0.
        # If you prefer to drop warmup rows, do it outside and keep alignment consistent.
        Z = Z.fillna(0.0)

        Z = Z.astype(np.float32)
        if self.zscore_mode == "none":
            Z = Z
        elif self.zscore_mode == "expanding":
            means = Z.expanding(min_periods=2).mean()
            stds = Z.expanding(min_periods=2).std(ddof=0)
            stds = stds.replace(0.0, 1.0).fillna(1.0)
            Z = (Z - means) / stds
        elif self.zscore_mode == "rolling":
            if self.zscore_window is None or self.zscore_window <= 1:
                raise ValueError("zscore_window must be provided (>1) when zscore_mode='rolling'.")
            minp = max(2, int(self.zscore_window // 5))
            means = Z.rolling(self.zscore_window, min_periods=minp).mean()
            stds = Z.rolling(self.zscore_window, min_periods=minp).std(ddof=0)
            stds = stds.replace(0.0, 1.0).fillna(1.0)
            Z = (Z - means) / stds
        else:
            raise ValueError(f"Unknown zscore_mode: {self.zscore_mode}")

        Z = Z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        Z = Z.clip(lower=-5.0, upper=5.0)

        return Z.astype(np.float32)

    @staticmethod
    def _normalize_position_scale(arr: np.ndarray) -> np.ndarray:
        x = np.asarray(arr, dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if x.size == 0:
            return x

        xmin = float(np.min(x))
        xmax = float(np.max(x))
        if xmin >= 0.0 and xmax <= 1.0:
            x = 2.0 * x - 1.0
        else:
            x = np.tanh(x)

        return np.clip(x, -1.0, 1.0)

    def _validate_y(self, y: np.ndarray, n: int) -> np.ndarray:
        y = np.asarray(y, dtype=np.float32)
        if y.ndim != 1 or y.size != n:
            raise ValueError("y must be 1D and aligned with Z.")
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        return y

    def _validate_preds(self, preds: Dict[str, np.ndarray], n: int) -> Dict[str, np.ndarray]:
        if not preds:
            raise ValueError("preds_oof is empty.")
        out: Dict[str, np.ndarray] = {}
        for k, v in preds.items():
            arr = np.asarray(v, dtype=np.float32)
            if arr.ndim != 1 or arr.size != n:
                raise ValueError(f"preds_oof[{k}] must be 1D and aligned with Z.")
            out[k] = self._normalize_position_scale(arr)
        return out

    def _utility(self, y: np.ndarray, s: np.ndarray) -> np.ndarray:
        if self.utility_transform == "tanh":
            g = np.tanh(s)
        elif self.utility_transform == "clip":
            g = np.clip(s, -3.0, 3.0)
        elif self.utility_transform == "identity":
            g = s
        else:
            raise ValueError(f"Unknown utility_transform: {self.utility_transform}")
        return y * g

    def _stability_for_expert_on_mask(
        self,
        idx_mask: np.ndarray,
        y: np.ndarray,
        s_oof: np.ndarray,
        u_oof: Optional[np.ndarray] = None,
    ) -> float:
        """
        Stability score for one expert restricted to idx_mask (a leaf).
        Uses only validation sets, intersected via fast boolean AND.
        """
        fold_metrics: List[float] = []
        invalid_folds = 0
        total_val_samples = 0
        n_folds = len(self.fold_val_masks_)
        min_valid_folds = int(np.ceil(self.min_valid_fold_frac * n_folds))
        max_invalid_folds = max(0, n_folds - min_valid_folds)
        min_total_samples = self.min_total_leaf_val_samples
        if min_total_samples is None:
            min_total_samples = self.min_leaf_val_per_fold * max(1, min_valid_folds)

        for val_mask in self.fold_val_masks_:
            leaf_val_mask = idx_mask & val_mask
            n_leaf_val = int(leaf_val_mask.sum())
            if n_leaf_val < self.min_leaf_val_per_fold:
                invalid_folds += 1
                fold_metrics.append(np.nan)
                continue

            if u_oof is None:
                u = self._utility(y[leaf_val_mask], s_oof[leaf_val_mask])
            else:
                u = u_oof[leaf_val_mask]
            fold_metrics.append(_safe_sharpe(u))
            total_val_samples += n_leaf_val

        if invalid_folds > max_invalid_folds:
            return self.nan_score_floor

        if total_val_samples < min_total_samples:
            return self.nan_score_floor

        fm = np.array(fold_metrics, dtype=np.float32)
        fm = fm[np.isfinite(fm)]
        if fm.size == 0:
            return self.nan_score_floor
        score = stability_score_from_fold_metrics(
            fm,
            mode=self.stability_mode,
            disp_penalty=self.disp_penalty,
        )
        if not np.isfinite(score):
            return self.nan_score_floor
        return score

    def _stability_with_fold_info(
        self,
        idx_mask: np.ndarray,
        y: np.ndarray,
        s_oof: np.ndarray,
        u_oof: Optional[np.ndarray] = None,
    ) -> Tuple[float, List[float], float]:
        fold_metrics: List[float] = []
        total_val_samples = 0
        n_folds = len(self.fold_val_masks_)
        min_valid_folds = int(np.ceil(self.min_valid_fold_frac * n_folds))
        min_total_samples = self.min_total_leaf_val_samples
        if min_total_samples is None:
            min_total_samples = self.min_leaf_val_per_fold * max(1, min_valid_folds)

        valid_folds = 0
        for val_mask in self.fold_val_masks_:
            leaf_val_mask = idx_mask & val_mask
            n_leaf_val = int(leaf_val_mask.sum())
            if n_leaf_val < self.min_leaf_val_per_fold:
                continue

            if u_oof is None:
                u = self._utility(y[leaf_val_mask], s_oof[leaf_val_mask])
            else:
                u = u_oof[leaf_val_mask]
            fold_metrics.append(_safe_sharpe(u))
            total_val_samples += n_leaf_val
            valid_folds += 1

        valid_frac = valid_folds / max(1, n_folds)
        if valid_folds < min_valid_folds or total_val_samples < min_total_samples:
            return self.nan_score_floor, fold_metrics, valid_frac

        fm = np.array(fold_metrics, dtype=np.float32)
        fm = fm[np.isfinite(fm)]
        if fm.size == 0:
            return self.nan_score_floor, fold_metrics, valid_frac
        stability = stability_score_from_fold_metrics(
            fm,
            mode=self.stability_mode,
            disp_penalty=self.disp_penalty,
        )
        if not np.isfinite(stability):
            stability = self.nan_score_floor
        return stability, fold_metrics, valid_frac

    def _leaf_score_best_expert(
        self,
        idx_mask: np.ndarray,
        y: np.ndarray,
        preds_oof: Dict[str, np.ndarray],
    ) -> float:
        """
        Leaf score used in split search:
          max over experts of stability score on this leaf.
        """
        best = self.nan_score_floor
        for expert, s in preds_oof.items():
            u = self._utility_cache_.get(expert)
            sc = self._stability_for_expert_on_mask(idx_mask, y, s, u_oof=u)
            if sc > best:
                best = sc
        return best

    def _assign_leaf(
        self,
        leaf_id: int,
        idx_mask: np.ndarray,
        y: np.ndarray,
        preds_oof: Dict[str, np.ndarray],
    ) -> LeafAssignment:
        scores: Dict[str, float] = {}
        fold_scores: Dict[str, List[float]] = {}
        valid_folds: Dict[str, float] = {}

        # --- DEFENSIVE / ABSTAIN EXPERT ---
        # Add a virtual "ABSTAIN" expert that predicts 0.0 everywhere.
        # Its stability score is 0.0 (or slightly negative to penalize inaction).
        # This handles "Gray Zones" where no active expert has edge.
        abstain_score = 0.001  # Small positive score to beat pure noise
        scores["ABSTAIN_SPECIALIST"] = abstain_score
        fold_scores["ABSTAIN_SPECIALIST"] = [abstain_score] * len(self.fold_val_masks_)
        valid_folds["ABSTAIN_SPECIALIST"] = 1.0

        for expert, s in preds_oof.items():
            u = self._utility_cache_.get(expert)
            score, fold_metrics, valid_frac = self._stability_with_fold_info(idx_mask, y, s, u_oof=u)
            scores[expert] = score
            fold_scores[expert] = fold_metrics
            valid_folds[expert] = valid_frac

        best_expert = max(scores, key=scores.get)

        # --- ABSTAIN GATING: CROWD OUT WEAK EXPERTS ---
        # Require a minimum margin for activation.
        # Margin is data-driven: max(0.06, 60th percentile of leaf scores)
        if best_expert != "ABSTAIN_SPECIALIST":
            # Data-driven margin: q60 within leaf
            all_scores = [v for k, v in scores.items() if k != "ABSTAIN_SPECIALIST"]
            if all_scores:
                delta = np.nanpercentile(all_scores, 60)
            else:
                delta = 0.0

            # Combine with base threshold (slightly increased to 0.06)
            abstain_margin = max(0.06, delta)

            abstain_score = scores.get("ABSTAIN_SPECIALIST", 0.0)

            # The best expert must beat abstain + margin
            # Note: If delta (q60) is high, it means many experts are good, so barrier is higher?
            # User request: "delta = q^p(m)" where p=60% within leaf.
            # Interpretation: The margin itself IS the quantile.
            # So: score[best] > abstain + quantile(scores, 0.60) ??
            # Or: score[best] > abstain + max(0.06, quantile(scores, 0.60) - abstain)?
            # Let's interpret "delta = q60" as the margin value.
            # But q60 is an absolute score.
            # If q60 is e.g. 0.2, and abstain is 0.0, we need > 0.2.
            # This seems correct: "margin determined by the crowd".
            # However, usually margin is a delta.
            # If user means "delta should be the 60th percentile of the SPREAD vs abstain", that's different.
            # "delta = q^p(m)... e.g. p=60% within each leaf"
            # Let's assume absolute score quantile is the robust baseline.
            # So we check if best_score > max(abstain + 0.06, q60_score)

            barrier = max(abstain_score + 0.06, delta)

            if scores[best_expert] < barrier:
                best_expert = "ABSTAIN_SPECIALIST"

        best_score = float(scores[best_expert])

        pruned_scores = self._select_leaf_experts(
            scores,
            fold_scores,
            valid_folds,
            idx_mask,
            preds_oof,
        )
        if not pruned_scores:
            pruned_scores = {best_expert: best_score}

        weights = self._make_leaf_weights(pruned_scores) if self.soft_weights else {best_expert: 1.0}

        return LeafAssignment(
            leaf_id=leaf_id,
            expert_best=best_expert,
            expert_weights=weights,
            score_best=best_score,
            scores_by_expert=scores,
            n_samples=int(idx_mask.sum()),
            fold_scores_by_expert=fold_scores,
            valid_folds_by_expert=valid_folds,
        )

    def _select_leaf_experts(
        self,
        scores_by_expert: Dict[str, float],
        fold_scores_by_expert: Dict[str, List[float]],
        valid_folds_by_expert: Dict[str, float],
        idx_mask: np.ndarray,
        preds_oof: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        if not scores_by_expert:
            return {}

        best_expert = max(scores_by_expert, key=scores_by_expert.get)
        best_score = scores_by_expert[best_expert]
        candidates = sorted(scores_by_expert.items(), key=lambda kv: kv[1], reverse=True)

        n_before = len(candidates)
        kept: Dict[str, float] = {}

        def _min_fold_score(expert: str) -> float:
            folds = fold_scores_by_expert.get(expert, [])
            return min(folds) if folds else -np.inf

        def _corr(a: np.ndarray, b: np.ndarray) -> float:
            if a.size < 2 or b.size < 2:
                return 0.0
            if np.all(a == a[0]) or np.all(b == b[0]):
                return 0.0
            return float(np.corrcoef(a, b)[0, 1])

        n_leaf = int(idx_mask.sum())
        for expert, score in candidates:
            if score < best_score - self.expert_prune_gap:
                continue
            if valid_folds_by_expert.get(expert, 0.0) < self.min_valid_fold_frac:
                continue
            if _min_fold_score(expert) < self.expert_prune_worst_fold:
                continue
            if expert == "ABSTAIN_SPECIALIST":
                candidate_vals = np.zeros(n_leaf, dtype=np.float32)
            else:
                candidate_vals = preds_oof[expert][idx_mask]

            skip = False
            for kept_expert in kept:
                if kept_expert == "ABSTAIN_SPECIALIST":
                    kept_vals = np.zeros(n_leaf, dtype=np.float32)
                else:
                    kept_vals = preds_oof[kept_expert][idx_mask]
                
                corr = _corr(candidate_vals, kept_vals)
                if np.abs(corr) >= self.expert_prune_corr:
                    skip = True
                    break
            if skip:
                continue
            kept[expert] = score
            if len(kept) >= self.top_k_weights:
                break

        if not kept:
            kept[best_expert] = best_score

        if self.verbose:
            tprint_info(f"   ✂️ [RegimeTree] Pruned {n_before} -> {len(kept)} experts (gap={self.expert_prune_gap:.3f})")

        return kept

    def _make_leaf_weights(self, scores_by_expert: Dict[str, float]) -> Dict[str, float]:
        """
        Dynamic softmax temperature:
          - Use top_k experts.
          - Temperature is a deterministic function of the stability gap (top1 - top2).
          - If gap is small => blend (temp ~ 5).
          - If gap is large => decisive (temp ~ 12).
        """
        items = sorted(scores_by_expert.items(), key=lambda kv: kv[1], reverse=True)
        top = items[: max(1, self.top_k_weights)]

        vals = np.array([kv[1] for kv in top], dtype=np.float32)

        if not np.isfinite(vals).any():
            w = 1.0 / len(top)
            return {k: w for k, _ in top}

        vals = np.nan_to_num(vals, nan=-1e9, posinf=1e9, neginf=-1e9)

        temp_min, temp_max = 5.0, 12.0
        if len(vals) >= 2:
            gap = float(vals[0] - vals[1])
            g0, g1 = 0.05, 0.25
            if gap <= g0:
                temp = temp_min
            elif gap >= g1:
                temp = temp_max
            else:
                temp = temp_min + (temp_max - temp_min) * ((gap - g0) / (g1 - g0))
        else:
            temp = temp_min

        vals = vals - np.max(vals)
        probs = np.exp(temp * vals)
        probs = probs / (np.sum(probs) + 1e-12)

        if self.expert_prune_cum_weight > 0:
            full_vals = np.array([kv[1] for kv in items], dtype=np.float32)
            full_vals = np.nan_to_num(full_vals, nan=-1e9, posinf=1e9, neginf=-1e9)
            full_vals = full_vals - np.max(full_vals)
            full_probs = np.exp(temp * full_vals)
            full_probs = full_probs / (np.sum(full_probs) + 1e-12)
            cumulative = 0.0
            keep_count = 0
            for prob in full_probs:
                cumulative += float(prob)
                keep_count += 1
                if cumulative >= self.expert_prune_cum_weight:
                    break
            keep_count = max(keep_count, max(1, self.top_k_weights))
            kept_items = items[:keep_count]
            renorm = sum(full_probs[:keep_count]) or 1.0
            return {
                kept_items[i][0]: float(full_probs[i] / renorm)
                for i in range(len(kept_items))
            }

        return {top[i][0]: float(probs[i]) for i in range(len(top))}

    def _prune_leaf_experts(self) -> None:
        if self._y_cache_ is None or not self._preds_oof_cache_:
            return
        if self._leaf_ids_cache_ is None:
            return
        if len(self._preds_oof_cache_) <= self.top_k_weights:
            return
        tprint_info(f"🔍 [RegimeTree] Starting post-fit leaf expert pruning...")
        leaf_ids = self._leaf_ids_cache_
        for leaf_id, leaf in list(self.leaves_.items()):
            idx_mask = leaf_ids == leaf_id
            if not np.any(idx_mask):
                continue
            scores: Dict[str, float] = {}
            fold_scores: Dict[str, List[float]] = {}
            valid_folds: Dict[str, float] = {}
            
            # --- RE-ADD ABSTAIN SPECIALIST ---
            # Same logic as _assign_leaf: abstain gets a baseline score
            abstain_score = 0.001 
            scores["ABSTAIN_SPECIALIST"] = abstain_score
            fold_scores["ABSTAIN_SPECIALIST"] = [abstain_score] * len(self.fold_val_masks_)
            valid_folds["ABSTAIN_SPECIALIST"] = 1.0

            for expert, s in self._preds_oof_cache_.items():
                u = self._utility_cache_.get(expert)
                score, fold_metrics, valid_frac = self._stability_with_fold_info(
                    idx_mask,
                    self._y_cache_,
                    s,
                    u_oof=u,
                )
                scores[expert] = score
                fold_scores[expert] = fold_metrics
                valid_folds[expert] = valid_frac

            pruned_scores = self._select_leaf_experts(
                scores,
                fold_scores,
                valid_folds,
                idx_mask,
                self._preds_oof_cache_,
            )
            if not pruned_scores:
                continue

            weights = self._make_leaf_weights(pruned_scores) if self.soft_weights else {leaf.expert_best: 1.0}
            best_expert = max(weights, key=weights.get)
            leaf.expert_best = best_expert
            leaf.score_best = float(scores.get(best_expert, leaf.score_best))
            leaf.expert_weights = weights
            leaf.scores_by_expert = scores
            leaf.fold_scores_by_expert = fold_scores
            leaf.valid_folds_by_expert = valid_folds

    def merge_similar_leaves(self, eps: float = 0.2) -> None:
        n_before = len(self.leaves_)
        tprint_info(f"🤝 [RegimeTree] Attempting to merge similar leaves (eps={eps})...")

        def _l1_distance(a: Dict[str, float], b: Dict[str, float]) -> float:
            keys = set(a) | set(b)
            return float(sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys))

        def _merge_weights(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
            keys = set(a) | set(b)
            merged = {k: (a.get(k, 0.0) + b.get(k, 0.0)) / 2.0 for k in keys}
            total = sum(merged.values()) or 1.0
            return {k: v / total for k, v in merged.items()}

        def _merge_node(node: Optional[Node]) -> Optional[Node]:
            if node is None or node.is_leaf:
                return node
            node.left = _merge_node(node.left)
            node.right = _merge_node(node.right)
            if node.left is None or node.right is None:
                return node
            if not node.left.is_leaf or not node.right.is_leaf:
                return node
            left_leaf = self.leaves_.get(node.left.leaf_id)
            right_leaf = self.leaves_.get(node.right.leaf_id)
            if left_leaf is None or right_leaf is None:
                return node
            if _l1_distance(left_leaf.expert_weights, right_leaf.expert_weights) >= eps:
                return node

            merged_weights = _merge_weights(left_leaf.expert_weights, right_leaf.expert_weights)
            merged_scores = {**left_leaf.scores_by_expert}
            for k, v in right_leaf.scores_by_expert.items():
                merged_scores[k] = (merged_scores.get(k, v) + v) / 2.0
            best_expert = max(merged_weights, key=merged_weights.get)
            best_score = max(left_leaf.score_best, right_leaf.score_best)
            new_leaf_id = (max(self.leaves_.keys()) + 1) if self.leaves_ else 0
            merged_leaf = LeafAssignment(
                leaf_id=new_leaf_id,
                expert_best=best_expert,
                expert_weights=merged_weights,
                score_best=float(best_score),
                scores_by_expert=merged_scores,
                n_samples=left_leaf.n_samples + right_leaf.n_samples,
                fold_scores_by_expert={},
                valid_folds_by_expert={},
            )
            self.leaves_[new_leaf_id] = merged_leaf
            if node.left.leaf_id in self.leaves_:
                del self.leaves_[node.left.leaf_id]
            if node.right.leaf_id in self.leaves_:
                del self.leaves_[node.right.leaf_id]
            node.is_leaf = True
            node.left = None
            node.right = None
            node.leaf_id = new_leaf_id
            return node

        self.root_ = _merge_node(self.root_)
        n_after = len(self.leaves_)
        if n_after < n_before:
            tprint_success(f"🤝 [RegimeTree] Merged leaves: {n_before} -> {n_after}")

    def _find_best_split(
        self,
        idx_mask: np.ndarray,
        Z_np: np.ndarray,
        col_index: Dict[str, int],
        y: np.ndarray,
        preds_oof: Dict[str, np.ndarray],
    ) -> Optional[Tuple[str, float, np.ndarray, np.ndarray, float, float, float, float, str]]:
        """
        Greedy split: maximize stability gain.
        Returns (feature, threshold, left_mask, right_mask, gain, parent_score,
                 left_score, right_score, parent_best_expert)
        or None if no split meets criteria.
        """
        n_node = int(idx_mask.sum())
        if n_node < 2 * self._child_min_leaf_samples:
            self._record_split_diag("insufficient_node_samples")
            return None

        parent_score = self._leaf_score_best_expert(idx_mask, y, preds_oof)
        if not np.isfinite(parent_score):
            self._record_split_diag("nonfinite_parent_score")
            return None
        parent_best_expert = max(
            preds_oof.keys(),
            key=lambda k: self._stability_for_expert_on_mask(
                idx_mask,
                y,
                preds_oof[k],
                u_oof=self._utility_cache_.get(k),
            ),
        )

        best_gain = -np.inf
        best_split = None
        best_left_score = -np.inf
        best_right_score = -np.inf

        # Candidate thresholds: quantiles within node's samples (not global)
        adaptive_thresholds = max(3, min(self.n_thresholds, int(np.sqrt(n_node) / 5)))
        qs = np.linspace(0.1, 0.9, adaptive_thresholds, dtype=np.float32)

        for feat in self.features_:
            j = col_index[feat]
            x_node = Z_np[idx_mask, j]
            if np.nanstd(x_node) < 1e-12:
                self._record_split_diag("low_variance_feature")
                continue

            if self._Z_binned_ is not None:
                x_node_b = self._Z_binned_[idx_mask, j]
                uniq_bins = np.unique(x_node_b)
                if uniq_bins.size < 2:
                    self._record_split_diag("low_variance_feature")
                    continue

                if uniq_bins.size <= adaptive_thresholds:
                    candidate_bins = uniq_bins[:-1]
                else:
                    grid = np.linspace(0, uniq_bins.size - 2, adaptive_thresholds, dtype=int)
                    candidate_bins = uniq_bins[grid]

                edges = self._bin_edges_by_feature_.get(feat)
                if edges is not None and edges.size:
                    ks = candidate_bins.astype(int)
                    ks = ks[ks < edges.size]
                    thresholds = np.unique(edges[ks]) if ks.size else np.unique(np.nanquantile(x_node, qs))
                else:
                    thresholds = np.unique(np.nanquantile(x_node, qs))
            else:
                thresholds = np.unique(np.nanquantile(x_node, qs))
            # Evaluate each threshold
            x_full = Z_np[:, j]
            for thr in thresholds:
                cond = x_full <= thr
                left_mask = idx_mask & cond
                right_mask = idx_mask & ~cond

                if self._asset_id_codes_ is not None and self.min_asset_leaf_samples > 0:
                    parent_assets = np.unique(self._asset_id_codes_[idx_mask])
                    parent_assets = parent_assets[parent_assets >= 0]
                    if parent_assets.size:
                        left_counts = np.bincount(
                            self._asset_id_codes_[left_mask],
                            minlength=self._asset_id_max_ + 1,
                        )
                        right_counts = np.bincount(
                            self._asset_id_codes_[right_mask],
                            minlength=self._asset_id_max_ + 1,
                        )
                        if np.any(left_counts[parent_assets] < self.min_asset_leaf_samples) or np.any(
                            right_counts[parent_assets] < self.min_asset_leaf_samples
                        ):
                            self._record_split_diag("child_min_asset_samples_block")
                            continue

                if int(left_mask.sum()) < self._child_min_leaf_samples or int(right_mask.sum()) < self._child_min_leaf_samples:
                    self._record_split_diag("child_min_samples_block")
                    continue

                left_score = self._leaf_score_best_expert(left_mask, y, preds_oof)
                right_score = self._leaf_score_best_expert(right_mask, y, preds_oof)

                if not np.isfinite(left_score) or not np.isfinite(right_score):
                    self._record_split_diag("nonfinite_child_score")
                    continue

                n_left = int(left_mask.sum())
                n_right = int(right_mask.sum())
                weighted_child = (n_left * left_score + n_right * right_score) / max(1, n_node)

                p = n_left / max(1, n_node)
                if p <= 0.0 or p >= 1.0:
                    imbalance = 1.0
                else:
                    split_entropy = -p * np.log(p) - (1.0 - p) * np.log(1.0 - p)
                    imbalance = 1.0 - (split_entropy / np.log(2.0))

                gain = weighted_child - parent_score - self.split_imbalance_penalty * imbalance
                if gain > best_gain:
                    best_gain = float(gain)
                    best_left_score = float(left_score)
                    best_right_score = float(right_score)
                    best_split = (feat, float(thr), left_mask, right_mask, best_gain)

        if best_split is None:
            self._record_split_diag("no_valid_split")
            return None

        adaptive_min_gain = self.min_stability_gain
        if abs(parent_score) < self.min_gain_relax_parent_abs:
            adaptive_min_gain = min(adaptive_min_gain, self.min_gain_relaxed)

        # Minimum gain pruning (avoid micro-splits from noise)
        if best_gain < adaptive_min_gain:
            self._record_split_diag("gain_below_min")
            feat, thr, lm, rm, _ = best_split
            if self.verbose:
                tprint_info(
                    "   ⚠️ [RegimeTree] Best split below gain threshold: "
                    f"{feat} <= {thr:.4f} | gain={best_gain:.6f} < {adaptive_min_gain:.6f}, "
                    f"parent={parent_score:.6f}, left={best_left_score:.6f} (n={int(lm.sum())}), "
                    f"right={best_right_score:.6f} (n={int(rm.sum())})"
                )
            
            # --- DEBUG: Inspect why the split was invalid (check fold distribution) ---
            if self.verbose:
                if best_left_score <= self.nan_score_floor + 1.0 or best_right_score <= self.nan_score_floor + 1.0:
                    tprint_info("   🕵️ [RegimeTree] Debugging invalid split fold distribution:")

                    for label, mask in [("Left", lm), ("Right", rm)]:
                        is_invalid = (best_left_score <= self.nan_score_floor + 1.0) if label == "Left" else (best_right_score <= self.nan_score_floor + 1.0)
                        if is_invalid:
                            counts = []
                            valid_cnt = 0
                            for val_mask in self.fold_val_masks_:
                                c = int((mask & val_mask).sum())
                                counts.append(c)
                                if c >= self.min_leaf_val_per_fold:
                                    valid_cnt += 1
                            tprint_info(
                                f"      • {label} child ({int(mask.sum())} samples): Valid folds={valid_cnt}/{len(self.fold_val_masks_)} "
                                f"(req {int(np.ceil(self.min_valid_fold_frac * len(self.fold_val_masks_)))}), Counts={counts}"
                            )
            return None

        if self.verbose:
            feat, thr, lm, rm, gain = best_split
            print(
                f"[RegimeTree] split: {feat} <= {thr:.4g} | "
                f"gain={gain:.4g} | left={int(lm.sum())} right={int(rm.sum())}"
            )

        feat, thr, left_mask, right_mask, gain = best_split
        return (
            feat,
            thr,
            left_mask,
            right_mask,
            gain,
            float(parent_score),
            float(best_left_score),
            float(best_right_score),
            str(parent_best_expert),
        )

    def _record_split_diag(self, key: str) -> None:
        if self.split_diagnostics is None:
            self.split_diagnostics = Counter()
        self.split_diagnostics[key] += 1

    def _log_leaf_summaries(self) -> None:
        if not self.leaves_:
            return
        tprint_info("   🌿 [RegimeTree] Leaf summaries:")
        for leaf_id in sorted(self.leaves_):
            leaf = self.leaves_[leaf_id]
            weights = leaf.expert_weights or {}
            top_weights = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:3]
            weight_str = ", ".join(f"{k}:{v:.2f}" for k, v in top_weights)
            probs = np.array([v for v in weights.values() if v > 0], dtype=np.float32)
            if probs.size:
                entropy = float(-(probs * np.log(probs + 1e-12)).sum())
            else:
                entropy = 0.0
            scores = leaf.scores_by_expert or {}
            if scores:
                sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
                best_score = sorted_scores[0][1]
                third_score = sorted_scores[2][1] if len(sorted_scores) > 2 else sorted_scores[-1][1]
                score_spread = float(best_score - third_score)
            else:
                score_spread = 0.0
            tprint_info(
                f"      • Leaf {leaf_id}: n={leaf.n_samples}, best={leaf.expert_best}, "
                f"score={leaf.score_best:.4f}, weights=[{weight_str}], H={entropy:.3f}, "
                f"Δscore_top3={score_spread:.5f}"
            )

    def get_split_diagnostics(self) -> Dict[str, int]:
        return dict(self.split_diagnostics or {})

    def _traverse_to_leaf_id(self, row: np.ndarray, node: Node) -> int:
        """
        Traverse a single row to leaf_id.
        """
        while not node.is_leaf:
            if node.feature is None or node.threshold is None:
                raise RuntimeError("Malformed internal node.")

            if node.feature_idx is not None:
                j = int(node.feature_idx)
            else:
                j = int(self._feature_to_index_.get(node.feature, self.features_.index(node.feature)))
            if row[j] <= node.threshold:
                node = node.left  # type: ignore[assignment]
            else:
                node = node.right  # type: ignore[assignment]
            if node is None:
                raise RuntimeError("Malformed tree child pointer.")

        if node.leaf_id is None:
            raise RuntimeError("Malformed leaf node.")
        return node.leaf_id

    def prune(self, alpha: float = 0.03) -> None:
        """
        Cost-complexity-like pruning.
        Collapse an internal node into a leaf if the split does not improve
        stability enough to justify complexity.

        Rule:
          prune if children_weighted_score <= parent_best_score + alpha
        """
        if self.root_ is None:
            return
        if not isinstance(self.node_stats_, dict):
            raise RuntimeError("node_stats_ must be a dict[node_id -> NodeStats].")

        def _make_parent_leaf_assignment(node: Node) -> LeafAssignment:
            st = self.node_stats_.get(node.node_id)
            if st is None:
                raise RuntimeError(f"Missing NodeStats for node_id={node.node_id}")
            best_expert = st.parent_best_expert
            weights = {best_expert: 1.0} if not self.soft_weights else {best_expert: 1.0}

            new_leaf_id = (max(self.leaves_.keys()) + 1) if self.leaves_ else 0
            return LeafAssignment(
                leaf_id=new_leaf_id,
                expert_best=best_expert,
                expert_weights=weights,
                score_best=float(st.parent_best_score),
                scores_by_expert={best_expert: float(st.parent_best_score)},
                n_samples=int(st.n_left + st.n_right),
            )

        def _prune_recursive(node: Optional[Node]) -> Tuple[Optional[Node], float]:
            if node is None:
                return node, -np.inf

            if node.is_leaf:
                leaf_info = self.leaves_[node.leaf_id]  # type: ignore[index]
                return node, float(leaf_info.score_best)

            node.left, left_score = _prune_recursive(node.left)
            node.right, right_score = _prune_recursive(node.right)

            st = self.node_stats_.get(node.node_id)
            if st is None:
                return node, float((left_score + right_score) / 2.0)

            children_weighted = float(st.children_weighted_score)
            parent_best = float(st.parent_best_score)

            if children_weighted <= parent_best + float(alpha):
                if self.verbose:
                    print(f"[RegimeTree] Pruning node {node.node_id} at depth {node.depth}")

                parent_leaf = _make_parent_leaf_assignment(node)
                node.is_leaf = True
                node.feature = None
                node.threshold = None
                node.left = None
                node.right = None
                node.leaf_id = parent_leaf.leaf_id

                self.leaves_[parent_leaf.leaf_id] = parent_leaf
                return node, float(parent_leaf.score_best)

            return node, children_weighted

        self.root_, _ = _prune_recursive(self.root_)

    def _find_any_leaf(self, node: Optional[Node]) -> Optional[int]:
        """
        Utility for pruning: find any existing leaf_id beneath node.
        """
        if node is None:
            return None
        if node.is_leaf:
            return node.leaf_id
        left = self._find_any_leaf(node.left)
        if left is not None:
            return left
        return self._find_any_leaf(node.right)

    # -------------------------
    # Serialization
    # -------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Serialize tree for checkpointing."""
        return {
            "params": {
                "max_depth": self.max_depth,
                "min_leaf_samples": self.min_leaf_samples,
                "min_leaf_val_per_fold": self.min_leaf_val_per_fold,
                "n_thresholds": self.n_thresholds,
                "stability_mode": self.stability_mode,
                "disp_penalty": self.disp_penalty,
                "utility_transform": self.utility_transform,
                "soft_weights": self.soft_weights,
                "top_k_weights": self.top_k_weights,
                "weight_temperature": self.weight_temperature,
                "min_stability_gain": self.min_stability_gain,
                "min_valid_fold_frac": self.min_valid_fold_frac,
                "expert_prune_gap": self.expert_prune_gap,
                "expert_prune_cum_weight": self.expert_prune_cum_weight,
                "expert_prune_worst_fold": self.expert_prune_worst_fold,
                "expert_prune_corr": self.expert_prune_corr,
                "min_gain_relax_parent_abs": self.min_gain_relax_parent_abs,
                "min_gain_relaxed": self.min_gain_relaxed,
                "min_asset_leaf_samples": self.min_asset_leaf_samples,
                "zscore_mode": self.zscore_mode,
                "zscore_window": self.zscore_window,
                "split_imbalance_penalty": self.split_imbalance_penalty,
                "use_binned_splits": self.use_binned_splits,
                "n_feature_bins": self.n_feature_bins,
                "use_numba_inference": self.use_numba_inference,
                "verbose": self.verbose,
            },
            "_effective_min_leaf_samples": self._effective_min_leaf_samples,
            "root": self.root_.to_dict() if self.root_ else None,
            "leaves": {str(k): v.to_dict() for k, v in self.leaves_.items()},
            "features": self.features_,
            "experts": self.experts_,
            "feature_importances": self.feature_importances_,
            "node_stats": {str(k): v.to_dict() for k, v in self.node_stats_.items()},
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StabilityRegimeTree":
        """Deserialize tree from checkpoint data."""
        tree = cls(**d["params"])
        tree._effective_min_leaf_samples = d.get("_effective_min_leaf_samples", 0)
        if d.get("root"):
            tree.root_ = Node.from_dict(d["root"])
        if "leaves" in d:
            tree.leaves_ = {int(k): LeafAssignment.from_dict(v) for k, v in d["leaves"].items()}
        tree.features_ = d.get("features", [])
        tree.experts_ = d.get("experts", [])
        tree.feature_importances_ = d.get("feature_importances", {})
        if "node_stats" in d:
            tree.node_stats_ = {int(k): NodeStats.from_dict(v) for k, v in d["node_stats"].items()}
        return tree


# ============================================================
# Example usage (toy)
# ============================================================
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 20_000

    # State features Z (time-ordered)
    Z = pd.DataFrame(
        {
            "vol_shock": rng.normal(size=n),
            "liq_shock": rng.normal(size=n),
            "dispersion": rng.normal(size=n),
            "corr_shock": rng.normal(size=n),
        }
    )

    # Toy outcome
    y = rng.normal(scale=0.01, size=n)

    # Three specialists each "wins" in its own state
    preds_oof = {
        "LIQUIDITY_SPECIALIST": 2.0 * (Z["liq_shock"].to_numpy() < -0.5).astype(float) + 0.2 * rng.normal(size=n),
        "VOL_INNOVATION_SPECIALIST": 2.0 * (Z["vol_shock"].to_numpy() > 0.8).astype(float) + 0.2 * rng.normal(size=n),
        "DISPERSION_SPECIALIST": 2.0 * (Z["dispersion"].to_numpy() > 0.6).astype(float) + 0.2 * rng.normal(size=n),
    }

    # Make y respond to the right specialist in its regime (toy)
    y += 0.02 * np.tanh(preds_oof["LIQUIDITY_SPECIALIST"]) * (Z["liq_shock"].to_numpy() < -0.5)
    y += 0.02 * np.tanh(preds_oof["VOL_INNOVATION_SPECIALIST"]) * (Z["vol_shock"].to_numpy() > 0.8)
    y += 0.02 * np.tanh(preds_oof["DISPERSION_SPECIALIST"]) * (Z["dispersion"].to_numpy() > 0.6)

    # Purged K-folds (set purge/embargo to your label horizon in bars)
    folds = make_purged_kfold_folds(Z.index, n_folds=8, purge=0, embargo=0)

    tree = StabilityRegimeTree(
        max_depth=2,
        min_leaf_samples=1500,
        min_leaf_val_per_fold=200,
        stability_mode="min_minus_iqr",
        disp_penalty=0.5,
        utility_transform="tanh",
        soft_weights=True,
        top_k_weights=2,
        weight_temperature=8.0,
        min_stability_gain=0.02,
        verbose=True,
    ).fit(Z=Z, preds_oof=preds_oof, y=y, folds=folds)

    routed = tree.route(Z_new=Z, preds_by_expert=preds_oof)
    leaf_ids = tree.predict_leaf_ids(Z)

    print("n_leaves:", len(tree.leaves_))
    print("feature_importances_:", tree.feature_importances_)
    for leaf_id, leaf in tree.leaves_.items():
        print(
            f"leaf={leaf_id} n={leaf.n_samples} best={leaf.expert_best} "
            f"score={leaf.score_best:.4f} weights={leaf.expert_weights}"
        )
    print("routed sample:", routed[:5], "leaf ids:", leaf_ids[:5])
