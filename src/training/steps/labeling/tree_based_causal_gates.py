from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.tprint import tprint_info, tprint_success, tprint_warning


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
    fm = np.asarray(fm, dtype=float)
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


@dataclass
class Node:
    node_id: int
    depth: int
    is_leaf: bool
    leaf_id: Optional[int] = None

    feature: Optional[str] = None
    threshold: Optional[float] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None


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
        min_leaf_samples: float | int = 0.05,
        min_leaf_val_per_fold: int = 200,
        n_thresholds: int = 9,  # deciles by default
        stability_mode: str = "min_minus_iqr",
        disp_penalty: float = 0.5,
        utility_transform: str = "tanh",  # tanh / clip / identity
        soft_weights: bool = True,
        top_k_weights: int = 2,
        weight_temperature: float = 8.0,
        min_stability_gain: float = 0.02,
        min_valid_fold_frac: float = 0.8,
        min_total_leaf_val_samples: Optional[int] = None,
        expert_prune_gap: float = 0.1,
        expert_prune_cum_weight: float = 0.9,
        expert_prune_worst_fold: float = -0.25,
        expert_prune_corr: float = 0.9,
        merge_leaf_l1_eps: float = 0.2,
        verbose: bool = False,
    ):
        self.max_depth = int(max_depth)
        self.min_leaf_samples = min_leaf_samples
        self._effective_min_leaf_samples = 0 # Calculated in fit
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
        self.verbose = bool(verbose)

        self.root_: Optional[Node] = None
        self.leaves_: Dict[int, LeafAssignment] = {}
        self.features_: List[str] = []
        self.experts_: List[str] = []
        self.fold_val_masks_: List[np.ndarray] = []
        self.feature_importances_: Dict[str, float] = {}
        self.node_stats_: Dict[int, NodeStats] = {}
        self._preds_oof_cache_: Dict[str, np.ndarray] = {}
        self._y_cache_: Optional[np.ndarray] = None
        self._leaf_ids_cache_: Optional[np.ndarray] = None

    # -------------------------
    # Public API
    # -------------------------
    def fit(
        self,
        Z: pd.DataFrame,
        preds_oof: Dict[str, np.ndarray],
        y: np.ndarray,
        folds: List[Tuple[np.ndarray, np.ndarray]],
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
        
        tprint_info(f"   🌲 Effective min_leaf_samples: {self._effective_min_leaf_samples} ({self.min_leaf_samples})")

        self.features_ = list(Z.columns)
        self.experts_ = list(preds_oof.keys())
        self.feature_importances_ = {f: 0.0 for f in self.features_}
        self._preds_oof_cache_ = preds_oof
        self._y_cache_ = y

        # Precompute fold validation masks once (fast intersections later)
        self.fold_val_masks_ = []
        n = len(Z)
        for _, val_idx in folds:
            m = np.zeros(n, dtype=bool)
            m[val_idx] = True
            self.fold_val_masks_.append(m)

        # Numpy view for fast thresholding
        Z_np = Z.to_numpy(dtype=float)
        col_index = {c: j for j, c in enumerate(self.features_)}

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
            if depth >= self.max_depth or n_node < 2 * self._effective_min_leaf_samples:
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
                threshold=float(thr),
            )
            node.left = build_node(left_mask, depth + 1)
            node.right = build_node(right_mask, depth + 1)
            return node

        self.root_ = build_node(root_mask, 0)
        self._leaf_ids_cache_ = self.predict_leaf_ids(Z)
        self._prune_leaf_experts()
        if self.merge_leaf_l1_eps > 0:
            self.merge_similar_leaves(self.merge_leaf_l1_eps)
        tprint_success(f"✅ [RegimeTree] Fit complete: {len(self.leaves_)} leaves, {len(self.features_)} features")
        return self

    def predict_leaf_ids(self, Z_new: pd.DataFrame) -> np.ndarray:
        if self.root_ is None:
            raise RuntimeError("Tree is not fitted.")
        Z_new = self._validate_Z(Z_new)
        Z_new = Z_new[self.features_]
        X = Z_new.to_numpy(dtype=float)

        out = np.empty(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            out[i] = self._traverse_to_leaf_id(X[i], self.root_)
        return out

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

        leaf_ids = self.predict_leaf_ids(Z_new)
        signals = np.zeros(n, dtype=float)
        disagreements = np.zeros(n, dtype=float)
        entropies = np.zeros(n, dtype=float)
        best_scores = np.zeros(n, dtype=float)
        best_experts: List[str] = []

        for i, leaf_id in enumerate(leaf_ids):
            leaf = self.leaves_[leaf_id]
            current_signal = 0.0
            active_experts = []

            for e, w in leaf.expert_weights.items():
                if e == "ABSTAIN_SPECIALIST":
                    pred = 0.0
                else:
                    pred = float(preds_by_expert[e][i])
                current_signal += w * pred
                active_experts.append((pred, w))

            signals[i] = current_signal

            weighted_var = 0.0
            for pred, w in active_experts:
                weighted_var += w * (pred - current_signal) ** 2
            disagreements[i] = np.sqrt(max(0.0, weighted_var))

            weights_arr = np.array(list(leaf.expert_weights.values()), dtype=float)
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

        return Z

    def _validate_y(self, y: np.ndarray, n: int) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if y.ndim != 1 or y.size != n:
            raise ValueError("y must be 1D and aligned with Z.")
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        return y

    def _validate_preds(self, preds: Dict[str, np.ndarray], n: int) -> Dict[str, np.ndarray]:
        if not preds:
            raise ValueError("preds_oof is empty.")
        out: Dict[str, np.ndarray] = {}
        for k, v in preds.items():
            arr = np.asarray(v, dtype=float)
            if arr.ndim != 1 or arr.size != n:
                raise ValueError(f"preds_oof[{k}] must be 1D and aligned with Z.")
            out[k] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
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

            u = self._utility(y[leaf_val_mask], s_oof[leaf_val_mask])
            fold_metrics.append(_safe_sharpe(u))
            total_val_samples += n_leaf_val

        if invalid_folds > max_invalid_folds:
            return -np.inf

        if total_val_samples < min_total_samples:
            return -np.inf

        fm = np.array(fold_metrics, dtype=float)
        return stability_score_from_fold_metrics(
            fm,
            mode=self.stability_mode,
            disp_penalty=self.disp_penalty,
        )

    def _stability_with_fold_info(
        self,
        idx_mask: np.ndarray,
        y: np.ndarray,
        s_oof: np.ndarray,
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
            u = self._utility(y[leaf_val_mask], s_oof[leaf_val_mask])
            fold_metrics.append(_safe_sharpe(u))
            total_val_samples += n_leaf_val
            valid_folds += 1

        valid_frac = valid_folds / max(1, n_folds)
        if valid_folds < min_valid_folds or total_val_samples < min_total_samples:
            return -np.inf, fold_metrics, valid_frac

        fm = np.array(fold_metrics, dtype=float)
        stability = stability_score_from_fold_metrics(
            fm,
            mode=self.stability_mode,
            disp_penalty=self.disp_penalty,
        )
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
        best = -np.inf
        for s in preds_oof.values():
            sc = self._stability_for_expert_on_mask(idx_mask, y, s)
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
            score, fold_metrics, valid_frac = self._stability_with_fold_info(idx_mask, y, s)
            scores[expert] = score
            fold_scores[expert] = fold_metrics
            valid_folds[expert] = valid_frac

        best_expert = max(scores, key=scores.get)
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

        for expert, score in candidates:
            if score < best_score - self.expert_prune_gap:
                continue
            if valid_folds_by_expert.get(expert, 0.0) < self.min_valid_fold_frac:
                continue
            if _min_fold_score(expert) < self.expert_prune_worst_fold:
                continue
            if kept:
                candidate_vals = preds_oof[expert][idx_mask]
                skip = False
                for kept_expert in kept:
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

        vals = np.array([kv[1] for kv in top], dtype=float)

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
            full_vals = np.array([kv[1] for kv in items], dtype=float)
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
            for expert, s in self._preds_oof_cache_.items():
                score, fold_metrics, valid_frac = self._stability_with_fold_info(
                    idx_mask, self._y_cache_, s
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
        if n_node < 2 * self.min_leaf_samples:
            return None

        parent_score = self._leaf_score_best_expert(idx_mask, y, preds_oof)
        if not np.isfinite(parent_score):
            return None
        parent_best_expert = max(
            preds_oof.keys(),
            key=lambda k: self._stability_for_expert_on_mask(idx_mask, y, preds_oof[k]),
        )

        best_gain = -np.inf
        best_split = None
        best_left_score = -np.inf
        best_right_score = -np.inf

        # Candidate thresholds: quantiles within node's samples (not global)
        adaptive_thresholds = max(3, min(self.n_thresholds, int(np.sqrt(n_node) / 5)))
        qs = np.linspace(0.1, 0.9, adaptive_thresholds)

        for feat in self.features_:
            j = col_index[feat]
            x_node = Z_np[idx_mask, j]
            if np.nanstd(x_node) < 1e-12:
                continue

            thresholds = np.unique(np.nanquantile(x_node, qs))
            # Evaluate each threshold
            x_full = Z_np[:, j]
            for thr in thresholds:
                cond = x_full <= thr
                left_mask = idx_mask & cond
                right_mask = idx_mask & ~cond

                if int(left_mask.sum()) < self.min_leaf_samples or int(right_mask.sum()) < self.min_leaf_samples:
                    continue

                left_score = self._leaf_score_best_expert(left_mask, y, preds_oof)
                right_score = self._leaf_score_best_expert(right_mask, y, preds_oof)

                if not np.isfinite(left_score) or not np.isfinite(right_score):
                    continue

                gain = (left_score + right_score) - parent_score
                if gain > best_gain:
                    best_gain = float(gain)
                    best_left_score = float(left_score)
                    best_right_score = float(right_score)
                    best_split = (feat, float(thr), left_mask, right_mask, best_gain)

        if best_split is None:
            return None

        # Minimum gain pruning (avoid micro-splits from noise)
        if best_gain < self.min_stability_gain:
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

    def _traverse_to_leaf_id(self, row: np.ndarray, node: Node) -> int:
        """
        Traverse a single row to leaf_id.
        """
        while not node.is_leaf:
            if node.feature is None or node.threshold is None:
                raise RuntimeError("Malformed internal node.")

            j = self.features_.index(node.feature)
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
