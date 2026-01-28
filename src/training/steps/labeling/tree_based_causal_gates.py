from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


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
        min_leaf_samples: int = 2000,
        min_leaf_val_per_fold: int = 200,
        n_thresholds: int = 9,  # deciles by default
        stability_mode: str = "min_minus_iqr",
        disp_penalty: float = 0.5,
        utility_transform: str = "tanh",  # tanh / clip / identity
        soft_weights: bool = True,
        top_k_weights: int = 2,
        weight_temperature: float = 8.0,
        min_stability_gain: float = 0.02,
        verbose: bool = False,
    ):
        self.max_depth = int(max_depth)
        self.min_leaf_samples = int(min_leaf_samples)
        self.min_leaf_val_per_fold = int(min_leaf_val_per_fold)
        self.n_thresholds = int(n_thresholds)
        self.stability_mode = str(stability_mode)
        self.disp_penalty = float(disp_penalty)
        self.utility_transform = str(utility_transform)
        self.soft_weights = bool(soft_weights)
        self.top_k_weights = int(top_k_weights)
        self.weight_temperature = float(weight_temperature)
        self.min_stability_gain = float(min_stability_gain)
        self.verbose = bool(verbose)

        self.root_: Optional[Node] = None
        self.leaves_: Dict[int, LeafAssignment] = {}
        self.features_: List[str] = []
        self.experts_: List[str] = []
        self.fold_val_masks_: List[np.ndarray] = []
        self.feature_importances_: Dict[str, float] = {}

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
        Z = self._validate_Z(Z)
        y = self._validate_y(y, len(Z))
        preds_oof = self._validate_preds(preds_oof, len(Z))

        self.features_ = list(Z.columns)
        self.experts_ = list(preds_oof.keys())
        self.feature_importances_ = {f: 0.0 for f in self.features_}

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
            if depth >= self.max_depth or n_node < 2 * self.min_leaf_samples:
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

            feat, thr, left_mask, right_mask, gain = best
            self.feature_importances_[feat] += float(gain)

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
    ) -> np.ndarray:
        """
        Route and combine expert predictions for new data.
        If soft_weights=True -> weighted sum per leaf.
        Else -> hard route to leaf's best expert.
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
        out = np.zeros(n, dtype=float)

        for i, leaf_id in enumerate(leaf_ids):
            leaf = self.leaves_[leaf_id]
            if not self.soft_weights:
                out[i] = float(preds_by_expert[leaf.expert_best][i])
            else:
                s = 0.0
                for e, w in leaf.expert_weights.items():
                    s += float(w) * float(preds_by_expert[e][i])
                out[i] = s

        return out

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

        for val_mask in self.fold_val_masks_:
            leaf_val_mask = idx_mask & val_mask
            n_leaf_val = int(leaf_val_mask.sum())
            if n_leaf_val < self.min_leaf_val_per_fold:
                fold_metrics.append(np.nan)
                continue

            u = self._utility(y[leaf_val_mask], s_oof[leaf_val_mask])
            fold_metrics.append(_safe_sharpe(u))

        fm = np.array(fold_metrics, dtype=float)
        return stability_score_from_fold_metrics(
            fm,
            mode=self.stability_mode,
            disp_penalty=self.disp_penalty,
        )

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
        for expert, s in preds_oof.items():
            scores[expert] = self._stability_for_expert_on_mask(idx_mask, y, s)

        best_expert = max(scores, key=scores.get)
        best_score = float(scores[best_expert])

        weights = self._make_leaf_weights(scores) if self.soft_weights else {best_expert: 1.0}

        return LeafAssignment(
            leaf_id=leaf_id,
            expert_best=best_expert,
            expert_weights=weights,
            score_best=best_score,
            scores_by_expert=scores,
            n_samples=int(idx_mask.sum()),
        )

    def _make_leaf_weights(self, scores_by_expert: Dict[str, float]) -> Dict[str, float]:
        """
        Soft weights from stability scores:
          - select top_k experts
          - softmax with temperature
        """
        items = sorted(scores_by_expert.items(), key=lambda kv: kv[1], reverse=True)
        top = items[: max(1, self.top_k_weights)]

        scores = np.array([kv[1] for kv in top], dtype=float)
        if not np.isfinite(scores).any():
            w = 1.0 / len(top)
            return {k: w for k, _ in top}

        scores = np.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
        scores = scores - np.max(scores)
        probs = np.exp(self.weight_temperature * scores)
        probs = probs / (np.sum(probs) + 1e-12)

        return {top[i][0]: float(probs[i]) for i in range(len(top))}

    def _find_best_split(
        self,
        idx_mask: np.ndarray,
        Z_np: np.ndarray,
        col_index: Dict[str, int],
        y: np.ndarray,
        preds_oof: Dict[str, np.ndarray],
    ) -> Optional[Tuple[str, float, np.ndarray, np.ndarray, float]]:
        """
        Greedy split: maximize stability gain.
        Returns (feature, threshold, left_mask, right_mask, gain)
        or None if no split meets criteria.
        """
        n_node = int(idx_mask.sum())
        if n_node < 2 * self.min_leaf_samples:
            return None

        parent_score = self._leaf_score_best_expert(idx_mask, y, preds_oof)
        if not np.isfinite(parent_score):
            return None

        best_gain = -np.inf
        best_split = None

        # Candidate thresholds: quantiles within node's samples (not global)
        qs = np.linspace(0.1, 0.9, self.n_thresholds)

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

        return best_split

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
