"""
Grid Utilities for HPO

Centralized creation of coarse and fine parameter grids used across HPO
to avoid duplication and ensure consistent behavior.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import itertools
import numpy as np


def build_coarse_grid_from_search_space(search_space: Dict[str, Any], grid_points: int) -> List[Dict[str, Any]]:
    """Create a coarse parameter grid list from a search space.

    Each parameter produces a list of candidate values; we then take the Cartesian
    product to return a list of parameter dictionaries.
    """
    try:
        value_lists: List[List[Tuple[str, Any]]] = []
        for name, cfg in search_space.items():
            if not isinstance(cfg, dict):
                # Legacy tuple(low, high)
                if isinstance(cfg, tuple) and len(cfg) == 2:
                    low, high = cfg
                    vals = np.linspace(low, high, num=max(2, grid_points)).tolist()
                    value_lists.append([(name, v) for v in vals])
                continue

            typ = cfg.get('type', 'float')
            if typ == 'float':
                low, high = cfg['low'], cfg['high']
                vals = np.linspace(low, high, num=max(2, grid_points)).tolist()
                value_lists.append([(name, v) for v in vals])
            elif typ == 'int':
                low, high = cfg['low'], cfg['high']
                if high == low:
                    vals = [low]
                else:
                    pts = np.linspace(low, high, num=max(2, grid_points))
                    vals = sorted({int(round(v)) for v in pts})
                value_lists.append([(name, v) for v in vals])
            elif typ == 'categorical':
                vals = cfg.get('choices', [])
                value_lists.append([(name, v) for v in vals])

        if not value_lists:
            return []

        combinations = list(itertools.product(*value_lists))
        return [dict(combo) for combo in combinations]
    except Exception:
        return []


def build_fine_grid_around_best(search_space: Dict[str, Any], best_params: Dict[str, Any],
                                grid_points: int) -> List[Dict[str, Any]]:
    """Create a fine parameter grid around the best parameters discovered so far.

    For floats: +/- 20% of the original range; for ints: +/- 2; categorical: keep choices.
    """
    combos: List[List[Tuple[str, Any]]] = []
    for name, cfg in search_space.items():
        if name not in best_params:
            continue
        best_val = best_params[name]
        if isinstance(cfg, dict):
            typ = cfg.get('type', 'float')
            if typ == 'float':
                low, high = cfg['low'], cfg['high']
                rng = high - low
                fine_rng = rng * 0.2
                fine_min = max(low, best_val - fine_rng)
                fine_max = min(high, best_val + fine_rng)
                if cfg.get('log', False) and fine_min > 0 and fine_max > fine_min:
                    vals = np.logspace(np.log10(fine_min), np.log10(fine_max), grid_points)
                else:
                    vals = np.linspace(fine_min, fine_max, grid_points)
                combos.append([(name, v) for v in vals])
            elif typ == 'int':
                low, high = cfg['low'], cfg['high']
                fine_min = max(low, int(best_val) - 2)
                fine_max = min(high, int(best_val) + 2)
                vals = list(range(fine_min, fine_max + 1))
                combos.append([(name, v) for v in vals])
            elif typ == 'categorical':
                vals = cfg.get('choices', [])
                combos.append([(name, v) for v in vals])
        else:
            # Legacy tuple
            if isinstance(cfg, tuple) and len(cfg) == 2:
                low, high = cfg
                rng = high - low
                fine_rng = rng * 0.2
                fine_min = max(low, best_val - fine_rng)
                fine_max = min(high, best_val + fine_rng)
                vals = np.linspace(fine_min, fine_max, grid_points)
                combos.append([(name, v) for v in vals])

    if not combos:
        return []
    return [dict(c) for c in itertools.product(*combos)]


__all__ = [
    'build_coarse_grid_from_search_space',
    'build_fine_grid_around_best',
]

