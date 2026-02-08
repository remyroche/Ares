"""
Grid Search Constraint Analysis

Original grid: 10 × 10 × 7 = 700 configurations

With constraints:
1. TP > SL (asymmetric edge)
2. Trail < TP (can't trail before activation)
3. TP/SL ratio ∈ [1.2, 5.0] (reasonable asymmetry)
4. Trail/TP ratio ∈ [0.3, 0.9] (reasonable trail distance)
5. Skip tight SL + wide TP combos (SL<0.5 AND TP>2.0)

Estimated viable configs: ~200-300 (reduction of 60-70%)

Example valid combinations:
- TP=1.0, SL=0.5, Trail=0.5 ✓ (ratio=2.0, trail=0.5×TP)
- TP=1.5, SL=0.8, Trail=0.7 ✓ (ratio=1.88, trail=0.47×TP)
- TP=2.0, SL=1.0, Trail=1.0 ✓ (ratio=2.0, trail=0.5×TP)

Example invalid combinations:
- TP=1.0, SL=1.0, Trail=0.5 ✗ (TP ≤ SL)
- TP=1.0, SL=0.5, Trail=1.0 ✗ (Trail ≥ TP)
- TP=3.0, SL=0.3, Trail=1.0 ✗ (ratio=10.0 > 5.0)
- TP=1.0, SL=0.5, Trail=0.2 ✗ (trail=0.2×TP < 0.3)
- TP=2.5, SL=0.3, Trail=1.0 ✗ (SL<0.5 AND TP>2.0)

Time savings: ~60-70% reduction in grid search time
"""

print(__doc__)
