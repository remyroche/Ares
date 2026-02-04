import unittest
import numpy as np
from extreme_price_movements.optimization import composite_score_with_constraints

class TestOptimization(unittest.TestCase):
    def test_composite_score_logging(self):
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 100)
        raw_pos = np.random.normal(0, 1, 100)
        vol = np.random.uniform(0.01, 0.05, 100)

        # We are mainly running this to ensure no crashes and to check logs manually if needed.
        # But we can check return values too.
        score, metrics = composite_score_with_constraints(
            returns, raw_pos, vol=vol, target_vol=0.1
        )

        self.assertIsInstance(score, float)
        self.assertIsInstance(metrics, dict)
        self.assertIn("PnL", metrics)
        self.assertIn("Sortino", metrics)
        self.assertIn("MaxDD", metrics)

if __name__ == '__main__':
    unittest.main()
