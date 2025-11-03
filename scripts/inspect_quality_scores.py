"""
Inspect quality scores to verify they make sense.

Samples levels from each quality tier and displays their features
to manually verify that quality scores align with level characteristics.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def inspect_quality_scores(data_path: str = 'data_cache/sr_ml_training/sr_quality_training_data.parquet'):
    """Manually inspect quality scores."""
    
    if not Path(data_path).exists():
        logger.error(f"❌ Training data not found: {data_path}")
        logger.error(f"   Run step 2.5 to generate training data first")
        return
    
    data = pd.read_parquet(data_path)
    
    logger.info("\n" + "="*80)
    logger.info("  QUALITY SCORE INSPECTION")
    logger.info("="*80)
    logger.info(f"\nDataset: {len(data):,} samples")
    
    # Sample from each tier
    tiers = {
        'Noise (0.0-0.3)': (0.0, 0.3),
        'Weak (0.3-0.5)': (0.3, 0.5),
        'Medium (0.5-0.7)': (0.5, 0.7),
        'Strong (0.7-0.85)': (0.7, 0.85),
        'Critical (0.85-1.0)': (0.85, 1.0)
    }
    
    inconsistencies_found = 0
    
    for tier_name, (min_q, max_q) in tiers.items():
        tier_data = data[
            (data['quality_score'] >= min_q) &
            (data['quality_score'] < max_q)
        ]
        
        if len(tier_data) == 0:
            logger.info(f"\n{'='*80}")
            logger.info(f"  {tier_name}")
            logger.info(f"{'='*80}")
            logger.info(f"  No samples in this tier")
            continue
        
        # Sample 5 random levels
        sample_size = min(5, len(tier_data))
        samples = tier_data.sample(sample_size, random_state=42)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"  {tier_name}")
        logger.info(f"  Population: {len(tier_data):,} samples ({len(tier_data)/len(data)*100:.1f}%)")
        logger.info(f"{'='*80}")
        
        for i, (idx, row) in enumerate(samples.iterrows(), 1):
            logger.info(f"\nSample {i}/{sample_size}:")
            logger.info(f"  Quality Score:      {row['quality_score']:.3f}")
            
            # Core features
            touches = row.get('feature_touch_count', 0)
            strength = row.get('feature_strength', 0)
            consistency = row.get('feature_consistency', 0)
            volume_conf = row.get('feature_volume_confirmation', 0)
            bounce_ratio = row.get('feature_avg_bounce_ratio', 0)
            prominence = row.get('feature_prominence', 0)
            
            logger.info(f"  Touches:            {touches}")
            logger.info(f"  Strength:           {strength:.3f}")
            logger.info(f"  Consistency:        {consistency:.3f}")
            logger.info(f"  Volume Confirm:     {volume_conf:.3f}")
            logger.info(f"  Avg Bounce Ratio:   {bounce_ratio:.3f}")
            logger.info(f"  Prominence:         {prominence:.3f}")
            
            # Check for inconsistencies
            is_inconsistent = False
            
            if row['quality_score'] >= 0.7:  # Should be strong
                if touches < 3:
                    logger.warning(f"  ⚠️ HIGH quality ({row['quality_score']:.2f}) but LOW touches ({touches})!")
                    is_inconsistent = True
                if strength < 0.4:
                    logger.warning(f"  ⚠️ HIGH quality ({row['quality_score']:.2f}) but LOW strength ({strength:.2f})!")
                    is_inconsistent = True
                    
            elif row['quality_score'] < 0.3:  # Should be weak
                if touches >= 8 and strength >= 0.7:
                    logger.warning(f"  ⚠️ LOW quality ({row['quality_score']:.2f}) but HIGH metrics (touches={touches}, strength={strength:.2f})!")
                    is_inconsistent = True
            
            if is_inconsistent:
                inconsistencies_found += 1
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info(f"  INSPECTION SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"\nInconsistencies found: {inconsistencies_found}")
    
    if inconsistencies_found == 0:
        logger.info(f"✅ Quality scores appear correctly calculated!")
        logger.info(f"   Low R² for strong levels is likely due to variance restriction")
    elif inconsistencies_found <= 3:
        logger.info(f"🟡 Minor inconsistencies found (could be edge cases)")
    else:
        logger.info(f"❌ Multiple inconsistencies - quality calculation may be flawed!")
        logger.info(f"   Recommend reviewing _measure_level_performance() method")


if __name__ == "__main__":
    inspect_quality_scores()

