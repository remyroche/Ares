#!/usr/bin/env python3
"""
Add trail_mult to GridResult and OuterFoldResult instantiations in optimise_tpsl_ratio.py
"""

import re

file_path = "/Users/remyroche/Documents/Ares/extreme_price_movements/optimise_tpsl_ratio.py"

with open(file_path, 'r') as f:
    content = f.read()

# Fix 1: Add trail_mult to GridResult instantiation (line ~1089)
# Find the pattern and add trail_mult after sl_mult
pattern1 = r'(grid_metrics\.append\(GridResult\(\s+tp_mult=float\(tp_mult\),\s+sl_mult=float\(sl_mult\),)\s+(lo=float\(lo_val\),)'
replacement1 = r'\1\n                                    trail_mult=float(trail_mult),  # NEW\n                                    \2'
content = re.sub(pattern1, replacement1, content)

# Fix 2: Add trail_mult to OuterFoldResult instantiation
# Find where OuterFoldResult is created and add trail_mult
pattern2 = r'(OuterFoldResult\(\s+fold=ofold,\s+chosen_tp_mult=best\.tp_mult,\s+chosen_sl_mult=best\.sl_mult,)\s+(chosen_lo=best\.lo,)'
replacement2 = r'\1\n                chosen_trail_mult=best.trail_mult,  # NEW\n                \2'
content = re.sub(pattern2, replacement2, content)

# Fix 3: Update chosen_configs tuple to include trail_mult
pattern3 = r'chosen_configs\.append\(\(best\.tp_mult, best\.sl_mult, best\.lo, best\.hi, best\.z_max, best\.threshold_p\)\)'
replacement3 = 'chosen_configs.append((best.tp_mult, best.sl_mult, best.trail_mult, best.lo, best.hi, best.z_max, best.threshold_p))'
content = re.sub(pattern3, replacement3, content)

# Fix 4: Update final aggregation to include trail_mult
pattern4 = r'(final_tp_mult = np\.median\(\[c\[0\] for c in chosen_configs\]\)\s+final_sl_mult = np\.median\(\[c\[1\] for c in chosen_configs\]\))\s+(final_lo = np\.median\(\[c\[2\] for c in chosen_configs\]\))'
replacement4 = r'\1\n    final_trail_mult = np.median([c[2] for c in chosen_configs])\n    \2'
content = re.sub(pattern4, replacement4, content)

# Fix 5: Update index references for lo, hi, z_max, threshold_p (they shift by 1)
content = re.sub(r'\[c\[2\] for c in chosen_configs\]', '[c[3] for c in chosen_configs]', content)  # lo
content = re.sub(r'\[c\[3\] for c in chosen_configs\]', '[c[4] for c in chosen_configs]', content)  # hi  
content = re.sub(r'\[c\[4\] for c in chosen_configs\]', '[c[5] for c in chosen_configs]', content)  # z_max
content = re.sub(r'\[c\[5\] for c in chosen_configs\]', '[c[6] for c in chosen_configs]', content)  # threshold_p

# Fix 6: Update SelectionSummary return to include final_trail_mult
pattern6 = r'(return SelectionSummary\(\s+chosen_configs=chosen_configs,\s+outer_results=outer_results,\s+final_tp_mult=final_tp_mult,\s+final_sl_mult=final_sl_mult,)\s+(final_lo=final_lo,)'
replacement6 = r'\1\n        final_trail_mult=final_trail_mult,\n        \2'
content = re.sub(pattern6, replacement6, content)

# Fix 7: Update fallback return statement
pattern7 = r'return SelectionSummary\(\[\], \[\], 1\.0, 1\.0, lo, hi, z_max, 0\.5\)'
replacement7 = 'return SelectionSummary([], [], 1.0, 1.0, 0.5, lo, hi, z_max, 0.5)'  # Added 0.5 for trail_mult
content = re.sub(pattern7, replacement7, content)

with open(file_path, 'w') as f:
    f.write(content)

print("✅ Successfully updated optimise_tpsl_ratio.py with trail_mult support")
