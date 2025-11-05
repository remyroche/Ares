#!/usr/bin/env python3
"""Find unmatched triple quotes in a file"""

with open('src/training/steps/market_analysis/clusters/cluster_quality_assessor.py', 'r') as f:
    lines = f.readlines()

triple_quote_count = 0
for i, line in enumerate(lines, 1):
    count = line.count('"""')
    if count > 0:
        triple_quote_count += count
        print(f"Line {i}: {count} triple quotes - {line.strip()[:60]}")

print(f"\nTotal triple quotes: {triple_quote_count}")
if triple_quote_count % 2 != 0:
    print("❌ Odd number of triple quotes - there's an unmatched one!")
else:
    print("✅ Even number of triple quotes")
