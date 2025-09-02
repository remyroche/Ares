#!/usr/bin/env python3
"""
Step Formatting Demonstration

This script demonstrates what the step formatter does by showing before/after examples.
"""

import re

def demonstrate_step_formatting():
    """Show examples of step formatting."""
    
    print("🔍 STEP FORMATTER DEMONSTRATION")
    print("=" * 50)
    print()
    
    # Examples of what gets formatted
    examples = [
        "step01",
        "step02", 
        "step03",
        "step04",
        "step05",
        "step06",
        "step07",
        "step08",
        "step09",
        "step10",  # This won't change (already double digit)
        "step11",  # This won't change (already double digit)
        "step12"   # This won't change (already double digit)
    ]
    
    # Regex pattern used by the formatter
    step_pattern = re.compile(r'\bstep([1-9])\b')
    
    print("📝 EXAMPLES OF STEP FORMATTING:")
    print("-" * 40)
    
    for example in examples:
        # Check if it matches the pattern
        if step_pattern.search(example):
            # Format it
            formatted = step_pattern.sub(lambda m: f'step0{m.group(1)}', example)
            print(f"  {example:<25} → {formatted}")
        else:
            print(f"  {example:<25} → (no change)")
    
    print()
    print("📋 WHAT GETS FORMATTED:")
    print("-" * 40)
    print("  ✅ step01  → step01")
    print("  ✅ step02  → step02") 
    print("  ✅ step03  → step03")
    print("  ✅ step04  → step04")
    print("  ✅ step05  → step05")
    print("  ✅ step06  → step06")
    print("  ✅ step07  → step07")
    print("  ✅ step08  → step08")
    print("  ✅ step09  → step09")
    print("  ❌ step10 → step10 (no change - already double digit)")
    print("  ❌ step11 → step11 (no change - already double digit)")
    print("  ❌ step12 → step12 (no change - already double digit)")
    
    print()
    print("🎯 PATTERN EXPLANATION:")
    print("-" * 40)
    print("  The formatter uses this regex pattern:")
    print("  \\bstep([1-9])\\b")
    print()
    print("  This matches:")
    print("  • Word boundaries (\\b)")
    print("  • Literal 'step'")
    print("  • Single digit 1-9 ([1-9])")
    print("  • Word boundaries (\\b)")
    print()
    print("  So it only matches single-digit steps, not double-digit ones.")
    
    print()
    print("🚀 HOW TO USE:")
    print("-" * 40)
    print("  1. See what would change (dry run):")
    print("     python3 format_steps.py")
    print()
    print("  2. Apply the changes:")
    print("     python3 format_steps.py --apply")
    print()
    print("  3. Apply with backup files:")
    print("     python3 format_steps.py --apply --backup")
    print()
    print("  4. Get help:")
    print("     python3 format_steps.py --help")

if __name__ == "__main__":
    demonstrate_step_formatting()