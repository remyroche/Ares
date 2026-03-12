import re

with open("extreme_price_movements/mask_optimiser.py", "r") as f:
    code = f.read()

# Instead of patching by searching strings, let's write a simple python parser using AST or just robust regex
# Wait, let's check lines 4768-4793 for the Dominance Pruning and Complexity Penalties.
