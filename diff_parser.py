import re

with open("meta_diff.txt", "r") as f:
    text = f.read()

lines = text.split("\n")
additions = []
deletions = []

for line in lines:
    if line.startswith("+") and not line.startswith("+++"):
        additions.append(line[1:])
    elif line.startswith("-") and not line.startswith("---"):
        deletions.append(line[1:])

print("=== ADDITIONS ===")
for a in additions:
    print(a)

print("\n=== DELETIONS ===")
for d in deletions:
    print(d)
