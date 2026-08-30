import os
import json

def write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        data = json.load(sys.stdin)
        for path, content in data.items():
            write_file(path, content)
