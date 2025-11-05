# IDE Configuration for Ares Project
# Add this to your IDE settings to resolve import warnings

## VS Code Settings (.vscode/settings.json):
```json
{
    "python.defaultInterpreterPath": "/usr/bin/python3",
    "python.analysis.extraPaths": [
        "/Users/remyroche/Documents/Ares",
        "/Users/remyroche/Documents/Ares/src"
    ],
    "python.analysis.autoImportCompletions": true,
    "python.analysis.typeCheckingMode": "basic",
    "python.analysis.diagnosticSeverityOverrides": {
        "reportMissingImports": "information",
        "reportUndefinedVariable": "error"
    }
}
```

## PyCharm Settings:
1. File → Settings → Project → Python Interpreter
2. Add `/Users/remyroche/Documents/Ares` to "Content Roots"
3. Add `/Users/remyroche/Documents/Ares/src` to "Source Roots"

## Alternative: Use absolute imports
Instead of:
```python
from src.utils.tprint import tprint
```
Use:
```python
import sys
sys.path.insert(0, '/Users/remyroche/Documents/Ares')
from src.utils.tprint import tprint
```
