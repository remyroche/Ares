Code Quality Toolkit

This directory contains scripts to assess and improve code quality across the repository. The toolkit is modular; each script focuses on a specific concern and can run independently or via the aggregate runner.

Available categories:

- Auto-fixes (Python: black, ruff, isort, autoflake; Optional JS: prettier, eslint)
- Linting (Python: ruff/flake8; Optional JS: eslint if present)
- Formatting (Python: black; Optional JS: prettier)
- Type Checking (Python: mypy; Optional JS/TS: tsc if present)
- Dead Code Detection (Python: vulture)
- Circular Import Detection (Python: pylint)
- Dependency Mapping (Python: pydeps; tree via pipdeptree)

- Complexity Analysis (Python: radon)
- Test Coverage (pytest + coverage if tests configured)
- Dependency Audit (Python: pip-audit)

Usage

- Run all checks: ./run_all.sh
- Run an individual category: see scripts in ./scripts and ./python

Notes

- Scripts are designed to work even if some tools are missing; they will report and skip gracefully. Use ./bootstrap_tools.sh to install suggested tools in a virtual environment or your environment.
- The toolkit auto-detects languages based on repository contents.

