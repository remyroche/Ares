#!/bin/bash
set -e

echo "--- Ruff: Lint & Auto‑Fix"
ruff format .
ruff check . --fix
ruff check .

echo "--- MyPy: Static Type Checking"
mypy . --strict

echo "--- Radon: Complexity & Maintainability"
radon cc src/ -s -a -nc
radon mi src/ -s -a -nc

echo "--- Scalpel: CFG, Call Graph, Type Inference"
scalpel --call-graph src/ > scalpel_callgraph.json
scalpel --type-infer src/ > scalpel_typeinfer.json

echo "--- PyTracer: Numerical Stability Trace (via pytest)"
pytracer trace --command "pytest"
pytracer parse
pytracer visualize & # optionally background the dashboard

echo "Static + quality + numeric scan complete."
