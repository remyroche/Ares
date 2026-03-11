# AGENTS.md

## Project mission

This repository implements **machine learning infrastructure for systematic trading research**.

Primary goals:

- discover **predictive regimes**
- train **robust ML trading models**
- evaluate **economic tradability**
- maintain **strict out-of-sample discipline**

All work must prioritize:

1. **statistical validity**
2. **economic relevance**
3. **out-of-sample robustness**
4. **computational efficiency**

Where trade-offs occur, **correctness and robustness override speed**.

---

# Environment and tooling

## Python

Python version:

3.11

Defined in:

pyproject.toml

---

## Package manager

Dependency management:

Poetry

Typical commands:

poetry install  
poetry shell  
poetry run <command>

---

## Formatting and linting

Configured in `pyproject.toml`.

Tools:

- black (line length 88)
- isort (profile `black`)
- flake8
- pylint
- mypy

Before committing code, ensure formatting is correct.

---

## Testing

Testing framework:

pytest

Markers:

unit  
integration  
slow

Example usage:

pytest -m unit  
pytest -m "not slow"

---

# Validation checklist (minimum)

When code changes are made, run the smallest relevant checks first.

### Targeted tests

pytest tests/<path_or_file> -q

### If shared/core logic changed

pytest -q

### Formatting

black <files>  
isort <files>

---

# Core engineering principles

## Type safety

Use **type hints everywhere**.

Prefer:

```python
def compute_metric(x: np.ndarray) -> np.ndarray:

Avoid untyped functions.

Run mypy for validation.

Logging and progress monitoring

Use tprint() instead of raw print() for long-running operations.

Purpose:

track pipeline progress

monitor expensive loops

identify performance bottlenecks

Avoid excessive logging in hot loops.

Memory safety

The project must run on Mac M1/M2 environments with limited RAM.

Guidelines:

prefer float32 over float64

downcast whenever possible

reuse arrays instead of copying

release large objects after use

call gc.collect() in large batch pipelines

avoid large intermediate pandas objects

If processing large datasets:

use chunking

use shared cache structures

avoid recomputation of expensive tensors

Performance guidelines
Numeric stack

Preferred stack:

NumPy + Numba

Avoid heavy pandas usage inside hot loops.

Numba usage

Use Numba for:

barrier scans

rolling computations

vectorizable loops

event generation

Typical decorator:
@njit(cache=True, fastmath=True)

Rules:

avoid Python objects inside kernels

use typed NumPy arrays

avoid dynamic resizing

Numeric precision

Default numeric type:

float32

Use:

int32

for indices.

Use:

int8

for flags or masks.

Do not introduce float64 unless mathematically required.

Vectorization expectations

Prefer:

NumPy vectorization

over Python loops.

Exception: loops inside Numba kernels.

# Agent Research Protocols

Agents must follow the research methodology defined in:

agents/

start with agents/README.md
