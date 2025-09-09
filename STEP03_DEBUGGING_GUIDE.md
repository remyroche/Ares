### Step03 Debugging Guide

This guide explains how to run the Step03 debug suite, what it checks, and how to interpret results.

#### What the debug suite does
- Import checks for Step03 modules and core decorators
- Dependency summary (key versions)
- Data presence check under processed data directory
- Artifact validation via the Step03 validator
- Resource snapshot (CPU/memory/disk)
- Optional smoke test of the enhanced Step03 runner

#### How to run
```bash
python scripts/run_step03_debug.py --symbol ETHUSDT --exchange BINANCE --timeframe 1m --data-dir data_cache --smoke --timeout 60
```

Flags:
- `--symbol`, `--exchange`, `--timeframe`: Select dataset
- `--data-dir`: Override processed data base directory (optional)
- `--smoke`: Run a short smoke test of Step03
- `--timeout`: Smoke test timeout in seconds (default 30)
- `--output-dir`: Where to save the JSON report (default `results`)

#### Outputs
- JSON report saved to `results/step03_debug_report_<symbol>_<timeframe>.json`

#### Typical issues and fixes
- Missing processed inputs: run earlier steps or provide correct `--data-dir`
- Import failures: ensure your PYTHONPATH and package structure are intact; install missing dependencies
- Artifacts validation failed: re-run Step03 and inspect logs for errors
- Smoke test timeout: increase `--timeout` or check for long-running operations

