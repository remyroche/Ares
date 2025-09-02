#!/usr/bin/env bash
set -Eeuo pipefail

# Repo Health Sweep: detect tools, run checks, save reports, produce a summary.
# Usage:
#   scripts/repo_health.sh [--autofix] [--only=python,js,shell,docs] [--skip-tests] [--report-dir PATH] [--target-dir PATH] [--file PATH]
#
# Notes:
# - Writes reports to reports/YYYYmmdd_HHMMSS/ by default and records the path in reports/_latest.txt
# - Never deletes files. Only writes new reports.
# - --file option allows checking individual files instead of directories

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

timestamp="$(date +%Y%m%d_%H%M%S)"
report_dir="$repo_root/reports/$timestamp"
mkdir -p "$report_dir"
echo "$report_dir" > "$repo_root/reports/_latest.txt"

autofix="false"
only_sets=""
skip_tests="false"
custom_report_dir=""
target_dir="$repo_root"  # default to whole repo; can override with --target-dir
target_file=""           # for single file operations

for arg in "$@"; do
  case "$arg" in
    --autofix) autofix="true" ;;
    --skip-tests) skip_tests="true" ;;
    --only=*) only_sets="${arg#--only=}" ;;
    --report-dir=*) custom_report_dir="${arg#--report-dir=}" ;;
    --target-dir=*) target_dir="${arg#--target-dir=}" ;;
    --file=*) target_file="${arg#--file=}" ;;
    *) echo "Unknown arg: $arg" >&2; exit 2 ;;
  esac
done

if [[ -n "$custom_report_dir" ]]; then
  report_dir="$custom_report_dir"
  mkdir -p "$report_dir"
  echo "$report_dir" > "$repo_root/reports/_latest.txt"
fi

# If --file is specified, override target_dir and set it to the file's directory
if [[ -n "$target_file" ]]; then
  if [[ ! -f "$target_file" ]]; then
    echo "Error: File '$target_file' does not exist" >&2
    exit 1
  fi
  target_dir="$(dirname "$target_file")"
  # Ensure target_dir is absolute
  if [[ ! "$target_dir" = /* ]]; then
    target_dir="$(cd "$target_dir" && pwd)"
  fi
fi

log="$report_dir/run.log"
touch "$log"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$log"; }
have() { command -v "$1" >/dev/null 2>&1; }
any_file() { compgen -G "$1" >/dev/null 2>&1; }
has_pyproj() { [[ -f "$target_dir/pyproject.toml" || -f "$target_dir/setup.cfg" || -f "$target_dir/requirements.txt" || -f "$repo_root/requirements.txt" || -f "$repo_root/pyproject.toml" ]]; }
has_node() { [[ -f "$target_dir/package.json" || -f "$repo_root/package.json" ]]; }

# Try running Python-based tools via their module if binaries are missing
run_py_tool() {
  local bin_name="$1"; shift
  local module_name="$1"; shift
  if have "$bin_name"; then
    "$bin_name" "$@"
  else
    if have python3; then
      python3 -m "$module_name" "$@"
    else
      return 127
    fi
  fi
}

# Common find exclusions
exclude_prune=(-name .git -o -name node_modules -o -name .venv -o -name venv -o -name .mypy_cache -o -name .pytest_cache -o -name dist -o -name build -o -name reports)

# Split --only into booleans
want_python="true"; want_js="true"; want_shell="true"; want_docs="true"
if [[ -n "$only_sets" ]]; then
  want_python="false"; want_js="false"; want_shell="false"; want_docs="false"
  IFS=',' read -r -a parts <<< "$only_sets"
  for p in "${parts[@]}"; do
    case "$p" in
      python) want_python="true" ;;
      js|ts|node) want_js="true" ;;
      shell) want_shell="true" ;;
      docs|text) want_docs="true" ;;
    esac
  done
fi

say "Repo root: $repo_root"
say "Target dir: $target_dir"
if [[ -n "$target_file" ]]; then
  say "Target file: $target_file"
fi
say "Report dir: $report_dir"
say "Autofix: $autofix, Skip tests: $skip_tests, Only: ${only_sets:-all}"

# --------------------
# Python
# --------------------
if [[ "$want_python" == "true" ]] && has_pyproj; then
  say "Python detected."
  if have python3; then
    # Determine if we're checking a single file or directory
    if [[ -n "$target_file" ]]; then
      # Single file mode
      say "Running: ruff check (single file)"
      if [[ "$autofix" == "true" ]]; then
        run_py_tool ruff ruff check "$target_file" --fix --exit-zero --output-format=json > "$report_dir/ruff.json" 2>>"$log" || say "ruff not available, skipping ruff"
      else
        run_py_tool ruff ruff check "$target_file" --exit-zero --output-format=json > "$report_dir/ruff.json" 2>>"$log" || say "ruff not available, skipping ruff"
      fi

      say "Running: black (single file)"
      if [[ "$autofix" == "true" ]]; then
        run_py_tool black black "$target_file" --quiet 2>>"$log" || say "black not available, skipping black"
      else
        run_py_tool black black "$target_file" --check --diff > "$report_dir/black.diff" 2>>"$log" || say "black not available, skipping black"
      fi

      say "Running: isort (single file)"
      if [[ "$autofix" == "true" ]]; then
        run_py_tool isort isort "$target_file" --atomic --quiet 2>>"$log" || say "isort not available, skipping isort"
      else
        run_py_tool isort isort "$target_file" --check-only --diff > "$report_dir/isort.diff" 2>>"$log" || say "isort not available, skipping isort"
      fi

      say "Running: mypy (single file)"
      run_py_tool mypy mypy "$target_file" --pretty --show-error-codes > "$report_dir/mypy.txt" 2>>"$log" || say "mypy not available, skipping mypy"

      say "Running: bandit (single file)"
      run_py_tool bandit bandit -q "$target_file" -f json -o "$report_dir/bandit.json" 2>>"$log" || say "bandit not available, skipping bandit"
    else
      # Directory mode (original behavior)
      say "Running: ruff check"
      if [[ "$autofix" == "true" ]]; then
        run_py_tool ruff ruff check "$target_dir" --fix --exit-zero --output-format=json > "$report_dir/ruff.json" 2>>"$log" || say "ruff not available, skipping ruff"
      else
        run_py_tool ruff ruff check "$target_dir" --exit-zero --output-format=json > "$report_dir/ruff.json" 2>>"$log" || say "ruff not available, skipping ruff"
      fi

      say "Running: black"
      if [[ "$autofix" == "true" ]]; then
        run_py_tool black black "$target_dir" --quiet 2>>"$log" || say "black not available, skipping black"
      else
        run_py_tool black black "$target_dir" --check --diff > "$report_dir/black.diff" 2>>"$log" || say "black not available, skipping black"
      fi

      say "Running: isort"
      if [[ "$autofix" == "true" ]]; then
        run_py_tool isort isort "$target_dir" --atomic --quiet 2>>"$log" || say "isort not available, skipping isort"
      else
        run_py_tool isort isort "$target_dir" --check-only --diff > "$report_dir/isort.diff" 2>>"$log" || say "isort not available, skipping isort"
      fi

      say "Running: mypy"
      run_py_tool mypy mypy "$target_dir" --pretty --show-error-codes > "$report_dir/mypy.txt" 2>>"$log" || say "mypy not available, skipping mypy"

      say "Running: bandit"
      run_py_tool bandit bandit -q -r "$target_dir" -f json -o "$report_dir/bandit.json" 2>>"$log" || say "bandit not available, skipping bandit"
    fi

    if [[ "$skip_tests" != "true" ]]; then
      say "Running: pytest"
      if [[ -n "$target_file" ]]; then
        # For single files, run pytest on the file's directory to catch related tests
        (cd "$repo_root" && run_py_tool pytest pytest -q "$target_dir" > "$report_dir/pytest.txt" 2>>"$log") || say "pytest not available, skipping tests"
      else
        (cd "$repo_root" && run_py_tool pytest pytest -q "$target_dir" > "$report_dir/pytest.txt" 2>>"$log") || say "pytest not available, skipping tests"
      fi
    fi
  else
    say "python3 not found, skipping Python checks"
  fi
else
  say "Python not selected or not detected."
fi

# --------------------
# JS/TS
# --------------------
if [[ "$want_js" == "true" ]] && has_node; then
  say "Node detected."
  if have node; then
    if have npx; then
      if [[ -n "$target_file" ]]; then
        # Single file mode
        if [[ -f "$target_dir/.eslintrc.js" || -f "$target_dir/.eslintrc.cjs" || -f "$target_dir/.eslintrc.json" || -f "$repo_root/.eslintrc.js" || -f "$repo_root/.eslintrc.cjs" || -f "$repo_root/.eslintrc.json" || -f "$repo_root/package.json" || -f "$target_dir/package.json" ]]; then
          say "Running: eslint (single file)"
          if [[ "$autofix" == "true" ]]; then
            npx --yes eslint "$target_file" --fix --format json -o "$report_dir/eslint.json" 2>>"$log" || true
          else
            npx --yes eslint "$target_file" --format json -o "$report_dir/eslint.json" 2>>"$log" || true
          fi
        else
          say "ESLint config not found, skipping eslint"
        fi
        if [[ -f "$target_dir/tsconfig.json" || -f "$repo_root/tsconfig.json" ]]; then
          say "Running: tsc --noEmit (single file)"
          npx --yes tsc --noEmit --pretty false "$target_file" > "$report_dir/tsc.txt" 2>>"$log" || true
        fi
        if [[ -f "$target_dir/.prettierrc" || -f "$target_dir/.prettierrc.json" || -f "$target_dir/.prettierrc.js" || -f "$repo_root/.prettierrc" || -f "$repo_root/.prettierrc.json" || -f "$repo_root/.prettierrc.js" || -f "$repo_root/package.json" || -f "$target_dir/package.json" ]]; then
          say "Running: prettier check (single file)"
          if [[ "$autofix" == "true" ]]; then
            npx --yes prettier "$target_file" --write > "$report_dir/prettier.txt" 2>>"$log" || true
          else
            npx --yes prettier "$target_file" --check > "$report_dir/prettier.txt" 2>>"$log" || true
          fi
        fi
      else
        # Directory mode (original behavior)
        if [[ -f "$target_dir/.eslintrc.js" || -f "$target_dir/.eslintrc.cjs" || -f "$target_dir/.eslintrc.json" || -f "$repo_root/.eslintrc.js" || -f "$repo_root/.eslintrc.cjs" || -f "$repo_root/.eslintrc.json" || -f "$repo_root/package.json" || -f "$target_dir/package.json" ]]; then
          say "Running: eslint"
          if [[ "$autofix" == "true" ]]; then
            npx --yes eslint "$target_dir" --fix --format json -o "$report_dir/eslint.json" 2>>"$log" || true
          else
            npx --yes eslint "$target_dir" --format json -o "$report_dir/eslint.json" 2>>"$log" || true
          fi
        else
          say "ESLint config not found, skipping eslint"
        fi
        if [[ -f "$target_dir/tsconfig.json" || -f "$repo_root/tsconfig.json" ]]; then
          say "Running: tsc --noEmit"
          npx --yes tsc --noEmit --pretty false > "$report_dir/tsc.txt" 2>>"$log" || true
        fi
        if [[ -f "$target_dir/.prettierrc" || -f "$target_dir/.prettierrc.json" || -f "$target_dir/.prettierrc.js" || -f "$repo_root/.prettierrc" || -f "$repo_root/.prettierrc.json" || -f "$repo_root/.prettierrc.js" || -f "$repo_root/package.json" || -f "$target_dir/package.json" ]]; then
          say "Running: prettier check"
          if [[ "$autofix" == "true" ]]; then
            npx --yes prettier "$target_dir" --write > "$report_dir/prettier.txt" 2>>"$log" || true
          else
            npx --yes prettier "$target_dir" --check > "$report_dir/prettier.txt" 2>>"$log" || true
          fi
        fi
      fi
      if [[ "$skip_tests" != "true" ]]; then
        if [[ -f "$repo_root/package.json" || -f "$target_dir/package.json" ]]; then
          say "Running: npm test (if configured)"
          (cd "$repo_root" && npm test --silent > "$report_dir/npm_test.txt" 2>>"$log") || true
        fi
      fi
    else
      say "npx not found, skipping Node checks"
    fi
  else
    say "node not found, skipping Node checks"
  fi
else
  say "JS/TS not selected or not detected."
fi

# --------------------
# Shell
# --------------------
if [[ "$want_shell" == "true" ]]; then
  if have shellcheck; then
    say "Running: shellcheck"
    sc_json="$report_dir/shellcheck.json"
    if [[ -n "$target_file" ]]; then
      # Single file mode
      if [[ "$target_file" == *.sh ]]; then
        shellcheck -f json "$target_file" > "$sc_json" 2>>"$log" || true
      else
        say "Target file is not a shell script, skipping shellcheck"
      fi
    else
      # Directory mode (original behavior)
      # Collect shell files and run shellcheck
      mapfile -d '' sh_files < <(find "$target_dir" \( -type d \( "${exclude_prune[@]}" \) -prune \) -o -type f -name "*.sh" -print0)
      if (( ${#sh_files[@]} > 0 )); then
        shellcheck -f json "${sh_files[@]}" > "$sc_json" 2>>"$log" || true
      else
        say "No .sh files found"
      fi
    fi
  else
    say "shellcheck not found, skipping shellcheck"
  fi
  if have shfmt && [[ "$autofix" == "true" ]]; then
    say "Running: shfmt -w"
    if [[ -n "$target_file" ]]; then
      # Single file mode
      if [[ "$target_file" == *.sh ]]; then
        shfmt -w "$target_file"
      else
        say "Target file is not a shell script, skipping shfmt"
      fi
    else
      # Directory mode (original behavior)
      find "$target_dir" \( -type d \( "${exclude_prune[@]}" \) -prune \) -o -type f -name "*.sh" -exec shfmt -w {} +
    fi
  fi
fi

# --------------------
# Docs / Configs
# --------------------
if [[ "$want_docs" == "true" ]]; then
  if have yamllint; then
    say "Running: yamllint"
    if [[ -n "$target_file" ]]; then
      # Single file mode
      if [[ "$target_file" == *.yaml || "$target_file" == *.yml ]]; then
        yamllint -f parsable "$target_file" > "$report_dir/yamllint.txt" 2>>"$log" || true
      else
        say "Target file is not a YAML file, skipping yamllint"
      fi
    else
      # Directory mode (original behavior)
      yamllint -f parsable "$target_dir" > "$report_dir/yamllint.txt" 2>>"$log" || true
    fi
  else
    say "yamllint not found, skipping yamllint"
  fi
  if have markdownlint; then
    say "Running: markdownlint"
    if [[ -n "$target_file" ]]; then
      # Single file mode
      if [[ "$target_file" == *.md ]]; then
        markdownlint "$target_file" > "$report_dir/markdownlint.txt" 2>>"$log" || true
      else
        say "Target file is not a markdown file, skipping markdownlint"
      fi
    else
      # Directory mode (original behavior)
      markdownlint "$target_dir" > "$report_dir/markdownlint.txt" 2>>"$log" || true
    fi
  else
    say "markdownlint not found, skipping markdownlint"
  fi
  if have jq; then
    say "Running: JSON lint"
    : > "$report_dir/jsonlint.txt"
    if [[ -n "$target_file" ]]; then
      # Single file mode
      if [[ "$target_file" == *.json ]]; then
        if ! jq -e . "$target_file" >/dev/null 2>&1; then
          echo "Invalid JSON: $target_file" >> "$report_dir/jsonlint.txt"
        fi
      else
        say "Target file is not a JSON file, skipping JSON lint"
      fi
    else
      # Directory mode (original behavior)
      while IFS= read -r -d '' f; do
        if ! jq -e . "$f" >/dev/null 2>&1; then
          echo "Invalid JSON: $f" >> "$report_dir/jsonlint.txt"
        fi
      done < <(find "$target_dir" \( -type d \( "${exclude_prune[@]}" \) -prune \) -o -type f -name "*.json" -print0)
    fi
  else
    say "jq not found, skipping JSON lint"
  fi
fi

# --------------------
# Merge reports
# --------------------
if have python3; then
  say "Merging reports into summary"
  python3 "$script_dir/merge_reports.py" "$report_dir" 2>>"$log" || true
else
  say "python3 not found; cannot generate summary"
fi

say "Done. See: $report_dir"

