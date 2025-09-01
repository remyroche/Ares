#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

# Merge known tool outputs into summary.json and summary.md
# Usage: python scripts/merge_reports.py <report_dir>


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def parse_ruff(path: Path) -> List[Dict[str, Any]]:
    data = load_json(path)
    if not data:
        return []
    items = data if isinstance(data, list) else data.get("diagnostics", [])
    out = []
    for it in items:
        filename = it.get("filename") or it.get("file") or ""
        code = it.get("code") or it.get("rule") or ""
        message = it.get("message") or ""
        loc = it.get("location") or {}
        line = loc.get("row") or it.get("line") or 0
        col = loc.get("column") or it.get("col") or 0
        out.append({
            "tool": "ruff",
            "file": filename,
            "line": line,
            "col": col,
            "rule": code,
            "severity": "warning",
            "message": message,
            "fixable": bool(it.get("fix") or it.get("fixable", False)),
        })
    return out


def parse_eslint(path: Path) -> List[Dict[str, Any]]:
    data = load_json(path)
    if not data:
        return []
    out = []
    for file_result in data if isinstance(data, list) else data.get("results", []):
        file_path = file_result.get("filePath", "")
        for msg in file_result.get("messages", []):
            out.append({
                "tool": "eslint",
                "file": file_path,
                "line": msg.get("line", 0),
                "col": msg.get("column", 0),
                "rule": msg.get("ruleId", "") or "",
                "severity": {2: "error", 1: "warning"}.get(msg.get("severity", 1), "warning"),
                "message": msg.get("message", ""),
                "fixable": bool(msg.get("fix")),
            })
    return out


def parse_bandit(path: Path) -> List[Dict[str, Any]]:
    data = load_json(path)
    if not data:
        return []
    out = []
    for r in data.get("results", []):
        out.append({
            "tool": "bandit",
            "file": r.get("filename", ""),
            "line": r.get("line_number", 0),
            "col": 0,
            "rule": r.get("test_id", ""),
            "severity": r.get("issue_severity", "").lower() or "warning",
            "message": r.get("issue_text", ""),
            "fixable": False,
        })
    return out


def parse_shellcheck(path: Path) -> List[Dict[str, Any]]:
    data = load_json(path)
    if not data:
        return []
    out = []
    comments = data.get("comments", data if isinstance(data, list) else [])
    for c in comments:
        out.append({
            "tool": "shellcheck",
            "file": c.get("file", ""),
            "line": c.get("line", 0),
            "col": c.get("column", 0),
            "rule": f"SC{c.get('code', '')}",
            "severity": (c.get("level") or "warning").lower(),
            "message": c.get("message", ""),
            "fixable": False,
        })
    return out


def parse_line_kv(text: str, tool: str, regex: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    pat = re.compile(regex)
    for line in text.splitlines():
        m = pat.match(line.strip())
        if not m:
            continue
        file, lno, col, msg = m.group(1), int(m.group(2) or 0), int(m.group(3) or 0), m.group(4)
        out.append({
            "tool": tool,
            "file": file,
            "line": lno,
            "col": col,
            "rule": "",
            "severity": "error",
            "message": msg,
            "fixable": False,
        })
    return out


def parse_mypy(path: Path) -> List[Dict[str, Any]]:
    txt = path.read_text(encoding="utf-8")
    return parse_line_kv(txt, "mypy", r"^([^:]+):(\d+):(\d+):\s+(.*)$")


def parse_yamllint(path: Path) -> List[Dict[str, Any]]:
    txt = path.read_text(encoding="utf-8")
    out: List[Dict[str, Any]] = []
    pat = re.compile(r"^([^:]+):(\d+):(\d+):\s+\[(\w+)\]\s+(.*?)(?:\s+\(([^)]+)\))?$")
    for line in txt.splitlines():
        m = pat.match(line.strip())
        if not m:
            continue
        out.append({
            "tool": "yamllint",
            "file": m.group(1),
            "line": int(m.group(2)),
            "col": int(m.group(3)),
            "rule": m.group(6) or "",
            "severity": m.group(4).lower(),
            "message": m.group(5),
            "fixable": False,
        })
    return out


def parse_markdownlint(path: Path) -> List[Dict[str, Any]]:
    txt = path.read_text(encoding="utf-8")
    out: List[Dict[str, Any]] = []
    pat = re.compile(r"^([^:]+):(\d+)\s+([A-Z]+\d+)[^ ]*\s+(.*)$")
    for line in txt.splitlines():
        m = pat.match(line.strip())
        if not m:
            continue
        out.append({
            "tool": "markdownlint",
            "file": m.group(1),
            "line": int(m.group(2)),
            "col": 0,
            "rule": m.group(3),
            "severity": "warning",
            "message": m.group(4),
            "fixable": False,
        })
    return out


def collect(report_dir: Path) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    mapping_json = {
        "ruff.json": parse_ruff,
        "eslint.json": parse_eslint,
        "bandit.json": parse_bandit,
        "shellcheck.json": parse_shellcheck,
    }
    for name, fn in mapping_json.items():
        p = report_dir / name
        if p.exists():
            issues.extend(fn(p))

    mapping_txt = {
        "mypy.txt": parse_mypy,
        "yamllint.txt": parse_yamllint,
        "markdownlint.txt": parse_markdownlint,
    }
    for name, fn in mapping_txt.items():
        p = report_dir / name
        if p.exists():
            issues.extend(fn(p))

    return issues


def write_summary(report_dir: Path, issues: List[Dict[str, Any]]) -> None:
    summary_json: Dict[str, Any] = {
        "report_dir": str(report_dir),
        "total_issues": len(issues),
        "by_tool": {},
        "issues": issues,
    }
    for it in issues:
        summary_json["by_tool"].setdefault(it["tool"], 0)
        summary_json["by_tool"][it["tool"]] += 1
    (report_dir / "summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append(f"# Repo Health Summary")
    lines.append("")
    lines.append(f"- Report directory: `{report_dir}`")
    lines.append(f"- Total issues: {len(issues)}")
    if summary_json["by_tool"]:
        lines.append("- By tool: " + ", ".join(f"{k}={v}" for k, v in summary_json["by_tool"].items()))
    lines.append("")
    lines.append("## Top issues")
    if not issues:
        lines.append("No issues found.")
    else:
        for it in issues[:50]:
            lines.append(f"- [{it['tool']}] {it['file']}:{it['line']}:{it['col']} {it['severity'].upper()} {it.get('rule','')}: {it['message']}")
        if len(issues) > 50:
            lines.append(f"... and {len(issues)-50} more")
    (report_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: merge_reports.py <report_dir>", file=sys.stderr)
        sys.exit(2)
    report_dir = Path(sys.argv[1]).resolve()
    issues = collect(report_dir)
    write_summary(report_dir, issues)


if __name__ == "__main__":
    main()

