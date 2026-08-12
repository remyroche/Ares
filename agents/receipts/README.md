# Agent Evidence Receipts

Every commit that changes governed Python code must include one receipt in this
directory. Governed code is `extreme_price_movements/**/*.py` and `scripts/*.py`,
except the receipt validator itself. Tests alone do not require a receipt.

The receipt is a JSON object with this shape:

```json
{
  "task": "short-task-name",
  "scope": "What changed and why.",
  "changed_paths": ["extreme_price_movements/example.py"],
  "contracts": [
    {
      "path": "agents/dataset_contract.md",
      "sha256": "sha256 of the contract read for this task"
    }
  ],
  "validation": {
    "status": "passed",
    "commands": ["pytest -q tests/test_example.py"],
    "not_run_reason": ""
  },
  "agent_plan": {
    "subagents": [
      {
        "model": "luna",
        "deliverable": "bounded review",
        "read_paths": ["extreme_price_movements/example.py"],
        "write_paths": []
      }
    ]
  }
}
```

Use `model: "luna"` whenever Luna is available. A `gpt-5.6-terra` entry must
also include a non-empty `terra_exception` explaining why the work met the
exception threshold in `AGENTS.md`.
