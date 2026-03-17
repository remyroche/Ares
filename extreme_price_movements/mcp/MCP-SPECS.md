# 🛠️ Quant-Advisor Integration Specs

## 1. Installation & Runtime

Add this to your IDE's MCP configuration file.

JSON does not support comments, so keep the file comment-free.

```json
{
  "mcpServers": {
    "quant-advisor": {
      "command": "node",
      "args": ["/absolute/path/to/extreme_price_movements/mcp/dist/index.js"],
      "env": {
        "QUANT_PROVIDER": "codex",
        "OPENAI_API_KEY": "sk-...",
        "MAX_PARALLEL_ADVISORS": "3",
        "DEBUG": "1"
      }
    }
  }
}
```

Set `QUANT_PROVIDER` to `windsurf` or `antigravity` as needed.

If using Windsurf or Antigravity, also provide the appropriate base URL and API key for your compatible inference gateway:

```json
{
  "mcpServers": {
    "quant-advisor": {
      "command": "node",
      "args": ["/absolute/path/to/extreme_price_movements/mcp/dist/index.js"],
      "env": {
        "QUANT_PROVIDER": "windsurf",
        "WINDSURF_BASE_URL": "https://your-compatible-endpoint/v1",
        "WINDSURF_API_KEY": "key-...",
        "MAX_PARALLEL_ADVISORS": "3",
        "DEBUG": "1"
      }
    }
  }
}
```

## 2. Required Project Rules (System Instructions)

Copy this block into your IDE's project rules, custom instructions, or memory:

> MCP TOOL: quant_advisor
>
> Trigger Policy:
> - Call `quant_advisor` for any change involving alpha signals, feature engineering, backtesting logic, labeling logic, execution logic, or risk management.
>
> Execution Flow:
> 1. Provide the current `diff` and `relevant_code` in the `context` object.
> 2. If `problem_stage` is `"implementation"`, you MUST include `changed_files`.
>
> Enforcement:
> - If `confidence < 0.6`: stop and ask for missing context.
> - If `blockers` is not empty: do not apply code changes until blockers are resolved.
> - Treat `checks` as mandatory unit/integration test requirements.
> - Add `risks` to PR descriptions, code comments, or implementation notes where relevant.

## 3. Key Behavioral Logic

| Field | Interpretation for AI Agent |
|---|---|
| `modes` | Which advisory modes were activated by the classifier |
| `advisors_used` | Which expert lenses were actually run |
| `risks` | Points to add to PR descriptions or code comments |
| `actions` | Refactors or implementation steps to perform now |
| `checks` | Assertions, regression tests, NaN/off-by-one checks, or edge cases that must be verified |
| `confidence` | If below `0.6`, ask for more context before proceeding |
| `blockers` | Hard stop; do not apply the change yet |

## Why this works

- **Unified Context**: the IDE uses your advisor instead of inventing ad hoc quant guidance.
- **Schema Enforcement**: `problem_stage` and structured `context` ensure the classifier activates the right advisors.
- **Token Efficiency**: sending `diff`, `relevant_code`, and `data_schema` improves signal quality and increases prompt-cache reuse.
