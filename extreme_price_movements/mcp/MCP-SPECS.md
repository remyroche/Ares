# 🛠️ Quant-Advisor Integration Specs

## 1. Installation & Runtime

Add this to your IDE's MCP configuration file.

Note: JSON does not support comments, so keep the file comment-free.

```json
{
  "mcpServers": {
    "quant-advisor": {
      "command": "node",
      "args": ["/path/to/your/dist/index.js"],
      "env": {
        "QUANT_PROVIDER": "codex",
        "OPENAI_API_KEY": "sk-...",
        "MAX_PARALLEL_ADVISORS": "3",
        "DEBUG": "1"
      }
    }
  }
}
