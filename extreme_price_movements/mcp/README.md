# quant-advisor MCP

A single-tool MCP server for quant research, AFML review, implementation review, validation design, and optimization.

## Folder

This project is laid out for storage in:

`extreme_price_movements/mcp/`

## Supported IDE / provider setups

- **Codex / OpenAI Responses API**
- **Windsurf** via MCP plus an OpenAI-compatible inference endpoint you configure
- **Antigravity** via MCP plus an OpenAI-compatible inference endpoint you configure

## Quick start

```bash
npm install
npm run build
node dist/index.js
```

## Environment examples

### Codex / OpenAI

```bash
export QUANT_PROVIDER=codex
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_CLASSIFIER_MODEL=gpt-5-mini
export OPENAI_ADVISOR_MODEL=gpt-5.3-codex
export MAX_PARALLEL_ADVISORS=3
export MAX_RETRIES=2
export REQUEST_TIMEOUT_MS=20000
export CLASSIFIER_TIMEOUT_MS=8000
export DEBUG=1
```

### Windsurf-compatible gateway

```bash
export QUANT_PROVIDER=windsurf
export WINDSURF_BASE_URL=https://your-compatible-endpoint/v1
export WINDSURF_API_KEY=key-...
export WINDSURF_CLASSIFIER_MODEL=gpt-5-mini
export WINDSURF_ADVISOR_MODEL=gpt-5.3-codex
export MAX_PARALLEL_ADVISORS=3
export MAX_RETRIES=2
export REQUEST_TIMEOUT_MS=20000
export CLASSIFIER_TIMEOUT_MS=8000
export DEBUG=1
```

### Antigravity-compatible gateway

```bash
export QUANT_PROVIDER=antigravity
export ANTIGRAVITY_BASE_URL=https://your-compatible-endpoint/v1
export ANTIGRAVITY_API_KEY=key-...
export ANTIGRAVITY_CLASSIFIER_MODEL=gpt-5-mini
export ANTIGRAVITY_ADVISOR_MODEL=gpt-5.3-codex
export MAX_PARALLEL_ADVISORS=3
export MAX_RETRIES=2
export REQUEST_TIMEOUT_MS=20000
export CLASSIFIER_TIMEOUT_MS=8000
export DEBUG=1
```

## Example MCP payload

```json
{
  "request": "Review this rewrite of triple-barrier labeling and tell me what can break.",
  "problem_stage": "implementation",
  "context": {
    "market": "crypto_perps",
    "horizon": "15m to 4h",
    "changed_files": ["labels/triple_barrier.py"],
    "diff": "diff --git a/labels/triple_barrier.py ...",
    "relevant_code": "def label_events(...): ...",
    "data_schema": "events: ts, side, price ..."
  }
}
```
