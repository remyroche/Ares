import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";

const ENV = {
  QUANT_PROVIDER: process.env.QUANT_PROVIDER ?? "codex",

  OPENAI_API_KEY: process.env.OPENAI_API_KEY ?? "",
  OPENAI_BASE_URL: process.env.OPENAI_BASE_URL ?? "https://api.openai.com/v1",
  OPENAI_CLASSIFIER_MODEL: process.env.OPENAI_CLASSIFIER_MODEL ?? "gpt-5-mini",
  OPENAI_ADVISOR_MODEL: process.env.OPENAI_ADVISOR_MODEL ?? "gpt-5.3-codex",

  WINDSURF_API_KEY: process.env.WINDSURF_API_KEY ?? "",
  WINDSURF_BASE_URL: process.env.WINDSURF_BASE_URL ?? "",
  WINDSURF_CLASSIFIER_MODEL: process.env.WINDSURF_CLASSIFIER_MODEL ?? "gpt-5-mini",
  WINDSURF_ADVISOR_MODEL: process.env.WINDSURF_ADVISOR_MODEL ?? "gpt-5.3-codex",

  ANTIGRAVITY_API_KEY: process.env.ANTIGRAVITY_API_KEY ?? "",
  ANTIGRAVITY_BASE_URL: process.env.ANTIGRAVITY_BASE_URL ?? "",
  ANTIGRAVITY_CLASSIFIER_MODEL: process.env.ANTIGRAVITY_CLASSIFIER_MODEL ?? "gpt-5-mini",
  ANTIGRAVITY_ADVISOR_MODEL:
    process.env.ANTIGRAVITY_ADVISOR_MODEL ?? "gpt-5.3-codex",

  REQUEST_TIMEOUT_MS: Number(process.env.REQUEST_TIMEOUT_MS ?? "20000"),
  CLASSIFIER_TIMEOUT_MS: Number(process.env.CLASSIFIER_TIMEOUT_MS ?? "8000"),
  MAX_RETRIES: Number(process.env.MAX_RETRIES ?? "2"),
  MAX_PARALLEL_ADVISORS: Number(process.env.MAX_PARALLEL_ADVISORS ?? "3"),
  DEBUG: process.env.DEBUG === "1",
};

const InputSchema = z.object({
  request: z.string().min(1),
  problem_stage: z
    .enum([
      "exploration",
      "research_design",
      "implementation",
      "verification",
      "optimization",
    ])
    .default("exploration"),
  context: z
    .object({
      market: z.string().optional(),
      horizon: z.string().optional(),
      strategy_summary: z.string().optional(),
      changed_files: z.array(z.string()).optional(),
      diff: z.string().optional(),
      relevant_code: z.string().optional(),
      data_schema: z.string().optional(),
      constraints: z.array(z.string()).optional(),
      extra_context: z.string().optional(),
    })
    .default({}),
});

type Input = z.infer<typeof InputSchema>;

const AdvisoryModeSchema = z.enum([
  "idea_formulation",
  "hypothesis_refinement",
  "regime_design",
  "target_label_design",
  "feature_design",
  "feature_filtering",
  "model_design",
  "execution_design",
  "risk_design",
  "implementation_review",
  "validation_review",
  "performance_review",
  "bug_risk_review",
]);

type AdvisoryMode = z.infer<typeof AdvisoryModeSchema>;

const ClassifierOutputSchema = z.object({
  modes: z.array(AdvisoryModeSchema).min(1),
  rationale: z.string().min(1),
});

type ClassifierOutput = z.infer<typeof ClassifierOutputSchema>;

const AdvisorFindingSchema = z.object({
  severity: z.enum(["blocker", "major", "minor", "note"]),
  summary: z.string().min(1),
  findings: z.array(z.string()),
  required_checks: z.array(z.string()),
  actions: z.array(z.string()),
});

type AdvisorFindingPayload = z.infer<typeof AdvisorFindingSchema>;

type AdvisorKey =
  | "quant"
  | "afml"
  | "regime"
  | "feature_model"
  | "performance"
  | "reliability"
  | "bug_hunter";

type AdvisorFinding = AdvisorFindingPayload & {
  advisor: AdvisorKey;
};

const OutputSchema = z.object({
  modes: z.array(AdvisoryModeSchema),
  advisors_used: z.array(
    z.enum([
      "quant",
      "afml",
      "regime",
      "feature_model",
      "performance",
      "reliability",
      "bug_hunter",
    ])
  ),
  classifier_rationale: z.string(),
  summary: z.string(),
  blockers: z.array(z.string()),
  risks: z.array(z.string()),
  actions: z.array(z.string()),
  checks: z.array(z.string()),
  confidence: z.number().min(0).max(1),
  uncertainty: z.array(z.string()),
  findings: z.array(
    z.object({
      advisor: z.enum([
        "quant",
        "afml",
        "regime",
        "feature_model",
        "performance",
        "reliability",
        "bug_hunter",
      ]),
      severity: z.enum(["blocker", "major", "minor", "note"]),
      summary: z.string(),
      findings: z.array(z.string()),
      required_checks: z.array(z.string()),
      actions: z.array(z.string()),
    })
  ),
});

type Output = z.infer<typeof OutputSchema>;

function debug(...args: unknown[]) {
  if (ENV.DEBUG) {
    console.error("[quant-advisor]", ...args);
  }
}

function stableStringify(value: unknown): string {
  return JSON.stringify(value);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function fnv1a(input: string): string {
  let hash = 2166136261;
  for (let i = 0; i < input.length; i++) {
    hash ^= input.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(16);
}

function buildSharedContextPrefix(input: Input): string {
  const lines = [
    "QUANT_ADVISOR_SHARED_CONTEXT_V1",
    "",
    "REQUEST",
    input.request,
    "",
    "PROBLEM_STAGE",
    input.problem_stage,
  ];

  if (input.context.market) lines.push("", "MARKET", input.context.market);
  if (input.context.horizon) lines.push("", "HORIZON", input.context.horizon);
  if (input.context.strategy_summary) lines.push("", "STRATEGY_SUMMARY", input.context.strategy_summary);
  if (input.context.data_schema) lines.push("", "DATA_SCHEMA", input.context.data_schema);
  if (input.context.relevant_code) lines.push("", "RELEVANT_CODE", input.context.relevant_code);
  if (input.context.diff) lines.push("", "DIFF", input.context.diff);
  if (input.context.changed_files && input.context.changed_files.length > 0) {
    lines.push("", "CHANGED_FILES", stableStringify(input.context.changed_files));
  }
  if (input.context.constraints && input.context.constraints.length > 0) {
    lines.push("", "CONSTRAINTS", stableStringify(input.context.constraints));
  }
  if (input.context.extra_context) lines.push("", "EXTRA_CONTEXT", input.context.extra_context);

  return lines.join("\n");
}

function buildPromptCacheKey(input: Input): string {
  return `quant-advisor:${fnv1a(buildSharedContextPrefix(input))}`;
}

function withTimeout(parentSignal: AbortSignal | undefined, timeoutMs: number) {
  const controller = new AbortController();

  const onParentAbort = () => {
    controller.abort(parentSignal?.reason ?? new Error("Parent aborted"));
  };

  if (parentSignal) {
    if (parentSignal.aborted) {
      controller.abort(parentSignal.reason);
    } else {
      parentSignal.addEventListener("abort", onParentAbort, { once: true });
    }
  }

  const timeoutId = setTimeout(() => {
    controller.abort(new Error(`Timed out after ${timeoutMs}ms`));
  }, timeoutMs);

  const cleanup = () => {
    clearTimeout(timeoutId);
    if (parentSignal) {
      parentSignal.removeEventListener("abort", onParentAbort);
    }
  };

  return { signal: controller.signal, cleanup };
}

function isRetriableStatus(status: number): boolean {
  return status === 408 || status === 409 || status === 425 || status === 429 || status >= 500;
}

function retryAfterMs(headers: Headers): number | null {
  const value = headers.get("retry-after");
  if (!value) return null;

  const seconds = Number(value);
  if (!Number.isNaN(seconds)) return seconds * 1000;

  const dateMs = Date.parse(value);
  if (!Number.isNaN(dateMs)) {
    return Math.max(0, dateMs - Date.now());
  }

  return null;
}

async function fetchWithRetry(
  url: string,
  init: RequestInit,
  opts: {
    retries: number;
    timeoutMs: number;
    signal?: AbortSignal;
    tag: string;
  }
): Promise<Response> {
  let lastError: unknown;

  for (let attempt = 0; attempt <= opts.retries; attempt++) {
    const { signal, cleanup } = withTimeout(opts.signal, opts.timeoutMs);

    try {
      debug(`${opts.tag}: attempt ${attempt + 1} ${url}`);
      const res = await fetch(url, { ...init, signal });

      if (!res.ok && isRetriableStatus(res.status) && attempt < opts.retries) {
        const retryMs =
          retryAfterMs(res.headers) ??
          Math.min(1000 * 2 ** attempt, 8000) + Math.floor(Math.random() * 250);
        cleanup();
        debug(`${opts.tag}: retryable status ${res.status}, sleeping ${retryMs}ms`);
        await sleep(retryMs);
        continue;
      }

      cleanup();
      return res;
    } catch (err) {
      cleanup();
      lastError = err;

      if (signal.aborted) {
        throw err;
      }

      if (attempt >= opts.retries) {
        break;
      }

      const backoff =
        Math.min(1000 * 2 ** attempt, 8000) + Math.floor(Math.random() * 250);
      debug(`${opts.tag}: retrying after error in ${backoff}ms`, err);
      await sleep(backoff);
    }
  }

  throw lastError instanceof Error ? lastError : new Error(`${opts.tag}: request failed`);
}

function extractTextFromUnknownResponse(json: unknown): string {
  if (typeof json === "string") return json;

  if (json && typeof json === "object") {
    const obj = json as Record<string, unknown>;

    if (typeof obj.output_text === "string") return obj.output_text;

    if (Array.isArray(obj.output)) {
      const pieces: string[] = [];
      for (const item of obj.output) {
        if (!item || typeof item !== "object") continue;
        const content = (item as Record<string, unknown>).content;
        if (!Array.isArray(content)) continue;
        for (const c of content) {
          if (!c || typeof c !== "object") continue;
          const text = (c as Record<string, unknown>).text;
          if (typeof text === "string") pieces.push(text);
        }
      }
      if (pieces.length > 0) return pieces.join("\n");
    }

    const choices = obj.choices;
    if (Array.isArray(choices) && choices.length > 0) {
      const message = (choices[0] as Record<string, unknown>).message;
      if (message && typeof message === "object") {
        const content = (message as Record<string, unknown>).content;
        if (typeof content === "string") return content;
      }
    }
  }

  throw new Error("Could not extract text content from provider response");
}

function extractJsonObject(text: string): unknown {
  const trimmed = text.trim();

  try {
    return JSON.parse(trimmed);
  } catch {
    // continue
  }

  const firstBrace = trimmed.indexOf("{");
  const lastBrace = trimmed.lastIndexOf("}");
  if (firstBrace >= 0 && lastBrace > firstBrace) {
    const candidate = trimmed.slice(firstBrace, lastBrace + 1);
    return JSON.parse(candidate);
  }

  throw new Error("No JSON object found in model output");
}

async function mapBatches<T, R>(
  items: T[],
  batchSize: number,
  fn: (item: T) => Promise<R>
): Promise<R[]> {
  const out: R[] = [];
  for (let i = 0; i < items.length; i += batchSize) {
    const batch = items.slice(i, i + batchSize);
    const results = await Promise.all(batch.map(fn));
    out.push(...results);
  }
  return out;
}

type ProviderConfig = {
  name: "codex" | "windsurf" | "antigravity";
  kind: "openai_responses" | "openai_compatible_chat";
  baseUrl: string;
  apiKey: string;
  classifierModel: string;
  advisorModel: string;
};

function getProviderConfig(): ProviderConfig {
  switch (ENV.QUANT_PROVIDER) {
    case "codex":
      return {
        name: "codex",
        kind: "openai_responses",
        baseUrl: ENV.OPENAI_BASE_URL.replace(/\/$/, ""),
        apiKey: ENV.OPENAI_API_KEY,
        classifierModel: ENV.OPENAI_CLASSIFIER_MODEL,
        advisorModel: ENV.OPENAI_ADVISOR_MODEL,
      };

    case "windsurf":
      return {
        name: "windsurf",
        kind: "openai_compatible_chat",
        baseUrl: ENV.WINDSURF_BASE_URL.replace(/\/$/, ""),
        apiKey: ENV.WINDSURF_API_KEY,
        classifierModel: ENV.WINDSURF_CLASSIFIER_MODEL,
        advisorModel: ENV.WINDSURF_ADVISOR_MODEL,
      };

    case "antigravity":
      return {
        name: "antigravity",
        kind: "openai_compatible_chat",
        baseUrl: ENV.ANTIGRAVITY_BASE_URL.replace(/\/$/, ""),
        apiKey: ENV.ANTIGRAVITY_API_KEY,
        classifierModel: ENV.ANTIGRAVITY_CLASSIFIER_MODEL,
        advisorModel: ENV.ANTIGRAVITY_ADVISOR_MODEL,
      };

    default:
      throw new Error(`Unsupported QUANT_PROVIDER: ${ENV.QUANT_PROVIDER}`);
  }
}

async function callJsonModel<T>({
  provider,
  model,
  system,
  user,
  timeoutMs,
  signal,
  responseSchema,
  cacheKey,
  tag,
}: {
  provider: ProviderConfig;
  model: string;
  system: string;
  user: string;
  timeoutMs: number;
  signal?: AbortSignal;
  responseSchema: z.ZodSchema<T>;
  cacheKey?: string;
  tag: string;
}): Promise<T> {
  if (!provider.baseUrl) {
    throw new Error(`Missing base URL for provider ${provider.name}`);
  }

  if (!provider.apiKey) {
    throw new Error(`Missing API key for provider ${provider.name}`);
  }

  if (provider.kind === "openai_responses") {
    const res = await fetchWithRetry(
      `${provider.baseUrl}/responses`,
      {
        method: "POST",
        headers: {
          "content-type": "application/json",
          authorization: `Bearer ${provider.apiKey}`,
        },
        body: JSON.stringify({
          model,
          prompt_cache_key: cacheKey,
          input: [
            {
              role: "system",
              content: [{ type: "input_text", text: system }],
            },
            {
              role: "user",
              content: [{ type: "input_text", text: user }],
            },
          ],
          text: {
            format: {
              type: "json_object",
            },
          },
        }),
      },
      {
        retries: ENV.MAX_RETRIES,
        timeoutMs,
        signal,
        tag,
      }
    );

    const json = await res.json();

    const text =
      typeof json.output_text === "string"
        ? json.output_text
        : extractTextFromUnknownResponse(json);

    return responseSchema.parse(extractJsonObject(text));
  }

  const res = await fetchWithRetry(
    `${provider.baseUrl}/chat/completions`,
    {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${provider.apiKey}`,
      },
      body: JSON.stringify({
        model,
        temperature: 0.1,
        prompt_cache_key: cacheKey,
        response_format: { type: "json_object" },
        messages: [
          { role: "system", content: system },
          { role: "user", content: user },
        ],
      }),
    },
    {
      retries: ENV.MAX_RETRIES,
      timeoutMs,
      signal,
      tag,
    }
  );

  const json = await res.json();
  const text = extractTextFromUnknownResponse(json);
  return responseSchema.parse(extractJsonObject(text));
}

const CLASSIFIER_SYSTEM = `
You are a low-cost routing classifier for a quant research and coding advisor.

Classify the request into one or more advisory modes.

Use the minimum useful set of modes.
Do not activate implementation/performance/bug modes for high-level ideation unless the request clearly asks for code review, optimization, or breakage analysis.

Return strict JSON only:
{
  "modes": [...],
  "rationale": "..."
}

Allowed modes:
- idea_formulation
- hypothesis_refinement
- regime_design
- target_label_design
- feature_design
- feature_filtering
- model_design
- execution_design
- risk_design
- implementation_review
- validation_review
- performance_review
- bug_risk_review
`.trim();

function buildClassifierPrompt(input: Input): string {
  const lightContext: Record<string, unknown> = {
    has_diff: Boolean(input.context.diff),
    changed_files_count: input.context.changed_files?.length ?? 0,
    has_relevant_code: Boolean(input.context.relevant_code),
    has_data_schema: Boolean(input.context.data_schema),
  };

  if (input.context.market) lightContext.market = input.context.market;
  if (input.context.horizon) lightContext.horizon = input.context.horizon;
  if (input.context.constraints && input.context.constraints.length > 0) {
    lightContext.constraints = input.context.constraints;
  }

  return [
    "CLASSIFY_THIS_REQUEST",
    "",
    "REQUEST",
    input.request,
    "",
    "PROBLEM_STAGE",
    input.problem_stage,
    "",
    "LIGHT_CONTEXT",
    stableStringify(lightContext),
  ].join("\n");
}

async function classifyModes(
  provider: ProviderConfig,
  input: Input,
  signal?: AbortSignal
): Promise<ClassifierOutput> {
  const out = await callJsonModel<ClassifierOutput>({
    provider,
    model: provider.classifierModel,
    system: CLASSIFIER_SYSTEM,
    user: buildClassifierPrompt(input),
    timeoutMs: ENV.CLASSIFIER_TIMEOUT_MS,
    signal,
    responseSchema: ClassifierOutputSchema,
    cacheKey: `${buildPromptCacheKey(input)}:classifier`,
    tag: "classifier",
  });

  return {
    modes: [...new Set(out.modes)],
    rationale: out.rationale,
  };
}

type AdvisorDefinition = {
  key: AdvisorKey;
  rolePrompt: string;
  shouldRun: (input: Input, modes: AdvisoryMode[]) => boolean;
};

const ADVISORS: AdvisorDefinition[] = [
  {
    key: "quant",
    rolePrompt: `
You are a quant and financial advisor.

Focus on:
- financial meaning
- tradability
- execution realism
- PnL implications
- funding, fee, carry, and slippage assumptions
- hidden semantic changes in alpha logic or risk logic

Return JSON only:
{
  "severity": "blocker|major|minor|note",
  "summary": "...",
  "findings": ["..."],
  "required_checks": ["..."],
  "actions": ["..."]
}
`.trim(),
    shouldRun: (_input, modes) =>
      modes.some((m) =>
        [
          "idea_formulation",
          "hypothesis_refinement",
          "regime_design",
          "target_label_design",
          "feature_design",
          "model_design",
          "execution_design",
          "risk_design",
          "validation_review",
          "implementation_review",
        ].includes(m)
      ),
  },
  {
    key: "afml",
    rolePrompt: `
You are an AFML advisor.

Focus on:
- causality
- leakage prevention
- event framing
- label correctness
- purge/embargo requirements
- overfitting and false discovery risk
- temporal integrity of features and targets

Return JSON only:
{
  "severity": "blocker|major|minor|note",
  "summary": "...",
  "findings": ["..."],
  "required_checks": ["..."],
  "actions": ["..."]
}
`.trim(),
    shouldRun: (_input, modes) =>
      modes.some((m) =>
        [
          "hypothesis_refinement",
          "regime_design",
          "target_label_design",
          "feature_design",
          "feature_filtering",
          "model_design",
          "validation_review",
          "implementation_review",
          "risk_design",
        ].includes(m)
      ),
  },
  {
    key: "regime",
    rolePrompt: `
You are a market regime and crypto microstructure advisor.

Focus on:
- state definition logic
- regime boundaries
- transition logic
- funding/liquidity/perp-specific realism
- where and why the idea should fail

Return JSON only:
{
  "severity": "blocker|major|minor|note",
  "summary": "...",
  "findings": ["..."],
  "required_checks": ["..."],
  "actions": ["..."]
}
`.trim(),
    shouldRun: (_input, modes) =>
      modes.some((m) =>
        ["regime_design", "execution_design", "risk_design"].includes(m)
      ),
  },
  {
    key: "feature_model",
    rolePrompt: `
You are a feature engineering and modeling advisor.

Focus on:
- feature usefulness
- redundancy
- filtering logic
- target-feature-model alignment
- whether the idea belongs in feature, target, model, regime, or execution logic

Return JSON only:
{
  "severity": "blocker|major|minor|note",
  "summary": "...",
  "findings": ["..."],
  "required_checks": ["..."],
  "actions": ["..."]
}
`.trim(),
    shouldRun: (_input, modes) =>
      modes.some((m) =>
        [
          "feature_design",
          "feature_filtering",
          "model_design",
          "target_label_design",
          "hypothesis_refinement",
        ].includes(m)
      ),
  },
  {
    key: "performance",
    rolePrompt: `
You are a performance and numerical optimization advisor.

Focus on:
- vectorization
- rolling/window efficiency
- array layout
- memory copies
- hot path efficiency
- batchability
- unnecessary dataframe overhead

Only give strong opinions when implementation detail exists.

Return JSON only:
{
  "severity": "blocker|major|minor|note",
  "summary": "...",
  "findings": ["..."],
  "required_checks": ["..."],
  "actions": ["..."]
}
`.trim(),
    shouldRun: (input, modes) =>
      input.problem_stage === "optimization" ||
      modes.includes("performance_review") ||
      (modes.includes("implementation_review") &&
        Boolean(input.context.diff || input.context.relevant_code)),
  },
  {
    key: "reliability",
    rolePrompt: `
You are a reliability and correctness advisor.

Focus on:
- invariants
- compatibility
- edge cases
- downstream breakage
- failure handling
- maintainability

Return JSON only:
{
  "severity": "blocker|major|minor|note",
  "summary": "...",
  "findings": ["..."],
  "required_checks": ["..."],
  "actions": ["..."]
}
`.trim(),
    shouldRun: (_input, modes) =>
      modes.some((m) =>
        [
          "implementation_review",
          "validation_review",
          "execution_design",
          "risk_design",
        ].includes(m)
      ),
  },
  {
    key: "bug_hunter",
    rolePrompt: `
You are a bug hunter.

Focus on:
- off-by-one errors
- alignment bugs
- NaN propagation
- shape/dtype mismatches
- timezone/timestamp bugs
- silent behavioral regressions

Only give strong opinions when code context exists.

Return JSON only:
{
  "severity": "blocker|major|minor|note",
  "summary": "...",
  "findings": ["..."],
  "required_checks": ["..."],
  "actions": ["..."]
}
`.trim(),
    shouldRun: (input, modes) =>
      modes.includes("bug_risk_review") ||
      (modes.includes("implementation_review") &&
        Boolean(input.context.diff || input.context.relevant_code)),
  },
];

function getRelevantAdvisors(input: Input, modes: AdvisoryMode[]): AdvisorDefinition[] {
  const selected = ADVISORS.filter((advisor) => advisor.shouldRun(input, modes));

  if (selected.length === 0) {
    return ADVISORS.filter((a) => a.key === "quant" || a.key === "afml");
  }

  return selected;
}

function buildAdvisorPrompt(
  input: Input,
  modes: AdvisoryMode[],
  advisor: AdvisorDefinition,
  sharedPrefix: string
): string {
  return [
    sharedPrefix,
    "",
    "ADVISORY_MODES",
    stableStringify(modes),
    "",
    "ADVISOR_ROLE",
    advisor.rolePrompt,
    "",
    "INSTRUCTIONS",
    "- Review only from your assigned lens.",
    "- Be concise and concrete.",
    "- Use blocker only for serious issues.",
    "- required_checks must be testable.",
    "- actions should be implementation-ready.",
    "",
    "RETURN_ONLY_JSON",
  ].join("\n");
}

async function runAdvisor(
  provider: ProviderConfig,
  advisor: AdvisorDefinition,
  input: Input,
  modes: AdvisoryMode[],
  signal?: AbortSignal
): Promise<AdvisorFinding> {
  try {
    const payload = await callJsonModel<AdvisorFindingPayload>({
      provider,
      model: provider.advisorModel,
      system: "You are a quant research routing and advisory assistant.",
      user: buildAdvisorPrompt(input, modes, advisor, buildSharedContextPrefix(input)),
      timeoutMs: ENV.REQUEST_TIMEOUT_MS,
      signal,
      responseSchema: AdvisorFindingSchema,
      cacheKey: buildPromptCacheKey(input),
      tag: `advisor:${advisor.key}`,
    });

    return {
      advisor: advisor.key,
      ...payload,
    };
  } catch (err) {
    debug(`advisor ${advisor.key} failed`, err);
    return {
      advisor: advisor.key,
      severity: "note",
      summary: "Advisor unavailable or timed out.",
      findings: [],
      required_checks: [],
      actions: [],
    };
  }
}

function dedupe(values: string[]): string[] {
  return [...new Set(values.map((x) => x.trim()).filter(Boolean))];
}

function synthesize(
  input: Input,
  classifier: ClassifierOutput,
  findings: AdvisorFinding[]
): Output {
  const blockers = findings
    .filter((f) => f.severity === "blocker")
    .map((f) => `${f.advisor}: ${f.summary}`);

  const majorCount = findings.filter((f) => f.severity === "major").length;

  const risks = dedupe(findings.flatMap((f) => f.findings));
  const actions = dedupe(findings.flatMap((f) => f.actions));
  const checks = dedupe(findings.flatMap((f) => f.required_checks));
  const advisorsUsed = findings.map((f) => f.advisor);

  const uncertainty: string[] = [];

  if (
    classifier.modes.includes("implementation_review") &&
    !input.context.diff &&
    !input.context.relevant_code
  ) {
    uncertainty.push("Implementation review ran without diff or relevant_code.");
  }

  if (
    input.problem_stage === "implementation" &&
    (!input.context.changed_files || input.context.changed_files.length === 0)
  ) {
    uncertainty.push("changed_files missing for implementation stage.");
  }

  if (
    classifier.modes.some((m) =>
      ["feature_design", "target_label_design", "validation_review"].includes(m)
    ) &&
    !input.context.data_schema
  ) {
    uncertainty.push("Missing data schema.");
  }

  if (classifier.modes.includes("regime_design") && !input.context.market) {
    uncertainty.push("Missing market context.");
  }

  let confidence = 0.72;
  let summary = "Advisory review completed.";

  if (blockers.length > 0) {
    confidence = 0.42;
    summary = "Blocker-level issues were found.";
  } else if (majorCount > 0) {
    confidence = 0.6;
    summary = "Material risks were found; revise before proceeding.";
  } else if (actions.length === 0 && checks.length === 0) {
    confidence = 0.5;
    summary = "The request is still too high-level for strong implementation guidance.";
  }

  if (uncertainty.length > 0) {
    confidence = Math.max(0.3, Math.min(confidence, 0.58));
  }

  return OutputSchema.parse({
    modes: classifier.modes,
    advisors_used: advisorsUsed,
    classifier_rationale: classifier.rationale,
    summary,
    blockers,
    risks,
    actions,
    checks,
    confidence,
    uncertainty,
    findings,
  });
}

async function handleQuantAdvisor(
  input: Input,
  signal?: AbortSignal
): Promise<Output> {
  const provider = getProviderConfig();

  const classifier = await classifyModes(provider, input, signal);
  const selectedAdvisors = getRelevantAdvisors(input, classifier.modes);

  debug("provider", provider.name);
  debug("modes", classifier.modes);
  debug("selected advisors", selectedAdvisors.map((a) => a.key));

  const findings = await mapBatches(
    selectedAdvisors,
    ENV.MAX_PARALLEL_ADVISORS,
    (advisor) => runAdvisor(provider, advisor, input, classifier.modes, signal)
  );

  return synthesize(input, classifier, findings);
}

const server = new Server(
  { name: "quant-advisor", version: "2.1.0" },
  { capabilities: { tools: {} } }
);

server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "quant_advisor",
        description:
          "Unified quant advisor for alpha logic, AFML, implementation review, validation design, and optimization.",
        inputSchema: {
          type: "object",
          properties: {
            request: { type: "string" },
            problem_stage: {
              type: "string",
              enum: [
                "exploration",
                "research_design",
                "implementation",
                "verification",
                "optimization",
              ],
            },
            context: {
              type: "object",
              properties: {
                market: { type: "string" },
                horizon: { type: "string" },
                strategy_summary: { type: "string" },
                changed_files: { type: "array", items: { type: "string" } },
                diff: { type: "string" },
                relevant_code: { type: "string" },
                data_schema: { type: "string" },
                constraints: { type: "array", items: { type: "string" } },
                extra_context: { type: "string" },
              },
            },
          },
          required: ["request"],
        },
      },
    ],
  };
});

server.setRequestHandler(CallToolRequestSchema, async (req) => {
  if (req.params.name !== "quant_advisor") {
    throw new Error(`Unknown tool: ${req.params.name}`);
  }

  const input = InputSchema.parse(req.params.arguments ?? {});
  const result = await handleQuantAdvisor(input);

  return {
    content: [
      {
        type: "text",
        text: JSON.stringify(result, null, 2),
      },
    ],
  };
});

await server.connect(new StdioServerTransport());
