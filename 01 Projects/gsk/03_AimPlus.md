# AIM Plus GenAI V2 — Summary

This document is a consolidated guide to the GenAI V2 backend. The [documentation index](00_INDEX.md) links to the individual deep dives; this page focuses on how the service fits together and the details most useful when developing, debugging, or operating it.

## What the service does

The backend is an asynchronous FastAPI service for pharmaceutical and medical-insights questions. A request can use either private enterprise data or live web information. LangGraph orchestrates the work, specialized agents use retrieval and LLM tools, and SQL Server stores conversation threads, turns, and rolling summaries.

The main routes are:

- `POST /api/aim-service/genai-backend/v2/predict` — non-streaming prediction.
- `POST /api/aim-service/genai-backend/v2/predict/stream` — Server-Sent Events (SSE) progress and answer tokens.
- `POST /filter` — structured-filter search used to preview or prefetch documents.
- `GET /`, `/health`, `/ready`, and `/docs` — service metadata, health/readiness, and Swagger UI.

## Request lifecycle

Each prediction receives a thread ID (`ParentConversationId`) and a per-turn ID (`ConversationId`); both must be GUIDs for SQL Server. The service stores the incoming turn as `processing`, loads eligible prior turns and any summary, runs the graph, then saves the answer or error. Follow-up turns are sequenced within the thread. The thread's `isWebSearch` value is read from SQL, so the selected pipeline remains sticky across turns.

The graph's shared `GraphState` carries request data, routing flags, history, results, and errors. Large DataFrames stay in a session-scoped `DataContext`; the graph state carries reference handles to them instead.

### Internal-data path

1. Query processing uses Azure OpenAI to assess clarity, rewrite the question using context, and decompose it into sub-queries tagged with an intent and data-source type.
2. A clarification can be returned directly. Otherwise, the history checker tries a fuzzy cache match, then asks an LLM whether a simple question can be answered from the thread's history.
3. The supervisor processes each sub-query in turn. Azure AI Search retrieves structured `OTHER` records and/or unstructured `MUN` and `ADB` documents, limited by the selected context IDs. A structured filter request can provide prefetched rows and skip the structured search call.
4. The QA, summary, trends, or analytical agent invokes the matching RAG tools. RAG formats and deduplicates records, chunks them, maps over chunks, and reduces the results while retaining source citations.
5. The combine node joins a single result directly or synthesizes multiple/analytical results with source headers and citations preserved.

### Web-search path

The web agent uses Gemini with Google Search grounding. On follow-ups it can rewrite the question from conversation context, validates medical/pharma relevance, and applies only the optional `Date` filter. Gemini returns answer text and grounding metadata, which is formatted into inline citations and a Sources list. Exact-repeat cache hits and history-answerable follow-ups can avoid a new grounded search.

## Models, retrieval, and memory

- **Azure OpenAI (`gpt-4o`)** handles internal query processing, tool use, history judgments, synthesis, and some RAG tasks. Azure OpenAI authorization uses cached, proactively refreshed OAuth bearer tokens.
- **Google Gemini** handles grounded web search, web-history judgments, and selected large-context RAG tasks.
- **Azure AI Search** provides hybrid keyword/vector retrieval for the internal path. `contextIdList` determines which selected records or indexes are searched; structured filters may use the bundled filter wrapper to prefetch records.
- **Conversation memory** is stored in `gen_query_new` (thread), `gen_query_conversations` (one row per question/answer turn), and `gen_query_summaries` (rolling summary). Completed, non-error turns are eligible for history; cancelled and failed turns are excluded.
- **Caching** is scoped to the loaded thread history. Internal requests use RapidFuzz similarity (threshold 0.85); web requests require an exact normalized query match because web information changes over time. A separate history judge may answer follow-ups from earlier context.
- **Summarization** bounds long conversations using an approximate character-based token count. When the documented threshold is exceeded (about 80,000 tokens), older messages are compressed and the most recent three turns are retained alongside the summary.

## Streaming, errors, and concurrency

Streaming sends `metadata`, `status`, and `token` events, followed by completion. If a client disconnects, the graph task is cancelled to stop ongoing model work, and an independent cleanup updates the turn to `cancelled` so it is not loaded into future history.

Known and unexpected failures are normalized to error codes and safe user-facing messages. The error path still persists the turn and its error details. LLM calls use retry/backoff behavior, including provider rate-limit guidance; SQL writes retry transient stale-connection errors. A shared per-pod semaphore limits LLM calls to 20 concurrent requests, and RAG fan-out defaults to 2. The Azure Storage Queue manager is optional load-leveling infrastructure; the primary prediction routes invoke the graph directly.

## Configuration and operations

Configuration resolves from existing environment variables first, then Key Vault for missing mapped secrets, then code defaults. The Key Vault loader maps 13 required secrets, including AI Search, Azure OpenAI, Gemini endpoints, and SQL credentials. When configured, a Key Vault load failure aborts application startup. In AKS, prefer managed/workload identity; local development can supply values through ignored environment files.

Run the application from the `GenAI/` directory because prompt YAML paths are relative to the working directory. The documented Docker flow builds the V2 image for `linux/amd64`, injects `docker.env` at runtime, and maps a host port to container port 8000. Access to private SQL, AI Search, and model endpoints requires the corporate VPN for local end-to-end predictions. Kubernetes startup, readiness, and liveness probes use the health endpoints.

Logs are emitted to console and rotating files with thread and turn IDs. Use those IDs and `execution_path` to reconstruct a request's route and locate the underlying exception when the API reports a generic error.

## Important caveats and improvement areas

- The documented JWT handling decodes the bearer token and reads its `sub` claim without verifying the signature. This is a security gap that should be addressed before relying on the token for trusted identity.
- Gemini safety categories are documented as disabled; the architecture docs also identify missing content-safety screening, PII protection, and jailbreak defenses.
- Health checks verify configuration rather than proving every dependency is reachable. `/health` checks critical OpenAI and AI Search settings; `/ready` additionally checks index configuration.
- Token estimates for conversation memory are approximate, and caching is limited to the current thread. Evaluation/groundedness scoring, distributed tracing, adaptive quota-aware throttling, and consistent response/citation schemas are identified as areas to improve.
- The SQL `Role` column and legacy `save_conversation` path do not describe the active V2 turn shape; active turns store both query and answer in one row.

## Deep-dive guide

- **Architecture and request paths:** [Architecture](01_CHAT_BOT_ARCHITECTURE.md), [internal workflow](02_INTERNAL_WORKFLOW.md), [web workflow](03_WEB_SEARCH_WORKFLOW.md), [query processing](04_QUERY_PROCESSING.md), [supervisor](05_SUPERVISOR_ORCHESTRATION.md), [specialized agents](06_SPECIALIZED_AGENTS.md).
- **Memory and grounding:** [history and caching](07_HISTORY_AND_CACHING.md), [AI Search retrieval](08_AI_SEARCH_RETRIEVAL.md), [RAG generation](09_RAG_GENERATION.md), [filter API](10_FILTER_API.md), [SQL data flow](18_DATA_FLOW.md), [graph state](17_GRAPH_STATE.md).
- **Models and runtime behavior:** [LLM models](11_LLM_MODELS.md), [token management](12_TOKEN_MANAGEMENT.md), [prompts](13_PROMPTS.md), [streaming](14_STREAMING.md), [error handling](15_ERROR_HANDLING.md), [concurrency and queueing](16_CONCURRENCY_AND_QUEUEING.md).
- **Operations:** [configuration and secrets](19_CONFIG_AND_SECRETS.md), [logging and observability](20_LOGGING_OBSERVABILITY.md), [lifecycle and health](21_APP_LIFECYCLE_HEALTH.md), [running in Docker](22_RUNNING_IN_DOCKER.md).
