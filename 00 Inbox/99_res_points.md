# Section 1 - Points

- **Architected a token-level streaming inference pipeline** for a production RAG/agentic LLM system using FastAPI Server-Sent Events (SSE) and `asyncio.Queue`-based producer–consumer decoupling, cutting perceived response latency (time-to-first-token) from multiple seconds to near-instant.

- **Built a provider-agnostic streaming abstraction** over Azure OpenAI (`stream=True`) and Google Gemini (`streamGenerateContent`), normalizing heterogeneous chunk formats behind a single interface so the multi-agent graph consumes tokens without provider-specific code.

- **Designed a client-disconnect cancellation mechanism** that propagates `asyncio` cancellation into in-flight LLM HTTP calls to abort generation on abandonment, **reducing wasted token/compute spend** while safely persisting partial state to the database.

- **Integrated streaming into a LangGraph multi-agent orchestration flow**, threading a shared event queue through graph state to emit real-time `status`, `token`, and `citation` events, giving users progressive feedback across retrieval, reasoning, and synthesis stages.

- **Hardened the streaming transport for production**, adding SSE keep-alive heartbeats and anti-buffering headers (`X-Accel-Buffering: no`) to survive proxy/load-balancer idle timeouts, plus a sentinel-based completion protocol for deterministic stream termination.

- **Deployed a production LLM inference service on Azure Kubernetes Service (AKS)**, containerizing a FastAPI application with a hardened, multi-stage Docker image (non-root user, dropped Linux capabilities, `seccompProfile: RuntimeDefault`) aligned to container security best practices.

- **Implemented keyless authentication with Azure Workload Identity + Managed Identity**, eliminating stored credentials in the cluster and letting pods securely access Azure OpenAI, Key Vault, and SQL without secrets in code or environment.

- **Configured Horizontal Pod Autoscaling (2–8 replicas)** on CPU/memory targets with tuned scale-up/scale-down stabilization windows, enabling the GenAI backend to elastically absorb variable LLM traffic while controlling compute cost.

- **Engineered zero-downtime rolling deployments** with `RollingUpdate` (25% maxSurge/maxUnavailable) plus three-tier startup/readiness/liveness probes on a `/health` endpoint, ensuring traffic only reaches fully-initialized pods during releases.

- **Built a multi-environment Azure DevOps CI/CD pipeline (DEV/UAT/PROD)** with automated pre-deployment validation, token-replacement config templating, and gated environment approvals — auto-deploying to DEV on successful builds while requiring manual promotion to production.

- **Tuned Kubernetes and Uvicorn for long-lived streaming (SSE) workloads**, setting extended keep-alive/graceful-shutdown timeouts and disabling proxy buffering so token-by-token LLM responses survive load-balancer idle limits without dropped connections.

- Built an end-to-end RAG pipeline over multilingual healthcare insights—denormalizing SQL star-schema data into retrieval-ready documents, generating Azure OpenAI text-embedding-3-small (1536-dim) embeddings with tiktoken-based tokenization/context management and Unicode normalization, and serving hybrid (vector + BM25) retrieval via an Azure AI Search HNSW/cosine ANN index with RRF fusion—scaled through Dask parallel batch inference, delta/incremental embedding, checkpointing, and OAuth2 token-refresh with rate-limit backoff, and enriched by upstream NER, GPT-3.5 summarization, and aspect-based sentiment as filterable metadata.

- Developed RAG pipeline using NER to extract biomedical entities using azure-ai-textanalytics & scispacy, running aspect-based sentiment alongside multi-class classification using a fine-tuned BERT model, and reconciling model-predicted vs. gold entities through fuzzy matches (fuzzywuzzy), with the resulting entities and sentiment surfaced as filterable metadata in an Azure AI Search vector index.
  
- Built a robust, end-to-end transfer-learning pipeline using BERT (Hugging Face Transformers) for enterprise multiclass text classification—standardizing automated data preprocessing, label indexing, class-balance mitigation, and rigorous train/validation/test evaluation methodologies while ensuring reproducible model artifact versioning and dependency compatibility.

- Architected an enterprise RAG + multi-agent chatbot used by medical business teams to deliver high-trust answers via inline source citations leveraging Glean, brandwatch social listening etc. using orchestrating retrieval, query decomposition, intent classification, and answer synthesis via LangGraph agent workflows, with real-time token-by-token response streaming (FastAPI Server-Sent Events) and automatic cross-provider fallback between Azure OpenAI (GPT) and Google Gemini with exponential-backoff retries (Tenacity), semaphore-based concurrency control/backpressure, timeout handling.

# Section 2 - Example JDs 

## eBay

About the job

Job DescriptionAs an LLM Inference Engineer on our AI Platform team, you’ll remove the compute-scaling bottleneck for production LLMs. Your job is to make frontier-model inference fast, efficient, reliable, and observable—the “last mile” from GPUs to APIs that products depend on. This role sits at the intersection of HPC, GPU systems, and MLOps, and requires strong intuition for how model architecture, runtimes, and hardware interact.What You’ll Do
Own production inference: Take models from handoff to production-grade serving, including release engineering, capacity planning, cost optimization, and incident response.
Tune inference performance: reduce end-to-end latency and increase throughput across real production traffic patterns.
Optimize runtimes and servers: Scale inference across heterogeneous GPU fleets; optimize stacks such as vLLM, Triton, and related components (e.g., schedulers, KV cache, batching, memory).
Benchmark and measure: Build benchmarking suites, metrics, and tooling to quantify latency, throughput, GPU utilization, memory, and cost.
Reliability and observability: Improve monitoring, tracing, and alerting; participate in incident response and postmortems to harden systems.
Apply and ship new optimizations: Evaluate research and implement pragmatic inference optimizations (e.g., quantization, paging, kernel/runtimes improvements).
Partner with cross-functional teams: Work with data science and product teams to translate business requirements into performance and availability SLOs.
What We’re Looking For
8+ years of strong development experience
Experience deploying and operating LLM inference services in production.
Strong production coding skills in Python plus Go or Rust (systems-level implementation and debugging).
Experience with ML frameworks and runtimes: PyTorch, vLLM, SGLang (and/or TensorRT).
Knowledge of GPU architecture and performance (profiling, memory bandwidth/latency tradeoffs); CUDA/kernel programming is a strong plus.
Solid understanding of LLM inference and optimization techniques: continuous batching, KV cache management, quantization, speculative decoding (nice-to-have), etc.
3+ years hands-on experience in performance optimization and systems programming for AI/ML workloads.
Demonstrated ability to deliver measurable production improvements (e.g., 2X throughput, lower p95/p99 latency, reduced GPU cost).
Proven skill in root-cause analysis: finding bottlenecks across model, runtime, networking, and infrastructure.
Demonstrated proficiency in applying autonomous AI coding agents to speed up software delivery pipelines. This includes advanced prompting and careful human-in-the-loop code review to improve development speed and code accuracy.


## Sarvam

About the job
About Sarvam

Sarvam is building the bedrock of Sovereign AI for India. The company is developing India’s full-stack sovereign AI platform, building across research, models, infrastructure and applications with a singular focus on making AI genuinely work for India. Sarvam works with leading enterprises and public institutions and is backed by Lightspeed, Peak XV, and Khosla Ventures. Sarvam partners with India’s leading brands, including Tata Capital, SBI Life, CRED, IDFC, and LIC.

About The Role

You will own the infrastructure that our next family of foundational models is trained on. This is a deep systems role: distributed training at scale, parallelism strategies, GPU kernel work, throughput and stability optimization, and the unglamorous reliability engineering that determines whether a frontier training run actually finishes.

The bar is high. A bad MFU number, a slow data loader, a switch flap — these are problems whose cost is measured in weeks and millions of dollars. You will be expected to find them before they happen, and to fix them fast when they do.

What You’ll Do

Build, maintain, and continually push the limits of our distributed training stack across large GPU clusters. 
Design and implement parallelism strategies — data, tensor, pipeline, sequence, expert — and reason about which combinations make sense for which architectures and which scales. 
Profile and optimize end-to-end training throughput: kernel performance, communication overlap, memory layout, checkpointing, data loading. 
Write and tune custom GPU kernels (CUDA, Triton) where off-the-shelf is leaving performance on the table. 
Own the reliability of long-running training jobs — fault tolerance, checkpoint integrity, deterministic restarts, automated detection of slow nodes and silent corruption. 
Partner closely with researchers to make sure architectural ideas can actually be trained efficiently, and with the data team to keep the pipeline from being the bottleneck. 

What We’re Looking For

BS or MS in Computer Science or a closely related technical field (or equivalent demonstrated experience). 
3+ years of experience building ML training infrastructure or large-scale distributed systems. Exceptional early-career candidates with a strong systems background will be considered. 
Hands-on experience training large models with distributed training frameworks — Megatron-LM, DeepSpeed, FSDP, NeMo, or equivalent. You should have been on-call for a real pretraining run. 
Deep working knowledge of GPU architecture, the CUDA programming model, and standard tools for profiling GPU workloads (Nsight, PyTorch profiler, etc.). 
Strong PyTorch internals knowledge. You're comfortable reading and modifying low-level training code, not just using high-level APIs. 
Meaningful open-source contributions in the training infrastructure ecosystem — Megatron, DeepSpeed, PyTorch, vLLM, Triton, NCCL, or similar. 

Bonus Points

Custom CUDA or Triton kernel development with measurable wins on real workloads. 
Direct experience training models at 10B+ parameters or on 1000+ GPU clusters. 
Familiarity with cluster orchestration and job scheduling (Slurm, Kubernetes) at scale. 
Experience with mixed precision (BF16, FP8), quantization-aware training, or other numerical work that has shown up in production training runs. 
First-author papers or technical reports on training systems, scaling, or model efficiency. 

Why this role?

Frontier model training is one of the hardest systems problems in the world right now, and most of the people doing it well sit at a small handful of labs. You will be doing the same work, with the same autonomy, and your impact will be visible in every model Sarvam ships.
