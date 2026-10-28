# AI Models Primer — From Model Names to Running Models Locally

A practical reference for understanding model names, architecture, quantization, benchmarks, inference settings, and the settings shown in LM Studio for **GPT-OSS 20B**.

> **Scope note:** Model-specific values and UI labels can change between model releases and LM Studio versions. Treat memory estimates as estimates, not guarantees. The values in the screenshot are recorded in the dedicated LM Studio section below.

## 1. What is an AI model?

An AI model is a learned set of numerical parameters that maps an input to an output. A language model typically receives tokens (pieces of text) and predicts the next token repeatedly.

- **Training:** parameters are adjusted using data so the model learns statistical patterns.
- **Inference:** the trained model is run to generate an answer.
- **Weights / parameters:** the learned numbers that encode what the model has learned.
- **Checkpoint:** a saved version of model weights, often accompanied by configuration and tokenizer files.
- **Runtime / inference engine:** software that loads the model and executes it, for example LM Studio's runtime.

A model is not the same thing as the application around it. The model generates predictions; the runtime handles loading, memory, batching, tokenization, and hardware execution; the chat application manages conversations and tools.

## 2. Reading a model name

Examples:

- `Qwen3-14B`
- `DeepSeek-R1-Distill-Qwen-14B`
- `gpt-oss-20b`
- `Qwen2.5-Coder-14B-Instruct-GGUF`
- `Qwen2.5-14B-Instruct-Q4_K_M.gguf`

Names often encode several clues:

| Name fragment | Typical meaning |
|---|---|
| `Qwen`, `Llama`, `GPT-OSS`, `DeepSeek` | Model family or publisher |
| `2.5`, `3`, etc. | Model generation / release family |
| `14B`, `20B`, `70B` | Approximate parameter count; B means billion |
| `Coder` | Optimized or trained for programming tasks |
| `Instruct` / `Chat` | Tuned to follow user instructions or conduct dialogue |
| `Base` | General pretrained checkpoint, not necessarily instruction-tuned |
| `Reasoning` / `R1` | A model or training recipe emphasizing multi-step reasoning |
| `Distill` | A smaller model trained to imitate outputs or behavior from a stronger teacher |
| `GGUF` | File format commonly used by llama.cpp-based local inference runtimes |
| `Q4_K_M`, `Q5_K_M` | Quantization scheme / weight representation |

These labels are useful hints, not universal guarantees. Always check the model card and the exact file you downloaded.

## 3. Parameters, model size, dense models, and MoE

### Parameters

A 14B model has roughly 14 billion learned parameters. A 20B model has roughly 20 billion. Parameter count is a rough measure of model capacity, not a direct measure of quality, speed, or memory use.

### Dense models

In a dense model, most of the model's parameters participate in each forward pass for a token. Larger dense models generally require more compute and memory.

### Mixture of Experts (MoE)

An MoE model contains multiple expert subnetworks and a routing mechanism. For a given token, the router activates only some experts.

- **Total parameters:** weights stored across all experts and other model components.
- **Active parameters:** the subset used for a token's computation.
- **Why it matters:** an MoE model may have a large total parameter count but compute per token closer to a smaller model. It still generally needs memory to store the experts that may be selected.

**GPT-OSS 20B** is an MoE model. Its approximately 21-billion total parameters are not all active for each token; the commonly stated active parameter count is around 3.6B per token. Do not confuse active parameters with the amount of weight memory needed to keep the model available.

## 4. Transformer basics

Most modern large language models use the Transformer architecture.

- **Tokenization:** text is split into tokens. A token can be a word, word fragment, punctuation mark, or other symbol.
- **Token embeddings:** each token is represented as a vector.
- **Attention:** lets the model use information from other tokens in the context.
- **Layers / blocks:** repeated computational stages that transform representations.
- **Feed-forward network / MLP:** processes representations within each layer; in MoE models, this may be implemented using experts.
- **Positional information / RoPE:** supplies information about token positions.
- **Output head:** converts the final representation into scores for possible next tokens.

### Context window

The **context window** is the maximum number of tokens the model/runtime can consider in a single request, including relevant prompt and conversation history and, depending on the application, generated tokens. A model may support a very large context in principle, while the runtime's selected context is much smaller.

A larger context can improve performance on long documents, but it increases memory use and can reduce speed. More context does not automatically mean better answers.

### Token count is not word count

Tokenization varies by language, vocabulary, and content. As a rough English estimate, one token is often around three-quarters of a word, but this is not a reliable conversion for code, numbers, or other languages.

## 5. Quantization and model file sizes

**Quantization** stores model weights using fewer bits than full-precision formats. It usually reduces memory requirements and may improve inference speed, at the cost of some possible quality loss.

Approximate *weight-only* storage before metadata and runtime overhead:

| Representation | Bits per value (roughly) | Approximate storage per 1B parameters |
|---|---:|---:|
| FP32 | 32 | 4 GB |
| FP16 / BF16 | 16 | 2 GB |
| INT8 / 8-bit | 8 | 1 GB |
| 6-bit | 6 | 0.75 GB |
| 5-bit | 5 | 0.625 GB |
| 4-bit | 4 | 0.5 GB |

These are simplified estimates. Actual quantized file sizes vary because quantization often uses blocks, scales, metadata, and mixed precision. Runtime memory is higher than the weight file size.

Common GGUF quantization labels include:

- **`Q4_K_M`:** a 4-bit K-quant family variant that uses mixed choices for different tensors to balance size and quality.
- **`Q5_K_M`:** a similar family with a higher bit budget, usually larger and potentially more accurate than Q4_K_M.
- **Q8 / 8-bit:** larger, often closer to higher-precision weights.
- **FP16:** substantially larger, with less quantization-related loss.
- **MXFP4:** a microscaling 4-bit floating-point format used by some model architectures / runtimes; it is not interchangeable with every GGUF Q4 format.

Do not compare quantization names as if every format were exactly the same number of bits for every tensor. Test the actual model on your workload.

## 6. Model variants and training terms

- **Base model:** pretrained to predict text, but may not reliably follow instructions.
- **Instruction-tuned model:** further trained on instruction/response examples.
- **Chat model:** tuned for conversational interactions, sometimes with a required chat template.
- **Reasoning model:** trained or configured to spend more computation/tokens on multi-step problem solving. Reasoning behavior does not guarantee correctness.
- **Distillation:** a smaller student learns from a teacher's outputs or behavior. It can retain useful capabilities but need not match the teacher.
- **Fine-tuning:** additional training on selected data to adapt behavior or knowledge.
- **SFT (supervised fine-tuning):** training on examples of desired inputs and outputs.
- **LoRA / QLoRA:** parameter-efficient fine-tuning methods that train a smaller set of adapter parameters rather than all base weights; QLoRA typically quantizes the base model during training.
- **Alignment / preference tuning:** methods that encourage outputs to better match desired behavior or preferences.
- **System prompt:** high-priority instructions supplied by the application.
- **Chat template:** formatting rules that serialize roles and messages into the exact token structure expected by the model. An incorrect template can noticeably harm output quality or tool use.

## 7. Benchmarks: what they tell you—and what they don't

Benchmarks measure performance on particular test sets. They do not provide a universal ranking for every real task.

| Benchmark | Broad area |
|---|---|
| MMLU / MMLU-Pro | Broad academic and professional knowledge |
| GPQA Diamond | Difficult science questions |
| AIME | Competition-style mathematics |
| GSM8K | Grade-school mathematical reasoning |
| HumanEval / MBPP | Programming problems |
| SWE-bench Verified | Resolving real software issues in repository environments |
| IFEval | Following verifiable instructions |
| BFCL | Tool/function-calling behavior |
| Long-context tests | Retrieval/reasoning over extended input |

Caveats:
1. Different model cards may use different benchmark versions, prompts, sampling settings, or evaluation harnesses.
2. A high benchmark score does not guarantee good performance on your own documents or codebase.
3. Some benchmarks may overlap with training data.
4. Small score differences may not be meaningful.
5. Compare models on a repeatable set of your own representative tasks as well.

## 8. Capabilities to distinguish

- **Reasoning:** solving tasks that require multiple dependent steps.
- **Tool / function calling:** producing structured requests for an application to execute tools. The model does not automatically have access to tools; the application must expose them and execute the calls.
- **Structured output / JSON mode:** helping constrain output to a schema or JSON format. Always validate the result.
- **Vision / multimodal input:** processing images, and sometimes audio or video, if the exact model and runtime support it.
- **Audio:** speech recognition, audio understanding, or speech generation, depending on model type.
- **Multilingual capability:** performance across languages; this can vary widely by language and task.
- **Embeddings:** vectors used to represent text or other content for semantic search, clustering, and retrieval. An embedding model is not the same as a chat-generation model.
- **RAG (retrieval-augmented generation):** retrieve relevant external material and include it in the prompt so the model can answer with that context. RAG does not itself guarantee factuality.
- **Agent:** an application pattern in which a model can plan, call tools, observe results, and repeat within defined permissions and limits.
- **MCP (Model Context Protocol):** a protocol for exposing tools and resources to compatible AI clients. It is a connection/protocol layer, not a model architecture.

## 9. How to read a model card

Check these fields before downloading or deploying a model:

1. Publisher and canonical model ID.
2. Intended use and limitations.
3. Architecture: dense or MoE, total and active parameters if relevant.
4. Training and instruction-tuning details.
5. Supported languages and modalities.
6. Context length and any conditions on using it.
7. License and usage restrictions.
8. Benchmark methods and reported results.
9. Hardware / memory requirements.
10. Supported runtime and file formats.
11. Quantization provenance and any known quality degradation.
12. Chat template, tool-calling format, and recommended inference settings.
13. Safety limitations and known failure modes.

## 10. Local inference: RAM, VRAM, and offloading

- **VRAM:** memory on the GPU. It is usually the fastest place to store model weights and inference state for GPU execution.
- **System RAM:** main memory. A runtime may keep some weights or other data in RAM and transfer or execute parts on the CPU.
- **GPU offload:** the number of layers (or equivalent model components) assigned to the GPU. More offload can improve speed but uses more VRAM.
- **CPU inference:** computes model operations on the CPU; often slower than a well-supported GPU path.
- **Unified memory:** some systems share a memory pool between CPU and GPU; the exact behavior depends on hardware and runtime.
- **Memory estimate:** an estimate of expected memory consumption under the selected settings. Real use can differ due to driver overhead, runtime buffers, context, batch sizes, concurrency, and other applications.

A model loading successfully does not guarantee that it will generate quickly. It may fit only by offloading parts to RAM/CPU, or it may run out of memory when a longer prompt or multiple concurrent requests are used.

## 11. LM Studio settings shown in the screenshot

This section explains the labels visible in the advanced settings for **GPT-OSS 20B**. The screenshot shows an estimated GPU use of **11.51 GB**, estimated total use of **12.01 GB**, a selected context length of **8192**, GPU offload **22**, CPU thread pool size **6**, evaluation batch size **2048**, physical batch size **512**, maximum concurrent predictions **4**, unified KV cache enabled, context checkpoints **32**, GPU KV-cache offload enabled, model kept in memory enabled, `mmap()` enabled, Flash Attention enabled, speculative decoding off, and number of experts **4**. These are the screenshot's current selections, not universal recommended values.

### Estimated Memory Usage (GPU / Total)

LM Studio's estimate of the memory required with the selected configuration. **GPU** is the estimated GPU portion; **Total** includes the broader estimated footprint. The estimate can change when context, offload, batching, cache, or concurrency settings change. Leave headroom for the operating system, display, and other GPU processes.

### Context Length

Maximum context tokens allocated for a request/session in this load configuration. The screenshot selects **8192 tokens**. The UI also says the model supports up to **131072 tokens**, which is the model's stated maximum supported context in that UI—not the context currently selected.

- Increasing context can help with long prompts/documents.
- It usually increases KV-cache memory and can affect speed.
- Choose a value based on actual prompt length; do not select the model's maximum just because it is available.
- If you get memory errors or slow generation, reduce context before changing many other settings.

### GPU Offload

How much of the model is assigned to GPU execution; the screenshot shows **22**. In LM Studio, this is commonly expressed as the number of layers offloaded, although exact behavior can depend on architecture/runtime.

- More GPU offload often improves generation speed if there is enough VRAM.
- If VRAM is insufficient, the model may fail to load or performance may deteriorate due to memory pressure.
- The best value is usually the highest stable setting that leaves memory headroom, not necessarily the maximum slider value.

### CPU Thread Pool Size

The number of CPU worker threads used by the runtime for CPU-side operations. The screenshot selects **6**.

- More threads can help when CPU work is a bottleneck.
- More threads do not always mean faster inference; memory bandwidth and contention matter.
- Start near the number of physical CPU cores available to the workload, then benchmark. Avoid assuming that logical-core count is always optimal.

### Evaluation Batch Size

The number of tokens processed together in an evaluation batch (typically prompt processing / prefill behavior in the runtime). The screenshot selects **2048**.

- Larger batches can improve prompt-processing throughput.
- They require more temporary memory and can make loading or prompt evaluation fail on constrained hardware.
- If memory use is high or long prompts fail, try reducing this value.

### Physical Batch Size

The physical chunk size used to execute a batch internally; the screenshot selects **512**. This can affect memory/throughput trade-offs and is related to, but not necessarily identical to, the logical evaluation batch size.

- Larger physical batches may improve throughput but use more memory.
- Smaller values can be more memory-friendly but may be slower.
- Keep the default unless you are diagnosing memory pressure or benchmarking.

### Max Concurrent Predictions (Experimental)

The maximum number of predictions / requests the runtime may process concurrently; the screenshot selects **4**.

- Higher concurrency can improve aggregate throughput for multiple users or parallel requests.
- It can increase memory requirements and contention, and may reduce the speed of an individual response.
- For a single-user local chat, start at **1** unless you have a clear reason to run multiple requests concurrently.
- Experimental features may change between LM Studio versions.

### Unified KV Cache (Experimental)

An experimental option for sharing/managing the key-value cache across concurrent sequences, where supported by the runtime. The screenshot has it **enabled**.

A KV cache stores attention keys and values for tokens already processed, avoiding recomputing them for every generated token. Unified-cache behavior and benefits depend on the runtime and model. If you see instability or unexpected memory behavior, compare with the default setting and consult the version-specific LM Studio documentation.

### Context Checkpoints

The screenshot sets **32**. Checkpoints relate to saving intermediate context/state so the runtime can manage or reuse parts of prompt processing, depending on the implementation. This is a runtime-specific optimization rather than a model capability. Its exact memory/performance behavior may vary by LM Studio version; leave it at default unless you are tuning a measured bottleneck.

### Reasoning Budget Message

A text field that can supply a reasoning-budget instruction/message for supported reasoning models. It is blank in the screenshot. It is not a universal model parameter, and its effect depends on the model's prompt format and training.

For GPT-OSS or other reasoning models, do not assume a custom message has an effect unless the model/runtime documentation says how it is interpreted. An instruction in the prompt is not a hard guarantee on token use or reasoning quality.

### RoPE Frequency Base and RoPE Frequency Scale

RoPE (Rotary Position Embedding) is a method used to encode token position information in attention.

- **Frequency base** and **frequency scale** are low-level positional-encoding parameters.
- The screenshot leaves both unchecked / **Auto**.
- Incorrectly overriding these can degrade position handling, especially for long contexts.
- Keep them at **Auto** unless the model's official configuration explicitly recommends a different value.

### Offload KV Cache to GPU Memory

The screenshot has this **enabled**. It places the attention KV cache in GPU memory where supported.

- This can improve generation speed.
- The cache grows with context length, batch/concurrency, and model architecture.
- If VRAM becomes the limiting factor, disabling GPU KV-cache offload may move pressure elsewhere but can reduce performance. Test rather than assume it will be better.

### Keep Model in Memory

The screenshot has this **enabled**. LM Studio keeps the loaded model resident after use, reducing the need to unload and reload it between interactions.

- Useful when repeatedly chatting with the same model.
- Uses memory even while idle.
- Disable it if you need to free resources for other models or GPU applications.

### Try `mmap()`

The screenshot has this **enabled**. Memory mapping allows a process to map model-file contents into its virtual address space rather than reading the entire file into a separate application buffer in the same way. The operating system manages pages as needed.

- Often helps loading behavior and memory efficiency for supported local model files.
- It is not the same as GPU offload and does not mean the entire model is stored in VRAM.
- Leave enabled unless troubleshooting a specific platform or filesystem issue.

### Seed / Random Seed

The screenshot's seed checkbox is unchecked and the UI says **Random Seed**. A seed initializes the random-number generator used during sampling.

- A fixed seed can make results more repeatable under the same model, prompt, runtime, and settings.
- Exact reproducibility is not guaranteed across different hardware, software versions, parallel execution paths, or nondeterministic kernels.
- A random seed allows different samples across runs.

### Number of Experts

The screenshot shows **4**. This is an MoE-related runtime setting for the number of experts used/selected in the supported model configuration. The exact interpretation is architecture- and runtime-specific. It should not be treated as a universal control for all models.

For GPT-OSS 20B, leave this at the model/runtime's intended default unless documentation explicitly recommends tuning it. Changing expert routing/selection settings can change behavior, performance, or compatibility.

### Number of layers for which to force MoE weights onto CPU (Experimental)

The screenshot sets this to **0**. This option can force MoE weights for a number of layers to remain on or be handled by the CPU, where supported.

- A nonzero value may reduce VRAM use.
- It can slow inference if CPU execution or transfers become a bottleneck.
- **0** means no layers are being forced to CPU through this particular setting.
- Use only when VRAM pressure requires it and compare performance.

### Speculative Decoding

The screenshot shows **Off**. Speculative decoding uses a draft model or other proposal mechanism to predict candidate tokens that a larger target model verifies. It can accelerate generation when the draft/verification arrangement is effective.

- It requires runtime/model support and sometimes an appropriate draft model.
- It can add complexity and may not help every model or workload.
- Leave off unless you have a supported configuration and can measure a benefit.

### Chat Template

The screenshot has a collapsible **Chat Template** section. The chat template converts conversation roles and content into the special token format the model expects.

A mismatch can cause poor instruction following, malformed outputs, or broken tool calls. Prefer the model's recommended template and avoid custom edits unless you know the expected format.

### Flash Attention

The screenshot has **Flash Attention enabled**. Flash Attention is an optimized attention implementation designed to reduce memory traffic and improve efficiency, especially for long sequences.

- It can improve speed and/or memory behavior on supported hardware and models.
- Support depends on architecture, backend, driver, and runtime.
- If you encounter errors or unusual output, test the default/disabled path to isolate the issue.

### K Cache Quantization Type and V Cache Quantization Type (Experimental)

The screenshot shows the K and V cache quantization checkboxes unchecked.

Attention stores keys (**K**) and values (**V**) for previously processed tokens in the KV cache. Quantizing these can reduce cache memory use, especially at long contexts or with concurrency, but may affect speed or quality depending on implementation.

- Leave disabled/default initially.
- Consider enabling only if KV-cache memory is a real constraint and your runtime supports the chosen format.
- Validate quality and speed on long prompts, not just short chats.

### Remember settings for `gpt-oss-20b`

The screenshot has **Remember settings** enabled. This asks LM Studio to remember the selected load settings for this model, so they can be reused next time.

### Show advanced settings

The screenshot has **Show advanced settings** enabled. This exposes the additional controls described above; it does not itself change the model's capabilities.

## 12. Suggested starting configuration for a single-user local setup

Use this as a cautious starting point, not a universal optimum:

1. **Context length:** 8192 is a reasonable initial test if your tasks fit within it.
2. **GPU offload:** use the highest stable level that leaves VRAM headroom; lower it if loading fails or the system becomes unstable.
3. **CPU threads:** begin around the number of physical cores you can spare, then benchmark.
4. **Evaluation / physical batch sizes:** keep defaults initially; reduce them if prompt processing runs out of memory.
5. **Concurrent predictions:** use 1 for one interactive user; increase only when serving parallel requests.
6. **Unified KV cache, context checkpoints, MoE CPU forcing, cache quantization:** leave at defaults unless documentation or measurements justify changing them.
7. **RoPE values:** leave on Auto unless official model configuration says otherwise.
8. **KV cache on GPU:** keep enabled if it fits; watch VRAM at longer contexts.
9. **Keep model in memory / mmap:** generally useful for repeated local use if memory allows.
10. **Flash Attention:** keep enabled when supported and stable.
11. **Speculative decoding:** leave off unless you have a supported draft setup.
12. Change **one setting at a time**, record the change, and compare speed, memory use, and output quality.

### How to measure

- Test the same representative prompts before and after a change.
- Include short prompts and long prompts.
- Record time to first token, generation tokens per second, prompt-processing time, peak GPU memory, and any errors.
- Repeat tests; one run can be noisy.
- A configuration that is fastest on a short prompt may be worse on a long-context task.

## 13. Inference settings and speed metrics

- **Temperature:** controls randomness; lower values tend to be more predictable, but do not guarantee correctness.
- **Top-p:** samples from a probability mass covering a chosen cumulative probability.
- **Top-k:** restricts sampling to the k most likely next tokens.
- **Max output tokens:** upper bound on generated output length.
- **Stop sequences:** strings or tokens that tell the runtime to stop generation.
- **Tokens per second (tok/s):** generation throughput; compare under similar settings and prompts.
- **TTFT (time to first token):** delay before the first generated token appears.
- **Latency:** total time to receive a response.
- **Throughput:** work completed per unit time, often important for multiple users.
- **Prefill / prompt processing:** processing input tokens before generation starts.
- **Decode / generation:** producing output tokens one by one.

Always distinguish prompt-processing speed from generation speed; they can respond differently to batch and attention settings.

## 14. RAG, fine-tuning, tools, and agents

### RAG workflow

1. Collect documents.
2. Parse and clean them.
3. Split them into meaningful chunks.
4. Generate embeddings.
5. Store vectors and metadata.
6. Retrieve relevant chunks for a question.
7. Build a prompt with the retrieved evidence.
8. Generate an answer and, where required, cite the source documents.
9. Evaluate retrieval and answer quality separately.

### When to use RAG versus fine-tuning

Use **RAG** when the model needs current, private, or source-grounded information that changes over time. Use **fine-tuning** when you need to teach a consistent response style, format, task behavior, or domain-specific transformation. They can be combined.

Fine-tuning is not a reliable way to keep a model's factual knowledge current. A fine-tuned model can still hallucinate.

### Tool use

A tool-enabled application typically:
1. Gives the model tool descriptions and schemas.
2. Receives a structured tool call from the model.
3. Validates and authorizes the call.
4. Executes the tool in application code.
5. Returns the result to the model.
6. Presents the final response.

Never treat a model-generated tool call as inherently safe. Validate arguments, enforce permissions, and restrict side effects.

## 15. Evaluating models for a real project

For a quantitative-research or data-engineering workflow, evaluate models on your actual tasks instead of choosing from model size or leaderboard position alone.

Build a small fixed test set covering:
- extracting facts from economic releases or PDFs;
- identifying dates, figures, units, and source passages;
- summarizing news without adding unsupported claims;
- producing schema-valid JSON;
- writing and reviewing Python/SQL;
- following tool schemas correctly;
- refusing to invent unavailable data;
- performance on long documents and tables.

Track:
- task correctness and factual grounding;
- source/citation correctness;
- JSON/schema validity;
- tool-call success rate;
- latency / TTFT and tokens per second;
- peak RAM/VRAM;
- failure rate and recovery behavior;
- cost per successful task (including retries and human review).

For a news-signal research pipeline, keep model evaluation separate from the trading hypothesis test. A model may extract news well without the extracted signal predicting returns after transaction costs.

## 16. Troubleshooting checklist

**Model will not load**
- Reduce context length.
- Reduce evaluation/physical batch size.
- Reduce GPU offload or free VRAM.
- Close other GPU-heavy applications.
- Check whether the model file and runtime backend are compatible.

**Model loads but is slow**
- Check GPU utilization and VRAM.
- See whether many layers are running on CPU.
- Try a shorter context and one concurrent prediction.
- Measure prompt-processing and token-generation speeds separately.
- Compare one setting at a time.

**Long prompts cause memory problems**
- Reduce context length, concurrency, or batch sizes.
- Check KV-cache memory.
- Consider supported KV-cache quantization only after a baseline test.

**Poor instruction following or broken tool calls**
- Verify the model is instruction-tuned.
- Check the chat template.
- Validate tool definitions and the runtime's supported calling format.
- Test with a small, deterministic example.

**Answers are confident but wrong**
- Provide source evidence through RAG.
- Require citations or structured extraction where appropriate.
- Validate numerical outputs in code.
- Evaluate against a gold set; do not rely on fluency as a quality metric.

## 17. Hands-on LM Studio CLI examples

LM Studio's CLI commands can vary by version. Check `lms --help` and the official documentation if an option is unavailable.

List downloaded models:

```bash
lms ls
```

Estimate memory for a model key (replace the placeholder with a key from your local list):

```bash
lms load --estimate-only <model_key>
```

Load a model with a selected context length and maximum GPU offload, if supported by your installed CLI version:

```bash
lms load <model_key> --gpu max --context-length 8192
```

Monitor NVIDIA GPU use on Linux:

```bash
watch -n 1 nvidia-smi
```

Use a repeatable prompt and compare the LM Studio UI's reported generation speed and system memory. The CLI flags above are examples; consult the installed version's help output for exact syntax.

## 18. Glossary

| Term | Meaning |
|---|---|
| AI / language model | A learned system that maps input tokens to predicted output tokens |
| Parameter | A learned numerical value in the model |
| Dense model | A model where most parameters are used for each token |
| MoE | Mixture of Experts; routes each token through selected expert subnetworks |
| Active parameters | Parameters used for a token's computation in an MoE model |
| Token | A text or code unit processed by the model |
| Context window | Maximum token context for a request |
| Weight | A learned model parameter |
| Quantization | Representing weights/cache with fewer bits to reduce memory |
| GGUF | Model-file format used by several local inference runtimes |
| VRAM | GPU memory |
| GPU offload | Assigning model computation/weights to GPU |
| KV cache | Cached attention keys and values for processed tokens |
| RoPE | Rotary positional embeddings |
| Flash Attention | An optimized attention implementation |
| Prefill | Processing input tokens before generation |
| Decode | Generating output tokens |
| TTFT | Time to first token |
| RAG | Retrieval-augmented generation |
| Embedding | Vector representation used for similarity/retrieval |
| SFT | Supervised fine-tuning |
| LoRA / QLoRA | Parameter-efficient fine-tuning approaches |
| Distillation | Training a student model from a teacher's outputs/behavior |
| Chat template | Formatting rules for model conversation tokens |
| Tool calling | Model output that requests an application to execute a tool |
| MCP | Model Context Protocol for connecting AI clients to tools/resources |
| Speculative decoding | Using proposed draft tokens that a target model verifies |
| mmap | Memory mapping a file into virtual memory |
| Seed | Random-number-generator initialization value |
| Batch size | Number of items/tokens processed together |
| Concurrency | Number of requests or sequences processed at once |

## 19. Quick model-card checklist

Before choosing a model, answer:

- What exact task will it perform?
- Is it base, instruct, reasoning, coder, vision, or embedding?
- Is it dense or MoE?
- How many total and active parameters does it have?
- Which quantized file am I using?
- What context length do I actually need?
- Does the license allow my intended use?
- Does my runtime support its architecture and chat template?
- Does it fit within memory with headroom?
- How accurate is it on my own fixed evaluation set?
- What are its latency, throughput, and failure rates?
- Does it support the tools or modalities my application needs?

## 20. Useful official references

- Hugging Face model hub and model cards: https://huggingface.co/models
- Hugging Face documentation: https://huggingface.co/docs
- LM Studio documentation: https://lmstudio.ai/docs
- LM Studio CLI documentation: https://lmstudio.ai/docs/cli
- llama.cpp project and GGUF-related documentation: https://github.com/ggml-org/llama.cpp
- OpenAI GPT-OSS documentation / model information: https://openai.com/index/introducing-gpt-oss/

## Final mental model

When looking at a local model, keep five questions separate:

1. **What can it do?** Architecture, training, modalities, and evaluation results.
2. **How much memory does it need?** Weights, context/KV cache, batching, concurrency, and runtime overhead.
3. **How quickly does it run?** Hardware, offload, kernels, prompt length, and inference settings.
4. **How should I interact with it?** Chat template, system prompt, sampling, structured output, and tools.
5. **Does it work for my use case?** Test it on real representative tasks and measure correctness, reliability, and speed.

No single model name, parameter count, benchmark score, or LM Studio setting answers all five.

