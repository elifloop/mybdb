
# 0. Metric Hierarchy & Summary

                    BUSINESS
                       │
              ┌────────┴────────┐
              │                 │
          Task Success       Cost / ROI
              │
       ┌──────┴───────┐
       │              │
   Answer Quality   UX Quality
       │              │
   ┌───┴────┐     ┌───┴─────┐
   │        │     │         │
Correct  Grounded  TTFT    E2E latency
   │        │       │         │
   └───┬────┘       └────┬────┘
       │                 │
    Retrieval          Inference
       │                 │
  Recall@K            tok/sec
  Precision@K         queue latency
  MRR                  GPU utilization
  NDCG                 batching
       │
   ┌───┴────┐
   │        │
Embedding  Reranking
   │        │
Recall@K   NDCG
MRR        MRR

| Category    | Metric                             | Remember this                               |
| ----------- | ---------------------------------- | ------------------------------------------- |
| UX          | **TTFT**                           | How quickly user sees something             |
| UX          | **P95 latency**                    | Tail user experience                        |
| UX          | **P99 latency**                    | Worst-case tail                             |
| UX          | **E2E latency**                    | Complete response time                      |
| Inference   | **Tokens/sec**                     | Generation speed                            |
| Inference   | **Queue latency**                  | Capacity bottleneck                         |
| Cost        | **Cost/request**                   | Unit economics                              |
| Cost        | **Tokens/request**                 | Prompt/output efficiency                    |
| Retrieval   | **Recall@K**                       | Did we retrieve the evidence?               |
| Retrieval   | **Precision@K**                    | Is retrieved content relevant?              |
| Retrieval   | **Hit@K**                          | Did we retrieve at least one useful result? |
| Retrieval   | **MRR**                            | How high was first relevant result?         |
| Retrieval   | **NDCG@K**                         | Overall ranking quality                     |
| Retrieval   | **Context precision**              | How much retrieved context is useful?       |
| Retrieval   | **Context recall**                 | Did we retrieve enough evidence?            |
| Generation  | **Faithfulness**                   | Is answer supported by context?             |
| Generation  | **Answer relevance**               | Did we answer the question?                 |
| Generation  | **Answer correctness**             | Is the answer actually right?               |
| Generation  | **Hallucination rate**             | How often do we invent things?              |
| Generation  | **Citation accuracy**              | Does citation support claim?                |
| Agent       | **Task completion rate**           | Did the agent accomplish the task?          |
| Agent       | **Tool selection accuracy**        | Did it choose right tool?                   |
| Agent       | **Tool success rate**              | Did tool execution work?                    |
| Agent       | **Trajectory length**              | How many steps to accomplish task?          |
| Agent       | **Loop/retry rate**                | Is agent getting stuck?                     |
| Reliability | **Error rate**                     | How often does system fail?                 |
| Reliability | **Timeout rate**                   | How often does system exceed SLA?           |
| Data        | **Drift**                          | Has production data changed?                |
| Safety      | **Jailbreak success rate**         | Can users bypass safeguards?                |
| Business    | **Task success / resolution rate** | Does AI actually create value?              |
# 1. LLM / Chatbot System Performance

|Metric|What it measures|High/Low is good?|What it tells you|Typical action if bad|
|---|---|--:|---|---|
|**TTFT — Time to First Token**|Time from request submission until first generated token|Lower|How quickly the user sees a response beginning|Reduce queueing, prompt size, model latency, retrieval latency; streaming; KV/prompt caching|
|**TTLT / E2E Latency**|Total request → complete response time|Lower|Overall user-perceived response time|Profile retrieval, generation, tool calls, post-processing|
|**Time per Output Token**|Generation time per token|Lower|How quickly the model generates|Smaller/faster model, quantization, batching, optimized inference|
|**Tokens/sec**|Generated tokens per second|Higher|Model inference throughput|vLLM/TGI, batching, quantization, GPU optimization|
|**Input tokens/sec**|Prompt processing speed|Higher|Prefill performance|Prompt reduction, prefix caching, optimized serving|
|**Queueing latency**|Time request waits before processing|Lower|Capacity/concurrency problem|Scale workers/GPUs, load balancing, batching|
|**P50 latency**|Median latency|Lower|Typical user experience|General performance baseline|
|**P95 latency**|95th percentile latency|Lower|Experience for slower users|Investigate tail latency|
|**P99 latency**|99th percentile latency|Lower|Worst-case production experience|Capacity, noisy neighbors, long prompts, slow tools|
|**Request throughput**|Requests processed per unit time|Higher|System capacity|Horizontal scaling, batching, concurrency|
|**Token throughput**|Tokens processed/generated per second|Higher|Infrastructure efficiency|Model/server optimization|
|**Concurrency**|Simultaneous active requests|Context-dependent|Load characteristics|Capacity planning|
|**Requests/sec (RPS)**|Incoming request rate|Context-dependent|Traffic/load|Autoscaling|
|**Error rate**|% failed requests|Lower|Reliability|Debug API/model/tool/infrastructure failures|
|**Timeout rate**|% requests exceeding timeout|Lower|Requests taking too long|Optimize slow path; increase capacity|
|**Cancellation rate**|% users abandoning requests|Lower|UX/perceived latency|Reduce TTFT/E2E latency|
|**Availability**|% time system is operational|Higher|Reliability|HA, failover, infrastructure|
|**SLA/SLO compliance**|% requests satisfying latency/reliability objective|Higher|Whether system meets contract|Capacity/performance/reliability work|
|**Cost/request**|Infrastructure/model cost per request|Lower|Unit economics|Smaller model, caching, prompt reduction|
|**Cost/token**|Cost per input/output token|Lower|Model economics|Model selection/routing|
|**GPU utilization**|GPU compute utilization|Usually higher|Whether hardware is being used efficiently|Batch/parallelize; but investigate memory/IO bottlenecks|
|**GPU memory utilization**|VRAM consumption|Context-dependent|Memory pressure|Quantization, model size, batching|
|**CPU utilization**|CPU consumption|Context-dependent|CPU bottleneck|Scale/optimize preprocessing|
|**Memory utilization**|RAM consumption|Lower/appropriate|Memory pressure/leaks|Optimize memory lifecycle|
|**Cache hit rate**|% requests served partly/full from cache|Higher|Effectiveness of caching|Improve cache key/prefix/cache policy|
|**Retry rate**|% requests requiring retry|Lower|Instability or transient failures|Fix dependency/API/network issues|
|**Rate-limit rate**|% requests rejected due to limits|Lower|Capacity/API quota issue|Increase limits or throttle intelligently|



# 2. LLM Generation / Output Metrics

|Metric|What it measures|Good direction|What it tells you|
|---|---|--:|---|
|**Output token count**|Number of generated tokens|Lower when possible|Verbosity/cost/latency|
|**Input token count**|Prompt size|Lower when possible|Prompt efficiency|
|**Total token count**|Input + output|Lower when possible|Cost and latency|
|**Length distribution**|Distribution of response lengths|Context-dependent|Detect unexpectedly verbose responses|
|**Completion rate**|% requests producing valid responses|Higher|Basic generation reliability|
|**JSON validity rate**|% outputs valid JSON|Higher|Structured output reliability|
|**Schema adherence**|% outputs matching required schema|Higher|Production integration quality|
|**Tool-call success rate**|Successful tool invocations|Higher|Agent reliability|
|**Tool-call accuracy**|Correct tool chosen/arguments|Higher|Agent decision quality|
|**Hallucination rate**|% outputs containing unsupported claims|Lower|Factual reliability|
|**Refusal rate**|% requests refused|Context-dependent|Safety/over-refusal behavior|
|**Appropriate refusal rate**|Correctly refused unsafe/out-of-scope requests|Higher|Safety alignment|
|**Citation accuracy**|% citations actually supporting claims|Higher|RAG answer grounding|
|**Citation completeness**|% claims with supporting citations|Higher|Coverage of evidence|
|**Answer relevance**|How directly response answers question|Higher|Helpfulness|
|**Answer correctness**|Whether answer is factually correct|Higher|Core task quality|
|**Instruction adherence**|Whether model followed instructions|Higher|Prompt/system reliability|
|**Consistency**|Similarity across repeated runs|Higher|Stability/reproducibility|
|**Toxicity rate**|Toxic outputs|Lower|Safety|
|**PII leakage rate**|Outputs exposing sensitive information|Lower|Privacy/security|
|**Bias/fairness metrics**|Performance disparity across groups|Lower disparity|Responsible AI|

# 3. RAG Retrieval Metrics

|Metric|What it measures|Good direction|Interpretation|
|---|---|--:|---|
|**Recall@K**|Whether relevant documents appear in top K|Higher|Can retrieval find the required evidence?|
|**Precision@K**|Fraction of top K results that are relevant|Higher|Are retrieved results actually useful?|
|**Hit Rate@K**|% queries where at least one relevant result appears in top K|Higher|Basic retrieval success|
|**MRR — Mean Reciprocal Rank**|How high the first relevant result appears|Higher|Rewards relevant result appearing near top|
|**NDCG@K**|Ranking quality accounting for graded relevance|Higher|Better than simple relevance/no relevance|
|**MAP — Mean Average Precision**|Precision across ranking positions|Higher|Overall ranking quality|
|**Recall@1**|Relevant result in first result|Higher|Very strict retrieval test|
|**Recall@5**|Relevant result in top 5|Higher|Common retrieval diagnostic|
|**Recall@10**|Relevant result in top 10|Higher|Common RAG benchmark|
|**Precision@5**|Relevance of top 5|Higher|Retrieval precision|
|**Context relevance**|Retrieved context relevance to query|Higher|Are we giving LLM useful context?|
|**Context precision**|Relevant chunks / retrieved chunks|Higher|Detect noisy retrieval|
|**Context recall**|Relevant information retrieved / relevant information available|Higher|Detect missing evidence|
|**Context utilization**|How much retrieved context contributes to answer|Higher|Detect irrelevant/ignored context|
|**Retrieval latency**|Query → retrieved documents|Lower|Search performance|
|**Embedding latency**|Query → embedding|Lower|Embedding infrastructure performance|
|**Reranking latency**|Retrieval → reranked results|Lower|Cost of advanced retrieval|
|**Index size**|Number/size of indexed vectors|Context-dependent|Infrastructure scale|
|**Vector search latency**|ANN search duration|Lower|Vector DB performance|
|**Duplicate rate**|Duplicate chunks retrieved|Lower|Poor chunking/indexing|
|**Empty retrieval rate**|Queries returning no results|Lower|Index/query problem|
|**Retrieval coverage**|% questions answerable using retrieved corpus|Higher|Knowledge-base completeness|

# 5. Hybrid Search Metrics

|Metric|What it tells you|
|---|---|
|BM25 Recall@K|Keyword retrieval effectiveness|
|BM25 Precision@K|Keyword retrieval relevance|
|Dense Recall@K|Semantic retrieval effectiveness|
|Hybrid Recall@K|Whether combined search finds more relevant documents|
|Fusion effectiveness|Whether combining lexical + semantic improves ranking|
|Reranker NDCG|Quality of reranked candidates|
|Reranker MRR|Position of first relevant result|
|Candidate recall|Whether reranker received relevant documents at all|
|Reranker precision|Whether reranker promotes relevant documents|
|Search latency|Search performance|
|Filter selectivity|How much metadata filtering reduces candidate space|
|Filtered recall|Whether filtering accidentally removes relevant documents|

# 6. Embedding Metrics

|Metric|What it measures|Why it matters|
|---|---|---|
|Embedding dimensionality|Vector dimensions|Memory/index size|
|Embedding latency|Time to generate vector|Query latency|
|Embedding throughput|Vectors/sec|Ingestion/query scalability|
|Cosine similarity|Angular similarity|Common semantic similarity measure|
|Dot-product similarity|Vector similarity|Used by many embedding systems|
|Euclidean distance|Geometric distance|Alternative distance measure|
|Recall@K|Relevant vector found in top K|Embedding/search quality|
|MRR|Rank of relevant vector|Ranking quality|
|NDCG|Graded ranking quality|Advanced ranking evaluation|
|Clustering quality|Semantic separation|Embedding-space quality|
|Duplicate similarity|Similarity between supposedly different documents|Detect bad embeddings/data|
|Embedding drift|Distribution change over time|Detect model/data changes|
|Cross-domain performance|Retrieval performance across domains|Detect embedding specialization|
|Multilingual retrieval accuracy|Cross-language retrieval|Important for multilingual systems|

# 7. Chunking Metrics

|Metric|Meaning|Problem detected|
|---|---|---|
|Chunk size|Tokens/characters per chunk|Too large/small context|
|Chunk overlap|Shared tokens between chunks|Missing cross-boundary context|
|Retrieval recall by chunk size|Recall vs chunking strategy|Optimal chunking|
|Chunk relevance|How relevant retrieved chunks are|Bad segmentation|
|Boundary completeness|Whether chunks contain complete concepts|Bad splitting|
|Duplicate chunk rate|Duplicate/near-duplicate chunks|Excessive overlap|
|Orphan rate|Chunks lacking useful context|Bad chunking|
|Metadata coverage|% chunks with useful metadata|Filtering quality|
|Context compression ratio|Original context / compressed context|Compression efficiency|

# 8. RAG Generation / Grounding Metrics

|Metric|Measures|Good direction|
|---|---|--:|
|**Faithfulness**|Whether answer is supported by retrieved context|Higher|
|**Groundedness**|Degree answer is grounded in evidence|Higher|
|**Answer relevance**|Whether answer addresses question|Higher|
|**Answer correctness**|Whether answer is actually correct|Higher|
|**Context relevance**|Whether retrieved context is useful|Higher|
|**Context precision**|Relevant retrieved context / retrieved context|Higher|
|**Context recall**|Relevant information successfully retrieved|Higher|
|**Citation correctness**|Citation supports associated claim|Higher|
|**Citation completeness**|Important claims have citations|Higher|
|**Hallucination rate**|Unsupported claims|Lower|
|**Abstention accuracy**|Correctly says “I don't know” when evidence is absent|Higher|
|**Grounded answer rate**|Answers supported by available evidence|Higher|

# 9. Advanced RAG / Agentic RAG Metrics

|Metric|What it measures|Why important|
|---|---|---|
|Query rewrite success rate|% rewritten queries improving retrieval|Determines value of query rewriting|
|Query expansion recall|Whether expansion finds missing evidence|Advanced retrieval quality|
|Multi-query recall|Recall across multiple generated queries|Detects retrieval improvements|
|Retrieval iterations|Number of retrieval loops|Efficiency|
|Retrieval success rate|% tasks finding sufficient evidence|Agent effectiveness|
|Tool selection accuracy|Correct tool selected|Agent reasoning|
|Tool argument accuracy|Correct parameters passed|Integration reliability|
|Tool success rate|Successful tool execution|Reliability|
|Tool latency|Time spent in tools|Agent latency|
|Planning success rate|Plans leading to correct outcome|Agent quality|
|Step success rate|Successful individual agent steps|Debugging|
|Task completion rate|% tasks successfully completed|Most important agent metric|
|Agent trajectory length|Number of steps|Efficiency|
|Average tool calls/task|Tool utilization|Cost/efficiency|
|Retry rate|Agent retries|Instability|
|Loop rate|Agent getting stuck in loops|Agent failure|
|Recovery rate|Successful recovery from tool/error failures|Robustness|
|Handoff success rate|Successful agent-to-agent transfers|Multi-agent reliability|
|State consistency|Correct state across steps|Long-running workflow quality|
|Memory retrieval accuracy|Correct memory retrieved|Agent personalization|
|Memory contamination rate|Incorrect memory affecting response|Reliability|
|Human escalation rate|Tasks requiring humans|Automation effectiveness|

# 10. Query-Rewriting / Advanced Retrieval Metrics

|Metric|Meaning|
|---|---|
|Original query Recall@K|Baseline retrieval|
|Rewritten query Recall@K|Retrieval after rewriting|
|Recall uplift|`Recall_rewritten - Recall_original`|
|Precision change|Whether rewriting adds noise|
|Query expansion rate|Number of generated search queries|
|Rewrite latency|Cost of query rewriting|
|Rewrite token cost|LLM cost|
|Retrieval diversity|Whether multiple queries retrieve different useful documents|
|Redundancy|Duplicate results across queries|
|End-to-end answer uplift|Whether retrieval improvements actually improve answers|

# 11. LLM-as-a-Judge Metrics

|Metric|What it means|
|---|---|
|Judge score|LLM evaluator's rating|
|Pairwise win rate|Model A preferred over B|
|Agreement with human labels|Judge reliability|
|Judge consistency|Same answer evaluated similarly|
|Inter-rater agreement|Agreement between evaluators|
|Cohen's Kappa|Agreement beyond chance|
|Krippendorff's Alpha|General annotation agreement|
|Position bias|Judge favoring first/second answer|
|Verbosity bias|Judge favoring longer responses|
|Judge calibration|Correlation with human evaluation|
|Judge false-positive rate|Bad answers incorrectly rated good|
|Judge false-negative rate|Good answers incorrectly rated bad|

# 12. Classification Metrics

|Metric|Formula / Meaning|When useful|
|---|---|---|
|**Accuracy**|Correct / total|Balanced classification|
|**Precision**|TP / (TP+FP)|False positives expensive|
|**Recall / Sensitivity**|TP / (TP+FN)|False negatives expensive|
|**Specificity**|TN / (TN+FP)|Negative class important|
|**F1**|Harmonic mean of precision/recall|Balanced tradeoff|
|**Fβ**|Weighted precision/recall|One error type more important|
|**ROC-AUC**|Ranking across thresholds|General binary classification|
|**PR-AUC**|Precision-recall curve area|Imbalanced datasets|
|**Log loss / Cross entropy**|Probability quality|Probabilistic models|
|**Brier score**|Probability calibration|Risk prediction|
|**MCC**|Balanced classification correlation|Severe imbalance|
|**Confusion matrix**|TP/TN/FP/FN|Error diagnosis|

# 13. Ranking / Recommendation Metrics

|Metric|What it measures|
|---|---|
|Precision@K|Relevant recommendations among top K|
|Recall@K|Relevant items retrieved in top K|
|Hit Rate@K|At least one relevant item in top K|
|MRR|Position of first relevant result|
|NDCG@K|Ranking quality with graded relevance|
|MAP@K|Average precision over ranking|
|CTR|Click-through rate|
|Conversion rate|% recommendations resulting in desired action|
|Coverage|% catalog/user space recommended|
|Diversity|Difference between recommended items|
|Novelty|How unexpected recommendations are|
|Serendipity|Useful unexpected recommendations|
|Revenue/user|Business impact|
|Engagement/session|User engagement|
|Churn reduction|Business impact|

# 14. Speech / Multimodal AI Metrics

|Metric|Meaning|
|---|---|
|WER|Word Error Rate|
|CER|Character Error Rate|
|SER|Sentence Error Rate|
|Speaker diarization error rate|Speaker identification quality|
|VAD accuracy|Voice activity detection|
|Audio latency|Input → output latency|
|Vision detection mAP|Object detection quality|
|IoU|Bounding-box overlap|
|CLIP similarity|Image-text alignment|
|OCR accuracy|Text extraction quality|
|Image-groundedness|Whether answer reflects image|
|Multimodal task accuracy|Overall task correctness|

# 15. Model Training Metrics

|Metric|What it tells you|
|---|---|
|Training loss|Whether model learns training objective|
|Validation loss|Generalization|
|Training/validation gap|Overfitting|
|Perplexity|Language-model uncertainty|
|Learning rate|Optimization behavior|
|Gradient norm|Training stability|
|Gradient clipping rate|Exploding-gradient control|
|GPU utilization|Training efficiency|
|Samples/sec|Training throughput|
|Tokens/sec|LLM training throughput|
|Training cost|Economics|
|Epoch time|Training efficiency|
|Convergence rate|Optimization efficiency|
|Checkpoint quality|Best model selection|

# 16. Fine-Tuning / LoRA Metrics

|Metric|What to monitor|
|---|---|
|Training loss|Learning|
|Validation loss|Generalization|
|Task accuracy|Actual task performance|
|Exact-match|Structured output correctness|
|JSON validity|Output reliability|
|Schema accuracy|Structured integration|
|Instruction adherence|Following desired behavior|
|Catastrophic forgetting|Retention of general capabilities|
|Base-vs-finetuned delta|Whether fine-tuning helped|
|Parameter count|Model adaptation size|
|Trainable parameters|LoRA/PEFT efficiency|
|VRAM consumption|Fine-tuning infrastructure|
|Training time|Cost|
|Inference latency|Production impact|
|Adapter size|Deployment/storage cost|

# 17. Data Quality Metrics

|Metric|What it tells you|
|---|---|
|Missing-value rate|Data completeness|
|Duplicate rate|Data cleanliness|
|Null rate|Missing information|
|Outlier rate|Abnormal observations|
|Schema violation rate|Data contract problems|
|Label consistency|Annotation quality|
|Label noise|Incorrect training labels|
|Class imbalance|Distribution problem|
|Data freshness|Staleness|
|Distribution drift|Production data changed|
|Concept drift|Relationship between features/target changed|
|Training-serving skew|Offline vs production difference|
|Feature drift|Feature distribution changed|
|Population Stability Index (PSI)|Distribution shift|
|KL divergence|Distribution difference|
|Jensen-Shannon divergence|Symmetric distribution difference|
|Wasserstein distance|Distribution shift|
|Data coverage|Whether important cases are represented|
|Slice coverage|Representation across important segments|

# 18. Model Drift / Production Monitoring

|Metric|What it detects|
|---|---|
|Feature drift|Input distribution changed|
|Prediction drift|Output distribution changed|
|Embedding drift|Semantic input distribution changed|
|Performance drift|Accuracy deteriorating|
|Error-rate drift|Increasing failures|
|Calibration drift|Probabilities becoming unreliable|
|Data freshness|Stale source data|
|Ground-truth delay|How long before performance can be measured|
|Slice performance|Model degradation for specific segments|
|PSI|Distribution shift|
|KL divergence|Distribution divergence|
|JS divergence|Stable symmetric divergence|
|Wasserstein distance|Distribution movement|

# 19. Safety / Security Metrics

|Metric|What it measures|
|---|---|
|Prompt injection success rate|% attacks bypassing instructions|
|Jailbreak success rate|% safety bypasses|
|Toxicity rate|Harmful generation|
|PII leakage rate|Sensitive data exposure|
|Secret leakage rate|Credentials/system information exposure|
|Unsafe tool-call rate|Dangerous agent actions|
|Policy violation rate|Safety violations|
|False refusal rate|Legitimate requests incorrectly rejected|
|Unsafe compliance rate|Harmful requests incorrectly fulfilled|
|Data exfiltration rate|Information extraction attacks|
|Indirect prompt injection success|Malicious retrieved content influencing model|
|Guardrail bypass rate|Guardrail effectiveness|

# 20. Evaluation Dataset Metrics

|Metric|Meaning|
|---|---|
|Dataset size|Number of evaluation examples|
|Slice coverage|Coverage of important scenarios|
|Label quality|Human annotation quality|
|Inter-annotator agreement|Human consistency|
|Difficulty distribution|Easy/medium/hard balance|
|Long-tail coverage|Rare but important cases|
|Adversarial coverage|Robustness testing|
|Multilingual coverage|Language coverage|
|Temporal coverage|Old/new data|
|Domain coverage|Business/domain coverage|
|Leakage rate|Test examples appearing in training|
|Duplicate rate|Duplicate evaluation examples|

# 21. Business / Product Metrics

|Metric|Technical interpretation|Business interpretation|
|---|---|---|
|Task completion rate|Agent succeeds|Users accomplish goal|
|Resolution rate|AI solves issue|Reduced support workload|
|Escalation rate|AI fails/uncertain|Human support cost|
|CSAT|User satisfaction|Product quality|
|NPS|User loyalty|Business value|
|Adoption rate|Usage|Product acceptance|
|DAU/MAU|Engagement|Product stickiness|
|Retention|Continued usage|Long-term value|
|Conversion rate|Desired action|Revenue|
|Revenue/session|Monetization|Financial impact|
|Cost/task|AI operational cost|Unit economics|
|ROI|Benefit/cost|Business justification|
