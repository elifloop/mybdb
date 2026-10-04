# RAG

## Ingestion side

- dedup + preprocess - alternate - dont dedup now ; do it later at retrieval using *Maximum Marginal Relevance* - early dedup saves cost etc.
- chunking strategies - fixed size vs semantic vs late chunking - late chunking was popularized by anthropic to solve `lost in the middle` problem where an isolated chunks looses its meaning so you just label a chunk by passign the whole document to a large context before hand. Example - prepends a short, generated summary of the parent document to every individual chunk so that a standalone chunk like "The revenue grew by 4%." becomes "This chunk is from Q3 Financial Report of Acme Corp: The revenue grew by 4%."
- models - dense (complete vectors like e.g., OpenAI text-embedding-3, Voyage-3, BGE-M3) Vs Sparse / Lexical Embeddings (BM25, SPLADE) - sparse embeddings which are based on some advanced tfidf can help with `out of vocab` prob

#### Scenario-Based Staff Interview Questions

###### Scenario A

"Our enterprise RAG system has 50 million documents, and our daily ingestion pipeline costs thousands of dollars and takes 8 hours to run. The business wants real-time document updates. How do you redesign the preprocessing layer?"

Staff-Level Answer Strategy:

Move from Batch to Event-Driven (CDC): Implement Change Data Capture (CDC) or webhook connectors (e.g., Kafka/Debezium) so documents are processed incrementally (only when modified or created), rather than re-ingesting the whole corpus.

Tiered Embeddings / Lazy Processing: Don't run expensive semantic chunking or heavy embedding calls on documents nobody reads. Use lightweight heuristics or lazy-load chunking for cold data, reserving heavy vectorization for hot paths.

Asynchronous Ingestion Queues: Decouple parsing (CPU-bound) from embedding generation (Network/API-bound) using a worker pool (e.g., Celery/Temporal) with backpressure handling.

###### Scenario B

"Users complain that chunks retrieved from long PDF manuals lack context. The LLM gets a chunk saying 'Turn the valve clockwise,' but doesn't know which machine it refers to."

Staff-Level Answer Strategy:

Diagnose: This is the classic context-fragmentation failure mode of naive fixed-size chunking.

Propose Mitigation: Implement Parent-Child Chunking (retrieve small child chunks for precise vector matching, but feed the broader parent chunk/section to the LLM) or Contextual Retrieval (injecting document-level summaries into chunk headers).

#### Metrics

- Chunk Size & Overlap Ratio: Measured in tokens (e.g., 512 token size with a 10% overlap). Determines the balance between context granularity and boundary fragmentation.
- Ingestion Throughput: Measured in documents-per-second or tokens-per-second during bulk processing.
- Deduplication Rate: Percentage of duplicate or near-duplicate documents successfully filtered out prior to embedding generation to measure storage and cost savings.
- Embedding Drift: Measure of distributional shift in vector space when updating or mixing embedding models over time.

## Retrieval side 

- query rewriting - users write poor queries - use HyDE why hypothetically guesses the answer and recomputes the embedding
- hybrid search - use fusion mechanism like RRF which has standard retrieval score OR weighted score fusion using learnable param
- rerankers - Cross-Encoders vs. Bi-Encoders -
  - Bi-Encoder (Retrieval): The standard vector search. Documents and queries are embedded independently into vectors, and cosine similarity is computed. Fast ($O(N)$ lookup via index), but sacrifices fine-grained interaction between query and text tokens.
  - Cross-Encoder (Reranking): The query and the retrieved document chunk are fed together into a transformer model at the same time. The model evaluates them jointly. Highly accurate, but computationally expensive ($O(K)$ where $K$ is the number of candidate chunks, e.g., top 50).
- vector db indexing algo - HNSW vs. IVF-
  - HNSW - How it works: A multi-layer graph structure where searching navigates from coarse global layers down to fine local layers. - Pros: Blazing fast query latency, extremely high recall ($>98\%$). - Cons: Massive RAM overhead (the graph must live primarily in memory) and slow index build times.
  - IVF: Clusters the vector space using $k$-means. At query time, it only searches the clusters closest to the query vector. - Pros: Memory-efficient, scales to billions of vectors without needing all vectors in RAM. - Cons: Lower recall unless you increase the number of clusters to search ($nprobes$), which drives up latency.

#### Scenario-Based Staff Interview Questions

###### Scenario A

"Your vector search returns great semantic matches, but users complain that when they search for specific part numbers (like 'TX-9000-REV2'), the system returns documents about completely different parts because the dense embedding model treats 'TX-9000' and 'TX-8000' as semantically identical. How do you fix this?"

Staff-Level Answer Strategy:

Identify the Root Cause: Pure dense embedding failure on out-of-vocabulary tokens and alphanumeric codes.

Architectural Fix: Implement Hybrid Search with a sparse retriever (BM25 or neural sparse like SPLADE) explicitly tuned for exact token matching.

Data Layer Enhancement: Apply metadata filtering. Extract part numbers during preprocessing using regex/NER, store them in structured metadata fields in the vector DB, and force hard metadata filters when a part number pattern is detected in the user query.

###### Scenario B

"Management wants to reduce vector database infrastructure costs because RAM usage for HNSW indexes is blowing up the cloud bill. They suggest switching everything to flat file scans or cheaper indexes. What are the production implications of this choice?"

Staff-Level Answer Strategy:

Quantify Trade-offs: Explain that dropping HNSW for unindexed flat searches (Brute-force KNN) kills latency (scaling linearly with dataset size), failing real-time SLA requirements ($<200\text{ms}$).

Alternative Proposals: Move to DiskANN or compressed indexes like PQ (Product Quantization) combined with IVF to compress vector memory footprints by 4x–8x while maintaining acceptable recall. 
Implement Tiered Storage: Keep hot/recent data in high-performance HNSW indexes in RAM, and cold historical data in compressed IVF-PQ or disk-backed indexes.

#### Metrics
- MetricsRecall@K: The proportion of relevant documents captured within the top-$K$ retrieved results (crucial for ensuring the right answer is actually present in the candidate pool).
- Precision@K: The proportion of retrieved documents in the top-$K$ that are actually relevant (measures how much noise you are filtering out).
- MRR (Mean Reciprocal Rank): Evaluates how high up the first correct answer appears in the ranked retrieval list.
- NDCG (Normalized Discounted Cumulative Gain): Measures ranking quality, penalizing systems when relevant items are buried lower down in the retrieved list.
- Query Latency (P95 / P99): End-to-end time taken from user query submission to receiving the final reranked context chunks.


##### Generation side

Retrieved Chunks ──> Context Compression / Packing ──> Prompt Assembly ──> LLM Generation (Streaming/Caching) ──> Guardrails & Safety Filters ──> Response to User

- Context Compression & Packing - while dealing with scenarios like - You retrieved 10 chunks, but they contain fluff, or they exceed the prompt token window, or putting too much irrelevant text in the middle causes the LLM to suffer from the "Lost in the Middle" phenomenon (where models ignore instructions or details buried in the center of long contexts). - Techiniques - Prompt Compression: Using small auxiliary models to strip redundant tokens and filler words from retrieved chunks while preserving key facts, shrinking context by up to 50% without losing semantic meaning. - Reordering Chunks: Placing the highest-scoring chunks at the very beginning and very end of the prompt context window, as transformer models attend much more strongly to the edges of their input.
- Caching (optimize cost & latency) - Semantic Caching: Instead of exact-string caching, you embed incoming user queries and check against a cache vector DB (like Redis). If a user asks "How do I reset my password?" and another asks "Steps to reset password", the semantic distance is near zero, and you serve the cached LLM response instantly, saving 100% of LLM inference cost and latency.
- Guardrails and Safety Layers - Input Guardrails (Pre-generation): Prompt injection defense, PII masking, and out-of-domain detection (e.g., stopping a user from asking an internal HR RAG bot about cooking recipes). - Output Guardrails (Post-generation): Groundedness checking. Ensuring the generated text is strictly derived from the retrieved context and does not hallucinate facts outside the provided sources. Frameworks like NeMo Guardrails or Llama Guard are standard here.
- Evaluation - Context Relevance: Did the retrieval layer pull chunks that actually answer the query, or is there a lot of noise? (Measured via retriever evaluation).- Groundedness / Faithfulness: Is the generated answer comp - Answer Relevance: Does the final response actually address the user's original question?

#### Scenario-Based Staff Interview Questions

###### Scenario A
"Users are noticing that when the retrieved context contains conflicting information across different versions of a document, the LLM hallucinates a hybrid answer or picks an arbitrary older version. How do you solve this at the generation layer?"

Staff-Level Answer Strategy:

Metadata Injection: Ensure your preprocessing and chunking layers preserve explicit timestamps, version numbers, or document status (e.g., [Version: 3.2, Date: 2026]) directly in the chunk text or metadata block.

Prompt Instructions: Update the system prompt to explicitly enforce temporal hierarchy: "If conflicting instructions or facts appear across chunks, prioritize the document with the most recent timestamp and cite its version."

Deterministic Filtering: Filter out outdated chunks at the retrieval layer using metadata filters before they even reach the generation phase.

###### Scenario B
"Your CFO points out that your enterprise RAG token costs have doubled month-over-month, and LLM latency is creeping up to 4 seconds per response. How do you architect a cost-reduction and speed optimization plan without degrading accuracy?"

Staff-Level Answer Strategy:

Implement Semantic Caching: Catch repetitive and similar queries instantly via Redis vector cache.

Tiered LLM Routing: Use a fast, cheap small language model (SLM) like Llama-3-8B or GPT-4o-mini for straightforward queries and standard RAG synthesis, escalating to a heavy frontier model (like GPT-4o or Claude 3.5 Sonnet) only when a classifier detects high complexity or ambiguity.

Context Window Optimization: Prune low-scoring reranked chunks instead of blindly passing all top-$k$ chunks to the LLM.

#### Metrics
- Faithfulness / Groundedness: Measures whether the generated answer is strictly derived from the retrieved context without introducing external hallucinations.
- Context Relevance: Measures whether the retrieved context chunks contain only necessary information or include excessive noise.
- Answer Relevance: Measures whether the final response directly addresses the user's intent or drifts into unrelated topics.
- TTFT (Time to First Token): The latency duration before the streaming generation begins rendering on the client side.
- Token Cost Efficiency: Cost per 1,000 queries tracked across embedding calls, reranking APIs, and LLM token consumption.







