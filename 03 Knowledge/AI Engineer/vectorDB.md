# Vector Database Concepts with Azure AI Search

A learning guide using the real index definition `social-listening-index`.

---

## 1. What is a vector database?

A traditional search matches **keywords**. A vector database matches **meaning**.

- An **embedding model** (e.g. Azure OpenAI `text-embedding-ada-002` / `text-embedding-3-small`) converts text into a list of numbers (a **vector**).
- Texts with similar meaning end up **close together** in this high-dimensional space.
- A query is also converted into a vector, and the database returns the **nearest neighbours**.

Example: "my knee hurts when I climb stairs" will match "joint pain on stairs" even though they share no words.

Azure AI Search is not a pure vector DB. It is a **hybrid search engine**: keyword (BM25) + vector + semantic reranking + filters in one index.

---

## 2. Anatomy of your index

| Section | Purpose |
|---|---|
| `fields` | The schema (columns) of each document |
| `scoringProfiles` | Boosts relevance of keyword matches |
| `similarity` | Keyword ranking algorithm (BM25) |
| `semantic` | Semantic reranker configuration |
| `vectorSearch` | Vector algorithms and profiles |

Each record (a social media post) is a **document**. Its key is `Id`.

---

## 3. The vector field

```json
{
  "name": "vector",
  "type": "Collection(Edm.Single)",
  "searchable": true,
  "retrievable": true,
  "dimensions": 1536,
  "vectorSearchProfile": "patientplus-vector-profile"
}
```

| Property | Meaning |
|---|---|
| `Collection(Edm.Single)` | An array of 32-bit floats, which is the vector type |
| `dimensions: 1536` | Must exactly match the embedding model output (ada-002 and text-embedding-3-small produce 1536) |
| `vectorSearchProfile` | Links the field to an algorithm configuration |
| `searchable: true` | Required so the field can be queried |
| `retrievable: true` | Returns the raw vector in results (usually wasteful. Set `false`, or use `stored: false`, to save storage) |
| `filterable/sortable/facetable: false` | These are not supported on vector fields |

**Rule:** the query vector must be produced by the **same model** as the indexed vectors.

---

## 4. Vector search configuration

```json
"vectorSearch": {
  "algorithms": [{
    "name": "patientplus-vector-algorithm",
    "kind": "hnsw",
    "hnswParameters": { "metric": "cosine", "m": 4, "efConstruction": 400, "efSearch": 500 }
  }],
  "profiles": [{
    "name": "patientplus-vector-profile",
    "algorithm": "patientplus-vector-algorithm"
  }],
  "vectorizers": [],
  "compressions": []
}
```

The chain is: **field → profile → algorithm** (plus optionally a vectorizer and compression).

### 4.1 Similarity metric: `cosine`

How "closeness" is measured:

- **cosine**: angle between vectors (ignores length). Best for text embeddings.
- **dotProduct**: angle and magnitude. Equivalent to cosine for normalized vectors.
- **euclidean**: straight-line distance.
- **hamming**: for binary vectors.

Cosine score ranges from 1 (identical direction) to -1 (opposite). Azure converts it into a `@search.score` where higher is better.

### 4.2 Algorithms: HNSW vs Exhaustive KNN

| | HNSW (`hnsw`) | Exhaustive KNN (`exhaustiveKnn`) |
|---|---|---|
| Approach | Approximate Nearest Neighbour (ANN) graph | Compares the query to **every** vector |
| Speed | Very fast, scales to millions | Slow at scale |
| Accuracy | High but not perfect (recall < 100%) | 100% exact |
| Use when | Production, large data | Small data, or as a baseline to measure recall |

**HNSW (Hierarchical Navigable Small World)** builds a layered graph. Search starts at a sparse top layer, jumps toward the query's neighbourhood, and descends to dense lower layers.

### 4.3 HNSW parameters in your index

| Parameter | Your value | Meaning | Trade-off |
|---|---|---|---|
| `m` | 4 | Connections per node (bi-directional links) | Higher means better recall and more memory and slower indexing. Range 4-10, default 4 |
| `efConstruction` | 400 | Candidate list size while **building** the graph | Higher means better graph quality and slower indexing. Range 100-1000 |
| `efSearch` | 500 | Candidate list size while **querying** | Higher means better recall and higher latency. Range 100-1000 |

Your settings favour **recall over speed** (a high ef and the minimum m). That is reasonable for a mid-sized corpus.

### 4.4 Empty `vectorizers` and `compressions`

- `vectorizers: []`: no integrated vectorization is configured. Your **application must compute the query embedding** and send it as a raw vector. Adding a vectorizer (Azure OpenAI) lets you send plain text with `kind: "text"` and have the service embed it.
- `compressions: []`: no quantization. Options are **scalar quantization** (float32 to int8, about 4x smaller) or **binary quantization** (about 28x smaller), with optional rescoring. Useful to reduce cost at scale.

---

## 5. Keyword (full-text) search side

### 5.1 `searchable`
Text fields are analyzed (tokenized, lowercased, stemmed) into an **inverted index**. This powers BM25.

### 5.2 BM25 similarity
```json
"similarity": { "@odata.type": "#Microsoft.Azure.Search.BM25Similarity" }
```
Ranks documents by term frequency, inverse document frequency and document length. It is good for exact terms such as drug names and codes.

### 5.3 Analyzers
`"analyzer": "en.microsoft"` applies English linguistic processing, such as stemming ("running" to "run") and lemmatization. Fields without an analyzer use the default `standard.lucene`.

### 5.4 Synonym maps
`"synonymMaps": ["patientplus-synonym-map-profile"]` expands queries, so a search for "heart attack" can also match "myocardial infarction". They apply to **keyword search only**, not vector search (vectors capture synonymy by themselves).

### 5.5 Scoring profile
```json
"scoringProfiles": [{
  "name": "sp_social_listening",
  "text": { "weights": { "Text": 5, "CountryName": 2 } }
}]
```
A match in `Text` counts 5x and in `CountryName` 2x relative to other fields. It is set as `defaultScoringProfile`. It affects **BM25 only**, not vector scores.

---

## 6. Field attributes cheat sheet

| Attribute | What it enables | Example in your index |
|---|---|---|
| `key` | Unique document ID (exactly one per index) | `Id` |
| `searchable` | Full-text search | `Text`, `Title` |
| `filterable` | `$filter` expressions | `Date`, `Product`, `Sentiment` |
| `sortable` | `$orderby` | `Date` only |
| `facetable` | Count buckets for navigation | `Product`, `CountryName` |
| `retrievable` | Returned in results | Most fields |
| `stored` | Kept in storage for retrieval (false saves space, mainly for vectors) | all `true` |

Examples of how types are used:
- `Edm.DateTimeOffset` (`Date`) supports range filters.
- `Collection(Edm.String)` (`Region`, `Indication`, `Insightshashtag`) is multi-valued, filtered with `any()`.
- `Collection(Edm.ComplexType)` (`raw_ner_entities`, `raw_age_entities`) is nested objects, such as extracted entities with sentiment.

Design note: `Text` is `facetable: true`, which is unusual and costly for long free text. Facet on short categorical fields only.

---

## 7. Semantic ranking (reranker)

```json
"semantic": {
  "configurations": [{
    "name": "semantic_confi_patientplus_sl",
    "rankingOrder": "BoostedRerankerScore",
    "prioritizedFields": {
      "prioritizedContentFields": [{ "fieldName": "Text" }],
      "prioritizedKeywordsFields": [{ "fieldName": "CountryName" }]
    }
  }]
}
```

A Microsoft-trained **language model reranks the top 50 results** from BM25/vector/hybrid, using deep language understanding.

- `prioritizedContentFields`: the main body to read (`Text`).
- `prioritizedKeywordsFields`: short keyword-like fields (`CountryName`).
- A title field can be added with `prioritizedTitleField`. Your index has `Title` but does not use it here, which is an easy improvement.
- `BoostedRerankerScore` factors the scoring profile into the final reranker order.
- It also produces **captions** and **answers**.

---

## 8. Query types

### 8.1 Keyword only
```json
{ "search": "headache side effects", "queryType": "simple" }
```

### 8.2 Pure vector
```json
{
  "count": true,
  "select": "Id, Title, Text, Sentiment",
  "vectorQueries": [{
    "kind": "vector",
    "vector": [0.012, -0.034, "... 1536 floats ..."],
    "fields": "vector",
    "k": 10
  }]
}
```
`k` is the number of nearest neighbours to return.

### 8.3 Hybrid (keyword + vector), usually best
```json
{
  "search": "patients complaining about injection pain",
  "select": "Id, Title, Text, CountryName, Sentiment",
  "vectorQueries": [{
    "kind": "vector",
    "vector": [ "... 1536 floats ..." ],
    "fields": "vector",
    "k": 50
  }],
  "top": 10
}
```
Both retrievers run in parallel, and their rankings are merged with **Reciprocal Rank Fusion (RRF)**.

### 8.4 Hybrid + filter + semantic reranker
```json
{
  "search": "injection pain",
  "vectorQueries": [{ "kind": "vector", "vector": ["..."], "fields": "vector", "k": 50 }],
  "filter": "Product eq 'DrugX' and Sentiment eq 'Negative' and Date ge 2025-01-01T00:00:00Z",
  "queryType": "semantic",
  "semanticConfiguration": "semantic_confi_patientplus_sl",
  "captions": "extractive",
  "answers": "extractive",
  "top": 10
}
```

### 8.5 Facets (analytics on results)
```json
{ "search": "*", "facets": ["Product", "Sentiment", "CountryName"], "top": 0 }
```
Returns counts per Product, Sentiment and Country, which suits dashboards.

### 8.6 Filtering collections
```
Region/any(r: r eq 'EMEA')
Indication/any(i: i eq 'Diabetes')
```

---

## 9. Filtering with vector search

Filters can be applied **before** (pre-filter, default) or **after** (post-filter) the vector search:

- **Pre-filter**: restricts candidates first. It is accurate and always returns up to `k` results that match.
- **Post-filter**: retrieves the top `k` first, then filters. It can return fewer than `k` results.

Set it via `"vectorFilterMode": "preFilter" | "postFilter"`. Filter fields must be `filterable`, but the vector field itself cannot be filtered.

---

## 10. Typical end-to-end flow (RAG)

```
Social post  ->  Embedding model  ->  1536-d vector  ->  stored in `vector` field
                                                               |
User question -> Embedding model -> query vector ------------->|  HNSW + BM25 hybrid
                                                               v
                                      Top-N docs -> Semantic reranker -> Top-K
                                                               v
                                       LLM (GPT) gets Text + metadata -> grounded answer
```

Steps to build it:
1. Clean and enrich the data (NER, sentiment, age entities, as in your index).
2. Embed `Text` (or `Title + Text`) and store the vector in `vector`.
3. Upload documents in batches (`mergeOrUpload`).
4. Query with hybrid search plus filters.
5. Pass the top results to an LLM as context (RAG).

---

## 11. Python examples

```python
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

aoai = AzureOpenAI(azure_endpoint="<endpoint>", api_key="<key>", api_version="2024-06-01")

def embed(text: str) -> list[float]:
    return aoai.embeddings.create(model="<embedding-deployment>", input=text).data[0].embedding

search = SearchClient("https://<service>.search.windows.net",
                      "social-listening-index",
                      AzureKeyCredential("<key>"))

question = "What are patients saying about injection pain?"
results = search.search(
    search_text=question,
    vector_queries=[VectorizedQuery(vector=embed(question), k_nearest_neighbors=50, fields="vector")],
    filter="Sentiment eq 'Negative'",
    query_type="semantic",
    semantic_configuration_name="semantic_confi_patientplus_sl",
    select=["Id", "Title", "Text", "CountryName", "Sentiment"],
    top=10,
)
for r in results:
    print(r["@search.score"], r["@search.reranker_score"], r["Title"])
```

Prefer **Entra ID auth** (`DefaultAzureCredential`) over keys, and never hardcode keys.

---

## 12. Tuning guide

| Goal | Action |
|---|---|
| Better recall | Raise `efSearch`, `m` or `efConstruction`, or raise `k` |
| Lower latency | Lower `efSearch` |
| Lower storage and cost | Add scalar or binary quantization, set vector `retrievable: false`, drop unneeded facets |
| Better keyword relevance | Tune scoring profile weights and synonym maps |
| Better final ranking | Use hybrid + semantic reranker, and add a title field |
| Measure ANN accuracy | Compare HNSW results with an `exhaustiveKnn` profile |
| Simpler queries | Add a `vectorizer` so clients send text |

---

## 13. Common pitfalls

- Embedding dimension mismatch with `dimensions` causes an indexing error.
- Different embedding models for indexing and querying give meaningless results.
- Changing HNSW parameters or dimensions requires **re-creating the index**.
- Synonym maps and scoring profiles do not affect vector scores.
- Too many `facetable`/`filterable` fields increase index size.
- Embedding only long text can hurt retrieval. Chunk long documents (not needed for short social posts).
- Semantic ranker only reranks the top 50 results.

---

## 14. Glossary

- **Embedding**: numeric representation of meaning.
- **Dimension**: length of the vector (1536 here).
- **ANN**: approximate nearest neighbour search.
- **HNSW**: graph-based ANN algorithm.
- **BM25**: keyword relevance algorithm.
- **RRF**: Reciprocal Rank Fusion, which merges multiple ranked lists.
- **Hybrid search**: keyword + vector in one query.
- **Semantic ranker**: language-model reranker.
- **RAG**: Retrieval-Augmented Generation. Retrieve relevant data, then let an LLM answer from it.
- **Recall**: share of true nearest neighbours that were found.
- **Quantization**: compressing vectors to save memory.

---

## 15. Suggested next steps

1. Run a pure vector query, then hybrid, then hybrid + semantic, and compare results for the same question.
2. Experiment with `efSearch` (100, 500, 1000) and observe latency vs recall.
3. Add `Title` as `prioritizedTitleField`.
4. Try scalar quantization on a copy of the index and compare size.
5. Read: Microsoft Learn, "Vector search in Azure AI Search" and "Hybrid search".
