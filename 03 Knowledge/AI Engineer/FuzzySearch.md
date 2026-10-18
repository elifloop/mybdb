# Fuzzy Duplicate Filtering in GenAI

## 1. Theoretical Foundations

### What it does

The implementation identifies rows whose `Text_en` strings are sufficiently similar, then removes rows considered near-duplicates before RAG document preparation. It is **fuzzy deduplication**, not fuzzy retrieval: it does not take a user's query and search for similar corpus entries. Search/retrieval is handled elsewhere in the application.

Fuzzy similarity is useful when two records express essentially the same content with superficial differences such as casing, spacing, or punctuation. Unlike exact deduplication, a string similarity score can tolerate some textual variation. This also creates a precision/recall trade-off: a low threshold removes more variants but risks deleting distinct records; a high threshold preserves distinctions but leaves more duplicates.

### Similarity and selection

`_find_fuzzy_duplicates()` calls RapidFuzz `process.extract()` with `fuzz.QRatio`, `utils.default_process`, and `score_cutoff=80`. Scores are on a 0-100 scale; candidates scoring below 80 are omitted. `default_process` normalizes case and non-alphanumeric characters before comparison. The helper uses the default `process.extract` result limit (five choices), and then slices off the first result under the assumption that it is the query row itself.

There is no model training, learned embedding, or tensor operation here. The method is deterministic string matching. For a normalized pair of strings, the scorer returns a normalized edit-similarity score; the implementation keeps candidates satisfying:

$$s(a,b) \geq 80$$

The threshold is hard-coded, not calibrated from labeled duplicate/non-duplicate examples. It does not optimize a loss function. In particular, the score should not be interpreted as a probability that two records are duplicates.

### Alternatives and trade-offs

| Method | Advantages | Limitations |
|---|---|---|
| Exact `drop_duplicates` | Fast, predictable, preserves one exact copy | Does not catch spelling or formatting variants |
| RapidFuzz `QRatio` | Fast native string scoring; understandable threshold; no model deployment | Character-level surface similarity can miss semantic duplicates and conflate distinct short texts |
| Token/embedding similarity | Can capture reordered wording or semantic equivalence | Requires tokenization/embedding infrastructure, index management, threshold evaluation, and greater compute |

For short fields such as names, a high character-similarity score may still be ambiguous. For long text, a single local edit contributes less to the total score, while semantically equivalent paraphrases can score poorly.

## 2. Practical & Engineering Implementation

### Code walkthrough

`RAGResponseBuilder.remove_duplicates_fuzzy(df)` in [rag_utility.py](../aim_plus/GenAI/src/rag_components/rag_utility.py#L32) returns immediately for an empty DataFrame. Otherwise it logs the starting shape, adds temporary columns, applies `_find_fuzzy_duplicates()` to each row, passes the collected indices to `_drop_duplicate_rows()`, logs the final shape, and returns the filtered DataFrame.

The matching call is:

```python
matches = process.extract(
	row['Text_en'],
	df['Text_en'],
	scorer=fuzz.QRatio,
	processor=utils.default_process,
	score_cutoff=80
)[1:]  # Exclude self-match

index_list = [match[2] for match in matches]
return matches, len(matches), index_list
```

Source: [rag_utility.py](../aim_plus/GenAI/src/rag_components/rag_utility.py#L68), `_find_fuzzy_duplicates()`.

The helper records every extracted candidate index, then the drop function concatenates those lists and drops the resulting indices:

```python
indices_to_drop = []
for _, row in df.iterrows():
	if isinstance(row['index_fuzz_text'], list):
		indices_to_drop.extend(row['index_fuzz_text'])

return df.drop(indices_to_drop, errors='ignore')
```

Source: [rag_utility.py](../aim_plus/GenAI/src/rag_components/rag_utility.py#L82), `_drop_duplicate_rows()`.

The dependency is pinned as `rapidfuzz==3.14.1` in [requirements.txt](../aim_plus/GenAI/requirements.txt#L34).

### Pipeline integration status

The helper has call sites in the QA and summarization flows, but both invocations are commented out. For example, `process_query()` currently proceeds directly to `prepare_documents()`:

```python
# data_df = self.preprocessor.remove_duplicates_fuzzy(data_df)
prepared_data = self.prepare_documents(data_df)
```

Source: [question_answering.py](../aim_plus/GenAI/src/rag_components/question_answering.py#L139). The corresponding preprocessing call is also commented out in [summarization.py](../aim_plus/GenAI/src/rag_components/summarization.py#L220). Therefore this fuzzy helper is implemented but **not active in these flows**. QA's `prepare_documents()` instead applies exact deduplication on `Text_en`.

### Data shape and complexity

There are no tensors. The input is a Pandas DataFrame with $N$ rows and at least a `Text_en` column (the current helper also assumes a `Name` column). Each row is compared against the full `Text_en` series. Each `process.extract()` result is a tuple of candidate text, score, and candidate index; the row-wise `apply()` expands the matches, match count, and index list into three temporary columns.

Because it scores the full choice series for each input row, the broad scaling is approximately $O(N^2)$ string comparisons, with additional work dependent on string lengths. RapidFuzz is implemented in optimized native code, but that does not remove the quadratic number of candidate comparisons. The default five-result cap limits stored matches, not the need to score the candidate set.

### Performance, reliability, and observability

- Log input/output row counts and elapsed time; current logs record before/after shapes but do not measure duplicate precision or recall.
- Track the distribution of similarity scores, number of candidate matches per row, rows removed, and text lengths. Evaluate the cutoff against a reviewed duplicate set before enabling the helper.
- For larger data, block candidates by a stable key (for example, source or normalized prefix), use an index/LSH strategy, or deduplicate in batches with overlap handling rather than comparing every row with every other row.
- Preserve deterministic survivor rules (for example, keep earliest or highest-quality record) and provenance. A deduplication pass should remove one member of a duplicate group, not erase the group.
- Monitor null/empty text, Unicode normalization, repeated records, short strings, and changes in document source. The `Name` and `Text_en` column assumptions should be validated before processing.
- The method mutates the supplied DataFrame by adding `fuzz_elements`, `duplicate_fuzz_text`, and `index_fuzz_text`; it also assigns `Name` to itself. Consider working on a copy and dropping internal helper columns before returning.

## 3. Technical Interview Preparation

### System design scenarios

**1. Deduplicate 10 million incoming records while keeping the near-real-time RAG index fresh.**

- Define a canonical text field, normalization policy, duplicate scope, and deterministic survivor policy.
- Avoid all-pairs scoring: partition by source/time/entity or use a candidate-generation index such as character n-grams/LSH, then apply RapidFuzz to candidates.
- Use idempotency keys and durable state so retries do not create duplicate writes.
- Separate candidate generation, exact scoring, human review for uncertain matches, and index updates.
- Benchmark throughput and memory by text length; monitor candidate counts, score distribution, false merges, false splits, and processing lag.

**2. Combine fuzzy deduplication with semantic RAG retrieval without losing distinct evidence.**

- Keep deduplication separate from ranking/retrieval; do not drop a document only because its wording is similar.
- Group near-duplicates while retaining source IDs, timestamps, metadata, and a representative document.
- Use lexical similarity for text variants and embeddings for paraphrase candidates; calibrate separate thresholds on labeled pairs.
- Preserve citations to all supporting records and test query-answer quality before and after deduplication.
- Add reversible decisions, audit logs, and a review path for high-impact or low-confidence merges.

### Deep-dive questions and strong answers

**Q1: What does `QRatio` with cutoff 80 mean, and is 80% a probability?**

**Strong answer:** It is a normalized string-similarity score, not a posterior probability. The cutoff is a manually selected operating point. We need labeled pairs and category-specific precision/recall analysis to select a threshold; score calibration would be required before interpreting scores probabilistically.

**Q2: Why can `[1:]` fail to exclude the current row?**

**Strong answer:** The first result is assumed to be the current row, but equal-scoring strings can tie and result order may put a different duplicate first. The safer approach is to identify and exclude the current row by its stable DataFrame index or document ID, not by result position.

**Q3: What is the complexity and how would you scale this implementation?**

**Strong answer:** It calls `process.extract()` once per DataFrame row against the full series, giving roughly quadratic candidate comparisons in row count. First reduce the candidate set with blocking or an approximate index, then run exact RapidFuzz scoring on candidate pairs. For batch processing, make boundary overlap explicit and maintain a stable cross-batch duplicate key.

### Common gotchas and edge cases

1. **A duplicate group can be entirely removed.** Each row can nominate the other rows, and `_drop_duplicate_rows()` drops the union of every nominated index. Similarity is symmetric, so a pair can cause both rows to be dropped. Additionally, `[1:]` can skip a different tied match instead of self. Fix by building undirected candidate pairs, selecting connected groups, and retaining one deterministic representative per group.
2. **The returned frame leaks internal columns and mutates its input.** The helper adds `fuzz_elements`, `duplicate_fuzz_text`, and `index_fuzz_text` without removing them, and edits the caller's DataFrame in place. Copy the frame at entry, use local results, and return only the intended schema.
3. **The algorithm can be expensive and over-merge text.** Full-series matching for each row scales quadratically; an 80 cutoff may also merge short but distinct texts. Benchmark by corpus size, block candidates, and validate threshold performance separately for short and long texts. Handle missing `Text_en` values before calling the scorer.

### Repository evidence

- Matcher and row removal: [rag_utility.py](../aim_plus/GenAI/src/rag_components/rag_utility.py)
- QA call site: [question_answering.py](../aim_plus/GenAI/src/rag_components/question_answering.py)
- Summarization call site: [summarization.py](../aim_plus/GenAI/src/rag_components/summarization.py)
- Dependency pin: [requirements.txt](../aim_plus/GenAI/requirements.txt)
