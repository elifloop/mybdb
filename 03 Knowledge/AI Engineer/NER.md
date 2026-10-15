# Named Entity Recognition (NER) in This Repository

## 1. Theoretical Foundations

### What NER solves

Named Entity Recognition identifies spans in text and assigns each span a type. For input tokens $x_1,\ldots,x_n$, a sequence tagger predicts labels $y_1,\ldots,y_n$; a BIO scheme commonly represents entity starts as `B-TYPE`, continuations as `I-TYPE`, and non-entities as `O`. The resulting spans can carry a surface string, normalized value, type, confidence, and character offsets.

This repository uses NER to extract medically useful facts from mVoC text, including diseases/diagnoses, symptoms, medication concepts and attributes, age, and study names. Those annotations are later represented as rows keyed to the source insight, with character positions into the English text.

### Model objective and decoding

The BioBERT token-classification path in `gsk-biobert_drug.py` produces a score vector for every token. If the encoded batch has shape $B\times L$ and the model has hidden width $H$ and $K$ labels, the classifier scores have shape $B\times L\times K$. The implementation selects the highest-scoring label independently at each token:

$$\hat y_{b,t}=\arg\max_{k\in\{1,\ldots,K\}} z_{b,t,k}$$

For a standard supervised token classifier, training minimizes token-level cross-entropy over non-padding positions:

$$\mathcal{L}=-\sum_{b,t} m_{b,t}\log p(y_{b,t}\mid x_b)$$

where $m_{b,t}$ masks padding/special positions. The checked-in class exposes `predict()` but its `train()` method is a placeholder, so this repository does not show the training objective or training loop for those saved BioBERT artifacts. The Azure Healthcare model is an external service; its model architecture and training objective are not implemented here.

The `EntityDictGenerator` converts token labels back into text spans. It groups wordpieces, discards special tokens, starts an entity on `B-`, and extends it on `I-`. The Azure API instead supplies entity offsets and lengths directly, so its span end is computed as `Offset + Length`.

### Architecture and alternatives

The repository contains complementary approaches, not one unified NER model:

| Approach | Where used | Strength | Limitation |
|---|---|---|---|
| Hosted Azure Healthcare NER | Main mVoC flow in `azure-healthcare-ner.py` | Medical categories, normalized text, confidence, and source metadata without hosting model weights | Network/service dependency, quota and cost exposure, external model versioning |
| Dictionary/span extraction | `ner-studyNames-extract.py` and related helpers | Deterministic, explainable, straightforward to update with a curated study list | Misses unseen names; matching quality depends on dictionary and span-matching rules |
| Transformer token classification | BioBERT evaluation POC in `gsk-biobert_drug.py` | Domain-adapted context-sensitive predictions and locally controlled inference | Artifact/GPU/memory management; token limits and label-decoding complexity |
| spaCy Med7 | `med7-ner.py` | Focused extraction of drug, dosage, form, route, frequency, duration, and strength | Separate path from the main generation notebook; model coverage and version compatibility matter |

Compared with regular-expression or dictionary rules, contextual models handle ambiguous mentions better but are harder to explain and maintain. Compared with training an in-house model, the Azure service reduces model operations but adds external availability, cost, data-governance, and version-control considerations.

## 2. Practical & Engineering Implementation

### End-to-end mVoC flow

1. `generate-ners(Main).py` loads insight text and metadata from SQL or Delta, joins the tables, and constructs `Text_en`. For source `4`, it prepends `Question_en` to the text. It creates `tmp_mvoc_data` with the insight ID, composite key, original/processed text, and language.
2. The notebook runs `azure-healthcare-ner` and `ner-studyNames-extract` in parallel. The Azure notebook submits groups of 100 input rows to a `ThreadPoolExecutor` with up to 30 workers. The ADF `Deltas Insights Processor` pipeline schedules the generation notebook after its NER checkpoint activity, with a configured maximum of 5,000 API records.
3. The Azure path uses `TextAnalyticsClient` with the endpoint and service principal credentials loaded from Databricks secrets. It parses each response entity into `Entity`, `Normalized_Text`, `Category`, `Sub_Category`, `Confidence_Score`, `Offset`, `Length`, and source metadata. The downstream mapping keeps categories `Age`, `Diagnosis`, `MedicationClass`, `MedicationForm`, `MedicationRoute`, and `SymptomOrSign`, mapping them to repository labels such as `Disease`, `DrugClass`, and `RouteOfAdministration`. Azure drug entities also pass through `azh-ners_downstream_tasks`.
4. The study-name path loads `/dbfs/mnt/models/dictionary/Study_Names.csv`, tags matches, computes exact spans, maps study origin to `TypeNer`, and assigns uppercase normalized text and confidence `1.0`. It also extracts curated custom terms using `custom_ner_extraction_data_file.csv` and `ner_common_preprocess` helpers.
5. `generate-ners(Main).py` unions Azure and study-name rows, removes duplicate `(Id, TextNer, StartPosition, EndPosition)` annotations, and writes the result to the `insights_azh_ners` Hive table. The Azure notebook uses shared CSV/master-file checkpointing to avoid reprocessing already-seen text.

### Other implemented paths

- `med7-ner.py` loads `en_core_med7_lg`, sentence-splits input, divides rows into five chunks, and runs `med7_ner_processing` in parallel. Its stated labels are Drug, Dosage, Form, Route, Frequency, Duration, and Strength. It is a distinct Med7 workflow; the main notebook's listed parallel calls are Azure Healthcare NER and study-name extraction, not Med7.
- `gsk-biobert_drug.py` defines `GSKNERBioBERT` for token-classification inference. It loads a label list and Hugging Face `AutoConfig`, `AutoTokenizer`, and `AutoModelForTokenClassification` from a configured local/DBFS directory. `predict()` tokenizes with padding and truncation at 512 tokens, uses batches of 256 by default, runs under `torch.no_grad()`, argmax-decodes labels, then calls `EntityDictGenerator`. Related POC files evaluate disease and drug predictions. `train()`, `score()`, and other lifecycle methods are stubs; the artifacts are loaded, not trained in this code.
- `Social_Listening/SL_cleanup_script_Entire_AIM.py` and `SL_secondary_cleanup_script.py` load scispaCy BC5CDR, spaCy English NER, and serialized BioBERT/SimpleTransformers models for disease/chemical extraction and cleanup. These are separate Social Listening workflows.

### Data and tensor shape tracking

For the BioBERT `predict(sentences)` route:

```text
sentences                         list[str], size B
tokenizer(..., padding=True)      input_ids / attention_mask: [B, L], L <= 512
AutoModelForTokenClassification   logits: [B, L, K]
argmax(axis=2)                    predicted label IDs: [B, L]
EntityDictGenerator               list of entity dictionaries per input sentence
```

`H` (the transformer hidden width) is internal to the model; it is not materialized by this code. Padding makes each batch rectangular. The prediction `DataLoader` batches at 256 by default, but tokenization is performed for the full `sentences` list before the loader is created. Azure input/output is row- and entity-oriented rather than tensor-oriented; API offsets and lengths refer to the submitted text and must stay aligned with that exact text.

### Performance, scale, and bottlenecks

- **Azure throughput:** External request latency, service quota, throttling, and failures dominate. The code groups 100 records and fans out up to 30 futures, which increases throughput but can also create bursts. The ADF cap of 5,000 records and checkpoint/master CSV strategy bound or avoid repeated work; failed records are written to an error CSV in the Azure notebook.
- **Driver memory:** Several notebooks convert Spark data to Pandas (`toPandas()`), so the active input and intermediate output must fit driver memory. This limits safe scale even where Spark is used elsewhere.
- **BioBERT memory/latency:** Transformer inference cost grows approximately with the number of tokens and attention's quadratic sequence-length term. Padding to the longest item in a batch wastes work for mixed-length text. `max_length=512` bounds memory but truncates long inputs. GPU transfer and model loading add startup cost; CPU inference is supported but likely slower.
- **Med7 execution:** Five chunks and parallel notebook execution distribute work, but all chunks are still coordinated by the driver and outputs are unioned there. Sentence splitting changes the unit of processing; offsets must be interpreted relative to the sentence or restored to document coordinates consistently.
- **Artifact/service I/O:** The BioBERT POC expects model files and `labels.txt` at a hard-coded DBFS path. The Azure path depends on secrets, endpoint availability, and shared checkpoint storage.

### Production deployment and observability

- Track input document/character counts, entities per document and category, API latency and throughput, retries/errors, throttling, duplicate rate, empty-result rate, and end-to-end pipeline duration.
- Monitor confidence distributions by category and source, normalization coverage, span validity (`0 <= start <= end <= text length`), and annotation yield over time. Compare reviewed samples for precision/recall and detect model/service or source-text drift; confidence alone is not a quality metric.
- Preserve the exact submitted text and a stable text/version hash when caching results. Offsets become invalid if text is translated, normalized, concatenated, or otherwise edited after inference.
- Make external calls idempotent and retry only transient failures with bounded exponential backoff and quota-aware concurrency. The ADF configuration disables automatic retry for NER generation to avoid duplicate paid API calls; a robust recovery design should use per-record status/checkpointing rather than blindly replaying whole batches.
- Version endpoint/model, dictionary, label map, and text preprocessing. Protect service credentials in a secret store and verify data retention/region requirements before sending sensitive text to a hosted API.
- Route uncertain, high-impact, or newly emerging entity types to human review; retain a rollback path for model, endpoint, and dictionary changes.

## 3. Technical Interview Preparation

### System design scenarios

**1. Scale medical NER from batch processing to 10,000 documents per minute.**

- Separate ingestion, inference, and persistence with a durable queue and bounded worker pools.
- Batch by both document count and character/token budget; enforce provider quotas with a centralized rate limiter.
- Use idempotency keys based on document ID, text hash, model/version, and preprocessing version.
- Add bounded retries with jitter, dead-letter handling, per-document checkpoints, and replay tooling.
- Partition outputs by date/source and write incrementally; avoid collecting the full corpus into Pandas on the driver.
- Load-test p50/p95/p99 latency, throughput, throttling, error rate, cost/document, and backlog drain time.

**2. Design a reliable hybrid NER platform combining hosted healthcare extraction, a custom transformer, and curated study dictionaries.**

- Define a shared annotation contract for entity type, surface text, normalized text, confidence, offsets, source, and model/version.
- Route entity classes to the appropriate extractor; normalize category names and offset conventions at adapters.
- Deduplicate using document ID, span, and canonical type while preserving provenance from each extractor.
- Evaluate each route against a reviewed gold set; monitor category-specific precision/recall and normalization accuracy.
- Use confidence/disagreement thresholds for review, canary model/dictionary updates, and support rollback.
- Keep source text immutable during inference and test Unicode, translated text, and long-document handling.

### Deep-dive questions and strong answers

**Q1: Why use BIO labels, and what can go wrong when decoding them?**

**Strong answer:** BIO labels encode span boundaries as token-level classification. Decoding combines `B-TYPE` with following `I-TYPE` labels and maps tokens back through tokenizer character offsets. A robust decoder should also validate that an `I-X` follows an entity of type `X`; malformed transitions should be repaired or rejected. The repository's `EntityDictGenerator` extends the previous entity for any `I-` label without checking type equality, so inconsistent transitions can produce mixed-type spans.

**Q2: What does truncation at 512 do to NER quality and offsets?**

**Strong answer:** The BioBERT tokenizer truncates each encoded input to 512 tokens. Entities after the cutoff are never predicted, and a tokenized word near the cutoff may be incomplete. Character offsets from `token_to_chars` are relative to the original input encoding, but there is no windowing/reconciliation code here. For long documents, sentence/chunk splitting or overlapping windows with offset translation and boundary deduplication is needed; evaluate recall by document length.

**Q3: Why is `argmax` not a confidence score, and how would you calibrate it?**

**Strong answer:** `argmax` only selects the most likely label; the implementation discards logits/probabilities when it converts them to IDs. The Azure path returns a service confidence score, while the BioBERT prediction output shown does not attach per-entity confidence. Calibration requires retaining token probabilities, aggregating them over spans with a defined rule, then evaluating reliability (for example, expected calibration error) on held-out labeled data. Thresholds should be tuned per entity type and risk.

### Common gotchas and edge cases

1. **Offsets refer to a different text than the persisted text.** The main notebook can prepend `Question_en` for source `4`, while study extraction reads `Text_en_orig`; ASCII normalization is also applied in the study and Azure paths. Combining offsets from transformed and original strings can silently misplace spans, especially with non-ASCII characters. Keep an explicit canonical inference text and test offsets against it, or maintain a character-level transform map.
2. **Empty or malformed API results can poison parsing.** The Azure response parser catches exceptions, but construction of output rows and intermediate entity lists still assumes expected response fields. Treat per-document API errors, empty entity lists, null normalized values, and schema changes as explicit cases; validate output schema and preserve failed document IDs for retry.
3. **Inference configuration silently limits or changes coverage.** BioBERT inputs over 512 tokens are truncated; its artifact path and label file are local assumptions, and `get_abs_path()` has its download logic commented out. Its decoder also does not enforce matching BIO entity types. Validate artifact/version availability at startup, report truncation, use long-document windowing where needed, and test malformed BIO transitions. Separately, study-name extraction only finds names represented in its dictionary and assigns confidence `1.0` by construction, which is not a calibrated probability.

### Repository evidence

- Main generation and Azure integration: `Databricks/notebooks/mvoc/ner/generate-ners(Main).py`, `azure-healthcare-ner.py`, `ner-studyNames-extract.py`
- Med7 workflow: `Databricks/notebooks/mvoc/ner/med7-ner.py`
- BioBERT inference and span decoding: `Databricks/notebooks/poc/cortellis_gskbiobert_expertai_eval/gsk-biobert_drug.py`
- Social Listening models: `Databricks/notebooks/Social_Listening/SL_cleanup_script_Entire_AIM.py`, `SL_secondary_cleanup_script.py`
- Scheduled pipeline: `DataFactory/pipeline/Deltas Insights Processor.json`
