
### AI reasearch (theory) ###

-> was reading Deep Learning on kindle
-> continue with multi head attention
-> tensor & kernel puzzles [ON HOLD]

### optrade ###
-> vectorize and entity extraction

### FDE ###
-> fde interview - rag etc. going on 
-> work - reformulation of ai nba??; alternate view of nba as transformer [ON HOLD]


fred - 0ec87ebf93d05b1b2c57d115c1abd1c1

Before you paste the brief, copy two things from this repo into the new data repo:

# run in this simulator repo; <data-repo> is the path to the new Dagster repo
mkdir -p <data-repo>/contracts/examples
uv run simevidence schema > <data-repo>/contracts/evidence-v1.schema.json
cp examples/nickel_toy/evidence/*.json <data-repo>/contracts/examples/

# Brief: build the data platform (Dagster) for the simulator

## Context
I am building a decision intelligence platform in a separate repository, "the simulator". It turns
what-if questions (e.g. "What happens to nickel markets if Indonesia bans ore exports?") into
system dynamics models and runs Monte Carlo studies. Every number in a model must trace back to a
source.

This repository is the **data platform**:
- It fetches, stores and curates evidence: time series, and values quoted from documents.
- It serves that evidence through a small read-only HTTP API.
- The simulator only reads the API (its client package is `simevidence`) and never writes back.
- The simulator re-checks everything it receives (units, bounds, provenance). Your job is to serve
  correct, well-sourced records in the agreed format. This repo knows nothing about models.

Constraints:
- Personal use, local-first: one Linux PC in production, macOS for development.
- No cloud services.
- Python 3.12 managed with uv.
- Must keep working offline for data already fetched.

## The contract: evidence/v1 (do not change it)
The JSON Schema is in `contracts/evidence-v1.schema.json`, and two valid examples are in
`contracts/examples/`. Every record the API serves must validate against the schema and must also
meet these rules, which the schema cannot express.

Fields common to both kinds of record:
- `contract`: always `"evidence/v1"`.
- `kind`: `"series"` or `"value"`.
- `id`
  - Stable and unique; pattern `^[a-z0-9][a-z0-9_.:-]{0,127}$`.
  - Convention: `<source>.<native id>`, e.g. `fred.pnickusdm`. Curated values use
    `<firstauthor><year>.<what>`.
  - Never reuse an id for a different thing. When the data behind an id is updated (for example
    a new month), the id stays the same.
- `quantity`
  - A key from the quantity catalog; pattern `^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$`.
  - Convention: `<subject>.<concept>[.<detail>]`, e.g. `nickel.price`,
    `nickel.demand.price_elasticity`.
- `unit`
  - A standard unit string in Pint syntax, in the unit the source itself uses. Do not convert
    to model units.
  - Examples: `USD / tonne`, `tonne / year`, `USD / oil_barrel`, and `1` for dimensionless.
  - Never use the ambiguous `ton`, `tons`, `kt`, `oz`, `barrel` or `bbl`. Use `tonne`,
    `kilotonne`, `troy_ounce` or `oil_barrel` instead.
  - Currencies are upper-case ISO codes from this list only: USD, EUR, GBP, CNY, JPY, IDR, PHP,
    AUD, CAD, CHF, INR, BRL.
  - No model-specific custom units.
- `source`
  - `title` and `publisher` are required; `authors` is a list.
  - `url` and/or `doi`, at least one. Write the DOI without `https://doi.org/`.
  - `published` (a date).
  - `retrieved_at`: the actual fetch time, ISO 8601 in UTC, e.g. `2026-10-27T09:15:00Z`.
  - `license`.
- `context`: a string-to-string map of what a reader needs to interpret the number: frequency,
  region, period, method, notes.
- `version`: optional; your own tag, such as the Dagster run id or the snapshot hash.
- No other fields; extra fields are rejected. All numbers must be finite (no NaN or Infinity).

Series records (`kind: "series"`):
- `observations` is a list of `[ISO date, number]` pairs with strictly increasing dates.
- Monthly data is dated on the first of the month.

Value records (`kind: "value"`):
- `value`, plus optional `low` and `high` with `low <= value <= high`.
- `quote`: the source's own words stating the value, copied verbatim from the stored document text.
- Before publishing, check that:
  - the quote appears in the stored text (after normalising whitespace);
  - the value appears in the quote as written (allow for formatting differences such as a
    Unicode minus sign or thousands separators).

## The API (read-only)
- `GET /v1/records?quantity=KEY` returns `200 {"records": [...]}`: all current records for that
  quantity.
  - An unknown quantity gives `200 {"records": []}`, not 404.
  - Never return a record whose `quantity` differs from KEY.
- `GET /v1/records/{id}` returns `200` with the record, or `404`. Record ids may contain `.`, `:`
  and `-`.
- `GET /v1/health` returns `200 {"status": "ok"}`.
- Responses are JSON in UTF-8. Serve at the root path (`/v1/...`); the simulator appends `/v1/...`
  to the base URL it is given.
- Authentication: if `DATA_API_TOKEN` is set, require `Authorization: Bearer <token>`; otherwise
  the API is open. Bind to localhost by default.
- Serve only records that passed every check. Never compute or modify values when reading.

## Architecture
- **Dagster** handles ingestion, with one asset group per source:
  1. **raw**: fetch the response and store its bytes unchanged. Address them by SHA-256 and record
     the URL and fetch time, so every record traces back to exact bytes.
  2. **parse**: turn raw bytes into records.
  3. **asset checks**: the contract, the extra rules above, and source-specific sanity checks
     (no gaps in a monthly series, values in a plausible range). A record that fails is not
     published.
  4. **publish**: upsert the record into the store and keep its previous versions in a history
     table.
- **Schedules**: FRED series weekly; curated values run on demand.
- **Record store**: Postgres 16. A `records` table (id, quantity, kind, content as JSONB, version,
  published_at) and a `record_history` table. Raw blobs live on disk under `DATA_DIR`.
- **API**: a FastAPI app, run as its own process separate from Dagster, that only reads the
  record store.
- **Docker Compose**: postgres, the Dagster webserver and daemon, and the api. Everything local.
- **Configuration**: environment variables or `.env`; never commit secrets. `DATABASE_URL`,
  `DATA_DIR`, `DATA_API_TOKEN`, and optionally `FRED_API_KEY` and `OPENALEX_API_KEY`.
- **Tooling**: ruff, pyright, pytest and a Makefile with `make check`.
- **The contract in code**: pydantic v2 models in this repo that mirror the contract and its extra
  rules. Do not depend on the simulator's packages, which pull in JAX.

## Milestone 1 (do this first): the nickel slice, end to end
1. Repository skeleton, the contract models, and tests showing that `contracts/examples/*.json`
   pass both the models and the JSON Schema, and that broken variants fail.
2. `catalog/quantities.yaml` (key, description, unit, source):

   | Key | Unit | Source |
   |---|---|---|
   | `nickel.price` | USD / tonne | FRED `PNICKUSDM`, IMF monthly |
   | `copper.price` | USD / tonne | FRED `PCOPPUSDM` |
   | `cocoa.price` | USD / tonne | FRED `PCOCOUSDM` |
   | `oil.brent.price` | USD / oil_barrel | FRED `POILBREUSDM`, IMF monthly |
   | `nickel.demand.price_elasticity` | 1 | curated value (step 4) |

3. FRED source:
   - Download `https://fred.stlouisfed.org/graph/fredgraph.csv?id=SERIES`, which needs no key.
     Use the FRED API instead when `FRED_API_KEY` is set.
   - Parse into a series record with id `fred.<series id in lowercase>` and url
     `https://fred.stlouisfed.org/series/SERIES`.
   - Treat FRED's "." placeholders for missing values as gaps, not zeros.
4. Curated values:
   - `curated/*.yaml` holds hand-entered value records (value, quote, DOI, context).
   - Fetch the document text the quote comes from, e.g. the abstract from OpenAlex looked up by
     DOI. OpenAlex returns the abstract as an inverted index, so rebuild the text. Store the text
     as a raw asset and run the quote check against it.
   - First entry: nickel's price elasticity of demand, -0.09 (Shojaeddini, Alonso & Nassar 2024,
     DOI 10.1016/j.resconrec.2024.107664). It must match `contracts/examples/`.
5. The API, with tests using FastAPI's TestClient. Use an in-memory store implementation for unit
   tests, plus one integration test against Postgres marked `integration`.
6. Unit tests never touch the network: use recorded responses (a small saved FRED CSV) with respx
   or pytest-httpx.

Milestone 1 is done when, with Compose up and the assets materialised:
- `curl 'localhost:8000/v1/records?quantity=nickel.price'` returns the FRED record.
- In the simulator repo, `uv run simevidence find nickel.price --source http://localhost:8000`
  lists it.
- In the simulator repo, this command succeeds with the same `reference_price` (16,815.97, the
  2024 average), unless FRED has revised 2024 values since:
  `uv run simevidence ground examples/nickel_toy/model.yaml examples/nickel_toy/bindings.yaml --source http://localhost:8000`

## Later milestones (not now)
- **M2**: literature search with OpenAlex (a key is optional). A local OpenAI-compatible LLM
  (LM Studio on Linux, Docker Model Runner on the Mac) proposes candidate values, but code decides:
  - the quote must be present in the stored text;
  - the value must appear in the quote;
  - the unit must parse.

  Candidates that fail stay candidates, never records. Add a review queue for a person to accept
  or reject candidates.
- **M3**: more sources: USGS Mineral Commodity Summaries, World Bank commodity prices, EIA,
  FAOSTAT, and user PDFs (text extraction, then candidates).
- **M4**: discovery endpoints (`GET /v1/quantities`, text search), if the simulator's model
  builder needs them.

## Rules
- No simulation or model logic here, and no knowledge of which parameter a record will ground.
  The simulator's bindings decide that.
- Do not change evidence/v1. If something needs to change, propose `evidence/v2` and stop.
- Respect licences (fill `license`), rate limits and robots rules. Send a descriptive User-Agent.
- Work in small steps and run `make check` after each one. Every published record must trace back
  to a stored raw snapshot.
